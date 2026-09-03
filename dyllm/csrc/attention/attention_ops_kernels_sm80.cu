#include "common.h"

#include <cuda_bf16.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <tuple>
#include <type_traits>

#include "attn_tile_config.h"

// DIM=64/128/256.
namespace dyllm_sm80 {

// Only QSMEM and BLOCK_K vary by dimension.
template <int DIM>
struct Sm80Cfg;
template <>
struct Sm80Cfg<64> {
  static constexpr bool QSMEM = false;
  static constexpr int BLOCK_K = 64;
};
template <>
struct Sm80Cfg<128> {
  static constexpr bool QSMEM = false;
  static constexpr int BLOCK_K = 64;
};
template <>
struct Sm80Cfg<256> {
  static constexpr bool QSMEM = true;
  static constexpr int BLOCK_K = 32;
};


// [KV_PIPE] Replace K/V double buffering with one buffer each and pipeline them against each other.
// This reduces smem to max(BLOCK_Q, 2*BLOCK_K)*DIM*2, allowing BLOCK_K to be doubled.

struct BlockInfo {
  __device__ BlockInfo(const int* cu_seqlens_q, const int* cu_seqlens_k, int batch_id)
      : cu_seqlens_q_curr(cu_seqlens_q[batch_id]), cu_seqlens_q_next(cu_seqlens_q[batch_id + 1]),
        cu_seqlens_k_curr(cu_seqlens_k[batch_id]), cu_seqlens_k_next(cu_seqlens_k[batch_id + 1]),
        seqlen_q(cu_seqlens_q_next - cu_seqlens_q_curr), seqlen_kv(cu_seqlens_k_next - cu_seqlens_k_curr) {}
  const int cu_seqlens_q_curr;
  const int cu_seqlens_q_next;
  const int cu_seqlens_k_curr;
  const int cu_seqlens_k_next;
  const int seqlen_q;
  const int seqlen_kv;
};

__global__ void cosine_similarity_reduce_kernel(const float* __restrict__ stats, // [total_tokens, 3]
                                                bool* __restrict__ out,          // [total_tokens]
                                                const int total_tokens, const int* __restrict__ cu_seqlens_q,
                                                const float threshold) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_tokens)
    return;

  float dot = stats[idx * 3 + 0];
  float norm_a = stats[idx * 3 + 1];
  float norm_b = stats[idx * 3 + 2];

  float rsqrt_norm_a = rsqrtf(fmaxf(norm_a, 1e-8f));
  float rsqrt_norm_b = rsqrtf(fmaxf(norm_b, 1e-8f));

  float cos_sim = dot * rsqrt_norm_a * rsqrt_norm_b;
  out[idx] = (cos_sim < threshold);
}

void compute_cosine_similarity(const float* stats, bool* out, const int total_tokens, const int* cu_seqlens_q,
                               const float threshold) {
  int threads = 256;
  int blocks = (total_tokens + threads - 1) / threads;
  cosine_similarity_reduce_kernel<<<blocks, threads, 0>>>(stats, out, total_tokens, cu_seqlens_q, threshold);
}

__global__ void overwrite_salient_kernel(const nv_bfloat16* __restrict__ c,
                                          const nv_bfloat16* __restrict__ o_sal,
                                          nv_bfloat16* __restrict__ o,
                                          const int* __restrict__ idx_salient_row,
                                          float* __restrict__ cosine_stats, const int vector_size) {
  const int salient_idx = blockIdx.x;
  const int token_idx = idx_salient_row[salient_idx];
  const int pair_count = vector_size / 2;
  const nv_bfloat162* old_row = reinterpret_cast<const nv_bfloat162*>(c + token_idx * vector_size);
  const nv_bfloat162* new_row = reinterpret_cast<const nv_bfloat162*>(o_sal + salient_idx * vector_size);
  nv_bfloat162* out_row = reinterpret_cast<nv_bfloat162*>(o + token_idx * vector_size);

  float3 local = {0.0f, 0.0f, 0.0f};
  for (int pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
    const nv_bfloat162 old_v = old_row[pair];
    const nv_bfloat162 new_v = new_row[pair];
    out_row[pair] = new_v;
    const float2 old_f = __bfloat1622float2(old_v);
    const float2 new_f = __bfloat1622float2(new_v);
    local.x += new_f.x * old_f.x + new_f.y * old_f.y;
    local.y += new_f.x * new_f.x + new_f.y * new_f.y;
    local.z += old_f.x * old_f.x + old_f.y * old_f.y;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local.x += __shfl_down_sync(0xffffffff, local.x, offset);
    local.y += __shfl_down_sync(0xffffffff, local.y, offset);
    local.z += __shfl_down_sync(0xffffffff, local.z, offset);
  }

  __shared__ float3 warp_stats[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0)
    warp_stats[warp] = local;
  __syncthreads();

  if (warp == 0) {
    local = lane < blockDim.x / 32 ? warp_stats[lane] : make_float3(0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local.x += __shfl_down_sync(0xffffffff, local.x, offset);
      local.y += __shfl_down_sync(0xffffffff, local.y, offset);
      local.z += __shfl_down_sync(0xffffffff, local.z, offset);
    }
    if (lane == 0) {
      cosine_stats[token_idx * 3 + 0] = local.x;
      cosine_stats[token_idx * 3 + 1] = local.y;
      cosine_stats[token_idx * 3 + 2] = local.z;
    }
  }
}

__global__ void compute_k_block_mask_kernel(const int* __restrict__ idx_salient_row,
                                            const int* __restrict__ cu_salientlens,
                                            const int* __restrict__ cu_seqlens_k, uint64_t* __restrict__ row_masks,
                                            const int total_blocks, const int num_blk_k, const int block_k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_blocks)
    return;

  const int batch_id = idx / num_blk_k;
  const int block_id = idx - batch_id * num_blk_k;
  const int seq_start = cu_seqlens_k[batch_id];
  const int seq_end = cu_seqlens_k[batch_id + 1];
  const int block_start = seq_start + block_id * block_k;
  if (block_start >= seq_end) {
    row_masks[idx] = 0;
    return;
  }
  const int block_end = min(block_start + block_k, seq_end);

  int left = cu_salientlens[batch_id];
  int right = cu_salientlens[batch_id + 1];
  while (left < right) {
    const int mid = (left + right) >> 1;
    if (idx_salient_row[mid] < block_start)
      left = mid + 1;
    else
      right = mid;
  }
  uint64_t mask = 0;
  const int salient_end = cu_salientlens[batch_id + 1];
  while (left < salient_end && idx_salient_row[left] < block_end) {
    mask |= 1ULL << (idx_salient_row[left] - block_start);
    ++left;
  }
  row_masks[idx] = mask;
}

void compute_k_masks(const int batch_size, const int* cu_seqlens_k, const int* cu_salientlens_k,
                     const int* idx_salient_row_k, const int num_salient_k, const int BLOCK_K, const int max_seqlen_k,
                     uint64_t* row_masks) {

  int num_blk_k = (max_seqlen_k + BLOCK_K - 1) / BLOCK_K;
  const int total_blocks = batch_size * num_blk_k;
  constexpr int threads = 256;
  const int blocks = cdiv(total_blocks, threads);
  compute_k_block_mask_kernel<<<blocks, threads>>>(idx_salient_row_k, cu_salientlens_k, cu_seqlens_k, row_masks,
                                                   total_blocks, num_blk_k, BLOCK_K);
}

// Tile Q@K by BLOCK_K, and tile V loads and P@V by BLOCK_V (=MMA_K).
template <int BLOCK_Q, int BLOCK_K, int BLOCK_V, int DIM, int NUM_WARPS, bool MASK_V_ROWS, int D_SPLIT = 1>
__launch_bounds__(NUM_WARPS* WARP_SIZE) __global__
    void attention_sparse_varlen_kernel(const nv_bfloat16* Q,     // [sum(len_q), H_q, D]
                                        const nv_bfloat16* K,     // [sum(len_kv), H_kv, D]
                                        const nv_bfloat16* V,     // [sum(len_kv), H_kv, D]
                                        const nv_bfloat16* C,     // [sum(len_kv), H_q, D]
                                        const nv_bfloat16* O_sal, // [sum(salientlen), H_q, D]
                                        nv_bfloat16* O,           // [sum(len_q), H_q, D]
                                        const int B, const int H, const int H_kv, const int* cu_seqlens_q,
                                        const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k,
                                        const int* cu_salientlens, const uint64_t* row_masks, float* cosine_stats) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  // Each thread block handles one BLOCK_Q tile.
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int num_blk_q = cdiv(max_seqlen_q, BLOCK_Q);
  const int num_blk_kv = cdiv(max_seqlen_k, BLOCK_K);
  const int batch_id = (bid / (H * num_blk_q));
  const int head_id = (bid % (H * num_blk_q)) / num_blk_q;
  const int blk_q_id = bid % num_blk_q;

  // GQA: query head -> KV head
  const int num_queries_per_kv = H / H_kv;
  const int kv_head_id = head_id / num_queries_per_kv;

  BlockInfo binfo(cu_seqlens_q, cu_seqlens_k, batch_id);

  const int num_kv_iter = cdiv(binfo.seqlen_kv, BLOCK_K);

  const uint64_t* row_masks_batch = row_masks + batch_id * num_blk_kv;

  const int ldg_q = H * DIM;     // Q, C, O use num_q_heads
  const int ldg_kv = H_kv * DIM; // K, V use num_kv_heads

  // QKVO offsets
  const int batch_offset_q = binfo.cu_seqlens_q_curr * ldg_q;
  const int block_offset_q = blk_q_id * BLOCK_Q * ldg_q;
  const int head_offset_q = head_id * DIM;
  const int head_offset_kv = kv_head_id * DIM;
  const int batch_offset_kv = binfo.cu_seqlens_k_curr * ldg_kv;

  // Return if the block is out of range.
  if (batch_offset_q + block_offset_q >= binfo.cu_seqlens_q_next * ldg_q)
    return;


  Q += batch_offset_q + block_offset_q + head_offset_q;
  K += batch_offset_kv + head_offset_kv;
  V += batch_offset_kv + head_offset_kv;
  C += batch_offset_q + block_offset_q + head_offset_q;
  O_sal += cu_salientlens[batch_id] * ldg_q + head_offset_q;
  O += batch_offset_q + block_offset_q + head_offset_q;
  

  // Q_smem is loaded only once, so overlap it with (K_smem + V_smem).
  extern __shared__ nv_bfloat16 smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  // With Q_SMEM, Q must remain live throughout the KV loop, so K starts after it.
  constexpr bool Q_SMEM = Sm80Cfg<DIM>::QSMEM;
  const uint32_t K_smem = Q_SMEM ? Q_smem + BLOCK_Q * DIM * sizeof(nv_bfloat16) : Q_smem;
  const uint32_t V_smem = K_smem + BLOCK_K * DIM * sizeof(nv_bfloat16);

  constexpr int NUM_ROW_WARPS = NUM_WARPS / D_SPLIT;
  constexpr int WARP_Q = BLOCK_Q / NUM_ROW_WARPS;
  constexpr int DIM_W = DIM / D_SPLIT;
  static_assert(NUM_WARPS % D_SPLIT == 0, "NUM_WARPS must be a multiple of D_SPLIT");
  static_assert(BLOCK_Q % NUM_ROW_WARPS == 0, "BLOCK_Q must be a multiple of NUM_ROW_WARPS");
  static_assert(DIM % D_SPLIT == 0, "DIM must be a multiple of D_SPLIT");
  const int warp_row = warp_id % NUM_ROW_WARPS;
  const int warp_d = warp_id / NUM_ROW_WARPS;

  // mma.m16n8k16
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  static_assert(BLOCK_V == MMA_K, "BLOCK_V must equal MMA_K");
  static_assert(BLOCK_K % BLOCK_V == 0, "BLOCK_K must be a multiple of BLOCK_V");

  // Prepare registers. K_rmem/V_rmem each appear to require 128 registers, but ptxas
  // shortens their live ranges, so actual usage is lower.
  constexpr int Q_RMEM_M = Q_SMEM ? 1 : (WARP_Q / MMA_M);
  constexpr int Q_RMEM_D = Q_SMEM ? 1 : (DIM / MMA_K);
  uint32_t Q_rmem[Q_RMEM_M][Q_RMEM_D][4];
  constexpr int K_RMEM_KV = BLOCK_K / MMA_N;
  constexpr int K_RMEM_D = DIM / MMA_K;
  uint32_t K_rmem[K_RMEM_KV][K_RMEM_D][2];

  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_K / MMA_K][4];

  constexpr int V_RMEM_KV = BLOCK_K / MMA_K;
  constexpr int V_RMEM_D = DIM_W / MMA_N;
  uint32_t V_rmem[V_RMEM_KV][V_RMEM_D][2];
  static_assert(DIM_W % MMA_N == 0, "DIM/D_SPLIT must be a multiple of MMA_N");

  float O_rmem[WARP_Q / MMA_M][DIM_W / MMA_N][4] = {};


  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    const int row_off = warp_row * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    // A tile
    Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    // B tile
    K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // Transposed B tile.
    const int row_off = lane_id % 16;
    const int col_off = warp_d * DIM_W + lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));
  // Log2-domain softmax.
  const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // Load Q [BLOCK_Q, DIM].
  global_to_shared_swizzle_zero_pad<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, ldg_q, tid, blk_q_id, 0, binfo.seqlen_q);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // Shared memory -> registers
  if constexpr (!Q_SMEM) {
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
        uint32_t addr = Q_smem_thread;
        addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16); // row
        addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);       // col
        ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
      }
    __syncthreads();
  }

  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      const nv_bfloat16* src_k = K + kv_id * BLOCK_K * ldg_kv;
      global_to_shared_swizzle_zero_pad<BLOCK_K, DIM, TB_SIZE>(K_smem, src_k, ldg_kv, tid, kv_id, 0,
                                                               binfo.seqlen_kv);
    }
    asm volatile("cp.async.commit_group;");
  };

  auto load_V = [&](int kv_id, uint64_t row_mask) {
    if (kv_id < num_kv_iter && row_mask != 0) {
      {
        if constexpr (MASK_V_ROWS) {
          // Load V in BLOCK_V-row chunks, skipping cp.async for chunks with a zero mask.
          constexpr int NUM_V_CHUNKS = BLOCK_K / BLOCK_V;
          const int v_chunk_base = kv_id * NUM_V_CHUNKS;
#pragma unroll
          for (int c = 0; c < NUM_V_CHUNKS; c++) {
            const uint64_t sub_mask = (row_mask >> (c * BLOCK_V)) & ((1ULL << BLOCK_V) - 1);
            if (sub_mask == 0)
              continue;
            const int v_chunk_id = v_chunk_base + c;
            const uint32_t dst_v = V_smem + c * BLOCK_V * DIM * sizeof(nv_bfloat16);
            const nv_bfloat16* src_v = V + v_chunk_id * BLOCK_V * ldg_kv;
            global_to_shared_swizzle_zero_pad_with_row_mask<BLOCK_V, DIM, TB_SIZE>(
                dst_v, src_v, ldg_kv, tid, v_chunk_id, 0, binfo.seqlen_kv, sub_mask);
          }
        } else {
          const uint32_t dst_v = V_smem;
          const nv_bfloat16* src_v = V + kv_id * BLOCK_K * ldg_kv;
          global_to_shared_swizzle_zero_pad<BLOCK_K, DIM, TB_SIZE>(dst_v, src_v, ldg_kv, tid, kv_id, 0,
                                                                   binfo.seqlen_kv);
        }
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  uint64_t this_row_mask = num_kv_iter > 0 ? row_masks_batch[0] : 0;

  load_K(0);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    float S_rmem[WARP_Q / MMA_M][BLOCK_K / MMA_N][4] = {};
    load_V(kv_id, this_row_mask);

    const bool sparse_block = this_row_mask == 0;

    // K shared memory -> registers
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_K / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread;                         // [KV_PIPE] single buffer
        addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16); // row
        addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);        // col
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_K]
    if constexpr (Q_SMEM) {
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
          uint32_t q_reg[4];
          uint32_t addr = Q_smem_thread;
          addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16); // row
          addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);       // col
          ldmatrix_x4(q_reg, addr);
          for (int mma_id_kv = 0; mma_id_kv < BLOCK_K / MMA_N; mma_id_kv++)
            mma_m16n8k16(q_reg, K_rmem[mma_id_kv][mma_id_d], S_rmem[mma_id_q][mma_id_kv]);
        }
    } else {
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_K / MMA_N; mma_id_kv++)
          for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
            mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[mma_id_q][mma_id_kv]);
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    const uint64_t next_row_mask = kv_id + 1 < num_kv_iter ? row_masks_batch[kv_id + 1] : 0;
    load_K(kv_id + 1);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_K / MMA_N; mma_id_kv++) {
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        const int key_col = kv_id * BLOCK_K + mma_id_kv * MMA_N + (lane_id % 4) * 2;
        if (key_col >= binfo.seqlen_kv) {
          regs[0] = -FLT_MAX;
          regs[2] = -FLT_MAX;
        }
        if (key_col + 1 >= binfo.seqlen_kv) {
          regs[1] = -FLT_MAX;
          regs[3] = -FLT_MAX;
        }
      }
      // rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_K / MMA_N; mma_id_kv++) {
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        if (mma_id_kv == 0) {
          this_rowmax[0] = max(regs[0], regs[1]); // c0 and c1
          this_rowmax[1] = max(regs[2], regs[3]); // c2 and c3
        } else {
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1])); // c0 and c1
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3])); // c2 and c3
        }
      }

      // Four-lane butterfly reduction
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // New rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      // Rescale the previous O.
      float rescale[2];
      rescale[0] = exp2f((rowmax[mma_id_q][0] - this_rowmax[0]) * softmax_scale_log2);
      rescale[1] = exp2f((rowmax[mma_id_q][1] - this_rowmax[1]) * softmax_scale_log2);
      for (int mma_id_d = 0; mma_id_d < DIM_W / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // Store the new rowmax.
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2];
      const float m2_0 = rowmax[mma_id_q][0] * softmax_scale_log2;
      const float m2_1 = rowmax[mma_id_q][1] * softmax_scale_log2;
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_K / MMA_N; mma_id_kv++) {
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] = exp2f(fmaf(regs[0], softmax_scale_log2, -m2_0)); // c0
        regs[1] = exp2f(fmaf(regs[1], softmax_scale_log2, -m2_0)); // c1
        regs[2] = exp2f(fmaf(regs[2], softmax_scale_log2, -m2_1)); // c2
        regs[3] = exp2f(fmaf(regs[3], softmax_scale_log2, -m2_1)); // c3

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = regs[0] + regs[1];
          this_rowsumexp[1] = regs[2] + regs[3];
        } else {
          this_rowsumexp[0] += regs[0] + regs[1];
          this_rowsumexp[1] += regs[2] + regs[3];
        }

        // Pack into P for the next MMA (m16n8 -> m16k16).
        nv_bfloat162* this_P_rmem = reinterpret_cast<nv_bfloat162*>(P_rmem[mma_id_q][mma_id_kv / 2]);
        if (sparse_block)
          continue;
        this_P_rmem[(mma_id_kv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      // Accumulate into the overall rowsumexp.
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }
    // V shared memory -> registers, then MMA O += P @ V [BLOCK_Q, DIM]
    if (!sparse_block) {

    // One mma_id_kv step corresponds to one BLOCK_V-row chunk of V. For a zero-mask
    // chunk, skip both ldmatrix and mma.
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_K / MMA_K; mma_id_kv++) {
        const uint64_t sub_mask = (this_row_mask >> (mma_id_kv * BLOCK_V)) & ((1ULL << BLOCK_V) - 1);
        if (sub_mask == 0)
          continue;


        for (int mma_id_d = 0; mma_id_d < DIM_W / MMA_N; mma_id_d += 2) {
          uint32_t addr = V_smem_thread;                         // [KV_PIPE] single buffer
          addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16); // row
          addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);        // col
          ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
        }

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
          for (int mma_id_d = 0; mma_id_d < DIM_W / MMA_N; mma_id_d++)
            mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d], O_rmem[mma_id_q][mma_id_d]);
      }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    this_row_mask = next_row_mask;
  }

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowsumexp[mma_id_q][0] += __shfl_xor_sync(0xFFFF'FFFF, rowsumexp[mma_id_q][0], 1);
    rowsumexp[mma_id_q][0] += __shfl_xor_sync(0xFFFF'FFFF, rowsumexp[mma_id_q][0], 2);
    rowsumexp[mma_id_q][1] += __shfl_xor_sync(0xFFFF'FFFF, rowsumexp[mma_id_q][1], 1);
    rowsumexp[mma_id_q][1] += __shfl_xor_sync(0xFFFF'FFFF, rowsumexp[mma_id_q][1], 2);
  }

  // Write to O.
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    const int row = warp_row * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
    const int global_row_0 = binfo.cu_seqlens_q_curr + blk_q_id * BLOCK_Q + row;
    if (global_row_0 >= binfo.cu_seqlens_q_next)
      continue;

    const bool valid_1 = global_row_0 + 8 < binfo.cu_seqlens_q_next;
    const float inv_rowsum_0 = __frcp_rn(rowsumexp[mma_id_q][0]);
    const float inv_rowsum_1 = valid_1 ? __frcp_rn(rowsumexp[mma_id_q][1]) : 0.0f;

    float3 stats_0 = {0.0f, 0.0f, 0.0f};
    float3 stats_1 = {0.0f, 0.0f, 0.0f};
    for (int mma_id_d = 0; mma_id_d < DIM_W / MMA_N; mma_id_d++) {
      const int col = warp_d * DIM_W + mma_id_d * MMA_N + (lane_id % 4) * 2; // [D_SPLIT] this warp's column slice

      nv_bfloat162 old_c_0, old_c_1, new_c_0, new_c_1;
      float2 old_c_0_f2, old_c_1_f2, new_c_0_f2, new_c_1_f2;
      old_c_0 = reinterpret_cast<const nv_bfloat162*>(C + (row + 0) * ldg_q + col)[0];
      old_c_0_f2 = __bfloat1622float2(old_c_0);

      if (valid_1) {
        old_c_1 = reinterpret_cast<const nv_bfloat162*>(C + (row + 8) * ldg_q + col)[0];
        old_c_1_f2 = __bfloat1622float2(old_c_1);
      }

      float* regs = O_rmem[mma_id_q][mma_id_d];

      regs[0] *= inv_rowsum_0;
      regs[1] *= inv_rowsum_0;
      new_c_0_f2 = {old_c_0_f2.x + regs[0], old_c_0_f2.y + regs[1]};
      new_c_0 = __float22bfloat162_rn(new_c_0_f2);

      if (valid_1) {
        regs[2] *= inv_rowsum_1;
        regs[3] *= inv_rowsum_1;
        new_c_1_f2 = {old_c_1_f2.x + regs[2], old_c_1_f2.y + regs[3]};
        new_c_1 = __float22bfloat162_rn(new_c_1_f2);
      }

      stats_0.x += new_c_0_f2.x * old_c_0_f2.x + new_c_0_f2.y * old_c_0_f2.y;
      stats_0.y += new_c_0_f2.x * new_c_0_f2.x + new_c_0_f2.y * new_c_0_f2.y;
      stats_0.z += old_c_0_f2.x * old_c_0_f2.x + old_c_0_f2.y * old_c_0_f2.y;

      if (valid_1) {
        stats_1.x += new_c_1_f2.x * old_c_1_f2.x + new_c_1_f2.y * old_c_1_f2.y;
        stats_1.y += new_c_1_f2.x * new_c_1_f2.x + new_c_1_f2.y * new_c_1_f2.y;
        stats_1.z += old_c_1_f2.x * old_c_1_f2.x + old_c_1_f2.y * old_c_1_f2.y;
      }

      // Store O.
      reinterpret_cast<nv_bfloat162*>(O + (row + 0) * ldg_q + col)[0] = new_c_0;
      if (valid_1) {
        reinterpret_cast<nv_bfloat162*>(O + (row + 8) * ldg_q + col)[0] = new_c_1;
      }
    }

    // Four-lane (lane_id % 4) butterfly reduction
#pragma unroll
    for (int i = 1; i <= 2; i *= 2) {
      stats_0.x += __shfl_xor_sync(0xffffffff, stats_0.x, i);
      stats_0.y += __shfl_xor_sync(0xffffffff, stats_0.y, i);
      stats_0.z += __shfl_xor_sync(0xffffffff, stats_0.z, i);

      stats_1.x += __shfl_xor_sync(0xffffffff, stats_1.x, i);
      stats_1.y += __shfl_xor_sync(0xffffffff, stats_1.y, i);
      stats_1.z += __shfl_xor_sync(0xffffffff, stats_1.z, i);
    }

    // Reduce across thread blocks and store in global memory.
    if (lane_id % 4 == 0) {
      int global_token_idx_0 = global_row_0;

      if (global_token_idx_0 < binfo.cu_seqlens_q_next) {
        atomicAdd(&cosine_stats[global_token_idx_0 * 3 + 0], stats_0.x);
        atomicAdd(&cosine_stats[global_token_idx_0 * 3 + 1], stats_0.y);
        atomicAdd(&cosine_stats[global_token_idx_0 * 3 + 2], stats_0.z);
      }

      int global_token_idx_1 = global_token_idx_0 + 8;
      if (global_token_idx_1 < binfo.cu_seqlens_q_next) {
        atomicAdd(&cosine_stats[global_token_idx_1 * 3 + 0], stats_1.x);
        atomicAdd(&cosine_stats[global_token_idx_1 * 3 + 1], stats_1.y);
        atomicAdd(&cosine_stats[global_token_idx_1 * 3 + 2], stats_1.z);
      }
    }
  }
}

// Instantiate the same template for all three dimensions with different (QSMEM, BLOCK_K) values.
template <int DIM>
struct Sm80Tile {
  static constexpr int BLOCK_V = 16; // [BLOCK_V] V load + P@V tiling granularity (== MMA_K)
  static constexpr int BLOCK_Q = DYLLM_TILE_PINNED ? DyllmTile<DIM>::BLOCK_Q : 64;
  static constexpr int BLOCK_K = DYLLM_TILE_PINNED ? DyllmTile<DIM>::BLOCK_K : Sm80Cfg<DIM>::BLOCK_K;
  static constexpr int D_SPLIT = DYLLM_TILE_PINNED ? DyllmTile<DIM>::D_SPLIT : 1;
  static constexpr int WARP_Q_ = DYLLM_TILE_PINNED ? DyllmTile<DIM>::WARP_Q : 16;
  static constexpr int NUM_WARPS = (BLOCK_Q / WARP_Q_) * D_SPLIT;
  static constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  // [KV_PIPE] Single K/V buffers.
  static constexpr int SMEM_SIZE =
      (Sm80Cfg<DIM>::QSMEM ? (BLOCK_Q + 2 * BLOCK_K) : (BLOCK_Q > 2 * BLOCK_K ? BLOCK_Q : 2 * BLOCK_K)) * DIM *
      static_cast<int>(sizeof(nv_bfloat16));
  static_assert(BLOCK_K >= 16, "row_masks is allocated host-side for a 16-row block granularity");
};

template <int DIM>
static void launch_for_dim(const nv_bfloat16* q_ptr, const nv_bfloat16* k_ptr, const nv_bfloat16* v_ptr,
                           const nv_bfloat16* c_ptr, const nv_bfloat16* o_sal_ptr, nv_bfloat16* o_ptr, const int B,
                           const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                           const int max_seqlen_q, const int max_seqlen_k, const int num_salient,
                           const int* cu_salientlens, const int* idx_salient_row_k, uint64_t* row_masks,
                           float* cosine_stats, int force_mask_v_rows) {
  using T = Sm80Tile<DIM>;
  compute_k_masks(B, cu_seqlens_k, cu_salientlens, idx_salient_row_k, num_salient, T::BLOCK_K, max_seqlen_k, row_masks);

  const int num_blocks = B * H * cdiv(max_seqlen_q, T::BLOCK_Q);

  // On H100, masked zero-filling wins on short grids dominated by K/V bandwidth and underutilization.
  // Once there are enough CTAs to fill the GPU, direct copying has lower instruction overhead.
  const bool mask_v_rows = (force_mask_v_rows >= 0) ? (force_mask_v_rows != 0) : (num_blocks <= 1024);

  auto go = [&](auto kernel) {
    launch_kernel(kernel, num_blocks, T::TB_SIZE, T::SMEM_SIZE, q_ptr, k_ptr, v_ptr, c_ptr, o_sal_ptr, o_ptr, B, H,
                  H_kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, cu_salientlens, row_masks,
                  cosine_stats);
  };
  if (mask_v_rows)
    go(attention_sparse_varlen_kernel<T::BLOCK_Q, T::BLOCK_K, T::BLOCK_V, DIM, T::NUM_WARPS, true, T::D_SPLIT>);
  else
    go(attention_sparse_varlen_kernel<T::BLOCK_Q, T::BLOCK_K, T::BLOCK_V, DIM, T::NUM_WARPS, false, T::D_SPLIT>);
}

template <typename scalar_t>
void attention_sparse_varlen(const scalar_t* q,     // [sum(len_q), H_q, D]
                             const scalar_t* k,     // [sum(len_kv), H_kv, D]
                             const scalar_t* v,     // [sum(len_kv), H_kv, D]
                             const scalar_t* c,     // [sum(len_kv), H_q, D]
                             const scalar_t* o_sal, // [sum(salientlen), H_q, D]
                             scalar_t* o,           // [sum(len_q), H_q, D]
                             const int B, const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                             const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                             const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                             const int* idx_salient_row_q, uint64_t* row_masks,
                             float* cosine_stats, // [sum(len_q), 3] - intermediate buffer, must be zero-initialized
                             bool* cosine_out,    // [sum(len_q)] - output bool mask (true if cos_sim < threshold)
                             float threshold, int dim,
                             int force_mask_v_rows) { // BENCH-ONLY override: -1=auto, 0=force B, 1=force A

  // Only BFloat16 is currently supported.
  if constexpr (!std::is_same_v<scalar_t, at::BFloat16>) {
    std::cerr << "Only BFloat16 is supported" << std::endl;
    exit(1);
  }

  const nv_bfloat16* q_ptr = reinterpret_cast<const nv_bfloat16*>(q);
  const nv_bfloat16* k_ptr = reinterpret_cast<const nv_bfloat16*>(k);
  const nv_bfloat16* v_ptr = reinterpret_cast<const nv_bfloat16*>(v);
  const nv_bfloat16* c_ptr = reinterpret_cast<const nv_bfloat16*>(c);
  const nv_bfloat16* o_sal_ptr = reinterpret_cast<const nv_bfloat16*>(o_sal);
  nv_bfloat16* o_ptr = reinterpret_cast<nv_bfloat16*>(o);

#define DYLLM_SM80_LAUNCH(D)                                                                                           \
  launch_for_dim<D>(q_ptr, k_ptr, v_ptr, c_ptr, o_sal_ptr, o_ptr, B, H, H_kv, cu_seqlens_q, cu_seqlens_k,              \
                    max_seqlen_q, max_seqlen_k, num_salient, cu_salientlens, idx_salient_row_k, row_masks,             \
                    cosine_stats, force_mask_v_rows)
  bool handled = false;
  if constexpr (DYLLM_DIM_ENABLED(64)) {
    if (dim == 64) {
      DYLLM_SM80_LAUNCH(64);
      handled = true;
    }
  }
  if constexpr (DYLLM_DIM_ENABLED(128)) {
    if (!handled && dim == 128) {
      DYLLM_SM80_LAUNCH(128);
      handled = true;
    }
  }
  if constexpr (DYLLM_DIM_ENABLED(256)) {
    if (!handled && dim == 256) {
      DYLLM_SM80_LAUNCH(256);
      handled = true;
    }
  }
  if (!handled) {
    std::cerr << "attention_ops_kernels_sm80.cu implements dim=64/128/256 (got dim=" << dim
              << "). Co-link attention_ops_kernels_sm80_d512.cu for dim=512." << std::endl;
    exit(1);
  }
#undef DYLLM_SM80_LAUNCH

  if (num_salient > 0) {
    overwrite_salient_kernel<<<num_salient, 256>>>(c_ptr, o_sal_ptr, o_ptr, idx_salient_row_q, cosine_stats, H * dim);
  }

  compute_cosine_similarity(cosine_stats, cosine_out, total_seqlen_q, cu_seqlens_q, threshold);
}

// Explicit instantiations
template void attention_sparse_varlen<at::BFloat16>(
    const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v, const at::BFloat16* c,
    const at::BFloat16* o_sal, at::BFloat16* o, const int B, const int H, const int H_kv, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
    const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
    uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim, int force_mask_v_rows);

// Kernel launch resources for sweeps. Nonzero local bytes indicate that ptxas spilled registers.
// Returns [registers, local bytes, static smem, dynamic smem, tb_size,
//          max active blocks per SM, block_q, block_k, d_split, ok].
template <int DIM>
static std::vector<int64_t> kernel_info_for_dim(bool mask_v_rows) {
  using T = Sm80Tile<DIM>;
  const void* kernel =
      mask_v_rows ? reinterpret_cast<const void*>(&attention_sparse_varlen_kernel<T::BLOCK_Q, T::BLOCK_K, T::BLOCK_V,
                                                                                 DIM, T::NUM_WARPS, true, T::D_SPLIT>)
                  : reinterpret_cast<const void*>(&attention_sparse_varlen_kernel<T::BLOCK_Q, T::BLOCK_K, T::BLOCK_V,
                                                                                 DIM, T::NUM_WARPS, false, T::D_SPLIT>);

  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) {
    cudaGetLastError();
    return {0, 0, 0, T::SMEM_SIZE, T::TB_SIZE, 0, T::BLOCK_Q, T::BLOCK_K, T::D_SPLIT, 0};
  }

  int64_t ok = 1;
  if (T::SMEM_SIZE > 48'000) {
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_SIZE) != cudaSuccess) {
      cudaGetLastError();
      ok = 0;
    }
  }
  int blocks = 0;
  if (ok && cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, T::TB_SIZE, T::SMEM_SIZE) != cudaSuccess) {
    cudaGetLastError();
    blocks = 0;
  }

  return {attr.numRegs, static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes),
          T::SMEM_SIZE, T::TB_SIZE, blocks, T::BLOCK_Q, T::BLOCK_K, T::D_SPLIT, ok};
}

std::vector<int64_t> attention_kernel_info(int dim, bool mask_v_rows) {
  if constexpr (DYLLM_DIM_ENABLED(64))
    if (dim == 64)
      return kernel_info_for_dim<64>(mask_v_rows);
  if constexpr (DYLLM_DIM_ENABLED(128))
    if (dim == 128)
      return kernel_info_for_dim<128>(mask_v_rows);
  if constexpr (DYLLM_DIM_ENABLED(256))
    if (dim == 256)
      return kernel_info_for_dim<256>(mask_v_rows);
  return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

__global__ void fused_offset_kernel(const int* idxs, const int* num_idxs_ptr, const int* cu_promptlens,
                                    const int* cu_salientlens, const int num_groups, long* out_tensor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int limit = *num_idxs_ptr;
  if (tid >= limit)
    return;

  int left = 0;
  int right = num_groups;
  int group_idx = 0;

  while (left < right) {
    int mid = left + (right - left) / 2;
    if (cu_salientlens[mid + 1] <= tid) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  group_idx = left;

  long offset_val = (long)cu_promptlens[group_idx + 1];

  out_tensor[tid] = (long)idxs[tid] + offset_val;
}

void fused_offset_launch(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                         torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads) {
  const int threads = 256;
  const int blocks = (max_threads + threads - 1) / threads;

  fused_offset_kernel<<<blocks, threads>>>(idxs.data_ptr<int>(), num_idxs.data_ptr<int>(),
                                           cu_promptlens.data_ptr<int>(), cu_salientlens.data_ptr<int>(),
                                           cu_promptlens.size(0) - 1, out_tensor.data_ptr<long>());
}

} // namespace dyllm_sm80

// Global entry point. This is the primary TU, so only the plain name needs to be exported.
#ifndef DYLLM_SM80_SECONDARY
template <typename scalar_t>
void attention_sparse_varlen(const scalar_t* q, const scalar_t* k, const scalar_t* v, const scalar_t* c,
                             const scalar_t* o_sal, scalar_t* o, const int B, const int H, const int H_kv,
                             const int* cu_seqlens_q, const int* cu_seqlens_k, const int max_seqlen_q,
                             const int max_seqlen_k, const int total_seqlen_q, const int num_salient,
                             const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
                             uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim,
                             int force_mask_v_rows) {
  dyllm_sm80::attention_sparse_varlen<scalar_t>(q, k, v, c, o_sal, o, B, H, H_kv, cu_seqlens_q, cu_seqlens_k,
      max_seqlen_q, max_seqlen_k, total_seqlen_q, num_salient, cu_salientlens, idx_salient_row_k, idx_salient_row_q,
      row_masks, cosine_stats, cosine_out, threshold, dim, force_mask_v_rows);
}

template void attention_sparse_varlen<at::BFloat16>(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v, const at::BFloat16* c,
    const at::BFloat16* o_sal, at::BFloat16* o, const int B, const int H, const int H_kv, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
    const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
    uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim, int force_mask_v_rows);

std::vector<int64_t> attention_kernel_info(int dim, bool mask_v_rows) {
  return dyllm_sm80::attention_kernel_info(dim, mask_v_rows);
}

void fused_offset_launch(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                         torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads) {
  dyllm_sm80::fused_offset_launch(idxs, num_idxs, cu_promptlens, cu_salientlens, out_tensor, max_threads);
}
#endif // DYLLM_SM80_SECONDARY
