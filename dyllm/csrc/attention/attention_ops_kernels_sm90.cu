#include "common.h"
#include "hopper_common.cuh"

#include <cuda_bf16.h>
#include <float.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>
#include <type_traits>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>

#include "attn_tile_config.h"
namespace dyllm_sm90 {

// Only BLOCK_K and MASK_FORM vary by DIM (64/256/0, 128/192/0,
// 256/64/1).
template <int DIM>
struct Sm90Cfg;
template <>
struct Sm90Cfg<64> {
  static constexpr int BLOCK_K = 256;
  static constexpr int MASK_FORM = 0;
};
template <>
struct Sm90Cfg<128> {
  static constexpr int BLOCK_K = 192;
  static constexpr int MASK_FORM = 0;
};
template <>
struct Sm90Cfg<256> {
  static constexpr int BLOCK_K = 64;
  static constexpr int MASK_FORM = 1;
};

// Constants shared by all three dimensions.
static constexpr int SM90_MMA_WG = 1;
static constexpr int SM90_STAGES = 1;
static constexpr bool SM90_WS = true;
[[maybe_unused]] static constexpr bool SM90_OVERLAP = false;
[[maybe_unused]] static constexpr bool SM90_PINGPONG = true;
static constexpr int SM90_BLOCK_Q = 64 * SM90_MMA_WG;
// row_masks contains one uint64 per key block. Its 64 bits represent 64 rows regardless of tile size.
[[maybe_unused]] static constexpr int SM90_MASK_BLOCK = 64;
// Warp specialization adds one warpgroup for the TMA producer.
[[maybe_unused]] static constexpr int SM90_TB = SM90_WS ? (1 + SM90_MMA_WG) * 128 : 128;

template <int DIM>
struct Sm90Tile {
  static constexpr int BLOCK_K = Sm90Cfg<DIM>::BLOCK_K;
  // Place the mbarriers after the Q + STAGES*(K,V) tiles: one for Q and one
  // full/empty pair per stage, rounded up to a multiple of 128 bytes to preserve tile alignment.
  static constexpr int SMEM =
      (SM90_BLOCK_Q + 2 * SM90_STAGES * BLOCK_K) * DIM * 2 + 128 * (1 + 2 * SM90_STAGES);
};


using namespace dyllm_hopper;

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

// Main kernel
template <int BLOCK_Q, int BLOCK_K, int DIM, int STAGES, bool WS, int NUM_MMA_WG, bool OVERLAP,
          int MASK_FORM>
__launch_bounds__(WS ? (1 + NUM_MMA_WG) * 128 : 128, NUM_MMA_WG == 1 ? 2 : 1) __global__
    void attention_h100_kernel(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                               const __grid_constant__ CUtensorMap tmV, const nv_bfloat16* __restrict__ C,
                               nv_bfloat16* __restrict__ O, const int B, const int H, const int H_kv,
                               const int* __restrict__ cu_seqlens_q, const int* __restrict__ cu_seqlens_k,
                               const int max_seqlen_q, const int max_seqlen_k,
                               const int* __restrict__ cu_salientlens, const uint64_t* __restrict__ row_masks,
                               float* __restrict__ cosine_stats) {
#if DYLLM_SM90A
  // Each warpgroup owns exactly 64 Q rows. More warpgroups reduce d64 occupancy
  // to one CTA per SM, so the default uses one warpgroup and a wide key tile.
  static_assert(BLOCK_Q == 64 * NUM_MMA_WG, "one MMA warpgroup per 64 Q rows");
  static_assert(WS || NUM_MMA_WG == 1, "multiple MMA warpgroups require warp specialization");
  static_assert(DIM % SWZ_COLS == 0, "DIM must be a whole number of 128B swizzle blocks");
  static_assert(BLOCK_K % 16 == 0, "BLOCK_K must be a whole number of wgmma k16 steps");

  constexpr int NCB = DIM / SWZ_COLS;              // Number of 128-byte column blocks per row (1 for DIM=64)
  constexpr int Q_CB = BLOCK_Q * SWZ_ROW_BYTES;    // Bytes in one Q column block
  constexpr int KV_CB = BLOCK_K * SWZ_ROW_BYTES;   // Bytes in one K/V column block
  constexpr int SQ_BYTES = NCB * Q_CB;
  constexpr int SKV_BYTES = NCB * KV_CB;
  constexpr int NKSTEP = BLOCK_K / 16;             // Number of wgmma k16 steps in the P@V reduction
  // Accumulate S in 64-key subtiles because one uint64 mask word corresponds
  // exactly to 64 keys. Use the n32 form only when BLOCK_K=32.
  constexpr int SUBK = BLOCK_K < 64 ? BLOCK_K : 64;
  constexpr int NS = BLOCK_K / SUBK;
  constexpr int NNB = SUBK / 8;                    // Number of n-blocks per subtile
  constexpr int NREG = SUBK / 2;                   // Number of accumulator registers per subtile
  constexpr int KPS = SUBK / 16;                   // Number of k16 steps per subtile
  static_assert(BLOCK_K == 32 || BLOCK_K % 64 == 0, "BLOCK_K must be 32 or a multiple of 64");
  constexpr int MNMAJOR_K16 = K16_STEP_MNMAJOR_BYTES;
  // P@V output width. DIM=64 uses the n64 form, requiring 32 rather than 64
  // accumulator registers; the saved registers accommodate a wider BLOCK_K.
  constexpr int OW = DIM < 128 ? DIM : 128;
  constexpr int NPV = DIM / OW;
  constexpr int ONB = OW / 8;   // Number of accumulator n-blocks per P@V group
  static_assert(DIM % OW == 0, "P@V output width must divide the head dim");
  // 128*32 + 128*224 == 32768 registers, allowing two CTAs per SM. Registers
  // limit occupancy before the 72 KB of shared memory used by BLOCK_K=256.

  //   1 MMA warpgroup: initial allocation 128. (128-32)*128 = 12288 = (224-128)*128
  //   2 MMA warpgroups: initial allocation 168. (168-24)*128 = 18432 = (240-168)*256
  constexpr int PRODUCER_REGS = NUM_MMA_WG == 1 ? 32 : 24;
  constexpr int CONSUMER_REGS = NUM_MMA_WG == 1 ? 224 : 240;
  static_assert(128 * PRODUCER_REGS + NUM_MMA_WG * 128 * CONSUMER_REGS <= 65536,
                "setmaxnreg split exceeds the SM register file; consumers would block forever");
  constexpr int CONSUMER_BAR = 1;                  // First named-barrier ID, one per consumer warpgroup
  // [Ping-pong] Offset two MMA warpgroups by half a step so one uses the tensor
  // cores while the other runs softmax. This matters only with two warpgroups.
  constexpr int SCHED_BAR = CONSUMER_BAR + NUM_MMA_WG;
  constexpr bool PINGPONG = (NUM_MMA_WG == 2) && SM90_PINGPONG;
  static_assert(!OVERLAP || STAGES >= 2, "the pipeline needs K(j+1) live while P@V(j) runs");

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sQ = smem;
  uint8_t* sK = sQ + SQ_BYTES;
  uint8_t* sV = sK + STAGES * SKV_BYTES;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sV + STAGES * SKV_BYTES);

  const uint32_t aQ = static_cast<uint32_t>(__cvta_generic_to_shared(sQ));
  const uint32_t aK = static_cast<uint32_t>(__cvta_generic_to_shared(sK));
  const uint32_t aV = static_cast<uint32_t>(__cvta_generic_to_shared(sV));
  const uint32_t aBar = static_cast<uint32_t>(__cvta_generic_to_shared(bars));

  const uint32_t bQ = aBar;

  auto bKFull = [&](int s) { return aBar + (1 + s) * 8; };
  auto bKEmpty = [&](int s) { return aBar + (1 + STAGES + s) * 8; };
  auto bVFull = [&](int s) { return aBar + (1 + 2 * STAGES + s) * 8; };
  auto bVEmpty = [&](int s) { return aBar + (1 + 3 * STAGES + s) * 8; };

  const int tid = threadIdx.x;

  const int num_blk_q = cdiv(max_seqlen_q, BLOCK_Q);

  const int num_blk_kv = cdiv(max_seqlen_k, 64);
  const int bid = blockIdx.x;
  const int batch_id = bid / (H * num_blk_q);
  const int head_id = (bid % (H * num_blk_q)) / num_blk_q;
  const int blk_q_id = bid % num_blk_q;
  const int kv_head_id = head_id / (H / H_kv);

  BlockInfo binfo(cu_seqlens_q, cu_seqlens_k, batch_id);
  const int num_kv_iter = cdiv(binfo.seqlen_kv, BLOCK_K);
  const uint64_t* row_masks_batch = row_masks + batch_id * num_blk_kv;

  const int ldg_q = H * DIM;
  const int q_row0 = binfo.cu_seqlens_q_curr + blk_q_id * BLOCK_Q;
  if (blk_q_id * BLOCK_Q >= binfo.seqlen_q)
    return;

  if (tid == 0) {
    mbar_init(bQ, 1);
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(bKFull(s), 1);
      mbar_init(bVFull(s), 1);
      mbar_init(bKEmpty(s), NUM_MMA_WG);
      mbar_init(bVEmpty(s), NUM_MMA_WG);
    }
  }
  __syncthreads();
  fence_async_shared();

  auto issue_k = [&](int kv_id, int stage) {
    mbar_expect_tx(bKFull(stage), SKV_BYTES);
    const int krow = binfo.cu_seqlens_k_curr + kv_id * BLOCK_K;
    for (int cb = 0; cb < NCB; ++cb)
      tma_load_3d(aK + stage * SKV_BYTES + cb * KV_CB, &tmK, bKFull(stage), cb * SWZ_COLS, kv_head_id, krow);
  };
  auto issue_v = [&](int kv_id, int stage) {
    mbar_expect_tx(bVFull(stage), SKV_BYTES);
    const int krow = binfo.cu_seqlens_k_curr + kv_id * BLOCK_K;
    for (int cb = 0; cb < NCB; ++cb)
      tma_load_3d(aV + stage * SKV_BYTES + cb * KV_CB, &tmV, bVFull(stage), cb * SWZ_COLS, kv_head_id, krow);
  };
  auto issue_q = [&] {
    mbar_expect_tx(bQ, SQ_BYTES);
    for (int cb = 0; cb < NCB; ++cb)
      tma_load_3d(aQ + cb * Q_CB, &tmQ, bQ, cb * SWZ_COLS, head_id, q_row0);
  };

  // ---------------------------------------------------------------- Producer
  if constexpr (WS) {
    if (tid < 128) {
      setmaxnreg_dec<PRODUCER_REGS>();
      if (tid == 0) {
        issue_q();
        for (int kv_id = 0; kv_id < num_kv_iter; ++kv_id) {
          const int stage = kv_id % STAGES;
          const uint32_t ph = ((kv_id / STAGES) - 1) & 1;
          if (kv_id >= STAGES)
            mbar_wait(bKEmpty(stage), ph);
          issue_k(kv_id, stage);
          if (kv_id >= STAGES)
            mbar_wait(bVEmpty(stage), ph);
          issue_v(kv_id, stage);
        }
      }
      return;
    }
    setmaxnreg_inc<CONSUMER_REGS>();
  } else {
    if (tid == 0) {
      issue_q();
      for (int s = 0; s < STAGES && s < num_kv_iter; ++s)
      {
        issue_k(s, s);
        issue_v(s, s);
      }
    }
  }

  // ---------------------------------------------------------------- Consumer
  const int cwg = NUM_MMA_WG == 1 ? 0 : __shfl_sync(0xFFFFFFFFu, static_cast<int>(threadIdx.x / 128) - 1, 0);
  const int math_tid = tid - (WS ? (cwg + 1) * 128 : 0);
  const int warp_id = math_tid / WARP_SIZE;
  const int lane_id = math_tid % WARP_SIZE;

  const int r0 = warp_id * 16 + lane_id / 4;
  const int c0 = (lane_id % 4) * 2;

  float O_acc[NPV][OW / 2];
#pragma unroll
  for (int g = 0; g < NPV; ++g)
#pragma unroll
    for (int i = 0; i < OW / 2; ++i)
      O_acc[g][i] = 0.f;

  float rowmax[2] = {-FLT_MAX, -FLT_MAX};
  float rowsum[2] = {0.f, 0.f};

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));
  const float scale_log2 = softmax_scale * 1.4426950408889634f;

  if constexpr (PINGPONG) {
    if (cwg == 0)
      named_barrier_arrive<SCHED_BAR, 256>();
  }

  mbar_wait(bQ, 0);  // Q is resident in shared memory

  float S[NS][NREG];
  auto zero_S = [&](float (&s_)[NS][NREG]) {
#pragma unroll
    for (int h = 0; h < NS; ++h)
#pragma unroll
      for (int i = 0; i < NREG; ++i)
        s_[h][i] = 0.f;
  };

  const uint64_t descQ = desc_kmajor(aQ);
  auto issue_qk = [&](float (&s_)[NS][NREG], int st) {
    const uint64_t dK = desc_kmajor(aK + st * SKV_BYTES);
#pragma unroll
    for (int h = 0; h < NS; ++h)
#pragma unroll
      for (int cb = 0; cb < NCB; ++cb)
#pragma unroll
        for (int kk = 0; kk < SWZ_COLS / 16; ++kk) {
          const uint64_t da = desc_add(descQ, cb * Q_CB + cwg * 64 * SWZ_ROW_BYTES + kk * K16_STEP_BYTES);
          const uint64_t db = desc_add(dK, cb * KV_CB + h * SUBK * SWZ_ROW_BYTES + kk * K16_STEP_BYTES);
          if constexpr (SUBK == 32)
            wgmma_m64n32k16_ss<1>(s_[h], da, db);
          else
            wgmma_m64n64k16_ss<1>(s_[h], da, db);
        }
  };

  auto rescale_O = [&](const float (&r)[2]) {
#pragma unroll
    for (int g = 0; g < NPV; ++g)
#pragma unroll
      for (int nb = 0; nb < ONB; ++nb) {
        O_acc[g][nb * 4 + 0] *= r[0];
        O_acc[g][nb * 4 + 1] *= r[0];
        O_acc[g][nb * 4 + 2] *= r[1];
        O_acc[g][nb * 4 + 3] *= r[1];
      }
  };

  auto sched_sync = [&] {
    if constexpr (PINGPONG) {
      if (cwg == 0)
        named_barrier_sync<SCHED_BAR, 256>();
      else
        named_barrier_sync<SCHED_BAR + 1, 256>();
    }
  };
  auto sched_arrive = [&] {
    if constexpr (PINGPONG) {
      if (cwg == 0)
        named_barrier_arrive<SCHED_BAR + 1, 256>();
      else
        named_barrier_arrive<SCHED_BAR, 256>();
    }
  };
  auto release_bar = [&](uint32_t bar) {
    if constexpr (NUM_MMA_WG == 1) {
      named_barrier_sync<CONSUMER_BAR, 128>();
    } else if (cwg == 0) {
      named_barrier_sync<CONSUMER_BAR, 128>();
    } else {
      named_barrier_sync<CONSUMER_BAR + 1, 128>();
    }
    if (math_tid == 0)
      mbar_arrive(bar);
  };
  auto release_stage = [&](int st) {
    release_bar(bKEmpty(st));
    release_bar(bVEmpty(st));
  };

  if constexpr (OVERLAP) {
    if (num_kv_iter > 0) {
      mbar_wait(bKFull(0), 0);
      zero_S(S);
      wgmma_fence();
      issue_qk(S, 0);
      wgmma_commit();
      wgmma_wait<0>();
      release_bar(bKEmpty(0));
    }
  }

  for (int kv_id = 0; kv_id < num_kv_iter; ++kv_id) {
    const int stage = kv_id % STAGES;
    if constexpr (!OVERLAP)
      mbar_wait(bKFull(stage), (kv_id / STAGES) & 1);


    // ---------------- S = Q @ K^T (SS, both operands K-major) -----------------
    if constexpr (!OVERLAP) {
      zero_S(S);
      wgmma_fence();
      issue_qk(S, stage);
      wgmma_commit();
      wgmma_wait<0>();
      release_bar(bKEmpty(stage));
    }


    // ---------------- Tail mask + online softmax ---------------------
    const bool tail = (kv_id + 1) * BLOCK_K > binfo.seqlen_kv;
    if (tail) {
#pragma unroll
      for (int h = 0; h < NS; ++h)
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int key = kv_id * BLOCK_K + h * SUBK + nb * 8 + c0 + (i & 1);
            if (key >= binfo.seqlen_kv)
              S[h][nb * 4 + i] = -FLT_MAX;
          }
    }

    float m_new[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
    for (int h = 0; h < NS; ++h)
#pragma unroll
      for (int nb = 0; nb < NNB; ++nb) {
        m_new[0] = fmaxf(m_new[0], fmaxf(S[h][nb * 4 + 0], S[h][nb * 4 + 1]));
        m_new[1] = fmaxf(m_new[1], fmaxf(S[h][nb * 4 + 2], S[h][nb * 4 + 3]));
      }
    // The four lanes in a quad share a row, so reduce rowmax with a butterfly pattern.
#pragma unroll
    for (int d = 1; d <= 2; d *= 2) {
      m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xFFFFFFFFu, m_new[0], d));
      m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xFFFFFFFFu, m_new[1], d));
    }

    float rescale[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float m = fmaxf(rowmax[r], m_new[r]);
      rescale[r] = (rowmax[r] == -FLT_MAX) ? 1.f : exp2f((rowmax[r] - m) * scale_log2);
      rowmax[r] = m;
    }
    const float m2_0 = rowmax[0] * scale_log2;
    const float m2_1 = rowmax[1] * scale_log2;

    float psum[2] = {0.f, 0.f};
#pragma unroll
    for (int h = 0; h < NS; ++h)
#pragma unroll
      for (int nb = 0; nb < NNB; ++nb) {
        S[h][nb * 4 + 0] = exp2f(fmaf(S[h][nb * 4 + 0], scale_log2, -m2_0));
        S[h][nb * 4 + 1] = exp2f(fmaf(S[h][nb * 4 + 1], scale_log2, -m2_0));
        S[h][nb * 4 + 2] = exp2f(fmaf(S[h][nb * 4 + 2], scale_log2, -m2_1));
        S[h][nb * 4 + 3] = exp2f(fmaf(S[h][nb * 4 + 3], scale_log2, -m2_1));
        psum[0] += S[h][nb * 4 + 0] + S[h][nb * 4 + 1];
        psum[1] += S[h][nb * 4 + 2] + S[h][nb * 4 + 3];
      }
    rowsum[0] = rowsum[0] * rescale[0] + psum[0];
    rowsum[1] = rowsum[1] * rescale[1] + psum[1];


    // ---------------- O += P @ V (RS, V read as MN-major) -----------------
    constexpr int NMW = (BLOCK_K + 63) / 64;
    uint64_t mw[NMW];
    uint64_t mask_any = 0;
#pragma unroll
    for (int w = 0; w < NMW; ++w) {
      const int widx = (kv_id * BLOCK_K) / 64 + w;
      if constexpr (NMW == 1) {
        mw[w] = row_masks_batch[widx];
      } else {
        mw[w] = widx < num_blk_kv ? row_masks_batch[widx] : 0ull;
      }
      mask_any |= mw[w];
    }
    uint32_t Afrag[NKSTEP][4];
    bool live[NKSTEP];
    auto pack_afrag = [&] {
#pragma unroll
    for (int j = 0; j < NKSTEP; ++j) {
      const int h = j / KPS;   // S subtile containing this step
      const int jj = j % KPS;  // k16 step within that subtile

      const uint64_t m = mw[j / 4];
      const int kb = (j % 4) * 16;
      const uint64_t sub = (m >> kb) & 0xFFFFull;
      live[j] = sub != 0;

      const float* Sh = S[h];

      if constexpr (MASK_FORM == 0) {
        auto keep = [&](float v, int koff) { return ((m >> (kb + koff)) & 1ull) ? v : 0.f; };
        Afrag[j][0] = pack_bf16x2(keep(Sh[(2 * jj) * 4 + 0], c0), keep(Sh[(2 * jj) * 4 + 1], c0 + 1));
        Afrag[j][1] = pack_bf16x2(keep(Sh[(2 * jj) * 4 + 2], c0), keep(Sh[(2 * jj) * 4 + 3], c0 + 1));
        Afrag[j][2] = pack_bf16x2(keep(Sh[(2 * jj + 1) * 4 + 0], c0 + 8), keep(Sh[(2 * jj + 1) * 4 + 1], c0 + 9));
        Afrag[j][3] = pack_bf16x2(keep(Sh[(2 * jj + 1) * 4 + 2], c0 + 8), keep(Sh[(2 * jj + 1) * 4 + 3], c0 + 9));
      } else {
        auto pairmask = [&](int koff) -> uint32_t {
          const uint32_t b = static_cast<uint32_t>((sub >> koff) & 3ull);
          const uint32_t lo = static_cast<uint32_t>(-static_cast<int32_t>(b & 1u)) & 0x0000FFFFu;
          const uint32_t hi = static_cast<uint32_t>(-static_cast<int32_t>(b >> 1)) & 0xFFFF0000u;
          return lo | hi;
        };
        const uint32_t m01 = pairmask(c0);
        const uint32_t m89 = pairmask(c0 + 8);
        Afrag[j][0] = pack_bf16x2(Sh[(2 * jj) * 4 + 0], Sh[(2 * jj) * 4 + 1]) & m01;
        Afrag[j][1] = pack_bf16x2(Sh[(2 * jj) * 4 + 2], Sh[(2 * jj) * 4 + 3]) & m01;
        Afrag[j][2] = pack_bf16x2(Sh[(2 * jj + 1) * 4 + 0], Sh[(2 * jj + 1) * 4 + 1]) & m89;
        Afrag[j][3] = pack_bf16x2(Sh[(2 * jj + 1) * 4 + 2], Sh[(2 * jj + 1) * 4 + 3]) & m89;
      }
    }
    };

    auto issue_pv = [&](int st) {
#pragma unroll
      for (int g = 0; g < NPV; ++g) {
        const uint64_t dV = OW == 64 ? desc_mnmajor(aV + st * SKV_BYTES + g * KV_CB)
                                     : desc_mnmajor_n128(aV + st * SKV_BYTES + 2 * g * KV_CB, KV_CB);
#pragma unroll
        for (int j = 0; j < NKSTEP; ++j) {
          if (!live[j])
            continue;
          if constexpr (OW == 64)
            wgmma_m64n64k16_rs<1, 1>(O_acc[g], Afrag[j], desc_add(dV, j * MNMAJOR_K16));
          else
            wgmma_m64n128k16_rs<1>(O_acc[g], Afrag[j], desc_add(dV, j * MNMAJOR_K16));
        }
      }
    };

    if constexpr (OVERLAP) {
      wgmma_wait<0>();
      if (kv_id >= 1)
        release_bar(bVEmpty((kv_id - 1) % STAGES));
      pack_afrag();
      rescale_O(rescale);

      const int next = kv_id + 1;
      const int nstage = next % STAGES;
      if (next < num_kv_iter) {
        mbar_wait(bKFull(nstage), (next / STAGES) & 1);
        zero_S(S);
      }
      mbar_wait(bVFull(stage), (kv_id / STAGES) & 1);

      sched_sync();
      wgmma_fence();
      if (next < num_kv_iter)
        issue_qk(S, nstage);
      wgmma_commit();
      issue_pv(stage);
      wgmma_commit();
      sched_arrive();
      if (next < num_kv_iter) {
        wgmma_wait<1>();               // Q@K^T(next) is done; P@V(kv_id) is still running
        release_bar(bKEmpty(nstage));  // Release K now; keep V until its corresponding P@V finishes
      } else {
        wgmma_wait<0>();
        release_bar(bVEmpty(stage));
      }
    } else {
      pack_afrag();
      rescale_O(rescale);
      mbar_wait(bVFull(stage), (kv_id / STAGES) & 1);
      if (mask_any != 0) {
        wgmma_fence();
        issue_pv(stage);
        wgmma_commit();
        wgmma_wait<0>();
      }
      release_bar(bVEmpty(stage));
    }
  }


  // ---- Epilogue: O = C + acc/rowsum and cosine statistics ------------------------
#pragma unroll
  for (int d = 1; d <= 2; d *= 2) {
    rowsum[0] += __shfl_xor_sync(0xFFFFFFFFu, rowsum[0], d);
    rowsum[1] += __shfl_xor_sync(0xFFFFFFFFu, rowsum[1], d);
  }
  const float inv0 = rowsum[0] > 0.f ? 1.f / rowsum[0] : 0.f;
  const float inv1 = rowsum[1] > 0.f ? 1.f / rowsum[1] : 0.f;

  const int grow0 = q_row0 + cwg * 64 + r0;
  const int grow1 = grow0 + 8;
  const bool ok0 = grow0 < binfo.cu_seqlens_q_next;
  const bool ok1 = grow1 < binfo.cu_seqlens_q_next;
  float3 st0 = {0.f, 0.f, 0.f}, st1 = {0.f, 0.f, 0.f};

#pragma unroll
  for (int g = 0; g < NPV; ++g)
#pragma unroll
    for (int nb = 0; nb < ONB; ++nb) {
      const int col = g * OW + nb * 8 + c0;
      const int off0 = grow0 * ldg_q + head_id * DIM + col;
      const int off1 = grow1 * ldg_q + head_id * DIM + col;
      if (ok0) {
        float2 oldc = __bfloat1622float2(reinterpret_cast<const nv_bfloat162*>(C + off0)[0]);
        float2 newc = {oldc.x + O_acc[g][nb * 4 + 0] * inv0, oldc.y + O_acc[g][nb * 4 + 1] * inv0};
        reinterpret_cast<nv_bfloat162*>(O + off0)[0] = __float22bfloat162_rn(newc);
        st0.x += newc.x * oldc.x + newc.y * oldc.y;
        st0.y += newc.x * newc.x + newc.y * newc.y;
        st0.z += oldc.x * oldc.x + oldc.y * oldc.y;
      }
      if (ok1) {
        float2 oldc = __bfloat1622float2(reinterpret_cast<const nv_bfloat162*>(C + off1)[0]);
        float2 newc = {oldc.x + O_acc[g][nb * 4 + 2] * inv1, oldc.y + O_acc[g][nb * 4 + 3] * inv1};
        reinterpret_cast<nv_bfloat162*>(O + off1)[0] = __float22bfloat162_rn(newc);
        st1.x += newc.x * oldc.x + newc.y * oldc.y;
        st1.y += newc.x * newc.x + newc.y * newc.y;
        st1.z += oldc.x * oldc.x + oldc.y * oldc.y;
      }
    }

#pragma unroll
  for (int d = 1; d <= 2; d *= 2) {
    st0.x += __shfl_xor_sync(0xFFFFFFFFu, st0.x, d);
    st0.y += __shfl_xor_sync(0xFFFFFFFFu, st0.y, d);
    st0.z += __shfl_xor_sync(0xFFFFFFFFu, st0.z, d);
    st1.x += __shfl_xor_sync(0xFFFFFFFFu, st1.x, d);
    st1.y += __shfl_xor_sync(0xFFFFFFFFu, st1.y, d);
    st1.z += __shfl_xor_sync(0xFFFFFFFFu, st1.z, d);
  }
  if (lane_id % 4 == 0) {
    if (ok0) {
      atomicAdd(&cosine_stats[grow0 * 3 + 0], st0.x);
      atomicAdd(&cosine_stats[grow0 * 3 + 1], st0.y);
      atomicAdd(&cosine_stats[grow0 * 3 + 2], st0.z);
    }
    if (ok1) {
      atomicAdd(&cosine_stats[grow1 * 3 + 0], st1.x);
      atomicAdd(&cosine_stats[grow1 * 3 + 1], st1.y);
      atomicAdd(&cosine_stats[grow1 * 3 + 2], st1.z);
    }
  }
#else
  (void)tmQ; (void)tmK; (void)tmV; (void)C; (void)O; (void)B; (void)H; (void)H_kv;
  (void)cu_seqlens_q; (void)cu_seqlens_k; (void)max_seqlen_q; (void)max_seqlen_k;
  (void)cu_salientlens; (void)row_masks; (void)cosine_stats;
#endif
}

static PFN_cuTensorMapEncodeTiled_v12000 dyllm_tma_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qr);
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }();
  return fn;
}

// View [total_seq, heads, DIM] as a three-dimensional (DIM, heads, total_seq) TMA tensor.
// The box width is one swizzle block; rows beyond total_seq are filled with zero.
static CUtensorMap dyllm_make_map(const void* base, int total_seq, int heads, int dim, int box_rows) {
  CUtensorMap tm{};
  uint64_t gdim[3] = {static_cast<uint64_t>(dim), static_cast<uint64_t>(heads), static_cast<uint64_t>(total_seq)};
  uint64_t gstr[2] = {static_cast<uint64_t>(dim) * 2ull, static_cast<uint64_t>(heads) * dim * 2ull};
  uint32_t bdim[3] = {static_cast<uint32_t>(dyllm_hopper::SWZ_COLS), 1u, static_cast<uint32_t>(box_rows)};
  uint32_t estr[3] = {1u, 1u, 1u};
  CUresult r = dyllm_tma_encode()(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), gdim, gstr, bdim,
                                  estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return tm;
}

template <int DIM>
static void launch_for_dim(const nv_bfloat16* q_ptr, const nv_bfloat16* k_ptr, const nv_bfloat16* v_ptr,
                           const nv_bfloat16* c_ptr, nv_bfloat16* o_ptr, const int B, const int H, const int H_kv,
                           const int* cu_seqlens_q, const int* cu_seqlens_k, const int max_seqlen_q,
                           const int max_seqlen_k, const int total_seqlen_q, const int num_salient,
                           const int* cu_salientlens, const int* idx_salient_row_k, uint64_t* row_masks,
                           float* cosine_stats, int total_kv) {
  using T = Sm90Tile<DIM>;

  CUtensorMap tmQ = dyllm_make_map(q_ptr, total_seqlen_q, H, DIM, SM90_BLOCK_Q);
  CUtensorMap tmK = dyllm_make_map(k_ptr, total_kv, H_kv, DIM, T::BLOCK_K);
  CUtensorMap tmV = dyllm_make_map(v_ptr, total_kv, H_kv, DIM, T::BLOCK_K);

  compute_k_masks(B, cu_seqlens_k, cu_salientlens, idx_salient_row_k, num_salient, SM90_MASK_BLOCK, max_seqlen_k,
                  row_masks);

  const int num_blocks = B * H * cdiv(max_seqlen_q, SM90_BLOCK_Q);
  auto kernel = attention_h100_kernel<SM90_BLOCK_Q, T::BLOCK_K, DIM, SM90_STAGES, SM90_WS, SM90_MMA_WG, SM90_OVERLAP,
                                      Sm90Cfg<DIM>::MASK_FORM>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
  kernel<<<num_blocks, SM90_TB, T::SMEM>>>(tmQ, tmK, tmV, c_ptr, o_ptr, B, H, H_kv, cu_seqlens_q, cu_seqlens_k,
                                           max_seqlen_q, max_seqlen_k, cu_salientlens, row_masks, cosine_stats);
  CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void attention_sparse_varlen(const scalar_t* q, const scalar_t* k, const scalar_t* v, const scalar_t* c,
                             const scalar_t* o_sal, scalar_t* o, const int B, const int H, const int H_kv,
                             const int* cu_seqlens_q, const int* cu_seqlens_k, const int max_seqlen_q,
                             const int max_seqlen_k, const int total_seqlen_q, const int num_salient,
                             const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
                             uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim,
                             int force_mask_v_rows, int total_kv) {
  if constexpr (!std::is_same_v<scalar_t, at::BFloat16>) {
    std::cerr << "Only BFloat16 is supported" << std::endl;
    exit(1);
  }

  const auto* q_ptr = reinterpret_cast<const nv_bfloat16*>(q);
  const auto* k_ptr = reinterpret_cast<const nv_bfloat16*>(k);
  const auto* v_ptr = reinterpret_cast<const nv_bfloat16*>(v);
  const auto* c_ptr = reinterpret_cast<const nv_bfloat16*>(c);
  const auto* o_sal_ptr = reinterpret_cast<const nv_bfloat16*>(o_sal);
  auto* o_ptr = reinterpret_cast<nv_bfloat16*>(o);

#define DYLLM_SM90_LAUNCH(D)                                                                                           \
  launch_for_dim<D>(q_ptr, k_ptr, v_ptr, c_ptr, o_ptr, B, H, H_kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,           \
                    max_seqlen_k, total_seqlen_q, num_salient, cu_salientlens, idx_salient_row_k, row_masks,           \
                    cosine_stats, total_kv)
  bool handled = false;
  if constexpr (DYLLM_DIM_ENABLED(64)) {
    if (dim == 64) {
      DYLLM_SM90_LAUNCH(64);
      handled = true;
    }
  }
  if constexpr (DYLLM_DIM_ENABLED(128)) {
    if (!handled && dim == 128) {
      DYLLM_SM90_LAUNCH(128);
      handled = true;
    }
  }
  if constexpr (DYLLM_DIM_ENABLED(256)) {
    if (!handled && dim == 256) {
      DYLLM_SM90_LAUNCH(256);
      handled = true;
    }
  }
  TORCH_CHECK(handled, "attention_ops_kernels_sm90.cu implements dim=64/128/256, got ", dim);
#undef DYLLM_SM90_LAUNCH

  if (num_salient > 0)
    overwrite_salient_kernel<<<num_salient, 256>>>(c_ptr, o_sal_ptr, o_ptr, idx_salient_row_q, cosine_stats, H * dim);
  compute_cosine_similarity(cosine_stats, cosine_out, total_seqlen_q, cu_seqlens_q, threshold);
  (void)force_mask_v_rows;
}

template void attention_sparse_varlen<at::BFloat16>(const at::BFloat16*, const at::BFloat16*, const at::BFloat16*,
                                                    const at::BFloat16*, const at::BFloat16*, at::BFloat16*, const int,
                                                    const int, const int, const int*, const int*, const int, const int,
                                                    const int, const int, const int*, const int*, const int*,
                                                    uint64_t*, float*, bool*, float, int, int, int);

template <int DIM>
static std::vector<int64_t> kernel_info_for_dim() {
  using T = Sm90Tile<DIM>;
  cudaFuncAttributes attr{};
  auto kernel = attention_h100_kernel<SM90_BLOCK_Q, T::BLOCK_K, DIM, SM90_STAGES, SM90_WS, SM90_MMA_WG, SM90_OVERLAP,
                                      Sm90Cfg<DIM>::MASK_FORM>;
  if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess)
    return {0, 0, 0, 0, 0, 0, SM90_BLOCK_Q, T::BLOCK_K, 1, 0};
  int blocks = 0;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, SM90_TB, T::SMEM);
  return {static_cast<int64_t>(attr.numRegs),
          static_cast<int64_t>(attr.localSizeBytes),
          static_cast<int64_t>(attr.sharedSizeBytes),
          T::SMEM,
          SM90_TB,
          blocks,
          SM90_BLOCK_Q,
          T::BLOCK_K,
          1,
          1};
}

std::vector<int64_t> attention_kernel_info(int dim, bool mask_v_rows) {
  (void)mask_v_rows;
  if constexpr (DYLLM_DIM_ENABLED(64))
    if (dim == 64)
      return kernel_info_for_dim<64>();
  if constexpr (DYLLM_DIM_ENABLED(128))
    if (dim == 128)
      return kernel_info_for_dim<128>();
  if constexpr (DYLLM_DIM_ENABLED(256))
    if (dim == 256)
      return kernel_info_for_dim<256>();
  return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

} // namespace dyllm_sm90

void attention_sparse_varlen_h100(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                  const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                  const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                  const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                                  const int* idx_salient_row_q, uint64_t* row_masks, float* cosine_stats,
                                  bool* cosine_out, float threshold, int dim, int force_mask_v_rows, int total_kv) {
  dyllm_sm90::attention_sparse_varlen<at::BFloat16>(q, k, v, c, o_sal, o, B, H, H_kv, cu_seqlens_q, cu_seqlens_k,
      max_seqlen_q, max_seqlen_k, total_seqlen_q, num_salient, cu_salientlens, idx_salient_row_k, idx_salient_row_q,
      row_masks, cosine_stats, cosine_out, threshold, dim, force_mask_v_rows, total_kv);
}

std::vector<int64_t> attention_kernel_info_h100(int dim, bool mask_v_rows) {
  return dyllm_sm90::attention_kernel_info(dim, mask_v_rows);
}

int attention_sm90_binary_version() {
#if DYLLM_DIM_ENABLED(128)
  constexpr int PROBE_DIM = 128;
#elif DYLLM_DIM_ENABLED(64)
  constexpr int PROBE_DIM = 64;
#else
  constexpr int PROBE_DIM = 256;
#endif
  cudaFuncAttributes attr{};
  auto kernel = dyllm_sm90::attention_h100_kernel<
      dyllm_sm90::SM90_BLOCK_Q, dyllm_sm90::Sm90Tile<PROBE_DIM>::BLOCK_K, PROBE_DIM, dyllm_sm90::SM90_STAGES,
      dyllm_sm90::SM90_WS, dyllm_sm90::SM90_MMA_WG, dyllm_sm90::SM90_OVERLAP,
      dyllm_sm90::Sm90Cfg<PROBE_DIM>::MASK_FORM>;
  if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return attr.binaryVersion;
}
