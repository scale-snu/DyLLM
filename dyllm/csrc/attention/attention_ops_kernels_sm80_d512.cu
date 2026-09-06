#include "common.h"
#include "attention_aux.h"
#include "attn_tile_config.h"

#include <cuda_bf16.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <tuple>
#include <vector>

namespace dyllm_sm80_d512 {

// Log2-domain softmax. Because the scale is positive and does not change argmax,
// track rowmax before scaling (values published to md_smem use the same convention).
#ifndef DYLLM_D512_LOG2
#define DYLLM_D512_LOG2 1
#endif

// Number of Q rows per warp
#ifndef DYLLM_D512_WARP_Q
#define DYLLM_D512_WARP_Q 16
#endif

// Warp grid. NUM_WARPS_KV is the VO partition factor
// (O_rmem = WARP_Q/16 * DIM/WARPS_KV/8 * 4). The default is 4.
#ifndef DYLLM_D512_WARPS_Q
#define DYLLM_D512_WARPS_Q 4
#endif
#ifndef DYLLM_D512_WARPS_KV
#define DYLLM_D512_WARPS_KV 2
#endif

// Keys per CTA iteration. At least NUM_WARPS_KV*16 (an integral number of P tiles),
// and at most 64 (uint64 mask).
#ifndef DYLLM_D512_BLOCK_K
#define DYLLM_D512_BLOCK_K 32
#endif

#define DYLLM_D512_BLOCK_Q (DYLLM_D512_WARPS_Q * DYLLM_D512_WARP_Q)
#define DYLLM_D512_NUM_WARPS (DYLLM_D512_WARPS_Q * DYLLM_D512_WARPS_KV)

template <int BLOCK_Q, int BLOCK_K, int BLOCK_V, int DIM, int WARP_Q, int NUM_WARPS_Q, int NUM_WARPS_KV,
          bool MASK_V_ROWS>
__launch_bounds__(NUM_WARPS_Q* NUM_WARPS_KV* WARP_SIZE) __global__
    void attention_sparse_varlen_kernel(const nv_bfloat16* Q,     // [sum(len_q), H_q, D]
                                        const nv_bfloat16* K,     // [sum(len_kv), H_kv, D]
                                        const nv_bfloat16* V,     // [sum(len_kv), H_kv, D]
                                        const nv_bfloat16* C,     // [sum(len_kv), H_q, D]
                                        const nv_bfloat16* O_sal, // [sum(salientlen), H_q, D]
                                        nv_bfloat16* O,           // [sum(len_q), H_q, D]
                                        const int B, const int H, const int H_kv, const int* cu_seqlens_q,
                                        const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k,
                                        const int* cu_salientlens, const uint64_t* row_masks, float* cosine_stats) {

  constexpr int NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_KV;
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  // mma.m16n8k16
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  // Q tiles owned by one warp / the entire CTA
  constexpr int Q_MMA_TILES = WARP_Q / MMA_M;
  constexpr int Q_MMA_TILES_CTA = BLOCK_Q / MMA_M;
  // Key rows assigned to one warp during Q@K^T and the n8 tiles covering them
  constexpr int KV_PER_WARP = BLOCK_K / NUM_WARPS_KV;
  constexpr int NUM_MMA_KV = KV_PER_WARP / MMA_N;
  // P tiles produced by one warp / covering the entire BLOCK_K (MMA_K wide)
  constexpr int P_TILES_PER_WARP = KV_PER_WARP / MMA_K;
  constexpr int P_TILES_TOTAL = BLOCK_K / MMA_K;
  // Output columns owned by one warp during P@V
  constexpr int DIM_VO = DIM / NUM_WARPS_KV;

  static_assert(BLOCK_Q == NUM_WARPS_Q * WARP_Q, "BLOCK_Q must be NUM_WARPS_Q * WARP_Q");
  static_assert(WARP_Q % MMA_M == 0, "WARP_Q must be a multiple of MMA_M");
  static_assert(BLOCK_K % NUM_WARPS_KV == 0, "BLOCK_K must be a multiple of NUM_WARPS_KV");
  // [P_SMEM] A warp publishes a complete MMA_K-wide A operand, so a key shard cannot be narrower.
  static_assert(KV_PER_WARP % MMA_K == 0, "BLOCK_K/NUM_WARPS_KV must be a multiple of MMA_K (raise BLOCK_K)");
  static_assert(BLOCK_K <= 64, "row_masks packs one bit per key row into a uint64");
  static_assert(BLOCK_V == MMA_K, "BLOCK_V must equal MMA_K");
  static_assert(BLOCK_K % BLOCK_V == 0, "BLOCK_K must be a multiple of BLOCK_V");
  static_assert(DIM % NUM_WARPS_KV == 0, "DIM must be a multiple of NUM_WARPS_KV");
  static_assert(DIM_VO % (2 * MMA_N) == 0, "one ldmatrix.x4.trans covers 2*MMA_N output columns");
  static_assert((DIM / MMA_K) % 2 == 0, "one ldmatrix.x4 of K covers 2 head-dim tiles");

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int warp_q = warp_id % NUM_WARPS_Q;
  const int warp_kv = warp_id / NUM_WARPS_Q;

  const int num_blk_q = cdiv(max_seqlen_q, BLOCK_Q);
  const int num_blk_kv = cdiv(max_seqlen_k, BLOCK_K);
  const int batch_id = (bid / (H * num_blk_q));
  const int head_id = (bid % (H * num_blk_q)) / num_blk_q;
  const int blk_q_id = bid % num_blk_q;

  // GQA: query head -> KV head
  const int num_queries_per_kv = H / H_kv;
  const int kv_head_id = head_id / num_queries_per_kv;

  DyllmBlockInfo binfo(cu_seqlens_q, cu_seqlens_k, batch_id);

  const int num_kv_iter = cdiv(binfo.seqlen_kv, BLOCK_K);

  const uint64_t* row_masks_batch = row_masks + batch_id * num_blk_kv;

  const int ldg_q = H * DIM;     // Q, C, and O use num_q_heads.
  const int ldg_kv = H_kv * DIM; // K and V use num_kv_heads.

  const int batch_offset_q = binfo.cu_seqlens_q_curr * ldg_q;
  const int block_offset_q = blk_q_id * BLOCK_Q * ldg_q;
  const int head_offset_q = head_id * DIM;
  const int head_offset_kv = kv_head_id * DIM;
  const int batch_offset_kv = binfo.cu_seqlens_k_curr * ldg_kv;

  if (batch_offset_q + block_offset_q >= binfo.cu_seqlens_q_next * ldg_q)
    return;

  Q += batch_offset_q + block_offset_q + head_offset_q;
  K += batch_offset_kv + head_offset_kv;
  V += batch_offset_kv + head_offset_kv;
  C += batch_offset_q + block_offset_q + head_offset_q;
  O_sal += cu_salientlens[batch_id] * ldg_q + head_offset_q;
  O += batch_offset_q + block_offset_q + head_offset_q;

  // ---- Shared memory -----------------------------------------------------
  // Single Q | K | V buffers. All three start at multiples of eight rows and use the same swizzle pattern.
  extern __shared__ nv_bfloat16 smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem + BLOCK_Q * DIM * sizeof(nv_bfloat16);
  const uint32_t V_smem = K_smem + BLOCK_K * DIM * sizeof(nv_bfloat16);
  static_assert(BLOCK_Q % 8 == 0 && BLOCK_K % 8 == 0, "smem regions must start on an 8-row boundary");

  // [P_SMEM] One m16n8k16 A operand per lane for each (Q tile, P tile) pair (4 registers = 16 B).
  uint4* p_smem = reinterpret_cast<uint4*>(smem + (BLOCK_Q + 2 * BLOCK_K) * DIM);
  // Inter-warp softmax exchange: .x = partial rowmax, .y = partial rowsumexp.
  float2* md_smem = reinterpret_cast<float2*>(p_smem + Q_MMA_TILES_CTA * P_TILES_TOTAL * WARP_SIZE);

  // ---- Registers ---------------------------------------------------------
  uint32_t P_rmem[Q_MMA_TILES][P_TILES_PER_WARP][4];
  // [VO_SPLIT] DIM_VO columns instead of DIM columns.
  float O_rmem[Q_MMA_TILES][DIM_VO / MMA_N][4] = {};

  // Precompute ldmatrix addresses and swizzles.
  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    const int row_off = warp_q * WARP_Q + (lane_id % 16);
    const int col_off = (lane_id / 16) * 8;
    Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // B tile
    const int row_off = warp_kv * KV_PER_WARP + (lane_id % 8);
    const int col_off = (lane_id / 8) * 8;
    K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // Transposed B tile
    const int row_off = lane_id % 16;
    const int col_off = warp_kv * DIM_VO + (lane_id / 16) * 8;
    V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));
  
  constexpr bool LOG2_SOFTMAX = DYLLM_D512_LOG2 != 0;
  const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;

  float rowmax[Q_MMA_TILES][2];
  float rowsumexp[Q_MMA_TILES][2] = {};
  for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      const nv_bfloat16* src_k = K + kv_id * BLOCK_K * ldg_kv;
      global_to_shared_swizzle_zero_pad<BLOCK_K, DIM, TB_SIZE>(K_smem, src_k, ldg_kv, tid, kv_id, 0, binfo.seqlen_kv);
    }
    asm volatile("cp.async.commit_group;");
  };

  auto load_V = [&](int kv_id, uint64_t row_mask) {
    if (kv_id < num_kv_iter && row_mask != 0) {
      if constexpr (MASK_V_ROWS) {
        // Load V in BLOCK_K/BLOCK_V subchunks. Skip cp.async entirely for chunks with no
        // salient rows; this corresponds to skipping P@V below.
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
        const nv_bfloat16* src_v = V + kv_id * BLOCK_K * ldg_kv;
        global_to_shared_swizzle_zero_pad<BLOCK_K, DIM, TB_SIZE>(V_smem, src_v, ldg_kv, tid, kv_id, 0,
                                                                 binfo.seqlen_kv);
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  // ---- Prologue: Q and K(0) ----------------------------------------------
  global_to_shared_swizzle_zero_pad<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, ldg_q, tid, blk_q_id, 0, binfo.seqlen_q);
  asm volatile("cp.async.commit_group;");
  load_K(0);
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  uint64_t this_row_mask = num_kv_iter > 0 ? row_masks_batch[0] : 0;

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    load_V(kv_id, this_row_mask);

    float S_rmem[Q_MMA_TILES][NUM_MMA_KV][4] = {};

    // ---- Stage 1: S = Q @ K^T, partitioned by key (no redundant computation) ----
#pragma unroll
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
      const uint32_t d_xor = mma_id_d * MMA_K * sizeof(nv_bfloat16);

      uint32_t K_frag[NUM_MMA_KV][4];
#pragma unroll
      for (int mma_id_kv = 0; mma_id_kv < NUM_MMA_KV; mma_id_kv++)
        ldmatrix_x4(K_frag[mma_id_kv], (K_smem_thread + mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16)) ^ d_xor);

#pragma unroll
      for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
        const uint32_t q_base = Q_smem_thread + mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
        uint32_t Q_frag[2][4];
        ldmatrix_x4(Q_frag[0], q_base ^ d_xor);
        ldmatrix_x4(Q_frag[1], q_base ^ (d_xor + MMA_K * sizeof(nv_bfloat16)));
#pragma unroll
        for (int mma_id_kv = 0; mma_id_kv < NUM_MMA_KV; mma_id_kv++) {
          // K_frag[..][0..1] is the B operand for mma_id_d; [2..3] is for mma_id_d+1.
          mma_m16n8k16(Q_frag[0], &K_frag[mma_id_kv][0], S_rmem[mma_id_q][mma_id_kv]);
          mma_m16n8k16(Q_frag[1], &K_frag[mma_id_kv][2], S_rmem[mma_id_q][mma_id_kv]);
        }
      }
    }

    // ---- Stage 2a: Publish partial rowmax to md_smem ----------------------
    float this_rowmax[Q_MMA_TILES][2];
#pragma unroll
    for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
#pragma unroll
      for (int mma_id_kv = 0; mma_id_kv < NUM_MMA_KV; mma_id_kv++) {
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        if constexpr (!LOG2_SOFTMAX) {
#pragma unroll
          for (int reg_id = 0; reg_id < 4; reg_id++)
            regs[reg_id] *= softmax_scale;
        }

        const int key_col = kv_id * BLOCK_K + warp_kv * KV_PER_WARP + mma_id_kv * MMA_N + (lane_id % 4) * 2;
        if (key_col >= binfo.seqlen_kv) {
          regs[0] = -FLT_MAX;
          regs[2] = -FLT_MAX;
        }
        if (key_col + 1 >= binfo.seqlen_kv) {
          regs[1] = -FLT_MAX;
          regs[3] = -FLT_MAX;
        }

        if (mma_id_kv == 0) {
          this_rowmax[mma_id_q][0] = max(regs[0], regs[1]);
          this_rowmax[mma_id_q][1] = max(regs[2], regs[3]);
        } else {
          this_rowmax[mma_id_q][0] = max(this_rowmax[mma_id_q][0], max(regs[0], regs[1]));
          this_rowmax[mma_id_q][1] = max(this_rowmax[mma_id_q][1], max(regs[2], regs[3]));
        }
      }

      float* trm = this_rowmax[mma_id_q];
      trm[0] = max(trm[0], __shfl_xor_sync(0xFFFF'FFFF, trm[0], 1));
      trm[0] = max(trm[0], __shfl_xor_sync(0xFFFF'FFFF, trm[0], 2));
      trm[1] = max(trm[1], __shfl_xor_sync(0xFFFF'FFFF, trm[1], 1));
      trm[1] = max(trm[1], __shfl_xor_sync(0xFFFF'FFFF, trm[1], 2));

      if (lane_id % 4 == 0) {
        const int row0 = warp_q * WARP_Q + mma_id_q * MMA_M + lane_id / 4;
        md_smem[warp_kv * BLOCK_Q + row0].x = trm[0];
        md_smem[warp_kv * BLOCK_Q + row0 + 8].x = trm[1];
      }
    }

    __syncthreads();

    const uint64_t next_row_mask = kv_id + 1 < num_kv_iter ? row_masks_batch[kv_id + 1] : 0;
    load_K(kv_id + 1);

    const bool sparse_block = this_row_mask == 0;

    // ---- Stage 2b: Inter-warp rowmax, O rescaling, P -> p_smem ------------
    float rescale[Q_MMA_TILES][2];
#pragma unroll
    for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
      const int row0 = warp_q * WARP_Q + mma_id_q * MMA_M + lane_id / 4;
      float tile_max[2] = {md_smem[row0].x, md_smem[row0 + 8].x};
#pragma unroll
      for (int w = 1; w < NUM_WARPS_KV; w++) {
        tile_max[0] = max(tile_max[0], md_smem[w * BLOCK_Q + row0].x);
        tile_max[1] = max(tile_max[1], md_smem[w * BLOCK_Q + row0 + 8].x);
      }

      const float new_rowmax[2] = {max(rowmax[mma_id_q][0], tile_max[0]), max(rowmax[mma_id_q][1], tile_max[1])};
      
      if constexpr (LOG2_SOFTMAX) {
        rescale[mma_id_q][0] = exp2f((rowmax[mma_id_q][0] - new_rowmax[0]) * softmax_scale_log2);
        rescale[mma_id_q][1] = exp2f((rowmax[mma_id_q][1] - new_rowmax[1]) * softmax_scale_log2);
      } else {
        rescale[mma_id_q][0] = __expf(rowmax[mma_id_q][0] - new_rowmax[0]);
        rescale[mma_id_q][1] = __expf(rowmax[mma_id_q][1] - new_rowmax[1]);
      }
      rowmax[mma_id_q][0] = new_rowmax[0];
      rowmax[mma_id_q][1] = new_rowmax[1];

#pragma unroll
      for (int mma_id_d = 0; mma_id_d < DIM_VO / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[mma_id_q][0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[mma_id_q][0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[mma_id_q][1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[mma_id_q][1];
      }

      float this_rowsumexp[2];
      const float m2_0 = LOG2_SOFTMAX ? rowmax[mma_id_q][0] * softmax_scale_log2 : 0.0f;
      const float m2_1 = LOG2_SOFTMAX ? rowmax[mma_id_q][1] * softmax_scale_log2 : 0.0f;
#pragma unroll
      for (int mma_id_kv = 0; mma_id_kv < NUM_MMA_KV; mma_id_kv++) {
        float* regs = S_rmem[mma_id_q][mma_id_kv];
        if constexpr (LOG2_SOFTMAX) {
          regs[0] = exp2f(fmaf(regs[0], softmax_scale_log2, -m2_0)); // c0
          regs[1] = exp2f(fmaf(regs[1], softmax_scale_log2, -m2_0)); // c1
          regs[2] = exp2f(fmaf(regs[2], softmax_scale_log2, -m2_1)); // c2
          regs[3] = exp2f(fmaf(regs[3], softmax_scale_log2, -m2_1)); // c3
        } else {
          regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]); // c0
          regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]); // c1
          regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]); // c2
          regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]); // c3
        }

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = regs[0] + regs[1];
          this_rowsumexp[1] = regs[2] + regs[3];
        } else {
          this_rowsumexp[0] += regs[0] + regs[1];
          this_rowsumexp[1] += regs[2] + regs[3];
        }

        if (sparse_block)
          continue;
        nv_bfloat162* this_P_rmem = reinterpret_cast<nv_bfloat162*>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_rmem[(mma_id_kv % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      if (lane_id % 4 == 0) {
        md_smem[warp_kv * BLOCK_Q + row0].y = this_rowsumexp[0];
        md_smem[warp_kv * BLOCK_Q + row0 + 8].y = this_rowsumexp[1];
      }

      if (!sparse_block) {
#pragma unroll
        for (int t = 0; t < P_TILES_PER_WARP; t++) {
          const int slot = (warp_q * Q_MMA_TILES + mma_id_q) * P_TILES_TOTAL + warp_kv * P_TILES_PER_WARP + t;
          const uint32_t* src = P_rmem[mma_id_q][t];
          p_smem[slot * WARP_SIZE + lane_id] = make_uint4(src[0], src[1], src[2], src[3]);
        }
      }
    }
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

#pragma unroll
    for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
      const int row0 = warp_q * WARP_Q + mma_id_q * MMA_M + lane_id / 4;
      float tile_sum[2] = {md_smem[row0].y, md_smem[row0 + 8].y};
#pragma unroll
      for (int w = 1; w < NUM_WARPS_KV; w++) {
        tile_sum[0] += md_smem[w * BLOCK_Q + row0].y;
        tile_sum[1] += md_smem[w * BLOCK_Q + row0 + 8].y;
      }
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[mma_id_q][0] + tile_sum[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[mma_id_q][1] + tile_sum[1];
    }

    // ---- Stage 3: O += P @ V, partitioned by output column ----------------
    if (!sparse_block) {
#pragma unroll
      for (int t = 0; t < P_TILES_TOTAL; t++) {
        // Recompute the mask fragment for this MMA_K-row chunk, and skip both ldmatrix
        // and mma if it contains no salient rows.
        const uint64_t sub_mask = (this_row_mask >> (t * MMA_K)) & ((1ULL << MMA_K) - 1);
        if (sub_mask == 0)
          continue;
        uint32_t P_frag[Q_MMA_TILES][4];
#pragma unroll
        for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
          const int slot = (warp_q * Q_MMA_TILES + mma_id_q) * P_TILES_TOTAL + t;
          const uint4 f = p_smem[slot * WARP_SIZE + lane_id];
          P_frag[mma_id_q][0] = f.x;
          P_frag[mma_id_q][1] = f.y;
          P_frag[mma_id_q][2] = f.z;
          P_frag[mma_id_q][3] = f.w;
        }

        const uint32_t v_base = V_smem_thread + t * MMA_K * DIM * sizeof(nv_bfloat16);
#pragma unroll
        for (int mma_id_d = 0; mma_id_d < DIM_VO / MMA_N; mma_id_d += 2) {
          uint32_t V_frag[4];
          ldmatrix_x4_trans(V_frag, v_base ^ (mma_id_d * MMA_N * sizeof(nv_bfloat16)));
#pragma unroll
          for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
            mma_m16n8k16(P_frag[mma_id_q], &V_frag[0], O_rmem[mma_id_q][mma_id_d]);
            mma_m16n8k16(P_frag[mma_id_q], &V_frag[2], O_rmem[mma_id_q][mma_id_d + 1]);
          }
        }
      }
    }

    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    this_row_mask = next_row_mask;
  }

  // ---- Epilogue ----------------------------------------------------------
  // No inter-warp O reduction
  for (int mma_id_q = 0; mma_id_q < Q_MMA_TILES; mma_id_q++) {
    const int row = warp_q * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
    const int global_row_0 = binfo.cu_seqlens_q_curr + blk_q_id * BLOCK_Q + row;
    if (global_row_0 >= binfo.cu_seqlens_q_next)
      continue;

    const bool valid_1 = global_row_0 + 8 < binfo.cu_seqlens_q_next;
    const float inv_rowsum_0 = __frcp_rn(rowsumexp[mma_id_q][0]);
    const float inv_rowsum_1 = valid_1 ? __frcp_rn(rowsumexp[mma_id_q][1]) : 0.0f;

    float3 stats_0 = {0.0f, 0.0f, 0.0f};
    float3 stats_1 = {0.0f, 0.0f, 0.0f};
    for (int mma_id_d = 0; mma_id_d < DIM_VO / MMA_N; mma_id_d++) {
      const int col = warp_kv * DIM_VO + mma_id_d * MMA_N + (lane_id % 4) * 2;

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

// Host side

namespace {

constexpr int D512_BLOCK_Q = DYLLM_D512_BLOCK_Q;
constexpr int D512_BLOCK_K = DYLLM_D512_BLOCK_K;
constexpr int D512_BLOCK_V = 16;
constexpr int D512_WARP_Q = DYLLM_D512_WARP_Q;
constexpr int D512_WARPS_Q = DYLLM_D512_WARPS_Q;
constexpr int D512_WARPS_KV = DYLLM_D512_WARPS_KV;
constexpr int D512_NUM_WARPS = DYLLM_D512_NUM_WARPS;
constexpr int D512_TB_SIZE = D512_NUM_WARPS * WARP_SIZE;

#ifndef DYLLM_D512_DIM
#define DYLLM_D512_DIM 512
#endif
constexpr int D512_DIM = DYLLM_D512_DIM;

// Q | K | V (single buffers) + p_smem + md_smem.
constexpr int d512_smem_bytes() {
  return (D512_BLOCK_Q + 2 * D512_BLOCK_K) * D512_DIM * static_cast<int>(sizeof(nv_bfloat16))  // Q, K, V
         + (D512_BLOCK_Q / 16) * (D512_BLOCK_K / 16) * WARP_SIZE * 16                        // p_smem
         + D512_WARPS_KV * D512_BLOCK_Q * static_cast<int>(sizeof(float2));                  // md_smem
}

} // namespace

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
                             float* cosine_stats, // [sum(len_q), 3] - intermediate buffer; must be zero-initialized
                             bool* cosine_out,    // [sum(len_q)] - output bool mask (true if cos_sim < threshold)
                             float threshold, int dim,
                             int force_mask_v_rows) { // Benchmark-only override: -1=auto, 0=force B, 1=force A

  if constexpr (!std::is_same_v<scalar_t, at::BFloat16>) {
    TORCH_CHECK(false, "Only BFloat16 is supported");
  }

  TORCH_CHECK(dim == D512_DIM, "attention_ops_kernels_sm80_d512.cu only implements dim=", D512_DIM,
              ", got ", dim);

  const nv_bfloat16* q_ptr = reinterpret_cast<const nv_bfloat16*>(q);
  const nv_bfloat16* k_ptr = reinterpret_cast<const nv_bfloat16*>(k);
  const nv_bfloat16* v_ptr = reinterpret_cast<const nv_bfloat16*>(v);
  const nv_bfloat16* c_ptr = reinterpret_cast<const nv_bfloat16*>(c);
  const nv_bfloat16* o_sal_ptr = reinterpret_cast<const nv_bfloat16*>(o_sal);
  nv_bfloat16* o_ptr = reinterpret_cast<nv_bfloat16*>(o);

  const int num_blocks = B * H * cdiv(max_seqlen_q, D512_BLOCK_Q);

  static_assert(D512_BLOCK_K >= 16 && D512_BLOCK_K <= 64 && D512_BLOCK_K % 16 == 0,
                "BLOCK_K must fit one uint64 row mask and preserve MMA tiling");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dyllm_compute_k_masks(B, cu_seqlens_k, cu_salientlens, idx_salient_row_k, D512_BLOCK_K, max_seqlen_k, row_masks,
                        stream);

  auto launch_kernel_fn = [&](auto kernel) {
    launch_kernel(kernel, num_blocks, D512_TB_SIZE, d512_smem_bytes(), q_ptr, k_ptr, v_ptr, c_ptr, o_sal_ptr, o_ptr, B, H,
                  H_kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, cu_salientlens, row_masks,
                  cosine_stats);
  };

  const bool mask_v_rows = (force_mask_v_rows >= 0) ? (force_mask_v_rows != 0) : (num_blocks <= 1024);
  if (mask_v_rows)
    launch_kernel_fn(attention_sparse_varlen_kernel<D512_BLOCK_Q, D512_BLOCK_K, D512_BLOCK_V, D512_DIM, D512_WARP_Q, D512_WARPS_Q,
                                                    D512_WARPS_KV, true>);
  else
    launch_kernel_fn(attention_sparse_varlen_kernel<D512_BLOCK_Q, D512_BLOCK_K, D512_BLOCK_V, D512_DIM, D512_WARP_Q, D512_WARPS_Q,
                                                    D512_WARPS_KV, false>);

  dyllm_overwrite_salient_and_accumulate_stats(c_ptr, o_sal_ptr, o_ptr, idx_salient_row_q, cosine_stats,
                                               num_salient, H * dim, stream);
  dyllm_compute_cosine_mask(cosine_stats, cosine_out, total_seqlen_q, threshold, stream);
}

template void attention_sparse_varlen<at::BFloat16>(
    const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v, const at::BFloat16* c,
    const at::BFloat16* o_sal, at::BFloat16* o, const int B, const int H, const int H_kv, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
    const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
    uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim, int force_mask_v_rows);

// Kernel launch resources for sweeps. The d_split field reports NUM_WARPS_KV.
// Returns [registers, local bytes, static smem, dynamic smem, tb_size,
//          max active blocks per SM, block_q, block_k, d_split, ok].
std::vector<int64_t> attention_kernel_info(int dim, bool mask_v_rows) {
  const int smem_size = d512_smem_bytes();

  if (dim != D512_DIM)
    return {0, 0, 0, smem_size, D512_TB_SIZE, 0, D512_BLOCK_Q, D512_BLOCK_K, D512_WARPS_KV, 0};

  const void* kernel =
      mask_v_rows
          ? reinterpret_cast<const void*>(&attention_sparse_varlen_kernel<D512_BLOCK_Q, D512_BLOCK_K, D512_BLOCK_V, D512_DIM,
                                                                          D512_WARP_Q, D512_WARPS_Q, D512_WARPS_KV, true>)
          : reinterpret_cast<const void*>(&attention_sparse_varlen_kernel<D512_BLOCK_Q, D512_BLOCK_K, D512_BLOCK_V, D512_DIM,
                                                                          D512_WARP_Q, D512_WARPS_Q, D512_WARPS_KV, false>);

  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, kernel) != cudaSuccess) {
    cudaGetLastError();
    return {0, 0, 0, smem_size, D512_TB_SIZE, 0, D512_BLOCK_Q, D512_BLOCK_K, D512_WARPS_KV, 0};
  }

  // Exceeding the limit causes the >48 KB smem opt-in to fail. Report ok=0 instead
  // of aborting so the sweep can continue.
  int64_t ok = 1;
  if (smem_size > 48'000) {
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) != cudaSuccess) {
      cudaGetLastError();
      ok = 0;
    }
  }
  int blocks = 0;
  if (ok && cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, D512_TB_SIZE, smem_size) != cudaSuccess) {
    cudaGetLastError();
    blocks = 0;
  }

  return {attr.numRegs,
          static_cast<int64_t>(attr.localSizeBytes),
          static_cast<int64_t>(attr.sharedSizeBytes),
          smem_size,
          D512_TB_SIZE,
          blocks,
          D512_BLOCK_Q,
          D512_BLOCK_K,
          D512_WARPS_KV,
          ok};
}

} // namespace dyllm_sm80_d512

// Global entry point.
void attention_sparse_varlen_sm80_d512(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v, const at::BFloat16* c,
    const at::BFloat16* o_sal, at::BFloat16* o, const int B, const int H, const int H_kv, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
    const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
    uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim, int force_mask_v_rows) {
  dyllm_sm80_d512::attention_sparse_varlen<at::BFloat16>(q, k, v, c, o_sal, o, B, H, H_kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
      total_seqlen_q, num_salient, cu_salientlens, idx_salient_row_k, idx_salient_row_q, row_masks, cosine_stats,
      cosine_out, threshold, dim, force_mask_v_rows);
}

std::vector<int64_t> attention_kernel_info_sm80_d512(int dim, bool mask_v_rows) {
  return dyllm_sm80_d512::attention_kernel_info(dim, mask_v_rows);
}

#ifndef DYLLM_D512_SECONDARY

template <typename scalar_t>
void attention_sparse_varlen(const scalar_t* q, const scalar_t* k, const scalar_t* v, const scalar_t* c,
                             const scalar_t* o_sal, scalar_t* o, const int B, const int H, const int H_kv,
                             const int* cu_seqlens_q, const int* cu_seqlens_k, const int max_seqlen_q,
                             const int max_seqlen_k, const int total_seqlen_q, const int num_salient,
                             const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
                             uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim,
                             int force_mask_v_rows) {
  dyllm_sm80_d512::attention_sparse_varlen<scalar_t>(q, k, v, c, o_sal, o, B, H, H_kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
      total_seqlen_q, num_salient, cu_salientlens, idx_salient_row_k, idx_salient_row_q, row_masks, cosine_stats,
      cosine_out, threshold, dim, force_mask_v_rows);
}

template void attention_sparse_varlen<at::BFloat16>(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v, const at::BFloat16* c,
    const at::BFloat16* o_sal, at::BFloat16* o, const int B, const int H, const int H_kv, const int* cu_seqlens_q,
    const int* cu_seqlens_k, const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
    const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
    uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim, int force_mask_v_rows);

std::vector<int64_t> attention_kernel_info(int dim, bool mask_v_rows) {
  return dyllm_sm80_d512::attention_kernel_info(dim, mask_v_rows);
}

#endif // DYLLM_D512_SECONDARY
