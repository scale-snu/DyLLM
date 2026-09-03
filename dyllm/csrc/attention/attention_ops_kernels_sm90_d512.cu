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

#ifndef DYLLM_H100_D512_OVERLAP
#define DYLLM_H100_D512_OVERLAP 0
#endif

#ifndef DYLLM_H100_D512_MASK_FORM
#define DYLLM_H100_D512_MASK_FORM 1
#endif

namespace dyllm_h100_d512 {

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

template <int BLOCK_Q, int BLOCK_K, int DIM, int STAGES, bool WS, int NUM_MMA_WG, bool OVERLAP>
__launch_bounds__(WS ? (1 + NUM_MMA_WG) * 128 : 128, 1) __global__
    void attention_h100_kernel(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                               const __grid_constant__ CUtensorMap tmV, const nv_bfloat16* __restrict__ C,
                               nv_bfloat16* __restrict__ O, const int B, const int H, const int H_kv,
                               const int* __restrict__ cu_seqlens_q, const int* __restrict__ cu_seqlens_k,
                               const int max_seqlen_q, const int max_seqlen_k,
                               const int* __restrict__ cu_salientlens, const uint64_t* __restrict__ row_masks,
                               float* __restrict__ cosine_stats) {
#if DYLLM_SM90A
  static_assert(BLOCK_Q == 64, "VO-split: both MMA warpgroups own the same 64 Q rows");
  static_assert(NUM_MMA_WG == 2, "VO-split needs exactly two MMA warpgroups");
  static_assert(WS, "the split is expressed against a separate producer warpgroup");
  static_assert(!OVERLAP, "intra-warpgroup overlap does not compose with the P exchange");
  static_assert(DIM % (2 * 128) == 0, "each warpgroup's output half is issued n128 at a time");
  static_assert(BLOCK_K % 32 == 0, "each warpgroup owns BLOCK_K/2 keys, a whole k16 step");
  static_assert(BLOCK_K <= 64, "sP is one 128 B swizzle block per row");

  constexpr int NCB = DIM / SWZ_COLS;            // Number of 128-byte column blocks per row (8 for DIM=512)
  constexpr int Q_CB = BLOCK_Q * SWZ_ROW_BYTES;  // Bytes in one Q column block
  constexpr int KV_CB = BLOCK_K * SWZ_ROW_BYTES; // Bytes in one K/V column block
  constexpr int SQ_BYTES = NCB * Q_CB;
  constexpr int SKV_BYTES = NCB * KV_CB;
  constexpr int NKSTEP = BLOCK_K / 16;           // Number of k16 steps in the *entire* tile processed by P@V
  constexpr int KPW = BLOCK_K / 2;               // Number of Q@K^T keys owned by this warpgroup
  constexpr int NNB = KPW / 8;                   // Number of n-blocks in S
  constexpr int NREG = KPW / 2;                  // Number of S accumulator registers
  constexpr int DIM_VO = DIM / 2;                // Number of output columns owned by this warpgroup
  constexpr int NPV = DIM_VO / 128;              // Number of n128 P@V groups -> O_acc uses 2*64=128 registers
  constexpr int SP_BYTES = 64 * SWZ_ROW_BYTES;   // sP: 64 rows x one swizzle block = 8 KB
  constexpr int MNMAJOR_K16 = K16_STEP_MNMAJOR_BYTES;

  // [Registers]
  //     producer 128 * 24  +  consumers 256 * 240  =  168 * 384 = 64512 exactly
  constexpr int PRODUCER_REGS = 24;
  constexpr int CONSUMER_REGS = 240;
  static_assert((168 - PRODUCER_REGS) * 128 >= (CONSUMER_REGS - 168) * 256,
                "setmaxnreg would deadlock: consumers ask for more than the producer frees");
  constexpr int CONSUMER_BAR = 1; // IDs 1 and 2 protect buffer release for each warpgroup
  constexpr int MD_BAR = 3;       // Exchange rowmax between the two MMA warpgroups
  constexpr int P_BAR = 4;        // Exchange P between the two MMA warpgroups
  constexpr int MMA_THREADS = NUM_MMA_WG * 128;

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sQ = smem;
  uint8_t* sK = sQ + SQ_BYTES;
  uint8_t* sV = sK + STAGES * SKV_BYTES;
  uint8_t* sP = sV + STAGES * SKV_BYTES;
  float* sMD = reinterpret_cast<float*>(sP + SP_BYTES);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sMD + 2 * 64);

  const uint32_t aQ = static_cast<uint32_t>(__cvta_generic_to_shared(sQ));
  const uint32_t aK = static_cast<uint32_t>(__cvta_generic_to_shared(sK));
  const uint32_t aV = static_cast<uint32_t>(__cvta_generic_to_shared(sV));
  const uint32_t aP = static_cast<uint32_t>(__cvta_generic_to_shared(sP));
  const uint32_t aBar = static_cast<uint32_t>(__cvta_generic_to_shared(bars));
  // [Split K/V] K and V each have a full/empty barrier pair. K is no longer
  // needed after Q@K^T and V after P@V, so loads overlap compute even with STAGES=1.
  //
  // [0] Q, [1..S] K full, [1+S..1+2S] K empty, [1+2S..1+3S] V full, [1+3S..1+4S] V empty
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

  // ---------------------------------------------------------------- Producer
  if (tid < 128) {
    setmaxnreg_dec<PRODUCER_REGS>();
    if (tid == 0) {
      mbar_expect_tx(bQ, SQ_BYTES);
      for (int cb = 0; cb < NCB; ++cb)
        tma_load_3d(aQ + cb * Q_CB, &tmQ, bQ, cb * SWZ_COLS, head_id, q_row0);
      for (int kv_id = 0; kv_id < num_kv_iter; ++kv_id) {
        const int stage = kv_id % STAGES;
        const uint32_t phase = ((kv_id / STAGES) - 1) & 1;
        // Loading K first and releasing it earlier overlaps this load with the previous tile's softmax and P@V.
        if (kv_id >= STAGES)
          mbar_wait(bKEmpty(stage), phase);
        issue_k(kv_id, stage);
        if (kv_id >= STAGES)
          mbar_wait(bVEmpty(stage), phase);
        issue_v(kv_id, stage);
      }
    }
    return;
  }
  setmaxnreg_inc<CONSUMER_REGS>();

  // ---------------------------------------------------------------- Consumer
  const int cwg = __shfl_sync(0xFFFFFFFFu, static_cast<int>(threadIdx.x / 128) - 1, 0);
  const int math_tid = tid - (cwg + 1) * 128;
  const int warp_id = math_tid / WARP_SIZE;
  const int lane_id = math_tid % WARP_SIZE;

  const int r0 = warp_id * 16 + lane_id / 4;
  const int c0 = (lane_id % 4) * 2;
  const int key0 = cwg * KPW;      // First key in the S fragment owned by this warpgroup
  const int cb0 = cwg * (DIM_VO / SWZ_COLS); // First V/O column block owned by this warpgroup

  float O_acc[NPV][64];
#pragma unroll
  for (int g = 0; g < NPV; ++g)
#pragma unroll
    for (int i = 0; i < 64; ++i)
      O_acc[g][i] = 0.f;

  float rowmax[2] = {-FLT_MAX, -FLT_MAX}; // Kept in the original scale domain and scaled only once
  float rowsum[2] = {0.f, 0.f};           // Includes only this warpgroup's keys; summed in the epilogue

  const float softmax_scale = rsqrtf(static_cast<float>(DIM));
  const float scale_log2 = softmax_scale * 1.4426950408889634f;

  mbar_wait(bQ, 0); // Q is resident in shared memory

  float S[NREG];
  const uint64_t descQ = desc_kmajor(aQ);

  auto issue_qk = [&](int st) {
    const uint64_t dK = desc_kmajor(aK + st * SKV_BYTES);
#pragma unroll
    for (int cb = 0; cb < NCB; ++cb)
#pragma unroll
      for (int kk = 0; kk < SWZ_COLS / 16; ++kk) {
        const uint64_t da = desc_add(descQ, cb * Q_CB + kk * K16_STEP_BYTES);
        const uint64_t db = desc_add(dK, cb * KV_CB + key0 * SWZ_ROW_BYTES + kk * K16_STEP_BYTES);
        if constexpr (KPW == 16)
          wgmma_m64n16k16_ss<1>(S, da, db);
        else
          wgmma_m64n32k16_ss<1>(S, da, db);
      }
  };
  auto rescale_O = [&](const float (&r)[2]) {
#pragma unroll
    for (int g = 0; g < NPV; ++g)
#pragma unroll
      for (int nb = 0; nb < 16; ++nb) {
        O_acc[g][nb * 4 + 0] *= r[0];
        O_acc[g][nb * 4 + 1] *= r[0];
        O_acc[g][nb * 4 + 2] *= r[1];
        O_acc[g][nb * 4 + 3] *= r[1];
      }
  };

  auto release = [&](uint32_t bar) {
    if (cwg == 0)
      named_barrier_sync<CONSUMER_BAR, 128>();
    else
      named_barrier_sync<CONSUMER_BAR + 1, 128>();
    if (math_tid == 0)
      mbar_arrive(bar);
  };
  auto store_P = [&](int row, int chunk, uint32_t v) {
    *reinterpret_cast<uint32_t*>(sP + row * SWZ_ROW_BYTES + ((chunk ^ (row & 7)) * 16) + c0 * 2) = v;
  };

  for (int kv_id = 0; kv_id < num_kv_iter; ++kv_id) {
    const int stage = kv_id % STAGES;
    const uint32_t phase = (kv_id / STAGES) & 1;
    mbar_wait(bKFull(stage), phase);

    // ---------------- S = Q @ K^T, process only this warpgroup's keys ----------------
#pragma unroll
    for (int i = 0; i < NREG; ++i)
      S[i] = 0.f;
    wgmma_fence();
    issue_qk(stage);
    wgmma_commit();
    wgmma_wait<0>();
    // K is no longer used, so release its buffer immediately. The next K load
    // overlaps the softmax and P@V below instead of waiting for them to finish.
    release(bKEmpty(stage));

    // ---------------- Tail mask + online softmax ---------------------
    const bool tail = (kv_id + 1) * BLOCK_K > binfo.seqlen_kv;
    if (tail) {
#pragma unroll
      for (int nb = 0; nb < NNB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int key = kv_id * BLOCK_K + key0 + nb * 8 + c0 + (i & 1);
          if (key >= binfo.seqlen_kv)
            S[nb * 4 + i] = -FLT_MAX;
        }
    }

    float m_new[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
    for (int nb = 0; nb < NNB; ++nb) {
      m_new[0] = fmaxf(m_new[0], fmaxf(S[nb * 4 + 0], S[nb * 4 + 1]));
      m_new[1] = fmaxf(m_new[1], fmaxf(S[nb * 4 + 2], S[nb * 4 + 3]));
    }
    // The four lanes in a quad share a row, so reduce rowmax with a butterfly pattern.
#pragma unroll
    for (int d = 1; d <= 2; d *= 2) {
      m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xFFFFFFFFu, m_new[0], d));
      m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xFFFFFFFFu, m_new[1], d));
    }
    // At this point, rowmax includes only the keys owned by this warpgroup.
    if (lane_id % 4 == 0) {
      sMD[cwg * 64 + r0] = m_new[0];
      sMD[cwg * 64 + r0 + 8] = m_new[1];
    }
    named_barrier_sync<MD_BAR, MMA_THREADS>();
    m_new[0] = fmaxf(m_new[0], sMD[(1 - cwg) * 64 + r0]);
    m_new[1] = fmaxf(m_new[1], sMD[(1 - cwg) * 64 + r0 + 8]);

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
    for (int nb = 0; nb < NNB; ++nb) {
      S[nb * 4 + 0] = exp2f(fmaf(S[nb * 4 + 0], scale_log2, -m2_0));
      S[nb * 4 + 1] = exp2f(fmaf(S[nb * 4 + 1], scale_log2, -m2_0));
      S[nb * 4 + 2] = exp2f(fmaf(S[nb * 4 + 2], scale_log2, -m2_1));
      S[nb * 4 + 3] = exp2f(fmaf(S[nb * 4 + 3], scale_log2, -m2_1));
      psum[0] += S[nb * 4 + 0] + S[nb * 4 + 1];
      psum[1] += S[nb * 4 + 2] + S[nb * 4 + 3];
    }
    // Both warpgroups now have the same rowmax and rescale values, so their
    // partial rowsums share a common basis and can be combined in the epilogue.
    rowsum[0] = rowsum[0] * rescale[0] + psum[0];
    rowsum[1] = rowsum[1] * rescale[1] + psum[1];

    // ---------------- Publish P, then run P@V over all keys -------------
    constexpr int NMW = (BLOCK_K + 63) / 64;
    uint64_t mw[NMW];
    uint64_t mask_any = 0;
#pragma unroll
    for (int w = 0; w < NMW; ++w) {
      mw[w] = row_masks_batch[(kv_id * BLOCK_K) / 64 + w];
      mask_any |= mw[w];
    }
#pragma unroll
    for (int nb = 0; nb < NNB; ++nb) {
      const int gk = kv_id * BLOCK_K + key0 + nb * 8;
      const uint64_t m = mw[(key0 + nb * 8) >> 6];
      const int kb = gk & 63; // An 8-key n-block never crosses a mask-word boundary.
#if DYLLM_H100_D512_MASK_FORM == 0
      auto keep = [&](float v, int koff) { return ((m >> (kb + koff)) & 1ull) ? v : 0.f; };
      const int chunk = (key0 >> 3) + nb;
      store_P(r0, chunk, pack_bf16x2(keep(S[nb * 4 + 0], c0), keep(S[nb * 4 + 1], c0 + 1)));
      store_P(r0 + 8, chunk, pack_bf16x2(keep(S[nb * 4 + 2], c0), keep(S[nb * 4 + 3], c0 + 1)));
#else
      const uint32_t bits = static_cast<uint32_t>((m >> (kb + c0)) & 3ull);
      const uint32_t lo = static_cast<uint32_t>(-static_cast<int32_t>(bits & 1u)) & 0x0000FFFFu;
      const uint32_t hi = static_cast<uint32_t>(-static_cast<int32_t>(bits >> 1)) & 0xFFFF0000u;
      const uint32_t pair_mask = lo | hi;
      const int chunk = (key0 >> 3) + nb;
      store_P(r0, chunk, pack_bf16x2(S[nb * 4 + 0], S[nb * 4 + 1]) & pair_mask);
      store_P(r0 + 8, chunk, pack_bf16x2(S[nb * 4 + 2], S[nb * 4 + 3]) & pair_mask);
#endif
    }
    fence_async_shared();
    named_barrier_sync<P_BAR, MMA_THREADS>();

    bool live[NKSTEP];
#pragma unroll
    for (int j = 0; j < NKSTEP; ++j) {
      const int gk = kv_id * BLOCK_K + j * 16;
      live[j] = ((mw[(j * 16) >> 6] >> (gk & 63)) & 0xFFFFull) != 0;
    }

    rescale_O(rescale);
    mbar_wait(bVFull(stage), phase);
    if (mask_any != 0) {
      const uint64_t dP = desc_kmajor(aP);
      wgmma_fence();
#pragma unroll
      for (int g = 0; g < NPV; ++g) {
        const uint64_t dV = desc_mnmajor_n128(aV + stage * SKV_BYTES + (cb0 + 2 * g) * KV_CB, KV_CB);
#pragma unroll
        for (int j = 0; j < NKSTEP; ++j) {
          if (!live[j])
            continue; // The entire k16 step is masked: skip the instruction instead of loading zeros.
          wgmma_m64n128k16_ss<1, 1>(O_acc[g], desc_add(dP, j * K16_STEP_BYTES),
                                    desc_add(dV, j * MNMAJOR_K16));
        }
      }
      wgmma_commit();
      wgmma_wait<0>();
    }
    release(bVEmpty(stage));
  }

  // ---- Epilogue: O = C + acc/rowsum and cosine statistics ------------------------
#pragma unroll
  for (int d = 1; d <= 2; d *= 2) {
    rowsum[0] += __shfl_xor_sync(0xFFFFFFFFu, rowsum[0], d);
    rowsum[1] += __shfl_xor_sync(0xFFFFFFFFu, rowsum[1], d);
  }
  if (lane_id % 4 == 0) {
    sMD[cwg * 64 + r0] = rowsum[0];
    sMD[cwg * 64 + r0 + 8] = rowsum[1];
  }
  named_barrier_sync<MD_BAR, MMA_THREADS>();
  rowsum[0] += sMD[(1 - cwg) * 64 + r0];
  rowsum[1] += sMD[(1 - cwg) * 64 + r0 + 8];

  const float inv0 = rowsum[0] > 0.f ? 1.f / rowsum[0] : 0.f;
  const float inv1 = rowsum[1] > 0.f ? 1.f / rowsum[1] : 0.f;

  const int grow0 = q_row0 + r0;
  const int grow1 = grow0 + 8;
  const bool ok0 = grow0 < binfo.cu_seqlens_q_next;
  const bool ok1 = grow1 < binfo.cu_seqlens_q_next;
  float3 st0 = {0.f, 0.f, 0.f}, st1 = {0.f, 0.f, 0.f};

#pragma unroll
  for (int g = 0; g < NPV; ++g)
#pragma unroll
    for (int nb = 0; nb < 16; ++nb) {
      const int col = cb0 * SWZ_COLS + g * 128 + nb * 8 + c0;
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
// The box width is one swizzle block, so a DIM=512 row spans eight boxes. Out-of-bounds data is zero-filled.
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

static constexpr int H100_DIM = 512;
// Fixed by the VO-split design rather than a tuning parameter. The two MMA
// warpgroups split the head dimension, not Q rows, so BLOCK_Q is 64 and the CTA contains two warpgroups.
static constexpr int H100_MMA_WG = 2;
static constexpr int H100_BLOCK_Q = 64;
// BLOCK_K may be only 32 or 64; the default is 64 with STAGES=1. At 32, the
// VO split halves the tile again across warpgroups, issuing only m64n16k16 per warpgroup.
#ifndef DYLLM_H100_D512_BLOCK_K
#define DYLLM_H100_D512_BLOCK_K 64
#endif
static constexpr int H100_BLOCK_K = DYLLM_H100_D512_BLOCK_K;
// row_masks contains one uint64 per key block. Its 64 bits represent 64 rows regardless of tile size.
static constexpr int H100_MASK_BLOCK = 64;
#ifndef DYLLM_H100_D512_STAGES
#define DYLLM_H100_D512_STAGES 1
#endif
static constexpr int H100_STAGES = DYLLM_H100_D512_STAGES;
#ifndef DYLLM_H100_D512_WARPSPEC
#define DYLLM_H100_D512_WARPSPEC 1
#endif
static constexpr bool H100_WS = DYLLM_H100_D512_WARPSPEC != 0;
static constexpr bool H100_OVERLAP = DYLLM_H100_D512_OVERLAP != 0;

static constexpr int H100_TB = H100_WS ? (1 + H100_MMA_WG) * 128 : 128;

// After the Q + STAGES*(K,V) tiles, place sP (64x64 bf16 P exchange), sMD
// (2x64 f32 rowmax/rowsum exchange), and mbarriers in that order.
static constexpr int H100_SMEM = (H100_BLOCK_Q + 2 * H100_STAGES * H100_BLOCK_K) * H100_DIM * 2 // Q, K, V
                                 + 64 * 128                                                     // sP
                                 + 2 * 64 * 4                                                   // sMD
                                 + 128 * (1 + 2 * H100_STAGES);                                 // mbarrier

static_assert(H100_SMEM <= 227 * 1024, "DIM=512 tile does not fit in shared memory");

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
  TORCH_CHECK(dim == H100_DIM, "attention_ops_kernels_sm90_d512.cu only implements dim=512, got ", dim);

  const auto* q_ptr = reinterpret_cast<const nv_bfloat16*>(q);
  const auto* k_ptr = reinterpret_cast<const nv_bfloat16*>(k);
  const auto* v_ptr = reinterpret_cast<const nv_bfloat16*>(v);
  const auto* c_ptr = reinterpret_cast<const nv_bfloat16*>(c);
  const auto* o_sal_ptr = reinterpret_cast<const nv_bfloat16*>(o_sal);
  auto* o_ptr = reinterpret_cast<nv_bfloat16*>(o);


  CUtensorMap tmQ = dyllm_make_map(q_ptr, total_seqlen_q, H, H100_DIM, H100_BLOCK_Q);
  CUtensorMap tmK = dyllm_make_map(k_ptr, total_kv, H_kv, H100_DIM, H100_BLOCK_K);
  CUtensorMap tmV = dyllm_make_map(v_ptr, total_kv, H_kv, H100_DIM, H100_BLOCK_K);

  compute_k_masks(B, cu_seqlens_k, cu_salientlens, idx_salient_row_k, num_salient, H100_MASK_BLOCK,
                  max_seqlen_k, row_masks);

  const int num_blocks = B * H * cdiv(max_seqlen_q, H100_BLOCK_Q);
  auto kernel = attention_h100_kernel<H100_BLOCK_Q, H100_BLOCK_K, H100_DIM, H100_STAGES, H100_WS, H100_MMA_WG, H100_OVERLAP>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, H100_SMEM));
  kernel<<<num_blocks, H100_TB, H100_SMEM>>>(tmQ, tmK, tmV, c_ptr, o_ptr, B, H, H_kv, cu_seqlens_q, cu_seqlens_k,
                                             max_seqlen_q, max_seqlen_k, cu_salientlens, row_masks, cosine_stats);
  CUDA_CHECK(cudaGetLastError());

  if (num_salient > 0)
    overwrite_salient_kernel<<<num_salient, 256>>>(c_ptr, o_sal_ptr, o_ptr, idx_salient_row_q, cosine_stats,
                                                   H * H100_DIM);
  compute_cosine_similarity(cosine_stats, cosine_out, total_seqlen_q, cu_seqlens_q, threshold);
  (void)force_mask_v_rows;
}

template void attention_sparse_varlen<at::BFloat16>(const at::BFloat16*, const at::BFloat16*, const at::BFloat16*,
                                                    const at::BFloat16*, const at::BFloat16*, at::BFloat16*, const int,
                                                    const int, const int, const int*, const int*, const int, const int,
                                                    const int, const int, const int*, const int*, const int*,
                                                    uint64_t*, float*, bool*, float, int, int, int);

std::vector<int64_t> attention_kernel_info(int dim, bool mask_v_rows) {
  (void)mask_v_rows;
  cudaFuncAttributes attr{};
  auto kernel = attention_h100_kernel<H100_BLOCK_Q, H100_BLOCK_K, H100_DIM, H100_STAGES, H100_WS, H100_MMA_WG, H100_OVERLAP>;
  if (dim != H100_DIM || cudaFuncGetAttributes(&attr, kernel) != cudaSuccess)
    return {0, 0, 0, 0, 0, 0, H100_BLOCK_Q, H100_BLOCK_K, 1, 0};
  int blocks = 0;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, H100_SMEM);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, H100_TB, H100_SMEM);
  return {static_cast<int64_t>(attr.numRegs),
          static_cast<int64_t>(attr.localSizeBytes),
          static_cast<int64_t>(attr.sharedSizeBytes),
          H100_SMEM,
          H100_TB,
          blocks,
          H100_BLOCK_Q,
          H100_BLOCK_K,
          1,
          1};
}

} // namespace dyllm_h100_d512

void attention_sparse_varlen_h100_d512(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                       const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                       const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                       const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                       const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                                       const int* idx_salient_row_q, uint64_t* row_masks, float* cosine_stats,
                                       bool* cosine_out, float threshold, int dim, int force_mask_v_rows, int total_kv) {
  dyllm_h100_d512::attention_sparse_varlen<at::BFloat16>(
      q, k, v, c, o_sal, o, B, H, H_kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, total_seqlen_q,
      num_salient, cu_salientlens, idx_salient_row_k, idx_salient_row_q, row_masks, cosine_stats, cosine_out, threshold,
      dim, force_mask_v_rows, total_kv);
}
std::vector<int64_t> attention_kernel_info_h100_d512(int dim, bool mask_v_rows) {
  return dyllm_h100_d512::attention_kernel_info(dim, mask_v_rows);
}
