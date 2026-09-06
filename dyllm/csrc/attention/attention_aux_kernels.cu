#include "attention_aux.h"
#include "common.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

__global__ void cosine_mask_from_stats_kernel(const float* __restrict__ stats, bool* __restrict__ out,
                                              int total_tokens, float threshold) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_tokens)
    return;

  const float dot = stats[idx * 3];
  const float norm_a = stats[idx * 3 + 1];
  const float norm_b = stats[idx * 3 + 2];
  out[idx] = dot * rsqrtf(fmaxf(norm_a, 1e-8f)) * rsqrtf(fmaxf(norm_b, 1e-8f)) < threshold;
}

__global__ void overwrite_salient_and_accumulate_stats_kernel(
    const nv_bfloat16* __restrict__ c, const nv_bfloat16* __restrict__ o_sal, nv_bfloat16* __restrict__ out,
    const int* __restrict__ idx_salient_row, float* __restrict__ cosine_stats, int vector_size) {
  const int salient_idx = blockIdx.x;
  const int token_idx = idx_salient_row[salient_idx];
  const int pair_count = vector_size / 2;
  const auto* old_row = reinterpret_cast<const nv_bfloat162*>(c + token_idx * vector_size);
  const auto* new_row = reinterpret_cast<const nv_bfloat162*>(o_sal + salient_idx * vector_size);
  auto* out_row = reinterpret_cast<nv_bfloat162*>(out + token_idx * vector_size);

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
      cosine_stats[token_idx * 3] = local.x;
      cosine_stats[token_idx * 3 + 1] = local.y;
      cosine_stats[token_idx * 3 + 2] = local.z;
    }
  }
}

__global__ void compute_k_block_mask_kernel(const int* __restrict__ idx_salient_row,
                                            const int* __restrict__ cu_salientlens,
                                            const int* __restrict__ cu_seqlens_k,
                                            uint64_t* __restrict__ row_masks, int total_blocks, int num_blk_k,
                                            int block_k) {
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

__global__ void fused_offset_kernel(const int* idxs, const int* num_idxs_ptr, const int* cu_promptlens,
                                    const int* cu_salientlens, int num_groups, int64_t* out_tensor) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= *num_idxs_ptr)
    return;

  int left = 0;
  int right = num_groups;
  while (left < right) {
    const int mid = left + (right - left) / 2;
    if (cu_salientlens[mid + 1] <= tid)
      left = mid + 1;
    else
      right = mid;
  }
  out_tensor[tid] = static_cast<int64_t>(idxs[tid]) + cu_promptlens[left + 1];
}

void check_cuda_int32_vector(const torch::Tensor& tensor, const char* name, const torch::Device& device) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
  TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), name, " must be a contiguous 1D tensor");
}

} // namespace

void dyllm_compute_cosine_mask(const float* stats, bool* out, int total_tokens, float threshold,
                               cudaStream_t stream) {
  if (total_tokens == 0)
    return;
  constexpr int threads = 256;
  const int blocks = cdiv(total_tokens, threads);
  cosine_mask_from_stats_kernel<<<blocks, threads, 0, stream>>>(stats, out, total_tokens, threshold);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dyllm_overwrite_salient_and_accumulate_stats(const nv_bfloat16* c, const nv_bfloat16* o_sal,
                                                  nv_bfloat16* out, const int* idx_salient_row,
                                                  float* cosine_stats, int num_salient, int vector_size,
                                                  cudaStream_t stream) {
  if (num_salient == 0)
    return;
  overwrite_salient_and_accumulate_stats_kernel<<<num_salient, 256, 0, stream>>>(
      c, o_sal, out, idx_salient_row, cosine_stats, vector_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dyllm_compute_k_masks(int batch_size, const int* cu_seqlens_k, const int* cu_salientlens_k,
                           const int* idx_salient_row_k, int block_k, int max_seqlen_k, uint64_t* row_masks,
                           cudaStream_t stream) {
  TORCH_CHECK(block_k > 0 && block_k <= 64, "row-mask block size must be in [1, 64], got ", block_k);
  const int num_blk_k = cdiv(max_seqlen_k, block_k);
  const int total_blocks = batch_size * num_blk_k;
  if (total_blocks == 0)
    return;
  constexpr int threads = 256;
  compute_k_block_mask_kernel<<<cdiv(total_blocks, threads), threads, 0, stream>>>(
      idx_salient_row_k, cu_salientlens_k, cu_seqlens_k, row_masks, total_blocks, num_blk_k, block_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dyllm_launch_fused_offset(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                               torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads) {
  const auto device = idxs.device();
  const at::cuda::CUDAGuard device_guard(device);
  check_cuda_int32_vector(idxs, "idxs", device);
  check_cuda_int32_vector(num_idxs, "num_idxs", device);
  check_cuda_int32_vector(cu_promptlens, "cu_promptlens", device);
  check_cuda_int32_vector(cu_salientlens, "cu_salientlens", device);
  TORCH_CHECK(num_idxs.numel() == 1, "num_idxs must contain exactly one element");
  TORCH_CHECK(cu_promptlens.numel() == cu_salientlens.numel(),
              "cu_promptlens and cu_salientlens must have the same length");
  TORCH_CHECK(out_tensor.is_cuda() && out_tensor.device() == device && out_tensor.scalar_type() == at::kLong &&
                  out_tensor.dim() == 1 && out_tensor.is_contiguous(),
              "out_tensor must be a contiguous int64 CUDA vector on ", device);
  // max_threads is the output capacity; the device-side num_idxs value is the
  // actual number of entries consumed from idxs.
  TORCH_CHECK(max_threads >= 0 && max_threads <= out_tensor.numel(),
              "max_threads must fit out_tensor");
  if (max_threads == 0)
    return;

  constexpr int threads = 256;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_offset_kernel<<<cdiv(max_threads, threads), threads, 0, stream>>>(
      idxs.data_ptr<int>(), num_idxs.data_ptr<int>(), cu_promptlens.data_ptr<int>(), cu_salientlens.data_ptr<int>(),
      cu_promptlens.size(0) - 1, out_tensor.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
