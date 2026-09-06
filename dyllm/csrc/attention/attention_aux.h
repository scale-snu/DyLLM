#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void dyllm_compute_cosine_mask(const float* stats, bool* out, int total_tokens, float threshold,
                               cudaStream_t stream);

void dyllm_overwrite_salient_and_accumulate_stats(const nv_bfloat16* c, const nv_bfloat16* o_sal,
                                                  nv_bfloat16* out, const int* idx_salient_row,
                                                  float* cosine_stats, int num_salient, int vector_size,
                                                  cudaStream_t stream);

void dyllm_compute_k_masks(int batch_size, const int* cu_seqlens_k, const int* cu_salientlens_k,
                           const int* idx_salient_row_k, int block_k, int max_seqlen_k, uint64_t* row_masks,
                           cudaStream_t stream);

void dyllm_launch_fused_offset(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                               torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads);
