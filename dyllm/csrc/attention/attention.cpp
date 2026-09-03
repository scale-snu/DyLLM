#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attn_tile_config.h"

#ifndef DYLLM_NO_H100
void attention_sparse_varlen_h100(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                  const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                  const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                  const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                                  const int* idx_salient_row_q, uint64_t* row_masks, float* cosine_stats,
                                  bool* cosine_out, float threshold, int dim, int force_mask_v_rows, int total_kv);
void attention_sparse_varlen_h100_d512(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                       const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                       const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                       const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                       const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                                       const int* idx_salient_row_q, uint64_t* row_masks, float* cosine_stats,
                                       bool* cosine_out, float threshold, int dim, int force_mask_v_rows, int total_kv);
std::vector<int64_t> attention_kernel_info_h100(int dim, bool mask_v_rows);
std::vector<int64_t> attention_kernel_info_h100_d512(int dim, bool mask_v_rows);
int attention_sm90_binary_version();

// Without sm_90a, the Hopper implementation is a stub that runs but silently produces incorrect results.
// Compute capability alone cannot detect this, so also check the architecture of the loaded cubin.
static bool dyllm_use_sm90() {
  static const bool ok = [] {
    const auto* props = at::cuda::getCurrentDeviceProperties();
    if (props == nullptr || props->major < 9)
      return false;
    return attention_sm90_binary_version() == 90;
  }();
  return ok;
}
#endif // DYLLM_NO_H100

template <typename scalar_t>
void attention_sparse_varlen(const scalar_t* q, const scalar_t* k, const scalar_t* v, const scalar_t* c,
                             const scalar_t* o_sal, scalar_t* o, const int B, const int H, const int H_kv,
                             const int* cu_seqlens_q, const int* cu_seqlens_k, const int max_seqlen_q,
                             const int max_seqlen_k, const int total_seqlen_q, const int num_salient,
                             const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
                             uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold, int dim,
                             int force_mask_v_rows);
std::vector<int64_t> attention_kernel_info(int dim, bool mask_v_rows);

#ifdef DYLLM_D512_SECONDARY
void attention_sparse_varlen_d512(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                  const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                  const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                  const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                                  const int* idx_salient_row_q, uint64_t* row_masks, float* cosine_stats,
                                  bool* cosine_out, float threshold, int dim, int force_mask_v_rows);
std::vector<int64_t> attention_kernel_info_d512(int dim, bool mask_v_rows);
#endif

at::Tensor attention_sparse_varlen_cuda(const at::Tensor& Q, // [T_q, H, D], bf16, CUDA
                                        const at::Tensor& K, // [T_k, H, D], bf16, CUDA
                                        const at::Tensor& V, // [T_k, H, D], bf16, CUDA
                                        const at::Tensor& C, const at::Tensor& o_sal,
                                        const at::Tensor& cu_seqlens_q, // [B+1], CUDA
                                        const at::Tensor& cu_seqlens_k, // [B+1], CUDA
                                        const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen,
                                        const at::Tensor& cu_salientlens, const at::Tensor& idx_salient_row,
                                        const at::Tensor& cosine_stats,
                                        const at::Tensor& cosine_out, // [total_seqlen], bool mask
                                        float threshold = 0.0f, bool is_q_pruned = false,
                                        const at::Tensor& idx_salient_row_k = at::Tensor(),
                                        // Benchmark only: -1=auto, 0=force off, 1=force on
                                        int force_mask_v_rows = -1) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA");
  TORCH_CHECK(Q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16 && V.scalar_type() == at::kBFloat16,
              "Q, K, V must be bfloat16");
  TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be [T, H, D]");

  auto Oc = at::empty_like(Q);

  const int* cu_q_ptr = cu_seqlens_q.data_ptr<int>();
  const int* cu_k_ptr = cu_seqlens_k.data_ptr<int>();
  const int* cu_sal_ptr = cu_salientlens.data_ptr<int>();
  const int* idx_sal_ptr = (is_q_pruned) ? idx_salient_row_k.data_ptr<int>() : idx_salient_row.data_ptr<int>();
  const int* idx_sal_q_ptr = idx_salient_row.data_ptr<int>();

  auto cosine_stats_ptr = cosine_stats.data_ptr<float>();
  auto cosine_out_ptr = cosine_out.data_ptr<bool>();

  const int B = static_cast<int>(cu_seqlens_q.size(0) - 1);
  const int H = static_cast<int>(Q.size(1));
  const int H_kv = static_cast<int>(K.size(1));
  const int dim = static_cast<int>(Q.size(2));
  // row_masks stores one bit per key row for each BLOCK_K-row block, so it must
  // be sized with the same BLOCK_K that the kernel actually uses for tiling.
  const int BLOCK_K = dyllm_block_k_for_dim(dim);
  const int block_kv = (max_seqlen_k + BLOCK_K - 1) / BLOCK_K;

  auto opts_gpu = at::TensorOptions().device(at::kCUDA);
  auto row_masks = at::empty({B * block_kv}, opts_gpu.dtype(at::kLong));

  uint64_t* row_masks_ptr = reinterpret_cast<uint64_t*>(row_masks.data_ptr<int64_t>());
  const int num_salient = static_cast<int>(idx_salient_row.numel());

  using scalar_t = at::BFloat16;
#define DYLLM_ARGS                                                                                                     \
  Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),                      \
      o_sal.data_ptr<scalar_t>(), Oc.data_ptr<scalar_t>(), B, H, H_kv, cu_q_ptr, cu_k_ptr, max_seqlen_q,               \
      max_seqlen_k, total_seqlen, num_salient, cu_sal_ptr, idx_sal_ptr, idx_sal_q_ptr, row_masks_ptr,                  \
      cosine_stats_ptr, cosine_out_ptr, threshold, dim, force_mask_v_rows

#ifndef DYLLM_NO_H100
  if (dyllm_use_sm90()) {
    TORCH_CHECK(BLOCK_K <= 64, "row_masks was sized at a ", BLOCK_K,
                "-key granularity but the H100 kernels index it at 64");
    const int total_kv = static_cast<int>(K.size(0));
    if (dim == 512)
      attention_sparse_varlen_h100_d512(DYLLM_ARGS, total_kv);
    else
      attention_sparse_varlen_h100(DYLLM_ARGS, total_kv);
    return Oc;
  }
#endif
#ifdef DYLLM_D512_SECONDARY
  if (dim == 512) {
    attention_sparse_varlen_d512(DYLLM_ARGS);
    return Oc;
  }
#endif
  attention_sparse_varlen<scalar_t>(DYLLM_ARGS);
#undef DYLLM_ARGS

  return Oc;
}

void fused_offset_launch(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                         torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads);

static std::vector<int64_t> attention_kernel_info_dispatch(int dim, bool mask_v_rows) {
#ifndef DYLLM_NO_H100
  if (dyllm_use_sm90())
    return dim == 512 ? attention_kernel_info_h100_d512(dim, mask_v_rows)
                      : attention_kernel_info_h100(dim, mask_v_rows);
#endif
#ifdef DYLLM_D512_SECONDARY
  if (dim == 512)
    return attention_kernel_info_d512(dim, mask_v_rows);
#endif
  return attention_kernel_info(dim, mask_v_rows);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_sparse_varlen", &attention_sparse_varlen_cuda, "Sparse Variable Length Attention (CUDA)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("C"), py::arg("o_sal"), py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"), py::arg("max_seqlen_q"), py::arg("max_seqlen_k"), py::arg("total_seqlen"),
        py::arg("cu_salientlens"), py::arg("idx_salient_row"), py::arg("cosine_stats"), py::arg("cosine_out"),
        py::arg("threshold") = 0.0f, py::arg("is_q_pruned") = false, py::arg("idx_salient_row_k") = at::Tensor(),
        py::arg("force_mask_v_rows") = -1);
  m.def("fused_offset_launch", &fused_offset_launch, "Fused Offset Launch");
  m.def("attention_kernel_info", &attention_kernel_info_dispatch,
        "Launch resources of this build's main kernel: [regs, local_bytes, static_smem, dynamic_smem, tb_size, "
        "max_active_blocks_per_sm, block_q, block_k, d_split, ok]",
        py::arg("dim"), py::arg("mask_v_rows") = false);
}
