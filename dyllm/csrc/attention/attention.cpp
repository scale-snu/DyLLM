#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_tile_config.h"

#ifndef DYLLM_NO_H100
void attention_sparse_varlen_sm90(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                  const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                  const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                  const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                                  const int* idx_salient_row_q, uint64_t* row_masks, float* cosine_stats,
                                  bool* cosine_out, float threshold, int dim, int force_mask_v_rows, int total_kv);
void attention_sparse_varlen_sm90_d512(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                       const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                       const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                       const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                       const int num_salient, const int* cu_salientlens, const int* idx_salient_row_k,
                                       const int* idx_salient_row_q, uint64_t* row_masks, float* cosine_stats,
                                       bool* cosine_out, float threshold, int dim, int force_mask_v_rows, int total_kv);
std::vector<int64_t> attention_kernel_info_sm90(int dim, bool mask_v_rows);
std::vector<int64_t> attention_kernel_info_sm90_d512(int dim, bool mask_v_rows);
int attention_sm90_binary_version();

// Without sm_90a, the Hopper implementation is a stub that runs but silently produces incorrect results.
// Compute capability alone cannot detect this, so also check the architecture of the loaded cubin.
static bool dyllm_use_sm90() {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  return props != nullptr && props->major >= 9 && attention_sm90_binary_version() == 90;
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

static void check_cuda_tensor(const at::Tensor& tensor, const char* name, const at::Device& device,
                              at::ScalarType dtype, int dim) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.dim() == dim && tensor.is_contiguous(), name, " must be a contiguous ", dim, "D tensor");
}

#ifdef DYLLM_D512_SECONDARY
void attention_sparse_varlen_sm80_d512(const at::BFloat16* q, const at::BFloat16* k, const at::BFloat16* v,
                                       const at::BFloat16* c, const at::BFloat16* o_sal, at::BFloat16* o, const int B,
                                       const int H, const int H_kv, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                       const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen_q,
                                       const int num_salient, const int* cu_salientlens,
                                       const int* idx_salient_row_k, const int* idx_salient_row_q,
                                       uint64_t* row_masks, float* cosine_stats, bool* cosine_out, float threshold,
                                       int dim, int force_mask_v_rows);
std::vector<int64_t> attention_kernel_info_sm80_d512(int dim, bool mask_v_rows);
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
  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
  const at::cuda::CUDAGuard device_guard(Q.device());
  const auto device = Q.device();
  check_cuda_tensor(Q, "Q", device, at::kBFloat16, 3);
  check_cuda_tensor(K, "K", device, at::kBFloat16, 3);
  check_cuda_tensor(V, "V", device, at::kBFloat16, 3);
  check_cuda_tensor(C, "C", device, at::kBFloat16, 3);
  check_cuda_tensor(o_sal, "o_sal", device, at::kBFloat16, 3);
  check_cuda_tensor(cu_seqlens_q, "cu_seqlens_q", device, at::kInt, 1);
  check_cuda_tensor(cu_seqlens_k, "cu_seqlens_k", device, at::kInt, 1);
  check_cuda_tensor(cu_salientlens, "cu_salientlens", device, at::kInt, 1);
  check_cuda_tensor(idx_salient_row, "idx_salient_row", device, at::kInt, 1);
  check_cuda_tensor(cosine_stats, "cosine_stats", device, at::kFloat, 2);
  check_cuda_tensor(cosine_out, "cosine_out", device, at::kBool, 1);

  TORCH_CHECK(K.sizes() == V.sizes(), "K and V must have the same shape");
  TORCH_CHECK(Q.size(0) > 0 && K.size(0) > 0, "Q and K/V token dimensions must be non-empty");
  TORCH_CHECK(Q.size(2) == K.size(2), "Q, K, and V must have the same head dimension");
  TORCH_CHECK(C.sizes() == Q.sizes(), "C must have the same shape as Q");
  TORCH_CHECK(o_sal.size(0) == idx_salient_row.numel() && o_sal.size(1) == Q.size(1) &&
                  o_sal.size(2) == Q.size(2),
              "o_sal must be [idx_salient_row.numel(), Q.size(1), Q.size(2)]");
  TORCH_CHECK(Q.size(1) > 0 && K.size(1) > 0 && Q.size(1) % K.size(1) == 0,
              "local TP query heads must be a positive multiple of local KV heads; got ", Q.size(1), " and ",
              K.size(1));
  TORCH_CHECK(dyllm_supported_head_dim(Q.size(2)), "supported head dimensions are 64, 128, 256, and 512; got ",
              Q.size(2));
  TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_q.numel() == cu_seqlens_k.numel() &&
                  cu_seqlens_q.numel() == cu_salientlens.numel(),
              "cu_seqlens_q, cu_seqlens_k, and cu_salientlens must have the same B+1 length");
  TORCH_CHECK(total_seqlen == Q.size(0), "total_seqlen must equal Q.size(0)");
  TORCH_CHECK(cosine_stats.size(0) == Q.size(0) && cosine_stats.size(1) == 3,
              "cosine_stats must be [Q.size(0), 3]");
  TORCH_CHECK(cosine_out.numel() == Q.size(0), "cosine_out must contain Q.size(0) elements");
  TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_q <= Q.size(0) && max_seqlen_k > 0 && max_seqlen_k <= K.size(0),
              "maximum sequence lengths must be positive and fit their packed token dimensions");
  TORCH_CHECK(idx_salient_row.numel() <= Q.size(0), "salient Q index count cannot exceed Q.size(0)");
  TORCH_CHECK(force_mask_v_rows >= -1 && force_mask_v_rows <= 1, "force_mask_v_rows must be -1, 0, or 1");
  if (is_q_pruned) {
    check_cuda_tensor(idx_salient_row_k, "idx_salient_row_k", device, at::kInt, 1);
    TORCH_CHECK(idx_salient_row_k.numel() == idx_salient_row.numel(),
                "parallel Q/K pruning requires matching salient Q and K index counts");
    TORCH_CHECK(idx_salient_row_k.numel() <= K.size(0), "salient K index count cannot exceed K.size(0)");
  }

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
  const bool use_sm90 =
#ifndef DYLLM_NO_H100
      dyllm_use_sm90();
#else
      false;
#endif
  const int mask_block = dyllm_row_mask_block_for_dim(dim, use_sm90);
  const int block_kv = (max_seqlen_k + mask_block - 1) / mask_block;

  auto row_masks = at::empty({B * block_kv}, Q.options().dtype(at::kLong));

  uint64_t* row_masks_ptr = reinterpret_cast<uint64_t*>(row_masks.data_ptr<int64_t>());
  const int num_salient = static_cast<int>(idx_salient_row.numel());

  using scalar_t = at::BFloat16;
#define DYLLM_ARGS                                                                                                     \
  Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),                      \
      o_sal.data_ptr<scalar_t>(), Oc.data_ptr<scalar_t>(), B, H, H_kv, cu_q_ptr, cu_k_ptr, max_seqlen_q,               \
      max_seqlen_k, total_seqlen, num_salient, cu_sal_ptr, idx_sal_ptr, idx_sal_q_ptr, row_masks_ptr,                  \
      cosine_stats_ptr, cosine_out_ptr, threshold, dim, force_mask_v_rows

#ifndef DYLLM_NO_H100
  if (use_sm90) {
    const int total_kv = static_cast<int>(K.size(0));
    if (dim == 512)
      attention_sparse_varlen_sm90_d512(DYLLM_ARGS, total_kv);
    else
      attention_sparse_varlen_sm90(DYLLM_ARGS, total_kv);
    return Oc;
  }
#endif
#ifdef DYLLM_D512_SECONDARY
  if (dim == 512) {
    attention_sparse_varlen_sm80_d512(DYLLM_ARGS);
    return Oc;
  }
#endif
  attention_sparse_varlen<scalar_t>(DYLLM_ARGS);
#undef DYLLM_ARGS

  return Oc;
}

void dyllm_launch_fused_offset(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                               torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads);

static std::vector<int64_t> attention_kernel_info_dispatch(int dim, bool mask_v_rows) {
#ifndef DYLLM_NO_H100
  if (dyllm_use_sm90())
    return dim == 512 ? attention_kernel_info_sm90_d512(dim, mask_v_rows)
                      : attention_kernel_info_sm90(dim, mask_v_rows);
#endif
#ifdef DYLLM_D512_SECONDARY
  if (dim == 512)
    return attention_kernel_info_sm80_d512(dim, mask_v_rows);
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
  m.def("fused_offset_launch", &dyllm_launch_fused_offset, "Fused Offset Launch");
  m.def("attention_kernel_info", &attention_kernel_info_dispatch,
        "Launch resources of this build's main kernel: [regs, local_bytes, static_smem, dynamic_smem, tb_size, "
        "max_active_blocks_per_sm, block_q, block_k, d_split, ok]",
        py::arg("dim"), py::arg("mask_v_rows") = false);
}
