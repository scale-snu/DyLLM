#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace dyllm_hopper {

inline PFN_cuTensorMapEncodeTiled_v12000 get_tma_encode_tiled() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult result;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault,
                                                    &result));
    TORCH_CHECK(ptr != nullptr && result == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

// View contiguous [total_seq, heads, dim] memory as a (dim, heads, total_seq)
// TMA tensor. A box is one 128-byte swizzle block wide.
inline CUtensorMap make_tma_tensor_map(const void* base, int total_seq, int heads, int dim, int box_rows) {
  CUtensorMap map{};
  uint64_t global_dims[3] = {static_cast<uint64_t>(dim), static_cast<uint64_t>(heads),
                             static_cast<uint64_t>(total_seq)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(dim) * 2, static_cast<uint64_t>(heads) * dim * 2};
  uint32_t box_dims[3] = {64u, 1u, static_cast<uint32_t>(box_rows)};
  uint32_t element_strides[3] = {1u, 1u, 1u};
  const CUresult result = get_tma_encode_tiled()(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), global_dims, global_strides, box_dims,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
  return map;
}

} // namespace dyllm_hopper
