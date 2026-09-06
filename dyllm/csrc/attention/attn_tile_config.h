#pragma once

#if defined(DYLLM_BLOCK_Q) || defined(DYLLM_BLOCK_K) || defined(DYLLM_D_SPLIT) || \
    defined(DYLLM_WARP_Q)
#define DYLLM_TILE_PINNED 1
#else
#define DYLLM_TILE_PINNED 0
#endif

#ifndef DYLLM_BLOCK_K
#define DYLLM_BLOCK_K 64
#endif

#ifndef DYLLM_D_SPLIT
#define DYLLM_D_SPLIT 1
#endif

#ifndef DYLLM_BLOCK_Q
#define DYLLM_BLOCK_Q 64
#endif

// WARP_Q: Number of Q rows owned by one warp. Must be a multiple of MMA_M=16.
#ifndef DYLLM_WARP_Q
#define DYLLM_WARP_Q 16
#endif

// DIM=512 has a separate kernel implementation and therefore separate tuning
// knobs.  Keep its host-side mask geometry sourced from the same knob.
#ifndef DYLLM_D512_BLOCK_K
#define DYLLM_D512_BLOCK_K 32
#endif

template <int DIM>
struct DyllmTileDefault {
  static constexpr int BLOCK_Q = 64;
  static constexpr int BLOCK_K = 64;
  static constexpr int D_SPLIT = 1;
  static constexpr int WARP_Q = 16;
};

template <>
struct DyllmTileDefault<256> {
  // 254 registers / 0 B spills / 32 KB shared memory. No D_SPLIT.
  static constexpr int BLOCK_Q = 64;
  static constexpr int BLOCK_K = 32;
  static constexpr int D_SPLIT = 1;
  static constexpr int WARP_Q = 16;
};

template <>
struct DyllmTileDefault<512> {
  static constexpr int BLOCK_Q = 64;
  static constexpr int BLOCK_K = 32;
  static constexpr int D_SPLIT = 2;
  static constexpr int WARP_Q = 16;
};

template <int DIM>
struct DyllmTile {
#if DYLLM_TILE_PINNED
  static constexpr int BLOCK_Q = DYLLM_BLOCK_Q;
  static constexpr int BLOCK_K = DYLLM_BLOCK_K;
  static constexpr int D_SPLIT = DYLLM_D_SPLIT;
  static constexpr int WARP_Q = DYLLM_WARP_Q;
#else
  static constexpr int BLOCK_Q = DyllmTileDefault<DIM>::BLOCK_Q;
  static constexpr int BLOCK_K = DyllmTileDefault<DIM>::BLOCK_K;
  static constexpr int D_SPLIT = DyllmTileDefault<DIM>::D_SPLIT;
  static constexpr int WARP_Q = DyllmTileDefault<DIM>::WARP_Q;
#endif
};

constexpr bool dyllm_supported_head_dim(int dim) {
  return dim == 64 || dim == 128 || dim == 256 || dim == 512;
}

constexpr int dyllm_sm80_block_k_for_dim(int dim) {
  return dim == 64    ? DyllmTile<64>::BLOCK_K
         : dim == 128 ? DyllmTile<128>::BLOCK_K
         : dim == 256 ? DyllmTile<256>::BLOCK_K
         : dim == 512 ? DYLLM_D512_BLOCK_K
                      : 0;
}

// Hopper kernels use one uint64 mask word per 64 key rows, independently of
// their wider attention tile. Generic kernels use their actual BLOCK_K.
constexpr int dyllm_row_mask_block_for_dim(int dim, bool use_sm90) {
  return use_sm90 ? 64 : dyllm_sm80_block_k_for_dim(dim);
}

#ifndef DYLLM_BUILD_DIM
#define DYLLM_BUILD_DIM 0
#endif

#define DYLLM_DIM_ENABLED(D) (DYLLM_BUILD_DIM == 0 || DYLLM_BUILD_DIM == (D))
