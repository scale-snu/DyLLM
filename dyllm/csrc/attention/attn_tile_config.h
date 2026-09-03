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
  static constexpr int BLOCK_K = 16;
  static constexpr int D_SPLIT = 1;
  static constexpr int WARP_Q = 16;
};

template <>
struct DyllmTileDefault<512> {
  static constexpr int BLOCK_Q = 32;
  static constexpr int BLOCK_K = 16;
  static constexpr int D_SPLIT = 4;
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

constexpr int dyllm_block_k_for_dim(int dim) {
  return dim == 64    ? DyllmTile<64>::BLOCK_K
         : dim == 128 ? DyllmTile<128>::BLOCK_K
         : dim == 256 ? DyllmTile<256>::BLOCK_K
         : dim == 512 ? DyllmTile<512>::BLOCK_K
                      : 16;
}

#ifndef DYLLM_BUILD_DIM
#define DYLLM_BUILD_DIM 0
#endif

#define DYLLM_DIM_ENABLED(D) (DYLLM_BUILD_DIM == 0 || DYLLM_BUILD_DIM == (D))
