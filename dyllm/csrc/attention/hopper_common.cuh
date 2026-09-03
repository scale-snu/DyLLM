#pragma once
// Hopper (SM90a) primitives: TMA loads, mbarrier, and WGMMA.
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#define DYLLM_SM90A (defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_SPECIFIC__))

namespace dyllm_hopper {

static constexpr int SWZ_COLS = 64;                    // Number of bf16 elements per swizzle block
static constexpr int SWZ_ROW_BYTES = SWZ_COLS * 2;     // 128

#if DYLLM_SM90A

// ---------------------------------------------------------------- descriptor
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
  uint64_t d = 0;
  d |= (static_cast<uint64_t>(addr >> 4) & 0x3FFFull);
  d |= (static_cast<uint64_t>(lbo >> 4) & 0x3FFFull) << 16;
  d |= (static_cast<uint64_t>(sbo >> 4) & 0x3FFFull) << 32;
  d |= (static_cast<uint64_t>(swizzle) & 0x3ull) << 62;
  return d;
}

__device__ __forceinline__ uint64_t desc_swz128(uint32_t addr) {
  return make_smem_desc(addr, 0, 8 * SWZ_ROW_BYTES, 1);
}

__device__ __forceinline__ uint64_t desc_kmajor(uint32_t addr) { return desc_swz128(addr); }

__device__ __forceinline__ uint64_t desc_mnmajor(uint32_t addr) { return desc_swz128(addr); }

__device__ __forceinline__ uint64_t desc_mnmajor_n128(uint32_t addr, uint32_t colblock_stride) {
  return make_smem_desc(addr, colblock_stride, 8 * SWZ_ROW_BYTES, 1);
}

__device__ __forceinline__ uint64_t desc_add(uint64_t desc, uint32_t byte_off) {
  return desc + (static_cast<uint64_t>(byte_off) >> 4);
}

static constexpr int K16_STEP_BYTES = 32;
static constexpr int K16_STEP_MNMAJOR_BYTES = 16 * SWZ_ROW_BYTES;

// ------------------------------------------------------------------ mbarrier
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar));
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
  uint32_t done = 0;
  while (!done) {
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.b32 %0, 1, 0, P;}"
                 : "=r"(done)
                 : "r"(bar), "r"(phase));
  }
}

// ----------------------------------------------------------------------- TMA
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* tm, uint32_t bar, int c0, int c1,
                                            int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3, %4}], [%5];"
               :
               : "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void tma_prefetch(const CUtensorMap* tm) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tm) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  nv_bfloat162 v = __float22bfloat162_rn(make_float2(lo, hi));
  uint32_t u;
  __builtin_memcpy(&u, &v, sizeof(u));
  return u;
}

// ------------------------------------------------- warp specialization
template <int N> __device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}
template <int N> __device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}
template <int ID, int COUNT> __device__ __forceinline__ void named_barrier_sync() {
  asm volatile("bar.sync %0, %1;" ::"n"(ID), "n"(COUNT));
}
template <int ID, int COUNT> __device__ __forceinline__ void named_barrier_arrive() {
  asm volatile("bar.arrive %0, %1;" ::"n"(ID), "n"(COUNT));
}

// ---------------------------------------------------------------------- wgmma
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

#define DYLLM_D32_SLOTS                                                                                                \
  "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                                                             \
  "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31"
#define DYLLM_D32_OUT(d)                                                                                               \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),          \
      "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),           \
      "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),          \
      "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

#define DYLLM_D8_SLOTS "%0,%1,%2,%3,%4,%5,%6,%7"
#define DYLLM_D8_OUT(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])

#define DYLLM_D16_SLOTS "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15"
#define DYLLM_D16_OUT(d)                                                                                               \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),          \
      "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])

template <int SCALE_D>
__device__ __forceinline__ void wgmma_m64n16k16_ss(float d[8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {" DYLLM_D8_SLOTS "}, %8, %9, %10, 1, 1, 0, 0;"
               : DYLLM_D8_OUT(d)
               : "l"(da), "l"(db), "n"(SCALE_D));
}

template <int SCALE_D>
__device__ __forceinline__ void wgmma_m64n32k16_ss(float d[16], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {" DYLLM_D16_SLOTS "}, %16, %17, %18, 1, 1, 0, 0;"
               : DYLLM_D16_OUT(d)
               : "l"(da), "l"(db), "n"(SCALE_D));
}

template <int SCALE_D>
__device__ __forceinline__ void wgmma_m64n64k16_ss(float d[32], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {" DYLLM_D32_SLOTS "}, %32, %33, %34, 1, 1, 0, 0;"
               : DYLLM_D32_OUT(d)
               : "l"(da), "l"(db), "n"(SCALE_D));
}
#define DYLLM_D64_SLOTS                                                                                                \
  "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"     \
  "%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,"   \
  "%58,%59,%60,%61,%62,%63"
#define DYLLM_D64_OUT(d)                                                                                               \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),          \
      "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),           \
      "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),          \
      "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),          \
      "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),          \
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),          \
      "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),          \
      "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])


template <int SCALE_D>
__device__ __forceinline__ void wgmma_m64n128k16_rs(float d[64], const uint32_t a[4], uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {" DYLLM_D64_SLOTS "}, {%64,%65,%66,%67}, %68, "
               "%69, 1, 1, 1;"
               : DYLLM_D64_OUT(d)
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(SCALE_D));
}

template <int SCALE_D, int TRANS_B>
__device__ __forceinline__ void wgmma_m64n128k16_ss(float d[64], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {" DYLLM_D64_SLOTS "}, %64, %65, "
               "%66, 1, 1, 0, %67;"
               : DYLLM_D64_OUT(d)
               : "l"(da), "l"(db), "n"(SCALE_D), "n"(TRANS_B));
}

template <int SCALE_D, int TRANS_B>
__device__ __forceinline__ void wgmma_m64n64k16_rs(float d[32], const uint32_t a[4], uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {" DYLLM_D32_SLOTS "}, {%32,%33,%34,%35}, %36, "
               "%37, 1, 1, %38;"
               : DYLLM_D32_OUT(d)
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(SCALE_D), "n"(TRANS_B));
}

#endif // DYLLM_SM90A

} // namespace dyllm_hopper
