#pragma once
// EXL3 ROCm port: compatibility shims. ptx.cuh includes this first so every kernel sees it.
//
// Wave64 strategy: the kernels keep their 32-lane "virtual warp" decomposition
// (warp = tid/32, lane = tid%32). Every warp-collective goes through the EXL3_SHFL_* macros
// with an explicit width of 32. HIP's __shfl/__shfl_xor/__shfl_down mask or clamp the source
// lane by `width` (amd_warp_functions.h), so a 64-lane wavefront behaves as two independent
// 32-lane segments exactly like two CUDA warps. On CUDA the macros expand to the original
// *_sync forms with width 32, which is the default, so CUDA codegen is unchanged.
#include <stdint.h>

#if defined(EXL3_ROCM) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP__) || defined(__HIPCC__)
#ifndef EXL3_ROCM
#define EXL3_ROCM 1
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define EXL3_SHFL(v, src)     __shfl((v), (src), 32)
#define EXL3_SHFL_XOR(v, m)   __shfl_xor((v), (m), 32)
#define EXL3_SHFL_DOWN(v, d)  __shfl_down((v), (d), 32)

// ld.global.cs (evict-first streaming load) -> clang nontemporal load
#define EXL3_LDCS(p) __builtin_nontemporal_load(p)

#if defined(__HIPCC__)  // device helpers: only visible to hipcc-compiled units (host-only TUs skip them)
// dp4a: byte-wise dot product with 32-bit accumulate. Only the codebook byte-sum
// (b == 0x01010101) is on any hot path and only for the mul1 codebook (cb 2).
__device__ __forceinline__ uint32_t exl3_dp4a(uint32_t a, uint32_t b, uint32_t c)
{
    return c + (a & 0xffu) * (b & 0xffu)
             + ((a >> 8) & 0xffu) * ((b >> 8) & 0xffu)
             + ((a >> 16) & 0xffu) * ((b >> 16) & 0xffu)
             + (a >> 24) * (b >> 24);
}
#define EXL3_DP4A(a, b, c) exl3_dp4a((uint32_t)(a), (uint32_t)(b), (uint32_t)(c))

// lop3.b32 immLut 0x6a == (a & b) ^ c
#define EXL3_LOP3_6A(x, b, c) ((x) = (((x) & (b)) ^ (c)))

// half2 max/min: HIP has __hmax/__hmin on __half but no matching __hmax2/__hmin2 overloads
__device__ __forceinline__ __half2 exl3_hmax2(const __half2 a, const __half2 b)
{ return __halves2half2(__hmax(__low2half(a), __low2half(b)), __hmax(__high2half(a), __high2half(b))); }
__device__ __forceinline__ __half2 exl3_hmin2(const __half2 a, const __half2 b)
{ return __halves2half2(__hmin(__low2half(a), __low2half(b)), __hmin(__high2half(a), __high2half(b))); }
#define EXL3_HMAX2(a, b) exl3_hmax2((a), (b))
#define EXL3_HMIN2(a, b) exl3_hmin2((a), (b))

#endif  // __HIPCC__

#else  // CUDA

#define EXL3_SHFL(v, src)     __shfl_sync(0xffffffffu, (v), (src), 32)
#define EXL3_SHFL_XOR(v, m)   __shfl_xor_sync(0xffffffffu, (v), (m), 32)
#define EXL3_SHFL_DOWN(v, d)  __shfl_down_sync(0xffffffffu, (v), (d), 32)
#define EXL3_LDCS(p) __ldcs(p)
#define EXL3_DP4A(a, b, c) __dp4a((a), (b), (c))
#define EXL3_LOP3_6A(x, b, c) asm("lop3.b32 %0, %0, " #b ", " #c ", 0x6a;" : "+r"(x))
#define EXL3_HMAX2(a, b) __hmax2((a), (b))
#define EXL3_HMIN2(a, b) __hmin2((a), (b))

#endif
