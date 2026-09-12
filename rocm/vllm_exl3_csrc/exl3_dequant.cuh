#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

namespace vllm_exl3 {

constexpr std::uint32_t MCG_MULTIPLIER = 0xCBAC1FEDu;
constexpr std::uint32_t MUL1_MULTIPLIER = 0x83DCD12Du;

inline std::uint32_t lop3_6a(std::uint32_t a, std::uint32_t b,
                             std::uint32_t c) {
    std::uint32_t out = 0;
    for (unsigned bit = 0; bit < 32; ++bit) {
        // PTX LOP3 indexes the LUT as a:b:c (a is the most-significant bit).
        const unsigned index = (((a >> bit) & 1u) << 2u) |
                               (((b >> bit) & 1u) << 1u) |
                               ((c >> bit) & 1u);
        if ((0x6Au >> index) & 1u) out |= 1u << bit;
    }
    return out;
}

inline float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits >> 15) & 1u;
    const std::uint32_t exp = (bits >> 10) & 0x1Fu;
    const std::uint32_t mant = bits & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float value = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -value : value;
    }
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    const float value = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f,
                                   static_cast<int>(exp) - 15);
    return sign ? -value : value;
}

inline std::uint16_t float_to_half_bits(float value) {
    std::uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    const std::uint32_t sign = (raw >> 16) & 0x8000u;
    const std::uint32_t exp_bits = (raw >> 23) & 0xFFu;
    const std::uint32_t mant = raw & 0x7FFFFFu;
    if (exp_bits == 0xFFu) {
        if (mant == 0) return static_cast<std::uint16_t>(sign | 0x7C00u);
        // Preserve a non-zero NaN payload and force the quiet-NaN bit.
        std::uint32_t payload = mant >> 13;
        payload |= 0x200u;
        return static_cast<std::uint16_t>(sign | 0x7C00u | (payload & 0x3FFu));
    }

    const int exp = static_cast<int>(exp_bits) - 127;
    const std::uint32_t significand = mant | 0x800000u;
    if (exp < -14) {
        // A half subnormal is significand * 2^-24.  Round the float
        // significand to that unit with ties-to-even, including the carry
        // into the smallest normal half (0x0400).
        const int shift = -exp - 1;
        if (shift >= 32) return static_cast<std::uint16_t>(sign);
        std::uint32_t rounded = significand >> shift;
        const std::uint32_t remainder = significand & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | (rounded & 0x7FFu));
    }

    std::uint32_t rounded_mant = mant >> 13;
    const std::uint32_t remainder = mant & 0x1FFFu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded_mant & 1u))) {
        ++rounded_mant;
    }
    int half_exp = exp + 15;
    if (rounded_mant == 0x400u) {
        rounded_mant = 0;
        ++half_exp;
    }
    if (half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(half_exp) << 10) | rounded_mant);
}

inline std::uint16_t half_add_bits(std::uint16_t a, std::uint16_t b) {
    return float_to_half_bits(half_bits_to_float(a) + half_bits_to_float(b));
}

inline std::uint16_t half_mul_bits(std::uint16_t a, std::uint16_t b) {
    return float_to_half_bits(half_bits_to_float(a) * half_bits_to_float(b));
}

inline std::uint16_t half_fma_bits(std::uint16_t a, std::uint16_t b,
                                   std::uint16_t c) {
    return float_to_half_bits(half_bits_to_float(a) * half_bits_to_float(b) +
                              half_bits_to_float(c));
}

// EXL3's three procedural codebooks.  The CUDA implementation returns half at
// every codebook operation; retaining those roundings here is important for
// parity with the native kernel.
inline std::uint16_t decode_codebook_bits(std::uint16_t window, int cb) {
    std::uint32_t x = window;
    if (cb == 0) {
        x = x * 89226354u + 64248484u;
        x = lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
        return half_add_bits(static_cast<std::uint16_t>(x),
                             static_cast<std::uint16_t>(x >> 16));
    }
    if (cb == 1) {
        x *= MCG_MULTIPLIER;
        x = lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
        return half_add_bits(static_cast<std::uint16_t>(x),
                             static_cast<std::uint16_t>(x >> 16));
    }
    // cb == 2 (MUL1): byte dot product followed by half FMA.
    x *= MUL1_MULTIPLIER;
    const std::uint32_t sum = 0x6400u + (x & 0xffu) + ((x >> 8) & 0xffu) +
                               ((x >> 16) & 0xffu) + ((x >> 24) & 0xffu);
    return half_fma_bits(static_cast<std::uint16_t>(sum), 0x1eeeu, 0xc931u);
}

inline float decode_mcg(std::uint16_t window) {
    std::uint32_t x = static_cast<std::uint32_t>(window) * MCG_MULTIPLIER;
    x = lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
    return half_bits_to_float(static_cast<std::uint16_t>(x)) +
           half_bits_to_float(static_cast<std::uint16_t>(x >> 16));
}

inline float decode_mul1(std::uint16_t window) {
    const std::uint32_t x = static_cast<std::uint32_t>(window) * MUL1_MULTIPLIER;
    const std::uint32_t sum = 0x6400u + (x & 0xFFu) + ((x >> 8) & 0xFFu) +
                              ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
    return half_bits_to_float(static_cast<std::uint16_t>(sum)) *
           half_bits_to_float(0x1eeeu) + half_bits_to_float(0xc931u);
}

inline std::uint16_t read_window(const std::uint16_t* packed, std::size_t words,
                                 std::size_t bit) {
    // The packed stream is addressed as uint32 words by EXL3's dq helpers.
    // Reproduce their funnel-shift window exactly instead of treating each
    // uint16 as an independently wrapped scalar word.
    bit %= words * 16;
    const std::size_t b0 = bit;
    const std::size_t b1 = b0 + 16;
    const std::size_t n32 = words / 2;
    const std::size_t i0 = (b0 / 32) % n32;
    const std::size_t i1 = ((b1 - 1) / 32) % n32;
    const unsigned shift = static_cast<unsigned>((i1 + 1) * 32 - b1);
    const std::uint32_t a = static_cast<std::uint32_t>(packed[2 * i0]) |
                            (static_cast<std::uint32_t>(packed[2 * i0 + 1]) << 16);
    const std::uint32_t b = static_cast<std::uint32_t>(packed[2 * i1]) |
                            (static_cast<std::uint32_t>(packed[2 * i1 + 1]) << 16);
    const std::uint64_t merged = (static_cast<std::uint64_t>(a) << 32) | b;
    return static_cast<std::uint16_t>((merged >> shift) & 0xffffu);
}

inline float decode_weight(const std::uint16_t* packed, std::size_t words,
                           int index, int bits, bool mcg) {
    const std::size_t start = (static_cast<std::size_t>(index) + 257u) * bits - 16u;
    const std::uint16_t window = read_window(packed, words, start);
    return half_bits_to_float(decode_codebook_bits(window, mcg ? 1 : 2));
}

inline void hadamard(float* values, int n) {
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int base = 0; base < n; base += stride << 1) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
}

#if defined(__CUDACC__)
// Device equivalents used by the CUDA kernel variants.  Keeping this primitive
// header-only lets future GEMV kernels share exactly the same trellis window
// extraction and MCG codebook as the extension harness.
__device__ __forceinline__ std::uint32_t device_lop3_6a(std::uint32_t a,
                                                        std::uint32_t b,
                                                        std::uint32_t c) {
    std::uint32_t out = 0;
    #pragma unroll
    for (unsigned bit = 0; bit < 32; ++bit) {
        const unsigned index = (((a >> bit) & 1u) << 2u) |
                               (((b >> bit) & 1u) << 1u) |
                               ((c >> bit) & 1u);
        if ((0x6Au >> index) & 1u) out |= 1u << bit;
    }
    return out;
}

__device__ __forceinline__ std::uint16_t device_read_window(
    const std::uint16_t* packed, int words, std::size_t bit) {
    bit %= static_cast<std::size_t>(words * 16);
    const std::size_t b1 = bit + 16;
    const int n32 = words / 2;
    const int i0 = static_cast<int>((bit / 32) % n32);
    const int i1 = static_cast<int>(((b1 - 1) / 32) % n32);
    const unsigned shift = static_cast<unsigned>((i1 + 1) * 32 - b1);
    const std::uint32_t a = static_cast<std::uint32_t>(packed[2 * i0]) |
                            (static_cast<std::uint32_t>(packed[2 * i0 + 1]) << 16);
    const std::uint32_t b = static_cast<std::uint32_t>(packed[2 * i1]) |
                            (static_cast<std::uint32_t>(packed[2 * i1 + 1]) << 16);
    const std::uint64_t merged = (static_cast<std::uint64_t>(a) << 32) | b;
    return static_cast<std::uint16_t>((merged >> shift) & 0xffffu);
}

template <bool MCG>
__device__ __forceinline__ float dequant_weight_device(
    const std::uint16_t* packed, int words, int index, int bits) {
    const std::size_t start = (static_cast<std::size_t>(index) + 257u) * bits - 16u;
    const std::uint32_t window = device_read_window(packed, words, start);
    if constexpr (MCG) {
        std::uint32_t x = window * MCG_MULTIPLIER;
        x = device_lop3_6a(x, 0x8fff8fffu, 0x3b603b60u);
        return __half2float(__ushort_as_half(static_cast<std::uint16_t>(x))) +
               __half2float(__ushort_as_half(static_cast<std::uint16_t>(x >> 16)));
    }
    const std::uint32_t x = window * MUL1_MULTIPLIER;
    const int sum = 0x6400 + static_cast<int>(x & 0xffu) +
                    static_cast<int>((x >> 8) & 0xffu) +
                    static_cast<int>((x >> 16) & 0xffu) +
                    static_cast<int>((x >> 24) & 0xffu);
    return __half2float(__ushort_as_half(static_cast<std::uint16_t>(sum))) *
           __half2float(__ushort_as_half(0x1eeeu)) +
                        __half2float(__ushort_as_half(0xc931u));
}

__device__ __forceinline__ void hadamard_device(float* values, int n) {
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int base = 0; base < n; base += stride << 1) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
}
#endif

}  // namespace vllm_exl3
