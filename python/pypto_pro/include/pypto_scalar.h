/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include <type_traits>

#if defined(__CCE_AICORE__)
#define PYPTO_SCALAR_INLINE __aicore__ inline __attribute__((always_inline))
#else
#define PYPTO_SCALAR_INLINE inline
#endif

namespace pypto_scalar {

// Round to nearest, ties to even. A NaN must never round into an infinity or zero.
PYPTO_SCALAR_INLINE uint16_t FloatBitsToBFloat16(uint32_t bits)
{
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (bits >> 16) | 0x0040u;
    }
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

// Convert integers directly: going through FP32 can round twice near a BF16 midpoint.
template <typename T>
PYPTO_SCALAR_INLINE uint16_t IntegerToBFloat16(T value)
{
    bool negative = false;
    if constexpr (std::is_signed<T>::value) {
        negative = value < 0;
    }
    uint64_t magnitude = negative ? uint64_t(0) - uint64_t(value) : uint64_t(value);
    if (magnitude == 0) {
        return 0;
    }
    const unsigned exponent = 63u - __builtin_clzll(magnitude);
    uint64_t significand;
    if (exponent <= 7) {
        significand = magnitude << (7 - exponent);
    } else {
        const unsigned shift = exponent - 7;
        significand = magnitude >> shift;
        const uint64_t remainder = magnitude & ((uint64_t(1) << shift) - 1);
        const uint64_t halfway = uint64_t(1) << (shift - 1);
        significand += remainder > halfway || (remainder == halfway && (significand & 1));
    }
    // Addition propagates a rounded significand carry into the exponent.
    return (negative ? 0x8000u : 0u) | (((exponent + 126u) << 7) + significand);
}

template <typename T>
PYPTO_SCALAR_INLINE T BFloat16ToInteger(uint16_t bits)
{
    if constexpr (std::is_same<T, bool>::value) {
        return (bits & 0x7fffu) != 0;
    } else {
        const int exponent = int((bits >> 7) & 0xffu) - 127;
        // The caller must supply an in-range finite value. Avoid undefined shifts
        // even for inputs outside that contract.
        if (exponent < 0 || exponent > 63) {
            return 0;
        }
        const uint64_t significand = (bits & 0x7fu) | 0x80u;
        const uint64_t magnitude = exponent >= 7 ? significand << (exponent - 7) : significand >> (7 - exponent);
        return static_cast<T>((bits & 0x8000u) ? uint64_t(0) - magnitude : magnitude);
    }
}

#if defined(__CCE_AICORE__)
template <typename To, typename From>
PYPTO_SCALAR_INLINE To Cast(From value)
{
    if constexpr (std::is_same<To, From>::value) {
        return value;
    } else if constexpr (std::is_same<From, bfloat16_t>::value) {
        if constexpr (std::is_integral<To>::value) {
            return BFloat16ToInteger<To>(reinterpret_cast<uint16_t&>(value));
        } else {
            uint32_t bits = uint32_t(reinterpret_cast<uint16_t&>(value)) << 16;
            return static_cast<To>(reinterpret_cast<float&>(bits));
        }
    } else if constexpr (std::is_same<To, bfloat16_t>::value) {
        uint16_t bits;
        if constexpr (std::is_integral<From>::value) {
            bits = IntegerToBFloat16(value);
        } else {
            float fp32 = static_cast<float>(value);
            bits = FloatBitsToBFloat16(reinterpret_cast<uint32_t&>(fp32));
        }
        return reinterpret_cast<bfloat16_t&>(bits);
    } else {
        return static_cast<To>(value);
    }
}

template <bool Absolute>
PYPTO_SCALAR_INLINE bfloat16_t ChangeSign(bfloat16_t value)
{
    uint16_t bits = reinterpret_cast<uint16_t&>(value);
    bits = Absolute ? (bits & 0x7fffu) : (bits ^ 0x8000u);
    return reinterpret_cast<bfloat16_t&>(bits);
}
#endif

} // namespace pypto_scalar

#undef PYPTO_SCALAR_INLINE
