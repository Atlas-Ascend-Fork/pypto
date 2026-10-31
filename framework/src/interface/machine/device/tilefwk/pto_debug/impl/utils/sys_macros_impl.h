/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sys_macros_impl.h
 * \brief PyPTO-vendored slim macros for AICore printf/dump (no g_tilingKey strong symbol).
 */
#ifndef TILEFWK_PTO_DEBUG_IMPL_UTILS_SYS_MACROS_IMPL_H
#define TILEFWK_PTO_DEBUG_IMPL_UTILS_SYS_MACROS_IMPL_H

#include <cstdint>
#include "utils/base/sys_constants.h"

#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
#ifdef __ASCC_PRE__
#ifdef __global__
#undef __global__
#endif
#else

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#ifndef __host__
#define __host__ __attribute__((cce_host))
#endif

#ifndef __host_aicore__
#define __host_aicore__ [ host, aicore ]
#endif

#ifndef __disable_kernel_type_autoinfer__
#define __disable_kernel_type_autoinfer__
#endif

#endif
#endif

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

#if (defined(__DAV_CUBE__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201))
#define SPLIT_CORE_CUBE
#endif

#if (defined(__DAV_VEC__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201))
#define SPLIT_CORE_VEC
#endif

#if defined(ASCENDC_CPU_DEBUG)
extern int32_t g_coreType;
#define PTO_IS_AIV (g_coreType == AscendC::AIV)
#define PTO_IS_AIC (g_coreType == AscendC::AIC)
#define PTO_IS_NOT_AIV (g_coreType != AscendC::AIV)
#define PTO_IS_NOT_AIC (g_coreType != AscendC::AIC)
#else
#if defined(SPLIT_CORE_CUBE)
#define PTO_IS_AIV constexpr(false)
#define PTO_IS_AIC constexpr(true)
#define PTO_IS_NOT_AIV constexpr(true)
#define PTO_IS_NOT_AIC constexpr(false)
#elif defined(SPLIT_CORE_VEC)
#define PTO_IS_AIV constexpr(true)
#define PTO_IS_AIC constexpr(false)
#define PTO_IS_NOT_AIV constexpr(false)
#define PTO_IS_NOT_AIC constexpr(true)
#else
#define PTO_IS_AIV constexpr(false)
#define PTO_IS_AIC constexpr(false)
#define PTO_IS_NOT_AIV constexpr(true)
#define PTO_IS_NOT_AIC constexpr(true)
#endif
#endif

#if defined(__PTO_DISABLE_RESERVED_UBUF__)
#define __PTO_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg) __attribute__((unavailable(msg)))
#else
#define __PTO_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg)
#endif

#define __PTO_RESERVED_UBUF_CAT__(a, b) __PTO_RESERVED_UBUF_CAT_IMPL__(a, b)
#define __PTO_RESERVED_UBUF_CAT_IMPL__(a, b) a##b
#define __PTO_RESERVED_UBUF_PROBE__() ~, 1
#define __PTO_RESERVED_UBUF_CHECK_N__(_0, n, ...) n
#define __PTO_RESERVED_UBUF_CHECK__(...) __PTO_RESERVED_UBUF_CHECK_N__(__VA_ARGS__, 0)
#define __PTO_RESERVED_UBUF_IS_PROBE__(...) __PTO_RESERVED_UBUF_CHECK__(__VA_ARGS__)

#define __PTO_RESERVED_UBUF_ARCH_2201 __PTO_RESERVED_UBUF_PROBE__()
#define __PTO_RESERVED_UBUF_ARCH_3510 __PTO_RESERVED_UBUF_PROBE__()
#define __PTO_RESERVED_UBUF_IS_SUPPORTED_ARCH__(arch) \
    __PTO_RESERVED_UBUF_IS_PROBE__(__PTO_RESERVED_UBUF_CAT__(__PTO_RESERVED_UBUF_ARCH_, arch))

#define __PTO_RESERVED_UBUF_PRAGMA__(x) _Pragma(#x)
#define __PTO_RESERVED_UBUF_UNSUPPORTED_ARCH_ERROR__() \
    __PTO_RESERVED_UBUF_PRAGMA__(                      \
        GCC error                                      \
        "unsupported chip architecture in __PTO_USE_RESERVED_UBUF__; supported architectures are 2201 and 3510")

#if defined(__PTO_DISABLE_RESERVED_UBUF__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#define __PTO_USE_RESERVED_UBUF_2201(msg) __PTO_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg)
#else
#define __PTO_USE_RESERVED_UBUF_2201(msg)
#endif

#if defined(__PTO_DISABLE_RESERVED_UBUF__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define __PTO_USE_RESERVED_UBUF_3510(msg) __PTO_RESERVED_UBUF_UNAVAILABLE_ATTR__(msg)
#else
#define __PTO_USE_RESERVED_UBUF_3510(msg)
#endif

#define __PTO_USE_RESERVED_UBUF_SUPPORTED_0(arch, msg) __PTO_RESERVED_UBUF_UNSUPPORTED_ARCH_ERROR__()
#define __PTO_USE_RESERVED_UBUF_SUPPORTED_1(arch, msg) __PTO_RESERVED_UBUF_CAT__(__PTO_USE_RESERVED_UBUF_, arch)(msg)
#define __PTO_USE_RESERVED_UBUF_CALL__(supported, arch, msg) __PTO_USE_RESERVED_UBUF_CALL_IMPL__(supported, arch, msg)
#define __PTO_USE_RESERVED_UBUF_CALL_IMPL__(supported, arch, msg) \
    __PTO_RESERVED_UBUF_CAT__(__PTO_USE_RESERVED_UBUF_SUPPORTED_, supported)(arch, msg)
#define __PTO_USE_RESERVED_UBUF_IMPL__(arch, msg) \
    __PTO_USE_RESERVED_UBUF_CALL__(__PTO_RESERVED_UBUF_IS_SUPPORTED_ARCH__(arch), arch, msg)

#define __PTO_USE_RESERVED_UBUF_1(arch1, msg) __PTO_USE_RESERVED_UBUF_IMPL__(arch1, msg)
#define __PTO_USE_RESERVED_UBUF_2(arch1, arch2, msg) \
    __PTO_USE_RESERVED_UBUF_IMPL__(arch1, msg)       \
    __PTO_USE_RESERVED_UBUF_IMPL__(arch2, msg)
#define __PTO_USE_RESERVED_UBUF_3(arch1, arch2, arch3, msg) \
    __PTO_USE_RESERVED_UBUF_2(arch1, arch2, msg)            \
    __PTO_USE_RESERVED_UBUF_IMPL__(arch3, msg)
#define __PTO_USE_RESERVED_UBUF_4(arch1, arch2, arch3, arch4, msg) \
    __PTO_USE_RESERVED_UBUF_3(arch1, arch2, arch3, msg)            \
    __PTO_USE_RESERVED_UBUF_IMPL__(arch4, msg)
#define __PTO_USE_RESERVED_UBUF_5(arch1, arch2, arch3, arch4, arch5, msg) \
    __PTO_USE_RESERVED_UBUF_4(arch1, arch2, arch3, arch4, msg)            \
    __PTO_USE_RESERVED_UBUF_IMPL__(arch5, msg)

#define __PTO_USE_RESERVED_UBUF_PICK__(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define __PTO_USE_RESERVED_UBUF__(...)                                                                              \
    __PTO_USE_RESERVED_UBUF_PICK__(__VA_ARGS__, __PTO_USE_RESERVED_UBUF_5, __PTO_USE_RESERVED_UBUF_4,               \
                                   __PTO_USE_RESERVED_UBUF_3, __PTO_USE_RESERVED_UBUF_2, __PTO_USE_RESERVED_UBUF_1) \
    (__VA_ARGS__)

// PyPTO multi-TU link: never emit a strong g_tilingKey. Tiling-key macros evaluate to 0.
#ifndef ASCENDC_TILING_KEY_VAR_IMPL
#ifdef TILING_KEY_VAR
#define ASCENDC_TILING_KEY_VAR_IMPL TILING_KEY_VAR
#else
#define ASCENDC_TILING_KEY_VAR_IMPL 0
#endif
#endif

#endif // TILEFWK_PTO_DEBUG_IMPL_UTILS_SYS_MACROS_IMPL_H
