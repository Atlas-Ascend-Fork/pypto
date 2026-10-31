/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_ASSERT_H
#define PYPTO_ASSERT_H

// Device-side validation: pypto_panic reports a message and traps; pypto_assert compiles
// only for sanitizer-enabled kernels (pl.jit(sanitizer=True)).

#if defined(__ENABLE_PTO_PRINTF__)
#ifndef __PTO_INCLUDE_INTERNAL_HEADERS__
#define __PTO_INCLUDE_INTERNAL_HEADERS__
#define __PYPTO_ASSERT_DEFINED_INTERNAL_HEADERS__
#endif
#ifndef PYPTO_PTO_DEBUG
#define PYPTO_PTO_DEBUG 1
#define __PYPTO_ASSERT_DEFINED_PTO_DEBUG__
#endif
#include "impl/utils/debug/pto_aicore_printf_impl.h"
#ifdef __PYPTO_ASSERT_DEFINED_PTO_DEBUG__
#undef PYPTO_PTO_DEBUG
#undef __PYPTO_ASSERT_DEFINED_PTO_DEBUG__
#endif
#ifdef __PYPTO_ASSERT_DEFINED_INTERNAL_HEADERS__
#undef __PTO_INCLUDE_INTERNAL_HEADERS__
#undef __PYPTO_ASSERT_DEFINED_INTERNAL_HEADERS__
#endif

// Report the failure (printf-enabled builds) and trap.
template <class... Args>
__aicore__ inline __attribute__((noinline)) void pypto_panic(const __gm__ char* fmt, Args&&... args)
{
    pypto::aicore::printf(fmt, static_cast<Args&&>(args)...);
    trap();
}
#else
// Debug printing is compiled out; keep the trap.
template <class... Args>
__aicore__ inline void pypto_panic(const __gm__ char*, Args&&...)
{
    trap();
}
#endif

// Runtime validation: release kernels do not evaluate the condition or the message arguments.
#if defined(__PYPTO_SANITIZER__)
#define pypto_assert(cond, ...)       \
    do {                              \
        if (!(cond)) {                \
            pypto_panic(__VA_ARGS__); \
        }                             \
    } while (0)
#else
#define pypto_assert(cond, ...) ((void)sizeof(cond))
#endif

#endif // PYPTO_ASSERT_H
