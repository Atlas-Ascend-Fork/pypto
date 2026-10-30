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
 * \file aikernel_define.h
 * \brief
 */

#ifndef AIKERNEL_DEFINE_H
#define AIKERNEL_DEFINE_H

#ifndef __gm__

// Host or aicpu
#define IS_AICORE 0
#define __gm__
#ifndef __aicore__
#define __aicore__
#endif
#define INLINE inline
#define __TILE_FWK_HOST__
#define BLOCK_LOCAL

#else

// aicore
#define IS_AICORE 1
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#define INLINE __attribute__((always_inline)) inline __aicore__
#define BLOCK_LOCAL [[block_local]]

#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#ifndef UNUSED
#define UNUSED_1(a) (void)(a);
#define UNUSED_2(a, ...) \
    (void)(a);           \
    UNUSED_1(__VA_ARGS__)
#define UNUSED_3(a, ...) \
    (void)(a);           \
    UNUSED_2(__VA_ARGS__)
#define UNUSED_4(a, ...) \
    (void)(a);           \
    UNUSED_3(__VA_ARGS__)
#define UNUSED_5(a, ...) \
    (void)(a);           \
    UNUSED_4(__VA_ARGS__)
#define UNUSED_6(a, ...) \
    (void)(a);           \
    UNUSED_5(__VA_ARGS__)
#define UNUSED_7(a, ...) \
    (void)(a);           \
    UNUSED_6(__VA_ARGS__)
#define UNUSED_8(a, ...) \
    (void)(a);           \
    UNUSED_7(__VA_ARGS__)
#define UNUSED_9(a, ...) \
    (void)(a);           \
    UNUSED_8(__VA_ARGS__)
#define UNUSED_10(a, ...) \
    (void)(a);            \
    UNUSED_9(__VA_ARGS__)
#define UNUSED_11(a, ...) \
    (void)(a);            \
    UNUSED_10(__VA_ARGS__)
#define UNUSED_12(a, ...) \
    (void)(a);            \
    UNUSED_11(__VA_ARGS__)
#define UNUSED_13(a, ...) \
    (void)(a);            \
    UNUSED_12(__VA_ARGS__)
#define UNUSED_14(a, ...) \
    (void)(a);            \
    UNUSED_13(__VA_ARGS__)
#define UNUSED_15(a, ...) \
    (void)(a);            \
    UNUSED_14(__VA_ARGS__)
#define UNUSED_16(a, ...) \
    (void)(a);            \
    UNUSED_15(__VA_ARGS__)

#define UNUSED_SELECT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, NAME, ...) NAME
#define UNUSED(...)                                                                                                   \
    UNUSED_SELECT(__VA_ARGS__, UNUSED_16, UNUSED_15, UNUSED_14, UNUSED_13, UNUSED_12, UNUSED_11, UNUSED_10, UNUSED_9, \
                  UNUSED_8, UNUSED_7, UNUSED_6, UNUSED_5, UNUSED_4, UNUSED_3, UNUSED_2, UNUSED_1)                     \
    (__VA_ARGS__)
#endif

#ifndef ENABLE_AICORE_RESOLVE
#define ENABLE_AICORE_RESOLVE 0
#endif

#ifndef ENABLE_AICORE_BN_PARALLEL
#define ENABLE_AICORE_BN_PARALLEL 0
#endif

#endif
