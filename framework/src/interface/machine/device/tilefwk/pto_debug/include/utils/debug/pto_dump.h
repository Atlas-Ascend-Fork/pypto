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
 * \file pto_dump.h
 * \brief PyPTO-vendored AICore pointer dump API (GM/UB/L1/L0C). No SIMT/SIMD-VF.
 */
#ifndef INCLUDE_UTILS_DEBUG_PTO_DUMP_H
#define INCLUDE_UTILS_DEBUG_PTO_DUMP_H

#include <cstdint>

#if !defined(__PTO_INCLUDE_INTERNAL_HEADERS__)
#define __PTO_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_PTO_INCLUDE_COMPILER_INTERNAL_HEADERS_PTO_DUMP_H__
#endif

#include "impl/utils/sys_macros_impl.h"

namespace pypto::aicore {
enum class DumpLayout : uint8_t {
    ND,
    DN,
    NZ,
    ZN,
    ZZ,
    NN,
};

struct DumpTensorView {
    const int64_t* sourceShape;
    const int64_t* offsets;
    const int64_t* shape;
    const int64_t* stride;
    uint32_t rank;
    int64_t innerRows;
    int64_t innerCols;
};
} // namespace pypto::aicore

#if (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)
#include "impl/utils/debug/pto_aicore_dump_impl.h"
#endif

namespace pypto::aicore {
template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dump_size);

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dump_size);

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_cbuf(__cc__ T* input, uint32_t desc, uint32_t dump_size);

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_l1buf(__cbuf__ T* input, uint32_t desc, uint32_t dump_size);

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__gm__ T* input, uint32_t desc, uint32_t dump_size);

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__ubuf__ T* input, uint32_t desc, uint32_t dump_size);

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__cc__ T* input, uint32_t desc, uint32_t dump_size);

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__cbuf__ T* input, uint32_t desc, uint32_t dump_size);

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__gm__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view);

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__ubuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view);

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__cc__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view);

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__cbuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view);

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__biasbuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view);

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__fbuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view);
} // namespace pypto::aicore

#if defined(__UNDEF_PTO_INCLUDE_COMPILER_INTERNAL_HEADERS_PTO_DUMP_H__)
#undef __PTO_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_PTO_INCLUDE_COMPILER_INTERNAL_HEADERS_PTO_DUMP_H__
#endif

#endif // INCLUDE_UTILS_DEBUG_PTO_DUMP_H
