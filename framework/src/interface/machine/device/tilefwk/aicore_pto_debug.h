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
 * \file aicore_pto_debug.h
 * \brief PyPTO AICore wrappers around vendored pto_debug (printf + pointer dump).
 */

#pragma once

#if IS_AICORE && defined(__ENABLE_PTO_PRINTF__)
#ifndef __PTO_INCLUDE_INTERNAL_HEADERS__
#define __PTO_INCLUDE_INTERNAL_HEADERS__
#define __AICORE_PTO_DEBUG_DEFINED_INTERNAL__
#endif
// Skip CANN print_common_head / enable PyPTO vendored adaptations.
#ifndef PYPTO_PTO_DEBUG
#define PYPTO_PTO_DEBUG 1
#endif
#include "impl/utils/debug/pto_aicore_printf_impl.h"
#include "utils/debug/pto_dump.h"
#ifdef __AICORE_PTO_DEBUG_DEFINED_INTERNAL__
#undef __PTO_INCLUDE_INTERNAL_HEADERS__
#undef __AICORE_PTO_DEBUG_DEFINED_INTERNAL__
#endif

// Prefer PYPTO_AICORE_PRINTF_WITH_CORE(ctx->blockIdx, ...): get_subblockdim() may become 0 after
// KernelEntry, so recomputing the mix formula collapses to get_block_num().
#if defined(__AIV__) && defined(__MIX__)
#define PYPTO_AICORE_PRINTF_CORE_ID() (get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num())
#else
#define PYPTO_AICORE_PRINTF_CORE_ID() (get_block_idx())
#endif

#define PYPTO_AICORE_PRINTF_WITH_CORE(core_id, fmt, ...) \
    __pto_aicore::printf("[core=%u][t=%lu] " fmt "\n", static_cast<uint32_t>(core_id), get_sys_cnt(), ##__VA_ARGS__)
#define PYPTO_AICORE_PRINTF(fmt, ...) PYPTO_AICORE_PRINTF_WITH_CORE(PYPTO_AICORE_PRINTF_CORE_ID(), fmt, ##__VA_ARGS__)

// Pointer-based dump (PyPTO machine / codegen has no AscendC LocalTensor / GlobalTensor).
// desc: user tag (e.g. line id); dump_size: element count. Disabled when ASCENDC_DUMP=0.

template <typename T>
INLINE void AicorePtoDumpGm(__gm__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __pto_aicore::pto_dump_gm(input, desc, dumpSize);
}

template <typename T>
INLINE void AicorePtoDumpUbuf(__ubuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __pto_aicore::pto_dump_ubuf(input, desc, dumpSize);
}

template <typename T>
INLINE void AicorePtoDumpL1(__cbuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __pto_aicore::pto_dump_l1buf(input, desc, dumpSize);
}

template <typename T>
INLINE void AicorePtoDumpL0c(__cc__ T* input, uint32_t desc, uint32_t dumpSize)
{
    __pto_aicore::pto_dump_cbuf(input, desc, dumpSize);
}

template <typename T>
INLINE void AicorePtoDump(__gm__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicorePtoDumpGm(input, desc, dumpSize);
}

template <typename T>
INLINE void AicorePtoDump(__ubuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicorePtoDumpUbuf(input, desc, dumpSize);
}

template <typename T>
INLINE void AicorePtoDump(__cbuf__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicorePtoDumpL1(input, desc, dumpSize);
}

template <typename T>
INLINE void AicorePtoDump(__cc__ T* input, uint32_t desc, uint32_t dumpSize)
{
    AicorePtoDumpL0c(input, desc, dumpSize);
}

#define PYPTO_AICORE_DUMP_GM(ptr, desc, n) AicorePtoDumpGm((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP_UBUF(ptr, desc, n) AicorePtoDumpUbuf((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP_L1(ptr, desc, n) AicorePtoDumpL1((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP_L0C(ptr, desc, n) AicorePtoDumpL0c((ptr), (desc), (n))
#define PYPTO_AICORE_DUMP(ptr, desc, n) AicorePtoDump((ptr), (desc), (n))
#else
#define PYPTO_AICORE_PRINTF_WITH_CORE(core_id, fmt, ...) ((void)0)
#define PYPTO_AICORE_PRINTF(fmt, ...) ((void)0)
#define PYPTO_AICORE_DUMP_GM(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP_UBUF(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP_L1(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP_L0C(ptr, desc, n) ((void)0)
#define PYPTO_AICORE_DUMP(ptr, desc, n) ((void)0)
#endif
