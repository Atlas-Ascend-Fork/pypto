/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pypto_block_meta.h
 * \brief Block num / block idx accessors for PyPTO kernels (SuperKernel-aware).
 *
 * The CCE backend emits these accessors unconditionally, so the generated kernel source
 * is identical for SK and non-SK builds. The compile flag ``-D__PYPTO_SK_BLOCK__`` (added
 * by the JIT whenever ``sk_cap`` is configured, together with the asc-devkit include
 * paths) selects the implementation:
 *
 * - SK build: per-core ``__gm__`` records. A fused sub-kernel's hardware grid is the
 *   *container's*, not its own, so the raw builtins return container-wide values. The
 *   ``__sk__`` entry seeds the records from ``sysArgs``; the non-fused ``__global__``
 *   entry (which has no sysArgs) seeds the same records from the raw builtins; every
 *   core only ever reads the record it seeded itself, and each seed is flushed with a
 *   single-line DCCI. An earlier ``__BLOCK_LOCAL__`` revision faulted the device
 *   (507035) on early-start kernels (cap bits 1/2) whose entry prologue may overlap the
 *   previous sub-kernel, and a plain ``__gm__`` revision without the DCCI went
 *   numerically wrong: raw scalar GM access carries no implicit cache maintenance.
 * - non-SK build: the accessors are thin wrappers over the raw builtins and inline
 *   away; no records, no DCCI, no asc-devkit include.
 */
#pragma once
#include <cstdint>

#ifdef __PYPTO_SK_BLOCK__

#include "kernel_operator.h"

static __aicore__ inline int64_t __pypto_raw_block_num() { return get_block_num(); }
static __aicore__ inline int64_t __pypto_raw_block_idx()
{
#if defined(__DAV_VEC__)
    return get_block_idx() * get_subblockdim() + get_subblockid();
#else
    return get_block_idx();
#endif
}

// One record per raw block-index slot, one slot per DCCI cache line:
//
// - The second dimension is exactly one 512B line of uint32_t (512 / 4 = 128).
//   A single-line DCCI writes back the whole line, so records must never share a
//   line: otherwise concurrent seeds lose each other's update (last flush wins)
//   and the impl reads a stale count/index. Only words [0] (count) and [1]
//   (index) are used; the rest pads the slot to one line.
// - The first dimension is one slot per raw block index. The index is the
//   hardware block index (AIV: blockIdx * subblockdim + subblockid), so the table
//   needs max vector_core_cnt rows. 72 is the maximum across the platforms this
//   CANN release supports (A5/DAV_3510 959x/95Ax: 36 AIC x 2 subblocks = 72;
//   DAV_2201 max 48; smaller SKUs fewer). Bump PYPTO_SK_BLOCK_SLOTS whenever a
//   supported platform exceeds it.
// - The seed helpers validate the slot through __pypto_checked_block_idx(): an
//   out-of-range index traps instead of silently writing past the table. The
//   read-only accessors rely on the seed having run first, which the JIT
//   guarantees by seeding exactly the accessors the impl uses.
constexpr uint32_t PYPTO_SK_BLOCK_SLOTS = 72;
__gm__ uint32_t g_pypto_block_meta_vec[PYPTO_SK_BLOCK_SLOTS][128];
__gm__ uint32_t g_pypto_block_meta_cube[PYPTO_SK_BLOCK_SLOTS][128];

static __aicore__ inline int64_t __pypto_checked_block_idx()
{
    const int64_t idx = __pypto_raw_block_idx();
    if (idx < 0 || idx >= static_cast<int64_t>(PYPTO_SK_BLOCK_SLOTS)) {
        AscendC::Trap();
    }
    return idx;
}

static __aicore__ inline int64_t __pypto_block_num_value()
{
#if defined(__DAV_VEC__)
    return (int64_t)g_pypto_block_meta_vec[__pypto_raw_block_idx()][0];
#else
    return (int64_t)g_pypto_block_meta_cube[__pypto_raw_block_idx()][0];
#endif
}

static __aicore__ inline int64_t __pypto_block_idx_value()
{
#if defined(__DAV_VEC__)
    return (int64_t)g_pypto_block_meta_vec[__pypto_raw_block_idx()][1];
#else
    return (int64_t)g_pypto_block_meta_cube[__pypto_raw_block_idx()][1];
#endif
}

static __aicore__ inline void __pypto_seed_block_num(uint32_t v)
{
#if defined(__DAV_VEC__)
    const int64_t idx = __pypto_checked_block_idx();
    g_pypto_block_meta_vec[idx][0] = v;
    dcci(reinterpret_cast<__gm__ void*>(&g_pypto_block_meta_vec[idx][0]), SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
    const int64_t idx = __pypto_checked_block_idx();
    g_pypto_block_meta_cube[idx][0] = v;
    dcci(reinterpret_cast<__gm__ void*>(&g_pypto_block_meta_cube[idx][0]), SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
}

static __aicore__ inline void __pypto_seed_block_idx(uint32_t v)
{
#if defined(__DAV_VEC__)
    const int64_t idx = __pypto_checked_block_idx();
    g_pypto_block_meta_vec[idx][1] = v;
    dcci(reinterpret_cast<__gm__ void*>(&g_pypto_block_meta_vec[idx][1]), SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
    const int64_t idx = __pypto_checked_block_idx();
    g_pypto_block_meta_cube[idx][1] = v;
    dcci(reinterpret_cast<__gm__ void*>(&g_pypto_block_meta_cube[idx][1]), SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
}

#else // __PYPTO_SK_BLOCK__

// Non-SK build: the raw builtins are already the launch truth; the wrappers keep the
// codegen emission identical across builds and inline away.
static __aicore__ inline int64_t __pypto_block_num_value() { return get_block_num(); }

static __aicore__ inline int64_t __pypto_block_idx_value()
{
#if defined(__DAV_VEC__)
    return get_block_idx() * get_subblockdim() + get_subblockid();
#else
    return get_block_idx();
#endif
}

#endif // __PYPTO_SK_BLOCK__
