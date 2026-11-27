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
 * ACLNN recompilation defines ``__PYPTO_ACLNN__`` and uses AscendC's native grid
 * accessors. The OPC pipeline specializes the count and provides the SK index.
 * Under the same flag this header also carries the ACLNN static-build Tiling size
 * check and the temporary PTO block-intrinsic adaptation; the compatibility was
 * previously provided by pypto_aclnn_compat.h.
 *
 * The CCE backend emits these accessors unconditionally, so the generated kernel source
 * is identical for SK and non-SK builds. The compile flag ``-D__PYPTO_SK_BLOCK__`` (added
 * by the JIT whenever ``sk_cap`` is configured, together with the asc-devkit include
 * paths) selects the implementation:
 *
 * - SK build: per-core ``__BLOCK_LOCAL__`` scalars. A fused sub-kernel's hardware grid
 *   is the *container's*, not its own, so the raw builtins return container-wide
 *   values. The ``__sk__`` entry seeds the block-local count/index from ``sysArgs``;
 *   the non-fused ``__global__`` entry (which has no sysArgs) seeds the same scalars
 *   from the raw builtins. Block-local storage is private to one core and a core runs
 *   one block at a time, so concurrent launches of one symbol on different streams can
 *   no longer overwrite each other's values: no GM table and no DCCI.
 * - non-SK build: the accessors are thin wrappers over the raw builtins and inline
 *   away; no block-local state, no DCCI, no asc-devkit include.
 */
#pragma once
#include <cstdint>

#if defined(__PYPTO_ACLNN__)

#include "kernel_operator.h"

// CANN 9.3's static GET_TILING_DATA_WITH_STRUCT materializes this byte array.
// convert_from_bytes alone does not reject a Host/Device layout-size mismatch.
// Check before accepting a binary compiled against another Host Tiling branch.
#define PYPTO_ACLNN_CHECK_TILING_SIZE(type, name) \
    static_assert(sizeof(type) == sizeof(__ascendc_arr_##name), "PyPTO Host/Device Tiling sizes differ")

// PTO headers contain unused sync/communication helpers that read the raw launch
// grid. CANN checks all preprocessed functions before dead-code removal. Adapt those
// reads while including PTO, then restore the caller's macro state.
// GetBlockIdx is linear on AIV; dividing by the sub-block count preserves the raw
// group-index convention expected inside those PTO helpers.
#pragma push_macro("get_block_num")
#pragma push_macro("get_block_idx")
#pragma push_macro("get_task_ration")
#undef get_block_num
#undef get_block_idx
#undef get_task_ration
#define get_block_num() AscendC::GetBlockNum()
#define get_block_idx() (AscendC::GetBlockIdx() / AscendC::GetSubBlockNum())
#define get_task_ration() AscendC::GetTaskRatio()
#include <pto/pto-inst.hpp>
#pragma pop_macro("get_task_ration")
#pragma pop_macro("get_block_idx")
#pragma pop_macro("get_block_num")

// The AscendC OPC pipeline specializes the sub-kernel block count and compiles the
// SK entry with its native block-index convention. No process-global GM state is needed.
static __aicore__ inline int64_t __pypto_block_num_value() { return AscendC::GetBlockNum(); }
static __aicore__ inline int64_t __pypto_block_idx_value() { return AscendC::GetBlockIdx(); }
static __aicore__ inline int64_t __pypto_subblock_num_value() { return AscendC::GetSubBlockNum(); }
static __aicore__ inline int64_t __pypto_subblock_idx_value() { return AscendC::GetSubBlockIdx(); }

#elif defined(__PYPTO_SK_BLOCK__)

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

// Per-core block-local scalars: one core executes one block at a time, so seeding in
// the entry and reading in the impl stays launch-private without any GM table or DCCI.
#if defined(__DAV_VEC__)
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_num_v;
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_idx_v;
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_subblock_num_v;
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_subblock_idx_v;
#define PYPTO_BLOCK_NUM g_pypto_block_num_v
#define PYPTO_BLOCK_IDX g_pypto_block_idx_v
#define PYPTO_SUBBLOCK_NUM g_pypto_subblock_num_v
#define PYPTO_SUBBLOCK_IDX g_pypto_subblock_idx_v
#else
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_num_c;
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_idx_c;
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_subblock_num_c;
__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_subblock_idx_c;
#define PYPTO_BLOCK_NUM g_pypto_block_num_c
#define PYPTO_BLOCK_IDX g_pypto_block_idx_c
#define PYPTO_SUBBLOCK_NUM g_pypto_subblock_num_c
#define PYPTO_SUBBLOCK_IDX g_pypto_subblock_idx_c
#endif

static __aicore__ inline int64_t __pypto_block_num_value() { return (int64_t)PYPTO_BLOCK_NUM; }
static __aicore__ inline int64_t __pypto_block_idx_value() { return (int64_t)PYPTO_BLOCK_IDX; }
static __aicore__ inline int64_t __pypto_subblock_num_value() { return (int64_t)PYPTO_SUBBLOCK_NUM; }
static __aicore__ inline int64_t __pypto_subblock_idx_value() { return (int64_t)PYPTO_SUBBLOCK_IDX; }
static __aicore__ inline void __pypto_seed_block_num(uint32_t v) { PYPTO_BLOCK_NUM = v; }
static __aicore__ inline void __pypto_seed_block_idx(uint32_t v) { PYPTO_BLOCK_IDX = v; }
static __aicore__ inline void __pypto_seed_subblock_num(uint32_t v) { PYPTO_SUBBLOCK_NUM = v; }
static __aicore__ inline void __pypto_seed_subblock_idx(uint32_t v) { PYPTO_SUBBLOCK_IDX = v; }

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

static __aicore__ inline int64_t __pypto_subblock_num_value()
{
#if defined(__DAV_VEC__)
    return get_subblockdim();
#else
    return 1;
#endif
}

static __aicore__ inline int64_t __pypto_subblock_idx_value()
{
#if defined(__DAV_VEC__)
    return get_subblockid();
#else
    return 0;
#endif
}

#endif // __PYPTO_SK_BLOCK__
