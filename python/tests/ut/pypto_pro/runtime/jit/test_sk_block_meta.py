#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SK block adaptation: unconditional Tile block-local seeding + adapted SIMT VF.

The Tile block num/idx accessors are emitted unconditionally by the CCE backend and
implemented in ``pypto_pro/include/pypto_block_meta.h`` (SK per-core block-local
scalars vs raw-builtin wrappers selected by ``-D__PYPTO_SK_BLOCK__``). SK entries
always seed both scalars. SIMT VF kernels may carry ``sk_cap`` when the VF receives
the adapted block count/index from the outer SIMD kernel instead of reading
``gridDim``/``blockIdx``.
"""
import pathlib

import pypto_pro
import pypto_pro.language as pl
from pypto_pro.runtime.jit import KERNEL_HEADER_SINGLE, _codegen
from pypto_pro.runtime.platform import NpuArch


@pl.vector_function(mode="simt", max_threads=128)
def _simt_copy(x, y, grid_x, core_id):
    tid = pl.simt.linear_thread_idx()
    for i in pl.range(core_id * 128 + tid, 1024, grid_x * 128):
        y[0, i] = x[0, i]


@pl.jit(sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
def _simt_sk_kernel(x: pl.Tensor[[1, 1024], pl.DT_FP32], y: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        grid_x = pl.get_block_num() * pl.get_subblock_num()
        core_id = pl.get_block_idx()
        _simt_copy[128](x, y, grid_x, core_id)


@pl.jit(auto_mutex=False)
def _simt_no_sk_kernel(x: pl.Tensor[[1, 1024], pl.DT_FP32], y: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        grid_x = pl.get_block_num() * pl.get_subblock_num()
        core_id = pl.get_block_idx()
        _simt_copy[128](x, y, grid_x, core_id)


@pl.jit(sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
def _simd_sk_kernel(out: pl.Tensor[[64], pl.DT_INT64]):
    out[0] = pl.get_block_num()


def test_block_meta_header_carries_both_implementations():
    """The ACLNN branch carries the static Tiling/PTO compatibility and native
    accessors; the SK branch holds per-core block-local scalars; the else branch
    keeps the raw builtin wrappers."""
    assert "#include <pypto_block_meta.h>" in KERNEL_HEADER_SINGLE
    header = pathlib.Path(pypto_pro.__file__).parent / "include" / "pypto_block_meta.h"
    text = header.read_text()
    assert "#elif defined(__PYPTO_SK_BLOCK__)" in text
    aclnn_start = text.index("#if defined(__PYPTO_ACLNN__)")
    sk_start = text.index("#elif defined(__PYPTO_SK_BLOCK__)")
    else_start = text.index("#else // __PYPTO_SK_BLOCK__")
    aclnn_branch = text[aclnn_start:sk_start]
    sk_branch = text[sk_start:else_start]
    binary_branch = text[:sk_start]
    assert "return AscendC::GetBlockNum();" in aclnn_branch
    assert "return AscendC::GetBlockIdx();" in aclnn_branch
    assert "return AscendC::GetSubBlockNum();" in aclnn_branch
    assert "return AscendC::GetSubBlockIdx();" in aclnn_branch
    assert "PYPTO_ACLNN_CHECK_TILING_SIZE" in aclnn_branch
    assert "#include <pto/pto-inst.hpp>" in aclnn_branch
    assert '#pragma push_macro("get_block_num")' in aclnn_branch
    assert '#pragma pop_macro("get_block_num")' in aclnn_branch
    assert "#define get_block_num() AscendC::GetBlockNum()" in aclnn_branch
    assert "#define get_block_idx()" in aclnn_branch
    assert "__gm__ uint32_t" not in binary_branch
    assert "__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_num_v;" in sk_branch
    assert "__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_idx_v;" in sk_branch
    assert "__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_num_c;" in sk_branch
    assert "__BLOCK_LOCAL__ __inline__ uint32_t g_pypto_block_idx_c;" in sk_branch
    assert "return (int64_t)PYPTO_BLOCK_NUM;" in sk_branch
    assert "return (int64_t)PYPTO_BLOCK_IDX;" in sk_branch
    assert "PYPTO_BLOCK_NUM = v;" in sk_branch
    assert "PYPTO_BLOCK_IDX = v;" in sk_branch
    assert "__pypto_subblock_num_value" in sk_branch
    assert "__pypto_seed_subblock_num" in sk_branch
    assert "PYPTO_SUBBLOCK_NUM" in sk_branch
    assert "PYPTO_SUBBLOCK_IDX" in sk_branch
    assert "__pypto_subblock_idx_value" in sk_branch
    assert "__pypto_seed_subblock_idx" in sk_branch
    assert "g_pypto_block_meta" not in text
    assert "PYPTO_SK_BLOCK_SLOTS" not in text
    assert "dcci(" not in " ".join(text.split())
    assert "AscendC::Trap();" not in text
    non_sk_branch = text[else_start:]
    assert "return get_block_num();" in non_sk_branch
    assert "__pypto_subblock_num_value" in non_sk_branch
    assert "return get_subblockdim();" in non_sk_branch
    assert "__pypto_subblock_idx_value" in non_sk_branch
    assert "return get_subblockid();" in non_sk_branch


def test_simt_kernel_with_sk_cap_compiles(tmp_path):
    """Adapted SIMT VF may carry sk_cap; the outer SIMD entry supplies block values."""
    result = _codegen(_simt_sk_kernel.to_kernel_def(), NpuArch.DAV_3510, False, out_dir=str(tmp_path))
    assert result is not None
    assert "gridDim" not in result.content
    assert "__pypto_block_num_value()" in result.content
    assert "__pypto_block_idx_value()" in result.content


def test_simt_kernel_without_sk_cap_compiles(tmp_path):
    result = _codegen(_simt_no_sk_kernel.to_kernel_def(), NpuArch.DAV_3510, False, out_dir=str(tmp_path))
    assert result is not None


def test_simd_kernel_with_sk_cap_still_compiles(tmp_path):
    result = _codegen(_simd_sk_kernel.to_kernel_def(), NpuArch.DAV_3510, False, out_dir=str(tmp_path))
    assert result is not None
