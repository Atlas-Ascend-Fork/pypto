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
"""SK block adaptation: Tile accessor detection + SIMT rejection.

The Tile block num/idx accessors are emitted unconditionally by the CCE backend and
implemented in ``pypto_pro/include/pypto_block_meta.h`` (SK per-core records vs
raw-builtin wrappers selected by ``-D__PYPTO_SK_BLOCK__``) — the jit only detects their
presence to gate the entry-side seeds. SIMT is not adapted yet: under a fused container
the SIMT block grid would be the container's, so ``sk_cap`` on a kernel that launches
SIMT is rejected in ``_codegen`` before any SK artifact is produced.
"""
import pathlib

import pypto_pro
from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
from pypto_pro.runtime.jit import KERNEL_HEADER_SINGLE, _codegen, _detect_sk_block_usage
from pypto_pro.runtime.platform import NpuArch
import pytest


@pl.vector_function(mode="simt", max_threads=128)
def _simt_copy(x: pl.Tensor[[1, 1024], pl.DT_FP32], y: pl.Tensor[[1, 1024], pl.DT_FP32]):
    tid = pl.simt.linear_thread_idx()
    bid = pl.simt.block_idx().x
    grid = pl.simt.grid_dim().x
    for i in pl.range(bid * 128 + tid, 1024, grid * 128):
        y[0, i] = x[0, i]


@pl.jit(sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
def _simt_sk_kernel(x: pl.Tensor[[1, 1024], pl.DT_FP32], y: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        _simt_copy[128](x, y)


@pl.jit(auto_mutex=False)
def _simt_no_sk_kernel(x: pl.Tensor[[1, 1024], pl.DT_FP32], y: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        _simt_copy[128](x, y)


@pl.jit(sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
def _simd_sk_kernel(out: pl.Tensor[[64], pl.DT_INT64]):
    out[0] = pl.get_block_num()


def test_tile_usage_detection_matches_backend_emission():
    content = (
        "__aicore__ inline void f()\n"
        "{\n"
        "    auto n = __pypto_block_num_value();\n"
        "    auto i = __pypto_block_idx_value();\n"
        "}\n"
    )
    assert _detect_sk_block_usage(content) == (True, True)
    assert _detect_sk_block_usage("void f() {}\n") == (False, False)
    assert _detect_sk_block_usage("auto n = __pypto_block_num_value();\n") == (True, False)
    assert _detect_sk_block_usage("auto i = __pypto_block_idx_value();\n") == (False, True)


def test_block_meta_header_carries_both_implementations():
    """The SK branch holds the DCCI-guarded __gm__ records; the else branch the raw
    builtin wrappers. The TU header includes it for every kernel (the codegen-emitted
    accessors resolve either way), and the header must not shadow the builtins with
    macros."""
    assert "#include <pypto_block_meta.h>" in KERNEL_HEADER_SINGLE
    header = pathlib.Path(pypto_pro.__file__).parent / "include" / "pypto_block_meta.h"
    text = header.read_text()
    assert "#ifdef __PYPTO_SK_BLOCK__" in text
    assert "constexpr uint32_t PYPTO_SK_BLOCK_SLOTS = 72;" in text
    assert "__gm__ uint32_t g_pypto_block_meta_vec[PYPTO_SK_BLOCK_SLOTS][128];" in text
    assert "__gm__ uint32_t g_pypto_block_meta_cube[PYPTO_SK_BLOCK_SLOTS][128];" in text
    assert "AscendC::Trap();" in text
    # Whitespace-normalized: the header may wrap the long DCCI call across lines.
    flattened = " ".join(text.split())
    assert (
        "dcci(reinterpret_cast<__gm__ void*>(&g_pypto_block_meta_vec[idx][0]), "
        "SINGLE_CACHE_LINE, CACHELINE_OUT)" in flattened
    )
    assert "__BLOCK_LOCAL__ uint32_t" not in text
    assert "#define get_block_num()" not in text
    assert "#define get_block_idx()" not in text


def test_simt_kernel_with_sk_cap_is_rejected(tmp_path):
    """A SIMT launch combined with sk_cap fails the compile, not the device run."""
    with pytest.raises(NotSupported, match="sk_cap is not supported for SIMT kernels"):
        _codegen(_simt_sk_kernel.to_kernel_def(), NpuArch.DAV_3510, False, out_dir=str(tmp_path), sk_cap=4)


def test_simt_kernel_without_sk_cap_compiles(tmp_path):
    result = _codegen(_simt_no_sk_kernel.to_kernel_def(), NpuArch.DAV_3510, False, out_dir=str(tmp_path))
    assert result is not None


def test_simd_kernel_with_sk_cap_still_compiles_and_detects_usage(tmp_path):
    result = _codegen(_simd_sk_kernel.to_kernel_def(), NpuArch.DAV_3510, False, out_dir=str(tmp_path), sk_cap=4)
    assert result is not None
    assert result.uses_block_num is True
    assert result.uses_block_idx is False
