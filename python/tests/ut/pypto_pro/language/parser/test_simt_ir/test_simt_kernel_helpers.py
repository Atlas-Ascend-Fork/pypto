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

"""Integer Tensor reads in SIMT entries and nested helpers."""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl
import pytest


@pl.vector_function(mode="simt")
def _identity(value):
    return value


@pl.vector_function(mode="simt")
def _load(src, index):
    return _identity(src[index])


@pl.vector_function(mode="simt")
def _store(dst, src):
    index = pl.simt.linear_thread_idx()
    dst[index] = _load(src, index)


@pytest.mark.parametrize("dtype", [
    pl.DT_INT8, pl.DT_INT16, pl.DT_INT32, pl.DT_INT64,
    pl.DT_UINT8, pl.DT_UINT16, pl.DT_UINT32, pl.DT_UINT64,
], ids=lambda dtype: dtype.to_string())
@pytest.mark.parametrize("mode", ["simt", "simd"])
def test_integer_load_preserves_dtype_in_entry_and_nested_helper(dtype, mode):
    @pl.vector_function(mode="simt", max_threads=32)
    def mixed(dst, src):
        dst[0] = _identity(src[0])
        _store(dst, src)

    @pl.jit(mode="simt", max_threads=32)
    def pure(dst: pl.Tensor[[32], dtype], src: pl.Tensor[[32], dtype]):
        dst[0] = _identity(src[0])
        _store(dst, src)

    @pl.jit
    def host(dst: pl.Tensor[[32], dtype], src: pl.Tensor[[32], dtype]):
        with pl.section_vector():
            mixed[32](dst, src)

    program, _ = (pure if mode == "simt" else host).to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    printed = str(program)
    golden_path = Path(__file__).with_name(f"{Path(__file__).stem}__{mode}_{dtype.to_string()}.pypto")
    golden = golden_path.read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
