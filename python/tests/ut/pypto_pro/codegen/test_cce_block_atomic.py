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

"""CCE emission for the ``pl.atomic_*`` GM Tensor scalar atomics.

Both IR families must lower to the device scalar-atomic intrinsics with a typed
``__gm__`` element pointer and the element-typed operand; the widened old value
is cast where the IR result type is wider.
"""

import pypto_pro.language as pl


def _compile_to_cce(kernel, arch: str = "3510") -> str:
    from pypto_pro.runtime.jit import _add_kernel_header, _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import normalize_arch

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), normalize_arch(arch), "")
    return _add_kernel_header(_assemble_cv_source(cube, vector)).content


@pl.jit(arch="3510", auto_mutex=False)
def _atomic_ops_codegen_kernel(
    i32: pl.Tensor[[64], pl.DT_INT32],
    u64: pl.Tensor[[64], pl.DT_UINT64],
    f32: pl.Tensor[[64], pl.DT_FP32],
    out: pl.Tensor[[2], pl.DT_INT64],
    fout: pl.Tensor[[1], pl.DT_FP32],
):
    with pl.section_vector():
        pl.atomic_add(i32, 0, 1)
        pl.atomic_sub(i32, 1, 2)
        pl.atomic_exch(i32, 2, 3)
        pl.atomic_max(i32, 3, 4)
        pl.atomic_min(i32, 4, 5)
        pl.atomic_and(i32, 5, 6)
        pl.atomic_or(i32, 6, 7)
        pl.atomic_xor(i32, 7, 8)
        pl.atomic_inc(u64, 0, 5)
        pl.atomic_dec(u64, 1, 5)
        pl.atomic_cas(i32, 8, 0, 9)
        pl.atomic_add(f32, 0, 1.5)
        pl.atomic_store(i32, 10, 11)
        pl.atomic_store(f32, 1, 2.5)
        pl.setval(out, 0, pl.atomic_add(i32, 9, 1))
        pl.setval(out, 1, pl.atomic_load(i32, 10))
        pl.setval(fout, 0, pl.atomic_load(f32, 2))


def test_atomic_ops_aiv(assert_golden):
    assert_golden(_compile_to_cce(_atomic_ops_codegen_kernel), "cce")


def test_atomic_ops_aic(assert_golden):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(
        state: pl.Tensor[[64], pl.DT_INT32],
        u64: pl.Tensor[[64], pl.DT_UINT64],
        f32: pl.Tensor[[64], pl.DT_FP32],
        out: pl.Tensor[[2], pl.DT_INT64],
    ):
        with pl.section_cube():
            pl.atomic_add(state, 0, 1)
            pl.atomic_sub(state, 1, 2)
            pl.atomic_exch(state, 2, 3)
            pl.atomic_max(state, 3, 4)
            pl.atomic_min(state, 4, 5)
            pl.atomic_and(state, 5, 6)
            pl.atomic_or(state, 6, 7)
            pl.atomic_xor(state, 7, 8)
            pl.atomic_inc(u64, 0, 5)
            pl.atomic_dec(u64, 1, 5)
            pl.atomic_store(state, 8, 9)
            pl.atomic_store(f32, 0, 2.5)
            out[0] = pl.atomic_load(state, 8)
            out[1] = pl.atomic_cas(state, 9, 0, 3)

    assert_golden(_compile_to_cce(kernel), "cce")
