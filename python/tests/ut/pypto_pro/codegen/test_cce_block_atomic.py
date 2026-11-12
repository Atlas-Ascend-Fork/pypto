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

import re

import pypto_pro.language as pl
import pytest


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
        pl.setval(out, 0, pl.atomic_add(i32, 9, 1))


def test_rmw_kinds_emit_their_intrinsics_on_typed_gm_pointers():
    cpp = _compile_to_cce(_atomic_ops_codegen_kernel)

    for intrinsic, ctype in [
        ("atomicAdd", "int32_t"),
        ("atomicSub", "int32_t"),
        ("atomicExch", "int32_t"),
        ("atomicMax", "int32_t"),
        ("atomicMin", "int32_t"),
        ("atomicAnd", "int32_t"),
        ("atomicOr", "int32_t"),
        ("atomicXOr", "int32_t"),
        ("atomicInc", "uint64_t"),
        ("atomicDec", "uint64_t"),
        ("atomicAdd", "float"),
    ]:
        assert f"{intrinsic}((__gm__ {ctype}*)" in cpp


def test_cas_emits_atomiccas_with_compare_and_value():
    cpp = _compile_to_cce(_atomic_ops_codegen_kernel)

    assert re.search(r"atomicCAS\(\(__gm__ int32_t\*\)[^;]*, 0, 9\)", cpp)


def test_old_value_flows_into_the_consumer_without_an_atomic_cast():
    cpp = _compile_to_cce(_atomic_ops_codegen_kernel)

    # The intrinsic already returns the element dtype, so the old value enters
    # the scalar expression uncast; only the consumer (setval into an INT64
    # tensor) may add its own conversion.
    assert not re.search(r"\(\(int64_t\)__atomic_old_\d+\)", cpp)
    assert re.search(r"_expr_tmp_\d+_\d+ = __atomic_old_\d+;", cpp)


def test_discarded_old_value_still_emits_the_call():
    cpp = _compile_to_cce(_atomic_ops_codegen_kernel)

    # Every call binds the old value (the RMW returns it); discarded ones simply
    # never read the binding, but the side effect stays in the source.
    assert cpp.count("auto __atomic_old_") == 13


def test_cube_section_is_rejected():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(state: pl.Tensor[[64], pl.DT_INT32]):
        with pl.section_cube():
            pl.atomic_add(state, 0, 1)

    with pytest.raises(NotImplementedError, match="Vector section"):
        _compile_to_cce(kernel)
