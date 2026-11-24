# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generated-C++ checks for ``pl.array`` scalar arrays."""

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    return _assemble_cv_source(cube, vector).content


@pl.jit(auto_mutex=False)
def _basic_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    datas = pl.array(8, dtype=pl.DT_INT32)
    datas[0] = 7
    with pl.section_vector():
        pl.setval(out, 0, datas[0])


@pl.jit(auto_mutex=False)
def _alias_kernel(out: pl.Tensor[[3], pl.DT_INT32]):
    datas = pl.array(8, dtype=pl.DT_INT32)
    datas[0] = 7
    a = datas[0]
    alias = datas
    alias[0] = 9
    datas[1] = a
    with pl.section_vector():
        pl.setval(out, 0, a)
        pl.setval(out, 1, datas[0])
        pl.setval(out, 2, datas[1])


@pl.jit(auto_mutex=False)
def _independent_kernel(out: pl.Tensor[[2], pl.DT_INT32]):
    first = pl.array(2, dtype=pl.DT_INT32)
    second = pl.array(2, dtype=pl.DT_INT32)
    first[0] = 1
    second[0] = 2
    with pl.section_vector():
        pl.setval(out, 0, first[0])
        pl.setval(out, 1, second[0])


@pl.jit(auto_mutex=False)
def _outside_loop_kernel(out: pl.Tensor[[8], pl.DT_INT32]):
    datas = pl.array(8, dtype=pl.DT_INT32)
    for i in pl.range(0, 8):
        datas[i] = i
    with pl.section_vector():
        for i in pl.range(0, 8):
            pl.setval(out, i, datas[i])


@pl.jit(auto_mutex=False)
def _inside_loop_kernel(out: pl.Tensor[[8], pl.DT_INT32]):
    with pl.section_vector():
        for i in pl.range(0, 8):
            datas = pl.array(8, dtype=pl.DT_INT32)
            datas[0] = i
            pl.setval(out, i, datas[0])


def test_array_basic_codegen(assert_golden):
    assert_golden(_compile_to_cce(_basic_kernel), "cce")


def test_array_alias_codegen(assert_golden):
    assert_golden(_compile_to_cce(_alias_kernel), "cce")


def test_array_independent_codegen(assert_golden):
    assert_golden(_compile_to_cce(_independent_kernel), "cce")


def test_array_outside_loop_codegen(assert_golden):
    assert_golden(_compile_to_cce(_outside_loop_kernel), "cce")


def test_array_inside_loop_codegen(assert_golden):
    assert_golden(_compile_to_cce(_inside_loop_kernel), "cce")
