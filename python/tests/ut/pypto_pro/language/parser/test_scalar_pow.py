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
"""Unsupported scalar power syntax is rejected while parsing the DSL."""

from pypto_pro import ir
from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
import pytest


@pytest.mark.parametrize("dtype", [pl.DT_INT32, pl.DT_FP16, pl.DT_BF16, pl.DT_FP32])
@pytest.mark.parametrize("vf", [False, True])
def test_power_syntax_is_rejected_before_codegen(dtype, vf):
    @pl.vector_function
    def power_vf(operand):
        _result = operand ** 2

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        operand = pl.astype(value, dtype)
        with pl.section_vector():
            if vf:
                power_vf(operand)
            else:
                _result = operand ** 2

    with pytest.raises(NotSupported, match="Unsupported binary operator: Pow"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
