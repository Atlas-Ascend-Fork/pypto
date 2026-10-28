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
"""Tests for function type attribute feature."""

import inspect

from pypto_pro import DataType, ir
from pypto_pro.ir import IRBuilder
import pypto_pro.language as pl


def test_ir_builder_with_function_type():
    """Test IR Builder with function type parameter."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Build function with InCore type
    with ib.function("aicore_kernel", span=span, func_type=ir.FunctionType.InCore) as f:
        y = f.param("y", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(y, span=span)

    func2 = f.get_result()
    assert func2.name == "aicore_kernel"
    assert func2.func_type == ir.FunctionType.InCore


def test_jit_kernel_uses_opaque_function_type():
    """pl.jit captures only Opaque kernel entries."""
    assert "func_type" not in inspect.signature(pl.jit).parameters

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[4], pl.DT_INT64]):
        _test_result = x

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    kernel = program.get_function(kernel.__name__)

    assert kernel.name == "kernel"
    assert kernel.func_type == ir.FunctionType.Opaque


def test_function_type_language_export():
    """Test that FunctionType is exported from language module."""

    assert hasattr(pl, "FunctionType")
    assert pl.FunctionType.Opaque
    assert pl.FunctionType.InCore
