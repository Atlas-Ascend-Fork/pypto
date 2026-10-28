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
"""Extracted from ir/test_function_type.py: test_function_type_python_print.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import DataType, ir
from pypto_pro.ir import IRBuilder


def test_function_type_python_print():
    """Test that function type is correctly printed in Python syntax."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Ordinary functions print without a frontend decorator.
    with ib.function("default_func", span=span) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_opaque = f.get_result()

    # FunctionType remains IR metadata and is not serialized as a removed decorator.
    with ib.function("kernel", span=span, func_type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_incore = f.get_result()
    printed = "\n".join(
        s.rstrip("\n")
        for s in (str(func_opaque), str(func_incore))
    )
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
