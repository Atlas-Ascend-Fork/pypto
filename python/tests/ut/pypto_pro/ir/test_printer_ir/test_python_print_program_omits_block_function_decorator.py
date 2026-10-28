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
"""Extracted from ir/test_printer.py: test_python_print_program_omits_block_function_decorator.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import DataType, ir


def test_python_print_program_omits_block_function_decorator():
    span = ir.Span.unknown()
    x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
    fn = ir.Function(
        "f",
        [x],
        [ir.ScalarType(DataType.INT64)],
        ir.YieldStmt([x], span),
        span,
        type=ir.FunctionType.InCore,
    )
    program = ir.Program([fn], "BlockProgram", span)

    result = str(program)

    assert fn.func_type == ir.FunctionType.InCore
    printed = result
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
