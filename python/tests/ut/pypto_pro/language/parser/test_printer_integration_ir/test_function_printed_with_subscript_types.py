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
"""Extracted from language/parser/test_printer_integration.py: test_function_printed_with_subscript_types.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl


def test_function_printed_with_subscript_types():
    """Test that function parameters use subscript notation."""

    @pl.jit(auto_mutex=False)
    def test_func(x: pl.Tensor[[64, 128], pl.DT_FP16]):
        result: pl.Tensor[[64, 128], pl.DT_FP32] = pl.make_tensor(x, [64, 128], [128, 1], dtype=pl.DT_FP32)
        _test_result = result

    test_func_program, _ = test_func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_func = test_func_program.get_function(test_func.__name__)

    # Print the function
    printed = str(test_func)

    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
