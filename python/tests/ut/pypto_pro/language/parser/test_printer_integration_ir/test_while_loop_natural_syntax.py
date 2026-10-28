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
"""Extracted from language/parser/test_printer_integration.py: test_while_loop_natural_syntax.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl


def test_while_loop_natural_syntax():
    """Test that natural while loop can be parsed and printed."""

    @pl.jit(auto_mutex=False)
    def while_natural(n: pl.DT_INT64):
        x: pl.DT_INT64 = 0
        while x < n:
            x = x + 1
        _test_result = x

    while_natural_program, _ = while_natural.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_natural = while_natural_program.get_function(while_natural.__name__)

    # Print the function
    printed = str(while_natural)

    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")

    # Verify structural properties
    assert isinstance(while_natural, ir.Function)
    assert while_natural.name == "while_natural"
