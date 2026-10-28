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
"""Extracted from language/parser/test_printer_integration.py: test_parsed_function_printer_round_trip.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl


def test_parsed_function_printer_round_trip():
    """Test that parsed functions can be printed correctly."""

    @pl.jit(auto_mutex=False)
    def round_trip(
        x: pl.Tensor[[64], pl.DT_FP32],
        y: pl.Tensor[[64], pl.DT_FP32],
    ):
        sum_val: pl.Tensor[[64], pl.DT_FP32] = pl.make_tensor(x, [64], [1])
        result: pl.Tensor[[64], pl.DT_FP32] = pl.make_tensor(sum_val, [64], [1])
        _test_result = result

    round_trip_program, _ = round_trip.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    round_trip = round_trip_program.get_function(round_trip.__name__)

    # Print and check syntax
    printed = str(round_trip)

    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
