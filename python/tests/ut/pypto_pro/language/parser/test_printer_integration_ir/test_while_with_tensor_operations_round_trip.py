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
"""Extracted from language/parser/test_printer_integration.py: test_while_with_tensor_operations_round_trip.
Auto-assigned memref ids are process-global, so they are normalized to ``<id>``
before the whole-file comparison.
The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path
import re

from pypto_pro import ir
import pypto_pro.language as pl


def test_while_with_tensor_operations_round_trip():
    """Test while loop with tensor operations."""

    @pl.jit(auto_mutex=False)
    def while_tensors(n: pl.DT_INT64, x: pl.Tensor[[64], pl.DT_FP32]):
        i: pl.DT_INT64 = 0
        tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        acc = pl.make_tile(tile_type, addr=0)
        while i < n:
            i = i + 1
            pl.add(acc, acc, acc)
        _test_result = acc

    while_tensors_program, _ = while_tensors.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    while_tensors = while_tensors_program.get_function(while_tensors.__name__)

    # Print the function
    printed = str(while_tensors)

    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
