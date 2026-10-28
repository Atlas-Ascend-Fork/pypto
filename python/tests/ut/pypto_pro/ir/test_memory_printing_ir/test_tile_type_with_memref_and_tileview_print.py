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
"""Extracted from ir/test_memory_printing.py: test_tile_type_with_memref_and_tileview_print.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import DataType, ir


def test_tile_type_with_memref_and_tileview_print():
    """Test printing TileType with MemRef variable name and TileView."""
    span = ir.Span.unknown()
    shape = [
        ir.ConstInt(16, DataType.INT64, span),
        ir.ConstInt(16, DataType.INT64, span),
    ]

    addr = ir.ConstInt(0x2000, DataType.INT64, span)
    memref = ir.MemRef(ir.MemorySpace.Left, addr, 512, 8)

    valid_shape = [
        ir.ConstInt(16, DataType.INT64, span),
        ir.ConstInt(16, DataType.INT64, span),
    ]
    stride = [
        ir.ConstInt(1, DataType.INT64, span),
        ir.ConstInt(16, DataType.INT64, span),
    ]
    start_offset = ir.ConstInt(0, DataType.INT64, span)
    tv = ir.TileView(valid_shape, stride, start_offset)

    tile_type = ir.TileType(shape, DataType.FP16, memref, tv)
    printed = str(tile_type)

    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
