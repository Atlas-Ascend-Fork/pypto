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
"""Extracted from ir/test_printer.py: test_python_print_block_tensor_and_tile_types.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import DataType, ir


def test_python_print_block_tensor_and_tile_types():
    span = ir.Span.unknown()
    tensor_type = ir.TensorType(
        [ir.ConstInt(64, DataType.INDEX, span), ir.ConstInt(128, DataType.INDEX, span)],
        DataType.FP16,
    )
    memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INDEX, span), 512, 0)
    tile_type = ir.TileType(
        [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(16, DataType.INDEX, span)],
        DataType.FP16,
        memref,
    )

    printed = "\n".join(
        s.rstrip("\n")
        for s in (str(tensor_type), str(tile_type))
    )
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
