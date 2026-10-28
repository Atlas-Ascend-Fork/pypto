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
"""Extracted from ir/test_printer.py: test_python_print_block_load_store_ops.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import DataType, ir


def test_python_print_block_load_store_ops():
    span = ir.Span.unknown()
    dim = ir.ConstInt(64, DataType.INDEX, span)
    tensor = ir.Var("x", ir.TensorType([dim], DataType.FP16), span)
    memref = ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, DataType.INDEX, span), 128, 0)
    tile = ir.Var("tile", ir.TileType([dim], DataType.FP16, memref), span)
    zero = ir.ConstInt(0, DataType.INDEX, span)

    load = ir.Call(ir.Op("block.load"), [tile, tensor, ir.MakeTuple([zero], span)], span)
    store = ir.Call(ir.Op("block.store"), [tensor, tile, ir.MakeTuple([zero], span)], span)
    body = ir.SeqStmts([ir.EvalStmt(load, span), ir.EvalStmt(store, span)], span)

    result = str(body)

    printed = result
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
