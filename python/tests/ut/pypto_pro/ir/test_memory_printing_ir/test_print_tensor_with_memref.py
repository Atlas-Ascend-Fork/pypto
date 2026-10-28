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
"""Extracted from ir/test_memory_printing.py: test_print_tensor_with_memref.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import DataType, ir


def test_print_tensor_with_memref():
    """Test Python printing of TensorType with MemRef."""
    span = ir.Span.unknown()
    shape = [ir.ConstInt(10, DataType.INT64, span)]

    memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 40, 25)

    tensor_type = ir.TensorType(shape, DataType.FP32, memref)
    tensor_var = ir.Var("tensor", tensor_type, span)
    stmt = ir.AssignStmt(tensor_var, ir.ConstInt(0, DataType.INT64, span), span)

    printed = str(stmt)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
