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
"""Extracted from ir/test_memory_printing.py: test_tensor_type_with_memref_and_tensorview_print.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import DataType, ir


def test_tensor_type_with_memref_and_tensorview_print():
    """Test printing TensorType with both MemRef and TensorView."""
    span = ir.Span.unknown()
    shape = [
        ir.ConstInt(64, DataType.INT64, span),
        ir.ConstInt(64, DataType.INT64, span),
    ]
    stride = [
        ir.ConstInt(1, DataType.INT64, span),
        ir.ConstInt(64, DataType.INT64, span),
    ]

    addr = ir.ConstInt(0x5000, DataType.INT64, span)
    memref = ir.MemRef(ir.MemorySpace.Left, addr, 4096, 42)
    tensor_view = ir.TensorView(stride, ir.TensorLayout.NZ)
    tensor_type = ir.TensorType(shape, DataType.FP16, memref=memref, tensor_view=tensor_view)

    printed = str(tensor_type)

    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
