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
"""Extracted from codegen/test_dynamic_shape.py: test_dynamic_shape_subscript_uses_parameter_shape_var.
Auto-assigned memref ids are process-global, so they are normalized to ``<id>``
before the whole-file comparison.
The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path
import re

from pypto_pro import DataType, ir
import pypto_pro.language as pl


@pl.jit(auto_mutex=False)
def add_kernel_shape_subscript(
    a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32],
    output: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP32],
):
    """Uses the canonical dynamic dimension stored in a TensorType."""
    m_var = a.shape[0]
    tile_type = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a_tile = pl.make_tile(tile_type, addr=0x0000)
    b_tile = pl.make_tile(tile_type, addr=0x0400)
    result = pl.make_tile(tile_type, addr=0x0800)
    out = output
    for i in pl.range(0, m_var, 2):
        offset_1 = i * 2
        pl.load(a_tile, a, [offset_1, 0])
        pl.load(b_tile, b, [offset_1, 0])
        pl.add(result, a_tile, b_tile)
        out = pl.store(output, result, [offset_1, 0])
    _test_result = out


add_kernel_shape_subscript_program, _ = add_kernel_shape_subscript.to_kernel_def().parse_target_program(
    ir.SectionKind.Vector
)
add_kernel_shape_subscript = add_kernel_shape_subscript_program.get_function(add_kernel_shape_subscript.__name__)


def test_dynamic_shape_subscript_uses_parameter_shape_var():
    dynamic_dim = add_kernel_shape_subscript.params[0].type.shape[0]

    assert isinstance(dynamic_dim, ir.Var)
    assert dynamic_dim.name == "__pypto_dyn_a_0"
    assert dynamic_dim.type.dtype == DataType.INT64
    printed = str(add_kernel_shape_subscript)
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
