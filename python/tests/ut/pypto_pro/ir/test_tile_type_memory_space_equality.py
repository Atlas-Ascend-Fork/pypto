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
"""Tile type equality compares buffer kind while keeping allocation identity separate."""

from pypto_pro import DataType, ir
import pytest


def _tile(space=None, address=0, dtype=DataType.INT64):
    span = ir.Span.unknown()
    dims = [ir.ConstInt(d, DataType.INDEX, span) for d in (1, 64)]
    memref = ir.MemRef(space, ir.ConstInt(address, DataType.INDEX, span), 512, span) if space else None
    return ir.TileType(dims, dtype, memref) if memref is not None else ir.TileType(dims, dtype)


def test_distinct_buffers_have_distinct_types():
    assert not ir.structural_equal(_tile(ir.MemorySpace.Mat), _tile(ir.MemorySpace.Scaling))


def test_addresses_are_not_part_of_the_type():
    assert ir.structural_equal(_tile(ir.MemorySpace.Mat), _tile(ir.MemorySpace.Mat, 4096))


@pytest.mark.parametrize("space", [ir.MemorySpace.Mat, ir.MemorySpace.Vec,
                                  ir.MemorySpace.ScaleLeft, ir.MemorySpace.ScaleRight])
@pytest.mark.parametrize("reverse", [False, True])
def test_known_and_unspecified_memory_spaces_are_not_equal(space, reverse):
    lhs, rhs = _tile(space), _tile()
    if reverse:
        lhs, rhs = rhs, lhs
    assert not ir.structural_equal(lhs, rhs)
    with pytest.raises(RuntimeError, match="memory reference presence mismatch"):
        ir.assert_structural_equal(lhs, rhs)


def test_both_unspecified_memory_spaces_are_equal():
    assert ir.structural_equal(_tile(), _tile())


def test_dtype_still_participates_in_equality():
    assert not ir.structural_equal(_tile(ir.MemorySpace.Mat), _tile(ir.MemorySpace.Mat, dtype=DataType.FP16))
