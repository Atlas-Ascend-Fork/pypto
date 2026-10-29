# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Mutex counts belong to tile value types, independently of candidate IDs."""

from pypto_pro import ir
from pypto_pro.ir.op.block_ops import _ir_matmul_acc, _ir_reinterpret, make_mutex_tile_expr
import pytest


def test_subtype_preserves_tile_properties_and_exact_equality():
    base = ir.TileType([8, 32], ir.DataType.FP32)
    first = ir.MutexTileType(base, 2)
    same = ir.MutexTileType(base, 2)
    other = ir.MutexTileType(base, 1)
    assert isinstance(first, ir.TileType)
    assert isinstance(first, ir.ShapedType)
    assert first.dtype == base.dtype
    assert [dim.value for dim in first.shape] == [8, 32]
    assert first.mutex_count == 2
    assert ir.structural_equal(first, same)
    assert ir.structural_hash(first) == ir.structural_hash(same)
    assert not ir.structural_equal(first, other)
    assert not ir.structural_equal(base, ir.MutexTileType(base, 0))
    assert not ir.structural_equal(ir.TupleType([first]), ir.TupleType([other]))


@pytest.mark.parametrize("ids", [[], [1], [1, 3]])
def test_descriptor_operations_preserve_count(ids):
    tile = make_mutex_tile_expr([8, 32], ir.DataType.FP32, ir.MemorySpace.Vec, addr=0, mutex_ids=ids)
    assert isinstance(tile.type, ir.MutexTileType)
    assert tile.type.mutex_count == len(ids)
    view = _ir_reinterpret(tile, shape=[4, 64])
    assert isinstance(view.type, ir.MutexTileType)
    assert view.type.mutex_count == len(ids)
    span = tile.span

    def pair(a, b):
        return ir.MakeTuple([ir.ConstInt(a, ir.DataType.INT64, span), ir.ConstInt(b, ir.DataType.INT64, span)], span)

    subview = ir.create_op_call("block.subview", [tile, pair(1, 0), pair(4, 32)], {}, span)
    assert isinstance(subview.type, ir.MutexTileType)
    assert subview.type.mutex_count == len(ids)


def test_actual_ids_do_not_affect_type_equality():
    first = make_mutex_tile_expr([8, 32], ir.DataType.FP32, ir.MemorySpace.Vec, addr=0, mutex_ids=[1, 2])
    second = make_mutex_tile_expr([8, 32], ir.DataType.FP32, ir.MemorySpace.Vec, addr=1024, mutex_ids=[3, 4])
    assert ir.structural_equal(first.type, second.type)
    call = ir.create_op_call("block.move", [first, second], {}, first.span)
    assert isinstance(call.type, ir.NoneType)


def test_op_accepts_different_counts_and_returns_none():
    first = make_mutex_tile_expr([8, 32], ir.DataType.FP32, ir.MemorySpace.Vec, addr=0, mutex_ids=[1])
    second = make_mutex_tile_expr([8, 32], ir.DataType.FP32, ir.MemorySpace.Vec, addr=1024, mutex_ids=[2, 3])
    call = ir.create_op_call("block.add", [first, second, first], {}, first.span)
    assert isinstance(call.type, ir.NoneType)
    assert call.args[0].type.mutex_count == 1


def test_matmul_acc_compares_data_type_without_mutex_count():
    dst = make_mutex_tile_expr([64, 64], ir.DataType.FP32, ir.MemorySpace.Acc, addr=0, mutex_ids=[1])
    acc = make_mutex_tile_expr([64, 64], ir.DataType.FP32, ir.MemorySpace.Acc, addr=16384, mutex_ids=[2, 3])
    lhs = make_mutex_tile_expr([64, 64], ir.DataType.FP16, ir.MemorySpace.Left, addr=0, mutex_ids=[])
    rhs = make_mutex_tile_expr([64, 64], ir.DataType.FP16, ir.MemorySpace.Right, addr=0, mutex_ids=[])
    call = _ir_matmul_acc(dst, acc, lhs, rhs)
    assert isinstance(call.type, ir.NoneType)
