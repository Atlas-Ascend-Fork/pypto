# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tile views store valid shape only, with dynamic defaults supplied by TileType."""

from pypto_pro import ir
from pypto_pro.ir.op.block_ops import make_mutex_tile_expr
import pytest


def _memref():
    return ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, ir.DataType.INT64, ir.Span.unknown()), 4096, 0)


@pytest.mark.parametrize("shape", [[16], [16, 32], [2, 16, 32]])
@pytest.mark.parametrize("with_memref", [False, True])
def test_default_valid_shape_matches_rank(shape, with_memref):
    tile = ir.TileType(shape, ir.DataType.FP32, memref=_memref() if with_memref else None)
    assert [dim.value for dim in tile.tile_view.valid_shape] == [-1] * len(shape)
    expected_view = "tile_view<" + " x ".join(["-1"] * len(shape)) + ">"
    assert expected_view in str(tile)
    assert str(ir.text_load_type(str(tile))) == str(tile)


def test_empty_and_explicit_dynamic_views_match_defaults():
    memref = _memref()
    default = ir.TileType([16, 32], ir.DataType.FP32, memref)
    empty = ir.TileType([16, 32], ir.DataType.FP32, memref, ir.TileView())
    explicit = ir.TileType([16, 32], ir.DataType.FP32, memref, ir.IRBuilder().tile_view([-1, -1]))
    for tile in (empty, explicit):
        assert ir.structural_equal(default, tile)
        assert ir.structural_hash(default) == ir.structural_hash(tile)
        assert str(default) == str(tile)


def test_explicit_valid_shape_preserves_expressions():
    span = ir.Span.unknown()
    extent = ir.Var("extent", ir.ScalarType(ir.DataType.INT64), span)
    view = ir.IRBuilder().tile_view([8, -1, extent], span=span)
    tile = ir.TileType([16, 32, 64], ir.DataType.FP32, _memref(), view)
    assert [dim.value for dim in tile.tile_view.valid_shape[:2]] == [8, -1]
    assert ir.structural_equal(tile.tile_view.valid_shape[2], extent)
    assert not hasattr(view, "stride")
    assert not hasattr(view, "start_offset")


@pytest.mark.parametrize("mutex_ids", [[], [1, 2]])
def test_allocation_and_mutex_tile_default_valid_shape(mutex_ids):
    tile = make_mutex_tile_expr([16, 32], ir.DataType.FP32, ir.MemorySpace.Vec, addr=0, mutex_ids=mutex_ids)
    assert isinstance(tile.type, ir.MutexTileType)
    assert tile.type.mutex_count == len(mutex_ids)
    assert [dim.value for dim in tile.type.tile_view.valid_shape] == [-1, -1]
    assert "tile_view<-1 x -1>" in str(tile.type)
    assert str(ir.text_load_type(str(tile.type))) == str(tile.type)


def test_empty_text_view_is_normalized():
    tile = ir.text_load_type("tile<16 x 32, float, tile_view<>, hw_info<>>")
    assert str(tile) == "tile<16 x 32, float, tile_view<-1 x -1>, hw_info<>>"


def test_removed_tile_view_arguments_and_text_are_rejected():
    view = ir.IRBuilder().tile_view([16, 32])
    with pytest.raises(TypeError):
        ir.TileView(view.valid_shape, [], None)
    with pytest.raises(RuntimeError):
        ir.text_load_type("tile<16 x 32, float, tile_view<16 x 32, 32 x 1, 0>, hw_info<>>")
