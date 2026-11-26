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

"""Parser validation tests for Tile subscript access."""

from pypto_pro._errors import InvalidFormat, InvalidOperation, InvalidShape, InvalidType, InvalidVal, OutOfRange
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _parse_vector_kernel(kernel) -> None:
    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def _parse_cube_kernel(kernel) -> None:
    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def test_tile_subscript_rejects_negative_constant_index():
    @pl.vector_function(mode="simt", max_threads=32)
    def negative_index(dst):
        dst[-1, 0] = 0

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            negative_index[32](dst)

    with pytest.raises(InvalidVal, match="axis 0 must be non-negative, got -1"):
        _parse_vector_kernel(kernel)


def test_tensor_subscript_rejects_constant_index_at_dimension_size():
    @pl.vector_function(mode="simt", max_threads=32)
    def out_of_bounds(dst):
        dst[8, 0] = 0

    @pl.jit
    def kernel(dst: pl.Tensor[[8, 64], pl.DT_FP32]):
        with pl.section_vector():
            out_of_bounds[32](dst)

    with pytest.raises(OutOfRange, match="axis 0 is out of range for dimension size 8"):
        _parse_vector_kernel(kernel)


def test_tile_subscript_read_requires_ub_memory():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _value = tile[0, 0]

    with pytest.raises(
        InvalidOperation,
        match=r"getval: Tile element access requires a Vec-memory Tile \(UB\), got Mat",
    ):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_negative_static_bound():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[-1:4, 0:32]

    with pytest.raises(InvalidVal, match="slice start for axis 0 must be non-negative"):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_non_integer_static_bound():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[1.5:4, 0:32]

    with pytest.raises(InvalidType, match="slice start for axis 0 must be an integer scalar"):
        _parse_vector_kernel(kernel)


@pytest.mark.parametrize("start,valid_rows", [(0, None), (4, None), (8, None), (4, 4)])
def test_tile_slice_accepts_empty_static_range(start, valid_rows):
    valid_shape = None if valid_rows is None else [valid_rows, 64]

    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                                valid_shape=valid_shape)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[start:start, 0:32]

    _parse_vector_kernel(kernel)


def test_tile_slice_accepts_empty_static_tail():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                                valid_shape=[4, 64])
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _row_empty = tile[4:, 0:32]
            _col_empty = tile[0:4, 64:]
            _both_empty = tile[4:, 64:]

    _parse_vector_kernel(kernel)


def test_tile_slice_rejects_reversed_static_range():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[6:4, 0:32]

    with pytest.raises(InvalidShape, match="axis 0 must satisfy start <= end"):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_static_end_beyond_physical_shape():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[0:9, 0:32]

    with pytest.raises(InvalidShape, match=r"slice end \(9\) exceeds physical shape dim 0 \(8\)"):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_static_end_beyond_static_valid_shape():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[4, 64]
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[0:5, 0:32]

    with pytest.raises(InvalidShape, match=r"slice end \(5\) exceeds valid_shape dim 0 \(4\)"):
        _parse_vector_kernel(kernel)


def test_tile_slice_does_not_infer_set_validshape_bounds():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.set_validshape(tile, [4, 64])
            _sub = tile[0:5, 0:32]

    _parse_vector_kernel(kernel)


def test_tile_slice_omitted_end_uses_known_valid_shape():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[4, 64]
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub_exact = tile[0:4, 0:64]
            _sub = tile[0:, :]

    _parse_vector_kernel(kernel)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("row_end,col_end", [(6, 24), (4, 40)])
def test_nested_tile_slice_does_not_infer_parent_valid_shape_bounds(inline, row_end, col_end):
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.set_validshape(tile, [6, 40])
            if inline:
                _sub = tile[2:6, 16:40][2:row_end, 16:col_end]
            else:
                sub = tile[2:6, 16:40]
                alias = sub
                _sub = alias[2:row_end, 16:col_end]

    _parse_vector_kernel(kernel)


def test_nested_tile_slice_accepts_valid_window():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            sub = tile[2:6, 16:40]
            _sub = sub[2:4, 16:24]
            _inline = tile[2:6, 16:40][2:4, 16:24]

    _parse_vector_kernel(kernel)


def test_nested_tile_slice_omitted_end_preserves_valid_shape():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            sub = tile[2:6, 16:40]
            tail = sub[2:, :]
            _sub = tail[:, :]

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    subviews = [
        stmt for stmt in program.get_function(kernel.__name__).body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.name == "block.subview"
    ]
    assert len(subviews) == 3
    tail_shape = subviews[1].value.args[2].elements
    assert isinstance(tail_shape[0], ir.Sub)
    assert isinstance(tail_shape[0].left, ir.Call)
    assert tail_shape[0].left.name == "block.tile_valid_shape"
    assert tail_shape[0].left.kwargs["axis"] == 0
    assert tail_shape[0].right.value == 2
    assert ir.structural_equal(tail_shape[0].left.args[0], subviews[0].targets[0], enable_auto_mapping=False)
    for axis, dim in enumerate(subviews[2].value.args[2].elements):
        assert isinstance(dim, ir.Call)
        assert dim.name == "block.tile_valid_shape"
        assert dim.kwargs["axis"] == axis
        assert ir.structural_equal(dim.args[0], subviews[1].targets[0], enable_auto_mapping=False)


def test_nested_tile_slice_accepts_dynamic_valid_shape():
    @pl.jit(auto_mutex=False)
    def kernel(col_end: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            sub = tile[2:6, 16:col_end]
            _sub = sub[0:4, 0:16]

    _parse_vector_kernel(kernel)


def test_tile_slice_rejects_non_unit_step():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[0:8:2, 0:32]

    with pytest.raises(InvalidShape, match="slice step for axis 0 must be the compile-time integer 1"):
        _parse_vector_kernel(kernel)


def test_tile_slice_with_dynamic_valid_shape():
    @pl.jit(auto_mutex=False)
    def kernel(rows: pl.DT_INT64, cols: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[8, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.set_validshape(tile, [rows, cols])
            _sub = tile[1:, :]

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    subview = next(
        stmt.value for stmt in program.get_function(kernel.__name__).body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.name == "block.subview"
    )
    rows, cols = subview.args[2].elements
    assert isinstance(rows, ir.Sub)
    assert isinstance(rows.left, ir.Call)
    assert rows.left.name == "block.tile_valid_shape"
    assert rows.left.kwargs["axis"] == 0
    assert rows.right.value == 1
    assert isinstance(cols, ir.Call)
    assert cols.name == "block.tile_valid_shape"
    assert cols.kwargs["axis"] == 1
    assert ir.structural_equal(rows.left.args[0], subview.args[0], enable_auto_mapping=False)
    assert ir.structural_equal(cols.args[0], subview.args[0], enable_auto_mapping=False)


@pytest.mark.parametrize(
    ("layout", "row_start", "col_start"),
    [
        (pl.ND, 0, 20),
        (pl.DN, 20, 0),
    ],
)
def test_plain_tile_slice_rejects_unaligned_address(layout, row_start, col_start):
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=layout
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[row_start:row_start + 32, col_start:col_start + 64]

    with pytest.raises(InvalidShape, match="Tile slice start must produce a 32-byte-aligned Tile address"):
        _parse_vector_kernel(kernel)


@pytest.mark.parametrize(
    ("layout", "row_start", "col_start"),
    [
        (pl.ND, 1, 16),
        (pl.DN, 16, 1),
    ],
)
def test_plain_tile_slice_accepts_aligned_address(layout, row_start, col_start):
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=layout
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[row_start:row_start + 32, col_start:col_start + 64]

    _parse_vector_kernel(kernel)


def test_plain_tile_slice_accepts_dynamic_start():
    @pl.jit(auto_mutex=False)
    def kernel(col_start: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.ND
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[0:32, col_start:128]

    _parse_vector_kernel(kernel)


def test_nz_tile_slice_accepts_fractal_aligned_start():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[16:32, 16:32]

    _parse_vector_kernel(kernel)


@pytest.mark.parametrize("memory,layout,fractal,row,col,alignment", [
    (pl.MemorySpace.Acc, pl.NZ, 512, 16, 8, None),
    (pl.MemorySpace.Acc, pl.NZ, 1024, 16, 8, (16, 16)),
    (pl.MemorySpace.Mat, pl.ZN, 512, 8, 16, None),
    (pl.MemorySpace.Mat, pl.ZN, 1024, 8, 16, (16, 16)),
])
def test_tile_slice_alignment_follows_fractal(memory, layout, fractal, row, col, alignment):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=memory,
                                layout=layout, fractal=fractal)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_cube():
            _sub = tile[row:32, col:32]

    if alignment is None:
        _parse_cube_kernel(kernel)
    else:
        with pytest.raises(InvalidShape, match=rf"slice start must align to \[{alignment[0]}, {alignment[1]}\]"):
            _parse_cube_kernel(kernel)


def test_zn_tile_slice_accepts_fractal_aligned_start():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_cube():
            _sub = tile[16:32, 16:32]

    _parse_cube_kernel(kernel)


@pytest.mark.parametrize(
    ("memory_space", "layout"),
    [
        (pl.MemorySpace.Left, pl.NZ),
        (pl.MemorySpace.Right, pl.ZN),
    ],
)
def test_l0a_l0b_tile_slice_is_not_supported(memory_space, layout):
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[32, 32], dtype=pl.DT_FP16, target_memory=memory_space, layout=layout
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_cube():
            _sub = tile[16:32, 16:32]

    with pytest.raises(InvalidFormat, match="Tile slice supports Vec ND/DN/NZ, Mat ND/DN/NZ/ZN, and Acc NZ tiles"):
        _parse_cube_kernel(kernel)


@pytest.mark.parametrize(("row_start", "col_start"), [(1, 16), (16, 1)])
def test_nz_tile_slice_rejects_unaligned_start(row_start, col_start):
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[row_start:32, col_start:32]

    with pytest.raises(InvalidShape, match=r"NZ Tile slice start must align to \[16, 16\]"):
        _parse_vector_kernel(kernel)


def test_nz_tile_slice_accepts_dynamic_start():
    @pl.jit(auto_mutex=False)
    def kernel(row_start: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[row_start:32, 16:32]

    _parse_vector_kernel(kernel)


def test_nz_tile_slice_with_dynamic_row_rejects_unaligned_static_col():
    @pl.jit(auto_mutex=False)
    def kernel(row_start: pl.DT_INT64):
        tile_type = pl.TileType(
            shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
        )
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[row_start:32, 1:32]

    with pytest.raises(InvalidShape, match=r"NZ Tile slice start must align to \[16, 16\]"):
        _parse_vector_kernel(kernel)
