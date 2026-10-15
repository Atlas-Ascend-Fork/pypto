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
    def out_of_bounds(dst: pl.Tensor[[8, 64], pl.DT_FP32]):
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


def test_tile_slice_rejects_empty_static_range():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            _sub = tile[4:4, 0:32]

    with pytest.raises(InvalidOperation, match=r"axis 0 must satisfy start < min\(end, shape\)"):
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
