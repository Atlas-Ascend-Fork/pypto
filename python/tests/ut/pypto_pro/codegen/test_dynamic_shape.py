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
"""Unit tests for codegen with dynamic shape tensor parameters."""

from pypto_pro import ir
from pypto_pro._errors import InvalidShape, InvalidType
import pypto_pro.language as pl
import pytest


@pl.jit(auto_mutex=False)
def add_kernel_dynamic(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    output: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    """Adds two tensors element-wise with dynamic shapes: result = a + b"""
    tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a_tile = pl.make_tile(tile_type, addr=0x0000)
    b_tile = pl.make_tile(tile_type, addr=0x10000)
    result = pl.make_tile(tile_type, addr=0x20000)
    pl.load(a_tile, a, [0, 0])
    pl.load(b_tile, b, [0, 0])
    pl.add(result, a_tile, b_tile)
    out = pl.store(output, result, [0, 0])
    _test_result = out


add_kernel_dynamic_program, _ = add_kernel_dynamic.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
add_kernel_dynamic = add_kernel_dynamic_program.get_function(add_kernel_dynamic.__name__)


@pl.jit(auto_mutex=False)
def add_kernel_valid_shape(
    a: pl.Tensor[[128, 128], pl.DT_FP32],
    b: pl.Tensor[[128, 128], pl.DT_FP32],
    output: pl.Tensor[[128, 128], pl.DT_FP32],
    m_var: pl.DT_INT64,
    n_var: pl.DT_INT64,
):
    """Loads 128x128 tiles but marks only [M, N] as valid: result = a + b"""
    tile_type = pl.TileType(
        shape=[128, 128],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    a_tile = pl.make_tile(tile_type, addr=0x0000)
    b_tile = pl.make_tile(tile_type, addr=0x10000)
    result = pl.make_tile(tile_type, addr=0x20000)
    pl.set_validshape(a_tile, [m_var, n_var])
    pl.set_validshape(b_tile, [m_var, n_var])
    pl.set_validshape(result, [m_var, n_var])
    pl.load(a_tile, a, [0, 0])
    pl.load(b_tile, b, [0, 0])
    pl.add(result, a_tile, b_tile)
    pl.set_validshape(a_tile, [m_var, n_var])
    pl.set_validshape(b_tile, [m_var, n_var])
    pl.set_validshape(result, [m_var, n_var])
    out = pl.store(output, result, [0, 0])
    _test_result = out


add_kernel_valid_shape_program, _ = add_kernel_valid_shape.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
add_kernel_valid_shape = add_kernel_valid_shape_program.get_function(add_kernel_valid_shape.__name__)


@pl.jit(auto_mutex=False)
def static_shape_negative_subscript(
    a: pl.Tensor[[2, 128], pl.DT_FP32],
    output: pl.Tensor[[2, 128], pl.DT_FP32],
):
    n = a.shape[-1]
    out = output
    for _ in pl.range(0, n, 1):
        out = output
    _test_result = out


static_shape_negative_subscript_program, _ = static_shape_negative_subscript.to_kernel_def().parse_target_program(
    ir.SectionKind.Vector
)
static_shape_negative_subscript = static_shape_negative_subscript_program.get_function(
    static_shape_negative_subscript.__name__
)


@pl.jit(auto_mutex=False)
def dynamic_shape_tuple_unpack(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 128, 64], pl.DT_FP32],
):
    m, n, _, _ = a.shape
    out = a
    for _ in pl.range(0, m, 1):
        out = a
    _test_result = out


dynamic_shape_tuple_unpack_program, _ = dynamic_shape_tuple_unpack.to_kernel_def().parse_target_program(
    ir.SectionKind.Vector
)
dynamic_shape_tuple_unpack = dynamic_shape_tuple_unpack_program.get_function(dynamic_shape_tuple_unpack.__name__)


@pl.jit(auto_mutex=False)
def compile_time_shape_axis(
    a: pl.Tensor[[2, 128], pl.DT_FP32],
    output: pl.Tensor[[2, 128], pl.DT_FP32],
):
    n = a.shape[0]
    out = output
    for _ in pl.range(0, n, 1):
        out = output
    _test_result = out


compile_time_shape_axis_program, _ = compile_time_shape_axis.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
compile_time_shape_axis = compile_time_shape_axis_program.get_function(compile_time_shape_axis.__name__)


def test_negative_shape_subscript_uses_fixed_dimension():
    assert isinstance(static_shape_negative_subscript.params[0].type.shape[-1], ir.ConstInt)
    assert static_shape_negative_subscript.params[0].type.shape[-1].value == 128


def test_shape_tuple_unpack_uses_canonical_tensor_dimensions():
    assert isinstance(dynamic_shape_tuple_unpack.params[0].type.shape[0], ir.Var)
    assert dynamic_shape_tuple_unpack.params[0].type.shape[0].name == "__pypto_dyn_a_0"


def test_shape_tuple_unpack_rejects_arity_mismatch():
    with pytest.raises(InvalidShape, match="unpack"):

        @pl.jit(auto_mutex=False)
        def shape_unpack(x: pl.Tensor[[2, 128], pl.DT_FP32]):
            m, n, k = x.shape
            _test_result = x

        shape_unpack.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("index", [True, 0.0, 2, -3])
def test_shape_subscript_rejects_invalid_constant_index(index):
    with pytest.raises((InvalidShape, InvalidType)):

        @pl.jit(auto_mutex=False)
        def shape_index(x: pl.Tensor[[2, 128], pl.DT_FP32]):
            n = x.shape[index]  # noqa: F841
            _test_result = x

        shape_index.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_shape_subscript_rejects_non_tensor_base():
    with pytest.raises(InvalidShape):

        @pl.jit(auto_mutex=False)
        def shape_index(x: pl.DT_INT64):
            _test_result = x.shape[0]

        shape_index.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_shape_subscript_rejects_runtime_axis():
    with pytest.raises(InvalidType, match="tensor.shape index must be a compile-time integer"):

        @pl.jit(auto_mutex=False)
        def shape_index(
            x: pl.Tensor[[2, 128], pl.DT_FP32],
            axis: pl.DT_INT64,
        ):
            _test_result = x.shape[axis]

        shape_index.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
