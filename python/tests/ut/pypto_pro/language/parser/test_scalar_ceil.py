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
"""Tests for the scalar ceiling API ``pl.ceil``.

``pl.ceil`` mirrors ``pl.scalar_abs``: one scalar operand, value-expression form,
building an ``ir.Ceil`` scalar expression that folds constant operands at parse
time (the rounded-up value is emitted as a constant, no device node).
"""

from pypto_pro._errors import InvalidType, PyptoProError
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _ceil_nodes(function):
    return [
        stmt.value for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Ceil)
    ]


def test_scalar_ceil_builds_a_ceil_expr():
    """pl.ceil(scalar) dispatches to ir.Ceil and keeps the operand dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_FP32],
        out: pl.Tensor[[1], pl.DT_FP32],
    ):
        x: pl.DT_FP32 = pl.getval(config, 0)
        c = pl.ceil(x)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceil_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_FP32


def test_scalar_ceil_keeps_an_fp16_operand_dtype():
    """A half-precision operand reaches ir.Ceil with its own dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_FP16],
        out: pl.Tensor[[1], pl.DT_FP16],
    ):
        x: pl.DT_FP16 = pl.getval(config, 0)
        c = pl.ceil(x)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceil_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_FP16


def test_scalar_ceil_keeps_a_bf16_operand_dtype():
    """A bfloat16 operand reaches ir.Ceil with its own dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_BF16],
        out: pl.Tensor[[1], pl.DT_BF16],
    ):
        x: pl.DT_BF16 = pl.getval(config, 0)
        c = pl.ceil(x)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceil_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_BF16


def test_ceil_folds_a_float_constant_at_parse_time():
    """A compile-time operand folds to a rounded-up constant, with no ir.Ceil."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.ceil(pl.const(2.3, pl.DT_FP32))
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _ceil_nodes(function)
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstFloat)
        and stmt.value.value == 3.0
        and stmt.value.type.dtype == pl.DT_FP32
    ]
    assert len(assigns) == 1


def test_an_integer_constant_folds_identically():
    """ceil of an integer constant is the integer itself."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_INT64]):
        c = pl.ceil(pl.const(5, pl.DT_INT64))
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _ceil_nodes(function)
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstInt)
        and stmt.value.value == 5
        and stmt.value.type.dtype == pl.DT_INT64
    ]
    assert len(assigns) == 1


def test_a_tile_argument_is_refused():
    """The one-argument op takes a scalar only; a Tile is not one."""

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 8], pl.DT_FP32]):
        source_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(source_type, addr=0)
        pl.load(tile, x, [0, 0])
        c = pl.ceil(tile)
        _test_result = c

    with pytest.raises(PyptoProError) as excinfo:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "expects Union[int, float]" in str(excinfo.value)


def test_a_none_argument_is_refused():
    """None parses to an expression of ir.NoneType, which the scalar declaration refuses."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.ceil(None)
        _test_result = c

    with pytest.raises(InvalidType) as excinfo:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "pl.ceil: x expects Union[int, float], got NoneType" in str(excinfo.value)


def test_an_extra_argument_is_refused():
    """The scalar form takes exactly one positional argument."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_FP32],
    ):
        x: pl.DT_FP32 = pl.getval(config, 0)
        c = pl.ceil(x, x)
        _test_result = c

    with pytest.raises(PyptoProError):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_ceil_of_an_unsigned_constant_folds_identically():
    """An unsigned integer operand is already integral; ceil leaves it unchanged."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_UINT32]):
        c = pl.ceil(pl.const(7, pl.DT_UINT32))
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _ceil_nodes(function)
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstInt)
        and stmt.value.value == 7
        and stmt.value.type.dtype == pl.DT_UINT32
    ]
    assert len(assigns) == 1


def test_ceil_builds_a_uint_expr_for_a_runtime_unsigned_scalar():
    """A runtime UINT64 scalar lowers to ir.Ceil with the unsigned dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_UINT64],
        out: pl.Tensor[[1], pl.DT_UINT64],
    ):
        x: pl.DT_UINT64 = pl.getval(config, 0)
        c = pl.ceil(x)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceil_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_UINT64


def test_ceil_folds_a_zero_result():
    """A constant operand in (-1, 0) folds to a plain zero and leaves no device node.

    ``math.ceil`` returns an integer, so the folded constant does not carry the
    operand sign the device would produce; only the value is asserted here.
    """

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.ceil(pl.const(-0.4, pl.DT_FP32))
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _ceil_nodes(function)
    folded = [
        stmt.value for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.ConstFloat)
    ]
    assert len(folded) == 1
    assert folded[0].value == 0.0
