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
"""Tests for the scalar absolute-value API ``pl.scalar_abs``.

The declaration takes ``Union[int, float]`` only, and the parser checks that
annotation against the parsed operand before the builder runs: a None expression
and a Tile (element-wise ``pl.abs`` territory) are both refused at parse time.
A valid operand builds ``ir.Abs``, and a constant operand folds at parse time.
"""

from pypto_pro._errors import InvalidType
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _abs_nodes(function):
    return [
        stmt.value for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Abs)
    ]


def test_scalar_abs_builds_an_abs_expr():
    """pl.scalar_abs(scalar) dispatches to ir.Abs and keeps the operand dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_FP32],
        out: pl.Tensor[[1], pl.DT_FP32],
    ):
        x: pl.DT_FP32 = pl.getval(config, 0)
        a = pl.scalar_abs(x)
        out[0] = a
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _abs_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_FP32


def test_abs_folds_a_negative_float_constant():
    """A compile-time operand folds to its magnitude, with no ir.Abs."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        a = pl.scalar_abs(pl.const(-2.5, pl.DT_FP32))
        out[0] = a
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _abs_nodes(function)
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstFloat)
        and stmt.value.value == 2.5
        and stmt.value.type.dtype == pl.DT_FP32
    ]
    assert len(assigns) == 1


def test_a_none_argument_is_refused():
    """None parses to an expression of ir.NoneType, which the scalar declaration refuses."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.scalar_abs(None)
        _test_result = c

    with pytest.raises(InvalidType) as excinfo:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "pl.scalar_abs: x expects Union[int, float], got NoneType" in str(excinfo.value)


def test_a_tile_argument_is_refused():
    """A Tile is not a scalar: the gate names the declared role and the value kind."""

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 8], pl.DT_FP32]):
        source_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(source_type, addr=0)
        pl.load(tile, x, [0, 0])
        c = pl.scalar_abs(tile)
        _test_result = c

    with pytest.raises(InvalidType) as excinfo:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "pl.scalar_abs: x expects Union[int, float], got TileType" in str(excinfo.value)
