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
"""Tests for the scalar convenience APIs ``pl.floor`` and ``pl.ceildiv``.

``pl.floor`` mirrors ``pl.scalar_abs``/``pl.ceil``: one scalar operand lowering to an
``ir.Floor`` CCE scalar that folds constant operands at parse time. ``pl.ceildiv``
lowers to ``ir.CeilDiv`` and folds constant operands at parse time; runtime operands
stay as a single ``ir.CeilDiv`` node.
"""

import math

from pypto_pro._errors import InvalidType, NotSupported, PyptoProError
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _floor_nodes(function):
    return [
        stmt.value for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Floor)
    ]


def _ceildiv_nodes(function):
    return [
        stmt.value
        for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.CeilDiv)
    ]


def test_scalar_floor_builds_a_floor_expr():
    """pl.floor(scalar) dispatches to ir.Floor and keeps the operand dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_FP32],
        out: pl.Tensor[[1], pl.DT_FP32],
    ):
        x: pl.DT_FP32 = pl.getval(config, 0)
        c = pl.floor(x)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _floor_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_FP32


def test_scalar_floor_keeps_an_fp16_operand_dtype():
    """A half-precision operand reaches ir.Floor with its own dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_FP16],
        out: pl.Tensor[[1], pl.DT_FP16],
    ):
        x: pl.DT_FP16 = pl.getval(config, 0)
        c = pl.floor(x)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _floor_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_FP16


def test_scalar_floor_keeps_a_bf16_operand_dtype():
    """A bfloat16 operand reaches ir.Floor with its own dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_BF16],
        out: pl.Tensor[[1], pl.DT_BF16],
    ):
        x: pl.DT_BF16 = pl.getval(config, 0)
        c = pl.floor(x)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _floor_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_BF16


def test_floor_folds_a_negative_float_constant():
    """A compile-time operand folds to a rounded-down constant, with no ir.Floor."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.floor(pl.const(-2.3, pl.DT_FP32))
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _floor_nodes(function)
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstFloat)
        and stmt.value.value == -3.0
        and stmt.value.type.dtype == pl.DT_FP32
    ]
    assert len(assigns) == 1


def test_ceildiv_builds_a_ceil_div_expr():
    """pl.ceildiv dispatches to ir.CeilDiv and never folds for runtime operands."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[1], pl.DT_INT64],
    ):
        a: pl.DT_INT64 = pl.getval(config, 0)
        b: pl.DT_INT64 = pl.getval(config, 1)
        c = pl.ceildiv(a, b)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceildiv_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_INT64


def test_ceildiv_keeps_an_fp16_operand_dtype():
    """Half-precision operands reach ir.CeilDiv with their own dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[2], pl.DT_FP16],
        out: pl.Tensor[[1], pl.DT_FP16],
    ):
        a: pl.DT_FP16 = pl.getval(config, 0)
        b: pl.DT_FP16 = pl.getval(config, 1)
        c = pl.ceildiv(a, b)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceildiv_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_FP16


def test_ceildiv_keeps_a_bf16_operand_dtype():
    """Bfloat16 operands reach ir.CeilDiv with their own dtype."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[2], pl.DT_BF16],
        out: pl.Tensor[[1], pl.DT_BF16],
    ):
        a: pl.DT_BF16 = pl.getval(config, 0)
        b: pl.DT_BF16 = pl.getval(config, 1)
        c = pl.ceildiv(a, b)
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceildiv_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_BF16


def test_ceildiv_folds_constant_operands_at_parse_time():
    """Two compile-time operands fold to ceil(lhs/rhs) with no device node."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[3], pl.DT_INT32]):
        c0 = pl.ceildiv(pl.const(7, pl.DT_INT32), pl.const(2, pl.DT_INT32))
        c1 = pl.ceildiv(pl.const(-7, pl.DT_INT32), pl.const(2, pl.DT_INT32))
        c2 = pl.ceildiv(pl.const(8, pl.DT_INT32), pl.const(-2, pl.DT_INT32))
        out[0] = c0
        out[1] = c1
        out[2] = c2
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _ceildiv_nodes(function)
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstInt)
        and stmt.value.type.dtype == pl.DT_INT32
    ]
    assert [stmt.value.value for stmt in assigns] == [4, -3, -4]


def test_ceildiv_rejects_a_zero_divisor_constant():
    """A compile-time zero divisor is refused at parse time."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_INT64],
        out: pl.Tensor[[1], pl.DT_INT64],
    ):
        a: pl.DT_INT64 = pl.getval(config, 0)
        c = pl.ceildiv(a, 0)
        out[0] = c
        _test_result = out

    with pytest.raises(PyptoProError) as excinfo:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "zero divisor" in str(excinfo.value)


def test_tile_arguments_are_refused():
    """The ops take scalars only; a Tile is not one."""

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 8], pl.DT_FP32]):
        source_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(source_type, addr=0)
        pl.load(tile, x, [0, 0])
        c = pl.floor(tile) + pl.ceildiv(tile, tile)
        _test_result = c

    with pytest.raises(PyptoProError) as excinfo:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "expects Union[int, float]" in str(excinfo.value)


def test_a_none_operand_is_refused():
    """None parses to an expression of ir.NoneType, which both declarations refuse."""

    @pl.jit(auto_mutex=False)
    def floor_kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.floor(None)
        _test_result = c

    @pl.jit(auto_mutex=False)
    def ceildiv_lhs_kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.ceildiv(None, 2.0)
        _test_result = c

    @pl.jit(auto_mutex=False)
    def ceildiv_rhs_kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.ceildiv(2.0, None)
        _test_result = c

    expected = [
        (floor_kernel, "pl.floor: x expects Union[int, float], got NoneType"),
        (ceildiv_lhs_kernel, "pl.ceildiv: lhs expects Union[int, float], got NoneType"),
        (ceildiv_rhs_kernel, "pl.ceildiv: rhs expects Union[int, float], got NoneType"),
    ]
    for kernel, message in expected:
        with pytest.raises(InvalidType) as excinfo:
            kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        assert message in str(excinfo.value)


def test_extra_arguments_are_refused():
    """Each op takes exactly its positional arguments."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_INT64],
    ):
        x: pl.DT_INT64 = pl.getval(config, 0)
        c = pl.floor(x, x) + pl.ceildiv(x, x, x)
        _test_result = c

    with pytest.raises(PyptoProError):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_floor_of_an_unsigned_constant_folds_identically():
    """An unsigned integer operand is already integral; floor leaves it unchanged."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_UINT32]):
        c = pl.floor(pl.const(7, pl.DT_UINT32))
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _floor_nodes(function)
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstInt)
        and stmt.value.value == 7
        and stmt.value.type.dtype == pl.DT_UINT32
    ]
    assert len(assigns) == 1


def test_ceildiv_supports_unsigned_operands():
    """ceildiv of unsigned constants folds; a runtime UINT64 divisor builds ir.CeilDiv."""

    @pl.jit(auto_mutex=False)
    def kernel(
        config: pl.Tensor[[1], pl.DT_UINT64],
        out: pl.Tensor[[2], pl.DT_UINT64],
    ):
        x: pl.DT_UINT64 = pl.getval(config, 0)
        folded = pl.ceildiv(pl.const(7, pl.DT_UINT64), pl.const(2, pl.DT_UINT64))
        runtime = pl.ceildiv(x, pl.const(3, pl.DT_UINT64))
        out[0] = folded
        out[1] = runtime
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    nodes = _ceildiv_nodes(function)
    assert len(nodes) == 1
    assert nodes[0].type.dtype == pl.DT_UINT64
    assigns = [
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.ConstInt)
        and stmt.value.value == 4
        and stmt.value.type.dtype == pl.DT_UINT64
    ]
    assert len(assigns) == 1


def test_floor_folds_a_zero_result_to_positive_zero():
    """The device keeps the operand sign on a zero result, so folding must too."""

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        c = pl.floor(pl.const(0.4, pl.DT_FP32))
        out[0] = c
        _test_result = out

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")

    assert not _floor_nodes(function)
    folded = [
        stmt.value for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.ConstFloat)
    ]
    assert len(folded) == 1
    assert folded[0].value == 0.0
    assert math.copysign(1.0, folded[0].value) == 1.0


def test_float_ceildiv_is_refused_inside_a_vector_function():
    """Float ceiling division emits fmod-driven scalar math, which a VF body cannot hold."""

    @pl.vector_function
    def probe_vf(src, dst, count):
        c = pl.ceildiv(pl.const(3.0, pl.DT_FP32), pl.const(2.0, pl.DT_FP32))
        _test_result = c

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[1, 128], pl.DT_FP32],
        y: pl.Tensor[[1, 128], pl.DT_FP32],
        count: pl.DT_INT64,
    ):
        tile_type = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
        outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
        with pl.section_vector():
            probe_vf(inputs.next(), outputs.next(), count)

    with pytest.raises(NotSupported, match="Floating-point 'pl.ceildiv'"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
