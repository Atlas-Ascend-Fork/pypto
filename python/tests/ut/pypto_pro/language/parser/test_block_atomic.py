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

"""Parse-level contracts for the ``pl.atomic_*`` GM Tensor scalar atomics.

Every public name lowers to one of two IR ops: ``block.atomic_rmw`` carries the
kind, ``block.atomic_cas`` is the compare-and-swap form. These tests pin that
mapping, the linear-offset/value typing, and the rejection paths.
"""

from pypto_pro._errors import InvalidType, NotSupported
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _parse(kernel):
    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return program.get_function(kernel.__name__)


def _op_calls(func, op_name):
    """Every IR call to *op_name* in the parsed body, in program order."""
    calls = []
    for stmt in func.body.stmts:
        value = getattr(stmt, "value", None) if isinstance(stmt, ir.AssignStmt) else getattr(stmt, "expr", None)
        if isinstance(value, ir.Call) and value.name == op_name:
            calls.append(value)
    return calls


@pl.jit(arch="3510", auto_mutex=False)
def _rmw_kernel(
    i32: pl.Tensor[[16], pl.DT_INT32],
    u32: pl.Tensor[[16], pl.DT_UINT32],
    f32: pl.Tensor[[16], pl.DT_FP32],
):
    with pl.section_vector():
        pl.atomic_add(i32, 0, 1)
        pl.atomic_sub(i32, 1, 1)
        pl.atomic_exch(i32, 2, 1)
        pl.atomic_max(i32, 3, 1)
        pl.atomic_min(i32, 4, 1)
        pl.atomic_and(i32, 5, 1)
        pl.atomic_or(u32, 0, 1)
        pl.atomic_xor(i32, 6, 1)
        pl.atomic_inc(u32, 1, 5)
        pl.atomic_dec(u32, 2, 5)
        pl.atomic_add(f32, 0, 1.5)


@pl.jit(arch="3510", auto_mutex=False)
def _cas_kernel(i32: pl.Tensor[[16], pl.DT_INT32], u64: pl.Tensor[[16], pl.DT_UINT64]):
    with pl.section_vector():
        pl.atomic_cas(i32, 0, 0, 7)
        pl.atomic_exch(u64, 0, 18446744073709551615)


@pl.jit(arch="3510", auto_mutex=False)
def _result_kernel(i32: pl.Tensor[[16], pl.DT_INT32], out: pl.Tensor[[16], pl.DT_INT64]):
    with pl.section_vector():
        old = pl.atomic_add(i32, 0, 1)
        pl.setval(out, 0, old)


@pl.jit(arch="3510", auto_mutex=False)
def _runtime_value_kernel(i32: pl.Tensor[[16], pl.DT_INT32], value: pl.DT_INT32):
    with pl.section_vector():
        pl.atomic_add(i32, 0, value)


def test_every_rmw_name_lowers_to_one_family_with_its_kind():
    function = _parse(_rmw_kernel)
    calls = _op_calls(function, "block.atomic_rmw")

    kinds = [call.kwargs["kind"] for call in calls]
    assert kinds == [
        int(ir.AtomicRMWMode.ADD),
        int(ir.AtomicRMWMode.SUB),
        int(ir.AtomicRMWMode.EXCH),
        int(ir.AtomicRMWMode.MAX),
        int(ir.AtomicRMWMode.MIN),
        int(ir.AtomicRMWMode.AND),
        int(ir.AtomicRMWMode.OR),
        int(ir.AtomicRMWMode.XOR),
        int(ir.AtomicRMWMode.INC),
        int(ir.AtomicRMWMode.DEC),
        int(ir.AtomicRMWMode.ADD),
    ]
    assert _op_calls(function, "block.atomic_cas") == []


def test_cas_lowers_to_its_own_family_with_compare_and_value():
    call = _op_calls(_parse(_cas_kernel), "block.atomic_cas")[0]

    assert [arg.type.dtype for arg in call.args] == [
        ir.DataType.INT32,
        ir.DataType.INT64,
        ir.DataType.INT32,
        ir.DataType.INT32,
    ]


def test_numeric_literals_are_retyped_to_the_element_dtype():
    calls = _op_calls(_parse(_rmw_kernel), "block.atomic_rmw")

    assert calls[0].args[2].type.dtype == ir.DataType.INT32
    assert calls[6].args[2].type.dtype == ir.DataType.UINT32
    assert calls[10].args[2].type.dtype == ir.DataType.FP32
    # Offset stays a linear integer scalar, like getval/setval.
    assert calls[0].args[1].type.dtype == ir.DataType.INT64


def test_uint64_literal_preserves_the_unsigned_value():
    call = next(call for call in _op_calls(_parse(_cas_kernel), "block.atomic_rmw"))

    assert call.args[2].type.dtype == ir.DataType.UINT64
    assert call.args[2].value == -1


def test_runtime_value_keeps_its_scalar_type():
    call = _op_calls(_parse(_runtime_value_kernel), "block.atomic_rmw")[0]

    assert call.args[2].type.dtype == ir.DataType.INT32


def test_old_value_keeps_the_element_dtype():
    function = _parse(_result_kernel)
    call = _op_calls(function, "block.atomic_rmw")[0]
    assignment = next(
        stmt for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call) and stmt.value.name == "block.atomic_rmw"
    )

    assert call.args[0].type.dtype == ir.DataType.INT32
    assert assignment.targets[0].type.dtype == ir.DataType.INT32


def test_float_literal_on_an_integer_element_is_rejected():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            pl.atomic_add(i32, 0, 1.5)

    with pytest.raises(InvalidType, match="requires an integer value"):
        _parse(kernel)


def test_arithmetic_rejects_narrow_integer_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(u8: pl.Tensor[[16], pl.DT_UINT8]):
        with pl.section_vector():
            pl.atomic_add(u8, 0, 1)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_bitwise_rejects_floating_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(f32: pl.Tensor[[16], pl.DT_FP32]):
        with pl.section_vector():
            pl.atomic_and(f32, 0, 1.0)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_counter_rejects_signed_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            pl.atomic_inc(i32, 0, 1)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_cas_rejects_boolean_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(flag: pl.Tensor[[16], pl.DT_BOOL]):
        with pl.section_vector():
            pl.atomic_cas(flag, 0, 0, 1)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_runtime_value_dtype_must_match_the_element_dtype():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32], value: pl.DT_INT64):
        with pl.section_vector():
            pl.atomic_add(i32, 0, value)

    with pytest.raises(TypeError, match="dtype must match the element dtype"):
        _parse(kernel)


def test_runtime_linear_offset_and_keyword_arguments_are_accepted():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(t: pl.Tensor[[64], pl.DT_INT32], i: pl.DT_INT64, j: pl.DT_INT64):
        with pl.section_vector():
            pl.atomic_add(tensor=t, offset=i * 8 + j, value=1)

    call = _op_calls(_parse(kernel), "block.atomic_rmw")[0]

    assert not isinstance(call.args[1], ir.ConstInt)
    assert call.args[2].type.dtype == ir.DataType.INT32


def test_ub_tile_is_rejected():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel():
        tile_type = pl.TileType(shape=[1, 16], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.atomic_add(tile, 0, 1)

    with pytest.raises(InvalidType, match="Tensor"):
        _parse(kernel)


def test_atomic_is_rejected_inside_a_simt_function():
    @pl.vector_function(mode="simt", max_threads=1)
    def simt_body(i32):
        pl.atomic_add(i32, 0, 1)

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            simt_body[1](i32)

    with pytest.raises(NotSupported, match="not supported inside a SIMT function"):
        _parse(kernel)
