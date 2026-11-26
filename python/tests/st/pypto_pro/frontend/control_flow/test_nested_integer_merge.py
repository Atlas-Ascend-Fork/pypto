# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Nested if/for integer merge STs.

Covers the four structural combinations of the if merge promotion and the
loop-carried widening:

  1. if nested in if (both levels merge mixed integer branch dtypes)
  2. for nested in for (wide and narrow loop-carried slots at both levels)
  3. if wrapping a for (the if merges a loop result with a scalar constant)
  4. for wrapping an if (an inner if merge feeding a loop-carried slot)
"""

import os

from pypto_pro import ir
import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


def _walk_stmts(stmt):
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _walk_stmts(child)
    elif isinstance(stmt, ir.SectionStmt):
        yield from _walk_stmts(stmt.body)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
        yield stmt
        yield from _walk_stmts(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield stmt
        yield from _walk_stmts(stmt.then_body)
        if stmt.else_body is not None:
            yield from _walk_stmts(stmt.else_body)
    else:
        yield stmt


def _device():
    device = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device)
    return device


@pl.jit
def nested_if_kernel(
    flag0: pl.DT_INT32,
    flag1: pl.DT_INT32,
    s16: pl.DT_INT16,
    s32: pl.DT_INT32,
    out: pl.Tensor[[1, 3], pl.DT_INT64],
):
    with pl.section_vector():
        if flag0:
            if flag1:
                v0 = s16
            else:
                v0 = s32
            value = v0
        else:
            value = pl.const(101, pl.DT_INT64)
        pl.setval(out, 0, value)

        value2 = pl.const(200, pl.DT_INT64)
        if flag0:
            if flag1:
                value2 = pl.const(201, pl.DT_INT16)
            else:
                value2 = s32
        pl.setval(out, 1, value2)

        value3 = pl.const(0, pl.DT_INT32)
        if flag0:
            if flag1:
                value3 = pl.const(5, pl.DT_UINT16)
            else:
                value3 = pl.const(7, pl.DT_INT16)
        pl.setval(out, 2, value3)


@pl.jit
def nested_for_kernel(
    s16: pl.DT_INT16,
    u16: pl.DT_UINT16,
    out: pl.Tensor[[1, 3], pl.DT_INT64],
):
    with pl.section_vector():
        total = pl.const(0, pl.DT_INT64)
        for i in pl.range(2):
            inner = pl.const(0, pl.DT_INT64)
            for j in pl.range(2):
                inner = s16
            total = inner
        pl.setval(out, 0, total)

        outer32 = pl.const(0, pl.DT_INT32)
        for i in pl.range(2):
            inner32 = pl.const(0, pl.DT_INT32)
            for j in pl.range(2):
                inner32 = u16
            outer32 = inner32
        pl.setval(out, 1, outer32)

        broke = pl.const(0, pl.DT_INT64)
        for i in pl.range(2):
            for j in pl.range(2):
                broke = u16
                if j == 1:
                    break
            if i == 1:
                break
        pl.setval(out, 2, broke)


@pl.jit
def if_wraps_for_kernel(
    flag: pl.DT_INT32,
    s16: pl.DT_INT16,
    out: pl.Tensor[[1, 2], pl.DT_INT64],
):
    with pl.section_vector():
        total = pl.const(0, pl.DT_INT64)
        if flag:
            for _ in pl.range(3):
                total = s16
        else:
            total = pl.const(77, pl.DT_INT32)
        pl.setval(out, 0, total)

        count = pl.const(0, pl.DT_INT64)
        if flag == 0:
            for _ in pl.range(2):
                count = pl.const(5, pl.DT_UINT16)
        else:
            count = pl.const(6, pl.DT_INT64)
        pl.setval(out, 1, count)


@pl.jit
def for_wraps_if_kernel(
    flag: pl.DT_INT32,
    s16: pl.DT_INT16,
    s32: pl.DT_INT32,
    out: pl.Tensor[[1, 2], pl.DT_INT64],
):
    with pl.section_vector():
        total = pl.const(0, pl.DT_INT64)
        for _ in pl.range(2):
            if flag:
                inner = s16
            else:
                inner = s32
            total = inner
        pl.setval(out, 0, total)

        acc = pl.const(0, pl.DT_INT32)
        for i in pl.range(2):
            if i == 0:
                acc = pl.const(11, pl.DT_INT16)
            else:
                acc = pl.const(13, pl.DT_UINT16)
        pl.setval(out, 1, acc)


def _parse(kernel):
    program, matched = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert matched
    return program.get_function(kernel.__name__)


def _if_scalar_dtypes(func):
    return [
        var.type.dtype
        for stmt in _walk_stmts(func.body)
        if isinstance(stmt, ir.IfStmt)
        for var in stmt.return_vars
        if isinstance(var.type, ir.ScalarType)
    ]


def _loop_carry(for_stmt, name):
    for arg, ret in zip(for_stmt.iter_args, for_stmt.return_vars):
        var_name = arg.iterVar.name
        if var_name == name or var_name.startswith(name + "_"):
            return arg.iterVar.type.dtype, ret.type.dtype
    raise AssertionError(
        f"loop does not carry {name!r}; carried names: "
        f"{[arg.iterVar.name for arg in for_stmt.iter_args]}"
    )


@pytest.mark.soc("950", "960")
def test_nested_if_integer_merge_ir_types():
    func = _parse(nested_if_kernel)
    assert _if_scalar_dtypes(func) == [
        ir.DataType.INT64,
        ir.DataType.INT32,
        ir.DataType.INT64,
        ir.DataType.INT32,
        ir.DataType.INT32,
        ir.DataType.INT32,
    ]


@pytest.mark.soc("950", "960")
def test_nested_if_integer_merge_runs_on_device():
    device = _device()
    s16 = -29
    s32 = -37
    expected_by_flags = {
        (0, 0): [101, 200, 0],
        (1, 1): [s16, 201, 5],
        (1, 0): [s32, s32, 7],
    }
    for (flag0, flag1), expected in expected_by_flags.items():
        out = torch.full((1, 3), -1, dtype=torch.int64, device=f"npu:{device}")
        nested_if_kernel[None, 1](flag0, flag1, s16, s32, out)
        torch.npu.synchronize()
        assert out.cpu().tolist() == [expected]


@pytest.mark.soc("950", "960")
def test_nested_for_integer_merge_ir_types():
    func = _parse(nested_for_kernel)
    loops = [stmt for stmt in _walk_stmts(func.body) if isinstance(stmt, ir.ForStmt)]
    loop_types = [
        _loop_carry(loops[0], "total"),
        _loop_carry(loops[1], "inner"),
        _loop_carry(loops[2], "outer32"),
        _loop_carry(loops[3], "inner32"),
        _loop_carry(loops[4], "broke"),
        _loop_carry(loops[5], "broke"),
    ]
    assert loop_types == [
        (ir.DataType.INT64, ir.DataType.INT64),
        (ir.DataType.INT64, ir.DataType.INT64),
        (ir.DataType.INT32, ir.DataType.INT32),
        (ir.DataType.INT32, ir.DataType.INT32),
        (ir.DataType.INT64, ir.DataType.INT64),
        (ir.DataType.INT64, ir.DataType.INT64),
    ]


@pytest.mark.soc("950", "960")
def test_nested_for_integer_merge_runs_on_device():
    device = _device()
    out = torch.full((1, 3), -1, dtype=torch.int64, device=f"npu:{device}")
    nested_for_kernel[None, 1](-29, 65530, out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [[-29, 65530, 65530]]


@pytest.mark.soc("950", "960")
def test_if_wraps_for_integer_merge_ir_types():
    func = _parse(if_wraps_for_kernel)
    loops = [stmt for stmt in _walk_stmts(func.body) if isinstance(stmt, ir.ForStmt)]
    assert _loop_carry(loops[0], "total") == (ir.DataType.INT64, ir.DataType.INT64)
    assert _loop_carry(loops[1], "count") == (ir.DataType.INT64, ir.DataType.INT64)
    assert _if_scalar_dtypes(func) == [ir.DataType.INT64, ir.DataType.INT64]


@pytest.mark.soc("950", "960")
def test_if_wraps_for_integer_merge_runs_on_device():
    device = _device()
    expected_by_flag = {1: [-29, 6], 0: [77, 5]}
    for flag, expected in expected_by_flag.items():
        out = torch.full((1, 2), -1, dtype=torch.int64, device=f"npu:{device}")
        if_wraps_for_kernel[None, 1](flag, -29, out)
        torch.npu.synchronize()
        assert out.cpu().tolist() == [expected]


@pytest.mark.soc("950", "960")
def test_for_wraps_if_integer_merge_ir_types():
    func = _parse(for_wraps_if_kernel)
    loops = [stmt for stmt in _walk_stmts(func.body) if isinstance(stmt, ir.ForStmt)]
    assert _loop_carry(loops[0], "total") == (ir.DataType.INT64, ir.DataType.INT64)
    assert _loop_carry(loops[1], "acc") == (ir.DataType.INT32, ir.DataType.INT32)
    assert _if_scalar_dtypes(func) == [ir.DataType.INT32, ir.DataType.INT32]


@pytest.mark.soc("950", "960")
def test_for_wraps_if_integer_merge_runs_on_device():
    device = _device()
    expected_by_flag = {1: [-29, 13], 0: [-37, 13]}
    for flag, expected in expected_by_flag.items():
        out = torch.full((1, 2), -1, dtype=torch.int64, device=f"npu:{device}")
        for_wraps_if_kernel[None, 1](flag, -29, -37, out)
        torch.npu.synchronize()
        assert out.cpu().tolist() == [expected]
