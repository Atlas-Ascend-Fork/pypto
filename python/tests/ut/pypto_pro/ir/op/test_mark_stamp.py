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
"""IR structure and printing contracts for explicit profiling stamps."""

import inspect
from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl
import pytest
from test_common import assert_ir_matches_dump


def _assert_stamp(stmt, pipe, stamp_id):
    assert isinstance(stmt, ir.AssignStmt)
    assert not stmt.targets
    call = stmt.value
    assert isinstance(call, ir.Call)
    assert call.name == "system.mark_stamp"
    assert not call.args
    assert call.kwargs == {"pipe": pipe.value, "stamp_id": stamp_id}
    assert isinstance(call.type, ir.NoneType)
    return call


@pytest.mark.parametrize("target,pipe", [
    (ir.SectionKind.Cube, pl.PipeType.S),
    (ir.SectionKind.Cube, pl.PipeType.MTE2),
    (ir.SectionKind.Cube, pl.PipeType.MTE1),
    (ir.SectionKind.Cube, pl.PipeType.M),
    (ir.SectionKind.Cube, pl.PipeType.FIX),
    (ir.SectionKind.Cube, pl.PipeType.MTE3),
    (ir.SectionKind.Vector, pl.PipeType.S),
    (ir.SectionKind.Vector, pl.PipeType.MTE2),
    (ir.SectionKind.Vector, pl.PipeType.V),
    (ir.SectionKind.Vector, pl.PipeType.MTE3),
])
@pytest.mark.parametrize("stamp_id", [0, 4095])
def test_mark_stamp_ir_attributes_and_source_span(target, pipe, stamp_id):
    @pl.jit(auto_mutex=False)
    def kernel():
        pl.mark_stamp(pipe, stamp_id)

    program, _ = kernel.to_kernel_def().parse_target_program(target)
    function = program.get_function("kernel")
    statements = [stmt for stmt in function.body.stmts if isinstance(stmt, ir.AssignStmt)]
    assert len(statements) == 1
    call = _assert_stamp(statements[0], pipe, stamp_id)
    lines, start = inspect.getsourcelines(kernel._func)
    expected_line = start + next(i for i, line in enumerate(lines) if "pl.mark_stamp(" in line)
    assert Path(call.span.filename) == Path(__file__)
    assert call.span.begin_line == expected_line


@pytest.mark.parametrize("target,pipes,golden", [
    (ir.SectionKind.Cube, [pl.PipeType.S, pl.PipeType.M, pl.PipeType.FIX], "cube.pypto"),
    (ir.SectionKind.Vector, [pl.PipeType.S, pl.PipeType.V, pl.PipeType.MTE3], "vector.pypto"),
])
def test_mark_stamp_mixed_target_ir(target, pipes, golden):
    @pl.jit(auto_mutex=False)
    def kernel():
        with pl.section_cube():
            pl.mark_stamp(pl.PipeType.S, 0)
            pl.mark_stamp(pipe=pl.PipeType.M, id=1 + 2)
            pl.mark_stamp(pl.PipeType.FIX, id=4095)
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.S, 0)
            pl.mark_stamp(pipe=pl.PipeType.V, id=1 + 2)
            pl.mark_stamp(pl.PipeType.MTE3, id=4095)

    program, matched = kernel.to_kernel_def().parse_target_program(target)
    assert matched
    function = program.get_function("kernel")
    statements = [stmt for stmt in function.body.stmts if isinstance(stmt, ir.AssignStmt)]
    assert len(statements) == 3
    for stmt, pipe, stamp_id in zip(statements, pipes, [0, 3, 4095]):
        _assert_stamp(stmt, pipe, stamp_id)
    assert_ir_matches_dump(function, Path(__file__).with_suffix("") / golden)


def test_mark_stamp_stays_inside_loop_and_branch_ir():
    @pl.jit(auto_mutex=False)
    def kernel(count: pl.DT_INT32):
        for i in pl.range(0, count):
            pl.mark_stamp(pl.PipeType.V, 9)
            if i == 0:
                pl.mark_stamp(pl.PipeType.S, 10)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("kernel")
    loop = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.ForStmt))
    loop_statements = [stmt for stmt in loop.body.stmts if isinstance(stmt, ir.AssignStmt) and not stmt.targets]
    assert len(loop_statements) == 1
    _assert_stamp(loop_statements[0], pl.PipeType.V, 9)
    branch = next(stmt for stmt in loop.body.stmts if isinstance(stmt, ir.IfStmt))
    _assert_stamp(branch.then_body.stmts[0], pl.PipeType.S, 10)
    assert_ir_matches_dump(function, Path(__file__).with_suffix("") / "control_flow.pypto")
