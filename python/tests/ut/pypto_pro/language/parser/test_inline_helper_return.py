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

from pypto_pro import ir
from pypto_pro._errors import InvalidOperation, NameNotFound, NotSupported, PyptoProError
import pypto_pro.language as pl
import pytest


def _walk_statements(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _walk_statements(child)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
        yield from _walk_statements(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from _walk_statements(stmt.then_body)
        if stmt.else_body is not None:
            yield from _walk_statements(stmt.else_body)


def test_inline_helper_multiple_value_returns_use_one_wrapper():
    def choose(value):
        if value > 0:
            return value + 1
        return value - 1

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        _test_result = choose(value)

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller = caller_program.get_function(caller.__name__)

    statements = list(_walk_statements(caller.body))
    helper_loops = [stmt for stmt in statements if isinstance(stmt, ir.WhileStmt)]
    return_assigns = [
        stmt
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0_return_val")
    ]

    assert len(helper_loops) == 1
    assert len(return_assigns) == 2
    assert len({stmt.var.name for stmt in return_assigns}) == 2
    return_iter_arg = next(
        arg for arg in helper_loops[0].iter_args if arg.iterVar.name.startswith("__inline_0_return_val")
    )
    assert isinstance(return_iter_arg.initValue.type, ir.NoneType)


def test_inline_helper_top_level_return_flattens_lowered_wrapper():
    def choose(value):
        adjusted = value + 1
        return adjusted
        unreachable = missing_name + 1  # noqa: F821, F841

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        selected = choose(value)
        _test_result = selected + 1

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller_ir = caller_program.get_function(caller.__name__)

    statements = list(_walk_statements(caller_ir.body))
    assert not any(isinstance(stmt, ir.WhileStmt) for stmt in statements)
    assert any(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0_return_val")
        for stmt in statements
    )
    assert any(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_test_result")
        for stmt in statements
    )


def test_inline_helper_tuple_return_after_nested_loop_flattens_wrapper():
    def resolve(value):
        result = value
        for i in pl.range(1):
            if i > 0:
                break
            result = result + i
        return result, value

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        first, second = resolve(value)
        _test_result = first + second

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller_ir = caller_program.get_function(caller.__name__)

    statements = list(_walk_statements(caller_ir.body))
    assert not any(isinstance(stmt, ir.WhileStmt) for stmt in statements)
    assert any(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0_return_val")
        for stmt in statements
    )


def test_inline_helper_return_inside_loop_adds_propagation_guard():
    def choose(value):
        index = 0
        while index < value:
            if index >= 2:
                return index
            index = index + 1
        return value

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        _test_result = choose(value)

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller = caller_program.get_function(caller.__name__)

    wrapper = next(stmt for stmt in caller.body.stmts if isinstance(stmt, ir.WhileStmt))
    nested_loop_index = next(i for i, stmt in enumerate(wrapper.body.stmts) if isinstance(stmt, ir.WhileStmt))
    guard = wrapper.body.stmts[nested_loop_index + 1]

    assert isinstance(guard, ir.IfStmt)
    assert isinstance(guard.condition, ir.Var)
    assert guard.condition.name.startswith("__inline_0_returned")
    assert any(isinstance(stmt, ir.BreakStmt) for stmt in _walk_statements(guard.then_body))


def test_inline_helper_dynamic_loop_return_does_not_merge_initial_empty_value():
    def choose(value):
        index = 0
        while index < value:
            if index >= 2:
                return value
            index = index + 1
        return value

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        selected = choose(value)
        _test_result = selected + 1

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller_ir = caller_program.get_function(caller.__name__)
    result_assign = next(
        stmt
        for stmt in _walk_statements(caller_ir.body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_test_result")
    )
    assert isinstance(result_assign.var.type, ir.ScalarType)


def test_inline_helper_nested_if_returns_preserve_fallthrough_retval_state():
    def choose(branch, left, right, value):
        if branch:
            if left:
                return value
        else:
            if right:
                return value
        return value

    @pl.jit(auto_mutex=False)
    def caller(
        branch: pl.DT_BOOL,
        left: pl.DT_BOOL,
        right: pl.DT_BOOL,
        value: pl.DT_INT64,
    ):
        selected = choose(branch, left, right, value)
        _test_result = selected + 1

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller_ir = caller_program.get_function(caller.__name__)
    result_assign = next(
        stmt
        for stmt in _walk_statements(caller_ir.body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("_test_result")
    )
    retval_results = [
        var
        for stmt in _walk_statements(caller_ir.body)
        if isinstance(stmt, ir.IfStmt)
        for var in stmt.return_vars
        if var.name.startswith("__inline_0_return_val")
    ]
    assert isinstance(result_assign.var.type, ir.ScalarType)
    assert any(isinstance(var.type, ir.NoneType) for var in retval_results)
    assert all(not isinstance(var.type, ir.UnknownType) for var in retval_results)


def test_inline_helper_dynamic_loop_uses_one_int64_return_type():
    def choose(value):
        index = 0
        while index < value:
            if index >= 2:
                return index
            index = index + 1
        return value

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        selected = choose(value)
        _test_result = selected + 1

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller_ir = caller_program.get_function(caller.__name__)
    selected = next(
        stmt.var
        for stmt in _walk_statements(caller_ir.body)
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("selected")
    )
    assert isinstance(selected.type, ir.ScalarType)
    assert selected.type.dtype == ir.DataType.INT64


def test_inline_helper_loop_guard_folds_constant_false():
    def inner(value):
        if value > 0:
            return value
        return value + 1

    def outer(value):
        result = value
        for _ in pl.range(1):
            result = inner(value)
        # The second return is what puts `outer` on the lowering path at all. It has to come
        # after the loop -- an early return ahead of the loop would leave `returned` non-constant
        # there and the guard would survive -- and one statement clear of it, so that an `if`
        # sitting right behind the loop can only be the guard itself.
        doubled = result + result
        if doubled > 0:
            return doubled
        return result

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        _test_result = outer(value)

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller = caller_program.get_function(caller.__name__)

    outer_wrapper = next(stmt for stmt in caller.body.stmts if isinstance(stmt, ir.WhileStmt))
    outer_for_index = next(i for i, stmt in enumerate(outer_wrapper.body.stmts) if isinstance(stmt, ir.ForStmt))
    statement_after_for = outer_wrapper.body.stmts[outer_for_index + 1]

    assert not isinstance(statement_after_for, ir.IfStmt)


def test_inline_helper_bare_returns_share_return_state():
    def stop_early(value):
        if value > 0:
            return
        return None

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        stop_early(value)
        _test_result = value

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller = caller_program.get_function(caller.__name__)

    statements = list(_walk_statements(caller.body))
    assert sum(isinstance(stmt, ir.WhileStmt) for stmt in statements) == 1
    return_assigns = [
        stmt
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0_return_val")
    ]
    assert len(return_assigns) == 2
    assert all(isinstance(stmt.var.type, ir.NoneType) for stmt in return_assigns)


def test_inline_helper_mixed_bare_and_value_returns_share_return_val():
    def choose(value):
        if value > 0:
            return value
        return

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        _test_result = choose(value)

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller = caller_program.get_function(caller.__name__)

    statements = list(_walk_statements(caller.body))
    return_assigns = [
        stmt
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0_return_val")
    ]
    assert len(return_assigns) == 2
    assert any(isinstance(stmt.var.type, ir.NoneType) for stmt in return_assigns)
    assert any(isinstance(stmt.var.type, ir.ScalarType) for stmt in return_assigns)


def test_inline_helper_without_return_uses_default_return_val():
    def update(value):
        result = value
        for _ in pl.range(1):
            result = result + 1

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        _test_result = update(value)

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller = caller_program.get_function(caller.__name__)

    statements = list(_walk_statements(caller.body))
    assert not any(isinstance(stmt, ir.WhileStmt) for stmt in statements)
    return_assigns = [
        stmt
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0_return_val")
    ]
    assert len(return_assigns) == 1
    assert isinstance(return_assigns[0].var.type, ir.NoneType)


def test_inline_helper_target_section_does_not_finalize_wrapper_loop():
    def update(value):
        with pl.section_vector():
            _updated = value + 1

    @pl.jit(auto_mutex=False)
    def caller(value: pl.DT_INT64):
        update(value)
        _test_result = value

    caller_program, matched = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller_ir = caller_program.get_function(caller.__name__)

    assert matched
    assert any(
        isinstance(stmt, ir.AssignStmt) and stmt.var.name.startswith("__inline_0__updated")
        for stmt in _walk_statements(caller_ir.body)
    )


def test_inline_helper_value_return_fallthrough_merges_conservative_none():
    def choose(value):
        if value > 0:
            return value

    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):

        @pl.jit(auto_mutex=False)
        def caller(value: pl.DT_INT64):
            selected = choose(value)
            _test_result = selected + 1

        caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_kernel_uses_runtime_tile_valid_shape_ir():
    @pl.jit(auto_mutex=False)
    def caller(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        _kernel_rows = tile.valid_shape[0]
        _kernel_cols = tile.valid_shape[1]

    caller_program, _ = caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    caller = caller_program.get_function(caller.__name__)

    function_ir = str(caller)
    assert function_ir.count("block.tile_valid_shape") == 2


@pytest.mark.parametrize("index", [True, 0.0, 2, -3])
def test_tile_valid_shape_rejects_invalid_index(index):
    with pytest.raises((PyptoProError, PyptoProError)):
        @pl.jit(auto_mutex=False)
        def invalid_valid_shape(_jit_entry: pl.DT_INT64):
            tile_type = pl.TileType(shape=[16, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0)
            value = tile.valid_shape[index]  # noqa: F841

        invalid_valid_shape.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_vector_function_rejects_return_value():
    @pl.vector_function
    def invalid_vf():
        return 1

    with pytest.raises(NotSupported, match="returning values is not supported"):

        @pl.jit(auto_mutex=False)
        def caller(value: pl.DT_INT64):
            invalid_vf()
            _test_result = value

        caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_vector_function_rejects_non_vector_helper_call():
    def helper(value):
        return value

    @pl.vector_function
    def invalid_vf(value):
        helper(value)

    with pytest.raises(InvalidOperation, match="cannot call non-vector inline function 'helper'"):

        @pl.jit(auto_mutex=False)
        def caller(value: pl.DT_INT64):
            invalid_vf(value)
            _test_result = value

        caller.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
