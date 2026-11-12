# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression coverage for assignment result lists and discarded expressions."""

import pytest

from pypto.pypto_impl import ir


def _program(stmts):
    span = ir.Span.unknown()
    body = ir.SeqStmts([*stmts, ir.ReturnStmt(span)], span)
    func = ir.Function("test", [], [], body, span)
    return ir.IRBuilder().create_program([func], "main", span)


def _verify(program):
    diagnostics = ir.IRVerifier.create_default().verify(program)
    assert not diagnostics, ir.IRVerifier.generate_report(diagnostics)


def test_assignment_target_lists_preserve_definitions():
    span = ir.Span.unknown()
    scalar = ir.ScalarType(ir.DataType.INT64)
    x, y = [ir.Var(name, scalar, span) for name in ("x", "y")]
    value = ir.MakeTuple([ir.ConstInt(1, ir.DataType.INT64, span), ir.ConstInt(2, ir.DataType.INT64, span)], span)
    assign = ir.AssignStmt([x, y], value, span)
    use = ir.AssignStmt([], ir.Call("consume", [x, y], span), span)
    program = _program([assign, use])
    _verify(program)
    assert assign.targets == [x, y]
    other_x, other_y = [ir.Var(name, scalar, span) for name in ("a", "b")]
    other = ir.AssignStmt([other_x, other_y], value, span)
    assert ir.structural_equal(assign, other, enable_auto_mapping=True)
    assert ir.structural_hash(assign, True) == ir.structural_hash(other, True)
    assert not ir.structural_equal(assign, ir.AssignStmt([other_x], value, span))

    duplicate = ir.AssignStmt([x, x], value, span)
    assert ir.IRVerifier.create_default().verify(_program([duplicate]))


def test_assignment_target_lists_preserve_ssa_order():
    span = ir.Span.unknown()
    tensor_type = ir.LogicalTensorType()
    targets = [ir.Var(name, tensor_type, span) for name in ("x", "y")]
    value = ir.Call("produce", [], ir.TupleType([tensor_type, tensor_type]), span)
    assign = ir.AssignStmt(targets, value, span)
    use = ir.AssignStmt([], ir.Call("consume", targets, span), span)
    _verify(_program([assign, use]))
    diagnostics = ir.IRVerifier.create_default().verify(_program([use, assign]))
    assert len(diagnostics) == 2
    assert "used before its dominating definition" in ir.IRVerifier.generate_report(diagnostics)

    self_use = ir.AssignStmt(targets, ir.MakeTuple(targets, span), span)
    diagnostics = ir.IRVerifier.create_default().verify(_program([self_use]))
    assert len(diagnostics) == 2
    assert "used before its dominating definition" in ir.IRVerifier.generate_report(diagnostics)


@pytest.mark.parametrize("discard_call", [False, True])
def test_discarded_expression_keeps_its_dependencies(discard_call):
    span = ir.Span.unknown()
    scalar = ir.ScalarType(ir.DataType.INT64)
    source = ir.Var("source", scalar, span)
    dead = ir.Var("dead", scalar, span)
    value = ir.ConstInt(7, ir.DataType.INT64, span)
    dependency = ir.AssignStmt([source], value, span)
    unused = ir.AssignStmt([dead], value, span)
    expr = ir.Call("consume", [source], span) if discard_call else source
    discard = ir.AssignStmt([], expr, span)
    program = _program([dependency, unused, discard])
    _verify(program)
    result = ir.Pass.aggressive_dce()(program)
    _verify(result)
    stmts = result.functions["test"].body.stmts
    assert len(stmts) == 3
    assert stmts[0].targets == [source]
    assert not stmts[1].targets
    assert ir.structural_equal(stmts[1].value, expr, enable_auto_mapping=False)


@pytest.mark.parametrize("used_index", [None, 0, 1])
def test_dce_preserves_whole_target_list_if_any_result_is_live(used_index):
    span = ir.Span.unknown()
    scalar = ir.ScalarType(ir.DataType.INT64)
    targets = [ir.Var(name, scalar, span) for name in ("x", "y")]
    value = ir.MakeTuple([ir.ConstInt(n, ir.DataType.INT64, span) for n in (1, 2)], span)
    stmts = [ir.AssignStmt(targets, value, span)]
    if used_index is not None:
        stmts.append(ir.AssignStmt([], targets[used_index], span))
    program = _program(stmts)
    _verify(program)
    result = ir.Pass.aggressive_dce()(program)
    _verify(result)
    body = result.functions["test"].body.stmts
    if used_index is None:
        assert len(body) == 1
        assert isinstance(body[0], ir.ReturnStmt)
    else:
        assert len(body) == 3
        assert body[0].targets == targets


@pytest.mark.parametrize("use_loop_result", [False, True])
def test_canonicalize_preserves_discarded_expression_carry_behavior(use_loop_result):
    span = ir.Span.unknown()
    scalar = ir.ScalarType(ir.DataType.INT64)
    zero = ir.ConstInt(0, ir.DataType.INT64, span)
    one = ir.ConstInt(1, ir.DataType.INT64, span)
    stop = ir.ConstInt(4, ir.DataType.INT64, span)
    index = ir.Var("i", scalar, span)
    carried = ir.IterArg("carried", scalar, zero, span)
    updated = ir.Var("updated", scalar, span)
    result = ir.Var("result", scalar, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt([], ir.Call("consume", [carried.iterVar], span), span),
            ir.AssignStmt([updated], ir.Add(carried.iterVar, one, ir.DataType.INT64, span), span),
            ir.YieldStmt([updated], span),
        ],
        span,
    )
    loop = ir.ForStmt(index, zero, stop, one, [carried], body, [result], span)
    stmts = [loop]
    if use_loop_result:
        stmts.append(ir.AssignStmt([], result, span))
    program = _program(stmts)
    _verify(program)
    result = ir.Pass.canonicalize()(program)
    _verify(result)
    new_loop = result.functions["test"].body.stmts[0]
    # Preserve the pre-migration EvalStmt behavior: discarded expressions in
    # the body do not create carry-liveness roots. An external result use does.
    assert len(new_loop.iter_args) == int(use_loop_result)
    assert len(new_loop.return_vars) == int(use_loop_result)
    assert len(new_loop.body.stmts[-1].value) == int(use_loop_result)
    assert not new_loop.body.stmts[0].targets
