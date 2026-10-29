# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Backend pipe registrations and parameter-dependent InsertSync decisions."""

import ast

from pypto_pro.ir.op.block_ops import make_mutex_tile_expr
from pypto_pro.runtime.pipeline._cross_core_scanner import (
    _block_op_pipe,
    get_expands_pipe,
    get_move_pipe,
    get_op_pipe,
    get_store_pipe,
)
import pytest

from pypto.pypto_impl import ir
from pypto.pypto_impl.backend import BackendCCE

M = ir.MemorySpace
P = ir.PipeType


def _insert_sync_pipes(name, memories):
    span = ir.Span.unknown()
    definitions = [
        make_mutex_tile_expr([16, 16], ir.DataType.FP32, memory, addr=i * 4096, mutex_ids=[i])
        for i, memory in enumerate(memories)
    ]
    tiles = [ir.Var(f"tile_{i}", definition.type, span) for i, definition in enumerate(definitions)]
    args = list(tiles)
    params = []
    zero = ir.ConstInt(0, ir.DataType.INT64, span)
    if name == "insert":
        args += [zero, zero]
    elif name == "store":
        tensor = ir.Var("tensor", ir.TensorType([16, 16], ir.DataType.FP32), span)
        params.append(tensor)
        args = [tensor, tiles[0], ir.MakeTuple([zero, zero], span)]
    elif name in ("expands", "full"):
        args.append(ir.ConstFloat(0.0, ir.DataType.FP32, span))
    call = ir.Call(ir.Op(f"block.{name}"), args, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(tile, definition, span) for tile, definition in zip(tiles, definitions, strict=True)]
        + [ir.EvalStmt(call, span)],
        span,
    )
    function = ir.Function("kernel", params, [], body, span)
    program = ir.Pass.insert_sync()(ir.Program([function], "pipes", span))
    sync = [
        stmt.expr
        for stmt in program.get_function("kernel").body.stmts
        if isinstance(stmt, ir.EvalStmt) and stmt.expr.name in ("system.mutex_lock_dyn", "system.mutex_unlock_dyn")
    ]
    assert len(sync) == 2 * len(tiles)
    return {ir.PipeType(call.kwargs["pipe"]) for call in sync}


@pytest.mark.parametrize("name", ["move", "insert", "store", "expands", "full"])
def test_argument_dependent_ops_have_no_fixed_pipe(name):
    assert BackendCCE.instance().get_op_pipe(f"block.{name}") is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("load", P.MTE2),
        ("load_tile", P.MTE2),
        ("matmul", P.M),
        ("matmul_mx", P.M),
        ("add", P.V),
        ("sum", P.V),
        ("select", P.V),
        ("getval", P.S),
        ("setval", P.S),
        ("make_tile", P.S),
        ("subview", P.S),
    ],
)
def test_fixed_pipes_and_dsl_aliases(name, expected):
    assert get_op_pipe(name) == expected


@pytest.mark.parametrize("name", ["move", "insert"])
@pytest.mark.parametrize(
    "src,dst,expected",
    [
        (M.Mat, M.Left, P.MTE1),
        (M.Mat, M.Right, P.MTE1),
        (M.Mat, M.Bias, P.MTE1),
        (M.Mat, M.ScaleLeft, P.MTE1),
        (M.Mat, M.ScaleRight, P.MTE1),
        (M.Mat, M.Vec, P.V),
        (M.Acc, M.Vec, P.FIX),
        (M.Acc, M.Mat, P.FIX),
        (M.Vec, M.Mat, P.MTE3),
        (M.Vec, M.Vec, P.V),
    ],
)
def test_move_pipe_matches_source_and_destination(name, src, dst, expected):
    assert get_move_pipe(src, dst) == expected
    assert _insert_sync_pipes(name, [dst, src]) == {expected}


@pytest.mark.parametrize("src,expected", [(M.Acc, P.FIX), (M.Vec, P.MTE3)])
def test_store_pipe_matches_source(src, expected):
    assert get_store_pipe(src) == expected
    assert _insert_sync_pipes("store", [src]) == {expected}


@pytest.mark.parametrize("name", ["expands", "full"])
@pytest.mark.parametrize("dst,expected", [(M.Mat, P.MTE2), (M.Vec, P.V)])
def test_broadcast_pipe_matches_destination_in_both_sync_paths(name, dst, expected):
    assert get_expands_pipe(dst) == expected
    assert _insert_sync_pipes(name, [dst]) == {expected}
    call = ast.parse(f"pl.{name}(tile, 0)").body[0].value
    assert _block_op_pipe(name, call, {"tile": "buffer"}, {"buffer": dst}) == expected.name
