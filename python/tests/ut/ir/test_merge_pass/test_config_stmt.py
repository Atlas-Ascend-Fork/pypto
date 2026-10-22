# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

import pytest

import pypto
from pypto import ir, pil
from pypto.pil.pir import BuildContext, CollectContext, InsertPoint

from ..test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent


@pytest.mark.parametrize("stage", ["pil", "merge"])
def test_pass_config_statements(stage):
    """Keep the configuration-only branches from the pre-!606 MLA prolog pattern."""
    def kernel(x, y, out):
        pypto.set_vec_tile_shapes(16, 16)
        if pypto.SymbolicScalar("token_count") > 64:
            pypto.set_pass_options(cube_l1_reuse_setting={-1: 4, 3: 1})
            pypto.set_pass_options(cube_nbuffer_setting={-1: 4, 3: 3})
        else:
            pypto.set_pass_options(cube_l1_reuse_setting={-1: 4, 2: 1})
            pypto.set_pass_options(cube_nbuffer_setting={-1: 4, 2: 3})
        pypto.assemble(x + y, [0, 0], out)

    x = pypto.Tensor([16, 16], pypto.DT_FP32, "x")
    y = pypto.Tensor([16, 16], pypto.DT_FP32, "y")
    out = pypto.Tensor([16, 16], pypto.DT_FP32, "out")
    compile_ir = pil.compile if stage == "pil" else run_merge_pass
    func = compile_ir(kernel, x, y, out)
    check_snapshot(func, _GOLDEN_DIR / f"test_pass_config_statements_{stage}.pypto")


@pytest.mark.parametrize("stage", ["pil", "merge"])
def test_tile_config_statements(stage):
    """Keep tile and non-tile changes in order when a following loop sinks into an if."""
    def kernel(x, y, out):
        pypto.set_options(vec_tile_shapes=[8, 8], host_options={"compile_timeout": 1000})
        if pypto.SymbolicScalar("cond") > 0:
            pypto.set_vec_tile_shapes(16, 16)
            pypto.set_pass_options(cube_nbuffer_setting={-1: 2})
        else:
            pypto.set_vec_tile_shapes(8, 16)
            pypto.set_pass_options(cube_nbuffer_setting={-1: 4})
        value = x + y
        for _ in pypto.loop(2):
            pypto.set_vec_tile_shapes(4, 16)
            value = value + y
        pypto.set_vec_tile_shapes(16, 8)
        pypto.set_host_options(compile_timeout=2000)
        pypto.assemble(value, [0, 0], out)

    x = pypto.Tensor([16, 16], pypto.DT_FP32, "x")
    y = pypto.Tensor([16, 16], pypto.DT_FP32, "y")
    out = pypto.Tensor([16, 16], pypto.DT_FP32, "out")
    compile_ir = pil.compile if stage == "pil" else run_merge_pass
    func = compile_ir(kernel, x, y, out)
    check_snapshot(func, _GOLDEN_DIR / f"test_tile_config_statements_{stage}.pypto")


def test_conv1d_internal_tile_config_statements():
    """C++ temporary tiles and their restorations surround the corresponding reshapes."""
    def kernel(x, weight, out):
        pypto.set_vec_tile_shapes(1, 16, 64)
        pypto.set_conv_tile_shapes(
            pypto.pypto_impl.TileL1Info(
                tileHin=1, tileHout=1, tileWin=64, tileWout=64,
                tileCinFmap=8, tileCinWeight=8, tileN=32, tileBatch=1,
            ),
            pypto.pypto_impl.TileL0Info(tileH=1, tileW=64, tileK=24, tileN=32),
        )
        value = pypto.conv(x, weight, pypto.DT_FP32, [1], [1, 1], [1], extend_params={}, groups=1)
        pypto.assemble(value + 1, [0, 0, 0], out)

    x = pypto.Tensor([1, 16, 64], pypto.DT_FP32, "x")
    weight = pypto.Tensor([64, 16, 3], pypto.DT_FP32, "weight")
    out = pypto.Tensor([1, 64, 64], pypto.DT_FP32, "out")
    func = run_merge_pass(kernel, x, weight, out)
    check_snapshot(func, _GOLDEN_DIR / "test_conv1d_internal_tile_config_statements_merge.pypto")


def test_config_recording_context_cleanup():
    """Collection and failed nested builds must not leak configurations into the parent build."""
    span = ir.Span.unknown()
    body = ir.SeqStmts(span)
    collected = ir.SeqStmts(span)
    x = pypto.Tensor([16, 16], pypto.DT_FP32, "x")
    with pypto.function("__entry__", x):
        with BuildContext(span) as ctx, InsertPoint(body):
            pypto.set_vec_tile_shapes(16, 16)
            with CollectContext(span) as collect_ctx, InsertPoint(collected):
                pypto.set_vec_tile_shapes(8, 8)
                collect_ctx.emit_tensor_stmts()
            with pytest.raises(RuntimeError, match="abort recording"):
                with BuildContext(span), InsertPoint(ir.SeqStmts(span)):
                    pypto.set_vec_tile_shapes(4, 4)
                    raise RuntimeError("abort recording")
            pypto.set_host_options(compile_timeout=1000)
            ctx.emit_tensor_stmts()
    # Collection emits configurations into its own temporary body, not the parent body.
    func = ir.Function("recording_contexts", [], [], ir.SeqStmts([body, collected], span), span)
    check_snapshot(func, _GOLDEN_DIR / "test_config_recording_context_cleanup.pypto")


def test_config_ops_remain_tokenless_after_passes():
    def kernel(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        value = x + 1
        pypto.set_vec_tile_shapes(8, 16)
        pypto.assemble(value, [0, 0], out)

    x = pypto.Tensor([16, 16], pypto.DT_FP32, "x")
    out = pypto.Tensor([16, 16], pypto.DT_FP32, "out")
    func = pil.compile(kernel, x, out, create_new_logical_tensor=True)
    ops = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.TensorOpStmt)]
    configs = [stmt for stmt in ops if stmt.opcode == "OP_CONFIG"]
    assert len(configs) == 2
    add_index = next(i for i, op in enumerate(ops) if op.opcode == "ADDS")
    assert ops.index(configs[0]) < add_index < ops.index(configs[1])
    assert configs[0].attrs["vec_tile_shapes"] == [16, 16]
    assert configs[1].attrs["vec_tile_shapes"] == [8, 16]

    prog = ir.IRBuilder().create_program([func], "main", ir.Span.unknown())
    for transform in (ir.Pass.infer_token_pass(), ir.Pass.canonicalize(), ir.Pass.aggressive_dce(),
                      ir.Pass.remove_redundant_token_pass()):
        prog = transform(prog)
        configs = [stmt for stmt in prog.functions[func.name].body.stmts
                   if isinstance(stmt, ir.TensorOpStmt) and stmt.opcode == "OP_CONFIG"]
        assert len(configs) == 2
        for op in configs:
            assert not op.args and not op.result
            assert not op.tokens and not op.result_token
