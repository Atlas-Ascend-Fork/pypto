# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""DCE tests for LogicalTensor versions that share memory."""

from pathlib import Path

import pypto
from pypto import ir
from pypto.pil.compile_pipeline import _build_default_pipeline, compile_new_ir

from ..test_common import check_snapshot

_GOLDEN_DIR = Path(__file__).parent


def _run_dce(func, *args):
    pipeline = [("dce", lambda p:ir.Pass.aggressive_dce()(ir.Pass.canonicalize()(p)))]
    return compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)


def _run_token_dce(func, *args):
    pipeline = []
    for name, transform in _build_default_pipeline():
        pipeline.append((name, transform))
        if name == "first_canonicalize_dce":
            break
    return compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)


def test_dead_view_shared_token():
    def foo(out):
        pypto.set_vec_tile_shapes(16, 16)
        pypto.assemble(pypto.full([32, 16], 0.0, pypto.DT_FP32), [0, 0], out)
        unused = pypto.view(out, [16, 16], [0, 0], valid_shape=[16, 16])  # noqa: F841
        used = pypto.view(out, [16, 16], [16, 0], valid_shape=[16, 16])
        pypto.assemble(pypto.add(used, used), [16, 0], out)

    tensor = pypto.Tensor((32, 16), pypto.DT_FP32, name="out")
    func = _run_token_dce(foo, tensor)
    check_snapshot(func, _GOLDEN_DIR / "test_dce_dead_view_shared_token.pypto")


def test_dead_view_loop_writes():
    def foo(out):
        pypto.set_vec_tile_shapes(16, 16)
        for _ in pypto.loop(2):
            pypto.assemble(pypto.full([16, 16], 0.0, pypto.DT_FP32), [0, 0], out)
            unused = pypto.view(out, [16, 16], [0, 0], valid_shape=[16, 16])  # noqa: F841
            pypto.assemble(pypto.full([16, 16], 1.0, pypto.DT_FP32), [0, 0], out)

    tensor = pypto.Tensor((32, 16), pypto.DT_FP32, name="out")
    func = _run_token_dce(foo, tensor)
    check_snapshot(func, _GOLDEN_DIR / "test_dce_dead_view_loop_writes.pypto")


def test_dce_keeps_writes_to_consumed_raw_tensor():
    """DCE keeps every LogicalTensor version that writes a consumed RawTensor."""

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        aux = pypto.tensor([32, 16], pypto.DT_FP32, name="aux")
        pypto.assemble(pypto.full([8, 16], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([8, 16], 2.0, pypto.DT_FP32), [16, 0], aux)
        out = pypto.add(x, aux)  # noqa: F841

    x = pypto.Tensor((32, 16), pypto.DT_FP32, name="x")
    out = pypto.Tensor((32, 16), pypto.DT_FP32, name="out")
    func = _run_dce(foo, x, out)
    assembles = [
        op
        for op in func.body
        if isinstance(op, ir.TensorOpStmt) and op.opcode == "ASSEMBLE"
    ]

    assert len(assembles) == 2
    assert assembles[0].result[0].name != assembles[1].result[0].name


def test_dce_keeps_write_to_inplace_reshape_memory():
    """DCE keeps a write consumed through an inplace-reshape memory alias."""

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        aux = pypto.tensor([32, 16], pypto.DT_FP32, name="aux")
        alias = pypto.reshape(aux, [16, 32], inplace=True)
        src = pypto.full([8, 32], 1.0, pypto.DT_FP32)
        pypto.assemble(src, [0, 0], alias)
        out = pypto.add(x, aux)  # noqa: F841

    x = pypto.Tensor((32, 16), pypto.DT_FP32, name="x")
    out = pypto.Tensor((32, 16), pypto.DT_FP32, name="out")
    func = _run_dce(foo, x, out)
    assembles = [
        op
        for op in func.body
        if isinstance(op, ir.TensorOpStmt) and op.opcode == "ASSEMBLE"
    ]

    assert len(assembles) == 1
