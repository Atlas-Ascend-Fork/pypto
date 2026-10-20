# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""IR snapshots for a loop-local write followed by a read.

A buffer created outside the loop is carried in. The hidden function keeps the
loop-entry version as an extra incast, then reads the version written in this
iteration. A buffer created inside the loop is defined by that write and is
not an extra incast.
"""

from pathlib import Path

import pypto
from pypto import ir, pil

from .test_common import _ssa_verify, check_snapshot

_DIR = Path(__file__).parent
OUTSIDE_IR = _DIR / "test_loop_storage_incast_outside.pypto"
INSIDE_IR = _DIR / "test_loop_storage_incast_inside.pypto"


def _compile_root(func, *args):
    compiled = pil.compile(func, *args, create_new_logical_tensor=True)
    program = ir.IRBuilder().create_program([compiled], "main", ir.Span.unknown())
    verifier = ir.IRVerifier.create_default()
    _ssa_verify(verifier, program, "original")
    program = ir.Pass.aggressive_dce()(ir.Pass.canonicalize()(program))
    _ssa_verify(verifier, program, "dce")
    program = ir.Pass.create_root_functions()(program)
    return program


def _loop_hidden(program):
    hidden = [func for name, func in program.functions.items() if "hiddenfunc" in name and "loop_idx" in name]
    assert len(hidden) == 1, "expected one loop hidden function, got %s" % [f.name for f in hidden]
    return hidden[0]


def _param_bases(func):
    return [param.name.rsplit("_", 1)[0] for param in func.params]


def test_outside_loop_tensor_write_then_read():
    """buf is allocated before the loop, written, then read inside the loop."""

    def outside(src, out):
        pypto.set_vec_tile_shapes(16, 16)
        buf = pypto.tensor([16, 16], pypto.DT_FP32, "buf")
        for i in pypto.loop(2):
            tile = pypto.view(src, [16, 16], [i * 16, 0])
            pypto.assemble(tile, [0, 0], buf)
            read = pypto.view(buf, [16, 16], [0, 0])
            pypto.assemble(read, [i * 16, 0], out)

    src = pypto.Tensor([32, 16], pypto.DT_FP32, name="src")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    hidden = _loop_hidden(_compile_root(outside, src, out))

    assert _param_bases(hidden) == ["src", "buf", "buf", "out"]
    check_snapshot(hidden, OUTSIDE_IR)


def test_inside_loop_tensor_write_then_read():
    """buf is allocated inside the loop, written, then read there."""

    def inside(src, out):
        pypto.set_vec_tile_shapes(16, 16)
        for i in pypto.loop(2):
            buf = pypto.tensor([16, 16], pypto.DT_FP32, "buf")
            tile = pypto.view(src, [16, 16], [i * 16, 0])
            pypto.assemble(tile, [0, 0], buf)
            read = pypto.view(buf, [16, 16], [0, 0])
            pypto.assemble(read, [i * 16, 0], out)

    src = pypto.Tensor([32, 16], pypto.DT_FP32, name="src")
    out = pypto.Tensor([32, 16], pypto.DT_FP32, name="out")
    hidden = _loop_hidden(_compile_root(inside, src, out))

    assert _param_bases(hidden) == ["src", "out"]
    check_snapshot(hidden, INSIDE_IR)
