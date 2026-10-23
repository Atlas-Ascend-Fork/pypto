# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Assemble slot NeedAlloc: each PATH's constructAssembleSlotList (source of RUNTIME_SlotMarkNeedAlloc) lists
exactly the inner assemble targets that PATH defines; kernel params are never listed (host IR only, no NPU)."""

import re

import pytest

import pypto
from pypto import pypto_impl

from .test_common import run_root_function

BS = 32
HIDDEN = 64
TILE = 8
N_ROW = BS // TILE


def _marked_tensors(kernel):
    """{PATH name without kernel prefix: source variable names in its constructAssembleSlotList}."""
    prog = run_root_function(kernel, pypto.Tensor([BS, HIDDEN], pypto.DT_FP32, name="src"),
                             pypto.Tensor([BS, HIDDEN], pypto.DT_FP32, name="out"))
    # Slot aliases: SSA versions "buf_3" -> "buf"; drop anonymous temp-var names (all digits, or pass-made
    # "$N") and pass-made "OUTCAST_*" names.
    var_of = {slot: re.sub(r"_\d+$", "", name) for name, slot in pypto_impl.GetSlotInfo().items()
              if not re.match(r"\d|\$|OUTCAST_", name)}
    prefix = kernel.__name__ + "_"
    return {f.GetRawName()[len(prefix):]: {var_of[s] for s in f.GetConstructAssembleSlotList()}
            for f in prog.functions.values() if f.GetConstructAssembleSlotList()}


def _ir_full_cross_loop(src, out):
    pypto.set_vec_tile_shapes(TILE, TILE)
    buf = pypto.full([BS, HIDDEN], 1.0, pypto.DT_FP32)
    for i in pypto.loop(N_ROW, name="CV_FCL_W", idx_name="i"):
        v = pypto.view(src, [TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], buf)
    for j in pypto.loop(N_ROW, name="CV_FCL_R", idx_name="j"):
        w = pypto.view(buf, [TILE, HIDDEN], [j * TILE, 0])
        pypto.assemble(w + 1.0, [j * TILE, 0], out)


def _ir_full_outer_inner(src, out):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for i in pypto.loop(N_ROW, name="CV_FOI_I", idx_name="i"):
        buf = pypto.full([TILE, HIDDEN], 1.0, pypto.DT_FP32)
        for j in pypto.loop(HIDDEN // TILE, name="CV_FOI_J", idx_name="j"):
            v = pypto.view(src, [TILE, TILE], [i * TILE, j * TILE])
            pypto.assemble(pypto.mul(v, 4.0), [0, j * TILE], buf)
        pypto.assemble(buf, [i * TILE, 0], out)


def _ir_loop_carry(src, out):
    pypto.set_vec_tile_shapes(TILE, TILE)
    buf = pypto.full([BS, HIDDEN], 0.0, pypto.DT_FP32)
    for k in pypto.loop(3, name="CV_LC_ACC", idx_name="k"):
        buf[:] = pypto.add(buf, 1.0)
    for i in pypto.loop(N_ROW, name="CV_LC_W", idx_name="i"):
        v = pypto.view(src, [TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], buf)
    for j in pypto.loop(N_ROW, name="CV_LC_R", idx_name="j"):
        w = pypto.view(buf, [TILE, HIDDEN], [j * TILE, 0])
        pypto.assemble(w + 1.0, [j * TILE, 0], out)


def _ir_if_full_in_branch(src, out):
    pypto.set_vec_tile_shapes(TILE, TILE)
    for i in pypto.loop(N_ROW, name="CV_IFF", idx_name="i"):
        v = pypto.view(src, [TILE, HIDDEN], [i * TILE, 0])
        if pypto.cond(i < 2):
            f = pypto.full([TILE, HIDDEN], 1.0, pypto.DT_FP32)
            for j in pypto.loop(HIDDEN // TILE, name="CV_IFF_J", idx_name="j"):
                u = pypto.view(src, [TILE, TILE], [i * TILE, j * TILE])
                pypto.assemble(pypto.mul(u, 4.0), [0, j * TILE], f)
            pypto.assemble(f, [i * TILE, 0], out)
        else:
            pypto.assemble(pypto.add(v, 1.0), [i * TILE, 0], out)


def _ir_atomic_full(src, out):
    pypto.set_vec_tile_shapes(TILE, TILE)
    buf = pypto.full([TILE, HIDDEN], 0.0, pypto.DT_FP32)
    for i in pypto.loop(N_ROW, name="CV_ATF", idx_name="i"):
        v = pypto.view(src, [TILE, HIDDEN], [i * TILE, 0])
        pypto.atomic_add(pypto.mul(v, 4.0), [0, 0], buf)
    for j in pypto.loop(N_ROW, name="CV_ATF_R", idx_name="j"):
        pypto.assemble(buf + 1.0, [j * TILE, 0], out)


def _ir_tensor_assemble_only(src, out):
    pypto.set_vec_tile_shapes(TILE, TILE)
    buf = pypto.tensor([BS, HIDDEN], pypto.DT_FP32, "buf")
    for i in pypto.loop(N_ROW, name="CV_TAO_W", idx_name="i"):
        v = pypto.view(src, [TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], buf)
    for j in pypto.loop(N_ROW, name="CV_TAO_R", idx_name="j"):
        w = pypto.view(buf, [TILE, HIDDEN], [j * TILE, 0])
        pypto.assemble(w + 1.0, [j * TILE, 0], out)


def _ir_out_compute_then_assemble(src, out):
    pypto.set_vec_tile_shapes(TILE, TILE)
    out[:] = pypto.mul(pypto.view(src, [BS, HIDDEN], [0, 0]), 2.0)
    for i in pypto.loop(N_ROW, name="CV_OCA", idx_name="i"):
        v = pypto.view(src, [TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], out)


@pytest.mark.parametrize(
    "kernel, expected",
    [
        # buf defined by full at top level; the i/j loops only assemble into / read it.
        pytest.param(_ir_full_cross_loop, {"PATH0": {"buf"}}, id="full_cross_loop"),
        # buf redefined on every i iteration: marked on entry of the i-loop PATH that defines it.
        pytest.param(_ir_full_outer_inner, {"loop_idx_i_Unroll1_PATH0": {"buf"}}, id="full_outer_inner"),
        # buf written whole by both the top-level full and the k-loop add: marked by each writer.
        pytest.param(_ir_loop_carry, {"PATH0": {"buf"}, "loop_idx_k_Unroll1_PATH0": {"buf"}}, id="loop_carry"),
        # f defined only in the if branch: marked by that branch's PATH alone.
        pytest.param(_ir_if_full_in_branch, {"loop_idx_i_Unroll1_PATH0": {"f"}}, id="if_full_in_branch"),
        # atomic_add (ATOMIC_RMW) into buf counts as assemble.
        pytest.param(_ir_atomic_full, {"PATH0": {"buf"}}, id="atomic_full"),
        # buf only declared (no compute writer) still needs a buffer where it is declared.
        pytest.param(_ir_tensor_assemble_only, {"PATH0": {"buf"}}, id="tensor_assemble_only"),
        # out is a kernel param bound to external memory: never marked.
        pytest.param(_ir_out_compute_then_assemble, {}, id="out_compute_then_assemble"),
    ],
)
def test_construct_assemble_slot_list(kernel, expected):
    assert _marked_tensors(kernel) == expected
