# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto
from pypto import ir

from .test_common import run_root_function

BS = 32
HIDDEN = 64
TILE = 8
TILE_N = 16


def _tensor(*shape):
    return pypto.Tensor(list(shape), pypto.DT_FP32)


def _multi_iter_no_overlap(prog):
    """Function name → MultiIterNoOverlap dump string (only funcs that have the Mark)."""
    out = {}
    for name, func in sorted(prog.functions.items()):
        dumped = func.DumpAttrs().get("MultiIterNoOverlap", "")
        if dumped:
            out[name] = dumped
    return out


def _run_overlap_pass(kernel, *args):
    prog = run_root_function(kernel, *args, create_new_logical_tensor=True)
    prog = ir.Pass.infer_multi_iter_overlap()(prog)
    return _multi_iter_no_overlap(prog)


def _kernel_assemble_disjoint(src, out):
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_DJ", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = src.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], out)


def _kernel_assemble_overlap(src, out):
    stride = TILE // 2
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_OV", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = src.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * stride, 0], out)


def _kernel_nested_disjoint(src, out):
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN + TILE_N - 1) // TILE_N
    for i in pypto.loop(n_row, name="ST_NDJ_I", idx_name="i"):
        for j in pypto.loop(n_col, name="ST_NDJ_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = src.view([TILE, TILE_N], [i * TILE, j * TILE_N])
            pypto.assemble(pypto.mul(v, 4.0), [i * TILE, j * TILE_N], out)


def _kernel_nested_overlap(src, out):
    stride_c = TILE_N // 2
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN - TILE_N) // stride_c + 1
    for i in pypto.loop(n_row, name="ST_NOV_I", idx_name="i"):
        for j in pypto.loop(n_col, name="ST_NOV_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = src.view([TILE, TILE_N], [i * TILE, j * stride_c])
            pypto.assemble(pypto.mul(v, 4.0), [i * TILE, j * stride_c], out)


def _kernel_fixed_offset(src, out):
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_FIX", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = src.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [0, 0], out)


def _kernel_same_idx_diff_step(src, out_ov, out_dj):
    stride = TILE // 2
    n_ov = (BS + TILE - 1) // TILE
    for i in pypto.loop(0, n_ov, 1, name="ST_SIB_OV", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = src.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * stride, 0], out_ov)

    n_dj = 2 * ((BS + TILE - 1) // TILE)
    for i in pypto.loop(0, n_dj, 2, name="ST_SIB_DJ", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = src.view([TILE, HIDDEN], [i * stride, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * stride, 0], out_dj)


def _kernel_nested_inner_immediate_offset(src, out):
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN + TILE_N - 1) // TILE_N
    for i in pypto.loop(n_row, name="ST_IMM_I", idx_name="i"):
        for j in pypto.loop(n_col, name="ST_IMM_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = src.view([TILE, TILE_N], [i * TILE, j * TILE_N])
            pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], out)


def _kernel_nested_outer_overlap_inner_disjoint(src, out):
    stride_r = TILE // 2
    n_row = (BS - TILE) // stride_r + 1
    n_col = (HIDDEN + TILE_N - 1) // TILE_N
    for i in pypto.loop(n_row, name="ST_OOV_I", idx_name="i"):
        for j in pypto.loop(n_col, name="ST_OOV_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = src.view([TILE, TILE_N], [i * stride_r, j * TILE_N])
            pypto.assemble(pypto.mul(v, 4.0), [i * stride_r, j * TILE_N], out)


def _kernel_tensor_create_outer_inner_overlap(src, out):
    stride_c = TILE // 2
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN - TILE) // stride_c + 1
    for i in pypto.loop(n_row, name="ST_TCO_I", idx_name="i"):
        buf = pypto.tensor([TILE, HIDDEN], pypto.DT_FP32, "buf")
        for j in pypto.loop(n_col, name="ST_TCO_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = src.view([TILE, TILE], [i * TILE, j * stride_c])
            pypto.assemble(pypto.mul(v, 4.0), [0, j * stride_c], buf)
        pypto.assemble(buf, [i * TILE, 0], out)


def _kernel_full_create_disjoint(src, out):
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_FULL_I", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        f = pypto.full([TILE, HIDDEN], 1.0, pypto.DT_FP32)
        v = src.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [0, 0], f)
        out[:] = f + 1.0


def _kernel_tensor_create_outer_inner_disjoint(src, out):
    n_row = (BS + TILE - 1) // TILE
    n_col = (HIDDEN + TILE - 1) // TILE
    for i in pypto.loop(n_row, name="ST_TCD_I", idx_name="i"):
        buf = pypto.tensor([TILE, HIDDEN], pypto.DT_FP32, "buf")
        for j in pypto.loop(n_col, name="ST_TCD_J", idx_name="j"):
            pypto.set_vec_tile_shapes(TILE, TILE)
            v = src.view([TILE, TILE], [i * TILE, j * TILE])
            pypto.assemble(pypto.mul(v, 4.0), [0, j * TILE], buf)
        pypto.assemble(buf, [i * TILE, 0], out)


def _kernel_local_tensor_create_disjoint(src, out):
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_LTC", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        buf = pypto.tensor([BS, HIDDEN], pypto.DT_FP32, "buf")
        v = src.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], buf)
        w = buf.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(w + 1.0, [i * TILE, 0], out)


def _kernel_local_full_create_disjoint(src, out):
    n = (BS + TILE - 1) // TILE
    for i in pypto.loop(n, name="ST_LFC", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        f = pypto.full([BS, HIDDEN], 0.0, pypto.DT_FP32)
        v = src.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * TILE, 0], f)
        w = f.view([TILE, HIDDEN], [i * TILE, 0])
        pypto.assemble(w + 1.0, [i * TILE, 0], out)


def _kernel_outer_reshape_outside(src, out):
    rows = TILE // 2
    n = BS // TILE
    buf = pypto.tensor([BS, HIDDEN], pypto.DT_FP32, "buf")
    alias = pypto.reshape(buf, [BS // 2, HIDDEN * 2], inplace=True)
    for i in pypto.loop(n, name="ST_RS1", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        v = src.view([rows, HIDDEN * 2], [i * rows, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * rows, 0], alias)
    for j in pypto.loop(n, name="ST_RS1_OUT", idx_name="j"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        w = buf.view([TILE, HIDDEN], [j * TILE, 0])
        pypto.assemble(w + 1.0, [j * TILE, 0], out)


def _kernel_outer_reshape_inside(src, out):
    rows = TILE // 2
    n = BS // TILE
    buf = pypto.tensor([BS, HIDDEN], pypto.DT_FP32, "buf")
    for i in pypto.loop(n, name="ST_RS2", idx_name="i"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        alias = pypto.reshape(buf, [BS // 2, HIDDEN * 2], inplace=True)
        v = src.view([rows, HIDDEN * 2], [i * rows, 0])
        pypto.assemble(pypto.mul(v, 4.0), [i * rows, 0], alias)
    for j in pypto.loop(n, name="ST_RS2_OUT", idx_name="j"):
        pypto.set_vec_tile_shapes(TILE, TILE)
        w = buf.view([TILE, HIDDEN], [j * TILE, 0])
        pypto.assemble(w + 1.0, [j * TILE, 0], out)


# ---------- expected MultiIterNoOverlap only (name → dump) ----------

_EXPECTED_ASSEMBLE_DISJOINT = {
    "_kernel_assemble_disjoint_loop_idx_i_Unroll1_PATH0_hiddenfunc_4": "[6]",
}
_EXPECTED_ASSEMBLE_OVERLAP = {}
_EXPECTED_NESTED_DISJOINT = {
    "_kernel_nested_disjoint_loop_idx_j_Unroll1_PATH0_hiddenfunc_4": "[6]",
}
_EXPECTED_NESTED_OVERLAP = {}
_EXPECTED_FIXED_OFFSET = {}
_EXPECTED_SAME_IDX_DIFF_STEP = {
    "_kernel_same_idx_diff_step_loop_idx_i_Unroll1_PATH1_hiddenfunc_6": "[16]",
}
_EXPECTED_NESTED_INNER_IMMEDIATE = {}
_EXPECTED_NESTED_OUTER_OV_INNER_DJ = {}
_EXPECTED_TENSOR_CREATE_OUTER_INNER_OV = {
    "_kernel_tensor_create_outer_inner_overlap_loop_idx_i_Unroll1_PATH1_hiddenfunc_8": "[14]",
}
_EXPECTED_FULL_CREATE_DISJOINT = {}
_EXPECTED_TENSOR_CREATE_OUTER_INNER_DJ = {
    "_kernel_tensor_create_outer_inner_disjoint_loop_idx_i_Unroll1_PATH1_hiddenfunc_8": "[14]",
    "_kernel_tensor_create_outer_inner_disjoint_loop_idx_j_Unroll1_PATH0_hiddenfunc_6": "[7]",
}
_EXPECTED_LOCAL_TENSOR_CREATE = {
    "_kernel_local_tensor_create_disjoint_loop_idx_i_Unroll1_PATH0_hiddenfunc_4": "[9]",
}
_EXPECTED_LOCAL_FULL_CREATE = {
    "_kernel_local_full_create_disjoint_loop_idx_i_Unroll1_PATH0_hiddenfunc_4": "[9]",
}
_EXPECTED_OUTER_RESHAPE_OUTSIDE = {
    "_kernel_outer_reshape_outside_loop_idx_i_Unroll1_PATH0_hiddenfunc_6": "[18]",
    "_kernel_outer_reshape_outside_loop_idx_j_Unroll1_PATH0_hiddenfunc_8": "[25]",
}
_EXPECTED_OUTER_RESHAPE_INSIDE = {
    "_kernel_outer_reshape_inside_loop_idx_j_Unroll1_PATH0_hiddenfunc_8": "[18]",
}


# ---------- tests ----------


def test_assemble_disjoint_marked():
    """Single loop, assemble offset i*TILE abutting → Mark."""
    got = _run_overlap_pass(_kernel_assemble_disjoint, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_ASSEMBLE_DISJOINT


def test_assemble_overlap_unmarked():
    """Single loop, assemble offset i*(TILE//2) half-window → no Mark."""
    got = _run_overlap_pass(_kernel_assemble_overlap, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_ASSEMBLE_OVERLAP


def test_nested_disjoint_marked():
    """Nested i/j, offsets i*TILE / j*TILE_N both abutting → Mark."""
    got = _run_overlap_pass(_kernel_nested_disjoint, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_NESTED_DISJOINT


def test_nested_overlap_unmarked():
    """Nested i/j, column stride j*(TILE_N//2) overlaps → no Mark."""
    got = _run_overlap_pass(_kernel_nested_overlap, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_NESTED_OVERLAP


def test_fixed_offset_unmarked():
    """Every iter assembles to fixed [0,0] → same region overlaps → no Mark."""
    got = _run_overlap_pass(_kernel_fixed_offset, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_FIXED_OFFSET


def test_same_idx_diff_step_attrs():
    """Sibling loops share idx_name i: ov half-window unmarked; dj step=2 abutting Marked."""
    got = _run_overlap_pass(
        _kernel_same_idx_diff_step, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN), _tensor(BS, HIDDEN)
    )
    assert got == _EXPECTED_SAME_IDX_DIFF_STEP


def test_nested_inner_immediate_offset_unmarked():
    """Inner j not in assemble offset (window invariant in j) → no Mark."""
    got = _run_overlap_pass(_kernel_nested_inner_immediate_offset, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_NESTED_INNER_IMMEDIATE


def test_nested_outer_overlap_inner_disjoint_unmarked():
    """Outer i half-window overlaps; inner j abutting — any layer fail → no Mark."""
    got = _run_overlap_pass(
        _kernel_nested_outer_overlap_inner_disjoint, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN)
    )
    assert got == _EXPECTED_NESTED_OUTER_OV_INNER_DJ


def test_tensor_create_outer_inner_overlap_attrs():
    """Outer tensor(buf); inner half-window into buf unmarked; buf→out i*TILE Marked."""
    got = _run_overlap_pass(
        _kernel_tensor_create_outer_inner_overlap, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN)
    )
    assert got == _EXPECTED_TENSOR_CREATE_OUTER_INNER_OV


def test_full_create_disjoint_unmarked():
    """full(f) each iter + assemble into f (local, skipped); out[:] = f + 1 rewrites out at [0,0] → no Mark."""
    got = _run_overlap_pass(_kernel_full_create_disjoint, _tensor(BS, HIDDEN), _tensor(TILE, HIDDEN))
    assert got == _EXPECTED_FULL_CREATE_DISJOINT


def test_tensor_create_outer_inner_disjoint_attrs():
    """Outer tensor(buf); inner j*TILE and buf→out i*TILE both abutting → both Marked."""
    got = _run_overlap_pass(
        _kernel_tensor_create_outer_inner_disjoint, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN)
    )
    assert got == _EXPECTED_TENSOR_CREATE_OUTER_INNER_DJ


def test_local_tensor_create_not_outcast_skipped():
    """Loop-local tensor(buf) written i*TILE and read in same hidden → not outcast → only out Marked."""
    got = _run_overlap_pass(_kernel_local_tensor_create_disjoint, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_LOCAL_TENSOR_CREATE


def test_local_full_create_not_outcast_skipped():
    """Loop-local full(f) written i*TILE and read in same hidden → not outcast → only out Marked."""
    got = _run_overlap_pass(_kernel_local_full_create_disjoint, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_LOCAL_FULL_CREATE


def test_outer_reshape_outside_loop_marked():
    """Outer tensor(buf) + inplace reshape outside loop; alias is hidden outcast, i*rows abutting → Marked."""
    got = _run_overlap_pass(_kernel_outer_reshape_outside, _tensor(BS // 2, HIDDEN * 2), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_OUTER_RESHAPE_OUTSIDE


def test_outer_reshape_inside_loop_not_outcast_skipped():
    """Outer tensor(buf) + inplace reshape inside loop; alias raw is not a hidden outcast → skipped."""
    got = _run_overlap_pass(_kernel_outer_reshape_inside, _tensor(BS // 2, HIDDEN * 2), _tensor(BS, HIDDEN))
    assert got == _EXPECTED_OUTER_RESHAPE_INSIDE


def test_multiround_disjoint_then_overlap_clears_marks():
    """Same process 5×: disjoint Marks then overlap must stay empty (Infer ClearMarks)."""
    for _ in range(5):
        marked = _run_overlap_pass(_kernel_assemble_disjoint, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
        assert marked == _EXPECTED_ASSEMBLE_DISJOINT
        unmarked = _run_overlap_pass(_kernel_assemble_overlap, _tensor(BS, HIDDEN), _tensor(BS, HIDDEN))
        assert unmarked == {}
