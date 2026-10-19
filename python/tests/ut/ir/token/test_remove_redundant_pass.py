# -*- coding: utf-8 -*-
# ruff: noqa: E501
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for removing redundant token dependencies after MergeStmtsIntoIf."""

from pathlib import Path

import pypto
from pypto import ir
from pypto.pil.compile_pipeline import compile_new_ir

from ..test_common import check_snapshot

_GOLDEN_DIR = Path(__file__).parent / "test_remove_redundant_pass_data"


class _Snapshot:
    def __init__(self, name, text):
        self.name = name
        self.text = text

    def __str__(self):
        return self.text


def _run_passes(func, *args):
    before_remove = []

    def capture_before_remove(program):
        compiled_func = program.functions[func.__name__]
        before_remove.append(_Snapshot(compiled_func.name, str(compiled_func)))
        return program

    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    pipeline = [
        ("infer_token_pass", ir.Pass.infer_token_pass()),
        ("first_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        ("capture_before_remove_redundant_token_pass", capture_before_remove),
        ("remove_redundant_token_pass", ir.Pass.remove_redundant_token_pass()),
    ]
    after_remove = compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)
    return before_remove[0], after_remove


def _tensor(shape, name):
    return pypto.Tensor(shape=shape, dtype=pypto.DT_FP32, name=name)




BEFORE_IR_0 = _GOLDEN_DIR / "BEFORE_IR_0.pypto"


IR_0 = _GOLDEN_DIR / "IR_0.pypto"


def test_remove_ssa_dependency():
    def foo(x, out):
        value = x + 1.0
        pypto.assemble(value, [0, 0], out)

    before_remove, after_remove = _run_passes(foo, _tensor((16, 16), "x"), _tensor((16, 16), "out"))
    check_snapshot(before_remove, BEFORE_IR_0)
    check_snapshot(after_remove, IR_0)


BEFORE_IR_1 = _GOLDEN_DIR / "BEFORE_IR_1.pypto"


IR_1 = _GOLDEN_DIR / "IR_1.pypto"


def test_forward_disjoint_write_dependency():
    def foo(src, aux, result):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [16, 0], aux)
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_1)
    check_snapshot(after_remove, IR_1)


BEFORE_IR_2 = _GOLDEN_DIR / "BEFORE_IR_2.pypto"


IR_2 = _GOLDEN_DIR / "IR_2.pypto"


def test_keep_overlapping_write_dependency():
    def foo(src, aux, result):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [4, 0], aux)
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_2)
    check_snapshot(after_remove, IR_2)


BEFORE_IR_3 = _GOLDEN_DIR / "BEFORE_IR_3.pypto"


IR_3 = _GOLDEN_DIR / "IR_3.pypto"


def test_keep_dynamic_write_dependency():
    def foo(src, aux, shape):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [shape.shape[0], 0], aux)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((-1,), "shape"),
    )
    check_snapshot(before_remove, BEFORE_IR_3)
    check_snapshot(after_remove, IR_3)


BEFORE_IR_4 = _GOLDEN_DIR / "BEFORE_IR_4.pypto"


IR_4 = _GOLDEN_DIR / "IR_4.pypto"


def test_assemble_scenario_1_parallel_non_overlapping_writes():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_4)
    check_snapshot(after_remove, IR_4)


BEFORE_IR_5 = _GOLDEN_DIR / "BEFORE_IR_5.pypto"


IR_5 = _GOLDEN_DIR / "IR_5.pypto"


def test_assemble_scenario_2_war_between_non_overlapping_writes():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_5)
    check_snapshot(after_remove, IR_5)


BEFORE_IR_6 = _GOLDEN_DIR / "BEFORE_IR_6.pypto"


IR_6 = _GOLDEN_DIR / "IR_6.pypto"


def test_assemble_scenario_3_overlapping_waw():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([129, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [64, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_6)
    check_snapshot(after_remove, IR_6)


BEFORE_IR_7 = _GOLDEN_DIR / "BEFORE_IR_7.pypto"


IR_7 = _GOLDEN_DIR / "IR_7.pypto"


def test_assemble_scenario_4_all_versions_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value1 = aux + 1.0
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        value2 = aux + 2.0
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value1 + value2 + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_7)
    check_snapshot(after_remove, IR_7)


BEFORE_IR_8 = _GOLDEN_DIR / "BEFORE_IR_8.pypto"


IR_8 = _GOLDEN_DIR / "IR_8.pypto"


def test_assemble_scenario_5_no_intermediate_version_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(a + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_8)
    check_snapshot(after_remove, IR_8)


BEFORE_IR_9 = _GOLDEN_DIR / "BEFORE_IR_9.pypto"


IR_9 = _GOLDEN_DIR / "IR_9.pypto"


def test_assemble_scenario_6_middle_version_not_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_9)
    check_snapshot(after_remove, IR_9)


BEFORE_IR_10 = _GOLDEN_DIR / "BEFORE_IR_10.pypto"


IR_10 = _GOLDEN_DIR / "IR_10.pypto"


def test_assemble_scenario_7_first_version_not_consumed():
    def foo(a, aux, out):
        pypto.assemble(pypto.full([32, 128], 1.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(pypto.full([32, 128], 2.0, pypto.DT_FP32), [96, 0], aux)
        value = aux + 1.0
        pypto.assemble(pypto.full([64, 128], 3.0, pypto.DT_FP32), [16, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_10)
    check_snapshot(after_remove, IR_10)


BEFORE_IR_11 = _GOLDEN_DIR / "BEFORE_IR_11.pypto"


IR_11 = _GOLDEN_DIR / "IR_11.pypto"


def test_assemble_scenario_8_separate_writes_after_shared_read():
    def foo(a, aux, out):
        value = a + aux
        pypto.assemble(pypto.full([64, 128], 1.0, pypto.DT_FP32), [0, 0], a)
        pypto.assemble(pypto.full([64, 128], 2.0, pypto.DT_FP32), [0, 0], aux)
        pypto.assemble(value, [0, 0], out)

    shape = (129, 128)
    before_remove, after_remove = _run_passes(foo, _tensor(shape, "a"), _tensor(shape, "aux"), _tensor(shape, "out"))
    check_snapshot(before_remove, BEFORE_IR_11)
    check_snapshot(after_remove, IR_11)


BEFORE_IR_12 = _GOLDEN_DIR / "BEFORE_IR_12.pypto"


IR_12 = _GOLDEN_DIR / "IR_12.pypto"


def test_rewire_four_op_token_chain():
    def foo(src, aux, result):
        # A: write aux[0:8].
        pypto.assemble(src, [0, 0], aux)
        # B: write aux[16:24].
        pypto.assemble(src, [16, 0], aux)
        # C: write aux[0:8].
        pypto.assemble(src, [0, 0], aux)
        # D: read aux.
        result = aux + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_12)
    check_snapshot(after_remove, IR_12)


BEFORE_IR_13 = _GOLDEN_DIR / "BEFORE_IR_13.pypto"


IR_13 = _GOLDEN_DIR / "IR_13.pypto"


def test_rewire_middle_edge_to_diamond():
    def foo(src_a, src_b, src_c, aux, result):
        # A overlaps both B and C; B and C are disjoint.
        pypto.assemble(src_a, [0, 0], aux)   # A: write aux[0:32].
        pypto.assemble(src_b, [0, 0], aux)   # B: write aux[0:8].
        pypto.assemble(src_c, [24, 0], aux)  # C: write aux[24:32].
        result = aux + 1.0  # D: read the whole tensor.  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((32, 16), "src_a"),
        _tensor((8, 16), "src_b"),
        _tensor((8, 16), "src_c"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_13)
    check_snapshot(after_remove, IR_13)


BEFORE_IR_14 = _GOLDEN_DIR / "BEFORE_IR_14.pypto"


IR_14 = _GOLDEN_DIR / "IR_14.pypto"


def test_remove_disjoint_dependency_before_loop_first_stage():
    def foo(src, aux, out):
        pypto.assemble(src, [0, 0], aux)
        pypto.assemble(src, [16, 0], aux)
        for _ in pypto.loop(2):
            value = aux + 1.0
        pypto.assemble(value, [0, 0], out)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "out"),
    )
    check_snapshot(before_remove, BEFORE_IR_14)
    check_snapshot(after_remove, IR_14)


BEFORE_IR_15 = _GOLDEN_DIR / "BEFORE_IR_15.pypto"


IR_15 = _GOLDEN_DIR / "IR_15.pypto"


def test_remove_disjoint_dependency_inside_loop_first_stage():
    def foo(src, aux, out):
        for _ in pypto.loop(2):
            pypto.assemble(src, [0, 0], aux)
            pypto.assemble(src, [16, 0], aux)
        pypto.assemble(aux + 1.0, [0, 0], out)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "out"),
    )
    check_snapshot(before_remove, BEFORE_IR_15)
    check_snapshot(after_remove, IR_15)


BEFORE_IR_16 = _GOLDEN_DIR / "BEFORE_IR_16.pypto"


IR_16 = _GOLDEN_DIR / "IR_16.pypto"


def test_remove_disjoint_write_dependency_for_input_tensor():
    def foo(src, kj_assemble, result):
        block_size = 128
        valid_rows = src.shape[0].min(block_size)
        for i in range(4):
            block = pypto.view(src, [block_size, 128], [i * block_size, 0],
                               valid_shape=[valid_rows, 128])
            pypto.assemble(block, [i * block_size, 0], kj_assemble)
        result = kj_assemble + 1.0  # noqa: F841

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((-1, 128), "src"),
        _tensor((512, 128), "kj_assemble"),
        _tensor((512, 128), "result"),
    )
    check_snapshot(before_remove, BEFORE_IR_16)
    check_snapshot(after_remove, IR_16)


BEFORE_IR_17 = _GOLDEN_DIR / "BEFORE_IR_17.pypto"


IR_17 = _GOLDEN_DIR / "IR_17.pypto"


def test_shared_view_keeps_overlap():
    """A shared read token must account for every view that produces it."""
    def foo_two_views(io, src):
        pypto.view(io, [8, 16], [0, 0])
        pypto.view(io, [8, 16], [16, 0])
        pypto.assemble(src, [16, 0], io)
    before_remove, after_remove = _run_passes(
        foo_two_views,
        _tensor((32, 16), "io"),
        _tensor((8, 16), "src"),
    )
    check_snapshot(before_remove, BEFORE_IR_17)
    check_snapshot(after_remove, IR_17)


BEFORE_IR_18 = _GOLDEN_DIR / "BEFORE_IR_18.pypto"


IR_18 = _GOLDEN_DIR / "IR_18.pypto"


def test_atomic_add_prior_read_keeps_consumed_version():
    """An unread atomic version merges forward; a consumed version stays before the next atomic write."""
    def foo(src, aux, out):
        pypto.atomic_add(src, [0, 0], aux)
        pypto.atomic_add(src, [16, 0], aux)
        value = aux + 1.0
        pypto.atomic_add(src, [0, 0], aux)
        pypto.assemble(value + aux, [0, 0], out)

    before_remove, after_remove = _run_passes(
        foo,
        _tensor((8, 16), "src"),
        _tensor((32, 16), "aux"),
        _tensor((32, 16), "out"),
    )
    check_snapshot(before_remove, BEFORE_IR_18)
    check_snapshot(after_remove, IR_18)
