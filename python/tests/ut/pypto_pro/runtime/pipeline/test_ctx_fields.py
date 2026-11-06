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

"""The type each ctx field is declared with.

A field takes the type its stage receives at the call. Cube and vector compile separate ctx
arrays, so a name bound to different types in the two sections declares the field
differently in each.
"""

import ast
import inspect
import textwrap

from pipeline_test_common import assert_targets_match_dumps, transform
import pypto_pro.language as pl

from pypto.pypto_impl import ir

TM, K = 64, 128
NT = 4
_CONFIG = pl.pipeline.PipelineConfig(preload=1)


def _ctx_declarations(result) -> list:
    """Each `pl.struct_array` declaration, as (enclosing section or None, its keywords)."""
    found = []
    for stmt in result.tree.body:
        section = None
        body = [stmt]
        if isinstance(stmt, ast.With):
            section = stmt.items[0].context_expr.func.attr
            body = stmt.body
        for inner in body:
            if (isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Call)
                    and getattr(inner.value.func, "attr", "") == "struct_array"):
                found.append((section, {k.arg: ast.literal_eval(k.value) for k in inner.value.keywords}))
    return found


def _ctx_struct_type(result, target) -> str:
    """The type the target's IR gives the first ctx slot."""
    text = str(result.parse_target(target)[0])
    return next(line for line in text.splitlines() if "_pl_ctx_arr_0_0 = struct.create" in line).split("%")[0].strip()


def _mat():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=[0x0000, 0x4000], mutex_ids=[2, 3])


def _ub():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x0000, 0x4000], mutex_ids=[0, 1])


def _gm(ws):
    return pl.make_tensor(ws, [TM, K], dtype=pl.DT_FP16, depth=2, fwd_ids=[0, 1], bwd_ids=[2, 3])


@pl.pipeline.stage
def scale(i, s, src, gm, ub):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.muls(t, t, s)
    pl.store(gm, t, [0, 0])


@pl.pipeline.stage
def take(i, s, gm, mat):
    m = mat.next()
    pl.load(m, gm, [0, 0])


@pl.pipeline.stage
def scale_by_every_field(i, s, r, cfg, src, gm, ub):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.muls(t, t, s)
    pl.muls(t, t, r)
    pl.muls(t, t, cfg.f)
    pl.store(gm, t, [0, 0])


@pl.pipeline.stage
def take_every_field(i, s, r, cfg, gm, mat):
    m = mat.next()
    pl.load(m, gm, [0, 0])


@pl.pipeline.stage
def scale_twice(i, s, r, src, gm, ub):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.muls(t, t, s)
    pl.muls(t, t, r)
    pl.store(gm, t, [0, 0])


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="ctx_typed_per_section")
def typed_per_section(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_UINT8]):
    gm = _gm(ws)
    ub = _ub()
    mat = _mat()
    with pl.section_vector():
        cfg = pl.struct("Cfg", f=0.5)
    with pl.section_cube():
        cfg = pl.struct("Cfg", f=1)
    for i in pl.range(0, NT):
        with pl.section_vector():
            s = i * 0.5
            r = i + 1
        with pl.section_cube():
            s = i + 1
            r = i * 0.5
        with pl.section_vector():
            scale_by_every_field(i, s, r, cfg, src, gm, ub)
        with pl.section_cube():
            take_every_field(i, s, r, cfg, gm, mat)


def test_fields_typed_differently_per_section_are_declared_per_section():
    result = transform(typed_per_section)
    declarations = _ctx_declarations(result)
    assert [section for section, _ in declarations] == ["section_cube", "section_vector"]
    by_section = dict(declarations)
    # A scalar either way round, and a struct field.
    assert {name: repr(by_section["section_cube"][name]) for name in ("s", "r", "f")} == {
        "s": "0", "r": "0.0", "f": "0"}
    assert {name: repr(by_section["section_vector"][name]) for name in ("s", "r", "f")} == {
        "s": "0.0", "r": "0", "f": "0.0"}
    # PipeCtx is (_pl_is_valid, f, i, r, s, _pl_task_id).
    assert _ctx_struct_type(result, ir.SectionKind.Cube) == "tuple<int64_t, int64_t, int64_t, float, int64_t, int64_t>"
    assert _ctx_struct_type(result, ir.SectionKind.Vector) == "tuple<int64_t, float, int64_t, int64_t, float, int64_t>"
    assert_targets_match_dumps(result, __file__, "typed_per_section")


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="ctx_typed_once")
def typed_once(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_UINT8]):
    gm = _gm(ws)
    ub = _ub()
    mat = _mat()
    r = 0
    r = 0.5
    for i in pl.range(0, NT):
        with pl.section_vector():
            s = i * 0.5
        with pl.section_cube():
            s = i * 0.25
        with pl.section_vector():
            scale_twice(i, s, r, src, gm, ub)
        with pl.section_cube():
            take(i, s, gm, mat)


def test_one_declaration_serves_both_sections_when_their_types_agree():
    result = transform(typed_once)
    declarations = _ctx_declarations(result)
    assert len(declarations) == 1 and declarations[0][0] is None
    # `s` has one type in both sections. `r` is rebound before the loop and only vector
    # hands it over: the type at the stage call wins, and cube declares it the same.
    assert repr(declarations[0][1]["s"]) == "0.0"
    assert repr(declarations[0][1]["r"]) == "0.0"
    for target in (ir.SectionKind.Cube, ir.SectionKind.Vector):
        assert _ctx_struct_type(result, target) == "tuple<int64_t, int64_t, float, float, int64_t>"


def _ident(v):
    return v


def _halve(x):
    """Halve ``x``.

    The call below sits at the same position in this function's own source as `s` in
    position_shared_with_helper's call to scale: both ASTs count from their own first line.
    Its argument is a float, while that `s` is an integer.

    The test asserts the two positions coincide, so keep this docstring's length.
    """
    halfway = _ident(x * 0.5)
    return halfway


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="ctx_position_shared_with_helper")
def position_shared_with_helper(src: pl.Tensor[[NT * TM, K], pl.DT_FP16], ws: pl.Ptr[pl.DT_UINT8]):
    gm = _gm(ws)
    ub = _ub()
    mat = _mat()
    _halve(3)
    for i in pl.range(0, NT):
        s = i + 1
        with pl.section_vector():
            scale(i, s, src, gm, ub)
        with pl.section_cube():
            take(i, s, gm, mat)


def _argument_positions(tree: ast.AST, callee: str) -> list:
    """(lineno, col_offset) of each argument of the calls to ``callee`` in ``tree``."""
    return [
        (arg.lineno, arg.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == callee
        for arg in node.args
    ]


def test_an_argument_inside_a_helper_does_not_type_a_stage_argument_at_the_same_position():
    # The case only tests something while the two positions really coincide.
    helper = ast.parse(textwrap.dedent(inspect.getsource(_halve)))
    kernel = position_shared_with_helper.to_kernel_def()._func_def
    assert _argument_positions(helper, "_ident")[0] == _argument_positions(kernel, "scale")[1]
    result = transform(position_shared_with_helper)
    declarations = _ctx_declarations(result)
    assert len(declarations) == 1 and declarations[0][0] is None
    assert repr(declarations[0][1]["s"]) == "0"
    for target in (ir.SectionKind.Cube, ir.SectionKind.Vector):
        result.parse_target(target)
