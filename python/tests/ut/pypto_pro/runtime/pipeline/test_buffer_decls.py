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

"""What a pipeline kernel's buffer declarations amount to: where they may sit, and the
slot count and addresses they resolve to.

A buffer is declared in the kernel body, or returned there by a function, bound to a
kernel-body variable, and outside every loop.
"""

import linecache
import re
from types import SimpleNamespace

from pipeline_test_common import assert_targets_match_dumps, transform
from pypto_pro._errors import InvalidOperation
import pypto_pro.language as pl
from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts
from pypto_pro.runtime.pipeline._cross_core_scanner import buffer_addr_ranges, buffer_slot_counts
from pypto_pro.runtime.pipeline._sync_graph import _slot_count
import pytest

TM, K = 64, 128
NT = 4
_CONFIG = pl.pipeline.PipelineConfig(preload=1)
_UNBOUND = "is not bound to a variable in the kernel body"
_IN_LOOP = "is declared inside a loop"


def _pointed_at(error: InvalidOperation) -> str:
    """The source line an error points at."""
    line = error.span.get("line") or error.span.get("begin_line")
    return linecache.getline(error.span["filename"], line).strip()


def _ub():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x0000, 0x4000], mutex_ids=[0, 1])


def _left():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x0000, 0x4000], mutex_ids=[2, 3])


def _shared():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x4000], mutex_ids=[12, 13], fwd_ids=[0, 1], bwd_ids=[2, 3])


def _vec_group(base, first_id):
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[base, base + 0x4000], mutex_ids=[first_id, first_id + 1])


def _mat_group(base, first_id):
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=[base, base + 0x4000], mutex_ids=[first_id, first_id + 1])


def _ub_and_mat():
    return _vec_group(0x8000, 4), _mat_group(0x8000, 6)


def _more_ub_and_mat():
    return _vec_group(0x10000, 8), _mat_group(0x10000, 10)


@pl.pipeline.stage
def produce(i, src, sh, ub):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.move(sh.next(), t)


@pl.pipeline.stage
def consume(i, sh, left):
    pl.move(left.next(), sh.next())


@pl.pipeline.stage
def produce_own_ub(i, src, sh):
    ub = _ub()
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.move(sh.next(), t)


def _load_into_own_ub(i, src, sh):
    ub = _ub()
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.move(sh.next(), t)


@pl.pipeline.stage
def produce_through_helper(i, src, sh):
    _load_into_own_ub(i, src, sh)


def _warm_up(src):
    ub = _ub()
    pl.load(ub.next(), src, [0, 0])


# ---------------------------------------------------------------------------
# Accepted: declared in the kernel body or by a factory, bound to a kernel-body variable.
# ---------------------------------------------------------------------------


@pl.pipeline.stage
def produce_into_every_group(i, src, sh, ub, ub_a, ub_b):
    t = ub.next()
    a = ub_a.next()
    b = ub_b.next()
    pl.load(t, src, [i * TM, 0])
    pl.muls(a, t, 2.0)
    pl.muls(b, a, 2.0)
    pl.move(sh.next(), b)


@pl.pipeline.stage
def consume_into_every_group(i, src, sh, left, mat_a, mat_b):
    pl.move(left.next(), sh.next())
    pl.load(mat_a.next(), src, [i * TM, 0])
    pl.load(mat_b.next(), src, [i * TM, 0])


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_every_accepted_spelling")
def every_accepted_spelling(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    # Declared in the kernel body.
    sh = pl.make_tile_group(
        type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x4000], mutex_ids=[12, 13], fwd_ids=[0, 1], bwd_ids=[2, 3])
    # Returned by a factory, in the kernel body and inside a section.
    left = _left()
    with pl.section_vector():
        ub = _ub()
    # Returned as a pair and unpacked, at once or through a tuple.
    ub_a, mat_a = _ub_and_mat()
    pair = _more_ub_and_mat()
    ub_b, mat_b = pair
    for i in pl.range(0, NT):
        # A plain view in the loop declares no buffer.
        rows = pl.make_tensor(src, [NT * TM, K], [K, 1])
        with pl.section_vector():
            produce_into_every_group(i, rows, sh, ub, ub_a, ub_b)
        with pl.section_cube():
            consume_into_every_group(i, src, sh, left, mat_a, mat_b)


def test_accepted_declarations_are_named_by_their_kernel_variable():
    result = transform(every_accepted_spelling)
    assert set(result.facts.buffer_facts) == {"sh", "left", "ub", "ub_a", "mat_a", "ub_b", "mat_b"}
    assert_targets_match_dumps(result, __file__, "every_accepted_spelling")


def test_declarations_are_listed_in_the_kernel_order_of_their_sites():
    # _ub is defined before _mat, and each pair factory makes its Vec group first; the
    # kernel takes mat before ub.
    assert list(probe_kernel_facts(every_accepted_spelling.to_kernel_def()).buffer_facts) == [
        "sh", "left", "ub", "ub_a", "mat_a", "ub_b", "mat_b"]


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_in_kernel_helper")
def in_kernel_helper(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    ub = _ub()
    left = _left()
    with pl.section_vector():
        _warm_up(src)
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce(i, src, sh, ub)
        with pl.section_cube():
            consume(i, sh, left)


def test_a_declaration_outside_every_loop_is_left_to_the_code_that_made_it():
    # _warm_up declares its own group and keeps it: nothing the pipeline tracks.
    assert set(transform(in_kernel_helper).facts.buffer_facts) == {"sh", "ub", "left"}


# ---------------------------------------------------------------------------
# Refused: no kernel-body variable holds the buffer.
# ---------------------------------------------------------------------------


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_in_stage")
def in_stage(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    left = _left()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce_own_ub(i, src, sh)
        with pl.section_cube():
            consume(i, sh, left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_in_stage_helper")
def in_stage_helper(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    left = _left()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce_through_helper(i, src, sh)
        with pl.section_cube():
            consume(i, sh, left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_passed_directly")
def passed_directly(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    left = _left()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce(i, src, sh, _ub())
        with pl.section_cube():
            consume(i, sh, left)


@pytest.mark.parametrize(
    ("kernel", "pointed_at"),
    [
        # A declaration made by a called function is reported at the kernel-body code
        # running it, naming where the declaration itself is written.
        (in_stage, "produce_own_ub(i, src, sh)"),
        (in_stage_helper, "produce_through_helper(i, src, sh)"),
        (passed_directly, "produce(i, src, sh, _ub())"),
    ],
)
def test_a_declaration_no_kernel_variable_holds_is_refused(kernel, pointed_at):
    with pytest.raises(InvalidOperation, match=_UNBOUND) as refused:
        transform(kernel)
    assert _pointed_at(refused.value) == pointed_at
    assert f"{__file__}:" in str(refused.value).splitlines()[0]


# ---------------------------------------------------------------------------
# Refused: declared inside a loop.
# ---------------------------------------------------------------------------


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_tile_group_in_loop")
def tile_group_in_loop(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    left = _left()
    for i in pl.range(0, NT):
        ub = pl.make_tile_group(
            type=pl.TileType(shape=[TM, K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0000, 0x4000], mutex_ids=[0, 1])
        with pl.section_vector():
            produce(i, src, sh, ub)
        with pl.section_cube():
            consume(i, sh, left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_in_outer_loop")
def in_outer_loop(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    left = _left()
    for o in pl.range(0, 2):
        ub = _ub()
        for i in pl.range(0, NT // 2):
            with pl.section_vector():
                produce(o * (NT // 2) + i, src, sh, ub)
            with pl.section_cube():
                consume(o * (NT // 2) + i, sh, left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_factory_in_loop")
def factory_in_loop(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    ub = _ub()
    for i in pl.range(0, NT):
        with pl.section_cube():
            left = _left()
        with pl.section_vector():
            produce(i, src, sh, ub)
        with pl.section_cube():
            consume(i, sh, left)


@pytest.mark.parametrize(
    ("kernel", "pointed_at", "declared_elsewhere"),
    [
        (tile_group_in_loop, "ub = pl.make_tile_group(", False),
        (in_outer_loop, "ub = _ub()", True),
        (factory_in_loop, "left = _left()", True),
    ],
)
def test_a_declaration_inside_a_loop_is_refused(kernel, pointed_at, declared_elsewhere):
    with pytest.raises(InvalidOperation, match=_IN_LOOP) as refused:
        transform(kernel)
    assert _pointed_at(refused.value) == pointed_at
    assert ("(declared at " in str(refused.value)) == declared_elsewhere


# ---------------------------------------------------------------------------
# Refused: one variable name for two buffers.
# ---------------------------------------------------------------------------


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_same_name_in_two_sections")
def same_name_in_two_sections(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    with pl.section_vector():
        buf = _ub()
    with pl.section_cube():
        buf = _left()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce(i, src, sh, buf)
        with pl.section_cube():
            consume(i, sh, buf)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="decl_same_name_rebound")
def same_name_rebound(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    left = _left()
    ub = _ub()
    ub = _ub()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce(i, src, sh, ub)
        with pl.section_cube():
            consume(i, sh, left)


@pytest.mark.parametrize(
    ("kernel", "pointed_at"),
    [(same_name_in_two_sections, "buf = _left()"), (same_name_rebound, "ub = _ub()")],
)
def test_two_buffers_under_one_name_are_refused(kernel, pointed_at):
    with pytest.raises(InvalidOperation, match="is bound to a second buffer") as refused:
        transform(kernel)
    # The later of the two is the one reported.
    pointed = refused.value.span.get("line") or refused.value.span.get("begin_line")
    assert _pointed_at(refused.value) == pointed_at
    # Both locations in the message name a binding, so they are comparable.
    first = re.search(r"but \S+:(\d+) already bound it", str(refused.value))
    assert int(first.group(1)) < pointed


# ---------------------------------------------------------------------------
# What a declaration resolves to: slot count and addresses.
# ---------------------------------------------------------------------------

_ADDR_BASE = 0x1000
_ADDR_STRIDE = 0x400
_DEPTH = 3
_MUTEX_BASE = 4


def _buffer_facts(kernel_def):
    return probe_kernel_facts(kernel_def.to_kernel_def()).buffer_facts


def test_auto_mutex_uses_depth_without_becoming_a_manual_id_list():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[16, 64], pl.DT_FP16]):
        auto_group = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=0, mutex_ids="auto", depth=2)
        manual_group = pl.make_tile_group(
            type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=256, mutex_ids=[4, 5])
        pl.load(auto_group.next(), a, [0, 0])
        pl.load(manual_group.next(), a, [0, 0])

    facts = _buffer_facts(k)
    slot_counts = buffer_slot_counts(facts)
    addr_ranges = buffer_addr_ranges(facts)

    # "auto" is a request for allocation, not a list of ids: the slot count comes from depth,
    # and a group that leaves depth out takes it from how many ids it names.
    assert slot_counts == {"auto_group": 2, "manual_group": 2}
    assert addr_ranges["auto_group"][1] == [(0, 128), (128, 256)]
    info = SimpleNamespace(sync=SimpleNamespace(slot_counts=slot_counts, addr_ranges=addr_ranges))
    assert _slot_count(info, "auto_group") == 2


def test_addrs_from_a_list_comprehension_over_closure_constants():
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[16, 32], pl.DT_FP16]):
        manual_group = pl.make_tile_group(
            type=pl.TileType(shape=[16, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[_ADDR_BASE + i * _ADDR_STRIDE for i in range(_DEPTH)],
            mutex_ids=[_MUTEX_BASE + i for i in range(_DEPTH)])
        pl.load(manual_group.next(), a, [0, 0])

    facts = _buffer_facts(k)

    assert buffer_slot_counts(facts) == {"manual_group": 3}
    assert buffer_addr_ranges(facts) == {
        "manual_group": (
            pl.MemorySpace.Vec,
            [(0x1000, 0x1400), (0x1400, 0x1800), (0x1800, 0x1C00)],
        )
    }
