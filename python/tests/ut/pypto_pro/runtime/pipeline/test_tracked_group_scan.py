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

"""What the scan makes of a group the sync graph tracks.

Three things it must not depend on: where the group was declared, how an op spells its read
and its write of a slot, and whether the op is a block op or a vf instruction.
"""

from pipeline_test_common import assert_targets_match_dumps, transform
from pypto_pro._errors import InvalidArgument, InvalidOperation
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest

TM, TN = 64, 64
NT = 4
FULL_M = TM * NT
_CONFIG = pl.pipeline.PipelineConfig(preload=1)


# ---------------------------------------------------------------------------
# A group a factory returns is a group: it decides whether a slot taken outside the pipeline
# loop travels as an index in ctx, and whether a slot may be taken in the loop body.
# ---------------------------------------------------------------------------


def _shared():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])


def _left():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=[0x0000, 0x2000], mutex_ids=[4, 5])


def _ub():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x0000, 0x2000], mutex_ids=[6, 7])


@pl.pipeline.stage
def produce(k, src, sh, tile):
    pl.load(tile, src, [k * TM, 0])
    shared = sh.next()
    pl.move(shared, tile)


@pl.pipeline.stage
def consume(k, sh, leftg):
    shared = sh.next()
    left = leftg.next()
    pl.move(left, shared)


@pl.pipeline.stage
def produce_from_both(k, src, sh, t, u):
    pl.load(t, src, [k * TM, 0])
    pl.muls(u, t, 2.0)
    shared = sh.next()
    pl.move(shared, u)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="outer_slots")
def outer_slots(src: pl.Tensor[[FULL_M, TN], pl.DT_FP16]):
    sh = _shared()
    with pl.section_cube():
        leftg = _left()
    with pl.section_vector():
        ub_factory = _ub()
        ub_kernel = pl.make_tile_group(
            type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
            addrs=[0x4000, 0x6000], mutex_ids=[8, 9])
    for o in pl.range(0, 2):
        with pl.section_vector():
            t = ub_factory.next()
            u = ub_kernel.next()
        for k in pl.range(0, NT // 2):
            with pl.section_vector():
                produce_from_both(o * (NT // 2) + k, src, sh, t, u)
            with pl.section_cube():
                consume(o * (NT // 2) + k, sh, leftg)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="factory_slot_in_loop_body")
def slot_of_a_factory_group_in_loop_body(src: pl.Tensor[[FULL_M, TN], pl.DT_FP16]):
    sh = _shared()
    with pl.section_cube():
        leftg = _left()
    with pl.section_vector():
        ub = _ub()
    for k in pl.range(0, NT):
        _spare = sh.next()
        with pl.section_vector():
            produce(k, src, sh, ub.next())
        with pl.section_cube():
            consume(k, sh, leftg)


def test_a_slot_taken_outside_the_pipeline_loop_travels_in_ctx():
    # The stage is handed the slot its own task picked, re-selected from the index in ctx,
    # whether the group was declared in the kernel body or returned by a factory.
    source = transform(outer_slots).source
    assert "_pl_idx_ub_factory" in source
    assert "_pl_idx_ub_kernel" in source


def test_a_tracked_factory_group_slot_taken_in_the_loop_body_is_refused():
    with pytest.raises(InvalidOperation, match="takes a slot from tile group"):
        transform(slot_of_a_factory_group_in_loop_body)


# ---------------------------------------------------------------------------
# The roles an op's arguments carry, and the sync they imply. An op handed the same buffer at
# a read and a write argument folds to one "RW" access, which takes part on both sides of the
# graph: it consumes what the previous write left, and its own write must wait for the readers
# of the batch still living on the slot. The second half is what a pure writer gets.
# ---------------------------------------------------------------------------


def _cross_core():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x0000, 0x2000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])


def _mat():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=[0x0000, 0x8000], mutex_ids=[4, 5])


def _ub_scratch():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x4000, 0x6000], mutex_ids=[6, 7])


@pl.pipeline.stage
def write_the_slot(k, src, sh, ub):
    t = ub.next()
    pl.load(t, src, [k * TM, 0])
    s = sh.next()
    # move's dst is a plain W, the shape every cross-core producer has.
    pl.move(s, t)


@pl.pipeline.stage
def modify_the_slot(k, src, sh, ub):
    t = ub.next()
    pl.load(t, src, [k * TM, 0])
    s = sh.next()
    # axpy's out is "RW" — the stage's only access to `sh`, so nothing else can carry the
    # handshake the write needs.
    pl.axpy(s, t, 2.0)


@pl.pipeline.stage
def read_the_slot(k, sh, mat):
    s = sh.next()
    m = mat.next()
    pl.move(m, s)


def _in_place_kernel(producer, name):
    @pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name=name)
    def kern(src: pl.Tensor[[NT * TM, TN], pl.DT_FP16]):
        sh = _cross_core()
        with pl.section_vector():
            ub = _ub_scratch()
        with pl.section_cube():
            mat = _mat()
        for k in pl.range(0, NT):
            with pl.section_vector():
                producer(k, src, sh, ub)
            with pl.section_cube():
                read_the_slot(k, sh, mat)
    return kern


pure_write = _in_place_kernel(write_the_slot, "in_place_sync_pure_write")
in_place = _in_place_kernel(modify_the_slot, "in_place_sync_rw")


def _sync_lines(source: str) -> list:
    """The cross-core wait/set in ``source``, in source order."""
    return [line.strip() for line in source.splitlines() if "cross_core" in line]


def test_an_in_place_producer_gets_the_same_handshake_as_a_pure_writer():
    # The two kernels differ only in the op that fills the slot, so every wait/set matches.
    assert _sync_lines(transform(in_place).source) == _sync_lines(transform(pure_write).source)


def test_an_in_place_write_waits_for_the_readers_of_the_batch_on_its_slot():
    result = transform(in_place)
    # The WAR edge of the pipeline loop: before overwriting the slot, the producer waits on
    # the backward ids the consumer releases for the task two rotations back.
    assert "pl.system.wait_cross_core(pipe=pl.PipeType.V, " \
           "event_id=_pl_bwd_ids_sh[_pl_ctx_0._pl_task_id % 2])" in _sync_lines(result.source)
    assert_targets_match_dumps(result, __file__, "in_place")


# ---------------------------------------------------------------------------
# A slot a helper hands back is still the group's slot, so the stage reading it is
# synchronised like any other.
# ---------------------------------------------------------------------------


def _hand_back_slot(g):
    return g.next()


@pl.pipeline.stage
def read_a_returned_slot(k, sh, leftg):
    shared = _hand_back_slot(sh)
    left = leftg.next()
    pl.move(left, shared)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="helper_returns_a_slot")
def helper_returns_a_slot(src: pl.Tensor[[FULL_M, TN], pl.DT_FP16]):
    sh = _shared()
    with pl.section_cube():
        leftg = _left()
    with pl.section_vector():
        ub = _ub()
    for k in pl.range(0, NT):
        with pl.section_vector():
            write_the_slot(k, src, sh, ub)
        with pl.section_cube():
            read_a_returned_slot(k, sh, leftg)


def test_a_slot_a_helper_hands_back_is_synchronised():
    lines = [line.strip() for line in transform(helper_returns_a_slot).source.splitlines()]
    call = next(i for i, line in enumerate(lines) if line.startswith("read_a_returned_slot("))
    # The read `pl.move` makes of the returned slot is a read of `sh`, so the consumer's
    # call waits for the producer and hands the slot back.
    assert "wait_cross_core(pipe=pl.PipeType.MTE1, event_id=_pl_fwd_ids_sh[" in lines[call - 1]
    assert "set_cross_core(pipe=pl.PipeType.MTE1, event_id=_pl_bwd_ids_sh[" in lines[call + 1]


# ---------------------------------------------------------------------------
# A vf instruction is scanned the way a block op is: one access per recorded op, with being
# inside a vector function deciding only the pipe and which roles table is consulted. An
# argument the table does not reach is a gap in that table, named by the refusal.
# ---------------------------------------------------------------------------


@pl.vector_function
def store_with_the_tile_as_mask(tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    v = vf.load_align(tile, 0)
    v = vf.muls(v, 2.0, preg)
    # store_align is tabled as (dst, src), so the third argument is past the table's roles —
    # the buffer reaches an argument the scan has no verdict for.
    vf.store_align(tile, v, tile)


@pl.pipeline.stage
def modify_in_a_vf(k, sh):
    store_with_the_tile_as_mask(sh.next())


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="vf_scan_uncovered_arg")
def uncovered_vf_argument(src: pl.Tensor[[NT * TM, TN], pl.DT_FP16]):
    sh = _cross_core()
    with pl.section_cube():
        mat = _mat()
    for k in pl.range(0, NT):
        with pl.section_vector():
            modify_in_a_vf(k, sh)
        with pl.section_cube():
            read_the_slot(k, sh, mat)


def test_a_tracked_buffer_at_an_untabled_vf_argument_names_the_vf_table():
    with pytest.raises(InvalidArgument, match="_VF_OP_TILE_ROLES gives 'store_align' no role"):
        transform(uncovered_vf_argument)


# ---------------------------------------------------------------------------
# A GM buffer's slot is selected at the call site, by the declared name. An argument that
# only aliases the buffer must select it the same way: the transform resolves the argument
# through the probe's facts rather than matching the name as written.
# ---------------------------------------------------------------------------

TK = 128


def _gm(ws):
    return pl.make_tensor(ws, [TM, TK], dtype=pl.DT_FP16, depth=2, fwd_ids=[4, 5], bwd_ids=[6, 7])


def _ub_rows():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=[0x0000, 0x4000], mutex_ids=[8, 9])


def _mat_rows():
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=[0x0000, 0x4000], mutex_ids=[10, 11])


@pl.pipeline.stage
def write_the_gm_slot(k, src, gm, ub):
    t = ub.next()
    pl.load(t, src, [k * TM, 0])
    pl.store(gm, t, [0, 0])


@pl.pipeline.stage
def read_the_gm_slot(k, gm, mat):
    pl.load(mat.next(), gm, [0, 0])


def _gm_kernel(alias: bool, name: str):
    if alias:
        @pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name=name)
        def kern(src: pl.Tensor[[NT * TM, TK], pl.DT_FP16], ws: pl.Ptr[pl.DT_UINT8]):
            gm = _gm(ws)
            gm_alias = gm
            ub = _ub_rows()
            mat = _mat_rows()
            for k in pl.range(0, NT):
                with pl.section_vector():
                    write_the_gm_slot(k, src, gm_alias, ub)
                with pl.section_cube():
                    read_the_gm_slot(k, gm_alias, mat)
        return kern

    @pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name=name)
    def kern(src: pl.Tensor[[NT * TM, TK], pl.DT_FP16], ws: pl.Ptr[pl.DT_UINT8]):
        gm = _gm(ws)
        ub = _ub_rows()
        mat = _mat_rows()
        for k in pl.range(0, NT):
            with pl.section_vector():
                write_the_gm_slot(k, src, gm, ub)
            with pl.section_cube():
                read_the_gm_slot(k, gm, mat)
    return kern


def test_an_alias_of_a_gm_buffer_selects_the_slot_the_declared_name_would():
    direct = transform(_gm_kernel(False, "gm_slot_direct")).source
    result = transform(_gm_kernel(True, "gm_slot_aliased"))
    aliased = result.source
    # The slot is bound and passed under the DECLARED name, so the two agree but for the
    # alias's own binding.
    assert "_pl_gm_gm = " in aliased
    assert "gm_alias" not in aliased.replace("gm_alias = gm", "")
    assert aliased.replace("    gm_alias = gm\n", "") == direct
    assert_targets_match_dumps(result, __file__, "gm_alias")
