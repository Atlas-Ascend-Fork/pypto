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

"""A pipeline kernel may not launch a SIMT vector function.

The launch is refused wherever it is written — in a stage, in a function a stage calls, or
in the pipeline loop body — and the refusal points at the launch itself.
"""

import linecache

from pipeline_test_common import transform
from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
import pytest

TM, K = 64, 128
NT = 4
THREADS = 64
_CONFIG = pl.pipeline.PipelineConfig(preload=1)
_REFUSED = "does not support SIMT"


def _pointed_at(error: NotSupported) -> str:
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


@pl.vector_function(mode="simt", max_threads=THREADS)
def simt_scale(dst, src):
    tid = pl.simt.linear_thread_idx()
    dst[tid] = pl.simt.abs(src[tid])


@pl.pipeline.stage
def produce(i, src, sh, ub):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    pl.move(sh.next(), t)


@pl.pipeline.stage
def consume(i, sh, left):
    pl.move(left.next(), sh.next())


@pl.pipeline.stage
def produce_and_launch(i, src, sh, ub, vec):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    simt_scale[THREADS](vec, vec)
    pl.move(sh.next(), t)


def _launch(vec):
    simt_scale[THREADS](vec, vec)


@pl.pipeline.stage
def produce_through_helper(i, src, sh, ub, vec):
    t = ub.next()
    pl.load(t, src, [i * TM, 0])
    _launch(vec)
    pl.move(sh.next(), t)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="simt_launch_in_stage")
def launch_in_stage(
    src: pl.Tensor[[NT * TM, K], pl.DT_FP16], vec: pl.Tensor[[THREADS], pl.DT_FP32]
):
    sh = _shared()
    left = _left()
    ub = _ub()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce_and_launch(i, src, sh, ub, vec)
        with pl.section_cube():
            consume(i, sh, left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="simt_launch_in_stage_helper")
def launch_in_stage_helper(
    src: pl.Tensor[[NT * TM, K], pl.DT_FP16], vec: pl.Tensor[[THREADS], pl.DT_FP32]
):
    sh = _shared()
    left = _left()
    ub = _ub()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce_through_helper(i, src, sh, ub, vec)
        with pl.section_cube():
            consume(i, sh, left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="simt_launch_in_loop_body")
def launch_in_loop_body(
    src: pl.Tensor[[NT * TM, K], pl.DT_FP16], vec: pl.Tensor[[THREADS], pl.DT_FP32]
):
    sh = _shared()
    left = _left()
    ub = _ub()
    for i in pl.range(0, NT):
        with pl.section_vector():
            simt_scale[THREADS](vec, vec)
        with pl.section_vector():
            produce(i, src, sh, ub)
        with pl.section_cube():
            consume(i, sh, left)


@pl.jit(arch="3510", auto_mutex=True, pipeline=_CONFIG, name="simt_none")
def no_simt(src: pl.Tensor[[NT * TM, K], pl.DT_FP16]):
    sh = _shared()
    left = _left()
    ub = _ub()
    for i in pl.range(0, NT):
        with pl.section_vector():
            produce(i, src, sh, ub)
        with pl.section_cube():
            consume(i, sh, left)


@pytest.mark.parametrize(
    ("kernel", "pointed_at"),
    [
        (launch_in_stage, "simt_scale[THREADS](vec, vec)"),
        (launch_in_stage_helper, "simt_scale[THREADS](vec, vec)"),
        (launch_in_loop_body, "simt_scale[THREADS](vec, vec)"),
    ],
)
def test_a_simt_launch_is_refused(kernel, pointed_at):
    with pytest.raises(NotSupported, match=_REFUSED) as refused:
        transform(kernel)
    assert _pointed_at(refused.value) == pointed_at
    # The refusal shows the user the launch's own file and line.
    assert f"--> {__file__}:" in str(refused.value)


def test_a_kernel_without_simt_still_transforms():
    assert "for " in transform(no_simt).source
