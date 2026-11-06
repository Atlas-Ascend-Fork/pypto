#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A cross-core tile group a factory function declares and returns.

Production kernels do not declare their buffers in the kernel body. They call a factory
(``q_l1 = make_q_l1()``) that lives in a module of its own and configures itself from names the
kernel's closure never sees, and some of them take arguments (``make_y_ub(v1w)``).

The judge is the generated code. A declaration the framework fails to recognise is not refused,
it is dropped, and a dropped cross-core declaration means NO handshake at all -- the two cores
then race over the shared L1. So each factory kernel below is compared line by line against an
inline-declared twin that is otherwise the same kernel: the declaration statement is the only
difference allowed, and the twin has to carry a handshake for the comparison to mean anything.
One kernel also runs, so the addresses and ids the factory chose are checked for real.
"""

import ast
import os

from _buffer_factories import make_shared_l1
from pypto_pro._errors import span_source
import pypto_pro.language as pl
from pypto_pro.language.parser._span_tracker import SpanTracker
from pypto_pro.runtime.pipeline import transform_pipeline
from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts
import pytest
import torch

pytestmark = pytest.mark.soc("950")

ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
TM, TN, TK = 64, 64, 64
NT = 4
FULL_M = TM * (NT + 2)      # the fill and drain beats get rows of their own
SCALE = 2.0

# Names the factories below configure themselves from. The kernel body never reads them, so a
# factory that was inlined into the kernel's namespace instead of called would not resolve.
SHARED_ADDRS = [0x0000, 0x8000]
SHARED_PITCH = 0x8000
SHARED_MUTEX_IDS = [0, 1]
SHARED_FWD_IDS = [0, 1]
SHARED_BWD_IDS = [2, 3]


def _generated(kernel) -> str:
    """The transformed kernel as source, the way the framework hands it to codegen."""
    kd = kernel.to_kernel_def()
    facts = probe_kernel_facts(kd, None)
    spans = SpanTracker(kd._source_file, kd._source_lines, kd._line_offset, kd._col_offset)
    with span_source(spans.get_span):
        tree = transform_pipeline(
            kd._func_def, kd._closure_vars, kd._pipeline, facts=facts,
            tilingkey_consts=kd._tilingkey_consts, datatype_consts=kd._datatype_consts,
        )
    return ast.unparse(tree)


def _split_off_the_declaration(generated: str) -> tuple:
    """The generated lines, with the one that binds ``sh`` held back.

    The transform never rewrites a declaration -- it only adds its own statements beside it --
    so the declaration is the one line that legitimately differs between the variants.
    """
    kept, declaration = [], []
    for line in generated.splitlines():
        (declaration if line.strip().startswith("sh = ") else kept).append(line)
    return kept, declaration


def _handshake(generated: str) -> list:
    return [line.strip() for line in generated.splitlines() if "_cross_core(" in line]


def _shared_l1():
    """The same buffer, declared by a factory in this module."""
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=SHARED_ADDRS, mutex_ids=SHARED_MUTEX_IDS,
        fwd_ids=SHARED_FWD_IDS, bwd_ids=SHARED_BWD_IDS)


def _shared_l1_at(base):
    """A factory that takes the base address, so its slots are spanned from an argument."""
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=[base, base + SHARED_PITCH], mutex_ids=SHARED_MUTEX_IDS,
        fwd_ids=SHARED_FWD_IDS, bwd_ids=SHARED_BWD_IDS)


@pl.pipeline.stage
def scale_into_l1(k, src, sh, ub, nz):
    """Vector side: scales a tile and hands it to the cube core through the shared L1."""
    tile = ub.next()
    pl.load(tile, src, [k * TM, 0])
    pl.muls(tile, tile, SCALE)
    packed = nz.next()
    pl.move(packed, tile)
    shared = sh.next()
    pl.move(shared, packed)


@pl.pipeline.stage
def multiply(k, b, out, sh, matb, la, lb, acc):
    """Cube side: the other half of the handover, and the result reaches GM."""
    shared = sh.next()
    mb = matb.next()
    left = la.next()
    right = lb.next()
    c = acc.next()
    pl.load(mb, b, [0, 0])
    pl.move(left, shared)
    pl.move(right, mb)
    pl.matmul(c, left, right)
    pl.store(out, c, [k * TM, 0])


def _kernel(how: str, name: str):
    """One kernel, four ways of declaring its cross-core L1 and nothing else different."""

    @pl.jit(arch="3510", auto_mutex=True, name=name,
            pipeline=pl.pipeline.PipelineConfig(preload=1))
    def kernel(
        src: pl.Tensor[[FULL_M, TK], pl.DT_FP16],
        b: pl.Tensor[[TK, TN], pl.DT_FP16],
        out: pl.Tensor[[FULL_M, TN], pl.DT_FP32],
    ):
        if how == "inline":
            sh = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
                addrs=[0x0000, 0x8000], mutex_ids=[0, 1], fwd_ids=[0, 1], bwd_ids=[2, 3])
        elif how == "factory":
            sh = _shared_l1()
        elif how == "factory_arg":
            sh = _shared_l1_at(0x0000)
        else:
            sh = make_shared_l1()
        with pl.section_cube():
            matb = pl.make_tile_group(
                type=pl.TileType(shape=[TK, TN], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
                addrs=0x10000, mutex_ids=[2])
            la = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Left, compact=1),
                addrs=0x0000, mutex_ids=[3])
            lb = pl.make_tile_group(
                type=pl.TileType(shape=[TK, TN], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right, compact=1),
                addrs=0x0000, mutex_ids=[4])
            acc = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=0x0000, mutex_ids=[5])
        with pl.section_vector():
            ub = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addrs=0x0000, mutex_ids=[6])
            nz = pl.make_tile_group(
                type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Vec, layout=pl.NZ, compact=1),
                addrs=0x8000, mutex_ids=[7])
        for ki in pl.range(0, NT):
            with pl.section_vector():
                scale_into_l1(ki, src, sh, ub, nz)
            with pl.section_cube():
                multiply(ki, b, out, sh, matb, la, lb, acc)

    return kernel


@pytest.mark.parametrize(
    "how",
    [
        "factory",       # a factory of this module's own, configured from names the kernel lacks
        "factory_arg",   # the base address comes in as an argument
        "imported",      # the factory lives in another module, like fag_buffers.py
    ],
)
def test_a_factory_declared_group_is_handed_over_like_an_inline_one(how):
    inline = _generated(_kernel("inline", f"tgf_inline_{how}"))
    assert _handshake(inline), "the inline twin emits no handshake, so there is nothing to match"

    expected, inline_declaration = _split_off_the_declaration(inline)
    produced, factory_declaration = _split_off_the_declaration(_generated(_kernel(how, f"tgf_{how}")))
    assert len(inline_declaration) == 1 and len(factory_declaration) == 1
    assert produced == expected


def test_a_factory_declared_group_hands_over_the_right_tile():
    """The factory's own addresses and ids on device: what cube multiplies is what vector wrote."""
    torch.npu.set_device(ST_DEVICE)
    generator = torch.Generator().manual_seed(23)
    src = torch.randn((FULL_M, TK), generator=generator, dtype=torch.float16)
    b = torch.randn((TK, TN), generator=generator, dtype=torch.float16)
    out = torch.full((FULL_M, TN), 7.0, device=ST_DEVICE, dtype=torch.float32)

    _kernel("factory", "tile_group_factory_on_device")(src.to(ST_DEVICE), b.to(ST_DEVICE), out)
    torch.npu.synchronize()

    golden = (src[:NT * TM].float() * SCALE) @ b.float()
    torch.testing.assert_close(out[:NT * TM].cpu(), golden, rtol=1e-3, atol=1e-3)
