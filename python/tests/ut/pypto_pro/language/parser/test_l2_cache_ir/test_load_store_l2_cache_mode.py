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
"""Full IR golden for per-transfer L2 cache modes."""

from pathlib import Path
import re

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidType
import pypto_pro.language as pl
import pytest


@pl.jit(arch="3510", auto_mutex=False)
def l2_cache_mode_kernel(
    src: pl.Tensor[[1, 64], pl.DT_FP32],
    dst: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile(tt, addr=0)
    with pl.section_vector():
        pl.load(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NormalFirstVictim)
        pl.load(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NormalLastVictim)
        pl.load(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NormalPersistent)
        pl.load(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
        pl.load_tile(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NotAllocClean)
        pl.load_tile(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NotAllocDrop)
        pl.store(dst, tile, [0, 0], l2_cache_mode=pl.TStoreL2Hint.NormalFirstVictim)
        pl.store(dst, tile, [0, 0], l2_cache_mode=pl.TStoreL2Hint.NormalLastVictim)
        pl.store(dst, tile, [0, 0], l2_cache_mode=pl.TStoreL2Hint.NormalPersistent)
        pl.store_tile(dst, tile, [0, 0], l2_cache_mode=pl.TStoreL2Hint.NotAllocClean)


def test_load_store_l2_cache_mode():
    program, _ = l2_cache_mode_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(program))
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
    assert pl.TLoadL2Hint.NotAllocKeep.value == 3
    assert pl.TStoreL2Hint.NotAllocClean.value == 3


def test_l2_cache_mode_rejects_plain_integer():
    @pl.jit(arch="3510", auto_mutex=False)
    def invalid(src: pl.Tensor[[1, 64], pl.DT_FP32]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tt, addr=0)
        with pl.section_vector():
            pl.load(tile, src, [0, 0], l2_cache_mode=1)

    with pytest.raises(InvalidType, match="l2_cache_mode expects"):
        invalid.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_l2_cache_mode_rejects_wrong_hint_type():
    @pl.jit(arch="3510", auto_mutex=False)
    def invalid(src: pl.Tensor[[1, 64], pl.DT_FP32]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tt, addr=0)
        with pl.section_vector():
            pl.load(tile, src, [0, 0], l2_cache_mode=pl.TStoreL2Hint.NotAllocClean)

    with pytest.raises(InvalidType, match="l2_cache_mode expects TLoadL2Hint"):
        invalid.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pl.jit(arch="3510", auto_mutex=False)
def l1_load_not_alloc_keep_kernel(src: pl.Tensor[[16, 16], pl.DT_FP16]):
    tt = pl.TileType(
        shape=[16, 16],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
    )
    tile = pl.make_tile(tt, addr=0)
    with pl.section_cube():
        pl.load(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)


def test_l1_load_accepts_not_alloc_keep():
    program, _ = l1_load_not_alloc_keep_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(program))
    golden = Path(__file__).with_name("test_l1_load_not_alloc_keep.pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")


@pytest.mark.parametrize(
    "mode",
    [pl.TLoadL2Hint.NotAllocClean, pl.TLoadL2Hint.NotAllocDrop],
)
def test_l1_load_rejects_unsupported_l2_cache_mode(mode):
    @pl.jit(arch="3510", auto_mutex=False)
    def invalid(src: pl.Tensor[[16, 16], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[16, 16],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        )
        tile = pl.make_tile(tt, addr=0)
        with pl.section_cube():
            pl.load(tile, src, [0, 0], l2_cache_mode=mode)

    with pytest.raises(InvalidArgument, match="L1 loads support only"):
        invalid.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
