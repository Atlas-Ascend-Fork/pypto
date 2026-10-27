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
"""CCE codegen tests for per-tile-aware dynamic mutex deduplication."""

import re

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    return _assemble_cv_source(cube, vector).content


@pl.jit(auto_mutex=True)
def _mixed_multi_mutex_ids_kernel(x: pl.Tensor[[64, 32], pl.DT_FP16]):
    tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    source_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    output_group = pl.make_tile_group(type=tile_type, addrs=0x1000, mutex_ids=[[0, 2], [1, 3]])
    with pl.section_vector():
        for index in pl.range(0, 2):
            source_tile = source_group[index]
            output_tile = output_group[index]
            pl.load(source_tile, x, [index * 32, 0])
            pl.move(output_tile, source_tile)
            value = output_tile[0, 0]
            output_tile[1, 0] = value


def test_cce_dynamic_mutex_dedup_skips_same_tile_comparisons():
    cpp = _compile_to_cce(_mixed_multi_mutex_ids_kernel)

    assert "source_tile__mutexid" not in cpp
    assert "output_tile__mutexid" not in cpp
    assert ".mutex_ids[0]" in cpp and ".mutex_ids[1]" in cpp
    # Only the first output slot can overlap the source candidates.
    # Compute that comparison once and reuse it at acquisition/release.
    assert cpp.count(" != ") == 1
    guards = re.findall(r"if \((__sync_needed_\w+)\)", cpp)
    assert len(guards) == 2 and guards[0] == guards[1]
    assert cpp.count("get_buf(PIPE_S,") == cpp.count("rls_buf(PIPE_S,") == 4
    acquire_ids = re.findall(r"get_buf\(PIPE_V, (\w+), 0\);", cpp)
    release_ids = re.findall(r"rls_buf\(PIPE_V, (\w+), 0\);", cpp)
    assert len(acquire_ids) == 3
    assert acquire_ids == release_ids


@pl.jit(auto_mutex=True)
def _auto_mutex_ids_kernel(x: pl.Tensor[[64, 32], pl.DT_FP16]):
    tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    manual_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])  # noqa: F841
    auto_group = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids="auto", depth=2)
    with pl.section_vector():
        for index in pl.range(0, 2):
            auto_tile = auto_group[index]
            pl.load(auto_tile, x, [index * 32, 0])
            pl.store(x, auto_tile, [index * 32, 0])


def test_cce_resolved_auto_mutex_ids_use_scalar_sync():
    cpp = _compile_to_cce(_auto_mutex_ids_kernel)

    assert ".mutex_ids[0]" in cpp
    assert "__pypto_auto_mutex_id_" not in cpp
    for pipe in ("MTE2", "MTE3"):
        assert cpp.count(f"get_buf(PIPE_{pipe},") == 1
        assert cpp.count(f"rls_buf(PIPE_{pipe},") == 1
