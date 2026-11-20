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

import os
from pathlib import Path
import re

from pypto_pro._errors import InvalidArgument
import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    return _assemble_cv_source(cube, vector).content


def _check_cce_golden(source: str, name: str) -> None:
    """Compare the complete source; PYPTO_RENDER_IR=1 refreshes the golden."""
    # Normalize generated line-end spaces to match repository whitespace checks.
    source = re.sub(r"[ \t]+$", "", source, flags=re.MULTILINE)
    path = Path(__file__).with_suffix("") / f"{name}.pypto"
    if os.environ.get("PYPTO_RENDER_IR") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.rstrip("\n") + "\n", encoding="utf-8")
        return
    assert source.rstrip("\n") == path.read_text(encoding="utf-8").rstrip("\n")


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


@pl.jit(auto_mutex=True)
def _auto_mutex_split_n_transpose_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    ga = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0, mutex_ids="auto", depth=1,
    )
    gb = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids="auto", depth=1,
    )
    la = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, compact=1),
        addrs=0, mutex_ids="auto", depth=2,
    )
    lb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, compact=1),
        addrs=0, mutex_ids="auto", depth=2,
    )
    gc = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, compact=1),
        addrs=0, mutex_ids="auto", depth=1,
    )
    acc_leaf_type = pl.TileType(
        shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, compact=1,
    )
    pl.make_tile_group(type=acc_leaf_type, addrs=0, mutex_ids=[20])
    pl.make_tile_group(type=acc_leaf_type, addrs=0x2000, mutex_ids=[21])
    with pl.section_cube():
        x = ga.current()
        y = gb.current()
        z = gc.current()
        pl.load(x, a, [0, 0])
        pl.load(y, b, [0, 0])
        pl.gemm(z, x, y, l0a=la, l0b=lb, splitn=32, n=64, k=128, trans_b=True)
        pl.store(out, z, [0, 0])


def test_cce_auto_mutex_split_n_transpose_materializes_dynamic_strip_tuple():
    cpp = _compile_to_cce(_auto_mutex_split_n_transpose_kernel)
    _check_cce_golden(cpp, "auto_mutex_split_n_transpose")

    assert "TMATMUL" in cpp
    assert re.search(r"SyncedTile<Tile<[^;]*>, 2>", cpp)
    assert "__pypto_auto_mutex_id_" not in cpp


@pl.jit(auto_mutex=True)
def _auto_containment_parent_kernel(x: pl.Tensor[[1, 32], pl.DT_FP16]):
    root_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    parent_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    leaf_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    pl.make_tile_group(type=root_type, addrs=0, mutex_ids=[[2, 5, 7, 9]])
    parent = pl.make_tile_group(type=parent_type, addrs=0, mutex_ids="auto", depth=1)
    pl.make_tile_group(type=leaf_type, addrs=0, mutex_ids="auto", depth=1)
    pl.make_tile_group(type=leaf_type, addrs=32, mutex_ids=[5])
    pl.make_tile_group(type=parent_type, addrs=64, mutex_ids=[[7, 9]])
    with pl.section_vector():
        tile = parent.current()
        pl.load(tile, x, [0, 0])
        pl.store(x, tile, [0, 0])


def test_cce_auto_containment_parent_uses_two_resolved_mutex_ids():
    cpp = _compile_to_cce(_auto_containment_parent_kernel)
    _check_cce_golden(cpp, "auto_containment_parent")

    assert re.search(r"SyncedTile<Tile<[^;]*>, 2>", cpp)
    for pipe in ("MTE2", "MTE3"):
        for mutex_id in (2, 5):
            assert f"get_buf(PIPE_{pipe}, {mutex_id}, 0);" in cpp
            assert f"rls_buf(PIPE_{pipe}, {mutex_id}, 0);" in cpp


@pl.jit(auto_mutex=True)
def _different_group_sibling_counts_kernel(x: pl.Tensor[[1, 64], pl.DT_FP16]):
    root_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    child_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    leaf_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    pl.make_tile_group(type=root_type, addrs=0, mutex_ids=[[0, 1, 2]])
    two_id_child = pl.make_tile_group(type=child_type, addrs=0, mutex_ids="auto", depth=1)
    one_id_sibling = pl.make_tile_group(type=child_type, addrs=64, mutex_ids=[2])
    pl.make_tile_group(type=leaf_type, addrs=0, mutex_ids=[[0, 1]])
    with pl.section_vector():
        pl.load(two_id_child[0], x, [0, 0])
        pl.store(x, two_id_child[0], [0, 0])
        pl.load(one_id_sibling[0], x, [0, 32])
        pl.store(x, one_id_sibling[0], [0, 32])


def test_cce_different_tile_groups_allow_different_sibling_mutex_counts():
    cpp = _compile_to_cce(_different_group_sibling_counts_kernel)
    _check_cce_golden(cpp, "different_group_sibling_counts")

    for pipe in ("MTE2", "MTE3"):
        for mutex_id in (0, 1, 2):
            assert f"get_buf(PIPE_{pipe}, {mutex_id}, 0);" in cpp
            assert f"rls_buf(PIPE_{pipe}, {mutex_id}, 0);" in cpp


@pl.jit(auto_mutex=True)
def _same_group_incompatible_counts_kernel(x: pl.Tensor[[1, 64], pl.DT_FP16]):
    root_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    child_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    leaf_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    pl.make_tile_group(type=root_type, addrs=0, mutex_ids=[[0, 1, 2]])
    pl.make_tile_group(type=root_type, addrs=128, mutex_ids=[2])
    tiles = pl.make_tile_group(type=child_type, addrs=[0, 128], mutex_ids="auto", depth=2)
    pl.make_tile_group(type=leaf_type, addrs=0, mutex_ids=[[0, 1]])
    with pl.section_vector():
        pl.load(tiles[0], x, [0, 0])


def test_cce_same_tile_group_reports_incompatible_mutex_counts():
    with pytest.raises(InvalidArgument, match="separate TileGroups"):
        _compile_to_cce(_same_group_incompatible_counts_kernel)
