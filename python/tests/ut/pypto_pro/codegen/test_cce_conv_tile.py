#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A conv-layout TileType lowers to a multi-dim pto::ConvTile, not a 2-D pto::Tile.

The frontend selects the pto::ConvTile path from the layout alone; codegen must emit
``ConvTile<TileType::Mat, <ctype>, <BufferSize>, pto::Layout::<X>, ConvTileShape<...>>`` for it
(and keep emitting ``Tile<...>`` for every other tile). Tiles are declared at their allocation
site as ``SyncedTile<PtoTile, N>``, so the ConvTile is the synced tile's underlying type and its
instance passes no 2-D ``(rows, cols)`` constructor arguments. These read the generated C++
directly, so they need no device and no bisheng.
"""

import os
from pathlib import Path
import re

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch

_GOLDEN_DIR = Path(__file__).parent / "test_cce_conv_tile_src"


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    return _assemble_cv_source(cube, vector).content


def _source_golden(kernel, golden_name: str) -> str:
    """Whole generated CCE source, compared against the .pypto golden.

    Set PYPTO_RENDER_IR=1 to (re)generate the golden files.
    """
    source = _compile_to_cce(kernel)
    golden_path = _GOLDEN_DIR / f"{golden_name}.pypto"
    if os.environ.get("PYPTO_RENDER_IR"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(source.rstrip("\n") + "\n", encoding="utf-8")
        return source
    golden = golden_path.read_text(encoding="utf-8")
    assert source.rstrip("\n") == golden.rstrip("\n")
    return source


def _tile_alias(source: str, tile_group: str) -> str:
    """The underlying pto tile type of the group's first allocated slot."""
    match = re.search(rf"using _tg_{re.escape(tile_group)}_tiles_0_0_Type = SyncedTile<(.*), \d+>;", source)
    assert match is not None, f"no tile alias found for '{tile_group}'"
    return match.group(1)


def _conv_tile_alias(source: str, tile_group: str) -> str:
    alias = _tile_alias(source, tile_group)
    assert alias.startswith("ConvTile<"), f"'{tile_group}' did not lower to a ConvTile: {alias}"
    return alias


def _conv_tile_decl_args(source: str, tile_group: str) -> str:
    """The constructor arguments of the first ConvTile instance of a group."""
    tile = re.escape(f"_tg_{tile_group}_tiles_0_0")
    match = re.search(rf"{tile}_Type {tile}\{{[A-Za-z0-9_]+::PtoTile\(([^)]*)\), \{{", source)
    assert match is not None, f"no ConvTile instance found for '{tile_group}'"
    return match.group(1)


@pl.jit(auto_mutex=True)
def _fmap_kernel(x: pl.Tensor[[64], pl.DT_FP16]):
    tt = pl.TileType(
        shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0
    )
    g = pl.make_tile_group(type=tt, addrs=0, depth=1, mutex_ids=[0])
    _ = g.next()


@pl.jit(auto_mutex=True)
def _fmap3d_kernel(x: pl.Tensor[[64], pl.DT_FP16]):
    tt = pl.TileType(
        shape=[1, 2, 4, 8, 8, 16],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.Layout.NDC1HWC0,
    )
    g = pl.make_tile_group(type=tt, addrs=0, depth=1, mutex_ids=[0])
    _ = g.next()


@pl.jit(auto_mutex=True)
def _weight_kernel(x: pl.Tensor[[64], pl.DT_FP16]):
    tt = pl.TileType(
        shape=[64, 4, 16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.Layout.FRACTAL_Z
    )
    g = pl.make_tile_group(type=tt, addrs=0, depth=1, mutex_ids=[0])
    _ = g.next()


@pl.jit(auto_mutex=True)
def _weight3d_kernel(x: pl.Tensor[[64], pl.DT_FP16]):
    tt = pl.TileType(
        shape=[64, 4, 16, 16],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.Layout.FRACTAL_Z_3D,
    )
    g = pl.make_tile_group(type=tt, addrs=0, depth=1, mutex_ids=[0])
    _ = g.next()


def test_conv_nc1hwc0_lowers_to_conv_tile():
    # BufferSize = 1*4*8*8*16 = 4096 elements; C0 = 16 for FP16.
    src = _source_golden(_fmap_kernel, "fmap_nc1hwc0")
    alias = _conv_tile_alias(src, "g")
    assert alias == ("ConvTile<TileType::Mat, half, 4096, pto::Layout::NC1HWC0, ConvTileShape<1, 4, 8, 8, 16>>")


def test_conv_fractal_z_lowers_to_conv_tile():
    # BufferSize = 64*4*16*16 = 65536 elements.
    src = _source_golden(_weight_kernel, "weight_fractal_z")
    alias = _conv_tile_alias(src, "g")
    assert alias == ("ConvTile<TileType::Mat, half, 65536, pto::Layout::FRACTAL_Z, ConvTileShape<64, 4, 16, 16>>")


def test_conv_ndc1hwc0_lowers_to_conv_tile():
    # BufferSize = 1*2*4*8*8*16 = 8192 elements.
    src = _source_golden(_fmap3d_kernel, "fmap3d_ndc1hwc0")
    alias = _conv_tile_alias(src, "g")
    assert alias == (
        "ConvTile<TileType::Mat, half, 8192, pto::Layout::NDC1HWC0, ConvTileShape<1, 2, 4, 8, 8, 16>>"
    )


def test_conv_fractal_z_3d_lowers_to_conv_tile():
    # BufferSize = 64*4*16*16 = 65536 elements.
    src = _source_golden(_weight3d_kernel, "wgt3d_fractal_z_3d")
    alias = _conv_tile_alias(src, "g")
    assert alias == (
        "ConvTile<TileType::Mat, half, 65536, pto::Layout::FRACTAL_Z_3D, ConvTileShape<64, 4, 16, 16>>"
    )


def test_plain_tile_still_lowers_to_tile():
    @pl.jit(auto_mutex=True)
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, depth=1, mutex_ids=[0])
        _ = g.next()

    src = _source_golden(k, "plain_tile")
    assert "ConvTile<" not in src
    assert _tile_alias(src, "g").startswith("Tile<")


def test_conv_tile_declared_without_2d_ctor_args():
    # A pto::ConvTile carries its whole shape in ConvTileShape<...> and has no ``(rows, cols)``
    # constructor; emitting the 2-D Tile ctor args makes the device build fail (pto's ConvTile
    # static_asserts that ctor values are int64_t and match the dynamic-dimension count).
    for kernel, golden in (
        (_fmap_kernel, "fmap_nc1hwc0"),
        (_fmap3d_kernel, "fmap3d_ndc1hwc0"),
        (_weight_kernel, "weight_fractal_z"),
        (_weight3d_kernel, "wgt3d_fractal_z_3d"),
    ):
        assert _conv_tile_decl_args(_source_golden(kernel, golden), "g") == ""
