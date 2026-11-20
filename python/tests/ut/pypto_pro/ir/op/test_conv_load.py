#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conv-form pl.load: framework-native NCHW/NCDHW sources into ConvTiles.

The destination ConvTile's layout decides the source interpretation:
NC1HWC0 / FRACTAL_Z load NCHW [N, C, H, W] (rank 4); NDC1HWC0 / FRACTAL_Z_3D load
NCDHW [N, C, D, H, W] (rank 5).
"""

from pypto_pro._errors import InvalidType, NotSupported
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir

C0 = 16


def _fmap2d_type():
    # NC1HWC0 [N, C1, H, W, C0] for cin=16: C1=1
    return pl.TileType(shape=[1, 1, 8, 8, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.NC1HWC0)


def _fmap3d_type():
    # NDC1HWC0 [N, D, C1, H, W, C0]
    return pl.TileType(shape=[1, 2, 1, 8, 8, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.NDC1HWC0)


def _wgt2d_type():
    # FRACTAL_Z [C1*KH*KW, Cout/16, 16, C0] for cin=16, 3x3 kernel, cout=16
    return pl.TileType(shape=[9, 1, 16, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.FRACTAL_Z)


def _wgt3d_type():
    # FRACTAL_Z_3D [C1*KD*KH*KW, Cout/16, 16, C0] for cin=16, 2x3x3 kernel, cout=16
    return pl.TileType(shape=[18, 1, 16, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.FRACTAL_Z_3D)


def _parse(kernel_def, section=ir.SectionKind.Cube):
    return kernel_def.to_kernel_def().parse_target_program(section)[0]


# ---------------------------------------------------------------------------
# conv2d: NCHW rank-4 sources
# ---------------------------------------------------------------------------


def test_load_conv2d_fmap_parses():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            tile = pl.make_tile(_fmap2d_type(), addr=0x0)
            pl.load(tile, x, [0, 0, 0, 0])

    text = str(_parse(kernel))
    assert "block.load" in text


def test_load_conv2d_weight_parses():
    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            tile = pl.make_tile(_wgt2d_type(), addr=0x0)
            pl.load(tile, w, [0, 0, 0, 0])

    text = str(_parse(kernel))
    assert "block.load" in text


# ---------------------------------------------------------------------------
# conv3d: NCDHW rank-5 sources
# ---------------------------------------------------------------------------


def test_load_conv3d_fmap_parses():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 2, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            tile = pl.make_tile(_fmap3d_type(), addr=0x0)
            pl.load(tile, x, [0, 0, 0, 0, 0])

    text = str(_parse(kernel))
    assert "block.load" in text


def test_load_conv3d_weight_parses():
    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 2, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            tile = pl.make_tile(_wgt3d_type(), addr=0x0)
            pl.load(tile, w, [0, 0, 0, 0, 0])

    text = str(_parse(kernel))
    assert "block.load" in text


# ---------------------------------------------------------------------------
# rank contract: the destination layout decides the source rank
# ---------------------------------------------------------------------------


def test_load_conv2d_rejects_rank5_source():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 2, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            tile = pl.make_tile(_fmap2d_type(), addr=0x0)
            pl.load(tile, x, [0, 0, 0, 0, 0])

    with pytest.raises(NotSupported, match="NC1HWC0 destination loads from an NCHW"):
        _parse(kernel)


def test_load_conv3d_rejects_rank4_source():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            tile = pl.make_tile(_fmap3d_type(), addr=0x0)
            pl.load(tile, x, [0, 0, 0, 0])

    with pytest.raises(NotSupported, match="NDC1HWC0 destination loads from an NCDHW"):
        _parse(kernel)


def test_load_conv_rejects_dtype_mismatch():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP32]):
        with pl.section_cube():
            tile = pl.make_tile(_fmap2d_type(), addr=0x0)
            pl.load(tile, x, [0, 0, 0, 0])

    with pytest.raises(InvalidType, match="dtype mismatch"):
        _parse(kernel)


def test_load_conv_rejects_order():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            tile = pl.make_tile(_fmap2d_type(), addr=0x0)
            pl.load(tile, x, [0, 0, 0, 0], order=[0, 1])

    with pytest.raises(NotSupported, match="order is not supported"):
        _parse(kernel)
