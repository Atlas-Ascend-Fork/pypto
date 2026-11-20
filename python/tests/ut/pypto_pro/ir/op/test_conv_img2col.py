#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conv-form pl.img2col: one L1 feature-map window expanded into the L0A matrix.

The source is a single conv-layout window (NC1HWC0 [1, C1, H, W, C0] for 2-D,
NDC1HWC0 [1, 1, C1, H, W, C0] for 3-D — the D loop is caller-side). The channel
depth is derived from the src tile (C1 * C0), not configured in Img2ColConfig.
"""

from pypto_pro._errors import InvalidArgument, InvalidFormat, InvalidShape, InvalidType, InvalidVal
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir

C0 = 16  # 32B / 2B for DT_FP16


def _parse(kernel_def, section=ir.SectionKind.Cube):
    return kernel_def.to_kernel_def().parse_target_program(section)[0]


def _l1_2d_type():
    # One 2-D window: NC1HWC0 [1, C1, H, W, C0], cin=16 -> C1=1
    return pl.TileType(shape=[1, 1, 8, 8, C0], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0)


def _l1_3d_type():
    # One 3-D depth slice: NDC1HWC0 [1, 1, C1, H, W, C0] (D=1, caller loops D)
    return pl.TileType(shape=[1, 1, 1, 8, 8, C0], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NDC1HWC0)


def _l0a_type(m=64, k=144):
    return pl.TileType(shape=[m, k], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Left, layout=pl.NZ)


def _cfg_2d(kernel_size=3):
    return pl.Img2ColConfig(kernel=[kernel_size, kernel_size], stride=[1, 1], dilation=[1, 1],
                            padding=[1, 1, 1, 1])


def _cfg_3d(kernel_size=3):
    return pl.Img2ColConfig(kernel=[kernel_size, kernel_size], stride=[1, 1], dilation=[1, 1],
                            padding=[1, 1, 1, 1])


# ---------------------------------------------------------------------------
# conv2d: NC1HWC0 window
# ---------------------------------------------------------------------------


def test_img2col_conv2d_parses():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            window = pl.make_tile(_l1_2d_type(), addr=0x0)
            a = pl.make_tile(_l0a_type(), addr=0x0)
            pl.load(window, x, [0, 0, 0, 0])
            # Ho = Wo = 8, K = C1*Kh*Kw*C0 = 1*3*3*16 = 144 -> dst [64, 144]
            pl.img2col(a, window, config=_cfg_2d(), pos_m=0, pos_k=0)

    text = str(_parse(kernel))
    assert "block.img2col" in text
    assert "kernel_h=3" in text and "kernel_w=3" in text
    # channel_size is derived from the src tile (C1 * C0), not configured.
    assert "channel_size=16" in text


def test_img2col_conv2d_kernel_geometry_parses():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            window = pl.make_tile(_l1_2d_type(), addr=0x0)
            # 5x5 kernel: Ho=Wo=6, K=1*5*5*16=400 -> dst [align16(36)=48, 400]
            a = pl.make_tile(_l0a_type(48, 400), addr=0x0)
            pl.load(window, x, [0, 0, 0, 0])
            pl.img2col(a, window, config=_cfg_2d(5), pos_m=0, pos_k=0)

    text = str(_parse(kernel))
    assert "block.img2col" in text
    assert "kernel_h=5" in text and "kernel_w=5" in text


# ---------------------------------------------------------------------------
# conv3d: NDC1HWC0 depth slice (D loop is caller-side)
# ---------------------------------------------------------------------------


def test_img2col_conv3d_parses():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 2, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            window = pl.make_tile(_l1_3d_type(), addr=0x0)
            a = pl.make_tile(_l0a_type(), addr=0x0)
            pl.load(window, x, [0, 0, 0, 0, 0])
            pl.img2col(a, window, config=_cfg_3d(), pos_m=0, pos_k=0)

    text = str(_parse(kernel))
    assert "block.img2col" in text
    assert "kernel_h=3" in text and "kernel_w=3" in text


# ---------------------------------------------------------------------------
# window contract: exactly one feature-map window per call
# ---------------------------------------------------------------------------


def test_img2col_rejects_multi_batch_window():
    # NC1HWC0 with N=2 is not a single window.
    @pl.jit()
    def kernel(x: pl.Tensor[[2, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            window = pl.make_tile(
                pl.TileType(shape=[2, 1, 8, 8, C0], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0),
                addr=0x0,
            )
            a = pl.make_tile(_l0a_type(128, 144), addr=0x0)
            pl.load(window, x, [0, 0, 0, 0])
            pl.img2col(a, window, config=_cfg_2d(), pos_m=0, pos_k=0)

    with pytest.raises(InvalidShape, match=r"N=1"):
        _parse(kernel)


def test_img2col_rejects_multi_depth_window():
    # NDC1HWC0 with D=2 is not a single depth slice.
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 2, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            window = pl.make_tile(
                pl.TileType(shape=[1, 2, 1, 8, 8, C0], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NDC1HWC0),
                addr=0x0,
            )
            a = pl.make_tile(_l0a_type(128, 144), addr=0x0)
            pl.load(window, x, [0, 0, 0, 0, 0])
            pl.img2col(a, window, config=_cfg_3d(), pos_m=0, pos_k=0)

    with pytest.raises(InvalidShape, match=r"D=1"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# source tile validation
# ---------------------------------------------------------------------------


def test_img2col_rejects_wrong_c0():
    # C0=8 is not a 32B block for DT_FP16 (which needs 16).
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            window = pl.make_tile(
                pl.TileType(shape=[1, 2, 8, 8, 8], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0),
                addr=0x0,
            )
            a = pl.make_tile(_l0a_type(), addr=0x0)
            pl.load(window, x, [0, 0, 0, 0])
            pl.img2col(a, window, config=_cfg_2d(), pos_m=0, pos_k=0)

    with pytest.raises(InvalidShape, match="C0"):
        _parse(kernel)


def test_img2col_rejects_non_l1_source():
    # ConvTile 只能声明在 L1；Acc 上的 NC1HWC0 TileType 在构造时即被拒绝。
    with pytest.raises(InvalidVal, match="must live in L1"):
        pl.TileType(shape=[1, 1, 8, 8, C0], dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Acc, layout=pl.Layout.NC1HWC0)


def test_img2col_rejects_non_conv_source():
    # A plain 2-D L1 tile is not a feature-map window.
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            window = pl.make_tile(_l1_2d_type(), addr=0x0)
            plain = pl.make_tile(
                pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.Layout.ND),
                addr=0x10000,
            )
            a = pl.make_tile(_l0a_type(), addr=0x0)
            pl.load(window, x, [0, 0, 0, 0])
            pl.img2col(a, plain, config=_cfg_2d(), pos_m=0, pos_k=0)

    with pytest.raises(InvalidFormat, match="must be a conv-layout tile"):
        _parse(kernel)


def test_img2col_rejects_bad_dtype():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP32]):
        with pl.section_cube():
            window = pl.make_tile(
                pl.TileType(shape=[1, 2, 8, 8, 8], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0),
                addr=0x0,
            )
            a = pl.make_tile(
                pl.TileType(shape=[64, 144], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Left, layout=pl.NZ),
                addr=0x0,
            )
            pl.load(window, x, [0, 0, 0, 0])
            pl.img2col(a, window, config=_cfg_2d(), pos_m=0, pos_k=0)

    with pytest.raises(InvalidType):
        _parse(kernel)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_img2col_config_rejects_out_of_range_kernel():
    with pytest.raises(InvalidVal, match="kernel"):
        pl.Img2ColConfig(kernel=[256, 256], stride=[1, 1], dilation=[1, 1],
                         padding=[1, 1, 1, 1])


def test_img2col_config_rejects_out_of_range_stride():
    with pytest.raises(InvalidVal, match="stride"):
        pl.Img2ColConfig(kernel=[3, 3], stride=[64, 1], dilation=[1, 1],
                         padding=[1, 1, 1, 1])


def test_img2col_config_rejects_out_of_range_padding():
    with pytest.raises(InvalidVal, match="padding"):
        pl.Img2ColConfig(kernel=[3, 3], stride=[1, 1], dilation=[1, 1],
                         padding=[256, 1, 1, 1])


def test_img2col_config_rejects_wrong_length():
    with pytest.raises(InvalidArgument, match="exactly 2 elements"):
        pl.Img2ColConfig(kernel=[3], stride=[1, 1], dilation=[1, 1],
                         padding=[1, 1, 1, 1])
