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

"""Conv-form pl.move: a 4-D packed conv weight tile folded into an L0B Right matrix.

The source must be a FRACTAL_Z / FRACTAL_Z_3D ConvTile in L1 (the fmap layouts
NC1HWC0 / NDC1HWC0 are expanded with pl.img2col instead). The path requires an
explicit offset [offset_k, offset_n] naming the window start inside the folded
matrix, and rejects the Acc→Vec/fixpipe fusion parameters. The lowering is a
format-converting TEXTRACT rather than a plain TMOV.
"""

from pypto_pro._errors import InvalidArgument, NotSupported
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir

C0 = 16


def _parse(kernel_def, section=ir.SectionKind.Cube):
    return kernel_def.to_kernel_def().parse_target_program(section)[0]


def _wgt2d_type():
    # FRACTAL_Z [C1*KH*KW, Cout/16, 16, C0] for cin=16, 3x3 kernel, cout=16
    return pl.TileType(shape=[9, 1, 16, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.FRACTAL_Z)


def _wgt3d_type():
    # FRACTAL_Z_3D [C1*KD*KH*KW, Cout/16, 16, C0] for cin=16, 2x3x3 kernel, cout=16
    return pl.TileType(shape=[18, 1, 16, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.FRACTAL_Z_3D)


def _fmap2d_type():
    # NC1HWC0 [N, C1, H, W, C0] — an fmap window, not a foldable weight
    return pl.TileType(shape=[1, 1, 8, 8, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.NC1HWC0)


def _fmap3d_type():
    # NDC1HWC0 [N, D, C1, H, W, C0] — a depth-slice window, not a foldable weight
    return pl.TileType(shape=[1, 1, 1, 8, 8, C0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                       layout=pl.Layout.NDC1HWC0)


def _right_type(rows=144, cols=16):
    # The folded L0B Right matrix: [Cin1HkWk * C0, Cout1 * Cout0]
    return pl.TileType(shape=[rows, cols], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right,
                       layout=pl.Layout.ZN)


# ---------------------------------------------------------------------------
# conv2d / conv3d weight extraction parses
# ---------------------------------------------------------------------------


def test_move_conv2d_weight_parses():
    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt2d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(144, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0])
            pl.move(dst, src, [0, 0])

    text = str(_parse(kernel))
    assert "block.move" in text


def test_move_conv3d_weight_parses():
    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 2, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt3d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(288, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0, 0])
            pl.move(dst, src, [0, 0])

    text = str(_parse(kernel))
    assert "block.move" in text


# ---------------------------------------------------------------------------
# source layout contract: only the packed weight layouts fold
# ---------------------------------------------------------------------------


def test_move_rejects_nc1hwc0_source():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_fmap2d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(), addr=0x0)
            pl.load(src, x, [0, 0, 0, 0])
            pl.move(dst, src, [0, 0])

    with pytest.raises(NotSupported, match="must be FRACTAL_Z or FRACTAL_Z_3D"):
        _parse(kernel)


def test_move_rejects_ndc1hwc0_source():
    @pl.jit()
    def kernel(x: pl.Tensor[[1, 16, 1, 8, 8], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_fmap3d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(), addr=0x0)
            pl.load(src, x, [0, 0, 0, 0, 0])
            pl.move(dst, src, [0, 0])

    with pytest.raises(NotSupported, match="must be FRACTAL_Z or FRACTAL_Z_3D"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# offset contract: the window start is mandatory
# ---------------------------------------------------------------------------


def test_move_conv_weight_requires_offset():
    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt2d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(144, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0])
            pl.move(dst, src)

    with pytest.raises(InvalidArgument, match="explicit offset"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# fusion parameters are rejected on the conv path
# ---------------------------------------------------------------------------


def test_move_conv_weight_rejects_scale():
    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt2d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(144, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0])
            pl.move(dst, src, [0, 0], scale=0.5)

    with pytest.raises(NotSupported, match="scale are not supported"):
        _parse(kernel)


def test_move_conv_weight_rejects_relu_pre_mode():
    relu = pl.ReluPreMode.NormalRelu

    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt2d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(144, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0])
            pl.move(dst, src, [0, 0], relu_pre_mode=relu)

    with pytest.raises(NotSupported, match="relu_pre_mode"):
        _parse(kernel)


def test_move_conv_weight_rejects_phase():
    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt2d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(144, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0])
            pl.move(dst, src, [0, 0], phase=pl.STPhase.Final)

    with pytest.raises(NotSupported, match="phase"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# codegen: the conv path lowers to a format-converting TEXTRACT
# ---------------------------------------------------------------------------


def test_move_conv_weight_lowers_to_textract():
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import NpuArch

    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt2d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(144, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0])
            pl.move(dst, src, [0, 0])

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    src = _assemble_cv_source(cube, vector).content
    assert "TEXTRACT<" in src


def test_move_conv3d_weight_lowers_to_textract():
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import NpuArch

    @pl.jit()
    def kernel(w: pl.Tensor[[16, 16, 2, 3, 3], pl.DT_FP16]):
        with pl.section_cube():
            src = pl.make_tile(_wgt3d_type(), addr=0x0)
            dst = pl.make_tile(_right_type(288, 16), addr=0x0)
            pl.load(src, w, [0, 0, 0, 0, 0])
            pl.move(dst, src, [0, 0])

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    src = _assemble_cv_source(cube, vector).content
    assert "TEXTRACT<" in src
