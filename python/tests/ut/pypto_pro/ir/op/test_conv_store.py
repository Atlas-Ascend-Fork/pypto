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

"""Conv-form pl.store: an Acc tile written back to an NCHW / NCDHW annotated Tensor.

The layout marker on the output parameter selects the fixpipe TStoreAccNCHW /
TStoreAccNCDHW path: the source must be a rank-2 Acc (L0C) tile, and order /
scale / relu_pre_mode / phase / atomic / l2_cache_mode are all rejected.
"""

from pypto_pro._errors import InvalidShape, NotSupported
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _parse(kernel_def, section=ir.SectionKind.Cube):
    return kernel_def.to_kernel_def().parse_target_program(section)[0]


def _acc_type(dtype=pl.DT_FP32):
    return pl.TileType(shape=[64, 16], dtype=dtype, target_memory=pl.MemorySpace.Acc, layout=pl.NZ)


# ---------------------------------------------------------------------------
# conv2d output: NCHW annotation
# ---------------------------------------------------------------------------


def test_store_conv2d_nchw_parses():
    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16, pl.NCHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0])

    text = str(_parse(kernel))
    assert "block.store" in text


def test_store_conv2d_nchw_lowers_to_conv_global_tensor():
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import NpuArch

    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16, pl.NCHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0])

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    src = _assemble_cv_source(cube, vector).content
    # The conv store path spells the NCHW layout tag over the offset window.
    assert "pto::Layout::NCHW" in src
    assert "TSTORE<" in src


# ---------------------------------------------------------------------------
# conv3d output: NCDHW annotation
# ---------------------------------------------------------------------------


def test_store_conv3d_ncdhw_parses():
    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 4, 8, 8], pl.DT_FP16, pl.NCDHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0, 0])

    text = str(_parse(kernel))
    assert "block.store" in text


def test_store_conv3d_ncdhw_lowers_to_conv_global_tensor():
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import NpuArch

    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 4, 8, 8], pl.DT_FP16, pl.NCDHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0, 0])

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    src = _assemble_cv_source(cube, vector).content
    assert "pto::Layout::NCDHW" in src
    assert "TSTORE<" in src


# ---------------------------------------------------------------------------
# annotation rank contract
# ---------------------------------------------------------------------------


def test_nchw_annotation_requires_rank4():
    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 4, 8, 8], pl.DT_FP16, pl.NCHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0, 0])

    with pytest.raises(InvalidShape, match="NCHW Tensor requires rank 4"):
        _parse(kernel)


def test_ncdhw_annotation_requires_rank5():
    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16, pl.NCDHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0])

    with pytest.raises(InvalidShape, match="NCDHW Tensor requires rank 5"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# source tile contract: rank-2 Acc only
# ---------------------------------------------------------------------------


def test_store_conv_rejects_vec_tile():
    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16, pl.NCHW]):
        with pl.section_cube():
            tile = pl.make_tile(
                pl.TileType(shape=[64, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
                addr=0x0,
            )
            pl.store(out, tile, [0, 0, 0, 0])

    with pytest.raises(NotSupported, match="requires a rank-2 Acc tile"):
        _parse(kernel)


def test_store_conv_rejects_order():
    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16, pl.NCHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0], order=[0, 1])

    with pytest.raises(NotSupported, match="order/scale/relu_pre_mode/phase are not supported"):
        _parse(kernel)


def test_store_conv_rejects_atomic():
    from pypto_pro.language import AtomicType

    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16, pl.NCHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0], atomic=AtomicType.AtomicAdd)

    with pytest.raises(NotSupported, match="atomic is not supported"):
        _parse(kernel)


def test_store_conv_rejects_l2_cache_mode():
    @pl.jit()
    def kernel(out: pl.Tensor[[1, 16, 8, 8], pl.DT_FP16, pl.NCHW]):
        with pl.section_cube():
            acc = pl.make_tile(_acc_type(), addr=0x0)
            pl.store(out, acc, [0, 0, 0, 0], l2_cache_mode=pl.TStoreL2Hint.NormalFirstVictim)

    with pytest.raises(NotSupported, match="l2_cache_mode is not supported"):
        _parse(kernel)
