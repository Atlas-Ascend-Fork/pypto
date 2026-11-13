# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A5 numerical coverage for SwigLU gradient implemented with SIMD tile operations."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


def _tile(tile_type, addr):
    group = pl.make_tile_group(type=tile_type, addrs=addr, mutex_ids="auto", depth=1)
    return group.current()


@pl.vector_function(mode="simd_tile")
def swiglu_grad(a, b, grad, sigmoid, grad_s, grad_tmp, da, db):
    pl.neg(sigmoid, a)
    pl.exp(sigmoid, sigmoid)
    pl.add(sigmoid, sigmoid, 1.0)
    pl.recip(sigmoid, sigmoid)
    pl.mul(grad_s, grad, sigmoid)
    pl.mul(db, grad_s, a)
    pl.sub(grad_tmp, sigmoid, 1.0)
    pl.mul(grad_tmp, grad_tmp, db)
    pl.sub(grad_tmp, grad_s, grad_tmp)
    pl.mul(da, grad_tmp, b)


@pl.jit(auto_mutex=True)
def swiglu_grad_kernel(
    a: pl.Tensor[[16, 128], pl.DT_FP32],
    b: pl.Tensor[[16, 128], pl.DT_FP32],
    grad: pl.Tensor[[16, 128], pl.DT_FP32],
    da_out: pl.Tensor[[16, 128], pl.DT_FP32],
    db_out: pl.Tensor[[16, 128], pl.DT_FP32],
):
    with pl.section_vector():
        tile_type = pl.TileType(
            shape=[16, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[16, 128]
        )
        a_tile = _tile(tile_type, 0x00000)
        b_tile = _tile(tile_type, 0x02000)
        grad_tile = _tile(tile_type, 0x04000)
        da = _tile(tile_type, 0x06000)
        db = _tile(tile_type, 0x08000)
        sigmoid = _tile(tile_type, 0x0A000)
        grad_s = _tile(tile_type, 0x0C000)
        grad_tmp = _tile(tile_type, 0x0E000)
        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.load(grad_tile, grad, [0, 0])
        swiglu_grad(a_tile, b_tile, grad_tile, sigmoid, grad_s, grad_tmp, da, db)
        pl.store(da_out, da, [0, 0])
        pl.store(db_out, db, [0, 0])


@pytest.mark.soc("950")
def test_simd_tile_swiglu_grad(expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    generator = torch.Generator().manual_seed(23)
    a, b, grad = [torch.randn(16, 128, generator=generator) for _ in range(3)]
    da = torch.empty_like(a, device="npu")
    db = torch.empty_like(a, device="npu")
    with expect_pgo_vf():
        swiglu_grad_kernel[1](a.npu(), b.npu(), grad.npu(), da, db)
        s = torch.sigmoid(a)
        g = grad * s
        expected_db = g * a
        expected_da = b * (g - expected_db * (s - 1))
        torch.testing.assert_close(da.cpu(), expected_da, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(db.cpu(), expected_db, rtol=1e-3, atol=1e-4)


@pl.jit(auto_mutex=True)
def swiglu_grad_dynamic_valid_shape_kernel(
    a: pl.Tensor[[31, 326], pl.DT_FP32],
    b: pl.Tensor[[31, 326], pl.DT_FP32],
    grad: pl.Tensor[[31, 326], pl.DT_FP32],
    da_out: pl.Tensor[[31, 326], pl.DT_FP32],
    db_out: pl.Tensor[[31, 326], pl.DT_FP32],
):
    with pl.section_vector():
        tile_type = pl.TileType(
            shape=[16, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        )
        a_tile = _tile(tile_type, 0x00000)
        b_tile = _tile(tile_type, 0x02000)
        grad_tile = _tile(tile_type, 0x04000)
        da = _tile(tile_type, 0x06000)
        db = _tile(tile_type, 0x08000)
        sigmoid = _tile(tile_type, 0x0A000)
        grad_s = _tile(tile_type, 0x0C000)
        grad_tmp = _tile(tile_type, 0x0E000)
        for row in pl.range(0, 31, 16):
            for col in pl.range(0, 326, 128):
                valid_rows = pl.min(16, 31 - row)
                valid_cols = pl.min(128, 326 - col)
                pl.set_validshape(a_tile, [valid_rows, valid_cols])
                pl.set_validshape(b_tile, [valid_rows, valid_cols])
                pl.set_validshape(grad_tile, [valid_rows, valid_cols])
                pl.set_validshape(da, [valid_rows, valid_cols])
                pl.set_validshape(db, [valid_rows, valid_cols])
                pl.set_validshape(sigmoid, [valid_rows, valid_cols])
                pl.set_validshape(grad_s, [valid_rows, valid_cols])
                pl.set_validshape(grad_tmp, [valid_rows, valid_cols])
                pl.load(a_tile, a, [row, col])
                pl.load(b_tile, b, [row, col])
                pl.load(grad_tile, grad, [row, col])
                swiglu_grad(a_tile, b_tile, grad_tile, sigmoid, grad_s, grad_tmp, da, db)
                pl.store(da_out, da, [row, col])
                pl.store(db_out, db, [row, col])


@pytest.mark.soc("950")
def test_simd_tile_swiglu_grad_dynamic_valid_shape(expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    generator = torch.Generator().manual_seed(23)
    a, b, grad = [torch.randn(31, 326, generator=generator) for _ in range(3)]
    da = torch.full_like(a, float("nan"), device="npu")
    db = torch.full_like(a, float("nan"), device="npu")
    with expect_pgo_vf():
        swiglu_grad_dynamic_valid_shape_kernel[1](a.npu(), b.npu(), grad.npu(), da, db)
        s = torch.sigmoid(a)
        expected_da = grad * b * (s + a * s * (1 - s))
        expected_db = grad * a * s
        torch.testing.assert_close(da.cpu(), expected_da, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(db.cpu(), expected_db, rtol=1e-3, atol=1e-4)
