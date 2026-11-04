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
"""End-to-end coverage for ICache preload on AIV and AIC."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 - installs torch.npu

import pypto

DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
DEVICE = f"npu:{DEVICE_ID}"


@pl.jit(auto_mutex=True)
def icache_preload_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_out_group = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        tile_x = tile_x_group.current()
        tile_out = tile_out_group.current()
        pl.system.icache_preload(2)
        pl.load(tile_x, x, [0, 0])
        pl.add(tile_out, tile_x, tile_x)
        pl.store(out, tile_out, [0, 0])


@pl.jit(auto_mutex=True)
def icache_preload_cube_kernel(
    lhs: pl.Tensor[[64, 64], pl.DT_FP16],
    rhs: pl.Tensor[[64, 64], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP32],
):
    lhs_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    rhs_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x4000,
        mutex_ids=[1],
    )
    lhs_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    rhs_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        lhs_l1_tile = lhs_l1.current()
        rhs_l1_tile = rhs_l1.current()
        lhs_l0_tile = lhs_l0.current()
        rhs_l0_tile = rhs_l0.current()
        acc_tile = acc.current()
        pl.system.icache_preload(2)
        pl.load(lhs_l1_tile, lhs, [0, 0])
        pl.load(rhs_l1_tile, rhs, [0, 0])
        pl.move(lhs_l0_tile, lhs_l1_tile)
        pl.move(rhs_l0_tile, rhs_l1_tile)
        pl.matmul(acc_tile, lhs_l0_tile, rhs_l0_tile)
        pl.store(dst, acc_tile, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_icache_preload_aiv_execution():
    torch.npu.set_device(DEVICE)
    torch.manual_seed(0)
    x = torch.randn((64, 64), device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    icache_preload_kernel[None, 1](x, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + x, rtol=1e-4, atol=1e-4)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_icache_preload_aic_execution():
    torch.npu.set_device(DEVICE)
    lhs = torch.arange(64 * 64, device=DEVICE, dtype=torch.float16).reshape(64, 64) % 17
    rhs = torch.eye(64, device=DEVICE, dtype=torch.float16)
    dst = torch.empty((64, 64), device=DEVICE, dtype=torch.float32)

    icache_preload_cube_kernel[None, 1](lhs, rhs, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst, lhs.float())
