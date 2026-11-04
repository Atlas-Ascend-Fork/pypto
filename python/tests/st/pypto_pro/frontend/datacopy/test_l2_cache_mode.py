# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

"""Compile and run Vector and Cube GM transfers with explicit L2 cache modes."""

import glob
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.jit(auto_mutex=True)
def l2_cache_mode_vector_kernel(
    src: pl.Tensor[[1, 64], pl.DT_FP32],
    dst: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
    with pl.section_vector():
        tile = group.current()
        pl.load(tile, src, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NormalFirstVictim)
        pl.store(dst, tile, [0, 0], l2_cache_mode=pl.TStoreL2Hint.NotAllocClean)


@pl.jit(auto_mutex=True)
def l2_cache_mode_cube_kernel(
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
        pl.load(lhs_l1_tile, lhs, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)
        pl.load(rhs_l1_tile, rhs, [0, 0], l2_cache_mode=pl.TLoadL2Hint.NormalLastVictim)
        pl.move(lhs_l0_tile, lhs_l1_tile)
        pl.move(rhs_l0_tile, rhs_l1_tile)
        pl.matmul(acc_tile, lhs_l0_tile, rhs_l0_tile)
        pl.store(dst, acc_tile, [0, 0], l2_cache_mode=pl.TStoreL2Hint.NormalPersistent)


def _generated_kernel_source(kernel_name):
    generated = glob.glob(os.path.join("build", f"*{kernel_name}*", "**", "kernel.cpp"), recursive=True)
    assert generated, f"no generated kernel.cpp found for {kernel_name}"
    with open(max(generated, key=os.path.getmtime), encoding="utf-8") as handle:
        return handle.read()


@pytest.mark.soc("950")
def test_l2_cache_mode_vector_round_trip():
    torch.npu.set_device(ST_DEVICE)
    src = torch.arange(64, device=ST_DEVICE, dtype=torch.float32).reshape(1, 64)
    dst = torch.zeros_like(src)

    l2_cache_mode_vector_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst, src)
    source = _generated_kernel_source("l2_cache_mode_vector_kernel")
    assert "TLOAD<TLoadL2Hint::NormalFirstVictim>" in source
    assert "TSTORE<TStoreL2Hint::NotAllocClean" in source


@pytest.mark.soc("950")
def test_l2_cache_mode_cube_round_trip():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.arange(64 * 64, device=ST_DEVICE, dtype=torch.float16).reshape(64, 64) % 17
    rhs = torch.eye(64, device=ST_DEVICE, dtype=torch.float16)
    dst = torch.zeros((64, 64), device=ST_DEVICE, dtype=torch.float32)

    l2_cache_mode_cube_kernel[None, 1](lhs, rhs, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst, lhs.float())
    source = _generated_kernel_source("l2_cache_mode_cube_kernel")
    assert "TLOAD<TLoadL2Hint::NotAllocKeep>" in source
    assert "TLOAD<TLoadL2Hint::NormalLastVictim>" in source
    assert "TSTORE<TStoreL2Hint::NormalPersistent" in source
