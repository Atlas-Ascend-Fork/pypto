# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for the SIMT fdim interface with Tile operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def fdim_tile(lhs_tile, rhs_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.fdim(lhs_tile[0, tid], rhs_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_fdim_tile(
    lhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    lhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_tile = lhs_tile_group.current()
    rhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_tile = rhs_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_tile, lhs, [0, 0])
        pl.load(rhs_tile, rhs, [0, 0])
        fdim_tile[ELEMENTS](lhs_tile, rhs_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_fdim_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.linspace(-8.0, 8.0, ELEMENTS)
    rhs = torch.linspace(4.0, -4.0, ELEMENTS)
    lhs_source = lhs.to(torch.float32).reshape(1, -1)
    rhs_source = rhs.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(lhs_source, device=ST_DEVICE)
    simt_fdim_tile(lhs_source.to(ST_DEVICE), rhs_source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(),
        (lambda x, y: torch.clamp(x - y, min=0.0))(lhs_source, rhs_source),
        rtol=1e-05,
        atol=1e-06,
        equal_nan=True,
    )
