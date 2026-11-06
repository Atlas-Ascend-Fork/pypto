# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 system tests for the SIMT nextafter interface with Tile operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def nextafter_tile(lhs_tile, rhs_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.nextafter(lhs_tile[0, tid], rhs_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_nextafter_tile(
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
        nextafter_tile[ELEMENTS](lhs_tile, rhs_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_nextafter_fp32_with_tile_operands():
    lhs = torch.tensor([-1.0, -0.0, 0.0, 1.0, 2.0, -2.0, float("inf"), float("-inf")] * 8)
    rhs = torch.tensor([0.0, -1.0, 1.0, 2.0, 1.0, -3.0, 0.0, 0.0] * 8)
    torch.npu.set_device(ST_DEVICE)
    lhs = lhs.reshape(1, ELEMENTS)
    rhs = rhs.reshape(1, ELEMENTS)
    lhs[0, 8:16] = torch.tensor([-0.0, 0.0, -0.0, 0.0, 1.0, -1.0, float("inf"), float("-inf")])
    rhs[0, 8:16] = torch.tensor([0.0, -0.0, -1.0, 1.0, 1.0, -1.0, float("inf"), float("-inf")])
    nan_value = torch.tensor([0x7FC12345], dtype=torch.int32).view(torch.float32)[0]
    lhs[0, 16], rhs[0, 16] = nan_value, 1.0
    lhs[0, 17], rhs[0, 17] = 1.0, nan_value
    output = torch.empty_like(lhs, device=ST_DEVICE)
    simt_nextafter_tile(lhs.to(ST_DEVICE), rhs.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.nextafter(lhs, rhs)
    torch.testing.assert_close(actual[:, :8], expected[:, :8], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[:, 18:], expected[:, 18:], rtol=0.0, atol=0.0)
    expected_bits = torch.tensor(
        [-2147483648, 0, -2147483647, 1, 0x3F800000, -1082130432, 0x7F800000, -8388608],
        dtype=torch.int32,
    )
    assert torch.equal(actual[0, 8:16].view(torch.int32), expected_bits)
    assert torch.equal(actual[0, 16:18].view(torch.int32), torch.full((2,), 0x7FFFFFFF, dtype=torch.int32))
