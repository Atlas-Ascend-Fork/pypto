# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for the SIMT remainder interface with Tile operands."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def remainder_tile(lhs_tile, rhs_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.remainder(lhs_tile[0, tid], rhs_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_remainder_tile(
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
        remainder_tile[ELEMENTS](lhs_tile, rhs_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


def remainder_golden(lhs, rhs):
    values = []
    for x, y in zip(lhs.reshape(-1), rhs.reshape(-1)):
        x_value = float(x)
        y_value = float(y)
        if math.isnan(x_value) or math.isnan(y_value) or y_value == 0.0 or math.isinf(x_value):
            values.append(float("nan"))
        else:
            values.append(math.remainder(x_value, y_value))
    return torch.tensor(values, dtype=torch.float32).reshape_as(lhs)


@pytest.mark.soc("950")
def test_remainder_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([5.5, -5.5, 7.0, 6.0, 1.0, -3.0, 8.5, -9.5] * 8)
    rhs = torch.tensor([2.0, 2.0, 2.0, 4.0, -2.0, -4.0, 3.0, 3.0] * 8)
    lhs[:8] = torch.tensor([1.0, float("inf"), float("nan"), 2.0, -0.0, 5.0, -5.0, 3.0])
    rhs[:8] = torch.tensor([0.0, 2.0, 2.0, float("nan"), 3.0, float("inf"), float("-inf"), 2.0])
    lhs_source = lhs.to(torch.float32).reshape(1, -1)
    rhs_source = rhs.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(lhs_source, device=ST_DEVICE)
    simt_remainder_tile(lhs_source.to(ST_DEVICE), rhs_source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), remainder_golden(lhs_source, rhs_source), rtol=1e-05, atol=1e-06, equal_nan=True
    )
