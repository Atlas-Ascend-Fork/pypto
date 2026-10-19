# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for the SIMT mul24 interface with Tile operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def mul24_tile(lhs_i32_tile, rhs_i32_tile, lhs_u32_tile, rhs_u32_tile, out_i32_tile, out_u32_tile):
    tid = pl.simt.linear_thread_idx()
    out_i32_tile[0, tid] = pl.simt.mul24(lhs_i32_tile[0, tid], rhs_i32_tile[0, tid])
    out_u32_tile[0, tid] = pl.simt.mul24(lhs_u32_tile[0, tid], rhs_u32_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_mul24_tile(
    lhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
):
    lhs_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_i32_tile = lhs_i32_tile_group.current()
    rhs_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_i32_tile = rhs_i32_tile_group.current()
    lhs_u32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_u32_tile = lhs_u32_tile_group.current()
    rhs_u32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_u32_tile = rhs_u32_tile_group.current()
    out_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_i32_tile = out_i32_tile_group.current()
    out_u32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_u32_tile = out_u32_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_i32_tile, lhs_i32, [0, 0])
        pl.load(rhs_i32_tile, rhs_i32, [0, 0])
        pl.load(lhs_u32_tile, lhs_u32, [0, 0])
        pl.load(rhs_u32_tile, rhs_u32, [0, 0])
        mul24_tile[ELEMENTS](
            lhs_i32_tile,
            rhs_i32_tile,
            lhs_u32_tile,
            rhs_u32_tile,
            out_i32_tile,
            out_u32_tile,
        )
        pl.store(out_i32, out_i32_tile, [0, 0])
        pl.store(out_u32, out_u32_tile, [0, 0])


def signed_mul24_golden(lhs, rhs):
    def signed24(value):
        value &= 0xFFFFFF
        return value - (1 << 24) if value & 0x800000 else value

    result = (signed24(lhs) * signed24(rhs)) & 0xFFFFFFFF
    return result - (1 << 32) if result & 0x80000000 else result


@pytest.mark.soc("950")
def test_mul24_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    signed_lhs = [-(1 << 31), -8388609, -8388608, -1, 0, 1, 8388607, (1 << 31) - 1] * 8
    signed_rhs = [-1, 2, -3, 4, 5, -6, 7, -8] * 8
    unsigned_lhs = list(range(ELEMENTS))
    unsigned_rhs = list(reversed(range(ELEMENTS)))
    inputs = (
        torch.tensor(signed_lhs, dtype=torch.int32).reshape(1, ELEMENTS),
        torch.tensor(signed_rhs, dtype=torch.int32).reshape(1, ELEMENTS),
        torch.tensor(unsigned_lhs, dtype=torch.uint32).reshape(1, ELEMENTS),
        torch.tensor(unsigned_rhs, dtype=torch.uint32).reshape(1, ELEMENTS),
    )
    out_i32 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    out_u32 = torch.empty((1, ELEMENTS), dtype=torch.uint32, device=ST_DEVICE)
    simt_mul24_tile(*(value.to(ST_DEVICE) for value in inputs), out_i32, out_u32)
    torch.npu.synchronize()
    expected_i32 = torch.tensor(
        [signed_mul24_golden(x, y) for x, y in zip(signed_lhs, signed_rhs)], dtype=torch.int32
    ).reshape(1, ELEMENTS)
    expected_u32 = torch.tensor([x * y for x, y in zip(unsigned_lhs, unsigned_rhs)], dtype=torch.uint32).reshape(
        1, ELEMENTS
    )
    torch.testing.assert_close(out_i32.cpu(), expected_i32, rtol=0, atol=0)
    torch.testing.assert_close(out_u32.cpu(), expected_u32, rtol=0, atol=0)
