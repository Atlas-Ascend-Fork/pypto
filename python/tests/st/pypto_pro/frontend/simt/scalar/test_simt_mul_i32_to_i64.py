# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for the SIMT mul_i32_to_i64 interface with Tile operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def mul_i32_to_i64_tile(lhs_i32_tile, rhs_i32_tile, lhs_u32_tile, rhs_u32_tile, out_i64_tile, out_u64_tile):
    tid = pl.simt.linear_thread_idx()
    out_i64_tile[0, tid] = pl.simt.mul_i32_to_i64(lhs_i32_tile[0, tid], rhs_i32_tile[0, tid])
    out_u64_tile[0, tid] = pl.simt.mul_i32_to_i64(lhs_u32_tile[0, tid], rhs_u32_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_mul_i32_to_i64_tile(
    lhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
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
    out_i64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_i64_tile = out_i64_tile_group.current()
    out_u64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_u64_tile = out_u64_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_i32_tile, lhs_i32, [0, 0])
        pl.load(rhs_i32_tile, rhs_i32, [0, 0])
        pl.load(lhs_u32_tile, lhs_u32, [0, 0])
        pl.load(rhs_u32_tile, rhs_u32, [0, 0])
        mul_i32_to_i64_tile[ELEMENTS](
            lhs_i32_tile,
            rhs_i32_tile,
            lhs_u32_tile,
            rhs_u32_tile,
            out_i64_tile,
            out_u64_tile,
        )
        pl.store(out_i64, out_i64_tile, [0, 0])
        pl.store(out_u64, out_u64_tile, [0, 0])


@pytest.mark.soc("950")
def test_mul_i32_to_i64_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    signed_lhs = torch.tensor([-(1 << 31), -100000, -1, 0, 1, 17, 100000, (1 << 31) - 1] * 8, dtype=torch.int32)
    signed_rhs = torch.tensor([-1, 2, -3, 4, 5, -6, 7, -8] * 8, dtype=torch.int32)
    unsigned_lhs_values = list(range(ELEMENTS))
    unsigned_rhs_values = list(reversed(range(ELEMENTS)))
    unsigned_lhs = torch.tensor(unsigned_lhs_values, dtype=torch.uint32)
    unsigned_rhs = torch.tensor(unsigned_rhs_values, dtype=torch.uint32)
    inputs = tuple(value.reshape(1, ELEMENTS) for value in (signed_lhs, signed_rhs, unsigned_lhs, unsigned_rhs))
    out_i64 = torch.empty((1, ELEMENTS), dtype=torch.int64, device=ST_DEVICE)
    out_u64 = torch.empty((1, ELEMENTS), dtype=torch.uint64, device=ST_DEVICE)
    simt_mul_i32_to_i64_tile(*(value.to(ST_DEVICE) for value in inputs), out_i64, out_u64)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out_i64.cpu(), (signed_lhs.to(torch.int64) * signed_rhs).reshape(1, ELEMENTS), rtol=0, atol=0
    )
    expected_u64 = torch.tensor(
        [lhs * rhs for lhs, rhs in zip(unsigned_lhs_values, unsigned_rhs_values)], dtype=torch.uint64
    ).reshape(1, ELEMENTS)
    torch.testing.assert_close(out_u64.cpu(), expected_u64, rtol=0, atol=0)
