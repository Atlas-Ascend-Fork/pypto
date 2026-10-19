# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar integer arithmetic interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ------------------------------------------------------- hadd --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def hadd_tile(lhs_i32_tile, rhs_i32_tile, lhs_u32_tile, rhs_u32_tile, out_i32_tile, out_u32_tile):
    tid = pl.simt.linear_thread_idx()
    out_i32_tile[0, tid] = pl.simt.hadd(lhs_i32_tile[0, tid], rhs_i32_tile[0, tid])
    out_u32_tile[0, tid] = pl.simt.hadd(lhs_u32_tile[0, tid], rhs_u32_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_hadd_tile(
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
        hadd_tile[ELEMENTS](
            lhs_i32_tile,
            rhs_i32_tile,
            lhs_u32_tile,
            rhs_u32_tile,
            out_i32_tile,
            out_u32_tile,
        )
        pl.store(out_i32, out_i32_tile, [0, 0])
        pl.store(out_u32, out_u32_tile, [0, 0])


@pytest.mark.soc("950")
def test_hadd_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    signed_lhs = [-100, -50, -1, 0, 1, 50, 100, 200] * 8
    signed_rhs = [10, -20, 30, -40, 50, -60, 70, -80] * 8
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
    simt_hadd_tile(*(value.to(ST_DEVICE) for value in inputs), out_i32, out_u32)
    torch.npu.synchronize()
    expected_i32 = torch.tensor([(x + y) // 2 for x, y in zip(signed_lhs, signed_rhs)], dtype=torch.int32)
    expected_u32 = torch.tensor([(x + y) // 2 for x, y in zip(unsigned_lhs, unsigned_rhs)], dtype=torch.uint32)
    torch.testing.assert_close(out_i32.cpu(), expected_i32.reshape(1, ELEMENTS), rtol=0, atol=0)
    torch.testing.assert_close(out_u32.cpu(), expected_u32.reshape(1, ELEMENTS), rtol=0, atol=0)

# ------------------------------------------------------- rhadd -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def rhadd_tile(lhs_i32_tile, rhs_i32_tile, lhs_u32_tile, rhs_u32_tile, out_i32_tile, out_u32_tile):
    tid = pl.simt.linear_thread_idx()
    out_i32_tile[0, tid] = pl.simt.rhadd(lhs_i32_tile[0, tid], rhs_i32_tile[0, tid])
    out_u32_tile[0, tid] = pl.simt.rhadd(lhs_u32_tile[0, tid], rhs_u32_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_rhadd_tile(
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
        rhadd_tile[ELEMENTS](
            lhs_i32_tile,
            rhs_i32_tile,
            lhs_u32_tile,
            rhs_u32_tile,
            out_i32_tile,
            out_u32_tile,
        )
        pl.store(out_i32, out_i32_tile, [0, 0])
        pl.store(out_u32, out_u32_tile, [0, 0])


@pytest.mark.soc("950")
def test_rhadd_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    signed_lhs = [-100, -50, -1, 0, 1, 50, 100, 200] * 8
    signed_rhs = [10, -20, 30, -40, 50, -60, 70, -80] * 8
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
    simt_rhadd_tile(*(value.to(ST_DEVICE) for value in inputs), out_i32, out_u32)
    torch.npu.synchronize()
    expected_i32 = torch.tensor([(x + y + 1) // 2 for x, y in zip(signed_lhs, signed_rhs)], dtype=torch.int32)
    expected_u32 = torch.tensor([(x + y + 1) // 2 for x, y in zip(unsigned_lhs, unsigned_rhs)], dtype=torch.uint32)
    torch.testing.assert_close(out_i32.cpu(), expected_i32.reshape(1, ELEMENTS), rtol=0, atol=0)
    torch.testing.assert_close(out_u32.cpu(), expected_u32.reshape(1, ELEMENTS), rtol=0, atol=0)

# -------------------------------------------------------- sad --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def sad_tile(lhs_i32_tile, rhs_i32_tile, lhs_u32_tile, rhs_u32_tile, addend_tile, out_i32_tile, out_u32_tile):
    tid = pl.simt.linear_thread_idx()
    out_i32_tile[0, tid] = pl.simt.sad(lhs_i32_tile[0, tid], rhs_i32_tile[0, tid], addend_tile[0, tid])
    out_u32_tile[0, tid] = pl.simt.sad(lhs_u32_tile[0, tid], rhs_u32_tile[0, tid], addend_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_sad_tile(
    lhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    addend: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_i32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
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
    addend_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    addend_tile = addend_tile_group.current()
    out_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_i32_tile = out_i32_tile_group.current()
    out_u32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    out_u32_tile = out_u32_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_i32_tile, lhs_i32, [0, 0])
        pl.load(rhs_i32_tile, rhs_i32, [0, 0])
        pl.load(lhs_u32_tile, lhs_u32, [0, 0])
        pl.load(rhs_u32_tile, rhs_u32, [0, 0])
        pl.load(addend_tile, addend, [0, 0])
        sad_tile[ELEMENTS](
            lhs_i32_tile,
            rhs_i32_tile,
            lhs_u32_tile,
            rhs_u32_tile,
            addend_tile,
            out_i32_tile,
            out_u32_tile,
        )
        pl.store(out_i32, out_i32_tile, [0, 0])
        pl.store(out_u32, out_u32_tile, [0, 0])


@pytest.mark.soc("950")
def test_sad_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    signed_lhs = torch.tensor([-100, -50, -1, 0, 1, 50, 100, 200] * 8, dtype=torch.int32)
    signed_rhs = torch.tensor([10, -20, 30, -40, 50, -60, 70, -80] * 8, dtype=torch.int32)
    unsigned_lhs = torch.tensor(list(range(ELEMENTS)), dtype=torch.uint32)
    unsigned_rhs = torch.tensor(list(reversed(range(ELEMENTS))), dtype=torch.uint32)
    addend = torch.tensor(list(range(ELEMENTS)), dtype=torch.uint32)
    inputs = tuple(value.reshape(1, ELEMENTS) for value in (signed_lhs, signed_rhs, unsigned_lhs, unsigned_rhs, addend))
    outputs = tuple(torch.empty((1, ELEMENTS), dtype=torch.uint32, device=ST_DEVICE) for _ in range(2))
    simt_sad_tile(*(value.to(ST_DEVICE) for value in inputs), *outputs)
    torch.npu.synchronize()
    expected_i32 = torch.tensor(
        [abs(int(x) - int(y)) + int(z) for x, y, z in zip(signed_lhs, signed_rhs, addend)], dtype=torch.uint32
    )
    expected_u32 = torch.tensor(
        [abs(int(x) - int(y)) + int(z) for x, y, z in zip(unsigned_lhs, unsigned_rhs, addend)], dtype=torch.uint32
    )
    torch.testing.assert_close(outputs[0].cpu(), expected_i32.reshape(1, ELEMENTS), rtol=0, atol=0)
    torch.testing.assert_close(outputs[1].cpu(), expected_u32.reshape(1, ELEMENTS), rtol=0, atol=0)
