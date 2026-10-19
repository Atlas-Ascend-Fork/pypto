# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar bit-manipulation interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ------------------------------------------------------- brev --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def brev_tile(src_u32_tile, src_u64_tile, out_u32_tile, out_u64_tile):
    tid = pl.simt.linear_thread_idx()
    out_u32_tile[0, tid] = pl.simt.brev(src_u32_tile[0, tid])
    out_u64_tile[0, tid] = pl.simt.brev(src_u64_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_brev_tile(
    src_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    src_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_u32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_u64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    src_u32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_u32_tile = src_u32_tile_group.current()
    src_u64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_u64_tile = src_u64_tile_group.current()
    out_u32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_u32_tile = out_u32_tile_group.current()
    out_u64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_u64_tile = out_u64_tile_group.current()
    with pl.section_vector():
        pl.load(src_u32_tile, src_u32, [0, 0])
        pl.load(src_u64_tile, src_u64, [0, 0])
        brev_tile[ELEMENTS](src_u32_tile, src_u64_tile, out_u32_tile, out_u64_tile)
        pl.store(out_u32, out_u32_tile, [0, 0])
        pl.store(out_u64, out_u64_tile, [0, 0])


def brev_golden(value, bits):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


@pytest.mark.soc("950")
def test_brev_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values_u32 = list(range(ELEMENTS))
    values_u64 = [0, 1, 2, 3, 1 << 31, 1 << 32, 1 << 62, (1 << 64) - 1] * 8
    src_u32 = torch.tensor(values_u32, dtype=torch.uint32).reshape(1, ELEMENTS)
    src_u64 = torch.tensor(values_u64, dtype=torch.uint64).reshape(1, ELEMENTS)
    out_u32 = torch.empty_like(src_u32, device=ST_DEVICE)
    out_u64 = torch.empty_like(src_u64, device=ST_DEVICE)
    simt_brev_tile(src_u32.to(ST_DEVICE), src_u64.to(ST_DEVICE), out_u32, out_u64)
    torch.npu.synchronize()
    expected_u32 = torch.tensor([brev_golden(value, 32) for value in values_u32], dtype=torch.uint32)
    expected_u64 = torch.tensor([brev_golden(value, 64) for value in values_u64], dtype=torch.uint64)
    torch.testing.assert_close(out_u32.cpu(), expected_u32.reshape(1, ELEMENTS), rtol=0, atol=0)
    torch.testing.assert_close(out_u64.cpu(), expected_u64.reshape(1, ELEMENTS), rtol=0, atol=0)

# ----------------------------------------------------- byte_perm -----------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def byte_perm_tile(lhs_tile, rhs_tile, selector_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.byte_perm(lhs_tile[0, tid], rhs_tile[0, tid], selector_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_byte_perm_tile(
    lhs: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    selector: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
):
    lhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_tile = lhs_tile_group.current()
    rhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_tile = rhs_tile_group.current()
    selector_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    selector_tile = selector_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_tile, lhs, [0, 0])
        pl.load(rhs_tile, rhs, [0, 0])
        pl.load(selector_tile, selector, [0, 0])
        byte_perm_tile[ELEMENTS](lhs_tile, rhs_tile, selector_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


def byte_perm_golden(lhs, rhs, selector):
    source = lhs | (rhs << 32)
    return sum(((source >> (8 * ((selector >> (4 * index)) & 7))) & 0xFF) << (8 * index) for index in range(4))


@pytest.mark.soc("950")
def test_byte_perm_uint32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs_values = list(range(ELEMENTS))
    rhs_values = list(reversed(range(ELEMENTS)))
    selectors = [291, 17767, 17, 8755, 17493, 26231, 30292, 12816] * 8
    lhs = torch.tensor(lhs_values, dtype=torch.uint32).reshape(1, ELEMENTS)
    rhs = torch.tensor(rhs_values, dtype=torch.uint32).reshape(1, ELEMENTS)
    selector = torch.tensor(selectors, dtype=torch.uint32).reshape(1, ELEMENTS)
    output = torch.empty_like(lhs, device=ST_DEVICE)
    simt_byte_perm_tile(lhs.to(ST_DEVICE), rhs.to(ST_DEVICE), selector.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.tensor(
        [byte_perm_golden(x, y, s) for x, y, s in zip(lhs_values, rhs_values, selectors)], dtype=torch.uint32
    ).reshape(1, ELEMENTS)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

# -------------------------------------------------------- clz --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def clz_tile(src_i32_tile, src_i64_tile, out_i32_tile, out_i64_tile):
    tid = pl.simt.linear_thread_idx()
    out_i32_tile[0, tid] = pl.simt.clz(src_i32_tile[0, tid])
    out_i64_tile[0, tid] = pl.simt.clz(src_i64_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_clz_tile(
    src_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    src_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_i32_tile = src_i32_tile_group.current()
    src_i64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_i64_tile = src_i64_tile_group.current()
    out_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_i32_tile = out_i32_tile_group.current()
    out_i64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_i64_tile = out_i64_tile_group.current()
    with pl.section_vector():
        pl.load(src_i32_tile, src_i32, [0, 0])
        pl.load(src_i64_tile, src_i64, [0, 0])
        clz_tile[ELEMENTS](src_i32_tile, src_i64_tile, out_i32_tile, out_i64_tile)
        pl.store(out_i32, out_i32_tile, [0, 0])
        pl.store(out_i64, out_i64_tile, [0, 0])


def clz_golden(value, bits):
    return bits - (value & ((1 << bits) - 1)).bit_length()


@pytest.mark.soc("950")
def test_clz_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values_i32 = [0, 1, -1, 2, -2, 1 << 16, -(1 << 16), -(1 << 31)] * 8
    values_i64 = [0, 1, -1, 2, -2, 1 << 32, -(1 << 32), -(1 << 63)] * 8
    src_i32 = torch.tensor(values_i32, dtype=torch.int32).reshape(1, ELEMENTS)
    src_i64 = torch.tensor(values_i64, dtype=torch.int64).reshape(1, ELEMENTS)
    out_i32 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    out_i64 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    simt_clz_tile(src_i32.to(ST_DEVICE), src_i64.to(ST_DEVICE), out_i32, out_i64)
    torch.npu.synchronize()
    expected_i32 = torch.tensor([clz_golden(value, 32) for value in values_i32], dtype=torch.int32)
    expected_i64 = torch.tensor([clz_golden(value, 64) for value in values_i64], dtype=torch.int32)
    torch.testing.assert_close(out_i32.cpu(), expected_i32.reshape(1, ELEMENTS), rtol=0, atol=0)
    torch.testing.assert_close(out_i64.cpu(), expected_i64.reshape(1, ELEMENTS), rtol=0, atol=0)

# -------------------------------------------------------- ffs --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def ffs_tile(src_i32_tile, src_i64_tile, out_i32_tile, out_i64_tile):
    tid = pl.simt.linear_thread_idx()
    out_i32_tile[0, tid] = pl.simt.ffs(src_i32_tile[0, tid])
    out_i64_tile[0, tid] = pl.simt.ffs(src_i64_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_ffs_tile(
    src_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_i32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_i64: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
):
    src_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_i32_tile = src_i32_tile_group.current()
    src_i64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_i64_tile = src_i64_tile_group.current()
    out_i32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_i32_tile = out_i32_tile_group.current()
    out_i64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_i64_tile = out_i64_tile_group.current()
    with pl.section_vector():
        pl.load(src_i32_tile, src_i32, [0, 0])
        pl.load(src_i64_tile, src_i64, [0, 0])
        ffs_tile[ELEMENTS](src_i32_tile, src_i64_tile, out_i32_tile, out_i64_tile)
        pl.store(out_i32, out_i32_tile, [0, 0])
        pl.store(out_i64, out_i64_tile, [0, 0])


def ffs_golden(value, bits):
    value &= (1 << bits) - 1
    return 0 if value == 0 else (value & -value).bit_length()


@pytest.mark.soc("950")
def test_ffs_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values_i32 = [0, 1, -1, 2, -2, 1 << 16, -(1 << 16), -(1 << 31)] * 8
    values_i64 = [0, 1, -1, 2, -2, 1 << 32, -(1 << 32), -(1 << 63)] * 8
    src_i32 = torch.tensor(values_i32, dtype=torch.int32).reshape(1, ELEMENTS)
    src_i64 = torch.tensor(values_i64, dtype=torch.int64).reshape(1, ELEMENTS)
    out_i32 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    out_i64 = torch.empty((1, ELEMENTS), dtype=torch.int32, device=ST_DEVICE)
    simt_ffs_tile(src_i32.to(ST_DEVICE), src_i64.to(ST_DEVICE), out_i32, out_i64)
    torch.npu.synchronize()
    expected_i32 = torch.tensor([ffs_golden(value, 32) for value in values_i32], dtype=torch.int32)
    expected_i64 = torch.tensor([ffs_golden(value, 64) for value in values_i64], dtype=torch.int32)
    torch.testing.assert_close(out_i32.cpu(), expected_i32.reshape(1, ELEMENTS), rtol=0, atol=0)
    torch.testing.assert_close(out_i64.cpu(), expected_i64.reshape(1, ELEMENTS), rtol=0, atol=0)
