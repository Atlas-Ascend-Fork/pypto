# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 system tests for SIMT scalar minimum and maximum interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# --------------------------------------------------- min ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def min_float_dtypes(
    lhs_fp16,
    rhs_fp16,
    lhs_bf16,
    rhs_bf16,
    lhs_fp32,
    rhs_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.min(lhs_fp16[0, tid], rhs_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.min(lhs_bf16[0, tid], rhs_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.min(lhs_fp32[0, tid], rhs_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_min_float_dtypes(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    lhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    lhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp16_tile = lhs_fp16_tile_group.current()
    rhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp16_tile = rhs_fp16_tile_group.current()
    lhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_bf16_tile = lhs_bf16_tile_group.current()
    rhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_bf16_tile = rhs_bf16_tile_group.current()
    lhs_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp32_tile = lhs_fp32_tile_group.current()
    rhs_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp32_tile = rhs_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        pl.load(lhs_fp32_tile, lhs_fp32, [0, 0])
        pl.load(rhs_fp32_tile, rhs_fp32, [0, 0])
        min_float_dtypes[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            lhs_fp32_tile,
            rhs_fp32_tile,
            out_fp16_tile,
            out_bf16_tile,
            out_fp32_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])

@pytest.mark.soc("950")
def test_min_all_supported_float_dtypes():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.linspace(-4.0, 4.0, ELEMENTS)
    rhs = torch.linspace(2.0, -2.0, ELEMENTS)
    lhs[:7] = torch.tensor([float("nan"), 1.0, float("nan"), -0.0, 0.0, float("inf"), float("-inf")])
    rhs[:7] = torch.tensor([1.0, float("nan"), float("nan"), 0.0, -0.0, 2.0, 2.0])
    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple(
        (lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    outputs = tuple(torch.empty_like(lhs_source, device=ST_DEVICE) for lhs_source, _ in operand_pairs)
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_min_float_dtypes(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = torch.fmin(lhs_source.float(), rhs_source.float()).to(lhs_source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def min_signed_dtypes(
    lhs_int8,
    rhs_int8,
    lhs_int16,
    rhs_int16,
    lhs_int32,
    rhs_int32,
    lhs_int64,
    rhs_int64,
    out_int8,
    out_int16,
    out_int32,
    out_int64,
):
    tid = pl.simt.linear_thread_idx()
    out_int8[0, tid] = pl.simt.min(lhs_int8[0, tid], rhs_int8[0, tid])
    out_int16[0, tid] = pl.simt.min(lhs_int16[0, tid], rhs_int16[0, tid])
    out_int32[0, tid] = pl.simt.min(lhs_int32[0, tid], rhs_int32[0, tid])
    out_int64[0, tid] = pl.simt.min(lhs_int64[0, tid], rhs_int64[0, tid])

@pl.jit(auto_mutex=True)
def simt_min_signed_dtypes(
    lhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    rhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    lhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    rhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    lhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    rhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    lhs_int8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int8_tile = lhs_int8_tile_group.current()
    rhs_int8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int8_tile = rhs_int8_tile_group.current()
    lhs_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int16_tile = lhs_int16_tile_group.current()
    rhs_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int16_tile = rhs_int16_tile_group.current()
    lhs_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int32_tile = lhs_int32_tile_group.current()
    rhs_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int32_tile = rhs_int32_tile_group.current()
    lhs_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int64_tile = lhs_int64_tile_group.current()
    rhs_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int64_tile = rhs_int64_tile_group.current()
    out_int8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_int8_tile = out_int8_tile_group.current()
    out_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_int16_tile = out_int16_tile_group.current()
    out_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2800,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_tile = out_int32_tile_group.current()
    out_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x2C00,
        mutex_ids="auto",
        depth=1,
    )
    out_int64_tile = out_int64_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_int8_tile, lhs_int8, [0, 0])
        pl.load(rhs_int8_tile, rhs_int8, [0, 0])
        pl.load(lhs_int16_tile, lhs_int16, [0, 0])
        pl.load(rhs_int16_tile, rhs_int16, [0, 0])
        pl.load(lhs_int32_tile, lhs_int32, [0, 0])
        pl.load(rhs_int32_tile, rhs_int32, [0, 0])
        pl.load(lhs_int64_tile, lhs_int64, [0, 0])
        pl.load(rhs_int64_tile, rhs_int64, [0, 0])
        min_signed_dtypes[ELEMENTS](
            lhs_int8_tile,
            rhs_int8_tile,
            lhs_int16_tile,
            rhs_int16_tile,
            lhs_int32_tile,
            rhs_int32_tile,
            lhs_int64_tile,
            rhs_int64_tile,
            out_int8_tile,
            out_int16_tile,
            out_int32_tile,
            out_int64_tile,
        )
        pl.store(out_int8, out_int8_tile, [0, 0])
        pl.store(out_int16, out_int16_tile, [0, 0])
        pl.store(out_int32, out_int32_tile, [0, 0])
        pl.store(out_int64, out_int64_tile, [0, 0])

@pytest.mark.soc("950")
def test_min_all_supported_signed_dtypes():
    torch.npu.set_device(ST_DEVICE)
    lhs = (torch.arange(ELEMENTS, dtype=torch.int64) - 32).reshape(1, ELEMENTS)
    rhs = (12 - lhs).reshape(1, ELEMENTS)
    dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    lhs_sources = tuple(lhs.to(dtype) for dtype in dtypes)
    rhs_sources = tuple(rhs.to(dtype) for dtype in dtypes)
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in lhs_sources)
    arguments = [tensor.to(ST_DEVICE) for pair in zip(lhs_sources, rhs_sources) for tensor in pair]
    simt_min_signed_dtypes(*arguments, *outputs)
    torch.npu.synchronize()
    expected = torch.minimum(lhs, rhs)
    for dtype, output in zip(dtypes, outputs):
        torch.testing.assert_close(output.cpu(), expected.to(dtype), rtol=0, atol=0)

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def min_unsigned_dtypes(
    lhs_uint8,
    rhs_uint8,
    lhs_uint16,
    rhs_uint16,
    lhs_uint32,
    rhs_uint32,
    lhs_uint64,
    rhs_uint64,
    out_uint8,
    out_uint16,
    out_uint32,
    out_uint64,
):
    tid = pl.simt.linear_thread_idx()
    out_uint8[0, tid] = pl.simt.min(lhs_uint8[0, tid], rhs_uint8[0, tid])
    out_uint16[0, tid] = pl.simt.min(lhs_uint16[0, tid], rhs_uint16[0, tid])
    out_uint32[0, tid] = pl.simt.min(lhs_uint32[0, tid], rhs_uint32[0, tid])
    out_uint64[0, tid] = pl.simt.min(lhs_uint64[0, tid], rhs_uint64[0, tid])

@pl.jit(auto_mutex=True)
def simt_min_unsigned_dtypes(
    lhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    rhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    lhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    rhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    lhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    lhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    rhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    lhs_uint8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint8_tile = lhs_uint8_tile_group.current()
    rhs_uint8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint8_tile = rhs_uint8_tile_group.current()
    lhs_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint16_tile = lhs_uint16_tile_group.current()
    rhs_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint16_tile = rhs_uint16_tile_group.current()
    lhs_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint32_tile = lhs_uint32_tile_group.current()
    rhs_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint32_tile = rhs_uint32_tile_group.current()
    lhs_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint64_tile = lhs_uint64_tile_group.current()
    rhs_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint64_tile = rhs_uint64_tile_group.current()
    out_uint8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_uint8_tile = out_uint8_tile_group.current()
    out_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint16_tile = out_uint16_tile_group.current()
    out_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2800,
        mutex_ids="auto",
        depth=1,
    )
    out_uint32_tile = out_uint32_tile_group.current()
    out_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x2C00,
        mutex_ids="auto",
        depth=1,
    )
    out_uint64_tile = out_uint64_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_uint8_tile, lhs_uint8, [0, 0])
        pl.load(rhs_uint8_tile, rhs_uint8, [0, 0])
        pl.load(lhs_uint16_tile, lhs_uint16, [0, 0])
        pl.load(rhs_uint16_tile, rhs_uint16, [0, 0])
        pl.load(lhs_uint32_tile, lhs_uint32, [0, 0])
        pl.load(rhs_uint32_tile, rhs_uint32, [0, 0])
        pl.load(lhs_uint64_tile, lhs_uint64, [0, 0])
        pl.load(rhs_uint64_tile, rhs_uint64, [0, 0])
        min_unsigned_dtypes[ELEMENTS](
            lhs_uint8_tile,
            rhs_uint8_tile,
            lhs_uint16_tile,
            rhs_uint16_tile,
            lhs_uint32_tile,
            rhs_uint32_tile,
            lhs_uint64_tile,
            rhs_uint64_tile,
            out_uint8_tile,
            out_uint16_tile,
            out_uint32_tile,
            out_uint64_tile,
        )
        pl.store(out_uint8, out_uint8_tile, [0, 0])
        pl.store(out_uint16, out_uint16_tile, [0, 0])
        pl.store(out_uint32, out_uint32_tile, [0, 0])
        pl.store(out_uint64, out_uint64_tile, [0, 0])

@pytest.mark.soc("950")
def test_min_all_supported_unsigned_dtypes():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.arange(ELEMENTS, dtype=torch.int64).reshape(1, ELEMENTS)
    rhs = (ELEMENTS - 1 - lhs).reshape(1, ELEMENTS)
    dtypes = (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    lhs_sources = tuple(lhs.to(dtype) for dtype in dtypes)
    rhs_sources = tuple(rhs.to(dtype) for dtype in dtypes)
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in lhs_sources)
    arguments = [tensor.to(ST_DEVICE) for pair in zip(lhs_sources, rhs_sources) for tensor in pair]
    simt_min_unsigned_dtypes(*arguments, *outputs)
    torch.npu.synchronize()
    expected = torch.minimum(lhs, rhs)
    for dtype, output in zip(dtypes, outputs):
        torch.testing.assert_close(output.cpu(), expected.to(dtype), rtol=0, atol=0)

# ------------------------------------------------- min_nan -------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def min_nan_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16_tile, out_bf16_tile):
    tid = pl.simt.linear_thread_idx()
    out_fp16_tile[0, tid] = pl.simt.min_nan(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid])
    out_bf16_tile[0, tid] = pl.simt.min_nan(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_min_nan_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    lhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp16_tile = lhs_fp16_tile_group.current()
    rhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp16_tile = rhs_fp16_tile_group.current()
    lhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_bf16_tile = lhs_bf16_tile_group.current()
    rhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_bf16_tile = rhs_bf16_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        min_nan_tile[ELEMENTS](lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16_tile, out_bf16_tile)
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])

@pytest.mark.soc("950")
def test_min_nan_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * 8)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * 8)
    lhs[:2] = torch.tensor([-0.0, 0.0])
    rhs[:2] = torch.tensor([0.0, -0.0])
    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(torch.empty_like(lhs_source, device=ST_DEVICE) for lhs_source, _ in operand_pairs)
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_min_nan_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = torch.minimum(lhs_source.float(), rhs_source.float()).to(lhs_source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(output.cpu()[:, :2].view(torch.int16), torch.full((1, 2), -32768, dtype=torch.int16))

# --------------------------------------------------- max ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def max_float_dtypes(
    lhs_fp16,
    rhs_fp16,
    lhs_bf16,
    rhs_bf16,
    lhs_fp32,
    rhs_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.max(lhs_fp16[0, tid], rhs_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.max(lhs_bf16[0, tid], rhs_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.max(lhs_fp32[0, tid], rhs_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_max_float_dtypes(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    lhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    lhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp16_tile = lhs_fp16_tile_group.current()
    rhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp16_tile = rhs_fp16_tile_group.current()
    lhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_bf16_tile = lhs_bf16_tile_group.current()
    rhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_bf16_tile = rhs_bf16_tile_group.current()
    lhs_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp32_tile = lhs_fp32_tile_group.current()
    rhs_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp32_tile = rhs_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        pl.load(lhs_fp32_tile, lhs_fp32, [0, 0])
        pl.load(rhs_fp32_tile, rhs_fp32, [0, 0])
        max_float_dtypes[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            lhs_fp32_tile,
            rhs_fp32_tile,
            out_fp16_tile,
            out_bf16_tile,
            out_fp32_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])

@pytest.mark.soc("950")
def test_max_all_supported_float_dtypes():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.linspace(-4.0, 4.0, ELEMENTS)
    rhs = torch.linspace(2.0, -2.0, ELEMENTS)
    lhs[:7] = torch.tensor([float("nan"), 1.0, float("nan"), -0.0, 0.0, float("inf"), float("-inf")])
    rhs[:7] = torch.tensor([1.0, float("nan"), float("nan"), 0.0, -0.0, 2.0, 2.0])
    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple(
        (lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    outputs = tuple(torch.empty_like(lhs_source, device=ST_DEVICE) for lhs_source, _ in operand_pairs)
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_max_float_dtypes(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = torch.fmax(lhs_source.float(), rhs_source.float()).to(lhs_source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def max_signed_dtypes(
    lhs_int8,
    rhs_int8,
    lhs_int16,
    rhs_int16,
    lhs_int32,
    rhs_int32,
    lhs_int64,
    rhs_int64,
    out_int8,
    out_int16,
    out_int32,
    out_int64,
):
    tid = pl.simt.linear_thread_idx()
    out_int8[0, tid] = pl.simt.max(lhs_int8[0, tid], rhs_int8[0, tid])
    out_int16[0, tid] = pl.simt.max(lhs_int16[0, tid], rhs_int16[0, tid])
    out_int32[0, tid] = pl.simt.max(lhs_int32[0, tid], rhs_int32[0, tid])
    out_int64[0, tid] = pl.simt.max(lhs_int64[0, tid], rhs_int64[0, tid])

@pl.jit(auto_mutex=True)
def simt_max_signed_dtypes(
    lhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    rhs_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    lhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    rhs_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    lhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    rhs_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    lhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    rhs_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_int8: pl.Tensor[[1, ELEMENTS], pl.DT_INT8],
    out_int16: pl.Tensor[[1, ELEMENTS], pl.DT_INT16],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
):
    lhs_int8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int8_tile = lhs_int8_tile_group.current()
    rhs_int8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int8_tile = rhs_int8_tile_group.current()
    lhs_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int16_tile = lhs_int16_tile_group.current()
    rhs_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int16_tile = rhs_int16_tile_group.current()
    lhs_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int32_tile = lhs_int32_tile_group.current()
    rhs_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int32_tile = rhs_int32_tile_group.current()
    lhs_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_int64_tile = lhs_int64_tile_group.current()
    rhs_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_int64_tile = rhs_int64_tile_group.current()
    out_int8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_int8_tile = out_int8_tile_group.current()
    out_int16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_int16_tile = out_int16_tile_group.current()
    out_int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2800,
        mutex_ids="auto",
        depth=1,
    )
    out_int32_tile = out_int32_tile_group.current()
    out_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x2C00,
        mutex_ids="auto",
        depth=1,
    )
    out_int64_tile = out_int64_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_int8_tile, lhs_int8, [0, 0])
        pl.load(rhs_int8_tile, rhs_int8, [0, 0])
        pl.load(lhs_int16_tile, lhs_int16, [0, 0])
        pl.load(rhs_int16_tile, rhs_int16, [0, 0])
        pl.load(lhs_int32_tile, lhs_int32, [0, 0])
        pl.load(rhs_int32_tile, rhs_int32, [0, 0])
        pl.load(lhs_int64_tile, lhs_int64, [0, 0])
        pl.load(rhs_int64_tile, rhs_int64, [0, 0])
        max_signed_dtypes[ELEMENTS](
            lhs_int8_tile,
            rhs_int8_tile,
            lhs_int16_tile,
            rhs_int16_tile,
            lhs_int32_tile,
            rhs_int32_tile,
            lhs_int64_tile,
            rhs_int64_tile,
            out_int8_tile,
            out_int16_tile,
            out_int32_tile,
            out_int64_tile,
        )
        pl.store(out_int8, out_int8_tile, [0, 0])
        pl.store(out_int16, out_int16_tile, [0, 0])
        pl.store(out_int32, out_int32_tile, [0, 0])
        pl.store(out_int64, out_int64_tile, [0, 0])

@pytest.mark.soc("950")
def test_max_all_supported_signed_dtypes():
    torch.npu.set_device(ST_DEVICE)
    lhs = (torch.arange(ELEMENTS, dtype=torch.int64) - 32).reshape(1, ELEMENTS)
    rhs = (12 - lhs).reshape(1, ELEMENTS)
    dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    lhs_sources = tuple(lhs.to(dtype) for dtype in dtypes)
    rhs_sources = tuple(rhs.to(dtype) for dtype in dtypes)
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in lhs_sources)
    arguments = [tensor.to(ST_DEVICE) for pair in zip(lhs_sources, rhs_sources) for tensor in pair]
    simt_max_signed_dtypes(*arguments, *outputs)
    torch.npu.synchronize()
    expected = torch.maximum(lhs, rhs)
    for dtype, output in zip(dtypes, outputs):
        torch.testing.assert_close(output.cpu(), expected.to(dtype), rtol=0, atol=0)

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def max_unsigned_dtypes(
    lhs_uint8,
    rhs_uint8,
    lhs_uint16,
    rhs_uint16,
    lhs_uint32,
    rhs_uint32,
    lhs_uint64,
    rhs_uint64,
    out_uint8,
    out_uint16,
    out_uint32,
    out_uint64,
):
    tid = pl.simt.linear_thread_idx()
    out_uint8[0, tid] = pl.simt.max(lhs_uint8[0, tid], rhs_uint8[0, tid])
    out_uint16[0, tid] = pl.simt.max(lhs_uint16[0, tid], rhs_uint16[0, tid])
    out_uint32[0, tid] = pl.simt.max(lhs_uint32[0, tid], rhs_uint32[0, tid])
    out_uint64[0, tid] = pl.simt.max(lhs_uint64[0, tid], rhs_uint64[0, tid])

@pl.jit(auto_mutex=True)
def simt_max_unsigned_dtypes(
    lhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    rhs_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    lhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    rhs_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    lhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    rhs_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    lhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    rhs_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
    out_uint8: pl.Tensor[[1, ELEMENTS], pl.DT_UINT8],
    out_uint16: pl.Tensor[[1, ELEMENTS], pl.DT_UINT16],
    out_uint32: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    out_uint64: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    lhs_uint8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint8_tile = lhs_uint8_tile_group.current()
    rhs_uint8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint8_tile = rhs_uint8_tile_group.current()
    lhs_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint16_tile = lhs_uint16_tile_group.current()
    rhs_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint16_tile = rhs_uint16_tile_group.current()
    lhs_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint32_tile = lhs_uint32_tile_group.current()
    rhs_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint32_tile = rhs_uint32_tile_group.current()
    lhs_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_uint64_tile = lhs_uint64_tile_group.current()
    rhs_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_uint64_tile = rhs_uint64_tile_group.current()
    out_uint8_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    out_uint8_tile = out_uint8_tile_group.current()
    out_uint16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec),
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_uint16_tile = out_uint16_tile_group.current()
    out_uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2800,
        mutex_ids="auto",
        depth=1,
    )
    out_uint32_tile = out_uint32_tile_group.current()
    out_uint64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x2C00,
        mutex_ids="auto",
        depth=1,
    )
    out_uint64_tile = out_uint64_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_uint8_tile, lhs_uint8, [0, 0])
        pl.load(rhs_uint8_tile, rhs_uint8, [0, 0])
        pl.load(lhs_uint16_tile, lhs_uint16, [0, 0])
        pl.load(rhs_uint16_tile, rhs_uint16, [0, 0])
        pl.load(lhs_uint32_tile, lhs_uint32, [0, 0])
        pl.load(rhs_uint32_tile, rhs_uint32, [0, 0])
        pl.load(lhs_uint64_tile, lhs_uint64, [0, 0])
        pl.load(rhs_uint64_tile, rhs_uint64, [0, 0])
        max_unsigned_dtypes[ELEMENTS](
            lhs_uint8_tile,
            rhs_uint8_tile,
            lhs_uint16_tile,
            rhs_uint16_tile,
            lhs_uint32_tile,
            rhs_uint32_tile,
            lhs_uint64_tile,
            rhs_uint64_tile,
            out_uint8_tile,
            out_uint16_tile,
            out_uint32_tile,
            out_uint64_tile,
        )
        pl.store(out_uint8, out_uint8_tile, [0, 0])
        pl.store(out_uint16, out_uint16_tile, [0, 0])
        pl.store(out_uint32, out_uint32_tile, [0, 0])
        pl.store(out_uint64, out_uint64_tile, [0, 0])

@pytest.mark.soc("950")
def test_max_all_supported_unsigned_dtypes():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.arange(ELEMENTS, dtype=torch.int64).reshape(1, ELEMENTS)
    rhs = (ELEMENTS - 1 - lhs).reshape(1, ELEMENTS)
    dtypes = (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    lhs_sources = tuple(lhs.to(dtype) for dtype in dtypes)
    rhs_sources = tuple(rhs.to(dtype) for dtype in dtypes)
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in lhs_sources)
    arguments = [tensor.to(ST_DEVICE) for pair in zip(lhs_sources, rhs_sources) for tensor in pair]
    simt_max_unsigned_dtypes(*arguments, *outputs)
    torch.npu.synchronize()
    expected = torch.maximum(lhs, rhs)
    for dtype, output in zip(dtypes, outputs):
        torch.testing.assert_close(output.cpu(), expected.to(dtype), rtol=0, atol=0)

# ------------------------------------------------- max_nan -------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def max_nan_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16_tile, out_bf16_tile):
    tid = pl.simt.linear_thread_idx()
    out_fp16_tile[0, tid] = pl.simt.max_nan(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid])
    out_bf16_tile[0, tid] = pl.simt.max_nan(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_max_nan_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    lhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp16_tile = lhs_fp16_tile_group.current()
    rhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp16_tile = rhs_fp16_tile_group.current()
    lhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_bf16_tile = lhs_bf16_tile_group.current()
    rhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_bf16_tile = rhs_bf16_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        max_nan_tile[ELEMENTS](lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16_tile, out_bf16_tile)
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])

@pytest.mark.soc("950")
def test_max_nan_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * 8)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * 8)
    lhs[:2] = torch.tensor([-0.0, 0.0])
    rhs[:2] = torch.tensor([0.0, -0.0])
    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(torch.empty_like(lhs_source, device=ST_DEVICE) for lhs_source, _ in operand_pairs)
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_max_nan_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = torch.maximum(lhs_source.float(), rhs_source.float()).to(lhs_source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    for output in outputs:
        assert torch.equal(output.cpu()[:, :2].view(torch.int16), torch.zeros((1, 2), dtype=torch.int16))
