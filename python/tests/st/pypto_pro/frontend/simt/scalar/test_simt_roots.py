# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar root interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# -------------------------------------------------- sqrt ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def sqrt_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.sqrt(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.sqrt(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.sqrt(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_sqrt_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_fp16_tile = src_fp16_tile_group.current()
    src_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_bf16_tile = src_bf16_tile_group.current()
    src_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    src_fp32_tile = src_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    with pl.section_vector():
        pl.load(src_fp16_tile, src_fp16, [0, 0])
        pl.load(src_bf16_tile, src_bf16, [0, 0])
        pl.load(src_fp32_tile, src_fp32, [0, 0])
        sqrt_all_dtypes[ELEMENTS](
            src_fp16_tile,
            src_bf16_tile,
            src_fp32_tile,
            out_fp16_tile,
            out_bf16_tile,
            out_fp32_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])

@pytest.mark.soc("950")
def test_sqrt_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.25, 16.0, ELEMENTS)
    values[:6] = torch.tensor(
        [0.0, -0.0, torch.finfo(torch.float32).tiny / 2, torch.finfo(torch.float32).tiny, 1.0, float("inf")]
    )
    values = values.reshape(1, ELEMENTS).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_sqrt_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.sqrt(source.float()).to(source.dtype)
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if source.dtype == torch.float32:
            torch.testing.assert_close(actual[0, 2], expected[0, 2], rtol=1e-5, atol=0)

# -------------------------------------------------- rsqrt --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def rsqrt_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.rsqrt(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.rsqrt(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.rsqrt(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_rsqrt_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_fp16_tile = src_fp16_tile_group.current()
    src_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_bf16_tile = src_bf16_tile_group.current()
    src_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    src_fp32_tile = src_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    with pl.section_vector():
        pl.load(src_fp16_tile, src_fp16, [0, 0])
        pl.load(src_bf16_tile, src_bf16, [0, 0])
        pl.load(src_fp32_tile, src_fp32, [0, 0])
        rsqrt_all_dtypes[ELEMENTS](
            src_fp16_tile,
            src_bf16_tile,
            src_fp32_tile,
            out_fp16_tile,
            out_bf16_tile,
            out_fp32_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])

@pytest.mark.soc("950")
def test_rsqrt_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.25, 16.0, ELEMENTS)
    values[:6] = torch.tensor(
        [0.0, -0.0, torch.finfo(torch.float32).tiny / 2, torch.finfo(torch.float32).tiny, 1.0, float("inf")]
    )
    values = values.reshape(1, ELEMENTS).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_rsqrt_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.rsqrt(source.float()).to(source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True)

# -------------------------------------------------- cbrt ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cbrt_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.cbrt(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_cbrt_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        cbrt_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_cbrt_fp32_with_tile_operands():
    values = torch.cat((torch.linspace(-64.0, -0.125, ELEMENTS // 2), torch.linspace(0.125, 64.0, ELEMENTS // 2)))
    torch.npu.set_device(ST_DEVICE)
    source = values.reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-float("inf"), -3.4e38, -8.0, -1e-20, -1.17549435e-38, -1.401298464324817e-45, -0.0, 0.0,
         1.401298464324817e-45, 1.17549435e-38, 1e-20, 8.0, 3.4e38, float("inf"), float("nan"), 1.0]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_cbrt_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.sign(source) * torch.pow(torch.abs(source), 1.0 / 3.0)
    actual = output.cpu()
    torch.testing.assert_close(actual[:, :16], expected[:, :16], rtol=2e-5, atol=0.0, equal_nan=True)
    torch.testing.assert_close(actual[:, 16:], expected[:, 16:], rtol=2e-5, atol=2e-6)

# -------------------------------------------------- rcbrt --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def rcbrt_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.rcbrt(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_rcbrt_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        rcbrt_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_rcbrt_fp32_with_tile_operands():
    values = torch.cat((torch.linspace(-64.0, -0.125, ELEMENTS // 2), torch.linspace(0.125, 64.0, ELEMENTS // 2)))
    torch.npu.set_device(ST_DEVICE)
    source = values.reshape(1, ELEMENTS)
    source[0, :8] = torch.tensor([-0.0, 0.0, float("-inf"), float("inf"), float("nan"), 1.0, -1.0, 8.0])
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_rcbrt_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected_bits = torch.tensor([-0x00800000, 0x7F800000, -0x80000000, 0], dtype=torch.int32)
    assert torch.equal(actual[0, :4].view(torch.int32), expected_bits)
    assert torch.isnan(actual[0, 4])
    torch.testing.assert_close(actual[0, 5:8], torch.tensor([1.0, -1.0, 0.5]), rtol=5e-4, atol=5e-6)
    expected = torch.sign(source[:, 8:]) / torch.pow(torch.abs(source[:, 8:]), 1.0 / 3.0)
    torch.testing.assert_close(actual[:, 8:], expected, rtol=5e-4, atol=5e-6)
