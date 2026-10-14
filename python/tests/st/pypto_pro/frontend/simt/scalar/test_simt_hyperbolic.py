# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar hyperbolic interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# -------------------------------------------------- tanh ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def tanh_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.tanh(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.tanh(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.tanh(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_tanh_all_dtypes(
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
        tanh_all_dtypes[ELEMENTS](
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
def test_tanh_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-4.0, 4.0, ELEMENTS)
    values[:10] = torch.tensor(
        [float("-inf"), -10.0, -(2.0**-24), -1.0e-7, -0.0, 0.0, 1.0e-7, 2.0**-24, 10.0, float("inf")]
    )
    values = values.reshape(1, ELEMENTS).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_tanh_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.tanh(source.float()).to(source.dtype)
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if source.dtype == torch.float32:
            torch.testing.assert_close(actual[0, 2:8], expected[0, 2:8], rtol=1e-5, atol=0)

# -------------------------------------------------- sinh ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def sinh_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.sinh(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_sinh_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        sinh_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_sinh_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-8.0, 8.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :6] = torch.tensor([-89.0, -88.8, -88.0, 88.0, 88.8, 89.0])
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_sinh_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:, :6], torch.sinh(source[:, :6].double()).float(), rtol=2e-5, atol=0.0)
    torch.testing.assert_close(actual[:, 6:], torch.sinh(source[:, 6:]), rtol=2e-5, atol=2e-6)

# -------------------------------------------------- cosh ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cosh_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.cosh(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_cosh_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        cosh_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_cosh_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-8.0, 8.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :6] = torch.tensor([-89.0, -88.8, -88.0, 88.0, 88.8, 89.0])
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_cosh_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual[:, :6], torch.cosh(source[:, :6].double()).float(), rtol=2e-5, atol=0.0)
    torch.testing.assert_close(actual[:, 6:], torch.cosh(source[:, 6:]), rtol=2e-5, atol=2e-6)

# -------------------------------------------------- asinh --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def asinh_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.asinh(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_asinh_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        asinh_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_asinh_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-32.0, 32.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-float("inf"), -1e20, -1e19, -1.0, -0.5, -1e-8, -0.0, 0.0,
         1e-8, 0.5, 1.0, 1e19, 1e20, float("inf"), float("nan"), 1e-4]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_asinh_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.asinh(source)
    torch.testing.assert_close(actual[:, :16], expected[:, :16], rtol=5e-5, atol=1e-12, equal_nan=True)
    torch.testing.assert_close(actual[:, 16:], expected[:, 16:], rtol=5e-4, atol=5e-6)

# -------------------------------------------------- acosh --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def acosh_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.acosh(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_acosh_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        acosh_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_acosh_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(1.0, 32.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [0.5, 1.0, 1.0000001192, 1.0001, 1.25, 1.5, 1.5000001, 2.0,
         10.0, 8388608.0, 8388609.0, 8388610.0, 1e19, 1e20, float("inf"), float("nan")]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_acosh_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.acosh(source)
    torch.testing.assert_close(actual[:, :16], expected[:, :16], rtol=5e-5, atol=1e-7, equal_nan=True)
    torch.testing.assert_close(actual[:, 16:], expected[:, 16:], rtol=5e-4, atol=5e-6)

# -------------------------------------------------- atanh --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def atanh_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.atanh(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_atanh_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        atanh_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_atanh_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-0.95, 0.95, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :6] = torch.tensor([-1e-8, -1e-6, -1e-4, 1e-8, 1e-6, 1e-4])
    source[0, 6:12] = torch.tensor([-1.5, -1.0, -0.99999994, 0.99999994, 1.0, 1.5])
    source[0, 12:14] = torch.tensor([-0.0, 0.0])
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_atanh_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.atanh(source)
    torch.testing.assert_close(actual[:, :6], expected[:, :6], rtol=2e-5, atol=1e-12)
    torch.testing.assert_close(actual[:, 6:12], expected[:, 6:12], rtol=5e-4, atol=1e-6, equal_nan=True)
    torch.testing.assert_close(actual[:, 14:], expected[:, 14:], rtol=5e-4, atol=5e-6)
    assert torch.equal(actual[:, 12:14].view(torch.int32), source[:, 12:14].view(torch.int32))
