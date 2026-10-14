# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar trigonometric interfaces."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# --------------------------------------------------- sin ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def sin_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.sin(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.sin(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.sin(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_sin_all_dtypes(
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
        sin_all_dtypes[ELEMENTS](
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
def test_sin_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    angles = torch.linspace(-3.14159265, 3.14159265, ELEMENTS)
    angles[:10] = torch.tensor(
        [-0.0, 0.0, -3.14159265, 3.14159265, -314.159265, 314.159265, -10000.0, 10000.0, float("-inf"), float("inf")]
    )
    angles[10:14] = torch.tensor([0.2093505859375, -0.2093505859375, 300.0, -300.0])
    values = angles.reshape(1, ELEMENTS).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_sin_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.sin(source.float()).to(source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True)
    # Check the corrected half results exactly at rounding-sensitive inputs.
    expected_bits = torch.tensor([0x32A7, 0xB2A7, 0xBBFF, 0x3BFF], dtype=torch.int32).to(torch.int16)
    torch.testing.assert_close(outputs[0].cpu().view(torch.int16)[0, 10:14], expected_bits, rtol=0, atol=0)

# --------------------------------------------------- cos ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cos_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.cos(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.cos(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.cos(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_cos_all_dtypes(
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
        cos_all_dtypes[ELEMENTS](
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
def test_cos_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    angles = torch.linspace(-3.14159265, 3.14159265, ELEMENTS)
    angles[:10] = torch.tensor(
        [-0.0, 0.0, -3.14159265, 3.14159265, -314.159265, 314.159265, -10000.0, 10000.0, float("-inf"), float("inf")]
    )
    angles[10:12] = torch.tensor([0.0584716796875, -0.0584716796875])
    values = angles.reshape(1, ELEMENTS).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_cos_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.cos(source.float()).to(source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True)
    # Both input signs must produce the same corrected half result.
    expected_bits = torch.tensor([0x3BFD, 0x3BFD], dtype=torch.int16)
    torch.testing.assert_close(outputs[0].cpu().view(torch.int16)[0, 10:12], expected_bits, rtol=0, atol=0)

# --------------------------------------------------- tan ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def tan_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.tan(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_tan_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        tan_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_tan_fp32_with_tile_operands():
    values = torch.linspace(-1.25, 1.25, ELEMENTS)
    torch.npu.set_device(ST_DEVICE)
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_tan_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.tan(source), rtol=2e-5, atol=2e-6)

# -------------------------------------------------- atan ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def atan_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.atan(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_atan_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        atan_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_atan_fp32_with_tile_operands():
    values = torch.linspace(-32.0, 32.0, ELEMENTS)
    torch.npu.set_device(ST_DEVICE)
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_atan_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.atan(source), rtol=2e-5, atol=2e-6)

# -------------------------------------------------- asin ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def asin_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.asin(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_asin_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        asin_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_asin_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-1.0, 1.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-float("inf"), -1.5, -1.0, -0.0, 0.0, 1.0, 1.5, float("inf"),
         float("nan"), -0.5, 0.5, -1.0001, 1.0001, -0.25, 0.25, 0.75]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_asin_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.asin(source)
    torch.testing.assert_close(actual[:, :16], expected[:, :16], rtol=2e-5, atol=2e-6, equal_nan=True)
    torch.testing.assert_close(actual[:, 16:], expected[:, 16:], rtol=2e-5, atol=2e-6)

# -------------------------------------------------- acos ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def acos_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.acos(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_acos_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        acos_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_acos_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-1.0, 1.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-float("inf"), -1.5, -1.0, -0.0, 0.0, 1.0, 1.5, float("inf"),
         float("nan"), -0.5, 0.5, -1.0001, 1.0001, -0.25, 0.25, 0.75]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_acos_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.acos(source)
    torch.testing.assert_close(actual[:, :16], expected[:, :16], rtol=2e-5, atol=2e-6, equal_nan=True)
    torch.testing.assert_close(actual[:, 16:], expected[:, 16:], rtol=2e-5, atol=2e-6)

# -------------------------------------------------- atan2 --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def atan2_tile(lhs_tile, rhs_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.atan2(lhs_tile[0, tid], rhs_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_atan2_tile(
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
        atan2_tile[ELEMENTS](lhs_tile, rhs_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_atan2_fp32_with_tile_operands():
    lhs = torch.linspace(-8.0, 8.0, ELEMENTS)
    rhs = torch.tensor([-4.0, 4.0, 4.0, -4.0] * (ELEMENTS // 4), dtype=torch.float32)
    torch.npu.set_device(ST_DEVICE)
    lhs = lhs.reshape(1, ELEMENTS)
    rhs = rhs.reshape(1, ELEMENTS)
    lhs[0, :16] = torch.tensor(
        [-0.0, 0.0, -0.0, 0.0, 1.0, -1.0, 1.0, -1.0,
         float("inf"), float("-inf"), float("inf"), float("-inf"),
         float("nan"), 1.0, 1.0, -1.0]
    )
    rhs[0, :16] = torch.tensor(
        [0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0,
         float("inf"), float("inf"), float("-inf"), float("-inf"),
         1.0, float("nan"), float("inf"), float("-inf")]
    )
    output = torch.empty_like(lhs, device=ST_DEVICE)
    simt_atan2_tile(lhs.to(ST_DEVICE), rhs.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.atan2(lhs, rhs)
    torch.testing.assert_close(actual[:, :16], expected[:, :16], rtol=5e-4, atol=5e-6, equal_nan=True)
    torch.testing.assert_close(actual[:, 16:], expected[:, 16:], rtol=5e-4, atol=5e-6)
    non_nan = ~torch.isnan(expected[:, :16])
    assert torch.equal(torch.signbit(actual[:, :16][non_nan]), torch.signbit(expected[:, :16][non_nan]))

# -------------------------------------------------- sinpi --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def sinpi_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.sinpi(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_sinpi_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        sinpi_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_sinpi_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-3.0, 3.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5,
         1.0, 1.5, 2.0, 2.5, 8388608.0, 8388609.0, 16777216.0, -3.4e38]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_sinpi_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.tensor(
        [-1.0, -0.0, 1.0, -0.0, -1.0, -0.0, 0.0, 1.0,
         0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.0]
    )
    actual = output.cpu()
    assert torch.equal(actual[0, :16].view(torch.int32), expected.view(torch.int32))
    torch.testing.assert_close(actual[:, 16:], torch.sin(math.pi * source[:, 16:]), rtol=5e-4, atol=5e-6)

# -------------------------------------------------- cospi --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cospi_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.cospi(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_cospi_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        cospi_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_cospi_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-3.0, 3.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5,
         1.0, 1.5, 2.0, 2.5, 8388608.0, 8388609.0, 16777216.0, -3.4e38]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_cospi_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.tensor(
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0,
         -1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0]
    )
    actual = output.cpu()
    torch.testing.assert_close(actual[0, :16], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[:, 16:], torch.cos(math.pi * source[:, 16:]), rtol=5e-4, atol=5e-6)

# -------------------------------------------------- tanpi --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def tanpi_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.tanpi(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_tanpi_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        tanpi_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_tanpi_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-0.45, 0.45, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5,
         1.0, 1.5, 2.0, 2.5, 8388608.0, 8388609.0, 16777216.0, -3.4e38]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_tanpi_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.tensor(
        [-float("inf"), -0.0, float("inf"), -0.0, -float("inf"), -0.0, 0.0, float("inf"),
         0.0, -float("inf"), 0.0, float("inf"), 0.0, 0.0, 0.0, -0.0]
    )
    actual = output.cpu()
    assert torch.equal(actual[0, :16].view(torch.int32), expected.view(torch.int32))
    torch.testing.assert_close(actual[:, 16:], torch.tan(math.pi * source[:, 16:]), rtol=5e-4, atol=5e-6)
