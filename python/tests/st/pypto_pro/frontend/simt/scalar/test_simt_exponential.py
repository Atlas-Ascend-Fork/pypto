# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar exponential interfaces."""

from decimal import Decimal, localcontext
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# --------------------------------------------------- exp ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def exp_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.exp(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.exp(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.exp(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_exp_all_dtypes(
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
        exp_all_dtypes[ELEMENTS](
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
def test_exp_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-3.0, 3.0, ELEMENTS).reshape(1, ELEMENTS)
    values[0, :10] = torch.tensor(
        [
            float("-inf"),
            -104.0,
            -88.0,
            0.007297515869140625,
            0.0226898193359375,
            -1.0e-7,
            0.0,
            1.0e-7,
            88.0,
            float("inf"),
        ]
    )
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_exp_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.exp(source.float()).to(source.dtype)
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        if source.dtype == torch.float16:
            # Round the high-precision reference directly to IEEE FP16, without an FP32 intermediate.
            with localcontext() as context:
                context.prec = 80
                expected_bits = [
                    struct.unpack("<h", struct.pack("<e", float(Decimal.from_float(value).exp())))[0]
                    for value in source[0, 3:5].tolist()
                ]
            expected_exact = torch.tensor(expected_bits, dtype=torch.int16)
            assert torch.equal(actual[0, 3:5].view(torch.int16), expected_exact)
        elif source.dtype == torch.float32:
            torch.testing.assert_close(actual[0, 2], expected[0, 2], rtol=1e-5, atol=0)

# -------------------------------------------------- exp2 ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def exp2_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.exp2(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.exp2(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.exp2(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_exp2_all_dtypes(
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
        exp2_all_dtypes[ELEMENTS](
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
def test_exp2_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-3.0, 3.0, ELEMENTS).reshape(1, ELEMENTS)
    values[0, :9] = torch.tensor(
        [
            float("-inf"),
            -150.0,
            -126.0,
            0.000704288482666015625,
            -1.0e-7,
            0.0,
            1.0e-7,
            127.0,
            float("inf"),
        ]
    )
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_exp2_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.exp2(source.float()).to(source.dtype)
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        if source.dtype == torch.float16:
            # Round the high-precision reference directly to IEEE FP16, without an FP32 intermediate.
            with localcontext() as context:
                context.prec = 80
                log2 = Decimal(2).ln()
                expected_bits = [
                    struct.unpack(
                        "<h", struct.pack("<e", float((Decimal.from_float(value) * log2).exp()))
                    )[0]
                    for value in source[0, 3:4].tolist()
                ]
            expected_exact = torch.tensor(expected_bits, dtype=torch.int16)
            assert torch.equal(actual[0, 3:4].view(torch.int16), expected_exact)

# -------------------------------------------------- exp10 --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def exp10_tile(
    src_fp16,
    src_bf16_tile,
    src_fp32,
    out_fp16,
    out_bf16_tile,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.exp10(src_fp16[0, tid])
    out_bf16_tile[0, tid] = pl.simt.exp10(src_bf16_tile[0, tid])
    out_fp32[0, tid] = pl.simt.exp10(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_exp10_tile(
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
        exp10_tile[ELEMENTS](
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
def test_exp10_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-2.0, 2.0, ELEMENTS).reshape(1, ELEMENTS)
    fp16_source = values.half()
    bf16_source = values.bfloat16()
    fp32_source = values.clone()
    fp16_source[0, :2] = torch.tensor([0.30419921875, -1.759765625], dtype=torch.float16)
    fp32_source[0, :8] = torch.tensor([-44.0, -40.0, -20.0, -1.0, 0.0, 1.0, 38.0, 38.5])
    sources = (fp16_source, bf16_source, fp32_source)
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_exp10_tile(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        expected = torch.pow(10.0, source.float()).to(source.dtype)
        actual = output.cpu()
        if source.dtype == torch.float32:
            extreme_expected = torch.pow(10.0, source[:, :8].to(torch.float64)).to(torch.float32)
            torch.testing.assert_close(actual[:, :8], extreme_expected, rtol=2e-4, atol=2e-45)
            torch.testing.assert_close(actual[:, 8:], expected[:, 8:], rtol=2e-4, atol=1e-6)
        else:
            atol = 5e-3 if source.dtype == torch.float16 else 2e-2
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=atol)
    rounding_expected = torch.pow(10.0, fp16_source[:, :2].to(torch.float64)).to(torch.float16)
    assert torch.equal(outputs[0].cpu()[:, :2].view(torch.int16), rounding_expected.view(torch.int16))

# -------------------------------------------------- expm1 --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def expm1_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.expm1(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_expm1_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        expm1_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_expm1_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-8.0, 8.0, ELEMENTS).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [-1e-8, -1e-6, -1e-4, -0.000244140625, -0.0003, -0.001, -0.1, -0.0,
         0.0, 0.1, 0.001, 0.0003, 0.000244140625, 1e-4, 1e-6, 1e-8]
    )
    source[0, 16:32] = torch.tensor(
        [-float("inf"), -100.0, -90.0, -88.7, -1.0, -0.0, 0.0, 1.0,
         80.0, 88.0, 88.7, 89.0, 100.0, float("inf"), float("nan"), 0.0]
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_expm1_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.expm1(source)
    torch.testing.assert_close(actual[:, :16], expected[:, :16], rtol=2e-5, atol=1e-12)
    torch.testing.assert_close(actual[:, 16:32], expected[:, 16:32], rtol=2e-5, atol=0.0, equal_nan=True)
    torch.testing.assert_close(actual[:, 32:], expected[:, 32:], rtol=2e-5, atol=2e-6)
    assert torch.equal(actual[:, 7:9].view(torch.int32), source[:, 7:9].view(torch.int32))
