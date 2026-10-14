# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar logarithmic interfaces."""

from decimal import Decimal, localcontext
import os
import struct

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# --------------------------------------------------- log ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def log_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.log(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.log(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.log(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_log_all_dtypes(
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
        log_all_dtypes[ELEMENTS](
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
def test_log_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.125, 8.0, ELEMENTS)
    values[:10] = torch.tensor(
        [
            -1.0,
            -0.0,
            0.0,
            torch.finfo(torch.float32).tiny / 2,
            torch.finfo(torch.float32).tiny,
            0.005340576171875,
            1.0e-7,
            1.0,
            2.0,
            float("inf"),
        ]
    )
    values = values.reshape(1, ELEMENTS).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_log_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.log(source.float()).to(source.dtype)
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if source.dtype == torch.float16:
            # Round the high-precision reference directly to IEEE FP16, without an FP32 intermediate.
            with localcontext() as context:
                context.prec = 80
                expected_bits = [
                    struct.unpack("<h", struct.pack("<e", float(Decimal.from_float(value).ln())))[0]
                    for value in source[0, 5:6].tolist()
                ]
            expected_exact = torch.tensor(expected_bits, dtype=torch.int16)
            assert torch.equal(actual[0, 5:6].view(torch.int16), expected_exact)

# -------------------------------------------------- log2 ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def log2_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.log2(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.log2(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.log2(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_log2_all_dtypes(
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
        log2_all_dtypes[ELEMENTS](
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
def test_log2_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.125, 8.0, ELEMENTS)
    values[:8] = torch.tensor([-1.0, -0.0, 0.0, torch.finfo(torch.float32).tiny, 1.0e-7, 1.0, 2.0, float("inf")])
    values = values.reshape(1, ELEMENTS).to(torch.float32)
    sources = tuple(values.to(dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_log2_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        expected = torch.log2(source.float()).to(source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True)

# -------------------------------------------------- log1p --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def log1p_fp32(
    source,
    output,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.log1p(source[0, tid])

@pl.jit(auto_mutex=True)
def simt_log1p_fp32(
    source: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    output: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    source_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    source_tile = source_tile_group.current()
    output_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    output_tile = output_tile_group.current()
    with pl.section_vector():
        pl.load(source_tile, source, [0, 0])
        log1p_fp32[ELEMENTS](source_tile, output_tile)
        pl.store(output, output_tile, [0, 0])

@pytest.mark.soc("950")
def test_log1p_supported_dtype():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-0.75, 8.0, ELEMENTS)
    values[:10] = torch.tensor(
        [-1.0, -0.0, 0.0, -(2.0**-24), 2.0**-24, -1.0e-7, 1.0e-7, -(2.0**-20), 2.0**-20, 1.0]
    )
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_log1p_fp32(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = torch.log1p(source)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(actual[0, 3:9], expected[0, 3:9], rtol=2e-5, atol=0)

# -------------------------------------------------- log10 --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def log10_tile(
    src_fp16,
    src_bf16_tile,
    src_fp32,
    out_fp16,
    out_bf16_tile,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.log10(src_fp16[0, tid])
    out_bf16_tile[0, tid] = pl.simt.log10(src_bf16_tile[0, tid])
    out_fp32[0, tid] = pl.simt.log10(src_fp32[0, tid])

@pl.jit(auto_mutex=True)
def simt_log10_tile(
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
        log10_tile[ELEMENTS](
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
def test_log10_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.125, 32.0, ELEMENTS).reshape(1, ELEMENTS)
    rounding_inputs = [0.2362060546875, 0.2490234375, 3.0703125, 126.0625, 11496.0, 2976.0]
    fp16_source = values.half()
    bf16_source = values.bfloat16()
    fp32_source = values.clone()
    fp16_source[0, :6] = torch.tensor(rounding_inputs, dtype=torch.float16)
    fp32_source[0, :8] = torch.tensor(
        [1.401298464324817e-45, 1e-40, 1.17549435e-38, 1e-10, 1.0, 10.0, 0.0, -1.0]
    )
    sources = (fp16_source, bf16_source, fp32_source)
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_log10_tile(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        expected = torch.log10(source.float()).to(source.dtype)
        actual = output.cpu()
        if source.dtype == torch.float32:
            torch.testing.assert_close(
                actual[:, :8], torch.log10(source[:, :8]), rtol=2e-5, atol=2e-5, equal_nan=True
            )
            torch.testing.assert_close(actual[:, 8:], expected[:, 8:], rtol=2e-4, atol=1e-6)
        else:
            atol = 5e-3 if source.dtype == torch.float16 else 2e-2
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=atol)
    # Round the high-precision reference directly to IEEE FP16, without an FP32 intermediate.
    with localcontext() as context:
        context.prec = 80
        expected_bits = [
            struct.unpack("<h", struct.pack("<e", float(Decimal.from_float(value).log10())))[0]
            for value in rounding_inputs
        ]
    expected = torch.tensor(expected_bits, dtype=torch.int16).reshape(1, 6)
    assert torch.equal(outputs[0].cpu()[:, :6].view(torch.int16), expected)

# -------------------------------------------------- logb ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def logb_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.logb(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_logb_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32]):
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
        logb_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_logb_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(0.125, 32.0, ELEMENTS).reshape(1, ELEMENTS)
    nan_bits = torch.tensor([0x7FC12345, -0x003EDCBB], dtype=torch.int32)
    source[0, :2] = nan_bits.view(torch.float32)
    special_bits = torch.tensor([0, -2147483648, 1, 0x00400000, 0x007FFFFF, 0x00800000,
                                 0x7F800000, -0x00800000], dtype=torch.int32)
    source[0, 2:10] = special_bits.view(torch.float32)
    special_expected = torch.tensor([float("-inf"), float("-inf"), -149, -127, -127, -126,
                                     float("inf"), float("inf")])
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_logb_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    assert torch.equal(actual[0, :2].view(torch.int32), nan_bits)
    torch.testing.assert_close(actual[0, 2:10], special_expected, rtol=0, atol=0)
    expected = torch.floor(torch.log2(torch.abs(source[:, 10:])))
    torch.testing.assert_close(actual[:, 10:], expected, rtol=1e-5, atol=1e-6)

# -------------------------------------------------- ilogb --------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def ilogb_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.ilogb(src_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_ilogb_tile(src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32], out: pl.Tensor[[1, ELEMENTS], pl.DT_INT32]):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        ilogb_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])

@pytest.mark.soc("950")
def test_ilogb_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    base = torch.tensor([-128.0, -8.0, -1.0, -0.75, 0.125, 0.5, 1.0, 31.0], dtype=torch.float32)
    source = base.repeat(ELEMENTS // base.numel()).reshape(1, ELEMENTS)
    special_bits = torch.tensor([0, -2147483648, 1, 0x00400000, 0x007FFFFF, 0x00800000,
                                 0x7F800000, 0x7FC00000], dtype=torch.int32)
    source[0, :8] = special_bits.view(torch.float32)
    output = torch.empty(source.shape, dtype=torch.int32, device=ST_DEVICE)
    simt_ilogb_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    special_expected = torch.tensor([-2147483648, -2147483648, -149, -127, -127, -126,
                                     2147483647, -2147483648], dtype=torch.int32)
    actual = output.cpu()
    torch.testing.assert_close(actual[0, :8], special_expected, rtol=0, atol=0)
    expected = torch.floor(torch.log2(torch.abs(source[:, 8:]))).to(torch.int32)
    torch.testing.assert_close(actual[:, 8:], expected, rtol=0, atol=0)
