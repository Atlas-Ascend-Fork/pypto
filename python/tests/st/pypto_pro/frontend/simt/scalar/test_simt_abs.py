# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 generalized system test for the SIMT abs interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 2048

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def abs_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    src_int32,
    src_int64,
    out_fp16,
    out_bf16,
    out_fp32,
    out_int32,
    out_int64,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.abs(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.abs(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.abs(src_fp32[0, tid])
    out_int32[0, tid] = pl.simt.abs(src_int32[0, tid])
    out_int64[0, tid] = pl.simt.abs(src_int64[0, tid])


@pl.jit(auto_mutex=True)
def simt_abs_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    src_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_int32: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out_int64: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
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
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    src_bf16_tile = src_bf16_tile_group.current()
    src_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    src_fp32_tile = src_fp32_tile_group.current()
    src_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    src_int64_tile = src_int64_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x9000,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0xA000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    out_int64_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0xC000,
        mutex_ids="auto",
        depth=1,
    )
    out_int64_tile = out_int64_tile_group.current()
    with pl.section_vector():
        pl.load(src_fp16_tile, src_fp16, [0, 0])
        pl.load(src_bf16_tile, src_bf16, [0, 0])
        pl.load(src_fp32_tile, src_fp32, [0, 0])
        pl.load(src_int64_tile, src_int64, [0, 0])
        abs_all_dtypes[32, 64](
            src_fp16_tile,
            src_bf16_tile,
            src_fp32_tile,
            src_int32,
            src_int64_tile,
            out_fp16_tile,
            out_bf16_tile,
            out_fp32_tile,
            out_int32,
            out_int64_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])
        pl.store(out_int64, out_int64_tile, [0, 0])


@pytest.mark.soc("950")
def test_abs_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    fp32 = torch.linspace(-8.0, 8.0, ELEMENTS).reshape(1, ELEMENTS)
    fp32[0, :6] = torch.tensor([-0.0, 0.0, float("-inf"), float("inf"), float("nan"), -1.0])
    float_sources = (fp32.to(torch.float16), fp32.to(torch.bfloat16), fp32)
    # Preserve raw NaN payloads (including signaling NaNs), signed zeros and subnormals.
    low_precision_bits = torch.tensor(
        [
            0x8000, 0x0000, 0x0001, 0x8001, 0x03FF, 0x83FF,
            0x7C00, 0xFC00, 0x7C01, 0xFC01, 0x7E01, 0xFE01, 0x7FFF, 0xFFFF,
            0x7F80, 0xFF80, 0x7F81, 0xFF81, 0x7FC1, 0xFFC1,
        ],
        dtype=torch.int32,
    ).to(torch.int16)
    for source in float_sources[:2]:
        source.view(torch.int16)[0, :low_precision_bits.numel()] = low_precision_bits
    int32_source = (torch.arange(ELEMENTS, dtype=torch.int32) - 32).reshape(1, ELEMENTS)
    int64_source = (torch.arange(ELEMENTS, dtype=torch.int64) - 32).reshape(1, ELEMENTS)
    float_outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in float_sources)
    int32_output = torch.empty_like(int32_source, device=ST_DEVICE)
    int64_output = torch.empty_like(int64_source, device=ST_DEVICE)

    simt_abs_all_dtypes(
        *(source.to(ST_DEVICE) for source in float_sources),
        int32_source.to(ST_DEVICE),
        int64_source.to(ST_DEVICE),
        *float_outputs,
        int32_output,
        int64_output,
    )
    torch.npu.synchronize()

    for source, output in zip(float_sources, float_outputs):
        expected = torch.abs(source.to(torch.float32)).to(source.dtype)
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (2e-2, 2e-2),
            torch.float32: (1e-5, 1e-6),
        }[source.dtype]
        torch.testing.assert_close(output.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True)
        if source.dtype in (torch.float16, torch.bfloat16):
            # Only the sign bit changes, even for NaNs.
            expected_bits = source.view(torch.int16) & 0x7FFF
            torch.testing.assert_close(output.cpu().view(torch.int16), expected_bits, rtol=0, atol=0)
    torch.testing.assert_close(int32_output.cpu(), torch.abs(int32_source), rtol=0, atol=0)
    torch.testing.assert_close(int64_output.cpu(), torch.abs(int64_source), rtol=0, atol=0)
    assert not torch.signbit(float_outputs[2].cpu()[0, 0]).item()
