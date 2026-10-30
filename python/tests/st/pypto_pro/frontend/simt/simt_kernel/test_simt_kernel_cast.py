# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Common SIMT cast types, rounding modes, and bitcast operations."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
ROUNDING = ("none", "rint", "round", "floor", "ceil", "trunc")


def _round_float(value, dtype, rounding):
    nearest_tensor = torch.tensor(value, dtype=dtype)
    nearest = nearest_tensor.item()
    if nearest == value or rounding in ("none", "rint"):
        return nearest
    bits_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    bits = nearest_tensor.view(bits_dtype).item()
    step = 1 if nearest >= 0 else -1
    other_bits = bits - step if nearest > value else bits + step
    other = torch.tensor(other_bits, dtype=bits_dtype).view(dtype).item()
    lower, upper = min(nearest, other), max(nearest, other)
    if rounding == "floor":
        return lower
    if rounding == "ceil":
        return upper
    if rounding == "trunc":
        return lower if value > 0 else upper
    if rounding == "odd":
        return nearest if bits & 1 else other
    if value - lower == upper - value:
        return upper if value > 0 else lower
    return nearest


@pl.vector_function(mode="simt")
def _cast_to_fp16(value):
    return pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)


@pl.vector_function(mode="simt")
def _cast_from_fp32(out_fp16, out_bf16, out_int32, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    out_fp16[0, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_NONE)
    out_fp16[1, tid] = _cast_to_fp16(value)
    out_fp16[2, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_ROUND)
    out_fp16[3, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_FLOOR)
    out_fp16[4, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_CEIL)
    out_fp16[5, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_TRUNC)
    out_fp16[6, tid] = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_ODD)
    out_bf16[0, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_NONE)
    out_bf16[1, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
    out_bf16[2, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_ROUND)
    out_bf16[3, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_FLOOR)
    out_bf16[4, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_CEIL)
    out_bf16[5, tid] = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_TRUNC)
    out_int32[0, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_NONE)
    out_int32[1, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_RINT)
    out_int32[2, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_ROUND)
    out_int32[3, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_FLOOR)
    out_int32[4, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_CEIL)
    out_int32[5, tid] = pl.simt.cast(value, pl.DT_INT32, mode=pl.RoundMode.CAST_TRUNC)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_cast_from_fp32(
    out_fp16: pl.Tensor[[7, THREADS], pl.DT_FP16],
    out_bf16: pl.Tensor[[6, THREADS], pl.DT_BF16],
    out_int32: pl.Tensor[[6, THREADS], pl.DT_INT32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _cast_from_fp32(out_fp16, out_bf16, out_int32, src)


@pytest.mark.soc("950")
def test_simt_kernel_cast_from_fp32():
    torch.npu.set_device(ST_DEVICE)
    samples = [
        -50000.5, -140.75, -7.75, -2.5, -1.5, -0.0, 0.0, 0.25,
        0.5, 1.0001, 1.00390625, 1.5, 2.5, 7.75, 140.75, 50000.5,
    ]
    values = torch.tensor(samples * (THREADS // len(samples)), dtype=torch.float32)
    out_fp16 = torch.empty((7, THREADS), dtype=torch.float16, device=ST_DEVICE)
    out_bf16 = torch.empty((6, THREADS), dtype=torch.bfloat16, device=ST_DEVICE)
    out_int32 = torch.empty((6, THREADS), dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_cast_from_fp32[1, THREADS](out_fp16, out_bf16, out_int32, values.to(ST_DEVICE))
    torch.npu.synchronize()

    for output, dtype, modes in (
        (out_fp16, torch.float16, (*ROUNDING, "odd")),
        (out_bf16, torch.bfloat16, ROUNDING),
    ):
        expected = torch.tensor(
            [[_round_float(value, dtype, mode) for value in values.tolist()] for mode in modes], dtype=dtype
        )
        assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
    expected_int32 = []
    for mode in ROUNDING:
        rounded = []
        for value in values.tolist():
            if mode == "rint":
                result = round(value)
            elif mode == "round":
                result = math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
            elif mode == "floor":
                result = math.floor(value)
            elif mode == "ceil":
                result = math.ceil(value)
            else:
                result = math.trunc(value)
            rounded.append(result)
        expected_int32.append(rounded)
    torch.testing.assert_close(out_int32.cpu(), torch.tensor(expected_int32, dtype=torch.int32), rtol=0, atol=0)


@pl.vector_function(mode="simt")
def _cast_from_int32(dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    dst[0, tid] = pl.simt.cast(value, pl.DT_FP32, mode=pl.RoundMode.CAST_NONE)
    dst[1, tid] = pl.simt.cast(value, pl.DT_FP32, mode=pl.RoundMode.CAST_RINT)
    dst[2, tid] = pl.simt.cast(value, pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
    dst[3, tid] = pl.simt.cast(value, pl.DT_FP32, mode=pl.RoundMode.CAST_FLOOR)
    dst[4, tid] = pl.simt.cast(value, pl.DT_FP32, mode=pl.RoundMode.CAST_CEIL)
    dst[5, tid] = pl.simt.cast(value, pl.DT_FP32, mode=pl.RoundMode.CAST_TRUNC)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_cast_from_int32(
    dst: pl.Tensor[[6, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_INT32],
):
    _cast_from_int32(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_cast_from_int32():
    torch.npu.set_device(ST_DEVICE)
    samples = [0, 1, 255, 257, -7, -3, 2**24 + 1, 2**24 + 3]
    values = torch.tensor(samples * (THREADS // len(samples)), dtype=torch.int32)
    dst = torch.empty((6, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_cast_from_int32[1, THREADS](dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    expected = torch.tensor(
        [[_round_float(value, torch.float32, mode) for value in values.tolist()] for mode in ROUNDING],
        dtype=torch.float32,
    )
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


@pl.vector_function(mode="simt")
def _reinterpret(value):
    return pl.simt.bitcast(value, pl.DT_INT32)


@pl.vector_function(mode="simt")
def _bitcast(bits, restored, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    result = _reinterpret(value)
    bits[0, tid] = pl.simt.bitcast(value, pl.DT_INT32)
    bits[1, tid] = result
    restored[tid] = pl.simt.bitcast(result, pl.DT_FP32)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_bitcast(
    bits: pl.Tensor[[2, THREADS], pl.DT_INT32],
    restored: pl.Tensor[[THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _bitcast(bits, restored, src)


@pytest.mark.soc("950")
def test_simt_kernel_bitcast():
    torch.npu.set_device(ST_DEVICE)
    patterns = [0, 0x80000000, 0x3F800000, 0x40200000, 0x7F800000, 0x7FC00001, 0xFF800000, 0xFFFFFFFF]
    raw = torch.tensor(patterns * (THREADS // len(patterns)), dtype=torch.uint32)
    bits = torch.empty((2, THREADS), dtype=torch.int32, device=ST_DEVICE)
    restored = torch.empty((THREADS,), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_bitcast[1, THREADS](bits, restored, raw.view(torch.float32).to(ST_DEVICE))
    torch.npu.synchronize()

    assert torch.equal(bits.cpu(), raw.view(torch.int32).repeat(2, 1))
    assert torch.equal(restored.cpu().view(torch.uint8), raw.view(torch.uint8))
