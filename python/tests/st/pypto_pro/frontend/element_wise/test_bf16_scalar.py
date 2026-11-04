#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ordinary scalar BF16 arithmetic, casts and control flow (no SIMT)."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
DTYPES = [
    (pl.DT_BOOL, torch.bool), (pl.DT_INT8, torch.int8), (pl.DT_UINT8, torch.uint8),
    (pl.DT_INT16, torch.int16), (pl.DT_UINT16, torch.uint16),
    (pl.DT_INT32, torch.int32), (pl.DT_UINT32, torch.uint32),
    (pl.DT_INT64, torch.int64), (pl.DT_UINT64, torch.uint64),
    (pl.DT_FP16, torch.float16), (pl.DT_FP32, torch.float32), (pl.DT_BF16, torch.bfloat16),
]


def _arithmetic(a, b, result, comparisons, control):
    for i in pl.range(a.shape[0]):
        x = a[i]
        y = b[i]
        result[0, i] = x + y
        result[1, i] = x - y
        result[2, i] = x * y
        result[3, i] = x / y
        result[4, i] = x // y
        result[5, i] = x % y
        result[6, i] = -x
        comparisons[0, i] = x == y
        comparisons[1, i] = x != y
        comparisons[2, i] = x < y
        comparisons[3, i] = x <= y
        comparisons[4, i] = x > y
        comparisons[5, i] = x >= y
        comparisons[6, i] = x and y
        comparisons[7, i] = x or y
        comparisons[8, i] = not x
        if x:
            selected = pl.const(1, pl.DT_INT32)
        else:
            selected = pl.const(0, pl.DT_INT32)
        control[0, i] = selected
        # A BF16 loop condition must also use scalar conversion.
        count = pl.const(0, pl.DT_INT32)
        condition = x
        while condition:
            count = count + pl.const(1, pl.DT_INT32)
            condition = pl.const(0, pl.DT_BF16)
        control[1, i] = count
        # The nested addition must round before subtraction.
        result[7, i] = (x + y) - y


def make_arithmetic_kernel(cube):
    @pl.jit(auto_mutex=False)
    def kernel(a: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16], b: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16],
               result: pl.Tensor[[8, pl.DYNAMIC], pl.DT_BF16],
               comparisons: pl.Tensor[[9, pl.DYNAMIC], pl.DT_BOOL],
               control: pl.Tensor[[2, pl.DYNAMIC], pl.DT_INT32]):
        if cube:
            with pl.section_cube():
                _arithmetic(a, b, result, comparisons, control)
        else:
            with pl.section_vector():
                _arithmetic(a, b, result, comparisons, control)
    return kernel


@pytest.mark.soc("950")
@pytest.mark.parametrize("cube", [False, True], ids=["vector", "cube"])
def test_bf16_scalar_arithmetic_and_control(cube):
    torch.npu.set_device(DEVICE)
    a = torch.tensor([-127, -9.5, -7, -1.1, -0.0, 0.0, 0.1, 1.1, 1.5, 7, 127, 256], dtype=torch.bfloat16)
    b = torch.tensor([3, -3, 2, 0.1, -3, 3, 1.1, -0.1, 1.5, -2, 256, 1.1], dtype=torch.bfloat16)
    x, y = a.float(), b.float()
    expected = torch.stack([
        x + y, x - y, x * y, x / y, torch.div(x, y, rounding_mode="floor"), torch.remainder(x, y),
        -x, (x + y).bfloat16().float() - y,
    ]).bfloat16()
    comparisons = torch.stack([
        x == y, x != y, x < y, x <= y, x > y, x >= y,
        x.bool() & y.bool(), x.bool() | y.bool(), ~x.bool(),
    ])
    result = torch.empty_like(expected, device=DEVICE)
    actual_comparisons = torch.empty_like(comparisons, device=DEVICE)
    control = torch.empty((2, a.numel()), dtype=torch.int32, device=DEVICE)
    make_arithmetic_kernel(cube)[None, 1](a.to(DEVICE), b.to(DEVICE), result, actual_comparisons, control)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu().view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
    torch.testing.assert_close(actual_comparisons.cpu(), comparisons)
    torch.testing.assert_close(control.cpu(), x.bool().int().expand(2, -1))


def make_cast_kernel(source, destination, cube=False, implicit=False):
    @pl.jit(auto_mutex=False)
    def kernel(a: pl.Tensor[[pl.DYNAMIC], source], out: pl.Tensor[[pl.DYNAMIC], destination]):
        if cube:
            with pl.section_cube():
                for i in pl.range(a.shape[0]):
                    if implicit:
                        out[i] = a[i]
                    else:
                        out[i] = pl.astype(a[i], destination)
        else:
            with pl.section_vector():
                for i in pl.range(a.shape[0]):
                    if implicit:
                        out[i] = a[i]
                    else:
                        out[i] = pl.astype(a[i], destination)
    return kernel


def _integer_bf16_reference(values):
    encodings = []
    for value in values:
        magnitude = abs(value)
        if magnitude == 0:
            bits = 0
        else:
            exponent = magnitude.bit_length() - 1
            shift = exponent - 7
            if shift <= 0:
                significand = magnitude << -shift
            else:
                significand, remainder = divmod(magnitude, 1 << shift)
                midpoint = 1 << (shift - 1)
                significand += remainder > midpoint or (remainder == midpoint and significand % 2)
            bits = ((exponent + 126) << 7) + significand
        encodings.append(bits | (0x8000 if value < 0 else 0))
    return torch.tensor(encodings, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


@pytest.mark.soc("950")
@pytest.mark.parametrize("dtype, torch_dtype", DTYPES, ids=[str(pair[0]) for pair in DTYPES])
@pytest.mark.parametrize("from_bf16", [False, True], ids=["to_bf16", "from_bf16"])
@pytest.mark.parametrize("cube", [False, True], ids=["vector", "cube"])
@pytest.mark.parametrize("implicit", [False, True], ids=["astype", "store"])
def test_bf16_scalar_cast(dtype, torch_dtype, from_bf16, cube, implicit):
    torch.npu.set_device(DEVICE)
    source, destination = (pl.DT_BF16, dtype) if from_bf16 else (dtype, pl.DT_BF16)
    source_torch, destination_torch = (torch.bfloat16, torch_dtype) if from_bf16 else (torch_dtype, torch.bfloat16)
    values = [0, 1, 2, 3, 7, 15, 63, 127]
    floating = torch_dtype in (torch.float16, torch.float32, torch.bfloat16)
    if floating:
        values = [-float("inf"), -127, -1.1, -0.0, 0.0, 0.1, 1.00390625, 1.01171875,
                  1.1, 127, float("inf"), float("nan")]
    elif torch_dtype != torch.bool and torch.iinfo(torch_dtype).min < 0:
        values += [-127, -3, -1]
    if from_bf16 and not floating and torch_dtype != torch.bool:
        values += [1.5, 2.75]  # Integer conversion truncates, rather than rounding to nearest.
    if not from_bf16 and torch_dtype in (torch.int32, torch.int64, torch.uint32, torch.uint64):
        values += [(128 << 20) + (1 << 19) + offset for offset in (-1, 0, 1)]
    if not from_bf16 and torch_dtype in (torch.int64, torch.uint64):
        values += [(128 << 48) + (1 << 47) + offset for offset in (-1, 0, 1)]
        values += [torch.iinfo(torch_dtype).min, torch.iinfo(torch_dtype).max]
    a = torch.tensor(values, dtype=source_torch)
    expected = a.to(destination_torch)
    if not from_bf16 and not floating:
        expected = _integer_bf16_reference(a.tolist())
    out = torch.empty(a.shape, dtype=destination_torch, device=DEVICE)
    make_cast_kernel(source, destination, cube, implicit)[None, 1](a.to(DEVICE), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.soc("950")
@pytest.mark.parametrize("cube", [False, True], ids=["vector", "cube"])
def test_bf16_to_fp32_all_encodings(cube):
    torch.npu.set_device(DEVICE)
    bits = torch.arange(65536, dtype=torch.int32)
    a = bits.to(torch.int16).view(torch.bfloat16)
    out = torch.empty(a.shape, dtype=torch.float32, device=DEVICE)
    make_cast_kernel(pl.DT_BF16, pl.DT_FP32, cube)[None, 1](a.to(DEVICE), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu().view(torch.int32), bits << 16, rtol=0, atol=0)


@pl.jit(auto_mutex=False)
def bf16_mixed_scalars(a: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16],
                       b: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
                       out: pl.Tensor[[4, pl.DYNAMIC], pl.DT_FP32],
                       comparisons: pl.Tensor[[2, pl.DYNAMIC], pl.DT_BOOL]):
    with pl.section_vector():
        for i in pl.range(a.shape[0]):
            out[0, i] = a[i] + b[i]
            out[1, i] = b[i] + a[i]
            out[2, i] = pl.min(a[i], b[i])
            out[3, i] = pl.min(b[i], a[i])
            comparisons[0, i] = a[i] < b[i]
            comparisons[1, i] = b[i] < a[i]


@pytest.mark.soc("950")
def test_bf16_mixed_scalars():
    torch.npu.set_device(DEVICE)
    a = torch.tensor([-1.1, 0.0, 1.1, 7.0], dtype=torch.bfloat16)
    b = torch.tensor([-1.101, -0.1, 1.101, 7.25], dtype=torch.float32)
    expected = torch.stack([
        a.float() + b, b + a.float(), torch.minimum(a.float(), b), torch.minimum(b, a.float()),
    ])
    expected_comparisons = torch.stack([a.float() < b, b < a.float()])
    out = torch.empty_like(expected, device=DEVICE)
    comparisons = torch.empty_like(expected_comparisons, device=DEVICE)
    bf16_mixed_scalars[None, 1](a.to(DEVICE), b.to(DEVICE), out, comparisons)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(comparisons.cpu(), expected_comparisons)
