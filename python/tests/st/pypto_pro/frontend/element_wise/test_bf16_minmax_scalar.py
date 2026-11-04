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
"""BF16 scalar min/max preserves native FP32 selection, including exceptional values."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


@pl.jit(auto_mutex=False)
def bf16_minmax_vector(
    a: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16],
    b: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16],
    a_fp32: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    b_fp32: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    result: pl.Tensor[[2, pl.DYNAMIC], pl.DT_BF16],
    reference: pl.Tensor[[2, pl.DYNAMIC], pl.DT_FP32],
):
    with pl.section_vector():
        for i in pl.range(a.shape[0]):
            result[0, i] = pl.min(a[i], b[i])
            result[1, i] = pl.max(a[i], b[i])
            reference[0, i] = pl.min(a_fp32[i], b_fp32[i])
            reference[1, i] = pl.max(a_fp32[i], b_fp32[i])


@pl.jit(auto_mutex=False)
def bf16_minmax_cube(
    a: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16],
    b: pl.Tensor[[pl.DYNAMIC], pl.DT_BF16],
    a_fp32: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    b_fp32: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    result: pl.Tensor[[2, pl.DYNAMIC], pl.DT_BF16],
    reference: pl.Tensor[[2, pl.DYNAMIC], pl.DT_FP32],
):
    with pl.section_cube():
        for i in pl.range(a.shape[0]):
            result[0, i] = pl.min(a[i], b[i])
            result[1, i] = pl.max(a[i], b[i])
            reference[0, i] = pl.min(a_fp32[i], b_fp32[i])
            reference[1, i] = pl.max(a_fp32[i], b_fp32[i])


@pytest.mark.soc("950")
@pytest.mark.parametrize("kernel", [bf16_minmax_vector, bf16_minmax_cube], ids=["vector", "cube"])
def test_bf16_minmax_matches_fp32_bits(kernel):
    torch.npu.set_device(ST_DEVICE)
    encodings = torch.arange(65536, dtype=torch.int32)
    # Cartesian pairs cover both operand orders for zeros, subnormals, infinities,
    # and signed quiet/signaling NaNs. Numeric casts could quiet a signaling NaN.
    special = torch.tensor([
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
        0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7f81, 0xff81,
        0x7fc0, 0xffc0, 0x7fff, 0xffff,
    ], dtype=torch.int32)
    left = torch.cat((encodings, special.repeat_interleave(special.numel())))
    right = torch.cat((encodings ^ 0xa55a, special.repeat(special.numel())))
    a = left.to(torch.int16).view(torch.bfloat16).to(ST_DEVICE)
    b = right.to(torch.int16).view(torch.bfloat16).to(ST_DEVICE)
    a_fp32 = (left << 16).view(torch.float32).to(ST_DEVICE)
    b_fp32 = (right << 16).view(torch.float32).to(ST_DEVICE)
    result = torch.empty((2, left.numel()), dtype=torch.bfloat16, device=ST_DEVICE)
    reference = torch.empty((2, left.numel()), dtype=torch.float32, device=ST_DEVICE)

    kernel[None, 1](a, b, a_fp32, b_fp32, result, reference)
    torch.npu.synchronize()

    expected_bits = (reference.cpu().view(torch.int32) >> 16).to(torch.int16)
    torch.testing.assert_close(result.cpu().view(torch.int16), expected_bits, rtol=0, atol=0)


@pl.vector_function(mode="simt")
def _bf16_minmax_callee(a, b, result):
    i = pl.simt.linear_thread_idx()
    result[2, i] = min(a[i], b[i])
    result[3, i] = max(a[i], b[i])


@pl.vector_function(mode="simt", max_threads=64)
def _bf16_minmax_vf(a, b, result):
    i = pl.simt.linear_thread_idx()
    result[0, i] = min(a[i], b[i])
    result[1, i] = max(a[i], b[i])
    _bf16_minmax_callee(a, b, result)


@pl.jit(auto_mutex=False)
def bf16_minmax_simt_vf(
    a: pl.Tensor[[64], pl.DT_BF16],
    b: pl.Tensor[[64], pl.DT_BF16],
    result: pl.Tensor[[4, 64], pl.DT_BF16],
):
    with pl.section_vector():
        _bf16_minmax_vf[64](a, b, result)


@pl.jit(mode="simt", max_threads=64)
def bf16_minmax_simt_kernel(
    a: pl.Tensor[[64], pl.DT_BF16],
    b: pl.Tensor[[64], pl.DT_BF16],
    result: pl.Tensor[[4, 64], pl.DT_BF16],
):
    i = pl.simt.linear_thread_idx()
    result[0, i] = min(a[i], b[i])
    result[1, i] = max(a[i], b[i])
    _bf16_minmax_callee(a, b, result)


@pytest.mark.soc("950")
@pytest.mark.parametrize("pure_simt", [False, True], ids=["vf-and-callee", "kernel-and-callee"])
def test_bf16_minmax_in_simt(pure_simt):
    torch.npu.set_device(ST_DEVICE)
    # Every lane differs; include both selection orders and BF16 extrema.
    a = torch.tensor([-3.5, 2.5, 1.0078125, -1.0078125, 1e30, -1e30, 1e-30, 0.5] * 8, dtype=torch.bfloat16)
    b = a.roll(1)
    result = torch.empty((4, 64), dtype=torch.bfloat16, device=ST_DEVICE)
    kernel = bf16_minmax_simt_kernel if pure_simt else bf16_minmax_simt_vf
    launch = kernel[1, 64] if pure_simt else kernel[None, 1]
    launch(a.to(ST_DEVICE), b.to(ST_DEVICE), result)
    torch.npu.synchronize()
    expected = torch.stack([torch.minimum(a, b), torch.maximum(a, b)] * 2)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
