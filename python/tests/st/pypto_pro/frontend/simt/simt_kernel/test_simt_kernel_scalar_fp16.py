# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT kernel scalar interfaces restricted to FP16/BF16, using FP16 operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64


@pl.vector_function(mode="simt")
def _half_math(dst, lhs, rhs):
    tid = pl.simt.linear_thread_idx()
    x = lhs[tid]
    y = rhs[tid]
    dst[0, tid] = pl.simt.rcp(x)
    dst[1, tid] = pl.simt.fma_relu(x, y, x)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_half_math(
    dst: pl.Tensor[[2, THREADS], pl.DT_FP16],
    lhs: pl.Tensor[[THREADS], pl.DT_FP16],
    rhs: pl.Tensor[[THREADS], pl.DT_FP16],
):
    _half_math(dst, lhs, rhs)


@pytest.mark.soc("950")
def test_simt_kernel_half_math():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([-4.0, -2.0, -1.0, -0.5, 0.25, 0.5, 1.0, 4.0] * (THREADS // 8), dtype=torch.float16)
    rhs = torch.tensor([1.5, -0.25, 2.0, -2.0, 4.0, -1.0, 0.0, 0.5] * (THREADS // 8), dtype=torch.float16)
    dst = torch.empty((2, THREADS), dtype=torch.float16, device=ST_DEVICE)
    simt_kernel_half_math[1, THREADS](dst, lhs.to(ST_DEVICE), rhs.to(ST_DEVICE))
    torch.npu.synchronize()

    x, y = lhs.float(), rhs.float()
    expected = torch.stack([x.reciprocal(), (x * y + x).clamp_min(0.0)]).half()
    torch.testing.assert_close(dst.cpu(), expected, rtol=5e-3, atol=5e-3)


@pl.vector_function(mode="simt")
def _half_comparisons(dst, predicates, lhs, rhs):
    tid = pl.simt.linear_thread_idx()
    x = lhs[tid]
    y = rhs[tid]
    dst[0, tid] = pl.simt.max_nan(x, y)
    dst[1, tid] = pl.simt.min_nan(x, y)
    predicates[0, tid] = pl.simt.equ(x, y)
    predicates[1, tid] = pl.simt.geu(x, y)
    predicates[2, tid] = pl.simt.gtu(x, y)
    predicates[3, tid] = pl.simt.leu(x, y)
    predicates[4, tid] = pl.simt.ltu(x, y)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_half_comparisons(
    dst: pl.Tensor[[2, THREADS], pl.DT_FP16],
    predicates: pl.Tensor[[5, THREADS], pl.DT_BOOL],
    lhs: pl.Tensor[[THREADS], pl.DT_FP16],
    rhs: pl.Tensor[[THREADS], pl.DT_FP16],
):
    _half_comparisons(dst, predicates, lhs, rhs)


@pytest.mark.soc("950")
def test_simt_kernel_half_comparisons():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * (THREADS // 8), dtype=torch.float16)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * (THREADS // 8), dtype=torch.float16)
    dst = torch.empty((2, THREADS), dtype=torch.float16, device=ST_DEVICE)
    predicates = torch.empty((5, THREADS), dtype=torch.bool, device=ST_DEVICE)
    simt_kernel_half_comparisons[1, THREADS](dst, predicates, lhs.to(ST_DEVICE), rhs.to(ST_DEVICE))
    torch.npu.synchronize()

    expected = torch.stack([torch.maximum(lhs, rhs), torch.minimum(lhs, rhs)])
    unordered = torch.isnan(lhs) | torch.isnan(rhs)
    expected_predicates = torch.stack(
        [
            unordered | (lhs == rhs),
            unordered | (lhs >= rhs),
            unordered | (lhs > rhs),
            unordered | (lhs <= rhs),
            unordered | (lhs < rhs),
        ]
    )
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(predicates.cpu(), expected_predicates)
