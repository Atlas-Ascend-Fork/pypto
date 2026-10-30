# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT kernel bit manipulation and integer arithmetic with 32-bit operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64


@pl.vector_function(mode="simt")
def _bit_manipulation(dst, counts, lhs, rhs, selectors, signed_src):
    tid = pl.simt.linear_thread_idx()
    x = lhs[tid]
    y = rhs[tid]
    signed = signed_src[tid]
    dst[0, tid] = pl.simt.brev(x)
    dst[1, tid] = pl.simt.byte_perm(x, y, selectors[tid])
    counts[0, tid] = pl.simt.popcount(x)
    counts[1, tid] = pl.simt.clz(signed)
    counts[2, tid] = pl.simt.ffs(signed)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_bit_manipulation(
    dst: pl.Tensor[[2, THREADS], pl.DT_UINT32],
    counts: pl.Tensor[[3, THREADS], pl.DT_INT32],
    lhs: pl.Tensor[[THREADS], pl.DT_UINT32],
    rhs: pl.Tensor[[THREADS], pl.DT_UINT32],
    selectors: pl.Tensor[[THREADS], pl.DT_UINT32],
    signed_src: pl.Tensor[[THREADS], pl.DT_INT32],
):
    _bit_manipulation(dst, counts, lhs, rhs, selectors, signed_src)


@pytest.mark.soc("950")
def test_simt_kernel_bit_manipulation():
    torch.npu.set_device(ST_DEVICE)
    lhs_values = [0, 1, 2, 0x80000000, 0xFFFFFFFF, 0x12345678, 0x55555555, 0x00010000] * (THREADS // 8)
    rhs_values = list(reversed(lhs_values))
    selectors = [0x0123, 0x4567, 0x0011, 0x2233, 0x4455, 0x6677, 0x7654, 0x3210] * (THREADS // 8)
    lhs = torch.tensor(lhs_values, dtype=torch.uint32)
    rhs = torch.tensor(rhs_values, dtype=torch.uint32)
    selector_tensor = torch.tensor(selectors, dtype=torch.uint32)
    dst = torch.empty((2, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    counts = torch.empty((3, THREADS), dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_bit_manipulation[1, THREADS](
        dst,
        counts,
        lhs.to(ST_DEVICE),
        rhs.to(ST_DEVICE),
        selector_tensor.to(ST_DEVICE),
        lhs.view(torch.int32).to(ST_DEVICE),
    )
    torch.npu.synchronize()

    permuted = []
    for x, y, selector in zip(lhs_values, rhs_values, selectors):
        source = x | (y << 32)
        permuted.append(
            sum(((source >> (8 * ((selector >> (4 * index)) & 7))) & 0xFF) << (8 * index) for index in range(4))
        )
    expected = torch.tensor(
        [[int(f"{value:032b}"[::-1], 2) for value in lhs_values], permuted], dtype=torch.uint32
    )
    expected_counts = torch.tensor(
        [
            [value.bit_count() for value in lhs_values],
            [32 - value.bit_length() for value in lhs_values],
            [(value & -value).bit_length() for value in lhs_values],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(dst.cpu().to(torch.int64), expected.to(torch.int64))
    assert torch.equal(counts.cpu(), expected_counts)


@pl.vector_function(mode="simt")
def _integer_arithmetic(dst, wide, differences, lhs, rhs, addends):
    tid = pl.simt.linear_thread_idx()
    x = lhs[tid]
    y = rhs[tid]
    dst[0, tid] = pl.simt.hadd(x, y)
    dst[1, tid] = pl.simt.rhadd(x, y)
    dst[2, tid] = pl.simt.mul24(x, y)
    dst[3, tid] = pl.simt.mul_hi(x, y)
    wide[tid] = pl.simt.mul_i32_to_i64(x, y)
    differences[tid] = pl.simt.sad(x, y, addends[tid])


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_integer_arithmetic(
    dst: pl.Tensor[[4, THREADS], pl.DT_INT32],
    wide: pl.Tensor[[THREADS], pl.DT_INT64],
    differences: pl.Tensor[[THREADS], pl.DT_UINT32],
    lhs: pl.Tensor[[THREADS], pl.DT_INT32],
    rhs: pl.Tensor[[THREADS], pl.DT_INT32],
    addends: pl.Tensor[[THREADS], pl.DT_UINT32],
):
    _integer_arithmetic(dst, wide, differences, lhs, rhs, addends)


@pytest.mark.soc("950")
def test_simt_kernel_integer_arithmetic():
    torch.npu.set_device(ST_DEVICE)
    lhs_values = [-100000, -17, -1, 0, 1, 17, 100000, 8388607] * (THREADS // 8)
    rhs_values = [100000, -2, 3, -1, 2, -3, 100000, -7] * (THREADS // 8)
    lhs = torch.tensor(lhs_values, dtype=torch.int32)
    rhs = torch.tensor(rhs_values, dtype=torch.int32)
    addends = torch.arange(THREADS, dtype=torch.int64).to(torch.uint32)
    dst = torch.empty((4, THREADS), dtype=torch.int32, device=ST_DEVICE)
    wide = torch.empty(THREADS, dtype=torch.int64, device=ST_DEVICE)
    differences = torch.empty(THREADS, dtype=torch.uint32, device=ST_DEVICE)
    simt_kernel_integer_arithmetic[1, THREADS](
        dst, wide, differences, lhs.to(ST_DEVICE), rhs.to(ST_DEVICE), addends.to(ST_DEVICE)
    )
    torch.npu.synchronize()

    products = [x * y for x, y in zip(lhs_values, rhs_values)]
    expected = torch.tensor(
        [
            [(x + y) // 2 for x, y in zip(lhs_values, rhs_values)],
            [(x + y + 1) // 2 for x, y in zip(lhs_values, rhs_values)],
            [value & 0xFFFFFFFF for value in products],
            [value >> 32 for value in products],
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    expected_differences = (lhs.to(torch.int64) - rhs.to(torch.int64)).abs() + addends.to(torch.int64)
    assert torch.equal(dst.cpu(), expected)
    assert torch.equal(wide.cpu(), torch.tensor(products, dtype=torch.int64))
    assert torch.equal(differences.cpu().to(torch.int64), expected_differences)
