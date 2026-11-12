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

"""GM Tensor scalar atomics from SIMD kernels (``pl.atomic_*``).

Covers multi-block contention through one shared address, the per-op update
semantics and returned old values, the wider/float kernels, and the discarded
statement form.
"""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
CORES = 8


def _require_3510():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not 3510 (Ascend950). Skip.")


@pl.jit(arch="3510", auto_mutex=True)
def atomic_contention_kernel(
    counter: pl.Tensor[[1], pl.DT_INT32],
    total: pl.Tensor[[1], pl.DT_INT32],
):
    # Both the count and the sum of the returned old values are published with
    # atomics: plain per-core GM stores would race on shared cache lines.
    with pl.section_vector():
        old = pl.atomic_add(counter, 0, 1)
        pl.atomic_add(total, 0, old)


@pl.jit(arch="3510", auto_mutex=True)
def atomic_semantics_kernel(
    state: pl.Tensor[[8], pl.DT_INT32],
    counter: pl.Tensor[[2], pl.DT_UINT32],
    old: pl.Tensor[[12], pl.DT_INT64],
):
    with pl.section_vector():
        old[0] = pl.atomic_add(state, 0, 5)
        old[1] = pl.atomic_sub(state, 1, 1)
        old[2] = pl.atomic_exch(state, 2, 4)
        old[3] = pl.atomic_max(state, 3, 7)
        old[4] = pl.atomic_min(state, 4, 5)
        old[5] = pl.atomic_and(state, 5, 0b1100)
        old[6] = pl.atomic_or(state, 6, 0b0101)
        old[7] = pl.atomic_xor(state, 7, 0xFF)
        old[8] = pl.atomic_inc(counter, 0, 5)
        old[9] = pl.atomic_dec(counter, 1, 5)
        old[10] = pl.atomic_inc(counter, 0, 5)
        old[11] = pl.atomic_dec(counter, 0, 5)


@pl.jit(arch="3510", auto_mutex=True)
def atomic_cas_kernel(
    state: pl.Tensor[[2], pl.DT_INT32],
    old: pl.Tensor[[2], pl.DT_INT64],
):
    with pl.section_vector():
        old[0] = pl.atomic_cas(state, 0, 3, 9)
        old[1] = pl.atomic_cas(state, 1, 7, 1)


@pl.jit(arch="3510", auto_mutex=True)
def atomic_wide_kernel(
    i64: pl.Tensor[[2], pl.DT_INT64],
    u64: pl.Tensor[[2], pl.DT_UINT64],
    f32: pl.Tensor[[2], pl.DT_FP32],
    old_i64: pl.Tensor[[2], pl.DT_INT64],
    old_u64: pl.Tensor[[2], pl.DT_UINT64],
    old_f32: pl.Tensor[[2], pl.DT_FP32],
):
    with pl.section_vector():
        old_i64[0] = pl.atomic_add(i64, 0, 100)
        old_i64[1] = pl.atomic_cas(i64, 1, 5, 6)
        old_u64[0] = pl.atomic_inc(u64, 0, 9)
        old_u64[1] = pl.atomic_dec(u64, 1, 9)
        old_f32[0] = pl.atomic_add(f32, 0, 1.5)
        old_f32[1] = pl.atomic_max(f32, 1, 2.5)
        pl.atomic_add(i64, 0, 1)  # discarded: the side effect must still land


@pytest.mark.soc("950")
def test_atomic_add_across_blocks_keeps_every_update():
    _require_3510()
    counter = torch.zeros(1, dtype=torch.int32).npu()
    total = torch.zeros(1, dtype=torch.int32).npu()

    atomic_contention_kernel[None, CORES](counter, total)
    torch.npu.synchronize()

    assert counter.cpu().tolist() == [CORES]
    # The old values of CORES serialized increments are exactly 0..CORES-1.
    assert total.cpu().tolist() == [CORES * (CORES - 1) // 2]


@pytest.mark.soc("950")
def test_rmw_semantics_and_returned_old_values():
    _require_3510()
    state = torch.tensor([10, 3, 9, 0, 7, 0b1010, 0b0010, 0b0001], dtype=torch.int32).npu()
    counter = torch.tensor([4, 0], dtype=torch.int32).npu().view(torch.uint32)
    old = torch.full((12,), -1, dtype=torch.int64).npu()

    atomic_semantics_kernel[None, 1](state, counter, old)
    torch.npu.synchronize()

    assert state.cpu().tolist() == [15, 2, 4, 7, 5, 0b1000, 0b0111, 0b11111110]
    assert counter.cpu().tolist() == [5, 5]
    assert old.cpu().tolist() == [10, 3, 9, 0, 7, 0b1010, 0b0010, 0b0001, 4, 0, 5, 0]


@pytest.mark.soc("950")
def test_cas_swaps_only_on_a_matching_old_value():
    _require_3510()
    state = torch.tensor([3, 5], dtype=torch.int32).npu()
    old = torch.full((2,), -1, dtype=torch.int64).npu()

    atomic_cas_kernel[None, 1](state, old)
    torch.npu.synchronize()

    assert state.cpu().tolist() == [9, 5]
    assert old.cpu().tolist() == [3, 5]


@pytest.mark.soc("950")
def test_wide_and_float_atomics_including_a_discarded_call():
    _require_3510()
    i64 = torch.tensor([10, 5], dtype=torch.int64).npu()
    u64 = torch.tensor([4, 0], dtype=torch.int64).npu().view(torch.uint64)
    f32 = torch.tensor([1.0, 3.0], dtype=torch.float32).npu()
    old_i64 = torch.full((2,), -1, dtype=torch.int64).npu()
    old_u64 = torch.zeros(2, dtype=torch.int64).npu().view(torch.uint64)
    old_f32 = torch.full((2,), -1.0, dtype=torch.float32).npu()

    atomic_wide_kernel[None, 1](i64, u64, f32, old_i64, old_u64, old_f32)
    torch.npu.synchronize()

    assert i64.cpu().tolist() == [111, 6]
    assert u64.cpu().tolist() == [5, 9]
    assert f32.cpu().tolist() == [2.5, 3.0]
    assert old_i64.cpu().tolist() == [10, 5]
    assert old_u64.cpu().tolist() == [4, 0]
    assert old_f32.cpu().tolist() == [1.0, 3.0]
