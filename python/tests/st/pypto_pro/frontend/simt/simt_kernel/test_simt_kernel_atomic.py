# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT kernel GM atomic operations and return values."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
GRID_THREAD_BLOCKS = 4


@pl.vector_function(mode="simt")
def _integer_atomics(state, old, counters, old_counters):
    tid = pl.simt.linear_thread_idx()
    old[0, tid] = pl.simt.atomic_add(state[0, tid], 3)
    old[1, tid] = pl.simt.atomic_sub(state[1, tid], 1)
    old[2, tid] = pl.simt.atomic_exch(state[2, tid], 3)
    old[3, tid] = pl.simt.atomic_max(state[3, tid], 3)
    old[4, tid] = pl.simt.atomic_min(state[4, tid], 3)
    old[5, tid] = pl.simt.atomic_cas(state[5, tid], 7, 3)
    old[6, tid] = pl.simt.atomic_and(state[6, tid], 3)
    old[7, tid] = pl.simt.atomic_or(state[7, tid], 3)
    old[8, tid] = pl.simt.atomic_xor(state[8, tid], 3)
    old_counters[0, tid] = pl.simt.atomic_inc(counters[0, tid], 5)
    old_counters[1, tid] = pl.simt.atomic_dec(counters[1, tid], 5)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_integer_atomics(
    state: pl.Tensor[[9, THREADS], pl.DT_INT32],
    old: pl.Tensor[[9, THREADS], pl.DT_INT32],
    counters: pl.Tensor[[2, THREADS], pl.DT_UINT32],
    old_counters: pl.Tensor[[2, THREADS], pl.DT_UINT32],
):
    _integer_atomics(state, old, counters, old_counters)


@pytest.mark.soc("950")
def test_simt_kernel_integer_atomics():
    torch.npu.set_device(ST_DEVICE)
    values = [1, 3, 7, 9]
    initial = torch.tensor(values * (THREADS // len(values)), dtype=torch.int64).repeat(9, 1)
    expected = initial.clone()
    expected[0] += 3
    expected[1] -= 1
    expected[2].fill_(3)
    expected[3].clamp_min_(3)
    expected[4].clamp_max_(3)
    expected[5] = torch.where(initial[5] == 7, 3, initial[5])
    expected[6] &= 3
    expected[7] |= 3
    expected[8] ^= 3
    initial_counters = torch.tensor([0, 4, 5, 6] * (THREADS // 4), dtype=torch.int64).repeat(2, 1)
    expected_counters = torch.tensor([[1, 5, 0, 0] * (THREADS // 4), [5, 3, 4, 5] * (THREADS // 4)], dtype=torch.int64)
    state = initial.to(torch.int32).to(ST_DEVICE)
    old = torch.empty_like(state)
    counters = initial_counters.to(torch.uint32).to(ST_DEVICE)
    old_counters = torch.empty_like(counters)
    simt_kernel_integer_atomics[1, THREADS](state, old, counters, old_counters)
    torch.npu.synchronize()

    assert torch.equal(state.cpu().to(torch.int64), expected)
    assert torch.equal(old.cpu().to(torch.int64), initial)
    assert torch.equal(counters.cpu().to(torch.int64), expected_counters)
    assert torch.equal(old_counters.cpu().to(torch.int64), initial_counters)


@pl.vector_function(mode="simt")
def _atomic_tickets(state, tickets):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    tickets[index] = pl.simt.atomic_add(state[0], 1)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_atomic_tickets(
    state: pl.Tensor[[1], pl.DT_INT32],
    tickets: pl.Tensor[[GRID_THREAD_BLOCKS * THREADS], pl.DT_INT32],
):
    _atomic_tickets(state, tickets)


@pytest.mark.soc("950")
def test_simt_kernel_atomic_add_contention():
    torch.npu.set_device(ST_DEVICE)
    count = GRID_THREAD_BLOCKS * THREADS
    state = torch.zeros(1, dtype=torch.int32, device=ST_DEVICE)
    tickets = torch.full((count,), -1, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_atomic_tickets[GRID_THREAD_BLOCKS, THREADS](state, tickets)
    torch.npu.synchronize()

    assert state.cpu().item() == count
    assert torch.equal(tickets.cpu().sort().values, torch.arange(count, dtype=torch.int32))


@pl.vector_function(mode="simt")
def _float_atomics(state):
    pl.simt.atomic_add(state[0, 0], 1.0)
    pl.simt.atomic_max(state[1, 0], 2.0)
    pl.simt.atomic_min(state[2, 0], -2.0)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_float_atomics(state: pl.Tensor[[3, 1], pl.DT_FP32]):
    _float_atomics(state)


@pytest.mark.soc("950")
def test_simt_kernel_float_atomics():
    torch.npu.set_device(ST_DEVICE)
    state = torch.zeros((3, 1), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_float_atomics[GRID_THREAD_BLOCKS, THREADS](state)
    torch.npu.synchronize()

    expected = torch.tensor([[GRID_THREAD_BLOCKS * THREADS], [2], [-2]], dtype=torch.float32)
    torch.testing.assert_close(state.cpu(), expected, rtol=0, atol=0)


@pl.vector_function(mode="simt")
def _float_atomic_returns(state, old):
    tid = pl.simt.linear_thread_idx()
    old[0, tid] = pl.simt.atomic_add(state[0, tid], 3.0)
    old[1, tid] = pl.simt.atomic_sub(state[1, tid], 1.0)
    old[2, tid] = pl.simt.atomic_exch(state[2, tid], 3.0)
    old[3, tid] = pl.simt.atomic_max(state[3, tid], 3.0)
    old[4, tid] = pl.simt.atomic_min(state[4, tid], 3.0)
    old[5, tid] = pl.simt.atomic_cas(state[5, tid], 7.0, 3.0)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_float_atomic_returns(
    state: pl.Tensor[[6, THREADS], pl.DT_FP32],
    old: pl.Tensor[[6, THREADS], pl.DT_FP32],
):
    _float_atomic_returns(state, old)


@pytest.mark.soc("950")
def test_simt_kernel_float_atomic_return_values():
    torch.npu.set_device(ST_DEVICE)
    initial = torch.tensor([-1.5, 3.0, 7.0, 9.0] * (THREADS // 4), dtype=torch.float32).repeat(6, 1)
    expected = initial.clone()
    expected[0] += 3.0
    expected[1] -= 1.0
    expected[2].fill_(3.0)
    expected[3].clamp_min_(3.0)
    expected[4].clamp_max_(3.0)
    expected[5] = torch.where(initial[5] == 7.0, 3.0, initial[5])
    state = initial.to(ST_DEVICE)
    old = torch.empty_like(state)
    simt_kernel_float_atomic_returns[1, THREADS](state, old)
    torch.npu.synchronize()
    torch.testing.assert_close(state.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(old.cpu(), initial, rtol=0, atol=0)


@pl.vector_function(mode="simt")
def _contended_atomics(state, old, counters, old_counters):
    tid = pl.simt.linear_thread_idx()
    old[0, tid] = pl.simt.atomic_add(state[0, 0], 3)
    old[1, tid] = pl.simt.atomic_sub(state[1, 0], 1)
    old[2, tid] = pl.simt.atomic_exch(state[2, 0], 3)
    old[3, tid] = pl.simt.atomic_max(state[3, 0], 3)
    old[4, tid] = pl.simt.atomic_min(state[4, 0], 3)
    old[5, tid] = pl.simt.atomic_cas(state[5, 0], 7, 3)
    old[6, tid] = pl.simt.atomic_and(state[6, 0], 3)
    old[7, tid] = pl.simt.atomic_or(state[7, 0], 3)
    old[8, tid] = pl.simt.atomic_xor(state[8, 0], 3)
    old_counters[0, tid] = pl.simt.atomic_inc(counters[0, 0], 5)
    old_counters[1, tid] = pl.simt.atomic_dec(counters[1, 0], 5)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_contended_atomics(
    state: pl.Tensor[[9, 1], pl.DT_INT32],
    old: pl.Tensor[[9, THREADS], pl.DT_INT32],
    counters: pl.Tensor[[2, 1], pl.DT_UINT32],
    old_counters: pl.Tensor[[2, THREADS], pl.DT_UINT32],
):
    _contended_atomics(state, old, counters, old_counters)


@pytest.mark.soc("950")
def test_simt_kernel_all_atomics_under_contention():
    torch.npu.set_device(ST_DEVICE)
    initial = [7, 100, 7, 1, 7, 7, 7, 4, 4]
    state = torch.tensor(initial, dtype=torch.int32, device=ST_DEVICE).reshape(9, 1)
    old = torch.empty((9, THREADS), dtype=torch.int32, device=ST_DEVICE)
    counters = torch.zeros((2, 1), dtype=torch.uint32).to(ST_DEVICE)
    old_counters = torch.empty((2, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    simt_kernel_contended_atomics[1, THREADS](state, old, counters, old_counters)
    torch.npu.synchronize()

    expected_state = [7 + 3 * THREADS, 100 - THREADS, 3, 3, 3, 3, 3, 7, 4]
    expected_old = torch.tensor(
        [
            [7 + 3 * index for index in range(THREADS)],
            [100 - index for index in range(THREADS)],
            [7] + [3] * (THREADS - 1),
            [1] + [3] * (THREADS - 1),
            [7] + [3] * (THREADS - 1),
            [7] + [3] * (THREADS - 1),
            [7] + [3] * (THREADS - 1),
            [4] + [7] * (THREADS - 1),
            [4, 7] * (THREADS // 2),
        ],
        dtype=torch.int32,
    )
    inc, dec = 0, 0
    inc_old, dec_old = [], []
    for _ in range(THREADS):
        inc_old.append(inc)
        dec_old.append(dec)
        inc = 0 if inc >= 5 else inc + 1
        dec = 5 if dec == 0 or dec > 5 else dec - 1
    assert torch.equal(state.cpu().flatten(), torch.tensor(expected_state, dtype=torch.int32))
    assert torch.equal(old.cpu().sort(dim=1).values, expected_old.sort(dim=1).values)
    assert counters.cpu().flatten().tolist() == [inc, dec]
    expected_counter_old = torch.tensor([inc_old, dec_old], dtype=torch.int64)
    assert torch.equal(old_counters.cpu().to(torch.int64).sort(dim=1).values, expected_counter_old.sort(dim=1).values)
