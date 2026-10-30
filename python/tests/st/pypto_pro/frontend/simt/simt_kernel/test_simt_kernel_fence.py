# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT kernel barriers and publication with thread block and device memory fences."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
GRID_THREAD_BLOCKS = 4


@pl.vector_function(mode="simt")
def _read_value(values, index):
    return values[index]


@pl.vector_function(mode="simt")
def _exchange_neighbors(out, values):
    tid = pl.simt.linear_thread_idx()
    offset = pl.simt.block_idx().x * THREADS
    values[offset + tid] = pl.simt.cast(offset + tid, pl.DT_INT32)
    pl.simt.syncthreads()
    out[offset + tid] = _read_value(values, offset + (tid + THREADS // 2) % THREADS)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_syncthreads(
    out: pl.Tensor[[GRID_THREAD_BLOCKS * THREADS], pl.DT_INT32],
    values: pl.Tensor[[GRID_THREAD_BLOCKS * THREADS], pl.DT_INT32],
):
    _exchange_neighbors(out, values)


@pytest.mark.soc("950")
def test_simt_kernel_syncthreads():
    torch.npu.set_device(ST_DEVICE)
    out = torch.full((GRID_THREAD_BLOCKS * THREADS,), -1, dtype=torch.int32, device=ST_DEVICE)
    values = torch.full_like(out, -1)
    simt_kernel_syncthreads[GRID_THREAD_BLOCKS, THREADS](out, values)
    torch.npu.synchronize()
    expected = torch.arange(GRID_THREAD_BLOCKS * THREADS, dtype=torch.int32).reshape(GRID_THREAD_BLOCKS, THREADS)
    assert torch.equal(out.cpu(), expected.roll(THREADS // 2, dims=1).flatten())


@pl.vector_function(mode="simt")
def _publish_in_thread_block(out, values, completed):
    tid = pl.simt.linear_thread_idx()
    values[tid] = 1
    pl.simt.threadfence_block()
    ticket = pl.simt.atomic_add(completed[0], 1)
    if ticket == THREADS - 1:
        total: pl.DT_INT32 = 0
        for index in pl.range(THREADS):
            total = total + values[index]
        out[0] = total


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_threadfence_block(
    out: pl.Tensor[[1], pl.DT_INT32],
    values: pl.Tensor[[THREADS], pl.DT_INT32],
    completed: pl.Tensor[[1], pl.DT_INT32],
):
    _publish_in_thread_block(out, values, completed)


@pytest.mark.soc("950")
def test_simt_kernel_threadfence_block():
    torch.npu.set_device(ST_DEVICE)
    out = torch.full((1,), -1, dtype=torch.int32, device=ST_DEVICE)
    values = torch.zeros(THREADS, dtype=torch.int32, device=ST_DEVICE)
    completed = torch.zeros(1, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_threadfence_block[1, THREADS](out, values, completed)
    torch.npu.synchronize()

    assert completed.cpu().item() == THREADS
    assert out.cpu().item() == THREADS


@pl.vector_function(mode="simt")
def _threadfence(out, values, completed):
    thread_block_id = pl.simt.block_idx().x
    values[thread_block_id] = 1
    pl.simt.threadfence()
    ticket = pl.simt.atomic_add(completed[0], 1)
    if ticket == GRID_THREAD_BLOCKS - 1:
        total: pl.DT_INT32 = 0
        for index in pl.range(GRID_THREAD_BLOCKS):
            total = total + values[index]
        out[0] = total


@pl.jit(mode="simt", max_threads=1)
def simt_kernel_threadfence(
    out: pl.Tensor[[1], pl.DT_INT32],
    values: pl.Tensor[[GRID_THREAD_BLOCKS], pl.DT_INT32],
    completed: pl.Tensor[[1], pl.DT_INT32],
):
    _threadfence(out, values, completed)


@pytest.mark.soc("950")
def test_simt_kernel_threadfence():
    torch.npu.set_device(ST_DEVICE)
    out = torch.full((1,), -1, dtype=torch.int32, device=ST_DEVICE)
    values = torch.zeros(GRID_THREAD_BLOCKS, dtype=torch.int32, device=ST_DEVICE)
    completed = torch.zeros(1, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_threadfence[GRID_THREAD_BLOCKS, 1](out, values, completed)
    torch.npu.synchronize()

    assert completed.cpu().item() == GRID_THREAD_BLOCKS
    assert out.cpu().item() == GRID_THREAD_BLOCKS
