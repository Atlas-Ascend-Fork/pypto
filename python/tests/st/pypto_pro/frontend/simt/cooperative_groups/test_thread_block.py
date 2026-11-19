# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""System tests for the ThreadBlock cooperative group."""

import os

import pypto_pro.language as pl
import pytest
import torch

pytestmark = pytest.mark.soc("950")
ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
THREADS = 64


# SIMT function
@pl.vector_function(mode="simt", max_threads=THREADS)
def _block_probe(scratch, out: pl.Tensor[[15, THREADS], pl.DT_UINT32]):
    block = pl.simt.this_thread_block()
    tid = block.thread_rank()
    thread = block.thread_index()
    index = block.group_index()
    dims = block.group_dim()
    out[0, tid] = tid
    out[1, tid] = block.size()
    out[2, tid] = thread.x
    out[3, tid] = thread.y
    out[4, tid] = thread.z
    out[5, tid] = index.x
    out[6, tid] = index.y
    out[7, tid] = index.z
    out[8, tid] = dims.x
    out[9, tid] = dims.y
    out[10, tid] = dims.z
    for phase in pl.range(0, 4, 1):
        scratch[0, tid] = pl.simt.cast(phase * THREADS + tid, pl.DT_UINT32)
        block.sync()
        out[11 + phase, tid] = scratch[0, tid ^ 32]
        # Finish all reads before any thread reuses the shared storage.
        block.sync()


# JIT kernel
@pl.jit(mode="simt", max_threads=THREADS)
def _block_queries_kernel(out: pl.Tensor[[11, 2 * THREADS], pl.DT_UINT32]):
    block = pl.simt.this_thread_block()
    tid = block.thread_rank()
    thread = block.thread_index()
    index = block.group_index()
    dims = block.group_dim()
    slot = index.x * THREADS + tid
    out[0, slot] = tid
    out[1, slot] = block.size()
    out[2, slot] = thread.x
    out[3, slot] = thread.y
    out[4, slot] = thread.z
    out[5, slot] = index.x
    out[6, slot] = index.y
    out[7, slot] = index.z
    out[8, slot] = dims.x
    out[9, slot] = dims.y
    out[10, slot] = dims.z
    block.sync()


@pl.jit(arch="3510")
def _block_kernel(out: pl.Tensor[[15, THREADS], pl.DT_UINT32]):
    scratch = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0,
    )
    with pl.section_vector():
        _block_probe[8, 4, 2](scratch, out)


# Host launch and validation
@pytest.mark.parametrize(
    "blocks,shape",
    [(1, (1, 1, 1)), (1, (31, 1, 1)), (2, (33, 1, 1)), (2, (8, 8, 1)), (2, (8, 4, 2))],
)
def test_thread_block_standalone_queries(blocks, shape):
    torch.npu.set_device(ST_DEVICE)
    rows = [[0xFFFFFFFF] * (2 * THREADS) for _ in range(11)]
    out = torch.tensor(rows, dtype=torch.uint32, device=ST_DEVICE)
    _block_queries_kernel[blocks, shape](out)
    torch.npu.synchronize()
    threads = shape[0] * shape[1] * shape[2]
    for block in range(blocks):
        for tid in range(threads):
            expected = (
                tid, threads, tid % shape[0], tid // shape[0] % shape[1],
                tid // (shape[0] * shape[1]), block, 0, 0, *shape,
            )
            for row, value in zip(rows, expected):
                row[block * THREADS + tid] = value
    torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)


def test_thread_block_queries_and_barriers():
    out = torch.empty((15, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    _block_kernel[None, 1](out)
    torch.npu.synchronize()
    rows = [
        list(range(THREADS)),
        [THREADS] * THREADS,
        [i % 8 for i in range(THREADS)],
        [(i // 8) % 4 for i in range(THREADS)],
        [i // 32 for i in range(THREADS)],
    ]
    rows += [[0] * THREADS for _ in range(3)]
    rows += [[dim] * THREADS for dim in (8, 4, 2)]
    rows += [[phase * THREADS + (i ^ 32) for i in range(THREADS)] for phase in range(4)]
    torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)
