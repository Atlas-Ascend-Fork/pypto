# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT context queries for one-, two-, and three-dimensional launches."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
ELEMENTS = 6 * THREADS


@pl.vector_function(mode="simt")
def _write_context(out):
    thread = pl.simt.thread_idx()
    thread_block = pl.simt.block_dim()
    thread_block_id = pl.simt.block_idx()
    grid = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()
    index = (thread_block_id.x + grid.x * (thread_block_id.y + grid.y * thread_block_id.z)) * THREADS + tid
    out[0, index] = thread.x
    out[1, index] = thread.y
    out[2, index] = thread.z
    out[3, index] = thread_block.x
    out[4, index] = thread_block.y
    out[5, index] = thread_block.z
    out[6, index] = thread_block_id.x
    out[7, index] = thread_block_id.y
    out[8, index] = thread_block_id.z
    out[9, index] = grid.x
    out[10, index] = grid.y
    out[11, index] = grid.z
    out[12, index] = tid
    out[13, index] = pl.simt.cast(pl.simt.warp_size(), pl.DT_UINT32)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_context(out: pl.Tensor[[14, ELEMENTS], pl.DT_UINT32]):
    _write_context(out)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "grid, thread_block",
    [((6, 1, 1), (64, 1, 1)), ((3, 2, 1), (16, 4, 1)), ((2, 1, 3), (8, 4, 2))],
)
def test_simt_kernel_context(grid, thread_block):
    torch.npu.set_device(ST_DEVICE)
    records = []
    for thread_block_index in range(math.prod(grid)):
        thread_block_id = (
            thread_block_index % grid[0],
            thread_block_index // grid[0] % grid[1],
            thread_block_index // (grid[0] * grid[1]),
        )
        for tid in range(math.prod(thread_block)):
            thread = (
                tid % thread_block[0],
                tid // thread_block[0] % thread_block[1],
                tid // (thread_block[0] * thread_block[1]),
            )
            records.append([*thread, *thread_block, *thread_block_id, *grid, tid, 32])
    expected = torch.tensor(records, dtype=torch.int64).T.contiguous()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = torch.empty((14, ELEMENTS), dtype=torch.uint32, device=ST_DEVICE)
        simt_kernel_context[stream, grid, thread_block](out)
    stream.synchronize()
    assert torch.equal(out.cpu().to(torch.int64), expected)


@pl.vector_function(mode="simt")
def _write_grid_context(out):
    grid = pl.simt.grid_dim()
    thread_block_id = pl.simt.block_idx()
    index = thread_block_id.x + grid.x * (thread_block_id.y + grid.y * thread_block_id.z)
    out[0, index] = grid.x
    out[1, index] = grid.y
    out[2, index] = grid.z
    out[3, index] = thread_block_id.x
    out[4, index] = thread_block_id.y
    out[5, index] = thread_block_id.z


@pl.jit(mode="simt", max_threads=1)
def simt_kernel_grid_3d(out: pl.Tensor[[6, 24], pl.DT_INT32]):
    _write_grid_context(out)


@pytest.mark.soc("950")
def test_simt_kernel_grid_3d():
    torch.npu.set_device(ST_DEVICE)
    out = torch.full((6, 24), -1, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_grid_3d[(2, 3, 4), 1](out)
    torch.npu.synchronize()

    expected = torch.tensor(
        [[2, 3, 4, x, y, z] for z in range(4) for y in range(3) for x in range(2)],
        dtype=torch.int32,
    ).T.contiguous()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
