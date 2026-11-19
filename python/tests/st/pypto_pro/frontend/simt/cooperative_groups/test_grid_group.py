# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GridGroup queries and repeated barriers in standalone SIMT kernels."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

pytestmark = pytest.mark.soc("950")
ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
THREADS = 32
BLOCKS = 4  # Keep the barrier grid below the device's physical AIV core count.
TOTAL = THREADS * BLOCKS
PHASES = 4


# SIMT function
@pl.vector_function(mode="simt")
def _grid_barrier():
    grid = pl.simt.this_grid()
    grid.sync()


# JIT kernels
@pl.jit(mode="simt", max_threads=64)
def _grid_queries(out: pl.Tensor[[11, TOTAL], pl.DT_UINT64]):
    grid = pl.simt.this_grid()
    rank = grid.thread_rank()
    dims = grid.group_dim()
    index = grid.block_index()
    out[0, rank] = rank
    out[1, rank] = grid.block_rank()
    out[2, rank] = grid.size()
    out[3, rank] = grid.num_blocks()
    out[4, rank] = grid.is_valid()
    out[5, rank] = pl.simt.cast(dims.x, pl.DT_UINT64)
    out[6, rank] = pl.simt.cast(dims.y, pl.DT_UINT64)
    out[7, rank] = pl.simt.cast(dims.z, pl.DT_UINT64)
    out[8, rank] = pl.simt.cast(index.x, pl.DT_UINT64)
    out[9, rank] = pl.simt.cast(index.y, pl.DT_UINT64)
    out[10, rank] = pl.simt.cast(index.z, pl.DT_UINT64)


@pl.jit(mode="simt", max_threads=THREADS)
def _grid_sync(counter: pl.Tensor[[1], pl.DT_INT32], out: pl.Tensor[[PHASES, TOTAL], pl.DT_INT32]):
    grid = pl.simt.this_grid()
    rank = grid.thread_rank()
    for phase in pl.range(0, PHASES, 1):
        pl.simt.atomic_add(counter, [0], 1)
        grid.sync()
        # Atomic reads do not rely on cross-block data-cache coherence.
        out[phase, rank] = pl.simt.atomic_add(counter, [0], 0)
        _grid_barrier()


# Host launch and validation
@pytest.mark.parametrize(
    "grid,block",
    [
        ((1, 1, 1), (1, 1, 1)),
        ((2, 1, 1), (31, 1, 1)),
        ((1, 2, 1), (33, 1, 1)),
        ((4, 1, 1), (32, 1, 1)),
        ((2, 1, 2), (4, 4, 2)),
        ((1, 2, 2), (2, 4, 2)),
        ((2, 2, 2), (2, 2, 2)),
    ],
)
def test_grid_group_queries(grid, block):
    torch.npu.set_device(ST_DEVICE)
    # Keep unused columns initialized to detect writes beyond the launch geometry.
    out = torch.full((11, TOTAL), -1, dtype=torch.int64).to(torch.uint64).to(ST_DEVICE)
    _grid_queries[grid, block](out)
    torch.npu.synchronize()
    total = math.prod(grid) * math.prod(block)
    rows = []
    for rank in range(total):
        block_rank = rank // math.prod(block)
        block_index = (
            block_rank % grid[0],
            block_rank // grid[0] % grid[1],
            block_rank // (grid[0] * grid[1]),
        )
        rows.append([rank, block_rank, total, math.prod(grid), 1, *grid, *block_index])
    expected = torch.full((11, TOTAL), -1, dtype=torch.int64)
    expected[:, :total] = torch.tensor(rows, dtype=torch.int64).T
    torch.testing.assert_close(out.cpu().to(torch.int64), expected, rtol=0, atol=0)


def test_grid_group_repeated_sync_and_independent_launches():
    torch.npu.set_device(ST_DEVICE)
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    outputs = []
    for stream in streams:
        with torch.npu.stream(stream):
            counter = torch.zeros((1,), dtype=torch.int32, device=ST_DEVICE)
            out = torch.empty((PHASES, TOTAL), dtype=torch.int32, device=ST_DEVICE)
            # Reuse the compiled kernel; each launch has its own grid barrier state.
            for _ in range(3):
                counter.zero_()
                _grid_sync[stream, (2, 2, 1), THREADS](counter, out)
            outputs.append((counter, out))
    for stream in streams:
        stream.synchronize()
    expected = torch.tensor([[(phase + 1) * TOTAL] * TOTAL for phase in range(PHASES)], dtype=torch.int32)
    for counter, out in outputs:
        assert counter.cpu().item() == PHASES * TOTAL
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
