# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 system test for multicore SIMT execution and three-dimensional thread context."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
GRID_BLOCKS = 4
THREADS_X = 4
THREADS_Y = 2
THREADS_Z = 8
THREADS = THREADS_X * THREADS_Y * THREADS_Z
CONTEXT_ROWS = 14


@pl.vector_function(mode="simt", max_threads=THREADS)
def write_thread_context(dst):
    thread = pl.simt.thread_idx()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    grid = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()
    warp = pl.simt.cast(pl.simt.warp_size(), pl.DT_UINT32)
    dst[0, tid] = tid
    dst[1, tid] = thread.x
    dst[2, tid] = thread.y
    dst[3, tid] = thread.z
    dst[4, tid] = block.x
    dst[5, tid] = block.y
    dst[6, tid] = block.z
    dst[7, tid] = block_id.x
    dst[8, tid] = block_id.y
    dst[9, tid] = block_id.z
    dst[10, tid] = grid.x
    dst[11, tid] = grid.y
    dst[12, tid] = grid.z
    dst[13, tid] = warp


@pl.jit(auto_mutex=True)
def simt_multicore(out: pl.Tensor[[GRID_BLOCKS * CONTEXT_ROWS, THREADS], pl.DT_UINT32]):
    tile_type = pl.TileType(shape=[CONTEXT_ROWS, THREADS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    dst_group = pl.make_tile_group(
        type=tile_type,
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    dst = dst_group.current()
    with pl.section_vector():
        core_id = pl.get_block_idx()
        write_thread_context[THREADS_X, THREADS_Y, THREADS_Z](dst)
        pl.store(out, dst, [core_id * CONTEXT_ROWS, 0])


@pytest.mark.soc("950")
def test_multicore():
    torch.npu.set_device(ST_DEVICE)

    out = torch.full((GRID_BLOCKS * CONTEXT_ROWS, THREADS), -1, dtype=torch.int32).to(torch.uint32).to(ST_DEVICE)
    simt_multicore[None, GRID_BLOCKS](out)
    torch.npu.synchronize()

    tid = torch.arange(THREADS, dtype=torch.int64)
    expected = torch.zeros((GRID_BLOCKS, CONTEXT_ROWS, THREADS), dtype=torch.int64)
    expected[:, 0, :] = tid
    expected[:, 1, :] = tid % THREADS_X
    expected[:, 2, :] = (tid // THREADS_X) % THREADS_Y
    expected[:, 3, :] = tid // (THREADS_X * THREADS_Y)
    expected[:, 4, :] = THREADS_X
    expected[:, 5, :] = THREADS_Y
    expected[:, 6, :] = THREADS_Z
    expected[:, 7, :] = torch.arange(GRID_BLOCKS, dtype=torch.int64).reshape(-1, 1)
    expected[:, 10, :] = GRID_BLOCKS
    expected[:, 11:13, :] = 1
    expected[:, 13, :] = 32
    actual = out.cpu().to(torch.int64).reshape(GRID_BLOCKS, CONTEXT_ROWS, THREADS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_multicore()
