# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""System tests for member synchronization across cooperative-group types."""

import os

import pypto_pro.language as pl
import pytest
import torch

pytestmark = pytest.mark.soc("950")
ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
THREADS = 64


# SIMT function
@pl.vector_function(mode="simt", max_threads=THREADS)
def _sync_probe(scratch, out: pl.Tensor[[3, THREADS], pl.DT_UINT32]):
    block = pl.simt.this_thread_block()
    tid = block.thread_rank()

    scratch[0, tid] = pl.simt.cast(tid, pl.DT_UINT32)
    block.sync()
    out[0, tid] = scratch[0, tid ^ 32]
    block.sync()

    tile = pl.simt.tiled_partition(block, 32)
    scratch[0, tid] = pl.simt.cast(1000 + tid, pl.DT_UINT32)
    tile.sync()
    out[1, tid] = scratch[0, tid ^ 31]
    block.sync()

    if tid % 2 == 0:
        active = pl.simt.coalesced_threads()
        active.sync()
        out[2, tid] = tid


# JIT kernel
@pl.jit(arch="3510")
def _sync_kernel(out: pl.Tensor[[3, THREADS], pl.DT_UINT32]):
    scratch = pl.make_tile(
        pl.TileType(shape=[1, THREADS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0,
    )
    with pl.section_vector():
        _sync_probe[THREADS](scratch, out)


# Host launch and validation
def test_group_sync_methods_coordinate_shared_storage():
    rows = [
        [index ^ 32 for index in range(THREADS)],
        [1000 + (index ^ 31) for index in range(THREADS)],
        [index if index % 2 == 0 else 0xFFFFFFFF for index in range(THREADS)],
    ]
    out = torch.full((3, THREADS), -1, dtype=torch.int32).to(torch.uint32).to(ST_DEVICE)
    _sync_kernel[None, 1](out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)
