# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""每个线程写入共享UB，块内同步后读取另一线程写入的数据。"""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def exchange_after_syncthreads(out, shared):
    tid = pl.simt.linear_thread_idx()
    shared[0, tid] = tid
    pl.simt.syncthreads()
    out[0, tid] = shared[0, THREADS - 1 - tid]


@pl.jit(auto_mutex=True)
def simt_syncthreads(out: pl.Tensor[[1, THREADS], pl.DT_UINT32]):
    shared_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, THREADS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    shared = shared_group.current()
    with pl.section_vector():
        exchange_after_syncthreads[THREADS](out, shared)


@pytest.mark.soc("950")
def test_syncthreads():
    torch.npu.set_device(ST_DEVICE)
    out = torch.empty((1, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    simt_syncthreads(out)
    torch.npu.synchronize()

    expected = torch.arange(THREADS - 1, -1, -1, dtype=torch.int64).to(torch.uint32).reshape(1, THREADS)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
