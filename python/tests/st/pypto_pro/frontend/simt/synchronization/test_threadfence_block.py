# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""每个线程先写共享UB，再发布完成计数，最后完成的线程读取全部数据。"""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def publish_with_threadfence_block(
    out: pl.Tensor[[1, 1], pl.DT_INT32],
    values,
    completed,
):
    tid = pl.simt.linear_thread_idx()
    if tid == 0:
        completed[0, 0] = 0
    values[0, tid] = 0
    pl.simt.syncthreads()

    values[0, tid] = 1
    pl.simt.threadfence_block()
    ticket = pl.simt.atomic_add(completed, [0, 0], 1)
    if ticket == THREADS - 1:
        total: pl.DT_INT32 = 0
        for index in pl.range(THREADS):
            total = total + values[0, index]
        out[0, 0] = total


@pl.jit(auto_mutex=True)
def simt_threadfence_block(out: pl.Tensor[[1, 1], pl.DT_INT32]):
    values_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, THREADS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    values = values_group.current()
    completed_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=THREADS * 4,
        mutex_ids="auto",
        depth=1,
    )
    completed = completed_group.current()
    with pl.section_vector():
        publish_with_threadfence_block[THREADS](out, values, completed)


@pytest.mark.soc("950")
def test_threadfence_block():
    torch.npu.set_device(ST_DEVICE)
    out = torch.full((1, 1), -1, dtype=torch.int32, device=ST_DEVICE)
    simt_threadfence_block(out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([[THREADS]], dtype=torch.int32), rtol=0, atol=0)
