# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""每个block先写GM，再发布完成计数，最后完成的block读取全部数据。"""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
GRID_BLOCKS = 4


@pl.vector_function(mode="simt", max_threads=1)
def publish_with_threadfence(
    out: pl.Tensor[[1, 1], pl.DT_INT32],
    values: pl.Tensor[[1, GRID_BLOCKS], pl.DT_INT32],
    completed: pl.Tensor[[1, 1], pl.DT_INT32],
):
    block_id = pl.simt.block_idx().x
    values[0, block_id] = 1
    pl.simt.threadfence()
    ticket = pl.simt.atomic_add(completed, [0, 0], 1)
    if ticket == GRID_BLOCKS - 1:
        total: pl.DT_INT32 = 0
        for index in pl.range(GRID_BLOCKS):
            total = total + values[0, index]
        out[0, 0] = total


@pl.jit()
def simt_threadfence(
    out: pl.Tensor[[1, 1], pl.DT_INT32],
    values: pl.Tensor[[1, GRID_BLOCKS], pl.DT_INT32],
    completed: pl.Tensor[[1, 1], pl.DT_INT32],
):
    with pl.section_vector():
        publish_with_threadfence[1](out, values, completed)


@pytest.mark.soc("950")
def test_threadfence():
    torch.npu.set_device(ST_DEVICE)
    out = torch.full((1, 1), -1, dtype=torch.int32, device=ST_DEVICE)
    values = torch.zeros((1, GRID_BLOCKS), dtype=torch.int32, device=ST_DEVICE)
    completed = torch.zeros((1, 1), dtype=torch.int32, device=ST_DEVICE)
    simt_threadfence[None, GRID_BLOCKS](out, values, completed)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([[GRID_BLOCKS]], dtype=torch.int32), rtol=0, atol=0)
