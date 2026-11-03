# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check the sequence produced by fill_index on device."""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

START = 0


# ============================================================================
# fill_index
# ============================================================================


@pl.jit(auto_mutex=True)
def fill_index_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        m_dim = out.shape[0]
        n_dim = out.shape[1]
        for i in pl.range(0, m_dim, 1):
            for j in pl.range(0, n_dim, 64):
                cur_out = tile_out.current()
                pl.fill_index(cur_out, START)
                pl.store(out, cur_out, [i, j])


# ============================================================================
# Test functions
# ============================================================================


@pytest.mark.soc("950")
def test_fill_index():
    torch.npu.set_device(ST_DEVICE)
    m_val, n_val = 1, 64
    out = torch.empty(m_val, n_val, device=ST_DEVICE, dtype=torch.int32)
    out_ref = torch.arange(START, START + n_val, dtype=torch.int32, device=ST_DEVICE).unsqueeze(0).contiguous()
    fill_index_kernel(out)
    torch.npu.synchronize()
    assert torch.equal(out, out_ref), f"fill_index mismatch:\n{out}\nvs\n{out_ref}"
    logging.info("test_fill_index passed!")
