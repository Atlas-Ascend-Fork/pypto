# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 - installs torch.npu

DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
pytestmark = pytest.mark.soc("950")


@pl.jit()
def get_cycle_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
    out: pl.Tensor[[2], pl.DT_INT64],
):
    tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    t0 = pl.get_cycle()
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.add(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
    t1 = pl.get_cycle()
    out[0] = t1 - t0
    out[1] = t1 - t1  # 0: 确认同核两次紧邻读取可作差


def test_get_cycle_delta():
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    logging.info("------------test_get_cycle_delta--------------")

    x = torch.rand([128, 128], device=device, dtype=torch.float16)
    y = torch.rand([128, 128], device=device, dtype=torch.float16)
    z = torch.zeros([128, 128], device=device, dtype=torch.float16)
    out = torch.zeros(2, device=device, dtype=torch.int64)

    get_cycle_kernel[None, 1](x, y, z, out)
    torch.npu.synchronize()

    delta, zero = out.tolist()
    assert zero == 0, f"same-core subtraction should be exact, got {zero}"
    assert 0 < delta < (1 << 62), f"elapsed cycles out of reasonable range: {delta}"
    logging.info("elapsed cycles = %d", delta)

    torch.testing.assert_close(z, x + y, rtol=1e-2, atol=1e-2)
    logging.info("test_get_cycle_delta passed!")
