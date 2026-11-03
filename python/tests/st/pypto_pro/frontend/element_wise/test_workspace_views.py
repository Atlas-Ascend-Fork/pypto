# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Workspace pointer views with explicit strides and untouched padding checks."""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"



def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


# ===========================================================================
# addptr.md / make_tensor.md / Ptr.md —— 用 Ptr 接收 workspace 裸指针，
#   addptr 切出 workspace，make_tensor 用非连续行 stride 包装成 tensor view，
#   分两次完成 a*2 写回 out，并覆盖非零 offset 的 load/store。
#   vector kernel 开 `auto_mutex`，同步由 `make_tile_group` 自动管理。
# ===========================================================================
@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [256, 1])

    tt = pl.TileType(shape=[32, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])

        pl.load(t, a, [32, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [32, 0])
        pl.load(t, ws_buf, [32, 0])
        pl.store(out, t, [32, 0])


@pytest.mark.soc("950")
def test_workspace_kernel():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    shape = [64, 128]
    a = torch.rand(shape, device=device, dtype=torch.float16)
    out = torch.empty(shape, device=device, dtype=torch.float16)
    workspace = torch.full((64 * 128 + 64 * 256,), -7.0, device=device, dtype=torch.float16)

    workspace_kernel(a, workspace, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a * 2, rtol=1e-2, atol=1e-2)
    pitched_workspace = workspace[64 * 128:].view(64, 256)
    torch.testing.assert_close(pitched_workspace[:, 128:], torch.full_like(pitched_workspace[:, 128:], -7.0))
    logging.info("addptr/make_tensor workspace result equal!")
