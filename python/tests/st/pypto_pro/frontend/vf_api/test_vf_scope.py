# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device coverage for scalar operations permitted inside a VF loop."""

import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch


@pl.vector_function
def _add_one_vf(src, dst, count):
    for offset in pl.range(0, count, 64):
        active = pl.max(0, pl.min(count - offset, 64))
        mask = vf.update_mask(active, dtype=pl.DT_FP32)
        reg = vf.load_align(src, offset)
        reg = vf.adds(reg, pl.const(1.0, pl.DT_FP32), mask)
        vf.store_align(dst, reg, mask, offset)


@pl.jit(auto_mutex=True)
def _scalar_vf_kernel(
    x: pl.Tensor[[1, 128], pl.DT_FP32],
    y: pl.Tensor[[1, 128], pl.DT_FP32],
    count: pl.DT_INT64,
):
    tile_type = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    inputs = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
    outputs = pl.make_tile_group(type=tile_type, addrs=[512], mutex_ids=[1])
    with pl.section_vector():
        src = inputs.next()
        dst = outputs.next()
        pl.load(src, x, [0, 0])
        pl.load(dst, y, [0, 0])
        _add_one_vf(src, dst, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950", "960")
def test_scalar_operations_in_vf():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    x = torch.arange(128, dtype=torch.float32).reshape(1, 128)
    x_npu = x.to(device)
    for count in (0, 17, 79, 128):
        y = torch.full_like(x, -10.0, device=device)
        _scalar_vf_kernel[None, 1](x_npu, y, count)
        torch.npu.synchronize()
        expected = torch.full_like(x, -10.0)
        expected[:, :count] = x[:, :count] + 1.0
        actual = y.cpu()
        max_abs_error = (actual - expected).abs().max().item()
        print(f"VF scalar bounds: count={count}, max_abs_error={max_abs_error}")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
