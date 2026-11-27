# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A5 numerical coverage for a scalar loop whose accumulated result feeds a SIMD tile operation."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


@pl.vector_function(mode="simd_tile")
def add_accumulated_bias(dst, src, scale):
    total = pl.const(0, pl.DT_INT32)
    for iteration in pl.range(4):
        delta = scale * pl.const(2, pl.DT_INT32)
        total = total + delta
    pl.add(dst, src, total)


def _make_kernel():
    @pl.jit(auto_mutex=True)
    def accumulated_bias_kernel(
        x: pl.Tensor[[4, 64], pl.DT_FP32],
        y: pl.Tensor[[4, 64], pl.DT_FP32],
        scale: pl.DT_INT32,
    ):
        with pl.section_vector():
            tile_type = pl.TileType(
                shape=[4, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[4, 64]
            )
            src_group = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids="auto", depth=1)
            dst_group = pl.make_tile_group(type=tile_type, addrs=[1024], mutex_ids="auto", depth=1)
            src = src_group.current()
            dst = dst_group.current()
            pl.load(src, x, [0, 0])
            add_accumulated_bias(dst, src, scale)
            pl.store(y, dst, [0, 0])

    return accumulated_bias_kernel


@pytest.mark.soc("950")
def test_simd_tile_scalar_loop(pgo_vf_available):
    if not pgo_vf_available:
        pytest.skip("SIMD tile execution requires the PGO VF plugin")

    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.linspace(-2, 3, 4 * 64).reshape(4, 64)
    y = torch.full_like(x, float("nan"), device="npu")
    kernel = _make_kernel()
    kernel[1](x.npu(), y, 2)
    torch.testing.assert_close(y.cpu(), x + 16, rtol=1e-3, atol=1e-4)
