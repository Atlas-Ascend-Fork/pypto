# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar gamma interfaces."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ------------------------------------------------------ lgamma -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def lgamma_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.lgamma(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_lgamma_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        lgamma_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_lgamma_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.1, 32.0, ELEMENTS)
    values[:8] = torch.tensor([-4.5, -3.5, -2.5, -1.5, -0.5, 0.0, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_lgamma_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.lgamma(source), rtol=0.0005, atol=5e-06, equal_nan=True)

# ------------------------------------------------------ tgamma -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def tgamma_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.tgamma(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_tgamma_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        tgamma_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_tgamma_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.1, 8.0, ELEMENTS)
    values[:8] = torch.tensor([-4.5, -3.5, -2.5, -1.5, -0.5, 0.1, 1.0, 1.5])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_tgamma_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.tensor([math.gamma(float(value)) for value in source.reshape(-1)], dtype=torch.float32)
    torch.testing.assert_close(output.cpu(), expected.reshape_as(source), rtol=0.0005, atol=5e-06, equal_nan=True)
