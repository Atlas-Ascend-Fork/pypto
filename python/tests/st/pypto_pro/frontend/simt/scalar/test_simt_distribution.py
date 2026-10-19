# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar normal-distribution interfaces."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ------------------------------------------------------ normcdf ------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def normcdf_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.normcdf(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_normcdf_tile(
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
        normcdf_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_normcdf_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-6.0, 6.0, ELEMENTS)
    values[:7] = torch.tensor([float("-inf"), -10.0, -0.0, 0.0, 10.0, float("inf"), float("nan")])

    def golden(x):
        return 0.5 * torch.erfc(-x / math.sqrt(2.0))

    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_normcdf_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), golden(source), rtol=0.0005, atol=5e-06, equal_nan=True)

# ---------------------------------------------------- normcdfinv -----------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def normcdfinv_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.normcdfinv(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_normcdfinv_tile(
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
        normcdfinv_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_normcdfinv_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.01, 0.99, ELEMENTS)
    values[:7] = torch.tensor([0.0, 1.0e-7, 0.5, 1.0 - 1.0e-7, 1.0, float("nan"), 0.01])

    def golden(x):
        return math.sqrt(2.0) * torch.erfinv(2.0 * x - 1.0)

    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_normcdfinv_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    actual = output.cpu()
    expected = golden(source)
    torch.testing.assert_close(actual[:, [1, 3]], expected[:, [1, 3]], rtol=0.005, atol=5e-06)
    torch.testing.assert_close(actual[:, 6:], expected[:, 6:], rtol=0.0005, atol=5e-06)
    torch.testing.assert_close(actual[:, [0, 2, 4, 5]], expected[:, [0, 2, 4, 5]], rtol=0.0, atol=0.0, equal_nan=True)
