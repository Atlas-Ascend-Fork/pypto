# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 end-to-end tests for the SIMT atomic_or interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
ELEMENTS = 64

@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_or_ub_all_dtypes(
    int32_tile,
    uint32_tile,
):
    pl.simt.atomic_or(int32_tile, [0, 0], 0xF)
    pl.simt.atomic_or(uint32_tile, [0, 0], 0xF)

@pl.jit(auto_mutex=True)
def simt_atomic_or_ub_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
):
    int32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    int32_tile = int32_tile_group.current()
    uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    uint32_tile = uint32_tile_group.current()
    with pl.section_vector():
        pl.load(int32_tile, int32_state, [0, 0])
        pl.load(uint32_tile, uint32_state, [0, 0])
        atomic_or_ub_all_dtypes[THREADS](int32_tile, uint32_tile)
        pl.store(int32_state, int32_tile, [0, 0])
        pl.store(uint32_state, uint32_tile, [0, 0])

@pytest.mark.soc("950")
def test_atomic_or_ub_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    states = [
        torch.full((1, ELEMENTS), 0, dtype=dtype).to(ST_DEVICE)
        for dtype in (torch.int32, torch.uint32)
    ]
    simt_atomic_or_ub_all_dtypes(*states)
    torch.npu.synchronize()
    for state in states:
        assert state.cpu()[0, 0].item() == 0xF

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_or_gm_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    int64_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    pl.simt.atomic_or(int32_state, [0, 0], 0xF)
    pl.simt.atomic_or(uint32_state, [0, 0], 0xF)
    pl.simt.atomic_or(int64_state, [0, 0], 0xF)
    pl.simt.atomic_or(uint64_state, [0, 0], 0xF)

@pl.jit()
def simt_atomic_or_gm_all_dtypes(
    int32_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    int64_state: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    with pl.section_vector():
        atomic_or_gm_all_dtypes[THREADS](int32_state, uint32_state, int64_state, uint64_state)

@pytest.mark.soc("950")
def test_atomic_or_gm_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    states = [
        torch.full((1, ELEMENTS), 0, dtype=dtype).to(ST_DEVICE)
        for dtype in (torch.int32, torch.uint32, torch.int64, torch.uint64)
    ]
    simt_atomic_or_gm_all_dtypes(*states)
    torch.npu.synchronize()
    for state in states:
        assert state.cpu()[0, 0].item() == 0xF

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=1)
def atomic_or_return_value_gm(
    state: pl.Tensor[[1, 1], pl.DT_INT32],
    old_values: pl.Tensor[[1, 1], pl.DT_INT32],
):
    old_values[0, 0] = pl.simt.atomic_or(state, [0, 0], 0x0F)

@pl.jit()
def simt_atomic_or_return_value_gm(
    state: pl.Tensor[[1, 1], pl.DT_INT32],
    old_values: pl.Tensor[[1, 1], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_or_return_value_gm[1](state, old_values)

@pytest.mark.soc("950")
def test_atomic_or_returns_old_value():
    torch.npu.set_device(ST_DEVICE)
    state = torch.tensor([[0]], dtype=torch.int32).to(ST_DEVICE)
    old_values = torch.zeros_like(state)
    simt_atomic_or_return_value_gm[None, 1](state, old_values)
    torch.npu.synchronize()
    assert state.cpu().item() == 0x0F
    assert old_values.cpu().item() == 0
