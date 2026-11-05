# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 end-to-end tests for the SIMT atomic_inc interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
ELEMENTS = 64

@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_inc_ub(uint32_tile):
    pl.simt.atomic_inc(uint32_tile, [0, 0], 255)

@pl.jit(auto_mutex=True)
def simt_atomic_inc_ub(uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32]):
    uint32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    uint32_tile = uint32_tile_group.current()
    with pl.section_vector():
        pl.load(uint32_tile, uint32_state, [0, 0])
        atomic_inc_ub[THREADS](uint32_tile)
        pl.store(uint32_state, uint32_tile, [0, 0])

@pytest.mark.soc("950")
def test_atomic_inc_ub_uint32():
    torch.npu.set_device(ST_DEVICE)
    state = torch.full((1, ELEMENTS), 0, dtype=torch.uint32).to(ST_DEVICE)
    simt_atomic_inc_ub[None, 1](state)
    torch.npu.synchronize()
    assert state.cpu()[0, 0].item() == THREADS

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=THREADS)
def atomic_inc_gm_all_dtypes(
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    pl.simt.atomic_inc(uint32_state, [0, 0], 255)
    pl.simt.atomic_inc(uint64_state, [0, 0], 255)

@pl.jit()
def simt_atomic_inc_gm_all_dtypes(
    uint32_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT32],
    uint64_state: pl.Tensor[[1, ELEMENTS], pl.DT_UINT64],
):
    with pl.section_vector():
        atomic_inc_gm_all_dtypes[THREADS](uint32_state, uint64_state)

@pytest.mark.soc("950")
def test_atomic_inc_gm_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    states = [
        torch.full((1, ELEMENTS), 0, dtype=dtype).to(ST_DEVICE)
        for dtype in (torch.uint32, torch.uint64)
    ]
    simt_atomic_inc_gm_all_dtypes[None, 1](*states)
    torch.npu.synchronize()
    for state in states:
        assert state.cpu()[0, 0].item() == THREADS

# -----------------------------------------------------------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=1)
def atomic_inc_return_value_gm(
    state: pl.Tensor[[1, 1], pl.DT_UINT32],
    old_values: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    old_values[0, 0] = pl.simt.atomic_inc(state, [0, 0], 5)

@pl.jit()
def simt_atomic_inc_return_value_gm(
    state: pl.Tensor[[1, 1], pl.DT_UINT32],
    old_values: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    with pl.section_vector():
        atomic_inc_return_value_gm[1](state, old_values)

@pytest.mark.soc("950")
def test_atomic_inc_returns_old_value():
    torch.npu.set_device(ST_DEVICE)
    state = torch.tensor([[5]], dtype=torch.uint32).to(ST_DEVICE)
    old_values = torch.tensor([[0]], dtype=torch.uint32).to(ST_DEVICE)
    simt_atomic_inc_return_value_gm[None, 1](state, old_values)
    torch.npu.synchronize()
    assert state.cpu().item() == 0
    assert old_values.cpu().item() == 5
