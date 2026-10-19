# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar error-function interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# -------------------------------------------------------- erf --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def erf_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.erf(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_erf_tile(
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
        erf_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_erf_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-4.0, 4.0, ELEMENTS)
    values[:7] = torch.tensor([float("-inf"), -4.0, -0.0, 0.0, 4.0, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_erf_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.erf(source), rtol=0.0005, atol=5e-06, equal_nan=True)

# ------------------------------------------------------- erfc --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def erfc_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.erfc(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_erfc_tile(
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
        erfc_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_erfc_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-4.0, 4.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -10.055, -0.0, 0.0, 10.055, 10.056, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_erfc_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.erfc(source), rtol=0.0005, atol=5e-06, equal_nan=True)

# ------------------------------------------------------ erfinv -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def erfinv_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.erfinv(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_erfinv_tile(
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
        erfinv_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_erfinv_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-0.99, 0.99, ELEMENTS)
    values[:7] = torch.tensor([-1.0, -0.999, -0.0, 0.0, 0.999, 1.0, float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_erfinv_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.erfinv(source), rtol=0.0005, atol=5e-06, equal_nan=True)

# ------------------------------------------------------ erfcinv ------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def erfcinv_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.erfcinv(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_erfcinv_tile(
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
        erfcinv_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_erfcinv_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.01, 1.99, ELEMENTS)
    values[:8] = torch.tensor([0.0, 0.0033, 0.0034, 1.0, 1.9966, 1.9967, 2.0, float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_erfcinv_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), (lambda x: torch.erfinv(1.0 - x))(source), rtol=0.0005, atol=5e-06, equal_nan=True
    )

# ------------------------------------------------------- erfcx -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def erfcx_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.erfcx(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_erfcx_tile(
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
        erfcx_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_erfcx_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-3.0, 8.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -9.0, -0.0, 0.0, 8.0, 10.0, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_erfcx_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.special.erfcx(source), rtol=0.0005, atol=5e-06, equal_nan=True)
