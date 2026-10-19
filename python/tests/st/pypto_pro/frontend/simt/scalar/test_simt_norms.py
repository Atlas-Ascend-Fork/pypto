# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar norm interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ------------------------------------------------------- hypot -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def hypot_tile(lhs_tile, rhs_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.hypot(lhs_tile[0, tid], rhs_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_hypot_tile(
    lhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    lhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_tile = lhs_tile_group.current()
    rhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_tile = rhs_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_tile, lhs, [0, 0])
        pl.load(rhs_tile, rhs, [0, 0])
        hypot_tile[ELEMENTS](lhs_tile, rhs_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_hypot_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.linspace(-32.0, 32.0, ELEMENTS)
    rhs = torch.linspace(64.0, -64.0, ELEMENTS)
    lhs[:8] = torch.tensor(
        [
            0.0,
            -0.0,
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).max,
            float("inf"),
            float("nan"),
            3.0,
            5.0,
        ]
    )
    rhs[:8] = torch.tensor(
        [
            -0.0,
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).tiny,
            1.0,
            float("nan"),
            float("inf"),
            4.0,
            12.0,
        ]
    )
    lhs_source = lhs.to(torch.float32).reshape(1, -1)
    rhs_source = rhs.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(lhs_source, device=ST_DEVICE)
    simt_hypot_tile(lhs_source.to(ST_DEVICE), rhs_source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.hypot(lhs_source, rhs_source), rtol=1e-05, atol=1e-06, equal_nan=True
    )

# ------------------------------------------------------ rhypot -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def rhypot_tile(lhs_tile, rhs_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.rhypot(lhs_tile[0, tid], rhs_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_rhypot_tile(
    lhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    lhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_tile = lhs_tile_group.current()
    rhs_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_tile = rhs_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_tile, lhs, [0, 0])
        pl.load(rhs_tile, rhs, [0, 0])
        rhypot_tile[ELEMENTS](lhs_tile, rhs_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_rhypot_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.linspace(-32.0, 32.0, ELEMENTS)
    rhs = torch.linspace(64.0, -64.0, ELEMENTS)
    lhs[:8] = torch.tensor(
        [
            0.0,
            -0.0,
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).max,
            float("inf"),
            float("nan"),
            3.0,
            5.0,
        ]
    )
    rhs[:8] = torch.tensor(
        [
            -0.0,
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).tiny,
            1.0,
            float("nan"),
            float("inf"),
            4.0,
            12.0,
        ]
    )
    lhs_source = lhs.to(torch.float32).reshape(1, -1)
    rhs_source = rhs.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(lhs_source, device=ST_DEVICE)
    simt_rhypot_tile(lhs_source.to(ST_DEVICE), rhs_source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(),
        (lambda x, y: torch.reciprocal(torch.hypot(x, y)))(lhs_source, rhs_source),
        rtol=1e-05,
        atol=1e-06,
        equal_nan=True,
    )

# ------------------------------------------------------ norm3d -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def norm3d_tile(x_tile, y_tile, z_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.norm3d(x_tile[0, tid], y_tile[0, tid], z_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_norm3d_tile(
    x: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    y: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    z: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    x_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    x_tile = x_tile_group.current()
    y_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    y_tile = y_tile_group.current()
    z_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    z_tile = z_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(x_tile, x, [0, 0])
        pl.load(y_tile, y, [0, 0])
        pl.load(z_tile, z, [0, 0])
        norm3d_tile[ELEMENTS](x_tile, y_tile, z_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_norm3d_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    x = torch.linspace(-8.0, 8.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    y = torch.linspace(0.25, 16.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    z = torch.linspace(-32.0, -0.5, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    x[0, :6] = torch.tensor(
        [0.0, torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max, float("inf"), float("nan"), 3.0]
    )
    y[0, :6] = torch.tensor([-0.0, torch.finfo(torch.float32).tiny, 1.0, float("nan"), float("inf"), 4.0])
    z[0, :6] = torch.tensor([0.0, -torch.finfo(torch.float32).tiny, -1.0, 2.0, 2.0, 12.0])
    output = torch.empty_like(x, device=ST_DEVICE)
    simt_norm3d_tile(x.to(ST_DEVICE), y.to(ST_DEVICE), z.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.hypot(torch.hypot(x, y), z), rtol=1e-05, atol=1e-06, equal_nan=True)

# ------------------------------------------------------ norm4d -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def norm4d_tile(x_tile, y_tile, z_tile, w_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.norm4d(x_tile[0, tid], y_tile[0, tid], z_tile[0, tid], w_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_norm4d_tile(
    x: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    y: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    z: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    w: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    x_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    x_tile = x_tile_group.current()
    y_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    y_tile = y_tile_group.current()
    z_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    z_tile = z_tile_group.current()
    w_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    w_tile = w_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(x_tile, x, [0, 0])
        pl.load(y_tile, y, [0, 0])
        pl.load(z_tile, z, [0, 0])
        pl.load(w_tile, w, [0, 0])
        norm4d_tile[ELEMENTS](
            x_tile,
            y_tile,
            z_tile,
            w_tile,
            out_tile,
        )
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_norm4d_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    x = torch.linspace(-8.0, 8.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    y = torch.linspace(0.25, 16.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    z = torch.linspace(-32.0, -0.5, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    w = torch.linspace(64.0, 1.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    x[0, :6] = torch.tensor(
        [0.0, torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max, float("inf"), float("nan"), 1.0]
    )
    y[0, :6] = torch.tensor([-0.0, torch.finfo(torch.float32).tiny, 1.0, float("nan"), float("inf"), 2.0])
    z[0, :6] = torch.tensor([0.0, -torch.finfo(torch.float32).tiny, -1.0, 2.0, 2.0, 2.0])
    w[0, :6] = torch.tensor([-0.0, torch.finfo(torch.float32).tiny, 2.0, 3.0, 3.0, 4.0])
    output = torch.empty_like(x, device=ST_DEVICE)
    simt_norm4d_tile(x.to(ST_DEVICE), y.to(ST_DEVICE), z.to(ST_DEVICE), w.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(),
        torch.hypot(torch.hypot(x, y), torch.hypot(z, w)),
        rtol=1e-05,
        atol=1e-06,
        equal_nan=True,
    )

# ------------------------------------------------------ rnorm3d ------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def rnorm3d_tile(x_tile, y_tile, z_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.rnorm3d(x_tile[0, tid], y_tile[0, tid], z_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_rnorm3d_tile(
    x: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    y: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    z: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    x_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    x_tile = x_tile_group.current()
    y_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    y_tile = y_tile_group.current()
    z_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    z_tile = z_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(x_tile, x, [0, 0])
        pl.load(y_tile, y, [0, 0])
        pl.load(z_tile, z, [0, 0])
        rnorm3d_tile[ELEMENTS](x_tile, y_tile, z_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_rnorm3d_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    x = torch.linspace(-8.0, 8.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    y = torch.linspace(0.25, 16.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    z = torch.linspace(-32.0, -0.5, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    x[0, :6] = torch.tensor(
        [0.0, torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max, float("inf"), float("nan"), 3.0]
    )
    y[0, :6] = torch.tensor([-0.0, torch.finfo(torch.float32).tiny, 1.0, float("nan"), float("inf"), 4.0])
    z[0, :6] = torch.tensor([0.0, -torch.finfo(torch.float32).tiny, -1.0, 2.0, 2.0, 12.0])
    output = torch.empty_like(x, device=ST_DEVICE)
    simt_rnorm3d_tile(x.to(ST_DEVICE), y.to(ST_DEVICE), z.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.reciprocal(torch.hypot(torch.hypot(x, y), z))
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-05, atol=1e-06, equal_nan=True)

# ------------------------------------------------------ rnorm4d ------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def rnorm4d_tile(x_tile, y_tile, z_tile, w_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.rnorm4d(x_tile[0, tid], y_tile[0, tid], z_tile[0, tid], w_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_rnorm4d_tile(
    x: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    y: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    z: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    w: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    x_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    x_tile = x_tile_group.current()
    y_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    y_tile = y_tile_group.current()
    z_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    z_tile = z_tile_group.current()
    w_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    w_tile = w_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(x_tile, x, [0, 0])
        pl.load(y_tile, y, [0, 0])
        pl.load(z_tile, z, [0, 0])
        pl.load(w_tile, w, [0, 0])
        rnorm4d_tile[ELEMENTS](
            x_tile,
            y_tile,
            z_tile,
            w_tile,
            out_tile,
        )
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_rnorm4d_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    x = torch.linspace(-8.0, 8.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    y = torch.linspace(0.25, 16.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    z = torch.linspace(-32.0, -0.5, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    w = torch.linspace(64.0, 1.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    x[0, :6] = torch.tensor(
        [0.0, torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max, float("inf"), float("nan"), 1.0]
    )
    y[0, :6] = torch.tensor([-0.0, torch.finfo(torch.float32).tiny, 1.0, float("nan"), float("inf"), 2.0])
    z[0, :6] = torch.tensor([0.0, -torch.finfo(torch.float32).tiny, -1.0, 2.0, 2.0, 2.0])
    w[0, :6] = torch.tensor([-0.0, torch.finfo(torch.float32).tiny, 2.0, 3.0, 3.0, 4.0])
    output = torch.empty_like(x, device=ST_DEVICE)
    simt_rnorm4d_tile(x.to(ST_DEVICE), y.to(ST_DEVICE), z.to(ST_DEVICE), w.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.reciprocal(torch.hypot(torch.hypot(x, y), torch.hypot(z, w)))
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-05, atol=1e-06, equal_nan=True)
