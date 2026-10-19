# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar scaling interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# ------------------------------------------------------- ldexp -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def ldexp_tile(src_tile, exponent_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.ldexp(src_tile[0, tid], exponent_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_ldexp_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    exponent: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    exponent_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    exponent_tile = exponent_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        pl.load(exponent_tile, exponent, [0, 0])
        ldexp_tile[ELEMENTS](src_tile, exponent_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_ldexp_fp32_int32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-4.0, 4.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    exponent = torch.tensor([index % 21 - 10 for index in range(ELEMENTS)], dtype=torch.int32).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            torch.finfo(torch.float32).tiny,
        ]
    )
    exponent[0, :16] = torch.tensor(
        [
            torch.iinfo(torch.int32).min,
            -300,
            -278,
            -149,
            -127,
            -101,
            0,
            101,
            127,
            149,
            278,
            300,
            torch.iinfo(torch.int32).max,
            torch.iinfo(torch.int32).min,
            300,
            -24,
        ],
        dtype=torch.int32,
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_ldexp_tile(source.to(ST_DEVICE), exponent.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.ldexp(source, exponent)
    expected[0, 10:15] = source[0, 10:15]
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-05, atol=1e-06, equal_nan=True)
    assert torch.equal(torch.signbit(actual[0, 10:14]), torch.signbit(expected[0, 10:14]))

# ------------------------------------------------------ scalbn -------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def scalbn_tile(src_tile, exponent_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.scalbn(src_tile[0, tid], exponent_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_scalbn_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    exponent: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    exponent_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    exponent_tile = exponent_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        pl.load(exponent_tile, exponent, [0, 0])
        scalbn_tile[ELEMENTS](src_tile, exponent_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_scalbn_fp32_int32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-4.0, 4.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    exponent = torch.tensor([index % 21 - 10 for index in range(ELEMENTS)], dtype=torch.int32).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            torch.finfo(torch.float32).tiny,
        ]
    )
    exponent[0, :16] = torch.tensor(
        [
            torch.iinfo(torch.int32).min,
            -300,
            -278,
            -149,
            -127,
            -101,
            0,
            101,
            127,
            149,
            278,
            300,
            torch.iinfo(torch.int32).max,
            torch.iinfo(torch.int32).min,
            300,
            -24,
        ],
        dtype=torch.int32,
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_scalbn_tile(source.to(ST_DEVICE), exponent.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.ldexp(source, exponent.clamp(-278, 278))
    expected[0, 10:15] = source[0, 10:15]
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-05, atol=1e-06, equal_nan=True)
    assert torch.equal(torch.signbit(actual[0, 10:14]), torch.signbit(expected[0, 10:14]))

# --------------------------------------------------- scalbn_int64 ----------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def scalbn_int64_tile(src_tile, exponent_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.scalbn(src_tile[0, tid], exponent_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_scalbn_int64_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    exponent: pl.Tensor[[1, ELEMENTS], pl.DT_INT64],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    exponent_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    exponent_tile = exponent_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        pl.load(exponent_tile, exponent, [0, 0])
        scalbn_int64_tile[ELEMENTS](src_tile, exponent_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_scalbn_fp32_int64_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(-4.0, 4.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    exponent = torch.tensor([index % 21 - 10 for index in range(ELEMENTS)], dtype=torch.int64).reshape(1, ELEMENTS)
    source[0, :16] = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            torch.finfo(torch.float32).tiny,
        ]
    )
    exponent[0, :16] = torch.tensor(
        [
            torch.iinfo(torch.int64).min,
            -1000000000000,
            -278,
            -149,
            -127,
            -101,
            0,
            101,
            127,
            149,
            278,
            1000000000000,
            torch.iinfo(torch.int64).max,
            torch.iinfo(torch.int64).min,
            1000000000000,
            -24,
        ],
        dtype=torch.int64,
    )
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_scalbn_int64_tile(source.to(ST_DEVICE), exponent.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.ldexp(source, exponent.clamp(-278, 278).to(torch.int32))
    expected[0, 10:15] = source[0, 10:15]
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-05, atol=1e-06, equal_nan=True)
    assert torch.equal(torch.signbit(actual[0, 10:14]), torch.signbit(expected[0, 10:14]))
