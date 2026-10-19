# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for the SIMT fma_relu interface with Tile operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def fma_relu_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16_tile, out_bf16_tile):
    tid = pl.simt.linear_thread_idx()
    out_fp16_tile[0, tid] = pl.simt.fma_relu(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid], lhs_fp16_tile[0, tid])
    out_bf16_tile[0, tid] = pl.simt.fma_relu(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid], lhs_bf16_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_fma_relu_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    lhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp16_tile = lhs_fp16_tile_group.current()
    rhs_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp16_tile = rhs_fp16_tile_group.current()
    lhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_bf16_tile = lhs_bf16_tile_group.current()
    rhs_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_bf16_tile = rhs_bf16_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        fma_relu_tile[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            out_fp16_tile,
            out_bf16_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])


@pytest.mark.soc("950")
def test_fma_relu_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.linspace(-4.0, 4.0, ELEMENTS)
    rhs = torch.linspace(3.0, -3.0, ELEMENTS)

    def golden(x, y):
        return torch.clamp(x * y + x, min=0.0)

    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(torch.empty_like(lhs_source, device=ST_DEVICE) for lhs_source, _ in operand_pairs)
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_fma_relu_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = golden(lhs_source.float(), rhs_source.float()).to(lhs_source.dtype)
        rtol, atol = {torch.float16: (0.005, 0.005), torch.bfloat16: (0.02, 0.02)}[lhs_source.dtype]
        torch.testing.assert_close(output.cpu(), expected, rtol=rtol, atol=atol, equal_nan=True)
