# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 generalized system test for the SIMT fma interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 105
TILE_ELEMENTS = 112

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def fma_all_dtypes(
    lhs_fp16,
    rhs_fp16,
    addend_fp16,
    lhs_bf16,
    rhs_bf16,
    addend_bf16,
    lhs_fp32,
    rhs_fp32,
    addend_fp32,
    out_fp16,
    out_bf16,
    out_fp32,
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.fma(lhs_fp16[0, tid], rhs_fp16[0, tid], addend_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.fma(lhs_bf16[0, tid], rhs_bf16[0, tid], addend_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.fma(lhs_fp32[0, tid], rhs_fp32[0, tid], addend_fp32[0, tid])


@pl.jit(auto_mutex=True)
def simt_fma_all_dtypes(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    addend_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    addend_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    lhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    rhs_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    addend_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    fp16_type = pl.TileType(
        shape=[1, TILE_ELEMENTS], dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec, valid_shape=[1, ELEMENTS],
    )
    bf16_type = pl.TileType(
        shape=[1, TILE_ELEMENTS], dtype=pl.DT_BF16,
        target_memory=pl.MemorySpace.Vec, valid_shape=[1, ELEMENTS],
    )
    fp32_type = pl.TileType(
        shape=[1, TILE_ELEMENTS], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec, valid_shape=[1, ELEMENTS],
    )
    lhs_fp16_tile_group = pl.make_tile_group(
        type=fp16_type,
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp16_tile = lhs_fp16_tile_group.current()
    rhs_fp16_tile_group = pl.make_tile_group(
        type=fp16_type,
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp16_tile = rhs_fp16_tile_group.current()
    addend_fp16_tile_group = pl.make_tile_group(
        type=fp16_type,
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    addend_fp16_tile = addend_fp16_tile_group.current()
    lhs_bf16_tile_group = pl.make_tile_group(
        type=bf16_type,
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    lhs_bf16_tile = lhs_bf16_tile_group.current()
    rhs_bf16_tile_group = pl.make_tile_group(
        type=bf16_type,
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    rhs_bf16_tile = rhs_bf16_tile_group.current()
    addend_bf16_tile_group = pl.make_tile_group(
        type=bf16_type,
        addrs=0x1400,
        mutex_ids="auto",
        depth=1,
    )
    addend_bf16_tile = addend_bf16_tile_group.current()
    lhs_fp32_tile_group = pl.make_tile_group(
        type=fp32_type,
        addrs=0x1800,
        mutex_ids="auto",
        depth=1,
    )
    lhs_fp32_tile = lhs_fp32_tile_group.current()
    rhs_fp32_tile_group = pl.make_tile_group(
        type=fp32_type,
        addrs=0x1C00,
        mutex_ids="auto",
        depth=1,
    )
    rhs_fp32_tile = rhs_fp32_tile_group.current()
    addend_fp32_tile_group = pl.make_tile_group(
        type=fp32_type,
        addrs=0x2000,
        mutex_ids="auto",
        depth=1,
    )
    addend_fp32_tile = addend_fp32_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=fp16_type,
        addrs=0x2400,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=bf16_type,
        addrs=0x2800,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    out_fp32_tile_group = pl.make_tile_group(
        type=fp32_type,
        addrs=0x2C00,
        mutex_ids="auto",
        depth=1,
    )
    out_fp32_tile = out_fp32_tile_group.current()
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(addend_fp16_tile, addend_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        pl.load(addend_bf16_tile, addend_bf16, [0, 0])
        pl.load(lhs_fp32_tile, lhs_fp32, [0, 0])
        pl.load(rhs_fp32_tile, rhs_fp32, [0, 0])
        pl.load(addend_fp32_tile, addend_fp32, [0, 0])
        fma_all_dtypes[7, 5, 3](
            lhs_fp16_tile,
            rhs_fp16_tile,
            addend_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            addend_bf16_tile,
            lhs_fp32_tile,
            rhs_fp32_tile,
            addend_fp32_tile,
            out_fp16_tile,
            out_bf16_tile,
            out_fp32_tile,
        )
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])
        pl.store(out_fp32, out_fp32_tile, [0, 0])


@pytest.mark.soc("950")
def test_fma_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.linspace(-2.0, 2.0, ELEMENTS)
    rhs = torch.linspace(0.5, 1.5, ELEMENTS)
    addend = torch.linspace(-0.75, 0.75, ELEMENTS)
    lhs[:4] = torch.tensor([1.0 + 2.0**-23, -(1.0 + 2.0**-23), 4097.0, -4097.0])
    rhs[:4] = torch.tensor([1.0 + 2.0**-23, 1.0 + 2.0**-23, 4097.0, 4097.0])
    addend[:4] = torch.tensor([-(1.0 + 2.0**-22), 1.0 + 2.0**-22, -16785408.0, 16785408.0])
    fp32_operands = tuple(values.to(torch.float32).reshape(1, -1) for values in (lhs, rhs, addend))
    operand_triples = tuple(
        tuple(operand.to(dtype) for operand in fp32_operands)
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    outputs = tuple(torch.empty_like(lhs_source, device=ST_DEVICE) for lhs_source, _, _ in operand_triples)
    arguments = [operand.to(ST_DEVICE) for triple in operand_triples for operand in triple]
    simt_fma_all_dtypes(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source, addend_source), output in zip(operand_triples, outputs):
        expected = (lhs_source.double() * rhs_source.double() + addend_source.double()).float().to(lhs_source.dtype)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
