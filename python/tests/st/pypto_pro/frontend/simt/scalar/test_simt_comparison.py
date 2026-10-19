# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar comparison interfaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# -------------------------------------------------------- equ --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def equ_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16, out_bf16):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.equ(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid])
    out_bf16[0, tid] = pl.simt.equ(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_equ_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
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
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        equ_tile[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            out_fp16,
            out_bf16,
        )


@pytest.mark.soc("950")
def test_equ_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * 8)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * 8)

    def golden(x, y):
        return torch.isnan(x) | torch.isnan(y) | (x == y)

    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(
        torch.empty(lhs_source.shape, dtype=torch.bool, device=ST_DEVICE) for lhs_source, _ in operand_pairs
    )
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_equ_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = golden(lhs_source, rhs_source)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

# -------------------------------------------------------- geu --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def geu_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16, out_bf16):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.geu(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid])
    out_bf16[0, tid] = pl.simt.geu(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_geu_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
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
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        geu_tile[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            out_fp16,
            out_bf16,
        )


@pytest.mark.soc("950")
def test_geu_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * 8)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * 8)

    def golden(x, y):
        return torch.isnan(x) | torch.isnan(y) | (x >= y)

    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(
        torch.empty(lhs_source.shape, dtype=torch.bool, device=ST_DEVICE) for lhs_source, _ in operand_pairs
    )
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_geu_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = golden(lhs_source, rhs_source)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

# -------------------------------------------------------- gtu --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def gtu_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16, out_bf16):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.gtu(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid])
    out_bf16[0, tid] = pl.simt.gtu(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_gtu_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
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
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        gtu_tile[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            out_fp16,
            out_bf16,
        )


@pytest.mark.soc("950")
def test_gtu_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * 8)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * 8)

    def golden(x, y):
        return torch.isnan(x) | torch.isnan(y) | (x > y)

    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(
        torch.empty(lhs_source.shape, dtype=torch.bool, device=ST_DEVICE) for lhs_source, _ in operand_pairs
    )
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_gtu_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = golden(lhs_source, rhs_source)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

# -------------------------------------------------------- leu --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def leu_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16, out_bf16):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.leu(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid])
    out_bf16[0, tid] = pl.simt.leu(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_leu_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
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
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        leu_tile[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            out_fp16,
            out_bf16,
        )


@pytest.mark.soc("950")
def test_leu_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * 8)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * 8)

    def golden(x, y):
        return torch.isnan(x) | torch.isnan(y) | (x <= y)

    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(
        torch.empty(lhs_source.shape, dtype=torch.bool, device=ST_DEVICE) for lhs_source, _ in operand_pairs
    )
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_leu_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = golden(lhs_source, rhs_source)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

# -------------------------------------------------------- ltu --------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def ltu_tile(lhs_fp16_tile, rhs_fp16_tile, lhs_bf16_tile, rhs_bf16_tile, out_fp16, out_bf16):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.ltu(lhs_fp16_tile[0, tid], rhs_fp16_tile[0, tid])
    out_bf16[0, tid] = pl.simt.ltu(lhs_bf16_tile[0, tid], rhs_bf16_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_ltu_tile(
    lhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    rhs_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    lhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    rhs_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
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
    with pl.section_vector():
        pl.load(lhs_fp16_tile, lhs_fp16, [0, 0])
        pl.load(rhs_fp16_tile, rhs_fp16, [0, 0])
        pl.load(lhs_bf16_tile, lhs_bf16, [0, 0])
        pl.load(rhs_bf16_tile, rhs_bf16, [0, 0])
        ltu_tile[ELEMENTS](
            lhs_fp16_tile,
            rhs_fp16_tile,
            lhs_bf16_tile,
            rhs_bf16_tile,
            out_fp16,
            out_bf16,
        )


@pytest.mark.soc("950")
def test_ltu_all_supported_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([1.0, -2.0, 3.5, -4.0, float("nan"), 6.0, -7.0, 8.0] * 8)
    rhs = torch.tensor([2.0, 3.0, -1.0, -5.0, 4.0, float("nan"), -7.0, 0.5] * 8)

    def golden(x, y):
        return torch.isnan(x) | torch.isnan(y) | (x < y)

    lhs_fp32 = lhs.to(torch.float32).reshape(1, -1)
    rhs_fp32 = rhs.to(torch.float32).reshape(1, -1)
    operand_pairs = tuple((lhs_fp32.to(dtype), rhs_fp32.to(dtype)) for dtype in (torch.float16, torch.bfloat16))
    outputs = tuple(
        torch.empty(lhs_source.shape, dtype=torch.bool, device=ST_DEVICE) for lhs_source, _ in operand_pairs
    )
    arguments = [operand.to(ST_DEVICE) for pair in operand_pairs for operand in pair]
    simt_ltu_tile(*arguments, *outputs)
    torch.npu.synchronize()
    for (lhs_source, rhs_source), output in zip(operand_pairs, outputs):
        expected = golden(lhs_source, rhs_source)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
