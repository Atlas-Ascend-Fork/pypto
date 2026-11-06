# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 system tests for the SIMT rcp interface with Tile operands."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def rcp_tile(src_fp16_tile, src_bf16_tile, out_fp16_tile, out_bf16_tile):
    tid = pl.simt.linear_thread_idx()
    out_fp16_tile[0, tid] = pl.simt.rcp(src_fp16_tile[0, tid])
    out_bf16_tile[0, tid] = pl.simt.rcp(src_bf16_tile[0, tid])

@pl.jit(auto_mutex=True)
def simt_rcp_tile(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
):
    src_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_fp16_tile = src_fp16_tile_group.current()
    src_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_bf16_tile = src_bf16_tile_group.current()
    out_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_fp16_tile = out_fp16_tile_group.current()
    out_bf16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0C00,
        mutex_ids="auto",
        depth=1,
    )
    out_bf16_tile = out_bf16_tile_group.current()
    with pl.section_vector():
        pl.load(src_fp16_tile, src_fp16, [0, 0])
        pl.load(src_bf16_tile, src_bf16, [0, 0])
        rcp_tile[ELEMENTS](src_fp16_tile, src_bf16_tile, out_fp16_tile, out_bf16_tile)
        pl.store(out_fp16, out_fp16_tile, [0, 0])
        pl.store(out_bf16, out_bf16_tile, [0, 0])

@pytest.mark.soc("950")
def test_rcp_dtypes_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.125, 32.0, ELEMENTS).reshape(1, ELEMENTS)
    values[0, :8] = torch.tensor([-0.0, 0.0, float("-inf"), float("inf"), -2.0, 2.0, float("nan"), 0.5])
    sources = (values.to(torch.float16), values.to(torch.bfloat16))
    outputs = tuple(torch.empty_like(source, device=ST_DEVICE) for source in sources)
    simt_rcp_tile(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        atol = 5e-3 if source.dtype == torch.float16 else 2e-2
        expected = torch.reciprocal(source.float()).to(source.dtype)
        actual = output.cpu()
        torch.testing.assert_close(
            actual[:, 8:], expected[:, 8:], rtol=2e-4, atol=atol,
        )
        rtol, atol = (5e-3, 5e-3) if source.dtype == torch.float16 else (2e-2, 2e-2)
        torch.testing.assert_close(actual[:, :8], expected[:, :8], rtol=rtol, atol=atol, equal_nan=True)
        assert torch.equal(torch.signbit(actual[0, :4]), torch.signbit(expected[0, :4]))
