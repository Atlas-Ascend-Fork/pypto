# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 generalized system test for the SIMT isnan interface."""

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def isnan_all_dtypes(
    src_fp16,
    src_bf16,
    src_fp32,
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
):
    tid = pl.simt.linear_thread_idx()
    out_fp16[0, tid] = pl.simt.isnan(src_fp16[0, tid])
    out_bf16[0, tid] = pl.simt.isnan(src_bf16[0, tid])
    out_fp32[0, tid] = pl.simt.isnan(src_fp32[0, tid])


@pl.jit(auto_mutex=True)
def simt_isnan_all_dtypes(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BF16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_bf16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    out_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
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
    src_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    src_fp32_tile = src_fp32_tile_group.current()
    with pl.section_vector():
        pl.load(src_fp16_tile, src_fp16, [0, 0])
        pl.load(src_bf16_tile, src_bf16, [0, 0])
        pl.load(src_fp32_tile, src_fp32, [0, 0])
        isnan_all_dtypes[ELEMENTS](
            src_fp16_tile,
            src_bf16_tile,
            src_fp32_tile,
            out_fp16,
            out_bf16,
            out_fp32,
        )


@pytest.mark.soc("950")
def test_isnan_all_supported_dtypes():
    torch.npu.set_device(ST_DEVICE)
    values = torch.zeros(ELEMENTS)
    values[0] = float("nan")
    values[1] = float("inf")
    values[2] = float("-inf")
    values[3] = 1.0
    fp32 = values.to(torch.float32).reshape(1, -1)
    sources = (fp32.half(), fp32.bfloat16(), fp32)
    outputs = tuple(torch.empty(source.shape, dtype=torch.bool, device=ST_DEVICE) for source in sources)
    simt_isnan_all_dtypes(*(source.to(ST_DEVICE) for source in sources), *outputs)
    torch.npu.synchronize()
    for source, output in zip(sources, outputs):
        torch.testing.assert_close(output.cpu(), torch.isnan(source), rtol=0, atol=0)
