# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64
ACTIVE_THREADS = 33

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def classify(
    src_fp16,
    src_fp32,
    finite_fp16,
    finite_fp32,
):
    tid = pl.simt.linear_thread_idx()
    finite_fp16[0, tid] = pl.simt.isfinite(src_fp16[0, tid])
    finite_fp32[0, tid] = pl.simt.isfinite(src_fp32[0, tid])


@pl.jit(auto_mutex=True)
def simt_isfinite(
    src_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_FP16],
    src_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    finite_fp16: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
    finite_fp32: pl.Tensor[[1, ELEMENTS], pl.DT_BOOL],
):
    src_fp16_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_fp16_tile = src_fp16_tile_group.current()
    src_fp32_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_fp32_tile = src_fp32_tile_group.current()
    with pl.section_vector():
        pl.load(src_fp16_tile, src_fp16, [0, 0])
        pl.load(src_fp32_tile, src_fp32, [0, 0])
        classify[ACTIVE_THREADS](
            src_fp16_tile,
            src_fp32_tile,
            finite_fp16,
            finite_fp32,
        )


@pytest.mark.soc("950")
def test_isfinite_fp16_fp32():
    torch.npu.set_device(ST_DEVICE)
    values = [0.0, -0.0, 1.0, -1.0, 7.5, float("inf"), float("-inf"), float("nan")]
    repeat = ELEMENTS // len(values)
    source_fp16 = torch.tensor(values, dtype=torch.float16).repeat(repeat).reshape(1, ELEMENTS)
    source_fp32 = torch.tensor(values, dtype=torch.float32).repeat(repeat).reshape(1, ELEMENTS)
    finite_fp16 = torch.zeros((1, ELEMENTS), dtype=torch.bool, device=ST_DEVICE)
    finite_fp32 = torch.zeros((1, ELEMENTS), dtype=torch.bool, device=ST_DEVICE)
    simt_isfinite(source_fp16.to(ST_DEVICE), source_fp32.to(ST_DEVICE), finite_fp16, finite_fp32)
    torch.npu.synchronize()

    for source, output in ((source_fp16, finite_fp16), (source_fp32, finite_fp32)):
        expected = torch.zeros((1, ELEMENTS), dtype=torch.bool)
        expected[:, :ACTIVE_THREADS] = torch.isfinite(source[:, :ACTIVE_THREADS])
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
