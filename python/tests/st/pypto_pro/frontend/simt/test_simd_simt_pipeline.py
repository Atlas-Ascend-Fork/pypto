# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""3510 system test for a SIMD/SIMT pipeline with multiple launches and nested SIMT callees."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 256


@pl.vector_function(mode="simt")
def affine(value: pl.DT_FP32, scale: pl.DT_FP32, delta: pl.DT_FP32) -> pl.DT_FP32:
    return value * scale + delta


@pl.vector_function(mode="simt")
def store_value(
    dst,
    index: pl.DT_UINT32,
    value: pl.DT_FP32,
):
    dst[0, index] = value


@pl.vector_function(mode="simt")
def transform_one(
    dst,
    src,
    index: pl.DT_UINT32,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
):
    value = affine(src[0, index], scale, delta)
    store_value(dst, index, value)


@pl.vector_function(mode="simt", max_threads=THREADS)
def transform_tile(
    dst,
    src,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    transform_one(dst, src, tid, scale, delta)


@pl.vector_function(mode="simt", max_threads=THREADS)
def mul_inplace(data, scale: pl.DT_FP32):
    tid = pl.simt.linear_thread_idx()
    data[0, tid] = data[0, tid] * scale


@pl.jit(auto_mutex=True)
def simd_simt_pipeline(
    x: pl.Tensor[[1, THREADS], pl.DT_FP32],
    out: pl.Tensor[[1, THREADS], pl.DT_FP32],
    pre_scale: pl.DT_FP32,
    scale: pl.DT_FP32,
    delta: pl.DT_FP32,
    simt_scale: pl.DT_FP32,
    post_scale: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, THREADS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    data_group = pl.make_tile_group(
        type=tile_type,
        addrs=0,
        mutex_ids="auto",
        depth=1,
    )
    data = data_group.current()
    scaled_group = pl.make_tile_group(
        type=tile_type,
        addrs=1024,
        mutex_ids="auto",
        depth=1,
    )
    scaled = scaled_group.current()
    result_group = pl.make_tile_group(
        type=tile_type,
        addrs=2048,
        mutex_ids="auto",
        depth=1,
    )
    result = result_group.current()
    with pl.section_vector():
        pl.load(data, x, [0, 0])
        pl.muls(scaled, data, pre_scale)
        transform_tile[THREADS](data, scaled, scale, delta)
        mul_inplace[THREADS](data, simt_scale)
        pl.muls(result, data, post_scale)
        pl.store(out, result, [0, 0])


@pytest.mark.soc("950")
def test_simd_simt_pipeline():
    torch.npu.set_device(ST_DEVICE)

    pre_scale = 2.0
    scale = 1.5
    delta = -2.0
    simt_scale = 2.5
    post_scale = 0.5
    x = torch.arange(THREADS, dtype=torch.float32).reshape(1, THREADS).to(ST_DEVICE)
    out = torch.empty_like(x)

    simd_simt_pipeline(x, out, pre_scale, scale, delta, simt_scale, post_scale)
    torch.npu.synchronize()

    expected = ((x.cpu() * pre_scale * scale + delta) * simt_scale) * post_scale
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_simd_simt_pipeline()
