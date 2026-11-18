# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Minimal SIMT kernel compilation smoke test and call-site syntax coverage."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.jit(mode="simt", max_threads=32)
def simt_kernel_int32_write_one(out: pl.Tensor[[1], pl.DT_INT32]):
    out[0] = 1


@pytest.mark.soc("950")
def test_simt_kernel_int32_write_one():
    torch.npu.set_device(ST_DEVICE)
    out = torch.zeros(1, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_int32_write_one[1, 1](out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.ones(1, dtype=torch.int32), rtol=0, atol=0)


THREADS = 64


@pl.vector_function(mode="simt")
def _scale_store(dst, src, index: pl.DT_UINT32, scale: pl.DT_FP32):
    dst[0, index] = src[0, index] * scale


@pl.vector_function(mode="simt", max_threads=THREADS)
def _scale_store_entry(dst_positional, dst_keyword, src, scale: pl.DT_FP32):
    tid = pl.simt.linear_thread_idx()
    _scale_store(dst_positional, src, tid, scale)
    _scale_store(scale=scale, src=src, index=tid, dst=dst_keyword)


@pl.jit(auto_mutex=True)
def simt_kernel_keyword_call(
    x: pl.Tensor[[1, THREADS], pl.DT_FP32],
    positional: pl.Tensor[[1, THREADS], pl.DT_FP32],
    keyword: pl.Tensor[[1, THREADS], pl.DT_FP32],
    scale: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, THREADS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    src_group = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids="auto", depth=1)
    positional_group = pl.make_tile_group(type=tile_type, addrs=1024, mutex_ids="auto", depth=1)
    keyword_group = pl.make_tile_group(type=tile_type, addrs=2048, mutex_ids="auto", depth=1)
    with pl.section_vector():
        src = src_group.current()
        positional_tile = positional_group.current()
        keyword_tile = keyword_group.current()
        pl.load(src, x, [0, 0])
        _scale_store_entry[THREADS](
            dst_keyword=keyword_tile,
            scale=scale,
            src=src,
            dst_positional=positional_tile,
        )
        pl.store(positional, positional_tile, [0, 0])
        pl.store(keyword, keyword_tile, [0, 0])


@pytest.mark.soc("950")
def test_simt_kernel_keyword_arguments_and_inferred_dtypes():
    torch.npu.set_device(ST_DEVICE)
    scale = 2.5
    x = torch.arange(THREADS, dtype=torch.float32).reshape(1, THREADS).to(ST_DEVICE)
    positional = torch.full_like(x, -1.0)
    keyword = torch.full_like(x, -1.0)
    simt_kernel_keyword_call(x, positional, keyword, scale)
    torch.npu.synchronize()

    expected = x.cpu() * scale
    torch.testing.assert_close(positional.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(keyword.cpu(), expected, rtol=0, atol=0)
