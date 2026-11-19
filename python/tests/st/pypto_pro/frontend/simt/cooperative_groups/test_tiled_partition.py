# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""System tests for the tiled_partition cooperative-group operation."""

from __future__ import annotations

import os

import pypto_pro.language as pl
import pytest
import torch

pytestmark = pytest.mark.soc("950")
ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
THREADS = 64


# SIMT functions
@pl.vector_function(mode="simt", max_threads=THREADS)
def _tiled_partition_probe(
    widths: pl.Tensor[[2], pl.DT_UINT32],
    out: pl.Tensor[[6, THREADS], pl.DT_UINT32],
):
    block = pl.simt.this_thread_block()
    parent = pl.simt.tiled_partition(block, widths[0])
    child = pl.simt.tiled_partition(parent, widths[1])
    tid = block.thread_rank()
    out[0, tid] = child.thread_rank()
    out[1, tid] = child.size()
    out[2, tid] = child.meta_group_rank()
    out[3, tid] = child.meta_group_size()
    out[4, tid] = parent.meta_group_rank()
    out[5, tid] = child.shfl(tid, child.size() - 1)


# JIT kernels
@pl.jit(arch="3510")
def _tiled_partition_kernel(
    widths: pl.Tensor[[2], pl.DT_UINT32],
    out: pl.Tensor[[6, THREADS], pl.DT_UINT32],
):
    with pl.section_vector():
        _tiled_partition_probe[THREADS](widths, out)


@pl.jit(mode="simt", max_threads=THREADS)
def _deep_tiled_partition_kernel(
    widths: pl.Tensor[[3], pl.DT_UINT32],
    out: pl.Tensor[[10, THREADS], pl.DT_UINT32],
):
    block = pl.simt.this_thread_block()
    parent = pl.simt.tiled_partition(block, widths[0])
    child = pl.simt.tiled_partition(parent, widths[1])
    leaf = pl.simt.tiled_partition(child, widths[2])
    tid = block.thread_rank()
    out[0, tid] = parent.meta_group_rank()
    out[1, tid] = parent.meta_group_size()
    out[2, tid] = child.meta_group_rank()
    out[3, tid] = child.meta_group_size()
    out[4, tid] = leaf.thread_rank()
    out[5, tid] = leaf.size()
    out[6, tid] = leaf.meta_group_rank()
    out[7, tid] = leaf.meta_group_size()
    out[8, tid] = leaf.shfl(tid, 0)
    out[9, tid] = leaf.shfl(tid, leaf.size() - 1)


# Host launch and validation
@pytest.mark.parametrize(
    "parent_width,width",
    [(parent, child) for parent in (2, 4, 8, 16, 32) for child in (1, 2, 4, 8, 16) if child < parent],
)
def test_nested_tiled_partition_with_runtime_widths(parent_width, width):
    runtime_widths = torch.tensor([parent_width, width], dtype=torch.uint32, device=ST_DEVICE)
    out = torch.empty((6, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    _tiled_partition_kernel[None, 1](runtime_widths, out)
    torch.npu.synchronize()
    rows = [
        [i % width for i in range(THREADS)],
        [width] * THREADS,
        [(i % parent_width) // width for i in range(THREADS)],
        [parent_width // width] * THREADS,
        [i // parent_width for i in range(THREADS)],
        [(i // width + 1) * width - 1 for i in range(THREADS)],
    ]
    torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)


@pytest.mark.parametrize("parent_width,child_width,width", [(4, 2, 1), (16, 8, 2), (32, 16, 8), (32, 4, 1)])
def test_three_level_tiled_partition_with_runtime_widths(parent_width, child_width, width):
    torch.npu.set_device(ST_DEVICE)
    runtime_widths = torch.tensor([parent_width, child_width, width], dtype=torch.uint32, device=ST_DEVICE)
    out = torch.empty((10, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    _deep_tiled_partition_kernel[1, THREADS](runtime_widths, out)
    torch.npu.synchronize()
    rows = [
        [i // parent_width for i in range(THREADS)],
        [THREADS // parent_width] * THREADS,
        [i % parent_width // child_width for i in range(THREADS)],
        [parent_width // child_width] * THREADS,
        [i % width for i in range(THREADS)],
        [width] * THREADS,
        [i % child_width // width for i in range(THREADS)],
        [child_width // width] * THREADS,
        [i // width * width for i in range(THREADS)],
        [(i // width + 1) * width - 1 for i in range(THREADS)],
    ]
    torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)
