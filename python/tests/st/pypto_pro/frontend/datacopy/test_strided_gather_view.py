# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for gathering non-contiguous GM lines on A5 CCE.

``pl.make_tensor(source, shape, stride)`` builds a tensor view with its own stride, so a single
``pl.load`` can gather rows that are far apart in GM. The stride is supplied at runtime (here
read from an input tensor), and because the view is a tensor in its own right it gets its own
GlobalTensor declaration -- its stride applies to that view alone and to every access through
it, offsets included.

This replaces the former ``pl.set_stride``, which rewrote one tensor's descriptor in place: it
left the offset arithmetic on the *original* stride, leaked its effect to every later access of
that tensor, and wrote to the tensor's plain declaration even when the access needed a
different layout variant. Positioning a view by pointer (``pl.addptr``) expresses the same
gather without any of that.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


LINE = 128  # elements per line (fp16)
N_LINES = 512  # total lines in the source GM tensor
N_PAIRS = 32  # number of pl.load ops, each loading a [2, LINE] tile
OUT_LINES = N_PAIRS * 2  # 64 gathered lines


def _is_a5() -> bool:
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        logging.info("NPU unavailable: %s", exc)
        return False
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        logging.info("Current device is %s, not A5 (Ascend950). Skip.", name)
        return False
    return True


# ---------------------------------------------------------------------------
# Test 1: basic two-line gather using a single pl.load through a strided view.
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def strided_view_basic_kernel(
    x: pl.Tensor[[N_LINES, LINE], pl.DT_FP16],
    strides: pl.Tensor[[1, 1], pl.DT_INT32],
    out: pl.Tensor[[2, LINE], pl.DT_FP16],
):
    ub_type = pl.TileType(shape=[2, LINE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ub_db = pl.make_tile_group(type=ub_type, addrs=0x0000, mutex_ids=[0, 1])

    with pl.section_vector():
        s = strides[0, 0]  # element (row) stride, read from GM input
        tile = ub_db.next()
        view = pl.make_tensor(x, [N_LINES, LINE], [s, 1])  # same data, custom row stride
        pl.load(tile, view, [0, 0])  # row0 = line 0, row1 = line 0 + s/LINE
        pl.store(out, tile, [0, 0])


@pytest.mark.soc("950")
def test_strided_view_basic():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(0)

    x = torch.randn(N_LINES, LINE, device=device, dtype=torch.float16)
    stride_lines = 137
    strides = torch.tensor([[stride_lines * LINE]], device=device, dtype=torch.int32)
    out = torch.zeros(2, LINE, device=device, dtype=torch.float16)

    strided_view_basic_kernel(x, strides, out)
    torch.npu.synchronize()

    expected = torch.stack([x[0], x[stride_lines]], dim=0)

    logging.info("basic strided view: line strides = %d", stride_lines)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    logging.info("test_strided_view_basic passed.")


# ---------------------------------------------------------------------------
# Test 2: gather 64 separate lines from a [512, 128] GM tensor into UB using
# 32 pl.load operations. Each load fetches a [2, 128] tile whose two lines are
# spaced by a random, positive stride read from an input.
#
# The view is positioned by pointer rather than by a load offset: its stride is
# the *pair* spacing, so an offset through the view would step by s, not by one
# line. pl.addptr moves the base to line i, and the view then supplies the
# spacing between the pair's two rows.
# ---------------------------------------------------------------------------
@pl.jit(auto_mutex=True)
def gather_view_kernel(
    x: pl.Tensor[[N_LINES, LINE], pl.DT_FP16],
    strides: pl.Tensor[[1, N_PAIRS], pl.DT_INT32],
    out: pl.Tensor[[OUT_LINES, LINE], pl.DT_FP16],
):
    ub_type = pl.TileType(shape=[2, LINE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ub_db = pl.make_tile_group(type=ub_type, addrs=0x0000, mutex_ids=[0, 1])

    with pl.section_vector():
        for i in pl.range(0, N_PAIRS, 1):
            s = strides[0, i]  # per-pair element (row) stride from GM input
            tile = ub_db.next()
            view = pl.make_tensor(pl.addptr(pl.make_ptr(x), i * LINE), [2, LINE], [s, 1])
            pl.load(tile, view, [0, 0])  # row0 = line i, row1 = line i + s/LINE
            pl.store(out, tile, [2 * i, 0])


@pytest.mark.soc("950")
def test_gather_64_lines_with_strided_views():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(1)

    x = torch.randn(N_LINES, LINE, device=device, dtype=torch.float16)

    # Random positive line strides; keep i + stride_line < N_LINES for every pair.
    stride_lines = torch.randint(1, 400, (N_PAIRS,), dtype=torch.int64)
    for i in range(N_PAIRS):
        if i + int(stride_lines[i]) >= N_LINES:
            stride_lines[i] = N_LINES - 1 - i
    stride_elems = (stride_lines * LINE).to(torch.int32).reshape(1, N_PAIRS).to(device)

    out = torch.zeros(OUT_LINES, LINE, device=device, dtype=torch.float16)

    gather_view_kernel(x, stride_elems, out)
    torch.npu.synchronize()

    expected = torch.zeros(OUT_LINES, LINE, device=device, dtype=torch.float16)
    for i in range(N_PAIRS):
        k = int(stride_lines[i])
        expected[2 * i] = x[i]
        expected[2 * i + 1] = x[i + k]

    logging.info("gather line strides: %s", stride_lines.tolist())
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    logging.info("test_gather_64_lines_with_strided_views passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_strided_view_basic()
    test_gather_64_lines_with_strided_views()
    logging.info("\nAll strided-view tests passed!")
