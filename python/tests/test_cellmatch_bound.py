#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
ST for cellMatchTable fill/read range when tail-iteration validShape < declared shape.

Port of framework/tests/st/machine/src/ops/test_dev_cellmatch_bound.cpp
(test_cellmatch_tail_valid_shape):

  q [b=2, sq=-1(18), d=8] --L1--> addTmp[b, sq, d]
                           --Reshape--> qReshape[b*sq, d] = [36, d]
                           --L2--> out[36, d], offSet=32
  L2 last tile: declared [32, d], validShape [4, d].

Do NOT add ``from __future__ import annotations`` — it breaks @jit parameter parsing.
"""

import os

from numpy.testing import assert_allclose
import torch
import torch_npu

import pypto
from pypto import pypto_impl

B = 2
D = 8
SQ_RUNTIME = 18
OFFSET = 32
OP_COUNT = 2
RTOL = 1e-3
ATOL = 1e-3


@pypto.frontend.jit()
def k_cellmatch_tail_valid_shape(
    q: pypto.Tensor([B, pypto.DYNAMIC, D], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, D], pypto.DT_FP32),
):
    b, sq, _ = q.shape
    add_tmp = pypto.tensor([b, sq, D], pypto.DT_FP32, "addTmp")

    pypto.set_vec_tile_shapes(1, 4, 32)
    for sq_idx in pypto.loop(sq, name="L1", idx_name="sqIdx"):
        view_tmp = pypto.view(q, [B, 1, D], [0, sq_idx, 0])
        pypto.assemble(view_tmp + 0.01, [0, sq_idx, 0], add_tmp)

    q_reshape = pypto.reshape(add_tmp, [b * sq, D], inplace=True)

    n = b * sq
    for loop_idx in pypto.loop(pypto.ceildiv(n, OFFSET), name="L2", idx_name="loopIdx"):
        pypto.set_vec_tile_shapes(4, 32)
        valid0 = pypto.min(n - loop_idx * OFFSET, OFFSET)
        tmp0 = pypto.view(
            q_reshape,
            [OFFSET, D],
            [loop_idx * OFFSET, 0],
            valid_shape=[valid0, D],
        )
        pypto.assemble(tmp0 + 0.01, [loop_idx * OFFSET, 0], out)


def test_cellmatch_tail_valid_shape():
    """Tail tile validShape < declared shape: cellMatch fill/read must use validShape."""
    pypto_impl.SetPassConfig("PVC2_OOO", "SplitReshape", "disable_pass", True)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    elem_cnt = B * SQ_RUNTIME * D
    q_cpu = torch.arange(elem_cnt, dtype=torch.float32).reshape(B, SQ_RUNTIME, D)
    golden = (torch.arange(elem_cnt, dtype=torch.float32) + 0.01 * OP_COUNT).reshape(B * SQ_RUNTIME, D)

    q_npu = q_cpu.to(device)
    out_npu = torch.full((B * SQ_RUNTIME, D), 0.001, dtype=torch.float32, device=device)

    k_cellmatch_tail_valid_shape(q_npu, out_npu)
    torch_npu.npu.synchronize()

    assert_allclose(
        out_npu.cpu().numpy(),
        golden.numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="cellmatch_tail_valid_shape mismatch",
    )


if __name__ == "__main__":
    test_cellmatch_tail_valid_shape()
