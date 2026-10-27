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

"""ST: pl.dynamic_input_dim on Ascend A5.

A runtime-variadic (DYNAMIC) input descriptor in the ``dim != 0`` form carries one shape
value per tensor axis.  The ST synthesizes such a descriptor (dim == 1 / 2 / 3) and checks
that a @pl.jit kernel reads every valid axis of a listed tensor via pl.dynamic_input_dim.

Note: ``dim == 0`` (the pointer-only form) has no shape block and is out of scope; an axis
outside ``[0, dim)`` is guarded on device (traps) rather than reading a wrong word.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

MAX_AXIS = 4  # upper bound on the axes this ST exercises (dim <= MAX_AXIS)


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _build_list_tensor_desc_shaped(addrs, dim, shapes):
    """list-tensor descriptor with dim >= 1.

    Layout: [dataPtrOffset, <count> shape structs of (1+dim) words each, addr_0..].
    Tensor i's shape[axis] is word[1 + i*(1+dim) + 1 + axis].
    """
    n = len(addrs)
    struct_sz = 1 + dim
    data_ptr_offset = 8 + n * struct_sz * 8
    w = [0] * (1 + n * struct_sz + n)
    w[0] = data_ptr_offset
    w[1] = (dim & 0xFFFFFFFF) | (n << 32)  # low32 dim, high32 count
    for i in range(n):
        for k in range(dim):
            w[1 + i * struct_sz + 1 + k] = shapes[k]
    addr_off = data_ptr_offset // 8
    for i, a in enumerate(addrs):
        w[addr_off + i] = a
    return torch.tensor(w, dtype=torch.int64)


@pl.jit(auto_mutex=True)
def dynamic_input_dim_read(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_INT64],
    dim: pl.DT_INT32,
):
    # Read every valid axis of tensor 1 (axis in [0, dim)); dim comes from the host.
    with pl.section_vector():
        out_t = pl.make_tensor(out, [1, MAX_AXIS], [MAX_AXIS, 1], dtype=pl.DT_INT64)
        for axis in pl.range(0, dim):
            pl.setval(out_t, axis, pl.dynamic_input_dim(desc, 1, axis))


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_input_dim():
    _require_a5()

    torch.manual_seed(0)
    # Generic over dim: the descriptor's shape stride is 1 + dim at runtime.
    for dim, shapes in ((1, [11]), (2, [22, 23]), (3, [33, 34, 35])):
        tensors = [torch.zeros(16, device=ST_DEVICE, dtype=torch.float32) for _ in range(2)]
        desc = _build_list_tensor_desc_shaped([t.data_ptr() for t in tensors], dim, shapes).to(ST_DEVICE)
        out = torch.full((1, MAX_AXIS), -1, device=ST_DEVICE, dtype=torch.int64)
        dynamic_input_dim_read[None, 1](desc, out, dim)
        torch.npu.synchronize()
        got = [int(v) for v in out.cpu().flatten().tolist()][:dim]
        torch.testing.assert_close(got, shapes, rtol=0, atol=0)
        logging.info("  dim=%d shapes=%s OK", dim, shapes)

    logging.info("test_dynamic_input_dim passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_dynamic_input_dim()
    logging.info("\nAll tests passed!")
