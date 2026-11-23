#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test the write-after-read dependency of ``index_put_`` on NPU."""

import os

import torch
import torch_npu

import pypto

SHAPE = (16, 16)
INDEX = (0, 2, 4, 6, 8, 10, 12, 14)
VALUE = 3.0


@pypto.frontend.jit
def index_put_write_after_read_kernel(
    target: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    index: pypto.Tensor([pypto.STATIC], pypto.DT_INT32),
    values: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 16)
    value = target + 1.0
    pypto.index_put_(target, (index,), values, False)
    pypto.assemble(value + target, [0, 0], out)


def test_index_put_keeps_write_after_read_dependency():
    """The read value must remain available after the in-place update."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch_npu.npu.set_device(device_id)
    device = f"npu:{device_id}"

    target = torch.arange(256, dtype=torch.float32).reshape(SHAPE)
    index = torch.tensor(INDEX, dtype=torch.int32)
    values = torch.full((len(INDEX), SHAPE[1]), VALUE, dtype=torch.float32)
    target_npu, index_npu, values_npu = (x.to(device) for x in (target, index, values))
    out = torch.empty(SHAPE, device=device)

    index_put_write_after_read_kernel(target_npu, index_npu, values_npu, out)

    expected = target.clone()
    expected.index_put_((index.to(torch.int64),), values, accumulate=False)
    assert torch.allclose(target_npu.cpu(), expected)
    assert torch.allclose(out.cpu(), target + 1.0 + expected)


if __name__ == "__main__":
    test_index_put_keeps_write_after_read_dependency()
