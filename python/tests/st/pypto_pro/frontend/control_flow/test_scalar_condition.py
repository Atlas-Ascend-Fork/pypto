# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scalar truth conversion in if/while, including signed zero, infinity and NaN."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

DTYPES = [
    (pl.DT_BOOL, torch.bool),
    (pl.DT_INT32, torch.int32),
    (pl.DT_INT64, torch.int64),
    (pl.DT_FP16, torch.float16),
    (pl.DT_BF16, torch.bfloat16),
    (pl.DT_FP32, torch.float32),
]


def make_condition_kernel(dtype, cube):
    def check_conditions(a, out):
        for i in pl.range(a.shape[0]):
            # Integer tensor reads widen to INT64; keep the loop-carried dtype explicit.
            x = pl.astype(a[i], dtype)
            if x:
                selected = pl.const(1, pl.DT_INT32)
            else:
                selected = pl.const(0, pl.DT_INT32)
            out[0, i] = selected
            count = pl.const(0, pl.DT_INT32)
            condition = x
            while condition:
                count = count + pl.const(1, pl.DT_INT32)
                condition = pl.const(0, dtype)
            out[1, i] = count

    @pl.jit(auto_mutex=False)
    def kernel(a: pl.Tensor[[pl.DYNAMIC], dtype], out: pl.Tensor[[2, pl.DYNAMIC], pl.DT_INT32]):
        if cube:
            with pl.section_cube():
                check_conditions(a, out)
        else:
            with pl.section_vector():
                check_conditions(a, out)

    return kernel


@pytest.mark.soc("950")
@pytest.mark.parametrize("dtype, torch_dtype", DTYPES, ids=[str(pair[0]) for pair in DTYPES])
@pytest.mark.parametrize("cube", [False, True], ids=["vector", "cube"])
def test_scalar_conditions(dtype, torch_dtype, cube):
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    values = [-3, -1, 0, 1, 3]
    if torch_dtype.is_floating_point:
        values = [-float("inf"), -3.5, -0.0, 0.0, 0.5, float("inf"), float("nan")]
    a = torch.tensor(values, dtype=torch_dtype)
    expected = a.bool().int().expand(2, -1)
    out = torch.empty(expected.shape, dtype=torch.int32, device=device)
    make_condition_kernel(dtype, cube)[None, 1](a.to(device), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
