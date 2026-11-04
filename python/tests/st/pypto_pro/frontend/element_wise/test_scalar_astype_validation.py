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
"""JIT validation and device coverage for scalar casts used by min/max."""

import os

from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


def _make_scalar_minmax_cast_kernel(dtype):
    @pl.jit()
    def kernel(src: pl.Tensor[[2, 8], pl.DT_FP32], out: pl.Tensor[[2, 8], pl.DT_FP32]):
        with pl.section_vector():
            for i in pl.range(8):
                a = pl.astype(src[0, i], dtype)
                b = pl.astype(src[1, i], dtype)
                out[0, i] = pl.astype(pl.min(a, b), pl.DT_FP32)
                out[1, i] = pl.astype(pl.max(a, b), pl.DT_FP32)

    return kernel


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "dtype",
    [pl.DT_FP8E4M3FN, pl.DT_FP8E5M2, pl.DT_FP8E8M0, pl.DT_FP4E2M1, pl.DT_FP4E1M2],
    ids=str,
)
def test_minmax_rejects_scalar_cast_to_fp8_fp4(dtype):
    torch.npu.set_device(ST_DEVICE)
    src = torch.ones((2, 8), dtype=torch.float32, device=ST_DEVICE)
    out = torch.full_like(src, 123.0)
    kernel = _make_scalar_minmax_cast_kernel(dtype)
    with pytest.raises(NotSupported, match=r"pl\.astype\(\).*destination dtype.*storage-only"):
        kernel[None, 1](src, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.full((2, 8), 123.0), rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [(pl.DT_FP16, torch.float16), (pl.DT_FP32, torch.float32), (pl.DT_INT32, torch.int32)],
    ids=["fp16", "fp32", "int32"],
)
def test_minmax_with_supported_scalar_cast_runs_on_device(dtype, torch_dtype):
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [-3.5, 2.5, 1.0003, -1.0003, 64.0, -64.0, 0.0, 0.5],
        [2.5, -3.5, 1.001, -1.001, -64.0, 64.0, 0.0, 0.5],
    ])
    out = torch.empty_like(src, device=ST_DEVICE)
    kernel = _make_scalar_minmax_cast_kernel(dtype)
    kernel[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()
    converted = src.to(torch_dtype)
    expected = torch.stack((torch.minimum(converted[0], converted[1]), torch.maximum(converted[0], converted[1])))
    torch.testing.assert_close(out.cpu(), expected.float(), rtol=0, atol=0)
