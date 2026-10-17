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
Test silu_mul codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def silu_mul_golden(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x) * y


def make_silu_mul_kernel(soc_version, name, dtype, tile_shapes):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        x: pypto.Tensor([...], dtype),
        y: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        sigmoid_x = pypto.sigmoid(x)
        silu_x = pypto.mul(x, sigmoid_x)
        output[:] = pypto.mul(silu_x, y)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_dtype: torch data type (float16)
    # pypto_dtype: pypto data type
    # tile_shapes: tile shape for pypto kernel
    # shape: input tensor shape (x and y have same shape)
    # marks: pytest marks
    pytest.param(
        "silu_mul_prefill",
        torch.float16,
        pypto.DT_FP16,
        (1, 3072),
        (64, 3072),
        marks=[],
        id="001",
    ),
    pytest.param(
        "silu_mul_decoder",
        torch.float16,
        pypto.DT_FP16,
        (1, 768),
        (1, 3072),
        marks=[pytest.mark.skip()],
        id="002",
    ),
]


def run_silu_mul_test(kernels, kernel_name, dtype, shape):
    device = "cpu"
    x = torch.rand(shape, dtype=dtype, device=device)
    y = torch.rand(shape, dtype=dtype, device=device)
    output = torch.rand(shape, dtype=dtype, device=device)

    golden = silu_mul_golden(x, y)

    kernels[kernel_name](x, y, output)

    check_nan(output, name=kernel_name)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_silu_mul_kernels(soc_version):
    return {
        p.values[0]: make_silu_mul_kernel(soc_version, p.values[0], p.values[2], p.values[3])
        for p in TEST_CASES
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
