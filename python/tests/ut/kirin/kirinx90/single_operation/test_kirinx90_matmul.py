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
Test matmul codegen for KirinX90
"""

from kirin.common.common_matmul import (
    MATMUL_CASES,
    QUANT_CASES,
    build_test_cases,
    create_test_matmul_module,
    create_test_matmul_quant_module,
    run_matmul_test,
    run_quant_matmul_test,
)
import pytest

SOC_VERSION = "KirinX90"

# 平台支持用例（allow-list），不在列表中的用例不参与收集。
_MATMUL_IDS = {f"{i:03d}" for i in range(1, 56)}
_QUANT_IDS = {"056", "057", "058", "059", "060", "061", "062", "063", "064", "065", "066"}

TEST_CASES = build_test_cases(MATMUL_CASES, _MATMUL_IDS)
QUANT_TEST_CASES = build_test_cases(QUANT_CASES, _QUANT_IDS)

KERNELS = create_test_matmul_module(SOC_VERSION, TEST_CASES)
QUANT_KERNELS = create_test_matmul_quant_module(SOC_VERSION, QUANT_TEST_CASES)


@pytest.mark.parametrize(
    "kernel_name,torch_in_dtype,torch_out_dtype,pypto_in_dtype,pypto_out_dtype,"
    "shape_a,shape_b,shape_c,cube_tile_shape,has_bias,bias_shape,a_trans,b_trans",
    TEST_CASES,
)
def test_matmul(
    kernel_name,
    torch_in_dtype,
    torch_out_dtype,
    pypto_in_dtype,
    pypto_out_dtype,
    shape_a,
    shape_b,
    shape_c,
    cube_tile_shape,
    has_bias,
    bias_shape,
    a_trans,
    b_trans,
):
    run_matmul_test(
        KERNELS,
        kernel_name,
        torch_in_dtype,
        torch_out_dtype,
        shape_a,
        shape_b,
        shape_c,
        cube_tile_shape,
        has_bias,
        bias_shape,
        a_trans,
        b_trans,
        SOC_VERSION,
    )


@pytest.mark.parametrize(
    "kernel_name,torch_in_dtype,torch_out_dtype,pypto_in_dtype,pypto_out_dtype,shape_a,shape_b,shape_c,"
    "cube_tile_shape,has_bias,bias_shape,a_trans,b_trans",
    QUANT_TEST_CASES,
)
def test_matmul_quant(
    kernel_name,
    torch_in_dtype,
    torch_out_dtype,
    pypto_in_dtype,
    pypto_out_dtype,
    shape_a,
    shape_b,
    shape_c,
    cube_tile_shape,
    has_bias,
    bias_shape,
    a_trans,
    b_trans,
):
    run_quant_matmul_test(
        QUANT_KERNELS,
        kernel_name,
        torch_in_dtype,
        torch_out_dtype,
        shape_a,
        shape_b,
        shape_c,
        cube_tile_shape,
        has_bias,
        bias_shape,
        a_trans,
        b_trans,
        SOC_VERSION,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
