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
Test Kirin dynamic-to-static codegen (loop unroll + dynamic shape staticization) for KirinX90.
"""

from kirin.common_controlflow import (
    ADAM_TEST_CASES,
    DYNAMIC_MULTI_TEST_CASES,
    DYNAMIC_TEST_CASES,
    LOOP_TEST_CASES,
    create_adam_kernels,
    create_dynamic_kernels,
    create_loop_kernels,
    run_adam_test,
    run_dynamic_multi_test,
    run_dynamic_test,
    run_from_torch_test,
    run_loop_test,
)
import pytest

LOOP_KERNELS = create_loop_kernels("KirinX90")
DYNAMIC_KERNELS = create_dynamic_kernels("KirinX90")
ADAM_KERNELS = create_adam_kernels("KirinX90")


@pytest.mark.parametrize(
    "kernel_name,torch_dtype,pypto_dtype,tile_shapes,loop_num,block,shape,kind",
    LOOP_TEST_CASES,
)
def test_loop(kernel_name, torch_dtype, pypto_dtype, tile_shapes, loop_num, block, shape, kind):
    run_loop_test(LOOP_KERNELS, kernel_name, torch_dtype, shape, loop_num, block, kind)


@pytest.mark.parametrize(
    "kernel_name,torch_dtype,pypto_dtype,tile_shapes,shape",
    DYNAMIC_TEST_CASES,
)
def test_dynamic(kernel_name, torch_dtype, pypto_dtype, tile_shapes, shape):
    run_dynamic_test(DYNAMIC_KERNELS, kernel_name, torch_dtype, shape)


@pytest.mark.parametrize(
    "kernel_name,torch_dtype,pypto_dtype,tile_shapes,shape",
    DYNAMIC_MULTI_TEST_CASES,
)
def test_dynamic_multi(kernel_name, torch_dtype, pypto_dtype, tile_shapes, shape):
    run_dynamic_multi_test(DYNAMIC_KERNELS, kernel_name, torch_dtype, shape)


@pytest.mark.parametrize(
    "kernel_name,torch_dtype,pypto_dtype,tile_shapes,shape",
    ADAM_TEST_CASES,
)
def test_nested_loop_adam(kernel_name, torch_dtype, pypto_dtype, tile_shapes, shape):
    run_adam_test(ADAM_KERNELS, kernel_name, torch_dtype, shape, tile_shapes)


def test_from_torch():
    run_from_torch_test()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
