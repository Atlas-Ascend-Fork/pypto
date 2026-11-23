#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import os

from numpy.testing import assert_allclose
import pytest
import torch

import pypto

RANGE_CPP_ST_CASES = (
    # case_index, input dtypes, output shape/dtype, view_shape, tile_shape, start, end, step
    (0, (pypto.DT_INT32, pypto.DT_INT32, pypto.DT_INT32), (5,), pypto.DT_INT32, (1,), (8,), -204, 807, 247),
    (
        1,
        (pypto.DT_INT32, pypto.DT_INT32, pypto.DT_FP32),
        (33,),
        pypto.DT_FP32,
        (8,),
        (32,),
        -2000,
        3680,
        174.5386,
    ),
)


@pytest.mark.parametrize(
    "case_index,input_dtypes,output_shape,output_dtype,view_shape,tile_shape,start,end,step",
    RANGE_CPP_ST_CASES,
    ids=("cpp_case_0", "cpp_case_1"),
)
def test_range_cpp_st_onboard(
    case_index,
    input_dtypes,
    output_shape,
    output_dtype,
    view_shape,
    tile_shape,
    start,
    end,
    step,
):
    """Python guards for TestRange cases 0 and 1."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch_output_dtype = torch.int32 if output_dtype == pypto.DT_INT32 else torch.float32

    pypto.runtime._device_init()
    try:
        inputs = [
            pypto.tensor((1,), dtype, f"RANGE_{case_index}_input_{index}") for index, dtype in enumerate(input_dtypes)
        ]
        output = pypto.tensor(output_shape, output_dtype, f"RANGE_{case_index}_output")

        with pypto.function(f"RANGE_CPP_ST_{case_index}", *inputs, output):
            for batch_index in pypto.loop(1, name="RANGE_LOOP", idx_name="batch_index"):
                pypto.set_vec_tile_shapes(*tile_shape)
                result = pypto.tensor()
                result.move(pypto.arange(start, end, step))
                output[batch_index * view_shape[0]:] = result

        input_data = [torch.zeros((1,), dtype=torch.int32) for _ in inputs]
        if input_dtypes[-1] == pypto.DT_FP32:
            input_data[-1] = input_data[-1].to(torch.float32)
        output_data = torch.zeros(output_shape, dtype=torch_output_dtype)
        pypto.runtime._device_run_once_data_from_host(
            *(
                pypto.from_torch(data, f"RANGE_{case_index}_input_data_{index}")
                for index, data in enumerate(input_data)
            ),
            pypto.from_torch(output_data, f"RANGE_{case_index}_output_data"),
        )

        expected = torch.arange(start, end, step, dtype=torch_output_dtype)
        assert expected.shape == output_data.shape
        assert_allclose(output_data, expected, rtol=1e-5, atol=1e-5)
    finally:
        pypto.runtime._device_fini()


def test_vector_operation_range():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    _dtype = pypto.DT_FP32
    size = 32
    view_shape = (16,)
    tile_shape = (8,)
    start_data = 1.0
    end_data = 32.1
    step_data = 1.0

    pypto.runtime._device_init()

    a = pypto.tensor((1, 1, 1), pypto.DT_FP32, "Range_TENSOR_a")
    b = pypto.tensor((size,), pypto.DT_FP32, "Range_TENSOR_b")
    start = 1.0
    end = 32.1
    step = 1.0

    with pypto.function("RANGE", a, b):
        for b_idx in pypto.loop(1, name="LOOP_L0_b_idex", idx_name="b_idx"):
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.tensor()
            res.move(pypto.arange(start, end, step))
            pypto.assemble(res, [b_idx * view_shape[0]], b)

    a_tensor = torch.rand([1, 1, 1], dtype=torch.float32) * 99.999 + 0.001
    res_tensor = torch.zeros(size, dtype=torch.float32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_res_tensor = pypto.from_torch(res_tensor, "res_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_res_tensor)

    expected = torch.arange(start_data, end_data, step_data)
    assert_allclose(res_tensor.flatten(), expected.flatten(), rtol=1e-6, atol=1e-7)

    pypto.runtime._device_fini()
