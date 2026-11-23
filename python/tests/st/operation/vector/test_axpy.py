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

import math
import os

from numpy.testing import assert_allclose
import pytest
import torch

import pypto

AXPY_CPP_ST_CASES = (
    # case_index, y_shape, x_shape, dtype(y, x, out), view_shape, tile_shape, alpha
    (0, (32, 32), (32, 32), (pypto.DT_FP32, pypto.DT_FP32, pypto.DT_FP32), (16, 16), (16, 16), 2.0),
    (4, (32, 32), (32, 32), (pypto.DT_FP16, pypto.DT_FP16, pypto.DT_FP16), (16, 16), (16, 16), 2.0),
    (6, (64, 64), (64, 64), (pypto.DT_FP32, pypto.DT_FP16, pypto.DT_FP32), (64, 64), (64, 64), 2.0),
    (12, (64, 128), (64, 1), (pypto.DT_FP16, pypto.DT_FP16, pypto.DT_FP16), (64, 128), (64, 128), 2.0),
    (13, (64, 64), (64, 1), (pypto.DT_FP32, pypto.DT_FP16, pypto.DT_FP32), (64, 64), (64, 64), 2.0),
)


@pytest.mark.parametrize(
    "case_index,y_shape,x_shape,dtypes,view_shape,tile_shape,alpha",
    AXPY_CPP_ST_CASES,
    ids=("cpp_case_0", "cpp_case_4", "cpp_case_6", "cpp_case_12", "cpp_case_13"),
)
def test_axpy_cpp_st_onboard(case_index, y_shape, x_shape, dtypes, view_shape, tile_shape, alpha):
    """Python guards for TestAxpy cases 0, 4, 6, 12 and 13."""
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch_dtypes = {
        pypto.DT_FP16: torch.float16,
        pypto.DT_FP32: torch.float32,
    }
    y_dtype, x_dtype, output_dtype = dtypes
    output_shape = y_shape
    loop_ranges = tuple(math.ceil(dim / view_dim) for dim, view_dim in zip(output_shape, view_shape))

    pypto.runtime._device_init()
    try:
        y = pypto.tensor(y_shape, y_dtype, f"AXPY_{case_index}_y")
        x = pypto.tensor(x_shape, x_dtype, f"AXPY_{case_index}_x")
        output = pypto.tensor(output_shape, output_dtype, f"AXPY_{case_index}_output")

        with pypto.function(f"AXPY_CPP_ST_{case_index}", y, x, output):
            for row_index in pypto.loop(loop_ranges[0], name="AXPY_ROW", idx_name="row_index"):
                for column_index in pypto.loop(loop_ranges[1], name="AXPY_COLUMN", idx_name="column_index"):
                    row_offset = row_index * view_shape[0]
                    column_offset = column_index * view_shape[1]
                    x_column_offset = 0 if x_shape[1] == 1 else column_offset
                    x_view_width = min(x_shape[1], view_shape[1])

                    y_view = y[
                        row_offset:row_offset + view_shape[0],
                        column_offset:column_offset + view_shape[1],
                    ]
                    x_view = x[
                        row_offset:row_offset + view_shape[0],
                        x_column_offset:x_column_offset + x_view_width,
                    ]
                    pypto.set_vec_tile_shapes(*tile_shape)
                    y_view.axpy_(x_view, alpha)
                    output[row_offset:, column_offset:] = y_view

        y_data = torch.empty(y_shape, dtype=torch_dtypes[y_dtype]).uniform_(-10, 10)
        x_data = torch.empty(x_shape, dtype=torch_dtypes[x_dtype]).uniform_(-10, 10)
        output_data = torch.zeros(output_shape, dtype=torch_dtypes[output_dtype])
        pypto.runtime._device_run_once_data_from_host(
            pypto.from_torch(y_data, f"AXPY_{case_index}_y_data"),
            pypto.from_torch(x_data, f"AXPY_{case_index}_x_data"),
            pypto.from_torch(output_data, f"AXPY_{case_index}_output_data"),
        )

        expected = y_data + alpha * x_data
        tolerance = 3e-3 if output_dtype == pypto.DT_FP32 else 1e-2
        assert_allclose(output_data, expected, rtol=tolerance, atol=tolerance)
    finally:
        pypto.runtime._device_fini()


@pytest.mark.skip(reason="冒烟跳过")
def test_axpy_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    shape = (64, 64)
    view_shape = (32, 32)
    tile_shape = (32, 32)
    alpha = 2.0
    pypto.runtime._device_init()

    input1 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_input1")
    input2 = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_input2")
    output = pypto.tensor(shape, pypto.DT_FP32, "PTO_TENSOR_output")

    b_loop_num = math.ceil(shape[0] / view_shape[0])
    s_loop_num = math.ceil(shape[1] / view_shape[1])
    with pypto.function("MAIN", input1, input2, output):
        for b_idx in pypto.loop(b_loop_num, name="b0", idx_name="bidx"):
            for s_idx in pypto.loop(s_loop_num, name="s0", idx_name="sidx"):
                view_tensor_a = pypto.view(
                    input1,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                view_tensor_b = pypto.view(
                    input2,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        pypto.min(
                            pypto.symbolic_scalar(shape[0]) - b_idx * view_shape[0],
                            pypto.symbolic_scalar(view_shape[0]),
                        ),
                        pypto.min(
                            pypto.symbolic_scalar(shape[1]) - s_idx * view_shape[1],
                            pypto.symbolic_scalar(view_shape[1]),
                        ),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                view_tensor_a.axpy_(view_tensor_b, alpha)
                pypto.assemble(view_tensor_a, [b_idx * view_shape[0], s_idx * view_shape[1]], output)

    assert isinstance(output, pypto.tensor)

    a_tensor = torch.randn(size=[shape[0], shape[1]], dtype=torch.float32)
    b_tensor = torch.randn(size=[shape[0], shape[1]], dtype=torch.float32)
    c_tensor = torch.zeros(shape[0], shape[1], dtype=torch.float32)
    pto_a_tensor = pypto.from_torch(a_tensor, "a_tensor")
    pto_b_tensor = pypto.from_torch(b_tensor, "b_tensor")
    pto_c_tensor = pypto.from_torch(c_tensor, "c_tensor")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor)

    golden = alpha * b_tensor + a_tensor
    assert_allclose(c_tensor.flatten(), golden.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()
