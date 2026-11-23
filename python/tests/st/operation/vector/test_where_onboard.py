#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Where migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.where_onboard_test_case import WHERE_ONBOARD_TESTS, WhereOnboardConfig

import pypto


def _where_value(tensor, scalar, use_scalar, dtype):
    if use_scalar:
        return pypto.Element(dtype, scalar).base()
    return tensor.base()


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def where_2d_kernel(
    condition: pypto.Tensor(), x: pypto.Tensor(), y: pypto.Tensor(), output: pypto.Tensor(), config: WhereOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for row_index in pypto.loop(config.loop_ranges[0]):
        for column_index in pypto.loop(config.loop_ranges[1]):
            offsets = [row_index * config.execution_view_shape[0], column_index * config.execution_view_shape[1]]
            condition_view = pypto.view(condition, config.input_view_shapes[0], offsets)
            x_view = pypto.view(
                x,
                config.input_view_shapes[1],
                [0 if config.input_shapes[1][axis] == 1 else offsets[axis] for axis in range(2)],
            )
            y_view = pypto.view(
                y,
                config.input_view_shapes[2],
                [0 if config.input_shapes[2][axis] == 1 else offsets[axis] for axis in range(2)],
            )
            x_value = config.x_scalar if config.flag in (2, 3) else x_view
            y_value = config.y_scalar if config.flag in (1, 3) else y_view
            pypto.assemble(pypto.where(condition_view, x_value, y_value), offsets, output)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def where_packed_kernel(
    condition: pypto.Tensor(), x: pypto.Tensor(), y: pypto.Tensor(), output: pypto.Tensor(), config: WhereOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    x_base = _where_value(x, config.x_scalar, config.flag in (2, 3), config.output_dtype)
    y_base = _where_value(y, config.y_scalar, config.flag in (1, 3), config.output_dtype)
    result = pypto.Tensor.from_base(pypto.pypto_impl.Where(condition.base(), x_base, y_base))
    output[:] = result


def where_golden(inputs, config):
    condition = inputs[0]
    if config.condition_is_packed:
        shifts = torch.arange(8, dtype=torch.uint8)
        condition = ((condition.unsqueeze(-1) >> shifts) & 1).bool().reshape(config.output_shapes[0])
    x = config.x_scalar if config.flag in (2, 3) else inputs[1]
    y = config.y_scalar if config.flag in (1, 3) else inputs[2]
    return [torch.where(condition.bool(), x, y)]


def run_migrated_where_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = WhereOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = where_golden(inputs_cpu, config)
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    kernel = where_packed_kernel if config.condition_is_packed else where_2d_kernel
    kernel(*inputs, *outputs, config)
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", WHERE_ONBOARD_TESTS, ids=[case["case_name"] for case in WHERE_ONBOARD_TESTS])
def test_migrated_where(case: dict):
    run_migrated_where_test(case)


import os  # noqa: E402, F811

import numpy as np  # noqa: E402, F811
from numpy.testing import assert_allclose  # noqa: E402, F811
import torch  # noqa: E402, F811

import pypto  # noqa: E402, F811


def test_vector_operation_where():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    dtype = pypto.DT_FP32
    tiling = 32
    n, m = int(tiling * 2.3), int(tiling * 2.7)
    shape = (n, m)
    view_shape = (16, 16)
    tile_shape = (8, 8)
    pypto.runtime._device_init()
    condition = pypto.tensor(shape, pypto.DT_BOOL, "WHERE_TENSOR_cond")
    input_base = pypto.tensor(shape, dtype, "WHERE_TENSOR_input")
    other_base = pypto.tensor(shape, dtype, "WHERE_TENSOR_other")
    out = pypto.tensor(shape, dtype, "WHERE_TENSOR_out")

    with pypto.function("WHERE", condition, input_base, other_base, out):
        for b_idx in pypto.loop(int(np.ceil(n / view_shape[0])), name="LOOP_ADD_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(m / view_shape[1])), name="LOOP_ADD_L1", idx_name="s_idx"):
                tile_cond = pypto.view(
                    condition,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                tile_input = pypto.view(
                    input_base,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                tile_other = pypto.view(
                    other_base,
                    view_shape,
                    [b_idx * view_shape[0], s_idx * view_shape[1]],
                    valid_shape=[
                        (pypto.symbolic_scalar(n) - b_idx * view_shape[0]).min(pypto.symbolic_scalar(view_shape[0])),
                        (pypto.symbolic_scalar(m) - s_idx * view_shape[1]).min(pypto.symbolic_scalar(view_shape[1])),
                    ],
                )
                pypto.set_vec_tile_shapes(tile_shape[0], tile_shape[1])
                tile_input.move(pypto.where(tile_cond, tile_input, tile_other))
                pypto.assemble(tile_input, [b_idx * view_shape[0], s_idx * view_shape[1]], out)

    cond_tensor = torch.randint(0, 2, (n, m), dtype=torch.bool)
    input_tensor = torch.rand(n, m, dtype=torch.float32)
    other_tensor = torch.zeros(n, m, dtype=torch.float32)
    out_tensor = torch.zeros(n, m, dtype=torch.float32)

    pto_input_tensor = pypto.from_torch(input_tensor, "input_tensor")
    pto_other_tensor = pypto.from_torch(other_tensor, "other_tensor")
    pto_cond_tensor = pypto.from_torch(cond_tensor, "cond_tensor")
    pto_out_tensor = pypto.from_torch(out_tensor, "out_tensor")

    pypto.runtime._device_run_once_data_from_host(pto_cond_tensor, pto_input_tensor, pto_other_tensor, pto_out_tensor)

    expected = torch.where(cond_tensor, input_tensor, other_tensor)
    assert_allclose(out_tensor.flatten(), expected.flatten(), rtol=1e-3, atol=1e-3)
    pypto.runtime._device_fini()
