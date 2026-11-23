#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Quantize migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.quantize_onboard_test_case import QUANTIZE_ONBOARD_TESTS, QuantizeOnboardConfig
from vector_testcase.vector_test_case import TORCH_DTYPES

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def quantize_onboard_2d_2input_2d_output_kernel(
    input0: pypto.Tensor(), input1: pypto.Tensor(), output0: pypto.Tensor(), config: QuantizeOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_view = input0[:]
            input1_view = input1[:]
            result = pypto.quantize(input0_view, input1_view, config.output_dtype, config.axis, None)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(2)
            ]
            pypto.assemble(result, output_offset, output0)


def _channel_param(value, source, axis):
    normalized_axis = axis if axis >= 0 else source.dim() + axis
    return value.unsqueeze(1) if normalized_axis == 1 else value.unsqueeze(0)


def quantize_golden(inputs, config):
    result = inputs[0] * _channel_param(inputs[1], inputs[0], config.axis)
    if config.use_zero_points:
        result += _channel_param(inputs[2], inputs[0], config.axis)
    outputs_dtype = TORCH_DTYPES[config.output_tensors[0].dtype]
    limits = torch.iinfo(outputs_dtype)
    return [torch.round(result).clamp(limits.min, limits.max).to(outputs_dtype)]


def dequantize_golden(inputs, config):
    result = inputs[0].to(torch.float32)
    if config.use_zero_points:
        result -= _channel_param(inputs[2], inputs[0], config.axis)
    return [result * _channel_param(inputs[1], inputs[0], config.axis)]


KERNELS = {
    (2, 2, 1, 2): quantize_onboard_2d_2input_2d_output_kernel,
}


def run_quantize_onboard_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = QuantizeOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = quantize_golden(inputs_cpu, config)
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", QUANTIZE_ONBOARD_TESTS, ids=[case["case_name"] for case in QUANTIZE_ONBOARD_TESTS])
def test_quantize_onboard(case: dict):
    run_quantize_onboard_test(case)
