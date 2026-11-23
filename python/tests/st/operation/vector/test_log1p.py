#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Log1p migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.log1p_test_case import LOG1P_TESTS, Log1pConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def log1p_2d_1input_kernel(input0: pypto.Tensor(), output: pypto.Tensor(), config: Log1pConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(2)]
            input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
            result = pypto.log1p(input0_view)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(len(config.execution_view_shape))
            ]
            pypto.assemble(result, output_offset, output)


KERNELS = {
    (2, 1): log1p_2d_1input_kernel,
}


def run_log1p_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = Log1pConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.log1p(inputs_cpu[0])]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs))](*inputs, *outputs, config)
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", LOG1P_TESTS, ids=[case["case_name"] for case in LOG1P_TESTS])
def test_log1p(case: dict):
    run_log1p_test(case)
