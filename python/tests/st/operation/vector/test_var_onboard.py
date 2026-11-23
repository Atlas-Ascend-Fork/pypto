#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Var migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.var_onboard_test_case import VAR_ONBOARD_TESTS, VarOnboardConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def var_onboard_2d_1input_1d_output_kernel(input0: pypto.Tensor(), output0: pypto.Tensor(), config: VarOnboardConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(2)]
            input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
            result = pypto.var(input0_view, config.dims, config.correction, config.keep_dim)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(1)
            ]
            pypto.assemble(result, output_offset, output0)


KERNELS = {
    (2, 1, 1, 1): var_onboard_2d_1input_1d_output_kernel,
}


def run_var_onboard_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = VarOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.var(inputs_cpu[0], dim=config.dims, correction=config.correction, keepdim=config.keep_dim)]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", VAR_ONBOARD_TESTS, ids=[case["case_name"] for case in VAR_ONBOARD_TESTS])
def test_var_onboard(case: dict):
    run_var_onboard_test(case)
