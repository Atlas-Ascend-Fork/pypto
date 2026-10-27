#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Prod migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.prod_onboard_test_case import PROD_ONBOARD_TESTS, ProdOnboardConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def prod_onboard_2d_1input_2d_output_kernel(input0: pypto.Tensor(), output0: pypto.Tensor(), config: ProdOnboardConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(2)]
            input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
            result = pypto.prod(input0_view, config.dims[0], config.keep_dim)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(2)
            ]
            pypto.assemble(result, output_offset, output0)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def prod_onboard_3d_1input_2d_output_kernel(input0: pypto.Tensor(), output0: pypto.Tensor(), config: ProdOnboardConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            for index_2 in pypto.loop(config.loop_ranges[2]):
                offset_0 = index_0 * config.execution_view_shape[0]
                offset_1 = index_1 * config.execution_view_shape[1]
                offset_2 = index_2 * config.execution_view_shape[2]
                input0_view = input0[
                    offset_0:offset_0 + config.execution_view_shape[0],
                    offset_1:offset_1 + config.execution_view_shape[1],
                    offset_2:offset_2 + config.execution_view_shape[2],
                ]
                result = pypto.prod(input0_view, config.dims[0], config.keep_dim)
                output0[
                    offset_1:offset_1 + config.execution_view_shape[1],
                    offset_2:offset_2 + config.execution_view_shape[2],
                ] = result


KERNELS = {
    (2, 1, 1, 2): prod_onboard_2d_1input_2d_output_kernel,
    (3, 1, 1, 2): prod_onboard_3d_1input_2d_output_kernel,
}


def run_prod_onboard_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = ProdOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.prod(inputs_cpu[0], dim=config.dims[0], keepdim=config.keep_dim)]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", PROD_ONBOARD_TESTS, ids=[case["case_name"] for case in PROD_ONBOARD_TESTS])
@pypto.options(pass_options={"enable_slice": True})
def test_prod_onboard(case: dict):
    run_prod_onboard_test(case)
