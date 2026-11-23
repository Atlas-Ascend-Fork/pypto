#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Exp2 migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.exp2_test_case import EXP2_TESTS, Exp2Config

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def exp2_2d_1input_kernel(input0: pypto.Tensor(), output: pypto.Tensor(), config: Exp2Config):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(2)]
            input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
            result = pypto.exp2(input0_view)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(len(config.execution_view_shape))
            ]
            pypto.assemble(result, output_offset, output)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def exp2_3d_1input_kernel(input0: pypto.Tensor(), output: pypto.Tensor(), config: Exp2Config):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            for index_2 in pypto.loop(config.loop_ranges[2]):
                offsets = [
                    index_0 * config.execution_view_shape[0],
                    index_1 * config.execution_view_shape[1],
                    index_2 * config.execution_view_shape[2],
                ]
                input0_offset = [
                    0 if config.input_shapes[0][axis] == 1 else offsets[axis]
                    for axis in range(len(config.execution_view_shape))
                ]
                input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
                result = pypto.exp2(input0_view)
                output_offset = [
                    0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                    for axis in range(len(config.output_shapes[0]))
                ]
                pypto.assemble(result, output_offset, output)


KERNELS = {
    (2, 1): exp2_2d_1input_kernel,
    (3, 1): exp2_3d_1input_kernel,
}


def run_exp2_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = Exp2Config.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.exp2(inputs_cpu[0])]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs))](*inputs, *outputs, config)
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", EXP2_TESTS, ids=[case["case_name"] for case in EXP2_TESTS])
def test_exp2(case: dict):
    run_exp2_test(case)
