#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Fmod migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.fmod_test_case import FMOD_TESTS, FmodConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def fmod_2d_2input_kernel(input0: pypto.Tensor(), input1: pypto.Tensor(), output: pypto.Tensor(), config: FmodConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(2)]
            input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
            input1_offset = [0 if config.input_shapes[1][axis] == 1 else offsets[axis] for axis in range(2)]
            input1_view = pypto.view(input1, config.input_view_shapes[1], input1_offset)
            result = pypto.fmod(input0_view, input1_view)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(len(config.execution_view_shape))
            ]
            pypto.assemble(result, output_offset, output)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def fmod_4d_2input_kernel(input0: pypto.Tensor(), input1: pypto.Tensor(), output: pypto.Tensor(), config: FmodConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            for index_2 in pypto.loop(config.loop_ranges[2]):
                for index_3 in pypto.loop(config.loop_ranges[3]):
                    offsets = [
                        index_0 * config.execution_view_shape[0],
                        index_1 * config.execution_view_shape[1],
                        index_2 * config.execution_view_shape[2],
                        index_3 * config.execution_view_shape[3],
                    ]
                    input0_offset = [
                        0 if config.input_shapes[0][axis] == 1 else offsets[axis]
                        for axis in range(len(config.execution_view_shape))
                    ]
                    input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
                    input1_offset = [0 if config.input_shapes[1][axis] == 1 else offsets[axis] for axis in range(4)]
                    input1_view = pypto.view(input1, config.input_view_shapes[1], input1_offset)
                    result = pypto.fmod(input0_view, input1_view)
                    output_offset = [
                        0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                        for axis in range(len(config.execution_view_shape))
                    ]
                    pypto.assemble(result, output_offset, output)


KERNELS = {
    (2, 2): fmod_2d_2input_kernel,
    (4, 2): fmod_4d_2input_kernel,
}


def run_fmod_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = FmodConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.fmod(*inputs_cpu)]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs))](*inputs, *outputs, config)
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", FMOD_TESTS, ids=[case["case_name"] for case in FMOD_TESTS])
def test_fmod(case: dict):
    run_fmod_test(case)
