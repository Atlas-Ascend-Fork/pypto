#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for Gather migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.gather_onboard_test_case import GATHER_ONBOARD_TESTS, GatherOnboardConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gather_onboard_2d_2input_2d_output_kernel(
    input0: pypto.Tensor(), input1: pypto.Tensor(), output0: pypto.Tensor(), config: GatherOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_view = input0[:]
            input1_view = input1[:]
            result = pypto.index_select(input0_view, config.axis, input1_view)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(2)
            ]
            pypto.assemble(result, output_offset, output0)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gather_onboard_3d_2input_3d_output_kernel(
    input0: pypto.Tensor(), input1: pypto.Tensor(), output0: pypto.Tensor(), config: GatherOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            for index_2 in pypto.loop(config.loop_ranges[2]):
                offsets = [
                    index_0 * config.execution_view_shape[0],
                    index_1 * config.execution_view_shape[1],
                    index_2 * config.execution_view_shape[2],
                ]
                input0_view = input0[:]
                input1_view = input1[:]
                result = pypto.index_select(input0_view, config.axis, input1_view)
                output_offset = [
                    0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                    for axis in range(3)
                ]
                pypto.assemble(result, output_offset, output0)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gather_onboard_3d_2input_4d_output_kernel(
    input0: pypto.Tensor(), input1: pypto.Tensor(), output0: pypto.Tensor(), config: GatherOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            for index_2 in pypto.loop(config.loop_ranges[2]):
                offsets = [
                    index_0 * config.execution_view_shape[0],
                    index_1 * config.execution_view_shape[1],
                    index_2 * config.execution_view_shape[2],
                ]
                input0_view = input0[:]
                input1_view = input1[:]
                result = pypto.index_select(input0_view, config.axis, input1_view)
                output_offset = [
                    0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                    for axis in range(4)
                ]
                pypto.assemble(result, output_offset, output0)


KERNELS = {
    (2, 2, 1, 2): gather_onboard_2d_2input_2d_output_kernel,
    (3, 2, 1, 3): gather_onboard_3d_2input_3d_output_kernel,
    (3, 2, 1, 4): gather_onboard_3d_2input_4d_output_kernel,
}


def run_gather_onboard_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = GatherOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [
        torch.index_select(inputs_cpu[0], config.axis, inputs_cpu[1].long().flatten()).reshape(config.output_shapes[0])
    ]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", GATHER_ONBOARD_TESTS, ids=[case["case_name"] for case in GATHER_ONBOARD_TESTS])
@pytest.mark.skip(reason="ci skip")
@pypto.options(pass_options={"enable_slice": True})
def test_gather_onboard(case: dict):
    run_gather_onboard_test(case)
