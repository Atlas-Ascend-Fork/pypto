#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for GatherMask migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.gathermask_test_case import GATHERMASK_TESTS, GathermaskConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gathermask_3d_1input_3d_output_kernel(input0: pypto.Tensor(), output0: pypto.Tensor(), config: GathermaskConfig):
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
                result = pypto.gathermask(input0_view, config.pattern_mode)
                output_offset = [
                    0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                    for axis in range(3)
                ]
                pypto.assemble(result, output_offset, output0)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gathermask_4d_1input_4d_output_kernel(input0: pypto.Tensor(), output0: pypto.Tensor(), config: GathermaskConfig):
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
                    input0_view = input0[:]
                    result = pypto.gathermask(input0_view, config.pattern_mode)
                    output_offset = [
                        0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                        for axis in range(4)
                    ]
                    pypto.assemble(result, output_offset, output0)


def gathermask_golden(inputs, config):
    if config.pattern_mode == 7:
        return [inputs[0]]
    step = 2 if config.pattern_mode <= 2 else 4
    start = config.pattern_mode - 1 if config.pattern_mode <= 2 else config.pattern_mode - 3
    return [inputs[0][..., start::step]]


KERNELS = {
    (3, 1, 1, 3): gathermask_3d_1input_3d_output_kernel,
    (4, 1, 1, 4): gathermask_4d_1input_4d_output_kernel,
}


def run_gathermask_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = GathermaskConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = gathermask_golden(inputs_cpu, config)
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", GATHERMASK_TESTS, ids=[case["case_name"] for case in GATHERMASK_TESTS])
def test_gathermask(case: dict):
    run_gathermask_test(case)
