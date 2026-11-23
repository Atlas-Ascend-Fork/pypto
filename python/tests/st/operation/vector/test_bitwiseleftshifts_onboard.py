#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for BitwiseLeftShifts migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.bitwiseleftshifts_onboard_test_case import (
    BITWISELEFTSHIFTS_ONBOARD_TESTS,
    BitwiseleftshiftsOnboardConfig,
)

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def bitwiseleftshifts_onboard_2d_1input_kernel(
    input0: pypto.Tensor(), output: pypto.Tensor(), config: BitwiseleftshiftsOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(2)]
            input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
            result = pypto.bitwise_left_shift(input0_view, config.scalar)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(len(config.execution_view_shape))
            ]
            pypto.assemble(result, output_offset, output)


KERNELS = {
    (2, 1): bitwiseleftshifts_onboard_2d_1input_kernel,
}


def run_bitwiseleftshifts_onboard_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = BitwiseleftshiftsOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.bitwise_left_shift(inputs_cpu[0], config.scalar)]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs))](*inputs, *outputs, config)
    assert_outputs(outputs, expected)


@pytest.mark.parametrize(
    "case", BITWISELEFTSHIFTS_ONBOARD_TESTS, ids=[case["case_name"] for case in BITWISELEFTSHIFTS_ONBOARD_TESTS]
)
def test_bitwiseleftshifts_onboard(case: dict):
    run_bitwiseleftshifts_onboard_test(case)
