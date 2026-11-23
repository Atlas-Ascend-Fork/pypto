#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for GatherElement migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.gatherelement_test_case import GATHERELEMENT_TESTS, GatherelementConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gatherelement_2d_2input_2d_output_kernel(
    input0: pypto.Tensor(), input1: pypto.Tensor(), output0: pypto.Tensor(), config: GatherelementConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_view = input0[:]
            input1_view = input1[:]
            result = pypto.gather(input0_view, config.axis, input1_view)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(2)
            ]
            pypto.assemble(result, output_offset, output0)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def gatherelement_3d_2input_3d_output_kernel(
    input0: pypto.Tensor(), input1: pypto.Tensor(), output0: pypto.Tensor(), config: GatherelementConfig
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
                result = pypto.gather(input0_view, config.axis, input1_view)
                output_offset = [
                    0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                    for axis in range(3)
                ]
                pypto.assemble(result, output_offset, output0)


KERNELS = {
    (2, 2, 1, 2): gatherelement_2d_2input_2d_output_kernel,
    (3, 2, 1, 3): gatherelement_3d_2input_3d_output_kernel,
}


def run_gatherelement_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = GatherelementConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.gather(inputs_cpu[0], config.axis, inputs_cpu[1].long())]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", GATHERELEMENT_TESTS, ids=[case["case_name"] for case in GATHERELEMENT_TESTS])
def test_gatherelement(case: dict):
    run_gatherelement_test(case)
