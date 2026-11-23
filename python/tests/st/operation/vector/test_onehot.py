#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for OneHot migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.onehot_test_case import ONEHOT_TESTS, OnehotConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def onehot_1d_1input_2d_output_kernel(input0: pypto.Tensor(), output0: pypto.Tensor(), config: OnehotConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        offsets = [index_0 * config.execution_view_shape[0]]
        input0_view = input0[:]
        result = pypto.one_hot(input0_view, config.num_classes)
        output_offset = [
            0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]] for axis in range(2)
        ]
        pypto.assemble(result, output_offset, output0)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def onehot_2d_1input_3d_output_kernel(input0: pypto.Tensor(), output0: pypto.Tensor(), config: OnehotConfig):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_view = input0[:]
            result = pypto.one_hot(input0_view, config.num_classes)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(3)
            ]
            pypto.assemble(result, output_offset, output0)


KERNELS = {
    (1, 1, 1, 2): onehot_1d_1input_2d_output_kernel,
    (2, 1, 1, 3): onehot_2d_1input_3d_output_kernel,
}


def run_onehot_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = OnehotConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.nn.functional.one_hot(inputs_cpu[0].long(), config.num_classes)]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", ONEHOT_TESTS, ids=[case["case_name"] for case in ONEHOT_TESTS])
def test_onehot(case: dict):
    run_onehot_test(case)
