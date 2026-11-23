#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for UnPack migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.unpack_onboard_test_case import UNPACK_ONBOARD_TESTS, UnpackOnboardConfig
from vector_testcase.vector_test_case import TORCH_DTYPES

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def unpack_onboard_1d_1input_1d_output_kernel(
    input0: pypto.Tensor(), output0: pypto.Tensor(), config: UnpackOnboardConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        offsets = [index_0 * config.execution_view_shape[0]]
        input0_view = input0[:]
        result = pypto.unpack(input0_view, config.output_dtype)
        output_offset = [
            0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]] for axis in range(1)
        ]
        pypto.assemble(result, output_offset, output0)


KERNELS = {
    (1, 1, 1, 1): unpack_onboard_1d_1input_1d_output_kernel,
}


def run_unpack_onboard_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = UnpackOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [
        inputs_cpu[0].flatten().view(TORCH_DTYPES[config.output_tensors[0].dtype]).reshape(config.output_shapes[0])
    ]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", UNPACK_ONBOARD_TESTS, ids=[case["case_name"] for case in UNPACK_ONBOARD_TESTS])
def test_unpack_onboard(case: dict):
    run_unpack_onboard_test(case)
