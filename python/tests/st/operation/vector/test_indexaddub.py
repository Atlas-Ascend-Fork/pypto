#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for IndexAddUB migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.indexaddub_test_case import INDEXADDUB_TESTS, IndexaddubConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def indexaddub_2d_3input_2d_output_kernel(
    input0: pypto.Tensor(),
    input1: pypto.Tensor(),
    input2: pypto.Tensor(),
    output0: pypto.Tensor(),
    config: IndexaddubConfig,
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_view = input0[:]
            input1_view = input1[:]
            input2_view = input2[:]
            result = pypto.index_add_ub(input0_view, config.axis, input2_view, input1_view, alpha=config.alpha)
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(2)
            ]
            pypto.assemble(result, output_offset, output0)


KERNELS = {
    (2, 3, 1, 2): indexaddub_2d_3input_2d_output_kernel,
}


def run_indexaddub_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = IndexaddubConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.index_add(inputs_cpu[0], config.axis, inputs_cpu[2].long(), inputs_cpu[1], alpha=config.alpha)]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", INDEXADDUB_TESTS, ids=[case["case_name"] for case in INDEXADDUB_TESTS])
def test_indexaddub(case: dict):
    run_indexaddub_test(case)
