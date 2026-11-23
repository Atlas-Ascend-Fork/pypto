#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for ScatterUpdate migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs
from vector_testcase.scatterupdate_onboard_test_case import SCATTERUPDATE_ONBOARD_TESTS, ScatterupdateOnboardConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scatterupdate_onboard_2d_3input_2d_output_kernel(
    input0: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    input1: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    input2: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    config: ScatterupdateOnboardConfig,
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    index_rows, index_columns = input1.shape
    source_view_rows = config.view_shape[0]
    index_view_rows = source_view_rows // index_columns
    row_loops = (index_rows + index_view_rows - 1) // index_view_rows
    for row_index in pypto.loop(row_loops):
        source_offset = row_index * source_view_rows
        index_offset = row_index * index_view_rows
        source_view = input0[source_offset:source_offset + source_view_rows, :]
        index_view = input1[index_offset:index_offset + index_view_rows, :]
        input2.move(pypto.scatter_update(input2, -2, index_view, source_view))


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def scatterupdate_onboard_4d_3input_4d_output_kernel(
    input0: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    input1: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC]),
    input2: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC]),
    config: ScatterupdateOnboardConfig,
):
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
                    index_view = input1[
                        offsets[0]:offsets[0] + config.execution_view_shape[0],
                        offsets[1]:offsets[1] + config.execution_view_shape[1],
                    ]
                    source_view = input0[
                        offsets[0]:offsets[0] + config.execution_view_shape[0],
                        offsets[1]:offsets[1] + config.execution_view_shape[1],
                        :,
                        :,
                    ]
                    input2.move(pypto.scatter_update(input2, -2, index_view, source_view))


def scatter_update_golden(inputs, config):
    source, index, destination = inputs
    result = destination.clone()
    if source.dim() == 4:
        block_size = destination.shape[1]
        for batch in range(index.shape[0]):
            for sequence in range(index.shape[1]):
                position = int(index[batch, sequence])
                result[position // block_size, position % block_size] = source[batch, sequence]
    else:
        for batch in range(index.shape[0]):
            for sequence in range(index.shape[1]):
                result[int(index[batch, sequence])] = source[batch * index.shape[1] + sequence]
    return [result]


KERNELS = {
    (2, 3): scatterupdate_onboard_2d_3input_2d_output_kernel,
    (4, 3): scatterupdate_onboard_4d_3input_4d_output_kernel,
}


def run_scatterupdate_onboard_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = ScatterupdateOnboardConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    index = inputs_cpu[1]
    index_capacity = inputs_cpu[2].shape[0] * (inputs_cpu[2].shape[1] if inputs_cpu[2].dim() == 4 else 1)
    inputs_cpu[1] = torch.randperm(index_capacity, dtype=index.dtype)[:index.numel()].reshape(index.shape)
    expected = scatter_update_golden(inputs_cpu, config)
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    KERNELS[(len(config.execution_view_shape), len(inputs))](*inputs, config)
    assert_outputs([inputs[2]], expected)


@pytest.mark.parametrize(
    "case", SCATTERUPDATE_ONBOARD_TESTS, ids=[case["case_name"] for case in SCATTERUPDATE_ONBOARD_TESTS]
)
def test_migrated_scatterupdate(case: dict):
    run_scatterupdate_onboard_test(case)
