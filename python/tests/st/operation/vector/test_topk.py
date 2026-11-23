#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for TopK migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.topk_test_case import TOPK_TESTS, TopkConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def topk_2d_1input_2d_output_kernel(
    input0: pypto.Tensor(), output0: pypto.Tensor(), output1: pypto.Tensor(), config: TopkConfig
):
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop(config.loop_ranges[0]):
        for index_1 in pypto.loop(config.loop_ranges[1]):
            offsets = [index_0 * config.execution_view_shape[0], index_1 * config.execution_view_shape[1]]
            input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(2)]
            input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
            results = pypto.topk(input0_view, config.count[0], config.dims[0], config.is_largest[0])
            output_offset = [
                0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                for axis in range(2)
            ]
            pypto.assemble(results[0], output_offset, output0)
            pypto.assemble(results[1], output_offset, output1)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def topk_3d_1input_3d_output_kernel(
    input0: pypto.Tensor(), output0: pypto.Tensor(), output1: pypto.Tensor(), config: TopkConfig
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
                input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(3)]
                input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
                results = pypto.topk(input0_view, config.count[0], config.dims[0], config.is_largest[0])
                output_offset = [
                    0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                    for axis in range(3)
                ]
                pypto.assemble(results[0], output_offset, output0)
                pypto.assemble(results[1], output_offset, output1)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def topk_4d_1input_4d_output_kernel(
    input0: pypto.Tensor(), output0: pypto.Tensor(), output1: pypto.Tensor(), config: TopkConfig
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
                    input0_offset = [0 if config.input_shapes[0][axis] == 1 else offsets[axis] for axis in range(4)]
                    input0_view = pypto.view(input0, config.input_view_shapes[0], input0_offset)
                    results = pypto.topk(input0_view, config.count[0], config.dims[0], config.is_largest[0])
                    output_offset = [
                        0 if config.output_offset_map[axis] < 0 else offsets[config.output_offset_map[axis]]
                        for axis in range(4)
                    ]
                    pypto.assemble(results[0], output_offset, output0)
                    pypto.assemble(results[1], output_offset, output1)


KERNELS = {
    (2, 1, 2, 2): topk_2d_1input_2d_output_kernel,
    (3, 1, 2, 3): topk_3d_1input_3d_output_kernel,
    (4, 1, 2, 4): topk_4d_1input_4d_output_kernel,
}


def run_topk_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = TopkConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = list(torch.topk(inputs_cpu[0], config.count[0], config.dims[0], config.is_largest[0]))
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    KERNELS[(len(config.execution_view_shape), len(inputs), len(outputs), len(config.output_shapes[0]))](
        *inputs, *outputs, config
    )
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", TOPK_TESTS, ids=[case["case_name"] for case in TOPK_TESTS])
def test_topk(case: dict):
    run_topk_test(case)
