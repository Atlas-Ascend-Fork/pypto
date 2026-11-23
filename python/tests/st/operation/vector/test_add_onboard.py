#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""System tests for pypto.add migrated from the C++ vector ST."""

import os

import pytest
import torch
from vector_test_utils import assert_outputs, make_inputs, make_outputs
from vector_testcase.add_test_case import ADD_TESTS, AddConfig

import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def add_2d_kernel(a: pypto.Tensor(), b: pypto.Tensor(), output: pypto.Tensor(), config: AddConfig):
    row_view, column_view = config.view_shape
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for row_index in pypto.loop((config.output_shape[0] + row_view - 1) // row_view):
        for column_index in pypto.loop((config.output_shape[1] + column_view - 1) // column_view):
            row_offset = row_index * row_view
            column_offset = column_index * column_view
            a_offset = [0 if config.a_shape[0] == 1 else row_offset, 0 if config.a_shape[1] == 1 else column_offset]
            b_offset = [0 if config.b_shape[0] == 1 else row_offset, 0 if config.b_shape[1] == 1 else column_offset]
            a_shape = [1 if config.a_shape[0] == 1 else row_view, 1 if config.a_shape[1] == 1 else column_view]
            b_shape = [1 if config.b_shape[0] == 1 else row_view, 1 if config.b_shape[1] == 1 else column_view]
            result = pypto.add(pypto.view(a, a_shape, a_offset), pypto.view(b, b_shape, b_offset))
            pypto.assemble(result, [row_offset, column_offset], output)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def add_4d_kernel(a: pypto.Tensor(), b: pypto.Tensor(), output: pypto.Tensor(), config: AddConfig):
    view_0, view_1, view_2, view_3 = config.view_shape
    pypto.set_vec_tile_shapes(*config.tile_shape)
    for index_0 in pypto.loop((config.output_shape[0] + view_0 - 1) // view_0):
        for index_1 in pypto.loop((config.output_shape[1] + view_1 - 1) // view_1):
            for index_2 in pypto.loop((config.output_shape[2] + view_2 - 1) // view_2):
                for index_3 in pypto.loop((config.output_shape[3] + view_3 - 1) // view_3):
                    offset = [index_0 * view_0, index_1 * view_1, index_2 * view_2, index_3 * view_3]
                    a_offset = [0 if config.a_shape[index] == 1 else offset[index] for index in range(4)]
                    b_offset = [0 if config.b_shape[index] == 1 else offset[index] for index in range(4)]
                    a_shape = [1 if config.a_shape[index] == 1 else config.view_shape[index] for index in range(4)]
                    b_shape = [1 if config.b_shape[index] == 1 else config.view_shape[index] for index in range(4)]
                    result = pypto.add(pypto.view(a, a_shape, a_offset), pypto.view(b, b_shape, b_offset))
                    pypto.assemble(result, offset, output)


def run_add_test(case: dict):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    config = AddConfig.from_test_case(case)
    inputs_cpu = make_inputs(config)
    expected = [torch.add(*inputs_cpu)]
    inputs = [tensor.to(f"npu:{device_id}") for tensor in inputs_cpu]
    outputs = make_outputs(config, f"npu:{device_id}")
    kernel = add_2d_kernel if len(config.output_shape) == 2 else add_4d_kernel
    kernel(*inputs, *outputs, config)
    assert_outputs(outputs, expected)


@pytest.mark.parametrize("case", ADD_TESTS, ids=[case["case_name"] for case in ADD_TESTS])
def test_add(case: dict):
    run_add_test(case)
