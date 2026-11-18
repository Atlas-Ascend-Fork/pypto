#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for DivMod vector ST."""

from dataclasses import dataclass

import torch

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class DivModConfig(VectorConfig):
    OPERATION = 'DivMod'


def _tensor(name, shape, data_range=None):
    tensor = {
        'name': name,
        'shape': tuple(shape),
        'dtype': 'int32',
        'format': 'ND',
        'need_trans': False,
    }
    if data_range is not None:
        tensor['data_range'] = {'min': data_range[0], 'max': data_range[1]}
    return tensor


def _case(case_index, case_name, dtype, a_shape, b_shape, a_range, b_range, view_shape, tile_shape):
    out_shape = tuple(torch.broadcast_shapes(tuple(a_shape), tuple(b_shape)))
    inputs = [_tensor('input0', a_shape, a_range), _tensor('input1', b_shape, b_range)]
    outputs = [_tensor('output0', out_shape), _tensor('output1', out_shape)]
    for tensor in inputs + outputs:
        tensor['dtype'] = dtype
    return {
        'case_index': case_index,
        'case_name': case_name,
        'operation': 'DivMod',
        'input_tensors': inputs,
        'output_tensors': outputs,
        'view_shape': tuple(view_shape),
        'tile_shape': tuple(tile_shape),
        'params': {},
        'source_case_index': case_index,
    }


# More sample types than FloorDiv: positive/negative/mixed signs, int8/uint8/int16/int32,
# broadcast row/column/scalar, and a larger tiled shape.
DIVMOD_TESTS = [
    _case(0, 'DivMod_int32_positive', 'int32', (128, 128), (128, 128), (1, 1000), (1, 1000), (64, 64), (64, 64)),
    _case(1, 'DivMod_int32_negative_dividend', 'int32', (128, 128), (128, 128), (-1000, -1), (1, 1000),
          (64, 64), (64, 64)),
    _case(2, 'DivMod_int32_negative_divisor', 'int32', (128, 128), (128, 128), (1, 1000), (-1000, -1),
          (64, 64), (64, 64)),
    _case(3, 'DivMod_int32_mixed', 'int32', (128, 128), (128, 128), (-1000, 1000), (-1000, -1),
          (64, 64), (64, 64)),
    _case(4, 'DivMod_int8_mixed', 'int8', (128, 128), (128, 128), (-128, 127), (-128, -1),
          (64, 64), (64, 64)),
    _case(5, 'DivMod_uint8_positive', 'uint8', (128, 128), (128, 128), (0, 255), (1, 255),
          (64, 64), (64, 64)),
    _case(6, 'DivMod_int16_mixed', 'int16', (128, 128), (128, 128), (-30000, 30000), (1, 1000),
          (64, 64), (64, 64)),
    _case(7, 'DivMod_broadcast_row', 'int32', (1, 128), (128, 128), (1, 1000), (1, 1000),
          (64, 64), (64, 64)),
    _case(8, 'DivMod_broadcast_col', 'int32', (128, 1), (128, 128), (1, 1000), (1, 1000),
          (64, 64), (64, 64)),
    _case(9, 'DivMod_broadcast_scalar', 'int32', (1, 1), (64, 64), (1, 1000), (1, 1000),
          (64, 64), (64, 64)),
    _case(10, 'DivMod_int32_tiled', 'int32', (256, 128), (256, 128), (-1000, 1000), (1, 1000),
          (128, 64), (32, 64)),
]
