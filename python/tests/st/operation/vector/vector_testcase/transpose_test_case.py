#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Transpose vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class TransposeConfig(VectorConfig):
    OPERATION = 'Transpose'


TRANSPOSE_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Transpose_test_0',
        'operation': 'Transpose',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32, 512),
                'dtype': 'fp32',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32, 512), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32, 512),
        'tile_shape': (2, 2, 512),
        'params': {'dims': [1, 0, 2], 'first_dim': 0, 'second_dim': 1},
        'source_case_index': 0,
    },
    {
        'case_index': 2,
        'case_name': 'Transpose_test_2',
        'operation': 'Transpose',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 32, 32, 32),
                'dtype': 'fp32',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (2, 32, 32, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 32, 32, 32),
        'tile_shape': (1, 16, 16, 16),
        'params': {'dims': [0, 1, 3, 2], 'first_dim': 2, 'second_dim': 3},
        'source_case_index': 2,
    },
    {
        'case_index': 13,
        'case_name': 'Transpose_test_13',
        'operation': 'Transpose',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 3, 4, 16),
                'dtype': 'fp32',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 3, 2, 16), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 3, 4, 16),
        'tile_shape': (2, 3, 4, 16),
        'params': {'dims': [2, 1, 0, 3], 'first_dim': 0, 'second_dim': 2},
        'source_case_index': 13,
    },
    {
        'case_index': 15,
        'case_name': 'Transpose_test_15',
        'operation': 'Transpose',
        'input_tensors': [
            {'name': 'input0', 'shape': (128, 128), 'dtype': 'fp16', 'data_range': {'min': 0, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (128, 128), 'dtype': 'fp16', 'format': 'ND'}],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'dims': [1, 0], 'first_dim': 0, 'second_dim': 1},
        'source_case_index': 15,
    },
    {
        'case_index': 18,
        'case_name': 'Transpose_test_18',
        'operation': 'Transpose',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 3, 4, 16),
                'dtype': 'fp16',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (2, 16, 4, 3), 'dtype': 'fp16', 'format': 'ND'}],
        'view_shape': (2, 3, 4, 16),
        'tile_shape': (2, 3, 4, 16),
        'params': {'dims': [0, 3, 2, 1], 'first_dim': 1, 'second_dim': 3},
        'source_case_index': 18,
    },
    {
        'case_index': 31,
        'case_name': 'Transpose_test_31',
        'operation': 'Transpose',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 3, 1, 16),
                'dtype': 'fp32',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (1, 3, 2, 16), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 3, 1, 16),
        'tile_shape': (1, 3, 1, 16),
        'params': {'dims': [2, 1, 0, 3], 'first_dim': 0, 'second_dim': 2},
        'source_case_index': 31,
    },
    {
        'case_index': 32,
        'case_name': 'Transpose_test_32',
        'operation': 'Transpose',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 3, 1, 16),
                'dtype': 'fp32',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (2, 3, 16, 1), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 3, 1, 16),
        'tile_shape': (2, 3, 1, 8),
        'params': {'dims': [0, 1, 3, 2], 'first_dim': 2, 'second_dim': 3},
        'source_case_index': 32,
    },
]
