#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for CumProd vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class CumprodConfig(VectorConfig):
    OPERATION = 'CumProd'


CUMPROD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'CumProd_test_3',
        'operation': 'CumProd',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (256, 7168),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -2, 'max': 2},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (256, 7168), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (75, 2534),
        'tile_shape': (32, 512),
        'params': {'axis': -2},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'CumProd_test_4',
        'operation': 'CumProd',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 10),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 10), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (2, 2),
        'tile_shape': (3, 4),
        'params': {'axis': 1},
        'source_case_index': 1,
    },
]
