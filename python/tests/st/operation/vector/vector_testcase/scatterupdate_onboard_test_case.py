#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for ScatterUpdate vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ScatterupdateOnboardConfig(VectorConfig):
    OPERATION = 'ScatterUpdate'


SCATTERUPDATE_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'ScatterUpdate_test_1',
        'operation': 'ScatterUpdate',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1, 1, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (1, 1),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 2},
            },
            {
                'name': 'input2',
                'shape': (2, 1, 1, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 0},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2, 1, 1, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 1, 1, 64),
        'tile_shape': (1, 1, 1, 64),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'ScatterUpdate_test_2',
        'operation': 'ScatterUpdate',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 8),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (5, 2),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 15},
            },
            {
                'name': 'input2',
                'shape': (15, 8),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 0},
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (15, 8), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (4, 8),
        'tile_shape': (2, 8),
        'params': {},
        'source_case_index': 1,
    },
    {
        'case_index': 2,
        'case_name': 'ScatterUpdate_test_3',
        'operation': 'ScatterUpdate',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1, 1, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (1, 1),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 2},
            },
            {
                'name': 'input2',
                'shape': (2, 1, 1, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 0},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2, 1, 1, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 1, 1, 64),
        'tile_shape': (1, 1, 1, 32),
        'params': {},
        'source_case_index': 0,
    },
]
