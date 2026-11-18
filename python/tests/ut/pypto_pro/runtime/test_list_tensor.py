#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the host-side list-tensor descriptor builder."""

from pypto_pro._errors import InvalidArgument, InvalidShape
from pypto_pro.runtime import make_list_tensor_desc
import pytest
import torch


def test_dim0_form_matches_layout():
    desc = make_list_tensor_desc([0x1000, 0x2000, 0x3000])

    words = desc.tolist()
    assert words == [24, 3 << 32, 0xFFFFFFFF, 0x1000, 0x2000, 0x3000]
    assert desc.dtype == torch.int64


def test_dim1_form_matches_layout():
    addrs = [0xA000, 0xB000]

    desc = make_list_tensor_desc(addrs, shape=[11])

    # [offset, (count<<32)|dim, struct_0 meta+shape, struct_1 meta+shape, addrs]
    assert desc.tolist() == [8 + 2 * 2 * 8, (2 << 32) | 1, 11, 0, 11, 0xA000, 0xB000]


def test_dim2_form_places_shape_structs_and_address_array():
    addrs = [0x10, 0x20]

    desc = make_list_tensor_desc(addrs, shape=[3, 5])

    struct_size = 3  # 1 meta + dim=2 shape words
    offset = 8 + 2 * struct_size * 8
    assert desc.tolist() == [
        offset,
        (2 << 32) | 2,  # struct 0: meta carries dim and count
        3,
        5,
        0,  # struct 1 meta stays 0
        3,
        5,
        0x10,
        0x20,
    ]


def test_shared_shape_is_written_for_every_tensor():
    desc = make_list_tensor_desc([1, 2, 3, 4], shape=[7])

    # Every struct i has shape[0] at word 1 + i*2 + 1.
    for i in range(4):
        assert desc[1 + i * 2 + 1].item() == 7


def test_per_tensor_shapes_match_layout():
    desc = make_list_tensor_desc([0xA000, 0xB000], shape=[[33, 34, 35], [23, 24, 25]])

    # [offset, (count<<32)|dim, struct_0 meta+shape, struct_1 meta(0)+shape, addrs]
    assert desc.tolist() == [8 + 2 * 4 * 8, (2 << 32) | 3, 33, 34, 35, 0, 23, 24, 25, 0xA000, 0xB000]


def test_per_tensor_shapes_single_tensor():
    desc = make_list_tensor_desc([0x1000], shape=[[3, 5]])

    assert desc.tolist() == [8 + 1 * 3 * 8, (1 << 32) | 2, 3, 5, 0x1000]


def test_per_tensor_count_mismatch_is_rejected():
    with pytest.raises(InvalidArgument):
        make_list_tensor_desc([0x1000, 0x2000], shape=[[3, 5]])


def test_per_tensor_dim_mismatch_is_rejected():
    with pytest.raises(InvalidShape):
        make_list_tensor_desc([0x1000, 0x2000], shape=[[3, 5], [2]])


def test_mixed_flat_and_nested_shape_is_rejected():
    with pytest.raises(InvalidShape):
        make_list_tensor_desc([0x1000], shape=[3, [5]])


def test_empty_addrs_is_rejected():
    with pytest.raises(InvalidArgument):
        make_list_tensor_desc([])


def test_empty_shape_is_rejected():
    with pytest.raises(InvalidShape):
        make_list_tensor_desc([0x1000], shape=[])


@pytest.mark.parametrize("bad", [0, -3, True, 2.5, "8"])
def test_non_positive_shape_axis_is_rejected(bad):
    with pytest.raises(InvalidShape):
        make_list_tensor_desc([0x1000], shape=[bad])
