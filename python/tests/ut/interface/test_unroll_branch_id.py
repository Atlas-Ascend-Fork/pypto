#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unroll branch context lifecycle tests.

The frontend stamps every op recorded inside an unrolled-loop branch copy with a fresh
unroll branch id (pypto_impl.SetUnrollBranchId -> Function::AddRawOperation), so
VFFusionClusterIdentify can refuse fusing ops from different branches into one VF cluster.
These tests verify the thread-local context lifecycle: fresh id per loop-body dispatch,
restore of the enclosing id on loop exit, and nesting.
"""

import pytest

import pypto
from pypto import pypto_impl


def _tensors():
    dtype = pypto.DT_FP16
    shape = (8, 8)
    return (pypto.tensor(shape, dtype, "tensor_a"), pypto.tensor(shape, dtype, "tensor_b"))


@pytest.fixture(autouse=True)
def _clear_unroll_branch_id():
    pypto_impl.SetUnrollBranchId(-1)
    yield
    pypto_impl.SetUnrollBranchId(-1)


def test_unroll_branch_id_roundtrip():
    assert pypto_impl.GetUnrollBranchId() == -1
    pypto_impl.SetUnrollBranchId(5)
    assert pypto_impl.GetUnrollBranchId() == 5
    pypto_impl.SetUnrollBranchId(-1)
    assert pypto_impl.GetUnrollBranchId() == -1


def test_loop_body_gets_fresh_band_and_restores_enclosing_id():
    a, b = _tensors()
    with pypto.function("UNROLL_BAND_LOOP", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        pypto_impl.SetUnrollBranchId(9)
        body_bands = []
        for _ in pypto.loop(0, 4, 1, name="LOOP_BAND", idx_name="band_k"):
            # Ops recorded here are stamped with the fresh band assigned per body dispatch.
            body_bands.append(pypto_impl.GetUnrollBranchId())
            _ = pypto.add(a, b)
        # Loop exit restores the enclosing id: post-loop ops belong to the outer context.
        assert pypto_impl.GetUnrollBranchId() == 9
        assert len(body_bands) > 0
        assert all(band >= 0 for band in body_bands)


def test_nested_loop_restores_outer_band():
    a, b = _tensors()
    with pypto.function("UNROLL_BAND_NESTED", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        for _ in pypto.loop(0, 2, 1, name="LOOP_BAND_OUTER", idx_name="outer_k"):
            outer_band = pypto_impl.GetUnrollBranchId()
            assert outer_band >= 0
            for _ in pypto.loop(0, 2, 1, name="LOOP_BAND_INNER", idx_name="inner_k"):
                inner_band = pypto_impl.GetUnrollBranchId()
                # Inner body ops get their own fresh band, distinct from the outer one.
                assert inner_band >= 0
                assert inner_band != outer_band
                _ = pypto.add(a, b)
            # Inner loop exit restores the outer loop's band.
            assert pypto_impl.GetUnrollBranchId() == outer_band
        # Outer loop exit restores the enclosing (wildcard) context.
        assert pypto_impl.GetUnrollBranchId() == -1


def test_loop_unroll_assigns_distinct_band_per_dispatch():
    a, b = _tensors()
    with pypto.function("UNROLL_BAND_FACTORS", a, b):
        pypto.set_vec_tile_shapes(8, 8)
        seen = []
        for _, factor in pypto.loop_unroll(0, 5, 1, name="LOOP_BAND_UNROLL", idx_name="lu_k", unroll_list=[2, 1]):
            seen.append((factor, pypto_impl.GetUnrollBranchId()))
            _ = pypto.add(a, b)
        assert pypto_impl.GetUnrollBranchId() == -1
        # Every dispatched body copy carries a fresh non-negative band, all distinct.
        assert seen
        assert all(band >= 0 for _, band in seen)
        bands = [band for _, band in seen]
        assert len(bands) == len(set(bands))
        assert set(factor for factor, _ in seen) == {2, 1}
