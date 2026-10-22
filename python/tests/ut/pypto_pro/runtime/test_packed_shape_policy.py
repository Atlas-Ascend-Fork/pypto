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
"""Host storage shapes are normalized once; explicit static shapes are already logical."""

from types import SimpleNamespace

from pypto_pro._errors import InvalidShape, InvalidType
import pypto_pro.language as pl
from pypto_pro.runtime.shape_policy import KernelSignatureSpec
import pytest


def _signature(dtype, shape):
    def kernel(x: pl.Tensor[shape, dtype]):
        pass
    return KernelSignatureSpec.from_callable(kernel)


@pytest.mark.parametrize("dtype", [pl.DT_FP4E2M1, pl.DT_FP4E1M2])
@pytest.mark.parametrize("shape", [[1, 64], [pl.DYNAMIC, 64], [1, pl.STATIC], [1, ...]])
def test_runtime_shapes_count_logical_elements(dtype, shape):
    bound = _signature(dtype, shape).bind_runtime_args((SimpleNamespace(shape=(1, 32)),))
    assert bound.tensor("x").dimensions[-1].value == 64


def test_explicit_static_shapes_are_not_unpacked_twice():
    signature = _signature(pl.DT_FP4E2M1, [1, pl.STATIC])
    assert signature.bind_static_shapes({"x": [1, 64]}).static_signature == ((0, 1, 64),)


@pytest.mark.parametrize("shape,error", [((1, 0), InvalidShape), ((1, -1), InvalidShape),
                                        ((1, True), InvalidType), ((32,), InvalidType)])
def test_invalid_carrier_shapes_remain_rejected(shape, error):
    with pytest.raises(error):
        _signature(pl.DT_FP4E2M1, [1, 64]).bind_runtime_args((SimpleNamespace(shape=shape),))


def test_byte_wide_types_and_other_axes_remain_unchanged():
    for dtype in (pl.DT_FP32, pl.DT_INT8):
        bound = _signature(dtype, [7, 32]).bind_runtime_args((SimpleNamespace(shape=(7, 32)),))
        assert [d.value for d in bound.tensor("x").dimensions] == [7, 32]
    bound = _signature(pl.DT_FP4E2M1, [7, 64]).bind_runtime_args((SimpleNamespace(shape=(7, 32)),))
    assert [d.value for d in bound.tensor("x").dimensions] == [7, 64]


def test_dynamic_packed_dimensions_keep_dynamic_policy():
    bound = _signature(pl.DT_FP4E2M1, [pl.DYNAMIC, pl.DYNAMIC]).bind_runtime_args(
        (SimpleNamespace(shape=(7, 32)),))
    assert all(dim.dynamic_name for dim in bound.tensor("x").dimensions)
