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
"""Block-facing smoke tests for DataType exports.

Full DataType behavior is covered by the C++ IR tests and the generic Python IR
binding tests. The block suite only checks the pypto_pro entry points that the
Block DSL depends on.
"""

import pypto_pro
from pypto_pro import DataType, ir
from pypto_pro.ir import _limits
import pypto_pro.language as pl
import pytest


def test_pypto_pro_reexports_common_block_dtypes():
    assert pypto_pro.DT_FP16 == DataType.FP16
    assert pypto_pro.DT_FP32 == DataType.FP32
    assert pypto_pro.DT_INT32 == DataType.INT32
    assert pypto_pro.DT_BOOL == DataType.BOOL


def test_language_dtype_aliases_are_usable_in_block_annotations():
    tensor_type = pl.Tensor[[16, 32], pl.DT_FP16]

    assert isinstance(tensor_type, pl.Tensor)
    assert tensor_type.dtype == DataType.FP16
    assert tensor_type.shape == [16, 32]


def test_int64_dtype_is_available_for_block_shape_symbols():
    shape_var = ir.Var("m", ir.ScalarType(DataType.INT64), ir.Span.unknown())
    tensor_type = ir.TensorType([shape_var], DataType.FP16)

    assert tensor_type.shape[0] is shape_var
    assert shape_var.type.dtype == DataType.INT64
    assert DataType.INT64.to_string() == "int64"


# ---------------------------------------------------------------------------
# Numeric limits (pypto_pro.ir._limits)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dtype", "lo", "hi"),
    [
        (DataType.INT4, -8, 7),
        (DataType.INT8, -128, 127),
        (DataType.INT16, -32768, 32767),
        (DataType.INT32, -(2**31), 2**31 - 1),
        (DataType.INT64, -(2**63), 2**63 - 1),
        (DataType.UINT4, 0, 15),
        (DataType.UINT8, 0, 255),
        (DataType.UINT16, 0, 65535),
        (DataType.UINT32, 0, 2**32 - 1),
        (DataType.UINT64, 0, 2**64 - 1),
    ],
)
def test_limits_of_integer_dtypes(dtype, lo, hi):
    assert _limits.limits_of(dtype) == _limits.NumericLimits(lo, hi, False)


def test_limits_of_bool_is_zero_to_one_not_eight_bits():
    """BOOL occupies a byte, so get_bit() reports 8; its value range is still [0, 1]."""
    assert DataType.BOOL.get_bit() == 8
    assert _limits.limits_of(DataType.BOOL) == _limits.NumericLimits(0, 1, False)


@pytest.mark.parametrize(
    ("dtype", "max_finite"),
    [
        (DataType.FP64, 1.7976931348623157e308),
        (DataType.FP32, 3.4028234663852886e38),
        (DataType.BF16, 3.3895313892515355e38),
        (DataType.FP16, 65504.0),
        (DataType.FP8E5M2, 57344.0),
        (DataType.FP8E4M3FN, 448.0),
        (DataType.FP4E2M1, 6.0),
    ],
)
def test_limits_of_signed_float_dtypes(dtype, max_finite):
    assert _limits.limits_of(dtype) == _limits.NumericLimits(-max_finite, max_finite, True)


def test_limits_of_fp8e8m0_is_unsigned():
    """E8M0 is a bare exponent: it carries no sign bit."""
    limits = _limits.limits_of(DataType.FP8E8M0)

    assert limits.lo == 0.0
    assert limits.hi == float(2**127)


@pytest.mark.parametrize("dtype", [DataType.FP4, DataType.FP4E1M2, DataType.HF4, DataType.HF8])
def test_limits_of_unspecified_float_formats_is_none(dtype):
    """These encodings are not specified in-tree, so callers skip the check rather than guess."""
    assert _limits.limits_of(dtype) is None


@pytest.mark.parametrize("dtype", [DataType.FP4, DataType.HF8])
def test_fits_skips_dtypes_without_limits(dtype):
    ok, reason, limits = _limits.fits(1e300, dtype)

    assert ok
    assert reason == ""
    assert limits is None


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (127, DataType.INT8, True),
        (-128, DataType.INT8, True),
        (128, DataType.INT8, False),
        (-129, DataType.INT8, False),
        (255, DataType.UINT8, True),
        (-1, DataType.UINT8, False),
        (65504.0, DataType.FP16, True),
        (70000.0, DataType.FP16, False),
        (1e300, DataType.FP32, False),
        (2**200, DataType.FP32, False),
        (1.5, DataType.INT32, True),  # a fraction is dropped by the backend, not rejected
        (True, DataType.BOOL, True),
        (2, DataType.BOOL, False),
    ],
)
def test_fits(value, dtype, expected):
    ok, reason, _ = _limits.fits(value, dtype)

    assert ok is expected
    # The reason is the diagnostic's hint, so it must be present exactly when the value is rejected.
    assert (reason == "") is expected


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_fits_passes_non_finite_floats_for_a_float_dtype(value):
    """inf / nan are legitimate reduction seeds and the backend emits them explicitly."""
    assert _limits.fits(value, DataType.FP16)[0]


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("dtype", [DataType.INT8, DataType.INT32, DataType.UINT64])
def test_fits_rejects_non_finite_floats_for_an_integer_dtype(value, dtype):
    """The backend lowers a float scalar via static_cast<int64_t>, undefined for inf / nan."""
    ok, reason, _ = _limits.fits(value, dtype)

    assert not ok
    assert str(dtype) in reason


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, True),
        (2**63 - 1, True),
        (2**63, True),
        (2**64 - 1, True),
        (2**64, False),
        (-(2**63), True),
        (-(2**63) - 1, False),
    ],
)
def test_fits_storage_int_is_the_int64_uint64_union_band(value, expected):
    assert _limits.fits_storage_int(value) is expected


@pytest.mark.parametrize(
    ("value", "storage"),
    [(0, 0), (-1, -1), (2**63 - 1, 2**63 - 1), (2**63, -(2**63)), (2**64 - 1, -1)],
)
def test_to_storage_int_folds_above_int64_max(value, storage):
    assert _limits.to_storage_int(value) == storage


@pytest.mark.parametrize(
    ("storage", "dtype", "expected"),
    [
        (-1, DataType.UINT64, 2**64 - 1),
        (-(2**63), DataType.UINT64, 2**63),
        (-1, DataType.INT64, -1),
        (255, DataType.UINT8, 255),
    ],
)
def test_from_storage_int_recovers_the_logical_value(storage, dtype, expected):
    assert _limits.from_storage_int(storage, dtype) == expected
