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

"""Numeric limits of the IR data types.

Pure data and predicates over :class:`DataType`; no dependency on the parser or on any third-party
package. Two bands matter:

* the *storage* band ``[INT64_MIN, UINT64_MAX]`` that ``ir.ConstInt`` can carry -- its ``value_`` field
  is an ``int64_t`` that doubles as the bit image of a ``uint64_t``;
* the *representable* band of a concrete dtype, used to reject scalar operands that an operator could
  not encode.
"""

from __future__ import annotations

__all__ = [
    "INT32_MAX",
    "INT64_MAX",
    "INT64_MIN",
    "UINT64_MAX",
    "NumericLimits",
    "fits",
    "fits_storage_int",
    "from_storage_int",
    "limits_of",
    "to_storage_int",
]


import math
from typing import NamedTuple

from pypto.pypto_impl.ir import DataType

INT32_MAX = 2 ** 31 - 1
INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1
UINT64_MAX = 2 ** 64 - 1

_UINT64_MODULUS = 2 ** 64


class NumericLimits(NamedTuple):
    """Inclusive ``[lo, hi]`` range of the finite values a dtype can represent."""

    lo: float
    hi: float
    is_float: bool


# Largest finite magnitude of every float format whose encoding is specified. Each entry is derived
# from the exponent/mantissa widths and cross-checked against the constants already used by the
# backend (FP16_MAX in vector/indexing.cpp, 448.0f / 57344.0f in fp_convert.cpp, 6.0f in calc_torch.cpp).
#
# fp4, fp4e1m2, hf4 and hf8 are deliberately absent: their encodings are not specified anywhere in the
# tree, and guessing one would produce wrong rejections. `limits_of` returns None for them so callers
# skip the check rather than assume an unbounded range.
_FLOAT_MAX_FINITE: dict[str, float] = {
    "fp64": 1.7976931348623157e308,
    "fp32": 3.4028234663852886e38,
    "bfloat16": 3.3895313892515355e38,
    "fp16": 65504.0,
    "fp8e5m2": 57344.0,
    "fp8e4m3fn": 448.0,
    "fp8e8m0": float(2 ** 127),
    "fp4e2m1": 6.0,
}

# fp8e8m0 encodes a bare exponent, so it carries no sign bit and no negative value.
_UNSIGNED_FLOATS = frozenset({"fp8e8m0"})


def limits_of(dtype: DataType) -> NumericLimits | None:
    """Inclusive value range of *dtype*, or ``None`` when the format is not specified in-tree.

    ``None`` means "unknown", not "unbounded": callers skip the check for such a dtype.
    """
    if dtype == DataType.BOOL:
        # BOOL is stored in a byte, so get_bit() reports 8; the value range is still just [0, 1].
        return NumericLimits(0, 1, False)
    if dtype.is_float():
        max_finite = _FLOAT_MAX_FINITE.get(str(dtype))
        if max_finite is None:
            return None
        lo = 0.0 if str(dtype) in _UNSIGNED_FLOATS else -max_finite
        return NumericLimits(lo, max_finite, True)
    bits = dtype.get_bit()
    if dtype.is_signed_int():
        # INDEX is a 64-bit signed placeholder, and reports itself as a signed int.
        return NumericLimits(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1, False)
    if dtype.is_unsigned_int():
        return NumericLimits(0, 2 ** bits - 1, False)
    return None


def fits(value: int | float, dtype: DataType) -> tuple[bool, str, NumericLimits | None]:
    """Whether *value* is representable in *dtype*.

    Returns ``(ok, reason, limits)``. *reason* is empty when *ok*, and *limits* is ``None`` for a dtype
    whose encoding is not specified in-tree -- such a dtype has no bounds to check, so it always fits.

    Non-finite floats (``inf`` / ``nan``) pass for a float dtype, where they are legitimate reduction
    seeds the backend emits explicitly; against an integer dtype they are rejected, because the backend
    lowers a float scalar through ``static_cast<int64_t>``, which is undefined for them. A fractional
    value against an integer dtype is only range-checked, never rejected for having a fraction -- the
    backend truncates it.
    """
    limits = limits_of(dtype)
    if limits is None:
        return True, "", limits
    if isinstance(value, float) and not math.isfinite(value):
        if dtype.is_float():
            return True, "", limits
        else:
            return False, f"{value} value does not match integer dtype {dtype}", limits
    if limits.lo <= value <= limits.hi:
        return True, "", limits
    else:
        return False, f"{value} is out of range of current dtype {dtype}", limits


def fits_storage_int(value: int) -> bool:
    """Whether *value* fits the ``[INT64_MIN, UINT64_MAX]`` band that ``ir.ConstInt`` can carry."""
    return INT64_MIN <= value <= UINT64_MAX


def to_storage_int(value: int) -> int:
    """Fold *value* into the ``int64_t`` field of ``ir.ConstInt``.

    Values above ``INT64_MAX`` are stored as their two's-complement image and rendered back according
    to the dtype's signedness by the code generators.
    """
    if value > INT64_MAX:
        return value - _UINT64_MODULUS
    return value


def from_storage_int(value: int, dtype: DataType) -> int:
    """Recover the logical value of a constant stored by :func:`to_storage_int`.

    Only a 64-bit unsigned constant can hold a folded (negative) image: every narrower unsigned dtype
    stores its value directly, so the un-fold is a no-op there.
    """
    if value < 0 and dtype.is_unsigned_int():
        return value + _UINT64_MODULUS
    return value
