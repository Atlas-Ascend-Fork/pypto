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

"""Range checks for numeric values on their way into the IR.

One message grammar -- ``<subject> must be in [lo, hi], got <value>`` -- for the three kinds of bound
the front end enforces: the storage band of ``ir.ConstInt``, the representable range of a dtype, and
plain explicit intervals such as an event id.

All checks raise :class:`OutOfRange`. ``ExpressionParserMixin.parse_expression`` recovers from a
failed IR build by re-evaluating the expression in Python, but only for a rejection whose site opted
in with ``parser_retry``; these checks do not, so a value they refuse stays refused.
"""

from __future__ import annotations

__all__ = [
    "check_const_expr_fits_dtype",
    "make_const_int",
    "check_fits_dtype",
    "check_in_range",
    "check_ir_int",
    "range_message",
]


from pypto.pypto_impl import ir
from pypto_pro.ir._limits import (
    INT64_MAX,
    INT64_MIN,
    UINT64_MAX,
    fits,
    fits_storage_int,
    from_storage_int,
    to_storage_int,
)

from ...._errors import OutOfRange, current_span


def _format(value: int | float) -> str:
    """Render *value* so it round-trips: full precision for floats, plain digits for integers."""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def range_message(
    value: int | float,
    lo: int | float,
    hi: int | float,
    *,
    subject: str,
    api: str | None = None,
    dtype: ir.DataType | None = None,
) -> str:
    """Build the shared out-of-range message."""
    prefix = f"{api}: " if api else ""
    bound = f"in [{_format(lo)}, {_format(hi)}]"
    if dtype is not None:
        bound = f"representable in {dtype}, i.e. {bound}"
    return f"{prefix}{subject} must be {bound}, got {_format(value)}"


def check_in_range(
    value: int | float,
    lo: int | float,
    hi: int | float,
    *,
    subject: str,
    api: str | None = None,
    hint: str | None = None,
    error: type[Exception] | None = None,
) -> None:
    """Reject *value* unless ``lo <= value <= hi``. Both bounds are inclusive.

    Pass *error* to raise a plain exception class instead -- builder-level checks in ``ir.op`` report a
    ``ValueError``, since they are not reached through the parser's retry path.
    """
    if lo <= value <= hi:
        return
    message = range_message(value, lo, hi, subject=subject, api=api)
    if error is not None:
        raise error(message)
    raise OutOfRange(message, hint=hint)


def check_fits_dtype(
    value: int | float,
    dtype: ir.DataType,
    *,
    subject: str = "scalar operand",
    api: str | None = None,
) -> None:
    """Reject *value* unless *dtype* can represent it.

    A dtype whose encoding is not specified in-tree has no limits, and the check is skipped rather than
    guessed at. ``inf`` and ``nan`` pass for a float dtype but not for an integer one; see
    :func:`pypto_pro.ir._limits.fits`.
    """
    result, hint, limits = fits(value, dtype)
    if result:
        return
    raise OutOfRange(
        range_message(value, limits.lo, limits.hi, subject=subject, api=api, dtype=dtype),
        hint=hint,
    )


def check_ir_int(value: int, *, subject: str = "integer constant") -> int:
    """Reject an integer the IR cannot carry, and return its storage image.

    ``ir.ConstInt`` holds an ``int64_t`` that doubles as the bit pattern of a ``uint64_t``, so the band
    it can carry is the union ``[INT64_MIN, UINT64_MAX]``. Arbitrary precision stays legal *before* this
    point: the parser evaluates intermediates in Python, and only materialising a constant is bounded.
    """
    if not fits_storage_int(value):
        raise OutOfRange(
            range_message(value, INT64_MIN, UINT64_MAX, subject=subject),
            hint="a constant must be representable as either int64 or uint64",
        )
    return to_storage_int(value)


def check_const_expr_fits_dtype(
    expr: ir.Expr,
    dtype: ir.DataType | None,
    *,
    subject: str = "scalar operand",
    api: str | None = None,
) -> None:
    """Check a parsed expression against *dtype* when, and only when, it is a literal constant.

    Runtime operands carry no value to check, so they are skipped -- the same rule the existing
    ``isinstance(expr, ConstInt)`` guards in ``ir/op`` already follow.
    """
    if dtype is None:
        return
    if isinstance(expr, ir.ConstInt):
        value: int | float = from_storage_int(expr.value, expr.type.dtype)
    elif isinstance(expr, ir.ConstFloat):
        value = expr.value
    else:
        return
    check_fits_dtype(value, dtype, subject=subject, api=api)


def make_const_int(
    value: int,
    dtype: ir.DataType | None = None,
    *,
    span: ir.Span,
    subject: str = "integer constant",
    api: str | None = None,
) -> ir.Expr:
    """Materialise an integer literal as ``ir.ConstInt``, range-checked and folded.

    With no dtype, values use ``INT64`` unless they require ``UINT64``. A dtype the caller named explicitly
    is taken at its word: the value has to fit it.
    """
    with current_span(span):
        if dtype is None:
            storage = check_ir_int(value, subject=subject)
            settled = ir.DataType.UINT64 if value > INT64_MAX else ir.DataType.INT64
            return ir.ConstInt(storage, settled, span)
        check_fits_dtype(value, dtype, subject=subject, api=api)
        return ir.ConstInt(to_storage_int(value), dtype, span)
