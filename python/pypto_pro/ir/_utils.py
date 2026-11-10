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

"""Utility functions for IR construction."""

from __future__ import annotations

__all__ = [
    "_const_int_value",
    "_const_int_values",
    "_const_scalar_value",
    "_get_span_or_capture",
    "_is_int",
    "_is_int_type",
    "_is_none",
    "_normalize_expr",
    "_normalize_shape",
    "_to_make_tuple",
    "require_const_bool",
    "require_const_int",
    "require_const_int_list",
    "require_const_number",
    "require_dtype_in",
    "require_dtypes_match",
    "require_enum",
    "require_int",
    "require_list",
    "require_scalar_dtype",
    "unwrap_type",
]


from collections.abc import Sequence
import inspect
from typing import TypeVar

from pypto.pypto_impl import ir as _ir
from pypto.pypto_impl.ir import DataType

from .._errors import InvalidType, InvalidVal, NotSupported, get_current_span

_EnumT = TypeVar("_EnumT")


def require_scalar_dtype(dtype: DataType, *, subject: str) -> None:
    """Reject storage-only formats at a scalar construction or access boundary."""
    if dtype.is_storage_only():
        raise NotSupported(
            f"{subject} dtype {dtype}; low-precision types are storage-only "
            "and cannot be used in scalar expressions",
            hint="Use a supported scalar dtype; for supported FP8/FP4 register conversions, use pl.Vf.astype",
            parser_retry=True,
        )


def _is_none(value: object) -> bool:
    return value is None or isinstance(value, _ir.Expr) and isinstance(value.type, _ir.NoneType)


def _const_scalar_value(value: object):
    """Unwrap a scalar constant; leave runtime expressions unchanged."""
    if isinstance(value, _ir.ConstBool):
        return bool(value.value)
    if isinstance(value, _ir.ConstInt):
        return bool(value.value) if value.type.dtype == DataType.BOOL else int(value.value)
    if isinstance(value, _ir.ConstFloat):
        return float(value.value)
    if isinstance(value, _ir.Cast):
        operand = _const_scalar_value(value.operand)
        if isinstance(operand, (bool, int, float)):
            dtype = value.type.dtype
            if dtype == DataType.BOOL:
                return bool(operand)
            if dtype.is_int():
                return int(operand)
            if dtype.is_float():
                return float(operand)
    return value


def _const_int_value(value: _ir.Expr) -> int | None:
    """Extract an IR integer constant; return None for other inputs."""
    if isinstance(value, _ir.ConstInt):
        return int(_const_scalar_value(value))
    return None


def _const_int_values(values: Sequence[_ir.Expr]) -> list[int] | None:
    """Extract IR integer dimensions; return None if any element is not constant."""
    constants = [_const_int_value(value) for value in values]
    return None if any(value is None for value in constants) else constants


def require_const_int(
    value: _ir.Expr,
    *,
    subject: str,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
) -> int:
    """Require a compile-time integer and validate its inclusive bounds."""
    const_value = _const_scalar_value(value) if isinstance(value, _ir.Expr) else None
    if not _is_int(const_value):
        raise InvalidType(f"{subject} must be a compile-time integer constant")
    if lower_bound is not None and const_value < lower_bound:
        raise InvalidVal(f"{subject} must be at least {lower_bound}, got {const_value}")
    if upper_bound is not None and const_value > upper_bound:
        raise InvalidVal(f"{subject} must be at most {upper_bound}, got {const_value}")
    return const_value


def require_int(
    value: _ir.Expr,
    *,
    subject: str,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
) -> int | None:
    """Require an integer scalar and validate constant values when available.

    Runtime integer expressions are accepted and return ``None`` because their
    bounds cannot be checked while constructing the IR.
    """
    if _const_int_value(value) is not None:
        return require_const_int(
            value,
            subject=subject,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    if _is_int_type(getattr(value, "type", None)):
        return None
    raise InvalidType(f"{subject} must be an integer scalar")


def require_list(
    value: _ir.Expr,
    *,
    subject: str,
    min_length: int = 0,
) -> list:
    """Read an IR tuple without emitting statements; reject Python containers."""
    if not isinstance(value, _ir.Expr) or not isinstance(value.type, _ir.TupleType):
        raise InvalidType(f"{subject} must be an IR tuple")
    elements = list(_to_make_tuple(value, get_current_span()).elements)
    if len(elements) < min_length:
        raise InvalidVal(f"{subject} must contain at least {min_length} elements, got {len(elements)}")
    return elements


def require_const_int_list(
    value: _ir.Expr,
    *,
    subject: str,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
    min_length: int = 0,
) -> list[int]:
    """Require integer constants and check each element against inclusive bounds."""
    elements = require_list(value, subject=subject, min_length=min_length)
    constants = []
    for index, element in enumerate(elements):
        constants.append(require_const_int(
            element, subject=f"{subject}[{index}]", lower_bound=lower_bound, upper_bound=upper_bound,
        ))
    return constants


def require_const_bool(value: _ir.Expr, *, subject: str) -> bool:
    """Require a boolean constant without accepting integer 0/1."""
    constant = _const_scalar_value(value) if isinstance(value, _ir.Expr) else None
    if not isinstance(constant, bool):
        raise InvalidType(f"{subject} must be a compile-time bool")
    return constant


def require_const_number(
    value: _ir.Expr,
    *,
    subject: str,
    lower_bound: int | float | None = None,
    upper_bound: int | float | None = None,
) -> int | float:
    """Require a numeric constant, excluding bool, and check inclusive bounds."""
    constant = _const_scalar_value(value) if isinstance(value, _ir.Expr) else None
    if isinstance(constant, bool) or not isinstance(constant, (int, float)):
        raise InvalidType(f"{subject} must be a compile-time number")
    if lower_bound is not None and constant < lower_bound:
        raise InvalidVal(f"{subject} must be at least {lower_bound}, got {constant}")
    if upper_bound is not None and constant > upper_bound:
        raise InvalidVal(f"{subject} must be at most {upper_bound}, got {constant}")
    return constant


def require_enum(
    value: object, enum_class: type[_EnumT], *, subject: str,
) -> _EnumT:
    """Require an instance of the specified enum class; integers are not converted."""
    if not isinstance(value, enum_class):
        raise InvalidType(
            f"{subject} must be a {enum_class.__name__}, got {type(value).__name__}",
        )
    return value


def require_dtype_in(
    dtype: DataType | None, allowed_dtypes: tuple[DataType, ...], *, subject: str,
) -> None:
    """Check a known dtype against the caller's supported types."""
    if dtype is not None and dtype not in allowed_dtypes:
        allowed_names = ", ".join(str(d) for d in allowed_dtypes)
        raise InvalidType(
            f"{subject}: unsupported dtype {dtype}, supported: {allowed_names}",
        )


def require_dtypes_match(
    dtypes: Sequence[DataType | None], expected_dtype: DataType | None, *, subject: str,
) -> None:
    """Require each known dtype in the sequence to match expected_dtype."""
    for index, dtype in enumerate(dtypes):
        if dtype is not None and dtype != expected_dtype:
            raise InvalidType(
                f"{subject}: dtype mismatch at index {index}: expected {expected_dtype}, got {dtype}",
            )


def unwrap_type(ty: _ir.Type) -> _ir.Type:
    """Project a tile value type to the data properties checked by an op."""
    if isinstance(ty, _ir.MutexTileType):
        return _ir.TileType(ty.shape, ty.dtype, ty.memref, ty.tile_view, ty.hardware_info)
    return ty


def _get_span_or_capture(span: _ir.Span | None = None, frame_offset: int = 1) -> _ir.Span:
    """Get explicit span or capture from caller.

    Args:
        span: Explicit span if provided
        frame_offset: Additional frames to skip beyond immediate caller

    Returns:
        Provided span or captured span from call site
    """
    if span is not None:
        return span

    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back

    for _ in range(frame_offset):
        if frame is None:
            break
        frame = frame.f_back

    if frame is not None:
        info = inspect.getframeinfo(frame)
        return _ir.Span(info.filename, info.lineno, -1)

    return _ir.Span.unknown()


def _is_int(value: object) -> bool:
    """Whether *value* is a plain integer (bool is not, despite subclassing int)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_int_type(value_type: object) -> bool:
    """Whether an IR type describes an integer scalar."""
    return isinstance(value_type, _ir.ScalarType) and value_type.dtype.is_int()


def _normalize_expr(
    value: int | float | _ir.Expr,
    span: _ir.Span | None = None,
    int_dtype: DataType | None = None,
    float_dtype: DataType = DataType.FP32,
) -> _ir.Expr:
    """Convert Python values to IR expressions.

    Args:
        value: Python int/float or existing Expr
        span: Optional span for created constants
        int_dtype: Explicit data type for integer constants. By default, use
            INT64 unless the value requires UINT64.
        float_dtype: Data type to use for float constants (default: FP32)

    Returns:
        IR expression node

    Raises:
        InvalidType: If value is not int, float, or ir.Expr
        OutOfRange: If an integer is outside the range the IR can carry
    """
    if isinstance(value, _ir.Expr):
        return value

    actual_span = span if span is not None else _ir.Span.unknown()

    if isinstance(value, int):
        from pypto_pro.language.parser.diagnostics import make_const_int

        return make_const_int(value, int_dtype, span=actual_span)
    elif isinstance(value, float):
        return _ir.ConstFloat(value, float_dtype, actual_span)
    else:
        raise InvalidType(f"Cannot convert {type(value)} to IR expression")


def _normalize_shape(
    shape: Sequence[int | _ir.Expr],
    span: _ir.Span | None = None,
) -> list[_ir.Expr]:
    """Convert shape dimensions to IR expressions.

    Args:
        shape: Sequence of integers or Expr nodes representing shape dimensions
        span: Optional span for created constants

    Returns:
        List of IR expression nodes

    Raises:
        TypeError: If shape contains non-int, non-Expr values
    """
    return [_normalize_expr(dim, span, int_dtype=DataType.INT64) for dim in shape]


def _to_make_tuple(
    value: _ir.MakeTuple | Sequence[int | float | _ir.Expr],
    span: _ir.Span | None = None,
) -> _ir.MakeTuple:
    """Normalize a sequence or MakeTuple into a MakeTuple IR node.

    Args:
        value: Either an existing MakeTuple (returned as-is) or a sequence
            of ints/floats/Exprs to wrap
        span: Optional span for created constants

    Returns:
        MakeTuple IR expression
    """
    if isinstance(value, _ir.MakeTuple):
        return value
    actual_span = span if span is not None else _ir.Span.unknown()
    # A tuple-typed Expr (e.g. a helper-function parameter bound to a ``[i, j]`` offset
    # list at the call site) is not a Python sequence and cannot be iterated directly.
    # Expand it into per-element ``GetItemExpr`` accesses so offsets can be passed through
    # helper functions.
    if isinstance(value, _ir.Expr):
        value_type = getattr(value, "type", None)
        if isinstance(value_type, _ir.TupleType):
            elements = [
                _ir.GetItemExpr(value, _ir.ConstInt(i, DataType.INT64, actual_span), actual_span)
                for i in range(len(value_type.types))
            ]
            return _ir.MakeTuple(elements, actual_span)
        raise InvalidType(
            f"Cannot convert Expr of type {value_type} to an offset tuple; expected a MakeTuple or a tuple-typed value"
        )
    elements = [_normalize_expr(v, actual_span) for v in value]
    return _ir.MakeTuple(elements, actual_span)
