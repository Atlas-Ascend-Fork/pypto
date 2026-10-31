# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Debug operations for PyPTO IR."""

from __future__ import annotations

__all__ = ["pto_assert", "dump_data", "printf", "trap"]

from collections.abc import Sequence
from typing import Any

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call, DataType, Expr, ScalarType, Span, TensorType, TileType

from ..._errors import InvalidArgument, InvalidShape, InvalidType, InvalidVal, NotSupported, argument_span
from .._utils import _const_int_value, _get_span_or_capture, _to_make_tuple, require_int
from ._op_registry import op_impl

_INTEGER_CONVERSIONS = {"d", "i", "u", "x"}
_FLOAT_CONVERSIONS = {"f"}
_POINTER_CONVERSIONS = {"p"}
_SUPPORTED_CONVERSIONS = _INTEGER_CONVERSIONS | _FLOAT_CONVERSIONS | _POINTER_CONVERSIONS
_SIGNED_INTEGER_DTYPES = tuple(
    getattr(DataType, dtype_name) for dtype_name in ("INT8", "INT16", "INT32", "INT64") if hasattr(DataType, dtype_name)
)

_UNSIGNED_INTEGER_DTYPES = tuple(
    getattr(DataType, dtype_name)
    for dtype_name in ("UINT8", "UINT16", "UINT32", "UINT64")
    if hasattr(DataType, dtype_name)
)

_UNSUPPORTED_DUMP_DTYPES = (DataType.INT4, DataType.UINT4, DataType.FP4, DataType.HF4)


def _is_signed_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsSignedInt"):
        return dtype.IsSignedInt()
    return dtype in _SIGNED_INTEGER_DTYPES


def _is_unsigned_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsUnsignedInt"):
        return dtype.IsUnsignedInt()
    return dtype in _UNSIGNED_INTEGER_DTYPES


def _is_index_dtype(dtype: DataType) -> bool:
    return dtype == DataType.INDEX


def _is_bool_dtype(dtype: DataType) -> bool:
    return dtype == DataType.BOOL


def _normalize_location_flag(loc: bool, op_name: str) -> bool:
    if not isinstance(loc, bool):
        raise InvalidType(f"debug.{op_name} requires bool loc flag, but got {type(loc).__name__}")
    return loc


def _scan_printf_format(format_str: str) -> list[str]:
    specs: list[str] = []
    i = 0
    while i < len(format_str):
        if format_str[i] != "%":
            i += 1
            continue
        if i + 1 < len(format_str) and format_str[i + 1] == "%":
            raise NotSupported("printf does not support literal '%%'")

        j = i + 1
        if j >= len(format_str):
            raise InvalidVal("printf format string ends with an incomplete conversion")

        conversion = format_str[j]
        if conversion not in _SUPPORTED_CONVERSIONS:
            if conversion in "-+ #0." or conversion.isdigit():
                raise NotSupported("printf does not support flags, width, or precision")
            raise NotSupported(f"printf does not support conversion '%{conversion}'")

        specs.append(format_str[i:j + 1])
        i = j + 1

    return specs


def _validate_printf_arguments(
    format_str: str, normalized_args: Sequence[Expr], *, op_name: str
) -> None:
    specs = _scan_printf_format(format_str)
    if len(specs) != len(normalized_args):
        raise InvalidArgument(f"{op_name} format expects {len(specs)} scalar arguments, but got {len(normalized_args)}")

    for idx, (spec, arg) in enumerate(zip(specs, normalized_args)):
        arg_type = getattr(arg, "type", None)
        conversion = spec[-1]

        # %p accepts pointer types (PtrType)
        if conversion == "p":
            if not isinstance(arg_type, _ir_core.PtrType):
                raise InvalidVal(
                    f"debug.{op_name} conversion '{spec}' requires pointer, but got {type(arg_type).__name__}"
                )
            continue

        if not isinstance(arg_type, ScalarType):
            raise InvalidType(
                f"debug.{op_name} argument {idx} requires ScalarType input, but got {type(arg_type).__name__}"
            )
        scalar_type = arg_type

        if conversion in {"d", "i"} and not (
            _is_signed_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
        ):
            raise InvalidVal(
                f"debug.{op_name} conversion '{spec}' requires signed integer, bool, or index scalar, "
                f"but got {scalar_type.dtype}"
            )
        if conversion == "u" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
        ):
            raise InvalidVal(
                f"debug.{op_name} conversion '{spec}' requires unsigned integer, bool, or index scalar, "
                f"but got {scalar_type.dtype}"
            )
        if conversion == "x" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype) or _is_index_dtype(scalar_type.dtype)
        ):
            raise InvalidVal(
                f"debug.{op_name} conversion '{spec}' requires unsigned integer or index scalar, "
                f"but got {scalar_type.dtype}"
            )
        if conversion in _FLOAT_CONVERSIONS and scalar_type.dtype != DataType.FP32:
            raise InvalidVal(f"debug.{op_name} conversion '{spec}' requires FP32 scalar, but got {scalar_type.dtype}")


def _validate_dump_offsets_shapes(
    offsets_tuple: _ir_core.MakeTuple,
    shapes_tuple: _ir_core.MakeTuple,
    input_shape: Sequence[Expr],
    op_name: str,
) -> None:
    rank = len(input_shape)
    if len(offsets_tuple.elements) != rank or len(shapes_tuple.elements) != rank:
        raise InvalidShape(
            f"debug.{op_name} offsets/shapes must match input rank {rank}, got "
            f"{len(offsets_tuple.elements)} offsets and {len(shapes_tuple.elements)} shapes"
        )
    for idx, (offset_expr, shape_expr, input_dim) in enumerate(
        zip(offsets_tuple.elements, shapes_tuple.elements, input_shape, strict=True)
    ):
        input_extent = _const_int_value(input_dim)
        offset_value = require_int(
            offset_expr,
            subject=f"debug.{op_name} offset at axis {idx}",
            lower_bound=0,
            upper_bound=input_extent - 1 if input_extent is not None else None,
        )
        require_int(
            shape_expr,
            subject=f"debug.{op_name} shape at axis {idx}",
            lower_bound=1,
            upper_bound=input_extent - offset_value if input_extent is not None and offset_value is not None else None,
        )


def dump_data(
    data: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    workspace: Expr | None = None,
    loc: bool = False,
    span: Span | None = None,
    section: _ir_core.SectionKind | None = None,
    flag: str | None = None,
) -> Call:
    """Print a Tensor or Tile, or a logical window of it, for debugging."""
    actual_span = _get_span_or_capture(span)
    show_location = _normalize_location_flag(loc, "dump_data")
    data_type = data.type
    if flag is not None and not isinstance(flag, str):
        raise InvalidType(f"debug.dump_data requires str flag, but got {type(flag).__name__}")
    if not isinstance(data_type, (TensorType, TileType)):
        raise InvalidType(f"debug.dump_data requires Tensor or Tile input, but got {type(data_type).__name__}")
    if data_type.dtype in _UNSUPPORTED_DUMP_DTYPES:
        dtype_name = f"DT_{data_type.dtype.to_string().upper()}"
        raise NotSupported(
            f"debug.dump_data does not support {dtype_name}; use DT_FP4E2M1 or DT_FP4E1M2 for packed FP4 data"
        )
    if (offsets is None) != (shapes is None):
        raise InvalidArgument("debug.dump_data offsets and shapes must be provided together")

    dump_kwargs: dict[str, Any] = {"show_location": show_location}
    if flag:
        dump_kwargs["dump_flag"] = flag

    if isinstance(data_type, TensorType):
        op_name = "debug.dump_tensor"
    else:
        op_name = "debug.dump_tile"
        if data_type.memref is None:
            raise InvalidArgument("debug.dump_data requires Tile input to have an allocated memory space")
        memory_space = data_type.memref.memory_space_
        if memory_space in (
            _ir_core.MemorySpace.Left,
            _ir_core.MemorySpace.Right,
            _ir_core.MemorySpace.ScaleLeft,
            _ir_core.MemorySpace.ScaleRight,
        ):
            raise NotSupported(f"debug.dump_data does not support {memory_space.name} tiles")
        if section == _ir_core.SectionKind.Vector and memory_space != _ir_core.MemorySpace.Vec:
            raise NotSupported(
                f"pl.dump_data of a {memory_space.name} tile is not allowed in a Vector section; "
                "only Vec (UB) tiles can be dumped there",
                span=actual_span,
            )
        if section == _ir_core.SectionKind.Cube and memory_space not in (
            _ir_core.MemorySpace.Mat,
            _ir_core.MemorySpace.Acc,
            _ir_core.MemorySpace.Bias,
            _ir_core.MemorySpace.Scaling,
        ):
            raise NotSupported(
                f"pl.dump_data of a {memory_space.name} tile is not allowed in a Cube section",
                span=actual_span,
            )

    operands = [data]
    if offsets is not None:
        offsets_tuple = _to_make_tuple(offsets, actual_span)
        shapes_tuple = _to_make_tuple(shapes, actual_span)
        _validate_dump_offsets_shapes(offsets_tuple, shapes_tuple, data_type.shape, "dump_data")
        operands.extend([offsets_tuple, shapes_tuple])
    return _ir_core.create_op_call(op_name, operands, dump_kwargs, actual_span)


def printf(format_str: str, *args: Expr, loc: bool = False, span: Span | None = None) -> Call:
    """Print scalar values using a compile-time format string."""
    actual_span = _get_span_or_capture(span)
    show_location = _normalize_location_flag(loc, "printf")
    if not isinstance(format_str, str):
        raise InvalidType(f"debug.printf requires string format literal, but got {type(format_str).__name__}")

    _validate_printf_arguments(format_str, args, op_name="printf")

    kwargs: dict[str, Any] = {"format": format_str, "show_location": show_location}
    return _ir_core.create_op_call("debug.printf", list(args), kwargs, actual_span)


def pto_assert(
    condition: Expr,
    format_str: str | None = None,
    *args: Expr,
    condition_text: str | None = None,
    loc: bool = False,
    span: Span | None = None,
) -> Call:
    """Abort execution when a scalar boolean condition is false."""
    actual_span = _get_span_or_capture(span)
    show_location = _normalize_location_flag(loc, "pto_assert")
    condition_type = getattr(condition, "type", None)
    if not isinstance(condition_type, ScalarType) or condition_type.dtype != DataType.BOOL:
        raise InvalidType(
            "debug.pto_assert requires a scalar bool condition, "
            f"but got {type(condition_type).__name__}({getattr(condition_type, 'dtype', condition_type)})"
        )

    if condition_text is None:
        condition_text = "condition"
    if not isinstance(condition_text, str):
        raise InvalidType(
            f"debug.pto_assert requires string condition_text metadata, but got {type(condition_text).__name__}"
        )

    normalized_args: list[Expr] = []
    if format_str is None:
        format_value = ""
    else:
        if not isinstance(format_str, str):
            raise InvalidType(f"debug.pto_assert requires string literal format, but got {type(format_str).__name__}")
        normalized_args = list(args)
        _validate_printf_arguments(format_str, normalized_args, op_name="pto_assert")
        format_value = format_str

    kwargs: dict[str, Any] = {
        "condition_text": condition_text,
        "format": format_value,
        "show_location": show_location,
    }
    return _ir_core.create_op_call("debug.assert", [condition, *normalized_args], kwargs, actual_span)


def trap(*, span: Span | None = None) -> Call:
    """Abort execution by inserting a trap."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("debug.trap", [], {}, actual_span)


# ---------------------------------------------------------------------------
# Value-facing implementations
# ---------------------------------------------------------------------------


@op_impl("dump_data")
def _build_dump_data(self, *args: Expr, span: Span, **kwargs):
    if "loc" in kwargs:
        kwargs["loc"] = self.require_const_value(
            kwargs["loc"], "loc", expects="bool",
            hint="loc must be known at compile time.",
        )
    return dump_data(*args, section=self.target, **kwargs, span=span)


@op_impl("pto_assert")
def _build_pto_assert(self, *args: Expr, span: Span, **kwargs):
    if "loc" in kwargs:
        kwargs["loc"] = self.require_const_value(
            kwargs["loc"], "loc", expects="bool",
            hint="loc must be known at compile time.",
        )
    condition = args[0] if args else kwargs.get("condition")
    condition_span = argument_span(0 if args else "condition", getattr(condition, "span", span))
    condition_text = self.span_tracker.get_source_text(condition_span)
    return pto_assert(*args, condition_text=condition_text or None, **kwargs, span=span)


@op_impl("printf")
def _build_printf(self, *args, span: Span, **kwargs):
    if "loc" in kwargs:
        kwargs["loc"] = self.require_const_value(
            kwargs["loc"], "loc", expects="bool",
            hint="loc must be known at compile time.",
        )
    return printf(*args, span=span, **kwargs)


@op_impl("trap")
def _build_trap(self, *, span: Span):
    return trap(span=span)
