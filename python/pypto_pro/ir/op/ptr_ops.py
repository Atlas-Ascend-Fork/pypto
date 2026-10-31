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

"""Pointer operations for PyPTO IR (ptoas scene).

These ops emit PTO MLIR instructions (pto.addptr, pto.make_tensor_view)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call, DataType, Expr, Span

from ..._errors import InvalidShape, InvalidType
from .._operators import make_binary
from .._utils import _const_int_value, _get_span_or_capture, _normalize_expr, _to_make_tuple
from ._op_registry import op_impl


def make_tensor(
    ptr: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    stride: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    dtype: DataType | None = None,
    span: Span | None = None,
    *,
    depth: int | None = None,
    fwd_ids: Sequence[int] | None = None,
    bwd_ids: Sequence[int] | None = None,
) -> Call:
    """Create a tensor view from a raw pointer or an existing tensor.

    Emits ``ptr.make_tensor``. In the ptoas codegen this lowers to
    ``pto.make_tensor_view``; in the CCE codegen the view is materialized as a
    ``GlobalTensor`` (handled exactly like a ``pl.Tensor`` parameter).

    The first argument may be either:

    - a raw pointer (``pl.Ptr[dtype]``), or
    - an existing ``pl.Tensor`` — in which case the new view reuses the source
      tensor's underlying data pointer but with the given ``shape``/``stride``
      (and optionally a different ``dtype``).

    Args:
        ptr: Raw pointer expression (PtrType) or source tensor (TensorType)
        shape: New shape dimensions (int or Expr per dimension), or a MakeTuple
        stride: Optional stride per dimension (int or Expr), or a MakeTuple.
            If omitted, the IR records an implicit contiguous row-major layout;
            codegen derives the stride from ``shape`` when needed.
        dtype: Optional element dtype for the view. If omitted, it is derived
            from the source pointer/tensor's element type; if given, the view is
            created with this dtype (the source is reinterpreted as ``dtype``).
        span: Optional source span for debugging (auto-captured if not provided)
        depth: Read by the automatic pipeline transform only; not part of the IR
        fwd_ids: Read by the automatic pipeline transform only; not part of the IR
        bwd_ids: Read by the automatic pipeline transform only; not part of the IR

    Returns:
        Call expression creating a tensor view with the given shape and strides
    """
    actual_span = _get_span_or_capture(span)
    shape_tuple = _to_make_tuple(shape, actual_span)
    if stride is None:
        stride_tuple = _ir_core.MakeTuple([], actual_span)
    else:
        stride_tuple = _to_make_tuple(stride, actual_span)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return _ir_core.create_op_call("ptr.make_tensor", [ptr, shape_tuple, stride_tuple], kwargs, actual_span)


def make_ptr(ptr: Expr, dtype: DataType | None = None, span: Span | None = None) -> Call:
    """Extract a raw pointer from a tensor, or reinterpret an existing pointer's element dtype.

    Emits ``ptr.make_ptr``. The result is a raw pointer (PtrType) sharing the source's
    underlying address:

    - When ``ptr`` is a ``pl.Tensor``, its underlying data pointer is extracted (a tensor
      address-extraction). Pass ``dtype`` to also reinterpret the element type in one step.
    - When ``ptr`` is a ``pl.Ptr[dtype]``, this is a pointer reinterpret-cast: same address,
      (usually) different element type, e.g. turning a ``pl.Ptr[pl.DT_UINT8]`` parameter into
      a ``pl.Ptr[pl.DT_FP16]`` so it can be sliced with element semantics (``pl.addptr``) or
      wrapped as a typed tensor view (``pl.make_tensor``).

    Args:
        ptr: Source expression — a tensor (TensorType) whose data pointer is extracted, or
            an existing raw pointer (PtrType) to reinterpret.
        dtype: Target element dtype. If omitted, the source dtype is kept
            (an identity reinterpret / same-dtype address extraction).
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for the resulting pointer (PtrType with ``dtype``)
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return _ir_core.create_op_call("ptr.make_ptr", [ptr], kwargs, actual_span)


def _tensor_strides(tensor_type: _ir_core.TensorType, span: Span) -> list:
    """Element strides of the source tensor: its declared stride, else row-major."""
    shape = list(tensor_type.shape)
    view = tensor_type.tensor_view
    if view is not None and len(view.stride) == len(shape):
        return list(view.stride)
    strides: list = [0] * len(shape)
    running: int | Expr = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        dim = _const_int_value(shape[axis])
        if dim is not None:
            running *= dim
        else:
            running = make_binary("mul", _normalize_expr(running, span), shape[axis], span)
    return strides


def _linear_coordinate(coords: Sequence[int | Expr], strides: Sequence[int | Expr], span: Span) -> Expr:
    """Row-major linear element offset of per-axis coordinates."""
    total: int | Expr = 0
    for coord, stride in zip(coords, strides):
        coord_value, stride_value = _const_int_value(coord), _const_int_value(stride)
        term: int | Expr = (
            coord_value * stride_value
            if coord_value is not None and stride_value is not None
            else make_binary("mul", _normalize_expr(coord, span), _normalize_expr(stride, span), span)
        )
        if isinstance(total, int) and isinstance(term, int):
            total += term
        else:
            total = make_binary("add", _normalize_expr(total, span), _normalize_expr(term, span), span)
    return _normalize_expr(total, span)


def tensor_view(
    tensor: Expr,
    offset: Sequence[int | Expr] | int | Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Create a no-copy sub-view of an existing GM tensor.

    Emits ``ptr.make_tensor`` over the tensor's data pointer advanced by ``offset``,
    so the ptoas codegen lowers it to ``pto.make_tensor_view`` and the CCE codegen
    materializes the view as a ``GlobalTensor`` exactly like ``pl.make_tensor``.
    The view keeps the source tensor's element dtype and per-axis strides, so it
    addresses a window of the source's layout without a copy.

    Args:
        tensor: Source tensor (TensorType) to view into
        offset: Per-axis element coordinates (one per source axis), or a scalar
            linear element offset
        shape: Extent of the sub-view per axis, same rank as the source
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression creating the sub-view
    """
    actual_span = _get_span_or_capture(span)
    tensor_type = tensor.type if isinstance(tensor, Expr) else None
    if not isinstance(tensor_type, _ir_core.TensorType):
        source_kind = type(tensor_type).__name__ if tensor_type is not None else type(tensor).__name__
        raise InvalidType(
            f"pl.tensor_view source must be a GM tensor (pl.Tensor), but got {source_kind}",
            span=actual_span,
        )
    source_rank = len(tensor_type.shape)
    shape_tuple = _to_make_tuple(shape, actual_span)
    if len(shape_tuple.elements) != source_rank:
        raise InvalidShape(
            f"pl.tensor_view shape rank ({len(shape_tuple.elements)}) must match the source "
            f"tensor rank ({source_rank}); use pl.make_tensor to reshape",
            span=actual_span,
        )
    strides = _tensor_strides(tensor_type, actual_span)

    if isinstance(getattr(offset, "type", None), _ir_core.TupleType):
        coords = list(_to_make_tuple(offset, actual_span).elements)
        if len(coords) != source_rank:
            raise InvalidShape(
                f"pl.tensor_view offset has {len(coords)} coordinates, but the source tensor "
                f"has {source_rank} axes",
                span=actual_span,
            )
        linear_offset = _linear_coordinate(coords, strides, actual_span)
    else:
        linear_offset = _normalize_expr(offset, actual_span)

    ptr = addptr(make_ptr(tensor, span=actual_span), linear_offset, span=actual_span)
    return make_tensor(ptr, shape_tuple, strides, span=actual_span)


def _check_addptr_dtype(ptr: Expr) -> None:
    """Reject pointer arithmetic on sub-byte element types.

    ``pl.addptr`` (and ``ptr + offset``) advance a pointer by a whole number of
    *elements*. For the 4-bit dtypes (``DT_INT4``/``DT_UINT4``/``DT_FP4``/``DT_HF4``)
    two elements share a byte, so an element offset cannot address a half-byte and
    there is no valid C element type to lower the arithmetic to (``ToCTypeString``
    yields ``"unknown"``). Forbid it with a clear error instead of miscompiling.
    """
    ptr_type = ptr.type
    if isinstance(ptr_type, _ir_core.PtrType) and ptr_type.dtype.get_bit() < 8:
        raise InvalidType(
            f"pl.addptr / pointer arithmetic is not supported on sub-byte element type "
            f"'{ptr_type.dtype.to_string()}': offsetting by elements cannot address a "
            f"half-byte. Reinterpret the pointer as a byte-addressable dtype via "
            f"pl.make_ptr (e.g. pl.DT_UINT8) before doing pointer arithmetic."
        )


def addptr(ptr: Expr, offset: Expr, span: Span | None = None) -> Call:
    """Advance a raw pointer by an integer offset.

    Emits ``pto.addptr`` in the ptoas codegen.

    Args:
        ptr: Raw pointer expression (must have PtrType)
        offset: IR expression with integer/index ScalarType
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for pointer arithmetic (same PtrType as input)
    """
    _check_addptr_dtype(ptr)
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("ptr.addptr", [ptr, offset], {}, actual_span)


@op_impl("make_tensor")
def _build_make_tensor(self, *args, span: Span, **kwargs):
    return make_tensor(*args, span=span, **kwargs)


@op_impl("make_ptr")
def _build_make_ptr(self, *args, span: Span, **kwargs):
    return make_ptr(*args, span=span, **kwargs)


@op_impl("tensor_view")
def _build_tensor_view(self, *args, span: Span, **kwargs):
    return tensor_view(*args, span=span, **kwargs)


@op_impl("addptr")
def _build_addptr(self, *args, span: Span, **kwargs):
    return addptr(*args, span=span, **kwargs)
