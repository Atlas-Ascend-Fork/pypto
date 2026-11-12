# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Expr, Span

from ..._errors import InvalidType
from .._limits import from_storage_int
from ._op_registry import op_impl


def _atomic_target_dtype(op_name: str, tensor: Expr):
    """The element dtype of the GM Tensor an atomic targets."""
    tensor_type = tensor.type
    if not isinstance(tensor_type, _ir_core.TensorType):
        raise InvalidType(
            f"pl.{op_name}() requires a GM Tensor, got {type(tensor_type).__name__}",
            hint=(
                "Scalar atomics operate on global memory only"
            ),
            parser_retry=True,
        )
    return tensor_type.dtype


def _adapt_atomic_operand(self, op_name: str, argument_name: str, value: Expr, dtype) -> Expr:
    """Retype a numeric literal to the element dtype."""
    if not isinstance(value, (_ir_core.ConstInt, _ir_core.ConstFloat)):
        return value
    if dtype.is_int() and isinstance(value, _ir_core.ConstFloat):
        raise InvalidType(
            f"pl.{op_name}() requires an integer {argument_name} for dtype {dtype}",
            parser_retry=True,
        )
    literal = from_storage_int(value.value, value.type.dtype) if isinstance(value, _ir_core.ConstInt) else value.value
    return self._make_scalar_constant(literal, dtype, value.span)


def _build_atomic_rmw(
    self,
    tensor: Expr,
    offset: Expr,
    value: Expr,
    *,
    op_name: str,
    kind,
    argument_name: str = "value",
    span: Span,
) -> Expr:
    dtype = _atomic_target_dtype(op_name, tensor)
    value = _adapt_atomic_operand(self, op_name, argument_name, value, dtype)
    return _ir_core.create_op_call("block.atomic_rmw", [tensor, offset, value], {"kind": int(kind)}, span)


def _atomic_cas_call(self, tensor: Expr, offset: Expr, compare: Expr, value: Expr, *, span: Span) -> Expr:
    dtype = _atomic_target_dtype("atomic_cas", tensor)
    compare = _adapt_atomic_operand(self, "atomic_cas", "compare", compare, dtype)
    value = _adapt_atomic_operand(self, "atomic_cas", "value", value, dtype)
    return _ir_core.create_op_call("block.atomic_cas", [tensor, offset, compare, value], {}, span)


@op_impl("atomic_add")
def _build_atomic_add(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_add", kind=_ir_core.AtomicRMWMode.ADD, span=span
    )


@op_impl("atomic_sub")
def _build_atomic_sub(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_sub", kind=_ir_core.AtomicRMWMode.SUB, span=span
    )


@op_impl("atomic_exch")
def _build_atomic_exch(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_exch", kind=_ir_core.AtomicRMWMode.EXCH, span=span
    )


@op_impl("atomic_max")
def _build_atomic_max(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_max", kind=_ir_core.AtomicRMWMode.MAX, span=span
    )


@op_impl("atomic_min")
def _build_atomic_min(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_min", kind=_ir_core.AtomicRMWMode.MIN, span=span
    )


@op_impl("atomic_inc")
def _build_atomic_inc(self, tensor, offset, limit, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, limit, op_name="atomic_inc", kind=_ir_core.AtomicRMWMode.INC,
        argument_name="limit", span=span,
    )


@op_impl("atomic_dec")
def _build_atomic_dec(self, tensor, offset, limit, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, limit, op_name="atomic_dec", kind=_ir_core.AtomicRMWMode.DEC,
        argument_name="limit", span=span,
    )


@op_impl("atomic_and")
def _build_atomic_and(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_and", kind=_ir_core.AtomicRMWMode.AND, span=span
    )


@op_impl("atomic_or")
def _build_atomic_or(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_or", kind=_ir_core.AtomicRMWMode.OR, span=span
    )


@op_impl("atomic_xor")
def _build_atomic_xor(self, tensor, offset, value, *, span: Span) -> Expr:
    return _build_atomic_rmw(
        self, tensor, offset, value, op_name="atomic_xor", kind=_ir_core.AtomicRMWMode.XOR, span=span
    )


@op_impl("atomic_cas")
def _build_atomic_cas(self, tensor, offset, compare, value, *, span: Span) -> Expr:
    return _atomic_cas_call(self, tensor, offset, compare, value, span=span)
