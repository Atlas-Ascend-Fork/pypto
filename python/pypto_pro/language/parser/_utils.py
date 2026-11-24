# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small shared parser utilities."""

from __future__ import annotations

__all__ = ["_check_expression_value", "_is_const_expr", "_is_simd_func_type", "_is_simt_func_type"]

from pypto.pypto_impl import ir

from ..._errors import InvalidType


def _contains_valueless_type(value_type: ir.Type) -> bool:
    if isinstance(value_type, (ir.NoneType, ir.UnknownType)):
        return True
    return isinstance(value_type, ir.TupleType) and any(_contains_valueless_type(t) for t in value_type.types)


def _check_expression_value(value: object, span: ir.Span, *, nested: bool) -> None:
    """Reject valueless results where an operand is required, before any folding can discard them."""
    value_type = value.type if isinstance(value, ir.Expr) else None
    if isinstance(value_type, ir.TupleType) and _contains_valueless_type(value_type):
        raise InvalidType(
            "Tuple elements must have values; NoneType and UnknownType elements are not supported",
            span=span,
        )
    # Direct statements and name assignments can discard an operation's result.
    if not nested:
        return
    if value is None or isinstance(value_type, (ir.NoneType, ir.UnknownType)):
        # The builder's canonical None represents literal/captured Python None,
        # which optional API parameters accept. Calls and assigned void results
        # are distinct IR objects even when they have the same NoneType.
        if isinstance(value_type, ir.NoneType) and value is ir.IRBuilder().none():
            return
        raise InvalidType(
            "Expression returns no value and cannot be used as an operand or argument",
            span=span,
            hint="Call the operation separately; do not use its result in another expression",
        )


def _is_const_expr(expr: object) -> bool:
    """Return whether an IR expression is a recursively constant value."""
    if isinstance(expr, (ir.ConstInt, ir.ConstBool, ir.ConstFloat)):
        return True
    return (
        isinstance(expr, ir.MakeTuple)
        and all(_is_const_expr(element) for element in expr.elements)
    )


def _is_simt_func_type(func_type: ir.FunctionType) -> bool:
    """Return whether a function uses the SIMT execution context."""
    return func_type in (
        ir.FunctionType.SimtVF, ir.FunctionType.SimtCallee, ir.FunctionType.SimtKernel
    )


def _is_simd_func_type(func_type: ir.FunctionType) -> bool:
    """Return whether a function uses the SIMD vector-function context."""
    return func_type in (
        ir.FunctionType.SimdVF,
        ir.FunctionType.SimdCallee,
        ir.FunctionType.SimdTileVF,
        ir.FunctionType.SimdTileCallee,
    )
