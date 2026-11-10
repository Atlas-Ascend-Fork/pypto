# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar and tile arithmetic implementations receiving parsed operands."""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Expr, Span

from ..._errors import InvalidType
from .._utils import require_dtype_in, require_dtypes_match, require_scalar_dtype
from ._op_registry import op_impl, op_impl_ast


@op_impl("min")
def _build_min(self, lhs: Expr, rhs: Expr, *, span: Span) -> Expr:
    result = _ir_core.min_(lhs, rhs, span)
    folded = self._fold_const_binop("min", result, span) if isinstance(result, _ir_core.BinaryExpr) else None
    return result if folded is None else folded


@op_impl("max")
def _build_max(self, lhs: Expr, rhs: Expr, *, span: Span) -> Expr:
    result = _ir_core.max_(lhs, rhs, span)
    folded = self._fold_const_binop("max", result, span) if isinstance(result, _ir_core.BinaryExpr) else None
    return result if folded is None else folded


# ---------------------------------------------------------------------------
# Builders for binary ops (tile-tile / tile-scalar dispatch)
#   rhs is Tile   -> block.add / block.sub / ... / block.and
#   rhs is Scalar -> block.adds / block.subs / ... / block.ands
# ---------------------------------------------------------------------------
@op_impl("adds")
def _build_adds(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call("block.adds", [out, lhs, rhs], {}, span)


@op_impl("subs")
def _build_subs(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call("block.subs", [out, lhs, rhs], {}, span)


@op_impl("muls")
def _build_muls(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call("block.muls", [out, lhs, rhs], {}, span)


@op_impl("add")
def _ir_add(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _BINARY_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BINARY_DTYPES, subject="add")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="add")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="add", scalar_op="adds", span=span)


@op_impl("sub")
def _ir_sub(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _BINARY_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BINARY_DTYPES, subject="sub")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="sub")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="sub", scalar_op="subs", span=span)


@op_impl("mul")
def _ir_mul(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _MUL_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _MUL_DTYPES, subject="mul")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="mul")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="mul", scalar_op="muls", span=span)


@op_impl("div")
def _ir_div(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _DIV_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _DIV_DTYPES, subject="div")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="div")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="div", scalar_op="divs", span=span)


@op_impl("rem")
def _ir_rem(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _REM_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _REM_DTYPES, subject="rem")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="rem")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="rem", scalar_op="rems", span=span)


@op_impl("and_")
def _ir_and(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _BITWISE_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BITWISE_DTYPES, subject="and")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="and")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="and", scalar_op="ands", span=span)


@op_impl("or_")
def _ir_or(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _BITWISE_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BITWISE_DTYPES, subject="or")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="or")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="or", scalar_op="ors", span=span)


@op_impl("shl")
def _ir_shl(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _BITWISE_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BITWISE_DTYPES, subject="shl")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="shl")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="shl", scalar_op="shls", span=span)


@op_impl("shr")
def _ir_shr(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.ir.op.block_ops import _BITWISE_DTYPES, _create_tile_scalar_op

    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BITWISE_DTYPES, subject="shr")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="shr")
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="shr", scalar_op="shrs", span=span)


# ---------------------------------------------------------------------------
# Builders for min/max (element-wise + reduce overload via dim kwarg)
# ---------------------------------------------------------------------------
@op_impl("minimum")
def _ir_minimum(self,
    out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, dim: int | Expr | None = None,
) -> Expr:
    from pypto_pro.ir.op.block_ops import (
        _BINARY_DTYPES,
        _create_dim_op,
        _create_tile_scalar_op,
    )

    if isinstance(getattr(dim, "type", None), _ir_core.NoneType):
        dim = None
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BINARY_DTYPES, subject="minimum")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="minimum")
    if dim is not None:
        return _create_dim_op([out, lhs, rhs], row_op="row_min", col_op="col_min", dim=dim, span=span)
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="minimum", scalar_op="mins", span=span)


@op_impl("maximum")
def _ir_maximum(self,
    out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, dim: int | Expr | None = None,
) -> Expr:
    from pypto_pro.ir.op.block_ops import (
        _BINARY_DTYPES,
        _create_dim_op,
        _create_tile_scalar_op,
    )

    if isinstance(getattr(dim, "type", None), _ir_core.NoneType):
        dim = None
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BINARY_DTYPES, subject="maximum")
    # rhs consistency is only enforced for the tile-tile path; the
    # tile-scalar path keeps the fits-dtype contract in
    # _create_tile_scalar_op (mirrors the AscendC same-type template).
    rhs_type = getattr(rhs, "type", None)
    rhs_dt = getattr(rhs_type, "dtype", None) if isinstance(rhs_type, _ir_core.TileType) else None
    require_dtypes_match([getattr(lhs.type, "dtype", None), rhs_dt], dt, subject="maximum")
    if dim is not None:
        return _create_dim_op([out, lhs, rhs], row_op="row_max", col_op="col_max", dim=dim, span=span)
    return _create_tile_scalar_op(out, lhs, rhs, tile_op="maximum", scalar_op="maxs", span=span)


# Keep AST registration: construct literals directly in the requested dtype so
# values such as FP64 1e200 are not rejected by default expression range checks.
# This also preserves the existing numeric-literal syntax restriction.
@op_impl_ast("const")
def _parse_typed_constant(self, call: ast.Call) -> Expr:
    from pypto_pro.language.parser.diagnostics import check_fits_dtype, make_const_int

    span = self.span_tracker.get_span(call)
    node = call.args[0]
    literal = node.operand if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) else node
    if not isinstance(literal, ast.Constant) or not isinstance(literal.value, (int, float)):
        raise InvalidType(
            "pl.const() first argument must be a numeric literal",
            hint="Use an int or float literal: pl.const(42, pl.DT_INT32)",
        )
    value = -literal.value if literal is not node else literal.value
    dtype = self.parse_expression(call.args[1])
    self._check_declared_types("const", [value, dtype], {})
    require_scalar_dtype(dtype, subject="pl.const() does not support")
    if dtype == _ir_core.DataType.BOOL:
        check_fits_dtype(value, dtype, subject="value", api="pl.const")
        return _ir_core.ConstBool(bool(value), span)
    if dtype.is_float():
        check_fits_dtype(float(value), dtype, subject="value", api="pl.const")
        return _ir_core.ConstFloat(float(value), dtype, span)
    return make_const_int(int(value), dtype, span=span, subject="value", api="pl.const")


@op_impl("astype")
def _build_scalar_astype(self, value: Expr, dtype, /, *, span: Span) -> Expr:
    for role, scalar_dtype in (("source", value.type.dtype), ("destination", dtype)):
        require_scalar_dtype(
            scalar_dtype, subject=f"pl.astype() does not support {role}",
        )
    return _ir_core.cast(value, dtype, span)
