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

"""VF op front-end: ``_OP_REGISTRY`` handlers for the ``vf.*`` namespace."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pypto.pypto_impl import ir as _ir_core

from ..._errors import InvalidType
from ._op_registry import op_impl


def _check_vf_scalar_operands(op_name: str, args: list, dtype: _ir_core.DataType | None, span: _ir_core.Span) -> None:
    """Reject a literal scalar operand of ``vf.<op_name>`` that ``dtype`` cannot represent.

    Non-constant operands carry no value to check and are skipped. Only the
    builder of the scalar-value family (and the muls_cast implementation with
    ``_infer_src_dtype``) calls in here; no registry flag is consulted.
    """
    if dtype is None:
        return
    from ...language.parser.diagnostics import check_const_expr_fits_dtype

    for arg in args:
        check_const_expr_fits_dtype(arg, dtype, span=span, api=f"vf.{op_name}")


def _infer_src_dtype(parsed_src_args: list) -> _ir_core.DataType | None:
    """The dtype of the first data-carrying source, or None if no source does.

    Only matches types whose dtype represents actual data: ScalarType,
    RegType (RegTensor kind), and ShapedType (base of TileType/TensorType).
    Auxiliary register kinds (MaskReg, AddrReg, UnalignReg) are ignored
    because their dtype is an implementation detail (e.g. UnalignReg is
    always UINT32), not the data dtype that should determine the output
    register's type.

    Mask-to-mask ops (and_, or_, xor, not_, move) have only MaskReg
    sources; for those the MaskReg dtype mirrors the underlying data dtype
    (a comparison mask carries the compared tensors' dtype), so fall back
    to the first MaskReg source instead of failing inference.
    """
    mask_fallback = None
    for src_expr in parsed_src_args:
        ty = getattr(src_expr, 'type', None)
        if isinstance(ty, (_ir_core.ScalarType, _ir_core.ShapedType)):
            return ty.dtype
        if isinstance(ty, _ir_core.RegType):
            if ty.regtype == _ir_core.RegKind.VecReg:
                return ty.dtype
            if mask_fallback is None and ty.regtype == _ir_core.RegKind.PredReg:
                mask_fallback = ty.dtype
    return mask_fallback


def _unwrap_const_kwargs(kwargs: dict[str, Any]) -> None:
    """Fold constant IR kwarg values back to their Python values.

    ``_dispatch_op`` parses keyword values with ``parse_expression``, so a
    literal kwarg (``datablock=True``, ``merge_mode=0``) arrives here as a
    ``ConstBool`` / ``ConstInt`` expr. The C++ op registry's attribute
    validation expects the plain Python scalar, so unwrap any parse-time
    constant (lenient exactly like the pre-refactor ``parse_op_kwargs``:
    a value with no parse-time form keeps its IR).
    """
    from pypto_pro.language.parser._expr_evaluator import ExprEvaluator

    for key, value in list(kwargs.items()):
        found, python_value = ExprEvaluator.ir_to_python_value(value)
        if found:
            kwargs[key] = python_value


def _make_vf_call(full_name: str, args: tuple, kwargs: dict[str, Any], span: _ir_core.Span) -> _ir_core.Expr:
    """Build one vf op call: fold const kwargs, then wrap the registry TypeError."""
    _unwrap_const_kwargs(kwargs)
    try:
        return _ir_core.create_op_call(full_name, list(args), kwargs, span)
    except TypeError as exc:
        raise InvalidType(str(exc), span=span, parser_retry=True) from exc


def _make_vf_builder(full_name: str) -> Callable:
    """Shared value-facing builder for one uniformly-shaped ``vf.<op>``."""
    def builder(self, *args: Any, span: _ir_core.Span, **kwargs: Any) -> _ir_core.Expr:
        return _make_vf_call(full_name, args, kwargs, span)

    return builder


def _make_vf_scalar_builder(full_name: str) -> Callable:
    """Builder of the scalar-value family: destination-dtype range check."""
    build = _make_vf_builder(full_name)

    def builder(self, *args: Any, span: _ir_core.Span, **kwargs: Any) -> _ir_core.Expr:
        result = build(self, *args, span=span, **kwargs)
        _check_vf_scalar_operands(full_name[3:], list(args), getattr(result.type, "dtype", None), span)
        return result

    return builder


_IMPLICIT_DST_ACC = {
    "axpy": "axpy_acc",
    "mul_add_dst": "mul_add_dst_acc",
    "histograms": "histogram_acc",
    "mul_dst_add": "mul_dst_add_acc",
    "move": "move_acc",
}


def _make_implicit_acc_builder(full_name: str) -> Callable:
    """Build an implicit-dst call, routing rebound assignments to ``*_acc``.

    The implicit form is the ``*_acc`` form minus its dst argument. When the
    assignment target already binds a register, that register supplies the
    accumulate source, so the call is rewritten to ``op_acc(existing, *args)``.
    ``vf.move`` only merges in the masked (2-source) form; an unmasked copy
    overwrites every lane, so it stays a plain ``vf.move`` on a rebind.
    """
    acc_name = "vf." + _IMPLICIT_DST_ACC[full_name[3:]]

    def builder(self, *args: Any, span: _ir_core.Span, **kwargs: Any) -> _ir_core.Expr:
        existing = None
        target = getattr(self, "current_target_name", None)
        if target:
            existing = self.scope_manager.lookup_var_bounded(target)
        existing_type = getattr(existing, "type", None)
        inject = isinstance(existing_type, _ir_core.RegType) and existing_type.regtype in (
            _ir_core.RegKind.VecReg,
            _ir_core.RegKind.PredReg,
        )
        if full_name == "vf.move" and len(args) != 2:
            inject = False
        if inject:
            result = _make_vf_call(acc_name, (existing,) + tuple(args), kwargs, span)
        else:
            result = _make_vf_call(full_name, args, kwargs, span)
        if full_name == "vf.axpy":
            _check_vf_scalar_operands("axpy", list(args), _infer_src_dtype(list(args)), span)
        return result

    return builder


@op_impl("vf.muls_cast")
def _build_vf_muls_cast(self, *args: Any, span: _ir_core.Span, **kwargs: Any) -> _ir_core.Expr:
    result = _make_vf_call("vf.muls_cast", args, kwargs, span)
    _check_vf_scalar_operands("muls_cast", list(args), _infer_src_dtype(list(args)), span)
    return result


_SCALAR_VALUE_OPS: frozenset[str] = frozenset(
    {
        "full", "muls", "adds", "mins", "maxs", "leaky_relu", "axpy_acc",
        "eq", "ne", "lt", "gt", "le", "ge", "arange",
    }
)

_PLAIN_VF_OPS: frozenset[str] = frozenset(
    {
        "add", "sub", "mul", "div", "max", "min", "and_", "or_", "xor", "abs_sub",
        "select", "shift_left", "shift_right", "prelu", "ln", "log", "exp", "abs",
        "not_", "sqrt", "relu", "neg", "pair_reduce_sum", "squeeze", "truncate",
        "astype", "log2", "log10", "reduce_sum", "reduce_max", "reduce_min",
        "exp_sub", "mul_dst_add_acc", "mul_add_dst_acc", "histogram_acc", "pack", "unpack",
        "unsqueeze", "load_align", "load", "load_unalign", "gather",
        "move_acc", "bit_cast",
        "interleave", "de_interleave", "mull", "addc", "subc",
        "store_align", "store_unalign", "store_unalign_post", "squeeze_store_unalign",
        "squeeze_store_unalign_post", "scatter", "store",
        "mem_bar", "clear_spr", "load_unalign_pre", "load_unalign_init",
        "create_mask", "update_mask", "get_mask_spr", "mask_gen_with_reg_tensor",
        "create_addr_reg", "unalign_reg_for_store",
    }
)

for _vf_name in _PLAIN_VF_OPS:
    op_impl(f"vf.{_vf_name}")(_make_vf_builder(f"vf.{_vf_name}"))
for _vf_name in _SCALAR_VALUE_OPS:
    op_impl(f"vf.{_vf_name}")(_make_vf_scalar_builder(f"vf.{_vf_name}"))
for _implicit_name in _IMPLICIT_DST_ACC:
    op_impl(f"vf.{_implicit_name}")(_make_implicit_acc_builder(f"vf.{_implicit_name}"))
