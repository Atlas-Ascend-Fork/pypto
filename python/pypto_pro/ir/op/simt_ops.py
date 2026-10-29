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

"""IR builders and value-facing implementations for the PyPTO Pro SIMT frontend."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from typing import Any

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call, Expr, Function, Span

from ..._errors import (
    CommonExternal,
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NotSupported,
    error_class_of_spec_message,
)
from .._utils import _get_span_or_capture
from ._op_registry import op_impl, op_impl_ast

_DIM3_FIELDS = ("x", "y", "z")
_DIM3_CONTEXT_NAME = "dim3_context"


def _create_simt_call(op_name: str, *operands: Expr, span: Span | None) -> Call:
    """Create an IR call for a SIMT operation without attributes."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(f"simt.{op_name}", list(operands), {}, actual_span)


def _make_dim3_components(op_name: str, span: Span) -> list[Call]:
    return [_ir_core.create_op_call(op_name, [], {"axis": axis}, span) for axis in range(len(_DIM3_FIELDS))]


def _make_dim3_context(op_name: str, span: Span | None = None) -> Expr:
    """Build a three-component SIMT context tuple."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.MakeTuple(_make_dim3_components(op_name, actual_span), actual_span)


def linear_thread_idx(span: Span | None = None) -> Call:
    """Return the x-major flattened thread index within the current block."""
    return _create_simt_call("linear_thread_idx", span=span)


def warp_size(span: Span | None = None) -> Call:
    """Return the target SIMT warp size."""
    return _create_simt_call("warp_size", span=span)


def syncthreads(span: Span | None = None) -> Call:
    """Build a block-wide SIMT thread barrier."""
    return _create_simt_call("syncthreads", span=span)


def threadfence_block(span: Span | None = None) -> Call:
    """Build a block-scoped SIMT memory fence."""
    return _create_simt_call("threadfence_block", span=span)


def threadfence(span: Span | None = None) -> Call:
    """Build a device-scoped SIMT memory fence."""
    return _create_simt_call("threadfence", span=span)


def lane_id(span: Span | None = None) -> Call:
    """Return the current lane index in the warp."""
    return _create_simt_call("lane_id", span=span)


def lanemask_eq(span: Span | None = None) -> Call:
    """Return a mask containing only the current lane bit."""
    return _create_simt_call("lanemask_eq", span=span)


def lanemask_le(span: Span | None = None) -> Call:
    """Return a mask containing lanes up to and including the current lane."""
    return _create_simt_call("lanemask_le", span=span)


def lanemask_lt(span: Span | None = None) -> Call:
    """Return a mask containing lanes below the current lane."""
    return _create_simt_call("lanemask_lt", span=span)


def lanemask_ge(span: Span | None = None) -> Call:
    """Return a mask containing lanes at or above the current lane."""
    return _create_simt_call("lanemask_ge", span=span)


def lanemask_gt(span: Span | None = None) -> Call:
    """Return a mask containing lanes above the current lane."""
    return _create_simt_call("lanemask_gt", span=span)


def warp_all(predicate: Expr, span: Span | None = None) -> Call:
    """Return whether every active lane satisfies predicate."""
    return _create_simt_call("warp_all", predicate, span=span)


def warp_any(predicate: Expr, span: Span | None = None) -> Call:
    """Return whether any active lane satisfies predicate."""
    return _create_simt_call("warp_any", predicate, span=span)


def warp_ballot(predicate: Expr, span: Span | None = None) -> Call:
    """Return the active-lane predicate bit mask."""
    return _create_simt_call("warp_ballot", predicate, span=span)


def warp_active_mask(span: Span | None = None) -> Call:
    """Return the currently active lane mask."""
    return _create_simt_call("warp_active_mask", span=span)


def warp_shfl(value: Expr, src_lane: Expr, width: Expr, span: Span | None = None) -> Call:
    """Read a scalar from one source lane in the logical subgroup."""
    return _create_simt_call("warp_shfl", value, src_lane, width, span=span)


def warp_shfl_up(value: Expr, delta: Expr, width: Expr, span: Span | None = None) -> Call:
    """Read a scalar from a lower-numbered lane in the logical subgroup."""
    return _create_simt_call("warp_shfl_up", value, delta, width, span=span)


def warp_shfl_down(value: Expr, delta: Expr, width: Expr, span: Span | None = None) -> Call:
    """Read a scalar from a higher-numbered lane in the logical subgroup."""
    return _create_simt_call("warp_shfl_down", value, delta, width, span=span)


def warp_shfl_xor(value: Expr, lane_mask: Expr, width: Expr, span: Span | None = None) -> Call:
    """Read a scalar from the lane selected by XOR with lane_mask."""
    return _create_simt_call("warp_shfl_xor", value, lane_mask, width, span=span)


def warp_reduce_add(value: Expr, span: Span | None = None) -> Call:
    """Reduce a scalar sum across active lanes."""
    return _create_simt_call("warp_reduce_add", value, span=span)


def warp_reduce_max(value: Expr, span: Span | None = None) -> Call:
    """Reduce a scalar maximum across active lanes."""
    return _create_simt_call("warp_reduce_max", value, span=span)


def warp_reduce_min(value: Expr, span: Span | None = None) -> Call:
    """Reduce a scalar minimum across active lanes."""
    return _create_simt_call("warp_reduce_min", value, span=span)


def cast(
    value: Expr,
    dtype: _ir_core.DataType,
    mode: _ir_core.RoundMode = _ir_core.RoundMode.CAST_NONE,
    span: Span | None = None,
) -> Call:
    """Build a registered SIMT scalar-cast operation."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.cast",
        [value],
        {"target_type": dtype, "mode": mode},
        actual_span,
    )


def bitcast(value: Expr, dtype: _ir_core.DataType, span: Span | None = None) -> Call:
    """Build a registered SIMT scalar bit-reinterpretation operation."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "simt.bitcast",
        [value],
        {"target_type": dtype},
        actual_span,
    )


def abs(value: Expr, span: Span | None = None) -> Call:
    """Build a supported scalar absolute-value call."""
    return _create_simt_call("abs", value, span=span)


def min(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype scalar minimum call."""
    return _create_simt_call("min", lhs, rhs, span=span)


def max(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype scalar maximum call."""
    return _create_simt_call("max", lhs, rhs, span=span)


def sqrt(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar square-root call."""
    return _create_simt_call("sqrt", value, span=span)


def rsqrt(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar reciprocal-square-root call."""
    return _create_simt_call("rsqrt", value, span=span)


def exp(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar exponential call."""
    return _create_simt_call("exp", value, span=span)


def exp2(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar base-two exponential call."""
    return _create_simt_call("exp2", value, span=span)


def log(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar natural-logarithm call."""
    return _create_simt_call("log", value, span=span)


def log2(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar base-two logarithm call."""
    return _create_simt_call("log2", value, span=span)


def log1p(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar log-one-plus call."""
    return _create_simt_call("log1p", value, span=span)


def sin(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar sine call."""
    return _create_simt_call("sin", value, span=span)


def cos(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar cosine call."""
    return _create_simt_call("cos", value, span=span)


def tanh(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar hyperbolic-tangent call."""
    return _create_simt_call("tanh", value, span=span)


def rint(value: Expr, *, dtype: _ir_core.DataType | None = None, span: Span | None = None) -> Call:
    """Build a scalar round-to-nearest-even call with an optional result dtype."""
    actual_span = _get_span_or_capture(span)
    target_type = value.type.dtype if dtype is None else dtype
    return _ir_core.create_op_call("simt.rint", [value], {"target_type": target_type}, actual_span)


def round(value: Expr, *, dtype: _ir_core.DataType | None = None, span: Span | None = None) -> Call:
    """Build a scalar round-half-away-from-zero call with an optional result dtype."""
    actual_span = _get_span_or_capture(span)
    target_type = value.type.dtype if dtype is None else dtype
    return _ir_core.create_op_call("simt.round", [value], {"target_type": target_type}, actual_span)


def floor(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar floor call."""
    return _create_simt_call("floor", value, span=span)


def ceil(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar ceiling call."""
    return _create_simt_call("ceil", value, span=span)


def trunc(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar truncate-toward-zero call."""
    return _create_simt_call("trunc", value, span=span)


def isnan(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar NaN-classification call."""
    return _create_simt_call("isnan", value, span=span)


def isinf(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar infinity-classification call."""
    return _create_simt_call("isinf", value, span=span)


def isfinite(value: Expr, span: Span | None = None) -> Call:
    """Build an FP16 or FP32 scalar finiteness-classification call."""
    return _create_simt_call("isfinite", value, span=span)


def popcount(value: Expr, span: Span | None = None) -> Call:
    """Build an unsigned scalar population count with an INT32 result."""
    return _create_simt_call("popcount", value, span=span)


def mul_hi(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype integer multiply-high call."""
    return _create_simt_call("mul_hi", lhs, rhs, span=span)


def exp10(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar base-ten exponential call."""
    return _create_simt_call("exp10", value, span=span)


def log10(value: Expr, span: Span | None = None) -> Call:
    """Build a floating-point scalar base-ten logarithm call."""
    return _create_simt_call("log10", value, span=span)


def rcp(value: Expr, span: Span | None = None) -> Call:
    """Build an FP16 or BF16 scalar reciprocal call."""
    return _create_simt_call("rcp", value, span=span)


def tan(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar tangent call."""
    return _create_simt_call("tan", value, span=span)


def atan(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar arc-tangent call."""
    return _create_simt_call("atan", value, span=span)


def expm1(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar exponential-minus-one call."""
    return _create_simt_call("expm1", value, span=span)


def logb(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar unbiased-exponent call."""
    return _create_simt_call("logb", value, span=span)


def cosh(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar hyperbolic-cosine call."""
    return _create_simt_call("cosh", value, span=span)


def acos(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar arc-cosine call."""
    return _create_simt_call("acos", value, span=span)


def sinh(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar hyperbolic-sine call."""
    return _create_simt_call("sinh", value, span=span)


def asin(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar arc-sine call."""
    return _create_simt_call("asin", value, span=span)


def cbrt(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scalar cube-root call."""
    return _create_simt_call("cbrt", value, span=span)


def max_nan(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP16 or BF16 maximum call that propagates NaN."""
    return _create_simt_call("max_nan", lhs, rhs, span=span)


def min_nan(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP16 or BF16 minimum call that propagates NaN."""
    return _create_simt_call("min_nan", lhs, rhs, span=span)


def atan2(y: Expr, x: Expr, span: Span | None = None) -> Call:
    """Build an FP32 quadrant-aware arc-tangent call."""
    return _create_simt_call("atan2", y, x, span=span)


def copysign(magnitude: Expr, sign: Expr, span: Span | None = None) -> Call:
    """Build an FP32 sign-selection call."""
    return _create_simt_call("copysign", magnitude, sign, span=span)


def nextafter(value: Expr, direction: Expr, span: Span | None = None) -> Call:
    """Build an FP32 next-representable-value call."""
    return _create_simt_call("nextafter", value, direction, span=span)


def tanpi(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 tangent-of-pi-multiple call."""
    return _create_simt_call("tanpi", value, span=span)


def atanh(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 inverse-hyperbolic-tangent call."""
    return _create_simt_call("atanh", value, span=span)


def cospi(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 cosine-of-pi-multiple call."""
    return _create_simt_call("cospi", value, span=span)


def acosh(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 inverse-hyperbolic-cosine call."""
    return _create_simt_call("acosh", value, span=span)


def sinpi(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 sine-of-pi-multiple call."""
    return _create_simt_call("sinpi", value, span=span)


def asinh(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 inverse-hyperbolic-sine call."""
    return _create_simt_call("asinh", value, span=span)


def rcbrt(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 reciprocal-cube-root call."""
    return _create_simt_call("rcbrt", value, span=span)


def ilogb(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32-to-INT32 unbiased-exponent call."""
    return _create_simt_call("ilogb", value, span=span)


def signbit(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32-to-INT32 sign-bit call."""
    return _create_simt_call("signbit", value, span=span)


def fmod(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP32 remainder call with a quotient truncated toward zero."""
    return _create_simt_call("fmod", lhs, rhs, span=span)


def fma(lhs: Expr, rhs: Expr, addend: Expr, span: Span | None = None) -> Call:
    """Build a same-dtype floating-point fused multiply-add call."""
    return _create_simt_call("fma", lhs, rhs, addend, span=span)


def atomic_add(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic add on one element of a SIMT Tile or Tensor."""
    return _create_simt_call("atomic_add", container, offset, value, span=span)


def atomic_sub(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic subtract on one element of a SIMT Tile or Tensor."""
    return _create_simt_call("atomic_sub", container, offset, value, span=span)


def atomic_exch(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic exchange on one element of a SIMT Tile or Tensor."""
    return _create_simt_call("atomic_exch", container, offset, value, span=span)


def atomic_max(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic maximum on one element of a SIMT Tile or Tensor."""
    return _create_simt_call("atomic_max", container, offset, value, span=span)


def atomic_min(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic minimum on one element of a SIMT Tile or Tensor."""
    return _create_simt_call("atomic_min", container, offset, value, span=span)


def atomic_inc(container: Expr, offset: Expr, limit: Expr, span: Span | None = None) -> Call:
    """Build an atomic wrapping increment on one SIMT counter element."""
    return _create_simt_call("atomic_inc", container, offset, limit, span=span)


def atomic_dec(container: Expr, offset: Expr, limit: Expr, span: Span | None = None) -> Call:
    """Build an atomic wrapping decrement on one SIMT counter element."""
    return _create_simt_call("atomic_dec", container, offset, limit, span=span)


def atomic_cas(
    container: Expr,
    offset: Expr,
    compare: Expr,
    value: Expr,
    span: Span | None = None,
) -> Call:
    """Build an atomic compare-and-swap on one SIMT element."""
    return _create_simt_call("atomic_cas", container, offset, compare, value, span=span)


def atomic_and(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic bitwise AND on one SIMT element."""
    return _create_simt_call("atomic_and", container, offset, value, span=span)


def atomic_or(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic bitwise OR on one SIMT element."""
    return _create_simt_call("atomic_or", container, offset, value, span=span)


def atomic_xor(container: Expr, offset: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an atomic bitwise XOR on one SIMT element."""
    return _create_simt_call("atomic_xor", container, offset, value, span=span)


def launch(
    callee: Function,
    *,
    threads: Expr | tuple[Expr, ...],
    args: Sequence[Expr],
    span: Span | None = None,
) -> Call:
    """Build a SIMT launch call.

    Normal DSL source is parsed specially so the callee Function is retained in
    the enclosing Program. This builder is primarily useful for direct IR tests.
    """
    actual_span = _get_span_or_capture(span)
    thread_dims = list(threads) if isinstance(threads, tuple) else [threads]
    if not 1 <= len(thread_dims) <= 3:
        raise InvalidShape("SIMT launch threads must contain one to three dimensions")
    thread_dims.extend(_ir_core.ConstInt(1, _ir_core.DataType.INT64, actual_span) for _ in range(3 - len(thread_dims)))
    return _ir_core.create_op_call(
        "simt.launch",
        [*thread_dims, *args],
        {"callee": callee.name, "max_threads": callee.get_attr("max_threads")},
        actual_span,
    )


def _parse_simt_launch(parser: Any, call: ast.Call, local_name: str, callee_template: Callable) -> Expr:
    """Parse ``simt_func[threads](args...)`` and lower it to ``simt.launch`` IR."""
    from pypto_pro.language.parser.decorator import get_simt_max_threads

    span = parser.span_tracker.get_span(call)
    if parser._current_func_type in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise NotSupported("Nested SIMT vector-function invocation is not supported", span=span)
    if parser.target != _ir_core.SectionKind.Vector:
        raise InvalidOperation(
            "SIMT vector functions must be invoked inside 'with pl.section_vector():'",
            span=span,
            hint="SIMT vector functions execute on AIV and cannot be invoked from Cube or shared kernel code.",
        )

    if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
        raise InvalidArgument(
            "SIMT vector-function invocation accepts positional arguments only",
            span=span,
            hint="Use: simt_func[threads](arg0, arg1, ...)",
        )

    if get_simt_max_threads(callee_template) is None:
        raise InvalidVal(
            f"'{local_name}' is not a launchable "
            '@pl.vector_function(mode="simt", max_threads=...)',
            span=parser.span_tracker.get_span(call.func.value),
            parser_retry=True,
        )

    threads_node = call.func.slice
    if isinstance(threads_node, ast.Tuple):
        if not 1 <= len(threads_node.elts) <= 3:
            raise InvalidShape(
                "SIMT thread configuration must contain one to three dimensions",
                span=parser.span_tracker.get_span(threads_node),
                hint="Use simt_func[N](...), simt_func[x, y](...), or simt_func[x, y, z](...).",
            )
        thread_nodes = list(threads_node.elts)
    else:
        thread_nodes = [threads_node]
    thread_dims = [parser.parse_expression(dim) for dim in thread_nodes]

    launch_args = [parser.parse_expression(arg) for arg in call.args]
    callee = parser._instantiate_simt_function(local_name, callee_template, launch_args, call.args, span)
    parser._validate_simt_function_arguments(callee, launch_args, call.args, span)
    result = _simt_builder(launch, callee, threads=tuple(thread_dims), args=launch_args, span=span)
    parser.requires_simt = True
    return result


def _simt_builder(builder, *args, span, **kwargs):
    try:
        return builder(*args, **kwargs, span=span)
    except (ValueError, RuntimeError) as error:
        message = str(error)
        if builder in (cast, bitcast):
            message = message.replace(f"simt.{builder.__name__}", f"pl.simt.{builder.__name__}()")
        raise (error_class_of_spec_message(message) or CommonExternal)(
            message, span=span, parser_retry=True,
        ) from error


@op_impl("simt.thread_idx")
def _build_thread_idx(self, *, span: Span) -> Expr:
    return self.make_named_tuple(
        _make_dim3_components("simt.thread_idx", span), _DIM3_FIELDS, span, name=_DIM3_CONTEXT_NAME,
    )


@op_impl("simt.block_dim")
def _build_block_dim(self, *, span: Span) -> Expr:
    return self.make_named_tuple(
        _make_dim3_components("simt.block_dim", span), _DIM3_FIELDS, span, name=_DIM3_CONTEXT_NAME,
    )


@op_impl("simt.block_idx")
def _build_block_idx(self, *, span: Span) -> Expr:
    return self.make_named_tuple(
        _make_dim3_components("simt.block_idx", span), _DIM3_FIELDS, span, name=_DIM3_CONTEXT_NAME,
    )


@op_impl("simt.grid_dim")
def _build_grid_dim(self, *, span: Span) -> Expr:
    return self.make_named_tuple(
        _make_dim3_components("simt.grid_dim", span), _DIM3_FIELDS, span, name=_DIM3_CONTEXT_NAME,
    )


def brev(value: Expr, span: Span | None = None) -> Call:
    """Build a UINT32 or UINT64 bit-reversal call."""
    return _create_simt_call("brev", value, span=span)

def byte_perm(lhs: Expr, rhs: Expr, selector: Expr, span: Span | None = None) -> Call:
    """Build a UINT32 byte-permutation call."""
    return _create_simt_call("byte_perm", lhs, rhs, selector, span=span)

def clz(value: Expr, span: Span | None = None) -> Call:
    """Build an INT32 or INT64 leading-zero count call."""
    return _create_simt_call("clz", value, span=span)

def cyl_bessel_i0(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 order-zero modified-Bessel call."""
    return _create_simt_call("cyl_bessel_i0", value, span=span)

def cyl_bessel_i1(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 order-one modified-Bessel call."""
    return _create_simt_call("cyl_bessel_i1", value, span=span)

def equ(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an unordered FP16 or BF16 equality comparison call."""
    return _create_simt_call("equ", lhs, rhs, span=span)

def erf(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 error-function call."""
    return _create_simt_call("erf", value, span=span)

def erfc(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 complementary-error-function call."""
    return _create_simt_call("erfc", value, span=span)

def erfcinv(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 inverse-complementary-error-function call."""
    return _create_simt_call("erfcinv", value, span=span)

def erfcx(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 scaled-complementary-error-function call."""
    return _create_simt_call("erfcx", value, span=span)

def erfinv(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 inverse-error-function call."""
    return _create_simt_call("erfinv", value, span=span)

def fdim(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP32 positive-difference call."""
    return _create_simt_call("fdim", lhs, rhs, span=span)

def ffs(value: Expr, span: Span | None = None) -> Call:
    """Build an INT32 or INT64 first-set-bit call."""
    return _create_simt_call("ffs", value, span=span)

def fma_relu(lhs: Expr, rhs: Expr, addend: Expr, span: Span | None = None) -> Call:
    """Build an FP16 or BF16 fused multiply-add with ReLU call."""
    return _create_simt_call("fma_relu", lhs, rhs, addend, span=span)

def geu(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an unordered FP16 or BF16 greater-or-equal comparison call."""
    return _create_simt_call("geu", lhs, rhs, span=span)

def gtu(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an unordered FP16 or BF16 greater-than comparison call."""
    return _create_simt_call("gtu", lhs, rhs, span=span)

def hadd(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a signed or unsigned integer average call."""
    return _create_simt_call("hadd", lhs, rhs, span=span)

def hypot(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP32 overflow-resistant hypotenuse call."""
    return _create_simt_call("hypot", lhs, rhs, span=span)

def j0(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 order-zero Bessel-J call."""
    return _create_simt_call("j0", value, span=span)

def j1(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 order-one Bessel-J call."""
    return _create_simt_call("j1", value, span=span)

def jn(order: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an integer-order FP32 Bessel-J call."""
    return _create_simt_call("jn", order, value, span=span)

def ldexp(value: Expr, exponent: Expr, span: Span | None = None) -> Call:
    """Build an FP32/INT32 power-of-two scaling call."""
    return _create_simt_call("ldexp", value, exponent, span=span)

def leu(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an unordered FP16 or BF16 less-or-equal comparison call."""
    return _create_simt_call("leu", lhs, rhs, span=span)

def lgamma(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 log-absolute-Gamma call."""
    return _create_simt_call("lgamma", value, span=span)

def ltu(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an unordered FP16 or BF16 less-than comparison call."""
    return _create_simt_call("ltu", lhs, rhs, span=span)

def mul24(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a signed or unsigned low-24-bit multiplication call."""
    return _create_simt_call("mul24", lhs, rhs, span=span)

def mul_i32_to_i64(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a widening INT32 or UINT32 multiplication call."""
    return _create_simt_call("mul_i32_to_i64", lhs, rhs, span=span)

def norm3d(x: Expr, y: Expr, z: Expr, span: Span | None = None) -> Call:
    """Build an overflow-resistant three-dimensional FP32 norm call."""
    return _create_simt_call("norm3d", x, y, z, span=span)

def norm4d(x: Expr, y: Expr, z: Expr, w: Expr, span: Span | None = None) -> Call:
    """Build an overflow-resistant four-dimensional FP32 norm call."""
    return _create_simt_call("norm4d", x, y, z, w, span=span)

def normcdf(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 standard-normal-CDF call."""
    return _create_simt_call("normcdf", value, span=span)

def normcdfinv(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 inverse-standard-normal-CDF call."""
    return _create_simt_call("normcdfinv", value, span=span)

def pow(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP32 power call."""
    return _create_simt_call("pow", lhs, rhs, span=span)

def remainder(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP32 nearest-even remainder call."""
    return _create_simt_call("remainder", lhs, rhs, span=span)

def rhadd(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build a signed or unsigned upward-rounded integer average call."""
    return _create_simt_call("rhadd", lhs, rhs, span=span)

def rhypot(lhs: Expr, rhs: Expr, span: Span | None = None) -> Call:
    """Build an FP32 reciprocal-hypotenuse call."""
    return _create_simt_call("rhypot", lhs, rhs, span=span)

def rnorm3d(x: Expr, y: Expr, z: Expr, span: Span | None = None) -> Call:
    """Build a reciprocal three-dimensional FP32 norm call."""
    return _create_simt_call("rnorm3d", x, y, z, span=span)

def rnorm4d(x: Expr, y: Expr, z: Expr, w: Expr, span: Span | None = None) -> Call:
    """Build a reciprocal four-dimensional FP32 norm call."""
    return _create_simt_call("rnorm4d", x, y, z, w, span=span)

def sad(lhs: Expr, rhs: Expr, addend: Expr, span: Span | None = None) -> Call:
    """Build a signed or unsigned sum-of-absolute-difference call."""
    return _create_simt_call("sad", lhs, rhs, addend, span=span)

def saturate(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 saturate-to-unit-interval call."""
    return _create_simt_call("saturate", value, span=span)

def scalbn(value: Expr, exponent: Expr, span: Span | None = None) -> Call:
    """Build an FP32 power-of-two scaling call with an INT32 or INT64 exponent."""
    return _create_simt_call("scalbn", value, exponent, span=span)

def tgamma(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 Gamma-function call."""
    return _create_simt_call("tgamma", value, span=span)

def y0(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 order-zero Bessel-Y call."""
    return _create_simt_call("y0", value, span=span)

def y1(value: Expr, span: Span | None = None) -> Call:
    """Build an FP32 order-one Bessel-Y call."""
    return _create_simt_call("y1", value, span=span)

def yn(order: Expr, value: Expr, span: Span | None = None) -> Call:
    """Build an integer-order FP32 Bessel-Y call."""
    return _create_simt_call("yn", order, value, span=span)


def _register_scalar_math_parser(op_name: str, builder: Callable[..., Call]) -> None:
    """Register scalar math after the shared declaration check."""
    @op_impl(op_name)
    def _impl(self, *args: Expr, span: Span, **kwargs) -> Expr:
        return _simt_builder(builder, *args, **kwargs, span=span)


@op_impl("simt.rint")
def _build_simt_rint(self, value: Expr, *, dtype=None, span: Span) -> Expr:
    return _build_rint_round(rint, value, dtype, span)


@op_impl("simt.round")
def _build_simt_round(self, value: Expr, *, dtype=None, span: Span) -> Expr:
    return _build_rint_round(round, value, dtype, span)


def _build_rint_round(builder: Callable[..., Call], value: Expr, dtype, span: Span) -> Expr:
    if isinstance(dtype, Expr) and isinstance(dtype.type, _ir_core.NoneType):
        dtype = None
    return _simt_builder(builder, value, dtype=dtype, span=span)


_register_scalar_math_parser("simt.abs", abs)
_register_scalar_math_parser("simt.min", min)
_register_scalar_math_parser("simt.max", max)
_register_scalar_math_parser("simt.sqrt", sqrt)
_register_scalar_math_parser("simt.rsqrt", rsqrt)
_register_scalar_math_parser("simt.exp", exp)
_register_scalar_math_parser("simt.exp2", exp2)
_register_scalar_math_parser("simt.log", log)
_register_scalar_math_parser("simt.log2", log2)
_register_scalar_math_parser("simt.log1p", log1p)
_register_scalar_math_parser("simt.sin", sin)
_register_scalar_math_parser("simt.cos", cos)
_register_scalar_math_parser("simt.tanh", tanh)
_register_scalar_math_parser("simt.floor", floor)
_register_scalar_math_parser("simt.ceil", ceil)
_register_scalar_math_parser("simt.trunc", trunc)
_register_scalar_math_parser("simt.isnan", isnan)
_register_scalar_math_parser("simt.isinf", isinf)
_register_scalar_math_parser("simt.isfinite", isfinite)
_register_scalar_math_parser("simt.popcount", popcount)
_register_scalar_math_parser("simt.mul_hi", mul_hi)
_register_scalar_math_parser("simt.exp10", exp10)
_register_scalar_math_parser("simt.log10", log10)
_register_scalar_math_parser("simt.rcp", rcp)
_register_scalar_math_parser("simt.tan", tan)
_register_scalar_math_parser("simt.atan", atan)
_register_scalar_math_parser("simt.expm1", expm1)
_register_scalar_math_parser("simt.logb", logb)
_register_scalar_math_parser("simt.cosh", cosh)
_register_scalar_math_parser("simt.acos", acos)
_register_scalar_math_parser("simt.sinh", sinh)
_register_scalar_math_parser("simt.asin", asin)
_register_scalar_math_parser("simt.cbrt", cbrt)
_register_scalar_math_parser("simt.max_nan", max_nan)
_register_scalar_math_parser("simt.min_nan", min_nan)
_register_scalar_math_parser("simt.atan2", atan2)
_register_scalar_math_parser("simt.copysign", copysign)
_register_scalar_math_parser("simt.nextafter", nextafter)
_register_scalar_math_parser("simt.tanpi", tanpi)
_register_scalar_math_parser("simt.atanh", atanh)
_register_scalar_math_parser("simt.cospi", cospi)
_register_scalar_math_parser("simt.acosh", acosh)
_register_scalar_math_parser("simt.sinpi", sinpi)
_register_scalar_math_parser("simt.asinh", asinh)
_register_scalar_math_parser("simt.rcbrt", rcbrt)
_register_scalar_math_parser("simt.ilogb", ilogb)
_register_scalar_math_parser("simt.signbit", signbit)
_register_scalar_math_parser("simt.brev", brev)
_register_scalar_math_parser("simt.byte_perm", byte_perm)
_register_scalar_math_parser("simt.clz", clz)
_register_scalar_math_parser("simt.cyl_bessel_i0", cyl_bessel_i0)
_register_scalar_math_parser("simt.cyl_bessel_i1", cyl_bessel_i1)
_register_scalar_math_parser("simt.equ", equ)
_register_scalar_math_parser("simt.erf", erf)
_register_scalar_math_parser("simt.erfc", erfc)
_register_scalar_math_parser("simt.erfcinv", erfcinv)
_register_scalar_math_parser("simt.erfcx", erfcx)
_register_scalar_math_parser("simt.erfinv", erfinv)
_register_scalar_math_parser("simt.fdim", fdim)
_register_scalar_math_parser("simt.ffs", ffs)
_register_scalar_math_parser("simt.fma_relu", fma_relu)
_register_scalar_math_parser("simt.geu", geu)
_register_scalar_math_parser("simt.gtu", gtu)
_register_scalar_math_parser("simt.hadd", hadd)
_register_scalar_math_parser("simt.hypot", hypot)
_register_scalar_math_parser("simt.j0", j0)
_register_scalar_math_parser("simt.j1", j1)
_register_scalar_math_parser("simt.jn", jn)
_register_scalar_math_parser("simt.ldexp", ldexp)
_register_scalar_math_parser("simt.leu", leu)
_register_scalar_math_parser("simt.lgamma", lgamma)
_register_scalar_math_parser("simt.ltu", ltu)
_register_scalar_math_parser("simt.mul24", mul24)
_register_scalar_math_parser("simt.mul_i32_to_i64", mul_i32_to_i64)
_register_scalar_math_parser("simt.norm3d", norm3d)
_register_scalar_math_parser("simt.norm4d", norm4d)
_register_scalar_math_parser("simt.normcdf", normcdf)
_register_scalar_math_parser("simt.normcdfinv", normcdfinv)
_register_scalar_math_parser("simt.pow", pow)
_register_scalar_math_parser("simt.remainder", remainder)
_register_scalar_math_parser("simt.rhadd", rhadd)
_register_scalar_math_parser("simt.rhypot", rhypot)
_register_scalar_math_parser("simt.rnorm3d", rnorm3d)
_register_scalar_math_parser("simt.rnorm4d", rnorm4d)
_register_scalar_math_parser("simt.sad", sad)
_register_scalar_math_parser("simt.saturate", saturate)
_register_scalar_math_parser("simt.scalbn", scalbn)
_register_scalar_math_parser("simt.tgamma", tgamma)
_register_scalar_math_parser("simt.y0", y0)
_register_scalar_math_parser("simt.y1", y1)
_register_scalar_math_parser("simt.yn", yn)
_register_scalar_math_parser("simt.fmod", fmod)
_register_scalar_math_parser("simt.fma", fma)
_register_scalar_math_parser("simt.warp_reduce_add", warp_reduce_add)
_register_scalar_math_parser("simt.warp_reduce_max", warp_reduce_max)
_register_scalar_math_parser("simt.warp_reduce_min", warp_reduce_min)


def _parse_warp_scalar(
    parser: Any,
    op_name: str,
    argument_name: str,
    node: ast.expr,
    literal_dtype: _ir_core.DataType | None = None,
) -> Expr:
    """Parse one warp scalar, contextually typing integer controls."""

    argument_span = parser.span_tracker.get_span(node)
    literal_value = _numeric_literal_value(node)
    if literal_dtype is not None and literal_value is None:
        known, evaluated_value = parser.expr_evaluator.try_eval_expr(node)
        if known and type(evaluated_value) in (int, float):
            literal_value = evaluated_value
    if literal_dtype is not None and literal_value is not None:
        if type(literal_value) is not int:
            raise InvalidType(
                f"pl.{op_name}() {argument_name} must be an integer scalar",
                span=argument_span,
                parser_retry=True,
            )
        result = parser._make_scalar_constant(literal_value, literal_dtype, argument_span)
    else:
        result = parser.parse_expression(node)

    if not isinstance(result, Expr) or not isinstance(result.type, _ir_core.ScalarType):
        raise InvalidType(
            f"pl.{op_name}() {argument_name} must be a scalar expression",
            span=argument_span,
            hint="Subscript a Tile or Tensor first to obtain one scalar element.",
            parser_retry=True,
        )
    return result


def _parse_warp_predicate_call(
    parser: Any,
    op_name: str,
    builder: Callable[..., Call],
    call: ast.Call,
) -> Expr:
    """Parse one warp vote operation."""

    span = parser.span_tracker.get_span(call)
    predicate_node = _simt_argument_node(call, 0, "predicate")
    predicate = _parse_warp_scalar(parser, op_name, "predicate", predicate_node, _ir_core.DataType.INT32)
    return _simt_builder(builder, predicate, span=span)


def _register_warp_predicate_parser(op_name: str, builder: Callable[..., Call]) -> None:
    """Register one warp vote operation."""

    # Keep AST registration: evaluate the whole predicate expression before
    # constructing INT32, so wide intermediate literals avoid default IR range checks.
    @op_impl_ast(op_name)
    def _parser_impl(parser: Any, call: ast.Call) -> Expr:
        return _parse_warp_predicate_call(parser, op_name, builder, call)


# Predicate constants use INT32 after whole-expression evaluation.
_register_warp_predicate_parser("simt.warp_all", warp_all)
# Predicate constants use INT32 after whole-expression evaluation.
_register_warp_predicate_parser("simt.warp_any", warp_any)
# Predicate constants use INT32 after whole-expression evaluation.
_register_warp_predicate_parser("simt.warp_ballot", warp_ballot)


def _parse_warp_shuffle_call(
    parser: Any,
    op_name: str,
    builder: Callable[..., Call],
    control_name: str,
    control_dtype: _ir_core.DataType,
    call: ast.Call,
) -> Expr:
    """Parse one warp shuffle operation, including its optional width."""

    span = parser.span_tracker.get_span(call)
    width_keyword = next((keyword for keyword in call.keywords if keyword.arg == "width"), None)

    value_node = _simt_argument_node(call, 0, "value")
    control_node = _simt_argument_node(call, 1, control_name)
    value = _parse_warp_scalar(parser, op_name, "value", value_node)
    control = _parse_warp_scalar(parser, op_name, control_name, control_node, control_dtype)
    if len(call.args) == 3:
        width = _parse_warp_scalar(parser, op_name, "width", call.args[2], _ir_core.DataType.INT32)
    elif width_keyword is not None:
        width = _parse_warp_scalar(parser, op_name, "width", width_keyword.value, _ir_core.DataType.INT32)
    else:
        width = _ir_core.ConstInt(32, _ir_core.DataType.INT32, span)

    return _simt_builder(builder, value, control, width, span=span)


def _register_warp_shuffle_parser(
    op_name: str,
    builder: Callable[..., Call],
    control_name: str,
    control_dtype: _ir_core.DataType,
) -> None:
    """Register one warp shuffle operation."""

    # Keep AST registration: lane/delta/width literals need INT32/UINT32 after
    # whole-expression evaluation rather than the default scalar dtype.
    @op_impl_ast(op_name)
    def _parser_impl(parser: Any, call: ast.Call) -> Expr:
        return _parse_warp_shuffle_call(parser, op_name, builder, control_name, control_dtype, call)


# src_lane and width constants use INT32 after whole-expression evaluation.
_register_warp_shuffle_parser("simt.warp_shfl", warp_shfl, "src_lane", _ir_core.DataType.INT32)
# delta constants use UINT32 and width uses INT32 after whole-expression evaluation.
_register_warp_shuffle_parser("simt.warp_shfl_up", warp_shfl_up, "delta", _ir_core.DataType.UINT32)
# delta constants use UINT32 and width uses INT32 after whole-expression evaluation.
_register_warp_shuffle_parser("simt.warp_shfl_down", warp_shfl_down, "delta", _ir_core.DataType.UINT32)
# lane_mask and width constants use INT32 after whole-expression evaluation.
_register_warp_shuffle_parser("simt.warp_shfl_xor", warp_shfl_xor, "lane_mask", _ir_core.DataType.INT32)


@op_impl("simt.cast")
def _build_cast(self, value: Expr, dtype, mode=_ir_core.RoundMode.CAST_NONE, *, span: Span) -> Expr:
    return _simt_builder(cast, value, dtype, mode, span=span)


@op_impl("simt.bitcast")
def _build_bitcast(self, value: Expr, dtype, *, span: Span) -> Expr:
    return _simt_builder(bitcast, value, dtype, span=span)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_add")
def _parse_atomic_add(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_add", atomic_add, ("value",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_sub")
def _parse_atomic_sub(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_sub", atomic_sub, ("value",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_exch")
def _parse_atomic_exch(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_exch", atomic_exch, ("value",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_max")
def _parse_atomic_max(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_max", atomic_max, ("value",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_min")
def _parse_atomic_min(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_min", atomic_min, ("value",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_inc")
def _parse_atomic_inc(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_inc", atomic_inc, ("limit",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_dec")
def _parse_atomic_dec(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_dec", atomic_dec, ("limit",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; compare/value literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_cas")
def _parse_atomic_cas(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_cas", atomic_cas, ("compare", "value"), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_and")
def _parse_atomic_and(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_and", atomic_and, ("value",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_or")
def _parse_atomic_or(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_or", atomic_or, ("value",), call)


# Keep AST registration: target must remain a container/index reference, not a
# scalar load; operand literals are constructed in the target element dtype.
@op_impl_ast("simt.atomic_xor")
def _parse_atomic_xor(self, call: ast.Call) -> Expr:
    return _parse_atomic_call(self, "simt.atomic_xor", atomic_xor, ("value",), call)


def _numeric_literal_value(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)
    ):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def _simt_argument_node(call: ast.Call, position: int, name: str) -> ast.expr:
    """Return one argument node after the public declaration has bound the call."""

    if position < len(call.args):
        return call.args[position]
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _parse_atomic_operand(
    parser: Any,
    op_name: str,
    operand_name: str,
    operand_node: ast.expr,
    dtype: _ir_core.DataType,
) -> Expr:
    """Parse one atomic operand, contextually typing numeric literals."""

    operand_span = parser.span_tracker.get_span(operand_node)
    literal_value = _numeric_literal_value(operand_node)
    if literal_value is not None:
        if dtype.is_int() and type(literal_value) is not int:
            raise InvalidType(
                f"pl.{op_name}() requires an integer {operand_name} for target dtype {dtype}",
                span=operand_span,
                hint=f"Use an integer literal or a Scalar with dtype {dtype}.",
                parser_retry=True,
            )
        return parser._make_scalar_constant(literal_value, dtype, operand_span)

    return parser.parse_expression(operand_node)


def _parse_atomic_call(
    parser: Any,
    op_name: str,
    builder: Callable[..., Call],
    operand_names: tuple[str, ...],
    call: ast.Call,
) -> Expr:
    """Parse one ``pl.simt.atomic_*`` call without loading its target."""

    span = parser.span_tracker.get_span(call)

    short_name = op_name[len("simt."):]
    target_node = _simt_argument_node(call, 0, "target")
    if not isinstance(target_node, ast.Subscript):
        raise InvalidVal(
            f"pl.simt.{short_name}() target must be a direct Tile or Tensor subscript",
            span=parser.span_tracker.get_span(target_node),
            hint=f"Use pl.simt.{short_name}(tile[row, col], ...), not a previously loaded Scalar.",
        )
    if isinstance(target_node.slice, ast.Slice) or (
        isinstance(target_node.slice, ast.Tuple)
        and any(isinstance(index, ast.Slice) for index in target_node.slice.elts)
    ):
        raise NotSupported(
            f"pl.simt.{short_name}() target does not support slices",
            span=parser.span_tracker.get_span(target_node),
            hint="Select exactly one element with integer indices.",
        )

    container = parser.parse_expression(target_node.value)
    container_type = container.type if isinstance(container, Expr) else None
    if not isinstance(container_type, (_ir_core.TileType, _ir_core.TensorType)):
        raise InvalidType(
            f"pl.simt.{short_name}() target container must be a Tile or Tensor",
            span=parser.span_tracker.get_span(target_node.value),
            parser_retry=True,
        )

    dtype = container_type.dtype
    target_span = parser.span_tracker.get_span(target_node)
    offset = parser._parse_scalar_subscript_index(container, target_node.slice, target_span)
    operands = [
        _parse_atomic_operand(
            parser,
            op_name,
            operand_name,
            _simt_argument_node(call, index, operand_name),
            dtype,
        )
        for index, operand_name in enumerate(operand_names, start=1)
    ]
    return _simt_builder(builder, container, offset, *operands, span=span)


@op_impl("simt.linear_thread_idx")
def _build_linear_thread_idx(self, *, span: Span):
    return linear_thread_idx(span=span)


@op_impl("simt.warp_size")
def _build_warp_size(self, *, span: Span):
    return warp_size(span=span)


@op_impl("simt.syncthreads")
def _build_syncthreads(self, *, span: Span):
    return syncthreads(span=span)


@op_impl("simt.threadfence_block")
def _build_threadfence_block(self, *, span: Span):
    return threadfence_block(span=span)


@op_impl("simt.threadfence")
def _build_threadfence(self, *, span: Span):
    return threadfence(span=span)


@op_impl("simt.lane_id")
def _build_lane_id(self, *, span: Span):
    return lane_id(span=span)


@op_impl("simt.lanemask_eq")
def _build_lanemask_eq(self, *, span: Span):
    return lanemask_eq(span=span)


@op_impl("simt.lanemask_le")
def _build_lanemask_le(self, *, span: Span):
    return lanemask_le(span=span)


@op_impl("simt.lanemask_lt")
def _build_lanemask_lt(self, *, span: Span):
    return lanemask_lt(span=span)


@op_impl("simt.lanemask_ge")
def _build_lanemask_ge(self, *, span: Span):
    return lanemask_ge(span=span)


@op_impl("simt.lanemask_gt")
def _build_lanemask_gt(self, *, span: Span):
    return lanemask_gt(span=span)


@op_impl("simt.warp_active_mask")
def _build_warp_active_mask(self, *, span: Span):
    return warp_active_mask(span=span)
