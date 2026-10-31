# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call, Expr, Span

from ..._errors import InvalidArgument, InvalidOperation, InvalidShape, InvalidType, InvalidVal
from .._operators import make_binary
from .._utils import (
    _const_int_value,
    _get_span_or_capture,
    _is_int,
    _normalize_expr,
    _to_make_tuple,
)
from ._op_registry import op_impl, op_impl_ast

# kwargs cross the binding as C++ int (32-bit); timeouts are device cycles.
_MAX_TIMEOUT_CYCLES = (1 << 31) - 1


def barrier(
    window: Expr,
    timeout: int | None = None,
    *,
    sync_local: bool = True,
    flush_store: bool = True,
    span: Span | None = None,
) -> Call:
    """``pld.barrier(win[, timeout][, sync_local][, flush_store])`` -> dist.barrier.

    Semantics are documented on :func:`pypto_pro.language.distribute.barrier`.
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call(
        "dist.barrier", [window],
        {
            "timeout_cycles": _timeout_value(timeout, "pld.barrier", actual_span),
            "sync_local": sync_local,
            "flush_store": flush_store,
        },
        actual_span,
    )


@op_impl("dist.barrier")
def _build_barrier(self, window, timeout=None, *, sync_local=None, flush_store=None, span: Span | None = None):
    timeout = _none_value(timeout)
    timeout_cycles = None
    if timeout is not None:
        timeout_cycles = self.require_const_value(
            timeout, "timeout", expects="integer", check=_is_int,
            hint="timeout is a compile-time device-cycle bound; pass a literal or a variable "
            "bound to one, not a runtime value.",
        )
    flags = {}
    for name, value in (("sync_local", sync_local), ("flush_store", flush_store)):
        if value is None:
            continue
        flags[name] = self.require_const_value(
            value, name, expects="bool", check=_is_bool,
            hint=f"{name} is a compile-time flag; pass a True/False literal or a variable bound to one.",
        )
    return barrier(window, timeout_cycles, span=span, **flags)


def _is_bool(value):
    return isinstance(value, bool)


def _none_value(value):
    """An explicit ``None`` argument parses as a NoneType constant, not Python None."""
    if isinstance(getattr(value, "type", None), _ir_core.NoneType):
        return None
    return value


def _parse_op_kwargs(parser, call):
    """Parse ``ch.*`` call keywords; parse-time constants unwrap to Python values."""
    from ...language.parser._expr_evaluator import ExprEvaluator

    kwargs = {}
    for keyword in call.keywords:
        parsed = parser.parse_expression(keyword.value)
        found, value = ExprEvaluator.ir_to_python_value(parsed)
        kwargs[keyword.arg] = value if found else parsed
    return kwargs


def _shape_dim(dim, span):
    """DTensor view dim: compile-time int (folded) or a runtime integer scalar expression."""
    value = _const_int_value(dim)
    if value is not None:
        return value
    dim_type = getattr(dim, "type", None)
    if isinstance(dim_type, _ir_core.ScalarType) and dim_type.dtype.is_int():
        return dim
    raise InvalidType(
        "pld.make_dtensor shape dims must be integers or integer scalar expressions",
        span=span,
    )


def _coord_elements(value, api, arg_name, expected_len=None, span=None):
    """Per-axis coordinates of a parsed offset operand: list/tuple form -> elementwise."""
    try:
        offsets = _to_make_tuple(value, span)
    except InvalidType:
        raise InvalidType(f"{api} {arg_name} must be a list of per-axis coordinates", span=span) from None
    coords = list(offsets.elements)
    if expected_len is not None and len(coords) != expected_len:
        raise InvalidArgument(
            f"{api} {arg_name} has {len(coords)} coordinates, but the DTensor view has {expected_len} axes",
            span=span,
        )
    for coord in coords:
        if isinstance(coord, _ir_core.Expr):
            continue
        raise InvalidType(f"{api} {arg_name} coordinates must be integers or in-kernel expressions", span=span)
    return coords


def _coord_scale(block_idx, dim, span):
    """Tile-block index x static tile extent -> element offset on that axis."""
    return make_binary("mul", _normalize_expr(block_idx, span), _normalize_expr(dim, span), span)


@op_impl("dist.make_dtensor")
def _build_make_dtensor(self, window, offset, shape, dtype, *, span: Span | None = None):
    """``dt = pld.make_dtensor(win, offset, shape, dtype)`` -> parse-time DTensor handle."""
    from ...language.distribute import DTensor

    if not isinstance(window, _ir_core.Var):
        raise InvalidType("pld.make_dtensor win must be a kernel window parameter", span=span)
    if not isinstance(getattr(shape, "type", None), _ir_core.TupleType):
        raise InvalidType("pld.make_dtensor shape must be a list of ints or integer scalar expressions", span=span)
    dims = tuple(_shape_dim(dim, span) for dim in _to_make_tuple(shape, span).elements)
    if not dims or any(_is_int(dim) and dim <= 0 for dim in dims) or len(dims) > 5:
        raise InvalidVal(f"pld.make_dtensor shape must have 1-5 positive dims, got {dims}", span=span)
    if isinstance(getattr(offset, "type", None), _ir_core.TupleType):
        raise InvalidType(
            "pld.make_dtensor offset is a scalar byte offset into the payload, not a coordinate list",
            span=span,
        )
    byte_offset = _int_operand(offset, "pld.make_dtensor", "offset", span)
    if isinstance(byte_offset, _ir_core.ConstInt) and byte_offset.value < 0:
        raise InvalidVal(f"pld.make_dtensor offset must be >= 0 bytes, got {byte_offset.value}", span=span)
    if not isinstance(dtype, _ir_core.DataType):
        raise InvalidType("pld.make_dtensor dtype must be a pl.DT_* constant", span=span)
    return DTensor(window, byte_offset, dims, dtype)


def _order_axes(self, order, api):
    """``order=`` tuple operand -> Python axis list (compile-time), ``None`` passthrough."""
    order = _none_value(order)
    if order is None:
        return None
    return list(
        self.require_const_value(
            order,
            "order",
            expects="integer list",
            check=lambda value: isinstance(value, (list, tuple)) and all(_is_int(axis) for axis in value),
            hint="order selects tensor axes at compile time; pass a constant list "
            "(e.g. order=[1, 3]) or a variable bound to one, not a runtime value.",
        )
    )


_PUT_ATOMIC_DTYPES = (
    _ir_core.DataType.FP32,
    _ir_core.DataType.FP16,
    _ir_core.DataType.BF16,
    _ir_core.DataType.INT32,
    _ir_core.DataType.INT16,
    _ir_core.DataType.INT8,
)


def _put_atomic_mode(atomic_mode, api, dtype, span):
    """``atomic_mode=`` enum -> int for the IR ``atomic`` kwarg (0 = AtomicNone, unset)."""
    if atomic_mode is None:
        return 0
    if not isinstance(atomic_mode, _ir_core.AtomicType):
        raise InvalidType(f"{api} atomic_mode must be a pl.AtomicType member, got {atomic_mode!r}", span=span)
    atomic = int(atomic_mode)
    if atomic == int(_ir_core.AtomicType.AtomicNone):
        return 0
    if dtype not in _PUT_ATOMIC_DTYPES:
        supported = "/".join(str(d) for d in _PUT_ATOMIC_DTYPES)
        raise InvalidType(
            f"{api} atomic_mode=AtomicAdd does not support dtype {dtype}; supported: {supported}", span=span
        )
    return atomic


def _build_transfer(self, tile, dtensor, offset, peer, order, *, op_name, api, is_tile, span, atomic_mode=None):
    """Shared get/put/get_tile/put_tile build: coordinates + order -> one dist.get/dist.put op."""
    from ...language.distribute import DTensor
    from .block_ops import _resolve_order, _static_shape_ints

    order = _order_axes(self, order, api)

    if not isinstance(dtensor, DTensor):
        name = "second" if op_name == "dist.get" else "first"
        raise InvalidType(
            f"{api} {name} argument must be a pld.DTensor built by pld.make_dtensor", span=span
        )
    if not isinstance(tile, _ir_core.Expr) or not isinstance(tile.type, _ir_core.TileType):
        raise InvalidType(f"{api} tile must be an in-kernel tile expression", span=span)
    if tile.type.dtype != dtensor.dtype:
        raise InvalidType(
            f"{api} tile dtype {tile.type.dtype} does not match the DTensor view dtype {dtensor.dtype}",
            span=span,
        )
    if not isinstance(peer, _ir_core.Expr):
        raise InvalidType(f"{api} peer must be an in-kernel expression", span=span)

    view_ndim = len(dtensor.shape)
    tile_ndim = len(tile.type.shape)
    if view_ndim == 1:
        if order is not None:
            raise InvalidArgument(f"{api}: order is not supported for a rank-1 DTensor view", span=span)
        tile_dims, is_transpose = None, False
    else:
        tile_dims, is_transpose = _resolve_order(order, view_ndim, tile_ndim, api)
        if order is None:
            if tile_ndim > view_ndim:
                raise InvalidShape(
                    f"{api}: Tile rank {tile_ndim} exceeds the DTensor view rank {view_ndim}", span=span
                )
        else:
            if len(tile_dims) != tile_ndim:
                raise InvalidShape(
                    f"{api}: order must name one axis per Tile dimension ({tile_ndim}), got {tile_dims}; "
                    "a merged-row order has no single block stride",
                    span=span,
                )
            if len(set(tile_dims)) != len(tile_dims):
                raise InvalidShape(f"{api}: order axes must be unique, got {tile_dims}", span=span)
            if any(axis < 0 or axis >= view_ndim for axis in tile_dims):
                raise InvalidShape(
                    f"{api}: order axis out of range for the DTensor view rank {view_ndim}: {tile_dims}", span=span
                )
    if is_tile:
        try:
            tile_shape = _static_shape_ints(tile.type.shape, api)
        except InvalidType as exc:
            raise InvalidType(f"{api}: tile-block offsets need a static tile shape ({exc})", span=span) from None
        block_offsets = _coord_elements(offset, api, "tile_offsets", expected_len=view_ndim, span=span)
        if tile_dims is None:
            # Rank-1 view: the Tile shape supplies the extent (block.load's access_size rule).
            extent = tile_shape[0] if len(tile_shape) > 1 and tile_shape[1] == 1 else tile_shape[-1]
            coords = [_coord_scale(off, extent, span) for off in block_offsets]
        else:
            coords = [
                _coord_scale(off, tile_shape[tile_dims.index(axis)], span) if axis in tile_dims else off
                for axis, off in enumerate(block_offsets)
            ]
    else:
        coords = _coord_elements(offset, api, "offset", expected_len=view_ndim, span=span)
    # make_dtensor's byte offset selects where the view starts; these element coordinates
    # address into it (the codegen applies base_bytes before the element offset).
    # Compile-time bounds check on constant coordinates (the tail block may still overhang,
    # the runtime valid window clips it — same contract as load).
    for axis, (coord, dim) in enumerate(zip(coords, dtensor.shape)):
        coord_value = _const_int_value(coord)
        dim_value = _const_int_value(dim)
        if coord_value is not None and dim_value is not None and not 0 <= coord_value < dim_value:
            raise InvalidShape(
                f"{api}: offset[{axis}] ({coord_value}) is outside the DTensor view shape[{axis}] ({dim_value})",
                span=span,
            )
    offsets_tuple = _to_make_tuple(coords, span)
    shape_tuple = _to_make_tuple(dtensor.shape, span)

    # No manual mutex emission here: auto-mutex ordering is configured at tile
    # declaration time (buffer parser), and emitting it per transfer would duplicate
    # the get_buf — a self-deadlock on the buffer semaphore (vector-core timeout on NPU).
    kwargs = {"dtype": dtensor.dtype}
    if is_transpose:
        kwargs["is_transpose"] = True
    if tile_dims is not None:
        kwargs["tile_dims"] = tile_dims
    atomic = _put_atomic_mode(atomic_mode, api, dtensor.dtype, span)
    if atomic:
        kwargs["atomic"] = atomic
    return _ir_core.create_op_call(
        op_name,
        [dtensor.window, dtensor.byte_offset, offsets_tuple, shape_tuple, tile, peer],
        kwargs,
        span,
    )


@op_impl("dist.get")
def _build_dist_get(self, tile, dtensor, offset, peer, *, order=None, span: Span | None = None):
    """``pld.get(tile, dtensor, offset, peer)``: tile <- peer's window, element coordinates."""
    return _build_transfer(
        self, tile, dtensor, offset, peer, order, op_name="dist.get", api="pld.get", is_tile=False, span=span
    )


@op_impl("dist.get_tile")
def _build_dist_get_tile(self, tile, dtensor, tile_offsets, peer, *, order=None, span: Span | None = None):
    """``pld.get_tile(tile, dtensor, tile_offsets, peer)``: block-index variant of pld.get."""
    return _build_transfer(
        self, tile, dtensor, tile_offsets, peer, order,
        op_name="dist.get", api="pld.get_tile", is_tile=True, span=span,
    )


@op_impl("dist.put")
def _build_dist_put(self, dtensor, tile, offset, peer, *, order=None, atomic_mode=None, span: Span | None = None):
    """``pld.put(dtensor, tile, offset, peer)``: peer's window <- tile, element coordinates."""
    return _build_transfer(
        self, tile, dtensor, offset, peer, order, op_name="dist.put", api="pld.put", is_tile=False, span=span,
        atomic_mode=atomic_mode,
    )


@op_impl("dist.put_tile")
def _build_dist_put_tile(self, dtensor, tile, tile_offsets, peer, *, order=None, atomic_mode=None,
                         span: Span | None = None):
    """``pld.put_tile(dtensor, tile, tile_offsets, peer)``: block-index variant of pld.put."""
    return _build_transfer(
        self, tile, dtensor, tile_offsets, peer, order,
        op_name="dist.put", api="pld.put_tile", is_tile=True, span=span, atomic_mode=atomic_mode,
    )


def _window_param(window, api, arg_name, span):
    """A parsed operand that must be a kernel window parameter (``pld.Window`` Var)."""
    if not isinstance(window, _ir_core.Var) or not isinstance(window.type, _ir_core.WindowType):
        raise InvalidType(f"{api} {arg_name} must be a kernel window parameter", span=span)
    return window


def _int_operand(value, api, arg_name, span):
    """A parsed operand that must be an integer scalar expression."""
    value_type = getattr(value, "type", None)
    if isinstance(value, _ir_core.Expr) and isinstance(value_type, _ir_core.ScalarType) and value_type.dtype.is_int():
        return value
    raise InvalidType(f"{api} {arg_name} must be an integer or an integer scalar expression", span=span)


def _parse_int_operand(parser, node, api, arg_name):
    """Parse a positional operand that must be an int or an integer scalar expression."""
    return _int_operand(parser.parse_expression(node), api, arg_name, parser.span_tracker.get_span(node))


def _enum_kwarg_to_int(value, api, arg_name, span):
    """Enum member (``pld.NotifyOp.SET``, ``pld.Cmp.GE``) or an int literal -> int."""
    if isinstance(value, _ir_core.ConstInt):
        return int(value.value)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and _is_int(enum_value):
        return enum_value
    raise InvalidType(f"{api} {arg_name} must be an enum value or an int, got {value!r}", span=span)


@op_impl("dist.signal")
def _build_signal(self, window, idx, *, span: Span | None = None):
    """``sig = pld.signal(win, idx)`` -> dist.signal."""
    from ...language.distribute import Signal

    window = _window_param(window, "pld.signal", "win", span)
    slot = _int_operand(idx, "pld.signal", "idx", span)
    if isinstance(slot, _ir_core.ConstInt) and slot.value < 0:
        raise InvalidVal(f"pld.signal idx must be >= 0, got {slot.value}", span=span)
    return Signal(window, slot)


@op_impl("dist.notify")
def _build_notify(self, signal, peer, val=None, *, op=None, span: Span | None = None):
    """``pld.notify(signal, peer, val)`` / local ``pld.notify(signal, val)`` -> dist.notify."""
    from ...language.distribute import NotifyOp, Signal

    val = _none_value(val)
    if not isinstance(signal, Signal):
        raise InvalidType("pld.notify first argument must be a pld.Signal built by pld.signal", span=span)
    if val is None:
        # Local form: the second positional argument is the value; the peer is this
        # rank, read from the window descriptor at runtime.
        val_operand = _int_operand(peer, "pld.notify", "val", span)
        field = _ir_core.ConstInt(0, _ir_core.DataType.INT64, span)
        peer_operand = _ir_core.create_op_call("dist.window_get", [signal.window, field], {}, span)
    else:
        peer_operand = _int_operand(peer, "pld.notify", "peer", span)
        val_operand = _int_operand(val, "pld.notify", "val", span)

    notify_op = _enum_kwarg_to_int(op if op is not None else NotifyOp.SET, "pld.notify", "op", span)
    if notify_op not in (int(NotifyOp.SET), int(NotifyOp.ATOMIC_ADD)):
        raise InvalidVal(
            f"pld.notify op must be NotifyOp.SET or NotifyOp.ATOMIC_ADD, got {notify_op}", span=span
        )

    return _ir_core.create_op_call(
        "dist.notify",
        [signal.window, _normalize_expr(signal.slot, span), val_operand, peer_operand],
        {"notify_op": notify_op},
        span,
    )


@op_impl("dist.notify_all")
def _build_notify_all(self, signal, val, *, op=None, span: Span | None = None):
    """``pld.notify_all(signal, val[, op=])`` -> dist.notify_all."""
    from ...language.distribute import NotifyOp, Signal

    if not isinstance(signal, Signal):
        raise InvalidType("pld.notify_all first argument must be a pld.Signal built by pld.signal", span=span)
    val_operand = _int_operand(val, "pld.notify_all", "val", span)
    notify_op = _enum_kwarg_to_int(op if op is not None else NotifyOp.SET, "pld.notify_all", "op", span)
    if notify_op not in (int(NotifyOp.SET), int(NotifyOp.ATOMIC_ADD)):
        raise InvalidVal(
            f"pld.notify_all op must be NotifyOp.SET or NotifyOp.ATOMIC_ADD, got {notify_op}", span=span
        )
    return _ir_core.create_op_call(
        "dist.notify_all",
        [signal.window, _normalize_expr(signal.slot, span), val_operand],
        {"notify_op": notify_op},
        span,
    )


def _cmp_kwarg(cmp, api, span):
    """``cmp=`` enum/int -> CmpMode integer (0-5)."""
    from ...language.distribute import Cmp

    cmp_mode = _enum_kwarg_to_int(cmp if cmp is not None else Cmp.EQ, api, "cmp", span)
    if not 0 <= cmp_mode <= 5:
        raise InvalidVal(f"{api} cmp must be a pld.Cmp member (0-5), got {cmp_mode}", span=span)
    return cmp_mode


def _timeout_value(timeout, api, span, *, minimum=0):
    """``timeout=`` compile-time cycles (``None`` -> 0 = forever)."""
    timeout = _none_value(timeout)
    if timeout is None:
        return 0
    if isinstance(timeout, _ir_core.ConstInt):
        timeout = int(timeout.value)
    if not _is_int(timeout):
        raise InvalidType(f"{api} timeout must be an int or None, got {timeout!r}", span=span)
    if not minimum <= timeout <= _MAX_TIMEOUT_CYCLES:
        forever = " (None = forever)" if minimum else ""
        raise InvalidVal(
            f"{api} timeout must be in [{minimum}, {_MAX_TIMEOUT_CYCLES}] device cycles"
            f"{forever}, but got {timeout}",
            span=span,
        )
    return timeout


@op_impl("dist.wait")
def _build_wait(self, signal, val, *, cmp=None, timeout=None, span: Span | None = None):
    """``pld.wait(signal, val, cmp=, timeout=)`` -> dist.wait."""
    from ...language.distribute import Signal

    if not isinstance(signal, Signal):
        raise InvalidType("pld.wait first argument must be a pld.Signal built by pld.signal", span=span)
    val_operand = _int_operand(val, "pld.wait", "val", span)
    cmp_mode = _cmp_kwarg(cmp, "pld.wait", span)

    return _ir_core.create_op_call(
        "dist.wait",
        [signal.window, _normalize_expr(signal.slot, span), val_operand],
        {"cmp": cmp_mode, "timeout_cycles": _timeout_value(timeout, "pld.wait", span)},
        span,
    )


def _shape_byte_count(shape, dtype, span):
    total = _normalize_expr(dtype.get_bit(), span)
    for dim in shape:
        total = make_binary("mul", total, _normalize_expr(dim, span), span)
    total = make_binary("floordiv", total, _normalize_expr(8, span), span)
    return total


def _parse_write_scalar(source, node, dtype, span, *, api="ch.write"):
    """Scalar write form: one word cast to the DTensor dtype, inline in the WQE."""
    if dtype not in _ATOMIC_DTYPES:
        supported = "/".join(str(d) for d in _ATOMIC_DTYPES)
        raise InvalidType(
            f"{api} scalar form does not support dtype {dtype}; supported: {supported}", span=span
        )
    source_type = getattr(source, "type", None)
    if isinstance(source, _ir_core.Expr) and isinstance(source_type, _ir_core.ScalarType) and (
        source_type.dtype.is_int() or source_type.dtype.is_float()
    ):
        if dtype.is_int() and source_type.dtype.is_float():
            raise InvalidType(
                f"{api} scalar must be an integer for dtype {dtype}, "
                f"got a {source_type.dtype} scalar",
                span=span,
            )
        return source
    raise InvalidType(
        f"{api} source must be a kernel GM tensor or an int/float scalar, got {ast.unparse(node)}",
        span=span,
    )


class _ChannelCreate:
    """Internal: ``pld.channel`` deferred to the assignment parser, which knows the
    let-binding target (a nested call must not steal the outer name)."""

    __slots__ = ("window", "peer", "ub_addr", "call")

    def __init__(self, window, peer, ub_addr, call):
        self.window = window
        self.peer = peer
        self.ub_addr = ub_addr
        self.call = call


def _require_const_int(parser, node, api, arg_name):
    """Compile-time integer argument (a literal or a folded constant)."""
    span = parser.span_tracker.get_span(node)
    value = parser.parse_expression(node)
    constant = _const_int_value(value)
    if constant is None:
        raise InvalidType(
            f"{api} {arg_name} must be a compile-time integer, got {ast.unparse(node)}", span=span
        )
    return constant


@op_impl_ast("dist.channel")
def _parse_channel(parser, call):
    """``ch = pld.channel(win, buffer[, peer])`` -> assignment-time handle.

    Without ``peer`` the device constructor receives ``-1`` (multi-issuer)."""
    from ...language.distribute import CHANNEL_UB_BYTES

    span = parser.span_tracker.get_span(call)
    if len(call.args) != 2:
        raise InvalidArgument(
            f"pld.channel requires (win, buffer), got {len(call.args)} arguments",
            span=span,
        )

    window = _window_param(parser.parse_expression(call.args[0]), "pld.channel", "win", span)
    ub_addr = _require_const_int(parser, call.args[1], "pld.channel", "buffer")
    if ub_addr < 0 or ub_addr % 64:
        raise InvalidVal(
            f"pld.channel buffer must be a non-negative 64-byte aligned address of at least "
            f"{CHANNEL_UB_BYTES} bytes, got {ub_addr}",
            span=span,
        )

    peer = None
    for kw in call.keywords:  # at most one entry: the declared-param gate rejected the rest
        peer = _parse_int_operand(parser, kw.value, "pld.channel", "peer")
        if isinstance(peer, _ir_core.ConstInt) and peer.value < 0:
            raise InvalidVal(
                f"pld.channel peer must be a destination rank, got {peer.value}; "
                "omit 'peer' for the multi-issuer (per-AIV jetty) form",
                span=span,
            )

    peer_ir = peer if peer is not None else _normalize_expr(-1, span)
    create = _ir_core.create_op_call("dist.channel", [window, peer_ir], {"ub_addr": ub_addr}, span)
    return _ChannelCreate(window, peer, ub_addr, create)


def _channel_dtensor(parser, api, node, channel, span):
    """Parse and validate the DTensor destination of a channel transfer."""
    from ...language.distribute import DTensor

    dtensor = parser.parse_expression(node)
    if not isinstance(dtensor, DTensor):
        raise InvalidType(
            f"{api} destination must be a pld.DTensor built by pld.make_dtensor, got {ast.unparse(node)}",
            span=span,
        )
    if dtensor.window.name != channel.window.name:
        raise InvalidType(f"{api} DTensor must belong to the channel's window", span=span)
    return dtensor


def _channel_peer_operand(parser, call, channel, span):
    """Resolve the transfer's destination peer (per-call arg on multi-issuer channels)."""
    if channel.peer is None:
        from ...language.distribute import DTensor

        if not call.args:
            raise InvalidArgument(
                "ch.write on a peer-less channel requires the destination peer first: "
                "ch.write(peer, dtensor, ...)",
                span=span,
            )
        first = parser.parse_expression(call.args[0])
        if isinstance(first, DTensor):
            raise InvalidArgument(
                "ch.write on a peer-less channel requires the destination peer first: "
                "ch.write(peer, dtensor, ...)",
                span=span,
            )
        return _parse_int_operand(parser, call.args[0], "ch.write", "peer"), 1
    return channel.peer, 0


def _build_channel_write(parser, call, channel):
    """``ch.write([peer,] dtensor, tensor|scalar, submit=, timeout=)`` -> channel_write[_value]."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("submit", "timeout"):
            raise InvalidArgument(f"ch.write does not accept keyword argument '{name}'", span=span)
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.write does not accept star-args", span=span)

    peer, base_arg = _channel_peer_operand(parser, call, channel, span)
    nargs = len(call.args) - base_arg
    if nargs != 2:
        expected = "peer, dtensor, tensor or scalar" if channel.peer is None else "dtensor, tensor or scalar"
        raise InvalidArgument(
            f"ch.write requires ({expected}), got {nargs} arguments after the channel", span=span
        )

    dtensor = _channel_dtensor(parser, "ch.write", call.args[base_arg], channel, span)
    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.write submit must be a compile-time bool, got {submit!r}", span=span)
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.write", span, minimum=1)

    source = parser.parse_expression(call.args[base_arg + 1])
    if isinstance(source, _ir_core.Var) and isinstance(source.type, _ir_core.TensorType):
        if source.type.dtype != dtensor.dtype:
            raise InvalidType(
                f"ch.write source dtype {source.type.dtype} does not match the DTensor view dtype "
                f"{dtensor.dtype}",
                span=span,
            )
        num_bytes_arg = _shape_byte_count(source.type.shape, dtensor.dtype, span)
        return _ir_core.create_op_call(
            "dist.channel_write",
            [channel.token, peer, dtensor.byte_offset, num_bytes_arg, source],
            {"dtype": dtensor.dtype, "submit": submit, "timeout_cycles": timeout},
            span,
        )

    value = _parse_write_scalar(source, call.args[base_arg + 1], dtensor.dtype, span, api="ch.write")
    # The scalar rides inline in the WQE; no MTE3 flush is needed.
    return _ir_core.create_op_call(
        "dist.channel_write_value",
        [channel.token, peer, dtensor.byte_offset, value],
        {"dtype": dtensor.dtype, "submit": submit, "timeout_cycles": timeout},
        span,
    )


def _build_channel_write_notify(parser, call, channel):
    """``ch.write_notify([peer,] dtensor, tensor, signal, val, timeout=)`` -> channel_write_notify."""
    from ...language.distribute import Signal

    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "timeout":
            raise InvalidArgument(f"ch.write_notify does not accept keyword argument '{name}'", span=span)
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.write_notify does not accept star-args", span=span)

    peer, base_arg = _channel_peer_operand(parser, call, channel, span)
    nargs = len(call.args) - base_arg
    if nargs != 4:
        expected = "peer, dtensor, tensor, signal, val" if channel.peer is None else "dtensor, tensor, signal, val"
        raise InvalidArgument(
            f"ch.write_notify requires ({expected}), got {nargs} arguments after the channel",
            span=span,
        )

    dtensor = _channel_dtensor(parser, "ch.write_notify", call.args[base_arg], channel, span)
    source = parser.parse_expression(call.args[base_arg + 1])
    if not isinstance(source, _ir_core.Var) or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            "ch.write_notify tensor must be a kernel GM tensor (the fused WQE reads its source "
            f"from GM; use ch.write for the scalar-word form), got {ast.unparse(call.args[base_arg + 1])}",
            span=span,
        )
    if source.type.dtype != dtensor.dtype:
        raise InvalidType(
            f"ch.write_notify source dtype {source.type.dtype} does not match the DTensor view dtype "
            f"{dtensor.dtype}",
            span=span,
        )
    signal = parser.parse_expression(call.args[base_arg + 2])
    if not isinstance(signal, Signal):
        raise InvalidType("ch.write_notify signal must be a pld.Signal built by pld.signal", span=span)
    if signal.window.name != channel.window.name:
        raise InvalidType("ch.write_notify signal must belong to the channel's window", span=span)
    val = _parse_int_operand(parser, call.args[base_arg + 3], "ch.write_notify", "val")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.write_notify", span, minimum=1)

    num_bytes_arg = _shape_byte_count(source.type.shape, dtensor.dtype, span)
    return _ir_core.create_op_call(
        "dist.channel_write_notify",
        [
            channel.token,
            peer,
            dtensor.byte_offset,
            num_bytes_arg,
            source,
            _normalize_expr(signal.slot, span),
            val,
        ],
        {"dtype": dtensor.dtype, "timeout_cycles": timeout},
        span,
    )


def _build_channel_submit(parser, call, channel):
    """``ch.submit()`` -> dist.channel_submit."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.submit takes no arguments", span=span)
    return _ir_core.create_op_call("dist.channel_submit", [channel.token], {}, span)


def _build_channel_drain(parser, call, channel):
    """``ch.drain(timeout=)`` / ``ch.drain(timeout)`` -> dist.channel_drain."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "timeout":
            raise InvalidArgument(f"ch.drain does not accept keyword argument '{name}'", span=span)
    if len(call.args) > 1:
        raise InvalidArgument("ch.drain accepts at most one positional timeout", span=span)
    if call.args:
        if "timeout" in kwargs_parsed:
            raise InvalidArgument("ch.drain timeout given both positionally and by keyword", span=span)
        kwargs_parsed = {"timeout": _require_const_int(parser, call.args[0], "ch.drain", "timeout")}
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.drain", span, minimum=1)
    return _ir_core.create_op_call(
        "dist.channel_drain", [channel.token], {"timeout_cycles": timeout}, span
    )


def _build_channel_is_exclusive(parser, call, channel):
    """``ch.is_exclusive()`` -> dist.channel_is_exclusive (runtime bool)."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.is_exclusive takes no arguments", span=span)
    return _ir_core.create_op_call("dist.channel_is_exclusive", [channel.token], {}, span)


def parse_channel_method(parser, channel, method, call):
    """Dispatch ``ch.<method>(...)`` (receiver-routed by the parser)."""
    if method == "write":
        return _build_channel_write(parser, call, channel)
    if method == "write_notify":
        return _build_channel_write_notify(parser, call, channel)
    if method == "submit":
        return _build_channel_submit(parser, call, channel)
    if method == "drain":
        return _build_channel_drain(parser, call, channel)
    if method == "is_exclusive":
        return _build_channel_is_exclusive(parser, call, channel)
    raise InvalidOperation(
        f"pld.Channel has no method '{method}'", span=parser.span_tracker.get_span(call)
    )


_ATOMIC_DTYPES = (
    _ir_core.DataType.INT32,
    _ir_core.DataType.UINT32,
    _ir_core.DataType.INT64,
    _ir_core.DataType.UINT64,
    _ir_core.DataType.FP32,
)


def _negate_atomic_expr(expr, dtype, span):
    if isinstance(expr, _ir_core.ConstInt):
        if dtype.is_unsigned():
            # Modular negation keeps the constant representable in the unsigned target;
            # the hardware add then wraps modulo 2^N.
            return _normalize_expr((-expr.value) % (1 << dtype.get_bit()), span, int_dtype=dtype)
        return _normalize_expr(-expr.value, span, int_dtype=dtype if dtype.is_signed() else None)
    if isinstance(expr, _ir_core.ConstFloat):
        return _normalize_expr(-expr.value, span)
    return -expr


def _atomic_value(value, api, arg_name, dtype, span, *, negate=False):
    value_type = getattr(value, "type", None)
    if not isinstance(value, _ir_core.Expr) or not isinstance(value_type, _ir_core.ScalarType) or not (
        value_type.dtype.is_int() or value_type.dtype.is_float()
    ):
        raise InvalidType(f"{api} {arg_name} must be an int/float or a scalar expression", span=span)
    if dtype.is_int() and value_type.dtype.is_float():
        raise InvalidType(
            f"{api} {arg_name} must be an integer for dtype {dtype}, "
            f"got a {value_type.dtype} scalar",
            span=span,
        )
    return _negate_atomic_expr(value, dtype, span) if negate else value


def _atomic_offsets(offset, dtensor, api, span):
    """Offset operand: per-axis coordinate list, or a scalar linear element index."""
    if isinstance(getattr(offset, "type", None), _ir_core.TupleType):
        # One coordinate per view axis; each may be a constant or a runtime expression.
        coords = _coord_elements(offset, api, "offset", expected_len=len(dtensor.shape), span=span)
        return _to_make_tuple(coords, span)
    return _int_operand(offset, api, "offset", span)


def _build_atomic(self, dtensor, peer, offset, values, *, op_name, api, value_names, negate=False, span=None):
    from ...language.distribute import DTensor

    if not isinstance(dtensor, DTensor):
        raise InvalidType(
            f"{api} first argument must be a pld.DTensor built by pld.make_dtensor", span=span
        )
    dtype = dtensor.dtype
    if dtype not in _ATOMIC_DTYPES:
        supported = "/".join(str(d) for d in _ATOMIC_DTYPES)
        raise InvalidType(f"{api} does not support dtype {dtype}; supported: {supported}", span=span)

    peer_operand = _int_operand(peer, api, "peer", span)
    offsets_arg = _atomic_offsets(offset, dtensor, api, span)
    parsed_values = [
        _atomic_value(value, api, name, dtype, span, negate=negate)
        for value, name in zip(values, value_names)
    ]
    shape_tuple = _to_make_tuple(dtensor.shape, span)
    return _ir_core.create_op_call(
        op_name,
        [dtensor.window, dtensor.byte_offset, offsets_arg, shape_tuple, peer_operand, *parsed_values],
        {"dtype": dtype},
        span,
    )


@op_impl("dist.atomic_add")
def _build_atomic_add(self, dtensor, peer, offset, val, *, span: Span | None = None):
    """``old = pld.atomic_add(dtensor, peer, offset, val)`` -> dist.atomic_add."""
    return _build_atomic(
        self, dtensor, peer, offset, [val], op_name="dist.atomic_add", api="pld.atomic_add",
        value_names=("val",), span=span,
    )


@op_impl("dist.atomic_sub")
def _build_atomic_sub(self, dtensor, peer, offset, val, *, span: Span | None = None):
    """``old = pld.atomic_sub(dtensor, peer, offset, val)`` -> dist.atomic_add with ``-val``."""
    return _build_atomic(
        self, dtensor, peer, offset, [val], op_name="dist.atomic_add", api="pld.atomic_sub",
        value_names=("val",), negate=True, span=span,
    )


@op_impl("dist.atomic_exchange")
def _build_atomic_exchange(self, dtensor, peer, offset, val, *, span: Span | None = None):
    """``old = pld.atomic_exchange(dtensor, peer, offset, val)`` -> dist.atomic_exchange."""
    return _build_atomic(
        self, dtensor, peer, offset, [val], op_name="dist.atomic_exchange", api="pld.atomic_exchange",
        value_names=("val",), span=span,
    )


@op_impl("dist.atomic_cas")
def _build_atomic_cas(self, dtensor, peer, offset, expect, desired, *, span: Span | None = None):
    """``old = pld.atomic_cas(dtensor, peer, offset, expect, desired)`` -> dist.atomic_cas."""
    return _build_atomic(
        self, dtensor, peer, offset, [expect, desired], op_name="dist.atomic_cas", api="pld.atomic_cas",
        value_names=("expect", "desired"), span=span,
    )
