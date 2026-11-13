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
    require_const_bool,
    require_const_int,
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
            "timeout_cycles": _timeout_value(timeout, "pld.barrier"),
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
        try:
            timeout_cycles = require_const_int(timeout, subject="timeout")
        except InvalidType as error:
            error.hint = (
                "timeout is a compile-time device-cycle bound; pass a literal or a variable "
                "bound to one, not a runtime value."
            )
            raise
    flags = {}
    for name, value in (("sync_local", sync_local), ("flush_store", flush_store)):
        if value is None:
            continue
        try:
            flags[name] = require_const_bool(value, subject=name)
        except InvalidType as error:
            error.hint = f"{name} is a compile-time flag; pass a True/False literal or a variable bound to one."
            raise
    return barrier(window, timeout_cycles, span=span, **flags)


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


def _ptr_operand(value, api, arg_name, span):
    """Operand that must identify a registered memory region by address.

    A raw ``pl.Ptr`` passes through; a kernel tensor parameter is converted to its
    base pointer. Locals and intermediates are rejected by the device region lookup.
    """
    from .ptr_ops import make_ptr

    if isinstance(value, _ir_core.Expr) and isinstance(value.type, _ir_core.PtrType):
        return value
    if isinstance(value, _ir_core.Var) and isinstance(value.type, _ir_core.TensorType):
        return make_ptr(value, None, span=span)
    raise InvalidType(
        f"{api} {arg_name} must be a kernel tensor parameter registered by "
        "Window.register_memory or a pl.Ptr inside such a region",
    )


@op_impl("dist.remote_ptr")
def _build_remote_ptr(self, window, tensor, peer, *, span: Span | None = None):
    """``ptr = pld.remote_ptr(win, tensor_or_ptr, peer)`` -> typed UB_MEM remote pointer."""
    window = _window_param(window, "pld.remote_ptr", "win")
    peer_operand = _int_operand(peer, "pld.remote_ptr", "peer")
    return _ir_core.create_op_call(
        "dist.remote_ptr",
        [window, _ptr_operand(tensor, "pld.remote_ptr", "first argument", span), peer_operand],
        {},
        span,
    )


def _window_param(window, api, arg_name):
    """A parsed operand that must be a kernel window parameter (``pld.Window`` Var)."""
    if not isinstance(window, _ir_core.Var) or not isinstance(window.type, _ir_core.WindowType):
        raise InvalidType(f"{api} {arg_name} must be a kernel window parameter")
    return window


def _int_operand(value, api, arg_name):
    """A parsed operand that must be an integer scalar expression."""
    value_type = getattr(value, "type", None)
    if isinstance(value, _ir_core.Expr) and isinstance(value_type, _ir_core.ScalarType) and value_type.dtype.is_int():
        return value
    raise InvalidType(f"{api} {arg_name} must be an integer or an integer scalar expression")


def _parse_int_operand(parser, node, api, arg_name):
    """Parse a positional operand that must be an int or an integer scalar expression."""
    return _int_operand(parser.parse_expression(node), api, arg_name)


def _enum_kwarg_to_int(value, api, arg_name):
    """Enum member (``pld.NotifyOp.SET``, ``pld.Cmp.GE``) or an int literal -> int."""
    if isinstance(value, _ir_core.ConstInt):
        return int(value.value)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and _is_int(enum_value):
        return enum_value
    raise InvalidType(f"{api} {arg_name} must be an enum value or an int, got {value!r}")


def _signal_ptr_operand(value, api, arg_name, span, *, allow_tensor, allow_32=True):
    """Signal-word operand: a pointer to an integer slot following its dtype.

    ``notify``/``wait`` accept 32/64-bit words and also accept a local tensor
    parameter (converted to its base pointer); channel notifications take the remote
    pointer from ``remote_ptr`` and require a 64-bit word (the hardware notify value
    is a ``uint64_t``). The framework cannot verify where a pointer came from.
    """
    if allow_tensor:
        ptr = _ptr_operand(value, api, arg_name, span)
    elif isinstance(value, _ir_core.Expr) and isinstance(value.type, _ir_core.PtrType):
        ptr = value
    else:
        raise InvalidType(
            f"{api} {arg_name} must be a remote pointer resolved by ch.remote_ptr(...)")
    dtype = _ptr_dtype(ptr, api)
    if allow_32:
        supported = dtype.is_int() and dtype.get_bit() in (32, 64)
        kinds = "32/64-bit integer signal words"
    else:
        supported = dtype in (_ir_core.DataType.INT64, _ir_core.DataType.UINT64)
        kinds = "64-bit integer signal words"
    if not supported:
        raise InvalidType(f"{api} {arg_name} must point at {kinds}, got {dtype}")
    return ptr


@op_impl("dist.notify")
def _build_notify(self, dst, val, *, op=None, flush_store=None, span: Span | None = None):
    """``pld.notify(dst_ptr_or_tensor, val[, op=, flush_store=])`` -> dist.notify."""
    from ...language.distribute import NotifyOp

    dst_ptr = _signal_ptr_operand(dst, "pld.notify", "dst", span, allow_tensor=True)
    val_operand = _int_operand(val, "pld.notify", "val")
    notify_op = _enum_kwarg_to_int(op if op is not None else NotifyOp.SET, "pld.notify", "op")
    if notify_op not in (int(NotifyOp.SET), int(NotifyOp.ATOMIC_ADD)):
        raise InvalidVal(
            f"pld.notify op must be NotifyOp.SET or NotifyOp.ATOMIC_ADD, got {notify_op}"
        )

    kwargs = {
        "notify_op": notify_op,
        "flush_store": True if flush_store is None else require_const_bool(
            flush_store, subject="pld.notify 'flush_store'",
        ),
    }
    return _ir_core.create_op_call("dist.notify", [dst_ptr, val_operand], kwargs, span)


def _cmp_kwarg(cmp, api):
    """``cmp=`` enum/int -> CmpMode integer (0-5)."""
    from ...language.distribute import Cmp

    cmp_mode = _enum_kwarg_to_int(cmp if cmp is not None else Cmp.EQ, api, "cmp")
    if not 0 <= cmp_mode <= 5:
        raise InvalidVal(f"{api} cmp must be a pld.Cmp member (0-5), got {cmp_mode}")
    return cmp_mode


def _timeout_value(timeout, api, *, minimum=0):
    """``timeout=`` compile-time cycles (``None`` -> 0 = forever)."""
    timeout = _none_value(timeout)
    if timeout is None:
        return 0
    if isinstance(timeout, _ir_core.ConstInt):
        timeout = int(timeout.value)
    if not _is_int(timeout):
        raise InvalidType(f"{api} timeout must be an int or None, got {timeout!r}")
    if not minimum <= timeout <= _MAX_TIMEOUT_CYCLES:
        forever = " (None = forever)" if minimum else ""
        raise InvalidVal(
            f"{api} timeout must be in [{minimum}, {_MAX_TIMEOUT_CYCLES}] device cycles"
            f"{forever}, but got {timeout}",
        )
    return timeout


@op_impl("dist.wait")
def _build_wait(self, src, val, *, cmp=None, timeout=None, span: Span | None = None):
    """``pld.wait(src_ptr_or_tensor, val, cmp=, timeout=)`` -> dist.wait."""
    src_ptr = _signal_ptr_operand(src, "pld.wait", "src", span, allow_tensor=True)
    val_operand = _int_operand(val, "pld.wait", "val")
    cmp_mode = _cmp_kwarg(cmp, "pld.wait")

    return _ir_core.create_op_call(
        "dist.wait",
        [src_ptr, val_operand],
        {"cmp": cmp_mode, "timeout_cycles": _timeout_value(timeout, "pld.wait")},
        span,
    )


def _shape_byte_count(shape, dtype, span):
    total = _normalize_expr(dtype.get_bit(), span)
    for dim in shape:
        total = make_binary("mul", total, _normalize_expr(dim, span), span)
    total = make_binary("floordiv", total, _normalize_expr(8, span), span)
    return total


def _static_shape_byte_count(shape, dtype):
    """Byte size of a fully static shape, or None when any dim is runtime."""
    total = 1
    for dim in shape:
        value = _const_int_value(dim)
        if value is None:
            return None
        total *= value
    return total * dtype.get_bit() // 8


_WRITE_VALUE_DTYPES = (
    _ir_core.DataType.INT32,
    _ir_core.DataType.UINT32,
    _ir_core.DataType.INT64,
    _ir_core.DataType.UINT64,
    _ir_core.DataType.FP32,
)


def _parse_write_scalar(source, node, dtype, *, api="ch.write"):
    """Scalar write form: one word cast to the destination pointer dtype, inline in the WQE."""
    if dtype not in _WRITE_VALUE_DTYPES:
        supported = "/".join(str(d) for d in _WRITE_VALUE_DTYPES)
        raise InvalidType(
            f"{api} scalar form does not support dtype {dtype}; supported: {supported}"
        )
    source_type = getattr(source, "type", None)
    if isinstance(source, _ir_core.Expr) and isinstance(source_type, _ir_core.ScalarType) and (
        source_type.dtype.is_int() or source_type.dtype.is_float()
    ):
        if dtype.is_int() and source_type.dtype.is_float():
            raise InvalidType(
                f"{api} scalar must be an integer for dtype {dtype}, "
                f"got a {source_type.dtype} scalar",
            )
        return source
    raise InvalidType(
        f"{api} source must be a kernel GM tensor or an int/float scalar, got {ast.unparse(node)}",
    )


class _ChannelCreate:
    """Internal: ``pld.channel`` deferred to the assignment parser, which knows the
    let-binding target (a nested call must not steal the outer name)."""

    __slots__ = ("window", "ub_addr", "call")

    def __init__(self, window, ub_addr, call):
        self.window = window
        self.ub_addr = ub_addr
        self.call = call


def _require_const_int(parser, node, api, arg_name):
    """Compile-time integer argument (a literal or a folded constant)."""
    value = parser.parse_expression(node)
    constant = _const_int_value(value)
    if constant is None:
        raise InvalidType(
            f"{api} {arg_name} must be a compile-time integer, got {ast.unparse(node)}"
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
        )

    window = _window_param(parser.parse_expression(call.args[0]), "pld.channel", "win")
    ub_addr = _require_const_int(parser, call.args[1], "pld.channel", "buffer")
    if ub_addr < 0 or ub_addr % 64:
        raise InvalidVal(
            f"pld.channel buffer must be a non-negative 64-byte aligned address of at least "
            f"{CHANNEL_UB_BYTES} bytes, got {ub_addr}",
        )

    peer = None
    for kw in call.keywords:  # at most one entry: the declared-param gate rejected the rest
        peer = _parse_int_operand(parser, kw.value, "pld.channel", "peer")
        if isinstance(peer, _ir_core.ConstInt) and peer.value < 0:
            raise InvalidVal(
                f"pld.channel peer must be a destination rank, got {peer.value}; "
                "omit 'peer' for the multi-issuer (per-AIV jetty) form",
            )

    # ubbuf may be use by simt code, update the max_vec_tile_end
    parser.max_vec_tile_end = max(parser.max_vec_tile_end, ub_addr + CHANNEL_UB_BYTES)

    peer_ir = peer if peer is not None else _normalize_expr(-1, span)
    create = _ir_core.create_op_call("dist.channel", [window, peer_ir], {"ub_addr": ub_addr}, span)
    return _ChannelCreate(window, ub_addr, create)


def _channel_dst_ptr(parser, api, node, span):
    """Parse a channel transfer destination: a Ptr or a registered tensor parameter."""
    return _ptr_operand(parser.parse_expression(node), api, "destination", span)


def _ptr_dtype(ptr, api):
    ptr_type = getattr(ptr, "type", None)
    if not isinstance(ptr_type, _ir_core.PtrType):
        raise InvalidType(f"{api} pointer operand must have PtrType")
    return ptr_type.dtype


def _build_channel_remote_ptr(parser, call, channel):
    """``ch.remote_ptr(tensor_or_ptr)`` -> dist.channel_remote_ptr."""
    span = parser.span_tracker.get_span(call)
    if len(call.args) != 1 or call.keywords:
        raise InvalidArgument("ch.remote_ptr requires exactly one tensor or pointer argument")
    if isinstance(call.args[0], ast.Starred):
        raise InvalidArgument("ch.remote_ptr does not accept star-args")
    dst = _channel_dst_ptr(parser, "ch.remote_ptr", call.args[0], span)
    return _ir_core.create_op_call("dist.channel_remote_ptr", [channel.token, dst], {}, span)


def _build_channel_write(parser, call, channel):
    """``ch.write(dst_ptr, tensor|scalar, submit=, timeout=)`` -> channel_write[_value]."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("submit", "timeout"):
            raise InvalidArgument(f"ch.write does not accept keyword argument '{name}'")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.write does not accept star-args")
    if len(call.args) != 2:
        raise InvalidArgument(
            f"ch.write requires (dst_ptr, tensor or scalar), got {len(call.args)} arguments after the channel",
            hint="Resolve the destination with rptr = ch.remote_ptr(tensor) first; the peer is bound "
                 "when the channel is opened",
        )

    dst = _channel_dst_ptr(parser, "ch.write", call.args[0], span)
    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.write submit must be a compile-time bool, got {submit!r}")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.write", minimum=1)

    source = parser.parse_expression(call.args[1])
    if isinstance(source, _ir_core.Var) and isinstance(source.type, _ir_core.TensorType):
        # The destination pointer is a byte address; the payload dtype and length come
        # from the source tensor.
        num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
        return _ir_core.create_op_call(
            "dist.channel_write",
            [channel.token, dst, num_bytes_arg, source],
            {"submit": submit, "timeout_cycles": timeout},
            span,
        )

    value = _parse_write_scalar(source, call.args[1], _ptr_dtype(dst, "ch.write"), api="ch.write")
    # The scalar rides inline in the WQE; no MTE3 flush is needed.
    return _ir_core.create_op_call(
        "dist.channel_write_value",
        [channel.token, dst, value],
        {"submit": submit, "timeout_cycles": timeout},
        span,
    )


_INDEX_DTYPES = (
    _ir_core.DataType.INT32,
    _ir_core.DataType.INT64,
    _ir_core.DataType.UINT32,
    _ir_core.DataType.UINT64,
)


def _check_dense_stride(type_, stride, api, name, span):
    """Reject explicitly strided views: v1 copies dense row-major rows only."""
    shape = list(type_.shape)
    strides = list(stride) if stride else []
    if len(strides) != len(shape):
        return
    expected = 1
    for axis in range(len(shape) - 1, -1, -1):
        actual = _const_int_value(strides[axis])
        dim = _const_int_value(shape[axis])
        if actual is None:
            return
        if actual != expected:
            raise InvalidShape(
                f"{api} {name} must be dense row-major, but axis {axis} has stride {actual} "
                f"(expected {expected})",
                span=span,
            )
        if dim is None:
            return
        expected *= dim


def _check_tensor_source_layout(source, api, span):
    """v1 payload reads assume a contiguous ND source tensor."""
    tensor_view = getattr(source.type, "tensor_view", None)
    if tensor_view is None:
        return
    layout = getattr(tensor_view, "layout", None)
    if layout is not None and layout != _ir_core.Layout.ND:
        raise InvalidType(
            f"{api} v1 supports contiguous row-major (ND) sources only, got layout {layout}",
            span=span,
        )
    _check_dense_stride(source.type, getattr(tensor_view, "stride", None), api, "source", span)


def _check_index_tile_layout(tile_type, api, arg_name, span):
    """The index Tile must be a flat array so data()[j] is logical row-index j."""
    hardware_info = getattr(tile_type, "hardware_info", None)
    if hardware_info is None:
        return
    slayout = getattr(hardware_info, "slayout", None)
    if slayout is not None and slayout != _ir_core.TileLayout.none_box:
        raise InvalidType(
            f"{api} {arg_name} must use a flat (none_box) tile layout, got {slayout}", span=span
        )


def _index_table_length(tile_type, api, arg_name, span):
    """Length of the 1-D row-index table stored in a (possibly rank-2) UB Tile.

    Tile allocation only accepts rank-2 shapes, so a 1-D logical table is spelled
    ``[N, 1]`` or ``[1, N]``; the unit axis is not part of the logical length.
    """
    shape = list(tile_type.shape)
    if len(shape) == 1:
        return shape[0]
    if len(shape) == 2:
        first = _const_int_value(shape[0])
        second = _const_int_value(shape[1])
        if second == 1:
            return shape[0]
        if first == 1:
            return shape[1]
    raise InvalidShape(
        f"{api} {arg_name} must be a 1-D row-index Tile ([N] or the rank-2 [N, 1] / [1, N] "
        f"spelling), got shape rank {len(shape)}",
        span=span,
    )


def _channel_index_operand(value, api, arg_name, span):
    """v1 index table: one contiguous 1-D UB Tile with a 32/64-bit integer dtype."""
    tile_type = getattr(value, "type", None)
    if not isinstance(tile_type, _ir_core.TileType):
        raise InvalidType(
            f"{api} {arg_name} must be a 1-D int32/int64/uint32/uint64 UB Tile or None; "
            "copy a GM index table into a UB Tile with pl.load first",
            span=span,
        )
    _index_table_length(tile_type, api, arg_name, span)
    if tile_type.dtype not in _INDEX_DTYPES:
        raise InvalidType(
            f"{api} {arg_name} must be an INT32/INT64/UINT32/UINT64 Tile, got {tile_type.dtype}",
            span=span,
        )
    _check_index_tile_layout(tile_type, api, arg_name, span)
    memref = getattr(tile_type, "memref", None)
    space = getattr(memref, "memory_space", None) if memref is not None else None
    if space != _ir_core.MemorySpace.Vec:
        raise InvalidType(
            f"{api} {arg_name} must live in UB (MemorySpace.Vec), got {space}; "
            "copy a GM index table into a UB Tile with pl.load first",
            span=span,
        )
    return value


def _int_kwarg_operand(value, api, arg_name, span):
    """Keyword operand that is either a folded int constant or an integer scalar."""
    if _is_int(value):
        return _normalize_expr(value, span)
    return _int_operand(value, api, arg_name)


def _check_index_capacity(api, span, count, capacity, subject):
    if count is not None and capacity is not None and count > capacity:
        raise InvalidVal(
            f"{api} count {count} exceeds the {subject} capacity {capacity}", span=span
        )


def _build_channel_write_indexed(parser, call, channel):
    """``ch.write_indexed(dst_ptr, tensor, *, src_indices=, dst_indices=, count=, submit=, timeout=)``."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("src_indices", "dst_indices", "count", "submit", "timeout"):
            raise InvalidArgument(
                f"ch.write_indexed does not accept keyword argument '{name}'", span=span
            )
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.write_indexed does not accept star-args", span=span)
    if len(call.args) != 2:
        raise InvalidArgument(
            f"ch.write_indexed requires (dst_ptr, tensor), got {len(call.args)} arguments after the channel",
            span=span,
            hint="Resolve the destination with rptr = ch.remote_ptr(tensor) first",
        )

    dst = _channel_dst_ptr(parser, "ch.write_indexed", call.args[0], span)
    source = parser.parse_expression(call.args[1])
    if not isinstance(source, _ir_core.Var) or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            "ch.write_indexed source must be a kernel GM tensor (the payload rows are read from GM), "
            f"got {ast.unparse(call.args[1])}",
            span=span,
        )
    source_dtype = source.type.dtype
    _check_tensor_source_layout(source, "ch.write_indexed", span)

    src_shape = list(source.type.shape)
    if not src_shape:
        raise InvalidShape(
            "ch.write_indexed source must have at least one dimension (the row axis)", span=span
        )
    if len(src_shape) < 2:
        raise InvalidShape(
            "ch.write_indexed source must have a row payload axis (rank >= 2); a rank-1 source "
            "has no per-row payload",
            span=span,
        )

    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.write_indexed submit must be a compile-time bool, got {submit!r}", span=span)
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.write_indexed", minimum=1)

    if "count" not in kwargs_parsed:
        raise InvalidArgument(
            "ch.write_indexed requires the 'count' keyword",
            span=span,
            hint="count is the runtime number of valid rows; pass 0 for a complete no-op.",
        )
    count_value = _const_int_value(kwargs_parsed["count"])
    if count_value is not None and count_value < 0:
        raise InvalidVal(f"ch.write_indexed count must be >= 0, got {count_value}", span=span)
    count_arg = _int_kwarg_operand(kwargs_parsed["count"], "ch.write_indexed", "count", span)
    if count_arg.type.dtype not in _INDEX_DTYPES:
        raise InvalidType(
            f"ch.write_indexed count must be a 32/64-bit integer scalar, got {count_arg.type.dtype}",
            span=span,
        )

    src_index = _none_value(kwargs_parsed.get("src_indices"))
    dst_index = _none_value(kwargs_parsed.get("dst_indices"))
    if src_index is not None:
        src_index = _channel_index_operand(src_index, "ch.write_indexed", "src_indices", span)
    if dst_index is not None:
        dst_index = _channel_index_operand(dst_index, "ch.write_indexed", "dst_indices", span)
    if src_index is None and dst_index is None:
        raise InvalidArgument(
            "ch.write_indexed requires src_indices or dst_indices",
            span=span,
            hint="Use ch.write(dst_ptr, src) for one contiguous copy.",
        )

    if src_index is not None:
        src_capacity = _const_int_value(
            _index_table_length(src_index.type, "ch.write_indexed", "src_indices", span)
        )
    else:
        src_capacity = _const_int_value(src_shape[0])
    if dst_index is not None:
        dst_capacity = _const_int_value(
            _index_table_length(dst_index.type, "ch.write_indexed", "dst_indices", span)
        )
    else:
        dst_capacity = None
    src_subject = "src_indices" if src_index is not None else "source rows"
    dst_subject = "dst_indices"
    _check_index_capacity("ch.write_indexed", span, count_value, src_capacity, src_subject)
    _check_index_capacity("ch.write_indexed", span, count_value, dst_capacity, dst_subject)

    # Row addresses are index * row stride; v1 accepts contiguous row payloads only,
    # so the row stride is the payload byte size on both sides.
    sentinel = _normalize_expr(0, span)
    actual_row_bytes = _static_shape_byte_count(src_shape[1:], source_dtype)
    if actual_row_bytes is not None:
        if actual_row_bytes <= 0:
            raise InvalidVal(
                f"ch.write_indexed row_bytes must be positive, got {actual_row_bytes}", span=span
            )
        if actual_row_bytes % 32:
            raise InvalidVal(
                f"ch.write_indexed row_bytes must be a multiple of 32 bytes, got {actual_row_bytes}; "
                "pad the payload rows",
                span=span,
            )
        if actual_row_bytes > 256 * 1024 * 1024:
            raise InvalidVal(
                f"ch.write_indexed row_bytes must be <= 256 MiB, got {actual_row_bytes}", span=span
            )
        row_bytes = _normalize_expr(actual_row_bytes, span)
    else:
        row_bytes = _shape_byte_count(src_shape[1:], source_dtype, span)

    # write_indexed lowers to the header's SIMT device worker, which is invisible
    # to the DSL's tile tracking; mark the kernel here so its UB provision is
    # sized for the channel scratch.
    parser.requires_simt = True
    return _ir_core.create_op_call(
        "dist.channel_write_indexed",
        [
            channel.token,
            dst,
            source,
            count_arg,
            row_bytes,
            src_index if src_index is not None else sentinel,
            dst_index if dst_index is not None else sentinel,
        ],
        {"submit": submit, "timeout_cycles": timeout},
        span,
    )


def _build_channel_write_notify(parser, call, channel):
    """``ch.write_notify(dst_ptr, tensor, signal_ptr, val, timeout=)`` -> channel_write_notify."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "timeout":
            raise InvalidArgument(f"ch.write_notify does not accept keyword argument '{name}'")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.write_notify does not accept star-args")
    if len(call.args) != 4:
        raise InvalidArgument(
            f"ch.write_notify requires (dst_ptr, tensor, signal_ptr, val), got {len(call.args)} "
            "arguments after the channel",
            hint="Resolve both addresses with ch.remote_ptr(tensor) first",
        )

    dst = _channel_dst_ptr(parser, "ch.write_notify", call.args[0], span)
    source = parser.parse_expression(call.args[1])
    if not isinstance(source, _ir_core.Var) or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            "ch.write_notify tensor must be a kernel GM tensor (the fused WQE reads its source "
            f"from GM; use ch.write for the scalar-word form), got {ast.unparse(call.args[1])}",
        )
    signal = _signal_ptr_operand(
        parser.parse_expression(call.args[2]), "ch.write_notify", "signal", span,
        allow_tensor=False, allow_32=False)
    val = _parse_int_operand(parser, call.args[3], "ch.write_notify", "val")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.write_notify", minimum=1)

    num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
    return _ir_core.create_op_call(
        "dist.channel_write_notify",
        [
            channel.token,
            dst,
            num_bytes_arg,
            source,
            signal,
            val,
        ],
        {"timeout_cycles": timeout},
        span,
    )


def _build_channel_submit(parser, call, channel):
    """``ch.submit()`` -> dist.channel_submit."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.submit takes no arguments")
    return _ir_core.create_op_call("dist.channel_submit", [channel.token], {}, span)


def _build_channel_drain(parser, call, channel):
    """``ch.drain(timeout=)`` / ``ch.drain(timeout)`` -> dist.channel_drain."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "timeout":
            raise InvalidArgument(f"ch.drain does not accept keyword argument '{name}'")
    if len(call.args) > 1:
        raise InvalidArgument("ch.drain accepts at most one positional timeout")
    if call.args:
        if "timeout" in kwargs_parsed:
            raise InvalidArgument("ch.drain timeout given both positionally and by keyword")
        kwargs_parsed = {"timeout": _require_const_int(parser, call.args[0], "ch.drain", "timeout")}
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.drain", minimum=1)
    return _ir_core.create_op_call(
        "dist.channel_drain", [channel.token], {"timeout_cycles": timeout}, span
    )


def _build_channel_is_exclusive(parser, call, channel):
    """``ch.is_exclusive()`` -> dist.channel_is_exclusive (runtime bool)."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.is_exclusive takes no arguments")
    return _ir_core.create_op_call("dist.channel_is_exclusive", [channel.token], {}, span)


def parse_channel_method(parser, channel, method, call):
    """Dispatch ``ch.<method>(...)`` (receiver-routed by the parser)."""
    if method == "write":
        return _build_channel_write(parser, call, channel)
    if method == "remote_ptr":
        return _build_channel_remote_ptr(parser, call, channel)
    if method == "write_indexed":
        return _build_channel_write_indexed(parser, call, channel)
    if method == "write_notify":
        return _build_channel_write_notify(parser, call, channel)
    if method == "submit":
        return _build_channel_submit(parser, call, channel)
    if method == "drain":
        return _build_channel_drain(parser, call, channel)
    if method == "is_exclusive":
        return _build_channel_is_exclusive(parser, call, channel)
    raise InvalidOperation(
        f"pld.Channel has no method '{method}'"
    )
