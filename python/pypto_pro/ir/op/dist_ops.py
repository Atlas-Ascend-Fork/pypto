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
    require_enum,
)
from ._op_registry import op_impl, op_impl_ast

# kwargs cross the binding as C++ int (32-bit); timeouts are device cycles.
_MAX_TIMEOUT_CYCLES = (1 << 31) - 1


@op_impl("dist.Window.rank")
def _build_window_rank(self, window: Expr, /, *, span: Span) -> Call:
    field = _ir_core.ConstInt(0, _ir_core.DataType.INT64, span)
    return _ir_core.create_op_call("dist.window_get", [window, field], {}, span)


@op_impl("dist.Window.nrank")
def _build_window_nrank(self, window: Expr, /, *, span: Span) -> Call:
    field = _ir_core.ConstInt(1, _ir_core.DataType.INT64, span)
    return _ir_core.create_op_call("dist.window_get", [window, field], {}, span)


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


@op_impl("dist.remote_tensor")
def _build_remote_tensor(self, window, tensor, peer, *, span: Span | None = None):
    """``pld.remote_tensor(win, tensor, peer)`` -> tensor view over the peer's region.

    Sugar for ``ptr.make_tensor(pld.remote_ptr(win, tensor, peer), tensor.shape)``: the
    view keeps the source shape, so element offsets address the peer's element."""
    from .ptr_ops import make_tensor

    source = tensor if isinstance(tensor, _ir_core.Expr) else None
    if source is None or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            "pld.remote_tensor 'tensor' must be a kernel tensor parameter registered by "
            "Window.register_memory; use pld.remote_ptr + pl.make_tensor for a raw-pointer view",
        )
    window = _window_param(window, "pld.remote_tensor", "win")
    peer_operand = _int_operand(peer, "pld.remote_tensor", "peer")
    ptr = _ir_core.create_op_call(
        "dist.remote_ptr",
        [window, _ptr_operand(source, "pld.remote_tensor", "tensor", span), peer_operand],
        {},
        span,
    )
    return make_tensor(ptr, source.type.shape, None, None, span)


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


def _signal_ptr_operand(value, api, arg_name, span):
    """Signal-word operand: a pointer to a remote 64-bit integer slot.

    The fused write-with-notify forms take the remote pointer from ``remote_ptr`` and
    require a 64-bit word (the hardware notify value is a ``uint64_t``). The framework
    cannot verify where a pointer came from.
    """
    if isinstance(value, _ir_core.Expr) and isinstance(value.type, _ir_core.PtrType):
        ptr = value
    else:
        raise InvalidType(
            f"{api} {arg_name} must be a remote pointer resolved by ch.remote_ptr(...)")
    dtype = _ptr_dtype(ptr, api)
    supported = dtype in (_ir_core.DataType.INT64, _ir_core.DataType.UINT64)
    kinds = "64-bit integer signal words"
    if not supported:
        raise InvalidType(f"{api} {arg_name} must point at {kinds}, got {dtype}")
    return ptr


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


class _BufferChannelCreate:
    """Internal: ``pld.buffer_channel`` deferred to the assignment parser, same
    mechanism as :class:`_ChannelCreate`."""

    __slots__ = ("base", "buf_addr", "buf_bytes", "call")

    def __init__(self, base, buf_addr, buf_bytes, call):
        self.base = base
        self.buf_addr = buf_addr
        self.buf_bytes = buf_bytes
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


# WQE flag byte layout shared with hcomm's WqeEntry: odr[2:0], fence[3], cqe[5].
_URMA_OPTION_MASK = 0x2F
_URMA_OPTION_DEFAULT = 0x2D
_URMA_OPTION_CQE = 0x20
_URMA_OPTION_ODR_CONFLICT = 0x03


def _urma_option_bits(value, api):
    """Compile-time ``pld.UrmaOption`` (IntFlag or int) -> validated flag-bits int."""
    if isinstance(value, _ir_core.ConstInt):
        option = int(value.value)
    elif _is_int(value):
        option = int(value)
    else:
        raise InvalidType(
            f"{api} option must be a compile-time pld.UrmaOption, got {value!r}",
            hint="pass a pld.UrmaOption member (or an int literal), not a runtime value",
        )
    if option < 0 or option & ~_URMA_OPTION_MASK:
        raise InvalidVal(
            f"{api} option sets unsupported WQE flag bits, got {option:#x}"
        )
    if option & _URMA_OPTION_ODR_CONFLICT == _URMA_OPTION_ODR_CONFLICT:
        raise InvalidVal(
            f"{api} option cannot set ODR_RELAX and ODR_STRONG together"
        )
    return option


def _urma_option(value, api):
    """Compile-time ``pld.UrmaOption`` -> WQE flag byte with a per-WQE completion."""
    option = _urma_option_bits(value, api)
    if not option & _URMA_OPTION_CQE:
        raise InvalidVal(
            f"{api} option must include pld.UrmaOption.CQE (drain waits for one "
            f"completion per batch), got {option:#x}"
        )
    return option


def _urma_append_option(value, api):
    """Compile-time ``pld.UrmaOption`` -> BODY flag byte of a staged append.

    A caller-set cqe bit is ignored (stripped): bch.submit publishes one completion
    per batch (fence+cqe land on the last staged WQE), so staged WQEBBs never carry
    their own."""
    return _urma_option_bits(value, api) & ~_URMA_OPTION_CQE


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


@op_impl_ast("dist.buffer_channel")
def _parse_buffer_channel(parser, call):
    """``bch = pld.buffer_channel(ch, buf_addr, buf_bytes)`` -> assignment-time handle.

    The buffer is a compile-time UB address + byte size (the parser reserves the range
    like ``pld.channel``'s scratch); the handle keeps the base channel so method
    dispatch can select the staged-append lowering."""
    from ...language.distribute import Channel

    span = parser.span_tracker.get_span(call)
    if len(call.args) != 3:
        raise InvalidArgument(
            f"pld.buffer_channel requires (ch, buf_addr, buf_bytes), got {len(call.args)} arguments",
        )
    if call.keywords:
        raise InvalidArgument("pld.buffer_channel does not accept keyword arguments")

    base_node = call.args[0]
    if not isinstance(base_node, ast.Name):
        raise InvalidType(
            "pld.buffer_channel first argument must be a pld.channel handle, got "
            f"{ast.unparse(base_node)}",
        )
    binding = parser.local_binding(base_node.id)
    base = binding[1] if binding is not None and binding[0] == "parse_time" else None
    if not isinstance(base, Channel):
        raise InvalidType(
            f"pld.buffer_channel first argument must be a pld.channel handle, got '{base_node.id}'",
        )

    buf_addr = _require_const_int(parser, call.args[1], "pld.buffer_channel", "buf_addr")
    buf_bytes = _require_const_int(parser, call.args[2], "pld.buffer_channel", "buf_bytes")
    if buf_addr < 0 or buf_addr % 64:
        raise InvalidVal(
            f"pld.buffer_channel buf_addr must be a non-negative 64-byte aligned address, got {buf_addr}",
        )
    if buf_bytes < 2 * 64 or buf_bytes % 64:
        raise InvalidVal(
            f"pld.buffer_channel buf_bytes must be a 64-byte multiple of at least 128, got {buf_bytes}",
        )

    # The buffer may be used by SIMT code; update the max_vec_tile_end for the whole
    # reservation, like pld.channel does for its scratch.
    parser.max_vec_tile_end = max(parser.max_vec_tile_end, buf_addr + buf_bytes)

    create = _ir_core.create_op_call(
        "dist.buffer_channel", [base.token], {"buf_addr": buf_addr, "buf_bytes": buf_bytes}, span
    )
    return _BufferChannelCreate(base, buf_addr, buf_bytes, create)


def _channel_dst_ptr(parser, api, node, span):
    """Parse a channel transfer destination: a Ptr or a registered tensor parameter."""
    return _ptr_operand(parser.parse_expression(node), api, "destination", span)


def _ptr_dtype(ptr, api):
    ptr_type = getattr(ptr, "type", None)
    if not isinstance(ptr_type, _ir_core.PtrType):
        raise InvalidType(f"{api} pointer operand must have PtrType")
    return ptr_type.dtype


def _build_channel_remote_ptr(parser, call, channel):
    """``ch.remote_ptr(tensor_or_ptr)`` -> channel.remote_ptr."""
    span = parser.span_tracker.get_span(call)
    if len(call.args) != 1 or call.keywords:
        raise InvalidArgument("ch.remote_ptr requires exactly one tensor or pointer argument")
    if isinstance(call.args[0], ast.Starred):
        raise InvalidArgument("ch.remote_ptr does not accept star-args")
    dst = _channel_dst_ptr(parser, "ch.remote_ptr", call.args[0], span)
    return _ir_core.create_op_call("channel.remote_ptr", [channel.token, dst], {}, span)


def _build_channel_write(parser, call, channel):
    """``ch.write(dst_ptr, tensor|scalar, submit=, timeout=)`` -> channel.write[_value]."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("submit", "timeout", "option"):
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
    option = _urma_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "ch.write")

    source = parser.parse_expression(call.args[1])
    if isinstance(source, _ir_core.Var) and isinstance(source.type, _ir_core.TensorType):
        # The destination pointer is a byte address; the payload dtype and length come
        # from the source tensor.
        _check_tensor_source_layout(source, "ch.write", span)
        num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
        return _ir_core.create_op_call(
            "channel.write",
            [channel.token, dst, num_bytes_arg, source],
            {"submit": submit, "timeout_cycles": timeout, "option": option},
            span,
        )

    value = _parse_write_scalar(source, call.args[1], _ptr_dtype(dst, "ch.write"), api="ch.write")
    # The scalar rides inline in the WQE; no MTE3 flush is needed.
    return _ir_core.create_op_call(
        "channel.write_value",
        [channel.token, dst, value],
        {"submit": submit, "timeout_cycles": timeout, "option": option},
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


def _check_tensor_source_layout(source, api, span, name="source"):
    """v1 payload transfers assume a contiguous ND tensor."""
    tensor_view = getattr(source.type, "tensor_view", None)
    if tensor_view is None:
        return
    layout = getattr(tensor_view, "layout", None)
    if layout is not None and layout != _ir_core.Layout.ND:
        raise InvalidType(
            f"{api} v1 supports contiguous row-major (ND) {name}s only, got layout {layout}",
            span=span,
        )
    _check_dense_stride(source.type, getattr(tensor_view, "stride", None), api, name, span)


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


def _build_channel_read(parser, call, channel):
    """``ch.read(dst_tensor, src_ptr, submit=, timeout=)`` -> channel.read."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("submit", "timeout", "option"):
            raise InvalidArgument(f"ch.read does not accept keyword argument '{name}'")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.read does not accept star-args")
    if len(call.args) != 2:
        raise InvalidArgument(
            f"ch.read requires (dst_tensor, src_ptr), got {len(call.args)} arguments after the channel",
            hint="Resolve the remote source with rptr = ch.remote_ptr(tensor) first; the peer is bound "
                 "when the channel is opened",
        )

    source = parser.parse_expression(call.args[1])
    if not isinstance(source, _ir_core.Expr) or not isinstance(source.type, _ir_core.PtrType):
        raise InvalidType(
            "ch.read source must be a remote pointer resolved by ch.remote_ptr(...) "
            f"(the engine pulls from the peer's region), got {ast.unparse(call.args[1])}")
    dst = parser.parse_expression(call.args[0])
    if not isinstance(dst, _ir_core.Var) or not isinstance(dst.type, _ir_core.TensorType):
        raise InvalidType(
            "ch.read destination must be a kernel GM tensor (the engine writes the fetched bytes "
            f"into it; use ch.write for the scalar-word form), got {ast.unparse(call.args[0])}")
    _check_tensor_source_layout(dst, "ch.read", span, name="destination")
    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.read submit must be a compile-time bool, got {submit!r}")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.read", minimum=1)
    option = _urma_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "ch.read")

    num_bytes_arg = _shape_byte_count(dst.type.shape, dst.type.dtype, span)
    return _ir_core.create_op_call(
        "channel.read",
        [channel.token, source, num_bytes_arg, dst],
        {"submit": submit, "timeout_cycles": timeout, "option": option},
        span,
    )


_REDUCE_DTYPES = (
    _ir_core.DataType.INT8,
    _ir_core.DataType.INT16,
    _ir_core.DataType.INT32,
    _ir_core.DataType.UINT32,
    _ir_core.DataType.FP16,
    _ir_core.DataType.FP32,
    _ir_core.DataType.BF16,
)

_REDUCE_OPS = (0x8, 0x9, 0xA)  # ReduceOp.MAX / MIN / SUM wire codes


def _build_channel_write_reduce(parser, call, channel):
    """``ch.write_reduce(dst_ptr, tensor, op=, submit=, timeout=)`` -> channel.write_reduce."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("op", "submit", "timeout", "option"):
            raise InvalidArgument(f"ch.write_reduce does not accept keyword argument '{name}'")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.write_reduce does not accept star-args")
    if len(call.args) != 2:
        raise InvalidArgument(
            f"ch.write_reduce requires (dst_ptr, tensor), got {len(call.args)} arguments after the channel",
            hint="Resolve the destination with rptr = ch.remote_ptr(tensor) first",
        )
    if "op" not in kwargs_parsed:
        raise InvalidArgument(
            "ch.write_reduce requires op=pld.ReduceOp.SUM/MAX/MIN",
            hint="The reduction mode is explicit so a lossy MAX/MIN is never spelled as a silent default",
        )
    from ...language.distribute import ReduceOp

    reduce_op = int(require_enum(kwargs_parsed["op"], ReduceOp, subject="ch.write_reduce op").value)
    if reduce_op not in _REDUCE_OPS:
        raise InvalidVal(
            f"ch.write_reduce op must be a pld.ReduceOp member (SUM/MAX/MIN), got {reduce_op}")

    dst = _channel_dst_ptr(parser, "ch.write_reduce", call.args[0], span)
    source = parser.parse_expression(call.args[1])
    if not isinstance(source, _ir_core.Var) or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            "ch.write_reduce source must be a kernel GM tensor (the engine reads the reduction "
            f"operands from GM), got {ast.unparse(call.args[1])}")
    if source.type.dtype not in _REDUCE_DTYPES:
        raise InvalidType(
            "ch.write_reduce does not support dtype "
            f"{source.type.dtype}; supported: int8/int16/int32/uint32/float16/float32/bfloat16")
    _check_tensor_source_layout(source, "ch.write_reduce", span)
    if _ptr_dtype(dst, "ch.write_reduce") != source.type.dtype:
        raise InvalidType(
            "ch.write_reduce source dtype must match the destination pointer "
            f"({_ptr_dtype(dst, 'ch.write_reduce')}), got {source.type.dtype}")
    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.write_reduce submit must be a compile-time bool, got {submit!r}")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.write_reduce", minimum=1)
    option = _urma_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "ch.write_reduce")

    num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
    return _ir_core.create_op_call(
        "channel.write_reduce",
        [channel.token, dst, num_bytes_arg, source],
        {"reduce_op": reduce_op, "submit": submit, "timeout_cycles": timeout, "option": option},
        span,
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
        "channel.write_indexed",
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


_ATOMIC_DTYPES = (
    _ir_core.DataType.INT32,
    _ir_core.DataType.UINT32,
    _ir_core.DataType.INT64,
    _ir_core.DataType.UINT64,
)


def _atomic_fetch_operand(source, api):
    """Local GM tensor receiving the engine's fetch result: same dtype as the
    target pointer, exactly one element when the shape is static."""
    if not isinstance(source, _ir_core.Var) or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            f"{api} fetch must be a kernel GM tensor (the engine writes the previous value there), "
            f"got {ast.unparse(source)}")
    return source


def _build_channel_atomic_fetch_add(parser, call, channel):
    """``ch.atomic_fetch_add(dst_ptr, fetch_tensor, val, submit=, timeout=)`` -> channel_atomic_add."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("submit", "timeout", "option"):
            raise InvalidArgument(f"ch.atomic_fetch_add does not accept keyword argument '{name}'")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.atomic_fetch_add does not accept star-args")
    if len(call.args) != 3:
        raise InvalidArgument(
            f"ch.atomic_fetch_add requires (dst_ptr, fetch_tensor, val), got {len(call.args)} arguments "
            "after the channel",
            hint="Resolve the target with rptr = ch.remote_ptr(tensor) first",
        )

    dst = _channel_dst_ptr(parser, "ch.atomic_fetch_add", call.args[0], span)
    dtype = _ptr_dtype(dst, "ch.atomic_fetch_add")
    if dtype not in _ATOMIC_DTYPES:
        raise InvalidType(
            f"ch.atomic_fetch_add destination must point at a 32/64-bit integer word, got {dtype}")
    fetch = _atomic_fetch_operand(parser.parse_expression(call.args[1]), "ch.atomic_fetch_add")
    if fetch.type.dtype != dtype:
        raise InvalidType(
            f"ch.atomic_fetch_add fetch tensor dtype must match the destination pointer "
            f"({dtype}), got {fetch.type.dtype}")
    fetch_bytes = _static_shape_byte_count(fetch.type.shape, fetch.type.dtype)
    if fetch_bytes is not None and fetch_bytes != dtype.get_bit() // 8:
        raise InvalidVal(
            f"ch.atomic_fetch_add fetch tensor must hold exactly one {dtype} element, got shape "
            f"{fetch.type.shape}")
    val = _parse_int_operand(parser, call.args[2], "ch.atomic_fetch_add", "val")
    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.atomic_fetch_add submit must be a compile-time bool, got {submit!r}")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.atomic_fetch_add", minimum=1)
    option = _urma_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "ch.atomic_fetch_add")

    return _ir_core.create_op_call(
        "channel.atomic_fetch_add",
        [channel.token, dst, fetch, val],
        {"submit": submit, "timeout_cycles": timeout, "option": option},
        span,
    )


def _build_channel_atomic_cas(parser, call, channel):
    """``ch.atomic_cas(dst_ptr, fetch_tensor, expected, desired, submit=, timeout=)``."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("submit", "timeout", "option"):
            raise InvalidArgument(f"ch.atomic_cas does not accept keyword argument '{name}'")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("ch.atomic_cas does not accept star-args")
    if len(call.args) != 4:
        raise InvalidArgument(
            f"ch.atomic_cas requires (dst_ptr, fetch_tensor, expected, desired), got {len(call.args)} "
            "arguments after the channel",
            hint="Resolve the target with rptr = ch.remote_ptr(tensor) first",
        )

    dst = _channel_dst_ptr(parser, "ch.atomic_cas", call.args[0], span)
    dtype = _ptr_dtype(dst, "ch.atomic_cas")
    if dtype not in _ATOMIC_DTYPES:
        raise InvalidType(
            f"ch.atomic_cas destination must point at a 32/64-bit integer word, got {dtype}")
    fetch = _atomic_fetch_operand(parser.parse_expression(call.args[1]), "ch.atomic_cas")
    if fetch.type.dtype != dtype:
        raise InvalidType(
            f"ch.atomic_cas fetch tensor dtype must match the destination pointer "
            f"({dtype}), got {fetch.type.dtype}")
    fetch_bytes = _static_shape_byte_count(fetch.type.shape, fetch.type.dtype)
    if fetch_bytes is not None and fetch_bytes != dtype.get_bit() // 8:
        raise InvalidVal(
            f"ch.atomic_cas fetch tensor must hold exactly one {dtype} element, got shape "
            f"{fetch.type.shape}")
    expected = _parse_int_operand(parser, call.args[2], "ch.atomic_cas", "expected")
    desired = _parse_int_operand(parser, call.args[3], "ch.atomic_cas", "desired")
    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.atomic_cas submit must be a compile-time bool, got {submit!r}")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.atomic_cas", minimum=1)
    option = _urma_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "ch.atomic_cas")

    return _ir_core.create_op_call(
        "channel.atomic_cas",
        [channel.token, dst, fetch, expected, desired],
        {"submit": submit, "timeout_cycles": timeout, "option": option},
        span,
    )


def _build_channel_write_notify(parser, call, channel):
    """``ch.write_notify(dst_ptr, tensor, signal_ptr, val, submit=, timeout=)`` -> channel.write_notify."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("submit", "timeout", "option"):
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
    _check_tensor_source_layout(source, "ch.write_notify", span)
    signal = _signal_ptr_operand(
        parser.parse_expression(call.args[2]), "ch.write_notify", "signal", span)
    val = _parse_int_operand(parser, call.args[3], "ch.write_notify", "val")
    submit = kwargs_parsed.get("submit", True)
    if not isinstance(submit, bool):
        raise InvalidType(f"ch.write_notify submit must be a compile-time bool, got {submit!r}")
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "ch.write_notify", minimum=1)
    option = _urma_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "ch.write_notify")

    num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
    return _ir_core.create_op_call(
        "channel.write_notify",
        [
            channel.token,
            dst,
            num_bytes_arg,
            source,
            signal,
            val,
        ],
        {"submit": submit, "timeout_cycles": timeout, "option": option},
        span,
    )


def _build_channel_submit(parser, call, channel):
    """``ch.submit()`` -> channel.submit."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.submit takes no arguments")
    return _ir_core.create_op_call("channel.submit", [channel.token], {}, span)


def _build_channel_lock(parser, call, channel):
    """``ch.lock()`` -> channel.lock (cross-AIV exclusion over the channel's queue state)."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.lock takes no arguments")
    return _ir_core.create_op_call("channel.lock", [channel.token], {}, span)


def _build_channel_unlock(parser, call, channel):
    """``ch.unlock()`` -> channel.unlock."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.unlock takes no arguments")
    return _ir_core.create_op_call("channel.unlock", [channel.token], {}, span)


def _build_channel_drain(parser, call, channel):
    """``ch.drain(timeout=)`` / ``ch.drain(timeout)`` -> channel.drain."""
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
        "channel.drain", [channel.token], {"timeout_cycles": timeout}, span
    )


def _build_channel_is_exclusive(parser, call, channel):
    """``ch.is_exclusive()`` -> channel.is_exclusive (runtime bool)."""
    span = parser.span_tracker.get_span(call)
    if call.args or call.keywords:
        raise InvalidArgument("ch.is_exclusive takes no arguments")
    return _ir_core.create_op_call("channel.is_exclusive", [channel.token], {}, span)


def parse_channel_method(parser, channel, method, call):
    """Dispatch ``ch.<method>(...)`` (receiver-routed by the parser)."""
    if method == "write":
        return _build_channel_write(parser, call, channel)
    if method == "read":
        return _build_channel_read(parser, call, channel)
    if method == "write_reduce":
        return _build_channel_write_reduce(parser, call, channel)
    if method == "atomic_fetch_add":
        return _build_channel_atomic_fetch_add(parser, call, channel)
    if method == "atomic_cas":
        return _build_channel_atomic_cas(parser, call, channel)
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
    if method == "lock":
        return _build_channel_lock(parser, call, channel)
    if method == "unlock":
        return _build_channel_unlock(parser, call, channel)
    if method == "is_exclusive":
        return _build_channel_is_exclusive(parser, call, channel)
    raise InvalidOperation(
        f"pld.Channel has no method '{method}'"
    )


def parse_buffer_channel_method(parser, bchannel, method, call):
    """Dispatch ``bch.<method>(...)`` for a :class:`BufferChannel` receiver.

    Every engine op stages into the ring like ``write`` (one ``submit`` publishes
    appends and engine ops together); ``drain`` reuses the plain ``channel.drain`` op
    via the derived token — the device traps on staged WQEBBs, so submit-then-drain
    stays enforced."""
    if method == "write":
        return _build_buffer_channel_write(parser, call, bchannel)
    if method == "submit":
        return _build_buffer_channel_submit(parser, call, bchannel)
    if method == "drain":
        return parse_channel_method(parser, bchannel, method, call)
    if method == "read":
        return _build_buffer_channel_read(parser, call, bchannel)
    if method == "write_reduce":
        return _build_buffer_channel_write_reduce(parser, call, bchannel)
    if method == "write_notify":
        return _build_buffer_channel_write_notify(parser, call, bchannel)
    if method == "atomic_fetch_add":
        return _build_buffer_channel_atomic_fetch_add(parser, call, bchannel)
    if method == "atomic_cas":
        return _build_buffer_channel_atomic_cas(parser, call, bchannel)
    raise InvalidOperation(f"pld.BufferChannel has no method '{method}'")


def _build_buffer_channel_write(parser, call, bchannel):
    """``bch.write(dst_ptr, tensor|scalar, option=)`` -> buffer_channel.write[_value].

    Stages only: no ``submit``/``timeout`` exist here. ``option`` (default
    ``pld.UrmaOption.DEFAULT``) is the staged WQEs' BODY flag byte with any caller-set
    cqe bit ignored — ``bch.submit`` adds fence+cqe to the last WQEBB when it
    publishes, so one completion covers the batch."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "option":
            raise InvalidArgument(
                f"bch.write does not accept keyword argument '{name}' (it only stages)",
                hint="call bch.submit() to publish the staged batch",
            )
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("bch.write does not accept star-args")
    if len(call.args) != 2:
        raise InvalidArgument(
            f"bch.write requires (dst_ptr, tensor or scalar), got {len(call.args)} arguments after the buffer channel",
            hint="Resolve the destination with ch.remote_ptr(tensor) on the base channel first",
        )
    option = _urma_append_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "bch.write")

    dst = _channel_dst_ptr(parser, "bch.write", call.args[0], span)
    source = parser.parse_expression(call.args[1])
    if isinstance(source, _ir_core.Var) and isinstance(source.type, _ir_core.TensorType):
        _check_tensor_source_layout(source, "bch.write", span)
        num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
        return _ir_core.create_op_call(
            "buffer_channel.write", [bchannel.token, dst, num_bytes_arg, source], {"option": option}, span
        )

    value = _parse_write_scalar(source, call.args[1], _ptr_dtype(dst, "bch.write"), api="bch.write")
    return _ir_core.create_op_call(
        "buffer_channel.write_value", [bchannel.token, dst, value], {"option": option}, span
    )


def _build_buffer_channel_submit(parser, call, bchannel):
    """``bch.submit(timeout=)`` / ``bch.submit(timeout)`` -> buffer_channel.submit."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "timeout":
            raise InvalidArgument(f"bch.submit does not accept keyword argument '{name}'")
    if len(call.args) > 1:
        raise InvalidArgument("bch.submit accepts at most one positional timeout")
    if call.args:
        if "timeout" in kwargs_parsed:
            raise InvalidArgument("bch.submit timeout given both positionally and by keyword")
        kwargs_parsed = {"timeout": _require_const_int(parser, call.args[0], "bch.submit", "timeout")}
    timeout = _timeout_value(kwargs_parsed.get("timeout", None), "bch.submit", minimum=1)
    return _ir_core.create_op_call(
        "buffer_channel.submit", [bchannel.token], {"timeout_cycles": timeout}, span
    )


def _build_buffer_channel_read(parser, call, bchannel):
    """``bch.read(dst_tensor, src_ptr, option=)`` -> buffer_channel.read (staged)."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "option":
            raise InvalidArgument(f"bch.read does not accept keyword argument '{name}' (it only stages)")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("bch.read does not accept star-args")
    if len(call.args) != 2:
        raise InvalidArgument(
            f"bch.read requires (dst_tensor, src_ptr), got {len(call.args)} arguments after the buffer channel",
            hint="Resolve the remote source with ch.remote_ptr(tensor) on the base channel first",
        )

    source = parser.parse_expression(call.args[1])
    if not isinstance(source, _ir_core.Expr) or not isinstance(source.type, _ir_core.PtrType):
        raise InvalidType(
            "bch.read source must be a remote pointer resolved by ch.remote_ptr(...) "
            f"(the engine pulls from the peer's region), got {ast.unparse(call.args[1])}")
    dst = parser.parse_expression(call.args[0])
    if not isinstance(dst, _ir_core.Var) or not isinstance(dst.type, _ir_core.TensorType):
        raise InvalidType(
            "bch.read destination must be a kernel GM tensor (the engine writes the fetched bytes "
            f"into it), got {ast.unparse(call.args[0])}")
    _check_tensor_source_layout(dst, "bch.read", span, name="destination")
    option = _urma_append_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "bch.read")

    num_bytes_arg = _shape_byte_count(dst.type.shape, dst.type.dtype, span)
    return _ir_core.create_op_call(
        "buffer_channel.read",
        [bchannel.token, source, num_bytes_arg, dst],
        {"option": option},
        span,
    )


def _build_buffer_channel_write_reduce(parser, call, bchannel):
    """``bch.write_reduce(dst_ptr, tensor, op=, option=)`` -> buffer_channel.write_reduce (staged)."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("op", "option"):
            raise InvalidArgument(f"bch.write_reduce does not accept keyword argument '{name}' (it only stages)")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("bch.write_reduce does not accept star-args")
    if len(call.args) != 2:
        raise InvalidArgument(
            f"bch.write_reduce requires (dst_ptr, tensor), got {len(call.args)} arguments after the buffer channel",
            hint="Resolve the destination with ch.remote_ptr(tensor) on the base channel first",
        )
    if "op" not in kwargs_parsed:
        raise InvalidArgument(
            "bch.write_reduce requires op=pld.ReduceOp.SUM/MAX/MIN",
            hint="The reduction mode is explicit so a lossy MAX/MIN is never spelled as a silent default",
        )
    from ...language.distribute import ReduceOp

    reduce_op = int(require_enum(kwargs_parsed["op"], ReduceOp, subject="bch.write_reduce op").value)
    if reduce_op not in _REDUCE_OPS:
        raise InvalidVal(
            f"bch.write_reduce op must be a pld.ReduceOp member (SUM/MAX/MIN), got {reduce_op}")

    dst = _channel_dst_ptr(parser, "bch.write_reduce", call.args[0], span)
    source = parser.parse_expression(call.args[1])
    if not isinstance(source, _ir_core.Var) or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            "bch.write_reduce source must be a kernel GM tensor (the engine reads the reduction "
            f"operands from GM), got {ast.unparse(call.args[1])}")
    if source.type.dtype not in _REDUCE_DTYPES:
        raise InvalidType(
            "bch.write_reduce does not support dtype "
            f"{source.type.dtype}; supported: int8/int16/int32/uint32/float16/float32/bfloat16")
    _check_tensor_source_layout(source, "bch.write_reduce", span)
    if _ptr_dtype(dst, "bch.write_reduce") != source.type.dtype:
        raise InvalidType(
            "bch.write_reduce source dtype must match the destination pointer "
            f"({_ptr_dtype(dst, 'bch.write_reduce')}), got {source.type.dtype}")
    option = _urma_append_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "bch.write_reduce")

    num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
    return _ir_core.create_op_call(
        "buffer_channel.write_reduce",
        [bchannel.token, dst, num_bytes_arg, source],
        {"reduce_op": reduce_op, "option": option},
        span,
    )


def _build_buffer_channel_write_notify(parser, call, bchannel):
    """``bch.write_notify(dst_ptr, tensor, signal_ptr, val, option=)`` -> buffer_channel.write_notify."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name not in ("option",):
            raise InvalidArgument(f"bch.write_notify does not accept keyword argument '{name}'")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("bch.write_notify does not accept star-args")
    if len(call.args) != 4:
        raise InvalidArgument(
            f"bch.write_notify requires (dst_ptr, tensor, signal_ptr, val), got {len(call.args)} "
            "arguments after the buffer channel",
            hint="Resolve both addresses with ch.remote_ptr(tensor) first",
        )

    dst = _channel_dst_ptr(parser, "bch.write_notify", call.args[0], span)
    source = parser.parse_expression(call.args[1])
    if not isinstance(source, _ir_core.Var) or not isinstance(source.type, _ir_core.TensorType):
        raise InvalidType(
            "bch.write_notify tensor must be a kernel GM tensor (the fused WQE reads its source "
            f"from GM; use bch.write for the scalar-word form), got {ast.unparse(call.args[1])}",
        )
    signal = _signal_ptr_operand(
        parser.parse_expression(call.args[2]), "bch.write_notify", "signal", span)
    val = _parse_int_operand(parser, call.args[3], "bch.write_notify", "val")
    option = _urma_append_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "bch.write_notify")

    num_bytes_arg = _shape_byte_count(source.type.shape, source.type.dtype, span)
    return _ir_core.create_op_call(
        "buffer_channel.write_notify",
        [bchannel.token, dst, num_bytes_arg, source, signal, val],
        {"option": option},
        span,
    )


def _build_buffer_channel_atomic_fetch_add(parser, call, bchannel):
    """``bch.atomic_fetch_add(dst_ptr, fetch_tensor, val, option=)`` -> buffer_channel.atomic_fetch_add (staged)."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "option":
            raise InvalidArgument(
                f"bch.atomic_fetch_add does not accept keyword argument '{name}' (it only stages)")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("bch.atomic_fetch_add does not accept star-args")
    if len(call.args) != 3:
        raise InvalidArgument(
            f"bch.atomic_fetch_add requires (dst_ptr, fetch_tensor, val), got {len(call.args)} arguments "
            "after the buffer channel",
            hint="Resolve the target with ch.remote_ptr(tensor) on the base channel first",
        )

    dst = _channel_dst_ptr(parser, "bch.atomic_fetch_add", call.args[0], span)
    dtype = _ptr_dtype(dst, "bch.atomic_fetch_add")
    if dtype not in _ATOMIC_DTYPES:
        raise InvalidType(
            f"bch.atomic_fetch_add destination must point at a 32/64-bit integer word, got {dtype}")
    fetch = _atomic_fetch_operand(parser.parse_expression(call.args[1]), "bch.atomic_fetch_add")
    if fetch.type.dtype != dtype:
        raise InvalidType(
            f"bch.atomic_fetch_add fetch tensor dtype must match the destination pointer "
            f"({dtype}), got {fetch.type.dtype}")
    fetch_bytes = _static_shape_byte_count(fetch.type.shape, fetch.type.dtype)
    if fetch_bytes is not None and fetch_bytes != dtype.get_bit() // 8:
        raise InvalidVal(
            f"bch.atomic_fetch_add fetch tensor must hold exactly one {dtype} element, got shape "
            f"{fetch.type.shape}")
    val = _parse_int_operand(parser, call.args[2], "bch.atomic_fetch_add", "val")
    option = _urma_append_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "bch.atomic_fetch_add")

    return _ir_core.create_op_call(
        "buffer_channel.atomic_fetch_add",
        [bchannel.token, dst, fetch, val],
        {"option": option},
        span,
    )


def _build_buffer_channel_atomic_cas(parser, call, bchannel):
    """``bch.atomic_cas(dst_ptr, fetch_tensor, expected, desired, option=)`` -> buffer_channel.atomic_cas (staged)."""
    span = parser.span_tracker.get_span(call)
    kwargs_parsed = _parse_op_kwargs(parser, call)
    for name in kwargs_parsed:
        if name != "option":
            raise InvalidArgument(f"bch.atomic_cas does not accept keyword argument '{name}' (it only stages)")
    if any(isinstance(a, ast.Starred) for a in call.args):
        raise InvalidArgument("bch.atomic_cas does not accept star-args")
    if len(call.args) != 4:
        raise InvalidArgument(
            f"bch.atomic_cas requires (dst_ptr, fetch_tensor, expected, desired), got {len(call.args)} "
            "arguments after the buffer channel",
            hint="Resolve the target with ch.remote_ptr(tensor) on the base channel first",
        )

    dst = _channel_dst_ptr(parser, "bch.atomic_cas", call.args[0], span)
    dtype = _ptr_dtype(dst, "bch.atomic_cas")
    if dtype not in _ATOMIC_DTYPES:
        raise InvalidType(
            f"bch.atomic_cas destination must point at a 32/64-bit integer word, got {dtype}")
    fetch = _atomic_fetch_operand(parser.parse_expression(call.args[1]), "bch.atomic_cas")
    if fetch.type.dtype != dtype:
        raise InvalidType(
            f"bch.atomic_cas fetch tensor dtype must match the destination pointer "
            f"({dtype}), got {fetch.type.dtype}")
    fetch_bytes = _static_shape_byte_count(fetch.type.shape, fetch.type.dtype)
    if fetch_bytes is not None and fetch_bytes != dtype.get_bit() // 8:
        raise InvalidVal(
            f"bch.atomic_cas fetch tensor must hold exactly one {dtype} element, got shape "
            f"{fetch.type.shape}")
    expected = _parse_int_operand(parser, call.args[2], "bch.atomic_cas", "expected")
    desired = _parse_int_operand(parser, call.args[3], "bch.atomic_cas", "desired")
    option = _urma_append_option(kwargs_parsed.get("option", _URMA_OPTION_DEFAULT), "bch.atomic_cas")

    return _ir_core.create_op_call(
        "buffer_channel.atomic_cas",
        [bchannel.token, dst, fetch, expected, desired],
        {"option": option},
        span,
    )
