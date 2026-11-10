# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR construction helpers for block ops.

These functions accept IR ``Expr`` objects and return ``Call`` expressions.
Implementations registered via ``@op_impl`` receive parsed values from the
common frontend dispatcher.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
import enum
import struct
from typing import Any, Optional

from pypto.pypto_impl import NpuArch
from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import (
    AccPhase,
    AccToVecMode,
    AtomicType,
    CmpMode,
    ConstInt,
    DataType,
    Expr,
    Layout,
    MemorySpace,
    QuantMode,
    ReluPreMode,
    RoundMode,
    Span,
    STPhase,
    TilePad,
    TLoadL2Hint,
    TStoreL2Hint,
)
from pypto.pypto_impl.ir import TileType as _IRTileType  # IR-level TileType (C++ binding);
from pypto_pro.ir._utils import (
    _const_int_value,
    _const_int_values,
    _const_scalar_value,
    _is_int,
    _is_int_type,
    _is_none,
    _normalize_expr,
    _to_make_tuple,
    require_const_bool,
    require_const_int,
    require_const_int_list,
    require_const_number,
    require_dtype_in,
    require_dtypes_match,
    require_enum,
    require_int,
    require_scalar_dtype,
    unwrap_type,
)

from ..._errors import (
    CapacityUnknown,
    DynamicShapeUnsupported,
    InvalidArgument,
    InvalidFormat,
    InvalidOperation,
    InvalidShape,
    InvalidTile,
    InvalidType,
    InvalidVal,
    NotSupported,
    OutOfRange,
    PyptoProError,
    _warn_located,
    message_of,
)

# NOTE: a DSL-descriptor dataclass named ``TileType`` is defined later in this
# module and shadows this import, so use ``_IRTileType`` for isinstance checks.
from ._op_registry import op_impl, op_impl_ast


def _prepare_scale(scale: Expr | None) -> Expr | None:
    if _is_none(scale):
        return None
    if isinstance(scale, (_ir_core.ConstInt, _ir_core.ConstFloat, _ir_core.ConstBool)):
        value = require_const_number(scale, subject="'scale'")
        scale = ConstInt(
            _encode_deq_scalar(float(value)), DataType.UINT64, scale.span,
        )
    _validate_scaling_tile(scale)
    return scale


def _prepare_order(order: Expr | None) -> list[int] | None:
    return None if order is None else require_const_int_list(order, subject="order")


def _span() -> Span:
    return Span.unknown()


_BLOCK_OP_NAMESPACE = "block"


def block_ir_op(op_name: str) -> str:
    """Return the IR name for the explicit-output block DSL ops."""
    return f"{_BLOCK_OP_NAMESPACE}.{op_name}"


class FillPadMode(enum.Enum):
    """Fill-pad operating mode (lowered to different IR op names)."""

    NORMAL = 0
    EXPAND = 1
    INPLACE = 2


def _l2_cache_mode_attr(l2_cache_mode, expected_type: type, op_name: str) -> dict[str, int]:
    if l2_cache_mode is None or isinstance(getattr(l2_cache_mode, "type", None), _ir_core.NoneType):
        return {}
    if not isinstance(l2_cache_mode, expected_type):
        raise InvalidType(
            f"{op_name}: l2_cache_mode expects {expected_type.__name__}, got {type(l2_cache_mode).__name__}"
        )
    return {"l2_cache_mode": l2_cache_mode.value}


def _compute_absolute_offsets(
    tile_offsets: _ir_core.MakeTuple,
    tile_shape: list,
    tile_dims: list[int] | None,
    span: Span,
    *,
    access_size: Any | None = None,
) -> _ir_core.MakeTuple:
    """Convert tile-relative offsets to absolute tensor offsets."""
    offsets = []
    for i, tile_offset in enumerate(tile_offsets.elements):
        if access_size is not None:
            shape = access_size
        elif tile_dims is not None and i in tile_dims:
            shape = tile_shape[tile_dims.index(i)]
        else:
            offsets.append(tile_offset)
            continue

        if isinstance(tile_offset, ConstInt) and isinstance(shape, ConstInt):
            offset = ConstInt(tile_offset.value * shape.value, DataType.INT64, span)
        else:
            offset = _ir_core.Mul(tile_offset, shape, DataType.INT64, span)
        offsets.append(offset)
    return _ir_core.MakeTuple(offsets, span)


def _validate_offsets(
    op_name: str,
    offsets: Sequence[int | Expr],
    shape: Sequence[int | Expr],
    *,
    check_bounds: bool = True,
) -> None:
    """Validate offset rank, values, and optionally static starting-coordinate bounds."""
    if len(offsets) != len(shape):
        raise InvalidShape(
            f"{op_name}: Tensor rank {len(shape)} requires exactly {len(shape)} offsets, got {len(offsets)}"
        )

    for i, (off, limit) in enumerate(zip(offsets, shape, strict=True)):
        limit_val = _const_int_value(limit) if check_bounds else None
        upper_bound = limit_val - 1 if limit_val is not None else None
        require_int(
            off,
            subject=f"{op_name}: offset[{i}]",
            lower_bound=0,
            upper_bound=upper_bound,
        )


def _validate_offset_alignment(
    op_name: str,
    offsets: Sequence[int | Expr],
    alignments: Sequence[int],
) -> None:
    """Validate statically known offsets against per-axis element alignments."""
    for i, (off, alignment) in enumerate(zip(offsets, alignments, strict=True)):
        if alignment <= 1:
            continue
        off_val = _const_int_value(off)
        if off_val is not None and off_val % alignment != 0:
            raise InvalidVal(f"{op_name}: offset[{i}] must be a multiple of {alignment}, got {off_val}")


def _validate_transfer_offset_alignment(
    op_name: str,
    transfer_kind: str,
    src_tile: Expr,
    dst_tile: Expr,
    offsets: Sequence[int | Expr],
) -> None:
    """Validate TINSERT/TEXTRACT offsets using the transfer path's element alignments."""
    src_mem = src_tile.type.memref.memory_space_
    dst_mem = dst_tile.type.memref.memory_space_
    src_layout = _tile_layout(src_tile.type)
    dst_layout = _tile_layout(dst_tile.type)
    layouts = (src_layout, dst_layout)
    c0 = 256 // int(src_tile.type.dtype.get_bit())
    alignments = (1, 1)

    if (src_mem, dst_mem) == (MemorySpace.Vec, MemorySpace.Mat):
        alignments = (1, c0)
    elif (src_mem, dst_mem) == (MemorySpace.Vec, MemorySpace.Vec):
        is_fp4_extract = (
            transfer_kind == "extract"
            and layouts == (Layout.ND, Layout.ND)
            and src_tile.type.dtype in (
                DataType.FP4E2M1,
                DataType.FP4E1M2,
            )
        )
        if layouts == (Layout.NZ, Layout.NZ) or is_fp4_extract:
            alignments = (1, c0)

    _validate_offset_alignment(op_name, offsets, alignments)


def _validate_nz_transfer_axes(
    tensor_type: _ir_core.TensorType,
    tile_dims: list[int] | None,
    op_name: str,
) -> None:
    """Require a GM NZ transfer's 2D Tile to map to the Tensor's final M/N axes."""
    tensor_view = getattr(tensor_type, "tensor_view", None)
    if tensor_view is None or tensor_view.layout != Layout.NZ:
        return

    tensor_ndim = len(tensor_type.shape)
    expected_dims = list(range(tensor_ndim - 2, tensor_ndim))
    if tile_dims != expected_dims:
        raise InvalidFormat(f"{op_name}: NZ transfer only supports the last two tensor axes")


def _resolve_mx_scale_load_dims(
    dst_tile: Expr,
    src_tensor: Expr,
    offsets: Sequence[Expr],
    tile_dims: list[int] | None,
    order: list[int] | None,
    op_name: str,
) -> list[int] | None:
    """Validate and resolve the matrix axes selected by an MX scale load.

    An MX scale Tensor carries a trailing physical phase axis of static extent 2.
    That axis is storage metadata: its offset must be a compile-time zero and it
    cannot be selected as either Tile matrix axis.
    """
    dst_type = dst_tile.type
    src_type = src_tensor.type
    dst_mem = getattr(getattr(dst_type, "memref", None), "memory_space", None)
    is_mx_load = (
        src_type.dtype == DataType.FP8E8M0
        and dst_type.dtype == DataType.FP8E8M0
        and dst_mem == MemorySpace.Mat
        and _tile_layout(dst_type) in (Layout.ZZ, Layout.NN)
        and _hw_attr(getattr(dst_type, "hardware_info", None), "fractal") == _MX_SCALE_FRACTAL
    )
    if not is_mx_load:
        return tile_dims

    tensor_rank = len(src_type.shape)
    if tensor_rank < 3:
        raise InvalidShape(f"{op_name}: MX scale Tensor must contain two matrix axes and one physical phase axis")

    phase_extent = _const_int_value(src_type.shape[-1])
    if phase_extent != 2:
        raise InvalidShape(
            f"{op_name}: MX scale Tensor trailing physical phase axis must be statically equal to 2, "
            f"got {phase_extent}"
        )
    if _const_int_value(offsets[-1]) != 0:
        raise InvalidArgument(
            f"{op_name}: MX scale Tensor physical phase axis offset must be a compile-time zero"
        )

    # With no explicit order, MX lowering selects the two axes immediately before
    # the phase axis rather than the ordinary load default (the final two axes).
    selected_dims = [tensor_rank - 3, tensor_rank - 2] if order is None else tile_dims
    if tensor_rank - 1 in selected_dims:
        raise InvalidShape(
            f"{op_name}: MX scale order cannot select the trailing physical phase axis, "
            f"got {selected_dims}"
        )
    return selected_dims


def _ir_binary_cast(
    op_name: str,
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    # The fused multiply is computed in the source domain: both sources must
    # share one dtype (the out dtype differs by design -- it is target_type).
    lhs_dt = getattr(lhs.type, "dtype", None)
    require_dtype_in(lhs_dt, _CAST_BINARY_SRC_DTYPES[op_name], subject=op_name)
    require_dtypes_match([getattr(rhs.type, "dtype", None)], lhs_dt, subject=op_name)
    out_dt = getattr(out.type, "dtype", None)
    if isinstance(target_type, DataType) and out_dt is not None and out_dt != target_type:
        raise InvalidType(
            f"{op_name}: out dtype {out_dt} does not match target_type {target_type}; "
            f"declare the out tile with the target dtype"
        )
    return _ir_core.create_op_call(
        block_ir_op(op_name),
        [out, lhs, rhs],
        {"target_type": target_type, "mode": mode},
        span or _span(),
    )


def _ir_load(
    dst_tile: Expr,
    src_tensor: Expr,
    offsets: Expr,
    *,
    span: Span | None = None,
    order: list[int] | None = None,
    l2_cache_mode: TLoadL2Hint | None = None,
) -> Expr:
    op_name = "load"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(offsets, actual_span)

    tensor_ndim = len(src_tensor.type.shape)
    tile_ndim = len(dst_tile.type.shape)
    tile_dims, is_transpose = _resolve_order(order, tensor_ndim, tile_ndim, op_name)

    _, tile_dims, _ = _validate_load_operands(dst_tile, src_tensor, tile_dims, op_name)
    _validate_offsets(op_name, offsets_tuple.elements, src_tensor.type.shape)
    resolved_tile_dims = _resolve_mx_scale_load_dims(
        dst_tile, src_tensor, offsets_tuple.elements, tile_dims, order, op_name
    )
    _validate_nz_transfer_axes(src_tensor.type, resolved_tile_dims, op_name)
    nd_inner = _validate_merged_row_load(dst_tile, src_tensor, resolved_tile_dims, op_name, order=order)
    if nd_inner is not None and _const_int_value(offsets_tuple.elements[resolved_tile_dims[1]]) != 0:
        raise InvalidArgument(f"{op_name}: a merged-row load's middle axis offset must be a compile-time zero")

    kwargs: dict[str, Any] = _l2_cache_mode_attr(l2_cache_mode, TLoadL2Hint, op_name)
    if nd_inner is not None:
        kwargs["nd_inner"] = nd_inner
    if is_transpose:
        kwargs["is_transpose"] = is_transpose
    # Preserve explicit orders and MX-specific axis mappings in the IR.
    if resolved_tile_dims is not None and (order is not None or resolved_tile_dims != tile_dims):
        kwargs["tile_dims"] = resolved_tile_dims
    _check_layout_dtype(op_name, src_tensor, dst_tile, is_transpose=is_transpose)
    return _ir_core.create_op_call(block_ir_op(op_name), [dst_tile, src_tensor, offsets_tuple], kwargs, actual_span)


def _ir_load_tile(
    dst_tile: Expr,
    src_tensor: Expr,
    tile_offsets: Expr,
    *,
    span: Span | None = None,
    order: list[int] | None = None,
    l2_cache_mode: TLoadL2Hint | None = None,
) -> Expr:
    op_name = "load_tile"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(tile_offsets, actual_span)

    tensor_ndim = len(src_tensor.type.shape)
    tile_ndim = len(dst_tile.type.shape)
    tile_dims, is_transpose = _resolve_order(order, tensor_ndim, tile_ndim, op_name)
    if tile_dims is not None and len(tile_dims) > tile_ndim:
        raise InvalidArgument(
            f"{op_name}: a merged-row order (more axes than the Tile has dims) is only supported "
            f"by pl.load; tile-block offsets have no single stride to multiply once the Tile's "
            f"rows span two tensor axes. Use pl.load with element offsets."
        )

    tile_shape, tile_dims, access_size = _validate_load_operands(
        dst_tile,
        src_tensor,
        tile_dims,
        op_name,
    )
    _validate_offsets(op_name, offsets_tuple.elements, src_tensor.type.shape, check_bounds=False)
    resolved_tile_dims = _resolve_mx_scale_load_dims(
        dst_tile, src_tensor, offsets_tuple.elements, tile_dims, order, op_name
    )
    _validate_nz_transfer_axes(src_tensor.type, resolved_tile_dims, op_name)

    abs_offsets = _compute_absolute_offsets(
        offsets_tuple,
        tile_shape,
        resolved_tile_dims,
        actual_span,
        access_size=access_size,
    )
    _validate_offsets(op_name, abs_offsets.elements, src_tensor.type.shape)
    _check_layout_dtype("load_tile", src_tensor, dst_tile, is_transpose=is_transpose)
    kwargs: dict[str, Any] = _l2_cache_mode_attr(l2_cache_mode, TLoadL2Hint, op_name)
    if is_transpose:
        kwargs["is_transpose"] = is_transpose
    if resolved_tile_dims is not None and (order is not None or resolved_tile_dims != tile_dims):
        kwargs["tile_dims"] = resolved_tile_dims
    return _ir_core.create_op_call(block_ir_op("load"), [dst_tile, src_tensor, abs_offsets], kwargs, actual_span)


def _encode_deq_scalar(scale: float) -> int:
    scale_bits = struct.unpack("!I", struct.pack("!f", scale))[0]
    return scale_bits


def _resolve_scale_param(scale: Expr | None) -> tuple[Expr | None, Expr | None]:
    if scale is None:
        return None, None
    scale_type = getattr(scale, "type", None)
    if isinstance(scale_type, _ir_core.TileType):
        # User-prepared per-channel Scaling tile, validated before this builder.
        return None, scale
    if not isinstance(scale_type, _ir_core.ScalarType):
        raise InvalidType(
            f"scale must be a scalar expression or a Scaling Tile, got {type(scale_type).__name__}"
        )
    # Literal scales have already been encoded by the constant consumer.
    if isinstance(scale, ConstInt) and scale_type.dtype == DataType.UINT64:
        return scale, None
    # Runtime FP32 is bitcast by codegen; integer scalars carry pre-encoded bits.
    if scale_type.dtype not in (DataType.FP32, DataType.INT32, DataType.INT64):
        raise InvalidType(
            f"scale runtime scalar dtype {scale_type.dtype} is not supported — pass "
            f"an FP32 scalar (auto-reinterpreted as its IEEE-754 bit pattern) or an "
            f"INT32/INT64 scalar carrying the pre-encoded float32 bit pattern "
            f"(struct.pack(\"!f\", scale))"
        )
    return scale, None


def _ir_store(
    dst_tensor: Expr,
    src_tile: Expr,
    offsets: Expr,
    *,
    span: Span | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    scale: Any = None,
    order: list[int] | None = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: STPhase | None = None,
    l2_cache_mode: TStoreL2Hint | None = None,
) -> Expr:
    op_name = "store"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(offsets, actual_span)

    if not isinstance(dst_tensor.type, _ir_core.TensorType):
        raise InvalidType(f"{op_name}: dst must be a Tensor, got {type(dst_tensor.type).__name__}")

    if not isinstance(src_tile.type, _ir_core.TileType):
        raise InvalidType(f"{op_name}: src must be a Tile, got {type(src_tile.type).__name__}")
    _src_mem = getattr(getattr(src_tile.type, "memref", None), "memory_space", None)
    if _src_mem is not None and _src_mem not in (_ir_core.MemorySpace.Vec, _ir_core.MemorySpace.Acc):
        raise InvalidOperation(f"{op_name}: src tile must be in Vec (UB) or Acc (L0C) memory, got {_src_mem.name}")

    if phase is not None:
        require_enum(phase, STPhase, subject=f"{op_name}: phase")
    require_enum(atomic, AtomicType, subject=f"{op_name}: atomic")

    tensor_ndim = len(dst_tensor.type.shape)
    tile_shape = list(src_tile.type.shape)
    tile_dims, access_size = _validate_tile_dims(order, tensor_ndim, tile_shape, op_name)
    tile_ndim = len(tile_shape)
    if order is not None and order != sorted(order):
        raise InvalidVal(f"{op_name}: order must be ascending, got {order}")
    _validate_nz_transfer_axes(dst_tensor.type, tile_dims, op_name)
    _validate_offsets(op_name, offsets_tuple.elements, dst_tensor.type.shape)

    pre_quant_scalar, fp_tile = _resolve_scale_param(scale)
    is_quant = pre_quant_scalar is not None or fp_tile is not None
    _check_layout_dtype(op_name, src_tile, dst_tensor, quant=is_quant)
    if relu_pre_mode is not None and _src_mem is not None and _src_mem != _ir_core.MemorySpace.Acc:
        raise NotSupported(f"{op_name}: relu_pre_mode is only supported for Acc (L0C) source tiles")
    if fp_tile is not None and atomic != AtomicType.AtomicNone:
        raise InvalidOperation(
            "scale (per-channel) cannot be combined with atomic — the fixpipe quantization "
            "path does not emit atomic instructions; per-tensor scalar scale + atomic is supported"
        )
    kwargs = _build_store_kwargs(
        relu_pre_mode=relu_pre_mode,
        tile_dims=tile_dims,
        tensor_ndim=tensor_ndim,
        tile_ndim=tile_ndim,
        atomic=atomic,
        phase=phase,
    )
    kwargs.update(_l2_cache_mode_attr(l2_cache_mode, TStoreL2Hint, op_name))
    operands: list[Expr] = [dst_tensor, src_tile, offsets_tuple]
    if fp_tile is not None:
        operands.append(fp_tile)
    elif pre_quant_scalar is not None:
        operands.append(pre_quant_scalar)
    return _ir_core.create_op_call(block_ir_op(op_name), operands, kwargs, actual_span)


def _ir_store_tile(
    dst_tensor: Expr,
    src_tile: Expr,
    tile_offsets: Expr,
    *,
    span: Span | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    scale: Any = None,
    order: list[int] | None = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: STPhase | None = None,
    l2_cache_mode: TStoreL2Hint | None = None,
) -> Expr:
    op_name = "store_tile"
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(tile_offsets, actual_span)

    if not isinstance(dst_tensor.type, _ir_core.TensorType):
        raise InvalidType(f"{op_name}: dst must be a Tensor, got {type(dst_tensor.type).__name__}")

    if not isinstance(src_tile.type, _ir_core.TileType):
        raise InvalidType(f"{op_name}: src must be a Tile, got {type(src_tile.type).__name__}")
    _src_mem = getattr(getattr(src_tile.type, "memref", None), "memory_space", None)
    if _src_mem is not None and _src_mem not in (_ir_core.MemorySpace.Vec, _ir_core.MemorySpace.Acc):
        raise InvalidOperation(f"{op_name}: src tile must be in Vec (UB) or Acc (L0C) memory, got {_src_mem.name}")

    if phase is not None:
        require_enum(phase, STPhase, subject=f"{op_name}: phase")
    require_enum(atomic, AtomicType, subject=f"{op_name}: atomic")

    tensor_ndim = len(dst_tensor.type.shape)
    tile_shape = list(src_tile.type.shape)
    tile_dims, access_size = _validate_tile_dims(order, tensor_ndim, tile_shape, op_name)
    tile_ndim = len(tile_shape)
    if order is not None and order != sorted(order):
        raise InvalidVal(f"{op_name}: order must be ascending, got {order}")
    _validate_nz_transfer_axes(dst_tensor.type, tile_dims, op_name)
    _validate_offsets(op_name, offsets_tuple.elements, dst_tensor.type.shape, check_bounds=False)
    abs_offsets = _compute_absolute_offsets(
        offsets_tuple,
        tile_shape,
        tile_dims,
        actual_span,
        access_size=access_size,
    )
    _validate_offsets(op_name, abs_offsets.elements, dst_tensor.type.shape)

    pre_quant_scalar, fp_tile = _resolve_scale_param(scale)
    is_quant = pre_quant_scalar is not None or fp_tile is not None
    _check_layout_dtype(op_name, src_tile, dst_tensor, quant=is_quant)
    if relu_pre_mode is not None and _src_mem is not None and _src_mem != _ir_core.MemorySpace.Acc:
        raise NotSupported(f"{op_name}: relu_pre_mode is only supported for Acc (L0C) source tiles")
    if fp_tile is not None and atomic != AtomicType.AtomicNone:
        raise InvalidOperation(
            "scale (per-channel) cannot be combined with atomic — the fixpipe quantization "
            "path does not emit atomic instructions; per-tensor scalar scale + atomic is supported"
        )
    kwargs = _build_store_kwargs(
        relu_pre_mode=relu_pre_mode,
        tile_dims=tile_dims,
        tensor_ndim=tensor_ndim,
        tile_ndim=tile_ndim,
        atomic=atomic,
        phase=phase,
    )
    kwargs.update(_l2_cache_mode_attr(l2_cache_mode, TStoreL2Hint, op_name))
    operands: list[Expr] = [dst_tensor, src_tile, abs_offsets]
    if fp_tile is not None:
        operands.append(fp_tile)
    elif pre_quant_scalar is not None:
        operands.append(pre_quant_scalar)
    return _ir_core.create_op_call(block_ir_op("store"), operands, kwargs, actual_span)


def _init_output_const_int(expr: Expr) -> Optional[int]:
    """Compile-time value of a ConstInt, unfolded for unsigned dtypes; None otherwise."""
    if not isinstance(expr, ConstInt):
        return None
    from pypto_pro.ir._limits import from_storage_int

    return from_storage_int(expr.value, expr.type.dtype)


def _check_init_output_count(name: str, expr: Expr) -> None:
    """init_output offset/size are element counts: integer scalars only.

    A float offset/size (Python literal or float ScalarType) used to surface only
    at CCE compile time as ``ptr + 1.5f``; reject it here with a span instead.
    """
    expr_type = getattr(expr, "type", None)
    if _is_int_type(expr_type):
        return
    if isinstance(expr_type, _ir_core.ScalarType):
        got = f"a {expr_type.dtype} scalar"
    else:
        got = type(expr_type).__name__
    raise InvalidType(f"init_output: {name} must be an integer scalar (element count), got {got}")


def _check_init_output_bounds(
    tensor_type: _ir_core.TensorType,
    offset_expr: Expr,
    size_expr: Expr,
) -> None:
    """Statically check the flat element window, only where every value is a constant.

    Dynamic offset/size (per-core multicore pattern) or a dynamic tensor dim skips
    the check — never over-rejects. numel uses the flat [1, numel] view the op
    addresses GM through, so the layout (ND/NZ) is irrelevant.
    """
    offset_val = _init_output_const_int(offset_expr)
    size_val = _init_output_const_int(size_expr)
    if offset_val is not None and offset_val < 0:
        raise InvalidShape(f"init_output: offset must be non-negative, got {offset_val}")
    if size_val is not None and size_val <= 0:
        raise InvalidShape(f"init_output: size must be a positive element count, got {size_val}")
    if offset_val is None or size_val is None:
        return
    # TensorType.shape has the same dim structure as TileType.shape (ConstInt/int
    # for static dims, other Exprs for dynamic ones).
    static_shape = _tile_shape_ints(tensor_type)
    if static_shape is None:
        return
    numel = 1
    for dim in static_shape:
        numel *= dim
    if offset_val + size_val > numel:
        raise InvalidShape(
            f"init_output: offset ({offset_val}) + size ({size_val}) exceeds tensor element count ({numel})"
        )


@op_impl("init_output")
def _ir_init_output(self,
    tensor: Expr,
    *,
    offset: Expr | None = None,
    size: Expr,
    value: Expr | None = None,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    # Sectionless code is also parsed for Cube, where this Vector-pipe op is invalid.
    if self.target != _ir_core.SectionKind.Vector:
        raise InvalidOperation(
            "init_output must be called inside 'with pl.section_vector():'",
            hint="init_output is a Vector-pipe op; a sectionless kernel is shared by Cube and "
                 "Vector, which emits it into the Cube (AIC) target and fails to compile.",
        )

    if not isinstance(tensor.type, _ir_core.TensorType):
        raise InvalidVal(f"init_output: tensor must be a Tensor, got {type(tensor.type).__name__}")
    tensor_dtype = getattr(tensor.type, "dtype", None)
    require_dtype_in(tensor_dtype, _INIT_OUTPUT_DTYPES, subject="init_output")

    offset_expr = ConstInt(0, DataType.INT64, actual_span) if offset is None else offset
    size_expr = size
    value_expr = ConstInt(0, DataType.INT64, actual_span) if value is None else value

    _check_init_output_count("offset", offset_expr)
    _check_init_output_count("size", size_expr)
    _check_init_output_bounds(tensor.type, offset_expr, size_expr)
    # The fill value lands in the tensor's elements (TEXPANDS on a temp tile), so a
    # literal must be representable in the tensor dtype; runtime scalars carry no
    # value to check. Same policy as pl.expands, which lowers through TEXPANDS too.
    from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

    check_const_expr_fits_dtype(value_expr, tensor_dtype, api="pl.init_output")

    return _ir_core.create_op_call(
        block_ir_op("init_output"),
        [tensor, offset_expr, size_expr, value_expr],
        {},
        actual_span,
    )


def _tile_shape_ints(tile_type: "_IRTileType") -> list[int] | None:
    """Return the compile-time integer shape of a TileType, or None if any
    dimension is not a static constant (skip check — never over-rejects).

    bool is excluded (same convention as _const_int_value): without this, a
    True/False dim would slip through ``isinstance(dim, int)`` as 1/0 and be
    compared as a real static dim.
    """
    return _const_int_values(tile_type.shape)


def _check_move_shape_compat(
    out: Expr,
    src: Expr,
    offset: Expr | None,
    acc_to_vec_mode: AccToVecMode | None,
    actual_span: Span,
) -> None:
    """Validate src/dst tile shape compatibility for ``block.move``.

    Rules follow the PTO ISA dispatch:

    - ``acc_to_vec_mode`` set (Acc→Vec split move): shape is governed by the
      split mode and runtime valid shape.
    - ``offset`` given (TEXTRACT sub-block extraction): ``dst <= src`` per-dim.
    - No ``offset`` with a Mat source: ``dst == src``. PTO-ISA enforces equal
      physical Rows/Cols for every Mat-source TMOV.
    - Other TMOV paths derive their transfer extent from runtime valid shapes;
      there is no generic equal-shape or transposed-shape requirement.

    The complete ``offset + transfer extent`` window is not checked here because
    the effective extent may be set dynamically. Runtime DFX handles that bound.
    """
    out_type = out.type
    src_type = src.type
    dst_shape = _tile_shape_ints(out_type)
    src_shape = _tile_shape_ints(src_type)
    if dst_shape is None or src_shape is None:
        return  # symbolic shape — cannot verify statically
    if acc_to_vec_mode is not None:
        return  # Acc→Vec split mode: shape relation determined by mode
    src_mem = getattr(getattr(src_type, "memref", None), "memory_space_", None)
    if offset is None:
        if src_mem == MemorySpace.Mat and dst_shape != src_shape:
            raise InvalidShape(
                f"pl.move: dst tile shape {dst_shape} != src tile shape {src_shape} "
                "— without offset, a Mat-source move requires equal shapes.",
                span=actual_span,
            )
        return
    for axis, (d, s) in enumerate(zip(dst_shape, src_shape)):
        if d > s:
            raise InvalidShape(
                f"pl.move: with offset, dst dim {axis} ({d}) exceeds src dim ({s}) "
                f"— dst must be a sub-rectangle of src "
                f"(dst shape={dst_shape}, src shape={src_shape}).",
                span=actual_span,
            )


def _maybe_dispatch_to_insert(
    out: Expr,
    src: Expr,
    offset: Expr | None,
    acc_to_vec_mode: AccToVecMode | None,
) -> bool:
    """Whether ``move`` should dispatch to ``insert`` (TINSERT).

    Rule A: offset given, dst is a super-block of src (fractal-transpose
    pairs compare via dst[::-1]).
    Rule B: equal-shape fractal transpose on Vec->Mat (TMOV can't convert layout).
    """
    if acc_to_vec_mode is not None:
        return False

    out_type = out.type
    src_type = src.type
    if not isinstance(out_type, _IRTileType) or not isinstance(src_type, _IRTileType):
        return False
    dst_shape = _tile_shape_ints(out_type)
    src_shape = _tile_shape_ints(src_type)
    if dst_shape is None or src_shape is None:
        return False

    dst_layout = _tile_layout(out_type)
    src_layout = _tile_layout(src_type)
    layout_transpose = (
        dst_layout is not None
        and src_layout is not None
        and _is_fractal_transpose(dst_layout, src_layout)
    )
    dst_cmp = dst_shape[::-1] if layout_transpose else dst_shape

    is_shape_insert = (
        offset is not None
        and all(d >= s for d, s in zip(dst_cmp, src_shape))
        and any(d > s for d, s in zip(dst_cmp, src_shape))
    )

    out_mem = getattr(getattr(out_type, "memref", None), "memory_space_", None)
    src_mem = getattr(getattr(src_type, "memref", None), "memory_space_", None)
    is_equal_transpose_vec_mat = (
        layout_transpose
        and dst_shape == src_shape[::-1]
        and src_mem == MemorySpace.Vec
        and out_mem == MemorySpace.Mat
    )

    return is_shape_insert or is_equal_transpose_vec_mat


def _ir_move(
    dst_tile: Expr,
    src_tile: Expr,
    offset: Expr | None = None,
    *,
    span: Span | None = None,
    acc_to_vec_mode: AccToVecMode | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    scale: Any = None,
    phase: STPhase | None = None,
) -> Expr:
    actual_span = span or _span()
    if not isinstance(dst_tile.type, _ir_core.TileType):
        raise InvalidType(f"move: dst must be a Tile, got {type(dst_tile.type).__name__}")
    if not isinstance(src_tile.type, _ir_core.TileType):
        raise InvalidType(f"move: src must be a Tile, got {type(src_tile.type).__name__}")

    offset_tuple = None
    if offset is not None:
        offset_tuple = _to_make_tuple(offset, actual_span)
        if len(offset_tuple.elements) != 2:
            raise InvalidArgument("move: offset must contain exactly 2 elements")

    if phase is not None:
        require_enum(phase, STPhase, subject="move: phase")

    if _maybe_dispatch_to_insert(dst_tile, src_tile, offset_tuple, acc_to_vec_mode):
        actual_offset = offset_tuple if offset_tuple is not None else [0, 0]
        return _ir_insert(
            dst_tile,
            src_tile,
            actual_offset,
            span=actual_span,
            relu_pre_mode=relu_pre_mode,
            scale=scale,
            phase=phase,
        )

    _dst_mem = getattr(getattr(dst_tile.type, "memref", None), "memory_space_", None)
    _src_mem = getattr(getattr(src_tile.type, "memref", None), "memory_space_", None)
    _supported_move_paths = {
        (MemorySpace.Mat, MemorySpace.Left),
        (MemorySpace.Mat, MemorySpace.Right),
        (MemorySpace.Acc, MemorySpace.Mat),
        (MemorySpace.Acc, MemorySpace.Vec),
        (MemorySpace.Vec, MemorySpace.Vec),
        (MemorySpace.Mat, MemorySpace.Scaling),
        (MemorySpace.Vec, MemorySpace.Mat),
        (MemorySpace.Mat, MemorySpace.Bias),
        (MemorySpace.Mat, MemorySpace.ScaleLeft),
        (MemorySpace.Mat, MemorySpace.ScaleRight),
    }
    if (_src_mem, _dst_mem) not in _supported_move_paths:
        raise NotSupported(
            f"move: unsupported data path src({_src_mem.name})->dst({_dst_mem.name}), "
            f"supported paths: Mat->Left, Mat->Right, Mat->Scaling, Mat->Bias, Mat->ScaleLeft, "
            f"Mat->ScaleRight, Acc->Mat, Acc->Vec, Vec->Vec, Vec->Mat"
        )
    if acc_to_vec_mode is not None and (_src_mem, _dst_mem) not in {
        (MemorySpace.Acc, MemorySpace.Vec),
    }:
        raise NotSupported("move: acc_to_vec_mode is only supported for Acc-to-Vec moves")

    from pypto_pro.runtime.jit import get_current_arch

    _dual_modes = {AccToVecMode.DualModeSplitM, AccToVecMode.DualModeSplitN}
    if acc_to_vec_mode in _dual_modes:
        if get_current_arch() == NpuArch.DAV_9201:
            raise NotSupported(
                "move: acc_to_vec_mode DualModeSplitM/DualModeSplitN is not yet supported on "
                "arch '9201' because pto-isa does not currently support dual-destination Acc-to-Vec moves "
                "on dav-9201",
                span=actual_span,
            )

    if relu_pre_mode is not None and (_src_mem, _dst_mem) not in {
        (MemorySpace.Acc, MemorySpace.Vec),
        (MemorySpace.Acc, MemorySpace.Mat),
    }:
        raise NotSupported("move: relu_pre_mode is only supported for Acc-to-Vec or Acc-to-Mat moves")
    if phase is not None and (_src_mem, _dst_mem) not in {
        (MemorySpace.Acc, MemorySpace.Vec),
        (MemorySpace.Acc, MemorySpace.Mat),
    }:
        raise NotSupported("move: phase is only supported for Acc-to-Vec or Acc-to-Mat moves")
    if phase is not None and offset_tuple is not None and _dst_mem == MemorySpace.Vec:
        raise NotSupported("move: Acc-to-Vec phase cannot be combined with offset")

    _check_move_shape_compat(dst_tile, src_tile, offset_tuple, acc_to_vec_mode, actual_span)
    if offset_tuple is not None:
        _validate_offsets("move", offset_tuple.elements, src_tile.type.shape)

    pre_quant_scalar, fp_tile = _resolve_scale_param(scale)
    if fp_tile is not None and (_src_mem, _dst_mem) not in {
        (MemorySpace.Acc, MemorySpace.Vec),
        (MemorySpace.Acc, MemorySpace.Mat),
    }:
        raise NotSupported("move: Scaling Tile scale only supports Acc-to-Vec or Acc-to-Mat moves")

    is_quant = pre_quant_scalar is not None or fp_tile is not None
    is_acc_to_mat = _src_mem == MemorySpace.Acc and _dst_mem == MemorySpace.Mat
    if offset_tuple is not None and is_quant and not is_acc_to_mat:
        raise NotSupported(
            "move: offset cannot be combined with scale — the fixpipe quantization "
            "paths (per-channel Tile or per-tensor scalar) do not support sub-block "
            "extraction offsets"
        )

    # The CCE backend lowers every non-insert move with an offset to TEXTRACT.
    # Its layout/dtype matrix differs from plain TMOV (notably Vec -> Vec does
    # not perform ND -> NZ conversion), so validate it with the extract table.
    transfer_kind = "extract" if offset_tuple is not None else "move"
    _check_layout_dtype(transfer_kind, src_tile, dst_tile, quant=is_quant)
    if offset_tuple is not None:
        _validate_transfer_offset_alignment("move", "extract", src_tile, dst_tile, offset_tuple.elements)

    if is_quant and acc_to_vec_mode in _dual_modes:
        raise NotSupported(
            "scale cannot be combined with dual-mode acc_to_vec_mode — the fixpipe dual-destination "
            "control word does not support quantization (hardware limit); use a single-vec mode "
            "(SingleModeVec0/SingleModeVec1) or drop the scale"
        )
    kwargs: dict[str, Any] = {}
    if acc_to_vec_mode is not None:
        kwargs["acc_to_vec_mode"] = acc_to_vec_mode
    if relu_pre_mode is not None:
        kwargs["relu_pre_mode"] = relu_pre_mode
    if phase is not None:
        kwargs["phase"] = phase
    if fp_tile is not None:
        args = [dst_tile, src_tile]
        if offset_tuple is not None:
            args.append(offset_tuple)
        args.append(fp_tile)
        return _ir_core.create_op_call(block_ir_op("move"), args, kwargs, actual_span)
    args = [dst_tile, src_tile]
    if offset_tuple is not None:
        args.append(offset_tuple)
    if pre_quant_scalar is not None:
        args.append(pre_quant_scalar)
    return _ir_core.create_op_call(block_ir_op("move"), args, kwargs, actual_span)


def _normalize_2d_sequence(value: Any, parameter: str, span: Span) -> tuple[Expr, Expr]:
    sequence = _to_make_tuple(value, span)
    if len(sequence.elements) != 2:
        raise InvalidArgument(f"{parameter} must contain exactly 2 elements")
    return sequence.elements[0], sequence.elements[1]


def _ir_insert(
    dst_tile: Expr,
    src_tile: Expr,
    offset: Expr,
    *,
    span: Span | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    scale: Any = None,
    phase: STPhase | None = None,
) -> Expr:
    actual_span = span or _span()
    if phase is not None:
        require_enum(phase, STPhase, subject="insert: phase")

    dst_mem = getattr(getattr(dst_tile.type, "memref", None), "memory_space_", None)
    src_mem = getattr(getattr(src_tile.type, "memref", None), "memory_space_", None)
    if (scale is not None or relu_pre_mode is not None or phase is not None) and not (
        src_mem == MemorySpace.Acc and dst_mem == MemorySpace.Mat
    ):
        raise NotSupported("insert: scale, relu_pre_mode, and phase are only supported for Acc-to-Mat inserts")
    row, col = _normalize_2d_sequence(offset, "offset", actual_span)
    dst_shape = dst_tile.type.shape
    if _is_fractal_transpose(_tile_layout(dst_tile.type), _tile_layout(src_tile.type)):
        dst_shape = dst_shape[::-1]
    _validate_offsets("insert", [row, col], dst_shape)
    pre_quant_scalar, fp_tile = _resolve_scale_param(scale)
    _check_layout_dtype("insert", src_tile, dst_tile, quant=pre_quant_scalar is not None or fp_tile is not None)
    _validate_transfer_offset_alignment("insert", "insert", src_tile, dst_tile, (row, col))

    kwargs: dict[str, Any] = {}
    if relu_pre_mode is not None:
        kwargs["relu_pre_mode"] = relu_pre_mode
    if phase is not None:
        kwargs["phase"] = phase
    args = [dst_tile, src_tile, row, col]
    if fp_tile is not None:
        args.append(fp_tile)
    elif pre_quant_scalar is not None:
        args.append(pre_quant_scalar)
    return _ir_core.create_op_call(block_ir_op("insert"), args, kwargs, actual_span)


def _ir_sel(out: Expr, mask: Expr, lhs: Expr, rhs: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _SEL_DTYPES, subject="sel")
    # AscendC SelectImpl enforces dst/src0/src1 to share one type via its
    # template; the byte-mask tile is intentionally not dtype-constrained.
    require_dtypes_match([getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None)], dt, subject="sel")
    return _ir_core.create_op_call(block_ir_op("sel"), [out, mask, lhs, rhs, tmp], {}, span or _span())


def _ir_sels(out: Expr, mask: Expr, src: Expr, tmp: Expr, scalar: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _SEL_DTYPES, subject="sels")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="sels")
    actual_span = span or _span()
    from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

    check_const_expr_fits_dtype(scalar, dt, api="pl.select")
    return _ir_core.create_op_call(block_ir_op("sels"), [out, mask, src, tmp, scalar], {}, actual_span)


@op_impl("neg")
def _ir_neg(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _NEG_DTYPES, subject="neg")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="neg")
    return _ir_core.create_op_call(block_ir_op("neg"), [out, src], {}, span or _span())


@op_impl("not_")
def _ir_not(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BITWISE_DTYPES, subject="not")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="not")
    return _ir_core.create_op_call(block_ir_op("not"), [out, src], {}, span or _span())


@op_impl("abs")
def _ir_abs(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _ABS_DTYPES, subject="abs")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="abs")
    return _ir_core.create_op_call(block_ir_op("abs"), [out, src], {}, span or _span())


@op_impl("relu")
def _ir_relu(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _RELU_DTYPES, subject="relu")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="relu")
    return _ir_core.create_op_call(block_ir_op("relu"), [out, src], {}, span or _span())


@op_impl("exp")
def _ir_exp(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="exp")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="exp")
    return _ir_core.create_op_call(block_ir_op("exp"), [out, src], {}, span or _span())


@op_impl("log")
def _ir_log(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="log")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="log")
    return _ir_core.create_op_call(block_ir_op("log"), [out, src], {}, span or _span())


@op_impl("sqrt")
def _ir_sqrt(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="sqrt")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="sqrt")
    return _ir_core.create_op_call(block_ir_op("sqrt"), [out, src], {}, span or _span())


@op_impl("rsqrt")
def _ir_rsqrt(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="rsqrt")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="rsqrt")
    return _ir_core.create_op_call(block_ir_op("rsqrt"), [out, src], {}, span or _span())


@op_impl("recip")
def _ir_recip(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _RECIP_DTYPES, subject="recip")
    require_dtypes_match([getattr(src.type, "dtype", None)], dt, subject="recip")
    return _ir_core.create_op_call(block_ir_op("recip"), [out, src], {}, span or _span())


@op_impl("axpy")
def _ir_axpy(self, out: Expr, src: Expr, alpha: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    src_dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _AXPY_DTYPES, subject="axpy")
    require_dtype_in(src_dt, _AXPY_DTYPES, subject="axpy")
    if dt != src_dt:
        if not (dt == DataType.FP32 and src_dt == DataType.FP16):
            raise InvalidType(
                f"axpy: dtype mismatch between out ({dt}) and src ({src_dt}). "
                f"Supported: same type, or src=FP16 + out=FP32."
            )
    # TAXPY takes the multiplier in the src dtype (the half upconvert path
    # keeps a half scalar), so the scalar must be representable there.
    actual_span = span or _span()
    from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

    check_const_expr_fits_dtype(alpha, dt, api="pl.axpy")
    return _ir_core.create_op_call(block_ir_op("axpy"), [out, src, alpha], {}, actual_span)


@op_impl("add_relu")
def _ir_add_relu(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _RELU_DTYPES, subject="add_relu")
    require_dtypes_match([getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None)], dt, subject="add_relu")
    return _ir_core.create_op_call(block_ir_op("add_relu"), [out, lhs, rhs], {}, span or _span())


@op_impl("sub_relu")
def _ir_sub_relu(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _RELU_DTYPES, subject="sub_relu")
    require_dtypes_match([getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None)], dt, subject="sub_relu")
    return _ir_core.create_op_call(block_ir_op("sub_relu"), [out, lhs, rhs], {}, span or _span())


@op_impl("fused_mul_add")
def _ir_fused_mul_add(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    # TMADD semantics: out = lhs * out + rhs (out is an in-place operand).
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="fused_mul_add")
    require_dtypes_match(
        [getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None)],
        dt,
        subject="fused_mul_add",
    )
    return _ir_core.create_op_call(block_ir_op("fused_mul_add"), [out, lhs, rhs], {}, span or _span())


@op_impl("fused_mul_add_relu")
def _ir_fused_mul_add_relu(self, out: Expr, a: Expr, b: Expr, *, span: Span | None = None) -> Expr:
    # TMADD + ReLU fusion: out = relu(a * out + b).
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="fused_mul_add_relu")
    require_dtypes_match(
        [getattr(a.type, "dtype", None), getattr(b.type, "dtype", None)],
        dt,
        subject="fused_mul_add_relu",
    )
    return _ir_core.create_op_call(block_ir_op("fused_mul_add_relu"), [out, a, b], {}, span or _span())


@op_impl("mul_add_dst")
def _ir_mul_add_dst(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    # TMULA semantics: out = lhs * rhs + out (out is an in-place accumulator).
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="mul_add_dst")
    require_dtypes_match(
        [getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None)],
        dt,
        subject="mul_add_dst",
    )
    return _ir_core.create_op_call(block_ir_op("mul_add_dst"), [out, lhs, rhs], {}, span or _span())


@op_impl("xor")
def _ir_xor(self, out: Expr, lhs: Expr, rhs: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BITWISE_DTYPES, subject="xor")
    if not isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

        require_dtypes_match([getattr(lhs.type, "dtype", None)], dt, subject="xor")
        check_const_expr_fits_dtype(rhs, dt, api="pl.xor")
        # TXORS carries a trailing unused tmp operand in the pto-isa signature.
        return _ir_core.create_op_call(block_ir_op("xors"), [out, lhs, rhs, tmp], {}, span or _span())
    require_dtypes_match([getattr(lhs.type, "dtype", None), getattr(rhs.type, "dtype", None)], dt, subject="xor")
    return _ir_core.create_op_call(block_ir_op("xor"), [out, lhs, rhs, tmp], {}, span or _span())


@op_impl("expands")
def _ir_expands(self, out: Expr, scalar: Expr, *, span: Span | None = None) -> Expr:
    from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

    out_dtype = getattr(out.type, "dtype", None)
    require_dtype_in(out_dtype, _EXPANDS_DTYPES, subject="expands")
    # The splat value lands in the out tile, so it has to be representable there. Without this the
    # scalar was only bounded by the IR storage band, which is far wider than a narrow tile dtype.
    check_const_expr_fits_dtype(scalar, out_dtype, api="pl.expands")
    return _ir_core.create_op_call(block_ir_op("expands"), [out, scalar], {}, span or _span())


@op_impl("row_sum")
def _ir_row_sum(self, out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _SUM_DTYPES, subject="row_sum")
    require_dtypes_match([dt], getattr(out.type, "dtype", None), subject="row_sum")
    return _ir_core.create_op_call(block_ir_op("row_sum"), [out, src, tmp], {}, span or _span())


@op_impl("col_sum")
def _ir_col_sum(self, out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _SUM_DTYPES, subject="col_sum")
    require_dtypes_match([dt], getattr(out.type, "dtype", None), subject="col_sum")
    return _ir_core.create_op_call(block_ir_op("col_sum"), [out, src, tmp], {}, span or _span())


def _validate_validshape_bounds(tile: Expr, valid_shape: Sequence[Expr]) -> None:
    """Check each valid dimension against the tile's static bounds when available."""
    tile_type = getattr(tile, "type", None)
    if isinstance(tile_type, _IRTileType) and tile_type.tile_view is not None:
        if any(isinstance(dim, ConstInt) and dim.value != -1 for dim in tile_type.tile_view.valid_shape):
            raise InvalidOperation(
                "set_validshape cannot modify a TileType with static valid_shape; "
                "omit valid_shape or declare valid_shape=[-1, -1]"
            )

    for i, (dim_name, dim) in enumerate(zip(("row", "col"), valid_shape, strict=True)):
        upper_bound = _const_int_value(tile_type.shape[i]) if isinstance(tile_type, _IRTileType) else None
        # Zero denotes an empty valid region, e.g. a fillpad source.
        require_int(
            dim, subject=f"set_validshape {dim_name}", lower_bound=0, upper_bound=upper_bound,
        )


def _ir_set_validshape(
    tile: Expr,
    shape: Expr,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    shape_0, shape_1 = _normalize_2d_sequence(shape, "shape", actual_span)
    _validate_validshape_bounds(tile, (shape_0, shape_1))
    return _ir_core.create_op_call(block_ir_op("set_validshape"), [tile, shape_0, shape_1], {}, actual_span)


def _check_scalar_access_supported(op_name: str, container: Expr) -> None:
    """Validate the container constraints for scalar getval/setval access."""
    tile_type = container.type
    if isinstance(tile_type, _IRTileType):
        memref = getattr(tile_type, "memref", None)
        memory_space = getattr(memref, "memory_space_", None)
        if memory_space is not None and memory_space != MemorySpace.Vec:

            raise InvalidOperation(
                f"{op_name}: Tile element access requires a Vec-memory Tile (UB), got {memory_space.name}",
                parser_retry=True,
            )

    require_scalar_dtype(
        container.type.dtype, subject=f"{op_name} does not support container",
    )


def _validate_load_operands(
    out: Expr,
    tensor: Expr,
    tile_dims: list[int] | None,
    op_name: str,
) -> tuple[list[Any], list[int] | None, Any | None]:
    if not isinstance(out.type, _ir_core.TileType):
        raise InvalidType(f"{op_name}: dst must be a Tile, got {type(out.type).__name__}")
    dst_mem = getattr(getattr(out.type, "memref", None), "memory_space", None)
    if dst_mem is not None and dst_mem not in (_ir_core.MemorySpace.Vec, _ir_core.MemorySpace.Mat):
        raise InvalidOperation(f"{op_name}: dst tile must be in Vec (UB) or Mat (L1) memory, got {dst_mem.name}")

    if not isinstance(tensor.type, _ir_core.TensorType):
        raise InvalidType(f"{op_name}: src must be a Tensor, got {type(tensor.type).__name__}")

    tensor_ndim = len(tensor.type.shape)
    tile_shape = list(out.type.shape)
    tile_dims, access_size = _validate_tile_dims(tile_dims, tensor_ndim, tile_shape, op_name)
    return tile_shape, tile_dims, access_size


def _validate_merged_row_load(
    out: Expr, tensor: Expr, tile_dims, op_name: str, order: list[int] | None = None
) -> "int | None":
    """Validate the merged-row form of ``load``: ``order`` names 3 tensor axes for a rank-2 Tile.

    The Tile's row index is then the flattening of two tensor axes,
    ``row = i_outer * inner + i_inner`` -- which is what lets a Tensor [B, S, N, G, D] feed a
    matmul that wants (S, G) as one axis. 3510's nd2nz DMA issues it as ``ndNum`` ND matrices of
    ``inner`` rows each, through TLOAD. This returns ``inner``,
    or None when the call is the ordinary 2-axis form.

    Two spellings are accepted, and the dst Tile says which one is meant:

      ``order=[1, 3, 4]``  ascending, NZ dst [merged, cols] -- the operand as nd2nz writes it.
      ``order=[4, 3, 1]``  the exact reverse, ZN dst [cols, merged] -- the transposed operand.

    The reversed spelling is not a different transfer: nd2nz cannot transpose, and it does not
    have to. An NZ [M, K] tile and a ZN [K, M] tile at one address are the *same bytes* -- with
    the fractal's inner extents swapped by the layout (NZ: InnerRows R0, InnerCols C0; ZN: the
    reverse), both offsets reduce to ``M*C0*(c/C0) + C0*r + (c % C0)`` -- so the DMA is issued
    unchanged and only the destination's label differs. The Mat->L0 move then realizes the
    transpose. That is why the reverse is the whole-list flip, matching the 2-axis convention
    where ``order=[1, 0]`` means the transpose of ``order=[0, 1]``: it names the transpose of
    what the ascending order loads, so the merged axis still runs (outer, inner) along it.

    ``order`` is the caller's list before ``_resolve_order`` sorted it. The sorted form is not
    enough here: sorting maps [1, 4, 3], [1, 3, 4] and [4, 3, 1] onto the same ``tile_dims``,
    and they mean three different things (a different column axis, and the two spellings above),
    so validating the sorted list would accept [1, 4, 3] and silently transfer another one's
    data.
    """
    if tile_dims is None or len(tile_dims) != 3:
        return None

    if out.type.dtype != tensor.type.dtype:
        raise InvalidType(f"{op_name}: dtype mismatch between dst tile and src tensor")
    if int(tensor.type.dtype.get_bit()) not in (8, 16, 32):
        raise InvalidType(f"{op_name}: merged-row loads require 8, 16 or 32-bit elements")
    view = getattr(tensor.type, "tensor_view", None)
    if view is not None and view.stride and _const_int_value(view.stride[-1]) != 1:
        raise InvalidArgument(f"{op_name}: merged-row loads require a compile-time column stride of 1")

    tile_shape = list(out.type.shape)
    if len(tile_shape) != 2:
        raise InvalidArgument(
            f"{op_name}: a 3-axis order merges two tensor axes into the Tile's row axis, so the "
            f"Tile must be rank 2, got shape {tile_shape}"
        )

    ascending = sorted(tile_dims)
    raw_order = list(order) if order is not None else ascending
    is_transposed = raw_order == ascending[::-1]
    if raw_order != ascending and not is_transposed:
        raise InvalidArgument(
            f"{op_name}: a merged-row order must be ascending, or its exact reverse for a "
            f"transposed operand, got {raw_order}. nd2nz writes the two merged axes to the "
            f"Tile's rows and the Tensor's innermost axis to its columns; any other permutation "
            f"names a column axis it cannot walk. Use {ascending} for [merged, cols] into an NZ "
            f"Tile, or {ascending[::-1]} for the transpose, [cols, merged] into a ZN Tile."
        )

    dst_mem = getattr(getattr(out.type, "memref", None), "memory_space", None)
    if dst_mem is not None and dst_mem != _ir_core.MemorySpace.Mat:
        raise InvalidArgument(
            f"{op_name}: merged-row loads currently require a Mat (L1) destination, got {dst_mem.name}. "
            f"For Vec (UB), load the selected data into a packed tile before using pl.reinterpret; "
            f"reinterpret alone does not gather noncontiguous GM data."
        )

    dst_layout = _tile_layout(out.type)
    if _hw_attr(out.type.hardware_info, "fractal") not in (None, 512):
        raise InvalidFormat(f"{op_name}: merged-row loads require a 512-byte fractal")
    if is_transposed:
        # The transposed spelling is the same DMA into the same bytes under the other label, so
        # the Tile must carry that label: NZ's transpose pair, with the shape reversed to match.
        if dst_layout != Layout.ZN:
            raise InvalidArgument(
                f"{op_name}: the reversed merged order {raw_order} loads the transposed operand, "
                f"so the dst tile must be the ZN alias of the NZ tile {ascending} would fill: "
                f"shape [cols, merged] and layout=pl.ZN, got shape {tile_shape} and layout "
                f"{dst_layout.name if dst_layout is not None else None}. The bytes are identical "
                f"either way -- only the label decides which orientation the Mat->L0 move reads."
            )
    elif dst_layout != Layout.NZ:
        raise InvalidFormat(f"{op_name}: a 3-axis order requires an NZ dst tile (the nd2nz destination layout)")

    tensor_ndim = len(tensor.type.shape)
    if tile_dims[2] != tensor_ndim - 1:
        raise InvalidArgument(
            f"{op_name}: a 3-axis order's last axis must be the Tensor's innermost axis "
            f"({tensor_ndim - 1}), got {tile_dims[2]}: nd2nz reads each row as contiguous elements "
            f"and has no column stride"
        )

    # The merged axis is the Tile's rows as nd2nz writes them -- the ZN alias reverses the shape,
    # so it is that Tile's columns.
    merged_dim = tile_shape[1] if is_transposed else tile_shape[0]
    rows = _const_int_value(merged_dim)
    if rows is None:
        raise InvalidShape(f"{op_name}: a 3-axis order needs a compile-time Tile row count, got {merged_dim}")
    if not 0 < rows <= 0xFFFF:
        raise InvalidShape(f"{op_name}: nd2nz destination row stride is 16-bit; got {rows} merged rows")

    inner = _const_int_value(tensor.type.shape[tile_dims[1]])
    if inner is None:
        # Keep a dynamic middle extent for lowering. It splits the runtime valid
        # region into complete matrices and an optional partial final matrix.
        return -1
    if inner <= 0:
        raise InvalidArgument(
            f"{op_name}: a 3-axis order's middle axis {tile_dims[1]} must have a positive extent, got {inner}"
        )
    # Lowering splits the valid merged extent for static and dynamic inner extents alike.
    # An inner extent larger than the tile is clipped by TLOAD lowering, so it need not fit
    # the DMA's 16-bit nValue field. Defer extents exceeding the int attribute to the source
    # type instead of narrowing them before that normalization.
    if inner > 0x7FFFFFFF:
        return -1
    return inner


def _build_store_kwargs(
    *,
    relu_pre_mode: ReluPreMode | None,
    tile_dims: list[int] | None,
    tensor_ndim: int,
    tile_ndim: int,
    atomic: AtomicType,
    phase: STPhase | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if relu_pre_mode is not None:
        kwargs["relu_pre_mode"] = relu_pre_mode
    if tile_dims is not None:
        default_tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
        if tile_dims != default_tile_dims:
            kwargs["tile_dims"] = tile_dims
    if atomic != AtomicType.AtomicNone:
        kwargs["atomic"] = atomic
    if phase is not None:
        kwargs["phase"] = phase
    return kwargs


# Per-op supported dtype sets (aligned with ISA static_assert constraints)
_BINARY_DTYPES: tuple[DataType, ...] = (
    DataType.INT8, DataType.UINT8, DataType.INT16, DataType.UINT16,
    DataType.INT32, DataType.UINT32, DataType.INT64, DataType.UINT64,
    DataType.FP16, DataType.FP32, DataType.BF16,
)
_MUL_DTYPES: tuple[DataType, ...] = tuple(
    d for d in _BINARY_DTYPES if d not in (DataType.INT8, DataType.UINT8)
)
_DIV_DTYPES: tuple[DataType, ...] = tuple(
    d for d in _MUL_DTYPES if d != DataType.BF16
)
# Bitwise/shift ISA admits 64-bit integers: TAND/TOR/TXOR (and the *S scalar
# forms) are sizeof-based b8/b16/b32/b64, TSHL/TSHRS and TSHR/TSHRS pin
# int64/uint64 explicitly, TNOT covers u/i 8..64.
_BITWISE_DTYPES: tuple[DataType, ...] = (
    DataType.INT8, DataType.UINT8, DataType.INT16, DataType.UINT16,
    DataType.INT32, DataType.UINT32, DataType.INT64, DataType.UINT64,
)
_NEG_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.UINT16, DataType.INT32, DataType.UINT32,
    DataType.FP16, DataType.FP32, DataType.BF16,
)
_ABS_DTYPES: tuple[DataType, ...] = (
    DataType.INT8, DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32,
)
_RELU_DTYPES: tuple[DataType, ...] = (DataType.INT32, DataType.FP16, DataType.FP32)
_AXPY_DTYPES: tuple[DataType, ...] = (DataType.FP16, DataType.FP32)
_CMP_DTYPES: tuple[DataType, ...] = _BINARY_DTYPES
_EXPANDS_DTYPES: tuple[DataType, ...] = _BINARY_DTYPES
_SUM_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32,
    DataType.INT64, DataType.UINT64,
)
_SEL_DTYPES: tuple[DataType, ...] = _BINARY_DTYPES + (DataType.BOOL,)
# init_output fills GM through a flat [1, numel] view, so only the element width
# matters: the doc-supported b8/b16/b32/b64 integer and float types.
_INIT_OUTPUT_DTYPES: tuple[DataType, ...] = (
    DataType.UINT8, DataType.INT8,
    DataType.UINT16, DataType.INT16, DataType.BF16, DataType.FP16,
    DataType.UINT32, DataType.INT32, DataType.FP32,
    DataType.UINT64, DataType.INT64,
)
# Row/col expand variants broadcast a per-row (per-col) vector over a tile.
# Row/col expand ISA admits the full binary set: TROWEXPAND/TCOLEXPAND are
# sizeof-based b8/b16/b32/b64 and the dedicated TROWEXPANDADD/MAX/MIN/MUL/
# SUB/DIV (plus TCOLEXPANDADD via TCOLEXPANDOP) pin int64/uint64 explicitly.
_ROW_EXPAND_DTYPES: tuple[DataType, ...] = _BINARY_DTYPES
# Unary float math (TEXP/TLOG/TSQRT/TRSQRT: half/float only).
_FLOAT_UNARY_DTYPES: tuple[DataType, ...] = (DataType.FP16, DataType.FP32)
# TRECIP delegates to TDIVS(dst, 1, src), which also admits integer 1/x.
_RECIP_DTYPES: tuple[DataType, ...] = (DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32)
# TREM/TREMS (3510): half/float/int16/uint16/int32/uint32/int64/uint64.
_REM_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.UINT16, DataType.INT32, DataType.UINT32,
    DataType.INT64, DataType.UINT64, DataType.FP16, DataType.FP32,
)
# TROWPROD (3510): half/float/int16/int32; TCOLPROD adds uint16/uint32/bfloat16.
_ROW_PROD_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.INT32, DataType.FP16, DataType.FP32,
)
_COL_PROD_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.UINT16, DataType.INT32, DataType.UINT32,
    DataType.FP16, DataType.FP32, DataType.BF16,
)
# TROWARGMAX/TROWARGMIN (3510) 3-arg idx-only form: half/float source, 32-bit index.
_ARG_IDX32_DTYPES: tuple[DataType, ...] = (DataType.INT32, DataType.UINT32)
# TGATHER/TSCATTER index tiles are 16/32-bit integers.
_GATHER_IDX_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.UINT16, DataType.INT32, DataType.UINT32,
)
# TMRGSORT/TSORT32 (3510) operate on half/float only.
_SORT_DTYPES: tuple[DataType, ...] = (DataType.FP16, DataType.FP32)
# THISTOGRAM (3510) counts 16/32-bit unsigned source elements into a u32 bin tile
# with a u8 filter index (mirrors the block.histogram IR registration).
_HISTOGRAM_SRC_DTYPES: tuple[DataType, ...] = (DataType.UINT16, DataType.UINT32)
_HISTOGRAM_IDX_DTYPES: tuple[DataType, ...] = (DataType.UINT8,)
_HISTOGRAM_DST_DTYPES: tuple[DataType, ...] = (DataType.UINT32,)
# TCI (3510) fills 16/32-bit integer tiles with arithmetic sequences.
_FILL_INDEX_DTYPES: tuple[DataType, ...] = (
    DataType.INT16, DataType.UINT16, DataType.INT32, DataType.UINT32,
)
# Fused binary+cast ops compute in the source domain before TCVT:
# add/sub_relu_cast run TRELU on the source (half/float/int32), mul_cast runs
# TMUL (no 8-bit).
_CAST_BINARY_SRC_DTYPES: dict[str, tuple[DataType, ...]] = {
    "add_relu_cast": _RELU_DTYPES,
    "sub_relu_cast": _RELU_DTYPES,
    "mul_cast": _MUL_DTYPES,
}


def _resolve_order(
    order: list[int] | None,
    tensor_ndim: int,
    tile_ndim: int,
    op_name: str,
) -> tuple[list[int], bool]:
    """Resolve the ``order`` kwarg into ``(tile_dims, is_transpose)``.

    ``order`` maps each Tile dimension to a Tensor axis (absolute index).
    Ascending order means no transposition; descending order means transposition
    (DN layout). Internally we decompose into ``tile_dims`` (sorted ascending,
    consumed by C++ codegen) and ``is_transpose`` (bool, consumed by codegen).
    """
    if order is None:
        if tensor_ndim == 1:
            return None, False
        order = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    is_transpose = len(order) >= 2 and order[0] > order[1]
    tile_dims = sorted(order)
    return tile_dims, is_transpose


def _validate_rank1_tensor_tile_shape(tile_shape: list[Any], op_name: str) -> Any:
    shape_values = [_const_int_value(dim) for dim in tile_shape]
    if 1 not in shape_values:
        raise InvalidShape(
            f"{op_name}: rank-1 Tensor requires a rank-2 Tile with one dimension equal to 1, "
            f"got Tile shape {shape_values}"
        )

    access_size = tile_shape[0] if shape_values[1] == 1 else tile_shape[1]
    return access_size


def _validate_tile_dims(
    tile_dims: list[int] | None,
    tensor_ndim: int,
    tile_shape: list[Any],
    op_name: str,
) -> tuple[list[int] | None, Any | None]:
    if tensor_ndim == 1:
        if tile_dims is not None:
            raise InvalidArgument(
                f"{op_name}: order is not supported for rank-1 Tensors (the axis mapping is "
                f"expressed by the Tile shape), got {tile_dims}"
            )
        access_size = _validate_rank1_tensor_tile_shape(tile_shape, op_name)
        return None, access_size

    tile_ndim = len(tile_shape)
    if (tile_dims is not None and len(tile_dims) != tile_ndim
            and not (op_name == "load" and tile_ndim == 2 and len(tile_dims) == 3)):
        expected = f"{tile_ndim} axes"
        if op_name == "load" and tile_ndim == 2:
            expected += " (or 3 axes for a merged-row load)"
        raise InvalidShape(
            f"{op_name}: order must name {expected}, got {tile_dims}; omit order to use the default mapping"
        )

    if tile_dims is None:
        tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    if len(set(tile_dims)) != len(tile_dims):
        raise InvalidShape(f"{op_name}: order axes must be unique, got {tile_dims}")
    for dim in tile_dims:
        if dim < 0 or dim >= tensor_ndim:
            raise InvalidShape(f"{op_name}: order axis {dim} is out of range for Tensor rank {tensor_ndim}")
    return tile_dims, None


def _ir_getval(
    container: Expr, offset: Expr, *, preserve_dtype: bool = False, span: Span | None = None
) -> Expr:
    actual_span = span or _span()
    _ctype = container.type
    if not isinstance(_ctype, (_ir_core.TileType, _ir_core.TensorType)):

        raise InvalidVal(
            f"getval: 'container' must be a Tile or Tensor, got {type(_ctype).__name__}",
            hint="getval reads a scalar from a Tile/Tensor slot; to access a struct/tiling "
            "field, use attribute access (e.g. tiling.axis1) instead.",
            parser_retry=True,
        )
    _check_scalar_access_supported("getval", container)
    kwargs = {"preserve_dtype": True} if preserve_dtype else {}
    return _ir_core.create_op_call(block_ir_op("getval"), [container, offset], kwargs, actual_span)


def _ir_setval(container: Expr, offset: Expr, value: Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    _ctype = container.type
    if not isinstance(_ctype, (_ir_core.TileType, _ir_core.TensorType)):

        raise InvalidVal(
            f"setval: 'container' must be a Tile or Tensor, got {type(_ctype).__name__}",
            hint="setval writes a scalar into a Tile/Tensor slot; to write a struct/tiling "
            "field, use attribute assignment (e.g. tiling.axis1 = ...) instead.",
            parser_retry=True,
        )
    _check_scalar_access_supported("setval", container)
    from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

    check_const_expr_fits_dtype(value, container.type.dtype, api="pl.setval")
    return _ir_core.create_op_call(block_ir_op("setval"), [container, offset, value], {}, actual_span)


def _check_transpose_align(out: Expr, src: Expr, dt: DataType | None) -> None:
    """TTRANS (3510) statically asserts ``Cols * sizeof(T) % 32 == 0`` on both tiles.

    The swap always targets the last two dimensions; static-only (dynamic
    shapes are skipped — never over-rejects).
    """
    if dt is None or dt.get_bit() % 8 != 0:
        return
    esz = dt.get_bit() // 8
    for expr in (src, out):
        tile_type = getattr(expr, "type", None)
        if not isinstance(tile_type, _ir_core.TileType):
            continue
        shape = _tile_shape_ints(tile_type)
        if shape is None or len(shape) < 2:
            continue
        for axis in (len(shape) - 2, len(shape) - 1):
            if (shape[axis] * esz) % 32 != 0:
                raise InvalidType(
                    f"transpose: dimension {axis} of the tile ({shape[axis]} x {esz}B = "
                    f"{shape[axis] * esz}B) is not 32-byte aligned; TTRANS requires "
                    f"Cols * sizeof(T) % 32 == 0 on both dst and src"
                )


@op_impl("transpose")
def _ir_transpose(self, out: Expr, src: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    if dt is not None and dt.get_bit() not in (8, 16, 32):
        raise InvalidType(f"transpose: unsupported dtype {dt}, supported: b8/b16/b32")
    src_dt = getattr(src.type, "dtype", None)
    if src_dt is not None and dt is not None and src_dt.get_bit() != dt.get_bit():
        raise InvalidType(f"transpose: dtype size mismatch between dst ({dt}) and src ({src_dt})")
    _check_transpose_align(out, src, dt)
    return _ir_core.create_op_call(
        block_ir_op("transpose"),
        [out, src],
        {},
        span or _span(),
    )


@op_impl("cast")
def _ir_cast(self,
    out: Expr,
    src: Expr,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    target_type = out.type.dtype
    return _ir_core.create_op_call(
        block_ir_op("cast"),
        [out, src],
        {"target_type": target_type, "mode": mode},
        span or _span(),
    )


@op_impl("fillpad")
def _ir_fillpad(self,
    out: Expr,
    src: Expr,
    *,
    span: Span | None = None,
    mode: FillPadMode = FillPadMode.NORMAL,
) -> Expr:
    # TFillPad (3510) specializes on 1/2/4-byte elements (sub-byte types are
    # filled as byte patterns) and requires dst/src to share the element size.
    out_dt = getattr(out.type, "dtype", None)
    src_dt = getattr(src.type, "dtype", None)
    if out_dt is not None and out_dt.get_bit() not in (4, 8, 16, 32):
        raise InvalidType(f"fillpad: unsupported dtype {out_dt}, supported: b4/b8/b16/b32")
    if out_dt is not None and src_dt is not None and out_dt.get_bit() != src_dt.get_bit():
        raise InvalidType(f"fillpad: dtype size mismatch between dst ({out_dt}) and src ({src_dt})")
    name = {
        FillPadMode.NORMAL: "fillpad",
        FillPadMode.EXPAND: "fillpad_expand",
        FillPadMode.INPLACE: "fillpad_inplace",
    }[mode]
    return _ir_core.create_op_call(
        block_ir_op(name),
        [out, src],
        {},
        span or _span(),
    )


@op_impl("add_relu_cast")
def _ir_add_relu_cast(self,
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("add_relu_cast", out, lhs, rhs, target_type, span=span, mode=mode)


@op_impl("sub_relu_cast")
def _ir_sub_relu_cast(self,
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("sub_relu_cast", out, lhs, rhs, target_type, span=span, mode=mode)


@op_impl("mul_cast")
def _ir_mul_cast(self,
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("mul_cast", out, lhs, rhs, target_type, span=span, mode=mode)


def _ir_cmp(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, cmp_mode: int | Expr = 0) -> Expr:
    dt = getattr(lhs.type, "dtype", None)
    require_dtype_in(dt, _CMP_DTYPES, subject="cmp")
    require_dtypes_match([getattr(rhs.type, "dtype", None)], dt, subject="cmp")
    if _is_int(cmp_mode):
        cmp_mode = _normalize_expr(cmp_mode, span)
    cmp_mode = require_const_int(cmp_mode, subject="cmp_mode")
    return _ir_core.create_op_call(
        block_ir_op("cmp"),
        [out, lhs, rhs],
        {"cmp_mode": cmp_mode},
        span or _span(),
    )


def _ir_cmps(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, cmp_mode: int | Expr = 0) -> Expr:
    dt = getattr(lhs.type, "dtype", None)
    require_dtype_in(dt, _CMP_DTYPES, subject="cmps")
    from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

    check_const_expr_fits_dtype(rhs, dt, api="pl.cmps")
    if _is_int(cmp_mode):
        cmp_mode = _normalize_expr(cmp_mode, span)
    cmp_mode = require_const_int(cmp_mode, subject="cmp_mode")
    return _ir_core.create_op_call(
        block_ir_op("cmps"),
        [out, lhs, rhs],
        {"cmp_mode": cmp_mode},
        span or _span(),
    )


@op_impl("set_mask_count")
def _ir_set_mask_count(self, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_mask_count", [], {}, span or _span())


@op_impl("set_mask_norm")
def _ir_set_mask_norm(self, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_mask_norm", [], {}, span or _span())


@op_impl("set_vec_mask")
def _ir_set_vec_mask(self, mask_high: Expr, mask_low: Expr, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_vec_mask", [mask_high, mask_low], {}, span or _span())


@op_impl("reset_mask")
def _ir_reset_mask(self, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.reset_mask", [], {}, span or _span())


@op_impl("quant")
def _ir_quant(self,
    out: Expr,
    src: Expr,
    scale: Expr,
    *,
    span: Span | None = None,
    mode: QuantMode = QuantMode.SYM,
    offset: Expr | None = None,
) -> Expr:
    # 3510 TQUANT: INT8_SYM converts FP32→INT8, INT8_ASYM converts FP32→UINT8;
    # both consume FP32 scale/offset tiles.
    require_dtype_in(getattr(src.type, "dtype", None), (DataType.FP32,), subject="quant")
    require_dtype_in(getattr(scale.type, "dtype", None), (DataType.FP32,), subject="quant")
    if mode == QuantMode.SYM:
        require_dtype_in(getattr(out.type, "dtype", None), (DataType.INT8,), subject="quant")
    elif mode == QuantMode.ASYM:
        require_dtype_in(getattr(out.type, "dtype", None), (DataType.UINT8,), subject="quant")
        if offset is not None:
            require_dtype_in(getattr(offset.type, "dtype", None), (DataType.FP32,), subject="quant")
    ins = [out, src, scale]
    if mode == QuantMode.ASYM:
        if offset is None:
            raise InvalidVal("quant in 'asym' mode requires an offset argument")
        ins.append(offset)
    return _ir_core.create_op_call(block_ir_op("quant"), ins, {"mode": mode}, span or _span())


@op_impl("dequant")
def _ir_dequant(self, out: Expr, src: Expr, scale: Expr, offset: Expr, *, span: Span | None = None) -> Expr:
    # 3510 TDEQUANT: FP32 output from INT8/INT16 sources with FP32 scale/offset tiles.
    require_dtype_in(getattr(out.type, "dtype", None), (DataType.FP32,), subject="dequant")
    require_dtype_in(getattr(src.type, "dtype", None), (DataType.INT8, DataType.INT16), subject="dequant")
    require_dtype_in(getattr(scale.type, "dtype", None), (DataType.FP32,), subject="dequant")
    require_dtype_in(getattr(offset.type, "dtype", None), (DataType.FP32,), subject="dequant")
    return _ir_core.create_op_call(block_ir_op("dequant"), [out, src, scale, offset], {}, span or _span())


@op_impl("ssbuf_store")
def _ir_ssbuf_store(self, struct_var: Expr, offset: Expr, /, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("ssbuf_store"), [struct_var, offset], {}, span or _span())


@op_impl("ssbuf_load")
def _ir_ssbuf_load(self, struct_var: Expr, offset: Expr, /, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("ssbuf_load"), [struct_var, offset], {}, span or _span())


_LAYOUT_TO_BS: dict[Layout, tuple[int, int]] = {
    Layout.ND: (1, 0),
    Layout.DN: (2, 0),
    Layout.NZ: (2, 1),
    Layout.ZN: (1, 2),
    Layout.NN: (2, 2),
    Layout.ZZ: (1, 1),
}

_BS_TO_LAYOUT: dict[tuple[int, int], Layout] = {v: k for k, v in _LAYOUT_TO_BS.items()}


_FRACTAL_TRANSPOSE: dict[Layout, Layout] = {
    Layout.NZ: Layout.ZN,
    Layout.ZN: Layout.NZ,
    Layout.NN: Layout.ZZ,
    Layout.ZZ: Layout.NN,
    Layout.ND: Layout.DN,
    Layout.DN: Layout.ND,
}


def _is_fractal_transpose(a: Layout, b: Layout) -> bool:
    """True when layouts a, b are transpose-pairs (Z<->N flipped)."""
    return _FRACTAL_TRANSPOSE.get(a) == b


# 3510 cube buffers (L0A/L0B/L0C) do not support these layouts.
_REJECTED_LAYOUTS_ON_3510 = (Layout.ZZ, Layout.NN)


def _tile_layout(tile_type: "_IRTileType") -> "Layout | None":
    """Return the Layout of a TileType.

    A conv tile carries its layout in ``hardware_info.conv_layout`` (blayout/slayout do
    not describe it); every other tile derives it from blayout/slayout.
    """
    hw = getattr(tile_type, "hardware_info", None)
    if hw is None:
        return None
    if int(hw.conv_layout) >= 0:
        return Layout(int(hw.conv_layout))
    return _BS_TO_LAYOUT.get((int(hw.blayout), int(hw.slayout)))


_DEFAULT_LAYOUTS_3510: dict[MemorySpace, Layout] = {
    MemorySpace.Mat: Layout.NZ,
    MemorySpace.Left: Layout.NZ,
    MemorySpace.Right: Layout.ZN,
    MemorySpace.Bias: Layout.ND,
    MemorySpace.Scaling: Layout.ND,
    MemorySpace.Acc: Layout.NZ,
    MemorySpace.ScaleLeft: Layout.ZZ,
    MemorySpace.ScaleRight: Layout.NN,
}

_DEFAULT_FRACTAL = 512  # fractalABSize
_ACC_FRACTAL = 1024  # fractalCSize
_MX_SCALE_FRACTAL = 32  # fractalMxSize

mem_id: int = 0

_PAD_VALUES = {
    TilePad.null: 0,
    TilePad.zero: 1,
    TilePad.max: 2,
    TilePad.min: 3,
}


def _normalize_tile_pad(pad: "int | TilePad | None") -> "int | None":
    if pad is None:
        return None
    if pad in _PAD_VALUES:
        return _PAD_VALUES[pad]
    if isinstance(pad, int):
        if pad not in _PAD_VALUES.values():
            raise InvalidArgument("TileType.pad must be one of TilePad.null/zero/max/min")
        return pad
    raise InvalidType("TileType.pad must be a enum TilePad or compile-time integer 0/1/2/3")


def _static_last_axis(shape: "Sequence[int] | _ir_core.MakeTuple") -> "int | None":
    """The tile's innermost dimension when it is a compile-time integer, else None."""
    dims = shape.elements if isinstance(shape, _ir_core.MakeTuple) else shape
    if not dims:
        return None
    last = dims[-1]
    return last if _is_int(last) else _const_int_value(last)


def _validate_subbyte_tile_shape(tt: "TileType") -> None:
    """A sub-byte tile must span whole storage units along the axis its elements pack into.

    Several sub-byte elements share a byte, and only the innermost axis can carry that packing,
    so a tile that ends mid-byte has no address for its own edge. Tensors are exempt: their
    innermost extent may be dynamic, and it is the tile that decides what a transfer touches.
    """
    pack = 8 // tt.dtype.get_bit()
    if pack <= 1:
        return
    last = _static_last_axis(tt.shape)
    if last is not None and last % pack != 0:
        raise InvalidShape(
            f"a {tt.dtype.to_string()} tile's last dimension must be a multiple of {pack} "
            f"({pack} elements share a storage unit), got {last}. Pad the tile to the next "
            f"multiple of {pack}."
        )


_C0_SIZE_BYTE = 32  # pto C0_SIZE_BYTE: one fractal byte-block along the innermost axis

# Conv data layouts. A TileType whose `layout` is one of these selects the pto::ConvTile
# path (multi-dim, L1-only) instead of the 2-D pto::Tile path; see pto/common/type.hpp
# Layout and pto/common/pto_tile.hpp ConvTile/ConvTileShape.
_CONV_LAYOUTS = frozenset(
    {Layout.NC1HWC0, Layout.NDC1HWC0, Layout.FRACTAL_Z, Layout.FRACTAL_Z_3D}
)

# dtypes a pto::ConvTile may carry (a5 CheckConvTileData). Keyed by DataType.to_string()
# because the pybind DataType object is not hashable.
_CONV_TILE_DTYPE_NAMES = frozenset(
    {
        "int8", "uint8", "int16", "uint16", "int32", "uint32",
        "fp8e4m3fn", "fp8e5m2", "fp16", "bfloat16", "fp32",
    }
)


def _is_conv_layout(layout: "Layout | None") -> bool:
    """Whether ``layout`` selects the multi-dim pto::ConvTile path."""
    return layout in _CONV_LAYOUTS


def _conv_c0_elems(dtype: DataType) -> int:
    """C0: elements of one 32-byte block along the innermost conv-tile axis."""
    return _C0_SIZE_BYTE // ((dtype.get_bit() + 7) // 8)


def _validate_conv_tile_type(tt: "TileType") -> None:
    """Validate a conv-layout TileType (pto::ConvTile: L1-only, shape fixed by the layout).

    The per-layout shape contract mirrors pto (a5 TLoad/TExtract):
      NC1HWC0      [N, C1, H, W, C0]            (5D)
      NDC1HWC0     [N, D, C1, H, W, C0]         (6D)
      FRACTAL_Z    [C1HW, N/16, 16, C0] (4D) or [C1, H, W, N, C0] (5D, N a multiple of 16)
      FRACTAL_Z_3D [C1DHW, N/16, 16, C0]        (4D)
    where C0 = 32B / sizeof(dtype). The 4D forms spell out the packed ``N/16`` quotient, so only
    the 5D form carries an N that has to be checked for 16-alignment: a non-multiple N would make
    ``BufferSize = Πdims`` cover less than the packed box the layout actually occupies.
    """
    from pypto_pro.runtime.jit import get_current_arch

    arch = get_current_arch()
    if arch != NpuArch.DAV_3510:
        raise NotSupported(
            f"conv-layout tiles are only supported on 3510, got architecture '{arch}'."
        )
    if tt.target_memory != MemorySpace.Mat:
        raise InvalidVal(
            f"a conv-layout tile ({tt.layout.name}) is a pto::ConvTile and must live in L1; "
            f"set target_memory=MemorySpace.Mat, got {tt.target_memory.name}."
        )
    if tt.dtype.to_string() not in _CONV_TILE_DTYPE_NAMES:
        allowed = ", ".join(sorted(_CONV_TILE_DTYPE_NAMES))
        raise InvalidType(f"conv-layout tiles only support dtypes {{{allowed}}}, got {tt.dtype}.")
    if tt.fractal is not None:
        raise InvalidVal(
            f"conv-layout tile ({tt.layout.name}) is a pto::ConvTile; drop the 'fractal' argument "
            f"(got fractal={tt.fractal})."
        )
    if tt.valid_shape is not None:
        raise InvalidVal(
            f"conv-layout tile ({tt.layout.name}) is a pto::ConvTile; drop the 'valid_shape' argument "
            f"(valid_shape is not supported by the current ConvTile declaration-and-allocation scope; "
            f"pto::ConvTile's ValidShape serves the format-conversion data path, which is not wired yet; "
            f"got valid_shape={tt.valid_shape!r})."
        )
    if tt.pad is not None:
        raise InvalidVal(
            f"conv-layout tile ({tt.layout.name}) is a pto::ConvTile; drop the 'pad' argument "
            f"(a ConvTile has no padding mode, got pad={tt.pad!r})."
        )
    if tt.compact is not None and tt.compact != 1:
        raise InvalidVal(
            f"conv-layout tile ({tt.layout.name}) is a pto::ConvTile; drop the 'compact' argument "
            f"(a ConvTile is always normal compact mode, got compact={tt.compact!r})."
        )

    shape = _const_shape_ints(tt.shape)
    if shape is None:
        return  # symbolic dimensions: the layout shape contract cannot be checked yet

    c0 = _conv_c0_elems(tt.dtype)
    layout = tt.layout

    def require_rank(rank: int) -> None:
        if len(shape) != rank:
            raise InvalidShape(
                f"TileType with layout {layout.name} requires shape rank {rank}, got rank "
                f"{len(shape)} (shape {tt.shape!r})."
            )

    def require_dim(val: int, want: int, name: str) -> None:
        if val != want:
            raise InvalidShape(
                f"TileType with layout {layout.name} requires its {name} to be "
                f"{want} for dtype {tt.dtype.to_string()}, got {val} (shape {tt.shape!r})."
            )

    def require_multiple_of(val: int, mod: int, name: str) -> None:
        if val % mod != 0:
            raise InvalidShape(
                f"TileType with layout {layout.name} requires its {name} dimension to be a multiple of "
                f"{mod} (the packed fractal box width), got {val} (shape {tt.shape!r})."
            )

    if layout == Layout.NC1HWC0:
        require_rank(5)
        require_dim(shape[-1], c0, "last dimension (C0)")
    elif layout == Layout.NDC1HWC0:
        require_rank(6)
        require_dim(shape[-1], c0, "last dimension (C0)")
    elif layout == Layout.FRACTAL_Z:
        if len(shape) == 5:  # [C1, H, W, N, C0]
            require_multiple_of(shape[-2], _FIXED_ROW_SIZE, "N")
            require_dim(shape[-1], c0, "last dimension (C0)")
        elif len(shape) == 4:  # [C1HW, N/16, 16, C0]
            require_dim(
                shape[-2], _FIXED_ROW_SIZE, "second-to-last dimension (the fixed 16 of [N/16, 16, C0])"
            )
            require_dim(shape[-1], c0, "last dimension (C0)")
        else:
            raise InvalidShape(
                f"TileType with layout FRACTAL_Z requires shape rank 4 ([C1HW, N/16, 16, C0]) or "
                f"5 ([C1, H, W, N, C0]), got rank {len(shape)} (shape {tt.shape!r})."
            )
    elif layout == Layout.FRACTAL_Z_3D:
        require_rank(4)  # [C1DHW, N/16, 16, C0]
        require_dim(shape[-2], _FIXED_ROW_SIZE, "second-to-last dimension (the fixed 16 of [N/16, 16, C0])")
        require_dim(shape[-1], c0, "last dimension (C0)")


def _has_unit_last_axis(shape: "Sequence[int] | _ir_core.MakeTuple") -> bool:
    """Whether the tile's innermost dimension is statically 1."""
    return _static_last_axis(shape) == 1


def _apply_default_layout(tt: "TileType") -> None:
    from pypto_pro.runtime.jit import get_current_arch

    arch = get_current_arch()
    if arch not in (NpuArch.DAV_3510, NpuArch.DAV_9201):
        raise NotSupported(f"PyPTO Pro only supports arch '3510' or '9201', got '{arch}'")
    layout_dict = _DEFAULT_LAYOUTS_3510
    default_layout = layout_dict.get(tt.target_memory)

    if default_layout is None:
        return

    if tt.layout is None:
        tt.layout = default_layout

    # L1 ZZ/NN scale views and the dedicated scale buffers use the PTO MX fractal size.
    is_mx_scale_tile = tt.target_memory in (MemorySpace.ScaleLeft, MemorySpace.ScaleRight) or (
        tt.target_memory == MemorySpace.Mat and tt.layout in (Layout.ZZ, Layout.NN)
    )
    if is_mx_scale_tile:
        if tt.fractal not in (None, _MX_SCALE_FRACTAL):
            raise InvalidFormat(
                f"{tt.target_memory.name} MX scale tiles require fractal={_MX_SCALE_FRACTAL}, got {tt.fractal}"
            )
        tt.fractal = _MX_SCALE_FRACTAL

    allowed_layouts = {default_layout}
    if tt.target_memory == MemorySpace.Mat:
        allowed_layouts.update({Layout.ZN, Layout.ND, Layout.DN})
        if tt.dtype == DataType.FP8E8M0:
            allowed_layouts.update({Layout.ZZ, Layout.NN})

    if (
        tt.target_memory in (MemorySpace.Left, MemorySpace.Right, MemorySpace.Acc)
        and tt.layout in _REJECTED_LAYOUTS_ON_3510
    ):
        raise InvalidVal(
            f"{tt.target_memory.name} tiles do not support {tt.layout.name} layout on '{arch}'; "
            f"use the default {default_layout.name} layout instead."
        )

    if tt.layout not in allowed_layouts:
        space_name = tt.target_memory.name
        allowed_text = ", ".join(f.name for f in sorted(allowed_layouts, key=lambda x: x.name))
        raise InvalidVal(
            f"{space_name} tiles require layout in {{{allowed_text}}}, "
            f"got {tt.layout.name}. "
            f"Default for '{arch}' is {default_layout.name}."
        )

    if tt.target_memory == MemorySpace.Acc and tt.fractal is None:
        if tt.dtype in (DataType.FP32, DataType.INT32):
            tt.fractal = _ACC_FRACTAL


# 3510 MOVE / INSERT / MATMUL dtype and layout validation.
_ANY_LAYOUT = object()

# Layout/dtype combinations shared by TMOV and TEXTRACT on 3510.
_3510_MOVE_EXTRACT_COMMON_COMBOS = (
    # Mat -> Left.
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Left", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Left", Layout.ND, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Left", Layout.ZN, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    # Mat -> Right.
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.INT8, DataType.INT8),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.HF8, DataType.HF8),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.FP16, DataType.FP16),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.BF16, DataType.BF16),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.FP32, DataType.FP32),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Right", Layout.NZ, Layout.ZN, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.INT8, DataType.INT8),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.HF8, DataType.HF8),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.FP16, DataType.FP16),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.BF16, DataType.BF16),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.FP32, DataType.FP32),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Right", Layout.ZN, Layout.ZN, DataType.FP4E2M1, DataType.FP4E2M1),
    # Mat -> ScaleLeft.
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.INT8, DataType.INT8),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.HF8, DataType.HF8),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.FP16, DataType.FP16),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.BF16, DataType.BF16),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.FP32, DataType.FP32),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Mat", "ScaleLeft", Layout.ZZ, Layout.ZZ, DataType.FP8E8M0, DataType.FP8E8M0),

    # Mat -> ScaleRight.
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.INT8, DataType.INT8),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.HF8, DataType.HF8),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.FP16, DataType.FP16),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.BF16, DataType.BF16),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.FP32, DataType.FP32),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Mat", "ScaleRight", Layout.NN, Layout.NN, DataType.FP8E8M0, DataType.FP8E8M0),

    # Vec -> Vec.
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.HF8),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.INT8),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP16, DataType.FP16),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.BF16, DataType.BF16),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP32, DataType.FP32),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.INT32, DataType.INT32),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),

    # Vec -> Mat is a physical copy; the destination layout does not select a conversion path.
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.INT8, DataType.INT8),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.HF8, DataType.HF8),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP16, DataType.FP16),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.BF16, DataType.BF16),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP32, DataType.FP32),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.INT8, DataType.INT8),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.HF8, DataType.HF8),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP16, DataType.FP16),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.BF16, DataType.BF16),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP32, DataType.FP32),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP8E8M0, DataType.FP8E8M0),

    # Acc -> Vec.
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.INT32, DataType.INT32),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),

    # Acc -> Mat.
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),
)

_3510_MOVE_COMBOS = (
    *_3510_MOVE_EXTRACT_COMMON_COMBOS,

    # Additional layout/dtype combinations supported only by TMOV on 3510.
    # Mat -> Left/Right layouts rejected by TEXTRACT.
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Left", Layout.ZZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.INT8, DataType.INT8),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.HF8, DataType.HF8),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.FP16, DataType.FP16),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.BF16, DataType.BF16),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.FP32, DataType.FP32),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Right", Layout.ND, Layout.ZN, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.INT8, DataType.INT8),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.HF8, DataType.HF8),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.FP16, DataType.FP16),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.BF16, DataType.BF16),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.FP32, DataType.FP32),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Mat", "Right", Layout.ZZ, Layout.ZN, DataType.FP4E2M1, DataType.FP4E2M1),

    # Acc -> Vec DN layout rejected by TEXTRACT.
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.INT32, DataType.INT32),

    # Mat -> Bias.
    ("Mat", "Bias", _ANY_LAYOUT, Layout.ND, DataType.INT32, DataType.INT32),
    ("Mat", "Bias", _ANY_LAYOUT, Layout.ND, DataType.FP32, DataType.FP32),
    ("Mat", "Bias", _ANY_LAYOUT, Layout.ND, DataType.FP16, DataType.FP32),
    ("Mat", "Bias", _ANY_LAYOUT, Layout.ND, DataType.BF16, DataType.FP32),

    # Mat -> Scaling.
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP8E4M3FN, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP8E4M3FN, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP8E5M2, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP8E5M2, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP8E8M0, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP8E8M0, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP4E2M1, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP4E2M1, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP4E1M2, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP4E1M2, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.HF8, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.HF8, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT8, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT8, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP16, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP16, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.BF16, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.BF16, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT16, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT16, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP32, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.FP32, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT32, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT32, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT64, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.INT64, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT8, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT8, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT16, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT16, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT32, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT32, DataType.UINT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT64, DataType.INT64),
    ("Mat", "Scaling", _ANY_LAYOUT, _ANY_LAYOUT, DataType.UINT64, DataType.UINT64),

    # Vec -> Vec.
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.INT16, DataType.INT16),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.UINT8, DataType.UINT8),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.UINT16, DataType.UINT16),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.UINT32, DataType.UINT32),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.INT16, DataType.INT16),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.UINT8),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.UINT16),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.UINT32),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.UINT8, DataType.UINT8),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.INT32, DataType.INT32),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Vec", Layout.ND, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),

    # Acc -> Mat.
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP32),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.INT32, DataType.INT32),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP32),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.INT32, DataType.INT32),
)


_3510_MOVE_QUANT_COMBOS = (
    # Acc -> Vec, scalar/vector quantized.
    # Supported quantized dtype conversions:
    # FP32 -> INT8 / UINT8 / HF8 / FP16 / BF16 / FP8E4M3FN / FP32
    # INT32 -> INT8 / UINT8 / FP16 / BF16

    # ══ Acc -> Vec · NZ -> ND ══
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.INT8),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.UINT8),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.HF8),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.INT32, DataType.INT8),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.INT32, DataType.UINT8),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.INT32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.INT32, DataType.BF16),

    # ══ Acc -> Vec · NZ -> DN ══
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.INT8),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.UINT8),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.HF8),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.INT32, DataType.INT8),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.INT32, DataType.UINT8),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.INT32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.INT32, DataType.BF16),

    # ══ Acc -> Vec · NZ -> NZ ══
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.INT8),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.UINT8),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.HF8),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT8),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.UINT8),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.BF16),

    # Acc -> Mat, scalar/vector quantized.
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.HF8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.HF8),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.INT32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.INT32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.INT32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.ND, DataType.INT32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.HF8),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.INT32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.INT32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.INT32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.DN, DataType.INT32, DataType.BF16),
)


_3510_INSERT_COMBOS = (
    # Vec -> Vec; ND -> ND, NZ -> NZ
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.INT8),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.INT32, DataType.INT32),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP16, DataType.FP16),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.BF16, DataType.BF16),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP32, DataType.FP32),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.HF8),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Vec", Layout.ND, Layout.ND, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),

    # Vec -> Mat is a physical copy; the destination layout does not select a conversion path.
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.INT8, DataType.INT8),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.INT32, DataType.INT32),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP16, DataType.FP16),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.BF16, DataType.BF16),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP32, DataType.FP32),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.HF8, DataType.HF8),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Mat", Layout.ND, _ANY_LAYOUT, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.INT8, DataType.INT8),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.INT32, DataType.INT32),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP16, DataType.FP16),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.BF16, DataType.BF16),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP32, DataType.FP32),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.HF8, DataType.HF8),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "Mat", Layout.NZ, _ANY_LAYOUT, DataType.FP4E2M1, DataType.FP4E2M1),

    # Acc -> Vec.
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.ND, DataType.INT32, DataType.INT32),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.DN, DataType.INT32, DataType.INT32),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Acc", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),

    # Acc -> Mat.
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),
)

# A move carrying an offset is emitted as TEXTRACT, not TMOV. TEXTRACT is a
# strict subset of TMOV on 3510, so reuse only the explicitly shared combinations.
_3510_EXTRACT_COMBOS = _3510_MOVE_EXTRACT_COMMON_COMBOS


_3510_MATMUL_DTYPE_COMBOS = (
    # INT8
    (DataType.INT8, DataType.INT8, DataType.INT32),

    # FP16 / BF16 / FP32
    (DataType.FP16, DataType.FP16, DataType.FP32),
    (DataType.BF16, DataType.BF16, DataType.FP32),
    (DataType.FP32, DataType.FP32, DataType.FP32),

    # FP8
    (DataType.FP8E4M3FN, DataType.FP8E4M3FN, DataType.FP32),
    (DataType.FP8E4M3FN, DataType.FP8E5M2, DataType.FP32),
    (DataType.FP8E5M2, DataType.FP8E4M3FN, DataType.FP32),
    (DataType.FP8E5M2, DataType.FP8E5M2, DataType.FP32),

    # HF8
    (DataType.HF8, DataType.HF8, DataType.FP32),
)


_MATMUL_DTYPE_COMBOS = {
    NpuArch.DAV_3510: _3510_MATMUL_DTYPE_COMBOS,
    NpuArch.DAV_9201: _3510_MATMUL_DTYPE_COMBOS,
}


def _layout_matches(
    actual: Layout | None,
    expected: Any,
) -> bool:
    return actual is None or expected is _ANY_LAYOUT or actual == expected


def _layout_dtype_combo_matches(
    actual: tuple[Any, ...],
    rule: tuple[Any, ...],
) -> bool:
    src_loc, dst_loc, src_layout, dst_layout, src_dtype, dst_dtype = actual
    (
        rule_src_loc,
        rule_dst_loc,
        rule_src_layout,
        rule_dst_layout,
        rule_src_dtype,
        rule_dst_dtype,
    ) = rule

    return (
        src_loc == rule_src_loc
        and dst_loc == rule_dst_loc
        and _layout_matches(src_layout, rule_src_layout)
        and _layout_matches(dst_layout, rule_dst_layout)
        and src_dtype == rule_src_dtype
        and dst_dtype == rule_dst_dtype
    )


def _check_matmul_dtype(
    op_name: str,
    dst: Expr,
    lhs: Expr,
    rhs: Expr,
    *,
    bias: Expr | None = None,
    acc: Expr | None = None,
) -> None:
    """Validate MATMUL dtype combinations for configured architectures."""

    from pypto_pro.runtime.jit import get_current_arch

    arch = get_current_arch()
    combos = _MATMUL_DTYPE_COMBOS.get(arch)

    # 当前平台没有配置 MATMUL frontend dtype rules 时不做硬拦截。
    if combos is None:
        return

    lhs_dtype = getattr(lhs.type, "dtype", None)
    rhs_dtype = getattr(rhs.type, "dtype", None)
    dst_dtype = getattr(dst.type, "dtype", None)

    # 静态信息不足时避免误伤。
    if lhs_dtype is None or rhs_dtype is None or dst_dtype is None:
        return

    combo = (
        lhs_dtype,
        rhs_dtype,
        dst_dtype,
    )

    if combo not in combos:
        raise InvalidType(
            f"{op_name}: unsupported dtype combination "
            f"A={_frontend_dtype_name(lhs_dtype)}, "
            f"B={_frontend_dtype_name(rhs_dtype)}, "
            f"C={_frontend_dtype_name(dst_dtype)}; "
            "please check whether MATMUL supports the current A, B, and C dtypes."
        )

    if acc is not None:
        acc_dtype = getattr(acc.type, "dtype", None)
        if acc_dtype is not None and acc_dtype != dst_dtype:
            raise InvalidType(
                f"{op_name}: Acc dtype must match C dtype, "
                f"got Acc={_frontend_dtype_name(acc_dtype)}, "
                f"C={_frontend_dtype_name(dst_dtype)}."
            )

    if bias is not None:
        bias_dtype = getattr(bias.type, "dtype", None)
        if bias_dtype is not None and bias_dtype != dst_dtype:
            raise InvalidType(
                f"{op_name}: Bias dtype must match C dtype, "
                f"got Bias={_frontend_dtype_name(bias_dtype)}, "
                f"C={_frontend_dtype_name(dst_dtype)}."
            )


# CompactMode enum (pto/type.hpp): Null=0, Normal=1, RowPlusOne=2,
# RowAlignedPadding=3. Any other integer is not a valid compact mode.
_COMPACT_VALUES = frozenset({0, 1, 2, 3})


# ---------------------------------------------------------------------------
# Layout × dtype 合法组合表（按平台 arch 分；9201 当前复用 3510 规则）。
#
# 每个合法组合是一个六元组 (src_loc, dst_loc, src_layout, dst_layout, src_dtype, dst_dtype)，
# 均为 pto-isa 实际支持（static_assert 允许范围）且经真实上板实测。
# 匹配方式：把待校验的操作数解析为同构六元组，六个元素全部相等即命中——
# 不区分同类型/跨类型/量化子表，也不做 src_dtype == dst_dtype 预判。
#
# 表按 op 分组，每组按 src_loc→dst_loc 与排布分类罗列；规则完全相同的
# layout/dtype 组合可由公共 dtype 对生成，最终仍按完整六元组精确匹配。
# ---------------------------------------------------------------------------


def _loc_name(type_: "_IRTileType | _ir_core.TensorType") -> str | None:
    """Memory-space key of a Tile or Tensor: 'GM' for tensors, MemorySpace name for tiles."""
    if isinstance(type_, _ir_core.TensorType):
        return "GM"
    mem = getattr(getattr(type_, "memref", None), "memory_space", None)  # type: ignore[attr-defined]
    return mem.name if mem is not None else None


def _operand_layout(type_: "_IRTileType | _ir_core.TensorType") -> "Layout | None":
    """Layout of a Tile (hardware_info) or Tensor (tensor_view)."""
    if isinstance(type_, _ir_core.TensorType):
        tv = getattr(type_, "tensor_view", None)
        return getattr(tv, "layout", None)
    return _tile_layout(type_)  # type: ignore[arg-type]


def _operand_dtype(type_: "_IRTileType | _ir_core.TensorType") -> "DataType | None":
    return getattr(type_, "dtype", None)


# 3510 load 合法组合表（load / load_tile 共用）
_3510_LOAD_COMBOS = (
    # ND → ND（同位宽组合）
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP4E2M1, DataType.FP4E2M1),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP4E2M1, DataType.FP4E1M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP4E1M2, DataType.FP4E2M1),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP4E1M2, DataType.FP4E1M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.INT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.UINT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT8, DataType.HF8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT8, DataType.INT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT8, DataType.UINT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT8, DataType.HF8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.INT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.UINT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E8M0),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E5M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.HF8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.INT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.UINT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.HF8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.INT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.UINT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E8M0),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E5M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.HF8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.INT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.UINT8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.HF8, DataType.HF8),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP16, DataType.FP16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP16, DataType.BF16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP16, DataType.INT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP16, DataType.UINT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.BF16, DataType.FP16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.BF16, DataType.BF16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.BF16, DataType.INT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.BF16, DataType.UINT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT16, DataType.FP16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT16, DataType.BF16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT16, DataType.INT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT16, DataType.UINT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT16, DataType.FP16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT16, DataType.BF16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT16, DataType.INT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT16, DataType.UINT16),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP32, DataType.FP32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP32, DataType.INT32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.FP32, DataType.UINT32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT32, DataType.FP32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT32, DataType.INT32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT32, DataType.UINT32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT32, DataType.FP32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT32, DataType.INT32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT32, DataType.UINT32),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT64, DataType.INT64),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.INT64, DataType.UINT64),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT64, DataType.INT64),
    ("GM", "Vec", Layout.ND, Layout.ND, DataType.UINT64, DataType.UINT64),
    # DN → DN（同位宽组合，单列形态 [N,1]）
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP4E2M1, DataType.FP4E2M1),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP4E2M1, DataType.FP4E1M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP4E1M2, DataType.FP4E2M1),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT8, DataType.INT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT8, DataType.UINT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT8, DataType.HF8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT8, DataType.INT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT8, DataType.UINT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT8, DataType.HF8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.INT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.UINT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.FP8E8M0),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.FP8E5M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.HF8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.INT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.UINT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.HF8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.INT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.UINT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.FP8E8M0),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.HF8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.HF8, DataType.INT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.HF8, DataType.UINT8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.HF8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.HF8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.HF8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.HF8, DataType.HF8),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP16, DataType.FP16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP16, DataType.BF16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP16, DataType.INT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP16, DataType.UINT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.BF16, DataType.FP16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.BF16, DataType.BF16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.BF16, DataType.INT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.BF16, DataType.UINT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT16, DataType.FP16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT16, DataType.BF16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT16, DataType.INT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT16, DataType.UINT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT16, DataType.FP16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT16, DataType.BF16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT16, DataType.INT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT16, DataType.UINT16),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP32, DataType.FP32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP32, DataType.INT32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.FP32, DataType.UINT32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT32, DataType.FP32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT32, DataType.INT32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT32, DataType.UINT32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT32, DataType.FP32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT32, DataType.INT32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT32, DataType.UINT32),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT64, DataType.INT64),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.INT64, DataType.UINT64),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT64, DataType.INT64),
    ("GM", "Vec", Layout.DN, Layout.DN, DataType.UINT64, DataType.UINT64),
    # NZ → NZ（同位宽组合；NZ2NZ 无 64bit 限制）
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E1M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E2M1),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.INT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.UINT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT8, DataType.HF8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.INT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.UINT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.HF8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.INT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.UINT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E8M0),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E5M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.HF8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.INT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.UINT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.HF8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.INT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.UINT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E8M0),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.HF8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.INT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.UINT8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E8M0),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E4M3FN),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E5M2),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.HF8, DataType.HF8),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP16, DataType.FP16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP16, DataType.BF16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP16, DataType.INT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP16, DataType.UINT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.BF16, DataType.FP16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.BF16, DataType.BF16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.BF16, DataType.INT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.BF16, DataType.UINT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT16, DataType.FP16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT16, DataType.BF16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT16, DataType.INT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT16, DataType.UINT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.FP16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.BF16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.INT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.UINT16),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.INT32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.FP32, DataType.UINT32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT32, DataType.UINT32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.FP32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.INT32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.UINT32),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT64, DataType.INT64),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.INT64, DataType.UINT64),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT64, DataType.INT64),
    ("GM", "Vec", Layout.NZ, Layout.NZ, DataType.UINT64, DataType.UINT64),
    # ND → NZ（禁 64bit；同位宽组合）
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP4E2M1, DataType.FP4E1M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP4E1M2, DataType.FP4E2M1),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT8, DataType.INT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT8, DataType.UINT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT8, DataType.HF8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT8, DataType.INT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT8, DataType.UINT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT8, DataType.HF8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E8M0, DataType.INT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E8M0, DataType.UINT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E8M0, DataType.FP8E8M0),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E8M0, DataType.FP8E5M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E8M0, DataType.HF8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.INT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.UINT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E4M3FN, DataType.HF8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.INT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.UINT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.FP8E8M0),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP8E5M2, DataType.HF8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.HF8, DataType.INT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.HF8, DataType.UINT8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.HF8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.HF8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.HF8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.HF8, DataType.HF8),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP16, DataType.FP16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP16, DataType.BF16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP16, DataType.INT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP16, DataType.UINT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.BF16, DataType.FP16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.BF16, DataType.BF16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.BF16, DataType.INT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.BF16, DataType.UINT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT16, DataType.FP16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT16, DataType.BF16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT16, DataType.INT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT16, DataType.UINT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT16, DataType.FP16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT16, DataType.BF16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT16, DataType.INT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT16, DataType.UINT16),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP32, DataType.FP32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP32, DataType.INT32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.FP32, DataType.UINT32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT32, DataType.FP32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT32, DataType.INT32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.INT32, DataType.UINT32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT32, DataType.FP32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT32, DataType.INT32),
    ("GM", "Mat", Layout.ND, Layout.NZ, DataType.UINT32, DataType.UINT32),
    # DN → NZ（禁 64bit、禁 FP4；同位宽组合）
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT8, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT8, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT8, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT8, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT8, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT8, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E8M0, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E8M0, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E8M0, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E8M0, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E8M0, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E4M3FN, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E4M3FN, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E4M3FN, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E5M2, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E5M2, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E5M2, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP8E5M2, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.HF8, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.HF8, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.HF8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.HF8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.HF8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.HF8, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.BF16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.BF16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.BF16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.BF16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP32, DataType.FP32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP32, DataType.INT32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.FP32, DataType.UINT32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT32, DataType.FP32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT32, DataType.INT32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.INT32, DataType.UINT32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT32, DataType.FP32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT32, DataType.INT32),
    ("GM", "Mat", Layout.DN, Layout.NZ, DataType.UINT32, DataType.UINT32),
    # ND → ND（仅 INT64 / UINT64）
    ("GM", "Mat", Layout.ND, Layout.ND, DataType.INT64, DataType.INT64),
    ("GM", "Mat", Layout.ND, Layout.ND, DataType.UINT64, DataType.UINT64),
    # ND → ZZ（仅 FP8E8M0；MX_A ND→ZZ 通路）
    ("GM", "Mat", Layout.ND, Layout.ZZ, DataType.FP8E8M0, DataType.FP8E8M0),
    # DN → ZZ（仅 FP8E8M0；MX_A DN→ZZ 通路）
    ("GM", "Mat", Layout.DN, Layout.ZZ, DataType.FP8E8M0, DataType.FP8E8M0),
    # ND → NN（仅 FP8E8M0；MX_B ND→NN 通路）
    ("GM", "Mat", Layout.ND, Layout.NN, DataType.FP8E8M0, DataType.FP8E8M0),
    # DN → NN（仅 FP8E8M0；MX_B DN→NN 通路）
    ("GM", "Mat", Layout.DN, Layout.NN, DataType.FP8E8M0, DataType.FP8E8M0),
    # DN → ZN（同位宽组合）
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP4E2M1, DataType.FP4E2M1),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP4E2M1, DataType.FP4E1M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP4E1M2, DataType.FP4E2M1),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT8, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT8, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT8, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT8, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT8, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT8, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E8M0, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E8M0, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E8M0, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E8M0, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E8M0, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E4M3FN, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E4M3FN, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E4M3FN, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E5M2, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E5M2, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E5M2, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP8E5M2, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.HF8, DataType.INT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.HF8, DataType.UINT8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.HF8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.HF8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.HF8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.HF8, DataType.HF8),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.BF16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.BF16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.BF16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.BF16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT16, DataType.FP16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT16, DataType.BF16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT16, DataType.INT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT16, DataType.UINT16),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP32, DataType.FP32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP32, DataType.INT32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.FP32, DataType.UINT32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT32, DataType.FP32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT32, DataType.INT32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT32, DataType.UINT32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT32, DataType.FP32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT32, DataType.INT32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT32, DataType.UINT32),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT64, DataType.INT64),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.INT64, DataType.UINT64),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT64, DataType.INT64),
    ("GM", "Mat", Layout.DN, Layout.ZN, DataType.UINT64, DataType.UINT64),
    # NZ → NZ（同位宽组合）
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E1M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E2M1),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT8, DataType.INT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT8, DataType.UINT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT8, DataType.HF8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.INT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.UINT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.HF8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.INT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.UINT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E8M0),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E5M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.HF8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.INT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.UINT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.HF8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.INT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.UINT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E8M0),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.HF8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.HF8, DataType.INT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.HF8, DataType.UINT8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E8M0),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E4M3FN),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E5M2),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.HF8, DataType.HF8),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP16, DataType.FP16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP16, DataType.BF16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP16, DataType.INT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP16, DataType.UINT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.BF16, DataType.FP16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.BF16, DataType.BF16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.BF16, DataType.INT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.BF16, DataType.UINT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT16, DataType.FP16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT16, DataType.BF16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT16, DataType.INT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT16, DataType.UINT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.FP16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.BF16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.INT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.UINT16),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.INT32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.UINT32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.UINT32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.FP32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.INT32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.UINT32),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT64, DataType.INT64),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.INT64, DataType.UINT64),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT64, DataType.INT64),
    ("GM", "Mat", Layout.NZ, Layout.NZ, DataType.UINT64, DataType.UINT64),
)

# 3510 store 合法组合表（store / store_tile 共用，非量化路径）
_3510_STORE_COMBOS = (
    # ND → ND（17 种 × 同位宽组合）
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP4E2M1, DataType.FP4E1M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP4E1M2, DataType.FP4E2M1),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT8, DataType.INT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT8, DataType.UINT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT8, DataType.HF8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT8, DataType.INT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT8, DataType.UINT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT8, DataType.HF8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.INT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.UINT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.FP8E5M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E8M0, DataType.HF8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.INT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.UINT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E4M3FN, DataType.HF8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.INT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.UINT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E8M0),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP8E5M2, DataType.HF8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.HF8, DataType.INT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.HF8, DataType.UINT8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.HF8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.HF8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.HF8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.HF8, DataType.HF8),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP16, DataType.FP16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP16, DataType.BF16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP16, DataType.INT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP16, DataType.UINT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.BF16, DataType.FP16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.BF16, DataType.BF16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.BF16, DataType.INT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.BF16, DataType.UINT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT16, DataType.FP16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT16, DataType.BF16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT16, DataType.INT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT16, DataType.UINT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT16, DataType.FP16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT16, DataType.BF16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT16, DataType.INT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT16, DataType.UINT16),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP32, DataType.FP32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP32, DataType.INT32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.FP32, DataType.UINT32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT32, DataType.FP32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT32, DataType.INT32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT32, DataType.UINT32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT32, DataType.FP32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT32, DataType.INT32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT32, DataType.UINT32),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT64, DataType.INT64),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.INT64, DataType.UINT64),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT64, DataType.INT64),
    ("Vec", "GM", Layout.ND, Layout.ND, DataType.UINT64, DataType.UINT64),
    # DN → DN（17 种 × 同位宽组合，单列形态 [N,1]）
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP4E2M1, DataType.FP4E1M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP4E1M2, DataType.FP4E2M1),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT8, DataType.INT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT8, DataType.UINT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT8, DataType.HF8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT8, DataType.INT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT8, DataType.UINT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT8, DataType.HF8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.INT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.UINT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.FP8E5M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E8M0, DataType.HF8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.INT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.UINT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E4M3FN, DataType.HF8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.INT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.UINT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.FP8E8M0),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP8E5M2, DataType.HF8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.HF8, DataType.INT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.HF8, DataType.UINT8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.HF8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.HF8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.HF8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.HF8, DataType.HF8),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP16, DataType.FP16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP16, DataType.BF16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP16, DataType.INT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP16, DataType.UINT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.BF16, DataType.FP16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.BF16, DataType.BF16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.BF16, DataType.INT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.BF16, DataType.UINT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT16, DataType.FP16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT16, DataType.BF16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT16, DataType.INT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT16, DataType.UINT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT16, DataType.FP16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT16, DataType.BF16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT16, DataType.INT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT16, DataType.UINT16),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP32, DataType.FP32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP32, DataType.INT32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.FP32, DataType.UINT32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT32, DataType.FP32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT32, DataType.INT32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT32, DataType.UINT32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT32, DataType.FP32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT32, DataType.INT32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT32, DataType.UINT32),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT64, DataType.INT64),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.INT64, DataType.UINT64),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT64, DataType.INT64),
    ("Vec", "GM", Layout.DN, Layout.DN, DataType.UINT64, DataType.UINT64),
    # NZ → NZ（17 种 × 同位宽组合）
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E2M1),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP4E2M1, DataType.FP4E1M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E2M1),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP4E1M2, DataType.FP4E1M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT8, DataType.INT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT8, DataType.UINT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT8, DataType.HF8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.INT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.UINT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT8, DataType.HF8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.INT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.UINT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E8M0),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.FP8E5M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E8M0, DataType.HF8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.INT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.UINT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E8M0),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.FP8E5M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E4M3FN, DataType.HF8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.INT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.UINT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E8M0),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.FP8E5M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP8E5M2, DataType.HF8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.HF8, DataType.INT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.HF8, DataType.UINT8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E8M0),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E4M3FN),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.HF8, DataType.FP8E5M2),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.HF8, DataType.HF8),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP16, DataType.FP16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP16, DataType.BF16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP16, DataType.INT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP16, DataType.UINT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.BF16, DataType.FP16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.BF16, DataType.BF16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.BF16, DataType.INT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.BF16, DataType.UINT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT16, DataType.FP16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT16, DataType.BF16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT16, DataType.INT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT16, DataType.UINT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.FP16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.BF16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.INT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT16, DataType.UINT16),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.INT32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.UINT32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.UINT32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.FP32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.INT32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT32, DataType.UINT32),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT64, DataType.INT64),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.INT64, DataType.UINT64),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT64, DataType.INT64),
    ("Vec", "GM", Layout.NZ, Layout.NZ, DataType.UINT64, DataType.UINT64),
# ══ Acc → GM ══
    # 同类型：NZ → ND / NZ → NZ（CheckStaticAcc 对 ND/NZ dtype 白名单一致，
    # TStoreAccND/TStoreAccNZ 共用 fixpipe copy_matrix_cc_to_gm）
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP32),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.INT32, DataType.INT32),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT32),
    # 跨类型（fixpipe 数值转换，无 scale）：NZ → ND / NZ → NZ
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.BF16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.INT32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.INT32, DataType.BF16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.BF16),
)

# 3510 store 量化合法组合表（store / store_tile 带 scale）
_3510_STORE_QUANT_COMBOS = (
    # FP32 累加量化：NZ → ND / NZ → NZ
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.INT8),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.INT8),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.UINT8),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.UINT8),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.BF16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.HF8),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.HF8),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.FP32, DataType.FP32),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP32),
    # INT32 累加量化：NZ → ND / NZ → NZ
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.INT32, DataType.INT8),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT8),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.INT32, DataType.UINT8),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.UINT8),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.INT32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP16),
    ("Acc", "GM", Layout.NZ, Layout.ND, DataType.INT32, DataType.BF16),
    ("Acc", "GM", Layout.NZ, Layout.NZ, DataType.INT32, DataType.BF16),
)


_3510_INSERT_QUANT_COMBOS = (
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.BF16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.HF8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.FP32, DataType.FP8E4M3FN),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.INT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.UINT8),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.FP16),
    ("Acc", "Mat", Layout.NZ, Layout.NZ, DataType.INT32, DataType.BF16),
)

_LAYOUT_DTYPE_COMBOS = {
    "load": {arch: (_3510_LOAD_COMBOS, None) for arch in (NpuArch.DAV_3510, NpuArch.DAV_9201)},
    "load_tile": {arch: (_3510_LOAD_COMBOS, None) for arch in (NpuArch.DAV_3510, NpuArch.DAV_9201)},
    "store": {arch: (_3510_STORE_COMBOS, _3510_STORE_QUANT_COMBOS) for arch in (NpuArch.DAV_3510, NpuArch.DAV_9201)},
    "store_tile": {
        arch: (_3510_STORE_COMBOS, _3510_STORE_QUANT_COMBOS)
        for arch in (NpuArch.DAV_3510, NpuArch.DAV_9201)
    },
    "move": {arch: (_3510_MOVE_COMBOS, _3510_MOVE_QUANT_COMBOS) for arch in (NpuArch.DAV_3510, NpuArch.DAV_9201)},
    "extract": {
        arch: (_3510_EXTRACT_COMBOS, _3510_INSERT_QUANT_COMBOS)
        for arch in (NpuArch.DAV_3510, NpuArch.DAV_9201)
    },
    "insert": {arch: (_3510_INSERT_COMBOS, _3510_INSERT_QUANT_COMBOS) for arch in (NpuArch.DAV_3510, NpuArch.DAV_9201)},
}


def _effective_gm_layout(type_: "_IRTileType | _ir_core.TensorType", is_transpose: bool) -> "Layout":
    """Effective GM layout for load: NZ declaration wins, else order decides (降序→DN, 否则 ND)."""
    declared = _operand_layout(type_)
    if declared == Layout.NZ:
        return Layout.NZ
    return Layout.DN if is_transpose else Layout.ND


def _frontend_dtype_name(dtype: DataType) -> str:
    """前端数据类型名（如 ``DT_FP16``），与 ``pl.TileType(dtype=pl.DT_FP16)`` 一致。"""
    raw = str(dtype)
    if raw == "bfloat16":
        return "DT_BF16"
    return f"DT_{raw.upper()}"


def _check_layout_dtype(
    op: str,
    src: Expr,
    dst: Expr,
    *,
    is_transpose: bool | None = None,
    quant: bool = False,
) -> None:
    """Validate the layout × dtype combination of a data-movement op.

    把 src/dst 解析为六元组 (src_loc, dst_loc, src_layout, dst_layout, src_dtype, dst_dtype)，
    按操作对应的匹配规则与合法组合表比较，命中即合法。

    Args:
        op: data-movement kind（load / load_tile / store / store_tile / move / extract / insert）。
        src: source operand —— tile 或 tensor。
        dst: destination operand —— tile 或 tensor。
        is_transpose: load 专用——降序 order 时 GM 有效排布为 DN（仅 load 传入）；
                      为 None 时 GM 排布取 tensor 声明值（store 场景）。
        quant: 量化路径时查量化组合表，非量化时查主组合表（按 NpuArch 索引）。
    """
    from pypto_pro.runtime.jit import get_current_arch

    arch = get_current_arch()
    combos_by_arch = _LAYOUT_DTYPE_COMBOS.get(op)
    if combos_by_arch is None:
        return
    combos_entry = combos_by_arch.get(arch)
    if combos_entry is None:
        # 该架构暂未配置支持表——不做硬拦截，后续按平台补表后自动生效。
        return
    combos = combos_entry[1] if quant else combos_entry[0]
    if combos is None:
        return

    src_loc = _loc_name(src.type)
    dst_loc = _loc_name(dst.type)
    src_layout = _operand_layout(src.type)
    dst_layout = _operand_layout(dst.type)
    src_dtype = _operand_dtype(src.type)
    dst_dtype = _operand_dtype(dst.type)
    if (
        src_dtype is None
        or dst_dtype is None
        or src_loc is None
        or dst_loc is None
    ):
        # 解析不出确定信息（动态/未声明 layout）时不做硬拦截，避免误伤。
        return
    if op in ("load", "load_tile", "store", "store_tile") and (
        src_layout is None or dst_layout is None
    ):
        return
    if is_transpose is not None and src_loc == "GM":
        src_layout = _effective_gm_layout(src.type, is_transpose)
        # Mirror codegen's IsDNAccessLayout: a single-column access (tile last dim == 1)
        # declares the GM tensor as Layout::DN, so the emitted load is DN -> DN even when
        # the tensor itself is ND. Without this, a valid column load into a DN tile would
        # be misreported as an unsupported ND -> DN combination.
        if dst_layout == Layout.DN:
            if _has_unit_last_axis(dst.type.shape):
                src_layout = Layout.DN
            elif src_layout == Layout.ND:
                raise InvalidFormat(f"{op}: DN Tile requires a descending order mapping")
    elif src_loc != "GM" and dst_loc == "GM":
        # Mirror codegen's IsDNAccessLayout on the store side: a single-column tile
        # (last dim == 1) makes codegen declare the GM tensor as Layout::DN, so the
        # emitted store is DN -> DN even when the tensor is ND.
        if src_layout == Layout.DN and _has_unit_last_axis(src.type.shape):
            dst_layout = Layout.DN

    combo = (src_loc, dst_loc, src_layout, dst_layout, src_dtype, dst_dtype)
    if any(_layout_dtype_combo_matches(combo, rule) for rule in combos):
        return
    kind = "quantized " if quant else ""
    display_op = "move" if op == "extract" else op
    src_layout_name = src_layout.name if src_layout is not None else "unknown"
    dst_layout_name = dst_layout.name if dst_layout is not None else "unknown"
    hint = (
        " Hint: if the Tile dtype is not supported by store, use `pl.reinterpret` to view it as a "
        "same-bit-width supported dtype, then store it."
        if op in ("store", "store_tile")
        else ""
    )
    raise InvalidType(
        f"{display_op}: unsupported {kind}layout/dtype combination "
        f"src={src_loc}({src_layout_name},{_frontend_dtype_name(src_dtype)}) "
        f"-> dst={dst_loc}({dst_layout_name},{_frontend_dtype_name(dst_dtype)}); "
        f"please check whether the data path supports the current memory space, layout, and dtype.{hint}"
    )


@dataclass
class TileType:
    """Tile type descriptor containing shape, dtype, and TileView parameters."""

    shape: "Sequence[int] | _ir_core.MakeTuple"
    dtype: DataType
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None
    layout: Optional[Layout] = None
    fractal: Optional[int] = None
    pad: Optional[int] = None
    compact: Optional[int] = 1

    def __post_init__(self):
        _validate_tile_type_params(self)
        self.pad = _normalize_tile_pad(self.pad)
        if _is_conv_layout(self.layout):
            _validate_conv_tile_type(self)
            return
        _validate_subbyte_tile_shape(self)
        _apply_default_layout(self)
        _validate_tile_layout_alignments(self)


# Memory-space address alignment requirements (in bytes).
# Hardware constraint: misaligned tile addresses cause silent corruption or
# device-side runtime errors on move/load/store operations.
# - L1 (Mat) buffer:      32-byte alignment
# - L0A (Left) buffer:   512-byte alignment
# - L0B (Right) buffer:  512-byte alignment
# - L0C (Acc) buffer:     64-byte alignment
# - Vec (UB) buffer:      32-byte alignment
_MEMORY_ALIGNMENT: dict[MemorySpace, int] = {
    MemorySpace.Mat: 32,
    MemorySpace.Vec: 32,
    MemorySpace.Left: 512,
    MemorySpace.Right: 512,
    MemorySpace.Acc: 64,
    MemorySpace.ScaleLeft: 32,
    MemorySpace.ScaleRight: 32,
}


def _validate_tile_addr_alignment(
    addr: int,
    target_memory: MemorySpace,
    span: "Span | None" = None,
) -> None:
    """Validate that a tile address is properly aligned for its memory space.

    Raises ValueError with a descriptive message if the address is misaligned.
    """
    required = _MEMORY_ALIGNMENT.get(target_memory)
    if required is None:
        return  # DDR / Scaling / Bias — no enforced alignment
    if addr % required != 0:
        mem_name = str(target_memory).replace("MemorySpace.", "")
        raise InvalidTile(
            f"Tile address 0x{addr:05X} ({addr}) is not {required}-byte aligned "
            f"for memory space {mem_name}. "
            f"Address must be a multiple of {required}.",
            span=span,
        )


def _get_memory_capacity(target_memory: MemorySpace, span: Span | None = None) -> int:
    """On-chip buffer capacity (bytes) for ``target_memory``.

    The target platform comes from the pypto_pro compile flow: ``pl.jit``
    resolves arch (explicit argument, PYPTOPRO_JIT_ARCH, or hardware probe)
    and exports it via PYPTOPRO_JIT_ARCH before kernel parsing, so reading it
    here is reliable. The capacity is queried per-target via
    ``pypto_impl.GetMemoryLimitForArch`` (state-free, no platform-switch
    staleness), which also owns the memory-space key mapping and the arch ->
    platform ini selection. Returns 0 (caller warns and skips validation)
    when the capacity cannot be determined, instead of failing the build.
    """
    from pypto_pro.runtime.jit import get_current_arch
    from pypto_pro.runtime.platform import get_memory_limit

    arch = get_current_arch()
    cap = get_memory_limit(arch, target_memory.name)
    if cap <= 0 and target_memory in (
        MemorySpace.Vec, MemorySpace.Mat, MemorySpace.Left, MemorySpace.Right,
        MemorySpace.Acc, MemorySpace.Bias, MemorySpace.Scaling,
        MemorySpace.ScaleLeft, MemorySpace.ScaleRight,
    ):
        _warn_located(
            f"Cannot determine {target_memory.name} capacity for arch {arch}; tile capacity validation "
            "was skipped. Check the installed platform ini and native bindings.",
            CapacityUnknown,
            span,
        )
    # The platform ini reports the software-usable UB size, which may be smaller
    # than the physical buffer (e.g., ub_size=253952 = 256K - 8K reserved). Tile
    # address capacity must be checked against the physical buffer size.
    if target_memory == MemorySpace.Vec and cap == 256 * 1024 - 8 * 1024:
        cap = 256 * 1024
    return cap


def _validate_tile_addr_capacity(
    addr: int,
    size: int,
    target_memory: MemorySpace,
    span: "Span | None" = None,
) -> None:
    """Validate that addr + size does not exceed the buffer capacity.

    Mirrors SA-0353 in ``tassign_check.hpp`` (``end_addr <= capacity``).
    Also rejects negative addresses and non-positive sizes, which are
    nonsensical for on-chip buffer allocation and would silently bypass
    the capacity check (negative addr wraps to a large uint at runtime).
    Without this check, an out-of-bounds tile address silently passes
    compilation and triggers an AICore hardware trap (error 507015) at runtime.
    """
    mem_name = str(target_memory).replace("MemorySpace.", "")
    if addr < 0:
        raise InvalidTile(
            f"Tile addr {addr} is negative for memory space {mem_name}. "
            f"Address must be a non-negative integer.", span=span,
        )
    if size <= 0:
        raise InvalidTile(
            f"Tile size {size} must be positive for memory space {mem_name}.", span=span,
        )
    capacity = _get_memory_capacity(target_memory, span)
    if capacity <= 0:
        return
    end_addr = addr + size
    if end_addr > capacity:
        raise OutOfRange(
            f"Tile addr 0x{addr:X} + size {size} (0x{end_addr:X}) exceeds "
            f"{mem_name} capacity 0x{capacity:X} ({capacity} bytes). "
            f"Use a smaller address so that addr + tile_size <= {capacity}.", span=span,
        )


_MAKE_TILE_HINT = (
    "A tile occupies a fixed range of its memory space, so it needs an address, "
    "e.g. pl.make_tile(tile_type, addr=0x0); use pl.make_tile_group(type=..., "
    "addrs=..., depth=...) to lay several tiles out from one base address."
)


def tile_slot_size(shape: "Sequence[int] | _ir_core.MakeTuple", dtype: DataType) -> int:
    """Byte footprint of one tile, derived from its static shape and dtype.

    Shared by pl.make_tile()'s byte span and make_tile_group()'s slot stride,
    so a group slot is exactly as wide as a standalone tile of the same TileType.
    Sub-byte elements are packed before the total is rounded up to whole bytes.

    Raises ValueError when the shape is not a tuple of compile-time positive
    integers.
    """
    dims = list(shape.elements) if isinstance(shape, _ir_core.MakeTuple) else list(shape)
    static_dims = _const_shape_ints(dims)
    if static_dims is None:
        raise DynamicShapeUnsupported(f"tile shape must contain compile-time integers, got {dims}")
    if any(dim <= 0 for dim in static_dims):
        raise InvalidShape(f"tile shape dimensions must be positive, got {static_dims}")
    elems = 1
    for dim in static_dims:
        elems *= dim
    return (elems * int(dtype.get_bit()) + 7) // 8


def _const_shape_ints(shape) -> list[int] | None:
    """Read descriptor dimensions represented by Python ints or IR constants."""
    values = shape.elements if isinstance(shape, _ir_core.MakeTuple) else shape
    if not isinstance(values, (list, tuple)):
        return None
    dims = [value if _is_int(value) else _const_int_value(value) for value in values]
    return None if any(value is None for value in dims) else dims


# Tile config constants, mirrored from pto_tile.hpp TileConfig.
_ALIGNED_SIZE = 32  # alignedSize
_FIXED_ROW_SIZE = 16  # fixedRowSize
_FIXED_COL_SIZE = 16  # fixedColSize
_FIXED_MX_ROW_SIZE = 16  # fixedMxRowSize
_FIXED_MX_COL_SIZE = 2  # fixedMxColSize


def _validate_tile_layout_alignments(tt: "TileType") -> None:
    """Reject tile shapes the pto::Tile static_asserts would reject in kernel C++.

    Must run after ``_apply_default_layout`` so the effective layout and fractal
    defaults (Acc FP32/INT32 -> 1024, MX scale -> 32) are already in place;
    otherwise a boxed layout would be checked against the wrong inner box.
    """
    shape = _const_shape_ints(tt.shape)
    if shape is None or len(shape) < 2:
        return  # runtime dims or rank<2: no compile-time alignment to validate
    rows, cols = shape[0], shape[1]
    if rows <= 0 or cols <= 0:
        return
    dtype_size = max(1, (tt.dtype.get_bit() + 7) // 8)

    # A 1-wide tile has only its rows left to align, so the ISA accepts it in one
    # encoding only: BLayout::RowMajor needs Cols * sizeof(dtype) % 32 == 0, which
    # a 1-wide tile cannot satisfy, and codegen therefore emits ColMajor (DN). A
    # fractal layout would be silently rewritten (e.g. ZN on a [N, 1] tile -> NN).
    if tt.layout is not None and cols == 1 and tt.layout not in (
        Layout.DN,
    ):
        raise InvalidFormat(
            f"a tile whose last axis is 1 requires layout DN, got {tt.layout.name}: a "
            f"1-wide tile cannot meet the 32-byte column alignment a fractal layout needs, so "
            f"it is always emitted column-major. Use layout=pl.DN, or "
            f"widen the tile if you need {tt.layout.name}."
        )

    # Bias tiles are emitted NoneBox with Rows == 1: only the column stride matters.
    if tt.target_memory == MemorySpace.Bias:
        stride = cols * dtype_size
        if stride % _ALIGNED_SIZE != 0:
            raise InvalidShape(
                f"TileType Bias tile shape {tt.shape!r} {cols} * {dtype_size}B = "
                f"{stride}B must be {_ALIGNED_SIZE}-byte aligned (minimum "
                f"shape [1, {_ALIGNED_SIZE // dtype_size}] for "
                f"dtype {tt.dtype.to_string()})."
            )
        return

    slayout = _LAYOUT_TO_BS[tt.layout][1] if tt.layout is not None else 0

    # NoneBox (ND/DN, or Vec without a layout): the leading stride must be
    # 32-byte aligned; the stride axis is Cols for RowMajor and Rows for ColMajor.
    if slayout == 0:
        blayout = _LAYOUT_TO_BS[tt.layout][0] if tt.layout is not None else 1
        if cols == 1:
            blayout = 2  # codegen forces ColMajor on 1-wide tiles
        dim = cols if blayout == 1 else rows
        dim_name = "Cols" if blayout == 1 else "Rows"
        stride = dim * dtype_size
        if stride % _ALIGNED_SIZE != 0:
            raise InvalidType(
                f"Tile with {tt.layout.name if tt.layout is not None else 'default'} layout "
                f"(no inner box) requires {dim_name} * sizeof(dtype) = {dim} * {dtype_size} "
                f"= {stride} to be {_ALIGNED_SIZE}-byte aligned, got remainder "
                f"{stride % _ALIGNED_SIZE}. Pad {dim_name} so that "
                f"{dim_name} * sizeof(dtype) is a multiple of {_ALIGNED_SIZE}."
            )
        return

    # Boxed layout (NZ/ZN/ZZ/NN): inner box mirrors pto_tile.hpp Tile::getInnerRow/
    # getInnerCol — fractalCSize (_ACC_FRACTAL=1024) is fixed 16x16; fractalMxSize
    # (_MX_SCALE_FRACTAL=32) and fractalABSize (_DEFAULT_FRACTAL=512) follow slayout:
    # inner RowMajor → 16 x other, inner ColMajor → other x 16, where
    # other = alignedSize / sizeof(dtype) for 512 and fixedMxColSize(2) /
    # fixedMxRowSize(16) for 32.
    fractal = tt.fractal if tt.fractal is not None else _DEFAULT_FRACTAL
    if fractal == _ACC_FRACTAL:
        inner_rows, inner_cols = _FIXED_ROW_SIZE, _FIXED_COL_SIZE
    elif fractal == _MX_SCALE_FRACTAL:
        if slayout == 1:  # isInnerRowMajor → fixedMxRowSize=16, fixedMxColSize=2
            inner_rows, inner_cols = _FIXED_MX_ROW_SIZE, _FIXED_MX_COL_SIZE
        else:  # inner ColMajor → fixedMxColSize=2, fixedMxRowSize=16
            inner_rows, inner_cols = _FIXED_MX_COL_SIZE, _FIXED_MX_ROW_SIZE
    else:  # fractal == _DEFAULT_FRACTAL
        other_side = _ALIGNED_SIZE // dtype_size
        if slayout == 1:  # inner RowMajor (NZ, ZZ)
            inner_rows, inner_cols = _FIXED_ROW_SIZE, other_side
        else:  # inner ColMajor (ZN, NN)
            inner_rows, inner_cols = other_side, _FIXED_COL_SIZE

    layout_name = tt.layout.name if tt.layout is not None else "default"
    _pfx = f"Tile with {layout_name} layout (fractal={fractal})"
    if not (
        tt.target_memory == MemorySpace.Vec or fractal == _MX_SCALE_FRACTAL or rows == 1 or rows % inner_rows == 0
    ):
        raise InvalidVal(
            f"{_pfx} requires Rows ({rows}) to be a multiple of inner box rows "
            f"({inner_rows}), got remainder {rows % inner_rows}. "
            f"Pad Rows to a multiple of {inner_rows}, "
            f"or use Vec memory space, or set Rows=1."
        )
    if cols % inner_cols != 0:
        raise InvalidVal(
            f"{_pfx} requires Cols ({cols}) to be a multiple of inner box cols "
            f"({inner_cols}), got remainder {cols % inner_cols}. "
            f"Pad Cols to a multiple of {inner_cols}."
        )


def _validate_tile_type_params(tt: "TileType") -> None:
    """Reject invalid TileType parameter values early with actionable errors.

    Runs in ``TileType.__post_init__`` so both the AST-parsed path
    (``pl.TileType(...)`` inside a kernel) and direct Python construction are
    covered. Invalid values otherwise surface as opaque device/runtime errors
    (e.g. zero/negative tile sizes, misaligned derived addresses).
    """
    if not isinstance(tt.dtype, DataType):
        raise InvalidType(
            f"TileType dtype must be a pl.DT_* / DataType value, got {tt.dtype!r} "
            f"({type(tt.dtype).__name__})."
        )
    if not isinstance(tt.target_memory, MemorySpace):
        raise InvalidType(
            f"TileType target_memory must be a pl.MemorySpace value, got {tt.target_memory!r} "
            f"({type(tt.target_memory).__name__})."
        )
    if tt.target_memory == MemorySpace.DDR:
        raise NotSupported(
            "TileType target_memory does not support MemorySpace.DDR: a tile is a compile-time "
            "on-chip buffer, while DDR is off-chip GM.",
            hint="Use an on-chip memory space, e.g. target_memory=pl.MemorySpace.Vec.",
        )

    shape = _const_shape_ints(tt.shape)
    if shape is None:
        # Symbolic/dynamic dimensions (e.g. a runtime tensor dim) are allowed to
        # survive into the TileType; consuming code (make_tile/make_tile_group)
        # reports them when a compile-time size is actually required.
        pass
    else:
        if not shape:
            raise InvalidShape(f"TileType shape must not be empty, got {tt.shape!r}.")
        if any(s <= 0 for s in shape):
            raise InvalidArgument(
                f"TileType shape dimensions must be positive, got {tt.shape!r}. "
                f"Zero or negative tile dimensions produce invalid tile allocations."
            )

        if tt.target_memory == MemorySpace.Bias and shape[0] != 1:
            raise InvalidShape(
                f"TileType Bias tiles must have exactly 1 row (hardware Rows==1 constraint, "
                f"see TMatmul.hpp/TMov.hpp), got shape {tt.shape!r} with {shape[0]} rows. "
                f"Use shape=[1, N] for the Bias tile."
            )

    valid_shape = _const_shape_ints(tt.valid_shape)
    if valid_shape is not None and shape is not None:
        if len(valid_shape) != len(shape):
            raise InvalidShape(
                f"TileType valid_shape {tt.valid_shape!r} has rank {len(valid_shape)}, "
                f"which must match shape rank {len(shape)}."
            )
        for vs, s in zip(valid_shape, shape):
            if vs == -1:
                continue  # -1: dynamic valid dimension, set at runtime
            if vs <= 0:
                raise InvalidShape(
                    f"TileType valid_shape dimensions must be positive or -1 (dynamic), "
                    f"got {tt.valid_shape!r}."
                )
            if vs > s:
                raise InvalidShape(
                    f"TileType valid_shape dimension {vs} exceeds tile shape dimension {s}."
                )

    if tt.fractal is not None and not _is_int(tt.fractal):
        raise InvalidType(f"TileType fractal must be an integer, got {tt.fractal!r}.")

    if tt.compact is not None:
        if isinstance(tt.compact, bool) or not isinstance(tt.compact, int) or tt.compact not in _COMPACT_VALUES:
            raise InvalidArgument(
                f"TileType compact must be one of {sorted(_COMPACT_VALUES)} "
                f"(CompactMode: null=0/normal=1/row_plus_one=2/row_aligned_padding=3), "
                f"got {tt.compact!r}."
            )


def make_mutex_tile_expr(
    shape: "Sequence[int] | _ir_core.MakeTuple",
    dtype: DataType,
    target_memory: MemorySpace,
    *,
    addr: "int",
    size: "int | None" = None,
    valid_shape: "Sequence[int] | _ir_core.MakeTuple | None" = None,
    layout: "Layout | None" = None,
    fractal: "int | None" = None,
    pad: "int | None" = None,
    compact: "int | None" = 1,
    mutex_ids: "Sequence[int | Expr] | _ir_core.MakeTuple" = (),
    span: "Span | None" = None,
) -> Expr:
    """Create the block.make_mutex_tile allocation expression used by block buffers.

    This is the IR builder, which takes the tile's fields spread out. The DSL
    entry point is ``pl.make_tile(tile_type, *, addr)``, whose parse
    handler unpacks the TileType and calls this without mutex IDs. Framework
    callers can supply mutex_ids; omission binds no IDs. Both paths create
    the same internal op, with an empty mutex tuple for the zero-ID case.

    ``addr`` is required: it is what attaches a MemRef to the tile type, and a
    tile without one is placed nowhere in particular. ``size`` defaults to the
    footprint derived from ``shape`` and ``dtype``. The internal ``size`` override
    describes an allocation larger than that footprint; it is not exposed by the DSL.
    """
    actual_span = span or _span()
    shape_tuple = _to_make_tuple(shape, actual_span)
    valid_shape_tuple = (
        _to_make_tuple(valid_shape, actual_span) if valid_shape is not None else _ir_core.MakeTuple([], actual_span)
    )
    args = [shape_tuple, valid_shape_tuple, _to_make_tuple(mutex_ids, actual_span)]
    blayout: "int | None" = None
    slayout: "int | None" = None
    conv_layout: "int | None" = None
    if layout is not None:
        if _is_conv_layout(layout):
            # A conv layout has no blayout/slayout (2-D box) form; carry it as the
            # conv_layout attr so codegen can emit pto::ConvTile with the matching pto Layout.
            conv_layout = int(layout)
        else:
            b, s = _LAYOUT_TO_BS[layout]
            blayout = b
            slayout = s
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "target_memory": target_memory,
        "blayout": blayout,
        "slayout": slayout,
        "fractal": fractal,
        "pad": pad,
        "compact": compact,
        "conv_layout": conv_layout,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if addr is None:
        raise InvalidVal(f"make_mutex_tile_expr() requires 'addr', but it is missing. {_MAKE_TILE_HINT}", span=span)
    if size is None:
        try:
            size = tile_slot_size(shape_tuple, dtype)
        except PyptoProError as exc:
            raise InvalidOperation(
                f"pl.make_tile() cannot derive its byte span from the tile type: {message_of(exc)}. "
                "Use a static positive shape in pl.TileType.",
                span=span,
            ) from exc
    addr = _descriptor_integer(addr, "make_mutex_tile_expr addr")
    size = _descriptor_integer(size, "make_mutex_tile_expr size")
    _validate_tile_addr_alignment(addr, target_memory, span)
    _validate_tile_addr_capacity(addr, size, target_memory, span)
    global mem_id
    mem_id += 1
    kwargs["memref_addr"] = addr
    kwargs["memref_size"] = size
    kwargs["memref_id"] = mem_id
    return _ir_core.create_op_call(block_ir_op("make_mutex_tile"), args, kwargs, actual_span)


# ---------------------------------------------------------------------------
# Value-facing implementations — registered at import time
# ---------------------------------------------------------------------------


# Keep AST registration: parse shape dimensions without materializing the outer
# list/tuple as runtime IR, which would introduce unnecessary tuple anchors.
def _descriptor_integer(value, subject: str) -> int:
    """Decode a constant in the legacy descriptor parser, which owns Python values."""
    constant = _const_scalar_value(value)
    if not _is_int(constant):
        raise InvalidType(f"{subject} must be a compile-time integer constant")
    return constant


@op_impl_ast("TileType")
def _parse_tile_type(self, call: ast.Call):
    kwargs = {}
    for keyword in call.keywords:
        node = keyword.value
        if keyword.arg in ("shape", "valid_shape") and isinstance(node, (ast.List, ast.Tuple)):
            kwargs[keyword.arg] = [self.parse_expression(element) for element in node.elts]
        else:
            kwargs[keyword.arg] = self.parse_expression(node)
    self._check_declared_types("TileType", [], kwargs)
    for key, value in kwargs.items():
        if _is_none(value):
            kwargs[key] = None
        elif key in ("shape", "valid_shape"):
            elements = getattr(value, "elements", value)
            try:
                if not isinstance(elements, (list, tuple)):
                    raise InvalidType(f"{key} must be a list or tuple")
                dimensions = []
                for index, element in enumerate(elements):
                    dimensions.append(_descriptor_integer(element, f"{key}[{index}]"))
                kwargs[key] = dimensions
            except InvalidType as error:
                error.hint = (
                    "TileType shape/valid_shape must contain compile-time constants; "
                    "use pl.set_validshape() for a runtime valid shape."
                )
                raise
        elif key in ("fractal", "compact", "pad"):
            if key != "pad" or not isinstance(value, TilePad):
                kwargs[key] = _descriptor_integer(value, key)
    return TileType(**kwargs)


@op_impl("make_tile")
def _build_make_tile(self, tile_type, /, *, addr, span: Span) -> Expr:
    kwargs = vars(tile_type).copy()
    try:
        kwargs["addr"] = require_const_int(addr, subject="'addr'")
    except InvalidType as error:
        error.hint = (
            "addr is fixed while parsing; pass a literal, or a variable bound to "
            "literals/constants — not a runtime value such as a tensor shape or loop index"
        )
        raise
    tile = make_mutex_tile_expr(**kwargs, span=span)
    self._record_tile_high_water(tile)
    return tile


# ---------------------------------------------------------------------------
# reinterpret — Tile/TileGroup metadata reinterpretation (zero data movement)
# ---------------------------------------------------------------------------


def _hw_attr(hw, field: str) -> "int | None":
    """One hardware_info field as an int, or None when unset/negative sentinel."""
    if hw is None:
        return None
    value = getattr(hw, field, None)
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _ir_reinterpret(
    src: Expr, *, shape: "Sequence[int] | None" = None,
    dtype: "DataType | None" = None, layout: "Layout | None" = None,
    span: "Span | None" = None,
) -> Expr:
    """Keep the runtime source explicit; the backend binds its current address."""
    actual_span = span or _span()
    if not isinstance(src.type, _IRTileType):
        raise InvalidType(f"reinterpret: expected a Tile, got {type(src.type).__name__}")
    source_type = src.type
    if source_type.memref is None:
        raise InvalidArgument(
            "reinterpret: source tile has no MemRef (bind it with pl.make_tile first)",
        )
    new_shape = list(shape) if shape is not None else _static_shape_ints(source_type.shape, "reinterpret: source")
    new_dtype = dtype if dtype is not None else source_type.dtype
    new_layout = layout if layout is not None else _tile_layout(source_type)
    if _is_conv_layout(new_layout):
        raise NotSupported(
            "reinterpret: the target layout of a reinterpret cannot be a conv layout "
            "(conv-layout tiles support declaration and allocation only); to reinterpret a "
            "conv-layout tile, specify an explicit non-conv layout",
            span=span,
        )
    b, s = _LAYOUT_TO_BS[new_layout]
    return _ir_core.create_op_call(
        "block.reinterpret", [src, _to_make_tuple(new_shape, actual_span)],
        {"dtype": new_dtype, "blayout": b, "slayout": s}, actual_span,
    )


@op_impl("reinterpret")
def _build_reinterpret(
    self, tile: Expr, /, *, dtype: DataType | None = None, shape: Expr | None = None,
    layout: TensorLayout | None = None, span: Span,
) -> Expr:
    kwargs = {"dtype": dtype, "shape": shape, "layout": layout}
    for key, value in kwargs.items():
        if _is_none(value):
            kwargs[key] = None

    new_shape = kwargs.get("shape")
    if new_shape is not None:
        new_shape = require_const_int_list(
            new_shape, subject="shape", min_length=1,
        )
        kwargs["shape"] = new_shape

    if kwargs.get("dtype") is None and kwargs.get("shape") is None and kwargs.get("layout") is None:
        raise InvalidType(
            "pl.reinterpret() requires at least one of dtype/shape/layout to override",
            hint="e.g. pl.reinterpret(tile, shape=[64, 64], dtype=pl.DT_BF16)",
        )

    if kwargs.get("dtype") is not None and kwargs.get("shape") is None:
        raise InvalidShape(
            "pl.reinterpret() 'shape' is required when 'dtype' changes",
            hint="state the new element count explicitly: pl.reinterpret(tile, dtype=..., shape=[rows, cols])",
        )

    # All override args are validated once above, before dispatch: neither the
    # tile branch nor the tile_group branch re-validates them. The "is a Tile"
    # check lives in _ir_reinterpret, which both branches go through.
    if self.is_tile_group(tile):
        return self._build_reinterpreted_group(tile, kwargs, span)

    new_tile = _ir_reinterpret(tile, shape=new_shape, dtype=kwargs.get("dtype"), layout=kwargs.get("layout"), span=span)

    # The reinterpreted tile aliases the same buffer, so it inherits the source's
    # group origin (same propagation pattern as GetItemExpr in _expression_parser).
    self._transfer_tile_group_metadata(new_tile, tile)
    return new_tile


def _static_shape_ints(shape: Sequence[Expr], what: str) -> list[int]:
    """Read the compile-time integer dimensions of a tile/tensor type."""
    return [require_const_int(dim, subject=f"{what} shape dimension") for dim in shape]


def _validate_fp_shape_dtype(fp_shape_ints: list[int], scale_dtype: DataType, what: str) -> None:
    """Validate a per-channel scale Tile against the FixPipe deqTensor constraints.

    Hardware FixPipe deqTensor (pto-isa TMov.hpp): row must be 1;
    col * sizeof(int64) 128B-aligned (col % 16 == 0); col * sizeof(int64) <= 4KB FB (col <= 512).
    ``[N, 1]`` per-row scaling is NOT supported — it fails at compile time with
    "TMov: When TileType is Scaling, row must be 1."
    """
    if len(fp_shape_ints) != 2:
        raise InvalidShape(f"{what} must be 2D, got shape {fp_shape_ints}")
    if fp_shape_ints[0] != 1:
        raise NotSupported(
            f"{what} must have shape [1, N] (row == 1), got [{fp_shape_ints[0]}, {fp_shape_ints[1]}]. "
            f"Hardware FixPipe deqTensor only supports per-column scaling ([1, N]); "
            f"[N, 1] per-row scaling is not supported. "
            f"Per-token (row-wise) quantization must be done in the Vector (UB) domain."
        )
    if fp_shape_ints[1] % 16 != 0:
        raise InvalidTile(
            f"{what} [1, N]: N must be a multiple of 16 (128B alignment for INT64), "
            f"got [{fp_shape_ints[0]}, {fp_shape_ints[1]}]"
        )
    if fp_shape_ints[1] > 512:
        raise OutOfRange(
            f"{what} [1, N]: N must be <= 512 (4KB fixpipe buffer limit for INT64), "
            f"got [{fp_shape_ints[0]}, {fp_shape_ints[1]}]"
        )
    if scale_dtype == DataType.FP32:
        raise InvalidType(
            f"{what} dtype FP32 is not supported. "
            f"Please convert FP32 scale to INT64 using torch_npu.npu_trans_quant_param() before passing to kernel. "
            f"Example: scale_int64 = torch_npu.npu_trans_quant_param(scale_fp32.npu())"
        )
    if scale_dtype != DataType.INT64:
        raise InvalidType(
            f"{what} dtype must be INT64, got {scale_dtype}. "
            f"For FP32 scale, use torch_npu.npu_trans_quant_param() to convert to INT64."
        )


def _validate_scaling_tile(scale: Expr | None) -> None:
    """Pre-hook: validate a user-prepared Scaling tile when scale is a Tile (per-channel quantization).

    Per-channel quantization requires a user-prepared deqTensor tile: the user
    builds a Scaling tile (MemorySpace.Scaling, [1, N] INT64), owns the data
    flow (load -> move -> sync MTE1->FIX before the store/move/insert), and passes it
    as ``scale``. This hook only validates it — no auto-allocation of
    Scaling/Mat tiles and no sync events are emitted. The validated tile stays
    in the ``scale`` kwarg and is resolved by ``_resolve_scale_param`` in the
    builder to the store/move/insert Scaling Tile operand.

    A GM Tensor scale is rejected at parse time: the automatic per-channel path
    (auto-allocated Mat intermediate + auto sync events) has been removed; users
    must prepare the Scaling tile themselves.

    Scale tile shape constraint:
    - Hardware FixPipe deqTensor only supports per-column scaling: scale tile must be
      ``[1, N]`` (row == 1), with ``N % 16 == 0`` (128B alignment) and ``N <= 512`` (4KB
      fixpipe buffer). ``[N, 1]`` per-row scaling is rejected here with a clear error
      (hardware would otherwise fail compilation with "TMov: row must be 1").

    FP32 scale tile limitation:
    - Only INT64 scale tiles are supported; FP32 scale tensors must be converted
      on the host side via ``torch_npu.npu_trans_quant_param`` before being loaded
      into the Scaling tile.
    """
    if scale is None:
        return
    scale_type = getattr(scale, "type", None)

    if isinstance(scale_type, _ir_core.TileType):
        mem = getattr(getattr(scale_type, "memref", None), "memory_space", None)
        if mem != MemorySpace.Scaling:
            raise InvalidOperation(
                f"scale Tile must be allocated in MemorySpace.Scaling for per-channel quantization, got {mem}"
            )
        fp_shape_ints = _static_shape_ints(scale_type.shape, "scale tile")
        _validate_fp_shape_dtype(fp_shape_ints, scale_type.dtype, "scale tile")
        return

    if isinstance(scale_type, _ir_core.TensorType):
        raise InvalidVal(
            "scale Tensor is not supported for per-channel quantization — pass a "
            "user-prepared Scaling Tile (MemorySpace.Scaling, shape [1, N], INT64) "
            "instead, and ensure it is ready (load -> move -> sync MTE1->FIX) before "
            "the store/move/insert"
        )

    return  # scalar / runtime bits -> handled by _resolve_scale_param


# ---------------------------------------------------------------------------
# Builder helpers for merged interfaces
# ---------------------------------------------------------------------------


# Reduce input-dtype support per dim on Ascend 950PR/950DT, from the ISA
# instruction dtype lists (pto-isa docs/isa/TROWMAX_zh.md etc.). Row and col
# instructions differ: e.g. TROWMAX/TROWMIN have no bf16 while TCOLMAX/TCOLMIN
# do, and TROWSUM/TCOLSUM have no int8/uint8. Checked against the reduce input
# (src) dtype; argmax/argmin have no ISA dtype list and are not gated here.
_ROW_SUM_DTYPES = (
    DataType.FP16,
    DataType.FP32,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT64,
    DataType.INT16
)
_ROW_MAX_MIN_DTYPES = (
    DataType.FP16,
    DataType.FP32,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT64,
    DataType.INT16,
    DataType.INT8,
    DataType.UINT8
)
_COL_MAX_MIN_DTYPES = (
    DataType.FP16,
    DataType.FP32,
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
    DataType.INT64,
    DataType.UINT64,
    DataType.BF16,
)
_COL_SUM_DTYPES = (
    DataType.FP16,
    DataType.FP32,
    DataType.INT8,
    DataType.UINT8,
    DataType.INT16,
    DataType.UINT16,
    DataType.INT32,
    DataType.UINT32,
    DataType.INT64,
    DataType.UINT64,
    DataType.BF16,
)
_DIM_REDUCE_DTYPES: dict[str, tuple[tuple[DataType, ...], tuple[DataType, ...]]] = {
    "max": (_ROW_MAX_MIN_DTYPES, _COL_MAX_MIN_DTYPES),
    "min": (_ROW_MAX_MIN_DTYPES, _COL_MAX_MIN_DTYPES),
    "sum": (_ROW_SUM_DTYPES, _COL_SUM_DTYPES),
    "prod": (_ROW_PROD_DTYPES, _COL_PROD_DTYPES),
}

def _create_tile_scalar_op(
    out: Expr, lhs: Expr, rhs: Expr, *, tile_op: str, scalar_op: str, span: Span | None = None, **kwargs
) -> Expr:
    """Dispatch to tile-tile or tile-scalar IR op based on rhs type."""
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        target_op = block_ir_op(tile_op)
    else:
        from pypto_pro.language.parser.diagnostics import check_const_expr_fits_dtype

        target_op = block_ir_op(scalar_op)
        out_dtype = getattr(getattr(out, "type", None), "dtype", None)
        check_const_expr_fits_dtype(rhs, out_dtype, api=f"pl.{tile_op}")
    return _ir_core.create_op_call(target_op, [out, lhs, rhs], kwargs, span)


def _create_dim_op(
    args: list[Expr], *, row_op: str, col_op: str, dim: int | Expr = 0, span: Span | None = None,
) -> Expr:
    """Dispatch to row-wise or col-wise IR op based on dim."""
    if _is_int(dim):
        dim = _normalize_expr(dim, span)
    dim = require_const_int(dim, subject="'dim'", lower_bound=0, upper_bound=1)
    op_name = col_op.removeprefix("col_")
    src_dtype = getattr(getattr(args[1], "type", None), "dtype", None)
    dtype_sets = _DIM_REDUCE_DTYPES.get(op_name)
    if dtype_sets is not None and src_dtype is not None:
        # Row/col ISA instructions have different dtype ranges (e.g. TROWMAX
        # lacks bf16 while TCOLMAX supports it); validate the reduce input
        # dtype against the per-dim ISA support on Ascend 950.
        allowed = dtype_sets[dim]
        if src_dtype not in allowed:
            allowed_names = ", ".join(str(d) for d in allowed)
            raise NotSupported(
                f"{op_name}: dim={dim} reduce does not support src dtype {src_dtype}, supported: {allowed_names}"
            )
    ir_name = row_op if dim == 0 else col_op
    return _ir_core.create_op_call(block_ir_op(ir_name), args, {}, span)


@op_impl("select")
def _ir_select(self, out: Expr, mask: Expr, lhs: Expr, rhs: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_sel(out, mask, lhs, rhs, tmp, span=span)
    return _ir_sels(out, mask, lhs, tmp, rhs, span=span)


@op_impl("eq")
def _ir_eq(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.EQ.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.EQ.value)


@op_impl("ne")
def _ir_ne(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.NE.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.NE.value)


@op_impl("lt")
def _ir_lt(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.LT.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.LT.value)


@op_impl("le")
def _ir_le(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.LE.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.LE.value)


@op_impl("gt")
def _ir_gt(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.GT.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.GT.value)


@op_impl("ge")
def _ir_ge(self, out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None) -> Expr:
    if isinstance(getattr(rhs, "type", None), _ir_core.TileType):
        return _ir_cmp(out, lhs, rhs, span=span, cmp_mode=CmpMode.GE.value)
    return _ir_cmps(out, lhs, rhs, span=span, cmp_mode=CmpMode.GE.value)


@op_impl("sum")
def _ir_sum(self, out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _SUM_DTYPES, subject="sum")
    require_dtypes_match([dt], getattr(out.type, "dtype", None), subject="sum")
    return _create_dim_op([out, src, tmp], row_op="row_sum", col_op="col_sum", dim=dim, span=span)


@op_impl("prod")
def _ir_prod(self, out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    # TROWPROD and TCOLPROD admit different dtype sets on 3510; validate per dim
    # (TROWPROD: half/float/int16/int32; TCOLPROD adds uint16/uint32/bfloat16).
    dt = getattr(src.type, "dtype", None)
    require_dtypes_match([dt], getattr(out.type, "dtype", None), subject="prod")
    return _create_dim_op([out, src, tmp], row_op="row_prod", col_op="col_prod", dim=dim, span=span)


@op_impl("argmax")
def _ir_argmax(self, out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    # 3510 TROWARGMAX/TROWARGMIN (3-arg idx-only form): half/float source,
    # 32-bit index destination.
    require_dtype_in(getattr(src.type, "dtype", None), _FLOAT_UNARY_DTYPES, subject="argmax")
    require_dtype_in(getattr(out.type, "dtype", None), _ARG_IDX32_DTYPES, subject="argmax")
    return _create_dim_op([out, src, tmp], row_op="row_argmax", col_op="col_argmax", dim=dim, span=span)


@op_impl("argmin")
def _ir_argmin(self, out: Expr, src: Expr, tmp: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    # 3510 TROWARGMIN/TCOLARGMIN (3-arg idx-only form): half/float source,
    # 32-bit index destination.
    require_dtype_in(getattr(src.type, "dtype", None), _FLOAT_UNARY_DTYPES, subject="argmin")
    require_dtype_in(getattr(out.type, "dtype", None), _ARG_IDX32_DTYPES, subject="argmin")
    return _create_dim_op([out, src, tmp], row_op="row_argmin", col_op="col_argmin", dim=dim, span=span)


# Fixed-axis spellings were previously handled by the block-op fallback.
@op_impl("row_max")
def _build_row_max(self, out: Expr, src: Expr, tmp: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("row_max"), [out, src, tmp], {}, span)


@op_impl("row_min")
def _build_row_min(self, out: Expr, src: Expr, tmp: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("row_min"), [out, src, tmp], {}, span)


@op_impl("col_max")
def _build_col_max(self, out: Expr, src: Expr, tmp: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("col_max"), [out, src, tmp], {}, span)


@op_impl("col_min")
def _build_col_min(self, out: Expr, src: Expr, tmp: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("col_min"), [out, src, tmp], {}, span)


@op_impl("row_prod")
def _build_row_prod(self, out: Expr, src: Expr, tmp: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("row_prod"), [out, src, tmp], {}, span)


@op_impl("col_prod")
def _build_col_prod(self, out: Expr, src: Expr, tmp: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("col_prod"), [out, src, tmp], {}, span)


@op_impl("row_expand_sub")
def _build_row_expand_sub(self, out: Expr, src: Expr, row_vec: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("row_expand_sub"), [out, src, row_vec], {}, span)


@op_impl("row_expand_mul")
def _build_row_expand_mul(self, out: Expr, src: Expr, row_vec: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("row_expand_mul"), [out, src, row_vec], {}, span)


@op_impl("row_expand_div")
def _build_row_expand_div(self, out: Expr, src: Expr, row_vec: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("row_expand_div"), [out, src, row_vec], {}, span)


@op_impl("col_expand_sub")
def _build_col_expand_sub(self, out: Expr, src: Expr, col_vec: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("col_expand_sub"), [out, src, col_vec], {}, span)


@op_impl("col_expand_mul")
def _build_col_expand_mul(self, out: Expr, src: Expr, col_vec: Expr, *, span: Span) -> Expr:
    return _ir_core.create_op_call(block_ir_op("col_expand_mul"), [out, src, col_vec], {}, span)


@op_impl("expand_max")
def _ir_expand_max(self, out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _ROW_EXPAND_DTYPES, subject="expand_max")
    require_dtypes_match([getattr(out.type, "dtype", None)], dt, subject="expand_max")
    return _create_dim_op([out, src, scalar], row_op="row_expand_max", col_op="col_expand_max", dim=dim, span=span)


@op_impl("expand_min")
def _ir_expand_min(self, out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _ROW_EXPAND_DTYPES, subject="expand_min")
    require_dtypes_match([getattr(out.type, "dtype", None)], dt, subject="expand_min")
    return _create_dim_op([out, src, scalar], row_op="row_expand_min", col_op="col_expand_min", dim=dim, span=span)


@op_impl("expand_mul")
def _ir_expand_mul(self, out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _ROW_EXPAND_DTYPES, subject="expand_mul")
    require_dtypes_match([getattr(out.type, "dtype", None)], dt, subject="expand_mul")
    return _create_dim_op([out, src, scalar], row_op="row_expand_mul", col_op="col_expand_mul", dim=dim, span=span)


@op_impl("expand_sub")
def _ir_expand_sub(self, out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _ROW_EXPAND_DTYPES, subject="expand_sub")
    require_dtypes_match([getattr(out.type, "dtype", None)], dt, subject="expand_sub")
    return _create_dim_op([out, src, scalar], row_op="row_expand_sub", col_op="col_expand_sub", dim=dim, span=span)


@op_impl("expand_div")
def _ir_expand_div(self, out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _ROW_EXPAND_DTYPES, subject="expand_div")
    require_dtypes_match([getattr(out.type, "dtype", None)], dt, subject="expand_div")
    return _create_dim_op([out, src, scalar], row_op="row_expand_div", col_op="col_expand_div", dim=dim, span=span)


@op_impl("expand_add")
def _ir_expand_add(self, out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int | Expr = 0) -> Expr:
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _ROW_EXPAND_DTYPES, subject="expand_add")
    require_dtypes_match([getattr(out.type, "dtype", None)], dt, subject="expand_add")
    return _create_dim_op([out, src, scalar], row_op="row_expand_add", col_op="col_expand_add", dim=dim, span=span)


@op_impl("expand_expdif")
def _ir_expand_expdif(
    self, out: Expr, src: Expr, scalar: Expr, *, span: Span | None = None, dim: int | Expr = 0,
) -> Expr:
    # TROWEXPANDEXPDIF/TCOLEXPANDEXPDIF only provide half/float variants.
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _FLOAT_UNARY_DTYPES, subject="expand_expdif")
    require_dtypes_match([getattr(out.type, "dtype", None)], dt, subject="expand_expdif")
    return _create_dim_op(
        [out, src, scalar], row_op="row_expand_expdif", col_op="col_expand_expdif", dim=dim, span=span
    )


# ---------------------------------------------------------------------------
# Partial merge / gather / scatter / sort / histogram ops
# (validation mirrors the pto-isa 3510 TxxxCheck functions)
# ---------------------------------------------------------------------------
@op_impl("partadd")
def _ir_partadd(self, out: Expr, src0: Expr, src1: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BINARY_DTYPES, subject="partadd")
    require_dtypes_match([getattr(src0.type, "dtype", None), getattr(src1.type, "dtype", None)], dt, subject="partadd")
    return _ir_core.create_op_call(block_ir_op("partadd"), [out, src0, src1], {}, span or _span())


@op_impl("partmax")
def _ir_partmax(self, out: Expr, src0: Expr, src1: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BINARY_DTYPES, subject="partmax")
    require_dtypes_match([getattr(src0.type, "dtype", None), getattr(src1.type, "dtype", None)], dt, subject="partmax")
    return _ir_core.create_op_call(block_ir_op("partmax"), [out, src0, src1], {}, span or _span())


@op_impl("partmin")
def _ir_partmin(self, out: Expr, src0: Expr, src1: Expr, *, span: Span | None = None) -> Expr:
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _BINARY_DTYPES, subject="partmin")
    require_dtypes_match([getattr(src0.type, "dtype", None), getattr(src1.type, "dtype", None)], dt, subject="partmin")
    return _ir_core.create_op_call(block_ir_op("partmin"), [out, src0, src1], {}, span or _span())


@op_impl("partmul")
def _ir_partmul(self, out: Expr, src0: Expr, src1: Expr, *, span: Span | None = None) -> Expr:
    # TPARTMUL (3510) has no 8-bit variants.
    dt = getattr(out.type, "dtype", None)
    require_dtype_in(dt, _MUL_DTYPES, subject="partmul")
    require_dtypes_match([getattr(src0.type, "dtype", None), getattr(src1.type, "dtype", None)], dt, subject="partmul")
    return _ir_core.create_op_call(block_ir_op("partmul"), [out, src0, src1], {}, span or _span())


@op_impl("gather")
def _ir_gather(
    self, out: Expr, src: Expr, indices_or_k_value: Expr, tmp_or_cdst: Expr | None = None,
    tmp: Expr | None = None, *, cmp_mode: Expr | None = None, offset: Expr | None = None,
    span: Span | None = None,
) -> Expr:
    # An omitted cmp_mode selects the index form, even though the public default is 0.
    kwargs = {}
    if cmp_mode is not None:
        kwargs["cmp_mode"] = require_const_int(
            cmp_mode, subject="'cmp_mode'",
        )
    if offset is not None:
        kwargs["offset"] = require_const_int(
            offset, subject="'offset'",
        )
    if cmp_mode is not None:
        # Compare form (TGATHER with CmpMode): 32-bit counter destination.
        # IR args: [out, src, k_value, cdst, tmp]
        require_dtype_in(getattr(out.type, "dtype", None), _ARG_IDX32_DTYPES, subject="gather")
    else:
        # Index form (TGATHER CheckValid): dst/src share one dtype; index tiles
        # are 16/32-bit integers and 64-bit data requires a 32-bit index.
        # IR args: [out, src, indices] or [out, src, indices, tmp]
        require_dtypes_match([getattr(out.type, "dtype", None)], getattr(src.type, "dtype", None), subject="gather")
        idx_dt = getattr(indices_or_k_value.type, "dtype", None)
        require_dtype_in(idx_dt, _GATHER_IDX_DTYPES, subject="gather")
        src_dt = getattr(src.type, "dtype", None)
        if src_dt is not None and idx_dt is not None and src_dt.get_bit() == 64 and idx_dt.get_bit() != 32:
            raise InvalidType(f"gather: 64-bit data requires a 32-bit index tile, got {idx_dt}")
    ir_args = [out, src, indices_or_k_value]
    if tmp_or_cdst is not None:
        ir_args.append(tmp_or_cdst)
    if tmp is not None:
        ir_args.append(tmp)
    return _ir_core.create_op_call(block_ir_op("gather"), ir_args, kwargs, span or _span())


@op_impl("gatherb")
def _ir_gatherb(self, out: Expr, src: Expr, offsets: Expr, *, span: Span | None = None) -> Expr:
    # TGATHERB (3510): 1/2/4-byte destination elements, 32-bit byte-offset tiles.
    out_dt = getattr(out.type, "dtype", None)
    if out_dt is not None and out_dt.get_bit() not in (8, 16, 32):
        raise InvalidType(f"gatherb: unsupported out dtype {out_dt}, supported: b8/b16/b32")
    require_dtype_in(getattr(offsets.type, "dtype", None), _ARG_IDX32_DTYPES, subject="gatherb")
    return _ir_core.create_op_call(block_ir_op("gatherb"), [out, src, offsets], {}, span or _span())


@op_impl("gathermask")
def _ir_gathermask(
    self, out: Expr, src: Expr, *,
    pattern_mode: Expr, span: Span | None = None,
) -> Expr:
    pattern_mode = require_const_int(
        pattern_mode, subject="'pattern_mode'",
    )
    require_dtypes_match([getattr(out.type, "dtype", None)], getattr(src.type, "dtype", None), subject="gathermask")
    return _ir_core.create_op_call(
        block_ir_op("gathermask"), [out, src], {"pattern_mode": pattern_mode}, span or _span(),
    )


@op_impl("scatter")
def _ir_scatter(self, out: Expr, src: Expr, idx: Expr, *, span: Span | None = None) -> Expr:
    # TSCATTER (3510) index pairing: b8→b16, b16→b16, b32→b32, b64→b32.
    src_dt = getattr(src.type, "dtype", None)
    require_dtypes_match([getattr(out.type, "dtype", None)], src_dt, subject="scatter")
    idx_dt = getattr(idx.type, "dtype", None)
    require_dtype_in(idx_dt, _GATHER_IDX_DTYPES, subject="scatter")
    if src_dt is not None and idx_dt is not None:
        required_idx_bit = {8: 16, 16: 16, 32: 32, 64: 32}.get(src_dt.get_bit())
        if required_idx_bit is not None and idx_dt.get_bit() != required_idx_bit:
            raise InvalidType(
                f"scatter: {src_dt.get_bit()}-bit data requires a {required_idx_bit}-bit index tile, got {idx_dt}"
            )
    return _ir_core.create_op_call(block_ir_op("scatter"), [out, src, idx], {}, span or _span())


@op_impl("sort32")
def _ir_sort32(self, dst: Expr, src: Expr, idx: Expr, tmp: Expr | None = None, *, span: Span | None = None) -> Expr:
    # TSORT32 (3510): half/float values with uint32 indices.
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _SORT_DTYPES, subject="sort32")
    require_dtypes_match([getattr(dst.type, "dtype", None)], dt, subject="sort32")
    require_dtype_in(getattr(idx.type, "dtype", None), (DataType.UINT32,), subject="sort32")
    args = [dst, src, idx] + ([tmp] if tmp is not None else [])
    return _ir_core.create_op_call(block_ir_op("sort32"), args, {}, span or _span())


@op_impl("mrgsort")
def _ir_mrgsort(
    self, dst: Expr, src: Expr, *,
    block_len: Expr, span: Span | None = None,
) -> Expr:
    block_len = require_const_int(
        block_len, subject="'block_len'",
    )
    # TMRGSORT (3510): half/float only.
    dt = getattr(src.type, "dtype", None)
    require_dtype_in(dt, _SORT_DTYPES, subject="mrgsort")
    require_dtypes_match([getattr(dst.type, "dtype", None)], dt, subject="mrgsort")
    return _ir_core.create_op_call(block_ir_op("mrgsort"), [dst, src], {"block_len": block_len}, span or _span())


@op_impl("mrgsort2")
def _ir_mrgsort2(
    self, src0: Expr, src1: Expr, dst: Expr, tmp: Expr, *args: Expr,
    exhausted: Expr | None = None, span: Span | None = None,
) -> Expr:
    # None marks omission so the default does not add a new IR attribute.
    kwargs = {}
    if exhausted is not None:
        kwargs["exhausted"] = require_const_bool(exhausted, subject="'exhausted'")
    # TMRGSORT (3510): half/float only. Reorder to the backend argument layout
    # [dst, src0, tmp, src1, ...] — the plain fall-through produced misordered
    # IR args that the CCE codegen read as (dst=src0, src0=src1, tmp=dst, src1=tmp).
    dt = getattr(src0.type, "dtype", None)
    require_dtype_in(dt, _SORT_DTYPES, subject="mrgsort2")
    require_dtypes_match(
        [
            getattr(src1.type, "dtype", None),
            getattr(dst.type, "dtype", None),
            *(getattr(arg.type, "dtype", None) for arg in args),
        ],
        dt,
        subject="mrgsort2",
    )
    return _ir_core.create_op_call(block_ir_op("mrgsort2"), [dst, src0, tmp, src1, *args], kwargs, span or _span())


@op_impl("histogram")
def _ir_histogram(
    self, dst: Expr, src: Expr, idx: Expr, *, is_msb: Expr, span: Span | None = None,
) -> Expr:
    is_msb = require_const_bool(is_msb, subject="'is_msb'")
    # THISTOGRAM (3510): u16/u32 source, u8 filter index, u32 bin destination.
    require_dtype_in(getattr(src.type, "dtype", None), _HISTOGRAM_SRC_DTYPES, subject="histogram")
    require_dtype_in(getattr(idx.type, "dtype", None), _HISTOGRAM_IDX_DTYPES, subject="histogram")
    require_dtype_in(getattr(dst.type, "dtype", None), _HISTOGRAM_DST_DTYPES, subject="histogram")
    return _ir_core.create_op_call(block_ir_op("histogram"), [dst, src, idx], {"is_msb": is_msb}, span or _span())


@op_impl("fill_index")
def _ir_fill_index(self, out: Expr, start: Expr, *, span: Span | None = None) -> Expr:
    # TCI (3510): 16/32-bit integer tiles.
    require_dtype_in(getattr(out.type, "dtype", None), _FILL_INDEX_DTYPES, subject="fill_index")
    return _ir_core.create_op_call(block_ir_op("fill_index"), [out, start], {}, span or _span())


@op_impl("set_validshape")
def _build_set_validshape(self, tile: Expr, shape: Expr, *, span: Span) -> Expr:
    if self.is_tile_group(tile):
        group_var = tile

        # tile_group_meta is keyed by the handle expression itself, not by id():
        # an id() lookup never hits, silently collapsing the group to one tile.
        meta = self.tile_group_meta.get(group_var, (1, None))
        n_tiles = meta[0]
        tiles = self.lower_attr_access(group_var, "tiles", span)

        for i in range(n_tiles):
            tile_ir = _ir_core.GetItemExpr(tiles, ConstInt(i, DataType.INT64, span), span)
            vs_call = _ir_set_validshape(tile_ir, shape, span=span)
            self.builder.emit(_ir_core.EvalStmt(vs_call, span))

        return ConstInt(0, DataType.INT64, span)

    result = _ir_set_validshape(tile, shape, span=span)

    if hasattr(self, 'record_tile_valid_shape'):
        self.record_tile_valid_shape(tile, shape)
    return result


def _check_tile_memory_space(
    op_name: str, operand_name: str, expr: Expr, expected: MemorySpace, expected_desc: str
) -> None:
    mem = getattr(getattr(getattr(expr, "type", None), "memref", None), "memory_space_", None)
    if mem is not None and mem != expected:
        raise InvalidOperation(f"{op_name}: {operand_name} must be in {expected_desc}, got {mem.name}")


def _validate_acc_phase(op_name: str, phase: AccPhase | None) -> None:
    if phase is not None:
        require_enum(phase, AccPhase, subject=f"{op_name}: phase")


def _ir_matmul(dst: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, phase: AccPhase | None = None) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul", phase)
    _check_tile_memory_space("matmul", "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul", "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space("matmul", "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    _check_matmul_dtype("matmul", dst, lhs, rhs)
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(block_ir_op("matmul"), [dst, lhs, rhs], kwargs, actual_span)


@op_impl("matmul_acc")
def _ir_matmul_acc(self,
    dst: Expr, acc: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_acc", phase)
    _check_tile_memory_space("matmul_acc", "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul_acc", "acc_tile", acc, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul_acc", "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space("matmul_acc", "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    _check_matmul_dtype("matmul_acc", dst, lhs, rhs, acc=acc)
    if not _ir_core.structural_equal(unwrap_type(dst.type), unwrap_type(acc.type), enable_auto_mapping=False):
        raise InvalidType("matmul_acc: acc_tile type must match dst_tile type")
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(block_ir_op("matmul_acc"), [dst, acc, lhs, rhs], kwargs, actual_span)


def _ir_matmul_bias(
    dst: Expr, lhs: Expr, rhs: Expr, bias: Expr, *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_bias", phase)
    _check_tile_memory_space("matmul_bias", "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space("matmul_bias", "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space("matmul_bias", "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    _check_tile_memory_space("matmul_bias", "bias_tile", bias, MemorySpace.Bias, "L0B (Bias)")
    _check_matmul_dtype("matmul_bias", dst, lhs, rhs, bias=bias)
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(block_ir_op("matmul_bias"), [dst, lhs, rhs, bias], kwargs, actual_span)


@op_impl("matmul")
def _build_matmul(
    self, dst_tile: Expr, lhs_tile: Expr, rhs_tile: Expr, bias_tile: Expr | None = None, /,
    *, phase: AccPhase | None = None, span: Span,
) -> Expr:
    if bias_tile is not None:
        return _ir_matmul_bias(dst_tile, lhs_tile, rhs_tile, bias_tile, phase=phase, span=span)
    return _ir_matmul(dst_tile, lhs_tile, rhs_tile, phase=phase, span=span)


_MX_FP8_DTYPES = (DataType.FP8E4M3FN, DataType.FP8E5M2)
_MX_FP4_DTYPES = (DataType.FP4E2M1, DataType.FP4E1M2)
_MX_GROUP_SIZE = 32
_MX_SCALE_ADDR_SHIFT = 4


def _check_mx_scale_tile(
    op_name: str,
    scale: Expr | None,
    data: Expr,
    *,
    is_left: bool,
) -> None:
    """Validate an MX scale tile and its shape/address relationship to the paired data tile."""
    if is_left:
        scale_name, data_name, group_axis = "scale_a", "lhs_tile", 1
        expected_space, expected_desc = MemorySpace.ScaleLeft, "L0A (ScaleLeft)"
    else:
        scale_name, data_name, group_axis = "scale_b", "rhs_tile", 0
        expected_space, expected_desc = MemorySpace.ScaleRight, "L0B (ScaleRight)"

    if scale is None:
        raise InvalidArgument(f"{op_name}: {scale_name} is required")
    scale_type = getattr(scale, "type", None)
    if not isinstance(scale_type, _IRTileType):
        raise InvalidType(f"{op_name}: {scale_name} must be a Tile")
    if scale_type.dtype != DataType.FP8E8M0:
        raise InvalidType(f"{op_name}: {scale_name} must use FP8E8M0 dtype, got {scale_type.dtype}")
    _check_tile_memory_space(op_name, scale_name, scale, expected_space, expected_desc)

    data_type = data.type
    data_shape = _tile_shape_ints(data_type)
    scale_shape = _tile_shape_ints(scale_type)
    if data_shape is not None and scale_shape is not None:
        expected_shape = data_shape.copy()
        expected_shape[group_axis] //= _MX_GROUP_SIZE
        if scale_shape != expected_shape:
            raise InvalidType(
                f"{op_name}: {scale_name} shape must match {data_name} MX groups, "
                f"expected {expected_shape} for {data_name} shape {data_shape}, got {scale_shape}."
            )

    # 3510 mad_mx receives only the L0A/L0B data pointers and derives the
    # corresponding L0AMX/L0BMX address by shifting right four bits.  Skip
    # auto-allocated tiles or non-constant addresses, which cannot be proven here.
    data_memref = data_type.memref
    scale_memref = scale_type.memref
    if data_memref is None or scale_memref is None:
        return

    data_addr = data_memref.addr
    scale_addr = scale_memref.addr
    if not isinstance(data_addr, ConstInt) or not isinstance(scale_addr, ConstInt):
        return

    expected_scale_addr = data_addr.value >> _MX_SCALE_ADDR_SHIFT
    if scale_addr.value != expected_scale_addr:
        raise InvalidVal(
            f"{op_name}: {scale_name} address must equal {data_name} address >> {_MX_SCALE_ADDR_SHIFT}, "
            f"got {data_name}=0x{data_addr.value:X}, expected {scale_name}=0x{expected_scale_addr:X}, "
            f"actual {scale_name}=0x{scale_addr.value:X}."
        )


def _check_mx_operands(
    op_name: str,
    dst: Expr,
    lhs: Expr,
    rhs: Expr,
    scale_a: Expr,
    scale_b: Expr,
    acc: Expr | None = None,
    bias: Expr | None = None,
) -> None:
    """Run the common declared-type and shape checks for MX matmul variants."""
    _check_tile_memory_space(op_name, "dst_tile", dst, MemorySpace.Acc, "L0C (Acc)")
    _check_tile_memory_space(op_name, "lhs_tile", lhs, MemorySpace.Left, "L0A (Left)")
    _check_tile_memory_space(op_name, "rhs_tile", rhs, MemorySpace.Right, "L0B (Right)")
    if acc is not None:
        _check_tile_memory_space(op_name, "acc_tile", acc, MemorySpace.Acc, "L0C (Acc)")
    if bias is not None:
        _check_tile_memory_space(op_name, "bias_tile", bias, MemorySpace.Bias, "BiasTable (Bias)")

    a_dtype, b_dtype, dst_dtype = lhs.type.dtype, rhs.type.dtype, dst.type.dtype
    valid_input_dtypes = (a_dtype in _MX_FP8_DTYPES and b_dtype in _MX_FP8_DTYPES) or (
        a_dtype in _MX_FP4_DTYPES and b_dtype in _MX_FP4_DTYPES
    )
    if not valid_input_dtypes or dst_dtype != DataType.FP32:
        raise InvalidType(
            f"{op_name}: (lhs,rhs) must be FP8/FP4 combo and dst FP32, "
            f"got ({a_dtype},{b_dtype},{dst_dtype})."
        )

    lhs_shape = _tile_shape_ints(lhs.type)
    rhs_shape = _tile_shape_ints(rhs.type)
    lhs_k = lhs_shape[1] if lhs_shape is not None else None
    rhs_k = rhs_shape[0] if rhs_shape is not None else None

    if bias is not None:
        bias_shape = _tile_shape_ints(bias.type)
        if rhs_shape is not None and bias_shape is not None and bias_shape[1] != rhs_shape[1]:
            raise InvalidShape(
                f"{op_name}: bias_tile columns must match rhs_tile N dimension, "
                f"expected {rhs_shape[1]}, got {bias_shape[1]}."
            )

    for name, k_value in (("lhs", lhs_k), ("rhs", rhs_k)):
        if k_value is not None and k_value % 64 != 0:
            raise InvalidShape(
                f"{op_name}: K dimension must be a multiple of 64 for MX matmul, got {name} K={k_value}."
            )

    if acc is not None:
        if acc.type.dtype != DataType.FP32:
            raise InvalidType(f"{op_name}: acc_tile must use FP32 dtype, got {acc.type.dtype}.")
        if not _ir_core.structural_equal(
            unwrap_type(dst.type), unwrap_type(acc.type), enable_auto_mapping=False
        ):
            raise InvalidType(f"{op_name}: acc_tile type must match dst_tile type")

    if bias is not None and bias.type.dtype != DataType.FP32:
        raise InvalidType(f"{op_name}: bias_tile must use FP32 dtype, got {bias.type.dtype}.")

    _check_mx_scale_tile(op_name, scale_a, lhs, is_left=True)
    _check_mx_scale_tile(op_name, scale_b, rhs, is_left=False)


def _ir_matmul_mx(
    dst: Expr, lhs: Expr, rhs: Expr, scale_a: Expr, scale_b: Expr,
    bias_tile: Expr | None = None, *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_mx", phase)
    _check_mx_operands("matmul_mx", dst, lhs, rhs, scale_a, scale_b, bias=bias_tile)
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    operands = [dst, lhs, rhs, scale_a, scale_b]
    if bias_tile is not None:
        operands.append(bias_tile)
    return _ir_core.create_op_call(block_ir_op("matmul_mx"), operands, kwargs, actual_span)


def _ir_matmul_mx_acc(
    dst: Expr, acc: Expr, lhs: Expr, rhs: Expr, scale_a: Expr, scale_b: Expr,
    *, span: Span | None = None, phase: AccPhase | None = None
) -> Expr:
    actual_span = span or _span()
    _validate_acc_phase("matmul_mx_acc", phase)
    _check_mx_operands("matmul_mx_acc", dst, lhs, rhs, scale_a, scale_b, acc)
    kwargs: dict[str, Any] = {}
    if phase is not None:
        kwargs["phase"] = phase
    return _ir_core.create_op_call(
        block_ir_op("matmul_mx_acc"), [dst, acc, lhs, rhs, scale_a, scale_b], kwargs, actual_span
    )


@op_impl("set_saturation_flag")
def _ir_set_saturation_flag(self, mode: Any, enable: Any, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    enable = require_const_bool(enable, subject="'enable'")
    kwargs = {"mode": int(mode), "enable": enable}
    return _ir_core.create_op_call("set_saturation_flag", [], kwargs, actual_span)


@op_impl("get_saturation_flag")
def _ir_get_saturation_flag(self, mode: Any, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    kwargs = {"mode": int(mode)}
    return _ir_core.create_op_call("get_saturation_flag", [], kwargs, actual_span)


@op_impl("set_ctrl_spr")
def _ir_set_ctrl_spr(self, start_bit: Expr, end_bit: Expr, value: Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    args = [start_bit, end_bit, value]
    return _ir_core.create_op_call("set_ctrl_spr", args, {}, actual_span)


@op_impl("get_ctrl_spr")
def _ir_get_ctrl_spr(self, start_bit: Expr, end_bit: Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    args = [start_bit, end_bit]
    return _ir_core.create_op_call("get_ctrl_spr", args, {}, actual_span)


@op_impl("reset_ctrl_spr")
def _ir_reset_ctrl_spr(self, start_bit: Expr, end_bit: Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    args = [start_bit, end_bit]
    return _ir_core.create_op_call("reset_ctrl_spr", args, {}, actual_span)


@op_impl("load")
def _build_load(
    self, dst_tile: Expr, src_tensor: Expr, /, offsets: Expr,
    *, order: Expr | None = None, l2_cache_mode: TLoadL2Hint | None = None, span: Span,
) -> Expr:
    order = _prepare_order(order)
    return _ir_load(dst_tile, src_tensor, offsets, order=order, l2_cache_mode=l2_cache_mode, span=span)


@op_impl("load_tile")
def _build_load_tile(
    self, dst_tile: Expr, src_tensor: Expr, /, tile_offsets: Expr,
    *, order: Expr | None = None, l2_cache_mode: TLoadL2Hint | None = None, span: Span,
) -> Expr:
    order = _prepare_order(order)
    return _ir_load_tile(dst_tile, src_tensor, tile_offsets, order=order, l2_cache_mode=l2_cache_mode, span=span)


@op_impl("store")
def _build_store(
    self, dst_tensor: Expr, src_tile: Expr, /, offsets: Expr, *,
    relu_pre_mode: ReluPreMode | None = None, scale: Expr | None = None, order: Expr | None = None,
    atomic: AtomicType = AtomicType.AtomicNone, phase: STPhase | None = None,
    l2_cache_mode: TStoreL2Hint | None = None, span: Span,
) -> Expr:
    scale = _prepare_scale(scale)
    order = _prepare_order(order)
    return _ir_store(
        dst_tensor, src_tile, offsets, relu_pre_mode=relu_pre_mode, scale=scale,
        order=order, atomic=atomic, phase=phase, l2_cache_mode=l2_cache_mode, span=span,
    )


@op_impl("store_tile")
def _build_store_tile(
    self, dst_tensor: Expr, src_tile: Expr, /, tile_offsets: Expr, *,
    relu_pre_mode: ReluPreMode | None = None, scale: Expr | None = None, order: Expr | None = None,
    atomic: AtomicType = AtomicType.AtomicNone, phase: STPhase | None = None,
    l2_cache_mode: TStoreL2Hint | None = None, span: Span,
) -> Expr:
    scale = _prepare_scale(scale)
    order = _prepare_order(order)
    return _ir_store_tile(
        dst_tensor, src_tile, tile_offsets, relu_pre_mode=relu_pre_mode, scale=scale,
        order=order, atomic=atomic, phase=phase, l2_cache_mode=l2_cache_mode, span=span,
    )


@op_impl("dynamic_input_ptr")
def _build_dynamic_input_ptr(self, x: Expr, i: Expr, /, *, span: Span) -> Expr:
    return _ir_core.create_op_call("dynamic_input_ptr", [x, i], {}, span)


@op_impl("dynamic_input_dim")
def _build_dynamic_input_dim(self, x: Expr, i: Expr, axis: Expr, /, *, span: Span) -> Expr:
    return _ir_core.create_op_call("dynamic_input_dim", [x, i, axis], {}, span)


@op_impl("move")
def _build_move(
    self, dst_tile: Expr, src_tile: Expr, /, offset: Expr | None = None, *,
    acc_to_vec_mode: AccToVecMode | None = None, relu_pre_mode: ReluPreMode | None = None,
    scale: Expr | None = None, phase: STPhase | None = None, span: Span,
) -> Expr:
    scale = _prepare_scale(scale)
    return _ir_move(
        dst_tile, src_tile, offset, acc_to_vec_mode=acc_to_vec_mode, relu_pre_mode=relu_pre_mode,
        scale=scale, phase=phase, span=span,
    )


@op_impl("insert")
def _build_insert(
    self, dst_tile: Expr, src_tile: Expr, /, offset: Expr, *,
    relu_pre_mode: ReluPreMode | None = None, scale: Expr | None = None,
    phase: STPhase | None = None, span: Span,
) -> Expr:
    scale = _prepare_scale(scale)
    return _ir_insert(dst_tile, src_tile, offset, relu_pre_mode=relu_pre_mode, scale=scale, phase=phase, span=span)


@op_impl("getval")
def _build_getval(self, container: Expr, offset: Expr, *, span: Span) -> Expr:
    return _ir_getval(container, offset, span=span)


@op_impl("setval")
def _build_setval(self, container: Expr, offset: Expr, value: Expr, *, span: Span) -> Expr:
    return _ir_setval(container, offset, value, span=span)


@op_impl("matmul_mx")
def _build_matmul_mx(
    self, dst_tile: Expr, lhs_tile: Expr, rhs_tile: Expr, /, scale_a: Expr, scale_b: Expr,
    bias_tile: Expr | None = None, *, phase: AccPhase | None = None, span: Span,
) -> Expr:
    result = _ir_matmul_mx(
        dst_tile, lhs_tile, rhs_tile, scale_a, scale_b, bias_tile, phase=phase, span=span,
    )
    self._check_mx_plane_slots("matmul_mx", (dst_tile, lhs_tile, rhs_tile, scale_a, scale_b))
    return result


@op_impl("matmul_mx_acc")
def _build_matmul_mx_acc(
    self, dst_tile: Expr, acc_tile: Expr, lhs_tile: Expr, rhs_tile: Expr, /, scale_a: Expr, scale_b: Expr,
    *, phase: AccPhase | None = None, span: Span,
) -> Expr:
    result = _ir_matmul_mx_acc(dst_tile, acc_tile, lhs_tile, rhs_tile, scale_a, scale_b, phase=phase, span=span)
    self._check_mx_plane_slots("matmul_mx_acc", (dst_tile, acc_tile, lhs_tile, rhs_tile, scale_a, scale_b))
    return result
