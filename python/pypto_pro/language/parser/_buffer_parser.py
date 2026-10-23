# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""make_tile_group parsing helpers for ASTParser.

``pl.make_tile_group(type=, addrs=, mutex_ids=, depth=)`` declares a rotating group of
tiles inside a kernel. The parser lowers it to an IR named-tuple handle with
fields ``tiles`` (tuple of tile vars), one or more ``mutex_ids`` column tuples,
and a mutable ``cursor`` struct. The handle's ``next()/current()/previous()`` methods
and its ``group[i]`` subscript return a bare tile (not a slot); the tile <->
mutex_id mapping, when configured, is kept as parser metadata
(``_tile_mutex_meta``) and consumed by auto_mutex, so the runtime never handles
mutex ids or manual lock/unlock.

``next()/current()/previous()`` walk the rotating cursor; ``group[i]`` addresses
a slot directly and leaves the cursor alone, which is what lets a kernel keep
several slots in flight at once (``load(g[k + 1])`` while ``matmul(g[k])``).
Both produce the same metadata, so auto_mutex cannot tell them apart.
"""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType
from pypto_pro.ir._utils import _is_int, _to_make_tuple
from pypto_pro.ir.op._op_registry import op_impl
from pypto_pro.ir.op.block_ops import TileType as _TileType
from pypto_pro.ir.op.block_ops import _ir_reinterpret, _static_shape_ints
from pypto_pro.ir.op.block_ops import make_tile_expr as _make_tile_expr
from pypto_pro.ir.op.block_ops import tile_slot_size as _tile_slot_size
from pypto_pro.ir.op.system_ops import MAX_MUTEX_ID

from ..._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    MutexMismatch,
    OutOfRange,
    PyptoProError,
    _warn_located,
    message_of,
)
from ._tuple_type_registry import TupleTypeKind
from .diagnostics import check_in_range

_TILE_GROUP_TYPE_NAME = "tile_group"


def _is_int_sequence(value) -> bool:
    """Whether *value* is a non-empty tuple of plain integers."""
    return isinstance(value, tuple) and bool(value) and all(_is_int(e) for e in value)


class BufferParserMixin:
    """make_tile_group DSL composite: parse pl.make_tile_group(...) into an IR named tuple."""

    # --- helpers --------------------------------------------------------------

    def warn_manual_mutex_aliases(self) -> None:
        """Check manual aliases after native AUTO validation, reusing group metadata."""
        if not self._auto_mutex:
            return
        seen = set()
        slots = []
        for group, (_, mutex_ids, memory) in self.tile_group_meta.items():
            if mutex_ids is None:
                continue
            tiles = group.type.types[self.named_fields(group).index("tiles")].types
            for tile, row in zip(tiles, mutex_ids.elements):
                if not all(isinstance(value, ir.ConstInt) for value in row.elements):
                    continue
                memref = tile.memref
                if memref is None or memref in seen or not isinstance(memref.addr, ir.ConstInt):
                    continue
                seen.add(memref)
                start = int(memref.addr.value)
                end = start + int(memref.size)
                ids = {int(value.value) for value in row.elements}
                for other_memory, lo, hi, other_ids in slots:
                    if memory == other_memory and start < hi and lo < end and not ids.intersection(other_ids):
                        _warn_located(
                            f"Tile buffers in {memory.name} overlap byte ranges [{start:#x}, {end:#x}) "
                            f"(mutex IDs {sorted(ids)}) and [{lo:#x}, {hi:#x}) (mutex IDs {sorted(other_ids)}) "
                            "without a shared mutex ID; auto_mutex will not order their accesses to the same bytes",
                            MutexMismatch, group.span,
                        )
                slots.append((memory, start, end, ids))

    @staticmethod
    def _normalize_mutex_ids(mutex_ids, span: ir.Span) -> tuple[tuple[int, ...], ...]:
        """Normalize mutex IDs to one fixed-size tuple per tile slot."""
        normalized_mutex_ids = []
        for mutex_id in mutex_ids:
            if isinstance(mutex_id, (list, tuple)):
                tile_mutex_ids = tuple(mutex_id)
                if not tile_mutex_ids:
                    raise InvalidVal(
                        "make_tile_group() mutex ID sequence for a tile must not be empty", span=span,
                        parser_retry=True,
                    )
            else:
                tile_mutex_ids = (mutex_id,)

            for value in tile_mutex_ids:
                if not _is_int(value):
                    raise InvalidType(f"mutex_ids must be ints, got {value!r}", span=span, parser_retry=True)
                check_in_range(value, 0, MAX_MUTEX_ID, subject="mutex_ids element", span=span)
            if len(set(tile_mutex_ids)) != len(tile_mutex_ids):
                raise InvalidVal(
                    f"make_tile_group() mutex IDs for one tile must not contain duplicates: {tile_mutex_ids}",
                    span=span,
                    parser_retry=True,
                )
            normalized_mutex_ids.append(tile_mutex_ids)

        expected_mutex_id_count = len(normalized_mutex_ids[0])
        for tile_mutex_ids in normalized_mutex_ids[1:]:
            if len(tile_mutex_ids) != expected_mutex_id_count:
                raise InvalidArgument(
                    f"all tiles in a tile group must have the same mutex ID count: "
                    f"{expected_mutex_id_count} and {len(tile_mutex_ids)}",
                    span=span,
                    parser_retry=True,
                )
        return tuple(normalized_mutex_ids)

    @staticmethod
    def _validate_depth(depth, span: ir.Span) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise InvalidType(
                f"make_tile_group() depth must be a positive compile-time integer, got {depth!r}",
                span=span,
                parser_retry=True,
            )

    def _parse_mutex_ids_and_depth(
        self,
        raw_mutex_ids,
        depth: int | None,
        mutex_ids_span: ir.Span,
        depth_span: ir.Span,
        span: ir.Span,
    ) -> tuple[ir.MakeTuple | None, int]:
        """Validate and lower one tile group's mutex IDs, inferring depth when omitted."""
        if depth is not None:
            self._validate_depth(depth, depth_span)

        if isinstance(raw_mutex_ids, str):
            if raw_mutex_ids != "auto":
                raise InvalidArgument(
                    'make_tile_group() string mutex_ids only supports "auto", '
                    f"got {raw_mutex_ids!r}",
                    span=mutex_ids_span,
                    parser_retry=True,
                )
            if not self._auto_mutex:
                raise InvalidOperation(
                    'mutex_ids="auto" requires @pl.jit(auto_mutex=True)',
                    span=mutex_ids_span,
                    parser_retry=True,
                )
            if depth is None:
                raise InvalidArgument(
                    'make_tile_group() depth is required when mutex_ids is "auto", None, or empty',
                    span=mutex_ids_span,
                    parser_retry=True,
                )
            auto_mutex_ids = []
            for _ in range(depth):
                placeholder = ir.Var(
                    f"__pypto_auto_mutex_id_{self._auto_mutex_placeholder_counter}",
                    ir.ScalarType(DataType.INT64),
                    span,
                )
                self._auto_mutex_placeholder_counter += 1
                auto_mutex_ids.append(_to_make_tuple([placeholder], span))
            return _to_make_tuple(auto_mutex_ids, span), depth

        if raw_mutex_ids is None or (
            isinstance(raw_mutex_ids, (list, tuple)) and not raw_mutex_ids
        ):
            if depth is None:
                raise InvalidArgument(
                    'make_tile_group() depth is required when mutex_ids is "auto", None, or empty',
                    span=mutex_ids_span,
                    parser_retry=True,
                )
            return None, depth

        if not isinstance(raw_mutex_ids, (list, tuple)):
            raise InvalidType(
                'make_tile_group() mutex_ids must be "auto", a list, tuple, or None',
                span=mutex_ids_span,
                parser_retry=True,
            )

        normalized_mutex_ids = self._normalize_mutex_ids(raw_mutex_ids, mutex_ids_span)
        if depth is None:
            depth = len(normalized_mutex_ids)
        elif len(normalized_mutex_ids) != depth:
            raise InvalidShape(
                f"make_tile_group() mutex_ids length {len(normalized_mutex_ids)} must equal depth {depth}",
                span=mutex_ids_span,
                parser_retry=True,
            )
        return (
            _to_make_tuple(
                [_to_make_tuple(tile_mutex_ids, span) for tile_mutex_ids in normalized_mutex_ids],
                span,
            ),
            depth,
        )

    @staticmethod
    def _tile_type_slot_size(tile_type: _TileType) -> int:
        """Per-tile byte size derived from the TileType static shape and dtype.

        Same helper make_tile() uses for its byte span, so a group slot is
        exactly as wide as a standalone tile of the same TileType.
        """
        try:
            return _tile_slot_size(tile_type.shape, tile_type.dtype)
        except PyptoProError as exc:
            raise InvalidVal(f"make_tile_group() {message_of(exc)}", parser_retry=True) from exc

    def is_tile_group_type(self, value_type: ir.Type) -> bool:
        if not isinstance(value_type, ir.TupleType):
            return False
        info = self.classify_tuple_type(value_type)
        return info.kind == TupleTypeKind.NAMED_TUPLE and info.name == _TILE_GROUP_TYPE_NAME

    def is_tile_group(self, expr) -> bool:
        return isinstance(expr, ir.Expr) and self.is_tile_group_type(expr.type)

    def _contains_tile_group_type(self, value_type: ir.Type) -> bool:
        if self.is_tile_group_type(value_type):
            return True
        return isinstance(value_type, ir.TupleType) and any(
            self._contains_tile_group_type(element_type) for element_type in value_type.types
        )

    # --- factory --------------------------------------------------------------
    @op_impl("make_tile_group")
    def _parse_make_tile_group(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if call.args:
            raise InvalidArgument(
                "pl.make_tile_group() takes keyword args only "
                "(type=, addrs=, mutex_ids=, depth=)",
                span=span,
            )
        kw = {k.arg: k for k in call.keywords if k.arg is not None}
        # bwd_ids/fwd_ids are accepted for backward compatibility with existing
        # operator repositories and ignored, as in the legacy parser behavior.
        unknown = sorted(set(kw) - {"type", "addrs", "mutex_ids", "depth", "bwd_ids", "fwd_ids"})
        if unknown:
            raise InvalidArgument(
                f"pl.make_tile_group() got unexpected keyword argument(s) {unknown}; "
                f"supported keywords are type, addrs, mutex_ids, depth, bwd_ids, fwd_ids",
                span=self.span_tracker.keyword_span(kw, unknown[0], span),
                parser_retry=True,
            )
        for required in ("type", "addrs"):
            if required not in kw:
                raise InvalidArgument(
                    f"pl.make_tile_group() missing required keyword '{required}'",
                    span=span,
                    parser_retry=True,
                )

        tile_type = self.parse_expression(kw["type"].value)
        if not isinstance(tile_type, _TileType):
            raise InvalidType(
                "pl.make_tile_group() 'type' must be a pl.TileType",
                span=self.span_tracker.keyword_span(kw, "type", span),
                parser_retry=True,
            )
        group_target = (
            ir.SectionKind.Vector
            if tile_type.target_memory == ir.MemorySpace.Vec
            else ir.SectionKind.Cube
        )

        mutex_ids_span = self.span_tracker.keyword_span(kw, "mutex_ids", span)
        raw_mutex_ids = self.expr_evaluator.eval_expr(kw["mutex_ids"].value) if "mutex_ids" in kw else None
        depth_span = self.span_tracker.keyword_span(kw, "depth", span)
        depth = self.expr_evaluator.eval_expr(kw["depth"].value) if "depth" in kw else None

        if group_target == self.target:
            mutex_id_irs, depth = self._parse_mutex_ids_and_depth(
                raw_mutex_ids,
                depth,
                mutex_ids_span,
                depth_span,
                span,
            )
        else:
            mutex_id_irs = None
            if depth is not None:
                self._validate_depth(depth, depth_span)
                if isinstance(raw_mutex_ids, (list, tuple)) and raw_mutex_ids and len(raw_mutex_ids) != depth:
                    raise InvalidShape(
                        f"make_tile_group() mutex_ids length {len(raw_mutex_ids)} must equal depth {depth}",
                        span=mutex_ids_span,
                        parser_retry=True,
                    )
            elif isinstance(raw_mutex_ids, (list, tuple)) and raw_mutex_ids:
                depth = len(raw_mutex_ids)
            else:
                raise InvalidArgument(
                    'make_tile_group() depth is required when mutex_ids is "auto", None, or empty',
                    span=mutex_ids_span,
                    parser_retry=True,
                )

        slot_size = self._tile_type_slot_size(tile_type)
        # addrs: a single base address -> contiguous tiles (base + i*slot_size);
        # a list -> one explicit address per tile (non-contiguous).
        addrs_node = kw["addrs"].value
        addrs = self.require_const_value(
            self.parse_expression(addrs_node),
            addrs_node,
            key="addrs",
            expects="integer, or list of integers",
            check=lambda value: _is_int(value) or _is_int_sequence(value),
            hint="addrs is fixed while parsing; pass a literal, or a variable bound to "
            "literals/constants — not a runtime value such as a tensor shape or loop index",
        )
        for addr in (list(addrs) if isinstance(addrs, (list, tuple)) else [addrs]):
            if addr < 0:
                raise InvalidArgument(
                    f"make_tile_group() addrs must be non-negative integers, got {addr!r}",
                    span=self.span_tracker.keyword_span(kw, "addrs", span),
                    parser_retry=True,
                )
        if isinstance(addrs, (list, tuple)):
            tile_addrs = list(addrs)
            if len(tile_addrs) != depth:
                raise InvalidShape(
                    f"make_tile_group() addrs length {len(tile_addrs)} must equal "
                    f"depth/len(mutex_ids) {depth}",
                    span=self.span_tracker.keyword_span(kw, "addrs", span),
                    parser_retry=True,
                )
        else:
            tile_addrs = [addrs + i * slot_size for i in range(depth)]

        return self._build_tile_group_ir(
            tile_type,
            tile_addrs,
            depth,
            mutex_id_irs,
            slot_size,
            span,
        )

    def _build_tile_group_ir(
        self,
        tile_type,
        tile_addrs,
        depth,
        mutex_id_irs,
        slot_size,
        span: ir.Span,
    ) -> ir.Expr:
        """Build the IR named-tuple handle for a tile group.

        Multi-tile groups carry a rotating ``cursor`` struct ({tiles, mutex_ids,
        cursor}). It starts at the last slot so that next(), which advances before
        selecting, returns slot 0 on its first call. Single-tile groups skip it
        ({tiles, mutex_ids}) since the only tile is statically determined.

        Writes group metadata directly to ``tile_group_meta`` keyed by the
        returned expression; ``_transfer_tile_sync_metadata`` re-keys it to
        the assigned variable after SSA binding creation.
        """
        var_name = self.current_target_name
        tile_vars = []
        for i, addr in enumerate(tile_addrs):
            t = _make_tile_expr(
                shape=tile_type.shape,
                dtype=tile_type.dtype,
                target_memory=tile_type.target_memory,
                layout=tile_type.layout,
                fractal=tile_type.fractal,
                pad=tile_type.pad,
                compact=tile_type.compact,
                valid_shape=tile_type.valid_shape,
                addr=addr,
                size=slot_size,
                span=span,
            )
            self._record_tile_high_water(t)
            tile_vars.append(self.builder.let(f"_tg_{var_name}_tiles_{i}", t, span=span))

        if self._auto_mutex and mutex_id_irs is not None:
            self._mutex_id_manager.collect_group(tile_vars, mutex_id_irs, var_name)

        tiles_tuple = self.builder.let(f"_tg_{var_name}_tiles", ir.MakeTuple(tile_vars, span), span=span)
        mutex_rows = (
            [tile_mutex_ids.elements for tile_mutex_ids in mutex_id_irs.elements]
            if mutex_id_irs is not None
            else []
        )
        mutex_columns = tuple(zip(*mutex_rows)) if mutex_rows else ((),)
        mut_tuples = []
        mut_fields = []
        for column, values in enumerate(mutex_columns):
            suffix = "" if column == 0 else f"_{column}"
            mut_tuples.append(
                self.builder.let(
                    f"_tg_{var_name}_mutex_ids{suffix}",
                    _to_make_tuple(values, span),
                    span=span,
                )
            )
            mut_fields.append(f"mutex_ids{suffix}")
        if depth == 1:
            result = self.make_named_tuple(
                [tiles_tuple, *mut_tuples],
                ["tiles", *mut_fields],
                span,
                name=_TILE_GROUP_TYPE_NAME,
            )
            self.tile_group_meta[result] = (depth, mutex_id_irs, tile_type.target_memory)
            return result
        cursor_call = ir.create_op_call(
            "struct.create",
            [ir.ConstInt(depth - 1, DataType.INT64, span)],
            {"name": "_TileGroupCursor", "fields": ["cursor"]},
            span,
        )
        self.register_struct_type(cursor_call, "_TileGroupCursor", ["cursor"])
        cursor = self.builder.let(f"_tg_{var_name}_cursor", cursor_call, span=span)
        result = self.make_named_tuple(
            [tiles_tuple, *mut_tuples, cursor],
            ["tiles", *mut_fields, "cursor"],
            span,
            name=_TILE_GROUP_TYPE_NAME,
        )
        self.tile_group_meta[result] = (depth, mutex_id_irs, tile_type.target_memory)
        return result

    def _build_reinterpreted_group(self, group, kwargs: dict, span: ir.Span) -> ir.Expr:
        """Rebuild a tile_group handle after a reinterpretation.

        Each slot is reinterpreted through the single-tile path (``_ir_reinterpret``):
        every slot is re-allocated at the ORIGINAL address and size (SameAllocation
        semantics keep the "same buffer" identity) and runs the same validation as
        standalone tiles (compile-time shape, element alignment, footprint bound).

        depth and group mutex_ids are inherited verbatim (mutex_id
        inheritance); the mutex columns and the rotating cursor are reused from
        the original group handle. The group handle shape mirrors what
        ``_build_tile_group_ir`` produces (tiles tuple, mutex_id columns,
        optional cursor), keeping ``next()/current()/previous()`` and
        ``tile_group_meta`` fully functional.
        """

        meta = self.tile_group_meta.get(group)
        if meta is None:
            raise InvalidType(
                "tile group handle has no group metadata; it must come directly from pl.make_tile_group()",
                span=span,
                hint="Pass the handle itself rather than a copy stored in a struct or container",
                parser_retry=True,
            )
        depth, mutex_ids, _memory = meta
        tiles = self.lower_attr_access(group, "tiles", span)

        slot0 = ir.GetItemExpr(tiles, ir.ConstInt(0, DataType.INT64, span), span)
        t = getattr(slot0, "type", None)
        memref = getattr(t, "memref", None)
        try:
            old_shape = _static_shape_ints(t.shape, "reinterpret: source")
        except PyptoProError as exc:
            raise InvalidType(message_of(exc), span=span, parser_retry=True) from exc

        # Override args are validated ONCE at the DSL entry (_parse_reinterpret);
        # the group dispatch never re-validates them. Per-slot validation
        # (alignment / footprint / compile-time shape) happens inside
        # _ir_reinterpret for every slot.
        new_shape = list(kwargs["shape"]) if kwargs.get("shape") is not None else old_shape

        # ① reinterpret every slot through the single-tile path: same address &
        #    size (fresh MemRef inside _ir_reinterpret) and identical validation.
        var_name = self.current_target_name
        tile_vars = []
        for i in range(depth):
            slot = ir.GetItemExpr(tiles, ir.ConstInt(i, DataType.INT64, span), span)
            slot_tile = _ir_reinterpret(slot, shape=new_shape if kwargs.get("shape") is not None else None,
                                        dtype=kwargs.get("dtype"), layout=kwargs.get("layout"), span=span)
            tile_vars.append(self.builder.let(f"_tg_{var_name}_tiles_{i}", slot_tile, span=span))
        tiles_tuple = self.builder.let(f"_tg_{var_name}_tiles", ir.MakeTuple(tile_vars, span), span=span)

        # ② mutex columns: reuse the original group's own column expressions.
        n_columns = len(mutex_ids.elements[0].elements) if mutex_ids is not None else 0
        mut_tuples = []
        mut_fields = []
        for column in range(n_columns):
            suffix = "" if column == 0 else f"_{column}"
            mut_tuples.append(self.lower_attr_access(group, f"mutex_ids{suffix}", span))
            mut_fields.append(f"mutex_ids{suffix}")

        # ③ cursor: reused from the original handle so the rotation position
        #    (the original group keeps its own independent cursor) is preserved.
        if depth == 1:
            result = self.make_named_tuple(
                [tiles_tuple, *mut_tuples],
                ["tiles", *mut_fields],
                span,
                name=_TILE_GROUP_TYPE_NAME,
            )
            self.tile_group_meta[result] = (depth, mutex_ids, memref.memory_space)
            return result
        cursor = self.lower_attr_access(group, "cursor", span)
        result = self.make_named_tuple(
            [tiles_tuple, *mut_tuples, cursor],
            ["tiles", *mut_fields, "cursor"],
            span,
            name=_TILE_GROUP_TYPE_NAME,
        )
        self.tile_group_meta[result] = (depth, mutex_ids, memref.memory_space)
        return result

    # --- accessors ------------------------------------------------------------

    def _emit_cursor_advance(self, cursor_struct, new_val, span: ir.Span) -> None:
        self.builder.emit(
            ir.EvalStmt(ir.create_op_call("struct.set", [cursor_struct, new_val], {"field": "cursor"}, span), span)
        )

    def _group_meta(self, group_var, span: ir.Span) -> tuple:
        """Return the group depth, mutex IDs, and candidate IDs."""
        meta = self.tile_group_meta.get(group_var)
        if meta is None:
            raise InvalidType(
                "tile group handle has no group metadata; it must come directly from "
                "pl.make_tile_group()",
                span=span,
                hint="Pass the handle itself rather than a copy stored in a struct or container",
                parser_retry=True,
            )
        depth, mutex_ids, _memory = meta
        candidate_ids = _to_make_tuple(
            list(
                dict.fromkeys(
                    value
                    for tile_mutex_ids in (mutex_ids.elements if mutex_ids is not None else ())
                    for value in tile_mutex_ids.elements
                )
            ),
            span,
        )
        return depth, mutex_ids, candidate_ids

    def _select_static_slot(
        self, tiles, slot: int, mutex_ids, candidate_ids, span: ir.Span
    ) -> ir.Expr:
        """Select tiles[slot] with a compile-time index.

        When mutex metadata exists, attach the slot's static id; it needs no
        ``% n`` and no dynamic lock at lock time.
        """
        tile_ir = ir.GetItemExpr(tiles, ir.ConstInt(slot, DataType.INT64, span), span)
        if mutex_ids is not None:
            tile_id_irs = mutex_ids.elements[slot]
            self._tile_mutex_meta[tile_ir] = (tile_id_irs, candidate_ids)
        return tile_ir

    def _select_dynamic_slot(
        self,
        group_var,
        tiles,
        raw_index,
        mutex_ids,
        candidate_ids,
        span: ir.Span,
    ):
        """Select a slot with one materialized runtime index.

        The supplied index is used unchanged. When mutex metadata exists, the
        Tile IDs are selected from the mutex-ID columns by ``idx``, so
        auto_mutex takes the dynamic lock path.
        """
        idx = self.builder.let(
            f"_bufidx_{self._tuple_idx_counter}",
            raw_index,
            span=span,
        )
        self._tuple_idx_counter += 1
        tile_ir = ir.GetItemExpr(tiles, idx, span)
        self._tile_slot_source[tile_ir] = tiles
        if mutex_ids is not None:
            tile_id_irs = []
            for column in range(len(mutex_ids.elements[0].elements)):
                suffix = "" if column == 0 else f"_{column}"
                mut = self.lower_attr_access(group_var, f"mutex_ids{suffix}", span)
                tile_id_irs.append(ir.GetItemExpr(mut, idx, span))
            self._tile_mutex_meta[tile_ir] = (_to_make_tuple(tile_id_irs, span), candidate_ids)
        return tile_ir

    def _lower_group_accessor(self, group_var, method_name: str, span: ir.Span):
        """Lower group.next()/current()/previous() to a bare tile.

        next() advances the cursor (+1) then returns the tile at the new index;
        current()/previous() leave the cursor unchanged. When configured, records
        the selected tile's mutex id via _tile_mutex_meta for auto_mutex.
        """
        depth, mutex_ids, candidate_ids = self._group_meta(group_var, span)
        tiles = self.lower_attr_access(group_var, "tiles", span)
        if depth == 1:
            # Single tile: no cursor or modulo; optional mutex metadata stays static.
            return self._select_static_slot(tiles, 0, mutex_ids, candidate_ids, span)
        cursor_struct = self.lower_attr_access(group_var, "cursor", span)
        cur_read = self.lower_attr_access(cursor_struct, "cursor", span)
        if method_name == "next":
            new_val = cur_read + ir.ConstInt(1, DataType.INT64, span)
            self._emit_cursor_advance(cursor_struct, new_val, span)
            raw = self.lower_attr_access(cursor_struct, "cursor", span)
        elif method_name == "current":
            raw = cur_read
        else:  # previous
            raw = cur_read + ir.ConstInt(depth - 1, DataType.INT64, span)
        wrapped = raw % ir.ConstInt(depth, DataType.INT64, span)
        return self._select_dynamic_slot(
            group_var,
            tiles,
            wrapped,
            mutex_ids,
            candidate_ids,
            span,
        )

    def lower_group_subscript(self, group_var, slice_node: ast.expr, span: ir.Span) -> ir.Expr:
        """Lower ``group[i]`` to a bare tile — direct slot access, cursor untouched.

        Emits the same optional mutex metadata as next()/current()/previous(). A
        constant index resolves the slot and, when configured, its mutex id at parse
        time; a runtime index is used unchanged and selects a dynamic mutex id only
        when the group has mutex metadata. The caller must keep it in ``[0, n)``.

        The cursor is deliberately neither read nor advanced: subscript is random
        access, so interleaving it with next()/previous() leaves the rotation
        sequence exactly where the last next() put it.
        """
        depth, mutex_ids, candidate_ids = self._group_meta(group_var, span)
        tiles = self.lower_attr_access(group_var, "tiles", span)
        index_expr = self._reject_non_index_subscript(slice_node, span)

        if isinstance(index_expr, ir.ConstInt):
            slot = int(index_expr.value)
            if not 0 <= slot < depth:
                raise OutOfRange(
                    f"tile group index {index_expr.value} is out of range for a "
                    f"{depth}-tile group",
                    span=span,
                    hint=f"Valid indices are in [0, {depth})",
                    parser_retry=True,
                )
            return self._select_static_slot(tiles, slot, mutex_ids, candidate_ids, span)

        return self._select_dynamic_slot(
            group_var,
            tiles,
            index_expr,
            mutex_ids,
            candidate_ids,
            span,
        )

    def _reject_non_index_subscript(self, slice_node: ast.expr, span: ir.Span) -> ir.Expr:
        """Parse and validate one integer tile-group index."""
        index_expr = self.parse_expression(slice_node)
        index_type = getattr(index_expr, "type", None)
        if not (isinstance(index_type, ir.ScalarType) and index_type.dtype.is_int()):
            actual_type = index_type if index_type is not None else type(index_expr).__name__
            raise InvalidType(
                f"tile group index must be an integer scalar, but got {actual_type}",
                span=span,
                hint="Use an integer scalar expression, for example g[0] or g[i]",
                parser_retry=True,
            )
        return index_expr
