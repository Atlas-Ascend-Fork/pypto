# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Expand L1 GEMM calls through the normal statement parser.

Generated moves, matmuls and loops use the existing parser's mutex, slot and SSA
handling. Emitting their IR directly would bypass that handling.
"""

from __future__ import annotations

import ast
import copy

from pypto.pypto_impl import ir

from ..._errors import current_span
from ...ir.op.block_ops import _check_tile_memory_space, _tile_layout, tile_slot_size
from ._argument_checker import declared_api
from .diagnostics import InvalidArgument, InvalidType, InvalidVal, NotSupported

_EXTENTS = ("m", "n", "k")
_POSITIONAL = {
    "gemm": ("acc", "l1a", "l1b"),
    "gemm_mx": ("acc", "l1a", "l1b", "l1a_scale", "l1b_scale"),
}
_STAGING = {
    "gemm": ("l0a", "l0b"),
    "gemm_mx": ("l0a", "l0b", "scale_a", "scale_b"),
}
_MX_GROUP = 32
_MX_K_UNIT = 64


def _name(value, ctx=None):
    return ast.Name(id=value, ctx=ctx or ast.Load())


def _const(value):
    return ast.Constant(value=value)


def _list(values):
    return ast.List(elts=values, ctx=ast.Load())


def _tile_type(declared):
    if isinstance(declared, ir.TileType):
        return declared
    for element in getattr(declared, "types", ()):
        found = _tile_type(element)
        if found is not None:
            return found
    return None


def lower_gemm(parser, stmt: ast.Expr) -> list[ast.stmt] | None:
    call = stmt.value
    if not isinstance(call, ast.Call):
        return None
    op = parser._extract_op_name(call.func)
    if op not in _POSITIONAL:
        return None
    while True:
        parser._gemm_counter += 1
        prefix = f"_pl_gemm{parser._gemm_counter}_"
        if not any(name.startswith(prefix) for name in parser._gemm_used_names):
            break
    return _GemmCall(parser, stmt, op).lower()


class _GemmCall:
    """State local to one expansion; inspecting types never re-evaluates operands."""

    def __init__(self, parser, stmt, op):
        self.parser = parser
        self.stmt = stmt
        self.call = stmt.value
        self.op = op
        self.api = f"pl.{op}()"
        self.mx = op == "gemm_mx"
        self.prefix = f"_pl_gemm{parser._gemm_counter}_"
        self.bound = {}
        self.refs = {}
        self.operands = {}
        self.keywords = {}
        if isinstance(self.call.func, ast.Attribute):
            self.base = self.call.func.value
        else:
            # A bare imported alias also needs constants such as ZN and DT_FP32.
            import pypto_pro.language as language

            alias = self.prefix + "api"
            parser.expr_evaluator.closure_vars[alias] = language
            self.base = _name(alias)

    def span(self, node):
        return self.parser.span_tracker.get_span(node)

    def attr(self, name):
        node = copy.deepcopy(self.base)
        for part in name.split("."):
            node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
        return node

    def call_op(self, op, args, **keywords):
        return ast.Call(
            func=self.attr(op), args=args,
            keywords=[ast.keyword(arg=k, value=v) for k, v in keywords.items()],
        )

    def emit(self, op, args, **keywords):
        return ast.copy_location(ast.Expr(value=self.call_op(op, args, **keywords)), self.stmt)

    def assign(self, suffix, value):
        return ast.copy_location(
            ast.Assign(targets=[_name(self.prefix + suffix, ast.Store())], value=value), self.stmt
        )

    def bind(self, suffix, value):
        if isinstance(value, (ast.Name, ast.Constant)):
            self.refs[suffix] = value
        else:
            self.bound[suffix] = self.assign(suffix, value)
            self.refs[suffix] = _name(self.prefix + suffix)

    def read(self, suffix):
        return copy.deepcopy(self.refs[suffix])

    def bindings(self, *suffixes):
        return [self.bound[s] for s in suffixes if s in self.bound]

    def resolve_types(self):
        sources = {**self.operands, **{key: self.keywords[key] for key in _STAGING[self.op]}}
        if all(isinstance(node, (ast.Name, ast.Constant)) or self.type(node) is not None
               for node in sources.values()) and all(
            isinstance(node, (ast.Name, ast.Constant)) or self.known_int(node) is not None
            for key, node in self.keywords.items() if key in _EXTENTS
        ):
            return
        # Preserve operand order, including cursors beside an inline view. Let
        # the normal parser bind each expression once before reading its type.
        keys = [*self.operands, "bias", *_EXTENTS]
        if "splitk" not in self.keywords and "splitn" not in self.keywords:
            keys += _STAGING[self.op]
        for key in keys:
            values = self.operands if key in self.operands else self.keywords
            node = values.get(key)
            if node is None:
                continue
            value = self.staging(node) if key in _STAGING[self.op] else node
            if isinstance(value, (ast.Name, ast.Constant)):
                continue
            assignment = ast.fix_missing_locations(self.assign(key, value))
            self.parser.parse_statement(assignment)
            values[key] = ast.copy_location(_name(self.prefix + key), node)

    def value(self, node):
        """Look up a named handle without evaluating an operand."""
        if isinstance(node, ast.Name):
            return self.parser.scope_manager.lookup_var(node.id)
        return None

    def type(self, node):
        value = self.value(node)
        if value is None and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("next", "current", "previous"):
                value = self.value(node.func.value)
        if value is None and isinstance(node, ast.Subscript):
            value = self.value(node.value)
        return _tile_type(getattr(value, "type", None))

    def staging(self, node):
        value = self.value(node)
        if isinstance(value, ir.Expr) and value in self.parser.tile_group_meta:
            return ast.copy_location(
                ast.Call(func=ast.Attribute(value=node, attr="next", ctx=ast.Load()), args=[], keywords=[]),
                node,
            )
        return node

    def dims(self, node, flag):
        if not isinstance(node, ast.Name):
            raise NotSupported(
                f"{self.api} needs the operand '{flag}' works on bound to a variable",
                span=self.span(node), hint="Assign the operand to a variable before this call",
            )
        value = self.value(node)
        declared = getattr(value, "type", None)
        if not isinstance(declared, ir.TileType):
            raise InvalidType(f"{self.api} '{flag}' needs a Tile", span=self.span(node))
        dims = declared.shape
        if len(dims) != 2 or not all(isinstance(d, ir.ConstInt) for d in dims):
            raise InvalidVal(
                f"{self.api} '{flag}' needs two compile-time dimensions", span=self.span(node)
            )
        return tuple(d.value for d in dims)

    def compact(self, node, reason):
        declared = self.type(node)
        if declared is None or declared.hardware_info.compact != ir.CompactMode.normal:
            raise InvalidVal(
                f"{self.api} needs '{ast.unparse(node)}' declared with compact=1 {reason}",
                span=self.span(node),
                hint="Narrowed Tiles must use the multiply's compact layout",
            )

    def static_int(self, name):
        node = self.keywords[name]
        value = self.parser.resolve_single_kwarg(name, node)
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidType(
                f"{self.api} '{name}' must be a compile-time int", span=self.span(node)
            )
        if value <= 0:
            raise InvalidVal(f"{self.api} '{name}' must be positive, got {value}", span=self.span(node))
        return value

    def known_int(self, node):
        if node is None:
            return None
        # Inspect pure constant arithmetic without emitting IR or calling a cursor.
        if any(isinstance(n, ast.expr) and not isinstance(n, (ast.Constant, ast.Name, ast.UnaryOp, ast.BinOp))
               for n in ast.walk(node)):
            return None
        found, value = ((True, node.value) if isinstance(node, ast.Constant)
                        else self.parser.expr_evaluator.try_eval_expr(node))
        if found and (not isinstance(value, int) or isinstance(value, bool)):
            raise InvalidType(f"{self.api} effective extents must be integers", span=self.span(node))
        return value if found else None

    def arguments(self):
        for node in self.call.args:
            if isinstance(node, ast.Starred):
                raise NotSupported(f"{self.api} does not take *args", span=self.span(node))
        for keyword in self.call.keywords:
            key, node = keyword.arg, keyword.value
            if key is None:
                raise NotSupported(f"{self.api} does not take **kwargs", span=self.span(node))
            if self.mx and key in ("bias", "splitn", "splitk"):
                reason = "pl.matmul_mx() has no bias operand" if key == "bias" else "pl.gemm_mx() does not split"
                raise NotSupported(reason, span=self.span(node), hint="Use explicit native operations")
        self.parser._check_declared_params(self.op, self.call, self.span(self.call))
        bound = declared_api(self.op).bind_declared(
            *self.call.args, **{kw.arg: kw.value for kw in self.call.keywords}
        ).arguments
        self.operands = {key: bound.pop(key) for key in _POSITIONAL[self.op]}
        self.keywords = dict(bound)
        # None is the public default for optional extents, splits and bias.
        for key in (*_EXTENTS, "splitn", "splitk", "bias"):
            node = self.keywords.get(key)
            if isinstance(node, (ast.Constant, ast.Name)):
                found, value = self.parser.expr_evaluator.try_eval_expr(node)
                if found and value is None:
                    self.keywords.pop(key)
        flags = {}
        for key, default in (("init", True), ("trans_a", False), ("trans_b", False)):
            node = self.keywords.get(key)
            value = default if node is None else self.parser.resolve_single_kwarg(key, node)
            if not isinstance(value, bool):
                raise InvalidType(f"{self.api} '{key}' must be a compile-time bool", span=self.span(node))
            flags[key] = value
        self.init = flags["init"]
        self.trans_a, self.trans_b = flags["trans_a"], flags["trans_b"]
        if "bias" in self.keywords and not self.init:
            raise InvalidVal("pl.gemm() takes a bias only with init=True", span=self.span(self.keywords["bias"]))
        if "splitn" in self.keywords and "splitk" in self.keywords:
            raise InvalidArgument("pl.gemm() 'splitn' and 'splitk' are mutually exclusive")
        if "splitn" in self.keywords and "m" in self.keywords:
            raise NotSupported(
                "pl.gemm() cannot combine 'm' with 'splitn'", span=self.span(self.keywords["m"]),
                hint="Narrowing M changes the pitch between accumulator column strips",
            )

    def transpose(self, node, flag):
        rows, cols = self.dims(node, flag)
        value = self.value(node)
        if value in self.parser._tile_slot_source:
            raise NotSupported(
                f"{self.api} '{flag}': the operand is a tile selected by a runtime index; "
                "the transpose alias would silently bind to slot 0",
                span=self.span(node),
                hint="Reinterpret the whole TileGroup as ZN first, then select a slot without this flag",
            )
        if _tile_layout(value.type) != ir.TensorLayout.NZ:
            raise NotSupported(
                f"{self.api} '{flag}' requires an NZ L1 Tile", span=self.span(node),
                hint="A ZN view already represents the transpose; pass it without this flag",
            )
        return self.call_op("reinterpret", [node], shape=_list([_const(cols), _const(rows)]), layout=self.attr("ZN"))

    def set_shape(self, key, dims, *, compact_axes=(0, 1)):
        node = self.operands[key] if key == "acc" else self.keywords[key]
        declared = self.type(node)
        if declared is None or len(declared.shape) != 2:
            raise InvalidType(f"{self.api} '{key}' needs a two-dimensional Tile", span=self.span(node))
        known = [self.known_int(dim) for dim in dims]
        if any(known[i] is None or not isinstance(declared.shape[i], ir.ConstInt)
               or known[i] != declared.shape[i].value for i in compact_axes):
            reason = "to narrow the MX scale group axis" if key.startswith("scale_") else "to take an effective extent"
            self.compact(node, reason)
        # Fixed descriptors need no update when they already describe this call.
        if declared.tile_view is not None and len(declared.tile_view.valid_shape) == 2 and all(
            isinstance(dim, ir.ConstInt) and dim.value == size
            for dim, size in zip(declared.tile_view.valid_shape, known)
        ):
            return []
        return [self.emit("set_validshape", [self.read(key), _list(dims)])]

    def stage_move(self, dst, src, transposed=False):
        args = [self.read(dst), self.read(src)]
        source_type = self.type(self.operands[src])
        target_type = self.type(self.keywords[dst])
        if source_type is not None and target_type is not None and all(
            isinstance(d, ir.ConstInt) for t in (source_type, target_type) for d in t.shape
        ):
            source_shape = [d.value for d in source_type.shape]
            if transposed:
                source_shape.reverse()
            if source_shape != [d.value for d in target_type.shape]:
                # Smaller staging uses TEXTRACT; plain TMOV requires equal shapes.
                args.append(_list([_const(0), _const(0)]))
        return self.emit("move", args)

    def matmul(self, init):
        acc = self.read("acc")
        if self.mx:
            args = [acc, self.read("l0a"), self.read("l0b"), self.read("scale_a"), self.read("scale_b")]
            return self.emit("matmul_mx" if init else "matmul_mx_acc", args if init else [self.read("acc"), *args])
        args = [acc, self.read("l0a"), self.read("l0b")]
        if not init:
            return self.emit("matmul_acc", [self.read("acc"), *args])
        if "bias" in self.refs:
            args.append(self.read("bias"))
        return self.emit("matmul", args)


    def narrow(self):
        result = []
        for key, axes in (
            ("l0a", ("m", "k")), ("l0b", ("k", "n")), ("acc", ("m", "n")),
        ):
            result += self.set_shape(key, [self.read(axis) for axis in axes])
        return result

    def check_shapes(self):
        # A move may legally copy a smaller L0 Tile. GEMM's omitted axes mean
        # the full L1 extent, so silently using that smaller Tile would truncate
        # the multiply. Explicit extents and split axes select their own bounds.
        shapes = []
        for key in ("l1a", "l1b"):
            declared = self.type(self.operands[key])
            if declared is None or len(declared.shape) != 2 or not all(
                isinstance(d, ir.ConstInt) for d in declared.shape
            ):
                raise InvalidType(
                    f"{self.api} '{key}' needs a Tile with two compile-time dimensions",
                    span=self.span(self.operands[key]),
                )
            shapes.append([d.value for d in declared.shape])
        a, b = shapes
        if self.trans_a:
            a.reverse()
        if self.trans_b:
            b.reverse()
        if a[1] != b[0]:
            raise InvalidVal(f"{self.api} operands disagree on K: {a[1]} and {b[0]}")
        self.sizes = dict(zip(_EXTENTS, (a[0], b[1], a[1])))
        for axis, size in self.sizes.items():
            node = self.keywords.get(axis, _const(size))
            known = self.known_int(node)
            if known is None:
                scalar_type = getattr(self.value(node), "type", None)
                if not isinstance(scalar_type, ir.ScalarType) or not (
                    scalar_type.dtype.is_int() or scalar_type.dtype == ir.DataType.INDEX
                ):
                    raise InvalidType(
                        f"{self.api} '{axis}' must be an integer scalar", span=self.span(node)
                    )
            if known is not None and not 0 < known <= size:
                raise InvalidVal(
                    f"{self.api} '{axis}' is {known}, outside the declared extent 1..{size}",
                    span=self.span(node),
                )
            self.bind(axis, _const(known) if known is not None else node)
        for node, axes, sizes in (
            (self.operands["acc"], ("m", "n"), (a[0], b[1])),
            (self.keywords["l0a"], ("m", "k"), a),
            (self.keywords["l0b"], ("k", "n"), b),
        ):
            declared = self.type(node)
            if declared is None or len(declared.shape) != 2:
                continue
            for axis, actual, size in zip(axes, declared.shape, sizes):
                if "splitk" in self.keywords and axis == "k":
                    continue
                if "splitn" in self.keywords and axis == "n" and node is self.keywords["l0b"]:
                    continue
                if axis in self.keywords:
                    known = self.known_int(self.keywords[axis])
                    if known is not None and isinstance(actual, ir.ConstInt) and known > actual.value:
                        raise InvalidVal(
                            f"{self.api} '{axis}' is {known}, past '{ast.unparse(node)}' "
                            f"capacity {actual.value}", span=self.span(self.keywords[axis]),
                        )
                    continue
                if isinstance(actual, ir.ConstInt) and actual.value != size:
                    raise InvalidVal(
                        f"{self.api} '{ast.unparse(node)}' {axis.upper()} extent must match "
                        f"the L1 operand ({size}), got {actual.value}", span=self.span(node),
                        hint="Supply an explicit effective extent for a partial multiply",
                    )

    def mx_k(self):
        declared = self.sizes["k"]
        if declared % _MX_K_UNIT:
            raise InvalidVal(
                f"{self.api} operands declare K={declared}, which is not a multiple of {_MX_K_UNIT}",
                hint="MX K extents must cover pairs of 32-element scale groups",
            )
        node = self.keywords.get("k")
        known = self.known_int(node)
        if known is not None and known % _MX_K_UNIT:
            raise InvalidVal(
                f"{self.api} 'k' is {known}, which is not a multiple of {_MX_K_UNIT}",
                span=self.span(node), hint="Pad K or use a non-MX multiply for the tail",
            )

    def narrow_planes(self):
        result = []
        known = self.known_int(self.read("k"))
        groups = (
            _const(known // _MX_GROUP) if known is not None
            else ast.BinOp(left=self.read("k"), op=ast.FloorDiv(), right=_const(_MX_GROUP))
        )
        for key, axis, group_axis in (("scale_a", "m", 1), ("scale_b", "n", 0)):
            dims = [self.read(axis), copy.deepcopy(groups)]
            if group_axis == 0:
                dims.reverse()
            # M/N alone preserve the group pitch; narrowing K must compact it.
            result += self.set_shape(key, dims, compact_axes=(group_axis,))
        return result

    def loop(self, axis, extent, split, body):
        return ast.copy_location(
            ast.For(
                target=_name(self.prefix + axis, ast.Store()),
                iter=self.call_op("range", [_const(0), _const(extent), _const(split)]),
                body=body, orelse=[],
            ), self.stmt,
        )

    def check_staging_split(self, key, axis, split):
        declared = self.type(self.keywords[key])
        if declared is None or len(declared.shape) != 2:
            raise InvalidType(f"{self.api} '{key}' needs a two-dimensional staging Tile",
                              span=self.span(self.keywords[key]))
        size = declared.shape[axis]
        if not isinstance(size, ir.ConstInt) or size.value != split:
            raise InvalidVal(
                f"{self.api} '{key}' dimension {axis} must equal the split size {split}",
                span=self.span(self.keywords[key]),
                hint="The staging move's physical width must match the generated loop step",
            )

    def split_k(self, outer):
        split = self.static_int("splitk")
        self.check_staging_split("l0a", 1, split)
        self.check_staging_split("l0b", 0, split)
        extent = self.static_int("k") if "k" in self.keywords else self.sizes["k"]
        m, n = self.read("m"), self.read("n")
        outer += self.set_shape("acc", [copy.deepcopy(m), copy.deepcopy(n)])
        body = self.bindings("l0a", "l0b")
        step = _name(self.prefix + "k")
        if extent % split:
            body.append(self.assign(
                "vk", self.call_op("min", [_const(split), ast.BinOp(left=_const(extent), op=ast.Sub(), right=step)])
            ))
            valid = _name(self.prefix + "vk")
        else:
            valid = _const(split)
        body += self.set_shape("l0a", [m, copy.deepcopy(valid)])
        body += self.set_shape("l0b", [copy.deepcopy(valid), n])
        body += [
            self.emit("move", [self.read("l0a"), self.read("l1a"), _list([_const(0), copy.deepcopy(step)])]),
            self.emit("move", [self.read("l0b"), self.read("l1b"), _list([copy.deepcopy(step), _const(0)])]),
        ]
        if self.init:
            body.append(ast.If(
                test=ast.Compare(left=copy.deepcopy(step), ops=[ast.Eq()], comparators=[_const(0)]),
                body=[self.matmul(True)], orelse=[self.matmul(False)],
            ))
        else:
            body.append(self.matmul(False))
        return [*outer, self.loop("k", extent, split, body)]

    def strip_group(self, suffix, rows, width, addresses, ids, dtype):
        kwargs = {
            "type": self.call_op(
                "TileType", [], shape=_list([_const(rows), _const(width)]),
                dtype=self.attr(dtype), target_memory=self.attr("MemorySpace.Acc"),
            ),
            "addrs": _list([_const(addr) for addr in addresses]),
        }
        if ids == "auto":
            kwargs["mutex_ids"] = _const("auto")
            kwargs["depth"] = _const(len(addresses))
        elif ids:
            kwargs["mutex_ids"] = _list([_list([_const(i) for i in ids]) for _ in addresses])
        else:
            kwargs["depth"] = _const(len(addresses))
        return self.assign(suffix, self.call_op("make_tile_group", [], **kwargs))

    def split_n(self, outer):
        split = self.static_int("splitn")
        if split % 16:
            raise InvalidVal(
                f"pl.gemm() 'splitn' must be a multiple of 16, got {split}",
                span=self.span(self.keywords["splitn"]),
            )
        if "bias" in self.keywords:
            raise NotSupported(
                "pl.gemm() cannot combine 'bias' with 'splitn'", span=self.span(self.keywords["bias"]),
                hint="Stage the corresponding bias strip in an explicit N loop",
            )
        self.check_staging_split("l0b", 1, split)
        acc_node = self.operands["acc"]
        rows, cols = self.dims(acc_node, "splitn")
        b_cols = self.sizes["n"]
        if cols != b_cols:
            raise InvalidVal(
                f"pl.gemm() cannot split along N: the accumulator holds {cols} columns and l1b holds {b_cols}",
            )
        value = self.value(acc_node)
        with current_span(self.span(acc_node)):
            _check_tile_memory_space(self.api, "acc", value, ir.MemorySpace.Acc, "L0C (Acc)")
        if value in self.parser._tile_slot_source:
            raise NotSupported(
                "pl.gemm() cannot split along N through a runtime-selected accumulator",
                span=self.span(acc_node), hint="Use a static accumulator slot; strips need fixed addresses",
            )
        meta = self.parser.tile_mutex_lock_meta(value)
        # Static selection uses only the chosen slot's IDs. For AUTO, the native
        # manager unifies contained strip aliases with their parent allocation.
        row = meta[0].elements if meta else []
        ids = "auto" if any(isinstance(i, ir.Var) for i in row) else sorted({i.value for i in row})
        declared = value.type
        addr = declared.memref.addr
        if not isinstance(addr, ir.ConstInt):
            raise NotSupported("pl.gemm() cannot split along N through a computed address", span=self.span(acc_node))
        limit = self.static_int("n") if "n" in self.keywords else cols
        count, tail = divmod(limit, split)
        tail16 = ((tail + 15) // 16) * 16
        if count == 0:
            raise InvalidVal(
                f"pl.gemm() 'splitn' is {split}, wider than the {limit} columns to write",
                span=self.span(self.keywords["splitn"]),
                hint="Drop splitn and pass the extent as n",
            )
        if count * split + tail16 > cols:
            raise InvalidVal("pl.gemm() cannot place the last strip inside the accumulator")
        # NZ column fractals are contiguous at full M, with 16-column alignment.
        stride = tile_slot_size((rows, split), declared.dtype)
        import pypto_pro.language as language

        dtype = next(
            name for name in dir(language) if name.startswith("DT_") and getattr(language, name) == declared.dtype
        )
        declares = [
            self.strip_group("strips", rows, split, [addr.value + j * stride for j in range(count)], ids, dtype)
        ]
        if tail:
            declares.append(self.strip_group("tailstrip", rows, tail16, [addr.value + count * stride], ids, dtype))
        outer += self.set_shape("acc", [_const(rows), self.read("n")])
        prologue = self.bindings("l0a")
        k = self.read("k")
        prologue += self.set_shape("l0a", [_const(rows), copy.deepcopy(k)])
        prologue.append(self.emit("move", [self.read("l0a"), self.read("l1a")]))
        step = _name(self.prefix + "n")

        def one_step(group, index, width):
            body = self.set_shape("l0b", [copy.deepcopy(k), _const(width)])
            body.append(self.emit("move", [
                self.read("l0b"), self.read("l1b"), _list([_const(0), copy.deepcopy(step)]),
            ]))
            body.append(self.assign("strip", ast.Subscript(
                value=_name(self.prefix + group), slice=index, ctx=ast.Load(),
            )))
            parent = self.refs["acc"]
            self.refs["acc"] = _name(self.prefix + "strip")
            body.append(self.matmul(self.init))
            self.refs["acc"] = parent
            return body

        ordinal = ast.BinOp(left=copy.deepcopy(step), op=ast.FloorDiv(), right=_const(split))
        body = self.bindings("l0b")
        if tail:
            body.append(ast.If(
                test=ast.Compare(left=step, ops=[ast.Eq()], comparators=[_const(count * split)]),
                body=one_step("tailstrip", _const(0), tail16),
                orelse=one_step("strips", ordinal, split),
            ))
        else:
            body += one_step("strips", ordinal, split)
        return [*outer, *declares, *prologue, self.loop("n", limit, split, body)]

    def lower(self):
        self.arguments()
        self.parser._validate_op_scope("matmul_mx" if self.mx else "matmul", self.call)
        self.resolve_types()
        self.check_shapes()
        self.bind("acc", self.operands["acc"])
        for key, transposed in (("l1a", self.trans_a), ("l1b", self.trans_b)):
            node = self.operands[key]
            self.bind(key, self.transpose(node, "trans_a" if key == "l1a" else "trans_b") if transposed else node)
        if self.mx:
            for key in ("l1a_scale", "l1b_scale"):
                self.bind(key, self.operands[key])
        for key in _STAGING[self.op]:
            self.bind(key, self.staging(self.keywords[key]))
        if "bias" in self.keywords:
            self.bind("bias", self.keywords["bias"])
        if self.mx:
            self.mx_k()
        outer = self.bindings("acc", "l1a", "l1b", "l1a_scale", "l1b_scale", "bias", *_EXTENTS)
        if "splitk" in self.keywords:
            result = self.split_k(outer)
        elif "splitn" in self.keywords:
            result = self.split_n(outer)
        else:
            narrowing = self.narrow()
            if self.mx:
                narrowing += self.narrow_planes()
            moves = [self.stage_move("l0a", "l1a", self.trans_a)]
            if self.mx:
                moves.append(self.stage_move("scale_a", "l1a_scale"))
            moves.append(self.stage_move("l0b", "l1b", self.trans_b))
            if self.mx:
                moves.append(self.stage_move("scale_b", "l1b_scale"))
            result = [*outer, *self.bindings(*_STAGING[self.op]), *narrowing, *moves, self.matmul(self.init)]
        for stmt in result:
            ast.fix_missing_locations(ast.copy_location(stmt, self.stmt))
        return result
