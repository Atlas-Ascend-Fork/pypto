# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call and operation parsing helpers for ASTParser."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import inspect
import linecache
import logging
import sys
from typing import Any, Callable

from pypto.pypto_impl import ir
from pypto_pro.ir._utils import _is_int
from pypto_pro.ir.op._op_registry import _AST_OP_REGISTRY, _OP_REGISTRY

from ..._errors import (
    BadFd,
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NotSupported,
    argument_span,
    current_span,
)
from .. import _api as _language_api
from .. import distribute as _distribute_api
from .._vf_api import Vf
from . import _argument_checker
from ._control_flow_parser import _is_bare_return
from ._expr_evaluator import ExprEvaluator
from ._span_tracker import SpanTracker
from .diagnostics import check_in_range

# Largest thread count a SIMT function may declare.
_MAX_SIMT_THREADS = 2048

logger = logging.getLogger(__name__)


def _get_source_file(entity: Callable | type) -> str:
    """Get source filename for an entity, with fallback to code object attributes.

    Args:
        entity: Function or class to get source file for

    Returns:
        Source filename string
    """
    try:
        return inspect.getfile(entity)
    except (OSError, TypeError):
        pass

    # Fallback: extract from code object
    if callable(entity) and hasattr(entity, "__code__"):
        return entity.__code__.co_filename

    # For classes, find a method with a code object
    if isinstance(entity, type):
        for attr in entity.__dict__.values():
            if callable(attr) and hasattr(attr, "__code__"):
                return attr.__code__.co_filename

    return "<unknown>"


def _find_entity_in_source(
    all_lines: list[str], name: str, entity_type: str, start_line_hint: int | None = None
) -> tuple[list[str], int] | None:
    """Find an entity definition in source lines using AST parsing.

    Args:
        all_lines: All source lines from the file
        name: Name of the entity to find
        entity_type: "function" or "class"
        start_line_hint: Optional line number to disambiguate entities with the same name

    Returns:
        Tuple of (source_lines, starting_line_1based) or None if not found
    """
    source_text = "".join(all_lines)
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None

    node_type = ast.FunctionDef if entity_type == "function" else ast.ClassDef
    candidates = [node for node in ast.walk(tree) if isinstance(node, node_type) and node.name == name]

    if not candidates:
        return None

    if len(candidates) == 1:
        node = candidates[0]
    elif start_line_hint is not None:
        # Disambiguate using the code object's line number
        node = min(candidates, key=lambda n: abs(n.lineno - start_line_hint))
    else:
        node = candidates[0]

    # Start from the first decorator line if present
    start_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    end_line = node.end_lineno or node.lineno
    # Lines are 1-based in AST
    source_lines = all_lines[start_line - 1:end_line]
    return source_lines, start_line


def _source_from_inspect(entity: Callable | type) -> tuple[str, list[str], int] | None:
    """Get source information via inspect, returning None when unavailable."""
    try:
        source_file = inspect.getfile(entity)
        source_lines_raw, starting_line = inspect.getsourcelines(entity)
        return source_file, source_lines_raw, starting_line
    except (OSError, TypeError):
        return None


def _source_from_linecache(
    source_file: str,
    name: str,
    entity_type: str,
    start_line_hint: int | None,
) -> tuple[str, list[str], int] | None:
    """Get source information from linecache."""
    all_lines = linecache.getlines(source_file)
    if not all_lines:
        return None
    result = _find_entity_in_source(all_lines, name, entity_type, start_line_hint)
    return None if result is None else (source_file, result[0], result[1])


def _restore_linecache_entry(prev_entry) -> None:
    """Restore the synthetic <string> linecache entry used for python -c parsing."""
    if prev_entry is not None:
        linecache.cache["<string>"] = prev_entry
    else:
        linecache.cache.pop("<string>", None)


def _source_from_python_c(
    source_file: str,
    name: str,
    entity_type: str,
    start_line_hint: int | None,
) -> tuple[str, list[str], int] | None:
    """Get source information from sys.orig_argv for python -c invocations."""
    if source_file != "<string>" or not hasattr(sys, "orig_argv"):
        return None
    try:
        c_index = sys.orig_argv.index("-c")
    except ValueError:
        return None
    if c_index + 1 >= len(sys.orig_argv):
        return None

    code_str = sys.orig_argv[c_index + 1]
    code_lines = code_str.splitlines(keepends=True)
    prev_entry = linecache.cache.get("<string>")
    linecache.cache["<string>"] = (len(code_str), None, code_lines, "<string>")
    try:
        result = _find_entity_in_source(code_lines, name, entity_type, start_line_hint)
        return None if result is None else (source_file, result[0], result[1])
    finally:
        _restore_linecache_entry(prev_entry)


def _get_source_info(entity: Callable | type, entity_type: str) -> tuple[str, list[str], int]:
    """Get source file, source lines, and starting line for an entity.

    Tries multiple strategies:
    1. Standard inspect.getsourcelines()
    2. linecache fallback (handles IPython, pre-populated cache)
    3. sys.orig_argv for `python -c` invocations
    4. Clear error with actionable hint

    Args:
        entity: Function or class to get source for
        entity_type: "function" or "class"

    Returns:
        Tuple of (source_file, source_lines_raw, starting_line)

    Raises:
        InvalidOperation: If source cannot be retrieved by any strategy
    """
    name = entity.__name__ if hasattr(entity, "__name__") else str(entity)

    # Get a line number hint from the code object to disambiguate same-name entities
    start_line_hint: int | None = None
    if callable(entity) and hasattr(entity, "__code__"):
        start_line_hint = entity.__code__.co_firstlineno

    # Strategy 1: Standard inspect
    inspect_result = _source_from_inspect(entity)
    if inspect_result is not None:
        return inspect_result

    # Get source file via fallback for strategies 2-3
    source_file = _get_source_file(entity)

    # Strategy 2: linecache fallback
    linecache_result = _source_from_linecache(source_file, name, entity_type, start_line_hint)
    if linecache_result is not None:
        return linecache_result

    # Strategy 3: sys.orig_argv for `python -c`
    python_c_result = _source_from_python_c(source_file, name, entity_type, start_line_hint)
    if python_c_result is not None:
        return python_c_result

    # Strategy 4: Clear error
    raise BadFd(
        f"Cannot retrieve source code for {entity_type} '{name}'",
        hint="Define the callable in a Python source file so its AST can be inspected",
    )


# Mutex carrier for one selected Tile (runtime IDs, possible values, memory, dedup id).


@dataclass(frozen=True)
class _InlineFunctionTemplate:
    """Immutable source template for one directly-expanded Python callable."""

    func_def: ast.FunctionDef
    closure_vars: dict[str, Any]
    span_tracker: SpanTracker


class _InlineLocalRenamer(ast.NodeTransformer):
    """Rename bindings local to one helper expansion without touching closures."""

    def __init__(self, names: set[str], prefix: str):
        self._names = names
        self._rename = {name: f"{prefix}{name}" for name in names}

    def generic_visit(self, node):
        if isinstance(node, ast.Name) and node.id in self._rename:
            node.id = self._rename[node.id]
        return super().generic_visit(node)


class _InlineReturnLowerer(ast.NodeTransformer):
    """Lower helper returns and their loop-propagation checks in Python AST."""

    # ast.NodeTransformer dispatches by reflection on `visit_<NodeClassName>`, so the CamelCase
    # halves of visit_Return / visit_For / visit_While are the library's spelling, not ours.
    # Renaming them to snake_case silently turns the visitor into a no-op.
    # pylint: disable=huawei-invalid-name

    def __init__(self, return_val_name: str, returned_name: str):
        self._return_val_name = return_val_name
        self._returned_name = returned_name

    @staticmethod
    def _assignment(name: str, value: ast.expr, location: ast.AST) -> ast.Assign:
        target = ast.copy_location(ast.Name(id=name, ctx=ast.Store()), location)
        return ast.copy_location(ast.Assign(targets=[target], value=value), location)

    def _return_checkpoint(self, location: ast.AST) -> ast.If:
        condition = ast.copy_location(ast.Name(id=self._returned_name, ctx=ast.Load()), location)
        break_stmt = ast.copy_location(ast.Break(), location)
        return ast.copy_location(ast.If(test=condition, body=[break_stmt], orelse=[]), location)

    def visit_Return(self, node: ast.Return):
        value = (
            ast.copy_location(ast.Constant(value=None), node)
            if _is_bare_return(node)
            else node.value
        )
        lowered: list[ast.stmt] = [self._assignment(self._return_val_name, value, node)]
        returned = ast.copy_location(ast.Constant(value=True), node)
        lowered.append(self._assignment(self._returned_name, returned, node))
        lowered.append(ast.copy_location(ast.Break(), node))
        return lowered

    def _lower_loop(self, node: ast.For | ast.While) -> ast.stmt | list[ast.stmt]:
        has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
        lowered_loop = self.generic_visit(node)
        if not has_return:
            return lowered_loop
        return [lowered_loop, self._return_checkpoint(node)]

    def visit_For(self, node: ast.For):
        return self._lower_loop(node)

    def visit_While(self, node: ast.While):
        return self._lower_loop(node)

    def lower(self, body: list[ast.stmt], location: ast.AST) -> list[ast.stmt]:
        module = ast.Module(body=body, type_ignores=[])
        self.visit(module)
        returned_init = self._assignment(
            self._returned_name,
            ast.copy_location(ast.Constant(value=False), location),
            location,
        )
        fallback_return = self.visit_Return(ast.copy_location(ast.Return(value=None), location))
        wrapper = ast.copy_location(
            ast.While(
                test=ast.copy_location(ast.Constant(value=True), location),
                body=[returned_init, *module.body, *fallback_return],
                orelse=[],
            ),
            location,
        )
        ast.fix_missing_locations(wrapper)
        return [wrapper]


# Builtin function names that map to pl.* ops (syntax sugar).
_BUILTIN_TO_OP: dict[str, str] = {
    "min": "min",
    "max": "max",
}


def _dtypes_compatible(a: ir.DataType, b: ir.DataType) -> bool:
    """Lenient dtype compatibility: same dtype, or both integer-like / both float-like.

    Different numeric dtypes of the same family (e.g. INT32 and INT64, FP16 and FP32)
    are interconvertible and treated as compatible.
    """
    return a == b or (a.is_int() and b.is_int()) or (a.is_float() and b.is_float())


def _is_const_arg(arg: ir.Expr) -> bool:
    """True for call-site arguments that SIMD VF instantiation specializes out."""
    return isinstance(arg, (ir.ConstBool, ir.ConstInt))


def _const_name_value(const: ir.Expr) -> str:
    """Render a compile-time constant for a specialized function-name suffix.

    Booleans carry a ``b`` prefix so a ``True`` specialization can never collide
    with an integer ``1`` specialization under the same parameter name.
    """
    value = const.value
    if isinstance(const, ir.ConstBool):
        return "b1" if value else "b0"
    return str(value).replace("-", "n")


def _types_compatible(annotated: ir.Type, actual: ir.Type) -> bool:
    """Lenient type compatibility: same kind + compatible dtype.

    Ignores shape/memref/layout details. ``UnknownType`` is compatible with anything.
    Used to validate a function annotation against the type derived from the actual
    argument/return-value expression.
    """
    if isinstance(annotated, ir.UnknownType) or isinstance(actual, ir.UnknownType):
        return True
    if isinstance(annotated, ir.ScalarType) and isinstance(actual, ir.ScalarType):
        return _dtypes_compatible(annotated.dtype, actual.dtype)
    if isinstance(annotated, ir.PtrType) and isinstance(actual, ir.PtrType):
        return _dtypes_compatible(annotated.dtype, actual.dtype)
    # TensorType / TileType both derive from ShapedType; compare dtype, ignore shape/memref.
    if isinstance(annotated, ir.ShapedType) and isinstance(actual, ir.ShapedType):
        return type(annotated) is type(actual) and _dtypes_compatible(annotated.dtype, actual.dtype)
    if isinstance(annotated, ir.TupleType) and isinstance(actual, ir.TupleType):
        if len(annotated.types) != len(actual.types):
            return False
        return all(_types_compatible(a, b) for a, b in zip(annotated.types, actual.types))
    return False


def _check_type_compatible(annotated: ir.Type, actual: ir.Type, *, what: str, name: str, span) -> None:
    """Raise InvalidType if ``annotated`` is not compatible with ``actual``."""
    if not _types_compatible(annotated, actual):
        raise InvalidVal(
            f"{what} '{name}' annotated as {annotated} but called/returned with {actual}",
            span=span,
            hint="Make the annotation match the actual argument/return type, or remove it.",
            parser_retry=True,
        )


def _validate_simt_tensor_type(tensor_type: ir.TensorType, subject: str, span: ir.Span) -> None:
    if tensor_type.dtype.get_bit() < 8:
        raise NotSupported(
            f"{subject} uses an unsupported sub-byte dtype",
            span=span,
            hint="Use a byte-addressable Tensor element type such as pl.DT_UINT8 or wider.",
            parser_retry=True,
        )
    tensor_view = tensor_type.tensor_view
    if tensor_view is None or tensor_view.layout != ir.Layout.ND:
        raise InvalidVal(
            f"{subject} requires ND layout",
            span=span,
            hint="Use pl.Tensor[[shape], dtype] or specify pl.ND explicitly.",
            parser_retry=True,
        )


def _validate_simt_parameters(func: ir.Function, role: str, *, entry_type: ir.FunctionType) -> None:
    """Validate the common native CCE ABI accepted by SIMT functions."""
    for param in func.params:
        if entry_type == ir.FunctionType.SimtKernel and isinstance(param.type, ir.TileType):
            raise InvalidVal(
                f"{role} parameter '{param.name}' must be a scalar or Tensor, not Tile",
                span=param.span,
            )
        if not isinstance(param.type, (ir.ScalarType, ir.TensorType, ir.TileType)):
            raise InvalidVal(
                f"{role} parameter '{param.name}' must resolve to a scalar, Tensor, or Tile value, got {param.type}",
                span=param.span,
                hint="Ptr, Tuple, and tiling parameters are not supported.",
                parser_retry=True,
            )
        if isinstance(param.type, ir.TensorType):
            _validate_simt_tensor_type(param.type, f"{role} Tensor parameter '{param.name}'", param.span)
        if isinstance(param.type, ir.TileType):
            if len(param.type.shape) != 2 or not all(isinstance(dim, ir.ConstInt) for dim in param.type.shape):
                raise InvalidShape(
                    f"{role} Tile parameter '{param.name}' must have a static two-dimensional shape",
                    span=param.span,
                    hint="Pass a static two-dimensional Tile created with pl.make_tile().",
                    parser_retry=True,
                )
            if param.type.dtype.get_bit() < 8:
                raise NotSupported(
                    f"{role} Tile parameter '{param.name}' uses an unsupported sub-byte dtype",
                    span=param.span,
                    hint="Use a byte-addressable Tile element type such as pl.DT_UINT8 or wider.",
                    parser_retry=True,
                )

class CallParserMixin:
    """Mixin containing call and operation parsing methods for ``ASTParser``."""

    # -------------------------------------------------------------------------
    # VF execution-domain rules
    # -------------------------------------------------------------------------

    # The pl.* calls a VF body may still make. Everything in the body lowers into
    # one VEC_SCOPE, which bisheng requires to hold VF instructions only, so the
    # survivors are exactly the ops that lower to a scalar expression or a loop
    # bound and emit no pipe, buffer or section work of their own.
    _VF_SCALAR_PL_OPS: frozenset[str] = frozenset({"range", "min", "max", "const"})

    # Tile-group cursor accessors, lowered by _route_ir_node_method rather than by
    # the op registry. Named here because that route emits a struct.set plus scalar
    # arithmetic, which is plain CCE work and must not land in a VEC_SCOPE.
    _TILE_GROUP_ACCESSORS: frozenset[str] = frozenset({"next", "current", "previous"})

    def _extract_op_name(self, func: ast.expr) -> str | None:
        """Extract a normalized op name from a call's callee node.

        pl.system.bar_all -> "system.bar_all"
        pl.add_scalar     -> "add_scalar"
        pld.barrier       -> "dist.barrier"  (distribute module prefix)
        vf.add            -> "vf.add"  (vf prefix retained)
        bare_name         -> the op an imported alias stands for, otherwise None

        A bare name is resolved by identity against the API declarations, not by
        their ``__module__`` string: an alias such as ``m = vf.create_mask`` would
        otherwise reach parse_call as a plain callable and smuggle a VF
        instruction past the execution-domain check in _validate_op_scope, and a
        renamed declaration module would silently reopen that bypass instead of
        failing loudly. The distribute module is matched by identity for the same
        reason: without the explicit "dist." prefix, ``pld.barrier`` would collapse
        to the bare name "barrier" and collide with the top-level op namespace.
        """
        if isinstance(func, ast.Name):
            resolved = self.expr_evaluator.closure_vars.get(func.id)
            name = getattr(resolved, "__name__", None)
            if not isinstance(name, str):
                return None
            if getattr(Vf, name, None) is resolved:
                return f"vf.{name}"
            if getattr(_distribute_api, name, None) is resolved:
                return f"dist.{name}"
            if getattr(_language_api, name, None) is resolved:
                return name
            return None

        attrs: list[str] = []
        node: ast.expr = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)
        if len(attrs) < 2:
            return None
        if attrs[0] == "vf" or self.expr_evaluator.closure_vars.get(attrs[0]) is Vf:
            return ".".join(["vf", *attrs[1:]])
        if attrs[1] == "Vf":
            return ".".join(["vf", *attrs[2:]])
        if self.expr_evaluator.closure_vars.get(attrs[0]) is _distribute_api:
            return ".".join(["dist", *attrs[1:]])
        return ".".join(attrs[1:])

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        """Check if an AST statement is a docstring (string constant expression)."""
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)

    @staticmethod
    def _inline_param_list(func_def: ast.FunctionDef) -> list[ast.arg]:
        return list(func_def.args.args)

    @staticmethod
    def _inline_assigned_names(func_def: ast.FunctionDef) -> set[str]:
        """Collect every name the helper body assigns to (Store/Del), skipping nested functions.

        One walk serves both callers: local names are the params plus the assigned names, and
        reassigned parameters are the intersection of the two.
        """
        assigned: set[str] = set()

        class _Collector(ast.NodeVisitor):
            def generic_visit(self, node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return None
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    assigned.add(node.id)
                    return None
                return super().generic_visit(node)

        collector = _Collector()
        for stmt in func_def.body:
            collector.visit(stmt)
        return assigned

    @staticmethod
    def _has_unsupported_inline_params(args: ast.arguments) -> bool:
        if args.posonlyargs or args.vararg is not None:
            return True
        if args.kwarg is not None or args.kwonlyargs:
            return True
        return bool(args.kw_defaults)

    @classmethod
    def _make_call_with_return_type(
        cls,
        op: ir.Op,
        args: list[ir.Expr],
        return_types: list[ir.Type],
        span: ir.Span,
    ) -> ir.Expr:
        """Create an ir.Call, attaching the return type when known.

        Args:
            op: Op identifying the callee
            args: Parsed argument expressions
            return_types: The callee's return type list (may be empty)
            span: Source span for the call
        """
        if not return_types:
            return ir.Call(op, args, span)
        if len(return_types) == 1:
            return ir.Call(op, args, return_types[0], span)
        return ir.Call(op, args, ir.TupleType(return_types), span)

    @classmethod
    def _retrieve_function_source(
        cls,
        func_name: str,
        fn: Callable,
        span: ir.Span,
        decorator_hint: str,
    ) -> tuple[str, list[str], int, int, ast.FunctionDef]:
        """Retrieve source, parse AST, and locate FunctionDef for a callable.

        Returns (source_file, source_lines, line_offset, col_offset, func_def).
        """
        import textwrap as _tw

        try:
            source_file, source_lines_raw, starting_line = _get_source_info(fn, "function")
        except Exception as e:
            raise InvalidOperation(
                f"Cannot compile '{func_name}': unable to retrieve source -{e}",
                span=span,
                hint=f"Define '{func_name}' in a .py file, or use {decorator_hint}",
                parser_retry=True,
            ) from e

        source_code = _tw.dedent("".join(source_lines_raw))
        col_offset = len(source_lines_raw[0]) - len(source_lines_raw[0].lstrip()) if source_lines_raw else 0
        line_offset = starting_line - 1
        source_lines = source_code.split("\n")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise InvalidOperation(
                f"Cannot parse '{func_name}': {e}",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
                parser_retry=True,
            ) from e

        func_def = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__),
            None,
        )
        if func_def is None:
            raise InvalidOperation(
                f"Cannot find function definition for '{func_name}' in source",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
                parser_retry=True,
            )

        return source_file, source_lines, line_offset, col_offset, func_def

    @classmethod
    def _build_function_closure(cls, fn: Callable) -> dict[str, Any]:
        """Build closure dict from a callable's globals and free variables."""
        fn_closure: dict[str, Any] = {**fn.__globals__}
        if fn.__closure__ and fn.__code__.co_freevars:
            fn_closure.update(dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__))))
        return fn_closure

    def parse_call(self, call: ast.Call) -> Any:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func

        if isinstance(func, ast.Subscript):
            return self._parse_indexed_call(call)

        if isinstance(func, ast.Attribute) or self._extract_op_name(func) is not None:
            return self.parse_op_call(call)

        # Handle bare-name calls to external IR functions or inline Python callables.
        if isinstance(func, ast.Name):
            func_name = func.id

            # Builtin min/max -> route to pl.min/pl.max (scalar_ops.py)
            if func_name in _BUILTIN_TO_OP:
                op_name = _BUILTIN_TO_OP[func_name]
                op_func = _OP_REGISTRY.get(op_name)
                if op_func is not None:
                    return self._dispatch_op(op_name, call, op_func)

            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if callable(resolved) and not isinstance(resolved, type):
                return self._implicit_func_call(func_name, resolved, call)

        raise NotSupported(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations or call an inline Python helper by name",
            parser_retry=True,
        )

    def _parse_indexed_call(self, call: ast.Call) -> Any:
        """Route a call whose callee uses indexed-call syntax."""
        from pypto_pro.ir.op.simt_ops import _parse_simt_launch

        from .decorator import is_simt_function

        indexed = call.func
        if isinstance(indexed, ast.Subscript) and isinstance(indexed.value, ast.Name):
            local_name = indexed.value.id
            callee_template = self.expr_evaluator.closure_vars.get(local_name)
            if is_simt_function(callee_template):
                self._validate_op_scope("simt.launch", call)
                return _parse_simt_launch(self, call, local_name, callee_template)

        raise NotSupported(
            f"Unsupported indexed function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint='Indexed calls require a @pl.vector_function(mode="simt") target',
            parser_retry=True,
        )

    def parse_op_call(self, call: ast.Call) -> Any:
        """Parse operation call like pl.add() or pl.simt.abs().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        op_name = self._extract_op_name(call.func)

        if op_name in ("gemm", "gemm_mx"):
            raise NotSupported(
                f"pl.{op_name}() must be called as a standalone statement",
                hint="The operation writes its accumulator and returns no value",
            )

        # Ahead of _route_ir_node_method: that route parses the receiver and emits
        # the tile-group cursor bump, so a call from the wrong execution domain has
        # to be rejected before its side effects reach the section being built.
        if op_name is not None:
            self._validate_op_scope(op_name, call)

        result = self._route_ir_node_method(call)
        if result is not None:
            return result

        result = self._route_channel_method(call)
        if result is not None:
            return result

        span = self.span_tracker.get_span(call)
        if op_name is None:
            raise NotSupported(
                f"Unsupported operation call: {ast.unparse(call)}",
                span=span,
                hint="Use pl.*, pl.simt.*, or pl.system.* operations",
                parser_retry=True,
            )

        is_simt_function = self._current_func_type in (
            ir.FunctionType.SimtVF,
            ir.FunctionType.SimtCallee,
            ir.FunctionType.SimtKernel,
        )
        if is_simt_function and not op_name.startswith("simt."):
            raise NotSupported(
                f"Operation '{op_name}' is not supported inside a SIMT function",
                span=span,
                hint=(
                    "The current SIMT slice supports SIMT context queries, scalar expressions, restricted "
                    "loops, and scalar Tensor/Tile accesses."
                ),
                parser_retry=True,
            )
        elif not is_simt_function and op_name.startswith("simt."):
            raise InvalidOperation(
                f"pl.{op_name}() can only be used inside a SIMT function",
                span=span,
                hint='Move SIMT-context-dependent logic into @pl.vector_function(mode="simt").',
            )

        ast_handler = _AST_OP_REGISTRY.get(op_name)
        if ast_handler is not None:
            return self._dispatch_ast_op(op_name, call, ast_handler)
        op_func = _OP_REGISTRY.get(op_name)
        if op_func is not None:
            return self._dispatch_op(op_name, call, op_func)
        if op_name.startswith("vf."):
            raise InvalidOperation(
                f"Unknown vf operation: '{op_name}'",
                span=span,
                parser_retry=True,
            )
        raise NotSupported(f"Unsupported operation call: {ast.unparse(call)}", span=span, parser_retry=True)

    def _dispatch_ast_op(self, op_name: str, call: ast.Call, ast_handler: Callable):
        """Check declared parameters, then call the registered AST implementation."""
        span = self.span_tracker.get_span(call)
        self._check_declared_params(op_name, call, span)
        with current_span(span, self.span_tracker.argument_spans(call)):
            return ast_handler(self, call)

    def _dispatch_op(self, op_name: str, call: ast.Call, op_func: Callable):
        """Check and parse arguments, then call the registered implementation."""
        span = self.span_tracker.get_span(call)
        self._check_declared_params(op_name, call, span)
        args = [self.parse_expression(node) for node in call.args]
        kwargs = {keyword.arg: self.parse_expression(keyword.value) for keyword in call.keywords}
        argument_spans = self.span_tracker.argument_spans(call)
        self._check_declared_types(op_name, args, kwargs, span, argument_spans)
        with current_span(span, argument_spans):
            result = op_func(self, *args, **kwargs, span=span)
        self._record_op_access(
            op_name,
            [(position, None, arg) for position, arg in enumerate(args)]
            + [(None, keyword, value) for keyword, value in kwargs.items()],
            tuple(kwargs),
            span,
        )
        return result

    def _mx_slot_addrs(self, value: ir.Expr) -> tuple[int, ...] | None:
        if self.is_runtime_slot(value):
            refs = self.buffer_refs_of(value)
            facts = self.parse_facts.buffer_decls.get(refs[0].buffer) if len(refs) == 1 else None
            return facts.addrs if facts is not None else None
        memref = getattr(value.type, "memref", None)
        address = getattr(memref, "addr", None)
        if not isinstance(address, ir.ConstInt):
            return None
        return (int(address.value),)

    def _check_mx_plane_slots(self, op_name: str, args: tuple, kwargs: dict, span: ir.Span) -> None:
        """Check runtime slot layouts after operands have been parsed exactly once."""
        first = 1 if op_name == "matmul_mx" else 2
        for offset, role in enumerate(("scale_a", "scale_b")):
            plane_index = first + 2 + offset
            plane = args[plane_index] if len(args) > plane_index else kwargs[role]
            data = args[first + offset]
            if not self.is_runtime_slot(data) and not self.is_runtime_slot(plane):
                continue  # The builder already checks the selected static addresses.
            data_addrs = self._mx_slot_addrs(data)
            plane_addrs = self._mx_slot_addrs(plane)
            if data_addrs is None or plane_addrs is None:
                continue
            expected = tuple(address >> 4 for address in data_addrs)
            # A fixed plane may serve several slots that alias the same data address.
            if plane_addrs != expected and not (len(set(expected)) == 1 and set(plane_addrs) == set(expected)):
                raise InvalidVal(
                    f"{op_name}: every slot of {role} must match its data slot address >> 4; "
                    f"got {[hex(a) for a in plane_addrs]}, expected {[hex(a) for a in expected]}",
                    span=span,
                    hint="Pair runtime data/scale selections by slot; give the scale group explicit addresses",
                )

    def _check_declared_params(self, op_name: str, call: ast.Call, span: ir.Span) -> None:
        """Check which parameters the call names and how many it passes, before evaluation."""
        _argument_checker.check_declared_params(op_name, call, span, self.span_tracker)

    def _check_declared_types(
        self, op_name: str, args: list, kwargs: dict, span: ir.Span, argument_spans: dict
    ) -> None:
        """Check the parsed arguments against the roles their declaration names."""
        _argument_checker.check_declared_types(
            op_name, args, kwargs, span, argument_spans,
        )

    def _validate_op_scope(self, op_name: str, call: ast.Call) -> None:
        """Check VF execution scope before parsing operands or emitting IR.

        Every dispatch in parse_op_call runs behind this check, the tile-group
        route included: those routes lower their own operands and emit IR, so a
        check placed after them would reject the call only once its side effects
        already sat in the section being built.

        VF assignments bypass normal call dispatch and must also use this check.
        Match VF instructions by namespace so newly added operations inherit
        the same restriction.
        """
        span = self.span_tracker.get_span(call)
        if self.target == ir.SectionKind.Vector and op_name in {
            "system.bar_m", "system.bar_mte1", "system.bar_fix"
        }:
            raise NotSupported(
                f"pl.{op_name}() requires a Cube section; the current target is Vector",
                span=span,
                hint="Place the barrier with its Cube operations; use cross-core synchronization for a handover",
            )
        in_vf = self._current_func_type in (ir.FunctionType.SimdVF, ir.FunctionType.SimdCallee)
        if op_name.startswith("vf."):
            if not in_vf:
                raise InvalidOperation(
                    f"Operation '{op_name}' can only be used inside @pl.vector_function",
                    span=span,
                    hint="Move VF register operations into a @pl.vector_function helper and call it from the kernel.",
                )
            return
        if not in_vf:
            return
        if op_name in self._TILE_GROUP_ACCESSORS:
            raise InvalidOperation(
                f"Tile-group accessor '.{op_name}()' cannot be used inside @pl.vector_function",
                span=span,
                hint="Select the tile in the calling kernel and pass it to the vector function as an argument.",
            )
        if op_name not in self._VF_SCALAR_PL_OPS:
            raise NotSupported(
                f"Operation 'pl.{op_name}' is not supported inside @pl.vector_function",
                span=span,
                hint="VF bodies support vf.* operations, pl.range, and scalar pl.min/pl.max/pl.const. "
                "Move other pl.* operations outside the vector function and pass their results as arguments.",
            )

    def require_const_value(self, value, key, *, expects, hint, check=None, error=InvalidType):
        """Python value of an expression that has to be known while parsing.

        The single compile-time accessor: take the parse-time value of *value*,
        and report one consistent diagnostic when it has none or fails the caller's
        *check*. Callers add only what is specific to them — what the position
        expects and how to fix it.

        *value* is the caller's already parsed expression. A handler that
        parsed its arguments up front already holds it, and re-parsing to reach the
        check is not free — ``parse_expression`` can emit IR and falls back to
        evaluating the expression in Python. Prefer the call-site span for *key*,
        falling back to the value's IR span when that argument is not recorded.
        """
        found, constant = ExprEvaluator.ir_to_python_value(value)
        if not found or (check is not None and not check(constant)):
            span = argument_span(key, getattr(value, "span", None))
            text = self.span_tracker.get_source_text(span)
            detail = f", got '{text}'" if text else ""
            raise error(
                f"'{key}' must be a compile-time {expects}{detail}",
                span=span, hint=hint,
            )
        return constant

    def _parse_simt_template_call(self, local_name: str, fn: Callable, call: ast.Call) -> ir.Expr:
        """Instantiate and call one helper @pl.vector_function(mode="simt") template."""
        span = self.span_tracker.get_span(call)
        if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise InvalidArgument(
                'Helper @pl.vector_function(mode="simt") calls accept positional arguments only',
                span=span,
            )
        args = [self.parse_expression(arg) for arg in call.args]
        callee = self._instantiate_simt_function(local_name, fn, args, span)
        self._validate_vf_function_arguments(callee, args, call.args, span, kind="SIMT function")
        return self._make_call_with_return_type(ir.Op(callee.name), args, callee.return_types, span)

    def _instantiate_simt_function(
        self,
        local_name: str,
        fn: Callable,
        args: list[ir.Expr],
        span,
    ) -> ir.Function:
        """Parse a marked SIMT callable once its actual argument types are known."""
        from .decorator import get_simt_max_threads

        max_threads = get_simt_max_threads(fn)
        launchable = max_threads is not None
        if launchable:
            if not _is_int(max_threads):
                raise InvalidType("max_threads must be an integer")
            check_in_range(max_threads, 1, _MAX_SIMT_THREADS, subject="max_threads", error=ValueError)

        func_type = ir.FunctionType.SimtVF if launchable else ir.FunctionType.SimtCallee
        cached = self.vf_func_cache.get((id(fn), func_type, ()))
        if cached is not None:
            self._register_vf_external(cached, span)
            return cached

        parsed, _params = self._instantiate_vf_body(
            local_name,
            fn,
            args,
            span,
            decorator_label='@pl.vector_function(mode="simt")',
            recursion_label='helper @pl.vector_function(mode="simt")',
            role="SIMT function" if launchable else "SIMT callee",
            func_type=func_type,
            void_return_only=launchable,
            void_return_context='@pl.vector_function(mode="simt", max_threads=...)',
        )
        _validate_simt_parameters(
            parsed, "SIMT function" if launchable else "SIMT helper", entry_type=self._simt_entry_type
        )
        if launchable:
            result = ir.Function(
                parsed.name,
                list(parsed.params),
                list(parsed.return_types),
                parsed.body,
                parsed.span,
                func_type,
                parsed.entry,
                {"max_threads": max_threads},
            )
        else:
            result = parsed
        self._register_vf_external(result, span)
        self.vf_func_cache[(id(fn), func_type, ())] = result
        return result

    def _instantiate_vf_body(
        self,
        local_name: str,
        fn: Callable,
        args: list[ir.Expr],
        span,
        *,
        decorator_label: str,
        recursion_label: str,
        role: str,
        func_type: ir.FunctionType,
        void_return_only: bool,
        void_return_context: str,
    ):
        """Shared instantiation core for SIMT/SIMD VF templates.

        Checks recursion, retrieves the template source, validates the
        parameter list, and parses the body into a standalone function
        prototype. Returns ``(parsed, params)``: ``parsed`` is the prototype
        (params/return types/body) and ``params`` the AST parameter list.
        """
        from ._ast_parser import ASTParser

        if id(fn) in self.vf_call_stack:
            raise NotSupported(
                f"Recursive {recursion_label} call involving {fn.__name__!r} is not supported",
                span=span,
            )

        source_file, source_lines, line_offset, col_offset, func_def = self._retrieve_function_source(
            local_name,
            fn,
            span,
            decorator_label,
        )
        func_args = func_def.args
        if (
            func_args.posonlyargs
            or func_args.vararg is not None
            or func_args.kwarg is not None
            or func_args.kwonlyargs
            or func_args.defaults
            or func_args.kw_defaults
        ):
            raise NotSupported(
                f"{role} '{fn.__name__}' only supports required positional parameters",
                span=span,
            )
        params = list(func_args.args)
        if len(args) != len(params):
            raise InvalidVal(
                f"{role} '{fn.__name__}' expects {len(params)} argument(s), got {len(args)}",
                span=span,
                parser_retry=True,
            )

        const_param_values = None
        if func_type in (ir.FunctionType.SimdVF, ir.FunctionType.SimdCallee):
            const_param_values = {
                param.arg: arg for param, arg in zip(params, args) if _is_const_arg(arg)
            } or None
        # A parse recording accesses for the automatic pipeline collects the ops of the whole
        # kernel in one place, and a vector function is called as one op: the ops of its body
        # belong to the statement holding the call, so they are parsed into the caller's facts
        # under the caller's recording site.
        record_accesses = self.auto_pipeline_enabled and func_type in (
            ir.FunctionType.SimdVF,
            ir.FunctionType.SimdCallee,
        )
        callsite_param_buffers = (
            {param.arg: self.buffer_refs_of(actual) for param, actual in zip(params, args)}
            if record_accesses
            else None
        )

        parser = ASTParser(
            source_file,
            source_lines,
            ir.SectionKind.Vector,
            line_offset,
            col_offset,
            closure_vars=self._build_function_closure(fn),
            debug_info=self.debug_info,
            tilingkey_consts=self._tilingkey_consts,
            datatype_consts=self._datatype_consts,
            void_return_only=void_return_only,
            void_return_context=void_return_context,
            allow_early_return=True,
            auto_pipeline=record_accesses,
        )
        parser.external_funcs = self.external_funcs
        parser.vf_func_cache = self.vf_func_cache
        parser.vf_call_stack = self.vf_call_stack
        parser._simt_entry_type = self._simt_entry_type
        if record_accesses:
            parser.parse_facts = self.parse_facts
            parser._access_site = self._access_site

        self.vf_call_stack.append(id(fn))
        try:
            parsed = parser.parse_function(
                func_def,
                func_type=func_type,
                callsite_param_types={param.arg: actual.type for param, actual in zip(params, args)},
                const_param_values=const_param_values,
                callsite_param_buffers=callsite_param_buffers,
            )
        finally:
            self.vf_call_stack.pop()

        return parsed, params

    def _register_vf_external(self, func: ir.Function, span) -> None:
        """Register one instantiated SIMT/SIMD VF function in the enclosing Program."""
        if func.name in self.external_funcs and self.external_funcs[func.name] is not func:
            raise InvalidOperation(
                f"Conflicting external functions with name '{func.name}'",
                span=span,
            )
        self.external_funcs[func.name] = func

    def _validate_vf_function_arguments(
        self,
        callee: ir.Function,
        args: list[ir.Expr],
        arg_nodes: list[ast.expr],
        span,
        *,
        kind: str,
    ) -> None:
        """Validate call-site arguments against an instantiated VF signature."""
        if len(args) != len(callee.params):
            raise InvalidVal(
                f"{kind} '{callee.name}' expects {len(callee.params)} argument(s), got {len(args)}",
                span=span,
                parser_retry=True,
            )
        for param, actual, actual_node in zip(callee.params, args, arg_nodes):
            _check_type_compatible(
                param.type,
                actual.type,
                what=f"{kind} parameter",
                name=param.name,
                span=self.span_tracker.get_span(actual_node),
            )

    def _parse_simd_template_call(
        self,
        local_name: str,
        fn: Callable,
        call: ast.Call,
        func_type: ir.FunctionType,
    ) -> tuple[ir.Function, list[ir.Expr], ir.Span]:
        """Shared body of a SIMD VF call: parse args, instantiate the callee as
        *func_type* and validate the runtime arguments. The caller assembles the
        call form that matches its own function type: ``_parse_simd_kernel_call``
        builds the registered "simd.call" op (the SIMD analog of "simt.launch"),
        ``_parse_simd_inbody_call`` a direct named call (like SIMT callee calls).
        """
        span = self.span_tracker.get_span(call)
        if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise InvalidArgument(
                '@pl.vector_function calls accept positional arguments only',
                span=span,
            )
        args = [self.parse_expression(arg) for arg in call.args]

        callee = self._instantiate_simd_function(local_name, fn, args, span, func_type)
        runtime_args = [a for a in args if not _is_const_arg(a)]
        runtime_arg_nodes = [n for n, a in zip(call.args, args) if not _is_const_arg(a)]
        self._validate_vf_function_arguments(callee, runtime_args, runtime_arg_nodes, span, kind="VF function")
        return callee, runtime_args, span

    def _parse_simd_kernel_call(self, local_name: str, fn: Callable, call: ast.Call) -> ir.Expr:
        """Kernel-side SIMD VF call: the registered "simd.call" op carries the
        callee name as a kwarg. Kernel-side SIMD functions are always void, so
        the call is assembled directly like simt.launch (no return type to
        attach)."""
        callee, runtime_args, span = self._parse_simd_template_call(
            local_name, fn, call, ir.FunctionType.SimdVF
        )
        return ir.create_op_call("simd.call", runtime_args, {"callee": callee.name}, span)

    def _parse_simd_inbody_call(self, local_name: str, fn: Callable, call: ast.Call) -> ir.Expr:
        """In-body SIMD VF call: a direct named call (like SIMT callee calls),
        carrying the callee's return types."""
        callee, runtime_args, span = self._parse_simd_template_call(
            local_name, fn, call, ir.FunctionType.SimdCallee
        )
        return self._make_call_with_return_type(ir.Op(callee.name), runtime_args, callee.return_types, span)

    def _instantiate_simd_function(
        self,
        local_name: str,
        fn: Callable,
        args: list[ir.Expr],
        span,
        func_type: ir.FunctionType,
    ) -> ir.Function:
        """Parse a SIMD vector-function once per callable/type/constant specialization.

        Later call sites reuse the cached function and are validated against its
        parameter types by ``_parse_simd_template_call`` (the same contract SIMT
        follows). A tile argument is not part of the specialization: it is passed as a
        parameter, so one function serves every call site.

        A parse that records accesses for the automatic pipeline instantiates the body at
        every call site even so, and keeps the cached function. Which buffer a tile argument
        reaches is a fact about the call rather than about the function, and it is recorded
        only while the body is parsed, so a cached call site would otherwise contribute no
        accesses at all. Such a parse discards the IR it builds, hence the prototype of a
        repeat is dropped and the cached function is the one returned.
        """
        const_flags = [_is_const_arg(arg) for arg in args]
        const_spec = tuple(
            (idx, type(arg).__name__, arg.value)
            for idx, arg in enumerate(args)
            if const_flags[idx]
        )
        cache_key = (id(fn), func_type, const_spec)
        cached = self.vf_func_cache.get(cache_key)
        if cached is not None and not self.auto_pipeline_enabled:
            self._register_vf_external(cached, span)
            return cached

        parsed, params = self._instantiate_vf_body(
            local_name,
            fn,
            args,
            span,
            decorator_label='@pl.vector_function',
            recursion_label='@pl.vector_function',
            role="VF function" if func_type == ir.FunctionType.SimdVF else "VF callee",
            func_type=func_type,
            void_return_only=True,
            void_return_context='@pl.vector_function',
        )
        if cached is not None:
            # The body was parsed for this call site's accesses; the function is the cached one.
            self._register_vf_external(cached, span)
            return cached
        specialized_name = parsed.name
        if const_spec:
            suffix = "_".join(
                f"{param.arg}_{_const_name_value(arg)}"
                for param, arg in zip(params, args)
                if _is_const_arg(arg)
            )
            specialized_name = f"{parsed.name}__{suffix}"
        result = ir.Function(
            specialized_name,
            list(parsed.params),
            list(parsed.return_types),
            parsed.body,
            parsed.span,
            func_type,
            parsed.entry,
        )
        self._register_vf_external(result, span)
        self.vf_func_cache[cache_key] = result
        return result

    def _inline_template(self, func_name: str, fn: Callable, span) -> _InlineFunctionTemplate:
        template = self.inline_func_cache.get(id(fn))
        if template is not None:
            return template
        source_info = self._retrieve_function_source(
            func_name,
            fn,
            span,
            "an annotated Python helper",
        )
        template = _InlineFunctionTemplate(
            func_def=source_info[4],
            closure_vars=self._build_function_closure(fn),
            span_tracker=SpanTracker(source_info[0], source_info[1], source_info[2], source_info[3]),
        )
        self.inline_func_cache[id(fn)] = template
        return template

    def _bind_inline_arguments(
        self,
        func_name: str,
        func_def: ast.FunctionDef,
        call: ast.Call,
        span,
    ) -> dict[str, tuple[ir.Expr, ast.expr]]:
        """Bind one positional argument to each helper parameter."""
        args = func_def.args
        if self._has_unsupported_inline_params(args):
            raise NotSupported(
                f"Inline function '{func_name}' only supports positional parameters with optional defaults",
                span=span,
            )
        if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise NotSupported(
                f"Call to inline function '{func_name}' only supports plain positional arguments",
                span=span,
            )

        params = self._inline_param_list(func_def)
        required_count = len(params) - len(args.defaults)
        if not required_count <= len(call.args) <= len(params):
            raise InvalidArgument(
                f"Function '{func_name}' expects {required_count} to {len(params)} positional argument(s), "
                f"got {len(call.args)}",
                span=span,
                parser_retry=True,
            )
        argument_nodes = [*call.args, *args.defaults[len(call.args) - required_count:]]
        bound = {param.arg: (self.parse_expression(arg), arg) for param, arg in zip(params, argument_nodes)}
        for param in params:
            actual = bound[param.arg][0]
            if param.annotation is None or self.resolve_tiling_class(param.annotation):
                continue
            if self.type_resolver.annotation_has_shape_policy(param.annotation):
                self.type_resolver.validate_policy_parameter_type(param.annotation, param.arg, actual.type)
            else:
                annotated = self.type_resolver.resolve_param_type(param.annotation)
                _check_type_compatible(
                    annotated,
                    actual.type,
                    what="Parameter",
                    name=param.arg,
                    span=span,
                )
        return bound

    def _implicit_func_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr | None:
        """Expand a Python helper body directly into the caller's IR builder.

        For SIMT functions, routes to ``_parse_simt_template_call`` instead of
        inlining. For SIMD VF functions, routes by caller function type to
        ``_parse_simd_kernel_call`` / ``_parse_simd_inbody_call`` to emit a
        standalone ``__simd_vf__``/``__simd_callee__`` function.
        For regular callables, expands the body directly.
        """
        from .decorator import get_simt_max_threads, is_simt_function, is_vector_function

        if is_simt_function(fn):
            if self._current_func_type in (
                ir.FunctionType.SimtVF, ir.FunctionType.SimtCallee, ir.FunctionType.SimtKernel
            ):
                if get_simt_max_threads(fn) is not None:
                    raise InvalidVal(
                        f"SIMT helper '{func_name}' must omit max_threads in "
                        '@pl.vector_function(mode="simt")',
                        span=self.span_tracker.get_span(call),
                        parser_retry=True,
                    )
                return self._parse_simt_template_call(func_name, fn, call)
            raise InvalidOperation(
                f"SIMT function '{func_name}' cannot be called directly outside a SIMT function",
                span=self.span_tracker.get_span(call),
            )

        if is_vector_function(fn):
            if self._current_func_type in (ir.FunctionType.SimdVF, ir.FunctionType.SimdCallee):
                return self._parse_simd_inbody_call(func_name, fn, call)
            return self._parse_simd_kernel_call(func_name, fn, call)

        span = self.span_tracker.get_span(call)
        in_simd_vf = self._current_func_type in (ir.FunctionType.SimdVF, ir.FunctionType.SimdCallee)
        if in_simd_vf:
            # VF functions route to _parse_simd_inbody_call above, so any
            # helper reaching the inline path inside a VF body is a plain
            # Python function, which cannot be spliced into a VF body.
            raise InvalidOperation(
                f"Vector function cannot call non-vector inline function '{func_name}'",
                span=span,
            )
        template = self._inline_template(func_name, fn, span)
        if id(fn) in self.inline_call_stack:
            raise InvalidOperation(
                f"Recursive inline function call detected for '{func_name}'",
                span=span,
            )

        old_closure = self.expr_evaluator.closure_vars
        self.expr_evaluator.closure_vars = {**template.closure_vars, **old_closure}
        try:
            bound = self._bind_inline_arguments(func_name, template.func_def, call, span)
            params = self._inline_param_list(template.func_def)
            param_names = {param.arg for param in params}
            assigned_names = self._inline_assigned_names(template.func_def)
            inline_id = self.inline_counter
            self.inline_counter += 1
            prefix = f"__inline_{inline_id}_"
            return_val_name = f"{prefix}return_val"
            returned_name = f"{prefix}returned"
            local_names = param_names | assigned_names
            # The helper may itself use `return_val` or `returned`. Renaming
            # those locals must not capture the synthetic return value/flag.
            used_names = {f"{prefix}{name}" for name in local_names}
            used_names.update(node.id for node in ast.walk(template.func_def) if isinstance(node, ast.Name))
            while return_val_name in used_names:
                return_val_name += "_"
            while returned_name in used_names:
                returned_name += "_"
            # A parameter the helper rebinds must first become a real local IR
            # value; binding it directly to the caller expression leaves no
            # version for SSA to merge when the assignment is nested in control flow.
            reassigned_params = param_names & assigned_names
            func_def = copy.deepcopy(template.func_def)
            renamer = _InlineLocalRenamer(local_names, prefix)
            body = [renamer.visit(stmt) for stmt in func_def.body if not self._is_docstring(stmt)]
            body = _InlineReturnLowerer(return_val_name, returned_name).lower(body, func_def)

            # The ops this expansion emits are recorded under the call, which is also
            # registered here as expanded.
            old_access_site = self._access_site
            if not self.inline_call_stack:
                self._access_site = (call.lineno, call.col_offset)
                self.parse_facts.op_accesses.setdefault(self._access_site, [])
            self.inline_call_stack.append(id(fn))
            self.scope_manager.enter_scope("inline")

            old_span_tracker = self.span_tracker
            self.span_tracker = template.span_tracker

            try:
                for param in params:
                    expr, _arg_node = bound[param.arg]
                    renamed_name = f"{prefix}{param.arg}"
                    # A parameter that the helper rebinds must first become a
                    # real local IR value.  Binding it directly to the caller
                    # expression leaves no version for SSA to merge when the
                    # assignment is nested in an if/while.
                    if param.arg in reassigned_params:
                        value = self.builder.let(renamed_name, expr, span=span)
                        self._transfer_tile_group_metadata(value, expr)
                    else:
                        value = expr
                    self.scope_manager.define_var(renamed_name, value, allow_redef=True)
                    self._update_var_envs(value, expr)

                undefined_return = self.builder.var(return_val_name, ir.NoneType.get(), span)
                self._mark_empty_control_flow_value(undefined_return)
                self.scope_manager.define_var(
                    return_val_name, undefined_return, allow_redef=True, span=span
                )

                self._parse_statement_list(body)
                return self.lookup_expr_by_name(return_val_name)
            finally:
                self.span_tracker = old_span_tracker
                self.scope_manager.exit_scope(leak_vars=False)
                self.inline_call_stack.pop()
                self._access_site = old_access_site
        finally:
            self.expr_evaluator.closure_vars = old_closure
