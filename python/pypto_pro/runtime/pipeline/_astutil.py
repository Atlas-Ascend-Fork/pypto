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

"""What every pipeline module needs and none of them owns: the generated-code name
vocabulary, and the syntactic forms the transform has to recognise.

A leaf on purpose — it imports nothing from its siblings and knows nothing about
PipelineInfo, the sync graph or the transform. That is what lets ``_validate`` import
from here without a cycle, and what keeps one spelling of "a call to a stage" or "a slot
accessor" for every module that asks. Add nothing here that needs a sibling.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import textwrap

# ---------------------------------------------------------------------------
# Generated-code names
#
# The ``_pl_`` prefix is reserved for the transform (validate_names refuses a kernel that
# already uses one). Each name below is written by one module and read back by another, so
# it is spelled once, here.

# Which task the data in a ctx slot belongs to. The analyzer places it in the ctx layout,
# the transformer both assigns it and reads it back as the event-id index.
#
# This is the ctx FIELD only. The running counter that fills it is a variable of the same
# name with a different lifetime: the field travels with a snapshot, so a delayed stage
# reading `ctx._pl_task_id` gets ITS task's number, while the variable belongs to the loop
# and moves on every beat. Several pipeline loops each need their own counter, while the
# fields stay one per ctx struct. See task_id_var below.
PL_TASK_ID_FIELD = "_pl_task_id"
# The validity flag. Prefixed like every framework field so a user struct field named
# `is_valid` cannot collide with it.
PL_IS_VALID_FIELD = "_pl_is_valid"
# Marks a stage argument that is a whole struct: the ctx slot is passed in its place, so
# there is no single field to name (its fields are ctx fields under their own names).
PL_STRUCT_ARG = "_pl_struct_arg"


def slot_index_field(group: str) -> str:
    """Ctx field (and variable) name holding a group's current slot index."""
    return f"_pl_idx_{group}"


def task_id_var(loop_index: int) -> str:
    """The running task counter VARIABLE of one pipeline loop.

    Distinct from PL_TASK_ID_FIELD, which names the ctx field this counter is snapshotted
    into — see the comment there. Loop 0 keeps the bare name.
    """
    return PL_TASK_ID_FIELD if loop_index == 0 else f"{PL_TASK_ID_FIELD}_{loop_index}"


# ---------------------------------------------------------------------------
# Syntax recognition
# ---------------------------------------------------------------------------


def call_name(call: ast.Call) -> str:
    """The callee of a plainly-called function: ``f(...)`` -> ``"f"``; anything else -> ``""``.

    Deliberately blind to ``obj.f(...)``. Every caller uses this to recognise a call to a name
    bound in the kernel's closure — a ``@stage`` function — while an attribute call is a
    method on some object that merely shares the name.
    """
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
        return call.func.id
    return ""


# The accessors that resolve a tile group handle to one of its tiles. Only ``next``
# advances the group's cursor; the others read it where it stands, which is why the kind
# is reported rather than just the group.
_GROUP_ACCESSORS = ("next", "current", "previous")


def slot_accessor(node: ast.expr) -> tuple[str, str] | None:
    """``(group, kind)`` if ``node`` selects a slot of a tile group, else None.

    Both spellings are recognised: ``group.next()/current()/previous()`` gives that accessor
    as the kind, and ``group[i]`` gives ``"index"``.

    The group name is returned unfiltered — callers that only care about declared groups
    check membership themselves, since they do not all hold the same set.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _GROUP_ACCESSORS:
            if isinstance(func.value, ast.Name):
                return func.value.id, func.attr
        return None  # any other call is not an accessor, and must not fall through
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id, "index"
    return None


def is_vf_function(func_def: ast.FunctionDef) -> bool:
    """Return whether a function carries a SIMD vector-function decorator.

    Only the decorator counts: a plain function that merely calls a vector function is a
    different thing. Bare @pl.vector_function and explicit mode="simd" are recognized.
    """
    for dec in func_def.decorator_list:
        call = dec if isinstance(dec, ast.Call) else None
        target = call.func if call is not None else dec
        is_vector_decorator = (
            isinstance(target, ast.Attribute) and target.attr == "vector_function"
        ) or (isinstance(target, ast.Name) and target.id == "vector_function")
        if not is_vector_decorator:
            continue
        if call is None:
            return True
        mode = next((keyword.value for keyword in call.keywords if keyword.arg == "mode"), None)
        return isinstance(mode, ast.Constant) and mode.value == "simd"
    return False


def stmt_header(stmt: ast.stmt) -> ast.AST:
    """The part of ``stmt`` that names it without its body: the condition of an ``if`` or
    ``while``, the iterable of a ``for``, the context of a ``with``. Any other statement is
    returned as is.
    """
    if isinstance(stmt, (ast.If, ast.While)):
        return stmt.test
    if isinstance(stmt, ast.For):
        return stmt.iter
    if isinstance(stmt, ast.With) and stmt.items:
        return stmt.items[0].context_expr
    return stmt


def eval_const(node: ast.expr, closure_vars: dict):
    """Evaluate a constant expression node using closure_vars as namespace.
    Returns the value, or None if it cannot be evaluated (e.g. references a
    runtime variable)."""
    try:
        code = compile(ast.Expression(body=node), "<addr>", "eval")
        return eval(code, {"__builtins__": {}}, dict(closure_vars))
    except Exception:
        return None


def gm_slot_size(shape: ast.expr, stride: ast.expr | None, consts: dict) -> ast.expr | None:
    """Elements one slot of a GM buffer spans: ``max(shape[i] * stride[i])``, or
    ``prod(shape)`` without a stride.

    A constant when it evaluates statically; None when shape or stride is neither written
    as a list nor resolvable from ``consts``.
    """
    dims = _list_elts(shape, consts)
    steps = _list_elts(stride, consts) if stride is not None else None
    if dims is None or (stride is not None and steps is None):
        return None
    if steps is None:
        size = dims[0]
        for dim in dims[1:]:
            size = ast.BinOp(left=size, op=ast.Mult(), right=dim)
    else:
        spans = [ast.BinOp(left=d, op=ast.Mult(), right=st) for d, st in zip(dims, steps)]
        size = spans[0]
        for span in spans[1:]:
            size = ast.Call(func=ast.Name(id="max", ctx=ast.Load()), args=[size, span], keywords=[])
    size = ast.fix_missing_locations(copy.deepcopy(size))
    value = eval_const(size, {**consts, "max": max})
    if isinstance(value, int) and not isinstance(value, bool):
        return ast.Constant(value=value)
    return size


def _list_elts(node: ast.expr, consts: dict) -> list | None:
    """The elements of a list written literally, or of one ``consts`` resolves it to."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return node.elts
    value = eval_const(node, consts)
    if isinstance(value, (list, tuple)) and value:
        return [ast.Constant(value=v) for v in value]
    return None


@contextlib.contextmanager
def spans_in_function(fn):
    """Publish a span resolver for the tree :func:`get_funcdef` builds from *fn*.

    That tree carries the line numbers of the real file, while a kernel's tree is parsed
    from a dedented snippet and its resolver adds the kernel's offset back. Resolving a
    node of *fn*'s body with the kernel's resolver would add that offset a second time,
    so the body is given a resolver that adds none. Yields unchanged when *fn* has no
    reachable source, leaving the ambient resolver in place.
    """
    from pypto_pro._errors import source_lines_of, span_source
    from pypto_pro.language.parser._span_tracker import SpanTracker

    try:
        source_file = inspect.getsourcefile(fn) if fn is not None else None
        lines, _start = inspect.getsourcelines(fn) if source_file else ([], 0)
    except (OSError, TypeError):
        source_file = None
    if not source_file:
        yield
        return
    # get_funcdef dedents before parsing, so columns need the definition's own indent back.
    indent = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
    tracker = SpanTracker(source_file, source_lines_of(source_file) or [], 0, indent)
    with span_source(tracker.get_span):
        yield


def get_funcdef(fn) -> ast.FunctionDef | None:
    """The ``ast.FunctionDef`` for a Python function object, or None.

    Line numbers are shifted to the ones in the real file: re-parsing a source snippet numbers
    it from 1.
    """
    if fn is None:
        return None
    try:
        lines, start_lineno = inspect.getsourcelines(fn)
        mod = ast.parse(textwrap.dedent("".join(lines)))
        # getsourcelines is 1-based and so is the fresh parse, hence the -1.
        ast.increment_lineno(mod, start_lineno - 1)
        for node in mod.body:
            if isinstance(node, ast.FunctionDef):
                return node
    except (OSError, TypeError, SyntaxError):
        return None
    return None
