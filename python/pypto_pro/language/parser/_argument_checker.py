# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The two gates a gated call passes before its builder sees it.

Both read the same source of truth, the ``@_api_decl`` declarations, and differ in
when they run. ``check_declared_params`` runs before any argument is evaluated and
settles which parameters the call names and how many it passes, which is all its
written form can answer. ``check_declared_types`` runs after, on the parsed values,
and settles what each one turned out to be.
Whether a value is *legal* -- a dtype off the whitelist, two shapes that disagree --
is neither gate's business and stays in the builders.

A role is read from an annotation as it is *written*, never as typing resolves it:
the user-facing aliases in ``language/_api.py`` are all ``Any``, so the resolved form
says nothing. Anything a role does not describe is left alone -- an unannotated
parameter, ``Any``, a union with one unfamiliar branch -- because a role nobody wrote
down is not grounds to reject an argument the builder may well accept.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import re
from typing import Any, Callable

from pypto.pypto_impl import ir

from ..._errors import InvalidArgument, InvalidType
from .. import _api as _language_api
from .. import _simt_api, _system_api
from .._vf_api import Vf

# Where each gated namespace keeps its declarations. A call written without one --
# pl.add(...) -- is declared at the top level; a namespace this table does not name,
# such as the legacy mutex.*, is not gated at all.
_TOP_LEVEL_DECLARATIONS = _language_api
_NAMESPACE_DECLARATIONS = {
    "simt": _simt_api.Simt,
    "system": _system_api.System,
    "vf": Vf,
}


def spelled(op_name: str) -> str:
    """The op as a user writes it: vf.* is imported on its own, the rest hang off pl."""
    return op_name if op_name.startswith("vf.") else f"pl.{op_name}"


def declared_api(op_name: str) -> Callable | None:
    """The ``@_api_decl`` declaration for *op_name*, or None when it has none.

    Coverage follows the declaration files rather than ``_OP_REGISTRY``.
    """
    namespace, _, attribute = op_name.rpartition(".")
    declarations = _NAMESPACE_DECLARATIONS.get(namespace) if namespace else _TOP_LEVEL_DECLARATIONS
    if declarations is None:
        return None
    declared = getattr(declarations, attribute, None)
    return declared if getattr(declared, "__wrapped__", None) is not None else None


def _required_form(op_name: str, signature: inspect.Signature) -> str:
    """The shortest call the declaration accepts, keyword-only parameters included."""
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            required.append(name)
        elif parameter.kind is parameter.KEYWORD_ONLY:
            required.append(f"{name}=...")
    return f"{spelled(op_name)}({', '.join(required)})"


def _keyword_names(signature: inspect.Signature) -> list[str]:
    """Every parameter a caller may name."""
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    ]


# ---------------------------------------------------------------------------
# Gate 1 -- parameter names and counts, before any argument is evaluated
# ---------------------------------------------------------------------------


def check_declared_params(op_name: str, call: ast.Call, span: ir.Span, span_tracker: Any) -> None:
    """Bind the call's parameters to its declaration, before any argument is parsed.

    The rule is the declaration's own binding; AST nodes supply the written
    arguments without evaluating them. Callers run this after the execution-domain check, which
    reports a call from the wrong domain first, and after receiver-method routing,
    whose names no declaration covers. The binding is used only for validation.
    """
    for argument in call.args:
        if isinstance(argument, ast.Starred):
            raise InvalidArgument(
                f"{spelled(op_name)}() does not support argument unpacking", span=span_tracker.get_span(argument),
            )
    if any(keyword.arg is None for keyword in call.keywords):
        raise InvalidArgument(f"{spelled(op_name)}() does not support argument unpacking", span=span)
    declared = declared_api(op_name)
    if declared is None:
        return
    try:
        # Binding AST nodes checks only names/counts; it never evaluates them.
        declared.bind_declared(*call.args, **{kw.arg: kw.value for kw in call.keywords})
    except TypeError as error:
        _reject_declared_params(op_name, declared, call, span, span_tracker, error)


def _reject_declared_params(
    op_name: str,
    declared: Callable,
    call: ast.Call,
    span: ir.Span,
    span_tracker: Any,
    error: TypeError,
) -> None:
    """Report a declaration binding error at the argument that caused it.

    The declaration's binding checks the call; this function only locates the
    error and supplies a hint. Python compilation rejects repeated keywords.
    """
    signature = inspect.signature(declared)
    parameters = signature.parameters
    given = [keyword.arg for keyword in call.keywords]
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    unexpected = [name for name in given if name not in parameters]
    duplicated = [name for name in positional[:len(call.args)] if name in given]
    positional_only = [name for name, parameter in parameters.items() if parameter.kind is parameter.POSITIONAL_ONLY]
    as_keyword = [name for name in given if name in positional_only]
    takes_var_positional = any(p.kind is p.VAR_POSITIONAL for p in parameters.values())

    # Only the optional keyword-only parameters are worth naming: the required
    # ones already show up in the required form, written as ``name=...``.
    keyword_only = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is parameter.KEYWORD_ONLY and parameter.default is not inspect.Parameter.empty
    ]
    reason = str(error)
    if unexpected:
        marked: str | int | None = unexpected[0]
        # Binding reports whichever fault it meets first, and a keyword nobody
        # declared usually reads as a required one missing. Name the keyword the
        # caller actually wrote, so the reason and the mark agree.
        reason = f"got an unexpected keyword argument '{marked}'"
        names = _keyword_names(signature)
        close = difflib.get_close_matches(marked, [n for n in names if n not in given], n=1)
        suggestion = f"did you mean '{close[0]}'? " if close else ""
        takes = ", ".join(names) if names else "no arguments"
        hint = f"{suggestion}{spelled(op_name)}() takes {takes}"
    elif duplicated:
        marked = duplicated[0]
        hint = f"'{marked}' is already given as positional argument {positional.index(marked) + 1}"
    elif as_keyword:
        # Naming a positional-only parameter is a fault of the name, and the binding
        # already words it; only where to point is left to decide.
        marked = as_keyword[0]
        hint = f"write it by position: {_required_form(op_name, signature)}"
    elif not takes_var_positional and len(call.args) > len(positional):
        marked = len(positional)
        hint = _required_form(op_name, signature)
        if keyword_only:
            hint += f"; {', '.join(keyword_only)} must be passed by keyword"
    else:
        # An argument the call never wrote: the call itself is the only location.
        marked = None
        hint = _required_form(op_name, signature)
    if marked is None:
        located = span
    elif isinstance(marked, str):
        # Both keyword faults are faults of the name, so the name is what is marked.
        located = span_tracker.call_keyword_name_span(call, marked, span)
    else:
        located = span_tracker.call_argument_span(call, marked, span)
    raise InvalidArgument(f"{spelled(op_name)}() {reason}", span=located, hint=hint) from error


# ---------------------------------------------------------------------------
# Gate 2 -- what each parsed argument turned out to be
# ---------------------------------------------------------------------------

_NONE_TYPE = type(None)


# Every IR scalar shares one class, ScalarType, and carries what it really is in
# its dtype. A caller writing ``int`` means an integer, not "any scalar", so a
# scalar is described by its dtype family and these three stand for the families.
class _IntScalar:
    """An IR scalar whose dtype is an integer."""


class _FloatScalar:
    """An IR scalar whose dtype is a float."""


class _BoolScalar:
    """An IR scalar whose dtype is BOOL, which is neither integer nor float."""


# What each name written in a declaration accepts, as classes to match a value's
# kind against. Explicit scalar and sequence arguments arrive as IR expressions,
# regardless of whether they were written positionally or by keyword. Python
# representations are also accepted for defaults and direct builder calls.
_LEAF_TYPES: dict[str, tuple[type, ...]] = {
    "Tile": (ir.TileType,),
    "Tensor": (ir.TensorType,),
    # A group handle lowers to a MakeTuple, so a group is a TupleType here; that
    # it is a group of tiles rather than a struct is the builder's to tell apart.
    "TileGroup": (ir.TileType, ir.TensorType, ir.TupleType),
    "Struct": (ir.TupleType,),
    "Ptr": (ir.PtrType,),
    # Every scalar, written as a literal or lowered to an IR value. A bool is one
    # of them: whether it is a legal value in a given position -- an event id, a
    # loop bound -- is the builder's judgement, not this gate's.
    "Scalar": (_IntScalar, _FloatScalar, _BoolScalar, int, float, bool),
    "Shape": (ir.TupleType, list, tuple),
    "Offset": (ir.TupleType, list, tuple),
    # Python counts a bool as an int -- isinstance(True, int), min(True, 2) -- and
    # a declaration written in Python's own vocabulary says the same thing here.
    "int": (_IntScalar, _BoolScalar, int, bool),
    "float": (_FloatScalar, float),
    "bool": (_BoolScalar, bool),
    "str": (str,),
    "list": (ir.TupleType, list, tuple),
    "tuple": (ir.TupleType, list, tuple),
    "None": (_NONE_TYPE, ir.NoneType),
}

# Names a declaration writes that do not match the class they stand for.
_CLASS_ALIASES = {"DType": "DataType"}
_UNCHECKED = {"Any", "object"}
_NOTHING = (_NONE_TYPE, ir.NoneType)

_OPTIONAL_FORM = re.compile(r"^Optional\[(.*)\]$")
_UNION_FORM = re.compile(r"^Union\[(.*)\]$")
_SEQUENCE_FORM = re.compile(r"^(?:List|list|Sequence|tuple)\[.+\]$")


def _split_outside_brackets(text: str, separator: str) -> list[str]:
    """Split on *separator*, ignoring the ones nested inside square brackets."""
    parts: list[str] = []
    depth = 0
    current = ""
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        if character == separator and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += character
    parts.append(current.strip())
    return [part for part in parts if part]


def _named_class(name: str, declared: Callable) -> type | None:
    """The class a name stands for, resolved where the declaration was written.

    Resolution follows Python's own rule -- the declaring module -- and falls back to
    the IR module, which is where a name an aliased declaration file never imported
    (``DType``) actually lives.
    """
    if name == "TileType":
        # The declaration module binds this name to the TileType declaration rather
        # than to the class a caller passes, and that class is the frontend's, which
        # is a different class from the IR type of the same name.
        from pypto_pro.ir.op.block_ops import TileType

        return TileType
    wanted = _CLASS_ALIASES.get(name, name)
    written_in = getattr(declared, "__wrapped__", declared).__globals__
    for resolved in (written_in.get(wanted), getattr(ir, wanted, None)):
        if isinstance(resolved, type):
            return resolved
    return None


def _leaf_types(name: str, declared: Callable) -> tuple[type, ...] | None:
    if name in _UNCHECKED:
        return None
    if name in _LEAF_TYPES:
        return _LEAF_TYPES[name]
    if _SEQUENCE_FORM.match(name):
        # A sequence of anything is still a sequence; the element role is not checked.
        return _LEAF_TYPES["list"]
    named = _named_class(name, declared)
    return (named,) if named is not None else None


def accepted_types(annotation: str, declared: Callable) -> tuple[type, ...] | None:
    """The classes *annotation* accepts, or None when it names nothing worth checking.

    A union accepts any of its branches, so the branches' classes are pooled. One
    unrecognised branch leaves the whole annotation unchecked.
    """
    text = annotation.strip().strip("'\"")
    if not text or text in _UNCHECKED:
        return None
    if _SEQUENCE_FORM.match(text):
        return _LEAF_TYPES["list"]
    optional = _OPTIONAL_FORM.match(text)
    if optional:
        inner = accepted_types(optional.group(1), declared)
        return None if inner is None else tuple(dict.fromkeys(inner + _NOTHING))
    union = _UNION_FORM.match(text)
    if union:
        branches = _split_outside_brackets(union.group(1), ",")
    elif "|" in text:
        branches = _split_outside_brackets(text, "|")
    else:
        branches = [text]
    # An offset may be expressed as per-axis coordinates or as one scalar element
    # offset (for example, system.dcci). Keep bool out of that scalar branch even
    # though a plain ``int`` annotation follows Python's bool-is-an-int rule.
    offset_union = "int" in branches and any(_SEQUENCE_FORM.match(branch) for branch in branches)
    pooled: tuple[type, ...] = ()
    for branch in branches:
        types = (_IntScalar, int) if offset_union and branch == "int" else _leaf_types(branch, declared)
        if types is None:
            return None
        pooled += types
    return tuple(dict.fromkeys(pooled))


def _scalar_family(scalar_type: ir.ScalarType) -> type:
    """Which family an IR scalar's dtype belongs to."""
    dtype = scalar_type.dtype
    if dtype == ir.DataType.BOOL:
        return _BoolScalar
    return _IntScalar if dtype.is_int() else _FloatScalar


def value_kind(value: Any) -> type:
    """The one class that describes *value*, so it can be matched against a role.

    An IR expression is described by its IR type, everything else by its own type.
    A scalar is described one level deeper, by its dtype family, because that is
    the distinction a declaration writing ``int`` or ``float`` is drawing.
    """
    if isinstance(value, ir.Expr):
        value_type = getattr(value, "type", None)
        if isinstance(value_type, ir.ScalarType):
            return _scalar_family(value_type)
        if isinstance(value_type, ir.TileType):
            return ir.TileType
        return type(value_type)
    return type(value)


def describe_value(value: Any) -> str:
    """The value's kind as a user would recognise it, not its Python repr."""
    if value is None:
        return "None"
    if isinstance(value, ir.Expr):
        value_type = getattr(value, "type", None)
        if isinstance(value_type, ir.ScalarType):
            # Naming the class would say ScalarType for every scalar alike; the
            # dtype is what tells the caller which one they actually wrote.
            return f"a {value_type.dtype} scalar"
        if isinstance(value_type, ir.TileType):
            return "TileType"
        return type(value_type).__name__ if value_type is not None else "an expression"
    return type(value).__name__


def _written_as(kwargs: dict, name: str, signature: inspect.Signature) -> str | int:
    """Locate an argument using the parsed call's keyword names."""
    if name in kwargs:
        return name
    positional = [
        parameter_name
        for parameter_name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return positional.index(name) if name in positional else 0


def check_declared_types(
    op_name: str,
    args: list,
    kwargs: dict,
    span: ir.Span,
    argument_spans: dict,
) -> None:
    """Check parsed values without changing the arguments passed to the builder."""
    declared = declared_api(op_name)
    if declared is None:
        return
    signature = inspect.signature(declared)
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return  # the first gate already reported this, or the call is not gated
    for name, value in bound.arguments.items():
        parameter = signature.parameters[name]
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.annotation is inspect.Parameter.empty:
            continue
        allowed = accepted_types(str(parameter.annotation), declared)
        if allowed is None or value_kind(value) in allowed:
            continue
        marked = _written_as(kwargs, name, signature)
        raise InvalidType(
            f"{spelled(op_name)}: {name} expects {parameter.annotation}, got {describe_value(value)}",
            span=argument_spans.get(marked, span),
        )
