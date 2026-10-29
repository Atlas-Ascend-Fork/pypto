# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Struct parsing helpers for ASTParser (pl.struct / pl.struct_array / pl.make_tuple)."""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir
from pypto_pro.ir.op._op_registry import op_impl_ast

from ..._errors import InvalidArgument, InvalidOperation, InvalidType, InvalidVal, NotSupported

# C++ keywords. A struct type name or field name is emitted verbatim into the
# generated C++ (``class Name { int64_t field; }``, ``s.field``), so a name that
# is a valid Python identifier but a C++ keyword (``int``, ``true``, ``delete``)
# produces uncompilable C++. Reject such names in the frontend with a clear
# diagnostic instead of letting them reach codegen.
_CPP_KEYWORDS = frozenset({
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
    "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "compl", "concept", "const", "consteval", "constexpr", "constinit",
    "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected",
    "public", "register", "reinterpret_cast", "requires", "return", "short",
    "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
    "switch", "template", "this", "thread_local", "throw", "true", "try",
    "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual",
    "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
})


class StructParserMixin:
    """Mixin containing ``pl.struct`` / ``pl.struct_array`` / ``pl.make_tuple`` helpers."""

    def _check_cpp_identifier(self, name: str, what: str, span: ir.Span) -> None:
        """Reject a struct type name / field name that is a C++ keyword.

        The name is emitted verbatim into the generated C++, so a C++ keyword
        would produce uncompilable code. Reject it here with a clear diagnostic.
        """
        if name in _CPP_KEYWORDS:
            raise InvalidOperation(
                f"{what} '{name}' is a C++ keyword and cannot be used as a struct "
                f"type name or field name",
                span=span,
                hint=f"Rename it, e.g. '{name}_'",
            )

    def _check_scalar_or_array_field(self, field_name: str, elem: ir.Expr, span: ir.Span) -> None:
        """Reject a struct field value that is not a scalar or a uniform 1-D scalar array.

        A valid field is either a scalar (ScalarType) or a fixed-size array literal
        (MakeTuple of same-dtype scalars). Anything else -- a tensor/tile value, or a
        nested/multi-dimensional list -- would produce a ``fields=Unsupported``
        struct.create and later break codegen, so reject it here with a clear message.
        Nested named struct/tuple values are already rejected earlier via named_fields.
        """
        if not isinstance(elem, ir.MakeTuple):
            # Non-array field value: must be a scalar.
            if not isinstance(elem.type, ir.ScalarType):
                raise InvalidType(
                    f"struct field '{field_name}' must be a scalar or a fixed-size array; "
                    f"got a non-scalar value (e.g. tensor/tile)",
                    span=span,
                    hint="Use a scalar or a 1-D array of scalars.",
                )
            return
        # Array field literal: must be non-empty, one-dimensional, uniform scalar dtype.
        elems = elem.elements
        if not elems:
            raise InvalidVal(
                f"array field '{field_name}' is empty; a fixed-size array field "
                f"must have at least one element",
                span=span,
                hint="Provide initial values, e.g. [0, 0, 0, 0]",
            )
        for e in elems:
            if not isinstance(e.type, ir.ScalarType):
                raise InvalidType(
                    f"array field '{field_name}' has non-scalar elements; "
                    f"struct array fields must be one-dimensional with scalar elements",
                    span=span,
                    hint="Use a scalar or a 1-D array of scalars.",
                )
        first_dtype = elems[0].type.dtype
        for e in elems[1:]:
            if e.type.dtype != first_dtype:
                raise InvalidVal(
                    f"array field '{field_name}' has mixed element types; "
                    f"all elements must be the same scalar type",
                    span=span,
                    hint="Use a single dtype, e.g. [1.0, 2.5, 3.0] or [1, 2, 3]",
                )

    def _parse_struct_fields(
        self, keywords: list[ast.keyword], ctx: str, span: ir.Span
    ) -> tuple[list[str], list[ir.Expr]]:
        """Parse and validate ``field=value`` kwargs shared by struct / struct_array.

        ``ctx`` is the API name used in error messages ("pl.struct()" /
        "pl.struct_array()"). Returns (field_names, field value expressions).
        """
        field_names: list[str] = []
        elements: list[ir.Expr] = []
        for kw in keywords:
            if kw.arg is None:
                raise NotSupported(f"{ctx} does not support **kwargs", span=span)
            self._check_cpp_identifier(kw.arg, "struct field name", span)
            elem = self.parse_expression(kw.value, nested=True)
            if self.named_fields(elem):
                raise NotSupported(
                    f"{ctx} field '{kw.arg}' is a nested named tuple/struct, which is not "
                    f"supported; struct fields must be scalars or fixed-size arrays "
                    f"(list literals like [0, 0, 0, 0])",
                    span=self.span_tracker.get_span(kw.value),
                    hint="Use a scalar or a 1-D array of scalars.",
                )
            self._check_scalar_or_array_field(kw.arg, elem, span)
            field_names.append(kw.arg)
            elements.append(elem)
        return field_names, elements

    def _make_struct_create(
        self, elements: list[ir.Expr], struct_name: str, field_names: list[str], span: ir.Span
    ) -> ir.Expr:
        """Build a ``struct.create`` call and register its name + field names."""
        call = ir.create_op_call(
            "struct.create",
            elements,
            {"name": struct_name, "fields": field_names},
            span,
        )
        return self.register_struct_type(call, struct_name, field_names)

    # Keep AST registration: keyword names are user fields, including "span" and
    # "self", so framework parameters must not be injected into those fields.
    @op_impl_ast("make_tuple")
    def _parse_pl_make_tuple_expr(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if call.args:
            raise InvalidArgument(
                "pl.make_tuple() does not accept positional arguments; use keyword args",
                span=span,
            )
        if not call.keywords:
            raise InvalidArgument(
                "pl.make_tuple() requires at least one keyword argument (field=value)",
                span=span,
            )
        field_names: list[str] = []
        elements: list[ir.Expr] = []
        for kw in call.keywords:
            if kw.arg is None:
                raise NotSupported("pl.make_tuple() does not support **kwargs", span=span)
            field_names.append(kw.arg)
            elements.append(self.parse_expression(kw.value, nested=True))
        return self.make_named_tuple(elements, field_names, span)

    # Keep AST registration to preserve the string-literal name check and user
    # fields such as "span", which must not collide with framework parameters.
    @op_impl_ast("struct")
    def _parse_pl_struct_expr(self, call: ast.Call) -> ir.Expr:
        span = self.span_tracker.get_span(call)
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            raise InvalidVal(
                'pl.struct("Name", ...) requires exactly one string struct name as first argument',
                span=self.span_tracker.call_argument_span(call, 0, span),
                hint='Use pl.struct("Name", field1=val1, ...)',
            )
        struct_name = call.args[0].value
        self._check_cpp_identifier(struct_name, "struct type name", span)
        if not call.keywords:
            raise InvalidVal(
                'pl.struct("Name", ...) requires at least one keyword field',
                span=span,
            )
        field_names, elements = self._parse_struct_fields(call.keywords, "pl.struct()", span)
        return self._make_struct_create(elements, struct_name, field_names, span)

    # Keep AST registration to check size/name literals before parsing fields;
    # user field names such as "span" must remain independent of framework parameters.
    @op_impl_ast("struct_array")
    def _parse_struct_array_expr(self, call: ast.Call) -> ir.Expr:
        """Handle ``var = pl.struct_array(N, "StructName", field1=val1, ...)``.

        Lowers to N ``struct.create`` slots wrapped in a MakeTuple.
        The caller (_parse_name_assignment) emits the final let-binding.
        """
        span = self.span_tracker.get_span(call)
        if not call.args or not isinstance(call.args[0], ast.Constant):
            raise InvalidType(
                "pl.struct_array() requires an integer size as first argument",
                span=self.span_tracker.call_argument_span(call, 0, span),
                hint='Use pl.struct_array(N, "Name", field1=0, field2=0, ...)',
            )
        arr_size = call.args[0].value
        if not isinstance(arr_size, int) or arr_size < 1:
            raise InvalidType(
                f"pl.struct_array() size must be a positive integer, got {arr_size}",
                span=self.span_tracker.call_argument_span(call, 0, span),
            )
        if not (len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
            raise InvalidVal(
                'pl.struct_array(N, "Name", ...) requires a string struct name as second arg',
                span=self.span_tracker.call_argument_span(call, 1, span),
            )
        struct_name = call.args[1].value
        self._check_cpp_identifier(struct_name, "struct type name", span)
        if len(call.args) > 2:
            raise InvalidArgument(
                f"pl.struct_array() accepts exactly two positional arguments "
                f"(size and name), got {len(call.args)}",
                span=span,
            )
        if not call.keywords:
            raise InvalidVal(
                'pl.struct_array(N, "Name", ...) requires at least one keyword field',
                span=span,
            )
        var_name = self.current_target_name
        field_names, field_inits = self._parse_struct_fields(call.keywords, "pl.struct_array()", span)
        slot_vars: list[ir.Expr] = []
        for i in range(arr_size):
            slot_call = self._make_struct_create(field_inits, struct_name, field_names, span)
            slot_var = self.builder.let(f"{var_name}_{i}", slot_call, span=span)
            slot_vars.append(slot_var)
        result = ir.MakeTuple(slot_vars, span)
        self._struct_array_tuples.add(result)
        return result
