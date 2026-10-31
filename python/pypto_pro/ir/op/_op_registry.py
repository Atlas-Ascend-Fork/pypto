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

"""Op dispatch registry for CallParserMixin.

Kept in ir/op/ so that ir/op/*.py can use relative imports, avoiding the
circular-import cycle:
  ir/op/__init__.py -> block_ops.py -> language/parser/op_registry.py
    -> language/__init__.py -> pypto_pro.ir (partially initialized)
"""

from __future__ import annotations

from collections.abc import Callable

from ..._errors import InvalidOperation

_OP_REGISTRY: dict[str, Callable] = {}
_AST_OP_REGISTRY: dict[str, Callable] = {}


def op_impl(name: str) -> Callable:
    """Register a builder with parsed operands and keyword-only ``span``.

    Ordinary operands use parse_expression. Convert static instruction attributes
    in the implementation with require_const_value when Python values are needed.
    """
    def decorator(func: Callable) -> Callable:
        if name in _OP_REGISTRY or name in _AST_OP_REGISTRY:
            raise InvalidOperation(f"Duplicate op_impl registration: '{name}'")
        _OP_REGISTRY[name] = func
        return func

    return decorator


def op_impl_ast(name: str) -> Callable:
    """Register a special-parse ``(self, call: ast.Call)`` syntax handler.

    Use this for references, typed literals and descriptors whose source syntax
    cannot be treated as ordinary operands (e.g. a call whose let-binding target
    is part of its lowering). The handler owns argument parsing.
    """
    def decorator(func: Callable) -> Callable:
        if name in _OP_REGISTRY or name in _AST_OP_REGISTRY:
            raise InvalidOperation(f"Duplicate op_impl registration: '{name}'")
        _AST_OP_REGISTRY[name] = func
        return func

    return decorator
