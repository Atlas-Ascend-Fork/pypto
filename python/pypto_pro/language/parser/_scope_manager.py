# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scope and control-flow state used while parsing Python into SSA IR."""

from __future__ import annotations

__all__ = [
    "ConstantState",
    "ControlFlowInfo",
    "JumpInfo",
    "JumpKind",
    "LocalScope",
    "LoopVarState",
    "PhiState",
    "ScopeManager",
]


from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import enum
from typing import Any, Callable

from pypto.pypto_impl import ir

from ..._errors import InvalidOperation, InvalidVal
from ._utils import _is_const_expr


class JumpKind(enum.Enum):
    """Terminator kind for one parser-local code block."""

    YIELD = "yield"
    CONTINUE = "continue"
    BREAK = "break"
    RETURN = "return"


@dataclass(frozen=True)
class JumpInfo:
    """One emitted terminator and the source values carried by its CFG edge."""

    jump_op: ir.YieldStmt | ir.BreakStmt | ir.ContinueStmt | None
    outputs: tuple[ir.Expr, ...]


@dataclass
class ControlFlowInfo:
    """Merge state owned by one If or Loop operation."""

    merge_names: tuple[str, ...]
    flatten: bool = False
    jumps: list[JumpInfo] = field(default_factory=list)


class ConstantState(enum.Enum):
    """Constant propagation state for one control-flow value."""

    UNSET = 0
    MAY_BE_CONSTANT = 1
    NONCONSTANT = 2


def _type_equal(lhs: ir.Type, rhs: ir.Type) -> bool:
    return ir.structural_equal(lhs, rhs, enable_auto_mapping=False)


@dataclass
class PhiState:
    """Cutile-compatible type and constant inference for one control-flow result."""

    ty: ir.Type | None = None
    last_span: ir.Span | None = None
    constant_state: ConstantState = ConstantState.UNSET
    constant_value: ir.Expr | None = None
    type_equal: Callable[[ir.Type, ir.Type], bool] = field(default=_type_equal, repr=False, compare=False)

    def propagate(
        self,
        src: ir.Expr,
        *,
        fail_eagerly: bool = False,
    ) -> None:
        """Merge one reachable predecessor using cutile's propagation order."""
        src_ty = src.type
        if self.ty is None:
            self.ty = src_ty
            self.last_span = src.span
        elif isinstance(src_ty, ir.NoneType):
            if fail_eagerly and not isinstance(self.ty, ir.NoneType):
                raise InvalidVal(
                    "Loop-carried variable has an invalid type on a jump path",
                    span=src.span,
                    parser_retry=True,
                )
            self.ty = src_ty
        elif not isinstance(self.ty, ir.NoneType):
            if not self.type_equal(self.ty, src_ty):
                if fail_eagerly:
                    raise InvalidVal(
                        "Inconsistent types in control flow",
                        span=src.span,
                        hint="Check that the variable types are compatible. Use astype to ensure consistent types.",
                        parser_retry=True,
                    )
                self.ty = ir.NoneType.get()
            self.last_span = src.span

        if isinstance(src_ty, ir.NoneType):
            self.constant_state = ConstantState.NONCONSTANT
            self.constant_value = None
            return

        if not _is_const_expr(src):
            self.constant_state = ConstantState.NONCONSTANT
            self.constant_value = None
        elif self.constant_state is ConstantState.UNSET:
            self.constant_state = ConstantState.MAY_BE_CONSTANT
            self.constant_value = src
        elif (
            self.constant_state is ConstantState.MAY_BE_CONSTANT
            and not ir.structural_equal(
                self.constant_value, src, enable_auto_mapping=False
            )
        ):
            self.constant_state = ConstantState.NONCONSTANT
            self.constant_value = None


@dataclass
class LoopVarState:
    """Separate loop-header and loop-result phi state for one carried variable."""

    body_phi: PhiState
    result_phi: PhiState


@dataclass
class LocalScope:
    """Bindings and termination state for one lexical scope."""

    scope_type: str
    variables: dict[str, Any] = field(default_factory=dict)
    jump_kind: JumpKind | None = None

    def __contains__(self, name: str) -> bool:
        return name in self.variables

    def __getitem__(self, name: str) -> Any:
        return self.variables[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def items(self):
        return self.variables.items()

    def set_jump(self, kind: JumpKind) -> None:
        if self.jump_kind is None:
            self.jump_kind = kind


class ScopeManager:
    """Manage lexical bindings and the active If/Loop merge states."""

    def __init__(self, strict_ssa: bool = False):
        self.strict_ssa = strict_ssa
        self.scopes: list[LocalScope] = [LocalScope("global")]
        self.assignments: dict[str, int] = {}
        self.if_info: ControlFlowInfo | None = None
        self.loop_info: ControlFlowInfo | None = None

    @property
    def current_scope(self) -> LocalScope:
        return self.scopes[-1]

    def enter_scope(self, scope_type: str) -> None:
        self.scopes.append(LocalScope(scope_type))

    def exit_scope(self, leak_vars: bool = False) -> dict[str, Any]:
        if len(self.scopes) <= 1:
            raise InvalidOperation("Cannot exit global scope")

        local_scope = self.scopes.pop()
        if leak_vars:
            self.current_scope.variables.update(local_scope.variables)
        return local_scope.variables

    @contextmanager
    def change_if_info(self, info: ControlFlowInfo) -> Iterator[None]:
        old_info, self.if_info = self.if_info, info
        try:
            yield
        finally:
            self.if_info = old_info

    @contextmanager
    def change_loop_info(self, info: ControlFlowInfo) -> Iterator[None]:
        old_info, self.loop_info = self.loop_info, info
        try:
            yield
        finally:
            self.loop_info = old_info

    def define_var(self, name: str, value: Any, allow_redef: bool = False, span: Any | None = None) -> None:
        """Define a source variable in the current local scope."""
        local_scope = self.current_scope
        if name in local_scope and not allow_redef and self.strict_ssa:
            old_value = local_scope[name]
            previous_span = old_value.span if isinstance(old_value, ir.IRNode) else None
            raise InvalidOperation(
                f"Variable '{name}' is already defined",
                span=span,
                previous_span=previous_span,
                hint="Use a different variable name for each assignment (SSA form requires unique names)",
                note="Each variable can only be assigned once per scope",
            )

        local_scope[name] = value
        self.assignments[name] = self.assignments.get(name, 0) + 1

    def lookup_var(self, name: str) -> Any | None:
        for local_scope in reversed(self.scopes):
            if name in local_scope:
                return local_scope[name]
        return None

    def lookup_var_bounded(self, name: str, barrier: str = "inline") -> Any | None:
        """Look up a variable without crossing the nearest barrier scope."""
        for local_scope in reversed(self.scopes):
            if name in local_scope:
                return local_scope[name]
            if local_scope.scope_type == barrier:
                return None
        return None

    def is_defined(self, name: str) -> bool:
        return self.lookup_var(name) is not None

    def current_scope_type(self) -> str:
        return self.current_scope.scope_type

    def in_scope_type(self, scope_type: str) -> bool:
        return any(local_scope.scope_type == scope_type for local_scope in self.scopes)
