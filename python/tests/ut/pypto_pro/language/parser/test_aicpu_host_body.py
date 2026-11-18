#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests pinning that an aicpu kernel is NOT a vector function (DESIGN.md D9).

``mode="aicpu_kernel"`` keeps the host lane independent of the vector-function
family: the function carries no ``@pl.vector_function`` marker, and the entry
body parses outside a VF section scope -- so plain Python helpers inline
normally and while loops are allowed, unlike inside a vector function.
"""
from __future__ import annotations

# DSL function bodies are parsed as AST, not executed -suppress pyright errors
# from type-checking the annotations and kwargs inside parsed DSL bodies.
from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.decorator import is_vector_function

from .test_inline_helper_return import _walk_statements


def _parse(kernel):
    """Parse one jit kernel's target program and return (program, kernel_def)."""
    kdef = kernel.to_kernel_def()
    program, _ = kdef.parse_target_program(ir.SectionKind.Vector)
    return program, kdef


def test_aicpu_kernel_carries_no_vector_function_marker():
    @pl.jit(mode="aicpu_kernel", arch="3510")
    def explicit_kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64]):
        pass

    assert explicit_kernel._func_type == ir.FunctionType.AicpuKernel
    # The explicit AICPU mode does not mark the kernel as a vector function:
    # the host lane is not the VF family.
    assert is_vector_function(explicit_kernel._func) is False


def test_aicpu_body_parses_outside_vf_context():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64]):
        i = 0
        while i < 4:
            i = i + 1

    program, _ = _parse(kernel)
    # Host-style control flow parses at aicpu kernel level and stays a real loop in the IR.
    assert any(isinstance(stmt, ir.WhileStmt) for stmt in program.get_function(kernel.__name__).body.stmts)


def test_aicpu_body_calls_plain_inline_helpers():
    def plain_helper(v):
        return v + 1

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64]):
        y = plain_helper(1)  # noqa: F841 -- DSL body, parsed as AST

    program, _ = _parse(kernel)
    # The call inlined into the kernel body instead of being rejected by the
    # vector-function inline rule: the carrier assignments carry its fingerprint.
    stmts = list(_walk_statements(program.get_function(kernel.__name__).body))
    assert any(
        isinstance(stmt, ir.AssignStmt)
        and stmt.targets
        and stmt.targets[0].name.startswith("__inline_0_return_val")
        for stmt in stmts
    )
