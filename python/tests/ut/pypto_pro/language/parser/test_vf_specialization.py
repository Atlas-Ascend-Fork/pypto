# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared VF specialization identity and caller-domain contracts."""

import ast

from pypto_pro import ir
from pypto_pro._errors import InvalidOperation
import pypto_pro.language as pl
from pypto_pro.language.parser._ast_parser import ASTParser
import pytest


def _parser(closure=None):
    return ASTParser(
        source_file=__file__,
        source_lines=["def caller():", "    helper()"],
        target=ir.SectionKind.Vector,
        debug_info=ir.IRDebugInfo(),
        closure_vars={"pl": pl, **(closure or {})},
    )


@pytest.mark.parametrize("mode", ["simd", "simt"])
@pytest.mark.parametrize("hash_collision", [False, True])
def test_vf_cache_uses_actual_argument_types(mode, hash_collision, monkeypatch):
    if hash_collision:
        monkeypatch.setattr(ir, "structural_hash", lambda _: 0)

    @pl.vector_function(mode=mode)
    def helper(value):
        result = value + 1  # noqa: F841

    parser = _parser()
    span = ir.Span.unknown()

    def instantiate(dtype):
        args = [ir.Var("value", ir.ScalarType(dtype), span)]
        if mode == "simt":
            return parser._instantiate_simt_function("helper", helper, args)
        kind = ir.FunctionType.SimdVF
        return parser._instantiate_simd_function("helper", helper, args, kind)

    first = instantiate(pl.DT_INT32)
    second = instantiate(pl.DT_INT64)
    assert first.params[0].type.dtype == pl.DT_INT32
    assert second.params[0].type.dtype == pl.DT_INT64
    assert first.name != second.name
    assert instantiate(pl.DT_INT32) is first
    assert instantiate(pl.DT_INT64) is second


@pytest.mark.parametrize("mode", ["simd"])
def test_vf_constant_specialization_preserves_dtype(mode):
    @pl.vector_function(mode=mode)
    def helper(value, runtime):
        result = value + runtime  # noqa: F841

    parser = _parser()
    span = ir.Span.unknown()
    runtime = ir.Var("runtime", ir.ScalarType(pl.DT_INT32), span)
    kind = ir.FunctionType.SimdVF
    functions = [
        parser._instantiate_simd_function("helper", helper, [ir.ConstInt(1, dtype, span), runtime], kind)
        for dtype in (pl.DT_INT32, pl.DT_UINT32)
    ]
    assert functions[0].name != functions[1].name
    assert [f.body.stmts[0].value.type.dtype for f in functions] == [pl.DT_INT32, pl.DT_UINT32]
    assert all(len(f.params) == 1 for f in functions)


@pytest.mark.parametrize("mode", ["simd"])
def test_vf_entry_and_callee_have_distinct_names(mode):
    @pl.vector_function(mode=mode)
    def inner():
        return

    @pl.vector_function(mode=mode)
    def outer():
        inner()

    @pl.jit(auto_mutex=False)
    def kernel():
        inner()
        outer()

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    entry_kind = ir.FunctionType.SimdVF
    callee_kind = ir.FunctionType.SimdCallee
    assert program.get_function("inner").func_type == entry_kind
    assert program.get_function("inner__callee").func_type == callee_kind


@pytest.mark.parametrize("callee_mode", ["simd"])
@pytest.mark.parametrize(
    "caller_kind", [ir.FunctionType.SimtVF, ir.FunctionType.SimtCallee, ir.FunctionType.SimtKernel]
)
def test_simt_cannot_call_simd(callee_mode, caller_kind):
    @pl.vector_function(mode=callee_mode)
    def helper():
        return

    parser = _parser({"helper": helper})
    with pytest.raises(InvalidOperation, match="SIMT functions cannot call SIMD"):
        parser.parse_function(ast.parse("def caller():\n    helper()").body[0], func_type=caller_kind)
