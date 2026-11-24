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
from pypto_pro._errors import InvalidOperation, InvalidType, InvalidVal, OutOfRange
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
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
    def helper(value: pl.DT_INT8, runtime):
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


@pytest.mark.parametrize("mode", ["simd", "simt"])
@pytest.mark.parametrize("constant_type, dtype, first, second", [
    (ir.ConstInt, pl.DT_INT64, 1, 2),
    (ir.ConstBool, pl.DT_BOOL, True, False),
    (ir.ConstFloat, pl.DT_FP32, 1.5, 2.5),
])
def test_vf_scalar_specializations_distinguish_values_and_positions(mode, constant_type, dtype, first, second):
    @pl.vector_function(mode=mode)
    def helper(left, right):
        return

    parser = _parser()
    span = ir.Span.unknown()
    runtime = ir.Var("runtime", ir.ScalarType(dtype), span)
    other_runtime = ir.Var("other_runtime", ir.ScalarType(dtype), span)
    if constant_type is ir.ConstBool:
        first = constant_type(first, span)
        second = constant_type(second, span)
    else:
        first = constant_type(first, dtype, span)
        second = constant_type(second, dtype, span)

    def instantiate(args):
        if mode == "simt":
            return parser._instantiate_simt_function("helper", helper, args)
        return parser._instantiate_simd_function("helper", helper, args, ir.FunctionType.SimdVF)

    first_instance = instantiate([first, runtime])
    second_instance = instantiate([second, runtime])
    swapped_instance = instantiate([runtime, first])
    assert len({first_instance.name, second_instance.name, swapped_instance.name}) == 3
    assert all(len(function.params) == 1 for function in (first_instance, second_instance, swapped_instance))
    assert instantiate([first, other_runtime]) is first_instance


def test_vf_constant_specialization_distinguishes_raw_value_types():
    @pl.vector_function(mode="simd")
    def helper(value):
        return

    parser = _parser()
    kind = ir.FunctionType.SimdVF
    boolean = parser._instantiate_simd_function("helper", helper, [True], kind)
    integer = parser._instantiate_simd_function("helper", helper, [1], kind)
    fp32 = parser._instantiate_simd_function("helper", helper, [pl.DT_FP32], kind)
    fp16 = parser._instantiate_simd_function("helper", helper, [pl.DT_FP16], kind)
    assert boolean is not integer
    assert fp32 is not fp16
    assert len({boolean.name, integer.name, fp32.name, fp16.name}) == 4


@pytest.mark.parametrize("mode", ["simd", "simt"])
def test_vf_annotation_checks_each_runtime_dtype(mode):
    @pl.vector_function(mode=mode)
    def helper(value: pl.DT_INT32):
        return

    parser = _parser()
    span = ir.Span.unknown()

    def instantiate(dtype):
        args = [ir.Var("value", ir.ScalarType(dtype), span)]
        if mode == "simt":
            return parser._instantiate_simt_function("helper", helper, args)
        return parser._instantiate_simd_function("helper", helper, args, ir.FunctionType.SimdVF)

    first = instantiate(pl.DT_INT32)
    with pytest.raises(InvalidVal, match="VF parameter 'value' annotated as"):
        instantiate(pl.DT_INT64)
    assert instantiate(pl.DT_INT32) is first


@pytest.mark.parametrize("mode", ["simd", "simt"])
@pytest.mark.parametrize("value, error", [
    (128, OutOfRange),
    (2**64 - 1, OutOfRange),
    (True, InvalidType),
])
def test_vf_constant_annotation_rejects_invalid_range_or_category(mode, value, error):
    @pl.vector_function(mode=mode)
    def helper(value: pl.DT_INT8):
        return

    parser = _parser({"value": value})
    arg = parser.parse_expression(ast.parse("value", mode="eval").body)
    with pytest.raises(error, match="VF parameter 'value'"):
        if mode == "simt":
            parser._instantiate_simt_function("helper", helper, [arg])
        else:
            parser._instantiate_simd_function("helper", helper, [arg], ir.FunctionType.SimdVF)


@pytest.mark.parametrize("func_type", [ir.FunctionType.SimdVF, ir.FunctionType.SimtCallee])
@pytest.mark.parametrize("annotation, value", [
    ("pl.DT_FP16", 1),
    ("pl.DT_BOOL", True),
    ("pl.DT_BOOL", 1),
    ("pl.DT_BOOL", 0.5),
])
def test_vf_constant_annotations_follow_public_scalar_rules(func_type, annotation, value):
    parser = _parser({"value": value})
    constant = parser.parse_expression(ast.parse("value", mode="eval").body)
    func_def = ast.parse(f"def helper(value: {annotation}):\n    return").body[0]
    function = parser.parse_function(func_def, func_type=func_type, const_param_values={"value": constant})
    assert not function.params


@pytest.mark.parametrize(
    "value, error",
    [(pl.DT_FP32, InvalidType), (pl.RoundMode.CAST_RINT, InvalidType), ("fp32", InvalidType)],
    ids=["dtype", "enum", "string"],
)
def test_simd_compile_time_arguments_reject_incompatible_annotations(value, error):
    @pl.vector_function(mode="simd")
    def helper(value: pl.DT_INT32):
        return

    @pl.jit(auto_mutex=False)
    def kernel():
        helper(value)

    with pytest.raises(error, match="VF parameter 'value'"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simd_float_constants_specialize_across_vf_and_helper():
    @pl.vector_function
    def inner(value: pl.DT_FP16):
        mask = vf.create_mask(dtype=pl.DT_FP32)
        reg = vf.full(value, mask, dtype=pl.DT_FP32)  # noqa: F841

    @pl.vector_function
    def outer(value: pl.DT_FP16):
        inner(value)

    @pl.jit(auto_mutex=False)
    def kernel():
        outer(1.5)
        outer(2.5)
        outer(1.5)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    calls = [
        stmt.value for stmt in program.get_function("kernel").body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    ]
    assert [call.name for call in calls] == ["simd.call"] * 3
    names = [call.kwargs["callee"] for call in calls]
    assert names[0] == names[2] != names[1]
    assert all(not call.args for call in calls)
    for name, expected in zip(names[:2], ("1.5", "2.5")):
        entry = program.get_function(name)
        call = next(stmt.value for stmt in entry.body.stmts if isinstance(stmt, ir.AssignStmt))
        helper = program.get_function(call.name)
        assert not entry.params and not helper.params and not call.args
        full = next(
            stmt.value for stmt in helper.body.stmts
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call) and stmt.value.name == "vf.full"
        )
        assert isinstance(full.args[0], ir.ConstFloat)
        assert full.args[0].type.dtype == pl.DT_FP32
        assert repr(full.args[0].value) == expected


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
    assert program.get_function("inner_version_1").func_type == callee_kind


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
