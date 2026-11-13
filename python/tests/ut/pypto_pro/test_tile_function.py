# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""SIMD tile parser and public PGO VF plugin integration."""

import ast
from importlib import import_module
import textwrap
from types import SimpleNamespace

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidType, NotSupported, PyptoProError
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.runtime.platform import NpuArch
import pytest


def _parse_body(body, closure=None, func_type=ir.FunctionType.SimdTileVF, *, tile_params=False):
    params = "dst, src, n: pl.DT_INT64" if tile_params else "n: pl.DT_INT64"
    source = f"def body({params}):\n" + textwrap.indent(body, "    ")
    parser = ASTParser(
        source_file=__file__,
        source_lines=source.splitlines(),
        target=ir.SectionKind.Vector,
        debug_info=ir.IRDebugInfo(),
        closure_vars={"pl": pl, "vf": vf, **(closure or {})},
        void_return_only=True,
        void_return_context="simd_tile",
    )
    callsite_types = None
    if tile_params:
        span = ir.Span.unknown()
        tile_type = ir.TileType(
            [2, 128],
            pl.DT_FP32,
            ir.MemRef(ir.MemorySpace.Vec, ir.ConstInt(0, pl.DT_INT64, span), 1024),
            ir.TileView([ir.ConstInt(-1, pl.DT_INT64, span), ir.ConstInt(-1, pl.DT_INT64, span)]),
        )
        callsite_types = {"dst": tile_type, "src": tile_type}
    return parser.parse_function(
        ast.parse(source).body[0],
        func_type=func_type,
        callsite_param_types=callsite_types,
    )


def test_simd_tile_rejects_nonvoid_return():
    with pytest.raises(PyptoProError, match="only supports bare return or return None"):
        _parse_body("return n")


@pytest.mark.parametrize(
    "body,error",
    [
        ("pl.abs(dst)", InvalidArgument),
        ("pl.abs(n, src)", InvalidType),
    ],
)
def test_simd_tile_preserves_op_argument_validation(body, error):
    with pytest.raises(error):
        _parse_body(body, tile_params=True)


@pytest.mark.parametrize("func_type", [ir.FunctionType.SimdTileVF, ir.FunctionType.SimdTileCallee])
@pytest.mark.parametrize("barrier", ["bar_m", "bar_mte1", "bar_fix"])
def test_simd_tile_preserves_vector_section_barrier_validation(func_type, barrier):
    with pytest.raises(NotSupported, match="requires a Cube section; the current target is Vector"):
        _parse_body(f"pl.system.{barrier}()", func_type=func_type)


@pytest.mark.parametrize("func_type", [ir.FunctionType.SimdTileVF, ir.FunctionType.SimdTileCallee])
def test_simd_tile_preserves_register_operation_scope_validation(func_type):
    with pytest.raises(InvalidOperation, match="can only be used inside @pl.vector_function"):
        _parse_body("mask = vf.create_mask()", func_type=func_type)


@pytest.mark.parametrize(
    "body,op",
    [
        ("pl.neg(dst, src)", "block.neg"),
        ("pl.abs(dst, src)", "block.abs"),
        ("pl.maximum(dst, src, src)", "block.maximum"),
        ("pl.exp(dst, src)", "block.exp"),
        ("pl.add(dst, src, n)", "block.adds"),
        ("pl.add(dst, src, src)", "block.add"),
        ("pl.recip(dst, src)", "block.recip"),
        ("pl.mul(dst, src, src)", "block.mul"),
        ("pl.mul(dst, src, 2.0)", "block.muls"),
        ("pl.sub(dst, src, n)", "block.subs"),
        ("pl.sub(dst, src, src)", "block.sub"),
        ("offset = src + n", "block.subview"),
        ("value = src[0, 0]", "block.getval"),
        ("dst[0, 0] = 1.0", "block.setval"),
    ],
)
def test_simd_tile_parses_operations_without_plugin_support_checks(body, op):
    function = _parse_body(body, tile_params=True)
    assert function.body.stmts[0].value.name == op


def test_simd_tile_parses_imported_op_alias():
    function = _parse_body("absolute(dst, src)", {"absolute": pl.abs}, tile_params=True)
    assert function.body.stmts[0].value.name == "block.abs"


@pytest.mark.parametrize("body", ["result = pl.neg(dst, src)", "result: pl.DT_FP32 = pl.neg(dst, src)"])
def test_simd_tile_parses_assigned_operation(body):
    function = _parse_body(body, tile_params=True)
    assignment = function.body.stmts[0]
    assert isinstance(assignment, ir.AssignStmt)
    assert isinstance(assignment.targets[0].type, ir.NoneType)


def test_simd_tile_parses_shape_query_and_nested_operation():
    function = _parse_body(
        "cols = src.valid_shape[1]\ncount = pl.min(n, cols)\nfor i in pl.range(count):\n    pl.neg(dst, src)",
        tile_params=True,
    )
    query = function.body.stmts[0]
    assert query.value.name == "block.tile_valid_shape"
    loop = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.ForStmt))
    call = next(
        stmt.value for stmt in loop.body.stmts if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    )
    assert call.name == "block.neg"


@pytest.mark.parametrize("caller_mode,callee_mode", [("simd_tile", "simd"), ("simd", "simd_tile")])
def test_simd_tile_rejects_cross_mode_call(caller_mode, callee_mode):
    @pl.vector_function(mode=callee_mode)
    def helper():
        return

    func_type = ir.FunctionType.SimdTileVF if caller_mode == "simd_tile" else ir.FunctionType.SimdVF
    with pytest.raises(PyptoProError, match="simd_tile"):
        _parse_body("helper()", {"helper": helper}, func_type)


def test_simd_tile_rejects_plain_helper():
    def helper():
        return

    with pytest.raises(PyptoProError):
        _parse_body("helper()", {"helper": helper})


def test_simd_tile_rejects_tensor_parameter():
    @pl.vector_function(mode="simd_tile")
    def helper(x):
        # The call boundary must reject x before attempting this invalid body.
        pl.neg(missing_tile, x)  # noqa: F821

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 64], pl.DT_FP32]):
        helper(x)

    with pytest.raises(PyptoProError, match="TileType or ScalarType"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("func_type", [ir.FunctionType.SimdTileVF, ir.FunctionType.SimdTileCallee])
def test_simd_tile_instantiation_rejects_tensor_parameter(func_type):
    @pl.vector_function(mode="simd_tile")
    def helper(value):
        return

    parser = _parser()
    span = ir.Span.unknown()
    tensor = ir.Var("tensor", ir.TensorType([1, 64], pl.DT_FP32), span)
    with pytest.raises(InvalidType, match="TileType or ScalarType"):
        parser._instantiate_simd_function("helper", helper, [tensor], func_type)
    assert not parser.external_funcs
    assert not parser.vf_func_cache


def test_simd_tile_parameter_error_uses_call_location():
    @pl.vector_function(mode="simd_tile")
    def helper(value):
        return

    tensor = ir.Var("tensor", ir.TensorType([1, 64], pl.DT_FP32), ir.Span.unknown())
    with pytest.raises(InvalidType, match="TileType or ScalarType") as error:
        _parse_body("helper(tensor)", {"helper": helper, "tensor": tensor})
    assert error.value.span["filename"] == __file__
    assert error.value.span["line"] == 2


def test_simd_tile_specialization_cache_and_nested_callee():
    @pl.vector_function(mode="simd_tile")
    def inner(dst, value):
        pl.add(dst, dst, value)

    @pl.vector_function(mode="simd_tile")
    def outer(dst, value):
        inner(dst, value)

    @pl.jit(auto_mutex=False)
    def kernel():
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[1, 64])
        dst = pl.make_tile(tt, addr=0)
        outer(dst, 1)
        outer(dst, 1)
        outer(dst, 2)

    definition = kernel.to_kernel_def()
    program, _ = definition.parse_target_program(ir.SectionKind.Vector)
    assert definition.has_tile_function
    funcs = list(program.functions.values())
    assert sum(f.func_type == ir.FunctionType.SimdTileVF for f in funcs) == 2
    assert sum(f.func_type == ir.FunctionType.SimdTileCallee for f in funcs) == 2
    assert all(len(f.params) == 1 for f in funcs if f.func_type != ir.FunctionType.Opaque)


def test_simd_tile_recursion_is_rejected():
    @pl.vector_function(mode="simd_tile")
    def recursive(n):
        recursive(n)

    with pytest.raises(PyptoProError, match="Recursive"):
        _parse_body("recursive(n)", {"recursive": recursive})


def test_simd_tile_parses_simt_launch():
    @pl.vector_function(mode="simt", max_threads=32)
    def simt():
        return

    function = _parse_body("simt[32]()", {"simt": simt})
    assert function.body.stmts[0].value.name == "simt.launch"


def test_simd_tile_specializes_actual_types_and_reuses_allocations():
    @pl.vector_function(mode="simd_tile")
    def inner(dst, value):
        pl.sub(dst, dst, value)

    @pl.vector_function(mode="simd_tile")
    def outer(dst, value):
        inner(dst, value)

    @pl.jit(auto_mutex=False)
    def kernel(integer: pl.DT_UINT32, floating: pl.DT_FP32):
        compact = pl.TileType(shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 64])
        padded = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 64])
        full = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 128])
        a = pl.make_tile(compact, addr=0)
        b = pl.make_tile(padded, addr=0x1000)
        c = pl.make_tile(full, addr=0x2000)
        same_type = pl.make_tile(compact, addr=0x3000)
        outer(a, integer)
        outer(b, integer)
        outer(c, integer)
        outer(c, floating)
        outer(same_type, integer)
        inner(a, integer)

    definition = kernel.to_kernel_def()
    program, _ = definition.parse_target_program(ir.SectionKind.Vector)
    assert definition.has_tile_function
    funcs = list(program.functions.values())
    outer_funcs = [f for f in funcs if f.name.startswith("outer")]
    callees = [f for f in funcs if f.func_type == ir.FunctionType.SimdTileCallee]
    assert len(outer_funcs) == len(callees) == 4
    assert sum(f.func_type == ir.FunctionType.SimdTileVF for f in funcs) == 5
    signatures = [
        (
            tuple(d.value for d in f.params[0].type.shape),
            tuple(d.value for d in f.params[0].type.tile_view.valid_shape),
            f.params[1].type.dtype,
        )
        for f in outer_funcs
    ]
    expected_signatures = [
        ((2, 64), (2, 64), pl.DT_UINT32),
        ((2, 128), (2, 64), pl.DT_UINT32),
        ((2, 128), (2, 128), pl.DT_UINT32),
        ((2, 128), (2, 128), pl.DT_FP32),
    ]
    assert all(signature in signatures for signature in expected_signatures)
    assert all(any(all(p.type == q.type for p, q in zip(f.params, c.params)) for c in callees) for f in outer_funcs)


def test_specialized_simd_tile_preserves_alias_valid_shape():
    @pl.vector_function(mode="simd_tile")
    def helper(dst, src, flag, count):
        alias = dst
        if flag:
            selected = alias
        else:
            selected = dst
        for _ in pl.range(count):
            alias = selected
            pl.neg(alias, src)
        pl.neg(alias, src)

    @pl.jit(auto_mutex=False)
    def kernel(flag: pl.DT_BOOL, count: pl.DT_INT32):
        full = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 128])
        tail = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[1, 70])
        a = pl.make_tile(full, addr=0)
        b = pl.make_tile(tail, addr=0x1000)
        helper(a, a, flag, count)
        helper(b, b, flag, count)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert sum(f.func_type == ir.FunctionType.SimdTileVF for f in program.functions.values()) == 2


@pytest.mark.parametrize("control_flow", ["if", "for", "while", "break", "continue"])
@pytest.mark.parametrize("mismatched", [False, True])
def test_specialized_simd_tile_checks_all_control_flow_inputs(control_flow, mismatched):
    @pl.vector_function(mode="simd_tile")
    def helper(dst, src, flag, count):
        selected = dst
        if control_flow == "if":
            if flag:
                selected = dst
            else:
                selected = src
        elif control_flow == "while":
            while flag:
                selected = src
                break
        else:
            for _ in pl.range(count):
                selected = src
                if control_flow == "break":
                    break
                elif control_flow == "continue":
                    continue
        pl.neg(dst, selected)

    @pl.jit(auto_mutex=False)
    def kernel(flag: pl.DT_BOOL, count: pl.DT_INT32):
        full = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 128])
        tail = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[1, 70])
        a = pl.make_tile(full, addr=0)
        b = pl.make_tile(tail, addr=0x1000)
        helper(a, a, flag, count)
        if mismatched:
            helper(a, b, flag, count)
        else:
            helper(b, b, flag, count)

    if mismatched:
        with pytest.raises(PyptoProError):
            kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    else:
        program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        assert sum(f.func_type == ir.FunctionType.SimdTileVF for f in program.functions.values()) == 2


def _parser(closure=None):
    return ASTParser(
        source_file=__file__,
        source_lines=["def caller():", "    helper()"],
        target=ir.SectionKind.Vector,
        debug_info=ir.IRDebugInfo(),
        closure_vars={"pl": pl, **(closure or {})},
    )


@pytest.mark.parametrize("mode", ["simd_tile"])
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
        kind = ir.FunctionType.SimdVF if mode == "simd" else ir.FunctionType.SimdTileVF
        return parser._instantiate_simd_function("helper", helper, args, kind)

    first = instantiate(pl.DT_INT32)
    second = instantiate(pl.DT_INT64)
    assert first.params[0].type.dtype == pl.DT_INT32
    assert second.params[0].type.dtype == pl.DT_INT64
    assert first.name != second.name
    assert instantiate(pl.DT_INT32) is first
    assert instantiate(pl.DT_INT64) is second


@pytest.mark.parametrize("mode", ["simd_tile"])
def test_vf_constant_specialization_preserves_dtype(mode):
    @pl.vector_function(mode=mode)
    def helper(value, runtime):
        result = value + runtime  # noqa: F841

    parser = _parser()
    span = ir.Span.unknown()
    runtime = ir.Var("runtime", ir.ScalarType(pl.DT_INT32), span)
    kind = ir.FunctionType.SimdVF if mode == "simd" else ir.FunctionType.SimdTileVF
    functions = [
        parser._instantiate_simd_function("helper", helper, [ir.ConstInt(1, dtype, span), runtime], kind)
        for dtype in (pl.DT_INT32, pl.DT_UINT32)
    ]
    assert functions[0].name != functions[1].name
    assert [f.body.stmts[0].value.type.dtype for f in functions] == [pl.DT_INT32, pl.DT_UINT32]
    assert all(len(f.params) == 1 for f in functions)


@pytest.mark.parametrize("mode", ["simd_tile"])
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
    entry_kind = ir.FunctionType.SimdVF if mode == "simd" else ir.FunctionType.SimdTileVF
    callee_kind = ir.FunctionType.SimdCallee if mode == "simd" else ir.FunctionType.SimdTileCallee
    assert program.get_function("inner").func_type == entry_kind
    assert program.get_function("inner__callee").func_type == callee_kind


@pytest.mark.parametrize("callee_mode", ["simd_tile"])
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


def _kernel(mode, called):
    @pl.vector_function(mode=mode)
    def helper(dst):
        return

    @pl.jit(auto_mutex=False)
    def kernel():
        with pl.section_vector():
            tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
            dst = pl.make_tile(tt, addr=0)
            pl.neg(dst, dst)
            if called:
                helper(dst)
                helper(dst)

    return kernel.to_kernel_def()


@pytest.mark.parametrize(
    "mode,called,expected", [("simd_tile", True, True), ("simd_tile", False, False), ("simd", True, False)]
)
def test_tile_function_fact_tracks_each_target_parse(mode, called, expected):
    definition = _kernel(mode, called)
    assert not definition.has_tile_function
    for target, has_tile in [
        (ir.SectionKind.Vector, expected),
        (ir.SectionKind.Cube, False),
        (ir.SectionKind.Vector, expected),
    ]:
        parser_facts = []
        program, _ = definition.parse_target_program(
            target, observer=lambda program, parser: parser_facts.append(parser.has_tile_function)
        )
        assert parser_facts == [has_tile]
        assert definition.has_tile_function is has_tile
        tile_functions = [f for f in program.functions.values() if f.func_type == ir.FunctionType.SimdTileVF]
        assert len(tile_functions) == int(has_tile)


@pytest.mark.parametrize(
    "mode,called,expected", [("simd_tile", True, True), ("simd_tile", False, False), ("simd", True, False)]
)
def test_jit_calls_plugin_only_for_target_with_tile_functions(
    monkeypatch, mode, called, expected, expect_pgo_vf, pgo_vf_available
):
    jit = import_module("pypto_pro.runtime.jit")
    original = ir.PGOVFRun
    plugin_inputs = []
    plugin_outputs = []
    codegen_targets = []

    def run_plugin(program, function):
        assert function is None
        plugin_inputs.append(program)
        assert any(f.func_type == ir.FunctionType.SimdTileVF for f in program.functions.values())
        out_program, out_function = original(program, function)
        assert out_function is None
        plugin_outputs.append(out_program)
        return out_program, out_function

    def codegen(program, *args, target, **kwargs):
        codegen_targets.append(target)
        if expected:
            assert program is plugin_outputs[-1]
        return SimpleNamespace(sanitizer=False)

    monkeypatch.setattr(jit.ir, "PGOVFRun", run_plugin)
    monkeypatch.setattr(jit, "_codegen_target_cce", codegen)
    with expect_pgo_vf(required=expected):
        cube, vector = jit._parse_and_codegen_targets(_kernel(mode, called), NpuArch.DAV_3510, "")
        assert cube is None
        assert vector is not None
    assert codegen_targets == ([] if expected and not pgo_vf_available else [ir.SectionKind.Vector])
    assert len(plugin_inputs) == int(expected)


def test_simd_tile_decorator():
    @pl.vector_function(mode="simd_tile")
    def helper():
        return

    assert pl.parser.decorator.is_vector_function(helper)
    assert pl.parser.decorator.is_simd_tile_function(helper)
    assert not pl.parser.decorator.is_simt_function(helper)
