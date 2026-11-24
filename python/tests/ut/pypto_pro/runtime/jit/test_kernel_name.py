# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""JIT names propagate through IR, device entry, launch, SK and profiling."""

import ast
from dataclasses import replace
import importlib

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidType, NameNotFound
import pypto_pro.language as pl
from pypto_pro.runtime.compile_config import get_jit_compile_config
from pypto_pro.runtime.kernel import KernelDef
from pypto_pro.runtime.platform import NpuArch
import pytest

jit = importlib.import_module("pypto_pro.runtime.jit")


def _parse_named_kernel(name):
    @pl.jit(name=name)
    def kernel():
        pass

    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("name", [
    None, "custom_silu", "original_kernel", "out", "blockDim", "stream", "launchDim",
    "gridX", "threadBlockX", "pyptoProfOn",
])
@pytest.mark.parametrize("mode", ["simd", "simt", "sk"])
def test_name_is_shared_by_ir_launch_sk_and_profiling(name, mode, monkeypatch, tmp_path):
    expected = name or "original_kernel"
    sk_cap = pl.SkCap.CAP_4_DISABLE_DCCI_ALL if mode == "sk" else None
    if mode == "simt":
        @pl.jit(arch="3510", name=name, mode="simt", max_threads=32, compile_timeout=30)
        def original_kernel(out: pl.Tensor[[32], pl.DT_FP32]):
            out[pl.simt.thread_idx().x] = 1.0
    else:
        @pl.jit(arch="3510", name=name, auto_mutex=False, sk_cap=sk_cap, compile_timeout=30)
        def original_kernel(out: pl.Tensor[[1, 32], pl.DT_FP32]):
            with pl.section_vector():
                tile = pl.make_tile(
                    pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0,
                )
                pl.load(tile, out, [0, 0])
                pl.store(out, tile, [0, 0])

    definition = original_kernel.to_kernel_def()
    captured_ast = definition._func_def
    # All target parses reuse the AST with the effective kernel name.
    targets = (ir.SectionKind.Vector, ir.SectionKind.Vector) if mode == "simt" else (
        ir.SectionKind.Vector, ir.SectionKind.Cube, ir.SectionKind.Vector,
    )
    for target in targets:
        program, _ = definition.parse_target_program(target)
        assert jit._get_kernel_ir_function(program).name == expected
        assert definition._func_def is captured_ast
        assert captured_ast.name == expected
    assert original_kernel.__name__ == "original_kernel"

    cube, vector = jit._parse_and_codegen_targets(definition, NpuArch.DAV_3510, str(tmp_path))
    cg = jit._assemble_cv_source(cube, vector)
    assert cg.kernel_name == expected
    assert f"void {expected}_impl(" in cg.content
    monkeypatch.setattr(jit, "_compile_shared_library", lambda *_args: "unused.so")
    monkeypatch.setattr(jit, "_codegen", lambda *_args, **_kwargs: cg)
    monkeypatch.setattr(jit, "_setup_arch_env", lambda arch: arch)
    compiled = original_kernel._compile_variant(None, (None, None, None, None), None, ())
    assert compiled.kernel_name == expected
    assert cg.kernel_name == expected
    caller = (tmp_path / "call_kernel.cpp").read_text(encoding="utf-8")
    entry = next(line for line in caller.splitlines() if line.startswith('extern "C" __global__'))
    assert f"void {expected}(" in entry
    assert f"::{expected}<<<" in caller
    assert f"\n    ::{expected}_impl(" in caller
    assert f'aclprofStr2Id("{expected}")' in caller
    if mode == "sk":
        assert f"SK_BIND({expected}," in caller
        assert f"void {expected}_sk(" in caller


@pytest.mark.parametrize("name", [
    "1name", "bad-name", "class", "注意力", "", "   ", "bad\nname", "bad\tname", "bad\x00name",
    "\x7f", "\x80", "\x81", "\x9f", "a\x85b",
])
def test_invalid_kernel_name_is_rejected_before_ir_parsing(name):
    with pytest.raises(InvalidArgument, match=r"ASCII C\+\+ identifier"):
        _parse_named_kernel(name)


@pytest.mark.parametrize("name", [0, False, [], object()])
def test_kernel_name_requires_a_string(name):
    with pytest.raises(InvalidType, match="name must be a string"):
        _parse_named_kernel(name)


@pytest.mark.parametrize("name", ["\ud800", "\udfff", "a\ud800b"])
def test_name_rejects_surrogates_before_ir_parsing(name):
    with pytest.raises(InvalidArgument, match=r"ASCII C\+\+ identifier"):
        _parse_named_kernel(name)


@pytest.mark.parametrize("name", [None, "custom_name"])
def test_host_launcher_name_is_checked_after_applying_alias(name):
    def call_kernel(out: pl.Tensor[[32], pl.DT_INT32]):
        with pl.section_vector():
            out[0] = 1

    if name is None:
        kernel = pl.jit()(call_kernel)
        with pytest.raises(InvalidArgument, match="name 'call_kernel' is reserved"):
            kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    else:
        kernel = pl.jit(name=name)(call_kernel)
        program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        assert jit._get_kernel_ir_function(program).name == name


def test_custom_name_cannot_shadow_host_launcher():
    with pytest.raises(InvalidArgument, match="name 'call_kernel' is reserved"):
        _parse_named_kernel("call_kernel")


@pytest.mark.parametrize("name", ["helper", "named_kernel"])
def test_kernel_name_conflicting_with_vf_is_reported(name):
    @pl.vector_function(mode="simt")
    def helper(value: pl.DT_INT32) -> pl.DT_INT32:
        return value

    @pl.jit(arch="3510", mode="simt", max_threads=32, name=name)
    def original_kernel(out: pl.Tensor[[32], pl.DT_INT32], value: pl.DT_INT32):
        out[0] = helper(value)

    definition = original_kernel.to_kernel_def()
    if name == "helper":
        with pytest.raises(InvalidArgument, match="kernel name 'helper' conflicts with a referenced VF"):
            definition.parse_target_program(ir.SectionKind.Vector)
    else:
        program, _ = definition.parse_target_program(ir.SectionKind.Vector)
        assert jit._get_kernel_ir_function(program).name == name
        assert "helper" in program.functions


@pytest.mark.parametrize("name", ["__global__", "__pypto_seed_block_num", "_Reserved", "AICORE", "SK_BIND"])
def test_reserved_kernel_names_are_rejected(name):
    with pytest.raises(InvalidArgument, match="reserved"):
        _parse_named_kernel(name)


@pytest.mark.parametrize("name", ["int", "注意力"])
def test_default_kernel_name_has_the_same_constraints_as_explicit_name(name):
    def kernel():
        pass

    definition = pl.jit(kernel).to_kernel_def()
    definition._func.__name__ = name
    with pytest.raises(InvalidArgument, match=r"ASCII C\+\+ identifier"):
        definition.parse_target_program(ir.SectionKind.Vector)
    assert pl.jit(name="valid_kernel")(kernel)._name == "valid_kernel"


@pytest.mark.parametrize("name", ["int", "注意力", "__global__", "_Reserved", "call_kernel"])
@pytest.mark.parametrize("explicit", [False, True])
def test_direct_kernel_definition_validates_the_effective_name(name, explicit):
    def kernel():
        pass

    if not explicit:
        kernel.__name__ = name
    definition = KernelDef(
        func=kernel, source_file=__file__, source_lines=[], source_lines_raw=[],
        line_offset=0, col_offset=0, func_def=ast.parse("def kernel(): pass").body[0],
        closure_vars={}, name=name if explicit else None,
        func_type=ir.FunctionType.Opaque, strict_ssa=False, meta_data=None,
    )
    with pytest.raises((InvalidArgument, InvalidType)):
        definition.parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("simt", [False, True])
def test_generated_impl_calls_are_not_shadowed_by_parameters(simt, tmp_path):
    target = get_jit_compile_config().resolve_kernel_target(
        NpuArch.DAV_3510, has_cube=False, has_vector=True,
    )
    params = [("float", f"custom_kernel{suffix}", True) for suffix in ("_impl", "_impl_cube", "_impl_vector")]
    args = ", ".join(name for _, name, _ in params)
    entry = jit._make_global_entry(
        "custom_kernel", params, target=target, simt_kernel=simt, max_threads=32,
    )
    assert f"::custom_kernel_impl({args});" in entry
    cube = jit.CodegenResult(
        build_dir=str(tmp_path), content="", kernel_name="custom_kernel", bisheng_cross_sync=False,
        needs_print_debug=False, entry_params=params, has_cube=True,
    )
    vector = replace(cube, has_cube=False, has_vector=True)
    combined = jit._assemble_cv_source(cube, vector)
    assert f"::custom_kernel_impl_cube({args});" in combined.content
    assert f"::custom_kernel_impl_vector({args});" in combined.content
    sk = jit._make_sk_bind_block("custom_kernel", params, target, pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
    sk_args = ", ".join(f"args->{name}" for _, name, _ in params)
    assert f"\n    ::custom_kernel_impl({sk_args});\n" in sk


def test_main_is_allowed_for_ir_parsing():
    @pl.jit()
    def main():
        pass

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert jit._get_kernel_ir_function(program).name == "main"


def test_lambda_can_be_decorated_but_has_no_named_kernel_definition():
    kernel = pl.jit(mode="simt", max_threads=32, arch="3510")(lambda: None)
    assert kernel._arch == NpuArch.DAV_3510
    with pytest.raises(NameNotFound, match="Could not find function definition for <lambda>"):
        kernel.to_kernel_def()
