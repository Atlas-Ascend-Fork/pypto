# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser integration tests for ``mutex_ids="auto"``."""

import importlib
import re
from types import SimpleNamespace

from pypto_pro._errors import CommonInner, InvalidArgument, InvalidOperation, InvalidType
import pypto_pro.language as pl
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl import ir

jit_runtime = importlib.import_module("pypto_pro.runtime.jit")


def _kernel_def(kernel):
    return kernel.to_kernel_def()


def _parse_once_and_resolve(kernel):
    kernel_def = _kernel_def(kernel)
    programs = {}
    for target in (ir.SectionKind.Cube, ir.SectionKind.Vector):
        programs[target] = kernel_def.parse_target_program(target)[0]
    return programs


def _walk_statements(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _walk_statements(child)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
        yield from _walk_statements(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from _walk_statements(stmt.then_body)
        if stmt.else_body is not None:
            yield from _walk_statements(stmt.else_body)


def _mutex_calls(program, function_name):
    function = program.get_function(function_name)
    return [
        stmt.expr
        for stmt in _walk_statements(function.body)
        if isinstance(stmt, ir.EvalStmt)
        and isinstance(stmt.expr, ir.Call)
        and stmt.expr.name in ("system.mutex_lock_dyn", "system.mutex_unlock_dyn")
    ]


def _bound_ids(program, function_name):
    return [
        int(stmt.value.args[2].elements[0].value)
        for stmt in _walk_statements(program.get_function(function_name).body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        and stmt.value.name == "block.make_mutex_tile" and stmt.value.args[2].elements
    ]


def test_auto_mutex_parser_requires_mutex_id_manager():
    with pytest.raises(CommonInner, match="auto_mutex=True requires a mutex ID manager"):
        ASTParser(__file__, [], ir.SectionKind.Vector, auto_mutex=True)


def test_auto_ids_are_planned_and_resolved_into_first_parse_ir():
    @pl.jit(auto_mutex=True)
    def k(x: pl.Tensor[[1, 64], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=2)
        tile = group.next()
        pl.load(tile, x, [0, 0])
        pl.store(x, tile, [0, 0])

    programs = _parse_once_and_resolve(k)

    vector_ir = str(ir.Pass.insert_sync()(programs[ir.SectionKind.Vector]))
    assert "mutex_lock" in vector_ir
    assert "mutex_unlock" in vector_ir
    assert all("__pypto_auto_mutex_id_" not in str(program) for program in programs.values())


def test_jit_parses_each_target_once_when_auto_is_present(monkeypatch):
    class FakeKernelDef:
        _auto_mutex = True
        func_def = None

        def __init__(self):
            self.parse_targets = []
            self.last_param_directions = {}
            self.max_vec_tile_end = 0
            self.requires_simt = False

        def parse_target_program(self, target, bound_signature=None):
            self.parse_targets.append(target)
            return ir.Program([], "mock", ir.Span.unknown()), True

    kernel_def = FakeKernelDef()
    monkeypatch.setattr(
        jit_runtime,
        "_codegen_target_cce",
        lambda program, arch, build_dir, *, target, **kwargs: SimpleNamespace(target=target, sanitizer=False),
    )

    cube, vector = jit_runtime._parse_and_codegen_targets(kernel_def, NpuArch.DAV_3510, "")

    assert kernel_def.parse_targets == [ir.SectionKind.Cube, ir.SectionKind.Vector]
    assert cube.target == ir.SectionKind.Cube
    assert vector.target == ir.SectionKind.Vector


def test_jit_skips_mutex_id_resolution_without_auto_tile_group(monkeypatch):
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=[5])

    monkeypatch.setattr(
        jit_runtime,
        "_codegen_target_cce",
        lambda program, arch, build_dir, *, target, **kwargs: SimpleNamespace(target=target, sanitizer=False),
    )

    cube, vector = jit_runtime._parse_and_codegen_targets(_kernel_def(k), NpuArch.DAV_3510, "")

    assert cube.target == ir.SectionKind.Cube
    assert vector.target == ir.SectionKind.Vector


def test_jit_preserves_different_manual_ids_in_manual_only_alias_component(monkeypatch):
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=[0, 0], mutex_ids=[5, 9])

    monkeypatch.setattr(
        jit_runtime,
        "_codegen_target_cce",
        lambda program, arch, build_dir, *, target, **kwargs: SimpleNamespace(target=target, sanitizer=False),
    )

    cube, vector = jit_runtime._parse_and_codegen_targets(_kernel_def(k), NpuArch.DAV_3510, "")

    assert cube.target == ir.SectionKind.Cube
    assert vector.target == ir.SectionKind.Vector


def test_jit_rejects_multi_id_manual_tile_in_alias_component(monkeypatch):
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[5, 9]])
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)

    monkeypatch.setattr(
        jit_runtime,
        "_codegen_target_cce",
        lambda program, arch, build_dir, *, target, **kwargs: SimpleNamespace(target=target, sanitizer=False),
    )

    with pytest.raises(InvalidArgument, match="must have exactly one manual mutex ID"):
        jit_runtime._parse_and_codegen_targets(_kernel_def(k), NpuArch.DAV_3510, "")


def test_jit_rejects_crossing_addresses_in_auto_alias_component():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        pl.make_tile_group(type=tt, addrs=64, mutex_ids="auto", depth=1)

    with pytest.raises(InvalidOperation, match="possible buffer address trampling"):
        _parse_once_and_resolve(k)


def test_same_operation_auto_tiles_receive_different_ids_in_one_codegen_dedup_call():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        source_group = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        output_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids="auto", depth=1)
        pl.move(output_group.current(), source_group.current())

    programs = _parse_once_and_resolve(k)
    vector_program = programs[ir.SectionKind.Vector]
    ids = _bound_ids(vector_program, k.__name__)
    assert len(ids) == 2 and ids[0] != ids[1]


def test_exhausted_manual_candidates_preserve_overlap_for_codegen():
    all_manual_ids = list(range(32))

    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        manual_group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=all_manual_ids)
        auto_group = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids="auto", depth=1)
        for index in pl.range(0, 32):
            pl.move(auto_group.current(), manual_group[index])

    programs = _parse_once_and_resolve(k)
    vector_program = programs[ir.SectionKind.Vector]
    synced = ir.Pass.insert_sync()(vector_program)
    calls = _mutex_calls(synced, k.__name__)
    assert len(calls) == 4
    assert all(len(call.args) == 1 for call in calls)
    assert "__sync_needed" in str(synced)
    assert str(synced).count("block.tile_mutex_id") == 1


def test_auto_requires_depth():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto")

    with pytest.raises(InvalidArgument, match="depth is required"):
        _kernel_def(k).parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("mutex_ids", ["auto", [0], (0,), [[0, 1]]])
def test_mutex_ids_require_auto_mutex_enabled(mutex_ids):
    @pl.jit(auto_mutex=False)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=mutex_ids, depth=1)

    with pytest.raises(InvalidOperation, match="Configuring mutex_ids requires .*auto_mutex=True"):
        _kernel_def(k).parse_target_program(ir.SectionKind.Vector)


def test_unknown_string_mutex_ids_is_rejected():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="automatic", depth=1)

    with pytest.raises(InvalidArgument, match='only supports "auto"'):
        _kernel_def(k).parse_target_program(ir.SectionKind.Vector)


def test_auto_cannot_be_mixed_with_manual_ids():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=["auto", 3])

    with pytest.raises(InvalidType, match="Cannot convert .*str.* to IR expression"):
        _kernel_def(k).parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("mutex_ids", [[], ()])
def test_empty_mutex_ids_are_invisible_to_planner(mutex_ids):
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=mutex_ids, depth=2)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)

    programs = _parse_once_and_resolve(k)
    assert all("__pypto_auto_mutex_id_" not in str(program) for program in programs.values())


def test_none_mutex_ids_are_invisible_to_planner():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=None, depth=2)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)

    programs = _parse_once_and_resolve(k)
    assert all("__pypto_auto_mutex_id_" not in str(program) for program in programs.values())


def test_compile_time_if_only_collects_live_branch():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        if True:
            pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        else:
            pl.make_tile_group(type=tt, addrs=128, mutex_ids="auto", depth=4)

    programs = _parse_once_and_resolve(k)
    assert all(str(program).count("block.make_mutex_tile") == 1 for program in programs.values())


def test_compile_time_ternary_only_collects_live_branch():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        (
            pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
            if True
            else pl.make_tile_group(type=tt, addrs=128, mutex_ids="auto", depth=4)
        )

    programs = _parse_once_and_resolve(k)
    assert all(str(program).count("block.make_mutex_tile") == 1 for program in programs.values())


def test_same_helper_at_two_callsites_produces_two_stable_groups():
    def helper(tt, addr):
        return pl.make_tile_group(type=tt, addrs=addr, mutex_ids="auto", depth=1)

    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        helper(tt, 0)
        helper(tt, 128)

    first_programs = _parse_once_and_resolve(k)
    second_programs = _parse_once_and_resolve(k)
    first_ir = re.sub(r"memref_id=\d+", "memref_id=<id>", str(first_programs[ir.SectionKind.Vector]))
    second_ir = re.sub(r"memref_id=\d+", "memref_id=<id>", str(second_programs[ir.SectionKind.Vector]))
    assert first_ir == second_ir
    assert "__pypto_auto_mutex_id_" not in str(first_programs[ir.SectionKind.Vector])


def test_public_groups_are_routed_by_memory_space():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        vec_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        pl.make_tile_group(type=vec_type, addrs=0, mutex_ids="auto", depth=1)
        pl.make_tile_group(type=mat_type, addrs=0, mutex_ids="auto", depth=1)

    programs = _parse_once_and_resolve(k)
    assert all("__pypto_auto_mutex_id_" not in str(program) for program in programs.values())


def test_auto_placeholders_resolve_from_independent_target_plans():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        vec_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        pl.make_tile_group(type=vec_type, addrs=0, mutex_ids="auto", depth=1)
        pl.make_tile_group(type=mat_type, addrs=0, mutex_ids="auto", depth=1)

    programs = _parse_once_and_resolve(k)
    assert all("__pypto_auto_mutex_id_" not in str(program) for program in programs.values())


def test_explicit_section_groups_are_routed_by_memory_space():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        vec_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        with pl.section_cube():
            pl.make_tile_group(type=mat_type, addrs=0, mutex_ids="auto", depth=1)
        with pl.section_vector():
            pl.make_tile_group(type=vec_type, addrs=0, mutex_ids="auto", depth=1)

    programs = _parse_once_and_resolve(k)
    assert all("__pypto_auto_mutex_id_" not in str(program) for program in programs.values())
