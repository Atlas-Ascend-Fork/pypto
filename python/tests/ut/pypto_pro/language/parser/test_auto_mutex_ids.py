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

from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidType, NameNotFound
import pypto_pro.language as pl
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.language.parser._scope_manager import PhiState
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
        stmt.value
        for stmt in _walk_statements(function.body)
        if (isinstance(stmt, ir.AssignStmt) and not stmt.targets)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.name in ("system.mutex_lock_dyn", "system.mutex_unlock_dyn")
    ]


def _bound_ids(program, function_name):
    return [
        int(stmt.value.args[2].elements[0].value)
        for stmt in _walk_statements(program.get_function(function_name).body)
        if (isinstance(stmt, ir.AssignStmt) and bool(stmt.targets))
        and isinstance(stmt.value, ir.Call)
        and stmt.value.name == "block.make_mutex_tile"
        and stmt.value.args[2].elements
    ]


def _bound_id_rows(program, function_name):
    return [
        tuple(int(element.value) for element in stmt.value.args[2].elements)
        for stmt in _walk_statements(program.get_function(function_name).body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        and stmt.value.name == "block.make_mutex_tile"
    ]


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
        func_type = ir.FunctionType.Opaque
        has_tile_function = False

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


def test_auto_leaf_inherits_all_manual_ids_from_exact_alias():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[5, 9]])
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)

    vector = _parse_once_and_resolve(k)[ir.SectionKind.Vector]

    assert _bound_id_rows(vector, k.__name__) == [(5, 9), (5, 9)]


def test_auto_parent_binding_is_rebuilt_with_child_union_and_matching_type():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        root_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        parent_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        leaf_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=root_type, addrs=0, mutex_ids=[[2, 5, 7, 9]])
        pl.make_tile_group(type=parent_type, addrs=0, mutex_ids="auto", depth=1)
        pl.make_tile_group(type=leaf_type, addrs=0, mutex_ids="auto", depth=1)
        pl.make_tile_group(type=leaf_type, addrs=32, mutex_ids=[5])
        pl.make_tile_group(type=parent_type, addrs=64, mutex_ids=[[7, 9]])

    vector = _parse_once_and_resolve(k)[ir.SectionKind.Vector]

    assert _bound_id_rows(vector, k.__name__) == [
        (2, 5, 7, 9),
        (2, 5),
        (2,),
        (5,),
        (7, 9),
    ]


def test_jit_rejects_crossing_addresses_in_auto_alias_component():
    @pl.jit(auto_mutex=True)
    def k(_unused: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        left_group = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)  # noqa: F841
        right_group = pl.make_tile_group(type=tt, addrs=64, mutex_ids="auto", depth=1)  # noqa: F841

    with pytest.raises(InvalidOperation, match="possible buffer address trampling") as exc_info:
        _parse_once_and_resolve(k)
    assert str(exc_info.value).count("Vec Tile at line") == 2


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


def test_auto_if_merge_pads_standalone_tile_before_insert_sync():
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        full_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        half_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        parent = pl.make_tile_group(type=full_type, addrs=0, mutex_ids="auto", depth=1)
        standalone = pl.make_tile_group(type=full_type, addrs=256, mutex_ids="auto", depth=1)
        pl.make_tile_group(type=half_type, addrs=0, mutex_ids=[3])
        pl.make_tile_group(type=half_type, addrs=64, mutex_ids=[7])
        if control[0] > 0:
            selected = parent[0]
        else:
            selected = standalone[0]
        pl.load(selected, source, [0, 0])

    vector = _parse_once_and_resolve(k)[ir.SectionKind.Vector]
    rows = _bound_id_rows(vector, k.__name__)
    assert rows[0] == (3, 7)
    assert len(rows[1]) == 2
    branch = next(stmt for stmt in _walk_statements(vector[k.__name__].body) if isinstance(stmt, ir.IfStmt))
    assert not isinstance(branch.condition, (ir.ConstBool, ir.ConstInt))
    assert branch.return_vars[0].type.mutex_count == 2
    synced = ir.Pass.insert_sync()(vector)
    assert len(_mutex_calls(synced, k.__name__)) == 4
    assert _bound_id_rows(synced, k.__name__) == rows


def _verify_resolved_ssa(kernel, expected_if_count):
    program = _kernel_def(kernel).parse_target_program(ir.SectionKind.Vector)[0]
    statements = list(_walk_statements(program[kernel.__name__].body))
    branches = [stmt for stmt in statements if isinstance(stmt, ir.IfStmt)]
    assert len(branches) == expected_if_count
    assert all(not isinstance(branch.condition, (ir.ConstBool, ir.ConstInt)) for branch in branches)
    assert any(isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
               and stmt.value.name == "block.getval" for stmt in statements)
    diagnostics = ir.IRVerifier.create_default().verify(program)
    assert not diagnostics, ir.IRVerifier.generate_report(diagnostics)
    return program


@pytest.mark.parametrize("auto_first", [True, False])
def test_auto_if_merge_accepts_manual_multi_ids_in_either_order(auto_first):
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        auto = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        manual = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[[3, 7]])
        first, second = auto[0], manual[0]
        if not auto_first:
            first, second = second, first
        if control[0] > 0:
            selected = first
        else:
            selected = second
        pl.load(selected, source, [0, 0])

    program = _verify_resolved_ssa(k, expected_if_count=1)
    assert [len(row) for row in _bound_id_rows(program, k.__name__)] == [2, 2]
    branch = next(stmt for stmt in _walk_statements(program[k.__name__].body) if isinstance(stmt, ir.IfStmt))
    assert branch.return_vars[0].type.mutex_count == 2
    assert len(_mutex_calls(ir.Pass.insert_sync()(program), k.__name__)) == 4


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("auto_first", [True, False])
def test_auto_loop_merge_accepts_manual_multi_ids(loop_kind, auto_first):
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        limit = control[0]
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        auto = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        manual = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[[3, 7]])
        selected, following = auto[0], manual[0]
        if not auto_first:
            selected, following = following, selected
        if loop_kind == "for":
            for _ in pl.range(limit):
                selected = following
        else:
            index = 0
            while index < limit:
                selected = following
                if index == 2:
                    break
                index = index + 1
                continue
        pl.load(selected, source, [0, 0])

    program = _verify_resolved_ssa(k, expected_if_count=0 if loop_kind == "for" else 2)
    assert [len(row) for row in _bound_id_rows(program, k.__name__)] == [2, 2]
    loop = next(stmt for stmt in _walk_statements(program[k.__name__].body)
                if isinstance(stmt, (ir.ForStmt, ir.WhileStmt)))
    carried = [arg.iterVar.type.mutex_count for arg in loop.iter_args
               if isinstance(arg.iterVar.type, ir.MutexTileType)]
    assert carried == [2]


@pytest.mark.parametrize("view_kind", ["reinterpret", "subview", "group_reinterpret"])
def test_auto_view_merge_accepts_manual_multi_ids(view_kind):
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        auto = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        manual = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[[3, 7]])
        if view_kind == "group_reinterpret":
            viewed = pl.reinterpret(auto, shape=[1, 64])
            first = viewed[0]
            second = manual[0]
        elif view_kind == "reinterpret":
            first = pl.reinterpret(auto[0], shape=[1, 64])
            second = pl.reinterpret(manual[0], shape=[1, 64])
        else:
            first = auto[0][:, :]
            second = manual[0][:, :]
        if control[0] > 0:
            selected = first
        else:
            selected = second
        pl.load(selected, source, [0, 0])

    program = _verify_resolved_ssa(k, expected_if_count=1)
    assert [len(row) for row in _bound_id_rows(program, k.__name__)] == [2, 2]


def test_dynamic_auto_member_merge_expands_all_group_members():
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        slot = control[1]
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        auto = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=2)
        manual = pl.make_tile_group(type=tt, addrs=512, mutex_ids=[[3, 7]])
        if control[0] > 0:
            selected = auto[slot]
        else:
            selected = manual[0]
        pl.load(selected, source, [0, 0])

    program = _verify_resolved_ssa(k, expected_if_count=1)
    assert [len(row) for row in _bound_id_rows(program, k.__name__)] == [2, 2, 2]


def test_nested_auto_tuple_phi_merges_corresponding_positions():
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        auto = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        other = pl.make_tile_group(type=tt, addrs=256, mutex_ids="auto", depth=1)
        manual = pl.make_tile_group(type=tt, addrs=512, mutex_ids=[[3, 7]])
        passive = pl.make_tile_group(type=tt, addrs=768, mutex_ids=[9])
        if control[1] > 0:
            if control[0] > 0:
                inner = (manual[0], (passive[0],))
            else:
                inner = (auto[0], (passive[0],))
            first, rest = inner
            selected = (first, rest)
        else:
            selected = (other[0], (passive[0],))
        pl.load(selected[0], source, [0, 0])
        pl.load(selected[1][0], source, [0, 0])

    program = _verify_resolved_ssa(k, expected_if_count=2)
    assert [len(row) for row in _bound_id_rows(program, k.__name__)] == [2, 2, 2, 1]


def test_auto_phi_count_conflict_is_reported_by_allocator():
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        auto = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        two = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[[3, 7]])
        three = pl.make_tile_group(type=tt, addrs=512, mutex_ids=[[3, 7, 9]])
        if control[0] > 0:
            selected = auto[0]
        else:
            selected = two[0]
        if control[1] > 0:
            selected = three[0]
        pl.load(selected, source, [0, 0])

    with pytest.raises(InvalidArgument, match="TileGroup or SSA merge") as exc_info:
        _kernel_def(k).parse_target_program(ir.SectionKind.Vector)
    assert "requires at least 3 IDs but allows at most 2" in str(exc_info.value)


@pytest.mark.parametrize("second_ids", [[[7]], None])
def test_manual_count_mismatch_is_not_relaxed_by_unrelated_auto(second_ids):
    @pl.jit(auto_mutex=True)
    def k(source: pl.Tensor[[1, 64], pl.DT_FP16], control: pl.Tensor[[3], pl.DT_INT64]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)
        first = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[[3, 7]])
        second = pl.make_tile_group(type=tt, addrs=512, mutex_ids=second_ids, depth=1)
        if control[0] > 0:
            selected = first[0]
        else:
            selected = second[0]
        pl.load(selected, source, [0, 0])

    with pytest.raises(NameNotFound, match="potentially undefined variable"):
        _kernel_def(k).parse_target_program(ir.SectionKind.Vector)


def _ssa_type_parser(debug_info=None):
    return ASTParser(
        "ssa_mutex_test.py", [], ir.SectionKind.Vector, auto_mutex=True,
        debug_info=debug_info if debug_info is not None else ir.IRDebugInfo(),
    )


_SSA_SPAN = ir.Span.unknown()


def _ssa_tile(ids, *, dtype=ir.DataType.FP16, shape=(1, 64), memory=ir.MemorySpace.Vec):
    row = [ir.Var("__pypto_auto_mutex_id_test", ir.ScalarType(ir.DataType.INT64), _SSA_SPAN)] if ids == "auto" else [
        ir.ConstInt(value, ir.DataType.INT64, _SSA_SPAN) for value in ids
    ]
    return ir.create_op_call(
        "block.make_mutex_tile",
        [ir.MakeTuple([ir.ConstInt(value, ir.DataType.INT64, _SSA_SPAN) for value in shape], _SSA_SPAN),
         ir.MakeTuple([], _SSA_SPAN), ir.MakeTuple(row, _SSA_SPAN)],
        {"dtype": dtype, "target_memory": memory, "memref_addr": 0, "memref_size": 256, "memref_id": 0},
        _SSA_SPAN,
    )


@pytest.mark.parametrize("auto_first", [True, False])
@pytest.mark.parametrize("peer_ids", ["auto", [3], [3, 7]])
def test_auto_merge_keeps_first_input_type(auto_first, peer_ids):
    parser = _ssa_type_parser()
    auto, peer = _ssa_tile("auto"), _ssa_tile(peer_ids)
    parser._auto_mutex_types[id(auto.type)] = auto.type
    if peer_ids == "auto":
        parser._auto_mutex_types[id(peer.type)] = peer.type
    values = (auto, peer) if auto_first else (peer, auto)
    first_type = values[0].type
    state = PhiState(type_equal=parser._ssa_types_equal)
    for value in values:
        state.propagate(value, fail_eagerly=True)

    # Preserve PhiState's original representative, including equal-count merges.
    assert state.ty is first_type
    if peer_ids != "auto":
        different = _ssa_tile([7, 9] if len(peer_ids) == 1 else [7])
        assert not parser._ssa_types_equal(peer.type, different.type)
    assert not ir.structural_equal(auto.type, _ssa_tile([3, 7]).type, enable_auto_mapping=False)


@pytest.mark.parametrize("kwargs", [
    {"dtype": ir.DataType.INT8}, {"shape": (2, 64)}, {"memory": ir.MemorySpace.Mat},
])
def test_auto_still_requires_matching_non_count_fields(kwargs):
    parser = _ssa_type_parser()
    auto = _ssa_tile("auto")
    parser._auto_mutex_types[id(auto.type)] = auto.type
    assert not parser._ssa_types_equal(auto.type, _ssa_tile([3, 7], **kwargs).type)
    assert not parser._ssa_types_equal(auto.type, _ssa_tile([]).type)
    assert not parser._ssa_types_equal(auto.type, ir.ScalarType(ir.DataType.INT64))


def test_tuple_merge_checks_each_leaf_and_nominal_metadata():
    debug_info = ir.IRDebugInfo()
    parser = _ssa_type_parser(debug_info)
    auto, single, multiple = _ssa_tile("auto"), _ssa_tile([3]), _ssa_tile([3, 7])
    parser._auto_mutex_types[id(auto.type)] = auto.type
    lhs = ir.TupleType([multiple.type, ir.TupleType([single.type])])
    rhs = ir.TupleType([auto.type, ir.TupleType([single.type])])
    info = ir.TupleTypeInfo(ir.TupleTypeKind.NAMED_TUPLE, "Pair", ["tile", "rest"])
    for ty in (lhs, rhs):
        debug_info.register_tuple_type_info(ty, info)
    state = PhiState(type_equal=parser._ssa_types_equal)
    for ty in (lhs, rhs):
        state.propagate(ir.Var("pair", ty, _SSA_SPAN), fail_eagerly=True)
    assert state.ty is lhs
    assert debug_info.get_tuple_type_info(state.ty) == info
    assert not parser._ssa_types_equal(lhs.types[0], single.type)
    assert not parser._ssa_types_equal(lhs.types[1].types[0], multiple.type)

    # AUTO in one position must not mask a manual mismatch in another position.
    bad = ir.TupleType([auto.type, ir.TupleType([multiple.type])])
    debug_info.register_tuple_type_info(bad, info)
    assert not parser._ssa_types_equal(lhs, bad)
    assert not parser._ssa_types_equal(lhs, ir.TupleType(rhs.types))
    assert not parser._ssa_types_equal(lhs, ir.TupleType([auto.type]))


@pytest.mark.parametrize("op_name", ["block.reinterpret", "block.subview"])
@pytest.mark.parametrize("ids", ["auto", [3], []])
def test_auto_view_provenance_survives_alias_and_tuple_projection(op_name, ids):
    parser = _ssa_type_parser()
    tile, multiple = _ssa_tile(ids), _ssa_tile([3, 7])
    if ids == "auto":
        parser._auto_mutex_types[id(tile.type)] = tile.type
    alias = ir.Var("alias", tile.type, _SSA_SPAN)
    pair = ir.MakeTuple([ir.ConstInt(0, ir.DataType.INT64, _SSA_SPAN), alias], _SSA_SPAN)
    holder = ir.Var("holder", pair.type, _SSA_SPAN)
    selected = ir.GetItemExpr(holder, ir.ConstInt(1, ir.DataType.INT64, _SSA_SPAN), _SSA_SPAN)
    # The view has a fresh Type object, just as op deduction creates one.
    view = ir.Call(op_name, [selected], {}, ir.MutexTileType(tile.type, tile.type.mutex_count), _SSA_SPAN)
    parser._transfer_auto_mutex_type(view, selected)
    assert (id(view.type) in parser._auto_mutex_types) == (ids == "auto")
    assert parser._ssa_types_equal(view.type, multiple.type) == (ids == "auto")
