# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""IR binding propagation, synchronization deduplication and runtime descriptor codegen."""

import re

from pypto_pro._errors import NameNotFound
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl import ir


def parse(kernel):
    return kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)[0]


def synced(kernel):
    return ir.Pass.insert_sync()(parse(kernel))


def cpp(kernel):
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    return _assemble_cv_source(cube, vector).content


def statements(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from statements(child)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt, ir.SectionStmt)):
        yield from statements(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from statements(stmt.then_body)
        if stmt.else_body is not None:
            yield from statements(stmt.else_body)


def calls(program, name):
    for function in program.functions.values():
        for stmt in statements(function.body):
            value = (
                stmt.value if isinstance(stmt, ir.AssignStmt) else stmt.expr if isinstance(stmt, ir.EvalStmt) else None
            )
            if isinstance(value, ir.Call) and value.name == name:
                yield value


@pytest.mark.parametrize(
    "valid_shape,ctor",
    [(None, "32, 64"), ([-1, -1], "32, 64"), ([16, 48], ""), ([-1, 48], "32"), ([16, -1], "64")],
)
def test_tile_constructor_initializes_only_dynamic_dimensions(valid_shape, ctor):
    explicit_valid_shape = valid_shape is not None

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[32, 64], pl.DT_FP32]):
        if explicit_valid_shape:
            tt = pl.TileType(
                shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=valid_shape
            )
        else:
            tt = pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tt, addr=0)
        with pl.section_vector():
            pl.store(out, tile, [0, 0])

    declarations = list(calls(parse(kernel), "block.make_mutex_tile"))
    assert len(declarations) == 1
    assert isinstance(declarations[0].type, ir.MutexTileType)
    assert declarations[0].type.mutex_count == 0
    code = cpp(kernel)
    assert f"tile_0_Type::PtoTile({ctor}), {{}}" in code


@pl.jit(auto_mutex=True)
def mixed_counts(index: pl.DT_INT64):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    source = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
    output = pl.make_tile_group(type=tt, addrs=8192, mutex_ids=[[0, 2], [1, 3]])
    with pl.section_vector():
        pl.move(output[index], source[index])


def test_parser_records_ids_without_inserting_sync():
    program = parse(mixed_counts)
    assert not list(calls(program, "system.mutex_lock_dyn"))
    declarations = list(calls(program, "block.make_mutex_tile"))
    assert [len(call.args[2].elements) for call in declarations] == [1, 1, 2, 2]
    assert "__mutexid" not in str(program)
    assert "_mutex_ids" not in str(program)


@pytest.mark.parametrize("field_write", ["scalar", "array_element", "whole_array"])
def test_struct_field_tile_read_is_materialized_and_synchronized(field_write):
    @pl.jit(auto_mutex=True)
    def kernel(unused: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[5])
        with pl.section_vector():
            message = pl.struct("Message", value=0.0, values=[0.0])
            if field_write == "scalar":
                message.value = pl.getval(group[0], 0)
            elif field_write == "array_element":
                message.values[0] = pl.getval(group[0], 0)
            else:
                message.values = [pl.getval(group[0], 0)]

    program = parse(kernel)
    assert len(list(calls(program, "block.getval"))) == 1
    (write,) = calls(program, "struct.set")
    assert isinstance(write.args[-1], ir.Var)

    sequence = []
    for function in synced(kernel).functions.values():
        for stmt in statements(function.body):
            value = (
                stmt.value if isinstance(stmt, ir.AssignStmt) else stmt.expr if isinstance(stmt, ir.EvalStmt) else None
            )
            if isinstance(value, ir.Call) and value.name in (
                "system.mutex_lock_dyn", "block.getval", "system.mutex_unlock_dyn", "struct.set"
            ):
                sequence.append(value.name)
    assert sequence == ["system.mutex_lock_dyn", "block.getval", "system.mutex_unlock_dyn", "struct.set"]
    code = cpp(kernel)
    assert code.count("get_buf(PIPE_S, 5, 0)") == code.count("rls_buf(PIPE_S, 5, 0)") == 1


def test_mixed_counts_generate_scalar_sync_and_one_shared_guard():
    program = synced(mixed_counts)
    locks = list(calls(program, "system.mutex_lock_dyn"))
    unlocks = list(calls(program, "system.mutex_unlock_dyn"))
    assert len(locks) == len(unlocks) == 3
    assert all(len(call.args) == 1 for call in locks + unlocks)
    assert all(isinstance(call.type, ir.NoneType) for call in locks + unlocks)
    assert all(call.args[0].type.dtype == pl.DT_INT64 for call in locks + unlocks)
    assert [str(call.args[0]) for call in locks] == [str(call.args[0]) for call in unlocks]
    code = cpp(mixed_counts)
    assert "int64_t mutex_ids[N]" in code
    # Only output slot 0 can overlap the source; slot 1 has candidates {2, 3}.
    assert code.count(" != ") == 1
    guards = re.findall(r"if \((__sync_needed_\w+)\)", code)
    assert len(guards) == 2 and guards[0] == guards[1]


@pl.jit(auto_mutex=True)
def disjoint(index: pl.DT_INT64):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    source = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
    output = pl.make_tile_group(type=tt, addrs=8192, mutex_ids=[2, 3])
    with pl.section_vector():
        pl.move(output[index], source[index])


def test_disjoint_candidates_do_not_emit_comparisons_and_v_sync_remains():
    code = cpp(disjoint)
    assert " != " not in code
    assert code.count("get_buf(PIPE_V") == 2
    assert code.count("rls_buf(PIPE_V") == 2


@pytest.mark.parametrize("descriptor", ["subview", "reinterpret", "subview_reinterpret"])
@pytest.mark.parametrize("overlap", [False, True])
def test_derived_tile_inherits_candidates_after_inplace_operations(descriptor, overlap):
    output_ids = [1, 2] if overlap else [2, 3]

    @pl.jit(auto_mutex=True)
    def kernel(src: pl.Tensor[[32, 32], pl.DT_FP32], index: pl.DT_INT64):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        source = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        output = pl.make_tile_group(type=tt, addrs=8192, mutex_ids=output_ids)
        with pl.section_vector():
            tile = source[index]
            pl.load(tile, src, [0, 0])
            pl.set_validshape(tile, [32, 32])
            if descriptor == "subview":
                view = tile[0:, 0:]
            elif descriptor == "reinterpret":
                view = pl.reinterpret(tile, shape=[32, 32])
            else:
                view = pl.reinterpret(tile[0:, 0:], shape=[32, 32])
            pl.move(output[index], view)

    program = synced(kernel)
    for name in ("block.load", "block.set_validshape", "block.move"):
        (call,) = calls(program, name)
        assert isinstance(call.type, ir.NoneType)
    code = cpp(kernel)
    assert code.count("get_buf(PIPE_MTE2") == code.count("rls_buf(PIPE_MTE2") == 1
    assert code.count("get_buf(PIPE_V") == code.count("rls_buf(PIPE_V") == 2
    assert code.count(" != ") == int(overlap)


@pl.jit(auto_mutex=True)
def same_value(index: pl.DT_INT64):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[[0, 2], [1, 3]])
    with pl.section_vector():
        tile = group[index]
        pl.add(tile, tile, tile)


def test_repeated_tile_is_locked_once_per_slot():
    code = cpp(same_value)
    assert code.count("get_buf(PIPE_V") == 2
    assert " != " not in code


def test_distinct_tiles_with_the_same_static_id_are_deduplicated():
    @pl.jit(auto_mutex=True)
    def kernel(unused: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        out = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[3])
        lhs = pl.make_tile_group(type=tt, addrs=128, mutex_ids=[7])
        rhs = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[7])
        with pl.section_vector():
            pl.add(out[0], lhs[0], rhs[0])

    program = synced(kernel)
    assert [call.args[0].value for call in calls(program, "system.mutex_lock_dyn")] == [3, 7]
    assert [call.args[0].value for call in calls(program, "system.mutex_unlock_dyn")] == [3, 7]
    assert "__sync_needed" not in str(program)


@pl.jit(auto_mutex=True)
def loop_union(n: pl.DT_INT64):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
    with pl.section_vector():
        tile = group[0]
        other = group[1]
        for i in pl.range(n):
            pl.add(tile, tile, other)
            tile = other


def test_loop_candidates_include_the_backedge():
    code = cpp(loop_union)
    assert "__mutexid" not in code
    assert " != " in code
    assert ".mutex_ids[0]" in code


@pytest.mark.parametrize("use_while", [False, True])
def test_loop_candidates_are_merged_per_variable(use_while):
    @pl.jit(auto_mutex=True)
    def kernel(n: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        first = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        second = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[2, 3])
        with pl.section_vector():
            a = first[0]
            b = second[0]
            if use_while:
                while n > 0:
                    pl.move(a, b)
                    a = first[1]
                    b = second[1]
                    n = n - 1
            else:
                for _ in pl.range(n):
                    pl.move(a, b)
                    a = first[1]
                    b = second[1]
            pl.move(a, b)

    program = synced(kernel)
    assert len(list(calls(program, "system.mutex_lock_dyn"))) == 4
    # Each variable changes at runtime, but their candidate sets are disjoint.
    assert " != " not in cpp(kernel)
    loop = next(
        stmt for function in program.functions.values() for stmt in statements(function.body)
        if isinstance(stmt, (ir.ForStmt, ir.WhileStmt))
    )
    getters = [
        stmt.value for stmt in statements(loop.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
        and stmt.value.name == "block.tile_mutex_id"
    ]
    assert len(getters) == 2


def test_loop_candidates_propagate_through_multiple_carried_values():
    @pl.jit(auto_mutex=True)
    def kernel(n: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1, 2])
        with pl.section_vector():
            a = group[0]
            b = group[1]
            last = group[2]
            for _ in pl.range(n):
                pl.move(a, last)
                a = b
                b = last

    # After two backedges a may have last's ID, so the first operation needs a guard.
    code = cpp(kernel)
    assert code.count(" != ") == 1
    assert code.count("get_buf(PIPE_V") == 2


def test_while_iter_and_result_merge_their_own_inputs():
    @pl.jit(auto_mutex=True)
    def kernel(n: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        inputs = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        output = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[[2, 3]])
        with pl.section_vector():
            tile = inputs[0]
            while True:
                pl.move(tile, tile)
                if n <= 0:
                    tile = output[0]
                    break
                tile = inputs[1]
                n = n - 1
                if n == 0:
                    tile = output[0]
                    break
                continue
            pl.move(tile, tile)

    program = synced(kernel)
    locks = list(calls(program, "system.mutex_lock_dyn"))
    unlocks = list(calls(program, "system.mutex_unlock_dyn"))
    # The body sees the initial and continue IDs; both exits return IDs 2 and 3.
    assert len(locks) == len(unlocks) == 3
    assert isinstance(locks[0].args[0], ir.Var)
    assert [call.args[0].value for call in locks[1:]] == [2, 3]
    assert [str(call.args[0]) for call in locks] == [str(call.args[0]) for call in unlocks]
    assert len(list(calls(program, "block.tile_mutex_id"))) == 1


def test_unused_incompatible_branch_result_does_not_acquire_mutex_slots():
    @pl.jit(auto_mutex=True)
    def kernel(flag: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        with pl.section_vector():
            tile = group[0]
            if flag:
                _unused = tile
            else:
                _unused = 1
            pl.move(tile, tile)

    program = synced(kernel)
    assert [call.args[0].value for call in calls(program, "system.mutex_lock_dyn")] == [0]
    assert not list(calls(program, "block.tile_mutex_id"))


def test_inline_tile_return_merges_candidates_across_empty_loop_inputs():
    def choose_tile(first, second, n):
        for i in pl.range(n):
            if i == 1:
                return second
        return first

    @pl.jit(auto_mutex=True)
    def kernel(n: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        with pl.section_vector():
            tile = group[0]
            selected = choose_tile(tile, group[1], n)
            pl.move(selected, tile)

    program = synced(kernel)
    locks = list(calls(program, "system.mutex_lock_dyn"))
    assert len(locks) == 2
    assert isinstance(locks[0].args[0], ir.Var)
    assert locks[1].args[0].value == 0
    assert len(list(calls(program, "block.tile_mutex_id"))) == 1
    assert "__sync_needed" in str(program)


@pytest.mark.parametrize("phase", [pl.AccPhase.Unspecified, pl.AccPhase.Partial, pl.AccPhase.Final, None])
def test_matmul_with_bias_preserves_acc_phase_synchronization(phase):
    has_phase = phase is not None

    @pl.jit(auto_mutex=True)
    def kernel(unused: pl.DT_INT64):
        acc_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
        lhs_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
        rhs_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
        bias_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias)
        acc = pl.make_tile_group(type=acc_type, addrs=0, mutex_ids=[0])
        lhs = pl.make_tile_group(type=lhs_type, addrs=0, mutex_ids=[1])
        rhs = pl.make_tile_group(type=rhs_type, addrs=0, mutex_ids=[2])
        bias = pl.make_tile_group(type=bias_type, addrs=0, mutex_ids=[3])
        with pl.section_cube():
            if has_phase:
                pl.matmul(acc[0], lhs[0], rhs[0], bias[0], phase=phase)
            else:
                pl.matmul(acc[0], lhs[0], rhs[0], bias[0])

    parsed = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)[0]
    assert len(list(calls(parsed, "block.matmul_bias"))) == 1
    program = ir.Pass.insert_sync()(parsed)
    expected = [1, 2, 3] if phase in (pl.AccPhase.Partial, pl.AccPhase.Final) else [0, 1, 2, 3]
    assert [call.args[0].value for call in calls(program, "system.mutex_lock_dyn")] == expected
    assert [call.args[0].value for call in calls(program, "system.mutex_unlock_dyn")] == expected


@pl.vector_function
def vf_choose_tile(a, b, out, unused, flag):
    if flag:
        source = a
    else:
        source = b
    value = vf.load_align(source, 0)
    vf.store_align(out, value, vf.update_mask(32, dtype=pl.DT_FP32), 0)


@pl.vector_function
def vf_loop_tile(a, b, out, unused, count):
    source = a
    for _ in pl.range(count):
        value = vf.load_align(source, 0)
        vf.store_align(out, value, vf.update_mask(32, dtype=pl.DT_FP32), 0)
        source = b


@pl.vector_function
def vf_offset_tile(a, b, out, unused, offset):
    source = a + offset
    value = vf.load_align(source, 0)
    vf.store_align(out, value, vf.update_mask(32, dtype=pl.DT_FP32), 0)


@pl.vector_function
def vf_nested_tile(a, b, out, unused, offset):
    vf_offset_tile(a, b, out, unused, offset)


@pl.vector_function
def vf_rebind_tile(a, b, out, unused, flag):
    if flag:
        a = b
    value = vf.load_align(a, 0)
    vf.store_align(out, value, vf.update_mask(32, dtype=pl.DT_FP32), 0)


@pytest.mark.parametrize("helper", [vf_choose_tile, vf_loop_tile, vf_offset_tile, vf_nested_tile, vf_rebind_tile])
def test_vf_synchronizes_declared_arguments_at_section_boundary(helper):
    @pl.jit(auto_mutex=True)
    def kernel(index: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1, 2, 3])
        with pl.section_vector():
            helper(group[0], group[1], group[2], group[3], index)

    program = parse(kernel)
    assert not list(calls(program, "system.mutex_lock_dyn"))
    declarations = list(calls(program, "system.vf_scope_args"))
    assert len(declarations) == 1
    assert isinstance(declarations[0].type, ir.NoneType)
    assert len(declarations[0].args) == 4  # Scalar arguments are not synchronization operands.
    program = ir.Pass.insert_sync()(program)
    body = next(iter(program.functions.values())).body
    all_stmts = list(statements(body))
    section = next(
        stmt for stmt in all_stmts if isinstance(stmt, ir.SectionStmt) and stmt.section_kind == ir.SectionKind.VF
    )
    locks = list(calls(program, "system.mutex_lock_dyn"))
    unlocks = list(calls(program, "system.mutex_unlock_dyn"))
    assert section.body.stmts[0].expr.name == "system.vf_scope_args"
    # Match the original parser: all Tile arguments, including unused ones, participate.
    assert [call.args[0].value for call in locks] == [0, 1, 2, 3]
    assert [call.args[0].value for call in unlocks] == [0, 1, 2, 3]
    assert all(call.args[0].type.dtype == pl.DT_INT64 for call in locks + unlocks)
    assert not list(calls(program, "block.tile_mutex_id"))
    section_index = all_stmts.index(section)
    for i, stmt in enumerate(all_stmts):
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call):
            if stmt.expr.name == "system.mutex_lock_dyn":
                assert i < section_index
            elif stmt.expr.name == "system.mutex_unlock_dyn":
                assert i > section_index
    assert not any(
        isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
        and stmt.expr.name.startswith("system.mutex_") for stmt in statements(section.body)
    )


@pytest.mark.parametrize("auto_mutex", [False, True])
def test_vf_dynamic_mutex_setup_stays_outside_scope(auto_mutex):
    mutex_ids = [0, 1] if auto_mutex else None

    @pl.vector_function
    def copy_tile(source, target, count):
        value = vf.load_align(source, 0)
        vf.store_align(target, value, vf.update_mask(count, dtype=pl.DT_FP32), 0)

    @pl.jit(auto_mutex=auto_mutex)
    def kernel(i: pl.DT_INT64, j: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=mutex_ids, depth=2)
        with pl.section_vector():
            copy_tile(group[i], group[j], 32)

    assert len(list(calls(parse(kernel), "system.vf_scope_args"))) == 1
    code = cpp(kernel)
    assert "vf_scope_args" not in code
    scope = re.search(r"__VEC_SCOPE__\s*\{.*?\n\s*\}", code, re.DOTALL)
    assert scope, code
    assert "__sync_" not in scope.group()
    assert "get_buf(" not in scope.group() and "rls_buf(" not in scope.group()
    if auto_mutex:
        before, after = code[:scope.start()], code[scope.end():]
        assert before.count(".mutex_ids[0]") == 2
        assert before.count(" != ") == 1
        assert before.count("get_buf(PIPE_V") == 2
        assert after.count("rls_buf(PIPE_V") == 2
        assert "get_buf(" not in after and "rls_buf(" not in before
    else:
        assert "get_buf(" not in code and "rls_buf(" not in code


@pl.jit(auto_mutex=True)
def branch_reinterpret(flag: pl.DT_INT64):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=[0, 8192], mutex_ids=[[0, 2], [1, 3]])
    with pl.section_vector():
        if flag:
            tile = group[0]
        else:
            tile = group[1]
        view = pl.reinterpret(tile, shape=[16, 64])
        pl.move(view, view)


def test_reinterpret_uses_the_same_runtime_source_for_address_and_ids():
    program = parse(branch_reinterpret)
    assert len(list(calls(program, "block.make_mutex_tile"))) == 2
    assert len(list(calls(program, "block.reinterpret"))) == 1
    code = cpp(branch_reinterpret)
    match = re.search(r"TASSIGN\((\w+)\.tile, \(uint64_t\)(\w+)\.tile\.data\(\)\)", code)
    assert match, code
    target, source = match.groups()
    assert f"{target}{{{target}_Type::PtoTile(16, 64), {{{source}.mutex_ids[0], {source}.mutex_ids[1]}}}};" in code
    assert "runtime_address" not in code


def test_parser_rejects_incompatible_binding_counts():
    @pl.jit(auto_mutex=True)
    def kernel(flag: pl.DT_INT64):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        b = pl.make_tile_group(type=tt, addrs=8192, mutex_ids=[[1, 2]])
        with pl.section_vector():
            if flag:
                tile = a[0]
            else:
                tile = b[0]
            pl.move(tile, tile)

    with pytest.raises(NameNotFound, match="potentially undefined variable"):
        parse(kernel)


def test_parser_rejects_bound_unbound_merge():
    @pl.jit(auto_mutex=True)
    def kernel(flag: pl.DT_INT64):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        b = pl.make_tile(tt, addr=8192)
        with pl.section_vector():
            if flag:
                tile = a[0]
            else:
                tile = b
            pl.move(tile, tile)

    with pytest.raises(NameNotFound, match="potentially undefined variable"):
        parse(kernel)


@pl.jit(auto_mutex=True)
def equal_candidates(i: pl.DT_INT64, j: pl.DT_INT64):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
    with pl.section_vector():
        pl.move(group[i], group[j])


def test_equal_candidate_sets_still_need_a_runtime_comparison():
    code = cpp(equal_candidates)
    assert code.count("get_buf(PIPE_V") == 2
    assert code.count(" != ") == 1
    guards = re.findall(r"if \((__sync_needed_\w+)\)", code)
    assert len(guards) == 2 and guards[0] == guards[1]


def test_tile_names_do_not_determine_the_runtime_representation():
    @pl.jit(auto_mutex=True)
    def kernel(unused: pl.DT_INT64):
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        vf_tile_ptr_group = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        with pl.section_vector():
            tile = vf_tile_ptr_group[0]
            pl.add(tile, tile, tile)

    code = cpp(kernel)
    assert re.search(r"TADD\([^;\n]+\.tile,\s*[^;\n]+\.tile,\s*[^;\n]+\.tile\);", code), code


def _validate_expression(expr):
    function = ir.Function("validate", [], [], ir.EvalStmt(expr, expr.span), expr.span)
    return ir.Pass.insert_sync()(ir.Program([function], "validate", expr.span))


def test_mutex_getter_checks_slot_against_the_value_binding():
    from pypto_pro.ir.op.block_ops import make_mutex_tile_expr

    tile = make_mutex_tile_expr([1, 32], pl.DT_FP32, pl.MemorySpace.Vec, addr=0, mutex_ids=[1, 2])
    getter = ir.create_op_call(
        "block.tile_mutex_id", [tile, ir.ConstInt(1, pl.DT_INT64, tile.span)], {}, tile.span
    )
    assert getter.type.dtype == pl.DT_INT64
    _validate_expression(getter)
    with pytest.raises(ValueError, match="outside the tile's mutex binding"):
        ir.create_op_call(
            "block.tile_mutex_id", [tile, ir.ConstInt(2, pl.DT_INT64, tile.span)], {}, tile.span
        )


def test_binding_analysis_does_not_validate_reinterpret_storage():
    from pypto_pro.ir.op.block_ops import _ir_reinterpret, make_mutex_tile_expr

    tile = make_mutex_tile_expr([2, 32], pl.DT_FP32, pl.MemorySpace.Vec, addr=0, mutex_ids=[1])
    span = tile.span
    offsets = ir.MakeTuple([ir.ConstInt(1, pl.DT_INT64, span), ir.ConstInt(0, pl.DT_INT64, span)], span)
    window = ir.MakeTuple([ir.ConstInt(1, pl.DT_INT64, span), ir.ConstInt(32, pl.DT_INT64, span)], span)
    view = ir.create_op_call("block.subview", [tile, offsets, window], {}, span)
    result = _ir_reinterpret(view, shape=[2, 32])
    _validate_expression(result)


def test_reinterpret_skips_storage_checks_for_runtime_branch_selection():
    from pypto_pro.ir.op.block_ops import _ir_reinterpret, make_mutex_tile_expr

    from pypto.pypto_impl.codegen import CCECodegen

    first = make_mutex_tile_expr([1, 32], pl.DT_FP32, pl.MemorySpace.Vec, addr=0, size=256, mutex_ids=[0])
    second = make_mutex_tile_expr([1, 32], pl.DT_FP32, pl.MemorySpace.Vec, addr=512, size=128, mutex_ids=[1])
    span = first.span
    selected = ir.Var("selected", first.type, span)
    flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), span)
    first_var = ir.Var("first", first.type, span)
    second_var = ir.Var("second", second.type, span)
    branch = ir.IfStmt(flag, ir.YieldStmt([first_var], span), ir.YieldStmt([second_var], span), [selected], span)
    view = _ir_reinterpret(selected, shape=[2, 32])
    output = ir.Var("view", view.type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(first_var, first, span), ir.AssignStmt(second_var, second, span),
         branch, ir.AssignStmt(output, view, span)], span,
    )
    function = ir.Function("validate", [flag], [], body, span)
    program = ir.Pass.insert_sync()(ir.Program([function], "validate", span, ir.IRDebugInfo()))
    code = CCECodegen(ir.SectionKind.Vector).generate_single(program, NpuArch.DAV_3510)
    assert "selected.tile.data()" in code
