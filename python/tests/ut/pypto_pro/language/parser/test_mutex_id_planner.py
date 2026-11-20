# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from pathlib import Path

import pytest

from pypto.pypto_impl import ir


class TileGroupProgramBuilder:
    """Build the same named TileGroup tuples consumed by the allocation pass."""

    def __init__(self):
        self.declarations = []
        self.debug_info = ir.IRDebugInfo()
        self.groups = {}

    def collect_group(self, tiles, mutex_ids, group_name):
        members = []
        for tile, row in zip(tiles, mutex_ids.elements, strict=True):
            span = tile.span
            tile_type = tile.type
            memref = tile_type.memref
            one = ir.ConstInt(1, ir.DataType.INT64, span)
            call = ir.create_op_call(
                "block.make_mutex_tile",
                [ir.MakeTuple([one, one], span), ir.MakeTuple([], span), row],
                {"dtype": ir.DataType.INT8, "target_memory": memref.memory_space,
                 "memref_addr": memref.addr.value, "memref_size": memref.size,
                 "memref_id": len(self.declarations)},
                span,
            )
            member = ir.Var(tile.name, call.type, span)
            self.declarations.append(ir.AssignStmt([member], call, span))
            members.append(member)
        self.groups[group_name] = members
        span = tiles[0].span
        members_tuple = ir.MakeTuple(members, span)
        members_var = ir.Var(f"{group_name}_tiles", members_tuple.type, span)
        self.declarations.append(ir.AssignStmt([members_var], members_tuple, span))
        group = ir.MakeTuple([members_var], span)
        self.debug_info.register_tuple_type_info(
            group.type, ir.TupleTypeInfo(ir.TupleTypeKind.NAMED_TUPLE, "tile_group", ["tiles"]),
        )
        group_var = ir.Var(group_name, group.type, span)
        self.declarations.append(ir.AssignStmt([group_var], group, span))


_SPAN = ir.Span("planner_test.py", 1, 1)
_TILE_INDEX = 0
_PLACEHOLDER_INDEX = 0


def _tile(addr: int, *, size: int = 1, memory_space=ir.MemorySpace.Vec, span=_SPAN):
    global _TILE_INDEX
    address = ir.ConstInt(addr, ir.DataType.INT64, span)
    memref = ir.MemRef(memory_space, address, size, span)
    tile_type = ir.TileType([1], ir.DataType.INT8, memref)
    tile = ir.Var(f"tile_{_TILE_INDEX}", tile_type, span)
    _TILE_INDEX += 1
    return tile


def _collect_auto(manager, addrs, *, size=1, memory_space=ir.MemorySpace.Vec, group_name=None, span=_SPAN):
    global _PLACEHOLDER_INDEX
    group_name = group_name or f"auto_group_{_TILE_INDEX}"
    tiles = [_tile(addr, size=size, memory_space=memory_space, span=span) for addr in addrs]
    rows = []
    for _ in tiles:
        placeholder = ir.Var(
            f"__pypto_auto_mutex_id_{_PLACEHOLDER_INDEX}",
            ir.ScalarType(ir.DataType.INT64),
            span,
        )
        _PLACEHOLDER_INDEX += 1
        rows.append(ir.MakeTuple([placeholder], span))
    mutex_ids = ir.MakeTuple(rows, span)
    manager.collect_group(tiles, mutex_ids, group_name)
    return mutex_ids


def _collect_manual(manager, addrs, ids, *, size=1, memory_space=ir.MemorySpace.Vec, group_name=None):
    group_name = group_name or f"manual_group_{_TILE_INDEX}"
    tiles = [_tile(addr, size=size, memory_space=memory_space) for addr in addrs]
    mutex_ids = ir.MakeTuple(
        [
            ir.MakeTuple(
                [ir.ConstInt(mutex_id, ir.DataType.INT64, _SPAN) for mutex_id in tile_ids],
                _SPAN,
            )
            for tile_ids in ids
        ],
        _SPAN,
    )
    manager.collect_group(tiles, mutex_ids, group_name)


def _resolve_groups(manager, groups):
    rows = [
        tile_ids
        for group in groups
        for tile_ids in group.elements
    ]
    value = ir.MakeTuple(rows, _SPAN)
    holder = ir.Var("resolved_mutex_ids", value.type, _SPAN)
    body = ir.SeqStmts(manager.declarations + [ir.AssignStmt([holder], value, _SPAN)], _SPAN)
    func = ir.Function("main", [], [], body, _SPAN)
    program = ir.Pass.allocate_mutex_ids()(ir.Program([func], "mutex_plan", _SPAN, manager.debug_info))
    stmt = program["main"].body
    if isinstance(stmt, ir.SeqStmts):
        stmt = stmt.stmts[-1]
    resolved = tuple(
        tuple(element.value for element in row.elements)
        for row in stmt.value.elements
    )

    result = []
    offset = 0
    for group in groups:
        result.append(tuple(resolved[offset + index] for index in range(len(group.elements))))
        offset += len(group.elements)
    return tuple(result)


@pytest.mark.parametrize(
    ("count", "expected_min", "expected_max"),
    [(1, 0, 1), (32, 1, 1), (33, 1, 2), (64, 2, 2)],
)
def test_independent_auto_tiles_are_balanced(count, expected_min, expected_max):
    manager = TileGroupProgramBuilder()
    group = _collect_auto(manager, list(range(count)))

    assigned = [ids[0] for ids in _resolve_groups(manager, [group])[0]]
    usage_counts = [assigned.count(mutex_id) for mutex_id in range(32)]
    assert min(usage_counts) == expected_min
    assert max(usage_counts) == expected_max
    if count <= 32:
        assert len(set(assigned)) == count


def test_manual_multi_ids_are_flattened_before_auto_allocation():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0, 10, 20], [[0, 3], [1, 3], [2, 4]])
    auto = _collect_auto(manager, [100])

    assert _resolve_groups(manager, [auto]) == (((5,),),)


def test_discarded_mutex_tile_does_not_affect_allocation():
    manager = TileGroupProgramBuilder()
    one = ir.ConstInt(1, ir.DataType.INT64, _SPAN)
    seven = ir.ConstInt(7, ir.DataType.INT64, _SPAN)
    discarded = ir.create_op_call(
        "block.make_mutex_tile",
        [ir.MakeTuple([one, one], _SPAN), ir.MakeTuple([], _SPAN), ir.MakeTuple([seven], _SPAN)],
        {"dtype": ir.DataType.INT8, "target_memory": ir.MemorySpace.Vec,
         "memref_addr": 0, "memref_size": 16, "memref_id": 0},
        _SPAN,
    )
    manager.declarations.append(ir.AssignStmt([], discarded, _SPAN))
    auto = _collect_auto(manager, [0], size=16)

    assert _resolve_groups(manager, [auto]) == (((0,),),)


def test_group_members_tuple_is_resolved_through_an_alias():
    manager = TileGroupProgramBuilder()
    group = _collect_auto(manager, [0, 8], size=4)
    members = manager.declarations[-2]
    alias = ir.Var("members_alias", members.value.type, _SPAN)
    manager.declarations[-2:-1] = [
        ir.AssignStmt([alias], members.value, _SPAN),
        ir.AssignStmt(members.targets, alias, _SPAN),
    ]

    assert _resolve_groups(manager, [group]) == (((0,), (1,)),)


def test_containment_parent_is_the_union_of_its_children():
    manager = TileGroupProgramBuilder()
    first = _collect_auto(manager, [0], size=100)
    second = _collect_auto(manager, [0], size=220)
    third = _collect_auto(manager, [150], size=70)

    assert _resolve_groups(manager, [first, second, third]) == (((0,),), ((0, 1),), ((1,),))


def test_deeper_auto_parents_finish_before_root_limits_sibling_ids():
    manager = TileGroupProgramBuilder()
    root = _collect_auto(manager, [0], size=128, group_name="root")
    parents = _collect_auto(manager, [0, 64], size=64, group_name="parents")
    leaves = _collect_auto(manager, [0, 32, 64], size=32, group_name="leaves")

    root_ids, parent_ids, leaf_ids = _resolve_groups(manager, [root, parents, leaves])

    # A contains two leaves; B contains one but shares A's two-ID count.
    # Fixing R to the three leaf IDs first would force A and B to overlap.
    assert tuple(map(len, leaf_ids)) == (1, 1, 1)
    assert tuple(map(len, parent_ids)) == (2, 2)
    left, right = map(set, parent_ids)
    assert left.isdisjoint(right)
    assert set(root_ids[0]) == left | right
    assert len(root_ids[0]) == 4


@pytest.mark.parametrize(
    ("left_addr", "left_size", "right_addr", "right_size"),
    [(0, 32, 0, 32), (0, 64, 16, 16), (16, 16, 0, 64), (0, 32, 0, 64), (32, 32, 0, 64)],
)
def test_exact_and_contained_aliases_share_one_id(left_addr, left_size, right_addr, right_size):
    manager = TileGroupProgramBuilder()
    left = _collect_auto(manager, [left_addr], size=left_size)
    right = _collect_auto(manager, [right_addr], size=right_size)

    assert _resolve_groups(manager, [left, right]) == (((0,),), ((0,),))


@pytest.mark.parametrize(
    ("left_addr", "left_size", "right_addr", "right_size"),
    [(0, 32, 16, 32), (16, 32, 0, 32)],
)
def test_crossing_auto_aliases_are_rejected(left_addr, left_size, right_addr, right_size):
    manager = TileGroupProgramBuilder()
    left = _collect_auto(manager, [left_addr], size=left_size, group_name="left_group")
    right = _collect_auto(manager, [right_addr], size=right_size, group_name="right_group")

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(manager, [left, right])
    message = str(exc_info.value)
    assert "Enum: ExternalError::INVALID_OPERATION" in message
    assert message.count("Vec Tile at line 1") == 2
    overlap_start = max(left_addr, right_addr)
    overlap_end = min(left_addr + left_size, right_addr + right_size)
    assert f"overlap is [0x{overlap_start:x}, 0x{overlap_end:x})" in message


def test_crossing_same_name_groups_are_distinguished_by_source_line():
    manager = TileGroupProgramBuilder()
    left = _collect_auto(manager, [0], size=32, group_name="shared_group", span=ir.Span("planner_test.py", 10, 1))
    right = _collect_auto(
        manager, [16], size=32, group_name="shared_group", span=ir.Span("planner_test.py", 20, 1)
    )

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(manager, [left, right])
    message = str(exc_info.value)
    assert "Vec Tile at line 10" in message
    assert "Vec Tile at line 20" in message
    assert "--> planner_test.py:20:1" in message


def test_crossing_manual_tiles_are_rejected_when_component_contains_auto():
    manager = TileGroupProgramBuilder()
    auto = _collect_auto(manager, [0], size=128)
    _collect_manual(manager, [0], [[5]], size=80)
    _collect_manual(manager, [48], [[5]], size=80)

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(manager, [auto])
    assert "Enum: ExternalError::INVALID_OPERATION" in str(exc_info.value)


def test_adjacent_intervals_and_different_memory_spaces_do_not_alias():
    manager = TileGroupProgramBuilder()
    left0 = _collect_auto(manager, [0], size=16, memory_space=ir.MemorySpace.Left)
    left1 = _collect_auto(manager, [16], size=16, memory_space=ir.MemorySpace.Left)
    right = _collect_auto(manager, [0], size=16, memory_space=ir.MemorySpace.Right)

    assigned = _resolve_groups(manager, [left0, left1, right])
    assert {group[0][0] for group in assigned} == {0, 1, 2}


def test_cube_and_vector_use_independent_id_pools():
    cube = TileGroupProgramBuilder()
    vector = TileGroupProgramBuilder()
    cube_group = _collect_auto(cube, [0], memory_space=ir.MemorySpace.Mat)
    vector_group = _collect_auto(vector, [128])

    assert _resolve_groups(cube, [cube_group]) == (((0,),),)
    assert _resolve_groups(vector, [vector_group]) == (((0,),),)


def test_one_manual_id_is_reused_by_all_aliased_auto_tiles():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0], [[9]], size=64)
    first = _collect_auto(manager, [0], size=32)
    second = _collect_auto(manager, [32], size=32)

    assert _resolve_groups(manager, [first, second]) == (((9,),), ((9,),))


def test_exact_manual_alias_gives_auto_leaf_multiple_ids():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0], [[5, 9]], size=32, group_name="manual_group")
    auto = _collect_auto(manager, [0], size=32)

    assert _resolve_groups(manager, [auto]) == (((5, 9),),)


def test_restricted_candidate_job_is_allocated_before_unrestricted_job():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0], [[7]], size=16)
    restricted = _collect_auto(manager, [0], size=16)
    unrestricted = _collect_auto(manager, [100], size=16)

    assert _resolve_groups(manager, [restricted, unrestricted]) == (((7,),), ((0,),))


def test_larger_auto_component_wins_address_order_tie_break():
    manager = TileGroupProgramBuilder()
    singleton = _collect_auto(manager, [100], size=16)
    component = _collect_auto(manager, [0, 0, 0], size=16)

    assert _resolve_groups(manager, [singleton, component]) == (((1,),), ((0,), (0,), (0,)))


def test_allocation_is_deterministic_across_collection_order():
    assignments = []
    for order in ((300, 100, 200), (100, 200, 300)):
        manager = TileGroupProgramBuilder()
        groups = [_collect_auto(manager, [addr]) for addr in order]
        resolved = _resolve_groups(manager, groups)
        assignments.append({addr: ids[0][0] for addr, ids in zip(order, resolved)})

    assert assignments[0] == assignments[1]


def test_manual_only_alias_component_is_not_validated_without_auto():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0, 8], [[5], [9]], size=32)

    assert _resolve_groups(manager, []) == ()


def test_manual_only_alias_component_is_unchanged_when_auto_exists_elsewhere():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0], [[5]], size=32)
    _collect_manual(manager, [8], [[9]], size=32)
    auto = _collect_auto(manager, [100], size=16)

    assert _resolve_groups(manager, [auto]) == (((0,),),)


def test_auto_parent_contains_the_union_of_manual_children():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0], [[5]], size=32, group_name="manual_a")
    _collect_manual(manager, [32], [[9]], size=32, group_name="manual_b")
    auto = _collect_auto(manager, [0], size=64)

    assert _resolve_groups(manager, [auto]) == (((5, 9),),)


def test_same_layer_uses_maximum_child_union_count_and_pads_auto_nodes():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0], [[2, 5, 7, 9]], size=128, group_name="root")
    parent = _collect_auto(manager, [0], size=64, group_name="parent")
    leaf = _collect_auto(manager, [0], size=32, group_name="leaf")
    _collect_manual(manager, [32], [[5]], size=32, group_name="manual_leaf")
    _collect_manual(manager, [64], [[7, 9]], size=64, group_name="manual_sibling")

    assert _resolve_groups(manager, [parent, leaf]) == (((2, 5),), ((2,),))


def test_manual_descendant_must_be_subset_of_manual_ancestor_through_auto_parent():
    manager = TileGroupProgramBuilder()
    _collect_manual(manager, [0], [[2, 5]], size=128, group_name="root")
    auto = _collect_auto(manager, [0], size=64, group_name="auto_parent")
    _collect_manual(manager, [0], [[7]], size=32, group_name="manual_leaf")

    with pytest.raises(ValueError, match="containing Tile must include every mutex ID"):
        _resolve_groups(manager, [auto])


def test_cross_tree_group_can_expand_an_auto_leaf_to_match_parent_count():
    manager = TileGroupProgramBuilder()
    parents = _collect_auto(manager, [0, 256], size=128, group_name="parents")
    children = _collect_auto(manager, [0, 64], size=64, group_name="children")

    parent_ids, child_ids = _resolve_groups(manager, [parents, children])

    assert tuple(map(len, child_ids)) == (1, 1)
    assert tuple(map(len, parent_ids)) == (2, 2)
    assert set(parent_ids[0]) == {ids[0] for ids in child_ids}


@pytest.mark.parametrize("manual_ids", [[0], [0, 1]])
def test_saturated_auto_parent_reuses_ids_across_children(manual_ids):
    manager = TileGroupProgramBuilder()
    parents = _collect_auto(manager, [0, 256], size=192, group_name="parents")
    _collect_manual(manager, [256], [manual_ids], size=192)
    children = _collect_auto(manager, [0, 64, 128], size=64, group_name="children")
    _collect_manual(manager, [0], [manual_ids], size=64)

    parent_ids, child_ids = _resolve_groups(manager, [parents, children])

    expected_ids = tuple(manual_ids)
    assert parent_ids == (expected_ids, expected_ids)
    assert child_ids == (expected_ids, expected_ids, expected_ids)


def test_saturated_grandparent_restricts_ids_through_unassigned_parent():
    manager = TileGroupProgramBuilder()
    parents = _collect_auto(manager, [0, 256], size=192, group_name="parents")
    _collect_manual(manager, [256], [[0, 1]], size=192)
    intermediate = _collect_auto(manager, [0], size=128, group_name="intermediate")
    leaves = _collect_auto(manager, [0, 128], size=64, group_name="leaves")
    _collect_manual(manager, [128], [[0, 1]], size=64)

    parent_ids, intermediate_ids, leaf_ids = _resolve_groups(manager, [parents, intermediate, leaves])

    assert parent_ids == ((0, 1), (0, 1))
    assert intermediate_ids == ((0, 1),)
    assert leaf_ids == ((0, 1), (0, 1))


def test_auto_parent_with_spare_capacity_allows_new_ids():
    manager = TileGroupProgramBuilder()
    parents = _collect_auto(manager, [0, 256], size=192, group_name="parents")
    _collect_manual(manager, [256], [[0, 1]], size=192)
    children = _collect_auto(manager, [0, 64], size=64, group_name="children")
    _collect_manual(manager, [0], [[0]], size=64)

    parent_ids, child_ids = _resolve_groups(manager, [parents, children])

    assert parent_ids == ((0, 2), (0, 1))
    assert child_ids == ((0,), (2,))


def test_tile_group_count_constraint_can_make_cross_tree_plan_unsatisfiable():
    manager = TileGroupProgramBuilder()
    group = _collect_auto(manager, [0, 8], size=4, group_name="cross_tree_group")
    _collect_manual(manager, [0], [[0]], size=2, group_name="left_leaf")
    _collect_manual(manager, [2], [[1]], size=2, group_name="right_leaf")
    _collect_manual(manager, [8], [[5]], size=8, group_name="right_ancestor")

    with pytest.raises(ValueError, match="Unable to assign the same mutex ID count") as exc_info:
        _resolve_groups(manager, [group])
    message = str(exc_info.value)
    assert "requires at least 2 IDs but allows at most 1" in message
    assert "separate TileGroups" in message


def _ssa_fixture():
    manager = TileGroupProgramBuilder()
    _collect_auto(manager, [0], size=8, group_name="left")
    _collect_auto(manager, [32], size=8, group_name="right")
    _collect_manual(manager, [0], [[2, 5]], size=8, group_name="pin")
    flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), _SPAN)
    return manager, manager.groups["left"][0], manager.groups["right"][0], flag


@pytest.fixture
def check_ssa_golden(monkeypatch, request):
    # These counters name test declarations, so reset them for stable full-text goldens.
    monkeypatch.setitem(globals(), "_TILE_INDEX", 0)
    monkeypatch.setitem(globals(), "_PLACEHOLDER_INDEX", 0)

    def check(function):
        text = str(function).rstrip("\n") + "\n"
        path = Path(__file__).with_suffix("") / f"{request.node.name}.pypto"
        if os.environ.get("PYPTO_RENDER_IR") == "1":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return
        assert text == path.read_text(encoding="utf-8")

    return check


def _resolve_ssa(manager, statements, params, *, check_golden=None):
    body = ir.SeqStmts(manager.declarations + statements, _SPAN)
    function = ir.Function("main", params, [], body, _SPAN)
    program = ir.Program([function], "ssa_mutex_plan", _SPAN, manager.debug_info)
    resolved = ir.Pass.allocate_mutex_ids()(program)
    diagnostics = ir.IRVerifier.create_default().verify(resolved)
    assert not diagnostics, ir.IRVerifier.generate_report(diagnostics)
    if check_golden is not None:
        check_golden(resolved["main"])
    return resolved["main"].body.stmts[len(manager.declarations):]


def _if_values(flag, then_value, else_value, result):
    return ir.IfStmt(
        flag, ir.SeqStmts([ir.YieldStmt([then_value], _SPAN)], _SPAN),
        ir.SeqStmts([ir.YieldStmt([else_value], _SPAN)], _SPAN), [result], _SPAN,
    )


def _for_values(initials, iters, next_values, results, prefix=()):
    index = ir.Var("index", ir.ScalarType(ir.DataType.INT64), _SPAN)
    zero = ir.ConstInt(0, ir.DataType.INT64, _SPAN)
    one = ir.ConstInt(1, ir.DataType.INT64, _SPAN)
    return ir.ForStmt(
        index, zero, one, one,
        [ir.IterArg(var, init) for var, init in zip(iters, initials, strict=True)],
        ir.SeqStmts([*prefix, ir.ContinueStmt(next_values, _SPAN)], _SPAN), results, _SPAN,
    )


def test_ssa_if_pads_independent_auto_tile_and_remaps_uses(check_ssa_golden):
    manager, left, right, flag = _ssa_fixture()
    result = ir.Var("selected", left.type, _SPAN)
    alias = ir.Var("alias", result.type, _SPAN)
    branch, use = _resolve_ssa(
        manager, [_if_values(flag, left, right, result), ir.AssignStmt([alias], result, _SPAN)], [flag],
        check_golden=check_ssa_golden,
    )

    assert branch.return_vars[0].type.mutex_count == 2
    assert branch.then_body.stmts[-1].value[0].type.mutex_count == 2
    assert branch.else_body.stmts[-1].value[0].type.mutex_count == 2
    assert use.targets[0].type.mutex_count == use.value.type.mutex_count == 2


def test_ssa_for_backedge_and_swapped_slots_share_counts(check_ssa_golden):
    manager, left, right, _ = _ssa_fixture()
    first = ir.Var("first_iter", left.type, _SPAN)
    second = ir.Var("second_iter", right.type, _SPAN)
    results = [ir.Var("first_result", left.type, _SPAN), ir.Var("second_result", right.type, _SPAN)]
    loop, = _resolve_ssa(
        manager, [_for_values([left, right], [first, second], [second, first], results)], [],
        check_golden=check_ssa_golden,
    )

    assert [arg.iterVar.type.mutex_count for arg in loop.iter_args] == [2, 2]
    assert [value.type.mutex_count for value in loop.body.stmts[-1].value] == [2, 2]
    assert [var.type.mutex_count for var in loop.return_vars] == [2, 2]


def test_ssa_nested_loop_does_not_join_outer_and_inner_slots():
    manager, left, right, _ = _ssa_fixture()
    # Equal names must not connect distinct SSA definitions in nested scopes.
    outer_iter = ir.Var("carried", left.type, _SPAN)
    inner_iter = ir.Var("carried", right.type, _SPAN)
    outer_result = ir.Var("outer_result", left.type, _SPAN)
    inner_result = ir.Var("inner_result", right.type, _SPAN)
    inner = _for_values([right], [inner_iter], [inner_iter], [inner_result])
    outer = _for_values([left], [outer_iter], [outer_iter], [outer_result], [inner])
    resolved, = _resolve_ssa(manager, [outer], [])

    assert resolved.return_vars[0].type.mutex_count == 2
    assert resolved.body.stmts[0].return_vars[0].type.mutex_count == 1
    assert resolved.body.stmts[-1].value[0].same_as(resolved.iter_args[0].iterVar)
    inner_loop = resolved.body.stmts[0]
    assert inner_loop.body.stmts[-1].value[0].same_as(inner_loop.iter_args[0].iterVar)


def test_ssa_nested_if_result_updates_outer_loop_and_uses(check_ssa_golden):
    manager, left, right, flag = _ssa_fixture()
    carried = ir.Var("carried", right.type, _SPAN)
    selected = ir.Var("selected", right.type, _SPAN)
    result = ir.Var("result", right.type, _SPAN)
    alias = ir.Var("alias", right.type, _SPAN)
    branch = _if_values(flag, left, carried, selected)
    loop = _for_values([right], [carried], [selected], [result], [branch])
    resolved, use = _resolve_ssa(
        manager, [loop, ir.AssignStmt([alias], result, _SPAN)], [flag], check_golden=check_ssa_golden,
    )

    inner = resolved.body.stmts[0]
    assert resolved.iter_args[0].initValue.type.mutex_count == 2
    assert resolved.iter_args[0].iterVar.type.mutex_count == 2
    assert inner.return_vars[0].type.mutex_count == resolved.return_vars[0].type.mutex_count == 2
    assert inner.else_body.stmts[-1].value[0].same_as(resolved.iter_args[0].iterVar)
    assert resolved.body.stmts[-1].value[0].same_as(inner.return_vars[0])
    assert use.value.same_as(resolved.return_vars[0])
    assert use.targets[0].type.mutex_count == 2
    assert all(var.type.mutex_count == 1 for var in [right, carried, selected, result, alias])


@pytest.mark.parametrize("header_is_none", [False, True])
def test_ssa_while_exit_keeps_its_own_type(header_is_none):
    manager, left, right, flag = _ssa_fixture()
    initial = ir.Var("empty", ir.NoneType.get(), _SPAN) if header_is_none else left
    exit_value = left if header_is_none else right
    iter_var = ir.Var("carried", initial.type, _SPAN)
    result = ir.Var("exit_result", exit_value.type, _SPAN)
    guard = ir.IfStmt(
        flag, ir.SeqStmts([ir.BreakStmt([exit_value], _SPAN)], _SPAN),
        ir.SeqStmts([ir.YieldStmt([], _SPAN)], _SPAN), [], _SPAN,
    )
    loop = ir.WhileStmt(
        ir.ConstBool(True, _SPAN), [ir.IterArg(iter_var, initial)],
        ir.SeqStmts([guard, ir.ContinueStmt([iter_var], _SPAN)], _SPAN), [result], _SPAN,
    )
    params = [flag, initial] if header_is_none else [flag]
    resolved, = _resolve_ssa(manager, [loop], params)

    if header_is_none:
        assert isinstance(resolved.iter_args[0].iterVar.type, ir.NoneType)
        assert resolved.return_vars[0].type.mutex_count == 2
    else:
        assert resolved.iter_args[0].iterVar.type.mutex_count == 2
        assert resolved.return_vars[0].type.mutex_count == 1
    assert resolved.body.stmts[0].then_body.stmts[0].value[0].type.mutex_count == (2 if header_is_none else 1)


def test_ssa_while_guard_links_exit_to_backedge():
    manager, left, right, flag = _ssa_fixture()
    iter_var = ir.Var("carried", left.type, _SPAN)
    result = ir.Var("exit_result", left.type, _SPAN)
    guard = ir.IfStmt(
        flag, ir.SeqStmts([ir.BreakStmt([iter_var], _SPAN)], _SPAN),
        ir.SeqStmts([ir.YieldStmt([], _SPAN)], _SPAN), [], _SPAN,
    )
    loop = ir.WhileStmt(
        ir.ConstBool(True, _SPAN), [ir.IterArg(iter_var, left)],
        ir.SeqStmts([guard, ir.ContinueStmt([right], _SPAN)], _SPAN), [result], _SPAN,
    )
    resolved, = _resolve_ssa(manager, [loop], [flag])

    assert resolved.iter_args[0].iterVar.type.mutex_count == 2
    assert resolved.return_vars[0].type.mutex_count == 2
    assert resolved.body.stmts[-1].value[0].type.mutex_count == 2


def test_ssa_while_condition_uses_rebuilt_iter_var(check_ssa_golden):
    manager, left, right, flag = _ssa_fixture()
    initial = ir.MakeTuple([flag, left], _SPAN)
    next_value = ir.MakeTuple([flag, right], _SPAN)
    carried = ir.Var("carried", initial.type, _SPAN)
    result = ir.Var("result", initial.type, _SPAN)
    condition = ir.GetItemExpr(carried, ir.ConstInt(0, ir.DataType.INT64, _SPAN), _SPAN)
    guard = ir.IfStmt(
        flag, ir.SeqStmts([ir.BreakStmt([carried], _SPAN)], _SPAN),
        ir.SeqStmts([ir.YieldStmt([], _SPAN)], _SPAN), [], _SPAN,
    )
    loop = ir.WhileStmt(
        condition, [ir.IterArg(carried, initial)],
        ir.SeqStmts([guard, ir.ContinueStmt([next_value], _SPAN)], _SPAN), [result], _SPAN,
    )
    resolved, = _resolve_ssa(manager, [loop], [flag], check_golden=check_ssa_golden)

    assert resolved.iter_args[0].iterVar.type.types[1].mutex_count == 2
    assert resolved.return_vars[0].type.types[1].mutex_count == 2
    assert resolved.condition.value.same_as(resolved.iter_args[0].iterVar)
    assert resolved.body.stmts[0].then_body.stmts[0].value[0].same_as(resolved.iter_args[0].iterVar)
    assert resolved.body.stmts[-1].value[0].type.types[1].mutex_count == 2
    assert not resolved.condition.value.same_as(carried)


def test_ssa_tuple_merge_keeps_unrelated_positions_independent():
    manager, left, right, flag = _ssa_fixture()
    _collect_auto(manager, [64], size=8, group_name="other")
    first = manager.groups["other"][0]
    _collect_auto(manager, [96], size=8, group_name="last")
    last = manager.groups["last"][0]
    then_value = ir.MakeTuple([left, ir.MakeTuple([right], _SPAN)], _SPAN)
    else_value = ir.MakeTuple([first, ir.MakeTuple([last], _SPAN)], _SPAN)
    result = ir.Var("pair", then_value.type, _SPAN)
    branch, = _resolve_ssa(manager, [_if_values(flag, then_value, else_value, result)], [flag])

    merged = branch.return_vars[0].type
    assert merged.types[0].mutex_count == 2
    assert merged.types[1].types[0].mutex_count == 1
    assert first.type.mutex_count == 1  # Original IR remains immutable.


@pytest.mark.parametrize(("outer_index", "inner_index"), [(1, 0), (2, None), (3, 1)])
def test_ssa_tuple_projection_preserves_mutex_slot_offsets(outer_index, inner_index):
    manager, left, right, flag = _ssa_fixture()
    _collect_auto(manager, [64], size=8, group_name="before")
    _collect_auto(manager, [96], size=8, group_name="after")
    before = manager.groups["before"][0]
    after = manager.groups["after"][0]
    zero = ir.ConstInt(0, ir.DataType.INT64, _SPAN)
    value = ir.MakeTuple(
        [zero, ir.MakeTuple([before], _SPAN), right, ir.MakeTuple([zero, after], _SPAN)], _SPAN,
    )
    holder = ir.Var("holder", value.type, _SPAN)
    choice = ir.GetItemExpr(holder, ir.ConstInt(outer_index, ir.DataType.INT64, _SPAN), _SPAN)
    if inner_index is not None:
        choice = ir.GetItemExpr(choice, ir.ConstInt(inner_index, ir.DataType.INT64, _SPAN), _SPAN)
    selected = ir.Var("selected", choice.type, _SPAN)
    holder_stmt, branch = _resolve_ssa(
        manager, [ir.AssignStmt([holder], value, _SPAN), _if_values(flag, left, choice, selected)], [flag],
    )

    types = holder_stmt.targets[0].type.types
    counts = [types[1].types[0].mutex_count, types[2].mutex_count, types[3].types[1].mutex_count]
    assert counts == [2 if index == outer_index else 1 for index in range(1, 4)]
    assert branch.return_vars[0].type.mutex_count == 2
    assert branch.else_body.stmts[-1].value[0].type.mutex_count == 2


def test_ssa_tuple_unpacking_preserves_independent_mutex_counts():
    manager, left, right, flag = _ssa_fixture()
    _collect_auto(manager, [64], size=8, group_name="other")
    other = manager.groups["other"][0]
    zero = ir.ConstInt(0, ir.DataType.INT64, _SPAN)
    pair = ir.MakeTuple([right, other], _SPAN)
    value = ir.MakeTuple([zero, pair], _SPAN)
    holder = ir.Var("tuple_tmp", value.type, _SPAN)
    scalar = ir.Var("scalar", zero.type, _SPAN)
    tiles = ir.Var("tiles", pair.type, _SPAN)
    choice = ir.GetItemExpr(tiles, zero, _SPAN)
    selected = ir.Var("selected", choice.type, _SPAN)
    one = ir.ConstInt(1, ir.DataType.INT64, _SPAN)
    # Match parser unpacking: bind the tuple, then assign each GetItem separately.
    _, scalar_assign, tiles_assign, branch = _resolve_ssa(
        manager, [ir.AssignStmt([holder], value, _SPAN),
                  ir.AssignStmt([scalar], ir.GetItemExpr(holder, zero, _SPAN), _SPAN),
                  ir.AssignStmt([tiles], ir.GetItemExpr(holder, one, _SPAN), _SPAN),
                  _if_values(flag, left, choice, selected)], [flag],
    )

    assert isinstance(scalar_assign.targets[0].type, ir.ScalarType)
    assert [element.mutex_count for element in tiles_assign.targets[0].type.types] == [2, 1]
    assert [element.mutex_count for element in tiles_assign.value.type.types] == [2, 1]
    assert branch.return_vars[0].type.mutex_count == 2


def test_ssa_dynamic_tuple_selection_joins_all_possible_tiles():
    manager, left, right, flag = _ssa_fixture()
    _collect_auto(manager, [64], size=8, group_name="other")
    other = manager.groups["other"][0]
    elements = ir.MakeTuple([right, other], _SPAN)
    aggregate = ir.Var("choices", elements.type, _SPAN)
    index = ir.Var("choice_index", ir.ScalarType(ir.DataType.INT64), _SPAN)
    choice = ir.GetItemExpr(aggregate, index, _SPAN)
    selected = ir.Var("selected", choice.type, _SPAN)
    _, branch = _resolve_ssa(
        manager, [ir.AssignStmt([aggregate], elements, _SPAN), _if_values(flag, left, choice, selected)], [flag, index],
    )

    assert branch.return_vars[0].type.mutex_count == 2
    dynamic_choice = branch.else_body.stmts[-1].value[0]
    assert [element.mutex_count for element in dynamic_choice.value.type.types] == [2, 2]


def test_ssa_merge_reports_conflicting_fixed_counts_before_rewrite():
    manager, left, right, flag = _ssa_fixture()
    _collect_manual(manager, [32], [[7]], size=8, group_name="right_pin")
    selected = ir.Var("selected", left.type, _SPAN)

    with pytest.raises(ValueError, match="TileGroup or SSA merge") as exc_info:
        _resolve_ssa(manager, [_if_values(flag, left, right, selected)], [flag])
    assert "requires at least 2 IDs but allows at most 1" in str(exc_info.value)


def test_ssa_named_tuple_projection_and_subview_preserve_count_and_metadata(check_ssa_golden):
    manager, left, right, flag = _ssa_fixture()
    zero = ir.ConstInt(0, ir.DataType.INT64, _SPAN)
    one = ir.ConstInt(1, ir.DataType.INT64, _SPAN)
    holder_value = ir.MakeTuple([zero, right], _SPAN)
    manager.debug_info.register_tuple_type_info(
        holder_value.type, ir.TupleTypeInfo(ir.TupleTypeKind.NAMED_TUPLE, fields=["tag", "tile"]),
    )
    holder = ir.Var("holder", holder_value.type, _SPAN)
    projection = ir.GetItemExpr(holder, one, _SPAN)
    subview_call = ir.create_op_call("block.subview", [projection, zero, ir.MakeTuple([], _SPAN)], {}, _SPAN)
    view = ir.Var("view", subview_call.type, _SPAN)
    selected = ir.Var("selected", left.type, _SPAN)
    tuple_stmt, view_stmt, branch = _resolve_ssa(
        manager, [ir.AssignStmt([holder], holder_value, _SPAN), ir.AssignStmt([view], subview_call, _SPAN),
                  _if_values(flag, left, view, selected)], [flag],
        check_golden=check_ssa_golden,
    )

    assert tuple_stmt.value.type.types[1].mutex_count == 2
    assert view_stmt.value.type.mutex_count == 2
    assert branch.return_vars[0].type.mutex_count == 2
    info = manager.debug_info.get_tuple_type_info(tuple_stmt.value.type)
    assert info.kind == ir.TupleTypeKind.NAMED_TUPLE and list(info.fields) == ["tag", "tile"]
