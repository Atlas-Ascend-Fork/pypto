# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto.pypto_impl import ir

_SPAN = ir.Span("planner_test.py", 1, 1)


def _declare_tile(declarations, addr, size, memory_space, ids, span):
    # Use byte ranges directly so the planner tests can exercise partial overlaps.
    index = len(declarations)
    one = ir.ConstInt(1, ir.DataType.INT64, span)
    call = ir.create_op_call(
        "block.make_mutex_tile",
        [ir.MakeTuple([one, one], span), ir.MakeTuple([], span), ir.MakeTuple(ids, span)],
        {"dtype": ir.DataType.INT8, "target_memory": memory_space,
         "memref_addr": addr, "memref_size": size, "memref_id": index},
        span,
    )
    tile = ir.Var(f"tile_{index}", call.type, span)
    declarations.append(ir.AssignStmt(tile, call, span))
    return index


def _declare_auto(declarations, addrs, *, size=1, memory_space=ir.MemorySpace.Vec, span=_SPAN):
    group = []
    for addr in addrs:
        placeholder = ir.Var(f"__pypto_auto_mutex_id_{len(declarations)}", ir.ScalarType(ir.DataType.INT64), span)
        group.append(_declare_tile(declarations, addr, size, memory_space, [placeholder], span))
    return group


def _declare_manual(declarations, addrs, ids, *, size=1, memory_space=ir.MemorySpace.Vec):
    group = []
    for addr, row in zip(addrs, ids, strict=True):
        values = [ir.ConstInt(value, ir.DataType.INT64, _SPAN) for value in row]
        group.append(_declare_tile(declarations, addr, size, memory_space, values, _SPAN))
    return group


def _resolve_groups(declarations, groups):
    func = ir.Function("main", [], [], ir.SeqStmts(declarations, _SPAN), _SPAN)
    program = ir.Program([func], "mutex_plan", _SPAN, ir.IRDebugInfo())
    body = ir.Pass.allocate_mutex_ids()(program)["main"].body
    statements = body.stmts if isinstance(body, ir.SeqStmts) else [body]
    rows = [tuple(element.value for element in stmt.value.args[2].elements) for stmt in statements]
    return tuple(tuple(rows[index] for index in group) for group in groups)


@pytest.mark.parametrize(
    ("count", "expected_min", "expected_max"),
    [(1, 0, 1), (32, 1, 1), (33, 1, 2), (64, 2, 2)],
)
def test_independent_auto_tiles_are_balanced(count, expected_min, expected_max):
    declarations = []
    group = _declare_auto(declarations, list(range(count)))

    assigned = [ids[0] for ids in _resolve_groups(declarations, [group])[0]]
    usage_counts = [assigned.count(mutex_id) for mutex_id in range(32)]
    assert min(usage_counts) == expected_min
    assert max(usage_counts) == expected_max
    if count <= 32:
        assert len(set(assigned)) == count


def test_manual_multi_ids_are_flattened_before_auto_allocation():
    declarations = []
    _declare_manual(declarations, [0, 10, 20], [[0, 3], [1, 3], [2, 4]])
    auto = _declare_auto(declarations, [100])

    assert _resolve_groups(declarations, [auto]) == (((5,),),)


def test_transitive_alias_component_shares_one_id():
    declarations = []
    first = _declare_auto(declarations, [0], size=100)
    second = _declare_auto(declarations, [0], size=220)
    third = _declare_auto(declarations, [150], size=70)

    assert _resolve_groups(declarations, [first, second, third]) == (((0,),), ((0,),), ((0,),))


@pytest.mark.parametrize(
    ("left_addr", "left_size", "right_addr", "right_size"),
    [(0, 32, 0, 32), (0, 64, 16, 16), (16, 16, 0, 64), (0, 32, 0, 64), (32, 32, 0, 64)],
)
def test_exact_and_contained_aliases_share_one_id(left_addr, left_size, right_addr, right_size):
    declarations = []
    left = _declare_auto(declarations, [left_addr], size=left_size)
    right = _declare_auto(declarations, [right_addr], size=right_size)

    assert _resolve_groups(declarations, [left, right]) == (((0,),), ((0,),))


@pytest.mark.parametrize(
    ("left_addr", "left_size", "right_addr", "right_size"),
    [(0, 32, 16, 32), (16, 32, 0, 32)],
)
def test_crossing_auto_aliases_are_rejected(left_addr, left_size, right_addr, right_size):
    declarations = []
    left = _declare_auto(declarations, [left_addr], size=left_size)
    right = _declare_auto(declarations, [right_addr], size=right_size)

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(declarations, [left, right])
    message = str(exc_info.value)
    assert "Enum: ExternalError::INVALID_OPERATION" in message
    assert "Vec Tile at line 1 uses [0x0, 0x20)" in message
    assert "Vec Tile at line 1 uses [0x10, 0x30)" in message
    overlap_start = max(left_addr, right_addr)
    overlap_end = min(left_addr + left_size, right_addr + right_size)
    assert f"overlap is [0x{overlap_start:x}, 0x{overlap_end:x})" in message


def test_crossing_tiles_are_distinguished_by_source_line():
    declarations = []
    left = _declare_auto(declarations, [0], size=32, span=ir.Span("planner_test.py", 10, 1))
    right = _declare_auto(
        declarations, [16], size=32, span=ir.Span("planner_test.py", 20, 1)
    )

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(declarations, [left, right])
    message = str(exc_info.value)
    assert "Vec Tile at line 10 uses [0x0, 0x20)" in message
    assert "Vec Tile at line 20 uses [0x10, 0x30)" in message


def test_crossing_manual_tiles_are_rejected_when_component_contains_auto():
    declarations = []
    auto = _declare_auto(declarations, [0], size=128)
    _declare_manual(declarations, [0], [[5]], size=80)
    _declare_manual(declarations, [48], [[5]], size=80)

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(declarations, [auto])
    assert "Enum: ExternalError::INVALID_OPERATION" in str(exc_info.value)


def test_adjacent_intervals_and_different_memory_spaces_do_not_alias():
    declarations = []
    left0 = _declare_auto(declarations, [0], size=16, memory_space=ir.MemorySpace.Left)
    left1 = _declare_auto(declarations, [16], size=16, memory_space=ir.MemorySpace.Left)
    right = _declare_auto(declarations, [0], size=16, memory_space=ir.MemorySpace.Right)

    assigned = _resolve_groups(declarations, [left0, left1, right])
    assert {group[0][0] for group in assigned} == {0, 1, 2}


def test_cube_and_vector_use_independent_id_pools():
    cube = []
    vector = []
    cube_group = _declare_auto(cube, [0], memory_space=ir.MemorySpace.Mat)
    vector_group = _declare_auto(vector, [128])

    assert _resolve_groups(cube, [cube_group]) == (((0,),),)
    assert _resolve_groups(vector, [vector_group]) == (((0,),),)


def test_one_manual_id_is_reused_by_all_aliased_auto_tiles():
    declarations = []
    _declare_manual(declarations, [0], [[9]], size=64)
    first = _declare_auto(declarations, [0], size=32)
    second = _declare_auto(declarations, [32], size=32)

    assert _resolve_groups(declarations, [first, second]) == (((9,),), ((9,),))


def test_overlapping_tile_with_multiple_manual_ids_is_rejected():
    declarations = []
    _declare_manual(declarations, [0], [[5, 9]], size=32)
    auto = _declare_auto(declarations, [0], size=32)

    with pytest.raises(ValueError, match="must have exactly one manual mutex ID") as exc_info:
        _resolve_groups(declarations, [auto])
    message = str(exc_info.value)
    assert "Enum: ExternalError::INVALID_ARGUMENT" in message
    assert "when address-overlapping Tiles include a manually configured Tile" in message
    assert "Vec Tile at line 1 uses [0x0, 0x20)" in message
    assert "mutex_ids=[5, 9]" in message


def test_restricted_candidate_job_is_allocated_before_unrestricted_job():
    declarations = []
    _declare_manual(declarations, [0], [[7]], size=16)
    restricted = _declare_auto(declarations, [0], size=16)
    unrestricted = _declare_auto(declarations, [100], size=16)

    assert _resolve_groups(declarations, [restricted, unrestricted]) == (((7,),), ((0,),))


def test_larger_auto_component_wins_address_order_tie_break():
    declarations = []
    singleton = _declare_auto(declarations, [100], size=16)
    component = _declare_auto(declarations, [0, 0, 0], size=16)

    assert _resolve_groups(declarations, [singleton, component]) == (((1,),), ((0,), (0,), (0,)))


def test_allocation_is_deterministic_across_collection_order():
    assignments = []
    for order in ((300, 100, 200), (100, 200, 300)):
        declarations = []
        groups = [_declare_auto(declarations, [addr]) for addr in order]
        resolved = _resolve_groups(declarations, groups)
        assignments.append({addr: ids[0][0] for addr, ids in zip(order, resolved)})

    assert assignments[0] == assignments[1]


def test_manual_only_alias_component_is_not_validated_without_auto():
    declarations = []
    manual = _declare_manual(declarations, [0, 8], [[5], [9]], size=32)

    assert _resolve_groups(declarations, [manual]) == (((5,), (9,)),)


def test_manual_only_alias_component_is_unchanged_when_auto_exists_elsewhere():
    declarations = []
    _declare_manual(declarations, [0], [[5]], size=32)
    _declare_manual(declarations, [8], [[9]], size=32)
    auto = _declare_auto(declarations, [100], size=16)

    assert _resolve_groups(declarations, [auto]) == (((0,),),)


def test_transitive_manual_id_mismatch_rejects_aliased_auto():
    declarations = []
    _declare_manual(declarations, [0], [[5]], size=32)
    _declare_manual(declarations, [32], [[9]], size=32)
    auto = _declare_auto(declarations, [0], size=64)

    with pytest.raises(ValueError, match="must use the same mutex ID") as exc_info:
        _resolve_groups(declarations, [auto])
    message = str(exc_info.value)
    assert "Enum: ExternalError::INVALID_OPERATION" in message
    assert "Vec Tile at line 1 uses [0x0, 0x20) with mutex_id=5" in message
    assert "Vec Tile at line 1 uses [0x20, 0x40) with mutex_id=9" in message
    assert "address-overlap component is [0x0, 0x40)" in message
