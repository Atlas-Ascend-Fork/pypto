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


class AutoMutexIdManager:
    """Keep Tile Vars alive like the production IR builder does."""

    def __init__(self):
        self._manager = ir._AutoMutexIdManager()
        self._tiles = []

    def collect_group(self, tiles, mutex_ids, group_name):
        self._tiles.extend(tiles)
        self._manager.collect_group(tiles, mutex_ids, group_name)

    def assign_mutex_ids(self, program):
        return self._manager.assign_mutex_ids(program)


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
    flattened = [
        tile_ids.elements[0]
        for group in groups
        for tile_ids in group.elements
    ]
    value = ir.MakeTuple(flattened, _SPAN)
    holder = ir.Var("resolved_mutex_ids", value.type, _SPAN)
    body = ir.SeqStmts([ir.AssignStmt(holder, value, _SPAN)], _SPAN)
    func = ir.Function("main", [], [], body, _SPAN)
    program = manager.assign_mutex_ids(ir.Program([func], "mutex_plan", _SPAN, ir.IRDebugInfo()))
    stmt = program["main"].body
    if isinstance(stmt, ir.SeqStmts):
        stmt = stmt.stmts[0]
    resolved = tuple(element.value for element in stmt.value.elements)

    result = []
    offset = 0
    for group in groups:
        result.append(tuple((resolved[offset + index],) for index in range(len(group.elements))))
        offset += len(group.elements)
    return tuple(result)


@pytest.mark.parametrize(
    ("count", "expected_min", "expected_max"),
    [(1, 0, 1), (32, 1, 1), (33, 1, 2), (64, 2, 2)],
)
def test_independent_auto_tiles_are_balanced(count, expected_min, expected_max):
    manager = AutoMutexIdManager()
    group = _collect_auto(manager, list(range(count)))

    assigned = [ids[0] for ids in _resolve_groups(manager, [group])[0]]
    usage_counts = [assigned.count(mutex_id) for mutex_id in range(32)]
    assert min(usage_counts) == expected_min
    assert max(usage_counts) == expected_max
    if count <= 32:
        assert len(set(assigned)) == count


def test_manual_multi_ids_are_flattened_before_auto_allocation():
    manager = AutoMutexIdManager()
    _collect_manual(manager, [0, 10, 20], [[0, 3], [1, 3], [2, 4]])
    auto = _collect_auto(manager, [100])

    assert _resolve_groups(manager, [auto]) == (((5,),),)


def test_transitive_alias_component_shares_one_id():
    manager = AutoMutexIdManager()
    first = _collect_auto(manager, [0], size=100)
    second = _collect_auto(manager, [0], size=220)
    third = _collect_auto(manager, [150], size=70)

    assert _resolve_groups(manager, [first, second, third]) == (((0,),), ((0,),), ((0,),))


@pytest.mark.parametrize(
    ("left_addr", "left_size", "right_addr", "right_size"),
    [(0, 32, 0, 32), (0, 64, 16, 16), (16, 16, 0, 64), (0, 32, 0, 64), (32, 32, 0, 64)],
)
def test_exact_and_contained_aliases_share_one_id(left_addr, left_size, right_addr, right_size):
    manager = AutoMutexIdManager()
    left = _collect_auto(manager, [left_addr], size=left_size)
    right = _collect_auto(manager, [right_addr], size=right_size)

    assert _resolve_groups(manager, [left, right]) == (((0,),), ((0,),))


@pytest.mark.parametrize(
    ("left_addr", "left_size", "right_addr", "right_size"),
    [(0, 32, 16, 32), (16, 32, 0, 32)],
)
def test_crossing_auto_aliases_are_rejected(left_addr, left_size, right_addr, right_size):
    manager = AutoMutexIdManager()
    left = _collect_auto(manager, [left_addr], size=left_size, group_name="left_group")
    right = _collect_auto(manager, [right_addr], size=right_size, group_name="right_group")

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(manager, [left, right])
    message = str(exc_info.value)
    assert "Enum: ExternalError::INVALID_OPERATION" in message
    assert "TileGroup 'left_group' at line 1" in message
    assert "TileGroup 'right_group' at line 1" in message
    overlap_start = max(left_addr, right_addr)
    overlap_end = min(left_addr + left_size, right_addr + right_size)
    assert f"overlap is [0x{overlap_start:x}, 0x{overlap_end:x})" in message


def test_crossing_same_name_groups_are_distinguished_by_source_line():
    manager = AutoMutexIdManager()
    left = _collect_auto(manager, [0], size=32, group_name="shared_group", span=ir.Span("planner_test.py", 10, 1))
    right = _collect_auto(
        manager, [16], size=32, group_name="shared_group", span=ir.Span("planner_test.py", 20, 1)
    )

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(manager, [left, right])
    message = str(exc_info.value)
    assert "TileGroup 'shared_group' at line 10" in message
    assert "TileGroup 'shared_group' at line 20" in message


def test_crossing_manual_tiles_are_rejected_when_component_contains_auto():
    manager = AutoMutexIdManager()
    auto = _collect_auto(manager, [0], size=128)
    _collect_manual(manager, [0], [[5]], size=80)
    _collect_manual(manager, [48], [[5]], size=80)

    with pytest.raises(ValueError, match="possible buffer address trampling") as exc_info:
        _resolve_groups(manager, [auto])
    assert "Enum: ExternalError::INVALID_OPERATION" in str(exc_info.value)


def test_adjacent_intervals_and_different_memory_spaces_do_not_alias():
    manager = AutoMutexIdManager()
    left0 = _collect_auto(manager, [0], size=16, memory_space=ir.MemorySpace.Left)
    left1 = _collect_auto(manager, [16], size=16, memory_space=ir.MemorySpace.Left)
    right = _collect_auto(manager, [0], size=16, memory_space=ir.MemorySpace.Right)

    assigned = _resolve_groups(manager, [left0, left1, right])
    assert {group[0][0] for group in assigned} == {0, 1, 2}


def test_cube_and_vector_use_independent_id_pools():
    cube = AutoMutexIdManager()
    vector = AutoMutexIdManager()
    cube_group = _collect_auto(cube, [0], memory_space=ir.MemorySpace.Mat)
    vector_group = _collect_auto(vector, [128])

    assert _resolve_groups(cube, [cube_group]) == (((0,),),)
    assert _resolve_groups(vector, [vector_group]) == (((0,),),)


def test_one_manual_id_is_reused_by_all_aliased_auto_tiles():
    manager = AutoMutexIdManager()
    _collect_manual(manager, [0], [[9]], size=64)
    first = _collect_auto(manager, [0], size=32)
    second = _collect_auto(manager, [32], size=32)

    assert _resolve_groups(manager, [first, second]) == (((9,),), ((9,),))


def test_overlapping_tile_with_multiple_manual_ids_is_rejected():
    manager = AutoMutexIdManager()
    _collect_manual(manager, [0], [[5, 9]], size=32, group_name="manual_group")
    auto = _collect_auto(manager, [0], size=32)

    with pytest.raises(ValueError, match="must have exactly one manual mutex ID") as exc_info:
        _resolve_groups(manager, [auto])
    message = str(exc_info.value)
    assert "Enum: ExternalError::INVALID_ARGUMENT" in message
    assert "when address-overlapping Tiles include a manually configured Tile" in message
    assert "TileGroup 'manual_group' at line 1" in message
    assert "has an overlapping Tile with multiple mutex IDs" in message
    assert "mutex_ids=[5, 9]" in message


def test_restricted_candidate_job_is_allocated_before_unrestricted_job():
    manager = AutoMutexIdManager()
    _collect_manual(manager, [0], [[7]], size=16)
    restricted = _collect_auto(manager, [0], size=16)
    unrestricted = _collect_auto(manager, [100], size=16)

    assert _resolve_groups(manager, [restricted, unrestricted]) == (((7,),), ((0,),))


def test_larger_auto_component_wins_address_order_tie_break():
    manager = AutoMutexIdManager()
    singleton = _collect_auto(manager, [100], size=16)
    component = _collect_auto(manager, [0, 0, 0], size=16)

    assert _resolve_groups(manager, [singleton, component]) == (((1,),), ((0,), (0,), (0,)))


def test_allocation_is_deterministic_across_collection_order():
    assignments = []
    for order in ((300, 100, 200), (100, 200, 300)):
        manager = AutoMutexIdManager()
        groups = [_collect_auto(manager, [addr]) for addr in order]
        resolved = _resolve_groups(manager, groups)
        assignments.append({addr: ids[0][0] for addr, ids in zip(order, resolved)})

    assert assignments[0] == assignments[1]


def test_manual_only_alias_component_is_not_validated_without_auto():
    manager = AutoMutexIdManager()
    _collect_manual(manager, [0, 8], [[5], [9]], size=32)

    program = ir.Program([], "manual_only", _SPAN, ir.IRDebugInfo())
    manager.assign_mutex_ids(program)


def test_manual_only_alias_component_is_unchanged_when_auto_exists_elsewhere():
    manager = AutoMutexIdManager()
    _collect_manual(manager, [0], [[5]], size=32)
    _collect_manual(manager, [8], [[9]], size=32)
    auto = _collect_auto(manager, [100], size=16)

    assert _resolve_groups(manager, [auto]) == (((0,),),)


def test_transitive_manual_id_mismatch_rejects_aliased_auto():
    manager = AutoMutexIdManager()
    _collect_manual(manager, [0], [[5]], size=32, group_name="manual_a")
    _collect_manual(manager, [32], [[9]], size=32, group_name="manual_b")
    auto = _collect_auto(manager, [0], size=64)

    with pytest.raises(ValueError, match="must use the same mutex ID") as exc_info:
        _resolve_groups(manager, [auto])
    message = str(exc_info.value)
    assert "Enum: ExternalError::INVALID_OPERATION" in message
    assert "TileGroup 'manual_a' at line 1 uses [0x0, 0x20) with mutex_id=5" in message
    assert "TileGroup 'manual_b' at line 1 uses [0x20, 0x40) with mutex_id=9" in message
    assert "address-overlap component is [0x0, 0x40)" in message
