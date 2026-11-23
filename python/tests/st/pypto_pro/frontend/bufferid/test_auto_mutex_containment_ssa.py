#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Containment workloads exercising SsaMutexCountAnalysis through the frontend.

Coverage by workload:
* if_trees: manual multi-ID/AUTO and manual single-ID/AUTO, aliases and yields.
* dynamic_if: manual multi-ID/AUTO merges with independent runtime group indices.
* dynamic_tuple: AUTO/AUTO, MakeTuple, unpacking, constant/dynamic GetItem,
  scalar and zero-count positions, and independent tuple leaves.
* for_tuple: init/backedge/result relations, nested tuple rotation, continue/break.
* while_exits / while_jumps: independent header/exit phis, None, condition exit.
* nested_loops: inner results feeding outer carries and restoration of jump targets.
* views: reinterpret, group reinterpret, subview, aliases and dynamic selection.
* named_state: named tuple carries with scalar fields before independent Tile leaves.

Tuple unpacking follows the frontend lowering: a temporary tuple binding
and one single-target GetItem assignment for each extracted value.
All runtime conditions, indices and loop limits are read from an input Tensor;
the parser cannot fold them from the host test parameters.
"""

import os

import pypto_pro.language as pl
import pytest
import torch

from pypto.pypto_impl import ir

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
ROWS = 64
COLS = 32
HALF = ROWS // 2
_EXPECTED_IF_COUNTS = {
    "ssa_if_trees_kernel": 2,
    "ssa_dynamic_if_kernel": 1,
    "ssa_dynamic_tuple_kernel": 1,
    "ssa_for_tuple_kernel": 2,
    "ssa_while_exits_kernel": 1,
    "ssa_while_jumps_kernel": 3,
    "ssa_nested_loops_kernel": 5,
    "ssa_views_kernel": 1,
    "ssa_named_state_kernel": 2,
}


def _vec_type(rows):
    return pl.TileType(shape=[rows, COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _statements(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _statements(child)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
        yield from _statements(stmt.body)
    elif isinstance(stmt, ir.IfStmt):
        yield from _statements(stmt.then_body)
        if stmt.else_body is not None:
            yield from _statements(stmt.else_body)


def _check_plan(kernel, expected):
    """Check every declaration by (address, rows), independently of SSA names.

    Tuple values specify exact manual/forced ID sets; integers specify counts.
    Checking all declarations catches accidental joins with an unrelated leaf.
    """
    program, matched = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert matched
    statements = list(_statements(program[kernel.__name__].body))
    branches = [stmt for stmt in statements if isinstance(stmt, ir.IfStmt)]
    assert len(branches) == _EXPECTED_IF_COUNTS[kernel.__name__]
    assert all(not isinstance(branch.condition, (ir.ConstBool, ir.ConstInt)) for branch in branches)
    assert any(isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
               and stmt.value.name == "block.getval" for stmt in statements)
    actual = {}
    for stmt in statements:
        if not isinstance(stmt, ir.AssignStmt) or not isinstance(stmt.value, ir.Call):
            continue
        call = stmt.value
        if call.name != "block.make_mutex_tile":
            continue
        key = (call.kwargs["memref_addr"], call.type.shape[0].value)
        assert key not in actual, key
        ids = tuple(value.value for value in call.args[2].elements)
        assert len(ids) == len(set(ids)) == call.type.mutex_count
        assert all(0 <= value < 32 for value in ids)
        actual[key] = ids
    assert actual.keys() == expected.keys(), (actual, expected)
    for key, value in expected.items():
        if isinstance(value, tuple):
            assert actual[key] == value, (key, actual[key], value)
        else:
            assert len(actual[key]) == value, (key, actual[key], value)
    assert "__pypto_auto_mutex_id_" not in str(program)
    diagnostics = ir.IRVerifier.create_default().verify(program)
    assert not diagnostics, ir.IRVerifier.generate_report(diagnostics)
    return program


def _run_vector_case(kernel, input_blocks, output_blocks, golden, plan, *args):
    _require_a5()
    program = _check_plan(kernel, plan)
    torch.manual_seed(73)
    # Distinct block ranges make selecting the wrong descriptor observable.
    source_cpu = torch.rand([input_blocks * ROWS, COLS], dtype=torch.float16) * 0.25 - 0.125
    for index in range(input_blocks):
        source_cpu[index * ROWS:(index + 1) * ROWS] += index * 0.25
    source = source_cpu.to(ST_DEVICE)
    output = torch.full([output_blocks * ROWS, COLS], float("nan"), device=ST_DEVICE, dtype=torch.float16)
    control_values = [0, 0, 0]
    control_values[:len(args)] = args
    control = torch.tensor(control_values, device=ST_DEVICE, dtype=torch.int64)
    kernel(source, output, control)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), golden(source_cpu.float()), rtol=3e-3, atol=3e-3)
    return program


# Manual two-ID root -> AUTO children, with independent two-ID and one-ID phis.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_if_trees_kernel(
    source: pl.Tensor[[4 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[4 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids=[[8, 9]])
    children = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids="auto", depth=2)
    peer = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids="auto", depth=1)
    single = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x4000, mutex_ids=[4])
    single_auto = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x6000, mutex_ids="auto", depth=1)
    with pl.section_vector():
        condition = control[0] == 0
        pl.load(root[0], source, [0, 0])
        pl.load(peer[0], source, [ROWS, 0])
        pl.load(single[0], source, [2 * ROWS, 0])
        pl.load(single_auto[0], source, [3 * ROWS, 0])
        pl.add(children[0], children[0], children[0])
        pl.add(children[1], children[1], children[1])
        pl.add(children[1], children[1], children[1])
        first = root[0]
        second = peer[0]
        if condition:
            first = peer[0]
            second = root[0]
        alias = first
        pl.add(alias, alias, alias)
        pl.add(second, second, second)
        pl.add(second, second, second)
        one = single_auto[0]
        if condition:
            one = single[0]
        pl.add(one, one, one)
        pl.store(output, root[0], [0, 0])
        pl.store(output, peer[0], [ROWS, 0])
        pl.store(output, single[0], [2 * ROWS, 0])
        pl.store(output, single_auto[0], [3 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag", [True, False])
def test_ssa_manual_multi_and_single_id_if_trees(flag):
    def golden(source):
        result = source.clone()
        result[:HALF] *= 2
        result[HALF:ROWS] *= 4
        result[:ROWS] *= 2 if flag else 4
        result[ROWS:2 * ROWS] *= 4 if flag else 2
        block = 3 if flag else 2
        result[block * ROWS:(block + 1) * ROWS] *= 2
        return result

    plan = {(0, ROWS): (8, 9), (0, HALF): 1, (0x0800, HALF): 1,
            (0x2000, ROWS): 2, (0x4000, ROWS): (4,), (0x6000, ROWS): 1}
    _run_vector_case(ssa_if_trees_kernel, 4, 4, golden, plan, flag)


# Tensor-loaded indices select both Tile inputs to the if merge. Every AUTO
# member must grow to two IDs, while the contained children retain one ID each.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_dynamic_if_kernel(
    source: pl.Tensor[[4 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[5 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    manual = pl.make_tile_group(
        type=_vec_type(ROWS), addrs=[0, 0x2000], mutex_ids=[[8, 9], [10, 11]]
    )
    children = pl.make_tile_group(
        type=_vec_type(HALF), addrs=[0, 0x0800, 0x2000, 0x2800], mutex_ids="auto", depth=4
    )
    automatic = pl.make_tile_group(
        type=_vec_type(ROWS), addrs=[0x4000, 0x6000], mutex_ids="auto", depth=2
    )
    with pl.section_vector():
        condition = control[0] == 0
        for index in pl.range(0, 2):
            pl.load(manual[index], source, [index * ROWS, 0])
            pl.load(automatic[index], source, [(index + 2) * ROWS, 0])
        for child in pl.range(0, 4):
            pl.add(children[child], children[child], children[child])
        manual_slot = manual[control[1]]
        auto_slot = automatic[control[2]]
        selected = manual_slot
        if condition:
            selected = auto_slot
        pl.add(selected, selected, selected)
        for index in pl.range(0, 2):
            pl.store(output, manual[index], [index * ROWS, 0])
            pl.store(output, automatic[index], [(index + 2) * ROWS, 0])
        pl.store(output, selected, [4 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("manual_index", [0, 1])
@pytest.mark.parametrize("auto_index", [0, 1])
def test_ssa_manual_multi_auto_dynamic_if(flag, manual_index, auto_index):
    def golden(source):
        result = source.clone()
        result[:2 * ROWS] *= 2
        block = manual_index if flag else auto_index + 2
        result[block * ROWS:(block + 1) * ROWS] *= 2
        selected = result[block * ROWS:(block + 1) * ROWS]
        return torch.cat([result, selected])

    plan = {(0, ROWS): (8, 9), (0x2000, ROWS): (10, 11),
            (0, HALF): 1, (0x0800, HALF): 1, (0x2000, HALF): 1, (0x2800, HALF): 1,
            (0x4000, ROWS): 2, (0x6000, ROWS): 2}
    program = _run_vector_case(ssa_dynamic_if_kernel, 4, 5, golden, plan, flag, manual_index, auto_index)
    statements = list(_statements(program[ssa_dynamic_if_kernel.__name__].body))
    definitions = {
        stmt.targets[0]: stmt.value
        for stmt in statements if isinstance(stmt, ir.AssignStmt) and len(stmt.targets) == 1
    }
    branch, = [stmt for stmt in statements if isinstance(stmt, ir.IfStmt)]
    assert len(branch.return_vars) == 1
    assert branch.return_vars[0].type.mutex_count == 2
    for body in (branch.then_body, branch.else_body):
        terminator = body.stmts[-1]
        assert isinstance(terminator, ir.YieldStmt)
        value = terminator.value[0]
        while isinstance(value, ir.Var) and value in definitions:
            value = definitions[value]
        assert isinstance(value, ir.GetItemExpr)
        assert isinstance(value.slice, ir.Var)
        assert value.type.mutex_count == 2
        assert [tile.mutex_count for tile in value.value.type.types] == [2, 2]


# Manual two-ID leaves force a four-ID AUTO root. The ordinary tuple's choices
# belong to different groups, so dynamic GetItem must connect every candidate.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_dynamic_tuple_kernel(
    source: pl.Tensor[[5 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[5 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids="auto", depth=1)
    leaves = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids=[[8, 9], [10, 11]])
    choice_a = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids="auto", depth=1)
    choice_b = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x4000, mutex_ids="auto", depth=1)
    passive = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x6000, mutex_ids="auto", depth=1)
    unbound = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x8000, mutex_ids=None, depth=1)
    with pl.section_vector():
        condition = control[0] == 0
        slot = control[1]
        choices = (choice_a[0], choice_b[0])
        pl.load(root[0], source, [0, 0])
        pl.load(choices[0], source, [ROWS, 0])
        pl.load(choices[1], source, [2 * ROWS, 0])
        pl.load(passive[0], source, [3 * ROWS, 0])
        pl.load(unbound[0], source, [4 * ROWS, 0])
        for index in pl.range(0, 2):
            pl.add(leaves[index], leaves[index], leaves[index])
        state = (slot, root[0], (passive[0], unbound[0]))
        if condition:
            state = (slot, choices[slot], (passive[0], unbound[0]))
        tag, first, tail = state
        second, third = tail
        alias = first
        pl.add(alias, alias, alias)
        pl.add(second, second, second)
        pl.add(third, third, third)
        pl.store(output, root[0], [0, 0])
        pl.store(output, choices[0], [ROWS, 0])
        pl.store(output, choices[1], [2 * ROWS, 0])
        pl.store(output, passive[0], [3 * ROWS, 0])
        pl.store(output, unbound[0], [4 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("slot", [0, 1])
def test_ssa_auto_auto_dynamic_tuple_positions(flag, slot):
    def golden(source):
        result = source.clone()
        result[:ROWS] *= 2
        block = 0 if flag else slot + 1
        result[block * ROWS:(block + 1) * ROWS] *= 2
        result[3 * ROWS:] *= 2
        return result

    plan = {(0, ROWS): (8, 9, 10, 11), (0, HALF): (8, 9), (0x0800, HALF): (10, 11),
            (0x2000, ROWS): 4, (0x4000, ROWS): 4, (0x6000, ROWS): 1, (0x8000, ROWS): ()}
    _run_vector_case(ssa_dynamic_tuple_kernel, 5, 5, golden, plan, flag, slot)


# For break/continue share the header count groups, including swapped tuple leaves.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_for_tuple_kernel(
    source: pl.Tensor[[3 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[3 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids=[[16, 17, 18]])
    children = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids="auto", depth=2)
    peer = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids="auto", depth=1)
    passive = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x4000, mutex_ids="auto", depth=1)
    with pl.section_vector():
        steps = control[0]
        pl.load(root[0], source, [0, 0])
        pl.load(peer[0], source, [ROWS, 0])
        pl.load(passive[0], source, [2 * ROWS, 0])
        for child in pl.range(0, 2):
            pl.add(children[child], children[child], children[child])
        state = (0, root[0], (peer[0], passive[0]))
        for index in pl.range(0, steps):
            first = state[1]
            second = state[2][0]
            stable = state[2][1]
            if index == 3:
                break
            if index % 2 == 0:
                pl.add(second, second, second)
                state = (index, second, (first, stable))
                continue
            pl.add(first, first, first)
            state = (index, first, (second, stable))
        pl.store(output, state[1], [0, 0])
        pl.store(output, state[2][0], [ROWS, 0])
        pl.store(output, state[2][1], [2 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("steps", [0, 1, 2, 3, 5])
def test_ssa_manual_three_id_for_tuple_break_continue(steps):
    def golden(source):
        blocks = list(source.chunk(3))
        blocks[0] = blocks[0] * 2
        first, second = 0, 1
        for index in range(steps):
            if index == 3:
                break
            if index % 2 == 0:
                blocks[second] = blocks[second] * 2
                first, second = second, first
            else:
                blocks[first] = blocks[first] * 2
        return torch.cat([blocks[first], blocks[second], blocks[2]])

    plan = {(0, ROWS): (16, 17, 18), (0, HALF): 1, (0x0800, HALF): 1,
            (0x2000, ROWS): 3, (0x4000, ROWS): 1}
    _run_vector_case(ssa_for_tuple_kernel, 3, 3, golden, plan, steps)


# A while header with two IDs exits with a one-ID Tile in the same position.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_while_exits_kernel(
    source: pl.Tensor[[2 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[2 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids="auto", depth=1)
    leaves = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids=[5, 6])
    peer = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids="auto", depth=1)
    with pl.section_vector():
        steps = control[0]
        pl.load(root[0], source, [0, 0])
        pl.load(peer[0], source, [ROWS, 0])
        for child in pl.range(0, 2):
            pl.add(leaves[child], leaves[child], leaves[child])
        current = root[0]
        extra = None
        index = 0
        while True:
            if index >= steps:
                current = peer[0]
                extra = root[0]
                break
            pl.add(current, current, current)
            index = index + 1
            continue
        pl.store(output, current, [0, 0])
        pl.store(output, extra, [ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_ssa_while_header_exit_counts_and_none_are_independent(steps):
    def golden(source):
        return torch.cat([source[ROWS:], source[:ROWS] * (2 ** (steps + 1))])

    plan = {(0, ROWS): (5, 6), (0, HALF): (5,), (0x0800, HALF): (6,), (0x2000, ROWS): 1}
    _run_vector_case(ssa_while_exits_kernel, 2, 2, golden, plan, steps)


# Natural-condition while: continue, explicit break and the false-condition exit.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_while_jumps_kernel(
    source: pl.Tensor[[2 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[3 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids=[[12, 13]])
    children = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids="auto", depth=2)
    peer = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids="auto", depth=1)
    with pl.section_vector():
        limit = control[0]
        pl.load(root[0], source, [0, 0])
        pl.load(peer[0], source, [ROWS, 0])
        for child in pl.range(0, 2):
            pl.add(children[child], children[child], children[child])
        selected = root[0]
        index = 0
        while index < limit:
            if index % 2 == 0:
                selected = peer[0]
                pl.add(selected, selected, selected)
                index = index + 1
                continue
            selected = root[0]
            pl.add(selected, selected, selected)
            if index == 3:
                break
            index = index + 1
        pl.store(output, root[0], [0, 0])
        pl.store(output, peer[0], [ROWS, 0])
        pl.store(output, selected, [2 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 6])
def test_ssa_manual_multi_auto_while_jumps(limit):
    def golden(source):
        blocks = [source[:ROWS] * 2, source[ROWS:].clone()]
        selected = 0
        for index in range(min(limit, 4)):
            selected = 1 if index % 2 == 0 else 0
            blocks[selected] *= 2
        return torch.cat([blocks[0], blocks[1], blocks[selected]])

    plan = {(0, ROWS): (12, 13), (0, HALF): 1, (0x0800, HALF): 1, (0x2000, ROWS): 2}
    _run_vector_case(ssa_while_jumps_kernel, 2, 3, golden, plan, limit)


# Inner three-ID and one-ID carries feed separate outer slots. Wrong jump targets
# would either make the single-ID group grow or conflict with its manual ID.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_nested_loops_kernel(
    source: pl.Tensor[[4 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[4 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids=[[20, 21, 22]])
    children = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids="auto", depth=2)
    peer = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids="auto", depth=1)
    single = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x4000, mutex_ids=[7])
    single_auto = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x6000, mutex_ids="auto", depth=1)
    with pl.section_vector():
        outer_steps = control[0]
        inner_steps = control[1]
        pl.load(root[0], source, [0, 0])
        pl.load(peer[0], source, [ROWS, 0])
        pl.load(single[0], source, [2 * ROWS, 0])
        pl.load(single_auto[0], source, [3 * ROWS, 0])
        for child in pl.range(0, 2):
            pl.add(children[child], children[child], children[child])
        outer_tile = root[0]
        inner_tile = single[0]
        outer = 0
        while outer < outer_steps:
            for inner in pl.range(0, inner_steps):
                if inner % 2 == 0:
                    outer_tile = peer[0]
                    inner_tile = single_auto[0]
                    pl.add(inner_tile, inner_tile, inner_tile)
                    continue
                outer_tile = root[0]
                inner_tile = single[0]
                pl.add(inner_tile, inner_tile, inner_tile)
                if inner == 3:
                    break
            pl.add(outer_tile, outer_tile, outer_tile)
            outer = outer + 1
            if outer == 2:
                continue
            if outer == 3:
                break
        pl.store(output, root[0], [0, 0])
        pl.store(output, peer[0], [ROWS, 0])
        pl.store(output, single[0], [2 * ROWS, 0])
        pl.store(output, single_auto[0], [3 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("outer_steps, inner_steps", [(0, 2), (2, 0), (2, 3), (5, 6)])
def test_ssa_nested_for_while_restore_targets_and_connect_results(outer_steps, inner_steps):
    def golden(source):
        blocks = [block.clone() for block in source.chunk(4)]
        blocks[0] *= 2
        outer_tile = 0
        for _ in range(min(outer_steps, 3)):
            for inner in range(min(inner_steps, 4)):
                outer_tile = 1 if inner % 2 == 0 else 0
                inner_tile = 3 if inner % 2 == 0 else 2
                blocks[inner_tile] *= 2
            blocks[outer_tile] *= 2
        return torch.cat(blocks)

    plan = {(0, ROWS): (20, 21, 22), (0, HALF): 1, (0x0800, HALF): 1,
            (0x2000, ROWS): 3, (0x4000, ROWS): (7,), (0x6000, ROWS): 1}
    _run_vector_case(ssa_nested_loops_kernel, 4, 4, golden, plan, outer_steps, inner_steps)


@pl.jit(arch="3510", auto_mutex=True)
def ssa_views_kernel(
    source: pl.Tensor[[3 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[3 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids=[[20, 21]])
    children = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids="auto", depth=2)
    choices = pl.make_tile_group(type=_vec_type(ROWS), addrs=[0x2000, 0x4000], mutex_ids="auto", depth=2)
    with pl.section_vector():
        condition = control[0] == 0
        slot = control[1]
        pl.load(root[0], source, [0, 0])
        pl.load(choices[0], source, [ROWS, 0])
        pl.load(choices[1], source, [2 * ROWS, 0])
        for child in pl.range(0, 2):
            pl.add(children[child], children[child], children[child])
        manual_view = pl.reinterpret(root[0], shape=[32, 64])
        auto_views = pl.reinterpret(choices, shape=[32, 64])
        selected = manual_view[:, :]
        if condition:
            selected = auto_views[slot][:, :]
        alias = selected
        pl.add(alias, alias, alias)
        pl.store(output, root[0], [0, 0])
        pl.store(output, choices[0], [ROWS, 0])
        pl.store(output, choices[1], [2 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("slot", [0, 1])
def test_ssa_reinterpret_subview_and_dynamic_group(flag, slot):
    def golden(source):
        result = source.clone()
        result[:ROWS] *= 2
        block = 0 if flag else slot + 1
        result[block * ROWS:(block + 1) * ROWS] *= 2
        return result

    plan = {(0, ROWS): (20, 21), (0, HALF): 1, (0x0800, HALF): 1,
            (0x2000, ROWS): 2, (0x4000, ROWS): 2}
    program = _run_vector_case(ssa_views_kernel, 3, 3, golden, plan, flag, slot)
    for op in ("block.reinterpret", "block.subview"):
        assert op in str(program)


# A scalar prefix must not shift the following Tile positions in a named tuple
# or make its one-ID leaves join the two-ID leaves.
@pl.jit(arch="3510", auto_mutex=True)
def ssa_named_state_kernel(
    source: pl.Tensor[[4 * ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[4 * ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids="auto", depth=1)
    children = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids=[24, 25])
    leaves = pl.make_tile_group(
        type=_vec_type(16), addrs=[0, 0x0400, 0x0800, 0x0C00], mutex_ids="auto", depth=4
    )
    peer = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids="auto", depth=1)
    single = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x4000, mutex_ids=[6])
    single_auto = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x6000, mutex_ids="auto", depth=1)
    with pl.section_vector():
        steps = control[0]
        pl.load(root[0], source, [0, 0])
        pl.load(peer[0], source, [ROWS, 0])
        pl.load(single[0], source, [2 * ROWS, 0])
        pl.load(single_auto[0], source, [3 * ROWS, 0])
        for child in pl.range(0, 2):
            pl.add(children[child], children[child], children[child])
        for leaf in pl.range(0, 4):
            pl.add(leaves[leaf], leaves[leaf], leaves[leaf])
        cursor = 0
        if control[1] != 0:
            cursor = 1
        state = pl.make_tuple(index=cursor, multi=root[0], rest=(peer[0], single[0], single_auto[0]))
        for index in pl.range(0, steps):
            selected = state.multi
            one = state.rest[2]
            if state.index % 2 != 0:
                selected = state.rest[0]
                one = state.rest[1]
            pl.add(selected, selected, selected)
            pl.add(one, one, one)
            state = pl.make_tuple(
                index=state.index + 1, multi=state.rest[0], rest=(state.multi, state.rest[1], state.rest[2])
            )
        pl.store(output, root[0], [0, 0])
        pl.store(output, peer[0], [ROWS, 0])
        pl.store(output, single[0], [2 * ROWS, 0])
        pl.store(output, single_auto[0], [3 * ROWS, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("steps", [0, 1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_ssa_named_tuple_keeps_count_positions(steps, reverse):
    def golden(source):
        blocks = [block.clone() for block in source.chunk(4)]
        blocks[0] *= 4
        first, second = 0, 1
        for index in range(steps):
            even = (index + int(reverse)) % 2 == 0
            blocks[first if even else second] *= 2
            blocks[3 if even else 2] *= 2
            first, second = second, first
        return torch.cat(blocks)

    plan = {(0, ROWS): (24, 25), (0, HALF): (24,), (0x0800, HALF): (25,),
            (0, 16): (24,), (0x0400, 16): (24,), (0x0800, 16): (25,), (0x0C00, 16): (25,),
            (0x2000, ROWS): 2, (0x4000, ROWS): (6,), (0x6000, ROWS): 1}
    _run_vector_case(ssa_named_state_kernel, 4, 4, golden, plan, steps, reverse)
