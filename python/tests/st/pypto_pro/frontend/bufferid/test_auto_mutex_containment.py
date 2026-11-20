#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for AUTO mutex-ID allocation on contained Tile address ranges."""

import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

VEC_COLS = 32


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _vec_type(rows):
    return pl.TileType(shape=[rows, VEC_COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)


def _run_vector_case(kernel, rows, golden_fn, seed):
    _require_a5(ST_DEVICE)
    torch.manual_seed(seed)
    source = torch.rand([rows, VEC_COLS], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    output = torch.zeros_like(source)

    kernel(source, output)
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu().float(), golden_fn(source.cpu().float()), rtol=3e-3, atol=3e-3)


# =============================================================================
# Two levels: one parent and two adjacent children
# =============================================================================

TWO_LEVEL_ROWS = 64


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_two_level_vector_kernel(
    source: pl.Tensor[[TWO_LEVEL_ROWS, VEC_COLS], pl.DT_FP16],
    output: pl.Tensor[[TWO_LEVEL_ROWS, VEC_COLS], pl.DT_FP16],
):
    # FP16 [64, 32] occupies 0x1000 bytes; each child occupies one half.
    root = pl.make_tile_group(type=_vec_type(64), addrs=0x0000, mutex_ids="auto", depth=1)
    children = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x0000, 0x0800], mutex_ids="auto", depth=2
    )

    with pl.section_vector():
        pl.load(root[0], source, [0, 0])
        pl.add(children[0], children[0], children[0])
        pl.exp(children[1], children[1])
        pl.store(output, root[0], [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_two_level_vector_containment():
    def golden(source):
        result = source.clone()
        result[:32] *= 2.0
        result[32:] = torch.exp(result[32:])
        return result

    _run_vector_case(auto_mutex_two_level_vector_kernel, TWO_LEVEL_ROWS, golden, seed=31)


# =============================================================================
# Three levels, balanced: 1 root -> 2 children -> 4 leaves
# =============================================================================

BALANCED_ROWS = 64


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_balanced_three_level_vector_kernel(
    source: pl.Tensor[[BALANCED_ROWS, VEC_COLS], pl.DT_FP16],
    output: pl.Tensor[[BALANCED_ROWS, VEC_COLS], pl.DT_FP16],
):
    root = pl.make_tile_group(type=_vec_type(64), addrs=0x0000, mutex_ids="auto", depth=1)
    children = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x0000, 0x0800], mutex_ids="auto", depth=2
    )
    leaves = pl.make_tile_group(
        type=_vec_type(16),
        addrs=[0x0000, 0x0400, 0x0800, 0x0C00],
        mutex_ids="auto",
        depth=4,
    )

    with pl.section_vector():
        pl.load(root[0], source, [0, 0])
        for index in pl.range(0, 2):
            pl.add(children[index], children[index], children[index])
        for index in pl.range(0, 4):
            pl.add(leaves[index], leaves[index], leaves[index])
        pl.store(output, root[0], [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_balanced_three_level_vector_containment():
    _run_vector_case(
        auto_mutex_balanced_three_level_vector_kernel,
        BALANCED_ROWS,
        lambda source: source * 4.0,
        seed=32,
    )


# =============================================================================
# Three levels, unbalanced:
#   tree A: sibling children own three and two leaves;
#   tree B: sibling children own two and zero leaves.
# =============================================================================

UNBALANCED_A_ROWS = 80
UNBALANCED_B_ROWS = 64
UNBALANCED_ROWS = UNBALANCED_A_ROWS + UNBALANCED_B_ROWS


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_unbalanced_three_level_vector_kernel(
    source: pl.Tensor[[UNBALANCED_ROWS, VEC_COLS], pl.DT_FP16],
    output: pl.Tensor[[UNBALANCED_ROWS, VEC_COLS], pl.DT_FP16],
):
    # Tree A occupies [0x0000, 0x1400): two children contain 3 and 2 leaves.
    root_a = pl.make_tile_group(type=_vec_type(80), addrs=0x0000, mutex_ids="auto", depth=1)
    child_a_three = pl.make_tile_group(type=_vec_type(48), addrs=0x0000, mutex_ids="auto", depth=1)
    child_a_two = pl.make_tile_group(type=_vec_type(32), addrs=0x0C00, mutex_ids="auto", depth=1)
    leaves_a = pl.make_tile_group(
        type=_vec_type(16),
        addrs=[0x0000, 0x0400, 0x0800, 0x0C00, 0x1000],
        mutex_ids="auto",
        depth=5,
    )

    # Tree B occupies [0x2000, 0x3000): only the first child has leaves.
    root_b = pl.make_tile_group(type=_vec_type(64), addrs=0x2000, mutex_ids="auto", depth=1)
    children_b = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x2000, 0x2800], mutex_ids="auto", depth=2
    )
    leaves_b = pl.make_tile_group(
        type=_vec_type(16), addrs=[0x2000, 0x2400], mutex_ids="auto", depth=2
    )

    with pl.section_vector():
        pl.load(root_a[0], source, [0, 0])
        pl.add(child_a_three[0], child_a_three[0], child_a_three[0])
        pl.add(child_a_two[0], child_a_two[0], child_a_two[0])
        for index in pl.range(0, 5):
            pl.add(leaves_a[index], leaves_a[index], leaves_a[index])
        pl.store(output, root_a[0], [0, 0])

        pl.load(root_b[0], source, [UNBALANCED_A_ROWS, 0])
        for index in pl.range(0, 2):
            pl.add(children_b[index], children_b[index], children_b[index])
        for index in pl.range(0, 2):
            pl.add(leaves_b[index], leaves_b[index], leaves_b[index])
        pl.store(output, root_b[0], [UNBALANCED_A_ROWS, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_unbalanced_three_level_vector_containment():
    def golden(source):
        result = source.clone()
        result[:UNBALANCED_A_ROWS] *= 4.0
        result[UNBALANCED_A_ROWS:UNBALANCED_A_ROWS + 32] *= 4.0
        result[UNBALANCED_A_ROWS + 32:] *= 2.0
        return result

    _run_vector_case(auto_mutex_unbalanced_three_level_vector_kernel, UNBALANCED_ROWS, golden, seed=33)


# =============================================================================
# Four levels: root -> child -> grandchild -> leaf
# =============================================================================

FOUR_LEVEL_ROWS = 64


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_four_level_vector_kernel(
    source: pl.Tensor[[FOUR_LEVEL_ROWS, VEC_COLS], pl.DT_FP16],
    output: pl.Tensor[[FOUR_LEVEL_ROWS, VEC_COLS], pl.DT_FP16],
):
    root = pl.make_tile_group(type=_vec_type(64), addrs=0x0000, mutex_ids="auto", depth=1)
    child = pl.make_tile_group(type=_vec_type(48), addrs=0x0000, mutex_ids="auto", depth=1)
    grandchild = pl.make_tile_group(type=_vec_type(32), addrs=0x0000, mutex_ids="auto", depth=1)
    leaf = pl.make_tile_group(type=_vec_type(16), addrs=0x0000, mutex_ids="auto", depth=1)

    with pl.section_vector():
        pl.load(root[0], source, [0, 0])
        pl.add(child[0], child[0], child[0])
        pl.add(grandchild[0], grandchild[0], grandchild[0])
        pl.add(leaf[0], leaf[0], leaf[0])
        pl.store(output, root[0], [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_four_level_vector_containment():
    def golden(source):
        result = source.clone()
        result[:16] *= 8.0
        result[16:32] *= 4.0
        result[32:48] *= 2.0
        return result

    _run_vector_case(auto_mutex_four_level_vector_kernel, FOUR_LEVEL_ROWS, golden, seed=34)


# =============================================================================
# Manual constraints in two-level and three-level containment trees
# =============================================================================

MANUAL_TREE_ROWS = 64
MANUAL_TREE_COUNT = 7
MANUAL_ROWS = MANUAL_TREE_ROWS * MANUAL_TREE_COUNT


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_manual_containment_kernel(
    source: pl.Tensor[[MANUAL_ROWS, VEC_COLS], pl.DT_FP16],
    output: pl.Tensor[[MANUAL_ROWS, VEC_COLS], pl.DT_FP16],
):
    # Tree 0, two levels: the manual root restricts both AUTO children to ID 8.
    root_manual_two = pl.make_tile_group(type=_vec_type(64), addrs=0x0000, mutex_ids=[8])
    children_auto_two = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x0000, 0x0800], mutex_ids="auto", depth=2
    )

    # Tree 1, three levels: every AUTO descendant of the manual root must use ID 9.
    root_manual_three = pl.make_tile_group(type=_vec_type(64), addrs=0x2000, mutex_ids=[9])
    middle_auto_below_root = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x2000, 0x2800], mutex_ids="auto", depth=2
    )
    leaves_auto_below_root = pl.make_tile_group(
        type=_vec_type(16),
        addrs=[0x2000, 0x2400, 0x2800, 0x2C00],
        mutex_ids="auto",
        depth=4,
    )

    # Tree 2, three levels: both middle siblings are manual and each has two
    # IDs. Their AUTO leaves inherit one ID, while the AUTO root contains all four.
    root_auto_manual_siblings = pl.make_tile_group(
        type=_vec_type(64), addrs=0x4000, mutex_ids="auto", depth=1
    )
    middle_manual_siblings = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x4000, 0x4800], mutex_ids=[[10, 11], [12, 13]]
    )
    leaves_auto_manual_siblings = pl.make_tile_group(
        type=_vec_type(16),
        addrs=[0x4000, 0x4400, 0x4800, 0x4C00],
        mutex_ids="auto",
        depth=4,
    )

    # Tree 3, three levels: one middle sibling is manual and one is AUTO.
    root_auto_mixed_siblings = pl.make_tile_group(
        type=_vec_type(64), addrs=0x6000, mutex_ids="auto", depth=1
    )
    middle_manual = pl.make_tile_group(type=_vec_type(32), addrs=0x6000, mutex_ids=[14])
    middle_auto = pl.make_tile_group(
        type=_vec_type(32), addrs=0x6800, mutex_ids="auto", depth=1
    )
    leaves_auto_mixed_siblings = pl.make_tile_group(
        type=_vec_type(16),
        addrs=[0x6000, 0x6400, 0x6800, 0x6C00],
        mutex_ids="auto",
        depth=4,
    )

    # Tree 4, two levels: manual leaves may keep multiple IDs. The AUTO root
    # contains both fixed sets.
    root_auto_manual_leaves_two = pl.make_tile_group(
        type=_vec_type(64), addrs=0x8000, mutex_ids="auto", depth=1
    )
    leaves_manual_two = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x8000, 0x8800], mutex_ids=[[15, 16], [17, 18]]
    )

    # Tree 5, three levels: four bottom leaves each carry exactly one manual ID.
    root_auto_manual_leaves_three = pl.make_tile_group(
        type=_vec_type(64), addrs=0xA000, mutex_ids="auto", depth=1
    )
    middle_auto_manual_leaves = pl.make_tile_group(
        type=_vec_type(32), addrs=[0xA000, 0xA800], mutex_ids="auto", depth=2
    )
    leaves_manual_three = pl.make_tile_group(
        type=_vec_type(16),
        addrs=[0xA000, 0xA400, 0xA800, 0xAC00],
        mutex_ids=[17, 18, 19, 20],
    )

    # Tree 6, three unbalanced levels: the manual root fixes ID 21 for every
    # AUTO node. The first middle sibling has two leaves; the second has none.
    root_manual_unbalanced = pl.make_tile_group(type=_vec_type(64), addrs=0xC000, mutex_ids=[21])
    middle_auto_unbalanced = pl.make_tile_group(
        type=_vec_type(32), addrs=[0xC000, 0xC800], mutex_ids="auto", depth=2
    )
    leaves_auto_unbalanced = pl.make_tile_group(
        type=_vec_type(16), addrs=[0xC000, 0xC400], mutex_ids="auto", depth=2
    )

    with pl.section_vector():
        pl.load(root_manual_two[0], source, [0, 0])
        for index in pl.range(0, 2):
            pl.add(children_auto_two[index], children_auto_two[index], children_auto_two[index])
        pl.store(output, root_manual_two[0], [0, 0])

        pl.load(root_manual_three[0], source, [MANUAL_TREE_ROWS, 0])
        for index in pl.range(0, 2):
            pl.add(
                middle_auto_below_root[index],
                middle_auto_below_root[index],
                middle_auto_below_root[index],
            )
        for index in pl.range(0, 4):
            pl.add(
                leaves_auto_below_root[index],
                leaves_auto_below_root[index],
                leaves_auto_below_root[index],
            )
        pl.store(output, root_manual_three[0], [MANUAL_TREE_ROWS, 0])

        pl.load(root_auto_manual_siblings[0], source, [2 * MANUAL_TREE_ROWS, 0])
        for index in pl.range(0, 2):
            pl.add(middle_manual_siblings[index], middle_manual_siblings[index], middle_manual_siblings[index])
        for index in pl.range(0, 4):
            pl.add(
                leaves_auto_manual_siblings[index],
                leaves_auto_manual_siblings[index],
                leaves_auto_manual_siblings[index],
            )
        pl.store(output, root_auto_manual_siblings[0], [2 * MANUAL_TREE_ROWS, 0])

        pl.load(root_auto_mixed_siblings[0], source, [3 * MANUAL_TREE_ROWS, 0])
        pl.add(middle_manual[0], middle_manual[0], middle_manual[0])
        pl.add(middle_auto[0], middle_auto[0], middle_auto[0])
        for index in pl.range(0, 4):
            pl.add(
                leaves_auto_mixed_siblings[index],
                leaves_auto_mixed_siblings[index],
                leaves_auto_mixed_siblings[index],
            )
        pl.store(output, root_auto_mixed_siblings[0], [3 * MANUAL_TREE_ROWS, 0])

        pl.load(root_auto_manual_leaves_two[0], source, [4 * MANUAL_TREE_ROWS, 0])
        for index in pl.range(0, 2):
            pl.add(leaves_manual_two[index], leaves_manual_two[index], leaves_manual_two[index])
        pl.store(output, root_auto_manual_leaves_two[0], [4 * MANUAL_TREE_ROWS, 0])

        pl.load(root_auto_manual_leaves_three[0], source, [5 * MANUAL_TREE_ROWS, 0])
        for index in pl.range(0, 2):
            pl.add(
                middle_auto_manual_leaves[index],
                middle_auto_manual_leaves[index],
                middle_auto_manual_leaves[index],
            )
        for index in pl.range(0, 4):
            pl.add(leaves_manual_three[index], leaves_manual_three[index], leaves_manual_three[index])
        pl.store(output, root_auto_manual_leaves_three[0], [5 * MANUAL_TREE_ROWS, 0])

        pl.load(root_manual_unbalanced[0], source, [6 * MANUAL_TREE_ROWS, 0])
        for index in pl.range(0, 2):
            pl.add(
                middle_auto_unbalanced[index],
                middle_auto_unbalanced[index],
                middle_auto_unbalanced[index],
            )
        for index in pl.range(0, 2):
            pl.add(
                leaves_auto_unbalanced[index],
                leaves_auto_unbalanced[index],
                leaves_auto_unbalanced[index],
            )
        pl.store(output, root_manual_unbalanced[0], [6 * MANUAL_TREE_ROWS, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_manual_two_and_three_level_containment():
    def golden(source):
        result = source.clone()
        result[:MANUAL_TREE_ROWS] *= 2.0
        result[MANUAL_TREE_ROWS:4 * MANUAL_TREE_ROWS] *= 4.0
        result[4 * MANUAL_TREE_ROWS:5 * MANUAL_TREE_ROWS] *= 2.0
        result[5 * MANUAL_TREE_ROWS:6 * MANUAL_TREE_ROWS] *= 4.0
        result[6 * MANUAL_TREE_ROWS:6 * MANUAL_TREE_ROWS + 32] *= 4.0
        result[6 * MANUAL_TREE_ROWS + 32:] *= 2.0
        return result

    _run_vector_case(auto_mutex_manual_containment_kernel, MANUAL_ROWS, golden, seed=37)


# =============================================================================
# Cube and Vector: two-level trees in Mat, Left, Right, Acc and Vec
# =============================================================================

CV_M = 64
CV_K = 64
CV_N = 64
CV_HALF = 32
CV_QUARTER = 16


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_cube_vector_containment_kernel(
    lhs: pl.Tensor[[CV_M, CV_K], pl.DT_FP16],
    rhs: pl.Tensor[[CV_K, CV_N], pl.DT_FP16],
    output: pl.Tensor[[CV_M, CV_N], pl.DT_FP32],
):
    lhs_mat_root = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_mat_child = pl.make_tile_group(
        type=pl.TileType(shape=[CV_HALF, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    rhs_mat_root = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    rhs_mat_child = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_HALF], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )

    lhs_left_root = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    lhs_left_child = pl.make_tile_group(
        type=pl.TileType(shape=[CV_HALF, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    rhs_right_root = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    rhs_right_child = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_HALF], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )

    acc_root = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    acc_child = pl.make_tile_group(
        type=pl.TileType(shape=[CV_HALF, CV_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    vec_root = pl.make_tile_group(
        type=pl.TileType(shape=[CV_HALF, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    vec_child = pl.make_tile_group(
        type=pl.TileType(shape=[CV_QUARTER, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )

    with pl.section_cube():
        # Exercise child -> parent transitions in every Cube-side memory space.
        pl.load(lhs_mat_child[0], lhs, [0, 0])
        pl.load(rhs_mat_child[0], rhs, [0, 0])
        pl.load(lhs_mat_root[0], lhs, [0, 0])
        pl.load(rhs_mat_root[0], rhs, [0, 0])
        pl.move(lhs_left_child[0], lhs_mat_child[0])
        pl.move(rhs_right_child[0], rhs_mat_child[0])
        pl.move(lhs_left_root[0], lhs_mat_root[0])
        pl.move(rhs_right_root[0], rhs_mat_root[0])
        pl.matmul(acc_child[0], lhs_left_child[0], rhs_right_child[0])
        pl.matmul(acc_root[0], lhs_left_root[0], rhs_right_root[0])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(vec_root[0], acc_root[0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        subblock = pl.get_subblock_idx()
        row = subblock * CV_HALF
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.add(vec_child[0], vec_child[0], vec_child[0])
        pl.store(output, vec_root[0], [row, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_cube_vector_containment():
    _require_a5(ST_DEVICE)
    torch.manual_seed(35)
    lhs = torch.rand([CV_M, CV_K], device=ST_DEVICE, dtype=torch.float16) * 0.1
    torch.manual_seed(36)
    rhs = torch.rand([CV_K, CV_N], device=ST_DEVICE, dtype=torch.float16) * 0.1
    output = torch.zeros([CV_M, CV_N], device=ST_DEVICE, dtype=torch.float32)

    auto_mutex_cube_vector_containment_kernel(lhs, rhs, output)
    torch.npu.synchronize()

    golden = lhs.cpu().float() @ rhs.cpu().float()
    golden[:CV_QUARTER] *= 2.0
    golden[CV_HALF:CV_HALF + CV_QUARTER] *= 2.0
    torch.testing.assert_close(output.cpu().float(), golden, rtol=5e-3, atol=5e-3)


# =============================================================================
# Three levels, balanced: 1 root -> 4 children -> 8 leaves
# =============================================================================

BALANCED_FOUR_CHILDREN_ROWS = 128


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_balanced_three_level_four_children_vector_kernel(
    source: pl.Tensor[[BALANCED_FOUR_CHILDREN_ROWS, VEC_COLS], pl.DT_FP16],
    output: pl.Tensor[[BALANCED_FOUR_CHILDREN_ROWS, VEC_COLS], pl.DT_FP16],
):
    # FP16 [128, 32] occupies 0x2000 bytes; each child contains two leaves.
    root = pl.make_tile_group(type=_vec_type(128), addrs=0x0000, mutex_ids="auto", depth=1)
    children = pl.make_tile_group(
        type=_vec_type(32), addrs=[0x0000, 0x0800, 0x1000, 0x1800], mutex_ids="auto", depth=4
    )
    leaves = pl.make_tile_group(
        type=_vec_type(16),
        addrs=[0x0000, 0x0400, 0x0800, 0x0C00, 0x1000, 0x1400, 0x1800, 0x1C00],
        mutex_ids="auto",
        depth=8,
    )

    with pl.section_vector():
        pl.load(root[0], source, [0, 0])
        for index in pl.range(0, 4):
            pl.add(children[index], children[index], children[index])
        for index in pl.range(0, 8):
            pl.add(leaves[index], leaves[index], leaves[index])
        pl.store(output, root[0], [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_balanced_three_level_four_children_vector_containment():
    _run_vector_case(
        auto_mutex_balanced_three_level_four_children_vector_kernel,
        BALANCED_FOUR_CHILDREN_ROWS,
        lambda source: source * 4.0,
        seed=38,
    )


# =============================================================================
# Cube, three levels: a three-ID manual root -> 2 children -> 3 and 1 leaves
# =============================================================================

CUBE_MANUAL_THREE_ID_M = 96
CUBE_MANUAL_THREE_ID_K = 16
CUBE_MANUAL_THREE_ID_ROWS = 3 * CUBE_MANUAL_THREE_ID_M


@pl.jit(arch="3510", auto_mutex=True)
def auto_mutex_cube_manual_three_id_root_kernel(
    source: pl.Tensor[[CUBE_MANUAL_THREE_ID_ROWS, CUBE_MANUAL_THREE_ID_K], pl.DT_FP16],
    rhs: pl.Tensor[[CUBE_MANUAL_THREE_ID_K, CV_N], pl.DT_FP16],
    output: pl.Tensor[[CUBE_MANUAL_THREE_ID_ROWS, CV_N], pl.DT_FP32],
):
    # One FP16 NZ column block keeps the contained row ranges contiguous.
    root = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_MANUAL_THREE_ID_M, CUBE_MANUAL_THREE_ID_K], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
        ),
        addrs=0x0000,
        mutex_ids=[[24, 25, 26]],
    )
    children = pl.make_tile_group(
        type=pl.TileType(
            shape=[48, CUBE_MANUAL_THREE_ID_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=[0x0000, 0x0600],
        mutex_ids="auto",
        depth=2,
    )
    leaves = pl.make_tile_group(
        type=pl.TileType(
            shape=[16, CUBE_MANUAL_THREE_ID_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=[0x0000, 0x0200, 0x0400, 0x0600],
        mutex_ids="auto",
        depth=4,
    )
    rhs_mat = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_MANUAL_THREE_ID_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x4000,
        mutex_ids=[1],
    )
    lhs_left = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_MANUAL_THREE_ID_M, CUBE_MANUAL_THREE_ID_K], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    rhs_right = pl.make_tile_group(
        type=pl.TileType(shape=[CUBE_MANUAL_THREE_ID_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[2],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_MANUAL_THREE_ID_M, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc
        ),
        addrs=0x0000,
        mutex_ids=[3],
    )

    with pl.section_cube():
        pl.load(rhs_mat[0], rhs, [0, 0])
        pl.move(rhs_right[0], rhs_mat[0])

        # Store every stage so parent and intermediate loads remain observable.
        pl.load(root[0], source, [0, 0])
        pl.move(lhs_left[0], root[0])
        pl.matmul(acc[0], lhs_left[0], rhs_right[0])
        pl.store(output, acc[0], [0, 0])

        for index in pl.range(0, 2):
            pl.load(children[index], source, [CUBE_MANUAL_THREE_ID_M + index * 48, 0])
        pl.move(lhs_left[0], root[0])
        pl.matmul(acc[0], lhs_left[0], rhs_right[0])
        pl.store(output, acc[0], [CUBE_MANUAL_THREE_ID_M, 0])

        # Three leaves cover the first child; one leaf covers the second child's first 16 rows.
        for index in pl.range(0, 4):
            pl.load(leaves[index], source, [2 * CUBE_MANUAL_THREE_ID_M + index * 16, 0])
        pl.move(lhs_left[0], root[0])
        pl.matmul(acc[0], lhs_left[0], rhs_right[0])
        pl.store(output, acc[0], [2 * CUBE_MANUAL_THREE_ID_M, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_auto_mutex_cube_manual_three_id_root():
    _require_a5(ST_DEVICE)
    torch.manual_seed(39)
    source = torch.rand(
        [CUBE_MANUAL_THREE_ID_ROWS, CUBE_MANUAL_THREE_ID_K], device=ST_DEVICE, dtype=torch.float16
    ) * 0.1
    torch.manual_seed(40)
    rhs = torch.rand([CUBE_MANUAL_THREE_ID_K, CV_N], device=ST_DEVICE, dtype=torch.float16) * 0.1
    output = torch.zeros([CUBE_MANUAL_THREE_ID_ROWS, CV_N], device=ST_DEVICE, dtype=torch.float32)

    auto_mutex_cube_manual_three_id_root_kernel(source, rhs, output)
    torch.npu.synchronize()

    golden_source = source.cpu().float()
    # The last 32 rows are outside the leaves and retain the intermediate-stage values.
    golden_source[2 * CUBE_MANUAL_THREE_ID_M + 64:] = golden_source[
        CUBE_MANUAL_THREE_ID_M + 64:2 * CUBE_MANUAL_THREE_ID_M
    ]
    golden = golden_source @ rhs.cpu().float()
    torch.testing.assert_close(output.cpu().float(), golden, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    test_auto_mutex_two_level_vector_containment()
    test_auto_mutex_balanced_three_level_vector_containment()
    test_auto_mutex_balanced_three_level_four_children_vector_containment()
    test_auto_mutex_unbalanced_three_level_vector_containment()
    test_auto_mutex_four_level_vector_containment()
    test_auto_mutex_manual_two_and_three_level_containment()
    test_auto_mutex_cube_vector_containment()
    test_auto_mutex_cube_manual_three_id_root()
