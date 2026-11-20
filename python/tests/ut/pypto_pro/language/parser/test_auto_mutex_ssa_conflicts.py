# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only SSA mutex-count conflicts through the frontend allocation pass."""

from pypto_pro._errors import InvalidArgument
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir

ROWS = 64
COLS = 32
HALF = ROWS // 2


def _vec_type(rows):
    return pl.TileType(shape=[rows, COLS], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)


@pl.jit(arch="3510", auto_mutex=True)
def ssa_conflicting_counts_kernel(
    source: pl.Tensor[[ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    auto = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids="auto", depth=1)
    two = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids=[[8, 9]])
    three = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x4000, mutex_ids=[[8, 9, 10]])
    with pl.section_vector():
        selected = auto[0]
        if control[0] != 0:
            selected = auto[0]
        else:
            selected = two[0]
        if control[1] != 0:
            selected = three[0]
        pl.load(selected, source, [0, 0])
        pl.add(selected, selected, selected)
        pl.store(output, selected, [0, 0])


@pl.jit(arch="3510", auto_mutex=True)
def ssa_single_id_conflict_kernel(
    source: pl.Tensor[[ROWS, COLS], pl.DT_FP16],
    output: pl.Tensor[[ROWS, COLS], pl.DT_FP16],
    control: pl.Tensor[[3], pl.DT_INT64],
):
    root = pl.make_tile_group(type=_vec_type(ROWS), addrs=0, mutex_ids="auto", depth=1)
    leaves = pl.make_tile_group(type=_vec_type(HALF), addrs=[0, 0x0800], mutex_ids=[8, 9])
    single = pl.make_tile_group(type=_vec_type(ROWS), addrs=0x2000, mutex_ids=[10])
    with pl.section_vector():
        pl.load(root[0], source, [0, 0])
        pl.load(single[0], source, [0, 0])
        for child in pl.range(0, 2):
            pl.add(leaves[child], leaves[child], leaves[child])
        selected = root[0]
        if control[0] == 0:
            selected = single[0]
        pl.add(selected, selected, selected)
        pl.store(output, selected, [0, 0])


@pytest.mark.parametrize("kernel, lower, upper", [
    (ssa_conflicting_counts_kernel, 3, 2),
    (ssa_single_id_conflict_kernel, 2, 1),
])
def test_ssa_conflicting_manual_counts_are_rejected_by_allocation(kernel, lower, upper):
    # Includes equal placeholder counts: AUTO/count=1 must still constrain the
    # final count, rather than letting containment grow it beyond the manual ID.
    with pytest.raises(InvalidArgument, match="TileGroup or SSA merge") as exc_info:
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert f"requires at least {lower} IDs but allows at most {upper}" in str(exc_info.value)
