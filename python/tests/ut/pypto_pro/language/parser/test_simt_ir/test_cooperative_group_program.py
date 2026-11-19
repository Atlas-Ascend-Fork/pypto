# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Complete IR for cooperative-group construction and member operations."""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl
from test_common import assert_ir_matches_dump


def test_cooperative_group_program():
    @pl.vector_function(mode="simt")
    def helper():
        grid = pl.simt.this_grid()
        grid.sync()

    @pl.jit(mode="simt", max_threads=32)
    def kernel(
        out: pl.Tensor[[39, 32], pl.DT_UINT64],
        widths: pl.Tensor[[1], pl.DT_UINT32],
        width: pl.DT_UINT32,
    ):
        # Group creation and partitioning.
        block = pl.simt.this_thread_block()
        tile = pl.simt.tiled_partition(block, 8)
        child = pl.simt.tiled_partition(tile, 4)
        runtime_tile = pl.simt.tiled_partition(block, width)
        loaded_tile = pl.simt.tiled_partition(block, widths[0])
        active = pl.simt.coalesced_threads()
        tid = block.thread_rank()
        rank = tile.thread_rank()
        predicate = rank < 4
        subgroup = pl.simt.binary_partition(tile, predicate)
        active_subgroup = pl.simt.binary_partition(active, predicate)
        grid = pl.simt.this_grid()

        # Queries, shuffles and votes retain their scalar result types in IR.
        out[0, tid] = block.size()
        out[1, tid] = tile.size()
        out[2, tid] = tile.meta_group_rank()
        out[3, tid] = tile.meta_group_size()
        out[4, tid] = child.thread_rank()
        out[5, tid] = runtime_tile.size()
        out[6, tid] = loaded_tile.thread_rank()
        out[7, tid] = active.size()
        out[8, tid] = subgroup.size()
        out[9, tid] = subgroup.meta_group_rank()
        out[10, tid] = tile.shfl(tid, 0)
        out[11, tid] = tile.shfl_up(tid, 1)
        out[12, tid] = tile.shfl_down(tid, 1)
        out[13, tid] = tile.shfl_xor(tid, 1)
        out[14, tid] = subgroup.shfl(tid, 0)
        out[15, tid] = subgroup.shfl_up(tid, 1)
        out[16, tid] = subgroup.shfl_down(tid, 1)
        out[17, tid] = tile.any(predicate)
        out[18, tid] = tile.all(predicate)
        out[19, tid] = tile.ballot(predicate)
        out[20, tid] = subgroup.ballot(predicate)
        out[21, tid] = grid.is_valid()
        out[22, tid] = grid.thread_rank()
        out[23, tid] = grid.size()
        out[24, tid] = grid.block_rank()
        out[25, tid] = grid.num_blocks()
        out[26, tid] = active.thread_rank()
        out[27, tid] = subgroup.meta_group_size()
        out[28, tid] = subgroup.any(predicate)
        out[29, tid] = subgroup.all(predicate)
        out[30, tid] = child.meta_group_rank()
        out[31, tid] = child.meta_group_size()
        out[32, tid] = child.size()
        out[33, tid] = block.thread_index().x
        out[34, tid] = block.group_index().y
        out[35, tid] = block.group_dim().z
        out[36, tid] = grid.group_dim().x
        out[37, tid] = grid.block_index().y
        out[38, tid] = active_subgroup.meta_group_rank()

        # Synchronize each group kind and call a SIMT helper.
        block.sync()
        tile.sync()
        active.sync()
        helper()

    program, matched = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert not matched
    assert_ir_matches_dump(program, Path(__file__).with_suffix(".pypto"))
