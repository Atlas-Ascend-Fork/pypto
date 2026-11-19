# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GridGroup operations in a standalone SIMT kernel and its helper."""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl
from test_common import assert_ir_matches_dump


def test_grid_group():
    @pl.vector_function(mode="simt")
    def grid_size() -> pl.DT_UINT64:
        grid = pl.simt.this_grid()
        size = grid.size()
        grid.sync()
        return size

    @pl.jit(mode="simt", max_threads=64)
    def kernel(out: pl.Tensor[[12], pl.DT_UINT64]):
        grid = pl.simt.this_grid()
        out[0] = grid.is_valid()
        out[1] = grid.thread_rank()
        out[2] = grid.block_rank()
        out[3] = grid.size()
        out[4] = grid.num_blocks()
        out[5] = grid.group_dim().x
        out[6] = grid.group_dim().y
        out[7] = grid.group_dim().z
        out[8] = grid.block_index().x
        out[9] = grid.block_index().y
        out[10] = grid.block_index().z
        grid.sync()
        out[11] = grid_size()

    program, matched = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert not matched
    assert_ir_matches_dump(program, Path(__file__).with_suffix(".pypto"))
