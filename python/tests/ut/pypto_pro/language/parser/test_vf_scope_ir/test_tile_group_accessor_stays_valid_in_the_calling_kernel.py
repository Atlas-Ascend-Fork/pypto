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
"""Extracted from language/parser/test_vf_scope.py.

Original case: test_tile_group_accessor_stays_valid_in_the_calling_kernel.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path
import re

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813


def test_tile_group_accessor_stays_valid_in_the_calling_kernel():
    """The companion to the rejection above: the accessor is fine one scope out.

    Without this the rejection could be over-broad -- forbidding the tile itself rather than
    where the cursor is advanced -- and every VF kernel reading a rotating group would break.
    """

    @pl.vector_function
    def use(tile):
        reg = vf.load_align(tile, 0)
        vf.store_align(tile, reg, vf.create_mask(), 0)

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[1, 128], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tile_type, addrs=[0, 512], mutex_ids=[0, 1])
        with pl.section_vector():
            use(group.next())

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(kernel.__name__)
    sections = [stmt for stmt in func.body.stmts if isinstance(stmt, ir.SectionStmt)]
    assert [section.section_kind for section in sections] == [ir.SectionKind.VF]

    printed = str(func)
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
