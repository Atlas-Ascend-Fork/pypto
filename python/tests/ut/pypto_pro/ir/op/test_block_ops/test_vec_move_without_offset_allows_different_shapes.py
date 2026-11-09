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
"""The sibling ``.pypto`` file holds the complete TextDump output."""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl
from test_common import assert_ir_matches_dump


def test_vec_move_without_offset_allows_different_shapes():
    @pl.jit(auto_mutex=False)
    def main(_jit_entry: pl.DT_INT64):
        dst_type = pl.TileType(shape=[32, 16], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200)
        src = pl.make_tile(src_type, addr=0x31200)
        pl.move(dst, src)

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function(main.__name__)
    assert_ir_matches_dump(function, Path(__file__).with_suffix(".pypto"))
