#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""The sibling ``.pypto`` file holds the complete TextDump output."""

from pathlib import Path
import re

from pypto_pro import ir
import pypto_pro.language as pl


def test_cast_src_declared_dynamic_valid_shape_skips_region_check():
    # A -1 dim in a declared valid_shape marks it dynamic (it is narrowed later by
    # pl.set_validshape), so it carries no static region to compare and the
    # src-covers-out check is skipped.
    @pl.jit(auto_mutex=False)
    def main(src: pl.Tensor[[64, 64], pl.DT_FP32], output: pl.Tensor[[64, 128], pl.DT_FP32]):
        src_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, valid_shape=[-1, -1])
        dst_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, valid_shape=[64, 128])
        src_tile = pl.make_tile(src_type, addr=0x0000)
        dst_tile = pl.make_tile(dst_type, addr=0x4000)
        pl.load(src_tile, src, [0, 0])
        pl.cast(dst_tile, src_tile, mode=pl.RoundMode.CAST_NONE)
        _test_result = pl.store(output, dst_tile, [0, 0])

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(main.__name__)

    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(func))
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
