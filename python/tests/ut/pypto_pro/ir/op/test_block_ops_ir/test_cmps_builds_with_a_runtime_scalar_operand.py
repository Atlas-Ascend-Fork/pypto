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


def test_cmps_builds_with_a_runtime_scalar_operand():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[32, 32], pl.DT_FP32],
        output: pl.Tensor[[32, 32], pl.DT_UINT8],
        scalar: pl.DT_FP32,
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        mask_type = pl.TileType(shape=[32, 32], dtype=pl.DT_UINT8)
        tile_a = pl.make_tile(tile_type, addr=0x0000)
        mask = pl.make_tile(mask_type, addr=0x1000)
        pl.load(tile_a, a, [0, 0])
        pl.eq(mask, tile_a, scalar)
        _test_result = pl.store(output, mask, [0, 0])

    program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(main.__name__)

    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(func))
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
