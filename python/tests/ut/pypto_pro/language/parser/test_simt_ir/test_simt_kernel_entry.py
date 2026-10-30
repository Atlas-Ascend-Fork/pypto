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

"""SIMT kernel entry IR and launch-bound metadata."""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl


def test_simt_kernel_parser_records_root_type_and_launch_bound():
    @pl.jit(mode="simt", max_threads=256)
    def _simt_kernel_copy(
        out: pl.Tensor[[1024], pl.DT_FP32],
        src: pl.Tensor[[1024], pl.DT_FP32],
        count: pl.DT_UINT32,
    ):
        tid = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
        if tid < count:
            out[tid] = src[tid]

    program, matched = _simt_kernel_copy.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert not matched
    assert program.get_function("_simt_kernel_copy").get_attr("max_threads") == 256

    printed = str(program)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
