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

"""A SIMT VF definition is allowed when no VF launch is reached."""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl
import pytest


@pytest.mark.parametrize("through_helper", [False, True], ids=["direct", "helper"])
def test_simt_kernel_ignores_unlaunched_vf(through_helper):
    launch_vf = False

    @pl.vector_function(mode="simt", max_threads=32)
    def vf(value):
        return

    @pl.vector_function(mode="simt")
    def helper(value):
        if launch_vf:
            vf[32](value)

    @pl.jit(mode="simt", max_threads=32)
    def kernel(value: pl.DT_INT32):
        if through_helper:
            helper(value)
        elif launch_vf:
            vf[32](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    printed = str(program)
    case = "helper" if through_helper else "direct"
    golden = Path(__file__).with_name(f"{Path(__file__).stem}__{case}.pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
