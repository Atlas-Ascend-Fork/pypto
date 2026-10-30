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
"""Early-start set with ``force=True`` IR print style.

The sibling ``.pypto`` file holds the IR produced by ``str(<ir.Function>)``.
"""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl


def test_set_next_task_start_force_true():
    """Test unified printing for pl.system.set_next_task_start with force=True."""

    @pl.jit(auto_mutex=False)
    def main(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.set_next_task_start(force=True)
        _test_result = x

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    printed = str(main)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
