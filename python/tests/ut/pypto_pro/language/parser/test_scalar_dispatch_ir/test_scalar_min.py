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
"""Extracted from language/parser/test_scalar_dispatch.py: test_scalar_min.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl


def test_scalar_min():
    """Test pl.min(scalar, scalar) dispatches to ir.min_."""

    @pl.jit(auto_mutex=False)
    def test_min(
        config: pl.Tensor[[2], pl.DT_INT64],
        out: pl.Tensor[[2, 16, 128], pl.DT_FP32],
    ):
        a: pl.DT_UINT64 = pl.getval(config, 0)
        b: pl.DT_UINT64 = pl.getval(config, 1)
        c = pl.min(a, b)
        _ = c + 1
        _test_result = out

    test_min_program, _ = test_min.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_min = test_min_program.get_function(test_min.__name__)

    assert isinstance(test_min, ir.Function)
    printed = str(test_min)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
