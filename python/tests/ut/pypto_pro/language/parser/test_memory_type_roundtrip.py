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
"""Comprehensive tests for MemRef, MemorySpace, and TileView."""

from pypto_pro import ir
import pypto_pro.language as pl


def test_backwards_compat_three_args_layout():
    """Existing 3-arg [shape, dtype, layout] still works for Tensor."""

    @pl.jit(auto_mutex=False)
    def test_fn(x: pl.Tensor[[64], pl.DT_FP32]):
        y: pl.Tensor[[64], pl.DT_FP32, pl.NZ] = pl.make_tensor(x, [64], [1])  # noqa: F841

    test_fn_program, _ = test_fn.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    test_fn = test_fn_program.get_function(test_fn.__name__)

    assert isinstance(test_fn, ir.Function)
