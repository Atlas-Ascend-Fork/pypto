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
"""Parser tests for pointer null-check syntax on pl.Ptr and pl.Tensor parameters."""

from pypto_pro import ir
from pypto_pro._errors import InvalidVal
import pypto_pro.language as pl
import pytest


def test_is_none_requires_simple_variable():
    """'is None' must be used on simple variable names, not complex expressions."""
    with pytest.raises(InvalidVal, match="only supported on simple variable names"):
        @pl.jit(auto_mutex=False)
        def bad_is_none(_jit_entry: pl.DT_INT64):
            x: pl.DT_INT32 = 42
            if (x + 1) is None:  # Complex expression not allowed
                pass

        bad_is_none.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_is_none_on_tensor_supported():
    """'is None' on a pl.Tensor parameter lowers to its GM data-pointer null check.

    aicore and aicpu kernels share one path: pl.Ptr tests the pointer, a
    tensor tests its GM data pointer -- the flattened form of the cpp
    ``Tensor* != nullptr && GetData() != nullptr`` guard.
    """
    @pl.jit(auto_mutex=False)
    def tensor_none(src: pl.Tensor[[4], pl.DT_FP16], bias: pl.Tensor[[4], pl.DT_FP16]):
        if bias is None:
            src[0] = 0.0

    tensor_none.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_is_none_on_scalar_rejected():
    """'is None' on a non-pointer scalar parameter is rejected at parse time."""
    with pytest.raises(InvalidVal, match="pl.Ptr and pl.Tensor"):
        @pl.jit(auto_mutex=False)
        def scalar_none(x: pl.DT_INT64):
            if x is None:
                pass

        scalar_none.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
