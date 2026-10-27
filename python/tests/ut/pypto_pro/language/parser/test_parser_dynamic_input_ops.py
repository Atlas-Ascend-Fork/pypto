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
"""Parser tests for the DYNAMIC-input primitives.

Their expression parameters (i / axis) are positional-only, so naming them as keywords is rejected
by the declaration's own binding (Gate 1) instead of being silently dropped by the parser.  Their
first argument is an ordinary pointer (``pl.Ptr[dtype]``): the descriptor is decoded from the
address the framework passes, so no separate descriptor type is required.
"""

from pypto_pro._errors import InvalidArgument, InvalidType
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _parse_kernel(kernel_def) -> ir.Program:
    return kernel_def.to_kernel_def().parse_target_program(ir.SectionKind.Vector)[0]


@pl.jit
def _ptr_keyword_kernel(x: pl.Ptr[pl.DT_FP16]):
    p = pl.dynamic_input_ptr(x, i=0)  # noqa: F841


@pl.jit
def _dim_keyword_kernel(x: pl.Ptr[pl.DT_FP16]):
    n = pl.dynamic_input_dim(x, 0, axis=0)  # noqa: F841


@pl.jit
def _ptr_plain_ptr_kernel(x: pl.Ptr[pl.DT_FP16]):
    p = pl.dynamic_input_ptr(x, 0)  # noqa: F841


@pl.jit
def _dim_plain_ptr_kernel(x: pl.Ptr[pl.DT_FP16]):
    n = pl.dynamic_input_dim(x, 0, 0)  # noqa: F841


@pl.jit
def _ptr_float_index_kernel(x: pl.Ptr[pl.DT_FP16], f: pl.DT_FP32):
    p = pl.dynamic_input_ptr(x, f)  # noqa: F841


@pl.jit
def _dim_float_axis_kernel(x: pl.Ptr[pl.DT_FP16], f: pl.DT_FP32):
    n = pl.dynamic_input_dim(x, 0, f)  # noqa: F841


def test_ptr_keyword_index_rejected():
    """dynamic_input_ptr(x, i=0) must fail: i is positional-only."""
    with pytest.raises(InvalidArgument):
        _parse_kernel(_ptr_keyword_kernel)


def test_dim_keyword_axis_rejected():
    """dynamic_input_dim(x, 0, axis=0) must fail: axis is positional-only."""
    with pytest.raises(InvalidArgument):
        _parse_kernel(_dim_keyword_kernel)


def test_ptr_plain_pointer_accepted():
    """A plain pl.Ptr[dtype] is a valid descriptor argument."""
    assert _parse_kernel(_ptr_plain_ptr_kernel) is not None


def test_dim_plain_pointer_accepted():
    """A plain pl.Ptr[dtype] is a valid descriptor argument."""
    assert _parse_kernel(_dim_plain_ptr_kernel) is not None


def test_ptr_float_index_rejected():
    """dynamic_input_ptr(x, f) must fail: i must be an integer scalar."""
    with pytest.raises(InvalidType):
        _parse_kernel(_ptr_float_index_kernel)


def test_dim_float_axis_rejected():
    """dynamic_input_dim(x, 0, f) must fail: axis must be an integer scalar."""
    with pytest.raises(InvalidType):
        _parse_kernel(_dim_float_axis_kernel)
