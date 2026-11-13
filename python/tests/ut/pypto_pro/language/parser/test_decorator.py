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
"""Unit tests for @pl.inline and @pl.vector_function decorators.

The former @pl.function / @pl.program decorators have been removed.
Use @pl.jit to define kernels.
"""

from pypto_pro._errors import InvalidVal, NotSupported
import pypto_pro.language as pl
import pytest


def test_removed_frontend_apis_are_not_exported():
    removed_apis = (
        "function",
        "program",
        "kernel",
        "KernelDef",
        "parse",
        "loads",
        "parse_program",
        "loads_program",
    )

    for name in removed_apis:
        assert not hasattr(pl, name)
        assert not hasattr(pl.parser, name)

    assert not hasattr(pl.parser.decorator, "function")
    assert not hasattr(pl.parser.decorator, "program")


def test_inline_deprecated_warning():
    """@pl.inline is deprecated and should warn at decoration time."""

    with pytest.warns(DeprecationWarning, match="@pl.inline is deprecated"):

        @pl.inline
        def deprecated_inline(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            return x

    assert deprecated_inline.__name__ == "deprecated_inline"


def test_vector_function_marker():
    """@pl.vector_function sets the internal marker attribute."""

    @pl.vector_function
    def vf_body(in_tile, out_tile):
        pass

    decorator = pl.parser.decorator
    assert vf_body._pypto_vector_function_mode == "simd"
    assert decorator.is_vector_function(vf_body)
    assert not decorator.is_simt_function(vf_body)


def test_vector_function_supports_all_valid_forms():
    @pl.vector_function
    def implicit_simd():
        pass

    @pl.vector_function(mode="simd")
    def explicit_simd():
        pass

    @pl.vector_function(mode="simt", max_threads=256)
    def simt_vector_function():
        pass

    @pl.vector_function(mode="simt")
    def simt_helper():
        pass

    decorator = pl.parser.decorator
    assert decorator.is_vector_function(implicit_simd)
    assert decorator.is_vector_function(explicit_simd)
    assert decorator.is_simt_function(simt_vector_function)
    assert decorator.is_simt_function(simt_helper)
    assert decorator.get_simt_max_threads(simt_vector_function) == 256
    assert decorator.get_simt_max_threads(simt_helper) is None


def test_vector_function_empty_call_is_not_supported():
    with pytest.raises(NotSupported, match=r"@pl\.vector_function\(\) is not supported"):
        pl.vector_function()


@pytest.mark.parametrize("mode", ["scalar", "SIMT", 1])
def test_vector_function_rejects_invalid_mode(mode):
    with pytest.raises(ValueError, match="mode must be 'simd' or 'simt'"):
        pl.vector_function(mode=mode)


def test_vector_function_rejects_max_threads_in_simd_mode():
    with pytest.raises(InvalidVal, match="only supported when mode='simt'"):
        pl.vector_function(mode="simd", max_threads=32)


@pytest.mark.parametrize("max_threads", [True, 1.5, "32"])
def test_vector_function_rejects_non_integer_max_threads(max_threads):
    with pytest.raises(InvalidVal, match="max_threads must be an integer"):
        pl.vector_function(mode="simt", max_threads=max_threads)


@pytest.mark.parametrize("max_threads", [0, 2049])
def test_vector_function_rejects_out_of_range_max_threads(max_threads):
    with pytest.raises(ValueError, match=r"max_threads must be in \[1, 2048\]"):
        pl.vector_function(mode="simt", max_threads=max_threads)


def test_is_vector_function():
    """is_vector_function correctly detects decorated functions."""

    @pl.vector_function
    def vf_func(a, b):
        pass

    assert pl.parser.decorator.is_vector_function(vf_func) is True


def test_is_vector_function_false_for_plain():
    """is_vector_function returns False for undecorated functions."""

    def plain_func(a, b):
        pass

    assert pl.parser.decorator.is_vector_function(plain_func) is False
