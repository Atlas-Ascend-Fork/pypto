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

"""Parser tests for the public ``pl.simt.cast`` API.

Operation-level pl.simt.cast tests parse a captured kernel through
``KernelDef.parse_target_program`` so diagnostics and generated IR use the
same parser entry as production kernels.
"""

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidType, InvalidVal, NotSupported
import pypto_pro.language as pl
import pytest


def test_simt_cast_is_not_exported_as_shared_language_api():
    assert not hasattr(pl, "scalar_cast")


def test_simt_cast_rejects_ordinary_function():
    with pytest.raises(InvalidOperation, match="can only be used inside a SIMT function"):

        @pl.jit(auto_mutex=False)
        def unsupported_context(value: pl.DT_INT64):
            _test_result = pl.simt.cast(value, pl.DT_INT32)

        unsupported_context.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_is_available_inside_simt_helper():
    @pl.vector_function(mode="simt", max_threads=1)
    def cast_value(value: pl.DT_FP32):
        _test_result = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            cast_value[1](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    cast_value = program.get_function(cast_value.__name__)

    function_ir = str(cast_value)
    assert "simt.cast" in function_ir


def test_simt_cast_rejects_unsupported_dtype_pair():
    with pytest.raises(NotSupported, match=r"pl\.simt\.cast\(\) does not support"):

        @pl.vector_function(mode="simt", max_threads=1)
        def unsupported(value: pl.DT_FP16):
            _test_result = pl.simt.cast(value, pl.DT_INT32)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP16):
            with pl.section_vector():
                unsupported[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_odd_rounding_for_bfloat16():
    with pytest.raises(NotSupported, match=r"pl\.simt\.cast\(\) does not support"):

        @pl.vector_function(mode="simt", max_threads=1)
        def unsupported_mode(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_ODD)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                unsupported_mode[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_tile_operand():
    @pl.vector_function(mode="simt")
    def tile_operand(value):
        _ = pl.simt.cast(value, pl.DT_FP16)

    @pl.vector_function(mode="simt", max_threads=1)
    def entry(value):
        tile_operand(value)

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        value = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            entry[1](value)

    with pytest.raises(InvalidType, match="value must be a scalar expression"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_plain_integer_round_mode():
    with pytest.raises(InvalidVal, match="expects an enum value"):

        @pl.vector_function(mode="simt", max_threads=1)
        def integer_mode(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_FP16, mode=1)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                integer_mode[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_wrong_positional_arity():
    with pytest.raises(InvalidArgument, match="missing a required argument: 'dtype'"):

        @pl.vector_function(mode="simt", max_threads=1)
        def missing_dtype(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                missing_dtype[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_non_dtype_target():
    with pytest.raises(InvalidType, match=r"dtype must be a pl\.DT_\* value"):

        @pl.vector_function(mode="simt", max_threads=1)
        def invalid_dtype(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, 1)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                invalid_dtype[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_unexpected_keyword():
    with pytest.raises(InvalidArgument, match="unexpected keyword argument 'saturate'"):

        @pl.vector_function(mode="simt", max_threads=1)
        def unexpected_keyword(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_FP16, saturate=True)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                unexpected_keyword[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_bitcast_preserves_invalid_type_error():
    with pytest.raises(InvalidType, match="same bit width"):

        @pl.vector_function(mode="simt", max_threads=1)
        def invalid_bit_width(value: pl.DT_FP16):
            _test_result = pl.simt.bitcast(value, pl.DT_UINT32)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP16):
            with pl.section_vector():
                invalid_bit_width[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
