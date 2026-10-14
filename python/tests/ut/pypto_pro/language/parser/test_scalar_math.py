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

"""Parser tests for public ``pl.simt`` scalar math APIs.

Operation-level pl.simt scalar math tests parse a captured kernel through
``KernelDef.parse_target_program`` so diagnostics and generated IR use the
same parser entry as production kernels.
"""

from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidType, NotSupported
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _parse_tile_function(function, tile_specs):
    if len(tile_specs) == 1:
        (shape0, dtype0) = tile_specs[0]

        @pl.vector_function(mode="simt", max_threads=1)
        def entry(tile0):
            function(tile0)

        @pl.jit
        def kernel(_jit_entry: pl.DT_INT64):
            tile0 = pl.make_tile(
                pl.TileType(shape=shape0, dtype=dtype0, target_memory=pl.MemorySpace.Vec),
                addr=0,
            )
            with pl.section_vector():
                entry[1](tile0)

    elif len(tile_specs) == 2:
        (shape0, dtype0), (shape1, dtype1) = tile_specs

        @pl.vector_function(mode="simt", max_threads=1)
        def entry(tile0, tile1):
            function(tile0, tile1)

        @pl.jit
        def kernel(_jit_entry: pl.DT_INT64):
            tile0 = pl.make_tile(
                pl.TileType(shape=shape0, dtype=dtype0, target_memory=pl.MemorySpace.Vec),
                addr=0,
            )
            tile1 = pl.make_tile(
                pl.TileType(shape=shape1, dtype=dtype1, target_memory=pl.MemorySpace.Vec),
                addr=4096,
            )
            with pl.section_vector():
                entry[1](tile0, tile1)

    elif len(tile_specs) == 3:
        (shape0, dtype0), (shape1, dtype1), (shape2, dtype2) = tile_specs

        @pl.vector_function(mode="simt", max_threads=1)
        def entry(tile0, tile1, tile2):
            function(tile0, tile1, tile2)

        @pl.jit
        def kernel(_jit_entry: pl.DT_INT64):
            tile0 = pl.make_tile(
                pl.TileType(shape=shape0, dtype=dtype0, target_memory=pl.MemorySpace.Vec),
                addr=0,
            )
            tile1 = pl.make_tile(
                pl.TileType(shape=shape1, dtype=dtype1, target_memory=pl.MemorySpace.Vec),
                addr=4096,
            )
            tile2 = pl.make_tile(
                pl.TileType(shape=shape2, dtype=dtype2, target_memory=pl.MemorySpace.Vec),
                addr=8192,
            )
            with pl.section_vector():
                entry[1](tile0, tile1, tile2)

    else:
        raise ValueError("Only one to three Tile parameters are supported")

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def test_scalar_math_is_available_in_simt_functions():
    @pl.vector_function(mode="simt", max_threads=1)
    def scalar_math(value: pl.DT_FP32, rhs: pl.DT_FP32, addend: pl.DT_FP32):
        transformed = pl.simt.max(pl.simt.min(value, rhs), pl.simt.abs(addend))
        transformed = pl.simt.rsqrt(pl.simt.sqrt(transformed))
        transformed = pl.simt.exp2(pl.simt.exp(transformed))
        transformed = pl.simt.log2(pl.simt.log(pl.simt.log1p(transformed)))
        transformed = pl.simt.tanh(pl.simt.cos(pl.simt.sin(transformed)))
        transformed = pl.simt.trunc(
            pl.simt.floor(pl.simt.ceil(pl.simt.round(pl.simt.rint(transformed))))
        )
        classified = pl.simt.isnan(transformed) or pl.simt.isinf(transformed)
        if classified:
            transformed = value
        _test_result = pl.simt.fma(transformed, rhs, addend)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32, rhs: pl.DT_FP32, addend: pl.DT_FP32):
        with pl.section_vector():
            scalar_math[1](value, rhs, addend)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    scalar_math = program.get_function(scalar_math.__name__)

    function_ir = str(scalar_math)
    for name in (
        "abs",
        "min",
        "max",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "log",
        "log2",
        "log1p",
        "sin",
        "cos",
        "tanh",
        "rint",
        "round",
        "floor",
        "ceil",
        "trunc",
        "isnan",
        "isinf",
        "fma",
    ):
        assert f"simt.{name}(" in function_ir


def test_scalar_math_supports_fp16_and_bf16_scalars():
    @pl.vector_function(mode="simt")
    def fp16_math(
        src,
        dst,
    ):
        value = src[0, 0]
        dst[0, 0] = pl.simt.fma(pl.simt.round(pl.simt.sin(value)), value, value)

    @pl.vector_function(mode="simt")
    def bf16_math(
        src,
        dst,
        flags,
    ):
        value = src[0, 0]
        dst[0, 0] = pl.simt.tanh(pl.simt.exp2(value))
        flags[0, 0] = pl.simt.isinf(value)

    fp16_ir = str(_parse_tile_function(fp16_math, [([1, 32], pl.DT_FP16), ([1, 32], pl.DT_FP16)]))
    bf16_ir = str(
        _parse_tile_function(
            bf16_math,
            [([1, 32], pl.DT_BF16), ([1, 32], pl.DT_BF16), ([1, 32], pl.DT_BOOL)],
        )
    )
    assert "simt.fma(" in fp16_ir
    assert "simt.sin(" in fp16_ir
    assert "simt.tanh(" in bf16_ir
    assert "simt.isinf(" in bf16_ir


def test_scalar_math_supports_int64_abs_and_integer_min_max():
    @pl.vector_function(mode="simt")
    def integer_math(
        src,
        dst,
    ):
        dst[0, 0] = pl.simt.abs(src[0, 0])
        dst[0, 1] = pl.simt.max(pl.simt.min(src[0, 0], src[0, 1]), src[0, 0])

    function_ir = str(
        _parse_tile_function(integer_math, [([1, 32], pl.DT_INT64), ([1, 32], pl.DT_INT64)])
    )
    assert "simt.abs(" in function_ir
    assert "simt.min(" in function_ir
    assert "simt.max(" in function_ir


def test_scalar_math_rejects_ordinary_function():
    with pytest.raises(InvalidOperation, match="can only be used inside a SIMT function"):

        @pl.jit(auto_mutex=False)
        def unsupported(value: pl.DT_FP32):
            _test_result = pl.simt.exp(value)

        unsupported.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_scalar_math_rejects_unsupported_dtype_and_mixed_operands():
    @pl.vector_function(mode="simt")
    def unsupported_log1p(value):
        value[0, 0] = pl.simt.log1p(value[0, 0])

    with pytest.raises(NotSupported, match="supports only fp32"):
        _parse_tile_function(unsupported_log1p, [([1, 32], pl.DT_FP16)])

    @pl.vector_function(mode="simt")
    def unsupported_exp(value):
        value[0, 0] = pl.simt.exp(value[0, 0])

    with pytest.raises(NotSupported, match="supports only fp16, bfloat16, fp32"):
        _parse_tile_function(unsupported_exp, [([1, 32], pl.DT_INT32)])

    @pl.vector_function(mode="simt")
    def mixed_min(lhs, rhs):
        lhs[0, 0] = pl.simt.min(lhs[0, 0], rhs[0, 0])

    with pytest.raises(InvalidType, match="same dtype"):
        _parse_tile_function(mixed_min, [([1, 32], pl.DT_FP16), ([1, 32], pl.DT_FP32)])


def test_scalar_math_rejects_tile_operand():
    @pl.vector_function(mode="simt")
    def unsupported(value):
        _ = pl.simt.sqrt(value)

    with pytest.raises(InvalidType, match="must be a scalar expression"):
        _parse_tile_function(unsupported, [([1, 32], pl.DT_FP32)])


def test_scalar_math_rejects_wrong_arity_and_accepts_keywords():
    with pytest.raises(InvalidArgument, match="missing a required argument: 'addend'"):

        @pl.vector_function(mode="simt", max_threads=1)
        def missing_addend(value: pl.DT_FP32):
            _test_result = pl.simt.fma(value, value)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                missing_addend[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    @pl.vector_function(mode="simt", max_threads=1)
    def keyword_argument(value: pl.DT_FP32):
        _test_result = pl.simt.exp(value=value)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            keyword_argument[1](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "simt.exp(" in str(program.get_function(keyword_argument.__name__))


def test_simt_operations_reject_argument_unpacking_at_common_entry():
    @pl.vector_function(mode="simt", max_threads=1)
    def positional_unpacking(value: pl.DT_FP32):
        _test_result = pl.simt.exp(*(value,))

    @pl.vector_function(mode="simt", max_threads=1)
    def keyword_unpacking(value: pl.DT_FP32):
        _test_result = pl.simt.exp(**{"value": value})

    for function in (positional_unpacking, keyword_unpacking):

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                function[1](value)

        with pytest.raises(InvalidArgument, match="does not support argument unpacking"):
            kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_batch1_scalar_math_is_available_in_simt_functions():
    @pl.vector_function(mode="simt", max_threads=1)
    def batch1_math(value: pl.DT_FP32):
        transformed = pl.simt.exp10(value) + pl.simt.log10(value)
        transformed = transformed + pl.simt.tan(value) + pl.simt.atan(value)
        transformed = transformed + pl.simt.expm1(value) + pl.simt.logb(value)
        transformed = transformed + pl.simt.cosh(value) + pl.simt.acos(value)
        transformed = transformed + pl.simt.sinh(value) + pl.simt.asin(value)
        _test_result = transformed + pl.simt.cbrt(value)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            batch1_math[1](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function(batch1_math.__name__))
    for name in (
        "exp10",
        "log10",
        "tan",
        "atan",
        "expm1",
        "logb",
        "cosh",
        "acos",
        "sinh",
        "asin",
        "cbrt",
    ):
        assert f"simt.{name}(" in function_ir


def test_batch1_scalar_math_supports_fp16_and_bf16():
    @pl.vector_function(mode="simt")
    def batch1_math(src, dst):
        value = src[0, 0]
        dst[0, 0] = pl.simt.exp10(value)
        dst[0, 1] = pl.simt.log10(value)
        dst[0, 2] = pl.simt.rcp(value)

    for dtype in (pl.DT_FP16, pl.DT_BF16):
        function_ir = str(_parse_tile_function(batch1_math, [([1, 16], dtype), ([1, 16], dtype)]))
        for name in ("exp10", "log10", "rcp"):
            assert f"simt.{name}(" in function_ir


def test_batch4_math_is_available_in_simt_functions():
    @pl.vector_function(mode="simt", max_threads=1)
    def batch4_math(value: pl.DT_FP32):
        _tanpi = pl.simt.tanpi(value)
        _atanh = pl.simt.atanh(value)
        _cospi = pl.simt.cospi(value)
        _acosh = pl.simt.acosh(value)
        _sinpi = pl.simt.sinpi(value)
        _asinh = pl.simt.asinh(value)
        _rcbrt = pl.simt.rcbrt(value)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            batch4_math[1](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function(batch4_math.__name__))
    for name in (
        "tanpi",
        "atanh",
        "cospi",
        "acosh",
        "sinpi",
        "asinh",
        "rcbrt",
    ):
        assert f"simt.{name}(" in function_ir


def test_selected_nan_binary_and_exponent_math_is_available_in_simt_functions():
    @pl.vector_function(mode="simt")
    def nan_ops(src, dst):
        lhs = src[0, 0]
        rhs = src[0, 1]
        dst[0, 0] = pl.simt.max_nan(lhs, rhs)
        dst[0, 1] = pl.simt.min_nan(lhs, rhs)

    for dtype in (pl.DT_FP16, pl.DT_BF16):
        function_ir = str(_parse_tile_function(nan_ops, [([1, 16], dtype), ([1, 16], dtype)]))
        assert "simt.max_nan(" in function_ir
        assert "simt.min_nan(" in function_ir

    @pl.vector_function(mode="simt", max_threads=1)
    def fp32_ops(lhs: pl.DT_FP32, rhs: pl.DT_FP32):
        _atan2 = pl.simt.atan2(lhs, rhs)
        _copysign = pl.simt.copysign(lhs, rhs)
        _nextafter = pl.simt.nextafter(lhs, rhs)
        _ilogb = pl.simt.ilogb(lhs)
        _signbit = pl.simt.signbit(lhs)

    @pl.jit(auto_mutex=False)
    def kernel(lhs: pl.DT_FP32, rhs: pl.DT_FP32):
        with pl.section_vector():
            fp32_ops[1](lhs, rhs)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function(fp32_ops.__name__))
    for name in ("atan2", "copysign", "nextafter", "ilogb", "signbit"):
        assert f"simt.{name}(" in function_ir


def test_new_scalar_math_rejects_mixed_binary_and_unsupported_unary_dtype():
    @pl.vector_function(mode="simt")
    def mixed_nextafter(lhs, rhs):
        _ = pl.simt.nextafter(lhs[0, 0], rhs[0, 0])

    with pytest.raises(InvalidType, match="same dtype"):
        _parse_tile_function(mixed_nextafter, [([1, 16], pl.DT_FP32), ([1, 16], pl.DT_FP16)])

    @pl.vector_function(mode="simt")
    def invalid_signbit(value):
        _ = pl.simt.signbit(value[0, 0])

    with pytest.raises(NotSupported, match="supports only fp32"):
        _parse_tile_function(invalid_signbit, [([1, 16], pl.DT_FP16)])
