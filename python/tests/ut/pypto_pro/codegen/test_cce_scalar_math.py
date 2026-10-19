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

"""Direct CCE code-generation tests for A5 SIMT scalar math."""

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch


def _compile_to_cce(kernel) -> str:
    from pypto_pro.runtime.jit import _add_kernel_header, _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    return _add_kernel_header(_assemble_cv_source(cube, vector)).content


@pl.vector_function(mode="simt", max_threads=1)
def _fp32_math_intrinsics(
    out,
    flags,
    value: pl.DT_FP32,
):
    out[0, 0] = pl.simt.abs(value)
    out[0, 1] = pl.simt.min(value, value)
    out[0, 2] = pl.simt.max(value, value)
    out[0, 3] = pl.simt.sqrt(value)
    out[0, 4] = pl.simt.rsqrt(value)
    out[0, 5] = pl.simt.exp(value)
    out[0, 6] = pl.simt.exp2(value)
    out[0, 7] = pl.simt.log(value)
    out[0, 8] = pl.simt.log2(value)
    out[0, 9] = pl.simt.log1p(value)
    out[0, 10] = pl.simt.sin(value)
    out[0, 11] = pl.simt.cos(value)
    out[0, 12] = pl.simt.tanh(value)
    out[0, 13] = pl.simt.rint(value)
    out[0, 14] = pl.simt.round(value)
    out[0, 15] = pl.simt.floor(value)
    out[0, 16] = pl.simt.ceil(value)
    out[0, 17] = pl.simt.trunc(value)
    out[0, 18] = pl.simt.fma(value, value, value)
    flags[0, 0] = pl.simt.isnan(value)
    flags[0, 1] = pl.simt.isinf(value)


@pl.jit
def _fp32_math_codegen_kernel(value: pl.DT_FP32):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    flags_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    flags = pl.make_tile(flags_type, addr=0x0080)
    with pl.section_vector():
        _fp32_math_intrinsics[1](out, flags, value)


@pl.vector_function(mode="simt", max_threads=1)
def _fp16_math_intrinsics(
    out,
    flags,
    source,
):
    value = source[0, 0]
    out[0, 0] = pl.simt.abs(value)
    out[0, 1] = pl.simt.min(value, value)
    out[0, 2] = pl.simt.max(value, value)
    out[0, 3] = pl.simt.sqrt(value)
    out[0, 4] = pl.simt.rsqrt(value)
    out[0, 5] = pl.simt.exp(value)
    out[0, 6] = pl.simt.exp2(value)
    out[0, 7] = pl.simt.log(value)
    out[0, 8] = pl.simt.log2(value)
    out[0, 9] = pl.simt.sin(value)
    out[0, 10] = pl.simt.cos(value)
    out[0, 11] = pl.simt.tanh(value)
    out[0, 12] = pl.simt.rint(value)
    out[0, 13] = pl.simt.round(value)
    out[0, 14] = pl.simt.floor(value)
    out[0, 15] = pl.simt.ceil(value)
    out[0, 16] = pl.simt.trunc(value)
    out[0, 17] = pl.simt.fma(value, value, value)
    flags[0, 0] = pl.simt.isnan(value)
    flags[0, 1] = pl.simt.isinf(value)


@pl.jit
def _fp16_math_codegen_kernel(_jit_entry: pl.DT_INT64):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    flags_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    flags = pl.make_tile(flags_type, addr=0x0040)
    source = pl.make_tile(source_type, addr=0x0080)
    with pl.section_vector():
        _fp16_math_intrinsics[1](out, flags, source)


@pl.vector_function(mode="simt", max_threads=1)
def _bf16_math_intrinsics(
    out,
    flags,
    source,
):
    value = source[0, 0]
    out[0, 0] = pl.simt.abs(value)
    out[0, 1] = pl.simt.min(value, value)
    out[0, 2] = pl.simt.max(value, value)
    out[0, 3] = pl.simt.sqrt(value)
    out[0, 4] = pl.simt.rsqrt(value)
    out[0, 5] = pl.simt.exp(value)
    out[0, 6] = pl.simt.exp2(value)
    out[0, 7] = pl.simt.log(value)
    out[0, 8] = pl.simt.log2(value)
    out[0, 9] = pl.simt.sin(value)
    out[0, 10] = pl.simt.cos(value)
    out[0, 11] = pl.simt.tanh(value)
    out[0, 12] = pl.simt.rint(value)
    out[0, 13] = pl.simt.round(value)
    out[0, 14] = pl.simt.floor(value)
    out[0, 15] = pl.simt.ceil(value)
    out[0, 16] = pl.simt.trunc(value)
    out[0, 17] = pl.simt.fma(value, value, value)
    flags[0, 0] = pl.simt.isnan(value)
    flags[0, 1] = pl.simt.isinf(value)


@pl.jit
def _bf16_math_codegen_kernel(_jit_entry: pl.DT_INT64):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    flags_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 32], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    flags = pl.make_tile(flags_type, addr=0x0040)
    source = pl.make_tile(source_type, addr=0x0080)
    with pl.section_vector():
        _bf16_math_intrinsics[1](out, flags, source)


@pl.vector_function(mode="simt", max_threads=1)
def _int64_math_intrinsics(
    out,
    source,
):
    out[0, 0] = pl.simt.abs(source[0, 0])
    out[0, 1] = pl.simt.min(source[0, 0], source[0, 1])
    out[0, 2] = pl.simt.max(source[0, 0], source[0, 1])


@pl.jit
def _int64_math_codegen_kernel(_jit_entry: pl.DT_INT64):
    out_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    source = pl.make_tile(source_type, addr=0x0040)
    with pl.section_vector():
        _int64_math_intrinsics[1](out, source)


def test_fp32_scalar_math_codegen_maps_native_cce_intrinsics():
    cpp = _compile_to_cce(_fp32_math_codegen_kernel)

    for intrinsic in (
        "__fabsf(",
        "__fminf(",
        "__fmaxf(",
        "__sqrtf(",
        "__expf(",
        "__logf(",
        "__rintf(",
        "__roundf(",
        "__floorf(",
        "__ceilf(",
        "__isnan(",
        "__isinf(",
        "__fma(",
    ):
        assert intrinsic in cpp
    assert "1.0f / ({float __sqrt_x" in cpp
    assert "__exp2_exp_poly" in cpp
    assert "__log1p_reduced_poly" in cpp
    assert "__log2_log2_hi" in cpp
    assert "__tanh_abs_x >= 0.60000002384185791016f" in cpp
    assert "__fabsf(__t)" in cpp
    assert "float __t = __fma(" not in cpp
    assert "__t = __fma(__t, 0.0f, __t);" in cpp
    assert "simt_api/math_functions.h" not in cpp


def test_fp16_scalar_math_codegen_maps_cce_intrinsics_without_simt_headers():
    cpp = _compile_to_cce(_fp16_math_codegen_kernel)

    for intrinsic in (
        "__sqrtf(",
        "__expf(",
        "__logf(",
        "__rintf(",
        "__floorf(",
        "__ceilf(",
        "__isnan(",
        "__isinf(",
        "__fma(",
        "__hmin_nan(",
        "__hmax_nan(",
        "__cvt_float<",
        "__cvt_half<",
    ):
        assert intrinsic in cpp
    assert "1.0f / ({float __sqrt_x" in cpp
    assert "__cvt_half<ROUND::A," in cpp
    assert "simt_api/" not in cpp


def test_bf16_scalar_math_codegen_maps_cce_intrinsics_without_simt_headers():
    cpp = _compile_to_cce(_bf16_math_codegen_kernel)

    for intrinsic in (
        "__rintf(",
        "__floorf(",
        "__ceilf(",
        "__isnan(",
        "__isinf(",
        "__fma(",
        "__min(",
        "__max(",
        "__cvt_float<",
        "__cvt_bfloat16_t<",
    ):
        assert intrinsic in cpp
    assert "__cvt_bfloat16_t<ROUND::A," in cpp
    assert "simt_api/" not in cpp


def test_int64_scalar_math_codegen_maps_abs_min_max_and_header():
    cpp = _compile_to_cce(_int64_math_codegen_kernel)

    assert "= abs(" in cpp
    assert "min((int64_t)" in cpp
    assert "max((int64_t)" in cpp
    assert "simt_api/math_functions.h" not in cpp


@pl.vector_function(mode="simt", max_threads=1)
def _batch1_fp32_intrinsics(out, value: pl.DT_FP32):
    out[0, 0] = pl.simt.exp10(value)
    out[0, 1] = pl.simt.log10(value)
    out[0, 2] = pl.simt.tan(value)
    out[0, 3] = pl.simt.atan(value)
    out[0, 4] = pl.simt.expm1(value)
    out[0, 5] = pl.simt.logb(value)
    out[0, 6] = pl.simt.cosh(value)
    out[0, 7] = pl.simt.acos(value)
    out[0, 8] = pl.simt.sinh(value)
    out[0, 9] = pl.simt.asin(value)
    out[0, 10] = pl.simt.cbrt(value)


@pl.jit
def _batch1_fp32_codegen_kernel(value: pl.DT_FP32):
    out_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(out_type, addr=0x0000)
    with pl.section_vector():
        _batch1_fp32_intrinsics[1](out, value)


@pl.vector_function(mode="simt", max_threads=1)
def _batch1_fp16_intrinsics(out, source):
    value = source[0, 0]
    out[0, 0] = pl.simt.exp10(value)
    out[0, 1] = pl.simt.log10(value)
    out[0, 2] = pl.simt.rcp(value)


@pl.jit
def _batch1_fp16_codegen_kernel(_jit_entry: pl.DT_INT64):
    tile_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(tile_type, addr=0x0000)
    source = pl.make_tile(source_type, addr=0x0040)
    with pl.section_vector():
        _batch1_fp16_intrinsics[1](out, source)


@pl.vector_function(mode="simt", max_threads=1)
def _batch1_bf16_intrinsics(out, source):
    value = source[0, 0]
    out[0, 0] = pl.simt.exp10(value)
    out[0, 1] = pl.simt.log10(value)
    out[0, 2] = pl.simt.rcp(value)


@pl.jit
def _batch1_bf16_codegen_kernel(_jit_entry: pl.DT_INT64):
    tile_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    source_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    out = pl.make_tile(tile_type, addr=0x0000)
    source = pl.make_tile(source_type, addr=0x0040)
    with pl.section_vector():
        _batch1_bf16_intrinsics[1](out, source)


@pl.vector_function(mode="simt", max_threads=1)
def _nan_intrinsics(out_fp16, source_fp16, out_bf16, source_bf16):
    out_fp16[0, 0] = pl.simt.max_nan(source_fp16[0, 0], source_fp16[0, 1])
    out_fp16[0, 1] = pl.simt.min_nan(source_fp16[0, 0], source_fp16[0, 1])
    out_bf16[0, 0] = pl.simt.max_nan(source_bf16[0, 0], source_bf16[0, 1])
    out_bf16[0, 1] = pl.simt.min_nan(source_bf16[0, 0], source_bf16[0, 1])


@pl.jit
def _nan_intrinsics_codegen_kernel(_jit_entry: pl.DT_INT64):
    fp16_type = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    bf16_type = pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    out_fp16 = pl.make_tile(fp16_type, addr=0x0000)
    source_fp16 = pl.make_tile(fp16_type, addr=0x0040)
    out_bf16 = pl.make_tile(bf16_type, addr=0x0080)
    source_bf16 = pl.make_tile(bf16_type, addr=0x00C0)
    with pl.section_vector():
        _nan_intrinsics[1](out_fp16, source_fp16, out_bf16, source_bf16)


@pl.vector_function(mode="simt", max_threads=1)
def _batch4_math_intrinsics(out_fp32, value: pl.DT_FP32):
    out_fp32[0, 0] = pl.simt.tanpi(value)
    out_fp32[0, 1] = pl.simt.atanh(value)
    out_fp32[0, 2] = pl.simt.cospi(value)
    out_fp32[0, 3] = pl.simt.acosh(value)
    out_fp32[0, 4] = pl.simt.sinpi(value)
    out_fp32[0, 5] = pl.simt.asinh(value)
    out_fp32[0, 6] = pl.simt.rcbrt(value)


@pl.jit
def _batch4_math_codegen_kernel(value: pl.DT_FP32):
    out_fp32 = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    with pl.section_vector():
        _batch4_math_intrinsics[1](out_fp32, value)


def test_batch1_fp32_scalar_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_batch1_fp32_codegen_kernel)

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp
    for expression in (
        "0.0131822545081377029418945f",
        "__logf(10.0f)",
        "252.898206f",
        "0.00245002890005707741f",
        "0.00138624827377498149871826f",
        "8388608.0f",
        "12583037.0f",
        "0.03538220748305320740f",
        "0.00000281695110970758826f",
        "0.05025001987814903259f",
        "-0.6666666865348815918f",
    ):
        assert expression in cpp


def test_batch1_fp16_scalar_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_batch1_fp16_codegen_kernel)

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp
    for expression in (
        "0.30419921875f",
        "0.2362060546875f",
    ):
        assert expression in cpp
    assert "static_cast<half>(1.0f) / (" in cpp


def test_batch1_bf16_scalar_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_batch1_bf16_codegen_kernel)

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp
    for expression in (
        "0.0131822545081377029418945f",
        "__logf(10.0f)",
    ):
        assert expression in cpp
    assert "__cvt_float<" in cpp
    assert "__cvt_bfloat16_t<" in cpp
    assert "static_cast<bfloat16_t>(1.0f) / (" in cpp


def test_nan_select_codegen_uses_low_precision_intrinsics():
    cpp = _compile_to_cce(_nan_intrinsics_codegen_kernel)

    assert cpp.count("__hmax_nan(") == 2
    assert cpp.count("__hmin_nan(") == 2
    assert "__cvt_float<" not in cpp
    assert "simt_api/" not in cpp


def test_batch4_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_batch4_math_codegen_kernel)

    assert "pypto_simt_" not in cpp
    assert "simt_api/" not in cpp
    for expression in (
        "-8.7422776573475857731e-08f",
        "8.50705917302346158658e+37f",
        "__cospi_s * (__cospi_r * __cospi_z)",
        "0.000045124618889065459371f",
        "__sinpi_truncated_x == __sinpi_x",
        "-0.01396484375f",
        "__rcbrt_correction",
    ):
        assert expression in cpp


@pl.vector_function(mode="simt", max_threads=1)
def _bessel_rounding_and_saturate_intrinsics(out_fp32, out_i64, value: pl.DT_FP32, order: pl.DT_INT32):
    out_fp32[0, 0] = pl.simt.jn(order, value)
    out_fp32[0, 1] = pl.simt.yn(order, value)
    out_fp32[0, 2] = pl.simt.cyl_bessel_i0(value)
    out_fp32[0, 3] = pl.simt.cyl_bessel_i1(value)
    out_fp32[0, 4] = pl.simt.j0(value)
    out_fp32[0, 5] = pl.simt.j1(value)
    out_fp32[0, 6] = pl.simt.y0(value)
    out_fp32[0, 7] = pl.simt.y1(value)
    out_fp32[0, 8] = pl.simt.saturate(value)
    out_i64[0, 0] = pl.simt.rint(value, dtype=pl.DT_INT64)
    out_i64[0, 1] = pl.simt.round(value, dtype=pl.DT_INT64)


@pl.jit
def _bessel_rounding_and_saturate_codegen_kernel(value: pl.DT_FP32, order: pl.DT_INT32):
    out_fp32 = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    out_i64 = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    with pl.section_vector():
        _bessel_rounding_and_saturate_intrinsics[1](out_fp32, out_i64, value, order)


def test_bessel_rounding_and_saturate_codegen_is_self_contained():
    cpp = _compile_to_cce(_bessel_rounding_and_saturate_codegen_kernel)

    assert "simt_api/" not in cpp
    for marker in (
        "__cvt_int64_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(",
        "__cvt_int64_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(",
        "1.551427e-19f",
        "__a >= 9.0f",
        "__a >= 8.085f",
        "2.4048254f",
        "5.42968082f",
        "1.139649510383606f",
        "(float)__n < __ax * 0.1f",
        "(float)__n > __ax * __ax / 10.0f",
        "__prefix <= -87.3365478515625f",
        "__half_prefix * __half_prefix",
        "__x > (float)__n * 10.0f",
        "__saturatef(",
    ):
        assert marker in cpp
    for wrapper in ("j0f(", "j1f(", "y0f(", "y1f(", "jnf(", "ynf("):
        assert wrapper not in cpp


@pl.vector_function(mode="simt", max_threads=1)
def _batch2_integer_intrinsics(
    signed_out,
    counts,
    unsigned_out,
    unsigned_wide,
    signed_source,
    unsigned_source,
    signed64_source,
    unsigned64_source,
):
    signed_lhs = signed_source[0, 0]
    signed_rhs = signed_source[0, 1]
    unsigned_lhs = unsigned_source[0, 0]
    unsigned_rhs = unsigned_source[0, 1]
    selector = unsigned_source[0, 2]
    unsigned_out[0, 0] = pl.simt.brev(unsigned_lhs)
    unsigned_wide[0, 0] = pl.simt.brev(unsigned64_source[0, 0])
    counts[0, 0] = pl.simt.clz(signed_lhs)
    counts[0, 1] = pl.simt.clz(signed64_source[0, 0])
    counts[0, 2] = pl.simt.ffs(signed_lhs)
    counts[0, 3] = pl.simt.ffs(signed64_source[0, 0])
    unsigned_out[0, 1] = pl.simt.byte_perm(unsigned_lhs, unsigned_rhs, selector)
    unsigned_out[0, 2] = pl.simt.sad(signed_lhs, signed_rhs, selector)
    unsigned_out[0, 3] = pl.simt.sad(unsigned_lhs, unsigned_rhs, selector)
    signed_out[0, 0] = pl.simt.mul24(signed_lhs, signed_rhs)
    unsigned_out[0, 4] = pl.simt.mul24(unsigned_lhs, unsigned_rhs)
    signed_out[0, 1] = pl.simt.hadd(signed_lhs, signed_rhs)
    unsigned_out[0, 5] = pl.simt.hadd(unsigned_lhs, unsigned_rhs)
    signed_out[0, 2] = pl.simt.rhadd(signed_lhs, signed_rhs)
    unsigned_out[0, 6] = pl.simt.rhadd(unsigned_lhs, unsigned_rhs)


@pl.jit
def _batch2_integer_codegen_kernel(_jit_entry: pl.DT_INT64):
    signed_out = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    counts = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    unsigned_out = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x00C0,
    )
    unsigned_wide = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0100,
    )
    signed_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0140,
    )
    unsigned_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0180,
    )
    signed64_source = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addr=0x01C0,
    )
    unsigned64_source = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0200,
    )
    with pl.section_vector():
        _batch2_integer_intrinsics[1](
            signed_out,
            counts,
            unsigned_out,
            unsigned_wide,
            signed_source,
            unsigned_source,
            signed64_source,
            unsigned64_source,
        )


@pl.vector_function(mode="simt", max_threads=1)
def _batch3_low_precision_intrinsics(out, flags, source):
    lhs = source[0, 0]
    rhs = source[0, 1]
    out[0, 0] = pl.simt.fma_relu(lhs, rhs, lhs)
    flags[0, 0] = pl.simt.equ(lhs, rhs)
    flags[0, 1] = pl.simt.leu(lhs, rhs)
    flags[0, 2] = pl.simt.geu(lhs, rhs)
    flags[0, 3] = pl.simt.ltu(lhs, rhs)
    flags[0, 4] = pl.simt.gtu(lhs, rhs)


@pl.jit
def _batch3_fp16_codegen_kernel(_jit_entry: pl.DT_INT64):
    out = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    flags = pl.make_tile(
        pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    source = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    with pl.section_vector():
        _batch3_low_precision_intrinsics[1](out, flags, source)


@pl.jit
def _batch3_bf16_codegen_kernel(_jit_entry: pl.DT_INT64):
    out = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    flags = pl.make_tile(
        pl.TileType(shape=[1, 32], dtype=pl.DT_BOOL, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    source = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    with pl.section_vector():
        _batch3_low_precision_intrinsics[1](out, flags, source)


@pl.vector_function(mode="simt", max_threads=1)
def _batch3_fp32_intrinsics(out, lhs: pl.DT_FP32, rhs: pl.DT_FP32):
    out[0, 0] = pl.simt.hypot(lhs, rhs)
    out[0, 1] = pl.simt.rhypot(lhs, rhs)
    out[0, 2] = pl.simt.fdim(lhs, rhs)
    out[0, 3] = pl.simt.remainder(lhs, rhs)
    out[0, 4] = pl.simt.pow(lhs, rhs)


@pl.jit
def _batch3_fp32_codegen_kernel(lhs: pl.DT_FP32, rhs: pl.DT_FP32):
    out = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    with pl.section_vector():
        _batch3_fp32_intrinsics[1](out, lhs, rhs)


@pl.vector_function(mode="simt", max_threads=1)
def _recognized_special_math_intrinsics(
    out_fp32,
    value: pl.DT_FP32,
    order: pl.DT_INT32,
    long_exponent: pl.DT_INT64,
):
    out_fp32[0, 0] = pl.simt.ldexp(value, order)
    out_fp32[0, 1] = pl.simt.scalbn(value, order)
    out_fp32[0, 2] = pl.simt.scalbn(value, long_exponent)
    out_fp32[0, 3] = pl.simt.norm3d(value, value, value)
    out_fp32[0, 4] = pl.simt.rnorm3d(value, value, value)
    out_fp32[0, 5] = pl.simt.norm4d(value, value, value, value)
    out_fp32[0, 6] = pl.simt.rnorm4d(value, value, value, value)
    out_fp32[0, 7] = pl.simt.erf(value)
    out_fp32[0, 8] = pl.simt.erfc(value)
    out_fp32[0, 9] = pl.simt.erfinv(value)
    out_fp32[0, 10] = pl.simt.erfcinv(value)
    out_fp32[0, 11] = pl.simt.erfcx(value)
    out_fp32[0, 12] = pl.simt.tgamma(value)
    out_fp32[0, 13] = pl.simt.lgamma(value)
    out_fp32[0, 14] = pl.simt.normcdf(value)
    out_fp32[0, 15] = pl.simt.normcdfinv(value)


@pl.jit
def _recognized_special_math_codegen_kernel(
    value: pl.DT_FP32,
    order: pl.DT_INT32,
    long_exponent: pl.DT_INT64,
):
    out_fp32 = pl.make_tile(
        pl.TileType(shape=[1, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    with pl.section_vector():
        _recognized_special_math_intrinsics[1](out_fp32, value, order, long_exponent)


def test_recognized_special_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_recognized_special_math_codegen_kernel)

    assert "simt_api/" not in cpp
    for marker in (
        "int64_t __long_n",
        "__long_n > 278",
        "0xfe000000u",
        "0.0008912170887924731f",
        "10.055000305175781f",
        "__powf(2.0f, __exponent_arg)",
        "1.9966f",
        "-63.113224f",
        "41.099998474121094f",
        "__gamma_sinpi",
        "0.045882664f",
    ):
        assert marker in cpp
    for wrapper in (
        "erff(",
        "erfcf(",
        "erfinvf(",
        "erfcinvf(",
        "erfcxf(",
        "tgammaf(",
        "lgammaf(",
        "normcdff(",
        "normcdfinvf(",
    ):
        assert wrapper not in cpp


@pl.vector_function(mode="simt", max_threads=1)
def _mul_i32_to_i64_intrinsics(signed_out, unsigned_out, signed_source, unsigned_source):
    signed_out[0, 0] = pl.simt.mul_i32_to_i64(signed_source[0, 0], signed_source[0, 1])
    unsigned_out[0, 0] = pl.simt.mul_i32_to_i64(unsigned_source[0, 0], unsigned_source[0, 1])


@pl.jit
def _mul_i32_to_i64_codegen_kernel(_jit_entry: pl.DT_INT64):
    signed_out = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0000,
    )
    unsigned_out = pl.make_tile(
        pl.TileType(shape=[1, 4], dtype=pl.DT_UINT64, target_memory=pl.MemorySpace.Vec),
        addr=0x0040,
    )
    signed_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0080,
    )
    unsigned_source = pl.make_tile(
        pl.TileType(shape=[1, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x00C0,
    )
    with pl.section_vector():
        _mul_i32_to_i64_intrinsics[1](signed_out, unsigned_out, signed_source, unsigned_source)


def test_mul_i32_to_i64_codegen_uses_cce_intrinsic():
    cpp = _compile_to_cce(_mul_i32_to_i64_codegen_kernel)

    assert "__mul_i32toi64((int)(" in cpp
    assert "__mul_i32toi64((unsigned int)(" in cpp


def test_batch2_integer_math_codegen_uses_cce_intrinsics():
    cpp = _compile_to_cce(_batch2_integer_codegen_kernel)

    assert "simt_api/" not in cpp
    for intrinsic in (
        "__brev",
        "__clz",
        "__ffs",
        "__byte_perm",
        "__sad",
        "__usad",
        "__mul24",
        "__umul24",
        "__hadd",
        "__uhadd",
        "__rhadd",
        "__urhadd",
    ):
        assert f"{intrinsic}(" in cpp


def test_batch3_fp16_math_codegen_uses_cce_intrinsics():
    cpp = _compile_to_cce(_batch3_fp16_codegen_kernel)

    assert "simt_api/" not in cpp
    for intrinsic in ("__hfma_relu", "__hequ", "__hleu", "__hgeu", "__hltu", "__hgtu"):
        assert f"{intrinsic}(" in cpp


def test_batch3_bf16_math_codegen_uses_cce_intrinsics():
    cpp = _compile_to_cce(_batch3_bf16_codegen_kernel)

    assert "simt_api/" not in cpp
    for intrinsic in ("__hfma_relu", "__hequ", "__hleu", "__hgeu", "__hltu", "__hgtu"):
        assert f"{intrinsic}(" in cpp


def test_batch3_fp32_binary_math_codegen_is_self_contained():
    cpp = _compile_to_cce(_batch3_fp32_codegen_kernel)

    assert "simt_api/" not in cpp
    for local_code in (
        "0x7e800000u",
        "1.0f / __sqrtf(__square_sum)",
        "__min_bits == 0x7f800000u",
        "__x > __y ? __x - __y : 0.0f",
        "__result = 0x7fffffffu",
        "bool __quotient_odd = false",
        "__twice == __my && __quotient_odd",
        "__product_lo = __fma(__log2_lo - (__log2_abs_x - __log2_hi), __y, __product_lo)",
        "(__product_hi - __rounded) + __product_lo",
        "__half_y > 0.0f ? __floorf(__half_y) : __ceilf(__half_y)",
        "if (__y == 0.0f || __x == 1.0f)",
        "__x_is_zero_or_inf && __isnan(__y)",
        "__y_is_odd_int ? __special : __fabsf(__special)",
        "__y_is_inf && __x == -1.0f",
    ):
        assert local_code in cpp
    assert "? -1.5707963267948966f : 1.5707963267948966f" not in cpp
    for wrapper in ("hypotf", "rhypotf", "fdimf", "remainderf", "powf"):
        assert f"{wrapper}(" not in cpp
