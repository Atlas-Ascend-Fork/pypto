# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT kernel scalar arithmetic, elementary math, and classification interfaces."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64


@pl.vector_function(mode="simt")
def _sqrt(value):
    return pl.simt.sqrt(value)


@pl.vector_function(mode="simt")
def _float_math(dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    other = src[THREADS - 1 - tid]
    dst[0, tid] = pl.simt.abs(-value)
    dst[1, tid] = pl.simt.min(value, other)
    dst[2, tid] = pl.simt.max(value, other)
    dst[3, tid] = _sqrt(value)
    dst[4, tid] = pl.simt.rsqrt(value)
    dst[5, tid] = pl.simt.exp(value)
    dst[6, tid] = pl.simt.exp2(value)
    dst[7, tid] = pl.simt.log(value)
    dst[8, tid] = pl.simt.log2(value)
    dst[9, tid] = pl.simt.sin(value)
    dst[10, tid] = pl.simt.cos(value)
    dst[11, tid] = pl.simt.tanh(value)
    dst[12, tid] = pl.simt.fma(value, other, value)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_float_math(
    dst: pl.Tensor[[13, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _float_math(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_float_math():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.25, 4.0, THREADS, dtype=torch.float32)
    dst = torch.empty((13, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_float_math[1, THREADS](dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    reference = values.double()
    other = reference.flip(0)
    expected = torch.stack(
        [
            reference.abs(),
            torch.minimum(reference, other),
            torch.maximum(reference, other),
            reference.sqrt(),
            reference.rsqrt(),
            reference.exp(),
            reference.exp2(),
            reference.log(),
            reference.log2(),
            reference.sin(),
            reference.cos(),
            reference.tanh(),
            reference * other + reference,
        ]
    ).float()
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-5, atol=1e-6)


@pl.vector_function(mode="simt")
def _rounding(dst, integer_dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    dst[0, tid] = pl.simt.rint(value)
    dst[1, tid] = pl.simt.round(value)
    dst[2, tid] = pl.simt.floor(value)
    dst[3, tid] = pl.simt.ceil(value)
    dst[4, tid] = pl.simt.trunc(value)
    integer_dst[0, tid] = pl.simt.rint(value, dtype=pl.DT_INT64)
    integer_dst[1, tid] = pl.simt.round(value, dtype=pl.DT_INT64)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_rounding(
    dst: pl.Tensor[[5, THREADS], pl.DT_FP32],
    integer_dst: pl.Tensor[[2, THREADS], pl.DT_INT64],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _rounding(dst, integer_dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_rounding():
    torch.npu.set_device(ST_DEVICE)
    values = torch.tensor([-3.75, -2.5, -1.5, -0.0, 0.25, 1.5, 2.5, 3.75] * (THREADS // 8), dtype=torch.float32)
    dst = torch.empty((5, THREADS), dtype=torch.float32, device=ST_DEVICE)
    integer_dst = torch.empty((2, THREADS), dtype=torch.int64, device=ST_DEVICE)
    simt_kernel_rounding[1, THREADS](dst, integer_dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    reference = values.float()
    away = reference.sign() * (reference.abs() + 0.5).floor()
    expected = torch.stack([reference.round(), away, reference.floor(), reference.ceil(), reference.trunc()])
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(integer_dst.cpu(), expected[:2].to(torch.int64))


@pl.vector_function(mode="simt")
def _float_classification(dst, src):
    tid = pl.simt.linear_thread_idx()
    dst[0, tid] = pl.simt.isnan(src[tid])
    dst[1, tid] = pl.simt.isinf(src[tid])
    dst[2, tid] = pl.simt.isfinite(src[tid])


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_float_classification(
    dst: pl.Tensor[[3, THREADS], pl.DT_BOOL],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _float_classification(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_float_classification():
    torch.npu.set_device(ST_DEVICE)
    values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, -1.0, 1.0, 65504.0] * (THREADS // 8),
        dtype=torch.float32,
    )
    dst = torch.empty((3, THREADS), dtype=torch.bool, device=ST_DEVICE)
    simt_kernel_float_classification[1, THREADS](dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    assert torch.equal(dst.cpu(), torch.stack([torch.isnan(values), torch.isinf(values), torch.isfinite(values)]))


@pl.vector_function(mode="simt")
def _integer_minmax(dst, lhs, rhs):
    tid = pl.simt.linear_thread_idx()
    dst[0, tid] = pl.simt.min(lhs[tid], rhs[tid])
    dst[1, tid] = pl.simt.max(lhs[tid], rhs[tid])


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_integer_minmax(
    dst: pl.Tensor[[2, THREADS], pl.DT_INT32],
    lhs: pl.Tensor[[THREADS], pl.DT_INT32],
    rhs: pl.Tensor[[THREADS], pl.DT_INT32],
):
    _integer_minmax(dst, lhs, rhs)


@pytest.mark.soc("950")
def test_simt_kernel_integer_minmax():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.arange(THREADS, dtype=torch.int32) - THREADS // 2
    rhs = lhs.flip(0)
    expected = torch.stack([torch.minimum(lhs, rhs), torch.maximum(lhs, rhs)])
    dst = torch.empty((2, THREADS), dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_integer_minmax[1, THREADS](dst, lhs.to(ST_DEVICE), rhs.to(ST_DEVICE))
    torch.npu.synchronize()

    assert torch.equal(dst.cpu(), expected)


@pl.vector_function(mode="simt")
def _powers_and_scaling(dst, integer_dst, src, exponents):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    other = src[THREADS - 1 - tid]
    exponent = exponents[tid]
    dst[0, tid] = pl.simt.exp10(value)
    dst[1, tid] = pl.simt.log10(value)
    dst[2, tid] = pl.simt.log1p(value)
    dst[3, tid] = pl.simt.expm1(value)
    dst[4, tid] = pl.simt.logb(value)
    dst[5, tid] = pl.simt.cbrt(value)
    dst[6, tid] = pl.simt.rcbrt(value)
    dst[7, tid] = pl.simt.pow(value, other)
    dst[8, tid] = pl.simt.ldexp(value, exponent)
    dst[9, tid] = pl.simt.scalbn(value, exponent)
    integer_dst[tid] = pl.simt.ilogb(value)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_powers_and_scaling(
    dst: pl.Tensor[[10, THREADS], pl.DT_FP32],
    integer_dst: pl.Tensor[[THREADS], pl.DT_INT32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
    exponents: pl.Tensor[[THREADS], pl.DT_INT32],
):
    _powers_and_scaling(dst, integer_dst, src, exponents)


@pytest.mark.soc("950")
def test_simt_kernel_powers_and_scaling():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.25, 4.0, THREADS, dtype=torch.float32)
    exponents = torch.arange(THREADS, dtype=torch.int32) % 17 - 8
    dst = torch.empty((10, THREADS), dtype=torch.float32, device=ST_DEVICE)
    integer_dst = torch.empty(THREADS, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_powers_and_scaling[1, THREADS](dst, integer_dst, values.to(ST_DEVICE), exponents.to(ST_DEVICE))
    torch.npu.synchronize()

    reference = values.double()
    exponent = torch.frexp(reference).exponent - 1
    expected = torch.stack(
        [
            torch.pow(10.0, reference),
            reference.log10(),
            reference.log1p(),
            reference.expm1(),
            exponent.double(),
            reference.pow(1.0 / 3.0),
            reference.pow(-1.0 / 3.0),
            reference.pow(reference.flip(0)),
            torch.ldexp(reference, exponents),
            torch.ldexp(reference, exponents),
        ]
    ).float()
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(integer_dst.cpu(), exponent)


@pl.vector_function(mode="simt")
def _trigonometric(dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    other = src[(tid + 17) % THREADS]
    dst[0, tid] = pl.simt.tan(value)
    dst[1, tid] = pl.simt.atan(value)
    dst[2, tid] = pl.simt.acos(value)
    dst[3, tid] = pl.simt.asin(value)
    dst[4, tid] = pl.simt.atan2(value, other)
    dst[5, tid] = pl.simt.tanpi(value)
    dst[6, tid] = pl.simt.cospi(value)
    dst[7, tid] = pl.simt.sinpi(value)
    dst[8, tid] = pl.simt.cosh(value)
    dst[9, tid] = pl.simt.sinh(value)
    dst[10, tid] = pl.simt.atanh(value)
    dst[11, tid] = pl.simt.acosh(value + 2.0)
    dst[12, tid] = pl.simt.asinh(value)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_trigonometric(
    dst: pl.Tensor[[13, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _trigonometric(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_trigonometric():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-0.375, 0.375, THREADS, dtype=torch.float32)
    dst = torch.empty((13, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_trigonometric[1, THREADS](dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    reference = values.double()
    expected = torch.stack(
        [
            reference.tan(),
            reference.atan(),
            reference.acos(),
            reference.asin(),
            torch.atan2(reference, reference.roll(-17)),
            (math.pi * reference).tan(),
            (math.pi * reference).cos(),
            (math.pi * reference).sin(),
            reference.cosh(),
            reference.sinh(),
            reference.atanh(),
            (values + 2.0).double().acosh(),
            reference.asinh(),
        ]
    ).float()
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-5, atol=1e-6)


@pl.vector_function(mode="simt")
def _float_utilities(dst, signs, lhs, rhs):
    tid = pl.simt.linear_thread_idx()
    x = lhs[tid]
    y = rhs[tid]
    dst[0, tid] = pl.simt.copysign(x, y)
    dst[1, tid] = pl.simt.nextafter(x, y)
    dst[2, tid] = pl.simt.fmod(x, y)
    dst[3, tid] = pl.simt.remainder(x, y)
    dst[4, tid] = pl.simt.fdim(x, y)
    dst[5, tid] = pl.simt.saturate(x)
    signs[tid] = pl.simt.signbit(x)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_float_utilities(
    dst: pl.Tensor[[6, THREADS], pl.DT_FP32],
    signs: pl.Tensor[[THREADS], pl.DT_INT32],
    lhs: pl.Tensor[[THREADS], pl.DT_FP32],
    rhs: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _float_utilities(dst, signs, lhs, rhs)


@pytest.mark.soc("950")
def test_simt_kernel_float_utilities():
    torch.npu.set_device(ST_DEVICE)
    lhs = torch.tensor([-5.5, -2.5, -1.0, -0.0, 0.0, 0.25, 1.5, 5.5] * (THREADS // 8), dtype=torch.float32)
    rhs = torch.tensor([2.0, -2.0, 0.5, -3.0, 3.0, -0.5, 2.0, -2.0] * (THREADS // 8), dtype=torch.float32)
    dst = torch.empty((6, THREADS), dtype=torch.float32, device=ST_DEVICE)
    signs = torch.empty(THREADS, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_float_utilities[1, THREADS](dst, signs, lhs.to(ST_DEVICE), rhs.to(ST_DEVICE))
    torch.npu.synchronize()

    expected = torch.stack(
        [
            torch.copysign(lhs, rhs),
            torch.nextafter(lhs, rhs),
            torch.fmod(lhs, rhs),
            torch.tensor([math.remainder(x, y) for x, y in zip(lhs.tolist(), rhs.tolist())], dtype=torch.float32),
            (lhs - rhs).clamp_min(0.0),
            lhs.clamp(0.0, 1.0),
        ]
    )
    actual = dst.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual[:4].view(torch.int32), expected[:4].view(torch.int32))
    assert torch.equal(signs.cpu(), torch.signbit(lhs).to(torch.int32))


@pl.vector_function(mode="simt")
def _norms(dst, src):
    tid = pl.simt.linear_thread_idx()
    x = src[0, tid]
    y = src[1, tid]
    z = src[2, tid]
    w = src[3, tid]
    dst[0, tid] = pl.simt.hypot(x, y)
    dst[1, tid] = pl.simt.rhypot(x, y)
    dst[2, tid] = pl.simt.norm3d(x, y, z)
    dst[3, tid] = pl.simt.rnorm3d(x, y, z)
    dst[4, tid] = pl.simt.norm4d(x, y, z, w)
    dst[5, tid] = pl.simt.rnorm4d(x, y, z, w)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_norms(
    dst: pl.Tensor[[6, THREADS], pl.DT_FP32],
    src: pl.Tensor[[4, THREADS], pl.DT_FP32],
):
    _norms(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_norms():
    torch.npu.set_device(ST_DEVICE)
    values = torch.stack(
        [
            torch.linspace(-8.0, 8.0, THREADS),
            torch.linspace(0.25, 16.0, THREADS),
            torch.linspace(-32.0, -0.5, THREADS),
            torch.linspace(64.0, 1.0, THREADS),
        ]
    ).float()
    dst = torch.empty((6, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_norms[1, THREADS](dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    x, y, z, w = values.double()
    norm2 = torch.hypot(x, y)
    norm3 = torch.hypot(norm2, z)
    norm4 = torch.hypot(norm2, torch.hypot(z, w))
    expected = torch.stack([norm2, 1.0 / norm2, norm3, 1.0 / norm3, norm4, 1.0 / norm4]).float()
    torch.testing.assert_close(dst.cpu(), expected, rtol=1e-5, atol=1e-6)
