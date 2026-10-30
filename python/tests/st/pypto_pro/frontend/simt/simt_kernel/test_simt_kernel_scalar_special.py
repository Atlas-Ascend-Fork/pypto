# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT kernel error, distribution, Gamma, and Bessel interfaces with FP32 operands."""

import math
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64


@pl.vector_function(mode="simt")
def _error_and_distribution(dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    dst[0, tid] = pl.simt.erf(value)
    dst[1, tid] = pl.simt.erfc(value)
    dst[2, tid] = pl.simt.erfinv(value)
    dst[3, tid] = pl.simt.erfcinv(value + 1.0)
    dst[4, tid] = pl.simt.erfcx(value)
    dst[5, tid] = pl.simt.normcdf(value)
    dst[6, tid] = pl.simt.normcdfinv((value + 1.0) * 0.5)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_error_and_distribution(
    dst: pl.Tensor[[7, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _error_and_distribution(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_error_and_distribution():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-0.875, 0.875, THREADS, dtype=torch.float32)
    dst = torch.empty((7, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_error_and_distribution[1, THREADS](dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    reference = values.double()
    shifted = (values + 1.0).double()
    probability = ((values + 1.0) * 0.5).double()
    expected = torch.stack(
        [
            torch.erf(reference),
            torch.erfc(reference),
            torch.erfinv(reference),
            torch.erfinv(1.0 - shifted),
            torch.special.erfcx(reference),
            0.5 * torch.erfc(-reference / math.sqrt(2.0)),
            math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0),
        ]
    ).float()
    torch.testing.assert_close(dst.cpu(), expected, rtol=5e-4, atol=5e-6)


@pl.vector_function(mode="simt")
def _gamma(dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    dst[0, tid] = pl.simt.lgamma(value)
    dst[1, tid] = pl.simt.tgamma(value)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_gamma(
    dst: pl.Tensor[[2, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _gamma(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_gamma():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.25, 5.0, THREADS, dtype=torch.float32)
    dst = torch.empty((2, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_gamma[1, THREADS](dst, values.to(ST_DEVICE))
    torch.npu.synchronize()

    expected = torch.tensor(
        [[math.lgamma(value) for value in values.tolist()], [math.gamma(value) for value in values.tolist()]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(dst.cpu(), expected, rtol=5e-4, atol=5e-6)


@pl.vector_function(mode="simt")
def _bessel(dst, src, orders):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    order = orders[tid]
    dst[0, tid] = pl.simt.cyl_bessel_i0(value)
    dst[1, tid] = pl.simt.cyl_bessel_i1(value)
    dst[2, tid] = pl.simt.j0(value)
    dst[3, tid] = pl.simt.j1(value)
    dst[4, tid] = pl.simt.jn(order, value)
    dst[5, tid] = pl.simt.y0(value)
    dst[6, tid] = pl.simt.y1(value)
    dst[7, tid] = pl.simt.yn(order, value)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_bessel(
    dst: pl.Tensor[[8, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
    orders: pl.Tensor[[THREADS], pl.DT_INT32],
):
    _bessel(dst, src, orders)


@pytest.mark.soc("950")
def test_simt_kernel_bessel():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(8.0, 16.0, THREADS, dtype=torch.float32)
    orders = torch.arange(THREADS, dtype=torch.int32) % 4
    dst = torch.empty((8, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_bessel[1, THREADS](dst, values.to(ST_DEVICE), orders.to(ST_DEVICE))
    torch.npu.synchronize()

    reference = values.double()
    bessel_j = [torch.special.bessel_j0(reference), torch.special.bessel_j1(reference)]
    bessel_y = [torch.special.bessel_y0(reference), torch.special.bessel_y1(reference)]
    for order in range(1, 3):
        bessel_j.append(2.0 * order / reference * bessel_j[-1] - bessel_j[-2])
        bessel_y.append(2.0 * order / reference * bessel_y[-1] - bessel_y[-2])
    indices = torch.arange(THREADS)
    expected = torch.stack(
        [
            torch.special.i0(reference),
            torch.special.i1(reference),
            bessel_j[0],
            bessel_j[1],
            torch.stack(bessel_j)[orders.long(), indices],
            bessel_y[0],
            bessel_y[1],
            torch.stack(bessel_y)[orders.long(), indices],
        ]
    ).float()
    torch.testing.assert_close(dst.cpu(), expected, rtol=5e-4, atol=5e-6)
