# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 system tests for SIMT scalar Bessel interfaces."""

import logging
import math
import os

import pypto_pro.language as pl
import pytest
import torch

ELEMENTS = 64

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# --------------------------------------------------- cyl_bessel_i0 ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cyl_bessel_i0_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.cyl_bessel_i0(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_cyl_bessel_i0_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        cyl_bessel_i0_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_cyl_bessel_i0_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-12.0, 12.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -9.0, -8.999, -0.0, 8.999, 9.0, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_cyl_bessel_i0_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.special.i0(source)
    expected[0, [0, 6]] = float("inf")
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0005, atol=5e-06, equal_nan=True)

# --------------------------------------------------- cyl_bessel_i1 ---------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def cyl_bessel_i1_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.cyl_bessel_i1(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_cyl_bessel_i1_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        cyl_bessel_i1_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_cyl_bessel_i1_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-12.0, 12.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -8.085, -8.084, -0.0, 8.084, 8.085, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_cyl_bessel_i1_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.special.i1(source)
    expected[0, 0] = float("-inf")
    expected[0, 6] = float("inf")
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0005, atol=5e-06, equal_nan=True)

# -------------------------------------------------------- j0 ---------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def j0_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.j0(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_j0_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        j0_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_j0_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-32.0, 32.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -8.0, -0.0, 0.0, 2.4048254, 8.0, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_j0_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.special.bessel_j0(source)
    expected[0, [0, 6]] = 0.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0005, atol=5e-06, equal_nan=True)


def _check_bessel_fp32_tile_accuracy(kernel, input_value, reference):
    torch.npu.set_device(ST_DEVICE)
    source = torch.full((1, ELEMENTS), input_value, dtype=torch.float32)
    output = torch.empty_like(source, device=ST_DEVICE)
    kernel[1](source.to(ST_DEVICE), output)
    torch.npu.synchronize()

    actual = output.cpu().to(torch.float64)
    expected = torch.full_like(actual, reference)
    relative_error = (actual - expected).abs() / expected.abs()
    logging.info(
        "%s(fp32_input=%.17g): actual=%.9g reference=%.17g max_relative_error=%.9g",
        kernel.__name__, source[0, 0].item(), actual[0, 0].item(), reference, relative_error.max().item(),
    )
    # The usual 5e-6 absolute tolerance would hide relative errors in the decaying tail.
    torch.testing.assert_close(actual, expected, rtol=0.0005, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("input_value", "reference"),
    [
        (8.0, 0.17165080713755390609),
        (9.0, -9.03336111828761343360e-2),
        (16.0, -1.74899073983629184828e-1),
        (32.0, 1.38079009746555923759e-1),
        (105614.9921875, 2.42939547004085426370e-3),
        (105615.0, 2.42655054895566530700e-3),
        (105615.0078125, 2.42355752394779203569e-3),
        (1e6, 3.31043013739873740988e-4),
        (1e9, 2.46874718862691951144e-5),
        (9.9e12, -8.41918629080968205264e-8),
        (1e13, 2.17024994126219814694e-7),
        (10000000876544.0, 1.62297538219433337413e-7),
        (1e20, 7.95978945272244524996e-11),
        (3.4028234663852886e38, 1.01279820264082199891e-20),
    ],
    ids=[
        "8", "9", "16", "32", "below_reduction_cutoff", "reduction_cutoff", "above_reduction_cutoff",
        "1e6", "1e9", "9.9e12", "1e13", "above_1e13", "1e20", "fp32_max",
    ],
)
def test_j0_fp32_huge_arguments_with_tile_operands(input_value, reference):
    # References use the exact FP32 inputs, including 9900000280576 and 9999999827968.
    # High-precision references: J0 power series up to 32, https://dlmf.nist.gov/10.17.E3 otherwise.
    _check_bessel_fp32_tile_accuracy(simt_j0_tile, input_value, reference)

# -------------------------------------------------------- j1 ---------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def j1_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.j1(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_j1_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        j1_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_j1_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(-32.0, 32.0, ELEMENTS)
    values[:8] = torch.tensor([float("-inf"), -8.0, -0.0, 0.0, 3.831706, 8.0, float("inf"), float("nan")])
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_j1_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.special.bessel_j1(source)
    expected[0, 0] = -0.0
    expected[0, 6] = 0.0
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0.0005, atol=5e-06, equal_nan=True)
    assert torch.signbit(actual[0, 0])

# -------------------------------------------------------- jn ---------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def jn_tile(order_tile, src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.jn(order_tile[0, tid], src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_jn_tile(
    order: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    order_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    order_tile = order_tile_group.current()
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(order_tile, order, [0, 0])
        pl.load(src_tile, src, [0, 0])
        jn_tile[ELEMENTS](order_tile, src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


def bessel_jn_golden(order, value):
    value_fp64 = value.to(torch.float64)
    previous = torch.special.bessel_j0(value_fp64)
    current = torch.special.bessel_j1(value_fp64)
    result = torch.where(order == 0, previous, current)
    for current_order in range(1, int(order.max().item())):
        following = 2.0 * current_order * current / value_fp64 - previous
        result = torch.where(order == current_order + 1, following, result)
        previous, current = current, following
    result = torch.where((order > 1) & (value == 0.0), torch.zeros_like(result), result)
    result = torch.where(torch.isinf(value), torch.zeros_like(result), result)
    result = torch.where((order < 0) | torch.isnan(value), torch.full_like(result, float("nan")), result)
    return result.to(torch.float32)


@pytest.mark.soc("950")
def test_jn_int32_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(8.0, 32.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    order = torch.tensor([index % 6 for index in range(ELEMENTS)], dtype=torch.int32).reshape(1, ELEMENTS)
    source[0, :10] = torch.tensor([1.0, 1.0, 0.0, float("inf"), float("nan"), 32.0, 8.0, 2.0, 11.0, -8.0])
    order[0, :10] = torch.tensor([-1, 0, 2, 3, 4, 2, 5, 5, 12, 5], dtype=torch.int32)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_jn_tile(order.to(ST_DEVICE), source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), bessel_jn_golden(order, source), rtol=0.0005, atol=5e-06, equal_nan=True)


@pytest.mark.soc("950")
def test_jn_preserves_subnormal_series_result():
    torch.npu.set_device(ST_DEVICE)
    source = torch.ones((1, ELEMENTS), dtype=torch.float32)
    order = torch.full((1, ELEMENTS), 30, dtype=torch.int32)
    output = torch.empty_like(source, device=ST_DEVICE)

    simt_jn_tile(order.to(ST_DEVICE), source.to(ST_DEVICE), output)
    torch.npu.synchronize()

    actual = output.cpu()
    value = source.to(torch.float64)
    prefix = torch.pow(value * 0.5, 30) / float(math.factorial(30))
    multiplier = value * -0.5 * value * 0.5
    term = torch.ones_like(value)
    series = torch.zeros_like(value)
    for index in range(1, 15):
        series += term
        term *= multiplier / (index * (index + 30))
    expected = (prefix * series).to(torch.float32)

    assert torch.count_nonzero(actual).item() == actual.numel()
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.0)

# -------------------------------------------------------- y0 ---------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def y0_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.y0(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_y0_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        y0_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_y0_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.01, 32.0, ELEMENTS)
    values[:9] = torch.tensor(
        [-1.0, 0.0, torch.finfo(torch.float32).tiny, 0.5, 2.197141326, 7.999, 8.0, float("inf"), float("nan")]
    )
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_y0_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.special.bessel_y0(source)
    expected[0, 7] = 0.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0005, atol=5e-06, equal_nan=True)

# -------------------------------------------------------- y1 ---------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def y1_tile(src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.y1(src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_y1_tile(
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(src_tile, src, [0, 0])
        y1_tile[ELEMENTS](src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


@pytest.mark.soc("950")
def test_y1_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    values = torch.linspace(0.01, 32.0, ELEMENTS)
    values[:9] = torch.tensor(
        [-1.0, 0.0, torch.finfo(torch.float32).tiny, 1.2, 3.0, 7.999, 8.0, float("inf"), float("nan")]
    )
    source = values.to(torch.float32).reshape(1, -1)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_y1_tile(source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    expected = torch.special.bessel_y1(source)
    expected[0, 7] = 0.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0005, atol=5e-06, equal_nan=True)

# -------------------------------------------------------- yn ---------------------------------------------------------

@pl.vector_function(mode="simt", max_threads=ELEMENTS)
def yn_tile(order_tile, src_tile, out_tile):
    tid = pl.simt.linear_thread_idx()
    out_tile[0, tid] = pl.simt.yn(order_tile[0, tid], src_tile[0, tid])


@pl.jit(auto_mutex=True)
def simt_yn_tile(
    order: pl.Tensor[[1, ELEMENTS], pl.DT_INT32],
    src: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    out: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
):
    order_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids="auto",
        depth=1,
    )
    order_tile = order_tile_group.current()
    src_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0400,
        mutex_ids="auto",
        depth=1,
    )
    src_tile = src_tile_group.current()
    out_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, ELEMENTS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0800,
        mutex_ids="auto",
        depth=1,
    )
    out_tile = out_tile_group.current()
    with pl.section_vector():
        pl.load(order_tile, order, [0, 0])
        pl.load(src_tile, src, [0, 0])
        yn_tile[ELEMENTS](order_tile, src_tile, out_tile)
        pl.store(out, out_tile, [0, 0])


def bessel_yn_golden(order, value):
    value_fp64 = value.to(torch.float64)
    previous = torch.special.bessel_y0(value_fp64)
    current = torch.special.bessel_y1(value_fp64)
    result = torch.where(order == 0, previous, current)
    for current_order in range(1, int(order.max().item())):
        following = 2.0 * current_order * current / value_fp64 - previous
        result = torch.where(order == current_order + 1, following, result)
        previous, current = current, following
    result = torch.where(value == 0.0, torch.full_like(result, float("-inf")), result)
    result = torch.where(torch.isinf(value), torch.zeros_like(result), result)
    invalid = (order < 0) | (value < 0.0) | torch.isnan(value)
    result = torch.where(invalid, torch.full_like(result, float("nan")), result)
    return result.to(torch.float32)


@pytest.mark.soc("950")
def test_yn_int32_fp32_with_tile_operands():
    torch.npu.set_device(ST_DEVICE)
    source = torch.linspace(8.0, 32.0, ELEMENTS, dtype=torch.float32).reshape(1, ELEMENTS)
    order = torch.tensor([index % 6 for index in range(ELEMENTS)], dtype=torch.int32).reshape(1, ELEMENTS)
    source[0, :10] = torch.tensor([1.0, -1.0, 0.0, float("inf"), float("nan"), 32.0, 8.0, 0.5, 11.0, 1.0])
    order[0, :10] = torch.tensor([-1, 2, 2, 3, 4, 2, 5, 5, 12, 0], dtype=torch.int32)
    output = torch.empty_like(source, device=ST_DEVICE)
    simt_yn_tile(order.to(ST_DEVICE), source.to(ST_DEVICE), output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), bessel_yn_golden(order, source), rtol=0.0005, atol=5e-06, equal_nan=True)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    ("kernel", "input_value", "reference"),
    [
        pytest.param(simt_j1_tile, 9.9e12, -2.39200242285566645970e-7, id="j1_9.9e12"),
        pytest.param(simt_j1_tile, 1e13, 1.28693940247646966895e-7, id="j1_1e13"),
        pytest.param(simt_y0_tile, 9.9e12, -2.39200242285562393856e-7, id="y0_9.9e12"),
        pytest.param(simt_y0_tile, 1e13, 1.28693940247636115645e-7, id="y0_1e13"),
        pytest.param(simt_y1_tile, 9.9e12, 8.41918629080847397064e-8, id="y1_9.9e12"),
        pytest.param(simt_y1_tile, 1e13, -2.17024994126213379997e-7, id="y1_1e13"),
    ],
)
def test_bessel_fp32_huge_arguments_with_tile_operands(kernel, input_value, reference):
    # References use https://dlmf.nist.gov/10.17.E3 and E4 at the exact FP32 inputs.
    _check_bessel_fp32_tile_accuracy(kernel, input_value, reference)
