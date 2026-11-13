# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A5 numerical coverage for tile instructions expanded inside one SIMD VF."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401


def _tile(tile_type, addr):
    group = pl.make_tile_group(type=tile_type, addrs=addr, mutex_ids="auto", depth=1)
    return group.current()


@pl.vector_function(mode="simd_tile")
def select_and_negate(dst0, dst1, src0, src1, index):
    destinations = (dst0, dst1)
    sources = (src0, src1)
    dst = destinations[index]
    src = sources[index]
    pl.neg(dst, src)


@pl.jit(auto_mutex=True)
def tuple_kernel(
    x: pl.Tensor[[2, 64], pl.DT_FP32],
    y0: pl.Tensor[[2, 64], pl.DT_FP32],
    y1: pl.Tensor[[2, 64], pl.DT_FP32],
    rows: pl.DT_INT32,
    cols: pl.DT_INT32,
):
    with pl.section_vector():
        tt = pl.TileType(shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        dst0 = _tile(tt, 0)
        dst1 = _tile(tt, 512)
        src0 = _tile(tt, 1024)
        src1 = _tile(tt, 1536)
        pl.set_validshape(dst0, [1, 17])
        pl.set_validshape(src0, [1, 17])
        pl.set_validshape(dst1, [rows, cols])
        pl.set_validshape(src1, [rows, cols])
        pl.load(src0, x, [0, 0])
        pl.load(src1, x, [0, 0])
        pl.load(dst0, y0, [0, 0])
        pl.load(dst1, y1, [0, 0])
        for index in pl.range(2):
            select_and_negate(dst0, dst1, src0, src1, index)
        pl.store(y0, dst0, [0, 0])
        pl.store(y1, dst1, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("rows,cols", [(2, 37), (1, 64)])
def test_simd_tile_dynamic_tuple_metadata(rows, cols, expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.linspace(-2, 2, 128).reshape(2, 64)
    y0 = torch.full_like(x, 99, device="npu")
    y1 = torch.full_like(x, 99, device="npu")
    with expect_pgo_vf():
        tuple_kernel[1](x.npu(), y0, y1, rows, cols)
        expected0 = torch.full_like(x, 99)
        expected1 = torch.full_like(x, 99)
        expected0[:1, :17] = -x[:1, :17]
        expected1[:rows, :cols] = -x[:rows, :cols]
        torch.testing.assert_close(y0.cpu(), expected0, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(y1.cpu(), expected1, rtol=1e-3, atol=1e-4)


@pl.vector_function(mode="simd_tile")
def typed_elementwise(dst, src, value):
    pl.neg(dst, src)
    pl.exp(dst, dst)
    pl.add(dst, dst, value)
    pl.recip(dst, dst)
    pl.mul(dst, dst, src)
    pl.sub(dst, dst, value)
    pl.sub(dst, dst, src)


def _make_typed_kernel(dtype):
    @pl.jit(auto_mutex=True)
    def typed_kernel(
        x: pl.Tensor[[2, 256], dtype],
        y: pl.Tensor[[2, 256], dtype],
        cols: pl.DT_INT32,
        value: pl.DT_FP32,
    ):
        with pl.section_vector():
            src_type = pl.TileType(shape=[2, 256], dtype=dtype, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
            dst_type = pl.TileType(shape=[2, 384], dtype=dtype, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
            src = _tile(src_type, 0)
            dst = _tile(dst_type, 4096)
            pl.set_validshape(src, [2, cols])
            pl.set_validshape(dst, [2, cols])
            pl.load(src, x, [0, 0])
            typed_elementwise(dst, src, value)
            pl.store(y, dst, [0, 0])

    return typed_kernel


@pytest.mark.soc("950")
@pytest.mark.parametrize("cols", [63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("torch_dtype,pl_dtype", [(torch.float32, pl.DT_FP32), (torch.float16, pl.DT_FP16)])
def test_simd_tile_typed_register_steps(torch_dtype, pl_dtype, cols, expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.linspace(-2, 2, 512, dtype=torch_dtype).reshape(2, 256)
    y = torch.full_like(x, 99, device="npu")
    kernel = _make_typed_kernel(pl_dtype)
    with expect_pgo_vf():
        kernel[1](x.npu(), y, cols, 0.25)
        expected = torch.full_like(x, 99)
        valid = x[:, :cols]
        expected[:, :cols] = (torch.reciprocal(torch.exp(-valid) + 0.25) * valid - 0.25) - valid
        tolerance = 2e-3 if torch_dtype == torch.float16 else 1e-3
        torch.testing.assert_close(y.cpu(), expected, rtol=tolerance, atol=tolerance)


@pl.vector_function(mode="simd_tile")
def add_scalar(dst, src, value):
    result = pl.add(dst, src, value)  # noqa: F841 -- exercise a void tile op carried by AssignStmt


@pl.vector_function(mode="simd_tile")
def add_typed_constant(dst, src, value, runtime):
    pl.add(dst, src, value < runtime)


@pl.jit(auto_mutex=True)
def typed_constant_kernel(
    x: pl.Tensor[[3, 70], pl.DT_FP32],
    signed: pl.Tensor[[3, 70], pl.DT_FP32],
    unsigned: pl.Tensor[[3, 70], pl.DT_FP32],
    runtime: pl.DT_INT32,
):
    with pl.section_vector():
        tt = pl.TileType(shape=[4, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[3, 70])
        src = _tile(tt, 0)
        signed_tile = _tile(tt, 0x1000)
        unsigned_tile = _tile(tt, 0x2000)
        pl.load(src, x, [0, 0])
        add_typed_constant(signed_tile, src, pl.const(1, pl.DT_INT32), runtime)
        add_typed_constant(unsigned_tile, src, pl.const(1, pl.DT_UINT32), runtime)
        pl.store(signed, signed_tile, [0, 0])
        pl.store(unsigned, unsigned_tile, [0, 0])


@pytest.mark.soc("950")
def test_simd_tile_typed_constants_with_tail(expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.linspace(-2, 2, 210).reshape(3, 70)
    signed = torch.empty_like(x, device="npu")
    unsigned = torch.empty_like(x, device="npu")
    with expect_pgo_vf():
        typed_constant_kernel[1](x.npu(), signed, unsigned, -2)
        torch.testing.assert_close(signed.cpu(), x, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(unsigned.cpu(), x + 1, rtol=1e-3, atol=1e-4)


@pl.vector_function(mode="simd_tile")
def repeat_update(dst, src, value, count):
    pl.neg(dst, src)
    for i in pl.range(count):
        if i % 2 == 0:
            add_scalar(dst, dst, value)
        else:
            pl.sub(dst, dst, value)


@pl.jit(auto_mutex=True)
def tail_kernel(
    x: pl.Tensor[[3, 70], pl.DT_FP32],
    y: pl.Tensor[[3, 70], pl.DT_FP32],
    value: pl.DT_FP32,
    count: pl.DT_INT32,
):
    with pl.section_vector():
        src_type = pl.TileType(shape=[4, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[3, 70])
        dst_type = pl.TileType(shape=[4, 192], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[3, 70])
        src = _tile(src_type, 0)
        dst = _tile(dst_type, 0x2000)
        pl.load(src, x, [0, 0])
        repeat_update(dst, src, value, count)
        pl.store(y, dst, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("count", [1, 2, 5])
def test_simd_tile_tail_nested_loop(count, expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.linspace(-2, 2, 210).reshape(3, 70)
    y = torch.empty_like(x, device="npu")
    with expect_pgo_vf():
        tail_kernel[1](x.npu(), y, 0.25, count)
        torch.testing.assert_close(y.cpu(), -x + (count % 2) * 0.25, rtol=1e-3, atol=1e-4)


@pl.jit(auto_mutex=True)
def full_and_tail_kernel(x: pl.Tensor[[7, 326], pl.DT_FP32], y: pl.Tensor[[7, 326], pl.DT_FP32]):
    with pl.section_vector():
        full_src = _tile(
            pl.TileType(shape=[4, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[4, 128]), 0
        )
        full_dst = _tile(
            pl.TileType(shape=[4, 192], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[4, 128]),
            0x2000,
        )
        row_src = _tile(
            pl.TileType(shape=[4, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[3, 128]),
            0x4000,
        )
        row_dst = _tile(
            pl.TileType(shape=[4, 192], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[3, 128]),
            0x6000,
        )
        col_src = _tile(
            pl.TileType(shape=[4, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[4, 70]), 0x8000
        )
        col_dst = _tile(
            pl.TileType(shape=[4, 192], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[4, 70]), 0xA000
        )
        corner_src = _tile(
            pl.TileType(shape=[4, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[3, 70]), 0xC000
        )
        corner_dst = _tile(
            pl.TileType(shape=[4, 192], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[3, 70]), 0xE000
        )
        # Instantiate the corner tail first, then verify each static valid-shape
        # specialization writes its entire output region.
        pl.load(corner_src, x, [4, 256])
        repeat_update(corner_dst, corner_src, 0.25, 1)
        pl.store(y, corner_dst, [4, 256])
        for i in pl.range(2):
            pl.load(full_src, x, [0, i * 128])
            repeat_update(full_dst, full_src, 0.25, 1)
            pl.store(y, full_dst, [0, i * 128])
            pl.load(row_src, x, [4, i * 128])
            repeat_update(row_dst, row_src, 0.25, 1)
            pl.store(y, row_dst, [4, i * 128])
        pl.load(col_src, x, [0, 256])
        repeat_update(col_dst, col_src, 0.25, 1)
        pl.store(y, col_dst, [0, 256])


@pytest.mark.soc("950")
def test_simd_tile_full_and_tail_specializations(expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.linspace(-2, 2, 7 * 326).reshape(7, 326)
    y = torch.full_like(x, float("nan"), device="npu")
    with expect_pgo_vf():
        full_and_tail_kernel[1](x.npu(), y)
        torch.testing.assert_close(y.cpu(), -x + 0.25, rtol=1e-3, atol=1e-4)


@pl.vector_function(mode="simd_tile")
def subtract_scalar(dst, value):
    pl.sub(dst, dst, value)


@pl.jit(auto_mutex=True)
def unsigned_sub_kernel(x: pl.Tensor[[1, 64], pl.DT_FP32], y: pl.Tensor[[1, 64], pl.DT_FP32], value: pl.DT_UINT32):
    with pl.section_vector():
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[1, 64])
        tile = _tile(tt, 0)
        pl.load(tile, x, [0, 0])
        subtract_scalar(tile, value)
        pl.store(y, tile, [0, 0])


@pytest.mark.soc("950")
def test_simd_tile_unsigned_scalar_subtraction(expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.full((1, 64), 4.0, device="npu")
    y = torch.empty_like(x)
    with expect_pgo_vf():
        unsigned_sub_kernel[1](x, y, 1)
        torch.testing.assert_close(y.cpu(), torch.full((1, 64), 3.0), rtol=1e-3, atol=1e-4)


@pl.vector_function(mode="simd_tile")
def negate_tile(dst):
    pl.neg(dst, dst)


@pl.vector_function(mode="simd_tile")
def nested_negate_tile(dst):
    negate_tile(dst)


@pl.jit(auto_mutex=True)
def specialized_tiles_kernel(
    x: pl.Tensor[[2, 128], pl.DT_FP32],
    a_out: pl.Tensor[[2, 64], pl.DT_FP32],
    b_out: pl.Tensor[[2, 64], pl.DT_FP32],
    c_out: pl.Tensor[[2, 128], pl.DT_FP32],
    nested_a_out: pl.Tensor[[2, 64], pl.DT_FP32],
    nested_b_out: pl.Tensor[[2, 64], pl.DT_FP32],
    nested_c_out: pl.Tensor[[2, 128], pl.DT_FP32],
):
    with pl.section_vector():
        compact = pl.TileType(shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 64])
        padded = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 64])
        full = pl.TileType(shape=[2, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[2, 128])
        a = _tile(compact, 0)
        b = _tile(padded, 0x1000)
        c = _tile(full, 0x2000)
        pl.load(a, x, [0, 0])
        pl.load(b, x, [0, 0])
        pl.load(c, x, [0, 0])
        negate_tile(a)
        negate_tile(b)
        negate_tile(c)
        pl.store(a_out, a, [0, 0])
        pl.store(b_out, b, [0, 0])
        pl.store(c_out, c, [0, 0])
        nested_negate_tile(a)
        nested_negate_tile(b)
        nested_negate_tile(c)
        pl.store(nested_a_out, a, [0, 0])
        pl.store(nested_b_out, b, [0, 0])
        pl.store(nested_c_out, c, [0, 0])


@pytest.mark.soc("950")
def test_simd_tile_specialized_entry_and_nested_callee(expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.arange(1, 257, dtype=torch.float32).reshape(2, 128)
    outputs = [torch.empty((2, cols), device="npu") for cols in (64, 64, 128, 64, 64, 128)]
    with expect_pgo_vf():
        specialized_tiles_kernel[1](x.npu(), *outputs)
        for i, output in enumerate(outputs):
            expected = x[:, :output.shape[1]] * (-1 if i < 3 else 1)
            torch.testing.assert_close(output.cpu(), expected, rtol=1e-3, atol=1e-4)


@pl.jit(auto_mutex=True)
def boolean_add_kernel(x: pl.Tensor[[1, 64], pl.DT_FP32], y: pl.Tensor[[1, 64], pl.DT_FP32], value: pl.DT_BOOL):
    with pl.section_vector():
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[1, 64])
        tile = _tile(tt, 0)
        pl.load(tile, x, [0, 0])
        add_scalar(tile, tile, value)
        pl.store(y, tile, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("value", [False, True])
def test_simd_tile_boolean_scalar_addition(value, expect_pgo_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.linspace(-2, 2, 64).reshape(1, 64)
    y = torch.empty_like(x, device="npu")
    with expect_pgo_vf():
        boolean_add_kernel[1](x.npu(), y, value)
        torch.testing.assert_close(y.cpu(), x + float(value), rtol=1e-3, atol=1e-4)
