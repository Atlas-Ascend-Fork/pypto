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

"""ST coverage for pl.dump_data across GM and local memory spaces."""

import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
DEBUG_BLOCK_DIM = 1
FP4_DUMP_ROWS = 16
FP4_DUMP_COLS = 64
SPECIAL_BUFFER_N = 64


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        name = torch.npu.get_device_name()
        if "Ascend950" not in name:
            pytest.skip(f"Device {name} is not A5 (Ascend950). Skip.")
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")


@pl.jit()
def dump_data_tensor_kernel(
    out: pl.Tensor[[32], pl.DT_INT32],
):
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        for i in pl.range(0, 32):
            out[i] = i * 2
        if sub_id == 0:
            pl.dump_data(out, flag="tensor_full")
            pl.dump_data(out, offsets=[8], shapes=[8], flag="tensor_window")
            pl.dump_data(out, loc=True, flag="tensor_loc")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_tensor_full_window_and_loc():
    _check_npu()
    expected = torch.arange(32, device=ST_DEVICE, dtype=torch.int32) * 2
    out = torch.zeros(32, device=ST_DEVICE, dtype=torch.int32)
    dump_data_tensor_kernel[None, DEBUG_BLOCK_DIM](out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, expected)


@pl.jit(auto_mutex=True)
def dump_data_vec_kernel(
    a: pl.Tensor[[32, 64], pl.DT_INT32],
    b: pl.Tensor[[32, 64], pl.DT_INT32],
    out: pl.Tensor[[32, 64], pl.DT_INT32],
):
    tile_type = pl.TileType(shape=[32, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    a_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    b_group = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[1])
    result_group = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        a_tile = a_group.current()
        b_tile = b_group.current()
        result = result_group.current()
        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.add(result, a_tile, b_tile)
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            pl.dump_data(result, flag="vec_full")
            pl.dump_data(result, offsets=[4, 3], shapes=[8, 9], flag="vec_window")
            pl.dump_data(result, offsets=[0, 0], shapes=[8, 8], loc=True, flag="vec_loc")
        pl.store(out, result, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_vec_full_window_and_loc():
    _check_npu()
    a = torch.arange(32 * 64, device=ST_DEVICE, dtype=torch.int32).reshape(32, 64)
    b = a
    out = torch.empty_like(a)
    dump_data_vec_kernel[None, DEBUG_BLOCK_DIM](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)


def _make_fp4_dump_kernel(name: str, dtype):
    @pl.jit(name=name, auto_mutex=True)
    def kernel(
        inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
    ):
        l1_nz_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[FP4_DUMP_ROWS, FP4_DUMP_COLS],
                dtype=dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x0000,
            mutex_ids=[0],
        )
        l1_zn_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[FP4_DUMP_COLS, FP4_DUMP_ROWS],
                dtype=dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
            ),
            addrs=0x10000,
            mutex_ids=[1],
        )
        ub_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[FP4_DUMP_ROWS, FP4_DUMP_COLS],
                dtype=dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.ND,
            ),
            addrs=0x0000,
            mutex_ids=[2],
        )

        with pl.section_cube():
            l1_nz = l1_nz_group.current()
            l1_zn = l1_zn_group.current()
            pl.load(l1_nz, inp, [0, 0])
            pl.load(l1_zn, inp, [0, 0], order=[1, 0])
            pl.dump_data(l1_nz, flag="fp4_l1_nz_full")
            pl.dump_data(l1_nz, offsets=[1, 3], shapes=[7, 9], flag="fp4_l1_nz_window")
            pl.dump_data(l1_zn, flag="fp4_l1_zn_full")
            pl.dump_data(l1_zn, offsets=[1, 3], shapes=[7, 9], flag="fp4_l1_zn_window")

        with pl.section_vector():
            ub = ub_group.current()
            pl.load(ub, inp, [0, 0])
            sub_id = pl.get_subblock_idx()
            if sub_id == 0:
                pl.dump_data(inp, flag="fp4_gm_full")
                pl.dump_data(inp, offsets=[1, 3], shapes=[7, 9], flag="fp4_gm_window")
                pl.dump_data(ub, flag="fp4_ub_full")
                pl.dump_data(ub, offsets=[1, 3], shapes=[7, 9], flag="fp4_ub_window")
            pl.store(out, ub, [0, 0])

    return kernel


_FP4_DUMP_CASES = (
    ("fp4_e2m1", pl.DT_FP4E2M1),
    ("fp4_e1m2", pl.DT_FP4E1M2),
)
_FP4_DUMP_KERNELS = {label: _make_fp4_dump_kernel(f"dump_data_{label}", dtype) for label, dtype in _FP4_DUMP_CASES}


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
@pytest.mark.parametrize("label,dtype", _FP4_DUMP_CASES, ids=[case[0] for case in _FP4_DUMP_CASES])
def test_dump_data_fp4_supported_memory_levels(label: str, dtype):
    del dtype
    _check_npu()
    packed = (
        torch.arange(FP4_DUMP_ROWS * FP4_DUMP_COLS // 2, device=ST_DEVICE, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(FP4_DUMP_ROWS, FP4_DUMP_COLS // 2)
    )
    out = torch.zeros_like(packed)
    _FP4_DUMP_KERNELS[label][None, DEBUG_BLOCK_DIM](packed, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, packed)


@pl.jit(auto_mutex=True)
def dump_data_dynamic_offset_kernel(
    a: pl.Tensor[[32, 32], pl.DT_INT32],
    b: pl.Tensor[[32, 32], pl.DT_INT32],
    out: pl.Tensor[[32, 32], pl.DT_INT32],
):
    tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    a_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    b_group = pl.make_tile_group(type=tile_type, addrs=0x1000, mutex_ids=[1])
    result_group = pl.make_tile_group(type=tile_type, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        a_tile = a_group.current()
        b_tile = b_group.current()
        result = result_group.current()
        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.add(result, a_tile, b_tile)
        sub_id = pl.get_subblock_idx()
        if sub_id == 0:
            for row_offset in pl.range(0, 32, 8):
                pl.dump_data(result, offsets=[row_offset, 0], shapes=[8, 8], flag="dynamic_offset")
        pl.store(out, result, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_dynamic_offset():
    _check_npu()
    a = torch.arange(32 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(32, 32)
    b = a
    out = torch.empty_like(a)
    dump_data_dynamic_offset_kernel[None, DEBUG_BLOCK_DIM](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)


@pl.jit(auto_mutex=True)
def dump_data_control_flow_kernel(
    a: pl.Tensor[[64, 32], pl.DT_INT32],
    b: pl.Tensor[[64, 32], pl.DT_INT32],
    out: pl.Tensor[[64, 32], pl.DT_INT32],
):
    tile_type = pl.TileType(shape=[16, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    a_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    b_group = pl.make_tile_group(type=tile_type, addrs=0x0800, mutex_ids=[1])
    result_group = pl.make_tile_group(type=tile_type, addrs=0x1000, mutex_ids=[2])
    with pl.section_vector():
        a_tile = a_group.current()
        b_tile = b_group.current()
        result = result_group.current()
        sub_id = pl.get_subblock_idx()
        for offset in pl.range(0, 64, 16):
            pl.load(a_tile, a, [offset, 0])
            pl.load(b_tile, b, [offset, 0])
            pl.add(result, a_tile, b_tile)
            if sub_id == 0:
                if offset == 0:
                    pl.dump_data(result, flag="control_flow_full")
                else:
                    pl.dump_data(result, offsets=[0, 0], shapes=[8, 16], flag="control_flow_window")
            pl.store(out, result, [offset, 0])

        index: pl.DT_INT32 = 0
        while index < 2:
            if sub_id == 0:
                pl.dump_data(result, offsets=[index, 0], shapes=[4, 8], flag="control_flow_while")
            index = index + 1


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_control_flow():
    _check_npu()
    a = torch.arange(64 * 32, device=ST_DEVICE, dtype=torch.int32).reshape(64, 32)
    b = a
    out = torch.empty_like(a)
    dump_data_control_flow_kernel[None, DEBUG_BLOCK_DIM](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)


@pl.jit(auto_mutex=True)
def dump_data_cube_levels_kernel(
    a: pl.Tensor[[16, 16], pl.DT_FP16],
    b: pl.Tensor[[16, 16], pl.DT_FP16],
    out: pl.Tensor[[16, 16], pl.DT_FP32],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        acc = acc_group.current()
        pl.load(a_l1, a, [0, 0])
        pl.load(b_l1, b, [0, 0])
        pl.dump_data(a_l1, flag="l1_full")
        pl.dump_data(a_l1, offsets=[4, 3], shapes=[8, 9], flag="l1_window")
        pl.move(a_l0a, a_l1)
        pl.move(b_l0b, b_l1)
        pl.matmul(acc, a_l0a, b_l0b)
        pl.dump_data(acc, flag="l0c_full")
        pl.dump_data(acc, offsets=[4, 3], shapes=[8, 9], flag="l0c_window")
        pl.store(out, acc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_cube_levels():
    _check_npu()
    a = torch.randn(16, 16, device=ST_DEVICE, dtype=torch.float16)
    b = torch.randn(16, 16, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(16, 16, device=ST_DEVICE, dtype=torch.float32)
    dump_data_cube_levels_kernel[None, DEBUG_BLOCK_DIM](a, b, out)
    torch.npu.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def dump_data_special_cube_buffers_kernel(
    bias_src: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    fixpipe_src: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT64],
):
    bias_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, SPECIAL_BUFFER_N],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Mat,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    fixpipe_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, SPECIAL_BUFFER_N],
            dtype=pl.DT_UINT64,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ND,
        ),
        addrs=0x0400,
        mutex_ids=[1],
    )
    bias_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, SPECIAL_BUFFER_N],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Bias,
        ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    fixpipe_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, SPECIAL_BUFFER_N],
            dtype=pl.DT_UINT64,
            target_memory=pl.MemorySpace.Scaling,
        ),
        addrs=0x0000,
        mutex_ids=[3],
    )

    with pl.section_cube():
        bias_l1 = bias_l1_group.current()
        fixpipe_l1 = fixpipe_l1_group.current()
        bias = bias_group.current()
        fixpipe = fixpipe_group.current()
        pl.load(bias_l1, bias_src, [0, 0])
        pl.load(fixpipe_l1, fixpipe_src, [0, 0])
        pl.move(bias, bias_l1)
        pl.move(fixpipe, fixpipe_l1)

        pl.dump_data(bias, flag="bias_table_full")
        pl.dump_data(bias, offsets=[0, 16], shapes=[1, 32], flag="bias_table_window")
        pl.dump_data(fixpipe, flag="fixpipe_full")
        pl.dump_data(fixpipe, offsets=[0, 16], shapes=[1, 32], flag="fixpipe_window")


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dump_data_special_cube_buffers():
    _check_npu()
    observable_values = torch.arange(1, SPECIAL_BUFFER_N + 1)
    bias_src = observable_values.to(torch.int32).reshape(1, SPECIAL_BUFFER_N).to(ST_DEVICE)
    fixpipe_src = observable_values.to(torch.uint64).reshape(1, SPECIAL_BUFFER_N).to(ST_DEVICE)
    dump_data_special_cube_buffers_kernel[None, DEBUG_BLOCK_DIM](
        bias_src,
        fixpipe_src,
    )
    torch.npu.synchronize()
