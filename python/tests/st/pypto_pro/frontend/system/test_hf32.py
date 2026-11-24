# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Ascend 950 HF32 rounding, retained state and FP32/FP16 matmul regression."""

import os

import pypto_pro.language as pl
import pytest
import torch

DIM = 16
ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


def _make_kernel(dtype, accumulate):
    kernel_name = f"hf32_{'fp32' if dtype == pl.DT_FP32 else 'fp16'}_acc{int(accumulate)}"

    @pl.jit(auto_mutex=True, name=kernel_name)
    def kernel(
        a: pl.Tensor[[DIM, DIM], dtype],
        b: pl.Tensor[[DIM, DIM], dtype],
        out: pl.Tensor[[6 * DIM, DIM], pl.DT_FP32],
        mode: pl.DT_INT32,
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[DIM, DIM], dtype=dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[0],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[DIM, DIM], dtype=dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x2000, mutex_ids=[1],
        )
        a_left = pl.make_tile_group(
            type=pl.TileType(shape=[DIM, DIM], dtype=dtype, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0, mutex_ids=[2],
        )
        b_right = pl.make_tile_group(
            type=pl.TileType(shape=[DIM, DIM], dtype=dtype, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0, mutex_ids=[3],
        )
        accumulator = pl.make_tile_group(
            type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024),
            addrs=0, mutex_ids=[4],
        )
        with pl.section_cube():
            a_mat = a_l1.current()
            b_mat = b_l1.current()
            lhs = a_left.current()
            rhs = b_right.current()
            acc = accumulator.current()
            pl.load(a_mat, a, [0, 0])
            pl.load(b_mat, b, [0, 0])
            pl.move(lhs, a_mat)
            pl.move(rhs, b_mat)
            for step in pl.range(0, 6):
                if step == 0 or step == 2:
                    pl.system.disable_hf32()
                elif step == 1:
                    pl.system.enable_hf32()
                    pl.system.set_hf32_round_mode(pl.HF32RoundMode.NEAREST_AWAY)
                elif step == 3:
                    pl.system.enable_hf32()
                elif step == 4:
                    pl.system.set_hf32_round_mode(mode)
                else:
                    pl.system.set_hf32_round_mode(pl.HF32RoundMode.NEAREST_EVEN)
                pl.matmul(acc, lhs, rhs)
                if accumulate:
                    pl.system.bar_m()
                    pl.matmul_acc(acc, acc, lhs, rhs)
                pl.system.bar_m()
                pl.store(out, acc, [step * DIM, 0])
                pl.system.bar_fix()
            pl.system.disable_hf32()

    return kernel


@pytest.mark.soc("950")
@pytest.mark.parametrize("mode", [0, 1], ids=["away", "even"])
@pytest.mark.parametrize("accumulate", [False, True], ids=["matmul", "matmul_acc"])
@pytest.mark.parametrize("fp32", [False, True], ids=["fp16", "fp32"])
def test_hf32_rounding_and_state(mode, accumulate, fp32):
    torch.npu.set_device(ST_DEVICE)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("HF32 configuration requires Ascend 950")

    if fp32:
        # Halfway cases with even/odd retained mantissas, both signs, and values
        # outside FP16's normal exponent range. Identity B isolates input rounding.
        tie = 1 + 2 ** -11
        odd_tie = 1 + 3 * 2 ** -11
        values = [tie, -tie, odd_tie, -odd_tie, tie * 2 ** 20, tie * 2 ** -20, 1, -1] * 2
        up = 1 + 2 ** -10
        up_two = 1 + 2 ** -9
        away = [up, -up, up_two, -up_two, up * 2 ** 20, up * 2 ** -20, 1, -1] * 2
        even = [1, -1, up_two, -up_two, 2 ** 20, 2 ** -20, 1, -1] * 2
    else:
        values = [1.5, -1.5, 2, -2] * 4
        away = even = values

    full = torch.diag(torch.tensor(values, dtype=torch.float32))
    away = torch.diag(torch.tensor(away, dtype=torch.float32))
    even = torch.diag(torch.tensor(even, dtype=torch.float32))
    selected = away if mode == 0 else even
    expected = torch.cat([full, away, full, away, selected, even]) * (2 if accumulate else 1)
    dtype = torch.float32 if fp32 else torch.float16
    a = full.to(device=ST_DEVICE, dtype=dtype)
    b = torch.eye(DIM, device=ST_DEVICE, dtype=dtype)
    output = torch.empty((6 * DIM, DIM), device=ST_DEVICE, dtype=torch.float32)
    kernel = _make_kernel(pl.DT_FP32 if fp32 else pl.DT_FP16, accumulate)
    kernel[None, 1](a, b, output, mode)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pl.jit(auto_mutex=True)
def hf32_direct_default_kernel(
    a: pl.Tensor[[DIM, DIM], pl.DT_FP32],
    b: pl.Tensor[[DIM, DIM], pl.DT_FP32],
    out: pl.Tensor[[DIM, DIM], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x2000, mutex_ids=[1],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0, mutex_ids=[2],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0, mutex_ids=[3],
    )
    accumulator = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024),
        addrs=0, mutex_ids=[4],
    )
    with pl.section_cube():
        a_mat = a_l1.current()
        b_mat = b_l1.current()
        lhs = a_left.current()
        rhs = b_right.current()
        acc = accumulator.current()
        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.move(lhs, a_mat)
        pl.move(rhs, b_mat)
        pl.system.enable_hf32()
        pl.matmul(acc, lhs, rhs)
        pl.system.bar_m()
        pl.store(out, acc, [0, 0])
        pl.system.bar_fix()
        pl.system.disable_hf32()


@pl.jit(auto_mutex=True)
def hf32_direct_nearest_even_kernel(
    a: pl.Tensor[[DIM, DIM], pl.DT_FP32],
    b: pl.Tensor[[DIM, DIM], pl.DT_FP32],
    out: pl.Tensor[[DIM, DIM], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x2000, mutex_ids=[1],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0, mutex_ids=[2],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0, mutex_ids=[3],
    )
    accumulator = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024),
        addrs=0, mutex_ids=[4],
    )
    with pl.section_cube():
        a_mat = a_l1.current()
        b_mat = b_l1.current()
        lhs = a_left.current()
        rhs = b_right.current()
        acc = accumulator.current()
        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.move(lhs, a_mat)
        pl.move(rhs, b_mat)
        pl.system.enable_hf32()
        pl.system.set_hf32_round_mode(pl.HF32RoundMode.NEAREST_EVEN)
        pl.matmul(acc, lhs, rhs)
        pl.system.bar_m()
        pl.store(out, acc, [0, 0])
        pl.system.bar_fix()
        pl.system.disable_hf32()


@pl.jit(auto_mutex=True)
def hf32_direct_nearest_away_kernel(
    a: pl.Tensor[[DIM, DIM], pl.DT_FP32],
    b: pl.Tensor[[DIM, DIM], pl.DT_FP32],
    out: pl.Tensor[[DIM, DIM], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x2000, mutex_ids=[1],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0, mutex_ids=[2],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0, mutex_ids=[3],
    )
    accumulator = pl.make_tile_group(
        type=pl.TileType(shape=[DIM, DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024),
        addrs=0, mutex_ids=[4],
    )
    with pl.section_cube():
        a_mat = a_l1.current()
        b_mat = b_l1.current()
        lhs = a_left.current()
        rhs = b_right.current()
        acc = accumulator.current()
        pl.load(a_mat, a, [0, 0])
        pl.load(b_mat, b, [0, 0])
        pl.move(lhs, a_mat)
        pl.move(rhs, b_mat)
        pl.system.enable_hf32()
        pl.system.set_hf32_round_mode(pl.HF32RoundMode.NEAREST_AWAY)
        pl.matmul(acc, lhs, rhs)
        pl.system.bar_m()
        pl.store(out, acc, [0, 0])
        pl.system.bar_fix()
        # The AWAY result is complete; restore the default for subsequent tests.
        pl.system.set_hf32_round_mode(pl.HF32RoundMode.NEAREST_EVEN)
        pl.system.disable_hf32()



def _run_hf32_direct(kernel, rounded):
    torch.npu.set_device(ST_DEVICE)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("HF32 configuration requires Ascend 950")

    tie = 1 + 2 ** -11
    odd_tie = 1 + 3 * 2 ** -11
    values = [tie, -tie, odd_tie, -odd_tie, tie * 2 ** 20, tie * 2 ** -20, 1, -1] * 2
    a = torch.diag(torch.tensor(values, dtype=torch.float32)).to(ST_DEVICE)
    b = torch.eye(DIM, device=ST_DEVICE, dtype=torch.float32)
    output = torch.empty((DIM, DIM), device=ST_DEVICE, dtype=torch.float32)
    expected = torch.diag(torch.tensor(rounded, dtype=torch.float32))
    kernel[None, 1](a, b, output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950")
def test_hf32_direct_default():
    up_two = 1 + 2 ** -9
    rounded = [1, -1, up_two, -up_two, 2 ** 20, 2 ** -20, 1, -1] * 2
    _run_hf32_direct(hf32_direct_default_kernel, rounded)


@pytest.mark.soc("950")
def test_hf32_direct_nearest_even():
    up_two = 1 + 2 ** -9
    rounded = [1, -1, up_two, -up_two, 2 ** 20, 2 ** -20, 1, -1] * 2
    _run_hf32_direct(hf32_direct_nearest_even_kernel, rounded)


@pytest.mark.soc("950")
def test_hf32_direct_nearest_away():
    up = 1 + 2 ** -10
    up_two = 1 + 2 ** -9
    rounded = [up, -up, up_two, -up_two, up * 2 ** 20, up * 2 ** -20, 1, -1] * 2
    _run_hf32_direct(hf32_direct_nearest_away_kernel, rounded)
