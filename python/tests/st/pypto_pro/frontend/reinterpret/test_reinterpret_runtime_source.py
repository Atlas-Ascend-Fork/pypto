# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Check runtime-selected addresses and descriptor independence on A5."""

import os

import pypto_pro.language as pl
import pytest
import torch

pytestmark = pytest.mark.soc("950")


@pl.jit(auto_mutex=True)
def runtime_source(
    x: pl.Tensor[[2, 32], pl.DT_FP32],
    branch: pl.Tensor[[8, 16], pl.DT_FP32],
    carried: pl.Tensor[[8, 16], pl.DT_FP32],
    indexed: pl.Tensor[[8, 16], pl.DT_FP32],
    grouped: pl.Tensor[[8, 16], pl.DT_FP32],
    window: pl.Tensor[[1, 32], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=[0, 256], mutex_ids=[[0, 2], [1, 3]])
    with pl.section_vector():
        pl.load(group[0], x, [0, 0])
        pl.load(group[1], x, [1, 0])
        for i in pl.range(4):
            if i % 2 == 0:
                tile = group[0]
            else:
                tile = group[1]
            view = pl.reinterpret(tile, shape=[2, 16])
            pl.store(branch, view, [2 * i, 0])

        tile = group[0]
        for i in pl.range(4):
            view = pl.reinterpret(tile, shape=[2, 16])
            pl.store(carried, view, [2 * i, 0])
            tile = group[(i + 1) % 2]

        for i in pl.range(4):
            tile = group[i % 2]
            view = pl.reinterpret(tile, shape=[2, 16])
            pl.store(indexed, view, [2 * i, 0])

        views = pl.reinterpret(group, shape=[2, 16])
        for i in pl.range(4):
            view = views.next()
            pl.store(grouped, view, [2 * i, 0])

        source = group[0]
        pl.set_validshape(source, [1, 16])
        full = pl.reinterpret(source, shape=[1, 32])
        pl.store(window, full, [0, 0])
        # Building the full view must not widen the source's valid window.
        pl.muls(source, source, 2.0)
        pl.store(window, source, [0, 0])


def test_runtime_source_after_branch_loop_and_group_selection():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(device)
    source = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    x = source.to(device)
    outputs = [torch.full((8, 16), -1.0, device=device) for _ in range(4)]
    window = torch.full((1, 32), -1.0, device=device)
    runtime_source[None, 1](x, *outputs, window)
    torch.npu.synchronize()
    expected = source.repeat(2, 1).reshape(8, 16)
    for name, actual in zip(("branch", "carried", "indexed", "grouped"), outputs, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, msg=name)
    expected_window = source[:1].clone()
    expected_window[:, :16] *= 2
    torch.testing.assert_close(window.cpu(), expected_window, rtol=0, atol=0)


@pl.jit(auto_mutex=True)
def swap_tiles(x: pl.Tensor[[2, 32], pl.DT_FP32], out: pl.Tensor[[2, 32], pl.DT_FP32], n: pl.DT_INT64):
    tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    group = pl.make_tile_group(type=tt, addrs=[0, 256], mutex_ids=[0, 1])
    with pl.section_vector():
        pl.load(group[0], x, [0, 0])
        pl.load(group[1], x, [1, 0])
        a = group[0]
        b = group[1]
        for i in pl.range(n):
            a, b = b, a
        pl.store(out, a, [0, 0])
        pl.store(out, b, [1, 0])


@pytest.mark.parametrize("n", [0, 1, 2])
def test_loop_swap_preserves_complete_tile_values(n):
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(device)
    source = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    x = source.to(device)
    output = torch.empty_like(x)
    swap_tiles[None, 1](x, output, n)
    torch.npu.synchronize()
    expected = source.flip(0) if n % 2 else source
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pl.jit(auto_mutex=False)
def fillpad_runtime_view(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    noise: pl.Tensor[[8, 8], pl.DT_INT32],
    out: pl.Tensor[[8, 8], pl.DT_INT32],
):
    tt = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    unused = pl.make_tile(tt, addr=0)
    source = pl.make_tile(tt, addr=512)
    dest = pl.make_tile(tt, addr=1024)
    with pl.section_vector():
        pl.load(unused, noise, [0, 0])
        pl.load(source, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        view = pl.reinterpret(source, shape=[8, 8])
        pl.set_validshape(view, [4, 8])
        pl.fillpad(dest, view)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, dest, [0, 0])
        pl.system.bar_all()


def test_fillpad_uses_runtime_address_of_zero_mutex_view():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(device)
    x = torch.full((8, 8), 7, dtype=torch.int32, device=device)
    noise = torch.full_like(x, 99)
    output = torch.empty_like(x)
    fillpad_runtime_view[None, 1](x, noise, output)
    torch.npu.synchronize()
    expected = torch.zeros((8, 8), dtype=torch.int32)
    expected[:4] = 7
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
