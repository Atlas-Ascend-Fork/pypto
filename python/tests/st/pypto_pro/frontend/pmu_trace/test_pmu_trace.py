# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""PMU instrumentation must preserve device numerical results.

Raw hardware stamps are checked by profilling/test_profiling.py, sharing its
msprof collection with the eager operator-name checks.
"""
import os

import pypto_pro.language as pl
from pypto_pro.runtime.platform import get_platform_info
import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.soc("950")
def test_vector_add_with_pmu():
    @pl.pipeline.stage
    def load_inputs(a, b, x, y):
        pl.load(a, x, [0, 0])
        pl.load(b, y, [0, 0])

    @pl.pipeline.stage
    def compute(c, a, b):
        pl.add(c, a, b)

    @pl.pipeline.stage
    def store_output(z, c):
        pl.store(z, c, [0, 0])

    @pl.jit
    def vector_add(
        x: pl.Tensor[[128, 128], pl.DT_FP32],
        y: pl.Tensor[[128, 128], pl.DT_FP32],
        z: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile_group(type=tile, addrs=0, mutex_ids=[0]).next()
        b = pl.make_tile_group(type=tile, addrs=65536, mutex_ids=[1]).next()
        c = pl.make_tile_group(type=tile, addrs=131072, mutex_ids=[2]).next()
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.MTE2, 0)
            load_inputs(a, b, x, y)
            pl.mark_stamp(pl.PipeType.MTE2, 1)
            pl.mark_stamp(pl.PipeType.V, 2)
            compute(c, a, b)
            pl.mark_stamp(pl.PipeType.V, 3)
            pl.mark_stamp(pl.PipeType.MTE3, 4)
            store_output(z, c)
            pl.mark_stamp(pl.PipeType.MTE3, 5)

    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.rand((128, 128), device="npu", dtype=torch.float32)
    y = torch.rand_like(x)
    z = torch.empty_like(x)
    vector_add[None, 1](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z.cpu(), x.cpu() + y.cpu(), rtol=1e-5, atol=1e-6)


@pytest.mark.soc("950")
def test_mixed_stamps_preserve_all_workers():
    device = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device)
    platform = get_platform_info()
    blocks = min(platform.cube_core_num, platform.vector_core_num // 2)
    assert blocks > 1, platform
    output_size = blocks * 3 * 64
    # msprof monitors a subset of physical cores. Fill the device so that capture
    # coverage does not depend on where two logical blocks happen to be scheduled.
    # Each participant writes a separate cache line.
    # Branch-specific checkpoints identify nonzero blocks in the raw hardware capture.
    @pl.jit
    def write_workers(out: pl.Tensor[[output_size], pl.DT_INT32]):
        with pl.section_cube():
            pl.mark_stamp(pl.PipeType.S, 0)
            worker = 3 * pl.get_block_idx()
            pl.setval(out, worker * 64, worker + 1)
            if pl.get_block_idx() == 0:
                pl.mark_stamp(pl.PipeType.S, 2)
            else:
                pl.mark_stamp(pl.PipeType.S, 3)
            pl.mark_stamp(pl.PipeType.S, 1)
            # Flush the checked stamps out of BIU's partial 32-byte tail packet.
            for _ in pl.range(8):
                pl.mark_stamp(pl.PipeType.S, 4095)
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.S, 0)
            worker = (pl.get_block_idx() // 2) * 3 + 1 + pl.get_subblock_idx()
            pl.setval(out, worker * 64, worker + 1)
            if pl.get_subblock_idx() == 0:
                pl.mark_stamp(pl.PipeType.S, 2)
            else:
                pl.mark_stamp(pl.PipeType.S, 3)
            if pl.get_block_idx() >= 2:
                pl.mark_stamp(pl.PipeType.S, 4)
            else:
                pl.mark_stamp(pl.PipeType.S, 1)
            for _ in pl.range(8):
                pl.mark_stamp(pl.PipeType.S, 4095)

    output = torch.zeros(output_size, dtype=torch.int32, device="npu")
    for _ in range(2):
        output.zero_()
        write_workers[None, blocks](output)
        torch.npu.synchronize()
        assert output.cpu()[::64].tolist() == list(range(1, blocks * 3 + 1))


@pytest.mark.soc("950")
def test_stamp_preserves_early_return():
    @pl.jit
    def write_active_worker(out: pl.Tensor[[128], pl.DT_INT32]):
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.S, 0)
            worker = pl.get_block_idx()
            if worker == 0:
                return
            pl.setval(out, worker * 64, 7)
            pl.mark_stamp(pl.PipeType.S, 1)

    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    output = torch.full((128,), -1, dtype=torch.int32, device="npu")
    write_active_worker[None, 2](output)
    torch.npu.synchronize()
    assert output.cpu()[::64].tolist() == [-1, 7]


@pytest.mark.soc("950")
def test_all_supported_pipes_compile_and_execute():
    @pl.jit(auto_mutex=False)
    def all_pipes(out: pl.Tensor[[192], pl.DT_INT32]):
        with pl.section_cube():
            pl.mark_stamp(pl.PipeType.S, 0)
            pl.mark_stamp(pl.PipeType.MTE2, 1)
            pl.mark_stamp(pl.PipeType.MTE1, 2)
            pl.mark_stamp(pl.PipeType.M, 3)
            pl.mark_stamp(pl.PipeType.FIX, 4)
            pl.mark_stamp(pl.PipeType.MTE3, 4095)
            pl.setval(out, 0, 1)
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.S, 0)
            pl.mark_stamp(pl.PipeType.MTE2, 1)
            pl.mark_stamp(pl.PipeType.V, 2)
            pl.mark_stamp(pl.PipeType.MTE3, 4095)
            worker = pl.get_subblock_idx() + 1
            pl.setval(out, worker * 64, worker + 1)

    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    output = torch.zeros(192, dtype=torch.int32, device="npu")
    all_pipes[None, 1](output)
    torch.npu.synchronize()
    assert output.cpu()[::64].tolist() == [1, 2, 3]


@pytest.mark.soc("950")
@pytest.mark.parametrize("mixed", [False, True])
def test_vector_stamp_preserves_live_user_events(mixed):
    # Keep every V->S event live across a V stamp. Borrowing a fixed event ID
    # would double-set/consume a user's flag; mixing compiler tokens would not compile.
    @pl.jit
    def add_with_events(
        x: pl.Tensor[[2, 64], pl.DT_FP32],
        y: pl.Tensor[[2, 64], pl.DT_FP32],
        z: pl.Tensor[[2, 64], pl.DT_FP32],
    ):
        if mixed:
            with pl.section_cube():
                pl.system.bar_all()
        tile = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a = pl.make_tile_group(type=tile, addrs=0, mutex_ids=[0]).next()
        b = pl.make_tile_group(type=tile, addrs=256, mutex_ids=[1]).next()
        c = pl.make_tile_group(type=tile, addrs=512, mutex_ids=[2]).next()
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.MTE2, 0)
            row = pl.get_block_idx()
            pl.load(a, x, [row, 0])
            pl.load(b, y, [row, 0])
            pl.add(c, a, b)
            for event in pl.range(0, 8):
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=event)
            pl.mark_stamp(pl.PipeType.V, 1)
            for event in pl.range(0, 8):
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.S, event_id=event)
            pl.store(z, c, [row, 0])
            pl.mark_stamp(pl.PipeType.MTE3, 2)

    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    x = torch.arange(128, dtype=torch.float32, device="npu").reshape(2, 64)
    y = torch.ones_like(x)
    z = torch.empty_like(x)
    add_with_events[None, 1 if mixed else 2](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z.cpu(), x.cpu() + y.cpu(), rtol=0, atol=0)
