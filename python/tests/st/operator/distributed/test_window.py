#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import time

import pypto_pro.language as pl
import pypto_pro.language.distribute as pld
import pytest
import torch
import torch.distributed as dist


def init_dist(rank, nrank):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    dist.init_process_group(backend='hccl', init_method=f'tcp://{ip}:{port}', world_size=nrank, rank=rank)
    return dist.distributed_c10d._get_default_group()


def worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=1 << 20)
    print(str(win), flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_create_window(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(worker, args=(nprocs,), nprocs=nprocs)


@pl.jit
def probe(win: pld.Window, out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        if pl.get_block_idx() == 0:  # one writer / one printf line per rank
            pl.printf("rank %d / %d, size %d\n", win.rank(), win.nrank(), win.size())
            out[0] = win.rank()
            out[1] = win.nrank()
            out[2] = win.size()
            out[3] = win.num_signals()


def probe_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=1 << 20)
    # Fill on CPU + H2D copy: some CANN installs miss the NPU Fill operator package
    # (EZ1013 on aclnnInplaceFillScalar), which is unrelated to the code under test.
    out = torch.full((4,), -1, dtype=torch.int64).npu()
    probe(win, out)
    torch.npu.synchronize()
    expect = [win.rank(), win.nrank(), win.size(), win.num_signals()]
    actual = out.cpu().tolist()
    assert actual == expect, f"[rank {rank}] probe got {actual}, expected {expect}"
    print(f"[rank {rank}] probe ok: {expect}", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_window_probe(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(probe_worker, args=(nprocs,), nprocs=nprocs)


@pl.jit
def barrier_kernel(win: pld.Window, out: pl.Tensor[[3], pl.DT_INT64]):
    with pl.section_vector():
        # Unguarded: the SPMD contract requires every AIV of the launch to reach the barrier.
        pld.barrier(win)
        if pl.get_block_idx() == 0:
            out[0] = 1
        pld.barrier(win)
        if pl.get_block_idx() == 0:
            out[1] = 2
        pld.barrier(win, sync_local=False, flush_store=False)  # knob path: leader-only exchange, no store flush
        if pl.get_block_idx() == 0:
            out[2] = 3


def barrier_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=1 << 20)
    out = torch.full((3,), -1, dtype=torch.int64).npu()

    time.sleep(rank * 0.5)  # stagger launches: the barrier must block until the last rank arrives

    barrier_kernel(win, out)
    torch.npu.synchronize()

    assert out.cpu().tolist() == [1, 2, 3], f"[rank {rank}] markers got {out.cpu().tolist()}"

    # Sense-reversal end state after 3 barriers( counter @ +0, signals @ +8):
    # i0 counts signals[0] up to nrank, i1 counts signals[1] up to nrank, i2 counts signals[0]
    # back down to 0 — host-visible proof that every rank's remote atomics landed.
    off = win.size() + 512 * win.num_signals()
    region = win.device_buffer[off:off + 16]
    counter = region[:8].view(torch.int64).cpu().item()
    sig0, sig1 = region[8:16].view(torch.int32).cpu().tolist()
    assert counter == 3, f"[rank {rank}] barrier counter got {counter}, expected 3"
    assert sig0 == 0, f"[rank {rank}] signals[0] got {sig0}, expected 0 (counted back down)"
    assert sig1 == nrank, f"[rank {rank}] signals[1] got {sig1}, expected {nrank}"
    print(f"[rank {rank}] barrier ok: counter 3, signals [{sig0}, {sig1}]", flush=True)

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_window_barrier(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(barrier_worker, args=(nprocs,), nprocs=nprocs)


_SHIFT_N = 1024
_SHIFT_DTYPE = pl.DT_BF16


@pl.jit
def shift_load_right(win: pld.Window, src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                     out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tile_type, addr=0)
        t2 = pl.make_tile(tile_type, addr=_SHIFT_N * 2)
        pl.load(t, src, [0, 0])
        me = win.rank()
        nr = win.nrank()
        dt = pld.make_dtensor(win, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pld.put(dt, t, [0, 0], (me + 1) % nr)  # push my tile into the NEXT rank's window
        pld.barrier(win)  # flushes MTE3 (the put) before the arrivals; all pushes land before any pull
        pld.get(t2, dt, [0, 0], me)  # my own window now holds the PREVIOUS rank's push
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=3)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=3)
        pl.store(out, t2, [0, 0])


# Left shift on the auto-mutex path: tile groups + get_buf/rls_buf hardware ordering, no
# manual sync_src/sync_dst — the cross-rank counterpart of local_roundtrip_mutex.
@pl.jit(auto_mutex=True)
def shift_load_left(win: pld.Window, src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                    out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg_a = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        tg_b = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[1])
        t = tg_a.current()
        t2 = tg_b.current()
        pl.load(t, src, [0, 0])
        me = win.rank()
        nr = win.nrank()
        dt = pld.make_dtensor(win, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        pld.put(dt, t, [0, 0], (me + nr - 1) % nr)  # push my tile into the PREVIOUS rank's window
        pld.barrier(win)
        pld.get(t2, dt, [0, 0], me)  # my own window now holds the NEXT rank's push
        pl.store(out, t2, [0, 0])


def _shift_setup(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=1 << 20)
    src = torch.full((1, _SHIFT_N), float(rank), dtype=torch.bfloat16).npu()
    out = torch.zeros((1, _SHIFT_N), dtype=torch.bfloat16).npu()
    return win, src, out


@pl.jit(auto_mutex=True)
def local_roundtrip_mutex(win: pld.Window, src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                          out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg_a = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        tg_b = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[1])
        t = tg_a.current()
        t2 = tg_b.current()
        pl.load(t, src, [0, 0])
        me = win.rank()
        dt = pld.make_dtensor(win, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        pld.put(dt, t, [0, 0], me)  # peer == own rank: local GM copy through the window
        pld.barrier(win)
        pld.get_tile(t2, dt, [0, 0], me)
        pl.store(out, t2, [0, 0])


def local_roundtrip_mutex_worker(rank, nrank):
    win, src, out = _shift_setup(rank, nrank)
    local_roundtrip_mutex(win, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float(rank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] local_roundtrip_mutex got {out.cpu()[0, :3].tolist()}..., expected own data"
    )
    print(f"[rank {rank}] local_roundtrip_mutex ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_local_roundtrip(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(local_roundtrip_mutex_worker, args=(nprocs,), nprocs=nprocs)


def shift_right_worker(rank, nrank):
    win, src, out = _shift_setup(rank, nrank)
    shift_load_right(win, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float((rank - 1) % nrank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] shift_load_right got {out.cpu()[0, :3].tolist()}..., "
        f"expected rank {(rank - 1) % nrank}'s data"
    )
    print(f"[rank {rank}] shift_load_right ok", flush=True)
    win.close()
    dist.destroy_process_group()


def shift_left_worker(rank, nrank):
    win, src, out = _shift_setup(rank, nrank)
    shift_load_left(win, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float((rank + 1) % nrank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] shift_load_left got {out.cpu()[0, :3].tolist()}..., "
        f"expected rank {(rank + 1) % nrank}'s data"
    )
    print(f"[rank {rank}] shift_load_left ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_shift_load_right(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(shift_right_worker, args=(nprocs,), nprocs=nprocs)


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_shift_load_left(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(shift_left_worker, args=(nprocs,), nprocs=nprocs)


_WAIT_TIMEOUT = (1 << 31) - 1
_CH_UB_ADDR = 0x2000  # channel UB scratch (>= 256 B); tiles live below 0x1000 in these kernels


@pl.jit(auto_mutex=True)
def shift_load_right_urma(win: pld.Window, src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                          out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)  # rides the peer's jetty
        ch.write(dt, src, submit=False)  # stage the WQE on the peer jetty
        ch.submit()                      # doorbell: publish the staged WQE
        ch.drain()                       # engine completion (visible remotely)
        pld.barrier(win)                        # every rank pushed before any pull
        pld.get(t2, dt, [0, 0], me)             # read back my own window
        pl.store(out, t2, [0, 0])


@pl.jit(auto_mutex=True)
def shift_load_left_urma(win: pld.Window, src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                         out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        peer = (me + nr - 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)
        ch.write(dt, src, submit=False)
        ch.submit()
        ch.drain()
        pld.barrier(win)
        pld.get(t2, dt, [0, 0], me)
        pl.store(out, t2, [0, 0])


@pl.jit(auto_mutex=True)
def write_submit_kernel(win: pld.Window, src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                        out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    """Canonical handshake: submit=True + drain, then notify the consumer and wait."""
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        sig_done = pld.signal(win, 0)
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)
        ch.write(dt, src, submit=True)   # publish immediately
        ch.drain()                       # release: data visible before the signal
        pld.notify(sig_done, peer, 1, op=pld.NotifyOp.ATOMIC_ADD)
        pld.wait(sig_done, 1, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)
        pld.barrier(win)
        pld.get(t2, dt, [0, 0], me)
        pl.store(out, t2, [0, 0])


def shift_right_urma_worker(rank, nrank):
    win, src, out = _shift_setup(rank, nrank)
    shift_load_right_urma[None, 1](win, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float((rank - 1) % nrank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] shift_load_right_urma got {out.cpu()[0, :3].tolist()}..., "
        f"expected rank {(rank - 1) % nrank}'s data"
    )
    print(f"[rank {rank}] shift_load_right_urma ok", flush=True)
    win.close()
    dist.destroy_process_group()


def shift_left_urma_worker(rank, nrank):
    win, src, out = _shift_setup(rank, nrank)
    shift_load_left_urma[None, 1](win, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float((rank + 1) % nrank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] shift_load_left_urma got {out.cpu()[0, :3].tolist()}..., "
        f"expected rank {(rank + 1) % nrank}'s data"
    )
    print(f"[rank {rank}] shift_load_left_urma ok", flush=True)
    win.close()
    dist.destroy_process_group()


def write_submit_worker(rank, nrank):
    win, src, out = _shift_setup(rank, nrank)
    write_submit_kernel[None, 1](win, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float((rank - 1) % nrank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] write_submit got {out.cpu()[0, :3].tolist()}..., "
        f"expected rank {(rank - 1) % nrank}'s data"
    )
    print(f"[rank {rank}] write_submit ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_shift_load_right_urma(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(shift_right_urma_worker, args=(nprocs,), nprocs=nprocs)


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_shift_load_left_urma(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(shift_left_urma_worker, args=(nprocs,), nprocs=nprocs)


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_submit(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_submit_worker, args=(nprocs,), nprocs=nprocs)


_WV_INT64_BASE = 0     # one int64 word per rank
_WV_FP32_BASE = 1024   # one fp32 word per rank, no overlap with the int64 region


@pl.jit(auto_mutex=True)
def write_value_kernel(win: pld.Window, out: pl.Tensor[[1], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)

        # int64 word at a runtime byte offset: stage the inline WQE, then publish it
        dt_i64 = pld.make_dtensor(win, _WV_INT64_BASE + me * 8, [1], pl.DT_INT64)
        ch.write(dt_i64, me + 100, submit=False)
        ch.submit()

        # fp32 word: literal value, published immediately
        dt_f32 = pld.make_dtensor(win, _WV_FP32_BASE + me * 4, [1], pl.DT_FP32)
        ch.write(dt_f32, 1.5)

        ch.drain()
        pld.barrier(win)
        out[0] = me


def write_value_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=1 << 20)
    out = torch.full((1,), -1, dtype=torch.int64).npu()

    write_value_kernel[None, 1](win, out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [rank], f"[rank {rank}] write_value kernel did not complete"

    # Each rank writes one word to its successor, so this window must hold the previous
    # rank's word in that rank's slot. The payload is uninitialized otherwise, so check
    # only the word we own an expectation for.
    prev = (rank - 1) % nrank
    got_i64 = win.device_buffer[_WV_INT64_BASE:_WV_INT64_BASE + 8 * nrank].view(torch.int64).cpu()
    assert got_i64[prev].item() == prev + 100, (
        f"[rank {rank}] int64 word {prev} got {got_i64.tolist()}, expected {prev + 100}"
    )

    got_f32 = win.device_buffer[_WV_FP32_BASE:_WV_FP32_BASE + 4 * nrank].view(torch.float32).cpu()
    assert got_f32[prev].item() == 1.5, (
        f"[rank {rank}] fp32 word {prev} got {got_f32.tolist()}, expected 1.5"
    )
    print(f"[rank {rank}] write_value ok: {got_i64.tolist()} / {got_f32.tolist()}", flush=True)

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_value(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_value_worker, args=(nprocs,), nprocs=nprocs)


@pl.jit(auto_mutex=True)
def write_notify_kernel(win: pld.Window, src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                        out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    """Fused write + hardware notify: the signal fires on payload completion, with no drain."""
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        sig_done = pld.signal(win, 0)
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)
        ch.write_notify(dt, src, sig_done, me + 1)  # hardware writes sig_done on completion
        prev = (me + nr - 1) % nr
        pld.wait(sig_done, prev + 1, cmp=pld.Cmp.EQ, timeout=_WAIT_TIMEOUT)
        pld.get(t2, dt, [0, 0], me)                        # my window now holds the previous rank's data
        pl.store(out, t2, [0, 0])
        pld.barrier(win)                                   # all writes completed before any rank exits


def write_notify_worker(rank, nrank):
    win, src, out = _shift_setup(rank, nrank)
    write_notify_kernel[None, 1](win, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float((rank - 1) % nrank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] write_notify got {out.cpu()[0, :3].tolist()}..., "
        f"expected rank {(rank - 1) % nrank}'s data"
    )
    # The hardware notify wrote the producer's value into this rank's slot 0 (the signal area
    # starts at win.size(), one 512 B slot per signal).
    sig = win.device_buffer[win.size():win.size() + 8].view(torch.int64).cpu().item()
    expect_sig = (rank - 1) % nrank + 1
    assert sig == expect_sig, f"[rank {rank}] signal slot got {sig}, expected {expect_sig}"
    print(f"[rank {rank}] write_notify ok (signal {sig})", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_notify(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_notify_worker, args=(nprocs,), nprocs=nprocs)


@pl.jit(auto_mutex=True)
def channel_exclusive_per_peer_kernel(win: pld.Window, out: pl.Tensor[[1], pl.DT_INT64]):
    """Per-peer spelling: the channel rides the bound peer's jetty (is_exclusive() == 0)."""
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB_ADDR, peer=(me + 1) % nr)
        if ch.is_exclusive():
            out[0] = 1
        else:
            out[0] = 0


@pl.jit(auto_mutex=True)
def channel_exclusive_multi_issuer_kernel(win: pld.Window, out: pl.Tensor[[1], pl.DT_INT64]):
    """Multi-issuer spelling: no peer, the channel rides this AIV's own jetty (is_exclusive() == 1)."""
    with pl.section_vector():
        ch = pld.channel(win, _CH_UB_ADDR)
        if ch.is_exclusive():
            out[0] = 1
        else:
            out[0] = 0


def channel_exclusive_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=4096)
    out = torch.full((1,), -1, dtype=torch.int64).npu()

    # Pick the spelling the topology supports; the device answer must match the host query.
    kernel = channel_exclusive_multi_issuer_kernel if win.support_multi_issuer() \
        else channel_exclusive_per_peer_kernel
    kernel[None, 1](win, out)
    torch.npu.synchronize()
    got = out.cpu().tolist()[0]
    expect = 1 if win.support_multi_issuer() else 0
    assert got == expect, f"[rank {rank}] is_exclusive got {got}, expected {expect}"
    print(f"[rank {rank}] channel_exclusive ok ({got})", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_channel_exclusive(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(channel_exclusive_worker, args=(nprocs,), nprocs=nprocs)


@pl.jit
def notify_wait_kernel(win: pld.Window, out: pl.Tensor[[2], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        pl.printf("rank %d notify_wait: nrank %d\n", me, nr)

        sig_ring = pld.signal(win, 4)
        pld.notify(sig_ring, (me + 1) % nr, 1, op=pld.NotifyOp.ATOMIC_ADD)
        pld.wait(sig_ring, 1, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)

        sig_counter = pld.signal(win, 7)
        pld.notify(sig_counter, 0, 1, op=pld.NotifyOp.ATOMIC_ADD)

        sig_local = pld.signal(win, 5)
        pld.notify(sig_local, 5)
        pld.wait(sig_local, 5, cmp=pld.Cmp.EQ, timeout=_WAIT_TIMEOUT)

        sig_cmp = pld.signal(win, 6)
        pld.notify(sig_cmp, 5)
        pld.wait(sig_cmp, 5, cmp=pld.Cmp.EQ, timeout=_WAIT_TIMEOUT)
        pld.wait(sig_cmp, 4, cmp=pld.Cmp.NE, timeout=_WAIT_TIMEOUT)
        pld.wait(sig_cmp, 6, cmp=pld.Cmp.LT, timeout=_WAIT_TIMEOUT)
        pld.wait(sig_cmp, 5, cmp=pld.Cmp.LE, timeout=_WAIT_TIMEOUT)
        pld.wait(sig_cmp, 4, cmp=pld.Cmp.GT, timeout=_WAIT_TIMEOUT)
        pld.wait(sig_cmp, 5, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)

        pld.barrier(win)
        out[0] = 1
        out[1] = 2


def notify_wait_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=1 << 20)
    out = torch.full((2,), -1, dtype=torch.int64).npu()

    time.sleep(rank * 0.1)  # slight stagger: the waits must block until peers arrive
                            # (well under the timeout; 0.5 s/rank used to outlast it)
    notify_wait_kernel[None, 1](win, out)  # one AIV: the only spinner, no bid guards
    torch.npu.synchronize()
    assert out.tolist() == [1, 2], f"[rank {rank}] notify_wait kernel did not complete: {out.tolist()}"

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_notify_wait(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(notify_wait_worker, args=(nprocs,), nprocs=nprocs)


@pl.jit
def notify_all_kernel(win: pld.Window, out: pl.Tensor[[1], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        pl.printf("rank %d notify_all: nrank %d\n", me, nr)

        sig_add = pld.signal(win, 3)
        pld.notify_all(sig_add, 1, op=pld.NotifyOp.ATOMIC_ADD)
        pld.wait(sig_add, nr, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)

        sig_set = pld.signal(win, 4)
        pld.notify_all(sig_set, 5)
        pld.wait(sig_set, 5, cmp=pld.Cmp.EQ, timeout=_WAIT_TIMEOUT)
        out[0] = 1


def notify_all_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=1 << 20)
    out = torch.full((1,), -1, dtype=torch.int64).npu()

    notify_all_kernel[None, 1](win, out)  # one AIV: its leader is the only signaler
    torch.npu.synchronize()
    assert out.tolist() == [1], f"[rank {rank}] notify_all kernel did not complete: {out.tolist()}"
    print(f"[rank {rank}] notify_all ok", flush=True)

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_notify_all(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(notify_all_worker, args=(nprocs,), nprocs=nprocs)


_AG_SHARD = 1024    # per-rank shard (4096 elements over 4 ranks; offsets stay runtime)


@pl.jit(auto_mutex=True)
def all_gather_signals_kernel(win: pld.Window, src: pl.Tensor[[1, _AG_SHARD], pl.DT_BF16]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        tile_type = pl.TileType(shape=[1, _AG_SHARD], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        t = tg.current()
        sig_done = pld.signal(win, 0)
        pl.load(t, src, [0, 0])
        dt = pld.make_dtensor(win, 0, [nr, _AG_SHARD], pl.DT_BF16)
        for r in pl.range(nr):
            pld.put(dt, t, [me, 0], r)
            pld.notify(sig_done, r, 1, op=pld.NotifyOp.ATOMIC_ADD)
        pld.wait(sig_done, nr, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)


def all_gather_signals_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=8192)
    src = torch.full((1, _AG_SHARD), float(rank), dtype=torch.bfloat16).npu()

    all_gather_signals_kernel[None, 1](win, src)
    torch.npu.synchronize()

    gathered = win.device_buffer[:8192].view(torch.bfloat16).cpu().view(1, 4096)
    expect = torch.cat(
        [torch.full((1, _AG_SHARD), float(r), dtype=torch.bfloat16) for r in range(nrank)], dim=1
    )
    assert torch.equal(gathered, expect), (
        f"[rank {rank}] all_gather window got {gathered[0, :4].tolist()}...{gathered[0, -4:].tolist()}, "
        f"expected shards 0..{nrank - 1}"
    )

    # cross-check the same collective through torch.distributed
    ref = [torch.empty((1, _AG_SHARD), dtype=torch.bfloat16).npu() for _ in range(nrank)]
    dist.all_gather(ref, src)
    ref_t = torch.cat([t.cpu() for t in ref], dim=1)
    assert torch.equal(gathered, ref_t), f"[rank {rank}] mismatch vs dist.all_gather"
    print(f"[rank {rank}] all_gather_signals ok (matches dist.all_gather)", flush=True)

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_all_gather_signals(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(all_gather_signals_worker, args=(nprocs,), nprocs=nprocs)


_AR_SHARD = 512  # per-rank shard (bf16); one atomic-add put lands in every rank's window


# All-reduce sum through atomic puts: every rank accumulates its shard into every
# rank's window (atomic_mode=AtomicAdd), then reads the total back from its own.
@pl.jit(auto_mutex=True)
def all_reduce_put_kernel(win: pld.Window, src: pl.Tensor[[1, _AR_SHARD], pl.DT_BF16],
                          out: pl.Tensor[[1, _AR_SHARD], pl.DT_BF16]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _AR_SHARD], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
        tg_a = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        tg_b = pl.make_tile_group(type=tile_type, addrs=_AR_SHARD * 2, mutex_ids=[1])
        t = tg_a.current()
        t2 = tg_b.current()
        me, nr = win.rank(), win.nrank()
        pl.load(t, src, [0, 0])
        dt = pld.make_dtensor(win, 0, [1, _AR_SHARD], pl.DT_BF16)
        pld.barrier(win)  # every rank's host-side payload zeroing is complete
        for r in pl.range(nr):  # accumulate my shard into every rank's window
            pld.put(dt, t, [0, 0], r, atomic_mode=pl.AtomicType.AtomicAdd)
        pld.barrier(win)  # all adds have landed before any rank reads back
        pld.get(t2, dt, [0, 0], me)
        pl.store(out, t2, [0, 0])


def all_reduce_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=8192)
    # AtomicAdd accumulates, so zero the payload before the launch.
    win.device_buffer[:win.size()].copy_(torch.zeros(win.size(), dtype=torch.uint8))
    src = torch.full((1, _AR_SHARD), float(rank), dtype=torch.bfloat16).npu()
    out = torch.zeros((1, _AR_SHARD), dtype=torch.bfloat16).npu()

    all_reduce_put_kernel[None, 1](win, src, out)  # one AIV: redundant puts would multiply the sum
    torch.npu.synchronize()

    # cross-check the same collective through torch.distributed
    ref = src.clone()
    dist.all_reduce(ref)
    assert torch.equal(out.cpu(), ref.cpu()), (
        f"[rank {rank}] all_reduce got {out.cpu()[0, :4].tolist()}..., "
        f"expected {ref.cpu()[0, :4].tolist()}..."
    )
    print(f"[rank {rank}] all_reduce ok (sum {ref.cpu()[0, 0].item()})", flush=True)

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_all_reduce(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(all_reduce_worker, args=(nprocs,), nprocs=nprocs)


_AT_INT64 = 0        # [add, exch, cas, failing cas, contended counter] @ 0,8,16,24,32
_AT_UINT64 = 40      # [add, exch, cas] @ 40,48,56
_AT_INT32 = 64       # [add, exch, cas] @ 64,68,72
_AT_UINT32 = 76      # [add, exch, cas] @ 76,80,84
_AT_FP32 = 88        # [add, exch, cas] @ 88,92,96
_AT_FLAT = 128       # 2-D view [4, 4]
_AT_FLAT_WORD = 168  # scalar offset 5 -> linear element 5 = byte 128 + 5*8
_AT_FLAT_2D_WORD = 176   # coordinate [1, 2] -> linear element 6 = byte 128 + 6*8
_AT_FLAT_2D_DECOY = 136  # element 1: must stay 0 if [1, 2] is flattened, not read as 1
_AT_NESTED = 200     # direct `out[i] = pld.atomic_add(...)` (setval-nested) path
_AT_SUB = 256        # int64 [7]: const 5, const -3, runtime offset/value
_AT_SUB_U64 = 320    # uint64 modular wrap
_AT_SUB_F32 = 328    # fp32


@pl.jit
def atomic_kernel(win: pld.Window, out: pl.Tensor[[6], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        peer = (me + 1) % nr

        pld.barrier(win)  # every rank's host-side payload zeroing is complete

        i64 = pld.make_dtensor(win, _AT_INT64, [5], pl.DT_INT64)
        old_add = pld.atomic_add(i64, peer, [0], 1)
        old_exch = pld.atomic_exchange(i64, peer, [1], me + 100)
        old_cas = pld.atomic_cas(i64, peer, [2], 0, me + 200)
        old_fail = pld.atomic_cas(i64, peer, [3], 0xDEAD, 123)

        u64 = pld.make_dtensor(win, _AT_UINT64, [3], pl.DT_UINT64)
        pld.atomic_add(u64, peer, [0], 1)
        pld.atomic_exchange(u64, peer, [1], me + 300)
        pld.atomic_cas(u64, peer, [2], 0, me + 400)

        i32 = pld.make_dtensor(win, _AT_INT32, [3], pl.DT_INT32)
        pld.atomic_add(i32, peer, [0], 1)
        pld.atomic_exchange(i32, peer, [1], me + 500)
        pld.atomic_cas(i32, peer, [2], 0, me + 600)

        u32 = pld.make_dtensor(win, _AT_UINT32, [3], pl.DT_UINT32)
        pld.atomic_add(u32, peer, [0], 1)
        pld.atomic_exchange(u32, peer, [1], me + 700)
        pld.atomic_cas(u32, peer, [2], 0, me + 800)

        f32 = pld.make_dtensor(win, _AT_FP32, [3], pl.DT_FP32)
        pld.atomic_add(f32, peer, [0], 1.5)
        pld.atomic_exchange(f32, peer, [1], 2.25)
        pld.atomic_cas(f32, peer, [2], 0.0, 3.5)

        # Scalar offset form: linear element 5 inside a [4, 4] view (never [5, 0]).
        flat = pld.make_dtensor(win, _AT_FLAT, [4, 4], pl.DT_INT64)
        pld.atomic_add(flat, peer, 5, 1)
        # Coordinate form on the same 2-D view: [1, 2] flattens to linear element 6.
        pld.atomic_add(flat, peer, [1, 2], 2)
        # Runtime offset expressions are legal: coordinates and the scalar linear form.
        pld.atomic_add(flat, peer, [me, 3], 3)   # peer's element me*4 + 3
        pld.atomic_add(flat, peer, me * 4, 4)    # scalar linear expression: element me*4

        # atomic_sub is a frontend wrapper over atomic_add(-val).
        sub = pld.make_dtensor(win, _AT_SUB, [7], pl.DT_INT64)
        pld.atomic_sub(sub, peer, [0], 5)        # element 0 -> -5
        pld.atomic_sub(sub, peer, [1], -3)       # element 1 -> +3 (double negation)
        pld.atomic_sub(sub, peer, me + 2, 1)     # runtime scalar offset -> element me+2
        pld.atomic_sub(sub, peer, [6], me)       # runtime value -> element 6 = -me

        sub_u64 = pld.make_dtensor(win, _AT_SUB_U64, [1], pl.DT_UINT64)
        pld.atomic_sub(sub_u64, peer, [0], 1)    # 0 - 1 modulo 2^64

        sub_f32 = pld.make_dtensor(win, _AT_SUB_F32, [1], pl.DT_FP32)
        pld.atomic_sub(sub_f32, peer, [0], 0.5)  # -0.5

        # Contended counter: every rank adds 1 to rank 0's word.
        shared = pld.make_dtensor(win, 32, [1], pl.DT_INT64)
        old_shared = pld.atomic_add(shared, 0, [0], 1)

        # Nested value position: the atomic is an argument of block.setval, not a bound var.
        nested = pld.make_dtensor(win, _AT_NESTED, [1], pl.DT_INT64)
        out[5] = pld.atomic_add(nested, peer, [0], 7)

        pld.barrier(win)  # all atomics have landed before any rank reads its window

        out[0] = old_add
        out[1] = old_exch
        out[2] = old_cas
        out[3] = old_fail
        out[4] = old_shared


def atomic_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group, size=4096)
    # Zero only the payload before launch; the signal/barrier regions are framework-managed.
    win.device_buffer[:win.size()].copy_(torch.zeros(win.size(), dtype=torch.uint8))
    out = torch.full((6,), -1, dtype=torch.int64).npu()

    atomic_kernel[None, 1](win, out)
    torch.npu.synchronize()

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_atomics(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(atomic_worker, args=(nprocs,), nprocs=nprocs)


if __name__ == '__main__':
    test_create_window()
    test_window_probe()
    test_window_barrier()
    test_local_roundtrip()
    test_shift_load_right()
    test_shift_load_left()
    test_shift_load_right_urma()
    test_shift_load_left_urma()
    test_write_submit()
    test_write_value()
    test_write_notify()
    test_channel_exclusive()
    test_notify_wait()
    test_all_gather_signals()
    test_all_reduce()
    test_atomics()
