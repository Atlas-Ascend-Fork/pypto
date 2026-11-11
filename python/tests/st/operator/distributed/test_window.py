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

_WS_BYTES = 1 << 20  # payload region registered for the remote-transfer tests
_PROTOCOL = pld.Protocol.UB_MEM | pld.Protocol.URMA  # the workspace is used by both tiers
_ST_MAX_RANKS = 4  # world size of the distributed ST launches; sizes static readback buffers


def init_dist(rank, nrank):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    dist.init_process_group(backend='hccl', init_method=f'tcp://{ip}:{port}', world_size=nrank, rank=rank)
    return dist.distributed_c10d._get_default_group()


def worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    print(str(win), flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_create_window(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(worker, args=(nprocs,), nprocs=nprocs)


@pl.jit
def probe(win: pld.Window, out: pl.Tensor[[2], pl.DT_INT64]):
    with pl.section_vector():
        if pl.get_block_idx() == 0:  # one writer / one printf line per rank
            pl.printf("rank %d / %d\n", win.rank(), win.nrank())
            out[0] = win.rank()
            out[1] = win.nrank()


def probe_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    ws = torch.zeros(4096, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    win.connect()
    # Fill on CPU + H2D copy: some CANN installs miss the NPU Fill operator package
    # (EZ1013 on aclnnInplaceFillScalar), which is unrelated to the code under test.
    out = torch.full((2,), -1, dtype=torch.int64).npu()
    probe(win, out)
    torch.npu.synchronize()
    expect = [win.rank(), win.nrank()]
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


def layer_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)

    # Explicit layer: take the first id dump_topology reports (0 = intra-server mesh).
    topo = pld.dump_topology(group)
    layer = int(topo.split("Network Layer: ")[1].split()[0])
    win = pld.create_window(group, name="layer_probe", layer=layer)
    win.connect()  # layer selection happens at connect, not at creation
    assert f"layer {layer}" in str(win), f"[rank {rank}] window not built on layer {layer}:\n{win}"

    out = torch.full((2,), -1, dtype=torch.int64).npu()
    probe(win, out)
    torch.npu.synchronize()
    expect = [win.rank(), win.nrank()]
    assert out.cpu().tolist() == expect, f"[rank {rank}] probe got {out.cpu().tolist()}, expected {expect}"
    print(f"[rank {rank}] explicit layer ok: layer {layer}", flush=True)
    win.close()

    # A layer outside the rank graph must be rejected on every rank.
    try:
        win = pld.create_window(group, layer=9999)
        win.connect()
        raise AssertionError(f"[rank {rank}] invalid layer 9999 accepted")
    except RuntimeError as e:
        assert "not in rank graph" in str(e), f"[rank {rank}] unexpected error: {e}"
    print(f"[rank {rank}] invalid layer rejected", flush=True)

    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_create_window_layer(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(layer_worker, args=(nprocs,), nprocs=nprocs)


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
    win = pld.create_window(group)
    win.connect()
    out = torch.full((3,), -1, dtype=torch.int64).npu()

    time.sleep(rank * 0.5)  # stagger launches: the barrier must block until the last rank arrives

    barrier_kernel(win, out)
    torch.npu.synchronize()

    assert out.cpu().tolist() == [1, 2, 3], f"[rank {rank}] markers got {out.cpu().tolist()}"

    # Sense-reversal end state after 3 barriers (counter @ 0, phase signals @ 8):
    # i0 counts signals[0] up to nrank, i1 counts signals[1] up to nrank, i2 counts
    # signals[0] back down to 0 — host-visible proof that every rank's atomics landed.
    region = win.control_buffer[0:16]  # the Barrier is the first control-region slot
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
def shift_load_right(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                     src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                     out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tile_type, addr=0)
        t2 = pl.make_tile(tile_type, addr=_SHIFT_N * 2)
        pl.load(t, src, [0, 0])
        me = win.rank()
        nr = win.nrank()
        dt = pld.make_dtensor(win, ws, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
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
def shift_load_left(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                    src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
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
        dt = pld.make_dtensor(win, ws, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        pld.put(dt, t, [0, 0], (me + nr - 1) % nr)  # push my tile into the PREVIOUS rank's window
        pld.barrier(win)
        pld.get(t2, dt, [0, 0], me)  # my own window now holds the NEXT rank's push
        pl.store(out, t2, [0, 0])


def _shift_setup(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    win.connect()
    src = torch.full((1, _SHIFT_N), float(rank), dtype=torch.bfloat16).npu()
    out = torch.zeros((1, _SHIFT_N), dtype=torch.bfloat16).npu()
    return win, ws, src, out


@pl.jit(auto_mutex=True)
def local_roundtrip_mutex(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                          src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                          out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg_a = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        tg_b = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[1])
        t = tg_a.current()
        t2 = tg_b.current()
        pl.load(t, src, [0, 0])
        me = win.rank()
        dt = pld.make_dtensor(win, ws, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        pld.put(dt, t, [0, 0], me)  # peer == own rank: local GM copy through the window
        pld.barrier(win)
        pld.get_tile(t2, dt, [0, 0], me)
        pl.store(out, t2, [0, 0])


def local_roundtrip_mutex_worker(rank, nrank):
    win, ws, src, out = _shift_setup(rank, nrank)
    local_roundtrip_mutex(win, ws, src, out)
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
    win, ws, src, out = _shift_setup(rank, nrank)
    shift_load_right(win, ws, src, out)
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
    win, ws, src, out = _shift_setup(rank, nrank)
    shift_load_left(win, ws, src, out)
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
def shift_load_right_urma(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                          src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                          out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, ws, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)  # rides the peer's jetty
        ch.write(dt, src, submit=False)  # stage the WQE on the peer jetty
        ch.submit()                      # doorbell: publish the staged WQE
        ch.drain()                       # engine completion (visible remotely)
        pld.barrier(win)                        # every rank pushed before any pull
        pld.get(t2, dt, [0, 0], me)             # read back my own window
        pl.store(out, t2, [0, 0])


@pl.jit(auto_mutex=True)
def shift_load_left_urma(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                         src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                         out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, ws, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        peer = (me + nr - 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)
        ch.write(dt, src, submit=False)
        ch.submit()
        ch.drain()
        pld.barrier(win)
        pld.get(t2, dt, [0, 0], me)
        pl.store(out, t2, [0, 0])


@pl.jit(auto_mutex=True)
def write_submit_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                        src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                        out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    """Canonical handshake: submit=True + drain, then notify the consumer and wait.

    The consumer reads its window back between the wait and the trailing barrier so the
    check depends on the wait alone — not on the barrier or the kernel-exit sync.
    """
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, ws, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
        sig_done = pld.signal(win, 0)
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)
        ch.write(dt, src, submit=True)   # publish immediately
        ch.drain()                       # release: data visible before the signal
        pld.notify(sig_done, peer, 1, op=pld.NotifyOp.ATOMIC_ADD)
        pld.wait(sig_done, 1, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)
        pld.get(t2, dt, [0, 0], me)      # the signal already guarantees the payload
        pl.store(out, t2, [0, 0])
        pld.barrier(win)


def shift_right_urma_worker(rank, nrank):
    win, ws, src, out = _shift_setup(rank, nrank)
    shift_load_right_urma[None, 1](win, ws, src, out)
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
    win, ws, src, out = _shift_setup(rank, nrank)
    shift_load_left_urma[None, 1](win, ws, src, out)
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
    win, ws, src, out = _shift_setup(rank, nrank)
    write_submit_kernel[None, 1](win, ws, src, out)
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


_WI_ROWS = 8
_WI_ELEMS = 16
_WI_SLOT = _WI_ROWS * _WI_ELEMS * 2          # 256 B per sub-case, three slots
_WI_IDX_SRC_UB = 0x1000                      # channel scratch starts at 0x2000
_WI_IDX_DST_UB = 0x1400
_WI_PERM_A = (3, 1, 7, 5, 0, 6, 2, 4)        # gather: source row ids
_WI_PERM_B = (0, 5, 2, 7, 1, 4, 6, 3)        # scatter: destination row ids (unique)
_WI_BUF = 1 << 20                            # payload region every rank registers as "wi"


def _wi_setup(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    buf = torch.zeros(_WI_BUF, dtype=torch.uint8, device='npu')
    pld.register_memory(win, buf, "wi", _PROTOCOL)
    win.connect()
    return win, buf


@pl.jit(auto_mutex=False)
def write_indexed_kernel(win: pld.Window,
                         wi: pl.Tensor[[_WI_BUF], pl.DT_UINT8],
                         src: pl.Tensor[[_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE],
                         src_idx: pl.Tensor[[1, _WI_ROWS], pl.DT_INT32],
                         dst_idx: pl.Tensor[[1, _WI_ROWS], pl.DT_INT32],
                         out: pl.Tensor[[_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE]):
    with pl.section_vector():
        it_src = pl.make_tile(
            pl.TileType(shape=[1, _WI_ROWS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=_WI_IDX_SRC_UB)
        it_dst = pl.make_tile(
            pl.TileType(shape=[1, _WI_ROWS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=_WI_IDX_DST_UB)
        ret = pl.make_tile(
            pl.TileType(shape=[_WI_ROWS, _WI_ELEMS], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec),
            addr=0x800)
        pl.load(it_src, src_idx, [0, 0])
        pl.load(it_dst, dst_idx, [0, 0])
        # The index tables are read by the scalar pipe inside write_indexed.
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)

        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB_ADDR, peer=(me + 1) % nr)
        dt_g = pld.make_dtensor(win, wi, 0, [_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        dt_s = pld.make_dtensor(win, wi, _WI_SLOT, [_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        dt_gs = pld.make_dtensor(win, wi, 2 * _WI_SLOT, [_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE)

        ch.write_indexed(dt_g, src, src_indices=it_src, count=_WI_ROWS)
        ch.drain()
        ch.write_indexed(dt_s, src, dst_indices=it_dst, count=_WI_ROWS)
        ch.drain()
        ch.write_indexed(dt_gs, src, src_indices=it_src, dst_indices=it_dst, count=_WI_ROWS, submit=False)
        ch.submit()
        ch.drain()
        pld.barrier(win)
        pld.get(ret, dt_g, [0, 0], me)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.store(out, ret, [0, 0])


def write_indexed_worker(rank, nrank):
    torch.npu.set_device(rank)
    win, buf = _wi_setup(rank, nrank)
    rows = torch.arange(_WI_ROWS, dtype=torch.float32) + rank * 10
    src = rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16).npu()
    src_idx = torch.tensor(_WI_PERM_A, dtype=torch.int32).view(1, _WI_ROWS).npu()
    dst_idx = torch.tensor(_WI_PERM_B, dtype=torch.int32).view(1, _WI_ROWS).npu()
    out = torch.zeros((_WI_ROWS, _WI_ELEMS), dtype=torch.bfloat16).npu()

    write_indexed_kernel[None, 1](win, buf, src, src_idx, dst_idx, out)
    torch.npu.synchronize()

    prev = (rank - 1) % nrank
    prev_rows = torch.arange(_WI_ROWS, dtype=torch.float32) + prev * 10
    prev_src = prev_rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16)
    perm_a = torch.tensor(_WI_PERM_A)
    perm_b = torch.tensor(_WI_PERM_B)
    exp_g = prev_src[perm_a]
    exp_s = torch.zeros_like(prev_src)
    exp_s[perm_b] = prev_src
    exp_gs = torch.zeros_like(prev_src)
    exp_gs[perm_b] = prev_src[perm_a]

    payload = buf.view(torch.bfloat16).view(-1, _WI_ELEMS).cpu()
    assert torch.equal(payload[0:_WI_ROWS], exp_g), f"[rank {rank}] gather slot mismatch"
    assert torch.equal(payload[_WI_ROWS:2 * _WI_ROWS], exp_s), f"[rank {rank}] scatter slot mismatch"
    assert torch.equal(payload[2 * _WI_ROWS:3 * _WI_ROWS], exp_gs), f"[rank {rank}] gather-scatter slot mismatch"
    assert torch.equal(out.cpu(), exp_g), f"[rank {rank}] device readback mismatch"
    print(f"[rank {rank}] write_indexed ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_indexed(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_indexed_worker, args=(nprocs,), nprocs=nprocs)


_WI_REPEAT_LAUNCHES = 3
_WI_DRAIN_TIMEOUT = 2_000_000_000  # device cycles; a stalled CQE fails the test instead of hanging


@pl.jit(auto_mutex=False)
def write_indexed_repeat_kernel(win: pld.Window,
                                wi: pl.Tensor[[_WI_BUF], pl.DT_UINT8],
                                src: pl.Tensor[[_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE],
                                src_idx: pl.Tensor[[1, _WI_ROWS], pl.DT_INT32],
                                out: pl.Tensor[[_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE]):
    with pl.section_vector():
        it_src = pl.make_tile(
            pl.TileType(shape=[1, _WI_ROWS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=_WI_IDX_SRC_UB)
        ret = pl.make_tile(
            pl.TileType(shape=[_WI_ROWS, _WI_ELEMS], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec),
            addr=0x800)
        pl.load(it_src, src_idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)

        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB_ADDR, peer=(me + 1) % nr)
        dt = pld.make_dtensor(win, wi, 0, [_WI_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        ch.write_indexed(dt, src, src_indices=it_src, count=_WI_ROWS)
        ch.drain(_WI_DRAIN_TIMEOUT)
        pld.barrier(win, _WI_DRAIN_TIMEOUT)
        pld.get(ret, dt, [0, 0], me)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.store(out, ret, [0, 0])


@pl.jit(auto_mutex=False)
def write_indexed_one_kernel(win: pld.Window,
                             wi: pl.Tensor[[_WI_BUF], pl.DT_UINT8],
                             src: pl.Tensor[[1, _WI_ELEMS], _SHIFT_DTYPE],
                             src_idx: pl.Tensor[[1, _WI_ROWS], pl.DT_INT32],
                             out: pl.Tensor[[1, _WI_ELEMS], _SHIFT_DTYPE]):
    with pl.section_vector():
        it_src = pl.make_tile(
            pl.TileType(shape=[1, _WI_ROWS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=_WI_IDX_SRC_UB)
        ret = pl.make_tile(
            pl.TileType(shape=[1, _WI_ELEMS], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec),
            addr=0x800)
        pl.load(it_src, src_idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)

        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB_ADDR, peer=(me + 1) % nr)
        dt = pld.make_dtensor(win, wi, 0, [1, _WI_ELEMS], _SHIFT_DTYPE)
        # One indexed row per launch, so consecutive launches advance the SQ head
        # by one and the second launch reuses physical slot 1.
        ch.write_indexed(dt, src, src_indices=it_src, count=1)
        ch.drain(_WI_DRAIN_TIMEOUT)
        pld.barrier(win, _WI_DRAIN_TIMEOUT)
        pld.get(ret, dt, [0, 0], me)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.store(out, ret, [0, 0])


def write_indexed_repeat_worker(rank, nrank):
    """Relaunch the indexed writer on the same window/channel and recheck the payload.

    The single-launch tests cannot catch SQ/CQ accounting that only breaks when the
    packed head and completion counters carry over from a previous launch.
    """
    torch.npu.set_device(rank)
    win, buf = _wi_setup(rank, nrank)
    src_idx = torch.tensor(_WI_PERM_A, dtype=torch.int32).view(1, _WI_ROWS).npu()
    out = torch.zeros((_WI_ROWS, _WI_ELEMS), dtype=torch.bfloat16).npu()
    perm_a = torch.tensor(_WI_PERM_A)
    prev = (rank - 1) % nrank

    for it in range(_WI_REPEAT_LAUNCHES):
        rows = torch.arange(_WI_ROWS, dtype=torch.float32) + rank * 10 + it * 100
        src = rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16).npu()
        write_indexed_repeat_kernel[None, 1](win, buf, src, src_idx, out)
        torch.npu.synchronize()

        prev_rows = torch.arange(_WI_ROWS, dtype=torch.float32) + prev * 10 + it * 100
        exp = prev_rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16)[perm_a]
        payload = buf.view(torch.bfloat16).view(-1, _WI_ELEMS).cpu()
        assert torch.equal(payload[0:_WI_ROWS], exp), f"[rank {rank}] repeat {it} gather mismatch"
        assert torch.equal(out.cpu(), exp), f"[rank {rank}] repeat {it} readback mismatch"

    # count=1 twice: launch 0 lands in SQ slot 0, launch 1 in slot 1.
    one_idx = torch.zeros((1, _WI_ROWS), dtype=torch.int32).npu()
    one_out = torch.zeros((1, _WI_ELEMS), dtype=torch.bfloat16).npu()
    for it in range(2):
        src = (torch.tensor([[rank * 10 + it * 100]], dtype=torch.float32)
               .repeat(1, _WI_ELEMS).to(torch.bfloat16).npu())
        write_indexed_one_kernel[None, 1](win, buf, src, one_idx, one_out)
        torch.npu.synchronize()

        exp = (torch.tensor([[prev * 10 + it * 100]], dtype=torch.float32)
               .repeat(1, _WI_ELEMS).to(torch.bfloat16))
        payload = buf.view(torch.bfloat16).view(-1, _WI_ELEMS).cpu()
        assert torch.equal(payload[0:1], exp), f"[rank {rank}] count=1 launch {it} mismatch"
        assert torch.equal(one_out.cpu(), exp), f"[rank {rank}] count=1 launch {it} readback mismatch"

    print(f"[rank {rank}] write_indexed repeat ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_indexed_repeat(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_indexed_repeat_worker, args=(nprocs,), nprocs=nprocs)


_WI_DTYPE_ROWS = 64


@pl.jit(auto_mutex=False)
def write_indexed_dtype_kernel(win: pld.Window,
                               wi: pl.Tensor[[_WI_BUF], pl.DT_UINT8],
                               src: pl.Tensor[[_WI_DTYPE_ROWS, _WI_ELEMS], _SHIFT_DTYPE],
                              src_idx: pl.Tensor[[1, _WI_DTYPE_ROWS], pl.DT_INT32],
                              src_idx64: pl.Tensor[[1, _WI_DTYPE_ROWS], pl.DT_INT64],
                              out: pl.Tensor[[_WI_DTYPE_ROWS, _WI_ELEMS], _SHIFT_DTYPE]):
    with pl.section_vector():
        it_src = pl.make_tile(
            pl.TileType(shape=[1, _WI_DTYPE_ROWS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=0x1000)
        it_src64 = pl.make_tile(
            pl.TileType(shape=[1, _WI_DTYPE_ROWS], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
            addr=0x1600)
        ret = pl.make_tile(
            pl.TileType(shape=[_WI_DTYPE_ROWS, _WI_ELEMS], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec),
            addr=0x800)
        pl.load(it_src, src_idx, [0, 0])
        pl.load(it_src64, src_idx64, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)

        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB_ADDR, peer=(me + 1) % nr)
        dt_g = pld.make_dtensor(win, wi, 0, [_WI_DTYPE_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        dt_64 = pld.make_dtensor(win, wi, 2048, [_WI_DTYPE_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        ch.write_indexed(dt_g, src, src_indices=it_src, count=_WI_DTYPE_ROWS)
        ch.drain()
        ch.write_indexed(dt_64, src, src_indices=it_src64, count=_WI_DTYPE_ROWS)
        ch.drain()
        pld.barrier(win)
        pld.get(ret, dt_g, [0, 0], me)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.store(out, ret, [0, 0])


def write_indexed_dtype_worker(rank, nrank):
    torch.npu.set_device(rank)
    win, buf = _wi_setup(rank, nrank)
    rows = torch.arange(_WI_DTYPE_ROWS, dtype=torch.float32) + rank * 10
    src = rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16).npu()
    perm_a = [(i * 37) % _WI_DTYPE_ROWS for i in range(_WI_DTYPE_ROWS)]
    src_idx = torch.tensor(perm_a, dtype=torch.int32).view(1, _WI_DTYPE_ROWS).npu()
    src_idx64 = torch.tensor(perm_a, dtype=torch.int64).view(1, _WI_DTYPE_ROWS).npu()
    out = torch.zeros((_WI_DTYPE_ROWS, _WI_ELEMS), dtype=torch.bfloat16).npu()

    write_indexed_dtype_kernel[None, 1](win, buf, src, src_idx, src_idx64, out)
    torch.npu.synchronize()

    prev = (rank - 1) % nrank
    prev_rows = torch.arange(_WI_DTYPE_ROWS, dtype=torch.float32) + prev * 10
    prev_src = prev_rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16)
    exp = prev_src[torch.tensor(perm_a)]
    payload = buf.view(torch.bfloat16).view(-1, _WI_ELEMS).cpu()
    assert torch.equal(payload[0:_WI_DTYPE_ROWS], exp), f"[rank {rank}] dtype int32 gather slot mismatch"
    assert torch.equal(payload[_WI_DTYPE_ROWS:2 * _WI_DTYPE_ROWS], exp), f"[rank {rank}] dtype int64 slot mismatch"
    assert torch.equal(out.cpu(), exp), f"[rank {rank}] dtype device readback mismatch"
    print(f"[rank {rank}] write_indexed_dtype ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_indexed_dtype(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_indexed_dtype_worker, args=(nprocs,), nprocs=nprocs)


_WI_COUNT_ROWS = 13
_WI_COUNT_IDX = 16
_WI_COUNT_STRIDE = 512


@pl.jit(auto_mutex=False)
def write_indexed_count_kernel(win: pld.Window,
                               wi: pl.Tensor[[_WI_BUF], pl.DT_UINT8],
                               src: pl.Tensor[[_WI_COUNT_ROWS, _WI_ELEMS], _SHIFT_DTYPE],
                               idx: pl.Tensor[[1, _WI_COUNT_IDX], pl.DT_INT32],
                               n0: pl.DT_INT32, n1: pl.DT_INT32, n2: pl.DT_INT32, n3: pl.DT_INT32,
                               out: pl.Tensor[[_WI_COUNT_ROWS, _WI_ELEMS], _SHIFT_DTYPE]):
    with pl.section_vector():
        it = pl.make_tile(
            pl.TileType(shape=[1, _WI_COUNT_IDX], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=0x1000)
        ret = pl.make_tile(
            pl.TileType(shape=[_WI_COUNT_ROWS, _WI_ELEMS], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec),
            addr=0x800)
        pl.load(it, idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)

        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB_ADDR, peer=(me + 1) % nr)
        dt0 = pld.make_dtensor(win, wi, 0 * _WI_COUNT_STRIDE, [_WI_COUNT_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        dt1 = pld.make_dtensor(win, wi, 1 * _WI_COUNT_STRIDE, [_WI_COUNT_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        dt2 = pld.make_dtensor(win, wi, 2 * _WI_COUNT_STRIDE, [_WI_COUNT_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        dt3 = pld.make_dtensor(win, wi, 3 * _WI_COUNT_STRIDE, [_WI_COUNT_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        ch.write_indexed(dt0, src, src_indices=it, count=n0)
        ch.write_indexed(dt1, src, src_indices=it, count=n1)
        ch.write_indexed(dt2, src, src_indices=it, count=n2)
        ch.write_indexed(dt3, src, src_indices=it, count=n3)
        ch.drain()
        pld.barrier(win)
        pld.get(ret, dt2, [0, 0], me)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.store(out, ret, [0, 0])


def write_indexed_count_worker(rank, nrank):
    torch.npu.set_device(rank)
    win, buf = _wi_setup(rank, nrank)
    rows = torch.arange(_WI_COUNT_ROWS, dtype=torch.float32) + rank * 10
    src = rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16).npu()
    perm = [(i * 5) % _WI_COUNT_ROWS for i in range(_WI_COUNT_ROWS)]
    idx = torch.tensor(perm + [0] * (_WI_COUNT_IDX - _WI_COUNT_ROWS), dtype=torch.int32).view(
        1, _WI_COUNT_IDX).npu()
    out = torch.zeros((_WI_COUNT_ROWS, _WI_ELEMS), dtype=torch.bfloat16).npu()
    counts = (0, 1, 12, 13)

    write_indexed_count_kernel[None, 1](win, buf, src, idx, *counts, out)
    torch.npu.synchronize()

    prev = (rank - 1) % nrank
    prev_rows = torch.arange(_WI_COUNT_ROWS, dtype=torch.float32) + prev * 10
    prev_src = prev_rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16)
    payload = buf.view(torch.bfloat16)
    for slot, n in enumerate(counts):
        start = slot * _WI_COUNT_STRIDE // 2
        got = payload[start:start + _WI_COUNT_ROWS * _WI_ELEMS].view(-1, _WI_ELEMS).cpu()
        exp = torch.zeros_like(got)
        if n > 0:
            exp[:n] = prev_src[torch.tensor(perm[:n])]
        assert torch.equal(got, exp), f"[rank {rank}] count={n} slot mismatch"
    assert torch.equal(out.cpu()[:12], prev_src[torch.tensor(perm[:12])]), f"[rank {rank}] count readback mismatch"
    print(f"[rank {rank}] write_indexed_count ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_indexed_count(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_indexed_count_worker, args=(nprocs,), nprocs=nprocs)


_WI_LARGE_ROWS = 192
_WI_LARGE_UB = 0x8000
_WI_LARGE_IDX_UB = 0x1000


@pl.jit(auto_mutex=False)
def write_indexed_large_kernel(win: pld.Window,
                               wi: pl.Tensor[[_WI_BUF], pl.DT_UINT8],
                               src: pl.Tensor[[_WI_LARGE_ROWS, _WI_ELEMS], _SHIFT_DTYPE],
                               idx: pl.Tensor[[1, _WI_LARGE_ROWS], pl.DT_INT32],
                               out: pl.Tensor[[_WI_LARGE_ROWS, _WI_ELEMS], _SHIFT_DTYPE]):
    with pl.section_vector():
        it = pl.make_tile(
            pl.TileType(shape=[1, _WI_LARGE_ROWS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=_WI_LARGE_IDX_UB)
        ret = pl.make_tile(
            pl.TileType(shape=[_WI_LARGE_ROWS, _WI_ELEMS], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec),
            addr=_WI_LARGE_UB)
        pl.load(it, idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=4)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=7)

        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB_ADDR, peer=(me + 1) % nr)
        dt = pld.make_dtensor(win, wi, 0, [_WI_LARGE_ROWS, _WI_ELEMS], _SHIFT_DTYPE)
        ch.write_indexed(dt, src, src_indices=it, count=_WI_LARGE_ROWS)
        ch.drain()
        pld.barrier(win)
        pld.get(ret, dt, [0, 0], me)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=5)
        pl.store(out, ret, [0, 0])


def write_indexed_large_worker(rank, nrank):
    torch.npu.set_device(rank)
    win, buf = _wi_setup(rank, nrank)
    rows = torch.arange(_WI_LARGE_ROWS, dtype=torch.float32) + rank * 10
    src = rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16).npu()
    perm = [(i * 37) % _WI_LARGE_ROWS for i in range(_WI_LARGE_ROWS)]
    idx = torch.tensor(perm, dtype=torch.int32).view(1, _WI_LARGE_ROWS).npu()
    out = torch.zeros((_WI_LARGE_ROWS, _WI_ELEMS), dtype=torch.bfloat16).npu()

    write_indexed_large_kernel[None, 1](win, buf, src, idx, out)
    torch.npu.synchronize()

    prev = (rank - 1) % nrank
    prev_rows = torch.arange(_WI_LARGE_ROWS, dtype=torch.float32) + prev * 10
    prev_src = prev_rows.view(-1, 1).repeat(1, _WI_ELEMS).to(torch.bfloat16)
    exp = prev_src[torch.tensor(perm)]
    payload = buf.view(torch.bfloat16).view(-1, _WI_ELEMS).cpu()
    assert torch.equal(payload[0:_WI_LARGE_ROWS], exp), f"[rank {rank}] large indexed slot mismatch"
    assert torch.equal(out.cpu(), exp), f"[rank {rank}] large indexed readback mismatch"
    print(f"[rank {rank}] write_indexed_large ok", flush=True)
    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_write_indexed_large(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(write_indexed_large_worker, args=(nprocs,), nprocs=nprocs)


_WV_INT64_BASE = 0     # one int64 word per rank
_WV_FP32_BASE = 1024   # one fp32 word per rank, no overlap with the int64 region


@pl.jit(auto_mutex=True)
def write_value_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                       out: pl.Tensor[[_ST_MAX_RANKS, 1], pl.DT_INT64],
                       out_f32: pl.Tensor[[_ST_MAX_RANKS * 2, 1], pl.DT_FP32]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)
        # One word per rank; the tile row needs 32 B, so the readback covers all ranks
        # (8 fp32 slots or 4 int64 slots).
        tg_i64 = pl.make_tile_group(
            type=pl.TileType(shape=[_ST_MAX_RANKS, 1], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
            addrs=0, mutex_ids=[0])
        tg_f32 = pl.make_tile_group(
            type=pl.TileType(shape=[_ST_MAX_RANKS * 2, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x100, mutex_ids=[1])

        # int64 word at a runtime byte offset: stage the inline WQE, then publish it
        dt_i64 = pld.make_dtensor(win, ws, _WV_INT64_BASE + me * 8, [1], pl.DT_INT64)
        ch.write(dt_i64, me + 100, submit=False)
        ch.submit()

        # fp32 word: literal value, published immediately
        dt_f32 = pld.make_dtensor(win, ws, _WV_FP32_BASE + me * 4, [1], pl.DT_FP32)
        ch.write(dt_f32, 1.5)

        ch.drain()
        pld.barrier(win)
        # Read the received words back right after the barrier: the check must not lean
        # on the kernel-exit synchronization.
        rb_i64 = pld.make_dtensor(win, ws, _WV_INT64_BASE, [_ST_MAX_RANKS, 1], pl.DT_INT64)
        rb_f32 = pld.make_dtensor(win, ws, _WV_FP32_BASE, [_ST_MAX_RANKS * 2, 1], pl.DT_FP32)
        pld.get(tg_i64.current(), rb_i64, [0, 0], me)
        pld.get(tg_f32.current(), rb_f32, [0, 0], me)
        pl.store(out, tg_i64.current(), [0, 0])
        pl.store(out_f32, tg_f32.current(), [0, 0])


def write_value_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    win.connect()
    out = torch.full((_ST_MAX_RANKS, 1), -1, dtype=torch.int64).npu()
    out_f32 = torch.full((_ST_MAX_RANKS * 2, 1), -1.0, dtype=torch.float32).npu()

    write_value_kernel[None, 1](win, ws, out, out_f32)
    torch.npu.synchronize()
    prev = (rank - 1) % nrank
    # The kernel read the words back directly after the barrier, so this is what the
    # consumer saw at the sync point — not what the window holds after the kernel exits.
    # Rank r writes r + 100 into slot r of every window.
    at_barrier = out.cpu().view(-1).tolist()[:nrank]
    assert at_barrier == [r + 100 for r in range(nrank)], (
        f"[rank {rank}] write_value at barrier got {at_barrier}, expected {[r + 100 for r in range(nrank)]}"
    )
    at_barrier_f32 = out_f32.cpu().view(-1).tolist()[:nrank]
    assert at_barrier_f32 == [1.5] * nrank, (
        f"[rank {rank}] fp32 at barrier got {at_barrier_f32}, expected {[1.5] * nrank}"
    )

    # Each rank writes one word to its successor, so this window must hold the previous
    # rank's word in that rank's slot. The payload is uninitialized otherwise, so check
    # only the word we own an expectation for.
    got_i64 = ws[_WV_INT64_BASE:_WV_INT64_BASE + 8 * nrank].view(torch.int64).cpu()
    assert got_i64[prev].item() == prev + 100, (
        f"[rank {rank}] int64 word {prev} got {got_i64.tolist()}, expected {prev + 100}"
    )

    got_f32 = ws[_WV_FP32_BASE:_WV_FP32_BASE + 4 * nrank].view(torch.float32).cpu()
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
def two_region_kernel(win: pld.Window, ra: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                      rb: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                      out: pl.Tensor[[_ST_MAX_RANKS * 2, 1], pl.DT_INT64]):
    """One word into each of two independently registered regions."""
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        peer = (me + 1) % nr
        ch = pld.channel(win, _CH_UB_ADDR, peer=peer)
        # One word per rank; the tile row needs 32 B, so the readback covers all ranks.
        tg_a = pl.make_tile_group(
            type=pl.TileType(shape=[_ST_MAX_RANKS, 1], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
            addrs=0, mutex_ids=[0])
        tg_b = pl.make_tile_group(
            type=pl.TileType(shape=[_ST_MAX_RANKS, 1], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec),
            addrs=0x100, mutex_ids=[1])
        dt_a = pld.make_dtensor(win, ra, me * 8, [1], pl.DT_INT64)
        dt_b = pld.make_dtensor(win, rb, me * 8, [1], pl.DT_INT64)
        ch.write(dt_a, me + 100)
        ch.write(dt_b, me + 200)
        ch.drain()
        pld.barrier(win)
        # Read both regions back right after the barrier: the check must not lean on the
        # kernel-exit synchronization.
        rd_a = pld.make_dtensor(win, ra, 0, [_ST_MAX_RANKS, 1], pl.DT_INT64)
        rd_b = pld.make_dtensor(win, rb, 0, [_ST_MAX_RANKS, 1], pl.DT_INT64)
        pld.get(tg_a.current(), rd_a, [0, 0], me)
        pld.get(tg_b.current(), rd_b, [0, 0], me)
        pl.store(out, tg_a.current(), [0, 0])
        pl.store(out, tg_b.current(), [_ST_MAX_RANKS, 0])


def two_region_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    ra = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    rb = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ra, "ra", _PROTOCOL)
    pld.register_memory(win, rb, "rb", _PROTOCOL)
    win.connect()
    out = torch.full((_ST_MAX_RANKS * 2, 1), -1, dtype=torch.int64).npu()

    two_region_kernel[None, 1](win, ra, rb, out)
    torch.npu.synchronize()
    prev = (rank - 1) % nrank
    # The kernel read both regions back directly after the barrier, so this is what the
    # consumer saw at the sync point — not what they hold after the kernel exits.
    # Rank r writes r + 100 / r + 200 into slot r of every window.
    at_barrier = out.cpu().view(-1).tolist()
    exp_a = [r + 100 for r in range(nrank)]
    exp_b = [r + 200 for r in range(nrank)]
    assert at_barrier[0:nrank] == exp_a, (
        f"[rank {rank}] region 'ra' at barrier got {at_barrier[0:nrank]}, expected {exp_a}"
    )
    assert at_barrier[_ST_MAX_RANKS:_ST_MAX_RANKS + nrank] == exp_b, (
        f"[rank {rank}] region 'rb' at barrier got {at_barrier[_ST_MAX_RANKS:_ST_MAX_RANKS + nrank]}, "
        f"expected {exp_b}"
    )

    got_a = ra[prev * 8:(prev + 1) * 8].view(torch.int64).cpu().item()
    got_b = rb[prev * 8:(prev + 1) * 8].view(torch.int64).cpu().item()
    assert got_a == prev + 100, f"[rank {rank}] region 'ra' word {prev} got {got_a}, expected {prev + 100}"
    assert got_b == prev + 200, f"[rank {rank}] region 'rb' word {prev} got {got_b}, expected {prev + 200}"

    # Writes target their region only: the next rank's slot stays zero in both regions.
    other = (prev + 1) % nrank
    stray_a = ra[other * 8:(other + 1) * 8].view(torch.int64).cpu().item()
    stray_b = rb[other * 8:(other + 1) * 8].view(torch.int64).cpu().item()
    assert stray_a == 0 and stray_b == 0, f"[rank {rank}] stray writes: ra={stray_a}, rb={stray_b}"
    print(f"[rank {rank}] two_regions ok: ra {got_a}, rb {got_b}", flush=True)

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_two_regions(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(two_region_worker, args=(nprocs,), nprocs=nprocs)


@pl.jit(auto_mutex=True)
def write_notify_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                        src: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE],
                        out: pl.Tensor[[1, _SHIFT_N], _SHIFT_DTYPE]):
    """Fused write + hardware notify: the signal fires on payload completion, with no drain."""
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_SHIFT_DTYPE, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=_SHIFT_N * 2, mutex_ids=[0])
        t2 = tg.current()
        me, nr = win.rank(), win.nrank()
        dt = pld.make_dtensor(win, ws, 0, [1, _SHIFT_N], _SHIFT_DTYPE)
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
    win, ws, src, out = _shift_setup(rank, nrank)
    write_notify_kernel[None, 1](win, ws, src, out)
    torch.npu.synchronize()
    expected = torch.full((1, _SHIFT_N), float((rank - 1) % nrank), dtype=torch.bfloat16)
    assert torch.equal(out.cpu(), expected), (
        f"[rank {rank}] write_notify got {out.cpu()[0, :3].tolist()}..., "
        f"expected rank {(rank - 1) % nrank}'s data"
    )
    # The hardware notify wrote the producer's value into signal slot 0, which follows
    # the Barrier slot (offset 512 in the control region).
    sig = win.control_buffer[512:520].view(torch.int64).cpu().item()
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
    win = pld.create_window(group)
    win.connect()
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

        # Same ring without the store flush: there is no prior put to order here.
        sig_ring_nf = pld.signal(win, 8)
        pld.notify(sig_ring_nf, (me + 1) % nr, 1, op=pld.NotifyOp.ATOMIC_ADD, flush_store=False)
        pld.wait(sig_ring_nf, 1, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)

        # notify_all without the flush: every rank adds 1 to every rank's slot 9.
        sig_all_nf = pld.signal(win, 9)
        pld.notify_all(sig_all_nf, 1, op=pld.NotifyOp.ATOMIC_ADD, flush_store=False)
        pld.wait(sig_all_nf, nr, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)

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
    win = pld.create_window(group)
    win.connect()
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
    win = pld.create_window(group)
    win.connect()
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
def all_gather_signals_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                              src: pl.Tensor[[1, _AG_SHARD], pl.DT_BF16],
                              out: pl.Tensor[[_ST_MAX_RANKS, _AG_SHARD], pl.DT_BF16]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        tile_type = pl.TileType(shape=[1, _AG_SHARD], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
        tg = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        tg_rcv = pl.make_tile_group(type=tile_type, addrs=_AG_SHARD * 2, mutex_ids=[1])
        t = tg.current()
        t_rcv = tg_rcv.current()
        sig_done = pld.signal(win, 0)
        pl.load(t, src, [0, 0])
        dt = pld.make_dtensor(win, ws, 0, [nr, _AG_SHARD], pl.DT_BF16)
        for r in pl.range(nr):
            pld.put(dt, t, [me, 0], r)
            pld.notify(sig_done, r, 1, op=pld.NotifyOp.ATOMIC_ADD)
        pld.wait(sig_done, nr, cmp=pld.Cmp.GE, timeout=_WAIT_TIMEOUT)
        # Read the shards back the moment every rank's arrival proves they landed:
        # the check must not lean on the kernel-exit synchronization.
        for r in pl.range(nr):
            pld.get(t_rcv, dt, [r, 0], me)
            pl.store(out, t_rcv, [r, 0])


def all_gather_signals_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    win.connect()
    src = torch.full((1, _AG_SHARD), float(rank), dtype=torch.bfloat16).npu()
    out = torch.zeros((_ST_MAX_RANKS, _AG_SHARD), dtype=torch.bfloat16).npu()

    all_gather_signals_kernel[None, 1](win, ws, src, out)
    torch.npu.synchronize()

    expect = torch.cat(
        [torch.full((1, _AG_SHARD), float(r), dtype=torch.bfloat16) for r in range(nrank)], dim=1
    )
    # The kernel readback happens directly after the arrival wait, so this is what the
    # consumer saw at wait time — not what the window holds after the kernel exits.
    at_wait = out.cpu()[:nrank].view(1, nrank * _AG_SHARD)
    assert torch.equal(at_wait, expect), (
        f"[rank {rank}] all_gather at wait got {at_wait[0, :4].tolist()}...{at_wait[0, -4:].tolist()}, "
        f"expected shards 0..{nrank - 1}"
    )

    gathered = ws[:nrank * _AG_SHARD * 2].view(torch.bfloat16).cpu().view(1, nrank * _AG_SHARD)
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
def all_reduce_put_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                          src: pl.Tensor[[1, _AR_SHARD], pl.DT_BF16],
                          out: pl.Tensor[[1, _AR_SHARD], pl.DT_BF16]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _AR_SHARD], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
        tg_a = pl.make_tile_group(type=tile_type, addrs=0, mutex_ids=[0])
        tg_b = pl.make_tile_group(type=tile_type, addrs=_AR_SHARD * 2, mutex_ids=[1])
        t = tg_a.current()
        t2 = tg_b.current()
        me, nr = win.rank(), win.nrank()
        pl.load(t, src, [0, 0])
        dt = pld.make_dtensor(win, ws, 0, [1, _AR_SHARD], pl.DT_BF16)
        pld.barrier(win)  # every rank's host-side payload zeroing is complete
        for r in pl.range(nr):  # accumulate my shard into every rank's window
            pld.put(dt, t, [0, 0], r, atomic_mode=pl.AtomicType.AtomicAdd)
        pld.barrier(win)  # all adds have landed before any rank reads back
        pld.get(t2, dt, [0, 0], me)
        pl.store(out, t2, [0, 0])


def all_reduce_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    win.connect()
    # AtomicAdd accumulates, so zero the payload before the launch.
    ws.zero_()
    src = torch.full((1, _AR_SHARD), float(rank), dtype=torch.bfloat16).npu()
    out = torch.zeros((1, _AR_SHARD), dtype=torch.bfloat16).npu()

    all_reduce_put_kernel[None, 1](win, ws, src, out)  # one AIV: redundant puts would multiply the sum
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


_RP_ROWS = 8
_RP_COLS = 16
_RP_READ_OFF = 0     # this rank's prefilled slot; peers load it through the remote pointer
_RP_WRITE_OFF = 4096  # slot peers store into; independent of the read slot
_RP_BYTES = _RP_ROWS * _RP_COLS * 2  # fp16 payload


def _rp_pattern(rank):
    base = torch.arange(_RP_ROWS * _RP_COLS, dtype=torch.float32).reshape(_RP_ROWS, _RP_COLS)
    return (base + rank * 1000).to(torch.float16)


@pl.jit
def remote_ptr_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                      src: pl.Tensor[[_RP_ROWS, _RP_COLS], pl.DT_FP16],
                      out: pl.Tensor[[_RP_ROWS, _RP_COLS], pl.DT_FP16]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        peer = (me + 1) % nr
        pld.barrier(win)  # every rank's host-side prefill is complete

        tile_type = pl.TileType(shape=[_RP_ROWS, _RP_COLS], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Vec)
        rd = pl.make_tile(tile_type, addr=0)
        wr = pl.make_tile(tile_type, addr=_RP_BYTES)

        # Ordinary pl.load through a raw remote pointer at the read slot.
        rptr = pld.get_remote_ptr(win, ws, peer) + _RP_READ_OFF
        rview = pl.make_tensor(rptr, [_RP_ROWS, _RP_COLS], dtype=pl.DT_FP16)
        pl.load(rd, rview, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.store(out, rd, [0, 0])

        # Ordinary pl.store through a raw remote pointer resolved from an interior local
        # pointer (region base + write slot): the in-region offset must be preserved.
        wptr = pld.get_remote_ptr(win, pl.addptr(pl.make_ptr(ws), _RP_WRITE_OFF), peer)
        wview = pl.make_tensor(wptr, [_RP_ROWS, _RP_COLS], dtype=pl.DT_FP16)
        pl.load(wr, src, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=3)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=3)
        pl.store(wview, wr, [0, 0])

        pld.barrier(win)  # flush the remote store; every rank's write has landed


def remote_ptr_worker(rank, nrank):
    torch.npu.set_device(rank)
    group = init_dist(rank, nrank)
    win = pld.create_window(group)
    ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    win.connect()

    peer = (rank + 1) % nrank
    prev = (rank - 1) % nrank
    own = _rp_pattern(rank)
    ws[_RP_READ_OFF:_RP_READ_OFF + _RP_BYTES].view(torch.float16).copy_(own.view(-1).npu())
    out = torch.full((_RP_ROWS, _RP_COLS), -1.0, dtype=torch.float16).npu()

    remote_ptr_kernel[None, 1](win, ws, own.npu(), out)
    torch.npu.synchronize()

    got = out.cpu()
    exp = _rp_pattern(peer)
    assert torch.equal(got, exp), (
        f"[rank {rank}] remote load got {got[0, :4].tolist()}..., expected {exp[0, :4].tolist()}"
    )
    got = ws[_RP_WRITE_OFF:_RP_WRITE_OFF + _RP_BYTES].view(torch.float16).cpu()
    exp = _rp_pattern(prev).view(-1)
    assert torch.equal(got, exp), (
        f"[rank {rank}] remote store got {got[:4].tolist()}..., expected {exp[:4].tolist()}"
    )
    print(f"[rank {rank}] remote_ptr ok", flush=True)

    win.close()
    dist.destroy_process_group()


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_remote_ptr(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(remote_ptr_worker, args=(nprocs,), nprocs=nprocs)


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
_AT_MINMAX = 336     # int64 [4]: min/max sequences
_AT_MINMAX_U64 = 368  # uint64 [2]
_AT_MINMAX_F32 = 384  # fp32 [2]
_AT_MINMAX_I32 = 392  # int32 [2]
_AT_MINMAX_U32 = 400  # uint32 [2]


@pl.jit
def atomic_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                  out: pl.Tensor[[6], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        peer = (me + 1) % nr

        pld.barrier(win)  # every rank's host-side payload zeroing is complete

        i64 = pld.make_dtensor(win, ws, _AT_INT64, [5], pl.DT_INT64)
        old_add = pld.atomic_add(i64, peer, [0], 1)
        old_exch = pld.atomic_exchange(i64, peer, [1], me + 100)
        old_cas = pld.atomic_cas(i64, peer, [2], 0, me + 200)
        old_fail = pld.atomic_cas(i64, peer, [3], 0xDEAD, 123)

        u64 = pld.make_dtensor(win, ws, _AT_UINT64, [3], pl.DT_UINT64)
        pld.atomic_add(u64, peer, [0], 1)
        pld.atomic_exchange(u64, peer, [1], me + 300)
        pld.atomic_cas(u64, peer, [2], 0, me + 400)

        i32 = pld.make_dtensor(win, ws, _AT_INT32, [3], pl.DT_INT32)
        pld.atomic_add(i32, peer, [0], 1)
        pld.atomic_exchange(i32, peer, [1], me + 500)
        pld.atomic_cas(i32, peer, [2], 0, me + 600)

        u32 = pld.make_dtensor(win, ws, _AT_UINT32, [3], pl.DT_UINT32)
        pld.atomic_add(u32, peer, [0], 1)
        pld.atomic_exchange(u32, peer, [1], me + 700)
        pld.atomic_cas(u32, peer, [2], 0, me + 800)

        f32 = pld.make_dtensor(win, ws, _AT_FP32, [3], pl.DT_FP32)
        pld.atomic_add(f32, peer, [0], 1.5)
        pld.atomic_exchange(f32, peer, [1], 2.25)
        pld.atomic_cas(f32, peer, [2], 0.0, 3.5)

        # Scalar offset form: linear element 5 inside a [4, 4] view (never [5, 0]).
        flat = pld.make_dtensor(win, ws, _AT_FLAT, [4, 4], pl.DT_INT64)
        pld.atomic_add(flat, peer, 5, 1)
        # Coordinate form on the same 2-D view: [1, 2] flattens to linear element 6.
        pld.atomic_add(flat, peer, [1, 2], 2)
        # Runtime offset expressions are legal: coordinates and the scalar linear form.
        pld.atomic_add(flat, peer, [me, 3], 3)   # peer's element me*4 + 3
        pld.atomic_add(flat, peer, me * 4, 4)    # scalar linear expression: element me*4

        # atomic_sub is a frontend wrapper over atomic_add(-val).
        sub = pld.make_dtensor(win, ws, _AT_SUB, [7], pl.DT_INT64)
        pld.atomic_sub(sub, peer, [0], 5)        # element 0 -> -5
        pld.atomic_sub(sub, peer, [1], -3)       # element 1 -> +3 (double negation)
        pld.atomic_sub(sub, peer, me + 2, 1)     # runtime scalar offset -> element me+2
        pld.atomic_sub(sub, peer, [6], me)       # runtime value -> element 6 = -me

        sub_u64 = pld.make_dtensor(win, ws, _AT_SUB_U64, [1], pl.DT_UINT64)
        pld.atomic_sub(sub_u64, peer, [0], 1)    # 0 - 1 modulo 2^64

        sub_f32 = pld.make_dtensor(win, ws, _AT_SUB_F32, [1], pl.DT_FP32)
        pld.atomic_sub(sub_f32, peer, [0], 0.5)  # -0.5

        # atomic_min/max keep the smaller/larger value; a plain add would move both.
        minmax = pld.make_dtensor(win, ws, _AT_MINMAX, [4], pl.DT_INT64)
        pld.atomic_min(minmax, peer, 0, -5)      # 0 -> -5 (scalar linear offset)
        pld.atomic_max(minmax, peer, [1], 9)     # 0 -> 9
        pld.atomic_min(minmax, peer, [1], 4)     # 9 -> 4
        pld.atomic_max(minmax, peer, [2], 3)     # 0 -> 3
        pld.atomic_min(minmax, peer, [3], 8)     # 0 -> 0 (unchanged)

        minmax_u64 = pld.make_dtensor(win, ws, _AT_MINMAX_U64, [2], pl.DT_UINT64)
        pld.atomic_max(minmax_u64, peer, [0], 7)  # 0 -> 7
        pld.atomic_min(minmax_u64, peer, [1], 9)  # 0 -> 0

        minmax_i32 = pld.make_dtensor(win, ws, _AT_MINMAX_I32, [2], pl.DT_INT32)
        pld.atomic_min(minmax_i32, peer, [0], -5)  # 0 -> -5
        pld.atomic_max(minmax_i32, peer, [1], 9)   # 0 -> 9

        minmax_u32 = pld.make_dtensor(win, ws, _AT_MINMAX_U32, [2], pl.DT_UINT32)
        pld.atomic_min(minmax_u32, peer, [0], 5)   # 0 -> 0
        pld.atomic_max(minmax_u32, peer, [1], 7)   # 0 -> 7

        minmax_f32 = pld.make_dtensor(win, ws, _AT_MINMAX_F32, [2], pl.DT_FP32)
        pld.atomic_max(minmax_f32, peer, [0], 2.5)  # 0 -> 2.5
        pld.atomic_min(minmax_f32, peer, [0], 1.5)  # 2.5 -> 1.5
        pld.atomic_min(minmax_f32, peer, [1], 2.5)  # 0 -> 0

        # Contended counter: every rank adds 1 to rank 0's word.
        shared = pld.make_dtensor(win, ws, 32, [1], pl.DT_INT64)
        old_shared = pld.atomic_add(shared, 0, [0], 1)

        # Nested value position: the atomic is an argument of block.setval, not a bound var.
        nested = pld.make_dtensor(win, ws, _AT_NESTED, [1], pl.DT_INT64)
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
    win = pld.create_window(group)
    ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device='npu')
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    win.connect()
    # Zero only the payload before launch; the signal/barrier regions are framework-managed.
    ws.zero_()
    out = torch.full((6,), -1, dtype=torch.int64).npu()

    atomic_kernel[None, 1](win, ws, out)
    torch.npu.synchronize()

    # Every rank's payload received the min/max sequence from its predecessor.
    i64 = ws.view(torch.int64)
    got = i64[_AT_MINMAX // 8:_AT_MINMAX // 8 + 4].cpu().tolist()
    assert got == [-5, 4, 3, 0], f"[rank {rank}] int64 min/max got {got}"
    u64 = ws.view(torch.uint64)
    got = u64[_AT_MINMAX_U64 // 8:_AT_MINMAX_U64 // 8 + 2].cpu().tolist()
    assert got == [7, 0], f"[rank {rank}] uint64 min/max got {got}"
    i32 = ws.view(torch.int32)
    got = i32[_AT_MINMAX_I32 // 4:_AT_MINMAX_I32 // 4 + 2].cpu().tolist()
    assert got == [-5, 9], f"[rank {rank}] int32 min/max got {got}"
    u32 = ws.view(torch.uint32)
    got = u32[_AT_MINMAX_U32 // 4:_AT_MINMAX_U32 // 4 + 2].cpu().tolist()
    assert got == [0, 7], f"[rank {rank}] uint32 min/max got {got}"
    f32 = ws.view(torch.float32)
    got = f32[_AT_MINMAX_F32 // 4:_AT_MINMAX_F32 // 4 + 2].cpu().tolist()
    assert got == [1.5, 0.0], f"[rank {rank}] fp32 min/max got {got}"

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
    test_create_window_layer()
    test_window_barrier()
    test_local_roundtrip()
    test_shift_load_right()
    test_shift_load_left()
    test_shift_load_right_urma()
    test_shift_load_left_urma()
    test_write_submit()
    test_write_indexed()
    test_write_indexed_repeat()
    test_write_indexed_dtype()
    test_write_indexed_count()
    test_write_indexed_large()
    test_write_value()
    test_two_regions()
    test_write_notify()
    test_channel_exclusive()
    test_notify_wait()
    test_all_gather_signals()
    test_all_reduce()
    test_remote_ptr()
    test_atomics()
