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

"""Distributed Channel / BufferChannel micro-benchmarks (manual, 4 ranks).

All timings use ``pl.get_cycle()`` on the device; the host reports cycles and the
equivalent microseconds assuming the 1 GHz AIV clock (``cycles / 1000``). Cases are
skipped under pytest; run this file as a script on a free four-card group:

    task-submit --device 0,1,2,3 --run "cd <repo> && PYTHONPATH=<repo>/python \\
        python python/tests/st/operator/distributed/test_dist_bench.py"

Channel per-post cases issue ``_CH_ITERS`` operations; BufferChannel cases stage
``_BCH_ITERS`` single-WQE or ``_BCH_2WQE_ITERS`` two-WQE operations into a 64 KiB
buffer and time only the staging, with submit/drain outside the timed span.
"""

import os

import pypto_pro.language as pl
import pypto_pro.language.distribute as pld
import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

_WS_BYTES = 1 << 20
_SIG_SLOTS = 16
_PROTOCOL = pld.Protocol.UB_MEM | pld.Protocol.URMA
_CH_UB = 0x2000
_BCH_UB = 0x4000
_BCH_BYTES = 65536
_CH_ITERS = 64
_CH_NOTIFY_ITERS = 16  # 2 WQEBBs and one CQE each; the deferred batch shares one doorbell
_BCH_ITERS = 1000
_BCH_2WQE_ITERS = 512
_PAYLOAD_BYTES = 64
_PAYLOAD_2K = 2048
_REDUCE_WORDS = 16  # int32: 64 B
_BASELINE_FILE = "test_dist_stat.txt"
_REGRESSION_TOL = 0.10  # 5% is run-to-run noise; fail only beyond 10%
_SAMPLES = 3


def init_dist(rank, nrank):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    dist.init_process_group(backend="hccl", init_method=f"tcp://{ip}:{port}", world_size=nrank, rank=rank)


# --- Channel: one WQE per call, submit inside the timed loop unless noted -----------


@pl.jit(auto_mutex=True)
def bench_ch_write_post(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                        src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(ws)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_ITERS):
            ch.write(dst, src, submit=False)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.submit()
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_write_submit(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                          src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(ws)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_ITERS):
            ch.write(dst, src)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_write_value(win: pld.Window, sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                         out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_ITERS):
            ch.write(dst, 1, submit=False)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.submit()
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_read(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                  dst: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        src = ch.remote_ptr(ws)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_ITERS):
            ch.read(dst, src)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_write_reduce(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                          src: pl.Tensor[[_REDUCE_WORDS], pl.DT_INT32], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = pl.make_ptr(ch.remote_ptr(ws), pl.DT_INT32)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_ITERS):
            ch.write_reduce(dst, src, op=pld.ReduceOp.SUM)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_write_notify_post(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                               sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                               src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(ws)
        signal = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_NOTIFY_ITERS):
            ch.write_notify(dst, src, signal, 1, submit=False)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_NOTIFY_ITERS
        ch.submit()
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_write_notify_submit(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                                 sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                                 src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(ws)
        signal = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_NOTIFY_ITERS):
            ch.write_notify(dst, src, signal, 1)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_NOTIFY_ITERS
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_atomic_faa(win: pld.Window, sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                        fetch: pl.Tensor[[1], pl.DT_INT64], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_ITERS):
            ch.atomic_fetch_add(dst, fetch, 1)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_atomic_cas(win: pld.Window, sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                        fetch: pl.Tensor[[1], pl.DT_INT64], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_CH_ITERS):
            ch.atomic_cas(dst, fetch, 0, 1)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_submit(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                    src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(ws)
        for _ in pl.range(_CH_ITERS):
            ch.write(dst, src, submit=False)
        t0 = pl.get_cycle()
        ch.submit()
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        ch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_ch_drain(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                   src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        dst = ch.remote_ptr(ws)
        for _ in pl.range(_CH_ITERS):
            ch.write(dst, src, submit=False)
        ch.submit()
        t0 = pl.get_cycle()
        ch.drain()
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _CH_ITERS
        pld.barrier(win)


# --- BufferChannel: pure UB staging; one submit/drain after the timed loop ----------


@pl.jit(auto_mutex=True)
def bench_bch_write(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                    src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(ws)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_ITERS):
            bch.write(dst, src)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_write_2k(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                       src: pl.Tensor[[_PAYLOAD_2K], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(ws)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_ITERS):
            bch.write(dst, src)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_write_value(win: pld.Window, sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                          out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_ITERS):
            bch.write(dst, 1)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_read(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                   dst: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        src = ch.remote_ptr(ws)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_2WQE_ITERS):
            bch.read(dst, src)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_2WQE_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_write_reduce(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                           src: pl.Tensor[[_REDUCE_WORDS], pl.DT_INT32], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = pl.make_ptr(ch.remote_ptr(ws), pl.DT_INT32)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_2WQE_ITERS):
            bch.write_reduce(dst, src, op=pld.ReduceOp.SUM)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_2WQE_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_write_notify(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                           sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                           src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(ws)
        signal = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_2WQE_ITERS):
            bch.write_notify(dst, src, signal, 1)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_2WQE_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_atomic_faa(win: pld.Window, sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                         fetch: pl.Tensor[[1], pl.DT_INT64], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_2WQE_ITERS):
            bch.atomic_fetch_add(dst, fetch, 1)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_2WQE_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_atomic_cas(win: pld.Window, sig: pl.Tensor[[_SIG_SLOTS], pl.DT_INT64],
                         fetch: pl.Tensor[[1], pl.DT_INT64], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(sig)
        t0 = pl.get_cycle()
        for _ in pl.range(_BCH_2WQE_ITERS):
            bch.atomic_cas(dst, fetch, 0, 1)
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_2WQE_ITERS
        bch.submit()
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_submit(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                     src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(ws)
        for _ in pl.range(_BCH_ITERS):
            bch.write(dst, src)
        t0 = pl.get_cycle()
        bch.submit()
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_ITERS
        bch.drain()
        pld.barrier(win)


@pl.jit(auto_mutex=True)
def bench_bch_drain(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                    src: pl.Tensor[[_PAYLOAD_BYTES], pl.DT_UINT8], out: pl.Tensor[[4], pl.DT_INT64]):
    with pl.section_vector():
        me, nr = win.rank(), win.nrank()
        ch = pld.channel(win, _CH_UB, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, _BCH_UB, _BCH_BYTES)
        dst = ch.remote_ptr(ws)
        for _ in pl.range(_BCH_ITERS):
            bch.write(dst, src)
        bch.submit()
        t0 = pl.get_cycle()
        bch.drain()
        t1 = pl.get_cycle()
        out[0] = t1 - t0
        out[1] = _BCH_ITERS
        pld.barrier(win)


def _load_baseline():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _BASELINE_FILE)
    if not os.path.exists(path):
        return None
    baseline = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.rsplit("|", 2)]
            name, value = parts[0], float(parts[1])
            noise = float(parts[2]) if len(parts) == 3 else _REGRESSION_TOL
            baseline[name] = (value, noise)
    return baseline


def _report(rank, name, cycles_per_op, baseline):
    text = f"[rank {rank}] {name:<28} {cycles_per_op:8.2f} cyc/op  {cycles_per_op / 1000:8.4f} us/op"
    if baseline is not None and name in baseline:
        base, noise = baseline[name]
        text += (f"  base {base:8.2f} +/-{noise:.0%}"
                 f"  {100.0 * (cycles_per_op - base) / base:+6.1f}%")
    print(text, flush=True)


def _run_case(rank, name, kernel, args, out, baseline, failures):
    kernel[None, 1](*args)  # warmup
    torch.npu.synchronize()
    best = None
    for _ in range(_SAMPLES):
        kernel[None, 1](*args)
        torch.npu.synchronize()
        cycles, count = out.cpu().tolist()[:2]
        cycles_per_op = cycles / count
        best = cycles_per_op if best is None else min(best, cycles_per_op)
    _report(rank, name, best, baseline)
    if baseline is None:
        return
    if name not in baseline:
        failures.append(f"{name}: missing from {_BASELINE_FILE}")
    else:
        base, noise = baseline[name]
        if best > base * (1.0 + noise):
            failures.append(f"{name}: {best:.2f} cyc/op > baseline {base:.2f} + {noise:.0%}")


def worker(rank, nrank):
    torch.npu.set_device(rank)
    init_dist(rank, nrank)
    win = pld.create_window(dist.distributed_c10d._get_default_group())
    ws = torch.zeros(_WS_BYTES, dtype=torch.uint8, device="npu")
    sig = torch.zeros(_SIG_SLOTS, dtype=torch.int64, device="npu")
    pld.register_memory(win, ws, "ws", _PROTOCOL)
    pld.register_memory(win, sig, "sig", _PROTOCOL)
    win.connect()

    src = torch.zeros(_PAYLOAD_BYTES, dtype=torch.uint8, device="npu")
    src2k = torch.zeros(_PAYLOAD_2K, dtype=torch.uint8, device="npu")
    src32 = torch.zeros(_REDUCE_WORDS, dtype=torch.int32, device="npu")
    rbuf = torch.zeros(_PAYLOAD_BYTES, dtype=torch.uint8, device="npu")
    fetch = torch.zeros(1, dtype=torch.int64, device="npu")
    out = torch.zeros(4, dtype=torch.int64, device="npu")

    cases = [
        ("ch.write (post)", bench_ch_write_post, (win, ws, src, out)),
        ("ch.write (submit)", bench_ch_write_submit, (win, ws, src, out)),
        ("ch.write value", bench_ch_write_value, (win, sig, out)),
        ("ch.read", bench_ch_read, (win, ws, rbuf, out)),
        ("ch.write_reduce", bench_ch_write_reduce, (win, ws, src32, out)),
        ("ch.write_notify (submit)", bench_ch_write_notify_submit, (win, ws, sig, src, out)),
        ("ch.write_notify (post)", bench_ch_write_notify_post, (win, ws, sig, src, out)),
        ("ch.atomic_fetch_add", bench_ch_atomic_faa, (win, sig, fetch, out)),
        ("ch.atomic_cas", bench_ch_atomic_cas, (win, sig, fetch, out)),
        ("bch.write 64B", bench_bch_write, (win, ws, src, out)),
        ("bch.write 2KB", bench_bch_write_2k, (win, ws, src2k, out)),
        ("bch.write value", bench_bch_write_value, (win, sig, out)),
        ("bch.read", bench_bch_read, (win, ws, rbuf, out)),
        ("bch.write_reduce", bench_bch_write_reduce, (win, ws, src32, out)),
        ("bch.write_notify", bench_bch_write_notify, (win, ws, sig, src, out)),
        ("bch.atomic_fetch_add", bench_bch_atomic_faa, (win, sig, fetch, out)),
        ("bch.atomic_cas", bench_bch_atomic_cas, (win, sig, fetch, out)),
    ]
    slow_cases = [
        ("ch.submit", bench_ch_submit, (win, ws, src, out)),
        ("ch.drain", bench_ch_drain, (win, ws, src, out)),
        ("bch.submit", bench_bch_submit, (win, ws, src, out)),
        ("bch.drain", bench_bch_drain, (win, ws, src, out)),
    ]

    selected = os.getenv("DIST_BENCH_CASE")
    if selected:
        cases = [case for case in cases + slow_cases if case[0] == selected]
        if not cases:
            raise ValueError(f"unknown DIST_BENCH_CASE={selected!r}")
    # Opt-in cases: their cost is dominated by the deferred-batch bookkeeping rather
    # than a steady per-op rate. Run one with DIST_BENCH_CASE="ch.submit" (or
    # "ch.drain", "bch.submit", "bch.drain").

    baseline = None if os.getenv("DIST_BENCH_NO_BASELINE") else _load_baseline()
    failures = []
    for name, kernel, args in cases:
        _run_case(rank, name, kernel, args, out, baseline, failures)

    win.close()
    dist.destroy_process_group()
    if failures:
        raise AssertionError(f"[rank {rank}] dist bench regressions: " + "; ".join(failures))


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_dist_bench(world_size: int = 4):
    nprocs = min(world_size, torch.npu.device_count())
    torch.multiprocessing.spawn(worker, args=(nprocs,), nprocs=nprocs)


if __name__ == "__main__":
    test_dist_bench()
