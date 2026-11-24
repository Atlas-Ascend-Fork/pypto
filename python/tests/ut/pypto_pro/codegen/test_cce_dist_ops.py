#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CCE emission for the pointer-based distributed window API.

Remote transfers now spell ``pl.load``/``pl.store`` over a ``pl.make_tensor`` view of
``pld.remote_ptr``, and channel writes take pointers resolved by
``ch.remote_ptr``. These tests pin the generated device calls.
"""

import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


def _compile_to_cce(kernel, arch: str = "3510") -> str:
    from pypto_pro.runtime.jit import _add_kernel_header, _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import normalize_arch

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), normalize_arch(arch), "")
    return _add_kernel_header(_assemble_cv_source(cube, vector)).content


@pl.jit(arch="3510", auto_mutex=False)
def _dist_pointer_codegen_kernel(
    win: pld.Window,
    ws: pl.Tensor[[4096], pl.DT_UINT8],
    sig: pl.Tensor[[16], pl.DT_INT64],
    sig32: pl.Tensor[[16], pl.DT_INT32],
    src: pl.Tensor[[8, 16], pl.DT_BF16],
    out: pl.Tensor[[8, 16], pl.DT_BF16],
    out2: pl.Tensor[[1], pl.DT_INT64],
    out2f: pl.Tensor[[1, 8], pl.DT_FP32],
):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()
        peer = (me + 1) % nr
        rptr = pld.remote_ptr(win, ws, peer)
        view = pl.make_tensor(rptr, [8, 16], dtype=pl.DT_BF16)
        t = pl.make_tile(pl.TileType(shape=[8, 16], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec), addr=0)
        pl.load(t, view, [0, 0])
        pl.store(view, t, [0, 0], atomic=pl.AtomicType.AtomicAdd)
        a64 = pl.make_tensor(pl.addptr(rptr, 0), [1], dtype=pl.DT_INT64)
        out2[0] = pl.atomic_add(a64, 0, 1)

        # Signals are plain user memory: a one-element view over the slot is what the
        # frontend atomics address, and the pointer decides local vs remote.
        local_slot = pl.make_tensor(pl.addptr(pl.make_ptr(sig), me), [1], dtype=pl.DT_INT64)
        pl.atomic_exch(local_slot, 0, 1)
        while True:
            if local_slot[0] >= 1:
                break
        # The signal word width follows the view's element type.
        sig32_slot = pl.make_tensor(pl.make_ptr(sig32), [1], dtype=pl.DT_INT32)
        pl.atomic_exch(sig32_slot, 0, 1)
        while True:
            if sig32_slot[0] >= 1:
                break

        ch = pld.channel(win, 0x2000, peer=peer)
        wptr = ch.remote_ptr(ws)
        sig_rptr = ch.remote_ptr(sig)
        sig_peer = pl.make_tensor(pl.addptr(sig_rptr, me), [1], dtype=pl.DT_INT64)
        pl.atomic_add(sig_peer, 0, 1)
        ch.lock()
        ch.write(wptr, src, submit=False)
        ch.submit()
        ch.write(pl.make_ptr(pl.addptr(wptr, 64), pl.DT_INT64), 7)
        ch.write_notify(wptr, src, pl.addptr(sig_rptr, me), me + 1)
        ch.unlock()
        ch.read(out, pl.addptr(wptr, 512))
        ch.write_reduce(pl.make_ptr(pl.addptr(wptr, 768), pl.DT_FP32), out2f, op=pld.ReduceOp.SUM)
        ch.atomic_fetch_add(pl.make_ptr(pl.addptr(wptr, 1024), pl.DT_INT64), out2, 1)
        ch.atomic_cas(pl.make_ptr(pl.addptr(wptr, 1032), pl.DT_INT64), out2, 0, 1)
        ch.drain()
        bch = pld.buffer_channel(ch, 0x4000, 256)
        bch.write(wptr, src)
        bch.write(pl.make_ptr(pl.addptr(wptr, 64), pl.DT_INT64), 7)
        bch.write_notify(wptr, src, pl.addptr(sig_rptr, me), me + 1)
        bch.submit()
        bch.read(out, pl.addptr(wptr, 512))
        bch.atomic_fetch_add(pl.make_ptr(pl.addptr(wptr, 1024), pl.DT_INT64), out2, 1)
        bch.write_reduce(pl.make_ptr(pl.addptr(wptr, 768), pl.DT_FP32), out2f, op=pld.ReduceOp.SUM)
        bch.atomic_cas(pl.make_ptr(pl.addptr(wptr, 1032), pl.DT_INT64), out2, 0, 1)
        bch.submit()
        bch.drain()
        ch2 = pld.channel(win, 0x3000, peer=peer)
        ch2.write(pl.addptr(ch2.remote_ptr(ws), 256), src, option=pld.UrmaOption.CQE)
        ch2.drain()
        pld.barrier(win)
        pl.store(out, t, [0, 0])


def test_generated_source_matches_golden(assert_golden):
    assert_golden(_compile_to_cce(_dist_pointer_codegen_kernel), "cce")
