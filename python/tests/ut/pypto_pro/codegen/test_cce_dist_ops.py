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

import os
from pathlib import Path

import pypto_pro.language as pl
import pypto_pro.language.distribute as pld

_GOLDEN_DIR = Path(__file__).parent / "test_cce_dist_ops_src"


def _compile_to_cce(kernel, arch: str = "3510") -> str:
    from pypto_pro.runtime.jit import _add_kernel_header, _assemble_cv_source, _parse_and_codegen_targets
    from pypto_pro.runtime.platform import normalize_arch

    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), normalize_arch(arch), "")
    return _add_kernel_header(_assemble_cv_source(cube, vector)).content


def _source_golden(kernel, golden_name: str) -> str:
    """Whole generated CCE source, compared against the .pypto golden.

    Set PYPTO_RENDER_IR=1 to (re)generate the golden files.
    """
    source = _compile_to_cce(kernel)
    golden_path = _GOLDEN_DIR / f"{golden_name}.pypto"
    if os.environ.get("PYPTO_RENDER_IR"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(source.rstrip("\n") + "\n", encoding="utf-8")
        return source
    golden = golden_path.read_text(encoding="utf-8")
    assert source.rstrip("\n") == golden.rstrip("\n")
    return source


@pl.jit(arch="3510", auto_mutex=False)
def _dist_pointer_codegen_kernel(
    win: pld.Window,
    ws: pl.Tensor[[4096], pl.DT_UINT8],
    sig: pl.Tensor[[16], pl.DT_INT64],
    sig32: pl.Tensor[[16], pl.DT_INT32],
    src: pl.Tensor[[8, 16], pl.DT_BF16],
    idx: pl.Tensor[[1, 8], pl.DT_INT32],
    out: pl.Tensor[[8, 16], pl.DT_BF16],
    out2: pl.Tensor[[1], pl.DT_INT64],
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

        # Signals are plain user memory: the caller offsets the slot and the pointer
        # determines whether the write is local or remote.
        local_slot = pl.addptr(pl.make_ptr(sig), me)
        pld.notify(local_slot, 1)
        pld.wait(local_slot, 1, cmp=pld.Cmp.GE)
        # The signal word width follows the pointer's element type.
        pld.notify(pl.make_ptr(sig32), 1)
        pld.wait(sig32, 1)

        it = pl.make_tile(pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec), addr=0x1000)
        pl.load(it, idx, [0, 0])
        ch = pld.channel(win, 0x2000, peer=peer)
        wptr = ch.remote_ptr(ws)
        sig_rptr = ch.remote_ptr(sig)
        pld.notify(pl.addptr(sig_rptr, me), 1, op=pld.NotifyOp.ATOMIC_ADD)
        ch.write(wptr, src, submit=False)
        ch.submit()
        ch.write(pl.make_ptr(pl.addptr(wptr, 64), pl.DT_INT64), 7)
        ch.write_notify(wptr, src, pl.addptr(sig_rptr, me), me + 1)
        ch.write_indexed(pl.addptr(wptr, 128), src, src_indices=it, count=8)
        ch.drain()
        pld.barrier(win)
        pl.store(out, t, [0, 0])


def test_generated_source_matches_golden():
    _source_golden(_dist_pointer_codegen_kernel, "dist_pointer_codegen")


def test_ub_remote_pointer_lowers_to_remote_ptr():
    cpp = _compile_to_cce(_dist_pointer_codegen_kernel)
    assert "pypto::dist::RemotePtr(" in cpp
    assert "pto::GlobalTensor" in cpp


def test_channel_address_lowers_to_remote_ptr():
    cpp = _compile_to_cce(_dist_pointer_codegen_kernel)
    assert ".RemoteAddr<uint8_t>(" in cpp


def test_channel_write_forms_take_pointer_destinations():
    cpp = _compile_to_cce(_dist_pointer_codegen_kernel)
    assert ".Write<" in cpp and "reinterpret_cast<GM_ADDR>(wptr_0)" in cpp
    assert ".WriteValue<" in cpp
    assert ".WriteNotify<" in cpp
    assert ".WriteIndexed<" in cpp


def test_signal_words_lower_to_pointer_addresses():
    cpp = _compile_to_cce(_dist_pointer_codegen_kernel)
    # notify: SET and ATOMIC_ADD modes; wait: Cmp.GE == 5.
    assert "pypto::dist::Notify<int64_t, 0, true>(" in cpp
    assert "pypto::dist::Notify<int64_t, 1, true>(" in cpp
    assert "pypto::dist::Wait<int64_t, 5>(" in cpp
    # A 32-bit signal tensor lowers to the 32-bit device helpers.
    assert "pypto::dist::Notify<int32_t, 0, true>(" in cpp
    assert "pypto::dist::Wait<int32_t, 0>(" in cpp


def test_pointer_atomic_lowers_to_typed_gm_intrinsic():
    cpp = _compile_to_cce(_dist_pointer_codegen_kernel)
    assert "atomicAdd((__gm__ int64_t*)" in cpp
