#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import enum
import os
from pathlib import Path
import re

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidType, InvalidVal
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld
import pytest

_GOLDEN_DIR = Path(__file__).parent / "test_dist_op_dispatch_ir"


def _parse(kernel, target=ir.SectionKind.Vector):
    program, _ = kernel.to_kernel_def().parse_target_program(target)
    return program.get_function(kernel.__name__)


def _assert_ir(kernel, golden_name):
    """Whole parsed function text (memref ids normalized) vs the ``.pypto`` golden."""
    func = _parse(kernel)
    assert isinstance(func, ir.Function)
    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(func))
    golden_path = _GOLDEN_DIR / f"{golden_name}.pypto"
    if os.environ.get("PYPTO_RENDER_IR"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(printed.rstrip("\n") + "\n", encoding="utf-8")
        return
    assert printed.rstrip("\n") == golden_path.read_text(encoding="utf-8").rstrip("\n")


# =============================================================================
# Protocol — the pld-level region access-tier selector
# =============================================================================


def test_protocol_enum_members():
    assert pld.Protocol.UB_MEM == 1
    assert pld.Protocol.URMA == 2
    assert {m.name for m in pld.Protocol} == {"UB_MEM", "URMA"}
    assert issubclass(pld.Protocol, enum.IntFlag)
    assert int(pld.Protocol.UB_MEM | pld.Protocol.URMA) == 3


# =============================================================================
# barrier — the uniform-path reference op
# =============================================================================


def test_dist_barrier_defaults():
    """Default barrier attributes and the explicit knob spellings."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.barrier(win)
            pld.barrier(win, sync_local=False, flush_store=False)
    _assert_ir(kernel, "test_dist_barrier_defaults")

def test_dist_barrier_timeout_forms():
    """Timeout given positionally and by keyword, with knob attributes."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.barrier(win, 1000)
            pld.barrier(win, timeout=2000, sync_local=False, flush_store=True)
    _assert_ir(kernel, "test_dist_barrier_timeout_forms")

def test_dist_barrier_timeout_none():
    """``timeout=None`` parses like omission (forever)."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.barrier(win, timeout=None)
    _assert_ir(kernel, "test_dist_barrier_timeout_none")

def test_barrier_timeout_must_be_compile_time():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, i: pl.Tensor[[1], pl.DT_INT64]):
        with pl.section_vector():
            pld.barrier(win, i[0])

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "timeout" in str(caught.value)
    assert "compile-time" in str(caught.value)


# =============================================================================
# remote_ptr — typed device-side pointer over a registered region
# =============================================================================


def test_dist_remote_ptr_dtype():
    """A registered tensor argument becomes a typed pointer; the region dtype is kept."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, buf: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ptr = pld.remote_ptr(win, buf, 1)
            view = pl.make_tensor(pl.addptr(ptr, 8), [1, 8], dtype=pl.DT_INT32)
            tile = pl.make_tile(
                pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec), addr=0)
            pl.load(tile, view, [0, 0])
    _assert_ir(kernel, "test_dist_remote_ptr_dtype")

def test_dist_remote_ptr_pointer_parameter():
    """A pointer kernel parameter passes through with its dtype."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, workspace: pl.Ptr[pl.DT_INT32]):
        with pl.section_vector():
            ptr = pld.remote_ptr(win, workspace, 1)
            pl.printf("%p\n", ptr)
    _assert_ir(kernel, "test_dist_remote_ptr_pointer_parameter")

def test_dist_remote_ptr_arithmetic():
    """Direct pointer arithmetic on the returned typed pointer."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, buf: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            off = pld.remote_ptr(win, buf, 1) + 4
            view = pl.make_tensor(off, [1, 8], dtype=pl.DT_INT32)
            tile = pl.make_tile(
                pl.TileType(shape=[1, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec), addr=0)
            pl.load(tile, view, [0, 0])
    _assert_ir(kernel, "test_dist_remote_ptr_arithmetic")

def test_dist_remote_ptr_interior():
    """An interior pointer (addptr(make_ptr(buf), 8)) is accepted."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, buf: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ptr = pld.remote_ptr(win, pl.addptr(pl.make_ptr(buf), 8), 1)
            pl.printf("%p\n", ptr)
    _assert_ir(kernel, "test_dist_remote_ptr_interior")

def test_remote_ptr_rejects_a_non_region_operand():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, buf: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            pld.remote_ptr(win, 7, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "must be a kernel tensor parameter" in str(caught.value)


# =============================================================================
# notify / wait — pointer targets, enum and timeout kwargs
# =============================================================================


def test_dist_notify_forms():
    """Remote pointer slot, ATOMIC_ADD mode, and the local-tensor form."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64], peer: pl.DT_INT64):
        with pl.section_vector():
            remote = pld.remote_ptr(win, sig, peer)
            slot = pl.addptr(remote, 3)
            pld.notify(slot, 7)
            pld.notify(slot, 8, op=pld.NotifyOp.ATOMIC_ADD)
            pld.notify(sig, 5)  # local tensor -> its base pointer
    _assert_ir(kernel, "test_dist_notify_forms")

def test_dist_notify_uint32_pointer():
    """The signal word follows a 32-bit pointer dtype."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_UINT32]):
        with pl.section_vector():
            pld.notify(sig, 1)
    _assert_ir(kernel, "test_dist_notify_uint32_pointer")

def test_notify_requires_a_pointer_or_tensor():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.notify(3, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "kernel tensor parameter" in str(caught.value)

def test_notify_rejects_a_non_integer_signal_pointer():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_BF16]):
        with pl.section_vector():
            pld.notify(sig, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "integer signal words" in str(caught.value)

def test_notify_rejects_an_unknown_op_mode():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            pld.notify(sig, 1, op=3)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "op must be NotifyOp.SET or NotifyOp.ATOMIC_ADD" in str(caught.value)

def test_dist_notify_flush_store():
    """``flush_store=False`` lands on the notify IR attributes."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            pld.notify(sig, 7, flush_store=False)
    _assert_ir(kernel, "test_dist_notify_flush_store")

def test_notify_flush_store_must_be_compile_time():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64], flag: pl.DT_BOOL):
        with pl.section_vector():
            pld.notify(sig, 7, flush_store=flag)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "'flush_store' must be a compile-time bool" in str(caught.value)

def test_dist_wait_timeout_none():
    """A local pointer slot with ``cmp=GE`` and ``timeout=None``."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            local = pl.addptr(pl.make_ptr(sig), 2)
            pld.wait(local, 1, cmp=pld.Cmp.GE, timeout=None)
    _assert_ir(kernel, "test_dist_wait_timeout_none")

def test_dist_wait_cmp_enum():
    """A variable-bound Cmp enum is accepted."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            mode = pld.Cmp.GE
            pld.wait(sig, 1, cmp=mode)
    _assert_ir(kernel, "test_dist_wait_cmp_enum")

def test_dist_wait_int32_pointer():
    """The wait word follows a 32-bit pointer dtype."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            pld.wait(sig, 1)

    _assert_ir(kernel, "test_dist_wait_int32_pointer")


def test_wait_rejects_a_non_integer_signal_pointer():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_FP32]):
        with pl.section_vector():
            pld.wait(sig, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "integer signal words" in str(caught.value)


# =============================================================================
# channel — the AST-registered exception (lvalue let-binding)
# =============================================================================


def test_dist_channel_lifecycle():
    """Open, remote address, write, submit, drain on one channel."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32], sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            ch.write(rptr, src, submit=False)
            ch.submit()
            ch.drain()
            pld.notify(pl.addptr(pl.make_ptr(sig), 1), 1)
    _assert_ir(kernel, "test_dist_channel_lifecycle")

def test_dist_channel_write_value():
    """A scalar write takes its dtype from the destination pointer."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            ch.write(rptr, 7)
    _assert_ir(kernel, "test_dist_channel_write_value")

def test_channel_transfers_reject_per_call_peer():
    """The channel binds its destination at creation; transfers take no peer argument."""

    @pl.jit(arch="3510", auto_mutex=False)
    def write_kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32], src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            ch.write(1, rptr, src)

    with pytest.raises(InvalidArgument) as caught:
        _parse(write_kernel)
    assert "ch.write requires (dst_ptr, tensor or scalar)" in str(caught.value)

    @pl.jit(arch="3510", auto_mutex=False)
    def write_notify_kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
                            src: pl.Tensor[[1, 8], pl.DT_INT32], sig: pl.Ptr[pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            ch.write_notify(1, rptr, src, sig, 1)

    with pytest.raises(InvalidArgument) as caught:
        _parse(write_notify_kernel)
    assert "ch.write_notify requires (dst_ptr, tensor, signal_ptr, val)" in str(caught.value)

def test_dist_channel_write_notify():
    """A write fused with a hardware notify takes a remote signal pointer."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32], sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            sig_rptr = ch.remote_ptr(sig)
            ch.write_notify(dst_rptr, src, pl.addptr(sig_rptr, 2), 5)
    _assert_ir(kernel, "test_dist_channel_write_notify")

def test_channel_write_notify_rejects_a_32bit_signal_pointer():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32], sig: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            sig_rptr = ch.remote_ptr(sig)
            ch.write_notify(dst_rptr, src, pl.addptr(sig_rptr, 2), 5)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    # The hardware notify value is an aligned uint64 word.
    assert "64-bit integer signal words" in str(caught.value)

def test_channel_write_notify_rejects_a_local_tensor_signal():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32], sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            ch.write_notify(rptr, src, sig, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "remote pointer resolved by ch.remote_ptr" in str(caught.value)

def test_channel_requires_assignment():
    """A channel must be the RHS of a simple assignment — the let-bound variable name is
    part of its lowering. The peer= spelling here keeps the negative example separate from
    the multi-issuer form: ``pld.channel(win, buffer)`` without ``peer`` is itself legal
    (clos meshes), only the missing assignment is rejected."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.channel(win, 64, peer=1)

    with pytest.raises(InvalidOperation) as caught:
        _parse(kernel)
    assert "must be assigned to a variable" in str(caught.value)

def test_dist_channel_multi_issuer():
    """Omitting ``peer`` is the multi-issuer spelling (peer == -1, AIV jetty)."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            ch = pld.channel(win, 64)
            ch.is_exclusive()
    _assert_ir(kernel, "test_dist_channel_multi_issuer")

def test_channel_buffer_alignment_is_checked():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            _ = pld.channel(win, 65)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "64-byte aligned" in str(caught.value)

def test_channel_remote_ptr_needs_one_operand():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            ch.remote_ptr(dst, dst)

    with pytest.raises(InvalidArgument) as caught:
        _parse(kernel)
    assert "exactly one tensor or pointer" in str(caught.value)
