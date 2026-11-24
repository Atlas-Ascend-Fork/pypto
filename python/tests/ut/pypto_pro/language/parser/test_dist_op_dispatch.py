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
from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidShape, InvalidType, InvalidVal
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
# UrmaOption — the WQE control flag set (mirrors hcomm WqeEntry)
# =============================================================================


def test_urma_option_enum_members():
    assert issubclass(pld.UrmaOption, enum.IntFlag)
    assert set(pld.UrmaOption.__members__) == {
        "ODR_NONE", "ODR_RELAX", "ODR_STRONG", "ODR_IN_ORDER_COMPLETION",
        "FENCE", "CQE", "DEFAULT",
    }
    assert int(pld.UrmaOption.ODR_NONE) == 0
    assert int(pld.UrmaOption.ODR_RELAX) == 1 << 0
    assert int(pld.UrmaOption.ODR_STRONG) == 1 << 1
    assert int(pld.UrmaOption.ODR_IN_ORDER_COMPLETION) == 1 << 2
    assert int(pld.UrmaOption.FENCE) == 1 << 3
    assert int(pld.UrmaOption.CQE) == 1 << 5
    assert int(pld.UrmaOption.DEFAULT) == 0x2D


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

def test_dist_channel_read():
    """An engine read takes a local GM destination and a remote source pointer."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            src_rptr = ch.remote_ptr(src)
            ch.read(dst, src_rptr, submit=False, timeout=101)
    _assert_ir(kernel, "test_dist_channel_read")

def test_channel_read_requires_a_tensor_destination():
    """The fetched bytes land in a local GM tensor, not in another remote pointer."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            src_rptr = ch.remote_ptr(src)
            ch.read(dst_rptr, src_rptr)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "must be a kernel GM tensor" in str(caught.value)

def test_channel_read_rejects_strided_destination():
    """The engine writes num_bytes contiguously, so a strided view must not be accepted."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 16], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            src_rptr = ch.remote_ptr(src)
            view = pl.make_tensor(dst, [1, 8], [16, 1])
            ch.read(view, src_rptr)

    with pytest.raises(InvalidShape) as caught:
        _parse(kernel)
    assert "ch.read destination must be dense row-major" in str(caught.value)

def test_channel_write_rejects_strided_source():
    """The engine reads num_bytes contiguously, so a strided view must not be accepted."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 16], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            view = pl.make_tensor(src, [1, 8], [16, 1])
            ch.write(ch.remote_ptr(dst), view)

    with pytest.raises(InvalidShape) as caught:
        _parse(kernel)
    assert "ch.write source must be dense row-major" in str(caught.value)

def test_dist_channel_write_reduce():
    """Engine reduction carries the wire reduce_op from a supported source dtype."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_BF16],
               src: pl.Tensor[[1, 8], pl.DT_BF16]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.write_reduce(dst_rptr, src, op=pld.ReduceOp.MAX)
    _assert_ir(kernel, "test_dist_channel_write_reduce")

def test_channel_write_reduce_rejects_unsupported_dtype():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT64],
               src: pl.Tensor[[1, 8], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.write_reduce(dst_rptr, src, op=pld.ReduceOp.SUM)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "does not support dtype" in str(caught.value)

def test_channel_write_reduce_requires_op():
    """The reduction mode is explicit: a lossy MAX/MIN is never a silent default."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.write_reduce(dst_rptr, src)

    with pytest.raises(InvalidArgument) as caught:
        _parse(kernel)
    assert "requires op=pld.ReduceOp.SUM/MAX/MIN" in str(caught.value)

def test_channel_write_reduce_rejects_bad_op():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.write_reduce(dst_rptr, src, op=1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "must be a ReduceOp" in str(caught.value)

def test_dist_channel_atomic_fetch_add():
    """Fetch-and-add takes the previous remote value into a local one-element tensor."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT64],
               fetch: pl.Tensor[[1], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.atomic_fetch_add(dst_rptr, fetch, 5, submit=False, timeout=101)
    _assert_ir(kernel, "test_dist_channel_atomic_fetch_add")

def test_dist_channel_atomic_cas():
    """Compare-and-swap carries expected and desired with the fetch destination."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT64],
               fetch: pl.Tensor[[1], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.atomic_cas(dst_rptr, fetch, 3, 4)
    _assert_ir(kernel, "test_dist_channel_atomic_cas")

def test_channel_atomic_rejects_a_float_pointer():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_FP32],
               fetch: pl.Tensor[[1], pl.DT_FP32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.atomic_fetch_add(dst_rptr, fetch, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "32/64-bit integer word" in str(caught.value)

def test_channel_atomic_fetch_dtype_must_match():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT64],
               fetch: pl.Tensor[[1], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.atomic_fetch_add(dst_rptr, fetch, 1)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "fetch tensor dtype must match" in str(caught.value)

def test_channel_atomic_fetch_must_be_single_element():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT64],
               fetch: pl.Tensor[[2], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            ch.atomic_fetch_add(dst_rptr, fetch, 1)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "exactly one" in str(caught.value)

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


def test_dist_channel_write_notify_submit_false():
    """The fused notify WQE can be staged and published by a later submit."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32], sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            dst_rptr = ch.remote_ptr(dst)
            sig_rptr = ch.remote_ptr(sig)
            ch.write_notify(dst_rptr, src, pl.addptr(sig_rptr, 2), 5, submit=False)
    _assert_ir(kernel, "test_dist_channel_write_notify_submit_false")


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
    the shared form: ``pld.channel(win, buffer)`` without ``peer`` is itself legal
    (clos meshes), only the missing assignment is rejected."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.channel(win, 64, peer=1)

    with pytest.raises(InvalidOperation) as caught:
        _parse(kernel)
    assert "must be assigned to a variable" in str(caught.value)

def test_dist_channel_per_aiv():
    """Omitting ``peer`` is the shared spelling (peer == -1, the calling AIV's jetty)."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            ch = pld.channel(win, 64)
            ch.is_shared()
    _assert_ir(kernel, "test_dist_channel_per_aiv")

def test_channel_buffer_alignment_is_checked():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            _ = pld.channel(win, 65)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "64-byte aligned" in str(caught.value)


def test_dist_channel_urma_option():
    """A per-call WQE option rides the write op as a compile-time flag byte."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, src: pl.Tensor[[8, 16], pl.DT_BF16], dst: pl.Tensor[[8, 16], pl.DT_BF16],
               sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 0x2000, peer=1)
            ch.write(ch.remote_ptr(dst), src, option=pld.UrmaOption.CQE)
            ch.write_notify(ch.remote_ptr(dst), src, ch.remote_ptr(sig), 1, option=pld.UrmaOption.CQE)
    _assert_ir(kernel, "test_dist_channel_urma_option")


def test_channel_option_requires_cqe():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, src: pl.Tensor[[8, 16], pl.DT_BF16], dst: pl.Tensor[[8, 16], pl.DT_BF16]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            ch.write(ch.remote_ptr(dst), src, option=pld.UrmaOption.ODR_STRONG)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "CQE" in str(caught.value)


def test_channel_option_rejects_conflicting_order_bits():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, src: pl.Tensor[[8, 16], pl.DT_BF16], dst: pl.Tensor[[8, 16], pl.DT_BF16]):
        with pl.section_vector():
            # 0x2F = DEFAULT | ODR_STRONG: both execution-order bits set.
            ch = pld.channel(win, 64, peer=1)
            ch.write(ch.remote_ptr(dst), src, option=0x2F)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "ODR_RELAX" in str(caught.value)


def test_channel_option_rejects_unsupported_bits():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, src: pl.Tensor[[8, 16], pl.DT_BF16], dst: pl.Tensor[[8, 16], pl.DT_BF16]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            ch.write(ch.remote_ptr(dst), src, option=0x10)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel)
    assert "unsupported WQE flag bits" in str(caught.value)


def test_channel_remote_ptr_needs_one_operand():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            ch.remote_ptr(dst, dst)

    with pytest.raises(InvalidArgument) as caught:
        _parse(kernel)
    assert "exactly one tensor or pointer" in str(caught.value)


# =============================================================================
# buffer_channel — batch staging composed on an opened channel
# =============================================================================


def test_dist_channel_lock():
    """lock/unlock wrap a post sequence on the base channel."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            ch.lock()
            ch.write(rptr, src, submit=False)
            ch.submit()
            ch.drain()
            ch.unlock()
    _assert_ir(kernel, "test_dist_channel_lock")


def test_dist_buffer_channel_lifecycle():
    """Bind a staging buffer, append two writes, publish once, drain."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            bch = pld.buffer_channel(ch, 512, 256)
            bch.write(rptr, src)
            bch.write(rptr, src)
            bch.submit()
            bch.drain()
    _assert_ir(kernel, "test_dist_buffer_channel_lifecycle")


def test_dist_buffer_channel_value_write():
    """A scalar append takes its dtype from the destination pointer, like ch.write."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            bch = pld.buffer_channel(ch, 512, 256)
            bch.write(rptr, 7)
    _assert_ir(kernel, "test_dist_buffer_channel_value_write")


def test_dist_buffer_channel_engine_ops_stage_too():
    """read/atomics stage into the ring like writes; one submit publishes the batch."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32], fetch: pl.Tensor[[1], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            src_rptr = ch.remote_ptr(src)
            dst_rptr = ch.remote_ptr(dst)
            bch = pld.buffer_channel(ch, 512, 512)
            bch.write(dst_rptr, src)
            bch.read(dst, src_rptr)
            bch.write_reduce(dst_rptr, src, op=pld.ReduceOp.SUM)
            bch.atomic_fetch_add(dst_rptr, fetch, 3)
            bch.atomic_cas(dst_rptr, fetch, 0, 3)
            bch.submit()
            bch.drain()
    _assert_ir(kernel, "test_dist_buffer_channel_engine_ops_stage_too")


def test_buffer_channel_write_rejects_submit():
    """Appends only stage; submit/timeout do not exist on bch.write."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            bch = pld.buffer_channel(ch, 512, 256)
            bch.write(rptr, src, submit=True)

    with pytest.raises(InvalidArgument) as caught:
        _parse(kernel)
    assert "only stages" in str(caught.value)


def test_dist_buffer_channel_option():
    """option= selects the staged body flag byte; DEFAULT's CQE bit is stripped.

    DEFAULT (0x2D) lands as the fence-only body 0x0D: bch.submit adds fence+cqe to
    the last WQEBB when it publishes, so one completion covers the batch."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            bch = pld.buffer_channel(ch, 512, 256)
            bch.write(rptr, src, option=pld.UrmaOption.DEFAULT)
            bch.write(rptr, 7)
            bch.submit()
            bch.drain()
    _assert_ir(kernel, "test_dist_buffer_channel_option")


def test_dist_buffer_channel_write_notify():
    """write_notify stages a two-WQEBB append; the signal fires when the batch publishes."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32],
               src: pl.Tensor[[1, 8], pl.DT_INT32], sig: pl.Tensor[[16], pl.DT_INT64]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            rptr = ch.remote_ptr(dst)
            bch = pld.buffer_channel(ch, 512, 256)
            bch.write(rptr, src)
            bch.write_notify(rptr, src, pl.addptr(pl.make_ptr(sig), 1), 5)
            bch.submit()
            bch.drain()
    _assert_ir(kernel, "test_dist_buffer_channel_write_notify")


def test_buffer_channel_rejects_remote_ptr_and_is_shared():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window, dst: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            bch = pld.buffer_channel(ch, 512, 256)
            bch.remote_ptr(dst)

    with pytest.raises(InvalidOperation) as caught:
        _parse(kernel)
    assert "no method 'remote_ptr'" in str(caught.value)


def test_buffer_channel_validates_buffer_args():
    """buf_addr must be 64-B aligned; buf_bytes a 64-B multiple of >= 128."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel_misaligned(win: pld.Window):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            pld.buffer_channel(ch, 544, 256)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel_misaligned)
    assert "64-byte aligned" in str(caught.value)

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel_too_small(win: pld.Window):
        with pl.section_vector():
            ch = pld.channel(win, 64, peer=1)
            pld.buffer_channel(ch, 512, 64)

    with pytest.raises(InvalidVal) as caught:
        _parse(kernel_too_small)
    assert "at least 128" in str(caught.value)


def test_buffer_channel_requires_channel_handle():
    """The first argument must be a pld.channel handle, not a window."""

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(win: pld.Window):
        with pl.section_vector():
            pld.buffer_channel(win, 512, 256)

    with pytest.raises(InvalidType) as caught:
        _parse(kernel)
    assert "pld.channel handle" in str(caught.value)
