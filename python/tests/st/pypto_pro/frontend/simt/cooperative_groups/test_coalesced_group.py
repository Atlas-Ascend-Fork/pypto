# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""System tests for the CoalescedGroup cooperative group."""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
import pytest
import torch

pytestmark = pytest.mark.soc("950")
ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
THREADS = 64
DTYPES = [
    (pl.DT_FP16, torch.float16),
    (pl.DT_BF16, torch.bfloat16),
    (pl.DT_FP32, torch.float32),
    (pl.DT_INT32, torch.int32),
    (pl.DT_UINT32, torch.uint32),
    (pl.DT_INT64, torch.int64),
    (pl.DT_UINT64, torch.uint64),
]


# Launch-time keys specialize the file-level kernels at compile time.
class CoalescedGroupKey:
    NumThreads = TilingKeyField(bits=7, values=[1, 2, 20, 31, 32, 33, 64])


# SIMT functions
@pl.vector_function(mode="simt")
def _coalesced_group_probe(out):
    tid = pl.simt.linear_thread_idx()
    if tid % 2 == 0:
        active = pl.simt.coalesced_threads()
        out[0, tid] = active.thread_rank()
        out[1, tid] = active.size()
        out[2, tid] = active.meta_group_rank()
        out[3, tid] = active.meta_group_size()
        out[4, tid] = active.ballot(tid % 4 == 0)
        out[5, tid] = active.shfl(tid, 0)
        out[6, tid] = active.shfl_up(tid, 1)
        out[7, tid] = active.shfl_down(tid, 1)
        # Observe the actual captured membership instead of assuming that every
        # eligible lane must be coalesced into the same group by the hardware.
        member_mask = pl.simt.cast(0, pl.DT_UINT32)
        for source_rank in pl.range(0, active.size(), 1):
            lane = active.shfl(tid, source_rank) % 32
            member_mask = pl.simt.cast(member_mask | (pl.simt.cast(1, pl.DT_UINT32) << lane), pl.DT_UINT32)
        out[12, tid] = member_mask
        if tid % 4 == 0:
            narrow = pl.simt.coalesced_threads()
            out[8, tid] = active.size()
            out[9, tid] = active.thread_rank()
            out[10, tid] = narrow.size()
            out[11, tid] = narrow.thread_rank()
            narrow_mask = pl.simt.cast(0, pl.DT_UINT32)
            for source_rank in pl.range(0, narrow.size(), 1):
                lane = narrow.shfl(tid, source_rank) % 32
                narrow_mask = pl.simt.cast(narrow_mask | (pl.simt.cast(1, pl.DT_UINT32) << lane), pl.DT_UINT32)
            out[13, tid] = narrow_mask
        active.sync()


@pl.vector_function(mode="simt", max_threads=THREADS)
def _coalesced_group_entry(out):
    _coalesced_group_probe(out)


@pl.vector_function(mode="simt", max_threads=THREADS)
def _coalesced_group_shuffle_probe(src, out):
    group = pl.simt.coalesced_threads()
    tid = pl.simt.linear_thread_idx()
    rank = group.thread_rank()
    value = src[0, tid]
    out[0, tid] = group.shfl(value, 31 - rank)
    out[1, tid] = group.shfl_up(value, 1)
    out[2, tid] = group.shfl_down(value, 1)


# JIT kernels
@pl.jit(mode="simt", max_threads=THREADS)
def _coalesced_group_simt_kernel(out: pl.Tensor[[14, THREADS], pl.DT_UINT32]):
    _coalesced_group_probe(out)


@pl.jit(arch="3510", tiling_key=CoalescedGroupKey)
def _coalesced_group_kernel(out: pl.Tensor[[14, THREADS], pl.DT_UINT32]):
    with pl.section_vector():
        _coalesced_group_entry[NumThreads](out)  # noqa: F821


@pl.jit(arch="3510", datatype={"src": "io_dtype", "out": "io_dtype"})
def _coalesced_group_shuffle_kernel(src: pl.Ptr[pl.DT_UINT8], out: pl.Ptr[pl.DT_UINT8]):
    src_tensor = pl.make_tensor(src, [1, THREADS], [THREADS, 1], dtype=io_dtype)  # noqa: F821
    out_tensor = pl.make_tensor(out, [3, THREADS], [THREADS, 1], dtype=io_dtype)  # noqa: F821
    with pl.section_vector():
        _coalesced_group_shuffle_probe[THREADS](src_tensor, out_tensor)


# Host launch and validation
@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("threads", [1, 2, 20, 31, 32, 33, 64])
def test_coalesced_group_divergence_and_membership_snapshot(threads, standalone):
    torch.npu.set_device(ST_DEVICE)
    rows = [[0xFFFFFFFF] * THREADS for _ in range(14)]
    out = torch.tensor(rows, dtype=torch.uint32, device=ST_DEVICE)
    if standalone:
        _coalesced_group_simt_kernel[1, threads](out)
    else:
        _coalesced_group_kernel[None, 1, {"NumThreads": threads}](out)
    torch.npu.synchronize()
    actual = out.cpu()
    for tid in range(0, threads, 2):
        start = tid // 32 * 32
        mask = actual[12, tid].item()
        members = [start + lane for lane in range(32) if mask & (1 << lane)]
        assert tid in members
        assert all(index < threads and index % 2 == 0 for index in members)
        assert all(actual[12, index].item() == mask for index in members)
        rows[12][tid] = mask
        rank = members.index(tid)
        expected = (
            rank,
            len(members),
            0,
            1,
            sum(1 << rank for rank, index in enumerate(members) if index % 4 == 0),
            members[0],
            members[rank - 1] if rank else tid,
            members[rank + 1] if rank + 1 < len(members) else tid,
        )
        for row, value in zip(rows, expected):
            row[tid] = value
        if tid % 4 == 0:
            narrow_mask = actual[13, tid].item()
            narrow = [start + lane for lane in range(32) if narrow_mask & (1 << lane)]
            assert tid in narrow
            assert all(index < threads and index % 4 == 0 for index in narrow)
            assert all(actual[13, index].item() == narrow_mask for index in narrow)
            rows[13][tid] = narrow_mask
            for row, value in zip(rows[8:12], (len(members), rank, len(narrow), narrow.index(tid))):
                row[tid] = value
    torch.testing.assert_close(actual, torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,torch_dtype", DTYPES)
def test_coalesced_group_shuffle_preserves_every_bit(dtype, torch_dtype):
    if torch_dtype in (torch.float16, torch.bfloat16, torch.float32):
        bits, integer, special = {
            torch.float16: (16, torch.int16, [0, 0x8000, 0x7C00, 0xFC00, 0x7E15, 0xFE37, 1, 0x03FF]),
            torch.bfloat16: (16, torch.int16, [0, 0x8000, 0x7F80, 0xFF80, 0x7FC5, 0xFFC7, 1, 0x007F]),
            torch.float32: (
                32,
                torch.int32,
                [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00125, 0xFFC00137, 1, 0x007FFFFF],
            ),
        }[torch_dtype]
        patterns = special + [
            0x3C00 + i * 17 if bits == 16 else 0x3F800000 + i * 65537 for i in range(THREADS - len(special))
        ]
        signed = [value - (1 << bits) if value & (1 << (bits - 1)) else value for value in patterns]
        source = torch.tensor(signed, dtype=integer).view(torch_dtype).reshape(1, THREADS)
    else:
        bits = 64 if torch_dtype in (torch.int64, torch.uint64) else 32
        values = [
            ((0x80000000 + i * 65537) << 32 | (0x12345678 ^ (i * 0x01010101)))
            if bits == 64
            else (0x80000000 ^ (i * 0x01010101))
            for i in range(THREADS)
        ]
        if torch_dtype in (torch.int32, torch.int64):
            values = [value - (1 << bits) if value & (1 << (bits - 1)) else value for value in values]
        source = torch.tensor(values, dtype=torch_dtype).reshape(1, THREADS)

    out = torch.empty((3, THREADS), dtype=torch_dtype, device=ST_DEVICE)
    _coalesced_group_shuffle_kernel[None, 1, {"src": dtype, "out": dtype}](source.to(ST_DEVICE), out)
    torch.npu.synchronize()
    indices = [[] for _ in range(3)]
    for tid in range(THREADS):
        expected = (
            tid // 32 * 32 + 31 - tid % 32,
            tid - 1 if tid % 32 else tid,
            tid + 1 if tid % 32 < 31 else tid,
        )
        for row, index in zip(indices, expected):
            row.append(index)
    # Byte comparisons preserve signed zeros, NaN payloads and both UINT64 words.
    source_bytes = source.view(torch.uint8).reshape(THREADS, source.element_size())
    expected = source_bytes[torch.tensor(indices, dtype=torch.int64)].reshape(3, -1)
    assert torch.equal(out.cpu().contiguous().view(torch.uint8), expected)
