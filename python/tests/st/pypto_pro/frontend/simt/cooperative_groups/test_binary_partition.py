# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""System tests for the binary_partition cooperative-group operation."""

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
class BinaryPartitionKey:
    UseCoalescedParent = TilingKeyField(bits=1, values=[0, 1])


class BinaryPartitionTileKey:
    TileWidth = TilingKeyField(bits=6, values=[1, 8, 32])


# SIMT functions
@pl.vector_function(mode="simt", max_threads=THREADS)
def _binary_partition_probe(
    predicates: pl.Tensor[[1, THREADS], pl.DT_INT32], out: pl.Tensor[[14, THREADS], pl.DT_UINT32]
):
    block = pl.simt.this_thread_block()
    tid = block.thread_rank()
    if UseCoalescedParent:  # noqa: F821
        parent = pl.simt.coalesced_threads()
    else:
        parent = pl.simt.tiled_partition(block, 32)
    predicate = predicates[0, tid] != 0
    group = pl.simt.binary_partition(parent, predicate)
    # Metadata must keep the predicate from creation, not this later assignment.
    predicate = predicates[0, tid] == 0
    out[0, tid] = group.thread_rank()
    out[1, tid] = group.size()
    out[2, tid] = group.meta_group_rank()
    out[3, tid] = group.meta_group_size()
    out[4, tid] = group.ballot(tid % 3 == 0)
    out[5, tid] = group.shfl(tid, group.size() + 1)
    out[6, tid] = group.shfl_up(tid, 1)
    out[7, tid] = group.shfl_down(tid, 1)
    nested = pl.simt.binary_partition(group, predicate or tid % 4 < 2)
    out[8, tid] = nested.thread_rank()
    out[9, tid] = nested.size()
    out[10, tid] = nested.meta_group_rank()
    out[11, tid] = nested.meta_group_size()
    out[12, tid] = pl.simt.cast(group.any(tid % 3 == 0), pl.DT_UINT32)
    out[13, tid] = pl.simt.cast(group.all(tid % 3 == 0), pl.DT_UINT32)


@pl.vector_function(mode="simt", max_threads=THREADS)
def _binary_partition_shuffle_probe(src, out):
    block = pl.simt.this_thread_block()
    tile = pl.simt.tiled_partition(block, TileWidth)  # noqa: F821
    tid = block.thread_rank()
    rank = tile.thread_rank()
    value = src[0, tid]
    group = pl.simt.binary_partition(tile, rank % 3 == 0)
    out[0, tid] = group.shfl(value, group.size() - 1 - group.thread_rank())
    out[1, tid] = group.shfl_up(value, 1)
    out[2, tid] = group.shfl_down(value, 1)
    # An out-of-range UINT32 offset must select the caller even if rank arithmetic would wrap.
    delta = pl.simt.cast(0xFFFFFFFF, pl.DT_UINT32)
    out[3, tid] = group.shfl_up(value, delta)
    out[4, tid] = group.shfl_down(value, delta)


# JIT kernels
@pl.jit(arch="3510", tiling_key=BinaryPartitionKey)
def _binary_partition_kernel(
    predicates: pl.Tensor[[1, THREADS], pl.DT_INT32],
    out: pl.Tensor[[14, THREADS], pl.DT_UINT32],
):
    with pl.section_vector():
        _binary_partition_probe[THREADS](predicates, out)


@pl.jit(arch="3510", tiling_key=BinaryPartitionTileKey, datatype={"src": "io_dtype", "out": "io_dtype"})
def _binary_partition_shuffle_kernel(src: pl.Ptr[pl.DT_UINT8], out: pl.Ptr[pl.DT_UINT8]):
    src_tensor = pl.make_tensor(src, [1, THREADS], [THREADS, 1], dtype=io_dtype)  # noqa: F821
    out_tensor = pl.make_tensor(out, [5, THREADS], [THREADS, 1], dtype=io_dtype)  # noqa: F821
    with pl.section_vector():
        _binary_partition_shuffle_probe[THREADS](src_tensor, out_tensor)


@pl.jit(mode="simt", max_threads=THREADS)
def _binary_partition_runtime_shuffle_kernel(
    predicates: pl.Tensor[[THREADS], pl.DT_INT32],
    offsets: pl.Tensor[[THREADS], pl.DT_UINT32],
    out: pl.Tensor[[3, THREADS], pl.DT_INT32],
):
    block = pl.simt.this_thread_block()
    tile = pl.simt.tiled_partition(block, 32)
    tid = block.thread_rank()
    group = pl.simt.binary_partition(tile, predicates[tid] != 0)
    value = pl.simt.cast(tid * 17 + 101, pl.DT_INT32)
    offset = offsets[tid]
    out[0, tid] = group.shfl(value, offset)
    out[1, tid] = group.shfl_up(value, offset)
    out[2, tid] = group.shfl_down(value, offset)


# Host launch and validation
@pytest.mark.parametrize("coalesced_parent", [False, True])
@pytest.mark.parametrize("mask", [0, 0xFFFFFFFF, 1, 0x80000000, 0xAAAAAAAA, 0x49249249])
def test_binary_partition_sparse_uniform_nested_and_predicate_snapshot(coalesced_parent, mask):
    flags = [bool((mask if i < 32 else mask ^ 0xFFFFFFFF) & (1 << (i % 32))) for i in range(THREADS)]
    predicates = torch.tensor(
        [[-7 if flag else 0 for flag in flags]],
        dtype=torch.int32,
        device=ST_DEVICE,
    )
    out = torch.empty((14, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    _binary_partition_kernel[None, 1, {"UseCoalescedParent": int(coalesced_parent)}](predicates, out)
    torch.npu.synchronize()
    rows = [[] for _ in range(14)]
    for tid in range(THREADS):
        parent = list(range(tid // 32 * 32, (tid // 32 + 1) * 32))
        members = [index for index in parent if flags[index] == flags[tid]]
        rank = members.index(tid)
        split = len(members) != len(parent)
        nested_flags = {index: not flags[index] or index % 4 < 2 for index in members}
        nested = [index for index in members if nested_flags[index] == nested_flags[tid]]
        nested_split = len(nested) != len(members)
        expected = (
            rank,
            len(members),
            int(flags[tid]) if split else 0,
            2 if split else 1,
            sum(1 << member_rank for member_rank, index in enumerate(members) if index % 3 == 0),
            members[1 % len(members)],
            members[rank - 1] if rank else tid,
            members[rank + 1] if rank + 1 < len(members) else tid,
            nested.index(tid),
            len(nested),
            int(nested_flags[tid]) if nested_split else 0,
            2 if nested_split else 1,
            int(any(index % 3 == 0 for index in members)),
            int(all(index % 3 == 0 for index in members)),
        )
        for row, value in zip(rows, expected):
            row.append(value)
    torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,torch_dtype", DTYPES)
@pytest.mark.parametrize("width", [1, 8, 32])
def test_binary_partition_shuffle_preserves_every_bit(dtype, torch_dtype, width):
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

    out = torch.empty((5, THREADS), dtype=torch_dtype, device=ST_DEVICE)
    _binary_partition_shuffle_kernel[None, 1, {"TileWidth": width}, {"src": dtype, "out": dtype}](
        source.to(ST_DEVICE), out
    )
    torch.npu.synchronize()
    indices = [[] for _ in range(5)]
    for tid in range(THREADS):
        start = tid // width * width
        rank = tid - start
        members = [index for index in range(start, start + width) if ((index - start) % 3 == 0) == (rank % 3 == 0)]
        group_rank = members.index(tid)
        expected = (
            members[-1 - group_rank],
            members[group_rank - 1] if group_rank else tid,
            members[group_rank + 1] if group_rank + 1 < len(members) else tid,
            tid,
            tid,
        )
        for row, index in zip(indices, expected):
            row.append(index)
    # Byte comparisons preserve signed zeros, NaN payloads and both UINT64 words.
    source_bytes = source.view(torch.uint8).reshape(THREADS, source.element_size())
    expected = source_bytes[torch.tensor(indices, dtype=torch.int64)].reshape(5, -1)
    assert torch.equal(out.cpu().contiguous().view(torch.uint8), expected)


@pytest.mark.parametrize(
    "mask,delta",
    [
        (0x5, 0),
        (0x5, 1),
        (0x5, 2),
        (0x5, 30),
        (0x5, 32),
        (0x5, 0xFFFFFFFF),
        (0x80000001, 1),
        (0x80000001, 2),
        (0x80000001, 0xFFFFFFFF),
        (0xAAAAAAAA, 15),
        (0xAAAAAAAA, 16),
        (0, 31),
        (0xFFFFFFFF, 32),
    ],
)
def test_binary_partition_runtime_shuffle_boundaries(mask, delta):
    torch.npu.set_device(ST_DEVICE)
    flags = [bool(mask & (1 << (tid % 32))) for tid in range(THREADS)]
    predicates = torch.tensor([-7 if flag else 0 for flag in flags], dtype=torch.int32, device=ST_DEVICE)
    offsets = torch.tensor([delta] * THREADS, dtype=torch.uint32).to(ST_DEVICE)
    out = torch.empty((3, THREADS), dtype=torch.int32, device=ST_DEVICE)
    _binary_partition_runtime_shuffle_kernel[1, THREADS](predicates, offsets, out)
    torch.npu.synchronize()
    rows = [[] for _ in range(3)]
    for tid in range(THREADS):
        start = tid // 32 * 32
        members = [index for index in range(start, start + 32) if flags[index] == flags[tid]]
        rank = members.index(tid)
        sources = (
            members[delta % len(members)],
            members[rank - delta] if delta <= rank else tid,
            members[rank + delta] if rank + delta < len(members) else tid,
        )
        for row, source in zip(rows, sources):
            row.append(source * 17 + 101)
    torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.int32), rtol=0, atol=0)
