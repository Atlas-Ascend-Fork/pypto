# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""System tests for the ThreadBlockTile cooperative group."""

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
class ThreadBlockTileKey:
    TileWidth = TilingKeyField(bits=6, values=[1, 2, 4, 8, 16, 32])


# SIMT functions
@pl.vector_function(mode="simt")
def _thread_block_tile_probe(
    pred: pl.Tensor[[1, THREADS], pl.DT_INT32], out: pl.Tensor[[7, THREADS], pl.DT_UINT32], width,
):
    block = pl.simt.this_thread_block()
    group = pl.simt.tiled_partition(block, width)
    tid = block.thread_rank()
    value = pred[0, tid]
    out[0, tid] = group.thread_rank()
    out[1, tid] = group.size()
    out[2, tid] = group.meta_group_rank()
    out[3, tid] = group.meta_group_size()
    out[4, tid] = group.ballot(value)
    out[5, tid] = pl.simt.cast(group.any(value), pl.DT_UINT32)
    out[6, tid] = pl.simt.cast(group.all(value), pl.DT_UINT32)
    group.sync()


@pl.vector_function(mode="simt", max_threads=THREADS)
def _thread_block_tile_entry(
    pred: pl.Tensor[[1, THREADS], pl.DT_INT32], out: pl.Tensor[[7, THREADS], pl.DT_UINT32], width,
):
    _thread_block_tile_probe(pred, out, width)


@pl.vector_function(mode="simt", max_threads=THREADS)
def _thread_block_tile_shuffle_probe(src, out):
    block = pl.simt.this_thread_block()
    tile = pl.simt.tiled_partition(block, TileWidth)  # noqa: F821
    tid = block.thread_rank()
    rank = tile.thread_rank()
    value = src[0, tid]
    out[0, tid] = tile.shfl(value, TileWidth - 1 - rank)  # noqa: F821
    out[1, tid] = tile.shfl_up(value, 1)
    out[2, tid] = tile.shfl_down(value, 1)
    out[3, tid] = tile.shfl_xor(value, TileWidth - 1)  # noqa: F821


# JIT kernels
@pl.jit(mode="simt", max_threads=THREADS)
def _thread_block_tile_simt_kernel(
    width: pl.DT_UINT32, pred: pl.Tensor[[1, THREADS], pl.DT_INT32], out: pl.Tensor[[7, THREADS], pl.DT_UINT32],
):
    _thread_block_tile_probe(pred, out, width)


@pl.jit(mode="simt", max_threads=THREADS)
def _thread_block_tile_runtime_shuffle_kernel(
    width: pl.DT_UINT32,
    offsets: pl.Tensor[[THREADS], pl.DT_UINT32],
    out: pl.Tensor[[3, THREADS], pl.DT_UINT32],
):
    block = pl.simt.this_thread_block()
    tile = pl.simt.tiled_partition(block, width)
    tid = block.thread_rank()
    offset = offsets[tid]
    out[0, tid] = tile.shfl(tid, offset)
    out[1, tid] = tile.shfl_up(tid, offset)
    out[2, tid] = tile.shfl_down(tid, offset)


@pl.jit(arch="3510", tiling_key=ThreadBlockTileKey)
def _thread_block_tile_kernel(pred: pl.Tensor[[1, THREADS], pl.DT_INT32], out: pl.Tensor[[7, THREADS], pl.DT_UINT32]):
    with pl.section_vector():
        _thread_block_tile_entry[THREADS](pred, out, TileWidth)  # noqa: F821


@pl.jit(arch="3510", tiling_key=ThreadBlockTileKey, datatype={"src": "io_dtype", "out": "io_dtype"})
def _thread_block_tile_shuffle_kernel(src: pl.Ptr[pl.DT_UINT8], out: pl.Ptr[pl.DT_UINT8]):
    src_tensor = pl.make_tensor(src, [1, THREADS], [THREADS, 1], dtype=io_dtype)  # noqa: F821
    out_tensor = pl.make_tensor(out, [4, THREADS], [THREADS, 1], dtype=io_dtype)  # noqa: F821
    with pl.section_vector():
        _thread_block_tile_shuffle_probe[THREADS](src_tensor, out_tensor)


# Host launch and validation
@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
def test_thread_block_tile_queries_votes_and_barrier(width, standalone):
    torch.npu.set_device(ST_DEVICE)
    # The three launches cover empty, full and mixed ballots; negative nonzero means true.
    for values in (
        [0] * THREADS,
        [-7] * THREADS,
        [9 if i % 3 == 0 or i % 32 == 31 else 0 for i in range(THREADS)],
    ):
        pred = torch.tensor([values], dtype=torch.int32, device=ST_DEVICE)
        out = torch.empty((7, THREADS), dtype=torch.uint32, device=ST_DEVICE)
        if standalone:
            _thread_block_tile_simt_kernel[1, THREADS](width, pred, out)
        else:
            _thread_block_tile_kernel[None, 1, {"TileWidth": width}](pred, out)
        torch.npu.synchronize()
        rows = [[] for _ in range(7)]
        for tid in range(THREADS):
            start = tid // width * width
            flags = [value != 0 for value in values[start:start + width]]
            expected = (
                tid - start,
                width,
                tid // width,
                THREADS // width,
                sum(1 << i for i, flag in enumerate(flags) if flag),
                int(any(flags)),
                int(all(flags)),
            )
            for row, value in zip(rows, expected):
                row.append(value)
        torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,torch_dtype", DTYPES)
@pytest.mark.parametrize("width", [1, 8, 32])
def test_thread_block_tile_shuffle_preserves_every_bit(dtype, torch_dtype, width):
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

    out = torch.empty((4, THREADS), dtype=torch_dtype, device=ST_DEVICE)
    _thread_block_tile_shuffle_kernel[None, 1, {"TileWidth": width}, {"src": dtype, "out": dtype}](
        source.to(ST_DEVICE), out
    )
    torch.npu.synchronize()
    indices = [[] for _ in range(4)]
    for tid in range(THREADS):
        start = tid // width * width
        rank = tid - start
        expected = (
            start + width - 1 - rank,
            tid - 1 if rank else tid,
            tid + 1 if rank + 1 < width else tid,
            start + (rank ^ (width - 1)),
        )
        for row, index in zip(indices, expected):
            row.append(index)
    # Byte comparisons preserve signed zeros, NaN payloads and both UINT64 words.
    source_bytes = source.view(torch.uint8).reshape(THREADS, source.element_size())
    expected = source_bytes[torch.tensor(indices, dtype=torch.int64)].reshape(4, -1)
    assert torch.equal(out.cpu().contiguous().view(torch.uint8), expected)


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
def test_thread_block_tile_runtime_shuffle_boundaries(width):
    torch.npu.set_device(ST_DEVICE)
    for delta in sorted({0, 1, width - 1, width}):
        offsets = torch.tensor([delta] * THREADS, dtype=torch.uint32).to(ST_DEVICE)
        out = torch.empty((3, THREADS), dtype=torch.uint32, device=ST_DEVICE)
        _thread_block_tile_runtime_shuffle_kernel[1, THREADS](width, offsets, out)
        torch.npu.synchronize()
        rows = [[] for _ in range(3)]
        for tid in range(THREADS):
            rank = tid % width
            expected = (
                tid - rank + delta % width,
                tid - delta if delta <= rank else tid,
                tid + delta if rank + delta < width else tid,
            )
            for row, value in zip(rows, expected):
                row.append(value)
        torch.testing.assert_close(out.cpu(), torch.tensor(rows, dtype=torch.uint32), rtol=0, atol=0)
