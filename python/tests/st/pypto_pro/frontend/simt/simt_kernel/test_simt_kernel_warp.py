# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT kernel lane masks, votes, shuffles, and reductions."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
THREADS = 64
WARP_SIZE = 32


@pl.vector_function(mode="simt")
def _warp_ops(info, masks, exchanged, src):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    info[0, tid] = lane
    info[1, tid] = pl.simt.lanemask_eq()
    info[2, tid] = pl.simt.lanemask_le()
    info[3, tid] = pl.simt.lanemask_lt()
    info[4, tid] = pl.simt.lanemask_ge()
    info[5, tid] = pl.simt.lanemask_gt()
    info[6, tid] = pl.simt.warp_all(lane < pl.simt.warp_size())
    info[7, tid] = pl.simt.warp_all(lane < 31)
    info[8, tid] = pl.simt.warp_any(lane == 31)
    info[9, tid] = pl.simt.warp_any(lane == 32)
    masks[0, tid] = pl.simt.warp_active_mask()
    masks[1, tid] = pl.simt.warp_ballot(lane % 2 == 0)
    value = src[tid]
    exchanged[0, tid] = pl.simt.warp_shfl(value, 0)
    exchanged[1, tid] = pl.simt.warp_shfl_up(value, 1)
    exchanged[2, tid] = pl.simt.warp_shfl_down(value, 1)
    exchanged[3, tid] = pl.simt.warp_shfl_xor(value, 1)
    exchanged[4, tid] = pl.simt.warp_reduce_add(value)
    exchanged[5, tid] = pl.simt.warp_reduce_max(value)
    exchanged[6, tid] = pl.simt.warp_reduce_min(value)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_warp(
    info: pl.Tensor[[10, THREADS], pl.DT_INT32],
    masks: pl.Tensor[[2, THREADS], pl.DT_UINT32],
    exchanged: pl.Tensor[[7, THREADS], pl.DT_INT32],
    src: pl.Tensor[[THREADS], pl.DT_INT32],
):
    _warp_ops(info, masks, exchanged, src)


@pytest.mark.soc("950")
def test_simt_kernel_warp_interfaces():
    torch.npu.set_device(ST_DEVICE)
    values = torch.arange(THREADS, dtype=torch.int32) - 40
    info = torch.empty((10, THREADS), dtype=torch.int32, device=ST_DEVICE)
    masks = torch.empty((2, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    exchanged = torch.empty((7, THREADS), dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_warp[1, THREADS](info, masks, exchanged, values.to(ST_DEVICE))
    torch.npu.synchronize()

    lanes = list(range(WARP_SIZE)) * (THREADS // WARP_SIZE)
    expected_info = torch.tensor(
        [
            lanes,
            [1 << lane for lane in lanes],
            [(1 << (lane + 1)) - 1 for lane in lanes],
            [(1 << lane) - 1 for lane in lanes],
            [0xFFFFFFFF ^ ((1 << lane) - 1) for lane in lanes],
            [0xFFFFFFFF ^ ((1 << (lane + 1)) - 1) for lane in lanes],
            [1] * THREADS,
            [0] * THREADS,
            [1] * THREADS,
            [0] * THREADS,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    expected_masks = torch.tensor([[0xFFFFFFFF] * THREADS, [0x55555555] * THREADS], dtype=torch.uint32)
    warps = values.reshape(-1, WARP_SIZE)
    up = warps.clone()
    up[:, 1:] = warps[:, :-1]
    down = warps.clone()
    down[:, :-1] = warps[:, 1:]
    expected_exchanged = torch.stack(
        [
            warps[:, 0].repeat_interleave(WARP_SIZE),
            up.flatten(),
            down.flatten(),
            warps[:, [lane ^ 1 for lane in range(WARP_SIZE)]].flatten(),
            warps.sum(dim=1, dtype=torch.int32).repeat_interleave(WARP_SIZE),
            warps.max(dim=1).values.repeat_interleave(WARP_SIZE),
            warps.min(dim=1).values.repeat_interleave(WARP_SIZE),
        ]
    )
    assert torch.equal(info.cpu(), expected_info)
    assert torch.equal(masks.cpu().to(torch.int64), expected_masks.to(torch.int64))
    assert torch.equal(exchanged.cpu(), expected_exchanged)


@pl.vector_function(mode="simt")
def _shuffle(dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    dst[0, tid] = pl.simt.warp_shfl(value, 0, 1)
    dst[1, tid] = pl.simt.warp_shfl_up(value, 1, 1)
    dst[2, tid] = pl.simt.warp_shfl_down(value, 1, 1)
    dst[3, tid] = pl.simt.warp_shfl_xor(value, 0, 1)
    dst[4, tid] = pl.simt.warp_shfl(value, 1, 2)
    dst[5, tid] = pl.simt.warp_shfl_up(value, 1, 2)
    dst[6, tid] = pl.simt.warp_shfl_down(value, 1, 2)
    dst[7, tid] = pl.simt.warp_shfl_xor(value, 1, 2)
    dst[8, tid] = pl.simt.warp_shfl(value, 3, 4)
    dst[9, tid] = pl.simt.warp_shfl_up(value, 1, 4)
    dst[10, tid] = pl.simt.warp_shfl_down(value, 1, 4)
    dst[11, tid] = pl.simt.warp_shfl_xor(value, 1, 4)
    dst[12, tid] = pl.simt.warp_shfl(value, 7, 8)
    dst[13, tid] = pl.simt.warp_shfl_up(value, 1, 8)
    dst[14, tid] = pl.simt.warp_shfl_down(value, 1, 8)
    dst[15, tid] = pl.simt.warp_shfl_xor(value, 1, 8)
    dst[16, tid] = pl.simt.warp_shfl(value, 15, 16)
    dst[17, tid] = pl.simt.warp_shfl_up(value, 1, 16)
    dst[18, tid] = pl.simt.warp_shfl_down(value, 1, 16)
    dst[19, tid] = pl.simt.warp_shfl_xor(value, 1, 16)
    dst[20, tid] = pl.simt.warp_shfl(value, 31, 32)
    dst[21, tid] = pl.simt.warp_shfl_up(value, 1, 32)
    dst[22, tid] = pl.simt.warp_shfl_down(value, 1, 32)
    dst[23, tid] = pl.simt.warp_shfl_xor(value, 1, 32)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_shuffle(
    dst: pl.Tensor[[24, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _shuffle(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_shuffle_widths():
    torch.npu.set_device(ST_DEVICE)
    source = (torch.arange(THREADS, dtype=torch.float32) - THREADS // 2) * 0.25
    dst = torch.empty((24, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_shuffle[1, THREADS](dst, source.to(ST_DEVICE))
    torch.npu.synchronize()

    samples = source.tolist()
    expected = []
    for width in (1, 2, 4, 8, 16, 32):
        lane_mask = 0 if width == 1 else 1
        expected.extend(
            [
                [samples[index // width * width + width - 1] for index in range(THREADS)],
                [samples[index - 1 if index % width else index] for index in range(THREADS)],
                [samples[index + 1 if index % width < width - 1 else index] for index in range(THREADS)],
                [samples[index ^ lane_mask] for index in range(THREADS)],
            ]
        )
    torch.testing.assert_close(dst.cpu(), torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)


@pl.vector_function(mode="simt")
def _reduce_full_warp(dst, src):
    tid = pl.simt.linear_thread_idx()
    value = src[tid]
    dst[0, tid] = pl.simt.warp_reduce_add(value)
    dst[1, tid] = pl.simt.warp_reduce_max(value)
    dst[2, tid] = pl.simt.warp_reduce_min(value)


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_reduce_full_warp(
    dst: pl.Tensor[[3, THREADS], pl.DT_FP32],
    src: pl.Tensor[[THREADS], pl.DT_FP32],
):
    _reduce_full_warp(dst, src)


@pytest.mark.soc("950")
def test_simt_kernel_reduce_full_warp():
    torch.npu.set_device(ST_DEVICE)
    source = torch.arange(THREADS, dtype=torch.float32) - THREADS // 2
    dst = torch.empty((3, THREADS), dtype=torch.float32, device=ST_DEVICE)
    simt_kernel_reduce_full_warp[1, THREADS](dst, source.to(ST_DEVICE))
    torch.npu.synchronize()

    warps = source.reshape(-1, WARP_SIZE)
    expected = torch.stack(
        [
            warps.sum(dim=1).repeat_interleave(WARP_SIZE),
            warps.max(dim=1).values.repeat_interleave(WARP_SIZE),
            warps.min(dim=1).values.repeat_interleave(WARP_SIZE),
        ]
    )
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


@pl.vector_function(mode="simt")
def _integer_votes(votes, masks, predicates):
    tid = pl.simt.linear_thread_idx()
    predicate = predicates[tid]
    votes[0, tid] = pl.simt.warp_all(predicate)
    votes[1, tid] = pl.simt.warp_any(predicate)
    masks[0, tid] = pl.simt.warp_ballot(predicate)
    if pl.simt.lane_id() < WARP_SIZE // 2:
        masks[1, tid] = pl.simt.warp_active_mask()
        masks[2, tid] = 0
    else:
        masks[1, tid] = 0
        masks[2, tid] = pl.simt.warp_active_mask()


@pl.jit(mode="simt", max_threads=THREADS)
def simt_kernel_integer_votes(
    votes: pl.Tensor[[2, THREADS], pl.DT_INT32],
    masks: pl.Tensor[[3, THREADS], pl.DT_UINT32],
    predicates: pl.Tensor[[THREADS], pl.DT_INT32],
):
    _integer_votes(votes, masks, predicates)


@pytest.mark.soc("950")
def test_simt_kernel_integer_votes_and_divergent_mask():
    torch.npu.set_device(ST_DEVICE)
    predicates = torch.tensor([0] + [2] * (WARP_SIZE - 1), dtype=torch.int32).repeat(THREADS // WARP_SIZE)
    votes = torch.empty((2, THREADS), dtype=torch.int32, device=ST_DEVICE)
    masks = torch.empty((3, THREADS), dtype=torch.uint32, device=ST_DEVICE)
    simt_kernel_integer_votes[1, THREADS](votes, masks, predicates.to(ST_DEVICE))
    torch.npu.synchronize()
    expected_votes = torch.tensor([[0] * THREADS, [1] * THREADS], dtype=torch.int32)
    expected_masks = torch.tensor(
        [
            [0xFFFFFFFE] * THREADS,
            ([0xFFFF] * 16 + [0] * 16) * (THREADS // WARP_SIZE),
            ([0] * 16 + [0xFFFF0000] * 16) * (THREADS // WARP_SIZE),
        ],
        dtype=torch.int64,
    )
    assert torch.equal(votes.cpu(), expected_votes)
    assert torch.equal(masks.cpu().to(torch.int64), expected_masks)
