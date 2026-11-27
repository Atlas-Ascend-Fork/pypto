# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tile computation workloads with AscendC references for VF fusion experiments.

Each ordinary Python helper describes one candidate computation region. Its
arguments are input tiles and scalars, then workspace tiles, then output tiles.
The tile helpers are inlined. SwiGLU and FA online softmax also have handwritten
register VFs. Each kernel exports its results after the end timestamp so that
the compiler retains every output in the measured region.
Run pytest with -s to see the median device cycles and microseconds per compute
region. The interval includes issuing the operations and draining the pipeline.
"""

import os
from statistics import median

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu


def _make_workload_tile(tile_type, addr):
    group = pl.make_tile_group(type=tile_type, addrs=addr, mutex_ids="auto", depth=1)
    return group.current()


def _synced_cycle():
    # The system counter is read on the scalar pipe. Drain outstanding work first.
    pl.system.bar_all()
    return pl.get_cycle()


def swiglu_grad(a, b, grad, sigmoid, one_minus_s, da, db):
    """SwiGLU backward: a straight element-wise graph with two outputs.

    AscendC: SwiGluGradBaseKernel<T>::Compute, arch35 (3510).
    https://gitcode.com/cann/ops-nn/blob/a59460abfe0623082657ed98d976092dda0b6eca/activation/swi_glu_grad/op_kernel/arch35/swi_glu_grad_base.h

    Forward: output = SiLU(a) * b.
    Backward, element-wise:
        s = 1 / (1 + exp(-a))
        g = grad * s
        db = g * a
        da = b * (g + db * (1 - s))

    The da tile holds g as the initial accumulator for the fused multiply-add.
    The benchmark fixes a, b and grad to separate [16, 128] FP32 tiles.
    For x=[16, 256] split on the last dimension, the upstream operator dispatches
    ComputeInterleaved, which reads a and b from adjacent halves of each row.
    This benchmark targets Compute with separate buffers and dataCount=2048.
    """
    pl.neg(sigmoid, a)
    pl.exp(sigmoid, sigmoid)
    pl.add(sigmoid, sigmoid, 1.0)
    pl.recip(sigmoid, sigmoid)

    pl.mul(da, grad, sigmoid)
    pl.mul(db, da, a)
    pl.expands(one_minus_s, 1.0)
    pl.sub(one_minus_s, one_minus_s, sigmoid)
    pl.mul_add_dst(da, db, one_minus_s)
    pl.mul(da, da, b)


@pl.jit(auto_mutex=True)
def swiglu_grad_kernel(
    a: pl.Tensor[[16, 128], pl.DT_FP32],
    b: pl.Tensor[[16, 128], pl.DT_FP32],
    grad: pl.Tensor[[16, 128], pl.DT_FP32],
    da_out: pl.Tensor[[16, 128], pl.DT_FP32, pl.Output],
    db_out: pl.Tensor[[16, 128], pl.DT_FP32, pl.Output],
    elapsed: pl.Tensor[[1], pl.DT_INT64, pl.Output],
):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[16, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        # Seven 8192-byte slots occupy 56 KiB.
        a_tile = _make_workload_tile(tile_type, 0x00000)
        b_tile = _make_workload_tile(tile_type, 0x02000)
        grad_tile = _make_workload_tile(tile_type, 0x04000)
        da = _make_workload_tile(tile_type, 0x06000)
        db = _make_workload_tile(tile_type, 0x08000)
        sigmoid = _make_workload_tile(tile_type, 0x0A000)
        one_minus_s = _make_workload_tile(tile_type, 0x0C000)

        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.load(grad_tile, grad, [0, 0])
        start = _synced_cycle()
        swiglu_grad(a_tile, b_tile, grad_tile, sigmoid, one_minus_s, da, db)
        elapsed[0] = _synced_cycle() - start
        pl.store(da_out, da, [0, 0])
        pl.store(db_out, db, [0, 0])


@pl.vector_function
def swiglu_grad_reg(a, b, grad, da, db):
    """Reuse db = g * a in da = b * (g + db * (1 - s)), with g = grad * s."""
    mask = vf.update_mask(16 * 128, dtype=pl.DT_FP32)
    one = vf.full(1.0, mask, dtype=pl.DT_FP32)
    for i in pl.range(0, 32):
        offset = i * 64
        x = vf.load_align(a, offset)
        y = vf.load_align(b, offset)
        dy = vf.load_align(grad, offset)
        denominator = vf.muls(x, -1.0, mask)
        denominator = vf.exp(denominator, mask)
        denominator = vf.adds(denominator, 1.0, mask)
        s = vf.div(one, denominator, mask)

        g = vf.mul(dy, s, mask)
        db_reg = vf.mul(g, x, mask)
        one_minus_s = vf.sub(one, s, mask)
        da_reg = vf.mul_add_dst_acc(g, db_reg, one_minus_s, mask)
        da_reg = vf.mul(y, da_reg, mask)
        vf.store_align(da, da_reg, mask, offset)
        vf.store_align(db, db_reg, mask, offset)


@pl.jit(auto_mutex=True)
def swiglu_grad_reg_kernel(
    a: pl.Tensor[[16, 128], pl.DT_FP32],
    b: pl.Tensor[[16, 128], pl.DT_FP32],
    grad: pl.Tensor[[16, 128], pl.DT_FP32],
    da_out: pl.Tensor[[16, 128], pl.DT_FP32, pl.Output],
    db_out: pl.Tensor[[16, 128], pl.DT_FP32, pl.Output],
    elapsed: pl.Tensor[[1], pl.DT_INT64, pl.Output],
):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[16, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        a_tile = _make_workload_tile(tile_type, 0x00000)
        b_tile = _make_workload_tile(tile_type, 0x02000)
        grad_tile = _make_workload_tile(tile_type, 0x04000)
        da = _make_workload_tile(tile_type, 0x06000)
        db = _make_workload_tile(tile_type, 0x08000)
        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.load(grad_tile, grad, [0, 0])
        start = _synced_cycle()
        swiglu_grad_reg(a_tile, b_tile, grad_tile, da, db)
        elapsed[0] = _synced_cycle() - start
        pl.store(da_out, da, [0, 0])
        pl.store(db_out, db, [0, 0])


def fa_online_softmax(
    qk,
    prev_max,
    prev_sum,
    scale,
    exp_tmp,
    reduce_tmp,
    block_max,
    block_sum,
    sum_tmp,
    p,
    new_max,
    new_sum,
    rescale,
):
    """Online softmax update for one subsequent KV block in DN layout.

    AscendC: ProcessVec1DnUpdateVF, arch35 (3510), FP32 scores / FP16 output,
    hasAtten=false, hasSink=false, pScale=1.
    https://gitcode.com/cann/ops-transformer/blob/ca5169fbc95c4e0044af1ef987b841a78455f758/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h

    Formula for qk[K, Q], reducing over K; scale must be positive:
        scaled_scores = scale * qk
        new_max = maximum(prev_max, max_K(scaled_scores))
        rescale = exp(prev_max - new_max)
        exp_tmp = exp(scaled_scores - new_max)
        new_sum = prev_sum * rescale + sum_K(exp_tmp)
        p = cast_fp16(exp_tmp)

    prev_max/new_max include scale. prev_sum is the previous exponential sum;
    sum_tmp = prev_sum * rescale converts it to the new_max reference.
    qk is [128, 64]; DSL dim=1 reduces its rows to [1, 64].
    p is unnormalized and stored in ND layout. This workload extracts
    the math region: the reference's NZ packing/DeInterleave is excluded.
    A baseline must match that layout boundary and use separate old/new
    state buffers, since different tile arguments here never alias.
    Like fa/test_fa_perf_tkv_preload_dn_tile.py::softmax_body, exp_tmp is updated
    in place from scaled scores to exponential weights. reduce_tmp is reduction
    scratch. expand_expdif combines subtraction and exp; this region uses 9 ops.
    """
    pl.mul(exp_tmp, qk, scale)
    pl.maximum(block_max, exp_tmp, reduce_tmp, dim=1)
    pl.maximum(new_max, prev_max, block_max)
    pl.expand_expdif(rescale, prev_max, new_max, dim=1)
    pl.expand_expdif(exp_tmp, exp_tmp, new_max, dim=1)
    pl.sum(block_sum, exp_tmp, reduce_tmp, dim=1)
    pl.mul(sum_tmp, prev_sum, rescale)
    pl.add(new_sum, sum_tmp, block_sum)
    pl.cast(p, exp_tmp, mode=pl.RoundMode.CAST_ROUND)


@pl.jit(auto_mutex=True)
def fa_online_softmax_kernel(
    qk: pl.Tensor[[128, 64], pl.DT_FP32],
    prev_max: pl.Tensor[[1, 64], pl.DT_FP32],
    prev_sum: pl.Tensor[[1, 64], pl.DT_FP32],
    scale: pl.DT_FP32,
    p_out: pl.Tensor[[128, 64], pl.DT_FP16, pl.Output],
    stats_out: pl.Tensor[[3, 64], pl.DT_FP32, pl.Output],
    elapsed: pl.Tensor[[1], pl.DT_INT64, pl.Output],
):
    with pl.section_vector():
        matrix_type = pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        row_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        output_type = pl.TileType(shape=[128, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        # Three FP32 matrices, one FP16 matrix, eight FP32 rows: 114 KiB.
        qk_tile = _make_workload_tile(matrix_type, 0x00000)
        exp_tmp = _make_workload_tile(matrix_type, 0x08000)
        reduce_tmp = _make_workload_tile(matrix_type, 0x10000)
        p = _make_workload_tile(output_type, 0x18000)
        prev_max_tile = _make_workload_tile(row_type, 0x1C000)
        prev_sum_tile = _make_workload_tile(row_type, 0x1C100)
        new_max = _make_workload_tile(row_type, 0x1C200)
        new_sum = _make_workload_tile(row_type, 0x1C300)
        rescale = _make_workload_tile(row_type, 0x1C400)
        block_max = _make_workload_tile(row_type, 0x1C500)
        block_sum = _make_workload_tile(row_type, 0x1C600)
        sum_tmp = _make_workload_tile(row_type, 0x1C700)

        pl.load(qk_tile, qk, [0, 0])
        pl.load(prev_max_tile, prev_max, [0, 0])
        pl.load(prev_sum_tile, prev_sum, [0, 0])
        start = _synced_cycle()
        fa_online_softmax(
            qk_tile,
            prev_max_tile,
            prev_sum_tile,
            scale,
            exp_tmp,
            reduce_tmp,
            block_max,
            block_sum,
            sum_tmp,
            p,
            new_max,
            new_sum,
            rescale,
        )
        elapsed[0] = _synced_cycle() - start
        pl.store(p_out, p, [0, 0])
        pl.store(stats_out, new_max, [0, 0])
        pl.store(stats_out, new_sum, [1, 0])
        pl.store(stats_out, rescale, [2, 0])


@pl.vector_function
def fa_online_softmax_reg(qk, prev_max, prev_sum, scale, p, new_max, new_sum, rescale):
    """One FP32 register holds all 64 queries for a key. Reduce over pairs of keys."""
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    half_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    even_max = vf.load_align(qk, 0)
    odd_max = vf.load_align(qk, 64)
    for pair in pl.range(1, 64):
        offset = pair * 128
        even = vf.load_align(qk, offset)
        odd = vf.load_align(qk, offset + 64)
        even_max = vf.max(even_max, even, mask)
        odd_max = vf.max(odd_max, odd, mask)

    block_max = vf.max(even_max, odd_max, mask)
    block_max = vf.muls(block_max, scale, mask)
    old_max = vf.load_align(prev_max, 0)
    maximum = vf.max(block_max, old_max, mask)
    correction = vf.exp_sub(old_max, maximum, mask)
    vf.store_align(new_max, maximum, mask)
    vf.store_align(rescale, correction, mask)

    even_sum = vf.full(0.0, mask)
    odd_sum = vf.full(0.0, mask)
    for pair in pl.range(0, 64):
        offset = pair * 128
        even = vf.load_align(qk, offset)
        odd = vf.load_align(qk, offset + 64)
        even = vf.muls(even, scale, mask)
        odd = vf.muls(odd, scale, mask)
        even = vf.exp_sub(even, maximum, mask)
        odd = vf.exp_sub(odd, maximum, mask)
        even_sum = vf.add(even_sum, even, mask)
        odd_sum = vf.add(odd_sum, odd, mask)
        # Each cast fills the even FP16 lanes. Pack both rows in ND order.
        even_half = vf.astype(even, mask, dtype=pl.DT_FP16, round_mode=pl.VFRoundMode.CAST_ROUND)
        odd_half = vf.astype(odd, mask, dtype=pl.DT_FP16, round_mode=pl.VFRoundMode.CAST_ROUND)
        weights, _ = vf.de_interleave(even_half, odd_half)
        vf.store_align(p, weights, half_mask, offset)

    block_sum = vf.add(even_sum, odd_sum, mask)
    old_sum = vf.load_align(prev_sum, 0)
    scaled_sum = vf.mul(old_sum, correction, mask)
    total = vf.add(block_sum, scaled_sum, mask)
    vf.store_align(new_sum, total, mask)


@pl.jit(auto_mutex=True)
def fa_online_softmax_reg_kernel(
    qk: pl.Tensor[[128, 64], pl.DT_FP32],
    prev_max: pl.Tensor[[1, 64], pl.DT_FP32],
    prev_sum: pl.Tensor[[1, 64], pl.DT_FP32],
    scale: pl.DT_FP32,
    p_out: pl.Tensor[[128, 64], pl.DT_FP16, pl.Output],
    stats_out: pl.Tensor[[3, 64], pl.DT_FP32, pl.Output],
    elapsed: pl.Tensor[[1], pl.DT_INT64, pl.Output],
):
    with pl.section_vector():
        matrix_type = pl.TileType(shape=[128, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        row_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        output_type = pl.TileType(shape=[128, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        qk_tile = _make_workload_tile(matrix_type, 0x00000)
        p = _make_workload_tile(output_type, 0x18000)
        prev_max_tile = _make_workload_tile(row_type, 0x1C000)
        prev_sum_tile = _make_workload_tile(row_type, 0x1C100)
        new_max = _make_workload_tile(row_type, 0x1C200)
        new_sum = _make_workload_tile(row_type, 0x1C300)
        rescale = _make_workload_tile(row_type, 0x1C400)

        pl.load(qk_tile, qk, [0, 0])
        pl.load(prev_max_tile, prev_max, [0, 0])
        pl.load(prev_sum_tile, prev_sum, [0, 0])
        start = _synced_cycle()
        fa_online_softmax_reg(qk_tile, prev_max_tile, prev_sum_tile, scale, p, new_max, new_sum, rescale)
        elapsed[0] = _synced_cycle() - start
        pl.store(p_out, p, [0, 0])
        pl.store(stats_out, new_max, [0, 0])
        pl.store(stats_out, new_sum, [1, 0])
        pl.store(stats_out, rescale, [2, 0])


def mhc_pre_sinkhorn(
    x0,
    x1,
    x2,
    x3,
    bias0,
    bias1,
    bias2,
    bias3,
    scale,
    eps,
    n_iter,
    w0,
    w1,
    w2,
    w3,
    reduce_tmp,
    row_max,
    row_sum,
    col_tmp,
    col_sum,
    y0,
    y1,
    y2,
    y3,
):
    """The batched 4x4 Sinkhorn region of MHC PreSinkhorn.

    AscendC: VFProcessCombFragCompact, arch35 (3510).
    https://gitcode.com/cann/ops-transformer/blob/ca5169fbc95c4e0044af1ef987b841a78455f758/mhc/mhc_pre_sinkhorn/op_kernel/arch35/mhc_pre_sinkhorn_base_arch35.h

    For each 4x4 matrix, i is its row and j its column:
        scores[i,j] = scale * x[i,j] + bias[i,j]
        w[i,j] = exp(scores[i,j] - max_j(scores[i,j]))
        w[i,j] = w[i,j] / sum_j(w[i,j]) + eps
        repeat n_iter times:
            w[i,j] = w[i,j] / (sum_i(w[i,j]) + eps)
            w[i,j] = w[i,j] / (sum_j(w[i,j]) + eps)
        y[i,j] = w[i,j] / (sum_i(w[i,j]) + eps)

    n_iter is AscendC's iters (iterTimes - 1 in the full operator).
    Zero still runs softmax and one column normalization.

    There are 64 independent 4x4 matrices. Each x<i>[4,64] stores logical row i
    with indices [column, batch]; w<i> and y<i> use the same layout.
    Bias is shared across batches and broadcast into bias0..bias3 before launch.
    Thus DSL dim=1 reduces a logical matrix ROW; adding four row tiles
    reduces logical matrix COLUMNS without mixing independent batches.
    The baseline must use this UB layout, or account for the reference's
    compact Gather/Scatter separately. This is the Sinkhorn math region,
    excluding the enclosing operator's matmul/RMSNorm and other outputs.

    w0..w3 hold intermediate values. row_max/row_sum are row statistics;
    col_tmp sums rows 2 and 3, and col_sum is the column denominator.
    reduce_tmp is reduction scratch. The final normalization writes y0..y3.
    Softmax and final normalization use 36 tile ops; each runtime iteration
    uses 20. Differently named tiles have disjoint addresses.
    """
    pl.mul(w0, x0, scale)
    pl.add(w0, w0, bias0)
    pl.maximum(row_max, w0, reduce_tmp, dim=1)
    pl.expand_expdif(w0, w0, row_max, dim=1)
    pl.sum(row_sum, w0, reduce_tmp, dim=1)
    pl.expand_div(w0, w0, row_sum, dim=1)
    pl.add(w0, w0, eps)

    pl.mul(w1, x1, scale)
    pl.add(w1, w1, bias1)
    pl.maximum(row_max, w1, reduce_tmp, dim=1)
    pl.expand_expdif(w1, w1, row_max, dim=1)
    pl.sum(row_sum, w1, reduce_tmp, dim=1)
    pl.expand_div(w1, w1, row_sum, dim=1)
    pl.add(w1, w1, eps)

    pl.mul(w2, x2, scale)
    pl.add(w2, w2, bias2)
    pl.maximum(row_max, w2, reduce_tmp, dim=1)
    pl.expand_expdif(w2, w2, row_max, dim=1)
    pl.sum(row_sum, w2, reduce_tmp, dim=1)
    pl.expand_div(w2, w2, row_sum, dim=1)
    pl.add(w2, w2, eps)

    pl.mul(w3, x3, scale)
    pl.add(w3, w3, bias3)
    pl.maximum(row_max, w3, reduce_tmp, dim=1)
    pl.expand_expdif(w3, w3, row_max, dim=1)
    pl.sum(row_sum, w3, reduce_tmp, dim=1)
    pl.expand_div(w3, w3, row_sum, dim=1)
    pl.add(w3, w3, eps)

    for _ in pl.range(0, n_iter):
        pl.add(col_sum, w0, w1)
        pl.add(col_tmp, w2, w3)
        pl.add(col_sum, col_sum, col_tmp)
        pl.add(col_sum, col_sum, eps)
        pl.div(w0, w0, col_sum)
        pl.div(w1, w1, col_sum)
        pl.div(w2, w2, col_sum)
        pl.div(w3, w3, col_sum)

        pl.sum(row_sum, w0, reduce_tmp, dim=1)
        pl.add(row_sum, row_sum, eps)
        pl.expand_div(w0, w0, row_sum, dim=1)
        pl.sum(row_sum, w1, reduce_tmp, dim=1)
        pl.add(row_sum, row_sum, eps)
        pl.expand_div(w1, w1, row_sum, dim=1)
        pl.sum(row_sum, w2, reduce_tmp, dim=1)
        pl.add(row_sum, row_sum, eps)
        pl.expand_div(w2, w2, row_sum, dim=1)
        pl.sum(row_sum, w3, reduce_tmp, dim=1)
        pl.add(row_sum, row_sum, eps)
        pl.expand_div(w3, w3, row_sum, dim=1)

    pl.add(col_sum, w0, w1)
    pl.add(col_tmp, w2, w3)
    pl.add(col_sum, col_sum, col_tmp)
    pl.add(col_sum, col_sum, eps)
    pl.div(y0, w0, col_sum)
    pl.div(y1, w1, col_sum)
    pl.div(y2, w2, col_sum)
    pl.div(y3, w3, col_sum)


@pl.jit(auto_mutex=True)
def mhc_pre_sinkhorn_kernel(
    x: pl.Tensor[[16, 64], pl.DT_FP32],
    bias: pl.Tensor[[16, 64], pl.DT_FP32],
    scale: pl.DT_FP32,
    eps: pl.DT_FP32,
    n_iter: pl.DT_INT64,
    y_out: pl.Tensor[[16, 64], pl.DT_FP32, pl.Output],
    elapsed: pl.Tensor[[1], pl.DT_INT64, pl.Output],
):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[4, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        row_type = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        # 19 matrix slots (1024 bytes) and two row slots (256 bytes): 19.5 KiB.
        x0 = _make_workload_tile(tile_type, 0x0000)
        x1 = _make_workload_tile(tile_type, 0x0400)
        x2 = _make_workload_tile(tile_type, 0x0800)
        x3 = _make_workload_tile(tile_type, 0x0C00)
        bias0 = _make_workload_tile(tile_type, 0x1000)
        bias1 = _make_workload_tile(tile_type, 0x1400)
        bias2 = _make_workload_tile(tile_type, 0x1800)
        bias3 = _make_workload_tile(tile_type, 0x1C00)
        y0 = _make_workload_tile(tile_type, 0x2000)
        y1 = _make_workload_tile(tile_type, 0x2400)
        y2 = _make_workload_tile(tile_type, 0x2800)
        y3 = _make_workload_tile(tile_type, 0x2C00)
        w0 = _make_workload_tile(tile_type, 0x3000)
        w1 = _make_workload_tile(tile_type, 0x3400)
        w2 = _make_workload_tile(tile_type, 0x3800)
        w3 = _make_workload_tile(tile_type, 0x3C00)
        reduce_tmp = _make_workload_tile(tile_type, 0x4000)
        col_tmp = _make_workload_tile(tile_type, 0x4400)
        col_sum = _make_workload_tile(tile_type, 0x4800)
        row_max = _make_workload_tile(row_type, 0x4C00)
        row_sum = _make_workload_tile(row_type, 0x4D00)

        pl.load(x0, x, [0, 0])
        pl.load(x1, x, [4, 0])
        pl.load(x2, x, [8, 0])
        pl.load(x3, x, [12, 0])
        pl.load(bias0, bias, [0, 0])
        pl.load(bias1, bias, [4, 0])
        pl.load(bias2, bias, [8, 0])
        pl.load(bias3, bias, [12, 0])
        start = _synced_cycle()
        mhc_pre_sinkhorn(
            x0,
            x1,
            x2,
            x3,
            bias0,
            bias1,
            bias2,
            bias3,
            scale,
            eps,
            n_iter,
            w0,
            w1,
            w2,
            w3,
            reduce_tmp,
            row_max,
            row_sum,
            col_tmp,
            col_sum,
            y0,
            y1,
            y2,
            y3,
        )
        elapsed[0] = _synced_cycle() - start
        pl.store(y_out, y0, [0, 0])
        pl.store(y_out, y1, [4, 0])
        pl.store(y_out, y2, [8, 0])
        pl.store(y_out, y3, [12, 0])


@pytest.fixture
def measure_vf(request):
    def measure(kernel, *args):
        launch = kernel[1]
        elapsed = torch.zeros(1, dtype=torch.int64, device="npu")
        for _ in range(10):
            launch(*args, elapsed)
        torch.npu.synchronize()

        samples = []
        for _ in range(101):
            launch(*args, elapsed)
            samples.append(elapsed.item())
        assert min(samples) > 0, f"Invalid device counter deltas: {samples}"
        ticks = median(samples)
        # Ascend 950 system counter frequency is 1 GHz.
        print(f"{request.node.name}  {ticks:g} cycles ({ticks / 1000:.4f} us), median of {len(samples)} launches")

    return measure


# Each gradient covers one FP32 [16, 128] block. Median times on Ascend 950
# exclude GM transfers. The PyPTO run used 101 samples after 10 warmups.
# The separate AscendC run used 200 samples, discarding 10 warmups per group.
# Speedup is the AscendC Compute median divided by each implementation's median.
#
# Implementation             | Median (cycles) | Median (us) | Speedup vs AscendC
# -------------------------- | --------------- | ----------- | ------------------
# AscendC Compute (baseline) |             305 |       0.305 |             1.000x
# PyPTO tile                 |             272 |       0.272 |             1.121x
# PyPTO reg                  |             250 |       0.250 |             1.220x
# AscendC ComputeInterleaved |             306 |       0.306 |             0.997x
#
# Tile, reg and Compute use separate a/b buffers. ComputeInterleaved reads
# a and b from the two halves of each row of the packed [16, 256] input.
@pytest.mark.soc("950")
@pytest.mark.parametrize("kernel", [swiglu_grad_kernel, swiglu_grad_reg_kernel], ids=["tile", "reg"])
def test_swiglu_grad_launch(kernel, measure_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    generator = torch.Generator().manual_seed(23)
    a, b, grad = (torch.randn(16, 128, generator=generator) for _ in range(3))
    a.requires_grad_()
    b.requires_grad_()
    expected_da, expected_db = torch.autograd.grad(torch.nn.functional.silu(a) * b, (a, b), grad)
    da = torch.full_like(a, float("nan"), device="npu")
    db = torch.full_like(b, float("nan"), device="npu")
    measure_vf(kernel, a.detach().npu(), b.detach().npu(), grad.npu(), da, db)
    torch.testing.assert_close(da.cpu(), expected_da, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(db.cpu(), expected_db, rtol=1e-4, atol=1e-5)


# One subsequent KV block has 128 keys x 64 queries (DN), FP32 scores and
# [1, 64] FP32 running statistics. The score scale is 1/sqrt(128).
# All three ran on Ascend950DT_9592 with the same input and one block per launch.
# Medians cover 101 launches after 10 warmups, with GM transfers outside the timed interval.
#
# Implementation     | Median (cycles) | Median (us) | Speedup vs AscendC
# ------------------ | --------------- | ----------- | ------------------
# AscendC (baseline) |             397 |       0.397 |             1.000x
# PyPTO tile         |            1420 |       1.420 |             0.280x
# PyPTO reg          |             416 |       0.416 |             0.954x
#
# Tile and reg write FP16 weights in ND and use separate max/sum outputs.
# AscendC ProcessVec1DnUpdateVF packs the weights into NZ and updates max/sum in place.
@pytest.mark.soc("950")
@pytest.mark.parametrize("kernel", [fa_online_softmax_kernel, fa_online_softmax_reg_kernel], ids=["tile", "reg"])
def test_fa_online_softmax_launch(kernel, measure_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    generator = torch.Generator().manual_seed(83)
    scale = 128**-0.5
    prev_scores = torch.randn(128, 64, dtype=torch.float32, generator=generator) * scale
    prev_max = prev_scores.amax(dim=0, keepdim=True)
    prev_sum = (prev_scores - prev_max).exp().sum(dim=0, keepdim=True)
    qk = torch.randn(128, 64, dtype=torch.float32, generator=generator)
    scores = qk * scale
    new_max = torch.maximum(prev_max, scores.amax(dim=0, keepdim=True))
    rescale = (prev_max - new_max).exp()
    weights = (scores - new_max).exp()
    new_sum = prev_sum * rescale + weights.sum(dim=0, keepdim=True)

    p = torch.full_like(qk, float("nan"), dtype=torch.float16, device="npu")
    stats = torch.full((3, 64), float("nan"), dtype=torch.float32, device="npu")
    measure_vf(kernel, qk.npu(), prev_max.npu(), prev_sum.npu(), scale, p, stats)
    torch.testing.assert_close(p.cpu(), weights.half(), rtol=1e-3, atol=0)
    torch.testing.assert_close(stats.cpu(), torch.cat((new_max, new_sum, rescale)), rtol=1e-5, atol=1e-6)


# One block holds 64 independent FP32 4x4 matrices, split into four [4, 64]
# row tiles. n_iter counts Sinkhorn iterations after the initial row softmax.
# Ascend 950 medians cover 101 launches after 10 warmups and exclude GM transfers.
# N/A indicates a missing AscendC measurement for that iteration count.
#
# Implementation | n_iter | Median (cycles) | Median (us) | Speedup vs AscendC
# -------------- | ------ | --------------- | ----------- | ------------------
# PyPTO tile     |      0 |             595 |       0.595 |                N/A
# PyPTO tile     |      1 |             991 |       0.991 |                N/A
# PyPTO tile     |      8 |            3762 |       3.762 |                N/A
# PyPTO tile     |     24 |           10096 |      10.096 |                N/A
@pytest.mark.soc("950")
@pytest.mark.parametrize("n_iter", [0, 1, 8, 24])
def test_mhc_pre_sinkhorn_launch(n_iter, measure_vf):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    # GM layout [logical row, logical column, batch], flattened to [16, 64].
    x = torch.randn(16, 64, dtype=torch.float32, device="npu")
    bias_matrix = torch.randn(4, 4, 1, dtype=torch.float32, device="npu")
    bias = bias_matrix.expand(4, 4, 64).contiguous().view(16, 64)
    y = torch.empty_like(x)
    measure_vf(mhc_pre_sinkhorn_kernel, x, bias, 0.5, 1e-6, n_iter, y)
