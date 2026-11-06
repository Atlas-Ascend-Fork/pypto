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
The current frontend inlines these helpers and emits the existing tile ops;
the helpers do not declare handwritten register-level vector functions.

Each kernel constructs disjoint UB tiles, loads its inputs, and calls its helper
once. Results stay in UB. The tests only check compilation and a single launch;
measurement belongs to the external performance tooling. The helper docstrings
specify the mathematical region and layout to match when extracting a baseline.
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu


def _make_workload_tile(tile_type, addr):
    group = pl.make_tile_group(type=tile_type, addrs=addr, mutex_ids="auto", depth=1)
    return group.current()


def swiglu_grad(a, b, grad, sigmoid, grad_s, grad_tmp, da, db):
    """SwiGLU backward: a straight element-wise graph with two outputs.

    AscendC: SwiGluGradBaseKernel<T>::Compute, arch35 (3510).
    https://gitcode.com/cann/ops-nn/blob/a59460abfe0623082657ed98d976092dda0b6eca/activation/swi_glu_grad/op_kernel/arch35/swi_glu_grad_base.h

    Forward: output = SiLU(a) * b.
    Backward, element-wise:
        s = 1 / (1 + exp(-a))
        g = grad * s
        db = g * a
        da = b * (g - db * (s - 1))

    sigmoid stores s; grad_s stores g; grad_tmp computes g - db * (s - 1).
    This region uses 10 tile ops.
    All tiles are [16, 128] FP32. Inputs and their gradients use separate buffers;
    splitting/concatenating the full operator's tensors is outside this region.
    """
    pl.neg(sigmoid, a)
    pl.exp(sigmoid, sigmoid)
    pl.add(sigmoid, sigmoid, 1.0)
    pl.recip(sigmoid, sigmoid)

    pl.mul(grad_s, grad, sigmoid)
    pl.mul(db, grad_s, a)
    # grad_tmp = grad_s - db * (sigmoid - 1).
    pl.sub(grad_tmp, sigmoid, 1.0)
    pl.mul(grad_tmp, grad_tmp, db)
    pl.sub(grad_tmp, grad_s, grad_tmp)
    pl.mul(da, grad_tmp, b)


@pl.jit(auto_mutex=True)
def swiglu_grad_kernel(
    a: pl.Tensor[[16, 128], pl.DT_FP32],
    b: pl.Tensor[[16, 128], pl.DT_FP32],
    grad: pl.Tensor[[16, 128], pl.DT_FP32],
):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[16, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        # Eight 8192-byte slots: [0, 0x10000), 64 KiB in total.
        a_tile = _make_workload_tile(tile_type, 0x00000)
        b_tile = _make_workload_tile(tile_type, 0x02000)
        grad_tile = _make_workload_tile(tile_type, 0x04000)
        da = _make_workload_tile(tile_type, 0x06000)
        db = _make_workload_tile(tile_type, 0x08000)
        sigmoid = _make_workload_tile(tile_type, 0x0A000)
        grad_s = _make_workload_tile(tile_type, 0x0C000)
        grad_tmp = _make_workload_tile(tile_type, 0x0E000)

        pl.load(a_tile, a, [0, 0])
        pl.load(b_tile, b, [0, 0])
        pl.load(grad_tile, grad, [0, 0])
        swiglu_grad(a_tile, b_tile, grad_tile, sigmoid, grad_s, grad_tmp, da, db)


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


@pytest.mark.soc("950")
def test_swiglu_grad_launch():
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    a = torch.randn(16, 128, dtype=torch.float32, device="npu")
    b = torch.randn_like(a)
    grad = torch.randn_like(a)
    swiglu_grad_kernel[1](a, b, grad)
    torch.npu.synchronize()


@pytest.mark.soc("950")
def test_fa_online_softmax_launch():
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    scale = 128**-0.5
    prev_qk = torch.randn(128, 64, dtype=torch.float32, device="npu")
    prev_scores = prev_qk * scale
    prev_max = prev_scores.amax(dim=0, keepdim=True)
    prev_sum = (prev_scores - prev_max).exp().sum(dim=0, keepdim=True)
    qk = torch.randn_like(prev_qk)
    fa_online_softmax_kernel[1](qk, prev_max, prev_sum, scale)
    torch.npu.synchronize()


@pytest.mark.soc("950")
@pytest.mark.parametrize("n_iter", [0, 1, 8, 24])
def test_mhc_pre_sinkhorn_launch(n_iter):
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    # GM layout [logical row, logical column, batch], flattened to [16, 64].
    x = torch.randn(16, 64, dtype=torch.float32, device="npu")
    bias_matrix = torch.randn(4, 4, 1, dtype=torch.float32, device="npu")
    bias = bias_matrix.expand(4, 4, 64).contiguous().view(16, 64)
    mhc_pre_sinkhorn_kernel[1](x, bias, 0.5, 1e-6, n_iter)
    torch.npu.synchronize()
