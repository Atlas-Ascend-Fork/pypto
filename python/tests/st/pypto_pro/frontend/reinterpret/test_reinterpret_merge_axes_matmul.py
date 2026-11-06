#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Full tiled matmul over a grouped Tensor, fed by the merged-row form of pl.load.

    x   [B, S, N, G, D]        grouped activations
    w   [D, NCOL]              weights
    out [B, N, S*G, NCOL]      out[b, h] = x[b, :, h, :, :].reshape(S*G, D) @ w

The matmul wants (S, G) flattened into its M axis. Those two axes are not adjacent in x -- N sits
between them -- so consecutive S rows are N*G*D apart while consecutive G rows are D apart. A
plain load can walk only one of those strides, which is what makes this shape awkward.

``order=[1, 3, 4]`` makes the destination Tile's row index the flattening of tensor axes 1 and 3:

    tile row r  <->  x[b, s0 + r // G, h, r % G, k0 ...]

so one instruction fills [M_TILE, K_TILE] in L1 with the (s, g) rows already in order. It lowers
to PTO TLOAD's multi-ND2NZ path, which also handles partial final matrices.

``order=[4, 3, 1]``, the exact reverse, loads the same rows as the transposed operand,
[K_TILE, M_TILE] into a ZN Tile. It is the very same transfer: the NZ and ZN tiles at one
address are the same bytes, so only the destination's label differs, and the Mat->L0 move turns
that label into the transpose. A matmul that contracts over the merged axis takes its left
operand this way; the alternative -- an ascending load plus a pl.reinterpret view of the buffer
it filled -- is the same trick spelled by hand, and the last two kernels here use it.

Every dimension here is pl.DYNAMIC, including G. That is the interesting case for this op:

  - B, S, N, D  become runtime strides in the DMA descriptor, as for any load.
  - G is different. It sets the rows per ND matrix. Lowering bounds it to a supported
    matrix extent, and TLOAD transfers the valid window, including a partial final matrix.
    The full-matmul kernels below derive their own S step as ``M_TILE // x.shape[3]``;
    they require G > 0 and M_TILE % G == 0. Floor division alone does not ensure this:
    G=3 gives a step of 42 for M_TILE=128 and overlapping 128-row windows. The test
    runners reject non-divisors before launching these full-matmul kernels. Separate
    single-load regressions cover G=3 remainder transfers and large dynamic inner extents.

Tiling: M_TILE rows of the merged axis, K_TILE down D with accumulation, N_TILE across NCOL.
No tail handling -- the tests choose S, D and NCOL as whole multiples of the tile sizes.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

DYN = pl.DYNAMIC

M_TILE = 128  # merged (s, g) rows per matmul
K_TILE = 64   # step down D
N_TILE = 64   # step across NCOL
MROW_TILE = 64  # rows of the plain left operand, for the merged-as-right-matrix case

# L1 / L0 addresses. Every group is depth 2 so the k-loop pipelines.
L1_X, L1_W = 0x00000, 0x10000
L0A, L0B, L0C = 0x0, 0x0, 0x0


def _require_3510(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("not 3510")


@pl.jit(auto_mutex=True)
def grouped_matmul_fp16(
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
    w: pl.Tensor[[DYN, DYN], pl.DT_FP16],
    out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
):
    """out[b, h, s*G + g, :] = sum_d x[b, s, h, g, d] * w[d, :], every dim dynamic.

    Requires G > 0 and M_TILE % G == 0; S, D and output columns must tile without tails.
    """
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0, 1])
    w_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_W, mutex_ids=[2, 3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, N_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

    with pl.section_cube():
        group = x.shape[3]
        s_step = M_TILE // group  # Requires M_TILE % group == 0; see the kernel precondition.
        for b in pl.range(0, x.shape[0]):
            for h in pl.range(0, x.shape[2]):
                for s0 in pl.range(0, x.shape[1], s_step):
                    for n0 in pl.range(0, w.shape[1], N_TILE):
                        ac = acc.next()
                        for k0 in pl.range(0, x.shape[4], K_TILE):
                            xl = x_l1.next()
                            wl = w_l1.next()
                            # One DMA: M_TILE (s, g) rows x K_TILE columns of D.
                            pl.load(xl, x, [b, s0, h, 0, k0], order=[1, 3, 4])
                            pl.load(wl, w, [k0, n0])
                            a_ = l0a.next()
                            b_ = l0b.next()
                            pl.move(a_, xl)
                            pl.move(b_, wl)
                            if k0 == 0:
                                pl.matmul(ac, a_, b_)
                            else:
                                pl.matmul_acc(ac, ac, a_, b_)
                        pl.store(out, ac, [b, h, s0 * group, n0], order=[2, 3])


@pl.jit(auto_mutex=True)
def grouped_matmul_bf16(
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_BF16],
    w: pl.Tensor[[DYN, DYN], pl.DT_BF16],
    out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
):
    """Same kernel in BF16 -- same C0 (16) as FP16, so the same dst strides.

    Requires G > 0 and M_TILE % G == 0; S, D and output columns must tile without tails.
    """
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0, 1])
    w_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_W, mutex_ids=[2, 3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, N_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

    with pl.section_cube():
        group = x.shape[3]
        s_step = M_TILE // group
        for b in pl.range(0, x.shape[0]):
            for h in pl.range(0, x.shape[2]):
                for s0 in pl.range(0, x.shape[1], s_step):
                    for n0 in pl.range(0, w.shape[1], N_TILE):
                        ac = acc.next()
                        for k0 in pl.range(0, x.shape[4], K_TILE):
                            xl = x_l1.next()
                            wl = w_l1.next()
                            pl.load(xl, x, [b, s0, h, 0, k0], order=[1, 3, 4])
                            pl.load(wl, w, [k0, n0])
                            a_ = l0a.next()
                            b_ = l0b.next()
                            pl.move(a_, xl)
                            pl.move(b_, wl)
                            if k0 == 0:
                                pl.matmul(ac, a_, b_)
                            else:
                                pl.matmul_acc(ac, ac, a_, b_)
                        pl.store(out, ac, [b, h, s0 * group, n0], order=[2, 3])


def _merged_s_step(g):
    if g <= 0 or M_TILE % g != 0:
        raise ValueError(f"grouped matmul requires G > 0 and G to divide M_TILE={M_TILE}, got G={g}")
    return M_TILE // g


def _make_static_g_matmul(g):
    """Same matmul with a static G > 0 that divides M_TILE."""
    s_step = _merged_s_step(g)

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[DYN, DYN, DYN, g, DYN], pl.DT_FP16],
        w: pl.Tensor[[DYN, DYN], pl.DT_FP16],
        out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
    ):
        x_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0, 1])
        w_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_W, mutex_ids=[2, 3])
        l0a = pl.make_tile_group(
            type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
        l0b = pl.make_tile_group(
            type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[M_TILE, N_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

        with pl.section_cube():
            for b in pl.range(0, x.shape[0]):
                for h in pl.range(0, x.shape[2]):
                    for s0 in pl.range(0, x.shape[1], s_step):
                        for n0 in pl.range(0, w.shape[1], N_TILE):
                            ac = acc.next()
                            for k0 in pl.range(0, x.shape[4], K_TILE):
                                xl = x_l1.next()
                                wl = w_l1.next()
                                pl.load(xl, x, [b, s0, h, 0, k0], order=[1, 3, 4])
                                pl.load(wl, w, [k0, n0])
                                a_ = l0a.next()
                                b_ = l0b.next()
                                pl.move(a_, xl)
                                pl.move(b_, wl)
                                if k0 == 0:
                                    pl.matmul(ac, a_, b_)
                                else:
                                    pl.matmul_acc(ac, ac, a_, b_)
                            pl.store(out, ac, [b, h, s0 * g, n0], order=[2, 3])

    return kernel


@pl.jit(auto_mutex=True)
def grouped_matmul_transposed_left(
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
    v: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP16],
    out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
):
    """The merged operand TRANSPOSED: out[b, h] = merged[b, h]^T @ v[b, h], i.e. [D, NCOL].

    ``order=[4, 3, 1]`` -- the exact reverse of the ascending ``[1, 3, 4]`` -- asks for the
    transpose of what that order loads: [D_TILE, merged] instead of [merged, D_TILE]. It is the
    same instruction, not a second DMA and not a UB round trip. nd2nz cannot transpose, and does
    not have to: an NZ [M, K] L1 tile and a ZN [K, M] tile at one address are the same bytes
    under the NZ/ZN offset formulas, so the reversed spelling issues the identical
    transfer and only labels the destination the other way round. Hence the ZN Tile below with
    its shape reversed -- the label IS the transpose, and the Mat->L0A move realizes it.

    That is also why the reversal is the whole-list flip rather than a per-axis permutation: it
    names the transpose of the ascending load, so the merged axis still runs (s outer, g inner)
    along it. ``[1, 4, 3]``, which merely swaps the column axis, stays rejected.

    grouped_matmul_both_merged_transposed and grouped_matmul_right_merged_transposed below reach
    the same orientation the other way, through a pl.reinterpret view of an ascending load; both
    spellings are supported and both are exercised here.

    Here the merged (s, g) axis becomes the CONTRACTION, so K-accumulation runs over s0 and D
    indexes the output rows.

    Requires G > 0 and M_TILE % G == 0; S, D and output columns must tile without tails.
    """
    # [D_TILE, merged], the transposed operand as loaded -- the reversed order's destination.
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=L1_X, mutex_ids=[0, 1])
    v_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, N_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_W, mutex_ids=[2, 3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, N_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

    with pl.section_cube():
        group = x.shape[3]
        s_step = M_TILE // group
        for b in pl.range(0, x.shape[0]):
            for h in pl.range(0, x.shape[2]):
                for d0 in pl.range(0, x.shape[4], K_TILE):
                    for n0 in pl.range(0, v.shape[3], N_TILE):
                        ac = acc.next()
                        for s0 in pl.range(0, x.shape[1], s_step):
                            xl = x_l1.next()
                            vl = v_l1.next()
                            # Reversed merged load: fills xl with [D_TILE, merged] = x^T.
                            pl.load(xl, x, [b, s0, h, 0, d0], order=[4, 3, 1])
                            pl.load(vl, v, [b, h, s0 * group, n0])
                            a_ = l0a.next()
                            b_ = l0b.next()
                            # Mat(ZN) -> Left(NZ): the move reads the label and transposes.
                            pl.move(a_, xl)
                            pl.move(b_, vl)
                            if s0 == 0:
                                pl.matmul(ac, a_, b_)
                            else:
                                pl.matmul_acc(ac, ac, a_, b_)
                        pl.store(out, ac, [b, h, d0, n0], order=[2, 3])


@pl.jit(auto_mutex=True)
def grouped_matmul_as_right(
    a: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP16],
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
    out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
):
    """The merged operand as the RIGHT matrix: out[b, h] = a[b, h] @ merged[b, h].

    Nothing in the merged load is specific to the left operand -- it fills an NZ L1 tile, and
    Mat(NZ) -> Right(ZN) is the ordinary untransposed move, the same one the weights take in the
    left-matrix kernels. The merged (s, g) axis is the contraction here, so K-accumulation runs
    over s0 and D indexes the output columns.

    The reversed ``order=[4, 3, 1]`` has no use in THIS kernel, and the reason is worth stating:
    a right operand must be [contraction, output cols], here [merged, D_TILE], and the ZN label
    would make the tile merged^T = [D_TILE, merged]. That names D as the contraction, i.e. a
    different product -- which is grouped_matmul_right_merged_transposed below, not this one.
    The cube cannot transpose L0B, so there is no spelling that keeps a @ merged with a ZN load.

    Requires G > 0 and M_TILE % G == 0; S, D and output columns must tile without tails.
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0, 1])
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_W, mutex_ids=[2, 3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, K_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

    with pl.section_cube():
        group = x.shape[3]
        s_step = M_TILE // group
        for b in pl.range(0, x.shape[0]):
            for h in pl.range(0, x.shape[2]):
                for m0 in pl.range(0, a.shape[2], MROW_TILE):
                    for d0 in pl.range(0, x.shape[4], K_TILE):
                        ac = acc.next()
                        for s0 in pl.range(0, x.shape[1], s_step):
                            al = a_l1.next()
                            xl = x_l1.next()
                            pl.load(al, a, [b, h, m0, s0 * group])
                            # Merged load feeding the RIGHT operand.
                            pl.load(xl, x, [b, s0, h, 0, d0], order=[1, 3, 4])
                            a_ = l0a.next()
                            b_ = l0b.next()
                            pl.move(a_, al)
                            pl.move(b_, xl)  # Mat(NZ) -> Right(ZN): no transpose
                            if s0 == 0:
                                pl.matmul(ac, a_, b_)
                            else:
                                pl.matmul_acc(ac, ac, a_, b_)
                        pl.store(out, ac, [b, h, m0, d0], order=[2, 3])


@pl.jit(auto_mutex=True)
def grouped_matmul_both_merged_transposed(
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
    out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
):
    """BOTH operands merged, from the same grouped Tensor, with S and D transposed.

    out[b, h] = merged[b, h]^T @ merged[b, h], a [D, D] Gram matrix.

    Two things compose here that the single-operand kernels test separately:

      * Both operands come from a merged load -- two `order=[1, 3, 4]` DMAs per step, one per
        D window, filling two independent L1 buffers.
      * The LEFT operand is transposed. The merged (s, g) axis is the contraction for both
        sides, so the left operand needs [D_TILE, merged] while the load as spelled writes
        [merged, D_TILE]. The ZN reinterpret view supplies the flip -- kept here on purpose as
        the by-hand spelling of what the reversed ``order=[4, 3, 1]`` does for the kernel
        above; the right operand takes the same buffer shape untransposed, so one merged load
        feeds each side in the orientation it needs without a second DMA or any UB traffic.

    Verified on device for both operand roles: the reinterpret view reproduces x^T exactly
    (max diff 0.0) through Mat(ZN)->Left(NZ) and through Mat(ZN)->Right(ZN).

    Accumulation runs over s0 (the contraction), and the two D windows index the output.

    Requires G > 0 and M_TILE % G == 0; S, D and output columns must tile without tails.
    """
    xa_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0, 1])
    # Transposed view of the left buffer: [D_TILE, merged] over the same bytes.
    xa_zn = pl.reinterpret(xa_l1, shape=[K_TILE, M_TILE], layout=pl.ZN)
    xb_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, N_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_W, mutex_ids=[2, 3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, N_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, N_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

    with pl.section_cube():
        group = x.shape[3]
        s_step = M_TILE // group
        for b in pl.range(0, x.shape[0]):
            for h in pl.range(0, x.shape[2]):
                for d0 in pl.range(0, x.shape[4], K_TILE):
                    for d1 in pl.range(0, x.shape[4], N_TILE):
                        ac = acc.next()
                        for s0 in pl.range(0, x.shape[1], s_step):
                            xa = xa_l1.next()
                            xb = xb_l1.next()
                            # Two merged loads: the same (s, g) rows, two different D windows.
                            pl.load(xa, x, [b, s0, h, 0, d0], order=[1, 3, 4])
                            pl.load(xb, x, [b, s0, h, 0, d1], order=[1, 3, 4])
                            a_ = l0a.next()
                            b_ = l0b.next()
                            # Left: transposed view of xa. Right: xb as loaded.
                            pl.move(a_, xa_zn.current())
                            pl.move(b_, xb)
                            if s0 == 0:
                                pl.matmul(ac, a_, b_)
                            else:
                                pl.matmul_acc(ac, ac, a_, b_)
                        pl.store(out, ac, [b, h, d0, d1], order=[2, 3])


@pl.jit(auto_mutex=True)
def grouped_matmul_right_merged_transposed(
    a: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP16],
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
    out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
):
    """The merged operand as the RIGHT matrix and TRANSPOSED: out[b, h] = a[b, h] @ merged^T.

    Distinct from grouped_matmul_as_right, where the right operand is used as loaded. Here D is
    the contraction and the merged (s, g) axis indexes the output columns, so the right operand
    must be [D_TILE, merged] -- the other orientation of the same bytes.

    The same reinterpret view supplies it, the by-hand spelling of the reversed order. The move
    is Mat(ZN) -> Right(ZN): source and destination name the same layout, unlike the
    left-operand case's Mat(ZN) -> Left(NZ) flip, and it was verified separately on device
    (max diff 0.0) before this kernel was written.

    Requires G > 0 and M_TILE % G == 0; S, D and output columns must tile without tails.
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0, 1])
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_W, mutex_ids=[2, 3])
    x_zn = pl.reinterpret(x_l1, shape=[K_TILE, M_TILE], layout=pl.ZN)
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, M_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

    with pl.section_cube():
        group = x.shape[3]
        s_step = M_TILE // group
        for b in pl.range(0, x.shape[0]):
            for h in pl.range(0, x.shape[2]):
                for m0 in pl.range(0, a.shape[2], MROW_TILE):
                    for s0 in pl.range(0, x.shape[1], s_step):
                        ac = acc.next()
                        for d0 in pl.range(0, x.shape[4], K_TILE):
                            al = a_l1.next()
                            xl = x_l1.next()
                            pl.load(al, a, [b, h, m0, d0])
                            pl.load(xl, x, [b, s0, h, 0, d0], order=[1, 3, 4])
                            a_ = l0a.next()
                            b_ = l0b.next()
                            pl.move(a_, al)
                            # Transposed view as the RIGHT operand: [D_TILE, merged].
                            pl.move(b_, x_zn.current())
                            if d0 == 0:
                                pl.matmul(ac, a_, b_)
                            else:
                                pl.matmul_acc(ac, ac, a_, b_)
                        pl.store(out, ac, [b, h, m0, s0 * group], order=[2, 3])


@pl.jit(auto_mutex=True)
def grouped_matmul_right_merged_transposed_by_order(
    a: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP16],
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
    out: pl.Tensor[[DYN, DYN, DYN, DYN], pl.DT_FP32],
):
    """The kernel above, with the transpose asked for in the load instead of by pl.reinterpret.

    Same product, out[b, h] = a[b, h] @ merged[b, h]^T, and -- the point of the test that pairs
    them -- the same instructions: ``order=[4, 3, 1]`` into a ZN [D_TILE, merged] group emits
    the identical multi-ND2NZ DMA that ``order=[1, 3, 4]`` into the NZ [merged, D_TILE] group above
    emits, because those two tiles are one buffer read under two labels. So the outputs must
    match bit for bit, not merely within tolerance.

    What it adds over grouped_matmul_transposed_left is the operand role: there a directly
    loaded ZN tile feeds Mat(ZN) -> Left(NZ), a fractal flip; here it feeds Mat(ZN) ->
    Right(ZN), where source and destination name the same layout and the move copies. Both
    reach L0 from a ZN tile the DMA wrote directly rather than one pl.reinterpret relabelled.

    Requires G > 0 and M_TILE % G == 0; S, D and output columns must tile without tails.
    """
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0, 1])
    # [D_TILE, merged] as loaded: the reversed order's destination, same bytes as the NZ
    # [merged, D_TILE] group the reinterpret-spelled kernel declares at this address.
    x_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=L1_W, mutex_ids=[2, 3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, K_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=L0A, mutex_ids=[4, 5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[K_TILE, M_TILE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=L0B, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[MROW_TILE, M_TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024), addrs=L0C, mutex_ids=[8, 9])

    with pl.section_cube():
        group = x.shape[3]
        s_step = M_TILE // group
        for b in pl.range(0, x.shape[0]):
            for h in pl.range(0, x.shape[2]):
                for m0 in pl.range(0, a.shape[2], MROW_TILE):
                    for s0 in pl.range(0, x.shape[1], s_step):
                        ac = acc.next()
                        for d0 in pl.range(0, x.shape[4], K_TILE):
                            al = a_l1.next()
                            xl = x_l1.next()
                            pl.load(al, a, [b, h, m0, d0])
                            # Reversed merged load: fills xl with [D_TILE, merged] = merged^T.
                            pl.load(xl, x, [b, s0, h, 0, d0], order=[4, 3, 1])
                            a_ = l0a.next()
                            b_ = l0b.next()
                            pl.move(a_, al)
                            # Mat(ZN) -> Right(ZN): same layout both sides, so the move copies
                            # and the label is what makes the operand the transpose.
                            pl.move(b_, xl)
                            if d0 == 0:
                                pl.matmul(ac, a_, b_)
                            else:
                                pl.matmul_acc(ac, ac, a_, b_)
                        pl.store(out, ac, [b, h, m0, s0 * group], order=[2, 3])


def _reference(x, w):
    """out[b, h] = x[b, :, h, :, :].reshape(S*G, D) @ w, in fp32."""
    b_n, s_n, n_n, g_n, d_n = x.shape
    ncol = w.shape[1]
    merged = x.permute(0, 2, 1, 3, 4).reshape(b_n, n_n, s_n * g_n, d_n)
    return torch.matmul(merged.float(), w.float()).reshape(b_n, n_n, s_n * g_n, ncol)


def _run(kernel, g, dtype, device, s_tiles=2, ncol=2 * N_TILE, d=2 * K_TILE, b_n=2, n_n=3):
    """Drive one grouped matmul and return (out, reference)."""
    s_step = _merged_s_step(g)
    s_n = s_tiles * s_step  # whole number of merged tiles: no tail handling in the kernel
    torch.manual_seed(0)
    x = torch.randn([b_n, s_n, n_n, g, d], device=device, dtype=dtype)
    w = torch.randn([d, ncol], device=device, dtype=dtype)
    out = torch.zeros([b_n, n_n, s_n * g, ncol], device=device, dtype=torch.float32)
    kernel(x, w, out)
    torch.npu.synchronize()
    return out, _reference(x, w)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 4, 8, 16, 32, 64])
def test_grouped_matmul_all_dynamic(g):
    """Full tiled matmul with every dim dynamic, G included: nValue and ndNum are runtime."""
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run(grouped_matmul_fp16, g, torch.float16, device)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul all-dynamic G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 4])
def test_grouped_matmul_all_dynamic_bf16(g):
    """Same, in BF16."""
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run(grouped_matmul_bf16, g, torch.bfloat16, device)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul bf16 G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=8e-2, atol=8e-2)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 8, 64])
def test_grouped_matmul_static_g(g):
    """G pinned in the signature, so codegen folds nValue/ndNum to constants."""
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run(_make_static_g_matmul(g), g, torch.float16, device)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul static-G G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
def test_grouped_matmul_rejects_wrong_row_grouping():
    """The result must be (s, g) ordered, not (g, s) -- the whole point of the merged load.

    Guards the tests above: without it a kernel that grouped rows the other way round would
    still look plausible, since both orderings produce the same set of rows.
    """
    device = ST_DEVICE
    _require_3510(device)
    g = 4
    out, ref = _run(grouped_matmul_fp16, g, torch.float16, device)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    b_n, n_n, rows, ncol = ref.shape
    swapped = ref.reshape(b_n, n_n, rows // g, g, ncol).transpose(2, 3).reshape(b_n, n_n, rows, ncol)
    assert (out - swapped).abs().max().item() > 1.0, "(s, g) and (g, s) orderings are indistinguishable here"


def _merged(x):
    """merged[b, h] = x[b, :, h, :, :].reshape(S*G, D), in fp32."""
    b_n, s_n, n_n, g_n, d_n = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b_n, n_n, s_n * g_n, d_n).float()


def _run_transposed(g, dtype, device, s_tiles=2, ncol=2 * N_TILE, d=2 * K_TILE, b_n=2, n_n=3):
    """out[b, h] = merged[b, h]^T @ v[b, h] -- the merged operand transposed, [D, NCOL]."""
    s_step = _merged_s_step(g)
    s_n = s_tiles * s_step
    rows = s_n * g
    torch.manual_seed(0)
    x = torch.randn([b_n, s_n, n_n, g, d], device=device, dtype=dtype)
    v = torch.randn([b_n, n_n, rows, ncol], device=device, dtype=dtype)
    out = torch.zeros([b_n, n_n, d, ncol], device=device, dtype=torch.float32)
    grouped_matmul_transposed_left(x, v, out)
    torch.npu.synchronize()
    ref = torch.matmul(_merged(x).transpose(2, 3), v.float())
    return out, ref


def _run_as_right(g, dtype, device, s_tiles=2, mrow=2 * MROW_TILE, d=2 * K_TILE, b_n=2, n_n=3):
    """out[b, h] = a[b, h] @ merged[b, h] -- the merged operand as the right matrix, [MROW, D]."""
    s_step = _merged_s_step(g)
    s_n = s_tiles * s_step
    rows = s_n * g
    torch.manual_seed(0)
    x = torch.randn([b_n, s_n, n_n, g, d], device=device, dtype=dtype)
    a = torch.randn([b_n, n_n, mrow, rows], device=device, dtype=dtype)
    out = torch.zeros([b_n, n_n, mrow, d], device=device, dtype=torch.float32)
    grouped_matmul_as_right(a, x, out)
    torch.npu.synchronize()
    ref = torch.matmul(a.float(), _merged(x))
    return out, ref


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 8, 64])
def test_grouped_matmul_transposed_left(g):
    """The merged operand transposed, loaded that way in one instruction by order=[4, 3, 1].

    nd2nz cannot transpose; the reversed order does not ask it to. It issues the identical
    transfer into the ZN [K, M] alias of the NZ [M, K] tile -- the same bytes under the other
    label -- and the Mat->L0 move realizes the transpose. Verified as pure data movement: the
    probe of this path reproduced x^T exactly.
    """
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run_transposed(g, torch.float16, device)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul transposed-left G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 8, 64])
def test_grouped_matmul_as_right(g):
    """The merged load feeding the RIGHT operand: Mat(NZ) -> Right(ZN) is the ordinary move."""
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run_as_right(g, torch.float16, device)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul as-right G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
def test_transposed_left_is_really_transposed():
    """Guard: the transposed kernel must not silently return the untransposed product.

    The reversed order and the Mat->L0 move each flip the operand, so getting one of them wrong
    yields merged @ v -- a plausible-looking result of a different shape only when D != NCOL.
    Pinning D == NCOL makes both products the same shape, so only the values distinguish them.
    """
    device = ST_DEVICE
    _require_3510(device)
    g = 4
    out, ref = _run_transposed(g, torch.float16, device, ncol=2 * K_TILE, d=2 * K_TILE)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    assert (out - ref.transpose(2, 3)).abs().max().item() > 1.0, "transposed and plain products are indistinguishable"


@pytest.mark.soc("950")
def test_merged_order_must_be_ascending_or_its_exact_reverse():
    """A merged order naming a non-innermost column axis is rejected, not silently reinterpreted.

    ``order=[1, 4, 3]`` sorts to the same tile_dims as the correct ``[1, 3, 4]`` -- and as the
    transposed ``[4, 3, 1]`` -- so before the raw order was checked this spelling was accepted
    and transferred the ascending one's data. Only the two orderings above have a meaning.
    """
    @pl.jit(auto_mutex=True)
    def bad(x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16]):
        t = pl.make_tile_group(
            type=pl.TileType(shape=[M_TILE, K_TILE], dtype=pl.DT_FP16,
                             target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=L1_X, mutex_ids=[0])
        with pl.section_cube():
            pl.load(t.current(), x, [0, 0, 0, 0, 0], order=[1, 4, 3])

    with pytest.raises(Exception, match="ascending"):
        from pypto_pro import ir

        bad.to_kernel_def().parse_target_program(ir.SectionKind.Cube)


def _run_both_merged(g, dtype, device, s_tiles=2, d=2 * K_TILE, b_n=2, n_n=3):
    """out[b, h] = merged[b, h]^T @ merged[b, h] -- both operands merged, left transposed."""
    s_step = _merged_s_step(g)
    s_n = s_tiles * s_step
    torch.manual_seed(0)
    x = torch.randn([b_n, s_n, n_n, g, d], device=device, dtype=dtype)
    out = torch.zeros([b_n, n_n, d, d], device=device, dtype=torch.float32)
    grouped_matmul_both_merged_transposed(x, out)
    torch.npu.synchronize()
    m = _merged(x)
    return out, torch.matmul(m.transpose(2, 3), m)


def _run_right_transposed(g, dtype, device, s_tiles=2, mrow=2 * MROW_TILE, d=2 * K_TILE, b_n=2, n_n=3,
                          kernel=grouped_matmul_right_merged_transposed):
    """out[b, h] = a[b, h] @ merged[b, h]^T -- merged operand on the right, transposed.

    ``kernel`` picks the spelling: the pl.reinterpret view (default) or the reversed order. The
    seed is fixed here, so both are driven with identical inputs and their outputs compare.
    """
    s_step = _merged_s_step(g)
    s_n = s_tiles * s_step
    rows = s_n * g
    torch.manual_seed(0)
    x = torch.randn([b_n, s_n, n_n, g, d], device=device, dtype=dtype)
    a = torch.randn([b_n, n_n, mrow, d], device=device, dtype=dtype)
    out = torch.zeros([b_n, n_n, mrow, rows], device=device, dtype=torch.float32)
    kernel(a, x, out)
    torch.npu.synchronize()
    return out, torch.matmul(a.float(), _merged(x).transpose(2, 3))


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 8, 64])
def test_grouped_matmul_right_merged_transposed(g):
    """Merged load feeding the RIGHT operand transposed: Mat(ZN) -> Right(ZN).

    Complements test_grouped_matmul_as_right (right operand as loaded) and
    test_grouped_matmul_transposed_left (transposed, but on the left): this is the fourth
    corner -- right AND transposed, where the move's source and destination name the same
    layout rather than a fractal flip.
    """
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run_right_transposed(g, torch.float16, device)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul right-merged-transposed G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 8, 64])
def test_grouped_matmul_right_merged_transposed_by_order(g):
    """The same fourth corner, with the transpose asked for in the load: order=[4, 3, 1].

    A directly loaded ZN tile driving Mat(ZN) -> Right(ZN). test_grouped_matmul_transposed_left
    covers the other destination for such a tile, Mat(ZN) -> Left(NZ); this one covers the move
    that does not flip the fractal, so the label alone carries the transpose all the way to L0B.
    """
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run_right_transposed(g, torch.float16, device,
                                     kernel=grouped_matmul_right_merged_transposed_by_order)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul right-merged-transposed by order G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 64])
def test_right_merged_transposed_spellings_agree_exactly(g):
    """The reversed order and the pl.reinterpret view must agree BIT FOR BIT, not within rtol.

    Both kernels issue the same multi-ND2NZ DMA into the same L1 address and run the same matmul
    over the same inputs; the only difference is where the ZN label comes from -- the load's
    order, or a relabelling of the buffer an ascending load filled. Anything the reversed
    spelling got wrong in ndNum, nValue, dValue or loop3DstStride would move some element, so
    exact equality is the assertion that has teeth here. A tolerance check would pass on a
    kernel that transferred nothing at all, since both sides accumulate the same zeros.
    """
    device = ST_DEVICE
    _require_3510(device)
    out_view, ref = _run_right_transposed(g, torch.float16, device)
    out_order, _ = _run_right_transposed(g, torch.float16, device,
                                         kernel=grouped_matmul_right_merged_transposed_by_order)
    logging.info("right-merged-transposed spellings G=%d: max|view - order| = %s",
                 g, (out_view - out_order).abs().max().item())
    # Guard against both being trivially zero: the shared reference must be reproduced too.
    torch.testing.assert_close(out_order, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_order, out_view, rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [2, 8, 64])
def test_grouped_matmul_both_merged_transposed(g):
    """Both operands merged from one grouped Tensor, with the left one transposed."""
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run_both_merged(g, torch.float16, device)
    diff = (out - ref).abs().max().item()
    logging.info("grouped_matmul both-merged G=%d: max|out - ref| = %s", g, diff)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.soc("950")
def test_both_merged_result_is_symmetric():
    """Guard on the both-merged kernel: merged^T @ merged is symmetric, its transpose is not new.

    A Gram matrix is a weak check on its own -- so this also pins that the result is NOT the
    plain merged @ merged^T, which has a different shape only when D != rows. Asserting
    symmetry catches a left/right mix-up that still produced a [D, D] output.
    """
    device = ST_DEVICE
    _require_3510(device)
    out, ref = _run_both_merged(4, torch.float16, device)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
    # merged^T @ merged is symmetric by construction; a swapped-operand result would not be.
    torch.testing.assert_close(out, out.transpose(2, 3), rtol=3e-2, atol=3e-2)
    assert out.abs().max().item() > 1.0, "degenerate all-zero result would satisfy symmetry trivially"


@pytest.mark.soc("950")
def test_right_merged_transposed_is_really_transposed():
    """Guard: the right-transposed kernel must not return the untransposed product.

    Pinning rows == D makes a @ merged and a @ merged^T the same shape, so only the values
    distinguish them.
    """
    device = ST_DEVICE
    _require_3510(device)
    g = 4
    # s_tiles=1 with g=4 gives rows = M_TILE = 128; d = 128 matches it.
    out, ref = _run_right_transposed(g, torch.float16, device, s_tiles=1, d=M_TILE)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    assert (out - ref.transpose(2, 3)).abs().max().item() > 1.0, "transposed and plain products are indistinguishable"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _g in (2, 4, 8, 16, 32, 64):
        test_grouped_matmul_all_dynamic(_g)
    for _g in (2, 4):
        test_grouped_matmul_all_dynamic_bf16(_g)
    for _g in (2, 8, 64):
        test_grouped_matmul_static_g(_g)
    test_grouped_matmul_rejects_wrong_row_grouping()
    for _g in (2, 8, 64):
        test_grouped_matmul_transposed_left(_g)
    for _g in (2, 8, 64):
        test_grouped_matmul_as_right(_g)
    test_transposed_left_is_really_transposed()
    for _g in (2, 8, 64):
        test_grouped_matmul_right_merged_transposed(_g)
    for _g in (2, 8, 64):
        test_grouped_matmul_right_merged_transposed_by_order(_g)
    for _g in (2, 64):
        test_right_merged_transposed_spellings_agree_exactly(_g)
    for _g in (2, 8, 64):
        test_grouped_matmul_both_merged_transposed(_g)
    test_both_merged_result_is_symmetric()
    test_right_merged_transposed_is_really_transposed()


@pl.jit(auto_mutex=True)
def merged_load_valid_rows(
    x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
    w: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[128, 64], pl.DT_FP32],
    valid: pl.DT_INT32,
    start: pl.DT_INT32,
):
    xl = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=0, mutex_ids=[0])
    wl = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=0x10000, mutex_ids=[1])
    a = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=0, mutex_ids=[2])
    b = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=0, mutex_ids=[3])
    c = pl.make_tile_group(type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP32,
                           target_memory=pl.MemorySpace.Acc, layout=pl.NZ), addrs=0, mutex_ids=[4])
    with pl.section_cube():
        t = xl.current()
        pl.load(t, x, [0, 0, 0, 0, 0], order=[1, 3, 4])
        pl.set_validshape(t, [valid, 64])
        pl.load(t, x, [0, start, 0, 0, 0], order=[1, 3, 4])
        pl.set_validshape(t, [128, 64])
        pl.load(wl.current(), w, [0, 0])
        pl.move(a.current(), t)
        pl.move(b.current(), wl.current())
        pl.matmul(c.current(), a.current(), b.current())
        pl.store(out, c.current(), [0, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("inner,valid", [(4, 16), (4, 18), (32, 16)])
def test_merged_load_respects_runtime_valid_rows(inner, valid):
    _require_3510(ST_DEVICE)
    x = torch.arange(64 * 2 * inner, device=ST_DEVICE, dtype=torch.float32)
    x = x.reshape(1, 64, 2, inner, 1).expand(1, 64, 2, inner, 64).contiguous().half()
    w = torch.eye(64, device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty([128, 64], device=ST_DEVICE, dtype=torch.float32)
    merged_load_valid_rows(x, w, out, valid, 32)
    torch.npu.synchronize()
    baseline = x[0, :, 0].reshape(-1, 64)[:128].float()
    replacement = x[0, 32:, 0].reshape(-1, 64)[:128].float()
    torch.testing.assert_close(out[:valid], replacement[:valid], rtol=0, atol=0)
    # Ignore any hardware padding immediately after the valid region. Farther rows must
    # retain the initial load, rather than being overwritten by a full-tile DMA.
    torch.testing.assert_close(out[64:], baseline[64:], rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize("inner", [3, 65536])
def test_merged_load_dynamic_inner_remainder_and_width(inner):
    _require_3510(ST_DEVICE)
    outer = (128 + inner - 1) // inner
    x = torch.full([1, outer, 2, inner, 64], 7.0, device=ST_DEVICE, dtype=torch.float16)
    x[:, :, 1] = 11.0
    w = torch.eye(64, device=ST_DEVICE, dtype=torch.float16)
    out = torch.empty([128, 64], device=ST_DEVICE, dtype=torch.float32)
    merged_load_valid_rows(x, w, out, 128, 0)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.full_like(out, 7.0), rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize("g", [0, 3, 256])
def test_full_grouped_matmul_rejects_nondivisor_before_launch(g):
    with pytest.raises(ValueError, match="G to divide M_TILE"):
        _run(grouped_matmul_fp16, g, torch.float16, "cpu")


def _make_merged_dtype_matmul(dtype, transposed, inner_extent=DYN):
    rows, reduction = (64, 32) if transposed else (32, 64)
    layout = pl.ZN if transposed else pl.NZ
    order = [4, 3, 1] if transposed else [1, 3, 4]

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[1, DYN, 2, inner_extent, 64], dtype],
        w: pl.Tensor[[reduction, reduction], dtype],
        out: pl.Tensor[[rows, reduction], pl.DT_FP32],
    ):
        xl = pl.make_tile_group(type=pl.TileType(shape=[rows, reduction], dtype=dtype,
                                 target_memory=pl.MemorySpace.Mat, layout=layout), addrs=0, mutex_ids=[0])
        wl = pl.make_tile_group(type=pl.TileType(shape=[reduction, reduction], dtype=dtype,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=0x10000, mutex_ids=[1])
        a = pl.make_tile_group(type=pl.TileType(shape=[rows, reduction], dtype=dtype,
                                target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=0, mutex_ids=[2])
        b = pl.make_tile_group(type=pl.TileType(shape=[reduction, reduction], dtype=dtype,
                                target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=0, mutex_ids=[3])
        c = pl.make_tile_group(type=pl.TileType(shape=[rows, reduction], dtype=pl.DT_FP32,
                                target_memory=pl.MemorySpace.Acc, layout=pl.NZ), addrs=0, mutex_ids=[4])
        with pl.section_cube():
            pl.load(xl.current(), x, [0, 1, 1, 0, 0], order=order)
            pl.load(wl.current(), w, [0, 0])
            pl.move(a.current(), xl.current())
            pl.move(b.current(), wl.current())
            pl.matmul(c.current(), a.current(), b.current())
            pl.store(out, c.current(), [0, 0])

    return kernel


@pytest.mark.soc("950")
@pytest.mark.parametrize("dtype,torch_dtype", [
    (pl.DT_FP8E4M3FN, torch.float8_e4m3fn),
    (pl.DT_FP8E5M2, torch.float8_e5m2),
    (pl.DT_FP32, torch.float32),
])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("inner", [2, 3])
def test_merged_load_element_widths(dtype, torch_dtype, transposed, inner):
    """Exercise C0=32 for FP8 and C0=8 for FP32, including a partial ND matrix."""
    _require_3510(ST_DEVICE)
    outer = 1 + (32 + inner - 1) // inner
    shape = (1, outer, 2, inner, 64)
    torch.manual_seed(9)
    host = torch.randint(-4, 5, shape, dtype=torch.int32).float().to(torch_dtype)
    x = host.to(ST_DEVICE)
    reduction = 32 if transposed else 64
    w = torch.eye(reduction).to(torch_dtype).to(ST_DEVICE)
    reference = host.float()[0, 1:, 1].reshape(-1, 64)[:32]
    if transposed:
        reference = reference.T.contiguous()
    out = torch.empty(reference.shape, device=ST_DEVICE, dtype=torch.float32)
    _make_merged_dtype_matmul(dtype, transposed)(x, w, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize("inner", [3, 64, 65536])
@pytest.mark.parametrize("transposed", [False, True])
def test_merged_load_static_inner_matches_dynamic(inner, transposed):
    """Static and dynamic G load the same rows, across matrix boundaries or within one matrix."""
    _require_3510(ST_DEVICE)
    outer = 1 + (32 + inner - 1) // inner
    shape = (1, outer, 2, inner, 64)
    torch.manual_seed(9)
    host = torch.randint(-4, 5, shape, dtype=torch.int32).half()
    x = host.to(ST_DEVICE)
    reduction = 32 if transposed else 64
    w = torch.eye(reduction, dtype=torch.float16, device=ST_DEVICE)
    reference = host.float()[0, 1:, 1].reshape(-1, 64)[:32]
    if transposed:
        reference = reference.T.contiguous()
    static_out = torch.empty(reference.shape, device=ST_DEVICE, dtype=torch.float32)
    dynamic_out = torch.empty_like(static_out)
    _make_merged_dtype_matmul(pl.DT_FP16, transposed, inner)(x, w, static_out)
    _make_merged_dtype_matmul(pl.DT_FP16, transposed)(x, w, dynamic_out)
    torch.npu.synchronize()
    static_result = static_out.cpu()
    dynamic_result = dynamic_out.cpu()
    torch.testing.assert_close(static_result, reference, rtol=0, atol=0)
    torch.testing.assert_close(dynamic_result, reference, rtol=0, atol=0)
    print(f"merged load G={inner}, transposed={transposed}: "
          f"static max error={(static_result - reference).abs().max().item()}, "
          f"dynamic max error={(dynamic_result - reference).abs().max().item()}")


def _make_merged_window_kernel(transposed, matrix_stride=0):
    rows, reduction = (64, 128) if transposed else (128, 64)
    shape = [rows, reduction]
    order = [4, 3, 1] if transposed else [1, 3, 4]
    layout = pl.ZN if transposed else pl.NZ
    outer_offset = 0 if matrix_stride else 1

    @pl.jit(auto_mutex=True)
    def window(
        x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
        w: pl.Tensor[[reduction, reduction], pl.DT_FP16],
        out: pl.Tensor[[rows, reduction], pl.DT_FP32],
        valid: pl.DT_INT32,
        columns: pl.DT_INT32,
    ):
        xl = pl.make_tile_group(type=pl.TileType(shape=shape, dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Mat, layout=layout), addrs=0, mutex_ids=[0])
        wl = pl.make_tile_group(type=pl.TileType(shape=[reduction, reduction], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=0x10000, mutex_ids=[1])
        a = pl.make_tile_group(type=pl.TileType(shape=shape, dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=0, mutex_ids=[2])
        b = pl.make_tile_group(type=pl.TileType(shape=[reduction, reduction], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=0, mutex_ids=[3])
        c = pl.make_tile_group(type=pl.TileType(shape=shape, dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Acc, layout=pl.NZ), addrs=0, mutex_ids=[4])
        if matrix_stride:
            source = pl.make_tensor(x, [1, 1, 2, x.shape[3], 64],
                                    [matrix_stride, matrix_stride, x.shape[3] * 64, 64, 1])
        else:
            source = x
        with pl.section_cube():
            t = xl.current()
            pl.load(t, source, [0, 0, 1, 0, 0], order=order)
            if transposed:
                pl.set_validshape(t, [columns, valid])
            else:
                pl.set_validshape(t, [valid, columns])
            pl.load(t, source, [0, outer_offset, 1, 0, 0], order=order)
            pl.set_validshape(t, shape)
            pl.load(wl.current(), w, [0, 0])
            pl.move(a.current(), t)
            pl.move(b.current(), wl.current())
            pl.matmul(c.current(), a.current(), b.current())
            pl.store(out, c.current(), [0, 0])

    return window


@pytest.mark.soc("950")
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("inner,valid", [(4, 16), (4, 18), (32, 16), (3, 32), (32, 1), (256, 16), (65536, 16)])
@pytest.mark.parametrize("columns", [35, 64])
def test_merged_load_runtime_window(transposed, inner, valid, columns):
    device = ST_DEVICE
    _require_3510(device)
    torch.manual_seed(73)
    host = torch.randint(-4, 5, (1, 1 + (128 + inner - 1) // inner, 2, inner, 64)).half()
    x = host.to(device)
    reduction = 128 if transposed else 64
    w = torch.eye(reduction, dtype=torch.float16, device=device)
    out = torch.empty((64, 128) if transposed else (128, 64), dtype=torch.float32, device=device)
    _make_merged_window_kernel(transposed)(x, w, out, valid, columns)
    torch.npu.synchronize()
    actual = out.cpu().T if transposed else out.cpu()
    initial = host[0, :, 1].reshape(-1, 64)[:128].float()
    replacement = host[0, 1:, 1].reshape(-1, 64)[:valid].float()
    torch.testing.assert_close(actual[:valid, :columns], replacement[:, :columns], rtol=0, atol=0)
    torch.testing.assert_close(actual[64:], initial[64:], rtol=0, atol=0)


@pytest.mark.soc("950")
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("matrix_stride", [2**31, 2**32])
def test_merged_single_matrix_does_not_use_outer_stride(transposed, matrix_stride):
    """Full and tail-only loads keep an unused wide matrix stride in their descriptor."""
    _require_3510(ST_DEVICE)
    torch.manual_seed(74)
    host = torch.randint(-4, 5, (1, 1, 2, 128, 64)).half()
    x = host.to(ST_DEVICE)
    reduction = 128 if transposed else 64
    w = torch.eye(reduction, dtype=torch.float16, device=ST_DEVICE)
    out = torch.empty((64, 128) if transposed else (128, 64), dtype=torch.float32, device=ST_DEVICE)
    _make_merged_window_kernel(transposed, matrix_stride)(x, w, out, 16, 64)
    torch.npu.synchronize()
    actual = out.cpu().T if transposed else out.cpu()
    torch.testing.assert_close(actual, host[0, 0, 1].float(), rtol=0, atol=0)


def _make_reused_merged_descriptor_kernel(use_view):
    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[DYN, DYN, DYN, DYN, DYN], pl.DT_FP16],
        out: pl.Tensor[[4, 64, 64], pl.DT_FP32],
    ):
        xl = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addrs=0, mutex_ids=[0])
        yl = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Mat, layout=pl.ZN), addrs=0x10000, mutex_ids=[1])
        a = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Left, layout=pl.NZ), addrs=0, mutex_ids=[2])
        b = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Right, layout=pl.ZN), addrs=0, mutex_ids=[3])
        c = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Acc, layout=pl.NZ), addrs=0, mutex_ids=[4])
        if use_view:
            source = pl.make_tensor(x, [1, x.shape[1], 2, x.shape[3], 80],
                                    [x.shape[1] * 2 * x.shape[3] * 80, 2 * x.shape[3] * 80,
                                     x.shape[3] * 80, 80, 1])
        else:
            source = x
        with pl.section_cube():
            pl.load(xl.current(), source, [0, 0, 1, 0, 8], order=[1, 3, 4])
            for i in pl.range(4):
                t = xl.current()
                u = yl.current()
                if i == 0:
                    pl.load(t, source, [0, 1, 1, 0, 8], order=[1, 4])
                else:
                    pl.set_validshape(t, [16, 35])
                    pl.load(t, source, [0, 0, 1, 0, 8], order=[1, 3, 4])
                    pl.set_validshape(t, [64, 64])
                    pl.load(t, source, [0, 1, 1, 0, 8], order=[1, 3, 4])
                pl.set_validshape(u, [35, 16])
                pl.load(u, source, [0, 0, 1, 0, 8], order=[4, 3, 1])
                pl.set_validshape(u, [64, 64])
                pl.load(u, source, [0, 2, 1, 0, 8], order=[4, 3, 1])
                if i == 3:
                    pl.load(t, source, [0, 1, 0, 1, 8], order=[1, 2, 4])
                    pl.load(u, source, [0, 2, 0, 1, 8], order=[4, 2, 1])
                pl.move(a.current(), t)
                pl.move(b.current(), u)
                pl.matmul(c.current(), a.current(), b.current())
                pl.store(out, c.current(), [i, 0, 0])

    return kernel


@pytest.mark.soc("950")
@pytest.mark.parametrize("use_view", [False, True])
@pytest.mark.parametrize("inner", [4, 32])
def test_merged_load_reuses_descriptors_across_windows_and_layouts(use_view, inner):
    """Reuse strides across short/full windows and select distinct strides for different axes."""
    _require_3510(ST_DEVICE)
    torch.manual_seed(91)
    host = torch.randint(-2, 3, (1, 128, 2, inner, 80)).half()
    x = host.to(ST_DEVICE)
    out = torch.empty((4, 64, 64), dtype=torch.float32, device=ST_DEVICE)
    _make_reused_merged_descriptor_kernel(use_view)(x, out)
    torch.npu.synchronize()
    ordinary = host[0, 1:65, 1, 0, 8:72].float()
    merged = host[0, 1:, 1, :, 8:72].reshape(-1, 64)[:64].float()
    right = host[0, 2:, 1, :, 8:72].reshape(-1, 64)[:64].float().T
    other_axes = host[0, 1:, :, 1, 8:72].reshape(-1, 64)[:64].float()
    other_right = host[0, 2:, :, 1, 8:72].reshape(-1, 64)[:64].float().T
    reference = torch.stack([ordinary @ right, merged @ right, merged @ right, other_axes @ other_right])
    torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)
