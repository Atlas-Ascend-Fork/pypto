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

"""3-D convolution ST on A5: end-to-end img2col + matmul data path vs torch.conv3d.

The kernel consumes 5-D NCDHW tensors (all-STATIC dims) and walks the full conv data
path: GM -> L1 ConvTiles (``NDC1HWC0`` fmap / ``FRACTAL_Z_3D`` weight) -> ``pl.img2col``
fold into L0A -> ``pl.matmul`` / ``pl.matmul_acc`` accumulation in L0C -> FP16 store
back to GM. Output extents are tiled across D (dk loop), H*W (m loop), C_in (k loop)
and C_out (n loop); the m tiles are partitioned across the cube cores via
``pl.range(core_id, m_num, core_num)``, and every L1/L0 tile is resized to its
per-tile valid region with ``pl.set_validshape``.

The result is checked against ``torch.conv3d`` on all-ones fmap/weight with a 1x1x1
kernel, C_in=1024, C_out=4096, D/H/W = 91/45/80: each output element is the exact
FP16-representable sum of 1024 ones, so a 1e-2 tolerance catches any mis-load /
mis-fold / mis-store without FP16 accumulation noise.

Doc: docs/zh/api/pro_api/SIMD-API/memory_data_movement/img2col.md
"""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.platform import get_platform_info
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the npu backend

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# Conv geometry: NCDHW 3-D conv, 1x1x1 kernel, C_in = 1024, C_out = 4096.
BATCH = 1
CIN = 1024
COUT = 4096
DIN, HIN, WIN = 91, 45, 80
KD, KH, KW = 1, 1, 1
GROUPS = 1

STRIDES = [1, 1, 1]
PADS = [0, 0, 0]
DILATIONS = [1, 1, 1]

DOUT = (DIN + 2 * PADS[0] - DILATIONS[0] * (KD - 1) - 1) // STRIDES[0] + 1
HOUT = (HIN + 2 * PADS[1] - DILATIONS[1] * (KH - 1) - 1) // STRIDES[1] + 1
WOUT = (WIN + 2 * PADS[2] - DILATIONS[2] * (KW - 1) - 1) // STRIDES[2] + 1

FMAP_SHAPE = (BATCH, CIN, DIN, HIN, WIN)
WEIGHT_SHAPE = (COUT, CIN // GROUPS, KD, KH, KW)
OUTPUT_SHAPE = (BATCH, COUT, DOUT, HOUT, WOUT)

DTYPE_TORCH = torch.float16


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def conv_kernel(
        fmap: pl.Tensor[[pl.STATIC, pl.STATIC, pl.STATIC, pl.STATIC, pl.STATIC], pl.DT_FP16, pl.NCDHW],
        weight: pl.Tensor[[pl.STATIC, pl.STATIC, pl.STATIC, pl.STATIC, pl.STATIC], pl.DT_FP16, pl.NCDHW],
        output: pl.Tensor[[pl.STATIC, pl.STATIC, pl.STATIC, pl.STATIC, pl.STATIC], pl.DT_FP16, pl.NCDHW]
):
    core_num = pl.get_block_num()
    core_id = pl.get_block_idx()

    dtype = pl.DT_FP16
    batch = output.shape[0]
    cout = output.shape[1]
    dout = output.shape[2]
    hout = output.shape[3]
    wout = output.shape[4]
    cin = fmap.shape[1]
    din = fmap.shape[2]
    hin = fmap.shape[3]
    win = fmap.shape[4]
    dk = weight.shape[2]
    hk = weight.shape[3]
    wk = weight.shape[4]
    hk_wk = weight.shape[3] * weight.shape[4]
    strides = STRIDES
    pads = PADS
    dilations = DILATIONS

    cin0 = 16
    cin1 = (cin + cin0 - 1) // cin0
    cin1_hk_wk_cin0 = cin1 * hk_wk * cin0
    cout0 = 16

    m_l1 = 160  # multiple of wout
    k_l1 = 128  # multiple of hk_wk*cin0
    n_l1 = 128  # multiple of cout0
    m_num = (hout * wout + m_l1 - 1) // m_l1
    m_l0, k_l0, n_l0 = 32, 64, 64  # multiple of 16

    cin1_l1 = k_l1 // (cin0 * hk_wk)
    hout_l1 = (m_l1 + wout - 1) // wout
    hin_l1 = (hout_l1 - 1) * strides[1] + (hk - 1) * dilations[1] + 1
    wout_l1 = wout if (m_l1 % wout == 0) else (m_l1 % wout)
    win_l1 = (wout_l1 - 1) * strides[2] + (wk - 1) * dilations[2] + 1
    cout1_l1 = n_l1 // cout0

    fmap_l1_type = pl.TileType(shape=[1, 1, cin1_l1, hin_l1, win_l1, cin0], dtype=dtype,
                               valid_shape=[-1, -1, -1, -1, -1, -1],
                               target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NDC1HWC0)
    weight_l1_type = pl.TileType(shape=[cin1_l1 * hk_wk, cout1_l1, cout0, cin0], dtype=dtype,
                                 valid_shape=[-1, -1, -1, -1],
                                 target_memory=pl.MemorySpace.Mat,
                                 layout=pl.Layout.FRACTAL_Z_3D)  # [Cin1HkWk, Cout1, Cout0, Cin0]
    fmap_l0_type = pl.TileType(shape=[m_l0, k_l0], dtype=dtype, valid_shape=[-1, -1], target_memory=pl.MemorySpace.Left,
                               layout=pl.NZ)
    weight_l0_type = pl.TileType(shape=[k_l0, n_l0], dtype=dtype, valid_shape=[-1, -1],
                                 target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    output_l0_type = pl.TileType(shape=[m_l0, n_l0], dtype=pl.DT_FP32, valid_shape=[-1, -1],
                                 target_memory=pl.MemorySpace.Acc, layout=pl.NZ)
    fmap_l1_buffer = pl.make_tile_group(type=fmap_l1_type, addrs=0x00000, mutex_ids=[0])
    weight_l1_buffer = pl.make_tile_group(type=weight_l1_type, addrs=0x20000, mutex_ids=[1])
    fmap_l0_db = pl.make_tile_group(type=fmap_l0_type, addrs=0x00000, mutex_ids=[2])
    weight_l0_db = pl.make_tile_group(type=weight_l0_type, addrs=0x00000, mutex_ids=[3])
    output_l0_db = pl.make_tile_group(type=output_l0_type, addrs=0x00000, mutex_ids=[4])

    with pl.section_cube():
        for batch_offset in pl.range(batch):
            for dout_offset in pl.range(dout):
                # din_offset && din_size
                din_start = dout_offset * strides[0] - pads[0]
                din_offset = din_start
                dk_l1 = dk
                dk_start = 0
                if din_start < 0:
                    tmp_dk = (-din_start + dilations[0] - 1) // dilations[0]
                    din_offset += tmp_dk * dilations[0]
                    dk_start = tmp_dk
                din_end = din_start + (dk - 1) * dilations[0] + 1
                if din_end > din:
                    tmp_dk = (din_end - din + dilations[0] - 1) // dilations[0]
                    dk_l1 -= tmp_dk
                din_base = din_offset

                for m_id in pl.range(core_id, m_num, core_num):
                    m_l1_offset = m_id * m_l1
                    m_l1_size = pl.min(m_l1, hout * wout - m_l1_offset)
                    hout_offset = m_l1_offset // wout
                    hout_size = (m_l1_size + wout - 1) // wout
                    wout_offset = m_l1_offset % wout
                    wout_size = pl.min(wout, m_l1_size)
                    # hin_offset && hin_size
                    hin_start = hout_offset * strides[1] - pads[1]
                    hin_offset = pl.max(hin_start, 0)
                    hin_end = hin_start + (hout_size - 1) * strides[1] + (hk - 1) * dilations[1] + 1
                    hin_size = hin_end - hin_start
                    if hin_start < 0:
                        hin_size += hin_start
                    if hin_end > hin:
                        hin_size -= (hin_end - hin)
                    # win_offset && win_size
                    win_start = wout_offset * strides[2] - pads[2]
                    win_offset = pl.max(win_start, 0)
                    win_end = win_start + (wout_size - 1) * strides[2] + (wk - 1) * dilations[2] + 1
                    win_size = win_end - win_start
                    if win_start < 0:
                        win_size += win_start
                    if win_end > win:
                        win_size -= (win_end - win)

                    for n_l1_offset in pl.range(0, cout, n_l1):
                        cout_offset = n_l1_offset
                        n_l1_size = pl.min(n_l1, cout - n_l1_offset)
                        cout1_size = (n_l1_size + cout0 - 1) // cout0

                        for m_l0_offset in pl.range(0, m_l1_size, m_l0):
                            m_l0_size = pl.min(m_l0, m_l1_size - m_l0_offset)
                            for n_l0_offset in pl.range(0, n_l1_size, n_l0):
                                n_l0_size = pl.min(n_l0, n_l1_size - n_l0_offset)
                                c_l0_tile = output_l0_db.next()
                                pl.set_validshape(c_l0_tile, [m_l0_size, n_l0_size])
                                din_offset = din_base

                                for dk_offset in pl.range(dk_start, dk_l1, 1):
                                    for k_l1_offset in pl.range(0, cin1_hk_wk_cin0, k_l1):
                                        cin_offset = k_l1_offset // hk_wk
                                        k_l1_size = pl.min(k_l1, cin1_hk_wk_cin0 - k_l1_offset)
                                        cin1_size = k_l1_size // (cin0 * hk_wk)

                                        a_l1_tile = fmap_l1_buffer.next()
                                        b_l1_tile = weight_l1_buffer.next()
                                        pl.set_validshape(a_l1_tile, [1, 1, cin1_size, hin_size, win_size, cin0])
                                        pl.set_validshape(b_l1_tile, [cin1_size * hk_wk, cout1_size, cout0, cin0])
                                        pl.load(a_l1_tile, fmap,
                                                [batch_offset, cin_offset, din_offset, hin_offset, win_offset])
                                        pl.load(b_l1_tile, weight, [cout_offset, cin_offset, dk_offset, 0, 0])

                                        for k_l0_offset in pl.range(0, k_l1_size, k_l0):
                                            k_l0_size = pl.min(k_l0, k_l1_size - k_l0_offset)
                                            a_l0_tile = fmap_l0_db.next()
                                            b_l0_tile = weight_l0_db.next()
                                            pl.set_validshape(a_l0_tile, [m_l0_size, k_l0_size])
                                            pl.set_validshape(b_l0_tile, [k_l0_size, n_l0_size])
                                            cfg = pl.Img2ColConfig(
                                                kernel=[hk, wk],
                                                stride=[strides[1], strides[2]],
                                                dilation=[dilations[1], dilations[2]],
                                                padding=[pads[1], pads[1], pads[2], pads[2]]
                                            )
                                            pl.img2col(a_l0_tile, a_l1_tile, config=cfg, pos_m=m_l0_offset,
                                                       pos_k=k_l0_offset)
                                            pl.move(b_l0_tile, b_l1_tile, [k_l0_offset, n_l0_offset])
                                            if dk_offset == dk_start and k_l1_offset == 0 and k_l0_offset == 0:
                                                pl.matmul(c_l0_tile, a_l0_tile, b_l0_tile)
                                            else:
                                                pl.matmul_acc(c_l0_tile, c_l0_tile, a_l0_tile, b_l0_tile)
                                    din_offset += dilations[0]
                                pl.store(output, c_l0_tile, [batch_offset, cout_offset + n_l0_offset, dout_offset,
                                                             hout_offset + m_l0_offset // wout,
                                                             wout_offset + m_l0_offset % wout])


@pytest.mark.soc("950")
def test_conv3d_matches_golden():
    _require_a5(ST_DEVICE)

    torch.manual_seed(0)
    fmap = torch.randn(FMAP_SHAPE, device=ST_DEVICE, dtype=DTYPE_TORCH)
    weight = torch.randn(WEIGHT_SHAPE, device=ST_DEVICE, dtype=DTYPE_TORCH)
    output = torch.zeros(OUTPUT_SHAPE, device=ST_DEVICE, dtype=DTYPE_TORCH)

    block_dim = get_platform_info().cube_core_num
    conv_kernel[None, block_dim](fmap, weight, output)
    torch.npu.synchronize()

    golden = torch.conv3d(fmap, weight, bias=None, padding=PADS, stride=STRIDES,
                          dilation=DILATIONS, groups=GROUPS)
    torch.testing.assert_close(output.cpu(), golden.cpu(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_conv3d_matches_golden()
    logging.info("test_conv3d_matches_golden passed!")
