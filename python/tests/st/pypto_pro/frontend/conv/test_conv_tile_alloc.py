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

"""ConvTile declaration and allocation smoke test on A5.

A conv-layout ``pl.TileType`` (NC1HWC0 / FRACTAL_Z here) lowers to a pto::ConvTile in L1
instead of the 2-D pto::Tile; a ConvTile's shape is carried by ``ConvTileShape<...>`` and it
has no ``(rows, cols)`` constructor, so the emitted declaration must not pass the 2-D Tile
ctor arguments (pto static_asserts on the ctor values). This ST is the on-device counterpart
of ``ut/pypto_pro/codegen/test_cce_conv_tile.py``: declaring and allocating a ConvTile must
survive bisheng compilation and launch.

The interface currently only provides ConvTile declaration and allocation -- there is no
conv data path yet (GM -> ConvTile loads and im2col folding are not implemented), so the
ConvTile cannot be exercised numerically. The kernel therefore fills an output tensor with a
constant to prove the launched kernel really executed; the ConvTile groups are allocated in
the same kernel, so a broken ConvTile declaration fails the compile/launch.

Doc: docs/zh/api/pro_api/SIMD-API/basic_data_structures/TileType.md
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the npu backend

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# NC1HWC0 = [N, C1, H, W, C0]; FP16 => C0 = 32B / 2B = 16.
FMAP_SHAPE = [1, 4, 8, 8, 16]
# FRACTAL_Z = [C1HW, N/16, 16, C0].
WGT_SHAPE = [64, 4, 16, 16]
OUT_NUMEL = 64
FILL_VALUE = 7.0


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def conv_tile_alloc_kernel(out: pl.Tensor[[OUT_NUMEL], pl.DT_FP16]):
    fmap_type = pl.TileType(
        shape=FMAP_SHAPE, dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0
    )
    wgt_type = pl.TileType(
        shape=WGT_SHAPE, dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.Layout.FRACTAL_Z
    )

    # Both are L1 (Mat) conv tiles; addrs are byte offsets in L1 and "auto" pairs the
    # mutex ids of the double buffer's two slots.
    fmap_db = pl.make_tile_group(type=fmap_type, addrs=0x00000, depth=2, mutex_ids="auto")
    wgt_db = pl.make_tile_group(type=wgt_type, addrs=0x20000, depth=2, mutex_ids="auto")

    with pl.section_cube():
        _ = fmap_db.next()
        _ = wgt_db.next()

    with pl.section_vector():
        pl.init_output(out, offset=0, size=OUT_NUMEL, value=FILL_VALUE)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_conv_tile_declaration_and_allocation_runs():
    _require_a5(ST_DEVICE)

    out = torch.full([OUT_NUMEL], -1.0, device=ST_DEVICE, dtype=torch.float16)

    conv_tile_alloc_kernel[None, 1](out)
    torch.npu.synchronize()

    expected = torch.full([OUT_NUMEL], FILL_VALUE, dtype=torch.float16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_conv_tile_declaration_and_allocation_runs()
    logging.info("test_conv_tile_declaration_and_allocation_runs passed!")
