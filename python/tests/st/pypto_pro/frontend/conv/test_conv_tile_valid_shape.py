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

"""ConvTile valid_shape declaration and runtime set_validshape smoke test on A5.

A conv-layout ``pl.TileType`` with ``valid_shape`` lowers to the optional 6th ``ValidShape_``
template parameter of pto's ``ConvTile``; ``pl.set_validshape`` on the tile lowers to
``ConvTile::SetValidShape(v0, ..., vN)`` (one value per shape dim, matching the DYNAMIC-dim
count). This ST is the on-device counterpart of ``ut/pypto_pro/codegen/test_cce_conv_tile.py``:
declaring the valid shape and setting it at runtime must survive bisheng compilation and launch
(the environment's pto headers must carry the six-parameter ConvTile template).

There is still no conv data path (GM -> ConvTile loads and im2col folding are not
implemented), so the valid extents are metadata only; the kernel fills an output tensor with
a constant to prove the launched kernel really executed, with the ConvTile groups allocated
and set in the same kernel.

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
def conv_tile_valid_shape_kernel(
    out: pl.Tensor[[OUT_NUMEL], pl.DT_FP16],
    valid_h: pl.DT_INT64,
    valid_w: pl.DT_INT64,
):
    # All-dynamic declaration: every valid dim is runtime-settable; the IR constructor
    # normalizes an omitted valid_shape to the same all--1 form.
    fmap_type = pl.TileType(
        shape=FMAP_SHAPE,
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.Layout.NC1HWC0,
        valid_shape=[-1, -1, -1, -1, -1],
    )
    # Mixed declaration: static dims keep their compile-time value; this tile is not
    # runtime-settable, which the frontend enforces (only emitted here for the golden form).
    wgt_type = pl.TileType(
        shape=WGT_SHAPE,
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.Layout.FRACTAL_Z,
        valid_shape=[-1, -1, 16, 16],
    )

    fmap_db = pl.make_tile_group(type=fmap_type, addrs=0x00000, depth=2, mutex_ids="auto")
    wgt_db = pl.make_tile_group(type=wgt_type, addrs=0x20000, depth=2, mutex_ids="auto")

    with pl.section_cube():
        fmap = fmap_db.next()
        _ = wgt_db.next()
        # One runtime value per shape dim (N, C1, H, W, C0); H and W come from launch args.
        pl.set_validshape(fmap, [1, 4, valid_h, valid_w, 16])

    with pl.section_vector():
        pl.init_output(out, offset=0, size=OUT_NUMEL, value=FILL_VALUE)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_conv_tile_valid_shape_set_and_runs():
    _require_a5(ST_DEVICE)

    out = torch.full([OUT_NUMEL], -1.0, device=ST_DEVICE, dtype=torch.float16)

    conv_tile_valid_shape_kernel[None, 1](out, 8, 8)
    torch.npu.synchronize()

    expected = torch.full([OUT_NUMEL], FILL_VALUE, dtype=torch.float16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_conv_tile_valid_shape_set_and_runs()
    logging.info("test_conv_tile_valid_shape_set_and_runs passed!")
