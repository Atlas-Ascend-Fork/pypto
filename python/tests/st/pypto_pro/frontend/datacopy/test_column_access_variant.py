# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

"""One tensor read with both a full window and a single-column window.

A tile whose last dim is 1 is emitted ``BLayout::ColMajor``, and the ISA requires the
GlobalTensor's layout to equal the tile's (``TLOAD(VecTile, GlobalTensor)`` only supports
ND2ND/DN2DN/NZ2NZ). So a single-column access needs a ``Layout::DN`` declaration while an
ordinary access of the *same tensor* needs ``Layout::ND`` -- two declarations, and each
access has to go through its own.

That is what ``IsColumnAccess`` is for: it is the only caller-visible input that lets
``TensorLayoutVariantKey`` tell those two accesses apart while the prescan is still working
from the op alone. Drop it and both accesses key the same, share one declaration, and one of
them is handed the wrong ``Layout`` -- which the ISA rejects with
"Src and dst layout must be same!", so this test fails at kernel compile time.

Note what a DN read actually transfers: ``TLoadGm2ubDn2dn`` sets nBurst=1 and
lenBurst=validRow, i.e. ROWS *contiguous* elements, not a strided column. Reading a real
column is what ``order=[1, 0]`` is for (see test_load_order_variants.py).
"""

import glob
import logging
import os
import re

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

logging.basicConfig(level=logging.INFO)

ROWS = 64
COLS = 64


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


@pl.jit(auto_mutex=True)
def _column_and_full_kernel(
    x: pl.Tensor[[ROWS, COLS], pl.DT_FP32],
    full_out: pl.Tensor[[ROWS, COLS], pl.DT_FP32],
    column_out: pl.Tensor[[ROWS, 1], pl.DT_FP32],
):
    # Same dtype, same tensor, same offsets: the *only* difference is the window's last dim,
    # which is what decides ND vs DN. No order kwarg anywhere.
    full_type = pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    column_type = pl.TileType(shape=[ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    full_group = pl.make_tile_group(type=full_type, addrs=0x00000, mutex_ids=[0])
    column_group = pl.make_tile_group(type=column_type, addrs=0x10000, mutex_ids=[1])

    with pl.section_vector():
        full_tile = full_group.next()
        column_tile = column_group.next()
        pl.load(full_tile, x, [0, 0])
        pl.load(column_tile, x, [0, 0])
        pl.store(full_out, full_tile, [0, 0])
        pl.store(column_out, column_tile, [0, 0])


@pytest.mark.soc("950")
def test_column_and_full_access_of_one_tensor():
    """Both windows read correctly, so each went through the declaration matching its layout."""
    device = ST_DEVICE
    _require_a5(device)

    # arange makes every element distinguishable, so a wrong layout cannot coincidentally pass.
    x = torch.arange(ROWS * COLS, device=device, dtype=torch.float32).reshape(ROWS, COLS)
    full_out = torch.zeros(ROWS, COLS, device=device, dtype=torch.float32)
    column_out = torch.zeros(ROWS, 1, device=device, dtype=torch.float32)

    _column_and_full_kernel[None, 1](x, full_out, column_out)
    torch.npu.synchronize()

    torch.testing.assert_close(full_out, x)
    # A DN transfer moves ROWS contiguous elements from the access offset, so the column tile
    # holds x's first ROWS elements in memory order -- not x[:, 0].
    torch.testing.assert_close(column_out.flatten(), x.flatten()[:ROWS])
    assert not torch.equal(x.flatten()[:ROWS], x[:, 0]), "inputs must differ for the check to bite"
    logging.info("test_column_and_full_access_of_one_tensor [%d, %d] passed!", ROWS, COLS)


@pl.jit(auto_mutex=True)
def _compact_column_kernel(
    x: pl.Tensor[[ROWS, COLS], pl.DT_FP32],
    column_out: pl.Tensor[[ROWS, 1], pl.DT_FP32],
):
    # compact=1 on a single-column tile used to be dropped on the floor: the shape rule that
    # forces ColMajor replaced the tile's whole hw_info instead of just its block layout. It now
    # survives into CompactMode::Normal, which TSTORE and TMOV branch on -- so run it for real.
    column_type = pl.TileType(
        shape=[ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, compact=1
    )
    column_group = pl.make_tile_group(type=column_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        column_tile = column_group.next()
        pl.load(column_tile, x, [0, 0])
        pl.store(column_out, column_tile, [0, 0])


@pytest.mark.soc("950")
def test_compact_column_tile_is_emitted_and_round_trips():
    """compact=1 survives onto a column tile, and the kernel still runs correctly.

    The numeric half alone cannot catch a dropped ``compact``: CompactMode::Normal only changes
    the fractal paths (``TStoreUb2gm``'s srcStride, ``TExtractToACompact`` in TMOV), and this
    tile is SLayout::NoneBox, so results are identical either way. The emitted tile type is
    therefore asserted directly -- that is the part that regressed before.
    """
    device = ST_DEVICE
    _require_a5(device)

    x = torch.arange(ROWS * COLS, device=device, dtype=torch.float32).reshape(ROWS, COLS)
    column_out = torch.zeros(ROWS, 1, device=device, dtype=torch.float32)

    _compact_column_kernel[None, 1](x, column_out)
    torch.npu.synchronize()

    torch.testing.assert_close(column_out.flatten(), x.flatten()[:ROWS])

    # Read back the kernel this run actually compiled, rather than re-deriving it.
    generated = glob.glob(
        os.path.join("build", "*_compact_column_kernel__*", "**", "kernel.cpp"), recursive=True
    )
    assert generated, "no generated kernel.cpp found for the compact column kernel"
    source = max(generated, key=os.path.getmtime)
    with open(source, encoding="utf-8") as handle:
        column_type = re.search(
            r"using _tg_column_group_tiles_0_0_Type = SyncedTile<(Tile<[^;]*>), [0-9]+>;", handle.read()
        )
    assert column_type is not None, f"no tile type alias for the column tile in {source}"
    assert "BLayout::ColMajor" in column_type.group(1), column_type.group(1)
    assert "CompactMode::Normal" in column_type.group(1), column_type.group(1)
    logging.info("test_compact_column_tile_is_emitted_and_round_trips [%d, 1] passed!", ROWS)
