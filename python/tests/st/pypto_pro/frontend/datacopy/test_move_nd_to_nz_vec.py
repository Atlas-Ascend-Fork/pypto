# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.move`` from a row-major Vec tile into an NZ Vec tile (pto-isa ``TMovToVecNd2Nz``).

The ISA walks the source one *column group* at a time -- a group being the
``CCE_VL / sizeof(T)`` elements one vector register holds, scattered across 8 fractal panels of
the destination. Within a group it walks rows; between groups it must land on both the next
group's source column and the next group's fractal base. That between-group step is the part
with nothing to fall back on if it is wrong.

Rather than convert the result back to ND, these tests alias the destination's UB address with
a row-major tile and store *that*, so the assertion is against the physical NZ image:
``dst[panel][row][lane] == src[row][panel * c0 + lane]``. A misplaced row or column group then
shows up as specific misplaced values, which a format conversion would hide.

Covered geometry: one column group and several; a single source row; a source with fewer valid
rows than the destination (the walk must stay inside the destination's fractal); ``compact=2``
(``CompactMode::RowPlusOne``), which pads each panel by one row and so changes the
between-group step; and 1-byte and 4-byte dtypes, which change how many elements a group spans.
"""

from functools import lru_cache
import logging
import os

import numpy as np
import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

logging.basicConfig(level=logging.INFO)

_C0_BYTES = 32
_FRACTAL_NZ_ROW = 16


def _require_a5_or_a6(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if not any(soc in name for soc in ("Ascend950", "Ascend960")):
        pytest.skip(f"Current device is {name}, not A5/A6 (Ascend950/Ascend960). Skip.")


def _nz_geometry(rows, cols, itemsize, compact):
    """(c0 lanes, panels, rows per panel) of the destination's physical NZ image."""
    c0 = _C0_BYTES // itemsize
    align_row = (rows + _FRACTAL_NZ_ROW - 1) // _FRACTAL_NZ_ROW * _FRACTAL_NZ_ROW
    virtual_row = align_row + 1 if compact == 2 else align_row
    assert cols % c0 == 0, f"cols {cols} must be a multiple of c0 {c0}"
    return c0, cols // c0, virtual_row


# Retain the JIT object across discovery and execution; only static factory
# arguments enter this cache, so runtime tensor values still run independently.
@lru_cache(maxsize=None)
def _make_kernel(rows, cols, torch_dtype, compact, src_rows, flat_rows, c0, nz_bytes):
    """ND -> NZ move, read back through a row-major alias of the destination's address."""

    dtype = {torch.float16: pl.DT_FP16, torch.float32: pl.DT_FP32, torch.int8: pl.DT_INT8}[torch_dtype]

    # Concurrent specializations need distinct source and library directories.
    kernel_name = f"nd2nz_{rows}_{cols}_{dtype}_{compact}_{src_rows}_{flat_rows}_{c0}_{nz_bytes}"

    @pl.jit(auto_mutex=True, name=kernel_name)
    def kernel(x: pl.Tensor[[rows, cols], dtype], out: pl.Tensor[[flat_rows, c0], dtype]):
        nd_type = pl.TileType(
            shape=[rows, cols], dtype=dtype, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
        )
        nz_type = pl.TileType(
            shape=[rows, cols], dtype=dtype, target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
            valid_shape=[-1, -1], compact=compact,
        )
        src_group = pl.make_tile_group(type=nd_type, addrs=0x00000, mutex_ids=[0])
        dst_group = pl.make_tile_group(type=nz_type, addrs=0x20000, mutex_ids=[1])
        # Same UB address as the destination, read row-major: this is the physical NZ image.
        flat_view = pl.make_tile(
            pl.TileType(shape=[flat_rows, c0], dtype=dtype, target_memory=pl.MemorySpace.Vec),
            addr=0x20000,
        )
        with pl.section_vector():
            src = src_group.next()
            dst = dst_group.next()
            pl.load(src, x, [0, 0])
            pl.set_validshape(src, [src_rows, cols])
            pl.move(dst, src)
            # dst and flat_view alias one address but are distinct IR values, so auto_mutex
            # cannot see the dependency; make the V -> MTE3 order explicit.
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=3)
            pl.store(out, flat_view, [0, 0])

    return kernel


def _run(rows, cols, torch_dtype, compact=1, src_rows=None):
    device = ST_DEVICE
    _require_a5_or_a6(device)
    src_rows = rows if src_rows is None else src_rows

    itemsize = torch.empty((), dtype=torch_dtype).element_size()
    c0, panels, virtual_row = _nz_geometry(rows, cols, itemsize, compact)
    flat_rows = panels * virtual_row

    # Distinct values so a misplaced row or column group is a specific wrong number, not a
    # coincidental match. 251 is prime, keeping the pattern non-repeating across a panel.
    x = (torch.arange(rows * cols, device=device, dtype=torch.int32).reshape(rows, cols) % 251).to(torch_dtype)
    out = torch.zeros((flat_rows, c0), device=device, dtype=torch_dtype)

    _make_kernel(rows, cols, torch_dtype, compact, src_rows, flat_rows, c0, flat_rows * c0 * itemsize)(x, out)
    torch.npu.synchronize()

    # dst[panel][row][lane] == src[row][panel * c0 + lane], for the rows actually moved.
    got = out.cpu().to(torch.float32).numpy().reshape(panels, virtual_row, c0)[:, :src_rows, :]
    src = x[:src_rows].cpu().to(torch.float32).numpy().reshape(src_rows, panels, c0)
    expected = np.transpose(src, (1, 0, 2))
    np.testing.assert_array_equal(got, expected)
    logging.info(
        "nd->nz move [%d, %d] %s compact=%d src_rows=%d (%d groups) passed!",
        rows, cols, torch_dtype, compact, src_rows, panels,
    )


@pytest.mark.soc("950", "960")
def test_single_column_group():
    """fp16 spans 128 elements per register, so 128 columns is exactly one group."""
    _run(32, 128, torch.float16)


@pytest.mark.soc("950", "960")
def test_two_column_groups():
    """Two groups, so the step between them is exercised at all."""
    _run(32, 256, torch.float16)


@pytest.mark.soc("950", "960")
def test_four_column_groups():
    """More groups than the 8 panels one store scatters across, so the step compounds."""
    _run(16, 512, torch.float16)


@pytest.mark.soc("950", "960")
def test_single_source_row():
    """One row per group: the row walk runs once and must still land each group."""
    _run(16, 256, torch.float16, src_rows=1)


@pytest.mark.soc("950", "960")
def test_source_shorter_than_destination():
    """The destination's fractal geometry comes from its own valid rows, not the source's."""
    _run(32, 256, torch.float16, src_rows=17)


@pytest.mark.soc("950", "960")
def test_row_plus_one_compact():
    """compact=2 pads each panel by a row, changing the between-group step."""
    _run(32, 256, torch.float16, compact=2)


@pytest.mark.soc("950", "960")
def test_row_plus_one_compact_partial_rows():
    """The padded-panel step together with a short source."""
    _run(32, 256, torch.float16, compact=2, src_rows=17)


@pytest.mark.soc("950", "960")
def test_fp32_narrower_group():
    """fp32 spans 64 elements per register, so 256 columns is four groups."""
    _run(32, 256, torch.float32)


@pytest.mark.soc("950", "960")
def test_int8_byte_path():
    """1-byte dtypes are reinterpreted as uint8_t before the walk."""
    _run(32, 512, torch.int8)


@pytest.mark.soc("950", "960")
def test_int8_byte_path_partial_rows():
    """The byte path with a short source, covering its pointer handling too."""
    _run(32, 512, torch.int8, src_rows=9)
