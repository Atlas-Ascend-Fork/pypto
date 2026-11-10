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
"""Unit tests for invalid-parameter validation of pl.TileType and pl.make_tile_group.

Invalid values must be intercepted at parse time with actionable errors instead
of surfacing as opaque device/runtime failures (bad addresses, zero/negative
tile sizes, misaligned base-address expansion, ...).
"""

import os
from pathlib import Path
import re

from pypto_pro._errors import (
    InvalidArgument,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NotSupported,
)
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _parse(kernel_def) -> ir.Program:
    return kernel_def.to_kernel_def().parse_target_program(ir.SectionKind.Vector)[0]


_GOLDEN_DIR = Path(__file__).parent / "test_tile_type_validation_ir"


def _ir_golden(kernel_def, golden_name: str) -> str:
    """Whole-program IR text (memref ids normalized), compared against the .pypto golden.

    Set PYPTO_RENDER_IR=1 to (re)generate the golden files.
    """
    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(_parse(kernel_def)))
    golden_path = _GOLDEN_DIR / f"{golden_name}.pypto"
    if os.environ.get("PYPTO_RENDER_IR"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(printed.rstrip("\n") + "\n", encoding="utf-8")
        return printed
    golden = golden_path.read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
    return printed


# ---------------------------------------------------------------------------
# pl.TileType invalid parameters
# ---------------------------------------------------------------------------


def test_tile_type_shape_zero_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="shape dimensions must be positive"):
        _parse(k)


def test_tile_type_shape_negative_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[-64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="shape dimensions must be positive"):
        _parse(k)


def test_tile_type_shape_empty_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="shape must not be empty"):
        _parse(k)


def test_tile_type_bias_multi_row_rejected():
    # Bias (L0B) tiles must have exactly 1 row: [128, 128] is a hardware-invalid
    # bias shape and previously surfaced only as a F0FFFF compile error.
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    with pytest.raises(InvalidShape, match="Bias tiles must have exactly 1 row"):
        _parse(k)


def test_tile_type_bias_single_row_allowed():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_mutex_tile" in ir_str


def test_tile_type_bias_non_nd_layout_rejected():
    @pl.jit(arch="3510")
    def k(x: pl.Tensor[[1, 128], pl.DT_FP32]):
        tt = pl.TileType(
            shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias, layout=pl.DN
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    with pytest.raises(InvalidVal, match="Bias tiles require layout in \\{ND\\}"):
        _parse(k)


def test_move_mat_fp16_to_bias_allowed():
    # FP16 Mat -> FP32 Bias is the standard bias-feed path (TMovToBt half->float).
    @pl.jit
    def k(x: pl.Tensor[[1, 128], pl.DT_FP16]):
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
            addrs=0x30000, mutex_ids=[10])
        bias_l0b = pl.make_tile_group(
            type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
            addrs=0x0000, mutex_ids=[12])
        cur = bias_l1.next()
        pl.load(cur, x, [0, 0])
        pl.move(bias_l0b.current(), cur)

    ir_str = str(_parse(k))
    assert "block.move" in ir_str


def test_tile_type_shape_non_int_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64.0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="must be a compile-time integer"):
        _parse(k)


def test_tile_type_shape_bool_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[True], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="must be a compile-time integer"):
        _parse(k)


def test_tile_type_bad_dtype_direct_construction_rejected():
    # dtype=0 in kernel source resolves to DataType.BOOL (a valid enum value), so
    # the direct-construction path is the right place to probe a non-DataType dtype.
    with pytest.raises(TypeError, match=r"TileType dtype must be a pl\.DT_\* / DataType value"):
        _ = pl.TileType(shape=[64], dtype="fp16", target_memory=pl.MemorySpace.Vec)


def test_tile_type_bad_target_memory_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory="Vec")
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="target_memory expects MemorySpace"):
        _parse(k)


def test_tile_type_ddr_target_memory_rejected():
    # Tiles are compile-time on-chip buffers; DDR (off-chip GM) is accessed via
    # pl.Tensor, so a DDR TileType is rejected instead of failing at codegen.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.DDR)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(NotSupported, match="does not support MemorySpace.DDR"):
        _parse(k)


def test_tile_type_valid_shape_exceeds_shape_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[65]
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="exceeds tile shape dimension"):
        _parse(k)


def test_tile_type_valid_shape_rank_mismatch_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[64, 1]
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="must match shape rank"):
        _parse(k)


def test_tile_type_valid_shape_bad_value_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-2]
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidShape, match="positive or -1 \\(dynamic\\)"):
        _parse(k)


def test_tile_type_bad_fractal_rejected():
    # fractal is a pass-through field on every memory space (any int is accepted,
    # e.g. fractal=256 spreads into make_tile IR); only non-int values are invalid.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, fractal=True
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    # Python counts True as an int, so the gate lets it by and the builder judges it
    with pytest.raises(InvalidType, match="fractal must be a compile-time integer"):
        _parse(k)


def test_tile_type_fractal_arbitrary_int_allowed():
    # Any valid integer fractal (512/1024/32) is legal and must spread into make_tile
    # unchanged; see test_make_tile.py::test_layout_and_fractal_are_spread_from_the_tile_type.
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, fractal=512
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0, 0])

    ir_str = str(_parse(k))
    assert "fractal=512" in ir_str


def test_tile_type_compact_out_of_range_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, compact=-1
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="compact must be one of \\[0, 1, 2, 3\\]"):
        _parse(k)


# ---------------------------------------------------------------------------
# pl.make_tile_group invalid addrs / mutex_ids
# ---------------------------------------------------------------------------


def test_make_tile_group_negative_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=-0x100, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidVal, match="addrs must be at least 0"):
        _parse(k)


def test_make_tile_group_float_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0.5, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match=r"pl\.make_tile_group: addrs expects int \| list, got a fp32 scalar"):
        _parse(k)


def test_make_tile_group_float_in_addrs_list_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=[0, 0.5], mutex_ids=[0, 1])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match=r"addrs\[1\] must be a compile-time integer"):
        _parse(k)


def test_make_tile_group_bool_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=True, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match=r"addrs must be a compile-time integer"):
        _parse(k)


def test_make_tile_group_none_addrs_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=None, mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match=r"pl\.make_tile_group: addrs expects int \| list, got NoneType"):
        _parse(k)


def test_make_tile_group_str_in_addrs_list_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=["x"], mutex_ids=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match="Cannot convert .*str.* to IR expression"):
        _parse(k)


def test_make_tile_group_bool_mutex_id_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[True])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match=r"mutex_ids\[0\] must be a compile-time integer"):
        _parse(k)


def test_make_tile_group_scalar_mutex_id_int_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=5)
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match=r"pl\.make_tile_group: mutex_ids expects .*got a int64 scalar"):
        _parse(k)


def test_make_tile_group_scalar_mutex_id_float_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=0.0)
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidType, match=r"pl\.make_tile_group: mutex_ids expects .*got a fp32 scalar"):
        _parse(k)


def test_make_tile_group_unknown_kwarg_rejected():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0], foo=1)
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument 'foo'"):
        _parse(k)


def test_make_tile_group_typo_kwarg_rejected():
    # mutex_ids -> mutex_id typo must point at the unknown keyword, not just
    # report a missing 'mutex_ids'.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_id=[0])
        pl.load(g.next(), x, [0])

    with pytest.raises(InvalidArgument, match="unexpected keyword argument 'mutex_id'") as excinfo:
        _parse(k)
    # a near miss is named, so the fix does not have to be looked up
    assert "did you mean 'mutex_ids'?" in str(excinfo.value)


def test_make_tile_group_bwd_fwd_ids_ignored_for_compat():
    # bwd_ids/fwd_ids were silently ignored by the legacy parser and are kept
    # accepted for backward compatibility with existing operator repositories.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0], bwd_ids=[0], fwd_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_mutex_tile" in ir_str


# ---------------------------------------------------------------------------
# Valid inputs must keep parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valid_shape", [None, [-1, -1]])
def test_valid_dynamic_valid_shape_allowed(valid_shape):
    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=valid_shape
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        pl.load(g.next(), x, [0, 0])

    ir_str = str(_parse(k))
    assert "block.make_mutex_tile" in ir_str
    assert "tile_view<-1 x -1>" in ir_str


def test_valid_contiguous_addrs_expansion():
    @pl.jit
    def k(x: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        pl.load(db.next(), x, [0, 0])
        pl.load(db.next(), x, [0, 0])

    ir_str = str(_parse(k))
    assert "32768" in ir_str  # second slot at base + 128*128*2


def test_valid_acc_default_fractal():
    @pl.jit
    def k(x: pl.Tensor[[64, 64], pl.DT_FP32]):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_mutex_tile" in ir_str


def test_valid_fractal_512_on_left_allowed():
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, fractal=512
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_mutex_tile" in ir_str


def test_valid_compact_row_plus_one_allowed():
    # CompactMode::RowPlusOne(=2) is a valid compact mode, must not be rejected.
    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, compact=2
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = str(_parse(k))
    assert "block.make_mutex_tile" in ir_str


# ---------------------------------------------------------------------------
# pl.TileType conv layouts: multi-dim => pto::ConvTile (L1-only, layout-shaped)
# ---------------------------------------------------------------------------


def _conv_tt(**kwargs):
    return pl.TileType(**kwargs)


def test_conv_nc1hwc0_declared_and_grouped():
    """A multi-dim NC1HWC0 TileType parses and can be allocated in L1."""

    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(
            shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0,
        )
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    ir_str = _ir_golden(k, "test_conv_nc1hwc0_declared_and_grouped")
    assert "block.make_mutex_tile" in ir_str


def test_conv_layered_groups_allocated():
    """L1 ConvTile + L0A/L0B/L0C 2-D tiles allocate as separate multi-buffer groups."""

    @pl.jit(auto_mutex=True)
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        fmap = pl.TileType(shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0)
        l0a = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
        l0b = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
        l0c = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
        g1 = pl.make_tile_group(type=fmap, addrs=0x00000, depth=2, mutex_ids="auto")
        g2 = pl.make_tile_group(type=l0a, addrs=0x0000, depth=2, mutex_ids="auto")
        g3 = pl.make_tile_group(type=l0b, addrs=0x0000, depth=2, mutex_ids="auto")
        g4 = pl.make_tile_group(type=l0c, addrs=0x0000, depth=2, mutex_ids="auto")
        _ = (g1.next(), g2.next(), g3.next(), g4.next())

    ir_str = _ir_golden(k, "test_conv_layered_groups_allocated")
    # depth=2 slots per group, 4 groups.
    assert ir_str.count("block.make_mutex_tile") >= 8


@pytest.mark.parametrize(
    "shape,dtype,layout",
    [
        ([1, 4, 8, 8, 16], pl.DT_FP16, pl.Layout.NC1HWC0),
        ([2, 4, 8, 8, 32], pl.DT_INT8, pl.Layout.NC1HWC0),
        ([64, 4, 16, 16], pl.DT_FP16, pl.Layout.FRACTAL_Z),
        ([4, 8, 8, 64, 16], pl.DT_FP16, pl.Layout.FRACTAL_Z),
        ([1, 2, 4, 8, 8, 16], pl.DT_FP16, pl.Layout.NDC1HWC0),
        ([64, 4, 16, 16], pl.DT_FP16, pl.Layout.FRACTAL_Z_3D),
    ],
)
def test_conv_layout_shapes_accepted(shape, dtype, layout):
    tt = _conv_tt(shape=shape, dtype=dtype, target_memory=pl.MemorySpace.Mat, layout=layout)
    assert tt.layout == layout


def test_conv_layout_requires_l1():
    with pytest.raises(InvalidVal, match="must live in L1"):
        _conv_tt(shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Vec, layout=pl.Layout.NC1HWC0)


def test_conv_layout_rank_mismatch_rejected():
    with pytest.raises(InvalidShape, match="requires shape rank 5"):
        _conv_tt(shape=[2, 4, 8, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0)


def test_conv_layout_inner_c0_rejected():
    # fp16 C0 is 16; a last axis of 8 is not a whole 32-byte block.
    with pytest.raises(InvalidShape, match="requires its last dimension"):
        _conv_tt(shape=[1, 4, 8, 8, 8], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0)


def test_conv_fractal_z_inner_box_rejected():
    # FRACTAL_Z_3D 4D is [C1DHW, N/16, 16, C0]; the 16 is fixed.
    with pytest.raises(InvalidShape, match="requires its second-to-last dimension"):
        _conv_tt(shape=[64, 4, 8, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.FRACTAL_Z_3D)


@pytest.mark.parametrize("n", [1, 15, 20, 63])
def test_conv_fractal_z_5d_rejects_n_not_multiple_of_16(n):
    # FRACTAL_Z 5D is [C1, H, W, N, C0]. The packed layout pads N to the 16-element box, so a
    # non-multiple N would leave BufferSize = Πdims short of the box the layout occupies.
    with pytest.raises(InvalidShape, match=r"requires its N dimension to be a multiple of 16"):
        _conv_tt(shape=[4, 8, 8, n, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.FRACTAL_Z)


@pytest.mark.parametrize("n", [16, 32])
def test_conv_fractal_z_5d_accepts_aligned_n(n):
    tt = _conv_tt(shape=[4, 8, 8, n, 16], dtype=pl.DT_FP16,
                  target_memory=pl.MemorySpace.Mat, layout=pl.Layout.FRACTAL_Z)
    assert tt.layout == pl.Layout.FRACTAL_Z


def test_multi_dim_without_conv_layout_rejected():
    """A multi-dim tile without a conv layout is rejected by the IR rank-2 gate."""

    @pl.jit
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        # Layout-less Vec tile: passes the 2-D checks, so the IR rank-2 gate rejects it.
        tt = pl.TileType(shape=[1, 64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0])
        _ = g.next()

    with pytest.raises(NotSupported, match="TileType only supports rank-2 shape"):
        _parse(k)


def test_conv_layout_dtype_not_supported_rejected():
    with pytest.raises(InvalidType, match="only support dtypes"):
        _conv_tt(shape=[1, 4, 8, 8, 8], dtype=pl.DT_INT4,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0)


def test_conv_layout_rejects_fractal_argument():
    with pytest.raises(InvalidVal, match="drop the 'fractal' argument"):
        _conv_tt(shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0, fractal=512)


def test_conv_layout_rejects_valid_shape_argument():
    with pytest.raises(InvalidVal, match="drop the 'valid_shape' argument"):
        _conv_tt(shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0,
                 valid_shape=[1, 4, 8, 8, 16])


def test_conv_layout_rejects_pad_argument():
    with pytest.raises(InvalidVal, match="drop the 'pad' argument"):
        _conv_tt(shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0, pad=1)


def test_conv_layout_rejects_non_normal_compact():
    with pytest.raises(InvalidVal, match="drop the 'compact' argument"):
        _conv_tt(shape=[1, 4, 8, 8, 16], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Mat, layout=pl.Layout.NC1HWC0, compact=2)
