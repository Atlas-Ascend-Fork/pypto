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

"""CPU replay regressions do not require an NPU or a SoC marker."""

import struct

import pypto_pro.runtime.sanitizer_replay as replay_module
from pypto_pro.runtime.sanitizer_replay import (
    Detection,
    Finding,
    PhaseAccessRecord,
    ReplayContext,
    TileAccessRecord,
    check_phase_state_machine,
    check_tile_bounds,
    run_replay,
)
import pytest

_CTX = ReplayContext("ut.py")


def test_phase_insert_final_read_records():
    records = [PhaseAccessRecord(op_kind=kind, phase=phase, acc_addr=0, line=10)
               for kind, phase in ((1, 1), (2, 2), (10, 2))]
    assert check_phase_state_machine(records, _CTX) == []


@pytest.mark.parametrize("previous_phase", [None, 0, 2], ids=["initial", "after-software", "after-hardware"])
def test_phase_unpaired_read_does_not_set_next_round_mode(previous_phase):
    records = []
    if previous_phase is not None:
        records.extend([
            PhaseAccessRecord(op_kind=1, phase=previous_phase, acc_addr=0x100, line=10),
            PhaseAccessRecord(op_kind=8, phase=previous_phase, acc_addr=0x100, line=11),
        ])
    records.extend([
        PhaseAccessRecord(op_kind=6, phase=0, acc_addr=0x100, line=12),
        PhaseAccessRecord(op_kind=1, phase=2, acc_addr=0x100, line=20),
        PhaseAccessRecord(op_kind=8, phase=2, acc_addr=0x100, line=21),
    ])
    assert check_phase_state_machine(records, _CTX) == []


def test_phase_mixing_reports_each_offending_location():
    records = [
        PhaseAccessRecord(op_kind=kind, phase=phase, acc_addr=0, line=line)
        for kind, phase, line in (
            (1, 0, 10), (2, 1, 11), (2, 2, 12),
            (6, 0, 13), (10, 0, 14), (8, 2, 15),
        )
    ]
    findings = check_phase_state_machine(records, _CTX)
    assert [f.location for f in findings] == ["ut.py:11", "ut.py:12", "ut.py:13", "ut.py:14"]
    assert all(f.kind == "PHASE_MISMATCH" for f in findings)
    assert all("matmul ops mix" in f.message for f in findings[:2])
    assert all("store/move" in f.message for f in findings[2:])


@pytest.mark.parametrize("next_kind,expected_detail", [
    (1, "matmul appears mid-chain"),
    (3, "matmul_bias appears mid-chain"),
    (4, "matmul appears mid-chain"),
    (2, None),
    (5, None),
], ids=["overwrite", "bias", "mx-overwrite", "accumulate", "mx-accumulate"])
def test_phase_unspecified_read_keeps_kchain_roles(next_kind, expected_detail):
    first_kind = 4 if next_kind in (4, 5) else 1
    records = [
        PhaseAccessRecord(op_kind=kind, phase=phase, acc_addr=0, line=line)
        for kind, phase, line in (
            (first_kind, 1, 10), (6, 0, 11), (next_kind, 2, 12), (8, 2, 13),
        )
    ]
    findings = check_phase_state_machine(records, _CTX)
    expected_locations = ["ut.py:11", "ut.py:12"] if expected_detail else ["ut.py:11"]
    assert [f.location for f in findings] == expected_locations
    assert all(f.kind == "PHASE_MISMATCH" for f in findings)
    assert "store/move" in findings[0].message
    if expected_detail:
        assert expected_detail in findings[1].message


@pytest.mark.parametrize("write_phases", [(0, 1), (1, 0)], ids=["software-first", "hardware-first"])
def test_phase_unspecified_read_keeps_mixed_round(write_phases):
    records = [
        PhaseAccessRecord(op_kind=kind, phase=phase, acc_addr=0, line=line)
        for kind, phase, line in (
            (1, write_phases[0], 10), (2, write_phases[1], 11),
            (6, 0, 12), (1, 2, 13), (8, 2, 14),
        )
    ]
    findings = check_phase_state_machine(records, _CTX)
    assert [f.location for f in findings] == ["ut.py:11", "ut.py:12", "ut.py:13", "ut.py:13"]
    assert all(f.kind == "PHASE_MISMATCH" for f in findings)
    assert "matmul ops mix" in findings[0].message
    assert "store/move" in findings[1].message
    assert "matmul ops mix" in findings[2].message
    assert "matmul appears mid-chain" in findings[3].message


def test_phase_invalid_reads_preserve_round_until_final_consumer():
    records = [
        PhaseAccessRecord(op_kind=kind, phase=phase, acc_addr=0, line=line)
        for kind, phase, line in (
            (1, 1, 10), (6, 0, 11), (8, 0, 12), (10, 0, 13),
            (2, 2, 14), (8, 2, 15), (1, 0, 16), (6, 0, 17),
            (1, 2, 18), (8, 2, 19),
        )
    ]
    findings = check_phase_state_machine(records, _CTX)
    assert [f.location for f in findings] == ["ut.py:11", "ut.py:12", "ut.py:13"]
    assert all(f.kind == "PHASE_MISMATCH" and "store/move" in f.message for f in findings)


def test_replay_keeps_distinct_findings_at_same_location(monkeypatch):
    """同一行的不同诊断都保留，完全重复项只折叠 hits。"""
    class _Record:
        pass

    first = Finding("PHASE_MISMATCH", "mode mismatch", "ut.py:10", "fix mode")
    second = Finding("PHASE_MISMATCH", "contract mismatch", "ut.py:10", "fix tile")
    duplicate = Finding("PHASE_MISMATCH", "mode mismatch", "ut.py:10", "fix mode")
    monkeypatch.setattr(
        replay_module,
        "DETECTIONS",
        [Detection(_Record, "ERROR", lambda records, ctx:[first, second, duplicate])],
    )

    findings = run_replay([_Record()], _CTX)
    assert len(findings) == 2
    assert any(
        f.message == "mode mismatch  (same defect hit 2 times at this location)" and f.hits == 2
        for f in findings
    )
    assert any(f.message == "contract mismatch" and f.hits == 1 for f in findings)


def _acc_record(kind=2, phase=0, **changes):
    record = PhaseAccessRecord(
        op_kind=kind, phase=phase, acc_addr=0, acc_src_addr=0,
        acc_valid_rows=32, acc_valid_cols=64,
        acc_src_valid_rows=32, acc_src_valid_cols=64,
        line=20,
    )
    return record._replace(**changes)


def _acc_chain(record):
    initial = PhaseAccessRecord(
        op_kind=1, phase=0 if record.phase == 0 else 1,
        acc_addr=record.acc_addr, line=10, region=record.region,
    )
    records = [initial, record]
    if record.phase == 2:
        records.append(PhaseAccessRecord(
            op_kind=6, phase=2, acc_addr=record.acc_addr, line=30, region=record.region,
        ))
    return records


@pytest.mark.parametrize("kind", [2, 5], ids=["matmul_acc", "matmul_mx_acc"])
@pytest.mark.parametrize("phase", [0, 1, 2], ids=["software", "partial", "final"])
@pytest.mark.parametrize("changes,expected_detail", [
    ({"acc_src_addr": 4096}, "address: dst=0x0, acc=0x1000"),
    ({"acc_src_valid_rows": 16}, "valid_shape: dst=(32, 64), acc=(16, 64)"),
    ({"acc_src_valid_cols": 32}, "valid_shape: dst=(32, 64), acc=(32, 32)"),
    ({"acc_src_valid_rows": 64, "acc_src_valid_cols": 32}, "valid_shape: dst=(32, 64), acc=(64, 32)"),
    ({"acc_src_addr": 4096, "acc_src_valid_rows": 16},
     "both address and valid_shape: address dst=0x0, acc=0x1000; valid_shape dst=(32, 64), acc=(16, 64)"),
])
def test_acc_identity_rejects_address_or_valid_shape_mismatch(kind, phase, changes, expected_detail):
    record = _acc_record(kind, phase, **changes)
    findings = check_phase_state_machine(_acc_chain(record), _CTX)
    assert len(findings) == 1
    assert findings[0].kind == "PHASE_MISMATCH"
    assert findings[0].location == "ut.py:20"
    assert findings[0].message == f"region 0, Acc tile 0x0: dst_tile and acc_tile differ in {expected_detail}"


@pytest.mark.parametrize("kind", [2, 5])
@pytest.mark.parametrize("phase", [0, 1, 2])
def test_acc_identity_accepts_matching_runtime_view(kind, phase):
    # An edge tile's valid_shape may be smaller than its allocation shape.
    assert check_phase_state_machine(_acc_chain(_acc_record(kind, phase)), _CTX) == []


def test_acc_identity_reads_each_operations_valid_shape():
    first = _acc_record(phase=1)
    second = _acc_record(phase=2, acc_valid_rows=16, acc_src_valid_rows=16, line=21)
    records = [*_acc_chain(first), second,
               PhaseAccessRecord(op_kind=6, phase=2, acc_addr=0, line=30)]
    assert check_phase_state_machine(records, _CTX) == []
    records[2] = second._replace(acc_src_valid_rows=32)
    findings = check_phase_state_machine(records, _CTX)
    assert len(findings) == 1
    assert findings[0].location == "ut.py:21"
    assert "valid_shape" in findings[0].message


def test_acc_identity_check_survives_blocked_timeline():
    records = [PhaseAccessRecord(op_kind=6, phase=2, acc_addr=0, line=10),
               _acc_record(phase=2, acc_src_valid_cols=32)]
    findings = check_phase_state_machine(records, _CTX)
    assert any(f.kind == "PHASE_DEADLOCK" for f in findings)
    assert any(f.kind == "PHASE_MISMATCH" and "valid_shape" in f.message for f in findings)


def test_phase_identity_fields_keep_constructor_defaults():
    record = PhaseAccessRecord(op_kind=1, phase=0, acc_addr=0)
    assert record.acc_src_addr == -1
    assert record.acc_valid_rows == record.acc_src_valid_cols == -1
    assert record.line == record.region == 0
    assert PhaseAccessRecord._field_defaults == {
        "acc_src_addr": -1, "acc_valid_rows": -1, "acc_valid_cols": -1,
        "acc_src_valid_rows": -1, "acc_src_valid_cols": -1, "line": 0, "region": 0}
    assert replay_module.GmAccessRecord._field_defaults == {"region": 0}
    assert check_phase_state_machine([record], _CTX) == []


def test_phase_identity_wire_round_trip(tmp_path):
    # Explicit field order checks the shared C++ schema as well as replay.
    words = [
        (replay_module.RECORD_MAGIC << 16) | 7, 2, 0, 0, 0,
        32, 64, 32, 32, 20,
    ]
    payload = struct.pack("<III" + "i" * 6 + "I", *words)
    chunk = struct.pack("<I", len(words)) + payload
    decoded = replay_module.decode_regions([struct.pack("<I", 0), chunk])
    assert decoded == [_acc_record(acc_src_valid_cols=32, region=1)]
    path = tmp_path / "same_tile.bin"
    replay_module.build_report_file([chunk], str(path))
    findings = replay_module.scan_report_file(str(path), "ut.py")
    assert len(findings) == 1
    assert findings[0].kind == "PHASE_MISMATCH"
    assert "valid_shape" in findings[0].message


def _slice_rec(off_row, off_col, acc_row, acc_col, dim_row=64, dim_col=64,
               layout_kind=2, inner_rows=16, inner_cols=16, base_addr=-1,
               shape_row=64, shape_col=64, dtype_bits=16):
    return TileAccessRecord(dim_row=dim_row, dim_col=dim_col,
                            off_row=off_row, off_col=off_col,
                            acc_row=acc_row, acc_col=acc_col,
                            layout_kind=layout_kind,
                            inner_rows=inner_rows, inner_cols=inner_cols,
                            base_addr=base_addr, shape_row=shape_row, shape_col=shape_col,
                            dtype_bits=dtype_bits, line=20)


def test_slice_alignment_records():
    """纯逻辑测试：L1 分形切片的动态对齐检测（TILE_SLICE_UNALIGNED）。"""
    ctx = ReplayContext(source_file="ut.py")

    # 对齐且在界内：无 finding
    assert check_tile_bounds([_slice_rec(16, 0, 32, 32)], ctx) == []

    # 动态未对齐 start（界内）：TILE_SLICE_UNALIGNED
    hits = check_tile_bounds([_slice_rec(8, 0, 32, 32)], ctx)
    assert len(hits) == 1
    assert hits[0].kind == "TILE_SLICE_UNALIGNED"
    assert "[8, 0] is unaligned to the [16, 16] fractal inner block" in hits[0].message

    # ZN 布局（INT8 inner=(32,16)）：行方向未对齐（C0 方向）
    hits = check_tile_bounds(
        [_slice_rec(16, 0, 32, 32, layout_kind=3, inner_rows=32, inner_cols=16)], ctx)
    assert len(hits) == 1
    assert hits[0].kind == "TILE_SLICE_UNALIGNED"
    assert "[16, 0] is unaligned to the [32, 16]" in hits[0].message

    # ZN 布局对齐通过
    assert check_tile_bounds(
        [_slice_rec(32, 0, 32, 32, layout_kind=3, inner_rows=32, inner_cols=16)], ctx) == []

    # 列方向未对齐
    hits = check_tile_bounds([_slice_rec(0, 8, 32, 32)], ctx)
    assert len(hits) == 1
    assert hits[0].kind == "TILE_SLICE_UNALIGNED"
    assert "[0, 8] is unaligned to the [16, 16]" in hits[0].message

    # 缺少 ND/DN 首地址字段（-1）的旧式构造不做地址对齐检查
    assert check_tile_bounds([_slice_rec(8, 0, 32, 32, layout_kind=0, inner_rows=0, inner_cols=0)], ctx) == []

    # 旧式构造（不填新字段，缺省 -1）：行为与现在完全一致
    legacy = TileAccessRecord(dim_row=64, dim_col=64, off_row=8, off_col=0, acc_row=32, acc_col=32, line=0)
    assert check_tile_bounds([legacy], ctx) == []

    # 起点对齐通过后，off+win 超维时报 TILE_OUT_OF_BOUNDS
    hits = check_tile_bounds([_slice_rec(48, 0, 32, 32)], ctx)
    assert len(hits) == 1
    assert hits[0].kind == "TILE_OUT_OF_BOUNDS"

    # move/insert 的 layout 哨兵为 -1，不触发对齐检查
    assert check_tile_bounds([_slice_rec(8, 0, 32, 32, layout_kind=-1)], ctx) == []
    move_rec = TileAccessRecord(dim_row=64, dim_col=64, off_row=8, off_col=0,
                                acc_row=32, acc_col=32,
                                layout_kind=-1, inner_rows=-1, inner_cols=-1, line=20)
    # Encode raw device words. The final namedtuple field is the host-only region.
    words = [(replay_module.RECORD_MAGIC << 16) | 2, *move_rec[:-1]]
    chunk = struct.pack(f"<{len(words) + 1}I", len(words), *(value & 0xFFFFFFFF for value in words))
    decoded = replay_module.decode_regions([chunk])
    assert decoded == [move_rec]
    assert check_tile_bounds(decoded, ctx) == []


def test_tile_access_fields_keep_constructor_defaults():
    assert TileAccessRecord._field_defaults == {
        "layout_kind": -1, "inner_rows": -1, "inner_cols": -1, "base_addr": -1,
        "shape_row": -1, "shape_col": -1, "dtype_bits": -1, "line": 0, "region": 0}
    record = TileAccessRecord(dim_row=64, dim_col=64, off_row=0, off_col=0, acc_row=32, acc_col=32)
    assert record.layout_kind == record.inner_rows == record.inner_cols == -1
    assert record.base_addr == record.shape_row == record.shape_col == record.dtype_bits == -1
    assert record.line == record.region == 0


def test_slice_bounds_use_runtime_valid_shape():
    """Slice records carry the parent's runtime valid_shape as the bound, not declared shape."""
    ctx = ReplayContext(source_file="ut.py")

    # valid_shape (32,64) is the bound: start 24 + size 16 = 40 > 32 → OOB on dim0
    rec = TileAccessRecord(dim_row=32, dim_col=64, off_row=24, off_col=0,
                           acc_row=16, acc_col=32, layout_kind=0,
                           inner_rows=0, inner_cols=0, line=20)
    hits = check_tile_bounds([rec], ctx)
    assert len(hits) == 1
    assert hits[0].kind == "TILE_OUT_OF_BOUNDS"
    assert "dim0: offset 24 + valid_shape 16 = 40 > 32" in hits[0].message

    # start 16 + size 16 = 32 == valid_shape 32 → in bounds
    rec2 = TileAccessRecord(dim_row=32, dim_col=64, off_row=16, off_col=0,
                            acc_row=16, acc_col=32, layout_kind=0,
                            inner_rows=0, inner_cols=0, line=20)
    assert check_tile_bounds([rec2], ctx) == []

    # declared shape is 64x64 but the record's dim is valid_shape 32x64
    # start 40 + size 16 = 56 < 64 (declared) but > 32 (valid) → OOB
    rec3 = TileAccessRecord(dim_row=32, dim_col=64, off_row=40, off_col=0,
                            acc_row=16, acc_col=32, layout_kind=0,
                            inner_rows=0, inner_cols=0, line=20)
    hits = check_tile_bounds([rec3], ctx)
    assert len(hits) == 1
    assert hits[0].kind == "TILE_OUT_OF_BOUNDS"


@pytest.mark.parametrize("layout", [0, 1, 2, 3])
@pytest.mark.parametrize("dims", [(0, 64), (64, 0), (0, 0)])
def test_l1_slice_zero_valid_dimension_is_a_known_empty_bound(layout, dims):
    record = _slice_rec(0, 0, 16, 16, dim_row=dims[0], dim_col=dims[1],
                        layout_kind=layout, base_addr=0)
    findings = check_tile_bounds([record], _CTX)
    assert [finding.kind for finding in findings] == ["TILE_OUT_OF_BOUNDS"]
    for axis, dim in enumerate(dims):
        assert (f"dim{axis}:" in findings[0].message) == (dim == 0)


@pytest.mark.parametrize("layout", [0, 1, 2, 3])
def test_l1_slice_alignment_precedes_zero_bound(layout):
    record = _slice_rec(1, 1, 16, 16, dim_row=0, layout_kind=layout, base_addr=0, dtype_bits=4)
    findings = check_tile_bounds([record], _CTX)
    assert [finding.kind for finding in findings] == ["TILE_SLICE_UNALIGNED"]


@pytest.mark.parametrize("dims", [(0, 64), (64, 0), (0, 0)])
def test_l1_slice_empty_window_fits_empty_bound(dims):
    record = _slice_rec(0, 0, *dims, dim_row=dims[0], dim_col=dims[1],
                        layout_kind=0, base_addr=0)
    assert check_tile_bounds([record], _CTX) == []


def test_non_slice_minus_one_bound_means_unknown():
    record = _slice_rec(0, 0, 16, 16, dim_row=-1, dim_col=-1, layout_kind=-1)
    assert check_tile_bounds([record], _CTX) == []


def test_non_slice_zero_bound_is_a_real_empty_dimension():
    record = _slice_rec(0, 0, 16, 16, dim_row=0, dim_col=64, layout_kind=-1)
    findings = check_tile_bounds([record], _CTX)
    assert [finding.kind for finding in findings] == ["TILE_OUT_OF_BOUNDS"]
    assert "dim0:" in findings[0].message
    assert "dim1:" not in findings[0].message


@pytest.mark.parametrize("layout,offset", [(0, (33554432, 0)), (1, (0, 33554432))])
def test_extreme_slice_offset_is_computed_without_integer_overflow(layout, offset):
    record = _slice_rec(*offset, 16, 16, layout_kind=layout, base_addr=0)
    findings = check_tile_bounds([record], _CTX)
    assert [finding.kind for finding in findings] == ["TILE_OUT_OF_BOUNDS"]
    assert "33554432" in findings[0].message


@pytest.mark.parametrize("layout,offset,base_addr,shape,dtype_bits,start_bits,expected", [
    (0, (0, 0), 0, (64, 64), 16, 0, None),
    (0, (0, 16), 0, (64, 64), 16, 256, None),
    (0, (0, 1), 0, (64, 64), 16, 16, "TILE_SLICE_UNALIGNED"),
    (1, (16, 0), 0, (64, 64), 16, 256, None),
    (1, (1, 0), 0, (64, 64), 16, 16, "TILE_SLICE_UNALIGNED"),
    # The parent address participates even when the relative offset is aligned.
    (0, (0, 0), 16, (64, 64), 16, 128, "TILE_SLICE_UNALIGNED"),
    (1, (0, 0), 16, (64, 64), 16, 128, "TILE_SLICE_UNALIGNED"),
    # FP4 coordinate 1 remains half a byte.
    (0, (0, 1), 0, (64, 64), 4, 4, "TILE_SLICE_UNALIGNED"),
    (1, (1, 0), 0, (64, 64), 4, 4, "TILE_SLICE_UNALIGNED"),
    # Alignment is checked before bounds, including negative coordinates.
    (0, (-1, 0), 0, (64, 64), 16, -1024, "TILE_OUT_OF_BOUNDS"),
    (0, (0, -1), 0, (64, 64), 16, -16, "TILE_SLICE_UNALIGNED"),
    (1, (60, 0), 0, (64, 64), 16, 960, "TILE_SLICE_UNALIGNED"),
    # Physical shape determines strides even when valid_shape is only 31 x 31.
    (0, (1, 0), 0, (64, 64), 16, 1024, None),
    (1, (0, 1), 0, (64, 64), 16, 1024, None),
    # An ND single-column Tile uses PTO's column-major stride for its invalid column.
    (0, (0, 1), 0, (64, 1), 16, 1024, "TILE_OUT_OF_BOUNDS"),
])
def test_plain_l1_slice_uses_exact_bit_address(layout, offset, base_addr, shape, dtype_bits,
                                              start_bits, expected):
    record = _slice_rec(*offset, 8, 1 if shape[1] == 1 else 8,
                        dim_row=31, dim_col=1 if shape[1] == 1 else 31,
                        layout_kind=layout, inner_rows=0, inner_cols=0,
                        base_addr=base_addr, shape_row=shape[0], shape_col=shape[1], dtype_bits=dtype_bits)
    findings = check_tile_bounds([record], _CTX)
    assert [finding.kind for finding in findings] == ([] if expected is None else [expected])
    if expected == "TILE_SLICE_UNALIGNED":
        assert "32-byte (256-bit) aligned" in findings[0].message
        assert str(start_bits) in findings[0].message
        assert findings[0].location == _CTX.location(20)


@pytest.mark.parametrize("layout,base_addr", [(0, 4), (1, 4), (2, -1), (3, -1)])
def test_slice_address_wire_round_trip(layout, base_addr):
    words = [(replay_module.RECORD_MAGIC << 16) | 2,
             64, 64, 0, 0, 16, 16, layout, 16, 16, base_addr, 64, 64, 16, 20]
    payload = struct.pack("<I" + "i" * 13 + "I", *words)
    chunk = struct.pack("<I", len(words)) + payload
    decoded = replay_module.decode_regions([chunk])
    expected = _slice_rec(0, 0, 16, 16, layout_kind=layout, base_addr=base_addr)
    assert decoded == [expected]
    assert check_tile_bounds(decoded, _CTX) == check_tile_bounds([expected], _CTX)


def test_transfer_records_do_not_apply_slice_address_alignment():
    record = _slice_rec(0, 1, 16, 16, layout_kind=-1, base_addr=0, dtype_bits=4)
    assert check_tile_bounds([record], _CTX) == []
