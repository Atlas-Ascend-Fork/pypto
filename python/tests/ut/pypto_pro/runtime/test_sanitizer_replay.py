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
    check_phase_state_machine,
    run_replay,
)
import pytest

_CTX = ReplayContext("ut.py")


def test_phase_insert_final_read_records():
    records = [PhaseAccessRecord(op_kind=kind, phase=phase, acc_addr=0, line=10)
               for kind, phase in ((1, 1), (2, 2), (10, 2))]
    assert check_phase_state_machine(records, _CTX) == []


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
