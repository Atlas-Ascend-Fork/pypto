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

import pypto_pro.runtime.sanitizer_replay as replay_module
from pypto_pro.runtime.sanitizer_replay import (
    Detection,
    Finding,
    PhaseAccessRecord,
    ReplayContext,
    check_phase_state_machine,
    run_replay,
)

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
