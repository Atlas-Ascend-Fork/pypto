# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host tests for SIMT log allocation and capacity reporting; no NPU execution."""

from contextlib import nullcontext
import importlib
import struct
from types import SimpleNamespace
import warnings

from pypto_pro.runtime.sanitizer_replay import (
    MAX_RECORD_WORDS,
    RECORD_MAGIC,
    SanitizerReplayError,
    SanitizerReportError,
    TileScalarRecord,
    scan_report_file,
)
import pytest
import torch

jit = importlib.import_module("pypto_pro.runtime.jit")


@pytest.fixture
def cpu_npu(monkeypatch):
    monkeypatch.setattr(jit.torch, "npu", SimpleNamespace(
        synchronize=lambda: None,
        current_stream=lambda: None,
        stream=lambda stream: nullcontext(),
    ), raising=False)


def _buffer(capacity, offset=None):
    words = torch.zeros(capacity + 1, dtype=torch.int32)
    if offset is not None:
        record = TileScalarRecord.STRUCT.pack((RECORD_MAGIC << 16) | 5, 1, 32, offset, 42)
        payload = struct.unpack("<" + "i" * (len(record) // 4), record)
        words[0] = len(payload)
        words[1:1 + len(payload)] = torch.tensor(payload, dtype=torch.int32)
    return words.view(torch.uint8)


@pytest.mark.parametrize("aiv_per_block,slots,blocks,regions,capacity", [
    (2, 3, 1, 3, 63), (2, 3, 4, 12, 15), (1, 2, 3, 6, 31), (0, 3, 3, 3, 63),
])
def test_sanitizer_allocates_one_region_per_core(
    cpu_npu, monkeypatch, aiv_per_block, slots, blocks, regions, capacity,
):
    observed = {}
    empty = torch.empty

    def cpu_empty(nbytes, *, dtype, device):
        assert device == "npu"
        return empty(nbytes, dtype=dtype)

    def launch(args, block_dim, stream):
        log, capacity = args[-2:]
        observed.update(log=log, capacity=capacity)
        assert args[0] == "user argument"
        assert log.view(torch.int32)[::capacity + 1].tolist() == [0] * regions

    monkeypatch.setattr(jit.torch, "empty", cpu_empty)
    monkeypatch.setattr(jit, "_build_launch_entry", lambda compiled: launch)
    monkeypatch.setattr(jit, "_sanitizer_sub_blocks", lambda compiled: slots)
    monkeypatch.setattr(jit, "KSANITIZER_LOG_BUDGET_BYTES", 768)
    monkeypatch.setattr(jit, "_replay_sanitizer", lambda *args: None)
    compiled = SimpleNamespace(kernel_name="simt", source_file="test.py",
                               target=SimpleNamespace(aiv_per_block=aiv_per_block))
    jit._build_sanitizer_entry(compiled, ".")(("user argument",), blocks, None)
    assert observed["log"].numel() == 768
    assert observed["capacity"] == capacity


@pytest.mark.parametrize("offset", (0, 32))
def test_capacity_warning_preserves_decoded_findings(cpu_npu, tmp_path, offset):
    capacity = TileScalarRecord.STRUCT.size // 4  # Exact fit: no unwritten tail.
    log = _buffer(capacity, offset)
    if offset == 32:
        with pytest.raises(SanitizerReplayError) as error:
            jit._replay_sanitizer(log, capacity, 1, str(tmp_path), "simt", "test.py")
        assert [f.kind for f in error.value.findings] == ["TILE_OUT_OF_BOUNDS"]
    else:
        with pytest.warns(RuntimeWarning, match="records may have been dropped"):
            jit._replay_sanitizer(log, capacity, 1, str(tmp_path), "simt", "test.py")
    report = (tmp_path / "pypto_sanitizer_report.txt").read_text()
    assert "records may have been dropped" in report
    decoded = scan_report_file(str(next(tmp_path.glob("sanitizer_report.*.bin"))), "test.py")
    assert len(decoded) == (1 if offset == 32 else 0)


def test_too_small_region_is_not_reported_as_fully_clean(cpu_npu, tmp_path):
    with pytest.warns(RuntimeWarning, match="capacity"):
        jit._replay_sanitizer(_buffer(1), 1, 1, str(tmp_path), "simt", "test.py")
    assert "full coverage is not guaranteed" in (tmp_path / "pypto_sanitizer_report.txt").read_text()


def test_sufficient_capacity_has_no_coverage_warning(cpu_npu, tmp_path):
    capacity = MAX_RECORD_WORDS * 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jit._replay_sanitizer(_buffer(capacity, 0), capacity, 1, str(tmp_path), "simt", "test.py")
    assert not caught
    assert "WARNING" not in (tmp_path / "pypto_sanitizer_report.txt").read_text()


def test_counter_beyond_capacity_still_fails_closed(cpu_npu, tmp_path):
    log = _buffer(32)
    log.view(torch.int32)[0] = 33
    with pytest.raises(SanitizerReportError, match="exceeds capacity"):
        jit._replay_sanitizer(log, 32, 1, str(tmp_path), "simt", "test.py")
