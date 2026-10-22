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
"""Diagnostic rendering separates source previews from messages and locations."""

import warnings

from pypto_pro import ir
from pypto_pro._errors import (
    CheckSkipped,
    InvalidType,
    _warn_located,
    warn_check_skipped,
)
import pytest

MARKER = "ONLY_IN_SOURCE_PREVIEW"


def _error():
    return InvalidType(
        "actual diagnostic",
        span={"filename": "kernel.py", "line": 2, "column": 1},
        source_lines=["first", MARKER, "third"],
        hint="use a Tile", note="a diagnostic note",
    )


def test_render_without_source_excludes_source_preview():
    error = _error()
    assert MARKER in str(error)
    assert MARKER not in error.render(with_source=False)
    assert error.message == "actual diagnostic"


def test_render_without_source_preserves_diagnostic_fields():
    diagnostic = _error().render(with_source=False)
    for pattern in ("actual diagnostic", "hint: use a Tile", "note: a diagnostic note", "kernel.py:2:1"):
        assert pattern in diagnostic


@pytest.mark.parametrize("setting,count", [("", 0), ("other", 0), ("skipped-checks", 1), ("other, all", 1)])
def test_skipped_checks_are_opt_in_and_located(monkeypatch, setting, count):
    monkeypatch.setenv("PYPTOPRO_INFO", setting)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warn_check_skipped("dynamic offset", ir.Span("a_kernel.py", 17, 3))
    assert len(recorded) == count
    if recorded:
        assert recorded[0].category is CheckSkipped
        assert recorded[0].filename == "a_kernel.py"
        assert recorded[0].lineno == 17


def test_warning_locations_have_separate_registries():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default", CheckSkipped)
        for filename in ("first_kernel.py", "second_kernel.py", "first_kernel.py"):
            _warn_located("same diagnostic", CheckSkipped, ir.Span(filename, 23, 1))
    assert [item.filename for item in recorded] == ["first_kernel.py", "second_kernel.py"]
