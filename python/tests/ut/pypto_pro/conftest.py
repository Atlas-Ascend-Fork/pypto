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
"""Shared fixtures for unit tests."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


def duration_estimate(seconds: float):
    """Annotate a test case with an estimated duration."""

    def decorator(func):
        func.duration_estimate = seconds
        return func

    return decorator


@pytest.fixture
def assert_a6_golden(request):
    """Compare the complete A6 regression output with its checked-in golden."""
    def check(actual, label="ir"):
        printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(actual))
        path = Path(request.path).with_suffix("") / f"{request.node.name}-{label}.pypto"
        golden = path.read_text(encoding="utf-8")
        assert printed.rstrip("\n") == golden.rstrip("\n")

    return check
