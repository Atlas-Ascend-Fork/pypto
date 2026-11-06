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
"""Helpers shared by the pypto_pro unit tests."""

import difflib
import os
from pathlib import Path
import re
from typing import Any


def assert_ir_matches_dump(ir_node: Any, golden_path: Path) -> None:
    """Compare the printed IR of ``ir_node`` with the dump at ``golden_path``.

    Memref ids are numbered process-wide, so they are compared as ``memref_id=<id>``.
    With ``PYPTO_RENDER_IR`` set, the dump is rewritten from ``ir_node`` instead.
    """
    actual = re.sub(r"memref_id=\d+", "memref_id=<id>", str(ir_node))
    if os.environ.get("PYPTO_RENDER_IR"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual.strip() + "\n", encoding="utf-8")
        print(f"Updated: {golden_path}")
        return

    golden = golden_path.read_text(encoding="utf-8")
    if golden.strip() != actual.strip():
        diff = "".join(
            difflib.unified_diff(
                golden.splitlines(keepends=True),
                actual.splitlines(keepends=True),
            )
        )
        raise AssertionError("IR dump mismatch in %s:\n%s" % (golden_path, diff))
