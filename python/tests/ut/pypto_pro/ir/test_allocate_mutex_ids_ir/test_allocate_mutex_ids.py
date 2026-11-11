# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compare the allocated Vector program with the sibling IR dump."""

from pathlib import Path
import re

from mutex_kernel import parse_copy_tiles


def test_allocate_mutex_ids():
    program = parse_copy_tiles()
    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(program))
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
