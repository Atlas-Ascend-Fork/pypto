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
"""Extracted from language/parser/test_gemm.py.

Original case: test_static_transpose_and_dynamic_slot_rejection (params: {'flag': 'trans_b'}).

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output of ``parse_gemm``.
"""

from pathlib import Path
import re

from common import parse_gemm


def test_static_transpose_and_dynamic_slot_rejection__flag_trans_b(tmp_path):
    printed = str(
        parse_gemm(
            tmp_path,
            """pl.gemm(c, a, b, l0a=la, l0b=lb, trans_b=True)""",
            trans_a=False,
            trans_b=True,
        )
    )
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
