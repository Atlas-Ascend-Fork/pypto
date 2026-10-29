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

Original case: test_mx_adjacent_pairs_and_native_mutex_equivalence (params: {'init': True}).

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output of ``parse_gemm``.
"""

from pathlib import Path
import re

from common import parse_gemm


def test_mx_adjacent_pairs_and_native_mutex_equivalence__init_true(tmp_path):
    printed = "\n".join(
        s.rstrip("\n")
        for s in (
            str(
                parse_gemm(
                    tmp_path,
                    """pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, init=True)""",
                    auto=True,
                    k=64,
                    mx=True,
                )
            ),
            str(
                parse_gemm(
                    tmp_path,
                    (
                        '\n'
                        'x = la.next()\n'
                        'y = lb.next()\n'
                        'sx = sa.next()\n'
                        'sy = sb.next()\n'
                        'pl.set_validshape(x, [64, 64])\n'
                        'pl.set_validshape(y, [64, 64])\n'
                        'pl.set_validshape(c, [64, 64])\n'
                        'pl.set_validshape(sx, [64, 2])\n'
                        'pl.set_validshape(sy, [2, 64])\n'
                        'pl.move(x, a)\n'
                        'pl.move(sx, s1)\n'
                        'pl.move(y, b)\n'
                        'pl.move(sy, s2)\n'
                        '\n'
                        'pl.matmul_mx(c, x, y, sx, sy)\n'
                    ),
                    auto=True,
                    k=64,
                    mx=True,
                )
            ),
        )
    )
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
