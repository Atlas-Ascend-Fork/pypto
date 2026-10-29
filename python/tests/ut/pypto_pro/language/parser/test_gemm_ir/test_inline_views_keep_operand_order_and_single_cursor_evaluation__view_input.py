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

Original case: test_inline_views_keep_operand_order_and_single_cursor_evaluation (params: {'view': 'input'}).

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output of ``parse_gemm``.
"""

from pathlib import Path
import re

from common import parse_gemm


def test_inline_views_keep_operand_order_and_single_cursor_evaluation__view_input(tmp_path):
    printed = "\n".join(
        s.rstrip("\n")
        for s in (
            str(
                parse_gemm(
                    tmp_path,
                    (
                        'pl.gemm(gc.next(), pl.reinterpret(a, shape=[64, 128], layout=pl.NZ), gb.next(), l0a=la.next(),'
                        ' l0b=lb.next())\n'
                    ),
                )
            ),
            str(
                parse_gemm(
                    tmp_path,
                    (
                        '\n'
                        '        z = gc.next()\n'
                        '        x = pl.reinterpret(a, shape=[64, 128], layout=pl.NZ)\n'
                        '        y = gb.next()\n'
                        '        u = la.next()\n'
                        '        v = lb.next()\n'
                        '        pl.set_validshape(u, [64, 128])\n'
                        '        pl.set_validshape(v, [128, 64])\n'
                        '        pl.set_validshape(z, [64, 64])\n'
                        '        pl.move(u, x)\n'
                        '        pl.move(v, y)\n'
                        '        pl.matmul(z, u, v)\n'
                        '    \n'
                    ),
                )
            ),
        )
    )
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
