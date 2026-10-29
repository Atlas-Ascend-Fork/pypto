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

Original case: test_omitted_extents_restore_reused_tiles (params: see file name).

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output of ``parse_gemm``.
"""

from pathlib import Path
import re

from common import parse_gemm


def test_omitted_extents_restore_reused_tiles__kwargs_none_options_none_partial_m_48_n_40_k_64(tmp_path):
    printed = "\n".join(
        s.rstrip("\n")
        for s in (
            str(
                parse_gemm(
                    tmp_path,
                    (
                        'u = la[0]\n'
                        'v = lb[0]\n'
                        'pl.gemm(c, a, b, l0a=u, l0b=v, m=48, n=40, k=64)\n'
                        'pl.gemm(c, a, b, l0a=u, l0b=v)\n'
                    ),
                )
            ),
            str(
                parse_gemm(
                    tmp_path,
                    (
                        'u = la[0]\n'
                        'v = lb[0]\n'
                        'pl.gemm(c, a, b, l0a=u, l0b=v, m=48, n=40, k=64)\n'
                        'pl.gemm(c, a, b, l0a=u, l0b=v, n=64, k=128, m=64)\n'
                    ),
                )
            ),
        )
    )
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
