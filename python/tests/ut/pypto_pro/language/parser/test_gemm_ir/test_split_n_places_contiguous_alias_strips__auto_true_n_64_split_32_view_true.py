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

Original case: test_split_n_places_contiguous_alias_strips (params: see file name).

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output of ``parse_gemm``.
"""

from pathlib import Path
import re

from common import parse_gemm


def test_split_n_places_contiguous_alias_strips__auto_true_n_64_split_32_view_true(tmp_path):
    printed = str(
        parse_gemm(
            tmp_path,
            (
                'wide = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,'
                ' target_memory=pl.MemorySpace.Acc, compact=1), addrs=65536, mutex_ids="auto", depth=1)\n'
                'c = pl.reinterpret(wide[0], shape=[64, 64])\n'
                'pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=32)\n'
            ),
            auto=True,
            l0n=32,
            n=64,
        )
    )
    printed = re.sub(r'memref_id=\d+', 'memref_id=<id>', printed)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
