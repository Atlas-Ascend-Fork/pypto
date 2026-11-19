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
"""pypto_pro text: dotted calls, kwargs, ``mutex_tile`` and empty view slots roundtrip verbatim."""

from pathlib import Path

from pypto.pypto_impl import ir

_GOLDEN = Path(__file__).with_suffix(".pypto")


def test_pro_text_roundtrip():
    text = _GOLDEN.read_text(encoding="utf-8")
    func = ir.text_load_function(text)

    assert str(func).rstrip("\n") == text.rstrip("\n")

    assigned = func.body.stmts[0]
    assert isinstance(assigned.value, ir.Call)
    assert assigned.value.name == "block.make_mutex_tile"
