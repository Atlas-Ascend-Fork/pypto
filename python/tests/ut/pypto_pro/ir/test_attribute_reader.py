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
"""Attribute snapshots retain supported metadata and reject unsupported values."""

from pypto_pro import ir
import pytest

import pypto


def test_call_snapshot_retains_lists_and_is_read_only():
    node = ir.Call("attribute_probe", [], {"ids": [2, 5], "fields": ["cursor", "tiles"],
                                         "space": ir.MemorySpace.Vec, "enabled": True, "scale": 1.25},
                   ir.Span.unknown())
    attrs = ir.get_attributes(node)
    assert ir.attributes_api_version == 1
    assert attrs == {"ids": [2, 5], "fields": ["cursor", "tiles"],
                     "space": ir.MemorySpace.Vec, "enabled": True, "scale": 1.25}
    attrs["ids"].append(9)
    assert ir.get_attributes(node)["ids"] == [2, 5]


def test_function_and_for_snapshots_retain_list_metadata():
    span = ir.Span.unknown()
    value = ir.ConstInt(2, ir.INT64, span)
    function = ir.Function("probe", [], [], ir.SeqStmts(span), span,
                           attrs={"ids": [2, 5], "label": "cursor"})
    assert ir.get_attributes(function) == {"ids": [2, 5], "label": "cursor"}
    loop = ir.ForStmt(ir.Var("i", ir.ScalarType(ir.INT64), span), value, value, value,
                      [], ir.SeqStmts(span), [], span, attrs={"ids": [1, 4]})
    assert ir.get_attributes(loop) == {"ids": [1, 4]}


def test_unknown_attribute_type_fails_with_its_key():
    span = ir.Span.unknown()
    node = ir.Function("probe", [], [], ir.SeqStmts(span), span,
                       attrs={"symbolic": pypto.SymbolicScalar("extent")})
    with pytest.raises(TypeError, match="symbolic.*unsupported type"):
        ir.get_attributes(node)


def test_unsupported_node_type_fails():
    with pytest.raises(TypeError):
        ir.get_attributes(ir.ConstInt(1, ir.INT64, ir.Span.unknown()))
