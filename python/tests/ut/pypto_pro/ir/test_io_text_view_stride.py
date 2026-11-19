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
"""tensor_view default stride (`-2`): expanded only for concrete shapes, kept empty for symbolic shapes."""

from pypto.pypto_impl import ir

_PREFIX = "function f incast(int64_t %entry_0, ptr<float> %p_0, "
_SUFFIX = " %x_0) outcast() #entry(false) #type(Opaque) {\n    return;\n}\n"


def _load(param_type: str):
    return ir.text_load_function(_PREFIX + param_type + _SUFFIX)


def test_default_stride_keeps_ir_stride_empty():
    func = _load("tensor<4 x 8, float, tensor_view<-2, ND, %p_0>>")
    view = func.params[-1].type.tensor_view
    assert view.stride == []
    assert "tensor_view<-2, ND, %p_0>" in str(func)


def test_default_stride_on_symbolic_shape_keeps_ir_stride_empty():
    func = _load("tensor<4 x %entry_0 x 128, float, tensor_view<-2, ND, %p_0>>")
    view = func.params[-1].type.tensor_view
    assert view.stride == []
    assert "tensor_view<-2, ND, %p_0>" in str(func)


def test_default_stride_round_trip_is_stable():
    for shape in ("4 x 8", "4 x %entry_0 x 128"):
        text = _PREFIX + f"tensor<{shape}, float, tensor_view<-2, ND, %p_0>>" + _SUFFIX
        first = ir.text_dump(ir.text_load_function(text))
        second = ir.text_dump(ir.text_load_function(first))
        assert first == second
        assert "tensor_view<-2, ND, %p_0>" in first


def test_explicit_stride_is_kept_verbatim():
    func = _load("tensor<4 x 8, float, tensor_view<8 x 1, ND, %p_0>>")
    view = func.params[-1].type.tensor_view
    assert [str(dim) for dim in view.stride] == ["8", "1"]
    assert "tensor_view<8 x 1, ND, %p_0>" in str(func)


def test_explicit_symbolic_stride_is_kept_verbatim():
    func = _load("tensor<4 x %entry_0 x 128, float, tensor_view<%entry_0 x 1, ND, %p_0>>")
    view = func.params[-1].type.tensor_view
    assert [str(dim) for dim in view.stride] == ["%entry_0", "1"]
    assert "tensor_view<%entry_0 x 1, ND, %p_0>" in str(func)
