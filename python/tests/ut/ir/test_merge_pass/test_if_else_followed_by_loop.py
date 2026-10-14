# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

import pypto

from ..test_common import check_snapshot, run_merge_pass

_GOLDEN_DIR = Path(__file__).parent


def test_sunk_loop_zero_trip():
    """A zero-trip sunk loop keeps its initial value for the consumer after the loop."""

    def foo(x, y, out):
        pypto.set_vec_tile_shapes(16, 16)
        if pypto.SymbolicScalar("cond") > 0:
            value = x + y
        else:
            value = x - y
        for _ in pypto.loop(0):
            value = value + y
        pypto.assemble(value + x, [0, 0], out)

    x = pypto.Tensor([16, 16], pypto.DT_FP32, 'x')
    y = pypto.Tensor([16, 16], pypto.DT_FP32, 'y')
    out = pypto.Tensor([16, 16], pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, x, y, out)
    check_snapshot(func, _GOLDEN_DIR / "test_sunk_loop_zero_trip.pypto")


def test_sunk_loop_dynamic_shape():
    """Each branch substitutes its own loop bound and view valid shape into the sunk loop."""

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        if pypto.SymbolicScalar("cond") > 0:
            width = 8
            count = 2
        else:
            width = 16
            count = 1
        for i in pypto.loop(count):
            view = pypto.view(x, [16, 16], [i * 16, 0], valid_shape=[width, 16])
            pypto.assemble(view, [i * 16, 0], out)

    x = pypto.Tensor([32, 16], pypto.DT_FP32, 'x')
    out = pypto.Tensor([32, 16], pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, x, out)
    check_snapshot(func, _GOLDEN_DIR / "test_sunk_loop_dynamic_shape.pypto")


def test_sunk_loop_add_dyn_valid_shape():
    """ADD after VIEW keeps the branch valid shape on the result tensor, not a dangling yield var."""

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        if pypto.SymbolicScalar("cond") > 0:
            width = 8
        else:
            width = 16
        for _ in pypto.loop(2):
            view = pypto.view(x, [16, 16], [0, 0], valid_shape=[width, 16])
            value = view + view
            pypto.assemble(value, [0, 0], out)

    x = pypto.Tensor([16, 16], pypto.DT_FP32, 'x')
    out = pypto.Tensor([16, 16], pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, x, out)
    check_snapshot(func, _GOLDEN_DIR / "test_sunk_loop_add_dyn_valid_shape.pypto")


def test_leading_loop_stays():
    """A loop before an if stays outside; the loop after the if sinks into both branches."""

    def foo(x, y, out):
        pypto.set_vec_tile_shapes(16, 16)
        value = x
        for _ in pypto.loop(2):
            value = value + y
        if pypto.SymbolicScalar("cond") > 0:
            value = value + x
        else:
            value = value - y
        for _ in pypto.loop(2):
            value = value + y
        pypto.assemble(value, [0, 0], out)

    x = pypto.Tensor([16, 16], pypto.DT_FP32, 'x')
    y = pypto.Tensor([16, 16], pypto.DT_FP32, 'y')
    out = pypto.Tensor([16, 16], pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, x, y, out)
    check_snapshot(func, _GOLDEN_DIR / "test_leading_loop_stays.pypto")


def test_sunk_nested_if_dynamic_shape():
    """An if nested in a sunk loop carries the branch valid shape on its return tensor."""

    def foo(x, out):
        pypto.set_vec_tile_shapes(16, 16)
        if pypto.SymbolicScalar("cond") > 0:
            width = 8
        else:
            width = 16
        value = pypto.view(x, [16, 16], [0, 0], valid_shape=[width, 16])
        for _ in pypto.loop(2):
            if pypto.SymbolicScalar("other") > 0:
                value = pypto.view(x, [16, 16], [0, 0], valid_shape=[width, 16])
            else:
                value = pypto.view(x, [16, 16], [0, 0], valid_shape=[width, 16])
        pypto.assemble(value, [0, 0], out)

    x = pypto.Tensor([16, 16], pypto.DT_FP32, 'x')
    out = pypto.Tensor([16, 16], pypto.DT_FP32, 'out')
    func = run_merge_pass(foo, x, out)
    check_snapshot(func, _GOLDEN_DIR / "test_sunk_nested_if_dynamic_shape.pypto")
