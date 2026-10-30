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
"""Front-end validation of the early-start sync ``force`` keyword.

``force`` selects an AscendC template argument, so it must be a compile-time
bool: a string or an int is refused by the declared-type gate, and a runtime
scalar by the compile-time-constant gate. The IR builders are directly callable
and enforce the same bool contract, so a bypassing caller cannot truthiness-coerce
``force``.
"""
from pypto_pro import ir
from pypto_pro._errors import InvalidType
from pypto_pro.ir.op.system_ops import set_next_task_start, wait_pre_task_end
import pypto_pro.language as pl
import pytest


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("force", [True, False])
def test_wait_pre_task_end_accepts_bool_literal(force):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.wait_pre_task_end(force=force)

    _parse(kernel)


@pytest.mark.parametrize("force", [True, False])
def test_set_next_task_start_accepts_bool_literal(force):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.set_next_task_start(force=force)

    _parse(kernel)


def test_force_defaults_accepted():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.wait_pre_task_end()

    _parse(kernel)


@pytest.mark.parametrize("force", ["False", 7])
def test_force_rejects_non_bool(force):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.wait_pre_task_end(force=force)

    with pytest.raises(InvalidType, match=r"force expects bool"):
        _parse(kernel)


def test_force_rejects_runtime_value():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32], flag: pl.DT_BOOL):
        pl.system.set_next_task_start(force=flag)

    with pytest.raises(InvalidType, match=r"force.*compile-time bool"):
        _parse(kernel)


@pytest.mark.parametrize("build", [wait_pre_task_end, set_next_task_start])
@pytest.mark.parametrize("force", ["False", 7, None])
def test_builders_reject_non_bool_force(build, force):
    """Direct builder calls bypass the parser; they must refuse non-bool values too."""
    with pytest.raises(InvalidType, match=r"force must be a bool"):
        build(force=force)
