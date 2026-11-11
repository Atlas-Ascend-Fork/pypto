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
"""Kernels for the nested-control-flow demo: one per branch of the conditional.

Three separate kernels, because each branch computes something different. Their tensor annotations are
left empty (``pypto.Tensor([...])``) so shapes and dtypes are supplied at deploy/run. Inference and the
CPU references live next door in ``op.py``.
"""
import pypto


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def add_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 + input1)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def sub_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 - input1)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def mul_kernel(
    input0: pypto.Tensor([...]),
    input1: pypto.Tensor([...]),
    output: pypto.Tensor([...]),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output.move(input0 * input1)
