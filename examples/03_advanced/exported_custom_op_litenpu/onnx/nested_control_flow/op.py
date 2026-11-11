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
"""The three ops of the nested-control-flow demo: inference, CPU references, and the declarations.

All three are elementwise and shape-preserving, so they share one ``infer_shape`` and one
``infer_dtype``. Importing this module registers ``torch.ops.pypto.{add,sub,mul}`` and enqueues their
ONNX wiring for ``finalize_pending_ops()``.

``op_type`` names the COMPUTE, so these are the plain ``Add`` / ``Sub`` / ``Mul``; the demo's subject is
the GRAPH shape around them, not the arithmetic.
"""
from kernel import add_kernel, mul_kernel, sub_kernel
import torch

import pypto.extensions.torch_custom_op_litenpu

ONNX_OPSET_VERSION = 12


# the CPU compute for each op, also used as the reference.
def add_torch(input0, input1):
    return input0 + input1


def sub_torch(input0, input1):
    return input0 - input1


def mul_torch(input0, input1):
    return input0 * input1


# Elementwise and shape-preserving, so one pair of inference functions serves all three ops.
def elementwise_infer_shape(input0_shape: torch.Size, input1_shape: torch.Size) -> torch.Size:
    return input0_shape


def elementwise_infer_dtype(input0_dtype: torch.dtype, input1_dtype: torch.dtype) -> torch.dtype:
    return input0_dtype


for _kernel, _torch_defn, _stem, _op_type in (
    (add_kernel, add_torch, "add", "Add"),
    (sub_kernel, sub_torch, "sub", "Sub"),
    (mul_kernel, mul_torch, "mul", "Mul"),
):
    pypto.extensions.torch_custom_op_litenpu.ExportedCustomOp(
        kernel=_kernel,
        infer_shape=elementwise_infer_shape,
        infer_dtype=elementwise_infer_dtype,
        torch_defn=_torch_defn,
        torch_op_qualname=f"pypto::{_stem}",
        onnx_spec=pypto.extensions.torch_custom_op_litenpu.OnnxSymbolicSpec(
            op_type=_op_type, opset_version=ONNX_OPSET_VERSION
        ),
    )
