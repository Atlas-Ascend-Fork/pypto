#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
The rawShape of the input/output tensor changes between the pass right after
tile splitting and the last pass, while temp space is sized by tileShape, so
block offsets inside temp space must be computed from the validShape instead
of the input/output rawShape. These kernels reshape inputs with explicit
valid_shape, apply an elementwise op, and write the sliced result back to the
output tensor, exercising the assembled temp path for both unary ops
(asin/acos/asinh/atanh/sinh) and binary ops (logical_and).
"""

import os

import pytest
import torch

import pypto


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def unary_assembled_scratch(
    x3: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    out3: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    operation: str,
):
    tile, seq_tile = 16, 8
    bs, sl = x3.shape[0], x3.shape[1]
    nb = (bs + tile - 1) // tile
    ns = (sl + seq_tile - 1) // seq_tile
    pypto.set_vec_tile_shapes(16, tile)
    for bi in pypto.loop(nb, name="b", idx_name="bi", unroll_list=[2, 1]):
        bv = (bs - bi * tile).min(tile)
        x_t = x3[bi * tile:(bi + 1) * tile, :, :]
        x2 = pypto.reshape(x_t, [tile, sl * tile], valid_shape=[bv, sl * tile])
        for si in pypto.loop(ns, name="s", idx_name="si", unroll_list=[3, 1]):
            sv = (sl - si * seq_tile).min(seq_tile)
            cs = si * seq_tile * tile
            xs = x2[0:tile, cs:cs + seq_tile * tile]
            flat = pypto.reshape(xs, [tile * seq_tile, tile], valid_shape=[bv * sv, tile])
            value = pypto.tanh(flat) * 0.5
            if operation == "asin":
                result = pypto.asin(value)
            elif operation == "atanh":
                result = pypto.atanh(value)
            elif operation == "sinh":
                result = pypto.sinh(value)
            back = pypto.reshape(result, [tile, seq_tile, tile], valid_shape=[bv, sv, tile])
            out3[bi * tile:(bi + 1) * tile, si * seq_tile:(si + 1) * seq_tile, :] = back


def run_unary_assembled_scratch_test(operation: str):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    generator = torch.Generator().manual_seed(42)
    x_cpu = torch.randn(32, 32, 16, generator=generator)
    expected = getattr(torch, operation)(torch.tanh(x_cpu) * 0.5)
    x = x_cpu.to(f"npu:{device_id}")
    output = torch.zeros_like(x)
    unary_assembled_scratch(x, output, operation)
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("operation", ["asin", "atanh", "sinh"])
def test_unary_assembled_scratch_onboard(operation: str):
    run_unary_assembled_scratch_test(operation)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def binary_assembled_scratch(
    x3: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    y3: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    out3: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC]),
    operation: str,
):
    tile, seq_tile = 16, 8
    bs, sl = x3.shape[0], x3.shape[1]
    nb = (bs + tile - 1) // tile
    ns = (sl + seq_tile - 1) // seq_tile
    pypto.set_vec_tile_shapes(16, tile)
    for bi in pypto.loop(nb, name="b", idx_name="bi", unroll_list=[2, 1]):
        bv = (bs - bi * tile).min(tile)
        x_t = x3[bi * tile:(bi + 1) * tile, :, :]
        y_t = y3[bi * tile:(bi + 1) * tile, :, :]
        x2 = pypto.reshape(x_t, [tile, sl * tile], valid_shape=[bv, sl * tile])
        y2 = pypto.reshape(y_t, [tile, sl * tile], valid_shape=[bv, sl * tile])
        for si in pypto.loop(ns, name="s", idx_name="si", unroll_list=[3, 1]):
            sv = (sl - si * seq_tile).min(seq_tile)
            cs = si * seq_tile * tile
            xs = x2[0:tile, cs:cs + seq_tile * tile]
            ys = y2[0:tile, cs:cs + seq_tile * tile]
            flat_x = pypto.reshape(xs, [tile * seq_tile, tile], valid_shape=[bv * sv, tile])
            flat_y = pypto.reshape(ys, [tile * seq_tile, tile], valid_shape=[bv * sv, tile])
            value0 = pypto.tanh(flat_x) * 0.5
            value1 = pypto.tanh(flat_y) * 0.5
            if operation == "logicaland":
                result = pypto.logical_and(value0, value1)
            back = pypto.reshape(result, [tile, seq_tile, tile], valid_shape=[bv, sv, tile])
            out3[bi * tile:(bi + 1) * tile, si * seq_tile:(si + 1) * seq_tile, :] = back


def run_binary_assembled_scratch_test(operation: str):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    generator = torch.Generator().manual_seed(42)
    x_cpu = torch.randn(32, 32, 16, generator=generator)
    y_cpu = torch.randn(32, 32, 16, generator=generator)
    x = x_cpu.to(f"npu:{device_id}")
    y = y_cpu.to(f"npu:{device_id}")
    if operation == "logicaland":
        expected = torch.logical_and(torch.tanh(x_cpu) * 0.5, torch.tanh(y_cpu) * 0.5)
        output = torch.zeros_like(x, dtype=torch.bool)
    binary_assembled_scratch(x, y, output, operation)
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("operation", ["logicaland"])
def test_binary_assembled_scratch_onboard(operation: str):
    run_binary_assembled_scratch_test(operation)
