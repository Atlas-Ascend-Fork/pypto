# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""NaN modes: exact values, NaN categories, first indices, signed zeros and output guards."""

import os

import pytest
import torch
import torch_npu

import pypto


@pypto.frontend.jit()
def _reduce_kernel(x: pypto.Tensor(), lo: pypto.Tensor(), hi: pypto.Tensor(),
                   ilo: pypto.Tensor(), ihi: pypto.Tensor(), axis: int, keepdim: bool,
                   propagate_nan: bool, tile: list):
    pypto.set_vec_tile_shapes(*tile)
    offset = [1] + [0] * (len(lo.shape) - 1)
    pypto.assemble(pypto.amin(x, axis, keepdim, propagate_nan), offset, lo)
    pypto.assemble(x.amax(axis, keepdim, propagate_nan), offset, hi)
    pypto.assemble(pypto.argmin(x, axis, keepdim, propagate_nan), offset, ilo)
    pypto.assemble(x.argmax(axis, keepdim, propagate_nan), offset, ihi)


@pypto.frontend.jit()
def _binary_kernel(x: pypto.Tensor(), y: pypto.Tensor(), lo: pypto.Tensor(), hi: pypto.Tensor()):
    pypto.set_vec_tile_shapes(4, 32)
    pypto.assemble(x.minimum(y, False), [1, 0], lo)
    pypto.assemble(pypto.maximum(x, y, False), [1, 0], hi)


@pypto.frontend.jit()
def _scalar_kernel(x: pypto.Tensor(), lo: pypto.Tensor(), hi: pypto.Tensor(),
                   scalar: float):
    pypto.set_vec_tile_shapes(4, 32)
    pypto.assemble(pypto.minimum(x, scalar, False), [1, 0], lo)
    pypto.assemble(pypto.maximum(scalar, x, False), [1, 0], hi)


def _device():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch_npu.npu.set_device(device_id)
    return f"npu:{device_id}"


def _check(actual, expected, check_zero_sign=False):
    actual = actual.cpu()
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    valid = ~torch.isnan(expected)
    assert torch.equal(actual[valid], expected[valid])
    if check_zero_sign:
        zero = valid & (expected == 0)
        assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


def _reduce_reference(x, axis, keepdim, is_max, return_index):
    rows = x.movedim(axis, -1)
    values = []
    for row in rows.reshape(-1, rows.shape[-1]):
        indices = torch.arange(row.numel())
        indices = indices[~torch.isnan(row)]
        if indices.numel() == 0:
            value = 0 if return_index else float("nan")
        else:
            selected = row[indices]
            if return_index:
                value = indices[torch.argmax(selected) if is_max else torch.argmin(selected)].item()
            else:
                value = (torch.max(selected) if is_max else torch.min(selected)).item()
                # IEEE zero rule is independent of Torch CPU's choice of an equal operand.
                if value == 0:
                    signs = torch.signbit(selected[selected == 0])
                    negative = signs.all().item() if is_max else signs.any().item()
                    value = -0.0 if negative else 0.0
        values.append(value)
    result = torch.tensor(values, dtype=torch.int32 if return_index else x.dtype).reshape(rows.shape[:-1])
    return result.unsqueeze(axis) if keepdim else result


def _special_rows(dtype):
    x = torch.full((8, 65), float("nan"), dtype=dtype)
    x[1, 64] = -float("inf")  # Real infinity must beat NaN, even after a full NaN tile.
    x[2, 33] = float("inf")
    x[3, 31] = x[3, 64] = 7  # First equal extremum across tiles.
    x[4] = 2
    x[4, 32] = x[4, 64] = float("nan")
    x[5] = 0.0
    x[5, 63] = -0.0
    x[6] = -0.0
    x[7] = torch.arange(65).to(dtype) - 32
    return x


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("axis,keepdim", [(1, False), (0, True)])
def test_nan_reductions(dtype, axis, keepdim):
    x = _special_rows(dtype)
    if axis == 0:
        x = x.t().contiguous()
    device = _device()
    results = [_reduce_reference(x, axis, keepdim, is_max, index)
               for index, is_max in [(False, False), (False, True), (True, False), (True, True)]]
    outputs = [torch.full((r.shape[0] + 2, *r.shape[1:]), -101, dtype=r.dtype, device=device)
               for r in results]
    tile = [32, 16] if axis == 0 else [4, 32]
    _reduce_kernel(x.to(device), *outputs, axis, keepdim, False, tile)
    for output, expected in zip(outputs, results):
        assert torch.all(output[0] == -101) and torch.all(output[-1] == -101)
        _check(output[1:-1], expected, check_zero_sign=expected.is_floating_point())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nan_binary(dtype):
    x = torch.tensor([[float("nan"), -float("inf"), float("inf"), 0.0, -0.0, 3, float("nan")]], dtype=dtype)
    y = torch.tensor([[2, float("nan"), -float("inf"), -0.0, 0.0, 3, float("nan")]], dtype=dtype)
    device = _device()
    outputs = [torch.full((x.shape[0] + 2, x.shape[1]), -101, dtype=dtype, device=device) for _ in range(2)]
    _binary_kernel(x.to(device), y.to(device), *outputs)
    for output, op, is_max in zip(outputs,
                                 (torch.fmin, torch.fmax),
                                 (False, True)):
        expected = op(x, y)
        both_zero = (x == 0) & (y == 0)
        negative = (torch.signbit(x) & torch.signbit(y)) if is_max else (torch.signbit(x) | torch.signbit(y))
        expected[both_zero] = torch.where(negative, -0.0, 0.0)[both_zero].to(dtype)
        assert torch.all(output[0] == -101) and torch.all(output[-1] == -101)
        _check(output[1:-1], expected, check_zero_sign=True)


@pytest.mark.parametrize("scalar", [float("nan"), 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_nan_scalar(scalar, dtype):
    x = _special_rows(dtype)
    device = _device()
    outputs = [torch.full((10, 65), -101, dtype=dtype, device=device) for _ in range(2)]
    _scalar_kernel(x.to(device), *outputs, scalar)
    for output, op in zip(outputs, [torch.fmin, torch.fmax]):
        assert torch.all(output[0] == -101) and torch.all(output[-1] == -101)
        _check(output[1:-1], op(x, torch.tensor(scalar, dtype=dtype)))
