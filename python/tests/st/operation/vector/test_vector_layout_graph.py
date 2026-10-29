# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""End-to-end regressions for subviews retaining a wider UB parent layout."""

import os

import pytest
import torch

import pypto

pytestmark = pytest.mark.soc("950")


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def unary(x: pypto.Tensor(), y: pypto.Tensor(), op: str):
    pypto.set_vec_tile_shapes(16, 128)
    z = pypto.add(x, 0.25)
    pypto.set_vec_tile_shapes(16, 80)
    v = pypto.view(z, [16, 80], [0, 0])
    if op == 'log1p':
        r = pypto.log1p(v)
    elif op == 'erf':
        r = pypto.erf(v)
    elif op == 'floordivs':
        r = pypto.floor_div(v, 0.2)
    elif op == 'cumsum':
        r = pypto.cumsum(v, 0)
    else:
        r = pypto.cumprod(v, 0)
    pypto.assemble(r, [0, 0], y)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def pad(x: pypto.Tensor(), y: pypto.Tensor()):
    pypto.set_vec_tile_shapes(16, 80)
    r = x.pad([0, 0, 0, 9], mode='constant', value=0)
    pypto.assemble(r, [0, 0], y)


@pytest.mark.parametrize("operation", ["log1p", "erf", "floordivs", "cumsum", "cumprod"])
def test_unary_parent_stride(operation):
    device = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device)
    source = (torch.arange(16 * 128).reshape(16, 128) % 97).float() / 97
    output = torch.empty((16, 80), device=f"npu:{device}")
    unary(source.to(output.device), output, operation)
    values = source[:, :80] + 0.25
    reference = {
        "log1p": torch.log1p,
        "erf": torch.erf,
        "floordivs": lambda x: torch.floor(x / 0.2),
        "cumsum": lambda x: x.cumsum(0),
        "cumprod": lambda x: x.cumprod(0),
    }[operation](values)
    torch.testing.assert_close(output.cpu(), reference, rtol=1e-4, atol=1e-5)


def test_pad_column_tiles():
    device = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device)
    source = (torch.arange(7 * 128).reshape(7, 128) % 97).float() / 97
    output = torch.empty((16, 128), device=f"npu:{device}")
    pad(source.to(output.device), output)
    torch.testing.assert_close(output.cpu(), torch.nn.functional.pad(source, (0, 0, 0, 9)), rtol=0, atol=0)
