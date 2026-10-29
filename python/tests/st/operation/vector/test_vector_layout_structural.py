# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Graph regressions for full tiles, wider UB parents, and tail tiles."""

import os

import pytest
import torch

import pypto

pytestmark = [pytest.mark.soc("950"), pytest.mark.forked]


@pytest.fixture(scope='session', autouse=True)
def configure():
    pypto.pypto_impl.SetGlobalConfig({'pass.enable_slice': True})


@pypto.frontend.jit(debug_options={'runtime_debug_mode': 0, 'compile_debug_mode': 0})
def structural(
    x: pypto.Tensor(), out: pypto.Tensor(), index_out: pypto.Tensor(), op: str, h: int, w: int
):
    pypto.set_vec_tile_shapes(16, 128)
    parent = pypto.add(x, 0.125)
    pypto.set_vec_tile_shapes(16, w)
    v = pypto.view(parent, [h, w], [0, 0])
    if op == 'signbit':
        r = pypto.signbit(v - 1.0)
    elif op == 'gathermask':
        # The public API requires a complete inner axis; native tests cover shared strides.
        r = pypto.gathermask(v + 0.0, 1)
    elif op == 'topk':
        r, i = pypto.topk(v, 8, 1, True)
        pypto.assemble(i, [0, 0], index_out)
    pypto.assemble(r, [0] * r.dim, out)


STRUCTURAL_OPS = ['signbit', 'gathermask', 'topk']


@pytest.mark.parametrize('op', STRUCTURAL_OPS)
@pytest.mark.parametrize('h,w', [(16, 128), (16, 80), (7, 93)], ids=['full', 'view', 'tail'])
def test_structural(op, h, w):
    # The public gathermask API requires a complete inner axis.
    if op == 'gathermask' and w == 93:
        w = 96
    n = int(os.environ.get('TILE_FWK_DEVICE_ID', '5'))
    torch.npu.set_device(n)
    device = f'npu:{n}'
    source = (torch.arange(h * 128).reshape(h, 128) % 193).float() / 1024 + 0.8
    v = source[:, :w] + 0.125
    if op == 'signbit':
        reference = torch.signbit(v - 1)
    elif op == 'gathermask':
        reference = v[:, ::2]
    else:
        reference, indices = torch.topk(v, 8, 1, largest=True)
    out = torch.empty(reference.shape, dtype=reference.dtype, device=device)
    io = torch.empty((h, 8), dtype=torch.int32, device=device)
    structural(source.to(device), out, io, op, h, w)
    torch.testing.assert_close(out.cpu(), reference, rtol=2e-4, atol=1e-5)
    if op.startswith('topk'):
        torch.testing.assert_close(io.cpu(), indices.to(torch.int32), rtol=0, atol=0)
