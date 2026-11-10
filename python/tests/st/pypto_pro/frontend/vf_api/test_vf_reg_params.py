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

"""ST for passing RegTensor/MaskReg values into a VF helper.

The helper is instantiated as a standalone ``__simd_callee__`` function whose
parameters are register references. This exercises the register-parameter path
through parser, IR, CCE codegen, and bisheng.
"""

import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

M = 64
TILE_BYTES = M * 4


@pl.vector_function
def _add_regs(reg_a, reg_b, preg, out_tile):
    reg_out = vf.add(reg_a, reg_b, preg)
    vf.store_align(out_tile, reg_out, preg)


@pl.vector_function
def _load_add_store(in_a, in_b, out_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(in_a, 0)
    reg_b = vf.load_align(in_b, 0)
    _add_regs(reg_a, reg_b, preg, out_tile)


@pl.jit()
def _reg_param_kernel(
    a: pl.Tensor[[1, M], pl.DT_FP32],
    b: pl.Tensor[[1, M], pl.DT_FP32],
    out: pl.Tensor[[1, M], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[1, M], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a = pl.make_tile(tile_type, addr=0)
    in_b = pl.make_tile(tile_type, addr=TILE_BYTES)
    t_out = pl.make_tile(tile_type, addr=TILE_BYTES * 2)
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        _load_add_store(in_a, in_b, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


@pytest.mark.soc("950", "960")
def test_vf_reg_parameters():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(123)

    a = torch.randn([1, M], device=device, dtype=torch.float32) * 2.0
    b = torch.randn([1, M], device=device, dtype=torch.float32) * 2.0
    out = torch.empty([1, M], device=device, dtype=torch.float32)

    _reg_param_kernel(a, b, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a + b, rtol=1e-6, atol=1e-6)
