#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
profiling of aicpu pref  test for PyPTO
"""

import multiprocessing as mp
import os

import pytest
import torch
import torch_npu

import pypto


@pypto.frontend.jit()
def matmul_add(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT8),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
):
    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes([tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        out.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))


def device_run_data_from_device_mix_nodep():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8

    # prepare data
    c_data_list = []
    d_data_list = []

    count = 1

    a_rawdata = torch.tensor([[1] * k] * n)
    b_rawdata = torch.tensor([[1] * m] * k)
    a_data = a_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')
    b_data = b_rawdata.to(dtype=torch.int8, device=f'npu:{device_id}')

    for idx in range(count):
        c_rawdata = torch.tensor([[idx] * m] * n)
        c_data = c_rawdata.to(dtype=torch.int32, device=f'npu:{device_id}')
        c_data_list.append(c_data)

        d_data = torch.zeros((n, m), dtype=torch.int32, device=f'npu:{device_id}')
        d_data_list.append(d_data)

        # def inputs and outputs
        matmul_add(a_data, b_data, c_data, d_data)

    torch_npu.npu.synchronize()

    for idx in range(count):
        golden = torch.matmul(a_rawdata.to(torch.int32), b_rawdata.to(torch.int32)) + torch.tensor([[idx] * m] * n)
        assert torch.equal(d_data_list[idx].cpu(), golden)


@pytest.mark.soc("910", "950")
def test_launch_blocking():
    os.environ["ASCEND_RT_LAUNCH_BLOCKING"] = "1"
    mp.set_start_method('spawn', force=True)
    p = mp.Process(target=device_run_data_from_device_mix_nodep)
    p.start()
    p.join()
    assert p.exitcode == 0, f"child process exited with code {p.exitcode}"
