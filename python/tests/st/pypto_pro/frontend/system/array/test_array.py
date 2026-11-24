# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU coverage for ``pl.array`` fixed-length kernel-local scalar arrays."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


@pl.jit()
def array_alias_kernel(out: pl.Tensor[[3], pl.DT_INT32]):
    datas = pl.array(8, dtype=pl.DT_INT32)
    datas[0] = 7
    a = datas[0]
    alias = datas
    alias[0] = 9
    datas[1] = a
    with pl.section_vector():
        pl.setval(out, 0, a)
        pl.setval(out, 1, datas[0])
        pl.setval(out, 2, datas[1])


@pytest.mark.soc("950", "960")
def test_array_alias_and_scalar_capture():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    array_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([7, 9, 7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def array_loop_kernel(out: pl.Tensor[[8], pl.DT_INT32]):
    datas = pl.array(8, dtype=pl.DT_INT32)
    for i in pl.range(0, 8):
        datas[i] = i * 2 + 1
    with pl.section_vector():
        for i in pl.range(0, 8):
            pl.setval(out, i, datas[i])


@pytest.mark.soc("950", "960")
def test_array_fill_and_read_back_in_separate_loops():
    _check_npu()
    out = torch.zeros(8, device=ST_DEVICE, dtype=torch.int32)
    array_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
