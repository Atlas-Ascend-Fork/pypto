# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Boundary tests for pl.struct: field count limit, and creation inside a loop / branch.

pl.struct_array is covered by system/struct_array/, which pins its values; the probes that
used to live here logged whether an operation happened to work and swallowed every exception,
so no struct_array behaviour was ever asserted from this file.
"""

from __future__ import annotations

import logging
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


# =============================================================================
# struct.md: 字段数量上限验证
# =============================================================================
@pl.jit()
def struct_many_fields_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("TMany", f1=1, f2=2, f3=3, f4=4, f5=5, f6=6, f7=7, f8=8)
    with pl.section_vector():
        s.f1 = 10
        s.f2 = 20
        s.f3 = 30
        s.f4 = 40
        s.f5 = 50
        s.f6 = 60
        s.f7 = 70
        s.f8 = 80
        pl.setval(out, 0, s.f1 + s.f2 + s.f3 + s.f4 + s.f5 + s.f6 + s.f7 + s.f8)


@pytest.mark.soc("950")
def test_struct_many_fields():
    """验证 struct 支持 8 个字段，且字段可多次修改"""
    _check_npu()
    logging.info("------------test_struct_many_fields--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_many_fields_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([360], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_struct_many_fields passed!")


# =============================================================================
# struct.md: 循环/条件分支内创建 struct
# =============================================================================
@pl.jit()
def struct_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    with pl.section_vector():
        total = 0
        for i in pl.range(0, 4):
            s = pl.struct("LoopS", v=0)
            s.v = i * 10
            total = total + s.v
        pl.setval(out, 0, total)


@pytest.mark.soc("950")
def test_struct_in_loop():
    """验证可以在 for 循环内创建 struct"""
    _check_npu()
    logging.info("------------test_struct_in_loop--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_in_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0 + 10 + 20 + 30], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_struct_in_loop passed!")


@pl.jit()
def struct_in_branch_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    flag: pl.DT_BOOL,
):
    s = pl.struct("BranchS", v=0)
    with pl.section_vector():
        if flag:
            s.v = 100
        else:
            s.v = 200
        pl.setval(out, 0, s.v)


@pytest.mark.soc("950")
def test_struct_in_branch():
    """验证可以在 if/else 分支内修改 struct 字段（struct 须在分支外创建）"""
    _check_npu()
    logging.info("------------test_struct_in_branch--------------")
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_in_branch_kernel(out, True)
    torch.npu.synchronize()
    expected = torch.tensor([100], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
    logging.info("test_struct_in_branch passed!")


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_struct_many_fields,
        test_struct_in_loop,
        test_struct_in_branch,
    ]
    for t in tests:
        t()
        logging.info("%s completed!", t.__name__)
    logging.info("All struct boundary tests completed!")
