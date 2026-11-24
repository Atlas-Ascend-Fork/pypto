# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone effects and value-returning expressions preserve branches and loops."""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


def plus_one(value):
    return value + 1


@pl.jit(auto_mutex=False)
def standalone_effects(out: pl.Tensor[[8], pl.DT_INT32]):
    with pl.section_vector():
        if pl.get_subblock_idx() == 0:
            pair = (pl.get_block_idx(), 1)
            pl.setval(out, 0, pair[0] + 7)
            pl.setval(out, 1, pair[1])
            pl.setval(out, 2, plus_one(10))
            _unused = pl.setval(out, 3, 13)
            pl.setval(out, 4, 17)
            pl.setval(out, 5, plus_one(out[4]))
            pl.setval(out, 6, 19)
            pl.setval(out, 7, 23)


@pl.jit(auto_mutex=False)
def conditional_effects(out: pl.Tensor[[8], pl.DT_INT32], flag: pl.DT_INT32):
    with pl.section_vector():
        if pl.get_subblock_idx() == 0:
            if flag:
                pl.setval(out, 0, 7)
                pl.setval(out, 2, 17)
            else:
                pl.setval(out, 0, 9)
                pl.setval(out, 3, 19)
            pl.setval(out, 1, 11 if flag else 13)
            chosen = 1 if flag else 0
            and_result = flag != 0 and pl.get_block_idx() == 0
            or_result = flag != 0 or pl.get_block_idx() != 0
            pl.setval(out, 4, chosen + and_result + or_result)
            i = 0
            pl.setval(out, 5, out[5] + 1)
            while i < 3:
                pl.setval(out, 6, out[6] + 1)
                pl.setval(out, 5, out[5] + 1)
                i = i + 1
            for _j in pl.range(2):
                pl.setval(out, 7, out[7] + 1)


def _run(kernel, expected, *args):
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    out = torch.zeros(8, dtype=torch.int32, device=device)
    kernel[None, 1](out, *args)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.soc("950")
def test_standalone_effects():
    _run(standalone_effects, [7, 1, 11, 13, 17, 18, 19, 23])


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag", [0, 1])
def test_conditional_effects(flag):
    expected = [7, 11, 17, 0, 3, 4, 3, 2] if flag else [9, 13, 0, 19, 0, 4, 3, 2]
    _run(conditional_effects, expected, flag)
