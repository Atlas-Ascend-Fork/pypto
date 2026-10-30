# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Minimal SIMT kernel compilation smoke test."""

import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.jit(mode="simt", max_threads=32)
def simt_kernel_int32_write_one(out: pl.Tensor[[1], pl.DT_INT32]):
    out[0] = 1


@pytest.mark.soc("950")
def test_simt_kernel_int32_write_one():
    torch.npu.set_device(ST_DEVICE)
    out = torch.zeros(1, dtype=torch.int32, device=ST_DEVICE)
    simt_kernel_int32_write_one[1, 1](out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.ones(1, dtype=torch.int32), rtol=0, atol=0)
