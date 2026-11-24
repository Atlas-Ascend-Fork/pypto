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

"""pl.ceil NPU 泛化测试 — 标量级向上取整。

参考文献:
  docs/zh/api/pro_api/Utils-API/python_syntax_sugar/ceil.md

被测 API:
  pl.ceil(x) — 取不小于 x 的最小整数;整数操作数取自身,浮点操作数向上取整。
  标量级别(区别于 tile 级逐元素的向量取整接口）。

数据类型覆盖:
  FP32 标量 / FP16 标量 / BF16 标量 (运行期元素读取)
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


# =============================================================================
# Kernel 1: FP32 标量 — pl.ceil (运行期值)
# =============================================================================
@pl.jit()
def kernel_scalar_ceil_fp32(src: pl.Tensor[[2, 7], pl.DT_FP32], out: pl.Tensor[[2, 7], pl.DT_FP32]):
    with pl.section_vector():
        for i in pl.range(7):
            out[0, i] = pl.ceil(src[0, i])
            out[1, i] = pl.ceil(src[1, i])


# =============================================================================
# Kernel 2: FP16 标量 — pl.ceil (运行期值)
# =============================================================================
@pl.jit()
def kernel_scalar_ceil_fp16(src: pl.Tensor[[2, 7], pl.DT_FP16], out: pl.Tensor[[2, 7], pl.DT_FP16]):
    with pl.section_vector():
        for i in pl.range(7):
            out[0, i] = pl.ceil(src[0, i])
            out[1, i] = pl.ceil(src[1, i])


# =============================================================================
# Kernel 3: BF16 标量 — pl.ceil (运行期值)
# =============================================================================
@pl.jit()
def kernel_scalar_ceil_bf16(src: pl.Tensor[[2, 7], pl.DT_BF16], out: pl.Tensor[[2, 7], pl.DT_BF16]):
    with pl.section_vector():
        for i in pl.range(7):
            out[0, i] = pl.ceil(src[0, i])
            out[1, i] = pl.ceil(src[1, i])


@pytest.mark.soc("950", "960")
def test_scalar_ceil_fp32_runs_on_device():
    """测试 pl.ceil 对运行期 FP32 标量向上取整的场景。

    测试目的:验证 pl.ceil(x) 在 NPU 上对 FP32 标量操作数求 ceil(x) 的正确性,
             覆盖正数、负数、零与小数边界。
    输入参数:src 为 [2, 7] FP32,两行各 7 个运行期值(含小数、负值与零）;
             out 为 [2, 7] FP32,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.ceil(src) 对应行(rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [1.5, -1.5, 2.0, -2.0, 0.0, 0.3, -0.3],
        [3.0, -3.0, 0.7, -0.7, 64.0, -64.0, 1.0],
    ], dtype=torch.float32)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_ceil_fp32[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.ceil(src), rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_scalar_ceil_fp16_runs_on_device():
    """测试 pl.ceil 对运行期 FP16 标量向上取整的场景。

    测试目的:验证 pl.ceil(x) 在 NPU 上对 FP16 标量操作数求 ceil(x) 的正确性;
              FP16 没有原生 ceil 重载,设备侧经 FP32 取整后回写 FP16。
    输入参数:src 为 [2, 7] FP16,两行各 7 个运行期值(含小数、负值与零）;
              out 为 [2, 7] FP16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.ceil(src.float()).to(torch.float16) 对应行
              (rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [1.5, -1.5, 2.0, -2.0, 0.0, 0.3, -0.3],
        [3.0, -3.0, 0.7, -0.7, 64.0, -64.0, 1.0],
    ], dtype=torch.float16)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_ceil_fp16[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    expected = torch.ceil(src.float()).to(torch.float16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_scalar_ceil_bf16_runs_on_device():
    """测试 pl.ceil 对运行期 BF16 标量向上取整的场景。

    测试目的:验证 pl.ceil(x) 在 NPU 上对 BF16 标量操作数求 ceil(x) 的正确性;
              BF16 没有原生 ceil 重载,设备侧经 FP32 取整后按 RNE 回写 BF16。
    输入参数:src 为 [2, 7] BF16,两行各 7 个运行期值(含小数、负值与零）;
              out 为 [2, 7] BF16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.ceil(src.float()).to(torch.bfloat16) 对应行
              (rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [1.5, -1.5, 2.0, -2.0, 0.0, 0.3, -0.3],
        [3.0, -3.0, 0.7, -0.7, 64.0, -64.0, 1.0],
    ], dtype=torch.bfloat16)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_ceil_bf16[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    expected = torch.ceil(src.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
