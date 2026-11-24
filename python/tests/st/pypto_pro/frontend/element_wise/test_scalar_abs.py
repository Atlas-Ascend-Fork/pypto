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

"""pl.scalar_abs NPU 泛化测试 — 标量级绝对值语法糖,覆盖多种数据类型与控制流。

参考文献:
  docs/zh/api/pro_api/Utils-API/python_syntax_sugar/scalar_abs.md

被测 API:
  pl.scalar_abs(x)  — 取单个标量的绝对值 |x|,恰好 1 个参数,不接受关键字参数；
  x 为标量 Expr 或 Python int。标量级别(区别于 tile 级逐元素的 pl.abs）。

数据类型覆盖:
  FP32 标量 / INT32 标量 / FP16 标量 / BF16 标量 (运行期元素读取)

控制流覆盖:
  for 循环

语法糖覆盖:
  Python 内置 abs(),与 pl.scalar_abs 等价(同一次解析走同一注册表条目）
"""

import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu

ST_DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


# =============================================================================
# Test 1: FP32 标量 — pl.scalar_abs 与裸内置 abs()
# =============================================================================
@pl.jit()
def kernel_scalar_abs_fp32(src: pl.Tensor[[2, 7], pl.DT_FP32], out: pl.Tensor[[2, 7], pl.DT_FP32]):
    with pl.section_vector():
        for i in pl.range(7):
            out[0, i] = pl.scalar_abs(src[0, i])
            out[1, i] = abs(src[1, i])


# =============================================================================
# Test 2: INT32 标量 — pl.scalar_abs 与裸内置 abs()
# =============================================================================
@pl.jit()
def kernel_scalar_abs_int32(src: pl.Tensor[[2, 8], pl.DT_INT32], out: pl.Tensor[[2, 8], pl.DT_INT32]):
    with pl.section_vector():
        for i in pl.range(8):
            out[0, i] = pl.scalar_abs(src[0, i])
            out[1, i] = abs(src[1, i])


# =============================================================================
# Test 3: FP16 标量 — pl.scalar_abs 与裸内置 abs()
# =============================================================================
@pl.jit()
def kernel_scalar_abs_fp16(src: pl.Tensor[[1], pl.DT_FP16], out: pl.Tensor[[2], pl.DT_FP16]):
    with pl.section_vector():
        out[0] = pl.scalar_abs(src[0])
        out[1] = abs(src[0])


# =============================================================================
# Test 4: BF16 标量 — pl.scalar_abs 与裸内置 abs()
# =============================================================================
@pl.jit()
def kernel_scalar_abs_bf16(src: pl.Tensor[[1], pl.DT_BF16], out: pl.Tensor[[2], pl.DT_BF16]):
    with pl.section_vector():
        out[0] = pl.scalar_abs(src[0])
        out[1] = abs(src[0])


@pytest.mark.soc("950", "960")
def test_scalar_abs_fp32_runs_on_device():
    """测试 pl.scalar_abs 与裸内置 abs() 对运行期 FP32 标量取绝对值的场景。

    测试目的:验证 pl.scalar_abs(x) 在 NPU 上对 FP32 标量操作数求绝对值的正确性,
             以及裸内置 abs(x) 与 pl.scalar_abs(x) 等价。
    输入参数:src 为 [2, 7] FP32,第 0 行 7 个运行期值(含负值、零与小数）,交给 pl.scalar_abs；
             第 1 行 7 个运行期值交给内置 abs()；
             out 为 [2, 7] FP32,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.abs(src) 对应行(rtol=0, atol=0 逐元素精确比较验证）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [-12.5, 2.25, 0.0, 64.0, -64.0, 1e-6, -1e-6],
        [3.5, -3.5, 8.0, -1.0, 2.0, -100.0, 100.0],
    ], dtype=torch.float32)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_abs_fp32[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.abs(src).cpu(), rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_scalar_abs_int32_runs_on_device():
    """测试 pl.scalar_abs 与裸内置 abs() 对运行期 INT32 标量取绝对值的场景。

    测试目的:验证 pl.scalar_abs(x) 在 NPU 上对 INT32 标量操作数求绝对值的正确性,
             覆盖 INT32 边界值(-2147483647,绝对值为 2147483647 不溢出）；
             以及裸内置 abs(x) 与 pl.scalar_abs(x) 等价。
    输入参数:src 为 [2, 8] INT32,第 0 行、第 1 行各 8 个运行期值(含边界值）；
             out 为 [2, 8] INT32,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.abs(src) 对应行(rtol=0, atol=0 逐元素精确比较验证）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [-2, 2, 0, -100, 1000, -1000, -5, 5],
        [-2147483647, 2147483647, -1, 100000, -100000, 7, -7, 3000],
    ], dtype=torch.int32)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_abs_int32[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.abs(src).cpu(), rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_scalar_abs_fp16_runs_on_device():
    """测试 pl.scalar_abs 与裸内置 abs() 对运行期 FP16 标量取绝对值的场景。

    测试目的:验证 pl.scalar_abs(x) 在 NPU 上对 FP16 标量操作数求绝对值的正确性,
             以及裸内置 abs(x) 与 pl.scalar_abs(x) 等价。
    输入参数:src 为 [1] FP16,1 个运行期值同时交给 pl.scalar_abs 与内置 abs()；
             out 为 [2] FP16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两个元素均等于 torch.abs(src[0])(rtol=0, atol=0 逐元素精确比较验证）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([-12.5], dtype=torch.float16, device=ST_DEVICE)
    out = torch.empty(2, dtype=torch.float16, device=ST_DEVICE)

    kernel_scalar_abs_fp16[None, 1](src, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.abs(src).cpu().repeat(2), rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_scalar_abs_bf16_runs_on_device():
    """测试 pl.scalar_abs 与裸内置 abs() 对运行期 BF16 标量取绝对值的场景。

    测试目的:验证 pl.scalar_abs(x) 在 NPU 上对 BF16 标量操作数求绝对值的正确性,
             以及裸内置 abs(x) 与 pl.scalar_abs(x) 等价。
    输入参数:src 为 [1] BF16,1 个运行期值同时交给 pl.scalar_abs 与内置 abs()；
             out 为 [2] BF16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两个元素均等于 torch.abs(src[0])(rtol=0, atol=0 逐元素精确比较验证）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([-3.5], dtype=torch.bfloat16, device=ST_DEVICE)
    out = torch.empty(2, dtype=torch.bfloat16, device=ST_DEVICE)

    kernel_scalar_abs_bf16[None, 1](src, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.abs(src).cpu().repeat(2), rtol=0, atol=0)
