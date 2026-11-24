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

"""pl.floor / pl.ceildiv NPU 泛化测试 — 标量级向下取整与向上取整除法。

参考文献:
  docs/zh/api/pro_api/Utils-API/python_syntax_sugar/{floor,ceildiv}.md

被测 API:
  pl.floor(x)     — 取不大于 x 的最大整数;整数操作数取自身,浮点操作数向下取整
  pl.ceildiv(a,b) — 计算 ceil(a / b),参数类型规则与整除运算符 // 一致

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
# Kernel 1: FP32 标量 — floor (运行期值)
# =============================================================================
@pl.jit()
def kernel_scalar_floor_fp32(src: pl.Tensor[[2, 7], pl.DT_FP32], out: pl.Tensor[[2, 7], pl.DT_FP32]):
    with pl.section_vector():
        for i in pl.range(7):
            out[0, i] = pl.floor(src[0, i])
            out[1, i] = pl.floor(src[1, i])


# =============================================================================
# Kernel 2: INT32 标量 — ceildiv (运行期除数 / 常量除数)
# =============================================================================
@pl.jit()
def kernel_ceildiv_int32(src: pl.Tensor[[2, 8], pl.DT_INT32], out: pl.Tensor[[2, 8], pl.DT_INT32]):
    with pl.section_vector():
        for i in pl.range(8):
            out[0, i] = pl.ceildiv(src[0, i], 3)
            out[1, i] = pl.ceildiv(src[0, i], src[1, i])


# =============================================================================
# Kernel 3: FP16 标量 — floor (运行期值)
# =============================================================================
@pl.jit()
def kernel_scalar_floor_fp16(src: pl.Tensor[[2, 7], pl.DT_FP16], out: pl.Tensor[[2, 7], pl.DT_FP16]):
    with pl.section_vector():
        for i in pl.range(7):
            out[0, i] = pl.floor(src[0, i])
            out[1, i] = pl.floor(src[1, i])


# =============================================================================
# Kernel 4: BF16 标量 — floor (运行期值)
# =============================================================================
@pl.jit()
def kernel_scalar_floor_bf16(src: pl.Tensor[[2, 7], pl.DT_BF16], out: pl.Tensor[[2, 7], pl.DT_BF16]):
    with pl.section_vector():
        for i in pl.range(7):
            out[0, i] = pl.floor(src[0, i])
            out[1, i] = pl.floor(src[1, i])


# =============================================================================
# Kernel 5: FP16 标量 — ceildiv (运行期被除数 / 除数)
# =============================================================================
@pl.jit()
def kernel_ceildiv_fp16(src: pl.Tensor[[2, 8], pl.DT_FP16], out: pl.Tensor[[2, 8], pl.DT_FP16]):
    with pl.section_vector():
        for i in pl.range(8):
            out[0, i] = pl.ceildiv(src[0, i], src[1, i])
            out[1, i] = pl.ceildiv(src[1, i], src[0, i])


# =============================================================================
# Kernel 6: BF16 标量 — ceildiv (运行期被除数 / 除数)
# =============================================================================
@pl.jit()
def kernel_ceildiv_bf16(src: pl.Tensor[[2, 8], pl.DT_BF16], out: pl.Tensor[[2, 8], pl.DT_BF16]):
    with pl.section_vector():
        for i in pl.range(8):
            out[0, i] = pl.ceildiv(src[0, i], src[1, i])
            out[1, i] = pl.ceildiv(src[1, i], src[0, i])


@pytest.mark.soc("950", "960")
def test_scalar_floor_fp32_runs_on_device():
    """测试 pl.floor 对运行期 FP32 标量向下取整的场景。

    测试目的:验证 pl.floor(x) 在 NPU 上对 FP32 标量操作数求 floor(x) 的正确性,
             覆盖正数、负数、零与小数边界。
    输入参数:src 为 [2, 7] FP32,两行各 7 个运行期值(含小数、负值与零）;
             out 为 [2, 7] FP32,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.floor(src) 对应行(rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [1.5, -1.5, 2.0, -2.0, 0.0, 0.3, -0.3],
        [3.0, -3.0, 0.7, -0.7, 64.0, -64.0, 1.0],
    ], dtype=torch.float32)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_floor_fp32[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), torch.floor(src), rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_ceildiv_int32_runs_on_device():
    """测试 pl.ceildiv 对运行期 INT32 标量向上取整除法的场景。

    测试目的:验证 pl.ceildiv(lhs, rhs) 在 NPU 上求 ceil(lhs / rhs) 的正确性,
             覆盖常量除数与运行期除数以及负被除数。
    输入参数:src 为 [2, 8] INT32,第 0 行 8 个被除数(含负值),第 1 行 8 个非零除数;
             out 为 [2, 8] INT32,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 第 0 行等于 torch.ceil(src0/3),第 1 行等于 torch.ceil(src0/src1)
             (rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [1, 2, 3, 4, 5, 6, -1, -2],
        [1, 2, 3, 4, 5, 6, 7, 3],
    ], dtype=torch.int32)
    out = torch.empty(2, 8, dtype=torch.int32, device=ST_DEVICE)

    kernel_ceildiv_int32[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    expected = torch.stack([
        torch.ceil(src[0].to(torch.float32) / 3.0).to(torch.int32),
        torch.ceil(src[0].to(torch.float32) / src[1].to(torch.float32)).to(torch.int32),
    ])
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_scalar_floor_fp16_runs_on_device():
    """测试 pl.floor 对运行期 FP16 标量向下取整的场景。

    测试目的:验证 pl.floor(x) 在 NPU 上对 FP16 标量操作数求 floor(x) 的正确性;
              FP16 没有原生 floor 重载,设备侧经 FP32 取整后回写 FP16。
    输入参数:src 为 [2, 7] FP16,两行各 7 个运行期值(含小数、负值与零）;
              out 为 [2, 7] FP16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.floor(src.float()).to(torch.float16) 对应行
              (rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [1.5, -1.5, 2.0, -2.0, 0.0, 0.3, -0.3],
        [3.0, -3.0, 0.7, -0.7, 64.0, -64.0, 1.0],
    ], dtype=torch.float16)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_floor_fp16[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    expected = torch.floor(src.float()).to(torch.float16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_scalar_floor_bf16_runs_on_device():
    """测试 pl.floor 对运行期 BF16 标量向下取整的场景。

    测试目的:验证 pl.floor(x) 在 NPU 上对 BF16 标量操作数求 floor(x) 的正确性;
              BF16 没有原生 floor 重载,设备侧经 FP32 取整后按 RNE 回写 BF16。
    输入参数:src 为 [2, 7] BF16,两行各 7 个运行期值(含小数、负值与零）;
              out 为 [2, 7] BF16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 两行分别等于 torch.floor(src.float()).to(torch.bfloat16) 对应行
              (rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [1.5, -1.5, 2.0, -2.0, 0.0, 0.3, -0.3],
        [3.0, -3.0, 0.7, -0.7, 64.0, -64.0, 1.0],
    ], dtype=torch.bfloat16)
    out = torch.empty_like(src, device=ST_DEVICE)

    kernel_scalar_floor_bf16[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    expected = torch.floor(src.float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_ceildiv_fp16_runs_on_device():
    """测试 pl.ceildiv 对运行期 FP16 标量向上取整除法的场景。

    测试目的:验证 pl.ceildiv(lhs, rhs) 在 NPU 上对 FP16 操作数求 ceil(lhs / rhs) 的正确性;
              FP16 结果经 FP32 计算后回写,覆盖正负被除数、正负除数与整除/非整除。
    输入参数:src 为 [2, 8] FP16,第 0 行 8 个非零被除数(含负值),第 1 行 8 个非零除数;
              out 为 [2, 8] FP16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 第 0 行等于 ceil(src0/src1),第 1 行等于 ceil(src1/src0)
              (rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [7.0, 8.0, -7.0, -8.0, 9.0, -9.0, 1.0, -1.0],
        [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 3.0, 3.0],
    ], dtype=torch.float16)
    out = torch.empty(2, 8, dtype=torch.float16, device=ST_DEVICE)

    kernel_ceildiv_fp16[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    lhs, rhs = src[0].float(), src[1].float()
    expected = torch.stack([torch.ceil(lhs / rhs), torch.ceil(rhs / lhs)]).to(torch.float16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.soc("950", "960")
def test_ceildiv_bf16_runs_on_device():
    """测试 pl.ceildiv 对运行期 BF16 标量向上取整除法的场景。

    测试目的:验证 pl.ceildiv(lhs, rhs) 在 NPU 上对 BF16 操作数求 ceil(lhs / rhs) 的正确性;
              BF16 结果经 FP32 计算后按 RNE 回写,覆盖正负被除数、正负除数与整除/非整除。
    输入参数:src 为 [2, 8] BF16,第 0 行 8 个非零被除数(含负值),第 1 行 8 个非零除数;
              out 为 [2, 8] BF16,分配在 NPU 上,由 kernel 写入结果。
    预期行为:out 第 0 行等于 ceil(src0/src1),第 1 行等于 ceil(src1/src0)
              (rtol=0, atol=0 逐元素精确比较）。
    """
    torch.npu.set_device(ST_DEVICE)
    src = torch.tensor([
        [7.0, 8.0, -7.0, -8.0, 9.0, -9.0, 1.0, -1.0],
        [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 3.0, 3.0],
    ], dtype=torch.bfloat16)
    out = torch.empty(2, 8, dtype=torch.bfloat16, device=ST_DEVICE)

    kernel_ceildiv_bf16[None, 1](src.to(ST_DEVICE), out)
    torch.npu.synchronize()

    lhs, rhs = src[0].float(), src[1].float()
    expected = torch.stack([torch.ceil(lhs / rhs), torch.ceil(rhs / lhs)]).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
