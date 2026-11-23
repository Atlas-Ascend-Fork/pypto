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
"""
MatmulL0C2UB 融合算子测试脚本
算子公式：C = A @ B + C（逐元素加）
支持 pytest 和直接执行两种模式
"""

import os
import struct

import numpy as np
import pytest
from testcase.matmul_l0c2ub_test_case import L0C2UB_TESTS, MatmulL0C2UBConfig
import torch
import torch_npu

import pypto


def fixpipe_mask_scale(scale_input):
    """Convert FP32 scale values to the precision consumed by fixpipe."""
    mask = 0xFFFFE000
    if isinstance(scale_input, torch.Tensor):
        tensor_data = scale_input.numpy().view(np.uint32) & mask
        golden_scale = tensor_data.view(np.float32)
        return torch.from_numpy(tensor_data.astype(np.uint64)), torch.from_numpy(golden_scale)

    scale_bits = struct.unpack("I", struct.pack("f", np.float32(scale_input)))[0] & mask
    golden_scale = struct.unpack("f", struct.pack("I", scale_bits))[0]
    return golden_scale, np.float32(golden_scale)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_l0c2ub_kernel_basic(
    a_tensor: pypto.Tensor(),
    b_tensor: pypto.Tensor(),
    c_tensor: pypto.Tensor(),
    out_tensor: pypto.Tensor(),
    config: MatmulL0C2UBConfig,
):
    """基础 kernel：matmul → add，支持 PerTensor scale value。

    适用场景：无量化或 PerTensor 量化。
    """
    m, k, n = config.shape
    m_view, n_view = config.view_shape

    pypto.set_pass_options(sg_set_scope=10000)
    pypto.set_cube_tile_shapes(*config.tile_shape)

    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view

    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if config.a_trans:
                a_view = a_tensor[:, m_idx * m_view:m_idx * m_view + m_view]
            else:
                a_view = a_tensor[m_idx * m_view:m_idx * m_view + m_view, :]

            if config.b_trans:
                b_view = b_tensor[n_idx * n_view:n_idx * n_view + n_view, :]
            else:
                b_view = b_tensor[:, n_idx * n_view:n_idx * n_view + n_view]

            if config.quant_type == 1:
                mat_result = pypto.matmul(
                    a_view,
                    b_view,
                    out_dtype=config.out_dtype,
                    a_trans=config.a_trans,
                    b_trans=config.b_trans,
                    extend_params={"scale": config.scale_value},
                )
            else:
                mat_result = pypto.matmul(
                    a_view,
                    b_view,
                    out_dtype=config.out_dtype,
                    a_trans=config.a_trans,
                    b_trans=config.b_trans,
                )

            c_view = c_tensor[
                m_idx * m_view:m_idx * m_view + m_view,
                n_idx * n_view:n_idx * n_view + n_view,
            ]

            pypto.set_vec_tile_shapes(*config.vec_tile_shape)
            result = pypto.add(mat_result, c_view)

            out_tensor[
                m_idx * m_view:m_idx * m_view + m_view,
                n_idx * n_view:n_idx * n_view + n_view,
            ] = result
    pypto.set_pass_options(sg_set_scope=-1)


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
def matmul_l0c2ub_kernel_perchannel(
    a_tensor: pypto.Tensor(),
    b_tensor: pypto.Tensor(),
    scale_tensor: pypto.Tensor(),
    c_tensor: pypto.Tensor(),
    out_tensor: pypto.Tensor(),
    config: MatmulL0C2UBConfig,
):
    """PerChannel kernel：matmul → L0C2UB quant → add。"""
    m, k, n = config.shape
    m_view, n_view = config.view_shape

    pypto.set_pass_options(sg_set_scope=10000)
    pypto.set_cube_tile_shapes(*config.tile_shape)

    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view

    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if config.a_trans:
                a_view = a_tensor[:, m_idx * m_view:m_idx * m_view + m_view]
            else:
                a_view = a_tensor[m_idx * m_view:m_idx * m_view + m_view, :]

            if config.b_trans:
                b_view = b_tensor[n_idx * n_view:n_idx * n_view + n_view, :]
            else:
                b_view = b_tensor[:, n_idx * n_view:n_idx * n_view + n_view]

            scale_view = scale_tensor[:, n_idx * n_view:n_idx * n_view + n_view]
            mat_result = pypto.matmul(
                a_view,
                b_view,
                out_dtype=config.out_dtype,
                a_trans=config.a_trans,
                b_trans=config.b_trans,
                extend_params={"scale_tensor": scale_view},
            )

            c_view = c_tensor[
                m_idx * m_view:m_idx * m_view + m_view,
                n_idx * n_view:n_idx * n_view + n_view,
            ]

            pypto.set_vec_tile_shapes(*config.vec_tile_shape)
            result = pypto.add(mat_result, c_view)

            out_tensor[
                m_idx * m_view:m_idx * m_view + m_view,
                n_idx * n_view:n_idx * n_view + n_view,
            ] = result
    pypto.set_pass_options(sg_set_scope=-1)


def matmul_l0c2ub_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    config: MatmulL0C2UBConfig,
    scale=None,
) -> torch.Tensor:
    """Golden 参考实现（纯 PyTorch）

    Args:
        a: 左矩阵 A
        b: 右矩阵 B
        c: 加数矩阵 C
        config: 矩阵转置、类型等相关信息

    Returns:
        golden: A@B + C
    """
    a_trans = config.a_trans
    b_trans = config.b_trans
    a_input = a.T if a_trans else a
    b_input = b.T if b_trans else b

    accum_dtype = torch.int32 if config.a_dtype == pypto.DT_INT8 else torch.float32
    mat_result = torch.matmul(a_input.to(accum_dtype), b_input.to(accum_dtype))
    if config.quant_type != 0:
        mat_result = mat_result * scale
        if config.out_dtype == pypto.DT_INT8:
            mat_result = torch.round(mat_result).clamp(-128, 127).to(torch.int8)
        else:
            mat_result = mat_result.to(torch.float16)
    c_input = c.to(accum_dtype)
    result = mat_result + c_input
    return result


def run_matmul_l0c2ub_test(case: dict):
    """执行单个测试用例
    basic_kernel
    """
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    config = MatmulL0C2UBConfig.from_test_case(case)

    m, k, n = config.shape
    a_shape = [k, m] if config.a_trans else [m, k]
    b_shape = [n, k] if config.b_trans else [k, n]
    c_shape = [m, n]

    a_dtype = MatmulL0C2UBConfig.get_torch_dtype(case["a_dtype"])
    b_dtype = MatmulL0C2UBConfig.get_torch_dtype(case["b_dtype"])
    c_dtype = MatmulL0C2UBConfig.get_torch_dtype(case["c_dtype"])

    if a_dtype == torch.int8:
        a_tensor_cpu = torch.randint(-5, 6, a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.randint(-5, 6, b_shape, dtype=b_dtype)
    elif config.quant_type != 0:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_dtype) * 2 - 1
        b_tensor_cpu = torch.rand(b_shape, dtype=b_dtype) * 2 - 1
    else:
        a_tensor_cpu = torch.rand(a_shape, dtype=a_dtype)
        b_tensor_cpu = torch.rand(b_shape, dtype=b_dtype)

    if config.quant_type != 0:
        c_tensor_cpu = torch.zeros(c_shape, dtype=c_dtype)
    else:
        c_tensor_cpu = torch.rand(c_shape, dtype=c_dtype)

    scale_input = None
    golden_scale = None
    if config.quant_type == 1:
        scale_input, golden_scale = fixpipe_mask_scale(0.25)
        config.scale_value = scale_input
    elif config.quant_type == 2:
        scale_input, golden_scale = fixpipe_mask_scale(torch.full((1, n), 0.25, dtype=torch.float32))

    golden = matmul_l0c2ub_golden(
        a_tensor_cpu, b_tensor_cpu, c_tensor_cpu, config, scale=golden_scale
    ).to(c_dtype)

    a_tensor = a_tensor_cpu.to(f"npu:{device_id}")
    b_tensor = b_tensor_cpu.to(f"npu:{device_id}")
    c_tensor = c_tensor_cpu.to(f"npu:{device_id}")
    c_out_tensor = torch.zeros(c_shape, dtype=c_dtype, device=f"npu:{device_id}")

    if case.get("a_format") == "NZ":
        a_tensor = torch_npu.npu_format_cast(a_tensor, 29)
    if case.get("b_format") == "NZ":
        b_tensor = torch_npu.npu_format_cast(b_tensor, 29)

    if config.quant_type == 2:
        if config.out_dtype == pypto.DT_INT8:
            scale_tensor = torch_npu.npu_trans_quant_param(golden_scale.to(f"npu:{device_id}"))
        else:
            scale_tensor = scale_input.to(f"npu:{device_id}")
        matmul_l0c2ub_kernel_perchannel(
            a_tensor, b_tensor, scale_tensor, c_tensor, c_out_tensor, config=config
        )
    else:
        matmul_l0c2ub_kernel_basic(a_tensor, b_tensor, c_tensor, c_out_tensor, config=config)

    atol, rtol = MatmulL0C2UBConfig.get_tolerance(case["c_dtype"])
    assert torch.allclose(c_out_tensor.cpu(), golden.cpu(), atol=atol, rtol=rtol), (
        f"Test case {case['id']} ({case['name']}) failed"
    )


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=pytest.mark.soc(*case["products"])) for case in L0C2UB_TESTS]
)
def test_matmul_l0c2ub(case: dict):
    """pytest 参数化测试"""
    run_matmul_l0c2ub_test(case)


def run_l0c2ub_demo(run_mode):
    """MatmulL0C2UB 算子演示（类似 run_matmul_demo）

    简化版演示，硬编码参数：
    - shape: [256, 256, 256]
    - dtype: FP16
    - device: npu:0
    - vec_tile_shape: [128, 128]
    """
    m_size, k_size, n_size = 256, 256, 256
    m_view_size, n_view_size = 128, 128
    vec_m_tile_size, vec_n_tile_size = 128, 128

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1}, runtime_options={"run_mode": mode}
    )
    def l0c2ub_demo_kernel(
        a: pypto.Tensor([], pypto.DT_FP16),
        b: pypto.Tensor([], pypto.DT_FP16),
        c: pypto.Tensor([], pypto.DT_FP16),
        out: pypto.Tensor([], pypto.DT_FP16),
    ):
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

        m_loop = (m_size + m_view_size - 1) // m_view_size
        n_loop = (n_size + n_view_size - 1) // n_view_size

        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
                a_view = a[m_idx * m_view_size:m_idx * m_view_size + m_view_size, :]
                b_view = b[:, n_idx * n_view_size:n_idx * n_view_size + n_view_size]

                mat_result = pypto.matmul(a_view, b_view, pypto.DT_FP16)

                c_view = c[
                    m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                    n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                ]

                pypto.set_vec_tile_shapes(vec_m_tile_size, vec_n_tile_size)
                result = pypto.add(mat_result, c_view)

                out[
                    m_idx * m_view_size:m_idx * m_view_size + m_view_size,
                    n_idx * n_view_size:n_idx * n_view_size + n_view_size,
                ] = result

    a = torch.randn([m_size, k_size], dtype=torch.float16, device="npu:0")
    b = torch.randn([k_size, n_size], dtype=torch.float16, device="npu:0")
    c = torch.randn([m_size, n_size], dtype=torch.float16, device="npu:0")
    out = torch.empty(m_size, n_size, dtype=torch.float16, device="npu:0")
    l0c2ub_demo_kernel(a, b, c, out)


if __name__ == "__main__":
    run_l0c2ub_demo("npu")
