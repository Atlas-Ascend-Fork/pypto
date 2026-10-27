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
Test matmul codegen - common functions for Kirin9030 and KirinX90
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from kirin.common import check_atol_rtol, check_nan, compare_cos, generated_cpp_contains
import pypto

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "tools" / "scripts"))
from lite_npu_quant import decode_quant_param, trans_quant_param

DTYPE_MAP = {
    torch.float16: pypto.DT_FP16,
    torch.float32: pypto.DT_FP32,
    torch.int8: pypto.DT_INT8,
    torch.int32: pypto.DT_INT32,
}


def get_matmul_kernel(
    soc_version,
    name,
    in_dtype,
    out_dtype,
    cube_tile_shape,
    has_bias=False,
    bias_shape=None,
    a_trans=False,
    b_trans=False,
    bias_dtype_override=None,
    quant_scale_shape=None,
):
    # kirin9050 宏指令量化：scale_tensor 复用为 quant_pre（per-channel, shape (1, n)）
    if quant_scale_shape is not None:
        if has_bias:
            # 宏指令 bias 走 BT 表：int32 输出用 s32 bias，fp16 输出用 fp16 bias
            # （通用 TMATMUL_BIAS 要求 bias dtype == 输出 dtype；TMovToBt 只支持 int32/half）。
            quant_bias_dtype = bias_dtype_override if bias_dtype_override is not None else pypto.DT_INT32

            @pypto.frontend.jit(
                codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
            )
            def matmul_kernel(
                a: pypto.Tensor([...], in_dtype),
                b: pypto.Tensor([...], in_dtype),
                bias: pypto.Tensor([...], quant_bias_dtype),
                scale: pypto.Tensor([...], pypto.DT_INT64),
                c: pypto.Tensor([...], out_dtype),
            ):
                pypto.set_cube_tile_shapes(cube_tile_shape[0], cube_tile_shape[1], cube_tile_shape[2])
                extend_params = {"bias_tensor": bias, "scale_tensor": scale}
                c[:] = pypto.matmul(
                    a, b, out_dtype, a_trans=a_trans, b_trans=b_trans, c_matrix_nz=False, extend_params=extend_params
                )

            matmul_kernel.__name__ = name
            return matmul_kernel

        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def matmul_kernel(
            a: pypto.Tensor([...], in_dtype),
            b: pypto.Tensor([...], in_dtype),
            scale: pypto.Tensor([...], pypto.DT_INT64),
            c: pypto.Tensor([...], out_dtype),
        ):
            pypto.set_cube_tile_shapes(cube_tile_shape[0], cube_tile_shape[1], cube_tile_shape[2])
            extend_params = {"scale_tensor": scale}
            c[:] = pypto.matmul(
                a, b, out_dtype, a_trans=a_trans, b_trans=b_trans, c_matrix_nz=False, extend_params=extend_params
            )

        matmul_kernel.__name__ = name
        return matmul_kernel

    # X90 硬件要求 bias 以 32 位 fp32 格式传入（低 16 位有效，高 16 位无效）
    bias_dtype = (
        bias_dtype_override
        if bias_dtype_override is not None
        else (pypto.DT_FP32 if (soc_version == "KirinX90" and out_dtype == pypto.DT_FP16) else out_dtype)
    )
    if has_bias:

        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def matmul_kernel(
            a: pypto.Tensor([...], in_dtype),
            b: pypto.Tensor([...], in_dtype),
            bias: pypto.Tensor([...], bias_dtype),
            c: pypto.Tensor([...], out_dtype),
        ):
            pypto.set_cube_tile_shapes(cube_tile_shape[0], cube_tile_shape[1], cube_tile_shape[2])
            extend_params = {"bias_tensor": bias}
            c[:] = pypto.matmul(
                a, b, out_dtype, a_trans=a_trans, b_trans=b_trans, c_matrix_nz=False, extend_params=extend_params
            )

        matmul_kernel.__name__ = name
        return matmul_kernel
    else:

        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def matmul_kernel(
            a: pypto.Tensor([...], in_dtype),
            b: pypto.Tensor([...], in_dtype),
            c: pypto.Tensor([...], out_dtype),
        ):
            pypto.set_cube_tile_shapes(cube_tile_shape[0], cube_tile_shape[1], cube_tile_shape[2])
            c[:] = pypto.matmul(a, b, out_dtype, a_trans=a_trans, b_trans=b_trans)

        matmul_kernel.__name__ = name
        return matmul_kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # torch_in_dtype: torch data type for input tensor a
    # torch_out_dtype: torch data type for output tensor c
    # pypto_in_dtype: pypto data type for input tensor a
    # pypto_out_dtype: pypto data type for output tensor c
    # shape_a: input tensor a shape
    # shape_b: input tensor b shape
    # shape_c: output tensor c shape
    # cube_tile_shape: cube tile shapes for matmul
    # has_bias: whether to use bias
    # bias_shape: bias tensor shape (or None)
    # a_trans: whether to transpose a
    # b_trans: whether to transpose b
    # marks: pytest marks
    pytest.param(
        "matmul_fp16_001",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (16, 16),
        (16, 16),
        (16, 16),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="001",
    ),
    pytest.param(
        "matmul_fp16_002",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 130),
        (130, 2),
        (2, 2),
        ([16, 16], [32, 32], [16, 16]),
        True,
        (1, 2),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "matmul_fp16_003",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (30, 150),
        (150, 60),
        (30, 60),
        ([32, 32], [32, 32], [64, 64]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "matmul_fp16_004",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (128, 32),
        (32, 100),
        (128, 100),
        ([64, 64], [32, 32], [64, 64]),
        True,
        (1, 100),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "matmul_fp16_005",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (40, 130),
        (40, 64),
        (130, 64),
        ([32, 32], [32, 32], [64, 64]),
        False,
        None,
        True,
        False,
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "matmul_fp16_006",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (5, 80, 64),
        (5, 64, 1),
        (5, 80, 1),
        ([32, 32], [64, 64], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "matmul_fp16_007",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (16, 1, 64),
        (16, 64, 64),
        (16, 1, 64),
        ([16, 16], [64, 64], [16, 16]),
        False,
        None,
        False,
        True,
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "matmul_fp16_008",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 16, 129, 64),
        (2, 16, 64, 35),
        (2, 16, 129, 35),
        ([16, 16], [64, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[],
        id="008",
    ),
    pytest.param(
        "matmul_fp16_009",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 8, 80, 160),
        (2, 8, 160, 30),
        (2, 8, 80, 30),
        ([64, 64], [64, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "matmul_fp16_010",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 4, 60, 80),
        (1, 4, 32, 60),
        (1, 4, 80, 32),
        ([64, 64], [32, 32], [32, 32]),
        False,
        None,
        True,
        True,
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "matmul_s8s8_001",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (16, 32),
        (32, 16),
        (16, 16),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "matmul_s8s8_002",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (16, 32),
        (32, 16),
        (16, 16),
        ([32, 32], [32, 32], [32, 32]),
        True,
        (1, 16),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "matmul_s8s8_003",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (16, 32, 64),
        (16, 64, 16),
        (16, 32, 16),
        ([32, 32], [64, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "matmul_s8s8_004",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (2, 16, 32, 64),
        (2, 16, 64, 32),
        (2, 16, 64, 64),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        True,
        True,
        marks=[],
        id="014",
    ),
    pytest.param(
        "matmul_s8s8_gemv_001",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (1, 16),
        (16, 16),
        (1, 16),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "matmul_s8s8_gemv_002",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (1, 20),
        (32, 20),
        (1, 32),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        True,
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "matmul_s8s8_gemv_003",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (3, 1, 130),
        (3, 130, 32),
        (3, 1, 32),
        ([16, 16], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "matmul_s8s8_gemv_004",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (1024, 1),
        (1024, 32),
        (1, 32),
        ([32, 32], [64, 64], [32, 32]),
        True,
        (1, 32),
        True,
        False,
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "matmul_s8s8_gemv_005",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (3, 1, 32),
        (3, 64, 32),
        (3, 1, 64),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        True,
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "matmul_s8s8_gemv_006",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (2, 3, 1, 32),
        (2, 3, 32, 33),
        (2, 3, 1, 33),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "matmul_s8s8_gemv_007",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (2, 2, 1, 1024),
        (2, 2, 1024, 32),
        (2, 2, 1, 32),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[],
        id="021",
    ),
    pytest.param(
        "matmul_s8s8_gemv_008",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (3, 1, 130),
        (3, 130, 32),
        (3, 1, 32),
        ([1, 1], [256, 256], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="022",
    ),
    pytest.param(
        "matmul_s8s8_gemv_009",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (1, 130),
        (130, 64),
        (1, 64),
        ([1, 1], [512, 512], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="023",
    ),
    pytest.param(
        "matmul_s8s8_gemv_010",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (4, 1, 16),
        (4, 16, 32),
        (4, 1, 32),
        ([1, 1], [256, 256], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="024",
    ),
    pytest.param(
        "matmul_s8s8_gemv_011",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (1, 128),
        (128, 32),
        (1, 32),
        ([32, 64], [32, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="025",
    ),
    pytest.param(
        "matmul_s8s8_gemv_012",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (3, 1, 128),
        (3, 128, 32),
        (3, 1, 32),
        ([32, 64], [32, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="026",
    ),
    pytest.param(
        "matmul_s8s8_gemv_013",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (2, 2, 1, 128),
        (2, 2, 128, 32),
        (2, 2, 1, 32),
        ([32, 64], [32, 64], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="027",
    ),
    pytest.param(
        "matmul_fp16_gemv_001",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 16),
        (16, 16),
        (1, 16),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="028",
    ),
    pytest.param(
        "matmul_fp16_gemv_002",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (3, 1, 32),
        (3, 32, 32),
        (3, 1, 32),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="029",
    ),
    pytest.param(
        "matmul_fp16_gemv_003",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (4, 1, 130),
        (4, 130, 16),
        (4, 1, 16),
        ([16, 16], [32, 32], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="030",
    ),
    pytest.param(
        "matmul_fp16_gemv_004",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 16),
        (16, 130),
        (1, 130),
        ([16, 16], [16, 16], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="031",
    ),
    pytest.param(
        "matmul_fp16_gemv_005",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 2, 1, 130),
        (2, 2, 130, 130),
        (2, 2, 1, 130),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="032",
    ),
    pytest.param(
        "matmul_fp16_gemv_006",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 1),
        (1, 16),
        (1, 16),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="033",
    ),
    pytest.param(
        "matmul_fp16_gemv_007",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 16),
        (16, 1),
        (1, 1),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="034",
    ),
    pytest.param(
        "matmul_fp16_gemv_008",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (3, 1, 17),
        (3, 17, 17),
        (3, 1, 17),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="035",
    ),
    pytest.param(
        "matmul_fp16_gemv_009",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 8),
        (8, 8),
        (1, 8),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="036",
    ),
    pytest.param(
        "matmul_fp16_gemv_010",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 3, 1, 64),
        (2, 3, 64, 64),
        (2, 3, 1, 64),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[],
        id="037",
    ),
    pytest.param(
        "matmul_fp16_gemv_011",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 150),
        (150, 60),
        (1, 60),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="038",
    ),
    pytest.param(
        "matmul_fp16_gemv_012",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (3, 1, 1024),
        (3, 1024, 32),
        (3, 1, 32),
        ([32, 32], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="039",
    ),
    pytest.param(
        "matmul_fp16_gemv_013",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 16),
        (16, 1024),
        (1, 1024),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="040",
    ),
    pytest.param(
        "matmul_fp16_gemv_014",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (4, 1, 64),
        (4, 64, 64),
        (4, 1, 64),
        ([64, 64], [32, 32], [32, 32]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="041",
    ),
    pytest.param(
        "matmul_fp16_gemv_015",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (3, 1, 32),
        (3, 64, 32),
        (3, 1, 64),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        True,
        marks=[pytest.mark.skip()],
        id="042",
    ),
    pytest.param(
        "matmul_fp16_gemv_016",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 2, 64, 1),
        (2, 2, 64, 16),
        (2, 2, 1, 16),
        ([16, 16], [64, 64], [16, 16]),
        False,
        None,
        True,
        False,
        marks=[pytest.mark.skip()],
        id="043",
    ),
    pytest.param(
        "matmul_fp16_gemv_017",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (128, 1),
        (16, 128),
        (1, 16),
        ([16, 16], [128, 128], [16, 16]),
        False,
        None,
        True,
        True,
        marks=[pytest.mark.skip()],
        id="044",
    ),
    pytest.param(
        "matmul_fp16_gemv_018",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (3, 1, 32),
        (3, 32, 32),
        (3, 1, 32),
        ([16, 16], [16, 16], [16, 16]),
        True,
        (1, 32),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="045",
    ),
    pytest.param(
        "matmul_fp16_gemv_019",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (4, 1, 130),
        (4, 130, 16),
        (4, 1, 16),
        ([16, 16], [32, 32], [16, 16]),
        True,
        (1, 16),
        False,
        False,
        marks=[pytest.mark.skip()],
        id="046",
    ),
    pytest.param(
        "matmul_fp16_gemv_020",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (3, 1, 32),
        (3, 64, 32),
        (3, 1, 64),
        ([16, 16], [16, 16], [16, 16]),
        True,
        (1, 64),
        False,
        True,
        marks=[pytest.mark.skip()],
        id="047",
    ),
    pytest.param(
        "matmul_fp16_gemv_021",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 1, 32),
        (2, 32, 32),
        (2, 1, 32),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="048",
    ),
    pytest.param(
        "matmul_fp16_gemv_022",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 2, 1, 16),
        (2, 2, 16, 16),
        (2, 2, 1, 16),
        ([16, 16], [16, 16], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="049",
    ),
    pytest.param(
        "matmul_fp16_gemv_023",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 130),
        (130, 32),
        (1, 32),
        ([1, 1], [256, 256], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="050",
    ),
    pytest.param(
        "matmul_fp16_gemv_024",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (5, 1, 130),
        (5, 130, 64),
        (5, 1, 64),
        ([1, 1], [512, 512], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="051",
    ),
    pytest.param(
        "matmul_fp16_gemv_025",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 16),
        (16, 32),
        (1, 32),
        ([1, 1], [256, 256], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="052",
    ),
    pytest.param(
        "matmul_fp16_gemv_026",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (1, 128),
        (128, 32),
        (1, 32),
        ([32, 64], [32, 64], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="053",
    ),
    pytest.param(
        "matmul_fp16_gemv_027",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (3, 1, 128),
        (3, 128, 32),
        (3, 1, 32),
        ([32, 64], [32, 64], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="054",
    ),
    pytest.param(
        "matmul_fp16_gemv_028",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (2, 2, 1, 128),
        (2, 2, 128, 32),
        (2, 2, 1, 32),
        ([32, 64], [32, 64], [16, 16]),
        False,
        None,
        False,
        False,
        marks=[pytest.mark.skip()],
        id="055",
    ),
]


def run_matmul_test(
    kernels,
    kernel_name,
    torch_in_dtype,
    torch_out_dtype,
    shape_a,
    shape_b,
    shape_c,
    cube_tile_shape,
    has_bias,
    bias_shape,
    a_trans,
    b_trans,
    soc_version,
):
    """Run a single matmul kernel test with given kernels dict."""
    device = "cpu"

    if torch_in_dtype == torch.float16:
        a = torch.rand(shape_a, dtype=torch_in_dtype, device=device)
        b = torch.rand(shape_b, dtype=torch_in_dtype, device=device)
        c = torch.rand(shape_c, dtype=torch_out_dtype, device=device)
    else:
        a = torch.randint(-128, 127, shape_a, dtype=torch.int8)
        b = torch.randint(-128, 127, shape_b, dtype=torch.int8)
        c = torch.zeros(shape_c).to(dtype=torch.int32)

    a_new = a
    if a_trans:
        if a.dim() == 2:
            a_new = a.permute(1, 0)
        elif a.dim() == 3:
            a_new = a.permute(0, 2, 1)
        else:
            a_new = a.permute(0, 1, 3, 2)

    b_new = b
    if b_trans:
        if b.dim() == 2:
            b_new = b.permute(1, 0)
        elif b.dim() == 3:
            b_new = b.permute(0, 2, 1)
        else:
            b_new = b.permute(0, 1, 3, 2)

    if has_bias:
        if torch_in_dtype == torch.float16:
            # X90 硬件要求 bias 以 32 位 fp32 格式传入（低 16 位有效，高 16 位无效）
            bias = torch.rand(bias_shape, dtype=torch_out_dtype, device=device)
            gold_bias = bias
            if soc_version == "KirinX90" and torch_out_dtype == torch.float16:
                bias = bias.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).view(torch.float32)
        else:
            bias = torch.randint(-128, 127, bias_shape, dtype=torch.int32)
            gold_bias = bias

        kernels[kernel_name](a, b, bias, c)

        if torch_out_dtype == torch.int32:
            golden_c = a_new.to(torch.int32) @ b_new.to(torch.int32) + gold_bias
        else:
            golden_c = a_new @ b_new + gold_bias
    else:
        kernels[kernel_name](a, b, c)

        if torch_out_dtype == torch.int32:
            golden_c = a_new.to(torch.int32) @ b_new.to(torch.int32)
        else:
            golden_c = a_new @ b_new

    is_gemv = not a_trans and len(shape_a) >= 2 and shape_a[-2] == 1
    if is_gemv:
        assert generated_cpp_contains("TGEMV"), f"{kernel_name}: generated cpp missing 'TGEMV'"

    check_nan(c, name=kernel_name)
    cos_value = compare_cos(np.array(c.cpu()), np.array(golden_c.cpu()))
    cos_value = abs(cos_value)
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_test_matmul_module(soc_version, test_cases=None):
    """Create a test module for matmul with specified soc_version."""
    if test_cases is None:
        test_cases = TEST_CASES
    kernels = {
        p.values[0]: get_matmul_kernel(
            soc_version,
            p.values[0],
            p.values[3],  # pypto_in_dtype
            p.values[4],  # pypto_out_dtype
            p.values[8],  # cube_tile_shape
            p.values[9],  # has_bias
            p.values[10],  # bias_shape
            p.values[11],  # a_trans
            p.values[12],  # b_trans
        )
        for p in test_cases
    }
    return kernels, lambda: run_matmul_test(
        kernels, None, None, None, None, None, None, None, False, None, False, False, soc_version
    )


# kirin9050 宏指令量化用例（pre-stage quant/dequant，per-channel quant_pre）
# id 在非量化 TEST_CASES（001-055）之后从 056 开始连续编号。
# 组合矩阵：输入 {f16×f16, s8×s8} × 输出 {s32, fp16, s8} × bias {无, 有(s32)}
# 覆盖要点：
#   - 对齐（M=64）与非对齐（M=60）两类 shape；
#   - 切m（m tile < M）、切k（k tile < K）、切n（n tile < N）三种切分；
#   - L1/L0B 切分不一致：cube_tile_shape 中 k[0]（L0B 侧 K 分片）≠ k[1]（L1 侧 K 分片）。
# 字段：kernel_name / torch_in_dtype / torch_out_dtype / pypto_in_dtype / pypto_out_dtype /
#       shape_a / shape_b / shape_c / cube_tile_shape / has_bias / bias_shape / a_trans / b_trans /
#       scale_value(None 表示无 quant_pre) / marks
# 说明：
#   - s32 输出为无量化路径（NoQuant，不需要 quant_pre）；
#   - fp16/s8 输出为 pre-stage quant（DeqF16/ReqS8），需要 per-channel quant_pre；
#   - requant(s8 输出) 的 golden 公式待硬件确认，暂 skip；
#   - s8 输入取小量级控制累加值域，f16 输入随机 [0,1)；
#   - 非对齐 M=60 用例暂 skip，待尾部处理逻辑确认。
QUANT_TEST_CASES = [
    # ---- s8×s8 ---- #
    # 切m：M=64 按 m tile 16 切 4 片
    pytest.param(
        "matmul_quant_s8s8_s32_056",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (64, 128),
        (128, 64),
        (64, 64),
        ([16, 16], [64, 64], [64, 64]),
        False,
        None,
        False,
        False,
        None,
        marks=[],
        id="056",
    ),
    # 切n：N=64 按 n tile 16 切 4 片
    pytest.param(
        "matmul_quant_s8s8_s32_bias_057",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (64, 64),
        (64, 64),
        (64, 64),
        ([64, 64], [64, 64], [32, 32]),
        True,
        (1, 64),
        False,
        False,
        None,
        marks=[],
        id="057",
    ),
    # 非对齐 M=60
    pytest.param(
        "matmul_quant_s8s8_s32_058",
        torch.int8,
        torch.int32,
        pypto.DT_INT8,
        pypto.DT_INT32,
        (60, 64),
        (64, 64),
        (60, 64),
        ([16, 16], [64, 64], [64, 64]),
        False,
        None,
        False,
        False,
        None,
        marks=[pytest.mark.skip(reason="non-aligned M=60 pending verification")],
        id="058",
    ),
    # 切k + L1/L0B 切分不一致：K=128，L0B 侧按 32 切（4 片），L1 侧按 64 切（2 片）
    # m tile 取 16：SIM 的 fixpipe 向量反量化在 m tile=64 的几何下读错 per-channel scale
    pytest.param(
        "matmul_quant_s8s8_fp16_059",
        torch.int8,
        torch.float16,
        pypto.DT_INT8,
        pypto.DT_FP16,
        (64, 128),
        (128, 64),
        (64, 64),
        ([16, 16], [32, 64], [64, 64]),
        False,
        None,
        False,
        False,
        0.5,
        marks=[],
        id="059",
    ),
    # 非对齐 M=60 + L1/L0B 切分不一致（k[0]=32 ≠ k[1]=64）
    pytest.param(
        "matmul_quant_s8s8_fp16_bias_060",
        torch.int8,
        torch.float16,
        pypto.DT_INT8,
        pypto.DT_FP16,
        (60, 64),
        (64, 64),
        (60, 64),
        ([32, 32], [32, 64], [32, 32]),
        True,
        (1, 64),
        False,
        False,
        0.5,
        marks=[pytest.mark.skip(reason="non-aligned M=60 pending verification")],
        id="060",
    ),
    # requant：golden 公式待硬件确认
    pytest.param(
        "matmul_quant_s8s8_s8_061",
        torch.int8,
        torch.int8,
        pypto.DT_INT8,
        pypto.DT_INT8,
        (64, 64),
        (64, 64),
        (64, 64),
        ([64, 64], [64, 64], [32, 32]),
        False,
        None,
        False,
        False,
        0.5,
        marks=[pytest.mark.skip(reason="requant golden formula pending hardware confirmation")],
        id="061",
    ),
    pytest.param(
        "matmul_quant_s8s8_s8_bias_062",
        torch.int8,
        torch.int8,
        pypto.DT_INT8,
        pypto.DT_INT8,
        (60, 64),
        (64, 64),
        (60, 64),
        ([16, 16], [64, 64], [64, 64]),
        True,
        (1, 64),
        False,
        False,
        0.5,
        marks=[pytest.mark.skip(reason="requant golden formula pending hardware confirmation")],
        id="062",
    ),
    # ---- f16×f16 ---- #
    pytest.param(
        "matmul_quant_f16f16_s32_063",
        torch.float16,
        torch.int32,
        pypto.DT_FP16,
        pypto.DT_INT32,
        (64, 128),
        (128, 64),
        (64, 64),
        ([16, 16], [64, 64], [64, 64]),
        False,
        None,
        False,
        False,
        None,
        marks=[pytest.mark.skip(reason="f16 input s32 accumulation semantics pending hardware confirmation")],
        id="063",
    ),
    pytest.param(
        "matmul_quant_f16f16_s32_bias_064",
        torch.float16,
        torch.int32,
        pypto.DT_FP16,
        pypto.DT_INT32,
        (60, 64),
        (64, 64),
        (60, 64),
        ([16, 16], [64, 64], [64, 64]),
        True,
        (1, 64),
        False,
        False,
        None,
        marks=[pytest.mark.skip(reason="f16 input s32 accumulation semantics pending hardware confirmation")],
        id="064",
    ),
    # 切n：N=64 按 n tile 16 切 4 片
    pytest.param(
        "matmul_quant_f16f16_fp16_065",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (64, 64),
        (64, 64),
        (64, 64),
        ([64, 64], [64, 64], [16, 16]),
        False,
        None,
        False,
        False,
        None,
        marks=[],
        id="065",
    ),
    pytest.param(
        "matmul_quant_f16f16_fp16_bias_066",
        torch.float16,
        torch.float16,
        pypto.DT_FP16,
        pypto.DT_FP16,
        (60, 64),
        (64, 64),
        (60, 64),
        ([16, 16], [64, 64], [64, 64]),
        True,
        (1, 64),
        False,
        False,
        None,
        marks=[pytest.mark.skip(reason="f16->fp16 with s32 bias pending pto-isa bias layout confirmation")],
        id="066",
    ),
    pytest.param(
        "matmul_quant_f16f16_s8_067",
        torch.float16,
        torch.int8,
        pypto.DT_FP16,
        pypto.DT_INT8,
        (64, 64),
        (64, 64),
        (64, 64),
        ([64, 64], [64, 64], [16, 16]),
        False,
        None,
        False,
        False,
        0.5,
        marks=[pytest.mark.skip(reason="requant golden formula pending hardware confirmation")],
        id="067",
    ),
    pytest.param(
        "matmul_quant_f16f16_s8_bias_068",
        torch.float16,
        torch.int8,
        pypto.DT_FP16,
        pypto.DT_INT8,
        (60, 64),
        (64, 64),
        (60, 64),
        ([16, 16], [64, 64], [64, 64]),
        True,
        (1, 64),
        False,
        False,
        0.5,
        marks=[pytest.mark.skip(reason="requant golden formula pending hardware confirmation")],
        id="068",
    ),
]


def create_test_matmul_quant_module(soc_version, test_cases=None):
    """Create a test module for kirin9050 quant matmul (macro instr) with specified soc_version."""
    if test_cases is None:
        test_cases = QUANT_TEST_CASES
    kernels = {}
    for p in test_cases:
        v = p.values
        scale_value = v[13]
        n_size = v[5][-2] if v[11] else v[5][-1]
        kernels[v[0]] = get_matmul_kernel(
            soc_version,
            v[0],
            v[3],  # pypto_in_dtype
            v[4],  # pypto_out_dtype
            v[8],  # cube_tile_shape
            v[9],  # has_bias
            v[10],  # bias_shape
            v[11],  # a_trans
            v[12],  # b_trans
            # f16×f16→fp16 走通用 TMATMUL_BIAS，bias 必须是 fp16；s8 输入与其余宏指令路径用 s32 bias
            # （CheckLiteNPUBiasParam：s8 输入只接受 int32 bias）。
            bias_dtype_override=(
                pypto.DT_FP16 if (v[3] == pypto.DT_FP16 and v[4] == pypto.DT_FP16) else pypto.DT_INT32
            ),
            quant_scale_shape=(1, n_size) if scale_value is not None else None,
        )
    return kernels


def check_quant_precision(actual, golden, out_dtype, kernel_name, input_dtype=None):
    """Quant 用例的 dtype 容差策略，细筛实现见 kirin.common.check_atol_rtol。"""
    if out_dtype == torch.int8:
        rtol, atol = 0.0, 1.0
    elif out_dtype == torch.int32:
        if input_dtype == torch.float16:
            # f16→s32 是 Q16.16 定点 MAC；1 LSB = 1/65536。
            # K=128 实测 max_diff 6~9 LSB，留到 16 LSB。
            rtol, atol = 1e-5, 16.0
        else:
            # s8→s32 是精确整数累加，保持严格。
            rtol, atol = 1e-6, 2.0
    else:
        rtol, atol = 1e-3, 1e-3
    check_atol_rtol(actual, golden, rtol, atol, name=kernel_name)


def run_quant_matmul_test(
    kernels,
    kernel_name,
    torch_in_dtype,
    torch_out_dtype,
    shape_a,
    shape_b,
    shape_c,
    cube_tile_shape,
    has_bias,
    bias_shape,
    a_trans,
    b_trans,
    scale_value,
):
    """Run a single kirin9050 quant matmul kernel test (combination matrix)."""
    device = "cpu"
    if torch_in_dtype == torch.float16:
        a = torch.rand(shape_a, dtype=torch.float16, device=device)
        b = torch.rand(shape_b, dtype=torch.float16, device=device)
    else:
        # s8 输入取小量级，控制 int32 累加值域，避免 dequant 后 fp16 溢出
        a = torch.randint(-8, 7, shape_a, dtype=torch.int8)
        b = torch.randint(-8, 7, shape_b, dtype=torch.int8)
    c = torch.zeros(shape_c, dtype=torch_out_dtype)

    a_new = a.permute(1, 0) if a_trans else a
    b_new = b.permute(1, 0) if b_trans else b
    # 硬件契约：f16 输入的 MAC 结果在定点域是 acc*65536（Q16.16）。
    # - s32 输出：golden = round(acc*65536) + bias（bias 在缩放后按原始整数加）
    # - s8 输出：硬件 bias 乘 65536、scale 除 65536；两个 65536 在实数域恰好抵消，
    #   golden 直接用 (acc+bias) * 实数 scale，不再额外乘 65536。
    f16_fixup = torch_in_dtype == torch.float16

    gold_bias = None
    hw_bias = None
    if has_bias:
        # 宏指令 bias 走 BT 表：fp16 输出用 fp16 bias，其余用 s32 bias。
        bias_dtype = (
            torch.float16 if (torch_in_dtype == torch.float16 and torch_out_dtype == torch.float16) else torch.int32
        )
        gold_bias = torch.randint(-8, 7, bias_shape, dtype=bias_dtype)
        hw_bias = gold_bias
        if f16_fixup and torch_out_dtype == torch.int8:
            hw_bias = (gold_bias.to(torch.int32) * 65536).to(torch.int32)

    n_size = shape_b[-2] if b_trans else shape_b[-1]
    scale_decode = None
    if scale_value is not None:
        # 宏指令 quant_pre 是 per-channel（shape (1, n)）：逐通道随机生成 scale，
        # 若用 scale_value 广播成 (1, n) 全同值，则退化为 per-tensor，覆盖不到 per-channel 路径
        rand_scale = torch.rand((1, n_size), dtype=torch.float32)
        scale_fp32 = torch.maximum(rand_scale, torch.tensor(1e-3, dtype=torch.float32))
        # f16→s8：硬件读 Q16.16 域，scale 需先除 65536；golden 用实数 scale 即可，
        # 因为硬件 bias*65536 与 scale/65536 在实数域抵消（65536 是 2 的幂，fp19 尾数不变）。
        scale_hw_fp32 = scale_fp32
        if f16_fixup and torch_out_dtype == torch.int8:
            scale_hw_fp32 = scale_fp32 / 65536.0
        scale_decode = decode_quant_param(trans_quant_param(scale_fp32))

    def dispatch():
        if has_bias and scale_value is not None:
            kernels[kernel_name](a, b, hw_bias, trans_quant_param(scale_hw_fp32), c)
        elif has_bias:
            kernels[kernel_name](a, b, hw_bias, c)
        elif scale_value is not None:
            kernels[kernel_name](a, b, trans_quant_param(scale_hw_fp32), c)
        else:
            kernels[kernel_name](a, b, c)

    dispatch()

    # golden
    if torch_in_dtype == torch.float16:
        acc_raw = a_new.float() @ b_new.float()
    else:
        acc_raw = (a_new.to(torch.int32) @ b_new.to(torch.int32)).float()
    acc_f = acc_raw + gold_bias.float() if gold_bias is not None else acc_raw

    if torch_out_dtype == torch.int32:
        if f16_fixup:
            golden_c = torch.round(acc_raw * 65536.0).to(torch.int32)
            if gold_bias is not None:
                golden_c = golden_c + gold_bias.to(torch.int32)
        else:
            golden_c = torch.round(acc_f).to(torch.int32)
    elif torch_out_dtype == torch.float16:
        # s8 输入：dequant = s32 acc * per-channel scale；f16 输入无 scale 为直接 fp16 输出
        golden_c = (acc_f * scale_decode).to(torch.float16) if scale_decode is not None else acc_f.to(torch.float16)
    else:
        # requant：s8 输入为 round((acc+bias)*scale)。
        # f16 输入时硬件在 Q16.16 域累加（bias*65536），scale 也按 /65536 归一化；
        # 两个 65536 在实数域抵消，golden 直接用原始 (acc+bias) * 实数 scale。
        golden_c = torch.round(acc_f * scale_decode).clamp(-128, 127).to(torch.int8)

    check_nan(c, name=kernel_name)
    check_quant_precision(c, golden_c, torch_out_dtype, kernel_name, torch_in_dtype)
    cos_value = abs(compare_cos(np.array(c.cpu()), np.array(golden_c.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
