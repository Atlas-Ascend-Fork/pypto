#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test attention codegen for Kirin9050
"""

from kirin.common.common_attention import (
    TEST_CASES_ATTENTION,
    create_attention_kernels,
    create_attention_quant_kernels,
    run_attention_test,
    run_cast_matmul_test,
    run_mul_matmul_test,
)
import pytest

# mul+matmul 融合用例（全部 skip）。
TEST_CASES_MUL_MATMUL = [
    pytest.param("mul_matmul", (16, 16), (16, 16), (16, 16), (16, 16), (16, 16),
                 ([16, 16], [16, 16], [16, 16]), marks=[pytest.mark.skip()], id="mul_matmul"),
    pytest.param("matmul_mul", (16, 16), (16, 16), (16, 16), (16, 16), (16, 16),
                 ([16, 16], [16, 16], [16, 16]), marks=[pytest.mark.skip()], id="matmul_mul"),
    pytest.param("matmul_mul_with_reshape_001", (1, 1, 1, 256), (1, 1, 256, 128), (1, 1, 1, 128),
                 (1, 1, 1, 128), (1, 1, 1, 64), ([16, 16], [64, 128], [64, 128]),
                 marks=[pytest.mark.skip()], id="matmul_mul_with_reshape_001"),
    pytest.param("matmul_mul_with_reshape_002", (1, 1, 1, 256), (1, 1, 256, 128), (1, 1, 1, 128),
                 (1, 1, 1, 128), (1, 1, 1, 64), ([16, 16], [64, 128], [64, 128]),
                 marks=[pytest.mark.skip()], id="matmul_mul_with_reshape_002"),
    pytest.param("matmul_mul_with_reshape_003", (1, 16, 1, 256), (1, 16, 256, 128), (1, 16, 1, 128),
                 (1, 16, 1, 128), (1, 1, 1, 64), ([16, 16], [64, 128], [64, 128]),
                 marks=[pytest.mark.skip()], id="matmul_mul_with_reshape_003"),
    pytest.param("mul_matmul_with_reshape_001", (1, 1, 16, 16), (1, 1, 16, 16), (1, 1, 16, 16),
                 (1, 1, 16, 16), (1, 1, 16, 16), ([16, 16], [16, 16], [16, 16]),
                 marks=[pytest.mark.skip()], id="mul_matmul_with_reshape_001"),
    pytest.param("mul_matmul_with_reshape_002", (1, 1, 1, 256), (1, 1, 1, 256), (1, 1, 256, 128),
                 (1, 1, 1, 128), (1, 1, 1, 64), ([16, 16], [64, 64], [64, 128]),
                 marks=[pytest.mark.skip()], id="mul_matmul_with_reshape_002"),
    pytest.param("mul_matmul_with_reshape_003", (1, 16, 1, 256), (1, 16, 1, 256), (1, 16, 256, 128),
                 (1, 16, 1, 128), (1, 1, 1, 64), ([16, 16], [64, 64], [64, 128]),
                 marks=[pytest.mark.skip()], id="mul_matmul_with_reshape_003"),
]

KERNELS = create_attention_kernels("Kirin9050", TEST_CASES_MUL_MATMUL)

# kirin9050 宏指令 pre-stage quant 的 cast+matmul 融合用例。
CAST_MATMUL_TEST_CASES = [
    pytest.param("cast_matmul_int8", "cast_matmul", (16, 16), (16, 16), (16, 16),
                 (32, 32), (32, 32), None, ([32, 32], [32, 32], [32, 32]), marks=[pytest.mark.skip()],
                 id="cast_matmul_int8"),
    pytest.param("matmul_cast_001_int8", "matmul_cast", (32, 32), (32, 32), (32, 32),
                 None, None, (32, 32), ([32, 32], [32, 32], [32, 32]), marks=[pytest.mark.skip()],
                 id="matmul_cast_001_int8"),
    pytest.param("matmul_cast_002_int32", "matmul_cast_002", (16, 16), (16, 16), (16, 16),
                 None, None, (32, 32), ([32, 32], [32, 32], [32, 32]), marks=[pytest.mark.skip()],
                 id="matmul_cast_002_int32"),
    pytest.param("cast_matmul_with_reshape_001_int8", "cast_matmul", (1, 1, 16, 32), (1, 1, 32, 32), (1, 1, 16, 32),
                 (1, 1, 16, 32), (1, 1, 32, 32), None, ([16, 16], [32, 32], [32, 32]), marks=[pytest.mark.skip()],
                 id="cast_matmul_with_reshape_001_int8"),
    pytest.param("cast_matmul_with_reshape_002_int8", "cast_matmul", (1, 1, 1, 256), (1, 1, 256, 128), (1, 1, 1, 128),
                 (1, 1, 32, 64), (1, 1, 64, 32), None, ([32, 32], [64, 64], [32, 32]), marks=[pytest.mark.skip()],
                 id="cast_matmul_with_reshape_002_int8"),
    pytest.param("cast_matmul_with_reshape_003_int8", "cast_matmul", (1, 16, 1, 256), (1, 16, 256, 128),
                 (1, 16, 1, 128), (1, 1, 32, 64), (1, 1, 64, 32), None, ([32, 32], [64, 64], [32, 32]),
                 marks=[pytest.mark.skip()], id="cast_matmul_with_reshape_003_int8"),
    pytest.param("matmul_cast_with_reshape_001_int8", "matmul_cast", (1, 1, 1, 256), (1, 1, 256, 128), (1, 1, 1, 128),
                 None, None, (1, 1, 32, 32), ([32, 32], [64, 64], [32, 32]), marks=[pytest.mark.skip()],
                 id="matmul_cast_with_reshape_001_int8"),
    pytest.param("matmul_cast_with_reshape_002_int8", "matmul_cast", (1, 1, 1, 256), (1, 1, 256, 128), (1, 1, 1, 128),
                 None, None, (1, 1, 32, 32), ([32, 32], [64, 64], [32, 32]), marks=[pytest.mark.skip()],
                 id="matmul_cast_with_reshape_002_int8"),
    pytest.param("matmul_cast_with_reshape_003_int8", "matmul_cast", (1, 16, 1, 256), (1, 16, 256, 128),
                 (1, 16, 1, 128), None, None, (1, 1, 32, 32), ([32, 32], [64, 64], [32, 32]),
                 marks=[pytest.mark.skip()], id="matmul_cast_with_reshape_003_int8"),
    pytest.param("matmul_cast_with_reshape_001_int32", "matmul_cast_002", (1, 1, 1, 256), (1, 1, 256, 128),
                 (1, 1, 1, 128), None, None, (1, 1, 32, 32), ([32, 32], [64, 64], [32, 32]), marks=[pytest.mark.skip()],
                 id="matmul_cast_with_reshape_001_int32"),
    pytest.param("matmul_cast_with_reshape_002_int32", "matmul_cast_002", (1, 1, 1, 256), (1, 1, 256, 128),
                 (1, 1, 1, 128), None, None, (1, 1, 32, 32), ([32, 32], [64, 64], [32, 32]), marks=[pytest.mark.skip()],
                 id="matmul_cast_with_reshape_002_int32"),
    pytest.param("matmul_cast_with_reshape_003_int32", "matmul_cast_002", (1, 16, 1, 256), (1, 16, 256, 128),
                 (1, 16, 1, 128), None, None, (1, 1, 32, 32), ([32, 32], [64, 64], [32, 32]),
                 marks=[pytest.mark.skip()], id="matmul_cast_with_reshape_003_int32"),
]

QUANT_KERNELS = create_attention_quant_kernels("Kirin9050", CAST_MATMUL_TEST_CASES)


@pytest.mark.parametrize("op_type,a_shape,b_shape,c_shape,out_shape,vec_tile,cube_tile", TEST_CASES_MUL_MATMUL)
def test_mul_matmul(op_type, a_shape, b_shape, c_shape, out_shape, vec_tile, cube_tile):
    shapes = (a_shape, b_shape, c_shape, out_shape)
    run_mul_matmul_test(KERNELS, op_type, shapes)


@pytest.mark.parametrize(
    "op_type,q_shape,k_shape,v_shape,attn_mask_shape,output_shape,vec_tile,cube_tile", TEST_CASES_ATTENTION
)
def test_attention(op_type, q_shape, k_shape, v_shape, attn_mask_shape, output_shape, vec_tile, cube_tile):
    shapes = (q_shape, k_shape, v_shape, attn_mask_shape, output_shape)
    run_attention_test(KERNELS, op_type, shapes)


@pytest.mark.parametrize(
    "kernel_name,op_type,a_shape,b_shape,out_shape,vec_a,vec_b,vec_out,cube_tile_shape",
    CAST_MATMUL_TEST_CASES,
)
def test_cast_matmul_fusion(
    kernel_name, op_type, a_shape, b_shape, out_shape, vec_a, vec_b, vec_out, cube_tile_shape
):
    run_cast_matmul_test(QUANT_KERNELS, kernel_name, op_type, a_shape, b_shape, out_shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
