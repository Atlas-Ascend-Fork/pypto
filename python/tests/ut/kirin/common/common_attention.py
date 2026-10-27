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
Test attention codegen - common functions for Kirin9030 and Kirin9050
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from kirin.common import check_atol_rtol, check_nan, compare_cos, generated_cpp_contains
import pypto
from pypto import CastMode, SaturationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "tools" / "scripts"))
from lite_npu_quant import decode_quant_param, trans_quant_param

TEST_CASES_ATTENTION = [
    # kernel_name: name of the kernel
    # q_shape: query tensor shape
    # k_shape: key tensor shape
    # v_shape: value tensor shape
    # attn_mask_shape: attention mask shape
    # output_shape: output tensor shape
    # vec_tile_shapes: vector tile shape 1 for concat, 2 for mul/add/softmax, 3 for transpose
    # cube_tile_shapes: cube tile shapes for matmul
    # marks: pytest marks
    pytest.param(
        "attention_prefill",
        (1, 16, 64, 128),
        (1, 8, 2048, 128),
        (1, 8, 2048, 128),
        (1, 1, 64, 2048),
        (1, 64, 16, 128),
        ((1, 1, 128, 128), (1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 128, 128), (1, 1, 64, 128)),
        (([64, 64], [64, 128], [128, 128]), ([64, 64], [64, 128], [128, 128])),
        marks=[pytest.mark.skip()],
        id="attention_prefill",
    ),
    pytest.param(
        "attention_decoder",
        (1, 16, 1, 128),
        (1, 8, 2048, 128),
        (1, 8, 2048, 128),
        (1, 1, 1, 2048),
        (1, 1, 16, 128),
        ((1, 1, 128, 128), (1, 1, 1, 128), (1, 1, 1, 128), (1, 1, 128, 128), (1, 1, 1, 128)),
        (([16, 16], [64, 128], [128, 128]), ([16, 16], [64, 128], [128, 128])),
        marks=[pytest.mark.skip()],
        id="attention_decoder",
    ),
]


def _make_mul_matmul_kernel(soc_version, name, a_shape, b_shape, c_shape, out_shape, vec_tile_shape, cube_tile_shape):
    vec_tile = vec_tile_shape if vec_tile_shape is not None else (16, 16)
    cube_tile = cube_tile_shape if cube_tile_shape is not None else ([16, 16], [16, 16], [16, 16])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP16),
        b: pypto.Tensor(b_shape, pypto.DT_FP16),
        c: pypto.Tensor(c_shape, pypto.DT_FP16),
        out: pypto.Tensor(out_shape, pypto.DT_FP16),
    ):
        if "mul_matmul" in name:
            pypto.set_vec_tile_shapes(*vec_tile)
            pypto.set_cube_tile_shapes(*cube_tile)
            mul = pypto.mul(a, b)
            out[:] = pypto.matmul(mul, c, pypto.DT_FP16, a_trans=False, b_trans=False)
        else:
            pypto.set_cube_tile_shapes(*cube_tile)
            pypto.set_vec_tile_shapes(*vec_tile)
            mul = pypto.matmul(a, b, pypto.DT_FP16, a_trans=False, b_trans=False)
            out[:] = pypto.mul(mul, c)

    kernel.__name__ = name
    return kernel


def _make_attention_kernel(
    soc_version, name, q_shape, k_shape, v_shape, attn_mask_shape, output_shape, vec_tile_shape, cube_tile_shape
):
    vec_tile = vec_tile_shape if vec_tile_shape is not None else (1, 1, 16, 16)
    cube_tile = cube_tile_shape if cube_tile_shape is not None else ([16, 16], [16, 16], [16, 16])

    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": pypto.RunMode.SIM},
        pass_options={"enable_slice": False},
    )
    def kernel(
        q: pypto.Tensor(q_shape, pypto.DT_FP16),
        k: pypto.Tensor(k_shape, pypto.DT_FP16),
        v: pypto.Tensor(v_shape, pypto.DT_FP16),
        attn_mask: pypto.Tensor(attn_mask_shape, pypto.DT_FP16),
        output: pypto.Tensor(output_shape, pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(*vec_tile[0])
        k_reshape = pypto.reshape(k, [8, 1, 2048, 128])
        k_concat = pypto.concat([k_reshape, k_reshape], dim=1)
        k_concat_reshape = pypto.reshape(k_concat, [1, 16, 2048, 128])
        pypto.set_cube_tile_shapes(*cube_tile[0])
        q_k_t = pypto.matmul(q, k_concat_reshape, pypto.DT_FP16, a_trans=False, b_trans=True)
        pypto.set_vec_tile_shapes(*vec_tile[1])
        q_k_t_mul = pypto.mul(q_k_t, 0.0883883461356163)
        pypto.set_vec_tile_shapes(*vec_tile[2])
        q_k_t_mul_add = pypto.add(q_k_t_mul, attn_mask)
        softmax_q_k_t = pypto.softmax(q_k_t_mul_add, dim=-1)
        pypto.set_vec_tile_shapes(*vec_tile[3])
        v_reshape = pypto.reshape(v, [8, 1, 2048, 128])
        v_concat = pypto.concat([v_reshape, v_reshape], dim=1)
        v_concat_reshape = pypto.reshape(v_concat, [1, 16, 2048, 128])
        pypto.set_cube_tile_shapes(*cube_tile[1])
        if q_shape[2] != 1:
            attention_res = pypto.matmul(softmax_q_k_t, v_concat_reshape, pypto.DT_FP16, a_trans=False, b_trans=False)
            pypto.set_vec_tile_shapes(*vec_tile[4])
            output[:] = pypto.transpose(attention_res, 1, 2)
        else:
            output[:] = pypto.matmul(softmax_q_k_t, v_concat_reshape, pypto.DT_FP16, a_trans=False, b_trans=False)

    kernel.__name__ = name
    return kernel


def create_attention_kernels(soc_version, test_cases):
    kernels_mul_matmul = {
        p.values[0]: _make_mul_matmul_kernel(
            soc_version, p.values[0], p.values[1], p.values[2], p.values[3], p.values[4], p.values[5], p.values[6]
        )
        for p in test_cases
    }

    kernels_attention = {
        p.values[0]: _make_attention_kernel(
            soc_version,
            p.values[0],
            p.values[1],
            p.values[2],
            p.values[3],
            p.values[4],
            p.values[5],
            p.values[6],
            p.values[7],
        )
        for p in TEST_CASES_ATTENTION
    }

    return {**kernels_mul_matmul, **kernels_attention}


def _compute_golden_mul_matmul(op_type, a, b, c):
    if "mul_matmul" in op_type:
        mul = torch.mul(a, b)
        return torch.matmul(mul, c)
    else:
        mul = torch.matmul(a, b)
        return torch.mul(mul, c)


def _compute_golden_attention(op_type, q, k, v, attn_mask):
    k_reshape = k.reshape(8, 1, 2048, 128)
    k_concat = torch.concat([k_reshape, k_reshape], dim=1)
    k_concat_reshape = k_concat.reshape(1, 16, 2048, 128)
    k_t = torch.transpose(k_concat_reshape, 2, 3)
    q_k_t = torch.matmul(q, k_t)
    q_k_t_mul = torch.mul(q_k_t, 0.0883883461356163)
    q_k_t_mul_add = torch.add(q_k_t_mul, attn_mask)
    softmax = torch.softmax(q_k_t_mul_add, -1)
    v_reshape = v.reshape(8, 1, 2048, 128)
    v_concat = torch.concat([v_reshape, v_reshape], dim=1)
    v_concat_reshape = v_concat.reshape(1, 16, 2048, 128)
    attention_res = torch.matmul(softmax, v_concat_reshape)
    return torch.transpose(attention_res, 1, 2)


def run_mul_matmul_test(kernels, op_type, shapes):
    a = torch.rand(shapes[0], dtype=torch.float16, device="cpu")
    b = torch.rand(shapes[1], dtype=torch.float16, device="cpu")
    c = torch.rand(shapes[2], dtype=torch.float16, device="cpu")
    out = torch.rand(shapes[3], dtype=torch.float16, device="cpu")

    kernels[op_type](a, b, c, out)

    expected_copy = "TCopyUB2L1" if "mul_matmul" in op_type else "TCopyL0C2UB"
    assert generated_cpp_contains(expected_copy), f"{op_type}: generated cpp missing '{expected_copy}'"

    golden = _compute_golden_mul_matmul(op_type, a, b, c)

    check_nan(out, name=op_type)
    cos_value = abs(compare_cos(np.array(out.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"cos_value {cos_value} < 0.9999")


def run_attention_test(kernels, op_type, shapes):
    q = torch.rand(shapes[0], dtype=torch.float16, device="cpu")
    k = torch.rand(shapes[1], dtype=torch.float16, device="cpu")
    v = torch.rand(shapes[2], dtype=torch.float16, device="cpu")
    attn_mask = torch.rand(shapes[3], dtype=torch.float16, device="cpu")
    output = torch.rand(shapes[4], dtype=torch.float16, device="cpu")

    golden = _compute_golden_attention(op_type, q, k, v, attn_mask)
    kernels[op_type](q, k, v, attn_mask, output)

    check_nan(output, name=op_type)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"cos_value {cos_value} < 0.9999")


def _make_cast_matmul_kernel(
    soc_version, op_type, name, a_shape, b_shape, out_shape, vec_a, vec_b, vec_out, cube_tile_shape
):
    scale_shape = (1, b_shape[-1])

    if op_type == "cast_matmul":

        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def kernel(
            a: pypto.Tensor(a_shape, pypto.DT_FP16),
            b: pypto.Tensor(b_shape, pypto.DT_FP16),
            scale: pypto.Tensor(scale_shape, pypto.DT_INT64),
            out: pypto.Tensor(out_shape, pypto.DT_FP16),
        ):
            pypto.set_vec_tile_shapes(*vec_a)
            a_s8 = pypto.cast(a, pypto.DT_INT8, CastMode.CAST_NONE, SaturationMode.ON)
            pypto.set_vec_tile_shapes(*vec_b)
            b_s8 = pypto.cast(b, pypto.DT_INT8, CastMode.CAST_NONE, SaturationMode.ON)
            pypto.set_cube_tile_shapes(*cube_tile_shape)
            out[:] = pypto.matmul(a_s8, b_s8, pypto.DT_FP16, extend_params={"scale_tensor": scale})

        kernel.__name__ = name
        return kernel

    if op_type == "matmul_cast":

        @pypto.frontend.jit(
            codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM}
        )
        def kernel(
            a: pypto.Tensor(a_shape, pypto.DT_INT8),
            b: pypto.Tensor(b_shape, pypto.DT_INT8),
            scale: pypto.Tensor(scale_shape, pypto.DT_INT64),
            out: pypto.Tensor(out_shape, pypto.DT_FP16),
        ):
            pypto.set_cube_tile_shapes(*cube_tile_shape)
            mm = pypto.matmul(a, b, pypto.DT_INT8, extend_params={"scale_tensor": scale})
            pypto.set_vec_tile_shapes(*vec_out)
            out[:] = pypto.cast(mm, pypto.DT_FP16, CastMode.CAST_NONE, SaturationMode.OFF)

        kernel.__name__ = name
        return kernel

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor(a_shape, pypto.DT_INT8),
        b: pypto.Tensor(b_shape, pypto.DT_INT8),
        out: pypto.Tensor(out_shape, pypto.DT_FP32),
    ):
        pypto.set_cube_tile_shapes(*cube_tile_shape)
        mm = pypto.matmul(a, b, pypto.DT_INT32)
        pypto.set_vec_tile_shapes(*vec_out)
        out[:] = pypto.cast(mm, pypto.DT_FP32, CastMode.CAST_NONE, SaturationMode.OFF)

    kernel.__name__ = name
    return kernel


def create_attention_quant_kernels(soc_version, test_cases):
    """Build kirin9050 cast+matmul quant fusion kernels, keyed by name."""
    return {
        p.values[0]: _make_cast_matmul_kernel(
            soc_version, p.values[1], p.values[0], p.values[2], p.values[3], p.values[4],
            p.values[5], p.values[6], p.values[7], p.values[8]
        )
        for p in test_cases
    }


def _gen_quant_scale(n_size):
    """生成 per-channel quant_pre scale（逐通道随机，clamp 到 >=1e-3）。

    返回 (packed, decode)：
      - packed: trans_quant_param 打包后的 int64，直接作为 kernel 的 scale_tensor 入参；
      - decode: fp19 截断尾数后按位解码回的 fp32 近似值，作为 golden 用 scale。
    """
    rand_scale = torch.rand((1, n_size), dtype=torch.float32)
    scale_fp32 = torch.maximum(rand_scale, torch.tensor(1e-3, dtype=torch.float32))
    scale_packed = trans_quant_param(scale_fp32)
    scale_decode = decode_quant_param(scale_packed)
    return scale_packed, scale_decode


def _cast_fp16_to_s8_golden(t):
    """fp16->s8 cast golden：CAST_NONE（float->int 解析为 CAST_TRUNC）+ SaturationMode.ON 截断并饱和。"""
    return torch.clamp(torch.trunc(t.float()), -128, 127).to(torch.int8)


def run_cast_matmul_test(kernels, kernel_name, op_type, a_shape, b_shape, out_shape):
    """Run a single kirin9050 cast+matmul quant fusion kernel test."""
    device = "cpu"
    n_size = b_shape[-1]

    if op_type == "cast_matmul":
        a = torch.rand(a_shape, dtype=torch.float16, device=device)
        b = torch.rand(b_shape, dtype=torch.float16, device=device)
        a_s8 = _cast_fp16_to_s8_golden(a)
        b_s8 = _cast_fp16_to_s8_golden(b)
        acc = (a_s8.to(torch.int32) @ b_s8.to(torch.int32)).float()
        scale_packed, scale_decode = _gen_quant_scale(n_size)
        golden = (acc * scale_decode).to(torch.float16)
        out = torch.zeros(out_shape, dtype=torch.float16, device=device)
        kernels[kernel_name](a, b, scale_packed, out)
        check_atol_rtol(out, golden, 1e-3, 1e-3, name=kernel_name)
    elif op_type == "matmul_cast":
        a = torch.randint(-8, 7, a_shape, dtype=torch.int8)
        b = torch.randint(-8, 7, b_shape, dtype=torch.int8)
        acc = (a.to(torch.int32) @ b.to(torch.int32)).float()
        scale_packed, scale_decode = _gen_quant_scale(n_size)
        mm_s8 = torch.round(acc * scale_decode).clamp(-128, 127).to(torch.int8)
        golden = mm_s8.to(torch.float16)
        out = torch.zeros(out_shape, dtype=torch.float16, device=device)
        kernels[kernel_name](a, b, scale_packed, out)
        check_atol_rtol(out, golden, 1e-3, 1e-3, name=kernel_name)
    else:
        a = torch.randint(-8, 7, a_shape, dtype=torch.int8)
        b = torch.randint(-8, 7, b_shape, dtype=torch.int8)
        mm_s32 = (a.to(torch.int32) @ b.to(torch.int32)).to(torch.int32)
        golden = mm_s32.to(torch.float32)
        out = torch.zeros(out_shape, dtype=torch.float32, device=device)
        kernels[kernel_name](a, b, out)
        check_atol_rtol(out, golden, 1e-6, 1e-3, name=kernel_name)

    check_nan(out, name=kernel_name)
    cos_value = abs(compare_cos(np.array(out.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
