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
Test qkv_rmsnorm_rope_scatternd codegen - common functions for Kirin9030 and Kirin9050
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def compute_rmsnorm_golden(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    hidden_dim = input_tensor.shape[-1]
    input_fp32 = input_tensor.float()
    rms = torch.sqrt(torch.sum(input_fp32 * input_fp32 * (1.0 / hidden_dim), dim=-1, keepdim=True) + epsilon)
    normalized = input_fp32 / rms
    if gamma is not None:
        # gamma shape (1, 128, 1, 1), transpose to (1, 1, 1, 128) for broadcast
        normalized = normalized * gamma.transpose(1, 3).float()
    return normalized.to(input_tensor.dtype)


def compute_rope_golden(query_tensor: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
    head_dim = query_tensor.shape[3]
    half_head_dim = head_dim // 2
    query_first_half, query_second_half = torch.split(query_tensor, half_head_dim, dim=3)
    query_second_half_negated = torch.mul(query_second_half, -1)
    rotated_qurey = torch.cat([query_second_half_negated, query_first_half], dim=3)
    query_scaled = torch.mul(query_tensor, cos_cache)
    rotated_scaled = torch.mul(rotated_qurey, sin_cache)
    output = torch.add(query_scaled, rotated_scaled)
    return output


def compute_qkv_fused_golden(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    past_key: torch.Tensor,
    past_value: torch.Tensor,
    indices: torch.Tensor,
    epsilon: float = 1e-6,
):
    """Fused QKV golden (prefill & decode). Cache write positions come from `indices` (N,1) int32."""
    q_rmsnorm = compute_rmsnorm_golden(q_input, q_gamma, epsilon)
    q_transposed = torch.transpose(q_rmsnorm, 1, 2)
    q_out = compute_rope_golden(q_transposed, cos_cache, sin_cache)

    k_rmsnorm = compute_rmsnorm_golden(k_input, k_gamma, epsilon)
    k_transposed = torch.transpose(k_rmsnorm, 1, 2)
    k_rope = compute_rope_golden(k_transposed, cos_cache, sin_cache)
    k_final = k_rope.transpose(0, 2)
    indices_squeezed = indices.squeeze(1)
    k_out = past_key.clone()
    k_out.index_put_((indices_squeezed,), k_final, accumulate=False)
    k_out = k_out.transpose(0, 2)

    v_final = v_input.permute(1, 2, 0, 3)
    v_out = past_value.clone()
    v_out.index_put_((indices_squeezed,), v_final, accumulate=False)
    v_out = v_out.transpose(0, 2)

    return q_out, k_out, v_out


def apply_rmsnorm(
    input_tensor: pypto.Tensor,
    gamma: pypto.Tensor,
    epsilon: float = 1e-6,
) -> pypto.Tensor:
    # gamma 入参 shape 为 (1, 128, 1, 1)，flatten 到 (128,) 给 rms_norm 用。
    # 不把 reshape 结果赋回 gamma 变量，避免被前端识别为"修改入参"触发 outcast symbol。
    normalized = pypto.rms_norm(input_tensor, pypto.reshape(gamma, [128]), epsilon=epsilon)
    return normalized


def apply_rope(input_tensor, cos_cache, sin_cache, tile_neg, tile_scale):
    batch_size = input_tensor.shape[0]
    num_heads = input_tensor.shape[1]
    seq_len = input_tensor.shape[2]
    head_dim = input_tensor.shape[3]
    half_head_dim = head_dim // 2

    first_half = pypto.view(input_tensor, [batch_size, num_heads, seq_len, half_head_dim], [0, 0, 0, 0])
    second_half = pypto.view(input_tensor, [batch_size, num_heads, seq_len, half_head_dim], [0, 0, 0, half_head_dim])
    pypto.set_vec_tile_shapes(*tile_neg)
    second_half_negated = pypto.mul(second_half, -1.0)
    rotated_tensor = pypto.concat([second_half_negated, first_half], dim=-1)
    pypto.set_vec_tile_shapes(*tile_scale)
    scaled_input = pypto.mul(input_tensor, cos_cache)
    scaled_rotated = pypto.mul(rotated_tensor, sin_cache)
    return pypto.add(scaled_input, scaled_rotated)


TEST_CASES = [
    # kernel_name: kernel to run ("qkv_fused_prefill" / "qkv_fused_decode")
    # torch_dtype: torch data type (float16)
    # q_input_shape / k_input_shape / v_input_shape: input tensor shapes
    # q_gamma_shape / k_gamma_shape: rmsnorm gamma tensor shapes
    # cos_shape / sin_shape: rope cache tensor shapes
    # past_key_shape / past_value_shape: cache tensor shapes
    # indices_shape: cache write position tensor shape (prefill (64,1), decode (1,1))
    # tiles: 12 vec tile shapes used by the kernel body
    # rope_tiles: 2 vec tile shapes used inside apply_rope
    # index_len: reshape length of the flattened index tensor
    pytest.param(
        "qkv_fused_prefill",
        torch.float16,
        (1, 64, 16, 128),
        (1, 64, 8, 128),
        (1, 64, 8, 128),
        (1, 128, 1, 1),
        (1, 128, 1, 1),
        (1, 1, 64, 128),
        (1, 1, 64, 128),
        (2048, 8, 1, 128),
        (2048, 8, 1, 128),
        (64, 1),
        [
            (1, 4, 16, 128),
            (1, 8, 16, 128),
            (1, 8, 8, 128),
            (1, 16, 8, 128),
            (1, 4, 64, 128),
            (1, 1, 64, 64),
            (32,),
            (32, 8, 1, 128),
            (1, 32, 8, 128),
            (1, 4, 64, 128),
            (32,),
            (32, 8, 1, 128),
        ],
        [
            (1, 8, 64, 64),
            (1, 4, 64, 128),
        ],
        64,
        marks=[pytest.mark.skip()],
        id="prefill",
    ),
    pytest.param(
        "qkv_fused_decode",
        torch.float16,
        (1, 1, 16, 128),
        (1, 1, 8, 128),
        (1, 1, 8, 128),
        (1, 128, 1, 1),
        (1, 128, 1, 1),
        (1, 1, 1, 128),
        (1, 1, 1, 128),
        (2048, 8, 1, 128),
        (2048, 8, 1, 128),
        (1, 1),
        [
            (1, 1, 16, 128),
            (1, 1, 16, 128),
            (1, 1, 8, 128),
            (1, 1, 8, 128),
            (1, 1, 1, 128),
            (1, 1, 1, 1),
            (1,),
            (32, 8, 1, 128),
            (1, 1, 8, 128),
            (1, 1, 1, 128),
            (1,),
            (32, 8, 1, 128),
        ],
        [
            (1, 1, 1, 64),
            (1, 1, 1, 128),
        ],
        1,
        marks=[pytest.mark.skip()],
        id="decoder",
    ),
]


def _make_fused_qkv_kernel(soc_version, tiles, rope_tiles, index_len):
    @pypto.frontend.jit(
        codegen_options={"soc_version": soc_version},
        runtime_options={"run_mode": pypto.RunMode.SIM},
        debug_options={"compile_debug_mode": 1},
    )
    def kernel(
        q_input: pypto.Tensor([...], pypto.DT_FP16),
        k_input: pypto.Tensor([...], pypto.DT_FP16),
        v_input: pypto.Tensor([...], pypto.DT_FP16),
        q_rmsnorm_gamma: pypto.Tensor([...], pypto.DT_FP16),
        k_rmsnorm_gamma: pypto.Tensor([...], pypto.DT_FP16),
        cos_cache: pypto.Tensor([...], pypto.DT_FP16),
        sin_cache: pypto.Tensor([...], pypto.DT_FP16),
        indices: pypto.Tensor([...], pypto.DT_INT32),
        past_key: pypto.Tensor([...], pypto.DT_FP16),
        past_value: pypto.Tensor([...], pypto.DT_FP16),
        q_out: pypto.Tensor([...], pypto.DT_FP16),
        k_out: pypto.Tensor([...], pypto.DT_FP16),
        v_out: pypto.Tensor([...], pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(*tiles[0])
        q_rmsnorm = apply_rmsnorm(q_input, q_rmsnorm_gamma, epsilon=1e-6)
        pypto.set_vec_tile_shapes(*tiles[1])
        q_transposed = pypto.transpose(q_rmsnorm, 1, 2)
        q_out[:] = apply_rope(q_transposed, cos_cache, sin_cache, *rope_tiles)

        pypto.set_vec_tile_shapes(*tiles[2])
        k_rmsnorm = apply_rmsnorm(k_input, k_rmsnorm_gamma, epsilon=1e-6)
        pypto.set_vec_tile_shapes(*tiles[3])
        k_transposed = pypto.transpose(k_rmsnorm, 1, 2)

        k_rope = apply_rope(k_transposed, cos_cache, sin_cache, *rope_tiles)
        pypto.set_vec_tile_shapes(*tiles[4])
        k_final = pypto.transpose(k_rope, 0, 2)
        pypto.set_vec_tile_shapes(*tiles[5])
        indices_squeezed = pypto.reshape(indices, [index_len])

        pypto.set_vec_tile_shapes(*tiles[6])
        pypto.index_put_(past_key, (indices_squeezed,), k_final, False)
        pypto.set_vec_tile_shapes(*tiles[7])
        k_out[:] = pypto.transpose(past_key, 0, 2)

        pypto.set_vec_tile_shapes(*tiles[8])
        v_transposed_12 = pypto.transpose(v_input, 1, 2)
        pypto.set_vec_tile_shapes(*tiles[9])
        v_final = pypto.transpose(v_transposed_12, 0, 2)

        pypto.set_vec_tile_shapes(*tiles[10])
        pypto.index_put_(past_value, (indices_squeezed,), v_final, False)
        pypto.set_vec_tile_shapes(*tiles[11])
        v_out[:] = pypto.transpose(past_value, 0, 2)

    return kernel


def create_fused_qkv_kernels(soc_version):
    """Build prefill and decode fused QKV kernels, keyed by name."""
    pypto.set_pass_options(sg_set_tunevf_mode=1)
    return {
        case.values[0]: _make_fused_qkv_kernel(soc_version, case.values[12], case.values[13], case.values[14])
        for case in TEST_CASES
    }


def _assert_cos_output(out, golden, name):
    check_nan(out, name=name)
    cos_value = abs(compare_cos(np.array(out.cpu()), np.array(golden.cpu())))
    print(f"[stage] {name} cos = {cos_value}")
    if cos_value < 0.9999:
        raise AssertionError(f"{name} cos_value {cos_value} < 0.9999")


def run_qkv_fused_test(
    kernels,
    kernel_name,
    dtype,
    q_input_shape,
    k_input_shape,
    v_input_shape,
    q_gamma_shape,
    k_gamma_shape,
    cos_shape,
    sin_shape,
    past_key_shape,
    past_value_shape,
    indices_shape,
):
    """Run a single fused QKV kernel test (prefill or decode)."""
    device = "cpu"

    torch.manual_seed(42)
    q_input = torch.randn(q_input_shape, dtype=dtype, device=device).contiguous()
    k_input = torch.randn(k_input_shape, dtype=dtype, device=device).contiguous()
    v_input = torch.randn(v_input_shape, dtype=dtype, device=device).contiguous()
    q_gamma = torch.randn(q_gamma_shape, dtype=dtype, device=device).contiguous()
    k_gamma = torch.randn(k_gamma_shape, dtype=dtype, device=device).contiguous()
    cos_cache = torch.randn(cos_shape, dtype=dtype, device=device).contiguous()
    sin_cache = torch.randn(sin_shape, dtype=dtype, device=device).contiguous()
    past_key = torch.randn(past_key_shape, dtype=dtype, device=device).contiguous()
    past_value = torch.randn(past_value_shape, dtype=dtype, device=device).contiguous()
    indices = torch.randperm(past_key_shape[0], dtype=torch.int32, device=device)[:indices_shape[0]].unsqueeze(1)

    q_golden, k_golden, v_golden = compute_qkv_fused_golden(
        q_input, k_input, v_input, q_gamma, k_gamma, cos_cache, sin_cache, past_key, past_value, indices
    )

    q_out = torch.empty_like(q_golden).contiguous()
    k_out = torch.empty_like(k_golden).contiguous()
    v_out = torch.empty_like(v_golden).contiguous()

    kernels[kernel_name](
        q_input,
        k_input,
        v_input,
        q_gamma,
        k_gamma,
        cos_cache,
        sin_cache,
        indices,
        past_key,
        past_value,
        q_out,
        k_out,
        v_out,
    )

    _assert_cos_output(q_out, q_golden, "q_out")
    _assert_cos_output(k_out, k_golden, "k_out")
    _assert_cos_output(v_out, v_golden, "v_out")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
