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
Test Kirin dynamic-to-static codegen - common functions for Kirin9030 and KirinX90.

Covers the ``LitenpuDynamicToStatic`` pass:

- Controlflow staticization: ``pypto.loop`` / ``pypto.cond`` / ``pypto.is_loop_begin`` /
  ``pypto.is_loop_end`` / ``pypto.loop_unroll`` must be statically unrolled / resolved at
  compile time. A dynamic loop or branch (one depending on a -1 dim) is unsupported on
  LiteNPU (Kirin) and must fail fast with a clear error.
- Dynamic shape staticization: a kernel declaring ``pypto.DYNAMIC`` (a -1 axis) is resolved
  to its concrete runtime shape via ``register_static_shape_tensor`` + the pass's ResolveShape.

Reference docs: docs/zh/api/controlflow/
- pypto-loop
- pypto-cond
- pypto-is_loop_begin
- pypto-is_loop_end
- pypto-loop_unroll
"""

import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def _jit_kernel(soc_version, name):
    """Apply the shared @pypto.frontend.jit options and set the kernel's __name__."""
    def decorator(fn):
        kernel = pypto.frontend.jit(
            codegen_options={"soc_version": soc_version},
            runtime_options={"run_mode": pypto.RunMode.SIM},
            create_new_logical_tensor=True,
        )(fn)
        kernel.__name__ = name
        return kernel
    return decorator


def make_loop_add_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        for idx in pypto.loop(loop_num):
            start = idx * block
            end = (idx + 1) * block
            output[start:end, :] = pypto.add(input0[start:end, :], input1[start:end, :])

    return kernel


def make_loop_cond_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        for idx in pypto.loop(loop_num):
            start = idx * block
            end = (idx + 1) * block
            if pypto.cond(idx % 2 == 0):
                output[start:end, :] = pypto.add(input0[start:end, :], input1[start:end, :])
            else:
                output[start:end, :] = pypto.sub(input0[start:end, :], input1[start:end, :])

    return kernel


def make_loop_boundary_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        for idx in pypto.loop(loop_num):
            start = idx * block
            end = (idx + 1) * block
            if pypto.is_loop_begin(idx):
                output[start:end, :] = pypto.add(input0[start:end, :], input1[start:end, :])
            elif pypto.is_loop_end(idx):
                output[start:end, :] = pypto.sub(input0[start:end, :], input1[start:end, :])
            else:
                output[start:end, :] = pypto.mul(input0[start:end, :], input1[start:end, :])

    return kernel


def make_loop_unroll_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor([...], dtype),
        input1: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        for idx, unroll_factor in pypto.loop_unroll(loop_num, unroll_list=[1]):
            start = idx * block
            end = (idx + 1) * block
            output[start:end, :] = pypto.add(input0[start:end, :], input1[start:end, :])

    return kernel


def make_dynamic_loop_bound_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    """A loop whose trip count depends on a dynamic (-1) dim, staticized via the registered static shape."""
    dynamic_shape = [pypto.DYNAMIC, tile_shapes[1]]

    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor(dynamic_shape, dtype),
        input1: pypto.Tensor(dynamic_shape, dtype),
        output: pypto.Tensor(dynamic_shape, dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        for idx in pypto.loop(input0.shape[0] // block):
            start = idx * block
            end = (idx + 1) * block
            output[start:end, :] = pypto.add(input0[start:end, :], input1[start:end, :])

    return kernel


def make_dynamic_cond_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    """A static loop whose branch condition depends on a dynamic (-1) dim, staticized via the registered shape."""
    dynamic_shape = [pypto.DYNAMIC, tile_shapes[1]]

    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor(dynamic_shape, dtype),
        input1: pypto.Tensor(dynamic_shape, dtype),
        output: pypto.Tensor(dynamic_shape, dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        for idx in pypto.loop(loop_num):
            start = idx * block
            end = (idx + 1) * block
            if pypto.cond(input0.shape[0] > block):
                output[start:end, :] = pypto.add(input0[start:end, :], input1[start:end, :])
            else:
                output[start:end, :] = pypto.sub(input0[start:end, :], input1[start:end, :])

    return kernel


def make_standalone_dynamic_cond_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    """A standalone if/else whose branch condition depends on a dynamic (-1) dim, staticized via registered shape."""
    dynamic_shape = [pypto.DYNAMIC, tile_shapes[1]]

    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor(dynamic_shape, dtype),
        input1: pypto.Tensor(dynamic_shape, dtype),
        output: pypto.Tensor(dynamic_shape, dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        if pypto.cond(input0.shape[0] > block):
            output[:] = pypto.add(input0, input1)
        else:
            output[:] = pypto.sub(input0, input1)

    return kernel


def make_dynamic_add_kernel(soc_version, name, dtype, tile_shapes):
    dynamic_shape = [pypto.DYNAMIC] * len(tile_shapes)

    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor(dynamic_shape, dtype),
        input1: pypto.Tensor(dynamic_shape, dtype),
        output: pypto.Tensor(dynamic_shape, dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output[:] = pypto.add(input0, input1)

    return kernel


def make_dynamic_multi_kernel(soc_version, name, dtype, tile_shapes):
    """Multi-op chain: the intermediate tensor (tmp) is not a function arg, only the pass can staticize its -1."""
    dynamic_shape = [pypto.DYNAMIC] * len(tile_shapes)

    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor(dynamic_shape, dtype),
        input1: pypto.Tensor(dynamic_shape, dtype),
        input2: pypto.Tensor(dynamic_shape, dtype),
        output: pypto.Tensor(dynamic_shape, dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        tmp = pypto.add(input0, input1)      # intermediate tensor, shape=[-1]
        output[:] = pypto.add(tmp, input2)

    return kernel


def make_static_loop_dynamic_kernel(soc_version, name, dtype, tile_shapes, loop_num, block):
    """Static loop + DYNAMIC input/output: the pass must staticize the -1 dim after unrolling."""
    dynamic_shape = [pypto.DYNAMIC] + list(tile_shapes[1:])

    @_jit_kernel(soc_version, name)
    def kernel(
        input0: pypto.Tensor(dynamic_shape, dtype),
        input1: pypto.Tensor(dynamic_shape, dtype),
        output: pypto.Tensor(dynamic_shape, dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        for idx in pypto.loop(loop_num):
            start = idx * block
            end = (idx + 1) * block
            output[start:end, ...] = pypto.add(input0[start:end, ...], input1[start:end, ...])

    return kernel


# ---------------------------------------------------------------------------
# AdamW nested-loop kernel (mirrors
# pypto-gym/src/pypto_gym/ops/pypto_tensor/experimental/vector/ApplyAdamWV2/apply_adam_w_v2_impl.py):
# a 2D nested ``pypto.loop`` whose trip counts depend on the dynamic (-1) dims,
# with in-loop ``pypto.view`` / ``pypto.assemble``.
# ---------------------------------------------------------------------------
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_LR = 0.01
_ADAM_WEIGHT_DECAY = 0.01
_ADAM_EPS = 1e-8


def make_nested_loop_adam_kernel(soc_version, name, dtype, tile_shapes):
    """Nested 2D loop + DYNAMIC shape: single AdamW step without bias correction.

    The outer loop tiles the M axis, the inner loop tiles the N axis; both trip
    counts are derived from the DYNAMIC (-1) dims and must be staticized by the
    LitenpuDynamicToStatic pass before unrolling.
    """
    m_tile, n_tile = tile_shapes
    dynamic_shape = [pypto.DYNAMIC, pypto.DYNAMIC]

    @_jit_kernel(soc_version, name)
    def kernel(
        weight: pypto.Tensor(dynamic_shape, dtype),
        grad: pypto.Tensor(dynamic_shape, dtype),
        m: pypto.Tensor(dynamic_shape, dtype),
        v: pypto.Tensor(dynamic_shape, dtype),
        weight_out: pypto.Tensor(dynamic_shape, dtype),
        m_out: pypto.Tensor(dynamic_shape, dtype),
        v_out: pypto.Tensor(dynamic_shape, dtype),
    ):
        pypto.set_vec_tile_shapes(m_tile, n_tile)
        m_dim = weight.shape[0]
        n_dim = weight.shape[1]
        m_loops = pypto.ceildiv(m_dim, m_tile)
        n_loops = pypto.ceildiv(n_dim, n_tile)
        for m_idx in pypto.loop(m_loops):
            m_offset = m_idx * m_tile
            valid_m = (m_dim - m_offset).min(m_tile)
            for n_idx in pypto.loop(n_loops):
                n_offset = n_idx * n_tile
                valid_n = (n_dim - n_offset).min(n_tile)
                valid_shape = [valid_m, valid_n]
                w_tile = pypto.view(weight, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)
                g_tile = pypto.view(grad, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)
                m_state = pypto.view(m, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)
                v_state = pypto.view(v, [m_tile, n_tile], [m_offset, n_offset], valid_shape=valid_shape)

                m_new = pypto.add(pypto.mul(m_state, _ADAM_BETA1), pypto.mul(g_tile, 1.0 - _ADAM_BETA1))
                grad_sq = pypto.mul(g_tile, g_tile)
                v_new = pypto.add(pypto.mul(v_state, _ADAM_BETA2), pypto.mul(grad_sq, 1.0 - _ADAM_BETA2))
                denom = pypto.add(pypto.sqrt(v_new), _ADAM_EPS)
                update = pypto.add(pypto.div(m_new, denom), pypto.mul(w_tile, _ADAM_WEIGHT_DECAY))
                w_new = pypto.sub(w_tile, pypto.mul(update, _ADAM_LR))

                pypto.assemble(w_new, [m_offset, n_offset], weight_out)
                pypto.assemble(m_new, [m_offset, n_offset], m_out)
                pypto.assemble(v_new, [m_offset, n_offset], v_out)

    return kernel


LOOP_TEST_CASES = [
    # kernel_name, torch_dtype, pypto_dtype, tile_shapes, loop_num, block, shape, kind
    pytest.param(
        "loop_add_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (64, 64),
        4,
        64,
        (256, 64),
        "loop",
        marks=[pytest.mark.skip()],
        id="001",
    ),
    pytest.param(
        "loop_cond_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (64, 64),
        4,
        64,
        (256, 64),
        "loop_cond",
        marks=[pytest.mark.skip()],
        id="002",
    ),
    pytest.param(
        "loop_cond_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (32, 32),
        3,
        32,
        (96, 32),
        "loop_cond",
        marks=[pytest.mark.skip()],
        id="003",
    ),
    pytest.param(
        "loop_boundary_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (64, 64),
        4,
        64,
        (256, 64),
        "loop_boundary",
        marks=[pytest.mark.skip()],
        id="004",
    ),
    pytest.param(
        "loop_boundary_fp16_002",
        torch.float16,
        pypto.DT_FP16,
        (32, 32),
        3,
        32,
        (96, 32),
        "loop_boundary",
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "loop_unroll_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (64, 64),
        4,
        64,
        (256, 64),
        "loop_unroll",
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "loop_dynamic_4d_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (2, 4, 8, 16),
        4,
        2,
        (8, 4, 8, 16),
        "loop_dynamic",
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "loop_dynamic_bound_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (64, 64),
        0,
        64,
        (256, 64),
        "loop_dynamic_bound",
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "loop_dynamic_cond_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (64, 64),
        4,
        64,
        (256, 64),
        "loop_dynamic_cond",
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "standalone_dynamic_cond_fp16_001",
        torch.float16,
        pypto.DT_FP16,
        (64, 64),
        0,
        64,
        (256, 64),
        "standalone_dynamic_cond",
        marks=[pytest.mark.skip()],
        id="010",
    ),
]

DYNAMIC_TEST_CASES = [
    # kernel_name, torch_dtype, pypto_dtype, tile_shapes, shape
    pytest.param(
        "dynamic_add_fp32_001", torch.float32, pypto.DT_FP32, (100,), (112,), marks=[pytest.mark.skip()], id="001"
    ),
    pytest.param(
        "dynamic_add_fp16_002", torch.float16, pypto.DT_FP16, (64,), (128,), marks=[pytest.mark.skip()], id="002"
    ),
    pytest.param(
        "dynamic_add4d_fp32_003",
        torch.float32,
        pypto.DT_FP32,
        (2, 4, 8, 16),
        (2, 4, 8, 16),
        marks=[pytest.mark.skip()],
        id="003",
    ),
]

DYNAMIC_MULTI_TEST_CASES = [
    # kernel_name, torch_dtype, pypto_dtype, tile_shapes, shape
    pytest.param(
        "dynamic_multi_fp32_004",
        torch.float32,
        pypto.DT_FP32,
        (100,),
        (112,),
        marks=[pytest.mark.skip()],
        id="004",
    ),
]

ADAM_TEST_CASES = [
    # kernel_name, torch_dtype, pypto_dtype, tile_shapes, shape
    pytest.param(
        "adam_nested_loop_fp32_001",
        torch.float32,
        pypto.DT_FP32,
        (64, 64),
        (128, 256),
        marks=[],
        id="001",
    ),
    pytest.param(
        "adam_nested_loop_fp32_002",
        torch.float32,
        pypto.DT_FP32,
        (32, 128),
        (64, 384),
        marks=[pytest.mark.skip()],
        id="002",
    ),
]


def _compute_golden(input0, input1, loop_num, block, kind):
    if kind == "loop_dynamic_bound":
        # trip count is derived from the dynamic (-1) dim, resolved to the concrete static shape
        loop_num = input0.shape[0] // block
    if kind == "standalone_dynamic_cond":
        # standalone if/else; input0.shape[0] > block is statically true for these shapes -> add
        return input0 + input1
    golden = torch.empty_like(input0)
    for idx in range(loop_num):
        start = idx * block
        end = (idx + 1) * block
        if kind == "loop_cond" and idx % 2 != 0:
            golden[start:end, ...] = input0[start:end, ...] - input1[start:end, ...]
        elif kind == "loop_boundary":
            if idx == 0:
                golden[start:end, ...] = input0[start:end, ...] + input1[start:end, ...]
            elif idx == loop_num - 1:
                golden[start:end, ...] = input0[start:end, ...] - input1[start:end, ...]
            else:
                golden[start:end, ...] = input0[start:end, ...] * input1[start:end, ...]
        else:  # loop / loop_unroll / loop_dynamic / loop_dynamic_bound / loop_dynamic_cond
            golden[start:end, ...] = input0[start:end, ...] + input1[start:end, ...]
    return golden


def _run_kernel_test(kernels, kernel_name, inputs, golden):
    """Run a kernel with `inputs`, then check NaN and cosine similarity against `golden`."""
    output = torch.zeros_like(golden)
    kernels[kernel_name](*inputs, output)
    check_nan(output, name=kernel_name)
    cos_value = abs(compare_cos(output.cpu().numpy(), golden.cpu().numpy()))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def run_loop_test(kernels, kernel_name, dtype, shape, loop_num, block, kind):
    """Run a single controlflow kernel test with the given kernels dict."""
    device = "cpu"
    input0 = torch.randn(shape, dtype=dtype, device=device)
    input1 = torch.randn(shape, dtype=dtype, device=device)
    golden = _compute_golden(input0, input1, loop_num, block, kind)
    _run_kernel_test(kernels, kernel_name, [input0, input1], golden)


def run_dynamic_test(kernels, kernel_name, dtype, shape):
    """Run a single pypto.DYNAMIC add kernel test with the given kernels dict."""
    device = "cpu"
    input0 = torch.randn(shape, dtype=dtype, device=device)
    input1 = torch.randn(shape, dtype=dtype, device=device)
    _run_kernel_test(kernels, kernel_name, [input0, input1], input0 + input1)


def run_dynamic_multi_test(kernels, kernel_name, dtype, shape):
    """Run a multi-op pypto.DYNAMIC kernel (input0 + input1 + input2) with an intermediate tensor."""
    device = "cpu"
    input0 = torch.randn(shape, dtype=dtype, device=device)
    input1 = torch.randn(shape, dtype=dtype, device=device)
    input2 = torch.randn(shape, dtype=dtype, device=device)
    _run_kernel_test(kernels, kernel_name, [input0, input1, input2], input0 + input1 + input2)


def _compute_adam_golden(weight, grad, m, v):
    """Single AdamW step (no bias correction), element-wise, matching the nested-loop kernel."""
    m_new = _ADAM_BETA1 * m + (1.0 - _ADAM_BETA1) * grad
    v_new = _ADAM_BETA2 * v + (1.0 - _ADAM_BETA2) * grad * grad
    denom = torch.sqrt(v_new) + _ADAM_EPS
    update = m_new / denom + _ADAM_WEIGHT_DECAY * weight
    w_new = weight - _ADAM_LR * update
    return w_new, m_new, v_new


def _check_output(output, golden, name):
    """Check NaN and cosine similarity of a single output tensor against its golden."""
    check_nan(output, name=name)
    cos_value = abs(compare_cos(output.cpu().numpy(), golden.cpu().numpy()))
    if cos_value < 0.9999:
        raise AssertionError(f"{name}: cos_value {cos_value} < 0.9999")


def run_adam_test(kernels, kernel_name, dtype, shape, tile_shapes):
    """Run the nested-loop AdamW kernel and check all three outputs (weight/m/v)."""
    del tile_shapes  # tiling is baked into the kernel factory; only the concrete shape drives the golden
    device = "cpu"
    weight = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    m = torch.randn(shape, dtype=dtype, device=device)
    v = torch.rand(shape, dtype=dtype, device=device) + 1.0

    g_weight, g_m, g_v = _compute_adam_golden(weight, grad, m, v)

    w_out = torch.zeros_like(weight)
    m_out = torch.zeros_like(m)
    v_out = torch.zeros_like(v)
    kernels[kernel_name](weight, grad, m, v, w_out, m_out, v_out)

    _check_output(w_out, g_weight, kernel_name + "_weight")
    _check_output(m_out, g_m, kernel_name + "_m")
    _check_output(v_out, g_v, kernel_name + "_v")


def run_from_torch_test():
    """Test the pypto.from_torch interface (basic conversion / name / dynamic_axis / dtype / tensor_format)."""
    # 1. Basic conversion: auto-named, shape preserved
    x = torch.randn(2, 3, dtype=torch.float32)
    x_pto = pypto.from_torch(x)
    if list(x_pto.shape) != [2, 3]:
        raise AssertionError(f"from_torch default conversion shape mismatch: {x_pto.shape}")

    # 2. name + dynamic_axis: axis 0 marked dynamic, ori_shape preserves the original concrete shape
    y = torch.randn(2, 3, dtype=torch.float32)
    y_pto = pypto.from_torch(y, "y", dynamic_axis=[0])
    if y_pto.name != "y":
        raise AssertionError(f"name mismatch: {y_pto.name}")
    if list(y_pto.ori_shape) != [2, 3]:
        raise AssertionError(f"ori_shape mismatch: {y_pto.ori_shape}")
    if isinstance(y_pto.shape[0], int):
        raise AssertionError(f"dynamic_axis[0] should be SymbolicScalar, got: {y_pto.shape[0]}")

    # 3. Explicit dtype override (overrides auto-inference)
    k = torch.randn(2, 3, dtype=torch.float32)
    k_pto = pypto.from_torch(k, "k", dtype=pypto.DT_FP16)
    if k_pto.dtype != pypto.DT_FP16:
        raise AssertionError(f"dtype mismatch: {k_pto.dtype}")

    # 4. Explicit tensor_format
    z = torch.randn(2, 3, dtype=torch.float32)
    z_pto = pypto.from_torch(z, "z", tensor_format=pypto.TileOpFormat.TILEOP_NZ)
    if z_pto.format != pypto.TileOpFormat.TILEOP_NZ:
        raise AssertionError(f"format mismatch: {z_pto.format}")


_LOOP_FACTORY_BY_KIND = {
    "loop": make_loop_add_kernel,
    "loop_cond": make_loop_cond_kernel,
    "loop_boundary": make_loop_boundary_kernel,
    "loop_unroll": make_loop_unroll_kernel,
    "loop_dynamic": make_static_loop_dynamic_kernel,
    "loop_dynamic_bound": make_dynamic_loop_bound_kernel,
    "loop_dynamic_cond": make_dynamic_cond_kernel,
    "standalone_dynamic_cond": make_standalone_dynamic_cond_kernel,
}


def create_loop_kernels(soc_version):
    kernels = {}
    for p in LOOP_TEST_CASES:
        kernel_name, _, dtype, tile_shapes, loop_num, block, _, kind = p.values
        kernels[kernel_name] = _LOOP_FACTORY_BY_KIND[kind](
            soc_version, kernel_name, dtype, tile_shapes, loop_num, block
        )
    return kernels


def create_dynamic_kernels(soc_version):
    kernels = {}
    for cases, factory in ((DYNAMIC_TEST_CASES, make_dynamic_add_kernel),
                           (DYNAMIC_MULTI_TEST_CASES, make_dynamic_multi_kernel)):
        for p in cases:
            name, _, dtype, tile_shapes, _ = p.values
            kernels[name] = factory(soc_version, name, dtype, tile_shapes)
    return kernels


def create_adam_kernels(soc_version):
    kernels = {}
    for p in ADAM_TEST_CASES:
        name, _, dtype, tile_shapes, _ = p.values
        kernels[name] = make_nested_loop_adam_kernel(soc_version, name, dtype, tile_shapes)
    return kernels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
