#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Conv backward input (dx) and conv backward filter (dw) system tests.
Tests run on actual NPU hardware and verify numerical correctness against PyTorch reference.
"""

import os

import pytest
import torch

import pypto
from pypto import pypto_impl


def create_convbp_kernel(
    grad_output_shape,
    input_size,
    weight_shape,
    dtype,
    tile_l1_info,
    tile_l0_info,
    strides,
    pads,
    dilations,
    groups=1,
):
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0})
    def convbp_kernel(
        grad_output: pypto.Tensor(grad_output_shape, dtype),
        weight: pypto.Tensor(weight_shape, dtype),
        input_grad: pypto.Tensor(input_size, dtype),
    ):
        pypto.set_convbp_input_tile_shapes(tile_l1_info, tile_l0_info)
        pypto.set_vec_tile_shapes(16, 16, 1, 16, 16)
        output = pypto.conv_backward_input(
            grad_output, input_size, weight, dtype, strides, pads, dilations, groups=groups
        )
        input_grad.move(output)

    return convbp_kernel


def get_torch_conv_bp_golden(
    grad_output, input_tensor, weight, strides, paddings, dilations, groups=1
):
    """Get PyTorch reference result for conv backward input"""
    grad_input, _, _ = torch.ops.aten.convolution_backward(
        grad_output,
        input_tensor,
        weight,
        bias_sizes=[],
        stride=strides,
        padding=paddings,
        dilation=dilations,
        transposed=False,
        output_padding=[0] * len(strides),
        groups=groups,
        output_mask=[True, False, False]
    )
    return grad_input


@pytest.mark.soc("910")
def test_convbp_2d_fp16_basic():
    """Conv2D backward dx: FP16, stride=1, pad=1, dilation=1, 3x3 kernel"""
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    # Conv parameters
    n, cin, cout = 1, 16, 16
    hin, win = 5, 32
    kh, kw = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dil_h, dil_w = 1, 1
    groups = 1

    # Compute output shape
    hout = (hin + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    wout = (win + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

    grad_output_shape = (n, cout, hout, wout)
    weight_shape = (cout, cin // groups, kh, kw)
    input_grad_shape = (n, cin, hin, win)

    dtype = pypto.DT_FP16
    dtype_torch = torch.float16

    tile_l1_info = pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=144, tileNL1=16
    )
    tile_l0_info = pypto_impl.ConvBpTileL0Info(
        tileML0=16, tileKL0=16, tileNL0=16
    )

    strides = [stride_h, stride_w]
    pads = [pad_h, pad_h, pad_w, pad_w]
    dilations = [dil_h, dil_w]

    # Prepare input tensors
    grad_out = torch.rand(grad_output_shape, dtype=dtype_torch, device='npu')
    weight = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    input_grad = torch.zeros(input_grad_shape, dtype=dtype_torch, device='npu')

    # Create dummy input for golden computation
    input_tensor = torch.rand(input_grad_shape, dtype=dtype_torch, device='npu')

    # Run PyPTO kernel
    create_convbp_kernel(
        grad_output_shape, input_grad_shape, weight_shape,
        dtype, tile_l1_info, tile_l0_info, strides, pads, dilations, groups
    )(grad_out, weight, input_grad)

    # Get PyTorch golden
    golden = get_torch_conv_bp_golden(
        grad_out.cpu(), input_tensor.cpu(), weight.cpu(),
        strides=(stride_h, stride_w), paddings=(pad_h, pad_w),
        dilations=(dil_h, dil_w), groups=groups
    )

    assert torch.allclose(
        input_grad.cpu().to(dtype_torch),
        golden.cpu().to(dtype_torch),
        atol=1e-2, rtol=1e-2
    )


@pytest.mark.soc("910")
def test_convbp_2d_fp16_stride():
    """Conv2D backward dx: FP16, stride=2, pad=1, dilation=1"""
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    n, cin, cout = 1, 16, 16
    hin, win = 13, 13
    kh, kw = 3, 3
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 1, 1
    dil_h, dil_w = 1, 1
    groups = 1

    hout = (hin + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    wout = (win + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

    grad_output_shape = (n, cout, hout, wout)
    weight_shape = (cout, cin // groups, kh, kw)
    input_grad_shape = (n, cin, hin, win)

    dtype = pypto.DT_FP16
    dtype_torch = torch.float16

    tile_l1_info = pypto_impl.ConvBpTileL1Info(
        tileML1=13, tileKL1=144, tileNL1=16
    )
    tile_l0_info = pypto_impl.ConvBpTileL0Info(
        tileML0=16, tileKL0=16, tileNL0=16
    )

    strides = [stride_h, stride_w]
    pads = [pad_h, pad_h, pad_w, pad_w]
    dilations = [dil_h, dil_w]

    grad_out = torch.rand(grad_output_shape, dtype=dtype_torch, device='npu')
    weight = torch.rand(weight_shape, dtype=dtype_torch, device='npu')
    input_grad = torch.zeros(input_grad_shape, dtype=dtype_torch, device='npu')
    input_tensor = torch.rand(input_grad_shape, dtype=dtype_torch, device='npu')

    create_convbp_kernel(
        grad_output_shape, input_grad_shape, weight_shape,
        dtype, tile_l1_info, tile_l0_info, strides, pads, dilations, groups
    )(grad_out, weight, input_grad)

    golden = get_torch_conv_bp_golden(
        grad_out.cpu(), input_tensor.cpu(), weight.cpu(),
        strides=(stride_h, stride_w), paddings=(pad_h, pad_w),
        dilations=(dil_h, dil_w), groups=groups
    )

    assert torch.allclose(
        input_grad.cpu().to(dtype_torch),
        golden.cpu().to(dtype_torch),
        atol=1e-2, rtol=1e-2
    )


def create_convbp_filter_kernel(
    grad_output_shape,
    input_shape,
    filter_size,
    dw_shape,
    dtype,
    tile_l1_info,
    tile_l0_info,
    strides,
    pads,
    dilations,
    groups=1,
):
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0},
        runtime_options={"run_mode": pypto.RunMode.NPU})
    def convbp_dw_kernel(
        grad_output: pypto.Tensor(grad_output_shape, dtype),
        input_: pypto.Tensor(input_shape, dtype),
        out: pypto.Tensor(dw_shape, dtype),
    ):
        pypto.set_convbp_filter_tile_shapes(tile_l1_info, tile_l0_info)
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        output = pypto.conv_backward_filter(
            grad_output, input_, filter_size, dtype, strides, pads, dilations, groups=groups
        )
        out.move(output)

    return convbp_dw_kernel


def get_torch_conv_dw_golden(
    grad_output, input_tensor, weight, strides, paddings, dilations, groups=1
):
    """Get PyTorch reference result for conv backward filter"""
    _, grad_weight, _ = torch.ops.aten.convolution_backward(
        grad_output,
        input_tensor,
        weight,
        bias_sizes=[],
        stride=strides,
        padding=paddings,
        dilation=dilations,
        transposed=False,
        output_padding=[0] * len(strides),
        groups=groups,
        output_mask=[False, True, False]
    )
    return grad_weight


@pytest.mark.soc("910")
def test_convbp_filter_2d_fp16_basic():
    """Conv2D backward dw: FP16, stride=1, pad=1, dilation=1, 2x2 kernel"""
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    # Conv parameters
    n, cin, cout = 1, 16, 16
    hin, win = 4, 15
    kh, kw = 2, 2
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dil_h, dil_w = 1, 1
    groups = 1

    # Compute output shape
    hout = (hin + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    wout = (win + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

    grad_output_shape = (n, cout, hout, wout)
    input_shape = (n, cin, hin, win)
    filter_size = [cout, cin // groups, kh, kw]
    dw_shape = (cout, cin // groups, kh, kw)

    dtype = pypto.DT_FP16
    dtype_torch = torch.float16

    # dw: tileNL1 must be aligned to cin0*Kh*Kw; tileKL1 must be multiple of wout
    tile_l1_info = pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=80, tileNL1=64
    )
    tile_l0_info = pypto_impl.ConvBpTileL0Info(
        tileML0=16, tileKL0=80, tileNL0=64
    )

    strides = [stride_h, stride_w]
    pads = [pad_h, pad_h, pad_w, pad_w]
    dilations = [dil_h, dil_w]

    # Prepare input tensors
    grad_out = torch.rand(grad_output_shape, dtype=dtype_torch, device='npu')
    input_ = torch.rand(input_shape, dtype=dtype_torch, device='npu')
    out = torch.zeros(dw_shape, dtype=dtype_torch, device='npu')

    # Create dummy weight for golden computation (only grad_weight is computed)
    weight = torch.zeros(dw_shape, dtype=dtype_torch, device='npu')

    # Run PyPTO kernel
    create_convbp_filter_kernel(
        grad_output_shape, input_shape, filter_size, dw_shape,
        dtype, tile_l1_info, tile_l0_info, strides, pads, dilations, groups
    )(grad_out, input_, out)

    # Get PyTorch golden
    golden = get_torch_conv_dw_golden(
        grad_out.cpu(), input_.cpu(), weight.cpu(),
        strides=(stride_h, stride_w), paddings=(pad_h, pad_w),
        dilations=(dil_h, dil_w), groups=groups
    )

    assert torch.allclose(
        out.cpu().to(dtype_torch),
        golden.cpu().to(dtype_torch),
        atol=1e-2, rtol=1e-2
    )


@pytest.mark.soc("910")
def test_convbp_filter_2d_fp16_stride():
    """Conv2D backward dw: FP16, stride=2, pad=1, dilation=1, 3x3 kernel"""
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    n, cin, cout = 1, 16, 16
    hin, win = 8, 8
    kh, kw = 3, 3
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 1, 1
    dil_h, dil_w = 1, 1
    groups = 1

    hout = (hin + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    wout = (win + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

    grad_output_shape = (n, cout, hout, wout)
    input_shape = (n, cin, hin, win)
    filter_size = [cout, cin // groups, kh, kw]
    dw_shape = (cout, cin // groups, kh, kw)

    dtype = pypto.DT_FP16
    dtype_torch = torch.float16

    tile_l1_info = pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=16, tileNL1=144
    )
    tile_l0_info = pypto_impl.ConvBpTileL0Info(
        tileML0=16, tileKL0=16, tileNL0=16
    )

    strides = [stride_h, stride_w]
    pads = [pad_h, pad_h, pad_w, pad_w]
    dilations = [dil_h, dil_w]

    grad_out = torch.rand(grad_output_shape, dtype=dtype_torch, device='npu')
    input_ = torch.rand(input_shape, dtype=dtype_torch, device='npu')
    out = torch.zeros(dw_shape, dtype=dtype_torch, device='npu')
    weight = torch.zeros(dw_shape, dtype=dtype_torch, device='npu')

    create_convbp_filter_kernel(
        grad_output_shape, input_shape, filter_size, dw_shape,
        dtype, tile_l1_info, tile_l0_info, strides, pads, dilations, groups
    )(grad_out, input_, out)

    golden = get_torch_conv_dw_golden(
        grad_out.cpu(), input_.cpu(), weight.cpu(),
        strides=(stride_h, stride_w), paddings=(pad_h, pad_w),
        dilations=(dil_h, dil_w), groups=groups
    )

    assert torch.allclose(
        out.cpu().to(dtype_torch),
        golden.cpu().to(dtype_torch),
        atol=1e-2, rtol=1e-2
    )


@pytest.mark.soc("910")
def test_convbp_filter_2d_fp16_batch():
    """Conv2D backward dw: FP16, batch=2"""
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))

    n, cin, cout = 2, 16, 16
    hin, win = 4, 15
    kh, kw = 2, 2
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dil_h, dil_w = 1, 1
    groups = 1

    hout = (hin + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    wout = (win + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

    grad_output_shape = (n, cout, hout, wout)
    input_shape = (n, cin, hin, win)
    filter_size = [cout, cin // groups, kh, kw]
    dw_shape = (cout, cin // groups, kh, kw)

    dtype = pypto.DT_FP16
    dtype_torch = torch.float16

    tile_l1_info = pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=80, tileNL1=64
    )
    tile_l0_info = pypto_impl.ConvBpTileL0Info(
        tileML0=16, tileKL0=80, tileNL0=64
    )

    strides = [stride_h, stride_w]
    pads = [pad_h, pad_h, pad_w, pad_w]
    dilations = [dil_h, dil_w]

    grad_out = torch.rand(grad_output_shape, dtype=dtype_torch, device='npu')
    input_ = torch.rand(input_shape, dtype=dtype_torch, device='npu')
    out = torch.zeros(dw_shape, dtype=dtype_torch, device='npu')
    weight = torch.zeros(dw_shape, dtype=dtype_torch, device='npu')

    create_convbp_filter_kernel(
        grad_output_shape, input_shape, filter_size, dw_shape,
        dtype, tile_l1_info, tile_l0_info, strides, pads, dilations, groups
    )(grad_out, input_, out)

    golden = get_torch_conv_dw_golden(
        grad_out.cpu(), input_.cpu(), weight.cpu(),
        strides=(stride_h, stride_w), paddings=(pad_h, pad_w),
        dilations=(dil_h, dil_w), groups=groups
    )

    assert torch.allclose(
        out.cpu().to(dtype_torch),
        golden.cpu().to(dtype_torch),
        atol=1e-2, rtol=1e-2
    )
