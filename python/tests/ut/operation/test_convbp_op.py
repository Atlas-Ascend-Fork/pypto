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
Conv backward input (dx) and conv backward filter (dw) operation unit tests.
Tests cover:
- Different dtype: FP16, BF16
- Different conv parameters: stride, padding, dilation, batch
"""

import pytest

import pypto
from pypto.error import PyptoError


def test_convbp_2d_fp16_op():
    """Conv2D backward dx: FP16, stride=1, pad=1, dilation=1, 3x3 kernel"""
    dtype = pypto.DT_FP16
    # grad_output [N, Cout, Hout, Wout] = [1, 16, 16, 16]
    # weight [Cout, Cin, Kh, Kw] = [16, 16, 3, 3]
    # hin = 1*(16-1) + 1*(3-1) + 1 - 1 - 1 = 16, win = 16
    grad_output = pypto.tensor((1, 16, 16, 16), dtype, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), dtype, "weight")
    input_size = [1, 16, 16, 16]

    with pypto.function("CONV_BP", grad_output, weight):
        pypto.set_convbp_input_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=16, tileKL1=144),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_input(
            grad_output, input_size, weight, dtype,
            [1, 1], [1, 1, 1, 1], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [1, 16, 16, 16]


def test_convbp_2d_bf16_op():
    """Conv2D backward dx: BF16, stride=1, pad=1, dilation=1"""
    dtype = pypto.DT_BF16
    grad_output = pypto.tensor((1, 16, 16, 16), dtype, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), dtype, "weight")
    input_size = [1, 16, 16, 16]

    with pypto.function("CONV_BP", grad_output, weight):
        pypto.set_convbp_input_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=16, tileKL1=144),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_input(
            grad_output, input_size, weight, dtype,
            [1, 1], [1, 1, 1, 1], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [1, 16, 16, 16]


def test_convbp_2d_dilation_op():
    """Conv2D backward dx: FP16, stride=1, pad=1, dilation=2, 3x3 kernel"""
    dtype = pypto.DT_FP16
    # hin = 1*(16-1) + 2*(3-1) + 1 - 1 - 1 = 15 + 4 + 1 - 2 = 18
    # win = 18
    grad_output = pypto.tensor((1, 16, 16, 16), dtype, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), dtype, "weight")
    input_size = [1, 16, 18, 18]

    with pypto.function("CONV_BP", grad_output, weight):
        pypto.set_convbp_input_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=16, tileKL1=144),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_input(
            grad_output, input_size, weight, dtype,
            [1, 1], [1, 1, 1, 1], [2, 2], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [1, 16, 18, 18]


def test_convbp_2d_no_pad_op():
    """Conv2D backward dx: FP16, stride=1, pad=0, dilation=1, 3x3 kernel"""
    dtype = pypto.DT_FP16
    # hin = 1*(14-1) + 1*(3-1) + 1 - 0 - 0 = 13 + 2 + 1 = 16
    # win = 16
    grad_output = pypto.tensor((1, 16, 14, 14), dtype, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), dtype, "weight")
    input_size = [1, 16, 16, 16]

    with pypto.function("CONV_BP", grad_output, weight):
        pypto.set_convbp_input_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=16, tileKL1=144),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_input(
            grad_output, input_size, weight, dtype,
            [1, 1], [0, 0, 0, 0], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [1, 16, 16, 16]


def test_convbp_2d_fp16_batch_op():
    """Conv2D backward dx: FP16, batch=2"""
    dtype = pypto.DT_FP16
    grad_output = pypto.tensor((2, 16, 16, 16), dtype, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), dtype, "weight")
    input_size = [2, 16, 16, 16]

    with pypto.function("CONV_BP", grad_output, weight):
        pypto.set_convbp_input_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=16, tileKL1=144),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_input(
            grad_output, input_size, weight, dtype,
            [1, 1], [1, 1, 1, 1], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [2, 16, 16, 16]


def test_convbp_grad_output_unsupported_dtype():
    """grad_output dtype must be in [bf16, fp16, fp32]; INT8 is not supported."""
    grad_output = pypto.tensor((1, 16, 16, 16), pypto.DT_INT8, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), pypto.DT_INT8, "weight")
    with pytest.raises(PyptoError, match="Input tensor data type must in"):
        pypto.conv_backward_input(
            grad_output, [1, 16, 16, 16], weight, pypto.DT_INT8,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_weight_unsupported_dtype():
    """weight dtype must be in [bf16, fp16, fp32]; INT8 is not supported."""
    grad_output = pypto.tensor((1, 16, 16, 16), pypto.DT_FP16, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), pypto.DT_INT8, "weight")
    with pytest.raises(PyptoError, match="Weight tensor data type must in"):
        pypto.conv_backward_input(
            grad_output, [1, 16, 16, 16], weight, pypto.DT_FP16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_grad_weight_dtype_inconsistency():
    """grad_output and weight must have the same data type."""
    grad_output = pypto.tensor((1, 16, 16, 16), pypto.DT_FP16, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), pypto.DT_BF16, "weight")
    with pytest.raises(PyptoError, match="grad_output and weight data types must be consistent"):
        pypto.conv_backward_input(
            grad_output, [1, 16, 16, 16], weight, pypto.DT_FP16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_out_dtype_grad_output_inconsistency():
    """out_dtype must match grad_output data type."""
    grad_output = pypto.tensor((1, 16, 16, 16), pypto.DT_FP16, "grad_output")
    weight = pypto.tensor((16, 16, 3, 3), pypto.DT_FP16, "weight")
    with pytest.raises(PyptoError, match="Output data type must be consistent with grad_output"):
        pypto.conv_backward_input(
            grad_output, [1, 16, 16, 16], weight, pypto.DT_BF16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_filter_2d_fp16_op():
    """Conv2D backward dw: FP16, stride=1, pad=1, dilation=1, 2x2 kernel"""
    dtype = pypto.DT_FP16
    # grad_output [N, Cout, Hout, Wout] = [1, 16, 5, 16]
    # input [N, Cin, Hin, Win] = [1, 16, 4, 15]
    # filter_size [Cout, Cin, Kh, Kw] = [16, 16, 2, 2]
    # hout = (4 + 1 + 1 - 1 * (2 - 1) - 1) // 1 + 1 = 5, wout = (15 + 1 + 1 - 1 - 1) // 1 + 1 = 16
    grad_output = pypto.tensor((1, 16, 5, 16), dtype, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), dtype, "input")
    filter_size = [16, 16, 2, 2]

    with pypto.function("CONV_BP_DW", grad_output, input_):
        pypto.set_convbp_filter_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=64, tileKL1=80),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=64, tileKL0=80),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_filter(
            grad_output, input_, filter_size, dtype,
            [1, 1], [1, 1, 1, 1], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [16, 16, 2, 2]


def test_convbp_filter_2d_bf16_op():
    """Conv2D backward dw: BF16, stride=1, pad=1, dilation=1, 2x2 kernel"""
    dtype = pypto.DT_BF16
    grad_output = pypto.tensor((1, 16, 5, 16), dtype, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), dtype, "input")
    filter_size = [16, 16, 2, 2]

    with pypto.function("CONV_BP_DW", grad_output, input_):
        pypto.set_convbp_filter_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=64, tileKL1=80),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=64, tileKL0=80),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_filter(
            grad_output, input_, filter_size, dtype,
            [1, 1], [1, 1, 1, 1], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [16, 16, 2, 2]


def test_convbp_filter_2d_dilation_op():
    """Conv2D backward dw: FP16, stride=1, pad=2, dilation=2, 3x3 kernel"""
    dtype = pypto.DT_FP16
    # hout = (8 + 2 + 2 - 2 * (3 - 1) - 1) // 1 + 1 = 8, wout = 8
    grad_output = pypto.tensor((1, 16, 8, 8), dtype, "grad_output")
    input_ = pypto.tensor((1, 16, 8, 8), dtype, "input")
    filter_size = [16, 16, 3, 3]

    with pypto.function("CONV_BP_DW", grad_output, input_):
        pypto.set_convbp_filter_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=144, tileKL1=16),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_filter(
            grad_output, input_, filter_size, dtype,
            [1, 1], [2, 2, 2, 2], [2, 2], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [16, 16, 3, 3]


def test_convbp_filter_2d_no_pad_op():
    """Conv2D backward dw: FP16, stride=1, pad=0, dilation=1, 3x3 kernel"""
    dtype = pypto.DT_FP16
    # hout = (16 + 0 + 0 - 1 * (3 - 1) - 1) // 1 + 1 = 14, wout = 14
    grad_output = pypto.tensor((1, 16, 14, 14), dtype, "grad_output")
    input_ = pypto.tensor((1, 16, 16, 16), dtype, "input")
    filter_size = [16, 16, 3, 3]

    with pypto.function("CONV_BP_DW", grad_output, input_):
        pypto.set_convbp_filter_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=144, tileKL1=28),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_filter(
            grad_output, input_, filter_size, dtype,
            [1, 1], [0, 0, 0, 0], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [16, 16, 3, 3]


def test_convbp_filter_2d_fp16_batch_op():
    """Conv2D backward dw: FP16, batch=2"""
    dtype = pypto.DT_FP16
    # hout = (8 + 1 + 1 - 1 * (3 - 1) - 1) // 1 + 1 = 8, wout = 8
    grad_output = pypto.tensor((2, 16, 8, 8), dtype, "grad_output")
    input_ = pypto.tensor((2, 16, 8, 8), dtype, "input")
    filter_size = [16, 16, 3, 3]

    with pypto.function("CONV_BP_DW", grad_output, input_):
        pypto.set_convbp_filter_tile_shapes(
            pypto.pypto_impl.ConvBpTileL1Info(tileML1=16, tileNL1=144, tileKL1=16),
            pypto.pypto_impl.ConvBpTileL0Info(tileML0=16, tileNL0=16, tileKL0=16),
        )
        pypto.set_vec_tile_shapes(16, 16, 16, 16)
        result = pypto.conv_backward_filter(
            grad_output, input_, filter_size, dtype,
            [1, 1], [1, 1, 1, 1], [1, 1], groups=1
        )

    assert isinstance(result, pypto.tensor)
    assert result.shape == [16, 16, 3, 3]


def test_convbp_filter_grad_output_unsupported_dtype():
    """dw: grad_output dtype must be in [bf16, fp16, fp32]; INT8 is not supported."""
    grad_output = pypto.tensor((1, 16, 5, 16), pypto.DT_INT8, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), pypto.DT_INT8, "input")
    with pytest.raises(PyptoError, match="Input tensor data type must in"):
        pypto.conv_backward_filter(
            grad_output, input_, [16, 16, 2, 2], pypto.DT_INT8,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_filter_input_unsupported_dtype():
    """dw: input dtype must be in [bf16, fp16, fp32]; INT8 is not supported."""
    grad_output = pypto.tensor((1, 16, 5, 16), pypto.DT_FP16, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), pypto.DT_INT8, "input")
    with pytest.raises(PyptoError, match="Input tensor data type must in"):
        pypto.conv_backward_filter(
            grad_output, input_, [16, 16, 2, 2], pypto.DT_FP16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_filter_grad_input_dtype_inconsistency():
    """dw: grad_output and input must have the same data type."""
    grad_output = pypto.tensor((1, 16, 5, 16), pypto.DT_FP16, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), pypto.DT_BF16, "input")
    with pytest.raises(PyptoError, match="grad_output and input data types must be consistent"):
        pypto.conv_backward_filter(
            grad_output, input_, [16, 16, 2, 2], pypto.DT_FP16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_filter_out_dtype_grad_output_inconsistency():
    """dw: out_dtype must match grad_output data type."""
    grad_output = pypto.tensor((1, 16, 5, 16), pypto.DT_FP16, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), pypto.DT_FP16, "input")
    with pytest.raises(PyptoError, match="Output data type must be consistent with grad_output"):
        pypto.conv_backward_filter(
            grad_output, input_, [16, 16, 2, 2], pypto.DT_BF16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_filter_not_2d_dim():
    """dw: only supports 2D conv, grad_output must be 4D."""
    grad_output = pypto.tensor((16, 16, 16), pypto.DT_FP16, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), pypto.DT_FP16, "input")
    with pytest.raises(PyptoError, match="ConvBackwardFilter only supports 2D conv"):
        pypto.conv_backward_filter(
            grad_output, input_, [16, 16, 2, 2], pypto.DT_FP16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )


def test_convbp_filter_invalid_filter_size_len():
    """dw: filter_size must have length 4 for 2D conv."""
    grad_output = pypto.tensor((1, 16, 5, 16), pypto.DT_FP16, "grad_output")
    input_ = pypto.tensor((1, 16, 4, 15), pypto.DT_FP16, "input")
    with pytest.raises(PyptoError, match="filter_size must have length 4"):
        pypto.conv_backward_filter(
            grad_output, input_, [16, 16, 2], pypto.DT_FP16,
            [1, 1], [1, 1, 1, 1], [1, 1]
        )
