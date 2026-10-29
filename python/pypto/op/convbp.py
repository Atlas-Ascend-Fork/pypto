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
"""PyPTO"""
from dataclasses import dataclass
from typing import Any, Type

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..enum import DataType
from ..error import PyptoError
from ..tensor import Tensor

_VALID_DATA_TYPES = (
    pypto_impl.DataType.DT_BF16,
    pypto_impl.DataType.DT_FP16,
    pypto_impl.DataType.DT_FP32
)


@dataclass
class ConvBpParams:
    out_dtype: Any
    input_size: list
    grad_output: Tensor
    weight: Tensor
    strides: list
    paddings: list
    dilations: list
    groups: int
    transposed: bool
    output_paddings: list
    extend_params: Any


@op_wrapper
def conv_backward_input(
    grad_output,
    input_size,
    weight,
    out_dtype,
    strides,
    paddings,
    dilations,
    *,
    groups=1
) -> Tensor:
    """
    Performs convolution backward operation with support for 1D/2D/3D convolution.

    Computes the gradient of the convolution operation with respect to the input tensor.

    Parameters
    ----------
    grad_output : Tensor
        The gradient of the loss with respect to the output (output_grad).
        Shape: (N, Cout, H_out, W_out) for 2D convolution.
    weight : Tensor
        The convolution kernel tensor (same as forward convolution).
        Shape: (Cout, Cin, Kh, Kw) for 2D convolution.
    out_dtype : dtype
        The data type of the output tensor (input gradient).
    strides : list/tuple of int
        Unidirectional stride values for convolution, with length matching the convolution dimension:
        - 1D conv: [stride_w]
        - 2D conv: [stride_h, stride_w]
        - 3D conv: [stride_d, stride_h, stride_w]
    paddings : list/tuple of int
        Bidirectional padding values for convolution, length is 2 x convolution dimension:
        - 1D conv: [padding_w_left, padding_w_right]
        - 2D conv: [padding_h_top, padding_h_bottom, padding_w_left, padding_w_right]
        - 3D conv: [padding_d_front, padding_d_back, padding_h_top, padding_h_bottom,
                    padding_w_left, padding_w_right]
    dilations : list/tuple of int
        Unidirectional dilation rates for convolution, with length matching the convolution dimension:
        - 1D conv: [dilation_w]
        - 2D conv: [dilation_h, dilation_w]
        - 3D conv: [dilation_d, dilation_h, dilation_w]

    Keyword Arguments
    ----------
    groups : int, default=1
        Number of groups for grouped convolution.
    transposed : bool, default=False
        If True, perform transposed convolution backward. Currently not supported.
    output_paddings : list/tuple of int, default=[]
        Output padding values for transposed convolution, only used when `transposed=True`.
        Length matches the convolution dimension (1D/2D/3D).
    extend_params : dict, optional
        Extended parameters:
        - 'bias_tensor': Tensor

    Returns
    -------
    Tensor
        A new tensor containing the input gradient.

    Raises
    ------
    RuntimeError
        If input/weight dimensions are invalid.
    ValueError
        If input parameters (strides/paddings/dilations) have invalid lengths, or if groups is not a divisor of
        input/weight channels.

    Examples
    --------
    output_grad = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "output_grad")
    weight = pypto.tensor((32, 32, 3, 3), pypto.DT_FP16, "weight")
    input_grad = pypto.conv_backward_input(
        output_grad, weight, pypto.DT_FP16,
        strides=[1, 1], paddings=[0, 0, 0, 0], dilations=[1, 1]
    )
    """
    params = ConvBpParams(
        out_dtype=out_dtype,
        grad_output=grad_output,
        input_size=input_size,
        weight=weight,
        strides=strides,
        paddings=paddings,
        dilations=dilations,
        groups=groups,
        transposed=False,
        output_paddings=None,
        extend_params=None
    )
    __validate_inputs(params)


    return pypto_impl.ConvBackwardInput(
        params.out_dtype, params.grad_output, params.input_size, params.weight,
        params.strides, params.paddings, params.dilations,
        pypto_impl.ConvBpExtendParam(), params.groups
    )


@op_wrapper
def conv_backward_filter(
    grad_output,
    input,
    filter_size,
    out_dtype,
    strides,
    paddings,
    dilations,
    *,
    groups=1
) -> Tensor:
    """
    Performs convolution backward operation with respect to the filter (weight), 2D only.

    Computes the gradient of the convolution operation with respect to the weight tensor.

    Parameters
    ----------
    grad_output : Tensor
        The gradient of the loss with respect to the output (output_grad).
        Shape: (N, Cout, Hout, Wout).
    input : Tensor
        The original input tensor of the forward convolution.
        Shape: (N, Cin, Hin, Win).
    filter_size : list of int
        The shape of the convolution kernel filter: [Cout, Cin_per_group, Kh, Kw].
    out_dtype : dtype
        The data type of the output tensor (weight gradient), must be consistent with inputs.
    strides : list of int
        Unidirectional stride values for convolution:
        - 2D conv: [stride_h, stride_w]
    paddings : list of int
        Bidirectional padding values for convolution:
        - 2D conv: [padding_top, padding_bottom, padding_left, padding_right]
    dilations : list of int
        Unidirectional dilation rates for convolution:
        - 2D conv: [dilation_h, dilation_w]

    Keyword Arguments
    ----------
    groups : int, default=1
        Number of groups for grouped convolution.

    Returns
    -------
    Tensor
        A new tensor containing the weight gradient.
        Shape: (Cout, Cin, Kh, Kw).

    Raises
    ------
    ValueError
        If input dimensions are invalid or not 2D, if input parameters
        (strides/paddings/dilations/filter_size) have invalid lengths,
        or if data types are inconsistent.

    Examples
    --------
    grad_output = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "grad_output")
    input = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input")
    weight_grad = pypto.conv_backward_filter(
        grad_output, input, [32, 32, 3, 3], pypto.DT_FP16,
        strides=[1, 1], paddings=[0, 0, 0, 0], dilations=[1, 1]
    )
    """
    __validate_filter_inputs(grad_output, input, filter_size, out_dtype, strides, paddings, dilations, groups)

    return pypto_impl.ConvBackwardFilter(
        out_dtype, grad_output, input, filter_size,
        strides, paddings, dilations, groups
    )


def __validate_type(value: Any, expect_type: Type, arg_name: str = "input") -> None:
    if value is None:
        return
    if not isinstance(value, expect_type):
        raise PyptoError(0xF00001, TypeError(
            f"Argument '{arg_name}' must be of type {expect_type.__name__}, "
            f"but got {type(value).__name__}."
        ))


def __validate_shape(grad_output: Tensor, weight: Tensor, transposed: bool) -> None:
    grad_output_dim = grad_output.Dim()
    weight_dim = weight.Dim()
    if grad_output_dim != weight_dim or grad_output_dim not in {3, 4, 5}:
        raise PyptoError(0xF00003, RuntimeError(
            "Tensor dimension mismatch. Expect grad_output_dim == weight_dim and both in [3, 4, 5], "
            f"got grad_output_dim: {grad_output_dim}, weight_dim: {weight_dim}."
        ))


def __validate_inputs(params: ConvBpParams) -> None:
    __validate_type(params.grad_output, pypto_impl.Tensor, "grad_output")
    __validate_type(params.weight, pypto_impl.Tensor, "weight")
    __validate_type(params.out_dtype, DataType, "out_dtype")
    __validate_type(params.strides, list, "strides")
    __validate_type(params.paddings, list, "paddings")
    __validate_type(params.dilations, list, "dilations")
    __validate_type(params.groups, int, "groups")
    __validate_type(params.transposed, bool, "transposed")
    __validate_type(params.output_paddings, list, "output_paddings")
    __validate_type(params.extend_params, dict, "extend_params")
    __validate_shape(params.grad_output, params.weight, False)

    if params.extend_params is not None and 'bias_tensor' in params.extend_params:
        bias = params.extend_params['bias_tensor']
        if bias is not None:
            __validate_type(bias, pypto_impl.Tensor, "bias_tensor")

    if params.grad_output.GetDataType() not in _VALID_DATA_TYPES:
        raise PyptoError(0xF00002, ValueError(
            "Input tensor data type must in [bf16, fp16, fp32],"
            f"but Input tensor got {params.grad_output.GetDataType()}"
        ))

    if params.weight.GetDataType() not in _VALID_DATA_TYPES:
        raise PyptoError(0xF00002, ValueError(
            "Weight tensor data type must in [bf16, fp16, fp32],"
            f"but Weight tensor got {params.weight.GetDataType()}"
        ))

    __validate_data_type_consistency(params)


def __validate_data_type_consistency(params: ConvBpParams) -> None:
    if params.grad_output.GetDataType() != params.weight.GetDataType():
        raise PyptoError(0xF00002, ValueError(
            f"grad_output and weight data types must be consistent, "
            f"but got grad_output: {params.grad_output.GetDataType()}, weight: {params.weight.GetDataType()}"
        ))

    if params.out_dtype != params.grad_output.GetDataType():
        raise PyptoError(0xF00002, ValueError(
            f"Output data type must be consistent with grad_output, "
            f"but got out_dtype: {params.out_dtype}, grad_output: {params.grad_output.GetDataType()}"
        ))


def __validate_filter_inputs(grad_output: Any, input: Any, filter_size: Any, out_dtype: Any,
                             strides: Any, paddings: Any, dilations: Any, groups: Any) -> None:
    __validate_type(grad_output, pypto_impl.Tensor, "grad_output")
    __validate_type(input, pypto_impl.Tensor, "input")
    __validate_type(filter_size, list, "filter_size")
    __validate_type(out_dtype, DataType, "out_dtype")
    __validate_type(strides, list, "strides")
    __validate_type(paddings, list, "paddings")
    __validate_type(dilations, list, "dilations")
    __validate_type(groups, int, "groups")

    if grad_output.Dim() != 4 or input.Dim() != 4:
        raise PyptoError(0xF00002, ValueError(
            "ConvBackwardFilter only supports 2D conv, expect grad_output.Dim() == input.Dim() == 4, "
            f"got grad_output_dim: {grad_output.Dim()}, input_dim: {input.Dim()}."
        ))

    if len(filter_size) != 4:
        raise PyptoError(0xF00002, ValueError(
            f"filter_size must have length 4 for 2D conv, got {len(filter_size)}."
        ))

    if len(strides) != 2 or len(dilations) != 2 or len(paddings) != 4:
        raise PyptoError(0xF00002, ValueError(
            "Invalid strides/dilations/paddings length. Expect strides=2, dilations=2, paddings=4, "
            f"got strides={len(strides)}, dilations={len(dilations)}, paddings={len(paddings)}."
        ))

    for value in (grad_output, input):
        if value.GetDataType() not in _VALID_DATA_TYPES:
            raise PyptoError(0xF00002, ValueError(
                "Input tensor data type must in [bf16, fp16, fp32],"
                f"but Input tensor got {value.GetDataType()}"
            ))

    __validate_filter_type_consistency(grad_output, input, out_dtype)


def __validate_filter_type_consistency(grad_output: Any, input: Any, out_dtype: Any) -> None:
    if grad_output.GetDataType() != input.GetDataType():
        raise PyptoError(0xF00002, ValueError(
            f"grad_output and input data types must be consistent, "
            f"but got grad_output: {grad_output.GetDataType()}, input: {input.GetDataType()}"
        ))

    if out_dtype != grad_output.GetDataType():
        raise PyptoError(0xF00002, ValueError(
            f"Output data type must be consistent with grad_output, "
            f"but got out_dtype: {out_dtype}, grad_output: {grad_output.GetDataType()}"
        ))
