#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO"""

from typing import Any, Type

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..config import get_codegen_options
from ..enum import DataType
from ..error import PyptoError
from ..tensor import Tensor


@op_wrapper
def matmul(input, mat2, out_dtype, *, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=None) -> Tensor:
    """
    Performs matrix multiplication with support for batched operations, broadcasting, transposition, and extended
    features like bias addition and dequantization.

    Supports two primary computation modes:
    1.  Standard matrix multiplication of two matrices.
    2.  Batched matrix-matrix multiplication.

    `input` and `mat2` can be 2-D, 3-D, or 4-D tensors, each containing the same number of matrices.
    - If `input` is (n x k) and `mat2` is (k x m), output is (n x m).
    - If `input` is (b x n x k) and `mat2` is (b x k x m), output is (b x n x m).

    Note: Broadcasting is supported for 3-D or 4-D tensors.
    Example: If `input` is (1 x n x k) and `mat2` is (b x k x m), output will be (b x n x m).

    Parameters
    ----------
    input : Tensor
        The left operand matrix.
    mat2 : Tensor
        The right operand matrix.
    out_dtype : dtype
        The data type of the output tensor.

    Keyword Arguments
    ----------
    a_trans : bool, default=False
        If True, transpose the left matrix (`input`) before multiplication.
    b_trans : bool, default=False
        If True, transpose the right matrix (`mat2`) before multiplication.
    c_matrix_nz : bool, default=False
        If True, output the result matrix in NZ (non-zero) format.
    extend_params : dict, optional
        A dictionary specifying extended computation features:
        - 'bias_tensor': Tensor
            Adds a learnable bias to the output: C = A @ B + bias.
        - 'scale': float
            For dequantization: C = DEQF16(ReLU(A @ B)) * scale.
        - 'scale_tensor': Tensor
            For dequantization with a per-channel scale: C = DEQF16(ReLU(A @ B)) * scale_tensor.
        - 'relu_type': ReLuType
            Type of ReLU activation to apply before dequantization (e.g., ReLuType.RELU).
        - 'trans_mode': TransMode
            The rounding mode for converting float to TF32 (e.g., TransMode.CAST_RINT):
            CAST_NONE: Disables the conversion of float data types to TF32.
            CAST_RINT: float will be rounded to TF32 by rounding to the nearest tie to even.
            CAST_ROUND: float will be rounded to TF32 by rounding to the nearest tie away from zero.

    Returns
    -------
    Tensor
        A new tensor containing the matrix multiplication result.

    Raises
    ------
    RuntimeError
        If input dimensions are invalid (<2-D or >4-D), or if matrix dimensions are incompatible.

    Examples
    --------
    # Standard matrix multiplication
    a = pypto.tensor((16, 32), pypto.DT_BF16, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_BF16, "tensor_b")
    pypto.matmul(a, b, pypto.DT_BF16)

    # Batched matrix multiplication
    a = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
    b = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
    pypto.matmul(a, b, pypto.DT_FP16)

    # Batched multiplication with broadcasting
    a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
    b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
    pypto.matmul(a, b, pypto.DT_FP32)

    # With bias addition
    a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
    bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
    extend_params = {'bias_tensor': bias}
    pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

    # With dequantization (scale)
    a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
    extend_params = {'scale': 0.2}
    pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

    # With dequantization (scale & ReLU)
    a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
    extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
    pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

    # With dequantization (scale_tensor & ReLU)
    a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
    scale_tensor = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
    extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
    pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

    # TF32 matrix multiplication
    a = pypto.tensor((16, 32), pypto.DT_FP32, "tensor_a")
    b = pypto.tensor((32, 64), pypto.DT_FP32, "tensor_b")
    extend_params = {'trans_mode': pypto.TransMode.CAST_ROUND}
    pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
    """
    __validate_inputs(input, mat2, out_dtype, [a_trans, b_trans, c_matrix_nz, extend_params], INPUT_COMBOS)
    if input.Dim() == 2:
        if extend_params is not None:
            extend_params = pypto_impl.MatmulExtendParam(**__convert_matmul_extend_params(extend_params))
            return pypto_impl.Matmul(out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz, extend_params)
        else:
            return pypto_impl.Matmul(out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz)
    else:
        if extend_params is not None:
            extend_params = pypto_impl.MatmulExtendParam(**__convert_matmul_extend_params(extend_params))
            return pypto_impl.BatchMatmul(out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz, extend_params)
        else:
            return pypto_impl.BatchMatmul(out_dtype, input, mat2, a_trans, b_trans, c_matrix_nz)


@op_wrapper
def scaled_mm(
    mat_a,
    mat_b,
    out_dtype,
    scale_a,
    scale_b,
    *,
    a_trans=False,
    b_trans=False,
    scale_a_trans=False,
    scale_b_trans=False,
    c_matrix_nz=False,
    extend_params=None,
) -> Tensor:
    """
    Performs matrix multiplication with support for transposition, and extended features like bias addition.

    `mat_a` and `mat_b` can be 2-D, 3-D or 4-D. Their batch dimensions support broadcasting.

    `scale_a` and `scale_b` must have one more dimension than the matrix inputs. The batch dimensions of `scale_a`
    must equal those of `mat_a`, and the batch dimensions of `scale_b` must equal those of `mat_b`. Each matrix and
    its paired scale use the same batch broadcasting index.
    - 2-D matrices use 3-D scales: (m, kScale, 2) and (kScale, n, 2).
    - 3-D matrices use 4-D scales: (batchA, m, kScale, 2) and (batchB, kScale, n, 2).
    - 4-D matrices use 5-D scales: (batchA0, batchA1, m, kScale, 2) and
      (batchB0, batchB1, kScale, n, 2).

    Parameters
    ----------
    mat_a : Tensor
        The left operand matrix.
    mat_b : Tensor
        The right operand matrix.
    out_dtype : dtype
        The data type of the output tensor.
    scale_a : Tensor
        The left scale matrix for left operand matrix.
    scale_b : Tensor
        The right scale matrix for right operand matrix.

    Keyword Arguments
    ----------
    a_trans : bool, default=False
        If True, transpose the left matrix (`mat_a`) before multiplication.
    b_trans : bool, default=False
        If True, transpose the right matrix (`mat_b`) before multiplication.
    scale_a_trans : bool, default=False
        If True, transpose the left scale matrix (`scale_a`) before multiplication.
    scale_b_trans : bool, default=False
        If True, transpose the right scale matrix (`scale_b`) before multiplication.
    c_matrix_nz : bool, default=False
        If True, output the result matrix in NZ (non-zero) format.
    extend_params : dict, optional
        A dictionary specifying extended computation features:
        - 'bias_tensor': Tensor
            Adds a learnable bias to the output: C = A @ B + bias.

    Returns
    -------
    Tensor
        A new tensor containing the matrix multiplication result.

    Raises
    ------
    RuntimeError
        If matrix or scale dimensions are invalid, scale batch dimensions do not equal their paired matrix batch
        dimensions, or matrix dimensions are incompatible.

    Examples
    --------
    # Standard matrix multiplication
    a = pypto.tensor((16, 128), pypto.DT_FP8E4M3, "tensor_a")
    b = pypto.tensor((128, 64), pypto.DT_FP8E4M3, "tensor_b")
    a_scale = pypto.tensor((16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    b_scale = pypto.tensor((2, 64, 2), pypto.DT_FP8E8M0, "scale_b")
    pypto.scaled_mm(a, b, pypto.DT_BF16, a_scale, b_scale)

    # Standard matrix with bias addition
    a = pypto.tensor((16, 128), pypto.DT_FP8E4M3, "tensor_a")
    b = pypto.tensor((128, 64), pypto.DT_FP8E4M3, "tensor_b")
    a_scale = pypto.tensor((16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    b_scale = pypto.tensor((2, 64, 2), pypto.DT_FP8E8M0, "scale_b")
    bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
    extend_params = {'bias_tensor': bias}
    pypto.scaled_mm(a, b, pypto.DT_FP16, a_scale, b_scale, extend_params=extend_params)

    # 3-D broadcasting: A and scale_a share batch 1; B and scale_b share batch 4
    a = pypto.tensor((1, 16, 128), pypto.DT_FP8E4M3, "tensor_a")
    b = pypto.tensor((4, 128, 64), pypto.DT_FP8E4M3, "tensor_b")
    a_scale = pypto.tensor((1, 16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    b_scale = pypto.tensor((4, 2, 64, 2), pypto.DT_FP8E8M0, "scale_b")
    pypto.scaled_mm(a, b, pypto.DT_BF16, a_scale, b_scale)

    # 4-D cross broadcasting: each scale follows its paired matrix batch shape
    a = pypto.tensor((2, 1, 16, 128), pypto.DT_FP8E4M3, "tensor_a")
    b = pypto.tensor((1, 3, 128, 64), pypto.DT_FP8E4M3, "tensor_b")
    a_scale = pypto.tensor((2, 1, 16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    b_scale = pypto.tensor((1, 3, 2, 64, 2), pypto.DT_FP8E8M0, "scale_b")
    pypto.scaled_mm(a, b, pypto.DT_BF16, a_scale, b_scale)

    # Standard batch matrix multiplication
    a = pypto.tensor((3, 16, 128), pypto.DT_FP8E4M3, "tensor_a")
    b = pypto.tensor((3, 128, 64), pypto.DT_FP8E4M3, "tensor_b")
    a_scale = pypto.tensor((3, 16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    b_scale = pypto.tensor((3, 2, 64, 2), pypto.DT_FP8E8M0, "scale_b")
    pypto.scaled_mm(a, b, pypto.DT_BF16, a_scale, b_scale)

    # Standard batch matrix with bias addition
    a = pypto.tensor((3, 16, 128), pypto.DT_FP8E4M3, "tensor_a")
    b = pypto.tensor((3, 128, 64), pypto.DT_FP8E4M3, "tensor_b")
    a_scale = pypto.tensor((3, 16, 2, 2), pypto.DT_FP8E8M0, "scale_a")
    b_scale = pypto.tensor((3, 2, 64, 2), pypto.DT_FP8E8M0, "scale_b")
    bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
    extend_params = {'bias_tensor': bias}
    pypto.scaled_mm(a, b, pypto.DT_FP16, a_scale, b_scale, extend_params=extend_params)
    """
    __validate_inputs(mat_a, mat_b, out_dtype, [a_trans, b_trans, c_matrix_nz, extend_params], MX_INPUT_COMBOS)
    __validate_scaled_inputs(mat_a, mat_b, scale_a, scale_b, extend_params)
    __validate_scaled_shape(mat_a, mat_b, scale_a, scale_b, [a_trans, b_trans, scale_a_trans, scale_b_trans])
    __validate_out_dtype(mat_a.GetDataType(), out_dtype, extend_params, MX_BASIC_OUT_DTYPES)

    if mat_a.Dim() == 2:
        if extend_params is not None:
            extend_params = pypto_impl.MatmulExtendParam(**__convert_matmul_extend_params(extend_params))
            return pypto_impl.MatmulMX(
                out_dtype,
                mat_a,
                scale_a,
                mat_b,
                scale_b,
                a_trans,
                scale_a_trans,
                b_trans,
                scale_b_trans,
                c_matrix_nz,
                extend_params,
            )
        else:
            return pypto_impl.MatmulMX(
                out_dtype, mat_a, scale_a, mat_b, scale_b, a_trans, scale_a_trans, b_trans, scale_b_trans, c_matrix_nz
            )
    else:
        if extend_params is not None:
            extend_params = pypto_impl.MatmulExtendParam(**__convert_matmul_extend_params(extend_params))
            return pypto_impl.BatchMatmulMX(
                out_dtype,
                mat_a,
                scale_a,
                mat_b,
                scale_b,
                a_trans,
                scale_a_trans,
                b_trans,
                scale_b_trans,
                c_matrix_nz,
                extend_params,
            )
        else:
            return pypto_impl.BatchMatmulMX(
                out_dtype, mat_a, scale_a, mat_b, scale_b, a_trans, scale_a_trans, b_trans, scale_b_trans, c_matrix_nz
            )


A2A3_ARCHS = ("DAV_1001", "DAV_2201")

INPUT_COMBOS = {
    (pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_FP16),
    (pypto_impl.DataType.DT_BF16, pypto_impl.DataType.DT_BF16),
    (pypto_impl.DataType.DT_FP32, pypto_impl.DataType.DT_FP32),
    (pypto_impl.DataType.DT_INT8, pypto_impl.DataType.DT_INT8),
    (pypto_impl.DataType.DT_FP8E5M2, pypto_impl.DataType.DT_FP8E5M2),
    (pypto_impl.DataType.DT_FP8E5M2, pypto_impl.DataType.DT_FP8E4M3),
    (pypto_impl.DataType.DT_FP8E4M3, pypto_impl.DataType.DT_FP8E5M2),
    (pypto_impl.DataType.DT_FP8E4M3, pypto_impl.DataType.DT_FP8E4M3),
    (pypto_impl.DataType.DT_HF8, pypto_impl.DataType.DT_HF8),
}

MX_INPUT_COMBOS = {
    (pypto_impl.DataType.DT_FP8E5M2, pypto_impl.DataType.DT_FP8E5M2),
    (pypto_impl.DataType.DT_FP8E5M2, pypto_impl.DataType.DT_FP8E4M3),
    (pypto_impl.DataType.DT_FP8E4M3, pypto_impl.DataType.DT_FP8E5M2),
    (pypto_impl.DataType.DT_FP8E4M3, pypto_impl.DataType.DT_FP8E4M3),
    (pypto_impl.DataType.DT_FP4_E2M1, pypto_impl.DataType.DT_FP4_E2M1),
}

_FLOAT_OUT_DTYPES = (
    pypto_impl.DataType.DT_FP16,
    pypto_impl.DataType.DT_BF16,
    pypto_impl.DataType.DT_FP32,
)

BASIC_OUT_DTYPES = {
    pypto_impl.DataType.DT_FP16: (pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_FP32),
    pypto_impl.DataType.DT_BF16: (pypto_impl.DataType.DT_BF16, pypto_impl.DataType.DT_FP32),
    pypto_impl.DataType.DT_FP32: (pypto_impl.DataType.DT_FP32,),
    pypto_impl.DataType.DT_INT8: (pypto_impl.DataType.DT_INT32,),
    pypto_impl.DataType.DT_FP8E5M2: _FLOAT_OUT_DTYPES,
    pypto_impl.DataType.DT_FP8E4M3: _FLOAT_OUT_DTYPES,
    pypto_impl.DataType.DT_HF8: _FLOAT_OUT_DTYPES,
}

MX_BASIC_OUT_DTYPES = {
    pypto_impl.DataType.DT_FP8E5M2: _FLOAT_OUT_DTYPES,
    pypto_impl.DataType.DT_FP8E4M3: _FLOAT_OUT_DTYPES,
    pypto_impl.DataType.DT_FP4_E2M1: _FLOAT_OUT_DTYPES,
}

NZ_UNSUPPORTED_INPUT_DTYPES = {
    pypto_impl.DataType.DT_FP32,
    pypto_impl.DataType.DT_FP8E5M2,
    pypto_impl.DataType.DT_HF8,
}

QUANT_OUT_DTYPES = {
    pypto_impl.DataType.DT_INT8: (pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_INT8),
    pypto_impl.DataType.DT_BF16: (pypto_impl.DataType.DT_INT8,),
    pypto_impl.DataType.DT_FP16: (pypto_impl.DataType.DT_INT8,),
    pypto_impl.DataType.DT_FP32: (pypto_impl.DataType.DT_INT8,),
    pypto_impl.DataType.DT_FP8E5M2: (pypto_impl.DataType.DT_INT8,),
    pypto_impl.DataType.DT_FP8E4M3: (pypto_impl.DataType.DT_INT8,),
    pypto_impl.DataType.DT_HF8: (pypto_impl.DataType.DT_INT8,),
    pypto_impl.DataType.DT_FP4_E2M1: (pypto_impl.DataType.DT_INT8,),
}


def __is_a2a3_arch() -> bool:
    return pypto_impl.GetNPUArch() in A2A3_ARCHS


def __is_kirin9050_arch() -> bool:
    codegen_options = get_codegen_options()
    soc_version = ""
    if "soc_version" in codegen_options:
        soc_version = codegen_options["soc_version"]
    return soc_version == "Kirin9050"


def __validate_type(value: Any, expect_type: Type, arg_name: str = "input") -> None:
    if value is None:
        return
    if not isinstance(value, expect_type):
        raise PyptoError(
            0xF00001,
            TypeError(f"Argument '{arg_name}' must be of type {expect_type.__name__}, but got {type(value).__name__}."),
        )


def __get_valid_shape(tensor):
    return tensor.GetValidShape()


def __validate_shape(input_tensor1: Tensor, input_tensor2: Tensor, a_trans: bool, b_trans: bool) -> None:
    input_dim = input_tensor1.Dim()
    mat2_dim = input_tensor2.Dim()
    if input_dim != mat2_dim or input_dim not in {2, 3, 4}:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Tensor dimension mismatch. Expect input_dim == mat2_dim and both in [2, 3, 4], "
                f"got input_dim: {input_dim}, mat2_dim: {mat2_dim}."
            ),
        )

    input_valid_shape = __get_valid_shape(input_tensor1)
    mat2_valid_shape = __get_valid_shape(input_tensor2)
    m_dim, ka_dim = (
        (input_valid_shape[-2], input_valid_shape[-1])
        if not a_trans
        else (input_valid_shape[-1], input_valid_shape[-2])
    )
    kb_dim, n_dim = (
        (mat2_valid_shape[-2], mat2_valid_shape[-1]) if not b_trans else (mat2_valid_shape[-1], mat2_valid_shape[-2])
    )
    if ka_dim.is_concrete() and kb_dim.is_concrete() and ka_dim != kb_dim:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "K-dimension valid shape mismatch. "
                f"Got input valid shape: {input_valid_shape}, mat2 valid shape: {mat2_valid_shape}, "
                f"a_trans: {a_trans}, b_trans: {b_trans}."
            ),
        )


def __validate_bias(input_tensor, extend_params) -> None:
    if (
        extend_params is not None
        and "bias_tensor" in extend_params
        and isinstance(extend_params["bias_tensor"], Tensor)
    ):
        input_dim = input_tensor.Dim()
        bias_dim = extend_params["bias_tensor"].Dim()
        if input_dim == 3 and bias_dim not in {2, 3}:
            raise PyptoError(
                0xF00003,
                RuntimeError(
                    "Bias tensor dimension mismatch for 3D. Expect bias_dim in [2, 3], "
                    f"got input_dim: {input_dim}, bias_dim: {bias_dim}."
                ),
            )
        elif input_dim == 4 and bias_dim != 2:
            raise PyptoError(
                0xF00003,
                RuntimeError(
                    "Bias tensor dimension mismatch for 4D. Expect bias_dim equals to 2, "
                    f"got input_dim: {input_dim}, bias_dim: {bias_dim}."
                ),
            )


def __validate_nz_input(input_tensor1, input_tensor2) -> None:
    for tensor in (input_tensor1, input_tensor2):
        unsupported_dtype = tensor.GetDataType() in NZ_UNSUPPORTED_INPUT_DTYPES
        if unsupported_dtype and tensor.Format() == pypto_impl.TileOpFormat.TILEOP_NZ:
            raise PyptoError(
                0xF00002,
                ValueError(
                    f"Input tensor with {tensor.GetDataType().name} must use ND format, "
                    "NZ format is not supported currently."
                ),
            )


def __validate_inputs(input_tensor1, input_tensor2, out_dtype, optional_param, input_combos) -> None:
    a_trans, b_trans, is_out_nz, extend_params = optional_param
    __validate_type(out_dtype, DataType, "out_dtype")
    __validate_type(a_trans, bool, "a_trans")
    __validate_type(b_trans, bool, "b_trans")
    __validate_type(is_out_nz, bool, "is_out_nz")
    __validate_type(extend_params, dict, "extend_params")
    __validate_shape(input_tensor1, input_tensor2, a_trans, b_trans)
    __validate_trans_mode(input_tensor1, input_tensor2, extend_params)
    __validate_bias(input_tensor1, extend_params)

    input1_dtype = input_tensor1.GetDataType()
    input2_dtype = input_tensor2.GetDataType()
    if is_out_nz:
        raise PyptoError(0xF00002, ValueError("Output tensor do not support NZ currently."))
    if (input1_dtype == pypto_impl.DataType.DT_HF8 or input2_dtype == pypto_impl.DataType.DT_HF8) and __is_a2a3_arch():
        raise PyptoError(
            0xF00002,
            ValueError("Input tensor with DT_HF8 is not supported on A2/A3 platforms, only supported on Ascend 950."),
        )
    __validate_nz_input(input_tensor1, input_tensor2)
    if (input1_dtype, input2_dtype) not in input_combos:
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Unsupported input dtype combination: (a={input1_dtype.name}, b={input2_dtype.name}). "
                "See the input dtype columns of the API docs."
            ),
        )
    __validate_out_dtype(input1_dtype, out_dtype, extend_params, BASIC_OUT_DTYPES)


def __has_quant_scale(extend_params) -> bool:
    if extend_params is None:
        return False
    return extend_params.get('scale_tensor') is not None or extend_params.get('scale', 0.0) != 0.0


def __is_out_dtype_allowed(input_dtype, out_dtype, allowed_out) -> bool:
    # kirin9050 宏指令：f16 输入支持 s32(NoQuant)/fp16/s8 输出，其中 s32 输出无需 quant scale
    kirin9050_fp16_to_s32 = (
        input_dtype == pypto_impl.DataType.DT_FP16
        and out_dtype == pypto_impl.DataType.DT_INT32
        and __is_kirin9050_arch()
    )
    return out_dtype in allowed_out or kirin9050_fp16_to_s32


def __validate_out_dtype(input_dtype, out_dtype, extend_params, basic_out_map) -> None:
    if input_dtype not in basic_out_map:
        return
    allowed_out = QUANT_OUT_DTYPES[input_dtype] if __has_quant_scale(extend_params) else basic_out_map[input_dtype]
    if not __is_out_dtype_allowed(input_dtype, out_dtype, allowed_out):
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Unsupported out_dtype {out_dtype} for input dtype {input_dtype}. "
                f"Allowed out_dtype: {allowed_out} (see dtype support tables in the API docs)."
            ),
        )


def __validate_bias_dimension(input_dim, bias_tensor):
    if bias_tensor is None:
        return
    bias_dim = bias_tensor.Dim()
    if input_dim in (2, 4):
        if bias_dim != 2:
            raise PyptoError(
                0xF00003,
                RuntimeError(f"{input_dim}D BatchMatmulMX only supports 2D bias tensor, but got {bias_dim}D bias."),
            )
    elif input_dim == 3:
        if bias_dim not in (2, 3):
            raise PyptoError(
                0xF00003,
                RuntimeError(f"3D BatchMatmulMX only supports 2D or 3D bias tensor, but got {bias_dim}D bias."),
            )


def __validate_scale_dimension(input_dim, scale_tensor):
    if scale_tensor is None:
        return
    scale_dim = scale_tensor.Dim()
    if scale_dim != 2:
        raise PyptoError(
            0xF00003,
            RuntimeError(f"{input_dim}D scaled_mm only supports 2D scale tensor, but got {scale_dim}D scale tensor."),
        )


def __validate_scaled_inputs(input_tensor1, input_tensor2, input_scale1, input_scale2, extend_params) -> None:
    input_dim = input_tensor1.Dim()
    other_dim = input_tensor2.Dim()
    scale_a_dim = input_scale1.Dim()
    scale_b_dim = input_scale2.Dim()
    supported_dims = (2, 3, 4)

    if (
        input_tensor1.GetDataType() == pypto_impl.DataType.DT_FP4_E1M2
        or input_tensor1.GetDataType() == pypto_impl.DataType.DT_FP4_E1M2X2
        or input_tensor1.GetDataType() == pypto_impl.DataType.DT_FP4_E2M1X2
    ):
        raise PyptoError(0xF00003, RuntimeError("scaled_mm fp4 input only supports DT_FP4_E2M1."))

    if (
        input_scale1.Format() == pypto_impl.TileOpFormat.TILEOP_NZ
        or input_scale2.Format() == pypto_impl.TileOpFormat.TILEOP_NZ
    ):
        raise PyptoError(0xF00002, ValueError("Scale tensor must use ND format, NZ format is not supported currently."))

    if input_dim not in supported_dims:
        raise PyptoError(
            0xF00003,
            RuntimeError(f"Unsupported tensor dimension for MX Matmul: {input_dim}, only support 2D, 3D or 4D tensor."),
        )
    if input_dim != other_dim:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Tensor dimension mismatch. Expect input_dim == other_dim, "
                f"got input_dim: {input_dim}, other_dim: {other_dim}."
            ),
        )
    expected_scale_dim = input_dim + 1
    if scale_a_dim != scale_b_dim or scale_a_dim != expected_scale_dim:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Tensor dimension mismatch. Expect scale_a_dim == scale_b_dim and both equal to input_dim + 1 "
                f"({expected_scale_dim}), "
                f"got scale_a_dim: {scale_a_dim}, scale_b_dim: {scale_b_dim}."
            ),
        )

    if extend_params is not None:
        __validate_bias_dimension(input_dim, extend_params.get('bias_tensor'))
        __validate_scale_dimension(input_dim, extend_params.get('scale_tensor'))


def __validate_scaled_shape(input_tensor1, input_tensor2, input_scale1, input_scale2, optional_param) -> None:
    a_trans, b_trans, a_scale_trans, b_scale_trans = optional_param
    align_64 = 64
    shape_dim_2 = 2
    input_valid_shape = __get_valid_shape(input_tensor1)
    other_valid_shape = __get_valid_shape(input_tensor2)
    m_dim, ka_dim = (
        (input_valid_shape[-2], input_valid_shape[-1])
        if not a_trans
        else (input_valid_shape[-1], input_valid_shape[-2])
    )
    n_dim = (other_valid_shape[-1]) if not b_trans else (other_valid_shape[-2])
    scale1_valid_shape = __get_valid_shape(input_scale1)
    scale2_valid_shape = __get_valid_shape(input_scale2)
    input_shape = input_tensor1.GetShape()
    other_shape = input_tensor2.GetShape()
    scale1_shape = input_scale1.GetShape()
    scale2_shape = input_scale2.GetShape()
    batch_dim = input_tensor1.Dim() - 2
    for batch_idx in range(batch_dim):
        if scale1_shape[batch_idx] != input_shape[batch_idx]:
            raise PyptoError(
                0xF00003,
                RuntimeError(
                    f"Scale batch dimension mismatch at axis {batch_idx}. Expected scale_a batch dimension to "
                    f"equal paired mat_a batch dimension {input_shape[batch_idx]}, got {scale1_shape[batch_idx]}."
                ),
            )
        if scale2_shape[batch_idx] != other_shape[batch_idx]:
            raise PyptoError(
                0xF00003,
                RuntimeError(
                    f"Scale batch dimension mismatch at axis {batch_idx}. Expected scale_b batch dimension to "
                    f"equal paired mat_b batch dimension {other_shape[batch_idx]}, got {scale2_shape[batch_idx]}."
                ),
            )
    m_scale_dim, k_a_scale0_dim, k_a_scale1_dim = (
        (scale1_valid_shape[-3], scale1_valid_shape[-2], scale1_valid_shape[-1])
        if not a_scale_trans
        else (scale1_valid_shape[-2], scale1_valid_shape[-3], scale1_valid_shape[-1])
    )
    k_b_scale0_dim, n_scale_dim, k_b_scale1_dim = (
        (scale2_valid_shape[-3], scale2_valid_shape[-2], scale2_valid_shape[-1])
        if not b_scale_trans
        else (scale2_valid_shape[-2], scale2_valid_shape[-3], scale2_valid_shape[-1])
    )

    __validate_scale_k0_dimensions(k_a_scale0_dim, k_b_scale0_dim)
    __validate_scale_k1_dimensions(k_a_scale1_dim, k_b_scale1_dim, shape_dim_2)
    __validate_scale_m_dimensions(m_dim, m_scale_dim)
    __validate_scale_n_dimensions(n_dim, n_scale_dim)
    __validate_scale_k_alignment(ka_dim, k_a_scale0_dim, align_64)


def __validate_scale_k0_dimensions(k_a_scale0_dim, k_b_scale0_dim):
    if k_a_scale0_dim.is_concrete() and k_b_scale0_dim.is_concrete() and k_a_scale0_dim != k_b_scale0_dim:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Scale Matrix Kscale0 dimension mismatch. Expect scale_ka_size == scale_kb_size, "
                f"got scale_ka_size: {k_a_scale0_dim}, scale_kb_size: {k_b_scale0_dim}."
            ),
        )


def __validate_scale_k1_dimensions(k_a_scale1_dim, k_b_scale1_dim, shape_dim_2):
    is_value_concrete = k_a_scale1_dim.is_concrete() and k_b_scale1_dim.is_concrete()
    if is_value_concrete and (
        k_a_scale1_dim != k_b_scale1_dim or k_a_scale1_dim != shape_dim_2 or k_b_scale1_dim != shape_dim_2
    ):
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Scale Matrix Kscale1 dimension mismatch. Expect the last dimensions of scale_a and scale_b "
                f"to both equal 2, got scale_a: {k_a_scale1_dim}, scale_b: {k_b_scale1_dim}."
            ),
        )


def __validate_scale_m_dimensions(m_dim, m_scale_dim):
    if m_dim.is_concrete() and m_scale_dim.is_concrete() and m_dim != m_scale_dim:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Matrix M dimension mismatch. Expect m_scale_size == m_size, "
                f"got m_scale_size: {m_scale_dim}, m_size: {m_dim}."
            ),
        )


def __validate_scale_n_dimensions(n_dim, n_scale_dim):
    if n_dim.is_concrete() and n_scale_dim.is_concrete() and n_dim != n_scale_dim:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Matrix N dimension mismatch. Expect n_scale_size == n_size, "
                f"got n_scale_size: {n_scale_dim}, n_size: {n_dim}."
            ),
        )


def __validate_scale_k_alignment(ka_dim, k_a_scale0_dim, align_64):
    if k_a_scale0_dim.is_concrete() and ka_dim.is_concrete() and k_a_scale0_dim != (ka_dim + align_64 - 1) // align_64:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "Input tensor K-dimension does not match expect scale K-dimension size. Scale K != ceil(input K, 64)."
                f"input tensor K: {ka_dim}, scale K: {k_a_scale0_dim}"
            ),
        )


def __validate_trans_mode(mat_a, mat_b, extend_params):
    if extend_params is not None:
        if (
            extend_params.get('trans_mode', pypto_impl.TransMode.CAST_NONE) != pypto_impl.TransMode.CAST_NONE
            and mat_a.GetDataType() != pypto_impl.DataType.DT_FP32
            and mat_b.GetDataType() != pypto_impl.DataType.DT_FP32
        ):
            raise PyptoError(
                0xF00003, RuntimeError("The trans_mode parameter is only supported when input data type is DT_FP32")
            )
        if (
            extend_params.get('trans_mode', pypto_impl.TransMode.CAST_NONE) != pypto_impl.TransMode.CAST_NONE
            and __is_a2a3_arch()
        ):
            raise PyptoError(
                0xF00002,
                ValueError(
                    "The trans_mode parameter is not supported on A2/A3 platforms, only supported on Ascend 950."
                ),
            )


def __convert_matmul_extend_params(extend_params) -> dict:
    extend_params.setdefault('bias_tensor', pypto_impl.Tensor())
    extend_params.setdefault('scale_tensor', pypto_impl.Tensor())
    extend_params.setdefault('relu_type', pypto_impl.ReLuType.NO_RELU)
    extend_params.setdefault('scale', 0.0)
    extend_params.setdefault('trans_mode', pypto_impl.TransMode.CAST_NONE)
    return extend_params
