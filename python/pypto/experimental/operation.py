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
""" """

from typing import Dict, List, Optional, Tuple, Union

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..config import get_current_scope, set_options
from ..error import PyptoError
from ..symbolic_scalar import SymbolicScalar
from ..tensor import Tensor


def assume_divisible(expr: "SymbolicScalar | int", divisor: int) -> "SymbolicScalar":
    """
    Assume that ``expr`` is divisible by ``divisor`` at runtime.

    Returns a ``SymbolicScalar`` representing ``expr``. ``SymbolicScalar`` inputs
    are returned unchanged; ``int`` inputs are converted. The normalized assumption
    is registered on the Program, enabling the compiler to simplify
    dynamic valid-shape expressions (e.g. ``min(max(vm - off, 0), T)`` → ``T``)
    when the user knows the runtime value is always a multiple of ``divisor``.

    By default, this assumption is unchecked at runtime. With
    ``runtime_debug_mode=4``, codegen lowers every explicitly registered,
    normalized assumption expression to a modulo assertion at the control-flow
    entry.

    If ``expr`` is a compile-time constant:
      - ``expr % divisor != 0`` → raises ``ValueError`` (assumption is provably false)
      - ``expr % divisor == 0`` → no-op (trivially true)

    Parameters
    ----------
    expr : SymbolicScalar | int
        The runtime scalar expression assumed to be divisible.
    divisor : int
        The divisor (must be positive).

    Returns
    -------
    SymbolicScalar
        A symbolic scalar representing ``expr``. Symbolic inputs are returned
        unchanged; integer inputs are converted.
    """
    if isinstance(expr, int):
        expr = SymbolicScalar(expr)
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if expr.is_concrete() and int(expr) % divisor != 0:
        raise ValueError(f"assume_divisible: constant {int(expr)} is not divisible by {divisor}")
    from ..functions import get_current_function

    func = get_current_function()
    if func is None:
        return expr
    func.base.RegisterDivisibleAssumption(expr, divisor)
    return expr


@op_wrapper
def transposed_batchmatmul(tensor_a: Tensor, tensor_b: Tensor, out_dtype) -> Tensor:
    """
    Performs a transposed batch matrix multiplication.

    This operator computes:
        1. Transpose tensor_a from shape (M, B, K) to (B, M, K).
        2. Perform a batch matrix multiplication between the transposed tensor_a
           (B, M, K) and tensor_b (B, K, N), yielding an intermediate result of
           shape (B, M, N).
        3. Transpose the intermediate result back to shape (M, B, N).

    Parameters
    ----------
    tensor_a : Tensor
        The left-hand input tensor with shape (M, B, K).
        Supported data types: DT_FP16, DT_BF16.

    tensor_b : Tensor
        The right-hand input tensor with shape (B, K, N).
        Supported data types: DT_FP16, DT_BF16.

    out_dtype : dtype
        The data type for the output tensor.

    Returns
    -------
    Tensor
        The output tensor of shape (M, B, N).

    Examples
    --------
    a = pypto.tensor((16, 2, 32), pypto.DT_FP16, "tensor_a")
    b = pypto.tensor((2, 32, 64), pypto.DT_FP16, "tensor_b")
    c = pypto.experimental.transposed_batchmatmul(a, b, pypto.DT_FP16)
    """
    __validate_transposed_batchmatmul_inputs(tensor_a, tensor_b, out_dtype)
    return pypto_impl.TransposedBatchMatmul(out_dtype, tensor_a, tensor_b)


def set_operation_options(*, combine_axis: Optional[bool] = None):
    """
    Set operation options.

    Parameters
    ---------
    combine_axis : bool
        Codegen forced axis fusion optimization.
    """

    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(operation_options=options_dict)


def get_operation_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get operation options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All operation options
    """

    scope = get_current_scope()
    return scope.get_operation_options()


@op_wrapper
def online_softmax(scores: Tensor, scale: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns axis-0 online softmax local statistics."""
    return pypto_impl.OnlineSoftmax(scores, scale)


@op_wrapper
def online_softmax_update(
    previous_max: Tensor,
    previous_sum: Tensor,
    previous_output: Tensor,
    current_max: Tensor,
    current_sum: Tensor,
    current_output: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Updates online softmax state for a new block."""
    return pypto_impl.OnlineSoftmaxUpdate(
        previous_max, previous_sum, previous_output, current_max, current_sum, current_output
    )


@op_wrapper
def nop(in_tensors: List[Tensor]) -> Tensor:
    return pypto_impl.Nop(in_tensors)


TRANSPOSED_BMM_INPUT_COMBOS = {
    (pypto_impl.DataType.DT_FP16, pypto_impl.DataType.DT_FP16),
    (pypto_impl.DataType.DT_BF16, pypto_impl.DataType.DT_BF16),
}
TRANSPOSED_BMM_OUT_DTYPES = (
    pypto_impl.DataType.DT_FP16,
    pypto_impl.DataType.DT_BF16,
)


def __validate_transposed_batchmatmul_inputs(tensor_a: Tensor, tensor_b: Tensor, out_dtype) -> None:
    for tensor, name in ((tensor_a, "tensor_a"), (tensor_b, "tensor_b")):
        if tensor.IsEmpty():
            raise PyptoError(
                0xF00003,
                RuntimeError(f"Empty tensor is not supported for transposed_batchmatmul, {name} is empty."),
            )
        shape = tensor.GetShape()
        if any(dim <= 0 for dim in shape):
            raise PyptoError(
                0xF00003,
                RuntimeError(f"Empty tensor is not supported for transposed_batchmatmul, {name} has shape {shape}."),
            )
    input_combination = (tensor_a.GetDataType(), tensor_b.GetDataType())
    if input_combination not in TRANSPOSED_BMM_INPUT_COMBOS:
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Unsupported input dtype combination: (a={tensor_a.GetDataType().name}, "
                f"b={tensor_b.GetDataType().name}). Supported combinations: DT_FP16 x DT_FP16, "
                "DT_BF16 x DT_BF16 (see the API docs)."
            ),
        )
    if out_dtype not in TRANSPOSED_BMM_OUT_DTYPES:
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Unsupported out_dtype {out_dtype.name} for transposed_batchmatmul. "
                "Allowed out_dtype: DT_FP16, DT_BF16 (see the API docs)."
            ),
        )
