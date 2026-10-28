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

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..error import PyptoError
from ..tensor import Tensor


@op_wrapper
def gather_in_l1(
    src: Tensor, indices: Tensor, block_table: Tensor, block_size: int, size: int, is_b_matrix: bool, is_trans: bool
) -> Tensor:
    """gather_in_l1."""

    __validate_gather_in_l1_inputs(src, indices, block_table)
    return pypto_impl.gather_in_l1(src, indices, block_table, block_size, size, is_b_matrix, is_trans)


@op_wrapper
def gather_in_ub(param: Tensor, indices: Tensor, block_table: Tensor, block_size: int, axis: int) -> Tensor:
    """gather_in_ub."""
    """
    Custom Operator for Sparse Attention Mechanism:
    Extracts selected key-value (KV) vectors from the PagedAttention KV cache based on token indices.

    This operator assumes that the KV cache is stored in GM (Global Memory),
    and the extracted results are written to UB (Unified Buffer).

    Parameters:
    -----------
    param : Tensor
        Input tensor representing the KV cache in GM.
        Only 2-D tensors are supported, with shape [token_num, hidden_size].
    indices : Tensor
        Input tensor containing the indices of selected tokens (e.g., TopK results).
        Only 2-D tensors are supported, with shape [1, k].
    blockTable : Tensor
        Input tensor representing the page table in PagedAttention.
        Only 2-D tensors are supported, with shape [1, block_table_size].
    blockSize : int
        Input scalar indicating the number of tokens per block in PagedAttention.
    axis : int
        Input scalar specifying the dimension along which the operation is applied.
        Only -2 (second-to-last dimension) is currently supported.
    out: Tensor
        Contains the KV vectors (either key or value, depending on the input param) corresponding to
        the k selected tokens specified by indices with shape [k, hidden_size].

    Examples
    --------
    param = pypto.tensor([6,4], pypto.DT_FP16, "src")
    offsets = pypto.tensor([1,3], pypto.DT_INT32, "offsets")
    pageTable = pypto.tensor([1,3], pypto.DT_INT32, "pageTable")
    blockSize = 2
    out = pypto.experimental.gather_in_ub(param, offsets, pageTable, blockSize, -2)

    Input param:
    [
        [  0,  1,  2,  3],  # 0
        [ 10, 11, 12, 13],  # 1
        [ 20, 21, 22, 23],  # 2
        [ 30, 31, 32, 33],  # 3
        [ 40, 41, 42, 43],  # 4
        [ 50, 51, 52, 53],  # 5
    ]
    Input indices:  [0, 4, 3]
    Input blockTable:  [0, 2, 1]

    Output out:
    [
        [  0,  1,  2,  3],
        [ 20, 21, 22, 23],
        [ 50, 51, 52, 53],
    ]
    """
    return pypto_impl.gather_in_ub(param, indices, block_table, block_size, axis)


GATHER_IN_L1_SRC_DTYPES = (
    pypto_impl.DataType.DT_FP32,
    pypto_impl.DataType.DT_FP16,
    pypto_impl.DataType.DT_BF16,
    pypto_impl.DataType.DT_INT8,
)
GATHER_IN_L1_INDICES_DTYPES = (
    pypto_impl.DataType.DT_INT32,
    pypto_impl.DataType.DT_INT64,
)


def __validate_gather_in_l1_inputs(src: Tensor, indices: Tensor, block_table: Tensor) -> None:
    if block_table.Dim() != 2:
        raise PyptoError(
            0xF00003,
            RuntimeError(
                "block_table dimension mismatch. Expect 2-dim block_table for gather_in_l1, "
                f"got {block_table.Dim()}-dim."
            ),
        )
    src_dtype = src.GetDataType()
    if src_dtype not in GATHER_IN_L1_SRC_DTYPES:
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Unsupported src dtype {src_dtype.name} for gather_in_l1. "
                "Supported dtypes: DT_FP32, DT_FP16, DT_BF16, DT_INT8 (see the API docs)."
            ),
        )
    indices_dtype = indices.GetDataType()
    if indices_dtype not in GATHER_IN_L1_INDICES_DTYPES:
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Unsupported indices dtype {indices_dtype.name} for gather_in_l1. "
                "Supported dtypes: DT_INT32, DT_INT64 (see the API docs)."
            ),
        )
    block_table_dtype = block_table.GetDataType()
    if block_table_dtype != pypto_impl.DataType.DT_INT32:
        raise PyptoError(
            0xF00002,
            ValueError(
                f"Unsupported block_table dtype {block_table_dtype.name} for gather_in_l1. "
                "Supported dtypes: DT_INT32 (see the API docs)."
            ),
        )
