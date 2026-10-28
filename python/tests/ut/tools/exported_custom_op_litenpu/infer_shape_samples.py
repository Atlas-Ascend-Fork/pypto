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
# Sample infer_shape / infer_dtype functions for codegen tests (must live in a real module for inspect.getsource).

import torch


def infer_shape_one_2d(x0_shape: torch.Size) -> torch.Size:
    return x0_shape


def infer_shape_two_by_two(
    x0_shape: torch.Size,
    x1_shape: torch.Size,
) -> torch.Size:
    return x0_shape


def infer_shape_4d_broadcast(
    a_shape: torch.Size,
    b_shape: torch.Size,
) -> torch.Size:
    return a_shape


def infer_shape_sum_last(
    x0_shape: torch.Size,
    x1_shape: torch.Size,
) -> torch.Size:
    return torch.Size((
        x0_shape[0],
        x0_shape[1],
        x0_shape[2] + x1_shape[2],
    ))


def infer_shape_nd_identity(x_shape: torch.Size) -> torch.Size:
    """Dynamic rank: output shape equals input shape."""
    return x_shape


def infer_shape_three_4d(
    a_shape: torch.Size,
    b_shape: torch.Size,
    c_shape: torch.Size,
) -> torch.Size:
    return a_shape


def infer_shape_two_outputs(
    a_shape: torch.Size,
    b_shape: torch.Size,
) -> tuple[torch.Size, torch.Size]:
    return (a_shape, b_shape)


def infer_shape_width_pick(x_shape: torch.Size, width: str) -> torch.Size:
    """A String shape-affecting attr: the attr value selects the output width."""
    out = list(x_shape)
    out[-1] = out[-1] if width == "FULL" else out[-1] // 2
    return torch.Size(out)


def infer_shape_scaled_last(x_shape: torch.Size, scale: float) -> torch.Size:
    """A Float shape-affecting attr: the attr value scales the last dim."""
    out = list(x_shape)
    out[-1] = int(out[-1] * scale)
    return torch.Size(out)


def infer_dtype_one(a_dtype: torch.dtype) -> torch.dtype:
    return a_dtype


def infer_dtype_two(a_dtype: torch.dtype, b_dtype: torch.dtype) -> torch.dtype:
    return a_dtype


def infer_dtype_three(
    a_dtype: torch.dtype, b_dtype: torch.dtype, c_dtype: torch.dtype
) -> torch.dtype:
    return a_dtype


def infer_dtype_two_outputs(
    a_dtype: torch.dtype, b_dtype: torch.dtype
) -> tuple[torch.dtype, torch.dtype]:
    return (a_dtype, b_dtype)
