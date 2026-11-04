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
"""Scalar casts reject storage-only dtypes before constructing cast IR."""

from pypto_pro import ir
from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
import pytest

STORAGE_DTYPES = (
    pl.DT_FP8E4M3FN, pl.DT_FP8E5M2, pl.DT_FP8E8M0,
    pl.DT_FP4E2M1, pl.DT_FP4E1M2, pl.DT_FP4,
    pl.DT_HF8, pl.DT_HF4, pl.DT_INT4, pl.DT_UINT4,
)


@pytest.mark.parametrize("dtype", STORAGE_DTYPES, ids=str)
@pytest.mark.parametrize("literal", [False, True], ids=["runtime", "literal"])
def test_astype_rejects_storage_only_destination(dtype, literal):
    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        if literal:
            _test_result = pl.astype(1.0, dtype)
        else:
            _test_result = pl.astype(value, dtype)

    with pytest.raises(NotSupported, match=r"pl\.astype\(\).*destination dtype.*storage-only"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("dtype", STORAGE_DTYPES, ids=str)
@pytest.mark.parametrize("identity", [False, True], ids=["to_fp32", "identity"])
def test_astype_rejects_storage_only_source(dtype, identity):
    destination = dtype if identity else pl.DT_FP32
    value = (ir.ConstFloat(1.0, dtype, ir.Span.unknown()) if dtype.is_float()
             else ir.ConstInt(1, dtype, ir.Span.unknown()))

    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1], pl.DT_FP32]):
        out[0] = pl.astype(value, destination)

    with pytest.raises(NotSupported, match=r"pl\.astype\(\).*source dtype.*storage-only"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize(
    "dtype",
    [pl.DT_BOOL, pl.DT_INT8, pl.DT_INT16, pl.DT_INT32, pl.DT_INT64,
     pl.DT_UINT8, pl.DT_UINT16, pl.DT_UINT32, pl.DT_UINT64, pl.DT_FP16, pl.DT_BF16, pl.DT_FP32],
    ids=str,
)
def test_astype_preserves_supported_destination_dtype(dtype):
    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        _test_result = pl.astype(value, dtype)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assignments = [stmt for stmt in program.get_function("kernel").body.stmts if isinstance(stmt, ir.AssignStmt)]
    assert assignments
    assert assignments[-1].value.type.dtype == dtype


@pytest.mark.parametrize("dtype", STORAGE_DTYPES, ids=str)
def test_const_rejects_storage_only_dtype(dtype):
    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        _result = pl.const(1, dtype)

    with pytest.raises(NotSupported, match=r"pl\.const\(\).*storage-only"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
