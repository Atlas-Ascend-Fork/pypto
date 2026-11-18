# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Enum usage on aicpu kernels follows the framework's compile-time enum contract.

Enums are first-class as *compile-time constants* only: a literal member is a
legal kwarg value, a literal comparison folds into a compile-time branch, and
an enum member never appears as a runtime op operand. Runtime construction
(``Mode(runtime_scalar)``) and runtime operand positions stay unsupported -- a
port replaces the C++ ``static_cast<Enum>(attr)`` pattern with plain integer
comparisons.
"""

import enum

from pypto_pro import ir
from pypto_pro._errors import InvalidType, NotSupported
import pypto_pro.language as pl
import pytest


class MaskMode(enum.IntEnum):
    NO_MASK = 0
    CAUSAL = 1


def test_enum_literal_comparison_folds_into_compile_time_branch():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64], mask_mode: pl.DT_INT32 = 0):
        if MaskMode.NO_MASK == MaskMode.CAUSAL:
            y = pl.getval(x, 0)  # noqa: F841 -- DSL body, parsed as AST
        else:
            y = pl.getval(x, 1)  # noqa: F841 -- DSL body, parsed as AST

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(kernel.__name__)
    # The enum equality folds to a ConstBool at parse time: the if lowers to a
    # compile-time branch, and only the taken side reaches the IR body.
    assert not any(
        isinstance(stmt, ir.IfStmt) for stmt in func.body.stmts
    ), "literal enum comparison must fold away the IfStmt"


def test_int_attribute_comparison_stays_a_runtime_branch():
    # The port idiom replacing C++ static_cast<Enum>(attr): plain int compare.
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64], mask_mode: pl.DT_INT32 = 0):
        if mask_mode == MaskMode.NO_MASK.value:
            y = pl.getval(x, 0)  # noqa: F841 -- DSL body, parsed as AST
        else:
            y = pl.getval(x, 1)  # noqa: F841 -- DSL body, parsed as AST

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(kernel.__name__)
    assert any(
        isinstance(stmt, ir.IfStmt) for stmt in func.body.stmts
    ), "runtime int comparison must keep the runtime branch"


def test_enum_literal_as_op_operand_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64]):
        y = pl.getval(x, MaskMode.CAUSAL)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidType):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_enum_runtime_construction_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64], mask_mode: pl.DT_INT32 = 0):
        mode = MaskMode(mask_mode)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(NotSupported, match="Unsupported function call"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
