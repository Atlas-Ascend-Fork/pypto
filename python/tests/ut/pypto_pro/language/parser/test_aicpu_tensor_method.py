#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the aicpu host-tensor descriptor surface.

In an aicpu kernel the cpp descriptor guards lower in the frontend; the
element count is the op-channel function form. Correspondence:

    tensor is None / is not None <=> Tensor* != nullptr && GetData() != nullptr
    pl.numel(tensor)             <=> tensor->NumElements()

``t.numel()`` (method form) is NOT supported: the receiver is not injected, so
the call dies in the declared-params binding. ``pl.size`` does not exist at all.
"""
from __future__ import annotations

# DSL function bodies are parsed as AST, not executed -suppress pyright errors
# from type-checking the annotations and kwargs inside parsed DSL bodies.
from pypto_pro import ir
from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
import pytest

from .test_inline_helper_return import _walk_statements


def _parse(kernel):
    """Parse one jit kernel's target program and return (program, kernel_def)."""
    kdef = kernel.to_kernel_def()
    program, _ = kdef.parse_target_program(ir.SectionKind.Vector)
    return program, kdef


def test_storage_numel_is_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        n = t.storage().numel()  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(NotSupported, match="Unsupported operation call"):
        _parse(kernel)


def test_numel_is_num_elements():
    @pl.jit(mode="aicpu_kernel")
    def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        n = pl.numel(t)  # noqa: F841 -- DSL body, parsed as AST

    _parse(kernel)


def test_size_is_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        n = t.size()  # noqa: F841 -- DSL body, parsed as AST
        m = t.size(0)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(NotSupported, match="Unsupported operation call"):
        _parse(kernel)


def test_is_none_merges_descriptor_guards():
    # `tensor is None` is the cpp `Tensor* != nullptr && GetData() != nullptr`
    # guard in its flattened form: the tensor slot is the data pointer, so one
    # null test covers both levels.
    @pl.jit(mode="aicpu_kernel")
    def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        if t is None:
            v = 1  # noqa: F841 -- DSL body, parsed as AST
        if t is not None:
            v = 2  # noqa: F841 -- DSL body, parsed as AST

    _parse(kernel)


def test_tile_rejected_on_surface_ops():
    # A Tile reaches the surface ops via make_tile (its argument is a Python
    # TileType, so the reverse surface rule never sees a Tensor); the surface
    # check must reject it before the impl emits AI-Core flavored IR.
    with pytest.raises(NotSupported, match="on a Tile is not supported in aicpu kernels"):
        @pl.jit(mode="aicpu_kernel")
        def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
            tf = pl.TileType(shape=[16, 16], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tf, addr=0)
            v = pl.getval(tile, 0)  # noqa: F841 -- DSL body, parsed as AST

        _parse(kernel)


def test_tensor_is_none_supported_on_tile_kernel():
    # aicore and aicpu share one null-check path: a tensor lowers to its GM
    # data-pointer comparison, so 't is None' parses on tile kernels too.
    @pl.jit(auto_mutex=False)
    def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        if t is None:
            v = 1  # noqa: F841 -- DSL body, parsed as AST

    _parse(kernel)


def test_tile_kernel_keeps_op_dispatch_rejection():
    # The surface is aicpu-only: the builder itself rejects other function types.
    @pl.jit(auto_mutex=False)
    def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        n = pl.numel(t)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(NotSupported, match="only supported in aicpu kernels"):
        _parse(kernel)


def test_scalar_argument_rejected():
    # numel takes a Tensor only; a scalar argument dies in the declared-types gate.
    from pypto_pro._errors import InvalidType

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.DT_INT32):
        n = pl.numel(x)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidType, match="tensor expects Tensor, got a int32 scalar"):
        _parse(kernel)


def test_method_form_is_rejected():
    # t.numel() is not a supported spelling: the receiver is not injected, so
    # the call dies in the declared-params binding (missing first argument).
    from pypto_pro._errors import InvalidArgument

    @pl.jit(mode="aicpu_kernel")
    def kernel(t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        n = t.numel()  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidArgument, match="missing a required argument"):
        _parse(kernel)


def test_numel_of_zero_dim_tensor_is_one():
    # A 0-d (scalar) tensor has one element: the empty shape product folds to
    # the unit value, exactly like the cpp NumElements().
    @pl.jit(mode="aicpu_kernel")
    def kernel(t: pl.Tensor[[], pl.DT_INT32]):
        n = pl.numel(t)
        pl.setval(t, 0, n)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(kernel.__name__)
    stmts = list(_walk_statements(func.body))
    setval_call = next(
        stmt.value for stmt in stmts
        if isinstance(getattr(stmt, "value", None), ir.Call) and stmt.value.name == "block.setval"
    )
    # The empty product folded to the constant 1 and flows into the setval value.
    value_operand = setval_call.args[2]
    assert isinstance(value_operand, ir.ConstInt) and value_operand.value == 1
