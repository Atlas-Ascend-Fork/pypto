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
"""Unit tests for TypeResolver."""
# DSL function bodies are parsed as AST, not executed -suppress pyright errors
# from type-checking the annotations and kwargs inside kernel bodies.
import ast
from typing import TYPE_CHECKING, Any

from pypto_pro import DataType, ir
from pypto_pro._errors import InvalidArgument, InvalidShape, InvalidType
import pypto_pro.language as pl
from pypto_pro.language.parser._expr_evaluator import ExprEvaluator
from pypto_pro.language.parser._type_resolver import TypeResolver
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_resolver(
    closure_vars: dict | None = None, scope_lookup: "Callable[[str], Any | None] | None" = None
) -> TypeResolver:
    """Create a TypeResolver with ExprEvaluator from closure_vars."""
    namespace = {"pl": pl, "ND": pl.ND, "NZ": pl.NZ}
    namespace.update(closure_vars or {})
    ev = ExprEvaluator(closure_vars=namespace)
    return TypeResolver(expr_evaluator=ev, scope_lookup=scope_lookup)


@pytest.mark.parametrize(
    "layout_str, expected_layout",
    [
        ("pl.NZ", ir.TensorLayout.NZ),
        ("pl.ND", ir.TensorLayout.ND),
    ],
)
def test_resolve_tensor_with_layout(layout_str, expected_layout):
    """Tensor with various layouts creates TensorType with TensorView."""
    resolver = _make_resolver()
    node = ast.parse(f"pl.Tensor[[64, 128], pl.DT_FP16, {layout_str}]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert len(result.shape) == 2
    assert result.dtype == DataType.FP16
    assert result.tensor_view is not None
    assert result.tensor_view.layout == expected_layout


def test_resolve_tensor_without_layout_has_no_view():
    """Tensor without an explicit layout has no TensorView."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is None


def test_resolve_tensor_layout_invalid():
    """Unsupported Tensor layout reports the supported layout set."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, pl.INVALID]", mode="eval").body

    with pytest.raises(
        InvalidArgument,
        match="Unknown layout: INVALID",
    ):
        resolver.resolve_type(node)


def test_resolve_nz_tensor_rejects_rank_less_than_two():
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64], pl.DT_FP16, pl.NZ]", mode="eval").body

    with pytest.raises(InvalidShape, match="NZ Tensor requires rank >= 2"):
        resolver.resolve_type(node)


def test_resolve_layout_bare_name():
    """Layout specified as bare name (NZ) instead of pl.NZ."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, NZ]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_resolve_layout_from_closure():
    """Layout from closure variable."""
    resolver = _make_resolver(closure_vars={"my_layout": ir.TensorLayout.NZ})
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, my_layout]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_resolve_layout_closure_invalid_type():
    """Non-TensorLayout closure variable raises error."""
    resolver = _make_resolver(closure_vars={"my_layout": "NZ"})
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, my_layout]", mode="eval").body

    with pytest.raises(InvalidType, match="must be a TensorLayout"):
        resolver.resolve_type(node)


def test_resolve_tensor_layout_with_dynamic_shape():
    """Layout works with dynamic shapes."""
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse("pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.NZ]", mode="eval").body
    result = resolver.resolve_param_type(node, parameter_name="x")

    assert isinstance(result, ir.TensorType)
    assert isinstance(result.shape[0], ir.Var)
    assert result.shape[0].name == "__pypto_dyn_x_0"
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_resolve_tensor_rejects_dn_with_shape_variable():
    """A shape variable must not bypass explicit DN annotation validation."""
    resolver = _make_resolver(closure_vars={"shape": [64, 128]})
    node = ast.parse("pl.Tensor[shape, pl.DT_FP16, pl.DN]", mode="eval").body
    with pytest.raises(InvalidArgument, match="Unknown layout: DN"):
        resolver.resolve_type(node)


@pytest.mark.parametrize(
    "annotation",
    [
        "pl.Tensor[[64, 128], pl.DT_FP16, pl.DN]",
        "pl.Tensor[[64, 128], pl.DT_FP16, DN]",
        "pl.Tensor[[64, 128], pl.DT_FP16, layout_alias]",
        "pl.Tensor[[64, 128], pl.DT_FP16, pl.DN, pl.MemRef(pl.MemorySpace.DDR, 0, 16384, 0)]",
        "pl.Tensor[[64, 128], pl.DT_FP16, pl.DN, pl.Input]",
        "pl.Tensor[[64], pl.DT_FP16, pl.DN]",
        "pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.DN]",
    ],
)
def test_resolve_tensor_rejects_explicit_dn_annotation(annotation):
    resolver = _make_resolver(closure_vars={"pl": pl, "layout_alias": pl.DN})
    node = ast.parse(annotation, mode="eval").body
    with pytest.raises(InvalidArgument, match="Unknown layout: DN"):
        resolver.resolve_param_type(node, parameter_name="x")


@pytest.mark.parametrize("layout_name", ["ZN", "NN", "ZZ"])
def test_resolve_tensor_rejects_other_explicit_layouts(layout_name):
    resolver = _make_resolver()
    node = ast.parse(f"pl.Tensor[[64, 128], pl.DT_FP16, pl.{layout_name}]", mode="eval").body

    with pytest.raises(
        InvalidArgument,
        match=rf"Unknown layout: {layout_name}",
    ):
        resolver.resolve_type(node)


@pytest.mark.parametrize("layout", [pl.ZN, pl.NN, pl.ZZ])
def test_resolve_tensor_rejects_other_layouts_from_closure(layout):
    resolver = _make_resolver(closure_vars={"layout_alias": layout})
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP16, layout_alias]", mode="eval").body

    with pytest.raises(InvalidArgument, match=rf"Unknown layout: {layout.name}"):
        resolver.resolve_type(node)


@pytest.mark.parametrize("shape", ["[pl.DYNAMIC, 128]", "[pl.STATIC, 128]", "[64, ...]"])
def test_policy_helper_parameter_rejects_explicit_dn(shape):
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse(f"pl.Tensor[{shape}, pl.DT_FP16, pl.DN]", mode="eval").body
    actual = ir.TensorType([64, 128], DataType.FP16)
    with pytest.raises(InvalidArgument, match="Unknown layout: DN"):
        resolver.validate_policy_parameter_type(node, "x", actual)


@pytest.mark.parametrize("layout_name", ["ZN", "NN", "ZZ"])
def test_policy_helper_parameter_rejects_other_explicit_layouts(layout_name):
    resolver = _make_resolver(closure_vars={"pl": pl})
    node = ast.parse(f"pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.{layout_name}]", mode="eval").body
    actual = ir.TensorType([64, 128], DataType.FP16)

    with pytest.raises(InvalidArgument, match=rf"Unknown layout: {layout_name}"):
        resolver.validate_policy_parameter_type(node, "x", actual)


def test_policy_helper_parameter_rejects_non_layout_value():
    resolver = _make_resolver(closure_vars={"pl": pl, "layout_alias": "DN"})
    node = ast.parse(
        "pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, layout_alias]", mode="eval"
    ).body
    actual = ir.TensorType([64, 128], DataType.FP16)

    with pytest.raises(InvalidType, match="Tensor layout must be a TensorLayout, got str"):
        resolver.validate_policy_parameter_type(node, "x", actual)


def test_function_with_tensor_layout():
    """Kernel parameter annotations preserve tensor layout."""

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    assert isinstance(func, ir.Function)
    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.dtype == DataType.FP16
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == ir.TensorLayout.NZ


def test_function_mixed_layout_and_no_layout():
    """Kernel with some params having layout and some not."""

    @pl.jit(auto_mutex=False)
    def func(
        a: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ],
        b: pl.Tensor[[64, 128], pl.DT_FP16],
    ):
        _test_result = a

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    a_type = func.params[0].type
    b_type = func.params[1].type
    assert isinstance(a_type, ir.TensorType)
    assert isinstance(b_type, ir.TensorType)
    assert a_type.tensor_view is not None
    assert a_type.tensor_view.layout == ir.TensorLayout.NZ
    assert b_type.tensor_view is not None


def test_program_with_tensor_layout():
    """Kernel with layout annotations."""

    @pl.jit(auto_mutex=False)
    def compute(
        x: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ],
    ):
        _test_result = x

    compute_program, _ = compute.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    compute = compute_program.get_function(compute.__name__)

    param_type = compute.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == ir.TensorLayout.NZ


def test_function_layout_from_closure_variable():
    """Kernel with layout from closure variable."""
    layout = pl.NZ

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[64, 128], pl.DT_FP16, layout],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == ir.TensorLayout.NZ


@pytest.mark.parametrize(
    "layout, expected",
    [
        (pl.NZ, ir.TensorLayout.NZ),
        (pl.ND, ir.TensorLayout.ND),
    ],
)
def test_parametrized_layout(layout, expected):
    """pytest.mark.parametrize with layout."""

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[64, 128], pl.DT_FP16, layout],
    ):
        _test_result = x

    func_program, _ = func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = func_program.get_function(func.__name__)

    param_type = func.params[0].type
    assert isinstance(param_type, ir.TensorType)
    assert param_type.tensor_view is not None
    assert param_type.tensor_view.layout == expected


def test_function_rejects_dn_annotation_without_memory_operations():
    """The declaration must fail even when the kernel never calls load/store."""

    @pl.jit(auto_mutex=False)
    def func(
        x: pl.Tensor[[16, 1], pl.DT_FP16, pl.DN],
    ):
        _test_result = x

    with pytest.raises(InvalidArgument, match="Unknown layout: DN"):
        func.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_backward_compat_bare_layout():
    """pl.Tensor[[shape], dtype, pl.NZ] layout syntax works."""
    resolver = _make_resolver()
    node = ast.parse("pl.Tensor[[64, 128], pl.DT_FP32, pl.NZ]", mode="eval").body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.tensor_view is not None
    assert result.tensor_view.layout == ir.TensorLayout.NZ


def test_backward_compat_memref():
    """pl.Tensor[[shape], dtype, pl.MemRef(...)] syntax works."""
    resolver = _make_resolver()
    node = ast.parse(
        "pl.Tensor[[64], pl.DT_FP32, pl.MemRef(pl.MemorySpace.DDR, 0, 1024, 0)]", mode="eval"
    ).body
    result = resolver.resolve_type(node)

    assert isinstance(result, ir.TensorType)
    assert result.memref is not None
    assert result.tensor_view is None
