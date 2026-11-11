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
"""Tests for ``exported_custom_op_litenpu.common.discovery.find_pypto_nodes``."""

from __future__ import annotations

from exported_custom_op_litenpu.common.discovery import find_pypto_nodes
import onnx
from onnx import TensorProto, helper
import pytest

from pypto.extensions.torch_custom_op_litenpu import extract_op_type as extract_op_type_attr
from pypto.extensions.torch_custom_op_litenpu.common.node_meta import (
    _LOCAL_NODE_LAYOUT_VERSION,
    _LOCAL_PYPTO_VERSION,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
)

# ---------------------------------------------------------------------------
# ONNX fixture helpers
# ---------------------------------------------------------------------------


def _onnx_pypto_node(
    op_type: str,
    *,
    name: str,
    fmt: str = _LOCAL_NODE_LAYOUT_VERSION,
    pkg: str = _LOCAL_PYPTO_VERSION,
) -> onnx.NodeProto:
    """Build a minimal ONNX node with the pypto marker attributes (+ op_type for dedup).

    *pkg* stays a parameter: attributes stack and the reader takes the FIRST, so a hardcoded stamp wins.
    """
    node = helper.make_node(
        op_type=op_type,
        inputs=["x"],
        outputs=[f"{name}_out"],
        domain="ai.onnx.contrib",
        name=name,
    )
    node.attribute.append(helper.make_attribute(_META_KEY__PYPTO_NODE_LAYOUT_VERSION, fmt))
    node.attribute.append(helper.make_attribute(_META_KEY__PYPTO_PACKAGE_VERSION, pkg))
    node.attribute.append(helper.make_attribute("op_type", op_type))
    return node


def _onnx_builtin_node(name: str) -> onnx.NodeProto:
    """Build a built-in ONNX node that carries none of pypto's node meta."""
    return helper.make_node(
        op_type="Relu",
        inputs=["x"],
        outputs=[f"{name}_out"],
        name=name,
    )


def _onnx_foreign_op_type_node(name: str) -> onnx.NodeProto:
    """A non-pypto node that happens to carry an ``op_type`` attribute (another
    framework's convention) but NOT the pypto marker — must be ignored."""
    node = helper.make_node(
        op_type="SomeOtherOp", inputs=["x"], outputs=[f"{name}_out"], name=name,
    )
    node.attribute.append(helper.make_attribute("op_type", "SomeOtherOp"))
    return node


def _build_onnx_model(*nodes: onnx.NodeProto) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=list(nodes),
        name="g",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    # opset_imports kept minimal — these cases put every node in the root graph.
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


# ---------------------------------------------------------------------------
# ONNX tests
# ---------------------------------------------------------------------------


def test_find_pypto_nodes_onnx_dedupes_and_orders():
    model = _build_onnx_model(
        _onnx_pypto_node("OpA", name="a1"),
        _onnx_builtin_node("relu"),
        _onnx_pypto_node("OpA", name="a2"),  # duplicate op_type
        _onnx_pypto_node("OpB", name="b1"),
    )
    nodes = find_pypto_nodes(model)
    assert [n.name for n in nodes] == ["a1", "b1"]


def test_find_pypto_nodes_onnx_op_types_filter():
    model = _build_onnx_model(
        _onnx_pypto_node("OpA", name="a1"),
        _onnx_pypto_node("OpB", name="b1"),
    )
    nodes = find_pypto_nodes(model, op_types=["OpB"])
    assert [n.name for n in nodes] == ["b1"]


def test_find_pypto_nodes_onnx_op_types_missing_raises():
    model = _build_onnx_model(_onnx_pypto_node("OpA", name="a1"))
    with pytest.raises(ValueError, match="OpC"):
        find_pypto_nodes(model, op_types=["OpC"])


def test_find_pypto_nodes_onnx_empty_model_returns_empty_list():
    model = _build_onnx_model(_onnx_builtin_node("relu"))
    assert not find_pypto_nodes(model)


def test_find_pypto_nodes_onnx_ignores_foreign_op_type_attr():
    # A node carrying only an ``op_type`` attribute (another framework) is NOT pypto —
    # the marker is what discriminates, so this must be skipped (the collision fix).
    model = _build_onnx_model(
        _onnx_foreign_op_type_node("foreign"),
        _onnx_pypto_node("OpA", name="a1"),
    )
    assert [n.name for n in find_pypto_nodes(model)] == ["a1"]


def test_find_pypto_nodes_onnx_rejects_newer_node_layout_version():
    newer = str(int(_LOCAL_NODE_LAYOUT_VERSION) + 1)
    model = _build_onnx_model(_onnx_pypto_node("OpA", name="a1", fmt=newer))
    with pytest.raises(RuntimeError, match="pypto_node_layout_version"):
        find_pypto_nodes(model)


def test_find_pypto_nodes_onnx_does_not_gate_the_package_version():
    # Discovery gates the node-attribute layout only: a node recording a newer pypto release than this one
    # is still returned, since the release a node was exported with says nothing about whether its
    # attributes can be read.
    model = _build_onnx_model(_onnx_pypto_node("OpA", name="a1", pkg="999.0.0"))
    assert [n.name for n in find_pypto_nodes(model)] == ["a1"]


# ---------------------------------------------------------------------------
# Nested-graph discovery: a pypto node in an ``If`` branch must reach deploy.
# ---------------------------------------------------------------------------

_F32 = TensorProto.FLOAT


def _vi(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, _F32, [1])


def _checked_model(graph: onnx.GraphProto, *, extra_opsets=()) -> onnx.ModelProto:
    """Assemble and validate: an invalid fixture would make any discovery result meaningless."""
    opsets = [helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.contrib", 1)]
    opsets += [helper.make_opsetid(domain, version) for domain, version in extra_opsets]
    model = helper.make_model(graph, opset_imports=opsets, ir_version=9)
    onnx.checker.check_model(model)
    return model


def _nested_pypto_node(op_type: str, *, name: str, outputs, inputs=("x",)) -> onnx.NodeProto:
    """``_onnx_pypto_node`` with caller-chosen wiring, so it can sit inside a subgraph."""
    node = _onnx_pypto_node(op_type, name=name)
    del node.input[:]
    node.input.extend(inputs)
    del node.output[:]
    node.output.extend(outputs)
    return node


def _root_control_model(op_type: str) -> onnx.ModelProto:
    """The positive control: the identical marked node, at the root graph."""
    node = _nested_pypto_node(op_type, name="at_root", outputs=["y"])
    return _checked_model(helper.make_graph([node], "root_control", [_vi("x")], [_vi("y")]))


def _if_nested_model(op_type: str) -> onnx.ModelProto:
    marked = helper.make_graph(
        [_nested_pypto_node(op_type, name="in_then", outputs=["b_out"])],
        "then_g", [], [_vi("b_out")])
    plain = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["b_out"], name="plain")], "else_g", [], [_vi("b_out")])
    node = helper.make_node("If", ["cond"], ["y"], name="the_if", then_branch=marked, else_branch=plain)
    return _checked_model(helper.make_graph(
        [node], "if_nested",
        [_vi("x"), helper.make_tensor_value_info("cond", TensorProto.BOOL, [])], [_vi("y")]))


def test_find_pypto_nodes_reaches_a_node_in_an_if_branch():
    # The paired root control is the evidence: without it a zero could equally be a broken fixture.
    assert [n.name for n in find_pypto_nodes(_root_control_model("OpInIfBranch"))] == ["at_root"]
    assert [extract_op_type_attr(n)
            for n in find_pypto_nodes(_if_nested_model("OpInIfBranch"))] == ["OpInIfBranch"]


def test_find_pypto_nodes_dedupes_a_root_and_a_nested_occurrence_keeping_the_root():
    # One op_type at both depths, the shape a model calling it in a branch and again at the root gives.
    # Deploy needs one entry per op_type, and the depth-first walk makes the kept one the root node.
    branch = helper.make_graph(
        [_nested_pypto_node("OpA", name="in_then", inputs=("t0",), outputs=["b_out"])],
        "then_g", [], [_vi("b_out")])
    plain = helper.make_graph(
        [helper.make_node("Identity", ["t0"], ["b_out"], name="plain")], "else_g", [], [_vi("b_out")])
    model = _checked_model(helper.make_graph(
        [_nested_pypto_node("OpA", name="at_root", outputs=["t0"]),
         helper.make_node("If", ["cond"], ["y"], name="the_if", then_branch=branch, else_branch=plain)],
        "root_and_nested",
        [_vi("x"), helper.make_tensor_value_info("cond", TensorProto.BOOL, [])], [_vi("y")]))
    assert [n.name for n in find_pypto_nodes(model)] == ["at_root"]


def test_find_pypto_nodes_op_types_filter_resolves_a_nested_node():
    # The explicit-subset path must see nested nodes too: otherwise asking for an op that lives in a
    # branch raises "op_types not found in model" even though the artifact carries it.
    model = _if_nested_model("OpNested")
    assert [extract_op_type_attr(n) for n in find_pypto_nodes(model, op_types=["OpNested"])] == ["OpNested"]


def test_find_pypto_nodes_still_ignores_an_unmarked_nested_node():
    # Negative case: descending into a subgraph does not relax the marker gate.
    # In another framework's own imported domain, so the checker does not demand an ai.onnx schema for it
    # (``_onnx_foreign_op_type_node`` leaves the domain empty, which only works for an unchecked model).
    foreign = helper.make_node("SomeOtherOp", ["x"], ["foreign_out"],
                               domain="other.framework", name="foreign_in_branch")
    foreign.attribute.append(helper.make_attribute("op_type", "SomeOtherOp"))
    branch = helper.make_graph([foreign], "then_g", [], [_vi("foreign_out")])
    plain = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["foreign_out"], name="plain")],
        "else_g", [], [_vi("foreign_out")])
    model = _checked_model(
        helper.make_graph(
            [helper.make_node("If", ["cond"], ["y"], name="the_if",
                              then_branch=branch, else_branch=plain)],
            "neg_nested",
            [_vi("x"), helper.make_tensor_value_info("cond", TensorProto.BOOL, [])], [_vi("y")]),
        extra_opsets=[("other.framework", 1)])
    assert not find_pypto_nodes(model)
