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
"""Tests for the pypto-node ONNX post-pass ``finalize_pypto_onnx_nodes``.

Given an ``onnx.ModelProto`` whose pypto custom-op NodeProtos carry their
``op_export_record`` (domain / domain_opset_version), ``finalize_pypto_onnx_nodes`` makes a
single pass that:

* stamps each custom domain's ``opset_import`` from the export record (single source of truth),
* rejects a domain mismatch between node and op_export_record,
* rejects intra-domain opset disagreement,
* ignores non-pypto NodeProtos and records that omit the domain/opset fields,
* logs the stamp when it changes the graph — INFO when filling torch's default/unset value,
  WARNING when overriding an explicitly-declared opset that disagrees with codegen.

The pass reads ``op_export_record`` directly from each node's attributes — no module-level side channel.
"""
from __future__ import annotations

import json

from exported_custom_op_litenpu.export.onnx_export import (
    _exported_pypto_nodes,
    _rewrite_standard_domain_for_ge,
    finalize_pypto_onnx_nodes,
)
import onnx
from onnx import TensorProto, helper
import pytest

from pypto.extensions.torch_custom_op_litenpu.common.node_meta import (
    _LOCAL_NODE_LAYOUT_VERSION,
    _LOCAL_PYPTO_VERSION,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
)


def _make_pypto_node(
    op_type: str,
    *,
    node_domain: str,
    export_record: dict | None,
) -> onnx.NodeProto:
    """Build a NodeProto that ``_exported_pypto_nodes`` will recognize.

    ``export_record=None`` omits ``op_export_record`` — the input that drives the "no record" branch below.
    """
    node = helper.make_node(op_type, ["in0"], ["out0"], domain=node_domain, name="n")
    node.attribute.append(helper.make_attribute(_META_KEY__PYPTO_NODE_LAYOUT_VERSION, _LOCAL_NODE_LAYOUT_VERSION))
    node.attribute.append(helper.make_attribute(_META_KEY__PYPTO_PACKAGE_VERSION, _LOCAL_PYPTO_VERSION))
    node.attribute.append(helper.make_attribute("op_type", op_type))
    if export_record is not None:
        node.attribute.append(helper.make_attribute("op_export_record", json.dumps(export_record)))
    return node


def _model_with_node(
    node: onnx.NodeProto,
    *,
    default_opset: int = 11,
    custom_domain_opset: tuple[str, int] | None = None,
) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])
    g = helper.make_graph([node], "g", [inp], [out])
    imports = [helper.make_opsetid("", default_opset)]
    if custom_domain_opset is not None:
        d, v = custom_domain_opset
        imports.append(helper.make_opsetid(d, v))
    model = helper.make_model(g, opset_imports=imports, producer_name="ut")
    model.ir_version = 7
    return model


def _export_record(op_type: str, *, domain: str, domain_opset_version: int) -> dict:
    # The op_export_record a pypto node carries; finalize reads (domain, domain_opset_version).
    return {
        "op_type": op_type,
        "domain": domain,
        "domain_opset_version": domain_opset_version,
    }


def _opsets(m: onnx.ModelProto) -> dict[str, int]:
    return {imp.domain: imp.version for imp in m.opset_import}


# --- accept / ignore ---


def test_ok_default_pypto_domain_is_noop():
    node = _make_pypto_node(
        "MyOp", node_domain="pypto",
        export_record=_export_record("MyOp", domain="pypto", domain_opset_version=1),
    )
    m = _model_with_node(node, custom_domain_opset=("pypto", 1))
    finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))
    assert _opsets(m)["pypto"] == 1


def test_ok_custom_domain_and_opset():
    node = _make_pypto_node(
        "MyOp", node_domain="vendor.x",
        export_record=_export_record("MyOp", domain="vendor.x", domain_opset_version=7),
    )
    m = _model_with_node(node, custom_domain_opset=("vendor.x", 7))
    finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))
    assert _opsets(m)["vendor.x"] == 7


def test_no_pypto_nodes_is_noop():
    # Plain ONNX node — no pypto marker, so the node selection skips it.
    inp = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])
    plain = helper.make_node("Relu", ["in0"], ["out0"], name="r")
    g = helper.make_graph([plain], "g", [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 7
    before = m.SerializeToString()
    finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))
    assert m.SerializeToString() == before  # a true no-op: the model is byte-identical


# --- reject ---


def test_rejects_domain_mismatch():
    node = _make_pypto_node(
        "MyOp",
        node_domain="vendor.x",  # symbolic emitted node in this domain
        export_record=_export_record("MyOp", domain="pypto", domain_opset_version=1),  # codegen used another
    )
    m = _model_with_node(node, custom_domain_opset=("vendor.x", 1))
    with pytest.raises(RuntimeError, match=r"domain 'vendor.x'.*domain='pypto'"):
        finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))


def test_rejects_intra_domain_disagreement():
    n1 = _make_pypto_node("OpA", node_domain="pypto",
                          export_record=_export_record("OpA", domain="pypto", domain_opset_version=1))
    n2 = _make_pypto_node("OpB", node_domain="pypto",
                          export_record=_export_record("OpB", domain="pypto", domain_opset_version=2))
    inp = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])
    g = helper.make_graph([n1, n2], "g", [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11), helper.make_opsetid("pypto", 1)])
    m.ir_version = 7
    with pytest.raises(RuntimeError, match="disagree"):
        finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))


def test_repeated_op_type_with_agreeing_records_is_accepted():
    # Two instances of one op: every node is visited, and the shared record stamps the domain once.
    n1 = _make_pypto_node("MyOp", node_domain="pypto",
                          export_record=_export_record("MyOp", domain="pypto", domain_opset_version=4))
    n2 = _make_pypto_node("MyOp", node_domain="pypto",
                          export_record=_export_record("MyOp", domain="pypto", domain_opset_version=4))
    inp = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])
    g = helper.make_graph([n1, n2], "g", [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11), helper.make_opsetid("pypto", 1)])
    m.ir_version = 7
    finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))
    assert _opsets(m)["pypto"] == 4


def test_rejects_repeated_op_type_whose_records_disagree():
    # No op_type dedupe stands between the second instance and the check, so a disagreeing record raises.
    n1 = _make_pypto_node("MyOp", node_domain="pypto",
                          export_record=_export_record("MyOp", domain="pypto", domain_opset_version=1))
    n2 = _make_pypto_node("MyOp", node_domain="pypto",
                          export_record=_export_record("MyOp", domain="pypto", domain_opset_version=2))
    inp = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])
    g = helper.make_graph([n1, n2], "g", [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11), helper.make_opsetid("pypto", 1)])
    m.ir_version = 7
    with pytest.raises(RuntimeError, match="disagree"):
        finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))


# --- stamp from the export record (single source of truth) + sync logging ---


def test_stamps_domain_opset_from_export_record():
    # Model imports pypto at the torch default (1); the export record says 3. After finalize the opset_import
    # is stamped to 3 (one entry per domain).
    node = _make_pypto_node(
        "MyOp", node_domain="pypto",
        export_record=_export_record("MyOp", domain="pypto", domain_opset_version=3),
    )
    m = _model_with_node(node, custom_domain_opset=("pypto", 1))
    finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))
    assert _opsets(m)["pypto"] == 3
    domains = [imp.domain for imp in m.opset_import]
    assert len(domains) == len(set(domains)), f"duplicate opset_import domains: {domains}"


def test_appends_missing_domain_opset():
    # torch didn't add a pypto opset_import at all → finalize inserts it from the export record.
    node = _make_pypto_node(
        "MyOp", node_domain="pypto",
        export_record=_export_record("MyOp", domain="pypto", domain_opset_version=2),
    )
    m = _model_with_node(node, custom_domain_opset=None)
    finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))
    assert _opsets(m).get("pypto") == 2


# --- Export-helpers GE-compat rewrite (lives in the helper; kept here for the pypto-node interaction) ---


def test_ge_standard_domain_rewrite():
    # GE-compat: standard (empty-domain) nodes get the explicit domain + a matching opset_import;
    # pypto nodes are left alone.
    pypto_node = _make_pypto_node(
        "MyOp", node_domain="pypto",
        export_record=_export_record("MyOp", domain="pypto", domain_opset_version=1),
    )
    mul = helper.make_node("Mul", ["in0", "in0"], ["out0"], name="mul")  # domain "" by default
    inp = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])
    g = helper.make_graph([mul, pypto_node], "g", [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12), helper.make_opsetid("pypto", 1)])
    m.ir_version = 7

    _rewrite_standard_domain_for_ge(m, "ai.onnx")

    domains = {n.op_type: n.domain for n in m.graph.node}
    assert domains["Mul"] == "ai.onnx"   # standard op got an explicit domain
    assert domains["MyOp"] == "pypto"    # custom op untouched
    opsets = {imp.domain: imp.version for imp in m.opset_import}
    assert opsets["ai.onnx"] == 12       # mirrors the base ai.onnx opset
    assert opsets[""] == 12 and opsets["pypto"] == 1  # originals preserved


# --- GE-compat rewrite, nested graphs ---
# A standard node in an ``If`` branch carries ``domain == ""`` like a root one, which GE's onnx parser
# rejects once a custom domain is imported (``domain_version[N] should be equal to one``).


def _all_nodes_in_tree(graph: onnx.GraphProto):
    """Every node of *graph* and of every graph nested in a node attribute (test-local, independent walk)."""
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _all_nodes_in_tree(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from _all_nodes_in_tree(sub)


def _nested_mixed_model() -> onnx.ModelProto:
    """Root ``Cast`` + ``If``; the pypto node in ``then_branch``, a ``Constant``/``Mul`` in ``else_branch``.

    The shape a scripted ``torch.onnx.export`` of one data-dependent conditional actually produces.
    """
    pypto_node = _make_pypto_node(
        "MyOp", node_domain="pypto",
        export_record=_export_record("MyOp", domain="pypto", domain_opset_version=1),
    )
    del pypto_node.output[:]
    pypto_node.output.extend(["b_out"])
    then_branch = helper.make_graph([pypto_node], "then_g", [],
                                    [helper.make_tensor_value_info("b_out", TensorProto.FLOAT, [4])])
    const = helper.make_node(
        "Constant", [], ["two"], name="const",
        value=helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0]))
    else_branch = helper.make_graph(
        [const, helper.make_node("Mul", ["in0", "two"], ["b_out"], name="mul_in_else")],
        "else_g", [], [helper.make_tensor_value_info("b_out", TensorProto.FLOAT, [4])])
    graph = helper.make_graph(
        [helper.make_node("Cast", ["cond_in"], ["cond"], name="cast_at_root", to=TensorProto.BOOL),
         helper.make_node("If", ["cond"], ["out0"], name="the_if",
                          then_branch=then_branch, else_branch=else_branch)],
        "nested_mixed",
        [helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4]),
         helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("pypto", 1)], ir_version=9)
    onnx.checker.check_model(model)   # an invalid fixture would make the rewrite result meaningless
    return model


def test_ge_standard_domain_rewrite_covers_nested_graphs():
    m = _nested_mixed_model()

    # Precondition, and the measurement the fix is judged against: 4 empty-domain nodes in the tree, of
    # which only one (the root Cast) a root-graph-only rewrite would reach.
    empty_before = [n.name for n in _all_nodes_in_tree(m.graph) if n.domain == ""]
    assert sorted(empty_before) == ["cast_at_root", "const", "mul_in_else", "the_if"]
    assert [n.name for n in m.graph.node if n.domain == ""] == ["cast_at_root", "the_if"]

    _rewrite_standard_domain_for_ge(m, "ai.onnx")

    # The discriminator: zero empty-domain nodes anywhere. A root-only rewrite leaves
    # ['const', 'mul_in_else'] — the else_branch nodes GE then rejects.
    assert [n.name for n in _all_nodes_in_tree(m.graph) if n.domain == ""] == []
    domains = {n.name: n.domain for n in _all_nodes_in_tree(m.graph)}
    assert domains["mul_in_else"] == "ai.onnx"    # inside else_branch, one graph down
    assert domains["const"] == "ai.onnx"          # likewise
    assert domains["cast_at_root"] == "ai.onnx"   # the root node the old rewrite already handled
    assert domains["n"] == "pypto"      # "n" is _make_pypto_node's own name; the nested pypto node
    assert _opsets(m)["ai.onnx"] == 17            # mirrors the base ai.onnx opset, added once
    assert _opsets(m)[""] == 17 and _opsets(m)["pypto"] == 1
