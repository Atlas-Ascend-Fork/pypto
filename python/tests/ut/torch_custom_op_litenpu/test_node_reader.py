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
"""Node-reader guard: the ``pypto_node_layout_version`` gate and the per-field extractor surface.

Needs onnx + torch and nothing else — no CANN, no device, no examples tree. Most cases read nodes
synthesized here under the names ``node_meta`` declares; one reads back a real ``torch.onnx.export``.
The gate's two version rejections raise a plain ``RuntimeError``, so each is pinned by the substring only
it emits — ``unparseable pypto_node_layout_version`` for a value that is not an integer string, and
``supports up to`` for one this reader is too old for.
"""
import json

import onnx
import pytest
import torch
import torch.nn as nn

import pypto
from pypto.extensions.torch_custom_op_litenpu import (
    ExportedCustomOp,
    OnnxSymbolicSpec,
    check_pypto_node_layout,
    extract_kernel_compile_snippet,
    extract_op_export_record,
    extract_op_type,
    extract_pypto_package_version,
    finalize_pending_ops,
    iter_pypto_nodes,
)
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _reset_export_state
from pypto.extensions.torch_custom_op_litenpu.common.node_meta import (
    _LOCAL_NODE_LAYOUT_VERSION,
    _LOCAL_PYPTO_VERSION,
    _META_KEY__KERNEL_COMPILE_SNIPPET,
    _META_KEY__OP_EXPORT_RECORD,
    _META_KEY__OP_TYPE,
    _META_KEY__PYPTO_NODE_LAYOUT_VERSION,
    _META_KEY__PYPTO_PACKAGE_VERSION,
)
from pypto.extensions.torch_custom_op_litenpu.common.torch_op import exporting_scope
from pypto.extensions.torch_custom_op_litenpu.onnx.export import _qualify_op_type

_QUALNAME = "pypto::node_probe"
_SPEC = OnnxSymbolicSpec(op_type="NodeProbe")
_OP_TYPE = _qualify_op_type(_SPEC.op_type)
_SHAPE = (1, 4, 1, 64)         # 256 elements; only the node meta is under test, so one shape is enough
_DTYPE = torch.float16


# ── authoring functions (module scope: the tracer reads their source off disk) ────────────────────────
def create_node_probe_kernel(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def node_probe_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                         out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b)

    return node_probe_inner


def _node_probe_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _node_probe_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _node_probe_torch(a, b):
    return a + b


class _NodeProbeModel(nn.Module):
    def forward(self, a, b):
        return torch.ops.pypto.node_probe(a, b)


def _meta_node(**attrs) -> onnx.NodeProto:
    """An ONNX node carrying each of *attrs* under its real node-meta key.

    Values are ``str``, so ``make_attribute`` emits ONNX STRING attributes — the one type readers accept.
    """
    node = onnx.helper.make_node(_OP_TYPE, inputs=["a", "b"], outputs=["y"], domain=_SPEC.domain)
    attrs = dict(attrs)
    # Every node carries this key (a real export always stamps it), but setdefault not append: attributes
    # stack and the reader takes the FIRST match, so a hardcoded stamp would win over a case's own value.
    attrs.setdefault(_META_KEY__PYPTO_PACKAGE_VERSION, _LOCAL_PYPTO_VERSION)
    for name, value in attrs.items():
        node.attribute.append(onnx.helper.make_attribute(name, value))
    return node


@pytest.fixture()
def exported_node(tmp_path):
    """The single pypto node of a real ``torch.onnx.export``, reloaded from disk."""
    _reset_export_state()
    ExportedCustomOp(
        kernel=create_node_probe_kernel,
        infer_shape=_node_probe_infer_shape,
        infer_dtype=_node_probe_infer_dtype,
        torch_defn=_node_probe_torch,
        torch_op_qualname=_QUALNAME,
        onnx_spec=_SPEC,
    )
    finalize_pending_ops()
    path = str(tmp_path / "node_probe.onnx")
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    with exporting_scope():
        torch.onnx.export(
            _NodeProbeModel(), (a, b), path,
            input_names=["a", "b"], output_names=["y"],
            opset_version=_SPEC.opset_version, do_constant_folding=False, dynamo=False,
            custom_opsets={_SPEC.domain: _SPEC.domain_opset_version},
        )
    model = onnx.load(path)
    nodes = [n for n in model.graph.node if n.op_type == _OP_TYPE]
    assert len(nodes) == 1, (
        f"expected exactly one {_OP_TYPE} node — a second means another declaration's symbolic is still "
        f"registered and the reads below would resolve its node meta — got "
        f"{[(n.op_type, n.domain) for n in model.graph.node]}"
    )
    yield nodes[0]
    _reset_export_state()


def test_node_layout_version_current_is_accepted():
    node = _meta_node(**{_META_KEY__PYPTO_NODE_LAYOUT_VERSION: _LOCAL_NODE_LAYOUT_VERSION})
    check_pypto_node_layout(node)   # this reader must accept the version this pypto itself writes


def test_node_layout_version_newer_is_refused_as_unsupported():
    # A readable version above this reader's: the artifact may carry attributes this pypto cannot parse,
    # so the gate refuses instead of mis-parsing downstream.
    newer = str(int(_LOCAL_NODE_LAYOUT_VERSION) + 1)
    with pytest.raises(RuntimeError) as excinfo:
        check_pypto_node_layout(_meta_node(**{_META_KEY__PYPTO_NODE_LAYOUT_VERSION: newer}))
    message = str(excinfo.value)
    assert f"pypto_node_layout_version={newer}" in message
    assert f"supports up to {_LOCAL_NODE_LAYOUT_VERSION}" in message
    assert "unparseable" not in message      # the value parsed fine; only the comparison failed


def test_node_layout_version_unparseable_is_refused_as_malformed():
    # Same exception type, different diagnosis: nothing was compared here because the value never
    # became an int, so the message must not claim a version range.
    with pytest.raises(RuntimeError) as excinfo:
        check_pypto_node_layout(_meta_node(**{_META_KEY__PYPTO_NODE_LAYOUT_VERSION: "1.0-beta"}))
    message = str(excinfo.value)
    assert "unparseable pypto_node_layout_version='1.0-beta'" in message
    assert "supports up to" not in message


def test_extractors_read_their_own_meta_keys():
    # One extractor per readable node-meta key, each resolving the name node_meta declares for it — a rename or a
    # crossed wiring shows up here rather than as a wrong field somewhere downstream.
    node = _meta_node(**{
        _META_KEY__OP_TYPE: _OP_TYPE,
        _META_KEY__KERNEL_COMPILE_SNIPPET: "def __pypto_compile():\n    return None\n",
        _META_KEY__OP_EXPORT_RECORD: json.dumps({"framework_kind": "onnx", "n_inputs": 2}),
    })
    assert extract_op_type(node) == _OP_TYPE
    # Surrounding whitespace is stripped off every string field, so the snippet's trailing newline is gone.
    assert extract_kernel_compile_snippet(node) == "def __pypto_compile():\n    return None"
    assert json.loads(extract_op_export_record(node)) == {"framework_kind": "onnx", "n_inputs": 2}


def test_extract_pypto_package_version_reads_the_recorded_value():
    # The public wrapper resolves the same key the private helper does, under the extract_* naming every
    # other readable field uses.
    node = _meta_node(**{_META_KEY__PYPTO_PACKAGE_VERSION: "1.2.3"})
    assert extract_pypto_package_version(node) == "1.2.3"


def test_read_back_an_exported_node(exported_node):
    # Write-then-read over the real boundary: what the export path serialized to disk is what the
    # extractors resolve off the reloaded node, under the same keys the synthesized cases use.
    check_pypto_node_layout(exported_node)   # the export path writes a version this reader accepts
    assert extract_op_type(exported_node) == _OP_TYPE
    # The snippet is the kernel source of truth: the factory arrives under its authored name.
    assert f"def {create_node_probe_kernel.__name__}(" in extract_kernel_compile_snippet(exported_node)
    record = json.loads(extract_op_export_record(exported_node))
    assert record["framework_kind"] == "onnx"
    assert record["mode"] == "factory"
    assert record["create_kernel_name"] == create_node_probe_kernel.__name__
    assert record["n_inputs"] == 2
    assert extract_pypto_package_version(exported_node) == _LOCAL_PYPTO_VERSION


# ── nested-graph discovery ────────────────────────────────────────────────────────────────────────────
# Dispatch is on the attribute's declared type (GRAPH / GRAPHS), so these cover every standard
# container plus the surface an op-name list misses: a GRAPHS attribute.

_NESTED_F32 = onnx.TensorProto.FLOAT
_NESTED_IR_VERSION = 9
_NESTED_OPSET = 17


def _vi(name, shape=(1,)):
    return onnx.helper.make_tensor_value_info(name, _NESTED_F32, list(shape))


def _marked_node(op_type, inputs, outputs, *, name, layout=_LOCAL_NODE_LAYOUT_VERSION):
    """A marked node in a custom domain, so no schema constrains it.

    *layout* is the raw marker value; a non-str makes it a non-STRING attribute the reader must reject.
    """
    node = onnx.helper.make_node(op_type, inputs, outputs, domain="pypto", name=name)
    node.attribute.append(onnx.helper.make_attribute(_META_KEY__PYPTO_NODE_LAYOUT_VERSION, layout))
    node.attribute.append(onnx.helper.make_attribute(_META_KEY__PYPTO_PACKAGE_VERSION, _LOCAL_PYPTO_VERSION))
    node.attribute.append(onnx.helper.make_attribute(_META_KEY__OP_TYPE, op_type))
    return node


def _plain_branch(feeds="x"):
    """The other ``If`` branch: ``If`` requires both and the checker enforces it; only one is under test."""
    return onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", [feeds], ["b_out"], name="plain")],
        "plain_g", [], [_vi("b_out")],
    )


def _nested_model(graph, *, extra_opsets=()):
    """Assemble and validate a model: a fixture that fails the checker proves nothing."""
    opsets = [onnx.helper.make_opsetid("", _NESTED_OPSET), onnx.helper.make_opsetid("pypto", 1)]
    opsets += [onnx.helper.make_opsetid(domain, version) for domain, version in extra_opsets]
    model = onnx.helper.make_model(graph, opset_imports=opsets, ir_version=_NESTED_IR_VERSION)
    onnx.checker.check_model(model)
    return model


def _found_op_types(model):
    """The op_types the reader yields, in yield order, with no dedupe."""
    return [extract_op_type(node) for node in iter_pypto_nodes(model)]


def _root_control(op_type):
    """The positive control: the identical marked node, at the root graph."""
    graph = onnx.helper.make_graph(
        [_marked_node(op_type, ["x"], ["y"], name="marked_at_root")],
        "root_control", [_vi("x")], [_vi("y")],
    )
    return _nested_model(graph)


def _if_model(branch, op_type):
    """pypto node inside ``If.<branch>``; the other branch is a plain Identity."""
    marked_branch = onnx.helper.make_graph(
        [_marked_node(op_type, ["x"], ["b_out"], name="marked_in_branch")],
        "marked_g", [], [_vi("b_out")],
    )
    other = "else_branch" if branch == "then_branch" else "then_branch"
    node = onnx.helper.make_node("If", ["cond"], ["y"], name="the_if",
                                 **{branch: marked_branch, other: _plain_branch()})
    graph = onnx.helper.make_graph(
        [node], "if_nested",
        [_vi("x"), onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])], [_vi("y")],
    )
    return _nested_model(graph)


def _loop_body(inner_nodes, tag):
    """A Loop body ``(iter, cond, v) -> (cond_out, v_out)``, names tagged so nesting stays SSA-clean."""
    return onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", [f"keep{tag}"], [f"cond_out{tag}"], name=f"c{tag}")]
        + list(inner_nodes),
        f"loop_body{tag}",
        [onnx.helper.make_tensor_value_info(f"iter_n{tag}", onnx.TensorProto.INT64, []),
         onnx.helper.make_tensor_value_info(f"keep{tag}", onnx.TensorProto.BOOL, []),
         _vi(f"v{tag}")],
        [onnx.helper.make_tensor_value_info(f"cond_out{tag}", onnx.TensorProto.BOOL, []),
         _vi(f"v_out{tag}")],
    )


def _loop_model(op_type, depth=1):
    """pypto node inside ``Loop.body``; *depth* > 1 wraps it in that many nested ``Loop`` bodies."""
    body = _loop_body([_marked_node(op_type, ["v_d1"], ["v_out_d1"], name="marked_in_loop")], "_d1")
    for level in range(2, depth + 1):
        tag = f"_d{level}"
        body = _loop_body(
            [onnx.helper.make_node("Loop", [f"iter_n{tag}", f"keep{tag}", f"v{tag}"], [f"v_out{tag}"],
                                   name=f"inner_loop{tag}", body=body)],
            tag,
        )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Loop", ["trip", "cond", "x"], ["y"], name="the_loop", body=body)],
        "loop_nested",
        [onnx.helper.make_tensor_value_info("trip", onnx.TensorProto.INT64, []),
         onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []), _vi("x")],
        [_vi("y")],
    )
    return _nested_model(graph)


def _scan_model(op_type):
    """pypto node inside ``Scan.body``."""
    body = onnx.helper.make_graph(
        [_marked_node(op_type, ["s_in"], ["s_out"], name="marked_in_scan"),
         onnx.helper.make_node("Identity", ["s_out"], ["y_out"], name="emit")],
        "scan_body", [_vi("s_in"), _vi("e_in")], [_vi("s_out"), _vi("y_out")],
    )
    node = onnx.helper.make_node("Scan", ["init", "seq"], ["s_final", "y_all"],
                                 name="the_scan", num_scan_inputs=1, body=body)
    graph = onnx.helper.make_graph(
        [node], "scan_nested", [_vi("init"), _vi("seq", (3, 1))],
        [_vi("s_final"), _vi("y_all", (3, 1))],
    )
    return _nested_model(graph)


def _sequence_map_model(op_type):
    """pypto node inside ``SequenceMap.body`` — the container the issue itself does not mention."""
    body = onnx.helper.make_graph(
        [_marked_node(op_type, ["elem"], ["out_elem"], name="marked_in_sequence_map")],
        "sequence_map_body", [_vi("elem")], [_vi("out_elem")],
    )
    node = onnx.helper.make_node("SequenceMap", ["seq"], ["out_seq"], name="the_sequence_map", body=body)
    graph = onnx.helper.make_graph(
        [node], "sequence_map_nested",
        [onnx.helper.make_tensor_sequence_value_info("seq", _NESTED_F32, [1])],
        [onnx.helper.make_tensor_sequence_value_info("out_seq", _NESTED_F32, [1])],
    )
    return _nested_model(graph)


def _graphs_attribute_model(op_type):
    """A custom container whose attribute is GRAPHS (a list), not GRAPH.

    No standard op uses GRAPHS today, so this can only come from a custom op — which is exactly why the
    walk dispatches on attribute type instead of on a list of op names.
    """
    plain = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["o"], name="plain")], "branch0", [], [_vi("o")])
    marked = onnx.helper.make_graph(
        [_marked_node(op_type, ["x"], ["o"], name="marked_in_graphs")], "branch1", [], [_vi("o")])
    node = onnx.helper.make_node("MultiBranch", ["x"], ["y"], domain="pypto", name="the_multibranch")
    node.attribute.append(onnx.helper.make_attribute("branches", [plain, marked]))
    assert node.attribute[0].type == onnx.AttributeProto.GRAPHS, "fixture must exercise GRAPHS, not GRAPH"
    return _nested_model(onnx.helper.make_graph([node], "graphs_nested", [_vi("x")], [_vi("y")]))


def _if_then_model(op_type):
    """``_if_model`` bound to the then branch, so the matrix entry stays a plain reference."""
    return _if_model("then_branch", op_type)


def _if_else_model(op_type):
    """``_if_model`` bound to the else branch."""
    return _if_model("else_branch", op_type)


def _four_nested_loops_model(op_type):
    """``_loop_model`` at the deepest homogeneous nesting the matrix covers."""
    return _loop_model(op_type, depth=4)


@pytest.mark.parametrize(
    ("op_type", "build"),
    [
        ("PyptoInIfThen", _if_then_model),
        ("PyptoInIfElse", _if_else_model),
        ("PyptoInLoopBody", _loop_model),
        ("PyptoInFourNestedLoops", _four_nested_loops_model),
        ("PyptoInScanBody", _scan_model),
        ("PyptoInSequenceMapBody", _sequence_map_model),
        ("PyptoInGraphsAttribute", _graphs_attribute_model),
    ],
)
def test_iter_pypto_nodes_reaches_a_node_in_every_graph_container(op_type, build):
    # The nested result alone is not the evidence: the paired root control, holding the identical marked
    # node, is what distinguishes "the walk is blind" from "the fixture or the marker is wrong".
    assert _found_op_types(_root_control(op_type)) == [op_type], "positive control at root must be found"
    assert _found_op_types(build(op_type)) == [op_type]


def test_iter_pypto_nodes_reaches_every_level_of_a_heterogeneous_container_chain():
    # Nested inside one another, five deep: a walk that handles each kind alone can still stop at the
    # first kind it will not recurse through, and the per-level op_type names which. The expected list
    # is the documented walk order, so it doubles as the depth sequence; deepest is the GRAPHS container.
    multi_branch = onnx.helper.make_node("MultiBranch", ["sm_marked"], ["sm_out"], domain="pypto",
                                         name="the_multibranch")
    multi_branch.attribute.append(onnx.helper.make_attribute("branches", [
        _plain_branch("sm_marked"),
        onnx.helper.make_graph(
            [_marked_node("PyptoComboInGraphs", ["sm_marked"], ["mb_out"], name="marked_in_graphs")],
            "combo_graphs_branch", [], [_vi("mb_out")]),
    ]))
    sequence_map_body = onnx.helper.make_graph(
        [_marked_node("PyptoComboInSequenceMap", ["sm_elem"], ["sm_marked"],
                      name="marked_in_sequence_map"), multi_branch],
        "combo_sequence_map_body", [_vi("sm_elem")], [_vi("sm_out")])
    # SequenceMap maps over a sequence while Scan carries tensors, so the Scan body lifts its state var
    # into a one-element sequence and concatenates the mapped sequence back into one tensor.
    scan_body = onnx.helper.make_graph(
        [_marked_node("PyptoComboInScan", ["sc_state"], ["sc_marked"], name="marked_in_scan"),
         onnx.helper.make_node("SequenceConstruct", ["sc_marked"], ["sc_seq"], name="to_sequence"),
         onnx.helper.make_node("SequenceMap", ["sc_seq"], ["sc_mapped"], name="the_sequence_map",
                               body=sequence_map_body),
         onnx.helper.make_node("ConcatFromSequence", ["sc_mapped"], ["sc_state_out"],
                               name="from_sequence", axis=0)],
        "combo_scan_body", [_vi("sc_state"), _vi("sc_elem")], [_vi("sc_state_out")])
    # The Scan's scan input and the Loop's trip count and condition are root-graph inputs: a nested
    # graph may read an outer-scope name, which keeps the chain itself free of filler operands.
    loop_body = _loop_body(
        [_marked_node("PyptoComboInLoop", ["v_combo"], ["lp_marked"], name="marked_in_loop"),
         onnx.helper.make_node("Scan", ["lp_marked", "seq"], ["v_out_combo"], name="the_scan",
                               num_scan_inputs=1, body=scan_body)],
        "_combo")
    then_branch = onnx.helper.make_graph(
        [_marked_node("PyptoComboInIf", ["x"], ["if_marked"], name="marked_in_if"),
         onnx.helper.make_node("Loop", ["trip", "cond", "if_marked"], ["if_out"], name="the_loop",
                               body=loop_body)],
        "combo_then_g", [], [_vi("if_out")])
    model = _nested_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("If", ["cond"], ["y"], name="the_if",
                                   then_branch=then_branch, else_branch=_plain_branch())],
            "combo_nested",
            [_vi("x"), onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
             onnx.helper.make_tensor_value_info("trip", onnx.TensorProto.INT64, []), _vi("seq", (1, 1))],
            [_vi("y")]))
    expected = [
        "PyptoComboInIf",             # If.then_branch, depth 1
        "PyptoComboInLoop",           # Loop.body, depth 2
        "PyptoComboInScan",           # Scan.body, depth 3
        "PyptoComboInSequenceMap",    # SequenceMap.body, depth 4
        "PyptoComboInGraphs",         # a GRAPHS-valued attribute, depth 5
    ]
    # Every marker proved findable at the root graph first, so a level missing below is the walk and
    # not the fixture — the pairing every nested case in this file uses.
    for op_type in expected:
        assert _found_op_types(_root_control(op_type)) == [op_type], f"root control lost {op_type}"
    assert _found_op_types(model) == expected


def test_iter_pypto_nodes_skips_an_unmarked_node_in_a_subgraph():
    # Negative case: descending into a subgraph must not relax the marker gate.
    plain = onnx.helper.make_node("Relu", ["x"], ["b_out"], name="unmarked_in_branch")
    branch = onnx.helper.make_graph([plain], "then_g", [], [_vi("b_out")])
    node = onnx.helper.make_node("If", ["cond"], ["y"], name="the_if",
                                 then_branch=branch, else_branch=_plain_branch())
    model = _nested_model(onnx.helper.make_graph(
        [node], "neg_unmarked",
        [_vi("x"), onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])], [_vi("y")]))
    assert _found_op_types(model) == []


def test_iter_pypto_nodes_skips_a_foreign_op_type_attr_in_a_subgraph():
    # Negative case: another framework's `op_type` attribute without pypto's marker is not pypto, nested
    # or not — the same discrimination the root-level case already pins, now one graph down.
    # In another framework's own imported domain, so the checker does not demand an ai.onnx schema for it.
    foreign = onnx.helper.make_node("SomeOtherOp", ["x"], ["b_out"],
                                    domain="other.framework", name="foreign_in_branch")
    foreign.attribute.append(onnx.helper.make_attribute(_META_KEY__OP_TYPE, "SomeOtherOp"))
    branch = onnx.helper.make_graph([foreign], "then_g", [], [_vi("b_out")])
    node = onnx.helper.make_node("If", ["cond"], ["y"], name="the_if",
                                 then_branch=branch, else_branch=_plain_branch())
    model = _nested_model(
        onnx.helper.make_graph(
            [node], "neg_foreign",
            [_vi("x"), onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])], [_vi("y")]),
        extra_opsets=[("other.framework", 1)])
    assert _found_op_types(model) == []


@pytest.mark.parametrize(
    ("op_type", "kwargs"),
    [
        ("PyptoMalformedBlankString", {"layout": "   "}),
        ("PyptoMalformedNonString", {"layout": 2}),
    ],
)
def test_iter_pypto_nodes_skips_a_malformed_marker_in_a_subgraph(op_type, kwargs):
    # Negative case: the existing marker gate (blank value / non-STRING attribute) still applies inside
    # a subgraph — the walk reaches further, it does not lower the bar. Paired with the positive control
    # so the zero cannot be a blind walk.
    assert _found_op_types(_root_control(op_type)) == [op_type], "positive control at root must be found"
    marked = onnx.helper.make_graph(
        [_marked_node(op_type, ["x"], ["b_out"], name="malformed_in_branch", **kwargs)],
        "then_g", [], [_vi("b_out")])
    node = onnx.helper.make_node("If", ["cond"], ["y"], name="the_if",
                                 then_branch=marked, else_branch=_plain_branch())
    model = _nested_model(onnx.helper.make_graph(
        [node], "neg_malformed",
        [_vi("x"), onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])], [_vi("y")]))
    assert _found_op_types(model) == []


def test_iter_pypto_nodes_descends_in_proto_attribute_order():
    # Proto attribute order is not the producer's source order: ``make_node`` sorts kwargs and stores
    # ``else_branch`` first, while torch.onnx writes ``then_branch`` first. So the attributes are appended
    # explicitly here and only proto order is asserted, leaving which branch wins per exporter unpinned.
    first = onnx.helper.make_graph(
        [_marked_node("PyptoFirstAttr", ["x"], ["b_out"], name="marked_first")],
        "first_g", [], [_vi("b_out")])
    second = onnx.helper.make_graph(
        [_marked_node("PyptoSecondAttr", ["x"], ["b_out"], name="marked_second")],
        "second_g", [], [_vi("b_out")])
    node = onnx.helper.make_node("If", ["cond"], ["y"], name="the_if")
    node.attribute.append(onnx.helper.make_attribute("then_branch", first))
    node.attribute.append(onnx.helper.make_attribute("else_branch", second))
    assert [a.name for a in node.attribute] == ["then_branch", "else_branch"], "fixture order precondition"
    model = _nested_model(onnx.helper.make_graph(
        [node], "attr_order",
        [_vi("x"), onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])], [_vi("y")]))
    assert _found_op_types(model) == ["PyptoFirstAttr", "PyptoSecondAttr"]
