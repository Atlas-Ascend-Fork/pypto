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
"""Export-stage guard: declare a :class:`ExportedCustomOp`, ``torch.onnx.export`` it, reload it and check the node.

Needs only torch + onnx: no CANN, no device, no compiler, and the kernel itself is never built.

Checks the reloaded node's ``op_type``/``domain``, arity and tensor element types, and every declared
:class:`AttrSpec`'s value against its ``onnx.AttributeProto`` type -- an Int arrives as INT (never FLOAT)
and a whole Float such as ``2.0`` arrives as FLOAT (never INT). Also checks that
``torch.ops.pypto.<op>(...)`` equals ``torch_defn(...)`` element-wise on CPU.
"""
import onnx
import pytest
import torch
import torch.nn as nn

import pypto
from pypto.extensions.torch_custom_op_litenpu import AttrSpec, ExportedCustomOp, OnnxSymbolicSpec, finalize_pending_ops
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _reset_export_state
from pypto.extensions.torch_custom_op_litenpu.common.torch_op import exporting_scope
from pypto.extensions.torch_custom_op_litenpu.onnx.export import _qualify_op_type, _set_symbolic_output_type

# The declaration under test. Two attrs of two different types, in declaration order: an Int and a Float
# whose value is a whole number (2.0) — the case that must not degrade to an ONNX INT attribute.
_QUALNAME = "pypto::scale_bias"
_SPEC = OnnxSymbolicSpec(op_type="ScaleBias", opset_version=12)
_ATTRS = (AttrSpec("bias", "Int", 0), AttrSpec("scale", "Float", 1.0))
_BIAS_VALUE = 3
_SCALE_VALUE = 2.0
_SHAPE = (1, 8, 1, 64)
_DTYPE = torch.float16
_ONNX_ELEM_TYPE = onnx.TensorProto.FLOAT16


# ── authoring functions (module scope: the tracer reads their source off disk) ────────────────────────
def _scale_bias_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    bias = int(attrs["bias"])       # attr values reach the factory as strings
    scale = float(attrs["scale"])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def scale_bias_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                             out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move((a + b) * scale + bias)

    return scale_bias_inner


def _scale_bias_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _scale_bias_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _scale_bias_torch(a, b, bias, scale):
    return (a + b) * scale + bias


class _ScaleBiasModel(nn.Module):
    def forward(self, a, b):
        return torch.ops.pypto.scale_bias(a, b, bias=_BIAS_VALUE, scale=_SCALE_VALUE)


@pytest.fixture()
def declared_op():
    """Declare the op, finalize its synthesized ONNX symbolic, and clear the export state around it."""
    _reset_export_state()
    op = ExportedCustomOp(
        kernel=_scale_bias_factory,
        infer_shape=_scale_bias_infer_shape,
        infer_dtype=_scale_bias_infer_dtype,
        torch_defn=_scale_bias_torch,
        torch_op_qualname=_QUALNAME,
        onnx_spec=_SPEC,
        attrs=list(_ATTRS),
    )
    finalize_pending_ops()
    yield op
    _reset_export_state()


@pytest.fixture()
def exported_model(declared_op, tmp_path):
    """The reloaded ``onnx.ModelProto`` produced by a real ``torch.onnx.export`` of the declared op."""
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    path = str(tmp_path / "scale_bias.onnx")
    with exporting_scope():
        torch.onnx.export(
            _ScaleBiasModel(), (a, b), path,
            input_names=["a", "b"], output_names=["y"],
            opset_version=_SPEC.opset_version, do_constant_folding=False, dynamo=False,
            custom_opsets={_SPEC.domain: _SPEC.domain_opset_version},
        )
    return onnx.load(path)


def _pypto_node(model):
    """The single node the declaration produced (matched on the qualified op_type)."""
    op_type = _qualify_op_type(_SPEC.op_type)
    nodes = [n for n in model.graph.node if n.op_type == op_type]
    assert len(nodes) == 1, (
        f"expected exactly one {op_type} node, got {[(n.op_type, n.domain) for n in model.graph.node]}"
    )
    return nodes[0]


def test_node_op_type_and_domain(exported_model):
    # The graph node carries the author's op name under the PyptoCustomOp namespace, in the spec's domain
    # — the <domain>::<domain_opset_version>::<op_type> triple GE looks the executor up by.
    node = _pypto_node(exported_model)
    assert node.op_type == "PyptoCustomOpScaleBias"
    assert node.op_type == _qualify_op_type(_SPEC.op_type)
    assert node.domain == _SPEC.domain == "pypto"
    opset = {o.domain: o.version for o in exported_model.opset_import}
    assert opset["pypto"] == 1
    assert opset[_SPEC.domain] == _SPEC.domain_opset_version


def test_node_and_graph_arity(exported_model):
    # Two tensor inputs (the attrs are node ATTRIBUTES, not operands) and the one output infer_shape
    # declares, on the node and on the graph alike.
    node = _pypto_node(exported_model)
    assert len(node.input) == 2
    assert len(node.output) == 1
    graph = exported_model.graph
    assert [i.name for i in graph.input] == ["a", "b"]
    assert [o.name for o in graph.output] == ["y"]
    assert list(node.input) == [i.name for i in graph.input]
    assert list(node.output) == [o.name for o in graph.output]


def test_graph_declared_tensor_dtypes(exported_model):
    # The graph's declared element types match the exported tensors and infer_dtype's promise.
    graph = exported_model.graph
    assert len(graph.input) == 2      # the loop below says nothing about an empty input list
    for value_info in graph.input:
        assert value_info.type.tensor_type.elem_type == _ONNX_ELEM_TYPE
    assert [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim] == list(_SHAPE)
    # Element type only here; the dims the symbolic now annotates are asserted by
    # test_symbolic_annotates_the_node_output_elem_type_and_dims below.
    assert graph.output[0].type.tensor_type.elem_type == _ONNX_ELEM_TYPE


def test_declared_attrs_present_with_value_and_onnx_type(exported_model):
    # Every declared AttrSpec reaches the node, and the ONNX AttributeProto TYPE matches the declared
    # kind: an Int must not arrive as a FLOAT, and a whole-number Float (2.0) must not arrive as an INT.
    node = _pypto_node(exported_model)
    by_name = {at.name: at for at in node.attribute}
    assert set(s.name for s in _ATTRS) <= set(by_name), (
        f"declared attrs missing from the node; node attrs = {sorted(by_name)}"
    )

    bias = by_name["bias"]
    assert bias.type == onnx.AttributeProto.INT
    assert bias.type != onnx.AttributeProto.FLOAT
    assert bias.i == _BIAS_VALUE

    scale = by_name["scale"]
    assert scale.type == onnx.AttributeProto.FLOAT
    assert scale.type != onnx.AttributeProto.INT   # 2.0 is a whole number and must still be a FLOAT attr
    assert scale.f == pytest.approx(_SCALE_VALUE, abs=1e-6)


def test_torch_op_matches_torch_defn_on_cpu(declared_op):
    # With no dispatch override active, the op's implementation IS the torch reference: the registered
    # torch.ops.pypto.<op> and torch_defn must agree element-wise on the same operands.
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    out = torch.ops.pypto.scale_bias(a, b, bias=_BIAS_VALUE, scale=_SCALE_VALUE)
    golden = _scale_bias_torch(a, b, _BIAS_VALUE, _SCALE_VALUE)
    assert out.shape == golden.shape == torch.Size(_SHAPE)
    assert out.dtype == golden.dtype == _DTYPE
    # Identical arithmetic on identical operands: agreement is exact, so no tolerance is allowed to hide
    # a divergence between the registered op and the declared reference.
    torch.testing.assert_close(out, golden, rtol=0, atol=0)
    assert declared_op._torch_defn_fn is _scale_bias_torch


@pytest.mark.parametrize("short,qualified", [("Add", "PyptoCustomOpAdd"), ("Softmax", "PyptoCustomOpSoftmax")])
def test_qualify_prefixes_a_short_name(short, qualified):
    assert _qualify_op_type(short) == qualified


def test_qualify_is_idempotent():
    once = _qualify_op_type("Add")
    assert _qualify_op_type(once) == once


# ── the synthesized symbolic's output type ────────────────────────────────────────────────────────────
# An untyped output is masked at the root graph, but in an ``If`` branch it becomes the subgraph's and
# then the model's output, where onnx.checker rejects a missing ``elem_type`` and nothing is saved.


class _NestedScaleBiasModel(nn.Module):
    """The op behind one data-dependent conditional. Must be scripted: tracing folds the branch away.

    The attrs are ``torch.jit.Final`` members rather than the module-level constants the traced model
    above closes over: TorchScript refuses a closed-over global ("python value of type 'int' cannot be
    used as a value"), so this is how the same two values reach a scripted forward.
    """

    bias: torch.jit.Final[int]
    scale: torch.jit.Final[float]

    def __init__(self):
        super().__init__()
        self.bias = _BIAS_VALUE
        self.scale = _SCALE_VALUE

    def forward(self, a, b, cond):
        if bool(cond):
            return torch.ops.pypto.scale_bias(a, b, bias=self.bias, scale=self.scale)
        return a * 2.0


def test_symbolic_annotates_the_node_output_elem_type_and_dims(exported_model):
    # Fully typed: element type and the dims infer_shape promises. At the ROOT graph torch propagates
    # dims too, so the load-bearing case is the nested one below.
    node = _pypto_node(exported_model)
    value_infos = {vi.name: vi for vi in list(exported_model.graph.value_info) + list(exported_model.graph.output)}
    typed = value_infos[node.output[0]]
    assert typed.type.WhichOneof("value") == "tensor_type", "output carries no tensor_type at all"
    assert typed.type.tensor_type.elem_type == _ONNX_ELEM_TYPE
    assert [d.dim_value for d in typed.type.tensor_type.shape.dim] == list(_SHAPE)


def _export_nested(model, args, spec, path, output_names, **export_kwargs):
    """``torch.onnx.export`` *model* and reload it, asserting the op really did end up nested.

    The branch must have survived scripting: a folded conditional would put the node at the root graph,
    where an untyped output is masked and the checker would pass for the wrong reason.
    """
    with exporting_scope():
        torch.onnx.export(
            torch.jit.script(model), args, path, output_names=output_names,
            opset_version=spec.opset_version, do_constant_folding=False, dynamo=False,
            custom_opsets={spec.domain: spec.domain_opset_version}, **export_kwargs,
        )
    loaded = onnx.load(path)
    assert any(n.op_type == "If" for n in loaded.graph.node), "scripting folded the conditional away"
    assert not [n for n in loaded.graph.node if n.op_type == _qualify_op_type(spec.op_type)], \
        "the pypto node is at the root graph, so this is not the nested case"
    return loaded


def _then_branch(model):
    """The ``then_branch`` graph of the model's single root ``If``, where the pypto node sits."""
    if_nodes = [n for n in model.graph.node if n.op_type == "If"]
    assert len(if_nodes) == 1, f"expected one root If, got {len(if_nodes)}"
    branches = [at.g for at in if_nodes[0].attribute if at.name == "then_branch"]
    assert len(branches) == 1, "the If carries no then_branch"
    return branches[0]


def test_nested_export_passes_the_onnx_checker(declared_op, tmp_path):
    # The whole of defect 2: a scripted export with the op inside an If branch must validate, which is
    # the gate that decides whether the export is finalized and saved at all.
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    cond = torch.tensor(True)
    model = _export_nested(_NestedScaleBiasModel(), (a, b, cond), _SPEC,
                           str(tmp_path / "nested_scale_bias.onnx"), ["y"])
    onnx.checker.check_model(model)


# ── the two op shapes the output annotation used to skip ───────────────────────────────────────────────
# A multi-output op, and an op whose ``infer_shape`` reads a declared attr. ``bias`` is declared first
# and affects no shape, so resolving ``keep`` by position would feed infer_shape the wrong value.
_SWAP_SPEC = OnnxSymbolicSpec(op_type="SwapPair", opset_version=12)
_CROP_SPEC = OnnxSymbolicSpec(op_type="CropLast", opset_version=12)
_CROP_KEEP = 32
_CROP_BIAS = 3
_CROP_ATTRS = (AttrSpec("bias", "Int", 0), AttrSpec("keep", "Int", _CROP_KEEP))


def _swap_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def swap_inner(a: pypto.Tensor([...], dtypes[0]), b: pypto.Tensor([...], dtypes[1]),
                   out0: pypto.Tensor([...], dtypes[1]), out1: pypto.Tensor([...], dtypes[0])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out0.move(b + b - b)
        out1.move(a + a - a)

    return swap_inner


def _swap_infer_shape(a: torch.Size, b: torch.Size) -> tuple[torch.Size, torch.Size]:
    return (b, a)


def _swap_infer_dtype(a: torch.dtype, b: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    return (b, a)


def _swap_torch(a, b):
    return b.clone(), a.clone()


def _crop_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    keep = int(attrs["keep"])
    bias = int(attrs["bias"])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def crop_inner(x: pypto.Tensor([...], dtypes[0]), out: pypto.Tensor([...], dtypes[0])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(x[:, :, :, :keep] + bias)

    return crop_inner


def _crop_infer_shape(x: torch.Size, keep: int) -> torch.Size:
    return torch.Size([x[0], x[1], x[2], keep])


def _crop_infer_dtype(x: torch.dtype) -> torch.dtype:
    return x


def _crop_torch(x, bias, keep):
    return x[:, :, :, :keep] + bias


class _NestedSwapModel(nn.Module):
    def forward(self, a, b, cond):
        if bool(cond):
            return torch.ops.pypto.swap_pair(a, b)
        return b.clone(), a.clone()


class _NestedCropModel(nn.Module):
    """``torch.jit.Final`` members for the same reason :class:`_NestedScaleBiasModel` uses them."""

    bias: torch.jit.Final[int]
    keep: torch.jit.Final[int]

    def __init__(self):
        super().__init__()
        self.bias = _CROP_BIAS
        self.keep = _CROP_KEEP

    def forward(self, x, cond):
        if bool(cond):
            return torch.ops.pypto.crop_last(x, bias=self.bias, keep=self.keep)
        return x[:, :, :, :self.keep] + self.bias


@pytest.fixture()
def swap_op():
    """A two-output op, declared and finalized with the export state cleared around it."""
    _reset_export_state()
    op = ExportedCustomOp(
        kernel=_swap_factory, infer_shape=_swap_infer_shape, infer_dtype=_swap_infer_dtype,
        torch_defn=_swap_torch, torch_op_qualname="pypto::swap_pair", onnx_spec=_SWAP_SPEC,
    )
    finalize_pending_ops()
    yield op
    _reset_export_state()


@pytest.fixture()
def crop_op():
    """An op whose ``infer_shape`` reads one of its two declared attrs."""
    _reset_export_state()
    op = ExportedCustomOp(
        kernel=_crop_factory, infer_shape=_crop_infer_shape, infer_dtype=_crop_infer_dtype,
        torch_defn=_crop_torch, torch_op_qualname="pypto::crop_last", onnx_spec=_CROP_SPEC,
        attrs=list(_CROP_ATTRS),
    )
    finalize_pending_ops()
    yield op
    _reset_export_state()


def test_multi_output_nested_export_passes_the_onnx_checker(swap_op, tmp_path):
    # Both outputs must be annotated: one untyped output of the pair is enough to fail the checker.
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    model = _export_nested(_NestedSwapModel(), (a, b, torch.tensor(True)), _SWAP_SPEC,
                           str(tmp_path / "nested_swap.onnx"), ["o0", "o1"])
    onnx.checker.check_model(model)
    nodes = [n for n in _then_branch(model).node if n.op_type == _qualify_op_type(_SWAP_SPEC.op_type)]
    assert len(nodes) == 1
    assert len(nodes[0].output) == 2, "the node must carry one output per infer_shape tuple entry"


def test_multi_output_annotation_is_per_output(swap_op, tmp_path):
    # Distinct input dtypes AND shapes, both swapped by the infer funcs: out0 takes input 1's and out1
    # takes input 0's. ``setType`` templates every output on input 0's type, so a dtype or a shape
    # carried over from the template instead of per output would give input 0's twice here.
    other_shape = (1, 4, 1, 32)
    a = torch.rand(_SHAPE, dtype=torch.float32)
    b = torch.rand(other_shape, dtype=torch.float16)
    model = _export_nested(_NestedSwapModel(), (a, b, torch.tensor(True)), _SWAP_SPEC,
                           str(tmp_path / "nested_swap_mixed.onnx"), ["o0", "o1"])
    onnx.checker.check_model(model)
    outputs = _then_branch(model).output
    assert [vi.type.tensor_type.elem_type for vi in outputs] == [onnx.TensorProto.FLOAT16,
                                                                 onnx.TensorProto.FLOAT]
    assert [tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)
            for vi in outputs] == [other_shape, _SHAPE]


def test_shape_attr_annotation_resolves_the_attr_by_name(crop_op, tmp_path):
    # The annotated dims are infer_shape's attr-dependent answer: the last dim is ``keep``, neither the
    # input's own last dim nor ``bias`` -- the attr that is declared first but affects no shape.
    x = torch.rand(_SHAPE, dtype=_DTYPE)
    model = _export_nested(_NestedCropModel(), (x, torch.tensor(True)), _CROP_SPEC,
                           str(tmp_path / "nested_crop_dims.onnx"), ["y"])
    onnx.checker.check_model(model)
    assert len(_then_branch(model).output) == 1
    typed = _then_branch(model).output[0].type.tensor_type
    assert typed.elem_type == _ONNX_ELEM_TYPE
    assert [d.dim_value for d in typed.shape.dim] == [_SHAPE[0], _SHAPE[1], _SHAPE[2], _CROP_KEEP]
    assert _CROP_KEEP != _SHAPE[-1] != _CROP_BIAS, "the constants must make the three cases tell apart"


# ── the annotation's two defensive exits ───────────────────────────────────────────────────────────────
class _RecordingOutput:
    """Stands in for the traced output value: records whether the annotation set a type on it."""

    def __init__(self):
        self.set_types = []

    def setType(self, tensor_type):  # noqa: N802 - mirrors torch._C.Value.setType
        self.set_types.append(tensor_type)


class _UntypedInput:
    """An operand whose type is not a tensor type, which is the AttributeError the handler documents."""

    def type(self):
        return object()         # no .dtype(): exactly what a non-tensor traced operand looks like


class _UnannotatableOp:
    _torch_op_qualname = "pypto::unannotatable"
    _attr_specs = ()

    @staticmethod
    def _infer_dtype_fn(dtype):
        raise AssertionError("must not be reached: the input type fails first")

    @staticmethod
    def _infer_shape_fn(shape) -> torch.Size:
        raise AssertionError("must not be reached: the input type fails first")


def test_annotation_warns_instead_of_raising_when_the_operand_has_no_tensor_type(caplog):
    # An export must not die because one op could not be annotated: at the root graph the result is
    # still valid, so the failure is a warning naming the op and the nesting consequence.
    out = _RecordingOutput()
    with caplog.at_level("WARNING"):
        _set_symbolic_output_type(_UnannotatableOp(), [_UntypedInput()], (), out, 1)
    assert out.set_types == [], "no type may be set when the infer funcs could not be called"
    assert len(caplog.records) == 1
    assert "pypto::unannotatable" in caplog.records[0].getMessage()
    assert "AttributeError" in caplog.records[0].getMessage()


def test_annotation_is_a_no_op_for_an_op_with_no_tensor_inputs(caplog):
    # With no operand there is no type to read, so the annotation declines silently rather than warning.
    out = _RecordingOutput()
    with caplog.at_level("WARNING"):
        _set_symbolic_output_type(_UnannotatableOp(), [], (), out, 1)
    assert out.set_types == []
    assert caplog.records == []


# ── the topology that decides whether the output SHAPE is load-bearing ─────────────────────────────────
# Torch merges the If branch types into the model output's, so one standard-op branch supplies a shape
# and masks a pypto node declaring none. All-pypto branches are what make infer_shape load-bearing.


class _PyptoOnlyBranchModel(nn.Module):
    """Both branches are the pypto op, with the operands swapped, so no standard op carries a shape."""

    bias: torch.jit.Final[int]
    scale: torch.jit.Final[float]

    def __init__(self):
        super().__init__()
        self.bias = _BIAS_VALUE
        self.scale = _SCALE_VALUE

    def forward(self, a, b, cond):
        if bool(cond):
            return torch.ops.pypto.scale_bias(a, b, bias=self.bias, scale=self.scale)
        return torch.ops.pypto.scale_bias(b, a, bias=self.bias, scale=self.scale)


def test_every_branch_of_a_pypto_only_conditional_is_typed(declared_op, tmp_path):
    # With no standard op anywhere on the path, the model output's shape can only come from the
    # annotation. Dropping infer_shape's answer fails this with "Field 'shape' of 'type' is required
    # but missing", while the mixed-branch models above keep passing.
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    model = _export_nested(_PyptoOnlyBranchModel(), (a, b, torch.tensor(True)), _SPEC,
                           str(tmp_path / "pypto_only_branches.onnx"), ["y"])
    onnx.checker.check_model(model)
    out_type = model.graph.output[0].type.tensor_type
    assert out_type.HasField("shape"), "the model output must carry a shape, not just an element type"
    assert [d.dim_value for d in out_type.shape.dim] == list(_SHAPE)
    nodes = [n for n in _then_branch(model).node if n.op_type == _qualify_op_type(_SPEC.op_type)]
    assert len(nodes) == 1


# -- a dynamic_axes export: the one case infer_shape cannot answer ------------------------------------
# An unknown operand dimension leaves the output shape uncomputable, so the element type is declared,
# the shape is not, and a warning names dynamic_axes, which nothing else reports.


def test_a_dynamic_axis_warns_and_declares_no_shape(declared_op, tmp_path, caplog):
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    with caplog.at_level("WARNING"):
        model = _export_nested(_NestedScaleBiasModel(), (a, b, torch.tensor(True)), _SPEC,
                               str(tmp_path / "nested_dynamic.onnx"), ["y"],
                               input_names=["a", "b", "cond"],
                               dynamic_axes={"a": {0: "batch"}, "b": {0: "batch"}, "y": {0: "batch"}})
    warned = [r.getMessage() for r in caplog.records if "dynamic_axes" in r.getMessage()]
    assert len(warned) == 1, f"expected one dynamic_axes warning, got {warned}"
    assert _QUALNAME in warned[0], "the warning must name the op whose output went undeclared"
    onnx.checker.check_model(model)
    typed = _then_branch(model).output[0].type.tensor_type
    assert typed.elem_type == _ONNX_ELEM_TYPE, "the element type is still declared"
    assert not typed.shape.dim, "no dimension may be declared from an unknown operand shape"


def test_a_static_export_of_the_same_model_does_not_warn(declared_op, tmp_path, caplog):
    # The control: the same model and op with dynamic_axes dropped. The declared dims below are what
    # ran, so a silently skipped annotation cannot pass this as an absent warning.
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    with caplog.at_level("WARNING"):
        model = _export_nested(_NestedScaleBiasModel(), (a, b, torch.tensor(True)), _SPEC,
                               str(tmp_path / "nested_static.onnx"), ["y"],
                               input_names=["a", "b", "cond"])
    warned = [r.getMessage() for r in caplog.records if "dynamic_axes" in r.getMessage()]
    assert not warned, f"a static export must not warn, got {warned}"
    dims = _then_branch(model).output[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == list(_SHAPE), "the static export declares every dim"
