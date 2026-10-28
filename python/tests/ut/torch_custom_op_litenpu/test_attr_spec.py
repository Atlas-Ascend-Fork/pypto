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
"""Tests for the operator-attribute pipeline (first slice — a single Int attr, the trailing-operand attr
model).

Covers: ``ExportedCustomOp(attrs=[...])`` storing the ordered list and validating entries; the ``"attrs"``
field round-tripping through the node's
``op_export_record`` JSON (present only when non-empty, so an attr-free op stays byte-identical); the
torch_defn arity (n_inputs + n_attrs); and the typed ONNX ``AttributeProto`` emitted after a real
``torch.onnx.export`` (``bias`` -> type INT(2), value == the call operand).
"""
import json

import postponed_annot_samples as postponed  # sibling fixture module (pytest puts this dir on sys.path)
import pytest
import torch
import torch.nn as nn

import pypto
from pypto.extensions.torch_custom_op_litenpu import AttrSpec, ExportedCustomOp, OnnxSymbolicSpec, finalize_pending_ops
from pypto.extensions.torch_custom_op_litenpu.common.exported_custom_op import (
    _build_node_meta_completer,
    exporting_scope,
)
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _PENDING_OPS, _reset_export_state
from pypto.extensions.torch_custom_op_litenpu.common.node_meta import _META_KEY__OP_EXPORT_RECORD
from pypto.extensions.torch_custom_op_litenpu.onnx.export import _onnx_encode_node_meta


# ── module-scope authoring fns (kernel_snippet inspects their source, so they must be importable) ──
def _bias_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    bias = int(attrs["bias"])  # attrs arrive as strings

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def add_bias_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype), out: pypto.Tensor([...])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b + bias)
    return add_bias_inner


def _plain_factory(shapes, dtypes, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def add_plain_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype), out: pypto.Tensor([...])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b)
    return add_plain_inner


def _bias_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _bias_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _bias_torch(a, b, bias):
    return a + b + bias


def _plain_torch(a, b):
    return a + b


def _two_attr_torch(a, b, bias, other):
    return a + b + bias + other


# ── typed-attr factories (single tensor input) for the per-type ONNX + shape-affecting tests ──
def _alpha_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    alpha = float(attrs["alpha"])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def alpha_inner(x: pypto.Tensor([...], dtype), out: pypto.Tensor([...])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(x * alpha)
    return alpha_inner


def _mode_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    mode = attrs["mode"]

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def mode_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype), out: pypto.Tensor([...])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b if mode == "sum" else a - b)
    return mode_inner


def _crop_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    import json as _json
    dtype = dtypes[0]
    shape = shapes[0]
    h, w = _json.loads(attrs["crop_size"])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def crop_inner(x: pypto.Tensor([...], dtype), out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(shape[0], shape[1], h, w)
        out[:] = x[:, :, :h, :w] * 2.0  # step-1 slice (supported) + real arithmetic
    return crop_inner


def _single_infer_shape(x: torch.Size) -> torch.Size:
    return x


def _single_infer_dtype(x: torch.dtype) -> torch.dtype:
    return x


def _alpha_torch(x, alpha):
    return x * alpha


def _mode_torch(a, b, mode):
    return a + b if mode == "sum" else a - b


def _crop_infer_shape(x: torch.Size, crop_size: list[int]) -> torch.Size:
    return torch.Size([x[0], x[1], crop_size[0], crop_size[1]])


def _crop_torch(x, crop_size):
    return x[:, :, :crop_size[0], :crop_size[1]] * 2.0


# --------------------------------------------------------------------------- ExportedCustomOp storage

def test_customop_stores_ordered_attr_specs():
    op = ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                          torch_defn=_bias_torch, torch_op_qualname="pypto::attr_store1",
                          attrs=[AttrSpec("bias", "Int", 0)])
    assert op._attr_specs == (AttrSpec("bias", "Int", 0),)


def test_customop_defaults_to_no_attrs():
    op = ExportedCustomOp(kernel=_plain_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                          torch_defn=_plain_torch, torch_op_qualname="pypto::attr_store2")
    assert op._attr_specs == ()


def test_customop_rejects_non_attrspec_entries():
    with pytest.raises(TypeError, match="AttrSpec"):
        ExportedCustomOp(kernel=_plain_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                         torch_defn=_plain_torch, torch_op_qualname="pypto::attr_store3",
                         attrs=[{"name": "bias", "type": "Int"}])


def test_customop_rejects_duplicate_attr_names():
    # AttrSpec validates one spec at a time and cannot see the repeat, and the torch schema parser accepts
    # a duplicated parameter, so this is the only place it can be caught. It has to be: the ONNX symbolic
    # keys its attribute kwargs by name, so the second spec would silently overwrite the first and the
    # exported node would carry fewer attributes than the op declares -- wrong values, no error anywhere.
    with pytest.raises(ValueError, match="duplicate"):
        ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                         torch_defn=_bias_torch, torch_op_qualname="pypto::attr_dupe1",
                         attrs=[AttrSpec("bias", "Int", 0), AttrSpec("bias", "Int", 1)])
    # Rejected at construction, so nothing was registered: the op never reached the pending-op list and
    # finalize has no torch schema to build for it.
    assert not any(getattr(op, "_torch_op_qualname", None) == "pypto::attr_dupe1" for op in _PENDING_OPS)
    # Two attrs differing only in name are fine -- it is the collision that is rejected, not the repetition
    # of a type.
    op = ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                          torch_defn=_two_attr_torch, torch_op_qualname="pypto::attr_dupe2",
                          attrs=[AttrSpec("bias", "Int", 0), AttrSpec("other", "Int", 1)])
    assert [s.name for s in op._attr_specs] == ["bias", "other"]


def test_torch_defn_arity_counts_inputs_plus_attrs():
    # torch_defn takes n_inputs + n_attrs (add_bias_torch(a, b, bias) with a 2-input infer_shape).
    ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                     torch_defn=_bias_torch, torch_op_qualname="pypto::attr_arity_ok",
                     attrs=[AttrSpec("bias", "Int", 0)])
    # A torch_defn missing the attr operand is a clear arity error.
    with pytest.raises(ValueError, match="attrs"):
        ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                         torch_defn=_plain_torch, torch_op_qualname="pypto::attr_arity_bad",
                         attrs=[AttrSpec("bias", "Int", 0)])


# --------------------------------------------------------------------------- op_export_record round-trip

def _drive_export_meta(op):
    """Run the op's kernel-export closure with a stub dtype extractor (no ONNX) so it populates
    ``op._node_meta[op_export_record]``, then return the parsed record dict."""
    def _stub_dtypes(*nodes):
        return [torch.float16, torch.float16]

    export_fn = _build_node_meta_completer(
        op=op, encode_node_meta=_onnx_encode_node_meta, input_dtypes=_stub_dtypes, framework_kind="onnx",
    )
    export_fn(object(), object(), op_type="PyptoCustomOpX", domain="pypto", domain_opset_version=1)
    return json.loads(op._node_meta[_META_KEY__OP_EXPORT_RECORD])


def test_op_export_record_carries_attrs():
    op = ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                          torch_defn=_bias_torch, torch_op_qualname="pypto::attr_params1",
                          attrs=[AttrSpec("bias", "Int", 0)])
    record = _drive_export_meta(op)
    assert record["attrs"] == [{"name": "bias", "type": "Int", "default": 0}]
    assert record["factory_signature"] == "full"  # attrs-aware factory


def test_op_export_record_omits_attrs_when_none():
    op = ExportedCustomOp(kernel=_plain_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                          torch_defn=_plain_torch, torch_op_qualname="pypto::attr_params2")
    record = _drive_export_meta(op)
    assert "attrs" not in record  # attr-free op => no "attrs" key (byte-identical to a pre-attrs op)


# --------------------------------------------------------------------------- ONNX AttributeProto after export

def test_onnx_export_emits_int_attr_on_node(tmp_path):
    import onnx
    _reset_export_state()
    _op = ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                           torch_defn=_bias_torch, torch_op_qualname="pypto::attr_onnx1",
                           onnx_spec=OnnxSymbolicSpec(op_type="AddBiasOnnx", opset_version=12),
                           attrs=[AttrSpec("bias", "Int", 7)])
    finalize_pending_ops()

    class M(nn.Module):
        def forward(self, a, b):
            return torch.ops.pypto.attr_onnx1(a, b, bias=7)

    a = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    b = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    path = str(tmp_path / "add_bias.onnx")
    with exporting_scope():
        torch.onnx.export(M(), (a, b), path, input_names=["a", "b"], output_names=["y"],
                          opset_version=12, do_constant_folding=False, dynamo=False,
                          custom_opsets={"pypto": 1})
    model = onnx.load(path)
    nodes = [n for n in model.graph.node if "AddBiasOnnx" in n.op_type]
    assert nodes, f"no AddBiasOnnx node found; ops={[n.op_type for n in model.graph.node]}"
    bias_attr = next((at for at in nodes[0].attribute if at.name == "bias"), None)
    assert bias_attr is not None, f"no bias attr; attrs={[at.name for at in nodes[0].attribute]}"
    assert bias_attr.type == onnx.AttributeProto.INT  # type code 2
    assert bias_attr.i == 7
    _reset_export_state()


# --------------------------------------------------------------------------- per-type ONNX AttributeProto

def test_onnx_export_emits_float_attr(tmp_path):
    import onnx
    _reset_export_state()
    ExportedCustomOp(kernel=_alpha_factory, infer_shape=_single_infer_shape, infer_dtype=_single_infer_dtype,
                     torch_defn=_alpha_torch, torch_op_qualname="pypto::attr_onnx_f",
                     onnx_spec=OnnxSymbolicSpec(op_type="AlphaOnnx", opset_version=12),
                     attrs=[AttrSpec("alpha", "Float", 0.01)])
    finalize_pending_ops()

    class M(nn.Module):
        def forward(self, x):
            return torch.ops.pypto.attr_onnx_f(x, alpha=0.25)

    path = str(tmp_path / "alpha.onnx")
    with exporting_scope():
        torch.onnx.export(M(), (torch.rand((1, 8, 1, 64), dtype=torch.float16),), path,
                          opset_version=12, do_constant_folding=False, dynamo=False, custom_opsets={"pypto": 1})
    node = next(n for n in onnx.load(path).graph.node if "AlphaOnnx" in n.op_type)
    at = next(a for a in node.attribute if a.name == "alpha")
    assert at.type == onnx.AttributeProto.FLOAT     # type 1
    assert abs(at.f - 0.25) < 1e-6
    _reset_export_state()


def test_onnx_export_emits_string_attr(tmp_path):
    import onnx
    _reset_export_state()
    ExportedCustomOp(kernel=_mode_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                     torch_defn=_mode_torch, torch_op_qualname="pypto::attr_onnx_s",
                     onnx_spec=OnnxSymbolicSpec(op_type="ModeOnnx", opset_version=12),
                     attrs=[AttrSpec("mode", "String", "sum")])
    finalize_pending_ops()

    class M(nn.Module):
        def forward(self, a, b):
            return torch.ops.pypto.attr_onnx_s(a, b, mode="diff")

    path = str(tmp_path / "mode.onnx")
    a = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    with exporting_scope():
        torch.onnx.export(M(), (a, torch.rand_like(a)), path,
                          opset_version=12, do_constant_folding=False, dynamo=False, custom_opsets={"pypto": 1})
    node = next(n for n in onnx.load(path).graph.node if "ModeOnnx" in n.op_type)
    at = next(a for a in node.attribute if a.name == "mode")
    assert at.type == onnx.AttributeProto.STRING    # type 3
    assert at.s == b"diff"
    _reset_export_state()


def test_onnx_export_emits_listint_attr(tmp_path):
    # ListInt goes out as INTS (type 7). pypto export never sets the ONNX output value_info, so every
    # pypto node's graph output dims read as 0 here; shape correctness is verified separately at the
    # C++ codegen level (see test_cpp_codegen_structure) and at GE/ATC deploy time, not in this graph.
    # This test only checks the node attribute, not the output shape.
    import onnx
    _reset_export_state()
    ExportedCustomOp(kernel=_crop_factory, infer_shape=_crop_infer_shape, infer_dtype=_single_infer_dtype,
                     torch_defn=_crop_torch, torch_op_qualname="pypto::attr_onnx_li",
                     onnx_spec=OnnxSymbolicSpec(op_type="CropOnnx", opset_version=12),
                     attrs=[AttrSpec("crop_size", "ListInt", [4, 4])])
    finalize_pending_ops()

    class M(nn.Module):
        def forward(self, x):
            return torch.ops.pypto.attr_onnx_li(x, crop_size=[4, 4])

    path = str(tmp_path / "crop.onnx")
    with exporting_scope():
        torch.onnx.export(M(), (torch.rand((1, 3, 8, 8), dtype=torch.float32),), path,
                          output_names=["y"], opset_version=12, do_constant_folding=False, dynamo=False,
                          custom_opsets={"pypto": 1})
    node = next(n for n in onnx.load(path).graph.node if "CropOnnx" in n.op_type)
    at = next(a for a in node.attribute if a.name == "crop_size")
    assert at.type == onnx.AttributeProto.INTS      # type 7
    assert list(at.ints) == [4, 4]
    # Torch-side shape-affecting sanity: the EAGER op (crop torch_defn) honors the attr, exercising
    # torch_defn<->infer_shape consistency (NOT the ONNX graph output shape).
    out = torch.ops.pypto.attr_onnx_li(torch.zeros(1, 3, 8, 8), crop_size=[4, 4])
    assert tuple(out.shape) == (1, 3, 4, 4)
    _reset_export_state()


def test_op_export_record_multi_attr_order_preserved():
    # Serialization-only test — no torch_defn needed (an op-export-record test doesn't run the op).
    op = ExportedCustomOp(kernel=_crop_factory, infer_shape=_crop_infer_shape, infer_dtype=_single_infer_dtype,
                          torch_op_qualname="pypto::attr_multi_order",
                          attrs=[AttrSpec("crop_size", "ListInt", [4, 4]), AttrSpec("dim", "Int", 0)])
    assert [s.name for s in op._attr_specs] == ["crop_size", "dim"]  # declaration order preserved

    def _stub_dtypes(*nodes):
        return [torch.float32]
    export_fn = _build_node_meta_completer(
        op=op, encode_node_meta=_onnx_encode_node_meta, input_dtypes=_stub_dtypes, framework_kind="onnx",
    )
    export_fn(object(), op_type="PyptoCustomOpX", domain="pypto", domain_opset_version=1)
    record = json.loads(op._node_meta[_META_KEY__OP_EXPORT_RECORD])
    assert [a["name"] for a in record["attrs"]] == ["crop_size", "dim"]  # == declared-attr / declaration order


def test_shape_affecting_attr_tensor_arity_from_infer_dtype():
    # The tensor-input count comes from infer_dtype (1 param), NOT infer_shape (2 params: x + crop_size).
    # So torch_defn arity = 1 tensor + 1 attr = 2, and the operand split stays correct.
    op = ExportedCustomOp(kernel=_crop_factory, infer_shape=_crop_infer_shape, infer_dtype=_single_infer_dtype,
                          torch_defn=_crop_torch, torch_op_qualname="pypto::attr_arity_shape",
                          attrs=[AttrSpec("crop_size", "ListInt", [4, 4])])
    assert op._attr_specs == (AttrSpec("crop_size", "ListInt", [4, 4]),)
    # a torch_defn with the WRONG arity (missing the attr operand) is rejected
    with pytest.raises(ValueError, match="attrs"):
        ExportedCustomOp(kernel=_crop_factory, infer_shape=_crop_infer_shape, infer_dtype=_single_infer_dtype,
                         torch_defn=lambda x: x, torch_op_qualname="pypto::attr_arity_shape_bad",
                         attrs=[AttrSpec("crop_size", "ListInt", [4, 4])])



def test_convert_shape_attr_typed_from_string():
    # The run-path shape-attr conversion mirrors the factory: list[int]->json.loads, int->int, and a
    # non-string (already typed) value passes through.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import _convert_shape_attr
    assert _convert_shape_attr("[2, 2]", list[int], "crop_size") == [2, 2]
    assert _convert_shape_attr("3", int, "k") == 3
    assert _convert_shape_attr([2, 2], list[int], "crop_size") == [2, 2]  # already typed -> passthrough
    assert _convert_shape_attr("2", float, "scale") == 2.0  # a Float attr, not the int json.loads gives


def test_run_path_infer_shape_threads_shape_affecting_attr():
    # Run-path regression (the shape-affecting-attr NPU failure): infer_shape for a shape-affecting-attr op
    # is called with the tensor shapes THEN the converted attr value, so out_shapes reflect the attr.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=1, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = _crop_infer_shape  # infer_shape(x_shape, crop_size) — 1 tensor + 1 shape-affecting attr
    args = entry._infer_shape_run_args([(1, 3, 8, 8)], {"crop_size": "[4, 4]"})
    assert args == [(1, 3, 8, 8), [4, 4]]                         # tensor shape THEN converted attr
    out_shapes, _ = entry._infer_out_shapes_dtypes([(1, 3, 8, 8)], [torch.float16], {"crop_size": "[4, 4]"})
    assert tuple(out_shapes[0]) == (1, 3, 4, 4)                   # cropped H/W = crop_size


def test_run_path_infer_shape_shape_invariant_unchanged():
    # A shape-INVARIANT op (infer_shape has no trailing attr param) is untouched: run args are just the
    # tensor shapes, even when a (non-shape) attr like bias rides in the dict.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=2, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = _bias_infer_shape  # (a, b) -> a; no attr param
    assert entry._infer_shape_run_args([(1, 8), (1, 8)], {"bias": "3"}) == [(1, 8), (1, 8)]


# ── infer_shape annotations that are not a plain evaluated ``str`` / ``int`` / ``list[int]`` ──────────


class _LayoutCallableInferShape:
    """A callable-object infer_shape hook whose CLASS annotates a name its ``__call__`` also declares."""
    layout: int = 0

    def __call__(self, x: torch.Size, layout: str) -> torch.Size:
        return torch.Size((*x, len(layout)))


def _layout_infer_shape_defaulting_to_none(x: torch.Size, layout: str = None) -> torch.Size:
    return torch.Size((*x, len(layout)))


@pytest.mark.parametrize("hook_name,attrs,expected", [
    ("str_attr", {"layout": "NHWC"}, "NHWC"),
    ("str_attr", {"layout": "123"}, "123"),
    ("int_attr", {"n": "2"}, 2),
    ("listint_attr", {"crop": "[4, 4]"}, [4, 4]),
], ids=["str", "str-json-lookalike", "int", "list-int"])
def test_run_path_infer_shape_resolves_a_postponed_annotation(hook_name, attrs, expected):
    # An author module with ``from __future__ import annotations`` (PEP 563) leaves the annotation SOURCE in
    # the signature, so the attr is converted against the resolved annotation: a String attr reaches
    # infer_shape as the declared str, both for 'NHWC' and for a JSON-parseable value like '123'.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=1, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = getattr(postponed, hook_name)
    assert entry._infer_shape_run_args([(1, 8)], attrs) == [(1, 8), expected]


@pytest.mark.parametrize("raw", ["NHWC", "123"])
@pytest.mark.parametrize("hook", [_LayoutCallableInferShape(), _layout_infer_shape_defaulting_to_none],
                         ids=["callable-object", "none-default"])
def test_run_path_infer_shape_keeps_an_evaluated_annotation(hook, raw):
    # An annotation the author's module already evaluated is used as it stands. Resolving it again would read
    # a callable object's CLASS namespace instead of its ``__call__`` parameters, which for 'NHWC' raises and
    # for '123' silently hands infer_shape an int.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=1, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = hook
    assert entry._infer_shape_run_args([(1, 8)], {"layout": raw}) == [(1, 8), raw]


def _layout_infer_shape_with_an_unresolvable_input(x: "Unresolvable", layout: str) -> torch.Size:  # noqa: F821
    return torch.Size((*x, len(layout)))


def test_run_path_infer_shape_does_not_resolve_an_evaluated_annotation():
    # An evaluated annotation is passed on as written, so one quoted name that resolves nowhere does not
    # stop a String attr from converting. A deployed compile snippet is that case: it carries no
    # TYPE_CHECKING import, and nothing re-derives the arity that would reject the hook first.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=1, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = _layout_infer_shape_with_an_unresolvable_input
    assert entry._infer_shape_run_args([(1, 8)], {"layout": "NHWC"}) == [(1, 8), "NHWC"]


@pytest.mark.parametrize("hook_name,attrs,expected", [
    ("CallableStrAttr", {"layout": "NHWC"}, "NHWC"),
    ("CallableListIntAttr", {"crop": "[4, 4]"}, [4, 4]),
], ids=["str", "list-int"])
def test_run_path_infer_shape_resolves_a_postponed_callable_object_from_its_call(hook_name, attrs, expected):
    # A callable object's parameters belong to its ``__call__``, so the annotations are read from there.
    # Both hooks carry a class attribute of the same name annotated as the OTHER type: reading the class
    # namespace instead converts 'NHWC' to an int, and hands ``crop`` the string '[4, 4]' silently.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=1, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = getattr(postponed, hook_name)()
    assert entry._infer_shape_run_args([(1, 8)], attrs) == [(1, 8), expected]


def test_run_path_infer_shape_keeps_refusing_a_postponed_annotated_attr():
    # ``Annotated[str, ...]`` stays refused, as it is at the root of an export: the deploy codegen reads
    # the same hints with include_extras=True, so the extras survive the lookup, and it accepts only the
    # plain int / float / str / list[int] annotations. Dropping the extras here would make the run path
    # take an op that cannot be built.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=1, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = postponed.annotated_str_attr
    with pytest.raises(ValueError, match="cannot convert"):
        entry._infer_shape_run_args([(1, 8)], {"layout": "NHWC"})


def test_run_path_infer_shape_names_an_unresolvable_annotation():
    # One unresolvable annotation fails the whole signature's resolution, and the name that did not
    # resolve is the report worth having, not a complaint about the attr being converted.
    from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry
    entry = CompileEntry(num_inputs=1, num_outputs=1, factory=lambda *a, **k: None, factory_signature="full")
    entry.infer_shape = postponed.str_attr_with_an_unresolvable_sibling
    with pytest.raises(NameError, match="Unresolvable"):
        entry._infer_shape_run_args([(1, 8)], {"layout": "NHWC", "mode": "x"})


@pytest.mark.parametrize("value,expected_key", [(7, "k_i"), (1.5, "k_f"), ("s", "k_s")])
def test_encode_node_meta_encodes_scalar_types_in_the_attribute_name(value, expected_key):
    assert _onnx_encode_node_meta({"k": value}) == {expected_key: value}


@pytest.mark.parametrize("value", [[1, 2], (1, 2), {"a": 1}, None, True])
def test_encode_node_meta_rejects_a_value_with_no_scalar_attribute_kind(value):
    with pytest.raises(TypeError, match=r"node_meta\['k'\] is \w+, which has no ONNX scalar attribute kind"):
        _onnx_encode_node_meta({"k": value})
