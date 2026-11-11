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
"""Export a pypto op to ONNX: qualify its op type, encode its attributes, register its symbolic.

``_synthesize_onnx`` turns a finalized ``OnnxSymbolicSpec`` into a registered ``torch.onnx`` symbolic
at finalize, recording the op's base ai.onnx opset in ``_ONNX_OPSET_FLOORS`` (read back via
``recorded_onnx_opset_floor``). ``_onnx_encode_node_meta`` encodes finished Python values;
``_onnx_encode_attrs`` encodes declared ``AttrSpec``s from traced operands.
"""
from dataclasses import dataclass
import inspect
import logging
from typing import Optional

from ..common import return_annotations, torch_op
from ..common.compile import _as_tuple_of
from ..common.exported_custom_op import _build_node_meta_completer
from ..common.finalize import _register_reset_hook, _register_synthesizer

__all__ = ("recorded_onnx_opset_floor",)

logger = logging.getLogger(__name__)


# Base ai.onnx opset each ONNX symbolic registered at, keyed by the torch op qualified name (the
# "floor"; the max drives the export opset). Recorded at torch.onnx registration, read back through
# recorded_onnx_opset_floor. Process-global: floors accumulate across every op authored in the process,
# so any test asserting an absolute floor MUST clear it first (_reset_onnx_opset_floors, a test hook).
_ONNX_OPSET_FLOORS: "dict[str, int]" = {}


def recorded_onnx_opset_floor(qualnames) -> Optional[int]:
    """Max base ai.onnx opset over the recorded ONNX symbolics of *qualnames*.

    The read is per-export scoped, so a higher-opset op elsewhere in the process cannot bleed in.
    """
    floors = [v for q, v in _ONNX_OPSET_FLOORS.items() if q in qualnames]
    return max(floors) if floors else None


def _reset_onnx_opset_floors() -> None:
    """Clear the recorded opset floors. Test-only isolation hook (not part of the public API)."""
    _ONNX_OPSET_FLOORS.clear()


# Namespace prefix stamped onto every op identifier. The qualified value is the EFFECTIVE op_type: the
# ONNX/GE graph node type, the C++ REG_OP / executor-class name, and the generated cpp_sources folder are
# all derived from it. The prefix keeps custom ops from colliding with GE's built-in op protos (e.g. a bare
# ``Add`` / ``Softmax`` would clash), so authors declare only the short semantic name (``Add``) on the spec.
_OP_TYPE_QUALIFIER = "PyptoCustomOp"


def _qualify_op_type(op_type: str) -> str:
    """Return the effective op identifier: the author's op name under the ``PyptoCustomOp`` namespace.

    Idempotent — an already-qualified name passes through unchanged.
    """
    return op_type if op_type.startswith(_OP_TYPE_QUALIFIER) else f"{_OP_TYPE_QUALIFIER}{op_type}"


# ONNX attribute kind -> (torch ``g.op`` kwarg-name suffix, ``_parse_arg`` descriptor). The single
# statement of the encoding: both encoders and the traced-operand reader resolve through it, so a kind is
# described once. INTS shares the INT suffix -- torch widens it from the value -- and differs only in how
# _parse_arg must read the operand.
@dataclass(frozen=True)
class _OnnxAttrEncoding:
    """How one ONNX attribute kind is written and read back."""
    suffix: str      # the torch ``g.op`` kwarg-name suffix
    descriptor: str  # what ``_parse_arg`` needs to read a traced operand of this kind


_ONNX_ATTR_ENCODING = {
    "INT": _OnnxAttrEncoding("i", "i"),
    "FLOAT": _OnnxAttrEncoding("f", "f"),
    "STRING": _OnnxAttrEncoding("s", "s"),
    "INTS": _OnnxAttrEncoding("i", "is"),
}


def _build_onnx_attr_kwarg(name: str, value, kind: str):
    """The ``(kwarg, value)`` pair ``g.op`` needs to emit *value* as an ONNX *kind* attribute.

    The suffix carries the ELEMENT kind: torch widens ``_i`` to INTS from the value being a list, not
    from the name, so a kind/value disagreement would silently serialize the wrong AttributeProto type.
    """
    try:
        suffix = _ONNX_ATTR_ENCODING[kind].suffix
    except KeyError:
        raise ValueError(f"unknown ONNX attribute kind {kind!r} for attribute {name!r}") from None
    if (kind == "INTS") != isinstance(value, (list, tuple)):
        raise ValueError(
            f"attribute {name!r} declares ONNX kind {kind} but its value is {type(value).__name__}; "
            "torch selects INT vs INTS from the value, so the two must agree"
        )
    return f"{name}_{suffix}", value


# Node-meta value type -> ONNX scalar attribute kind. Exact-type lookup rather than an isinstance chain:
# bool is an int subclass, and a widened lookup would emit it as an INT attribute instead of rejecting it.
_NODE_META_KIND_BY_TYPE = {int: "INT", float: "FLOAT", str: "STRING"}


def _onnx_encode_node_meta(node_meta: dict):
    """Build the ONNX attribute kwargs from the node meta, encoding each value's type in its name.

    Raises ``TypeError`` for a value whose type has no ONNX scalar attribute kind.
    """
    node_meta_kwargs = {}
    for k, v in node_meta.items():
        kind = _NODE_META_KIND_BY_TYPE.get(type(v))
        if kind is None:
            raise TypeError(
                f"node_meta[{k!r}] is {type(v).__name__}, which has no ONNX scalar attribute kind -- a "
                f"node meta value must be an int, a float or a str; serialize structured data at the call site"
            )
        kwarg, encoded = _build_onnx_attr_kwarg(k, v, kind)
        node_meta_kwargs[kwarg] = encoded
    return node_meta_kwargs


# AttrSpec.type -> ONNX attribute kind for a declared compute attr (the trailing-operand attr model).
# Only the translation lives here; the suffix and the _parse_arg descriptor for each kind come from
# ``_ONNX_ATTR_ENCODING`` above, so neither is restated.
_ATTR_ONNX_KIND = {"Int": "INT", "Float": "FLOAT", "String": "STRING", "ListInt": "INTS"}


def _onnx_encode_attrs(attr_specs, attr_values):
    """Map declared attrs (trailing operands) to typed ``g.op`` kwargs (see ``_ATTR_ONNX_KIND``).

    *attr_values* are the traced ``torch._C.Value`` operands trailing the tensor inputs. Each is read back
    to its Python constant via ``torch.onnx.symbolic_helper._parse_arg`` and emitted as ``<name>_<suffix>``
    so ``torch.onnx`` serializes it as the right ONNX ``AttributeProto`` type. The raw typed value is
    passed (not JSON), so a ListInt goes out as native INTS.
    """
    # torch.onnx.symbolic_helper._parse_arg is a private API that can drift across torch versions;
    # it is the pinned-toolchain path to read a traced operand's constant, mirroring the existing
    # @parse_args("v","v","i") precedent the mixed-op demos use. pypto declares no torch dependency, so
    # the caller's torch decides: on a bump, re-check this import and _ONNX_ATTR_ENCODING's descriptors.
    from torch.onnx.symbolic_helper import _parse_arg  # noqa: PLC0415,PLC2701
    out = {}
    for spec, value in zip(attr_specs, attr_values):
        kind = _ATTR_ONNX_KIND[spec.type]
        parsed = _parse_arg(value, _ONNX_ATTR_ENCODING[kind].descriptor)
        kwarg, encoded = _build_onnx_attr_kwarg(spec.name, parsed, kind)
        out[kwarg] = encoded
    return out


def _resolve_infer_shape_args(op, in_types, attr_values):
    """Build ``infer_shape``'s positional args from the traced operands: tensor shapes, then attrs.

    Each attr operand holds a constant, read back through the same ``_parse_arg`` + descriptor pair
    ``_onnx_encode_attrs`` uses, so ``infer_shape`` receives ints rather than IR nodes.
    """
    import torch  # noqa: PLC0415 - optional dependency: only the symbolic path needs torch
    from torch.onnx.symbolic_helper import _parse_arg  # noqa: PLC0415,PLC2701 - see _onnx_encode_attrs
    attr_specs = getattr(op, "_attr_specs", ())
    parsed = [_parse_arg(value, _ONNX_ATTR_ENCODING[_ATTR_ONNX_KIND[spec.type]].descriptor)
              for spec, value in zip(attr_specs, attr_values)]
    shapes = [torch.Size(t.sizes()) for t in in_types]
    return torch_op._infer_shape_call_args(op, shapes, parsed)


def _set_symbolic_output_type(op, inputs, attr_values, node_outputs, n_out) -> None:
    """Annotate each of the synthesized node's outputs from the op's own infer funcs.

    Load-bearing only when nested: the output becomes an ``If``/``Loop`` subgraph output, and torch
    merges the branch types into the model output's, which ``onnx.checker`` rejects when it is missing
    an element type or a shape. At the root graph the model output picks up a type from the trace.
    """
    if not inputs:
        return
    try:
        in_types = [value.type() for value in inputs]
        # Which operand feeds which output is not knowable here, so every output would take input 0's
        # device. Operands that disagree on it make that a guess, and such an op cannot run either.
        if len({str(t.device()) for t in in_types}) > 1:
            raise ValueError(
                f"pypto custom-op {op._torch_op_qualname!r}: operands are on more than one device, so an "
                "ONNX output type cannot declare one. Put every operand on the same device."
            )
        dtypes = _as_tuple_of(op._infer_dtype_fn(*[t.dtype() for t in in_types]), n_out)
        # An element type with no shape still satisfies an ``If`` branch whose sibling carries one.
        if any(t.sizes() is None for t in in_types):
            logger.warning(
                "pypto: %s has an operand dimension the trace left unknown, so its ONNX output "
                "declares an element type and no shape. Only a dynamic_axes export causes this: "
                "infer_shape takes concrete shapes and has no value for an unknown dimension. "
                "Export without dynamic_axes.", op._torch_op_qualname,
            )
            sizes = (None,) * n_out
        else:
            sizes = _as_tuple_of(
                op._infer_shape_fn(*_resolve_infer_shape_args(op, in_types, attr_values)), n_out)
        # Above one output g.op hands back a tuple of values; at one output it hands back the value.
        outs = (node_outputs,) if n_out == 1 else tuple(node_outputs)
        for value, dtype, size in zip(outs, dtypes, sizes):
            # Input 0's type is the template: it carries the device, which ``prim::type`` reads off an
            # output. Its strides ride along unreconciled; no exporter pass reads those.
            value.setType(in_types[0].with_dtype(dtype)
                          .with_sizes(None if size is None else list(size)))
    except (AttributeError, RuntimeError, TypeError) as exc:
        # An unknown dtype or rank (RuntimeError), a non-traced value (AttributeError) or None into
        # ``torch.Size`` (TypeError) defeat the annotation; anything else is the op's own defect.
        logger.warning(
            "pypto: could not annotate the ONNX output type of %s (%s: %s); nesting this op inside an "
            "If/Loop subgraph will fail onnx.checker, which requires the model output to declare both "
            "an element type and a shape.", op._torch_op_qualname, type(exc).__name__, exc,
        )


def _synthesize_onnx(op, spec) -> None:
    """Synthesize + register the ONNX symbolic for *op* from *spec* (skipped if manually attached)."""
    if op._onnx_symbolic_attached:
        return  # already synthesized (idempotency guard)
    n_out = return_annotations.infer_shape_output_arity(op._infer_shape_fn)
    op_type = _qualify_op_type(spec.op_type)
    domain, domain_opset_version = spec.domain, spec.domain_opset_version
    # The op's trailing scalar operands are the declared attrs; the leading n_in are the tensor inputs.
    # Split so only the tensors feed the kernel-export dtype extraction and the g.op node inputs, and
    # the attrs become typed g.op kwargs. n_in is the TENSOR count = infer_dtype's param count, which is
    # drift-free because infer_dtype never takes attrs (unlike infer_shape, once an attr is shape-affecting).
    n_in = len(inspect.signature(op._infer_dtype_fn).parameters)
    attr_specs = getattr(op, "_attr_specs", ())
    # Built once per op here rather than per symbolic invocation. "onnx" is the framework_kind recorded
    # in the node's export record, naming which flow authored the node.
    complete_node_meta = _build_node_meta_completer(
        op=op, encode_node_meta=_onnx_encode_node_meta,
        input_dtypes=lambda *input_nodes: [node.type().dtype() for node in input_nodes],
        framework_kind="onnx",
    )

    def _symbolic(g, *args):
        inputs = args[:n_in]
        attr_values = args[n_in:]
        node_meta_kwargs = complete_node_meta(
            *inputs, op_type=op_type, domain=domain, domain_opset_version=domain_opset_version,
        )
        attr_kwargs = _onnx_encode_attrs(attr_specs, attr_values)
        out = g.op(f"{domain}::{op_type}", *inputs, outputs=n_out,
                   **node_meta_kwargs, **attr_kwargs)
        _set_symbolic_output_type(op, inputs, attr_values, out, n_out)
        return out

    op._onnx_symbolic_attached = True
    import torch.onnx  # noqa: PLC0415 - optional dependency: only the synthesis path needs torch.onnx
    torch.onnx.register_custom_op_symbolic(op._torch_op_qualname, _symbolic, spec.opset_version)
    _ONNX_OPSET_FLOORS[op._torch_op_qualname] = spec.opset_version


# Bind this layer's wiring into the finalize registry; the import edge runs onnx -> common.
_register_synthesizer("onnx", _synthesize_onnx)
_register_reset_hook(_reset_onnx_opset_floors)
