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
"""Generic ONNX export orchestration for demos (export helpers).

The demo keeps the ``torch.onnx.export`` call visible; this provides the scope + finalize/validate around
it. Mixed-model safe: the finalize touches only pypto nodes (pypto core), and the generic checker / GE
domain rewrite / save apply to the whole graph. The base ai.onnx opset auto-syncs from the floors that
the synthesized ONNX symbolics (``OnnxSymbolicSpec.opset_version``, recorded at finalize) leave in pypto core;
:func:`resolve_pypto_export_opset` reads them back via
``pypto.extensions.torch_custom_op_litenpu.recorded_onnx_opset_floor``.
"""

import contextlib
import json
import logging

import onnx

# The example library imports pypto.extensions.torch_custom_op_litenpu: read the opset floor + call
# finalize_pending_ops() (finalize config-declared ops: synth torch op + symbolic). Legal (that
# package never imports these helpers, so no cycle); ``pypto`` is installed for demos.
from pypto.extensions.torch_custom_op_litenpu import (
    check_pypto_node_layout,
    exporting_scope,
    extract_op_export_record,
    finalize_pending_ops,
    iter_pypto_nodes,
    recorded_onnx_opset_floor,
)
from pypto.extensions.torch_custom_op_litenpu.onnx.node_reader import iter_nodes_in_node_tree

logger = logging.getLogger(__name__)


def _exported_pypto_nodes(model: onnx.ModelProto) -> list:
    """Return *model*'s pypto nodes, each gated against the node layout version this pypto supports.

    Every node is returned, repeats of one ``op_type`` included, so two instances whose export records
    disagree are caught by :func:`finalize_pypto_onnx_nodes` rather than reduced to the first.
    """
    nodes = list(iter_pypto_nodes(model))
    for node in nodes:
        check_pypto_node_layout(node)
    return nodes


def finalize_pypto_onnx_nodes(model: onnx.ModelProto, nodes) -> None:
    """Pypto-node-only ONNX post-pass: stamp each custom domain's opset from the node's export record.

    Single pass over *nodes*, the pypto nodes of *model* (mixed-model safe, since the caller selects
    them): the ``op_export_record`` written by ``complete_node_meta`` is the single source of truth for
    each custom op's ``(domain, domain_opset_version)``. For every pypto node this:

    * asserts the ONNX node's ``domain`` matches the recorded domain (the symbolic's ``g.op(...)`` prefix
      must equal the ``domain`` passed to ``complete_node_meta``, else the CANN parser can't dispatch),
    * asserts all ops sharing a domain agree on its opset,
    * stamps ``opset_import[domain]`` to the recorded opset (one entry per domain).

    The generic whole-graph steps (checker, GE domain rewrite, save) live in :func:`onnx_export_session`.
    """
    derived: dict = {}
    for node in nodes:
        # Every node here carries the pypto marker and passed the layout check, and op_export_record is
        # written unconditionally with both coordinates.
        record = json.loads(extract_op_export_record(node))
        recorded_domain = record["domain"]
        recorded_opset = record["domain_opset_version"]
        if node.domain != recorded_domain:
            raise RuntimeError(
                f"pypto custom-op {node.op_type!r}: ONNX node has domain {node.domain!r}, but codegen "
                f"recorded domain={recorded_domain!r}. The symbolic's g.op(...) prefix must match the "
                "domain kwarg passed to complete_node_meta."
            )
        if recorded_domain in derived and derived[recorded_domain] != recorded_opset:
            raise RuntimeError(
                f"pypto nodes disagree on the opset of domain {recorded_domain!r}: "
                f"{derived[recorded_domain]} vs {recorded_opset}. All ops in a domain must declare the "
                "same domain_opset_version."
            )
        derived[recorded_domain] = recorded_opset

    existing = {imp.domain: imp for imp in model.opset_import}
    for dom, ver in derived.items():
        imp = existing.get(dom)
        if imp is None:
            logger.info("pypto: domain %r opset set to %d from codegen", dom, ver)
            model.opset_import.append(onnx.helper.make_opsetid(dom, ver))
        elif imp.version != ver:
            logger.info("pypto: domain %r opset set to %d from codegen", dom, ver)
            imp.version = ver


def _rewrite_standard_domain_for_ge(model: onnx.ModelProto, standard_domain: str) -> None:
    """GE-compat: give standard ONNX nodes an explicit (non-empty) ``domain``, nested ones included.

    GE only auto-maps an empty domain to ai.onnx when one opset domain is declared; with pypto also
    imported it errors. Must run AFTER ``onnx.checker``, which rejects an unregistered domain string.
    """
    base_version = next((imp.version for imp in model.opset_import if imp.domain == ""), None)
    rewrote = False
    for node in iter_nodes_in_node_tree(model.graph.node):
        if node.domain == "":
            node.domain = standard_domain
            rewrote = True
    if rewrote and not any(imp.domain == standard_domain for imp in model.opset_import):
        model.opset_import.append(
            onnx.helper.make_opsetid(standard_domain, base_version if base_version is not None else 1)
        )


__all__ = ("resolve_pypto_export_opset", "onnx_export_session", "finalize_pypto_onnx_nodes")

# Base ai.onnx opset when the traced model invokes no pypto op and the caller passed no pin.
# 12 is a safe modern torch baseline; any pypto op's OnnxSymbolicSpec supplies its own floor.
_DEFAULT_BASE_OPSET = 12


def _discover_used_pypto_qualnames(model, example_inputs) -> set:
    """Run one forward under a TorchDispatchMode to record the torch.ops.pypto.* ops the model uses.

    exporting_scope() makes a no-torch_defn op return its shaped stub instead of raising.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    used = set()

    class _Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            used.add(func._schema.name)   # e.g. "pypto::add"
            return func(*args, **(kwargs or {}))

    with exporting_scope(), _Rec():
        model(*example_inputs)
    return used


def resolve_pypto_export_opset(opset_version=None, *, model=None, example_inputs=None):
    """Resolve the base ai.onnx opset to pass to ``torch.onnx.export(opset_version=...)``.

    *opset_version* is returned unchanged (the explicit pin; the ascendc path that must not drift).
    Otherwise *model* + *example_inputs* are required: a cheap pre-trace forward discovers the pypto op
    qualnames the model invokes and the floor is maxed over only those ops' recorded floors, so a
    higher-opset op elsewhere in the process cannot bleed in. *example_inputs* is a tuple of the model's
    forward args. ``_DEFAULT_BASE_OPSET`` when no symbolic supplies a floor (a model invoking no pypto
    op).
    """
    # Finalize first so config-declared ops record their symbolic/floor before we read it (double-trigger
    # with the session; idempotent). Demos call resolve_pypto_export_opset() before the `with` block, so
    # this is where the floor gets populated for the common case.
    finalize_pending_ops()
    if opset_version is not None:
        return opset_version
    used = _discover_used_pypto_qualnames(model, example_inputs)
    return recorded_onnx_opset_floor(used) or _DEFAULT_BASE_OPSET


@contextlib.contextmanager
def onnx_export_session(
    path,
    *,
    ge_standard_domain=None,
    check_model=True,
):
    """Scope a demo's ONNX export: finalize + validate the saved model on clean exit.

    Wrap a ``torch.onnx.export(..., dynamo=False)`` call (the legacy symbolic path pypto's custom ops
    require, since dynamo=True silently drops the custom symbolic), passing the export opset from
    :func:`resolve_pypto_export_opset`.
    """
    # Finalize config-declared ops (synth torch op + fake + onnx symbolic) BEFORE the wrapped export traces
    # the model. Idempotent: a no-op if resolve_pypto_export_opset already finalized, or if the demo is
    # hand-written.
    finalize_pending_ops()
    # Mark exporting so an op with no torch_defn returns the shaped-empty stub (export-safe) from its
    # synthesized impl instead of raising during the trace inside the wrapped torch.onnx.export call.
    with exporting_scope():
        yield
    # Reached only when the wrapped export did not raise.
    m = onnx.load(path)
    finalize_pypto_onnx_nodes(m, _exported_pypto_nodes(m))
    if check_model:
        onnx.checker.check_model(m)
    if ge_standard_domain is not None:
        _rewrite_standard_domain_for_ge(m, ge_standard_domain)
    onnx.save(m, path)
    logger.info("Exported ONNX model to %s", path)
    logger.info("ONNX graph:")
    logger.info(onnx.helper.printable_graph(m.graph))
