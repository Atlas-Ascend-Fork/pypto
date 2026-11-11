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
"""Read pypto's node meta back off ONNX nodes: node lookup, string/zip readers, and marker iterator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator

from ..common.node_meta import _META_KEY__PYPTO_NODE_LAYOUT_VERSION

if TYPE_CHECKING:  # annotations only — the runtime imports are function-scope (see below)
    import onnx


def _extract_string_attr_from_onnx_node(
    onnx_node: onnx.NodeProto,
    attr_name: str,
):
    """Read a string attribute from an ONNX node."""
    # Function-scope so this module carries no module-scope edge to onnx (the annotations above are
    # strings under ``from __future__ import annotations``, so they cost nothing at import time).
    import onnx  # noqa: PLC0415 - optional dependency: onnx is imported only when a node is read

    attr = next((a for a in onnx_node.attribute if a.name == attr_name), None)
    if attr is None:
        raise KeyError(f"Could not find {attr_name} attribute in onnx node")
    if attr.type != onnx.AttributeProto.STRING:
        raise TypeError(f"{attr.name} is not a STRING attribute")
    value = attr.s.decode("utf-8", errors="strict").strip()
    if not value:
        raise ValueError(f"{attr.name} is empty")
    return value


def _subgraphs_of_onnx_node(onnx_node: onnx.NodeProto) -> list[onnx.GraphProto]:
    """Return every graph nested in *onnx_node*'s attributes, in proto attribute order.

    Dispatch is on the attribute's declared type, not on ``op_type``: that covers ``SequenceMap``, a
    ``GRAPHS``-valued attribute and any custom control-flow op without an op-name list to maintain.
    """
    import onnx  # noqa: PLC0415 - optional dependency: onnx is imported only when a node is read

    nested: list[onnx.GraphProto] = []
    for attr in onnx_node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            nested.append(attr.g)
        elif attr.type == onnx.AttributeProto.GRAPHS:
            nested.extend(attr.graphs)
    return nested


def iter_nodes_in_node_tree(root_nodes: Iterable[Any]) -> Iterator[Any]:
    """Yield *root_nodes* and the nodes of every graph nested beneath them, depth first.

    Depth first: a node's own subgraphs are yielded before its next sibling.
    """
    for node in root_nodes:
        yield node
        for nested_graph in _subgraphs_of_onnx_node(node):
            yield from iter_nodes_in_node_tree(nested_graph.node)


def iter_all_onnx_nodes(onnx_model: onnx.ModelProto) -> Iterator[Any]:
    """Yield every node of the root graph and of every graph nested beneath it, depth first."""
    yield from iter_nodes_in_node_tree(onnx_model.graph.node)


def _iter_pypto_nodes_from_onnx_model(onnx_model: onnx.ModelProto) -> Iterator[Any]:
    """Yield every ONNX node carrying the ``pypto_node_layout_version`` marker, nested ones included.

    The walk order is observable, not an implementation detail: ``find_pypto_nodes`` dedupes by
    ``op_type`` keeping the first occurrence, and the walk is depth first, so a node nested under an
    earlier sibling wins over a root-graph node of its type that appears later.
    """
    for node in iter_all_onnx_nodes(onnx_model):
        try:
            _extract_string_attr_from_onnx_node(node, _META_KEY__PYPTO_NODE_LAYOUT_VERSION)
        except (KeyError, TypeError, ValueError):
            continue
        yield node
