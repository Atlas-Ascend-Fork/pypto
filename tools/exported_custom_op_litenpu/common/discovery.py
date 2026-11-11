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
"""Pypto custom-op discovery: load a model and find its pypto nodes.

``load_model`` loads an ``.onnx`` artifact; ``find_pypto_nodes`` walks a loaded ONNX model and returns
the pypto custom-op nodes (deduped by op_type), gating each against the supported node layout version. Both
delegate to pypto core's onnx node reader.
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import onnx

from pypto.extensions.torch_custom_op_litenpu import (
    check_pypto_node_layout,
    extract_op_type,
    iter_pypto_nodes,
)

__all__ = ("find_pypto_nodes", "load_model")


def load_model(path: Union[str, Path]) -> Any:
    """Load a pypto-exported ``.onnx`` file and return the ``onnx.ModelProto``."""
    path = Path(path)
    if path.suffix.lstrip(".").lower() == "onnx":
        return onnx.load(str(path))
    raise ValueError(f"Unsupported model format: {path.suffix!r}")


def find_pypto_nodes(
    model: Any,
    op_types: Optional[Sequence[str]] = None,
) -> list:
    """Return every pypto custom-op node in *model*, in walk order, deduped by op_type.

    A node counts as pypto iff it carries the ``pypto_node_layout_version`` marker — nothing else does.
    The walk is depth first, so the dedupe keeps the first occurrence of an op_type that it reaches.
    """
    if not isinstance(model, onnx.ModelProto):
        raise TypeError(f"find_pypto_nodes expects an onnx.ModelProto, got {type(model).__name__}")
    it = iter_pypto_nodes(model)

    wanted = set(op_types) if op_types else None
    seen: set = set()
    out: list = []
    for n in it:
        check_pypto_node_layout(n)
        ot = extract_op_type(n)
        if wanted is not None and ot not in wanted:
            continue
        if ot in seen:
            continue
        seen.add(ot)
        out.append(n)
    if wanted is not None:
        missing = wanted - seen
        if missing:
            raise ValueError(f"op_types not found in model: {sorted(missing)}")
    return out
