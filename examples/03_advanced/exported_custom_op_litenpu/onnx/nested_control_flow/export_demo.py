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
"""Export a model containing PyPTO operators in nested ONNX conditional subgraphs.

Scripting preserves the model's conditional branches. The export session supplies
the standard ONNX domain hint for the If nodes, and the exporter uses the opset
version shared by the operator declarations in op.py.

This file is the export entry point. run_demo.py executes the selected branch of
the same model. Reusable modules should not import either entry-point script.
"""
import argparse
import os
import pathlib as _pl
import sys as _sys
import tempfile

import torch

import pypto

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))  # <repo root>/tools
from exported_custom_op_litenpu.export.onnx_export import onnx_export_session
from model import SHAPE, CustomModel
from op import ONNX_OPSET_VERSION


def export_demo(model, path: str):
    pypto.set_codegen_options(support_dynamic_aligned=True)
    input_data0 = torch.rand(SHAPE, dtype=torch.float16)
    input_data1 = torch.rand(SHAPE, dtype=torch.float16)
    # Which branch the example inputs take does not matter: scripting keeps all three.
    use_add, use_sub = torch.tensor(True), torch.tensor(False)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # The opset is PINNED rather than resolved from a forward pass: resolve_pypto_export_opset()
    # discovers the ops one forward invokes, and a data-dependent conditional runs exactly one branch,
    # so it would see one of the three ops. All three declare the same opset in op.py.
    with onnx_export_session(path, ge_standard_domain="ai.onnx"):
        torch.onnx.export(
            torch.jit.script(model), (input_data0, input_data1, use_add, use_sub), path,
            input_names=["x0", "x1", "use_add", "use_sub"], output_names=["y"],
            opset_version=ONNX_OPSET_VERSION, do_constant_folding=False, dynamo=False, report=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs="?", help="onnx model path (default: a temp dir)")
    args = parser.parse_args()
    # Optional so the repo examples sweep can run this bare (a fresh temp dir when omitted).
    path = args.path or os.path.join(
        tempfile.mkdtemp(prefix="pypto_nested_control_flow_"), "nested_control_flow.onnx"
    )
    export_demo(CustomModel(), path=path)
