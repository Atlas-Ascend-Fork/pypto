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
"""compile_debug_mode=1 / runtime_debug_mode=1 产物看护。

覆盖：
1. 原有调试产物：program.json / dyn_topo.txt / merged_swimlane.json / tilefwk_L1_prof_data.json
2. IDE TOOLKIT 映射：kernel_aicore/cce_opmagic_map.json、cce_semantic_label.json
3. CCE 行内注释含 [opMagic:...][semanticLabel:...]
4. rundata.json 写入 kernel_aicore_path
"""

import json
import os
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch_npu

import pypto

_OUTPUT_CHECK_LIST = [
    "program.json",
    "dyn_topo.txt",
    "merged_swimlane.json",
    "tilefwk_L1_prof_data.json",
]
_CCE_OPMAGIC_MAP = "cce_opmagic_map.json"
_CCE_SEMANTIC_LABEL = "cce_semantic_label.json"
_KERNEL_AICORE = "kernel_aicore"
_RUNDATA_KEY = "kernel_aicore_path"


@pypto.frontend.jit(debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1})
def add_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=32,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_semantic_label("add_stage")
    c.move(a + b)


def _latest_subdir(root: str) -> str:
    if not os.path.isdir(root):
        return ""
    subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        return ""
    return max(subdirs, key=os.path.getctime)


def _assert_cce_debug_maps(latest_dir: str):
    kernel_dir = os.path.join(latest_dir, _KERNEL_AICORE)
    assert os.path.isdir(kernel_dir), f"missing kernel_aicore under {latest_dir}"

    op_map_path = os.path.join(kernel_dir, _CCE_OPMAGIC_MAP)
    label_map_path = os.path.join(kernel_dir, _CCE_SEMANTIC_LABEL)
    assert os.path.isfile(op_map_path), f"missing {_CCE_OPMAGIC_MAP}"
    assert os.path.isfile(label_map_path), f"missing {_CCE_SEMANTIC_LABEL}"

    op_map = json.loads(Path(op_map_path).read_text(encoding="utf-8"))
    label_map = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
    assert isinstance(op_map, list) and op_map, f"{_CCE_OPMAGIC_MAP} should be a non-empty list"
    assert isinstance(label_map, list) and label_map, f"{_CCE_SEMANTIC_LABEL} should be a non-empty list"
    assert any(rec.get("semanticLabel") == "add_stage" for rec in label_map), label_map

    cpp_files = list(Path(kernel_dir).glob("*.cpp"))
    assert cpp_files, f"no CCE cpp under {kernel_dir}"
    joined = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in cpp_files)
    assert "[opMagic:" in joined, "CCE sources missing [opMagic:...] comments"
    assert "[semanticLabel:add_stage]" in joined, "CCE sources missing semanticLabel comment"


def _assert_rundata_kernel_aicore_path(pypto_home: str, latest_dir: str):
    run_root = os.path.join(pypto_home, "run")
    rundata_dir = _latest_subdir(run_root)
    assert rundata_dir, f"no rundata dir under {run_root}"
    rundata_path = os.path.join(rundata_dir, "rundata.json")
    assert os.path.isfile(rundata_path), f"missing {rundata_path}"
    rundata = json.loads(Path(rundata_path).read_text(encoding="utf-8"))
    kernel_path = rundata.get(_RUNDATA_KEY, "")
    assert kernel_path, f"rundata missing {_RUNDATA_KEY}: {rundata}"
    expect_prefix = os.path.realpath(os.path.join(latest_dir, _KERNEL_AICORE))
    assert os.path.realpath(kernel_path) == expect_prefix, f"{kernel_path} != {expect_prefix}"


def check_output(pypto_home: str):
    out_path = os.environ.get("TILE_FWK_OUTPUT_DIR", "./output")
    latest_dir = _latest_subdir(out_path)
    assert latest_dir, f"no output subdir under {out_path}"

    lost_file = None
    for name in _OUTPUT_CHECK_LIST:
        file_path = os.path.join(latest_dir, name)
        if not os.path.exists(file_path):
            lost_file = file_path
            break
    assert lost_file is None, f"missing debug artifact: {lost_file}"

    _assert_cce_debug_maps(latest_dir)
    _assert_rundata_kernel_aicore_path(pypto_home, latest_dir)

    shutil.rmtree(out_path)


def device_run(device_id, pypto_home: str):
    tiling = 32
    n, m = tiling * 1, tiling * 1

    a_data = torch.ones((n, m), dtype=torch.int32, device=f"npu:{device_id}") * 2
    b_data = torch.ones((n, m), dtype=torch.int32, device=f"npu:{device_id}")
    c_data = torch.zeros((n, m), dtype=torch.int32, device=f"npu:{device_id}")

    add_kernel(a_data, b_data, c_data, tiling=tiling)
    torch_npu.npu.synchronize()

    check_output(pypto_home)


@pytest.mark.soc("910", "950")
def test_debug_mode():
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", 0)
    with tempfile.TemporaryDirectory(prefix="pypto_debug_mode_") as tmp_dir:
        out_dir = os.path.join(tmp_dir, "output")
        pypto_home = os.path.join(tmp_dir, "pypto_home")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(pypto_home, exist_ok=True)
        os.environ["TILE_FWK_OUTPUT_DIR"] = out_dir
        os.environ["PYPTO_HOME"] = pypto_home
        torch.npu.set_device(int(device_id))
        device_run(device_id, pypto_home)
