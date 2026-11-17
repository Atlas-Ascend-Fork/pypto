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
"""compile_debug_mode=2（固定 CCE）正式看护。

流程：
1. 设置 ASCEND_WORK_PATH，以 compile_debug_mode=2 跑第 1 遍，落盘固定路径 CCE
2. 向 kernel_aicore/*.cpp 注入唯一 marker 注释
3. 新进程再跑第 2 遍（强制重新走 codegen/DumpCode，避免进程内缓存跳过写盘）
4. 断言 marker 仍在 → 证明第二次未覆盖已有 CCE
"""

import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import uuid

import pytest
import torch
import torch_npu

import pypto

_KERNEL_AICORE = "kernel_aicore"
_MARKER_PREFIX = "// FIXED_CCE_ST_MARKER "


@pypto.frontend.jit(debug_options={"compile_debug_mode": 2, "runtime_debug_mode": 0})
def fixed_cce_add_kernel(
    a: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    b: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    c: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_INT32),
    tiling=32,
):
    pypto.set_vec_tile_shapes(tiling, tiling)
    c.move(a + b)


def _kernel_aicore_dir(work_path: str) -> Path:
    return Path(work_path) / "pypto" / _KERNEL_AICORE


def _list_cce_cpp(kernel_dir: Path) -> list[Path]:
    return sorted(p for p in kernel_dir.glob("*.cpp") if p.is_file())


def _run_kernel_once(work_path: str, device_id: int, result_queue: mp.Queue):
    """Child process entry: one compile+launch under fixed CCE mode."""
    try:
        os.environ["ASCEND_WORK_PATH"] = work_path
        # Fixed-CCE path must not depend on TILE_FWK_OUTPUT_DIR for device-side CCE.
        os.environ.pop("TILE_FWK_OUTPUT_DIR", None)
        pypto.set_pass_options(enable_slice=True)
        torch.npu.set_device(int(device_id))

        tiling = 32
        n = m = tiling
        a = torch.ones((n, m), dtype=torch.int32, device=f"npu:{device_id}") * 2
        b = torch.ones((n, m), dtype=torch.int32, device=f"npu:{device_id}")
        c = torch.zeros((n, m), dtype=torch.int32, device=f"npu:{device_id}")
        fixed_cce_add_kernel(a, b, c, tiling=tiling)
        torch_npu.npu.synchronize()

        kernel_dir = _kernel_aicore_dir(work_path)
        cpp_files = _list_cce_cpp(kernel_dir)
        result_queue.put(
            {
                "ok": True,
                "kernel_dir": str(kernel_dir.resolve()),
                "cpp_files": [str(p) for p in cpp_files],
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface any failure to parent
        result_queue.put({"ok": False, "error": repr(exc)})


def _launch_once(work_path: str, device_id: int) -> dict:
    result_queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_run_kernel_once, args=(work_path, device_id, result_queue))
    proc.start()
    proc.join(timeout=600)
    assert proc.exitcode == 0, f"child exited with code {proc.exitcode}"
    assert not result_queue.empty(), "child returned no result"
    result = result_queue.get()
    assert result.get("ok"), f"child failed: {result}"
    return result


def _inject_marker(cpp_files: list[str], marker: str) -> None:
    assert cpp_files, "no CCE cpp to inject marker"
    for path in cpp_files:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        p.write_text(marker + "\n" + text, encoding="utf-8")


def _assert_marker_preserved(cpp_files: list[str], marker: str) -> None:
    assert cpp_files, "no CCE cpp after second run"
    missing = [p for p in cpp_files if marker not in Path(p).read_text(encoding="utf-8", errors="replace")]
    assert not missing, f"marker lost after second compile (CCE overwritten?): {missing}"


@pytest.mark.soc("910", "950")
def test_fixed_cce_mode_does_not_overwrite_existing_cce():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    marker = _MARKER_PREFIX + uuid.uuid4().hex

    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory(prefix="pypto_fixed_cce_") as tmp_dir:
        work_path = os.path.join(tmp_dir, "ascend_work")
        os.makedirs(work_path, exist_ok=True)

        first = _launch_once(work_path, device_id)
        kernel_dir = Path(first["kernel_dir"])
        expect_dir = _kernel_aicore_dir(work_path).resolve()
        assert kernel_dir == expect_dir, f"CCE dir {kernel_dir} != fixed path {expect_dir}"
        assert kernel_dir.is_dir(), f"missing fixed CCE dir: {kernel_dir}"
        assert first["cpp_files"], f"no CCE cpp under {kernel_dir}"

        _inject_marker(first["cpp_files"], marker)

        second = _launch_once(work_path, device_id)
        assert Path(second["kernel_dir"]).resolve() == expect_dir
        _assert_marker_preserved(second["cpp_files"], marker)
