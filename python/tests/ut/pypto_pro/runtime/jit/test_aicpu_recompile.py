# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AICPU recompile decision: the shared variant cache.

AICPU kernels share the per-instance signature cache (``_compiled_by_signature``);
their tilingkey/datatype are rejected at decoration, so the key degenerates to
``(static_signature, None, None)``: a declared-static dimension
change selects a new variant and its own artifact dir, while a dynamic-axis change
reuses the cached one. These tests drive the real bind + cache path with the device
build stubbed out (``aicpu_compile.build_aicpu_jit_so`` is monkeypatched), so no
NPU or bisheng toolchain is needed.
"""

from importlib import import_module
import os
from pathlib import Path
from unittest.mock import MagicMock

import pypto_pro.language as pl
import pytest

jit = import_module("pypto_pro.runtime.jit")

_torch_mock = MagicMock()


class _MockTensor:
    """Minimal stand-in for torch.Tensor: only shape/dtype reach the bind path."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def data_ptr() -> int:
        return 0


# Assign the concrete class so isinstance(x, torch.Tensor) works in jit code.
_torch_mock.Tensor = _MockTensor


@pytest.fixture(autouse=True)
def _mock_torch(monkeypatch):
    """Inject mock torch for the duration of each test only."""
    import torch

    monkeypatch.setattr(jit, "torch", _torch_mock)
    monkeypatch.setattr(_torch_mock, "int32", torch.int32)


@pytest.fixture()
def aicpu_kernel():
    @pl.jit(mode="aicpu_kernel")
    def k(
        x: pl.Tensor[[pl.STATIC], pl.DT_INT32, pl.Input],
        out: pl.Tensor[[pl.STATIC], pl.DT_INT32, pl.Output],
        n: pl.DT_INT32 = 0,
    ):
        for i in pl.range(n):
            pl.setval(out, i, pl.getval(x, i))

    return k


@pytest.fixture()
def builds(monkeypatch, tmp_path, aicpu_kernel):
    """Stub the device build; record one (op_name, out_dir) per compile call."""
    calls = []

    def _fake_build(out_dir, op_name, cpp_path, host_path, so_path, arch, timeout):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(so_path).write_bytes(b"")  # CompiledKernel only stores the path string
        calls.append((op_name, out_dir))

    import pypto_pro.runtime.aicpu_compile as aicpu_compile

    monkeypatch.setattr(aicpu_compile, "build_aicpu_jit_so", _fake_build)

    return calls


def _launch(aicpu_kernel, n):
    x = _MockTensor([n], _torch_mock.int32)
    out = _MockTensor([n], _torch_mock.int32)
    return aicpu_kernel._ensure_compiled((x, out, n))


def test_static_axis_change_recompiles(aicpu_kernel, builds):
    first = _launch(aicpu_kernel, 4)
    second = _launch(aicpu_kernel, 8)
    assert first is not second, "a STATIC axis change must compile a new variant"
    assert len(builds) == 2
    assert {os.path.normpath(out) for _, out in builds} == {
        os.path.normpath(first.build_dir),
        os.path.normpath(second.build_dir),
    }, "each static shape gets its own artifact dir"
    assert os.path.normpath(first.build_dir) != os.path.normpath(second.build_dir)


def test_replay_hits_cache(aicpu_kernel, builds):
    first = _launch(aicpu_kernel, 4)
    again = _launch(aicpu_kernel, 4)
    assert first is again, "same static signature must reuse the cached variant"
    assert len(builds) == 1
    assert len(aicpu_kernel._compiled_by_signature) == 1


@pytest.fixture()
def dynamic_kernel():
    @pl.jit(mode="aicpu_kernel")
    def k(
        x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
        out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
        n: pl.DT_INT32 = 0,
    ):
        for i in pl.range(n):
            pl.setval(out, i, pl.getval(x, i))

    return k


def _launch_dynamic(dynamic_kernel, n):
    x = _MockTensor([n], _torch_mock.int32)
    out = _MockTensor([n], _torch_mock.int32)
    return dynamic_kernel._ensure_compiled((x, out, n))


def test_dynamic_axis_change_reuses_cache(dynamic_kernel, builds):
    first = _launch_dynamic(dynamic_kernel, 4)
    again = _launch_dynamic(dynamic_kernel, 8)
    assert first is again, "a DYNAMIC axis change must reuse the cached variant"
    assert len(builds) == 1
    assert len(dynamic_kernel._compiled_by_signature) == 1


def test_cache_key_is_static_signature(aicpu_kernel, builds):
    _launch(aicpu_kernel, 4)
    _launch(aicpu_kernel, 8)
    assert set(aicpu_kernel._compiled_by_signature) == {
        (((0, 0, 4), (1, 0, 4)), None, None),
        (((0, 0, 8), (1, 0, 8)), None, None),
    }
