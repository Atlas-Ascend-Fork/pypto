# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AICPU kernel codegen, entry ABI, and the recompile decision.

Three host-side stages of the AICPU JIT pipeline, no NPU or bisheng needed:

* codegen -- ``generate_aicpu_sources`` renders the kernel translation unit
  and the self-contained host launch translation unit (compat macros + flat
  entry declaration + the ``Run`` shim, which forwards the flattened
  arguments to ``Kernel<<<1, nullptr, stm>>>(args...)``; bisheng ASC dialect,
  the lowering dispatches through the ACL runtime). Full golden comparison
  keeps both TUs drift-free.
* entry -- a lite-AICPU kernel is a control-CPU function: the generated entry
  uses the ``__aicpu__`` attribute and ``call_kernel`` calls it directly (no
  device launch geometry). AICore entries keep ``__global__`` + the target
  qualifier.
* recompile -- AICPU kernels share the per-instance signature cache
  (``_compiled_by_signature``); their tilingkey/datatype are rejected at
  decoration, so the key degenerates to ``(static_signature, None, None)``:
  a declared-static dimension change selects a new variant and its own
  artifact dir, while a dynamic-axis change reuses the cached one. These
  tests drive the real bind + cache path with the device build stubbed out
  (``aicpu_compile.build_aicpu_jit_so`` is monkeypatched).
"""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.runtime.aicpu_compile import compile_aicpu_variant, generate_aicpu_bare_body, generate_aicpu_sources
from pypto_pro.runtime.jit import (
    _extract_param_specs,
    _generate_caller_cpp,
    _get_kernel_ir_function,
    _make_global_entry,
)
import pytest

# The runtime package re-exports the `jit` decorator, shadowing the module name;
# import_module still resolves the module object for attribute patching.
jit = import_module("pypto_pro.runtime.jit")


# ---------------------------------------------------------------------------
# Codegen: fast-launch sources (fast_kernel_launch_example shape)
# ---------------------------------------------------------------------------


def _generate(kernel):
    kdef = kernel.to_kernel_def()
    program, _ = kdef.parse_target_program(ir.SectionKind.Aicpu)
    specs = _extract_param_specs(program, None)
    name = _get_kernel_ir_function(program).name
    body = generate_aicpu_bare_body(program, __import__("pypto_pro").runtime.platform.NpuArch.DAV_3510, name)
    return generate_aicpu_sources(body, specs, name)


@pytest.mark.parametrize("delivery", ["fast_launch", "offline"])
def test_custom_name_matches_impl_call(delivery, monkeypatch, tmp_path):
    @pl.jit(mode="aicpu_kernel", arch="3510", name="opt_aicpu")
    def original_kernel(
        x: pl.Tensor[[4], pl.DT_INT32, pl.Input],
        out: pl.Tensor[[4], pl.DT_INT32, pl.Output],
    ):
        pl.setval(out, 0, pl.getval(x, 0))

    if delivery == "fast_launch":
        monkeypatch.setenv("ASCEND_WORK_PATH", str(tmp_path))
        sources = []

        def fake_build(out_dir, op_name, cpp_path, host_path, so_path, arch, timeout):
            sources.append(cpp_path.read_text(encoding="utf-8"))

        monkeypatch.setattr("pypto_pro.runtime.aicpu_compile.build_aicpu_jit_so", fake_build)
        compile_aicpu_variant(original_kernel, None, (None, None, None, None), None, ())
        source, = sources
        call = "return opt_aicpu_impl_aicpu("
    else:
        from pypto_pro.runtime.opc import aclnn_aicpu_codegen

        monkeypatch.setattr(aclnn_aicpu_codegen, "_setup_arch_env", lambda arch: arch)
        result = aclnn_aicpu_codegen._prepare_from_kernel(
            original_kernel, "3510", None, tmp_path, "libprobe.so",
        )
        source = Path(result.kernel_cpp_path).read_text(encoding="utf-8")
        call = "uint32_t status = opt_aicpu_impl_aicpu("
    assert "inline uint32_t opt_aicpu_impl_aicpu(" in source
    assert call in source


def test_generates_fast_launch_host_source():
    @pl.jit(mode="aicpu_kernel")
    def kernel(
        t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
        out: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT32, pl.Output],
        n: pl.DT_INT32 = 0,
    ):
        for i in pl.range(n):
            pl.setval(out, i, pl.getval(t, i))

    sources = _generate(kernel)
    kernel_src = sources["kernel_source"]
    host_src = sources["host_source"]

    # Full golden comparison (PRO-TEST-002): any line drift in any of the two
    # generated TUs fails here -- substring spot-checks cannot catch added/removed
    # lines or formatting drift.
    golden_dir = Path(__file__).parent
    for artifact, golden_name in (
        (kernel_src, "test_aicpu_codegen_compile__kernel_source.g.txt"),
        (host_src, "test_aicpu_codegen_compile__host_source.g.txt"),
    ):
        golden = (golden_dir / golden_name).read_text(encoding="utf-8")
        assert artifact.rstrip("\n") == golden.rstrip("\n"), f"artifact drifted from {golden_name}"

    # Key contract spot-checks (kept as human-readable anchors on top of the goldens).
    # Flattened launch ABI: no packed Args struct or void* raw on either side.
    assert "kernelKernel(void* raw)" not in kernel_src
    assert "struct kernelArgs" not in host_src
    assert 'extern "C" __global__ __aicpu__ uint32_t kernelKernel(int32_t* t_0,' in kernel_src
    # Bare-body contract: no device qualifiers or CCE scaffolding anywhere in the TU.
    for token in ("__gm__", "__aicore__", "#if defined(__DAV", "SyncedTile", "pypto_dump.h"):
        assert token not in kernel_src, f"bare body leaked {token}"
    # Host TU: the flattened <<<1, nullptr, stm>>> fast-launch form.
    assert (
        "kernelKernel<<<1, nullptr, static_cast<aclrtStream>(stm)>>>("
        "t_0, out_0, n_0, __pypto_dyn_t_0, __pypto_dyn_out_0);" in host_src
    )
    assert "sizeof(kernelArgs)" not in host_src

    if shutil.which("g++"):
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp, "kernel_aicpu.cpp")
            cpp.write_text(kernel_src)
            # Only the kernel TU is g++-checkable: the host TU carries the
            # <<<>>> compiler extension and the acl headers (bisheng only).
            subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", f"-I{tmp}", str(cpp)],
                check=True,
                capture_output=True,
                text=True,
            )


def test_status_abi_emission():
    @pl.jit(mode="aicpu_kernel")
    def status_kernel(
        t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
        out: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT32, pl.Output],
        n: pl.DT_INT32 = 0,
    ):
        if n < 0:
            return 1
        if n == 5:
            pl.setval(out, 0, 5)
            return
        for i in pl.range(n):
            pl.setval(out, i, pl.getval(t, i))

    sources = _generate(status_kernel)
    src = sources["kernel_source"]

    # The impl carries the status ABI: uint32_t return, explicit `return 1;`
    # failure exit, bare returns (branch + fall-through) as `return 0;`.
    assert "inline uint32_t status_kernel_impl_aicpu(" in src
    assert src.count("return 1;") == 1
    # one bare branch return + the fall-through tail
    assert src.count("return 0;") == 2
    assert "return;\n" not in src
    # The __global__ entry propagates the status instead of hardcoding 0.
    assert "return status_kernel_impl_aicpu(" in src

    if shutil.which("g++"):
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp, "status_kernel_aicpu.cpp")
            cpp.write_text(src)
            subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", f"-I{tmp}", str(cpp)],
                check=True,
                capture_output=True,
                text=True,
            )


# ---------------------------------------------------------------------------
# Entry/launcher qualifier: ``__aicpu__`` instead of ``__global__``
# ---------------------------------------------------------------------------

_PARAMS = [("int32_t", "x", True), ("int32_t", "n", False)]
_TARGET = SimpleNamespace(entry_qualifier="__mix__(1, 2)", aic_per_block=1, aiv_per_block=2)


def test_aicpu_entry_uses_aicpu_qualifier_and_direct_call():
    entry = _make_global_entry("k", _PARAMS, target=_TARGET, aicpu_kernel=True)
    assert entry.startswith("__aicpu__ void k(")
    assert "__global__" not in entry

    caller = _generate_caller_cpp(
        _PARAMS, "kernel.cpp", "k", global_entry=entry, target=_TARGET, aicpu_kernel=True
    )
    assert "__aicpu__ void k(" in caller
    assert "k<<<" not in caller
    assert "    k((int32_t *)x, n);" in caller
    assert "return 0;" in caller


def test_aicore_entry_keeps_global_qualifier():
    entry = _make_global_entry("k", _PARAMS, target=_TARGET)
    assert entry.startswith('extern "C" __global__ __mix__(1, 2) void k(')


# ---------------------------------------------------------------------------
# Recompile decision: the shared variant cache
# ---------------------------------------------------------------------------

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


@pytest.fixture()
def builds(monkeypatch, tmp_path, aicpu_kernel):
    """Stub torch and the device build; record one (op_name, out_dir) per compile call."""
    import torch

    monkeypatch.setattr(jit, "torch", _torch_mock)
    monkeypatch.setattr(_torch_mock, "int32", torch.int32)

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


def _launch_dynamic(dynamic_kernel, n):
    x = _MockTensor([n], _torch_mock.int32)
    out = _MockTensor([n], _torch_mock.int32)
    return dynamic_kernel._ensure_compiled((x, out, n))


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
