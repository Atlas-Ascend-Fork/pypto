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
"""Tests for the whole-model run path: ``common.run``'s run context, the DIRECT re-wrap, and the
run-mode/soc guards, plus the ``build_jit`` soc contract those guards depend on.

Covers: context propagation and clear-on-exit; ``_rewrap_direct`` overriding only ``run_mode`` (pinning
``soc_version`` only when given, tracked against ``jit()``'s live signature so upstream drift is caught);
``run_op``'s mode/device/soc validation; and ``build_jit(soc_version=None)`` forwarding ``None`` so the
framework auto-detects on-device. Needs the pypto backend on the box.
"""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

import pypto
from pypto.extensions.torch_custom_op_litenpu.common import run as _run
from pypto.extensions.torch_custom_op_litenpu.common.compile import CompileEntry

_TEST_DIR = Path(__file__).resolve().parent


def _load(unique_name, filename):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    path = _TEST_DIR / filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


samples = _load("kernel_snippet_samples_run_ut", "kernel_snippet_samples.py")


class _FakeOp:
    """Minimal stand-in exposing only what _rewrap_direct / run_op guards read."""

    def __init__(self, bare_kernel_fn=None, create_kernel_fn=None):
        self._bare_kernel_fn = bare_kernel_fn
        self._create_kernel_fn = create_kernel_fn


def test_run_context_sets_run_mode():
    # pypto_run_context(run_mode=…) builds a _RunContext carrying that RunMode; no context after the block.
    assert _run.get_run_context() is None
    with _run.pypto_run_context(run_mode=pypto.RunMode.SIM, soc_version="Kirin9030") as ctx:
        assert isinstance(ctx, _run._RunContext)
        assert ctx.run_mode == pypto.RunMode.SIM
        assert ctx.soc_version == "Kirin9030"
        assert _run.get_run_context() is ctx
    assert _run.get_run_context() is None


def test_rewrap_direct_injects_soc_only_when_given():
    # DIRECT re-wrap overrides run_mode ALWAYS; injects soc_version ONLY when explicitly given.
    op = _FakeOp(bare_kernel_fn=samples.direct_add_kernel_erased)
    # soc given -> pinned in codegen_options.
    w_soc = _run._rewrap_direct(op, "Kirin9030", pypto.RunMode.SIM)
    assert w_soc._codegen_options.get("soc_version") == "Kirin9030"
    assert w_soc._runtime_options["run_mode"] == pypto.RunMode.SIM
    # soc None -> NO soc_version key pinned (framework auto-detects on-device).
    w_none = _run._rewrap_direct(op, None, pypto.RunMode.NPU)
    assert "soc_version" not in (w_none._codegen_options or {})
    assert w_none._runtime_options["run_mode"] == pypto.RunMode.NPU


def test_rewrap_direct_preserves_author_options():
    # Author's non-soc decorator options are carried through unchanged; run_mode is overridden.
    op = _FakeOp(bare_kernel_fn=samples.direct_add_kernel_erased)
    src = samples.direct_add_kernel_erased
    w = _run._rewrap_direct(op, None, pypto.RunMode.SIM)
    assert w._pass_options == src._pass_options
    assert w._verify_options == src._verify_options
    assert w._debug_options == src._debug_options
    assert w._use_new_ir == src._use_new_ir


def test_collect_decorator_options_tracks_jit_signature():
    """Guard against upstream jit() signature drift.

    The carried option set is DERIVED from jit()'s live signature, so: a storable ADDITION is carried
    automatically (green, by design); a non-storable addition raises inside _collect_decorator_options
    (red); a REMOVAL or RENAME drops out of the frozen baseline below (red). NB the packed-snippet path
    (kernel_snippet) cannot drift the same way: it AST-edits the author's verbatim decorator text and only
    forces runtime_options["run_mode"], never enumerating jit parameters.
    """
    import inspect
    sig = inspect.signature(pypto.frontend.jit)
    expected = {
        n for n, p in sig.parameters.items()
        if n != "func" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    collected = set(_run._collect_decorator_options(samples.direct_add_kernel_erased))
    assert collected == expected
    assert {
        "host_options", "codegen_options", "pass_options", "runtime_options",
        "verify_options", "debug_options", "new_ir",
    } <= collected

    # Re-decorates the SAME function with non-default values and reads them back, proving a scalar and
    # a dict option survive with the author's VALUE, not just their key. run_mode is pinned because with
    # none, _set_run_mode auto-detects NPU from ASCEND_HOME_PATH and raises off-box, which would fail this
    # option-only assertion.
    probe = pypto.frontend.jit(
        debug_options={"runtime_debug_mode": True},
        runtime_options={"run_mode": pypto.RunMode.SIM},
    )(samples.direct_add_kernel_erased._original_func)
    probe_options = _run._collect_decorator_options(probe)
    assert probe_options["debug_options"] == {"runtime_debug_mode": True}


def test_run_op_npu_no_device_raises(monkeypatch):
    # run_op(run_mode=NPU) with no NPU available -> RunEnvError before any build (device-presence guard).
    import torch
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "is_available", lambda: False, raising=False)
    else:
        class _NoNpu:
            @staticmethod
            def is_available():
                return False
        monkeypatch.setattr(torch, "npu", _NoNpu(), raising=False)
    op = _FakeOp(bare_kernel_fn=samples.direct_add_kernel_erased)
    with pytest.raises(_run.RunEnvError):
        _run.run_op(op, [torch.zeros(4), torch.zeros(4)], run_mode=pypto.RunMode.NPU, soc_version=None)


def test_run_op_sim_requires_soc():
    # run_op(run_mode=SIM, soc_version=None) -> RunEnvError (sim needs an explicit soc; no device to detect).
    import torch
    op = _FakeOp(bare_kernel_fn=samples.direct_add_kernel_erased)
    with pytest.raises(_run.RunEnvError):
        _run.run_op(op, [torch.zeros(4), torch.zeros(4)], run_mode=pypto.RunMode.SIM, soc_version=None)


def test_run_op_rejects_invalid_run_mode():
    import torch
    op = _FakeOp(bare_kernel_fn=samples.direct_add_kernel_erased)
    with pytest.raises(ValueError):
        _run.run_op(op, [torch.zeros(4)], run_mode="npu")  # run_mode must be a pypto.RunMode, not a string


# factory build_jit must not pin a default soc onto a None soc on the run path.

def _add_factory_entry():
    entry = CompileEntry(
        num_inputs=2,
        num_outputs=1,
        factory=samples.create_add_kernel,
        factory_signature="single",
    )
    entry.infer_shape = samples.add_infer_shape
    entry.infer_dtype = samples.add_infer_dtype
    return entry


def test_build_jit_none_soc_not_pinned():
    # build_jit(soc_version=None) must forward None UNCHANGED to the factory (framework then auto-detects),
    # NOT pin a default SIM soc (e.g. Kirin9030) onto it. create_add_kernel writes the soc into its inner
    # jit's codegen_options, so the value must be None (never "Kirin9030"). set_codegen_options drops the
    # None key at BUILD time, so the framework auto-detects.
    entry = _add_factory_entry()
    jit_obj, _, _ = entry.build_jit(
        [(4,), (4,)], ["float32", "float32"], soc_version=None, run_mode=pypto.RunMode.SIM,
    )
    assert (jit_obj._codegen_options or {}).get("soc_version") is None
    assert (jit_obj._codegen_options or {}).get("soc_version") != "Kirin9030"


def test_build_jit_explicit_soc_pinned():
    # An explicit soc is still pinned on the factory inner jit.
    entry = _add_factory_entry()
    jit_obj, _, _ = entry.build_jit(
        [(4,), (4,)], ["float32", "float32"], soc_version="Kirin9030", run_mode=pypto.RunMode.SIM,
    )
    assert jit_obj._codegen_options.get("soc_version") == "Kirin9030"


# Runs in a fresh interpreter: the backend's platform info is process-global and first-touch-wins, so
# an earlier Kirin9030/LiteNPU build in this process (e.g. test_from_jit_kernel) would pin it, making a
# later real-NPU launch read the Lite platform INI (no ``CCEC_AIV_version`` -> ``F21003
# FeError::INVALID_VAL``). Targets the CURRENT device (bare "npu", no ordinal); skipped off-box.


def _current_device_scenario():
    """The check body, executed by the ``__main__`` block below in the child interpreter."""
    import torch
    target = 1 if torch.npu.device_count() > 1 else 0
    torch.npu.set_device(target)
    op = _FakeOp(create_kernel_fn=samples.create_add_kernel)
    op._infer_shape_fn = samples.add_infer_shape
    op._infer_dtype_fn = samples.add_infer_dtype
    op._bare_kernel_fn = None
    out = _run.run_op(
        op, [torch.zeros(4), torch.zeros(4)], run_mode=pypto.RunMode.NPU, soc_version=None,
    )
    # bare "npu" (no index in run_op) must resolve to the SET current device.
    assert out.device.type == "npu", out.device
    assert out.device.index == target, (out.device, target)


def test_run_op_npu_targets_current_device():
    import torch
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("the current-device check needs a real NPU device")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"current-device child interpreter failed (rc={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


if __name__ == "__main__":
    # Child entry point for test_run_op_npu_targets_current_device (see its comment): a fresh
    # interpreter whose platform info is unpinned, so the real-NPU launch picks the on-device soc.
    _current_device_scenario()
