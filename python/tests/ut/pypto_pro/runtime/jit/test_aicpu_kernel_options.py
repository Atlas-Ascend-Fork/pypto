# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND OF EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AICPU kernel decorator-option and bracket-form contract tests.

Two groups, no device needed (decoration and ``__getitem__`` raise before any
compile/launch):

* decorator options -- ``arch``/``name``/``compile_timeout``/``max_threads`` are
  accepted on ``@pl.jit(mode="aicpu_kernel")``; ``pipeline``/``tiling_key``/
  ``datatype``/``sanitizer``/``sk_cap``/``schedmode`` fail fast with the option
  names in the error message.
* bracket forms -- ``kernel[stream]``/``kernel[None]`` build a launcher;
  ``kernel[stream, block_dim]`` is rejected (the AICPU shim hardcodes
  ``<<<1>>>``, so a block_dim would be validated and silently dropped).
"""

import os

from pypto_pro._errors import InvalidArgument, NotSupported
import pypto_pro.language as pl
from pypto_pro.runtime.jit import _TileJitKernel
import pytest


def _decorate(**kwargs):
    """Decorate a minimal aicpu kernel with the given options (no launch)."""

    @pl.jit(mode="aicpu_kernel", **kwargs)
    def k(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
          y: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output]):
        if x is not None:
            pl.setval(y, 0, pl.getval(x, 0))

    return k


class TestAcceptedOptions:
    """arch / name / compile_timeout / max_threads decorate successfully."""

    @pytest.mark.parametrize("kwargs", [
        {},
        {"arch": "3510"},
        {"name": "renamed_aicpu"},
        {"compile_timeout": 120},
        {"max_threads": 256},
        {"arch": "3510", "name": "renamed_aicpu", "compile_timeout": 60, "max_threads": 128},
    ])
    def test_accepted(self, kwargs):
        kernel = _decorate(**kwargs)
        assert isinstance(kernel, _TileJitKernel)
        assert kernel._func_type.name == "AicpuKernel"

    def test_name_flows_to_symbol_field(self):
        kernel = _decorate(name="renamed_aicpu")
        assert kernel._name == "renamed_aicpu"


class TestRejectedOptions:
    """The six tile/SIMT-only options fail fast, naming every offender."""

    @pytest.mark.parametrize("kwargs, option", [
        ({"pipeline": object()}, "pipeline"),
        ({"tiling_key": {"A": 1}}, "tiling_key"),
        ({"datatype": {"x": "fp16"}}, "datatype"),
        ({"sanitizer": True}, "sanitizer"),
        ({"sk_cap": pl.SkCap.CAP_4_DISABLE_DCCI_ALL}, "sk_cap"),
        ({"schedmode": 1}, "schedmode"),
    ])
    def test_rejected_names_the_option(self, kwargs, option):
        with pytest.raises(NotSupported, match=option):
            _decorate(**kwargs)

    def test_rejected_lists_all_offenders(self):
        with pytest.raises(NotSupported) as exc_info:
            _decorate(datatype={"x": "fp16"}, sanitizer=True, schedmode=1, sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
        message = str(exc_info.value)
        for option in ("datatype", "sanitizer", "schedmode", "sk_cap"):
            assert option in message


class TestBracketForms:
    """Stream-only bracket for aicpu; block_dim form rejected."""

    def test_bracket_stream_and_none(self):
        kernel = _decorate()
        assert callable(kernel[None])
        assert callable(kernel[(None,)])
        assert callable(kernel[torch_stream])

    def test_bracket_block_dim_rejected(self):
        kernel = _decorate()
        with pytest.raises(InvalidArgument, match="block_dim has no effect"):
            kernel[(torch_stream, 1)]

    def test_bracket_arity_rejected(self):
        kernel = _decorate()
        with pytest.raises(InvalidArgument, match="Expected kernel\\[stream\\] for an aicpu kernel"):
            kernel[(torch_stream, 1, {"K": 1})]


class TestAcceptedOptionsStillRun:
    """On-device smoke: accepted options do not break the fast-launch path."""

    def test_arch_name_compile_timeout_launch(self):
        pytest.importorskip("torch_npu")
        import torch

        if not torch.npu.is_available():
            pytest.skip("NPU device not available")
        torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
        # Probe the CANN runtime before the smoke: a present-but-broken CANN install
        # (e.g. ASCEND_OPP_PATH unset) passes is_available() and then fails every
        # device op with a RuntimeError -- skip instead of FAIL on such hosts.
        try:
            torch.zeros(1, dtype=torch.int32, device="npu")
        except RuntimeError:
            pytest.skip("CANN runtime not usable on this host (device op probe failed)")

        kernel = _decorate(arch="3510", name="opt_aicpu", compile_timeout=300)
        t = torch.tensor([1, 2, 3], dtype=torch.int32, device="npu")
        o = torch.zeros(8, dtype=torch.int32, device="npu")
        kernel(t, o)
        torch.npu.synchronize()
        assert o.cpu().tolist()[0] == 1


torch_stream = None  # replaced by a real stream when torch is importable
try:
    import torch
    import torch_npu  # noqa: F401

    torch_stream = torch.npu.current_stream()
except Exception:  # noqa: BLE001 -- no CANN env in UT runners
    pass
