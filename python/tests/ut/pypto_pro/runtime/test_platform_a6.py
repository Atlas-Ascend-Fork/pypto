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
"""A6 arch detection."""

from pypto_pro.runtime.jit import _setup_arch_env
from pypto_pro.runtime.platform import PlatformInfo

from pypto import pypto_impl
from pypto.pypto_impl import NpuArch


def test_a6_runtime_arch_mapping(monkeypatch):
    original_arch = pypto_impl.GetNPUArch()
    try:
        pypto_impl.SetNPUArch("9201")
        assert pypto_impl.GetNPUArch() == "DAV_9201"
        assert PlatformInfo(soc_version=pypto_impl.GetNPUArch()).arch == NpuArch.DAV_9201
    finally:
        pypto_impl.SetNPUArch(original_arch.removeprefix("DAV_"))

    monkeypatch.delenv("PYPTOPRO_JIT_ARCH", raising=False)
    assert _setup_arch_env(NpuArch.DAV_9201) == NpuArch.DAV_9201
    assert pypto_impl.NpuArch.DAV_9201 == NpuArch.DAV_9201
