# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AICPU entry/launcher qualifier: ``__aicpu__`` instead of ``__global__``.

A lite-AICPU kernel is a control-CPU function: the generated entry uses the
``__aicpu__`` attribute and ``call_kernel`` calls it directly (no device launch
geometry). AICore entries keep ``__global__`` + the target qualifier.
"""
from __future__ import annotations

from types import SimpleNamespace

from pypto_pro.runtime.jit import _generate_caller_cpp, _make_global_entry

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
