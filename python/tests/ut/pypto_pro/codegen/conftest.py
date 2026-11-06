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
"""Capture the complete IR and CCE used by the 9201 codegen regressions."""

import importlib

import pytest


@pytest.fixture
def codegen_9201(monkeypatch, assert_9201_golden):
    jit_module = importlib.import_module("pypto_pro.runtime.jit")

    def codegen(kernel_def, arch):
        original = jit_module._codegen_target_cce
        captured = []

        def checked(program, *args, **kwargs):
            target = kwargs["target"].name.lower()
            assert_9201_golden(program, f"{target}-ir")
            result = original(program, *args, **kwargs)
            assert_9201_golden(result.content, f"{target}-cce")
            captured.append(target)
            return result

        with monkeypatch.context() as patch:
            patch.setenv("PYPTOPRO_JIT_ARCH", str(arch))
            patch.setattr(jit_module, "_codegen_target_cce", checked)
            results = jit_module._parse_and_codegen_targets(kernel_def, arch, "")
        assert captured == [target for target, result in zip(("cube", "vector"), results) if result is not None]
        return results

    return codegen
