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
"""Extracted from language/parser/test_vf_scope.py.

Original case: test_lookalike_callable_is_not_mistaken_for_a_declaration.

The sibling ``.pypto`` file holds the ``__str__`` (TextDump) output.
"""

from pathlib import Path

from pypto_pro import ir
import pypto_pro.language as pl


def test_lookalike_callable_is_not_mistaken_for_a_declaration():
    """Bare names resolve by identity against the declarations, not by their __module__ string.

    Renaming ``_vf_api``/``_api`` is the case that matters and no test can reach it: a string
    comparison would simply stop matching and reopen the alias bypass with no diagnostic. A
    callable that only *reports* the declaration module stands in for it -- matching on the
    string alone would take this ordinary helper for vf.create_mask and reject the kernel.
    """

    def create_mask():
        return 7

    create_mask.__module__ = "pypto_pro.language._vf_api"

    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64):
        value = create_mask()  # noqa: F841

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    func = program.get_function(kernel.__name__)
    assert func is not None

    printed = str(func)
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
