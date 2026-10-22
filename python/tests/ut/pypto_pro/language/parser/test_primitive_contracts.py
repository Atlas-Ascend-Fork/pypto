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
"""Public scalar aliases and core-side barriers have complete DSL contracts."""

from pypto_pro import ir
from pypto_pro._errors import InvalidOperation, NotSupported
import pypto_pro.language as pl
import pytest


@pl.jit(arch="3510", auto_mutex=False)
def scalar_aliases(x: pl.Tensor[[1], pl.DT_FP32]):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0)
    b = pl.make_tile(tt, addr=256)
    with pl.section_vector():
        pl.adds(b, a, 0.5)
        pl.subs(b, b, 0.25)
        pl.muls(b, b, 2.0)


def test_public_aliases_parse_and_export():
    program, _ = scalar_aliases.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    text = ir.python_print(program)
    for name in ("adds", "subs", "muls"):
        assert name in pl.__all__ and callable(getattr(pl, name))
        assert "block." + name in text
        with pytest.raises(InvalidOperation):
            getattr(pl, name)(None, None, 1)


@pl.jit(arch="3510", auto_mutex=False)
def cube_barriers(x: pl.Tensor[[1], pl.DT_FP32]):
    with pl.section_cube():
        pl.system.bar_m()
        pl.system.bar_mte1()
        pl.system.bar_fix()


def _vector_barrier(which):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(x: pl.Tensor[[1], pl.DT_FP32]):
        with pl.section_vector():
            if which == 0:
                pl.system.bar_m()
            elif which == 1:
                pl.system.bar_mte1()
            else:
                pl.system.bar_fix()
    return kernel


def test_cube_barriers_remain_supported():
    assert cube_barriers.to_kernel_def().parse_target_program(ir.SectionKind.Cube)[1]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_cube_only_barriers_reject_vector_target_with_location(which):
    with pytest.raises(NotSupported) as caught:
        _vector_barrier(which).to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "requires a Cube section" in caught.value.message
    assert caught.value.span["filename"] == __file__
