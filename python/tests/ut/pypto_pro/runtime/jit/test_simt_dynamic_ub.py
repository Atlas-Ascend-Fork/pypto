# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression tests for parser-collected SIMT dynamic UB inference."""

import importlib
from types import SimpleNamespace

import pypto_pro.language as pl
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl import ir

jit = importlib.import_module("pypto_pro.runtime.jit")


@pytest.mark.parametrize(
    ("requires_simt", "tile_high_water", "expected_dynamic_ub"),
    (
        (False, 216 * 1024 + 32, 0),
        (False, 256 * 1024, 0),
        (True, 0, 0),
        (True, 8 * 1024, 0),
        (True, 8 * 1024 + 1, 32),
        (True, 8 * 1024 + 32, 32),
        (True, 24 * 1024, 16 * 1024),
        (True, 216 * 1024 - 1, 208 * 1024),
        (True, 216 * 1024, 208 * 1024),
    ),
)
def test_dynamic_ub_inference(requires_simt, tile_high_water, expected_dynamic_ub):
    assert jit._infer_dynamic_ub_size(requires_simt, tile_high_water) == expected_dynamic_ub


@pytest.mark.parametrize(
    "tile_high_water",
    (216 * 1024 + 1, 216 * 1024 + 32, 240 * 1024, 256 * 1024),
)
def test_dynamic_ub_inference_caps_at_3510_limit(tile_high_water):
    assert jit._infer_dynamic_ub_size(True, tile_high_water) == 208 * 1024


def test_parser_collects_vec_high_water_from_tile_group():
    @pl.jit
    def kernel(_src: pl.Tensor[[1, 1], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        _tiles = pl.make_tile_group(type=tile_type, addrs=[2048, 8192], mutex_ids=[0, 1])

    kernel_def = kernel.to_kernel_def()
    kernel_def.parse_target_program(ir.SectionKind.Vector)

    assert kernel_def.max_vec_tile_end == 8192 + 32
    assert not kernel_def.requires_simt


def test_parser_marks_only_the_kernel_that_launches_simt():
    @pl.vector_function(mode="simt", max_threads=1)
    def simt_copy(src, out):
        out[0, 0] = src[0, 0]

    @pl.jit
    def simt_kernel(
        src: pl.Tensor[[1, 1], pl.DT_FP32],
        out: pl.Tensor[[1, 1], pl.DT_FP32],
    ):
        with pl.section_vector():
            simt_copy[1](src, out)

    @pl.jit
    def simd_kernel(_src: pl.Tensor[[1, 1], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        _tiles = pl.make_tile_group(type=tile_type, addrs=24 * 1024 - 32, mutex_ids=[0])

    simt_def = simt_kernel.to_kernel_def()
    simt_def.parse_target_program(ir.SectionKind.Vector)
    simd_def = simd_kernel.to_kernel_def()
    simd_def.parse_target_program(ir.SectionKind.Vector)

    assert simt_def.requires_simt
    assert not simd_def.requires_simt
    assert simd_def.max_vec_tile_end == 24 * 1024


@pytest.mark.parametrize(
    ("cube_high_water", "expected_dynamic_ub"),
    ((216 * 1024, 208 * 1024), (256 * 1024, 208 * 1024)),
)
def test_cube_and_vector_facts_are_aggregated_for_one_kernel(monkeypatch, cube_high_water, expected_dynamic_ub):
    class FakeKernelDef:
        _auto_mutex = False
        func_type = ir.FunctionType.Opaque
        has_tile_function = False

        def __init__(self):
            self.last_param_directions = {}
            self.max_vec_tile_end = 0
            self.requires_simt = False

        def parse_target_program(self, target, bound_signature=None):
            if target == ir.SectionKind.Cube:
                self.max_vec_tile_end = cube_high_water
                self.requires_simt = False
            else:
                self.max_vec_tile_end = 8 * 1024
                self.requires_simt = True
            return ir.Program([], str(target), ir.Span.unknown()), True

    recorded_dynamic_ub = {}

    def fake_codegen(
        prog,
        arch,
        build_dir,
        target,
        declared_directions=None,
        required_dynamic_ub_size=0,
    ):
        recorded_dynamic_ub[target] = required_dynamic_ub_size
        return SimpleNamespace(sanitizer=False)

    monkeypatch.setattr(jit, "_codegen_target_cce", fake_codegen)

    jit._parse_and_codegen_targets(FakeKernelDef(), NpuArch.DAV_3510, "unused")

    assert recorded_dynamic_ub == {
        ir.SectionKind.Cube: expected_dynamic_ub,
        ir.SectionKind.Vector: expected_dynamic_ub,
    }
