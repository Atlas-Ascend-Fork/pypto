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
"""Cross-core INTRA_BLOCK lowering follows the arch's Cube/Vector ratio."""

import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import NpuArch, ir
from pypto.pypto_impl.codegen import CCECodegen


@pl.jit
def _static_sync_kernel(x: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_cube():
        pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=3)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=3)


@pl.jit
def _dynamic_sync_kernel(x: pl.Tensor[[1], pl.DT_INT32], event_id: pl.DT_INT32):
    with pl.section_cube():
        pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=event_id)
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=event_id)


@pytest.mark.parametrize("kernel", [_static_sync_kernel, _dynamic_sync_kernel])
@pytest.mark.parametrize("arch,expected_calls", [(NpuArch.DAV_3510, 2), (NpuArch.DAV_9201, 1)])
def test_cube_intra_block_sync_uses_cv_ratio(kernel, arch, expected_calls, a6_codegen):
    cube, _ = a6_codegen(kernel.to_kernel_def(), arch)
    cpp = cube.content

    assert cpp.count("set_intra_block(PIPE_S,") == expected_calls
    assert cpp.count("wait_intra_block(PIPE_S,") == expected_calls
    assert ("+ 16);" in cpp or "19);" in cpp) == (arch == NpuArch.DAV_3510)
    assert "set_ffts_base_addr" not in cpp


@pytest.mark.parametrize("arch", [NpuArch.DAV_3510, NpuArch.DAV_9201])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("mode", [pl.CrossCoreSyncMode.INTER_BLOCK, pl.CrossCoreSyncMode.INTER_SUBBLOCK])
def test_cross_block_sync_codegen_matches_jit(arch, dynamic, mode, monkeypatch, a6_codegen, assert_a6_golden):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", str(arch))

    @pl.jit(auto_mutex=False)
    def kernel(event_id: pl.DT_INT32):
        with pl.section_cube():
            if dynamic:
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=event_id, sync_mode=mode)
            else:
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3, sync_mode=mode)
        with pl.section_vector():
            if dynamic:
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=event_id, sync_mode=mode)
            else:
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3, sync_mode=mode)

    kernel_def = kernel.to_kernel_def()
    cube, vector = a6_codegen(kernel_def, arch)
    for target, result in ((ir.SectionKind.Cube, cube), (ir.SectionKind.Vector, vector)):
        program, _ = kernel_def.parse_target_program(target)
        assert_a6_golden(program, f"{target.name.lower()}-ir")
        raw = CCECodegen(target).generate_single(program, arch)
        assert "ffts_cross_core_sync" in raw
        assert result.content == raw
        assert "TSyncCVID.hpp" not in raw
