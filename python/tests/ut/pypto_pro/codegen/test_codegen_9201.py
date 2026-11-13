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
"""Architecture-specific lowering retained when merging 9201 support into master."""

from dataclasses import dataclass
import re

from pypto_pro import ir
from pypto_pro._errors import NotSupported
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest

from pypto.pypto_impl import NpuArch
from pypto.pypto_impl.codegen import CCECodegen


@dataclass
class CopyTiling:
    total: int = 0


@pl.jit(auto_mutex=False)
def _tiling_copy(x: pl.Ptr[pl.DT_INT32], tiling: CopyTiling):
    with pl.section_vector():
        pl.printf("%d", tiling.total)


@pytest.mark.parametrize("arch,argc", [(NpuArch.DAV_3510, 11), (NpuArch.DAV_9201, 13)])
def test_tiling_copy_intrinsic_signature(arch, argc, monkeypatch, codegen_9201):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", str(arch))
    _, vector = codegen_9201(_tiling_copy.to_kernel_def(), arch)
    call = re.search(r"copy_gm_to_ubuf_align_v2\(([^\n]+)\);", vector.content)
    assert call is not None
    args = [arg.strip() for arg in call.group(1).split(",")]
    assert len(args) == argc
    if arch == NpuArch.DAV_9201:
        assert args[8] == "false"
        assert args[12] == "false"


@pytest.mark.parametrize("arch", [NpuArch.DAV_3510, NpuArch.DAV_9201])
@pytest.mark.parametrize("memory", [pl.MemorySpace.Left, pl.MemorySpace.Right])
@pytest.mark.parametrize("auto_mutex", [False, True])
@pytest.mark.parametrize("with_offset", [False, True], ids=["whole_tile", "explicit_zero_offset"])
def test_mat_to_operand_move_uses_offset_instruction(arch, memory, auto_mutex, with_offset, monkeypatch, codegen_9201):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", str(arch))
    @pl.jit(auto_mutex=auto_mutex)
    def kernel(x: pl.Ptr[pl.DT_FP16]):
        src_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        dst_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=memory)
        if auto_mutex:
            src_group = pl.make_tile_group(type=src_type, addrs=0, mutex_ids=[0])
            dst_group = pl.make_tile_group(type=dst_type, addrs=0, mutex_ids=[1])
        else:
            src_group = pl.make_tile_group(type=src_type, addrs=0, depth=1)
            dst_group = pl.make_tile_group(type=dst_type, addrs=0, depth=1)
        src = src_group.current()
        dst = dst_group.current()
        with pl.section_cube():
            if with_offset:
                pl.move(dst, src, [0, 0])
            else:
                pl.move(dst, src)

    cube, _ = codegen_9201(kernel.to_kernel_def(), arch)
    assert ("get_buf(PIPE_MTE1," in cube.content) == auto_mutex
    instruction = "TEXTRACT" if with_offset else "TMOV"
    assert instruction + "(" in cube.content or instruction + "<" in cube.content
    if with_offset:
        assert re.search(r"TEXTRACT[^;]+, 0, 0\);", cube.content)
    else:
        assert "TEXTRACT" not in cube.content


def test_9201_simd_kernel_codegen_remains_supported(monkeypatch, codegen_9201):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "9201")

    @pl.jit(arch="9201", auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 8], pl.DT_FP32], out: pl.Tensor[[1, 8], pl.DT_FP32]):
        tile_type = pl.TileType(shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0)
        dst = pl.make_tile(tile_type, addr=32)
        with pl.section_vector():
            pl.load(src, x, [0, 0])
            pl.add(dst, src, src)
            pl.store(out, dst, [0, 0])

    kernel_def = kernel.to_kernel_def()
    cube, vector = codegen_9201(kernel_def, NpuArch.DAV_9201)

    assert cube is None
    assert "TADD(" in vector.content or "TADD<" in vector.content
    assert "cce::async_invoke" not in vector.content
    assert vector.required_dynamic_ub_size == 0
    assert not kernel_def.requires_simt


@pl.vector_function
def _register_shift(src, shift, dst):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    value = vf.load_align(src, 0, dtype=pl.DT_INT32)
    amount = vf.load_align(shift, 0, dtype=pl.DT_INT32)
    result = vf.shift_left(value, amount, mask)
    vf.store_align(dst, result, mask)


@pl.jit(auto_mutex=False)
def _shift_kernel(x: pl.Ptr[pl.DT_INT32]):
    tile_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tile_type, addr=0)
    shift = pl.make_tile(tile_type, addr=256)
    dst = pl.make_tile(tile_type, addr=512)
    with pl.section_vector():
        _register_shift(src, shift, dst)


@pytest.mark.parametrize("arch", [NpuArch.DAV_3510, NpuArch.DAV_9201])
def test_register_shift_unsigned_operand_is_9201_only(arch, monkeypatch, assert_9201_golden):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", str(arch))
    program, matched = _shift_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert matched
    assert_9201_golden(program, "vector-ir")
    source = CCECodegen(ir.SectionKind.Vector).generate_single(program, arch)
    assert_9201_golden(source, "vector-cce")
    assert "vshl(" in source
    assert ("(RegTensor<uint32_t>&)" in source) == (arch == NpuArch.DAV_9201)
    params = []
    args = []
    for tile in ("src_0", "shift_0", "dst_0"):
        params.extend([f"__ubuf__ int32_t* {tile}", f"uint32_t {tile}__valid_row", f"uint32_t {tile}__valid_col"])
        args.extend([
            f"(__ubuf__ int32_t *){tile}.tile.data()",
            f"(uint32_t){tile}.tile.GetValidRow()",
            f"(uint32_t){tile}.tile.GetValidCol()",
        ])
    assert f"__simd_vf__ inline void _register_shift({', '.join(params)})" in source
    assert f"_register_shift({', '.join(args)});" in source


@pytest.mark.parametrize("arch", [NpuArch.DAV_3510, NpuArch.DAV_9201])
@pytest.mark.parametrize(
    "mode,shape",
    [(pl.AccToVecMode.DualModeSplitM, [32, 64]), (pl.AccToVecMode.DualModeSplitN, [64, 32])],
)
def test_dual_destination_move_retains_3510_and_rejects_9201(arch, mode, shape, monkeypatch, codegen_9201):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", str(arch))
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Ptr[pl.DT_FP32]):
        src = pl.make_tile(pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc), addr=0)
        dst = pl.make_tile(pl.TileType(shape=shape, dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=0)
        with pl.section_cube():
            pl.move(dst, src, acc_to_vec_mode=mode)

    if arch == NpuArch.DAV_9201:
        with pytest.raises(NotSupported, match="DualModeSplitM/DualModeSplitN"):
            codegen_9201(kernel.to_kernel_def(), arch)
    else:
        cube, _ = codegen_9201(kernel.to_kernel_def(), arch)
        assert "TMOV" in cube.content
