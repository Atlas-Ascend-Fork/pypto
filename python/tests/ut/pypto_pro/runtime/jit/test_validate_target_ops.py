# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Check target selection and placement diagnostics on Python DSL kernels."""

import importlib
import linecache
import re

from pypto_pro._errors import InvalidOperation
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pypto_pro.language.distribute as pld
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl import ir

jit = importlib.import_module("pypto_pro.runtime.jit")


def _codegen(kernel, tmp_path):
    return jit._parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, str(tmp_path))


def _placement_error(target, allowed):
    message = f"This operation cannot execute on {target}. Supported cores for this call: {allowed}."
    return re.escape(message)


def test_icache_preload_allows_both_targets():
    @pl.jit(auto_mutex=False)
    def kernel():
        pl.system.icache_preload(8)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    ir.Pass.validate_target_ops(ir.SectionKind.Cube)(program)
    ir.Pass.validate_target_ops(ir.SectionKind.Vector)(program)


def test_simd_call_checked_at_kernel_call_site(tmp_path):
    @pl.vector_function
    def inner():
        mask = vf.create_mask()  # noqa: F841

    @pl.vector_function
    def outer():
        inner()

    @pl.jit(auto_mutex=False)
    def vector_kernel(n: pl.DT_INT64):
        with pl.section_vector():
            outer()

    @pl.jit(auto_mutex=False)
    def cube_kernel(n: pl.DT_INT64):
        with pl.section_cube():
            outer()

    cube, vector = _codegen(vector_kernel, tmp_path)
    assert cube is None
    assert vector is not None
    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")) as caught:
        _codegen(cube_kernel, tmp_path)
    span = caught.value.span
    diagnostic = str(caught.value)
    assert "outer()" in diagnostic
    assert "simd.call" not in diagnostic
    assert span["filename"] == __file__
    assert linecache.getline(__file__, span["line"] - 1).strip() == "with pl.section_cube():"
    assert linecache.getline(__file__, span["line"]).strip() == "outer()"


@pytest.mark.parametrize("which", [0, 1, 2], ids=["bar_m", "bar_mte1", "bar_fix"])
def test_cube_pipeline_barrier_rejected_on_vector(which, tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        with pl.section_vector():
            if which == 0:
                pl.system.bar_m()
            elif which == 1:
                pl.system.bar_mte1()
            else:
                pl.system.bar_fix()

    with pytest.raises(InvalidOperation, match=_placement_error("Vector", "Cube")) as caught:
        _codegen(kernel, tmp_path)
    assert caught.value.span["filename"] == __file__


def test_common_cube_barriers_with_only_cube_target(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        pl.system.bar_m()
        pl.system.bar_mte1()
        pl.system.bar_fix()
        with pl.section_cube():
            pl.system.bar_all()

    cube, vector = _codegen(kernel, tmp_path)
    assert cube is not None and vector is None


@pytest.mark.parametrize("which", [0, 1, 2], ids=["enable_hf32", "disable_hf32", "set_hf32_round_mode"])
def test_hf32_configuration_rejected_on_vector(which, tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        with pl.section_vector():
            if which == 0:
                pl.system.enable_hf32()
            elif which == 1:
                pl.system.disable_hf32()
            else:
                pl.system.set_hf32_round_mode(pl.HF32RoundMode.NEAREST_EVEN)

    with pytest.raises(InvalidOperation, match=_placement_error("Vector", "Cube")):
        _codegen(kernel, tmp_path)


def test_hf32_configuration_allowed_on_cube():
    @pl.jit(auto_mutex=False)
    def kernel(mode: pl.DT_INT32):
        with pl.section_cube():
            pl.system.enable_hf32()
            pl.system.set_hf32_round_mode(pl.HF32RoundMode.NEAREST_AWAY)
            pl.system.set_hf32_round_mode(mode)
            pl.system.disable_hf32()

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    ir.Pass.validate_target_ops(ir.SectionKind.Cube)(program)


def test_common_init_output_with_only_vector_target(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1, 64], pl.DT_FP32]):
        pl.init_output(out, size=64, value=0)
        with pl.section_vector():
            pl.system.bar_all()

    cube, vector = _codegen(kernel, tmp_path)
    assert cube is None and vector is not None


def test_init_output_rejected_on_cube(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1, 64], pl.DT_FP32]):
        with pl.section_cube():
            pl.init_output(out, size=64, value=0)

    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")):
        _codegen(kernel, tmp_path)


def test_shared_pipeline_barriers_pass_on_both_targets():
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        pl.system.bar_mte2()
        pl.system.bar_mte3()
        pl.system.bar_all()

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    ir.Pass.validate_target_ops(ir.SectionKind.Cube)(program)
    ir.Pass.validate_target_ops(ir.SectionKind.Vector)(program)


@pytest.mark.parametrize(
    "memory,target,allowed",
    [(pl.MemorySpace.Vec, "Cube", "Vector"), (pl.MemorySpace.Bias, "Vector", "Cube"),
     (pl.MemorySpace.Scaling, "Vector", "Cube")],
)
def test_dump_tile_placement(memory, target, allowed):
    @pl.jit(auto_mutex=False)
    def kernel():
        tt = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=memory)
        tile = pl.make_tile(tt, addr=0)
        pl.dump_data(tile)

    kind = getattr(ir.SectionKind, target)
    program, _ = kernel.to_kernel_def().parse_target_program(kind)
    ir.Pass.validate_target_ops(getattr(ir.SectionKind, allowed))(program)
    with pytest.raises(ValueError, match=_placement_error(target, allowed)):
        ir.Pass.validate_target_ops(kind)(program)


def test_sectionless_cube_barrier_rejected_on_vector(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        pl.system.bar_m()

    with pytest.raises(InvalidOperation, match=_placement_error("Vector", "Cube")):
        _codegen(kernel, tmp_path)


def test_sectionless_init_output_rejected_on_cube(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1, 64], pl.DT_FP32]):
        pl.init_output(out, size=64, value=0)

    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")):
        _codegen(kernel, tmp_path)


def test_cube_barriers_and_vector_init_output_in_mixed_kernel(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(out: pl.Tensor[[1, 64], pl.DT_FP32]):
        with pl.section_cube():
            pl.system.bar_m()
            pl.system.bar_mte1()
            pl.system.bar_fix()
        with pl.section_vector():
            pl.init_output(out, size=64, value=0)

    cube, vector = _codegen(kernel, tmp_path)
    assert cube is not None and vector is not None


def test_common_load_checked_only_in_emitted_program(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 64], pl.DT_FP32]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, depth=1)
        pl.load(group.current(), x, [0, 0])
        with pl.section_vector():
            pl.system.bar_all()

    cube, vector = _codegen(kernel, tmp_path)
    assert cube is None
    assert vector is not None


def test_remote_ptr_rejected_on_cube(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(win: pld.Window, buf: pl.Tensor[[1, 8], pl.DT_INT32]):
        with pl.section_cube():
            ptr = pld.remote_ptr(win, buf, 1)  # noqa: F841

    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")):
        _codegen(kernel, tmp_path)


def test_unsectioned_load_checked_on_cube(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[1, 64], pl.DT_FP32]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, depth=1)
        pl.load(group.current(), x, [0, 0])

    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")) as caught:
        _codegen(kernel, tmp_path)
    assert caught.value.span["filename"] == __file__


def test_nested_cube_section_rejects_vector_add(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=0, depth=1)
        with pl.section_cube():
            if n > 0:
                tile = group.current()
                pl.add(tile, tile, tile)

    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")):
        _codegen(kernel, tmp_path)


def test_sync_all_uses_participating_core_type(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        with pl.section_vector():
            pl.system.sync_all(core_type=pl.SyncCoreType.AIC_ONLY)

    with pytest.raises(InvalidOperation, match=_placement_error("Vector", "Cube")):
        _codegen(kernel, tmp_path)


def test_move_to_cube_memory_can_require_vector(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        vec = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
        mat = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
        src = pl.make_tile_group(type=vec, addrs=0, depth=1)
        dst = pl.make_tile_group(type=mat, addrs=0, depth=1)
        with pl.section_cube():
            pl.move(dst.current(), src.current())

    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")):
        _codegen(kernel, tmp_path)


def test_cross_core_mode_restricts_shared_pipe(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        with pl.section_cube():
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE2, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
            )

    with pytest.raises(InvalidOperation, match=_placement_error("Cube", "Vector")):
        _codegen(kernel, tmp_path)


def test_shared_query_in_cube_condition(tmp_path):
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        with pl.section_cube():
            if n < pl.get_subblock_num():
                pl.system.bar_all()

    cube, vector = _codegen(kernel, tmp_path)
    assert cube is not None
    assert vector is None
