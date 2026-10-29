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

"""Parser tests for the Tile-first A5 SIMT frontend."""

import ast

from pypto_pro._errors import (
    InvalidArgument,
    InvalidOperation,
    InvalidShape,
    InvalidType,
    InvalidVal,
    NameNotFound,
    NotSupported,
    OutOfRange,
    PyptoProError,
)
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


@pl.vector_function(mode="simt", max_threads=256)
def _tile_add(
    dst,
    src,
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.vector_function(mode="simt", max_threads=256)
def _gm_add(
    dst: pl.Tensor[[1, 256], pl.DT_FP32],
    src: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.vector_function(mode="simt")
def _callee_add(value: pl.DT_INT32, delta: pl.DT_INT32) -> pl.DT_INT32:
    return value + delta


@pl.vector_function(mode="simt")
def _callee_store(
    dst,
    index: pl.DT_UINT32,
    value: pl.DT_INT32,
):
    dst[0, index] = value


@pl.vector_function(mode="simt")
def _callee_apply(
    dst,
    src,
    index: pl.DT_UINT32,
    delta: pl.DT_INT32,
):
    value = _callee_add(src[0, index], delta)
    _callee_store(dst, index, value)


@pl.vector_function(mode="simt", max_threads=32)
def _callee_entry(
    dst,
    src,
    delta: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    _callee_apply(dst, src, tid, delta)


@pl.vector_function(mode="simt", max_threads=256)
def _context_probe(dst):
    thread = pl.simt.thread_idx()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    grid = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()
    value = (
        thread.x
        + thread.y
        + thread.z
        + block.x
        + block.y
        + block.z
        + block_id.x
        + block_id.y
        + block_id.z
        + grid.x
        + grid.y
        + grid.z
        + pl.simt.warp_size()
    )
    dst[0, tid] = value


@pl.vector_function(mode="simt", max_threads=256)
def _tile_valid_shape_access(
    dst,
    src,
):
    tid = pl.simt.linear_thread_idx()
    rows = src.valid_shape[0]
    cols = src.valid_shape[1]
    row = tid // cols
    col = tid % cols
    if row < rows:
        dst[row, col] = src[row, col]


@pl.jit
def _simt_tile_kernel(n: pl.DT_UINT32, delta: pl.DT_FP32):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tile_type, addr=0x0000)
    dst = pl.make_tile(tile_type, addr=0x0400)
    with pl.section_vector():
        _tile_add[256](dst, src, n, delta)


@pl.jit
def _simt_gm_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    with pl.section_vector():
        _gm_add[256](out, x, n, delta)


@pl.jit
def _simt_context_kernel(_jit_entry: pl.DT_INT64):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    with pl.section_vector():
        _context_probe[8, 4, 8](dst)


@pl.jit
def _simt_callee_kernel(delta: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    src = pl.make_tile(tile_type, addr=0x0080)
    with pl.section_vector():
        _callee_entry[32](dst, src, delta)


def _make_tile_launch_kernel(shape, dtype, target_memory, layout):
    @pl.jit
    def kernel(n: pl.DT_UINT32, delta: pl.DT_FP32):
        tile_type = pl.TileType(
            shape=shape,
            dtype=dtype,
            target_memory=target_memory,
            layout=layout,
        )
        src = pl.make_tile(tile_type, addr=0x0000)
        dst = pl.make_tile(tile_type, addr=0x0400)
        with pl.section_vector():
            _tile_add[256](dst, src, n, delta)

    return kernel


def test_launchable_simt_vector_function_requires_indexed_invocation():
    @pl.vector_function(mode="simt", max_threads=32)
    def entry():
        return

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        with pl.section_vector():
            entry()

    with pytest.raises(InvalidOperation, match="cannot be called directly outside a SIMT function"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_helper_can_only_be_called_from_simt_function():
    @pl.vector_function(mode="simt")
    def helper():
        return

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        with pl.section_vector():
            helper()

    with pytest.raises(InvalidOperation, match="cannot be called directly outside a SIMT function"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_index_rejects_more_than_three_dimensions():
    @pl.vector_function(mode="simt", max_threads=32)
    def entry():
        return

    source = "def kernel(_jit_entry: pl.DT_INT64):\n    with pl.section_vector():\n        entry[1, 1, 1, 1]()\n"
    from pypto_pro.language.parser._ast_parser import ASTParser

    parser = ASTParser(
        source_file=__file__,
        source_lines=source.splitlines(),
        target=ir.SectionKind.Vector,
        debug_info=ir.IRDebugInfo(),
        closure_vars={"pl": pl, "entry": entry},
    )
    with pytest.raises(InvalidShape, match="one to three dimensions"):
        parser.parse_function(ast.parse(source).body[0])


def test_simt_indexed_invocation_rejects_keyword_arguments():
    @pl.vector_function(mode="simt", max_threads=32)
    def entry(value: pl.DT_INT32):
        return

    @pl.jit
    def kernel(value: pl.DT_INT32):
        with pl.section_vector():
            entry[32](value=value)

    with pytest.raises(InvalidArgument, match="accepts positional arguments only"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_tile_function_and_launch_build_vector_program():
    program, matched = _simt_tile_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("_tile_add")

    assert matched
    assert callable(_tile_add)
    assert not isinstance(_tile_add, ir.Function)
    assert function.func_type == ir.FunctionType.SimtVF
    assert function.has_attr("max_threads")
    assert function.get_attr("max_threads") == 256
    assert "#type(SimtVF)" in str(function)
    assert "simt.linear_thread_idx" in str(function)
    assert "block.getval" in str(function)
    assert "block.setval" in str(function)
    assert set(program.functions) == {"_tile_add", "_simt_tile_kernel"}
    assert "simt.launch" in str(program)


def test_simt_callee_records_type_calls_and_reachable_dependencies():
    program, matched = _simt_callee_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    callee = program.get_function("_callee_add")

    assert matched
    assert callee.func_type == ir.FunctionType.SimtCallee
    assert not callee.has_attr("max_threads")
    assert "#type(SimtCallee)" in str(callee)
    assert len(callee.return_types) == 1
    assert set(program.functions) == {
        "_callee_add",
        "_callee_store",
        "_callee_apply",
        "_callee_entry",
        "_simt_callee_kernel",
    }
    assert program.get_function("_callee_apply").func_type == ir.FunctionType.SimtCallee
    assert "_callee_add" in str(program.get_function("_callee_apply"))
    assert "_callee_store" in str(program.get_function("_callee_apply"))


def test_simt_gm_tensor_function_and_launch_reuse_scalar_tensor_access():
    program, matched = _simt_gm_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("_gm_add")
    function_ir = str(function)

    assert matched
    assert function.func_type == ir.FunctionType.SimtVF
    assert function_ir.count("block.getval") == 1
    assert function_ir.count("block.setval") == 1
    assert set(program.functions) == {"_gm_add", "_simt_gm_kernel"}
    assert "simt.launch" in str(program)


def test_simt_function_rejects_nested_launch():
    @pl.vector_function(mode="simt", max_threads=32)
    def nested_launch():
        _tile_add[32]()

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        with pl.section_vector():
            nested_launch[32]()

    with pytest.raises(NotSupported, match="Nested SIMT vector-function invocation"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_context_exposes_xyz_components_and_three_dimensional_launch():
    program, matched = _simt_context_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_context_probe"))

    assert matched
    assert function_ir.count("simt.thread_idx") >= 3
    assert function_ir.count("simt.block_dim") >= 3
    assert function_ir.count("simt.block_idx") >= 3
    assert function_ir.count("simt.grid_dim") >= 3
    assert "simt.linear_thread_idx" in function_ir
    assert "simt.warp_size" in function_ir
    assert "simt.launch" in str(program)


def test_simt_context_direct_call_uses_named_tuple_field_lowering():
    @pl.vector_function(mode="simt", max_threads=32)
    def direct_context(dst):
        value = (
            pl.simt.thread_idx().x
            + pl.simt.block_dim().y
            + pl.simt.block_idx().z
            + pl.simt.grid_dim().x
        )
        dst[0, 0] = value

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            direct_context[32](dst)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("direct_context"))

    assert function_ir.count("simt.thread_idx") >= 1
    assert function_ir.count("simt.block_dim") >= 1
    assert function_ir.count("simt.block_idx") >= 1
    assert function_ir.count("simt.grid_dim") >= 1


def test_simt_dim3_contexts_can_merge_across_control_flow():
    @pl.vector_function(mode="simt", max_threads=32)
    def merge_context(dst):
        tid = pl.simt.linear_thread_idx()
        if tid > 0:
            context = pl.simt.thread_idx()
        else:
            context = pl.simt.block_idx()
        dst[0, 0] = context.x

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            merge_context[32](dst)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("merge_context"))

    assert "if " in function_ir
    assert "simt.thread_idx" in function_ir
    assert "simt.block_idx" in function_ir


def test_simt_dim3_context_rejects_plain_tuple_merge():
    @pl.vector_function(mode="simt", max_threads=32)
    def merge_context(dst):
        tid = pl.simt.linear_thread_idx()
        if tid > 0:
            context = pl.simt.thread_idx()
        else:
            context = (tid, tid, tid)
        dst[0, 0] = context.x

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            merge_context[32](dst)

    with pytest.raises(NameNotFound, match="Use of potentially undefined variable"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_context_rejects_unknown_named_tuple_field():
    @pl.vector_function(mode="simt", max_threads=32)
    def invalid_context_field(dst):
        dst[0, 0] = pl.simt.thread_idx().w

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            invalid_context_field[32](dst)

    with pytest.raises(InvalidShape, match="Standalone attribute access not supported"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize(
    ("shape", "dtype", "target_memory", "layout", "message"),
    [
        pytest.param(
            [1, 256],
            pl.DT_FP32,
            pl.MemorySpace.Mat,
            pl.NZ,
            "Vec-memory Tile",
            id="memory",
        ),
        pytest.param(
            [8, 256],
            pl.DT_FP32,
            pl.MemorySpace.Vec,
            pl.DN,
            "ND Vec Tile",
            id="layout",
        ),
        pytest.param(
            [256],
            pl.DT_FP32,
            pl.MemorySpace.Vec,
            pl.ND,
            "rank-2 shape",
            id="rank",
        ),
    ],
)
def test_simt_launch_requires_compatible_tile(shape, dtype, target_memory, layout, message):
    kernel = _make_tile_launch_kernel(shape, dtype, target_memory, layout)

    with pytest.raises(PyptoProError, match=message):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_function_rejects_block_operation_before_default_dispatch():
    @pl.vector_function(mode="simt", max_threads=32)
    def block_add(
        dst,
        lhs,
        rhs,
    ):
        pl.add(dst, lhs, rhs)

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        lhs = pl.make_tile(tile_type, addr=128)
        rhs = pl.make_tile(tile_type, addr=256)
        with pl.section_vector():
            block_add[32](dst, lhs, rhs)

    with pytest.raises(NotSupported, match="not supported inside a SIMT function"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_function_rejects_tile_subview():
    @pl.vector_function(mode="simt", max_threads=32)
    def tile_subview(src):
        _ = src[0:4, 0:32]

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            tile_subview[32](src)

    with pytest.raises(InvalidOperation, match="Tile subview"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_tile_parameter_exposes_runtime_valid_shape():
    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        src = pl.make_tile(tile_type, addr=2048)
        with pl.section_vector():
            _tile_valid_shape_access[32](dst, src)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_tile_valid_shape_access"))

    assert function_ir.count("block.tile_valid_shape") == 2


def test_thread_idx_rejected_outside_simt_function():
    with pytest.raises(InvalidOperation, match="only be used inside"):

        @pl.jit(auto_mutex=False)
        def bad_thread_idx(_jit_entry: pl.DT_INT64):
            _test_result = pl.simt.thread_idx().x

        bad_thread_idx.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_launch_rejects_threads_above_bound():
    @pl.jit
    def too_many_threads(n: pl.DT_UINT32, delta: pl.DT_FP32):
        tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0x0000)
        dst = pl.make_tile(tile_type, addr=0x0400)
        with pl.section_vector():
            _tile_add[288](dst, src, n, delta)

    with pytest.raises(OutOfRange, match="exceed"):
        too_many_threads.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_launch_rejects_runtime_tuple_component():
    @pl.jit
    def runtime_dimension(n: pl.DT_UINT32, delta: pl.DT_FP32):
        tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        src = pl.make_tile(tile_type, addr=0x0000)
        dst = pl.make_tile(tile_type, addr=0x0400)
        with pl.section_vector():
            _tile_add[8, 4, n](dst, src, n, delta)

    with pytest.raises(InvalidShape, match=r"compile-time integers.*\[1, 2048\]"):
        runtime_dimension.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_launch_rejects_thread_count_above_hardware_limit():
    @pl.vector_function(mode="simt", max_threads=2048)
    def wide_function():
        return

    @pl.jit
    def too_wide(_jit_entry: pl.DT_INT64):
        with pl.section_vector():
            wide_function[2048, 2]()

    with pytest.raises(OutOfRange, match="must not exceed 2048"):
        too_wide.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_function_infers_parameter_and_callee_return_types_at_call_site():
    @pl.vector_function(mode="simt")
    def inferred_callee(value):
        return value

    @pl.vector_function(mode="simt", max_threads=32)
    def inferred_entry(dst, value):
        dst[0, 0] = inferred_callee(value)

    @pl.jit
    def kernel(value: pl.DT_INT32):
        tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            inferred_entry[32](dst, value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    callee = program.get_function("inferred_callee")
    entry = program.get_function("inferred_entry")

    assert isinstance(callee.params[0].type, ir.ScalarType)
    assert callee.params[0].type.dtype == pl.DT_INT32
    assert len(callee.return_types) == 1
    assert callee.return_types[0].dtype == pl.DT_INT32
    assert isinstance(entry.params[0].type, ir.TileType)
    assert entry.params[1].type.dtype == pl.DT_INT32


def test_simt_function_annotations_do_not_override_callsite_types():
    @pl.vector_function(mode="simt")
    def annotated_callee(value: pl.DT_INT32) -> pl.DT_INT32:
        return value

    @pl.vector_function(mode="simt", max_threads=32)
    def annotated_entry(value: pl.DT_INT32):
        annotated_callee(value)

    @pl.jit
    def kernel(value: pl.DT_INT64):
        with pl.section_vector():
            annotated_entry[32](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    callee = program.get_function("annotated_callee")
    entry = program.get_function("annotated_entry")

    assert entry.params[0].type.dtype == pl.DT_INT64
    assert callee.params[0].type.dtype == pl.DT_INT64
    assert callee.return_types[0].dtype == pl.DT_INT64


def test_simt_callee_rejects_return_incompatible_with_annotation():
    @pl.vector_function(mode="simt")
    def bad_return(value: pl.DT_FP32) -> pl.DT_INT32:
        return value

    @pl.vector_function(mode="simt", max_threads=32)
    def entry(value: pl.DT_FP32):
        bad_return(value)

    @pl.jit
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            entry[32](value)

    with pytest.raises(InvalidVal, match="Return 'bad_return' annotated as"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_callee_rejects_inconsistent_return_types_across_paths():
    @pl.vector_function(mode="simt")
    def inconsistent_return(condition: pl.DT_BOOL, integer: pl.DT_INT32, floating: pl.DT_FP32):
        if condition:
            return integer
        return floating

    @pl.vector_function(mode="simt", max_threads=32)
    def entry(condition: pl.DT_BOOL, integer: pl.DT_INT32, floating: pl.DT_FP32):
        inconsistent_return(condition, integer, floating)

    @pl.jit
    def kernel(condition: pl.DT_BOOL, integer: pl.DT_INT32, floating: pl.DT_FP32):
        with pl.section_vector():
            entry[32](condition, integer, floating)

    with pytest.raises(
        InvalidType,
        match="Function 'inconsistent_return' has inconsistent return types across return paths",
    ):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_function_rejects_argument_incompatible_with_annotation():
    @pl.vector_function(mode="simt", max_threads=32)
    def entry(value: pl.DT_INT32):
        return

    @pl.jit
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            entry[32](value)

    with pytest.raises(InvalidVal, match="SIMT parameter 'value' annotated as"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_cached_simt_function_rejects_incompatible_argument_type():
    @pl.vector_function(mode="simt", max_threads=32)
    def entry(value):
        return

    @pl.jit
    def kernel(integer: pl.DT_INT32, floating: pl.DT_FP32):
        with pl.section_vector():
            entry[32](integer)
            entry[32](floating)

    with pytest.raises(InvalidVal, match="SIMT parameter 'value_0' annotated as"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_entry_rejects_value_return():
    @pl.vector_function(mode="simt", max_threads=32)
    def entry(value: pl.DT_INT32):
        return value

    @pl.jit
    def kernel(value: pl.DT_INT32):
        with pl.section_vector():
            entry[32](value)

    with pytest.raises(NotSupported, match="only supports bare return or return None"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_scalar_simt_callee_accepts_early_return_when_all_paths_return():
    @pl.vector_function(mode="simt")
    def absolute(value: pl.DT_INT32) -> pl.DT_INT32:
        if value < 0:
            return -value
        return value

    @pl.vector_function(mode="simt", max_threads=32)
    def entry(value: pl.DT_INT32):
        absolute(value)

    @pl.jit
    def kernel(value: pl.DT_INT32):
        with pl.section_vector():
            entry[32](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function = program.get_function("absolute")
    assert function.func_type == ir.FunctionType.SimtCallee
    assert str(function).count("return") == 2


def test_simt_callee_cannot_be_launched_directly():
    @pl.jit
    def invalid_launch(value: pl.DT_INT32):
        with pl.section_vector():
            _callee_add[32](value, value)

    with pytest.raises(InvalidVal, match="not a launchable"):
        invalid_launch.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_recursive_simt_callee_is_rejected_during_instantiation():
    @pl.vector_function(mode="simt")
    def recursive(value: pl.DT_INT32) -> pl.DT_INT32:
        return recursive(value)

    @pl.vector_function(mode="simt", max_threads=32)
    def entry(value: pl.DT_INT32):
        recursive(value)

    @pl.jit
    def kernel(value: pl.DT_INT32):
        with pl.section_vector():
            entry[32](value)

    with pytest.raises(NotSupported, match="Recursive helper"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_launch_auto_mutex_inserts_pipe_v_lock_unlock():
    @pl.vector_function(mode="simt", max_threads=256)
    def inplace_add(data, delta: pl.DT_FP32):
        tid = pl.simt.linear_thread_idx()
        data[0, tid] = data[0, tid] + delta

    @pl.jit
    def kernel(
        x: pl.Tensor[[1, 256], pl.DT_FP32],
        out: pl.Tensor[[1, 256], pl.DT_FP32],
        delta: pl.DT_FP32,
    ):
        tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        data = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
        with pl.section_vector():
            pl.load(data.current(), x, [0, 0])
            inplace_add[256](data.current(), delta)
            pl.store(out, data.current(), [0, 0])

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "system.mutex_lock_dyn" not in str(program)
    program = ir.Pass.insert_sync()(program)
    ir_str = str(program)

    assert "system.mutex_lock_dyn" in ir_str
    assert "system.mutex_unlock_dyn" in ir_str
    assert "simt.launch" in ir_str

    kernel_ir = str(program.get_function("kernel"))
    lock_pos = kernel_ir.index("system.mutex_lock_dyn")
    launch_pos = kernel_ir.index("simt.launch")
    unlock_pos = kernel_ir.index("system.mutex_unlock_dyn", launch_pos)
    assert lock_pos < launch_pos < unlock_pos


@pl.vector_function(mode="simt", max_threads=32)
def _warp_ops(value: pl.DT_FP32, predicate_value: pl.DT_INT32):
    lane = pl.simt.lane_id()
    pl.simt.lanemask_eq()
    pl.simt.lanemask_le()
    pl.simt.lanemask_lt()
    pl.simt.lanemask_ge()
    pl.simt.lanemask_gt()
    predicate = lane < 16
    pl.simt.warp_all(predicate)
    pl.simt.warp_any(predicate_value)
    pl.simt.warp_ballot(predicate)
    pl.simt.warp_active_mask()
    warp_width = pl.simt.warp_size()
    subgroup_width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    pl.simt.warp_shfl(value, 0)
    pl.simt.warp_shfl_up(value, 1, subgroup_width)
    pl.simt.warp_shfl_down(value, 1, width=8)
    pl.simt.warp_shfl_xor(value, 1)
    pl.simt.warp_reduce_add(value)
    pl.simt.warp_reduce_max(value)
    pl.simt.warp_reduce_min(value)


@pl.jit
def _warp_kernel(value: pl.DT_FP32, predicate_value: pl.DT_INT32):
    with pl.section_vector():
        _warp_ops[32](value, predicate_value)


def test_warp_operations_parse_to_registered_ir_calls():
    program, matched = _warp_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_warp_ops"))

    assert matched
    for op_name in (
        "lane_id",
        "lanemask_eq",
        "lanemask_le",
        "lanemask_lt",
        "lanemask_ge",
        "lanemask_gt",
        "warp_all",
        "warp_any",
        "warp_ballot",
        "warp_active_mask",
        "warp_size",
        "warp_shfl",
        "warp_shfl_up",
        "warp_shfl_down",
        "warp_shfl_xor",
        "warp_reduce_add",
        "warp_reduce_max",
        "warp_reduce_min",
    ):
        assert f"simt.{op_name}" in function_ir


def test_shuffle_accepts_runtime_controls_with_asc_dtypes():
    @pl.vector_function(mode="simt", max_threads=32)
    def runtime_shuffle(
        value: pl.DT_FP16,
        src_lane: pl.DT_INT32,
        delta: pl.DT_UINT32,
        width: pl.DT_INT32,
    ):
        pl.simt.warp_shfl(value, src_lane, width)
        pl.simt.warp_shfl_up(value, delta, width)
        pl.simt.warp_shfl_down(value, delta, width)
        pl.simt.warp_shfl_xor(value, src_lane, width)

    @pl.jit
    def kernel(
        value: pl.DT_FP16,
        src_lane: pl.DT_INT32,
        delta: pl.DT_UINT32,
        width: pl.DT_INT32,
    ):
        with pl.section_vector():
            runtime_shuffle[32](value, src_lane, delta, width)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("runtime_shuffle"))
    assert function_ir.count("simt.warp_shfl") == 4


def test_warp_controls_evaluate_the_whole_constant_expression_before_ir_range_checks():
    @pl.vector_function(mode="simt", max_threads=32)
    def wide_controls(value: pl.DT_FP32):
        pl.simt.warp_all(18446744073709551616 - 18446744073709551615)
        pl.simt.warp_any(predicate=18446744073709551616 - 18446744073709551615)
        pl.simt.warp_ballot(18446744073709551616 - 18446744073709551615)
        pl.simt.warp_shfl(value, 18446744073709551616 - 18446744073709551615)
        pl.simt.warp_shfl_up(value, delta=18446744073709551616 - 18446744073709551615)
        pl.simt.warp_shfl_down(value, 18446744073709551616 - 18446744073709551615)
        pl.simt.warp_shfl_xor(value, 1, width=(1 << 70) - (1 << 70) + 32)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            wide_controls[32](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    calls = [
        stmt.expr for stmt in program.get_function("wide_controls").body.stmts
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]
    assert len(calls) == 7
    for call in calls:
        if len(call.args) == 1:
            assert call.args[0].value == 1
            assert call.args[0].type.dtype == ir.DataType.INT32
        else:
            expected = (
                ir.DataType.UINT32 if call.name in ("simt.warp_shfl_up", "simt.warp_shfl_down") else ir.DataType.INT32
            )
            assert call.args[1].value == 1 and call.args[1].type.dtype == expected
            assert call.args[2].value == 32 and call.args[2].type.dtype == ir.DataType.INT32


def test_warp_operation_rejected_outside_simt_function():
    with pytest.raises(InvalidOperation, match="only be used inside"):

        @pl.jit(auto_mutex=False)
        def bad_lane_id(_jit_entry: pl.DT_INT64):
            pl.simt.lane_id()

        bad_lane_id.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pl.vector_function(mode="simt", max_threads=32)
def _atomic_add_tile(
    dst,
    old_values,
    value: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.vector_function(mode="simt", max_threads=32)
def _atomic_add_tensor(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.vector_function(mode="simt")
def _atomic_rmw_ops(
    numeric,
    bitwise,
    counter,
    int_value: pl.DT_INT32,
    uint_value: pl.DT_UINT32,
):
    pl.simt.atomic_sub(numeric[0, 0], int_value)
    pl.simt.atomic_exch(numeric[0, 0], int_value)
    pl.simt.atomic_max(numeric[0, 0], int_value)
    pl.simt.atomic_min(numeric[0, 0], int_value)
    pl.simt.atomic_cas(numeric[0, 0], int_value, int_value)
    pl.simt.atomic_and(bitwise[0, 0], uint_value)
    pl.simt.atomic_or(bitwise[0, 0], uint_value)
    pl.simt.atomic_xor(bitwise[0, 0], uint_value)
    pl.simt.atomic_inc(counter[0, 0], uint_value)
    pl.simt.atomic_dec(counter[0, 0], uint_value)


@pl.vector_function(mode="simt", max_threads=1)
def _atomic_rmw_ops_entry(numeric, bitwise, counter, int_value, uint_value):
    _atomic_rmw_ops(numeric, bitwise, counter, int_value, uint_value)


@pl.jit
def _atomic_add_tile_kernel(value: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000)
    old_values = pl.make_tile(tile_type, addr=0x0080)
    with pl.section_vector():
        _atomic_add_tile[32](dst, old_values, value)


@pl.jit
def _atomic_add_tensor_kernel(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    with pl.section_vector():
        _atomic_add_tensor[32](dst, old_values, value)


@pl.jit
def _atomic_rmw_ops_kernel(int_value: pl.DT_INT32, uint_value: pl.DT_UINT32):
    int_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    uint_type = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    numeric = pl.make_tile(int_type, addr=0)
    bitwise = pl.make_tile(uint_type, addr=128)
    counter = pl.make_tile(uint_type, addr=256)
    with pl.section_vector():
        _atomic_rmw_ops_entry[1](numeric, bitwise, counter, int_value, uint_value)


def _parse_one_tile_function(function, dtype, shape=(1, 32)):
    @pl.vector_function(mode="simt", max_threads=1)
    def entry(tile):
        function(tile)

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=shape, dtype=dtype, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            entry[1](tile)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def _parse_two_tile_function(function, first_dtype, second_dtype):
    @pl.vector_function(mode="simt", max_threads=1)
    def entry(first, second):
        function(first, second)

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        first_type = pl.TileType(shape=[1, 32], dtype=first_dtype, target_memory=pl.MemorySpace.Vec)
        second_type = pl.TileType(shape=[1, 32], dtype=second_dtype, target_memory=pl.MemorySpace.Vec)
        first = pl.make_tile(first_type, addr=0)
        second = pl.make_tile(second_type, addr=4096)
        with pl.section_vector():
            entry[1](first, second)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def _parse_tile_scalar_function(function, tile_dtype, scalar_dtype):
    @pl.vector_function(mode="simt", max_threads=1)
    def entry(tile, value):
        function(tile, value)

    if scalar_dtype == pl.DT_INT32:

        @pl.jit
        def kernel(value: pl.DT_INT32):
            tile_type = pl.TileType(shape=[1, 32], dtype=tile_dtype, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0)
            with pl.section_vector():
                entry[1](tile, value)

    elif scalar_dtype == pl.DT_UINT32:

        @pl.jit
        def kernel(value: pl.DT_UINT32):
            tile_type = pl.TileType(shape=[1, 32], dtype=tile_dtype, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0)
            with pl.section_vector():
                entry[1](tile, value)

    elif scalar_dtype == pl.DT_FP32:

        @pl.jit
        def kernel(value: pl.DT_FP32):
            tile_type = pl.TileType(shape=[1, 32], dtype=tile_dtype, target_memory=pl.MemorySpace.Vec)
            tile = pl.make_tile(tile_type, addr=0)
            with pl.section_vector():
                entry[1](tile, value)

    else:
        raise ValueError(f"Unsupported scalar dtype: {scalar_dtype}")

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return program.get_function(function.__name__)


def test_atomic_add_preserves_tile_lvalue_and_returns_old_value():
    program, matched = _atomic_add_tile_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_add_tile"))

    assert matched
    assert function_ir.count("simt.atomic_add") == 1
    assert "block.getval" not in function_ir
    assert "block.setval" in function_ir
    assert "simt.launch" in str(program)


def test_atomic_add_supports_gm_int64_tensor():
    program, matched = _atomic_add_tensor_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_add_tensor"))

    assert matched
    assert function_ir.count("simt.atomic_add") == 1
    assert "block.getval" not in function_ir
    assert "block.setval" in function_ir
    assert "simt.launch" in str(program)


@pytest.mark.parametrize("dtype", [pl.DT_INT32, pl.DT_UINT32, pl.DT_FP16, pl.DT_BF16, pl.DT_FP32])
@pytest.mark.parametrize("use_tile", [False, True], ids=["tensor", "tile"])
def test_atomic_add_accepts_loaded_elements(dtype, use_tile):
    @pl.vector_function(mode="simt")
    def loaded_values(dst, values):
        value = values[0, 0]
        pl.simt.atomic_add(dst[0, 0], value)
        pl.simt.atomic_add(dst[0, 1], values[0, 1])

    if use_tile:
        function = _parse_two_tile_function(loaded_values, dtype, dtype)
    else:

        @pl.vector_function(mode="simt", max_threads=1)
        def entry(dst, values):
            loaded_values(dst, values)

        @pl.jit
        def kernel(dst: pl.Tensor[[1, 32], dtype], values: pl.Tensor[[1, 32], dtype]):
            with pl.section_vector():
                entry[1](dst, values)

        program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
        function = program.get_function("loaded_values")

    loads = [
        stmt.value
        for stmt in function.body.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call) and stmt.value.name == "block.getval"
    ]
    assert len(loads) == 2
    assert all(load.type.dtype == dtype for load in loads)
    assert str(function).count("simt.atomic_add") == 2


def test_atomic_add_contextually_types_numeric_literals():
    @pl.vector_function(mode="simt")
    def literal_values(unsigned, signed):
        old = pl.simt.atomic_add(unsigned[0, 0], 1)
        signed[0, 0] = pl.simt.atomic_add(signed[0, 0], -1)
        unsigned[0, 0] = old

    function_ir = str(_parse_two_tile_function(literal_values, pl.DT_UINT32, pl.DT_INT32))
    assert function_ir.count("simt.atomic_add") == 2


def test_atomic_add_requires_direct_subscript_target():
    @pl.vector_function(mode="simt")
    def scalar_alias(dst, value: pl.DT_INT32):
        current = dst[0, 0]
        pl.simt.atomic_add(current, value)

    with pytest.raises(InvalidVal, match="direct Tile or Tensor subscript"):
        _parse_tile_scalar_function(scalar_alias, pl.DT_INT32, pl.DT_INT32)


def test_atomic_add_rejects_slice_target():
    @pl.vector_function(mode="simt")
    def slice_target(dst, value: pl.DT_INT32):
        pl.simt.atomic_add(dst[0:1, 0:1], value)

    with pytest.raises(NotSupported, match="does not support slices"):
        _parse_tile_scalar_function(slice_target, pl.DT_INT32, pl.DT_INT32)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16])
def test_half_precision_atomic_result_cannot_be_returned(dtype):
    @pl.vector_function(mode="simt")
    def add_result(dst) -> None:
        return pl.simt.atomic_add(dst[0, 0], 1.0)

    with pytest.raises(InvalidVal, match="must return None or one scalar value"):
        _parse_one_tile_function(add_result, dtype)


def test_atomic_add_rejects_float_literal_for_integer_target():
    @pl.vector_function(mode="simt")
    def float_literal(dst):
        pl.simt.atomic_add(dst[0, 0], 1.0)

    with pytest.raises(InvalidType, match="requires an integer value"):
        _parse_one_tile_function(float_literal, pl.DT_INT32)


def test_atomic_rmw_interfaces_preserve_lvalue_and_build_distinct_ir_ops():
    program, _ = _atomic_rmw_ops_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_rmw_ops"))

    for op_name in (
        "atomic_sub",
        "atomic_exch",
        "atomic_max",
        "atomic_min",
        "atomic_cas",
        "atomic_and",
        "atomic_or",
        "atomic_xor",
        "atomic_inc",
        "atomic_dec",
    ):
        assert function_ir.count(f"simt.{op_name}") == 1
    assert "block.getval" not in function_ir


def test_atomic_cas_contextually_types_compare_and_value_literals():
    @pl.vector_function(mode="simt")
    def literal_operands(dst):
        old = pl.simt.atomic_cas(dst[0, 0], 0, 1)
        dst[0, 0] = old

    assert "simt.atomic_cas" in str(_parse_one_tile_function(literal_operands, pl.DT_UINT32))


def test_atomic_counter_supports_gm_uint64():
    @pl.vector_function(mode="simt", max_threads=1)
    def uint64_counter(dst: pl.Tensor[[1], pl.DT_UINT64], limit: pl.DT_UINT64):
        pl.simt.atomic_dec(dst[0], limit)

    @pl.jit(auto_mutex=False)
    def kernel(dst: pl.Tensor[[1], pl.DT_UINT64], limit: pl.DT_UINT64):
        with pl.section_vector():
            uint64_counter[1](dst, limit)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    uint64_counter = program.get_function(uint64_counter.__name__)

    assert "simt.atomic_dec" in str(uint64_counter)


def test_simt_cast_is_not_exported_as_shared_language_api():
    assert not hasattr(pl, "scalar_cast")


def test_simt_cast_rejects_ordinary_function():
    with pytest.raises(InvalidOperation, match="can only be used inside a SIMT function"):

        @pl.jit(auto_mutex=False)
        def unsupported_context(value: pl.DT_INT64):
            _test_result = pl.simt.cast(value, pl.DT_INT32)

        unsupported_context.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_is_available_inside_simt_helper():
    @pl.vector_function(mode="simt")
    def cast_value(value: pl.DT_FP32):
        _test_result = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)

    @pl.vector_function(mode="simt", max_threads=1)
    def entry(value: pl.DT_FP32):
        cast_value(value)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            entry[1](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    cast_value = program.get_function(cast_value.__name__)

    function_ir = str(cast_value)
    assert "simt.cast" in function_ir


def test_simt_cast_rejects_plain_integer_round_mode():
    with pytest.raises(InvalidType, match="mode expects"):

        @pl.vector_function(mode="simt", max_threads=1)
        def integer_mode(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_FP16, mode=1)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                integer_mode[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_wrong_positional_arity():
    with pytest.raises(InvalidArgument, match="missing a required argument: 'dtype'"):

        @pl.vector_function(mode="simt", max_threads=1)
        def missing_dtype(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                missing_dtype[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_non_dtype_target():
    with pytest.raises(InvalidType, match="dtype expects"):

        @pl.vector_function(mode="simt", max_threads=1)
        def invalid_dtype(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, 1)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                invalid_dtype[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_unexpected_keyword():
    with pytest.raises(InvalidArgument, match="unexpected keyword argument 'saturate'"):

        @pl.vector_function(mode="simt", max_threads=1)
        def unexpected_keyword(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_FP16, saturate=True)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                unexpected_keyword[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_scalar_math_is_available_in_simt_functions():
    @pl.vector_function(mode="simt", max_threads=1)
    def scalar_math(value: pl.DT_FP32, rhs: pl.DT_FP32, addend: pl.DT_FP32):
        transformed = pl.simt.min(value, pl.simt.abs(rhs))
        _test_result = pl.simt.fma(pl.simt.sqrt(transformed), rhs, addend)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32, rhs: pl.DT_FP32, addend: pl.DT_FP32):
        with pl.section_vector():
            scalar_math[1](value, rhs, addend)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    scalar_math = program.get_function(scalar_math.__name__)

    function_ir = str(scalar_math)
    for name in ("abs", "min", "sqrt", "fma"):
        assert f"simt.{name}(" in function_ir


def test_scalar_math_rejects_ordinary_function():
    with pytest.raises(InvalidOperation, match="can only be used inside a SIMT function"):

        @pl.jit(auto_mutex=False)
        def unsupported(value: pl.DT_FP32):
            _test_result = pl.simt.exp(value)

        unsupported.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_scalar_math_rejects_wrong_arity_and_accepts_keywords():
    with pytest.raises(InvalidArgument, match="missing a required argument: 'addend'"):

        @pl.vector_function(mode="simt", max_threads=1)
        def missing_addend(value: pl.DT_FP32):
            _test_result = pl.simt.fma(value, value)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                missing_addend[1](value)

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    @pl.vector_function(mode="simt", max_threads=1)
    def keyword_argument(value: pl.DT_FP32):
        _test_result = pl.simt.exp(value=value)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            keyword_argument[1](value)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert "simt.exp(" in str(program.get_function(keyword_argument.__name__))


def test_simt_operations_reject_argument_unpacking_at_common_entry():
    @pl.vector_function(mode="simt", max_threads=1)
    def positional_unpacking(value: pl.DT_FP32):
        _test_result = pl.simt.exp(*(value,))

    @pl.vector_function(mode="simt", max_threads=1)
    def keyword_unpacking(value: pl.DT_FP32):
        _test_result = pl.simt.exp(**{"value": value})

    for function in (positional_unpacking, keyword_unpacking):

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                function[1](value)

        with pytest.raises(InvalidArgument, match="does not support argument unpacking"):
            kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
