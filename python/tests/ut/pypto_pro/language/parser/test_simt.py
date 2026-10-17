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


def test_vector_function_empty_call_is_rejected():
    with pytest.raises(NotSupported, match=r"@pl\.vector_function\(\) is not supported"):
        pl.vector_function()


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


@pytest.mark.parametrize("max_threads", [0, 2049])
def test_simt_vector_function_rejects_invalid_max_threads(max_threads):
    with pytest.raises(ValueError, match=r"max_threads must be in \[1, 2048\]"):
        pl.vector_function(mode="simt", max_threads=max_threads)


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
    ir_str = str(program)

    assert "system.mutex_lock_dyn" in ir_str
    assert "system.mutex_unlock_dyn" in ir_str
    assert "simt.launch" in ir_str

    kernel_ir = str(program.get_function("kernel"))
    lock_pos = kernel_ir.index("system.mutex_lock_dyn")
    launch_pos = kernel_ir.index("simt.launch")
    unlock_pos = kernel_ir.index("system.mutex_unlock_dyn", launch_pos)
    assert lock_pos < launch_pos < unlock_pos
