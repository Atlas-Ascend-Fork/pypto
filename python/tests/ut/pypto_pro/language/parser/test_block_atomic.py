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
from pathlib import Path

from pypto_pro._errors import InvalidType, InvalidVal, NotSupported
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir

_GOLDEN_DIR = Path(__file__).parent / "test_block_atomic_ir"


def _parse(kernel):
    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return program.get_function(kernel.__name__)


def _assert_ir(kernel, golden_name):
    """Whole parsed function text vs the ``.pypto`` golden."""
    func = _parse(kernel)
    assert isinstance(func, ir.Function)
    golden_path = _GOLDEN_DIR / f"{golden_name}.pypto"
    assert str(func).rstrip("\n") == golden_path.read_text(encoding="utf-8").rstrip("\n"), golden_name


@pl.jit(arch="3510", auto_mutex=False)
def _atomic_kernel(
    i32: pl.Tensor[[16], pl.DT_INT32],
    u32: pl.Tensor[[16], pl.DT_UINT32],
    f32: pl.Tensor[[16], pl.DT_FP32],
    u64: pl.Tensor[[16], pl.DT_UINT64],
    out: pl.Tensor[[16], pl.DT_INT64],
    fout: pl.Tensor[[16], pl.DT_FP32],
    value: pl.DT_INT32,
    i: pl.DT_INT64,
    j: pl.DT_INT64,
):
    with pl.section_vector():
        pl.atomic_add(i32, 0, 1)
        pl.atomic_sub(i32, 1, 1)
        pl.atomic_exch(i32, 2, 1)
        pl.atomic_max(i32, 3, 1)
        pl.atomic_min(i32, 4, 1)
        pl.atomic_and(i32, 5, 1)
        pl.atomic_or(u32, 0, 1)
        pl.atomic_xor(i32, 6, 1)
        pl.atomic_inc(u32, 1, 5)
        pl.atomic_dec(u32, 2, 5)
        pl.atomic_add(f32, 0, 1.5)
        pl.atomic_cas(i32, 7, 0, 7)
        pl.atomic_exch(u64, 0, 18446744073709551615)
        pl.atomic_add(i32, 8, value)
        pl.atomic_add(tensor=i32, offset=i * 8 + j, value=1)
        pl.setval(out, 0, pl.atomic_add(i32, 9, 1))
        old = pl.atomic_load(i32, 1)
        pl.setval(out, 1, old)
        # atomic_store is a posted write (no dsb emitted), so order the read-back
        # explicitly; otherwise the load may observe the pre-store value.
        pl.atomic_store(i32, 1, 7)
        pl.system.dsb(scope=pl.MemDsb.DDR)
        pl.setval(out, 2, pl.atomic_load(i32, 1))
        pl.atomic_store(f32, 0, 1.5)
        pl.system.dsb(scope=pl.MemDsb.DDR)
        pl.setval(fout, 0, pl.atomic_load(f32, 0))
        pl.system.nop(4)


def test_atomic_lowering():
    _assert_ir(_atomic_kernel, "atomic")


def test_load_rejects_unsupported_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(u8: pl.Tensor[[16], pl.DT_UINT8]):
        with pl.section_vector():
            pl.atomic_load(u8, 0)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_store_rejects_a_mismatched_value_dtype():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32], value: pl.DT_INT64):
        with pl.section_vector():
            pl.atomic_store(i32, 0, value)

    with pytest.raises(TypeError, match="dtype must match the element dtype"):
        _parse(kernel)


def test_float_literal_on_an_integer_element_is_rejected():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            pl.atomic_add(i32, 0, 1.5)

    with pytest.raises(InvalidType, match="requires an integer value"):
        _parse(kernel)


def test_arithmetic_rejects_narrow_integer_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(u8: pl.Tensor[[16], pl.DT_UINT8]):
        with pl.section_vector():
            pl.atomic_add(u8, 0, 1)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_bitwise_rejects_floating_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(f32: pl.Tensor[[16], pl.DT_FP32]):
        with pl.section_vector():
            pl.atomic_and(f32, 0, 1.0)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_counter_rejects_signed_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            pl.atomic_inc(i32, 0, 1)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_cas_rejects_boolean_elements():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(flag: pl.Tensor[[16], pl.DT_BOOL]):
        with pl.section_vector():
            pl.atomic_cas(flag, 0, 0, 1)

    with pytest.raises(NotImplementedError, match="does not support dtype"):
        _parse(kernel)


def test_runtime_value_dtype_must_match_the_element_dtype():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32], value: pl.DT_INT64):
        with pl.section_vector():
            pl.atomic_add(i32, 0, value)

    with pytest.raises(TypeError, match="dtype must match the element dtype"):
        _parse(kernel)


def test_ub_tile_is_rejected():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel():
        tile_type = pl.TileType(shape=[1, 16], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_vector():
            pl.atomic_add(tile, 0, 1)

    with pytest.raises(InvalidType, match="Tensor"):
        _parse(kernel)


def test_atomic_is_rejected_inside_a_simt_function():
    @pl.vector_function(mode="simt", max_threads=1)
    def simt_body(i32):
        pl.atomic_add(i32, 0, 1)

    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(i32: pl.Tensor[[16], pl.DT_INT32]):
        with pl.section_vector():
            simt_body[1](i32)

    with pytest.raises(NotSupported, match="not supported inside a SIMT function"):
        _parse(kernel)


def test_nop_rejects_a_negative_count():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel():
        with pl.section_vector():
            pl.system.nop(-1)

    with pytest.raises(InvalidVal, match="at least 0"):
        _parse(kernel)


def test_nop_requires_a_constant_count():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(n: pl.DT_INT32):
        with pl.section_vector():
            pl.system.nop(n)

    with pytest.raises(InvalidType, match="compile-time integer constant"):
        _parse(kernel)


def test_nop_rejects_an_oversized_count():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel():
        with pl.section_vector():
            pl.system.nop(2 ** 31)

    with pytest.raises(InvalidVal, match="at most 2147483647"):
        _parse(kernel)
