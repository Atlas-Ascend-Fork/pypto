# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT for ``pl.array``: construction, indexed loads/stores, aliasing, and errors."""

from pathlib import Path

from pypto_pro import ir
from pypto_pro._errors import InvalidShape, InvalidType, InvalidVal, NotSupported, OutOfRange
import pypto_pro.language as pl
import pytest
from test_common import assert_ir_matches_dump

_GOLDEN_DIR = Path(__file__).with_name("test_array_ir")


def _parse(kernel):
    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    return program.get_function(kernel.__name__)


def _expect_parse_error(kernel, error_type):
    with pytest.raises(error_type) as exc_info:
        _parse(kernel)
    span = exc_info.value.span
    assert span is not None and span["line"] > 0


def test_array_parses_exact_dsl():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        datas[0] = 7
        a = datas[0]
        alias = datas
        alias[0] = 9
        datas[1] = a
        _test_result = datas[0]

    assert_ir_matches_dump(_parse(kernel), _GOLDEN_DIR / "scalar_array.pypto")


def test_array_load_is_not_constant_folded():
    """An earlier load keeps its value after a later write to the same element."""

    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        first = datas[0]
        datas[0] = 9
        second = datas[0]
        _test_result = first + second

    assert_ir_matches_dump(_parse(kernel), _GOLDEN_DIR / "load_not_folded.pypto")


def test_array_store_reads_another_element_directly():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        datas[0] = 7
        datas[1] = datas[0]
        _test_result = datas[1]

    assert_ir_matches_dump(_parse(kernel), _GOLDEN_DIR / "store_from_load.pypto")


def test_array_runtime_index_and_loop():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        for i in pl.range(0, 8):
            datas[i] = i

    assert_ir_matches_dump(_parse(kernel), _GOLDEN_DIR / "runtime_index_loop.pypto")


def test_array_ir_dump_and_load_round_trip():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        datas[0] = 7
        a = datas[0]
        alias = datas
        alias[0] = 9
        datas[1] = a
        _test_result = datas[0]

    fn = _parse(kernel)
    golden = _GOLDEN_DIR / "scalar_array.pypto"
    assert_ir_matches_dump(fn, golden)
    assert_ir_matches_dump(ir.text_load_function(str(fn)), golden)


def test_array_rejects_nonconstant_size():
    @pl.jit(auto_mutex=False)
    def kernel(n: pl.DT_INT64):
        datas = pl.array(n, dtype=pl.DT_INT32)
        datas[0] = 1

    _expect_parse_error(kernel, InvalidType)


def test_array_rejects_nonpositive_size():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(0, dtype=pl.DT_INT32)
        datas[0] = 1

    _expect_parse_error(kernel, InvalidVal)


def test_array_rejects_noninteger_index():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        _test_result = datas[1.5]

    _expect_parse_error(kernel, InvalidType)


def test_array_rejects_constant_out_of_bounds():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        datas[8] = 1

    _expect_parse_error(kernel, OutOfRange)


def test_array_rejects_multidimensional_index():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        _test_result = datas[0, 1]

    _expect_parse_error(kernel, InvalidShape)


def test_array_rejects_non_scalar_store_value():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT32)
        datas[0] = (1, 2)

    _expect_parse_error(kernel, InvalidType)


def test_array_rejects_storage_only_dtype():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        datas = pl.array(8, dtype=pl.DT_INT4)
        datas[0] = 1

    _expect_parse_error(kernel, NotSupported)
