# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ruff: noqa: F841

"""Constant-folding coverage for Python scalar expressions."""

import math

from pypto_pro._errors import InvalidType, InvalidVal, RuntimeFailure
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


def _assignments(func: ir.Function) -> dict[str, ir.Expr]:
    return {
        stmt.var.name.removesuffix("_0"): stmt.value
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt)
    }


def _assert_constant(expr, expected, dtype):
    if dtype == ir.DataType.BOOL:
        expected_type = ir.ConstBool
    elif dtype.is_float():
        expected_type = ir.ConstFloat
    else:
        expected_type = ir.ConstInt
    assert isinstance(expr, expected_type)
    assert expr.type.dtype == dtype
    if dtype.is_float():
        assert expr.value == pytest.approx(expected)
    else:
        assert expr.value == expected


def test_pl_const_uses_explicit_dtype_before_literal_range_checks():
    fp64 = ir.DataType.FP64

    @pl.jit(auto_mutex=False)
    def typed_literals(_jit_entry: pl.DT_INT64):
        large_float = pl.const(1e200, fp64)
        negative_float = pl.const(-1e200, fp64)
        large_integer = pl.const(1000000000000000000000000000000, pl.DT_FP32)
        negative_integer = pl.const(-1000000000000000000000000000000, pl.DT_FP32)
        negative_bool = pl.const(-True, pl.DT_INT32)

    program, _ = typed_literals.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    values = _assignments(program.get_function(typed_literals.__name__))
    _assert_constant(values["large_float"], 1e200, fp64)
    _assert_constant(values["negative_float"], -1e200, fp64)
    _assert_constant(values["large_integer"], 1e30, ir.DataType.FP32)
    _assert_constant(values["negative_integer"], -1e30, ir.DataType.FP32)
    _assert_constant(values["negative_bool"], -1, ir.DataType.INT32)


def test_pl_const_rejects_a_folded_constant_expression():
    @pl.jit(auto_mutex=False)
    def folded_const(_jit_entry: pl.DT_INT64):
        base = 5
        typed = pl.const(base + 2, pl.DT_INT32)

    with pytest.raises(InvalidType, match="numeric literal"):
        folded_const.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_pl_const_rejects_a_runtime_expression():
    @pl.jit(auto_mutex=False)
    def runtime_const(value: pl.DT_INT64):
        typed = pl.const(value, pl.DT_INT32)

    with pytest.raises(InvalidType, match="numeric literal"):
        runtime_const.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_tile_type_shapes_do_not_emit_tuple_anchors():
    @pl.jit(auto_mutex=False)
    def tile_shapes(_jit_entry: pl.DT_INT64):
        descriptor = pl.TileType(shape=[8, 16], valid_shape=(4, 16), dtype=pl.DT_FP32)
        tile = pl.make_tile(descriptor, addr=0)

    program, _ = tile_shapes.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    values = _assignments(program.get_function(tile_shapes.__name__))
    assert list(values["tile"].type.shape) == [8, 16]
    assert not any(name.startswith("_tuple_anchor_") for name in values)


def test_runtime_tuple_still_emits_an_anchor():
    @pl.jit(auto_mutex=False)
    def tuple_index(index: pl.DT_INT64):
        value = (11, 22)[index]

    program, _ = tuple_index.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    values = _assignments(program.get_function(tuple_index.__name__))
    assert any(name.startswith("_tuple_anchor_") for name in values)


def test_all_binary_and_comparison_operators_fold():
    @pl.jit(auto_mutex=False)
    def folded_ops(_jit_entry: pl.DT_INT64):
        add = 7 + 3
        sub = 7 - 3
        mul = 7 * 3
        truediv = 7 / 2
        floordiv = 7 // 3
        mod = 7 % 3
        bit_and = 7 & 3
        bit_or = 4 | 3
        bit_xor = 7 ^ 3
        shift_left = 3 << 2
        shift_right = 12 >> 2
        eq = 3 == 3
        ne = 3 != 4
        lt = 3 < 4
        le = 3 <= 3
        gt = 4 > 3
        ge = 4 >= 4

    folded_ops_program, _ = folded_ops.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    folded_ops = folded_ops_program.get_function(folded_ops.__name__)

    values = _assignments(folded_ops)
    int_expected = {
        "add": 10,
        "sub": 4,
        "mul": 21,
        "floordiv": 2,
        "mod": 1,
        "bit_and": 3,
        "bit_or": 7,
        "bit_xor": 4,
        "shift_left": 12,
        "shift_right": 3,
    }
    for name, expected in int_expected.items():
        _assert_constant(values[name], expected, ir.DataType.INT64)
    _assert_constant(values["truediv"], 3.5, ir.DataType.FP32)
    for name in ("eq", "ne", "lt", "le", "gt", "ge"):
        _assert_constant(values[name], True, ir.DataType.BOOL)


def test_mixed_numeric_constants_fold_to_fp32():
    @pl.jit(auto_mutex=False)
    def folded_numeric(_jit_entry: pl.DT_INT64):
        float_add = 1.25 + 2.5
        mixed_add = pl.const(2, pl.DT_INT32) + 0.5
        mixed_mul = 2 * pl.const(1.5, pl.DT_FP16)
        mixed_compare = pl.const(2, pl.DT_INT32) < 2.5
        typed_float = pl.const(2, pl.DT_FP32)
        typed_bool = pl.const(0, pl.DT_BOOL)

    folded_numeric_program, _ = folded_numeric.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    folded_numeric = folded_numeric_program.get_function(folded_numeric.__name__)

    values = _assignments(folded_numeric)
    _assert_constant(values["float_add"], 3.75, ir.DataType.FP32)
    _assert_constant(values["mixed_add"], 2.5, ir.DataType.FP32)
    _assert_constant(values["mixed_mul"], 3.0, ir.DataType.FP32)
    _assert_constant(values["mixed_compare"], True, ir.DataType.BOOL)
    _assert_constant(values["typed_float"], 2.0, ir.DataType.FP32)
    _assert_constant(values["typed_bool"], False, ir.DataType.BOOL)


def test_bool_numeric_and_unary_result_dtypes():
    @pl.jit(auto_mutex=False)
    def folded_bool_numeric(_jit_entry: pl.DT_INT64):
        add = True + True
        mixed_add = True + 0.5
        truediv = True / 2
        floordiv = True // True
        mod = True % True
        bit_bool = True & False
        bit_int = True | 2
        shift = True << True
        unary_plus = +True
        unary_minus = -True
        unary_invert = ~True
        unary_not = not 0.0
        float_plus = +1.5
        float_minus = -1.5
        int_not = not 3
        int_invert = ~3
        compare = False < 1

    folded_bool_numeric_program, _ = folded_bool_numeric.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    folded_bool_numeric = folded_bool_numeric_program.get_function(folded_bool_numeric.__name__)

    values = _assignments(folded_bool_numeric)
    _assert_constant(values["add"], 2, ir.DataType.INT64)
    _assert_constant(values["mixed_add"], 1.5, ir.DataType.FP32)
    _assert_constant(values["truediv"], 0.5, ir.DataType.FP32)
    _assert_constant(values["floordiv"], 1, ir.DataType.INT64)
    _assert_constant(values["mod"], 0, ir.DataType.INT64)
    _assert_constant(values["bit_bool"], 0, ir.DataType.INT64)
    _assert_constant(values["bit_int"], 3, ir.DataType.INT64)
    _assert_constant(values["shift"], 2, ir.DataType.INT64)
    _assert_constant(values["unary_plus"], True, ir.DataType.BOOL)
    _assert_constant(values["unary_minus"], -1, ir.DataType.INT64)
    _assert_constant(values["unary_invert"], -2, ir.DataType.INT64)
    _assert_constant(values["unary_not"], True, ir.DataType.BOOL)
    _assert_constant(values["float_plus"], 1.5, ir.DataType.FP32)
    _assert_constant(values["float_minus"], -1.5, ir.DataType.FP32)
    _assert_constant(values["int_not"], False, ir.DataType.BOOL)
    _assert_constant(values["int_invert"], -4, ir.DataType.INT64)
    _assert_constant(values["compare"], True, ir.DataType.BOOL)


def test_bool_ops_accept_bool_int_and_float_truthiness_but_return_bool():
    @pl.jit(auto_mutex=False)
    def folded_truth(_jit_entry: pl.DT_INT64):
        int_and = 2 and 0
        int_or = 0 or 3
        float_and = 1.5 and 0.0
        float_or = 0.0 or -2.5
        short_and = False and missing_name  # noqa: F821
        short_or = True or missing_name  # noqa: F821

    folded_truth_program, _ = folded_truth.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    folded_truth = folded_truth_program.get_function(folded_truth.__name__)

    values = _assignments(folded_truth)
    _assert_constant(values["int_and"], False, ir.DataType.BOOL)
    _assert_constant(values["int_or"], True, ir.DataType.BOOL)
    _assert_constant(values["float_and"], False, ir.DataType.BOOL)
    _assert_constant(values["float_or"], True, ir.DataType.BOOL)
    _assert_constant(values["short_and"], False, ir.DataType.BOOL)
    _assert_constant(values["short_or"], True, ir.DataType.BOOL)


def test_pl_and_builtin_min_max_fold_with_numeric_promotion():
    @pl.jit(auto_mutex=False)
    def folded_min_max(_jit_entry: pl.DT_INT64):
        pl_min = pl.min(3, 2)
        pl_max = pl.max(1.5, 2)
        builtin_min = min(True, 2)
        builtin_max = max(pl.const(1, pl.DT_INT32), 2.5)

    folded_min_max_program, _ = folded_min_max.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    folded_min_max = folded_min_max_program.get_function(folded_min_max.__name__)

    values = _assignments(folded_min_max)
    _assert_constant(values["pl_min"], 2, ir.DataType.INT64)
    _assert_constant(values["pl_max"], 2.0, ir.DataType.FP32)
    _assert_constant(values["builtin_min"], 1, ir.DataType.INT64)
    _assert_constant(values["builtin_max"], 2.5, ir.DataType.FP32)


def test_invalid_shift_keeps_validated_ir_node():
    @pl.jit(auto_mutex=False)
    def unsafe_constants(_jit_entry: pl.DT_INT64):
        negative_shift = 1 << -1

    unsafe_constants_program, _ = unsafe_constants.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    unsafe_constants = unsafe_constants_program.get_function(unsafe_constants.__name__)

    values = _assignments(unsafe_constants)
    assert isinstance(values["negative_shift"], ir.BitShiftLeft)


def test_compile_time_zero_divisor_is_rejected():
    @pl.jit(auto_mutex=False)
    def truediv_zero(value: pl.DT_INT32):
        result = value / 0

    @pl.jit(auto_mutex=False)
    def floordiv_zero(value: pl.DT_INT32):
        result = value // 0

    @pl.jit(auto_mutex=False)
    def modulo_zero(value: pl.DT_INT32):
        result = value % -0.0

    for kernel, operator in ((truediv_zero, "/"), (floordiv_zero, "//"), (modulo_zero, "%")):
        with pytest.raises(InvalidVal, match=rf"Operator '{operator}' does not allow a zero divisor"):
            kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16, pl.DT_FP32])
@pytest.mark.parametrize(
    "lhs,rhs",
    [
        (-7.0, 2.0),
        (7.0, -2.0),
        (-7.0, -2.0),
        (1.5, 0.5),
        (1.0, 0.1),
        (0.0, -2.0),
        (-0.0, 2.0),
    ],
)
def test_float_divmod_constants_fold_with_promoted_dtype(dtype, lhs, rhs):
    left = ir.ConstFloat(lhs, dtype, ir.Span.unknown())
    right = ir.ConstFloat(rhs, dtype, ir.Span.unknown())

    @pl.jit(auto_mutex=False)
    def folded_float(_jit_entry: pl.DT_INT64):
        quotient = left // right
        remainder = left % right

    program, _ = folded_float.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    values = _assignments(program.get_function(folded_float.__name__))
    _assert_constant(values["quotient"], lhs // rhs, dtype)
    _assert_constant(values["remainder"], lhs % rhs, dtype)
    assert math.copysign(1.0, values["quotient"].value) == math.copysign(1.0, lhs // rhs)
    assert math.copysign(1.0, values["remainder"].value) == math.copysign(1.0, lhs % rhs)


def test_float_divmod_constants_can_define_static_tile_shape():
    @pl.jit(auto_mutex=False)
    def static_float_shape(_jit_entry: pl.DT_INT64):
        cols = 16.0 // 2.0
        rows = 7.0 % 2.0
        tile_type = pl.TileType(shape=[int(rows), int(cols)], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0)

    program, _ = static_float_shape.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    values = _assignments(program.get_function(static_float_shape.__name__))
    assert list(values["tile"].type.shape) == [1, 8]


def test_nonconstant_operands_keep_runtime_ir():
    @pl.jit(auto_mutex=False)
    def runtime_ops(value: pl.DT_INT32, flag: pl.DT_BOOL):
        add = value + 1
        minimum = pl.min(value, 2)
        truth = value and 1.0
        positive = +value
        positive_bool = +flag

    runtime_ops_program, _ = runtime_ops.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    runtime_ops = runtime_ops_program.get_function(runtime_ops.__name__)

    values = _assignments(runtime_ops)
    assert isinstance(values["add"], ir.Add)
    assert isinstance(values["minimum"], ir.Min)
    assert isinstance(values["truth"], ir.And)
    assert values["truth"].type.dtype == ir.DataType.BOOL
    assert isinstance(values["positive"], ir.Var)
    assert values["positive"].type.dtype == ir.DataType.INT32
    assert isinstance(values["positive_bool"], ir.Var)
    assert values["positive_bool"].type.dtype == ir.DataType.BOOL


def test_inline_helper_constant_conditions_fold_independently_at_multiple_call_sites():
    def choose(cond):
        if cond == 0:
            return 11
        if cond == 1:
            return 22
        if cond == 2:
            return 33
        return 44

    @pl.jit(auto_mutex=False)
    def folded_calls(_jit_entry: pl.DT_INT64):
        first = choose(0)
        second = choose(1)
        third = choose(2)
        fallback = choose(9)
        first_is_constant = first == 11
        second_is_constant = second == 22
        third_is_constant = third == 33
        fallback_is_constant = fallback == 44
        combined = first + second + third + fallback

    program, _ = folded_calls.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    values = _assignments(program.get_function(folded_calls.__name__))

    for name in (
        "first_is_constant",
        "second_is_constant",
        "third_is_constant",
        "fallback_is_constant",
    ):
        _assert_constant(values[name], True, ir.DataType.BOOL)
    _assert_constant(values["combined"], 110, ir.DataType.INT64)


def test_unary_plus_is_identity_for_non_scalar_operand():
    @pl.jit(auto_mutex=False)
    def identity_pos(value: pl.Tensor[[1], pl.DT_FP32]):
        result = +value

    identity_pos_program, _ = identity_pos.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    identity_pos = identity_pos_program.get_function(identity_pos.__name__)

    result = _assignments(identity_pos)["result"]
    assert isinstance(result, ir.Var)
    assert isinstance(result.type, ir.TensorType)


def test_invalid_constant_operand_types_raise_from_ir_builders():
    with pytest.raises(RuntimeFailure, match="bit_not.*integer dtype"):

        @pl.jit(auto_mutex=False)
        def invalid_invert(_jit_entry: pl.DT_INT64):
            value = ~1.5

        invalid_invert.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    with pytest.raises(RuntimeFailure, match="bit_and.*integer dtype"):

        @pl.jit(auto_mutex=False)
        def invalid_bit_and(_jit_entry: pl.DT_INT64):
            value = 1.5 & 1

        invalid_bit_and.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
