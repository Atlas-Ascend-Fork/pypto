#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO Pro SIMT namespace (``pl.simt.*``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.simt.xxx`` call
- Python catches typos at import time
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.simt.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.
"""

from __future__ import annotations

from typing import Any, Union

from pypto.ir import RoundMode

from ._api import DType, Scalar, _api_decl


class Simt:
    """SIMT namespace (``pl.simt.*``).

    Provides SIMT context queries, synchronization, scalar operations, and atomics.
    """

    @staticmethod
    @_api_decl
    def thread_idx() -> Any:
        """Return the current thread coordinates within its thread block."""

    @staticmethod
    @_api_decl
    def block_dim() -> Any:
        """Return the dimensions of the current thread block."""

    @staticmethod
    @_api_decl
    def block_idx() -> Any:
        """Return the current block coordinates within the outer kernel grid."""

    @staticmethod
    @_api_decl
    def grid_dim() -> Any:
        """Return the dimensions of the outer kernel grid."""

    @staticmethod
    @_api_decl
    def linear_thread_idx() -> Scalar:
        """Return the x-major flattened thread index within the current block."""

    @staticmethod
    @_api_decl
    def warp_size() -> int:
        """Return the target SIMT warp size."""

    @staticmethod
    @_api_decl
    def syncthreads() -> None:
        """Synchronize all threads in the current block.

        The barrier must be reached uniformly by every thread in the block. It
        cannot be placed inside runtime ``if``, ``for``, or ``while`` control flow.
        This no-return operation must be used as a standalone statement.
        """

    @staticmethod
    @_api_decl
    def threadfence_block() -> None:
        """Order this thread's memory operations for threads in the current block.

        This memory fence does not wait for other threads and may be used in
        runtime control flow. It must be used as a standalone statement.
        """

    @staticmethod
    @_api_decl
    def threadfence() -> None:
        """Order this thread's memory operations with device-wide visibility.

        This memory fence does not wait for other threads and may be used in
        runtime control flow. It must be used as a standalone statement.
        """

    @staticmethod
    @_api_decl
    def lane_id() -> int:
        """Return the current lane index in the warp as an INT32 Scalar."""

    @staticmethod
    @_api_decl
    def lanemask_eq() -> int:
        """Return an INT32 mask containing only the current lane bit."""

    @staticmethod
    @_api_decl
    def lanemask_le() -> int:
        """Return an INT32 mask containing lanes up to and including the current lane."""

    @staticmethod
    @_api_decl
    def lanemask_lt() -> int:
        """Return an INT32 mask containing lanes below the current lane."""

    @staticmethod
    @_api_decl
    def lanemask_ge() -> int:
        """Return an INT32 mask containing lanes at or above the current lane."""

    @staticmethod
    @_api_decl
    def lanemask_gt() -> int:
        """Return an INT32 mask containing lanes above the current lane."""

    @staticmethod
    @_api_decl
    def warp_all(predicate: int) -> int:
        """Return INT32 one when predicate is true for every active lane, otherwise zero."""

    @staticmethod
    @_api_decl
    def warp_any(predicate: int) -> int:
        """Return INT32 one when predicate is true for any active lane, otherwise zero."""

    @staticmethod
    @_api_decl
    def warp_ballot(predicate: int) -> int:
        """Return a UINT32 bit mask of active lanes whose predicate is true."""

    @staticmethod
    @_api_decl
    def warp_active_mask() -> int:
        """Return a UINT32 bit mask of the currently active lanes."""

    @staticmethod
    @_api_decl
    def warp_shfl(value: Union[int, float], src_lane: int, width: int = 32) -> Scalar:
        """Read value from src_lane in the current logical warp subgroup, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_shfl_up(value: Union[int, float], delta: int, width: int = 32) -> Scalar:
        """Read value from a lower lane by delta within the subgroup, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_shfl_down(value: Union[int, float], delta: int, width: int = 32) -> Scalar:
        """Read value from a higher lane by delta within the subgroup, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_shfl_xor(value: Union[int, float], lane_mask: int, width: int = 32) -> Scalar:
        """Read value from the lane selected by XOR with lane_mask, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_reduce_add(value: Union[int, float]) -> Scalar:
        """Return the same-dtype sum of value across the active lanes in the warp."""

    @staticmethod
    @_api_decl
    def warp_reduce_max(value: Union[int, float]) -> Scalar:
        """Return the same-dtype maximum value across the active lanes in the warp."""

    @staticmethod
    @_api_decl
    def warp_reduce_min(value: Union[int, float]) -> Scalar:
        """Return the same-dtype minimum value across the active lanes in the warp."""

    @staticmethod
    @_api_decl
    def cast(
        value: Union[int, float],
        dtype: DType,
        *,
        mode: RoundMode = RoundMode.CAST_NONE,
    ) -> Scalar:
        """Convert one scalar expression to ``dtype`` inside a SIMT function.

        ``mode`` controls rounding when the target dtype cannot represent ``value``
        exactly. Low-precision floating-point conversions added beyond the default
        paths require an explicit supported mode. ``CAST_ODD`` is supported only
        for FP32-to-FP16 conversion. Saturation mode selection is not yet exposed.
        """

    @staticmethod
    @_api_decl
    def bitcast(value: Union[int, float], dtype: DType) -> Scalar:
        """Reinterpret one scalar bit pattern as a supported equal-width dtype.

        Unlike :meth:`cast`, this operation performs no numerical conversion,
        rounding, or saturation. Only the explicitly supported FP16/BF16 and
        FP32 integer bit-reinterpretation pairs are accepted.
        """

    @staticmethod
    @_api_decl
    def abs(value: Union[int, float]) -> Scalar:
        """Return the absolute value of a FP16, BF16, FP32, INT32, or INT64 Scalar."""

    @staticmethod
    @_api_decl
    def min(lhs: Union[int, float], rhs: Union[int, float]) -> Scalar:
        """Return the minimum of two same-dtype floating-point or integer Scalars."""

    @staticmethod
    @_api_decl
    def max(lhs: Union[int, float], rhs: Union[int, float]) -> Scalar:
        """Return the maximum of two same-dtype floating-point or integer Scalars."""

    @staticmethod
    @_api_decl
    def sqrt(value: float) -> float:
        """Return the square root of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def rsqrt(value: float) -> float:
        """Return the reciprocal square root of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def exp(value: float) -> float:
        """Return e raised to a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def exp2(value: float) -> float:
        """Return two raised to a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log(value: float) -> float:
        """Return the natural logarithm of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log2(value: float) -> float:
        """Return the base-two logarithm of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log1p(value: float) -> float:
        """Return the FP32 natural logarithm of one plus ``value``."""

    @staticmethod
    @_api_decl
    def sin(value: float) -> float:
        """Return the sine of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def cos(value: float) -> float:
        """Return the cosine of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def tanh(value: float) -> float:
        """Return the hyperbolic tangent of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def rint(value: float, *, dtype: DType | None = None) -> Scalar:
        """Round a floating-point Scalar to nearest-even.

        The result keeps the input dtype by default. Set ``dtype`` to
        ``pl.DT_INT64`` to return a saturated INT64 value.
        """

    @staticmethod
    @_api_decl
    def round(value: float, *, dtype: DType | None = None) -> Scalar:
        """Round a floating-point Scalar halfway away from zero.

        The result keeps the input dtype by default. Set ``dtype`` to
        ``pl.DT_INT64`` to return a saturated INT64 value.
        """

    @staticmethod
    @_api_decl
    def floor(value: float) -> float:
        """Round a floating-point Scalar down to an integer value."""

    @staticmethod
    @_api_decl
    def ceil(value: float) -> float:
        """Round a floating-point Scalar up to an integer value."""

    @staticmethod
    @_api_decl
    def trunc(value: float) -> float:
        """Round a floating-point Scalar toward zero to an integer value."""

    @staticmethod
    @_api_decl
    def isnan(value: float) -> bool:
        """Return a BOOL Scalar indicating whether a floating-point Scalar is NaN."""

    @staticmethod
    @_api_decl
    def isinf(value: float) -> bool:
        """Return a BOOL Scalar indicating whether a floating-point Scalar is infinite."""

    @staticmethod
    @_api_decl
    def isfinite(value: float) -> bool:
        """Test whether an FP16 or FP32 Scalar is finite, returning BOOL."""

    @staticmethod
    @_api_decl
    def popcount(value: int) -> int:
        """Count the set bits in a UINT32 or UINT64 Scalar, returning INT32."""

    @staticmethod
    @_api_decl
    def mul_hi(lhs: int, rhs: int) -> int:
        """Return the high half of the full product of two same-dtype integers.

        Supports INT32, UINT32, INT64, and UINT64; the result has the input dtype.
        """

    @staticmethod
    @_api_decl
    def exp10(value: float) -> float:
        """Return ten raised to a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log10(value: float) -> float:
        """Return the base-ten logarithm of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def rcp(value: float) -> float:
        """Return the reciprocal of a FP16 or BF16 Scalar."""

    @staticmethod
    @_api_decl
    def tan(value: float) -> float:
        """Return the tangent of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def atan(value: float) -> float:
        """Return the arc tangent of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def expm1(value: float) -> float:
        """Return the FP32 natural exponential of ``value`` minus one."""

    @staticmethod
    @_api_decl
    def logb(value: float) -> float:
        """Return the unbiased exponent of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def cosh(value: float) -> float:
        """Return the hyperbolic cosine of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def acos(value: float) -> float:
        """Return the arc cosine of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def sinh(value: float) -> float:
        """Return the hyperbolic sine of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def asin(value: float) -> float:
        """Return the arc sine of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def cbrt(value: float) -> float:
        """Return the cube root of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def max_nan(lhs: float, rhs: float) -> float:
        """Return the maximum FP16 or BF16 operand and propagate NaN."""

    @staticmethod
    @_api_decl
    def min_nan(lhs: float, rhs: float) -> float:
        """Return the minimum FP16 or BF16 operand and propagate NaN."""

    @staticmethod
    @_api_decl
    def atan2(y: float, x: float) -> float:
        """Return the FP32 arc tangent of ``y / x`` using the signs to select the quadrant."""

    @staticmethod
    @_api_decl
    def copysign(magnitude: float, sign: float) -> float:
        """Return the FP32 magnitude of ``magnitude`` with the sign of ``sign``."""

    @staticmethod
    @_api_decl
    def nextafter(value: float, direction: float) -> float:
        """Return the next representable FP32 value from ``value`` toward ``direction``."""

    @staticmethod
    @_api_decl
    def tanpi(value: float) -> float:
        """Return the FP32 tangent of ``pi * value``."""

    @staticmethod
    @_api_decl
    def atanh(value: float) -> float:
        """Return the FP32 inverse hyperbolic tangent."""

    @staticmethod
    @_api_decl
    def cospi(value: float) -> float:
        """Return the FP32 cosine of ``pi * value``."""

    @staticmethod
    @_api_decl
    def acosh(value: float) -> float:
        """Return the FP32 inverse hyperbolic cosine."""

    @staticmethod
    @_api_decl
    def sinpi(value: float) -> float:
        """Return the FP32 sine of ``pi * value``."""

    @staticmethod
    @_api_decl
    def asinh(value: float) -> float:
        """Return the FP32 inverse hyperbolic sine."""

    @staticmethod
    @_api_decl
    def rcbrt(value: float) -> float:
        """Return the reciprocal FP32 cube root."""

    @staticmethod
    @_api_decl
    def ilogb(value: float) -> int:
        """Return the INT32 unbiased exponent of an FP32 Scalar."""

    @staticmethod
    @_api_decl
    def signbit(value: float) -> int:
        """Return the FP32 sign bit as INT32 zero or one."""

    @staticmethod
    @_api_decl
    def fmod(lhs: float, rhs: float) -> float:
        """Return the FP32 remainder with a quotient truncated toward zero.

        Both operands must be FP32. The result retains the dividend's sign,
        including signed zero. A zero divisor, infinite dividend, or NaN operand
        produces NaN; a finite dividend modulo infinity returns the dividend.
        """

    @staticmethod
    @_api_decl
    def fma(lhs: float, rhs: float, addend: float) -> float:
        """Fused-multiply-add three same-dtype FP16, BF16, or FP32 Scalars."""

    @staticmethod
    @_api_decl
    def atomic_add(target: Union[int, float], value: Union[int, float]) -> Scalar | None:
        """Atomically add ``value`` to one Tile or Tensor element.

        ``target`` must be written directly as a subscript expression such as
        ``tile[row, col]`` or ``tensor[index]``. FP16/BF16 targets return no value;
        other supported dtypes return the element value observed before the update.
        """

    @staticmethod
    @_api_decl
    def atomic_sub(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically subtract ``value`` from one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_exch(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically replace one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_max(target: Union[int, float], value: Union[int, float]) -> Scalar | None:
        """Atomically update an element with its maximum; FP16/BF16 return no value."""

    @staticmethod
    @_api_decl
    def atomic_min(target: Union[int, float], value: Union[int, float]) -> Scalar | None:
        """Atomically update an element with its minimum; FP16/BF16 return no value."""

    @staticmethod
    @_api_decl
    def atomic_inc(target: Union[int, float], limit: Union[int, float]) -> Scalar:
        """Atomically increment and wrap one unsigned counter element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_dec(target: Union[int, float], limit: Union[int, float]) -> Scalar:
        """Atomically decrement and wrap one unsigned counter element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_cas(target: Union[int, float], compare: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically compare and exchange one Tile or Tensor element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_and(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically apply bitwise AND to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_or(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically apply bitwise OR to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_xor(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically apply bitwise XOR to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def brev(value: Scalar) -> Scalar:
        """Reverse the bits of a UINT32 or UINT64 Scalar."""

    @staticmethod
    @_api_decl
    def byte_perm(lhs: Scalar, rhs: Scalar, selector: Scalar) -> Scalar:
        """Select four bytes from two UINT32 Scalars according to a UINT32 selector."""

    @staticmethod
    @_api_decl
    def clz(value: Scalar) -> Scalar:
        """Count leading zero bits in an INT32 or INT64 Scalar, returning INT32."""

    @staticmethod
    @_api_decl
    def cyl_bessel_i0(value: Scalar) -> Scalar:
        """Return the order-zero modified cylindrical Bessel function."""

    @staticmethod
    @_api_decl
    def cyl_bessel_i1(value: Scalar) -> Scalar:
        """Return the order-one modified cylindrical Bessel function."""

    @staticmethod
    @_api_decl
    def equ(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return whether FP16 or BF16 operands are equal or unordered."""

    @staticmethod
    @_api_decl
    def erf(value: Scalar) -> Scalar:
        """Return the FP32 error function."""

    @staticmethod
    @_api_decl
    def erfc(value: Scalar) -> Scalar:
        """Return the FP32 complementary error function."""

    @staticmethod
    @_api_decl
    def erfcinv(value: Scalar) -> Scalar:
        """Return the FP32 inverse complementary error function."""

    @staticmethod
    @_api_decl
    def erfcx(value: Scalar) -> Scalar:
        """Return the FP32 scaled complementary error function."""

    @staticmethod
    @_api_decl
    def erfinv(value: Scalar) -> Scalar:
        """Return the FP32 inverse error function."""

    @staticmethod
    @_api_decl
    def fdim(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return ``lhs - rhs`` for positive FP32 differences and positive zero otherwise."""

    @staticmethod
    @_api_decl
    def ffs(value: Scalar) -> Scalar:
        """Return the one-based index of the least-significant set bit, returning INT32."""

    @staticmethod
    @_api_decl
    def fma_relu(lhs: Scalar, rhs: Scalar, addend: Scalar) -> Scalar:
        """Fused-multiply-add FP16 or BF16 Scalars and clamp negative results to zero."""

    @staticmethod
    @_api_decl
    def geu(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return whether FP16 or BF16 operands are greater-or-equal or unordered."""

    @staticmethod
    @_api_decl
    def gtu(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return whether FP16 or BF16 operands are greater-than or unordered."""

    @staticmethod
    @_api_decl
    def hadd(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return the average of two same-dtype INT32 or UINT32 Scalars."""

    @staticmethod
    @_api_decl
    def hypot(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return the overflow-resistant FP32 square root of ``lhs**2 + rhs**2``."""

    @staticmethod
    @_api_decl
    def j0(value: Scalar) -> Scalar:
        """Return the order-zero FP32 Bessel function of the first kind."""

    @staticmethod
    @_api_decl
    def j1(value: Scalar) -> Scalar:
        """Return the order-one FP32 Bessel function of the first kind."""

    @staticmethod
    @_api_decl
    def jn(order: Scalar, value: Scalar) -> Scalar:
        """Return the integer-order FP32 Bessel function of the first kind."""

    @staticmethod
    @_api_decl
    def ldexp(value: Scalar, exponent: Scalar) -> Scalar:
        """Scale an FP32 value by two raised to an INT32 exponent."""

    @staticmethod
    @_api_decl
    def leu(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return whether FP16 or BF16 operands are less-or-equal or unordered."""

    @staticmethod
    @_api_decl
    def lgamma(value: Scalar) -> Scalar:
        """Return the natural logarithm of the absolute FP32 Gamma function."""

    @staticmethod
    @_api_decl
    def ltu(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return whether FP16 or BF16 operands are less-than or unordered."""

    @staticmethod
    @_api_decl
    def mul24(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Multiply the low 24 bits of two same-dtype INT32 or UINT32 Scalars."""

    @staticmethod
    @_api_decl
    def mul_i32_to_i64(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Multiply two same-dtype INT32 or UINT32 Scalars, returning INT64 or UINT64."""

    @staticmethod
    @_api_decl
    def norm3d(x: Scalar, y: Scalar, z: Scalar) -> Scalar:
        """Return the overflow-resistant Euclidean norm of three FP32 Scalars."""

    @staticmethod
    @_api_decl
    def norm4d(x: Scalar, y: Scalar, z: Scalar, w: Scalar) -> Scalar:
        """Return the overflow-resistant Euclidean norm of four FP32 Scalars."""

    @staticmethod
    @_api_decl
    def normcdf(value: Scalar) -> Scalar:
        """Return the standard normal FP32 cumulative distribution function."""

    @staticmethod
    @_api_decl
    def normcdfinv(value: Scalar) -> Scalar:
        """Return the inverse standard normal FP32 cumulative distribution function."""

    @staticmethod
    @_api_decl
    def pow(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return FP32 ``lhs ** rhs``.

        Both operands must be FP32 Scalars. Finite negative bases require
        integer exponents; other finite exponents produce NaN. Zero, infinity
        and NaN follow ``powf`` special-value semantics, including a result of
        one when ``rhs == 0`` or ``lhs == 1``.
        """

    @staticmethod
    @_api_decl
    def remainder(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return the FP32 IEEE remainder using a nearest-even integer quotient."""

    @staticmethod
    @_api_decl
    def rhadd(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return the upward-rounded average of two same-dtype INT32 or UINT32 Scalars."""

    @staticmethod
    @_api_decl
    def rhypot(lhs: Scalar, rhs: Scalar) -> Scalar:
        """Return the reciprocal of the FP32 hypotenuse of two Scalars."""

    @staticmethod
    @_api_decl
    def rnorm3d(x: Scalar, y: Scalar, z: Scalar) -> Scalar:
        """Return the reciprocal Euclidean norm of three FP32 Scalars."""

    @staticmethod
    @_api_decl
    def rnorm4d(x: Scalar, y: Scalar, z: Scalar, w: Scalar) -> Scalar:
        """Return the reciprocal Euclidean norm of four FP32 Scalars."""

    @staticmethod
    @_api_decl
    def sad(lhs: Scalar, rhs: Scalar, addend: Scalar) -> Scalar:
        """Return ``abs(lhs - rhs) + addend`` as UINT32."""

    @staticmethod
    @_api_decl
    def saturate(value: Scalar) -> Scalar:
        """Clamp FP32 to ``[0, 1]``, mapping NaN to zero."""

    @staticmethod
    @_api_decl
    def scalbn(value: Scalar, exponent: Scalar) -> Scalar:
        """Scale an FP32 value by two raised to an INT32 or INT64 exponent."""

    @staticmethod
    @_api_decl
    def tgamma(value: Scalar) -> Scalar:
        """Return the FP32 Gamma function."""

    @staticmethod
    @_api_decl
    def y0(value: Scalar) -> Scalar:
        """Return the order-zero FP32 Bessel function of the second kind."""

    @staticmethod
    @_api_decl
    def y1(value: Scalar) -> Scalar:
        """Return the order-one FP32 Bessel function of the second kind."""

    @staticmethod
    @_api_decl
    def yn(order: Scalar, value: Scalar) -> Scalar:
        """Return the integer-order FP32 Bessel function of the second kind."""
