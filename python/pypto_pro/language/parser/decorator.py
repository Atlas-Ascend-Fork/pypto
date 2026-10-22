# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Decorators and markers supported by the PyPTO Pro parser."""

from __future__ import annotations

__all__ = ["inline", "vector_function"]


from collections.abc import Callable

from ..._errors import InvalidVal, NotSupported, OutOfRange

_SIMT_FUNCTION_MARKER = "_pypto_simt_function"
_SIMT_MAX_THREADS_ATTR = "_pypto_simt_max_threads"


def inline(fn: Callable) -> Callable:
    """Deprecated compatibility marker for inline callables."""
    import warnings

    warnings.warn(
        "@pl.inline is deprecated and will be removed. "
        "Call ordinary helpers inside @pl.jit; use @pl.vector_function for VF or SIMT helpers.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fn


def vector_function(
    fn: Callable | None = None,
    *,
    mode: str | None = None,
    max_threads: int | None = None,
) -> Callable:
    """Mark a callable as a SIMD or SIMT vector function.

    ``@pl.vector_function`` and ``@pl.vector_function(mode="simd")`` declare
    SIMD vector functions expanded at their call sites. ``mode="simt"`` marks
    a delayed SIMT template; providing ``max_threads`` makes it launchable via
    ``fn[threads](...)``, while omitting it declares a SIMT helper.
    """
    if fn is None and mode is None and max_threads is None:
        raise NotSupported("@pl.vector_function() is not supported; use @pl.vector_function")

    actual_mode = "simd" if mode is None else mode
    if actual_mode == "simd":
        if max_threads is not None:
            raise InvalidVal("max_threads is only supported when mode='simt'")
    elif actual_mode == "simt":
        if max_threads is not None:
            if isinstance(max_threads, bool) or not isinstance(max_threads, int):
                raise InvalidVal("max_threads must be an integer")
            if not 1 <= max_threads <= 2048:
                raise OutOfRange("max_threads must be in [1, 2048]")
    else:
        raise InvalidVal("mode must be 'simd' or 'simt'")

    def decorate(func: Callable) -> Callable:
        if not callable(func):
            raise InvalidVal("@pl.vector_function can only decorate a callable")
        if actual_mode == "simd":
            _mark_vector_function(func)
        else:
            _mark_simt_function(func, max_threads)
        return func

    return decorate(fn) if fn is not None else decorate


def _mark_vector_function(fn: Callable) -> None:
    """Set the internal marker used for SIMD vector-function expansion."""
    setattr(fn, "_pypto_vector_function", True)


def _mark_simt_function(fn: Callable, max_threads: int | None) -> None:
    """Set the existing internal markers used for delayed SIMT parsing."""
    setattr(fn, _SIMT_FUNCTION_MARKER, True)
    setattr(fn, _SIMT_MAX_THREADS_ATTR, max_threads)


def is_vector_function(fn: Callable) -> bool:
    """Return whether *fn* is a SIMD vector-function body."""
    return bool(getattr(fn, "_pypto_vector_function", False))


def is_simt_function(fn: Callable) -> bool:
    """Return whether *fn* is marked as a delayed SIMT function template."""
    return bool(getattr(fn, _SIMT_FUNCTION_MARKER, False))


def get_simt_max_threads(fn: Callable) -> int | None:
    """Return the launch bound recorded for a delayed SIMT function."""
    return getattr(fn, _SIMT_MAX_THREADS_ATTR, None)
