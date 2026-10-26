# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small shared parser utilities."""

from __future__ import annotations

__all__ = ["_is_const_expr"]

from pypto.pypto_impl import ir


def _is_const_expr(expr: object) -> bool:
    """Return whether an IR expression is a recursively constant value."""
    if isinstance(expr, (ir.ConstInt, ir.ConstBool, ir.ConstFloat)):
        return True
    return (
        isinstance(expr, ir.MakeTuple)
        and all(_is_const_expr(element) for element in expr.elements)
    )
