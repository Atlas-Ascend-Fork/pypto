# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CCE keeps its existing zero/single-result assignment boundary."""

from pypto_pro import ir
from pypto_pro.runtime.platform import NpuArch
import pytest

from pypto.pypto_impl.codegen import CCECodegen


def test_cce_rejects_multiple_assignment_targets():
    span = ir.Span.unknown()
    scalar = ir.ScalarType(ir.DataType.INT64)
    targets = [ir.Var(name, scalar, span) for name in ("x", "y")]
    value = ir.MakeTuple([ir.ConstInt(n, ir.DataType.INT64, span) for n in (1, 2)], span)
    body = ir.SeqStmts([ir.AssignStmt(targets, value, span)], span)
    function = ir.Function("test", [], [], body, span)
    program = ir.Program([function], "test", span, ir.IRDebugInfo())
    with pytest.raises(RuntimeError, match="does not support AssignStmt with multiple targets"):
        CCECodegen(ir.SectionKind.Vector).generate_single(program, NpuArch.DAV_3510)
