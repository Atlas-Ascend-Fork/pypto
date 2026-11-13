# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check PGO VF availability through the public plugin entry point."""

from contextlib import nullcontext

import pytest

_PGO_VF_UNAVAILABLE = "PGOVFRun is not implemented: the PGO VF implementation is unavailable"


def duration_estimate(seconds: float):
    """Annotate a test case with an estimated duration (seconds)."""

    def decorator(func):
        func.duration_estimate = seconds
        return func

    return decorator


@pytest.fixture(scope="session")
def pgo_vf_available():
    from pypto_pro import ir

    span = ir.Span.unknown()
    function = ir.Function("pgo_vf_probe", [], [], ir.ReturnStmt([], span), span)
    program = ir.Program([function], "pgo_vf_probe", span)
    try:
        out_program, out_function = ir.PGOVFRun(program, None)
    except RuntimeError as error:
        if _PGO_VF_UNAVAILABLE not in str(error):
            raise
        return False
    assert out_program is not None
    assert out_function is None
    return True


@pytest.fixture
def expect_pgo_vf(pgo_vf_available):
    def expectation(required=True):
        if not required or pgo_vf_available:
            return nullcontext()
        return pytest.raises(
            RuntimeError,
            match=_PGO_VF_UNAVAILABLE,
        )

    return expectation
