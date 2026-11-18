# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT of the AICPU status-ABI return rules (``return <int>`` in aicpu kernels).

Contract (mirroring the handwritten AICPU ``return KERNEL_STATUS_*``):

* An aicpu kernel may ``return <int>`` -- the status code (0 = OK,
  1 = PARAM_INVALID) the entry wrappers propagate.
* A bare ``return`` stays the normal OK exit; float/bool/tensor values are
  not status codes and are rejected.
* Non-aicpu void contexts (plain simd kernels) keep rejecting valued
  returns -- the status channel is aicpu-only.
"""

from pypto_pro import ir
from pypto_pro._errors import InvalidVal, NotSupported
import pypto_pro.language as pl
import pytest


def _parse_aicpu(kernel):
    kdef = kernel.to_kernel_def()
    kdef.parse_target_program(ir.SectionKind.Vector)


def test_aicpu_kernel_accepts_integer_status_return():
    @pl.jit(mode="aicpu_kernel")
    def kernel(out: pl.Tensor[[4], pl.DT_INT32, pl.Output], n: pl.DT_INT64):
        if n < 0:
            return 1
        pl.setval(out, 0, n)

    _parse_aicpu(kernel)


def test_aicpu_kernel_mixes_status_and_bare_returns():
    @pl.jit(mode="aicpu_kernel")
    def kernel(out: pl.Tensor[[4], pl.DT_INT32, pl.Output], n: pl.DT_INT64):
        if n < 0:
            return 1
        if n == 5:
            pl.setval(out, 0, 5)
            return
        pl.setval(out, 0, n)

    _parse_aicpu(kernel)


def test_aicpu_kernel_accepts_dynamic_status_expression():
    @pl.jit(mode="aicpu_kernel")
    def kernel(out: pl.Tensor[[4], pl.DT_INT32, pl.Output], code: pl.DT_INT64):
        if code != 0:
            return code
        pl.setval(out, 0, code)

    _parse_aicpu(kernel)


@pytest.mark.parametrize("bad", [1.5, True])
def test_aicpu_kernel_rejects_non_integer_status(bad):
    @pl.jit(mode="aicpu_kernel")
    def kernel(out: pl.Tensor[[4], pl.DT_INT32, pl.Output], n: pl.DT_INT64):
        if n < 0:
            return bad
        pl.setval(out, 0, n)

    with pytest.raises(InvalidVal, match="integer KERNEL_STATUS code"):
        _parse_aicpu(kernel)


def test_simd_kernel_still_rejects_valued_return():
    @pl.jit()
    def kernel(out: pl.Tensor[[4], pl.DT_INT32, pl.Output], n: pl.DT_INT64):
        if n < 0:
            return 1
        pl.setval(out, 0, n)

    with pytest.raises(NotSupported, match="only supports bare return or return None"):
        _parse_aicpu(kernel)
