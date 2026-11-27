# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement.
# You can use this software in accordance with the terms and conditions of
# the License. See the License for details.

"""Host interfaces rejected on the aicpu kernel surface, one case per interface.

Two rejection layers, both before any bisheng AICPU cross-compile:

- frontend: any ``with pl.*()`` context inside an aicpu kernel is rejected at
  parse time -- ``mode="aicpu_kernel"`` alone selects the execution domain and
  takes no section at all;
- backend (``ir.Pass.validate_target_ops(SectionKind.Aicpu)`` -- the Aicpu
  CoreMask): debug probes (printf / pto_assert / trap / dump_data), block and
  subblock queries, and the system barrier all die with
  "cannot execute on AICPU".
"""

from __future__ import annotations

import re

from pypto_pro._errors import InvalidOperation
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir

CORES_CV = "Supported cores for this call: Cube and Vector."


def _parse(kernel):
    """Parse one aicpu kernel for the Aicpu target and validate its ops."""
    from pypto_pro.runtime.kernel import validate_target_ops_or_raise

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Aicpu)
    validate_target_ops_or_raise(ir.SectionKind.Aicpu, program)
    return program


# ---------------------------------------------------------------------------
# Frontend: any 'with pl.*()' context is rejected (pl.system.nop here)
# ---------------------------------------------------------------------------


def test_with_system_nop_context_is_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        with pl.system.nop():
            pl.setval(x, 0, 1)

    with pytest.raises(InvalidOperation, match="Section 'pl.system' cannot be used inside an aicpu kernel"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Aicpu)


# ---------------------------------------------------------------------------
# Backend: debug probes never reach bisheng (Aicpu CoreMask)
# ---------------------------------------------------------------------------


def test_printf_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.printf("%d", pl.getval(x, 0))

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_pto_assert_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.pto_assert(pl.getval(x, 0) > 0)

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_trap_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.trap()

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_dump_data_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.dump_data(x)

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# Backend: block/subblock queries never reach bisheng (Aicpu CoreMask)
# ---------------------------------------------------------------------------


def test_get_block_idx_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        v = pl.get_block_idx()  # noqa: F841 -- DSL body, parsed as AST
        pl.setval(x, 0, v)

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_get_block_num_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        v = pl.get_block_num()  # noqa: F841 -- DSL body, parsed as AST
        pl.setval(x, 0, v)

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_get_subblock_idx_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        v = pl.get_subblock_idx()  # noqa: F841 -- DSL body, parsed as AST
        pl.setval(x, 0, v)

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_get_subblock_num_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        v = pl.get_subblock_num()  # noqa: F841 -- DSL body, parsed as AST
        pl.setval(x, 0, v)

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# Backend: system barrier never reaches bisheng (Aicpu CoreMask)
# ---------------------------------------------------------------------------


def test_system_bar_all_is_rejected_on_aicpu():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.system.bar_all()

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)
