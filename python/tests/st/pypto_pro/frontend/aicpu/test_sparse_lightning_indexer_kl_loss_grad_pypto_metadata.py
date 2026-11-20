# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST for the mixed Tensor/Ptr ``sparse_lightning_indexer_kl_loss_grad_pypto_metadata`` launch.

The kernel (defined inline below) is the JIT fast-launch twin of the
ops-transformer delivery kernel ``attention/sparse_lightning_indexer_kl_loss_grad_pypto_metadata/
op_kernel_aicpu/``, with the optional inputs deliberately split across both tensor
parameter kinds so one kernel exercises the full mixed contract:

* required tensors: ``cu_seqlens_q``/``cu_seqlens_k`` (``pl.Tensor``, TND shape-bearing)
  and the ``metadata`` output; required scalars (no default) bind-must-be-passed.
* optional inputs mix kinds: ``seqused_q`` is a shape-bearing ``pl.Tensor[[pl.DYNAMIC]]
  = None`` (batch-extent via ``pl.numel``), ``seqused_k``/``cmp_residual_k`` are
  ``pl.Ptr[pl.DT_INT32] = None`` (typed-view reads). At the launch boundary both kinds
  pack identically: a live argument becomes its data pointer, None/omitted a null
  pointer -- there is no optional/required distinction between the kinds.
* optional scalars carry Python defaults (batch_size/max_seqlen_q/max_seqlen_k/
  layout_q/layout_k) and may be omitted.

Every presence/emptiness/value judgment is kernel code (the ``is not None`` guards
plus ``ParamsCheck``), mirroring the arch35 cpp; the kernel writes the metadata words
only when all checks pass, so a failed validation leaves the output untouched.

The cases walk the combinations:

* golden cases -- every optional present (both kinds live), every optional
  omitted, each optional alone, BSND attr-driven batch, required tensors passed
  as null (no optional/required distinction at the launch boundary: the
  kernel's guards route), and the zero-batch int32 wraparound contract;
* kernel-side validation failures -- negative seqused_k/cmp_residual_k read
  through the Ptr views, negative seqused_q on the Tensor path, non-ascending
  cu_seqlens_q, and a seqused-derived batch that outgrows the cu extent: the
  check fails, the kernel returns KERNEL_STATUS_PARAM_INVALID, and the AICPU
  runtime surfaces the failed task as a stream error at synchronize time.
  Each such case runs in a subprocess (a failed AICPU task poisons the
  process's device context), and the parent asserts the child sensed the error;
* guards -- a null metadata output is the kernel's own ``is None`` early
  return (now likewise a sensed failure), and a missing required scalar is
  rejected at the launch bind (TypeError).

The goldens mirror the arch35 cpp schedule: with ``aic_core_num=24``,
``former = ceil(total/24)``, ``remain_proc = former - 1``,
``remain_num = former*24 - total``, ``used = min(total, 24)``.

``_main()`` at the bottom runs the cases without pytest.
"""

import os
import subprocess
import sys

import pypto_pro.language as pl
import pytest
import torch

pytestmark = pytest.mark.soc("950")

# ---------------------------------------------------------------------------
# Kernel: arch35 metadata schedule, mixed Tensor/Ptr optional inputs
# ---------------------------------------------------------------------------

# Metadata layout constants (sparse_lightning_indexer_kl_loss_grad_metadata_arch35.h)
TOTAL_NUM = 0
FORMER_CORE_PROCESS_NUM = 1
REMAIN_CORE_PROCESS_NUM = 2
REMAIN_CORE_NUM = 3
USED_CORE_NUM = 4

# Kernel-side comparison constants: enum members have no IR form, so the
# runtime int64 compare uses the folded .value (validated pattern).
LAYOUT_BSND = 0
LAYOUT_TND = 1


def _ceil_div(a, b):
    """``CeilDiv`` -- b == 0 returns 0, matching the C++ template's guard."""
    if b == 0:
        return 0
    return (a + b - 1) // b


def _ptr_view(ptr, n):
    """Typed int32 view over a raw ``pl.Ptr[pl.DT_INT32]`` with ``n`` elements."""
    return pl.make_tensor(ptr, [n], [1], dtype=pl.DT_INT32)


def _get_query_batch_size(seqused_q, cu_q, layout_q, batch_size):
    """``GetQueryBatchSize`` -- seqused dim0, else TND cu dim0 - 1, else the attr.

    Mixed-kind note: the seqused branch reads the shape-bearing ``pl.Tensor``
    optional; a ``pl.Ptr`` carries no extent, so only the Tensor form can own
    the batch here.
    """
    if seqused_q is not None:
        return pl.numel(seqused_q)
    if layout_q == LAYOUT_TND and cu_q is not None:
        return pl.numel(cu_q) - 1
    return batch_size


def _params_check(
    metadata,
    cu_q,
    cu_k,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    layout_q,
    layout_k,
    max_seqlen_q,
    batch_size,
):
    """arch35/92 ``ParamsCheck`` -- value validation over the mixed-kind inputs.

    ``seqused_q`` (Tensor) reads directly with an extent guard; the Ptr optionals
    read through ``[batch_size]`` typed views (a raw pointer carries no extent,
    so no extent guard is expressible for them, exactly like the cpp's raw
    ``GetData()`` pointer).
    """
    if metadata is None:
        # "Output metadata is nullptr" / "Output metadata data is nullptr"
        return False

    if layout_q == LAYOUT_TND and cu_q is not None:
        # (port guard) cu_seqlens_q needs batch + 1 entries
        if pl.numel(cu_q) < batch_size + 1:
            return False
        # "The first element of cu_seqlens_q should be 0, but got %d"
        if pl.getval(cu_q, 0) != 0:
            return False
        for i in pl.range(batch_size + 1):
            # "The elements in cu_seqlens_q must be in ascending order, ..."
            if i > 0 and pl.getval(cu_q, i - 1) > pl.getval(cu_q, i):
                return False
    if layout_k == LAYOUT_TND and cu_k is not None:
        # (port guard) cu_seqlens_k needs batch + 1 entries
        if pl.numel(cu_k) < batch_size + 1:
            return False
        # "The first element of cu_seqlens_k should be 0, but got %d"
        if pl.getval(cu_k, 0) != 0:
            return False
        for i in pl.range(batch_size + 1):
            # "The elements in cu_seqlens_k must be in ascending order, ..."
            if i > 0 and pl.getval(cu_k, i - 1) > pl.getval(cu_k, i):
                return False

    if seqused_q is not None:
        # (port guard) seqused_q needs batch entries
        if pl.numel(seqused_q) < batch_size:
            return False
        for i in pl.range(batch_size):
            # "The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d"
            if pl.getval(seqused_q, i) < 0:
                return False
            # "The elements in seqused_q should not be greater than max_seqlen_q %d, ..."
            if layout_q == LAYOUT_BSND and pl.getval(seqused_q, i) > max_seqlen_q:
                return False
            if layout_q == LAYOUT_TND and cu_q is not None:
                seq_len = pl.getval(cu_q, i + 1) - pl.getval(cu_q, i)
                # "The elements in seqused_q should not be greater than the sequence length ..."
                if pl.getval(seqused_q, i) > seq_len:
                    return False
    if seqused_k is not None:
        view_k = _ptr_view(seqused_k, batch_size)
        for i in pl.range(batch_size):
            # "The elements in seqused_k should be >= 0, but got seqused_k[%d] = %d"
            if pl.getval(view_k, i) < 0:
                return False
    if cmp_residual_k is not None:
        view_c = _ptr_view(cmp_residual_k, batch_size)
        for i in pl.range(batch_size):
            # "The elements in cmp_residual_k should be >= 0, but got cmp_residual_k[%d] = %d"
            if pl.getval(view_c, i) < 0:
                return False
    return True


def _get_s1_seq_size(seqused_q, cu_q, layout_q, b_idx, max_seqlen_q):
    """``GetS1SeqSize`` -- seqused (Tensor read), else TND cu diff, else max_seqlen_q."""
    if seqused_q is not None:
        return pl.astype(pl.getval(seqused_q, b_idx), pl.DT_INT64)
    if layout_q == LAYOUT_TND and cu_q is not None:
        return pl.astype(pl.getval(cu_q, b_idx + 1) - pl.getval(cu_q, b_idx), pl.DT_INT64)
    return max_seqlen_q


def _get_s2_seq_size(seqused_k, cu_k, layout_k, b_idx, b_size, max_seqlen_k):
    """``GetS2SeqSize`` -- the key-side twin; the Ptr optional reads through a view.

    Kept as the cpp API-surface mirror (the metadata schedule consumes only S1).
    """
    if seqused_k is not None:
        return pl.astype(pl.getval(_ptr_view(seqused_k, b_size), b_idx), pl.DT_INT64)
    if layout_k == LAYOUT_TND and cu_k is not None:
        return pl.astype(pl.getval(cu_k, b_idx + 1) - pl.getval(cu_k, b_idx), pl.DT_INT64)
    return max_seqlen_k


def _calc_split_info(batch_size, seqused_q, cu_q, layout_q, max_seqlen_q):
    """arch35 ``CalcSplitInfo`` -- totalNum = sum of per-batch S1 sizes."""
    total_num = 0
    for b_idx in pl.range(batch_size):
        total_num += _get_s1_seq_size(seqused_q, cu_q, layout_q, b_idx, max_seqlen_q)
    return total_num


def _gen_metadata(metadata, total_num, aic_core_num):
    """arch35 ``GenMetadata`` -- the ``SlikgMetadata`` (uint32-view) writeback.

    The published op output contract is ``DT_INT32``: every word is
    non-negative in two's complement except ``remain_core_process_num``
    underflow when ``total_num == 0`` (the cpp's ``uint32_t`` 0xFFFFFFFF
    member shows up as -1).
    """
    former_core_process_num = _ceil_div(total_num, aic_core_num)
    remain_core_process_num = former_core_process_num - 1
    remain_core_num = former_core_process_num * aic_core_num - total_num
    used_core_num = min(total_num, aic_core_num)

    pl.setval(metadata, TOTAL_NUM, total_num)
    pl.setval(metadata, FORMER_CORE_PROCESS_NUM, former_core_process_num)
    pl.setval(metadata, REMAIN_CORE_PROCESS_NUM, remain_core_process_num)
    pl.setval(metadata, REMAIN_CORE_NUM, remain_core_num)
    pl.setval(metadata, USED_CORE_NUM, used_core_num)
    return True


def _compute(
    metadata,
    cu_q,
    cu_k,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    num_heads_q,
    num_heads_k,
    aic_core_num,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    layout_q,
    layout_k,
):
    """arch35/92 ``Compute``: ParamsCheck && BalanceSchedule && GenMetadata."""
    batch_size = _get_query_batch_size(seqused_q, cu_q, layout_q, batch_size)
    if not _params_check(
        metadata,
        cu_q,
        cu_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        layout_q,
        layout_k,
        max_seqlen_q,
        batch_size,
    ):
        return False
    # BalanceSchedule
    total_num = _calc_split_info(batch_size, seqused_q, cu_q, layout_q, max_seqlen_q)
    # GenMetadata
    return _gen_metadata(metadata, total_num, aic_core_num)


@pl.jit(mode="aicpu_kernel")
def sparse_lightning_indexer_kl_loss_grad_pypto_metadata(
    cu_seqlens_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    cu_seqlens_k: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    metadata: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
    num_heads_q: pl.DT_INT64,
    num_heads_k: pl.DT_INT64,
    head_dim: pl.DT_INT64,
    aic_core_num: pl.DT_INT64,
    aiv_core_num: pl.DT_INT64,
    # Optional inputs, mixed kinds: seqused_q is a shape-bearing pl.Tensor
    # (batch extent via pl.numel), seqused_k/cmp_residual_k are raw pl.Ptr
    # (typed-view reads). Both pack identically at the launch boundary --
    # None/omitted becomes a null pointer, the kernel's ``is not None``
    # guards own the presence semantics (cpp OPTIONAL_INPUT).
    seqused_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input] = None,
    seqused_k: pl.Ptr[pl.DT_INT32] = None,
    cmp_residual_k: pl.Ptr[pl.DT_INT32] = None,
    batch_size: pl.DT_INT64 = 0,
    max_seqlen_q: pl.DT_INT64 = 0,
    max_seqlen_k: pl.DT_INT64 = 0,
    layout_q: pl.DT_INT64 = 0,
    layout_k: pl.DT_INT64 = 0,
):
    """``SparseLightningIndexerKLLossGradPyptoMetadataCpuKernel::Compute`` (arch35, mixed port).

    Status reporting mirrors the C++ (bool returns -> early return; the runtime
    maps the void exit to ``KERNEL_STATUS_*``): a failed check leaves the
    metadata output untouched.
    """
    ok = _compute(
        metadata,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        num_heads_q,
        num_heads_k,
        aic_core_num,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
    )
    if not ok:
        # KERNEL_STATUS_PARAM_INVALID: the explicit status return is the only
        # failure channel -- a bare return means KERNEL_STATUS_OK.
        return 1
    # KERNEL_STATUS_OK (implicit bare-return/fall-through)


kernel = sparse_lightning_indexer_kl_loss_grad_pypto_metadata

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _int32(values):
    return torch.tensor(values, dtype=torch.int32, device="npu")


def _zeros_out():
    return torch.zeros(64, dtype=torch.int32, device="npu")


def _metadata_words(metadata):
    torch.npu.synchronize()
    return metadata.cpu().tolist()[:5]


def _cu_pair():
    # batch 2 TND pair: diffs [4, 5] -> totalNum 9 -> [9, 1, 0, 15, 9]
    return _int32([0, 4, 9]), _int32([0, 2, 6])


def _launch(cu_q, cu_k, metadata, **kwargs):
    kernel(
        cu_q, cu_k, metadata, num_heads_q=16, num_heads_k=1, head_dim=128, aic_core_num=24, aiv_core_num=48, **kwargs
    )


# ---------------------------------------------------------------------------
# Golden cases: presence x kind x layout combinations
# ---------------------------------------------------------------------------


def test_full_inputs_tnd_mixed_kinds():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_full_inputs_tnd_mixed_kinds: NPU device not available")
        return

    # All optionals live: seqused_q as a Tensor argument, seqused_k and
    # cmp_residual_k as Ptr arguments -- the boundary packs both identically.
    cu_q, cu_k = _cu_pair()
    metadata = _zeros_out()
    _launch(
        cu_q,
        cu_k,
        metadata,
        seqused_q=_int32([4, 5]),
        seqused_k=_int32([2, 6]),
        cmp_residual_k=_int32([0, 0]),
        layout_q=LAYOUT_TND,
        layout_k=LAYOUT_TND,
    )
    assert _metadata_words(metadata) == [9, 1, 0, 15, 9]


def test_all_optionals_omitted_tnd():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_all_optionals_omitted_tnd: NPU device not available")
        return

    # Optionals omitted entirely (null pointers); TND takes the batch and the
    # per-batch S1 sizes from the cu diffs -- the golden is unchanged.
    cu_q, cu_k = _cu_pair()
    metadata = _zeros_out()
    _launch(cu_q, cu_k, metadata, layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)
    assert _metadata_words(metadata) == [9, 1, 0, 15, 9]


def test_seqused_q_tensor_only_drives_batch():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_seqused_q_tensor_only_drives_batch: NPU device not available")
        return

    # seqused_q alone: its numel (3) owns the batch extent and its values own
    # the S1 sizes (3+4+5 = 12); cu_q must carry batch+1 ascending entries with
    # per-batch diffs >= the seqused values. cu_k stays live but layout_k=BSND
    # skips its TND checks (cu_k has too few entries for the bigger batch).
    cu_q = _int32([0, 4, 9, 14, 20])
    cu_k = _int32([0, 2, 6])
    metadata = _zeros_out()
    _launch(cu_q, cu_k, metadata, seqused_q=_int32([3, 4, 5]), layout_q=LAYOUT_TND, layout_k=LAYOUT_BSND)
    assert _metadata_words(metadata) == [12, 1, 0, 12, 12]


def test_seqused_k_ptr_only_view_reads():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_seqused_k_ptr_only_view_reads: NPU device not available")
        return

    # seqused_k alone (Ptr kind): its values are validated through the typed
    # view but the metadata schedule consumes only S1, so the cu diffs keep the
    # golden unchanged.
    cu_q, cu_k = _cu_pair()
    metadata = _zeros_out()
    _launch(cu_q, cu_k, metadata, seqused_k=_int32([2, 6]), layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)
    assert _metadata_words(metadata) == [9, 1, 0, 15, 9]


def test_bsnd_attr_batch_all_optionals_omitted():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_bsnd_attr_batch_all_optionals_omitted: NPU device not available")
        return

    # BSND: the batch_size and max_seqlen_q attrs own every extent; the cu
    # tensors ride along unchecked. totalNum = 2 * 5 = 10.
    cu_q, cu_k = _cu_pair()
    metadata = _zeros_out()
    _launch(
        cu_q, cu_k, metadata, batch_size=2, max_seqlen_q=5, max_seqlen_k=6, layout_q=LAYOUT_BSND, layout_k=LAYOUT_BSND
    )
    assert _metadata_words(metadata) == [10, 1, 0, 14, 10]


def test_required_tensors_passed_null_bsnd_attr_batch():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_required_tensors_passed_null_bsnd_attr_batch: NPU device not available")
        return

    # The launch boundary draws no optional/required line: even the required
    # cu tensors accept None (null pointers), and the kernel's guards route
    # the batch/S1 authority to the attrs. Same golden as the BSND case.
    metadata = _zeros_out()
    _launch(None, None, metadata, batch_size=2, max_seqlen_q=5, layout_q=LAYOUT_BSND, layout_k=LAYOUT_BSND)
    assert _metadata_words(metadata) == [10, 1, 0, 14, 10]


def test_zero_batch_int32_wraparound_contract():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_zero_batch_int32_wraparound_contract: NPU device not available")
        return

    # batch_size omitted (default 0): totalNum = 0, remain_core_process_num
    # underflows to -1 -- the two's-complement twin of the cpp's uint32
    # 0xFFFFFFFF member on the published DT_INT32 output contract.
    metadata = _zeros_out()
    _launch(_int32([0]), _int32([0]), metadata, layout_q=LAYOUT_BSND, layout_k=LAYOUT_BSND)
    assert _metadata_words(metadata) == [0, 0, -1, 0, 0]


# ---------------------------------------------------------------------------
# Kernel-side validation failures: the kernel returns KERNEL_STATUS_PARAM_INVALID
# and the failed AICPU task surfaces as a stream error at synchronize time.
#
# A failed AICPU task poisons the launching process's device context, so every
# failure case runs in a subprocess: the child performs the invalid launch and
# synchronizes, and exits 0 only when the expected stream error was sensed.
# ---------------------------------------------------------------------------

_CHILD_ENV_VAR = "PYPTO_AICPU_ST_CHILD"


def _child_seqused_k_negative():
    cu_q, cu_k = _cu_pair()
    _launch(cu_q, cu_k, _zeros_out(), seqused_k=_int32([-1, 6]), layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)


def _child_cmp_residual_k_negative():
    cu_q, cu_k = _cu_pair()
    _launch(cu_q, cu_k, _zeros_out(), cmp_residual_k=_int32([-1, 0]), layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)


def _child_seqused_q_negative():
    cu_q, cu_k = _cu_pair()
    _launch(cu_q, cu_k, _zeros_out(), seqused_q=_int32([-1, 5]), layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)


def _child_cu_q_not_ascending():
    _launch(_int32([0, 5, 4]), _int32([0, 2, 6]), _zeros_out(), layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)


def _child_seqused_batch_exceeds_cu():
    # seqused_q (3 entries) derives a batch that cu_q (3 entries) cannot cover
    # (batch + 1 = 4): the (port guard) extent check fails fast instead of
    # letting the cpp's cu loop read past the tensor.
    cu_q, cu_k = _cu_pair()
    _launch(cu_q, cu_k, _zeros_out(), seqused_q=_int32([3, 4, 5]), layout_q=LAYOUT_TND, layout_k=LAYOUT_BSND)


def _child_metadata_null_output():
    # The output itself accepts None: the kernel's ``metadata is None`` guard
    # early-returns, a KERNEL_STATUS_PARAM_INVALID failure under the status ABI.
    cu_q, cu_k = _cu_pair()
    _launch(cu_q, cu_k, None, layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)


_FAILURE_LAUNCHERS = {
    "seqused_k_negative": _child_seqused_k_negative,
    "cmp_residual_k_negative": _child_cmp_residual_k_negative,
    "seqused_q_negative": _child_seqused_q_negative,
    "cu_q_not_ascending": _child_cu_q_not_ascending,
    "seqused_batch_exceeds_cu": _child_seqused_batch_exceeds_cu,
    "metadata_null_output": _child_metadata_null_output,
}


def _assert_kernel_failure_sensed(case_name):
    """Run one invalid-launch case in a subprocess and assert the failure was sensed."""
    result = subprocess.run(
        [sys.executable, __file__],
        env={**os.environ, _CHILD_ENV_VAR: case_name},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"child exit={result.returncode} (expected the stream error to be sensed)\n"
        f"stdout tail:\n{result.stdout[-1500:]}\nstderr tail:\n{result.stderr[-1500:]}"
    )


def test_seqused_k_negative_value_via_ptr_guard():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_seqused_k_negative_value_via_ptr_guard: NPU device not available")
        return

    # The Ptr-form value check reads through the [batch] view and rejects the
    # negative element: ParamsCheck fails and the kernel returns 1.
    _assert_kernel_failure_sensed("seqused_k_negative")


def test_cmp_residual_k_negative_value_via_ptr_guard():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_cmp_residual_k_negative_value_via_ptr_guard: NPU device not available")
        return

    _assert_kernel_failure_sensed("cmp_residual_k_negative")


def test_seqused_q_negative_value_tensor_guard():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_seqused_q_negative_value_tensor_guard: NPU device not available")
        return

    # The Tensor-form value check reads seqused_q directly (no view) and
    # rejects the negative element.
    _assert_kernel_failure_sensed("seqused_q_negative")


def test_cu_q_not_ascending_guard():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_cu_q_not_ascending_guard: NPU device not available")
        return

    # TND cu_seqlens_q must ascend: [0, 5, 4] fails the per-element check.
    _assert_kernel_failure_sensed("cu_q_not_ascending")


def test_seqused_batch_exceeds_cu_extent_guard():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_seqused_batch_exceeds_cu_extent_guard: NPU device not available")
        return

    _assert_kernel_failure_sensed("seqused_batch_exceeds_cu")

    # ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Guards: null output early-return and required-parameter bind
# ---------------------------------------------------------------------------


def test_metadata_null_output_guard():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_metadata_null_output_guard: NPU device not available")
        return

    # The output accepts None and the kernel's ``metadata is None`` guard
    # early-returns: under the status ABI that is a sensed failure in the
    # child. The healthy parent process then proves the compiled variant
    # still writes the ordinary golden on a live output.
    _assert_kernel_failure_sensed("metadata_null_output")

    cu_q, cu_k = _cu_pair()
    metadata = _zeros_out()
    _launch(cu_q, cu_k, metadata, layout_q=LAYOUT_TND, layout_k=LAYOUT_TND)
    assert _metadata_words(metadata) == [9, 1, 0, 15, 9]


def test_missing_required_scalar_rejected():
    torch.npu.set_device(0)
    if not torch.npu.is_available():
        print("[SKIP] test_missing_required_scalar_rejected: NPU device not available")
        return

    # Required scalars carry no Python default: omitting aic_core_num fails
    # the launch bind (TypeError), while the optional scalars stay omittable
    # everywhere else in this file.
    cu_q, cu_k = _cu_pair()
    metadata = _zeros_out()
    try:
        kernel(
            cu_q,
            cu_k,
            metadata,
            num_heads_q=16,
            num_heads_k=1,
            head_dim=128,
            aiv_core_num=48,
            layout_q=LAYOUT_TND,
            layout_k=LAYOUT_TND,
        )
    except TypeError:
        return
    raise AssertionError("omitting the required scalar 'aic_core_num' must fail the launch bind")


def _child_main() -> int:
    case = os.environ[_CHILD_ENV_VAR]
    torch.npu.set_device(0)
    try:
        _FAILURE_LAUNCHERS[case]()
        torch.npu.synchronize()
    except RuntimeError as exc:
        # Expected: the kernel's KERNEL_STATUS_PARAM_INVALID return marks the
        # AICPU task failed; the host observes it as a stream error at sync.
        print(f"child: sensed stream error: {str(exc).splitlines()[0][:120]}")
        return 0
    print(f"child: case {case!r} completed without the expected stream error")
    return 1


if os.environ.get(_CHILD_ENV_VAR):
    raise SystemExit(_child_main())


def _main():
    """Direct entry: run the cases without pytest.

    A failure is reported and does not stop the remaining cases; the process
    exits non-zero if any case failed.
    """
    cases = [
        test_full_inputs_tnd_mixed_kinds,
        test_all_optionals_omitted_tnd,
        test_seqused_q_tensor_only_drives_batch,
        test_seqused_k_ptr_only_view_reads,
        test_bsnd_attr_batch_all_optionals_omitted,
        test_required_tensors_passed_null_bsnd_attr_batch,
        test_zero_batch_int32_wraparound_contract,
        test_seqused_k_negative_value_via_ptr_guard,
        test_cmp_residual_k_negative_value_via_ptr_guard,
        test_seqused_q_negative_value_tensor_guard,
        test_cu_q_not_ascending_guard,
        test_seqused_batch_exceeds_cu_extent_guard,
        test_metadata_null_output_guard,
        test_missing_required_scalar_rejected,
    ]
    failed = []
    for case in cases:
        name = case.__name__
        try:
            case()
            print(f"[PASS] {name}")
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # report and keep running
            failed.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"direct entry: {len(cases) - len(failed)}/{len(cases)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
