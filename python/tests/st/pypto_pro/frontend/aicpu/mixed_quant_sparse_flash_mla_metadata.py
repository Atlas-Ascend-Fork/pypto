# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND OF EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MixedQuantSparseFlashMlaMetadata -- PyPTO Pro AICPU port of the cannbot-dsl sample.

Mirrors the computation flow of the cannbot-dsl AICPU planner
``samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_metadata.py``:
the fixed 1024-word int32 FA/FD metadata consumed by the MLA attention kernel,
with three plans dispatched by ``support_fd``:

* uniform whole-row partition (``support_fd == 0``, or the planner fallbacks);
* the exact 72-row selective plan (``support_fd >= 2`` with 72 rows of
  win_len=128 / cmp_len=512 on 32 cores);
* the cost-greedy FD planner: pass 1 row cost/tile totals, the savings gates,
  then pass 2 greedy row/tile assignment with a half-tail-tile tolerance and
  deferred FD tasks (rows with >= 5 tiles may split across cores).

Cost model (identical to the sample): full 128-token tile = 44, a partial tile
= 24 + 10 * ceil(rem/64).

Port divergence: the DSL sample stages per-core FA snapshots and FD tasks in
``zeros(I64, 36)`` scratch arrays and serializes them after pass 2. This port
has no local scratch arrays, so each core serializes its own FA entry and its
closed FD tasks directly inside the pass 2 loop (the array slots become scalar
snapshots, ``fa_m_end[core - 1]`` becomes ``prev_end``). The emitted metadata
words are identical.
"""

import pypto_pro.language as pl

AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
MQSMLA_METADATA_TOTAL_SIZE = 1024

FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
FD_METADATA_BASE = AIC_CORE_MAX_NUM * FA_METADATA_SIZE  # 324
FD_USED_VEC_NUM_WORD = FD_METADATA_BASE + AIV_CORE_MAX_NUM * FD_METADATA_SIZE  # 900

FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7

FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6

FA_TOLERANCE_RATIO = 2

FD_MIN_SAVED_TILES = 4
FD_TARGET_TILES_PER_CORE = 4
FD_MIN_ROW_TILES = 5

_COST_FULL_TILE = 44
_COST_M_PART = 24


def _cu(cu_q, i):
    # Int64 view of one cu_q entry; keeps int32 reads out of int64 plan math.
    return pl.astype(pl.getval(cu_q, i), pl.DT_INT64)


def _row_len(tensor, i):
    # Int64 view of one win_len/cmp_len entry.
    return pl.astype(pl.getval(tensor, i), pl.DT_INT64)


def _row_tiles(w, wc):
    n = 0
    if w > 0:
        n = (w + 127) // 128
    if wc > 0:
        n = n + (wc + 127) // 128
    return n


def _row_cost(w, wc):
    cost = 0
    if w > 0:
        cost = _COST_FULL_TILE * (w // 128)
        t = w % 128
        if t != 0:
            cost = cost + _COST_M_PART + 10 * ((t + 63) // 64)
    if wc > 0:
        costc = _COST_FULL_TILE * (wc // 128)
        tc = wc % 128
        if tc != 0:
            costc = costc + _COST_M_PART + 10 * ((tc + 63) // 64)
        cost = cost + costc
    return cost


def _row_last(w, wc):
    # Cost of the final tile; CMP follows ORI when present.
    if wc > 0:
        t = wc % 128
        if t == 0:
            return _COST_FULL_TILE
        return _COST_M_PART + 10 * ((t + 63) // 64)
    if w > 0:
        t = w % 128
        if t == 0:
            return _COST_FULL_TILE
        return _COST_M_PART + 10 * ((t + 63) // 64)
    return 0


def _tile_cost(w, wc, t):
    # Cost of row-local tile t; zero-cost rows let the tile loop advance.
    tw = 0
    if w > 0:
        tw = (w + 127) // 128
    if t < tw:
        wt = w % 128
        if wt != 0 and t == tw - 1:
            return _COST_M_PART + 10 * ((wt + 63) // 64)
        return _COST_FULL_TILE
    tc = 0
    if wc > 0:
        tc = (wc + 127) // 128
    if tw + tc == 0:
        return 0
    c = t - tw
    ct = wc % 128
    if ct != 0 and c == tc - 1:
        return _COST_M_PART + 10 * ((ct + 63) // 64)
    return _COST_FULL_TILE


def _batch_of(cu_q, batch, row):
    # Largest batch index whose cu_q entry is <= row.
    found = 0
    for b in pl.range(0, batch + 1):
        if _cu(cu_q, b) <= row:
            found = b
    return found


def _zero_metadata(metadata, size):
    for i in pl.range(0, size):
        pl.setval(metadata, i, 0)


def _uniform_metadata(metadata, cu_q, rows, blocks, batch):
    # Whole-row partition: every metadata field consumed by attention is
    # initialized, unused FA/FD entries included.
    _zero_metadata(metadata, MQSMLA_METADATA_TOTAL_SIZE)
    rows_per_core = rows // blocks
    extra = rows % blocks
    start = 0
    for core in pl.range(0, blocks):
        count = rows_per_core
        if core < extra:
            count = count + 1
        end = start + count
        if count > 0:
            bn2_s = _batch_of(cu_q, batch, start)
            m_s = start - _cu(cu_q, bn2_s)
            bn2_e = _batch_of(cu_q, batch, end)
            m_e = end - _cu(cu_q, bn2_e)
            fa = FA_METADATA_SIZE * core
            pl.setval(metadata, fa + FA_CORE_ENABLE_INDEX, 1)
            pl.setval(metadata, fa + FA_BN2_START_INDEX, bn2_s)
            pl.setval(metadata, fa + FA_M_START_INDEX, m_s)
            pl.setval(metadata, fa + FA_BN2_END_INDEX, bn2_e)
            pl.setval(metadata, fa + FA_M_END_INDEX, m_e)
        start = end
    return True


def _selective_fd_72(metadata, cu_q, rows, blocks, batch, has_cmp, win_len, cmp_len):
    # Exact 72 x 5-tile contiguous plan: eight 9-row groups on four cores
    # (11/11/11/12 tiles), three local split rows per group. Falls back to
    # the uniform plan when the shape is not the exact 72-row case.
    if has_cmp != 1:
        return _uniform_metadata(metadata, cu_q, rows, blocks, batch)
    for row in pl.range(0, 72):
        if _row_len(win_len, row) != 128 or _row_len(cmp_len, row) != 512:
            return _uniform_metadata(metadata, cu_q, rows, blocks, batch)
    _zero_metadata(metadata, FD_USED_VEC_NUM_WORD + 51)
    for core in pl.range(0, 32):
        group = core // 4
        local = core % 4
        base_tile = group * 45
        start_tile = base_tile
        end_tile = base_tile + 11
        first_fd = group * 6
        if local == 1:
            start_tile = base_tile + 11
            end_tile = base_tile + 22
            first_fd = first_fd + 1
        if local == 2:
            start_tile = base_tile + 22
            end_tile = base_tile + 33
            first_fd = first_fd + 3
        if local == 3:
            start_tile = base_tile + 33
            end_tile = base_tile + 45
            first_fd = first_fd + 5
        start = start_tile // 5
        end = end_tile // 5
        bs = _batch_of(cu_q, batch, start)
        be = _batch_of(cu_q, batch, end)
        base = core * FA_METADATA_SIZE
        pl.setval(metadata, base + FA_CORE_ENABLE_INDEX, 1)
        pl.setval(metadata, base + FA_BN2_START_INDEX, bs)
        pl.setval(metadata, base + FA_M_START_INDEX, start - _cu(cu_q, bs))
        pl.setval(metadata, base + FA_S2_START_INDEX, start_tile % 5)
        pl.setval(metadata, base + FA_BN2_END_INDEX, be)
        pl.setval(metadata, base + FA_M_END_INDEX, end - _cu(cu_q, be))
        pl.setval(metadata, base + FA_S2_END_INDEX, end_tile % 5)
        pl.setval(metadata, base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, first_fd)
    for task in pl.range(0, 24):
        group = task // 3
        which = task % 3
        row = group * 9 + 2 + 2 * which
        reducer_core = group * 4 + which
        batch_idx = _batch_of(cu_q, batch, row)
        for half in pl.range(0, 2):
            base = FD_METADATA_BASE + (2 * reducer_core + half) * FD_METADATA_SIZE
            pl.setval(metadata, base + FD_CORE_ENABLE_INDEX, 1)
            pl.setval(metadata, base + FD_BN2_IDX_INDEX, batch_idx)
            pl.setval(metadata, base + FD_M_IDX_INDEX, row - _cu(cu_q, batch_idx))
            pl.setval(metadata, base + FD_WORKSPACE_IDX_INDEX, 2 * task)
            pl.setval(metadata, base + FD_WORKSPACE_NUM_INDEX, 2)
            pl.setval(metadata, base + FD_M_START_INDEX, 32 * half)
            pl.setval(metadata, base + FD_M_NUM_INDEX, 32)
    pl.setval(metadata, FD_USED_VEC_NUM_WORD, 48)
    pl.setval(metadata, FD_USED_VEC_NUM_WORD + 50, 1)
    return True


def _general_fd_plan(metadata, cu_q, win_len, cmp_len, rows, blocks, batch, has_cmp):
    # Cost-greedy planner: pass 1 totals, savings gates, pass 2 greedy
    # assignment. FA/FD serialization is fused into pass 2 (see the module
    # docstring), so the sample's scratch arrays become scalar snapshots.
    _zero_metadata(metadata, MQSMLA_METADATA_TOTAL_SIZE)

    # Pass 1: total cost, tile count and maximum tiles per row.
    total_cost = 0
    total_tiles = 0
    max_row_tiles = 0
    for r in pl.range(0, rows):
        w = _row_len(win_len, r)
        wc = 0
        if has_cmp == 1:
            wc = _row_len(cmp_len, r)
        total_cost = total_cost + _row_cost(w, wc)
        rt = _row_tiles(w, wc)
        total_tiles = total_tiles + rt
        if rt > max_row_tiles:
            max_row_tiles = rt

    if max_row_tiles < FD_MIN_ROW_TILES:
        return _uniform_metadata(metadata, cu_q, rows, blocks, batch)

    # Limit split fanout while retaining at least whole-row parallelism.
    plan_blocks = total_tiles // FD_TARGET_TILES_PER_CORE
    if plan_blocks < rows:
        plan_blocks = rows
    if plan_blocks > blocks:
        plan_blocks = blocks
    # Require enough modeled tile savings to amortize staging and reduction.
    ceil_rows_blocks = (rows + blocks - 1) // blocks
    fd_floor = (total_tiles + plan_blocks - 1) // plan_blocks
    uniform_crit = ceil_rows_blocks * max_row_tiles
    if uniform_crit - fd_floor < FD_MIN_SAVED_TILES:
        return _uniform_metadata(metadata, cu_q, rows, blocks, batch)

    # Pass 2: greedy core assignment, following CalcSplitPlan/AssignBlocksToCore.
    cur_row = 0
    cur_tile = 0
    finished = 0
    wv = _row_len(win_len, 0)
    wcv = 0
    if has_cmp == 1:
        wcv = _row_len(cmp_len, 0)
    tiles_c = _row_tiles(wv, wcv)
    rem_cost = _row_cost(wv, wcv)
    last_c = _row_last(wv, wcv)
    unassigned = total_cost
    kv_split = 1
    pre_fd = 0
    num_fd = 0
    prev_end = 0
    for core in pl.range(0, plan_blocks):
        if finished == 1:
            break
        core_first_fd = pre_fd + kv_split - 1
        core_m_start = cur_row
        core_s2_start = cur_tile
        limit = unassigned // (plan_blocks - core)
        cost = 0
        blk = 0
        # Assign whole rows, advancing directly past zero-cost empty rows.
        while finished == 0:
            if (
                tiles_c == 0
                or limit + last_c // FA_TOLERANCE_RATIO >= cost + rem_cost
                or (cur_tile == 0 and tiles_c < FD_MIN_ROW_TILES and blk == 0)
            ):
                cost = cost + rem_cost
                blk = blk + (tiles_c - cur_tile)
                cur_row = cur_row + 1
                cur_tile = 0
                if cur_row >= rows:
                    finished = 1
                else:
                    wv = _row_len(win_len, cur_row)
                    wcv = 0
                    if has_cmp == 1:
                        wcv = _row_len(cmp_len, cur_row)
                    tiles_c = _row_tiles(wv, wcv)
                    rem_cost = _row_cost(wv, wcv)
                    last_c = _row_last(wv, wcv)
            else:
                break
        # Half-tile tolerance keeps a split boundary inside a long row.
        while finished == 0:
            if cur_tile == 0 and tiles_c < FD_MIN_ROW_TILES:
                break
            tcost = _tile_cost(wv, wcv, cur_tile)
            if limit + tcost // FA_TOLERANCE_RATIO >= cost + tcost:
                cost = cost + tcost
                blk = blk + 1
                rem_cost = rem_cost - tcost
                cur_tile = cur_tile + 1
                if cur_tile >= tiles_c:
                    cur_row = cur_row + 1
                    cur_tile = 0
                    if cur_row >= rows:
                        finished = 1
                    else:
                        wv = _row_len(win_len, cur_row)
                        wcv = 0
                        if has_cmp == 1:
                            wcv = _row_len(cmp_len, cur_row)
                        tiles_c = _row_tiles(wv, wcv)
                        rem_cost = _row_cost(wv, wcv)
                        last_c = _row_last(wv, wcv)
            else:
                break
        # Force one tile to guarantee forward progress.
        if blk == 0 and finished == 0:
            tcost = _tile_cost(wv, wcv, cur_tile)
            cost = cost + tcost
            blk = 1
            rem_cost = rem_cost - tcost
            cur_tile = cur_tile + 1
            if cur_tile >= tiles_c:
                cur_row = cur_row + 1
                cur_tile = 0
                if cur_row >= rows:
                    finished = 1
                else:
                    wv = _row_len(win_len, cur_row)
                    wcv = 0
                    if has_cmp == 1:
                        wcv = _row_len(cmp_len, cur_row)
                    tiles_c = _row_tiles(wv, wcv)
                    rem_cost = _row_cost(wv, wcv)
                    last_c = _row_last(wv, wcv)
        core_m_end = cur_row
        core_s2_end = cur_tile
        unassigned = unassigned - cost
        # A split row that closes on this core becomes a deferred FD task.
        if core > 0:
            if kv_split > 1:
                if core_m_end != prev_end:
                    fd_batch_idx = _batch_of(cu_q, batch, prev_end)
                    for half in pl.range(0, 2):
                        base = FD_METADATA_BASE + (2 * num_fd + half) * FD_METADATA_SIZE
                        pl.setval(metadata, base + FD_CORE_ENABLE_INDEX, 1)
                        pl.setval(metadata, base + FD_BN2_IDX_INDEX, fd_batch_idx)
                        pl.setval(metadata, base + FD_M_IDX_INDEX, prev_end - _cu(cu_q, fd_batch_idx))
                        pl.setval(metadata, base + FD_WORKSPACE_IDX_INDEX, pre_fd)
                        pl.setval(metadata, base + FD_WORKSPACE_NUM_INDEX, kv_split)
                        pl.setval(metadata, base + FD_M_START_INDEX, 32 * half)
                        pl.setval(metadata, base + FD_M_NUM_INDEX, 32)
                    num_fd = num_fd + 1
                    pre_fd = pre_fd + kv_split
                    kv_split = 1
        # An unfinished row at this core boundary adds one pending partial.
        if cur_tile > 0:
            kv_split = kv_split + 1
        # Fused FA serialization for this core.
        base = core * FA_METADATA_SIZE
        pl.setval(metadata, base + FA_CORE_ENABLE_INDEX, 1)
        batch_start = _batch_of(cu_q, batch, core_m_start)
        batch_end = _batch_of(cu_q, batch, core_m_end)
        pl.setval(metadata, base + FA_BN2_START_INDEX, batch_start)
        pl.setval(metadata, base + FA_BN2_END_INDEX, batch_end)
        pl.setval(metadata, base + FA_M_START_INDEX, core_m_start - _cu(cu_q, batch_start))
        pl.setval(metadata, base + FA_S2_START_INDEX, core_s2_start)
        pl.setval(metadata, base + FA_M_END_INDEX, core_m_end - _cu(cu_q, batch_end))
        pl.setval(metadata, base + FA_S2_END_INDEX, core_s2_end)
        pl.setval(metadata, base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, core_first_fd)
        prev_end = core_m_end
    # Publish the uniform reducer count; zero skips the FD barrier.
    pl.setval(metadata, FD_USED_VEC_NUM_WORD, 2 * num_fd)
    return True


@pl.jit(mode="aicpu_kernel")
def mixed_quant_sparse_flash_mla_metadata(
    win_len: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    cmp_len: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    metadata: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
    cu_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    rows: pl.DT_INT64,
    blocks: pl.DT_INT64,
    batch: pl.DT_INT64,
    has_cmp: pl.DT_INT64,
    support_fd: pl.DT_INT64,
):
    """``_mqsmla_metadata_kernel`` (cannbot-dsl) as a PyPTO Pro AICPU kernel.

    Dispatch mirrors the sample: invalid extents return KERNEL_STATUS_PARAM_INVALID
    (the early `return 1`); ``support_fd >= 2`` tries the exact 72-row selective
    plan first (``support_fd == 2`` falls back to uniform), ``support_fd == 0``
    is the uniform plan, everything else runs the cost-greedy FD planner. Every
    completed branch ends with a bare `return` -- KERNEL_STATUS_OK.
    """
    if rows <= 0 or blocks <= 0 or blocks > AIC_CORE_MAX_NUM:
        return 1
    if support_fd >= 2:
        selective = 0
        if rows == 72 and blocks == 32 and has_cmp == 1:
            selective = 1
            for row in pl.range(0, 72):
                if _row_len(win_len, row) != 128 or _row_len(cmp_len, row) != 512:
                    selective = 0
        if selective == 1:
            _selective_fd_72(metadata, cu_q, rows, blocks, batch, has_cmp, win_len, cmp_len)
            return
        if support_fd == 2:
            _uniform_metadata(metadata, cu_q, rows, blocks, batch)
            return
    if support_fd == 0:
        _uniform_metadata(metadata, cu_q, rows, blocks, batch)
        return
    _general_fd_plan(metadata, cu_q, win_len, cmp_len, rows, blocks, batch, has_cmp)
