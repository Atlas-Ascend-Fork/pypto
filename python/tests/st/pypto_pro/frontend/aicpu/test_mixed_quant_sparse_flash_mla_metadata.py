# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND OF EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST for ``mixed_quant_sparse_flash_mla_metadata`` against a host reference.

The device kernel is the fused-serialization port; the host reference below is
a direct transcription of the original cannbot-dsl sample's array-based
planner (scratch arrays + post-hoc serialize). Full 1024-word equality
therefore validates both the plan math and the fusion refactor.

Cases: uniform plan, exact 72-row selective plan, the cost-greedy FD plan
(split rows, deferred FD tasks), the short-row uniform fallback, the
``support_fd == 2`` non-eligible fallback, and the invalid-extent early
return (the kernel returns KERNEL_STATUS_PARAM_INVALID; the failed AICPU
task surfaces as a stream error, sensed in a subprocess because a failed
task poisons the launching process's device context).
"""

import os
import subprocess
import sys

import mixed_quant_sparse_flash_mla_metadata as mqsmla
import pytest
import torch

pytestmark = pytest.mark.soc("950")

kernel = mqsmla.mixed_quant_sparse_flash_mla_metadata


# ---------------------------------------------------------------------------
# Host reference: plain-Python transcription of the cannbot-dsl sample.
# ---------------------------------------------------------------------------


def _ref_row_tiles(w, wc):
    n = 0
    if w > 0:
        n = (w + 127) // 128
    if wc > 0:
        n = n + (wc + 127) // 128
    return n


def _ref_row_cost(w, wc):
    cost = 0
    if w > 0:
        cost = 44 * (w // 128)
        t = w % 128
        if t != 0:
            cost = cost + 24 + 10 * ((t + 63) // 64)
    if wc > 0:
        costc = 44 * (wc // 128)
        tc = wc % 128
        if tc != 0:
            costc = costc + 24 + 10 * ((tc + 63) // 64)
        cost = cost + costc
    return cost


def _ref_row_last(w, wc):
    if wc > 0:
        t = wc % 128
        if t == 0:
            return 44
        return 24 + 10 * ((t + 63) // 64)
    if w > 0:
        t = w % 128
        if t == 0:
            return 44
        return 24 + 10 * ((t + 63) // 64)
    return 0


def _ref_tile_cost(w, wc, t):
    tw = 0
    if w > 0:
        tw = (w + 127) // 128
    if t < tw:
        wt = w % 128
        if wt != 0 and t == tw - 1:
            return 24 + 10 * ((wt + 63) // 64)
        return 44
    tc = 0
    if wc > 0:
        tc = (wc + 127) // 128
    if tw + tc == 0:
        return 0
    c = t - tw
    ct = wc % 128
    if ct != 0 and c == tc - 1:
        return 24 + 10 * ((ct + 63) // 64)
    return 44


def _ref_uniform(win, cmpv, cu, rows, blocks, batch, has_cmp):
    md = [0] * 1024
    rows_per_core = rows // blocks
    extra = rows % blocks
    start = 0
    batch1 = batch + 1
    for core in range(blocks):
        count = rows_per_core
        if core < extra:
            count += 1
        end = start + count
        if count > 0:
            bn2_s = 0
            for b in range(batch1):
                if cu[b] <= start:
                    bn2_s = b
            m_s = start - cu[bn2_s]
            bn2_e = 0
            for b in range(batch1):
                if cu[b] <= end:
                    bn2_e = b
            m_e = end - cu[bn2_e]
            fa = 9 * core
            md[fa + 0] = 1
            md[fa + 1] = bn2_s
            md[fa + 2] = m_s
            md[fa + 4] = bn2_e
            md[fa + 5] = m_e
        start = end
    return md


def _ref_selective_72(win, cmpv, cu, rows, blocks, batch, has_cmp):
    if has_cmp != 1:
        return _ref_uniform(win, cmpv, cu, rows, blocks, batch, has_cmp)
    for row in range(72):
        if win[row] != 128 or cmpv[row] != 512:
            return _ref_uniform(win, cmpv, cu, rows, blocks, batch, has_cmp)
    md = [0] * 1024
    for core in range(32):
        group = core // 4
        local = core % 4
        base_tile = group * 45
        start_tile = base_tile
        end_tile = base_tile + 11
        first_fd = group * 6
        if local == 1:
            start_tile = base_tile + 11
            end_tile = base_tile + 22
            first_fd += 1
        if local == 2:
            start_tile = base_tile + 22
            end_tile = base_tile + 33
            first_fd += 3
        if local == 3:
            start_tile = base_tile + 33
            end_tile = base_tile + 45
            first_fd += 5
        start = start_tile // 5
        end = end_tile // 5
        bs = 0
        be = 0
        for bi in range(batch + 1):
            if cu[bi] <= start:
                bs = bi
            if cu[bi] <= end:
                be = bi
        base = core * 9
        md[base + 0] = 1
        md[base + 1] = bs
        md[base + 2] = start - cu[bs]
        md[base + 3] = start_tile % 5
        md[base + 4] = be
        md[base + 5] = end - cu[be]
        md[base + 6] = end_tile % 5
        md[base + 7] = first_fd
    for task in range(24):
        group = task // 3
        which = task % 3
        row = group * 9 + 2 + 2 * which
        reducer_core = group * 4 + which
        batch_idx = 0
        for bi in range(batch + 1):
            if cu[bi] <= row:
                batch_idx = bi
        for half in range(2):
            base = 324 + (2 * reducer_core + half) * 8
            md[base + 0] = 1
            md[base + 1] = batch_idx
            md[base + 2] = row - cu[batch_idx]
            md[base + 3] = 2 * task
            md[base + 4] = 2
            md[base + 5] = 32 * half
            md[base + 6] = 32
    md[900] = 48
    md[950] = 1
    return md


def _ref_general(win, cmpv, cu, rows, blocks, batch, has_cmp):
    md = [0] * 1024
    fa_m_start = [0] * 36
    fa_s2_start = [0] * 36
    fa_m_end = [0] * 36
    fa_s2_end = [0] * 36
    fa_first_fd = [0] * 36
    fd_row = [0] * 36
    fd_ws = [0] * 36
    fd_num = [0] * 36

    total_cost = 0
    total_tiles = 0
    max_row_tiles = 0
    r = 0
    while r < rows:
        w = win[r]
        wc = cmpv[r] if has_cmp == 1 else 0
        total_cost += _ref_row_cost(w, wc)
        rt = _ref_row_tiles(w, wc)
        total_tiles += rt
        if rt > max_row_tiles:
            max_row_tiles = rt
        r += 1

    if max_row_tiles < 5:
        return _ref_uniform(win, cmpv, cu, rows, blocks, batch, has_cmp)

    plan_blocks = total_tiles // 4
    if plan_blocks < rows:
        plan_blocks = rows
    if plan_blocks > blocks:
        plan_blocks = blocks
    ceil_rows_blocks = (rows + blocks - 1) // blocks
    fd_floor = (total_tiles + plan_blocks - 1) // plan_blocks
    uniform_crit = ceil_rows_blocks * max_row_tiles
    if uniform_crit - fd_floor < 4:
        return _ref_uniform(win, cmpv, cu, rows, blocks, batch, has_cmp)

    cur_row = 0
    cur_tile = 0
    finished = 0
    wv = win[0]
    wcv = cmpv[0] if has_cmp == 1 else 0
    tiles_c = _ref_row_tiles(wv, wcv)
    rem_cost = _ref_row_cost(wv, wcv)
    last_c = _ref_row_last(wv, wcv)
    unassigned = total_cost
    kv_split = 1
    pre_fd = 0
    num_fd = 0
    used = 0
    core = 0
    while core < plan_blocks:
        if finished == 1:
            break
        fa_first_fd[core] = pre_fd + kv_split - 1
        fa_m_start[core] = cur_row
        fa_s2_start[core] = cur_tile
        limit = unassigned // (plan_blocks - core)
        cost = 0
        blk = 0
        while finished == 0:
            if tiles_c == 0 or limit + last_c // 2 >= cost + rem_cost or (cur_tile == 0 and tiles_c < 5 and blk == 0):
                cost += rem_cost
                blk += tiles_c - cur_tile
                cur_row += 1
                cur_tile = 0
                if cur_row >= rows:
                    finished = 1
                else:
                    wv = win[cur_row]
                    wcv = cmpv[cur_row] if has_cmp == 1 else 0
                    tiles_c = _ref_row_tiles(wv, wcv)
                    rem_cost = _ref_row_cost(wv, wcv)
                    last_c = _ref_row_last(wv, wcv)
            else:
                break
        while finished == 0:
            if cur_tile == 0 and tiles_c < 5:
                break
            tcost = _ref_tile_cost(wv, wcv, cur_tile)
            if limit + tcost // 2 >= cost + tcost:
                cost += tcost
                blk += 1
                rem_cost -= tcost
                cur_tile += 1
                if cur_tile >= tiles_c:
                    cur_row += 1
                    cur_tile = 0
                    if cur_row >= rows:
                        finished = 1
                    else:
                        wv = win[cur_row]
                        wcv = cmpv[cur_row] if has_cmp == 1 else 0
                        tiles_c = _ref_row_tiles(wv, wcv)
                        rem_cost = _ref_row_cost(wv, wcv)
                        last_c = _ref_row_last(wv, wcv)
            else:
                break
        if blk == 0 and finished == 0:
            tcost = _ref_tile_cost(wv, wcv, cur_tile)
            cost += tcost
            blk = 1
            rem_cost -= tcost
            cur_tile += 1
            if cur_tile >= tiles_c:
                cur_row += 1
                cur_tile = 0
                if cur_row >= rows:
                    finished = 1
                else:
                    wv = win[cur_row]
                    wcv = cmpv[cur_row] if has_cmp == 1 else 0
                    tiles_c = _ref_row_tiles(wv, wcv)
                    rem_cost = _ref_row_cost(wv, wcv)
                    last_c = _ref_row_last(wv, wcv)
        fa_m_end[core] = cur_row
        fa_s2_end[core] = cur_tile
        unassigned -= cost
        if core > 0:
            if kv_split > 1:
                if cur_row != fa_m_end[core - 1]:
                    fd_row[num_fd] = fa_m_end[core - 1]
                    fd_ws[num_fd] = pre_fd
                    fd_num[num_fd] = kv_split
                    num_fd += 1
                    pre_fd += kv_split
                    kv_split = 1
        if cur_tile > 0:
            kv_split += 1
        used = core + 1
        core += 1

    i = 0
    while i < blocks:
        base = i * 9
        if i < used:
            md[base + 0] = 1
            batch_start = 0
            batch_end = 0
            for bi in range(batch + 1):
                if cu[bi] <= fa_m_start[i]:
                    batch_start = bi
                if cu[bi] <= fa_m_end[i]:
                    batch_end = bi
            md[base + 1] = batch_start
            md[base + 4] = batch_end
            md[base + 2] = fa_m_start[i] - cu[batch_start]
            md[base + 3] = fa_s2_start[i]
            md[base + 5] = fa_m_end[i] - cu[batch_end]
            md[base + 6] = fa_s2_end[i]
            md[base + 7] = fa_first_fd[i]
        i += 1
    j = 0
    while j < 2 * num_fd:
        base = 324 + j * 8
        t = j // 2
        md[base + 0] = 1
        fd_batch_index = 0
        for bi in range(batch + 1):
            if cu[bi] <= fd_row[t]:
                fd_batch_index = bi
        md[base + 1] = fd_batch_index
        md[base + 2] = fd_row[t] - cu[fd_batch_index]
        md[base + 3] = fd_ws[t]
        md[base + 4] = fd_num[t]
        md[base + 5] = (j % 2) * 32
        md[base + 6] = 32
        j += 1
    md[900] = 2 * num_fd
    return md


def _ref_metadata(win, cmpv, cu, rows, blocks, batch, has_cmp, support_fd):
    """Host reference dispatch; returns None for the invalid-extent early return."""
    if rows <= 0 or blocks <= 0 or blocks > 36:
        return None
    if support_fd >= 2:
        selective = 0
        if rows == 72 and blocks == 32 and has_cmp == 1:
            selective = 1
            for row in range(72):
                if win[row] != 128 or cmpv[row] != 512:
                    selective = 0
        if selective == 1:
            return _ref_selective_72(win, cmpv, cu, rows, blocks, batch, has_cmp)
        if support_fd == 2:
            return _ref_uniform(win, cmpv, cu, rows, blocks, batch, has_cmp)
    if support_fd == 0:
        return _ref_uniform(win, cmpv, cu, rows, blocks, batch, has_cmp)
    return _ref_general(win, cmpv, cu, rows, blocks, batch, has_cmp)


# ---------------------------------------------------------------------------
# Device launch helpers
# ---------------------------------------------------------------------------


def _launch(win, cmpv, cu, rows, blocks, batch, has_cmp, support_fd, fill=0):
    win_t = torch.tensor(win, dtype=torch.int32, device="npu")
    cmp_t = torch.tensor(cmpv, dtype=torch.int32, device="npu")
    cu_t = torch.tensor(cu, dtype=torch.int32, device="npu")
    metadata = torch.full((1024,), fill, dtype=torch.int32, device="npu")
    kernel(win_t, cmp_t, metadata, cu_t, rows=rows, blocks=blocks, batch=batch, has_cmp=has_cmp, support_fd=support_fd)
    torch.npu.synchronize()
    return metadata.cpu().tolist()


def _first_diff(got, expected):
    for i, (g, e) in enumerate(zip(got, expected)):
        if g != e:
            return f"word {i}: got {g}, expected {e}"
    return "no diff"


def test_uniform_plan_matches_host_reference():
    torch.npu.set_device(1)
    if not torch.npu.is_available():
        print("[SKIP] test_uniform_plan_matches_host_reference: NPU device not available")
        return
    win = [64] * 9
    cmpv = [0] * 9
    cu = [0, 4, 9]
    expected = _ref_metadata(win, cmpv, cu, rows=9, blocks=4, batch=2, has_cmp=0, support_fd=0)
    got = _launch(win, cmpv, cu, rows=9, blocks=4, batch=2, has_cmp=0, support_fd=0)
    assert got == expected, _first_diff(got, expected)


def test_selective_fd_72_plan_matches_host_reference():
    torch.npu.set_device(1)
    if not torch.npu.is_available():
        print("[SKIP] test_selective_fd_72_plan_matches_host_reference: NPU device not available")
        return
    win = [128] * 72
    cmpv = [512] * 72
    cu = [0, 36, 72]
    expected = _ref_metadata(win, cmpv, cu, rows=72, blocks=32, batch=2, has_cmp=1, support_fd=3)
    got = _launch(win, cmpv, cu, rows=72, blocks=32, batch=2, has_cmp=1, support_fd=3)
    assert got == expected, _first_diff(got, expected)


def test_general_fd_plan_matches_host_reference():
    torch.npu.set_device(1)
    if not torch.npu.is_available():
        print("[SKIP] test_general_fd_plan_matches_host_reference: NPU device not available")
        return
    # One 1280-token row (10 tiles) plus three 64-token rows: the savings
    # gates pass and the greedy planner must split the long row, producing
    # deferred FD tasks and a nonzero FD_USED_VEC_NUM word.
    win = [1280, 64, 64, 64]
    cmpv = [0, 0, 0, 0]
    cu = [0, 4]
    expected = _ref_metadata(win, cmpv, cu, rows=4, blocks=4, batch=1, has_cmp=0, support_fd=1)
    assert expected[900] > 0, "case must exercise the FD planner, not the uniform fallback"
    got = _launch(win, cmpv, cu, rows=4, blocks=4, batch=1, has_cmp=0, support_fd=1)
    assert got == expected, _first_diff(got, expected)


def test_short_rows_fall_back_to_uniform():
    torch.npu.set_device(1)
    if not torch.npu.is_available():
        print("[SKIP] test_short_rows_fall_back_to_uniform: NPU device not available")
        return
    win = [64, 64, 64, 64]
    cmpv = [0, 0, 0, 0]
    cu = [0, 4]
    expected = _ref_metadata(win, cmpv, cu, rows=4, blocks=4, batch=1, has_cmp=0, support_fd=1)
    got = _launch(win, cmpv, cu, rows=4, blocks=4, batch=1, has_cmp=0, support_fd=1)
    assert got == expected, _first_diff(got, expected)


def test_support_fd_2_non_eligible_falls_back_to_uniform():
    torch.npu.set_device(1)
    if not torch.npu.is_available():
        print("[SKIP] test_support_fd_2_non_eligible_falls_back_to_uniform: NPU device not available")
        return
    win = [64] * 9
    cmpv = [0] * 9
    cu = [0, 4, 9]
    expected = _ref_metadata(win, cmpv, cu, rows=9, blocks=4, batch=2, has_cmp=0, support_fd=2)
    got = _launch(win, cmpv, cu, rows=9, blocks=4, batch=2, has_cmp=0, support_fd=2)
    assert got == expected, _first_diff(got, expected)


_CHILD_ENV_VAR = "PYPTO_AICPU_ST_CHILD"


def _child_invalid_blocks():
    # blocks=40 exceeds AIC_CORE_MAX_NUM (36): the kernel's extent check
    # early-returns, a KERNEL_STATUS_PARAM_INVALID failure under the status ABI.
    _launch([64] * 4, [0] * 4, [0, 4], rows=4, blocks=40, batch=1, has_cmp=0, support_fd=1, fill=0x5A5A)


_FAILURE_LAUNCHERS = {
    "invalid_blocks": _child_invalid_blocks,
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


def test_invalid_blocks_senses_kernel_failure():
    torch.npu.set_device(1)
    if not torch.npu.is_available():
        print("[SKIP] test_invalid_blocks_senses_kernel_failure: NPU device not available")
        return

    # blocks=40 exceeds the core limit: the kernel's dispatch check fails and
    # returns KERNEL_STATUS_PARAM_INVALID, which the child senses as a stream
    # error at synchronize time.
    _assert_kernel_failure_sensed("invalid_blocks")


def _child_main() -> int:
    case = os.environ[_CHILD_ENV_VAR]
    torch.npu.set_device(1)
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
    """Direct entry: run the cases without pytest."""
    cases = [
        test_uniform_plan_matches_host_reference,
        test_selective_fd_72_plan_matches_host_reference,
        test_general_fd_plan_matches_host_reference,
        test_short_rows_fall_back_to_uniform,
        test_support_fd_2_non_eligible_falls_back_to_uniform,
        test_invalid_blocks_senses_kernel_failure,
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
