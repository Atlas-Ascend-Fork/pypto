#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.hiascend.com/software/licensing/community
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic cell-match pool reset must tolerate aclgraph capture (no direct H2D).

The DynamicCellMatchPool is initialized on the host and copied to the device with an
H2D memcpy whenever the pool is (re)allocated during a launch
(``KernelBinary::RefreshRuntimeDynamicCellMatchMeta`` ->
``ResetRuntimeDynamicCellMatchPoolHost(isDevice=true)``). A plain ``rtMemcpy`` H2D is
forbidden while the stream is capturing, so the copy must go through
``NormalizedRtMemcpy``, which switches to RELAXED capture mode.

This ST forces that H2D to execute INSIDE the capture region: the same JIT kernel is
warmed up eagerly with a small L (pool sized for L=64), then captured with a larger L
(L=2000). The launch inside the capture region re-evaluates the workspace, sees the
cell-match pool must grow, and re-runs the host-side init + H2D copy while capturing.
A regression to a direct ``rtMemcpy`` fails capture; a wrong reset fails the replay
precision check. Multiple replays with per-replay input refresh also cover pool
tag-state reuse across replays.
"""

import os

from numpy.testing import assert_allclose
import torch
import torch_npu

from st.test_cellmatch_case import B_STATIC, D_STATIC, H_STATIC, k_tmp_to_d_emb

WARM_L = 64  # eager warmup: pool is sized for this L, H2D reset runs outside capture
CAP_L = 2000  # captured graph: pool must grow, H2D reset runs inside the capture region
REPLAY_COUNT = 4

RTOL = 1e-3
ATOL = 1e-3


def _golden_d_emb_only(dy: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (dy[:, :, 0, :].reshape(-1, dy.shape[-1]) @ weight[0].T).reshape(dy.shape[0], dy.shape[1], dy.shape[-1])


def test_aclgraph_dynamic_cellmatch_pool_h2d_in_capture():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    # 1. Eager warmup with WARM_L: compile the kernel and size the dynamic cell-match
    #    pool outside of any capture region; also checks the eager baseline.
    torch.manual_seed(44)
    dy = torch.randn(B_STATIC, WARM_L, H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    weight = torch.randn(H_STATIC, D_STATIC, D_STATIC, dtype=torch.float32, device=device)
    d_emb = torch.zeros(B_STATIC, WARM_L, D_STATIC, dtype=torch.float32, device=device)
    k_tmp_to_d_emb(dy, weight, d_emb)
    torch_npu.npu.synchronize()
    assert_allclose(
        d_emb.cpu().numpy(),
        _golden_d_emb_only(dy, weight).cpu().numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="eager warmup mismatch",
    )

    # 2. Buffers for the captured graph; their addresses must stay stable across replays.
    dy_cap = torch.randn(B_STATIC, CAP_L, H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    weight_cap = torch.randn(H_STATIC, D_STATIC, D_STATIC, dtype=torch.float32, device=device)
    d_emb_cap = torch.zeros(B_STATIC, CAP_L, D_STATIC, dtype=torch.float32, device=device)

    # 3. Capture with CAP_L: the in-capture launch changes the required pool size
    #    (WARM_L -> CAP_L), so the pool reset H2D executes while the stream is capturing.
    assert not torch_npu.npu.is_current_stream_capturing()
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        g = torch_npu.npu.NPUGraph()
        torch_npu.npu.empty_cache()
        g.capture_begin()
        k_tmp_to_d_emb(dy_cap, weight_cap, d_emb_cap)
        assert torch_npu.npu.is_current_stream_capturing()
        g.capture_end()
    torch_npu.npu.current_stream().wait_stream(s)

    # 4. Replay with fresh input data each round; golden is recomputed per replay so a
    #    stale or corrupted cell-match pool state fails the check.
    for round_idx in range(REPLAY_COUNT):
        torch.manual_seed(100 + round_idx)
        dy_cap.copy_(torch.randn(B_STATIC, CAP_L, H_STATIC, D_STATIC, device=device))
        weight_cap.copy_(torch.randn(H_STATIC, D_STATIC, D_STATIC, device=device))
        g.replay()
        torch_npu.npu.current_stream().synchronize()
        assert_allclose(
            d_emb_cap.cpu().numpy(),
            _golden_d_emb_only(dy_cap, weight_cap).cpu().numpy(),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"replay round {round_idx} mismatch",
        )
    g.reset()


if __name__ == "__main__":
    test_aclgraph_dynamic_cellmatch_pool_h2d_in_capture()
    print("=========== pass ==========")
