#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Guard case for the loop-idx symbol dedup defect in EvaluateSymbolicScalar.

Minimal-shape driver over the vendored chunked_gated_delta_rule unaligned
kernel (the kernel that exposed the defect): its S loop runs
``pypto.loop(0, s, l, idx_name="s_idx", unroll_list=[16, 1])`` with a dynamic
bound ``s`` read from act_seq_len, so every factor calls
ctx.create_scalar_var("loop_idx_s_idx") and IRContext::GetVarName dedups the
later factors to ``loop_idx_s_idx_0``. The tail-block expressions
``actual_l = (s - s_idx).min(l)`` stored in the view DynValidShape reference
the *base* name created under the first-built factor variant, while the
pass_verify symbolDict of another variant registers only its dedup name.
Replaying that capture after InferParamIndex re-symbolizes valid shapes makes
EvaluateSymbolicScalar fail on the base name:

    ASSERT FAILED: ErrCode: F2FFFF! InternalError::FE_INNER_ERROR,
    func EvaluateSymbolicScalar, file symbolic_scalar_evaluate.cpp
    -> FB4001 VERIFY_RESULT_MISMATCH for every pass >= InferParamIndex

Fixed by the origin-name fallback lookup in symbolic_scalar_evaluate.cpp
(IRContext::GetOriginName + FindDedupNameInDict). Shapes are kept minimal:
B=1, Nqk=1, Nv=1, D=128, L=8, T=12 (act_seq_len=[0, 12]) so the S loop runs
one full chunk (s_idx=0, actual_l=8) plus one tail chunk (s_idx=8,
actual_l=4, is_loop_end branch).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ops"))
from chunked_gated_delta_rule_golden import chunked_gated_delta_rule_golden  # noqa: E402
from chunked_gated_delta_rule_impl import (  # noqa: E402
    chunk_gated_delta_rule_unaligned,
    prepare_chunk_helpers,
)


@pytest.mark.soc("950")
def test_loop_unroll_symbol_dedup():
    """S-loop unroll_list=[16,1] with a tail chunk: pass_verify must not hit FE_INNER_ERROR on the base name."""
    torch.manual_seed(42)
    b, nqk, nv, d, chunk_len, seq_len = 1, 1, 1, 128, 8, 12
    query = torch.rand(seq_len, nqk, d, dtype=torch.float32) * 1.644 - 1.644
    key = torch.rand(seq_len, nqk, d, dtype=torch.float32) * 1.7449 - 1.7449
    value = torch.rand(seq_len, nv, d, dtype=torch.float32) * 1.9273 - 1.9273
    beta = torch.rand(seq_len, nv, dtype=torch.float32) * 0.8038 - 0.8038
    gate = torch.rand(seq_len, nv, dtype=torch.float32) * 37.4109 - 37.4109
    states = torch.zeros(b, nv, d, d, dtype=torch.float32)
    act_seq_len = torch.tensor([0, seq_len], dtype=torch.int32)
    helpers = prepare_chunk_helpers(chunk_len)

    golden_out, golden_state = chunked_gated_delta_rule_golden(
        query, key, value, beta, gate, states,
        helpers["mask"], helpers["tril_mask"], helpers["eye"], act_seq_len, chunk_size=chunk_len,
    )
    set_verify_goldens([None] * 10 + [golden_out, golden_state])

    kernel = chunk_gated_delta_rule_unaligned(b, nqk, nv, d, chunk_len)
    core_attn_out = torch.zeros(seq_len, nv, d, dtype=torch.float32)
    last_state_data = torch.zeros(b, nv, d, d, dtype=torch.float32)
    kernel(query, key, value, beta, gate, states,
           helpers["mask"], helpers["tril_mask"], helpers["eye"], act_seq_len,
           core_attn_out, last_state_data)
    assert_pass_verify_ok()
