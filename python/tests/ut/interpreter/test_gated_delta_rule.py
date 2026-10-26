#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal gated_delta_rule verify case (Ascend950 compile + pass_verify + golden, no NPU)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ops.chunked_gated_delta_rule_golden import chunked_gated_delta_rule_golden  # noqa: E402
from _ops.chunked_gated_delta_rule_impl import chunked_gated_delta_rule_wrapper  # noqa: E402
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402

D = 128


def _gen_inputs(params, dtype=torch.float32):
    t, b, nqk, nv = params["T"], params["B"], params["Nqk"], params["Nv"]
    chunk_size = params["chunk_size"]
    query = torch.rand([t, nqk, D], dtype=dtype) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key = torch.rand([t, nqk, D], dtype=dtype) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value = torch.rand([t, nv, D], dtype=dtype) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta = torch.rand([t, nv], dtype=dtype) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate = torch.rand([t, nv], dtype=dtype) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states = torch.zeros([b, nv, D, D], dtype=dtype)
    seq_len_per_batch = t // b
    act_seq_len = torch.tensor([i * seq_len_per_batch for i in range(b + 1)], dtype=torch.int32)
    mask = torch.tril(-torch.ones([chunk_size, chunk_size], dtype=dtype), diagonal=-1)
    tril_mask = torch.ones([chunk_size, chunk_size], dtype=dtype).tril()
    eye = torch.eye(chunk_size, dtype=dtype)
    return {
        "query": query, "key": key, "value": value, "beta": beta, "gate": gate,
        "states": states, "act_seq_len": act_seq_len, "mask": mask,
        "tril_mask": tril_mask, "eye": eye, "chunk_size": chunk_size,
    }


@pytest.mark.soc("950")
def test_gated_delta_rule_minimal():
    """aligned_single_chunk_L32; pass_verify vs chunked_gdr golden."""
    params = {"B": 1, "Nqk": 2, "Nv": 4, "T": 32, "chunk_size": 32}
    torch.manual_seed(42)
    inputs = _gen_inputs(params)
    core_g, state_g = chunked_gated_delta_rule_golden(
        inputs["query"], inputs["key"], inputs["value"],
        inputs["beta"], inputs["gate"], inputs["states"],
        inputs["mask"], inputs["tril_mask"], inputs["eye"],
        inputs["act_seq_len"], chunk_size=inputs["chunk_size"],
    )
    set_verify_goldens([None] * 10 + [core_g.contiguous(), state_g.contiguous()])
    chunked_gated_delta_rule_wrapper(
        inputs["query"], inputs["key"], inputs["value"],
        inputs["beta"], inputs["gate"], inputs["states"],
        act_seq_len=inputs["act_seq_len"], chunk_size=inputs["chunk_size"],
        mask=inputs["mask"], tril_mask=inputs["tril_mask"], eye=inputs["eye"],
    )
    assert_pass_verify_ok()
