# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""litenpu_dynamic_to_static: loop if/else cond folding snapshot."""

from pathlib import Path

import pypto

from ..test_common import check_snapshot, npuarch, run_litenpu_pass

IR = Path(__file__).with_suffix(".pypto")


def test_fold_loop_cond():
    def foo(
        x: pypto.Tensor([256, 64], pypto.DT_FP32),
        y: pypto.Tensor([256, 64], pypto.DT_FP32),
        out: pypto.Tensor([256, 64], pypto.DT_FP32),
    ):
        for idx in pypto.loop(4):
            start = idx * 64
            end = (idx + 1) * 64
            if pypto.cond(idx % 2 == 0):
                out[start:end, :] = pypto.add(x[start:end, :], y[start:end, :])
            else:
                out[start:end, :] = pypto.sub(x[start:end, :], y[start:end, :])

    with npuarch("DAV_3113"):
        after = run_litenpu_pass(
            foo,
            pypto.Tensor([256, 64], pypto.DT_FP32, "x"),
            pypto.Tensor([256, 64], pypto.DT_FP32, "y"),
            pypto.Tensor([256, 64], pypto.DT_FP32, "out"),
        )
    check_snapshot(after, IR)
