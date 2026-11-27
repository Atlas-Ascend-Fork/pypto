# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""litenpu_dynamic_to_static: loop unroll keeps the raw of an assemble dst that is a
function-local accumulator (not a function parameter), so the writes stay visible."""

from pathlib import Path

import pypto

from ..test_common import check_snapshot, npuarch, run_litenpu_pass

IR = Path(__file__).with_suffix(".pypto")


def test_unroll_static_loop_local_accumulator():
    def foo(
        x: pypto.Tensor([32, 16], pypto.DT_FP32),
        out: pypto.Tensor([32, 16], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(16, 16)
        acc = pypto.tensor([32, 16], pypto.DT_FP32, "acc")
        for _ in pypto.loop(4):
            src = pypto.full([16, 16], 1.0, pypto.DT_FP32)
            pypto.assemble(src, [0, 0], acc)
        out[:, :] = pypto.add(acc, x)

    with npuarch("DAV_3113"):
        after = run_litenpu_pass(
            foo,
            pypto.Tensor([32, 16], pypto.DT_FP32, "x"),
            pypto.Tensor([32, 16], pypto.DT_FP32, "out"),
        )
    check_snapshot(after, IR)
