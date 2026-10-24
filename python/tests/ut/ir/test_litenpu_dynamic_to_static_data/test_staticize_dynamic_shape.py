# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""litenpu_dynamic_to_static: dynamic shape (-1) staticization snapshot."""

from pathlib import Path

import pypto

from ..test_common import check_snapshot, npuarch, run_litenpu_pass

IR = Path(__file__).with_suffix(".pypto")


def test_staticize_dynamic_shape():
    def foo(
        x: pypto.Tensor([-1], pypto.DT_FP32),
        y: pypto.Tensor([-1], pypto.DT_FP32),
        out: pypto.Tensor([-1], pypto.DT_FP32),
    ):
        out[:] = pypto.add(x, y)

    with npuarch("DAV_3113"):
        after = run_litenpu_pass(
            foo,
            pypto.Tensor([-1], pypto.DT_FP32, "x", ori_shape=(112,)),
            pypto.Tensor([-1], pypto.DT_FP32, "y", ori_shape=(112,)),
            pypto.Tensor([-1], pypto.DT_FP32, "out", ori_shape=(112,)),
        )
    check_snapshot(after, IR)
