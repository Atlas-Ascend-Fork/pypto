#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The graph this demo exports and runs: two nested conditionals over three pypto ops.

``forward`` is a plain ``if / elif / else``. The ``elif`` is what nests: TorchScript emits an outer
``If`` whose else branch holds a second ``If``, so ``add`` lands one level down and ``sub`` and ``mul``
land two. A consumer that only walks the root graph sees no pypto node at all.

The same module serves both entry points. ``run_demo.py`` calls it directly, where Python evaluates the
conditional; ``export_demo.py`` wraps it in ``torch.jit.script`` first, because TRACING would fold the
conditional down to the one branch the example inputs happen to take.
"""
import op  # noqa: F401 -- imported for its side effect: declaring the ops registers their torch.ops entries
import torch
import torch.nn as nn

SHAPE = (1, 8, 1, 64)  # 512 elems — kept small so the SIM run stays fast; tile (1, 4, 1, 64)


# the graph scripted for export and executed by run_demo.py.
class CustomModel(nn.Module):
    def forward(self, input0, input1, use_add, use_sub):
        if bool(use_add):
            out = torch.ops.pypto.add(input0, input1)
        elif bool(use_sub):
            out = torch.ops.pypto.sub(input0, input1)
        else:
            out = torch.ops.pypto.mul(input0, input1)
        return out
