#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer factories in a module of their own, the way a production kernel keeps them.

``attention/flash_attn_grad/op_kernel/fag_buffers.py`` is a module of exactly these: the kernel
body calls ``make_q_l1()`` and every address, id and shape is a name only this module knows.
"""

import pypto_pro.language as pl

TM, TK = 64, 64

_ADDRS = [0x0000, 0x8000]
_MUTEX_IDS = [0, 1]
_FWD_IDS = [0, 1]
_BWD_IDS = [2, 3]


def make_shared_l1():
    """The cross-core L1 handover buffer, configured from this module's own names."""
    return pl.make_tile_group(
        type=pl.TileType(shape=[TM, TK], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=_ADDRS, mutex_ids=_MUTEX_IDS, fwd_ids=_FWD_IDS, bwd_ids=_BWD_IDS)
