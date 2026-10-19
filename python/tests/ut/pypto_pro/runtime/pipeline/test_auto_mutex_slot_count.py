# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pipeline declaration scans for AUTO mutex TileGroups."""

import ast
from types import SimpleNamespace

import pypto_pro.language as pl
from pypto_pro.runtime.pipeline._cross_core_scanner import (
    scan_buffer_addr_ranges,
    scan_buffer_decls,
    scan_buffer_mutex_ids,
    scan_buffer_slot_counts,
)
from pypto_pro.runtime.pipeline._sync_graph import _slot_count


def test_auto_mutex_uses_depth_without_becoming_a_manual_id_list():
    module = ast.parse(
        """
def kernel():
    auto_group = pl.make_tile_group(
        type=pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0, mutex_ids="auto", depth=2)
    manual_group = pl.make_tile_group(
        type=pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=256, mutex_ids=[4, 5])
"""
    )
    decls = scan_buffer_decls(module.body[0])
    closure_vars = {"pl": pl}

    mutex_ids = scan_buffer_mutex_ids(decls, closure_vars)
    slot_counts = scan_buffer_slot_counts(decls, closure_vars)
    addr_ranges = scan_buffer_addr_ranges(decls, closure_vars)

    assert mutex_ids == {"manual_group": (4, 5)}
    assert slot_counts == {"auto_group": 2, "manual_group": 2}
    assert addr_ranges["auto_group"][1] == [(0, 128), (128, 256)]
    info = SimpleNamespace(sync=SimpleNamespace(slot_counts=slot_counts, addr_ranges=addr_ranges))
    assert _slot_count(info, "auto_group") == 2
