# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
import pypto
import pypto.pypto_impl as pypto_impl

from .test_common import run_root_function


def war_conflict_pypto(
    a: pypto.Tensor[[pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16],
    w: pypto.Tensor[[pypto.STATIC, pypto.STATIC], pypto.DT_BF16],
    out: pypto.Tensor[[pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16],
):
    tile_b = 128
    hidden = 128
    batch_size = a.shape[0]
    b_loop = (batch_size + tile_b - 1) // tile_b
    dtype = a.dtype

    for b_idx in pypto.loop(b_loop, unroll_list=[1, 2]):
        off = b_idx * tile_b

        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        pypto.set_vec_tile_shapes(128, 128)

        a_view = pypto.view(a, [tile_b, hidden], [off, 0])

        # ---- C1: Cube — matmul ----
        c1 = pypto.matmul(a_view, w, out_dtype=dtype)

        # ---- V1: Vector — add ----
        v1 = pypto.add(c1, a_view)

        # ---- Write output rows [off, off+tile_b) ----
        pypto.assemble(v1, [off, 0], out)

        # Read back the rows just written
        o_read = pypto.view(out, [tile_b, hidden], [off, 0])

        # ---- C2: Cube — matmul on read-back data ----
        c2 = pypto.matmul(o_read, w, out_dtype=dtype)

        # ---- V2: Vector — add ----
        v2 = pypto.add(c2, c1)

        # ---- Write to OVERLAPPING region: rows [off+tile_b//2, off+tile_b//2+tile_b) ----
        # This overlaps with the read region [off, off+tile_b) by tile_b//2 rows.
        pypto.assemble(v2, [off + tile_b // 2, 0], out)


def test_war_conflict():
    a = pypto.Tensor([-1, 128], pypto.DT_BF16)
    w = pypto.Tensor([128, 128], pypto.DT_BF16)
    out = pypto.Tensor([-1, 128], pypto.DT_BF16)

    with pypto.options(pass_options={"enable_slice": True}):
        prog = run_root_function(war_conflict_pypto, a, w, out, create_new_logical_tensor=True)

    slot_info = pypto_impl.GetSlotInfo()
    out_slot = slot_info['out']
    for name, func in prog.functions.items():
        if "_hiddenfunc" in name:
            # Includes one loop-entry incast and the outcasts.
            out_params = [x.name for x in func.params if x.name.startswith("out")]
            if "Unroll1" in name:
                assert len(out_params) == 3  # 1 incast + 2 outcasts
                assert set(slot_info[x] for x in out_params) == {out_slot}
            elif "Unroll2" in name:
                assert len(out_params) == 4  # 1 incast + 3 outcasts
                assert set(slot_info[x] for x in out_params) == {out_slot}
