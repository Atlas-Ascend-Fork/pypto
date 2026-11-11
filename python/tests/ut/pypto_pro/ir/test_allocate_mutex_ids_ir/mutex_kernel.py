# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Python DSL kernel shared by the allocation and synchronization dump tests."""

import pypto_pro.language as pl

from pypto.pypto_impl import ir


def parse_copy_tiles():
    @pl.jit(auto_mutex=True)
    def copy_tiles(a: pl.Tensor[[1, 64], pl.DT_FP32], out: pl.Tensor[[1, 64], pl.DT_FP32]):
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        group = pl.make_tile_group(type=tt, addrs=[0, 256], mutex_ids="auto", depth=2)
        with pl.section_vector():
            tile = group.next()
            pl.load(tile, a, [0, 0])
            pl.store(out, tile, [0, 0])
        # Declared after its aliased slot is used: allocation must see the whole program.
        pin = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[7])   # noqa: F841

    program, matched = copy_tiles.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert matched
    return program
