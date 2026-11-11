# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
from pathlib import Path
import re

import pypto_pro.language as pl

from pypto.pypto_impl import ir


def test_preserve_manual_and_unbound_tiles():
    @pl.jit(auto_mutex=True)
    def kernel():
        tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        auto = pl.make_tile_group(type=tt, addrs=0, mutex_ids="auto", depth=1)  # noqa: F841
        manual = pl.make_tile_group(type=tt, addrs=256, mutex_ids=[[3, 5]])     # noqa: F841
        unbound = pl.make_tile_group(type=tt, addrs=512, depth=1)               # noqa: F841

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    printed = re.sub(r"memref_id=\d+", "memref_id=<id>", str(program))
    golden = Path(__file__).with_suffix(".pypto").read_text(encoding="utf-8")
    assert printed.rstrip("\n") == golden.rstrip("\n")
