# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reinterpret ST: eight Vector byte-view checks share one binary; NZ->ZN uses a Cube binary.

Dtype and shape views must preserve bytes, rotating group views must share their
cursor, and an explicit valid window must leave the rest of the output untouched.
The Cube case checks that the NZ->ZN view is consumed as a transposed matmul input.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

pytestmark = pytest.mark.soc("950", "960")
ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
TILE = 64


@pl.jit(auto_mutex=True)
def vector_reinterpret_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    wide: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    dtype_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    wide_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    shape_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    group_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    tail_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    chain_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    move_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    nested_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
):
    """Eight independent outputs share one Vector binary; all views alias bytes."""
    tile_type = pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    wide_type = pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    shape_type = pl.TileType(shape=[TILE * 2, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    bf16_type = pl.TileType(shape=[TILE, TILE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    # Each independent path owns its buffers and mutexes (106,496 bytes total).
    dtype_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    wide_group = pl.make_tile_group(type=wide_type, addrs=0x2000, mutex_ids=[1])
    shape_group = pl.make_tile_group(type=shape_type, addrs=0x6000, mutex_ids=[2])
    rotating_group = pl.make_tile_group(type=tile_type, addrs=[0xA000, 0xC000], mutex_ids=[3, 4])
    tail_group = pl.make_tile_group(type=tile_type, addrs=0xE000, mutex_ids=[5])
    chain_group = pl.make_tile_group(type=shape_type, addrs=0x10000, mutex_ids=[6])
    move_source = pl.make_tile_group(type=tile_type, addrs=0x14000, mutex_ids=[7])
    move_target = pl.make_tile_group(type=bf16_type, addrs=0x16000, mutex_ids=[8])
    nested_group = pl.make_tile_group(type=tile_type, addrs=0x18000, mutex_ids=[9])

    with pl.section_vector():
        # Same-width dtype view: load the original type, then store its raw bits.
        for i in pl.range(0, x.shape[0], TILE):
            dtype_tile = dtype_group.current()
            pl.load(dtype_tile, x, [i, 0])
            dtype_view = pl.reinterpret(dtype_tile, shape=[TILE, TILE], dtype=pl.DT_BF16)
            pl.store(dtype_out, dtype_view, [i, 0])

        # FP32 -> FP16 changes element count, not bytes or their ordering.
        for i in pl.range(0, wide.shape[0], TILE):
            wide_tile = wide_group.current()
            pl.load(wide_tile, wide, [i, 0])
            wide_view = pl.reinterpret(wide_tile, shape=[TILE * 2, TILE], dtype=pl.DT_FP16)
            pl.store(wide_out, wide_view, [i * 2, 0])

        # Reshape a loaded 128x64 buffer into a 64x128 view without a transpose.
        for i in pl.range(0, x.shape[0], TILE * 2):
            shape_tile = shape_group.current()
            pl.load(shape_tile, x, [i, 0])
            shape_view = pl.reinterpret(shape_tile, shape=[TILE, TILE * 2])
            pl.store(shape_out, shape_view, [i // 2, 0])

        # Reinterpreting the group must share its rotating cursor. The original
        # advances it; current() on the view must select that same physical slot.
        rotating_view = pl.reinterpret(rotating_group, shape=[TILE, TILE], dtype=pl.DT_BF16)
        for i in pl.range(0, x.shape[0], TILE):
            pl.load(rotating_group.next(), x, [i, 0])
            pl.store(group_out, rotating_view.current(), [i, 0])

        tail_tile = tail_group.current()
        pl.load(tail_tile, x, [0, 0])
        tail_view = pl.reinterpret(tail_tile, shape=[TILE, TILE], dtype=pl.DT_BF16)
        pl.set_validshape(tail_view, [rows, cols])
        pl.store(tail_out, tail_view, [0, 0])

        for i in pl.range(0, x.shape[0], TILE * 2):
            chain_tile = chain_group.current()
            pl.load(chain_tile, x, [i, 0])
            chain_view = pl.reinterpret(
                pl.reinterpret(chain_tile, shape=[TILE, TILE * 2]), shape=[TILE, TILE * 2], dtype=pl.DT_BF16
            )
            pl.store(chain_out, chain_view, [i // 2, 0])

        # Vec -> Vec is a supported move path; Mat -> Vec is not.
        for i in pl.range(0, x.shape[0], TILE):
            move_tile = move_source.current()
            pl.load(move_tile, x, [i, 0])
            pl.move(move_target.current(), pl.reinterpret(move_tile, shape=[TILE, TILE], dtype=pl.DT_BF16))
            # The BF16 tile -> FP16 tensor store must preserve bits as well.
            pl.store(move_out, move_target.current(), [i, 0])

        # These APIs take tile indices; contrast with the element offsets above.
        for block in pl.range(0, x.shape[0] // TILE):
            pl.load_tile(pl.reinterpret(nested_group.current(), shape=[TILE, TILE], dtype=pl.DT_BF16), x, [block, 0])
            pl.store_tile(
                nested_out, pl.reinterpret(nested_group.current(), shape=[TILE, TILE], dtype=pl.DT_BF16), [block, 0]
            )


@pl.jit(auto_mutex=True)
def nz_to_zn_matmul_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    m0, k0 = TILE, TILE
    mat_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
    t1 = pl.make_tile_group(type=mat_type, addrs=0x0000, mutex_ids=[0])
    t2 = pl.make_tile_group(type=mat_type, addrs=0x10000, mutex_ids=[1])

    left_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ)
    right_type = pl.TileType(shape=[m0, k0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN)
    acc_type = pl.TileType(
        shape=[m0, m0], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
    )
    l0a = pl.make_tile_group(type=left_type, addrs=0x0000, mutex_ids=[2])
    l0b = pl.make_tile_group(type=right_type, addrs=0x0000, mutex_ids=[3])
    acc = pl.make_tile_group(type=acc_type, addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        pl.load_tile(t1.current(), a, [0, 0])
        pl.load_tile(t2.current(), a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        # The same NZ buffer re-declared as [K0, M0] ZN — hardware reads B' = A^T.
        t2r = pl.reinterpret(t2.current(), shape=[k0, m0], layout=pl.Layout.ZN)
        pl.move(l0a.current(), t1.current())
        pl.move(l0b.current(), t2r)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        pl.matmul(acc.current(), l0a.current(), l0b.current())
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        pl.store(out, acc.current(), [0, 0])


def test_vector_reinterpret_views():
    torch.npu.set_device(ST_DEVICE)
    generator = torch.Generator().manual_seed(20260915)
    checks = []
    # Multiple blocks expose wrong offsets and stale double-buffer cursors;
    # two windows check both copied bytes and untouched output outside the tail.
    for m, rows, cols in ((128, 32, 16), (256, 47, 31)):
        x_cpu = torch.randint(-2048, 2048, (m, TILE), generator=generator).to(torch.float16)
        wide_cpu = torch.randn(m // 2, TILE, generator=generator)
        x, wide = x_cpu.to(ST_DEVICE), wide_cpu.to(ST_DEVICE)
        outputs = {
            "dtype": torch.zeros(m, TILE, dtype=torch.bfloat16, device=ST_DEVICE),
            "width": torch.zeros(m, TILE, dtype=torch.float16, device=ST_DEVICE),
            "shape": torch.zeros(m // 2, TILE * 2, dtype=torch.float16, device=ST_DEVICE),
            "group": torch.zeros(m, TILE, dtype=torch.bfloat16, device=ST_DEVICE),
            "tail": torch.zeros(TILE, TILE, dtype=torch.bfloat16, device=ST_DEVICE),
            "chain": torch.zeros(m // 2, TILE * 2, dtype=torch.bfloat16, device=ST_DEVICE),
            "move": torch.zeros(m, TILE, dtype=torch.float16, device=ST_DEVICE),
            "nested": torch.zeros(m, TILE, dtype=torch.bfloat16, device=ST_DEVICE),
        }
        vector_reinterpret_kernel[None, 1](x, wide, *outputs.values(), rows, cols)
        torch.npu.synchronize()

        bits = x_cpu.view(torch.int16)
        tail_bits = torch.zeros(TILE, TILE, dtype=torch.int16)
        tail_bits[:rows, :cols] = bits[:rows, :cols]
        expected = {
            "dtype": bits,
            "width": wide_cpu.view(torch.int16).reshape(m, TILE),
            "shape": bits.reshape(m // 2, TILE * 2),
            "group": bits,
            "tail": tail_bits,
            "chain": bits.reshape(m // 2, TILE * 2),
            "move": bits,
            "nested": bits,
        }
        checks.append((m, outputs, expected))

    for m, outputs, expected in checks:
        for name, output in outputs.items():
            actual = output.cpu().view(torch.int16)
            assert torch.equal(actual, expected[name]), f"{name} reinterpret differs at m={m}"
            logging.info("%s reinterpret: m=%d, bit-identical=True", name, m)


def test_nz_to_zn_matmul():
    m = k = TILE
    torch.npu.set_device(ST_DEVICE)
    a = torch.randint(-8, 9, (m, k), dtype=torch.float16, device=ST_DEVICE)
    out = torch.zeros(m, m, dtype=torch.float32, device=ST_DEVICE)
    nz_to_zn_matmul_kernel[None, 1](a, out)
    torch.npu.synchronize()
    got = out.cpu().float()
    golden = a.cpu().float() @ a.cpu().float().T
    assert torch.allclose(got, golden, rtol=1e-2, atol=1e-1), (got, golden)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
