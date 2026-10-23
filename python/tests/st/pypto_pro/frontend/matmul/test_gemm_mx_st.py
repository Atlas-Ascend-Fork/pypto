# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MX GEMM versus native staging and independently dequantized operands."""

import os

import pypto_pro.language as pl
import pytest
from test_matmul_mx import _MX_DTYPES, _make_mx_operand_pair
import torch

import pypto  # noqa: F401

pytestmark = pytest.mark.soc("950")
DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


def _mx_kernel(a_label, b_label, *, trans_a=False, trans_b=False,
               m=64, n=64, k=128, accumulate=False, sugar=True, stage_k=None, plane_compact=0,
               compact=1, restore=False):
    a_dtype, _, fp4 = _MX_DTYPES[a_label]
    b_dtype = _MX_DTYPES[b_label][0]
    shape_a = [128, 64] if trans_a else [64, 128]
    shape_b = [64, 128] if trans_b else [128, 64]
    physical_k = stage_k or 128
    scale_slot = (64 * physical_k // (2 if fp4 else 1)) >> 4
    name = (f"gemm_mx_{a_label}_{b_label}_t{int(trans_a)}{int(trans_b)}_v{m}_{n}_{k}_"
            f"a{int(accumulate)}_s{int(sugar)}_c{compact}_r{int(restore)}")
    if stage_k is not None:
        name += f"_p{stage_k}"
    if plane_compact:
        name += "_pc1"

    @pl.jit(auto_mutex=True, name=name)
    def kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], a_dtype],
               b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], b_dtype],
               scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
               scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
        ga = pl.make_tile_group(
            type=pl.TileType(shape=shape_a, dtype=a_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0, mutex_ids=[0],
        )
        gb = pl.make_tile_group(
            type=pl.TileType(shape=shape_b, dtype=b_dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x10000, mutex_ids=[1],
        )
        gsa = pl.make_tile_group(
            type=pl.TileType(shape=[64, 4], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
            addrs=0x20000, mutex_ids=[2],
        )
        gsb = pl.make_tile_group(
            type=pl.TileType(shape=[4, 64], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.Mat, layout=pl.NN),
            addrs=0x21000, mutex_ids=[3],
        )
        la = pl.make_tile_group(
            type=pl.TileType(shape=[64, physical_k], dtype=a_dtype, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, compact=compact),
            addrs=0, mutex_ids=[4, 5],
        )
        lb = pl.make_tile_group(
            type=pl.TileType(shape=[physical_k, 64], dtype=b_dtype, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN, compact=compact),
            addrs=0, mutex_ids=[6, 7],
        )
        sa = pl.make_tile_group(
            type=pl.TileType(shape=[64, physical_k // 32], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.ScaleLeft,
                             layout=pl.ZZ, compact=plane_compact),
            addrs=[0, scale_slot], mutex_ids=[8, 9],
        )
        sb = pl.make_tile_group(
            type=pl.TileType(shape=[physical_k // 32, 64], dtype=pl.DT_FP8E8M0, target_memory=pl.MemorySpace.ScaleRight,
                             layout=pl.NN, compact=plane_compact),
            addrs=[0, scale_slot], mutex_ids=[10, 11],
        )
        gc = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             compact=compact),
            addrs=0, mutex_ids=[12],
        )
        with pl.section_cube():
            x = ga.current()
            y = gb.current()
            sx = gsa.current()
            sy = gsb.current()
            z = gc.current()
            pl.load(x, a, [0, 0])
            pl.load(y, b, [0, 0])
            pl.load(sx, scale_a, [0, 0, 0])
            if trans_b:
                pl.load(sy, scale_b, [0, 0, 0], order=[1, 0])
            else:
                pl.load(sy, scale_b, [0, 0, 0])
            if sugar and restore:
                al = la.current()
                br = lb.current()
                sal = sa.current()
                sbr = sb.current()
                pl.gemm_mx(z, x, y, sx, sy, l0a=al, l0b=br, scale_a=sal, scale_b=sbr,
                           m=48, n=40, k=64)
                pl.gemm_mx(z, x, y, sx, sy, l0a=al, l0b=br, scale_a=sal, scale_b=sbr)
            elif sugar:
                pl.gemm_mx(z, x, y, sx, sy, l0a=la, l0b=lb, scale_a=sa, scale_b=sb,
                           m=m, n=n, k=k, trans_a=trans_a, trans_b=trans_b)
                if accumulate:
                    pl.gemm_mx(z, x, y, sx, sy, l0a=la, l0b=lb, scale_a=sa, scale_b=sb,
                               m=m, n=n, k=k, init=False, trans_a=trans_a, trans_b=trans_b)
            else:
                if trans_a:
                    x = pl.reinterpret(x, shape=[64, 128], layout=pl.ZN)
                if trans_b:
                    y = pl.reinterpret(y, shape=[128, 64], layout=pl.ZN)
                al = la.next()
                br = lb.next()
                sal = sa.next()
                sbr = sb.next()
                pl.set_validshape(al, [m, k])
                pl.set_validshape(br, [k, n])
                pl.set_validshape(z, [m, n])
                pl.set_validshape(sal, [m, k // 32])
                pl.set_validshape(sbr, [k // 32, n])
                if stage_k is not None:
                    pl.move(al, x, [0, 0])
                    pl.move(sal, sx, [0, 0])
                    pl.move(br, y, [0, 0])
                    pl.move(sbr, sy, [0, 0])
                else:
                    pl.move(al, x)
                    pl.move(sal, sx)
                    pl.move(br, y)
                    pl.move(sbr, sy)
                pl.matmul_mx(z, al, br, sal, sbr)
                if accumulate:
                    pl.matmul_mx_acc(z, z, al, br, sal, sbr)
            pl.store(out, z, [0, 0])
    return kernel


_MX_CASES = [
    {},
    {"m": 48, "n": 40},
    {"k": 64, "plane_compact": 1},
    {"m": 48, "n": 40, "k": 64, "plane_compact": 1},
    {"trans_a": True},
    {"trans_b": True},
    {"trans_a": True, "trans_b": True},
    {"accumulate": True},
    {"stage_k": 64, "k": 64},
    {"compact": 0},
    {"restore": True, "plane_compact": 1},
]
_MX_IDS = [
    "plain", "mn_tail", "k_tail", "mnk_tail", "trans_a", "trans_b", "trans_both", "accumulate", "smaller_staging",
    "full_noncompact", "restore_default_mnk",
]


@pytest.mark.parametrize("labels", [
    ("mxfp8_e4m3", "mxfp8_e5m2"),
    ("mxfp4_e2m1", "mxfp4_e1m2"),
], ids=["fp8", "fp4"])
@pytest.mark.parametrize("options", _MX_CASES, ids=_MX_IDS)
def test_gemm_mx_matches_primitives_and_dequantized_reference(labels, options):
    torch.npu.set_device(DEVICE)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("MX GEMM regressions require Ascend950")
    args = _make_mx_operand_pair(
        *labels, 64, 128, 64, DEVICE, transpose_a=options.get("trans_a", False),
        direct_b_kn=not options.get("trans_b", False),
    )
    a, b, sa, sb, a_ref, b_ref, _ = args
    m, n, k = options.get("m", 64), options.get("n", 64), options.get("k", 128)
    golden = a_ref[:m, :k] @ b_ref[:k, :n]
    if options.get("accumulate"):
        golden *= 2
    outputs = []
    for sugar in (False, True):
        out = torch.full((64, 64), 7.0, device=DEVICE)
        _mx_kernel(*labels, sugar=sugar, **options)(a, b, sa, sb, out)
        torch.npu.synchronize()
        actual = out.cpu()
        torch.testing.assert_close(actual[:m, :n], golden, rtol=1e-3, atol=1e-3)
        assert torch.all(actual[m:, :] == 7) and torch.all(actual[:m, n:] == 7)
        outputs.append(actual)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
