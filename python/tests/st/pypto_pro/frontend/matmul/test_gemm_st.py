# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native GEMM numerical regressions; all buffers use make_tile_group."""

import os

import pypto_pro.language as pl
import pytest
import torch

import pypto  # noqa: F401

pytestmark = pytest.mark.soc("950")
DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"


def _require_a5():
    torch.npu.set_device(DEVICE)
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("GEMM regressions require Ascend950")


def _ordinary_kernel(*, trans_a=False, trans_b=False, split=0, split_n=False,
                     m=64, n=64, k=128, valid_m=64, valid_n=64, valid_k=128,
                     accumulate=False, with_bias=False, bf16=False, auto_ids=False, stage_k=None,
                     compact=1, restore=False, acc_view=False):
    dtype = pl.DT_BF16 if bf16 else pl.DT_FP16
    shape_a = [k, m] if trans_a else [m, k]
    shape_b = [n, k] if trans_b else [k, n]
    stage_a = [m, split if split and not split_n else (stage_k or k)]
    stage_b = [k if split_n else (split or stage_k or k), split if split_n else n]
    acc_shape = [m, n * 2 if acc_view else n]
    acc_addr = 65536 if acc_view else 0
    ids_a, ids_b, ids_la, ids_lb, ids_c = (
        ("auto", "auto", "auto", "auto", "auto") if auto_ids else ([0], [1], [2, 3], [4, 5], [6])
    )

    name = (f"gemm_t{int(trans_a)}{int(trans_b)}_s{split}_n{int(split_n)}_"
            f"v{valid_m}_{valid_n}_{valid_k}_b{int(with_bias)}_a{int(accumulate)}_"
            f"bf{int(bf16)}_auto{int(auto_ids)}_c{compact}_r{int(restore)}")
    if stage_k is not None:
        name += f"_p{stage_k}"
    if acc_view:
        name += "_view"

    @pl.jit(auto_mutex=True, name=name)
    def kernel(a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
               b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype],
               bias: pl.Tensor[[1, pl.DYNAMIC], pl.DT_FP32],
               out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
        ga = pl.make_tile_group(
            type=pl.TileType(shape=shape_a, dtype=dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0, mutex_ids=ids_a, depth=1,
        )
        gb = pl.make_tile_group(
            type=pl.TileType(shape=shape_b, dtype=dtype, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x10000, mutex_ids=ids_b, depth=1,
        )
        la = pl.make_tile_group(
            type=pl.TileType(shape=stage_a, dtype=dtype, target_memory=pl.MemorySpace.Left, compact=compact),
            addrs=0, mutex_ids=ids_la, depth=2,
        )
        lb = pl.make_tile_group(
            type=pl.TileType(shape=stage_b, dtype=dtype, target_memory=pl.MemorySpace.Right, compact=compact),
            addrs=0, mutex_ids=ids_lb, depth=2,
        )
        gc = pl.make_tile_group(
            type=pl.TileType(shape=acc_shape, dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Acc, compact=compact),
            addrs=acc_addr, mutex_ids=ids_c, depth=1,
        )
        if with_bias:
            bias_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[1, n], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
                addrs=0x20000, mutex_ids=[7],
            )
            bias_l0 = pl.make_tile_group(
                type=pl.TileType(shape=[1, n], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
                addrs=0, mutex_ids=[8],
            )
        with pl.section_cube():
            x = ga.current()
            y = gb.current()
            z = gc.current()
            if acc_view:
                z = pl.reinterpret(z, shape=[m, n])
            pl.load(x, a, [0, 0])
            pl.load(y, b, [0, 0])
            if restore:
                left = la.current()
                right = lb.current()
                if split_n:
                    pl.gemm(z, x, y, l0a=left, l0b=right, splitn=split, n=split, k=64)
                    pl.gemm(z, x, y, l0a=left, l0b=right, splitn=split)
                elif split:
                    pl.gemm(z, x, y, l0a=left, l0b=right, splitk=split, m=48, n=40, k=96)
                    pl.gemm(z, x, y, l0a=left, l0b=right, splitk=split)
                else:
                    pl.gemm(z, x, y, l0a=left, l0b=right, m=48, n=40, k=64)
                    pl.gemm(z, x, y, l0a=left, l0b=right)
            elif with_bias:
                bias_mat = bias_l1.current()
                bias_table = bias_l0.current()
                pl.load(bias_mat, bias, [0, 0])
                pl.move(bias_table, bias_mat)
                pl.gemm(z, x, y, l0a=la, l0b=lb, bias=bias_table, m=valid_m, n=valid_n, k=valid_k)
            elif split_n:
                pl.gemm(z, x, y, l0a=la, l0b=lb, splitn=split, n=valid_n, k=valid_k,
                        trans_a=trans_a, trans_b=trans_b)
            elif split:
                pl.gemm(z, x, y, l0a=la, l0b=lb, splitk=split, m=valid_m, n=valid_n, k=valid_k,
                        trans_a=trans_a, trans_b=trans_b)
            else:
                pl.gemm(z, x, y, l0a=la, l0b=lb, m=valid_m, n=valid_n, k=valid_k,
                        trans_a=trans_a, trans_b=trans_b)
            if accumulate:
                pl.gemm(z, x, y, l0a=la, l0b=lb, init=False, m=valid_m, n=valid_n, k=valid_k)
            pl.store(out, z, [0, 0])
    return kernel


_ORDINARY_CASES = [
    {},
    {"trans_a": True},
    {"trans_b": True},
    {"trans_a": True, "trans_b": True},
    {"valid_m": 37, "valid_n": 23, "valid_k": 19},
    {"split": 64, "valid_m": 48, "valid_n": 40, "valid_k": 104},
    {"split": 64, "trans_a": True},
    {"split": 32, "split_n": True},
    {"split": 32, "split_n": True, "valid_n": 40},
    {"split": 48, "split_n": True, "n": 80, "valid_n": 80, "valid_k": 104},
    {"split": 32, "split_n": True, "trans_b": True, "auto_ids": True},
    {"with_bias": True},
    {"with_bias": True, "valid_m": 48, "valid_n": 40},
    {"accumulate": True},
    {"bf16": True, "valid_m": 48, "valid_n": 40},
    {"stage_k": 64, "valid_k": 64},
    {"compact": 0},
    {"compact": 0, "split": 64},
    {"compact": 0, "split": 32, "split_n": True},
    {"restore": True},
    {"restore": True, "split": 64},
    {"restore": True, "split": 48, "split_n": True, "n": 80, "valid_n": 80},
    {"split": 32, "split_n": True, "acc_view": True},
    {"split": 48, "split_n": True, "acc_view": True, "n": 80, "valid_n": 80},
]
_ORDINARY_IDS = [
    "plain", "trans_a", "trans_b", "trans_both", "odd_extents", "split_k_tail",
    "split_k_transpose", "split_n", "split_n_rounded_tail", "split_n_declared_tail",
    "split_n_transpose_auto", "bias", "bias_tail", "accumulate", "bf16_tail", "smaller_staging",
    "full_noncompact", "split_k_full_noncompact", "split_n_full_noncompact",
    "restore_default_mnk", "split_k_restore_defaults", "split_n_restore_defaults",
    "split_n_acc_view", "split_n_acc_view_tail",
]


@pytest.mark.parametrize("options", _ORDINARY_CASES, ids=_ORDINARY_IDS)
def test_gemm_numbers_and_store_bounds(options):
    _require_a5()
    params = dict(m=64, n=64, k=128, valid_m=64, valid_n=64, valid_k=128)
    params.update(options)
    m, n, k = params["m"], params["n"], params["k"]
    dtype = torch.bfloat16 if params.get("bf16") else torch.float16
    generator = torch.Generator().manual_seed(42)
    a = torch.randn((m, k), generator=generator, dtype=dtype)
    b = torch.randn((k, n), generator=generator, dtype=dtype)
    bias = torch.randn((1, n), generator=generator)
    a_arg = a.T.contiguous() if params.get("trans_a") else a
    b_arg = b.T.contiguous() if params.get("trans_b") else b
    out = torch.full((m, n), 7.0, device=DEVICE)
    _ordinary_kernel(**params)(a_arg.to(DEVICE), b_arg.to(DEVICE), bias.to(DEVICE), out)
    torch.npu.synchronize()
    vm, vn, vk = params["valid_m"], params["valid_n"], params["valid_k"]
    golden = a[:vm, :vk].float() @ b[:vk, :vn].float()
    if params.get("accumulate"):
        golden *= 2
    if params.get("with_bias"):
        golden += bias[:, :vn]
    actual = out.cpu()
    torch.testing.assert_close(actual[:vm, :vn], golden, rtol=1e-3, atol=1e-3)
    assert torch.all(actual[vm:, :] == 7) and torch.all(actual[:vm, vn:] == 7)


@pl.jit(auto_mutex=True)
def gemm_double_buffered_rows(a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
                              b: pl.Tensor[[128, 64], pl.DT_FP16],
                              out: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP32]):
    ga = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0, mutex_ids=[0, 1],
    )
    gb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2],
    )
    la = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, compact=1),
        addrs=0, mutex_ids=[3, 4],
    )
    lb = pl.make_tile_group(
        type=pl.TileType(shape=[128, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0, mutex_ids=[5],
    )
    gc = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, compact=1),
        addrs=0, mutex_ids=[6, 7],
    )
    with pl.section_cube():
        y = gb.current()
        pl.load(y, b, [0, 0])
        for i in pl.range(0, out.shape[0], 64):
            rows = pl.min(64, out.shape[0] - i)
            x = ga.next()
            z = gc.next()
            pl.set_validshape(x, [rows, 128])
            pl.load(x, a, [i, 0])
            pl.gemm(z, x, y, l0a=la, l0b=lb, m=rows)
            pl.store(out, z, [i, 0])


def test_gemm_double_buffered_rows():
    _require_a5()
    generator = torch.Generator().manual_seed(53)
    a = torch.randn((264, 128), generator=generator, dtype=torch.float16)
    b = torch.randn((128, 64), generator=generator, dtype=torch.float16)
    out = torch.zeros((264, 64), device=DEVICE)
    gemm_double_buffered_rows(a.to(DEVICE), b.to(DEVICE), out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), a.float() @ b.float(), rtol=1e-3, atol=1e-3)
