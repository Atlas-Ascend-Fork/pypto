#!/usr/bin/env python3
# coding: utf-8
"""kirin9050 matmul 诊断脚本（不改 pypto，直接跑）。

用法:
  cd <pypto>/python/tests/ut/kirin/kirin9050/single_operation
  source ~/z30083490/Ascend-Learning/00-env/AscendC/sourcemekirin
  export KIRIN_SOC=Kirin9050 KIRIN_TEST_SUBDIR=kirin9050
  python diag_matmul.py
"""

from kirin.common import check_nan, compare_cos
from kirin.common.common_matmul import get_matmul_kernel
import numpy as np
import torch

import pypto

SOC = "Kirin9050"
TILE32 = ([32, 32], [32, 32], [64, 64])  # 003 用的 tile
TILE16 = ([16, 16], [32, 32], [16, 16])  # 007 用的 tile

# (name, shape_a, shape_b, shape_c, tile, a_trans, b_trans, 说明)
CASES = [
    # ---- 003 家族：分离 M 对齐 / K 余量 / 片数 ----
    ("d_003", (30, 150), (150, 60), (30, 60), TILE32, False, False, "对照：M30 K150(4x32+22)"),
    ("d_m32_k150", (32, 150), (150, 60), (32, 60), TILE32, False, False, "M 对齐, K 有余量22"),
    ("d_m30_k128", (30, 128), (128, 60), (30, 60), TILE32, False, False, "M 非对齐, K=4x32 全对齐"),
    ("d_m32_k128", (32, 128), (128, 60), (32, 60), TILE32, False, False, "全对齐 4 片"),
    ("d_m30_k160", (30, 160), (160, 60), (30, 60), TILE32, False, False, "M 非对齐, K=5x32 全对齐"),
    ("d_m32_k160", (32, 160), (160, 60), (32, 60), TILE32, False, False, "全对齐 5 片"),
    ("d_m30_k32", (30, 32), (32, 60), (30, 60), TILE32, False, False, "M 非对齐, 单 K 片"),
    ("d_m30_k64", (30, 64), (64, 60), (30, 60), TILE32, False, False, "M 非对齐, 2 片对齐"),
    ("d_m30_k96", (30, 96), (96, 60), (30, 60), TILE32, False, False, "M 非对齐, 3 片对齐"),
    ("d_m16_k150", (16, 150), (150, 60), (16, 60), TILE32, False, False, "M16 对齐, K 有余量22"),
    ("d_003_k16tile", (30, 150), (150, 60), (30, 60), TILE16, False, False, "003 形状但 k tile=16"),
    # ---- 007 家族：GEMV(M=1)+b_trans，分离 2D/3D、trans ----
    ("g_2d_bt", (1, 64), (64, 64), (1, 64), TILE16, False, True, "2D M=1 K64 b_trans"),
    ("g_2d_nbt", (1, 64), (64, 64), (1, 64), TILE16, False, False, "2D M=1 K64 非转置"),
    ("g_3d1_bt", (1, 1, 64), (1, 64, 64), (1, 1, 64), TILE16, False, True, "3D batch=1 b_trans"),
    ("g_3d16_bt", (16, 1, 64), (16, 64, 64), (16, 1, 64), TILE16, False, True, "3D batch=16 b_trans(=007)"),
    ("g_3d16_nbt", (16, 1, 64), (16, 64, 64), (16, 1, 64), TILE16, False, False, "3D batch=16 非转置"),
    # ---- a_trans（预期全挂，验证建模）----
    ("x_2d_at", (40, 130), (40, 64), (130, 64), TILE32, True, False, "2D a_trans(=005)"),
    ("x_2d_at_k32", (32, 130), (32, 64), (130, 64), TILE32, True, False, "a_trans K=32 单片"),
]


def run_case(name, sa, sb, sc, tile, a_trans, b_trans, note):
    in_dtype, out_dtype = pypto.DT_FP16, pypto.DT_FP16
    kern = get_matmul_kernel(SOC, name, in_dtype, out_dtype, tile, False, None, a_trans, b_trans)
    a = torch.rand(sa, dtype=torch.float16)
    b = torch.rand(sb, dtype=torch.float16)
    c = torch.rand(sc, dtype=torch.float16)
    a_new = (
        a.permute(0, 1, 3, 2)
        if (a_trans and a.dim() == 4)
        else a.permute(0, 2, 1)
        if (a_trans and a.dim() == 3)
        else (a.permute(1, 0) if a_trans else a)
    )
    b_new = (
        b.permute(0, 1, 3, 2)
        if (b_trans and b.dim() == 4)
        else b.permute(0, 2, 1)
        if (b_trans and b.dim() == 3)
        else (b.permute(1, 0) if b_trans else b)
    )
    try:
        kern(a, b, c)
        golden = a_new @ b_new
        check_nan(c, name=name)
        cos = abs(compare_cos(np.array(c.cpu()), np.array(golden.cpu())))
        print(f"{name:16s} cos={cos:.6f} {'PASS' if cos >= 0.9999 else 'FAIL'}   {note}")
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")[:160]
        print(f"{name:16s} ERROR  {msg}   {note}")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("suppress-seed note: torch.manual_seed(0) 固定输入，便于跨轮对比")
    for case in CASES:
        run_case(*case)
