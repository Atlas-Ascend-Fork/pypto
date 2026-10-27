#!/usr/bin/env python3
# coding: utf-8
"""临时诊断：002（128x130x32 fp16 + bias）/ 003（30x150x60）kernel 输出 vs golden。

跑法（P/N 一致，走 pytest 同款流程；结果同时打印并落盘）:
  pytest python/tests/ut/kirin/kirin9050/single_operation/test_kirin9050_dbg002.py -q -s -p no:skipping
  cat /tmp/dbg002_out.txt
可用环境变量 DBG002_LOG 指定落盘路径。
"""

import os

from kirin.common import check_nan, compare_cos
from kirin.common_matmul import get_matmul_kernel
import numpy as np
import torch

import pypto

SOC = "Kirin9050"
TILE = ([16, 16], [32, 32], [16, 16])
SA, SB, SC, BIAS = (128, 130), (130, 32), (128, 32), (1, 32)
DBG_LOG = os.environ.get("DBG002_LOG", "/tmp/dbg002_out.txt")


def _out(line):
    print(line)
    try:
        with open(DBG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _dump(tag, a, b, bias, c, c0, a0, b0, bias0):
    g_before = a0.float() @ b0.float() + bias0.float()
    g_after = a.float() @ b.float() + bias.float()
    cf = c.float().numpy()
    gf = g_before.numpy()
    _out(f"[DBG {tag}] c   min/max/mean/std = {cf.min():.4f}/{cf.max():.4f}/{cf.mean():.4f}/{cf.std():.4f}")
    _out(f"[DBG {tag}] gld min/max/mean/std = {gf.min():.4f}/{gf.max():.4f}/{gf.mean():.4f}/{gf.std():.4f}")
    _out(f"[DBG {tag}] max|g_before-g_after| = {np.abs((g_before - g_after).numpy()).max():.6f}")
    _out(
        f"[DBG {tag}] changed: a={int((a != a0).sum())} b={int((b != b0).sum())} "
        f"bias={int((bias != bias0).sum())} c={int((c != c0).sum())}/{c.numel()}"
    )
    idx = int(np.argmax(np.abs(cf.reshape(-1) - gf.reshape(-1))))
    _out(
        f"[DBG {tag}] worst idx={idx} row={idx // 32} col={idx % 32} "
        f"c={cf.reshape(-1)[idx]:.6f} g={gf.reshape(-1)[idx]:.6f}"
    )
    _out(f"[DBG {tag}] c[:6]   = {cf.reshape(-1)[:6]}")
    _out(f"[DBG {tag}] g[:6]   = {gf.reshape(-1)[:6]}")
    _out(f"[DBG {tag}] cbits[:6]= {c.reshape(-1)[:6].view(torch.int16).numpy()}")
    _out(f"[DBG {tag}] gbits[:6]= {g_before.half().reshape(-1)[:6].view(torch.int16).numpy()}")
    cos = compare_cos(c.numpy(), g_before.numpy())
    _out(f"[DBG {tag}] cos(c, g_before) = {abs(cos):.6f}")


def test_dbg002_rand():
    _out("===== test dbg002_rand =====")
    torch.manual_seed(0)
    kern = get_matmul_kernel(SOC, "dbg002_rand", pypto.DT_FP16, pypto.DT_FP16, TILE, True, BIAS, False, False)
    a = torch.rand(SA, dtype=torch.float16)
    b = torch.rand(SB, dtype=torch.float16)
    bias = torch.rand(BIAS, dtype=torch.float16)
    c = torch.rand(SC, dtype=torch.float16)
    a0, b0, bias0, c0 = a.clone(), b.clone(), bias.clone(), c.clone()
    kern(a, b, bias, c)
    check_nan(c, name="dbg002_rand")
    _dump("rand", a, b, bias, c, c0, a0, b0, bias0)


def test_dbg002_ones():
    _out("===== test dbg002_ones =====")
    kern = get_matmul_kernel(SOC, "dbg002_ones", pypto.DT_FP16, pypto.DT_FP16, TILE, True, BIAS, False, False)
    a = torch.full(SA, 1.0, dtype=torch.float16)
    b = torch.full(SB, 1.0, dtype=torch.float16)
    bias = torch.zeros(BIAS, dtype=torch.float16)
    c = torch.full(SC, 777.0, dtype=torch.float16)
    a0, b0, bias0, c0 = a.clone(), b.clone(), bias.clone(), c.clone()
    kern(a, b, bias, c)
    check_nan(c, name="dbg002_ones")
    _dump("ones", a, b, bias, c, c0, a0, b0, bias0)


def test_dbg003_ones():
    _out("===== test dbg003_ones =====")
    sa, sb, sc = (30, 150), (150, 60), (30, 60)
    kern = get_matmul_kernel(
        SOC, "dbg003_ones", pypto.DT_FP16, pypto.DT_FP16, ([32, 32], [32, 32], [64, 64]), False, None, False, False
    )
    a = torch.full(sa, 1.0, dtype=torch.float16)
    b = torch.full(sb, 1.0, dtype=torch.float16)
    c = torch.full(sc, 777.0, dtype=torch.float16)
    a0, b0, c0 = a.clone(), b.clone(), c.clone()
    kern(a, b, c)
    check_nan(c, name="dbg003_ones")
    cf = c.float().numpy()
    _out("[DBG 003ones] golden expect = 150")
    _out(f"[DBG 003ones] c min/max/mean = {cf.min():.4f}/{cf.max():.4f}/{cf.mean():.4f}  c[:6]={cf.reshape(-1)[:6]}")
    _out(f"[DBG 003ones] cbits[:6] = {c.reshape(-1)[:6].view(torch.int16).numpy()}")
    _out(
        f"[DBG 003ones] changed c={int((c != c0).sum())}/{c.numel()} a/b unchanged="
        f"{int((a != a0).sum())}/{int((b != b0).sum())}"
    )


if __name__ == "__main__":
    test_dbg002_rand()
    test_dbg002_ones()
    test_dbg003_ones()
