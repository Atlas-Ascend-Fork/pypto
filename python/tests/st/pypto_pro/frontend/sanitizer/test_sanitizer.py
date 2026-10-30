#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO Pro 自研日志回放 sanitizer v2 —— 检测 ST 用例。

覆盖：
  一、GM 越界（逻辑 shape，tensor 支持最多 5 维）—— GM_OUT_OF_BOUNDS
  二、tile 越界（valid_shape 超出 TileType 声明维度）—— TILE_OUT_OF_BOUNDS
  三、片上 tile 地址重叠                           —— TILE_OVERLAP
  四、mutex 配对（lock/unlock 按 region+id+pipe）  —— MUTEX_*

正样本保证不误报；负样本保证被检出；默认路径（sanitizer=False）行为不变。
"""

import logging
import os
import struct

import pypto_pro.language as pl
import pypto_pro.runtime.sanitizer_replay as replay_module
from pypto_pro.runtime.sanitizer_replay import (
    Detection,
    Finding,
    GmAccessRecord,
    MutexAccessRecord,
    PhaseAccessRecord,
    ReplayContext,
    SanitizerReplayError,
    SanitizerReportError,
    SyncAccessRecord,
    TileAccessRecord,
    check_gm_bounds,
    check_mutex_pairing,
    check_phase_state_machine,
    check_sync_pairing,
    check_tile_bounds,
    decode_regions,
    run_replay,
    scan_report_file,
)
import pytest
import torch

logging.basicConfig(level=logging.INFO)

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# Shared tile shapes for the phase/sync cases below.
M_DIM, N_DIM, K_DIM = 64, 64, 128
TILE_M, TILE_N, TILE_K = 64, 64, 64



def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _inputs(device, shape, dtype=torch.float16):
    torch.manual_seed(0)
    return torch.randn(shape, device=device, dtype=dtype)


# -----------------------------------------------------------------------------
# 一、GM 越界（逻辑 shape）
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def oob_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # ===== 错误注入点：行偏移 32 越界（32+64=96 > 逻辑 shape 64）→ GM_OUT_OF_BOUNDS =====
        pl.load(a, x, [32, 0])
        pl.store(z, a, [0, 0])


@pl.jit(sanitizer=True, auto_mutex=True)
def clean_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_gm_out_of_bounds_detected():
    """负样本：GM 访问超出逻辑 shape 应被回放检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_gm_out_of_bounds_detected OK")


@pytest.mark.soc("950")
def test_clean_no_false_positive():
    """正样本：合法访问不应有任何回放检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    clean_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x)
    logging.info("test_clean_no_false_positive OK")


# -----------------------------------------------------------------------------
# 二、片上 tile 地址重叠
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def overlap_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    b = pl.make_tile(tt, addr=0x0000)
    # ===== 错误注入点：c 的 addr=0x1000 落在 b 区间 [0x0, 0x2000) 内 → TILE_OVERLAP =====
    c = pl.make_tile(tt, addr=0x1000)
    with pl.section_vector():
        pl.load(b, x, [0, 0])
        pl.load(c, x, [0, 0])
        pl.store(z, b, [0, 0])


@pytest.mark.soc("950")
def test_tile_overlap_detected():
    """负样本：两个 tile 物理地址区间相交应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        overlap_kernel(x, z)
        torch.npu.synchronize()
    assert any("TILE_OVERLAP" in str(f) for f in exc_info.value.findings)
    logging.info("test_tile_overlap_detected OK")


# -----------------------------------------------------------------------------
# 三、tile 越界（valid_shape 超出 TileType 声明维度）
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def tile_ok_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_tile_ok_no_false_positive():
    """正样本：合法访问不误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    tile_ok_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x)
    logging.info("test_tile_ok_no_false_positive OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def tile_dim_oob_kernel(
    x: pl.Tensor[[TILE_M * 2, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M * 2, TILE_N], pl.DT_FP16],
    valid_rows: pl.DT_INT32,
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # 运行时传入 valid_rows=80，超过声明行数 64，由 sanitizer 回放检出。
        pl.set_validshape(a, [valid_rows, TILE_N // 2])
        pl.load(a, x, [0, 0])
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_tile_dim_oob_detected():
    """负样本：运行时有效窗口超出 TileType 声明维度应由回放检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M * 2, TILE_N])
    z = torch.zeros([TILE_M * 2, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        tile_dim_oob_kernel(x, z, TILE_M + 16)
        torch.npu.synchronize()
    assert any("TILE_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_tile_dim_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def tile_move_offset_oob_kernel(
    x: pl.Tensor[[TILE_M * 2, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M * 2, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    a = pl.make_tile(tt, addr=0x0000)
    b = pl.make_tile(tt, addr=0x2000)
    with pl.section_vector():
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        # ===== 错误注入点：move 目标偏移 32 + 窗口 64 = 96 > 声明行数 64 →
        # 运行时记录 offset+window，回放检出 TILE_OUT_OF_BOUNDS =====
        pl.move(b, a, offset=[32, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, b, [0, 0])


@pytest.mark.soc("950")
def test_tile_move_offset_oob_detected():
    """负样本：move 目标 offset+window 超出声明维度应被运行时回放检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M * 2, TILE_N])
    z = torch.zeros([TILE_M * 2, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        tile_move_offset_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("TILE_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_tile_move_offset_oob_detected OK")


# -----------------------------------------------------------------------------
# 三-c、tile 越界组合：逐维越界（纯逻辑验证，无需设备）
# -----------------------------------------------------------------------------


@pytest.mark.soc("950")
def test_tile_bounds_combined_check():
    """纯逻辑测试：验证 check_tile_bounds 逐维越界检测（含 offset）。"""
    ctx = ReplayContext(source_file="test.py")

    def rec(off_row, off_col, acc_row, acc_col, dim_row=64, dim_col=64):
        return TileAccessRecord(dim_row=dim_row, dim_col=dim_col,
                                off_row=off_row, off_col=off_col,
                                acc_row=acc_row, acc_col=acc_col, line=0)

    # 无 offset：窗口超出声明维度
    hits = check_tile_bounds([rec(0, 0, 80, 32)], ctx)
    assert len(hits) == 1, f"expected 1 finding, got {len(hits)}"
    assert "dim0: offset 0 + valid_shape 80 = 80 > 64 (over by 16)" in hits[0].message
    # 有 offset：offset+window 超出声明维度（核心回归场景）
    hits = check_tile_bounds([rec(32, 0, 64, 32)], ctx)
    assert len(hits) == 1, f"expected 1 finding, got {len(hits)}"
    assert "dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32)" in hits[0].message
    # 两维都越界
    hits2 = check_tile_bounds([rec(32, 16, 64, 64)], ctx)
    assert len(hits2) == 1, f"expected 1 finding, got {len(hits2)}"
    assert "dim0: offset 32 + valid_shape 64 = 96 > 64 (over by 32)" in hits2[0].message
    assert "dim1: offset 16 + valid_shape 64 = 80 > 64 (over by 16)" in hits2[0].message
    # 正样本：offset+window 在声明维度内
    hits3 = check_tile_bounds([rec(16, 0, 32, 32)], ctx)
    assert len(hits3) == 0, f"expected 0, got {len(hits3)}"
    # 按记录 dims 判定（无表）
    hits4 = check_tile_bounds([rec(32, 0, 64, 32)], ctx)
    assert len(hits4) == 1, f"expected 1 finding, got {len(hits4)}"
    # 动态负 offset 必须和 GM bounds 一样被拒绝。
    negative = check_tile_bounds([rec(-1, 0, 1, 32)], ctx)
    assert len(negative) == 1
    assert "negative offset -1" in negative[0].message


# -----------------------------------------------------------------------------
# 一-d、GM 窗口轴放置：win_ax 显式化（rank-1 乘积窗口 / 非尾部 order / 转置）
# -----------------------------------------------------------------------------


@pytest.mark.soc("950")
def test_gm_window_axis_placement():
    """纯逻辑测试：win_ax 驱动的窗口放置（rank-1 全 tile 投影、非尾部 order、转置交换）。"""
    ctx = ReplayContext(source_file="test.py")

    def rec(ndim, offs, dims, acc_row, acc_col, win_ax0, win_ax1):
        off5 = list(offs) + [0] * (5 - len(offs))
        dim5 = list(dims) + [0] * (5 - len(dims))
        return GmAccessRecord(ndim=ndim,
                              off0=off5[0], off1=off5[1], off2=off5[2], off3=off5[3], off4=off5[4],
                              dim0=dim5[0], dim1=dim5[1], dim2=dim5[2], dim3=dim5[3], dim4=dim5[4],
                              acc_row=acc_row, acc_col=acc_col, win_ax0=win_ax0, win_ax1=win_ax1, line=0)

    # rank-1 [N,1] 列向量：整 tile 投影（acc_row 为乘积窗口），修复前曾漏检
    hits = check_gm_bounds([rec(1, [64], [100], 64, 1, 0, -1)], ctx)
    assert len(hits) == 1, f"expected 1 finding, got {len(hits)}"
    assert "dim0: offset 64 + valid_shape 64 = 128 > 100 (over by 28)" in hits[0].message
    # rank-1 [1,N] 行向量：同样的乘积窗口
    hits = check_gm_bounds([rec(1, [64], [100], 64, 64, 0, -1)], ctx)
    assert len(hits) == 1, f"expected 1 finding, got {len(hits)}"
    # rank-1 边界贴合：36 + 64 = 100 ≤ 100，不报
    hits = check_gm_bounds([rec(1, [36], [100], 64, 1, 0, -1)], ctx)
    assert len(hits) == 0, f"expected 0, got {len(hits)}"
    # 非尾部 order=[1,3]：窗口贴轴 1/轴 3，其余轴 offset-only
    hits = check_gm_bounds([rec(4, [1, 48, 3, 0], [2, 64, 8, 128], 64, 128, 1, 3)], ctx)
    assert len(hits) == 1, f"expected 1 finding, got {len(hits)}"
    assert "dim1: offset 48 + valid_shape 64 = 112 > 64 (over by 48)" in hits[0].message
    # 转置 order=[1,0]：win_ax 交换，tile 第 0 维窗口（acc_row）贴轴 1
    hits = check_gm_bounds([rec(2, [0, 64], [128, 96], 64, 128, 1, 0)], ctx)
    assert len(hits) == 1, f"expected 1 finding, got {len(hits)}"
    assert "dim1: offset 64 + valid_shape 64 = 128 > 96 (over by 32)" in hits[0].message
    # 非尾部 order 干净访问：不报
    hits = check_gm_bounds([rec(4, [1, 0, 3, 0], [2, 64, 8, 128], 64, 128, 1, 3)], ctx)
    assert len(hits) == 0, f"expected 0, got {len(hits)}"


R1_LEN = 100
R1_TILE = 64


@pl.jit(sanitizer=True, auto_mutex=True)
def rank1_col_oob_kernel(
    x: pl.Tensor[[R1_LEN], pl.DT_FP16],
    z: pl.Tensor[[R1_LEN], pl.DT_FP16],
):
    tt = pl.TileType(shape=[R1_TILE, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # ===== 错误注入点：offset 64 + 64 = 128 > 100，[N,1] 列向量（修复前漏检）=====
        pl.load(a, x, [64])
        pl.store(z, a, [0])


@pl.jit(sanitizer=True, auto_mutex=True)
def rank1_row_oob_kernel(
    x: pl.Tensor[[R1_LEN], pl.DT_FP16],
    z: pl.Tensor[[R1_LEN], pl.DT_FP16],
):
    tt = pl.TileType(shape=[1, R1_TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # ===== 错误注入点：同样访问 64..127，[1,N] 行向量 =====
        pl.load(a, x, [64])
        pl.store(z, a, [0])


@pl.jit(sanitizer=True, auto_mutex=True)
def rank1_clean_kernel(
    x: pl.Tensor[[R1_LEN], pl.DT_FP16],
    z: pl.Tensor[[R1_LEN], pl.DT_FP16],
):
    tt = pl.TileType(shape=[R1_TILE, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # 边界贴合：36 + 64 = 100 = 100
        pl.load(a, x, [36])
        pl.store(z, a, [0])


@pytest.mark.soc("950")
def test_rank1_column_tile_oob_detected():
    """负样本：rank-1 列向量 tile 越界应被检出（win_ax 乘积窗口修复）。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [R1_LEN])
    z = torch.zeros([R1_LEN], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        rank1_col_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    assert any("over by 28" in str(f) for f in exc_info.value.findings)
    logging.info("test_rank1_column_tile_oob_detected OK")


@pytest.mark.soc("950")
def test_rank1_row_tile_oob_detected():
    """负样本：rank-1 行向量 tile 越界应被检出（回归）。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [R1_LEN])
    z = torch.zeros([R1_LEN], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        rank1_row_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    assert any("over by 28" in str(f) for f in exc_info.value.findings)
    logging.info("test_rank1_row_tile_oob_detected OK")


@pytest.mark.soc("950")
def test_rank1_tile_clean():
    """正样本：rank-1 边界贴合访问不应误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [R1_LEN])
    z = torch.zeros([R1_LEN], device=ST_DEVICE, dtype=torch.float16)
    rank1_clean_kernel(x, z)
    torch.npu.synchronize()
    logging.info("test_rank1_tile_clean OK")


# -----------------------------------------------------------------------------
# 一-e、非尾部 order：GM 窗口贴到非最内两维
# -----------------------------------------------------------------------------

O4 = (2, 64, 8, 128)  # [B, S, N, D]，order=[1, 3] 映射 S、D


@pl.jit(sanitizer=True, auto_mutex=True)
def order_nontrailing_oob_kernel(
    x: pl.Tensor[list(O4), pl.DT_FP16],
    z: pl.Tensor[list(O4), pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # ===== 错误注入点：S 轴 48 + 64 = 112 > 64（修复前该形态不插桩）=====
        pl.load(a, x, [1, 48, 3, 0], order=[1, 3])
        pl.store(z, a, [1, 0, 3, 0], order=[1, 3])


@pl.jit(sanitizer=True, auto_mutex=True)
def order_nontrailing_clean_kernel(
    x: pl.Tensor[list(O4), pl.DT_FP16],
    z: pl.Tensor[list(O4), pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [1, 0, 3, 0], order=[1, 3])
        pl.store(z, a, [1, 0, 3, 0], order=[1, 3])


@pytest.mark.soc("950")
def test_order_nontrailing_oob_detected():
    """负样本：非尾部 order 的窗口越界（S 轴）应被检出（win_ax 修复）。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, list(O4))
    z = torch.zeros(list(O4), device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        order_nontrailing_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    assert any("dim1" in str(f) for f in exc_info.value.findings)
    logging.info("test_order_nontrailing_oob_detected OK")


@pytest.mark.soc("950")
def test_order_nontrailing_clean():
    """正样本：非尾部 order 干净访问不应误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, list(O4))
    z = torch.zeros(list(O4), device=ST_DEVICE, dtype=torch.float16)
    order_nontrailing_clean_kernel(x, z)
    torch.npu.synchronize()
    logging.info("test_order_nontrailing_clean OK")


# -----------------------------------------------------------------------------
# 一-f、转置 load：win_ax 交换语义（tile 第 0 维窗口贴 Tensor 第 1 维）
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def transpose_oob_kernel(
    x: pl.Tensor[[128, 96], pl.DT_FP16],
):
    # [K=128, M=96] 转置搬入 L1：tile [M, K] = [64, 128]，order=[1, 0]
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_cube():
        # ===== 错误注入点：M 轴（dim1）偏移 64 + 64 = 128 > 96 =====
        pl.load(a, x, [0, 64], order=[1, 0])


@pytest.mark.soc("950")
def test_transpose_window_axes_detected():
    """负样本：转置 load 的窗口贴轴交换后，M 轴越界应按 acc_row 窗口检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 96])
    with pytest.raises(SanitizerReplayError) as exc_info:
        transpose_oob_kernel(x)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    assert any("dim1: offset 64 + valid_shape 64 = 128 > 96 (over by 32)" in str(f)
               for f in exc_info.value.findings)
    logging.info("test_transpose_window_axes_detected OK")


# -----------------------------------------------------------------------------
# 五维 tensor：GM 越界逐维判定（最多 5 维）
# -----------------------------------------------------------------------------

D5 = (2, 3, 4, TILE_M, TILE_N)


@pl.jit(sanitizer=True, auto_mutex=True)
def gm_5d_clean_kernel(
    x: pl.Tensor[[D5[0], D5[1], D5[2], D5[3], D5[4]], pl.DT_FP16],
    z: pl.Tensor[[D5[0], D5[1], D5[2], D5[3], D5[4]], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        for i0 in pl.range(0, D5[0]):
            for i1 in pl.range(0, D5[1]):
                for i2 in pl.range(0, D5[2]):
                    pl.load(a, x, [i0, i1, i2, 0, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.store(z, a, [i0, i1, i2, 0, 0])


@pl.jit(sanitizer=True, auto_mutex=True)
def gm_5d_oob_kernel(
    x: pl.Tensor[[D5[0], D5[1], D5[2], D5[3], D5[4]], pl.DT_FP16],
    z: pl.Tensor[[D5[0], D5[1], D5[2], D5[3], D5[4]], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        for i0 in pl.range(0, D5[0]):
            for i1 in pl.range(0, D5[1]):
                for i2 in pl.range(0, D5[2]):
                    # ===== 错误注入点：行偏移 +32，仅第 4 维越界（32+64>64）=====
                    pl.load(a, x, [i0, i1, i2, 32, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                    pl.store(z, a, [i0, i1, i2, 0, 0])


@pytest.mark.soc("950")
def test_gm_5d_oob_detected():
    """负样本：5 维 tensor 第 4 维偏移越界应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, list(D5))
    z = torch.zeros(list(D5), device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        gm_5d_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_gm_5d_oob_detected OK")


# -----------------------------------------------------------------------------
# 默认路径（sanitizer=False）不受影响
# -----------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def no_sanitizer_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_no_sanitizer_unchanged():
    """默认不开 sanitizer：kernel 正常执行且不产生回放报告。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    no_sanitizer_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x)
    logging.info("test_no_sanitizer_unchanged OK")


# -----------------------------------------------------------------------------
# 循环多轮不覆盖：前轮越界、后续轮合法也必须检出
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def loop_mid_oob_kernel(
    x: pl.Tensor[[TILE_M * 2 - 32, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M * 2 - 32, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        for i in pl.range(0, 2, 1):
            # ===== 错误注入点：i=1 时行偏移 64+64=128 > 逻辑 shape 96 → GM_OUT_OF_BOUNDS
            # （中间轮越界，末轮(不存在)不覆盖；验证循环内每轮记录均保留）=====
            pl.load(a, x, [i * TILE_M, 0])
            pl.store(z, a, [i * TILE_M, 0])


@pl.jit(sanitizer=True, auto_mutex=True)
def loop_ok_kernel(
    x: pl.Tensor[[TILE_M * 2, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M * 2, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        for i in pl.range(0, 2, 1):
            pl.load(a, x, [i * TILE_M, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(z, a, [i * TILE_M, 0])
            pl.system.bar_all()


@pytest.mark.soc("950")
def test_loop_mid_iteration_oob_detected():
    """循环中间轮越界必须被检出（记录追加，不被末轮覆盖）。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M * 2 - 32, TILE_N])
    z = torch.zeros([TILE_M * 2 - 32, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        loop_mid_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_loop_mid_iteration_oob_detected OK")


@pytest.mark.soc("950")
def test_loop_ok_no_false_positive():
    """合法循环（每轮读写各自区间）不应误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M * 2, TILE_N])
    z = torch.zeros([TILE_M * 2, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    loop_ok_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x)
    logging.info("test_loop_ok_no_false_positive OK")


# -----------------------------------------------------------------------------
# 复杂用例：多 if 分支 + 多层循环 + 多核 + 动态 shape（验证 GM/tile 检测稳定）
# -----------------------------------------------------------------------------

MM_TILE_M, MM_TILE_N = 256, 256
MM_KL0, MM_KL1 = 64, 128
MM_MAIN_WINDOW = 4  # ASW: #M-tiles 必须是其倍数
MM_L0_BASE = 0x0000
MM_A_L1_ADDR, MM_B_L1_BASE = 0x00000, 0x40000
MM_ROWS, MM_COLS, MM_K = 800, 304, 128  # N 尾块 48、M 尾块 32


@pl.jit(sanitizer=True, auto_mutex=True)
def mm_complex_clean_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """ASW matmul：round(多核) × ko(K 宽块) × ki(K 尾子块) 3 层循环 + 4 嵌套 if/else。"""
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    with pl.section_cube():
        a_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[MM_TILE_M, MM_KL1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1]),
            addrs=MM_A_L1_ADDR, mutex_ids=[0, 1, 10, 11],
        )
        b_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[MM_KL1, MM_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1]),
            addrs=MM_B_L1_BASE, mutex_ids=[2, 3, 12, 13],
        )
        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[MM_TILE_M, MM_KL0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, valid_shape=[-1, -1]),
            addrs=MM_L0_BASE, mutex_ids=[4, 5],
        )
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[MM_KL0, MM_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right,
                             valid_shape=[-1, -1]),
            addrs=MM_L0_BASE, mutex_ids=[6, 7],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[MM_TILE_M, MM_TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             fractal=1024, valid_shape=[-1, -1]),
            addrs=MM_L0_BASE, mutex_ids=[8],
        )
        m_tiles = (m + MM_TILE_M - 1) // MM_TILE_M
        n_tiles = (n + MM_TILE_N - 1) // MM_TILE_N
        total_cnt = m_tiles * n_tiles
        k_outer = (k + MM_KL1 - 1) // MM_KL1
        k_subs = (k + MM_KL0 - 1) // MM_KL0
        last_sub = k_subs - 1
        round_cnt = (total_cnt + num_cores - 1) // num_cores
        for r in pl.range(0, round_cnt):
            index = core_id + r * num_cores
            if index < total_cnt:
                row_idx = index // n_tiles // MM_MAIN_WINDOW
                mi = row_idx * MM_MAIN_WINDOW + index % MM_MAIN_WINDOW
                ni_normal = (index // MM_MAIN_WINDOW) % n_tiles
                parity = row_idx % 2  # 蛇形 N 走位（无分支写法）
                ni = ni_normal + parity * (n_tiles - 1 - 2 * ni_normal)
                i = mi * MM_TILE_M
                j = ni * MM_TILE_N
                valid_m = pl.min(MM_TILE_M, m - i)
                valid_n = pl.min(MM_TILE_N, n - j)
                for ko in pl.range(0, k_outer):
                    k_off = ko * MM_KL1
                    valid_k1 = pl.min(MM_KL1, k - k_off)
                    a_wide_slot = a_l1_wide.next()
                    pl.set_validshape(a_wide_slot, [valid_m, valid_k1])
                    pl.load(a_wide_slot, a, [i, k_off])
                    b_wide_slot = b_l1_wide.next()
                    pl.set_validshape(b_wide_slot, [valid_k1, valid_n])
                    pl.load(b_wide_slot, b, [k_off, j])
                    n_sub = (valid_k1 + MM_KL0 - 1) // MM_KL0  # 运行时 1 或 2 次
                    for ki in pl.range(0, n_sub):
                        sub_off = ki * MM_KL0
                        sub_k = pl.min(MM_KL0, valid_k1 - sub_off)
                        gsub = ko * (MM_KL1 // MM_KL0) + ki
                        pl.set_validshape(a_wide_slot, [valid_m, sub_k])
                        cur_a_left = a_left_db.next()
                        pl.set_validshape(cur_a_left, [valid_m, sub_k])
                        pl.move(cur_a_left, a_wide_slot, offset=[0, sub_off])
                        pl.set_validshape(b_wide_slot, [sub_k, MM_TILE_N])
                        cur_b_right = b_right_db.next()
                        pl.set_validshape(cur_b_right, [sub_k, MM_TILE_N])
                        pl.move(cur_b_right, b_wide_slot, offset=[sub_off, 0])
                        if gsub == 0:
                            if gsub == last_sub:
                                pl.matmul(acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Final)
                            else:
                                pl.matmul(acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial)
                        else:
                            if gsub == last_sub:
                                pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right,
                                              phase=pl.AccPhase.Final)
                            else:
                                pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right,
                                              phase=pl.AccPhase.Partial)
                pl.set_validshape(acc.current(), [valid_m, valid_n])
                pl.store(out, acc.current(), [i, j], phase=pl.STPhase.Final)


@pl.jit(sanitizer=True, auto_mutex=True)
def mm_complex_oob_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """与 clean 完全一致，仅注入：K 宽块 load b 的列偏移 +8 → 仅 N 尾块越界。"""
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    with pl.section_cube():
        a_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[MM_TILE_M, MM_KL1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1]),
            addrs=MM_A_L1_ADDR, mutex_ids=[0, 1, 10, 11],
        )
        b_l1_wide = pl.make_tile_group(
            type=pl.TileType(shape=[MM_KL1, MM_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
                             valid_shape=[-1, -1]),
            addrs=MM_B_L1_BASE, mutex_ids=[2, 3, 12, 13],
        )
        a_left_db = pl.make_tile_group(
            type=pl.TileType(shape=[MM_TILE_M, MM_KL0], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ, valid_shape=[-1, -1]),
            addrs=MM_L0_BASE, mutex_ids=[4, 5],
        )
        b_right_db = pl.make_tile_group(
            type=pl.TileType(shape=[MM_KL0, MM_TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right,
                             valid_shape=[-1, -1]),
            addrs=MM_L0_BASE, mutex_ids=[6, 7],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(shape=[MM_TILE_M, MM_TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             fractal=1024, valid_shape=[-1, -1]),
            addrs=MM_L0_BASE, mutex_ids=[8],
        )
        m_tiles = (m + MM_TILE_M - 1) // MM_TILE_M
        n_tiles = (n + MM_TILE_N - 1) // MM_TILE_N
        total_cnt = m_tiles * n_tiles
        k_outer = (k + MM_KL1 - 1) // MM_KL1
        k_subs = (k + MM_KL0 - 1) // MM_KL0
        last_sub = k_subs - 1
        round_cnt = (total_cnt + num_cores - 1) // num_cores
        for r in pl.range(0, round_cnt):
            index = core_id + r * num_cores
            if index < total_cnt:
                row_idx = index // n_tiles // MM_MAIN_WINDOW
                mi = row_idx * MM_MAIN_WINDOW + index % MM_MAIN_WINDOW
                ni_normal = (index // MM_MAIN_WINDOW) % n_tiles
                parity = row_idx % 2
                ni = ni_normal + parity * (n_tiles - 1 - 2 * ni_normal)
                i = mi * MM_TILE_M
                j = ni * MM_TILE_N
                valid_m = pl.min(MM_TILE_M, m - i)
                valid_n = pl.min(MM_TILE_N, n - j)
                for ko in pl.range(0, k_outer):
                    k_off = ko * MM_KL1
                    valid_k1 = pl.min(MM_KL1, k - k_off)
                    a_wide_slot = a_l1_wide.next()
                    pl.set_validshape(a_wide_slot, [valid_m, valid_k1])
                    pl.load(a_wide_slot, a, [i, k_off])
                    b_wide_slot = b_l1_wide.next()
                    pl.set_validshape(b_wide_slot, [valid_k1, valid_n])
                    # ================= 错误注入点 1 =================
                    # 与 clean 版本唯一区别：列偏移 +8。b 形状 [K=128, N=304]，
                    # 仅 N 尾块（j=256, valid_n=48）时 256+8+48=312 > 304 → GM_OUT_OF_BOUNDS
                    pl.load(b_wide_slot, b, [k_off, j + 8])
                    n_sub = (valid_k1 + MM_KL0 - 1) // MM_KL0
                    for ki in pl.range(0, n_sub):
                        sub_off = ki * MM_KL0
                        sub_k = pl.min(MM_KL0, valid_k1 - sub_off)
                        gsub = ko * (MM_KL1 // MM_KL0) + ki
                        pl.set_validshape(a_wide_slot, [MM_TILE_M, sub_k])
                        cur_a_left = a_left_db.next()
                        pl.move(cur_a_left, a_wide_slot, offset=[0, sub_off])
                        pl.set_validshape(cur_a_left, [MM_TILE_M, sub_k])
                        pl.set_validshape(b_wide_slot, [sub_k, MM_TILE_N])
                        cur_b_right = b_right_db.next()
                        pl.move(cur_b_right, b_wide_slot, offset=[sub_off, 0])
                        pl.set_validshape(cur_b_right, [sub_k, MM_TILE_N])
                        if gsub == 0:
                            if gsub == last_sub:
                                pl.matmul(acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Final)
                            else:
                                pl.matmul(acc.current(), cur_a_left, cur_b_right, phase=pl.AccPhase.Partial)
                        else:
                            if gsub == last_sub:
                                pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right,
                                              phase=pl.AccPhase.Final)
                            else:
                                pl.matmul_acc(acc.current(), acc.current(), cur_a_left, cur_b_right,
                                              phase=pl.AccPhase.Partial)
                pl.set_validshape(acc.current(), [valid_m, valid_n])
                pl.store(out, acc.current(), [i, j], phase=pl.STPhase.Final)


def _mm_inputs(device):
    torch.manual_seed(42)
    a = torch.randn(MM_ROWS, MM_K, device=device, dtype=torch.float16)
    b = torch.randn(MM_K, MM_COLS, device=device, dtype=torch.float16)
    return a, b


@pytest.mark.soc("950")
def test_mm_complex_clean_no_false_positive():
    """正样本：3 层循环 + 4 嵌套分支 + 8 核 ASW matmul，不误报且结果正确。"""
    _require_a5(ST_DEVICE)
    a, b = _mm_inputs(ST_DEVICE)
    out = torch.zeros([MM_ROWS, MM_COLS], device=ST_DEVICE, dtype=torch.float16)
    mm_complex_clean_kernel[None, 8](a, b, out)
    torch.npu.synchronize()
    golden = torch.matmul(a.float(), b.float()).half()
    torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
    logging.info("test_mm_complex_clean_no_false_positive OK")


@pytest.mark.soc("950")
def test_mm_complex_oob_detected():
    """负样本：K 宽块 load 列偏移 +8，仅 N 尾块（j=256, valid_n=48）越界。"""
    _require_a5(ST_DEVICE)
    a, b = _mm_inputs(ST_DEVICE)
    out = torch.zeros([MM_ROWS, MM_COLS], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        mm_complex_oob_kernel[None, 8](a, b, out)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_mm_complex_oob_detected OK")


# -----------------------------------------------------------------------------
# 四、mutex 配对检测
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=False)
def mutex_ok_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(a, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=1)
        pl.store(z, a, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=1)


@pl.jit(sanitizer=True, auto_mutex=False)
def mutex_unlock_before_lock_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # ===== 错误注入点：unlock 无前置 lock =====
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=7)
        pl.load(a, x, [0, 0])
        pl.store(z, a, [0, 0])


@pl.jit(sanitizer=True, auto_mutex=False)
def mutex_unpaired_lock_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # ===== 错误注入点：lock 后无 unlock =====
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=6)
        pl.load(a, x, [0, 0])
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_mutex_ok_no_false_positive():
    """正样本：mutex lock/unlock 正确配对，不误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    mutex_ok_kernel(x, z)
    torch.npu.synchronize()
    logging.info("test_mutex_ok_no_false_positive OK")


@pytest.mark.soc("950")
def test_mutex_unlock_before_lock_detected():
    """负样本：unlock 无前置 lock 应检出 MUTEX_UNLOCK_BEFORE_LOCK。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        mutex_unlock_before_lock_kernel(x, z)
        torch.npu.synchronize()
    assert any("MUTEX_UNLOCK_BEFORE_LOCK" in str(f) for f in exc_info.value.findings)
    logging.info("test_mutex_unlock_before_lock_detected OK")


@pytest.mark.soc("950")
def test_mutex_unpaired_lock_detected():
    """负样本：lock 无配对 unlock 应检出 UNPAIRED_MUTEX_LOCK。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        mutex_unpaired_lock_kernel(x, z)
        torch.npu.synchronize()
    assert any("UNPAIRED_MUTEX_LOCK" in str(f) for f in exc_info.value.findings)
    logging.info("test_mutex_unpaired_lock_detected OK")


def _mr(is_lock, mutex_id=0, pipe=1, line=10, region=0):
    return MutexAccessRecord(mutex_id=mutex_id, pipe=pipe, is_lock=is_lock,
                             line=line, region=region)


@pytest.mark.soc("950")
def test_mutex_double_lock_detected_without_launch():
    """重复 lock 直接报根因，避免用真实挂死样本验证。"""
    findings = check_mutex_pairing(
        [_mr(1, line=10), _mr(1, line=11), _mr(0, line=12), _mr(0, line=13)],
        ReplayContext("test.py"),
    )
    assert [f.kind for f in findings] == ["MUTEX_DOUBLE_LOCK"]
    assert findings[0].location.endswith(":11")


@pytest.mark.soc("950")
def test_decoder_rejects_truncated_region():
    with pytest.raises(SanitizerReportError, match="counter claims"):
        decode_regions([struct.pack("<I", 1)])


@pytest.mark.soc("950")
def test_decoder_rejects_unknown_record():
    unknown_head = (0x5A5A << 16) | 0xFFFF
    with pytest.raises(SanitizerReportError, match="unknown sanitizer record"):
        decode_regions([struct.pack("<II", 1, unknown_head)])


@pytest.mark.soc("950")
def test_scan_report_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(SanitizerReportError, match="region count"):
        scan_report_file(str(path), "test.py")


@pytest.mark.soc("950")
def test_scan_report_rejects_trailing_bytes(tmp_path):
    path = tmp_path / "trailing.bin"
    path.write_bytes(struct.pack("<I", 0) + b"\x00")
    with pytest.raises(SanitizerReportError, match="trailing bytes"):
        scan_report_file(str(path), "test.py")


# -----------------------------------------------------------------------------
# 八、跨程序 tile 登记（顶层 make_tile 同时进入 Cube/Vector 两个 program）
# -----------------------------------------------------------------------------

CV_M, CV_K, CV_N = 64, 64, 64
CV_VEC_ROWS = 32


@pl.jit(sanitizer=True, auto_mutex=True)
def cv_top_tiles_clean_kernel(
    a: pl.Tensor[[CV_M, CV_N], pl.DT_FP32],
    b: pl.Tensor[[CV_M, CV_K], pl.DT_FP16],
    c: pl.Tensor[[CV_K, CV_N], pl.DT_FP16],
    out: pl.Tensor[[CV_M, CV_N], pl.DT_FP32],
):
    """Cube+Vector 混合 kernel，全部 tile 声明在两个 section 之外的顶层。

    前端按 target 各解析一遍整个函数，顶层 make_tile(_group) 同时进入 Cube
    与 Vector 两个 program，sanitizer 用同一个 meta 跑两遍——同名同物理信息
    必须复用已有 slot 只登记一次，否则静态 TILE_OVERLAP 扫描会把同一逻辑
    tile 的两条登记判为地址重叠（tile 自重叠误报）。
    """
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[[0, 1]],
    )
    c_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids=[[2, 3]],
    )
    b_left = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4],
    )
    c_right = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    tile_acc = pl.make_tile(
        pl.TileType(shape=[CV_M, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=0x0000,
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[CV_VEC_ROWS, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[[6, 7], [8, 9], [10, 11], [12, 13]],
    )
    with pl.section_cube():
        pl.load(b_l1[0], b, [0, 0])
        pl.load(c_l1[0], c, [0, 0])
        pl.move(b_left[0], b_l1[0])
        pl.move(c_right[0], c_l1[0])
        pl.matmul(tile_acc, b_left[0], c_right[0])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(vec_group[0], tile_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        row_offset = sub_index * CV_VEC_ROWS
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.load(vec_group[1], a, [row_offset, 0])
        pl.add(vec_group[2], vec_group[1], vec_group[0])
        pl.move(vec_group[3], vec_group[2])
        pl.store(out, vec_group[3], [row_offset, 0])


@pytest.mark.soc("950")
def test_cv_top_level_tiles_no_self_overlap():
    """正样本：顶层 tile 声明进入两个 program 只登记一次，不得误报自重叠。"""
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([CV_M, CV_N], device=ST_DEVICE, dtype=torch.float32) * 2.0 - 1.0
    b = torch.rand([CV_M, CV_K], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    c = torch.rand([CV_K, CV_N], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([CV_M, CV_N], device=ST_DEVICE, dtype=torch.float32)
    cv_top_tiles_clean_kernel(a, b, c, out)
    torch.npu.synchronize()
    golden = a.cpu().float() + (b.cpu().float() @ c.cpu().float())
    torch.testing.assert_close(out.cpu().float(), golden, rtol=3e-3, atol=3e-3)
    logging.info("test_cv_top_level_tiles_no_self_overlap OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def cv_cube_oob_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    """混合 kernel 的 Cube 段注入 GM 越界（Vector 段干净）。

    region 冲突回归（#15）：修复前 Cube 与 Vector subblock-0 同写 region 0，
    Cube 记录被 Vector 覆盖 → 越界漏检。修复后 Cube 独占 region [0, N)，
    Vector 偏移到 [N, 3N)，记录互不干扰 → 检出。
    """
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000,
        mutex_ids=[[0, 1]],
    )
    c_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000,
        mutex_ids=[[2, 3]],
    )
    b_left = pl.make_tile_group(
        type=pl.TileType(shape=[CV_M, CV_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[4],
    )
    c_right = pl.make_tile_group(
        type=pl.TileType(shape=[CV_K, CV_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[5],
    )
    tile_acc = pl.make_tile(
        pl.TileType(shape=[CV_M, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=0x0000,
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[CV_VEC_ROWS, CV_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[[6, 7], [8, 9], [10, 11], [12, 13]],
    )
    with pl.section_cube():
        # ===== 错误注入点：load b 的行偏移 = b.shape[0]（运行时 64）越界 =====
        pl.load(b_l1[0], b, [b.shape[0], 0])
        pl.load(c_l1[0], c, [0, 0])
        pl.move(b_left[0], b_l1[0])
        pl.move(c_right[0], c_l1[0])
        pl.matmul(tile_acc, b_left[0], c_right[0])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(vec_group[0], tile_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        row_offset = sub_index * CV_VEC_ROWS
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.load(vec_group[1], a, [row_offset, 0])
        pl.add(vec_group[2], vec_group[1], vec_group[0])
        pl.move(vec_group[3], vec_group[2])
        pl.store(out, vec_group[3], [row_offset, 0])


@pytest.mark.soc("950")
def test_cv_cube_oob_detected():
    """负样本：混合 kernel 中 Cube 段的 GM 越界必须被检出（不被 Vector 记录覆盖）。"""
    _require_a5(ST_DEVICE)
    torch.manual_seed(0)
    a = torch.rand([CV_M, CV_N], device=ST_DEVICE, dtype=torch.float32) * 2.0 - 1.0
    b = torch.rand([CV_M, CV_K], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    c = torch.rand([CV_K, CV_N], device=ST_DEVICE, dtype=torch.float16) * 2.0 - 1.0
    out = torch.zeros([CV_M, CV_N], device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(SanitizerReplayError) as exc_info:
        cv_cube_oob_kernel(a, b, c, out)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_cv_cube_oob_detected OK")


# -----------------------------------------------------------------------------
# 九、形态扩展回归（转置 load / 动态 validshape / 标量访问 / group 下标）
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def transpose_load_clean_kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP16],
):
    """转置搬入：tile [128,64](DN) ← tensor [64,128](ND)，order=[1,0]。

    窗口到 tensor 维度的映射随 order 交换（tile 行 ↔ tensor dim1）。
    修复前窗口按默认最内两维硬套 → dim0 判 0+128 > 64 → 误报。
    """
    tt = pl.TileType(shape=[128, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
                     layout=pl.DN)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [0, 0], order=[1, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)


@pytest.mark.soc("950")
def test_transpose_load_clean_no_false_positive():
    """正样本：非方阵转置 load 不得误报 GM_OUT_OF_BOUNDS。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 128])
    transpose_load_clean_kernel(x)
    torch.npu.synchronize()
    logging.info("test_transpose_load_clean_no_false_positive OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def dynamic_validshape_oob_kernel(
    x: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, 64], pl.DT_FP16],
):
    """动态 set_validshape 窗口超限：valid_m = 128//2+1 = 65 > tile 声明 64。

    GM 侧 off+窗口（0+65 ≤ 128）合法不报；tile 侧窗口 65 > 64 应检出。
    修复前动态窗口无任何记录 → 漏检。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        valid_m = x.shape[0] // 2 + 1  # 运行时 65，忘写 min 封顶
        # ===== 错误注入点：动态窗口 65 超出 tile 声明行数 64 =====
        pl.set_validshape(a, [valid_m, 64])
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_dynamic_validshape_oob_detected():
    """负样本：动态 set_validshape 窗口超出 tile 声明维度应被运行时检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 64])
    z = torch.zeros([128, 64], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        dynamic_validshape_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("TILE_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_dynamic_validshape_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def setval_linear_oob_kernel(
    x: pl.Tensor[[64], pl.DT_INT32],
):
    """setval 线性偏移越界：tensor 64 元素，写第 64 个（末尾后 1）。

    修复前 getval/setval 无任何检测 → 漏检。
    """
    with pl.section_vector():
        # ===== 错误注入点：线性偏移 64 = 元素总数，越界 1 个 =====
        pl.setval(x, 64, 1)


@pytest.mark.soc("950")
def test_setval_linear_oob_detected():
    """负样本：setval 线性偏移超出 tensor 元素数应被检出。"""
    _require_a5(ST_DEVICE)
    x = torch.zeros([64], device=ST_DEVICE, dtype=torch.int32)
    with pytest.raises(SanitizerReplayError) as exc_info:
        setval_linear_oob_kernel(x)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_setval_linear_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def group_move_offset_oob_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    z: pl.Tensor[[128, 64], pl.DT_FP16],
):
    """tile group 静态下标引用的 move 越界：g[1] 窗口 64 + 偏移 32 > 声明 64。

    group slot 在 IR 层是 GetItemExpr(tiles, i) 而非 make_tile 的 Var。
    修复前 TileId 无法解析 → 记录缺失 → 漏检。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    g = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0, 1])
    with pl.section_vector():
        pl.load(g[0], x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        # ===== 错误注入点：g[1] 目标偏移 32 + 窗口 64 = 96 > 声明 64 =====
        pl.move(g[1], g[0], offset=[32, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, g[1], [0, 0])


@pytest.mark.soc("950")
def test_group_move_offset_oob_detected():
    """负样本：group 静态下标 move 的 offset+窗口超限应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 64])
    z = torch.zeros([128, 64], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        group_move_offset_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("TILE_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_group_move_offset_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def cursor_move_offset_oob_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    z: pl.Tensor[[128, 64], pl.DT_FP16],
):
    """tile group 游标（next()）引用的 move 越界。

    nxt = g.next() 的运行时 slot 编译期不可解析（#14 动态部分）——修复前
    记录缺失漏检。修复后 dims 来自访问点类型，游标也能出记录判定。
    注入：src 从行偏移 32 提取 64 行 → 32+64 = 96 > 声明 64。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    g = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0, 1])
    with pl.section_vector():
        cur = g.current()
        pl.load(cur, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        # ===== 错误注入点：从 src(cur) 的行偏移 32 提取 64 行 → 96 > 64 =====
        nxt = g.next()
        pl.move(nxt, cur, offset=[32, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, nxt, [0, 0])


@pytest.mark.soc("950")
def test_cursor_move_offset_oob_detected():
    """负样本：group 游标 next() 的 move 偏移越界应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 64])
    z = torch.zeros([128, 64], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        cursor_move_offset_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("TILE_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_cursor_move_offset_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def move_asymmetric_clean_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP16],
):
    """非对称规格 move 正样本（offset 归属修正回归）。

    move 的 offset 是 src 提取偏移、传输窗口是 dst 的 valid_shape（TEXTRACT
    语义）。src [64,64]、dst [32,64]、offset [32,0]、dst 窗口 [32,64]：
    正确判定 src 32+32=64 ≤ 64、dst 0+32=32 ≤ 32 → 干净。
    归属修正前（offset 记在 dst）：dst 32+32=64 > 32 → 误报。
    """
    tt_src = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tt_src, addr=0x0000)
    dst = pl.make_tile(tt_dst, addr=0x4000)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.set_validshape(dst, [32, 64])
        pl.move(dst, src, offset=[32, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, dst, [0, 0])


@pytest.mark.soc("950")
def test_move_asymmetric_clean_no_false_positive():
    """正样本：非对称 tile 的 move 不得因 offset 归属错误而误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([64, 64], device=ST_DEVICE, dtype=torch.float16)
    move_asymmetric_clean_kernel(x, z)
    torch.npu.synchronize()
    logging.info("test_move_asymmetric_clean_no_false_positive OK")


# -----------------------------------------------------------------------------
# 十、kernel 内创建的 tensor（make_tensor / 别名）——committer 意见 11 修复回归
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def make_tensor_oob_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """kernel 内 make_tensor 创建的视图越界。

    t2 是 ptr.make_tensor 的产物（非 launch 实参）——修复前无 tensor_id，
    记录被跳过漏检；修复后视图 shape [64,64] 进入 shapes 表，检出。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [64, 64])
        # ===== 错误注入点：行偏移 = x.shape[0]（运行时 64）+ 窗口 64 > 视图声明 64 =====
        pl.load(a, t2, [x.shape[0], 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_make_tensor_oob_detected():
    """负样本：kernel 内 make_tensor 视图的越界访问应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([64, 64], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        make_tensor_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_make_tensor_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def tensor_alias_oob_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """tensor 别名（t3 = x）后的越界访问。

    修复前 t3 查不到 tensor_id（非形参名）→ 漏检；修复后别名链解析回 x。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t3 = x
        # ===== 错误注入点：经别名访问，行偏移 = x.shape[0]（运行时 64）+ 窗口 64 > 64 =====
        pl.load(a, t3, [x.shape[0], 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_tensor_alias_oob_detected():
    """负样本：别名 tensor 的越界访问应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([64, 64], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        tensor_alias_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_tensor_alias_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def make_tensor_clean_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP16],
):
    """正样本：make_tensor 视图 + 别名的合法访问不误报。"""
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [64, 64])
        t3 = t2
        pl.load(a, t3, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_make_tensor_clean_no_false_positive():
    """正样本：kernel 内视图与别名的合法访问不误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([64, 64], device=ST_DEVICE, dtype=torch.float16)
    make_tensor_clean_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x)
    logging.info("test_make_tensor_clean_no_false_positive OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def view_over_source_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    """视图声明超源（#17）：make_tensor(x, [200,200])，x 实际 [128,128]。

    视图共享 x 的数据指针——声明 40000 元素 > x 的 16384，视图内访问
    （如 [130,0]）物理上越过 x 的分配。修复前按视图声明 shape 判放行 →
    漏检；修复后声明即报（不需要越界访问发生）。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [200, 200])   # ===== 错误注入点：声明超源 =====
        pl.load(a, t2, [130, 0])             # 视图内"合法"（130+64=194 ≤ 200）
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_view_over_source_detected():
    """负样本：视图声明超出源 tensor 分配应在声明期检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 128])
    z = torch.zeros([128, 128], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        view_over_source_kernel(x, z)
        torch.npu.synchronize()
    assert any("source tensor" in str(f) for f in exc_info.value.findings)
    logging.info("test_view_over_source_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def chained_view_clean_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    p: pl.Ptr[pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    """正样本：链式视图 + 指针源，全部不误报。

    t2 = make_tensor(x, [64,256])：同 numel 重排（16384 == 16384）；
    t3 = make_tensor(t2, [128,128])：链式到根 x，numel 相等；
    t4 = make_tensor(p, [64,64])：指针源 → 跳过校验（无已知边界）。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [64, 256])
        t3 = pl.make_tensor(t2, [128, 128])
        _t4 = pl.make_tensor(p, [64, 64])  # 指针源视图：声明即完成超源检查
        pl.load(a, t3, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_chained_view_clean_no_false_positive():
    """正样本：链式视图（递归到根）与指针源视图不误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 128])
    p_src = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([128, 128], device=ST_DEVICE, dtype=torch.float16)
    chained_view_clean_kernel(x, p_src, z)
    torch.npu.synchronize()
    # kernel 只搬 64x64 的 tile 到 z 左上角
    torch.testing.assert_close(z[:64, :64], x[:64, :64])
    logging.info("test_chained_view_clean_no_false_positive OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def wide_stride_view_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    """大跨度 stride 视图超源（#17 补强）：footprint 判据。

    t = make_tensor(x, [2, 2], [16384, 1])：numel 仅 4（旧 numel 判据放行），
    但 t[1,1] 地址 = 16384+1，footprint = 1*16384 + 1*1 + 1 = 16386 > 16384
    元素 → 物理越源。按 op 契约（元素地址 = Σ i*stride）以 footprint 判定。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        # ===== 错误注入点：stride 跨度使 footprint 超出 x 的 16384 元素 =====
        t = pl.make_tensor(x, [2, 2], [16384, 1])
        pl.load(a, t, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_wide_stride_view_detected():
    """负样本：大跨度 stride 的视图 footprint 超源应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 128])
    z = torch.zeros([128, 128], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        wide_stride_view_kernel(x, z)
        torch.npu.synchronize()
    assert any("make_tensor view" in str(f) for f in exc_info.value.findings)
    logging.info("test_wide_stride_view_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def dtype_reinterpret_view_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    """跨 dtype 视图按字节判定（#17 补强）。

    x 为 FP16 [128,128] = 32768 字节；视图按 FP32 解释 [64,128] 元素 =
    32768 字节（字节级合法；numel 级 8192 < 16384 也过）→ 干净。
    若按纯 numel 判据，跨 dtype 的边界场景会误判；字节级正确。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t = pl.make_tensor(x, [64, 128], dtype=pl.DT_FP32)   # 字节级恰好等宽
        pl.load(a, t, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)


@pytest.mark.soc("950")
def test_dtype_reinterpret_view_clean():
    """正样本：字节级等宽的跨 dtype 视图不误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 128])
    z = torch.zeros([128, 128], device=ST_DEVICE, dtype=torch.float16)
    dtype_reinterpret_view_kernel(x, z)
    torch.npu.synchronize()
    logging.info("test_dtype_reinterpret_view_clean OK")


# -----------------------------------------------------------------------------
# 十一、动态 shape 视图（#19）与 Tile 标量访问（#11 残余）——运行时真值回归
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def dynamic_view_oob_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """动态 shape 视图越界（#19）：dim 是运行时表达式（x.shape）。

    修复前动态维记 0 → 跳过判定漏检；修复后 ViewShape 记录落真值
    (64,128)，回放按真值判：off 64 + 窗口 64 = 128 > 64 → 检出。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [x.shape[0] // 2, x.shape[1]])   # 运行时 (64,128)
        # ===== 错误注入点：行偏移 64 + 窗口 64 > 视图行数 64 =====
        pl.load(a, t2, [64, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_dynamic_view_oob_detected():
    """负样本：动态 shape 视图的越界访问应由运行时真值检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [128, 128])
    z = torch.zeros([128, 128], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        dynamic_view_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_dynamic_view_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def dynamic_view_over_source_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """动态 shape 视图超源（#19 运行时化）：footprint 表达式落盘真值。

    视图声明 [x.shape[0], x.shape[1], 4]（运行时 64x64x4 = 16384 元素）
    > 源 x 的 4096 元素 → 运行时 footprint 校验检出。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [x.shape[0], x.shape[1] * 4])   # ===== 超源 =====
        pl.load(a, t2, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_dynamic_view_over_source_detected():
    """负样本：动态 shape 视图的 footprint 超源应被运行时检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([64, 64], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        dynamic_view_over_source_kernel(x, z)
        torch.npu.synchronize()
    assert any("source tensor" in str(f) for f in exc_info.value.findings)
    logging.info("test_dynamic_view_over_source_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def tile_getval_oob_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP16],
):
    """Tile 容器 getval 线性越界（#11 残余）。

    tile [64,64] 共 4096 元素，读线性偏移 4096（末尾后 1）。
    修复前 Tile 容器无记录漏检；修复后 TileScalar 记录检出。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        # ===== 错误注入点：线性偏移 4096 = 元素总数，越界 1 个 =====
        pl.getval(a, 4096)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_tile_getval_oob_detected():
    """负样本：Tile 容器 getval 线性越界应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([64, 64], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        tile_getval_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("TILE_OUT_OF_BOUNDS" in str(f) and "scalar tile access" in str(f) for f in exc_info.value.findings)
    logging.info("test_tile_getval_oob_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def tile_getval_clean_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP16],
):
    """正样本：Tile 容器 getval 合法偏移不误报。"""
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.getval(a, 4095)   # 最后一个元素，合法
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_tile_getval_clean_no_false_positive():
    """正样本：Tile 容器 getval 末元素访问不误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 64])
    z = torch.zeros([64, 64], device=ST_DEVICE, dtype=torch.float16)
    tile_getval_clean_kernel(x, z)
    torch.npu.synchronize()
    logging.info("test_tile_getval_clean_no_false_positive OK")


# -----------------------------------------------------------------------------
# 十二、别名重绑定（固定指向语义：别名绑定赋值时刻的 tensor 对象）
# 前端三处一致按赋值时刻的视图处理（文档 Tensor.md 别名规则 / load 偏移
# 校验 / load 操作数类型即 codegen 实例化依据），重绑源名不影响别名。
# offset 用 DYNAMIC 形参的 shape（真运行时值，前端无法常量折叠拦截）。
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True)
def alias_rebind_clean_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """正样本：只超重绑后视图、不超别名所指视图的访问不误报。

    t3 绑定 t2 时刻的 [64,256]；t2 重绑 [64,64]。经 t3 动态偏移 64：
    按固定指向 [128,128] 判 64+64=128 ≤ 128 → 干净（名字跟随实现按
    [64,64] 判 128 > 64 → 误报）。前端亦按 [128,128] 校验（行为一致）。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [128, 128])  # 与 x 等宽（16384 元素）
        t3 = t2                             # t3 固定指向 [128,128]
        t2 = pl.make_tensor(x, [64, 64])    # 重绑不影响 t3
        pl.load(a, t3, [x.shape[0], 0])     # 动态 64：64+64=128 ≤ 128 合法
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_alias_rebind_clean_no_false_positive():
    """正样本：固定指向语义下经别名的合法访问不误报。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 256])
    z = torch.zeros([64, 256], device=ST_DEVICE, dtype=torch.float16)
    alias_rebind_clean_kernel(x, z)
    torch.npu.synchronize()
    # 数据层验证固定指向：load 经 t3（[128,128] 视图，stride 128）在
    # 偏移 (64,0) 读 64x64 窗口 = x 的线性元素 [8192, 16384) 重排；
    # 若硬件按重绑后的 [64,64] 视图寻址（stride 64），读到的是线性
    # [4096, 8192) 段——两种语义数据不同，可直接分辨。
    x_flat = x.cpu().reshape(-1)
    expected = x_flat[8192:16384].reshape(64, 128)[:, :64]
    torch.testing.assert_close(z[:64, :64].cpu(), expected)
    logging.info("test_alias_rebind_clean_no_false_positive OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def alias_rebind_oob_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    """负样本：超出别名所指视图的访问仍须检出（摊平不丢检测）。

    t3 固定指向 [128,128]；动态偏移 64*4=256：256+64=320 > 128 → 检出。
    """
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        t2 = pl.make_tensor(x, [128, 128])
        t3 = t2
        t2 = pl.make_tensor(x, [64, 64])
        # ===== 错误注入点：动态偏移 256 + 窗口 64 > 别名所指视图的 128 =====
        pl.load(a, t3, [x.shape[0] * 4, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_alias_rebind_oob_detected():
    """负样本：超出别名所指视图的访问应被检出。"""
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [64, 256])
    z = torch.zeros([64, 256], device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        alias_rebind_oob_kernel(x, z)
        torch.npu.synchronize()
    assert any("GM_OUT_OF_BOUNDS" in str(f) for f in exc_info.value.findings)
    logging.info("test_alias_rebind_oob_detected OK")


# -----------------------------------------------------------------------------
# 十三、@pl.jit(name=...) 插桩回归（入口识别不依赖命名巧合）
# -----------------------------------------------------------------------------


@pl.jit(sanitizer=True, auto_mutex=True, name="sanitizer_named_entry_kernel")
def named_clean_kernel(
    x: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
    z: pl.Tensor[[TILE_M, TILE_N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(a, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, a, [0, 0])


@pytest.mark.soc("950")
def test_named_kernel_sanitizer_instrumented():
    """正样本：name= 与函数名不一致时仍应完成插桩并正常启动。

    回归：入口函数曾按 func->name_ == program->name_（命名巧合）识别，
    @pl.jit(name=...) 与函数名不同时插桩被静默跳过、隐藏参数未追加，
    launch 报 InvalidType: Expected 2 args, got 4。
    """
    _require_a5(ST_DEVICE)
    x = _inputs(ST_DEVICE, [TILE_M, TILE_N])
    z = torch.zeros([TILE_M, TILE_N], device=ST_DEVICE, dtype=torch.float16)
    named_clean_kernel(x, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x)
    logging.info("test_named_kernel_sanitizer_instrumented OK")


# ===========================================================================
# unit_flag phase detection (det 7) -- unit_flag_matching_full_analysis.md
# ===========================================================================

def _cube_tiles():
    """Mat(L1) 中转 + move 到 Left/Right（Left/Right 不能直接 load）。"""
    mat_a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000, mutex_ids=[0])
    mat_b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x8000, mutex_ids=[1])
    left_g = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    right_g = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])
    return mat_a, mat_b, left_g, right_g, acc_g


def _load_operands(cube_tiles, a, b, ka, kb):
    """load GM 到 Mat 再 move 到 Left/Right，返回当前迭代可用的 al/br。"""
    mat_a, mat_b, left_g, right_g, acc_g = cube_tiles
    ma = mat_a.current()
    mb = mat_b.current()
    pl.load(ma, a, [0, ka])
    pl.load(mb, b, [kb, 0])
    al = left_g.current()
    br = right_g.current()
    pl.move(al, ma)
    pl.move(br, mb)
    return al, br


def _phase_inputs(device, m=M_DIM):
    return torch.randn(m, K_DIM, device=device, dtype=torch.float16)

# =============================================================================
# 正样本（合法形态，不报）
# =============================================================================


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_clean_software_sync_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """模式 2：双方不带 phase，软件同步路径（CV 混合 kernel 形态）。"""
    cube_tiles = _cube_tiles()
    with pl.section_cube():
        ac = cube_tiles[4].current()
        al, br = _load_operands(cube_tiles, a, b, 0, 0)
        pl.matmul(ac, al, br)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, ac, [0, 0])


@pytest.mark.soc("950")
def test_phase_clean_software_sync():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_clean_software_sync--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    phase_clean_software_sync_kernel(a, b, out)
    torch.npu.synchronize()
    # 单块 tile [TILE_M, TILE_K] x [TILE_K, TILE_N]：只算 K 的前 TILE_K 段
    ref = torch.matmul(a[:, :TILE_K].float(), b[:TILE_K, :].float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    logging.info("test_phase_clean_software_sync OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_clean_kchain_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """模式 2（phase 版）：K 分块，首块 Partial、末块 Final、store Final 收尾。"""
    cube_tiles = _cube_tiles()
    with pl.section_cube():
        ac = cube_tiles[4].current()
        for k in pl.range(0, K_DIM, TILE_K):
            al, br = _load_operands(cube_tiles, a, b, k, k)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
def test_phase_clean_kchain():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_clean_kchain--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    phase_clean_kchain_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)
    logging.info("test_phase_clean_kchain OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_control_flow_group_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """Select an Acc tile through device control flow before the phase op.

    The selected tile expression must keep its runtime TileGroup slot. The
    sanitizer address record must follow that expression instead of resolving
    the last textual assignment in the host pass.
    """
    cube_tiles = _cube_tiles()
    acc_true = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc),
        addrs=0x4000, mutex_ids=[5],
    )
    acc_false = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc),
        addrs=0x8000, mutex_ids=[6],
    )
    with pl.section_cube():
        al, br = _load_operands(cube_tiles, a, b, 0, 0)
        ac = acc_true.current()
        if pl.get_block_idx() != 0:
            ac = acc_false.current()
        pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
def test_phase_control_flow_group_address_is_not_misresolved():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_control_flow_group_address_is_not_misresolved--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    try:
        phase_control_flow_group_kernel(a, b, out)
        torch.npu.synchronize()
    except SanitizerReplayError as exc:
        phase_findings = [f for f in exc.findings if f.kind.startswith("PHASE_")]
        assert not phase_findings, phase_findings
        raise
    logging.info("test_phase_control_flow_group_address_is_not_misresolved OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_clean_insert_final_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
):
    """insert(STPhase.Final) 消费 K 链：Acc→Mat 读端闭环。

    回归：insert 曾不在 phase 插桩注册表中，合法的 matmul(Final)→insert(Final)
    链读侧无记录，被终态检查误报 PHASE_DEADLOCK(C-T1)。
    """
    cube_tiles = _cube_tiles()
    mt = pl.make_tile(
        pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat),
        addr=0x10000)
    with pl.section_cube():
        ac = cube_tiles[4].current()
        al, br = _load_operands(cube_tiles, a, b, 0, 0)
        pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
        pl.insert(mt, ac, [0, 0], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
def test_phase_clean_insert_final():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_clean_insert_final--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    phase_clean_insert_final_kernel(a, b)  # 修复前: SanitizerReplayError(PHASE_DEADLOCK C-T1)
    torch.npu.synchronize()
    logging.info("test_phase_clean_insert_final OK")


# =============================================================================
# 负样本 A：配置匹配
# =============================================================================


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_a1_mixed_sync_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """A-1：matmul 带 phase、store 不带——两套同步机制混用，精度错误。"""
    cube_tiles = _cube_tiles()
    with pl.section_cube():
        ac = cube_tiles[4].current()
        al, br = _load_operands(cube_tiles, a, b, 0, 0)
        pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0])  # 违规点：未配 STPhase


@pytest.mark.soc("950")
def test_phase_a1_mixed_sync_detected():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_a1_mixed_sync_detected--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(SanitizerReplayError) as exc_info:
        phase_a1_mixed_sync_kernel(a, b, out)
        torch.npu.synchronize()
    assert any("PHASE_MISMATCH" in str(f) for f in exc_info.value.findings)
    logging.info("test_phase_a1_mixed_sync_detected OK")


# =============================================================================
# 负样本 B：flag 状态机
# =============================================================================


# =============================================================================
# 负样本 C：终态
# =============================================================================


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_ct1_dangling_final_write_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
):
    """C-T1：Final 写后无读消费——flag 悬挂，下次写该块死锁（kernel 本身能跑完）。"""
    cube_tiles = _cube_tiles()
    with pl.section_cube():
        ac = cube_tiles[4].current()
        al, br = _load_operands(cube_tiles, a, b, 0, 0)
        # 违规点：Final 写之后漏 store，无人释放块
        pl.matmul(ac, al, br, phase=pl.AccPhase.Final)


@pytest.mark.soc("950")
def test_phase_ct1_dangling_final_write_detected():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_ct1_dangling_final_write_detected--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    with pytest.raises(SanitizerReplayError) as exc_info:
        phase_ct1_dangling_final_write_kernel(a, b)
        torch.npu.synchronize()
    assert len(exc_info.value.findings) == 1
    assert any("PHASE_DEADLOCK" in str(f) for f in exc_info.value.findings)
    logging.info("test_phase_ct1_dangling_final_write_detected OK")


# =============================================================================
# 负样本 D：轮结构（静默精度错误）
# =============================================================================


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_d1_acc_chain_head_kernel(
    a: pl.Tensor[[2 * M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
    out: pl.Tensor[[2 * M_DIM, N_DIM], pl.DT_FP32],
):
    """D-1：切 M 复用 tile，第二链首块误用 matmul_acc——累加到上一轮残留值，静默错误。"""
    cube_tiles = _cube_tiles()
    with pl.section_cube():
        ac = cube_tiles[4].current()
        for i in pl.range(0, 2 * M_DIM, TILE_M):
            for k in pl.range(0, K_DIM, TILE_K):
                al, br = _load_operands(cube_tiles, a, b, i, k)
                if k == 0:
                    if i == 0:
                        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                    else:
                        # 违规点：第二链首块误用 matmul_acc（复制中段代码忘改）
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                else:
                    pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
            pl.store(out, ac, [i, 0], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
def test_phase_d1_acc_chain_head_detected():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_d1_acc_chain_head_detected--------------")
    a = _phase_inputs(ST_DEVICE, m=2 * M_DIM)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(2 * M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(SanitizerReplayError) as exc_info:
        phase_d1_acc_chain_head_kernel(a, b, out)
        torch.npu.synchronize()
    assert any("PHASE_MISMATCH" in str(f) for f in exc_info.value.findings)
    logging.info("test_phase_d1_acc_chain_head_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_mid_overwrite_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """链中段误用覆盖写（复制链首行忘改的变体）：部分和被静默丢弃。"""
    cube_tiles = _cube_tiles()
    with pl.section_cube():
        ac = cube_tiles[4].current()
        for k in pl.range(0, K_DIM, TILE_K):
            al, br = _load_operands(cube_tiles, a, b, k, k)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            else:
                # 违规点：末块应为 matmul_acc，复制了链首行忘改——覆盖写丢掉首块部分和
                pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
def test_phase_mid_overwrite_detected():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_mid_overwrite_detected--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(SanitizerReplayError) as exc_info:
        phase_mid_overwrite_kernel(a, b, out)
        torch.npu.synchronize()
    assert any("PHASE_MISMATCH" in str(f) for f in exc_info.value.findings)
    logging.info("test_phase_mid_overwrite_detected OK")


@pl.jit(sanitizer=True, auto_mutex=True)
def phase_write_mixing_kernel(
    a: pl.Tensor[[M_DIM, K_DIM], pl.DT_FP16],
    b: pl.Tensor[[K_DIM, N_DIM], pl.DT_FP16],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """写侧混用（P11）：链首 matmul 忘带 phase、后续块带——两套同步机制缺排序边。

    回归：该形态曾无任何报告（或被 D-1 以"未初始化内存"误报）——
    实际基值由无 phase 的覆盖写建立，数据未必错，但两种机制互不排序。
    """
    cube_tiles = _cube_tiles()
    with pl.section_cube():
        ac = cube_tiles[4].current()
        for k in pl.range(0, K_DIM, TILE_K):
            al, br = _load_operands(cube_tiles, a, b, k, k)
            if k == 0:
                pl.matmul(ac, al, br)  # 违规点：链首忘带 phase（软件同步）
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)  # 硬件握手
        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
def test_phase_write_mixing_detected():
    _require_a5(ST_DEVICE)
    logging.info("------------test_phase_write_mixing_detected--------------")
    a = _phase_inputs(ST_DEVICE)
    b = torch.randn(K_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float16)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(SanitizerReplayError) as exc_info:
        phase_write_mixing_kernel(a, b, out)
        torch.npu.synchronize()
    assert len(exc_info.value.findings) == 1
    assert any("mix phase and phase-less writes" in str(f) for f in exc_info.value.findings)
    assert not any("matmul_acc" in str(f) for f in exc_info.value.findings)
    logging.info("test_phase_write_mixing_detected OK")


# =============================================================================
# 单元测试（构造记录直测 check_phase_state_machine）
#
# 卡死类（A-2 / B 族）端到端时 kernel 在设备上等 flag 永久挂起，走不到回放，
# 因此用记录流直测判定逻辑（覆盖 14 场景中所有卡死形态 + D-2 + 链中段覆盖写）。
# 记录字段：op_kind / phase / acc_addr / acc_src_addr / line（acc_src_addr 默认 -1，region 默认 0）。
# =============================================================================

_CTX = ReplayContext("ut.py")

# op_kind: 1 matmul 2 matmul_acc 3 matmul_bias 4 matmul_mx 5 matmul_mx_acc
#          6 store 7 store_tile(保留) 8 move 9 move_fp(保留) 10 insert；
#          phase: 0 Unspec 1 Partial 2 Final
def _pr(kind, phase, acc_addr=0, line=10, acc_src_addr=None):
    fields = {"op_kind": kind, "phase": phase, "acc_addr": acc_addr, "line": line}
    if acc_src_addr is not None:
        fields["acc_src_addr"] = acc_src_addr
    return PhaseAccessRecord(**fields)


def _kinds(findings):
    return sorted({f.kind for f in findings})


@pytest.mark.soc("950")
def test_ut_phase_clean_kchain_records():
    """正样本：P -> P(F) -> store(F) 单链合法。"""
    recs = [_pr(1, 1), _pr(2, 2), _pr(6, 2)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_clean_software_sync_records():
    """正样本：全 Unspecified（软件同步路径）。"""
    recs = [_pr(1, 0), _pr(8, 0)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_clean_multiread_records():
    """正样本：多读链 F写 -> P读 -> F读。"""
    recs = [_pr(1, 2), _pr(6, 1), _pr(6, 2)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_a1_mixed_sync_records():
    """A-1：phased 写 + Unspecified 读 -> PHASE_MISMATCH。"""
    recs = [_pr(1, 2), _pr(6, 0)]
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_MISMATCH" in kinds


@pytest.mark.soc("950")
def test_ut_phase_a2_unspec_write_phased_read_records():
    """A-2：Unspecified 写 + phased 读 -> PHASE_DEADLOCK。"""
    recs = [_pr(1, 0), _pr(6, 2)]
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.kind for f in findings] == ["PHASE_DEADLOCK"]
    assert "unit_flag" in findings[0].message


@pytest.mark.soc("950")
def test_ut_phase_a2_reverse_stops_at_the_first_phased_read():
    """phased 读先发生时立即阻塞，不能把不可达的软件写当成错误点。"""
    recs = [_pr(6, 2, line=10), _pr(1, 0, line=11)]
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.kind for f in findings] == ["PHASE_DEADLOCK"]
    assert findings[0].location == "ut.py:10"


@pytest.mark.soc("950")
def test_ut_phase_b1a_partial_read_then_next_write_records():
    """B-1a：Partial 读不复位，下轮写撞 flag=1 -> PHASE_DEADLOCK。"""
    recs = [_pr(1, 2), _pr(6, 1), _pr(1, 1)]  # F写 P读（不复位） 新写
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_DEADLOCK" in kinds


@pytest.mark.soc("950")
def test_ut_phase_b1b_double_final_write_records():
    """B-1b：同链双 Final 写 -> PHASE_DEADLOCK。"""
    recs = [_pr(1, 2), _pr(2, 2)]
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.kind for f in findings] == ["PHASE_DEADLOCK"]


@pytest.mark.soc("950")
def test_ut_phase_b1c_dangling_then_next_write_records():
    """B-1c：悬挂 Final 后的写同时暴露 flag 和 K 链故障。"""
    recs = [_pr(1, 2), _pr(1, 1)]  # F写（无读消费） 新写
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.kind for f in findings] == ["PHASE_DEADLOCK", "PHASE_MISMATCH"]


@pytest.mark.soc("950")
def test_ut_phase_b2a_read_without_write_records():
    """B-2a：无写就读 -> PHASE_DEADLOCK。"""
    recs = [_pr(6, 2)]
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_DEADLOCK" in kinds


@pytest.mark.soc("950")
def test_ut_phase_b2b_all_partial_write_records():
    """B-2b：全 Partial 写 + Final 读（文档案例一）-> PHASE_DEADLOCK。"""
    recs = [_pr(1, 1), _pr(2, 1), _pr(6, 2)]
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_DEADLOCK" in kinds


@pytest.mark.soc("950")
def test_ut_phase_b2c_double_final_read_records():
    """B-2c：双 Final 读 -> PHASE_DEADLOCK。"""
    recs = [_pr(1, 2), _pr(6, 2), _pr(6, 2)]
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_DEADLOCK" in kinds


@pytest.mark.soc("950")
def test_ut_phase_b3_final_then_partial_write_records():
    """B-3：Final 写后同轮 Partial 写 -> PHASE_DEADLOCK。"""
    recs = [_pr(1, 1), _pr(2, 2), _pr(2, 1), _pr(6, 2)]
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_DEADLOCK" in kinds


@pytest.mark.soc("950")
def test_ut_phase_ct1_dangling_final_write_records():
    """C-T1：序列结束 flag 悬挂 -> PHASE_DEADLOCK。"""
    recs = [_pr(1, 2)]  # F写无读
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.kind for f in findings] == ["PHASE_DEADLOCK"]


@pytest.mark.soc("950")
def test_ut_phase_d1_acc_chain_head_records():
    """D-1：Final 读闭链后新链以 matmul_acc 开头 -> PHASE_MISMATCH。"""
    recs = [_pr(1, 1), _pr(2, 2), _pr(6, 2), _pr(2, 1)]  # 链1F收尾 链2 acc开头
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_MISMATCH" in kinds


@pytest.mark.soc("950")
def test_ut_phase_d2_mid_chain_bias_records():
    """D-2：链中段 matmul_bias -> PHASE_MISMATCH。"""
    recs = [_pr(1, 1), _pr(3, 2)]  # 首块后接 bias（非轮首）
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_MISMATCH" in kinds


@pytest.mark.soc("950")
def test_ut_phase_mid_overwrite_records():
    """链中段覆盖写（复制链首行忘改）：部分和被静默丢弃 -> PHASE_MISMATCH。"""
    recs = [_pr(1, 1), _pr(1, 1), _pr(2, 2), _pr(6, 2)]  # 首块后再来一个覆盖写
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_MISMATCH" in kinds


@pytest.mark.soc("950")
def test_ut_phase_mid_overwrite_mx_records():
    """链中段覆盖写变体：matmul_mx（kind 4）同样丢弃部分和。"""
    recs = [_pr(1, 1), _pr(4, 2), _pr(6, 2)]
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_MISMATCH" in kinds


@pytest.mark.soc("950")
def test_ut_phase_d3_wrong_loop_var_records():
    """D-3：分支条件误用变量 -> 部分轮落 B-2b（读前置失败）。"""
    # i=0 轮：P P + F读（漏 Final -> B-2b）
    recs = [_pr(1, 1), _pr(2, 1), _pr(6, 2)]
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_DEADLOCK" in kinds


@pytest.mark.soc("950")
def test_ut_phase_d4_mid_chain_final_read_records():
    """D-4：多读链中间 Final 读 + 后续读 -> PHASE_DEADLOCK（B-2c 形态）。"""
    recs = [_pr(1, 2), _pr(6, 1), _pr(6, 2), _pr(6, 1)]  # P F P：F后还有读
    kinds = _kinds(check_phase_state_machine(recs, _CTX))
    assert "PHASE_DEADLOCK" in kinds


@pytest.mark.soc("950")
def test_ut_phase_acc_addrs_independent_records():
    """两个 Acc tile 各自合法，互不干扰。"""
    recs = [_pr(1, 2, acc_addr=0), _pr(6, 2, acc_addr=0), _pr(1, 2, acc_addr=1), _pr(6, 2, acc_addr=1)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_two_kchains_sequential_records():
    """切 M 两轮复用（合法）：F收尾 -> 覆盖写开新链 -> F收尾。"""
    recs = [_pr(1, 1), _pr(2, 2), _pr(6, 2), _pr(1, 1), _pr(2, 2), _pr(6, 2)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_mode_switches_after_completed_rounds():
    """Completed phased/software rounds may reuse one Acc with a new mode."""
    phased_then_software = [
        _pr(1, 2, line=10),
        _pr(6, 2, line=11),
        _pr(1, 0, line=12),
        _pr(6, 0, line=13),
    ]
    software_then_phased = [
        _pr(1, 0, line=20),
        _pr(6, 0, line=21),
        _pr(1, 2, line=22),
        _pr(6, 2, line=23),
    ]
    assert check_phase_state_machine(phased_then_software, _CTX) == []
    assert check_phase_state_machine(software_then_phased, _CTX) == []


def test_ut_phase_insert_final_read_records():
    """正样本：matmul(F) -> insert(F) 读端闭环——insert 缺记录时曾误报 C-T1。"""
    recs = [_pr(1, 1), _pr(2, 2), _pr(10, 2)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_insert_read_without_write_records():
    """B-2a 形态：insert(F) 读一个无人写的块 -> PHASE_DEADLOCK。"""
    recs = [_pr(10, 2)]
    assert "PHASE_DEADLOCK" in _kinds(check_phase_state_machine(recs, _CTX))


@pytest.mark.soc("950")
def test_ut_phase_insert_unspec_mixed_records():
    """A-1 形态：phased 写 + Unspecified insert 读（漏 STPhase）-> PHASE_MISMATCH。"""
    recs = [_pr(1, 2), _pr(10, 0)]
    assert "PHASE_MISMATCH" in _kinds(check_phase_state_machine(recs, _CTX))


@pytest.mark.soc("950")
def test_ut_phase_write_mixing_records():
    """写侧混用（P11）：无 phase 覆盖写开链 + phased 累加 -> 混用报错，且 D-1 不再误报。"""
    recs = [_pr(1, 0), _pr(2, 1), _pr(2, 2), _pr(6, 2)]
    findings = check_phase_state_machine(recs, _CTX)
    msgs = [f.message for f in findings]
    assert len(findings) == 1
    assert any("mix phase and phase-less writes" in m for m in msgs)
    assert not any("matmul_acc" in m for m in msgs)  # 基值已由无 phase 写建立，D-1 不报


@pytest.mark.soc("950")
def test_ut_phase_write_mixing_reverse_records():
    """写侧混用与顺序无关：phased 写在前、无 phase 写在后同样报。"""
    recs = [_pr(1, 1), _pr(1, 0), _pr(6, 2)]
    msgs = [f.message for f in check_phase_state_machine(recs, _CTX)]
    assert any("mix phase and phase-less writes" in m for m in msgs)


@pytest.mark.soc("950")
def test_ut_phase_pure_software_no_mixing_records():
    """纯软件同步流（全 Unspecified）不报混用。"""
    recs = [_pr(1, 0), _pr(8, 0), _pr(6, 0)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_acc_src_addr_has_backward_compatible_default():
    """旧构造方式不传 acc_src_addr 时使用 -1 哨兵。"""
    assert _pr(1, 0).acc_src_addr == -1


@pytest.mark.soc("950")
def test_ut_phase_pure_software_accumulate_records():
    """纯软件流 W(0) -> A(0) 合法，UNSPECIFIED 不触发 K 链误报。"""
    recs = [_pr(1, 0), _pr(2, 0)]
    assert check_phase_state_machine(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_mode_and_kchain_findings_at_same_point_coexist():
    """同点的写模式混用、契约和 K 链故障全部保留。"""
    recs = [_pr(1, 0, acc_addr=0, line=10), _pr(2, 1, acc_addr=0, line=11, acc_src_addr=1)]
    findings = check_phase_state_machine(recs, _CTX)
    assert len(findings) == 2
    assert any("mix phase and phase-less writes" in f.message for f in findings)
    assert any("dst and acc must be the same Tile" in f.message for f in findings)


@pytest.mark.soc("950")
def test_ut_phase_findings_at_different_record_indexes_coexist():
    """MIXED 模式不能吞掉不同记录位置上的终态 flag 故障。"""
    recs = [_pr(1, 2, acc_addr=0, line=10), _pr(6, 0, acc_addr=0, line=11)]
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.location for f in findings] == ["ut.py:10", "ut.py:11"]
    assert [f.kind for f in findings] == ["PHASE_DEADLOCK", "PHASE_MISMATCH"]
    assert "unit_flag" in findings[0].message
    assert "mix" in findings[1].message


@pytest.mark.soc("950")
def test_ut_phase_mixed_stream_keeps_flag_findings():
    """同轮 MIXED 保留模式与 flag 故障；新轮只保留当前轮本身的故障。"""
    cases = [
        [_pr(1, 2, line=10), _pr(6, 0, line=11)],
        [_pr(1, 1, line=20), _pr(1, 0, line=21), _pr(6, 2, line=22)],
        [_pr(1, 2, line=30), _pr(6, 2, line=31), _pr(1, 0, line=32), _pr(6, 2, line=33)],
    ]
    for recs, expected_count in zip(cases, (2, 2, 1)):
        findings = check_phase_state_machine(recs, _CTX)
        assert len(findings) == expected_count
        if expected_count == 2:
            assert {f.kind for f in findings} == {"PHASE_MISMATCH", "PHASE_DEADLOCK"}
        else:
            assert [f.kind for f in findings] == ["PHASE_DEADLOCK"]


@pytest.mark.soc("950")
def test_ut_phase_out_of_place_acc_is_rejected_even_when_source_initialized():
    """官方契约要求 matmul_acc 的 dst_tile 与 acc_tile 必须相同。"""
    recs = [
        _pr(1, 1, acc_addr=0, line=10),
        _pr(2, 2, acc_addr=1, line=11, acc_src_addr=0),
        _pr(6, 2, acc_addr=1, line=12),
    ]
    findings = check_phase_state_machine(recs, _CTX)
    assert len(findings) == 2
    assert any("dst and acc must be the same Tile" in f.message for f in findings)
    assert any("first write of a chain" in f.message for f in findings)


@pytest.mark.soc("950")
def test_ut_phase_out_of_place_acc_rejects_uninitialized_source_tile():
    """异地累加即使源 Acc tile 未初始化，也按同 Tile 契约报错。"""
    recs = [_pr(2, 1, acc_addr=1, line=11, acc_src_addr=0)]
    findings = check_phase_state_machine(recs, _CTX)
    assert len(findings) == 2
    assert any("dst and acc must be the same Tile" in f.message for f in findings)
    assert any("first write of a chain" in f.message for f in findings)


@pytest.mark.soc("950")
def test_ut_phase_out_of_place_acc_contract_survives_flag_failure():
    """契约错误即使同点还有 flag 错误，也必须被报告。"""
    recs = [
        _pr(1, 2, acc_addr=0, line=10),
        _pr(2, 2, acc_addr=0, line=11, acc_src_addr=1),
    ]
    findings = check_phase_state_machine(recs, _CTX)
    assert any("dst and acc must be the same Tile" in f.message for f in findings)
    assert any(f.kind == "PHASE_DEADLOCK" for f in findings)


@pytest.mark.soc("950")
def test_ut_phase_mx_acc_out_of_place_is_rejected():
    """matmul_mx_acc 与 matmul_acc 共享同一个原地累加契约。"""
    findings = check_phase_state_machine(
        [_pr(5, 2, acc_addr=1, line=12, acc_src_addr=0)], _CTX
    )
    assert len(findings) == 3
    assert any("dst and acc must be the same Tile" in f.message for f in findings)
    assert any("first write of a chain" in f.message for f in findings)
    assert any("unit_flag still set" in f.message for f in findings)


@pytest.mark.soc("950")
def test_ut_phase_partial_write_at_eof_is_accepted_in_v1():
    """W(P) 不置位 unit_flag；v1 缺少用途证据时不判错。"""
    assert check_phase_state_machine([_pr(1, 1)], _CTX) == []


@pytest.mark.soc("950")
def test_ut_phase_interleaved_acc_addresses_keep_independent_flags():
    """一次全局扫描中，不同 acc_addr 的 unit_flag 不能互相干扰。"""
    recs = [
        _pr(1, 2, acc_addr=0, line=10),
        _pr(1, 2, acc_addr=1, line=11),
        _pr(6, 2, acc_addr=0, line=12),
        _pr(1, 2, acc_addr=1, line=13),
    ]
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.location for f in findings] == ["ut.py:13", "ut.py:13"]
    assert any(f.kind == "PHASE_DEADLOCK" for f in findings)
    assert any(f.kind == "PHASE_MISMATCH" for f in findings)


@pytest.mark.soc("950")
def test_ut_phase_fatal_address_does_not_stop_other_addresses():
    """一个 Acc 地址挂死后，其他地址仍须继续回放并报告问题。"""
    recs = [
        _pr(6, 2, acc_addr=0, line=10),
        _pr(1, 2, acc_addr=1, line=11),
        _pr(6, 2, acc_addr=1, line=12),
        _pr(6, 2, acc_addr=1, line=13),
    ]
    findings = check_phase_state_machine(recs, _CTX)
    assert [f.location for f in findings] == ["ut.py:10", "ut.py:13"]
    assert all(f.kind == "PHASE_DEADLOCK" for f in findings)


def test_ut_replay_keeps_distinct_findings_at_same_location(monkeypatch):
    """同一行的不同诊断都保留，完全重复项只折叠 hits。"""
    class _Record:
        pass

    first = Finding("PHASE_MISMATCH", "mode mismatch", "ut.py:10", "fix mode")
    second = Finding("PHASE_MISMATCH", "contract mismatch", "ut.py:10", "fix tile")
    duplicate = Finding("PHASE_MISMATCH", "mode mismatch", "ut.py:10", "fix mode")
    monkeypatch.setattr(
        replay_module,
        "DETECTIONS",
        [Detection(_Record, "ERROR", lambda records, ctx:[first, second, duplicate])],
    )

    findings = run_replay([_Record()], _CTX)
    assert len(findings) == 2
    assert any(
        f.message == "mode mismatch  (same defect hit 2 times at this location)" and f.hits == 2
        for f in findings
    )
    assert any(f.message == "contract mismatch" and f.hits == 1 for f in findings)




# ===========================================================================
# sync_src/sync_dst pairing detection (det 8) -- sync_pairing_detection_analysis.md
# ===========================================================================

def _vec_tiles():
    tt = pl.TileType(shape=[M_DIM, N_DIM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    ta = pl.make_tile(tt, addr=0x0000)
    tb = pl.make_tile(tt, addr=0x4000)
    tc = pl.make_tile(tt, addr=0x8000)
    return ta, tb, tc


# =============================================================================
# 正样本（合法配对，不报）
# =============================================================================


@pl.jit(sanitizer=True)
def sync_clean_basic_kernel(
    a: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
    b: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """API 文档示例原样：load -> src/dst(MTE2,V,0) -> add -> src/dst(V,MTE3,1) -> store。"""
    ta, tb, tc = _vec_tiles()
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_sync_clean_basic():
    _require_a5(ST_DEVICE)
    logging.info("------------test_sync_clean_basic--------------")
    a = torch.randn(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    sync_clean_basic_kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b)
    logging.info("test_sync_clean_basic OK")


@pl.jit(sanitizer=True)
def sync_clean_reuse_kernel(
    a: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """消费完成后反向 fence，再复用 MTE2->V event。"""
    ta, tb, tc = _vec_tiles()
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tc, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tb, ta, tc)
        # The first V consumer must complete before MTE2 overwrites tc.
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.load(tc, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(ta, tb, tc)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.store(out, ta, [0, 0])


@pytest.mark.soc("950")
def test_sync_clean_reuse():
    _require_a5(ST_DEVICE)
    logging.info("------------test_sync_clean_reuse--------------")
    a = torch.randn(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    sync_clean_reuse_kernel(a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + a + a)
    logging.info("test_sync_clean_reuse OK")


# =============================================================================
# 负样本（违规，回放报告）
# =============================================================================


@pl.jit(sanitizer=True)
def sync_s1_missing_dst_kernel(
    a: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
    b: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
    out: pl.Tensor[[M_DIM, N_DIM], pl.DT_FP32],
):
    """S-1：只有 src 没有 dst——等待方根本没等，数据竞争。"""
    ta, tb, tc = _vec_tiles()
    with pl.section_vector():
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        # 违规点：漏写 sync_dst
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tc, ta, tb)
        pl.store(out, tc, [0, 0])


@pytest.mark.soc("950")
def test_sync_s1_missing_dst_detected():
    _require_a5(ST_DEVICE)
    logging.info("------------test_sync_s1_missing_dst_detected--------------")
    a = torch.randn(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    b = torch.randn(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    out = torch.zeros(M_DIM, N_DIM, device=ST_DEVICE, dtype=torch.float32)
    with pytest.raises(SanitizerReplayError) as exc_info:
        sync_s1_missing_dst_kernel(a, b, out)
        torch.npu.synchronize()
    assert any("SYNC_UNPAIRED_SRC" in str(f) for f in exc_info.value.findings)
    logging.info("test_sync_s1_missing_dst_detected OK")


# =============================================================================
# 单元测试（记录流直测 check_sync_pairing）
# Unit tests share the _CTX / _kinds helpers of the phase block above.

def _sr(is_src, set_pipe=1, wait_pipe=4, event_id=0, line=10, region=0):
    return SyncAccessRecord(event_id=event_id, set_pipe=set_pipe, wait_pipe=wait_pipe,
                            is_src=is_src, line=line, region=region)




@pytest.mark.soc("950")
def test_ut_sync_clean_basic_records():
    recs = [_sr(1), _sr(0), _sr(1, set_pipe=4, wait_pipe=3, event_id=1), _sr(0, set_pipe=4, wait_pipe=3, event_id=1)]
    assert check_sync_pairing(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_sync_clean_reuse_records():
    recs = [_sr(1), _sr(0), _sr(1), _sr(0)]
    assert check_sync_pairing(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_sync_clean_multi_key_records():
    recs = [_sr(1, event_id=0), _sr(1, event_id=1), _sr(0, event_id=0), _sr(0, event_id=1)]
    assert check_sync_pairing(recs, _CTX) == []


@pytest.mark.soc("950")
def test_ut_sync_s1_unpaired_src_records():
    recs = [_sr(1)]
    assert "SYNC_UNPAIRED_SRC" in _kinds(check_sync_pairing(recs, _CTX))


@pytest.mark.soc("950")
def test_ut_sync_s2_unpaired_dst_records():
    recs = [_sr(0)]
    assert "SYNC_UNPAIRED_DST" in _kinds(check_sync_pairing(recs, _CTX))


@pytest.mark.soc("950")
def test_ut_sync_s4_event_reuse_records():
    recs = [_sr(1), _sr(1)]
    assert "SYNC_EVENT_REUSE" in _kinds(check_sync_pairing(recs, _CTX))


@pytest.mark.soc("950")
def test_ut_sync_s3_param_mismatch_records():
    """S-3：wait_pipe 不一致——落入不同分组，各自以漏配对暴露。"""
    recs = [_sr(1, wait_pipe=4), _sr(0, wait_pipe=5)]  # src 等 V，dst 等 MTE3
    kinds = _kinds(check_sync_pairing(recs, _CTX))
    assert "SYNC_UNPAIRED_SRC" in kinds and "SYNC_UNPAIRED_DST" in kinds


@pytest.mark.soc("950")
def test_ut_sync_s3_event_mismatch_records():
    """S-3 变体：event_id 不一致。"""
    recs = [_sr(1, event_id=0), _sr(0, event_id=1)]
    kinds = _kinds(check_sync_pairing(recs, _CTX))
    assert "SYNC_UNPAIRED_SRC" in kinds and "SYNC_UNPAIRED_DST" in kinds


@pytest.mark.soc("950")
def test_ut_sync_regions_independent_records():
    recs = [_sr(1, line=10, region=0), _sr(0, line=11, region=1)]
    assert check_sync_pairing(recs, _CTX) != []  # 不同 region 各自漏配对 -> 报


@pytest.mark.soc("950")
def test_ut_sync_s1_dangling_src_location_records():
    """S-1 定位回归：终态报告指向悬空 src 的行（曾指向组内第一条记录）。"""
    # [src@10, dst@11, src@20]：悬空的是行 20 的 src
    recs = [_sr(1, line=10), _sr(0, line=11), _sr(1, line=20)]
    s1 = [f for f in check_sync_pairing(recs, _CTX) if f.kind == "SYNC_UNPAIRED_SRC"]
    assert len(s1) == 1 and s1[0].location.endswith(":20")
    # 乱序变体 [dst@10, src@20]：S-1 不应指向 dst 的行
    recs = [_sr(0, line=10), _sr(1, line=20)]
    s1 = [f for f in check_sync_pairing(recs, _CTX) if f.kind == "SYNC_UNPAIRED_SRC"]
    assert len(s1) == 1 and s1[0].location.endswith(":20")
    # S-4 变体 [src@10, src@11]：悬空的是第二个 src（行 11）
    recs = [_sr(1, line=10), _sr(1, line=11)]
    s1 = [f for f in check_sync_pairing(recs, _CTX) if f.kind == "SYNC_UNPAIRED_SRC"]
    assert len(s1) == 1 and s1[0].location.endswith(":11")



if __name__ == "__main__":
    """python3 直接执行入口：跑本文件全部用例（选跑单个用例请用 pytest）。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short", "--no-header"]))
