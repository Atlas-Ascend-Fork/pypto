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
"""
AICPU-CTRL 构建任务性能看护。

覆盖 EXEC_DYN 树：ROOT_FUNC / ALLOCATE_WORKSPACE / FAST_STITCH / UPDATE_SLOT /
SUBMIT_AICORE / DECIDE_INCAST / STAGE_BUILD_TASK / RESOLVE_EARLY 等。
采集：DUMP_DEVICE_PERF=true；30 轮去头 2，看后 28 轮 EXEC_DYN 与首个 DEV_TASK_BUILD。
"""

import json
import multiprocessing as mp
import os
import re
import statistics
from typing import Dict, List, Tuple

import torch
import torch_npu

import pypto

# 冻结 shape：改任何一项都要重标定阈值。
_B = 16
_H = 8
_D = 128
_HIDDEN = 1024
_S2 = 2048
_S2_TILE = 256
_BLOCK = 128
_G_TILE = 2
_STITCH_FUNCTION_MAX_NUM = 64
_UNROLL_LIST = [8, 4, 2, 1]
_RUN_COUNT = 30
_SKIP_ROUNDS = 2
_EXPECTED_BUILD_PER_ROUND = 3

_MAX_BLOCKS = _S2 // _BLOCK
_KV_NUM_BLOCKS = _B * _MAX_BLOCKS
_BLOCK_NUM = _S2_TILE // _BLOCK

# 定标时 ENABLE_PERF_EVT 必须为 0（仓库默认）。这是编译期宏，打开后
# PerfBegin/End 会增加 ctrl 耗时，不能和 DUMP_DEVICE_PERF 看护一起用。
# 数据来自 0 卡 11 次压测（2026-09-09）：每次 30 轮、去掉前 2 轮、峰值先做 IQR 离群点过滤。
#   EXEC_DYN 均值 171.85~196.02us（最差 196.02），IQR 过滤后单轮最大最差 222.04
#   首个 DEV_TASK_BUILD 均值 93.70~107.77us（最差 107.77），IQR 过滤后单轮最大最差 124.66
# 门限 = 最差实测（过滤后）* 1.20，峰值门限按单轮波动大小微调（max_first * 1.28）。
_MEAN_EXEC_DYN_US = 235.0
_MEAN_FIRST_BUILD_US = 130.0
_MAX_EXEC_DYN_US = 266.0
_MAX_FIRST_BUILD_US = 160.0

# 单轮首个 DEV_TASK_BUILD 门禁超限提示：先排查当前修改是否引入 ctrl cpu 效率回退，再考虑环境波动。
_MAX_FIRST_GATE_HINT = (
    "max first DEV_TASK_BUILD 超限：请先确认当前修改是否影响 ctrl cpu 效率"
)

_BUILD_NAME = re.compile(r"^DEV_TASK_BUILD_(\d+)\((\d+)\)$")
_INIT_NAME = re.compile(r"^INIT_(\d+)$")


@pypto.frontend.jit(runtime_options={"stitch_function_max_num": _STITCH_FUNCTION_MAX_NUM})
def ctrl_perf_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    residual: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    gamma: pypto.Tensor([pypto.STATIC], pypto.DT_FP16),
    bias: pypto.Tensor([pypto.STATIC], pypto.DT_FP16),
    scale: pypto.Tensor([pypto.STATIC], pypto.DT_FP16),
    offset: pypto.Tensor([pypto.STATIC], pypto.DT_FP16),
    weight: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    q_tmp: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    k_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    v_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    block_table: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    act_seq: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    slot_mapping: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),
    unroll_list,
    eps,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    b = x.shape[0]
    n1 = _H
    mean_coff = 1.0 / float(_HIDDEN)
    softmax_scale = _D ** -0.5
    bs_tile = 4
    bs_loop = (b + bs_tile - 1) // bs_tile
    g_loop = n1 // _G_TILE
    kv_2d_shape = (k_cache.shape[0] * k_cache.shape[1], k_cache.shape[2] * k_cache.shape[3])

    pypto.set_vec_tile_shapes(1, _HIDDEN)
    gamma_2d = pypto.reshape(gamma, [1, _HIDDEN], inplace=True)
    bias_2d = pypto.reshape(bias, [1, _HIDDEN], inplace=True)
    scale_2d = pypto.reshape(scale, [1, _HIDDEN], inplace=True)
    offset_2d = pypto.reshape(offset, [1, _HIDDEN], inplace=True)
    k_cache_2d = pypto.reshape(k_cache, kv_2d_shape, inplace=True)
    v_cache_2d = pypto.reshape(v_cache, kv_2d_shape, inplace=True)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_PRE", idx_name="bs_idx"):
        act_bs = (b - bs_idx * bs_tile).min(bs_tile)
        x_tile = pypto.view(x, [bs_tile, _HIDDEN], [bs_idx * bs_tile, 0],
                            valid_shape=[act_bs, _HIDDEN])
        res_tile = pypto.view(residual, [bs_tile, _HIDDEN], [bs_idx * bs_tile, 0],
                              valid_shape=[act_bs, _HIDDEN])
        pypto.set_vec_tile_shapes(1, _HIDDEN)
        fused = pypto.add(pypto.cast(x_tile, pypto.DT_FP32), pypto.cast(res_tile, pypto.DT_FP32))
        square = pypto.mul(fused, fused)
        mean_res = pypto.mul(square, mean_coff)
        reduce_sum = pypto.add(pypto.sum(mean_res, -1, keepdim=True), eps)
        res_div = pypto.div(fused, pypto.sqrt(reduce_sum))
        gamma_fp32 = pypto.cast(gamma_2d, pypto.DT_FP32)
        bias_fp32 = pypto.cast(bias_2d, pypto.DT_FP32)
        norm = pypto.cast(pypto.add(pypto.mul(res_div, gamma_fp32), bias_fp32), pypto.DT_FP16)
        pypto.set_vec_tile_shapes(1, _HIDDEN)
        quant = pypto.add(pypto.mul(pypto.cast(norm, pypto.DT_FP32), pypto.cast(scale_2d, pypto.DT_FP32)),
                          pypto.cast(offset_2d, pypto.DT_FP32))
        pypto.set_cube_tile_shapes([32, 32], [128, 128], [128, 128])
        mm = pypto.matmul(pypto.cast(quant, pypto.DT_FP16), weight, pypto.DT_FP16)
        q_tmp[bs_idx * pypto.symbolic_scalar(bs_tile):, 0:] = mm

        b_ofs = bs_idx * bs_tile
        b_valid = (b - bs_idx * bs_tile).min(bs_tile)
        index_view = pypto.reshape(
            pypto.view(slot_mapping, [bs_tile], [b_ofs], valid_shape=[b_valid]),
            [bs_tile, 1], valid_shape=[b_valid, 1],
        )
        k_row = pypto.view(mm, [bs_tile, _D], [0, 0], valid_shape=[act_bs, _D])
        pypto.set_vec_tile_shapes(bs_tile, _D)
        k_cache.move(pypto.scatter_update(k_cache_2d, -2, index_view, k_row))
        v_cache.move(pypto.scatter_update(v_cache_2d, -2, index_view, k_row))

    q_2d = pypto.reshape(q_tmp, [b * n1, _D], inplace=True)
    pypto.set_pass_options(cube_l1_reuse_setting={0: 4})

    for b_idx in pypto.loop(b, name="LOOP_B", idx_name="b_idx"):
        cur_seq = act_seq[b_idx]
        s2_loop = (cur_seq + _S2_TILE - 1) // _S2_TILE
        for g_idx in pypto.loop(g_loop, name="LOOP_G", idx_name="g_idx"):
            oi_update = pypto.tensor([_G_TILE, _D], pypto.DT_FP32, "oi_update")
            sum_update = pypto.tensor([_G_TILE, 1], pypto.DT_FP32, "sum_update")
            max_update = pypto.tensor([_G_TILE, 1], pypto.DT_FP32, "max_update")
            for s2_idx in pypto.loop(s2_loop, name="LOOP_S2", idx_name="s2_idx", unroll_list=unroll_list):
                idx = s2_idx * _BLOCK_NUM
                n1g_ofs = g_idx * _G_TILE
                actual_s2 = (cur_seq - s2_idx * _S2_TILE).min(_S2_TILE)
                pypto.set_vec_tile_shapes(_G_TILE, _D)
                qi = pypto.view(q_2d, [_G_TILE, _D], [b_idx * n1 + n1g_ofs, 0])
                kj_assemble = pypto.tensor([_S2_TILE, _D], pypto.DT_FP16, "kj_assemble")
                vj_assemble = pypto.tensor([_S2_TILE, _D], pypto.DT_FP16, "vj_assemble")
                pypto.set_vec_tile_shapes(_BLOCK, _D)
                for i in range(_BLOCK_NUM):
                    block_idx = block_table[b_idx, idx + i].max(0)
                    kj_assemble[i * _BLOCK:(i + 1) * _BLOCK, 0:] = pypto.view(
                        k_cache_2d, [_BLOCK, _D], [block_idx * _BLOCK, 0]
                    )
                    vj_assemble[i * _BLOCK:(i + 1) * _BLOCK, 0:] = pypto.view(
                        v_cache_2d, [_BLOCK, _D], [block_idx * _BLOCK, 0]
                    )
                pypto.set_vec_tile_shapes(_G_TILE, _D)
                kj_assemble = pypto.view(kj_assemble, [_S2_TILE, _D], [0, 0],
                                         valid_shape=[actual_s2, _D])
                vj_assemble = pypto.view(vj_assemble, [_S2_TILE, _D], [0, 0],
                                         valid_shape=[actual_s2, _D])
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                sij = pypto.view(
                    pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True),
                    [_G_TILE, _S2_TILE], [0, 0], valid_shape=[_G_TILE, actual_s2],
                )
                pypto.set_vec_tile_shapes(_G_TILE, _S2_TILE)
                if pypto.is_loop_begin(s2_idx):
                    pypto.set_pass_options(sg_set_scope=3)
                    sij_scale = pypto.mul(sij, softmax_scale)
                    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                    tilda_pij = pypto.exp(pypto.sub(sij_scale, tilda_mij))
                    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                    max_update[:] = tilda_mij
                    pypto.set_pass_options(sg_set_scope=-1)
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    oi_update[:] = pypto.matmul(pypto.cast(tilda_pij, pypto.DT_FP16), vj_assemble, pypto.DT_FP32)
                else:
                    pypto.set_pass_options(sg_set_scope=1)
                    sij_scale = pypto.mul(sij, softmax_scale)
                    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                    max_new = pypto.maximum(max_update, tilda_mij)
                    tilda_pij = pypto.exp(pypto.sub(sij_scale, max_new))
                    sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                    pypto.set_pass_options(sg_set_scope=-1)
                    pypto.set_pass_options(sg_set_scope=2)
                    update_mul = pypto.exp(pypto.sub(max_update, max_new))
                    max_update[:] = max_new
                    sum_update[:] = sum_update * update_mul + sum_local
                    pypto.set_pass_options(sg_set_scope=-1)
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    oi_tmp = pypto.matmul(pypto.cast(tilda_pij, pypto.DT_FP16), vj_assemble, pypto.DT_FP32)
                    pypto.set_vec_tile_shapes(_G_TILE, _D)
                    oi_update[:] = oi_update * update_mul + oi_tmp
            pypto.set_vec_tile_shapes(_G_TILE, _D)
            o_norm = pypto.div(oi_update, sum_update)
            pypto.assemble(pypto.cast(o_norm, pypto.DT_FP16), [b_idx * n1 + g_idx * _G_TILE, 0], atten_out)


def _make_block_table(device):
    table = torch.zeros((_B, _MAX_BLOCKS), dtype=torch.int32, device=device)
    for b in range(_B):
        table[b] = torch.arange(b * _MAX_BLOCKS, (b + 1) * _MAX_BLOCKS, dtype=torch.int32, device=device)
    return table


def _device_run_ctrl_perf(queue):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    os.environ["DUMP_DEVICE_PERF"] = "true"
    device = f"npu:{device_id}"

    x = torch.rand(_B, _HIDDEN, dtype=torch.float16, device=device)
    residual = torch.rand(_B, _HIDDEN, dtype=torch.float16, device=device)
    gamma = torch.rand(_HIDDEN, dtype=torch.float16, device=device)
    bias = torch.rand(_HIDDEN, dtype=torch.float16, device=device)
    scale = torch.rand(_HIDDEN, dtype=torch.float16, device=device)
    offset = torch.rand(_HIDDEN, dtype=torch.float16, device=device)
    weight = torch.rand(_HIDDEN, _H * _D, dtype=torch.float16, device=device)
    q_tmp = torch.zeros(_B, _H * _D, dtype=torch.float16, device=device)
    k_cache = torch.rand(_KV_NUM_BLOCKS, _BLOCK, 1, _D, dtype=torch.float16, device=device)
    v_cache = torch.rand(_KV_NUM_BLOCKS, _BLOCK, 1, _D, dtype=torch.float16, device=device)
    block_table = _make_block_table(device)
    act_seq = torch.full((_B,), _S2, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(_B, dtype=torch.int32, device=device)
    atten_out = torch.zeros(_B * _H, _D, dtype=torch.float16, device=device)

    for _ in range(_RUN_COUNT):
        ctrl_perf_kernel(
            x, residual, gamma, bias, scale, offset, weight, q_tmp,
            k_cache, v_cache, block_table, act_seq, slot_mapping, atten_out,
            _UNROLL_LIST, 1e-5,
        )

    torch_npu.npu.synchronize()
    queue.put(pypto.pypto_impl.LogTopFolder())


def _load_trace_events(pref_path: str) -> List[Dict]:
    chrome_path = os.path.join(pref_path, "machine_runtime_operator_trace_0.json")
    raw_path = os.path.join(pref_path, "machine_trace_perf_data_0.json")
    if os.path.exists(chrome_path):
        with open(chrome_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "traceEvents" in data:
            return data["traceEvents"]
        if isinstance(data, list):
            return data
    assert os.path.exists(raw_path), f"missing dump json under {pref_path}"
    with open(raw_path, "r", encoding="utf-8") as f:
        cores = json.load(f)
    return _raw_ctrl_to_chrome_events(cores)


def _raw_ctrl_to_chrome_events(cores: List[Dict]) -> List[Dict]:
    ctrl = next((c for c in cores if str(c.get("coreType")) == "AICPU-CTRL"), None)
    assert ctrl is not None, "raw dump missing AICPU-CTRL"
    freq = float(ctrl.get("freq", 0)) or 1.0
    tasks = sorted(ctrl.get("tasks", []), key=lambda t: t.get("end", 0))
    events: List[Dict] = []
    prev_end = None
    for task in tasks:
        name = str(task.get("name", "UNKNOWN"))
        end_time = float(task.get("end", 0))
        start_time = (end_time - 1.0) if name.startswith("BEGIN") or prev_end is None else prev_end
        dur = (end_time - start_time) / freq
        events.append({"name": name, "cat": "AICPU-CTRL", "ph": "X", "dur": dur})
        prev_end = end_time
    return events


def _parse_ctrl_rounds(events: List[Dict]) -> Dict[int, Dict]:
    rounds: Dict[int, Dict] = {}

    def bucket(round_id: int) -> Dict:
        if round_id not in rounds:
            rounds[round_id] = {"init": [], "build": []}
        return rounds[round_id]

    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "AICPU-CTRL":
            continue
        name = str(event.get("name", ""))
        dur = float(event.get("dur", 0.0))
        init_m = _INIT_NAME.match(name)
        if init_m:
            bucket(int(init_m.group(1)))["init"].append(dur)
            continue
        build_m = _BUILD_NAME.match(name)
        if build_m:
            bucket(int(build_m.group(1)))["build"].append((int(build_m.group(2)), dur))
    return rounds


def _round_metrics(rounds: Dict[int, Dict]) -> List[Tuple[int, int, float, float]]:
    metrics = []
    for round_id in sorted(rounds):
        data = rounds[round_id]
        builds = sorted(data["build"], key=lambda x: x[0])
        init_us = data["init"][0] if data["init"] else 0.0
        first_build = builds[0][1] if builds else 0.0
        exec_dyn = init_us + sum(item[1] for item in builds)
        metrics.append((round_id, len(builds), exec_dyn, first_build))
    return metrics


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _filter_outliers_iqr(values: List[float], factor: float = 1.5) -> List[float]:
    """使用 IQR 方法剔除统计离群点，返回 [Q1 - factor*IQR, Q3 + factor*IQR] 范围内的值。

    标准因子 1.5 可剔除温和离群点，若环境波动较大可改用 2.0 或 3.0。
    """
    if len(values) < 4:
        return values
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    iqr = q3 - q1
    if iqr == 0.0:
        return values
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    n_before = len(values)
    filtered = [v for v in values if lower <= v <= upper]
    n_removed = n_before - len(filtered)
    if n_removed:
        print(f"  IQR outlier filter: removed {n_removed}/{n_before} values "
              f"(q1={q1:.2f}, q3={q3:.2f}, iqr={iqr:.2f}, bounds=[{lower:.2f}, {upper:.2f}])")
    return filtered


@pypto.options(pass_options={"enable_slice": True})
def test_ctrl_cpu_perf():
    """看护 AICPU-CTRL EXEC_DYN 与首个 DEV_TASK_BUILD（30 轮去头 2）。"""
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    proc = mp.Process(target=_device_run_ctrl_perf, args=(result_queue,))
    proc.start()
    proc.join()
    assert proc.exitcode == 0, f"ctrl perf subprocess failed, exitcode={proc.exitcode}"
    assert not result_queue.empty(), "Could not get perf output path"

    pref_path = result_queue.get()
    events = _load_trace_events(pref_path)
    metrics = _round_metrics(_parse_ctrl_rounds(events))
    assert len(metrics) >= _RUN_COUNT, (
        f"expected >= {_RUN_COUNT} CTRL rounds, got {len(metrics)} under {pref_path}"
    )

    for round_id, n_build, exec_dyn, first_build in metrics[:_RUN_COUNT]:
        print(f"  round={round_id} n_build={n_build} exec={exec_dyn:.2f} first={first_build:.2f}")
        if _EXPECTED_BUILD_PER_ROUND > 0:
            assert n_build == _EXPECTED_BUILD_PER_ROUND, (
                f"round={round_id}: DEV_TASK_BUILD count {n_build} != {_EXPECTED_BUILD_PER_ROUND}"
            )
        else:
            assert n_build >= 2, f"round={round_id}: need >=2 DEV_TASK_BUILD, got {n_build}"

    steady = metrics[_SKIP_ROUNDS:_RUN_COUNT]
    assert len(steady) == _RUN_COUNT - _SKIP_ROUNDS
    exec_vals = [item[2] for item in steady]
    first_vals = [item[3] for item in steady]
    mean_exec = _mean(exec_vals)
    mean_first = _mean(first_vals)
    # 对峰值做 IQR 离群点过滤，剔除环境波动引入的异常尖峰后再取最大值。
    # 均值看护（mean_exec / mean_first）已经足够稳定，不参与过滤。
    filtered_exec = _filter_outliers_iqr(exec_vals)
    filtered_first = _filter_outliers_iqr(first_vals)
    max_exec = max(filtered_exec) if filtered_exec else max(exec_vals)
    max_first = max(filtered_first) if filtered_first else max(first_vals)
    std_exec = statistics.pstdev(exec_vals) if len(exec_vals) > 1 else 0.0
    std_first = statistics.pstdev(first_vals) if len(first_vals) > 1 else 0.0
    print(
        "ctrl_cpu_perf calib: "
        f"n={len(steady)} n_build={steady[0][1]} "
        f"mean_exec={mean_exec:.2f} std_exec={std_exec:.2f} max_exec={max_exec:.2f} "
        f"(raw_max={max(exec_vals):.2f}) "
        f"mean_first={mean_first:.2f} std_first={std_first:.2f} max_first={max_first:.2f} "
        f"(raw_max={max(first_vals):.2f})"
    )

    assert mean_exec <= _MEAN_EXEC_DYN_US, (
        f"mean EXEC_DYN {mean_exec:.2f} us > gate {_MEAN_EXEC_DYN_US:.2f} us"
    )
    assert mean_first <= _MEAN_FIRST_BUILD_US, (
        f"mean first DEV_TASK_BUILD {mean_first:.2f} us > gate {_MEAN_FIRST_BUILD_US:.2f} us"
    )
    assert max_exec <= _MAX_EXEC_DYN_US, (
        f"max EXEC_DYN {max_exec:.2f} us > gate {_MAX_EXEC_DYN_US:.2f} us"
    )
    assert max_first <= _MAX_FIRST_BUILD_US, (
        f"max first DEV_TASK_BUILD {max_first:.2f} us > gate {_MAX_FIRST_BUILD_US:.2f} us. {_MAX_FIRST_GATE_HINT}"
    )
