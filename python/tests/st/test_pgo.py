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
PGO (Profiling Guided Optimization) 数据导出看护。

用例构造一个 PyPTO 算子（与 test.py 一致：1024x1024 fp32、32x32 tile、(A @ A + A) @ A + A），
并通过 generic_options 打开 PGO：`experimental_pgo_path=<tmp>/pgo`、`experimental_pgo_update=True`，
同时打开调试开关（`compile_debug_mode=1`、`runtime_debug_mode=1`）。

期望结果：launch 同步完成（PgoDump 的调用点）之后，`<tmp>/pgo` 目录下按 prof 类型各生成一个 json：

    <tmp>/pgo/per_core.json      // per-core prof 明细
    <tmp>/pgo/aicpu.json         // aicpu prof 明细

注意：PGO 数据来自 FedProf 记录，需要使能 prof 记录（DUMP_DEVICE_PERF=true / runtime debug），
否则对应的 prof 数据未分配、不会产出对应文件。
"""

import json
import os
import tempfile
from typing import Dict, List

import pytest
import torch
import torch_npu

import pypto

# 与 test.py 保持一致的问题规模与 tile 配置（必须编译期静态）
_N = 64
_TILE = 32
_NT = _N // _TILE

# PGO 数据按 prof 类型落盘的文件名，来自 C++ 侧 fed_prof_name_list
_PGO_KIND_FILES = ("per_core.json", "aicpu.json")


@pypto.options(pass_options={"enable_slice": True})
@pytest.mark.soc("910", "950")
def test_pgo_update_dump():
    """pgo_update=True 时，launch 同步后应在 <tmp>/pgo 下产出 prof 数据文件。"""
    with tempfile.TemporaryDirectory(prefix="pypto_pgo_") as tmp_dir:
        pgo_dir = os.path.join(tmp_dir, "pgo")

        @pypto.frontend.jit(
            debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1},
            generic_options={"experimental_pgo_path": pgo_dir, "experimental_pgo_update": True},
        )
        def matmul_add_kernel_npu(
            x: pypto.Tensor([_N, _N], pypto.DT_FP32),
            y: pypto.Tensor([_N, _N], pypto.DT_FP32),
            z: pypto.Tensor([_N, _N], pypto.DT_FP32),
            out: pypto.Tensor([_N, _N], pypto.DT_FP32),
        ):
            """out = x @ y + z，按 32x32 分块计算，K 方向在 kernel 内累加。"""
            pypto.set_cube_tile_shapes([_TILE, _TILE], [_TILE, _TILE], [_TILE, _TILE])
            pypto.set_vec_tile_shapes(_TILE, _TILE)

            for i in pypto.loop(_NT, name="LOOP_M", idx_name="i"):
                for j in pypto.loop(_NT, name="LOOP_N", idx_name="j"):
                    acc = pypto.tensor([_TILE, _TILE], pypto.DT_FP32, "acc")
                    for k in pypto.loop(_NT, name="LOOP_K", idx_name="k"):
                        x_tile = pypto.view(x, [_TILE, _TILE], [i * _TILE, k * _TILE])
                        y_tile = pypto.view(y, [_TILE, _TILE], [k * _TILE, j * _TILE])
                        prod = pypto.matmul(x_tile, y_tile, pypto.DT_FP32)
                        if pypto.is_loop_begin(k):
                            acc[:] = prod
                        else:
                            acc[:] = acc + prod
                    z_tile = pypto.view(z, [_TILE, _TILE], [i * _TILE, j * _TILE])
                    pypto.assemble(acc + z_tile, [i * _TILE, j * _TILE], out)

        device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        # PGO 数据来自 FedProf 记录，需要使能 prof 记录（必须在首次 launch 前设置）
        os.environ["DUMP_DEVICE_PERF"] = "true"

        a = torch.rand((_N, _N), dtype=torch.float32, device=f"npu:{device_id}")
        tmp = torch.empty((_N, _N), dtype=torch.float32, device=f"npu:{device_id}")
        out = torch.empty((_N, _N), dtype=torch.float32, device=f"npu:{device_id}")

        matmul_add_kernel_npu(a, a, a, tmp)      # tmp = A @ A + A
        matmul_add_kernel_npu(tmp, a, a, out)    # out = tmp @ A + A
        torch_npu.npu.synchronize()

        assert os.path.isdir(pgo_dir), f"PGO 目录未创建: {pgo_dir}"
        for name in _PGO_KIND_FILES:
            path = os.path.join(pgo_dir, name)
            assert os.path.exists(path), f"缺少 PGO 数据文件: {path}（目录内容: {os.listdir(pgo_dir)}）"
            with open(path, "r", encoding="utf-8") as f:
                data: List[Dict] = json.load(f)
            assert isinstance(data, list), f"{path} 应为 json 数组"
            assert len(data) > 0, f"{path} 内容为空"
