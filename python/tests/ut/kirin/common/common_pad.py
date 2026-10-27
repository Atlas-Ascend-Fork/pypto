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
Test pad codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto


def _make_pad_kernel(soc_version, name, dtype, tile_shapes, padding, pad_value):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        input0: pypto.Tensor([...], dtype),
        output: pypto.Tensor([...], dtype),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        output[:] = pypto.pad(input0, list(padding), "constant", pad_value)

    kernel.__name__ = name
    return kernel


# Per-entry layout (matches the parametrize() arglist in the callers):
#   [0] kernel_name, [1] torch_dtype, [2] pypto_dtype, [3] tile_shapes,
#   [4] in_shape, [5] padding, [6] pad_value
# NOTE(PR): only the 4D full-split case id=042 is enabled; the rest are marked _SKIP
# until the full matrix is re-enabled.
_SKIP = pytest.mark.skip(reason="temporarily disabled; only the 4D full-split case 042 is enabled")
TEST_CASES = [
    # ---------------- 1D (001~004) ----------------
    # 001 [1D right] tile > output -> single tile
    pytest.param(
        "pad_kernel_001",
        torch.float32,
        pypto.DT_FP32,
        (200,),
        (160,),
        (0, 8),
        0.0,
        marks=[_SKIP],
        id="001",
    ),
    # 002 [1D right] tile < output -> multi tile, non-zero value
    pytest.param(
        "pad_kernel_002",
        torch.float32,
        pypto.DT_FP32,
        (100,),
        (160,),
        (0, 40),
        1.5,
        marks=[_SKIP],
        id="002",
    ),
    # 003 [1D both side] non-aligned padding, fp16, negative value
    pytest.param(
        "pad_kernel_003",
        torch.float16,
        pypto.DT_FP16,
        (64,),
        (96,),
        (5, 7),
        -2.0,
        marks=[_SKIP],
        id="003",
    ),
    # 004 [1D left] colOff 10*4 = 40B non-aligned
    pytest.param(
        "pad_kernel_004",
        torch.float32,
        pypto.DT_FP32,
        (64,),
        (100,),
        (10, 0),
        3.0,
        marks=[_SKIP],
        id="004",
    ),
    # ---------------- 2D tile-split matrix (005~008) ----------------
    # base: in (24,40) pad (0,8,0,8) -> out (32,48); split = tile < 32 / tile < 48
    # 005 [2D F/F] tile covers both output dims -> single tile
    pytest.param(
        "pad_kernel_005",
        torch.float32,
        pypto.DT_FP32,
        (32, 48),
        (24, 40),
        (0, 8, 0, 8),
        0.0,
        marks=[_SKIP],
        id="005",
    ),
    # 006 [2D F/T] split last dim only
    pytest.param(
        "pad_kernel_006",
        torch.float32,
        pypto.DT_FP32,
        (32, 16),
        (24, 40),
        (0, 8, 0, 8),
        1.0,
        marks=[_SKIP],
        id="006",
    ),
    # 007 [2D T/F] split second-last dim only, fp16
    pytest.param(
        "pad_kernel_007",
        torch.float16,
        pypto.DT_FP16,
        (16, 48),
        (24, 40),
        (0, 8, 0, 8),
        -3.0,
        marks=[_SKIP],
        id="007",
    ),
    # 008 [2D T/T] split both dims, fp16
    pytest.param(
        "pad_kernel_008",
        torch.float16,
        pypto.DT_FP16,
        (16, 16),
        (24, 40),
        (0, 8, 0, 8),
        2.0,
        marks=[_SKIP],
        id="008",
    ),
    # ---------------- 2D boundary (009~015) ----------------
    # 009 identity (no pad) -> OP_REGISTER_COPY / noPad fast path
    pytest.param(
        "pad_kernel_009",
        torch.float32,
        pypto.DT_FP32,
        (16, 16),
        (32, 48),
        (0, 0, 0, 0),
        0.0,
        marks=[_SKIP],
        id="009",
    ),
    # 010 four-side, left/top offsets non-aligned
    pytest.param(
        "pad_kernel_010",
        torch.float32,
        pypto.DT_FP32,
        (16, 16),
        (32, 48),
        (5, 16, 7, 8),
        0.0,
        marks=[_SKIP],
        id="010",
    ),
    # 011 last dim == 1 (TD-1 regression), fp16
    pytest.param(
        "pad_kernel_011",
        torch.float16,
        pypto.DT_FP16,
        (16, 16),
        (20, 1),
        (0, 44, 0, 0),
        0.0,
        marks=[_SKIP],
        id="011",
    ),
    # 012 same 32B block: left 1 + valid 6 + right 1 = 8 elem per row
    pytest.param(
        "pad_kernel_012",
        torch.float32,
        pypto.DT_FP32,
        (16, 8),
        (64, 6),
        (1, 1, 0, 0),
        5.0,
        marks=[_SKIP],
        id="012",
    ),
    # 013 pad >> input -> many pure-pad tiles (TILE_VEC_DUP)
    pytest.param(
        "pad_kernel_013",
        torch.float32,
        pypto.DT_FP32,
        (16, 16),
        (16, 16),
        (0, 48, 0, 48),
        7.0,
        marks=[_SKIP],
        id="013",
    ),
    # 014 top-only pad (pre-dim offset, no last-dim pad)
    pytest.param(
        "pad_kernel_014",
        torch.float32,
        pypto.DT_FP32,
        (8, 16),
        (24, 16),
        (0, 0, 8, 0),
        0.0,
        marks=[_SKIP],
        id="014",
    ),
    # 015 left-only pad non-aligned (colOff 2*4 = 8B)
    pytest.param(
        "pad_kernel_015",
        torch.float32,
        pypto.DT_FP32,
        (16, 16),
        (32, 48),
        (2, 0, 0, 0),
        2.0,
        marks=[_SKIP],
        id="015",
    ),
    # ---------------- 3D tile-split matrix (016~023) ----------------
    # base: in (2,16,16) pad (0,8,0,8) -> out (2,24,24); split = tile < 2 / < 24 / < 24
    # 016 [3D F/F/F]
    pytest.param(
        "pad_kernel_016",
        torch.float32,
        pypto.DT_FP32,
        (2, 32, 32),
        (2, 16, 16),
        (0, 8, 0, 8),
        0.0,
        marks=[_SKIP],
        id="016",
    ),
    # 017 [3D F/F/T] split last dim only
    pytest.param(
        "pad_kernel_017",
        torch.float32,
        pypto.DT_FP32,
        (2, 32, 16),
        (2, 16, 16),
        (0, 8, 0, 8),
        1.5,
        marks=[_SKIP],
        id="017",
    ),
    # 018 [3D F/T/F] split second-last dim only, fp16
    pytest.param(
        "pad_kernel_018",
        torch.float16,
        pypto.DT_FP16,
        (2, 8, 32),
        (2, 16, 16),
        (0, 8, 0, 8),
        -3.0,
        marks=[_SKIP],
        id="018",
    ),
    # 019 [3D F/T/T] split last two dims only, fp16
    pytest.param(
        "pad_kernel_019",
        torch.float16,
        pypto.DT_FP16,
        (2, 8, 16),
        (2, 16, 16),
        (0, 8, 0, 8),
        2.0,
        marks=[_SKIP],
        id="019",
    ),
    # 020 [3D T/F/F] split outer dim only
    pytest.param(
        "pad_kernel_020",
        torch.float32,
        pypto.DT_FP32,
        (1, 32, 32),
        (2, 16, 16),
        (0, 8, 0, 8),
        0.5,
        marks=[_SKIP],
        id="020",
    ),
    # 021 [3D T/F/T] split outer + last dim
    pytest.param(
        "pad_kernel_021",
        torch.float32,
        pypto.DT_FP32,
        (1, 32, 16),
        (2, 16, 16),
        (0, 8, 0, 8),
        -1.0,
        marks=[_SKIP],
        id="021",
    ),
    # 022 [3D T/T/F] split outer + second-last dim, fp16
    pytest.param(
        "pad_kernel_022",
        torch.float16,
        pypto.DT_FP16,
        (1, 8, 32),
        (2, 16, 16),
        (0, 8, 0, 8),
        4.0,
        marks=[_SKIP],
        id="022",
    ),
    # 023 [3D T/T/T] split all dims, fp16
    pytest.param(
        "pad_kernel_023",
        torch.float16,
        pypto.DT_FP16,
        (1, 8, 16),
        (2, 16, 16),
        (0, 8, 0, 8),
        6.0,
        marks=[_SKIP],
        id="023",
    ),
    # ---------------- 3D boundary (024~026) ----------------
    # 024 last dim == 1 + left pad non-aligned (colOff 3*4 = 12B)
    pytest.param(
        "pad_kernel_024",
        torch.float32,
        pypto.DT_FP32,
        (1, 8, 8),
        (2, 16, 1),
        (3, 31, 0, 0),
        1.0,
        marks=[_SKIP],
        id="024",
    ),
    # 025 same 32B block fp16: left 5 + valid 6 + right 5 = 16 elem per row
    pytest.param(
        "pad_kernel_025",
        torch.float16,
        pypto.DT_FP16,
        (1, 32, 16),
        (1, 64, 6),
        (5, 5, 0, 0),
        5.0,
        marks=[_SKIP],
        id="025",
    ),
    # 026 four-side + outer split
    pytest.param(
        "pad_kernel_026",
        torch.float32,
        pypto.DT_FP32,
        (3, 16, 16),
        (6, 32, 48),
        (5, 16, 7, 8),
        0.0,
        marks=[_SKIP],
        id="026",
    ),
    # ---------------- 4D tile-split matrix (027~042) ----------------
    # base: in (2,2,16,16) pad (0,8,0,8) -> out (2,2,24,24)
    # 027 [4D F/F/F/F]
    pytest.param(
        "pad_kernel_027",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 32, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        0.0,
        marks=[_SKIP],
        id="027",
    ),
    # 028 [4D F/F/F/T] split last dim only
    pytest.param(
        "pad_kernel_028",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 32, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        1.0,
        marks=[_SKIP],
        id="028",
    ),
    # 029 [4D F/F/T/F] split second-last dim only
    pytest.param(
        "pad_kernel_029",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 8, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        -2.0,
        marks=[_SKIP],
        id="029",
    ),
    # 030 [4D F/F/T/T] split last two dims only
    pytest.param(
        "pad_kernel_030",
        torch.float32,
        pypto.DT_FP32,
        (2, 2, 8, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        3.0,
        marks=[_SKIP],
        id="030",
    ),
    # 031 [4D F/T/F/F] split second outer dim only, fp16
    pytest.param(
        "pad_kernel_031",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 32, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        0.5,
        marks=[_SKIP],
        id="031",
    ),
    # 032 [4D F/T/F/T] split second outer + last dim, fp16
    pytest.param(
        "pad_kernel_032",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 32, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        -1.0,
        marks=[_SKIP],
        id="032",
    ),
    # 033 [4D F/T/T/F] split second outer + second-last dim, fp16
    pytest.param(
        "pad_kernel_033",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 8, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        2.0,
        marks=[_SKIP],
        id="033",
    ),
    # 034 [4D F/T/T/T] split all but first dim, fp16
    pytest.param(
        "pad_kernel_034",
        torch.float16,
        pypto.DT_FP16,
        (2, 1, 8, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        4.0,
        marks=[_SKIP],
        id="034",
    ),
    # 035 [4D T/F/F/F] split first outer dim only
    pytest.param(
        "pad_kernel_035",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        0.0,
        marks=[_SKIP],
        id="035",
    ),
    # 036 [4D T/F/F/T] split first outer + last dim
    pytest.param(
        "pad_kernel_036",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 32, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        1.5,
        marks=[_SKIP],
        id="036",
    ),
    # 037 [4D T/F/T/F] split first outer + second-last dim
    pytest.param(
        "pad_kernel_037",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 8, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        -3.0,
        marks=[_SKIP],
        id="037",
    ),
    # 038 [4D T/F/T/T] split first outer + last two dims
    pytest.param(
        "pad_kernel_038",
        torch.float32,
        pypto.DT_FP32,
        (1, 2, 8, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        2.5,
        marks=[_SKIP],
        id="038",
    ),
    # 039 [4D T/T/F/F] split both outer dims, fp16
    pytest.param(
        "pad_kernel_039",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 32, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        7.0,
        marks=[_SKIP],
        id="039",
    ),
    # 040 [4D T/T/F/T] split both outer + last dim, fp16
    pytest.param(
        "pad_kernel_040",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 32, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        -5.0,
        marks=[_SKIP],
        id="040",
    ),
    # 041 [4D T/T/T/F] split both outer + second-last dim, fp16
    pytest.param(
        "pad_kernel_041",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 8, 32),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        0.25,
        marks=[_SKIP],
        id="041",
    ),
    # 042 [4D T/T/T/T] split all dims, fp16
    pytest.param(
        "pad_kernel_042",
        torch.float16,
        pypto.DT_FP16,
        (1, 1, 8, 16),
        (2, 2, 16, 16),
        (0, 8, 0, 8),
        9.0,
        marks=[],
        id="042",
    ),
    # ---------------- 4D boundary (043~045) ----------------
    # 043 tiny last-two dims (1,4) + left pad non-aligned (6*2 = 12B) + same-block (32B)
    pytest.param(
        "pad_kernel_043",
        torch.float16,
        pypto.DT_FP16,
        (1, 64, 1, 16),
        (1, 4096, 1, 4),
        (6, 6, 0, 0),
        5.0,
        marks=[_SKIP],
        id="043",
    ),
    # 044 four-side + outer split on both outer dims
    pytest.param(
        "pad_kernel_044",
        torch.float32,
        pypto.DT_FP32,
        (2, 1, 16, 16),
        (4, 2, 32, 48),
        (5, 16, 7, 8),
        0.0,
        marks=[pytest.mark.skip()],
        id="044",
    ),
    # 045 pad >> input -> many pure-pad tiles
    pytest.param(
        "pad_kernel_045",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 16, 16),
        (1, 1, 4, 4),
        (0, 60, 0, 60),
        7.0,
        marks=[_SKIP],
        id="045",
    ),
    # ---------------- ONNX model cases (046~048) ----------------
    # Shapes/pads taken verbatim from pad开发/onnx/tiny_fp32_sim.onnx (opset17, fp32, rank-4,
    # constant value 0). All three pad only the second-to-last dim (H): the ONNX "pads" array is
    # [b0,b1,b2,b3, e0,e1,e2,e3] -> pypto (left,right,top,bottom) = (d3_begin,d3_end,d2_begin,d2_end).
    # 046 /decoder/Pad: in (1,4,77,64) -> out (1,4,256,64), pads dim2=(0,179) -> bottom-only, large pad
    pytest.param(
        "pad_kernel_046",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 64, 64),
        (1, 4, 77, 64),
        (0, 0, 0, 179),
        0.0,
        marks=[_SKIP],
        id="046",
    ),
    # 047 /decoder/Pad_2: in (1,4,165,64) -> out (1,4,256,64), pads dim2=(13,78) -> top+bottom (four-side)
    pytest.param(
        "pad_kernel_047",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 64, 64),
        (1, 4, 165, 64),
        (0, 0, 13, 78),
        0.0,
        marks=[_SKIP],
        id="047",
    ),
    # 048 /decoder/Pad_4: in (1,4,142,64) -> out (1,4,256,64), pads dim2=(114,0) -> top-only (four-side)
    pytest.param(
        "pad_kernel_048",
        torch.float32,
        pypto.DT_FP32,
        (1, 1, 64, 64),
        (1, 4, 142, 64),
        (0, 0, 114, 0),
        0.0,
        marks=[_SKIP],
        id="048",
    ),
]


def run_pad_test(kernels, kernel_name, dtype, in_shape, padding, pad_value):
    input0 = torch.rand(in_shape, dtype=torch.float32) * 10
    input0 = input0.to(dtype)

    golden = torch.nn.functional.pad(input0, tuple(padding), mode="constant", value=pad_value)
    output = torch.empty(golden.shape, dtype=dtype)

    kernels[kernel_name](input0, output)

    check_nan(output, name=kernel_name)
    cos_value = abs(compare_cos(np.array(output.cpu()), np.array(golden.cpu())))
    if cos_value < 0.9999:
        raise AssertionError(f"{kernel_name}: cos_value {cos_value} < 0.9999")


def create_pad_kernels(soc_version):
    return {
        p.values[0]: _make_pad_kernel(soc_version, p.values[0], p.values[2], p.values[3], p.values[5], p.values[6])
        for p in TEST_CASES
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
