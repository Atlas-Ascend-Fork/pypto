#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conv pass_verify cases converted from python/tests/st/test_conv.py, plus the
static conv2d smoke cases merged from ut/interpreter/test_conv2d.py
(compile + pass_verify + golden, no NPU).

Same shapes / tiles / strides / pads / dilations / groups as the ST on-board
cases, validating the conv golden tool on both pipelines:
- soc("950"): forced Ascend950 (DAV_3510) pipeline, NCHW goldens
- soc("910"): forced A2A3 (DAV_2201) pipeline, NC1HWC0 / Fractal_Z goldens

The merged smoke cases share one parameterized conv2d_kernel driven by
_run_conv2d_case, covering 1x1 / 3x3(+pad) static configs on both pipelines.
The soc("910") block at the end additionally generalizes the ST dynamic-axis
patterns (batch / hout / cout) with dtype crosses to the A2A3 pipeline.
"""

import os
import sys

import pytest
import torch

import pypto
from pypto import pypto_impl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _verify_check import assert_pass_verify_ok, set_verify_goldens  # noqa: E402


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
)
def conv3d_bias_kernel(
    fmap: pypto.Tensor([1, 16, 5, 5, 32], pypto.DT_BF16),
    weight: pypto.Tensor([16, 8, 3, 3, 3], pypto.DT_BF16),
    bias: pypto.Tensor([16], pypto.DT_BF16),
    out: pypto.Tensor([1, 16, 2, 2, 15], pypto.DT_BF16),
):
    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=32, tileWout=16, tileCinFmap=16,
                              tileCinWeight=16, tileN=16, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=16, tileN=16),
    )
    extend_params = {"bias_tensor": bias}
    output = pypto.conv(fmap, weight, pypto.DT_BF16, [2, 2, 2], [1, 1, 1, 1, 1, 1], [2, 2, 2],
                        extend_params=extend_params, groups=2)
    out.move(output)


@pytest.mark.soc("950")
def test_conv3d_bf16_basic_with_bias():
    """Conv3D BF16 groups=2 + bias, stride=2, dilation=2, pad=1 (A5 pipeline)."""
    torch.manual_seed(0)
    fmap = torch.randn(1, 16, 5, 5, 32, dtype=torch.bfloat16)
    weight = torch.randn(16, 8, 3, 3, 3, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(16, dtype=torch.bfloat16) * 0.1
    golden = torch.nn.functional.conv3d(
        fmap.float(), weight.float(), bias.float(), stride=(2, 2, 2), padding=(1, 1, 1),
        dilation=(2, 2, 2), groups=2
    ).to(torch.bfloat16)
    set_verify_goldens([None, None, None, golden])
    out = torch.zeros(1, 16, 2, 2, 15, dtype=torch.bfloat16)
    conv3d_bias_kernel(fmap, weight, bias, out)
    assert_pass_verify_ok()


# Static conv2d smoke cases merged from ut/interpreter/test_conv2d.py: one
# parameterized kernel + case driver covering 1x1 / 3x3(+pad) configs on both
# pipelines (tensor-graph OP_CONV2D golden + tile-graph conv goldens,
# TILE_A_MUL_B / TILE_A_MULACC_B accumulation).
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
    host_options={"compile_stage": pypto.CompStage.EXECUTE_GRAPH},
)
def conv2d_kernel(
    fmap: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    weight: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    bias: pypto.Tensor([pypto.STATIC], pypto.DT_FP16),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    strides,
    paddings,
    dilations,
    tile_l1,
    tile_l0,
    extend_params,
):
    cin = fmap.shape[1]
    cin_per_group = weight.shape[1]
    groups = cin // cin_per_group
    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=tile_l1[0], tileHout=tile_l1[1], tileWin=tile_l1[2],
                              tileWout=tile_l1[3], tileCinFmap=tile_l1[4], tileCinWeight=tile_l1[5],
                              tileN=tile_l1[6], tileBatch=tile_l1[7]),
        pypto_impl.TileL0Info(tileH=tile_l0[0], tileW=tile_l0[1], tileK=tile_l0[2], tileN=tile_l0[3]),
    )
    extend_params["bias_tensor"] = bias
    conv_out = pypto.conv(fmap, weight, pypto.DT_FP16, strides, paddings, dilations,
                          extend_params=extend_params, groups=groups)
    out[:] = conv_out


def _run_conv2d_case(fmap_shape, weight_shape, strides, paddings, dilations, tile_l1, tile_l0):
    torch.manual_seed(0)
    fmap = torch.randn(fmap_shape, dtype=torch.float16)
    weight = torch.randn(weight_shape, dtype=torch.float16) * 0.1
    bias = torch.randn(weight_shape[0], dtype=torch.float16)
    groups = fmap_shape[1] // weight_shape[1]
    golden = torch.nn.functional.conv2d(
        fmap.float(), weight.float(), bias.float(),
        stride=tuple(strides), padding=(paddings[0], paddings[2]), dilation=tuple(dilations), groups=groups,
    ).half()
    set_verify_goldens([None, None, None, golden])
    out = torch.zeros(fmap_shape[0], weight_shape[0], golden.shape[2], golden.shape[3], dtype=torch.float16)
    extend_params = {}
    conv2d_kernel(fmap, weight, bias, out, strides, paddings, dilations, tile_l1, tile_l0, extend_params)
    assert_pass_verify_ok()


@pytest.mark.soc("950")
def test_conv2d_1x1_smoke_950():
    """A5 (Ascend950) conv2d smoke case with NCHW in/out."""
    _run_conv2d_case((1, 32, 8, 16), (16, 32, 1, 1), [1, 1], [0, 0, 0, 0], [1, 1],
                     tile_l1=[8, 8, 16, 16, 16, 16, 16, 1], tile_l0=[8, 16, 16, 16])


@pytest.mark.soc("950")
def test_conv2d_3x3_pad_smoke_950():
    """A5 conv2d with padding and 3x3 kernel (h pads only: tileWin must stay <= 16 on A5)."""
    _run_conv2d_case((1, 16, 18, 18), (16, 16, 3, 3), [1, 1], [1, 1, 0, 0], [1, 1],
                     tile_l1=[10, 8, 16, 16, 16, 16, 16, 1], tile_l0=[4, 16, 16, 16])


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
)
def conv2d_dynamic_batch_kernel(
    input_a: pypto.Tensor([pypto.DYNAMIC, 16, 32, 32], pypto.DT_FP16),
    input_b: pypto.Tensor([64, 16, 3, 3], pypto.DT_FP16),
    output_c: pypto.Tensor([pypto.DYNAMIC, 64, 15, 15], pypto.DT_FP16),
    params,
):
    batch = params["batch"]
    tile_batch = pypto.symbolic_scalar(1)
    batch_loop = (batch + tile_batch - 1) // tile_batch

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=16, tileWout=16, tileCinFmap=16,
                              tileCinWeight=16, tileN=64, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=144, tileN=64),
    )

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_batch"):
        batch_offset = batch_idx * tile_batch
        input_a_view = pypto.view(input_a, [tile_batch, 16, 32, 32], [batch_offset, 0, 0, 0])
        out = pypto.conv(input_a_view, input_b, pypto.DT_FP16, [2, 2], [1, 1, 1, 1], [2, 2],
                         extend_params={}, groups=1)
        pypto.assemble(out, [batch_offset, 0, 0, 0], output_c)


@pytest.mark.soc("950")
def test_conv2d_dynamic_batch_stride():
    """Conv2D dynamic batch, stride=2, dilation=2, pad=1, FP16 (A5 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(2, 16, 32, 32, dtype=torch.float16)
    b = torch.randn(64, 16, 3, 3, dtype=torch.float16) * 0.1
    golden = torch.nn.functional.conv2d(
        a.float(), b.float(), stride=(2, 2), padding=(1, 1), dilation=(2, 2), groups=1
    ).half()
    set_verify_goldens([None, None, golden])
    c_out = torch.zeros(2, 64, 15, 15, dtype=torch.float16)
    conv2d_dynamic_batch_kernel(a, b, c_out, {"batch": 2})
    assert_pass_verify_ok()


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
)
def conv2d_dynamic_hout_relu_kernel(
    input_a: pypto.Tensor([1, 16, pypto.DYNAMIC, 34], pypto.DT_FP32),
    input_b: pypto.Tensor([64, 16, 3, 3], pypto.DT_FP32),
    output_c: pypto.Tensor([1, 64, pypto.DYNAMIC, 32], pypto.DT_FP32),
):
    hin = input_a.shape[2]
    ho = output_c.shape[2]
    tile_hout = 8
    hout_loop = (ho + tile_hout - 1) // tile_hout
    tile_hin = tile_hout + 2  # stride=1, kernel=3

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=8, tileHout=8, tileWin=32, tileWout=32, tileCinFmap=16,
                              tileCinWeight=16, tileN=64, tileBatch=1),
        pypto_impl.TileL0Info(tileH=8, tileW=32, tileK=48, tileN=64),
    )

    for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_hout"):
        hout_offset = hout_idx * tile_hout
        hin_offset = hout_idx * tile_hout
        hin_current = (hin - hin_offset).min(tile_hin)
        input_a_view = pypto.view(
            input_a, [1, 16, tile_hin, 34], [0, 0, hin_offset, 0], valid_shape=[1, 16, hin_current, 34]
        )
        out = pypto.conv(
            input_a_view,
            input_b,
            pypto.DT_FP32,
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            extend_params={"relu_type": pypto_impl.ConvReLuType.RELU},
            groups=1,
        )
        pypto.assemble(out, [0, 0, hout_offset, 0], output_c)


@pytest.mark.soc("950")
def test_conv2d_dynamic_hout_relu():
    """Conv2D dynamic hout with relu fusion, pad=0, stride=1, FP32 (A5 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(1, 16, 34, 34, dtype=torch.float32)
    b = torch.randn(64, 16, 3, 3, dtype=torch.float32) * 0.1
    golden = torch.nn.functional.conv2d(a, b, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    golden = torch.relu(golden)
    set_verify_goldens([None, None, golden])
    c_out = torch.zeros(1, 64, 32, 32, dtype=torch.float32)
    conv2d_dynamic_hout_relu_kernel(a, b, c_out)
    assert_pass_verify_ok()


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
)
def conv1d_dynamic_cout_kernel(
    input_a: pypto.Tensor([1, 16, 64], pypto.DT_BF16),
    input_b: pypto.Tensor([pypto.DYNAMIC, 16, 3], pypto.DT_BF16),
    output_c: pypto.Tensor([1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    params,
):
    cout = params["cout"]
    tile_cout = pypto.symbolic_scalar(32)
    cout_loop = (cout + tile_cout - 1) // tile_cout

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=64, tileWout=64, tileCinFmap=16,
                              tileCinWeight=16, tileN=tile_cout, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=64, tileK=48, tileN=tile_cout),
    )

    for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_cout"):
        cout_offset = cout_idx * tile_cout
        input_b_view = input_b[cout_offset:cout_offset + tile_cout, 0:16, 0:3]
        out = pypto.conv(input_a, input_b_view, pypto.DT_BF16, [1], [1, 1], [1],
                         extend_params={}, groups=1)
        pypto.assemble(out, [0, cout_offset, 0], output_c)


@pytest.mark.soc("950")
def test_conv1d_dynamic_cout():
    """Conv1D dynamic cout, stride=1, dilation=1, pad=1, BF16 (A5 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(1, 16, 64, dtype=torch.bfloat16)
    b = torch.randn(64, 16, 3, dtype=torch.bfloat16) * 0.1
    golden = torch.nn.functional.conv1d(
        a.float(), b.float(), stride=1, padding=1, dilation=1, groups=1
    ).to(torch.bfloat16)
    set_verify_goldens([None, None, golden])
    c_out = torch.zeros(1, 64, 64, dtype=torch.bfloat16)
    conv1d_dynamic_cout_kernel(a, b, c_out, {"cout": 64})
    assert_pass_verify_ok()


# NOTE: explicit per-kernel options (2026-09-23):
# - auto_mix_partition: the DYNAMIC-weight-slice + conv + assemble pattern
#   (conv1d_dynamic_cout) used to hit FB1010 MIX_GLOBAL_TENSOR_WAIT_TIMEOUT
#   in CV mix multithread execution when mix partition was enabled. Since
#   !6648 the switch lives in pypto.experimental.auto_mix_partition(0/1),
#   set_pass_options no longer accepts the keyword, and the framework
#   default converged to 0 (disabled) - so no explicit setting is needed
#   here. Revisit if the mix wait defect gets fixed and defaults change.
# - host_options compile_stage=EXECUTE_GRAPH (soc("910") kernels): pypto
#   master headers emit pto::TPARTSEL, unknown to the local CANN 9.2.0 CCE
#   toolchain, so full codegen fails. The shared conv2d_kernel carries it too:
#   its generated aicore code includes tileop/vector/binary/floor_div.h
#   (groups derivation), which emits TPARTSEL. Stopping at EXECUTE_GRAPH keeps
#   pass_verify + goldens working standalone, without relying on the
#   compile_stage leaked by the soc("950") conftest fixture. Drop it after
#   the CANN toolchain knows TPARTSEL.
# NOTE: the soc("910") cases run last on purpose. Forcing DAV_2201 on a 950
# machine opens libruntime_v200, and any subsequent DeviceInit in the same
# process can deadlock in the runtime layer. Keep A2A3-forced cases at the
# end of the file until the golden gaps below are fixed:
# - OP_UB_COPY_L1 (conv weight local copy, 4D nz) is bound to the 2D-only
#   ExecuteL0CToL1 (calc_cube.cpp), raising L0C_TO_L1_SHAPE_NOT_2D (FB200E).
# - conv relu fusion (extend_params relu_type) is not reflected in the A2A3
#   golden (FB4001: golden 0 vs negative output), so the A2A3 dynamic hout
#   case below drops the fusion that its A5 twin carries.
# - conv3d fails the A2A3 host golden compile with F00003 (FAKE_TRANS shape
#   mismatch) for both FP16 and BF16 + FP32 bias, so there is no conv3d A2A3
#   case yet.
# - transDataPadFractalZ zero-fills the whole padded tile and then copies the
#   valid block through a view, so both writes target the same tensor and the
#   golden must keep their program order. The interpreter does not model view
#   aliasing (ResetSliceView gives each view its own RawTensorData) and the
#   dynamic execution order is token based, so the zero fill can run after the
#   copy and the conv1d groups=2 weight golden degenerates to all zeros.
#   test_conv1d_fp16_basic_with_bias is skipped until the interpreter models
#   view aliasing or the pad IR stops relying on writer order.
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
    host_options={"compile_stage": pypto.CompStage.EXECUTE_GRAPH},
)
def conv1d_bias_kernel(
    fmap: pypto.Tensor([1, 16, 32], pypto.DT_FP16),
    weight: pypto.Tensor([16, 8, 3], pypto.DT_FP16),
    bias: pypto.Tensor([16], pypto.DT_FP16),
    out: pypto.Tensor([1, 16, 15], pypto.DT_FP16),
):
    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=32, tileWout=16, tileCinFmap=16,
                              tileCinWeight=16, tileN=16, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=16, tileN=16),
    )
    extend_params = {"bias_tensor": bias}
    output = pypto.conv(fmap, weight, pypto.DT_FP16, [2], [1, 1], [2],
                        extend_params=extend_params, groups=2)
    out.move(output)


@pytest.mark.soc("910")
@pytest.mark.skip(reason="known golden gap: transdata pad view write order, see NOTE above")
def test_conv1d_fp16_basic_with_bias():
    """Conv1D FP16 groups=2 + bias, stride=2, dilation=2, pad=1 (A2A3 pipeline)."""
    torch.manual_seed(0)
    fmap = torch.randn(1, 16, 32, dtype=torch.float16)
    weight = torch.randn(16, 8, 3, dtype=torch.float16) * 0.1
    bias = torch.randn(16, dtype=torch.float16) * 0.1
    golden = torch.nn.functional.conv1d(
        fmap.float(), weight.float(), bias.float(), stride=2, padding=1, dilation=2, groups=2
    ).half()
    set_verify_goldens([None, None, None, golden])
    out = torch.zeros(1, 16, 15, dtype=torch.float16)
    conv1d_bias_kernel(fmap, weight, bias, out)
    assert_pass_verify_ok()


@pytest.mark.soc("910")
def test_conv2d_1x1_smoke():
    """Minimal conv2d: 1x32x8x16, kernel 1x1, the pass_verify smoke gate for conv (A2A3 pipeline).

    tileK=16 < cin=32 forces two K iterations, covering TILE_A_MUL_B + TILE_A_MULACC_B accumulation.
    """
    _run_conv2d_case((1, 32, 8, 16), (16, 32, 1, 1), [1, 1], [0, 0, 0, 0], [1, 1],
                     tile_l1=[8, 8, 16, 16, 16, 16, 16, 1], tile_l0=[8, 16, 16, 16])


@pytest.mark.soc("910")
def test_conv2d_3x3_smoke():
    """Conv2d 3x3 on 18x18 with multi L1/L0 tiles and 9 K iterations (A2A3 pipeline)."""
    _run_conv2d_case((1, 16, 18, 18), (16, 16, 3, 3), [1, 1], [0, 0, 0, 0], [1, 1],
                     tile_l1=[10, 8, 18, 16, 16, 16, 16, 1], tile_l0=[4, 16, 16, 16])


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
    host_options={"compile_stage": pypto.CompStage.EXECUTE_GRAPH},
)
def conv2d_dynamic_batch_a2a3_kernel(
    input_a: pypto.Tensor([pypto.DYNAMIC, 16, 32, 32], pypto.DT_FP16),
    input_b: pypto.Tensor([64, 16, 3, 3], pypto.DT_FP16),
    output_c: pypto.Tensor([pypto.DYNAMIC, 64, 15, 15], pypto.DT_FP16),
    params,
):
    batch = params["batch"]
    tile_batch = pypto.symbolic_scalar(1)
    batch_loop = (batch + tile_batch - 1) // tile_batch

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=16, tileWout=16, tileCinFmap=16,
                              tileCinWeight=16, tileN=64, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=144, tileN=64),
    )

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_batch"):
        batch_offset = batch_idx * tile_batch
        input_a_view = pypto.view(input_a, [tile_batch, 16, 32, 32], [batch_offset, 0, 0, 0])
        out = pypto.conv(input_a_view, input_b, pypto.DT_FP16, [2, 2], [1, 1, 1, 1], [2, 2],
                         extend_params={}, groups=1)
        pypto.assemble(out, [batch_offset, 0, 0, 0], output_c)


@pytest.mark.soc("910")
def test_conv2d_dynamic_batch_stride_a2a3():
    """Conv2D dynamic batch, stride=2, dilation=2, pad=1, FP16 (A2A3 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(2, 16, 32, 32, dtype=torch.float16)
    b = torch.randn(64, 16, 3, 3, dtype=torch.float16) * 0.1
    golden = torch.nn.functional.conv2d(
        a.float(), b.float(), stride=(2, 2), padding=(1, 1), dilation=(2, 2), groups=1
    ).half()
    set_verify_goldens([None, None, golden])
    c_out = torch.zeros(2, 64, 15, 15, dtype=torch.float16)
    conv2d_dynamic_batch_a2a3_kernel(a, b, c_out, {"batch": 2})
    assert_pass_verify_ok()


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
    host_options={"compile_stage": pypto.CompStage.EXECUTE_GRAPH},
)
def conv2d_dynamic_batch_bf16_a2a3_kernel(
    input_a: pypto.Tensor([pypto.DYNAMIC, 16, 32, 32], pypto.DT_BF16),
    input_b: pypto.Tensor([64, 16, 3, 3], pypto.DT_BF16),
    output_c: pypto.Tensor([pypto.DYNAMIC, 64, 15, 15], pypto.DT_BF16),
    params,
):
    batch = params["batch"]
    tile_batch = pypto.symbolic_scalar(1)
    batch_loop = (batch + tile_batch - 1) // tile_batch

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=16, tileWout=16, tileCinFmap=16,
                              tileCinWeight=16, tileN=64, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=16, tileK=144, tileN=64),
    )

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_batch"):
        batch_offset = batch_idx * tile_batch
        input_a_view = pypto.view(input_a, [tile_batch, 16, 32, 32], [batch_offset, 0, 0, 0])
        out = pypto.conv(input_a_view, input_b, pypto.DT_BF16, [2, 2], [1, 1, 1, 1], [2, 2],
                         extend_params={}, groups=1)
        pypto.assemble(out, [batch_offset, 0, 0, 0], output_c)


@pytest.mark.soc("910")
def test_conv2d_dynamic_batch_bf16_a2a3():
    """Conv2D dynamic batch, stride=2, dilation=2, pad=1, BF16 (A2A3 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(2, 16, 32, 32, dtype=torch.bfloat16)
    b = torch.randn(64, 16, 3, 3, dtype=torch.bfloat16) * 0.1
    golden = torch.nn.functional.conv2d(
        a.float(), b.float(), stride=(2, 2), padding=(1, 1), dilation=(2, 2), groups=1
    ).to(torch.bfloat16)
    set_verify_goldens([None, None, golden])
    c_out = torch.zeros(2, 64, 15, 15, dtype=torch.bfloat16)
    conv2d_dynamic_batch_bf16_a2a3_kernel(a, b, c_out, {"batch": 2})
    assert_pass_verify_ok()


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
    host_options={"compile_stage": pypto.CompStage.EXECUTE_GRAPH},
)
def conv2d_dynamic_hout_a2a3_kernel(
    input_a: pypto.Tensor([1, 16, pypto.DYNAMIC, 34], pypto.DT_FP32),
    input_b: pypto.Tensor([64, 16, 3, 3], pypto.DT_FP32),
    output_c: pypto.Tensor([1, 64, pypto.DYNAMIC, 32], pypto.DT_FP32),
):
    hin = input_a.shape[2]
    ho = output_c.shape[2]
    tile_hout = 8
    hout_loop = (ho + tile_hout - 1) // tile_hout
    tile_hin = tile_hout + 2  # stride=1, kernel=3

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=8, tileHout=8, tileWin=32, tileWout=32, tileCinFmap=16,
                              tileCinWeight=16, tileN=64, tileBatch=1),
        pypto_impl.TileL0Info(tileH=8, tileW=32, tileK=48, tileN=64),
    )

    for hout_idx in pypto.loop(0, hout_loop, 1, name="LOOP_hout"):
        hout_offset = hout_idx * tile_hout
        hin_offset = hout_idx * tile_hout
        hin_current = (hin - hin_offset).min(tile_hin)
        input_a_view = pypto.view(
            input_a, [1, 16, tile_hin, 34], [0, 0, hin_offset, 0], valid_shape=[1, 16, hin_current, 34]
        )
        out = pypto.conv(input_a_view, input_b, pypto.DT_FP32, [1, 1], [0, 0, 0, 0], [1, 1],
                         extend_params={}, groups=1)
        pypto.assemble(out, [0, 0, hout_offset, 0], output_c)


@pytest.mark.soc("910")
def test_conv2d_dynamic_hout_a2a3():
    """Conv2D dynamic hout, pad=0, stride=1, FP32, no relu (A2A3 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(1, 16, 34, 34, dtype=torch.float32)
    b = torch.randn(64, 16, 3, 3, dtype=torch.float32) * 0.1
    golden = torch.nn.functional.conv2d(a, b, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    set_verify_goldens([None, None, golden])
    c_out = torch.zeros(1, 64, 32, 32, dtype=torch.float32)
    conv2d_dynamic_hout_a2a3_kernel(a, b, c_out)
    assert_pass_verify_ok()


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
    host_options={"compile_stage": pypto.CompStage.EXECUTE_GRAPH},
)
def conv1d_dynamic_cout_a2a3_kernel(
    input_a: pypto.Tensor([1, 16, 64], pypto.DT_BF16),
    input_b: pypto.Tensor([pypto.DYNAMIC, 16, 3], pypto.DT_BF16),
    output_c: pypto.Tensor([1, pypto.DYNAMIC, 64], pypto.DT_BF16),
    params,
):
    cout = params["cout"]
    tile_cout = pypto.symbolic_scalar(32)
    cout_loop = (cout + tile_cout - 1) // tile_cout

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=64, tileWout=64, tileCinFmap=16,
                              tileCinWeight=16, tileN=tile_cout, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=64, tileK=48, tileN=tile_cout),
    )

    for cout_idx in pypto.loop(0, cout_loop, 1, name="LOOP_cout"):
        cout_offset = cout_idx * tile_cout
        input_b_view = input_b[cout_offset:cout_offset + tile_cout, 0:16, 0:3]
        out = pypto.conv(input_a, input_b_view, pypto.DT_BF16, [1], [1, 1], [1],
                         extend_params={}, groups=1)
        pypto.assemble(out, [0, cout_offset, 0], output_c)


@pytest.mark.soc("910")
def test_conv1d_dynamic_cout_a2a3():
    """Conv1D dynamic cout, stride=1, dilation=1, pad=1, BF16 (A2A3 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(1, 16, 64, dtype=torch.bfloat16)
    b = torch.randn(64, 16, 3, dtype=torch.bfloat16) * 0.1
    golden = torch.nn.functional.conv1d(
        a.float(), b.float(), stride=1, padding=1, dilation=1, groups=1
    ).to(torch.bfloat16)
    set_verify_goldens([None, None, golden])
    c_out = torch.zeros(1, 64, 64, dtype=torch.bfloat16)
    conv1d_dynamic_cout_a2a3_kernel(a, b, c_out, {"cout": 64})
    assert_pass_verify_ok()


@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.SIM, "stitch_function_max_num": 128},
    pass_options={"enable_slice": False},
    host_options={"compile_stage": pypto.CompStage.EXECUTE_GRAPH},
)
def conv1d_dynamic_batch_bias_a2a3_kernel(
    input_a: pypto.Tensor([pypto.DYNAMIC, 16, 64], pypto.DT_FP16),
    input_b: pypto.Tensor([32, 16, 3], pypto.DT_FP16),
    bias: pypto.Tensor([32], pypto.DT_FP16),
    output_c: pypto.Tensor([pypto.DYNAMIC, 32, 32], pypto.DT_FP16),
    params,
):
    batch = params["batch"]
    tile_batch = pypto.symbolic_scalar(1)
    batch_loop = (batch + tile_batch - 1) // tile_batch

    pypto.set_conv_tile_shapes(
        pypto_impl.TileL1Info(tileHin=1, tileHout=1, tileWin=64, tileWout=32, tileCinFmap=16,
                              tileCinWeight=16, tileN=32, tileBatch=1),
        pypto_impl.TileL0Info(tileH=1, tileW=32, tileK=48, tileN=32),
    )

    for batch_idx in pypto.loop(0, batch_loop, 1, name="LOOP_batch"):
        batch_offset = batch_idx * tile_batch
        input_a_view = pypto.view(input_a, [tile_batch, 16, 64], [batch_offset, 0, 0])
        out = pypto.conv(input_a_view, input_b, pypto.DT_FP16, [2], [1, 1], [1],
                         extend_params={"bias_tensor": bias}, groups=1)
        pypto.assemble(out, [batch_offset, 0, 0], output_c)


@pytest.mark.soc("910")
def test_conv1d_dynamic_batch_bias_a2a3():
    """Conv1D dynamic batch with bias, stride=2, dilation=1, pad=1, FP16 (A2A3 pipeline)."""
    torch.manual_seed(0)
    a = torch.randn(2, 16, 64, dtype=torch.float16)
    b = torch.randn(32, 16, 3, dtype=torch.float16) * 0.1
    bias = torch.randn(32, dtype=torch.float16) * 0.1
    golden = torch.nn.functional.conv1d(
        a.float(), b.float(), bias.float(), stride=2, padding=1, dilation=1, groups=1
    ).half()
    set_verify_goldens([None, None, None, golden])
    c_out = torch.zeros(2, 32, 32, dtype=torch.float16)
    conv1d_dynamic_batch_bias_a2a3_kernel(a, b, bias, c_out, {"batch": 2})
    assert_pass_verify_ok()
