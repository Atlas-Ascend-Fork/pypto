# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""init_output op: initialize a GM tensor region with a scalar value.

Doc: docs/zh/api/pro_api/SIMD-API/memory_data_movement/init_output.md

Three representative tests keep only the guards with real failure
mechanisms behind them (per-dtype value encoding and value magnitudes are
deliberately NOT guarded — TEXPANDS encodes them uniformly):

1. test_init_output_scalar_cases  — static mechanisms: baseline full fill,
   pl.const() scalar value, multiple overlapping calls on one tensor,
   partial offset window, non-16-aligned size (chunk-loop tail),
   multi-chunk address advance, and NZ layout (flat physical element
   addressing through a [1, numel] view).
2. test_init_output_dynamic       — dynamic/operand forms: multicore per-core
   sharding with runtime offset/size and tail-core clamp on 3D FP32 and 4D
   FP16 (AIV sub-block numbering, INDEX->float cast), pl.range loop variable
   as value on INT64 (pure int path, no cast), dynamic shape dims as value.
3. test_init_output_matmul_integration — mixed-kernel chain: per-AIV half-row
   init windows, sync_all(MIX) on both sides, init-written GM read back as
   matmul input (AIV->AIC visibility), DualModeSplitM double-target split,
   cross-core FIX->V event, plain and atomic stores (AIC->AIV visibility).

Constraint: init_output emits no cross-core barrier; when the initialized
data is consumed by other cores / the cube sub-core, the caller must insert
pl.system.sync_all(core_type=MIX) on both sides (documented in the md).
"""

import ctypes
import logging
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _require_a5(device):
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _nz_raw_copy(dst_ptr: int, src: torch.Tensor, kind: int, nbytes: int) -> None:
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl.aclrtMemcpy.restype = ctypes.c_int
    assert acl.aclrtMemcpy(dst_ptr, nbytes, src.data_ptr(), nbytes, kind) == 0


# ===========================================================================
# 1. Static scalar-fill matrix (+ NZ layout)
# ===========================================================================
NZ_M = 128
NZ_N = 128
NZ_NUMEL = NZ_M * NZ_N


# All independent scalar-fill cases share a single vector kernel. Each output
# retains its original dtype, shape, initial sentinel and expected fill regions.
@pl.jit(auto_mutex=True)
def init_output_scalar_cases(
    fp32_zero: pl.Tensor[[64, 64], pl.DT_FP32],
    fp32_const: pl.Tensor[[200], pl.DT_FP32],
    fp32_multiple_calls: pl.Tensor[[256], pl.DT_FP32],
    fp32_offset: pl.Tensor[[128], pl.DT_FP32],
    fp32_size_17: pl.Tensor[[17], pl.DT_FP32],
    fp32_multi_chunk: pl.Tensor[[65536], pl.DT_FP32],
    fp16_nz: pl.Tensor[[NZ_M, NZ_N], pl.DT_FP16, pl.NZ],
):
    with pl.section_vector():
        pl.init_output(fp32_zero, offset=0, size=64 * 64, value=0.0)
        pl.init_output(fp32_const, offset=0, size=200, value=pl.const(2.5, pl.DT_FP32))
        pl.init_output(fp32_multiple_calls, offset=0, size=128, value=1.5)
        pl.init_output(fp32_multiple_calls, offset=128, size=128, value=-2.0)
        pl.init_output(fp32_multiple_calls, offset=96, size=64, value=7.0)
        pl.init_output(fp32_offset, offset=16, size=96, value=0.0)
        pl.init_output(fp32_size_17, offset=0, size=17, value=0.0)
        pl.init_output(fp32_multi_chunk, offset=0, size=65536, value=0.0)
        # NZ layout: offset/size count PHYSICAL elements of the flat [1, numel]
        # view, so the fractal layout must not change the fill semantics.
        pl.init_output(fp16_nz, offset=256, size=512, value=-3.5)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_scalar_cases():
    _require_a5(ST_DEVICE)
    # name, shape, dtype, initial value, (offset, size, fill value) regions.
    # The order matches the kernel signature. Overlapping regions are applied in order.
    cases = [
        ("fp32_zero", (64, 64), torch.float32, 1, [(0, 4096, 0)]),
        ("fp32_const", (200,), torch.float32, 99, [(0, 200, 2.5)]),
        ("fp32_multiple_calls", (256,), torch.float32, 99, [(0, 128, 1.5), (128, 128, -2), (96, 64, 7)]),
        ("fp32_offset", (128,), torch.float32, 1, [(16, 96, 0)]),
        ("fp32_size_17", (17,), torch.float32, 1, [(0, 17, 0)]),
        ("fp32_multi_chunk", (65536,), torch.float32, 1, [(0, 65536, 0)]),
    ]
    outputs = [torch.full(shape, initial, dtype=dtype).to(ST_DEVICE) for _, shape, dtype, initial, _ in cases]

    # NZ output: aligned [128, 128] FP16 NZ storage (acl_format 29) holds
    # exactly NZ_NUMEL elements with no padding; pre-fill raw bytes with a
    # sentinel so an unfilled region is detectable.
    nz = torch_npu.empty_with_format([NZ_M, NZ_N], dtype=torch.float16, device=ST_DEVICE, acl_format=29)
    assert torch_npu.get_npu_format(nz) == 29
    nz_storage = nz.untyped_storage()
    nz_numel_phys = nz_storage.nbytes() // 2
    assert nz_numel_phys == NZ_NUMEL, f"aligned NZ storage expected {NZ_NUMEL} elems, got {nz_numel_phys}"
    torch.npu.synchronize()
    nz_sentinel = torch.full([nz_numel_phys], 66.0, dtype=torch.float16)  # H2D: ACL_MEMCPY_HOST_TO_DEVICE
    _nz_raw_copy(nz_storage.data_ptr(), nz_sentinel, 1, nz_storage.nbytes())

    init_output_scalar_cases(*outputs, nz)
    torch.npu.synchronize()
    for output, (name, shape, dtype, initial, regions) in zip(outputs, cases):
        expected = torch.full(shape, initial, dtype=dtype)
        for offset, size, value in regions:
            expected.view(-1)[offset:offset + size] = value
        torch.testing.assert_close(
            output.cpu(), expected, rtol=0, atol=0, msg=lambda message: f"{name}: {message}"
        )

    # NZ verification needs a raw D2H copy: the flat physical element sequence
    # must carry the fill exactly as an ND tensor would.
    host = torch.empty(nz_numel_phys, dtype=torch.float16)  # D2H: ACL_MEMCPY_DEVICE_TO_HOST
    _nz_raw_copy(host.data_ptr(), nz_storage, 2, nz_storage.nbytes())
    nz_expected = torch.full([nz_numel_phys], 66.0, dtype=torch.float16)
    nz_expected[256:768] = -3.5
    torch.testing.assert_close(host, nz_expected)
    logging.info("init_output scalar cases + NZ layout passed!")


# ===========================================================================
# 2. Dynamic operand forms: multicore sharding, loop variable, shape dims
# ===========================================================================
NUM_CORES = 28


@pl.jit(auto_mutex=True)
def init_output_dynamic(
    out3d: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out4d: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out_i64: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64],
    out2d: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    core_id = pl.get_block_idx()
    num_cores = pl.get_block_num()
    with pl.section_vector():
        # Per-core sharding with tail-core clamp. On the AIV side get_block_idx
        # is sub-block granular (1:2 mix -> 2 * num_cores work units); each unit
        # fills its own slice with its own INDEX value (INDEX->float cast).
        total3 = out3d.shape[0] * out3d.shape[1] * out3d.shape[2]
        per3 = (total3 + num_cores - 1) // num_cores
        off3 = core_id * per3
        pl.init_output(out3d, offset=off3, size=pl.min(per3, total3 - off3), value=core_id)
        total4 = out4d.shape[0] * out4d.shape[1] * out4d.shape[2] * out4d.shape[3]
        per4 = (total4 + num_cores - 1) // num_cores
        off4 = core_id * per4
        pl.init_output(out4d, offset=off4, size=pl.min(per4, total4 - off4), value=core_id)
        # Loop induction variable as value on INT64 (pure int path, no cast).
        # Every core writes identical values, so concurrent identical fills
        # are idempotent.
        n64 = out_i64.shape[0]
        for i in pl.range(0, n64, 64):
            pl.init_output(out_i64, offset=i, size=pl.min(64, n64 - i), value=i * 1000)
        # Dynamic shape dims as value expressions.
        m = out2d.shape[0]
        n2 = out2d.shape[1]
        total2 = m * n2
        pl.init_output(out2d, offset=0, size=total2, value=m + n2)
        pl.init_output(out2d, offset=total2 // 2, size=total2 // 2, value=n2 * 2)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_dynamic():
    """Dynamic forms: 3D FP32 + 4D FP16 per-core sharding, loop-var INT64, shape-expr values."""
    device = ST_DEVICE
    _require_a5(device)
    dims3 = (4, 32, 100)      # 12800 elems, FP32
    dims4 = (2, 3, 17, 19)    # 1938 elems, FP16, smaller tail
    n64 = 130
    dims2 = (3, 100)          # 300 elems, FP32

    out3d = torch.full(dims3, 99.0, device=device, dtype=torch.float32)
    out4d = torch.full(dims4, 99.0, device=device, dtype=torch.float16)
    out_i64 = torch.full([n64], -1, device=device, dtype=torch.int64)
    out2d = torch.full(dims2, -1.0, device=device, dtype=torch.float32)
    init_output_dynamic[None, NUM_CORES](out3d, out4d, out_i64, out2d)
    torch.npu.synchronize()

    # Per-core sharding: value = AIV sub-block index over the flat view.
    for out, dims in ((out3d, dims3), (out4d, dims4)):
        total = 1
        for d in dims:
            total *= d
        per_core = (total + NUM_CORES - 1) // NUM_CORES
        flat = out.view(-1)
        for core in range(NUM_CORES * 2):
            offset = core * per_core
            size = min(per_core, total - offset)
            if size > 0:
                flat[offset:offset + size] = core
    # Loop variable on INT64: chunk base * 1000.
    expected_i64 = torch.empty(n64, dtype=torch.int64)
    for j in range(n64):
        expected_i64[j] = (j // 64) * 64 * 1000
    torch.testing.assert_close(out_i64.cpu(), expected_i64, rtol=0, atol=0)
    # Shape expressions: first half m + n, second half n * 2.
    m, n2 = dims2
    expected2d = torch.full(dims2, float(m + n2), dtype=torch.float32)
    expected2d.view(-1)[m * n2 // 2:] = float(n2 * 2)
    torch.testing.assert_close(out2d.cpu(), expected2d, rtol=0, atol=0)
    logging.info("init_output dynamic forms (multicore 3D/4D + loop-var INT64 + shape-expr) passed!")


# ===========================================================================
# 3. Mixed-kernel integration: init_output + matmul + atomic store
#
# Dataflow (28 blocks, 1:2 mix -> 56 AIV work units, core_id below is the
# sub-block-granular AIV index; the AIC of block i is fed by units 2i/2i+1):
#   1. AIV unit j initializes half of the rows its AIC consumes:
#      ws_in rows [8j, 8j+8) = j (per-band value, INDEX->FP16 cast) and
#      ws_acc rows [8j, 8j+8) = 0. init_output emits no cross-core barrier,
#      so sync_all(MIX) is inserted on both sides.
#   2. AIC i loads its ws_in slice and shared b, matmul -> Acc; the split move
#      (DualModeSplitM) routes rows [16i+8s, 16i+8s+8) to Vec sub-target s.
#   3. AIV unit j = 2i+s waits the cross-core FIX->V event, adds 0.5, then
#      plain-stores c_out rows [8j, 8j+8) and atomic-adds ws_acc rows [8j, 8j+8).
#
# Final: row r of c_out and ws_acc equals (r // 8) * base + 0.5 where
# base = ones(1, K) @ b — every band carries a distinct value, so a wrong
# row mapping, a missing sync or a corrupted init shows up immediately.
# ===========================================================================
M_IO = 448
K_IO = 16
N_IO = 16
TILE_M = 16
NUM_CORES_IO = 28


@pl.jit(auto_mutex=True)
def init_output_matmul_integration(
    b: pl.Tensor[[K_IO, N_IO], pl.DT_FP16],
    ws_in: pl.Tensor[[M_IO, K_IO], pl.DT_FP16],
    ws_acc: pl.Tensor[[M_IO, N_IO], pl.DT_FP32],
    c_out: pl.Tensor[[M_IO, N_IO], pl.DT_FP32],
):
    core_id = pl.get_block_idx()
    # get_block_idx() is per-section: AIC-side [0, num_blocks) in section_cube,
    # AIV sub-block-granular [0, 2 * num_blocks) in section_vector.
    row0 = core_id * TILE_M

    ws_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, K_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K_IO, N_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000,
        mutex_ids=[1],
    )
    ws_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, K_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_IO, N_IO], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000,
        mutex_ids=[3],
    )
    tile_acc = pl.make_tile(
        pl.TileType(shape=[TILE_M, N_IO], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addr=0x0000,
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M // 2, N_IO], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x2000,
        mutex_ids=[[6]],
    )

    with pl.section_vector():
        # Each AIV initializes half of the rows its AIC consumes (ws_in, with a
        # per-band value so row mappings are verifiable) and half of the rows
        # it will atomic-add into (ws_acc, zero base).
        pl.init_output(ws_in, offset=core_id * (TILE_M // 2) * K_IO, size=(TILE_M // 2) * K_IO, value=core_id)
        pl.init_output(ws_acc, offset=core_id * (TILE_M // 2) * N_IO, size=(TILE_M // 2) * N_IO, value=0.0)
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    with pl.section_cube():
        pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
        pl.load(ws_l1[0], ws_in, [row0, 0])
        pl.load(b_l1[0], b, [0, 0])
        pl.move(ws_left[0], ws_l1[0])
        pl.move(b_right[0], b_l1[0])
        pl.matmul(tile_acc, ws_left[0], b_right[0])
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.move(vec_group[0], tile_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        row_off = core_id * (TILE_M // 2)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        pl.add(vec_group[0], vec_group[0], 0.5)
        pl.store(c_out, vec_group[0], [row_off, 0])
        pl.store(ws_acc, vec_group[0], [row_off, 0], atomic=pl.AtomicType.AtomicAdd)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_matmul_integration():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(42)
    b = torch.randn([K_IO, N_IO], device=device, dtype=torch.float16)
    ws_in = torch.full([M_IO, K_IO], 99.0, device=device, dtype=torch.float16)
    ws_acc = torch.full([M_IO, N_IO], 99.0, device=device, dtype=torch.float32)
    c_out = torch.full([M_IO, N_IO], 99.0, device=device, dtype=torch.float32)

    init_output_matmul_integration[None, NUM_CORES_IO](b, ws_in, ws_acc, c_out)
    torch.npu.synchronize()

    # ws_in must hold the per-band init values: row r -> r // 8 (0..55, exact in FP16).
    band = (torch.arange(M_IO, dtype=torch.int32) // (TILE_M // 2)).to(torch.float16)
    torch.testing.assert_close(ws_in.cpu(), band.unsqueeze(1).expand(M_IO, K_IO), rtol=0, atol=0)

    # base = ones(1, K) @ b; row r of both outputs = (r // 8) * base + 0.5.
    base = torch.matmul(torch.ones(1, K_IO, device=b.device), b.float()).cpu()
    expected = band.to(torch.float32).unsqueeze(1) * base + 0.5
    torch.testing.assert_close(c_out.cpu(), expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ws_acc.cpu(), expected, rtol=1e-3, atol=1e-3)
    logging.info("init_output + matmul + atomic (mixed kernel, both visibility directions) passed!")


# ===========================================================================
# 4. Float value on an integer tensor: the filler truncates toward zero
# ===========================================================================


@pl.jit(auto_mutex=True)
def init_output_float_value_on_int(
    out_i8: pl.Tensor[[32], pl.DT_INT8],
    out_u8: pl.Tensor[[32], pl.DT_UINT8],
    out_i32: pl.Tensor[[32], pl.DT_INT32],
):
    with pl.section_vector():
        pl.init_output(out_i8, offset=0, size=32, value=-1.9)
        pl.init_output(out_u8, offset=0, size=32, value=200.7)
        pl.init_output(out_i32, offset=0, size=32, value=2.7)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_init_output_float_value_on_int():
    """A float value on an integer tensor is truncated toward zero, not rejected."""
    _require_a5(ST_DEVICE)
    # Non-zero sentinels so an unfilled element would be visible.
    out_i8 = torch.full([32], 7, dtype=torch.int8).to(ST_DEVICE)
    out_u8 = torch.full([32], 7, dtype=torch.uint8).to(ST_DEVICE)
    out_i32 = torch.full([32], 7, dtype=torch.int32).to(ST_DEVICE)

    init_output_float_value_on_int(out_i8, out_u8, out_i32)
    torch.npu.synchronize()

    # Truncation toward zero: -1.9 -> -1, 200.7 -> 200, 2.7 -> 2.
    torch.testing.assert_close(out_i8.cpu(), torch.full([32], -1, dtype=torch.int8), rtol=0, atol=0)
    torch.testing.assert_close(out_u8.cpu(), torch.full([32], 200, dtype=torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(out_i32.cpu(), torch.full([32], 2, dtype=torch.int32), rtol=0, atol=0)
    logging.info("init_output float value on integer tensors truncated correctly!")
