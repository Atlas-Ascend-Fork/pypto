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

"""Tile subview tests for A[] slice syntax on A5.

Tests cover:

  1. Subview without valid_shape: row+col offset, stride mismatch (Vector section)
  2. Subview with runtime valid_shape (pl.set_validshape): intersection clamp
  3. VF subview: 2D row+col offset, per-row load/store with stride
  4. UB DN subview: column-major offset and parent-column stride in VF
  5. NZ subview: multi-fractal offset and byte conversion for C0=8/16/32/64
  6. L1 NZ/ZN subview: multi-fractal windows consumed by matmul
  7. L0C NZ subview: matmul destination and FixPipe source with ReLU/cast
  8. Cube composite: M/N/K loops, aligned tails, dynamic slices, and phases
"""

import ctypes
import logging
import os

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _copy_raw_npu_storage(tensor: torch.Tensor) -> torch.Tensor:
    torch.npu.synchronize()
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    host = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    aclrt_memcpy = ctypes.CDLL("libascendcl.so").aclrtMemcpy
    aclrt_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    aclrt_memcpy.restype = ctypes.c_int
    result = aclrt_memcpy(host.data_ptr(), nbytes, storage.data_ptr(), nbytes, 2)
    assert result == 0, f"aclrtMemcpy device-to-host failed with error {result}"
    return host


def _copy_raw_bytes_to_npu_storage(tensor: torch.Tensor, source: torch.Tensor) -> None:
    torch.npu.synchronize()
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    source = source.contiguous().view(torch.uint8).flatten()
    host = torch.zeros(nbytes, dtype=torch.uint8, device="cpu")
    host[:source.numel()].copy_(source)
    aclrt_memcpy = ctypes.CDLL("libascendcl.so").aclrtMemcpy
    aclrt_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    aclrt_memcpy.restype = ctypes.c_int
    result = aclrt_memcpy(storage.data_ptr(), nbytes, host.data_ptr(), nbytes, 1)
    assert result == 0, f"aclrtMemcpy host-to-device failed with error {result}"


def _nz_c0(dtype: torch.dtype) -> int:
    if dtype == torch.float4_e2m1fn_x2:
        return 64
    return 32 // torch.empty((), dtype=dtype).element_size()


def _pack_nz_cpu(logical: torch.Tensor) -> torch.Tensor:
    logical = logical.detach().cpu().contiguous()
    rows, cols = logical.shape
    c0 = _nz_c0(logical.dtype)
    padded_rows = (rows + 15) // 16 * 16
    padded_cols = (cols + c0 - 1) // c0 * c0

    if logical.dtype == torch.float4_e2m1fn_x2:
        logical_bytes = logical.view(torch.uint8).flatten()[:rows * cols // 2].reshape(rows, cols // 2)
        padded_bytes = torch.zeros((padded_rows, padded_cols // 2), dtype=torch.uint8)
        padded_bytes[:rows, :cols // 2] = logical_bytes
        return (
            padded_bytes.reshape(padded_rows // 16, 16, padded_cols // c0, c0 // 2)
            .permute(2, 0, 1, 3)
            .contiguous()
            .flatten()
        )

    padded = torch.zeros((padded_rows, padded_cols), dtype=logical.dtype)
    padded[:rows, :cols] = logical
    return (
        padded.reshape(padded_rows // 16, 16, padded_cols // c0, c0)
        .permute(2, 0, 1, 3)
        .contiguous()
        .view(torch.uint8)
        .flatten()
    )


def _to_nz(logical: torch.Tensor) -> torch.Tensor:
    nz = torch_npu.empty_with_format(
        list(logical.shape),
        dtype=logical.dtype,
        device=ST_DEVICE,
        acl_format=29,
    )
    _copy_raw_bytes_to_npu_storage(nz, _pack_nz_cpu(logical))
    return nz


def _assert_nz_storage_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    expected_storage = _pack_nz_cpu(expected)
    actual_storage = _copy_raw_npu_storage(actual)[:expected_storage.numel()]
    torch.testing.assert_close(actual_storage, expected_storage, rtol=0, atol=0)


def _is_a5() -> bool:
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        logging.info("NPU unavailable: %s", exc)
        return False
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        logging.info("Current device is %s, not A5 (Ascend950). Skip.", name)
        return False
    return True


# ===========================================================================
# Test T01: Subview without valid_shape — row+col offset, stride mismatch
#
# tile[2:6, 16:50] on [8, 64]
#   offset = 2*64+16 = 144, valid_shape = [4, 34], stride mismatch (34 != 64)
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_subview_kernel(
    a: pl.Tensor[[8, 64], pl.DT_FP16],
    out: pl.Tensor[[4, 34], pl.DT_FP16],
):
    tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000,
        mutex_ids=[0],
    )
    with pl.section_vector():
        tile = tile_group.next()
        pl.load(tile, a, [0, 0])
        sub = tile[2:6, 16:50]
        pl.store(out, sub, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(0)
    a = torch.rand(8, 64, device=device, dtype=torch.float16)
    out = torch.zeros(4, 34, device=device, dtype=torch.float16)

    tile_subview_kernel(a, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a[2:6, 16:50], rtol=0, atol=0)
    logging.info("test_tile_subview passed.")


# ===========================================================================
# Test T02: Subview with valid_shape intersection — runtime set_validshape
#
# tile shape [8, 64], valid_shape [6, 40] (via pl.set_validshape), slice [2:8, 8:64]
#
# Covers: rt_valid via set_validshape, intersection clamp
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_subview_vshape_kernel(
    a: pl.Tensor[[8, 64], pl.DT_FP16],
    out1: pl.Tensor[[4, 24], pl.DT_FP16],
):
    tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
        addrs=0x0000,
        mutex_ids=[0],
    )
    with pl.section_vector():
        tile = tile_group.next()
        pl.set_validshape(tile, [6, 40])
        pl.load(tile, a, [0, 0])
        sub1 = tile[2:6, 16:50]
        pl.store(out1, sub1, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_valid_shape():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(1)
    a = torch.rand(8, 64, device=device, dtype=torch.float16)
    out1 = torch.zeros(4, 24, device=device, dtype=torch.float16)

    tile_subview_vshape_kernel(a, out1)
    torch.npu.synchronize()

    expected = a[2:6, 16:40]
    torch.testing.assert_close(out1, expected, rtol=0, atol=0)
    logging.info("test_tile_subview_valid_shape passed.")


# ===========================================================================
# Test T03: VF subview — 2D row+col offset, per-row load/store with stride
#
# vf_src tile shape [4, 64], subview [1:4, 16:48] → 3 rows × 32 cols
# VF function loops over 3 rows, each row loads 32 FP16 from offset
# (row_start * 64 + col_start) and stores to dst row-by-row.
#
# Covers: VF pointer arithmetic with 2D subview, per-row stride walk
# ===========================================================================
@pl.vector_function
def vf_copy_subview_row(dst_tile, src_tile, n_rows, n_cols, stride):
    preg = vf.update_mask(n_cols, dtype=pl.DT_FP16)
    for r in pl.range(n_rows):
        vreg = vf.load_align(src_tile, r * stride)
        vf.store_align(dst_tile + r * n_cols, vreg, preg)


@pl.jit(auto_mutex=True)
def tile_subview_vf_kernel(
    vf_a: pl.Tensor[[4, 64], pl.DT_FP16],
    vf_out: pl.Tensor[[3, 32], pl.DT_FP16],
):
    vf_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[4, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000,
        mutex_ids=[1],
    )
    vf_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[3, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x5000,
        mutex_ids=[2],
    )
    with pl.section_vector():
        vf_src = vf_src_group.next()
        vf_dst = vf_dst_group.next()
        pl.load(vf_src, vf_a, [0, 0])
        vf_copy_subview_row(vf_dst, vf_src[1:4, 16:48], 3, 32, 64)
        pl.store(vf_out, vf_dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_vf():
    if not _is_a5():
        return
    device = ST_DEVICE
    torch.manual_seed(2)
    vf_a = torch.rand(4, 64, device=device, dtype=torch.float16)
    vf_out = torch.zeros(3, 32, device=device, dtype=torch.float16)

    tile_subview_vf_kernel(vf_a, vf_out)
    torch.npu.synchronize()

    torch.testing.assert_close(vf_out, vf_a[1:4, 16:48], rtol=0, atol=0)
    logging.info("test_tile_subview_vf passed.")


# ===========================================================================
# Test T04: UB DN subview consumed by VF
#
# Loading [16, 128] ND into a [128, 16] DN Tile with order=[1, 0] makes logical
# DN[r, c] correspond to ND[c, r]. The DN slice [32:93, 4:11] keeps a 32-byte
# aligned start for vf.load_align while using a non-aligned [61, 7] valid
# window. Each column retains the parent 128-element stride.
# ===========================================================================
@pl.vector_function
def vf_copy_dn_subview(dst_tile, src_tile):
    preg = vf.update_mask(61, dtype=pl.DT_FP16)
    for col in pl.range(7):
        vreg = vf.load_align(src_tile, col * 128)
        vf.store_align(dst_tile + col * 64, vreg, preg)


@pl.jit(auto_mutex=True)
def tile_subview_dn_kernel(
    inp: pl.Tensor[[16, 128], pl.DT_FP16],
    out: pl.Tensor[[8, 64], pl.DT_FP16],
):
    src_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 16],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.DN,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    dst_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[8, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=0x2000,
        mutex_ids=[1],
    )

    with pl.section_vector():
        src = src_group.next()
        dst = dst_group.next()
        pl.load(src, inp, [0, 0], order=[1, 0])
        pl.set_validshape(dst, [7, 61])
        vf_copy_dn_subview(dst, src[32:93, 4:11])
        pl.store(out, dst, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_dn():
    if not _is_a5():
        return

    inp_cpu = torch.arange(16 * 128, dtype=torch.float16).reshape(16, 128)
    inp = inp_cpu.to(ST_DEVICE)
    out = torch.zeros((8, 64), dtype=torch.float16, device=ST_DEVICE)

    tile_subview_dn_kernel(inp, out)
    torch.npu.synchronize()

    expected = torch.zeros((8, 64), dtype=torch.float16)
    expected[:7, :61] = inp_cpu[4:11, 32:93]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    logging.info("test_tile_subview_dn passed.")


# ===========================================================================
# Test T05: NZ subview — layout-aware element offset and dtype byte conversion
#
# Physical Tile [64, 4*C0] uses runtime valid_shape [48, 3*C0]. Its subview
# [16:64, C0:4*C0] is clamped to [32, 2*C0], spanning two M and two N
# fractals. Parameterization covers byte widths 32/16/8/4; FP4 detects whether
# the logical element offset is converted to bytes with /2.
# ===========================================================================
def _make_nz_subview_kernel(name, dtype, c0):
    @pl.jit(auto_mutex=True, name=name)
    def kernel(
        inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype, pl.NZ],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], dtype, pl.NZ],
    ):
        tile_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 4 * c0],
                valid_shape=[-1, -1],
                dtype=dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x0000,
            mutex_ids=[0],
        )
        with pl.section_vector():
            tile = tile_group.next()
            pl.set_validshape(tile, [48, 3 * c0])
            sub = tile[16:64, c0:4 * c0]
            pl.load(sub, inp, [16, c0])
            pl.store(out, sub, [16, c0])

    return kernel


_NZ_SUBVIEW_CASES = [
    ("fp32", pl.DT_FP32, torch.float32, 8),
    ("fp16", pl.DT_FP16, torch.float16, 16),
    ("int8", pl.DT_INT8, torch.int8, 32),
    ("fp8_e4m3", pl.DT_FP8E4M3FN, torch.float8_e4m3fn, 32),
    ("fp4_e2m1", pl.DT_FP4E2M1, torch.float4_e2m1fn_x2, 64),
]
_NZ_SUBVIEW_KERNELS = {
    label: _make_nz_subview_kernel(f"tile_subview_nz_{label}", dtype, c0) for label, dtype, _, c0 in _NZ_SUBVIEW_CASES
}


def _make_nz_subview_input(dtype: torch.dtype, c0: int) -> torch.Tensor:
    shape = (80, 5 * c0)
    if dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
        logical = torch.zeros(shape, dtype=dtype)
        storage_bytes = logical.numel() // 2 if dtype == torch.float4_e2m1fn_x2 else logical.numel()
        raw = logical.view(torch.uint8).flatten()[:storage_bytes]
        raw.copy_((torch.arange(storage_bytes, dtype=torch.int64) % 112 + 8).to(torch.uint8))
        return logical
    values = torch.arange(shape[0] * shape[1], dtype=torch.int64).reshape(shape) % 127
    return values.to(dtype)


def _make_nz_subview_expected(logical: torch.Tensor, c0: int) -> torch.Tensor:
    expected = torch.zeros(logical.shape, dtype=logical.dtype)
    if logical.dtype != torch.float4_e2m1fn_x2:
        expected[16:48, c0:3 * c0] = logical[16:48, c0:3 * c0]
        return expected

    source_bytes = logical.view(torch.uint8).flatten()[:logical.numel() // 2].reshape(80, 5 * c0 // 2)
    expected_bytes = expected.view(torch.uint8).flatten()[:expected.numel() // 2].reshape(80, 5 * c0 // 2)
    expected_bytes[16:48, c0 // 2:3 * c0 // 2] = source_bytes[16:48, c0 // 2:3 * c0 // 2]
    return expected


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
@pytest.mark.parametrize(
    "label,pl_dtype,torch_dtype,c0",
    _NZ_SUBVIEW_CASES,
    ids=[case[0] for case in _NZ_SUBVIEW_CASES],
)
def test_tile_subview_nz_dtype_byte_offset(label, pl_dtype, torch_dtype, c0):
    del pl_dtype
    if not _is_a5():
        return

    logical = _make_nz_subview_input(torch_dtype, c0)
    inp = _to_nz(logical)
    out = _to_nz(torch.zeros(logical.shape, dtype=torch_dtype))

    _NZ_SUBVIEW_KERNELS[label](inp, out)
    torch.npu.synchronize()

    expected = _make_nz_subview_expected(logical, c0)
    _assert_nz_storage_equal(out, expected)
    logging.info("test_tile_subview_nz_dtype_byte_offset passed: %s", label)


# ===========================================================================
# Test T06: L1 NZ/ZN subviews consumed by L0A/L0B
#
# Both L1 tiles use physical shape [64, 64]. Their aligned slice starts produce
# non-aligned [45, 34] NZ and [34, 37] ZN windows spanning multiple fractals.
# Compact moves into L0A/L0B preserve the tail valid shapes before matmul.
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_subview_l1_nz_zn_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=0x4000,
        mutex_ids=[1],
    )
    a_l0_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[48, 48],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.next()
        b_l1 = b_l1_group.next()
        a_l0 = a_l0_group.next()
        b_l0 = b_l0_group.next()
        acc = acc_group.next()

        pl.load(a_l1, a, [0, 0])
        pl.load(b_l1, b, [0, 0], order=[1, 0])
        a_view = a_l1[16:61, 16:50]
        b_view = b_l1[16:50, 16:53]
        pl.set_validshape(a_l0, [45, 34])
        pl.set_validshape(b_l0, [34, 37])
        pl.set_validshape(acc, [45, 37])
        pl.move(a_l0, a_view)
        pl.move(b_l0, b_view)
        pl.matmul(acc, a_l0, b_l0)
        pl.store(out, acc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_l1_nz_zn():
    if not _is_a5():
        return

    a_cpu = ((torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) % 31) - 15).to(torch.float16)
    a = _to_nz(a_cpu)
    b_cpu = torch.eye(64, dtype=torch.float16)
    b = b_cpu.to(ST_DEVICE)
    out = torch.zeros((64, 64), dtype=torch.float32, device=ST_DEVICE)

    tile_subview_l1_nz_zn_kernel(a, b, out)
    torch.npu.synchronize()

    expected = torch.zeros((64, 64), dtype=torch.float32)
    expected[:45, :37] = a_cpu[16:61, 16:50].float() @ b_cpu[16:50, 16:53].float()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    logging.info("test_tile_subview_l1_nz_zn passed.")


# ===========================================================================
# Test T07: L0C NZ subview as matmul destination and FixPipe source
#
# Matmul writes a non-aligned [50, 35] result across three N fractals of the
# parent Acc tile. Its physical M pitch is ceil16(50) = 64. The same subview is
# stored to a GM ND window with fused ReLU and FP32->FP16 conversion; using ND
# avoids imposing GM NZ's complete-fractal transfer requirement on this test.
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_subview_l0c_store_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
        ),
        addrs=0x4000,
        mutex_ids=[1],
    )
    a_l0_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.next()
        b_l1 = b_l1_group.next()
        a_l0 = a_l0_group.next()
        b_l0 = b_l0_group.next()
        acc = acc_group.next()

        pl.load(a_l1, a, [0, 0])
        pl.load(b_l1, b, [0, 0], order=[1, 0])
        pl.set_validshape(a_l0, [50, 34])
        pl.set_validshape(b_l0, [34, 35])
        pl.move(a_l0, a_l1)
        pl.move(b_l0, b_l1)
        acc_view = acc[0:50, 16:51]
        pl.matmul(acc_view, a_l0, b_l0)
        pl.store(out, acc_view, [0, 16], relu_pre_mode=pl.ReluPreMode.NormalRelu)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_l0c_store():
    if not _is_a5():
        return

    values = (torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) % 13) - 6
    a_cpu = (values / 4).to(torch.float16)
    b_cpu = torch.eye(64, dtype=torch.float16)
    a = _to_nz(a_cpu)
    b = b_cpu.to(ST_DEVICE)
    out = torch.full((64, 80), -9, dtype=torch.float16, device=ST_DEVICE)

    tile_subview_l0c_store_kernel(a, b, out)
    torch.npu.synchronize()

    expected = torch.full((64, 80), -9, dtype=torch.float16)
    result = torch.relu(a_cpu[:50, :34].float() @ b_cpu[:34, :35].float()).to(torch.float16)
    expected[:50, 16:51] = result
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    logging.info("test_tile_subview_l0c_store passed.")


# ===========================================================================
# Test T08: Composite Cube path with M/N/K loops and aligned tails
#
# Logical work covers rows/columns/K [16:96] in 32-element steps. The final
# block is 16, so M/N/K all exercise an aligned tail. Each L1 load includes a
# 16-element prefix and the dynamic subview removes it. K accumulation uses
# Partial/Final phases; the independent T07 case covers a nonzero L0C subview.
# ===========================================================================
@pl.jit(auto_mutex=True)
def tile_subview_cube_composite_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[0x0000, 0x4000],
        mutex_ids=[0, 1],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[0x8000, 0xC000],
        mutex_ids=[2, 3],
    )
    a_l0_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[0x0000, 0x4000],
        mutex_ids=[4, 5],
    )
    b_l0_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[0x0000, 0x4000],
        mutex_ids=[6, 7],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            compact=1,
        ),
        addrs=0x0000,
        mutex_ids=[8],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0 = a_l0_group.current()
        b_l0 = b_l0_group.current()
        acc = acc_group.current()
        for mi in pl.range(16, 96, 32):
            valid_m = pl.min(32, 96 - mi)
            for ni in pl.range(16, 96, 32):
                valid_n = pl.min(32, 96 - ni)
                pl.set_validshape(acc, [valid_m, valid_n])
                for ki in pl.range(16, 96, 32):
                    valid_k = pl.min(32, 96 - ki)
                    pl.set_validshape(a_l1, [16 + valid_m, 16 + valid_k])
                    pl.set_validshape(b_l1, [16 + valid_k, 16 + valid_n])
                    pl.set_validshape(a_l0, [valid_m, valid_k])
                    pl.set_validshape(b_l0, [valid_k, valid_n])
                    pl.load(a_l1, a, [mi - 16, ki - 16])
                    pl.load(b_l1, b, [ni - 16, ki - 16], order=[1, 0])

                    a_view = a_l1[16:16 + valid_m, 16:16 + valid_k]
                    b_view = b_l1[16:16 + valid_k, 16:16 + valid_n]
                    pl.move(a_l0, a_view)
                    pl.move(b_l0, b_view)
                    if ki == 16:
                        pl.matmul(acc, a_l0, b_l0, phase=pl.AccPhase.Partial)
                    elif ki + 32 < 96:
                        pl.matmul_acc(acc, acc, a_l0, b_l0, phase=pl.AccPhase.Partial)
                    else:
                        pl.matmul_acc(acc, acc, a_l0, b_l0, phase=pl.AccPhase.Final)
                pl.store(out, acc, [mi, ni], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_tile_subview_cube_composite():
    if not _is_a5():
        return

    a_values = (torch.arange(96 * 96, dtype=torch.float32).reshape(96, 96) % 11) - 5
    b_values = (torch.arange(96 * 96, dtype=torch.float32).reshape(96, 96) % 7) - 3
    a_cpu = (a_values / 8).to(torch.float16)
    b_cpu = (b_values / 8).to(torch.float16)
    a = _to_nz(a_cpu)
    b = b_cpu.to(ST_DEVICE)
    out = torch.full((96, 96), -9, dtype=torch.float32, device=ST_DEVICE)

    tile_subview_cube_composite_kernel(a, b, out)
    torch.npu.synchronize()

    expected = torch.full((96, 96), -9, dtype=torch.float32)
    expected[16:96, 16:96] = a_cpu[16:96, 16:96].float() @ b_cpu.t()[16:96, 16:96].float()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-2, atol=1e-2)
    logging.info("test_tile_subview_cube_composite passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_tile_subview()
    test_tile_subview_valid_shape()
    test_tile_subview_vf()
    test_tile_subview_dn()
    # T05 is parameterized over C0=8/16/32/64 and the corresponding dtypes.
    for nz_case in _NZ_SUBVIEW_CASES:
        test_tile_subview_nz_dtype_byte_offset(*nz_case)
    test_tile_subview_l1_nz_zn()
    test_tile_subview_l0c_store()
    test_tile_subview_cube_composite()
    logging.info("\nAll subview tests passed!")
