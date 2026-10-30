# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end ST coverage for GM tensors whose logical layout is NZ.

The inputs are created as logical Torch tensors and packed into device format
29 according to the PTO NZ rules. Outputs are copied as raw storage and
unpacked on CPU, avoiding TorchNPU format-cast restrictions and C0 differences.
FP4 is compared through its packed physical bytes.
"""

import ctypes
import os
import struct

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"
SENTINEL = -9.0


def _require_a5(device: str) -> None:
    try:
        torch.npu.set_device(device)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("NZ GM transfer ST requires Ascend950")


@pytest.fixture(scope="module")
def device() -> str:
    _require_a5(ST_DEVICE)
    return ST_DEVICE


def _to_nz(logical: torch.Tensor, device: str | torch.device | None = None) -> torch.Tensor:
    logical_cpu = logical.detach().cpu()
    target_device = logical.device if device is None else device
    nz = torch_npu.empty_with_format(
        list(logical.shape),
        dtype=logical.dtype,
        device=target_device,
        acl_format=29,
    )
    _copy_raw_bytes_to_npu_storage(nz, _pack_nz_cpu(logical_cpu))
    assert torch_npu.get_npu_format(nz) == 29
    return nz


def _nz_filled(shape: tuple[int, ...], dtype: torch.dtype, value: int | float, device: str) -> torch.Tensor:
    return _to_nz(torch.full(shape, value, dtype=dtype), device)


def _matrix(rows: int, cols: int, dtype: torch.dtype, device: str, bias: float = 0.0) -> torch.Tensor:
    values = (torch.arange(rows * cols, device=device, dtype=torch.float32) % 17 - 8) / 8 + bias
    return values.reshape(rows, cols).to(dtype)


def _diagonal(size: int, dtype: torch.dtype, device: str, factor: float = 1.0) -> torch.Tensor:
    diagonal = torch.linspace(0.5, 1.5, size, dtype=torch.float32, device=device) * factor
    return torch.diag(diagonal).to(dtype)


def _copy_raw_npu_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Copy device storage bytes without asking TorchNPU to recover its format."""
    # Raw ACL copies are outside TorchNPU's stream tracking.  Complete pending
    # kernel writes before reading storage that may come from the caching allocator.
    torch.npu.synchronize()
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    host = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    acl = ctypes.CDLL("libascendcl.so")
    aclrt_memcpy = acl.aclrtMemcpy
    aclrt_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    aclrt_memcpy.restype = ctypes.c_int
    # aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST
    result = aclrt_memcpy(host.data_ptr(), nbytes, storage.data_ptr(), nbytes, 2)
    assert result == 0, f"aclrtMemcpy device-to-host failed with error {result}"
    return host


def _copy_raw_bytes_to_npu_storage(tensor: torch.Tensor, source: torch.Tensor) -> None:
    """Initialize an internal-format tensor from its physical host bytes."""
    # The caching allocator may reuse storage touched by an earlier asynchronous
    # kernel.  Order that work before the raw ACL initialization.
    torch.npu.synchronize()
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    source = source.contiguous().view(torch.uint8).flatten()
    assert source.device.type == "cpu"
    assert source.numel() <= nbytes
    # Torch represents two logical FP4 values with an x2 dtype whose nominal
    # storage can be larger than the packed payload used by CCE. Zero the full
    # allocation so unused bytes remain deterministic for raw comparisons.
    host = torch.zeros(nbytes, dtype=torch.uint8, device="cpu")
    host[:source.numel()].copy_(source)
    acl = ctypes.CDLL("libascendcl.so")
    aclrt_memcpy = acl.aclrtMemcpy
    aclrt_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    aclrt_memcpy.restype = ctypes.c_int
    # aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE
    result = aclrt_memcpy(storage.data_ptr(), nbytes, host.data_ptr(), nbytes, 1)
    assert result == 0, f"aclrtMemcpy host-to-device failed with error {result}"


def _nz_c0(dtype: torch.dtype) -> int:
    if dtype == torch.float4_e2m1fn_x2:
        return 64
    return 32 // torch.empty((), dtype=dtype).element_size()


def _pack_nz_cpu(logical: torch.Tensor) -> torch.Tensor:
    """Pack a logical CPU tensor into its physically padded NZ storage."""
    assert logical.device.type == "cpu"
    logical = logical.contiguous()
    rows, cols = logical.shape[-2:]
    c0 = _nz_c0(logical.dtype)
    batch = logical.numel() // (rows * cols)

    if logical.dtype == torch.float4_e2m1fn_x2:
        # Torch exposes packed FP4 through an x2 byte dtype while the tensor
        # shape still counts logical FP4 elements. Keep pairs within each row.
        assert cols % 2 == 0
        padded_rows = (rows + 15) // 16 * 16
        padded_cols = (cols + c0 - 1) // c0 * c0
        packed_elements = logical.numel() // 2
        logical_bytes = logical.view(torch.uint8).flatten()[:packed_elements].reshape(batch, rows, cols // 2)
        padded_bytes = torch.zeros((batch, padded_rows, padded_cols // 2), dtype=torch.uint8)
        padded_bytes[:, :rows, :cols // 2] = logical_bytes
        return (
            padded_bytes.reshape(batch, padded_rows // 16, 16, padded_cols // c0, c0 // 2)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .flatten()
        )

    padded_rows = (rows + 15) // 16 * 16
    padded_cols = (cols + c0 - 1) // c0 * c0
    padded = torch.zeros((batch, padded_rows, padded_cols), dtype=logical.dtype)
    padded[:, :rows, :cols] = logical.reshape(batch, rows, cols)
    return (
        padded.reshape(batch, padded_rows // 16, 16, padded_cols // c0, c0)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(torch.uint8)
        .flatten()
    )


def _unpack_nz_cpu(storage_bytes: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Restore logical last-two-axis order from physical NZ storage bytes."""
    rows, cols = shape[-2:]
    c0 = _nz_c0(dtype)
    padded_rows = (rows + 15) // 16 * 16
    padded_cols = (cols + c0 - 1) // c0 * c0
    batch = 1
    for dim in shape[:-2]:
        batch *= dim
    element_size = torch.empty((), dtype=dtype).element_size()
    payload_bytes = batch * padded_rows * padded_cols * element_size
    physical = storage_bytes[:payload_bytes].view(dtype)
    padded = (
        physical.reshape(batch, padded_cols // c0, padded_rows // 16, 16, c0)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(batch, padded_rows, padded_cols)
    )
    return padded[:, :rows, :cols].reshape(shape)


def _assert_nz_close(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 0,
    atol: float = 0,
) -> None:
    assert torch_npu.get_npu_format(actual) == 29
    actual_bytes = _copy_raw_npu_storage(actual)
    if actual.dtype == torch.float4_e2m1fn_x2:
        expected_bytes = _pack_nz_cpu(expected)
        actual_bytes = actual_bytes[:expected_bytes.numel()]
        try:
            torch.testing.assert_close(actual_bytes, expected_bytes, rtol=0, atol=0)
        except AssertionError:
            mismatch = actual_bytes != expected_bytes
            first_mismatch = mismatch.nonzero()[0].tolist()
            print(
                f"[FAIL] {label}: NZ storage mismatch, first_byte={first_mismatch[0]}",
                flush=True,
            )
            raise
        print(
            f"[PASS] {label}: shape={tuple(actual.shape)}, dtype={actual.dtype}, raw storage equal",
            flush=True,
        )
        return

    actual_cpu = _unpack_nz_cpu(actual_bytes, tuple(actual.shape), actual.dtype)
    assert expected.device.type == "cpu"
    expected_cpu = expected.detach()
    if actual_cpu.dtype == torch.float8_e4m3fn:
        actual_bytes = actual_cpu.view(torch.uint8)
        expected_bytes = expected_cpu.view(torch.uint8)
        try:
            torch.testing.assert_close(actual_bytes, expected_bytes, rtol=0, atol=0)
        except AssertionError:
            mismatch = actual_bytes != expected_bytes
            first_mismatch = mismatch.nonzero()[0].tolist()
            print(
                f"[FAIL] {label}: raw storage mismatch, first_mismatch={first_mismatch}",
                flush=True,
            )
            raise
        print(
            f"[PASS] {label}: shape={tuple(actual_cpu.shape)}, dtype={actual_cpu.dtype}, raw storage equal",
            flush=True,
        )
        return
    if actual_cpu.is_floating_point():
        diff = (actual_cpu.to(torch.float64) - expected_cpu.to(torch.float64)).abs()
    else:
        diff = (actual_cpu.to(torch.int64) - expected_cpu.to(torch.int64)).abs()
    max_abs_diff = diff.max().item() if diff.numel() else 0
    try:
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=rtol, atol=atol)
    except AssertionError:
        actual_view = actual_cpu.reshape(-1, actual_cpu.shape[-1])
        expected_view = expected_cpu.reshape(-1, expected_cpu.shape[-1])
        threshold = atol + rtol * expected_cpu.to(torch.float64).abs()
        mismatch = diff > threshold
        first_mismatch = mismatch.nonzero()[0].tolist()
        print(
            f"[FAIL] {label}: shape={tuple(actual_cpu.shape)}, dtype={actual_cpu.dtype}, "
            f"max_abs_diff={max_abs_diff:.6g}, first_mismatch={first_mismatch}, "
            f"rtol={rtol:g}, atol={atol:g}",
            flush=True,
        )
        print(f"actual[:4, :8]=\n{actual_view[:4, :8]}", flush=True)
        print(f"golden[:4, :8]=\n{expected_view[:4, :8]}", flush=True)
        mismatch_view = mismatch.reshape(-1, mismatch.shape[-1])
        mismatch_row = mismatch_view.nonzero()[0, 0].item()
        row_end = min(mismatch_row + 4, actual_view.shape[0])
        print(
            f"actual[{mismatch_row}:{row_end}, :8]=\n{actual_view[mismatch_row:row_end, :8]}",
            flush=True,
        )
        print(
            f"golden[{mismatch_row}:{row_end}, :8]=\n{expected_view[mismatch_row:row_end, :8]}",
            flush=True,
        )
        raise
    print(
        f"[PASS] {label}: shape={tuple(actual_cpu.shape)}, dtype={actual_cpu.dtype}, "
        f"max_abs_diff={max_abs_diff:.6g}, rtol={rtol:g}, atol={atol:g}",
        flush=True,
    )


def _make_nz_e2e_kernel(
    name: str,
    valid_m: int,
    valid_n: int,
    row_offset: int,
    col_offset: int,
):
    @pl.jit(auto_mutex=True, name=name)
    def kernel(
        a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
        b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
        vec_in: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
        cube_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
        vec_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    ):
        a_l1_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[valid_m, 64],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x0000,
            mutex_ids=[0],
        )
        b_l1_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, valid_n],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x2000,
            mutex_ids=[1],
        )
        a_l0a_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[valid_m, 64],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            ),
            addrs=0x0000,
            mutex_ids=[2],
        )
        b_l0b_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, valid_n],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
            ),
            addrs=0x0000,
            mutex_ids=[3],
        )
        acc_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[valid_m, valid_n],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
            ),
            addrs=0x0000,
            mutex_ids=[4],
        )
        vec_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[valid_m, valid_n],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x0000,
            mutex_ids=[5],
        )

        with pl.section_cube():
            a_l1 = a_l1_group.current()
            b_l1 = b_l1_group.current()
            a_l0a = a_l0a_group.current()
            b_l0b = b_l0b_group.current()
            acc = acc_group.current()
            pl.load(a_l1, a, [row_offset, 0])
            pl.load(b_l1, b, [0, col_offset])
            pl.move(a_l0a, a_l1)
            pl.move(b_l0b, b_l1)
            pl.matmul(acc, a_l0a, b_l0b)
            pl.store(cube_out, acc, [row_offset, col_offset])

        with pl.section_vector():
            vec = vec_group.current()
            pl.load(vec, vec_in, [row_offset, col_offset])
            pl.store(vec_out, vec, [row_offset, col_offset])

    return kernel


_NZ_E2E_CASES = [
    ("full", _make_nz_e2e_kernel("nz_dynamic_full", 64, 64, 0, 0), 64, 64, 0, 0),
    ("aligned_tail", _make_nz_e2e_kernel("nz_dynamic_tail", 32, 16, 32, 48), 32, 16, 32, 48),
]


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
@pytest.mark.parametrize(
    "kernel,valid_m,valid_n,row_offset,col_offset",
    [case[1:] for case in _NZ_E2E_CASES],
    ids=[case[0] for case in _NZ_E2E_CASES],
)
def test_t01_2d_dynamic_nz_e2e(
    device: str,
    kernel,
    valid_m: int,
    valid_n: int,
    row_offset: int,
    col_offset: int,
) -> None:
    a_logical = _matrix(64, 64, torch.float16, device)
    b_logical = _diagonal(64, torch.float16, device)
    vec_logical = _matrix(64, 64, torch.float16, device, bias=2.0)
    cube_out = _nz_filled((64, 64), torch.float16, SENTINEL, device)
    vec_out = _nz_filled((64, 64), torch.float16, SENTINEL, device)

    a_nz = _to_nz(a_logical)
    b_nz = _to_nz(b_logical)
    vec_in_nz = _to_nz(vec_logical)
    kernel(
        a_nz,
        b_nz,
        vec_in_nz,
        cube_out,
        vec_out,
    )
    torch.npu.synchronize()

    a_cpu = a_logical.cpu()
    b_cpu = b_logical.cpu()
    vec_cpu = vec_logical.cpu()
    cube_expected = torch.full((64, 64), SENTINEL, dtype=torch.float16)
    vec_expected = torch.full((64, 64), SENTINEL, dtype=torch.float16)
    cube_expected[row_offset:row_offset + valid_m, col_offset:col_offset + valid_n] = (
        a_cpu[row_offset:row_offset + valid_m].float()
        @ b_cpu[:, col_offset:col_offset + valid_n].float()
    ).to(torch.float16)
    vec_expected[row_offset:row_offset + valid_m, col_offset:col_offset + valid_n] = vec_cpu[
        row_offset:row_offset + valid_m, col_offset:col_offset + valid_n
    ]
    case = f"t01 full/tail m={valid_m} n={valid_n} offset=({row_offset},{col_offset})"
    _assert_nz_close(f"{case} vector", vec_out, vec_expected)
    _assert_nz_close(f"{case} cube", cube_out, cube_expected, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def nz_vec_non_aligned_logical_shape_kernel(
    inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
):
    vec_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[16, 16],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )

    with pl.section_vector():
        vec = vec_group.current()
        pl.load(vec, inp, [64, 48])
        pl.store(out, vec, [64, 48])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t01_2d_dynamic_nz_non_aligned_logical_shape(device: str) -> None:
    logical = _matrix(70, 50, torch.float16, device, bias=2.0)
    inp = _to_nz(logical)
    out = _nz_filled((70, 50), torch.float16, SENTINEL, device)

    nz_vec_non_aligned_logical_shape_kernel(inp, out)
    torch.npu.synchronize()

    expected = torch.full((70, 50), SENTINEL, dtype=torch.float16)
    expected[64:70, 48:50] = logical.cpu()[64:70, 48:50]
    _assert_nz_close("t01 non-aligned logical shape [70, 50]", out, expected)


@pl.jit(auto_mutex=True)
def nz_cube_mnk_tiled_tail_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
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
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 16],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=0x2000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
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
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 16],
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
            shape=[64, 16],
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
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        acc = acc_group.current()
        m_blocks = (out.shape[0] + 63) // 64
        n_blocks = (out.shape[1] + 15) // 16
        k_blocks = (a.shape[1] + 63) // 64
        for m_idx in pl.range(0, m_blocks):
            valid_m = pl.min(64, out.shape[0] - m_idx * 64)
            for n_idx in pl.range(0, n_blocks):
                valid_n = pl.min(16, out.shape[1] - n_idx * 16)
                pl.set_validshape(acc, [valid_m, valid_n])
                for k_idx in pl.range(0, k_blocks):
                    valid_k = pl.min(64, a.shape[1] - k_idx * 64)
                    pl.set_validshape(a_l1, [valid_m, valid_k])
                    pl.set_validshape(a_l0a, [valid_m, valid_k])
                    pl.set_validshape(b_l1, [valid_k, valid_n])
                    pl.set_validshape(b_l0b, [valid_k, valid_n])
                    pl.load_tile(a_l1, a, [m_idx, k_idx])
                    pl.load_tile(b_l1, b, [k_idx, n_idx])
                    pl.move(a_l0a, a_l1)
                    pl.move(b_l0b, b_l1)
                    if k_idx == 0:
                        if k_blocks == 1:
                            pl.matmul(acc, a_l0a, b_l0b, phase=pl.AccPhase.Final)
                        else:
                            pl.matmul(acc, a_l0a, b_l0b, phase=pl.AccPhase.Partial)
                    elif k_idx == k_blocks - 1:
                        pl.matmul_acc(acc, acc, a_l0a, b_l0b, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul_acc(acc, acc, a_l0a, b_l0b, phase=pl.AccPhase.Partial)
                pl.store_tile(out, acc, [m_idx, n_idx], phase=pl.STPhase.Final)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t01_2d_nz_mnk_tiled_tail(device: str) -> None:
    a_logical = _matrix(96, 96, torch.float16, device)
    b_logical = _matrix(96, 32, torch.float16, device, bias=0.5)
    out = _nz_filled((96, 32), torch.float16, SENTINEL, device)

    a_nz = _to_nz(a_logical)
    b_nz = _to_nz(b_logical)
    nz_cube_mnk_tiled_tail_kernel(a_nz, b_nz, out)
    torch.npu.synchronize()

    expected = (a_logical.cpu().float() @ b_logical.cpu().float()).to(torch.float16)
    _assert_nz_close("t01 2D M/N/K tiled matmul", out, expected, rtol=1e-2, atol=1e-2)


def _make_vec_roundtrip_kernel(
    name: str,
    pl_dtype,
    tile_m: int,
    tile_n: int,
    element_offset: tuple[int, int],
    *,
    use_tile_offsets: bool = False,
):
    @pl.jit(auto_mutex=True, name=name)
    def kernel(
        inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl_dtype, pl.NZ],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl_dtype, pl.NZ],
    ):
        vec_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[tile_m, tile_n],
                dtype=pl_dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=0x0000,
            mutex_ids=[0],
        )
        with pl.section_vector():
            vec = vec_group.current()
            if use_tile_offsets:
                pl.load_tile(vec, inp, [1, 1])
                pl.store_tile(out, vec, [1, 1])
            else:
                pl.load(vec, inp, [element_offset[0], element_offset[1]])
                pl.store(out, vec, [element_offset[0], element_offset[1]])

    return kernel


vec_fp16_element_kernel = _make_vec_roundtrip_kernel(
    "vec_nz_element_offset", pl.DT_FP16, 32, 32, (32, 32)
)
vec_fp16_tile_kernel = _make_vec_roundtrip_kernel(
    "vec_nz_tile_offset", pl.DT_FP16, 32, 32, (32, 32), use_tile_offsets=True
)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t02_vec_nz_element_and_tile_offsets(device: str) -> None:
    logical = _matrix(96, 96, torch.float16, device)
    inp = _to_nz(logical)
    element_out = _nz_filled((96, 96), torch.float16, SENTINEL, device)
    tile_out = _nz_filled((96, 96), torch.float16, SENTINEL, device)

    vec_fp16_element_kernel(inp, element_out)
    vec_fp16_tile_kernel(inp, tile_out)
    torch.npu.synchronize()

    logical_cpu = logical.cpu()
    expected = torch.full((96, 96), SENTINEL, dtype=torch.float16)
    expected[32:64, 32:64] = logical_cpu[32:64, 32:64]
    _assert_nz_close("t02 vector element offsets", element_out, expected)
    _assert_nz_close("t02 vector tile offsets", tile_out, expected)


_VEC_DTYPE_CASES = [
    ("int8", pl.DT_INT8, torch.int8, 32),
    ("fp8_e4m3", pl.DT_FP8E4M3FN, torch.float8_e4m3fn, 32),
    ("hf8", pl.DT_HF8, torch.uint8, 32),
    ("fp4_e2m1", pl.DT_FP4E2M1, torch.float4_e2m1fn_x2, 64),
    ("fp16", pl.DT_FP16, torch.float16, 16),
    ("fp32", pl.DT_FP32, torch.float32, 8),
]
_VEC_DTYPE_KERNELS = {
    label: _make_vec_roundtrip_kernel(
        f"vec_nz_{label}_c0", pl_dtype, 32, 2 * c0, (16, c0)
    )
    for label, pl_dtype, _, c0 in _VEC_DTYPE_CASES
}


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
@pytest.mark.parametrize(
    "label,pl_dtype,torch_dtype,c0",
    _VEC_DTYPE_CASES,
    ids=[case[0] for case in _VEC_DTYPE_CASES],
)
def test_t02_vec_nz_dtype_c0(
    device: str,
    label: str,
    pl_dtype,
    torch_dtype: torch.dtype,
    c0: int,
) -> None:
    del pl_dtype
    shape = (70, 2 * c0 + 2) if torch_dtype == torch.float4_e2m1fn_x2 else (64, 4 * c0)
    # Torch has no native HF8 dtype, so HF8 values are carried as raw uint8 codes.
    is_low_precision = torch_dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.uint8)
    if is_low_precision:
        logical_cpu = torch.zeros(shape, dtype=torch_dtype)
        logical_storage = logical_cpu.view(torch.uint8).flatten()
        valid_storage_bytes = (
            logical_cpu.numel() // 2
            if torch_dtype == torch.float4_e2m1fn_x2
            else logical_cpu.numel()
        )
        logical_bytes = logical_storage[:valid_storage_bytes].reshape(shape[0], -1)
        pattern = torch.arange(logical_bytes.numel(), dtype=torch.int64).reshape(logical_bytes.shape)
        logical_bytes.copy_((pattern % 112 + 8).to(torch.uint8))
        logical = logical_cpu
    else:
        logical = torch.arange(shape[0] * shape[1], dtype=torch.int64, device=device).reshape(shape).to(torch_dtype)
        logical_cpu = logical.cpu()
    inp = _to_nz(logical, device)
    if is_low_precision:
        out = _to_nz(torch.zeros(shape, dtype=torch_dtype), device)
    else:
        sentinel = -9 if not torch_dtype.is_floating_point else SENTINEL
        out = _nz_filled(shape, torch_dtype, sentinel, device)

    _VEC_DTYPE_KERNELS[label](inp, out)
    torch.npu.synchronize()

    if is_low_precision:
        expected = torch.zeros(shape, dtype=torch_dtype)
        valid_storage_bytes = (
            expected.numel() // 2 if torch_dtype == torch.float4_e2m1fn_x2 else expected.numel()
        )
        expected_bytes = expected.view(torch.uint8).flatten()[:valid_storage_bytes].reshape(shape[0], -1)
        logical_bytes = logical_cpu.view(torch.uint8).flatten()[:valid_storage_bytes].reshape(shape[0], -1)
        logical_elements_per_byte = 2 if torch_dtype == torch.float4_e2m1fn_x2 else 1
        byte_start = c0 // logical_elements_per_byte
        byte_end = 3 * c0 // logical_elements_per_byte
        expected_bytes[16:48, byte_start:byte_end] = logical_bytes[16:48, byte_start:byte_end]
    else:
        expected = torch.full(shape, sentinel, dtype=torch_dtype)
        expected[16:48, c0:3 * c0] = logical_cpu[16:48, c0:3 * c0]
    _assert_nz_close(f"t02 vector dtype/C0 {label}", out, expected)


@pl.jit(auto_mutex=True)
def vec_nz_runtime_windows_kernel(
    inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
):
    vec_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
            compact=0,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    with pl.section_vector():
        vec = vec_group.current()
        pl.set_validshape(vec, [64, 32])
        pl.load(vec, inp, [16, 16])
        pl.store(out, vec, [16, 16])
        pl.set_validshape(vec, [32, 64])
        pl.load(vec, inp, [32, 32])
        pl.store(out, vec, [32, 32])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t02_vec_nz_runtime_windows(device: str) -> None:
    logical = _matrix(96, 96, torch.float16, device)
    out = _nz_filled((96, 96), torch.float16, SENTINEL, device)

    inp = _to_nz(logical)
    vec_nz_runtime_windows_kernel(inp, out)
    torch.npu.synchronize()

    logical_cpu = logical.cpu()
    expected = torch.full((96, 96), SENTINEL, dtype=torch.float16)
    expected[16:80, 16:48] = logical_cpu[16:80, 16:48]
    expected[32:64, 32:96] = logical_cpu[32:64, 32:96]
    _assert_nz_close("t02 vector runtime windows", out, expected)


def _scale_params(scale_values: list[float], device: str) -> torch.Tensor:
    encoded = []
    for value in scale_values:
        bits = struct.unpack("!I", struct.pack("!f", value))[0]
        encoded.append((1 << 46) | bits)
    return torch.tensor(encoded, dtype=torch.int64, device=device).reshape(1, -1)


@pl.jit(auto_mutex=True)
def l0c_nz_direct_store_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    scale_params: pl.Tensor[[1, 32], pl.DT_INT64],
    plain_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    scalar_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8, pl.NZ],
    channel_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8, pl.NZ],
    relu_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x1000,
        mutex_ids=[1],
    )
    b_plain_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x2000,
        mutex_ids=[2],
    )
    scale_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND),
        addrs=0x2800,
        mutex_ids=[3],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000,
        mutex_ids=[5],
    )
    b_plain_l0b_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x1000,
        mutex_ids=[6],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x0000,
        mutex_ids=[7],
    )
    plain_acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 16],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x1000,
        mutex_ids=[8],
    )
    scale_tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling),
        addrs=0x0000,
        mutex_ids=[9],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        b_plain_l1 = b_plain_l1_group.current()
        scale_l1 = scale_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        b_plain_l0b = b_plain_l0b_group.current()
        acc = acc_group.current()
        plain_acc = plain_acc_group.current()
        scale_tile = scale_tile_group.current()
        pl.load(a_l1, a, [32, 0])
        pl.load(b_l1, b, [0, 32])
        pl.load(b_plain_l1, b, [0, 32])
        pl.load(scale_l1, scale_params, [0, 0])
        pl.move(a_l0a, a_l1)
        pl.move(b_l0b, b_l1)
        pl.move(b_plain_l0b, b_plain_l1)
        pl.matmul(acc, a_l0a, b_l0b)
        pl.matmul(plain_acc, a_l0a, b_plain_l0b)
        pl.move(scale_tile, scale_l1)
        pl.store(plain_out, plain_acc, [32, 32])
        pl.store(scalar_out, acc, [32, 32], scale=0.5)
        pl.store(channel_out, acc, [32, 32], scale=scale_tile)
        pl.store(relu_out, plain_acc, [32, 32], relu_pre_mode=pl.ReluPreMode.NormalRelu)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t03_l0c_nz_direct_store_variants(device: str) -> None:
    a_logical = _matrix(64, 64, torch.float16, device)
    b_logical = torch.eye(64, dtype=torch.float16, device=device)
    scale_values = ([0.25, 0.5, 1.0, 2.0] * 8)[:32]
    plain_out = _nz_filled((64, 64), torch.float16, SENTINEL, device)
    scalar_out = _nz_filled((64, 64), torch.int8, -9, device)
    channel_out = _nz_filled((64, 64), torch.int8, -9, device)
    relu_out = _nz_filled((64, 64), torch.float16, SENTINEL, device)

    a_nz = _to_nz(a_logical)
    b_nz = _to_nz(b_logical)
    scale_params = _scale_params(scale_values, device)
    l0c_nz_direct_store_kernel(
        a_nz,
        b_nz,
        scale_params,
        plain_out,
        scalar_out,
        channel_out,
        relu_out,
    )
    torch.npu.synchronize()

    a_cpu = a_logical.cpu()
    b_cpu = b_logical.cpu()
    raw = a_cpu[32:64].float() @ b_cpu[:, 32:64].float()
    plain_expected = torch.full((64, 64), SENTINEL, dtype=torch.float16)
    scalar_expected = torch.full((64, 64), -9, dtype=torch.int8)
    channel_expected = torch.full((64, 64), -9, dtype=torch.int8)
    relu_expected = torch.full((64, 64), SENTINEL, dtype=torch.float16)
    plain_expected[32:64, 32:48] = raw[:, :16].to(torch.float16)
    scalar_expected[32:64, 32:64] = torch.clamp(torch.round(raw * 0.5), -128, 127).to(torch.int8)
    scales = torch.tensor(scale_values, dtype=torch.float32).reshape(1, 32)
    channel_expected[32:64, 32:64] = torch.clamp(torch.round(raw * scales), -128, 127).to(torch.int8)
    relu_expected[32:64, 32:48] = torch.relu(raw[:, :16]).to(torch.float16)
    _assert_nz_close("t03 L0C direct plain", plain_out, plain_expected, rtol=1e-2, atol=1e-2)
    _assert_nz_close("t03 L0C direct scalar quant", scalar_out, scalar_expected, atol=1)
    _assert_nz_close("t03 L0C direct per-channel quant", channel_out, channel_expected, atol=1)
    _assert_nz_close("t03 L0C direct ReLU", relu_out, relu_expected, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def l0c_nz_phase_store_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8, pl.NZ],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x1000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        acc = acc_group.current()
        pl.load(a_l1, a, [32, 0])
        pl.load(b_l1, b, [0, 32])
        pl.move(a_l0a, a_l1)
        pl.move(b_l0b, b_l1)
        pl.matmul(acc, a_l0a, b_l0b, phase=pl.AccPhase.Partial)
        pl.matmul_acc(acc, acc, a_l0a, b_l0b, phase=pl.AccPhase.Final)
        pl.store(out, acc, [32, 32], scale=0.5, phase=pl.STPhase.Partial)
        pl.store(out, acc, [32, 32], scale=0.5, phase=pl.STPhase.Final)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t03_l0c_nz_partial_final_phase(device: str) -> None:
    a_logical = _matrix(64, 64, torch.float16, device)
    b_logical = torch.eye(64, dtype=torch.float16, device=device)
    out = _nz_filled((64, 64), torch.int8, -9, device)

    a_nz = _to_nz(a_logical)
    b_nz = _to_nz(b_logical)
    l0c_nz_phase_store_kernel(a_nz, b_nz, out)
    torch.npu.synchronize()

    raw = 2 * (a_logical.cpu()[32:64].float() @ b_logical.cpu()[:, 32:64].float())
    expected = torch.full((64, 64), -9, dtype=torch.int8)
    expected[32:64, 32:64] = torch.clamp(torch.round(raw * 0.5), -128, 127).to(torch.int8)
    _assert_nz_close("t03 L0C Partial/Final phase", out, expected, atol=1)


@pl.jit(auto_mutex=True)
def l0c_nz_via_vec_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32, pl.NZ],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x1000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[5],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        acc = acc_group.current()
        pl.load(a_l1, a, [32, 0])
        pl.load(b_l1, b, [0, 32])
        pl.move(a_l0a, a_l1)
        pl.move(b_l0b, b_l1)
        pl.matmul(acc, a_l0a, b_l0b)
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=1)
        pl.move(vec_group[0], acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, vec_group[0], [32, 32])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t03_l0c_nz_partial_m_multi_fractal_via_vec(device: str) -> None:
    a_logical = _matrix(64, 64, torch.float16, device)
    b_logical = torch.eye(64, dtype=torch.float16, device=device)
    out = _nz_filled((64, 64), torch.float32, SENTINEL, device)

    a_nz = _to_nz(a_logical)
    b_nz = _to_nz(b_logical)
    l0c_nz_via_vec_kernel(a_nz, b_nz, out)
    torch.npu.synchronize()

    expected = torch.full((64, 64), SENTINEL, dtype=torch.float32)
    expected[32:64, 32:64] = a_logical.cpu()[32:64].float() @ b_logical.cpu()[:, 32:64].float()
    _assert_nz_close("t03 L0C via UB", out, expected, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def l0c_nz_multicore_atomic_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32, pl.NZ],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x1000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 16],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    k_offset = core_id * 64
    with pl.section_cube():
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        acc = acc_group.current()
        pl.load(a_l1, a, [0, k_offset])
        pl.load(b_l1, b, [k_offset, 16])
        pl.move(a_l0a, a_l1)
        pl.move(b_l0b, b_l1)
        pl.matmul(acc, a_l0a, b_l0b)
        pl.store(out, acc, [0, 16], atomic=pl.AtomicType.AtomicAdd)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t03_l0c_nz_multicore_atomic_tail(device: str) -> None:
    a_logical = _matrix(32, 128, torch.float16, device)
    b_logical = _matrix(128, 32, torch.float16, device, bias=0.25)
    out = _nz_filled((32, 32), torch.float32, 0.0, device)

    a_nz = _to_nz(a_logical)
    b_nz = _to_nz(b_logical)
    l0c_nz_multicore_atomic_kernel[None, 2](a_nz, b_nz, out)
    torch.npu.synchronize()

    expected = torch.zeros((32, 32), dtype=torch.float32)
    expected[:, 16:32] = a_logical.cpu().float() @ b_logical.cpu()[:, 16:32].float()
    _assert_nz_close("t03 multicore atomic store", out, expected, rtol=1e-2, atol=1e-2)


@pl.jit(auto_mutex=True)
def nz_high_dimensional_mnk_quant_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    cube_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8, pl.NZ],
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
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 32],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=0x2000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
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
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 32],
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
            shape=[64, 32],
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
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        acc = acc_group.current()
        m_blocks = (cube_out.shape[2] + 63) // 64
        n_blocks = (cube_out.shape[3] + 31) // 32
        k_blocks = (a.shape[3] + 63) // 64
        for batch in pl.range(0, cube_out.shape[0]):
            for head in pl.range(0, cube_out.shape[1]):
                for m_idx in pl.range(0, m_blocks):
                    valid_m = pl.min(64, cube_out.shape[2] - m_idx * 64)
                    for n_idx in pl.range(0, n_blocks):
                        valid_n = pl.min(32, cube_out.shape[3] - n_idx * 32)
                        pl.set_validshape(acc, [valid_m, valid_n])
                        for k_idx in pl.range(0, k_blocks):
                            valid_k = pl.min(64, a.shape[3] - k_idx * 64)
                            pl.set_validshape(a_l1, [valid_m, valid_k])
                            pl.set_validshape(a_l0a, [valid_m, valid_k])
                            pl.set_validshape(b_l1, [valid_k, valid_n])
                            pl.set_validshape(b_l0b, [valid_k, valid_n])
                            pl.load_tile(a_l1, a, [batch, head, m_idx, k_idx])
                            pl.load_tile(b_l1, b, [batch, head, k_idx, n_idx])
                            pl.move(a_l0a, a_l1)
                            pl.move(b_l0b, b_l1)
                            if k_idx == 0:
                                if k_blocks == 1:
                                    pl.matmul(acc, a_l0a, b_l0b, phase=pl.AccPhase.Final)
                                else:
                                    pl.matmul(acc, a_l0a, b_l0b, phase=pl.AccPhase.Partial)
                            elif k_idx == k_blocks - 1:
                                pl.matmul_acc(acc, acc, a_l0a, b_l0b, phase=pl.AccPhase.Final)
                            else:
                                pl.matmul_acc(acc, acc, a_l0a, b_l0b, phase=pl.AccPhase.Partial)
                        pl.store_tile(
                            cube_out,
                            acc,
                            [batch, head, m_idx, n_idx],
                            scale=0.125,
                            phase=pl.STPhase.Final,
                        )

@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t04_high_dimensional_dynamic_nz_mnk_quantized(device: str) -> None:
    a_shape = (2, 3, 96, 96)
    b_shape = (2, 3, 96, 64)
    cube_shape = (2, 3, 96, 64)
    a_logical = torch.empty(a_shape, dtype=torch.float16, device=device)
    b_logical = torch.empty(b_shape, dtype=torch.float16, device=device)
    for batch in range(a_shape[0]):
        for head in range(a_shape[1]):
            a_logical[batch, head] = _matrix(96, 96, torch.float16, device, bias=batch + head / 4)
            b_logical[batch, head] = _matrix(
                96, 64, torch.float16, device, bias=1 + batch + head / 4
            )

    cube_out = _nz_filled(cube_shape, torch.int8, -9, device)

    a_nz = _to_nz(a_logical)
    b_nz = _to_nz(b_logical)
    nz_high_dimensional_mnk_quant_kernel(
        a_nz,
        b_nz,
        cube_out,
    )
    torch.npu.synchronize()

    a_cpu = a_logical.cpu()
    b_cpu = b_logical.cpu()
    cube_expected = torch.empty(cube_shape, dtype=torch.int8)
    for batch in range(cube_shape[0]):
        for head in range(cube_shape[1]):
            raw = a_cpu[batch, head].float() @ b_cpu[batch, head].float()
            cube_expected[batch, head] = torch.clamp(torch.round(raw * 0.125), -128, 127).to(torch.int8)
    _assert_nz_close("t04 high-dimensional M/N/K quantized matmul", cube_out, cube_expected, atol=1)


@pl.jit(auto_mutex=True)
def nz_high_dimensional_vec_windows_kernel(
    inp: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
):
    vec_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64],
            valid_shape=[-1, -1],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
            compact=0,
        ),
        addrs=0x0000,
        mutex_ids=[0],
    )

    with pl.section_vector():
        vec = vec_group.current()
        pl.set_validshape(vec, [64, 32])
        pl.load(vec, inp, [0, 1, 16, 16])
        pl.store(out, vec, [0, 1, 16, 16])

        pl.set_validshape(vec, [32, 64])
        pl.load_tile(vec, inp, [1, 2, 1, 0])
        pl.store_tile(out, vec, [1, 2, 1, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t05_high_dimensional_dynamic_nz_vec_windows(device: str) -> None:
    shape = (2, 3, 96, 96)
    logical = torch.empty(shape, dtype=torch.float16, device=device)
    for batch in range(shape[0]):
        for head in range(shape[1]):
            logical[batch, head] = _matrix(96, 96, torch.float16, device, bias=batch + head / 4)
    out = _nz_filled(shape, torch.float16, SENTINEL, device)

    inp = _to_nz(logical)
    nz_high_dimensional_vec_windows_kernel(inp, out)
    torch.npu.synchronize()

    logical_cpu = logical.cpu()
    expected = torch.full(shape, SENTINEL, dtype=torch.float16)
    expected[0, 1, 16:80, 16:48] = logical_cpu[0, 1, 16:80, 16:48]
    expected[1, 2, 64:96, 0:64] = logical_cpu[1, 2, 64:96, 0:64]
    _assert_nz_close("t05 high-dimensional vector windows", out, expected)


@pl.jit(auto_mutex=True)
def nz_high_dimensional_l0c_via_vec_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16, pl.NZ],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32, pl.NZ],
):
    a_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[0],
    )
    b_l1_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x1000,
        mutex_ids=[1],
    )
    a_l0a_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[2],
    )
    b_l0b_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000,
        mutex_ids=[3],
    )
    acc_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[32, 32],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[32, 32], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
        addrs=0x0000,
        mutex_ids=[5],
    )

    with pl.section_cube():
        a_l1 = a_l1_group.current()
        b_l1 = b_l1_group.current()
        a_l0a = a_l0a_group.current()
        b_l0b = b_l0b_group.current()
        acc = acc_group.current()
        pl.load(a_l1, a, [1, 2, 32, 0])
        pl.load(b_l1, b, [1, 2, 0, 32])
        pl.move(a_l0a, a_l1)
        pl.move(b_l0b, b_l1)
        pl.matmul(acc, a_l0a, b_l0b)
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=1)
        pl.move(vec_group[0], acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
        if pl.get_subblock_idx() == 0:
            pl.store(out, vec_group[0], [1, 2, 32, 32])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_t06_high_dimensional_dynamic_nz_l0c_via_vec(device: str) -> None:
    shape = (2, 3, 64, 64)
    a_logical = torch.empty(shape, dtype=torch.float16)
    for batch in range(shape[0]):
        for head in range(shape[1]):
            a_logical[batch, head] = _matrix(64, 64, torch.float16, "cpu", bias=batch + head / 4)
    b_logical = torch.eye(64, dtype=torch.float16).expand(2, 3, 64, 64).clone()
    out = _nz_filled(shape, torch.float32, SENTINEL, device)

    a_nz = _to_nz(a_logical, device)
    b_nz = _to_nz(b_logical, device)
    nz_high_dimensional_l0c_via_vec_kernel(a_nz, b_nz, out)
    torch.npu.synchronize()

    expected = torch.full(shape, SENTINEL, dtype=torch.float32)
    expected[1, 2, 32:64, 32:64] = (
        a_logical[1, 2, 32:64].float() @ b_logical[1, 2, :, 32:64].float()
    )
    _assert_nz_close("t06 high-dimensional L0C via UB", out, expected, rtol=1e-2, atol=1e-2)
