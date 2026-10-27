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

"""ST: pl.dynamic_input_ptr on Ascend A5.

A PyPTO kernel consumes a runtime-variadic (DYNAMIC) input by decoding the
list-tensor descriptor the framework passes.  The ST synthesizes that descriptor
(dim==0 form, addressing N independently allocated tensors) and checks that a
@pl.jit kernel reads *each* tensor via pl.dynamic_input_ptr and sums them.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

P = 256


def _require_a5():
    try:
        torch.npu.set_device(ST_DEVICE)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def _build_list_tensor_desc(addrs):
    """list-tensor descriptor (dim==0 form): [dataPtrOffset, (count<<32)|dim, 0xffffffff, addr_0..addr_{N-1}]."""
    n = len(addrs)
    w = [0] * (3 + n)
    w[0] = 24  # byte offset from head to the address array -> dataPtr_ = base + 3
    w[1] = (n << 32) & 0xFFFFFFFFFFFFFFFF  # low32 dim=0, high32 count
    w[2] = 0xFFFFFFFF
    for i, a in enumerate(addrs):
        w[3 + i] = a
    return torch.tensor(w, dtype=torch.int64)


def _build_list_tensor_desc_dim1(addrs, shape0):
    """list-tensor descriptor with dim==1 (each entry carries one shape value).

    Layout: [dataPtrOffset, <count> shape structs of (1+dim)=2 words each, addr_0..].
    GetDataPtr(i) still reads a contiguous address array; the shape block sits between
    the header and the address array, so this exercises the dim != 0 form.
    """
    n = len(addrs)
    struct_sz = 2  # 1 (meta) + dim (=1)
    data_ptr_offset = 8 + n * struct_sz * 8
    w = [0] * (1 + n * struct_sz + n)
    w[0] = data_ptr_offset
    w[1] = (1 & 0xFFFFFFFF) | (n << 32)  # struct 0 meta: low32 dim=1, high32 count
    w[2] = shape0  # struct 0 shape[0]
    for i in range(1, n):
        w[1 + i * struct_sz] = 0
        w[1 + i * struct_sz + 1] = shape0
    addr_off = data_ptr_offset // 8
    for i, a in enumerate(addrs):
        w[addr_off + i] = a
    return torch.tensor(w, dtype=torch.int64)


@pl.jit(auto_mutex=True)
def dynamic_input_ptr_sum(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_FP32],
    count: pl.DT_INT32,
):
    out_t = pl.make_tensor(out, [1, P], [P, 1], dtype=pl.DT_FP32)
    tt = pl.TileType(
        shape=[1, P], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    src_db = pl.make_tile_group(type=tt, addrs=[0x0000, 0x1000], mutex_ids=[0, 1])
    acc_db = pl.make_tile_group(type=tt, addrs=[0x2000], mutex_ids=[2])
    with pl.section_vector():
        acc = acc_db.next()
        pl.set_validshape(acc, [1, P])
        # tensor 0 located through the descriptor
        p0 = pl.dynamic_input_ptr(desc, 0)
        pl.load(acc, pl.make_tensor(p0, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
        for i in pl.range(1, count):
            t = src_db.next()
            pl.set_validshape(t, [1, P])
            pi = pl.dynamic_input_ptr(desc, i)
            pl.load(t, pl.make_tensor(pi, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
            pl.add(acc, acc, t)
        pl.store(out_t, acc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_input_ptr():
    _require_a5()

    torch.manual_seed(0)
    # N tensors allocated independently (not contiguous), addressed via the descriptor.
    for count in (1, 3, 7):
        tensors = [torch.randn(P, device=ST_DEVICE, dtype=torch.float32) for _ in range(count)]
        desc = _build_list_tensor_desc([t.data_ptr() for t in tensors]).to(ST_DEVICE)
        out = torch.full((1, P), -1.0, device=ST_DEVICE, dtype=torch.float32)
        dynamic_input_ptr_sum[None, 1](desc, out, count)
        torch.npu.synchronize()
        ref = torch.stack(tensors).sum(0, keepdim=True)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
        logging.info("  count=%d OK", count)

    variants = len(dynamic_input_ptr_sum._compiled_by_signature)
    assert variants == 1, f"expected 1 compiled variant, got {variants}"
    logging.info("test_dynamic_input_ptr passed (variants=%d)!", variants)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_input_ptr_shaped():
    """dim != 0 descriptor form (each entry carries a shape value)."""
    _require_a5()

    torch.manual_seed(0)
    for count in (1, 3, 7):
        tensors = [torch.randn(P, device=ST_DEVICE, dtype=torch.float32) for _ in range(count)]
        desc = _build_list_tensor_desc_dim1([t.data_ptr() for t in tensors], P).to(ST_DEVICE)
        out = torch.full((1, P), -1.0, device=ST_DEVICE, dtype=torch.float32)
        dynamic_input_ptr_sum[None, 1](desc, out, count)
        torch.npu.synchronize()
        ref = torch.stack(tensors).sum(0, keepdim=True)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
        logging.info("  dim=1 count=%d OK", count)

    logging.info("test_dynamic_input_ptr_shaped passed!")


# ---------------------------------------------------------------------------
# dtype matrix: one representative per element width (64/32/16/8-bit).
# The descriptor itself is dtype-agnostic (addresses + shape words), so what is
# really being checked is that the pointer's element type -- and with it the whole
# load/add/store chain -- is right for each width.  4-bit (packed) elements are
# verified by the CCE codegen UT instead: torch has no 4-bit dtype to feed an ST.
# ---------------------------------------------------------------------------
DTYPE_MATRIX = (
    (pl.DT_INT64, torch.int64),
    (pl.DT_INT32, torch.int32),
    (pl.DT_FP32, torch.float32),
    (pl.DT_FP16, torch.float16),
    (pl.DT_BF16, torch.bfloat16),
    (pl.DT_INT8, torch.int8),
    (pl.DT_UINT8, torch.uint8),
)


def _build_ptr_sum_kernel(dt):
    """Sum `count` descriptor-listed tensors whose element dtype is `dt`."""

    @pl.jit(auto_mutex=True)
    def dynamic_input_ptr_sum_dt(
        desc: pl.Ptr[dt],
        out: pl.Ptr[dt],
        count: pl.DT_INT32,
    ):
        out_t = pl.make_tensor(out, [1, P], [P, 1], dtype=dt)
        tt = pl.TileType(shape=[1, P], dtype=dt, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        src_db = pl.make_tile_group(type=tt, addrs=[0x0000, 0x1000], mutex_ids=[0, 1])
        acc_db = pl.make_tile_group(type=tt, addrs=[0x2000], mutex_ids=[2])
        with pl.section_vector():
            acc = acc_db.next()
            pl.set_validshape(acc, [1, P])
            pl.load(acc, pl.make_tensor(pl.dynamic_input_ptr(desc, 0), [1, P], [P, 1], dtype=dt), [0, 0])
            for i in pl.range(1, count):
                t = src_db.next()
                pl.set_validshape(t, [1, P])
                pl.load(t, pl.make_tensor(pl.dynamic_input_ptr(desc, i), [1, P], [P, 1], dtype=dt), [0, 0])
                pl.add(acc, acc, t)
            pl.store(out_t, acc, [0, 0])

    return dynamic_input_ptr_sum_dt


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_input_ptr_dtype_matrix():
    """Every element width (64/32/16/8-bit) reads through the descriptor correctly."""
    _require_a5()

    for dt, torch_dt in DTYPE_MATRIX:
        kernel = _build_ptr_sum_kernel(dt)
        for count in (1, 3, 7):
            tensors = [torch.ones(P, device=ST_DEVICE, dtype=torch_dt) for _ in range(count)]
            desc = _build_list_tensor_desc([t.data_ptr() for t in tensors]).to(ST_DEVICE)
            out = torch.zeros((1, P), device=ST_DEVICE, dtype=torch_dt)
            kernel[None, 1](desc, out, count)
            torch.npu.synchronize()
            ref = torch.full((1, P), count, device=ST_DEVICE, dtype=torch_dt)
            assert torch.equal(out.cpu(), ref.cpu()), f"{dt} count={count} mismatch"
            logging.info("  %s count=%d OK", dt, count)

    logging.info("test_dynamic_input_ptr_dtype_matrix passed!")


@pl.jit(auto_mutex=True)
def dynamic_input_ptr_uint8_view(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_UINT8],
):
    """Re-view tensor 0 as raw bytes -- pl.make_tensor(dtype=) reinterprets the descriptor's tensor."""
    nb = P * 4
    out_t = pl.make_tensor(out, [1, nb], [nb, 1], dtype=pl.DT_UINT8)
    tt = pl.TileType(shape=[1, nb], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    db = pl.make_tile_group(type=tt, addrs=[0x0000], mutex_ids=[0])
    with pl.section_vector():
        acc = db.next()
        pl.set_validshape(acc, [1, nb])
        p0 = pl.dynamic_input_ptr(desc, 0)
        pl.load(acc, pl.make_tensor(p0, [1, nb], [nb, 1], dtype=pl.DT_UINT8), [0, 0])
        pl.store(out_t, acc, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_dynamic_input_ptr_uint8_view():
    """pl.make_tensor(dtype=) re-views the descriptor's tensor (here: raw uint8 bytes)."""
    _require_a5()

    torch.manual_seed(0)
    nb = P * 4
    tensors = [torch.randn(P, device=ST_DEVICE, dtype=torch.float32) for _ in range(2)]
    desc = _build_list_tensor_desc([t.data_ptr() for t in tensors]).to(ST_DEVICE)
    out = torch.zeros((1, nb), device=ST_DEVICE, dtype=torch.uint8)
    dynamic_input_ptr_uint8_view[None, 1](desc, out)
    torch.npu.synchronize()
    ref = tensors[0].view(torch.uint8).unsqueeze(0).cpu()
    assert torch.equal(out.cpu(), ref), "raw-byte re-view mismatch"
    logging.info("test_dynamic_input_ptr_uint8_view passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_dynamic_input_ptr()
    test_dynamic_input_ptr_shaped()
    test_dynamic_input_ptr_dtype_matrix()
    test_dynamic_input_ptr_uint8_view()
    logging.info("\nAll tests passed!")
