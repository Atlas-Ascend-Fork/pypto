# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SIMT sanitizer device cases. GM violations stay inside a larger backing allocation."""

import os

import pypto_pro.language as pl
from pypto_pro.runtime.sanitizer_replay import SanitizerReplayError
import pytest
import torch

pytestmark = pytest.mark.soc("950")


@pytest.fixture
def device():
    name = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    try:
        torch.npu.set_device(name)
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    if "Ascend950" not in torch.npu.get_device_name():
        pytest.skip("SIMT requires Ascend950")
    return name


@pl.vector_function(mode="simt")
def _read_value(src, index: pl.DT_INT32) -> pl.DT_INT32:
    return src[0, index]


@pl.vector_function(mode="simt")
def _copy_value(dst, src, tid: pl.DT_INT32, offset: pl.DT_INT32):
    dst[0, tid] = _read_value(src, tid + offset)


@pl.vector_function(mode="simt", max_threads=32)
def _access(dst, src, runs, offset: pl.DT_INT32, mode: pl.DT_INT32):
    tid = pl.simt.cast(pl.simt.linear_thread_idx(), pl.DT_INT32)
    if tid == 0:
        pl.simt.atomic_add(runs, [0, 0], 1)
    if mode == 0:
        _copy_value(dst, src, tid, offset)
    elif mode == 1:
        dst[0, tid + offset] = tid + 1
    elif mode == 2:
        pl.simt.atomic_add(dst, [0, tid + offset], 1)
    elif mode == 3:
        dst[0, tid] = pl.simt.atomic_add(src, [0, tid + offset], 1)
    else:
        dst[0, tid + offset] = pl.simt.atomic_add(src, [0, tid], 1)


@pl.jit(sanitizer=True)
def _gm_kernel(
    src: pl.Tensor[[1, 64], pl.DT_INT32],
    dst: pl.Tensor[[1, 64], pl.DT_INT32],
    runs: pl.Tensor[[1, 1], pl.DT_INT32],
    offset: pl.DT_INT32,
    mode: pl.DT_INT32,
):
    with pl.section_vector():
        logical_src = pl.make_tensor(src, [1, 32])
        logical_dst = pl.make_tensor(dst, [1, 32])
        _access[32](logical_dst, logical_src, runs, offset, mode)


@pytest.mark.parametrize("mode", (0, 1, 2, 3, 4))
@pytest.mark.parametrize("offset", (0, 32))
def test_simt_gm_read_write_atomic_and_nested_helpers(device, mode, offset):
    src = torch.arange(64, dtype=torch.int32, device=device).reshape(1, 64)
    dst = torch.zeros_like(src)
    runs = torch.zeros((1, 1), dtype=torch.int32, device=device)
    if offset:
        with pytest.raises(SanitizerReplayError) as error:
            _gm_kernel[None, 1](src, dst, runs, offset, mode)
        findings = error.value.findings
        assert all(f.kind == "GM_OUT_OF_BOUNDS" for f in findings)
        # Every active lane violates the same source statement, with differing offsets.
        assert sum(f.hits for f in findings) == 32 * runs.cpu().item()
        assert all(f.location.startswith(__file__ + ":") for f in findings)
    else:
        _gm_kernel[None, 1](src, dst, runs, offset, mode)
        torch.npu.synchronize()
        if mode == 0:
            torch.testing.assert_close(dst[:, :32], src[:, :32])
        elif mode == 1:
            torch.testing.assert_close(dst[:, :32], src[:, :32] + 1)
        elif mode == 2:
            assert torch.all(dst[:, :32] >= 1).item()
        else:
            initial = torch.arange(32, dtype=torch.int32, device=device).reshape(1, 32)
            torch.testing.assert_close(src[:, :32], initial + runs.cpu().item())
            # Concurrent blocks may store their returned old values in either order.
            assert torch.all(dst[:, :32] >= initial).item()
            assert torch.all(dst[:, :32] < src[:, :32]).item()


@pl.vector_function(mode="simt", max_threads=32)
def _tile_write(tile, src, offset: pl.DT_INT32):
    tid = pl.simt.cast(pl.simt.linear_thread_idx(), pl.DT_INT32)
    tile[0, tid + offset] = src[0, tid] + 1


@pl.jit(sanitizer=True)
def _tile_kernel(src: pl.Tensor[[1, 32], pl.DT_INT32], offset: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile(tile_type, addr=0)
    with pl.section_vector():
        pl.load(tile, src, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        _tile_write[32](tile, src, offset)


@pytest.mark.parametrize("offset", (0, 32))
def test_simt_tile_bounds_with_scalar_logging(device, offset):
    src = torch.zeros((1, 32), dtype=torch.int32, device=device)
    if offset:
        # [32,64) is outside the logical Tile but inside the static user UB reservation.
        with pytest.raises(SanitizerReplayError) as error:
            _tile_kernel[None, 1](src, offset)
        assert all(f.kind == "TILE_OUT_OF_BOUNDS" for f in error.value.findings)
        assert sum(f.hits for f in error.value.findings) >= 32
    else:
        _tile_kernel[None, 1](src, offset)


@pl.vector_function(mode="simt", max_threads=256)
def _many_records(dst):
    tid = pl.simt.linear_thread_idx()
    for iteration in pl.range(128):
        dst[0, tid] = pl.simt.cast(iteration, pl.DT_INT32)


@pl.jit(sanitizer=True)
def _full_log_kernel(dst: pl.Tensor[[1, 256], pl.DT_INT32]):
    with pl.section_vector():
        _many_records[8, 4, 8](dst)


def test_simt_log_capacity_exhaustion_keeps_complete_records(device, monkeypatch):
    import importlib

    jit_module = importlib.import_module("pypto_pro.runtime.jit")
    monkeypatch.setattr(jit_module, "KSANITIZER_LOG_BUDGET_BYTES", 1024**2)
    dst = torch.zeros((1, 256), dtype=torch.int32, device=device)
    with pytest.warns(RuntimeWarning, match="records may have been dropped"):
        _full_log_kernel[None, 1](dst)
    torch.testing.assert_close(dst, torch.full_like(dst, 127))


# ---------------------------------------------------------------------------
# Per-axis bounds, scalar arithmetic, and single-evaluation device regressions.
# ---------------------------------------------------------------------------

@pl.vector_function(mode="simt")
def _read_coordinate(src, row: pl.DT_INT32, col: pl.DT_INT32) -> pl.DT_INT32:
    return src[row, col]


@pl.vector_function(mode="simt", max_threads=1)
def _coordinate_probe(data, out, runs, row: pl.DT_INT32, col: pl.DT_INT32, mode: pl.DT_INT32):
    pl.simt.atomic_add(runs, [0, 0], 1)
    if mode == 0:
        out[0, 0] = _read_coordinate(data, row, col)
    elif mode == 1:
        data[row, col] = 7
    else:
        out[0, 0] = pl.simt.atomic_add(data, [row, col], 1)


@pl.jit(sanitizer=True)
def _gm_coordinate_kernel(
    data: pl.Tensor[[4, 4], pl.DT_INT32],
    out: pl.Tensor[[1, 1], pl.DT_INT32],
    runs: pl.Tensor[[1, 1], pl.DT_INT32],
    row: pl.DT_INT32, col: pl.DT_INT32, mode: pl.DT_INT32,
):
    with pl.section_vector():
        _coordinate_probe[1](data, out, runs, row, col, mode)


@pytest.mark.parametrize("mode", (0, 1, 2))
@pytest.mark.parametrize("row,col", [(3, 3), (1, 5), (1, -1), (-1, 5)])
def test_st_simt_per_axis_bounds_catch_compensating_coordinates(device, mode, row, col):
    import torch

    device = device
    # Even the invalid coordinates address existing backing storage: no device fault is needed.
    assert 0 <= row * 4 + col < 16
    data = torch.arange(16, dtype=torch.int32, device=device).reshape(4, 4)
    out = torch.zeros((1, 1), dtype=torch.int32, device=device)
    runs = torch.zeros_like(out)
    bad_axes = [axis for axis, index in enumerate((row, col)) if not 0 <= index < 4]
    if bad_axes:
        with pytest.raises(SanitizerReplayError) as error:
            _gm_coordinate_kernel[None, 1](data, out, runs, row, col, mode)
        findings = error.value.findings
        assert all(f.kind == "GM_OUT_OF_BOUNDS" for f in findings)
        assert len(findings) == len(bad_axes)
        for axis in bad_axes:
            finding = next(f for f in findings if f"dim{axis}" in f.hint)
            assert finding.hits == runs.cpu().item()
    else:
        _gm_coordinate_kernel[None, 1](data, out, runs, row, col, mode)
        if mode == 0:
            assert out.cpu().item() == 15
        elif mode == 1:
            assert data.cpu()[3, 3].item() == 7
        else:
            assert data.cpu()[3, 3].item() == 15 + runs.cpu().item()


@pl.jit(sanitizer=True)
def _tile_valid_coordinate_kernel(
    src: pl.Tensor[[2, 8], pl.DT_INT32],
    out: pl.Tensor[[1, 1], pl.DT_INT32],
    runs: pl.Tensor[[1, 1], pl.DT_INT32],
    valid_rows: pl.DT_INT32, valid_cols: pl.DT_INT32,
    row: pl.DT_INT32, col: pl.DT_INT32, mode: pl.DT_INT32,
):
    tt = pl.TileType(shape=[2, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile(tt, addr=0)
    with pl.section_vector():
        pl.load(tile, src, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.set_validshape(tile, [valid_rows, valid_cols])
        _coordinate_probe[1](tile, out, runs, row, col, mode)


@pytest.mark.parametrize("mode", (0, 1, 2))
@pytest.mark.parametrize(
    "valid_rows,valid_cols,row,col,bad_axis",
    [(1, 4, 0, 3, None), (1, 4, 0, 4, 1), (1, 4, 0, 6, 1),
     (1, 4, 1, 0, 0), (0, 4, 0, 0, 0), (1, 0, 0, 0, 1)],
)
def test_st_simt_tile_scalar_access_uses_runtime_valid_shape(
    device, mode, valid_rows, valid_cols, row, col, bad_axis,
):
    import torch

    device = device
    src = torch.arange(16, dtype=torch.int32, device=device).reshape(2, 8)
    out = torch.zeros((1, 1), dtype=torch.int32, device=device)
    runs = torch.zeros_like(out)
    args = (src, out, runs, valid_rows, valid_cols, row, col, mode)
    if bad_axis is None:
        _tile_valid_coordinate_kernel[None, 1](*args)
        if mode == 0:
            assert out.cpu().item() == col
    else:
        with pytest.raises(SanitizerReplayError) as error:
            _tile_valid_coordinate_kernel[None, 1](*args)
        findings = error.value.findings
        assert len(findings) == 1
        assert findings[0].kind == "TILE_OUT_OF_BOUNDS"
        assert "runtime valid_shape" in findings[0].message
        assert f"dim{bad_axis}" in findings[0].hint
        assert findings[0].hits == runs.cpu().item()


# ST: unchanged scalar addressing with sanitizer on/off, mixed integer widths,
# rank-one outputs, and static three-dimensional input shapes.
def _offset_regression_kernel(index_dtype, sanitizer):

    @pl.vector_function(mode="simt", max_threads=1)
    def _offset_access(data, out, first: pl.DT_UINT64, second: pl.DT_UINT64, third: pl.DT_INT64):
        typed_first = pl.simt.cast(first, index_dtype)
        out[0] = data[typed_first, second, third]
        out[1] = pl.simt.atomic_add(data, [typed_first, second, third], 1)
        data[typed_first, second, third] = out[0] + 10

    @pl.jit(sanitizer=sanitizer)
    def kernel(
        data: pl.Tensor[[2, 3, 4], pl.DT_INT32],
        out: pl.Tensor[[2], pl.DT_INT32],
        first: pl.DT_UINT64, second: pl.DT_UINT64, third: pl.DT_INT64,
    ):
        with pl.section_vector():
            # Exactly one producer makes the atomic return and final store deterministic.
            if pl.get_subblock_idx() == 0:
                _offset_access[1](data, out, first, second, third)

    return kernel


@pytest.mark.parametrize("sanitizer", [False, True])
@pytest.mark.parametrize("index_dtype", [pl.DT_INT32, pl.DT_UINT32, pl.DT_INT64, pl.DT_UINT64])
def test_st_simt_scalar_offsets_preserve_addressing(device, sanitizer, index_dtype):
    import torch

    kernel = _offset_regression_kernel(index_dtype, sanitizer)
    shapes = [(2, 3, 4)]
    for shape in shapes:
        count = shape[0] * shape[1] * shape[2]
        original = torch.arange(count, dtype=torch.int32).reshape(shape)
        data = original.to(device)
        out = torch.full((2,), -1, dtype=torch.int32, device=device)
        coordinates = tuple(size - 1 for size in shape)
        kernel[None, 1](data, out, *coordinates)
        expected = original.clone()
        expected[coordinates] += 10
        torch.testing.assert_close(data.cpu(), expected)
        assert out.cpu().tolist() == [count - 1, count - 1]


def _wrapping_coordinate_kernel(dtype, shape, sanitizer):
    @pl.vector_function(mode="simt", max_threads=1)
    def _wrapping_access(data, out, row: pl.DT_UINT64, col: pl.DT_UINT64):
        typed_row = pl.simt.cast(row, dtype)
        typed_col = pl.simt.cast(col, dtype)
        out[0] = data[typed_row, typed_col]
        out[1] = pl.simt.atomic_add(data, [typed_row, typed_col], 1)
        data[typed_row, typed_col] = 17

    @pl.jit(sanitizer=sanitizer)
    def kernel(
        data: pl.Tensor[shape, pl.DT_INT32], out: pl.Tensor[[2], pl.DT_INT32],
        row: pl.DT_UINT64, col: pl.DT_UINT64,
    ):
        with pl.section_vector():
            if pl.get_subblock_idx() == 0:
                _wrapping_access[1](data, out, row, col)

    return kernel


@pytest.mark.parametrize("sanitizer", [False, True])
@pytest.mark.parametrize("dtype,shape,row,col,bad_axes", [
    (pl.DT_UINT32, [4, 1], (1 << 32) - 1, 1, [0, 1]),
    (pl.DT_UINT64, [2, 2], 1 << 63, 0, [0]),
])
def test_st_simt_invalid_coordinates_preserve_original_unsigned_arithmetic(
    device, sanitizer, dtype, shape, row, col, bad_axes,
):
    import torch

    # Both original unsigned expressions wrap to offset zero. Widening the first
    # one or converting the second to signed arithmetic would change the access.
    width = 32 if dtype == pl.DT_UINT32 else 64
    assert (row * shape[1] + col) % (1 << width) == 0
    original = torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape)
    data = original.to(device)
    out = torch.full((2,), -1, dtype=torch.int32, device=device)
    kernel = _wrapping_coordinate_kernel(dtype, shape, sanitizer)
    if sanitizer:
        with pytest.raises(SanitizerReplayError) as error:
            kernel[None, 1](data, out, row, col)
        findings = error.value.findings
        assert len(findings) == 3 * len(bad_axes)
        from pathlib import Path

        access_lines = {
            number for number, line in enumerate(Path(__file__).read_text().splitlines(), 1)
            if line.strip().startswith(("out[0] = data[typed_row,", "out[1] = pl.simt.atomic_add(data, [typed_row,",
                                        "data[typed_row, typed_col] ="))
        }
        assert {int(f.location.rsplit(":", 1)[1]) for f in findings} == access_lines
        assert all(f.kind == "GM_OUT_OF_BOUNDS" and f.hits == 1 for f in findings)
        for axis in bad_axes:
            assert sum(f"dim{axis}" in f.hint for f in findings) == 3
    else:
        kernel[None, 1](data, out, row, col)
    expected = original.clone()
    expected[0, 0] = 17
    torch.testing.assert_close(data.cpu(), expected)
    assert out.cpu().tolist() == [0, 0]


# ST: a side-effecting index must be evaluated once for logs and the access.
def _once_only_index_kernel(sanitizer, mode):
    @pl.vector_function(mode="simt", max_threads=1)
    def _once_only_access(data, out, counter, col: pl.DT_INT32):
        if mode == 0:
            out[0] = data[pl.simt.atomic_add(counter, [0], 1), col]
        elif mode == 1:
            data[pl.simt.atomic_add(counter, [0], 1), col] = 19
        else:
            out[0] = pl.simt.atomic_add(data, [pl.simt.atomic_add(counter, [0], 1), col], 2)

    @pl.jit(sanitizer=sanitizer)
    def kernel(
        data: pl.Tensor[[4, 4], pl.DT_INT32], out: pl.Tensor[[1], pl.DT_INT32],
        counter: pl.Tensor[[1], pl.DT_INT32], col: pl.DT_INT32,
    ):
        with pl.section_vector():
            if pl.get_subblock_idx() == 0:
                _once_only_access[1](data, out, counter, col)
    return kernel


@pytest.mark.parametrize("sanitizer", [False, True])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("col", [2, 5])
def test_st_simt_atomic_index_evaluated_once(device, sanitizer, mode, col):
    import torch

    original = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    data = original.to(device)
    out = torch.full((1,), -1, dtype=torch.int32, device=device)
    counter = torch.ones((1,), dtype=torch.int32, device=device)
    kernel = _once_only_index_kernel(sanitizer, mode)
    if sanitizer and col == 5:
        with pytest.raises(SanitizerReplayError) as error:
            kernel[None, 1](data, out, counter, col)
        assert len(error.value.findings) == 1
        finding = error.value.findings[0]
        assert finding.kind == "GM_OUT_OF_BOUNDS" and finding.hits == 1
        assert "dim1" in finding.hint
    else:
        kernel[None, 1](data, out, counter, col)
    assert counter.cpu().item() == 2
    offset = 4 + col
    expected = original.clone()
    if mode == 1:
        expected.view(-1)[offset] = 19
    elif mode == 2:
        expected.view(-1)[offset] += 2
    torch.testing.assert_close(data.cpu(), expected)
    assert out.cpu().item() == (-1 if mode == 1 else offset)


@pl.vector_function(mode="simt", max_threads=32)
def _shared_log_threads(data):
    tid = pl.simt.linear_thread_idx()
    for iteration in pl.range(64):
        _value = data[tid]


@pl.jit(sanitizer=True, auto_mutex=False)
def _shared_log_kernel(data: pl.Tensor[[64], pl.DT_INT32]):
    with pl.section_vector():
        _shared_log_threads[32](data)
        for iteration in pl.range(64):
            _value = data[32 + iteration % 32]


@pytest.mark.parametrize("blocks", (1, 2))
@pytest.mark.parametrize("record_words", (9 * 2112, 9 * 7, 9 * 7 + 8))
def test_scalar_and_simt_share_complete_records_per_vcore(device, monkeypatch, blocks, record_words):
    from collections import Counter
    import importlib

    from pypto_pro.runtime.sanitizer_replay import ScalarAxisRecord, decode_regions

    jit_module = importlib.import_module("pypto_pro.runtime.jit")
    replay = jit_module._replay_sanitizer
    observed = {}
    # A pure Vector launch runs one AIV per block, inside the 950 mixed layout.
    regions = 3 * blocks
    active_regions = range(blocks, 2 * blocks)
    monkeypatch.setattr(jit_module, "KSANITIZER_LOG_BUDGET_BYTES", regions * (record_words + 1) * 4)

    def inspect(log, capacity, block_dim, *args):
        torch.npu.synchronize()
        words = log.view(torch.int32).cpu()
        counters = words[::capacity + 1].tolist()
        assert len(counters) == regions
        assert capacity == record_words
        chunks = []
        for region, count in enumerate(counters):
            assert 0 <= count <= capacity
            start = region * (capacity + 1)
            chunks.append(words[start:start + 1 + count].numpy().tobytes())
        observed.update(counters=counters, records=decode_regions(chunks))
        return replay(log, capacity, block_dim, *args)

    monkeypatch.setattr(jit_module, "_replay_sanitizer", inspect)
    data = torch.arange(64, dtype=torch.int32, device=device)
    # Exact fit and short tails must leave only complete records; a full log is warned about.
    with pytest.warns(RuntimeWarning, match="capacity"):
        _shared_log_kernel[None, blocks](data)

    expected = Counter({(index, 1): 64 for index in range(32)})
    expected.update({(index, 0): 2 for index in range(32, 64)})
    per_region = {region: Counter() for region in active_regions}
    for record in observed["records"]:
        assert isinstance(record, ScalarAxisRecord)
        assert record.region in per_region
        assert (record.is_tile, record.axis, record.index_hi, record.bound_lo, record.bound_hi) == (0, 0, 0, 64, 0)
        assert record.line > 0
        per_region[record.region][record.index_lo, record.is_unsigned] += 1
    record_count = min(2112, record_words // 9)
    assert observed["counters"] == [
        9 * record_count if region in active_regions else 0 for region in range(regions)
    ]
    for actual in per_region.values():
        assert sum(actual.values()) == record_count
        assert not actual - expected  # Reject corrupted fields, duplicates and unknown writers.
        if record_count == 2112:
            assert actual == expected
