# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""L1 slice records preserve the runtime tile, coordinates and layout contract."""

from pypto_pro import ir
import pypto_pro.language as pl
import pytest


def _statements(stmt):
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _statements(child)
    elif isinstance(stmt, ir.SectionStmt):
        yield from _statements(stmt.body)
    else:
        yield stmt


def _instrument(kernel, section=ir.SectionKind.Cube):
    program = kernel.to_kernel_def().parse_target_program(section)[0]
    result = ir.Pass.sanitizer()(program)
    entries = [f for f in result.functions.values() if f.func_type == ir.FunctionType.Opaque]
    assert len(entries) == 1
    return list(_statements(entries[0].body))


def _log_fields(stmt):
    assert isinstance(stmt, ir.AssignStmt) and not stmt.targets
    log = stmt.value
    assert isinstance(log, ir.Call) and log.name == "block.sanitizer_log"
    assert log.args[0].value & 0xFFFF == 2
    names = [name for name, _ in ir.sanitizer_record_layouts()[2][1]]
    assert len(log.args) == len(names) + 3
    return dict(zip(names[:-1], log.args[1:-3], strict=True))


def _slice_records(statements):
    result = []
    for position, stmt in enumerate(statements):
        if not (isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
                and stmt.value.name == "block.subview"):
            continue
        assert position > 0
        fields = _log_fields(statements[position - 1])
        result.append((stmt, fields))
    return result


def _assert_runtime_parent(stmt, fields):
    parent, coordinates, window = stmt.value.args
    for axis, name in enumerate(("dim_row", "dim_col")):
        getter = fields[name]
        assert getter.name == "block.tile_valid_shape"
        assert getter.kwargs["axis"] == axis
        assert ir.structural_equal(getter.args[0], parent)
    for axis, name in enumerate(("off_row", "off_col")):
        assert ir.structural_equal(fields[name], coordinates.elements[axis])
    for axis, name in enumerate(("acc_row", "acc_col")):
        assert ir.structural_equal(fields[name], window.elements[axis])


def _assert_address_inputs(stmt, fields, bits):
    parent, _, _ = stmt.value.args
    address = fields["base_addr"]
    assert address.name == "block.tile_addr"
    assert address.type.dtype == ir.DataType.INT64
    assert ir.structural_equal(address.args[0], parent)
    assert ir.structural_equal(fields["shape_row"], parent.type.shape[0])
    assert ir.structural_equal(fields["shape_col"], parent.type.shape[1])
    assert fields["dtype_bits"].value == bits


@pytest.mark.parametrize("layout", [pl.ND, pl.DN])
@pytest.mark.parametrize("dtype,bits", [(pl.DT_FP16, 16), (pl.DT_INT8, 8), (pl.DT_FP4E2M1, 4)])
def test_plain_l1_slice_records_address_inputs(layout, dtype, bits):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(row: pl.DT_INT64, col: pl.DT_INT64, rows: pl.DT_INT64, cols: pl.DT_INT64):
        tile_type = pl.TileType(shape=[64, 64], dtype=dtype, target_memory=pl.MemorySpace.Mat,
                                layout=layout, valid_shape=[-1, -1])
        tile = pl.make_tile(tile_type, addr=32)
        with pl.section_cube():
            pl.set_validshape(tile, [rows, cols])
            _view = tile[row:row + 16, col:col + 16]

    records = _slice_records(_instrument(kernel))
    assert len(records) == 1
    stmt, fields = records[0]
    _assert_runtime_parent(stmt, fields)
    _assert_address_inputs(stmt, fields, bits)
    assert fields["layout_kind"].value == int(layout)


@pytest.mark.parametrize("layout,dtype,fractal,inner", [
    (pl.NZ, pl.DT_FP16, 512, (16, 16)),
    (pl.ZN, pl.DT_FP16, 512, (16, 16)),
    (pl.NZ, pl.DT_INT8, 512, (16, 32)),
    (pl.ZN, pl.DT_INT8, 512, (32, 16)),
    (pl.NZ, pl.DT_FP32, 1024, (16, 16)),
    (pl.ZN, pl.DT_FP32, 1024, (16, 16)),
])
def test_fractal_l1_slice_records_logical_inner_axes(layout, dtype, fractal, inner):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(row: pl.DT_INT64, col: pl.DT_INT64):
        tile_type = pl.TileType(shape=[64, 64], dtype=dtype, target_memory=pl.MemorySpace.Mat,
                                layout=layout, fractal=fractal)
        tile = pl.make_tile(tile_type, addr=0)
        with pl.section_cube():
            _view = tile[row:row + 16, col:col + 16]

    records = _slice_records(_instrument(kernel))
    assert len(records) == 1
    stmt, fields = records[0]
    _assert_runtime_parent(stmt, fields)
    assert fields["layout_kind"].value == int(layout)
    assert (fields["inner_rows"].value, fields["inner_cols"].value) == inner
    assert all(fields[name].value == -1 for name in ("base_addr", "shape_row", "shape_col", "dtype_bits"))


@pytest.mark.parametrize("layout", [pl.ND, pl.DN])
def test_nested_group_slice_uses_selected_parent_descriptor(layout):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(index: pl.DT_INT64, row: pl.DT_INT64, col: pl.DT_INT64):
        tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Mat, layout=layout)
        group = pl.make_tile_group(type=tile_type, addrs=[0, 8192], depth=2)
        with pl.section_cube():
            selected = group[index]
            parent = selected[row:row + 32, col:col + 32]
            _child = parent[row:row + 16, col:col + 16]

    records = _slice_records(_instrument(kernel))
    assert len(records) == 2
    for stmt, fields in records:
        _assert_runtime_parent(stmt, fields)
        _assert_address_inputs(stmt, fields, 16)
    parent, _ = records[0]
    child, _ = records[1]
    assert ir.structural_equal(child.value.args[0], parent.targets[0])
    assert [dim.value for dim in child.value.args[0].type.shape] == [64, 64]


@pytest.mark.parametrize("memory,layout,section", [
    (pl.MemorySpace.Vec, pl.ND, ir.SectionKind.Vector),
    (pl.MemorySpace.Vec, pl.NZ, ir.SectionKind.Vector),
    (pl.MemorySpace.Acc, pl.NZ, ir.SectionKind.Cube),
])
def test_non_l1_slice_has_no_slice_record(memory, layout, section):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(row: pl.DT_INT64, col: pl.DT_INT64):
        tile = pl.make_tile(pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                                       target_memory=memory, layout=layout), addr=0)
        if memory == pl.MemorySpace.Vec:
            with pl.section_vector():
                _view = tile[row:row + 16, col:col + 16]
        else:
            with pl.section_cube():
                _view = tile[row:row + 16, col:col + 16]

    statements = _instrument(kernel, section)
    assert any(isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
               and stmt.value.name == "block.subview" for stmt in statements)
    logs = [stmt.value for stmt in statements
            if isinstance(stmt, ir.AssignStmt) and not stmt.targets
            and isinstance(stmt.value, ir.Call)
            and stmt.value.name == "block.sanitizer_log"]
    assert all(log.args[0].value & 0xFFFF != 2 for log in logs)


def test_existing_validshape_and_transfer_records_omit_slice_metadata():
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(rows: pl.DT_INT64):
        source = pl.make_tile(pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ), addr=0)
        target = pl.make_tile(pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                                         target_memory=pl.MemorySpace.Left), addr=0)
        with pl.section_cube():
            pl.set_validshape(source, [rows, 64])
            pl.move(target, source)

    names = [name for name, _ in ir.sanitizer_record_layouts()[2][1]]
    logs = [stmt.value for stmt in _instrument(kernel)
            if isinstance(stmt, ir.AssignStmt) and not stmt.targets
            and isinstance(stmt.value, ir.Call)
            and stmt.value.name == "block.sanitizer_log" and stmt.value.args[0].value & 0xFFFF == 2]
    assert len(logs) == 3  # set_validshape plus source/destination transfer windows
    for log in logs:
        assert len(log.args) == len(names) + 3
        fields = dict(zip(names[:-1], log.args[1:-3], strict=True))
        assert fields["layout_kind"].value == -1
        assert all(fields[name].value == -1 for name in ("base_addr", "shape_row", "shape_col", "dtype_bits"))
