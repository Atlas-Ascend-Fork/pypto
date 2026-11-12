# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Accumulator identity records retain runtime addresses and valid_shape."""

from pypto_pro import ir
from pypto_pro.ir.op.block_ops import make_mutex_tile_expr
import pypto_pro.language as pl
import pytest


def _calls(stmt):
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _calls(child)
    elif isinstance(stmt, ir.SectionStmt):
        yield from _calls(stmt.body)
    elif isinstance(stmt, ir.AssignStmt) and not stmt.targets and isinstance(stmt.value, ir.Call):
        yield stmt.value


def _phase_fields(log):
    names = [name for name, _ in ir.sanitizer_record_layouts()[7][1]]
    assert len(log.args) == len(names) + 3
    assert log.args[0].value & 0xFFFF == 7
    return dict(zip(names[:-1], log.args[1:-3], strict=True))


def _assert_tile_fields(fields, prefix, tile):
    address = fields[prefix + "_addr"]
    assert address.name == "block.tile_addr"
    assert ir.structural_equal(address.args[0], tile)
    for axis, suffix in enumerate(("_valid_rows", "_valid_cols")):
        getter = fields[prefix + suffix]
        assert getter.name == "block.tile_valid_shape"
        assert getter.kwargs["axis"] == axis
        assert ir.structural_equal(getter.args[0], tile)


@pytest.mark.parametrize("op_name,kind", [("matmul_acc", 2), ("matmul_mx_acc", 5)])
def test_acc_phase_records_addresses_and_runtime_valid_shapes(op_name, kind):
    span = ir.Span("acc_identity.py", 20, 0)
    specs = [
        ("dst", [64, 128], ir.DataType.FP32, ir.MemorySpace.Acc),
        ("acc", [64, 128], ir.DataType.FP32, ir.MemorySpace.Acc),
        ("lhs", [64, 64], ir.DataType.FP16, ir.MemorySpace.Left),
        ("rhs", [64, 128], ir.DataType.FP16, ir.MemorySpace.Right),
    ]
    if kind == 5:
        specs[2] = ("lhs", [64, 64], ir.DataType.FP8E4M3FN, ir.MemorySpace.Left)
        specs[3] = ("rhs", [64, 128], ir.DataType.FP8E4M3FN, ir.MemorySpace.Right)
        specs.extend([
            ("scale_a", [64, 2], ir.DataType.FP8E8M0, ir.MemorySpace.ScaleLeft),
            ("scale_b", [2, 128], ir.DataType.FP8E8M0, ir.MemorySpace.ScaleRight),
        ])
    definitions = [make_mutex_tile_expr(shape, dtype, memory, addr=0, span=span)
                   for _, shape, dtype, memory in specs]
    tiles = [ir.Var(spec[0], definition.type, span)
             for spec, definition in zip(specs, definitions, strict=True)]
    operation = ir.create_op_call("block." + op_name, tiles, {"phase": 1}, span)
    body = ir.SeqStmts(
        [ir.AssignStmt([tile], definition, span)
         for tile, definition in zip(tiles, definitions, strict=True)]
        + [ir.AssignStmt([], operation, span)], span,
    )
    program = ir.Program([ir.Function("kernel", [], [], body, span)], "identity", span)
    instrumented = ir.Pass.sanitizer()(program)
    calls = list(_calls(instrumented.get_function("kernel").body))
    log = calls[-2]
    assert log.name == "block.sanitizer_log"
    assert calls[-1].name == "block." + op_name
    assert log.args[-1].value == 20
    fields = _phase_fields(log)
    assert fields["op_kind"].value == kind
    _assert_tile_fields(fields, "acc", tiles[0])
    _assert_tile_fields(fields, "acc_src", tiles[1])


@pytest.mark.parametrize("use_group", [False, True])
def test_dsl_acc_identity_tracks_validshape_updates_and_selected_tile(use_group):
    @pl.jit(arch="3510", auto_mutex=False)
    def kernel(rows: pl.DT_INT64, cols: pl.DT_INT64, index: pl.DT_INT64):
        acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Acc)
        dst = pl.make_tile(acc_type, addr=0)
        if use_group:
            group = pl.make_tile_group(type=acc_type, addrs=[0, 16384], depth=2)
            acc = group[index]
        else:
            acc = pl.make_tile(acc_type, addr=0)
        lhs = pl.make_tile(pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                                       target_memory=pl.MemorySpace.Left), addr=0)
        rhs = pl.make_tile(pl.TileType(shape=[64, 64], dtype=pl.DT_FP16,
                                       target_memory=pl.MemorySpace.Right), addr=0)
        with pl.section_cube():
            pl.matmul_acc(dst, acc, lhs, rhs)
            pl.set_validshape(acc, [rows, cols])
            pl.matmul_acc(dst, acc, lhs, rhs)

    program = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)[0]
    instrumented = ir.Pass.sanitizer()(program)
    function = next(f for f in instrumented.functions.values() if f.func_type == ir.FunctionType.Opaque)
    calls = list(_calls(function.body))
    acc_positions = [i for i, call in enumerate(calls) if call.name == "block.matmul_acc"]
    update_positions = [i for i, call in enumerate(calls) if call.name == "block.set_validshape"]
    assert len(acc_positions) == 2
    assert len(update_positions) == 1
    assert acc_positions[0] < update_positions[0] < acc_positions[1]
    for position in acc_positions:
        log, operation = calls[position - 1], calls[position]
        assert log.name == "block.sanitizer_log"
        fields = _phase_fields(log)
        _assert_tile_fields(fields, "acc", operation.args[0])
        _assert_tile_fields(fields, "acc_src", operation.args[1])
