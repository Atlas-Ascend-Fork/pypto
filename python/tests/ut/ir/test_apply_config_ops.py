# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import Counter
from pathlib import Path

import pypto
from pypto import ir, pil

from .test_common import check_snapshot


def _walk(stmt):
    yield stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            yield from _walk(child)
    elif isinstance(stmt, ir.IfStmt):
        yield from _walk(stmt.then_body)
        if stmt.else_body is not None:
            yield from _walk(stmt.else_body)
    elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
        yield from _walk(stmt.body)


def _tensors():
    return [pypto.Tensor([32, 32], pypto.DT_FP32, name) for name in ("x", "y", "out")]


def test_config_replay():
    """apply_config_pass保留入口配置，function配置取末值"""
    def kernel(x, y, out):
        a = x + y
        pypto.set_vec_tile_shapes(16, 16)
        pypto.set_pass_options(cube_nbuffer_setting={-1: 2})
        b = a * y
        pypto.assemble(b, [0, 0], out)
        pypto.set_pass_options(cube_nbuffer_setting={-1: 4})

    with pypto.options("jit_scope", vec_tile_shapes=[8, 8]):
        entry = pypto.pypto_impl.CurrentScope()
        func = pil.compile(kernel, *_tensors())
        golden_dir = Path(__file__).with_suffix("")
        check_snapshot(func, golden_dir / "test_apply_config_ops_before.pypto")
        program = ir.IRBuilder().create_program([func], "main", ir.Span.unknown())
        result = ir.Pass.apply_config_ops()(program).functions[func.name]
        check_snapshot(result, golden_dir / "test_apply_config_ops_after.pypto")
        assert pypto.pypto_impl.CurrentScope() is not entry
        assert entry.GetAnyConfig("vec_tile_shapes") == [8, 8]
    rooted = ir.Pass.create_root_functions()(
        ir.IRBuilder().create_program([result], "main", ir.Span.unknown()))
    hidden = [f for name, f in rooted.functions.items() if "_hiddenfunc" in name]
    assert len(hidden) == 1
    # These backend fields are not represented in the IR text dump.
    tiles = {op.opcode: op.parse_attrs()["vec_tile_shapes"] for op in hidden[0].body.stmts}
    assert tiles["ADD"] == [8, 8]
    assert tiles["MUL"] == [16, 16]
    assert hidden[0].parse_attrs()["config"]["pass.cube_nbuffer_setting"] == {-1: 4}
    assert all(s.opcode != "OP_CONFIG" for s in hidden[0].body.stmts)


def test_branch_config():
    """后继 MUL 合入分支后使用分支配置"""
    def kernel(x, y, out):
        pypto.set_vec_tile_shapes(8, 8)
        if pypto.SymbolicScalar("cond") > 0:
            pypto.set_vec_tile_shapes(16, 16)
            pypto.set_pass_options(cube_nbuffer_setting={-1: 3}, sg_set_scope=11,
                                   experimental={"sg_set_atomic_scope": 21})
            value = x + y
        else:
            pypto.set_pass_options(cube_nbuffer_setting={-1: 4}, sg_set_scope=12,
                                   experimental={"sg_set_atomic_scope": 22})
            value = x - y
        pypto.assemble(value * y, [0, 0], out)

    rooted = _lower_config_function(pil.compile(kernel, *_tensors()), "branch_configs")
    hidden = [f for name, f in rooted.functions.items() if "_hiddenfunc" in name]
    assert len(hidden) == 2
    expected = {
        "ADD": ([16, 16], 11, 21, {-1: 3}),
        "SUB": ([8, 8], 12, 22, {-1: 4}),
    }
    seen = Counter()
    for func in hidden:
        ops = [op for op in func.body.stmts if op.opcode not in ("VIEW", "ASSEMBLE")]
        markers = [op.opcode for op in ops if op.opcode in expected]
        assert len(markers) == 1
        marker = markers[0]
        assert Counter(op.opcode for op in ops) == Counter({marker: 1, "MUL": 1})
        tile, scope, atomic_scope, nbuffer = expected[marker]
        assert func.parse_attrs()["config"]["pass.cube_nbuffer_setting"] == nbuffer
        for op in ops:
            metadata = op.parse_attrs()
            assert metadata["vec_tile_shapes"] == tile
            assert metadata["scope_id"] == scope
            assert metadata["atomic_scope_id"] // 10000 == atomic_scope
        seen[marker] += 1
    assert seen == Counter({"ADD": 1, "SUB": 1})


def test_config_only():
    """只有配置时不生成空function"""
    def kernel():
        pypto.set_pass_options(cube_nbuffer_setting={-1: 2})

    rooted = _lower_config_function(pil.compile(kernel), "config_only")
    assert all("_hiddenfunc" not in name for name in rooted.functions)
    assert all(not isinstance(s, ir.TensorOpStmt) or s.opcode != "OP_CONFIG"
               for s in _walk(rooted.functions["kernel"].body))


def test_loop_inheritance():
    """验证 global、JIT、嵌套及兄弟 loop 的配置继承与恢复"""
    def loop_scope(a, b, result):
        pypto.set_vec_tile_shapes(64, 64)
        a = a - 1
        for _ in pypto.loop(1, name="outer", idx_name="k"):
            pypto.set_vec_tile_shapes(32, 32)
            pypto.set_pass_options(vec_nbuffer_setting={-1: 4})
            result.move(a + b)
            for _ in pypto.loop(1, name="inner", idx_name="j"):
                pypto.set_vec_tile_shapes(16, 16)
                pypto.set_pass_options(vec_nbuffer_setting={-1: 2})
                result.move(result * b)
            result.move(result - b)
        for _ in pypto.loop(1, name="sibling", idx_name="k"):
            result.move(result + 1)

    tensors = [pypto.Tensor([64, 64], pypto.DT_INT32, name) for name in ("a", "b", "result")]
    # Global settings: JIT overrides vec, while cube is inherited through every scope.
    original = pypto.get_pass_options()
    try:
        pypto.set_pass_options(vec_nbuffer_setting={-1: 16}, cube_nbuffer_setting={-1: 7})
        with pypto.options("jit_scope", pass_options={"vec_nbuffer_setting": {-1: 8}}):
            rooted = _lower_config_function(pil.compile(loop_scope, *tensors), "nested_loop_scopes")
        assert pypto.get_pass_options()["vec_nbuffer_setting"] == {-1: 16}
    finally:
        pypto.set_pass_options(vec_nbuffer_setting=original["vec_nbuffer_setting"],
                               cube_nbuffer_setting=original["cube_nbuffer_setting"])

    hidden = [func for name, func in rooted.functions.items() if "_hiddenfunc" in name]
    # Distinct opcodes identify each scope independently of the configuration under test.
    expected = {
        "SUBS": ([64, 64], {-1: 8}),  # Before loops: inherit JIT config.
        "ADD": ([32, 32], {-1: 4}),   # Outer loop override.
        "MUL": ([16, 16], {-1: 2}),   # Nested loop override.
        "SUB": ([32, 32], {-1: 4}),   # After nested loop: restore outer config.
        "ADDS": ([64, 64], {-1: 8}),  # Sibling loop: inherit JIT config.
    }
    seen = Counter()
    for func in hidden:
        info = func.parse_attrs()
        assert info["config"]["pass.cube_nbuffer_setting"] == {-1: 7}
        for op in func.body.stmts:
            if op.opcode not in expected:
                assert op.opcode in ("VIEW", "ASSEMBLE"), op.opcode
                continue
            tile, nbuffer = expected[op.opcode]
            assert op.parse_attrs()["vec_tile_shapes"] == tile
            assert info["config"]["pass.vec_nbuffer_setting"] == nbuffer
            seen[op.opcode] += 1
    assert seen == Counter({opcode: 1 for opcode in expected})


def _lower_config_function(func, golden_name):
    from pypto.pil.compile_pipeline import _build_default_pipeline

    program = ir.IRBuilder().create_program([func], "main", ir.Span.unknown())
    for name, transform in _build_default_pipeline():
        if name == "apply_config_ops":
            golden_dir = Path(__file__).with_suffix("")
            check_snapshot(program.functions[func.name], golden_dir / f"{golden_name}_before.pypto")
        program = transform(program)
        if name == "apply_config_ops":
            check_snapshot(program.functions[func.name], golden_dir / f"{golden_name}_after.pypto")
        if name == "create_root_functions":
            assert all(not isinstance(stmt, ir.TensorOpStmt) or stmt.opcode != "OP_CONFIG"
                       for function in program.functions.values() for stmt in _walk(function.body))
            return program
    raise AssertionError("create_root_functions is missing from the pipeline")


def _check_hidden_configs(program, expected):
    # Each segment has a unique marker opcode; auxiliary ops cannot hide extra computations.
    hidden = [func for name, func in program.functions.items() if "_hiddenfunc" in name]
    assert len(hidden) == len(expected)
    seen = Counter()
    for func in hidden:
        info = func.parse_attrs()
        ops = [op for op in func.body.stmts if op.opcode not in ("VIEW", "ASSEMBLE")]
        markers = [op.opcode for op in ops if op.opcode in expected]
        assert len(markers) == 1, [op.opcode for op in ops]
        marker = markers[0]
        tiles, nbuffer = expected[marker]
        assert Counter(op.opcode for op in ops) == Counter({opcode: 1 for opcode in tiles})
        assert info["config"]["pass.vec_nbuffer_setting"] == {-1: nbuffer}
        for op in ops:
            assert op.parse_attrs()["vec_tile_shapes"] == tiles[op.opcode], (marker, op.opcode)
        seen[marker] += 1
    assert seen == Counter({marker: 1 for marker in expected})


def test_partial_branch_config():
    """分支仅覆盖部分配置，其余逐项继承"""
    def kernel(x, y, out):
        pypto.set_vec_tile_shapes(64, 64)
        for _ in pypto.loop(2):
            pypto.set_pass_options(vec_nbuffer_setting={-1: 4})
            if pypto.SymbolicScalar("cond") > 0:
                pypto.set_vec_tile_shapes(32, 32)
                value = x + y
            else:
                pypto.set_pass_options(vec_nbuffer_setting={-1: 2})
                value = x - y
            out.move(value * y)

    with pypto.options("jit_scope", pass_options={"vec_nbuffer_setting": {-1: 8}}):
        rooted = _lower_config_function(pil.compile(kernel, *_tensors()), "partial_branch_configs")
    _check_hidden_configs(rooted, {
        "ADD": ({"ADD": [32, 32], "MUL": [32, 32]}, 4),
        "SUB": ({"SUB": [64, 64], "MUL": [64, 64]}, 2),
    })


def test_segment_isolation():
    """后续设置不影响已结束的计算段"""
    def kernel(x, y, out):
        pypto.set_vec_tile_shapes(64, 64)
        pypto.set_pass_options(vec_nbuffer_setting={-1: 8})
        before = x + y
        for _ in pypto.loop(2):
            pypto.set_vec_tile_shapes(32, 32)
            pypto.set_pass_options(vec_nbuffer_setting={-1: 4})
            out.move(before * y)
        after = out / y
        pypto.set_vec_tile_shapes(16, 16)
        pypto.set_pass_options(vec_nbuffer_setting={-1: 2})
        out.move(after - before)

    rooted = _lower_config_function(pil.compile(kernel, *_tensors()), "later_segment")
    _check_hidden_configs(rooted, {
        "ADD": ({"ADD": [64, 64]}, 8),
        "MUL": ({"MUL": [32, 32]}, 4),
        "SUB": ({"DIV": [64, 64], "SUB": [16, 16]}, 2),
    })


def test_loop_sunk_into_if():
    """内层 loop 合入 if 后继承分支或外层 loop 配置"""
    def kernel(x, y, out):
        pypto.set_vec_tile_shapes(64, 64)
        for _ in pypto.loop(2, name="outer", idx_name="i"):
            pypto.set_vec_tile_shapes(16, 16)
            pypto.set_pass_options(vec_nbuffer_setting={-1: 2})
            if pypto.SymbolicScalar("cond") > 0:
                pypto.set_vec_tile_shapes(32, 32)
                pypto.set_pass_options(vec_nbuffer_setting={-1: 4})
                value = x + y
            else:
                # No override: inherit the outer loop, not JIT defaults.
                value = x - y
            for _ in pypto.loop(2, name="inner", idx_name="j"):
                out.move(value * y)

    with pypto.options("jit_scope", pass_options={"vec_nbuffer_setting": {-1: 8}}):
        rooted = _lower_config_function(pil.compile(kernel, *_tensors()), "nested_loop_after_if")
    # Verify the applied parameters and tiles that are not serialized in IR.
    hidden = [func for name, func in rooted.functions.items() if "_hiddenfunc" in name]
    assert len(hidden) == 4
    seen = Counter()
    for func in hidden:
        ops = [op for op in func.body.stmts if op.opcode not in ("VIEW", "ASSEMBLE")]
        assert len(ops) == 1
        op = ops[0]
        tile = tuple(op.parse_attrs()["vec_tile_shapes"])
        nbuffer = func.parse_attrs()["config"]["pass.vec_nbuffer_setting"]
        assert set(nbuffer) == {-1}
        seen[(op.opcode, tile, nbuffer[-1])] += 1
    assert seen == Counter({
        ("ADD", (32, 32), 4): 1,
        ("MUL", (32, 32), 4): 1,
        ("SUB", (16, 16), 2): 1,
        ("MUL", (16, 16), 2): 1,
    })


def test_entry_config():
    """后面的顶层设置不能改变前面 loop 继承的 JIT 配置"""
    def kernel(x, y, out):
        pypto.set_vec_tile_shapes(32, 32)
        for _ in pypto.loop(2):
            out.move(x + y)
        pypto.set_pass_options(vec_nbuffer_setting={-1: 2})
        out.move(out * y)

    with pypto.options("jit_scope", pass_options={"vec_nbuffer_setting": {-1: 8}}):
        rooted = _lower_config_function(pil.compile(kernel, *_tensors()), "entry_scope")
    # The later top-level setting must not change the earlier loop's inherited JIT config.
    _check_hidden_configs(rooted, {
        "ADD": ({"ADD": [32, 32]}, 8),
        "MUL": ({"MUL": [32, 32]}, 2),
    })
