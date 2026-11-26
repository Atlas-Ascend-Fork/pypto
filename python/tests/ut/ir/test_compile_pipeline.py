# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from pathlib import Path
import re

import pytest

import pypto
from pypto import ir, pil
from pypto.pil.compile_pipeline import (
    _build_default_pipeline,
    _run_pipeline,
    checkpoint,
    resume,
    resume_stage,
)
import pypto.pypto_impl

from .test_common import npuarch

_DATA_DIR = Path(__file__).parent / "test_compile_pipeline_data"


def _norm(text):
    """Fold the id and naming side effects so a direct run and a resumed run can be compared."""

    def _name(match):
        name = match.group(1)
        if re.fullmatch(r"[0-9_]+", name):
            name = "T"
        else:
            name = re.sub(r"(_\d+)+$", "", name)
        return "%" + name

    text = re.sub(r"@\d+", "@I", text)
    text = re.sub(r"!\d+", "!I", text)
    text = re.sub(r"%(?:INCAST|OUTCAST)_SYMBOL\d+", "%T", text)
    text = re.sub(r"%([A-Za-z0-9_.]+)", _name, text)
    text = re.sub(r"(_\d+)+", "_N", text)
    return text.strip()


def _loop_front():
    def loop_kernel(a, b, out):
        pypto.set_vec_tile_shapes(16, 16)
        for _ in pypto.loop(2):
            out[:] = a + b

    a = pypto.Tensor([64, 64], dtype=pypto.DT_FP32, name="a")
    b = pypto.Tensor([64, 64], dtype=pypto.DT_FP32, name="b")
    out = pypto.Tensor([64, 64], dtype=pypto.DT_FP32, name="out")
    return ir.IRBuilder().create_program([pil.compile(loop_kernel, a, b, out)], "main", ir.Span.unknown())


def test_resume_from_dumped_for_loop_matches_direct_run(tmp_path):
    """A for-loop checkpoint keeps typed attrs, so loading it and finishing the pipeline matches a direct run."""
    checkpoint = tmp_path / "loop.pypto"
    checkpoint.write_text(str(_loop_front()))

    pypto.pypto_impl.Reset()
    direct = _norm(str(_run_pipeline(_loop_front(), _build_default_pipeline())))

    pypto.pypto_impl.Reset()
    # A stale ambient must not leak in: only the header can restore [16, 16].
    pypto.set_vec_tile_shapes(32, 32)
    loaded = pypto.pypto_impl.ir.text_load_program(checkpoint.read_text())
    resumed = _norm(str(_run_pipeline(loaded, _build_default_pipeline())))

    assert resumed == direct


def test_resume_from_checkpoint_matches_direct_run():
    # A pre-dumped front program: compiling it here would leave a host-function name that collides
    # with the one create_root_functions registers.
    front_text = (_DATA_DIR / "add_front.pypto").read_text()
    stage = resume_stage()

    # A pipeline run leaves process state behind, so each run starts from a fresh program.
    pypto.pypto_impl.Reset()
    direct = _run_pipeline(pypto.pypto_impl.ir.text_load_program(front_text), _build_default_pipeline())
    pypto.pypto_impl.Reset()
    text = checkpoint(pypto.pypto_impl.ir.text_load_program(front_text), stage)
    pypto.pypto_impl.Reset()
    resumed = resume(text, stage)

    assert _norm(str(resumed)) == _norm(str(direct))


def test_checkpoint_stage_is_limited_to_the_ir_pipeline():
    stage = resume_stage()
    with pytest.raises(ValueError):
        checkpoint(None, stage + 1)
    with pytest.raises(ValueError):
        resume("", stage + 1)


_ARCH = "DAV_3510"
_DYNAMIC_CALL = "v0call(RUNTIME_GetInputShapeDim"


def _dynamic_kernel(a, out):
    bs, seq = a.shape[:2]
    tile = 8
    for bi in pypto.loop((bs + tile - 1) // tile, name="Loop_B", idx_name="bi"):
        for si in pypto.loop((seq + tile - 1) // tile, name="Loop_S", idx_name="si"):
            pypto.set_vec_tile_shapes(tile, tile)
            value = a[bi * tile:(bi + 1) * tile, si * tile:(si + 1) * tile]
            out[bi * tile:(bi + 1) * tile, si * tile:(si + 1) * tile] = value + 1.0


def _build_dynamic():
    b = pypto.symbolic_scalar("b")
    s = pypto.symbolic_scalar("s")
    func = pil.compile(_dynamic_kernel, pypto.tensor([b, s], pypto.DT_FP32), pypto.tensor([b, s], pypto.DT_FP32))
    return ir.IRBuilder().create_program([func], "main", ir.Span.unknown())


def test_checkpoint_text_keeps_dynamic_raw_shape():
    with npuarch(_ARCH):
        pypto.pypto_impl.Reset()
        text = checkpoint(_build_dynamic(), 0)

    assert _DYNAMIC_CALL in text
    assert "dynrawshape" in text


def test_load_dump_roundtrip_is_idempotent():
    with npuarch(_ARCH):
        pypto.pypto_impl.Reset()
        text = checkpoint(_build_dynamic(), 0)

        pypto.pypto_impl.Reset()
        assert str(pypto.pypto_impl.ir.text_load_program(text)) == text


def test_resume_keeps_dynamic_call_count():
    with npuarch(_ARCH):
        pypto.pypto_impl.Reset()
        direct = str(_run_pipeline(_build_dynamic(), _build_default_pipeline()))

        pypto.pypto_impl.Reset()
        text = checkpoint(_build_dynamic(), 0)
        resumed = str(resume(text, 0))

    assert direct.count(_DYNAMIC_CALL) > 0
    assert resumed.count(_DYNAMIC_CALL) == direct.count(_DYNAMIC_CALL)


def _tiled_matmul_kernel(a, b, out):
    for _ in pypto.loop(2, name="Loop_L0"):
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        pypto.set_vec_tile_shapes(16, 16)
        c = pypto.matmul(a, b, pypto.DT_FP32)
        pypto.assemble(c, [0, 0], out)


def test_dump_load_keeps_op_vec_tile():
    """An op-level tile that differs from the ambient one must not absorb it on load."""
    with npuarch(_ARCH):
        tensors = [pypto.tensor([128, 128], pypto.DT_FP32) for _ in range(3)]
        with pypto.options("jit_scope", vec_tile_shapes=[8, 8], cube_tile_shapes=[[64, 64], [64, 64], [64, 64]]):
            func = pil.compile(_tiled_matmul_kernel, *tensors)
            text = str(ir.IRBuilder().create_program([func], "main", ir.Span.unknown()))
    assert "#vec_tile([8, 8])" in text
    assert "#vec_tile([16, 16])" in text
    reloaded = str(pypto.pypto_impl.ir.text_load_program(text))
    assert "#vec_tile([8, 8, 16, 16])" not in reloaded
    assert reloaded == text


def _dump_and_load(tmp_path, name, names, func_name):
    span = ir.Span.unknown()
    scalar = ir.ScalarType(ir.INT32)
    vars_ = [ir.Var(var_name, scalar, span) for var_name in names]
    stmts = [ir.AssignStmt([var], ir.ConstInt(1, ir.INT32, span), span) for var in vars_]
    body = ir.SeqStmts(stmts, span) if len(stmts) > 1 else stmts[0]
    func = ir.Function(func_name, vars_, [scalar], body, span)
    program = ir.IRBuilder().create_program([func], "main", span)
    path = tmp_path / f"{name}.pypto"
    path.write_text(str(program))
    loaded = pypto.pypto_impl.ir.text_load_program(path.read_text())
    return str(loaded)


def test_ir_text_equal_ignores_history_labels(tmp_path):
    lhs = _dump_and_load(tmp_path, "lhs", ["x"], "f_1")
    rhs = _dump_and_load(tmp_path, "rhs", ["x"], "f_9")
    assert ir.text_equal(lhs, rhs)
    assert ir.normalize_ir_text(lhs) == ir.normalize_ir_text(rhs)
    assert not ir.text_equal(lhs, rhs + " #shape([1, 3])")


def test_ir_text_equal_keeps_tensor_identity(tmp_path):
    repeated = _dump_and_load(tmp_path, "repeated", ["x", "x"], "f")
    distinct = _dump_and_load(tmp_path, "distinct", ["x", "y"], "f")
    assert not ir.text_equal(repeated, distinct)


def test_ir_text_equal_handles_digit_names(tmp_path):
    lhs = _dump_and_load(tmp_path, "digit_lhs", ["0"], "f")
    rhs = _dump_and_load(tmp_path, "digit_rhs", ["5"], "f")
    assert ir.text_equal(lhs, rhs)

def _assemble_ssa_kernel(a, b, out):
    tile = 16
    pypto.set_vec_tile_shapes(tile, tile)
    for bi in pypto.loop(0, pypto.ceildiv(a.shape[0], tile), 1, name="LOOP_M", idx_name="bi"):
        valid = pypto.min(a.shape[0] - bi * tile, tile)
        left = pypto.view(a, [tile, tile], [bi * tile, 0], valid_shape=[valid, tile])
        right = pypto.view(b, [tile, tile], [bi * tile, 0], valid_shape=[valid, tile])
        pypto.assemble([(left + right, [bi * tile, 0]), (left + right, [bi * tile, tile])], out, parallel=False)


def test_resume_keeps_assemble_ssa_attribute():
    dim = pypto.symbolic_scalar("b")
    with npuarch(_ARCH):
        tensors = [
            pypto.tensor([dim, 16], pypto.DT_FP32),
            pypto.tensor([dim, 16], pypto.DT_FP32),
            pypto.tensor([dim, 32], pypto.DT_FP32),
        ]
        func = pil.compile(_assemble_ssa_kernel, *tensors)
        program = ir.IRBuilder().create_program([func], "main", ir.Span.unknown())

        pypto.pypto_impl.Reset()
        text = checkpoint(program, 0)
        assert "ASSEMBLE_SSA" in text

        pypto.pypto_impl.Reset()
        resumed = str(resume(text, 0))

    assert "ASSEMBLE_SSA" in resumed
    assert "#toOffset" in resumed
