# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import re

import pytest

import pypto
from pypto import ir, pil
from pypto.pil.compile_pipeline import _build_default_pipeline, _run_pipeline, compile_new_ir
import pypto.pypto_impl


@pytest.mark.skip(reason="write-after-write split-large-fanout-tensor not support yet")
def test_assemble():
    def assemble_kernel(a, out):
        pypto.set_vec_tile_shapes(64, 128)
        aux_mat = pypto.tensor([129, 128], pypto.DT_FP32, name="aux_mat")
        pypto.assemble(pypto.full([129, 128], 0.0, pypto.DT_FP32), [0, 0], aux_mat)
        pypto.assemble(a, [0, 0], aux_mat)
        pypto.assemble(aux_mat + a, [0, 0], out)  # out[:] = a + aux_mat # 对于outcast，暂不支持[:]赋值

    a = pypto.Tensor([129, 128], dtype=pypto.DT_FP32, name="a")
    out = pypto.Tensor([129, 128], dtype=pypto.DT_FP32, name="out")
    compile_new_ir(assemble_kernel, a, out)


# ---------- Write-After-Read ----------
@pytest.mark.skip(reason="FeError::OP_DEPENDENCY_CYCLE")
def test_war_conflict():
    """A read of y followed by an overlapping write of y -> WAR token edge."""

    def foo(x, y):
        pypto.set_vec_tile_shapes(16, 16)
        ry = pypto.view(y, [16, 16], [0, 0])  # read y rows [0, 16)
        a = pypto.view(x, [16, 16], [0, 0])
        s = pypto.add(ry, a)  # live read of ry
        pypto.assemble(s, [8, 0], y)  # write y rows [8, 24) -> overlaps [8, 16)

    x = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="x")
    y = pypto.Tensor(shape=(32, 16), dtype=pypto.DT_FP32, name="y")
    compile_new_ir(foo, x, y)


if __name__ == "__main__":
    test_assemble()


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
