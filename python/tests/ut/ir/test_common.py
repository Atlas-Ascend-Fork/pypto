# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Common helpers shared across ir.Pass tests (merge_stmts_into_if and friends)."""
from contextlib import contextmanager
import difflib
import logging
import os
from pathlib import Path
from typing import Any

import pypto
from pypto import ir, pil
from pypto.pil.compile_pipeline import compile_new_ir


@contextmanager
def npuarch(arch: str):
    try:
        old_arch = pypto.platform.npuarch
        pypto.platform.npuarch = arch
        yield
    finally:
        pypto.platform.npuarch = old_arch


def check_snapshot(func: Any, golden_path: Path) -> None:
    if os.environ.get("PYPTO_RENDER_IR"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(str(func).strip() + "\n")
        print(f"Updated: {golden_path}")
        return

    actual = str(func)
    golden = golden_path.read_text()
    if golden.strip() != actual.strip():
        diff = "".join(
            difflib.unified_diff(
                golden.splitlines(keepends=True),
                actual.splitlines(keepends=True),
            )
        )
        raise AssertionError("IR snapshot mismatch in %s:\n%s" % (func.name, diff))


def _ssa_verify(verifier, prog, name):
    diagnostic = verifier.verify(prog)
    if diagnostic:
        print(f"{prog}\n")
        print(ir.IRVerifier.generate_report(diagnostic))
        raise SyntaxError(f"IR verification failed after {name}")


def ssa_verify(func, desc: str = ""):
    verifier = ir.IRVerifier.create_default()
    b = ir.IRBuilder()
    prog = b.create_program([func], "main", ir.Span.unknown())
    _ssa_verify(verifier, prog, desc)


def _run_pass_pipeline(func, *args, passes, verify_skip=(), create_new_logical_tensor=True):
    ir_func = pil.compile(func, *args, create_new_logical_tensor=create_new_logical_tensor)
    prog = ir.IRBuilder().create_program([ir_func], "main", ir.Span.unknown())
    verifier = ir.IRVerifier.create_default()
    _ssa_verify(verifier, prog, "original")
    for name, transform in passes:
        prog = transform(prog)
        if name not in verify_skip:
            _ssa_verify(verifier, prog, name)
    return prog.functions[ir_func.name], prog


def run_merge_pass(func, *args, create_new_logical_tensor=True):
    dce = ir.Pass.aggressive_dce()
    canonical = ir.Pass.canonicalize()
    merge = ir.Pass.merge_stmts_into_if()
    passes = [
        ("canonicalize_dce", lambda p:dce(canonical(p))),
        ("canonicalize(merge_stmts)", lambda p:canonical(merge(p))),
        # only symbolic scalar are simplified, skip ssa_verify
        ("simplify_symbolic_scalar", ir.Pass.simplify_symbolic_scalar()),
    ]
    func, _ = _run_pass_pipeline(
        func,
        *args,
        passes=passes,
        verify_skip={"simplify_symbolic_scalar"},
        create_new_logical_tensor=create_new_logical_tensor,
    )
    logging.info("\nmerged:\n%s" % func.body)
    return func


def run_root_function(func, *args, create_new_logical_tensor=True):
    """Compile a kernel and run the compile_new_ir pass sequence up to and including
    create_root_functions, stopping before finalize so the root functions are inspectable.
    Returns the final program.
    """
    infer_token = ir.Pass.infer_token_pass()
    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    remove_redundant_tokens = ir.Pass.remove_redundant_token_pass()
    passes = [
        ("infer_token_pass", infer_token),
        ("canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize_dce2", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        # only symbolic scalar are simplified, skip ssa_verify
        ("simplify_symbolic_scalar", ir.Pass.simplify_symbolic_scalar()),
        ("remove_redundant_token_pass", remove_redundant_tokens),
        # create_root_functions does not support ssa_verify yet, skip it
        ("apply_config_ops", ir.Pass.apply_config_ops()),
        ("create_root_functions", ir.Pass.create_root_functions()),
    ]
    _, prog = _run_pass_pipeline(
        func,
        *args,
        passes=passes,
        verify_skip={"simplify_symbolic_scalar", "create_root_functions"},
        create_new_logical_tensor=create_new_logical_tensor,
    )
    return prog


def run_litenpu_pass(func, *args):
    """Run the litenpu_dynamic_to_static pass (Kirin default pipeline) and return the
    staticized Function IR.

    Callers must wrap this in `with npuarch("DAV_3113"):` so the pass is active. `args`
    are pypto.Tensor inputs; for dynamic-shape scenarios pass `ori_shape` on each input so
    compile_new_ir can register the concrete shape for the pass to resolve -1 dims.
    """
    dce = ir.Pass.aggressive_dce()
    canonicalize = ir.Pass.canonicalize()
    merge_stmts = ir.Pass.merge_stmts_into_if()
    pipeline = [
        ("litenpu_dynamic_to_static", ir.Pass.litenpu_dynamic_to_static()),
        ("infer_token_pass", ir.Pass.infer_token_pass()),
        ("first_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        ("simplify_symbolic_scalar", ir.Pass.simplify_symbolic_scalar()),
        ("remove_redundant_token_pass", ir.Pass.remove_redundant_token_pass()),
    ]
    return compile_new_ir(func, *args, pipeline=pipeline, create_new_logical_tensor=True)
