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
"""Whole-text goldens for the python->C++ bridge emitters in ``exported_custom_op_litenpu.deploy.codegen``.

The generated C++ is what the built ``.so`` actually compiles, so every byte of it is behaviour. The
substring checks in ``test_cpp_codegen_structure.py`` state contracts and keep their reasons; these
cases pin the complete output so an unasserted line cannot drift silently.
"""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` import below.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

import importlib.util

from codegen_golden_data.harness import GOLDEN_SUFFIX, claimed_cases, golden_text
from exported_custom_op_litenpu.deploy import codegen as cpp_mod
import pytest

_UT_DIR = _pl.Path(__file__).resolve().parent
DATA_DIR = _UT_DIR / "codegen_golden_data"

# The same deliberately unresolvable sentinel test_cpp_codegen_structure.py uses: an ``import_via="path"``
# wrapper bakes the path in as a literal, so a real location would make the golden machine-specific.
_INFER_PY_PATH = "/nonexistent/pypto_ut_infer_defs.py"

_BIAS_ATTR = {"name": "bias", "type": "Int", "default": 0}
_TWO_ATTRS = [{"name": "split_sizes", "type": "Int", "default": None}, _BIAS_ATTR]


def _load(unique_name: str, filename: str):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    spec = importlib.util.spec_from_file_location(unique_name, _UT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


shape_samples = _load("infer_shape_samples_ut_goldens", "infer_shape_samples.py")
helpers = _load("codegen_test_helpers_ut_goldens", "codegen_test_helpers.py")


def _pybind(func, cpp_name):
    """The ``import_via="path"`` form: standalone, used by the compile tests, bakes a path literal in."""
    return cpp_mod._generate_pybind_wrapper(func, cpp_name, import_via="path", py_path=_INFER_PY_PATH)


def _pybind_shipped(func, cpp_name):
    """The ``import_via="op_kernel"`` form, which is what the deployed op TU actually carries."""
    return cpp_mod._generate_pybind_wrapper(
        func, cpp_name, import_via="op_kernel", basename="add.py", stem="pypto_compile_Add")


def _shape_body(func, attr_specs=None):
    return cpp_mod._infer_shape_ge_impl_body(cpp_mod._parse_infer_shape_for_codegen(func, attr_specs))


def _dtype_body(func):
    return cpp_mod._infer_dtype_ge_impl_body(cpp_mod._parse_infer_dtype_for_codegen(func))


def _attr_block(spec_dict):
    return cpp_mod._attr_parse_param_block(cpp_mod._normalize_attr_specs([spec_dict])[0])


# A case name carries its own language extension; the golden is that name plus GOLDEN_SUFFIX.
GOLDEN_CASES = {
    "pybind_infer_shape_one_2d.inc": lambda: _pybind(
        shape_samples.infer_shape_one_2d, cpp_mod._CPP_BIND__INFER_SHAPE),
    "pybind_infer_shape_nd_identity.inc": lambda: _pybind(
        shape_samples.infer_shape_nd_identity, cpp_mod._CPP_BIND__INFER_SHAPE),
    "pybind_infer_shape_two_outputs.inc": lambda: _pybind(
        shape_samples.infer_shape_two_outputs, cpp_mod._CPP_BIND__INFER_SHAPE),
    "pybind_infer_shape_string_attr.inc": lambda: _pybind(
        shape_samples.infer_shape_width_pick, cpp_mod._CPP_BIND__INFER_SHAPE),
    "pybind_infer_shape_float_attr.inc": lambda: _pybind(
        shape_samples.infer_shape_scaled_last, cpp_mod._CPP_BIND__INFER_SHAPE),
    "pybind_shipped_infer_shape_one_2d.inc": lambda: _pybind_shipped(
        shape_samples.infer_shape_one_2d, cpp_mod._CPP_BIND__INFER_SHAPE),
    "pybind_shipped_infer_shape_two_outputs.inc": lambda: _pybind_shipped(
        shape_samples.infer_shape_two_outputs, cpp_mod._CPP_BIND__INFER_SHAPE),
    "pybind_shipped_infer_dtype_two.inc": lambda: _pybind_shipped(
        shape_samples.infer_dtype_two, cpp_mod._CPP_BIND__INFER_DTYPE),
    "pybind_infer_dtype_two.inc": lambda: _pybind(
        shape_samples.infer_dtype_two, cpp_mod._CPP_BIND__INFER_DTYPE),
    "pybind_infer_dtype_two_outputs.inc": lambda: _pybind(
        shape_samples.infer_dtype_two_outputs, cpp_mod._CPP_BIND__INFER_DTYPE),
    "infer_shape_ge_body_two_by_two.inc": lambda: _shape_body(shape_samples.infer_shape_two_by_two),
    "infer_shape_ge_body_nd_identity.inc": lambda: _shape_body(shape_samples.infer_shape_nd_identity),
    "infer_shape_ge_body_two_outputs.inc": lambda: _shape_body(shape_samples.infer_shape_two_outputs),
    "infer_shape_ge_body_string_attr.inc": lambda: _shape_body(
        shape_samples.infer_shape_width_pick, [{"name": "width", "type": "String", "default": "FULL"}]),
    "infer_shape_ge_body_float_attr.inc": lambda: _shape_body(
        shape_samples.infer_shape_scaled_last, [{"name": "scale", "type": "Float", "default": 0.5}]),
    "infer_dtype_ge_body_two.inc": lambda: _dtype_body(shape_samples.infer_dtype_two),
    "infer_dtype_ge_body_two_outputs.inc": lambda: _dtype_body(shape_samples.infer_dtype_two_outputs),
    "attr_parse_param_int.inc": lambda: _attr_block(_BIAS_ATTR),
    "attr_extract_attrs_two.inc": lambda: cpp_mod._attr_extract_attrs_method(
        cpp_mod._normalize_attr_specs(_TWO_ATTRS)),
    "dtype_bridge.py": cpp_mod._inline_dtype_conversion_src,
    "host_tu_infer_shape.cpp": lambda: helpers._generate_infer_shape_host_tu_for_test(
        shape_samples.infer_shape_two_by_two, _INFER_PY_PATH),
}


# Parametrized from the case table rather than the directory, so a missing golden fails instead of
# quietly shrinking the parameter set.
@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_generated_text_matches_golden(case):
    path = DATA_DIR / (case + GOLDEN_SUFFIX)
    assert path.exists(), f"{case}{GOLDEN_SUFFIX} is missing; run codegen_golden_data/render.py"
    golden = path.read_text(encoding="utf-8")
    assert golden.strip(), f"{case}{GOLDEN_SUFFIX} is empty"
    if case.endswith(".py"):
        compile(golden, case, "exec")
    assert golden_text(GOLDEN_CASES[case]()) == golden, (
        f"{case}{GOLDEN_SUFFIX} is out of sync with the emitter; run codegen_golden_data/render.py"
    )


def test_every_golden_on_disk_is_claimed_by_a_case():
    """A golden no case builds is dead weight that render.py would delete."""
    on_disk = {p.name[:-len(GOLDEN_SUFFIX)] for p in DATA_DIR.glob("*" + GOLDEN_SUFFIX)}
    claimed, n_suites = claimed_cases()
    assert on_disk <= claimed, f"unclaimed goldens: {sorted(on_disk - claimed)} (suites: {n_suites})"


def test_no_golden_would_be_rewritten_by_pre_commit():
    """trailing-whitespace and end-of-file-fixer have no file filter, so they apply to the goldens."""
    checked = 0
    for p in sorted(DATA_DIR.glob("*" + GOLDEN_SUFFIX)):
        text = p.read_text(encoding="utf-8")
        assert text.endswith("\n"), f"{p.name} does not end with a newline"
        assert "\r" not in text, f"{p.name} carries CR"
        for i, line in enumerate(text.split("\n"), 1):
            assert line == line.rstrip(), f"{p.name}:{i} has trailing whitespace"
        checked += 1
    claimed, n_suites = claimed_cases()
    assert checked == len(claimed), f"checked {checked} goldens, {len(claimed)} cases in {n_suites} suites"
