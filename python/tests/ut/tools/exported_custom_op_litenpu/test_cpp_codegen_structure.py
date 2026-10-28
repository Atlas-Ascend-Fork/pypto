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
"""Python-only structural checks for exported_custom_op_litenpu.deploy.codegen
(infer_shape + domi plugin + custom executor).
"""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below. Redundant under pytest: the sibling conftest.py
# already does this, and the guard there makes it a no-op.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

import importlib.util
from pathlib import Path
import sys as _sys
import types as _types
import typing

from exported_custom_op_litenpu.deploy import codegen as cpp_mod
import pytest
import torch

from pypto.extensions.torch_custom_op_litenpu.common import kernel_snippet

_EXPORT_TEST_DIR = Path(__file__).resolve().parent

# The ``domi::FrameworkType`` token the generated onnx plugin registers under. Held locally so
# these checks collect without importing the build module.
_FRAMEWORK_TYPE__ONNX = "ONNX"


def _load_samples_module(unique_name: str, filename: str):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    path = _EXPORT_TEST_DIR / filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


infer_shape_samples = _load_samples_module("infer_shape_samples_ut_structure", "infer_shape_samples.py")
kernel_compile_samples = _load_samples_module("kernel_compile_samples_ut_structure", "cpp_kernel_compile_samples.py")
codegen_test_helpers = _load_samples_module("codegen_test_helpers_ut_structure", "codegen_test_helpers.py")

# The ``import_via="path"`` wrappers bake a .py path in as a compile-time literal. These are TEXT-only
# checks (nothing is compiled or run here); the run tests in test_cpp_codegen_compile.py pass a real
# file. The sentinel is deliberately unresolvable so no generated text can ever name a real, writable
# location, it still exercises the repr/literal path.
_INFER_PY_PATH = "/nonexistent/pypto_ut_infer_defs.py"


def _executor_kwargs(dtypes=None):
    """Kwargs for the new ``_generate_custom_executor_cpp`` signature using the real kernel samples."""
    return dict(
        create_kernel_func=kernel_compile_samples.create_add_kernel,
        infer_shape_func=kernel_compile_samples.add_infer_shape,
        infer_dtype_func=kernel_compile_samples.add_infer_dtype,
        kernel_body_func=kernel_compile_samples.add_kernel_body,
        dtypes=dtypes if dtypes is not None else [torch.float16, torch.float16],
    )


@pytest.mark.parametrize(
    ("fn", "expected_n_params", "expected_n_outputs"),
    [
        (infer_shape_samples.infer_shape_two_by_two, 2, 1),
        (infer_shape_samples.infer_shape_4d_broadcast, 2, 1),
        (infer_shape_samples.infer_shape_sum_last, 2, 1),
        (infer_shape_samples.infer_shape_nd_identity, 1, 1),
    ],
)
def test_parse_infer_shape_metadata(fn, expected_n_params, expected_n_outputs):
    meta = cpp_mod._parse_infer_shape_for_codegen(fn)
    assert len(meta.param_names) == expected_n_params
    assert meta.n_outputs == expected_n_outputs
    assert meta.cpp_bind_name == "inferShape"


def test_parse_infer_shape_two_outputs_metadata():
    meta = cpp_mod._parse_infer_shape_for_codegen(infer_shape_samples.infer_shape_two_outputs)
    assert meta.param_names == ["a_shape", "b_shape"]
    assert meta.n_outputs == 2


def test_parse_infer_shape_rejects_non_torch_size_param():
    """Plain ``tuple[int, int]`` is no longer accepted, torch.Size only."""
    def bad(x_shape: tuple[int, int]) -> torch.Size:
        return torch.Size(x_shape)

    with pytest.raises(TypeError, match="torch.Size"):
        cpp_mod._parse_infer_shape_for_codegen(bad)


def test_parse_infer_shape_accepts_shape_param_named_like_an_emitted_local():
    """A torch.Size param is bridged positionally, so its name cannot shadow an InferShape local."""
    def ok(in0_shape: torch.Size, in1_shape: torch.Size) -> torch.Size:
        return in0_shape

    meta = cpp_mod._parse_infer_shape_for_codegen(ok)
    assert meta.param_names == ["in0_shape", "in1_shape"]


def test_parse_infer_shape_rejects_shape_param_colliding_with_a_wrapper_local():
    """Every param lands in the wrapper's own scope, which declares ``py::dict globals``."""
    def bad(globals: torch.Size) -> torch.Size:
        return globals

    with pytest.raises(ValueError, match="infer_shape parameter name .* pybind wrapper"):
        cpp_mod._parse_infer_shape_for_codegen(bad)


def test_parse_infer_dtype_accepts_param_named_like_an_emitted_local():
    """A dtype param is bridged positionally too, as ``in{i}_dtype_value``."""
    def ok(out_shape: torch.dtype) -> torch.dtype:
        return out_shape

    assert cpp_mod._parse_infer_dtype_for_codegen(ok).param_names == ["out_shape"]


def test_parse_infer_dtype_rejects_param_colliding_with_a_wrapper_local():
    """The dtype wrapper declares the same locals, so that scope still binds."""
    def bad(globals: torch.dtype) -> torch.dtype:
        return globals

    with pytest.raises(ValueError, match="infer_dtype parameter name .* pybind wrapper"):
        cpp_mod._parse_infer_dtype_for_codegen(bad)


def test_attr_name_colliding_with_an_emitted_local_is_still_rejected():
    """Narrowing the rule to attrs must not loosen it: an attr IS emitted into the InferShape body."""
    with pytest.raises(ValueError, match="attr name .* inference body"):
        cpp_mod._normalize_attr_specs([{"name": "in0_shape_vec", "type": "Int", "default": 0}])


def test_infer_shape_wrapper_and_ge_body_keep_like_named_locals_in_separate_scopes():
    """The GE body declares its own in0_shape and calls the wrapper positionally, never by param name."""
    def ok(in0_shape: torch.Size, in1_shape: torch.Size) -> torch.Size:
        return in0_shape

    body = cpp_mod._infer_shape_ge_impl_body(cpp_mod._parse_infer_shape_for_codegen(ok))
    assert "const gert::Shape* in0_shape = context->GetInputShape(0);" in body
    assert "inferShape(in0_shape_vec, in1_shape_vec)" in body


def test_parse_infer_shape_rejects_variadic_count_multi_output():
    """tuple[torch.Size, ...] (variadic count) is rejected, explicit count required."""
    def bad(x_shape: torch.Size) -> tuple[torch.Size, ...]:
        return (x_shape,)

    with pytest.raises(TypeError, match="variadic-arity multi-output"):
        cpp_mod._parse_infer_shape_for_codegen(bad)


@pytest.mark.parametrize(
    "fn",
    [infer_shape_samples.infer_shape_two_by_two, infer_shape_samples.infer_shape_4d_broadcast],
)
def test_infer_shape_ge_impl_body_contains_expected_ops(fn):
    meta = cpp_mod._parse_infer_shape_for_codegen(fn)
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    n_in = len(meta.param_names)
    for i in range(n_in):
        assert f"context->GetInputShape({i})" in body
        assert f"std::vector<int64_t> in{i}_shape_vec" in body
        assert f"in{i}_shape_vec.push_back" in body
    for k in range(meta.n_outputs):
        assert f"context->GetOutputShape({k})" in body
    # Fixed-rank emission is gone, no more std::make_tuple / gert::Shape{...}-list constructor.
    assert "std::make_tuple" not in body
    assert f"{meta.cpp_bind_name}(" in body
    assert "GRAPH_SUCCESS" in body


def test_infer_shape_ge_impl_body_variadic_uses_vector_and_setdims():
    meta = cpp_mod._parse_infer_shape_for_codegen(infer_shape_samples.infer_shape_nd_identity)
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    assert "std::vector<int64_t> in0_shape_vec" in body
    assert "in0_shape_vec.push_back" in body
    assert "SetDimNum(out_shape_vec.size())" in body
    assert "(*out_shape)[j]" in body
    assert "std::make_tuple" not in body


def test_embedded_pybind_single_output_casts_vector():
    fn = infer_shape_samples.infer_shape_two_by_two
    block = cpp_mod._generate_pybind_wrapper(
        fn, cpp_mod._CPP_BIND__INFER_SHAPE, import_via="path", py_path=_INFER_PY_PATH
    )
    assert "namespace {" in block
    assert "py::exec" in block
    assert "inferShape" in block
    # Single torch.Size output → std::vector<int64_t>.
    assert ".cast<std::vector<int64_t>>" in block
    assert "py::gil_scoped_acquire" in block


def test_embedded_pybind_variadic_casts_vector():
    fn = infer_shape_samples.infer_shape_nd_identity
    block = cpp_mod._generate_pybind_wrapper(
        fn, cpp_mod._CPP_BIND__INFER_SHAPE, import_via="path", py_path=_INFER_PY_PATH
    )
    assert ".cast<std::vector<int64_t>>" in block


def test_embedded_pybind_two_outputs_tuple_of_vector_cast():
    fn = infer_shape_samples.infer_shape_two_outputs
    block = cpp_mod._generate_pybind_wrapper(
        fn, cpp_mod._CPP_BIND__INFER_SHAPE, import_via="path", py_path=_INFER_PY_PATH
    )
    # Multi-output: outer std::tuple, each element std::vector<int64_t> from torch.Size.
    assert ".cast<std::tuple<std::vector<int64_t>, std::vector<int64_t>>>" in block


def test_embedded_pybind_imports_torch_and_wraps_shape_args():
    """Shape args annotated torch.Size are wrapped in torch.Size(...) inside the wrapper."""
    fn = infer_shape_samples.infer_shape_two_by_two
    block = cpp_mod._generate_pybind_wrapper(
        fn, cpp_mod._CPP_BIND__INFER_SHAPE, import_via="path", py_path=_INFER_PY_PATH
    )
    # Wrapper imports torch.Size once and wraps each shape arg.
    assert 'py::module_::import("torch").attr("Size")' in block
    assert "_torch_Size(py::cast(x0_shape))" in block
    assert "_torch_Size(py::cast(x1_shape))" in block
    # The inline bridge source carries ``import torch`` so ``torch.Size`` resolves in the wrapper.
    assert "import torch" in block


def test_to_cpp_type_torch_size_and_int():
    assert cpp_mod._to_cpp_type(torch.Size) == "std::vector<int64_t>"
    assert cpp_mod._to_cpp_type(int) == "int64_t"
    assert cpp_mod._to_cpp_type(str) == "std::string"          # a String attr param
    assert (cpp_mod._to_cpp_type(tuple[torch.Size, torch.Size])
            == "std::tuple<std::vector<int64_t>, std::vector<int64_t>>")
    # Generic tuple/list mapping still works for non-shape annotations.
    assert cpp_mod._to_cpp_type(typing.Tuple[int, int]) == "std::tuple<int64_t, int64_t>"
    assert cpp_mod._to_cpp_type(typing.Tuple[int, ...]) == "std::vector<int64_t>"
    assert cpp_mod._to_cpp_type(typing.Tuple[float, ...]) == "std::vector<float>"


def test_infer_shape_host_tu_for_test_is_single_ge_namespace():
    text = codegen_test_helpers._generate_infer_shape_host_tu_for_test(
        infer_shape_samples.infer_shape_two_by_two, _INFER_PY_PATH
    )
    assert 'namespace ge {' in text
    assert "InferShapeGeImpl" in text
    assert "AddCustom" not in text
    assert "OP_ADD" not in text


# --- domi plugin (REGISTER_CUSTOM_OP) ---


def test_local_framework_type_onnx_matches_the_build_module():
    """The locally held token must equal ``build``'s, the value the generated plugin registers under."""
    build = pytest.importorskip("exported_custom_op_litenpu.deploy.build")
    assert _FRAMEWORK_TYPE__ONNX == build._FRAMEWORK_TYPE__ONNX


def test_generate_op_custom_plugin_cpp_has_expected_preamble_and_includes():
    text = cpp_mod._generate_op_custom_plugin_cpp(
        "Add", framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert text.startswith("// Auto-generated\n")
    assert '#include "register/register.h"' in text


def test_generate_op_custom_plugin_cpp_domi_namespace_and_parse_param_shape():
    text = cpp_mod._generate_op_custom_plugin_cpp(
        "Add", framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert "namespace domi {" in text
    assert "Status ParseParamAdd(const ge::Operator& op_src, ge::Operator& op_dest)" in text
    assert "return SUCCESS;" in text
    assert 'REGISTER_CUSTOM_OP("Add")' in text
    assert f".FrameworkType({_FRAMEWORK_TYPE__ONNX})" in text
    # OriginOpType is the qualified ``<domain>::<opset>::<op_type>`` key the
    # CANN parser builds for each ONNX NodeProto, bare names are never
    # matched. Defaults: domain="pypto", opset_version=1.
    assert '.OriginOpType("pypto::1::Add")' in text
    assert ".ParseParamsByOperatorFn(ParseParamAdd)" in text


@pytest.mark.parametrize(
    "op_type",
    ["MyOp", "PyptoCustomOpAdd"],
)
def test_generate_op_custom_plugin_cpp_op_type_parameterizes_names(op_type: str):
    text = cpp_mod._generate_op_custom_plugin_cpp(
        op_type, framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert f"ParseParam{op_type}" in text
    assert f'REGISTER_CUSTOM_OP("{op_type}")' in text
    assert f'.OriginOpType("pypto::1::{op_type}")' in text
    assert f".ParseParamsByOperatorFn(ParseParam{op_type})" in text


@pytest.mark.parametrize(
    ("domain", "domain_opset_version"),
    [
        ("pypto", 1),
        ("ai.onnx.contrib", 1),
        ("custom.vendor", 7),
    ],
)
def test_generate_op_custom_plugin_cpp_domain_and_opset_parameterize_origin_op_type(
    domain: str, domain_opset_version: int,
):
    text = cpp_mod._generate_op_custom_plugin_cpp(
        "MyOp",
        framework_type=_FRAMEWORK_TYPE__ONNX,
        domain=domain,
        domain_opset_version=domain_opset_version,
    )
    assert f'.OriginOpType("{domain}::{domain_opset_version}::MyOp")' in text


def test_generate_custom_executor_cpp_has_system_includes_and_class():
    full = cpp_mod._generate_custom_executor_cpp("Add", **_executor_kwargs())
    # The thin executor only includes the shared base header; the base pulls in the GE headers.
    assert '#include "pto_custom_op.h"' in full
    assert "class Add : public PtoCustomOp" in full
    assert "REG_AUTO_MAPPING_OP(Add)" in full
    # Compile/DeclareLaunchArgs live in the base; the subclass provides only the per-op hooks
    # (module-stem, kernel-.py basename, InferShape/InferDataType).
    assert "SinkableExecuteOp" not in full
    assert "PrepareExecute" not in full
    assert 'return "pypto_compile_Add";' in full
    assert 'return "add.py";' in full
    assert "ge::graphStatus InferShape(gert::InferShapeContext *context) override" in full
    assert "ge::graphStatus InferDataType(gert::InferDataTypeContext *context) override" in full
    # Launch is now fully base-class-generic (ge 20260717 annotated-args launch): the per-op TU emits
    # NO launch code, so neither the deleted eager-launch API tokens nor the base-class launch tokens
    # (both the ge 20260629 offline-launch names and the ge 20260717 names) appear here.
    assert "LaunchKernel" not in full
    assert "KernelBin" not in full
    assert "LaunchCompiledKernel" not in full
    assert "EagerOpExecutionContext" not in full
    assert "DeclareOfflineLaunch" not in full
    assert "OfflineLaunchTask" not in full


def test_generate_op_tu_cpp_merges_executor_and_reg_op_prototype():
    # The single op_host TU carries BOTH the executor (class + REG_AUTO_MAPPING_OP) and the GE OpDef
    # REG_OP prototype, with graph/operator_reg.h pulled in next to the base header so the block compiles.
    tu, _snippet = cpp_mod._generate_op_tu_cpp_with_snippet("Add", **_executor_kwargs())
    # executor half
    assert '#include "pto_custom_op.h"' in tu
    assert "class Add : public PtoCustomOp" in tu
    assert "REG_AUTO_MAPPING_OP(Add)" in tu
    # op-def half (appended REG_OP prototype), absent from the standalone executor, added by the merge.
    assert '#include "graph/operator_reg.h"' in tu
    assert "REG_OP(Add)" in tu
    assert "OP_END_FACTORY_REG(Add)" in tu
    exe = cpp_mod._generate_custom_executor_cpp("Add", **_executor_kwargs())
    assert "REG_OP(Add)" not in exe
    # operator_reg.h appears exactly once; the class precedes the appended prototype.
    assert tu.count('#include "graph/operator_reg.h"') == 1
    assert tu.index("class Add") < tu.index("REG_OP(Add)")


def test_custom_executor_class_fragment_is_embedded_in_full_output():
    kwargs = _executor_kwargs()
    infer_shape_meta = cpp_mod._parse_infer_shape_for_codegen(kwargs["infer_shape_func"])
    infer_dtype_meta = cpp_mod._parse_infer_dtype_for_codegen(kwargs["infer_dtype_func"])
    fragment = cpp_mod._custom_executor_class_cpp(
        "Add",
        infer_shape_meta=infer_shape_meta,
        infer_dtype_meta=infer_dtype_meta,
    )
    full = cpp_mod._generate_custom_executor_cpp("Add", **kwargs)
    assert fragment in full
    assert '#include "pto_custom_op.h"' not in fragment
    # ShapeInferOp inheritance + the two infer member overrides live on the executor now.
    assert "public ge::ShapeInferOp" in fragment
    assert "InferShape(gert::InferShapeContext" in fragment
    assert "InferDataType(gert::InferDataTypeContext" in fragment


def test_executor_infer_wrapper_imports_from_file_single_path():
    # The executor's infer wrappers use import_via="op_kernel": they import the infer func from the shipped
    # op_kernel/<stem>.py, one path, no mode branch, no inline copy of the user source.
    full = cpp_mod._generate_custom_executor_cpp("Add", **_executor_kwargs())
    assert "PtoCustomOp::ImportKernelModule(" in full
    # The infer func is fetched by its ORIGINAL name (as the snippet defines it) from the imported module.
    assert '.attr("add_infer_shape")' in full
    assert '.attr("add_infer_dtype")' in full
    # stem (module memo key) vs basename (file), kept distinct, both baked as codegen literals.
    assert 'ImportKernelModule("pypto_compile_Add", "add.py")' in full
    # No mode branch, and a null module is FATAL (throws, caught by the member's try/catch).
    assert "EmbedModeStr" not in full
    assert "if (_mode" not in full
    assert "throw py::error_already_set();" in full


def test_no_generated_artifact_embeds_user_python_source():
    # The single-source-of-truth guard, across the WHOLE generated surface: no artifact carries user
    # Python as an R-string. Every wrapper imports its infer func from a module instead.
    executor = cpp_mod._generate_custom_executor_cpp("Add", **_executor_kwargs())
    merged, _snippet = cpp_mod._generate_op_tu_cpp_with_snippet("Add", **_executor_kwargs())
    host_tu = codegen_test_helpers._generate_infer_shape_host_tu_for_test(
        infer_shape_samples.infer_shape_two_by_two, _INFER_PY_PATH
    )
    for artifact in (executor, merged, host_tu):
        # No kernel-compile R-string and no inline copy of any user infer source.
        assert "kPyptoCompileSrc" not in artifact
        assert 'R"PYEMB(' not in artifact
        assert "_user_infer_src" not in artifact
        assert "GetEmbeddedCompileSrc" not in artifact
        assert "EmbedModeStr" not in artifact
        # The user's def bodies never appear, only the generic bridge/loader Python is inline.
        assert "def add_infer_shape" not in artifact
        assert "def add_infer_dtype" not in artifact
        assert "def infer_shape_two_by_two" not in artifact


def test_host_tu_imports_infer_func_by_path():
    # The standalone host TU has no PtoCustomOp, so it loads the .py itself via importlib with the
    # path baked in as a compile-time literal.
    text = codegen_test_helpers._generate_infer_shape_host_tu_for_test(
        infer_shape_samples.infer_shape_two_by_two, _INFER_PY_PATH
    )
    assert "spec_from_file_location" in text
    assert repr(_INFER_PY_PATH) in text
    assert "_pypto_spec.loader.exec_module(_pypto_infer_mod)" in text
    assert '.attr("infer_shape_two_by_two")' in text
    # Self-contained: the deployed executor's import path is absent.
    assert "ImportKernelModule" not in text
    assert "PtoCustomOp" not in text


def test_executor_infer_dtype_bridge_exec_inline():
    # The generic torch<->GE dtype bridge is still exec'd inline (it is never sourced from the .py),
    # while the user's infer func comes from the imported module rather than a re-exec.
    wrapper = cpp_mod._generate_pybind_wrapper(
        kernel_compile_samples.add_infer_dtype,
        cpp_mod._CPP_BIND__INFER_DTYPE,
        import_via="op_kernel",
        basename="add.py",
        stem="pypto_compile_Add",
    )
    assert "py::exec(_dtype_bridge_src, globals, globals)" in wrapper
    assert "_ge_data_type_enum_value_to_torch_dtype" in wrapper
    # The bridge exec precedes the import; the user infer source is nowhere in the wrapper.
    assert wrapper.index("py::exec(_dtype_bridge_src") < wrapper.index("ImportKernelModule")
    assert "_user_infer_src" not in wrapper
    assert "def add_infer_dtype" not in wrapper


def test_executor_member_infer_wrapped_in_try_catch():
    # A propagated infer error (e.g. a strict deploy_file resolve failure) is caught in the member and
    # turned into GRAPH_FAILED, never unwinding across the GE ABI.
    full = cpp_mod._generate_custom_executor_cpp("Add", **_executor_kwargs())
    assert "catch (py::error_already_set" in full
    assert "catch (const std::exception &ex)" in full
    # ERROR level with the exception text interpolated: GE runs the op_proto InferShape pass BEFORE
    # Compile, so this is the first failure a user hits and the old wording carried no reason at all.
    assert 'PTO_CUSTOM_LOGE("ERROR: Add::InferShape %s -> GRAPH_FAILED\\n", _what.c_str());' in full
    assert 'PTO_CUSTOM_LOGE("ERROR: Add::InferDataType %s -> GRAPH_FAILED\\n", _what.c_str());' in full
    assert "python error -> GRAPH_FAILED" not in full
    # GIL is re-acquired in the catch before PyErr_Print (the member runs outside the wrapper's GIL scope).
    assert "py::gil_scoped_acquire _g; e.restore(); PyErr_Print();" in full


def test_op_kernel_py_exposes_infer_and_compile():
    # The shipped op_kernel/<stem>.py (the artifact the deploy_file wrapper imports from) carries all
    # three: infer_shape, infer_dtype, and __pypto_compile, no snippet change was needed.
    kwargs = _executor_kwargs()
    snippet = kernel_snippet.build_kernel_compile_snippet(
        create_kernel_func=kwargs["create_kernel_func"],
        infer_shape_func=kwargs["infer_shape_func"],
        infer_dtype_func=kwargs["infer_dtype_func"],
        n_inputs=len(kwargs["dtypes"]),
        kernel_body_func=kwargs["kernel_body_func"],
    )
    assert "def add_infer_shape" in snippet
    assert "def add_infer_dtype" in snippet
    assert "def __pypto_compile" in snippet


def test_op_type_parameterizes_plugin_host_and_executor_cpp():
    op_type = "MyOp"
    plugin = cpp_mod._generate_op_custom_plugin_cpp(
        op_type, framework_type=_FRAMEWORK_TYPE__ONNX
    )
    assert "ParseParamMyOp" in plugin
    assert 'REGISTER_CUSTOM_OP("MyOp")' in plugin
    assert ".FrameworkType(ONNX)" in plugin
    assert '.OriginOpType("pypto::1::MyOp")' in plugin

    host = codegen_test_helpers._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_two_by_two,
        infer_shape_samples.infer_dtype_two,
        op_type=op_type,
        dtypes=[torch.float16, torch.float16],
    )
    # The op-def TU is now a bare REG_OP prototype: no IMPL_OP_INFERSHAPE, no infer free funcs,
    # no OpDef/OP_ADD. Inference is registered via the executor's ge::ShapeInferOp member methods.
    assert "REG_OP(MyOp)" in host
    assert "OP_END_FACTORY_REG(MyOp)" in host
    assert "IMPL_OP_INFERSHAPE" not in host
    assert "InferShapeGeImpl" not in host
    assert "OP_ADD" not in host
    assert '#include "graph/operator_reg.h"' in host

    exe = cpp_mod._generate_custom_executor_cpp(op_type, **_executor_kwargs())
    assert "class MyOp : public PtoCustomOp, public ge::ShapeInferOp" in exe
    assert "ge::graphStatus InferShape(gert::InferShapeContext *context) override" in exe
    assert "ge::graphStatus InferDataType(gert::InferDataTypeContext *context) override" in exe
    assert "REG_AUTO_MAPPING_OP(MyOp)" in exe


def test_op_kernel_py_is_returned_verbatim_and_is_not_compiled(tmp_path):
    # return_snippet still yields the kernel snippet alongside the TU, so build.py can ship
    # it verbatim as the dev-editable op_kernel/<stem>.py, the SINGLE source of this op's Python, which
    # the TU no longer duplicates, and it is NOT part of the compiled sources list.
    from exported_custom_op_litenpu.deploy.cpp_naming import camel_case_to_snake_case

    op_type = "Add"
    tu_cpp, embedded_py_src = cpp_mod._generate_op_tu_cpp_with_snippet(op_type, **_executor_kwargs())

    # The snippet carries the op's whole Python contract...
    assert "def __pypto_compile" in embedded_py_src
    assert "def add_infer_shape" in embedded_py_src
    assert "def add_infer_dtype" in embedded_py_src
    # ...and NONE of it is duplicated into the TU.
    assert "def __pypto_compile" not in tu_cpp
    assert "def add_infer_shape" not in tu_cpp
    assert "def add_infer_dtype" not in tu_cpp

    # Build write path (mirroring _codegen_cpp_sources_into): op_kernel/<stem>.py is written verbatim
    # into a SEPARATE dir from the compiled op_host/<stem>.cpp, and the compiled-sources list contains
    # only the .cpp, never the .py.
    stem = camel_case_to_snake_case(op_type)
    op_dir = tmp_path / op_type
    for rel, content in ((f"op_host/{stem}.cpp", tu_cpp), (f"op_kernel/{stem}.py", embedded_py_src)):
        path = op_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    kernel_py = op_dir / "op_kernel" / f"{stem}.py"
    assert kernel_py.is_file()
    assert kernel_py.read_text() == embedded_py_src   # verbatim, no framing
    assert (op_dir / "op_host" / f"{stem}.cpp").is_file()
    compiled_sources = [op_dir / "op_host" / f"{stem}.cpp"]  # what _codegen_cpp_sources_into compiles
    assert kernel_py not in compiled_sources


def test_generate_op_custom_plugin_cpp_framework_type_argument():
    text = cpp_mod._generate_op_custom_plugin_cpp("MyOp", framework_type="CAFFE")
    assert ".FrameworkType(CAFFE)" in text
    assert ".FrameworkType(ONNX)" not in text


def test_invalid_op_type_raises():
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod.validate_op_type_identifier("")
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._generate_custom_executor_cpp("1Bad", **_executor_kwargs())
    with pytest.raises(ValueError, match="op_type"):
        cpp_mod._generate_op_custom_plugin_cpp(
            "1Bad", framework_type=_FRAMEWORK_TYPE__ONNX
        )


def test_generate_op_custom_def_cpp_builds_inputs_from_dtypes():
    text = codegen_test_helpers._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_three_4d,
        infer_shape_samples.infer_dtype_three,
        op_type="MyOp",
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    # REG_OP IO is generated one ``.INPUT(inN, ...)`` / ``.OUTPUT(outK, ...)`` per
    # dtype; identifiers are bare and the DT_* tokens are unqualified (the block
    # lives inside ``namespace ge``).
    # Input dtypes come from codegen._torch_dtype_to_ge_dtype; output dtype matches infer_dtype (input0).
    assert '.INPUT(in0, TensorType({DT_FLOAT16}))' in text  # input 0
    assert '.INPUT(in1, TensorType({DT_FLOAT}))' in text    # input 1
    assert '.INPUT(in2, TensorType({DT_BF16}))' in text     # input 2
    assert '.OUTPUT(out0, TensorType({DT_FLOAT16}))' in text  # output matches input0


def test_generate_op_custom_def_cpp_rejects_empty_dtypes():
    with pytest.raises(ValueError, match="dtypes"):
        codegen_test_helpers._generate_op_custom_def_cpp(
            infer_shape_samples.infer_shape_one_2d,
            infer_shape_samples.infer_dtype_one,
            op_type="MyOp",
            dtypes=[],
        )


def _infer_dtype_copy_input1(input0_dtype: torch.dtype, input1_dtype: torch.dtype) -> torch.dtype:
    return input1_dtype


def _infer_dtype_const_fp32(input0_dtype: torch.dtype) -> torch.dtype:
    return torch.float32


def test_generate_op_custom_def_cpp_infer_dtype_copy_and_const():
    text_copy = codegen_test_helpers._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_two_by_two,
        _infer_dtype_copy_input1,
        op_type="MyOp",
        dtypes=[torch.float16, torch.float32],
    )
    # Static REG_OP output declaration reflects codegen-time call: returns input1 (float32) -> DT_FLOAT.
    assert '.OUTPUT(out0, TensorType({DT_FLOAT}))' in text_copy
    # The runtime InferDataType logic now lives on the executor (ge::ShapeInferOp member method); the
    # def TU carries no infer body. Verify the relocated logic via the body generator it shares.
    dtype_body = cpp_mod._infer_dtype_ge_impl_body(
        cpp_mod._parse_infer_dtype_for_codegen(_infer_dtype_copy_input1)
    )
    assert "in1_dtype_value = static_cast<int64_t>(context->GetInputDataType(1))" in dtype_body
    assert "SetOutputDataType(0, static_cast<ge::DataType>" in dtype_body
    # The dtype<->int helper is inlined into the pybind wrapper carried by the executor
    # (defined at module scope of the embedded source; the .so imports nothing
    # from pypto.extensions.torch_custom_op_litenpu).
    exe_copy = cpp_mod._generate_custom_executor_cpp("MyOp", **_executor_kwargs())
    assert "_torch_dtype_to_ge_data_type_enum_value_recursive" in exe_copy

    text_const = codegen_test_helpers._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_one_2d,
        _infer_dtype_const_fp32,
        op_type="MyOpConst",
        dtypes=[torch.float16],
    )
    # Static REG_OP output declaration matches the constant returned at codegen time.
    assert '.OUTPUT(out0, TensorType({DT_FLOAT}))' in text_const
    # Runtime emission (relocated to the executor body) still goes through the embedded wrapper.
    const_body = cpp_mod._infer_dtype_ge_impl_body(
        cpp_mod._parse_infer_dtype_for_codegen(_infer_dtype_const_fp32)
    )
    assert "SetOutputDataType(0, static_cast<ge::DataType>" in const_body


def test_generate_op_custom_def_cpp_two_outputs_and_infer_dtype_tuple():
    text = codegen_test_helpers._generate_op_custom_def_cpp(
        infer_shape_samples.infer_shape_two_outputs,
        infer_shape_samples.infer_dtype_two_outputs,
        op_type="DualOut",
        dtypes=[torch.float16, torch.float32],
    )
    assert '.OUTPUT(out0, TensorType({' in text
    assert '.OUTPUT(out1, TensorType({' in text
    # The def TU is bare REG_OP, the multi-output infer bodies moved to the executor. Verify the
    # std::tuple unpack via the shared body generators (what the executor's member methods embed).
    dtype_body = cpp_mod._infer_dtype_ge_impl_body(
        cpp_mod._parse_infer_dtype_for_codegen(infer_shape_samples.infer_dtype_two_outputs)
    )
    assert "std::get<0>(out_dt_tuple)" in dtype_body
    assert "std::get<1>(out_dt_tuple)" in dtype_body
    assert "SetOutputDataType(0, static_cast<ge::DataType>(std::get<0>(out_dt_tuple))" in dtype_body
    assert "SetOutputDataType(1, static_cast<ge::DataType>(std::get<1>(out_dt_tuple))" in dtype_body
    shape_body = cpp_mod._infer_shape_ge_impl_body(
        cpp_mod._parse_infer_shape_for_codegen(infer_shape_samples.infer_shape_two_outputs)
    )
    assert "std::get<0>(out_shape_tuple)" in shape_body
    assert "std::get<1>(out_shape_tuple)" in shape_body

# Operator-attribute (AttrSpec) emission: the REG_OP attr line, the plugin ParseParam and
# the executor ExtractAttrs, all driven from the one ordered attr list.


_BIAS_ATTR = {"name": "bias", "type": "Int", "default": 0}  # .ATTR(bias, Int, 0) (default given => optional)


def test_reg_op_block_emits_attr_line_with_default():
    block = cpp_mod._reg_op_block(
        kernel_compile_samples.add_infer_shape,
        kernel_compile_samples.add_infer_dtype,
        op_type="AddBias",
        dtypes=[torch.float16, torch.float16],
        attr_specs=[_BIAS_ATTR],
    )
    # Default given => .ATTR(name, Int, <default>); the attr line follows the IO lines and precedes the end.
    assert ".ATTR(bias, Int, 0)" in block
    assert block.index(".OUTPUT(out0") < block.index(".ATTR(bias, Int, 0)") < block.index("OP_END_FACTORY_REG")


def test_reg_op_block_required_attr_when_no_default():
    block = cpp_mod._reg_op_block(
        kernel_compile_samples.add_infer_shape,
        kernel_compile_samples.add_infer_dtype,
        op_type="AddBias",
        dtypes=[torch.float16, torch.float16],
        attr_specs=[{"name": "bias", "type": "Int", "default": None}],
    )
    assert ".REQUIRED_ATTR(bias, Int)" in block


def test_reg_op_block_no_attrs_is_byte_identical():
    kw = dict(op_type="AddBias", dtypes=[torch.float16, torch.float16])
    base = cpp_mod._reg_op_block(kernel_compile_samples.add_infer_shape,
                                 kernel_compile_samples.add_infer_dtype, **kw)
    empty = cpp_mod._reg_op_block(kernel_compile_samples.add_infer_shape,
                                  kernel_compile_samples.add_infer_dtype, attr_specs=[], **kw)
    none = cpp_mod._reg_op_block(kernel_compile_samples.add_infer_shape,
                                 kernel_compile_samples.add_infer_dtype, attr_specs=None, **kw)
    assert base == empty == none  # empty/None attrs => no attr line, identical to a pre-attrs op
    assert ".ATTR(" not in base and ".REQUIRED_ATTR(" not in base


def test_plugin_emits_parse_param_int_block_with_null_guard():
    text = cpp_mod._generate_op_custom_plugin_cpp(
        "AddBias", framework_type=_FRAMEWORK_TYPE__ONNX, attr_specs=[_BIAS_ATTR],
    )
    # Real ParseParam (not the stub): parses the ONNX "attribute" JSON, matches name+type==2, SetAttr.
    assert '#include <nlohmann/json.hpp>' in text
    assert 'nlohmann::json::parse(attrs_string.GetString())' in text
    assert 'if (attr["name"] == "bias" && attr["type"] == 2)' in text
    assert 'attr.contains("i") && !attr["i"].is_null()' in text  # protobuf-drops-default-0 null-guard
    assert 'op_dest.SetAttr("bias", value)' in text
    # Nothing may throw out of the registered domi callback: the throwing json/stof calls are wrapped.
    assert "catch (const nlohmann::json::exception &e)" in text
    assert "catch (const std::exception &e)" in text
    assert "return FAILED;" in text


def test_plugin_no_attrs_is_stub_and_byte_identical():
    stub = cpp_mod._generate_op_custom_plugin_cpp("AddBias", framework_type=_FRAMEWORK_TYPE__ONNX)
    empty = cpp_mod._generate_op_custom_plugin_cpp(
        "AddBias", framework_type=_FRAMEWORK_TYPE__ONNX, attr_specs=[],
    )
    assert stub == empty  # empty attrs => the historical no-op stub, byte-identical
    assert '#include <nlohmann/json.hpp>' not in stub
    assert "SetAttr" not in stub


def test_executor_emits_extract_attrs_with_get_int_at_index_0():
    exe = cpp_mod._generate_custom_executor_cpp("AddBias", attr_specs=[_BIAS_ATTR], **_executor_kwargs())
    assert "void ExtractAttrs(const gert::RuntimeAttrs *attrs" in exe
    assert "if (attrs == nullptr)" in exe
    assert 'out["bias"] = "0";' in exe            # default branch from AttrSpec.default
    assert "attrs->GetInt(0)" in exe              # bias is declaration-index 0 -> GetInt(0)
    assert 'out["bias"] = bias_ptr ? std::to_string(*bias_ptr) : "0";' in exe


def test_executor_no_attrs_omits_extract_attrs_and_is_byte_identical():
    base = cpp_mod._generate_custom_executor_cpp("AddBias", **_executor_kwargs())
    empty = cpp_mod._generate_custom_executor_cpp("AddBias", attr_specs=[], **_executor_kwargs())
    assert base == empty  # no attrs => no override emitted (base's empty stub is used), byte-identical
    assert "ExtractAttrs(" not in base


def test_extract_attrs_index_is_global_declaration_position_two_attrs():
    # anti-drift: the ExtractAttrs positional index is the GLOBAL 0-based attr position, so a second
    # attr reads at index 1 (NOT per-type). Both Int here (only the Int path is emitted in this slice).
    two = [{"name": "split_sizes", "type": "Int", "default": None}, {"name": "dim", "type": "Int", "default": 0}]
    exe = cpp_mod._generate_custom_executor_cpp("TwoAttr", attr_specs=two, **_executor_kwargs())
    assert "attrs->GetInt(0)" in exe   # split_sizes = declaration-index 0
    assert "attrs->GetInt(1)" in exe   # dim = declaration-index 1
    reg = cpp_mod._reg_op_block(
        kernel_compile_samples.add_infer_shape, kernel_compile_samples.add_infer_dtype,
        op_type="TwoAttr", dtypes=[torch.float16, torch.float16], attr_specs=two,
    )
    # REG_OP declaration order matches the ExtractAttrs index order (split_sizes before dim).
    assert reg.index(".REQUIRED_ATTR(split_sizes, Int)") < reg.index(".ATTR(dim, Int, 0)")


def test_attr_codegen_emits_from_a_dict_spec():
    dct = cpp_mod._generate_custom_executor_cpp("AddBias", attr_specs=[_BIAS_ATTR], **_executor_kwargs())
    assert "attrs->GetInt(0)" in dct


def test_op_tu_with_snippet_threads_attrs_into_both_reg_op_and_executor():
    tu, _py = cpp_mod._generate_op_tu_cpp_with_snippet("AddBias", attr_specs=[_BIAS_ATTR], **_executor_kwargs())
    assert ".ATTR(bias, Int, 0)" in tu          # REG_OP half
    assert "attrs->GetInt(0)" in tu             # executor half
    assert "ExtractAttrs(" in tu


# Float / String / ListInt attr emission + the shape-affecting InferShape path.


def _reg_line(spec):
    return cpp_mod._reg_op_block(
        kernel_compile_samples.add_infer_shape, kernel_compile_samples.add_infer_dtype,
        op_type="AttrOp", dtypes=[torch.float16, torch.float16], attr_specs=[spec],
    )


def test_reg_op_float_string_listint_lines():
    assert ".ATTR(alpha, Float, 0.01)" in _reg_line({"name": "alpha", "type": "Float", "default": 0.01})
    assert ".ATTR(mode, String, \"sum\")" in _reg_line({"name": "mode", "type": "String", "default": "sum"})
    assert ".ATTR(crop_size, ListInt, {2, 2})" in _reg_line({"name": "crop_size", "type": "ListInt", "default": [2, 2]})
    # required (default None) forms
    assert ".REQUIRED_ATTR(alpha, Float)" in _reg_line({"name": "alpha", "type": "Float", "default": None})
    assert ".REQUIRED_ATTR(crop_size, ListInt)" in _reg_line({"name": "crop_size", "type": "ListInt", "default": None})


def test_plugin_parse_param_float_string_listint():
    fl = cpp_mod._generate_op_custom_plugin_cpp("AlphaOp", framework_type=_FRAMEWORK_TYPE__ONNX,
                                                attr_specs=[{"name": "alpha", "type": "Float", "default": 0.01}])
    assert 'attr["type"] == 1' in fl
    assert 'std::stof(attr["f"].get<std::string>())' in fl        # FLOAT-as-string
    assert 'attr["f"].is_string()' in fl                          # defensive: fall back to numeric f
    # The absent-key fallback is the type's proto ZERO, never the declared default: protobuf drops the
    # "f" key exactly when the author set 0.0, so a declared-default fallback would rewrite an
    # author-set zero. The declared default rides on the REG_OP ``.ATTR(alpha, Float, 0.01)`` literal.
    # ``0.0f`` is also a valid C++ float literal, unlike the ``0f`` a %.9g value-string would give.
    assert 'float value = 0.0f;' in fl
    assert 'float value = 0.01f;' not in fl
    whole = cpp_mod._generate_op_custom_plugin_cpp("ScaleOp", framework_type=_FRAMEWORK_TYPE__ONNX,
                                                   attr_specs=[{"name": "scale", "type": "Float", "default": 1.0}])
    assert 'float value = 0.0f;' in whole and 'float value = 1.0f;' not in whole
    st = cpp_mod._generate_op_custom_plugin_cpp("ModeOp", framework_type=_FRAMEWORK_TYPE__ONNX,
                                                attr_specs=[{"name": "mode", "type": "String", "default": "sum"}])
    assert 'attr["type"] == 3' in st
    assert 'attr["s"].get<std::string>()' in st
    assert ': "";' in st and ': "sum";' not in st                 # String proto-zero fallback
    assert 'op_dest.SetAttr("mode", value.c_str())' in st
    li = cpp_mod._generate_op_custom_plugin_cpp(
        "CropOp", framework_type=_FRAMEWORK_TYPE__ONNX,
        attr_specs=[{"name": "crop_size", "type": "ListInt", "default": [2, 2]}])
    assert 'attr["type"] == 7' in li
    assert 'for (auto e : attr["ints"]) value.push_back(e.get<int64_t>());' in li
    assert 'op_dest.SetAttr("crop_size", value)' in li
    assert '#include <string>' in li and '#include <vector>' in li


def test_executor_extract_attrs_float_uses_percent_9g():
    exe = cpp_mod._generate_custom_executor_cpp(
        "AlphaOp", attr_specs=[{"name": "alpha", "type": "Float", "default": 0.01}],
        **_executor_kwargs())
    assert "attrs->GetFloat(0)" in exe
    assert 'snprintf(_buf, sizeof _buf, "%.9g", (double)*alpha_ptr)' in exe  # full float32 round-trip
    assert 'out["alpha"] = "0.01";' in exe                                    # default branch


def test_executor_extract_attrs_string_and_listint():
    st = cpp_mod._generate_custom_executor_cpp(
        "ModeOp", attr_specs=[{"name": "mode", "type": "String", "default": "sum"}],
        **_executor_kwargs())
    assert "attrs->GetStr(0)" in st
    assert 'out["mode"] = mode_ptr ? mode_ptr : "sum";' in st
    li = cpp_mod._generate_custom_executor_cpp(
        "CropOp", attr_specs=[{"name": "crop_size", "type": "ListInt", "default": [2, 2]}],
        **_executor_kwargs())
    assert "attrs->GetListInt(0)" in li
    assert "ListIntToJsonStr(_v)" in li                                        # base helper for the JSON-array string


# Attr edge values: empty ListInt default, negative and large Int (string assertions on the emitted text
# only; no TU compile covers edge-value defaults, because the plugin ParseParam body is built from the
# attr's name and type alone and its fallback is the proto zero, so the default never reaches the C++).
def test_reg_op_and_executor_empty_listint_default():
    # An empty ListInt default `[]` -> REG_OP ``.ATTR(k, ListInt, {})`` + the executor's null-branch
    # fallback string ``"[]"`` (JSON-array form). The ParseParam loop is unconditional (no fallback used).
    assert ".ATTR(k, ListInt, {})" in _reg_line({"name": "k", "type": "ListInt", "default": []})
    exe = cpp_mod._generate_custom_executor_cpp(
        "EmptyLi", attr_specs=[{"name": "k", "type": "ListInt", "default": []}], **_executor_kwargs())
    assert 'out["k"] = "[]";' in exe                                           # attrs==nullptr default branch
    assert "attrs->GetListInt(0)" in exe
    pl = cpp_mod._generate_op_custom_plugin_cpp(
        "EmptyLi", framework_type=_FRAMEWORK_TYPE__ONNX,
        attr_specs=[{"name": "k", "type": "ListInt", "default": []}])
    assert 'for (auto e : attr["ints"]) value.push_back(e.get<int64_t>());' in pl


def test_reg_op_and_executor_negative_and_large_int_default():
    # A negative Int default and a large (>int32) Int default must round-trip as valid C++ int64 literals
    # in the REG_OP ``.ATTR`` line and the executor null-branch fallback.
    neg = {"name": "d", "type": "Int", "default": -5}
    assert ".ATTR(d, Int, -5)" in _reg_line(neg)
    exe_neg = cpp_mod._generate_custom_executor_cpp("NegInt", attr_specs=[neg], **_executor_kwargs())
    assert 'out["d"] = d_ptr ? std::to_string(*d_ptr) : "-5";' in exe_neg
    pl_neg = cpp_mod._generate_op_custom_plugin_cpp(
        "NegInt", framework_type=_FRAMEWORK_TYPE__ONNX, attr_specs=[neg])
    # The ParseParam absent-key fallback is the Int proto zero, protobuf drops the "i" key exactly when
    # the author set 0, so the declared -5 must NOT replace an author-set 0. It rides on the REG_OP
    # ``.ATTR(d, Int, -5)`` literal, which applies when the attribute itself is absent.
    assert ": 0;" in pl_neg
    assert ": -5;" not in pl_neg

    big = {"name": "b", "type": "Int", "default": 10_000_000_000}              # > 2**31, fits int64
    assert ".ATTR(b, Int, 10000000000)" in _reg_line(big)
    exe_big = cpp_mod._generate_custom_executor_cpp("BigInt", attr_specs=[big], **_executor_kwargs())
    assert 'out["b"] = b_ptr ? std::to_string(*b_ptr) : "10000000000";' in exe_big


# GLOBAL declaration-index contract with a REAL ListInt + Int pair (mirrors reference split_v)
def test_extract_attrs_listint_then_int_global_index_contract():
    # split_sizes (ListInt) = declaration-index 0 -> GetListInt(0); dim (Int) = index 1 -> GetInt(1). This is
    # the reference split_v layout. NOTE: the reference leaky_relu reads GetStr(0) for its index-1 attr, a
    # BUG; our generator MUST use the declaration index, so do not "fix" this back to match that.
    two = [{"name": "split_sizes", "type": "ListInt", "default": None},
           {"name": "dim", "type": "Int", "default": 0}]
    exe = cpp_mod._generate_custom_executor_cpp("SplitVLike", attr_specs=two, **_executor_kwargs())
    assert "attrs->GetListInt(0)" in exe
    assert "attrs->GetInt(1)" in exe
    reg = cpp_mod._reg_op_block(kernel_compile_samples.add_infer_shape, kernel_compile_samples.add_infer_dtype,
                                op_type="SplitVLike", dtypes=[torch.float16, torch.float16], attr_specs=two)
    assert reg.index(".REQUIRED_ATTR(split_sizes, ListInt)") < reg.index(".ATTR(dim, Int, 0)")


# infer_shape reads a shape-affecting attr from GetAttrs at its declaration index
def _crop_infer_shape(x_shape: torch.Size, crop_size: list[int]) -> torch.Size:
    out = list(x_shape)
    out[-1] = out[-1] // crop_size[0]
    return torch.Size(out)


def _two_attr_infer_shape(x_shape: torch.Size, crop_size: list[int], dim: int) -> torch.Size:
    return x_shape


def test_parse_infer_shape_accepts_trailing_attr_params_and_maps_global_index():
    specs = [{"name": "crop_size", "type": "ListInt", "default": [2, 2]}]
    meta = cpp_mod._parse_infer_shape_for_codegen(_crop_infer_shape, specs)
    assert meta.param_names == ["x_shape"]                     # only the tensor-shape param (n_in stays tensor count)
    assert meta.attr_read_infos == (("crop_size", 0, "ListInt", [2, 2]),)


def test_parse_infer_shape_two_attrs_global_indices():
    specs = [{"name": "crop_size", "type": "ListInt", "default": None},
             {"name": "dim", "type": "Int", "default": 0}]
    meta = cpp_mod._parse_infer_shape_for_codegen(_two_attr_infer_shape, specs)
    assert meta.param_names == ["x_shape"]
    # global indices come from the AttrSpec order, not the infer_shape position
    assert meta.attr_read_infos == (("crop_size", 0, "ListInt", None), ("dim", 1, "Int", 0))


def test_infer_shape_body_reads_getattrs_listint_at_index():
    specs = [{"name": "crop_size", "type": "ListInt", "default": [2, 2]}]
    meta = cpp_mod._parse_infer_shape_for_codegen(_crop_infer_shape, specs)
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    assert "const gert::RuntimeAttrs *_attrs = context->GetAttrs();" in body
    assert "_attrs->GetListInt(0)" in body                     # ListInt at declaration index 0
    assert "crop_size.assign(" in body
    # the attr local is threaded into the pybind wrapper call after the shape vec
    assert "inferShape(in0_shape_vec, crop_size)" in body


def test_infer_shape_attr_ge_type_accepts_int_listint_and_string_only():
    # The gate behind the InferShape attr read: an annotation it answers None for is refused outright,
    # so a type with no emitter branch can never reach the generated C++.
    ge = cpp_mod._infer_shape_attr_ge_type
    assert (ge(int), ge(list[int]), ge(str), ge(float)) == ("Int", "ListInt", "String", "Float")
    assert (ge(list[float]), ge(typing.Optional[str]), ge(typing.Annotated[int, "doc"])) == (None, None, None)


def test_parse_infer_shape_maps_a_string_attr_to_its_declaration_index():
    # The emitted text is pinned whole by infer_shape_ge_body_string_attr.inc.golden; this pins the
    # metadata the emitter reads, which the golden cannot show.
    meta = cpp_mod._parse_infer_shape_for_codegen(
        infer_shape_samples.infer_shape_width_pick, [{"name": "width", "type": "String", "default": "FULL"}])
    assert meta.param_names == ["x_shape"]
    assert meta.attr_read_infos == (("width", 0, "String", "FULL"),)


def test_parse_infer_shape_maps_a_float_attr_to_its_declaration_index():
    # The Float twin of the String case above; the emitted text is pinned by
    # infer_shape_ge_body_float_attr.inc.golden.
    meta = cpp_mod._parse_infer_shape_for_codegen(
        infer_shape_samples.infer_shape_scaled_last, [{"name": "scale", "type": "Float", "default": 0.5}])
    assert meta.attr_read_infos == (("scale", 0, "Float", 0.5),)


@pytest.mark.parametrize("default,expected", [
    (None, "0.0f"),
    (0.0, "0.0f"),
    (1.0, "1.0f"),
    (-2.0, "-2.0f"),
    (-0.0, "-0.0f"),
    (0.5, "0.5f"),
    (1e-9, "1e-09f"),
], ids=["absent", "zero", "one", "negative", "negative-zero", "fraction", "exponent"])
def test_infer_shape_float_attr_default_emits_a_compilable_literal(default, expected):
    # ``%.9g`` writes an integral Float default without a decimal point, and ``float scale = 1f;`` is
    # not C++: the real package build failed with "unable to find numeric literal operator". The
    # emitted initializer therefore needs a point or an exponent before the ``f``. Nine significant
    # digits are preserved; the generator itself is exercised, not a copy of its formatter.
    meta = cpp_mod._parse_infer_shape_for_codegen(
        infer_shape_samples.infer_shape_scaled_last, [{"name": "scale", "type": "Float", "default": default}])
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    assert f"    float scale = {expected};" in body
    assert "_attrs->GetFloat(0)" in body                       # the read at declaration index 0 stays


def test_parse_infer_shape_rejects_a_string_param_declared_as_another_attr_type():
    with pytest.raises(TypeError, match="is String but its AttrSpec declares Int"):
        cpp_mod._parse_infer_shape_for_codegen(infer_shape_samples.infer_shape_width_pick,
                                               [{"name": "width", "type": "Int", "default": 0}])


def test_parse_infer_shape_rejects_shape_after_attr():
    def bad(x_shape: torch.Size, k: int, y_shape: torch.Size) -> torch.Size:
        return x_shape
    with pytest.raises(TypeError, match="precede"):
        cpp_mod._parse_infer_shape_for_codegen(bad, [{"name": "k", "type": "Int", "default": 0}])


def test_parse_infer_shape_shape_invariant_op_unchanged_no_attr_reads():
    # A shape-invariant-attr op's infer_shape has NO attr param -> attr_read_infos empty; body byte-identical.
    meta = cpp_mod._parse_infer_shape_for_codegen(kernel_compile_samples.add_infer_shape,
                                                  [{"name": "bias", "type": "Int", "default": 0}])
    assert not meta.attr_read_infos
    body = cpp_mod._infer_shape_ge_impl_body(meta)
    assert "GetAttrs()" not in body
    base = cpp_mod._infer_shape_ge_impl_body(
        cpp_mod._parse_infer_shape_for_codegen(kernel_compile_samples.add_infer_shape))
    assert body == base


# nlohmann include-root resolver + effective-include-dirs helper (hermetic)
# _resolve_nlohmann_include_dir tries installed pypto FIRST (on the box, pypto ships
# lib/framework/3rd/include/nlohmann/json.hpp), so every test neutralizes BOTH non-env sources, the
# installed-pypto branch (sys.modules['pypto'].__file__ -> a json-less dir) and codegen._REPO_ROOT
# (-> a json-less dir), so FATAL/form-agnostic assertions pass identically on box and local.


def _neutralize_nonenv_sources(monkeypatch, tmp_path):
    """Point the installed-pypto branch and _REPO_ROOT at json-less dirs so only
    `PYPTO_THIRD_PARTY_PATH` can resolve."""
    pypto_less = tmp_path / "pypto_pkg"
    pypto_less.mkdir()
    stub = _types.ModuleType("pypto")
    stub.__file__ = str(pypto_less / "__init__.py")
    monkeypatch.setitem(_sys.modules, "pypto", stub)
    repo_less = tmp_path / "repo_less"
    repo_less.mkdir()
    monkeypatch.setattr(cpp_mod, "_REPO_ROOT", repo_less)


def test_resolve_nlohmann_include_dir_fatal_when_absent(monkeypatch, tmp_path):
    _neutralize_nonenv_sources(monkeypatch, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PYPTO_THIRD_PARTY_PATH", str(empty))    # holds no nlohmann/json.hpp
    with pytest.raises(FileNotFoundError) as ei:
        cpp_mod._resolve_nlohmann_include_dir()
    msg = str(ei.value)
    assert "PYPTO_THIRD_PARTY_PATH" in msg
    assert "lib/framework/3rd/include" in msg   # installed-pypto knob named
    assert str(empty) in msg                    # probed roots listed


@pytest.mark.parametrize(
    "subpath", ["Release/include", "json-3.11.3/include", "json-3.11.3/single_include"]
)
def test_resolve_nlohmann_include_dir_form_agnostic(monkeypatch, tmp_path, subpath):
    """A multi-header (Release/include, json-3.11.3/include) and a single-header
    (json-3.11.3/single_include) root each resolve identically, the whole point of the include-root
    design. Pins each form via a crafted PYPTO_THIRD_PARTY_PATH root holding ONLY that sub-layout."""
    _neutralize_nonenv_sources(monkeypatch, tmp_path)
    # Craft a third-party root that exposes ONLY <subpath>/nlohmann, so the resolver's fixed candidate
    # order falls through to exactly this form (esp. single_include, reached only when Release/include and
    # json-<ver>/include are both absent under the root). The resolver only tests that
    # <root>/nlohmann/json.hpp IS A FILE, never its contents, so a synthesized header pins the layout
    # without a real third_party tree and the check runs in every environment.
    craft = tmp_path / "tp"
    dest = craft / subpath
    (dest / "nlohmann").mkdir(parents=True)
    (dest / "nlohmann" / "json.hpp").touch()
    monkeypatch.setenv("PYPTO_THIRD_PARTY_PATH", str(craft))
    assert cpp_mod._resolve_nlohmann_include_dir() == dest


def test_embedded_loader_carries_the_package_version_check():
    # The atc-side version gate reaches the .so ONLY as spliced text, and pto_custom_op.cpp fetches a
    # single symbol (load_embedded_compile_module) out of the spliced globals — so a comparator that is
    # present but never CALLED from that entry point is a silent no-op on the box. Both halves pinned.
    loader_src = cpp_mod._BUNDLED_EMBED_LOADER_PATH.read_text(encoding="utf-8")
    assert "def check_pypto_package_version(" in loader_src
    assert "def load_embedded_compile_module(" in loader_src
    # Searched from the loader's own def, so this pins the CALL without pinning the spelling of the
    # argument expression; .index raises when the loader carries no call at all.
    loader_def = loader_src.index("def load_embedded_compile_module(")
    call_index = loader_src.index("check_pypto_package_version(", loader_def)
    # Above the memo read: a cached module must not skip the gate.
    assert call_index < loader_src.index("cached = _loaded.get(key)")

    if not cpp_mod._BUNDLED_PTO_CUSTOM_OP_CPP_PATH.is_file():
        # The shared base .cpp is vendored by a later state than the one that adds the loader; the
        # splice cannot be exercised before it exists.
        pytest.skip("the shared PtoCustomOp base .cpp is not vendored at this state")

    spliced = cpp_mod._bundled_pto_custom_op_cpp_source()
    assert cpp_mod._EMBED_LOADER_PLACEHOLDER not in spliced
    assert "def check_pypto_package_version(" in spliced
    assert "pypto_version.info" in spliced
    # The .so must import no pypto module: the comparator re-reads the metadata itself.
    assert "importlib.metadata.version(\"pypto\")" in spliced
