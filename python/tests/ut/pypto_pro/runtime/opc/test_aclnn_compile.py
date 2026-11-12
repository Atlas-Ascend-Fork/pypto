# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""External ABI and capability checks for ACLNN static recompilation."""

import pathlib
from types import SimpleNamespace

from asc_op_compile_base.asc_op_compiler.ascendc_constants import KernelMetaType
import pypto_pro
from pypto_pro._errors import InvalidArgument, NotSupported
from pypto_pro.runtime.jit import ParamKind
from pypto_pro.runtime.opc import aclnn_compile
import pytest


def _entry():
    names = ["x", "optional_x", "y", "workspace", "tiling"]
    return SimpleNamespace(
        kernel_name="probe",
        entry_params=[("uint8_t", name, True) for name in names],
        param_specs=[SimpleNamespace(name=name, kind=ParamKind.PTR) for name in names[:-1]]
        + [SimpleNamespace(name="tiling", kind=ParamKind.TILING)],
    )


def _op_info():
    return {"origin_inputs": [{"param_name": "x"}, None], "outputs": [{"param_name": "y"}]}


def test_optional_input_retains_internal_slot_for_native_null_binding():
    cg = _entry()
    aclnn_compile._validate_entry(cg, _op_info())
    source = aclnn_compile._native_entry(cg, KernelMetaType.KERNEL_TYPE_AIV_ONLY)
    assert "GM_ADDR optional_x" in source
    assert "(__gm__ uint8_t*)optional_x" in source
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);" in source
    assert "GetUserWorkspace" not in source  # Native wrapper already converts it.


def test_dynamic_dimension_cannot_shift_workspace_or_tiling_slots():
    cg = _entry()
    cg.entry_params.insert(1, ("int64_t", "dynamic_n", False))
    with pytest.raises(InvalidArgument, match="physical parameters"):
        aclnn_compile._validate_entry(cg, _op_info())


def test_lowering_may_rename_parameters_without_changing_the_abi():
    cg = _entry()
    cg.param_specs[-2].name = "workspace_0"
    cg.param_specs[-1].name = "tiling_0"
    aclnn_compile._validate_entry(cg, _op_info())


@pytest.mark.parametrize("field", ["mc2_ctx", "output_shape_depend_on_compute"])
def test_unsupported_system_arguments_fail_before_compilation(field):
    info = _op_info()
    info[field] = [0]
    with pytest.raises(NotSupported, match="MC2 or computed output shapes"):
        aclnn_compile._validate_entry(_entry(), info)


def _sk_context(mode="static", combined=True):
    return SimpleNamespace(get_op_mode=lambda: mode, get_addition=lambda name: combined)


def _enable_sk(monkeypatch):
    monkeypatch.setattr(aclnn_compile, "get_current_build_config", lambda name: True)
    monkeypatch.setattr(aclnn_compile.global_var_storage, "get_variable", lambda name: True)
    native = SimpleNamespace(gen_sk_bind_source=lambda *args: b"")
    monkeypatch.setattr(aclnn_compile.importlib, "import_module", lambda name: native)


def test_sk_request_accepts_missing_sk_cap(monkeypatch):
    _enable_sk(monkeypatch)
    assert aclnn_compile._validate_sk_request(SimpleNamespace(_sk_cap=None), _sk_context()) is True


@pytest.mark.parametrize("cap", [1, 2, 4, 8, 16, 7])
def test_sk_request_ignores_sk_cap_bits(monkeypatch, cap):
    _enable_sk(monkeypatch)
    assert aclnn_compile._validate_sk_request(SimpleNamespace(_sk_cap=cap), _sk_context()) is True


def test_ordinary_static_compile_does_not_require_sk_capability(monkeypatch):
    monkeypatch.setattr(aclnn_compile, "get_current_build_config", lambda name: False)
    assert not aclnn_compile._validate_sk_request(SimpleNamespace(_sk_cap=None), None)


def test_sk_request_rejects_unsupported_target(monkeypatch):
    monkeypatch.setattr(aclnn_compile, "get_current_build_config", lambda name: True)
    monkeypatch.setattr(aclnn_compile.global_var_storage, "get_variable", lambda name: False)
    with pytest.raises(NotSupported, match="selected CANN target"):
        aclnn_compile._validate_sk_request(SimpleNamespace(_sk_cap=None), None)


@pytest.mark.parametrize("mode,combined,message", [
    ("dynamic", True, "static operator inputs"),
    ("static", False, "combined compile workflow"),
])
def test_sk_requires_static_combined_workflow(monkeypatch, mode, combined, message):
    monkeypatch.setattr(aclnn_compile, "get_current_build_config", lambda name: True)
    monkeypatch.setattr(aclnn_compile.global_var_storage, "get_variable", lambda name: True)
    context = SimpleNamespace(get_op_mode=lambda: mode, get_addition=lambda name: combined)
    with pytest.raises(NotSupported, match=message):
        aclnn_compile._validate_sk_request(SimpleNamespace(_sk_cap=None), context)


@pytest.mark.parametrize("native_binding", [False, True])
def test_sk_requires_native_capability_binding(monkeypatch, native_binding):
    monkeypatch.setattr(aclnn_compile, "get_current_build_config", lambda name: True)
    monkeypatch.setattr(aclnn_compile.global_var_storage, "get_variable", lambda name: True)
    native = SimpleNamespace()
    if native_binding:
        native.gen_sk_bind_source = lambda *args: b""
    monkeypatch.setattr(aclnn_compile.importlib, "import_module", lambda name: native)
    context = SimpleNamespace(get_op_mode=lambda: "static", get_addition=lambda name: True)
    kernel = SimpleNamespace(_sk_cap=None)
    if native_binding:
        assert aclnn_compile._validate_sk_request(kernel, context)
    else:
        with pytest.raises(NotSupported, match="macro-based SK binding"):
            aclnn_compile._validate_sk_request(kernel, context)


@pytest.mark.parametrize("shape", [[-1, 128], [-2]])
def test_static_context_rejects_unknown_shapes(shape):
    info = {"origin_inputs": [{"shape": shape}], "outputs": [{"shape": [128]}]}
    with pytest.raises(NotSupported, match="concrete shapes"):
        aclnn_compile._validate_static_inputs(info, {})


def test_static_input_validation_keeps_absent_optional_slots():
    info = {"origin_inputs": [{"shape": [128]}, None], "outputs": [{"shape": [128]}]}
    aclnn_compile._validate_static_inputs(info, {})


def test_static_input_validation_rejects_unresolved_value_dependency():
    info = {"origin_inputs": [{"shape": [1]}], "outputs": [{"shape": [128]}], "param_type_list": ["optional"]}
    with pytest.raises(NotSupported, match="resolved value dependencies"):
        aclnn_compile._validate_static_inputs(info, {"valueDepend": {0: "optional"}})
    info["origin_inputs"][0]["const_value"] = [128]
    aclnn_compile._validate_static_inputs(info, {"valueDepend": {0: "optional"}})


def test_host_tiling_uses_full_soc_and_captured_core_limits(monkeypatch):
    from asc_op_compile_base.common import platform as native_platform
    from tbe.common import platform as tiling_platform

    target = {"FULL_SOC_VERSION": "Ascend950DT_9592", "AICORE_TYPE": "AiCore", "CORE_NUM": 24}
    monkeypatch.setattr(native_platform, "get_soc_spec", target.__getitem__)
    received = []
    monkeypatch.setattr(tiling_platform, "set_current_compile_soc_info", lambda *args: received.append(args))
    aclnn_compile._sync_host_tiling_platform()
    assert received == [("Ascend950DT_9592", "AiCore", 24)]


def test_aclnn_compat_is_merged_into_block_meta():
    include_dir = pathlib.Path(pypto_pro.__file__).parent / "include"
    assert not (include_dir / "pypto_aclnn_compat.h").exists()
    text = (include_dir / "pypto_block_meta.h").read_text()
    assert "PYPTO_ACLNN_CHECK_TILING_SIZE" in text
    assert "#include <pto/pto-inst.hpp>" in text
    assert '#pragma push_macro("get_block_num")' in text
