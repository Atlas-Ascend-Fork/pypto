# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Adapt per-key PyPTO implementations to the native AscendC ACLNN compiler.

AscendC owns Host Tiling, static specialization, external parameter slots, normal/SK
variants, binding, linking and metadata. PyPTO supplies a template-compatible source
whose implementation consumes the native static Tiling macros.
"""

import importlib
import inspect
from pathlib import Path

from asc_op_compile_base.asc_op_compiler.ascendc_common_utility import CommonUtility
from asc_op_compile_base.asc_op_compiler.global_storage import global_var_storage
from asc_op_compile_base.common.buildcfg import get_current_build_config
from asc_op_compile_base.common.context import op_context

from pypto_pro._errors import InvalidArgument, NotSupported, RuntimeFailure
from pypto_pro.runtime.jit import ParamKind, _codegen
from pypto_pro.runtime.platform import NpuArch

from .pypto_compile import (
    _binary_tiling_parameter,
    _filter_tiling_keys,
    _kernel_type_from_codegen,
    _normalize_dfx_op_info,
    _op_info_get,
    _write_tilingkey_header,
)


def _validate_sk_request(kernel, context) -> bool:
    requested = bool(get_current_build_config("enable_super_kernel"))
    if not requested:
        return False
    if not global_var_storage.get_variable("ascendc_enable_super_kernel"):
        raise NotSupported("ACLNN SuperKernel compilation is not supported by the selected CANN target")
    if context.get_op_mode() not in {"static", "prestatic"}:
        raise NotSupported("PyPTO ACLNN SuperKernel compilation requires static operator inputs")
    if context.get_addition("super_kernel_sub_combine") is not True:
        raise NotSupported("PyPTO ACLNN SuperKernel requires the normal/sub-kernel combined compile workflow")
    native = importlib.import_module("asc_op_compile_base.asc_op_compiler.compile_op")
    if getattr(native, "gen_sk_bind_source", None) is None:
        raise NotSupported("ACLNN SuperKernel requires CANN with macro-based SK binding capabilities")
    return True


def _validate_entry(cg, op_info):
    """The native wrapper fills absent optional arguments with nullptr by position."""
    inputs = _op_info_get(op_info, "origin_inputs") or []
    outputs = _op_info_get(op_info, "outputs") or []
    if _op_info_get(op_info, "mc2_ctx") or _op_info_get(op_info, "output_shape_depend_on_compute"):
        raise NotSupported("PyPTO ACLNN recompilation does not yet support MC2 or computed output shapes")
    if any(isinstance(value, (list, tuple)) for value in [*inputs, *outputs]):
        raise NotSupported("PyPTO ACLNN recompilation does not yet support dynamic tensor lists")
    expected = len(inputs) + len(outputs) + 2
    if len(cg.entry_params) != expected or len(cg.param_specs) != expected:
        raise InvalidArgument(
            f"ACLNN expects {expected} parameters (inputs, outputs, workspace, tiling); "
            f"kernel '{cg.kernel_name}' has {len(cg.entry_params)} physical parameters"
        )
    if any(spec.kind not in {ParamKind.PTR, ParamKind.TENSOR} for spec in cg.param_specs[:-1]):
        raise NotSupported("ACLNN data and workspace parameters must be pointers or tensors without dynamic ABI fields")
    if cg.param_specs[-1].kind != ParamKind.TILING:
        raise InvalidArgument("ACLNN kernel must end with its declared Tiling parameter")
    if not all(is_pointer for _, _, is_pointer in cg.entry_params):
        raise NotSupported("ACLNN recompilation does not support scalar or dynamic-dimension ABI parameters")


def _validate_static_inputs(op_info, extend_options):
    from asc_op_compile_base.asc_op_compiler.get_op_tiling import is_static_shape

    inputs = _op_info_get(op_info, "origin_inputs") or []
    outputs = _op_info_get(op_info, "outputs") or []
    if any(isinstance(value, (list, tuple)) for value in [*inputs, *outputs]):
        raise NotSupported("PyPTO ACLNN recompilation does not yet support dynamic tensor lists")
    if not is_static_shape(
        inputs, outputs, extend_options.get("valueDepend"), _op_info_get(op_info, "param_type_list"), enable_vd=True
    ):
        raise NotSupported("PyPTO ACLNN recompilation requires concrete shapes and resolved value dependencies")


def _native_entry(cg, kernel_type):
    params = ", ".join(f"GM_ADDR {name}" for _, name, _ in cg.entry_params)
    arguments = ", ".join(f"(__gm__ {typ}*){name}" for typ, name, _ in cg.entry_params)
    return (
        "template <uint64_t... PyptoKeys>\n"
        f"__global__ AICORE void {cg.kernel_name}({params})\n"
        "{\n"
        f"    KERNEL_TASK_TYPE_DEFAULT({kernel_type.name});\n"
        f"    {cg.kernel_name}_impl({arguments});\n"
        "}\n"
    )


def _sync_host_tiling_platform():
    # asc_opc configures the AscendC platform API, while libregister Host Tiling
    # still reads the TBE platform state in CANN 9.3. Importing TBE initializes
    # that state to a default SoC; preserve the actual OPC target across import.
    from asc_op_compile_base.common.platform import get_soc_spec

    soc = get_soc_spec("FULL_SOC_VERSION")
    core_type = get_soc_spec("AICORE_TYPE")
    core_num = get_soc_spec("CORE_NUM")
    from tbe.common.platform import set_current_compile_soc_info

    set_current_compile_soc_info(soc, core_type, core_num)


def compile_aclnn_kernel(kernel, op_info, options, op_compile_option, extend_options, arch, dtype_consts):
    """Generate native input sources and run the complete static/combined compiler."""
    from asc_op_compile_base.asc_op_compiler.compile_op import compile_op

    if arch != NpuArch.DAV_3510:
        raise NotSupported("PyPTO ACLNN static recompilation currently supports Ascend950 (a5)")
    context = op_context.get_context()
    _validate_sk_request(kernel, context)
    _validate_static_inputs(op_info, extend_options)
    kernel_name = _op_info_get(op_info, "kernel_name")
    schema = kernel.tilingkey_schema
    _, tiling_class = _binary_tiling_parameter(kernel)
    # IR parameter names are uniquified (e.g. workspace_0). Validate source roles
    # before lowering, then use positions throughout the native ABI adapter.
    parameter_names = list(inspect.signature(kernel._func).parameters)
    if len(parameter_names) < 2 or parameter_names[-2] != "workspace":
        raise InvalidArgument("ACLNN kernel must place 'workspace' immediately before its Tiling parameter")
    combinations = {schema.pack(dict(zip(schema.field_names(), values))): values for values in schema.enumerate_valid()}
    keys = _filter_tiling_keys(list(combinations), extend_options, context, kernel_name)
    if not keys:
        raise InvalidArgument(f"No legal TilingKey for '{kernel_name}'")

    root = Path(CommonUtility.get_kernel_meta_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_tilingkey_header(schema, kernel.__name__, str(root))
    source = ['#include "pypto_block_meta.h"\n', f'#include "{schema.cls_name}_tilingkey.h"\n']
    expected_type = None
    for index, packed in enumerate(keys):
        packed = int(packed)
        key = dict(zip(schema.field_names(), combinations[packed]))
        cg = _codegen(
            kernel.to_kernel_def(key, dtype_consts), arch, clean_up=False, tilingkey_packed=packed,
            out_dir=str(root / f"_cg_aclnn_{packed}"),
        )
        if cg is None:
            raise RuntimeFailure(f"ACLNN codegen failed for TilingKey {packed}")
        _validate_entry(cg, op_info)
        kernel_type = _kernel_type_from_codegen(cg, arch)
        if expected_type is not None and kernel_type != expected_type:
            raise NotSupported("PyPTO ACLNN recompilation requires one kernel type across TilingKeys")
        expected_type = kernel_type
        condition = f"TILING_KEY_VAR == {packed}UL"
        if index == 0:
            # kernel_utils_macros.h always defines TILING_KEY_VAR (as g_tilingKey) when the
            # compiler passes no literal, so the precompile pass has no matching key and
            # would hit the #error below. ASCENDC_TPL_PRE is only defined by that pass.
            condition = "defined(ASCENDC_TPL_PRE) || " + condition
        source.append(f"{'#if' if index == 0 else '#elif'} {condition}\n")
        source.append(f'#include "_cg_aclnn_{packed}/kernel.cpp"\n')
        source.append(f"REGISTER_TILING_DEFAULT({tiling_class.__name__});\n")
        source.append(_native_entry(cg, kernel_type))
    source.append('#else\n#error "Unsupported PyPTO TilingKey"\n#endif\n')
    source_path = root / f"{kernel_name}_pypto_aclnn.cpp"
    source_path.write_text("".join(source), encoding="utf-8")
    native_options = [*options, "-D__PYPTO_ACLNN__"]
    _sync_host_tiling_platform()
    compile_op(
        str(source_path), kernel.__name__, _normalize_dfx_op_info(op_info), native_options,
        -1, op_compile_option, extend_options,
    )
    json_path = root / f"{kernel_name}.json"
    if not json_path.is_file() or not json_path.with_suffix(".o").is_file():
        raise RuntimeFailure(f"ACLNN compilation did not produce the binary and metadata for '{kernel_name}'")
    context.add_build_res("json_file_path", str(json_path))
