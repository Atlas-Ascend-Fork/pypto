# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND OF EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ACLNN AICPU offline codegen: source generation for the ops-transformer custom package.

The PyPTO role in the custom-package AICPU build is **configure-time source
generation only**: :func:`prepare_aicpu_kernel_sources` loads one
``@pl.jit(mode="aicpu_kernel")`` kernel, generates the bare CCE body
(``generate_aicpu_bare_body`` -- the same parser/codegen as the JIT fast-launch
path), wraps it in a ``CpuKernel::Compute`` adapter
(:func:`generate_cpu_kernel_sources`: an Args struct assembled from a
``CpuKernelContext`` with ``ctx.Input``/``ctx.Output`` tensor pointers and
``GetAttrValue``/``GetAttrValueOpt`` scalars, registered via
``REGISTER_CPU_KERNEL``) and writes the adapter TU, the Args
header and the opInfo json. Compile, link (``libtransformer_aicpu_kernels.so``
via ``--whole-archive``), json merge (``cust_aicpu_kernel.json``) and install
all happen in the ops-transformer CMake chain (``enable_pypto_aicpu_kernel`` ->
``add_aicpu_cust_kernel_modules`` -> ``gen_cust_aicpu_kernel_symbol`` /
``gen_cust_aicpu_json_symbol``), exactly like handwritten AICPU kernels.

Parameter semantics (aicpu kernel mode):

* ``pl.Tensor`` and ``pl.Ptr`` both represent tensors; the adapter makes no
  optional/required distinction between them. Each one maps to a null-tolerant
  ``ctx.Input``/``ctx.Output`` data-pointer extraction: a missing input packs a
  null pointer, and every presence/emptiness judgment lives in the kernel body
  (``param is None`` guards). The adapter emits no dtype/shape/non-null checks.
* Scalars are optional when the Python signature gives them a default value
  (``GetAttrValueOpt``, default assigned first) and required otherwise
  (``GetAttrValue``; a missing required attr returns ``kStatusParamInvalid``).

AICPU kernels take no string parameters: layout-like choices are plain
``pl.DT_INT32`` attributes (e.g. layout 0=BSND / 1=TND), so one generated
source serves every launch value.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

from pypto.pypto_impl import ir

from ..._errors import InvalidArgument, InvalidVal
from ..aicpu_compile import _impl_name, _parse_aicpu_program, generate_aicpu_bare_body, reject_unsupported_aicpu_params
from ..entry_params import collect_dyn_vars, cpp_type_from_dtype
from ..jit import _get_kernel_ir_function, _setup_arch_env
from ..platform import normalize_arch


def _args_fields(specs, dyn_vars) -> list[tuple[str, str]]:
    """Args struct fields: tensors/pointers as pointers, scalars by value, dynamic dims appended."""
    fields = []
    for spec in specs:
        kind = spec.kind.name
        if kind in ("TENSOR", "PTR"):
            fields.append((f"{cpp_type_from_dtype(spec.dtype, param_name=spec.name)}*", spec.name))
        elif kind == "SCALAR":
            fields.append((cpp_type_from_dtype(spec.dtype, param_name=spec.name), spec.name))
    fields.extend(("int64_t", dyn) for dyn in dyn_vars)
    return fields


def _cann_dtype_name(dtype) -> str:
    """CANN AICPU DataType enum name (cpu_types.h) for a pypto DataType."""
    # The pybind DataType is unhashable, so the mapping keys on str(dtype).
    mapping = {
        "bool": "DT_BOOL",
        "int8": "DT_INT8",
        "int16": "DT_INT16",
        "int32": "DT_INT32",
        "int64": "DT_INT64",
        "uint8": "DT_UINT8",
        "uint16": "DT_UINT16",
        "uint32": "DT_UINT32",
        "uint64": "DT_UINT64",
        "fp16": "DT_FLOAT16",
        "fp32": "DT_FLOAT",
    }
    name = mapping.get(str(dtype))
    if name is None:
        raise ValueError(f"AICPU offline delivery has no CANN dtype mapping for {dtype}")
    return name


def _scalar_attr_getter(dtype) -> str:
    """AttrValue getter family of a scalar dtype (validation oracle for offline specs)."""
    int_family = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
    key = str(dtype)
    if key == "bool":
        return "GetBool"
    if key in int_family:
        return "GetInt"
    if key in ("fp16", "fp32"):
        return "GetFloat"
    raise ValueError(f"AICPU offline delivery has no attr getter for scalar dtype {dtype}")


def _camel(name: str) -> str:
    """snake_case -> CamelCase for generated C++ class symbols."""
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


def _cpp_str_lit(value: str) -> str:
    """C++ string literal (JSON escaping matches for the ASCII values used)."""
    return json.dumps(value)


def _cpp_default_literal(value) -> str | None:
    """Python default literal -> C++ literal text; None when not representable."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return None


def _attr_helper_lines() -> list[str]:
    """Static C++ attr helpers emitted into the adapter TU (aicpu_common.h contract).

    Mirrors the handwritten AICPU ``aicpu_common.h`` helpers so the generated
    adaptation only calls ``GetAttrValue``/``GetAttrValueOpt``: required attrs
    report a miss through the bool return, optional attrs keep the field's
    pre-initialized default. ``static`` gives internal linkage, so the generated
    TU never collides with a repo's own inline helpers in the merged .so.
    """
    return [
        "// Attr marshalling helpers (handwritten AICPU aicpu_common.h contract,",
        "// internal linkage): GetAttrValue reports a missing required attr;",
        "// GetAttrValueOpt keeps the pre-initialized default when the attr is absent.",
        "namespace aicpu {",
        "",
        "template <typename T>",
        "static typename std::enable_if<std::is_integral_v<T>, bool>::type",
        "GetAttrValue(CpuKernelContext& ctx, const std::string& name, T& value)",
        "{",
        "    aicpu::AttrValue* attr = ctx.GetAttr(name);",
        "    if (attr == nullptr) {",
        "        return false;",
        "    }",
        "    value = static_cast<T>(attr->GetInt());",
        "    return true;",
        "}",
        "",
        "static bool GetAttrValue(CpuKernelContext& ctx, const std::string& name, bool& value)",
        "{",
        "    aicpu::AttrValue* attr = ctx.GetAttr(name);",
        "    if (attr == nullptr) {",
        "        return false;",
        "    }",
        "    value = attr->GetBool();",
        "    return true;",
        "}",
        "",
        "static bool GetAttrValue(CpuKernelContext& ctx, const std::string& name, float& value)",
        "{",
        "    aicpu::AttrValue* attr = ctx.GetAttr(name);",
        "    if (attr == nullptr) {",
        "        return false;",
        "    }",
        "    value = attr->GetFloat();",
        "    return true;",
        "}",
        "",
        "template <typename T>",
        "static typename std::enable_if<std::is_integral_v<T>, void>::type",
        "GetAttrValueOpt(CpuKernelContext& ctx, const std::string& name, T& value)",
        "{",
        "    aicpu::AttrValue* attr = ctx.GetAttr(name);",
        "    if (attr != nullptr) {",
        "        value = static_cast<T>(attr->GetInt());",
        "    }",
        "}",
        "",
        "static void GetAttrValueOpt(CpuKernelContext& ctx, const std::string& name, bool& value)",
        "{",
        "    aicpu::AttrValue* attr = ctx.GetAttr(name);",
        "    if (attr != nullptr) {",
        "        value = attr->GetBool();",
        "    }",
        "}",
        "",
        "static void GetAttrValueOpt(CpuKernelContext& ctx, const std::string& name, float& value)",
        "{",
        "    aicpu::AttrValue* attr = ctx.GetAttr(name);",
        "    if (attr != nullptr) {",
        "        value = attr->GetFloat();",
        "    }",
        "}",
        "",
        "} // namespace aicpu",
        "",
    ]


def generate_cpu_kernel_sources(
    cce_body, param_specs, op_name, op_type, param_names, defaults, kernel_so: str = "libtransformer_aicpu_kernels.so"
) -> dict[str, str]:
    """Generate the offline CpuKernel delivery sources: args header + adapter TU + op json.

    The kernel body is the same CCE output the fast-launch path compiles; this
    variant wraps it in a ``CpuKernel::Compute`` adapter that assembles an
    Args struct from a ``CpuKernelContext``. The ctx adaptation is marshalling
    only -- ``ctx.Input``/``ctx.Output`` for every tensor parameter (``pl.Tensor``
    and ``pl.Ptr`` alike, null-tolerant data-pointer extraction, no dtype/shape
    or non-null judgments), ``GetAttrValue`` for required scalars (no default)
    and ``GetAttrValueOpt`` for optional scalars (Python default pre-assigned).
    All presence/emptiness/value checks live in the kernel body and travel the
    AICPU status ABI: the impl returns uint32_t (0 = OK, 1 = PARAM_INVALID;
    an early bare ``return`` is a failed check) and the adapter maps it onto
    the CpuKernel status channel. ``kernel_so`` names the
    target library in the opInfo json -- the ops-transformer custom package
    merges all AICPU objects into one ``libtransformer_aicpu_kernels.so``.
    """
    reject_unsupported_aicpu_params(param_specs, "offline delivery", param_names)

    dyn_vars = collect_dyn_vars(param_specs)
    fields = _args_fields(param_specs, dyn_vars)
    forward = [f"args.{name}" for _typ, name in fields]
    impl_name = _impl_name(op_name)
    op_cls = _camel(op_name)

    fill_lines: list[str] = []
    dyn_source: dict[str, tuple[str, int]] = {}
    in_idx = out_idx = 0
    for spec in param_specs:
        kind = spec.kind.name
        if kind not in ("TENSOR", "PTR"):
            continue
        if kind == "TENSOR" and spec.direction is not None and spec.direction.name == "OUTPUT":
            local = f"t_out{out_idx}"
            fill_lines.append(f"    aicpu::Tensor* {local} = ctx.Output({out_idx});")
            out_idx += 1
        else:
            local = f"t_in{in_idx}"
            fill_lines.append(f"    aicpu::Tensor* {local} = ctx.Input({in_idx});")
            in_idx += 1
        # Both kinds carry the same optional-input edge semantics: an absent
        # tensor packs a null pointer, and the kernel body's ``is not None``
        # guards own the presence judgment. The pointee dtype is nominal -- the
        # body may reinterpret the bytes via pl.make_tensor -- so only the data
        # pointer is taken, with no dtype/shape/non-null checks here.
        cpp_t = cpp_type_from_dtype(spec.dtype, param_name=spec.name)
        fill_lines.append(
            f"    args.{spec.name} = ({local} != nullptr) ? static_cast<{cpp_t}*>({local}->GetData()) : nullptr;"
        )
        for axis, dim in enumerate(spec.shape or []):
            if isinstance(dim, str) and dim not in dyn_source:
                dyn_source[dim] = (local, axis)

    # Dynamic-dim tail of the launch ABI: a present tensor contributes its dim,
    # an absent one packs 0 (the fast-launch None packing keeps the same ABI).
    for dyn in dyn_vars:
        local, axis = dyn_source[dyn]
        fill_lines.append(
            f"    args.{dyn} = ({local} != nullptr) "
            f"? static_cast<int64_t>({local}->GetTensorShape()->GetDimSize({axis})) : 0;"
        )

    for spec, orig in zip(param_specs, param_names):
        kind = spec.kind.name
        if kind != "SCALAR":
            continue
        cpp_t = cpp_type_from_dtype(spec.dtype, param_name=spec.name)
        fallback = _cpp_default_literal(defaults.get(orig))
        if fallback is not None:
            # Optional attr: pre-assign the Python default, then let a present
            # attr override it (GetAttrValueOpt keeps the default when absent).
            fill_lines.append(f"    args.{spec.name} = static_cast<{cpp_t}>({fallback});")
            fill_lines.append(f'    GetAttrValueOpt(ctx, "{orig}", args.{spec.name});')
        else:
            fill_lines.append(
                f'    if (!GetAttrValue(ctx, "{orig}", args.{spec.name})) {{ return kStatusParamInvalid; }}'
            )
    has_scalars = any(spec.kind.name == "SCALAR" for spec in param_specs)

    # ── Args header: the offline adapter's own struct (minimal includes, no namespace imports) ──
    args_header = [
        "// Generated by PyPTO Pro AICPU offline glue: CpuKernel adapter Args struct.",
        "#pragma once",
        "#include <cstdint>",
        "",
        f"struct {op_name}Args {{",
    ]
    args_header.extend(f"    {typ} {name};" for typ, name in fields)
    args_header.extend(["};", ""])

    # ── Kernel TU: CCE body + CpuKernel adapter + registration ──
    kernel_source = [
        "// Generated by PyPTO Pro AICPU offline: CCE body + CpuKernel adapter + registration.",
        '#include "cpu_kernel.h"',
        f'#include "{op_name}_aicpu_args.h"',
        "#include <algorithm>",
        "#include <string>",
    ]
    if has_scalars:
        kernel_source.append("#include <type_traits>")
    kernel_source.extend(
        [
            "// AICPU runtime status channel: 0 = KERNEL_STATUS_OK, 1 = KERNEL_STATUS_PARAM_INVALID.",
            "constexpr uint32_t kStatusOk = 0;",
            "constexpr uint32_t kStatusParamInvalid = 1;",
            "// The CCE body calls min/max/abs unqualified; targeted imports keep the TU clean.",
            "using std::abs;",
            "using std::max;",
            "using std::min;",
            "",
        ]
    )
    if has_scalars:
        kernel_source.extend(_attr_helper_lines())
    kernel_source.extend(
        [
            "// Kernel body generated by the shared CCE codegen (same source as fast-launch).",
            cce_body,
            "",
            "namespace aicpu {",
            "",
            f"class {op_cls}CpuKernel final : public CpuKernel {{",
            " public:",
            "  uint32_t Compute(CpuKernelContext& ctx) override {",
            f"    {op_name}Args args{{}};",
        ]
    )
    kernel_source.extend(fill_lines)
    kernel_source.append(f"    uint32_t status = {impl_name}(")
    kernel_source.extend(f"        {expr}," for expr in forward[:-1])
    if forward:
        kernel_source.append(f"        {forward[-1]}")
    kernel_source.extend(
        [
            "    );",
            "    // AICPU status ABI: the kernel's early returns are failed checks;",
            "    // map the impl's KERNEL_STATUS onto the CpuKernel status channel",
            "    // (1 = KERNEL_STATUS_PARAM_INVALID) so the op caller receives it.",
            "    return status == 0 ? kStatusOk : kStatusParamInvalid;",
            "  }",
            "};",
            "",
            "// Internal linkage: the macro pastes the `kernelType` token into externally visible symbols.",
            "namespace {",
            f"static const char* kernelType = {_cpp_str_lit(op_type)};",
            f"REGISTER_CPU_KERNEL(kernelType, {op_cls}CpuKernel);",
            "} // namespace",
            "",
            "} // namespace aicpu",
            "",
        ]
    )

    # ── op json (schema of a handwritten AICPU opInfo entry) ──
    op_entry = {
        op_type: {
            "opInfo": {
                "computeCost": "100",
                "engine": "DNN_VM_AICPU",
                "flagAsync": "False",
                "flagPartial": "False",
                "functionName": "RunCpuKernel",
                "kernelSo": kernel_so,
                "opKernelLib": "CUSTAICPUKernel",
                "userDefined": "True",
                "workspaceSize": "100",
            }
        }
    }

    return {
        "args_header": "\n".join(args_header),
        "kernel_source": "\n".join(kernel_source),
        "op_json": json.dumps(op_entry, indent=4) + "\n",
        "op_entry": op_entry,
    }


@dataclasses.dataclass
class AicpuKernelSources:
    """Generated sources of one AICPU specialization (compile happens in CMake)."""

    op_type: str
    kernel_name: str
    args_header_path: str
    kernel_cpp_path: str
    json_path: str
    build_dir: str
    op_entry: dict


def _literal_defaults(func) -> dict:
    """Map of parameter name -> literal default (bool/int/float only; matches _cpp_default_literal)."""
    out: dict = {}
    for name, param in inspect.signature(func).parameters.items():
        if param.default is inspect.Parameter.empty:
            continue
        if isinstance(param.default, (bool, int, float)):
            out[name] = param.default
    return out


def _validate_offline_specs(param_specs) -> None:
    """Fail fast on parameter forms the CpuKernel adapter cannot express."""
    reject_unsupported_aicpu_params(param_specs, "offline delivery")
    for spec in param_specs:
        if spec.kind.name == "TENSOR":
            _cann_dtype_name(spec.dtype)
        elif spec.kind.name == "SCALAR":
            _scalar_attr_getter(spec.dtype)


def _prepare_from_kernel(kernel, arch: str, op_type: str | None, out_dir, kernel_so: str) -> AicpuKernelSources:
    """Generate one offline specialization from an already-loaded JIT kernel."""
    if kernel._func_type != ir.FunctionType.AicpuKernel:
        raise InvalidVal(f"kernel '{kernel.__name__}' is not mode='aicpu_kernel' (got func_type={kernel._func_type})")
    if getattr(kernel, "_tilingkey_schema", None) is not None:
        raise InvalidVal(f"AICPU offline v1 does not support tiling_key on kernel '{kernel.__name__}'")

    arch_npu = _setup_arch_env(normalize_arch(arch))

    kernel_def, program, param_specs = _parse_aicpu_program(kernel)
    op_name = _get_kernel_ir_function(program).name
    op_type = op_type or op_name
    _validate_offline_specs(param_specs)

    param_names = list(inspect.signature(kernel._func).parameters)
    defaults = _literal_defaults(kernel._func)

    cce_body = generate_aicpu_bare_body(program, arch_npu, op_name)
    sources = generate_cpu_kernel_sources(
        cce_body,
        param_specs,
        op_name,
        op_type,
        param_names,
        defaults,
        kernel_so=kernel_so,
    )

    out_dir = Path(out_dir) if out_dir else Path.cwd() / "build" / "aicpu_gen" / op_name
    out_dir.mkdir(parents=True, exist_ok=True)
    header_path = out_dir / f"{op_name}_aicpu_args.h"
    cpp_path = out_dir / f"{op_name}_aicpu_kernel.cpp"
    json_path = out_dir / f"{op_name}_aicpu.json"
    header_path.write_text(sources["args_header"], encoding="utf-8")
    cpp_path.write_text(sources["kernel_source"], encoding="utf-8")
    json_path.write_text(sources["op_json"], encoding="utf-8")
    return AicpuKernelSources(
        op_type=op_type,
        kernel_name=op_name,
        args_header_path=str(header_path),
        kernel_cpp_path=str(cpp_path),
        json_path=str(json_path),
        build_dir=str(out_dir),
        op_entry=sources["op_entry"],
    )


def prepare_aicpu_kernel_sources(
    op_path: str,
    arch: str = "3510",
    op_type: str | None = None,
    out_dir: str | None = None,
    kernel_so: str = "libtransformer_aicpu_kernels.so",
) -> AicpuKernelSources:
    """Load a DSL file with one ``@pl.jit(mode="aicpu_kernel")`` kernel and generate its sources.

    Entry for the ops-transformer custom-package chain: the ``pypto_aicpu_codegen_cli`` script
    calls this at CMake configure time (via ``enable_pypto_aicpu_kernel``); the returned
    ``<op>_aicpu_kernel.cpp`` joins ``add_aicpu_cust_kernel_modules`` objects and the json
    joins ``AICPU_JSON_FILES`` for ``gen_cust_aicpu_kernel_symbol``/``gen_cust_aicpu_json_symbol``.
    """
    from .pypto_compile import _load_kernel

    if not Path(op_path).is_file():
        raise InvalidArgument(f"kernel file not found: '{op_path}'")
    kernel = _load_kernel(op_path)
    return _prepare_from_kernel(kernel, arch, op_type, out_dir, kernel_so)
