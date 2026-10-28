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
# Put the repo's tools/ dir on sys.path so the `exported_custom_op_litenpu.*` import below resolves when this
# module is run as a script rather than imported.
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # tools

from collections import namedtuple as _namedtuple  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import inspect  # noqa: E402
import json as _json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
from typing import Any, Callable, get_args, get_origin, get_type_hints  # noqa: E402

import torch  # noqa: E402

from exported_custom_op_litenpu.deploy.cpp_naming import camel_case_to_snake_case  # noqa: E402
from pypto.extensions.torch_custom_op_litenpu.common.attr_spec import ATTR_NAME_RE, AttrSpec  # noqa: E402
from pypto.extensions.torch_custom_op_litenpu.common.authoring import validate_op_type_identifier  # noqa: E402
from pypto.extensions.torch_custom_op_litenpu.common.kernel_snippet import (  # noqa: E402
    build_kernel_compile_snippet,
)
from pypto.extensions.torch_custom_op_litenpu.common.return_annotations import (  # noqa: E402
    parse_output_arity,
)

# The names of the embedded pybind wrappers the generated C++ calls to reach the two infer hooks.
_CPP_BIND__INFER_SHAPE = "inferShape"
_CPP_BIND__INFER_DTYPE = "inferDtype"


# Operator-attribute (AttrSpec) codegen: the single ordered attr list drives every emission.
#
# A declared compute attr reaches the C++ codegen as a plain ``{"name","type","default"}`` dict
# (reconstructed from ``op_export_record``) or as a live ``AttrSpec`` object (structure UTs). It is
# normalized to ``_AttrSpecCG`` and drives THREE coordinated emissions from ONE ordered list, so the
# REG_OP declaration order == the ExtractAttrs positional index == the InferShape index:
#   * REG_OP  `.REQUIRED_ATTR(name, T)` / `.ATTR(name, T, <default>)`  (``_reg_op_block``)
#   * onnx plugin ParseParam  `attr["type"]==<code>` + `op_dest.SetAttr(...)`  (``_generate_op_custom_plugin_cpp``)
#   * executor `ExtractAttrs`  `GetXxx(<index>)`  (``_custom_executor_class_cpp``)
# Type tables cover all four declared kinds. Int/Float/String/ListInt are ALL emitted; the per-type
# emission bodies are driven off these tables so a new kind is a table + body-branch addition.

_AttrSpecCG = _namedtuple("_AttrSpecCG", ["name", "type", "default"])

# type -> ONNX AttributeProto type code matched in ParseParam.
_ATTR_ONNX_TYPE_CODE = {"Float": 1, "Int": 2, "String": 3, "ListInt": 7}
# type -> gert::RuntimeAttrs positional accessor used in ExtractAttrs / InferShape.
_ATTR_RT_ACCESSOR = {"Int": "GetInt", "Float": "GetFloat", "String": "GetStr", "ListInt": "GetListInt"}

# C++ keywords, including the alternative tokens. An attr name is spliced into the generated REG_OP block
# and into local declarations, so a keyword there produces a translation unit that cannot compile. The C++20
# contextual keywords (module, import, requires, concept) are legal identifiers and deliberately absent.
_CPP_KEYWORDS = frozenset("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class
compl const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default
delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long
mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public register
reinterpret_cast return short signed sizeof static static_assert static_cast struct switch template this
thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while
xor xor_eq
""".split())

# Names the generated InferShape body already declares in the scope the attr locals are emitted into. An attr
# called `_attrs` emits `int64_t _attrs = 0; if (_attrs != nullptr)`, shadowing the pointer it then
# dereferences. `context` is the GE API parameter, so renaming our own locals could not avoid that one.
_EMITTED_SCOPE_NAMES = frozenset({"_attrs", "context"})
_EMITTED_SCOPE_RE = re.compile(r"^(in\d+_shape(_vec)?|out_shape(_vec|_tuple|_\d+)?)$")

# Names the generated pybind wrapper declares alongside its own parameters. EVERY helper parameter
# becomes a wrapper parameter, so these bind for tensor-shape and dtype params too. The wrapper also
# declares ``<helper function name>_py``, which only a self-referential param name could hit.
_WRAPPER_SCOPE_NAMES = frozenset({
    "globals", "gil", "_m", "_mod", "_ret", "_torch_Size", "_ge_dt_to_torch", "_torch_dt_to_ge",
    "_dtype_bridge_src", "_import_by_path_src", "_pypto_py_bootstrapped",
})


def _validate_emitted_identifier(name, what: str, *, infer_body_scope: bool = True) -> None:
    """Reject a *name* the generated C++ could not carry as an identifier.

    *what* names the rejected thing in the message (e.g. ``"attr name"``). This runs at the code
    generator's own boundary rather than only in ``AttrSpec.__post_init__``, because the deploy path
    supplies plain dicts read back from ``op_export_record`` and Python helper signatures whose parameter
    names are spliced verbatim into the emitted C++ -- neither reaches the authoring-side checks.

    *infer_body_scope* additionally bars names the InferShape body declares; only attr names land there.
    """
    if not isinstance(name, str) or not ATTR_NAME_RE.match(name):
        raise ValueError(
            f"{what} must be an identifier matching {ATTR_NAME_RE.pattern}, got {name!r}"
        )
    if name in _CPP_KEYWORDS:
        raise ValueError(f"{what} {name!r} is a C++ keyword and cannot be emitted as an identifier")
    if name in _WRAPPER_SCOPE_NAMES:
        raise ValueError(
            f"{what} {name!r} collides with a local the generated pybind wrapper declares, "
            f"so the emitted C++ would redeclare it"
        )
    if infer_body_scope and (name in _EMITTED_SCOPE_NAMES or _EMITTED_SCOPE_RE.match(name)):
        raise ValueError(
            f"{what} {name!r} collides with a local the generated inference body already declares, "
            f"so the emitted C++ would shadow it"
        )


def _normalize_attr_specs(attr_specs) -> list:
    """Normalize a mixed list of ``AttrSpec`` objects / ``{"name","type","default"}`` dicts to
    ``_AttrSpecCG`` (name/type/default). ``None``/empty -> ``[]``.

    Every spec is validated at this boundary: the name against what the emitted C++ can carry
    (identifier, non-keyword, no collision with an emitted local), then the whole spec through
    ``AttrSpec`` so the per-type default rules (an Int default that is really a float, a non-finite
    Float, a String default carrying a quote or newline, a ListInt default that is not a list of ints)
    reject here rather than reaching the C++ emitters.
    """
    if not attr_specs:
        return []
    out = []
    for s in attr_specs:
        name, type_, default = s["name"], s["type"], s.get("default")
        _validate_emitted_identifier(name, "attr name")
        spec = AttrSpec(name, type_, default)
        out.append(_AttrSpecCG(spec.name, spec.type, spec.default))
    return out


def _attr_default_value_string(spec: "_AttrSpecCG") -> str:
    """The value-STRING an attr defaults to when absent, the ExtractAttrs / ParseParam fallback, and the
    ``attrs`` dict value the kernel factory converts. Matches the deployed stringified form per type:
    Int ``std::to_string``, Float ``%.9g``, String raw, ListInt the JSON array (``ListIntToJsonStr``)."""
    if spec.type == "Int":
        return "0" if spec.default is None else str(int(spec.default))
    if spec.type == "Float":
        return "0" if spec.default is None else f"{float(spec.default):.9g}"
    if spec.type == "String":
        return "" if spec.default is None else str(spec.default)
    if spec.type == "ListInt":
        return "[]" if spec.default is None else _json.dumps([int(v) for v in spec.default])


def _attr_reg_op_default_literal(spec: "_AttrSpecCG") -> str:
    """The C++ default LITERAL emitted in ``.ATTR(name, T, <literal>)`` (only when a default is set)."""
    if spec.type == "Int":
        return str(int(spec.default))                       # .ATTR(bias, Int, 0)
    if spec.type == "Float":
        return repr(float(spec.default))                    # .ATTR(alpha, Float, 0.01), valid C++ double literal
    if spec.type == "String":
        return f'"{spec.default}"'                           # .ATTR(mode, String, "sum")
    if spec.type == "ListInt":
        return "{" + ", ".join(str(int(v)) for v in spec.default) + "}"   # .ATTR(kernel_shape, ListInt, {2, 2})


def _attr_reg_op_line(spec: "_AttrSpecCG") -> str:
    """One REG_OP attr line: ``.REQUIRED_ATTR(name, T)`` when ``default is None`` else
    ``.ATTR(name, T, <default-literal>)``."""
    if spec.default is None:
        return f"    .REQUIRED_ATTR({spec.name}, {spec.type})"
    return f"    .ATTR({spec.name}, {spec.type}, {_attr_reg_op_default_literal(spec)})"


def _attr_parse_param_block(spec: "_AttrSpecCG") -> str:
    """The per-attr ParseParam body: match name+type in the ONNX ``"attribute"`` JSON and
    ``op_dest.SetAttr``. The value key is ABSENT exactly when the author set the type's proto zero
    (protobuf drops a zero-valued scalar field), so a missing key falls back to that proto zero, Int
    ``0``, Float ``0.0f``, String ``""``, never to the DECLARED default, which would replace an
    author-set zero. The declared default is carried by the REG_OP ``.ATTR`` literal instead, and applies
    when the attribute itself is absent from the node. Float reads FLOAT AS STRING (``std::stof``) but is
    defensive (falls back to a numeric ``f`` if the toolchain ever emits a number); String reads ``s``;
    ListInt loops ``ints`` into a ``std::vector<int64_t>``."""
    name, code = spec.name, _ATTR_ONNX_TYPE_CODE[spec.type]
    head = f'            if (attr["name"] == "{name}" && attr["type"] == {code}) {{'
    if spec.type == "Int":
        body = (
            f'                int64_t value = attr.contains("i") && !attr["i"].is_null()\n'
            f'                    ? attr["i"].get<int64_t>() : 0;\n'
            f'                op_dest.SetAttr("{name}", value);'
        )
    elif spec.type == "Float":
        # FLOAT-as-string quirk: the CANN parser hands FLOAT attrs as JSON strings, so read via
        # std::stof; defensive, fall back to a numeric `f` if a future toolchain emits a number instead.
        body = (
            f'                float value = 0.0f;\n'
            f'                if (attr.contains("f") && !attr["f"].is_null()) {{\n'
            f'                    value = attr["f"].is_string()\n'
            f'                        ? std::stof(attr["f"].get<std::string>()) : attr["f"].get<float>();\n'
            f'                }}\n'
            f'                op_dest.SetAttr("{name}", value);'
        )
    elif spec.type == "String":
        body = (
            f'                std::string value = attr.contains("s") && !attr["s"].is_null()\n'
            f'                    ? attr["s"].get<std::string>() : "";\n'
            f'                op_dest.SetAttr("{name}", value.c_str());'
        )
    else:  # ListInt
        body = (
            f'                std::vector<int64_t> value;\n'
            f'                for (auto e : attr["ints"]) value.push_back(e.get<int64_t>());\n'
            f'                op_dest.SetAttr("{name}", value);'
        )
    return f"{head}\n{body}\n            }}"


def _attr_extract_read_block(idx: int, spec: "_AttrSpecCG") -> str:
    """One attr's ExtractAttrs read at its GLOBAL declaration index *idx*. Value stringified to
    match the deployed contract: Int ``std::to_string``, Float ``%.9g`` (full float32 round-trip,
    unlike the reference's lossy ``std::to_string``), String the raw ``const char*``, ListInt
    ``ListIntToJsonStr`` (base helper)."""
    name, accessor = spec.name, _ATTR_RT_ACCESSOR[spec.type]
    fallback = _attr_default_value_string(spec)
    if spec.type == "Int":
        return (
            f"        const int64_t *{name}_ptr = attrs->{accessor}({idx});\n"
            f'        out["{name}"] = {name}_ptr ? std::to_string(*{name}_ptr) : "{fallback}";'
        )
    if spec.type == "Float":
        # %.9g round-trips a float32 exactly; std::to_string would fix 6 decimals and collapse small alphas.
        return (
            f"        const float *{name}_ptr = attrs->{accessor}({idx});\n"
            f'        if ({name}_ptr) {{ char _buf[32]; snprintf(_buf, sizeof _buf, "%.9g", '
            f'(double)*{name}_ptr); out["{name}"] = _buf; }}\n'
            f'        else {{ out["{name}"] = "{fallback}"; }}'
        )
    if spec.type == "String":
        return (
            f"        const char *{name}_ptr = attrs->{accessor}({idx});\n"
            f'        out["{name}"] = {name}_ptr ? {name}_ptr : "{fallback}";'
        )
    # ListInt: copy the continuous vector and JSON-encode via the base helper.
    return (
        f"        const auto *{name}_ptr = attrs->{accessor}({idx});\n"
        f"        if ({name}_ptr) {{ std::vector<int64_t> _v({name}_ptr->GetData(), "
        f"{name}_ptr->GetData() + {name}_ptr->GetSize()); out[\"{name}\"] = ListIntToJsonStr(_v); }}\n"
        f'        else {{ out["{name}"] = "{fallback}"; }}'
    )


def _attr_extract_attrs_method(attr_specs: list) -> str:
    """The generated ``ExtractAttrs`` override reading each attr at its GLOBAL declaration index.

    Returns an empty string when there are no attrs (the base's empty stub is used, no override emitted,
    keeping an attr-free op byte-identical). Otherwise emits an ``if (attrs==nullptr)`` default branch
    (defaults from ``AttrSpec.default``) then one positional accessor per attr at its ordinal index.
    """
    if not attr_specs:
        return ""
    null_defaults = "\n".join(
        f'            out["{s.name}"] = "{_attr_default_value_string(s)}";' for s in attr_specs
    )
    reads_block = "\n".join(_attr_extract_read_block(idx, s) for idx, s in enumerate(attr_specs))
    return f"""
    // Op attributes -> {{name: value-string}} at the declaration index (REG_OP order == this
    // positional index). Fed to the compile cache key and the Python kernel factory's attrs dict.
    void ExtractAttrs(const gert::RuntimeAttrs *attrs,
                      std::map<std::string, std::string> &out) const override {{
        if (attrs == nullptr) {{
{null_defaults}
            return;
        }}
{reads_block}
    }}
"""


# nlohmann json, the generated executor and onnx-plugin TUs ``#include <nlohmann/json.hpp>`` to read
# the kernel's ``*_aiv.json`` launch sidecar. The build puts an nlohmann include root (a dir holding
# ``nlohmann/json.hpp``) on the compile ``-I``; the resolution order and the accepted layouts are
# documented once, in vendors/README_en.md. ``_NLOHMANN_VERSION`` only shapes the ``json-<ver>``
# candidate paths.
_NLOHMANN_VERSION = "3.11.3"
# repo root = .../pypto (codegen.py is tools/exported_custom_op_litenpu/deploy/codegen.py -> parents[3]).
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_nlohmann_include_dir() -> Path:
    """Locate an nlohmann include ROOT, a dir ``D`` such that ``D/nlohmann/json.hpp`` is a file, for the
    build to place on ``-I`` so consumers resolve ``#include <nlohmann/json.hpp>`` (resolution order above)."""
    import os
    candidates: list[Path] = []
    # 1. installed pypto ships the nlohmann tree under lib/framework/3rd/include.
    try:
        import pypto
        if getattr(pypto, "__file__", None):
            candidates.append(Path(pypto.__file__).parent / "lib" / "framework" / "3rd" / "include")
    except ImportError:
        pass
    # 2. third-party roots (env root then the repo default), each shaping the known include-root layouts.
    roots = []
    tp = os.environ.get("PYPTO_THIRD_PARTY_PATH")
    if tp:
        roots.append(Path(tp))
    roots.append(_REPO_ROOT / "third_party_path")
    for root in roots:
        candidates.append(root / "Release" / "include")
        candidates.append(root / f"json-{_NLOHMANN_VERSION}" / "include")
        candidates.append(root / f"json-{_NLOHMANN_VERSION}" / "single_include")
        candidates.append(root / "include")
        candidates.append(root / "single_include")
    for d in candidates:
        if (d / "nlohmann" / "json.hpp").is_file():
            return d
    raise FileNotFoundError(
        "nlohmann include root (a dir holding nlohmann/json.hpp) not found. Probed:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nSet PYPTO_THIRD_PARTY_PATH, or install pypto (lib/framework/3rd/include)."
    )


# Shared custom-op base class shipped alongside the generated thin per-op executors.
# ``PtoCustomOp`` owns Compile/Execute orchestration, the compile cache, and the kernel
# launch; each generated executor is a small subclass.
_BUNDLED_PTO_CUSTOM_OP_H_PATH = Path(__file__).resolve().parent / "vendors" / "pto_custom_op.h"
_BUNDLED_PTO_CUSTOM_OP_CPP_PATH = Path(__file__).resolve().parent / "vendors" / "pto_custom_op.cpp"
# The kernel-module loader (imports the shipped op_kernel/<stem>.py by path) is injected
# INLINE into the base .cpp so the built .so imports no pypto Python module at runtime. Single source
# of truth (also unit-tested as ``deploy.embed``); the .cpp carries a placeholder for it.
_BUNDLED_EMBED_LOADER_PATH = Path(__file__).resolve().parent / "embed.py"
_EMBED_LOADER_PLACEHOLDER = "__PYPTO_EMBED_LOADER_SRC__"


def _bundled_pto_custom_op_h_source() -> str:
    """Return the shared ``PtoCustomOp`` base-class header (shipped with the executor)."""
    return _BUNDLED_PTO_CUSTOM_OP_H_PATH.read_text(encoding="utf-8")


def _bundled_pto_custom_op_cpp_source() -> str:
    """Return the shared ``PtoCustomOp`` base-class implementation with the inline embed loader injected.

    The ``.so`` must import no pypto Python module at op-compile time, so the loader (which imports the
    shipped ``op_kernel/<stem>.py`` by path) is compiled in: its source (``deploy.embed``) is spliced
    into the base ``.cpp``'s placeholder here.
    """
    cpp = _BUNDLED_PTO_CUSTOM_OP_CPP_PATH.read_text(encoding="utf-8")
    if _EMBED_LOADER_PLACEHOLDER not in cpp:
        raise RuntimeError(
            f"pto_custom_op.cpp is missing the {_EMBED_LOADER_PLACEHOLDER!r} placeholder for the "
            "inline embedded-compile loader"
        )
    loader_src = _BUNDLED_EMBED_LOADER_PATH.read_text(encoding="utf-8")
    return cpp.replace(_EMBED_LOADER_PLACEHOLDER, loader_src)


# Logging: route the generated trace/value logs through the Ascend slog backend (``dlog_debug`` /
# ``dlog_error``) instead of stdout ``printf``. ``GELOGD``/``ACL_LOG_DEBUG`` are ge/acl
# source-tree internals absent from the CANN install; the ``dlog_*`` backend they
# all funnel into ships in ``toolchain/slog.h`` with module ids in ``base/log_types.h``.
# ``PTO_CUSTOM_LOGD`` is gated by the runtime log level (``CheckLogLevel``), so trace lines emit only at
# debug level (``ASCEND_GLOBAL_LOG_LEVEL=0``); ``PTO_CUSTOM_LOGE`` carries the hard-failure diagnostics,
# which must be readable at the default level. The ``#ifndef`` guards make both safe to emit into
# every generated TU. Compile tests resolve the two headers from ``tools/exported_custom_op_litenpu/tests/ut/fixtures``
# (mock); real builds resolve them from the CANN install (link dep: ``unified_dlog``).
_LOG_PREAMBLE = """#include "toolchain/slog.h"
#include "base/log_types.h"
#ifndef PTO_CUSTOM_LOGD
#define PTO_CUSTOM_LOGD(fmt, ...) dlog_debug(OP, fmt, ##__VA_ARGS__)
#endif
#ifndef PTO_CUSTOM_LOGE
#define PTO_CUSTOM_LOGE(fmt, ...) dlog_error(OP, fmt, ##__VA_ARGS__)
#endif
"""


# torch dtype ↔ GE conversion (build-time codegen only)
# The torch↔GE dtype mapping is a codegen concern, so it lives here in the export helpers rather
# than in the pypto.extensions.torch_custom_op_litenpu core. Two uses, both build-time:
#   • the OpDef ``.INPUT/.OUTPUT`` dtype tokens (``_torch_dtype_to_ge_dtype`` → ``ge::DT_*``);
#   • the runtime enum↔torch bridge the infer pybind wrapper needs, emitted INLINE into the
#     generated wrapper Python (see ``_inline_dtype_conversion_src``) so the built ``.so``
#     imports NOTHING from ``pypto.extensions.torch_custom_op_litenpu`` (mirrors the inlined embed loader).
# The GE-int table is the single source of truth for the supported basename set.

# Numeric ``ge::DataType`` values (``gert_ge_minimal.hpp``) → torch dtype basename.
_GE_DATA_TYPE_VALUE_TO_TORCH_BASE: dict[int, str] = {
    0: "float32",  # DT_FLOAT
    1: "float16",  # DT_FLOAT16
    2: "int8",  # DT_INT8
    3: "int32",  # DT_INT32
    4: "uint8",  # DT_UINT8
    6: "int16",  # DT_INT16
    7: "uint16",  # DT_UINT16
    8: "uint32",  # DT_UINT32
    9: "int64",  # DT_INT64
    10: "uint64",  # DT_UINT64
    11: "float64",  # DT_DOUBLE
    12: "bool",  # DT_BOOL
    16: "complex64",  # DT_COMPLEX64
    17: "complex128",  # DT_COMPLEX128
    18: "qint8",  # DT_QINT8
    19: "qint16",  # DT_QINT16
    20: "qint32",  # DT_QINT32
    21: "quint8",  # DT_QUINT8
    22: "quint16",  # DT_QUINT16
    27: "bfloat16",  # DT_BF16
    33: "complex32",  # DT_COMPLEX32
}

# Inverse: torch dtype basename → ``ge::DataType`` enum value. Built from the same table so both
# directions, and the supported-base set ``frozenset(_TORCH_BASE_TO_GE_DATA_TYPE_VALUE)``, stay in
# lockstep from one source of truth.
_TORCH_BASE_TO_GE_DATA_TYPE_VALUE: dict[str, int] = {
    base: value for value, base in _GE_DATA_TYPE_VALUE_TO_TORCH_BASE.items()
}

# Torch basenames whose ``ge::DataType`` tail is not ``<base>.upper()`` (e.g. float32 → FLOAT).
_TORCH_BASE_TO_GE_DT_NAME_EXCEPTIONS: dict[str, str] = {
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "bfloat16": "BF16",
}


def _torch_dtype_to_ge_dtype(dtype) -> str:
    """Map torch dtype-like values to a GE dtype token string (e.g. ``ge::DT_FLOAT``).

    Only dtypes with a ``ge::DataType`` enum value are accepted; entries in
    ``_TORCH_BASE_TO_GE_DT_NAME_EXCEPTIONS`` override the default ``ge::DT_`` +
    uppercase-basename rule.
    """
    dtype_name = str(dtype)
    if not dtype_name.startswith("torch."):
        raise ValueError(f"Unsupported dtype for GE mapping: {dtype!r}")
    base = dtype_name.replace("torch.", "", 1)
    if base not in _TORCH_BASE_TO_GE_DATA_TYPE_VALUE:
        raise ValueError(f"Unsupported torch dtype for GE mapping: {dtype!r}")
    ge_dt_name = _TORCH_BASE_TO_GE_DT_NAME_EXCEPTIONS.get(base)
    if ge_dt_name is not None:
        return f"ge::DT_{ge_dt_name}"
    # int*/uint*, bool, complex*, qint*, quint*: ``DT_`` + basename uppercased.
    return f"ge::DT_{base.upper()}"


def _inline_dtype_conversion_src() -> str:
    """Return Python source defining the runtime torch↔``ge::DataType``-enum conversions.

    Emitted at module scope of the exec'd infer-wrapper source so the generated ``.so``
    imports NOTHING from ``pypto.extensions.torch_custom_op_litenpu``: the two conversions the wrapper needs
    (``_ge_data_type_enum_value_to_torch_dtype`` and
    ``_torch_dtype_to_ge_data_type_enum_value_recursive``) are defined inline from the
    same ``_GE_DATA_TYPE_VALUE_TO_TORCH_BASE`` table used by the codegen ge-token path.
    These emitted functions are the runtime torch↔GE dtype bridge the infer pybind wrapper needs.
    """
    items = ", ".join(
        f"{value}: {base!r}"
        for value, base in sorted(_GE_DATA_TYPE_VALUE_TO_TORCH_BASE.items())
    )
    return (
        "_GE_INT_TO_TORCH_BASE = {" + items + "}\n"
        "_TORCH_BASE_TO_GE_INT = {b: i for i, b in _GE_INT_TO_TORCH_BASE.items()}\n"
        "\n"
        "def _ge_data_type_enum_value_to_torch_dtype(value):\n"
        "    import torch\n"
        "    try:\n"
        "        base = _GE_INT_TO_TORCH_BASE[value]\n"
        "    except KeyError:\n"
        "        raise ValueError('Unsupported ge::DataType enum value for torch mapping: ' + repr(value))\n"
        "    td = getattr(torch, base, None)\n"
        "    if td is None:\n"
        "        raise ValueError('torch has no dtype attribute ' + repr(base))\n"
        "    return td\n"
        "\n"
        "def _torch_dtype_to_ge_data_type_enum_value(dtype):\n"
        "    dtype_name = str(dtype)\n"
        "    if not dtype_name.startswith('torch.'):\n"
        "        raise ValueError('Unsupported dtype for GE mapping: ' + repr(dtype))\n"
        "    base = dtype_name.replace('torch.', '', 1)\n"
        "    try:\n"
        "        return _TORCH_BASE_TO_GE_INT[base]\n"
        "    except KeyError:\n"
        "        raise ValueError('Unsupported torch dtype for GE enum mapping: ' + repr(dtype))\n"
        "\n"
        "def _torch_dtype_to_ge_data_type_enum_value_recursive(value):\n"
        "    if isinstance(value, tuple):\n"
        "        return tuple(_torch_dtype_to_ge_data_type_enum_value(v) for v in value)\n"
        "    return _torch_dtype_to_ge_data_type_enum_value(value)\n"
    )


@dataclass(frozen=True)
class _HelperFuncCodegenMeta:
    """Result of parsing an N-input / M-output helper (``infer_shape`` or ``infer_dtype``).

    Both helpers share the same shape: each parameter is a single ``torch.Size``
    or ``torch.dtype`` annotation, and the return is the same item type
    (single output) or a fixed-length tuple of it (multi output).
    """

    param_names: list[str]
    n_outputs: int
    cpp_bind_name: str
    # Shape-affecting attr params trailing the tensor-shape params of an ``infer_shape``. Each is
    # ``(name, global_attr_index, ge_type, default)``, the GLOBAL index into the op's ordered
    # ``_attr_specs``, NOT the param's position. Empty for ``infer_dtype`` and attr-free /
    # shape-invariant ops.
    attr_read_infos: tuple = ()


def _annotation_contains_torch_dtype(py_ann: Any) -> bool:
    """True if *py_ann* is ``torch.dtype`` or a tuple annotation containing it.

    Used by the pybind wrapper to decide whether the return value needs the
    recursive ``torch.dtype`` → ``ge::DataType`` enum-int conversion before
    pybind11 casts to ``int64_t`` / ``std::tuple<int64_t, ...>``.
    """
    if py_ann is torch.dtype:
        return True
    if get_origin(py_ann) is tuple:
        targs = get_args(py_ann)
        return any(_annotation_contains_torch_dtype(a) for a in targs)
    return False


def _to_cpp_type(py_ann: Any) -> str:
    """Map a Python type annotation to a C++ type string (recursively for tuple/list)."""
    if py_ann is torch.Size:
        return "std::vector<int64_t>"  # rank-erased shape: pybind iterates → fills vector
    if py_ann is torch.dtype:
        return "int64_t"  # raw ge::DataType enum value; wrapper converts to torch.dtype on call
    if py_ann is int:
        return "int64_t"  # Python int -> fixed-width C++ (aligns with gert::Shape dimensions)
    if py_ann is float:
        return "float"
    if py_ann is bool:
        return "bool"
    if py_ann is str:
        return "std::string"  # a String attr: pybind casts it to a Python str for the infer func
    origin = get_origin(py_ann)
    if origin is tuple:
        targs = get_args(py_ann)
        if len(targs) == 2 and targs[1] is Ellipsis:
            elem = _to_cpp_type(targs[0])
            return f"std::vector<{elem}>"
        inner = ", ".join(_to_cpp_type(a) for a in targs)
        return f"std::tuple<{inner}>"
    if origin is list:
        targs = get_args(py_ann)
        if not targs:
            raise TypeError(f"list annotation must specify element type: {py_ann!r}")
        return f"std::vector<{_to_cpp_type(targs[0])}>"
    raise TypeError(f"Unsupported annotation for C++ mapping: {py_ann!r}")


def _import_module_by_path_src(*, module_name: str, py_path: str) -> str:
    """Return Python that imports *py_path* as *module_name* and binds it to ``_pypto_infer_mod``.

    Generic ``importlib.util`` loader machinery only, it carries no user code. The path is baked in as
    a compile-time literal (``repr``'d here, so any path escapes correctly); *module_name* names the
    module for tracebacks (nothing is inserted into ``sys.modules``). Used by the ``import_via="path"``
    wrapper, which has no ``PtoCustomOp`` to resolve the file for it.
    """
    return (
        "import importlib.util as _pypto_ilu\n"
        f"_pypto_spec = _pypto_ilu.spec_from_file_location({module_name!r}, {py_path!r})\n"
        "_pypto_infer_mod = _pypto_ilu.module_from_spec(_pypto_spec)\n"
        "_pypto_spec.loader.exec_module(_pypto_infer_mod)"
    )


def _generate_pybind_wrapper(
    func: Callable,
    cpp_func_name: str,
    *,
    import_via: str,
    basename: str = "",
    stem: str = "",
    py_path: str = "",
) -> str:
    """Generate C++ pybind11 wrapper that runs the given Python function.

    Shape parameters annotated as ``torch.Size`` are wrapped in a real
    ``torch.Size`` instance before being handed to the user's Python function:
    the wrapper imports ``torch`` once and applies ``torch.Size(...)`` per
    shape argument. Dtype parameters annotated as ``torch.dtype`` are passed
    in as the raw ``ge::DataType`` enumerator value (``int64_t``) and converted
    to a real ``torch.dtype`` before the user function is invoked. The
    torch↔``ge::DataType``-enum conversions are emitted INLINE at module scope of
    the embedded source (see ``_inline_dtype_conversion_src``) so the built
    ``.so`` imports NOTHING from ``pypto.extensions.torch_custom_op_litenpu``, only ``torch`` must be
    importable in the embedded interpreter at call time.

    When the return annotation is ``torch.dtype`` (or a fixed-length tuple
    containing ``torch.dtype``), the wrapper applies the inline
    ``_torch_dtype_to_ge_data_type_enum_value_recursive`` to the result before
    pybind11 casts it to ``int64_t`` (or ``std::tuple<int64_t, ...>``). This is
    the inverse of the input-side conversion and is what makes the ``infer_dtype`` hook
    accept arbitrary Python.

    The function is emitted inside an anonymous namespace, for embedding in a host TU that already
    includes the pybind headers and ``namespace py = pybind11``.

    The wrapper NEVER carries *func*'s source: it always obtains the Python function from an IMPORTED
    module, and *import_via* selects the import mechanism. Both variants first ``py::exec`` the
    torch/dtype-bridge source into a fresh ``py::dict globals``, that generic bridge is the one piece of
    inline Python, and is never sourced from a ``.py``.

    * ``"op_kernel"``, the deployed executor's wrapper: fetch the module via
      ``PtoCustomOp::ImportKernelModule("<stem>", "<basename>")`` (strict, a null return throws
      ``py::error_already_set``), then read *func*'s name off it. *basename* is
      ``GetKernelPyBasename()`` (``"<snake>.py"``) and *stem* is ``GetCompileModuleStem()``
      (``"pypto_compile_<Op>"``), the module memo key, kept identical to ``Compile()``'s so the module
      imports once. Both are required.
    * ``"path"``, a self-contained wrapper with NO ``PtoCustomOp`` reference (the standalone test host
      TU): ``py::exec`` a small ``importlib.util`` snippet that loads *py_path*, baked in as a
      compile-time literal, and read *func*'s name off the resulting module. *py_path* is required and
      must hold a module defining *func*, which the codegen tests generate (see
      ``tests/ut/codegen_test_helpers.py``).
    """
    if import_via not in ("op_kernel", "path"):
        raise ValueError(f"import_via must be 'op_kernel' or 'path', got {import_via!r}")
    if import_via == "op_kernel" and not (basename and stem):
        raise ValueError("import_via='op_kernel' requires both basename and stem")
    if import_via == "path" and not py_path:
        raise ValueError("import_via='path' requires py_path")

    py_func_name = func.__name__
    py_sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    def ann_for(p: inspect.Parameter):
        if p.name in hints:
            return hints[p.name]
        return p.annotation

    param_anns = [ann_for(p) for p in py_sig.parameters.values()]
    has_shape_arg = any(a is torch.Size for a in param_anns)
    has_dtype_arg = any(a is torch.dtype for a in param_anns)
    param_cpp_types = [_to_cpp_type(a) for a in param_anns]
    cpp_args_list = ", ".join(
        [f"{param_cpp_types[i]} {py_arg.name}" for i, py_arg in enumerate(py_sig.parameters.values())]
    )
    ret_ann = hints.get("return", py_sig.return_annotation)
    cpp_return_type = _to_cpp_type(ret_ann)
    has_dtype_return = _annotation_contains_torch_dtype(ret_ann)
    needs_torch_import = has_shape_arg or has_dtype_arg or has_dtype_return

    def _arg_expr(p: inspect.Parameter) -> str:
        ann = ann_for(p)
        if ann is torch.Size:
            return f"_torch_Size(py::cast({p.name}))"
        if ann is torch.dtype:
            return f"_ge_dt_to_torch(py::cast({p.name}))"
        return f"py::cast({p.name})"

    pybind_args_list = ", ".join(_arg_expr(p) for p in py_sig.parameters.values())

    # The torch↔ge::DataType conversions. They land in the bridge source below, which BOTH wrapper
    # variants ``py::exec`` into ``globals`` (the bridge is generic and is never sourced from a .py),
    # and are fetched from there. Defining them inline is what keeps the ``.so`` from
    # importing anything out of ``pypto.extensions.torch_custom_op_litenpu``.
    dtype_conv = _inline_dtype_conversion_src() if (has_dtype_arg or has_dtype_return) else ""
    # The bridge itself: the conversions (+ ``import torch``) and nothing else.
    # Empty only when the wrapper needs no torch import at all.
    if has_dtype_arg or has_dtype_return:
        bridge_source = f"import torch\n\n{dtype_conv}"
    elif needs_torch_import:
        bridge_source = "import torch\n"
    else:
        bridge_source = ""
    helper_lookups: list[str] = []
    if has_shape_arg:
        helper_lookups.append(
            '    py::object _torch_Size = py::module_::import("torch").attr("Size");'
        )
    if has_dtype_arg:
        helper_lookups.append(
            '    py::object _ge_dt_to_torch = globals["_ge_data_type_enum_value_to_torch_dtype"];'
        )
    if has_dtype_return:
        helper_lookups.append(
            '    py::object _torch_dt_to_ge = globals["_torch_dtype_to_ge_data_type_enum_value_recursive"];'
        )
    helper_block = ("\n" + "\n".join(helper_lookups)) if helper_lookups else ""

    call_expr = f"{py_func_name}_py({pybind_args_list})"
    if has_dtype_return:
        # User returns torch.dtype (or a tuple of them); convert each item to the
        # ge::DataType enum int via the imported helper before pybind11 casts to
        # int64_t / std::tuple<int64_t, ...>.
        call_expr = f"_torch_dt_to_ge({call_expr})"

    # GIL prologue: bootstrap an interpreter if the host has none, then hold the
    # GIL for the WHOLE function, including the catch block below. Scoping the
    # acquire to the try-block would release the GIL before the catch runs, and
    # py::error_already_set handling (PyErr_Print) calls into CPython, so it
    # would fault without the GIL (e.g. _Py_HandleSystemExit -> _Py_GetConfig).
    gil_prologue = """\
    // One-time CPython bootstrap for a non-Python host (GE/Ascend executor that dlopen'd this .so);
    // guarded so a Python host is untouched, and never finalized (called repeatedly; re-init is fragile).
    static const bool _pypto_py_bootstrapped = []() {
        if (!Py_IsInitialized()) {
            Py_InitializeEx(0);   // 0: don't install signal handlers (library-friendly)
            PyEval_SaveThread();  // drop the GIL the init thread holds; reacquired below
        }
        return true;
    }();
    (void)_pypto_py_bootstrapped;

    // Held across the try AND the catch: error_already_set handling needs the GIL.
    py::gil_scoped_acquire gil;"""

    if import_via == "op_kernel":
        # The dtype bridge is exec'd into a FRESH ``py::dict globals``; the infer func comes from the
        # shipped op_kernel/<basename>, the single source of truth for this op's Python.
        py_work = f"""    const std::string _dtype_bridge_src = R"PYSRC(
{bridge_source}
)PYSRC";

    py::dict globals;                       // FRESH dict per call; py::exec auto-seeds __builtins__
    py::exec(_dtype_bridge_src, globals, globals);   // torch + dtype bridge
    PyObject *_m = PtoCustomOp::ImportKernelModule("{stem}", "{basename}");
    if (_m == nullptr) {{ throw py::error_already_set(); }}  // FATAL msg set by ImportKernelModule
    py::object _mod = py::reinterpret_steal<py::object>(_m);
    py::object {py_func_name}_py = _mod.attr("{py_func_name}");
    PTO_CUSTOM_LOGD("{cpp_func_name}: infer source=op_kernel/{basename}\\n");{helper_block}

    {cpp_return_type} _ret = {call_expr}.cast<{cpp_return_type}>();
    PTO_CUSTOM_LOGD("Exiting {cpp_func_name} (ok)\\n");
    return _ret;"""
    else:
        # Self-contained variant: no PtoCustomOp, so the module is loaded by an importlib snippet with
        # the .py path baked in as a compile-time literal. Only generic loader machinery is inline -
        # the infer function itself lives in that file.
        import_src = _import_module_by_path_src(
            module_name=f"_pypto_infer_{cpp_func_name}", py_path=py_path
        )
        py_work = f"""    const std::string _dtype_bridge_src = R"PYSRC(
{bridge_source}
)PYSRC";
    const std::string _import_by_path_src = R"PYSRC(
{import_src}
)PYSRC";

    py::dict globals;                       // FRESH dict per call; py::exec auto-seeds __builtins__
    py::exec(_dtype_bridge_src, globals, globals);     // torch + dtype bridge
    py::exec(_import_by_path_src, globals, globals);   // load the .py defining {py_func_name}
    py::object _mod = globals["_pypto_infer_mod"];
    py::object {py_func_name}_py = _mod.attr("{py_func_name}");{helper_block}

    {cpp_return_type} _ret = {call_expr}.cast<{cpp_return_type}>();
    PTO_CUSTOM_LOGD("Exiting {cpp_func_name} (ok)\\n");
    return _ret;"""

    fn_block = f"""{cpp_return_type} {cpp_func_name}({cpp_args_list}) {{
    PTO_CUSTOM_LOGD("Entering {cpp_func_name}\\n");
{gil_prologue}
{py_work}
}}"""

    return f"""namespace {{

{fn_block}

}}
"""


def _infer_shape_attr_ge_type(ann: Any) -> str | None:
    """The GE attr type of a NON-``torch.Size`` ``infer_shape`` param (a shape-affecting attr), or
    ``None`` if the annotation isn't an accepted attr shape (``int`` -> ``Int``, ``list[int]`` ->
    ``ListInt``, ``str`` -> ``String``, ``float`` -> ``Float``). ``InferShapeGeImpl`` reads the attr and
    passes it to the author's own ``infer_shape``, so what a value means is the author's to decide; this
    maps the annotation to the GE attr type that carries it."""
    if ann is int:
        return "Int"
    if ann is str:
        return "String"
    if ann is float:
        return "Float"
    if get_origin(ann) is list and get_args(ann) == (int,):
        return "ListInt"
    return None


def _parse_infer_shape_for_codegen(func: Callable, attr_specs=None) -> _HelperFuncCodegenMeta:
    """Validate infer_shape annotations and return codegen metadata.

    The LEADING parameters are the tensor input shapes, each annotated ``torch.Size``. An op whose
    output shape depends on an attr may declare TRAILING shape-affecting attr params AFTER all the
    shape params: ``int`` (-> ``Int``), ``float`` (-> ``Float``), ``str`` (-> ``String``) or ``list[int]``
    (-> ``ListInt``). ``param_names`` holds only the tensor-shape params (so ``n_in`` stays the TENSOR
    count); the trailing attr params land in ``attr_read_infos`` mapped to their GLOBAL ``_attr_specs``
    index via *attr_specs*. The return annotation must be ``torch.Size`` (single output) or
    ``tuple[torch.Size, torch.Size]`` (multi-output).
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(f"infer_shape requires type annotations resolvable by get_type_hints: {exc}") from exc
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError(
            "infer_shape must have a return annotation (torch.Size or tuple[torch.Size, torch.Size])"
        )
    ret_ann = hints.get("return", sig.return_annotation)
    n_outputs = parse_output_arity(ret_ann, torch.Size, "infer_shape return")
    params = list(sig.parameters.values())
    if not params:
        raise TypeError("infer_shape must accept at least one shape argument")

    # Partition: leading torch.Size (tensor shapes), then trailing shape-affecting attr params. A
    # torch.Size after an attr param is rejected, shapes must all precede the attrs.
    spec_by_name = {s.name: s for s in _normalize_attr_specs(attr_specs)}
    shape_names: list[str] = []
    attr_read_infos: list = []
    seen_attr = False
    for p in params:
        # The parameter name lands verbatim in the generated wrapper's C++ parameter list, so it has
        # to be an emittable identifier. Only an attr's name is ALSO emitted as an InferShape local.
        ann = hints.get(p.name, p.annotation)
        _validate_emitted_identifier(
            p.name, "infer_shape parameter name", infer_body_scope=ann is not torch.Size
        )
        if ann is torch.Size:
            if seen_attr:
                raise TypeError(
                    f"infer_shape parameter {p.name!r}: all torch.Size (tensor-shape) params must "
                    "precede any shape-affecting attr param"
                )
            shape_names.append(p.name)
            continue
        seen_attr = True
        ge_type = _infer_shape_attr_ge_type(ann)
        if ge_type is None:
            raise TypeError(
                f"infer_shape parameter {p.name!r}: a non-torch.Size param must be a shape-affecting "
                "attr annotated int (Int), float (Float), str (String) or list[int] (ListInt)"
            )
        spec = spec_by_name.get(p.name)
        if spec is None:
            raise TypeError(
                f"infer_shape reads attr {p.name!r} but no matching AttrSpec was declared (attrs="
                f"{sorted(spec_by_name)})"
            )
        if spec.type != ge_type:
            raise TypeError(
                f"infer_shape param {p.name!r} is {ge_type} but its AttrSpec declares {spec.type}"
            )
        global_index = list(spec_by_name).index(p.name)  # dict preserves _attr_specs order
        attr_read_infos.append((p.name, global_index, ge_type, spec.default))
    if not shape_names:
        raise TypeError("infer_shape must accept at least one torch.Size (tensor-shape) argument")
    return _HelperFuncCodegenMeta(
        param_names=shape_names,
        n_outputs=n_outputs,
        cpp_bind_name=_CPP_BIND__INFER_SHAPE,
        attr_read_infos=tuple(attr_read_infos),
    )


def _infer_shape_input_block(i: int) -> str:
    """One input shape: fill ``std::vector<int64_t>`` from ``gert::Shape`` dims.

    The context getter is null-guarded: the GE header documents a nullptr return for an index the node
    does not have, and dereferencing it is a segfault the executor's try/catch cannot turn into
    ``GRAPH_FAILED``.
    """
    return f"""    const gert::Shape* in{i}_shape = context->GetInputShape({i});
    if (in{i}_shape == nullptr) {{
        PTO_CUSTOM_LOGD("ERROR: InferShapeGeImpl: null input shape at index {i} -> GRAPH_FAILED\\n");
        return ge::GRAPH_FAILED;
    }}
    std::vector<int64_t> in{i}_shape_vec;
    in{i}_shape_vec.reserve(in{i}_shape->GetDimNum());
    for (size_t j = 0; j < in{i}_shape->GetDimNum(); ++j) {{
        in{i}_shape_vec.push_back((*in{i}_shape)[j]);
    }}
    PTO_CUSTOM_LOGD("InferShapeGeImpl: in{i} ndim=%zu\\n", in{i}_shape_vec.size());
"""


def _infer_shape_attr_read_block(name: str, global_index: int, ge_type: str, default) -> str:
    """One shape-affecting attr read for InferShapeGeImpl: read the attr at its GLOBAL declaration
    index off ``context->GetAttrs()`` into a local, marshalled to the pybind param type (Int ->
    ``int64_t``, ListInt -> ``std::vector<int64_t>``, String -> ``std::string``, Float -> ``float``).

    The local is initialized to the AttrSpec default, or to a type zero. That null-attrs path is
    unreachable in practice, since a shape attr is REQUIRED and GE guarantees it, but it stays
    well-defined. A Float initializer always carries a decimal point or an exponent before the ``f``
    suffix, because ``1f`` is not a C++ floating literal."""
    if ge_type == "String":
        init = '""' if default is None else f'"{default}"'
        return (
            f"    std::string {name} = {init};\n"
            f"    if (_attrs != nullptr) {{ const char *_p_{name} = _attrs->GetStr({global_index}); "
            f"if (_p_{name}) {name} = _p_{name}; }}"
        )
    if ge_type == "Float":
        if default is None:
            init = "0.0f"
        else:
            # ``%.9g`` drops the point of an integral value, and ``1f`` is not a C++ literal: the
            # package build fails with "unable to find numeric literal operator". A non-finite
            # default is left exactly as it was formatted; this PR does not accept one.
            text = f"{float(default):.9g}"
            if re.fullmatch(r"[+-]?[0-9]+", text):
                text += ".0"
            init = text + "f"
        return (
            f"    float {name} = {init};\n"
            f"    if (_attrs != nullptr) {{ const float *_p_{name} = _attrs->GetFloat({global_index}); "
            f"if (_p_{name}) {name} = *_p_{name}; }}"
        )
    if ge_type == "Int":
        init = "0" if default is None else str(int(default))
        return (
            f"    int64_t {name} = {init};\n"
            f"    if (_attrs != nullptr) {{ const int64_t *_p_{name} = _attrs->GetInt({global_index}); "
            f"if (_p_{name}) {name} = *_p_{name}; }}"
        )
    # ListInt
    init = "" if default is None else " = {" + ", ".join(str(int(v)) for v in default) + "}"
    return (
        f"    std::vector<int64_t> {name}{init};\n"
        f"    if (_attrs != nullptr) {{ const auto *_p_{name} = _attrs->GetListInt({global_index}); "
        f"if (_p_{name}) {name}.assign(_p_{name}->GetData(), _p_{name}->GetData() + _p_{name}->GetSize()); }}"
    )


def _infer_shape_ge_impl_body(meta: _HelperFuncCodegenMeta) -> str:
    """Return C++ statements for InferShapeGeImpl's body (no surrounding braces).

    Reads each input from ``context->GetInputShape(i)`` into a ``std::vector<int64_t>``, reads any
    shape-affecting attrs from ``context->GetAttrs()``, calls the embedded pybind wrapper
    ``meta.cpp_bind_name`` with the shape vectors THEN the attr locals (matching the Python
    ``infer_shape(x_shape, ..., attr, ...)`` order), and fills ``context->GetOutputShape(k)`` for each
    output. ``meta`` must come from ``_parse_infer_shape_for_codegen``.

    Both context getters are null-guarded and each output's rank is checked against
    ``gert::Shape::kMaxDimNum`` before ``SetDimNum``, so an index GE does not have and an over-rank
    Python result both return ``GRAPH_FAILED`` instead of faulting or writing past the fixed ``dims_``
    array.
    """
    n_in = len(meta.param_names)
    inputs_cpp = "\n".join(_infer_shape_input_block(i) for i in range(n_in))
    # Shape-affecting attrs: one GetAttrs() read per attr, appended to the wrapper call in
    # infer_shape declaration order (after the shape args).
    attr_reads = meta.attr_read_infos
    if attr_reads:
        attrs_cpp = "    const gert::RuntimeAttrs *_attrs = context->GetAttrs();\n" + "\n".join(
            _infer_shape_attr_read_block(name, gidx, ge_type, default)
            for (name, gidx, ge_type, default) in attr_reads
        ) + "\n"
    else:
        attrs_cpp = ""
    call_arg_names = [f"in{i}_shape_vec" for i in range(n_in)] + [info[0] for info in attr_reads]
    call_args = ", ".join(call_arg_names)
    bind = meta.cpp_bind_name
    n_out = meta.n_outputs

    if n_out == 1:
        return f"""    PTO_CUSTOM_LOGD("Entering InferShapeGeImpl\\n");
{inputs_cpp}
{attrs_cpp}    auto out_shape_vec = {bind}({call_args});
    PTO_CUSTOM_LOGD("InferShapeGeImpl: out0 ndim=%zu\\n", out_shape_vec.size());
    gert::Shape* out_shape = context->GetOutputShape(0);
    if (out_shape == nullptr) {{
        PTO_CUSTOM_LOGD("ERROR: InferShapeGeImpl: null output shape at index 0 -> GRAPH_FAILED\\n");
        return ge::GRAPH_FAILED;
    }}
    if (out_shape_vec.size() > gert::Shape::kMaxDimNum) {{
        PTO_CUSTOM_LOGD("ERROR: InferShapeGeImpl: out0 rank %zu exceeds gert::Shape::kMaxDimNum"
                        " -> GRAPH_FAILED\\n", out_shape_vec.size());
        return ge::GRAPH_FAILED;
    }}
    out_shape->SetDimNum(out_shape_vec.size());
    for (size_t j = 0; j < out_shape_vec.size(); ++j) {{
        (*out_shape)[j] = out_shape_vec[j];
    }}
    PTO_CUSTOM_LOGD("Exiting InferShapeGeImpl (status=%u)\\n", (unsigned)ge::GRAPH_SUCCESS);
    return ge::GRAPH_SUCCESS;
"""

    # Multi-output: pybind returns std::tuple<std::vector<int64_t>, ...> from the
    # Python tuple of torch.Size objects. Unpack each std::vector via std::get<k>.
    out_assign_lines: list[str] = []
    for k in range(n_out):
        inner = f"std::get<{k}>(out_shape_tuple)"
        out_assign_lines.append(
            f"""    PTO_CUSTOM_LOGD("InferShapeGeImpl: out{k} ndim=%zu\\n", {inner}.size());
    gert::Shape* out_shape_{k} = context->GetOutputShape({k});
    if (out_shape_{k} == nullptr) {{
        PTO_CUSTOM_LOGD("ERROR: InferShapeGeImpl: null output shape at index {k} -> GRAPH_FAILED\\n");
        return ge::GRAPH_FAILED;
    }}
    if ({inner}.size() > gert::Shape::kMaxDimNum) {{
        PTO_CUSTOM_LOGD("ERROR: InferShapeGeImpl: out{k} rank %zu exceeds gert::Shape::kMaxDimNum"
                        " -> GRAPH_FAILED\\n", {inner}.size());
        return ge::GRAPH_FAILED;
    }}
    out_shape_{k}->SetDimNum({inner}.size());
    for (size_t j = 0; j < {inner}.size(); ++j) {{
        (*out_shape_{k})[j] = {inner}[j];
    }}
"""
        )
    assigns = "\n".join(out_assign_lines)
    return f"""    PTO_CUSTOM_LOGD("Entering InferShapeGeImpl\\n");
{inputs_cpp}
{attrs_cpp}    auto out_shape_tuple = {bind}({call_args});
{assigns}
    PTO_CUSTOM_LOGD("Exiting InferShapeGeImpl (status=%u)\\n", (unsigned)ge::GRAPH_SUCCESS);
    return ge::GRAPH_SUCCESS;"""


def _parse_infer_dtype_for_codegen(func: Callable) -> _HelperFuncCodegenMeta:
    """Validate infer_dtype annotations and return codegen metadata.

    Each parameter annotation must be ``torch.dtype``. The return annotation
    must be ``torch.dtype`` (single output) or
    ``tuple[torch.dtype, torch.dtype, ...]`` with explicit count (multi-output).
    The body is arbitrary Python, runs inside the embedded pybind wrapper.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(
            f"infer_dtype requires type annotations resolvable by get_type_hints: {exc}"
        ) from exc
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError(
            "infer_dtype must have a return annotation (torch.dtype or tuple[torch.dtype, torch.dtype])"
        )
    ret_ann = hints.get("return", sig.return_annotation)
    n_outputs = parse_output_arity(ret_ann, torch.dtype, "infer_dtype return")
    params = list(sig.parameters.values())
    if not params:
        raise TypeError("infer_dtype must accept at least one dtype argument")
    for p in params:
        # Bridged positionally as ``in{i}_dtype_value``, so the name reaches only the wrapper's own
        # parameter list -- it cannot shadow an InferDataType local.
        _validate_emitted_identifier(p.name, "infer_dtype parameter name", infer_body_scope=False)
        ann = hints.get(p.name, p.annotation)
        if ann is not torch.dtype:
            raise TypeError(
                f"infer_dtype parameter {p.name!r}: expected torch.dtype, got {ann!r}"
            )
    return _HelperFuncCodegenMeta(
        param_names=[p.name for p in params],
        n_outputs=n_outputs,
        cpp_bind_name=_CPP_BIND__INFER_DTYPE,
    )


def _infer_dtype_input_block(i: int) -> str:
    """One input dtype: read raw ``ge::DataType`` enum as ``int64_t``."""
    return (
        f"    int64_t in{i}_dtype_value = "
        f"static_cast<int64_t>(context->GetInputDataType({i}));\n"
        f'    PTO_CUSTOM_LOGD("InferDataType: in{i} dtype=%lld\\n", (long long)in{i}_dtype_value);'
    )


def _infer_dtype_ge_impl_body(meta: _HelperFuncCodegenMeta) -> str:
    """Return C++ statements for InferDataType's body (no surrounding braces).

    Reads each input's ``GetInputDataType(i)`` as ``int64_t``, calls the
    embedded pybind wrapper ``meta.cpp_bind_name``, and writes
    ``context->SetOutputDataType(k, ...)`` for each output.
    """
    n_in = len(meta.param_names)
    inputs_cpp = "\n".join(_infer_dtype_input_block(i) for i in range(n_in))
    call_args = ", ".join(f"in{i}_dtype_value" for i in range(n_in))
    bind = meta.cpp_bind_name
    n_out = meta.n_outputs

    if n_out == 1:
        return f"""    PTO_CUSTOM_LOGD("Entering InferDataType\\n");
{inputs_cpp}
    int64_t out_dt_value = {bind}({call_args});
    context->SetOutputDataType(0, static_cast<ge::DataType>(out_dt_value));
    PTO_CUSTOM_LOGD("SetOutputDataType: index=0 value=%lld\\n", (long long)out_dt_value);
    PTO_CUSTOM_LOGD("Exiting InferDataType (status=%u)\\n", (unsigned)ge::GRAPH_SUCCESS);
    return ge::GRAPH_SUCCESS;
"""

    out_lines = "\n".join(
        f"    context->SetOutputDataType({k}, "
        f"static_cast<ge::DataType>(std::get<{k}>(out_dt_tuple)));\n"
        f'    PTO_CUSTOM_LOGD("SetOutputDataType: index={k} value=%lld\\n", '
        f"(long long)std::get<{k}>(out_dt_tuple));"
        for k in range(n_out)
    )
    return f"""    PTO_CUSTOM_LOGD("Entering InferDataType\\n");
{inputs_cpp}
    auto out_dt_tuple = {bind}({call_args});
{out_lines}
    PTO_CUSTOM_LOGD("Exiting InferDataType (status=%u)\\n", (unsigned)ge::GRAPH_SUCCESS);
    return ge::GRAPH_SUCCESS;
"""


# The registration inputs the plugin generator interpolates into C++ come from arbitrary ONNX model
# metadata, so each is validated at this boundary. The domain lands inside a C++ string literal and keeps
# dots (``ai.onnx.contrib``); the opset is emitted as a bare integer (``bool`` excluded, ``True`` would
# emit ``pypto::True::Op``); the framework type is a bare ``domi::FrameworkType`` enumerator token.
_PLUGIN_DOMAIN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _generate_op_custom_plugin_cpp(
    op_type: str,
    *,
    framework_type: str,
    domain: str = "pypto",
    domain_opset_version: int = 1,
    attr_specs=None,
) -> str:
    """Emit domi plugin registration.

    *framework_type* is the ``FrameworkType`` enum token (e.g. ``ONNX``).
    *domain* / *domain_opset_version* describe the ONNX node the registration must
    match: the CANN parser builds its lookup key as
    ``<domain>::<domain_opset_version>::<op_type>`` from each NodeProto and
    *requires* the registered ``OriginOpType`` to match that qualified form
    (verified against ``libfmk_onnx_parser.so``, a bare ``OriginOpType``
    is never matched). *domain_opset_version* is the opset assigned to the custom
    *domain* (NOT the base ai.onnx opset); ``torch.onnx.export`` imports the custom
    domain at opset 1 unless overridden via ``custom_opsets``.

    *attr_specs* is the op's declared compute attrs (``AttrSpec`` objects or ``{"name","type","default"}``
    dicts, in declaration order). When non-empty the ``ParseParam`` body parses the ONNX ``"attribute"``
    JSON and ``op_dest.SetAttr``s each matched attr (Int: name+``type==2`` with the null-guard), wrapped in
    a try/catch so a malformed attribute JSON returns ``FAILED`` instead of unwinding out of the domi
    callback; when empty the body is a no-op stub, so an attr-free op emits no JSON dependency at all.
    """
    validate_op_type_identifier(op_type)
    if not isinstance(domain, str) or not _PLUGIN_DOMAIN_RE.match(domain):
        raise ValueError(
            f"domain must match {_PLUGIN_DOMAIN_RE.pattern} -- it is emitted inside the OriginOpType C++ "
            f"string literal and is part of the parser's lookup key, got {domain!r}"
        )
    if not isinstance(domain_opset_version, int) or isinstance(domain_opset_version, bool) \
            or domain_opset_version < 1:
        raise ValueError(
            "domain_opset_version must be an int >= 1 (bool excluded) -- it is emitted into the "
            f"OriginOpType key, got {domain_opset_version!r}"
        )
    specs = _normalize_attr_specs(attr_specs)
    parse_fn = f"ParseParam{op_type}"
    origin_op_type = f"{domain}::{domain_opset_version}::{op_type}"

    if specs:
        # Real ParseParam: read the ONNX "attribute" JSON off op_src and SetAttr each declared attr onto
        # op_dest (name+type matched, order-independent, but generated from the ordered list for
        # determinism). Needs nlohmann/json.hpp (nlohmann::json), <string> (Float std::stof + String
        # std::string), and <vector> (ListInt std::vector<int64_t>).
        json_include = '#include <nlohmann/json.hpp>\n#include <string>\n#include <vector>\n'
        # One level deeper than the per-attr blocks are written: they sit inside the try below.
        parse_blocks = "\n".join(
            ("    " + line) if line.strip() else line
            for s in specs
            for line in _attr_parse_param_block(s).split("\n")
        )
        # The CANN parser calls this through a registered domi callback, so nothing may throw out of it:
        # nlohmann::json::parse / .get<> / std::stof all throw, and an exception crossing that boundary
        # aborts the parse with no diagnostic. Every failure becomes a logged FAILED return instead.
        parse_body = f"""    PTO_CUSTOM_LOGD("Entering {parse_fn}\\n");
    try {{
        ge::AscendString attrs_string;
        if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {{
            if (attrs_string.GetString() == nullptr) {{
                PTO_CUSTOM_LOGE("ERROR: {parse_fn}: null attribute string -> FAILED\\n");
                return FAILED;
            }}
            nlohmann::json attrs = nlohmann::json::parse(attrs_string.GetString());
            for (nlohmann::json attr : attrs["attribute"]) {{
{parse_blocks}
            }}
        }}
    }} catch (const nlohmann::json::exception &e) {{
        PTO_CUSTOM_LOGE("ERROR: {parse_fn}: attribute JSON error: %s -> FAILED\\n", e.what());
        return FAILED;
    }} catch (const std::exception &e) {{
        PTO_CUSTOM_LOGE("ERROR: {parse_fn}: %s -> FAILED\\n", e.what());
        return FAILED;
    }}
    PTO_CUSTOM_LOGD("Exiting {parse_fn} (status=%u)\\n", (unsigned)SUCCESS);
    return SUCCESS;"""
    else:
        json_include = ""
        parse_body = f"""    PTO_CUSTOM_LOGD("Entering {parse_fn}\\n");
    PTO_CUSTOM_LOGD("Exiting {parse_fn} (status=%u)\\n", (unsigned)SUCCESS);
    return SUCCESS;"""

    # Match real Ascend ``domi::ParseParamByOpFunc`` =
    # ``std::function<domi::Status(const ge::Operator&, ge::Operator&)>`` -
    # first arg is ``const ge::Operator&`` (NOT ``const Message*``).
    return f"""// Auto-generated

#include "register/register.h"
{json_include}#include <cstdio>
{_LOG_PREAMBLE}
namespace domi {{
// Parsing onnx params
Status {parse_fn}(const ge::Operator& op_src, ge::Operator& op_dest) {{
{parse_body}
}}

REGISTER_CUSTOM_OP("{op_type}")
    .FrameworkType({framework_type})
    .OriginOpType("{origin_op_type}")
    .ParseParamsByOperatorFn({parse_fn});
}}
"""


def _reg_op_block(
    infer_shape_func: Callable,
    infer_dtype_func: Callable,
    *,
    op_type: str,
    dtypes: list[Any],
    attr_specs=None,
) -> str:
    """Return the ``namespace ge { REG_OP(...) ... }`` op prototype block for *op_type* (no includes).

    Shared by the merged per-op TU (``_generate_op_tu_cpp_with_snippet``) and the standalone op-def TU
    the codegen tests assemble (``tests/ut/codegen_test_helpers.py``). One ``.INPUT``/``.OUTPUT`` line
    per input/output dtype; DT_* tokens are unqualified because the block lives inside ``namespace ge``.

    *attr_specs* (declaration order) add ``.REQUIRED_ATTR``/``.ATTR`` lines after the IO lines and before
    ``.OP_END_FACTORY_REG``, that order is the contract ``ExtractAttrs``'s positional index derives from.
    Empty => no attr lines (byte-identical to a pre-attrs op).
    """
    validate_op_type_identifier(op_type)
    if not dtypes:
        raise ValueError("dtypes must be a non-empty list")

    infer_shape_meta = _parse_infer_shape_for_codegen(infer_shape_func, attr_specs)
    infer_dtype_meta = _parse_infer_dtype_for_codegen(infer_dtype_func)
    n_out = infer_shape_meta.n_outputs
    if infer_dtype_meta.n_outputs != n_out:
        raise ValueError(
            f"infer_dtype output arity ({infer_dtype_meta.n_outputs}) must match "
            f"infer_shape output arity ({n_out})"
        )
    if len(infer_dtype_meta.param_names) != len(dtypes):
        raise ValueError(
            f"infer_dtype parameter count ({len(infer_dtype_meta.param_names)}) must "
            f"match the number of input dtypes ({len(dtypes)})"
        )

    # Codegen-time call to compute static OpDef Output().DataType({ge::DT_*}) declarations.
    # The runtime InferDataType callback is the source of truth at op execution time;
    # the OpDef declaration here reflects the dtypes seen at export time only.
    static_output_dtypes_raw = infer_dtype_func(*dtypes)
    if n_out == 1:
        static_output_dtypes = (static_output_dtypes_raw,)
    else:
        if not isinstance(static_output_dtypes_raw, tuple) or len(static_output_dtypes_raw) != n_out:
            raise TypeError(
                f"infer_dtype returned {static_output_dtypes_raw!r} at codegen time; "
                f"expected a tuple of {n_out} torch.dtype values"
            )
        static_output_dtypes = static_output_dtypes_raw
    for k, td in enumerate(static_output_dtypes):
        if not isinstance(td, torch.dtype):
            raise TypeError(
                f"infer_dtype output {k} returned {td!r} at codegen time; "
                "expected torch.dtype"
            )

    # REG_OP IO: one ``.INPUT(inN, TensorType({<dt>}))`` per input dtype and
    # one ``.OUTPUT(outK, TensorType({<dt>}))`` per (codegen-time) output dtype.
    # The block lives inside ``namespace ge {`` so dtype tokens are emitted
    # unqualified (strip the leading ``ge::`` from ``_torch_dtype_to_ge_dtype``).
    def _ge_dt_token(dt: Any) -> str:
        return _torch_dtype_to_ge_dtype(dt).removeprefix("ge::")

    reg_op_io_lines: list[str] = [
        f"    .INPUT(in{i}, TensorType({{{_ge_dt_token(dtype)}}}))"
        for i, dtype in enumerate(dtypes)
    ]
    reg_op_io_lines += [
        f"    .OUTPUT(out{k}, TensorType({{{_ge_dt_token(td)}}}))"
        for k, td in enumerate(static_output_dtypes)
    ]
    # Attr lines follow the IO lines, in declaration order (the declaration-index contract).
    reg_op_io_lines += [_attr_reg_op_line(s) for s in _normalize_attr_specs(attr_specs)]
    reg_op_io = "\n".join(reg_op_io_lines)

    # Op prototype only. Shape/dtype inference is NOT wired here: it lives on the executor as
    # ``ge::ShapeInferOp`` member methods (see ``_custom_executor_class_cpp``), which GE resolves
    # via ``CustomOpFactory`` + ``dynamic_cast<ShapeInferOp *>`` at compile time. ``IMPL_OP_INFERSHAPE``
    # is intentionally not used: a plain ``.so`` (dlopen, no op-package ``AddSoToRegistry``) never
    # populates the op-impl space registry it would target, so it would be inert.
    # ``ge::graphStatus`` is ``uint32_t`` in real Ascend (``graph/ge_error_codes.h``). Mirrors the
    # reference ``add_custom_pto`` ``op_host/add_custom.cpp`` (``graph/operator_reg.h`` + bare REG_OP).
    return f"""namespace ge {{
REG_OP({op_type})
{reg_op_io}
    .OP_END_FACTORY_REG({op_type});
}}
"""


def _custom_executor_class_cpp(
    op_type: str,
    *,
    infer_shape_meta: "_HelperFuncCodegenMeta",
    infer_dtype_meta: "_HelperFuncCodegenMeta",
    attr_specs=None,
) -> str:
    """Return C++ for the thin executor subclass named *op_type* plus ``REG_AUTO_MAPPING_OP``.

    The shared ``PtoCustomOp`` base (``pto_custom_op.{h,cpp}``) owns ``Compile()`` (import the shipped
    ``op_kernel/<stem>.py`` via the inline embed loader, call ``__pypto_compile``, parse the JSON launch
    sidecar into the compile cache) and ``DeclareLaunchArgs()`` (cache lookup + a generic, arity-driven
    ``AnnotatedKernelArgs`` build + ``AnnotatedKernelLaunchInfo`` + ``AddLaunch``). Launch is fully
    base-class-generic (ge 20260717 annotated-args launch), so this subclass emits no launch code -
    only the per-op hooks:

    - ``GetCompileModuleStem``, a unique module stem;
    - ``GetKernelPyBasename``, the shipped ``op_kernel/<stem>.py`` basename;
    - ``InferShape`` / ``InferDataType``, ``ge::ShapeInferOp`` member overrides (bodies from
      *infer_shape_meta* / *infer_dtype_meta*, calling the file-scope pybind infer wrappers). GE's
      compile-time custom-op shape inference resolves these via ``CustomOpFactory`` +
      ``dynamic_cast<ShapeInferOp *>`` (``OpDescUtilsEx::InferCustomOpShape``), that branch is tried
      *before* the op-impl space registry, and ``CustomOpFactory`` is populated by
      ``REG_AUTO_MAPPING_OP`` at plain-``.so`` dlopen, so no op-package / ``AddSoToRegistry`` is
      needed (``IMPL_OP_INFERSHAPE`` targets the op-impl space registry, which a plain ``.so`` never
      populates).

    The pybind infer wrappers themselves are emitted at file scope by
    ``_generate_custom_executor_cpp`` (before this class), so the member bodies can call them.

    Launch arity is read at runtime by the base class, so it is not a parameter here.
    """
    validate_op_type_identifier(op_type)
    stem = camel_case_to_snake_case(op_type)
    extract_attrs_method = _attr_extract_attrs_method(_normalize_attr_specs(attr_specs))
    infer_shape_body = _infer_shape_ge_impl_body(infer_shape_meta)
    infer_dtype_body = _infer_dtype_ge_impl_body(infer_dtype_meta)
    return f"""class {op_type} : public PtoCustomOp, public ge::ShapeInferOp {{
public:
    std::string GetCompileModuleStem() const override {{
        return "pypto_compile_{op_type}";
    }}

    // Basename of the shipped dev-editable kernel snippet (op_kernel/<stem>.py), the single source of
    // truth for this op's Python. The base resolves it via the ASCEND_CUSTOM_OPP_PATH entries
    // (<entry>/op_kernel/<basename>), with a dladdr upward-walk fallback, and loads it by path. The file
    // is REQUIRED, unresolved => FATAL. Emitted here so C++ uses the exact codegen basename.
    std::string GetKernelPyBasename() const override {{
        return "{stem}.py";
    }}
{extract_attrs_method}
    // GE compile-time shape/dtype inference: resolved via CustomOpFactory +
    // dynamic_cast<ShapeInferOp*> (OpDescUtilsEx::InferCustomOpShape), tried before the op-impl
    // space registry. Bodies call the file-scope pybind infer wrappers (Python self-bootstraps). A
    // propagated infer error (e.g. a strict deploy_file resolve failure) is caught here and turned into
    // GRAPH_FAILED so it never unwinds across the GE ABI; the catch re-acquires the GIL for PyErr_Print.
    ge::graphStatus InferShape(gert::InferShapeContext *context) override {{
        try {{
{infer_shape_body}
        }} catch (py::error_already_set &e) {{
            // what() BEFORE restore(): restore hands the error back to Python and what() is empty after.
            const std::string _what = e.what();
            {{ py::gil_scoped_acquire _g; e.restore(); PyErr_Print(); }}
            PTO_CUSTOM_LOGE("ERROR: {op_type}::InferShape %s -> GRAPH_FAILED\\n", _what.c_str());
            return ge::GRAPH_FAILED;
        }} catch (const std::exception &ex) {{
            PTO_CUSTOM_LOGE("ERROR: {op_type}::InferShape %s -> GRAPH_FAILED\\n", ex.what());
            return ge::GRAPH_FAILED;
        }}
    }}

    ge::graphStatus InferDataType(gert::InferDataTypeContext *context) override {{
        try {{
{infer_dtype_body}
        }} catch (py::error_already_set &e) {{
            // what() BEFORE restore(), as above.
            const std::string _what = e.what();
            {{ py::gil_scoped_acquire _g; e.restore(); PyErr_Print(); }}
            PTO_CUSTOM_LOGE("ERROR: {op_type}::InferDataType %s -> GRAPH_FAILED\\n", _what.c_str());
            return ge::GRAPH_FAILED;
        }} catch (const std::exception &ex) {{
            PTO_CUSTOM_LOGE("ERROR: {op_type}::InferDataType %s -> GRAPH_FAILED\\n", ex.what());
            return ge::GRAPH_FAILED;
        }}
    }}
}};

REG_AUTO_MAPPING_OP({op_type});
"""


def _generate_custom_executor_cpp(
    op_type: str,
    *,
    create_kernel_func: Callable | None = None,
    jit_kernel_func: Callable | None = None,
    infer_shape_func: Callable,
    infer_dtype_func: Callable,
    kernel_body_func: Callable | None = None,
    dtypes,
    factory_signature: str = "single",
    declared_annotations: list | None = None,
    mode: str = "factory",
    attr_specs=None,
    return_snippet: bool = False,
):
    """Assemble the thin executor TU: the pybind infer wrappers + the ``PtoCustomOp`` subclass.

    The shared ``PtoCustomOp`` base (shipped as ``pto_custom_op.{h,cpp}``) owns
    ``Compile()``/``DeclareLaunchArgs()``; this TU contributes only the per-op subclass (see
    ``_custom_executor_class_cpp``). The TU carries NO Python source: the op's Python lives solely
    in the shipped ``op_kernel/<stem>.py``, which ``Compile()`` and both infer wrappers import by path.
    Input count is ``len(dtypes)``; output arity is derived from ``infer_shape``. ``factory_signature``
    selects the factory call form (``"single"`` for homogeneous-rep factories; ``"lists"`` / ``"full"``
    for heterogeneous inputs).

    When *return_snippet* is True, return ``(tu_cpp, embedded_py_src)``, where *embedded_py_src* is the
    generated kernel snippet, so build.py can write it verbatim to ``op_kernel/<stem>.py`` with no
    re-derivation (drift-proof). Default False returns just the TU.
    """
    if not dtypes:
        raise ValueError("dtypes must be a non-empty list")
    infer_shape_meta = _parse_infer_shape_for_codegen(infer_shape_func, attr_specs)
    infer_dtype_meta = _parse_infer_dtype_for_codegen(infer_dtype_func)
    n_outputs = infer_shape_meta.n_outputs
    embedded_py_src = build_kernel_compile_snippet(
        create_kernel_func=create_kernel_func,
        jit_kernel_func=jit_kernel_func,
        infer_shape_func=infer_shape_func,
        infer_dtype_func=infer_dtype_func,
        n_inputs=len(dtypes),
        factory_signature=factory_signature,
        kernel_body_func=kernel_body_func,
        declared_annotations=declared_annotations,
        mode=mode,
    )
    # Shape/dtype inference lives on the executor (ge::ShapeInferOp member methods), so this TU carries
    # the pybind infer wrappers it calls. The base header (pto_custom_op.h) pulls in the GE custom_op
    # headers (incl. ShapeInferOp + Infer{Shape,DataType}Context), Python and PTO_CUSTOM_LOGD; we add
    # pybind + <tuple>/<vector> for the infer wrappers.
    # import_via="op_kernel": the wrappers import the infer func from the shipped
    # op_kernel/<basename>. The module *stem* MUST equal GetCompileModuleStem()
    # ("pypto_compile_<Op>") so Compile() and both wrappers share the ONE embed._loaded memo -> the .py
    # imports once; the *basename* is GetKernelPyBasename() ("<snake>.py"). Keep the two distinct.
    kernel_py_basename = f"{camel_case_to_snake_case(op_type)}.py"
    compile_module_stem = f"pypto_compile_{op_type}"
    infer_pybind = (
        _generate_pybind_wrapper(
            infer_shape_func, _CPP_BIND__INFER_SHAPE, import_via="op_kernel",
            basename=kernel_py_basename, stem=compile_module_stem,
        )
        + "\n\n"
        + _generate_pybind_wrapper(
            infer_dtype_func, _CPP_BIND__INFER_DTYPE, import_via="op_kernel",
            basename=kernel_py_basename, stem=compile_module_stem,
        )
    )
    tuple_inc = (
        "#include <tuple>\n" if (n_outputs > 1 or infer_dtype_meta.n_outputs > 1) else ""
    )
    preamble = f"""// Auto-generated

#include <cstdint>
#include <cstdio>
{tuple_inc}#include <string>
#include <vector>
#include "pto_custom_op.h"
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
"""
    class_cpp = _custom_executor_class_cpp(
        op_type,
        infer_shape_meta=infer_shape_meta,
        infer_dtype_meta=infer_dtype_meta,
        attr_specs=attr_specs,
    )
    tu_cpp = f"""{preamble}
{infer_pybind}

{class_cpp}"""
    if return_snippet:
        return tu_cpp, embedded_py_src
    return tu_cpp


def _generate_op_tu_cpp_with_snippet(
    op_type: str,
    *,
    create_kernel_func: Callable | None = None,
    jit_kernel_func: Callable | None = None,
    infer_shape_func: Callable,
    infer_dtype_func: Callable,
    kernel_body_func: Callable | None = None,
    dtypes,
    factory_signature: str = "single",
    declared_annotations: list | None = None,
    mode: str = "factory",
    attr_specs=None,
) -> tuple:
    """The single per-op ``op_host`` TU plus the generated kernel-compile snippet.

    The executor body is emitted by ``_generate_custom_executor_cpp``; the
    ``namespace ge { REG_OP(...) }`` prototype (``_reg_op_block``) is appended after it, and
    ``graph/operator_reg.h`` is pulled in next to the base header (``pto_custom_op.h``) so the appended
    block compiles. The two contribute complementary GE registrations keyed on the same *op_type*.

    Returns ``(tu_cpp, embedded_py_src)`` where *embedded_py_src* is the op's kernel snippet. build.py
    writes it verbatim to ``op_kernel/<stem>.py``, the shipped dev-editable file that ``Compile()`` and
    the infer wrappers import at op-compile time. *tu_cpp* itself carries no Python source.

    *attr_specs* (declaration order) drive the executor's ``ExtractAttrs`` override and the REG_OP attr
    lines from ONE ordered list.
    """
    executor, embedded_py_src = _generate_custom_executor_cpp(
        op_type,
        create_kernel_func=create_kernel_func,
        jit_kernel_func=jit_kernel_func,
        infer_shape_func=infer_shape_func,
        infer_dtype_func=infer_dtype_func,
        kernel_body_func=kernel_body_func,
        dtypes=dtypes,
        factory_signature=factory_signature,
        declared_annotations=declared_annotations,
        mode=mode,
        attr_specs=attr_specs,
        return_snippet=True,
    )
    base_include = '#include "pto_custom_op.h"\n'
    executor = executor.replace(
        base_include, base_include + '#include "graph/operator_reg.h"\n', 1
    )
    reg_op = _reg_op_block(
        infer_shape_func, infer_dtype_func, op_type=op_type, dtypes=dtypes, attr_specs=attr_specs,
    )
    return f"{executor}\n{reg_op}", embedded_py_src
