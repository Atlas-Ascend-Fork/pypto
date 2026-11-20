# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AICPU fast-launch glue generator + compile pipeline (extracted from ``runtime/jit.py``).

Source generation around the shared CCE body (all IR->C++ emission is done by
``CCECodegen``; this module owns only the AICPU-specific glue):

* **bare body** (``generate_aicpu_bare_body``): the shared codegen's
  ``aicpu_bare=True`` mode emits the impl without device
  qualifiers (``__gm__``/``__aicore__``/``__aicpu__``) and without the CCE
  TU scaffolding (``#if __DAV_VEC__`` guard, SyncedTile template, pypto
  framework headers); ``_validate_bare_body`` enforces that contract. The
  impl returns the AICPU status ABI (``uint32_t``, 0 = KERNEL_STATUS_OK,
  1 = KERNEL_STATUS_PARAM_INVALID): an early bare ``return`` in the kernel
  body is a failed check and emits ``return 1;``, the normal fall-through
  emits ``return 0;``.
* **kernel TU** (``kernel_source``): CCE body + the flattened
  ``__global__ __aicpu__ uint32_t Kernel(<flat params>)`` entry that returns
  the impl's status; a nonzero status marks the AICPU task failed and the
  host observes it as a stream error at synchronize time.
* **host TU** (``host_source``, self-contained): compatibility macros
  that erase CCE device qualifiers for the AICPU host compile, the flat
  ``Kernel`` entry declaration (no Args struct -- the launch passes every
  argument individually), and the ``Run`` C-ABI shim that forwards the
  flattened arguments to ``Kernel<<<1, nullptr, (aclrtStream)stm>>>(args...)``.

Compile pipeline: CANN install probing, the ``-x aicpu`` kernel TU
cross-compile, the ``--asc-aicore-lang`` host TU, the shared link, and the
per-variant orchestration that writes the generated sources and installs
the compiled kernel into the cache. ``jit.py`` keeps the recompile decision
and the launch entry. The offline CpuKernel delivery glue lives in
``opc/aclnn_aicpu_codegen.py``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re as _re
import subprocess

from pypto.pypto_impl import ir
from pypto.pypto_impl.codegen import CCECodegen
from pypto_pro.runtime.platform import NpuArch

from .._errors import InvalidOperation, RuntimeFailure
from .entry_params import entry_params_from_param_specs

__all__ = [
    "generate_aicpu_bare_body",
    "generate_aicpu_sources",
    "aicpu_cann_prefixes",
    "run_aicpu_build_step",
    "build_aicpu_jit_so",
    "compile_aicpu_variant",
    "reject_unsupported_aicpu_params",
]


# Word-boundary patterns: a leaked qualifier must be caught however it is spelled
# ("__gm__ x", "__gm__\n", "(__gm__*p)", trailing "__gm__"), not only with a space.
_FORBIDDEN_PATTERNS = tuple(
    _re.compile(_re.escape(token.strip()) + r"(?!\w)")
    for token in ("__gm__ ", "__aicore__ ", "__aicpu__ ", "#if defined(__DAV", "SyncedTile", "pypto_dump.h")
)


def _validate_bare_body(body: str, op_name: str) -> None:
    """Assert the bare-body contract: no device qualifiers, no CCE scaffolding.

    The aicpu_bare codegen mode never emits these; a hit means a reachable emitter
    bypassed GlobalMemoryQualifier()/ImplHeadQualifier() or the bare assembly branch, and the output
    would break under ``bisheng -x aicpu`` (where they are keywords, not macros).
    """
    leaked = [pattern.pattern for pattern in _FORBIDDEN_PATTERNS if pattern.search(body)]
    if leaked:
        raise InvalidOperation(
            f"bare codegen for '{op_name}' leaked {leaked}: a reachable emitter bypassed "
            f"GlobalMemoryQualifier()/ImplHeadQualifier() or the bare assembly branch"
        )


def generate_aicpu_bare_body(program, arch, op_name: str) -> str:
    """Single owner of the bare-body contract, shared by fast-launch and offline delivery.

    Asks the shared CCE codegen for an ``aicpu_bare`` TU (bare impl: no device
    qualifiers, no DAV guard/SyncedTile/framework headers; struct type definitions
    and the impl function only), then enforces the contract. The caller wraps the
    body in its own entry + compatibility header.
    """
    body = CCECodegen(ir.SectionKind.Vector).generate_single(program, arch)
    _validate_bare_body(body, op_name)
    return body


def _impl_name(op_name: str) -> str:
    """Bare impl symbol: CCE emits <name>_impl_<section> and AICPU bare mode is Vector-only."""
    return f"{op_name}_impl_vector"


def generate_aicpu_sources(cce_body: str, param_specs, op_name: str) -> dict[str, str]:
    """Generate the glue files around a CCE-generated kernel body.

    Args:
        cce_body: The cleaned CCE codegen output (impl function with body).
        param_specs: The kernel's ParamSpec list (parameter metadata).
        op_name: The kernel name used for C++ symbols and file names.

    Returns:
        Dict with keys ``kernel_source``, ``host_source``. The host TU is
        self-contained: it embeds the former compat-header content (qualifier
        erasure macros + the flat ``Kernel`` entry declaration), so fast-launch
        delivery is two files instead of three.
    """
    reject_unsupported_aicpu_params(param_specs, "fast-launch")
    entry = entry_params_from_param_specs(param_specs)
    fields = [(f"{typ}*" if is_ptr else typ, name) for typ, name, is_ptr in entry]

    impl_name = _impl_name(op_name)

    # ── Flattened launch ABI: every parameter passes through <<<>>> directly,
    # no packed Args struct on either side of the launch. ──
    flat_params = ", ".join(f"{typ} {name}" for typ, name in fields)
    pass_args = ", ".join(name for _typ, name in fields)

    # ── Kernel TU: CCE body + flat-argument entry (self-contained) ──
    kernel_source = [
        "// Generated by PyPTO Pro AICPU: CCE body + flat-argument entry.",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <algorithm>",
        "using namespace std;",
        "",
        "// Erase AICPU entry qualifiers when the compiler does not define them",
        "// (no-op under bisheng -x aicpu, where they are native keywords).",
        "#if !defined(__global__)",
        "#define __global__",
        "#endif",
        "#if !defined(__aicpu__)",
        "#define __aicpu__",
        "#endif",
        "",
        "// Kernel body generated by the shared CCE codegen.",
        cce_body,
        "",
        "// Entry: the flattened launch form bisheng -x aicpu expects for __global__ __aicpu__.",
        f'extern "C" __global__ __aicpu__ uint32_t {op_name}Kernel({flat_params}) {{',
        "// AICPU status ABI: propagate the impl's KERNEL_STATUS return (0 = OK,",
        "// 1 = PARAM_INVALID); the AICPU runtime marks a nonzero task status failed,",
        "// which the host observes as a stream error at synchronize time.",
        f"    return {impl_name}({pass_args});",
        "}",
        "",
    ]

    # ── Host TU: compat macros + flat entry declaration + Run shim ──
    host_source = [
        "// Generated by PyPTO Pro AICPU glue: compat macros + flat entry decl + host launch TU.",
        "#include <cstdint>",
        '#include "acl/acl_base.h"',
        "",
        "// Erase AICPU entry qualifiers when the compiler does not define them.",
        "#if !defined(__global__)",
        "#define __global__",
        "#endif",
        "#if !defined(__aicpu__)",
        "#define __aicpu__",
        "#endif",
        "",
        "extern \"C\" {",
        f"extern __global__ __aicpu__ uint32_t {op_name}Kernel({flat_params});",
        "}",
        "",
        f'extern "C" uint32_t {op_name}Run({flat_params}, void* stm) {{',
        "    // fast-launch: dispatch the AICPU binary carried in the linked",
        "    // .aicpu_binary section; the <<<>>> lowering resolves the symbol and",
        "    // launches on the stream with the arguments flattened (no packed",
        "    // Args struct). The launch is asynchronous; the kernel's",
        "    // uint32_t KERNEL_STATUS return does not travel back through this",
        "    // shim -- a nonzero status marks the task failed and surfaces as a",
        "    // stream error when the caller synchronizes.",
        f"    {op_name}Kernel<<<1, nullptr, static_cast<aclrtStream>(stm)>>>({pass_args});",
        "    return 0;",
        "}",
        "",
    ]

    return {
        "kernel_source": "\n".join(kernel_source),
        "host_source": "\n".join(host_source),
    }


def aicpu_cann_prefixes(home: str) -> list[str]:
    """The ``<prefix>-linux`` library prefixes of the CANN install at ``home``."""
    if not home:
        return []
    prefixes = []
    for prefix in ("x86_64", "aarch64"):
        if Path(home, f"{prefix}-linux").is_dir():
            prefixes.append(f"{prefix}-linux")
    return prefixes


def run_aicpu_build_step(what: str, command: list, kernel_name: str, timeout: int) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeFailure(f"AICPU {what} compiler '{command[0]}' was not found") from exc
    if result.returncode != 0:
        raise RuntimeFailure(
            f"AICPU {what} failed for kernel '{kernel_name}' (rc={result.returncode})\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def build_aicpu_jit_so(out_dir: str, op_name: str, cpp_path: Path, host_path: Path,
                       so_path: Path, arch: "NpuArch", timeout: int) -> None:
    """AICPU build: bisheng AICPU kernel TU + ASC host TU + shared link.

    Mirrors the reference ``fast_kernel_launch_example`` toolchain: the kernel
    TU is cross-compiled with ``-x aicpu`` (the AICPU binary lands in the
    ``.aicpu_binary`` section of the object), the host TU with
    ``--asc-aicore-lang`` so ``Kernel<<<1, nullptr, stm>>>(args, size)``
    lowers to the ACL runtime dispatch, and the link pulls in
    ``libascendcl``/``libascendc_runtime`` for the lowering and the kernel
    registration stub.
    """
    home = os.environ.get("ASCEND_HOME_PATH", "")
    prefixes = aicpu_cann_prefixes(home)
    if not prefixes:
        raise RuntimeFailure(
            "AICPU build needs ASCEND_HOME_PATH with a CANN install "
            "(no <prefix>-linux directory found)"
        )
    prefix = prefixes[0]
    sp = Path(home, prefix)
    hcc = Path(home, "toolkit", "toolchain", "hcc")
    if not hcc.is_dir():
        raise RuntimeFailure(
            f"AICPU build needs the hcc toolchain under '{hcc}' (not found)"
        )
    bisheng = str(Path(home, "bin", "bisheng"))

    # Kernel TU: the CANN CMakeAICPUInformation.cmake flag set.
    aicpu_cxx_includes = [
        Path(hcc, "aarch64-target-linux-gnu", "include"),
        Path(hcc, "aarch64-target-linux-gnu", "include", "c++", "7.3.0"),
        Path(hcc, "aarch64-target-linux-gnu", "include", "c++", "7.3.0", "aarch64-target-linux-gnu"),
        Path(hcc, "aarch64-target-linux-gnu", "include", "c++", "7.3.0", "backward"),
        Path(hcc, "aarch64-target-linux-gnu", "include", "c++", "14.3.0"),
        Path(hcc, "aarch64-target-linux-gnu", "include", "c++", "14.3.0", "aarch64-target-linux-gnu"),
        Path(hcc, "aarch64-target-linux-gnu", "include", "c++", "14.3.0", "backward"),
    ]
    kernel_obj = Path(out_dir, f"{op_name}_aicpu_kernel.o")
    kernel_cmd = [
        bisheng, "-O2", "-c", "-std=c++17",
        "-fvisibility=default", "-fvisibility-inlines-hidden",
        "-D_GLIBCXX_USE_CXX11_ABI=0", "-D_FORTIFY_SOURCE=2", "-D_GNU_SOURCE",
        f"-I{sp / 'asc' / 'include' / 'aicpu_api'}",
        f"--cce-aicpu-L{sp / 'lib64' / 'device' / 'lib64'}",
        "--cce-aicpu-laicpu_api",
        # Interface launch mode: the flat multi-arg entry needs it (legacy mode admits single-pointer kernels only).
        "--cce-aicpu-launch-with-interface",
        f"--cce-aicpu-toolkit-path={hcc / 'bin'}",
        f"--cce-aicpu-sysroot={hcc / 'sysroot'}",
        *[arg for p in aicpu_cxx_includes for arg in ("-isystem", str(p))],
        "-o", str(kernel_obj), "-x", "aicpu", str(cpp_path),
    ]
    run_aicpu_build_step("kernel compile", kernel_cmd, op_name, timeout)

    # Host TU: the ASC dialect understands <<<>>> on the __aicpu__ entry.
    npu_arch = arch.name.lower().replace("_", "-") if hasattr(arch, "name") else "dav-3510"
    host_obj = Path(out_dir, f"{op_name}_aicpu_host.o")
    host_cmd = [
        bisheng, "-O2", "-std=c++17", "-fPIC", "--asc-aicore-lang",
        # Keep the host <<<>>> lowering in the same interface launch mode as the kernel TU.
        "--cce-aicpu-launch-with-interface",
        f"--npu-arch={npu_arch}",
        f"-I{sp / 'include'}",
        f"-I{Path(home, 'include')}",
        f"-I{out_dir}",
        "-o", str(host_obj), "-c", str(host_path),
    ]
    run_aicpu_build_step("host compile", host_cmd, op_name, timeout)

    # Link: the kernel object registers its .aicpu_binary with the ACL runtime;
    # the host object's <<<>>> lowering resolves it at launch time.
    link_cmd = [
        bisheng, "-shared", "-fPIC", "-std=c++17",
        str(kernel_obj), str(host_obj), "-o", str(so_path),
        f"-L{sp / 'lib64'}", "-lascendcl", "-lascendc_runtime",
    ]
    run_aicpu_build_step("link", link_cmd, op_name, timeout)


def _parse_aicpu_program(kernel, concrete_key=None, datatype_consts=None, bound_signature=None):
    """Parse an AICPU kernel to its Vector KernelDef, program and ParamSpec list.

    Shared front-end chain for both deliveries (fast-launch and offline CpuKernel):
    KernelDef -> Vector program -> param specs. The fast-launch path passes its
    variant key, dtype consts and bound signature; the offline path keeps the
    defaults (single specialization, no tiling key).
    """
    # jit imports this module at load time; resolve its spec extractor lazily.
    from .jit import _extract_param_specs

    kernel_def = kernel.to_kernel_def(concrete_key, datatype_consts=datatype_consts)
    program, _ = kernel_def.parse_target_program(
        ir.SectionKind.Vector, bound_signature=bound_signature
    )
    param_specs = _extract_param_specs(program, kernel_def.last_param_directions)
    return kernel_def, program, param_specs


def compile_aicpu_variant(kernel, concrete_key, spec, bound_signature, static_signature):
    """Build one AICPU kernel variant in the fast_kernel_launch_example shape.

    The generated set is ``<op>_aicpu.cpp`` (kernel body +
    ``__global__ __aicpu__`` flat-argument entry) and
    ``<op>_aicpu_host.cpp`` (self-contained: qualifier erasure macros, the flat
    ``Kernel`` entry declaration, and the ``Run`` C-ABI shim). The build is the
    fast-launch toolchain (the reference ``fast_kernel_launch_example``
    mechanics): the kernel TU is cross-compiled with bisheng ``-x aicpu``
    embedding the AICPU binary in ``.aicpu_binary``; the host TU is compiled
    with ``--asc-aicore-lang`` so its ``Run`` shim launches
    ``Kernel<<<1, nullptr, stm>>>(args...)`` with the arguments flattened -- the bisheng
    lowering resolves the symbol via ``aclrtGetFuncBySymbol`` and
    dispatches through ``aclrtLaunchKernelWithArgsArray`` on the given
    stream. Tensors must live on the NPU; the launch is asynchronous.
    """
    from importlib import import_module

    _jit = import_module("pypto_pro.runtime.jit")

    tilingkey_packed, dtype_consts, dtype_metadata, dtype_hash = spec
    arch = kernel._arch or NpuArch.DAV_3510
    kernel_def, program, param_specs = _parse_aicpu_program(
        kernel, concrete_key, datatype_consts=dtype_consts, bound_signature=bound_signature
    )

    op_name = _jit._get_kernel_ir_function(program).name

    # Kernel body via the shared CCE codegen (all IR→C++ emission), bare mode.
    cce_body = generate_aicpu_bare_body(program, arch, op_name)
    sources = generate_aicpu_sources(cce_body, param_specs, op_name)
    build_dir = _jit._make_artifact_build_dir(
        kernel_def, arch,
        tilingkey_suffix=_jit._static_signature_suffix(static_signature),
    )
    out_dir = _jit._make_tilingkey_dir(build_dir, tilingkey_packed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cpp_path = Path(out_dir, f"{op_name}_aicpu.cpp")
    host_path = Path(out_dir, f"{op_name}_aicpu_host.cpp")
    _jit._atomic_write_text(cpp_path, sources["kernel_source"])
    _jit._atomic_write_text(host_path, sources["host_source"])
    # The .so name embeds a source digest: distinct source sets (e.g. different
    # static shapes folding different constants) get distinct libraries -- a
    # shared flat name overwritten in place would keep the old inode mapped in
    # this process, and dlopen would then silently run the first variant's code
    # for every later variant.
    so_digest = hashlib.sha256(
        (sources["kernel_source"] + sources["host_source"]).encode()
    ).hexdigest()[:12]
    so_path = Path(out_dir, f"lib{op_name}_aicpu_{so_digest}.so")

    build_aicpu_jit_so(
        str(out_dir), op_name, cpp_path, host_path, so_path, arch,
        kernel._compile_timeout or 600,
    )

    compiled = _jit.CompiledKernel(
        lib_path=str(so_path),
        param_specs=param_specs,
        kernel_name=op_name,
        build_dir=str(out_dir),
        aicpu=True,
        source_file=getattr(kernel_def, "_source_file", ""),
    )
    kernel._compiled_by_signature[(static_signature, dtype_hash, tilingkey_packed)] = compiled
    return compiled


def reject_unsupported_aicpu_params(param_specs, delivery: str, param_names=None) -> None:
    """Reject parameter forms no AICPU delivery can express (shared by all entrypoints).

    TENSOR and PTR travel identically in both deliveries: both represent tensors
    with no optional/required distinction -- the launch packs a None argument as
    a null pointer, the offline adapter maps both to the same null-tolerant
    Input/Output edge, and the kernel body's ``is not None`` guards own the
    presence judgment. TILING/WINDOW keep their dedicated fast-launch ABI and
    have no offline channel, so both kinds stay rejected everywhere. When
    ``param_names`` is given it must align 1:1 with ``param_specs`` (source
    order). ``delivery`` names the calling path in the error message (e.g.
    "fast-launch", "offline delivery").
    """
    from .._errors import InvalidVal

    for kind in ("TILING", "WINDOW"):
        bad = [spec.name for spec in param_specs if spec.kind.name == kind]
        if bad:
            raise InvalidVal(
                f"AICPU {delivery} does not support {kind} parameter(s): {bad}"
            )
    if param_names is not None and len(param_names) != len(param_specs):
        raise InvalidVal(
            f"param_names must align 1:1 with param_specs for AICPU {delivery} "
            f"(source order): got {len(param_names)} names vs {len(param_specs)} specs"
        )
