#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use this
# file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
# IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
# full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exception dump callback management for PyPTO Pro.

The C++ shim (``exception_dump_callback.cpp``) is compiled into
``libtile_fwk_interface.so`` at build time.  This module loads the
``extern "C"`` symbols from that library and provides a Python API to
register the callback and cache tensor info before each kernel launch.

On AICORE error, CANN asynchronously invokes the callback which writes
dump files (tensor data + ``_host.o``) to
``ASCEND_WORK_PATH/extra-info/data-dump/<device_id>/``.

The callback also automatically recompiles the JIT caller (``call_kernel.cpp``,
which includes ``kernel.cpp``) with ``-g`` via bisheng — same flags as the JIT
build, so the device code stays bit-identical to what was launched — and places
the resulting ``<kernel_name>_call_kernel.so`` in the dump directory. That one
artifact serves both offline reproduction (ctypes relaunch) and Error-PC
symbolization (fixedPCOffset -> source line). No manual intervention is
needed — the command is pre-constructed by :func:`set_dump_info` (called before
every kernel launch) and cached in the C++ shim for the callback to execute.
"""

from __future__ import annotations

import ctypes
import glob
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
from typing import TYPE_CHECKING

import torch

from pypto_pro import DataType

from .._errors import BadFd, PyptoProError

if TYPE_CHECKING:
    from pypto_pro.runtime.jit import CompiledKernel

_PL_DTYPE_TO_ACL_DTYPE: dict[str, int] = {
    str(DataType.FP32): 0,    # ACL_FLOAT
    str(DataType.FP16): 1,    # ACL_FLOAT16
    str(DataType.INT8): 2,    # ACL_INT8
    str(DataType.INT32): 3,   # ACL_INT32
    str(DataType.UINT8): 4,   # ACL_UINT8
    str(DataType.INT16): 6,   # ACL_INT16
    str(DataType.UINT16): 7,  # ACL_UINT16
    str(DataType.UINT32): 8,  # ACL_UINT32
    str(DataType.INT64): 9,   # ACL_INT64
    str(DataType.UINT64): 10, # ACL_UINT64
    str(DataType.BOOL): 12,   # ACL_BOOL
    str(DataType.BF16): 27,   # ACL_BF16
    str(DataType.FP8E5M2): 29,    # ACL_FP8_E5M2
    str(DataType.FP8E4M3FN): 30,  # ACL_FP8_E4M3FN
}

_MAX_DIMS = 8
_lib: ctypes.CDLL | None = None
_registered: bool = False


def _ensure_lib() -> ctypes.CDLL:
    """Load libtile_fwk_interface.so and resolve the extern "C" symbols."""
    global _lib
    if _lib is not None:
        return _lib

    import pypto

    lib_dir = Path(pypto.__file__).parent.resolve() / "lib"
    so_path = lib_dir / "libtile_fwk_interface.so"
    if not so_path.exists():
        raise BadFd(f"libtile_fwk_interface.so not found at {so_path}")

    _lib = ctypes.CDLL(str(so_path))

    _lib.pro_register_exception_dump_callback.restype = ctypes.c_int32
    _lib.pro_register_exception_dump_callback.argtypes = []

    _lib.pro_set_dump_info.restype = None
    _lib.pro_set_dump_info.argtypes = [
        ctypes.c_char_p,                   # kernelName
        ctypes.c_int32,                    # numTensors
        ctypes.POINTER(ctypes.c_int32),    # types
        ctypes.POINTER(ctypes.c_size_t),   # tensorSizes
        ctypes.POINTER(ctypes.c_int32),    # dataTypes
        ctypes.POINTER(ctypes.c_void_p),   # tensorAddrs
        ctypes.POINTER(ctypes.c_int64),    # flatShapes
        ctypes.POINTER(ctypes.c_int32),    # shapeCounts
        ctypes.c_int32,                    # maxDims
    ]

    _lib.pro_clear_dump_info.restype = None
    _lib.pro_clear_dump_info.argtypes = []

    _lib.pro_set_debug_cmd.restype = None
    _lib.pro_set_debug_cmd.argtypes = [ctypes.c_char_p]

    _lib.pro_set_launch_meta.restype = None
    _lib.pro_set_launch_meta.argtypes = [ctypes.c_char_p]
    return _lib


def register_callback() -> bool:
    """Register the exception dump callback with CANN. Returns True on success."""
    global _registered
    if _registered:
        return True
    try:
        lib = _ensure_lib()
    except PyptoProError as exc:
        logging.warning("Exception dump callback unavailable: %s", exc)
        return False
    ret = lib.pro_register_exception_dump_callback()
    if ret == 0:
        _registered = True
        logging.debug("Exception dump callback registered successfully")
    else:
        logging.warning("Failed to register exception dump callback (ret=%d)", ret)
    return _registered


def _dump_dir_path() -> str:
    """CANN brief_dump writes to ASCEND_WORK_PATH/extra-info/data-dump/<device_id>.

    Falls back to current working directory when ASCEND_WORK_PATH is not set,
    matching CANN's own default behavior.
    """
    work_path = os.environ.get("ASCEND_WORK_PATH", "") or os.getcwd()
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    return os.path.join(work_path, "extra-info", "data-dump", device_id)


def _build_debug_compile_cmd(compiled: "CompiledKernel") -> str:
    """Pre-construct the shell command for the C++ callback to execute.

    Returns a shell command string that:
      1. mkdir -p the dump directory
      2. recompiles call_kernel.cpp (which includes kernel.cpp) with -g and the
         exact same flags the JIT used, linking it into
         <dump_dir>/<kernel_name>_call_kernel.so

    The -g build carries DWARF for the device code while leaving the device
    .text bit-identical to the launched binary (verified: -g and linking never
    rewrite the embedded kernel image), so this single artifact serves BOTH
    offline reproduction (ctypes.CDLL + call_kernel relaunch) and Error-PC
    symbolization (fixedPCOffset -> source line via the embedded line table).

    Returns empty string if any prerequisite is missing (bisheng not found,
    call_kernel.cpp missing, env vars unset, etc.) so the callback skips
    compilation.
    """
    build_dir = compiled.build_dir
    kernel_name = compiled.kernel_name
    has_print_debug = compiled.needs_print_debug
    target = compiled.target
    jit_lib_path = compiled.lib_path

    if not build_dir:
        return ""
    caller_cpp = os.path.join(build_dir, "call_kernel.cpp")
    if not os.path.isfile(caller_cpp):
        return ""

    dump_dir = _dump_dir_path()
    so_path = os.path.join(dump_dir, f"{kernel_name}_call_kernel.so")

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        return ""

    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", ascend_home)

    bisheng = shutil.which("bisheng")
    if bisheng is None:
        return ""

    from pypto_pro.runtime.compile_config import get_jit_compile_config
    from pypto_pro.runtime.jit import get_current_arch

    arch = get_current_arch()

    cfg = get_jit_compile_config()

    if target is None:
        return ""

    # Mirror the JIT's own shared-library invocation (see jit._compile_shared_library):
    # same arch/fatobj/print-debug flags (all resolved from the same KernelTarget the
    # JIT used), same llvm args and link args, only -g and the output location differ.
    cmd_parts = [
        shlex.quote(bisheng),
        *[shlex.quote(f) for f in cfg.build_bisheng_flags(
            toolkit_home=toolkit_home,
            arch=arch,
            target=target,
            enable_print_debug=has_print_debug,
        )],
        *[shlex.quote(f) for f in cfg.runtime_include_flags(ascend_home)],
        *[shlex.quote(f) for f in cfg.build_llvm_args(arch)],
    ]
    cmd_parts.append("-g")
    cmd_parts.append(shlex.quote(caller_cpp))
    cmd_parts.extend(shlex.quote(f) for f in cfg.runtime_link_args(ascend_home))
    cmd_parts.extend(["-o", shlex.quote(so_path)])

    cmd = f"mkdir -p {shlex.quote(dump_dir)} && {' '.join(cmd_parts)}"

    # Copy the kernel sources next to the .so so the dump directory is fully
    # self-contained: the repro tool and the user never need the (ephemeral)
    # build directory to read the source lines the Error PC resolves to.
    # *.h matters as much as *.cpp here: kernels with a tiling struct include
    # <Tiling>_tiling.h from kernel.cpp, so without it the -g rebuild cannot
    # be replayed from the dump directory alone.
    src_artifacts = sorted(
        glob.glob(os.path.join(build_dir, "*.cpp"))
        + glob.glob(os.path.join(build_dir, "*.h"))
    )
    if src_artifacts:
        cmd += " && cp " + " ".join(shlex.quote(f) for f in src_artifacts) + f" {shlex.quote(dump_dir)}/"

    # Preserve the original JIT-launched library (call_kernel_<digest>.so) as
    # well: the -g rebuild is verified bit-identical in .text, but keeping the
    # exact launched bytes allows audit/compare and a highest-fidelity repro
    # path that does not depend on that verification.
    if jit_lib_path and os.path.isfile(jit_lib_path):
        cmd += f" && cp {shlex.quote(jit_lib_path)} {shlex.quote(dump_dir)}/"

    return cmd


def _record_launch_meta(compiled: "CompiledKernel", args: tuple, block_dim: int | None) -> None:
    """Cache the launch ABI (ptr/scalar/tiling args + block_dim) in the C++ shim.

    Same pattern as the tensor info (``pro_set_dump_info``): the meta is held
    in the shim and only persisted to
    ``<dump_dir>/<kernel>_launch_args.json`` when the exception-dump callback
    fires (``WriteLaunchMetaFile``), so nothing touches the dump directory at
    launch time. The CANN dump data file only carries tensor payloads;
    standalone scalar args (e.g. int64 m/k/n) and the launch block_dim cannot
    be recovered from the call_kernel.cpp signature (which only exposes
    shape-derived dyn args), so the ABI is recorded here, in signature order,
    for the offline repro tool to rebuild the exact call.
    """
    if not getattr(compiled, "kernel_name", ""):
        return

    abi = []
    for spec, arg in zip(getattr(compiled, "param_specs", None) or [], args):
        kind = spec.kind.name
        if kind in ("TENSOR", "PTR", "TILING"):
            if isinstance(arg, torch.Tensor):
                abi.append({
                    "name": spec.name,
                    "kind": "ptr",
                    "dtype": str(spec.dtype) if spec.dtype else "uint8",
                    "addr": arg.data_ptr(),
                    "size": arg.numel() * arg.element_size(),
                    "shape": list(arg.shape),
                })
            else:
                abi.append({"name": spec.name, "kind": "ptr", "dtype": "uint8", "addr": 0, "size": 0, "shape": []})
        elif kind == "SCALAR":
            value = arg if isinstance(arg, (int, float, bool)) else str(arg)
            abi.append({"name": spec.name, "kind": "scalar", "value": value})

    meta = {"kernel_name": compiled.kernel_name, "block_dim": block_dim, "abi": abi}
    _ensure_lib().pro_set_launch_meta(json.dumps(meta).encode("utf-8"))


def set_dump_info(compiled: "CompiledKernel", args: tuple, block_dim: int | None = None) -> None:
    """Cache tensor info from runtime args before kernel launch.

    Everything the exception-dump callback needs about the kernel's identity
    and artifacts (name, build dir, arch mix, launched .so) is read from the
    CompiledKernel; ``args`` is the only per-launch input. ``block_dim`` and
    the scalar members of ``args`` are cached in the C++ shim via
    :func:`_record_launch_meta` and only persisted to ``launch_args.json``
    when the exception-dump callback fires (``WriteLaunchMetaFile``), so
    nothing touches the dump directory at launch time. The offline repro tool
    reads the sidecar to rebuild the exact launch ABI.
    """
    if not compiled.build_dir or not os.path.isdir(compiled.build_dir):
        logging.warning("exception dump skipped: not a jit scenario (build_dir unavailable)")
        return

    if not register_callback():
        logging.warning("register_callback failed, exception dump will not be triggered")
        return

    types: list[int] = []
    sizes: list[int] = []
    dtypes: list[int] = []
    addrs: list[int] = []
    flat_shapes: list[int] = []
    shape_counts: list[int] = []

    for arg, spec in zip(args, compiled.param_specs):
        kind_name = spec.kind.name
        if kind_name in ("TENSOR", "PTR"):
            if arg is None:
                continue
            if not isinstance(arg, torch.Tensor):
                continue
            types.append(0)  # INPUT
            sizes.append(arg.numel() * arg.element_size())
            acl_dtype = _PL_DTYPE_TO_ACL_DTYPE.get(str(spec.dtype), 0)
            dtypes.append(acl_dtype)
            addrs.append(arg.data_ptr())
            shape = list(arg.shape)
            shape_counts.append(len(shape))
            padded = shape + [0] * (_MAX_DIMS - len(shape))
            flat_shapes.extend(padded[:_MAX_DIMS])
        elif kind_name == "TILING":
            if arg is None or not isinstance(arg, torch.Tensor):
                continue
            types.append(2)  # WORKSPACE
            sizes.append(arg.numel() * arg.element_size())
            dtypes.append(4)  # ACL_UINT8
            addrs.append(arg.data_ptr())
            shape_counts.append(0)
            flat_shapes.extend([0] * _MAX_DIMS)

    num_tensors = len(types)

    lib = _ensure_lib()

    _record_launch_meta(compiled, args, block_dim)

    lib.pro_set_debug_cmd(_build_debug_compile_cmd(compiled).encode("utf-8"))

    if num_tensors == 0:
        return

    c_types = (ctypes.c_int32 * num_tensors)(*types)
    c_sizes = (ctypes.c_size_t * num_tensors)(*sizes)
    c_dtypes = (ctypes.c_int32 * num_tensors)(*dtypes)
    c_addrs = (ctypes.c_void_p * num_tensors)(*addrs)
    c_shapes = (ctypes.c_int64 * (num_tensors * _MAX_DIMS))(*flat_shapes)
    c_counts = (ctypes.c_int32 * num_tensors)(*shape_counts)

    lib.pro_set_dump_info(
        compiled.kernel_name.encode("utf-8"),
        num_tensors,
        c_types,
        c_sizes,
        c_dtypes,
        c_addrs,
        c_shapes,
        c_counts,
        _MAX_DIMS,
    )


def clear_dump_info() -> None:
    """Clear cached tensor info."""
    global _lib
    if _lib is not None:
        _lib.pro_clear_dump_info()
