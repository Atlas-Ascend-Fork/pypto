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
register the callback and the kernel's static launch schema.

On AICORE error, CANN asynchronously invokes the callback which writes
dump files (tensor data + ``_host.o``) to
``ASCEND_WORK_PATH/extra-info/data-dump/<device_id>/``.

The callback also automatically recompiles the JIT caller (``call_kernel.cpp``,
which includes ``kernel.cpp``) with ``-g`` via bisheng — same flags as the JIT
build, so the device code stays bit-identical to what was launched — and places
the resulting ``<kernel_name>_call_kernel.so`` in the dump directory. That one
artifact serves both offline reproduction (ctypes relaunch) and Error-PC
symbolization (fixedPCOffset -> source line). No manual intervention is
needed — the command and Tensor/scalar ABI schema are registered when the
launch closure is built, and re-registered if another compiled variant with
the same kernel name becomes active. Pointer-like allocation sizes and block
dim are only refreshed when their values change.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from pathlib import Path
import shlex
import shutil
import threading
from typing import TYPE_CHECKING

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
    str(DataType.HF8): 34,    # ACL_HIFLOAT8
    str(DataType.FP8E5M2): 35,    # ACL_FLOAT8_E5M2
    str(DataType.FP8E4M3FN): 36,  # ACL_FLOAT8_E4M3FN
    str(DataType.FP8E8M0): 37,    # ACL_FLOAT8_E8M0
    str(DataType.FP4E2M1): 40,  # ACL_FLOAT4_E2M1
    str(DataType.FP4E1M2): 41,  # ACL_FLOAT4_E1M2
}

_MAX_DIMS = 8
_lib: ctypes.CDLL | None = None
_registered: bool = False
_dump_schema_lock = threading.RLock()
_active_dump_schemas: dict[str, tuple[CompiledKernel, tuple[int, tuple[int, ...]] | None]] = {}


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

    _lib.pro_register_dump_schema.restype = ctypes.c_int32
    _lib.pro_register_dump_schema.argtypes = [
        ctypes.c_char_p,                   # kernelName
        ctypes.c_int32,                    # numParams
        ctypes.POINTER(ctypes.c_int32),    # kinds
        ctypes.POINTER(ctypes.c_int32),    # tensorTypes
        ctypes.POINTER(ctypes.c_int32),    # dataTypes
        ctypes.POINTER(ctypes.c_int32),    # elementBits
        ctypes.POINTER(ctypes.c_char_p),   # names
        ctypes.POINTER(ctypes.c_char_p),   # dtypeNames
        ctypes.POINTER(ctypes.c_int32),    # shapeCounts
        ctypes.POINTER(ctypes.c_int64),    # flatShapes
        ctypes.POINTER(ctypes.c_int32),    # flatDynamicSlots
        ctypes.c_char_p,                   # debugCmd
    ]

    _lib.pro_unregister_dump_schema.restype = ctypes.c_int32
    _lib.pro_unregister_dump_schema.argtypes = [ctypes.c_char_p]

    _lib.pro_set_dump_runtime_info.restype = ctypes.c_int32
    _lib.pro_set_dump_runtime_info.argtypes = [
        ctypes.c_char_p,                   # kernelName
        ctypes.c_uint32,                   # blockDim
        ctypes.c_int32,                    # numSizes
        ctypes.POINTER(ctypes.c_size_t),   # argSizes
    ]

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
    # the same arch/fatobj/print-debug flags and include roots all come from one builder,
    # and the same llvm args / link args are used; only -g and the output location differ.
    cmd_parts = [
        shlex.quote(bisheng),
        *[shlex.quote(f) for f in cfg.build_bisheng_flags(
            toolkit_home=toolkit_home,
            arch=arch,
            target=target,
            enable_print_debug=has_print_debug,
            sk_cap=compiled.sk_cap,
        )],
        *[shlex.quote(f) for f in cfg.build_llvm_args(arch)],
    ]
    cmd_parts.append("-g")
    cmd_parts.append(shlex.quote(caller_cpp))
    cmd_parts.extend(shlex.quote(f) for f in cfg.runtime_link_args())
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


def prepare_dump_schema(compiled: "CompiledKernel") -> tuple[int, ...] | None:
    """Register launch-invariant dump metadata once for the exception callback.

    The returned tuple contains pointer-like parameter indices whose allocation
    sizes must be refreshed when the launch shape changes. ``None`` means the
    kernel ABI cannot currently be reconstructed by the exception callback.
    """
    if not compiled.build_dir or not os.path.isdir(compiled.build_dir):
        logging.warning(
            "Cannot register exception-dump schema for %s: invalid build directory %r",
            compiled.kernel_name,
            compiled.build_dir,
        )
        return None
    if not register_callback():
        return None

    specs = compiled.param_specs
    if any(spec.kind.name not in ("TENSOR", "SCALAR", "PTR", "TILING", "WINDOW") for spec in specs):
        return None

    from pypto_pro.runtime.jit import _collect_dyn_vars

    dynamic_vars = _collect_dyn_vars(specs)
    dynamic_slots = {name: len(specs) + index for index, name in enumerate(dynamic_vars)}

    kinds: list[int] = []
    tensor_types: list[int] = []
    data_types: list[int] = []
    element_bits: list[int] = []
    names: list[bytes] = []
    dtype_names: list[bytes] = []
    shape_counts: list[int] = []
    flat_shapes: list[int] = []
    flat_dynamic_slots: list[int] = []

    size_indices: list[int] = []
    for index, spec in enumerate(specs):
        kind_name = spec.kind.name
        is_tiling = kind_name == "TILING"
        dtype = DataType.UINT8 if kind_name in ("TILING", "WINDOW") else spec.dtype
        acl_dtype = _PL_DTYPE_TO_ACL_DTYPE.get(str(dtype))
        if acl_dtype is None:
            return None
        is_tensor = kind_name == "TENSOR"
        shape = list(spec.shape or [])
        if len(shape) > _MAX_DIMS:
            return None
        static_shape: list[int] = []
        dim_slots: list[int] = []
        for extent in shape:
            if isinstance(extent, str):
                static_shape.append(0)
                dim_slots.append(dynamic_slots[extent])
            elif isinstance(extent, int) and extent > 0:
                static_shape.append(extent)
                dim_slots.append(-1)
            else:
                return None

        kinds.append(int(spec.kind.value))
        direction = 2 if is_tiling else (getattr(spec.direction, "value", 0) if is_tensor else 0)
        tensor_types.append(direction)
        data_types.append(acl_dtype)
        element_bits.append(int(dtype.get_bit()))
        names.append(spec.name.encode("utf-8"))
        dtype_names.append(str(dtype).encode("utf-8"))
        shape_counts.append(len(shape))
        flat_shapes.extend((static_shape + [0] * _MAX_DIMS)[:_MAX_DIMS])
        flat_dynamic_slots.extend((dim_slots + [-1] * _MAX_DIMS)[:_MAX_DIMS])
        if kind_name in ("PTR", "TILING"):
            size_indices.append(index)

    num_params = len(specs)
    c_kinds = (ctypes.c_int32 * num_params)(*kinds)
    c_tensor_types = (ctypes.c_int32 * num_params)(*tensor_types)
    c_data_types = (ctypes.c_int32 * num_params)(*data_types)
    c_element_bits = (ctypes.c_int32 * num_params)(*element_bits)
    c_names = (ctypes.c_char_p * num_params)(*names)
    c_dtype_names = (ctypes.c_char_p * num_params)(*dtype_names)
    c_shape_counts = (ctypes.c_int32 * num_params)(*shape_counts)
    c_shapes = (ctypes.c_int64 * (num_params * _MAX_DIMS))(*flat_shapes)
    c_dynamic_slots = (ctypes.c_int32 * (num_params * _MAX_DIMS))(*flat_dynamic_slots)
    debug_cmd = _build_debug_compile_cmd(compiled).encode("utf-8")

    with _dump_schema_lock:
        result = _ensure_lib().pro_register_dump_schema(
            compiled.kernel_name.encode("utf-8"),
            num_params,
            c_kinds,
            c_tensor_types,
            c_data_types,
            c_element_bits,
            c_names,
            c_dtype_names,
            c_shape_counts,
            c_shapes,
            c_dynamic_slots,
            debug_cmd,
        )
        if result == 0:
            _active_dump_schemas[compiled.kernel_name] = (compiled, None)
    return tuple(size_indices) if result == 0 else None


def set_dump_runtime_info(compiled: "CompiledKernel", block_dim: int, sizes: tuple[int, ...]) -> None:
    """Update block dim and pointer-like allocation sizes after they change."""
    count = len(sizes)
    c_sizes = (ctypes.c_size_t * count)(*sizes)
    result = _ensure_lib().pro_set_dump_runtime_info(compiled.kernel_name.encode("utf-8"), block_dim, count, c_sizes)
    if result != 0:
        raise RuntimeError(f"Failed to update exception-dump runtime info for {compiled.kernel_name}")


def discard_dump_schema(compiled: "CompiledKernel") -> None:
    """Remove a stale schema before launching a variant that cannot use it."""
    with _dump_schema_lock:
        if compiled.kernel_name not in _active_dump_schemas:
            return
        try:
            result = _ensure_lib().pro_unregister_dump_schema(compiled.kernel_name.encode("utf-8"))
        except Exception:
            logging.exception("Failed to unregister exception-dump schema for %s", compiled.kernel_name)
            return
        if result != 0:
            logging.warning("Failed to unregister exception-dump schema for %s (ret=%d)", compiled.kernel_name, result)
            return
        _active_dump_schemas.pop(compiled.kernel_name, None)


def sync_dump_runtime_info(compiled: "CompiledKernel", block_dim: int, sizes: tuple[int, ...]) -> bool:
    """Keep the process-wide schema and dynamic state in sync with this launch."""
    with _dump_schema_lock:
        try:
            active = _active_dump_schemas.get(compiled.kernel_name)
            if active is None or active[0] is not compiled:
                if prepare_dump_schema(compiled) is None:
                    logging.warning("Failed to register exception-dump schema for %s", compiled.kernel_name)
                    discard_dump_schema(compiled)
                    return False
                active = _active_dump_schemas[compiled.kernel_name]
            state = (block_dim, sizes)
            if active[1] != state:
                set_dump_runtime_info(compiled, block_dim, sizes)
                _active_dump_schemas[compiled.kernel_name] = (compiled, state)
            return True
        except Exception:
            logging.exception("Failed to prepare exception dump for %s", compiled.kernel_name)
            discard_dump_schema(compiled)
            return False
