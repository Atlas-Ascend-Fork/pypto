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

r"""PyPTO Pro AICORE Error 离线复现工具。

从异常 dump 的 extra-info/data-dump 目录中读取 tensor 数据和编译产物，
构建单算子复现脚本并执行，验证 AICORE error 是否重现。

用法示例
--------
python debug_aicore_error_pro_repro.py -p <work_dir> [-d <device_id>]
"""

import argparse
import dataclasses
import glob
import inspect
import json
import logging
import math
import os
import re
import struct
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")


_STR_TO_TORCH = {
    "float16": "torch.float16", "float32": "torch.float32", "float64": "torch.float64",
    "int8": "torch.int8", "int16": "torch.int16", "int32": "torch.int32", "int64": "torch.int64",
    "uint8": "torch.uint8", "uint16": "torch.int16", "uint32": "torch.int32", "uint64": "torch.int64",
    "bfloat16": "torch.bfloat16", "bool": "torch.bool",
    "fp8e4m3fn": "torch.float8_e4m3fn", "fp8e5m2": "torch.float8_e5m2", "fp8e8m0": "torch.uint8",
    "hf8": "torch.uint8", "fp4e2m1": "torch.uint8", "fp4e1m2": "torch.uint8",
}

_STR_TO_NP = {
    "float16": "np.float16", "float32": "np.float32", "float64": "np.float64",
    "int8": "np.int8", "int16": "np.int16", "int32": "np.int32", "int64": "np.int64",
    "uint8": "np.uint8", "uint16": "np.int16", "uint32": "np.int32", "uint64": "np.int64",
    "bfloat16": "np.int16", "bool": "np.bool_",
    "fp8e4m3fn": "np.uint8", "fp8e5m2": "np.uint8", "fp8e8m0": "np.uint8",
    "hf8": "np.uint8", "fp4e2m1": "np.uint8", "fp4e1m2": "np.uint8",
}

_DTYPE_TO_SIZE = {
    "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
    "bfloat16": 2, "bool": 1,
    "fp8e4m3fn": 1, "fp8e5m2": 1, "fp8e8m0": 1, "hf8": 1,
}

_DT_NAME_TO_STR = {
    "FLOAT": "float32", "FLOAT16": "float16", "FLOAT64": "float64",
    "INT8": "int8", "INT16": "int16", "INT32": "int32", "INT64": "int64",
    "UINT8": "uint8", "UINT16": "uint16", "UINT32": "uint32", "UINT64": "uint64",
    "BOOL": "bool", "BFLOAT16": "bfloat16",
    "FP8E5M2": "fp8e5m2", "FP8E4M3FN": "fp8e4m3fn", "FLOAT8_E8M0": "fp8e8m0", "FP8E8M0": "fp8e8m0",
    "HIFLOAT8": "hf8", "HF8": "hf8",
    "FLOAT4_E2M1": "fp4e2m1", "FLOAT4_E1M2": "fp4e1m2",
    "FP4E2M1": "fp4e2m1", "FP4E1M2": "fp4e1m2",
}

_SIZE_TO_DTYPE = {1: "uint8", 2: "float16", 4: "float32", 8: "float64"}
_PACKED_FP4_DTYPES = frozenset(("fp4e2m1", "fp4e1m2"))


@dataclasses.dataclass
class TensorInfo:
    addr: int
    size: int
    dtype_str: str
    shape: tuple
    io_type: str
    index: int


def _parse_dump_file(dump_file: str) -> Tuple[str, List[TensorInfo]]:
    """从 dump 文件中解析 kernel name 和 tensor 信息。

    dump 文件格式:
    - 8 字节 uint64 total_size
    - protobuf 二进制头: 每个 tensor 的 dataType / shape / size / index
    - 文本元数据: [Dump][Exception] 行，记录 type / index / shape / dtype / address / size
    - 纯 tensor 数据（按 index 顺序连续排列）
    """
    with open(dump_file, "rb") as f:
        content = f.read()

    text = content.decode("utf-8", errors="replace")

    tensors: List[TensorInfo] = []
    for m in re.finditer(
        r"type=(\w+);\s*index=(\d+);\s*shape=\[([\d,]*)\];\s*format=\w+;\s*dtype=DT_(\w+);\s*address=(0x[0-9a-fA-F]+);\s*size=(\d+)\s*bytes",
        text
    ):
        io_type, index = m.group(1), int(m.group(2))
        shape = tuple(int(x) for x in m.group(3).split(",")) if m.group(3) else ()
        addr, size = int(m.group(5), 16), int(m.group(6))

        # dtype: trust the dump header; fall back to size/shape inference
        # (bytes per element) for unknown names or size mismatches.
        elems = 1
        for s in shape:
            elems *= s
        per_elem = size // elems if elems else 0
        dtype_str = _DT_NAME_TO_STR.get(m.group(4))
        if dtype_str is None:
            dtype_str = _SIZE_TO_DTYPE.get(per_elem, "float16") if elems else "uint8"
        elif elems and dtype_str not in _PACKED_FP4_DTYPES and _DTYPE_TO_SIZE.get(dtype_str, -1) != per_elem:
            fixed = _SIZE_TO_DTYPE.get(per_elem, dtype_str)
            logging.warning(f"  tensor[{index}] dtype {dtype_str} mismatch with size/shape, using {fixed}")
            dtype_str = fixed

        tensors.append(TensorInfo(addr, size, dtype_str, shape, io_type, index))
        logging.info(f"  tensor[{index}] {io_type} addr=0x{addr:x} size={size} dtype={dtype_str} shape={shape}")

    basename = os.path.basename(dump_file)
    parts = basename.split(".")
    kernel_name = parts[0] if parts else "unknown"

    return kernel_name, tensors


def _replay_storage_shape(tensor: TensorInfo) -> tuple:
    """Return the dumped storage shape, which is packed along the last axis for FP4."""
    if tensor.dtype_str not in _PACKED_FP4_DTYPES:
        return tensor.shape if tensor.shape else (-1,)
    if not tensor.shape or tensor.shape[-1] % 2:
        raise ValueError(f"FP4 tensor[{tensor.index}] needs an even innermost logical dimension")
    storage_shape = tensor.shape[:-1] + (tensor.shape[-1] // 2,)
    if math.prod(storage_shape) != tensor.size:
        raise ValueError(f"FP4 tensor[{tensor.index}] dump size does not match its packed shape")
    return storage_shape


def _replay_dynamic_dim(tensor: TensorInfo, var: str, axis: int) -> str:
    """Use the logical FP4 extent rather than the packed storage extent in the ABI tail."""
    if tensor.dtype_str in _PACKED_FP4_DTYPES:
        return str(tensor.shape[axis])
    return f"{var}.shape[{axis}]"


# C type name in call_kernel.cpp -> ctypes expression for argtypes binding.
_C_CTYPE_MAP = {
    "void*": "ctypes.c_void_p",
    "uint8_t*": "ctypes.c_void_p",
    "uint16_t*": "ctypes.c_void_p",
    "uint32_t*": "ctypes.c_void_p",
    "uint64_t*": "ctypes.c_void_p",
    "int8_t*": "ctypes.c_void_p",
    "int16_t*": "ctypes.c_void_p",
    "int32_t*": "ctypes.c_void_p",
    "int64_t*": "ctypes.c_void_p",
    "float*": "ctypes.c_void_p",
    "double*": "ctypes.c_void_p",
    "bool": "ctypes.c_bool",
    "int8_t": "ctypes.c_int8",
    "uint8_t": "ctypes.c_uint8",
    "int16_t": "ctypes.c_int16",
    "uint16_t": "ctypes.c_uint16",
    "int32_t": "ctypes.c_int32",
    "uint32_t": "ctypes.c_uint32",
    "int64_t": "ctypes.c_int64",
    "uint64_t": "ctypes.c_uint64",
    "float": "ctypes.c_float",
    "double": "ctypes.c_double",
}


def _load_launch_meta(dump_dir: str, kernel_name: str) -> Optional[dict]:
    """Load the <kernel>_launch_args.json sidecar written by the exception-dump callback.

    It records the full launch ABI in signature order (ptr / scalar / tiling
    args plus block_dim), which the dump data file alone cannot recover.
    """
    path = os.path.join(dump_dir, f"{kernel_name}_launch_args.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict) and isinstance(meta.get("abi"), list):
            return meta
        logging.warning(f"launch-args sidecar malformed, ignored: {path}")
    except (OSError, ValueError) as exc:
        logging.warning(f"launch-args sidecar unreadable ({exc}), ignored: {path}")
    return None


def _scalar_replay_expr(value, have_signature: bool) -> str:
    """Return a Python expression that preserves nonfinite launch scalars."""
    if isinstance(value, float):
        if math.isnan(value):
            literal = "float('nan')"
        elif math.isinf(value):
            literal = "float('-inf')" if value < 0 else "float('inf')"
        else:
            literal = repr(value)
        return literal if have_signature else f"ctypes.c_float({literal})"
    return repr(value) if have_signature else f"ctypes.c_int64({value!r})"


def _parse_call_kernel_signature(ck_cpp: str) -> Tuple[List[str], List[str]]:
    """Parse call_kernel.cpp's extern "C" entry into (arg names, ctypes exprs).

    Args are those after (blockDim, stream), in ABI order. The return type is
    ignored (void or int64_t alike). Unknown C types yield a None entry; the
    caller then skips argtypes binding rather than mis-binding values.
    """
    try:
        with open(ck_cpp, encoding="utf-8", errors="replace") as f:
            ck_text = f.read()
    except OSError:
        return [], []
    m = re.search(r'extern\s+"C"\s+\w+\s+call_kernel\s*\((.*?)\)', ck_text, re.S)
    if not m:
        return [], []
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    arg_names: List[str] = []
    arg_ctypes: List[str] = []
    for p in params[2:]:
        toks = p.split()
        if len(toks) < 2:
            return [], []
        arg_names.append(toks[-1].lstrip("*"))
        arg_ctypes.append(_C_CTYPE_MAP.get(" ".join(toks[:-1])))
    return arg_names, arg_ctypes


def _find_call_kernel_so(work_dir: str, dump_dir: Optional[str] = None,
                         kernel_name: Optional[str] = None) -> Optional[str]:
    """查找 call_kernel.so。

    优先使用异常回调复制到 dump 目录的 <kernel_name>_call_kernel.so 副本
    （随 dump 产物一起保存，不依赖 build 目录存活）；dump 目录没有副本时，
    再回退到 build 目录下搜索 call_kernel*.so（JIT 编译产物带 content digest 后缀）。
    """
    if dump_dir and os.path.isdir(dump_dir):
        if kernel_name:
            exact = os.path.join(dump_dir, f"{kernel_name}_call_kernel.so")
            if os.path.isfile(exact):
                return exact
        copies = sorted(f for f in os.listdir(dump_dir) if f.endswith("_call_kernel.so"))
        if copies:
            return os.path.join(dump_dir, copies[0])
    search_dirs = [
        os.path.join(work_dir, "build"),
        os.path.join(os.path.dirname(os.path.abspath(work_dir)), "build"),
        os.path.join(os.getcwd(), "build"),
        os.path.join(os.environ.get("ASCEND_WORK_PATH", ""), "build"),
    ]
    for build_dir in search_dirs:
        if os.path.isdir(build_dir):
            hits = glob.glob(os.path.join(build_dir, "**", "call_kernel*.so"), recursive=True)
            if hits:
                return hits[0]
    return None


def _elf_sections(data: bytes) -> Dict[str, Tuple[int, int]]:
    """Parse a little-endian ELF64's section table: name -> (file offset, size)."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return {}
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    if not e_shoff or not e_shnum:
        return {}

    def shdr(i):
        base = e_shoff + i * e_shentsize
        sh_name = struct.unpack_from("<I", data, base)[0]
        sh_offset = struct.unpack_from("<Q", data, base + 0x18)[0]
        sh_size = struct.unpack_from("<Q", data, base + 0x20)[0]
        sh_type = struct.unpack_from("<I", data, base + 4)[0]
        sh_link = struct.unpack_from("<I", data, base + 0x28)[0]
        sh_entsize = struct.unpack_from("<Q", data, base + 0x38)[0]
        return sh_name, sh_offset, sh_size, sh_type, sh_link, sh_entsize

    str_off = shdr(e_shstrndx)[1]
    out: Dict[str, Tuple[int, int]] = {}
    for i in range(e_shnum):
        sh_name, sh_offset, sh_size, sh_type, sh_link, sh_entsize = shdr(i)
        end = data.index(b"\0", str_off + sh_name)
        name = data[str_off + sh_name:end].decode("utf-8", "replace")
        out[name] = (sh_offset, sh_size)
    return out


def _find_bisheng_tools() -> Optional[Tuple[str, str, str]]:
    """Locate the bisheng toolchain's ld.lld + llvm-objdump + llvm-symbolizer.

    The pair relocations in the embedded device ELF's .debug_line are applied at
    link time by this lld, so linking the embedded ELF once through it yields a
    file whose line table stock LLVM consumers can decode.  llvm-symbolizer
    with --inlines then extracts the full inline call chain (intrinsic →
    user kernel) from the linked DWARF.

    Returns (ld_lld, llvm_objdump, llvm_symbolizer) or None when the toolchain
    is unavailable.
    """
    override = os.environ.get("PYPTO_BISHENG_BIN")
    roots = []
    if override:
        roots.append(override)
    ascend = os.environ.get("ASCEND_HOME_PATH")
    if ascend:
        roots.append(os.path.join(ascend, "tools", "bisheng_compiler", "bin"))
        roots.append(os.path.join(ascend, "..", "tools", "bisheng_compiler", "bin"))
        roots.append(os.path.join(ascend, "bin"))
        roots.append(os.path.join(ascend, "tools", "mssanitizer", "bin"))
        roots.append(os.path.join(ascend, "tools", "msopprof", "bin"))
    candidates = []
    for root in roots:
        ld = os.path.join(root, "ld.lld")
        objdump = os.path.join(root, "llvm-objdump")
        symbolizer = os.path.join(root, "llvm-symbolizer")
        if os.path.isfile(ld) and os.path.isfile(objdump) and os.path.isfile(symbolizer):
            candidates.append((ld, objdump, symbolizer))
    if candidates:
        return candidates[0]
    # Fallback: look for llvm-symbolizer separately if not in the same dir
    for root in roots:
        ld = os.path.join(root, "ld.lld")
        objdump = os.path.join(root, "llvm-objdump")
        if not os.path.isfile(ld) or not os.path.isfile(objdump):
            continue
        for sym_root in roots:
            symbolizer = os.path.join(sym_root, "llvm-symbolizer")
            if os.path.isfile(symbolizer):
                return (ld, objdump, symbolizer)
    return None


def _resolve_pc_to_source_via_tools(so_path: str, pc: int, tools: Tuple[str, str, str],
                                    symbol: str = "", symbol_offset: int = 0
                                    ) -> Optional[Tuple[str, int, List[Tuple[str, int]]]]:
    """Resolve a device Error-PC to (file, line, inline_chain) via bisheng tools.

    1. Cut the embedded device ELF (section __aicore_rel_binary) out of the .so.
    2. ld.lld -Ttext=0 links it, applying the CCE pair relocations to
       .debug_line and placing .text at address 0 (PC offsets need no rebase).
       (-shared is NOT used: kernels with pl.printf reference
       DebugTunnel globals whose R_AICORE_MOVADRP_G4 relocations are rejected
       in shared mode; plain relocation output links fine.)
    3. When the plog carries symbol+offset, rebase the PC through the symbol
       table: fatobj (mix) kernels have their AIV entry symbol at a non-zero
       .text offset, which shifts fixedPCOffset by exactly that amount.
    4. llvm-symbolizer --inlines maps the PC to the FULL inline call chain
       (innermost intrinsic → outer helper → user kernel code), upgrading the
       localization from "which intrinsic" to "which user-kernel call site".
       Falls back to llvm-objdump -d -l (single row) when the symbolizer is
       unavailable.

    The linked device ELF is persisted next to the .so as
    <kernel>_device_linked.so (relocations applied, .text at 0), so the dump
    directory keeps a directly-symbolizable device image for offline analysis.

    Returns (file, line, chain) where chain is a list of (file, line) from
    innermost to outermost inline frame; chain has a single entry on the
    objdump fallback path.
    """
    import shutil
    import tempfile

    ld, objdump, symbolizer = tools
    data = open(so_path, "rb").read()
    outer = _elf_sections(data)
    rel = outer.get("__aicore_rel_binary")
    if rel is None:
        return None
    tmp = tempfile.mkdtemp(prefix="pc_resolve_")
    try:
        obj = os.path.join(tmp, "device.o")
        with open(obj, "wb") as f:
            f.write(data[rel[0]:rel[0] + rel[1]])
        linked = os.path.join(tmp, "device_linked.so")
        try:
            subprocess.run([ld, "-Ttext=0", obj, "-o", linked],
                           capture_output=True, timeout=120, check=True)
        except (subprocess.SubprocessError, OSError):
            return None
        base = os.path.basename(so_path)
        stem = base[:-len("_call_kernel.so")] if base.endswith("_call_kernel.so") else "device"
        try:
            shutil.copyfile(linked, os.path.join(os.path.dirname(so_path), f"{stem}_device_linked.so"))
        except OSError:
            pass
        if symbol:
            try:
                syms = subprocess.run([objdump, "--syms", linked],
                                      capture_output=True, text=True, timeout=120, check=True)
                for line in syms.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 6 and parts[3] == ".text" and parts[5] == symbol:
                        pc = int(parts[0], 16) + symbol_offset
                        logging.info(f"symbol {symbol}+0x{symbol_offset:x} -> PC 0x{pc:x}")
                        break
            except (subprocess.SubprocessError, OSError, ValueError):
                pass
        chain = _resolve_pc_via_symbolizer(symbolizer, linked, pc)
        if chain is not None:
            # Pick the innermost user-source frame (not CANN include headers,
            # not call_kernel.cpp) as primary, so the user sees the exact
            # instruction call site (e.g. the TSTORE line) in their kernel,
            # not the outermost dispatch wrapper.
            primary_file, primary_line = chain[-1]
            for c_file, c_line in chain:
                if "/include/" not in c_file and not c_file.endswith("call_kernel.cpp"):
                    primary_file, primary_line = c_file, c_line
                    break
            return primary_file, primary_line, chain
        return _resolve_pc_via_objdump(objdump, linked, pc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _resolve_pc_via_symbolizer(symbolizer: str, linked: str, pc: int
                               ) -> Optional[List[Tuple[str, int]]]:
    """Map PC to the full inline chain via llvm-symbolizer --inlines.

    Output pairs (function name, file:line:col) from innermost to outermost;
    e.g. an inlined intrinsic reports the intrinsic header frame followed by
    each enclosing inline site up to the user kernel. Returns [(file, line)...]
    or None when the tool is unavailable or no frame resolves.
    """
    try:
        result = subprocess.run(
            [symbolizer, "--obj", linked, "--inlines", "--output-style=LLVM"],
            input=f"0x{pc:x}\n", capture_output=True, text=True, timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    raw = [s.strip() for s in result.stdout.splitlines()]
    # Strip empty lines; remainder is pairs of (func, file:line:col) from
    # innermost to outermost. Empty lines appear when a frame's DILocation
    # spans multiple scopes — skip them conservatively.
    lines = [s for s in raw if s]
    loc_re = re.compile(r"^(.+):(\d+):(\d+)$")
    chain: List[Tuple[str, int]] = []
    for i in range(0, len(lines) - 1, 2):
        m = loc_re.match(lines[i + 1])
        if m and "/" in m.group(1):
            chain.append((m.group(1), int(m.group(2))))
    return chain if chain else None


def _resolve_pc_via_objdump(objdump: str, linked: str, pc: int
                            ) -> Optional[Tuple[str, int, List[Tuple[str, int]]]]:
    """Fallback: map PC to a single (file, line) via llvm-objdump -d -l."""
    try:
        result = subprocess.run(
            [objdump, "-d", "-l", linked],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    rows: List[Tuple[int, str, int]] = []
    marker = re.compile(r"^(.+[^()]):(\d+)$")
    addr_re = re.compile(r"^\s+([0-9a-fA-F]+):")
    cur_file: Optional[str] = None
    cur_line = 0
    for raw in result.stdout.splitlines():
        line = raw.rstrip()
        m = marker.match(line)
        if m and "/" in m.group(1):
            cur_file, cur_line = m.group(1), int(m.group(2))
            continue
        ma = addr_re.match(line)
        if ma and cur_file is not None:
            rows.append((int(ma.group(1), 16), cur_file, cur_line))
    valid = [r for r in rows if r[2] and r[0] <= pc]
    if not valid:
        return None
    src = (valid[-1][1], valid[-1][2])
    return src[0], src[1], [src]


def _resolve_pc_to_source(so_path: str, pc: int, symbol: str = "", symbol_offset: int = 0) -> Optional[Tuple[str, int]]:
    """Map a device Error-PC offset to (file, line) via the bisheng toolchain.

    The device code's .debug_line carries CCE pair relocations that stock DWARF
    consumers cannot apply. The toolchain's own ld.lld applies them at link
    time, so the embedded device ELF is linked once (-Ttext=0, no rebase
    needed) and decoded with llvm-symbolizer (preferred) or llvm-objdump
    (fallback). Returns None when the toolchain is unavailable.
    """
    tools = _find_bisheng_tools()
    if tools is None:
        logging.warning("bisheng toolchain not found, Error-PC source line unavailable")
        return None
    result = _resolve_pc_to_source_via_tools(
        so_path, pc, tools, symbol=symbol, symbol_offset=symbol_offset)
    if result:
        return result[0], result[1]
    return None


def _resolve_pc_to_source_with_chain(so_path: str, pc: int, symbol: str = "", symbol_offset: int = 0
                                     ) -> Optional[Tuple[str, int, List[Tuple[str, int]]]]:
    """Like _resolve_pc_to_source, but returns the full inline chain.

    Returns (file, line, chain) where chain is [(file, line), ...] from
    innermost (intrinsic) to outermost (kernel entry), or None.
    """
    tools = _find_bisheng_tools()
    if tools is None:
        logging.warning("bisheng toolchain not found, Error-PC source line unavailable")
        return None
    return _resolve_pc_to_source_via_tools(so_path, pc, tools, symbol=symbol, symbol_offset=symbol_offset)


def _source_snippet(src_file: str, src_line: int, context: int = 3) -> str:
    """Render a few source lines around src_line for the localization report."""
    try:
        with open(src_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    start = max(src_line - context, 1)
    end = min(src_line + context, len(lines))
    out = []
    for n in range(start, end + 1):
        marker = ">" if n == src_line else " "
        out.append(f"  {marker} {n:5d} | {lines[n - 1].rstrip()}")
    return "\n".join(out) + "\n"


# -------------------------------------------------------------------
# Timeout-class diagnosis: msnpureport singlecommit (inter-core sync check)
# -------------------------------------------------------------------

_TIMEOUT_SIGNATURE = re.compile(r"times out|timeout|trap", re.IGNORECASE)


def _find_msnpureport() -> str:
    """Locate the msnpureport driver tool (mirrors debug_aicore_error.py)."""
    env_path = os.environ.get("MSNPUREPORT_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        ascend_root = os.path.dirname(os.path.abspath(ascend_home))
        cand = os.path.join(ascend_root, "driver", "tools", "msnpureport")
        if os.path.isfile(cand):
            return cand
    return "/usr/local/Ascend/driver/tools/msnpureport"


def _is_docker_env() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content:
                return True
    except (OSError, PermissionError):
        pass
    return False


def _msnpureport_set_singlecommit(enable: bool, device_id: int, is_docker: bool = False):
    """Enable/disable device singlecommit mode. Returns (success, cmd_str, output)."""
    val = "1" if enable else "0"
    cmd = [_find_msnpureport(), "config", "--set", "--singlecommit", val, "-d", str(device_id)]
    if is_docker:
        cmd.append("--docker")
    cmd_str = " ".join(cmd)
    action = "enable" if enable else "restore"
    logging.info(f"{action} singlecommit: {cmd_str}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.stderr:
            output += "\n" + result.stderr.strip()
        return result.returncode == 0, cmd_str, output
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, cmd_str, f"[ERROR] {exc}"


def run_singlecommit_diagnosis(script_path: str, device_id: int, out_dir: str,
                               timeout: int) -> dict:
    """Re-run the repro under singlecommit=1; the caller analyzes the result.

    singlecommit serializes task commit on the device, removing inter-core
    concurrency. For timeout-class faults the trap PC parks in compiler
    internals (e.g. PrintState) and is useless; under singlecommit the
    snapshot lands on the real stuck loop, so the caller re-parses the
    singlecommit run's plog for ALL stuck cores' ErrorPCs afterwards.

    Uses ``out_dir`` directly as the ASCEND_WORK_PATH for the singlecommit
    run (no subdirectory); the caller's ``since=sc_start`` time window isolates
    the plog records that belong to this run.

    Returns {executed, passed, sc_work, sc_start, output, log}. Device config
    is always restored (finally). ``executed`` is False only when msnpureport
    is unavailable/enabling failed; a subprocess timeout still counts as
    executed (the device may have written plog records before the kill).
    """
    result: dict = {"executed": False, "passed": False, "sc_work": "",
                    "sc_start": 0.0, "output": "", "log": ""}
    if not os.path.isfile(_find_msnpureport()):
        result["log"] = "msnpureport not found, singlecommit diagnosis skipped\n"
        logging.warning(result["log"].strip())
        return result

    is_docker = _is_docker_env()
    sc_work = out_dir
    result["sc_work"] = sc_work

    ok, cmd_str, msn_out = _msnpureport_set_singlecommit(True, device_id, is_docker)
    log_lines: List[str] = [f"msnpureport enable single-step: {cmd_str}", msn_out, ""]
    if not ok:
        logging.warning(f"enable singlecommit failed: {msn_out}")
        _msnpureport_set_singlecommit(False, device_id, is_docker)
        result["log"] = "\n".join(log_lines) + "\n"
        return result

    run_output = ""
    try:
        env = os.environ.copy()
        env["ASCEND_WORK_PATH"] = sc_work
        logging.info("========== Re-executing under singlecommit=1 ==========")
        result["sc_start"] = time.time()
        try:
            r = subprocess.run([sys.executable, script_path], capture_output=True,
                               text=True, timeout=timeout, env=env)
            run_output = r.stdout or ""
            if r.stderr:
                run_output += "\n[stderr]\n" + r.stderr
            result["passed"] = r.returncode == 0
            result["executed"] = True
        except subprocess.TimeoutExpired:
            run_output = f"[ERROR] repro timed out ({timeout}s) under singlecommit=1"
            result["executed"] = True
        result["output"] = run_output
    finally:
        ok2, cmd_str2, msn_out2 = _msnpureport_set_singlecommit(False, device_id, is_docker)
        log_lines.append(f"Re-execute (singlecommit=1):\n{run_output}")
        log_lines += ["", f"msnpureport restore: {cmd_str2}",
                      msn_out2 if ok2 else f"[ERROR] restore failed: {msn_out2}", ""]
    result["log"] = "\n".join(log_lines) + "\n"
    return result


def _locate_error_pc(so_path: str, work_dir: str, since: float) -> Optional[Tuple[int, str, Optional[Tuple[str, int]]]]:
    """Locate the Error PC from freshly written plogs and resolve it to source.

    Embedded verbatim (together with the resolver helpers) into generated
    repro scripts so they can print the localization themselves. Returns
    (pc, symbol, (file, line) | None) or None.

    Searches ``work_dir/log`` first, then falls back to the default ascend
    log path (``~/ascend/log``) so that scripts work even when
    ``ASCEND_WORK_PATH`` is not exported.

    The symbol line carries the true PC as symbol+offset; it is preferred over
    fixedPCOffset, whose zero-based normalization assumes a single kernel
    entry at .text 0. On fatobj (mix) kernels the AIV entry symbol sits at a
    non-zero .text offset, and fixedPCOffset is then shifted by exactly that
    amount, pointing at the wrong source row.
    """
    pc_line_re = re.compile(r"Error PC information\..*?fixedPCOffset=0x([0-9a-fA-F]+)")
    sym_re = re.compile(r"Error symbol information\..*?symbol=(\S+?)(?:\+0x([0-9a-fA-F]+))?\.?\s*$", re.MULTILINE)
    plogs: List[Tuple[float, str]] = []
    for base_dir in (os.path.join(work_dir, "log"),
                     os.path.join(os.path.expanduser("~"), "ascend", "log")):
        if os.path.isdir(base_dir):
            for root, _, fnames in os.walk(base_dir):
                for fname in fnames:
                    fpath = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                    except OSError:
                        continue
                    if mtime >= since:
                        plogs.append((mtime, fpath))
    for _, plog in sorted(plogs, reverse=True):
        try:
            with open(plog, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        m = pc_line_re.search(text)
        if m:
            sym, sym_off = "", 0
            ms = sym_re.search(text)
            if ms:
                sym = ms.group(1)
                sym_off = int(ms.group(2), 16) if ms.group(2) else 0
            pc = int(m.group(1), 16)
            src = _resolve_pc_to_source(so_path, pc, symbol=sym, symbol_offset=sym_off)
            return pc, sym, src
    return None


_PC_ALL_RE = re.compile(
    r"Error PC information\.\s*coreId=(\d+),\s*coreType=(\d+).*?"
    r"fixedPCOffset=0x([0-9a-fA-F]+)")
_SYM_ALL_RE = re.compile(
    r"Error symbol information\.\s*coreId=(\d+),\s*coreType=(\d+),\s*"
    r"symbol=(\S+?)(?:\+0x([0-9a-fA-F]+))?\.?\s*$",
    re.MULTILINE)


def _locate_error_pcs(so_path: str, work_dir: str, since: float) -> List[dict]:
    """Collect the ErrorPC of EVERY stuck core from a run's own plog.

    A cv-fusion (mix) kernel hangs with cube and vector cores parked at
    DIFFERENT wait points — the pair of stuck locations is the evidence of a
    deadlock, so only reporting the first record hides one side. This walks
    the same plog set as the embedded single-record locator but parses ALL
    "Error PC information" / "Error symbol information" lines, pairs them by
    (coreId, coreType), aggregates identical (coreType, PC, symbol) records
    and resolves each to a source line.

    Returns a list of {core_type, core_ids, pc, symbol, symbol_offset, src}
    in first-seen order. (Kept separate from _locate_error_pc, which is
    embedded verbatim into generated repro scripts and must stay
    self-contained.)
    """
    plogs: List[Tuple[float, str]] = []
    for base_dir in (os.path.join(work_dir, "log"),
                     os.path.join(os.path.expanduser("~"), "ascend", "log")):
        if os.path.isdir(base_dir):
            for root, _, fnames in os.walk(base_dir):
                for fname in fnames:
                    fpath = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                    except OSError:
                        continue
                    if mtime >= since:
                        plogs.append((mtime, fpath))

    pcs_by_core: Dict[Tuple[str, str], List[int]] = {}
    syms_by_core: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for _, plog in sorted(plogs, reverse=True):
        try:
            with open(plog, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if not _PC_ALL_RE.search(text):
            continue
        for m in _PC_ALL_RE.finditer(text):
            pcs_by_core.setdefault((m.group(1), m.group(2)), []).append(int(m.group(3), 16))
        for m in _SYM_ALL_RE.finditer(text):
            sym = m.group(3)
            off = int(m.group(4), 16) if m.group(4) else 0
            syms_by_core.setdefault((m.group(1), m.group(2)), []).append((sym, off))
        break  # all records belong to this (newest matching) plog

    aggregated: Dict[Tuple[str, int, str, int], dict] = {}
    for key, pcs in pcs_by_core.items():
        syms = syms_by_core.get(key, [])
        for i, pc in enumerate(pcs):
            sym, off = syms[i] if i < len(syms) else ("", 0)
            akey = (key[1], pc, sym, off)
            rec = aggregated.get(akey)
            if rec is None:
                rec = {"core_type": int(key[1]), "core_ids": set(), "pc": pc,
                       "symbol": sym, "symbol_offset": off, "src": None}
                aggregated[akey] = rec
            rec["core_ids"].add(int(key[0]))

    records = []
    for rec in aggregated.values():
        resolved = _resolve_pc_to_source_with_chain(so_path, rec["pc"],
                                                    symbol=rec["symbol"],
                                                    symbol_offset=rec["symbol_offset"])
        if resolved:
            rec["src"] = (resolved[0], resolved[1])
            rec["inline_chain"] = resolved[2]
        else:
            rec["src"] = None
            rec["inline_chain"] = None
        rec["core_ids"] = sorted(rec["core_ids"])
        records.append(rec)
    return records


def _format_error_pc_records(records: List[dict], dump_dir: str) -> str:
    """Render all stuck-core records (one per aggregated PC) for the report.

    Each record shows the inline chain (innermost → outermost frame) so the
    user sees not only the intrinsic that faulted but also which call site in
    their kernel code reached it.
    """
    if not records:
        return "Error PC not found in plog (symbol locator output absent)\n"
    out: List[str] = []
    for rec in records:
        label = "AICORE(cube)" if rec["core_type"] == 0 else "AIVECTOR(vec)"
        sym_repr = f"{rec['symbol']}+0x{rec['symbol_offset']:x}" if rec["symbol"] else "<unknown>"
        out.append(f"[{label} cores={','.join(str(c) for c in rec['core_ids'])}] "
                   f"ErrorPC: 0x{rec['pc']:x}  symbol={sym_repr}")
        chain = rec.get("inline_chain") or []
        if chain:
            src_file, src_line = rec["src"] if rec["src"] is not None else chain[-1]
            for idx, (c_file, c_line) in enumerate(chain):
                # Prefer the source copies shipped inside the dump dir so the
                # report never depends on the (ephemeral) build tree.
                local_copy = os.path.join(dump_dir, os.path.basename(c_file))
                if os.path.isfile(local_copy):
                    c_file = local_copy
                arrow = "->" if os.path.basename(c_file) == os.path.basename(src_file) and c_line == src_line else " |"
                out.append(f"    {arrow} {c_file}:{c_line}")
            src_file, src_line = rec["src"] if rec["src"] is not None else chain[-1]
            logging.info(f"[{label}] ErrorPC 0x{rec['pc']:x} ({sym_repr}) -> {src_file}:{src_line}")
            snippet = _source_snippet(src_file, src_line)
            if snippet:
                out.append(snippet.rstrip("\n"))
        else:
            out.append("    Source: <line table not resolvable in call_kernel.so>")
            logging.warning(f"[{label}] ErrorPC 0x{rec['pc']:x} ({sym_repr}) -> "
                            "source line not resolvable")
        out.append("")
    return "\n".join(out)


def codegen_test_script(
    call_kernel_so: str,
    kernel_name: str,
    dump_file: str,
    tensors: List[TensorInfo],
    device_id: int,
    output_path: str,
    block_dim: int = 1,
    launch_meta: Optional[dict] = None,
    work_dir: str = "",
) -> None:
    """生成单算子复现脚本。"""
    with open(dump_file, "rb") as f:
        all_data = f.read()

    # Tensor data sits after the [Dump] text header (which ends right after
    # the last "bytes\n" line within its first 4KB).
    dump_text_start = all_data.find(b"[Dump]")
    region = all_data[dump_text_start:dump_text_start + 4096] if dump_text_start != -1 else b""
    last_bytes = region.rfind(b"bytes\n")
    text_end = dump_text_start + last_bytes + len(b"bytes\n") if last_bytes != -1 else 0

    tensor_data_list = []
    offset = text_end
    for t in sorted(tensors, key=lambda x: x.index):
        tensor_data = all_data[offset:offset + t.size]
        tensor_data_list.append(tensor_data)
        offset += t.size

    bin_dir = os.path.dirname(output_path)
    load_lines: List[str] = []
    tensor_vars: List[str] = []
    tensors_sorted = sorted(tensors, key=lambda x: x.index)

    for t, data in zip(tensors_sorted, tensor_data_list):
        bin_path = os.path.join(bin_dir, f"tensor_{t.io_type}_{t.index}.bin")
        with open(bin_path, "wb") as f:
            f.write(data)

        var = f"t_{t.io_type}_{t.index}"
        torch_dtype = _STR_TO_TORCH.get(t.dtype_str, "torch.float16")
        np_dtype = _STR_TO_NP.get(t.dtype_str, "np.float16")
        shape_repr = repr(_replay_storage_shape(t))
        load_lines.append(f"{var}_bin = r'{bin_path}'")
        load_lines.append(f"{var}_np = np.fromfile({var}_bin, dtype={np_dtype})")
        load_lines.append(f"{var}_np = {var}_np.reshape({shape_repr})")
        load_lines.append(f"{var} = torch.tensor(")
        load_lines.append(f"    {var}_np, device=device).view({torch_dtype})")
        tensor_vars.append(var)
    tensor_by_var = dict(zip(tensor_vars, tensors_sorted))

    # Launch ABI binding. The C entry is
    # call_kernel(blockDim, stream, <ptr/scalar/tiling args...>, int32 dyn...)
    # where each __pypto_dyn_<param>_<k> arg takes the k-th dim of that tensor
    # param's runtime shape. Two sources rebuild it:
    #   1. <kernel>_launch_args.json sidecar (preferred): the exception-time
    #      record of every ABI arg plus block_dim — covers standalone scalars
    #      and multi-block launches that the signature alone cannot express.
    #   2. call_kernel.cpp signature alone (fallback, single-block, ptr-only):
    #      tensors in dump order, dyn args from shapes.
    ck_cpp = os.path.join(os.path.dirname(call_kernel_so), "call_kernel.cpp")
    arg_names: List[str] = []
    arg_ctypes: List[str] = []
    if os.path.isfile(ck_cpp):
        arg_names, arg_ctypes = _parse_call_kernel_signature(ck_cpp)
    have_signature = bool(arg_names) and all(ct is not None for ct in arg_ctypes)

    dyn_vals: List[str] = []
    launch_exprs: List[str] = []

    if launch_meta is not None and launch_meta.get("abi"):
        var_by_addr = {t.addr: var for t, var in zip(tensors_sorted, tensor_vars)}
        used_vars: set = set()
        var_by_name: Dict[str, str] = {}
        for entry in launch_meta["abi"]:
            if entry.get("kind") != "ptr":
                continue
            name = str(entry.get("name", ""))
            var = var_by_addr.get(int(entry.get("addr") or 0))
            if var is None:
                var = next((v for v in tensor_vars if v not in used_vars), None)
            if var is None:
                logging.warning(f"launch-args: no dumped tensor for ptr '{name}', passing null")
            else:
                used_vars.add(var)
                var_by_name[name] = var
        for entry in launch_meta["abi"]:
            if entry.get("kind") == "ptr":
                var = var_by_name.get(str(entry.get("name", "")))
                launch_exprs.append(f"ctypes.c_void_p({var}.data_ptr())" if var else "ctypes.c_void_p(0)")
            elif entry.get("kind") == "window":
                logging.warning("launch-args: distributed Window descriptor cannot be restored offline; passing null")
                launch_exprs.append("ctypes.c_void_p(0)")
            else:
                value = entry.get("value")
                # Plain literal when argtypes binds the width; otherwise pin
                # int64 so a wide scalar cannot be truncated by ctypes' default.
                launch_exprs.append(_scalar_replay_expr(value, have_signature))
        for name in arg_names:
            dm = re.fullmatch(r"__pypto_dyn_(.+)_(\d+)", name)
            if dm and dm.group(1) in var_by_name:
                var = var_by_name[dm.group(1)]
                dyn_vals.append(_replay_dynamic_dim(tensor_by_var[var], var, int(dm.group(2))))
    else:
        ptr_names = [n for n in arg_names if not n.startswith("__pypto_dyn_")]
        var_by_param = dict(zip(ptr_names, tensor_vars))
        for name in arg_names:
            dm = re.fullmatch(r"__pypto_dyn_(.+)_(\d+)", name)
            if dm and dm.group(1) in var_by_param:
                var = var_by_param[dm.group(1)]
                dyn_vals.append(_replay_dynamic_dim(tensor_by_var[var], var, int(dm.group(2))))
        launch_exprs = [f"ctypes.c_void_p({v}.data_ptr())" for v in tensor_vars]

    if not dyn_vals:
        logging.warning("call_kernel.cpp signature not usable, launching without dynamic-shape args")

    # Bind argtypes when the parsed signature covers every launch value — this
    # is what makes wide int64 scalars survive the ctypes call. arg_names
    # already includes the dyn tail, so arg_ctypes alone completes the list.
    argtypes_expr: Optional[str] = None
    if have_signature and len(arg_names) == len(launch_exprs) + len(dyn_vals):
        argtypes_expr = (
            "[ctypes.c_uint32, ctypes.c_void_p, " + ", ".join(arg_ctypes) + "]"
        )

    lines = [
        "#!/usr/bin/env python3",
        "# Auto-generated by debug_aicore_error_pro_repro.py",
        f"# Kernel: {kernel_name}",
        f"# Device: {device_id}",
        f"# block_dim: {block_dim}",
        "",
        "import ctypes",
        "import logging",
        "import os",
        "import re",
        "import struct",
        "import subprocess",
        "import sys",
        "import time",
        "from typing import Dict, List, Optional, Tuple",
        "",
        "import numpy as np",
        "import torch",
        "import torch_npu  # noqa: F401",
        "",
        "",
    ]

    # Embed the Error-PC line resolver so this script is self-contained: on
    # reproduction it prints the source file/line itself, without needing the
    # repro tool or the build directory. Sourced live via inspect so the copy
    # can never drift from the tool's own implementation.
    lines.append(inspect.getsource(_elf_sections).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_find_bisheng_tools).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_resolve_pc_via_symbolizer).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_resolve_pc_via_objdump).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_resolve_pc_to_source_via_tools).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_resolve_pc_to_source).rstrip())
    lines.append("")
    lines.append(inspect.getsource(_locate_error_pc).rstrip())
    lines.append("")
    lines.append("")

    lines += [
        f"device = torch.device('npu:{device_id}')",
        "torch.npu.set_device(device)",
        "",
        f"_so_path = r'{call_kernel_so}'",
        "_SO = ctypes.CDLL(_so_path)",
        "",
        f"_work_dir = r'{work_dir}'",
        "",
    ]
    lines.extend(load_lines)
    lines += [
        "",
        f"_launch_vals = [{', '.join(launch_exprs)}]",
        f"_dyn_args = [{', '.join(dyn_vals)}]",
        "_stream = torch.npu.current_stream()",
        f"_block_dim = {block_dim}",
        "",
    ]
    if argtypes_expr:
        lines += [
            f"_SO.call_kernel.argtypes = {argtypes_expr}",
            "",
        ]
    lines += [
        "print(f'Launching kernel with block_dim={_block_dim}')",
        "_SO.call_kernel(_block_dim, getattr(_stream, '_as_parameter_'), *_launch_vals, *_dyn_args)",
        "try:",
        "    torch.npu.synchronize()",
        "    print('PASS: No AICORE error — execution succeeded')",
        "    sys.exit(0)",
        "except Exception as e:",
        "    print(f'FAIL: AICORE error reproduced: {e}')",
        "    _work = os.environ.get('ASCEND_WORK_PATH') or _work_dir",
        "    _hit = _locate_error_pc(_so_path, _work, time.time() - 120.0)",
        "    if _hit:",
        "        _pc, _sym, _src = _hit",
        "        print(f'ErrorPC: 0x{_pc:x}')",
        "        print(f'Symbol: {_sym or \"<unknown>\"}')",
        "        if _src:",
        "            _local = os.path.join(os.path.dirname(_so_path), os.path.basename(_src[0]))",
        "            if os.path.isfile(_local):",
        "                _src = (_local, _src[1])",
        "            print(f'Source: {_src[0]}:{_src[1]}')",
        "    else:",
        "        print('ErrorPC: <not found in plog>')",
        "    sys.exit(1)",
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"Test script generated: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyPTO Pro AICORE Error Offline Reproduction Tool")
    parser.add_argument("-p", type=str, required=True,
                        help="Directory holding the dump artifacts for ONE device "
                             "(ASCEND_WORK_PATH/extra-info/data-dump/<device_id>, "
                             "where the dump data file and _call_kernel.so live)")
    parser.add_argument("-d", type=str, default=None, help="Device ID")
    parser.add_argument("-out", type=str, default=None,
                        help="Output directory (default: <work_dir>/aicore_error_debug)")
    parser.add_argument("-t", type=int, default=600, help="Timeout (seconds)")
    return parser.parse_args()


def _work_dir_from_dump_dir(dump_dir: str) -> str:
    """dump 目录位于 <work>/extra-info/data-dump/<device_id>,上溯三级得到 work 目录。"""
    return os.path.abspath(os.path.join(dump_dir, os.pardir, os.pardir, os.pardir))


def main():
    args = parse_args()

    dump_dir = os.path.abspath(args.p)
    if not os.path.isdir(dump_dir):
        logging.error(f"dump directory not found: {dump_dir}")
        sys.exit(1)
    work_dir = _work_dir_from_dump_dir(dump_dir)

    # Multi-card safe: the device id comes from the dump directory itself
    # (.../data-dump/<device_id>/), never defaulted to card 0. -d overrides.
    dev_name = os.path.basename(dump_dir.rstrip(os.sep))
    device_id = int(args.d) if args.d is not None else int(dev_name) if dev_name.isdigit() else 0
    logging.info(f"device_id = {device_id}")
    logging.info(f"dump dir = {dump_dir}")

    if args.out is None:
        args.out = os.path.join(work_dir, "aicore_error_debug")
    os.makedirs(args.out, exist_ok=True)
    logging.info(f"output dir = {args.out}")

    # ADDITIONAL mode can also write native exception_info.* files.
    dump_names = sorted(os.listdir(dump_dir))
    dump_names.sort(key=lambda name: not os.path.isfile(
        os.path.join(dump_dir, name.split(".")[0] + "_launch_args.json")))
    dump_file = None
    for fname in dump_names:
        fpath = os.path.join(dump_dir, fname)
        if os.path.isfile(fpath) and not fname.endswith(("_host.o", "_debug.o", ".so", ".cpp", ".h", ".json")):
            dump_file = fpath
            break
    if dump_file is None:
        logging.error("No dump data file found")
        sys.exit(1)
    logging.info(f"dump file = {dump_file}")

    kernel_name, tensors = _parse_dump_file(dump_file)
    if not tensors:
        logging.error("No tensors parsed from dump file")
        sys.exit(1)
    logging.info(f"kernel_name = {kernel_name}, {len(tensors)} tensors")

    call_kernel_so = _find_call_kernel_so(work_dir, dump_dir=dump_dir, kernel_name=kernel_name)
    if not call_kernel_so:
        logging.error("call_kernel.so not found in build directory")
        sys.exit(1)
    logging.info(f"call_kernel.so = {call_kernel_so}")

    launch_meta = _load_launch_meta(dump_dir, kernel_name)
    block_dim = 1
    if launch_meta is not None:
        recorded = launch_meta.get("block_dim")
        if isinstance(recorded, int) and recorded > 0:
            block_dim = recorded
            logging.info(f"block_dim = {block_dim} (from launch-args sidecar)")
        else:
            logging.warning("launch-args sidecar has no block_dim, defaulting to 1")

    script_path = os.path.join(args.out, "test_single_op.py")
    codegen_test_script(call_kernel_so, kernel_name, dump_file, tensors, device_id, script_path,
                        block_dim=block_dim, launch_meta=launch_meta, work_dir=work_dir)

    logging.info("========== Executing Single-Operator Test ==========")
    repro_start = time.time()
    # Pin ASCEND_WORK_PATH so the repro run's plog lands in work_dir/log,
    # where _locate_error_pcs looks for it (the caller's env may point
    # elsewhere or be unset, losing the Error PC records).
    repro_env = os.environ.copy()
    repro_env["ASCEND_WORK_PATH"] = work_dir
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True,
                                timeout=args.t, env=repro_env)
    except subprocess.TimeoutExpired:
        logging.error(f"Test script timed out ({args.t}s)")
        sys.exit(1)

    output = result.stdout or ""
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    print(output)

    pc_info = ""
    sc_info = ""
    if result.returncode != 0:
        logging.info("FAIL: AICORE error reproduced")
        records = _locate_error_pcs(call_kernel_so, work_dir, repro_start)
        pc_info = _format_error_pc_records(records, dump_dir)
        if _TIMEOUT_SIGNATURE.search(output):
            logging.info("Timeout-class failure: re-reproducing under singlecommit=1")
            sc = run_singlecommit_diagnosis(script_path, device_id, args.out, args.t)
            sc_info += "Singlecommit diagnosis (timeout-class):\n" + sc["log"]
            if sc["executed"]:
                if sc["passed"]:
                    conclusion = ("Conclusion: PASS in singlecommit mode -> inter-core "
                                  "synchronization issue (concurrency/timing)\n")
                    sc_info += conclusion
                    logging.info(conclusion.strip())
                else:
                    sc_info += ("Conclusion: still FAIL in singlecommit mode -> not an "
                                "inter-core sync issue; stuck cores of this run:\n")
                    sc_records = _locate_error_pcs(call_kernel_so, sc["sc_work"], sc["sc_start"])
                    sc_info += _format_error_pc_records(sc_records, dump_dir)
    else:
        logging.info("PASS: No AICORE error — execution succeeded")

    report_path = os.path.join(args.out, "reproduction_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Kernel: {kernel_name}\nDevice: {device_id}\n"
                f"call_kernel.so: {call_kernel_so}\ndump file: {dump_file}\n"
                f"Tensors: {len(tensors)}\n")
        for t in tensors:
            f.write(f"  {t.io_type}[{t.index}] addr=0x{t.addr:x} size={t.size} dtype={t.dtype_str} shape={t.shape}\n")
        if pc_info:
            f.write(f"\nError PC localization:\n{pc_info}\n")
        f.write(f"\nExecution result (exit code={result.returncode}):\n{output}")
    logging.info(f"Report saved: {report_path}")

    if sc_info:
        sc_path = os.path.join(args.out, "reproduction_report--singlecommit.txt")
        with open(sc_path, "w") as f:
            f.write(f"Kernel: {kernel_name}\nDevice: {device_id}\n"
                    f"call_kernel.so: {call_kernel_so}\ndump file: {dump_file}\n")
            for t in tensors:
                _fmt = f"  {t.io_type}[{t.index}] addr=0x{t.addr:x}"
                f.write(f"{_fmt} size={t.size} dtype={t.dtype_str} shape={t.shape}\n")
            f.write(f"\n{sc_info}")
            f.write(f"\nExecution result (exit code={result.returncode}):\n{output}")
        logging.info(f"Singlecommit report saved: {sc_path}")


if __name__ == "__main__":
    main()
