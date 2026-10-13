#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Check if a dtype is already supported in pypto operation config.

支持 dtype 的来源已由 operation 的 .cpp 源码迁移到 JSON 配置文件：
    framework/src/interface/configs/platform_op_supported_dtypes/{arch}_supported_op_dtypes.json

Usage:
    python3 check_dtype_support.py \
        --pypto-root /mnt/workspace/gitCode/cann/pypto \
        --operation add \
        --dtypes int64,uint64

Output: JSON with support status per dtype per architecture.
"""

import argparse
import json
from pathlib import Path
import sys

ARCH_FILES = {
    "a2a3": "a2a3_supported_op_dtypes.json",
    "a5": "a5_supported_op_dtypes.json",
    "kirin9030": "kirin9030_supported_op_dtypes.json",
    "kirinx90": "kirinx90_supported_op_dtypes.json",
}

# 用户常用名 -> JSON 字符串（STR_DATA_TYPE_MAP 小写友好名）
USER_TO_JSON_DTYPE = {
    "int4": "int4", "int8": "int8", "int16": "int16", "int32": "int32",
    "int64": "int64", "uint8": "uint8", "uint16": "uint16", "uint32": "uint32",
    "uint64": "uint64", "fp8": "fp8", "fp16": "fp16", "fp32": "float32",
    "bf16": "bf16", "bool": "bool", "double": "double",
    "fp8e4m3": "fp8e4m3", "fp8e5m2": "fp8e5m2", "fp8e8m0": "fp8e8m0",
    "hf4": "hf4", "hf8": "hf8",
}

# operation 名称 -> opcode 字符串列表（OpcodeManager 注册的 str）
OP_TO_OPCODES = {
    "add": ["ADD", "ADDS"],
    "sub": ["SUB", "SUBS"],
    "mul": ["MUL", "MULS"],
    "div": ["DIV", "DIVS"],
    "max": ["MAXIMUM", "MAXS"],
    "min": ["MINIMUM", "MINS"],
    "compare": ["CMP", "CMPS"], "eq": ["CMP", "CMPS"], "ne": ["CMP", "CMPS"],
    "gt": ["CMP", "CMPS"], "ge": ["CMP", "CMPS"], "lt": ["CMP", "CMPS"], "le": ["CMP", "CMPS"],
    "where": ["WHERE_TT", "WHERE_TS", "WHERE_ST", "WHERE_SS"],
    "gather": ["GATHER", "GATHER_ELEMENT"],
    "gather_elements": ["GATHER_ELEMENT"],
    "gather_in_ub": ["GATHER_IN_UB"],
    "scatter": ["SCATTER", "SCATTER_ELEMENT"],
    "scatter_update": ["SCATTER_UPDATE"],
    "transpose": ["TRANSPOSE_MOVEIN", "TRANSPOSE_MOVEOUT", "TRANSPOSE_VNCHWCONV"],
    "relu": ["RELU"],
    "abs": ["ABS"],
    "neg": ["NEG"],
    "round": ["ROUND"],
    "sign": ["SIGN"],
    "logical_not": ["LOGICALNOT"],
    "expand": ["EXPAND"],
    "full": ["VEC_DUP"],
    "amax": ["ROWMAX_SINGLE"], "amin": ["ROWMIN_SINGLE"], "sum": ["ROWSUM_SINGLE"],
    "remainder": ["REM", "REMS", "REMRS"],
    "floor_div": ["FLOORDIV", "FLOORDIVS"],
    "triul": ["TRIUL"],
    "bitwise_and": ["BITWISEAND", "BITWISEANDS"],
    "bitwise_or": ["BITWISEOR", "BITWISEORS"],
    "bitwise_xor": ["BITWISEXOR", "BITWISEXORS"],
    "bitwise_not": ["BITWISENOT"],
    "bitwise_right_shift": ["BITWISERIGHTSHIFT", "BITWISERIGHTSHIFTS", "SBITWISERIGHTSHIFT"],
    "bitwise_left_shift": ["BITWISELEFTSHIFT", "BITWISELEFTSHIFTS", "SBITWISELEFTSHIFT"],
}


def load_config_dir(pypto_root: Path) -> Path:
    return pypto_root / "framework" / "src" / "interface" / "configs" / "platform_op_supported_dtypes"


def get_opcodes(operation: str) -> list:
    return OP_TO_OPCODES.get(operation.lower(), [operation.upper()])


def _flatten_input_dtypes(input_dtypes) -> list:
    """将组合枚举（嵌套列表，空串表示可选槽位）展平为去重的 dtype 字符串列表。"""
    flat = []
    for item in input_dtypes:
        if isinstance(item, list):
            flat.extend(d for d in item if d)
        elif item:
            flat.append(item)
    return list(dict.fromkeys(flat))


def check_dtype(pypto_root: Path, operation: str, dtypes: list) -> dict:
    config_dir = load_config_dir(pypto_root)
    opcodes = get_opcodes(operation)

    per_arch = {}      # arch -> opcode -> list[dtype_str]
    matched_files = []

    for arch, filename in ARCH_FILES.items():
        f = config_dir / filename
        if not f.exists():
            per_arch[arch] = {}
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            per_arch[arch] = {}
            continue
        ops = (data.get("ops") or {})
        per_arch[arch] = {
            op: _flatten_input_dtypes(ops.get(op, {}).get("input_tensors_dtype_combinations", []))
            for op in opcodes
        }
        if any(op in ops for op in opcodes):
            matched_files.append(filename)

    all_types = {arch: set() for arch in ARCH_FILES}
    for arch, opmap in per_arch.items():
        for dts in opmap.values():
            all_types[arch].update(dts)

    dtype_results = {}
    for dt in dtypes:
        dt_norm = dt.lower().strip()
        json_dt = USER_TO_JSON_DTYPE.get(dt_norm, dt_norm)
        per_arch_status = {}
        for arch in ARCH_FILES:
            per_arch_status[arch] = json_dt in all_types[arch]
        supported_in = [a for a, ok in per_arch_status.items() if ok]
        if len(supported_in) == len(ARCH_FILES):
            status = "already_supported"
        elif supported_in:
            status = "partially_supported"
        else:
            status = "not_supported"
        dtype_results[dt] = {
            "json_dtype": json_dt,
            "status": status,
            "supported_in": supported_in,
        }

    return {
        "operation": operation,
        "opcodes": opcodes,
        "config_files": matched_files,
        "current_types_by_arch": {a: sorted(list(t)) for a, t in all_types.items()},
        "dtype_results": dtype_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Check dtype support in pypto operation config")
    parser.add_argument("--pypto-root", required=True, help="Path to pypto root directory")
    parser.add_argument("--operation", required=True, help="Operation name (e.g., add, sub, compare)")
    parser.add_argument("--dtypes", required=True, help="Comma-separated dtype names (e.g., int64,uint64)")
    args = parser.parse_args()

    pypto_root = Path(args.pypto_root)
    if not pypto_root.exists():
        print(json.dumps({"error": f"pypto root not found: {pypto_root}"}), file=sys.stderr)
        sys.exit(1)

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    result = check_dtype(pypto_root, args.operation, dtypes)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
