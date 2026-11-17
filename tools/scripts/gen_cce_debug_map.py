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
"""Scan kernel_aicore/*.cpp and emit cce_opmagic_map.json / cce_semantic_label.json for PYPTO TOOLKIT."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import re
import sys
import tempfile
from typing import Dict, Iterable, List, Set, Union

OPMAGIC_MAP_FILENAME = "cce_opmagic_map.json"
SEMANTIC_LABEL_FILENAME = "cce_semantic_label.json"

FUNC_HASH_RE = re.compile(r"^//\s*funcHash:\s*(\d+)\s*$")
OPMAGIC_RE = re.compile(r"//\s*\[opMagic:(\d+)\](?:\[semanticLabel:([^\]]+)\])?")

LineEntry = Union[int, List[int]]
# leafHash -> opMagic -> filename -> lines
OpMagicMap = Dict[str, Dict[str, Dict[str, Set[int]]]]
# leafHash -> filename -> label -> lines
SemanticLabelMap = Dict[str, Dict[str, Dict[str, Set[int]]]]


def compact_line_ranges(lines: Iterable[int]) -> List[LineEntry]:
    ordered = sorted(set(lines))
    if not ordered:
        return []
    result: List[LineEntry] = []
    start = end = ordered[0]
    for line in ordered[1:]:
        if line == end + 1:
            end = line
            continue
        result.append(start if start == end else [start, end])
        start = end = line
    result.append(start if start == end else [start, end])
    return result


def parse_cce_file(path: str) -> tuple[str, Dict[str, Set[int]], Dict[str, Set[int]]]:
    """Return (leaf_hash, opMagic->lines, label->lines)."""
    leaf_hash = ""
    op_lines: Dict[str, Set[int]] = defaultdict(set)
    label_lines: Dict[str, Set[int]] = defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.rstrip("\n\r")
            if not leaf_hash:
                m_hash = FUNC_HASH_RE.match(text)
                if m_hash:
                    leaf_hash = m_hash.group(1)
            m_op = OPMAGIC_RE.search(text)
            if not m_op:
                continue
            if not leaf_hash:
                continue
            op_magic = m_op.group(1)
            op_lines[op_magic].add(line_no)
            label = m_op.group(2)
            if label:
                label_lines[label].add(line_no)
    return leaf_hash, op_lines, label_lines


def scan_kernel_dir(cce_dir: str) -> tuple[OpMagicMap, SemanticLabelMap]:
    op_map: OpMagicMap = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    label_map: SemanticLabelMap = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    if not os.path.isdir(cce_dir):
        return op_map, label_map

    for name in sorted(os.listdir(cce_dir)):
        if not name.endswith(".cpp"):
            continue
        path = os.path.join(cce_dir, name)
        if not os.path.isfile(path):
            continue
        leaf_hash, op_lines, label_lines = parse_cce_file(path)
        if not leaf_hash:
            continue
        for op_magic, lines in op_lines.items():
            op_map[leaf_hash][op_magic][name] = set(lines)
        for label, lines in label_lines.items():
            label_map[leaf_hash][name][label] = set(lines)
    return op_map, label_map


def op_map_to_json(op_map: OpMagicMap) -> List[dict]:
    records: List[dict] = []
    for leaf_hash, by_op in sorted(op_map.items()):
        for op_magic, by_file in sorted(by_op.items()):
            for filename, lines in sorted(by_file.items()):
                records.append(
                    {
                        "leafhash": leaf_hash,
                        "opmagic": op_magic,
                        "cce": filename,
                        "line": compact_line_ranges(lines),
                    }
                )
    return records


def label_map_to_json(label_map: SemanticLabelMap) -> List[dict]:
    records: List[dict] = []
    for leaf_hash, by_file in sorted(label_map.items()):
        for filename, by_label in sorted(by_file.items()):
            for label, lines in sorted(by_label.items()):
                records.append(
                    {
                        "leafhash": leaf_hash,
                        "semanticLabel": label,
                        "cce": filename,
                        "line": compact_line_ranges(lines),
                    }
                )
    return records


def _write_json_tmp(path: str, payload: List[dict]) -> str:
    """Write JSON to a sibling temp file; caller publishes with os.replace."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


def write_maps(cce_dir: str) -> tuple[str, str]:
    """Emit both maps as a paired publish: stage temps, then os.replace each.

    Staging both temps before any replace means a failure while writing the second
    payload leaves the previously published pair untouched (no half-new / half-old).
    """
    op_map, label_map = scan_kernel_dir(cce_dir)
    op_path = os.path.join(cce_dir, OPMAGIC_MAP_FILENAME)
    label_path = os.path.join(cce_dir, SEMANTIC_LABEL_FILENAME)
    op_tmp = label_tmp = None
    try:
        op_tmp = _write_json_tmp(op_path, op_map_to_json(op_map))
        label_tmp = _write_json_tmp(label_path, label_map_to_json(label_map))
        os.replace(op_tmp, op_path)
        op_tmp = None
        os.replace(label_tmp, label_path)
        label_tmp = None
    finally:
        for tmp in (op_tmp, label_tmp):
            if tmp is None:
                continue
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return op_path, label_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate cce_opmagic_map.json and cce_semantic_label.json from kernel_aicore CCE sources."
    )
    parser.add_argument(
        "cce_dir",
        help="Directory containing generated *.cpp CCE files (typically kernel_aicore).",
    )
    args = parser.parse_args(argv)
    cce_dir = os.path.abspath(args.cce_dir)
    if not os.path.isdir(cce_dir):
        print(f"[gen_cce_debug_map] directory not found: {cce_dir}", file=sys.stderr)
        return 1
    write_maps(cce_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
