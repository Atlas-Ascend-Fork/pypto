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

"""Utility to build pipeline-transformed source files."""

from __future__ import annotations

import ast
import copy


def build_generated_file_source(
    transformed_func_def: ast.FunctionDef,
    source_file: str,
    line_offset: int,
    col_offset: int,
    source_lines_raw: list,
) -> str:
    """Build a complete, runnable .py source for the transformed kernel.

    Reads the original source file and replaces the kernel function definition
    (incl. decorators) with the transformed version. The ``pipeline=`` decorator kwarg
    is stripped so the generated file does not re-trigger the transform if run directly.
    """
    fd = copy.deepcopy(transformed_func_def)
    for dec in fd.decorator_list:
        if isinstance(dec, ast.Call):
            dec.keywords = [kw for kw in dec.keywords if kw.arg != "pipeline"]
    new_func_src = ast.unparse(fd)

    try:
        with open(source_file, encoding="utf-8") as f:
            orig_lines = f.read().split("\n")
    except OSError:
        return new_func_src

    start = line_offset
    end = start + len(source_lines_raw)
    indent = " " * col_offset
    new_block = [indent + ln if ln else ln for ln in new_func_src.split("\n")]
    result = orig_lines[:start] + new_block + orig_lines[end:]
    text = "\n".join(result) if isinstance(result, list) else result

    # Reparsed so every node carries its position in this text. The transformed tree's
    # generated statements have none, and the parse that follows reports errors against
    # this file.
    reparsed = ast.parse(text)
    # Found by position as well as by name: the kernel may be nested in a factory function,
    # and two factories may each hold one of the same name.
    span = range(start + 1, start + 1 + len(new_block))
    func_def = next(
        n for n in ast.walk(reparsed)
        if isinstance(n, ast.FunctionDef) and n.name == fd.name and n.lineno in span
    )
    return text, func_def
