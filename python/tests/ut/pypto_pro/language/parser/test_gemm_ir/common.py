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
"""Shared GEMM parser driver for the ``test_gemm_ir`` golden cases."""

import importlib.util
import itertools
import textwrap

from pypto_pro import ir

_SERIAL = itertools.count()


def parse_gemm(tmp_path, body, *, k=128, n=64, l0k=None, l0n=None, compact=1,
               acc_compact=1, acc_depth=1, trans_a=False, trans_b=False,
               auto=False, mx=False, plane_compact=0, bad_plane=False, static_validshape=False):
    """Use a real source file so both the parser and location diagnostics are exercised."""
    filename = tmp_path / f"gemm_case_{next(_SERIAL)}.py"
    ak, bn = l0k or k, l0n or n
    a_shape = [k, 64] if trans_a else [64, k]
    b_shape = [n, k] if trans_b else [k, n]
    dtype = "DT_FP8E4M3FN" if mx else "DT_FP16"
    def group_sync(values):
        return f'mutex_ids="auto", depth={len(values)}' if auto else f"mutex_ids={values}"
    decls = [
        ("ga", a_shape, dtype, "Mat", "NZ", 0, [0, 1], 0),
        ("gb", b_shape, dtype, "Mat", "NZ", 65536, [2, 3], 0),
        ("la", [64, ak], dtype, "Left", "NZ", 0, [4, 5], compact),
        ("lb", [ak, bn], dtype, "Right", "ZN", 0, [6, 7], compact),
        ("gc", [64, n], "DT_FP32", "Acc", "NZ", 0, list(range(8, 8 + acc_depth)), acc_compact),
    ]
    declarations = []
    for name, shape, dt, space, layout, addr, mutex, comp in decls:
        valid = f", valid_shape={shape}" if static_validshape else ""
        declarations.append(
            f"{name} = pl.make_tile_group(type=pl.TileType(shape={shape}, dtype=pl.{dt}, "
            f"target_memory=pl.MemorySpace.{space}, layout=pl.{layout}, compact={comp}{valid}), "
            f"addrs={addr}, {group_sync(mutex)})"
        )
    if mx:
        data_bytes = 64 * ak
        for name, shape, space, layout, addr, mutex in (
            ("gsa", [64, k // 32], "Mat", "ZZ", 131072, [10, 11]),
            ("gsb", [k // 32, n], "Mat", "NN", 135168, [12, 13]),
            ("sa", [64, ak // 32], "ScaleLeft", "ZZ",
             [0, (data_bytes >> 4) + (64 if bad_plane else 0)], [14, 15]),
            ("sb", [ak // 32, bn], "ScaleRight", "NN", [0, data_bytes >> 4], [16, 17]),
        ):
            declarations.append(
                f"{name} = pl.make_tile_group(type=pl.TileType(shape={shape}, dtype=pl.DT_FP8E8M0, "
                f"target_memory=pl.MemorySpace.{space}, layout=pl.{layout}, compact={plane_compact}), "
                f"addrs={addr}, {group_sync(mutex)})"
            )
        declarations += ["s1 = gsa[0]", "s2 = gsb[0]"]
    declarations += ["a = ga[0]", "b = gb[0]", "c = gc[0]"]
    source = (
        "import pypto_pro.language as pl\nfrom pypto_pro.language import gemm as product\n"
        "@pl.jit(auto_mutex=True)\n"
        "def kernel(arg: pl.Tensor[[1], pl.DT_FP16], extent: pl.DT_INT32):\n"
        + textwrap.indent("\n".join(declarations) + "\n" + textwrap.dedent(body).strip(), "    ") + "\n"
    )
    filename.write_text(source)
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program, _ = module.kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    program = ir.Pass.insert_sync()(program)
    return str(program.get_function("kernel"))
