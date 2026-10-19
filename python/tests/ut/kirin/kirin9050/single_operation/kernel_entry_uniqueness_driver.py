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

"""Build one Kirin9050 kernel and print the entry name recorded in its JSON sidecar.

Run as a subprocess by test_kirin9050_kernel_entry_uniqueness.py, which copies this file once per build. The
separate interpreter is the point: the function id feeding the magic name comes from a process-global counter,
so two builds in one interpreter never share a magic name and could not exercise entry-name uniqueness.

The kernel is always named add_kernel, so its source name cannot be what separates two builds; the operation
comes from argv. The test asserts that both builds collide on the magic name, so drift between the two
definitions below fails rather than passing silently.
"""

import glob
import json
import os
import re
import sys

import torch

import pypto

_OP = sys.argv[1] if len(sys.argv) > 1 else ""

if _OP == "add":

    @pypto.frontend.jit(
        codegen_options={"soc_version": "Kirin9050"},
        runtime_options={"run_mode": pypto.RunMode.SIM},
    )
    def add_kernel(
        input0: pypto.Tensor([...], pypto.DT_INT32),
        input1: pypto.Tensor([...], pypto.DT_INT32),
        output: pypto.Tensor([...], pypto.DT_INT32),
    ):
        pypto.set_vec_tile_shapes(200)
        output[:] = pypto.add(input0, input1)

elif _OP == "sub":

    @pypto.frontend.jit(
        codegen_options={"soc_version": "Kirin9050"},
        runtime_options={"run_mode": pypto.RunMode.SIM},
    )
    def add_kernel(
        input0: pypto.Tensor([...], pypto.DT_INT32),
        input1: pypto.Tensor([...], pypto.DT_INT32),
        output: pypto.Tensor([...], pypto.DT_INT32),
    ):
        pypto.set_vec_tile_shapes(200)
        output[:] = pypto.sub(input0, input1)

else:
    raise SystemExit(f"unknown operation {_OP!r}; expected one of: add, sub")


def main():
    shape = (160,)
    input0 = torch.randint(-100, 100, shape, dtype=torch.int32)
    input1 = torch.randint(-100, 100, shape, dtype=torch.int32)
    output = torch.empty(shape, dtype=torch.int32)

    # The sidecar is written during code generation, which precedes execution. Executing the kernel is merely
    # the public way to reach code generation, so a launch-side failure still leaves the artifact to inspect.
    try:
        add_kernel(input0, input1, output)
    except Exception as exc:
        print("EXECUTION_FAILED %s: %s" % (type(exc).__name__, exc), file=sys.stderr)

    # CodeGenCCE::PrepareDefaultOutputPath creates kernel_aicore whenever code generation runs, independently
    # of the CANN build that fills it, so an absent directory means code generation never happened -- a
    # breakage, not an environment limit. (KEY_FIXED_OUTPUT_PATH / IsFixedCceMode relocate the directory, so
    # this driver expects the default output layout.)
    sidecar_dir = os.path.join(pypto.pypto_impl.LogTopFolder(), "kernel_aicore")
    if not os.path.isdir(sidecar_dir):
        print("NO_CODEGEN %s" % sidecar_dir)
        return 5

    names = set()
    for path in sorted(glob.glob(os.path.join(sidecar_dir, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            name = json.load(handle).get("kernelName")
        if name:
            names.add(name)

    if not names:
        print("NO_SIDECAR %s" % sidecar_dir)
        return 3
    if len(names) > 1:
        print("AMBIGUOUS_SIDECAR %s" % sorted(names))
        return 4

    # The sidecar's kernelName and the symbol in the emitted kernel source are written by two independent
    # code paths -- GenConfigJson appends the suffix itself, while the source comes from a template that
    # spells it inline -- so agreement between them is a real assertion, not a restatement of one value.
    symbols = set()
    for path in sorted(glob.glob(os.path.join(sidecar_dir, "*.cpp"))):
        with open(path, encoding="utf-8") as handle:
            found = re.findall(r'extern\s+"C".*?void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', handle.read())
        symbols.update(found)

    print("ENTRY_NAME=%s" % names.pop())
    for symbol in sorted(symbols):
        print("CCE_SYMBOL=%s" % symbol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
