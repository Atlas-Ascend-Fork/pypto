# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND OF EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AICPU offline codegen CLI: the configure-time entry invoked by ops-transformer's
``enable_pypto_aicpu_kernel``.

Generates the CpuKernel adapter TU + shared Args header + opInfo json into
--out-dir; compile/link/json-merge/install all happen in
the regular ops-transformer AICPU custom-package chain.

Run directly (the cmake invokes it as)::

    python pypto_aicpu_codegen_cli.py --py-file <op>.py --out-dir <gen> \
        --op-file <op_file> --soc ascend950 [--arch 3510] [--op-type <T>]
"""

from __future__ import annotations

import argparse
import logging
import sys


def _install_build_only_torch_stub():
    # Configure-time codegen needs no real torch runtime; stub the runtime-only deps.
    from unittest.mock import MagicMock

    sys.modules["torch"] = MagicMock(name="torch")
    sys.modules["torch_npu"] = MagicMock(name="torch_npu")


# CMake ASCEND_COMPUTE_UNIT soc name -> default __NPU_ARCH__ version.
# Only ascend950 is supported today; append (soc_prefix, arch) pairs to extend.
_SOC_DEFAULT_ARCH = (("ascend950", "3510"),)


def _default_arch_for_soc(soc: str) -> str | None:
    """Resolve the default __NPU_ARCH__ version for a CMake ASCEND_COMPUTE_UNIT soc name.

    Soc names are matched case-insensitively by prefix; returns None when the soc
    has no default, so the caller can require an explicit --arch instead of guessing.
    """
    text = (soc or "").strip().lower()
    for prefix, arch in _SOC_DEFAULT_ARCH:
        if text.startswith(prefix):
            return arch
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="PyPTO AICPU kernel codegen CLI")
    parser.add_argument("--py-file", required=True, help="kernel python source file")
    parser.add_argument("--out-dir", required=True, help="directory to place generated artifacts")
    parser.add_argument("--op-file", required=True, help="op file stem")
    parser.add_argument("--soc", default=None, help="soc compute unit (e.g. ascend950)")
    parser.add_argument("--arch", default=None, help="__NPU_ARCH__ override (3510)")
    parser.add_argument("--op-type", default=None, help="opInfo op type key (default: kernel name)")
    args = parser.parse_args()

    from pathlib import Path

    py_file = Path(args.py_file).expanduser().resolve()
    if not py_file.is_file():
        logging.error("py-file not found: %s", py_file)
        return 1

    arch = args.arch or _default_arch_for_soc(args.soc or "")
    if arch is None:
        # Only ascend950 has a default today; unsupported socs must pass --arch.
        logging.error("soc %r is not supported; pass --arch explicitly", args.soc)
        return 1

    _install_build_only_torch_stub()
    from pypto_pro.runtime.opc import prepare_aicpu_kernel_sources

    result = prepare_aicpu_kernel_sources(
        str(py_file), arch=arch, op_type=args.op_type, out_dir=args.out_dir
    )
    generated = sorted(p.name for p in Path(result.build_dir).iterdir() if p.is_file())
    logging.info(
        "%s: kernel '%s' op_type '%s' soc '%s' arch '%s' generated %s into %s",
        args.op_file, result.kernel_name, result.op_type, args.soc, arch,
        generated, args.out_dir,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    sys.exit(main())
