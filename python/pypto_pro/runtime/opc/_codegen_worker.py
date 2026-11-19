# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate assigned tiling keys in an isolated interpreter."""

from multiprocessing.connection import Connection
from pathlib import Path
import sys
import traceback
from unittest.mock import MagicMock

_KERNEL = None
_ARCH = None
_DTYPE_CONSTS = None
_OUTPUT_DIR = None


def _initialize(request):
    global _KERNEL, _ARCH, _DTYPE_CONSTS, _OUTPUT_DIR
    from pypto_pro import DataType
    from pypto_pro.runtime.jit import _setup_arch_env
    from pypto_pro.runtime.opc.pypto_compile import _load_kernel
    from pypto_pro.runtime.platform import normalize_arch

    _ARCH = _setup_arch_env(normalize_arch(request["arch"]))
    _KERNEL = _load_kernel(request["source"])
    types = {
        str(value): value for name in dir(DataType)
        if isinstance(value := getattr(DataType, name), DataType)
    }
    _DTYPE_CONSTS = (
        {name: types[dtype] for name, dtype in request["dtype_consts"].items()}
        if request["dtype_consts"] is not None else None
    )
    _OUTPUT_DIR = request["output_dir"]


def _generate(task):
    from pypto_pro.runtime.jit import _codegen

    packed, concrete = task
    cg = _codegen(
        _KERNEL.to_kernel_def(concrete, _DTYPE_CONSTS),
        _ARCH,
        clean_up=False,
        tilingkey_packed=packed,
        out_dir=str(Path(_OUTPUT_DIR, f"_cg_tk_{packed}")),
    )
    if cg is None:
        raise RuntimeError("codegen returned no result")
    # Only plain metadata crosses the process boundary; IR and DataType
    # objects are native objects, and the generated C++ is already on disk.
    return {
        "build_dir": cg.build_dir,
        "kernel_name": cg.kernel_name,
        "entry_params": cg.entry_params,
        "has_cube": cg.has_cube,
        "has_vector": cg.has_vector,
    }


def main(control):
    request, task = control.recv()
    sys.path[:] = request["python_path"]
    for name in request["mock_modules"]:
        sys.modules[name] = MagicMock(name=name)
    _initialize(request)
    while task is not None:
        try:
            result = _generate(task)
        except Exception as exc:
            packed, concrete = task
            raise RuntimeError(f"dtype={request['dtype_consts']}: tilingkey {packed} ({concrete})") from exc
        control.send(result)
        task = control.recv()


if __name__ == "__main__":
    with Connection(int(sys.argv[1])) as connection:
        try:
            main(connection)
        except Exception:
            # Finish the diagnostic before EOF tells OPC to reap the process group.
            traceback.print_exc()
            sys.exit(1)
