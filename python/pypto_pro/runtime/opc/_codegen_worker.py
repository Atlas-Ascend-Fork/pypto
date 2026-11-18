# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone OPC codegen pool entry; never imports the caller's __main__ in workers."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import MagicMock

# Run this file by path, before importing pypto_pro (which imports torch). Neither
# the coordinator nor its spawned workers needs Torch, an NPU, or OPC runtime state.
if __name__ in ("__main__", "__mp_main__"):
    sys.modules["torch"] = MagicMock(name="torch")
    sys.modules["torch_npu"] = MagicMock(name="torch_npu")

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

    # Preserve an overlay or an embedding application's explicit module search path.
    # It is set before the pool is started, hence also before these imports.
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
    try:
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
    except Exception:
        # PyPTO exceptions can hold native Spans, which cannot be pickled.
        return {"error": f"tilingkey {packed} ({concrete}):\n{traceback.format_exc()}"}


def main():
    started = time.perf_counter()
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sys.path[:] = request["python_path"]
    print(f"PyPTO codegen: {len(request['tasks'])} tilingkeys, {request['workers']} workers", flush=True)
    pool = ProcessPoolExecutor(
        max_workers=request["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(request,),
    )
    futures = {}
    try:
        futures = {pool.submit(_generate, task): index for index, task in enumerate(request["tasks"])}
        results = [None] * len(futures)
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                raise RuntimeError(f"dtype={request['dtype_consts']}: {result['error']}")
            results[futures[future]] = result
        Path(sys.argv[2]).write_text(json.dumps(results), encoding="utf-8")
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    print(f"PyPTO codegen completed in {time.perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
