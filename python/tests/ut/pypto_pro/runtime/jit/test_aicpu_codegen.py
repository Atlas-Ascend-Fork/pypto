# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fast-launch codegen for AICPU kernels (fast_kernel_launch_example shape).

``generate_aicpu_sources`` renders the kernel translation unit and the
self-contained host launch translation unit (compat macros + flat entry
declaration + the ``Run`` shim, which forwards the flattened arguments to
``Kernel<<<1, nullptr, stm>>>(args...)``; bisheng ASC dialect, the lowering
dispatches through the ACL runtime).
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.runtime.aicpu_compile import generate_aicpu_bare_body, generate_aicpu_sources
from pypto_pro.runtime.jit import _extract_param_specs


def _generate(kernel):
    kdef = kernel.to_kernel_def()
    program, _ = kdef.parse_target_program(ir.SectionKind.Vector)
    specs = _extract_param_specs(program, None)
    body = generate_aicpu_bare_body(program, __import__("pypto_pro").runtime.platform.NpuArch.DAV_3510, kernel.__name__)
    return generate_aicpu_sources(body, specs, kernel.__name__)


def test_generates_fast_launch_host_source():
    @pl.jit(mode="aicpu_kernel")
    def kernel(
        t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
        out: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT32, pl.Output],
        n: pl.DT_INT32 = 0,
    ):
        for i in pl.range(n):
            pl.setval(out, i, pl.getval(t, i))

    sources = _generate(kernel)
    kernel_src = sources["kernel_source"]
    host_src = sources["host_source"]

    # Full golden comparison (PRO-TEST-002): any line drift in any of the two
    # generated TUs fails here -- substring spot-checks cannot catch added/removed
    # lines or formatting drift.
    golden_dir = Path(__file__).parent
    for artifact, golden_name in (
        (kernel_src, "test_aicpu_codegen__kernel_source.g.txt"),
        (host_src, "test_aicpu_codegen__host_source.g.txt"),
    ):
        golden = (golden_dir / golden_name).read_text(encoding="utf-8")
        assert artifact.rstrip("\n") == golden.rstrip("\n"), f"artifact drifted from {golden_name}"

    # Key contract spot-checks (kept as human-readable anchors on top of the goldens).
    # Flattened launch ABI: no packed Args struct or void* raw on either side.
    assert "kernelKernel(void* raw)" not in kernel_src
    assert "struct kernelArgs" not in host_src
    assert 'extern "C" __global__ __aicpu__ uint32_t kernelKernel(int32_t* t_0,' in kernel_src
    # Bare-body contract: no device qualifiers or CCE scaffolding anywhere in the TU.
    for token in ("__gm__", "__aicore__", "#if defined(__DAV", "SyncedTile", "pypto_dump.h"):
        assert token not in kernel_src, f"bare body leaked {token}"
    # Host TU: the flattened <<<1, nullptr, stm>>> fast-launch form.
    assert (
        "kernelKernel<<<1, nullptr, static_cast<aclrtStream>(stm)>>>("
        "t_0, out_0, n_0, __pypto_dyn_t_0, __pypto_dyn_out_0);" in host_src
    )
    assert "sizeof(kernelArgs)" not in host_src

    if shutil.which("g++"):
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp, "kernel_aicpu.cpp")
            cpp.write_text(kernel_src)
            # Only the kernel TU is g++-checkable: the host TU carries the
            # <<<>>> compiler extension and the acl headers (bisheng only).
            subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", f"-I{tmp}", str(cpp)],
                check=True,
                capture_output=True,
                text=True,
            )


def test_status_abi_emission():
    @pl.jit(mode="aicpu_kernel")
    def status_kernel(
        t: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
        out: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT32, pl.Output],
        n: pl.DT_INT32 = 0,
    ):
        if n < 0:
            return 1
        if n == 5:
            pl.setval(out, 0, 5)
            return
        for i in pl.range(n):
            pl.setval(out, i, pl.getval(t, i))

    sources = _generate(status_kernel)
    src = sources["kernel_source"]

    # The impl carries the status ABI: uint32_t return, explicit `return 1;`
    # failure exit, bare returns (branch + fall-through) as `return 0;`.
    assert "inline uint32_t status_kernel_impl_vector(" in src
    assert src.count("return 1;") == 1
    # one bare branch return + the fall-through tail
    assert src.count("return 0;") == 2
    assert "return;\n" not in src
    # The __global__ entry propagates the status instead of hardcoding 0.
    assert "return status_kernel_impl_vector(" in src

    if shutil.which("g++"):
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp, "status_kernel_aicpu.cpp")
            cpp.write_text(src)
            subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", f"-I{tmp}", str(cpp)],
                check=True,
                capture_output=True,
                text=True,
            )
