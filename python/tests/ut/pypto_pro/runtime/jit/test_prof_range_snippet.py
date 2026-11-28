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

"""Generated launch metadata for live reporting and captured graph replay."""

import importlib
from pathlib import Path
import re
import shutil
import subprocess

import pypto_pro.language as pl
from pypto_pro.runtime.compile_config import get_jit_compile_config
from pypto_pro.runtime.jit import (
    ParamKind,
    ParamSpec,
    _generate_caller_cpp,
    _generate_prof_range_snippet,
    _launch_error_message,
    _parse_and_codegen_targets,
)
from pypto_pro.runtime.platform import NpuArch
import pytest


def _snippet(kernel_name, specs, dims):
    target = get_jit_compile_config().resolve_kernel_target(NpuArch.DAV_3510, has_cube=False, has_vector=True)
    return _generate_prof_range_snippet(
        kernel_name, specs, dims, target=target, launch_stmt="    k<<<blockDim, nullptr, stream>>>(a, out);\n"
    )


def _spec(name, shape, dtype_str, direction=pl.Input):
    return ParamSpec(name, ParamKind.TENSOR, dtype_str, shape, direction)


def test_output_annotation_sets_direction():
    """pl.Output in the annotation flows to ParamSpec.direction; omitted defaults to INPUT."""
    @pl.jit()
    def marked_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.store(out, ta, [0, 0])

    kernel_def = marked_kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, NpuArch.DAV_3510, "")
    cg = cube or vector
    assert kernel_def.last_param_directions == {1: pl.Output}
    assert cg.param_specs[0].direction is pl.Input
    assert cg.param_specs[1].direction is pl.Output


def test_call_style_output_annotation_sets_direction():
    @pl.jit()
    def marked_kernel(
        a: pl.Tensor([64, 64], pl.DT_FP32),
        out: pl.Tensor([64, 64], pl.DT_FP32, pl.Output),
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.store(out, ta, [0, 0])

    kernel_def = marked_kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, NpuArch.DAV_3510, "")
    cg = cube or vector
    assert kernel_def.last_param_directions == {1: pl.Output}
    assert cg.param_specs[0].direction is pl.Input
    assert cg.param_specs[1].direction is pl.Output


def test_output_direction_uses_absolute_parameter_index():
    @pl.jit()
    def marked_kernel(
        size: pl.DT_INT32,
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.store(out, ta, [0, 0])

    kernel_def = marked_kernel.to_kernel_def()
    cube, vector = _parse_and_codegen_targets(kernel_def, NpuArch.DAV_3510, "")
    cg = cube or vector
    assert kernel_def.last_param_directions == {2: pl.Output}
    assert cg.param_specs[0].kind is ParamKind.SCALAR
    assert cg.param_specs[0].direction is None
    assert cg.param_specs[1].direction is pl.Input
    assert cg.param_specs[2].name == "out_0"
    assert cg.param_specs[2].direction is pl.Output


def test_omitted_direction_defaults_to_input():
    """An omitted direction defaults to INPUT even if the kernel stores into the tensor."""

    @pl.jit()
    def unmarked_kernel(
        out: pl.Tensor[[64, 64], pl.DT_FP32],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        with pl.section_vector():
            pl.store(out, ta, [0, 0])

    cube, vector = _parse_and_codegen_targets(unmarked_kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    cg = cube or vector
    assert cg.param_specs[0].direction is pl.Input


def test_static_shapes_are_literals():
    specs = [
        _spec("a", [64, 64], "fp32"),
        _spec("b", [64, 64], "fp32"),
        _spec("out", [64, 64], "fp32", pl.Output),
    ]
    _inc, push, pop = _snippet("k", specs, set())
    assert '#include "acl/acl_prof.h"' in _inc
    assert "{0, 2, 0, 2, {64, 64, 0, 0, 0, 0, 0, 0}}," in push  # a: input
    assert "{1, 2, 0, 2, {64, 64, 0, 0, 0, 0, 0, 0}}," in push  # out: output
    assert 'BeginCaptureTensorReport(pyptoProfOn)' in push
    assert 'ReportCaptureTensorInfo(pyptoProfInfo, pyptoProfBegin, pyptoCaptureOn)' in pop
    assert 'aclprofRange' not in push + pop
    assert 'aclprofEventAttributes' not in push
    assert "aclprofStr2Id(\"k\")" in push
    assert "pyptoProfInfo.tensorNum = 3;" in push
    assert "pyptoProfInfo.kernelType = 2;" in push


def test_dynamic_dims_reference_launcher_params():
    dims = {"a", "b", "out", "__pypto_dyn_a_0", "__pypto_dyn_a_1"}
    specs = [_spec("a", ["__pypto_dyn_a_0", "__pypto_dyn_a_1"], "fp32")]
    _inc, push, _pop = _snippet("k", specs, dims)
    assert "{static_cast<uint32_t>(__pypto_dyn_a_0), static_cast<uint32_t>(__pypto_dyn_a_1)," in push


def test_unknown_dynamic_degrades_to_zero():
    specs = [_spec("a", [-1, 8], "fp16")]
    _inc, push, _pop = _snippet("k", specs, set())
    assert "{0, 2, 1, 2, {0, 8, 0, 0, 0, 0, 0, 0}}," in push


def test_dtype_mapping():
    specs = [
        _spec("a", [4], "fp16"),
        _spec("b", [4], "bf16"),
        _spec("c", [4], "int32"),
        _spec("d", [4], "nonexistent"),
    ]
    _inc, push, _pop = _snippet("k", specs, set())
    assert "{0, 2, 1, 1," in push  # fp16 -> 1
    assert "{0, 2, 27, 1," in push  # bf16 -> 27
    assert "{0, 2, 3, 1," in push  # int32 -> 3
    assert "{0, 2, 0, 1," in push  # unknown dtype -> 0 fallback


def test_no_tensor_params_disables_snippet():
    assert _snippet("k", [], set()) == ("", "", "")


@pytest.mark.parametrize("simt_kernel", [False, True])
def test_caller_embeds_metadata_around_launch_static(simt_kernel):
    """End-to-end caller generation: pl.Output flows into the snippet."""
    @pl.jit()
    def static_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        tc = pl.make_tile(tt, addr=0x8000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.add(tc, ta, ta)
            pl.store(out, tc, [0, 0])

    @pl.jit(mode="simt", max_threads=32)
    def simt_static_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        index = pl.simt.linear_thread_idx()
        out[0, index] = a[0, index] + a[0, index]

    kernel = simt_static_kernel if simt_kernel else static_kernel
    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    from pypto_pro.runtime.compile_config import get_jit_compile_config

    target = get_jit_compile_config().resolve_kernel_target(NpuArch.DAV_3510, has_cube=cube is not None,
                                                            has_vector=vector is not None)
    cg = cube or vector
    content = _generate_caller_cpp(
        kernel_params=cg.entry_params,
        kernel_cpp_name="kernel.cpp",
        kernel_name=cg.kernel_name,
        target=target,
        prof_param_specs=cg.param_specs,
        simt_kernel=simt_kernel,
    )
    assert '#include "acl/acl_prof.h"' in content
    assert f'aclprofStr2Id("{cg.kernel_name}")' in content
    push_idx = content.index("BeginCaptureTensorReport")
    launch = (
        "<<<dim3(gridX, gridY, gridZ), dim3(threadBlockX, threadBlockY, threadBlockZ), 0, stream>>>"
        if simt_kernel else "aclrtLaunchKernelWithHostArgs("
    )
    launch_idx = content.index(launch)
    pop_idx = content.index("ReportCaptureTensorInfo")
    assert push_idx < launch_idx < pop_idx
    assert "pyptoProfInfo.kernelType = 2;" in content
    block_nums = "gridX * gridY * gridZ" if simt_kernel else "blockDim"
    assert f"pyptoProfInfo.blockNums = {block_nums};" in content
    else_idx = content.index("} else {")
    assert pop_idx < else_idx
    assert content.count(launch) == (2 if simt_kernel else 1)
    if not simt_kernel:
        assert push_idx < content.index("aclrtGetFuncBySymbol(") < launch_idx
        assert content.count("<<<blockDim") == 1
        assert "<<<blockDim" in content[else_idx:]
    assert "ReportCaptureTensorInfo" not in content[else_idx:]
    assert "{0, 2, 0, 2, {64, 64" in content   # a: input
    assert "{1, 2, 0, 2, {64, 64" in content   # out: output


def test_metadata_checks_collection_and_capture_independently():
    """Capture before collection caches tensors; live node reports need active collection."""
    @pl.jit()
    def bare_kernel(
        a: pl.Tensor[[64, 64], pl.DT_FP32],
        out: pl.Tensor[[64, 64], pl.DT_FP32, pl.Output],
    ):
        tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        ta = pl.make_tile(tt, addr=0x0000)
        tc = pl.make_tile(tt, addr=0x8000)
        with pl.section_vector():
            pl.load(ta, a, [0, 0])
            pl.add(tc, ta, ta)
            pl.store(out, tc, [0, 0])

    cube, vector = _parse_and_codegen_targets(bare_kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    from pypto_pro.runtime.compile_config import get_jit_compile_config

    target = get_jit_compile_config().resolve_kernel_target(NpuArch.DAV_3510, has_cube=cube is not None,
                                                            has_vector=vector is not None)
    cg = cube or vector
    content = _generate_caller_cpp(
        kernel_params=cg.entry_params,
        kernel_cpp_name="kernel.cpp",
        kernel_name=cg.kernel_name,
        target=target,
        prof_param_specs=cg.param_specs,
    )
    guard_idx = content.index("if (pyptoProfOn || pyptoCaptureOn) {")
    tensors_idx = content.index("aclprofTensor pyptoProfTensors[]")
    begin_idx = content.index("pypto::BeginCaptureTensorReport(pyptoProfOn)")
    lookup_idx = content.index("aclrtGetFuncBySymbol(")
    launch_idx = content.index("aclrtLaunchKernelWithHostArgs(")
    report_idx = content.index("pypto::ReportCaptureTensorInfo")
    else_idx = content.index("} else {")
    assert guard_idx < tensors_idx < begin_idx < lookup_idx < launch_idx < report_idx < else_idx
    assert "ACL_STREAM_ATTR_CACHE_OP_INFO" in content
    assert "pyptoCaptureAttr.cacheOpInfoSwitch" in content
    assert "const bool pyptoProfOn = true;" not in content
    assert "pypto::GetProfStatus()" in content
    assert "getenv" not in content
    assert content.count("<<<blockDim") == 1
    assert "ReportCaptureTensorInfo" not in content[else_idx:]


@pytest.mark.parametrize("kernel_name,expected_name", [("k", b"k"), ("custom_silu", b"custom_silu")])
def test_native_metadata_gate(tmp_path, kernel_name, expected_name):
    """Execute the emitted control flow with recording APIs; an ordinary launch must not report."""
    target = get_jit_compile_config().resolve_kernel_target(NpuArch.DAV_3510, has_cube=False, has_vector=True)
    _inc, push, pop = _generate_prof_range_snippet(
        kernel_name, [_spec("a", [64], "fp32", pl.Input)], set(), target=target, launch_stmt="    ++launches;\n",
    )
    # Only API types/functions and the device launch are stubs; compile the actual generated guard.
    harness = r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
static const unsigned char expectedName[] = {EXPECTED_NAME_BYTES, 0};
constexpr int ACL_SUCCESS = 0, ACL_STREAM_ATTR_CACHE_OP_INFO = 7;
struct aclrtStreamAttrValue { bool cacheOpInfoSwitch; };
struct aclprofTensor { uint32_t type, format, dataType, dimNum, shape[8]; };
struct aclprofTensorInfo {
    uint64_t opNameId, opTypeId;
    uint32_t resv, tensorNum, kernelType, blockNums;
    void* stream;
    aclprofTensor* tensors;
};
static int profiling, capture, queryError, queries, launches, reports, caches, strings, activeBegins;
namespace pypto {
static bool GetProfStatus() { return profiling; }
}
static int aclrtGetStreamAttribute(void* stream, int attribute, aclrtStreamAttrValue* value)
{
    assert(stream == reinterpret_cast<void*>(0x1234));
    assert(attribute == ACL_STREAM_ATTR_CACHE_OP_INFO);
    ++queries;
    value->cacheOpInfoSwitch = capture;
    return queryError;
}
static uint64_t aclprofStr2Id(const char* name) {
    const char* expected = strings == 0 ? reinterpret_cast<const char*>(expectedName) : "PyPTO";
    assert(std::strcmp(name, expected) == 0);
    return ++strings;
}
namespace pypto {
static uint64_t BeginCaptureTensorReport(bool profiling)
{
    activeBegins += profiling;
    return profiling ? 123 : 0;
}
static void ReportCaptureTensorInfo(const aclprofTensorInfo& info, uint64_t begin, bool cache)
{
    assert(launches == 1);
    assert(info.tensors[0].shape[0] == 64);
    reports += begin != 0;
    caches += cache;
}
}
static void launch(uint32_t blockDim, void* stream)
{
""" + push + "    ++launches;\n" + pop + r"""
}
int main()
{
    // Reuse the same stream as collection/capture changes to expose a cached guard.
    // Fields: process-wide collection, capture, query error.
    const int cases[][3] = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}, {0, 0, 0}};
    for (const auto& test : cases) {
        profiling = test[0]; capture = test[1]; queryError = test[2];
        queries = launches = reports = caches = strings = activeBegins = 0;
        launch(8, reinterpret_cast<void*>(0x1234));
        std::printf("%d %d %d %d %d %d\n", launches, reports, caches, strings, activeBegins, queries);
    }
}
"""
    source = tmp_path / "range_gate.cpp"
    source.write_text(harness.replace("EXPECTED_NAME_BYTES", ", ".join(map(str, expected_name))), encoding="utf-8")
    binary = tmp_path / "range_gate"
    subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    actual = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    # Counts: launches, live reports, caches, registered strings, active begins, capture queries.
    assert actual == [
        (1, 0, 0, 0, 0, 1),  # No collection, no capture.
        (1, 0, 1, 2, 0, 1),  # Capture before collection only caches.
        (1, 1, 0, 2, 1, 1),  # Collection alone reports live metadata.
        (1, 1, 1, 2, 1, 1),  # Capture during collection must also cache for later replay.
        (1, 1, 0, 2, 1, 1),  # Collection still reports despite a failed capture query.
        (1, 0, 0, 0, 0, 1),  # A failed capture query alone does not report.
        (1, 0, 0, 0, 0, 1),  # Returning to ordinary execution must stop all metadata calls.
    ]


@pytest.fixture(scope="module")
def native_profiler_launcher(tmp_path_factory):
    compiler = shutil.which("bisheng")
    if compiler is None:
        pytest.skip("bisheng is required to compile the physical scalar ABI")
    root = tmp_path_factory.mktemp("profiler_host_launch")
    (root / "acl").mkdir()
    (root / "acl/acl_rt.h").write_text("""#pragma once
#include <cstddef>
#include <cstdint>
using aclError = int32_t;
using aclrtStream = void*;
using aclrtFuncHandle = void*;
constexpr int ACL_SUCCESS = 0, ACL_STREAM_ATTR_CACHE_OP_INFO = 7;
enum aclrtDevResLimitType { ACL_RT_DEV_RES_CUBE_CORE, ACL_RT_DEV_RES_VECTOR_CORE };
struct aclrtStreamAttrValue { bool cacheOpInfoSwitch; };
constexpr int ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE = 2;
struct aclrtLaunchKernelAttr { int id; struct { uint32_t dynUBufSize; } value; };
struct aclrtLaunchKernelCfg { aclrtLaunchKernelAttr* attrs; size_t numAttrs; };
extern "C" aclError aclrtGetStreamResLimit(aclrtStream, aclrtDevResLimitType, uint32_t*);
extern "C" aclError aclrtGetStreamAttribute(aclrtStream, int, aclrtStreamAttrValue*);
extern "C" aclError aclrtGetFuncBySymbol(const void*, aclrtFuncHandle*);
extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*, void*, size_t, void*, size_t);
""")
    (root / "acl/acl_prof.h").write_text("""#pragma once
#include <cstdint>
struct aclprofTensor { uint32_t type, format, dataType, shapeDim, shape[8]; };
struct aclprofTensorInfo {
    uint64_t opNameId, opTypeId;
    uint32_t resv, tensorNum, kernelType, blockNums;
    void* stream;
    aclprofTensor* tensors;
};
uint64_t aclprofStr2Id(const char*);
""")
    (root / "pypto_profiler.h").write_text("""#pragma once
namespace pypto {
bool GetProfStatus();
uint64_t BeginCaptureTensorReport(bool);
void ReportCaptureTensorInfo(const aclprofTensorInfo&, uint64_t, bool);
}
""")
    (root / "kernel.h").write_text("#pragma once\ninline void probe() {}\n")
    params = [
        ("float", "x", True), ("int8_t", "narrow", False), ("float", "scale", False),
        ("float", "out", True), ("int64_t", "rows", False),
    ]
    specs = [
        ParamSpec("x", ParamKind.TENSOR, "fp32", ["rows", 64], pl.Input),
        ParamSpec("out", ParamKind.TENSOR, "fp32", ["rows", 64], pl.Output),
    ]
    target = get_jit_compile_config().resolve_kernel_target(
        NpuArch.DAV_3510, has_cube=False, has_vector=True,
    )
    caller = _generate_caller_cpp(
        params, "kernel.h", "probe", target=target, prof_param_specs=specs,
        required_dynamic_ub_size=256,
    )
    # Only ordinary ASC dispatch is replaced; execute the actual profiling path.
    caller, substitutions = re.subn(
        r"::probe<<<[^\n]+?>>>\([^\n]+?\);", ";", caller,
    )
    assert substitutions == 1
    harness = r"""
#include <acl/acl_rt.h>
#include <acl/acl_prof.h>
#include "kernel.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static int lookupError, launchError;
static uint32_t lookups, launches, reports, caches;
extern "C" aclError aclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t* value)
{
    assert(stream == reinterpret_cast<void*>(0x1234));
    assert(type == ACL_RT_DEV_RES_VECTOR_CORE);
    *value = 8;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtGetStreamAttribute(aclrtStream stream, int attr, aclrtStreamAttrValue* value)
{
    assert(stream == reinterpret_cast<void*>(0x1234) && attr == ACL_STREAM_ATTR_CACHE_OP_INFO);
    value->cacheOpInfoSwitch = true;
    return ACL_SUCCESS;
}
extern "C" aclError aclrtGetFuncBySymbol(const void* symbol, aclrtFuncHandle* function)
{
    ++lookups;
    assert(symbol == reinterpret_cast<const void*>(&probe));
    *function = reinterpret_cast<void*>(0x5678);
    return lookupError;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg* config,
    void* args, size_t size, void* placeholders, size_t placeholderCount)
{
    ++launches;
    assert(function == reinterpret_cast<void*>(0x5678) && blocks == 3);
    assert(stream == reinterpret_cast<void*>(0x1234));
    assert(config && config->numAttrs == 1);
    assert(config->attrs[0].id == ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE);
    assert(config->attrs[0].value.dynUBufSize == 256);
    assert(placeholders == nullptr && placeholderCount == 0);
    assert(size == 5 * sizeof(uint64_t));
    uint64_t slots[5]{};
    std::memcpy(slots, args, size);
    assert(slots[0] == 0x1000 && slots[3] == 0x2000);
    assert(static_cast<int64_t>(slots[1]) == -7);
    assert(slots[2] == 0x3fa00000 && slots[4] == 65);
    return launchError;
}
uint64_t aclprofStr2Id(const char*) { return 1; }
namespace pypto {
bool GetProfStatus() { return true; }
uint64_t BeginCaptureTensorReport(bool enabled) { return enabled ? 123 : 0; }
void ReportCaptureTensorInfo(const aclprofTensorInfo& info, uint64_t begin, bool cache)
{
    assert(launches == 1 && launchError == 0);
    assert(info.stream == reinterpret_cast<void*>(0x1234) && info.blockNums == 3);
    assert(info.tensorNum == 2 && info.tensors[0].type == 0 && info.tensors[1].type == 1);
    assert(info.tensors[0].shape[0] == 65 && info.tensors[1].shape[0] == 65);
    reports += begin != 0;
    caches += cache;
}
}
"""
    source = root / "launcher.cpp"
    source.write_text(harness + caller + r"""
int main(int argc, char** argv)
{
    assert(argc == 3);
    lookupError = std::atoi(argv[1]);
    launchError = std::atoi(argv[2]);
    const int64_t result = call_kernel(3, reinterpret_cast<void*>(0x1234), reinterpret_cast<uint8_t*>(0x1000),
                                       -7, 1.25f, reinterpret_cast<uint8_t*>(0x2000), 65);
    std::printf("%lld %u %u %u %u\n", static_cast<long long>(result), lookups, launches, reports, caches);
}
""")
    binary = root / "launcher"
    include = Path(importlib.import_module("pypto_pro.runtime.jit").__file__).resolve().parents[1] / "include"
    result = subprocess.run(
        [compiler, "-x", "c++", "-std=c++17", "-O2",
         f"-I{root}", f"-I{include}", str(source), "-o", str(binary)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize(
    "lookup_error,launch_error,expected,counts",
    [
        (0, 0, 3, (1, 1, 1, 1)),
        (507001, 0, -((6 << 32) | 507001), (1, 0, 0, 0)),
        (0, 507002, -((7 << 32) | 507002), (1, 1, 0, 0)),
    ],
)
def test_profiler_host_launch(native_profiler_launcher, lookup_error, launch_error, expected, counts):
    """Check the new HostArgs ABI and stop on either API failure; gates are tested above."""
    result = subprocess.run(
        [str(native_profiler_launcher), str(lookup_error), str(launch_error)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    actual, *actual_counts = map(int, result.stdout.split())
    assert actual == expected
    assert tuple(actual_counts) == counts
    if expected < 0:
        message = _launch_error_message(expected, 3, None)
        assert f"ACL error {(lookup_error or launch_error):#x}" in message
        assert ("looking up" if lookup_error else "launching") in message
