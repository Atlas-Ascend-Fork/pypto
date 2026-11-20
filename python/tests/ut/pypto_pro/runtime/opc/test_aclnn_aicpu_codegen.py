# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT of the AICPU offline CpuKernel adapter generation semantics.

``generate_cpu_kernel_sources`` is the configure-time adapter generator of the
ops-transformer custom package. Its contract (aicpu kernel mode):

* ``pl.Tensor`` and ``pl.Ptr`` parameters are both plain tensors: the adapter
  fills each slot with a null-tolerant ``ctx.Input``/``ctx.Output`` data-pointer
  extraction and emits no dtype/shape/non-null judgment (the kernel body's
  ``is None`` guards own presence); dynamic dims pack 0 for an absent tensor,
  like the fast-launch None packing.
* Scalars with a Python default are optional (default pre-assigned, then
  ``GetAttrValueOpt``); scalars without one are required (``GetAttrValue``,
  missing attr -> ``kStatusParamInvalid``).
* The ``REGISTER_CPU_KERNEL`` pair sits in an anonymous namespace, matching the
  handwritten convention, so the shared ``kernelType`` identifier stays TU-local.
"""

from importlib import import_module
import os
from pathlib import Path
import shutil
import subprocess

import pytest

jit = import_module("pypto_pro.runtime.jit")
opc_codegen = import_module("pypto_pro.runtime.opc.aclnn_aicpu_codegen")
pl = import_module("pypto_pro.language")

ParamKind = jit.ParamKind
ParamSpec = jit.ParamSpec
TensorDirection = import_module("pypto_pro.language.typing.direction").TensorDirection

_D = pl.DataType


def _spec_tensor(name, dtype, shape, direction=None):
    return ParamSpec(name, ParamKind.TENSOR, dtype, list(shape), direction)


def _spec_ptr(name, dtype):
    return ParamSpec(name, ParamKind.PTR, dtype, None)


def _spec_scalar(name, dtype):
    return ParamSpec(name, ParamKind.SCALAR, dtype, None)


def _generate(param_specs, param_names, defaults, cce_body="// body\n", op_type="ProbeOp"):
    return opc_codegen.generate_cpu_kernel_sources(cce_body, param_specs, "probe_op", op_type, param_names, defaults)


class TestTensorPtrUnifiedExtraction:
    """pl.Tensor and pl.Ptr fill their Input/Output slots identically, without checks."""

    def _sources(self):
        specs = [
            _spec_tensor("cu_q", _D.INT32, ["__pypto_dyn_cu_q_0"]),
            _spec_tensor("out", _D.INT32, ["__pypto_dyn_out_0"], TensorDirection.OUTPUT),
            _spec_ptr("seqused", _D.INT32),
            _spec_tensor("cmp_res", _D.INT32, ["__pypto_dyn_cmp_res_0"]),
        ]
        return _generate(specs, ["cu_q", "out", "seqused", "cmp_res"], {})

    def test_both_kinds_use_null_tolerant_pointer_extraction(self):
        src = self._sources()["kernel_source"]
        assert (
            "    aicpu::Tensor* t_in0 = ctx.Input(0);\n"
            "    args.cu_q = (t_in0 != nullptr) ? static_cast<int32_t*>(t_in0->GetData()) : nullptr;\n"
        ) in src
        assert (
            "    aicpu::Tensor* t_in1 = ctx.Input(1);\n"
            "    args.seqused = (t_in1 != nullptr) ? static_cast<int32_t*>(t_in1->GetData()) : nullptr;\n"
        ) in src

    def test_output_slot_uses_ctx_output(self):
        src = self._sources()["kernel_source"]
        assert "aicpu::Tensor* t_out0 = ctx.Output(0);" in src
        assert ("args.out = (t_out0 != nullptr) ? static_cast<int32_t*>(t_out0->GetData()) : nullptr;") in src

    def test_no_presence_dtype_or_shape_judgments_are_emitted(self):
        src = self._sources()["kernel_source"]
        assert "GetDataType" not in src
        assert "GetDimSize(0) !=" not in src
        # No early return guarding a tensor: the only allowed return paths are
        # the missing required attr and the kStatusOk tail.
        assert "{ return kStatusParamInvalid; }" not in src.split("REGISTER")[0].split("Compute(")[1]

    def test_dynamic_dims_pack_zero_for_absent_tensor(self):
        src = self._sources()["kernel_source"]
        assert (
            "args.__pypto_dyn_cu_q_0 = (t_in0 != nullptr) "
            "? static_cast<int64_t>(t_in0->GetTensorShape()->GetDimSize(0)) : 0;"
        ) in src
        assert (
            "args.__pypto_dyn_out_0 = (t_out0 != nullptr) "
            "? static_cast<int64_t>(t_out0->GetTensorShape()->GetDimSize(0)) : 0;"
        ) in src

    def test_args_header_keeps_pointers_then_dyn_tail(self):
        header = self._sources()["args_header"]
        assert header.index("int32_t* cu_q;") < header.index("int32_t* seqused;")
        assert header.index("int32_t* cmp_res;") < header.index("int64_t __pypto_dyn_cu_q_0;")

    def test_static_shape_dims_emit_no_check(self):
        specs = [_spec_tensor("data", _D.FP32, [8])]
        src = _generate(specs, ["data"], {})["kernel_source"]
        assert "GetDimSize" not in src
        assert "GetDataType" not in src


class TestScalarAttrAdaptation:
    """Scalar optionality comes from the Python default, via GetAttrValue/Opt."""

    def _sources(self):
        specs = [
            _spec_tensor("data", _D.FP32, []),
            _spec_scalar("must_i", _D.INT64),
            _spec_scalar("must_b", _D.BOOL),
            _spec_scalar("must_f", _D.FP32),
            _spec_scalar("opt_i", _D.INT64),
            _spec_scalar("opt_b", _D.BOOL),
            _spec_scalar("opt_f", _D.FP32),
        ]
        names = ["data", "must_i", "must_b", "must_f", "opt_i", "opt_b", "opt_f"]
        defaults = {"opt_i": 7, "opt_b": True, "opt_f": 1.5}
        return _generate(specs, names, defaults)

    def test_required_scalars_use_getattrvalue(self):
        src = self._sources()["kernel_source"]
        assert 'if (!GetAttrValue(ctx, "must_i", args.must_i)) { return kStatusParamInvalid; }' in src
        assert 'if (!GetAttrValue(ctx, "must_b", args.must_b)) { return kStatusParamInvalid; }' in src
        assert 'if (!GetAttrValue(ctx, "must_f", args.must_f)) { return kStatusParamInvalid; }' in src

    def test_optional_scalars_preassign_default_then_getattrvalueopt(self):
        src = self._sources()["kernel_source"]
        assert "args.opt_i = static_cast<int64_t>(7);" in src
        assert 'GetAttrValueOpt(ctx, "opt_i", args.opt_i);' in src
        assert "args.opt_b = static_cast<bool>(true);" in src
        assert 'GetAttrValueOpt(ctx, "opt_b", args.opt_b);' in src
        assert "args.opt_f = static_cast<float>(1.5);" in src
        assert 'GetAttrValueOpt(ctx, "opt_f", args.opt_f);' in src

    def test_legacy_getattr_api_is_gone(self):
        src = self._sources()["kernel_source"]
        assert "ctx.GetAttr(" not in src.split("namespace aicpu {")[0].split("// Kernel body")[0]
        assert "AttrValue* attr_" not in src.split("// Kernel body")[1]

    def test_helpers_emitted_only_with_scalars(self):
        with_scalars = _generate([_spec_tensor("data", _D.FP32, []), _spec_scalar("k", _D.INT64)], ["data", "k"], {})[
            "kernel_source"
        ]
        assert "GetAttrValue(CpuKernelContext& ctx" in with_scalars
        assert "#include <type_traits>" in with_scalars
        without_scalars = _generate([_spec_tensor("data", _D.FP32, [])], ["data"], {})["kernel_source"]
        assert "GetAttrValueOpt" not in without_scalars
        assert "#include <type_traits>" not in without_scalars

    def test_helpers_match_handwritten_contract(self):
        src = self._sources()["kernel_source"]
        helpers = src.split("namespace aicpu {", 1)[1].split("// Kernel body", 1)[0]
        assert "std::is_integral_v<T>" in helpers
        assert "attr->GetInt()" in helpers
        assert "attr->GetBool()" in helpers
        assert "attr->GetFloat()" in helpers
        assert helpers.count("static typename std::enable_if") == 2
        assert helpers.count("} // namespace aicpu") == 1


class TestUnsupportedParams:
    def test_tiling_and_window_stay_rejected(self):
        from pypto_pro._errors import InvalidVal

        specs = [_spec_tensor("data", _D.FP32, []), ParamSpec("tiling", ParamKind.TILING, None, None)]
        with pytest.raises(InvalidVal, match="TILING"):
            _generate(specs, ["data", "tiling"], {})
        specs = [_spec_tensor("data", _D.FP32, []), ParamSpec("win", ParamKind.WINDOW, None, None)]
        with pytest.raises(InvalidVal, match="WINDOW"):
            _generate(specs, ["data", "win"], {})

    def test_param_names_must_align_with_specs(self):
        from pypto_pro._errors import InvalidVal

        specs = [_spec_tensor("data", _D.FP32, [])]
        with pytest.raises(InvalidVal, match="param_names must align 1:1"):
            _generate(specs, ["data", "extra"], {})


class TestEndToEndAicpuKernel:
    """One real ``@pl.jit(mode="aicpu_kernel")`` kernel through _prepare_from_kernel."""

    @pytest.fixture()
    def kernel(self):
        @pl.jit(mode="aicpu_kernel")
        def offline_adapter_probe(
            cu_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
            out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
            seqused: pl.Ptr[pl.DT_INT32] = None,
            opt_buf: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input] = None,
            heads: pl.DT_INT64 = 1,
        ):
            total = pl.numel(cu_q)
            if seqused is not None:
                total += 1
            if opt_buf is not None:
                total += pl.numel(opt_buf)
            pl.setval(out, 0, total)

        return offline_adapter_probe

    def test_tensor_and_ptr_optionals_adapt_identically(self, kernel, tmp_path):
        result = opc_codegen._prepare_from_kernel(kernel, "3510", None, tmp_path, "libprobe.so")
        src = (tmp_path / "offline_adapter_probe_aicpu_kernel.cpp").read_text(encoding="utf-8")
        # Same slot ordering as the signature: cu_q -> Input(0), the Output is
        # independent, then seqused (Ptr) and opt_buf (Tensor) take Input(1)/(2).
        assert "aicpu::Tensor* t_in0 = ctx.Input(0);" in src
        assert "aicpu::Tensor* t_out0 = ctx.Output(0);" in src
        assert "aicpu::Tensor* t_in1 = ctx.Input(1);" in src
        assert "aicpu::Tensor* t_in2 = ctx.Input(2);" in src
        assert src.count("!= nullptr) ? static_cast<int32_t*>(t_") == 4
        # No adapter-level judgment remains: only the required-attr channel.
        assert "GetDataType" not in src
        # The single scalar has a default -> optional adaptation (the Args field
        # carries the IR name with the _0 suffix; the attr name is the source name).
        assert "args.heads_0 = static_cast<int64_t>(1);" in src
        assert 'GetAttrValueOpt(ctx, "heads", args.heads_0);' in src
        # Absent optional tensor's dynamic dim packs 0.
        assert (
            "args.__pypto_dyn_opt_buf_0 = (t_in2 != nullptr) "
            "? static_cast<int64_t>(t_in2->GetTensorShape()->GetDimSize(0)) : 0;"
        ) in src
        # Status ABI: the adapter maps the impl's KERNEL_STATUS return onto the
        # CpuKernel status channel instead of unconditionally returning OK.
        assert "uint32_t status = offline_adapter_probe_impl_vector(" in src
        assert "return status == 0 ? kStatusOk : kStatusParamInvalid;" in src
        assert "    return kStatusOk;" not in src
        assert result.kernel_cpp_path.endswith("offline_adapter_probe_aicpu_kernel.cpp")


# The registry stub stands in for libaicpu_context.a: the link under test is about
# the generated TUs' own symbols, not about the runtime registry implementation.
_REGIST_STUB = r"""
#include <functional>
#include <memory>
#include <string>
#include "cpu_kernel.h"
namespace aicpu {
bool RegistCpuKernel(const std::string& type, const KERNEL_CREATOR_FUN& fun) { return true; }
}
"""


def _aicpu_include_dirs():
    """The dir providing cpu_kernel.h; empty when no CANN install is reachable.

    ``ASCEND_HOME_PATH`` wins (the repo convention); the standard install prefix is
    the fallback for hosts that ship the headers without exporting the env var.
    """
    roots = []
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        roots.append(Path(home))
    roots.extend(sorted(Path("/usr/local/Ascend").glob("cann-*/*-linux"), reverse=True))
    for root in roots:
        for rel in (("include", "aicpu"), ("pkg_inc", "aicpu", "cpu_kernels")):
            candidate = root.joinpath(*rel)
            if (candidate / "cpu_kernel.h").is_file():
                return [str(candidate)]
    return []


def _write_generated_tu(tmp_path, op_name, op_type, unwrap=False):
    """Emit one generated adapter TU (+ Args header) shaped like the CMake chain's."""
    specs = [_spec_tensor("data", _D.FP32, [])]
    impl = opc_codegen._impl_name(op_name)
    body = f"uint32_t {impl}(float* data) {{ (void)data; return 0; }}\n"
    sources = opc_codegen.generate_cpu_kernel_sources(body, specs, op_name, op_type, ["data"], {})
    lines = sources["kernel_source"].splitlines()
    if unwrap:
        lines = [ln for ln in lines if ln not in ("namespace {", "} // namespace")]
    (tmp_path / f"{op_name}_aicpu_kernel.cpp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / f"{op_name}_aicpu_args.h").write_text(sources["args_header"], encoding="utf-8")
    return tmp_path / f"{op_name}_aicpu_kernel.cpp"


def _link_two_tus(tmp_path, unwrap):
    """Compile two generated TUs and link them into one .so, like gen_aicpu_kernel_symbol."""
    includes = _aicpu_include_dirs()
    if not shutil.which("g++") or not includes:
        pytest.skip("g++ and CANN aicpu headers required to link the generated adapters")
    stub = tmp_path / "regist_stub.cpp"
    stub.write_text(_REGIST_STUB, encoding="utf-8")
    tus = [
        _write_generated_tu(tmp_path, "probe_alpha", "ProbeAlpha", unwrap),
        _write_generated_tu(tmp_path, "probe_beta", "ProbeBeta", unwrap),
    ]
    flags = ["g++", "-std=c++17", "-fPIC", f"-I{tmp_path}", *(f"-I{d}" for d in includes)]
    objects = []
    for src in (stub, *tus):
        obj = src.with_suffix(".o")
        subprocess.run([*flags, "-c", str(src), "-o", str(obj)], check=True, capture_output=True, text=True)
        objects.append(str(obj))
    return subprocess.run(
        ["g++", "-shared", "-o", str(tmp_path / "libprobe.so"), *objects], capture_output=True, text=True
    )


class TestRegistrationLinkage:
    """REGISTER_CPU_KERNEL pastes the ``kernelType`` token into external symbol names.

    The macro emits ``Creator_##type##_Kernel()`` and ``g_##type##_Kernel_Creator``
    with external linkage, so every TU registering through the shared ``kernelType``
    identifier exports the same symbols. Handwritten ops keep them TU-local with an
    anonymous namespace (e.g. ``dense_lightning_indexer_kl_loss_grad_metadata_aicpu.cpp``
    declares the class at ``namespace aicpu`` scope and wraps only the registration
    pair); the generated adapter must match, or two generated objects collide once
    ``gen_aicpu_kernel_symbol`` merges them into ``libtransformer_aicpu_kernels.so``.
    Only the op_type string has to be globally unique -- it is the registry key.
    """

    def _source(self, op_type="ProbeOp"):
        return _generate([_spec_tensor("data", _D.FP32, [])], ["data"], {}, op_type=op_type)["kernel_source"]

    def test_registration_pair_is_wrapped_in_anonymous_namespace(self):
        src = self._source()
        assert (
            "namespace {\n"
            'static const char* kernelType = "ProbeOp";\n'
            "REGISTER_CPU_KERNEL(kernelType, ProbeOpCpuKernel);\n"
            "} // namespace\n"
        ) in src

    def test_registry_key_is_the_op_type_not_the_identifier(self):
        # The identifier is fixed by the template; the string is the per-op registry
        # key and must track op_type (it also keys the opInfo json).
        src = self._source("MixedQuantSparseFlashMlaPyptoMetadata")
        assert 'static const char* kernelType = "MixedQuantSparseFlashMlaPyptoMetadata";' in src
        assert "REGISTER_CPU_KERNEL(kernelType," in src

    def test_two_generated_tus_link_into_one_shared_object(self, tmp_path):
        result = _link_two_tus(tmp_path, unwrap=False)
        assert result.returncode == 0, result.stderr
        assert "multiple definition" not in result.stderr

    def test_unwrapped_registration_would_collide(self, tmp_path):
        # Negative control: dropping the anonymous namespace reproduces the exact
        # duplicate-symbol failure the wrapper prevents, so the test above is sensitive.
        result = _link_two_tus(tmp_path, unwrap=True)
        assert result.returncode != 0
        assert "multiple definition of `aicpu::Creator_kernelType_Kernel()'" in result.stderr
        assert "multiple definition of `aicpu::g_kernelType_Kernel_Creator'" in result.stderr
