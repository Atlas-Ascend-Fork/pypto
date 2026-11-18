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
"""

from importlib import import_module

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


def _generate(param_specs, param_names, defaults, cce_body="// body\n"):
    return opc_codegen.generate_cpu_kernel_sources(cce_body, param_specs, "probe_op", "ProbeOp", param_names, defaults)


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
