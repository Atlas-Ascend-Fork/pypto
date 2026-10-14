/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file backend_cce_simt_ops.cpp
 * \brief Direct CCE lowering for the A5 SIMT context and launch operations.
 */

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/scalar_expr.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace backend {
using npu::tile_fwk::ExternalError;

namespace {

const char* GetSimtAxisName(int axis)
{
    constexpr const char* axis_names[] = {"x", "y", "z"};
    return axis_names[axis];
}

std::string MakeSimtThreadIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/)
{
    return std::string("threadIdx.") + GetSimtAxisName(op->GetKwarg<int>("axis"));
}

std::string MakeSimtBlockDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/)
{
    return std::string("blockDim.") + GetSimtAxisName(op->GetKwarg<int>("axis"));
}

std::string MakeSimtBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/)
{
    return std::string("blockIdx.") + GetSimtAxisName(op->GetKwarg<int>("axis"));
}

std::string MakeSimtGridDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/)
{
    return std::string("gridDim.") + GetSimtAxisName(op->GetKwarg<int>("axis"));
}

std::string MakeSimtLinearThreadIdxCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& /*codegen_base*/)
{
    return "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";
}

std::string MakeSimtWarpSizeCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& /*codegen_base*/)
{
    return "warpSize";
}

std::string MakeSimtSyncthreadsCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    codegen.Emit("__sync_workitems();");
    return "";
}

std::string MakeSimtThreadfenceBlockCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    codegen.Emit("__threadfence_block();");
    return "";
}

std::string MakeSimtThreadfenceCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    codegen.Emit("__threadfence();");
    return "";
}

std::string MakeSimtWarpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const char* intrinsic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::ostringstream call;
    call << intrinsic << "(";
    for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i != 0)
            call << ", ";
        call << codegen.GetExprAsCode(op->args_[i]);
    }
    call << ")";
    return call.str();
}

const char* GetSimtCastRoundModeCCE(ir::RoundMode mode)
{
    switch (mode) {
        case ir::RoundMode::CAST_RINT:
            return "ROUND::R";
        case ir::RoundMode::CAST_ROUND:
            return "ROUND::A";
        case ir::RoundMode::CAST_FLOOR:
            return "ROUND::F";
        case ir::RoundMode::CAST_CEIL:
            return "ROUND::C";
        case ir::RoundMode::CAST_TRUNC:
            return "ROUND::Z";
        case ir::RoundMode::CAST_ODD:
            return "ROUND::O";
        default:
            return "ROUND::R";
    }
}

const char* GetSimtCastIntrinsicCCE(ir::DataType dtype)
{
    if (dtype == ir::DataType::FP16) {
        return "__cvt_half";
    }
    if (dtype == ir::DataType::BF16) {
        return "__cvt_bfloat16_t";
    }
    if (dtype == ir::DataType::FP32) {
        return "__cvt_float";
    }
    if (dtype == ir::DataType::INT32) {
        return "__cvt_int32_t";
    }
    if (dtype == ir::DataType::UINT32) {
        return "__cvt_uint32_t";
    }
    if (dtype == ir::DataType::INT64) {
        return "__cvt_int64_t";
    }
    return "__cvt_uint64_t";
}

std::string MakeSimtCastIntrinsicCallCCE(ir::DataType target_dtype, ir::RoundMode mode, const char* saturation,
                                         const std::string& operand)
{
    const char* intrinsic = GetSimtCastIntrinsicCCE(target_dtype);
    return std::string(intrinsic) + "<" + GetSimtCastRoundModeCCE(mode) + ", " + saturation + ">(" + operand + ")";
}

std::string MakeSimtCastNarrowIntegerCCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    // A5 has no direct low-float-to-8/16-bit scalar conversion. Match the reference semantics by converting to
    // a 32-bit carrier first, then clamp before narrowing. This is fixed per-conversion behavior, not public SatMode.
    const bool is_signed = target_dtype == ir::DataType::INT8 || target_dtype == ir::DataType::INT16;
    const ir::DataType carrier_dtype = is_signed ? ir::DataType::INT32 : ir::DataType::UINT32;
    const std::string converted = MakeSimtCastIntrinsicCallCCE(carrier_dtype, mode,
                                                               "RoundingSaturation::RS_ENABLE_VALUE", operand);

    const char* upper_bound = nullptr;
    const char* lower_bound = nullptr;
    if (target_dtype == ir::DataType::INT8) {
        upper_bound = "127";
        lower_bound = "-128";
    } else if (target_dtype == ir::DataType::UINT8) {
        upper_bound = "255U";
    } else if (target_dtype == ir::DataType::INT16) {
        upper_bound = "32767";
        lower_bound = "-32768";
    } else {
        upper_bound = "65535U";
    }

    const std::string target_type = target_dtype.ToCTypeString();
    const std::string carrier_type = carrier_dtype.ToCTypeString();
    std::ostringstream s;
    s << "({" << carrier_type << " __simt_cast_value = " << converted << ";"
      << "__simt_cast_value > " << upper_bound << " ? (" << target_type << ")" << upper_bound << " : ";
    if (is_signed) {
        s << "(__simt_cast_value < " << lower_bound << " ? (" << target_type << ")" << lower_bound << " : ("
          << target_type << ")__simt_cast_value);";
    } else {
        s << "(" << target_type << ")__simt_cast_value;";
    }
    s << "})";
    return s.str();
}

std::string MakeSimtCastPlainCCE(ir::DataType target_dtype, const std::string& operand)
{
    return "((" + target_dtype.ToCTypeString() + ")" + operand + ")";
}

std::string MakeSimtCastFromFp16OrBf16CCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    if (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16 ||
        target_dtype == ir::DataType::FP32) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT32 || target_dtype == ir::DataType::UINT32)) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_ENABLE_VALUE", operand);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT64 || target_dtype == ir::DataType::UINT64)) {
        const std::string fp32 = MakeSimtCastIntrinsicCallCCE(ir::DataType::FP32, mode,
                                                              "RoundingSaturation::RS_ENABLE_VALUE", operand);
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_ENABLE_VALUE", fp32);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT8 || target_dtype == ir::DataType::UINT8 ||
         target_dtype == ir::DataType::INT16 || target_dtype == ir::DataType::UINT16)) {
        return MakeSimtCastNarrowIntegerCCE(target_dtype, mode, operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, operand);
}

std::string MakeSimtCastFromFp32CCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    if (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::INT32 || target_dtype == ir::DataType::UINT32 ||
         target_dtype == ir::DataType::INT64 || target_dtype == ir::DataType::UINT64)) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_ENABLE_VALUE", operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, operand);
}

std::string MakeSimtCastFromInt16OrUint16CCE(ir::DataType source_dtype, ir::DataType target_dtype, ir::RoundMode mode,
                                             const std::string& operand)
{
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16)) {
        const ir::DataType carrier_dtype = source_dtype == ir::DataType::INT16 ? ir::DataType::INT32 :
                                                                                 ir::DataType::UINT32;
        const std::string widened = "((" + carrier_dtype.ToCTypeString() + ")" + operand + ")";
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", widened);
    }
    return MakeSimtCastPlainCCE(target_dtype, operand);
}

std::string MakeSimtCastFromInt32OrUint32CCE(ir::DataType target_dtype, ir::RoundMode mode, const std::string& operand)
{
    if (mode != ir::RoundMode::CAST_NONE && (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16 ||
                                             target_dtype == ir::DataType::FP32)) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, operand);
}

std::string MakeSimtCastFromInt64OrUint64CCE(ir::DataType source_dtype, ir::DataType target_dtype, ir::RoundMode mode,
                                             const std::string& operand)
{
    if (mode != ir::RoundMode::CAST_NONE &&
        (target_dtype == ir::DataType::FP16 || target_dtype == ir::DataType::BF16)) {
        // The A5 uint64-to-BF16 reference path always uses nearest-even for the uint64-to-FP32 carrier conversion.
        const auto fp32_mode = source_dtype == ir::DataType::UINT64 && target_dtype == ir::DataType::BF16 ?
                                   ir::RoundMode::CAST_RINT :
                                   mode;
        const std::string fp32 = MakeSimtCastIntrinsicCallCCE(ir::DataType::FP32, fp32_mode,
                                                              "RoundingSaturation::RS_DISABLE_VALUE", operand);
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", fp32);
    }
    if (mode != ir::RoundMode::CAST_NONE && target_dtype == ir::DataType::FP32) {
        return MakeSimtCastIntrinsicCallCCE(target_dtype, mode, "RoundingSaturation::RS_DISABLE_VALUE", operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, operand);
}

std::string MakeSimtCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto source_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    auto target_type = ir::As<ir::ScalarType>(op->GetType());

    const auto source_dtype = source_type->dtype_;
    const auto target_dtype = target_type->dtype_;
    const auto mode = static_cast<ir::RoundMode>(op->GetKwarg<int>("mode"));
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    if (source_dtype == target_dtype) {
        return operand;
    }

    if (source_dtype == ir::DataType::INT16 || source_dtype == ir::DataType::UINT16) {
        return MakeSimtCastFromInt16OrUint16CCE(source_dtype, target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::INT32 || source_dtype == ir::DataType::UINT32) {
        return MakeSimtCastFromInt32OrUint32CCE(target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::INT64 || source_dtype == ir::DataType::UINT64) {
        return MakeSimtCastFromInt64OrUint64CCE(source_dtype, target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::FP16 || source_dtype == ir::DataType::BF16) {
        return MakeSimtCastFromFp16OrBf16CCE(target_dtype, mode, operand);
    }
    if (source_dtype == ir::DataType::FP32) {
        return MakeSimtCastFromFp32CCE(target_dtype, mode, operand);
    }
    return MakeSimtCastPlainCCE(target_dtype, operand);
}

std::string MakeSimtBitcastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto source_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    auto target_type = ir::As<ir::ScalarType>(op->GetType());

    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    std::ostringstream s;
    s << "({union {" << source_type->dtype_.ToCTypeString() << " source;" << target_type->dtype_.ToCTypeString()
      << " target;} __simt_bitcast_data;"
      << "__simt_bitcast_data.source = (" << operand << ");"
      << "__simt_bitcast_data.target;})";
    return s.str();
}

std::string MakeSimtTrigFP32Codegen(const std::string& operand, bool is_sin)
{
    std::ostringstream s;
    s << "({"
      << "float __t = (" << operand << ");"
      << "__t = __fma(__t, 0.0f, __t);"
      << "int __q;"
      << "float __y;"
      << "if (__fabsf(__t) > 71476.0625f) {"
      << "uint32_t __bits = reinterpret_cast<uint32_t&>(__t);"
      << "int32_t __exp = ((__bits & 0x7F800000) >> 23) - 127;"
      << "uint32_t __ei = (uint32_t)__exp >> 5;"
      << "const uint32_t __tbl[] = {0x517cc1b7, 0x27220a94, 0xfe13abe8, 0xfa9a6ee0, 0x6db14acc, 0x9e21c820};"
      << "uint32_t __hi = __ei ? __tbl[__ei - 1] : 0;"
      << "uint32_t __mid = __tbl[__ei];"
      << "uint32_t __lo = __tbl[__ei + 1];"
      << "uint32_t __last = __tbl[__ei + 2];"
      << "int32_t __er = (uint32_t)__exp & 0x1F;"
      << "if (__er) {"
      << "__hi = (__hi << __er) | (__mid >> (32 - __er));"
      << "__mid = (__mid << __er) | (__lo >> (32 - __er));"
      << "__lo = (__lo << __er) | (__last >> (32 - __er));"
      << "}"
      << "uint32_t __mant = (__bits & 0x007FFFFF) | 0x4F000000;"
      << "uint32_t __nmant = (uint32_t)reinterpret_cast<float&>(__mant);"
      << "uint64_t __prod = (uint64_t)__nmant * __lo;"
      << "__prod = (uint64_t)__nmant * __mid + (__prod >> 32);"
      << "__prod = ((uint64_t)(__nmant * __hi) << 32) + __prod;"
      << "int32_t __quot = (int32_t)(__prod >> 62);"
      << "__prod &= 0x3FFFFFFFFFFFFFFFULL;"
      << "if (__prod & 0x2000000000000000ULL) { __prod -= 0x4000000000000000ULL; __quot++; }"
      << "int64_t __pi = (int64_t)__prod;"
      << "float __hf = (float)__pi;"
      << "__pi -= (int64_t)__hf;"
      << "float __lf = (float)__pi;"
      << "__y = (__hf + __lf) * 3.4061215800865545e-19f;"
      << "if (__t < 0.0f) { __y = -__y; __quot = -__quot; }"
      << "__q = __quot;"
      << "} else {"
      << "float __r = __fma(__t, 0.636619747f, 12582912.0f);"
      << "__q = reinterpret_cast<int&>(__r);"
      << "__r -= 12582912.0f;"
      << "__t = __fma(__r, -1.57079601e+00f, __t);"
      << "__t = __fma(__r, -3.13916473e-07f, __t);"
      << "__y = __fma(__r, -5.39030253e-15f, __t);"
      << "}"
      << "float __yy = __y * __y;"
      << "float __m = __fma(__y, __yy, 0.0f);"
      << "float __z = __fma(__yy, 2.86567956e-6f, -1.98559923e-4f);"
      << "__z = __fma(__yy, __z, 8.33338592e-3f);"
      << "__z = __fma(__yy, __z, -1.66666672e-1f);"
      << "float __s = __fma(__z, __m, __y);"
      << "float __c = __fma(__yy, 2.44677067e-5f, -1.38877297e-3f);"
      << "__c = __fma(__yy, __c, 4.16666567e-2f);"
      << "__c = __fma(__yy, __c, -5.00000000e-1f);"
      << "__c = __fma(__yy, __c, 1.00000000e+0f);"
      << "if (__q & 2) { __s = -__s; __c = -__c; }";
    if (is_sin) {
        s << "if (__q & 1) { __s = __c; }"
          << "__s;";
    } else {
        s << "if (__q & 1) { __c = -__s; }"
          << "__c;";
    }
    s << "})";
    return s.str();
}

// Scalar algorithms are emitted directly into the CCE expression.
std::string MakeSimtSqrtFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __sqrt_x = (" << operand << ");"
      << "float __sqrt_result;"
      << "if (__sqrt_x < 1.17549435e-38f) {"
      << "__sqrt_result = __sqrtf(__sqrt_x * 16777216.0f) * 0.000244140625f;"
      << "} else {"
      << "__sqrt_result = __sqrtf(__sqrt_x);"
      << "}"
      << "__sqrt_result;})";
    return s.str();
}

std::string MakeSimtExp2FP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __exp2_x = (" << operand << ");"
      << "float __exp2_res;"
      << "{"
      << "float __exp2_x_hi = __exp2_x, __exp2_x_lo = 0.0f;"
      << "constexpr float __exp2_ln2 = 0.69314718246459960938f;"
      << "constexpr float __exp2_overflow_abs_bound ="
      << "152.0f;"
      << "constexpr int32_t __exp2_fp32_exponent_shift = 23;"
      << "const float __exp2_rounded_x = __roundf(__exp2_x_hi);"
      << "const float __exp2_exp2_fraction = (__exp2_x_hi - __exp2_rounded_x) + __exp2_x_lo;"
      << "const int32_t __exp2_exp2_exponent = __cvt_int32_t<ROUND::Z,"
      << " RoundingSaturation::RS_ENABLE_VALUE>(__exp2_rounded_x);"
      << "float __exp2_exp_poly = __fma(__exp2_exp2_fraction, 0.00015239251661114395f, 0.0013391353422775864601f);"
      << "__exp2_exp_poly = __fma(__exp2_exp2_fraction, __exp2_exp_poly, 0.0096188392490148544312f);"
      << "__exp2_exp_poly = __fma(__exp2_exp2_fraction, __exp2_exp_poly, 0.055503588169813156128f);"
      << "__exp2_exp_poly = __fma(__exp2_exp2_fraction, __exp2_exp_poly, 0.24022644758224487305f);"
      << "__exp2_exp_poly = __fma(__exp2_exp2_fraction, __exp2_exp_poly, __exp2_ln2);"
      << "__exp2_exp_poly = __fma(__exp2_exp2_fraction, __exp2_exp_poly, 1.0f);"
      << "const bool __exp2_rounded_x_is_positive = __exp2_rounded_x > 0.0f;"
      << "const uint32_t __exp2_scale_hi_bits = __exp2_rounded_x_is_positive ? 0x7F000000U : 0x02000000U;"
      << "const uint32_t __exp2_scale_adjust = __exp2_rounded_x_is_positive ? 0U : 0x83000000U;"
      << "const uint32_t __exp2_scale_lo_bits = (static_cast<uint32_t>(__exp2_exp2_exponent) <<"
      << " __exp2_fp32_exponent_shift) - __exp2_scale_adjust;"
      << "float __exp2_output = __exp2_exp_poly * ({uint32_t __bits_value = (__exp2_scale_hi_bits);"
      << " reinterpret_cast<float&>(__bits_value);});"
      << "__exp2_output = __exp2_output * ({uint32_t __bits_value = (__exp2_scale_lo_bits);"
      << " reinterpret_cast<float&>(__bits_value);});"
      << "if (__fabsf(__exp2_x_hi) > __exp2_overflow_abs_bound) {"
      << "__exp2_output = __exp2_x_hi >= 0.0f ? __builtin_inff() : 0.0f;"
      << "}"
      << "__exp2_res = __exp2_output;"
      << "}"
      << "if (__isnan(__exp2_x)) {"
      << "__exp2_res = __exp2_x;"
      << "}"
      << "if (__exp2_x == __builtin_inff()) {"
      << "__exp2_res = __builtin_inff();"
      << "}"
      << "if (__exp2_x == -__builtin_inff()) {"
      << "__exp2_res = 0.0f;"
      << "}"
      << "__exp2_res;})";
    return s.str();
}

std::string MakeSimtLog2FP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __log2_x = (" << operand << ");"
      << "float __log2_log2_hi = 0.0f;"
      << "float __log2_log2_lo = 0.0f;"
      << "{"
      << "constexpr float __log2_subnormal_scale = 16777216.0f;"
      << "constexpr float __log2_subnormal_exponent_fix = -24.0f;"
      << "constexpr float __log2_log_exponent_scale ="
      << "1.1920928955078125e-07f;"
      << "constexpr uint32_t __log2_log_reduction_mask = 0xFF800000U;"
      << "constexpr uint32_t __log2_sqrt_half_bits ="
      << "0x3F3504F3U;"
      << "constexpr float __log2_log2e_hi = 1.4426950216293334961f;"
      << "constexpr float __log2_log2e_lo = 1.9251366722983220825e-08f;"
      << "const bool __log2_is_normal_x = __log2_x >= 1.17549435e-38f;"
      << "const float __log2_log_input = __log2_is_normal_x ? __log2_x : __log2_x * __log2_subnormal_scale;"
      << "const float __log2_exponent_base = __log2_is_normal_x ? 0.0f : __log2_subnormal_exponent_fix;"
      << "const uint32_t __log2_log_input_bits = ({float __bits_value = (__log2_log_input);"
      << " reinterpret_cast<uint32_t&>(__bits_value);});"
      << "const uint32_t __log2_reduction_bits = (__log2_log_input_bits - __log2_sqrt_half_bits) &"
      << " __log2_log_reduction_mask;"
      << "const float __log2_mantissa = ({uint32_t __bits_value = (__log2_log_input_bits - __log2_reduction_bits);"
      << " reinterpret_cast<float&>(__bits_value);});"
      << "const float __log2_exponent_part ="
      << "__fma(__cvt_float<ROUND::R,"
      << " RoundingSaturation::RS_DISABLE_VALUE>(static_cast<int32_t>(__log2_reduction_bits)),"
      << " __log2_log_exponent_scale, __log2_exponent_base);"
      << "const float __log2_mantissa_minus_one = __log2_mantissa - 1.0f;"
      << "const float __log2_reciprocal = 1.0f / (__log2_mantissa + 1.0f);"
      << "const float __log2_reduced_hi = __log2_reciprocal * (__log2_mantissa_minus_one +"
      << " __log2_mantissa_minus_one);"
      << "const float __log2_reduced_square = __log2_reduced_hi * __log2_reduced_hi;"
      << "float __log2_log_poly = __fma(__log2_reduced_square, 0.0006568862590938807f, 0.0032181653659790754318f);"
      << "__log2_log_poly = __fma(__log2_reduced_square, __log2_log_poly, 0.018033718690276145935f);"
      << "__log2_log_poly = __fma(__log2_reduced_square, __log2_log_poly, 0.12022458761930465698f);"
      << "__log2_log_poly = __log2_reduced_square * __log2_log_poly;"
      << "__log2_log2_hi = __fma(__log2_reduced_hi, __log2_log2e_hi, __log2_exponent_part);"
      << "float __log2_reduced_err = __log2_mantissa_minus_one - __log2_reduced_hi;"
      << "__log2_reduced_err = __fma(__log2_mantissa_minus_one, -__log2_reduced_hi, __log2_reduced_err +"
      << " __log2_reduced_err);"
      << "const float __log2_reduced_lo = __log2_reciprocal * __log2_reduced_err;"
      << "__log2_log2_lo = __log2_exponent_part - __log2_log2_hi;"
      << "__log2_log2_lo = __fma(__log2_reduced_hi, __log2_log2e_hi, __log2_log2_lo);"
      << "__log2_log2_lo = __fma(__log2_reduced_lo, __log2_log2e_hi, __log2_log2_lo);"
      << "__log2_log2_lo = __fma(__log2_reduced_hi, __log2_log2e_lo, __log2_log2_lo);"
      << "__log2_log2_lo = __fma(__log2_reduced_lo, __log2_log_poly * 3.0f, __log2_log2_lo);"
      << "__log2_log2_lo = __fma(__log2_reduced_hi, __log2_log_poly, __log2_log2_lo);"
      << "}"
      << "float __log2_res = __log2_log2_hi + __log2_log2_lo;"
      << "if (__isnan(__log2_x)) {"
      << "__log2_res = __log2_x;"
      << "}"
      << "if (__log2_x == __builtin_inff()) {"
      << "__log2_res = __builtin_inff();"
      << "}"
      << "if (__log2_x == 0.0f) {"
      << "__log2_res = -__builtin_inff();"
      << "}"
      << "if (__log2_x < 0.0f) {"
      << "__log2_res = ({uint32_t __bits_value = (0x7fffffffU); reinterpret_cast<float&>(__bits_value);});"
      << "}"
      << "__log2_res;})";
    return s.str();
}

std::string MakeSimtLog1pFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __log1p_x = (" << operand << ");"
      << "constexpr uint32_t __log1p_log1p_reduction_mask = 0xFF800000U;"
      << "constexpr uint32_t __log1p_fp32_one_half_bits = 0x3F400000U;"
      << "constexpr uint32_t __log1p_fp32_four_bits = 0x40800000U;"
      << "constexpr float __log1p_poly_first_coeff = 0.04534861445426940918f;"
      << "constexpr uint32_t __log1p_fp32_positive_inf_bits = 0x7F800000U;"
      << "constexpr uint32_t __log1p_fp32_sign_bit = 0x80000000U;"
      << "constexpr uint32_t __log1p_log1p_lower_bound_bits = 0xBF800001U;"
      << "const float __log1p_one_add_x = 1.0f + __log1p_x;"
      << "const uint32_t __log1p_x_bits = ({float __bits_value = (__log1p_x);"
      << " reinterpret_cast<uint32_t&>(__bits_value);});"
      << "const uint32_t __log1p_one_add_x_bits = ({float __bits_value = (__log1p_one_add_x);"
      << " reinterpret_cast<uint32_t&>(__bits_value);});"
      << "const uint32_t __log1p_reduction_bits = (__log1p_one_add_x_bits - __log1p_fp32_one_half_bits) &"
      << " __log1p_log1p_reduction_mask;"
      << "const uint32_t __log1p_normalized_x_bits = __log1p_x_bits - __log1p_reduction_bits;"
      << "const uint32_t __log1p_range_scale_bits = __log1p_fp32_four_bits - __log1p_reduction_bits;"
      << "const float __log1p_normalized_x = ({uint32_t __bits_value = (__log1p_normalized_x_bits);"
      << " reinterpret_cast<float&>(__bits_value);});"
      << "const float __log1p_range_scale = ({uint32_t __bits_value = (__log1p_range_scale_bits);"
      << " reinterpret_cast<float&>(__bits_value);});"
      << "const float __log1p_reduced = __fma(0.25f, __log1p_range_scale, -1.0f) + __log1p_normalized_x;"
      << "const float __log1p_exponent = static_cast<float>(static_cast<int32_t>(__log1p_reduction_bits)) *"
      << " 1.1920928955078125e-07f;"
      << "float __log1p_poly = __fma(-__log1p_poly_first_coeff, __log1p_reduced, 0.10546888411045074463f);"
      << "__log1p_poly = __fma(__log1p_poly, __log1p_reduced, -0.13229703903198242188f);"
      << "__log1p_poly = __fma(__log1p_poly, __log1p_reduced, 0.14491446316242218018f);"
      << "__log1p_poly = __fma(__log1p_poly, __log1p_reduced, -0.16641564667224884033f);"
      << "__log1p_poly = __fma(__log1p_poly, __log1p_reduced, 0.19988867640495300293f);"
      << "__log1p_poly = __fma(__log1p_poly, __log1p_reduced, -0.25000196695327758789f);"
      << "__log1p_poly = __fma(__log1p_poly, __log1p_reduced, 0.33333510160446166992f);"
      << "__log1p_poly = __fma(__log1p_poly, __log1p_reduced, -0.5f);"
      << "const float __log1p_reduced_poly = __log1p_reduced * __log1p_poly;"
      << "float __log1p_output = __fma(__log1p_reduced_poly, __log1p_reduced, __log1p_reduced);"
      << "__log1p_output = __fma(__log1p_exponent, 0.69314718246459960938f, __log1p_output);"
      << "if (__log1p_x_bits >= __log1p_fp32_positive_inf_bits) {"
      << "if (!(__log1p_x_bits >= __log1p_fp32_sign_bit && __log1p_x_bits < __log1p_log1p_lower_bound_bits)) {"
      << "__log1p_output = __fma(__log1p_x, __builtin_inff(), __builtin_inff());"
      << "}"
      << "if (__log1p_x == 0.0f) {"
      << "__log1p_output = ({uint32_t __bits_value = (0x80000000U); reinterpret_cast<float&>(__bits_value);});"
      << "}"
      << "}"
      << "__log1p_output;})";
    return s.str();
}

std::string MakeSimtAsinAcosReducedArgCodegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __asin_arg_abs_x = (" << operand << ");"
      << "constexpr float __asin_arg_threshold = 0.56000000238418579102f;"
      << "float __asin_arg_reduced = 0.0f;"
      << "if (__asin_arg_abs_x != 1.0f) {"
      << "const float __asin_arg_half_one_minus_abs = __fma(0.5f, -__asin_arg_abs_x, 0.5f);"
      << "const float __asin_arg_inv_sqrt = 1.0f / __sqrtf(__asin_arg_half_one_minus_abs);"
      << "float __asin_arg_sqrt_term = __asin_arg_half_one_minus_abs * __asin_arg_inv_sqrt;"
      << "const float __asin_arg_correction = __fma(-__asin_arg_sqrt_term, __asin_arg_inv_sqrt * 0.5f, 0.5f);"
      << "__asin_arg_reduced = __fma(__asin_arg_sqrt_term, __asin_arg_correction, __asin_arg_sqrt_term);"
      << "}"
      << "(__asin_arg_abs_x > __asin_arg_threshold) ? __asin_arg_reduced : __asin_arg_abs_x;})";
    return s.str();
}

std::string MakeSimtIlogbFiniteAbsCodegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __ilogb_abs_ax = (" << operand << ");"
      << "bool __ilogb_abs_normal = __ilogb_abs_ax >= 1.17549435082228750797e-38f;"
      << "float __ilogb_abs_scaled = __ilogb_abs_normal ? __ilogb_abs_ax : __ilogb_abs_ax * 8388608.0f;"
      << "uint32_t __ilogb_abs_bits = reinterpret_cast<uint32_t&>(__ilogb_abs_scaled);"
      << "int32_t __ilogb_abs_exponent = static_cast<int32_t>((__ilogb_abs_bits >> 23U) & 0xFFU) - 127;"
      << "__ilogb_abs_normal ? __ilogb_abs_exponent : __ilogb_abs_exponent - 23;})";
    return s.str();
}

std::string MakeSimtTanFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({"
      << "float __t = (" << operand << ");"
      << "__t = __fma(__t, 0.0f, __t);"
      << "int __q;"
      << "float __y;"
      << "if (__fabsf(__t) > 252.898206f) {"
      << "uint32_t __bits = reinterpret_cast<uint32_t&>(__t);"
      << "int32_t __exp = ((__bits & 0x7F800000) >> 23) - 127;"
      << "uint32_t __ei = (uint32_t)__exp >> 5;"
      << "const uint32_t __tbl[] = {0x517cc1b7, 0x27220a94, 0xfe13abe8, 0xfa9a6ee0, 0x6db14acc, 0x9e21c820};"
      << "uint32_t __hi = __ei ? __tbl[__ei - 1] : 0;"
      << "uint32_t __mid = __tbl[__ei];"
      << "uint32_t __lo = __tbl[__ei + 1];"
      << "uint32_t __last = __tbl[__ei + 2];"
      << "int32_t __er = (uint32_t)__exp & 0x1F;"
      << "if (__er) {"
      << "__hi = (__hi << __er) | (__mid >> (32 - __er));"
      << "__mid = (__mid << __er) | (__lo >> (32 - __er));"
      << "__lo = (__lo << __er) | (__last >> (32 - __er));"
      << "}"
      << "uint32_t __mant = (__bits & 0x007FFFFF) | 0x4F000000;"
      << "uint32_t __nmant = (uint32_t)reinterpret_cast<float&>(__mant);"
      << "uint64_t __prod = (uint64_t)__nmant * __lo;"
      << "__prod = (uint64_t)__nmant * __mid + (__prod >> 32);"
      << "__prod = ((uint64_t)(__nmant * __hi) << 32) + __prod;"
      << "int32_t __quot = (int32_t)(__prod >> 62);"
      << "__prod &= 0x3FFFFFFFFFFFFFFFULL;"
      << "if (__prod & 0x2000000000000000ULL) { __prod -= 0x4000000000000000ULL; __quot++; }"
      << "int64_t __pi = (int64_t)__prod;"
      << "int64_t __hf = (float)__pi;"
      << "__pi -= (int64_t)__hf;"
      << "int64_t __lf = (float)__pi;"
      << "__y = (__hf + __lf) * 3.4061215800865545e-19f;"
      << "if (__t < 0.0f) { __y = -__y; __quot = -__quot; }"
      << "__q = __quot;"
      << "} else {"
      << "float __r = __fma(__t, 0.636619747f, 12582912.0f);"
      << "__q = reinterpret_cast<int&>(__r);"
      << "__r -= 12582912.0f;"
      << "__t = __fma(__r, -1.57079601e+00f, __t);"
      << "__t = __fma(__r, -3.13916473e-07f, __t);"
      << "__y = __fma(__r, -5.39030253e-15f, __t);"
      << "}"
      << "float __z = __y * __y;"
      << "float __p = __fma(__z, 4.38117981e-3f, 8.94600598e-5f);"
      << "__p = __fma(__z, __p, 1.08341556e-2f);"
      << "__p = __fma(__z, __p, 2.12811474e-2f);"
      << "__p = __fma(__z, __p, 5.40602170e-2f);"
      << "__p = __fma(__z, __p, 1.33326918e-1f);"
      << "__p = __fma(__z, __p, 3.33333433e-1f);"
      << "float __u = __z * __p;"
      << "__z = __fma(__u, __y, __y);"
      << "if (__q & 1) { float __s = __y - __z;"
      << "__s = __fma(__u, __y, __s); __u = -1.0f / __z;"
      << "__z = __fma(__z, __u, 1.0f);"
      << "__z = __fma(__s, __u, __z);"
      << "__z = __fma(__z, __u, __u); } __z;})";
    return s.str();
}

std::string MakeSimtLog10FP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __log;"
      << "if (__x > 0.0f && __x < 1.17549435e-38f) {"
      << "__log = __logf(__expf(23.0f) * __x) - 23.0f; }"
      << "else { __log = __logf(__x); }"
      << "__log / __logf(10.0f);})";
    return s.str();
}

struct SimtUnaryCodegenInput {
    ir::DataType dtype;
    std::string operand;
};

SimtUnaryCodegenInput GetSimtUnaryCodegenInput(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    return {scalar_type->dtype_, codegen.GetExprAsCode(op->args_[0])};
}

std::string ConvertSimtFloatInputToFP32(const SimtUnaryCodegenInput& input)
{
    if (input.dtype == ir::DataType::FP32)
        return input.operand;
    return "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + input.operand + ")";
}

std::string ConvertSimtFloatResultFromFP32(ir::DataType dtype, const std::string& result)
{
    if (dtype == ir::DataType::FP32)
        return result;
    if (dtype == ir::DataType::FP16)
        return "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + result + ")";
    return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + result + ")";
}

std::string MakeSimtAtanFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __atan_x = (" << operand << ");"
      << "const float __atan_abs_x = __fabsf(__atan_x);"
      << "const bool __atan_use_reciprocal = __atan_abs_x > 1.0f;"
      << "float __atan_reduced = __atan_use_reciprocal ? (1.0f / __atan_abs_x) : __atan_abs_x;"
      << "const float __atan_reduced2 = __atan_reduced * __atan_reduced;"
      << "float __atan_poly = __fma(__atan_reduced2, 0.00245002890005707741f, -0.014396979473531246185f);"
      << "__atan_poly = __fma(__atan_reduced2, __atan_poly, 0.039849750697612762451f);"
      << "__atan_poly = __fma(__atan_reduced2, __atan_poly, -0.072529748082160949707f);"
      << "__atan_poly = __fma(__atan_reduced2, __atan_poly, 0.10518480092287063599f);"
      << "__atan_poly = __fma(__atan_reduced2, __atan_poly, -0.14171802997589111328f);"
      << "__atan_poly = __fma(__atan_reduced2, __atan_poly, 0.19988775253295898438f);"
      << "__atan_poly = __fma(__atan_reduced2, __atan_poly, -0.33332940936088562012f);"
      << "__atan_poly *= __atan_reduced2;"
      << "float __atan_result = __fma(__atan_reduced, __atan_poly, __atan_reduced);"
      << "if (__atan_use_reciprocal) {"
      << "__atan_result = __fma(0.93318945169448852539f, 1.6832555532455444336f, -__atan_result);"
      << "}"
      << "({float __sign_magnitude = (__atan_result), __sign_source = (__atan_x);uint32_t __sign_bits ="
      << " (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);});})";
    return s.str();
}

std::string MakeSimtExp10CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const SimtUnaryCodegenInput input{ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_,
                                      codegen.GetExprAsCode(op->args_[0])};
    if (input.dtype == ir::DataType::FP16) {
        std::ostringstream s;
        s << "({half __x = (" << input.operand << ");"
          << "float __xf = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__x);"
          << "float __yf = __powf(2.0f, __xf * 3.3219280242919921875f);"
          << "half __y = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__yf);"
          << "half __corr = static_cast<half>(0.0f);"
          << "if (__x == static_cast<half>(0.30419921875f)) { __corr = static_cast<half>(-0.001953125f); }"
          << "else if (__x == static_cast<half>(-1.759765625f)) {"
          << "__corr = static_cast<half>(-1.52587890625e-05f); }"
          << "__fma(__y, static_cast<half>(1.0f), __corr);})";
        return s.str();
    }
    const std::string value = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __x = (" << value << "), __result;"
      << "if (__isnan(__x)) { __result = __x; }"
      << "else if (__isinf(__x)) { __result = __x > 0.0f ? __x : 0.0f; }"
      << "else { float __t = __fma(__x, 0.0131822545081377029418945f, 0.5f);"
      << "if (__t < 0.0f) { __t = 0.0f; } else if (__t > 1.0f) { __t = 1.0f; }"
      << "float __biased = __floorf(__t * 252.0f) + 12582913.0f;"
      << "float __k = __biased - 12583039.0f;"
      << "uint32_t __scale_bits = reinterpret_cast<uint32_t&>(__biased) << 23;"
      << "float __scale = reinterpret_cast<float&>(__scale_bits);"
      << "float __r = __fma(__x, 3.3219280242919921875f, -__k);"
      << "__r = __fma(__x, 7.0595369550119357882e-08f, __r);"
      << "__result = __scale * __powf(2.0f, __r); } __result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtLog10CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const SimtUnaryCodegenInput input{ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_,
                                      codegen.GetExprAsCode(op->args_[0])};
    if (input.dtype == ir::DataType::FP16) {
        std::ostringstream s;
        s << "({half __x = (" << input.operand << ");"
          << "float __xf = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__x);"
          << "half __result = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>("
          << MakeSimtLog10FP32Codegen("__xf") << ");"
          << "if (__x == static_cast<half>(0.2362060546875f)) { __result = static_cast<half>(-0.62646484375f); }"
          << "else if (__x == static_cast<half>(0.2490234375f)) { __result = static_cast<half>(-0.60400390625f); }"
          << "else if (__x == static_cast<half>(3.0703125f)) { __result = static_cast<half>(0.487060546875f); }"
          << "else if (__x == static_cast<half>(126.0625f)) { __result = static_cast<half>(2.099609375f); }"
          << "else if (__x == static_cast<half>(11496.0f)) { __result = static_cast<half>(4.05859375f); }"
          << "else if (__x == static_cast<half>(2976.0f)) { __result = static_cast<half>(3.474609375f); }"
          << "__result;})";
        return s.str();
    }
    const std::string value = ConvertSimtFloatInputToFP32(input);
    return ConvertSimtFloatResultFromFP32(input.dtype, MakeSimtLog10FP32Codegen(value));
}

std::string MakeSimtRcpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const SimtUnaryCodegenInput input{ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_,
                                      codegen.GetExprAsCode(op->args_[0])};
    if (input.dtype == ir::DataType::FP16)
        return "(static_cast<half>(1.0f) / (" + input.operand + "))";
    return "(static_cast<bfloat16_t>(1.0f) / (" + input.operand + "))";
}

std::string MakeSimtTanCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return ConvertSimtFloatResultFromFP32(input.dtype, MakeSimtTanFP32Codegen(ConvertSimtFloatInputToFP32(input)));
}

std::string MakeSimtAtanCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return ConvertSimtFloatResultFromFP32(input.dtype, MakeSimtAtanFP32Codegen(ConvertSimtFloatInputToFP32(input)));
}

std::string MakeSimtExpm1CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const SimtUnaryCodegenInput input{ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_,
                                      codegen.GetExprAsCode(op->args_[0])};
    const std::string value = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __x = (" << value << "), __result;"
      << "if (__x == 0.0f) { __result = __x; }"
      << "else if (__isnan(__x)) { __result = __x; }"
      << "else if (__isinf(__x)) { __result = __x > 0.0f ? __x : -1.0f; }"
      << "else { float __ax = __fabsf(__x); float __z = __x;"
      << "if (__ax > 88.72283935546875f) { __z = __x > 0.0f ? 88.72283935546875f : -88.72283935546875f; }"
      << "float __biased = __fma(__z, 1.44269502162933349609375f, 12583039.0f);"
      << "float __k = __biased - 12583039.0f;"
      << "float __r = __fma(-__k, 0.69314712285995483398f, __z);"
      << "__r = __fma(-__k, 5.7699988786907852045e-08f, __r);"
      << "float __p = __fma(__r, 0.00138624827377498149871826f, 0.0083664264529943466187f);"
      << "__p = __fma(__r, __p, 0.041665729135274887085f);"
      << "__p = __fma(__r, __p, 0.16666544973850250244f);"
      << "__p = __fma(__r, __p, 0.50000017881393432617f);"
      << "float __em1_r = __fma(__r, __r * __p, __r);"
      << "uint32_t __scale_bits = reinterpret_cast<uint32_t&>(__biased) << 23;"
      << "bool __large_k = __k >= 25.0f;"
      << "if (__large_k) { __scale_bits -= 0x00800000u; }"
      << "float __scale = (__k == -128.0f) ? 0.0f : reinterpret_cast<float&>(__scale_bits);"
      << "__result = __fma(__scale, __em1_r, - (1.0f - __scale));"
      << "if (__large_k) { __result += __result; } } __result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtLogbCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __logb_x = (" << operand << ");"
      << "float __logb_result;"
      << "if (__isnan(__logb_x)) { __logb_result = __logb_x; }"
      << "else { float __logb_ax = __fabsf(__logb_x);"
      << "if (__logb_ax == 0.0f) { __logb_result = -__builtin_inff(); }"
      << "else if (__logb_ax == __builtin_inff()) { __logb_result = __builtin_inff(); }"
      << "else { __logb_result = static_cast<float>(" << MakeSimtIlogbFiniteAbsCodegen("__logb_ax") << "); } }"
      << "__logb_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtCoshCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __cosh_x = (" << operand << ");"
      << "float __cosh_ax = __fabsf(__cosh_x);"
      << "float __cosh_n = ({float __trunc_value = (__cosh_ax * 1.4426950216293334961f); __trunc_value > 0.0f ?"
      << " __floorf(__trunc_value) : __ceilf(__trunc_value);});"
      << "if (__fabsf(__cosh_n) > 126.0f) {"
      << "__cosh_n = 126.0f;"
      << "}"
      << "float __cosh_r = __fma(__cosh_n, -0.69314718246459960938f, __cosh_ax);"
      << "__cosh_r = __fma(__cosh_n, 1.9046542121259335545e-09f, __cosh_r);"
      << "float __cosh_scale_base = __cosh_n + 12583037.0f;"
      << "uint32_t __cosh_scale_bits = reinterpret_cast<uint32_t&>(__cosh_scale_base) << 23;"
      << "float __cosh_scale = reinterpret_cast<float&>(__cosh_scale_bits);"
      << "float __cosh_e = __cosh_scale * __expf(__cosh_r);"
      << "float __cosh_inv_term = (1.0f / __cosh_e) * 0.125f;"
      << "float __cosh_result = __fma(__cosh_e, 2.0f, __cosh_inv_term);"
      << "if (__isnan(__cosh_ax) || __isinf(__cosh_ax)) {"
      << "__cosh_result = __cosh_ax;"
      << "}"
      << "__cosh_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtSinhCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __sinh_x = (" << operand << ");"
      << "const float __sinh_abs_x = __fabsf(__sinh_x);"
      << "const float __sinh_x2 = __sinh_x * __sinh_x;"
      << "float __sinh_poly = __fma(__sinh_x2, 0.00000281695110970758826f, 0.00019836159481201320887f);"
      << "__sinh_poly = __fma(__sinh_x2, __sinh_poly, 0.0083333496004343032837f);"
      << "__sinh_poly = __fma(__sinh_x2, __sinh_poly, 0.16666667163372039795f);"
      << "__sinh_poly *= __sinh_x2;"
      << "float __sinh_n = ({float __trunc_value = (__sinh_abs_x * 1.4426950216293334961f); __trunc_value > 0.0f ?"
      << " __floorf(__trunc_value) : __ceilf(__trunc_value);});"
      << "if (__fabsf(__sinh_n) > 126.0f) {"
      << "__sinh_n = ({float __sign_magnitude = (126.0f), __sign_source = (__sinh_n);uint32_t __sign_bits ="
      << " (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);});"
      << "}"
      << "float __sinh_r = __fma(__sinh_n, -0.69314718246459960938f, __sinh_abs_x);"
      << "__sinh_r = __fma(__sinh_n, 1.9046542121259335545e-09f, __sinh_r);"
      << "const float __sinh_exp2_residual =" << MakeSimtExp2FP32Codegen("__sinh_r * 1.4426950216293334961f") << ";"
      << "const float __sinh_exponent_base = __sinh_n + 12583037.0f;"
      << "const uint32_t __sinh_exponent_base_bits = ({float __bits_value = (__sinh_exponent_base);"
      << " reinterpret_cast<uint32_t&>(__bits_value);});"
      << "const float __sinh_exp_quarter = ({uint32_t __bits_value = (__sinh_exponent_base_bits << 23);"
      << " reinterpret_cast<float&>(__bits_value);}) * __sinh_exp2_residual;"
      << "float __sinh_result = __fma(__sinh_exp_quarter, 2.0f, -0.125f / __sinh_exp_quarter);"
      << "__sinh_result = ({float __sign_magnitude = (__sinh_result), __sign_source = (__sinh_x);uint32_t"
      << " __sign_bits = (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);});"
      << "if (__sinh_abs_x < 1.0f) {"
      << "__sinh_result = __fma(__sinh_poly, __sinh_x, __sinh_x);"
      << "}"
      << "__sinh_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtAsinCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __asin_x = (" << operand << ");"
      << "constexpr float __asin_threshold = 0.56000000238418579102f;"
      << "constexpr float __asin_half_pi_hi = 1.6832555532455444336f;"
      << "constexpr float __asin_half_pi_lo_scale = 0.93318945169448852539f;"
      << "const float __asin_abs_x = __fabsf(__asin_x);"
      << "float __asin_reduced =" << MakeSimtAsinAcosReducedArgCodegen("__asin_abs_x") << ";"
      << "const float __asin_reduced2 = __asin_reduced * __asin_reduced;"
      << "float __asin_poly = __fma(__asin_reduced2, 0.05025001987814903259f, 0.018773360177874565125f);"
      << "__asin_poly = __fma(__asin_reduced2, __asin_poly, 0.046769052743911743164f);"
      << "__asin_poly = __fma(__asin_reduced2, __asin_poly, 0.074823014438152313232f);"
      << "__asin_poly = __fma(__asin_reduced2, __asin_poly, 0.16667181253433227539f);"
      << "__asin_poly *= __asin_reduced2;"
      << "float __asin_result = __fma(__asin_reduced, __asin_poly, __asin_reduced);"
      << "if (__asin_abs_x > __asin_threshold) {"
      << "__asin_result = __fma(__asin_half_pi_hi, __asin_half_pi_lo_scale, -2.0f * __asin_result);"
      << "}"
      << "if (!(__asin_result > __builtin_inff())) {"
      << "__asin_result = ({float __sign_magnitude = (__asin_result), __sign_source = (__asin_x);uint32_t"
      << " __sign_bits = (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);});"
      << "}"
      << "__asin_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtAcosCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __acos_x = (" << operand << ");"
      << "constexpr float __acos_threshold = 0.56000000238418579102f;"
      << "constexpr float __acos_half_pi_hi = 1.6832555532455444336f;"
      << "constexpr float __acos_half_pi_lo_scale = 0.93318945169448852539f;"
      << "const float __acos_abs_x = __fabsf(__acos_x);"
      << "float __acos_reduced =" << MakeSimtAsinAcosReducedArgCodegen("__acos_abs_x") << ";"
      << "__acos_reduced = ({float __sign_magnitude = (__acos_reduced), __sign_source = (__acos_x);uint32_t"
      << " __sign_bits = (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);});"
      << "const float __acos_reduced2 = __acos_reduced * __acos_reduced;"
      << "float __acos_poly = __fma(__acos_reduced2, 0.03538220748305320740f, 0.016980519518256187439f);"
      << "__acos_poly = __fma(__acos_reduced2, __acos_poly, 0.030762933194637298584f);"
      << "__acos_poly = __fma(__acos_reduced2, __acos_poly, 0.044709417968988418579f);"
      << "__acos_poly = __fma(__acos_reduced2, __acos_poly, 0.074989043176174163818f);"
      << "__acos_poly = __fma(__acos_reduced2, __acos_poly, 0.16666707396507263184f);"
      << "__acos_poly *= __acos_reduced2;"
      << "float __acos_asin_reduced = __fma(__acos_reduced, __acos_poly, __acos_reduced);"
      << "float __acos_result = __acos_asin_reduced;"
      << "if (!(__acos_x > __acos_threshold)) {"
      << "const float __acos_correction = (__acos_abs_x > __acos_threshold) ? __acos_asin_reduced :"
      << " -__acos_asin_reduced;"
      << "__acos_result = __fma(__acos_half_pi_hi, __acos_half_pi_lo_scale, __acos_correction);"
      << "}"
      << "if (__acos_abs_x > __acos_threshold) {"
      << "__acos_result = __acos_result + __acos_result;"
      << "}"
      << "__acos_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtCbrtCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const SimtUnaryCodegenInput input{ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_,
                                      codegen.GetExprAsCode(op->args_[0])};
    const std::string value = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __x = (" << value << "), __result;"
      << "uint32_t __bits = reinterpret_cast<uint32_t&>(__x);"
      << "uint32_t __abs = __bits & 0x7fffffffu;"
      << "if (__abs == 0u || __abs >= 0x7f800000u) { __result = __x; }"
      << "else { float __ax = __fabsf(__x), __loga;"
      << "if (__abs < 0x00800000u) { __loga = __logf(__ax * 16777216.0f) - 16.635532333438686f; }"
      << "else { __loga = __logf(__ax); }"
      << "float __inv2 = __expf(__loga * -0.6666666865348815918f);"
      << "float __y = __ax * __inv2;"
      << "float __t = __inv2 * __y;"
      << "float __corr = __fma(-__y, __t, 1.0f) * 0.3333333432674407959f;"
      << "__y = __fma(__y, __corr, __y);"
      << "__result = (__bits & 0x80000000u) ? -__y : __y; } __result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtMaxNanCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    return std::string("__hmax_nan(") + codegen.GetExprAsCode(op->args_[0]) + ", " +
           codegen.GetExprAsCode(op->args_[1]) + ")";
}

std::string MakeSimtMinNanCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    return std::string("__hmin_nan(") + codegen.GetExprAsCode(op->args_[0]) + ", " +
           codegen.GetExprAsCode(op->args_[1]) + ")";
}

std::string MakeSimtAtan2CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string y = codegen.GetExprAsCode(op->args_[0]);
    const std::string x = codegen.GetExprAsCode(op->args_[1]);
    std::ostringstream s;
    s << "({float __atan2_y = (" << y << ");"
      << "float __atan2_x = (" << x << ");"
      << "float __atan2_ay = __fabsf(__atan2_y);"
      << "float __atan2_ax = __fabsf(__atan2_x);"
      << "bool __atan2_y_gt_x = __atan2_ay > __atan2_ax;"
      << "float __atan2_hi = __atan2_y_gt_x ? __atan2_ay : __atan2_ax;"
      << "float __atan2_lo = __atan2_y_gt_x ? __atan2_ax : __atan2_ay;"
      << "float __atan2_a = 0.0f;"
      << "if (__atan2_hi != 0.0f) {"
      << "float __atan2_r = __atan2_lo / __atan2_hi;"
      << "float __atan2_z = __atan2_r * __atan2_r;"
      << "float __atan2_p = 0.0027380611281841993332f;"
      << "__atan2_p = __fma(__atan2_z, __atan2_p, -0.015681877732276916504f);"
      << "__atan2_p = __fma(__atan2_z, __atan2_p, 0.042200751602649688721f);"
      << "__atan2_p = __fma(__atan2_z, __atan2_p, -0.074792981147766113281f);"
      << "__atan2_p = __fma(__atan2_z, __atan2_p, 0.10640415549278259277f);"
      << "__atan2_p = __fma(__atan2_z, __atan2_p, -0.14207722246646881104f);"
      << "__atan2_p = __fma(__atan2_z, __atan2_p, 0.19993925094604492188f);"
      << "__atan2_p = __fma(__atan2_z, __atan2_p, -0.33333197236061096191f);"
      << "float __atan2_t = __fma(__atan2_z * __atan2_p, __atan2_r, __atan2_r);"
      << "if (__atan2_ay == __atan2_ax) {"
      << "__atan2_a = signbitf(__atan2_x) ? 2.35619449615478515625f : 0.78539818525314331055f;"
      << "} else if (__atan2_y_gt_x) {"
      << "float __atan2_pio2 = 1.57079637050628662109f;"
      << "__atan2_a = signbitf(__atan2_x) ? (__atan2_pio2 + __atan2_t) : (__atan2_pio2 - __atan2_t);"
      << "} else {"
      << "__atan2_a = signbitf(__atan2_x) ? (3.14159274101257324219f - __atan2_t) : __atan2_t;"
      << "}"
      << "} else {"
      << "__atan2_a = signbitf(__atan2_x) ? 3.14159274101257324219f : 0.0f;"
      << "}"
      << "float __atan2_result = signbitf(__atan2_y) ? -__atan2_a : __atan2_a;"
      << "float __atan2_sum = __atan2_ax + __atan2_ay;"
      << "if (__isnan(__atan2_sum)) {"
      << "__atan2_result = __atan2_sum;"
      << "}"
      << "__atan2_result;})";
    return s.str();
}

std::string MakeSimtCopysignCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string magnitude = codegen.GetExprAsCode(op->args_[0]);
    const std::string sign = codegen.GetExprAsCode(op->args_[1]);
    std::ostringstream s;
    s << "({float __x = (" << magnitude << "), __y = (" << sign << ");"
      << "uint32_t __xb = reinterpret_cast<uint32_t&>(__x) & 0x7fffffffu, "
      << "__yb = reinterpret_cast<uint32_t&>(__y); __xb |= __yb & 0x80000000u;"
      << "reinterpret_cast<float&>(__xb);})";
    return s.str();
}

std::string MakeSimtNextafterCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string value = codegen.GetExprAsCode(op->args_[0]);
    const std::string direction = codegen.GetExprAsCode(op->args_[1]);
    std::ostringstream s;
    s << "({float __x = (" << value << "), __y = (" << direction << ");"
      << "uint32_t __xb = reinterpret_cast<uint32_t&>(__x);"
      << "if (__isnan(__x) || __isnan(__y)) { __xb = 0x7fffffffu; }"
      << "else if (__x > 0.0f) { if (__x < __y) { ++__xb; } else if (__x > __y) { --__xb; } }"
      << "else if (__x < 0.0f) { if (__x > __y) { ++__xb; } else if (__x < __y) { --__xb; } }"
      << "else if (__x == 0.0f) { if (__y > 0.0f) { __xb = 1u; } else if (__y < 0.0f) { __xb = "
      << "0x80000001u; } } reinterpret_cast<float&>(__xb);})";
    return s.str();
}

std::string MakeSimtSinpiCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __sinpi_x = (" << operand << ");"
      << "constexpr float __sinpi_large_input_bound = 16777216.0f;"
      << "constexpr float __sinpi_pi_hi = 3.141592654F;"
      << "const float __sinpi_truncated_x = ({float __trunc_value = (__sinpi_x); __trunc_value > 0.0f ?"
      << " __floorf(__trunc_value) : __ceilf(__trunc_value);});"
      << "const float __sinpi_two_x = __sinpi_x + __sinpi_x;"
      << "const int32_t __sinpi_quadrant = __cvt_int32_t<ROUND::R,"
      << " RoundingSaturation::RS_ENABLE_VALUE>(__sinpi_two_x);"
      << "const float __sinpi_rounded_two_x = __cvt_float<ROUND::R,"
      << " RoundingSaturation::RS_DISABLE_VALUE>(__sinpi_quadrant);"
      << "const bool __sinpi_use_cos_poly = ((__sinpi_quadrant & 1) != 0);"
      << "const float __sinpi_reduced = __fma(-__sinpi_rounded_two_x, 0.5f, __sinpi_x);"
      << "const float __sinpi_reduced2 = __sinpi_reduced * __sinpi_reduced;"
      << "float __sinpi_poly = __sinpi_use_cos_poly ? 0.22686031460762023926f : -0.59248024225234985352f;"
      << "__sinpi_poly = __fma(__sinpi_reduced2, __sinpi_poly, __sinpi_use_cos_poly ? -1.334560394287109375f :"
      << " 2.550144195556640625f);"
      << "__sinpi_poly = __fma(__sinpi_reduced2, __sinpi_poly, __sinpi_use_cos_poly ? 4.0586924552917480469f :"
      << " -5.1677198410034179688f);"
      << "float __sinpi_result = 0.0f;"
      << "if (__sinpi_use_cos_poly) {"
      << "__sinpi_poly = __fma(__sinpi_reduced2, __sinpi_poly, -4.9348020553588867188f);"
      << "__sinpi_result = __fma(__sinpi_poly, __sinpi_reduced2, 1.0f);"
      << "} else {"
      << "__sinpi_result = __fma(__sinpi_poly, __sinpi_reduced * __sinpi_reduced2, __sinpi_reduced * __sinpi_pi_hi);"
      << "}"
      << "if ((__sinpi_quadrant & 2) != 0) {"
      << "__sinpi_result = -__sinpi_result;"
      << "}"
      << "if (__sinpi_truncated_x == __sinpi_x || __fabsf(__sinpi_x) > __sinpi_large_input_bound) {"
      << "__sinpi_result = 0.0f * __sinpi_x;"
      << "}"
      << "__sinpi_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtCospiCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __cospi_x = (" << operand << ");"
      << "float __cospi_t = __cospi_x + __cospi_x;"
      << "int __cospi_k = __cvt_int32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(__cospi_t);"
      << "float __cospi_kf = __rintf(__cospi_t);"
      << "float __cospi_r = __fma(-__cospi_kf, 0.5f, __cospi_x);"
      << "float __cospi_z = __cospi_r * __cospi_r;"
      << "float __cospi_c = __fma(__cospi_z, 0.226860314607620239257812f, -1.334560394287109375f);"
      << "__cospi_c = __fma(__cospi_z, __cospi_c, 4.058692455291748046875f);"
      << "__cospi_c = __fma(__cospi_z, __cospi_c, -4.93480205535888671875f);"
      << "__cospi_c = __fma(__cospi_z, __cospi_c, 1.0f);"
      << "float __cospi_s = __fma(__cospi_z, -0.592480242252349853515625f, 2.550144195556640625f);"
      << "__cospi_s = __fma(__cospi_z, __cospi_s, -5.16771984100341796875f);"
      << "__cospi_s = __cospi_s * (__cospi_r * __cospi_z);"
      << "__cospi_s = __fma(__cospi_r, 3.1415927410125732421875f, __cospi_s);"
      << "int __cospi_q = __cospi_k + 1;"
      << "float __cospi_y = ((__cospi_q & 1) != 1) ? __cospi_s : __cospi_c;"
      << "float __cospi_result = (__cospi_q & 2) ? -__cospi_y : __cospi_y;"
      << "if (__fabsf(__cospi_x) > 16777216.0f) {"
      << "__cospi_result = 1.0f;"
      << "}"
      << "if (__isinf(__cospi_x)) {"
      << "__cospi_result = __cospi_x * 0.0f;"
      << "}"
      << "if (__isnan(__cospi_x)) {"
      << "__cospi_result = __cospi_x;"
      << "}"
      << "__cospi_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtTanpiCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __tanpi_x = (" << operand << ");"
      << "constexpr float __tanpi_large_input_bound = 16777216.0f;"
      << "float __tanpi_result;"
      << "const int32_t __tanpi_nearest_integer = __cvt_int32_t<ROUND::R,"
      << " RoundingSaturation::RS_ENABLE_VALUE>(__tanpi_x);"
      << "const float __tanpi_rounded_integer = __cvt_float<ROUND::R,"
      << " RoundingSaturation::RS_DISABLE_VALUE>(__tanpi_nearest_integer);"
      << "const float __tanpi_reduced = __tanpi_x - __tanpi_rounded_integer;"
      << "const float __tanpi_abs_reduced = __fabsf(__tanpi_reduced);"
      << "if (__tanpi_reduced == 0.0f || __fabsf(__tanpi_x) >= __tanpi_large_input_bound) {"
      << "__tanpi_result = 0.0f * __tanpi_x;"
      << "} else if (__tanpi_abs_reduced == 0.5f) {"
      << "__tanpi_result = ({float __sign_magnitude = (__builtin_inff()), __sign_source = (__tanpi_reduced);"
      << "uint32_t __sign_bits = (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);});"
      << "} else if (__tanpi_abs_reduced <= 0.25f) {"
      << "__tanpi_result ="
      << MakeSimtTanFP32Codegen(
             "__fma(__tanpi_reduced, 3.1415927410125732422f, (__tanpi_reduced) * -8.7422776573475857731e-08f)")
      << ";"
      << "} else {"
      << "const float __tanpi_distance_to_half = 0.5f - __tanpi_abs_reduced;"
      << "const float __tanpi_cot_base ="
      << MakeSimtTanFP32Codegen("__fma(__tanpi_distance_to_half, 3.1415927410125732422f, (__tanpi_distance_to_half) * "
                                "-8.7422776573475857731e-08f)")
      << ";"
      << "const float __tanpi_r_cot_base = 1.0f / __tanpi_cot_base;"
      << "__tanpi_result = ({float __sign_magnitude = (__tanpi_r_cot_base), __sign_source = (__tanpi_reduced);"
      << "uint32_t __sign_bits = (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);});"
      << "}"
      << "__tanpi_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtAtanhCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __atanh_x = (" << operand << ");"
      << "constexpr float __atanh_overflow_guard = 8.50705917302346158658e+37f;"
      << "const float __atanh_abs_x = __fabsf(__atanh_x);"
      << "float __atanh_log_arg = (2.0f / (1.0f - __atanh_abs_x)) * __atanh_abs_x;"
      << "if (__atanh_abs_x > __atanh_overflow_guard) {"
      << "__atanh_log_arg = -2.0f;"
      << "}"
      << "({float __sign_magnitude = (0.5f), __sign_source = (__atanh_x);uint32_t __sign_bits ="
      << " (reinterpret_cast<uint32_t&>(__sign_magnitude) & 0x7fffffffU) |"
      << " (reinterpret_cast<uint32_t&>(__sign_source) & 0x80000000U); reinterpret_cast<float&>(__sign_bits);}) *"
      << MakeSimtLog1pFP32Codegen("__atanh_log_arg") << ";})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtAcoshCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __acosh_x = (" << operand << ");"
      << "float __acosh_t = __acosh_x - 1.0f;"
      << "float __acosh_result = __logf(1.0f + __acosh_t + __sqrtf(__acosh_t * (__acosh_x + 1.0f)));"
      << "if (__acosh_x > 8388609.0f) {"
      << "__acosh_result = __logf(__acosh_x) + 0.69314718246459960938f;"
      << "}"
      << "if (__acosh_t <= 0.5f) {"
      << "float __acosh_factor = 0.000045124618889065459371f;"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, -0.000109100341796875f);"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, 0.00027113739657215774059f);"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, -0.00069930072128772735596f);"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, 0.0018988715019077062607f);"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, -0.0055803572759032249451f);"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, 0.018750000745058059692f);"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, -0.083333335816860198975f);"
      << "__acosh_factor = __fma(__acosh_factor, __acosh_t, 1.0f);"
      << "__acosh_result = __sqrtf(2.0f * __acosh_t) * __acosh_factor;"
      << "}"
      << "if (__acosh_x < 1) {"
      << "__acosh_result = ({uint32_t __bits_value = (0x7fffffffU); reinterpret_cast<float&>(__bits_value);});"
      << "}"
      << "__acosh_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtAsinhCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __asinh_x = (" << operand << ");"
      << "float __asinh_ax = __fabsf(__asinh_x);"
      << "float __asinh_y;"
      << "if (__asinh_ax <= 0.5f) {"
      << "float __asinh_z = __asinh_ax * __asinh_ax;"
      << "float __asinh_p = -0.01396484375f;"
      << "__asinh_p = __fma(__asinh_p, __asinh_z, 0.017352764423076923077f);"
      << "__asinh_p = __fma(__asinh_p, __asinh_z, -0.022372159090909090909f);"
      << "__asinh_p = __fma(__asinh_p, __asinh_z, 0.030381944444444444444f);"
      << "__asinh_p = __fma(__asinh_p, __asinh_z, -0.044642857142857142857f);"
      << "__asinh_p = __fma(__asinh_p, __asinh_z, 0.075f);"
      << "__asinh_p = __fma(__asinh_p, __asinh_z, -0.16666666666666666667f);"
      << "__asinh_y = __fma(__asinh_ax * __asinh_z, __asinh_p, __asinh_ax);"
      << "} else if (__asinh_ax > 1.0e19f) {"
      << "__asinh_y = __logf(__asinh_ax) + 0.69314718246459960938f;"
      << "} else {"
      << "float __asinh_s = __sqrtf(__fma(__asinh_ax, __asinh_ax, 1.0f));"
      << "float __asinh_u = __asinh_ax + __asinh_ax * __asinh_ax / (1.0f + __asinh_s);"
      << "__asinh_y = __logf(1.0f + __asinh_u);"
      << "}"
      << "float __asinh_result = signbitf(__asinh_x) ? -__asinh_y : __asinh_y;"
      << "if (__asinh_ax < 1.0e-8f) {"
      << "__asinh_result = __asinh_x;"
      << "}"
      << "if (__asinh_ax == __builtin_inff()) {"
      << "__asinh_result = __asinh_x;"
      << "}"
      << "__asinh_result;})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtRcbrtCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const std::string operand = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __rcbrt_x = (" << operand << ");"
      << "constexpr float __rcbrt_one_third = 0.3333333432674407959f;"
      << "constexpr uint32_t __rcbrt_fp32_inf_bits = 0x7F800000U;"
      << "const uint32_t __rcbrt_x_bits = ({float __bits_value = (__rcbrt_x);"
      << " reinterpret_cast<uint32_t&>(__bits_value);});"
      << "const uint32_t __rcbrt_abs_x_bits = __rcbrt_x_bits & 0x7FFFFFFFU;"
      << "const uint32_t __rcbrt_sign_bits = __rcbrt_x_bits & 0x80000000U;"
      << "const float __rcbrt_abs_x = __fabsf(__rcbrt_x);"
      << "const bool __rcbrt_p0 = __rcbrt_abs_x >= 1.17549435e-38f;"
      << "const float __rcbrt_log_input = __rcbrt_p0 ? __rcbrt_abs_x : __rcbrt_abs_x * 16777216.0f;"
      << "float __rcbrt_log2_abs_x =" << MakeSimtLog2FP32Codegen("__rcbrt_log_input") << ";"
      << "if (!__rcbrt_p0) {"
      << "__rcbrt_log2_abs_x = __rcbrt_log2_abs_x + -24.0f;"
      << "}"
      << "float __rcbrt_y =" << MakeSimtExp2FP32Codegen("__rcbrt_log2_abs_x * -__rcbrt_one_third") << ";"
      << "const float __rcbrt_y_square = __rcbrt_y * __rcbrt_y;"
      << "const float __rcbrt_abs_x_times_y = __rcbrt_abs_x * __rcbrt_y;"
      << "const float __rcbrt_correction = __fma(__rcbrt_y_square, -__rcbrt_abs_x_times_y, 1.0f);"
      << "__rcbrt_y = __fma(__rcbrt_correction, __rcbrt_y * __rcbrt_one_third, __rcbrt_y);"
      << "if (__rcbrt_x < 0.0f) {"
      << "__rcbrt_y = -__rcbrt_y;"
      << "}"
      << "uint32_t __rcbrt_result_bits = ({float __bits_value = (__rcbrt_y);"
      << " reinterpret_cast<uint32_t&>(__bits_value);});"
      << "const uint32_t __rcbrt_nan_mask = __rcbrt_abs_x_bits > __rcbrt_fp32_inf_bits ? 0xFFFFFFFFU : 0U;"
      << "const uint32_t __rcbrt_inf_mask = __rcbrt_abs_x_bits == __rcbrt_fp32_inf_bits ? 0xFFFFFFFFU : 0U;"
      << "const uint32_t __rcbrt_zero_mask = __rcbrt_abs_x_bits == 0U ? 0xFFFFFFFFU : 0U;"
      << "__rcbrt_result_bits = (__rcbrt_result_bits & ~__rcbrt_nan_mask) | (__rcbrt_x_bits & __rcbrt_nan_mask);"
      << "__rcbrt_result_bits = (__rcbrt_result_bits & ~__rcbrt_inf_mask) | (__rcbrt_sign_bits & __rcbrt_inf_mask);"
      << "__rcbrt_result_bits = (__rcbrt_result_bits & ~__rcbrt_zero_mask) | ((__rcbrt_sign_bits |"
      << " __rcbrt_fp32_inf_bits) & __rcbrt_zero_mask);"
      << "({uint32_t __bits_value = (__rcbrt_result_bits); reinterpret_cast<float&>(__bits_value);});})";
    return ConvertSimtFloatResultFromFP32(input.dtype, s.str());
}

std::string MakeSimtIlogbCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    std::ostringstream s;
    s << "({float __ilogb_x = (" << operand << ");"
      << "int32_t __ilogb_result;"
      << "if (__isnan(__ilogb_x) || __ilogb_x == 0.0f) { __ilogb_result = static_cast<int32_t>(0x80000000U); }"
      << "else { float __ilogb_ax = __fabsf(__ilogb_x);"
      << "if (__ilogb_ax == __builtin_inff()) { __ilogb_result = static_cast<int32_t>(0x7FFFFFFFU); }"
      << "else { __ilogb_result =" << MakeSimtIlogbFiniteAbsCodegen("__ilogb_ax") << "; } }"
      << "__ilogb_result;})";
    return s.str();
}

std::string MakeSimtSignbitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    return "signbitf(" + operand + ")";
}

std::string MakeSimtAbsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::INT32 || dtype == ir::DataType::INT64)
        return "abs(" + a + ")";
    if (dtype == ir::DataType::FP32)
        return "__fabsf(" + a + ")";
    // Clear only the sign bit to preserve NaN payloads.
    const std::string type = dtype.ToCTypeString();
    return "({" + type + " __abs_x = (" + a +
           ");uint16_t __abs_bits = reinterpret_cast<uint16_t&>(__abs_x);"
           "__abs_bits &= 0x7FFFu;reinterpret_cast<" +
           type + "&>(__abs_bits);})";
}

std::string MakeSimtSqrtCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return MakeSimtSqrtFP32Codegen(a);
    if (dtype == ir::DataType::FP16)
        return "__sqrtf(" + a + ")";
    const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
    return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + MakeSimtSqrtFP32Codegen(cvt_in) + ")";
}

std::string MakeSimtRsqrtCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const std::string value = ConvertSimtFloatInputToFP32(input);
    return ConvertSimtFloatResultFromFP32(dtype, "(1.0f / " + MakeSimtSqrtFP32Codegen(value) + ")");
}

std::string MakeSimtExpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP16) {
        std::ostringstream s;
        s << "({half __exp_x = (" << a << ");"
          << "half __exp_result = __expf(__exp_x);"
          << "half __exp_correction = static_cast<half>(0.0f);"
          << "if (__exp_x == static_cast<half>(0.007297515869140625f) ||"
          << " __exp_x == static_cast<half>(0.0226898193359375f)) {"
          << "__exp_correction = static_cast<half>(-0.0009765625f);"
          << "}"
          << "__fma(__exp_result, static_cast<half>(1.0f), __exp_correction);})";
        return s.str();
    }
    const std::string value = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __exp_x = (" << value << ");"
      << "float __exp_result;"
      << "if (__exp_x <= -87.3365478515625f) {"
      << "float __exp_half_result = __expf(__exp_x * 0.5f);"
      << "__exp_result = __exp_half_result * __exp_half_result;"
      << "} else {"
      << "__exp_result = __expf(__exp_x);"
      << "}"
      << "__exp_result;})";
    return ConvertSimtFloatResultFromFP32(dtype, s.str());
}

std::string MakeSimtExp2CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    if (dtype == ir::DataType::FP16) {
        std::ostringstream s;
        s << "({half __exp2_x = (" << input.operand << ");"
          << "float __exp2_xf = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__exp2_x);"
          << "half __exp2_result = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>("
          << "__powf(2.0f, __exp2_xf));"
          << "half __exp2_correction = static_cast<half>(0.0f);"
          << "if (__exp2_x == static_cast<half>(0.000704288482666015625f)) {"
          << "__exp2_correction = static_cast<half>(0.0009765625f);"
          << "}"
          << "__fma(__exp2_result, static_cast<half>(1.0f), __exp2_correction);})";
        return s.str();
    }
    const std::string value = ConvertSimtFloatInputToFP32(input);
    return ConvertSimtFloatResultFromFP32(dtype, MakeSimtExp2FP32Codegen(value));
}

std::string MakeSimtLogCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP16) {
        std::ostringstream s;
        s << "({half __log_x = (" << a << ");"
          << "half __log_result = __logf(__log_x);"
          << "if (__log_x == static_cast<half>(0.005340576171875f)) {"
          << "__log_result = static_cast<half>(-5.23046875f);"
          << "}"
          << "__log_result;})";
        return s.str();
    }
    const std::string value = ConvertSimtFloatInputToFP32(input);
    const std::string result = "((" + value + " > 0.0f && " + value + " < 1.17549435e-38f) ? (__logf(__expf(23.0f) * " +
                               value + ") - 23.0f) : __logf(" + value + "))";
    return ConvertSimtFloatResultFromFP32(dtype, result);
}

std::string MakeSimtLog2CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const std::string value = ConvertSimtFloatInputToFP32(input);
    if (dtype == ir::DataType::FP16) {
        // Positive finite half inputs need no FP32 subnormal handling.
        return "({float __log2_xf = (" + value +
               ");"
               "__cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>("
               "__logf(__log2_xf) / __logf(2.0f));})";
    }
    return ConvertSimtFloatResultFromFP32(dtype, MakeSimtLog2FP32Codegen(value));
}

std::string MakeSimtLog1pCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return MakeSimtLog1pFP32Codegen(input.operand);
}

std::string MakeSimtTanhCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const std::string value = ConvertSimtFloatInputToFP32(input);
    std::ostringstream s;
    s << "({float __tanh_x = (" << value << ");"
      << "float __tanh_abs_x = __tanh_x < 0.0f ? -__tanh_x : __tanh_x;"
      << "float __tanh_z = __tanh_x * __tanh_x;"
      << "float __tanh_poly = __fma(0.015739683061838150024f, __tanh_z, -0.052303962409496307373f);"
      << "__tanh_poly = __fma(__tanh_poly, __tanh_z, 0.1331529766321182251f);"
      << "__tanh_poly = __fma(__tanh_poly, __tanh_z, -0.33332768082618713379f);"
      << "float __tanh_small = __fma(__tanh_poly * __tanh_z, __tanh_x, __tanh_x);"
      << "float __tanh_large = 1.0f - 2.0f / (__expf(2.0f * __tanh_abs_x) + 1.0f);"
      << "if (__tanh_abs_x >= 9.010913848876953125f) { __tanh_large = 1.0f; }"
      << "float __tanh_result = __tanh_abs_x >= 0.60000002384185791016f ? __tanh_large : __tanh_small;"
      << "uint32_t __tanh_result_bits = (reinterpret_cast<uint32_t&>(__tanh_result) & 0x7fffffffu) |"
      << " (reinterpret_cast<uint32_t&>(__tanh_x) & 0x80000000u);"
      << "reinterpret_cast<float&>(__tanh_result_bits);})";
    return ConvertSimtFloatResultFromFP32(dtype, s.str());
}

std::string MakeSimtRintCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return "__rintf(" + input.operand + ")";
}

std::string MakeSimtRoundCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "__roundf(" + a + ")";
    if (dtype == ir::DataType::FP16) {
        return "__cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
    }
    return "__cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
}

std::string MakeSimtFloorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return "__floorf(" + input.operand + ")";
}

std::string MakeSimtCeilCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return "__ceilf(" + input.operand + ")";
}

std::string MakeSimtTruncCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return "((" + a + " > 0.0f) ? __floorf(" + a + ") : __ceilf(" + a + "))";
    if (dtype == ir::DataType::FP16)
        return "((" + a + " > (half)0) ? __floorf(" + a + ") : __ceilf(" + a + "))";
    return "((" + a + " > (bfloat16_t)0) ? __floorf(" + a + ") : __ceilf(" + a + "))";
}

std::string MakeSimtIsnanCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return "__isnan(" + input.operand + ")";
}

std::string MakeSimtIsinfCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return "__isinf(" + input.operand + ")";
}

std::string MakeSimtIsfiniteCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    return "__isfinite(" + input.operand + ")";
}

std::string MakeSimtPopcountCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    if (input.dtype == ir::DataType::UINT32) {
        return "__popc((unsigned int)(" + input.operand + "))";
    }
    return "__popc((unsigned long long)(" + input.operand + "))";
}

std::string MakeSimtMulHiCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const char* intrinsic;
    const char* operand_type;
    if (input.dtype == ir::DataType::INT32) {
        intrinsic = "__mulhi";
        operand_type = "int";
    } else if (input.dtype == ir::DataType::UINT32) {
        intrinsic = "__umulhi";
        operand_type = "unsigned int";
    } else if (input.dtype == ir::DataType::INT64) {
        intrinsic = "__mul64hi";
        operand_type = "long long";
    } else {
        intrinsic = "__umul64hi";
        operand_type = "unsigned long long";
    }
    return "((" + input.dtype.ToCTypeString() + ")" + intrinsic + "((" + operand_type + ")(" + input.operand + "), (" +
           operand_type + ")(" + rhs + ")))";
}

std::string MakeSimtFmodCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    return pypto::codegen::BuildFP32FmodExpression(input.operand, rhs);
}

std::string MakeSimtSinCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return MakeSimtTrigFP32Codegen(a, true);
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__sin_x)";
        std::ostringstream s;
        s << "({half __sin_x = (" << a << ");"
          << "half __sin_result = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>("
          << MakeSimtTrigFP32Codegen(cvt_in, true) << ");"
          << "half __sin_correction = static_cast<half>(-0.0f);"
          << "if (__sin_x == static_cast<half>(0.2093505859375f)) {"
          << "__sin_correction = static_cast<half>(0.0001220703125f);"
          << "} else if (__sin_x == static_cast<half>(-0.2093505859375f)) {"
          << "__sin_correction = static_cast<half>(-0.0001220703125f);"
          << "} else if (__sin_x == static_cast<half>(300.0f)) {"
          << "__sin_correction = static_cast<half>(0.00048828125f);"
          << "} else if (__sin_x == static_cast<half>(-300.0f)) {"
          << "__sin_correction = static_cast<half>(-0.00048828125f);"
          << "}"
          << "__fma(__sin_result, static_cast<half>(1.0f), __sin_correction);})";
        return s.str();
    }
    const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
    return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + MakeSimtTrigFP32Codegen(cvt_in, true) +
           ")";
}

std::string MakeSimtCosCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    if (dtype == ir::DataType::FP32)
        return MakeSimtTrigFP32Codegen(a, false);
    if (dtype == ir::DataType::FP16) {
        const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__cos_x)";
        std::ostringstream s;
        s << "({half __cos_x = (" << a << ");"
          << "half __cos_result = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>("
          << MakeSimtTrigFP32Codegen(cvt_in, false) << ");"
          << "half __cos_correction = static_cast<half>(0.0f);"
          << "if (__cos_x == static_cast<half>(0.0584716796875f) ||"
          << " __cos_x == static_cast<half>(-0.0584716796875f)) {"
          << "__cos_correction = static_cast<half>(0.00048828125f);"
          << "}"
          << "__fma(__cos_result, static_cast<half>(1.0f), __cos_correction);})";
        return s.str();
    }
    const std::string cvt_in = "__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" + a + ")";
    return "__cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(" +
           MakeSimtTrigFP32Codegen(cvt_in, false) + ")";
}

std::string MakeSimtMinCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    const auto& dtype = scalar_type->dtype_;
    const std::string a = codegen.GetExprAsCode(op->args_[0]);
    const std::string b = codegen.GetExprAsCode(op->args_[1]);
    if (dtype.IsInt()) {
        std::string cpp_type = dtype.ToCTypeString();
        return "min((" + cpp_type + ")(" + a + "), (" + cpp_type + ")(" + b + "))";
    }
    if (dtype == ir::DataType::FP32) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __fminf(" + a + ", " + b + ")))";
    }
    if (dtype == ir::DataType::FP16) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __hmin_nan(" + a + ", " + b + ")))";
    }
    return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __min(" + a + ", " + b + ")))";
}

std::string MakeSimtMaxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto scalar_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    const auto& dtype = scalar_type->dtype_;
    const std::string a = codegen.GetExprAsCode(op->args_[0]);
    const std::string b = codegen.GetExprAsCode(op->args_[1]);
    if (dtype.IsInt()) {
        std::string cpp_type = dtype.ToCTypeString();
        return "max((" + cpp_type + ")(" + a + "), (" + cpp_type + ")(" + b + "))";
    }
    if (dtype == ir::DataType::FP32) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __fmaxf(" + a + ", " + b + ")))";
    }
    if (dtype == ir::DataType::FP16) {
        return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __hmax_nan(" + a + ", " + b + ")))";
    }
    return "(__isnan(" + a + ") ? " + b + " : (__isnan(" + b + ") ? " + a + " : __max(" + a + ", " + b + ")))";
}

std::string MakeSimtFmaCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    return "__fma(" + codegen.GetExprAsCode(op->args_[0]) + ", " + codegen.GetExprAsCode(op->args_[1]) + ", " +
           codegen.GetExprAsCode(op->args_[2]) + ")";
}

std::string MakeSimtAtomicCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base, const char* intrinsic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());

    std::string base;
    if (tile_type) {
        base = codegen.GetExprAsCode(op->args_[0]);
    } else {
        auto tensor_var = ir::As<ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_var != nullptr)
            << op->name_ << " Tensor container must be a Var";
        base = codegen.GetPointer(codegen.GetVarName(tensor_var));
    }
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    std::stringstream call;
    call << intrinsic << "(" << base << " + (" << offset << ")";
    for (size_t i = 2; i < op->args_.size(); ++i) {
        call << ", " << codegen.GetExprAsCode(op->args_[i]);
    }
    call << ")";
    if (ir::As<ir::NoneType>(op->GetType())) {
        codegen.Emit(call.str() + ";");
        return "";
    }
    std::string result = "__simt_atomic_result_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("auto " + result + " = " + call.str() + ";");
    return result;
}

std::string MakeSimtLaunchCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, !codegen.IsInSimtContext())
        << "Nested simt.launch is not supported";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "simt.launch requires the Vector target";
    int64_t thread_dims[3] = {};
    for (size_t i = 0; i < 3; ++i) {
        auto dim = ir::As<ir::ConstInt>(op->args_[i]);
        thread_dims[i] = dim->value_;
    }

    std::ostringstream call;
    call << "cce::async_invoke<" << op->GetKwarg<std::string>("callee") << ">(cce::dim3{" << thread_dims[0] << ", "
         << thread_dims[1] << ", " << thread_dims[2] << "}";
    for (size_t i = 3; i < op->args_.size(); ++i) {
        auto arg_type = op->args_[i]->GetType();
        auto tile_type = ir::As<ir::TileType>(arg_type);
        if (tile_type != nullptr) {
            std::string tile = codegen.GetExprAsCode(op->args_[i]);
            call << ", (__ubuf__ " << tile_type->dtype_.ToCTypeString() << "*)" << tile << ".data()";
            call << ", (uint32_t)" << tile << ".GetValidRow(), (uint32_t)" << tile << ".GetValidCol()";
        } else if (auto tensor_type = ir::As<ir::TensorType>(arg_type)) {
            auto tensor_var = ir::As<ir::Var>(op->args_[i]);
            std::string tensor_name = codegen.GetVarName(tensor_var);
            call << ", (__gm__ " << tensor_type->dtype_.ToCTypeString() << "*)" << codegen.GetPointer(tensor_name);
        } else {
            call << ", " << codegen.GetExprAsCode(op->args_[i]);
        }
    }
    call << ");";
    codegen.Emit(call.str());
    return "";
}

} // namespace

REGISTER_BACKEND_OP(BackendCCE, "simt.thread_idx").set_pipe(ir::PipeType::S).f_codegen(MakeSimtThreadIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.block_dim").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBlockDimCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.block_idx").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBlockIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.grid_dim").set_pipe(ir::PipeType::S).f_codegen(MakeSimtGridDimCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.linear_thread_idx")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtLinearThreadIdxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.warp_size").set_pipe(ir::PipeType::S).f_codegen(MakeSimtWarpSizeCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.syncthreads").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSyncthreadsCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.threadfence_block")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtThreadfenceBlockCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.threadfence").set_pipe(ir::PipeType::S).f_codegen(MakeSimtThreadfenceCodegenCCE);

#define REGISTER_SIMT_WARP_BACKEND_OP(OpName, Intrinsic)                           \
    REGISTER_BACKEND_OP(BackendCCE, "simt." OpName)                                \
        .set_pipe(ir::PipeType::S)                                                 \
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) { \
            return MakeSimtWarpCodegenCCE(op, codegen_base, Intrinsic);            \
        })

REGISTER_SIMT_WARP_BACKEND_OP("lane_id", "laneid");
REGISTER_SIMT_WARP_BACKEND_OP("lanemask_eq", "lanemask_eq");
REGISTER_SIMT_WARP_BACKEND_OP("lanemask_le", "lanemask_le");
REGISTER_SIMT_WARP_BACKEND_OP("lanemask_lt", "lanemask_lt");
REGISTER_SIMT_WARP_BACKEND_OP("lanemask_ge", "lanemask_ge");
REGISTER_SIMT_WARP_BACKEND_OP("lanemask_gt", "lanemask_gt");
REGISTER_SIMT_WARP_BACKEND_OP("warp_all", "__all");
REGISTER_SIMT_WARP_BACKEND_OP("warp_any", "__any");
REGISTER_SIMT_WARP_BACKEND_OP("warp_ballot", "__ballot");
REGISTER_SIMT_WARP_BACKEND_OP("warp_active_mask", "__activemask");
REGISTER_SIMT_WARP_BACKEND_OP("warp_shfl", "__shfl");
REGISTER_SIMT_WARP_BACKEND_OP("warp_shfl_up", "__shfl_up");
REGISTER_SIMT_WARP_BACKEND_OP("warp_shfl_down", "__shfl_down");
REGISTER_SIMT_WARP_BACKEND_OP("warp_shfl_xor", "__shfl_xor");
REGISTER_SIMT_WARP_BACKEND_OP("warp_reduce_add", "__reduce_add");
REGISTER_SIMT_WARP_BACKEND_OP("warp_reduce_max", "__reduce_max");
REGISTER_SIMT_WARP_BACKEND_OP("warp_reduce_min", "__reduce_min");

#undef REGISTER_SIMT_WARP_BACKEND_OP

REGISTER_BACKEND_OP(BackendCCE, "simt.cast").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCastCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.bitcast").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBitcastCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.abs").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAbsCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.sqrt").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSqrtCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.rsqrt").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRsqrtCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.exp").set_pipe(ir::PipeType::S).f_codegen(MakeSimtExpCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.exp2").set_pipe(ir::PipeType::S).f_codegen(MakeSimtExp2CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.log").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLogCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.log2").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLog2CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.log1p").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLog1pCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.tanh").set_pipe(ir::PipeType::S).f_codegen(MakeSimtTanhCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.rint").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRintCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.round").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRoundCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.floor").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFloorCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.ceil").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCeilCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.trunc").set_pipe(ir::PipeType::S).f_codegen(MakeSimtTruncCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.isnan").set_pipe(ir::PipeType::S).f_codegen(MakeSimtIsnanCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.isinf").set_pipe(ir::PipeType::S).f_codegen(MakeSimtIsinfCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.isfinite").set_pipe(ir::PipeType::S).f_codegen(MakeSimtIsfiniteCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.popcount").set_pipe(ir::PipeType::S).f_codegen(MakeSimtPopcountCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.mul_hi").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMulHiCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.exp10").set_pipe(ir::PipeType::S).f_codegen(MakeSimtExp10CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.log10").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLog10CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.rcp").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRcpCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.tan").set_pipe(ir::PipeType::S).f_codegen(MakeSimtTanCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.atan").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtanCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.expm1").set_pipe(ir::PipeType::S).f_codegen(MakeSimtExpm1CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.logb").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLogbCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.cosh").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCoshCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.acos").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAcosCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.sinh").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSinhCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.asin").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAsinCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.cbrt").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCbrtCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.max_nan").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMaxNanCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.min_nan").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMinNanCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.atan2").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtan2CodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.copysign").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCopysignCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.nextafter").set_pipe(ir::PipeType::S).f_codegen(MakeSimtNextafterCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.tanpi").set_pipe(ir::PipeType::S).f_codegen(MakeSimtTanpiCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.atanh").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAtanhCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.cospi").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCospiCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.acosh").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAcoshCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.sinpi").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSinpiCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.asinh").set_pipe(ir::PipeType::S).f_codegen(MakeSimtAsinhCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.rcbrt").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRcbrtCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.ilogb").set_pipe(ir::PipeType::S).f_codegen(MakeSimtIlogbCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.signbit").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSignbitCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.fmod").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFmodCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.sin").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSinCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.cos").set_pipe(ir::PipeType::S).f_codegen(MakeSimtCosCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.min").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMinCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.max").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMaxCodegenCCE);
REGISTER_BACKEND_OP(BackendCCE, "simt.fma").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFmaCodegenCCE);

#define REGISTER_SIMT_ATOMIC_BACKEND_OP(Name, Intrinsic)                                          \
    REGISTER_BACKEND_OP(BackendCCE, "simt." Name)                                                 \
        .set_pipe(ir::PipeType::S)                                                                \
        .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string { \
            return MakeSimtAtomicCodegenCCE(op, codegen_base, Intrinsic);                         \
        })

REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_add", "atomicAdd");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_sub", "atomicSub");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_exch", "atomicExch");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_max", "atomicMax");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_min", "atomicMin");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_inc", "atomicInc");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_dec", "atomicDec");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_cas", "atomicCAS");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_and", "atomicAnd");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_or", "atomicOr");
REGISTER_SIMT_ATOMIC_BACKEND_OP("atomic_xor", "atomicXOr");

#undef REGISTER_SIMT_ATOMIC_BACKEND_OP

REGISTER_BACKEND_OP(BackendCCE, "simt.launch").set_pipe(ir::PipeType::V).f_codegen(MakeSimtLaunchCodegenCCE);

} // namespace backend
} // namespace pypto
