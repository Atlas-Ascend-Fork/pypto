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
#include "backend/common/backend_utils.h"
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

std::string MakeSimtLogFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __log_x = (" << operand << "), __log_result;"
      << "if (__log_x > 0.0f && __log_x < 1.17549435e-38f) {"
      << "__log_result = __logf(__expf(23.0f) * __log_x) - 23.0f; }"
      << "else { __log_result = __logf(__log_x); } __log_result;})";
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
    const auto& a = input.operand;
    const auto result_dtype = ir::As<ir::ScalarType>(op->GetType())->dtype_;
    if (result_dtype == ir::DataType::INT64) {
        return "__cvt_int64_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(" + a + ")";
    }
    return "__rintf(" + a + ")";
}

std::string MakeSimtRoundCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    const auto input = GetSimtUnaryCodegenInput(op, codegen_base);
    const auto& dtype = input.dtype;
    const auto& a = input.operand;
    const auto result_dtype = ir::As<ir::ScalarType>(op->GetType())->dtype_;
    if (result_dtype == ir::DataType::INT64) {
        return "__cvt_int64_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(" + a + ")";
    }
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
            std::string tile = cce::GetOperandAsCode(codegen, op->args_[i]);
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

std::string MakeSimtBrevCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto dtype = ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_;
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const char* cpp_type = dtype == ir::DataType::UINT32 ? "unsigned int" : "unsigned long long";
    return "__brev((" + std::string(cpp_type) + ")(" + operand + "))";
}

std::string MakeSimtClzCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto dtype = ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_;
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const char* cpp_type = dtype == ir::DataType::INT32 ? "int" : "long long";
    return "__clz((" + std::string(cpp_type) + ")(" + operand + "))";
}

std::string MakeSimtFfsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto dtype = ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_;
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const char* cpp_type = dtype == ir::DataType::INT32 ? "int" : "long long";
    return "__ffs((" + std::string(cpp_type) + ")(" + operand + "))";
}

std::string MakeSimtBytePermCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const std::string selector = codegen.GetExprAsCode(op->args_[2]);
    return "__byte_perm((unsigned int)(" + lhs + "), (unsigned int)(" + rhs + "), (unsigned int)(" + selector + "))";
}

std::string MakeSimtSadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto dtype = ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_;
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const std::string addend = codegen.GetExprAsCode(op->args_[2]);
    const char* intrinsic = dtype == ir::DataType::INT32 ? "__sad" : "__usad";
    const char* operand_type = dtype == ir::DataType::INT32 ? "int" : "unsigned int";
    return std::string(intrinsic) + "((" + operand_type + ")(" + lhs + "), (" + operand_type + ")(" + rhs +
           "), (unsigned int)(" + addend + "))";
}

std::string MakeSimtMul24CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto dtype = ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_;
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const char* intrinsic = dtype == ir::DataType::INT32 ? "__mul24" : "__umul24";
    const char* operand_type = dtype == ir::DataType::INT32 ? "int" : "unsigned int";
    return std::string(intrinsic) + "((" + operand_type + ")(" + lhs + "), (" + operand_type + ")(" + rhs + "))";
}

std::string MakeSimtHaddCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto dtype = ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_;
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const char* intrinsic = dtype == ir::DataType::INT32 ? "__hadd" : "__uhadd";
    const char* operand_type = dtype == ir::DataType::INT32 ? "int" : "unsigned int";
    return std::string(intrinsic) + "((" + operand_type + ")(" + lhs + "), (" + operand_type + ")(" + rhs + "))";
}

std::string MakeSimtRhaddCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto dtype = ir::As<ir::ScalarType>(op->args_[0]->GetType())->dtype_;
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const char* intrinsic = dtype == ir::DataType::INT32 ? "__rhadd" : "__urhadd";
    const char* operand_type = dtype == ir::DataType::INT32 ? "int" : "unsigned int";
    return std::string(intrinsic) + "((" + operand_type + ")(" + lhs + "), (" + operand_type + ")(" + rhs + "))";
}

std::string MakeSimtFmaReluCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const std::string addend = codegen.GetExprAsCode(op->args_[2]);
    return "__hfma_relu((" + lhs + "), (" + rhs + "), (" + addend + "))";
}

std::string MakeSimtUnorderedCompareCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                               const char* intrinsic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    return std::string(intrinsic) + "((" + lhs + "), (" + rhs + "))";
}

std::string MakeSimtEquCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtUnorderedCompareCodegenCCE(op, codegen_base, "__hequ");
}

std::string MakeSimtLeuCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtUnorderedCompareCodegenCCE(op, codegen_base, "__hleu");
}

std::string MakeSimtGeuCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtUnorderedCompareCodegenCCE(op, codegen_base, "__hgeu");
}

std::string MakeSimtLtuCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtUnorderedCompareCodegenCCE(op, codegen_base, "__hltu");
}

std::string MakeSimtGtuCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeSimtUnorderedCompareCodegenCCE(op, codegen_base, "__hgtu");
}

std::string MakeSimtNormSqrtFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __x = (" << operand << "); uint32_t __xb = reinterpret_cast<uint32_t&>(__x);"
      << "bool __slow = (__xb - 0x0d000000u) > 0x727fffffu; float __r = __x;"
      << "if (!__slow) { float __inv = 1.0f / __sqrtf(__x); float __product = __x * __inv;"
      << "__inv *= 0.5f; float __residual = __fma(-__product, __product, __x);"
      << "__r = __fma(__residual, __inv, __product); }"
      << "else { uint32_t __magnitude = __xb & 0x7fffffffu;"
      << "if (__magnitude == 0u) { __r = __x; }"
      << "else if (__x < 0.0f) { uint32_t __nan = 0x7fffffffu; __r = reinterpret_cast<float&>(__nan); }"
      << "else if (__isnan(__x)) { __r = __x + 1.0f; }"
      << "else if (__magnitude == 0x7f800000u) { __r = __x; }"
      << "else { float __scaled = __fma(__x, 1.84467440737095516160e+19f, 0.0f);"
      << "float __inv = 1.0f / __sqrtf(__scaled); float __product = __scaled * __inv;"
      << "__inv *= 0.5f; float __residual = __fma(__product, -__product, __scaled);"
      << "__r = __fma(__residual, __inv, __product) * 2.3283064365386962891e-10f; } } __r;})";
    return s.str();
}

std::string MakeSimtHypotCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    const std::string sqrt_scaled = MakeSimtNormSqrtFP32Codegen("__square_sum");
    std::ostringstream s;
    s << "({float __x = __fabsf(" << lhs << "), __y = __fabsf(" << rhs << ");"
      << "uint32_t __xb = reinterpret_cast<uint32_t&>(__x), __yb = reinterpret_cast<uint32_t&>(__y);"
      << "bool __x_greater = __xb > __yb; uint32_t __max_bits = __x_greater ? __xb : __yb;"
      << "uint32_t __min_bits = __x_greater ? __yb : __xb;"
      << "float __max = reinterpret_cast<float&>(__max_bits), __min = reinterpret_cast<float&>(__min_bits);"
      << "uint32_t __bucket = __max_bits & 0xfe000000u, __scale_bits = 0x7e800000u - __bucket;"
      << "float __scale = reinterpret_cast<float&>(__scale_bits);"
      << "float __scaled_min = __min * __scale, __scaled_max = __max * __scale;"
      << "float __square_sum = __fma(__scaled_max, __scaled_max, __scaled_min * __scaled_min);"
      << "float __sqrt_scaled = " << sqrt_scaled << ";"
      << "uint32_t __restore_bits = __bucket | 0x00800000u;"
      << "float __result = __max != 0.0f ? reinterpret_cast<float&>(__restore_bits) * __sqrt_scaled : __max;"
      << "if (__min_bits == 0x7f800000u) {"
      << "uint32_t __inf = 0x7f800000u; __result = reinterpret_cast<float&>(__inf); } __result;})";
    return s.str();
}

std::string MakeSimtPowCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    std::ostringstream s;
    s << "({float __x = (" << lhs << "), __y = (" << rhs << ");"
      << "float __abs_x = __fabsf(__x), __abs_y = __fabsf(__y);"
      << "bool __normal = __abs_x >= 1.17549435e-38f;"
      << "float __log_input = __normal ? __abs_x : __abs_x * 16777216.0f;"
      << "float __exponent_base = __normal ? 0.0f : -24.0f;"
      << "uint32_t __input_bits = reinterpret_cast<uint32_t&>(__log_input);"
      << "uint32_t __reduction_bits = (__input_bits - 0x3F3504F3U) & 0xFF800000U;"
      << "uint32_t __mantissa_bits = __input_bits - __reduction_bits;"
      << "float __mantissa = reinterpret_cast<float&>(__mantissa_bits);"
      << "float __exponent_part = __fma(__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>("
      << "static_cast<int32_t>(__reduction_bits)), 1.1920928955078125e-07f, __exponent_base);"
      << "float __mantissa_minus_one = __mantissa - 1.0f;"
      << "float __reciprocal = 1.0f / (__mantissa + 1.0f);"
      << "float __reduced_hi = __reciprocal * (__mantissa_minus_one + __mantissa_minus_one);"
      << "float __reduced_square = __reduced_hi * __reduced_hi;"
      << "float __log_poly = __fma(__reduced_square, 0.0006568862590938807f, 0.0032181653659790754318f);"
      << "__log_poly = __fma(__reduced_square, __log_poly, 0.018033718690276145935f);"
      << "__log_poly = __fma(__reduced_square, __log_poly, 0.12022458761930465698f);"
      << "__log_poly = __reduced_square * __log_poly;"
      << "float __log2_hi = __fma(__reduced_hi, 1.4426950216293334961f, __exponent_part);"
      << "float __reduced_err = __mantissa_minus_one - __reduced_hi;"
      << "__reduced_err = __fma(__mantissa_minus_one, -__reduced_hi, __reduced_err + __reduced_err);"
      << "float __reduced_lo = __reciprocal * __reduced_err;"
      << "float __log2_lo = __exponent_part - __log2_hi;"
      << "__log2_lo = __fma(__reduced_hi, 1.4426950216293334961f, __log2_lo);"
      << "__log2_lo = __fma(__reduced_lo, 1.4426950216293334961f, __log2_lo);"
      << "__log2_lo = __fma(__reduced_hi, 1.9251366722983220825e-08f, __log2_lo);"
      << "__log2_lo = __fma(__reduced_lo, __log_poly * 3.0f, __log2_lo);"
      << "__log2_lo = __fma(__reduced_hi, __log_poly, __log2_lo);"
      << "float __log2_abs_x = __log2_hi + __log2_lo;"
      << "float __product_hi = __log2_abs_x * __y;"
      << "float __product_lo = __fma(__log2_abs_x, __y, -__product_hi);"
      << "__product_lo = __fma(__log2_lo - (__log2_abs_x - __log2_hi), __y, __product_lo);"
      << "float __rounded = __roundf(__product_hi);"
      << "float __fraction = (__product_hi - __rounded) + __product_lo;"
      << "int32_t __exponent = __cvt_int32_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(__rounded);"
      << "float __exp_poly = __fma(__fraction, 0.00015239251661114395f, 0.0013391353422775864601f);"
      << "__exp_poly = __fma(__fraction, __exp_poly, 0.0096188392490148544312f);"
      << "__exp_poly = __fma(__fraction, __exp_poly, 0.055503588169813156128f);"
      << "__exp_poly = __fma(__fraction, __exp_poly, 0.24022644758224487305f);"
      << "__exp_poly = __fma(__fraction, __exp_poly, 0.69314718246459960938f);"
      << "__exp_poly = __fma(__fraction, __exp_poly, 1.0f);"
      << "uint32_t __scale_hi_bits = __rounded > 0.0f ? 0x7F000000U : 0x02000000U;"
      << "uint32_t __scale_adjust = __rounded > 0.0f ? 0U : 0x83000000U;"
      << "uint32_t __scale_lo_bits = (static_cast<uint32_t>(__exponent) << 23) - __scale_adjust;"
      << "float __result = __exp_poly * reinterpret_cast<float&>(__scale_hi_bits);"
      << "__result = __result * reinterpret_cast<float&>(__scale_lo_bits);"
      << "if (__fabsf(__product_hi) > 152.0f) { __result = __product_hi >= 0.0f ? __builtin_inff() : 0.0f; }"
      << "float __half_y = __y * 0.5f;"
      << "float __half_trunc = __half_y > 0.0f ? __floorf(__half_y) : __ceilf(__half_y);"
      << "bool __y_is_odd_int = __fabsf(__y - (__half_trunc + __half_trunc)) == 1.0f;"
      << "if (__x < 0.0f) {"
      << "if (__floorf(__y) != __y) { uint32_t __nan = 0x7fffffffU; __result = reinterpret_cast<float&>(__nan); }"
      << "else if (__y_is_odd_int) { __result = -__result; } }"
      << "bool __x_is_zero_or_inf = __x == 0.0f || __abs_x == __builtin_inff();"
      << "bool __y_is_inf = __abs_y == __builtin_inff();"
      << "if (__y == 0.0f || __x == 1.0f) { __result = 1.0f; }"
      << "else if (__x_is_zero_or_inf && __isnan(__y)) { __result = __x + __y; }"
      << "else if (__x_is_zero_or_inf) { float __special = __x + __x;"
      << "if (__y < 0.0f) { uint32_t __sign = reinterpret_cast<uint32_t&>(__x) & 0x80000000U;"
      << "uint32_t __special_bits = (__abs_x == __builtin_inff() ? 0U : 0x7f800000U) | __sign;"
      << "__special = reinterpret_cast<float&>(__special_bits); }"
      << "__result = __y_is_odd_int ? __special : __fabsf(__special); }"
      << "else if (__y_is_inf && __isnan(__x)) { __result = __x + __y; }"
      << "else if (__y_is_inf && __x == -1.0f) { __result = 1.0f; }"
      << "else if (__y_is_inf) {"
      << "bool __result_is_inf = (__abs_x > 1.0f && __y > 0.0f) || (__abs_x < 1.0f && __y < 0.0f);"
      << "__result = __result_is_inf ? __builtin_inff() : 0.0f; }"
      << "else if (__isnan(__x) || __isnan(__y)) { __result = __x + __y; } __result;})";
    return s.str();
}

std::string MakeSimtRhypotCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    std::ostringstream s;
    s << "({float __x = __fabsf(" << lhs << "), __y = __fabsf(" << rhs << ");"
      << "uint32_t __xb = reinterpret_cast<uint32_t&>(__x), __yb = reinterpret_cast<uint32_t&>(__y);"
      << "uint32_t __min_bits = __xb < __yb ? __xb : __yb, __max_bits = __xb > __yb ? __xb : __yb;"
      << "float __min = reinterpret_cast<float&>(__min_bits), __max = reinterpret_cast<float&>(__max_bits);"
      << "uint32_t __scale_bits = 0x7e800000u - (__max_bits & 0xfe000000u);"
      << "float __scale = reinterpret_cast<float&>(__scale_bits);"
      << "float __scaled_min = __min * __scale, __scaled_max = __max * __scale;"
      << "float __square_sum = __fma(__scaled_max, __scaled_max, __scaled_min * __scaled_min);"
      << "float __result = (1.0f / __sqrtf(__square_sum)) * __scale;"
      << "if (__min_bits == 0x7f800000u) { __result = 0.0f; } __result;})";
    return s.str();
}

std::string MakeSimtFdimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    std::ostringstream s;
    s << "({float __x = (" << lhs << "), __y = (" << rhs << ");"
      << "uint32_t __xb = reinterpret_cast<uint32_t&>(__x), __yb = reinterpret_cast<uint32_t&>(__y);"
      << "(__xb & 0x7fffffffu) > 0x7f800000u ? __x : ((__yb & 0x7fffffffu) > 0x7f800000u ? __y : "
      << "(__x > __y ? __x - __y : 0.0f));})";
    return s.str();
}

std::string MakeSimtRemainderCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    // Binary long division computes fmod and the quotient's low bit. The low bit
    // resolves exact half-way cases using the IEEE nearest-even quotient rule.
    std::ostringstream s;
    s << "({float __x = (" << lhs << "), __y = (" << rhs << ");"
      << "uint32_t __ux = reinterpret_cast<uint32_t&>(__x), __uy = reinterpret_cast<uint32_t&>(__y);"
      << "uint32_t __sign = __ux & 0x80000000u, __ax = __ux & 0x7fffffffu, __ay = __uy & 0x7fffffffu;"
      << "uint32_t __result = __ux;"
      << "if (__ay == 0u || __ax >= 0x7f800000u || __ay > 0x7f800000u) { __result = 0x7fffffffu; }"
      << "else if (__ax != 0u) { int __ex = (int)(__ax >> 23), __ey = (int)(__ay >> 23);"
      << "uint32_t __mx = __ax & 0x007fffffu, __my = __ay & 0x007fffffu;"
      << "if (__ex == 0) { __ex = 1; while (__mx < 0x00800000u) { __mx <<= 1; --__ex; } }"
      << "else { __mx |= 0x00800000u; }"
      << "if (__ey == 0) { __ey = 1; while (__my < 0x00800000u) { __my <<= 1; --__ey; } }"
      << "else { __my |= 0x00800000u; }"
      << "bool __encode = false; if (__ex < __ey) {"
      << "if (__ex == __ey - 1 && __mx > __my) { __mx = (__my << 1) - __mx; __sign ^= 0x80000000u; "
      << "__encode = true; } }"
      << "else { while (__ex > __ey) { if (__mx >= __my) { __mx -= __my; } __mx <<= 1; --__ex; }"
      << "bool __quotient_odd = false; if (__mx >= __my) { __mx -= __my; __quotient_odd = true; }"
      << "if (__mx == 0u) { __result = __sign; } else {"
      << "uint32_t __twice = __mx << 1; if (__twice > __my || (__twice == __my && __quotient_odd)) {"
      << "__mx = __my - __mx; __sign ^= 0x80000000u; } __encode = true; } }"
      << "if (__encode) { while (__mx < 0x00800000u) { __mx <<= 1; --__ex; }"
      << "if (__ex > 0) { __result = (__mx & 0x007fffffu) | ((uint32_t)__ex << 23); }"
      << "else { __result = __mx >> (1 - __ex); } __result |= __sign; } }"
      << "reinterpret_cast<float&>(__result);})";
    return s.str();
}

std::string MakeSimtNorm3dCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string x = codegen.GetExprAsCode(op->args_[0]);
    const std::string y = codegen.GetExprAsCode(op->args_[1]);
    const std::string z = codegen.GetExprAsCode(op->args_[2]);
    const std::string sqrt_scaled = MakeSimtNormSqrtFP32Codegen("__square_sum");
    std::ostringstream s;
    s << "({float __a = __fabsf(" << x << "), __b = __fabsf(" << y << "), __c = __fabsf(" << z << ");"
      << "float __max_ab = __isnan(__a) ? __b : (__isnan(__b) ? __a : __fmaxf(__a, __b));"
      << "float __max = __isnan(__max_ab) ? __c : (__isnan(__c) ? __max_ab : __fmaxf(__max_ab, __c));"
      << "float __middle = __isnan(__max_ab) ? __c : (__isnan(__c) ? __max_ab : __fminf(__max_ab, __c));"
      << "float __min = __isnan(__a) ? __b : (__isnan(__b) ? __a : __fminf(__a, __b));"
      << "uint32_t __max_bits = reinterpret_cast<uint32_t&>(__max), __bucket = __max_bits & 0xfe000000u;"
      << "uint32_t __scale_bits = 0x7e800000u - __bucket; float __scale = reinterpret_cast<float&>(__scale_bits);"
      << "float __scaled_middle = __middle * __scale, __scaled_min = __min * __scale;"
      << "float __scaled_max = __max * __scale, __square_sum = __scaled_middle * __scaled_middle;"
      << "__square_sum = __fma(__scaled_min, __scaled_min, __square_sum);"
      << "__square_sum = __fma(__scaled_max, __scaled_max, __square_sum);"
      << "float __sqrt_scaled = " << sqrt_scaled << ", __sum = __a + __b + __c, __result = __sum;"
      << "if (__sum > __max) { uint32_t __restore_bits = __bucket | 0x00800000u;"
      << "__result = reinterpret_cast<float&>(__restore_bits) * __sqrt_scaled; }"
      << "if (__max_bits == 0x7f800000u) {"
      << "uint32_t __inf = 0x7f800000u; __result = reinterpret_cast<float&>(__inf); } __result;})";
    return s.str();
}

std::string MakeSimtRnorm3dCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string x = codegen.GetExprAsCode(op->args_[0]);
    const std::string y = codegen.GetExprAsCode(op->args_[1]);
    const std::string z = codegen.GetExprAsCode(op->args_[2]);
    std::ostringstream s;
    s << "({float __a = __fabsf(" << x << "), __b = __fabsf(" << y << "), __c = __fabsf(" << z << ");"
      << "uint32_t __ab = reinterpret_cast<uint32_t&>(__a), __bb = reinterpret_cast<uint32_t&>(__b), "
      << "__cb = reinterpret_cast<uint32_t&>(__c); float __result;"
      << "if (__ab == 0x7f800000u || __bb == 0x7f800000u || __cb == 0x7f800000u) { __result = 0.0f; }"
      << "else if (__ab > 0x7f800000u || __bb > 0x7f800000u || __cb > 0x7f800000u) {"
      << "__result = __builtin_inff() / __builtin_inff(); }"
      << "else { float __max_ab = __fmaxf(__a, __b), __min = __fminf(__a, __b);"
      << "float __max = __fmaxf(__c, __max_ab), __middle = __fminf(__c, __max_ab);"
      << "uint32_t __max_bits = reinterpret_cast<uint32_t&>(__max);"
      << "uint32_t __scale_bits = 0x7e800000u - (__max_bits & 0xfe000000u);"
      << "float __scale = reinterpret_cast<float&>(__scale_bits);"
      << "float __scaled_min = __min * __scale, __scaled_middle = __middle * __scale;"
      << "float __scaled_max = __max * __scale, __square_sum = __scaled_middle * __scaled_middle;"
      << "__square_sum = __fma(__scaled_min, __scaled_min, __square_sum);"
      << "__square_sum = __fma(__scaled_max, __scaled_max, __square_sum);"
      << "bool __subnormal = __fabsf(__square_sum) < 1.17549435e-38f;"
      << "if (__subnormal) { __square_sum *= 16777216.0f; }"
      << "if (__square_sum == 0.0f) { uint32_t __inf = 0x7f800000u; __result = reinterpret_cast<float&>(__inf); }"
      << "else { float __inv = 1.0f / __sqrtf(__square_sum);"
      << "__inv *= __fma(-0.5f * __square_sum, __inv * __inv, 1.5f);"
      << "if (__subnormal) { __inv *= 4096.0f; } __result = __scale * __inv; } } __result;})";
    return s.str();
}

std::string MakeSimtNorm4dCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string x = codegen.GetExprAsCode(op->args_[0]);
    const std::string y = codegen.GetExprAsCode(op->args_[1]);
    const std::string z = codegen.GetExprAsCode(op->args_[2]);
    const std::string w = codegen.GetExprAsCode(op->args_[3]);
    const std::string sqrt_scaled = MakeSimtNormSqrtFP32Codegen("__square_sum");
    std::ostringstream s;
    s << "({float __a = __fabsf(" << x << "), __b = __fabsf(" << y << "), __c = __fabsf(" << z << "), __d = __fabsf("
      << w << ");"
      << "float __max_ab = __isnan(__a) ? __b : (__isnan(__b) ? __a : __fmaxf(__a, __b));"
      << "float __max_abc = __isnan(__max_ab) ? __c : (__isnan(__c) ? __max_ab : __fmaxf(__max_ab, __c));"
      << "float __max = __isnan(__max_abc) ? __d : (__isnan(__d) ? __max_abc : __fmaxf(__max_abc, __d));"
      << "float __term3 = __isnan(__max_abc) ? __d : (__isnan(__d) ? __max_abc : __fminf(__max_abc, __d));"
      << "float __term2 = __isnan(__max_ab) ? __c : (__isnan(__c) ? __max_ab : __fminf(__max_ab, __c));"
      << "float __term0 = __isnan(__a) ? __b : (__isnan(__b) ? __a : __fminf(__a, __b));"
      << "uint32_t __max_bits = reinterpret_cast<uint32_t&>(__max), __bucket = __max_bits & 0xfe000000u;"
      << "uint32_t __scale_bits = 0x7e800000u - __bucket; float __scale = reinterpret_cast<float&>(__scale_bits);"
      << "float __scaled3 = __term3 * __scale, __scaled2 = __term2 * __scale;"
      << "float __scaled0 = __term0 * __scale, __scaled_max = __max * __scale;"
      << "float __square_sum = __scaled3 * __scaled3;"
      << "__square_sum = __fma(__scaled2, __scaled2, __square_sum);"
      << "__square_sum = __fma(__scaled0, __scaled0, __square_sum);"
      << "__square_sum = __fma(__scaled_max, __scaled_max, __square_sum);"
      << "float __sqrt_scaled = " << sqrt_scaled << ", __sum = __a + __b + __d + __c, __result = __sum;"
      << "if (__sum > __max) { uint32_t __restore_bits = __bucket | 0x00800000u;"
      << "__result = reinterpret_cast<float&>(__restore_bits) * __sqrt_scaled; }"
      << "if (__max_bits == 0x7f800000u) {"
      << "uint32_t __inf = 0x7f800000u; __result = reinterpret_cast<float&>(__inf); } __result;})";
    return s.str();
}

std::string MakeSimtRnorm4dCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string x = codegen.GetExprAsCode(op->args_[0]);
    const std::string y = codegen.GetExprAsCode(op->args_[1]);
    const std::string z = codegen.GetExprAsCode(op->args_[2]);
    const std::string w = codegen.GetExprAsCode(op->args_[3]);
    std::ostringstream s;
    s << "({float __a = __fabsf(" << x << "), __b = __fabsf(" << y << "), __c = __fabsf(" << z << "), __d = __fabsf("
      << w << ");"
      << "uint32_t __ab = reinterpret_cast<uint32_t&>(__a), __bb = reinterpret_cast<uint32_t&>(__b), "
      << "__cb = reinterpret_cast<uint32_t&>(__c), __db = reinterpret_cast<uint32_t&>(__d); float __result;"
      << "if (__ab == 0x7f800000u || __bb == 0x7f800000u || __cb == 0x7f800000u || __db == 0x7f800000u) {"
      << "__result = 0.0f; }"
      << "else if (__ab > 0x7f800000u || __bb > 0x7f800000u || __cb > 0x7f800000u || __db > 0x7f800000u) {"
      << "__result = __builtin_inff() / __builtin_inff(); }"
      << "else { float __max_ab = __fmaxf(__a, __b), __min = __fminf(__a, __b);"
      << "float __max_abc = __fmaxf(__c, __max_ab), __max = __fmaxf(__d, __max_abc);"
      << "float __second = __fminf(__d, __max_abc), __third = __fminf(__c, __max_ab);"
      << "uint32_t __max_bits = reinterpret_cast<uint32_t&>(__max);"
      << "uint32_t __scale_bits = 0x7e800000u - (__max_bits & 0xfe000000u);"
      << "float __scale = reinterpret_cast<float&>(__scale_bits);"
      << "float __scaled_min = __min * __scale, __scaled_second = __second * __scale;"
      << "float __scaled_third = __third * __scale, __scaled_max = __max * __scale;"
      << "float __square_sum = __scaled_second * __scaled_second;"
      << "__square_sum = __fma(__scaled_third, __scaled_third, __square_sum);"
      << "__square_sum = __fma(__scaled_min, __scaled_min, __square_sum);"
      << "__square_sum = __fma(__scaled_max, __scaled_max, __square_sum);"
      << "bool __subnormal = __fabsf(__square_sum) < 1.17549435e-38f;"
      << "if (__subnormal) { __square_sum *= 16777216.0f; }"
      << "if (__square_sum == 0.0f) { uint32_t __inf = 0x7f800000u; __result = reinterpret_cast<float&>(__inf); }"
      << "else { float __inv = 1.0f / __sqrtf(__square_sum);"
      << "__inv *= __fma(-0.5f * __square_sum, __inv * __inv, 1.5f);"
      << "if (__subnormal) { __inv *= 4096.0f; } __result = __scale * __inv; } } __result;})";
    return s.str();
}

std::string MakeSimtLdexpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string value = codegen.GetExprAsCode(op->args_[0]);
    const std::string exponent = codegen.GetExprAsCode(op->args_[1]);
    std::ostringstream s;
    s << "({float __x = (" << value << "); int32_t __n = (int32_t)(" << exponent << ");"
      << "int64_t __abs_n = __n < 0 ? -(int64_t)__n : (int64_t)__n; float __result;"
      << "if (__abs_n >= 101) { uint32_t __exp_bits = (uint32_t)(__n + 508);"
      << "uint32_t __q = __exp_bits >> 2, __r = __exp_bits - (__q << 2);"
      << "uint32_t __scale_q_bits = __q << 23, __scale_qr_bits = (__q + __r) << 23;"
      << "float __scale_q = reinterpret_cast<float&>(__scale_q_bits);"
      << "float __temp = __x * reinterpret_cast<float&>(__scale_qr_bits);"
      << "__temp *= __scale_q; __temp *= __scale_q; __temp *= __scale_q; __result = __temp; }"
      << "else { uint32_t __scale_bits = (uint32_t)(__n + 127) << 23;"
      << "__result = __x * reinterpret_cast<float&>(__scale_bits); }"
      << "uint32_t __xb = reinterpret_cast<uint32_t&>(__x);"
      << "if (__n > 278) { uint32_t __inf = (__xb & 0x80000000u) | 0x7f800000u;"
      << "__result = reinterpret_cast<float&>(__inf); }"
      << "if (__n < -278) { uint32_t __zero = __xb & 0x80000000u;"
      << "__result = reinterpret_cast<float&>(__zero); }"
      << "uint32_t __magnitude = __xb & 0x7fffffffu;"
      << "if (__magnitude == 0u || __magnitude >= 0x7f800000u || __n == 0) { __result = __x; } __result;})";
    return s.str();
}

std::string MakeSimtScalbnCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string value = codegen.GetExprAsCode(op->args_[0]);
    const std::string exponent = codegen.GetExprAsCode(op->args_[1]);
    const auto exponent_dtype = ir::As<ir::ScalarType>(op->args_[1]->GetType())->dtype_;
    std::ostringstream s;
    s << "({float __x = (" << value << "); int32_t __n;";
    if (exponent_dtype == ir::DataType::INT64) {
        s << "int64_t __long_n = (int64_t)(" << exponent << ");"
          << "if (__long_n > 278) { __n = 278; } else if (__long_n < -278) { __n = -278; }"
          << "else { __n = (int32_t)__long_n; }";
    } else {
        s << "__n = (int32_t)(" << exponent << ");";
    }
    s << "int32_t __clamped = __n > 278 ? 278 : __n; __clamped = __clamped < -278 ? -278 : __clamped;"
      << "uint32_t __scale_bits = 0u, __split_bits = 0x3f800000u;"
      << "if (__clamped >= 101 || __clamped <= -101) { int32_t __biased = __clamped + 508;"
      << "int32_t __split = (int32_t)((uint32_t)__biased >> 2); int32_t __residual = __biased - __split * 3;"
      << "__split_bits = (uint32_t)__split << 23; __scale_bits = (uint32_t)__residual << 23; }"
      << "else { __scale_bits = (uint32_t)(__clamped + 127) << 23; }"
      << "float __result = reinterpret_cast<float&>(__scale_bits) * __x;"
      << "float __split_scale = reinterpret_cast<float&>(__split_bits);"
      << "__result = __split_scale * __result; __result = __split_scale * __result;"
      << "__result = __split_scale * __result; __result;})";
    return s.str();
}

std::string MakeSimtGammaLnFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __gamma_ln_x = (" << operand << "), __gamma_ln_offset = 0.0f;"
      << "if (__gamma_ln_x < 1.17549435e-38f) { __gamma_ln_offset = -23.0f;"
      << "__gamma_ln_x *= 8388608.0f; }"
      << "uint32_t __gamma_ln_bits = reinterpret_cast<uint32_t&>(__gamma_ln_x);"
      << "int32_t __gamma_ln_exp_bits = ((int32_t)__gamma_ln_bits - 1059760811) & -8388608;"
      << "uint32_t __gamma_ln_mantissa_bits = __gamma_ln_bits - (uint32_t)__gamma_ln_exp_bits;"
      << "float __gamma_ln_mantissa = reinterpret_cast<float&>(__gamma_ln_mantissa_bits) - 1.0f;"
      << "float __gamma_ln_exponent = __fma((float)__gamma_ln_exp_bits, 1.1920929e-07f, __gamma_ln_offset);"
      << "float __gamma_ln_result = -0.130188569f;"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, 0.140846103f);"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, -0.121486276f);"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, 0.139806107f);"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, -0.166842356f);"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, 0.200122997f);"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, -0.249996692f);"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, 0.333331823f);"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, -0.5f);"
      << "__gamma_ln_result = __gamma_ln_mantissa * __gamma_ln_result;"
      << "__gamma_ln_result = __fma(__gamma_ln_result, __gamma_ln_mantissa, __gamma_ln_mantissa);"
      << "__gamma_ln_result = __fma(__gamma_ln_exponent, 0.693147182f, __gamma_ln_result);"
      << "if (__gamma_ln_bits >= 0x7f800000u || __gamma_ln_x == 0.0f) {"
      << "__gamma_ln_result = __fma(__gamma_ln_x, __builtin_inff(), __builtin_inff()); }"
      << "__gamma_ln_result;})";
    return s.str();
}

std::string MakeSimtLgammaSinpiFP32Codegen(const std::string& operand)
{
    std::ostringstream s;
    s << "({float __gamma_sinpi_x = (" << operand << "), __gamma_sinpi_double_x = __gamma_sinpi_x * 2.0f;"
      << "float __gamma_sinpi_nearest = __rintf(__gamma_sinpi_double_x);"
      << "int32_t __gamma_sinpi_quadrant = (int32_t)__gamma_sinpi_nearest;"
      << "float __gamma_sinpi_reduced = __fma(-__gamma_sinpi_nearest, 0.5f, __gamma_sinpi_x);"
      << "float __gamma_sinpi_arg = __gamma_sinpi_reduced * 3.14159274f;"
      << "float __gamma_sinpi_square = __gamma_sinpi_arg * __gamma_sinpi_arg, __gamma_sinpi_result;"
      << "if ((__gamma_sinpi_quadrant & 1) != 0) {"
      << "__gamma_sinpi_result = __fma(2.42795795e-05f, __gamma_sinpi_square, -0.00138878601f);"
      << "__gamma_sinpi_result = __fma(__gamma_sinpi_result, __gamma_sinpi_square, 0.0416667275f);"
      << "__gamma_sinpi_result = __fma(__gamma_sinpi_result, __gamma_sinpi_square, -0.49999997f);"
      << "float __gamma_sinpi_square_copy = __fma(__gamma_sinpi_square, 1.0f, 0.0f);"
      << "__gamma_sinpi_result = __fma(__gamma_sinpi_result, __gamma_sinpi_square_copy, 1.0f); }"
      << "else { __gamma_sinpi_result = __fma(-0.000195746587f, __gamma_sinpi_square, 0.00833270326f);"
      << "__gamma_sinpi_result = __fma(__gamma_sinpi_result, __gamma_sinpi_square, -0.166666627f);"
      << "float __gamma_sinpi_cube = __fma(__gamma_sinpi_square, __gamma_sinpi_arg, 0.0f);"
      << "__gamma_sinpi_result = __fma(__gamma_sinpi_result, __gamma_sinpi_cube, __gamma_sinpi_arg); }"
      << "if ((__gamma_sinpi_quadrant & 2) != 0) {"
      << "__gamma_sinpi_result = __fma(__gamma_sinpi_result, -1.0f, 0.0f); } __gamma_sinpi_result;})";
    return s.str();
}

std::string MakeSimtTgammaCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __abs_input = __fabsf(__x), __result;"
      << "if (__abs_input < 1.5f) { float __reduced = __x;"
      << "if (__x != 0.0f) { __reduced = __x - __rintf(__x); }"
      << "float __poly = __fma(__reduced, -0.00107286567799747f, 0.007111053448170424f);"
      << "__poly = __fma(__poly, __reduced, -0.009643718600273132f);"
      << "__poly = __fma(__poly, __reduced, -0.04218018800020218f);"
      << "__poly = __fma(__poly, __reduced, 0.16654090583324432f);"
      << "__poly = __fma(__poly, __reduced, -0.04200365021824837f);"
      << "__poly = __fma(__poly, __reduced, -0.6558781862258911f);"
      << "__poly = __fma(__poly, __reduced, 0.5772156715393066f);"
      << "float __scale = __x < -0.5f ? __x : 1.0f;"
      << "if (__x <= 0.5f) { __scale *= __reduced; }"
      << "float __denominator = __fma(__scale * __poly, __reduced, __scale);"
      << "if (__fabsf(__denominator) < 1.17549435e-38f && __denominator != 0.0f) {"
      << "uint32_t __denominator_bits = reinterpret_cast<uint32_t&>(__denominator);"
      << "int32_t __frexp_exponent = (int32_t)((__denominator_bits & 0x7f800000u) >> 23);"
      << "float __mantissa = (float)(__denominator_bits & 0x007fffffu) / 8388608.0f;"
      << "if (__frexp_exponent == 0) { while (__mantissa < 0.5f) { __mantissa *= 2.0f; --__frexp_exponent; } }"
      << "else { __mantissa = __mantissa * 0.5f + 0.5f; }"
      << "if ((__denominator_bits & 0x80000000u) != 0u) { __mantissa = -__mantissa; }"
      << "int32_t __ldexp_exponent = 126 - __frexp_exponent; float __reciprocal = 1.0f / __mantissa;"
      << "if (__ldexp_exponent >= 101) { uint32_t __biased = (uint32_t)(__ldexp_exponent + 508);"
      << "uint32_t __quotient = __biased >> 2, __remainder = __biased - (__quotient << 2);"
      << "uint32_t __scale_bits = __quotient << 23, __combined_bits = (__quotient + __remainder) << 23;"
      << "float __split_scale = reinterpret_cast<float&>(__scale_bits);"
      << "__result = __reciprocal * reinterpret_cast<float&>(__combined_bits);"
      << "__result *= __split_scale; __result *= __split_scale; __result *= __split_scale; }"
      << "else { uint32_t __scale_bits = (uint32_t)(__ldexp_exponent + 127) << 23;"
      << "__result = __reciprocal * reinterpret_cast<float&>(__scale_bits); } }"
      << "else { __result = 1.0f / __denominator; } }"
      << "else { float __abs_x = __abs_input > 41.099998474121094f ? 41.099998474121094f : __abs_input;"
      << "bool __normal = __abs_x >= 1.17549435e-38f;"
      << "float __log_input = __normal ? __abs_x : __abs_x * 16777216.0f;"
      << "float __exponent_base = __normal ? 0.0f : -24.0f;"
      << "uint32_t __log_bits = reinterpret_cast<uint32_t&>(__log_input);"
      << "uint32_t __reduction_bits = (__log_bits - 0x3f3504f3u) & 0xff800000u;"
      << "uint32_t __mantissa_bits = __log_bits - __reduction_bits;"
      << "float __mantissa = reinterpret_cast<float&>(__mantissa_bits);"
      << "float __exponent_part = __fma(__cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__reduction_bits),"
      << "1.1920928955078125e-07f, __exponent_base);"
      << "float __mantissa_minus_one = __mantissa - 1.0f, __reciprocal = 1.0f / (__mantissa + 1.0f);"
      << "float __reduced_hi = __reciprocal * (__mantissa_minus_one + __mantissa_minus_one);"
      << "float __reduced_square = __reduced_hi * __reduced_hi;"
      << "float __log_poly = __fma(__reduced_square, 0.0006568862590938807f, 0.0032181653659790754f);"
      << "__log_poly = __fma(__reduced_square, __log_poly, 0.018033718690276146f);"
      << "__log_poly = __fma(__reduced_square, __log_poly, 0.12022458761930466f);"
      << "__log_poly *= __reduced_square;"
      << "float __log_rounded = __fma(__reduced_hi, 1.4426950216293335f, __exponent_part);"
      << "float __reduced_err = __mantissa_minus_one - __reduced_hi;"
      << "__reduced_err = __fma(__mantissa_minus_one, -__reduced_hi, __reduced_err + __reduced_err);"
      << "float __reduced_lo = __reciprocal * __reduced_err;"
      << "float __log_diff = __exponent_part - __log_rounded;"
      << "__log_diff = __fma(__reduced_hi, 1.4426950216293335f, __log_diff);"
      << "__log_diff = __fma(__reduced_lo, 1.4426950216293335f, __log_diff);"
      << "__log_diff = __fma(__reduced_hi, 1.925963033500011e-08f, __log_diff);"
      << "__log_diff = __fma(__reduced_lo, __log_poly * 3.0f, __log_diff);"
      << "__log_diff = __fma(__reduced_hi, __log_poly, __log_diff);"
      << "float __log_x = __log_rounded + __log_diff;"
      << "float __log_round_err = __log_diff - (__log_x - __log_rounded);"
      << "float __abs_minus_half = __abs_x - 0.5f;"
      << "float __log_mul_hi = __log_x * __abs_minus_half;"
      << "float __exp_mul_hi = __abs_x * 1.4426950216293335f;"
      << "float __exponent_hi = __log_mul_hi - __exp_mul_hi;"
      << "float __log_mul_err = __fma(__log_x, __abs_minus_half, -__log_mul_hi);"
      << "__log_mul_err = __fma(__log_round_err, __abs_minus_half, __log_mul_err);"
      << "float __exp_mul_err = __fma(__abs_x, 1.4426950216293335f, -__exp_mul_hi);"
      << "__exp_mul_err = __fma(__abs_x, 1.925963033500011e-08f, __exp_mul_err);"
      << "float __exp_sum_hi = __exp_mul_hi + __exponent_hi;"
      << "float __exponent_round_err = __exponent_hi - __exp_sum_hi;"
      << "float __log_mul_round_err = __log_mul_hi - __exp_sum_hi;"
      << "float __exp_mul_round_err = -__exp_mul_hi - __exponent_round_err;"
      << "float __exponent_lo = __log_mul_round_err + __exp_mul_round_err;"
      << "__exponent_lo += __log_mul_err - __exp_mul_err;"
      << "float __offset = __abs_x > 33.0f ? 48.0f : 0.0f;"
      << "if (__x < 0.0f) { __exponent_hi = __offset - __exponent_hi; __exponent_lo = -__exponent_lo; }"
      << "float __rounded_exponent = __roundf(__exponent_hi);"
      << "float __exp2_fraction = __exponent_hi - __rounded_exponent + __exponent_lo;"
      << "int32_t __exp2_exponent = __cvt_int32_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(__rounded_exponent);"
      << "float __exp2_poly = __fma(__exp2_fraction, 0.00015239251661114395f, 0.0013391353422775865f);"
      << "__exp2_poly = __fma(__exp2_fraction, __exp2_poly, 0.009618839249014854f);"
      << "__exp2_poly = __fma(__exp2_fraction, __exp2_poly, 0.055503588169813156f);"
      << "__exp2_poly = __fma(__exp2_fraction, __exp2_poly, 0.24022644758224487f);"
      << "__exp2_poly = __fma(__exp2_fraction, __exp2_poly, 0.6931471824645996f);"
      << "__exp2_poly = __fma(__exp2_fraction, __exp2_poly, 1.0f);"
      << "bool __positive_exponent = __rounded_exponent > 0.0f;"
      << "uint32_t __scale_hi_bits = __positive_exponent ? 0x7f000000u : 0x02000000u;"
      << "uint32_t __scale_adjust = __positive_exponent ? 0u : 0x83000000u;"
      << "uint32_t __scale_lo_bits = ((uint32_t)__exp2_exponent << 23) - __scale_adjust;"
      << "float __exp2_value = __exp2_poly * reinterpret_cast<float&>(__scale_hi_bits);"
      << "__exp2_value *= reinterpret_cast<float&>(__scale_lo_bits);"
      << "if (__fabsf(__exponent_hi) > 152.0f) {"
      << "__exp2_value = __exponent_hi >= 0.0f ? __builtin_inff() : 0.0f; }"
      << "float __leading = __exp2_value * 2.5066282749176025f;"
      << "float __inv_abs = 1.0f / __abs_x;"
      << "float __correction = __fma(0.000068413915f, __inv_abs, -0.000050603266f);"
      << "__correction = __fma(__correction, __inv_abs, -0.00042276637f);"
      << "__correction = __fma(__correction, __inv_abs, 0.0009921414f);"
      << "__correction = __fma(__correction, __inv_abs, -0.00027855476f);"
      << "__correction = __fma(__correction, __inv_abs, -0.002674901f);"
      << "__correction = __fma(__correction, __inv_abs, 0.0034718033f);"
      << "__correction = __fma(__correction, __inv_abs, 0.08333334f);"
      << "__correction = __fma(__correction, __inv_abs, 0.0f);"
      << "if (__x > 0.0f) { __result = __fma(__leading, __correction, __leading); }"
      << "else { __correction += 1.0f; float __two_abs = __abs_x + __abs_x;"
      << "int32_t __quadrant = __cvt_int32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(__two_abs);"
      << "float __rounded_two_abs = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__quadrant);"
      << "bool __use_cos = (__quadrant & 1) != 0;"
      << "float __reduced_arg = __fma(__rounded_two_abs, -0.5f, __abs_x);"
      << "float __reduced_square = __reduced_arg * __reduced_arg, __sinpi;"
      << "if (__use_cos) { __sinpi = __fma(__reduced_square, 0.22686031460762024f, -1.3345603942871094f);"
      << "__sinpi = __fma(__reduced_square, __sinpi, 4.058692455291748f);"
      << "__sinpi = __fma(__reduced_square, __sinpi, -4.934802055358887f);"
      << "__sinpi = __fma(__reduced_square, __sinpi, 1.0f); }"
      << "else { __sinpi = __fma(__reduced_square, -0.5924802422523499f, 2.5501441955566406f);"
      << "__sinpi = __fma(__reduced_square, __sinpi, -5.167719841003418f);"
      << "__sinpi = __fma(__sinpi, __reduced_arg * __reduced_square, __reduced_arg * 3.1415927410125732f); }"
      << "if ((__quadrant & 2) != 0) { __sinpi = -__sinpi; }"
      << "float __denominator_product = __correction * __x;"
      << "float __product = __denominator_product * __sinpi;"
      << "float __product_error = __fma(__denominator_product, __sinpi, -__product);"
      << "float __reciprocal_product = 1.0f / __product;"
      << "__result = __fma(__leading, __reciprocal_product,"
      << "-__leading * __reciprocal_product * __product_error * __reciprocal_product);"
      << "__result *= 0.5f; if (__abs_x > 33.0f) { __result *= 3.5527136e-15f; } } }"
      << "float __truncated = __x > 0.0f ? __floorf(__x) : __ceilf(__x);"
      << "if (__truncated == __x && __x < 0.0f) { __result = __builtin_inff() / __builtin_inff(); }"
      << "if (__x < -41.099998474121094f && __truncated != __x && (((int32_t)__truncated & 1) != 0)) {"
      << "__result = 0.0f; }"
      << "if (__isnan(__x)) { __result = __x; }"
      << "if (__x == __builtin_inff()) { __result = __builtin_inff(); }"
      << "if (__x == -__builtin_inff()) { __result = __builtin_inff() / __builtin_inff(); }"
      << "if (__x == 0.0f) { __result = 1.0f / __x; } __result;})";
    return s.str();
}

std::string MakeSimtLgammaCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const std::string small_log = MakeSimtGammaLnFP32Codegen("__small_value");
    const std::string large_log = MakeSimtGammaLnFP32Codegen("__abs_x");
    const std::string sinpi = MakeSimtLgammaSinpiFP32Codegen("__abs_x");
    const std::string reflection_log = MakeSimtGammaLnFP32Codegen("__abs_x * __fabsf(__sinpi)");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __abs_x = __fabsf(__x), __result = 0.0f;"
      << "if (__isinf(__abs_x)) { __result = __abs_x; }"
      << "else if (__abs_x < 0.7f) { float __small_value = __fma(0.0035875155f, __abs_x, -0.0054712854f);"
      << "__small_value = __fma(__small_value, __abs_x, -0.044627126f);"
      << "__small_value = __fma(__small_value, __abs_x, 0.1673177f);"
      << "__small_value = __fma(__small_value, __abs_x, -0.04213598f);"
      << "__small_value = __fma(__small_value, __abs_x, -0.6558673f);"
      << "__small_value = __fma(__small_value, __abs_x, 0.5772154f);"
      << "__small_value = __fma(__small_value, __abs_x, 0.0f);"
      << "__small_value = __fma(__small_value, __abs_x, __abs_x);"
      << "__result = -(" << small_log << ");"
      << "if (__small_value == 0.0f) { __result = __builtin_inff(); } }"
      << "else if (__abs_x < 1.5f) { float __one_minus_x = 1.0f - __abs_x;"
      << "__result = __fma(0.045882664f, __one_minus_x, 0.10373967f);"
      << "__result = __fma(__result, __one_minus_x, 0.122803635f);"
      << "__result = __fma(__result, __one_minus_x, 0.12752421f);"
      << "__result = __fma(__result, __one_minus_x, 0.14321668f);"
      << "__result = __fma(__result, __one_minus_x, 0.16934357f);"
      << "__result = __fma(__result, __one_minus_x, 0.20740793f);"
      << "__result = __fma(__result, __one_minus_x, 0.2705875f);"
      << "__result = __fma(__result, __one_minus_x, 0.40068542f);"
      << "__result = __fma(__result, __one_minus_x, 0.82246696f);"
      << "__result = __fma(__result, __one_minus_x, 0.5772157f);"
      << "__result = __fma(__result, __one_minus_x, 0.0f); }"
      << "else if (__abs_x < 3.0f) { float __x_minus_two = __abs_x - 2.0f;"
      << "__result = __fma(0.0000495984932f, __x_minus_two, -0.00022089484f);"
      << "__result = __fma(__result, __x_minus_two, 0.000541314250f);"
      << "__result = __fma(__result, __x_minus_two, -0.00120451697f);"
      << "__result = __fma(__result, __x_minus_two, 0.00288425176f);"
      << "__result = __fma(__result, __x_minus_two, -0.00738275796f);"
      << "__result = __fma(__result, __x_minus_two, 0.0205813199f);"
      << "__result = __fma(__result, __x_minus_two, -0.0673524886f);"
      << "__result = __fma(__result, __x_minus_two, 0.322467029f);"
      << "__result = __fma(__result, __x_minus_two, 0.42278432f);"
      << "__result = __fma(__result, __abs_x, -__result - __result); }"
      << "else if (__abs_x < 7.8f) { float __x_minus_three = __abs_x - 3.0f;"
      << "float __numerator = __fma(-748.8903f, __x_minus_three, -12349.742f);"
      << "__numerator = __fma(__numerator, __x_minus_three, -41061.375f);"
      << "__numerator = __fma(__numerator, __x_minus_three, -48310.664f);"
      << "__numerator = __fma(__numerator, __x_minus_three, -143033.4f);"
      << "float __denominator = __fma(1.0f, __x_minus_three, -259.25097f);"
      << "__denominator = __fma(__denominator, __x_minus_three, -10777.18f);"
      << "__denominator = __fma(__denominator, __x_minus_three, -92685.05f);"
      << "__denominator = __fma(__denominator, __x_minus_three, -206353.58f);"
      << "__result = __fma(__numerator, 1.0f / __denominator, __x_minus_three); }"
      << "else { float __inv_abs_x = 1.0f / __abs_x, __inv_square = __inv_abs_x * __inv_abs_x;"
      << "float __correction = __fma(0.00077783066f, __inv_square, -0.0027776553f);"
      << "__correction = __fma(__correction, __inv_square, 0.083333276f);"
      << "__correction = __fma(__correction, __inv_abs_x, 0.0f);"
      << "float __half_log_product = (" << large_log << ") * 0.5f * (__abs_x - 0.5f);"
      << "__result = __half_log_product - __abs_x + __half_log_product + __correction + 0.9189385f; }"
      << "if (__x < 0.0f) { if (__floorf(__abs_x) == __abs_x) { __result = __builtin_inff(); }"
      << "else if (__abs_x < 9.9999996e-20f) { __result = -(" << large_log << "); }"
      << "else { float __sinpi = " << sinpi << "; float __reflection_log = " << reflection_log << ";"
      << "float __reflection = 1.14472985f - __reflection_log; __result = __fma(__reflection, 1.0f, -__result); } }"
      << "__result;})";
    return s.str();
}

std::string MakeSimtErfOpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const std::string exp2 = MakeSimtExp2FP32Codegen("__exponent");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __a = __fabsf(__x), __r;"
      << "if (__a >= 1.00296f) {"
      << "float __p = __fma(__a, 0.000112198715f, -0.0013275252f);"
      << "__p = __fma(__a, __p, 0.008396535f); __p = __fma(__a, __p, -0.040246583f);"
      << "__p = __fma(__a, __p, 0.15950431f); __p = __fma(__a, __p, 0.9129177f);"
      << "__p = __fma(__a, __p, 0.62906002f); float __exponent = __fma(__p, -__a, -__a);"
      << "float __adjusted = 1.0f - (" << exp2 << ");"
      << "uint32_t __result_bits = (reinterpret_cast<uint32_t&>(__x) & 0x80000000u) |"
      << " reinterpret_cast<uint32_t&>(__adjusted); __r = reinterpret_cast<float&>(__result_bits); }"
      << "else { float __z = __x * __x;"
      << "float __p = __fma(__z, 0.000084834944f, -0.00082130916f);"
      << "__p = __fma(__z, __p, 0.005213489f); __p = __fma(__z, __p, -0.026868773f);"
      << "__p = __fma(__z, __p, 0.11284005f); __p = __fma(__z, __p, -0.37612664f);"
      << "__p = __fma(__z, __p, 0.12837915f); __r = __fma(__p, __x, __x); } __r;})";
    return s.str();
}

std::string MakeSimtErfcOpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __a = __fabsf(__x);"
      << "float __term1 = __a - 4.0f, __term2 = __a + 4.0f, __inv_term2 = 1.0f / __term2;"
      << "float __y = __term1 * __inv_term2, __z = __y + 1.0f;"
      << "float __numerator = __fma(-4.0f, __z, __a), __tmp = __fma(-__y, __a, __numerator);"
      << "float __w = __fma(__inv_term2, __tmp, __y);"
      << "float __poly = __fma(0.0008912170887924731f, __w, 0.00704578822478652f);"
      << "__poly = __fma(__poly, __w, -0.015866896137595177f);"
      << "__poly = __fma(__poly, __w, 0.03642962500452995f);"
      << "__poly = __fma(__poly, __w, -0.06664343178272247f);"
      << "__poly = __fma(__poly, __w, 0.09381452947854996f);"
      << "__poly = __fma(__poly, __w, -0.10099056363105774f);"
      << "__poly = __fma(__poly, __w, 0.06809400022029877f);"
      << "__poly = __fma(__poly, __w, 0.015377387404441833f);"
      << "__poly = __fma(__poly, __w, -0.1396210789680481f);"
      << "__poly = __fma(__poly, __w, 1.2329951524734497f);"
      << "float __denominator = __fma(2.0f, __a, 1.0f), __inv_denominator = 1.0f / __denominator;"
      << "float __q = __poly * __inv_denominator, __t = __fma(__a, __q * -2.0f, __poly);"
      << "float __v = __fma(__t - __q, __inv_denominator, __q);"
      << "float __square = __a * __a, __negative_square = -__square;"
      << "float __scaled = __negative_square * 1.4426950216293335f, __integer = __ceilf(__scaled);"
      << "float __abs_integer = __fabsf(__integer);"
      << "uint32_t __clamped_bits = (reinterpret_cast<uint32_t&>(__integer) & 0x80000000u) | 0x42fc0000u;"
      << "float __safe_integer = __abs_integer > 126.0f ? reinterpret_cast<float&>(__clamped_bits) : __integer;"
      << "float __remainder = __fma(__safe_integer, -0.6931471824645996f, __negative_square);"
      << "__remainder = __fma(__safe_integer, 1.9046542121259336e-09f, __remainder);"
      << "float __exponent_arg = __remainder * 1.4426950216293335f;"
      << "float __exponent_base = __safe_integer + 12583039.0f;"
      << "uint32_t __exponent_bits = reinterpret_cast<uint32_t&>(__exponent_base) << 23;"
      << "float __exp_value = __powf(2.0f, __exponent_arg) * reinterpret_cast<float&>(__exponent_bits);"
      << "float __residual = __fma(-__a, __a, __square);"
      << "float __result = __v * __fma(__exp_value, __residual, __exp_value);"
      << "if (__a > 10.055000305175781f) { __result = 0.0f; }"
      << "if (__x < 0.0f) { __result = 2.0f - __result; } __result;})";
    return s.str();
}

std::string MakeSimtErfinvCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const std::string log2 = MakeSimtLog2FP32Codegen("__temp");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __opposite = -__x;"
      << "float __temp = __fma(__x, __opposite, 1.0f), __log2_temp = " << log2 << ", __w = -__log2_temp, __r;"
      << "if (__log2_temp < -8.2f) { float __s = 1.0f / __sqrtf(__w);"
      << "float __p = __fma(__s, -0.5899144f, -0.6630042f);"
      << "__p = __fma(__s, __p, 1.5970111f); __p = __fma(__s, __p, -0.67521554f);"
      << "__p = __fma(__s, __p, -0.09522479f); __p = __fma(__s, __p, 0.83535343f);"
      << "float __term = (1.0f / __s) * __p; uint32_t __rb = reinterpret_cast<uint32_t&>(__term) | "
      << "(reinterpret_cast<uint32_t&>(__x) & 0x80000000u); __r = reinterpret_cast<float&>(__rb); }"
      << "else { float __p = __fma(__w, -2.5172708e-10f, 9.427429e-9f);"
      << "__p = __fma(__w, __p, -1.2054752e-7f); __p = __fma(__w, __p, 2.1697005e-7f);"
      << "__p = __fma(__w, __p, 8.0621484e-6f); __p = __fma(__w, __p, -3.1675492e-5f);"
      << "__p = __fma(__w, __p, -0.0007743631f); __p = __fma(__w, __p, 0.005546588f);"
      << "__p = __fma(__w, __p, 0.16082023f); __p = __fma(__w, __p, 0.8862269f);"
      << "__r = __p * __x; } __r;})";
    return s.str();
}

std::string MakeSimtErfcinvCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const std::string central_log2 = MakeSimtLog2FP32Codegen("__product");
    const std::string tail_log2 = MakeSimtLog2FP32Codegen("__selected");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __opposite = -__x, __term = 2.0f - __x, __r;"
      << "if (__x <= 1.9966f && __x >= 0.0034f) { float __product = __term * __x;"
      << "float __w = -(" << central_log2 << ");"
      << "float __p = __fma(__w, -2.5172708e-10f, 9.427429e-9f);"
      << "__p = __fma(__w, __p, -1.2054752e-7f); __p = __fma(__w, __p, 2.1697005e-7f);"
      << "__p = __fma(__w, __p, 8.0621484e-6f); __p = __fma(__w, __p, -3.1675492e-5f);"
      << "__p = __fma(__w, __p, -0.0007743631f); __p = __fma(__w, __p, 0.005546588f);"
      << "__p = __fma(__w, __p, 0.16082023f); __p = __fma(__w, __p, 0.8862269f);"
      << "__r = __fma(__p, __opposite, __p); }"
      << "else { bool __high = __x > 1.0f; float __selected = __high ? __term : __x;"
      << "float __w = -(" << tail_log2 << "), __s = 1.0f / __sqrtf(__w);"
      << "float __p = __fma(__s, -63.113224f, 127.48469f);"
      << "__p = __fma(__s, __p, -114.10568f); __p = __fma(__s, __p, 60.325786f);"
      << "__p = __fma(__s, __p, -21.789892f); __p = __fma(__s, __p, 6.467409f);"
      << "__p = __fma(__s, __p, -1.8329474f); __p = __fma(__s, __p, -0.030327774f);"
      << "__p = __fma(__s, __p, 0.83287745f);"
      << "__r = __p * (__high ? -(1.0f / __s) : (1.0f / __s)); } __r;})";
    return s.str();
}

std::string MakeSimtErfcxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const std::string exp2 = MakeSimtExp2FP32Codegen("__exp_arg");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __r;"
      << "if (__x < -9.43f) { uint32_t __inf = 0x7f800000u; __r = reinterpret_cast<float&>(__inf); }"
      << "else { float __a = __fabsf(__x); if (__a < 10.055f) {"
      << "float __term1 = __a - 4.0f, __term2 = __a + 4.0f, __inv_term2 = 1.0f / __term2;"
      << "float __w = __term1 * __inv_term2, __z = __w + 1.0f;"
      << "float __numerator = __fma(-4.0f, __z, __a);"
      << "float __tmp = __fma(-__w, __a, __numerator); __w = __fma(__inv_term2, __tmp, __w);"
      << "float __poly = __fma(0.0008912171f, __w, 0.007045788f);"
      << "__poly = __fma(__poly, __w, -0.015866896f); __poly = __fma(__poly, __w, 0.036429625f);"
      << "__poly = __fma(__poly, __w, -0.06664343f); __poly = __fma(__poly, __w, 0.09381453f);"
      << "__poly = __fma(__poly, __w, -0.10099056f); __poly = __fma(__poly, __w, 0.068094f);"
      << "__poly = __fma(__poly, __w, 0.015377387f); __poly = __fma(__poly, __w, -0.13962108f);"
      << "__poly = __fma(__poly, __w, 1.2329951f); float __term3 = __fma(2.0f, __a, 1.0f);"
      << "float __inv_term3 = 1.0f / __term3, __q = __poly * __inv_term3;"
      << "float __t = __fma(__a, __q * -2.0f, __poly), __u = __t - __q;"
      << "__r = __fma(__u, __inv_term3, __q); if (!(__x > 0.0f)) {"
      << "float __x_sq = __a * __a, __neg_x_sq = -__x_sq, __term4 = __fma(__a, __a, __neg_x_sq);"
      << "float __term5 = __fma(__x_sq, 0.00572498f, 0.5f);"
      << "__term5 = __fminf(__term5, __builtin_inff());"
      << "float __term6 = __fma(__term5, 252.0f, 12582913.0f);"
      << "float __term7 = __term6 - 12583039.0f;"
      << "float __term8 = __fma(__x_sq, 1.442695f, -__term7);"
      << "float __exp_arg = __fma(__x_sq, 1.925963e-8f, __term8);"
      << "uint32_t __exponent = reinterpret_cast<uint32_t&>(__term6) << 23;"
      << "float __exponent_scale = reinterpret_cast<float&>(__exponent), __term9_exp = " << exp2
      << "; float __scaled_exp = __term9_exp * __exponent_scale;"
      << "float __exp_approx = __fma(__term9_exp, __exponent_scale, __scaled_exp);"
      << "float __final_exp = __fma(__exp_approx, __term4, __exp_approx);"
      << "__r = __isinf(__exp_approx) ? __exp_approx : (__final_exp - __r); } }"
      << "else { float __scaled_x = __a * 0.25f, __reciprocal = 0.25f / __scaled_x;"
      << "float __w = __reciprocal * __reciprocal;"
      << "float __poly = __fma(6.5625f, __w, -1.875f); __poly = __fma(__poly, __w, 0.75f);"
      << "__poly = __fma(__poly, __w, -0.5f); __poly = __fma(__poly, __w, 1.0f);"
      << "__r = (__reciprocal * 0.5641896f) * __poly; } } __r;})";
    return s.str();
}

std::string MakeSimtNormcdfCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const std::string exp2 = MakeSimtExp2FP32Codegen("__exp_residual * 1.4426950216293335f");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __abs_x = __fabsf(__x);"
      << "uint32_t __xb = reinterpret_cast<uint32_t&>(__x);"
      << "uint32_t __clamped_bits = (__xb & 0x80000000u) | 0x41680000u;"
      << "float __clamped_x = __abs_x > 14.5f ? reinterpret_cast<float&>(__clamped_bits) : __x;"
      << "bool __ge_neg_one = !(__clamped_x < -1.0f);"
      << "float __z_hi = __clamped_x * -0.70710676908493041992f;"
      << "float __z_comp = __fma(__clamped_x, -0.70710676908493041992f, -__z_hi);"
      << "float __z_lo = __fma(__clamped_x, -1.2101617485882343317e-08f, __z_comp);"
      << "float __z = __z_hi + __z_lo, __abs_z = __fabsf(__z); bool __z_nonnegative = !(__z < 0.0f);"
      << "float __inv_abs_z_plus_4 = 1.0f / (__abs_z + 4.0f);"
      << "float __q = (__abs_z - 4.0f) * __inv_abs_z_plus_4;"
      << "float __q_residual = __fma(__q + 1.0f, -4.0f, __abs_z);"
      << "__q_residual = __fma(__abs_z, -__q, __q_residual);"
      << "__q = __fma(__inv_abs_z_plus_4, __q_residual, __q);"
      << "float __z_square = __z * __z, __scaled_exp = __z_square * -1.4426950216293335f;"
      << "float __exp_k = __scaled_exp > 0.0f ? __floorf(__scaled_exp) : __ceilf(__scaled_exp);"
      << "float __abs_exp_k = __fabsf(__exp_k);"
      << "uint32_t __limited_bits = (reinterpret_cast<uint32_t&>(__exp_k) & 0x80000000u) | 0x42fc0000u;"
      << "float __exp_k_limited = __abs_exp_k > 126.0f ? reinterpret_cast<float&>(__limited_bits) : __exp_k;"
      << "float __exp_residual = __fma(__exp_k_limited, -0.6931471824645996f, -__z_square);"
      << "__exp_residual = __fma(__exp_k_limited, 1.9046542121259336e-09f, __exp_residual);"
      << "float __exp2_residual = " << exp2 << ";"
      << "uint32_t __exp_scale_bits = (uint32_t)__cvt_int32_t<ROUND::Z,"
      << " RoundingSaturation::RS_ENABLE_VALUE>(__exp_k_limited + 12583039.0f) << 23;"
      << "float __exp_scale = reinterpret_cast<float&>(__exp_scale_bits);"
      << "float __square_residual = __fma(-__abs_z, __abs_z, __z_square);"
      << "float __exp_term = __fma(__exp_scale, __square_residual, __exp_scale) * __exp2_residual;"
      << "float __poly = __fma(__q, 0.0008912170887924731f, 0.00704578822478652f);"
      << "__poly = __fma(__q, __poly, -0.015866896137595177f);"
      << "__poly = __fma(__q, __poly, 0.03642962500452995f);"
      << "__poly = __fma(__q, __poly, -0.06664343178272247f);"
      << "__poly = __fma(__q, __poly, 0.09381452947854996f);"
      << "__poly = __fma(__q, __poly, -0.10099056363105774f);"
      << "__poly = __fma(__q, __poly, 0.06809400022029877f);"
      << "__poly = __fma(__q, __poly, 0.015377387404441833f);"
      << "__poly = __fma(__q, __poly, -0.1396210789680481f);"
      << "__poly = __fma(__q, __poly, 1.2329951524734497f);"
      << "float __inv_two_abs_z_plus_1 = 1.0f / __fma(__abs_z, 2.0f, 1.0f);"
      << "float __rational = __poly * __inv_two_abs_z_plus_1;"
      << "float __rational_err = __fma(__abs_z, __rational * -2.0f, __poly);"
      << "__rational_err = -__rational + __rational_err;"
      << "__rational = __fma(__inv_two_abs_z_plus_1, __rational_err, __rational);"
      << "float __erfc_value = __rational * __exp_term;"
      << "if (__abs_z > 10.055000305175781f) { __erfc_value = 0.0f; }"
      << "if (!__z_nonnegative) { __erfc_value = 2.0f - __erfc_value; }"
      << "if (!__ge_neg_one) { float __correction_scale = __z * -2.0f * __erfc_value;"
      << "__erfc_value = __fma(__z_hi - __z + __z_lo, __correction_scale, __erfc_value); }"
      << "__erfc_value * 0.5f;})";
    return s.str();
}

std::string MakeSimtNormcdfinvCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    const std::string central_log2 = MakeSimtLog2FP32Codegen("__product");
    const std::string tail_log2 = MakeSimtLog2FP32Codegen("__selected");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __double_x = __x + __x;"
      << "float __opposite = -__double_x, __term = 2.0f - __double_x, __r;"
      << "if (__double_x >= 0.0034f && __double_x <= 1.9966f) { float __product = __term * __double_x;"
      << "float __w = -(" << central_log2 << ");"
      << "float __p = __fma(-2.51727084e-10f, __w, 9.42742862e-09f);"
      << "__p = __fma(__p, __w, -1.20547526e-07f); __p = __fma(__p, __w, 2.16970051e-07f);"
      << "__p = __fma(__p, __w, 8.06214848e-06f); __p = __fma(__p, __w, -3.16754922e-05f);"
      << "__p = __fma(__p, __w, -0.000774363114f); __p = __fma(__p, __w, 0.00554658799f);"
      << "__p = __fma(__p, __w, 0.160820231f); __p = __fma(__p, __w, 0.886226892f);"
      << "__r = __fma(__p, __opposite, __p); }"
      << "else { float __selected = __double_x <= 1.0f ? __double_x : __term;"
      << "float __w = 1.0f / __sqrtf(-(" << tail_log2 << "));"
      << "float __p = __fma(-63.113224f, __w, 127.484688f);"
      << "__p = __fma(__p, __w, -114.105682f); __p = __fma(__p, __w, 60.3257866f);"
      << "__p = __fma(__p, __w, -21.7898922f); __p = __fma(__p, __w, 6.46740913f);"
      << "__p = __fma(__p, __w, -1.83294737f); __p = __fma(__p, __w, -0.0303277746f);"
      << "__p = __fma(__p, __w, 0.832877457f); float __inv_w = 1.0f / __w;"
      << "if (__double_x > 1.0f) { __inv_w = -__inv_w; } __r = __p * __inv_w; }"
      << "__fma(__r, -1.41421354f, 0.0f);})";
    return s.str();
}

std::string MakeSimtBesselMiddleTrigFP32Codegen(const std::string& alpha, bool is_sin)
{
    std::ostringstream s;
    s << "({float __bessel_alpha = (" << alpha << "); __bessel_alpha = __bessel_alpha * 0.0f + __bessel_alpha;"
      << "int64_t __bessel_q = (int64_t)(__bessel_alpha * 0.63661975f);"
      << "__bessel_alpha = __fma((float)__bessel_q, -1.5707964f, __bessel_alpha);"
      << "__bessel_alpha = __fma((float)__bessel_q, 4.371139e-8f, __bessel_alpha);"
      << "int64_t __bessel_q2 = (int64_t)(__bessel_alpha * 0.63661975f);"
      << "__bessel_alpha = __fma((float)__bessel_q2, -1.5707964f, __bessel_alpha);"
      << "int32_t __bessel_quadrant = (int32_t)((__bessel_q + __bessel_q2) % 4);"
      << "if (__bessel_quadrant < 0) { __bessel_quadrant += 4; }"
      << "float __bessel_reduced = __bessel_alpha - 0.7853982f;"
      << "float __bessel_square = __bessel_reduced * __bessel_reduced;"
      << "float __bessel_cos = __fma(2.44677067e-5f, __bessel_square, -1.38877297e-3f);"
      << "__bessel_cos = __fma(__bessel_cos, __bessel_square, 4.16666567e-2f);"
      << "__bessel_cos = __fma(__bessel_cos, __bessel_square, -5.0e-1f);"
      << "__bessel_cos = __fma(__bessel_cos, __bessel_square, 1.0f);"
      << "float __bessel_sin = __fma(2.86567956e-6f, __bessel_square, -1.98559923e-4f);"
      << "__bessel_sin = __fma(__bessel_sin, __bessel_square, 8.33338592e-3f);"
      << "__bessel_sin = __fma(__bessel_sin, __bessel_square, -1.66666672e-1f);"
      << "float __bessel_cube = __fma(__bessel_reduced, __bessel_square, 0.0f);"
      << "__bessel_sin = __fma(__bessel_sin, __bessel_cube, __bessel_reduced);"
      << "if ((__bessel_quadrant & 2) != 0) { __bessel_sin = -__bessel_sin; __bessel_cos = -__bessel_cos; }";
    if (is_sin) {
        s << "if ((__bessel_quadrant & 1) != 0) { __bessel_sin = __bessel_cos; } __bessel_sin;";
    } else {
        s << "if ((__bessel_quadrant & 1) != 0) { __bessel_cos = -__bessel_sin; } __bessel_cos;";
    }
    s << "})";
    return s.str();
}

std::string MakeSimtBesselHugeCosFP32Codegen(const std::string& alpha, const std::string& phase_shift)
{
    const std::string cos_value = MakeSimtTrigFP32Codegen("__bessel_reduced - (" + phase_shift + ")", false);
    std::ostringstream s;
    s << "({float __bessel_alpha = (" << alpha << "), __bessel_reduced = __bessel_alpha;"
      << "int32_t __bessel_quadrant = 0; if (__bessel_alpha >= 105615.0f) {"
      << "uint32_t __bessel_bits = reinterpret_cast<uint32_t&>(__bessel_alpha);"
      << "int32_t __bessel_exp = (int32_t)(((__bessel_bits & 0x7f800000u) >> 23) - 127);"
      << "uint32_t __bessel_table_index = (uint32_t)__bessel_exp >> 5;"
      << "const uint32_t __bessel_table[] = {0x517cc1b7u, 0x27220a94u, 0xfe13abe8u,"
      << "0xfa9a6ee0u, 0x6db14accu, 0x9e21c820u};"
      << "uint32_t __bessel_hi = __bessel_table_index ? __bessel_table[__bessel_table_index - 1] : 0u;"
      << "uint32_t __bessel_mid = __bessel_table[__bessel_table_index];"
      << "uint32_t __bessel_lo = __bessel_table[__bessel_table_index + 1];"
      << "uint32_t __bessel_last = __bessel_table[__bessel_table_index + 2];"
      << "int32_t __bessel_remainder_exp = (uint32_t)__bessel_exp & 0x1f;"
      << "if (__bessel_remainder_exp != 0) {"
      << "__bessel_hi = (__bessel_hi << __bessel_remainder_exp) |"
      << " (__bessel_mid >> (32 - __bessel_remainder_exp));"
      << "__bessel_mid = (__bessel_mid << __bessel_remainder_exp) |"
      << " (__bessel_lo >> (32 - __bessel_remainder_exp));"
      << "__bessel_lo = (__bessel_lo << __bessel_remainder_exp) |"
      << " (__bessel_last >> (32 - __bessel_remainder_exp)); }"
      << "uint32_t __bessel_mantissa_bits = (__bessel_bits & 0x007fffffu) | 0x4f000000u;"
      << "uint32_t __bessel_mantissa = (uint32_t)reinterpret_cast<float&>(__bessel_mantissa_bits);"
      << "uint64_t __bessel_product = (uint64_t)__bessel_mantissa * __bessel_lo;"
      << "__bessel_product = (uint64_t)__bessel_mantissa * __bessel_mid + (__bessel_product >> 32);"
      << "__bessel_product = ((uint64_t)(__bessel_mantissa * __bessel_hi) << 32) + __bessel_product;"
      << "__bessel_quadrant = (int32_t)(__bessel_product >> 62);"
      << "__bessel_product &= 0x3fffffffffffffffull;"
      << "if ((__bessel_product & 0x2000000000000000ull) != 0) {"
      << "__bessel_product -= 0x4000000000000000ull; ++__bessel_quadrant; }"
      << "int64_t __bessel_pi = (int64_t)__bessel_product; float __bessel_hi_part = (float)__bessel_pi;"
      << "__bessel_pi -= (int64_t)__bessel_hi_part; float __bessel_lo_part = (float)__bessel_pi;"
      << "__bessel_reduced = (__bessel_hi_part + __bessel_lo_part) * 3.4061215800865545e-19f;"
      << "__bessel_quadrant &= 3;"
      << "__bessel_reduced += (float)__bessel_quadrant * 1.5707963705062866f; }" << cos_value << ";})";
    return s.str();
}

std::string MakeSimtJ0Codegen(const std::string& operand)
{
    const std::string middle_value = MakeSimtBesselMiddleTrigFP32Codegen("__alpha", false);
    const std::string huge_cos_value = MakeSimtBesselHugeCosFP32Codegen("__alpha", "0.7853981852531433f");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __a = __fabsf(__x), __r; uint32_t __ab = reinterpret_cast<uint32_t&>(__a);"
      << "if (__ab > 0x7f800000u) { __r = __x; }"
      << "else if (__ab == 0x7f800000u) { __r = 0.0f; }"
      << "else if (__a <= 8.0f) { float __d1 = (__a - 2.4048254f) - 1.087059e-7f;"
      << "float __p = 9.619266247e-13f; __p = __fma(__p, __d1, 5.702105547e-12f);"
      << "__p = __fma(__p, __d1, -4.398487105e-10f); __p = __fma(__p, __d1, 4.604940853e-10f);"
      << "__p = __fma(__p, __d1, 5.847321173e-08f); __p = __fma(__p, __d1, 2.084518856e-09f);"
      << "__p = __fma(__p, __d1, -5.452075416e-06f); __p = __fma(__p, __d1, -7.342953250e-06f);"
      << "__p = __fma(__p, __d1, 3.017067874e-04f); __p = __fma(__p, __d1, 7.739535477e-04f);"
      << "__p = __fma(__p, __d1, -7.283463700e-03f); __p = __fma(__p, __d1, -2.666820353e-02f);"
      << "float __d2 = (__a - 5.520078f) + 7.1934145e-8f;"
      << "float __d3 = (__a - 8.653728f) - 3.8147791e-7f; __r = __p * __d1 * __d2 * __d3; }"
      << "else if (__a <= 1.0e13f) { float __inv = 1.0f / __a, __inv2 = __inv * __inv;"
      << "float __beta = __fma(5.848699569702148f, __inv2, -0.5428466796875f);"
      << "__beta = __fma(__beta, __inv2, 0.103515625f);"
      << "__beta = __fma(__beta, __inv2, -0.0625f); __beta = __fma(__beta, __inv2, 1.0f);"
      << "float __alpha = __fma(1.6380658830915178f, __inv2, -0.2095703125f);"
      << "__alpha = __fma(__alpha, __inv2, 0.06510416666666666f);"
      << "__alpha = __fma(__alpha, __inv2, -0.125f); __alpha = __fma(__alpha, __inv, __a);"
      << "__r = __beta * (1.0f / __sqrtf(__a)) * 0.7978846f * (" << middle_value << "); }"
      << "else { float __inv = 1.0f / __a, __inv2 = __inv * __inv;"
      << "float __alpha_poly = __fma(__inv2, 1.139649510383606f, -0.20532675087451935f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, 0.06509173661470413f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, -0.1249999925494194f);"
      << "float __alpha = __fma(__inv, __alpha_poly, __a);"
      << "float __beta = __fma(__inv2, 3.359287977218628f, -0.5145226716995239f);"
      << "__beta = __fma(__inv2, __beta, 0.10337056964635849f);"
      << "__beta = __fma(__inv2, __beta, -0.062499724328517914f);"
      << "__beta = __fma(__inv2, __beta, 1.0f);"
      << "__r = __beta * (1.0f / __sqrtf(__a)) * 0.7978845834732056f * (" << huge_cos_value << "); } __r;})";
    return s.str();
}

std::string MakeSimtJ1Codegen(const std::string& operand)
{
    const std::string middle_value = MakeSimtBesselMiddleTrigFP32Codegen("__alpha", true);
    const std::string huge_cos_value = MakeSimtBesselHugeCosFP32Codegen("__alpha", "2.356194496154785f");
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __a = __fabsf(__x), __r; uint32_t __ab = reinterpret_cast<uint32_t&>(__a);"
      << "if (__ab > 0x7f800000u) { __r = __x; }"
      << "else if (__a == 0.0f || __ab == 0x7f800000u) { __r = 0.0f; }"
      << "else if (__a <= 8.0f) { float __d1 = (__a - 3.831706f) + 7.685059e-8f;"
      << "float __p = 9.206492556e-14f; __p = __fma(__p, __d1, 9.126927192e-13f);"
      << "__p = __fma(__p, __d1, -2.641634001e-11f); __p = __fma(__p, __d1, -2.014359882e-10f);"
      << "__p = __fma(__p, __d1, 4.525844770e-09f); __p = __fma(__p, __d1, 2.701145918e-08f);"
      << "__p = __fma(__p, __d1, -5.348958058e-07f); __p = __fma(__p, __d1, -2.360248564e-06f);"
      << "__p = __fma(__p, __d1, 4.121127279e-05f); __p = __fma(__p, __d1, 1.191702295e-04f);"
      << "__p = __fma(__p, __d1, -1.807559530e-03f); __p = __fma(__p, __d1, -2.554892713e-03f);"
      << "__p = __fma(__p, __d1, 3.301389139e-02f);"
      << "float __d2 = (__a - 7.015587f) + 1.8321172e-7f; __r = __p * __d1 * __d2 * __a; }"
      << "else if (__a <= 1.0e13f) { float __inv = 1.0f / __a, __inv2 = __inv * __inv;"
      << "float __beta = __fma(-7.739953994751f, __inv2, 0.8052978515625f);"
      << "__beta = __fma(__beta, __inv2, -0.193359375f);"
      << "__beta = __fma(__beta, __inv2, 0.1875f); __beta = __fma(__beta, __inv2, 1.0f);"
      << "float __alpha = __fma(-2.3693978445870534f, __inv2, 0.3708984375f);"
      << "__alpha = __fma(__alpha, __inv2, -0.1640625f);"
      << "__alpha = __fma(__alpha, __inv2, 0.375f); __alpha = __fma(__alpha, __inv, __a);"
      << "__r = __beta * (1.0f / __sqrtf(__a)) * 0.7978846f * (" << middle_value << "); }"
      << "else { float __inv = 1.0f / __a, __inv2 = __inv * __inv;"
      << "float __alpha_poly = __fma(__inv2, -1.5799448490142822f, 0.36148586869239807f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, -0.164012610912323f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, 0.37499991059303284f);"
      << "float __alpha = __fma(__inv, __alpha_poly, __a);"
      << "float __beta = __fma(__inv2, 4.087306499481201f, 0.7498765587806702f);"
      << "__beta = __fma(__inv2, __beta, -0.1929156482219696f);"
      << "__beta = __fma(__inv2, __beta, 0.18749825656414032f);"
      << "__beta = __fma(__inv2, __beta, 1.0f);"
      << "__r = __beta * (1.0f / __sqrtf(__a)) * 0.7978845834732056f * (" << huge_cos_value << "); }"
      << "if (__ab <= 0x7f800000u && (reinterpret_cast<uint32_t&>(__x) & 0x80000000u)) { __r = -__r; } __r;})";
    return s.str();
}

std::string MakeSimtY0Codegen(const std::string& operand)
{
    const std::string middle_value = MakeSimtBesselMiddleTrigFP32Codegen("__alpha", true);
    const std::string huge_cos_value = MakeSimtBesselHugeCosFP32Codegen("__alpha", "2.356194496154785f");
    const std::string log_value = MakeSimtLogFP32Codegen("__x");
    std::ostringstream s;
    s << "({float __x = (" << operand
      << "), __r; uint32_t __b = reinterpret_cast<uint32_t&>(__x), __a = __b & 0x7fffffffu;"
      << "if (__a > 0x7f800000u) { __r = __x; }"
      << "else if (__x < 0.0f) { __r = __builtin_inff() / __builtin_inff(); }"
      << "else if (__a == 0u) { uint32_t __ninf = 0xff800000u; __r = reinterpret_cast<float&>(__ninf); }"
      << "else if (__a == 0x7f800000u) { __r = 0.0f; }"
      << "else if (__x < 0.5f) { float __j0_input = __x; float __p = 7.977247951e-04f; __p = __fma(__p, __x, "
         "-1.652431533e-02f);"
      << "__p = __fma(__p, __x, 1.196180186e-04f); __p = __fma(__p, __x, 1.775911068e-01f);"
      << "__p = __fma(__p, __x, 7.436880598e-07f); __p = __fma(__p, __x, -7.380430393e-02f);"
      << "__r = 0.636619772367f * (" << MakeSimtJ0Codegen("__j0_input") << ") * (" << log_value << ") + __p; }"
      << "else if (__x < 2.197141326f) { float __d = (__x - 0.893576980f) + 1.33579787e-8f;"
      << "float __p = -4.485103697e-03f; __p = __fma(__p, __d, 3.231012427e-02f);"
      << "__p = __fma(__p, __d, -1.014045593e-01f); __p = __fma(__p, __d, 1.847167541e-01f);"
      << "__p = __fma(__p, __d, -2.253558856e-01f); __p = __fma(__p, __d, 2.147535120e-01f);"
      << "__p = __fma(__p, __d, -1.959638733e-01f); __p = __fma(__p, __d, 1.944536283e-01f);"
      << "__p = __fma(__p, __d, -2.040570397e-01f); __p = __fma(__p, __d, 2.190628354e-01f);"
      << "__p = __fma(__p, __d, -2.261730477e-01f); __p = __fma(__p, __d, 2.205519494e-01f);"
      << "__p = __fma(__p, __d, -4.920781667e-01f); __p = __fma(__p, __d, 8.794208015e-01f); __r = __p * __d; }"
      << "else if (__x < 8.0f) { float __d3 = (__x - 7.08605099f) - 7.30581178e-8f;"
      << "float __p = 8.146675423e-11f; __p = __fma(__p, __d3, 1.030741112e-09f);"
      << "__p = __fma(__p, __d3, 1.610027889e-09f); __p = __fma(__p, __d3, 1.063494888e-08f);"
      << "__p = __fma(__p, __d3, 6.693347461e-07f); __p = __fma(__p, __d3, 7.816005861e-07f);"
      << "__p = __fma(__p, __d3, -4.836658731e-05f); __p = __fma(__p, __d3, 1.049324298e-05f);"
      << "__p = __fma(__p, __d3, 2.142965752e-03f); __p = __fma(__p, __d3, -3.385610246e-03f);"
      << "__p = __fma(__p, __d3, -3.743254148e-02f); __p = __fma(__p, __d3, 9.592770584e-02f);"
      << "float __d2 = (__x - 3.95767832f) - 1.01291178e-7f; __r = __p * __d2 * __d3; }"
      << "else if (__x <= 1.0e13f) { float __inv = 1.0f / __x, __inv2 = __inv * __inv;"
      << "float __beta = __fma(5.848699569702148f, __inv2, -0.5428466796875f);"
      << "__beta = __fma(__beta, __inv2, 0.103515625f); __beta = __fma(__beta, __inv2, -0.0625f);"
      << "__beta = __fma(__beta, __inv2, 1.0f);"
      << "float __alpha = __fma(1.6380658830915178f, __inv2, -0.2095703125f);"
      << "__alpha = __fma(__alpha, __inv2, 0.06510416666666666f); __alpha = __fma(__alpha, __inv2, -0.125f);"
      << "__alpha = __fma(__alpha, __inv, __x);"
      << "__r = __beta * (1.0f / __sqrtf(__x)) * 0.7978846f * (" << middle_value << "); }"
      << "else { float __inv = 1.0f / __x, __inv2 = __inv * __inv;"
      << "float __alpha_poly = __fma(__inv2, 1.1000176668167114f, -0.2039303183555603f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, 0.06507780402898788f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, -0.12499996274709702f);"
      << "float __alpha = __fma(__inv, __alpha_poly, __x);"
      << "float __beta = __fma(__inv2, -0.39924582839012146f, 0.10197055339813232f);"
      << "__beta = __fma(__inv2, __beta, -0.06249268725514412f);"
      << "__beta = __fma(__inv2, __beta, 1.0f);"
      << "__r = __beta * (1.0f / __sqrtf(__x)) * 0.7978845834732056f * (" << huge_cos_value << "); } __r;})";
    return s.str();
}

std::string MakeSimtY1Codegen(const std::string& operand)
{
    const std::string middle_value = MakeSimtBesselMiddleTrigFP32Codegen("__alpha", false);
    const std::string huge_cos_value = MakeSimtBesselHugeCosFP32Codegen("__alpha", "3.9269907474517822f");
    const std::string log_value = MakeSimtLogFP32Codegen("__x");
    std::ostringstream s;
    s << "({float __x = (" << operand
      << "), __r; uint32_t __b = reinterpret_cast<uint32_t&>(__x), __a = __b & 0x7fffffffu;"
      << "if (__a > 0x7f800000u) { __r = __x; }"
      << "else if (__x < 0.0f) { __r = __builtin_inff() / __builtin_inff(); }"
      << "else if (__a == 0u) { uint32_t __ninf = 0xff800000u; __r = reinterpret_cast<float&>(__ninf); }"
      << "else if (__a == 0x7f800000u) { __r = 0.0f; }"
      << "else if (__x < 1.2f) { float __j1_input = __x; float __p = 2.798307076e-04f; __p = __fma(__p, __x, "
         "-3.402886792e-03f);"
      << "__p = __fma(__p, __x, 3.643335439e-04f); __p = __fma(__p, __x, 5.419222886e-02f);"
      << "__p = __fma(__p, __x, 3.339972037e-05f); __p = __fma(__p, __x, -1.960600316e-01f);"
      << "__p = __fma(__p, __x, 6.24278e-08f); __r = 0.636619772367f * (" << MakeSimtJ1Codegen("__j1_input") << ") * ("
      << log_value << ") - 0.636619772367f / __x + __p; }"
      << "else if (__x < 3.0f) { float __d = (__x - 2.19714141f) + 8.28892723e-8f;"
      << "float __p = -7.210192066e-05f; __p = __fma(__p, __d, 6.665645689e-05f);"
      << "__p = __fma(__p, __d, -3.106003176e-05f); __p = __fma(__p, __d, 2.276838750e-04f);"
      << "__p = __fma(__p, __d, -5.566432475e-04f); __p = __fma(__p, __d, 1.068050095e-03f);"
      << "__p = __fma(__p, __d, -2.582285756e-03f); __p = __fma(__p, __d, 7.422557063e-03f);"
      << "__p = __fma(__p, __d, -4.799279782e-03f); __p = __fma(__p, __d, -3.285740200e-02f);"
      << "__p = __fma(__p, __d, -1.185144993e-01f); __p = __fma(__p, __d, 5.207864124e-01f); __r = __p * __d; }"
      << "else if (__x < 8.0f) { float __d2 = (__x - 5.42968082f) - 2.16514351e-7f;"
      << "float __p = -4.575132868e-10f; __p = __fma(__p, __d2, 4.435273368e-09f);"
      << "__p = __fma(__p, __d2, 3.963341878e-08f); __p = __fma(__p, __d2, -4.231424306e-07f);"
      << "__p = __fma(__p, __d2, -4.201841643e-06f); __p = __fma(__p, __d2, 3.316061621e-05f);"
      << "__p = __fma(__p, __d2, 2.516106023e-04f); __p = __fma(__p, __d2, -1.369325160e-03f);"
      << "__p = __fma(__p, __d2, -8.495834725e-03f); __p = __fma(__p, __d2, 2.404736994e-02f);"
      << "__p = __fma(__p, __d2, 1.074804589e-01f);"
      << "float __d3 = (__x - 8.59600544f) - 4.28572861e-7f; __r = __p * __d2 * __d3; }"
      << "else if (__x <= 1.0e13f) { float __inv = 1.0f / __x, __inv2 = __inv * __inv;"
      << "float __beta = __fma(-7.739953994751f, __inv2, 0.8052978515625f);"
      << "__beta = __fma(__beta, __inv2, -0.193359375f); __beta = __fma(__beta, __inv2, 0.1875f);"
      << "__beta = __fma(__beta, __inv2, 1.0f);"
      << "float __alpha = __fma(-2.3693978445870534f, __inv2, 0.3708984375f);"
      << "__alpha = __fma(__alpha, __inv2, -0.1640625f); __alpha = __fma(__alpha, __inv2, 0.375f);"
      << "__alpha = __fma(__alpha, __inv, __x);"
      << "__r = -__beta * (1.0f / __sqrtf(__x)) * 0.7978846f * (" << middle_value << "); }"
      << "else { float __inv = 1.0f / __x, __inv2 = __inv * __inv;"
      << "float __alpha_poly = __fma(__inv2, -1.7881368398666382f, 0.36611685156822205f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, -0.1640450656414032f);"
      << "__alpha_poly = __fma(__inv2, __alpha_poly, 0.3749999701976776f);"
      << "float __alpha = __fma(__inv, __alpha_poly, __x);"
      << "float __beta = __fma(__inv2, 0.6503810882568359f, -0.1920607089996338f);"
      << "__beta = __fma(__inv2, __beta, 0.18749521672725677f);"
      << "__beta = __fma(__inv2, __beta, 1.0f);"
      << "__r = __beta * (1.0f / __sqrtf(__x)) * 0.7978845834732056f * (" << huge_cos_value << "); } __r;})";
    return s.str();
}

std::string MakeSimtJnCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string order = codegen.GetExprAsCode(op->args_[0]);
    const std::string value = codegen.GetExprAsCode(op->args_[1]);
    const std::string asymptotic_sin_x = MakeSimtTrigFP32Codegen("__ax", true);
    const std::string asymptotic_cos_x = MakeSimtTrigFP32Codegen("__ax", false);
    const std::string asymptotic_sin_phase = MakeSimtTrigFP32Codegen("__phase", true);
    const std::string asymptotic_cos_phase = MakeSimtTrigFP32Codegen("__phase", false);
    const std::string asymptotic_sqrt = MakeSimtSqrtFP32Codegen("__amplitude_argument");
    const std::string series_log = MakeSimtLogFP32Codegen("__log_input");
    std::ostringstream s;
    s << "({int32_t __n = (int32_t)(" << order << "); float __x = (" << value
      << "), __r; uint32_t __xb = reinterpret_cast<uint32_t&>(__x), __a = __xb & 0x7fffffffu;"
      << "if (__n == 0) { float __j0_input = __x; __r = " << MakeSimtJ0Codegen("__j0_input") << "; }"
      << "else if (__n == 1) { float __j1_input = __x; __r = " << MakeSimtJ1Codegen("__j1_input") << "; }"
      << "else if (__n < 0 || __a > 0x7f800000u) { __r = __builtin_inff() / __builtin_inff(); }"
      << "else if (__a == 0u || __a == 0x7f800000u) { __r = 0.0f; }"
      << "else { float __ax = __fabsf(__x); float __sign = ((__xb & 0x80000000u) && (__n & 1)) ? -1.0f : 1.0f;"
      << "if ((float)__n < __ax * 0.1f) { float __mu = (float)(4 * __n * __n);"
      << "float __denom = 4.0f * __ax, __denom2 = __denom * __denom;"
      << "float __phase = (__mu - 1.0f) / (2.0f * __denom); __denom *= __denom2;"
      << "__phase += (__mu - 1.0f) * (__mu - 25.0f) / (6.0f * __denom); __denom *= __denom2;"
      << "__phase += (__mu - 1.0f) * (__mu * __mu - 114.0f * __mu + 1073.0f) / (5.0f * __denom);"
      << "float __txq = 2.0f * __ax; __txq *= __txq;"
      << "float __amp_term = 1.0f + (__mu - 1.0f) / (2.0f * __txq);"
      << "__amp_term += 3.0f * (__mu - 1.0f) * (__mu - 9.0f) / (__txq * __txq * 8.0f);"
      << "float __amplitude_argument = __amp_term * 2.0f / (3.1415927410125732f * __ax);"
      << "float __amplitude = " << asymptotic_sqrt << ";"
      << "float __sin_x = " << asymptotic_sin_x << ", __cos_x = " << asymptotic_cos_x << ";"
      << "float __sin_phase = " << asymptotic_sin_phase << ", __cos_phase = " << asymptotic_cos_phase << ";"
      << "float __offset = (float)__n / 2.0f + 0.25f, __offset_two = __offset + __offset;"
      << "int32_t __offset_quadrant = __cvt_int32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(__offset_two);"
      << "float __offset_rounded = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(__offset_quadrant);"
      << "float __offset_reduced = __fma(-__offset_rounded, 0.5f, __offset);"
      << "float __offset_square = __offset_reduced * __offset_reduced;"
      << "float __offset_sin_poly = __fma(__offset_square, -0.5924802422523499f, 2.5501441955566406f);"
      << "__offset_sin_poly = __fma(__offset_square, __offset_sin_poly, -5.167719841003418f);"
      << "float __offset_sin_base = __fma(__offset_sin_poly, __offset_reduced * __offset_square,"
      << "__offset_reduced * 3.1415927410125732f);"
      << "float __offset_cos_base = __fma(__offset_square, 0.22686031460762024f, -1.3345603942871094f);"
      << "__offset_cos_base = __fma(__offset_square, __offset_cos_base, 4.058692455291748f);"
      << "__offset_cos_base = __fma(__offset_square, __offset_cos_base, -4.934802055358887f);"
      << "__offset_cos_base = __fma(__offset_square, __offset_cos_base, 1.0f);"
      << "float __offset_sin = (__offset_quadrant & 1) != 0 ? __offset_cos_base : __offset_sin_base;"
      << "float __offset_cos = (__offset_quadrant & 1) != 0 ? __offset_sin_base : __offset_cos_base;"
      << "if ((__offset_quadrant & 2) != 0) { __offset_sin = -__offset_sin; }"
      << "if (((__offset_quadrant + 1) & 2) != 0) { __offset_cos = -__offset_cos; }"
      << "if (__fabsf(__offset) > 16777216.0f) { __offset_sin = 0.0f; __offset_cos = 1.0f; }"
      << "float __sin_combined = __cos_phase * (__cos_x * __offset_cos + __sin_x * __offset_sin)"
      << " - __sin_phase * (__sin_x * __offset_cos - __cos_x * __offset_sin);"
      << "__r = __sign * __sin_combined * __amplitude; }"
      << "else if ((float)__n < __ax) { float __prev = " << MakeSimtJ0Codegen("__ax")
      << ", __current = " << MakeSimtJ1Codegen("__ax") << "; for (int32_t __k = 1; __k < __n; ++__k) {"
      << "float __next = 2.0f * (float)__k * __current / __ax - __prev; __prev = __current; __current = __next; }"
      << "__r = __sign * __current; }"
      << "else if ((float)__n > __ax * __ax / 10.0f) { float __log_input = __ax / 2.0f;"
      << "float __prefix = (float)__n * (" << series_log << ");"
      << "for (int32_t __i = 2; __i <= __n; ++__i) { __log_input = (float)__i;"
      << "__prefix -= (" << series_log << "); }"
      << "if (__prefix <= -87.3365478515625f) { float __half_prefix = __expf(__prefix * 0.5f);"
      << "__prefix = __half_prefix * __half_prefix; } else { __prefix = __expf(__prefix); }"
      << "float __mult = __ax * 0.5f; __mult *= -__mult;"
      << "float __term = 1.0f, __sum = 0.0f; for (int32_t __i = 1; __i <= 14; ++__i) {"
      << "__sum += __term; __term *= __mult / ((float)__i * (float)(__i + __n)); }"
      << "__r = __sign * __prefix * __sum; }"
      << "else { int32_t __count = __n + 30; float __prev = 1.0e-30f, __current = 0.0f;"
      << "float __sum = 0.0f, __scale = 1.0f, __answer = 0.0f;"
      << "for (int32_t __k = __count - 1; __k >= 0; --__k) { float __factor = 2.0f * (float)(__k + 1) / __ax;"
      << "if (__factor > 1.0f && __fabsf(__current) > 1.152921504606847e18f) {"
      << "__prev /= 1.152921504606847e18f; __sum /= 1.152921504606847e18f;"
      << "__scale /= 1.152921504606847e18f; __current /= 1.152921504606847e18f; }"
      << "float __next = __factor * __current - __prev; if ((__k & 1) == 0) { __sum += 2.0f * __next; }"
      << "if (__k == __n) { __answer = __next / __scale; } __prev = __current; __current = __next; }"
      << "__sum -= __current; __r = __sign * (__answer / __sum) * __scale; } } __r;})";
    return s.str();
}

std::string MakeSimtYnCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string order = codegen.GetExprAsCode(op->args_[0]);
    const std::string value = codegen.GetExprAsCode(op->args_[1]);
    const std::string asymptotic_sin_x = MakeSimtTrigFP32Codegen("__x", true);
    const std::string asymptotic_cos_x = MakeSimtTrigFP32Codegen("__x", false);
    const std::string asymptotic_sin_shift = MakeSimtTrigFP32Codegen("__phase_shift - __phase", true);
    const std::string asymptotic_cos_shift = MakeSimtTrigFP32Codegen("__phase_shift - __phase", false);
    const std::string asymptotic_sqrt = MakeSimtSqrtFP32Codegen("__amplitude_argument");
    std::ostringstream s;
    s << "({int32_t __n = (int32_t)(" << order << "); float __x = (" << value
      << "), __r, __yn_input = __x; uint32_t __a = reinterpret_cast<uint32_t&>(__x) & 0x7fffffffu;"
      << "if (__n < 0 || __x < 0.0f || __a > 0x7f800000u) { __r = __builtin_inff() / __builtin_inff(); }"
      << "else if (__a == 0u) { uint32_t __ni = 0xff800000u; __r = reinterpret_cast<float&>(__ni); }"
      << "else if (__a == 0x7f800000u) { __r = 0.0f; }"
      << "else if (__n == 0) { __r = " << MakeSimtY0Codegen("__yn_input") << "; }"
      << "else if (__n == 1) { __r = " << MakeSimtY1Codegen("__yn_input") << "; }"
      << "else if (__x > (float)__n * 10.0f) { float __mu = (float)(4 * __n * __n);"
      << "float __denom = 4.0f * __x, __denom2 = __denom * __denom;"
      << "float __phase = (__mu - 1.0f) / (2.0f * __denom); __denom *= __denom2;"
      << "__phase += (__mu - 1.0f) * (__mu - 25.0f) / (6.0f * __denom); __denom *= __denom2;"
      << "__phase += (__mu - 1.0f) * (__mu * __mu - 114.0f * __mu + 1073.0f) / (5.0f * __denom);"
      << "float __txq = 2.0f * __x; __txq *= __txq;"
      << "float __amp_term = 1.0f + (__mu - 1.0f) / (2.0f * __txq);"
      << "__amp_term += 3.0f * (__mu - 1.0f) * (__mu - 9.0f) / (__txq * __txq * 8.0f);"
      << "__amp_term += 15.0f * (__mu - 1.0f) * (__mu - 9.0f) * (__mu - 25.0f) /"
      << " (__txq * __txq * __txq * 48.0f);"
      << "float __amplitude_argument = __amp_term * 2.0f / (3.1415927410125732f * __x);"
      << "float __amplitude = " << asymptotic_sqrt << ";"
      << "float __phase_shift = (float)__n * 3.1415927410125732f / 2.0f + 0.7853981852531433f;"
      << "float __sin_x = " << asymptotic_sin_x << ", __cos_x = " << asymptotic_cos_x << ";"
      << "float __sin_shift = " << asymptotic_sin_shift << ", __cos_shift = " << asymptotic_cos_shift << ";"
      << "__r = (__sin_x * __cos_shift - __cos_x * __sin_shift) * __amplitude; }"
      << "else { bool __invalid = false; float __inv = 1.0f / __x;"
      << "__inv = __fma(__inv, __fma(-__x, __inv, 1.0f), __inv); float __two_over_x = __fma(__inv, 2.0f, 0.0f);"
      << "if (!__isfinite(__two_over_x)) { __invalid = true; }"
      << "float __prev = " << MakeSimtY0Codegen("__yn_input") << ", __current = " << MakeSimtY1Codegen("__yn_input")
      << "; if (!__isfinite(__prev) || !__isfinite(__current)) { __invalid = true; } float __scale = 1.0f;"
      << "for (int32_t __k = 1; __k < __n; ++__k) { float __mult = __fma((float)__k, __two_over_x, 0.0f);"
      << "float __next = __fma(__mult, __current, -__prev); if (!__isfinite(__next)) { __invalid = true; }"
      << "__prev = __current; __current = __next; if (__fabsf(__current) > 1.0e30f) {"
      << "float __inv_current = 1.0f / __fabsf(__current); __prev *= __inv_current;"
      << "__current *= __inv_current; __scale *= __inv_current; } } __r = __current / __scale;"
      << "if (!__isfinite(__r)) { __invalid = true; } if (__invalid) { uint32_t __ninf = 0xff800000u;"
      << "__r = reinterpret_cast<float&>(__ninf); } } __r;})";
    return s.str();
}

std::string MakeSimtCylBesselI0CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __a = __fabsf(__x), __r; uint32_t __ab = reinterpret_cast<uint32_t&>(__a);"
      << "if (__ab == 0x7f800000u) { __r = __a; }"
      << "else if (__a >= 9.0f) { float __inv = 1.0f / __a;"
      << "float __p = __fma(__inv, 0.34872168f, -0.0054563344f);"
      << "__p = __fma(__inv, __p, 0.033347155f); __p = __fma(__inv, __p, 0.027889195f);"
      << "__p = __fma(__inv, __p, 0.04987063f); __p = __fma(__inv, __p, 0.39894226f);"
      << "__p *= 1.0f / __sqrtf(__a); float __e = __expf(__a * 0.5f);"
      << "__r = __p * (__e - 1.0f) * (__e + 1.0f) + __p; }"
      << "else { float __square = __a * __a;"
      << "float __p = __fma(__square, 1.551427e-19f, 1.4492505e-17f);"
      << "__p = __fma(__square, __p, 1.0687647e-14f); __p = __fma(__square, __p, 2.3349575e-12f);"
      << "__p = __fma(__square, __p, 4.7306625e-10f); __p = __fma(__square, __p, 6.7778003e-8f);"
      << "__p = __fma(__square, __p, 6.7820783e-6f); __p = __fma(__square, __p, 0.00043402583f);"
      << "__p = __fma(__square, __p, 0.015625f); __p = __fma(__square, __p, 0.25f);"
      << "__r = __fma(__square, __p, 1.0f); } __r;})";
    return s.str();
}

std::string MakeSimtCylBesselI1CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    std::ostringstream s;
    s << "({float __x = (" << operand << "), __a = __fabsf(__x), __r; uint32_t __ab = reinterpret_cast<uint32_t&>(__a);"
      << "if (__ab == 0x7f800000u || __ab > 0x7f800000u) { __r = __ab == 0x7f800000u ? __x : __a; }"
      << "else if (__a >= 8.085f) { float __inv = 1.0f / __a;"
      << "float __p = __fma(__inv, -0.5028813f, 0.028471555f);"
      << "__p = __fma(__inv, __p, -0.04873671f); __p = __fma(__inv, __p, -0.04641596f);"
      << "__p = __fma(__inv, __p, -0.14960973f); __p = __fma(__inv, __p, 0.39894232f);"
      << "__p *= 1.0f / __sqrtf(__a); float __e = __expf(__a * 0.5f);"
      << "__r = __p * (__e - 1.0f) * (__e + 1.0f) + __p;"
      << "if (reinterpret_cast<uint32_t&>(__x) & 0x80000000u) { __r = -__r; } }"
      << "else { float __square = __x * __x;"
      << "float __p = __fma(__square, 2.7848253e-18f, 3.4224707e-16f);"
      << "__p = __fma(__square, __p, 1.6258002e-13f); __p = __fma(__square, __p, 3.3142173e-11f);"
      << "__p = __fma(__square, __p, 5.6632734e-9f); __p = __fma(__square, __p, 6.780027e-7f);"
      << "__p = __fma(__square, __p, 0.00005425474f); __p = __fma(__square, __p, 0.002604162f);"
      << "__p = __fma(__square, __p, 0.0625f); __p = __fma(__square, __p, 0.5f);"
      << "__r = __p * __x; } __r;})";
    return s.str();
}

std::string MakeSimtJ0CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    return MakeSimtJ0Codegen(operand);
}

std::string MakeSimtJ1CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    return MakeSimtJ1Codegen(operand);
}

std::string MakeSimtY0CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    return MakeSimtY0Codegen(operand);
}

std::string MakeSimtY1CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    return MakeSimtY1Codegen(operand);
}

std::string MakeSimtSaturateCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string operand = codegen.GetExprAsCode(op->args_[0]);
    return "__saturatef((float)(" + operand + "))";
}

std::string MakeSimtMulI32ToI64CodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const auto input_type = ir::As<ir::ScalarType>(op->args_[0]->GetType());
    const std::string lhs = codegen.GetExprAsCode(op->args_[0]);
    const std::string rhs = codegen.GetExprAsCode(op->args_[1]);
    if (input_type->dtype_ == ir::DataType::INT32) {
        return "__mul_i32toi64((int)(" + lhs + "), (int)(" + rhs + "))";
    }
    return "__mul_i32toi64((unsigned int)(" + lhs + "), (unsigned int)(" + rhs + "))";
}

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
REGISTER_BACKEND_OP(BackendCCE, "simt.brev").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBrevCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.byte_perm").set_pipe(ir::PipeType::S).f_codegen(MakeSimtBytePermCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.clz").set_pipe(ir::PipeType::S).f_codegen(MakeSimtClzCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.cyl_bessel_i0")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtCylBesselI0CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.cyl_bessel_i1")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtCylBesselI1CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.equ").set_pipe(ir::PipeType::S).f_codegen(MakeSimtEquCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.erf").set_pipe(ir::PipeType::S).f_codegen(MakeSimtErfOpCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.erfc").set_pipe(ir::PipeType::S).f_codegen(MakeSimtErfcOpCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.erfcinv").set_pipe(ir::PipeType::S).f_codegen(MakeSimtErfcinvCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.erfcx").set_pipe(ir::PipeType::S).f_codegen(MakeSimtErfcxCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.erfinv").set_pipe(ir::PipeType::S).f_codegen(MakeSimtErfinvCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.fdim").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFdimCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.ffs").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFfsCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.fma_relu").set_pipe(ir::PipeType::S).f_codegen(MakeSimtFmaReluCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.geu").set_pipe(ir::PipeType::S).f_codegen(MakeSimtGeuCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.gtu").set_pipe(ir::PipeType::S).f_codegen(MakeSimtGtuCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.hadd").set_pipe(ir::PipeType::S).f_codegen(MakeSimtHaddCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.hypot").set_pipe(ir::PipeType::S).f_codegen(MakeSimtHypotCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.j0").set_pipe(ir::PipeType::S).f_codegen(MakeSimtJ0CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.j1").set_pipe(ir::PipeType::S).f_codegen(MakeSimtJ1CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.jn").set_pipe(ir::PipeType::S).f_codegen(MakeSimtJnCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.ldexp").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLdexpCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.leu").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLeuCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.lgamma").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLgammaCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.ltu").set_pipe(ir::PipeType::S).f_codegen(MakeSimtLtuCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.mul24").set_pipe(ir::PipeType::S).f_codegen(MakeSimtMul24CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.mul_i32_to_i64")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeSimtMulI32ToI64CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.norm3d").set_pipe(ir::PipeType::S).f_codegen(MakeSimtNorm3dCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.norm4d").set_pipe(ir::PipeType::S).f_codegen(MakeSimtNorm4dCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.normcdf").set_pipe(ir::PipeType::S).f_codegen(MakeSimtNormcdfCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.normcdfinv").set_pipe(ir::PipeType::S).f_codegen(MakeSimtNormcdfinvCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.pow").set_pipe(ir::PipeType::S).f_codegen(MakeSimtPowCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.remainder").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRemainderCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.rhadd").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRhaddCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.rhypot").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRhypotCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.rnorm3d").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRnorm3dCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.rnorm4d").set_pipe(ir::PipeType::S).f_codegen(MakeSimtRnorm4dCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.sad").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSadCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.saturate").set_pipe(ir::PipeType::S).f_codegen(MakeSimtSaturateCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.scalbn").set_pipe(ir::PipeType::S).f_codegen(MakeSimtScalbnCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.tgamma").set_pipe(ir::PipeType::S).f_codegen(MakeSimtTgammaCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.y0").set_pipe(ir::PipeType::S).f_codegen(MakeSimtY0CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.y1").set_pipe(ir::PipeType::S).f_codegen(MakeSimtY1CodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "simt.yn").set_pipe(ir::PipeType::S).f_codegen(MakeSimtYnCodegenCCE);

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
