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
 * @file backend_950_cce_vf_ops.cpp
 * \brief CCE backend op registration for VF API operations (A5 target).
 *
 * VF ops directly emit CCE intrinsics (vlds, vmax, vdup, etc.) without
 * going through the PTO-ISA intermediate layer.
 */

#include <string>
#include <unordered_map>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "backend/common/backend_utils.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/pipe.h"
#include "ir/op_attr_types.h"
#include "ir/type.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;
using npu::tile_fwk::ExternalError;

static std::string VFEnumValueName(const char* fullName)
{
    const char* sep = std::strrchr(fullName, ':');
    return sep ? std::string(sep + 1) : std::string(fullName);
}

// The dist kwarg arrives as a raw int (the pybind enum's .value is extracted
// at the Python/C++ boundary); invalid enumerators yield "UNKNOWN" here.
static std::string VFCheckedDistName(int distVal, bool isLoad, const std::string& opName)
{
    const char* fullName = isLoad ? ir::EnumToString(static_cast<ir::LoadDist>(distVal)) :
                                    ir::EnumToString(static_cast<ir::StoreDist>(distVal));
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, std::strcmp(fullName, "UNKNOWN") != 0)
        << opName << " dist=" << distVal << " is not a valid " << (isLoad ? "ir::LoadDist" : "ir::StoreDist")
        << " enumerator";
    return VFEnumValueName(fullName);
}

// Format a DataType for log messages in the frontend DT_XXX style.
static std::string DTypeStr(const DataType& dt) { return "DT_" + ir::DTypeToString(dt); }

// True when a source operand is a MaskReg (v* vs p* intrinsic dispatch).
static bool IsSrcMaskReg(const ir::CallPtr& op, size_t idx = 0)
{
    if (idx >= op->args_.size())
        return false;
    return ir::IsMaskReg(op->args_[idx]->GetType());
}

static void EmitRtWrapDecl(codegen::CCECodegen& codegen, ir::DataType dtype, const std::string& name)
{
    codegen.Emit("RtWrap<" + dtype.ToCTypeString() + "> " + name + ";");
}

static std::string VFDstItemName(const std::string& target, size_t index)
{
    return target + "__item_" + std::to_string(index);
}

static std::string VFDstCType(const ir::TypePtr& type)
{
    auto reg = ir::As<ir::RegType>(type);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, reg != nullptr)
        << "vf value-form dst type must be a register, got " << type->TypeName();
    switch (reg->regtype_) {
        case ir::RegKind::VecReg:
            if (reg->dtype_ == DataType::INT4)
                return "vector_s4x2";
            if (reg->dtype_ == DataType::UINT4)
                return "vector_u4x2";
            return "RtWrap<" + reg->dtype_.ToCTypeString() + ">";
        case ir::RegKind::PredReg:
            return "MaskReg";
        case ir::RegKind::AddrReg:
            return "AddrReg";
        case ir::RegKind::UnalignReg:
            return "UnalignReg";
        default:
            break;
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "unknown vf register kind";
    return "";
}

static size_t VFDstCount(const ir::CallPtr& op)
{
    if (auto reg = ir::As<ir::RegType>(op->GetType())) {
        return (reg->regtype_ == ir::RegKind::VecReg || reg->regtype_ == ir::RegKind::PredReg) ? 1 : 0;
    }
    if (auto tuple = ir::As<ir::TupleType>(op->GetType()))
        return tuple->types_.size();
    if (ir::As<ir::NoneType>(op->GetType()) != nullptr || ir::As<ir::UnknownType>(op->GetType()) != nullptr)
        return 0;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
        << op->name_ << " has unexpected value-form result type " << op->GetType()->TypeName();
    return 0;
}

template <typename Fn>
static std::string EmitVFValueForm(const ir::CallPtr& op, codegen::CodegenBase& codegenBase, Fn emit)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    size_t dstCount = VFDstCount(op);
    std::string target = codegen.GetCurrentResultTarget();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, dstCount == 0 || !target.empty())
        << op->name_ << " value-form call requires an assignment target";
    std::vector<ir::ExprPtr> args;
    args.reserve(dstCount + op->args_.size());
    std::vector<ir::ExprPtr> dstVars;
    for (size_t i = 0; i < dstCount; ++i) {
        ir::TypePtr elem = (dstCount == 1) ? op->GetType() : ir::As<ir::TupleType>(op->GetType())->types_[i];
        std::string name = (dstCount == 1) ? target : VFDstItemName(target, i);
        auto var = std::make_shared<ir::Var>(name, elem, op->span_);
        codegen.RegisterVar(var, name);
        codegen.Emit(VFDstCType(elem) + " " + codegen.GetExprAsCode(var) + ";");
        args.push_back(var);
        dstVars.push_back(var);
    }
    args.insert(args.end(), op->args_.begin(), op->args_.end());
    if (dstCount > 1) {
        codegen.RegisterTuple(target, std::make_shared<ir::MakeTuple>(dstVars, op->span_));
    }
    auto patched = std::make_shared<ir::Call>(op->name_, std::move(args), op->kwargs_, op->GetType(), op->span_);
    return emit(patched, codegen);
}

// ============================================================================
// Dtype carried by an IR type: ScalarType / RegType / ShapedType, with a
// caller-supplied fallback. Of the register kinds only RegTensor and MaskReg
// carry a data dtype worth propagating; AddrReg/UnalignReg fall through to
// the caller's fallback (their UINT32 storage dtype is an implementation
// detail).
static DataType GetTypeDtype(const ir::TypePtr& type, DataType fallback = DataType::UINT32)
{
    if (auto st = ir::As<ir::ScalarType>(type))
        return st->dtype_;
    if (auto rt = ir::As<ir::RegType>(type)) {
        if (rt->regtype_ == ir::RegKind::VecReg || rt->regtype_ == ir::RegKind::PredReg) {
            return rt->dtype_;
        }
        return fallback;
    }
    if (auto sh = ir::As<ir::ShapedType>(type))
        return sh->dtype_;
    return fallback;
}

// Best-effort dtype extraction from an Expr's deduced type. Handles both
// ScalarType (RegTensor outputs) and ShapedType (Tile / Tensor expressions).
static DataType GetExprDtype(const ir::ExprPtr& expr, DataType fallback = DataType::UINT32)
{
    return GetTypeDtype(expr->GetType(), fallback);
}

// Coerce a float-constant scalar to an integer literal matching the src type,
// so that the emitted intrinsic (vmuls/vadds/vmins/vmaxs/vcmps_*/vaxpy)
// receives a correctly-typed integer literal instead of a float literal
// that would trigger -Wliteral-conversion. Shared with the elementwise/TEXPANDS
// backends via backend_utils so float->int truncation is uniform.

// For ops that only support ZEROING: return "MODE_ZEROING", reject MERGING.
static std::string VFZeroingOnly(const ir::CallPtr& op, const std::string& opName)
{
    if (op->HasKwarg("mode")) {
        auto mode = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, mode == ir::MergeMode::ZEROING)
            << opName << " only supports ZEROING mode on current device";
    }
    return "MODE_ZEROING";
}

// Check if a DataType is a b8-width type (8-bit storage).
// Includes INT8, UINT8, BOOL, and all FP8 types (FP8E4M3FN, FP8E5M2, HF8).
// FP4 types (FP4E2M1, FP4E1M2, FP4) are b4 but stored as b8 (packed 2-per-byte),
// so they are also treated as b8 for load/store mode selection.
static bool IsB8Type(DataType dt) { return dt.GetBit() <= 8; }

// Integer types with a native VF arithmetic instruction form (vadd/vsub/vmax/
// vcmp ... have no 4-bit form, so INT4/UINT4 must be rejected up front).
static bool IsArithIntType(DataType dt) { return dt.IsInt() && dt.GetBit() >= 8; }
static bool IsArithSignedIntType(DataType dt) { return dt.IsSignedInt() && dt.GetBit() >= 8; }

// Check if a DataType is a b16-width type (16-bit storage).
static bool IsB16Type(DataType dt) { return dt.GetBit() == 16; }

// ============================================================================
// B64 helper: deinterleave a b64 RegTensor into b32 low/high halves.
// Emits temp RegTensor declarations and a vdintlv instruction.
// After call, {prefix}_lo_ holds the low 32 bits and {prefix}_hi_ holds the
// high 32 bits of each b64 element (32 meaningful elements in positions 0-31).
// ============================================================================
static void EmitB64Deinterleave(codegen::CCECodegen& codegen, const std::string& prefix, const std::string& b64Reg)
{
    std::string lo = prefix + "_lo_";
    std::string hi = prefix + "_hi_";
    std::string zero = prefix + "_zero_";
    std::string dump = prefix + "_ddump_";
    std::string m = prefix + "_dm_";
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, lo);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, hi);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, zero);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, dump);
    codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
    codegen.Emit("vdup(" + zero + ", 0, " + m + ", MODE_ZEROING);");
    codegen.Emit("vdintlv(" + lo + ", " + hi + ", (RegTensor<uint32_t>&)" + b64Reg + ", " + zero + ");");
}

// ============================================================================
// B64 helper: interleave b32 low/high halves into a b64 RegTensor.
// Emits a vintlv instruction that combines lo and hi into b64_dst.
// ============================================================================
static void EmitB64Interleave(codegen::CCECodegen& codegen, const std::string& b64Dst, const std::string& lo,
                              const std::string& hi, const std::string& prefix)
{
    std::string dump = prefix + "_idump_";
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, dump);
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + b64Dst + ", " + dump + ", (RegTensor<uint32_t>&)" + lo +
                 ", (RegTensor<uint32_t>&)" + hi + ");");
}

// ============================================================================
// B64 helper: emit a b64 ZEROING mask application.
// Emits vsel(dst, src, zero_b64, mask) to zero out inactive b64 elements.
// The b64 mask is packed to b32 via ppack, then expanded via pintlv_b32
// (mirrors AscendC CopyMerging: ppack + pintlv_b32 to produce a b32 mask
// where each b64 element's 2 b32 halves share the same mask bit).
// ============================================================================
static void EmitB64Zeroing(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src,
                           const std::string& mask, const std::string& prefix)
{
    std::string zero = prefix + "_zero64_";
    std::string m = prefix + "_zm_";
    std::string packedM = prefix + "_pm_";
    std::string expandedM = prefix + "_em_";
    std::string dumpM = prefix + "_dm_";
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, zero);
    codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
    codegen.Emit("vdup(" + zero + ", 0, " + m + ", MODE_ZEROING);");
    // Pack b64 mask to b32, then expand back to b32 width (mirrors AscendC
    // CopyMerging: ppack + pintlv_b32)
    codegen.Emit("MaskReg " + packedM + ";");
    codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
    codegen.Emit("MaskReg " + expandedM + ", " + dumpM + ";");
    codegen.Emit("pintlv_b32(" + expandedM + ", " + dumpM + ", " + packedM + ", " + packedM + ");");
    codegen.Emit("vsel((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + src + ", " + zero + ", " +
                 expandedM + ");");
}

// ============================================================================
// B64 helper: emit a b32 bitwise instruction over b64 registers with a b64
// mask (mirrors AscendC AndImpl/XorImpl/OrImpl/NotImpl b64 path): pack the b64
// mask (2-bit-per-element) to a b32 mask via ppack, deinterleave each operand
// into low/high u32 halves, apply the b32 bitwise instruction on both halves
// with the packed mask, then interleave the halves back into the b64 dst.
// MODE_ZEROING zeroes both halves of an inactive element, matching b64 zeroing
// semantics. Bitwise ops are sign-agnostic, so u32 halves are used for both
// INT64 and UINT64.
// ============================================================================
static void EmitB64Bitwise(codegen::CCECodegen& codegen, const std::string& instruction,
                           const std::vector<std::string>& srcs, const std::string& dst, const std::string& mask,
                           const std::string& mode, const std::string& prefix)
{
    std::string packedM = prefix + "_pm_";
    codegen.Emit("MaskReg " + packedM + ";");
    codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
    std::string loArgs;
    std::string hiArgs;
    for (size_t i = 0; i < srcs.size(); ++i) {
        EmitB64Deinterleave(codegen, prefix + "s" + std::to_string(i), srcs[i]);
        loArgs += prefix + "s" + std::to_string(i) + "_lo_, ";
        hiArgs += prefix + "s" + std::to_string(i) + "_hi_, ";
    }
    std::string loD = prefix + "_lod_";
    std::string hiD = prefix + "_hid_";
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
    codegen.Emit(instruction + "(" + loD + ", " + loArgs + packedM + ", " + mode + ");");
    codegen.Emit(instruction + "(" + hiD + ", " + hiArgs + packedM + ", " + mode + ");");
    EmitB64Interleave(codegen, dst, loD, hiD, prefix + "_ilv");
}

// ============================================================================
// B64 helper: reduce SUM via the hierarchical 16-bit carry decomposition
// (mirrors AscendC ReduceSumB64Impl). The b64 src is deinterleaved into
// low/high u32 halves; each 64-bit element is split into 16-bit lanes whose
// vcadd partial sums are combined with carries.
// ============================================================================
static void EmitB64ReduceSum(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src,
                             const std::string& mask)
{
    const std::string p = dst + "_b64rs_";
    const std::string pm = p + "pm";
    codegen.Emit("MaskReg " + pm + ";");
    codegen.Emit("ppack(" + pm + ", " + mask + ", LOWER);");
    EmitB64Deinterleave(codegen, p + "s", src);
    const std::string loS = p + "s_lo_";
    const std::string hiS = p + "s_hi_";
    const std::string lowF = p + "lowf_";
    const std::string lowR = p + "lowr_";
    const std::string midR = p + "midr_";
    const std::string hiR = p + "hir_";
    const std::string tmpR = p + "tmpr_";
    const std::string lo32 = p + "lo32_";
    const std::string dump = p + "dump_";
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, lowF);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, lowR);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, midR);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiR);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, tmpR);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, lo32);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, dump);
    codegen.Emit("vdup(" + lowF + ", (int32_t)0xFFFF, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vand(" + lowR + ", " + lowF + ", " + loS + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vcadd(" + lowR + ", " + lowR + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vshrs(" + midR + ", " + loS + ", (int16_t)16, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vcadd(" + midR + ", " + midR + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vcadd(" + hiR + ", " + hiS + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vshrs(" + tmpR + ", " + lowR + ", (int16_t)16, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vadd(" + midR + ", " + midR + ", " + tmpR + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vshrs(" + tmpR + ", " + midR + ", (int16_t)16, " + pm + ", MODE_ZEROING);");
    codegen.Emit("vadd(" + hiR + ", " + hiR + ", " + tmpR + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vand(" + lowR + ", " + lowR + ", " + lowF + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vand(" + midR + ", " + midR + ", " + lowF + ", " + pm + ", MODE_ZEROING);");
    codegen.Emit("vintlv((RegTensor<uint16_t>&)" + lo32 + ", (RegTensor<uint16_t>&)" + tmpR +
                 ", (RegTensor<uint16_t>&)" + lowR + ", (RegTensor<uint16_t>&)" + midR + ");");
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + dump +
                 ", (RegTensor<uint32_t>&)" + lo32 + ", (RegTensor<uint32_t>&)" + hiR + ");");
}

static void EmitB64ReduceMaxMin(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src,
                                const std::string& mask, bool isMax, bool isSigned)
{
    const std::string p = dst + "_b64rm_";
    const std::string pm = p + "pm";
    codegen.Emit("MaskReg " + pm + ";");
    codegen.Emit("ppack(" + pm + ", " + mask + ", LOWER);");
    EmitB64Deinterleave(codegen, p + "s", src);
    const std::string loS = p + "s_lo_";
    const std::string hiS = p + "s_hi_";
    const std::string red = isMax ? "vcmax" : "vcmin";
    const std::string hiR = p + "hir_";
    const std::string bcast = p + "bc_";
    const std::string eqM = p + "eqm_";
    const std::string loR = p + "lor_";
    const std::string dump = p + "dump_";
    codegen.Emit("MaskReg " + eqM + ";");
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiR);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, bcast);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, loR);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, dump);
    // The hi half holds the SIGNED b32 view of an int64 element: the hi-half
    // reduce must compare signed for INT64 (a negative int64 has a hi half
    // that is huge when viewed as u32; mirrors Int64RowMinMax's s32/u32
    // split). The lo half is an unsigned magnitude in both cases.
    const std::string hiSc = isSigned ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
    codegen.Emit(red + "(" + hiSc + hiR + ", " + hiSc + hiS + ", " + pm + ", MODE_ZEROING);");
    // Register broadcast uses the 5-arg vdup (mask + POS + MODE); the 4-arg
    // form only accepts a scalar (mirrors Int64RowMinMax's highDup broadcast).
    codegen.Emit("vdup(" + bcast + ", " + hiR + ", " + pm + ", POS_LOWEST, MODE_ZEROING);");
    codegen.Emit("vcmp_eq(" + eqM + ", " + bcast + ", " + hiS + ", " + pm + ");");
    codegen.Emit(red + "(" + loR + ", " + loS + ", " + eqM + ", MODE_ZEROING);");
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + dump +
                 ", (RegTensor<uint32_t>&)" + loR + ", (RegTensor<uint32_t>&)" + hiR + ");");
}

static void EmitB64Div(codegen::CCECodegen& codegen, const std::string& dst, const std::string& src0,
                       const std::string& src1, const std::string& mask, bool isSigned)
{
    const std::string p = dst + "_div_";
    const std::string allM = p + "_allm_";
    const std::string pm = p + "_pm_";
    const std::string cy = p + "_cy_";
    const std::string cy1 = p + "_cy1_";
    codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
    codegen.Emit("MaskReg " + pm + ";");
    codegen.Emit("MaskReg " + cy + ";");
    codegen.Emit("MaskReg " + cy1 + ";");
    codegen.Emit("ppack(" + pm + ", " + mask + ", LOWER);");
    // Deinterleave the b64 sources into lo/hi u32 half pairs (each half has 64
    // lanes: 32 meaningful + 32 zeros from the deinterleave zero operand).
    const std::string s0 = p + "s0", s0Lo = s0 + "_lo", s0Hi = s0 + "_hi";
    const std::string s1 = p + "s1", s1Lo = s1 + "_lo", s1Hi = s1 + "_hi";
    const std::string z = p + "z", zLo = z + "_lo", zHi = z + "_hi";
    const std::string dz = p + "_dz";
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, s0Lo);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, s0Hi);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, s1Lo);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, s1Hi);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, zLo);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, zHi);
    EmitRtWrapDecl(codegen, ir::DataType::UINT32, dz);
    codegen.Emit("vdup(" + zLo + ", 0, " + allM + ", MODE_ZEROING);");
    codegen.Emit("vdup(" + zHi + ", 0, " + allM + ", MODE_ZEROING);");
    codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + s0Lo + ", (RegTensor<uint32_t>&)" + s0Hi +
                 ", (RegTensor<uint32_t>&)" + src0 + ", (RegTensor<uint32_t>&)" + zLo + ");");
    codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + s1Lo + ", (RegTensor<uint32_t>&)" + s1Hi +
                 ", (RegTensor<uint32_t>&)" + src1 + ", (RegTensor<uint32_t>&)" + zHi + ");");
    // Composite b64 helper emissions over {x_lo, x_hi} u32 half pairs.
    auto VmulUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + d + "_lo, (RegTensor<uint32_t>&)" + d +
                     "_hi, (RegTensor<uint32_t>&)" + a + "_lo, (RegTensor<uint32_t>&)" + b + "_lo, " + m + ");");
        codegen.Emit("vmula((RegTensor<uint32_t>&)" + d + "_hi, (RegTensor<uint32_t>&)" + a +
                     "_lo, (RegTensor<uint32_t>&)" + b + "_hi, " + m + ", MODE_ZEROING);");
        codegen.Emit("vmula((RegTensor<uint32_t>&)" + d + "_hi, (RegTensor<uint32_t>&)" + a +
                     "_hi, (RegTensor<uint32_t>&)" + b + "_lo, " + m + ", MODE_ZEROING);");
    };
    auto VnotInPlace = [&](const std::string& x, const std::string& m) {
        codegen.Emit("vnot((RegTensor<uint32_t>&)" + x + "_lo, (RegTensor<uint32_t>&)" + x + "_lo, " + m +
                     ", MODE_ZEROING);");
        codegen.Emit("vnot((RegTensor<uint32_t>&)" + x + "_hi, (RegTensor<uint32_t>&)" + x + "_hi, " + m +
                     ", MODE_ZEROING);");
    };
    auto AddB64 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vaddc(" + cy + ", " + d + "_lo, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + d + "_hi, " + a + "_hi, " + b + "_hi, " + cy + ", " + m + ");");
    };
    auto VsubUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vsubc(" + cy + ", " + d + "_lo, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vsubcs(" + cy + ", " + d + "_hi, " + a + "_hi, " + b + "_hi, " + cy + ", " + m + ");");
    };
    auto VaddUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vaddc(" + cy + ", " + d + "_lo, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + d + "_hi, " + a + "_hi, " + b + "_hi, " + cy + ", " + m + ");");
    };
    auto VselUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vsel((RegTensor<uint32_t>&)" + d + "_lo, (RegTensor<uint32_t>&)" + a +
                     "_lo, (RegTensor<uint32_t>&)" + b + "_lo, " + m + ");");
        codegen.Emit("vsel((RegTensor<uint32_t>&)" + d + "_hi, (RegTensor<uint32_t>&)" + a +
                     "_hi, (RegTensor<uint32_t>&)" + b + "_hi, " + m + ");");
    };
    auto VcmpEqUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vcmp_eq(" + d + "_l, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vcmp_eq(" + d + "_h, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("pand(" + d + ", " + d + "_l, " + d + "_h, " + m + ");");
    };
    auto VcmpGeUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vcmp_eq(" + d + "_heq, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("vcmp_ge(" + d + "_lge, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vcmp_ge(" + d + "_hge, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("psel(" + d + ", " + d + "_lge, " + d + "_hge, " + d + "_heq);");
    };
    auto VcmpLtUsingU32 = [&](const std::string& d, const std::string& a, const std::string& b, const std::string& m) {
        codegen.Emit("vcmp_eq(" + d + "_heq, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("vcmp_lt(" + d + "_llt, " + a + "_lo, " + b + "_lo, " + m + ");");
        codegen.Emit("vcmp_lt(" + d + "_hlt, " + a + "_hi, " + b + "_hi, " + m + ");");
        codegen.Emit("psel(" + d + ", " + d + "_llt, " + d + "_hlt, " + d + "_heq);");
    };
    auto VbrU32 = [&](const std::string& d, const std::string& hiVal, const std::string& loVal) {
        codegen.Emit("vdup(" + d + "_hi, (int32_t)(" + hiVal + "), " + allM + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + d + "_lo, (int32_t)(" + loVal + "), " + allM + ", MODE_ZEROING);");
    };
    auto DeclPair = [&](const std::string& base) {
        codegen.Emit("RtWrap<uint32_t> " + base + "_lo;");
        codegen.Emit("RtWrap<uint32_t> " + base + "_hi;");
    };
    auto DeclU32 = [&](const std::string& name) { EmitRtWrapDecl(codegen, ir::DataType::UINT32, name); };
    // B128Calc: vb = high64(va * vb + s) (mirrors AscendC B128Calc; vb is both
    // read and written, va and s are read-only). Uses two carry masks (mirrors
    // the carry0/carry1 pair in the reference: the second vaddc's carry-out
    // must not clobber the first, which is still consumed by the next vaddcs).
    auto B128Calc = [&](const std::string& tag, const std::string& va, const std::string& vb, const std::string& s,
                        const std::string& m) {
        const std::string m0l = tag + "m0l", m0h = tag + "m0h", m1l = tag + "m1l", m1h = tag + "m1h";
        const std::string m2l = tag + "m2l", m2h = tag + "m2h", m3l = tag + "m3l", m3h = tag + "m3h";
        const std::string dd0 = tag + "dd0", dd1 = tag + "dd1";
        DeclU32(m0l);
        DeclU32(m0h);
        DeclU32(m1l);
        DeclU32(m1h);
        DeclU32(m2l);
        DeclU32(m2h);
        DeclU32(m3l);
        DeclU32(m3h);
        DeclU32(dd0);
        DeclU32(dd1);
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m0l + ", (RegTensor<uint32_t>&)" + m0h +
                     ", (RegTensor<uint32_t>&)" + va + "_lo, (RegTensor<uint32_t>&)" + vb + "_lo, " + m + ");");
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m1l + ", (RegTensor<uint32_t>&)" + m1h +
                     ", (RegTensor<uint32_t>&)" + va + "_lo, (RegTensor<uint32_t>&)" + vb + "_hi, " + m + ");");
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m2l + ", (RegTensor<uint32_t>&)" + m2h +
                     ", (RegTensor<uint32_t>&)" + va + "_hi, (RegTensor<uint32_t>&)" + vb + "_lo, " + m + ");");
        codegen.Emit("vmull((RegTensor<uint32_t>&)" + m3l + ", (RegTensor<uint32_t>&)" + m3h +
                     ", (RegTensor<uint32_t>&)" + va + "_hi, (RegTensor<uint32_t>&)" + vb + "_hi, " + m + ");");
        codegen.Emit("vaddc(" + cy + ", " + dd0 + ", " + m0h + ", " + m1l + ", " + m + ");");
        codegen.Emit("vaddc(" + cy1 + ", " + dd1 + ", " + dd0 + ", " + m2l + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + dd0 + ", " + m3l + ", " + m1h + ", " + cy + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy1 + ", " + vb + "_lo, " + dd0 + ", " + m2h + ", " + cy1 + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + dd0 + ", " + s + ", " + m3h + ", " + cy + ", " + m + ");");
        codegen.Emit("vaddcs(" + cy + ", " + vb + "_hi, " + s + ", " + dd0 + ", " + cy1 + ", " + m + ");");
    };
    // Abs of src0/src1 via hi-half sign compare + borrow-chain neg + vsel
    // (mirrors AbsUsingS32). Unsigned passes through unchanged.
    const std::string abs0 = p + "abs0", abs1 = p + "abs1";
    const std::string abs0Lo = abs0 + "_lo", abs0Hi = abs0 + "_hi";
    const std::string abs1Lo = abs1 + "_lo", abs1Hi = abs1 + "_hi";
    DeclPair(abs0);
    DeclPair(abs1);
    const std::string ac0 = p + "ac0", ac1 = p + "ac1", ac2 = p + "ac2";
    const std::string at20 = p + "at20", at30 = p + "at30";
    const std::string bc0 = p + "bc0", bc1 = p + "bc1", bc2 = p + "bc2";
    const std::string bt20 = p + "bt20", bt30 = p + "bt30";
    codegen.Emit("MaskReg " + ac0 + ";");
    codegen.Emit("MaskReg " + ac1 + ";");
    codegen.Emit("MaskReg " + ac2 + ";");
    EmitRtWrapDecl(codegen, ir::DataType::INT32, at20);
    EmitRtWrapDecl(codegen, ir::DataType::INT32, at30);
    codegen.Emit("MaskReg " + bc0 + ";");
    codegen.Emit("MaskReg " + bc1 + ";");
    codegen.Emit("MaskReg " + bc2 + ";");
    EmitRtWrapDecl(codegen, ir::DataType::INT32, bt20);
    EmitRtWrapDecl(codegen, ir::DataType::INT32, bt30);
    if (isSigned) {
        codegen.Emit("vcmp_lt(" + p + "ac0, (RegTensor<int32_t>&)" + s0Hi + ", (RegTensor<int32_t>&)" + zHi + ", " +
                     pm + ");");
        codegen.Emit("vsubc(" + p + "ac1, (RegTensor<int32_t>&)" + p + "at20, (RegTensor<int32_t>&)" + zLo +
                     ", (RegTensor<int32_t>&)" + s0Lo + ", " + p + "ac0);");
        codegen.Emit("vsubcs(" + p + "ac2, (RegTensor<int32_t>&)" + p + "at30, (RegTensor<int32_t>&)" + zHi +
                     ", (RegTensor<int32_t>&)" + s0Hi + ", " + p + "ac1, " + pm + ");");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs0Lo + ", (RegTensor<int32_t>&)" + p +
                     "at20, (RegTensor<int32_t>&)" + s0Lo + ", " + p + "ac0);");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs0Hi + ", (RegTensor<int32_t>&)" + p +
                     "at30, (RegTensor<int32_t>&)" + s0Hi + ", " + p + "ac0);");
        codegen.Emit("vcmp_lt(" + p + "bc0, (RegTensor<int32_t>&)" + s1Hi + ", (RegTensor<int32_t>&)" + zHi + ", " +
                     pm + ");");
        codegen.Emit("vsubc(" + p + "bc1, (RegTensor<int32_t>&)" + p + "bt20, (RegTensor<int32_t>&)" + zLo +
                     ", (RegTensor<int32_t>&)" + s1Lo + ", " + p + "bc0);");
        codegen.Emit("vsubcs(" + p + "bc2, (RegTensor<int32_t>&)" + p + "bt30, (RegTensor<int32_t>&)" + zHi +
                     ", (RegTensor<int32_t>&)" + s1Hi + ", " + p + "bc1, " + pm + ");");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs1Lo + ", (RegTensor<int32_t>&)" + p +
                     "bt20, (RegTensor<int32_t>&)" + s1Lo + ", " + p + "bc0);");
        codegen.Emit("vsel((RegTensor<int32_t>&)" + abs1Hi + ", (RegTensor<int32_t>&)" + p +
                     "bt30, (RegTensor<int32_t>&)" + s1Hi + ", " + p + "bc0);");
    } else {
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs0Lo + ", (RegTensor<int32_t>&)" + s0Lo + ");");
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs0Hi + ", (RegTensor<int32_t>&)" + s0Hi + ");");
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs1Lo + ", (RegTensor<int32_t>&)" + s1Lo + ");");
        codegen.Emit("vmov((RegTensor<int32_t>&)" + abs1Hi + ", (RegTensor<int32_t>&)" + s1Hi + ");");
    }
    // 64-bit constants: qZero = all-ones (div-by-zero result); one = 1.
    const std::string qz = p + "qz", c1 = p + "c1";
    DeclU32(qz + "_lo");
    DeclU32(qz + "_hi");
    DeclU32(c1 + "_lo");
    DeclU32(c1 + "_hi");
    VbrU32(qz, "-1", "-1");
    VbrU32(c1, "0", "1");
    // zeroMask: divisor == 0 (VcmpEqUsingU32 on both halves).
    const std::string zeroM = p + "zm", nonzeroM = p + "nzm";
    const std::string oneM = p + "om", nononeM = p + "nom";
    codegen.Emit("MaskReg " + zeroM + ", " + zeroM + "_l, " + zeroM + "_h;");
    codegen.Emit("MaskReg " + nonzeroM + ";");
    codegen.Emit("MaskReg " + oneM + ", " + oneM + "_l, " + oneM + "_h;");
    codegen.Emit("MaskReg " + nononeM + ";");
    VcmpEqUsingU32(zeroM, s1, z, pm);
    codegen.Emit("pnot(" + nonzeroM + ", " + zeroM + ", " + pm + ");");
    // oneMask: abs(divisor) == 1.
    VcmpEqUsingU32(oneM, abs1, c1, pm);
    codegen.Emit("pnot(" + nononeM + ", " + oneM + ", " + pm + ");");
    // Newton work mask + (unsigned-only) pre-checks (mirrors DivU64Impl):
    // divisor > INT64_MAX (hi half negative as int32) invalidates the f32
    // reciprocal path; the quotient is 1 (src0 >= src1) or 0 (src0 < src1).
    // usrc1_m keeps the "large divisor" meaning until the final bypass select,
    // so the work-mask construction uses a separate notlarge_m mask.
    const std::string workM = p + "wm", usrc1M = p + "u1m", srccmpM = p + "scm", cmpdivM = p + "cdm";
    const std::string ge0 = p + "ge0";
    const std::string notlargeM = p + "nlm";
    const std::string q = p + "q", qLo = q + "_lo", qHi = q + "_hi";
    codegen.Emit("MaskReg " + workM + ";");
    codegen.Emit("MaskReg " + usrc1M + ";");
    codegen.Emit("MaskReg " + srccmpM + ";");
    codegen.Emit("MaskReg " + cmpdivM + ";");
    codegen.Emit("MaskReg " + ge0 + ";");
    codegen.Emit("MaskReg " + notlargeM + ";");
    // Temp masks written by VcmpGeUsingU32/VcmpLtUsingU32 (shared across call
    // sites, declared once here).
    codegen.Emit("MaskReg " + srccmpM + "_heq, " + srccmpM + "_lge, " + srccmpM + "_hge;");
    codegen.Emit("MaskReg " + srccmpM + "_llt, " + srccmpM + "_hlt;");
    codegen.Emit("MaskReg " + ge0 + "_heq, " + ge0 + "_lge, " + ge0 + "_hge;");
    DeclU32(qLo);
    DeclU32(qHi);
    if (!isSigned) {
        codegen.Emit("vcmp_lt(" + usrc1M + ", (RegTensor<int32_t>&)" + s1Hi + ", (RegTensor<int32_t>&)" + zHi + ", " +
                     pm + ");");
        VcmpGeUsingU32(srccmpM, s0, s1, pm);
        codegen.Emit("pand(" + cmpdivM + ", " + srccmpM + ", " + usrc1M + ", " + pm + ");");
        VselUsingU32(q, c1, z, cmpdivM);
        VcmpLtUsingU32(srccmpM, s0, s1, pm);
        codegen.Emit("pand(" + cmpdivM + ", " + srccmpM + ", " + usrc1M + ", " + pm + ");");
        VselUsingU32(q, z, c1, cmpdivM);
        codegen.Emit("vcmp_ge(" + notlargeM + ", (RegTensor<int32_t>&)" + s1Hi + ", (RegTensor<int32_t>&)" + zHi +
                     ", " + pm + ");");
        codegen.Emit("pand(" + workM + ", " + nonzeroM + ", " + notlargeM + ", " + pm + ");");
        codegen.Emit("pand(" + workM + ", " + workM + ", " + nononeM + ", " + pm + ");");
    } else {
        codegen.Emit("pand(" + workM + ", " + nononeM + ", " + nonzeroM + ", " + pm + ");");
    }
    // Newton-Raphson reciprocal refinement.
    const std::string t2 = p + "t2f", t3 = p + "t3f", t4 = p + "t4u";
    const std::string t2o = p + "t2o", t2c = p + "t2c", t2d = p + "t2d";
    const std::string prl = p + "prl", prh = p + "prh", cf0 = p + "cf0", cf1 = p + "cf1";
    const std::string ci0 = p + "ci0", ci1 = p + "ci1";
    const std::string t5 = p + "t5", t6 = p + "t6", t7 = p + "t7", t8 = p + "t8", t9 = p + "t9";
    const std::string d0 = p + "d0", d1 = p + "d1";
    const std::string t5Lo = t5 + "_lo", t5Hi = t5 + "_hi";
    const std::string t6Lo = t6 + "_lo", t6Hi = t6 + "_hi";
    EmitRtWrapDecl(codegen, ir::DataType::FP32, t2);
    EmitRtWrapDecl(codegen, ir::DataType::FP32, t3);
    EmitRtWrapDecl(codegen, ir::DataType::FP32, t2o);
    EmitRtWrapDecl(codegen, ir::DataType::FP32, t2c);
    EmitRtWrapDecl(codegen, ir::DataType::FP32, t2d);
    DeclU32(t4);
    // prl/prh receive the interleaved b64 divisor (s64 view), cf0/cf1 the
    // split f32 halves, ci0/ci1 the converted s64 reciprocal (mirrors
    // Int64ToFloat / FloatToInt64 in the reference).
    EmitRtWrapDecl(codegen, ir::DataType::INT64, prl);
    EmitRtWrapDecl(codegen, ir::DataType::INT64, prh);
    EmitRtWrapDecl(codegen, ir::DataType::FP32, cf0);
    EmitRtWrapDecl(codegen, ir::DataType::FP32, cf1);
    EmitRtWrapDecl(codegen, ir::DataType::INT64, ci0);
    EmitRtWrapDecl(codegen, ir::DataType::INT64, ci1);
    DeclPair(t5);
    DeclPair(t6);
    DeclPair(t7);
    DeclPair(t8);
    DeclPair(t9);
    DeclU32(d0);
    DeclU32(d1);
    // VcvtS642F32(t2c, abs1): re-interleave the halves to b64, convert to f32,
    // then compact. The s64->f32 vcvt writes its 32 results to the EVEN f32
    // lanes only, so the converted values must be gathered to consecutive
    // lanes 0-31 with a vdintlv before the divide (mirrors Int64ToFloat's
    // vcvt + vdintlv tail).
    // vintlv must use a single element view for all four arguments: the s32
    // v64 overload takes vector_s32 refs for the dst pair and vector_s32
    // values for the lo/hi sources (mirrors Int64ToFloat's (vector_s32&)
    // casts; mixing s32 dsts with u32 srcs matches no overload).
    codegen.Emit("vintlv((RegTensor<int32_t>&)" + prl + ", (RegTensor<int32_t>&)" + prh + ", (RegTensor<int32_t>&)" +
                 abs1Lo + ", (RegTensor<int32_t>&)" + abs1Hi + ");");
    codegen.Emit("vcvt(" + t2 + ", " + prl + ", " + allM + ", ROUND_R, PART_EVEN);");
    codegen.Emit("vcvt(" + t2o + ", " + prh + ", " + allM + ", ROUND_R, PART_EVEN);");
    codegen.Emit("vdintlv((RegTensor<float>&)" + t2c + ", (RegTensor<float>&)" + t2d + ", (RegTensor<float>&)" + t2 +
                 ", (RegTensor<float>&)" + t2o + ");");
    // F32PreProcess(t4, t3, t2c): 1.0 / divisor_f32, then the f32 exponent bias.
    codegen.Emit("vdup(" + t3 + ", 1.0f, " + allM + ", MODE_ZEROING);");
    codegen.Emit("vdiv(" + t3 + ", " + t3 + ", " + t2c + ", " + workM + ", MODE_ZEROING);");
    codegen.Emit("vadds(" + t4 + ", (RegTensor<uint32_t>&)" + t3 + ", (int32_t)0x1FFFFFFE, " + workM + ");");
    // VcvtF322S64(t5, t4): f32 -> s64 (the 6-arg overload: ROUND + RS + PART).
    codegen.Emit("vintlv(" + cf0 + ", " + cf1 + ", (RegTensor<float>&)" + t4 + ", (RegTensor<float>&)" + t4 + ");");
    codegen.Emit("vcvt(" + ci0 + ", " + cf0 + ", " + allM + ", ROUND_Z, RS_DISABLE, PART_EVEN);");
    codegen.Emit("vcvt(" + ci1 + ", " + cf1 + ", " + allM + ", ROUND_Z, RS_DISABLE, PART_EVEN);");
    codegen.Emit("vdintlv((RegTensor<int32_t>&)" + t5Lo + ", (RegTensor<int32_t>&)" + t5Hi + ", (RegTensor<int32_t>&)" +
                 ci0 + ", (RegTensor<int32_t>&)" + ci1 + ");");
    // Newton refinement: t6 = -(abs1 * t5) + 1.
    VmulUsingU32(t6, abs1, t5, workM);
    VnotInPlace(t6, workM);
    AddB64(t6, t6, c1, workM);
    codegen.Emit("vdup(" + t4 + ", 0, " + allM + ", MODE_ZEROING);");
    // Refinement rounds via B128Calc (128-bit square/product high parts).
    B128Calc(p + "r1", t5, t6, t4, workM);
    VaddUsingU32(t7, t5, t6, workM);
    VmulUsingU32(t6, abs1, t7, workM);
    VnotInPlace(t6, workM);
    AddB64(t6, t6, c1, workM);
    B128Calc(p + "r2", t7, t6, t4, workM);
    VaddUsingU32(t6, t7, t6, workM);
    B128Calc(p + "r3", abs0, t6, t4, workM);
    // Final correction: two conditional subtract + increment rounds.
    VmulUsingU32(t7, t6, abs1, workM);
    VsubUsingU32(t7, abs0, t7, workM);
    VcmpGeUsingU32(ge0, t7, abs1, workM);
    VsubUsingU32(t8, t7, abs1, ge0);
    AddB64(t9, t6, c1, ge0);
    VselUsingU32(t7, t8, t7, ge0);
    VselUsingU32(t6, t9, t6, ge0);
    VcmpGeUsingU32(ge0, t7, abs1, workM);
    AddB64(t9, t6, c1, ge0);
    VselUsingU32(t6, t9, t6, ge0);
    VselUsingU32(t6, abs0, t6, oneM);
    if (!isSigned) {
        VselUsingU32(t6, q, t6, usrc1M);
    }
    // DivSignCal + negate the quotient when the operand signs differ (signed).
    // The negation (0 - q) runs under the full mask (mirrors
    // Int64DivSignedRestoreSign: the negate happens on every lane and the
    // following vsel picks the original quotient on same-sign lanes).
    if (isSigned) {
        const std::string s0geM = p + "s0ge", s1geM = p + "s1ge", signM = p + "sign";
        codegen.Emit("MaskReg " + s0geM + ", " + s1geM + ", " + signM + ";");
        codegen.Emit("vcmp_ge(" + s0geM + ", (RegTensor<int32_t>&)" + s0Hi + ", (RegTensor<int32_t>&)" + zLo + ", " +
                     pm + ");");
        codegen.Emit("vcmp_ge(" + s1geM + ", (RegTensor<int32_t>&)" + s1Hi + ", (RegTensor<int32_t>&)" + zLo + ", " +
                     pm + ");");
        codegen.Emit("pxor(" + signM + ", " + s0geM + ", " + s1geM + ", " + pm + ");");
        codegen.Emit("pnot(" + signM + ", " + signM + ", " + pm + ");");
        VsubUsingU32(t8, z, t6, allM);
        VselUsingU32(t6, t6, t8, signM);
    }
    // Divisor == 0 -> all-ones quotient; assemble the interleaved dst.
    VselUsingU32(t6, qz, t6, zeroM);
    codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + d1 + ", (RegTensor<uint32_t>&)" +
                 t6Lo + ", (RegTensor<uint32_t>&)" + t6Hi + ");");
}

// Check if a DataType lacks a direct vdup/vlds/vsts intrinsic overload and
// must be reinterpreted as uint8_t. The bisheng __VF_VDUP/__VF_VLDS/__VF_VSTS
// macros only instantiate overloads for u8/s8/u16/s16/f16/u32/s32/f32/bf16/
// f8e4m3/f8e5m2/f8e8m0/f4e2m1x2/f4e1m2x2 and b64 types. FP4/HF4 (b4 packed as
// b8) and HF8 (b8) have no overload, but share the same physical b8 register
// layout as u8.
static bool NeedsB8Reinterpret(DataType dt)
{
    return dt == DataType::FP4 || dt == DataType::FP4E2M1 || dt == DataType::FP4E1M2 || dt == DataType::HF4 ||
           dt == DataType::HF8 || dt == DataType::INT4 || dt == DataType::UINT4;
}

// Map a DataType to the correct C pointer type for __ubuf__ load/store.
// VF load/store intrinsics (vlds/vsts/vld/vst/vsldb/vsstb) accept both
// signed and unsigned pointer types, so we use the native C type to match
// the RegTensor element type and avoid type-mismatch errors.
// B64 types (INT64/UINT64) are stored as pairs of 32-bit halves, so the
// pointer type is the 32-bit half type for non-B64-overload paths.
// (B64 vlds/vsts overloads override the pointer type via GetB64PtrType.)
// Types lacking a direct bisheng overload (FP4/HF4/HF8/INT4/UINT4) are
// reinterpreted as uint8_t.
static std::string DtypeToPtrType(DataType dt)
{
    if (NeedsB8Reinterpret(dt))
        return "uint8_t";
    if (dt == DataType::INT64)
        return "int32_t";
    if (dt == DataType::UINT64)
        return "uint32_t";
    return dt.ToCTypeString();
}

// Return the (RegTensor<uint8_t>&) cast prefix for types that need B8 reinterpret.
static std::string GetB8Cast(DataType dt) { return NeedsB8Reinterpret(dt) ? "(RegTensor<uint8_t>&)" : ""; }

// Resolve an offset argument to a C++ code string.
// If the argument is a 2-element MakeTuple [row, col], compute the linear
// offset `row * cols + col` using the tile's shape[1] (number of columns).
// Otherwise, emit the expression directly (integer offset, AddrReg, etc.).
// Returns empty string if the argument is not an offset (caller should skip).
static std::string ResolveOffsetArg(codegen::CCECodegen& codegen, const ir::ExprPtr& offsetExpr,
                                    const ir::ExprPtr& tileExpr)
{
    if (auto tuple = ir::As<ir::MakeTuple>(offsetExpr)) {
        if (tuple->elements_.size() == 2) {
            std::string rowStr = codegen.GetExprAsCode(tuple->elements_[0]);
            std::string colStr = codegen.GetExprAsCode(tuple->elements_[1]);
            // Get cols from tile shape[1]
            std::string colsStr = "1";
            if (auto tileType = ir::As<ir::TileType>(tileExpr->GetType())) {
                if (tileType->shape_.size() >= 2) {
                    colsStr = codegen.GetExprAsCode(tileType->shape_[1]);
                }
            }
            return "((" + rowStr + ") * (" + colsStr + ") + (" + colStr + "))";
        }
    }
    return codegen.GetExprAsCode(offsetExpr);
}

// ============================================================================
// CreateMask — declares MaskReg + emits VF init instruction
// ============================================================================

// For b64 element width (INT64/UINT64) on a single-register trait, the hardware
// interprets mask bits at 2-bit-per-element granularity. Since pset_b32 produces
// 1-bit-per-b32-element masks, we must:
//   1. Remap H/Q patterns to VL16/VL8 (punpack doubles the bit count, so the
//      source pattern must be halved to compensate).
//   2. Emit punpack(reg, reg, LOWER) after pset_b32 to expand each bit into a
//      pair, matching the 2-bit-per-b64-element mask width.
// This mirrors AscendC CreateMaskImpl<T, mode, RegTraitNumOne> for sizeof(T)==8.
static ir::MaskPattern RemapB64MaskPattern(ir::MaskPattern pattern)
{
    switch (pattern) {
        case ir::MaskPattern::H:
            return ir::MaskPattern::VL16;
        case ir::MaskPattern::Q:
            return ir::MaskPattern::VL8;
        default:
            return pattern;
    }
}

static std::string EmitVFCreateMask(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // pattern defaults to ALL, dtype defaults to FP32 — either kwarg may be omitted.
    auto pattern = op->HasKwarg("pattern") ? static_cast<ir::MaskPattern>(op->GetKwarg<int>("pattern")) :
                                             ir::MaskPattern::ALL;
    auto dtype = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") : DataType::FP32;
    // MaskReg dtype must be b8/b16/b32/b64 — determines mask granularity
    // FP4 types (GetBit()==4) are b8 storage (packed 2-per-byte), treated as b8
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dtype) || dtype.GetBit() == 16 || dtype.GetBit() == 32 || dtype.GetBit() == 64)
        << "vf.create_mask dtype must be b8/b16/b32/b64, got " << DTypeStr(dtype);
    bool isB64 = (dtype.GetBit() == 64);
    if (isB64) {
        pattern = RemapB64MaskPattern(pattern);
    }
    std::string regName = codegen.GetExprAsCode(op->args_[0]);
    // Declaration and name registration are handled by the value-form
    // synthesis (VisitExpr_(CallPtr)); only the pset initialization below.
    // Map the pattern enumerator to its CCE PAT_* constant. The names follow
    // the ir::MaskPattern declaration order (op_attr_types.h), so a static table
    // plus a bound check replaces the per-value switch while keeping the loud
    // failure for an out-of-enum (type-erased int) value.
    static const char* const kPatNames[] = {
        "PAT_ALL",  "PAT_ALLF", "PAT_VL1",   "PAT_VL2", "PAT_VL3", "PAT_VL4", "PAT_VL8", "PAT_VL16",
        "PAT_VL32", "PAT_VL64", "PAT_VL128", "PAT_M3",  "PAT_M4",  "PAT_H",   "PAT_Q",
    };
    const int patternIndex = static_cast<int>(pattern);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      patternIndex >= 0 && patternIndex < static_cast<int>(sizeof(kPatNames) / sizeof(kPatNames[0])))
        << "vf.create_mask pattern=" << patternIndex << " is not a valid ir::MaskPattern enumerator";
    std::string pat = kPatNames[patternIndex];

    // Select pset instruction based on data element size (not mask type)
    // float/int32 (4 bytes) → pset_b32, half/bf16 (2 bytes) → pset_b16, int8 (1 byte) → pset_b8
    // FP8/FP4 types are b8 storage → pset_b8
    // b64 (INT64/UINT64): pset_b32 + punpack(LOWER) — punpack expands each bit
    // into a pair to match the 2-bit-per-element mask width for 8-byte elements.
    if (IsB8Type(dtype)) {
        codegen.Emit(regName + " = pset_b8(" + pat + ");");
    } else if (dtype.GetBit() == 32 || isB64) {
        codegen.Emit(regName + " = pset_b32(" + pat + ");");
        if (isB64) {
            codegen.Emit("punpack(" + regName + ", " + regName + ", LOWER);");
        }
    } else {
        // FP16, BF16, UINT16, INT16 etc. (2 bytes)
        codegen.Emit(regName + " = pset_b16(" + pat + ");");
    }
    return "";
}

// ============================================================================
// Duplicate — scalar broadcast
// ============================================================================

static std::string EmitVFDuplicate(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, (optional) mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2 && op->args_.size() <= 3)
        << "vf.full requires 2-3 args (dst, src[, mask])";
    // vdup supports b8/b16/b32/b64 element widths (bool, int, float, FP8/FP4/HF8 types)
    DataType srcDt = GetExprDtype(op->args_[1], DataType::FP32);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(srcDt) || srcDt.GetBit() == 16 || srcDt.GetBit() == 32 || srcDt.GetBit() == 64))
        << "vf.full src only supports b8/b16/b32/b64 types, got " << DTypeStr(srcDt);
    // MERGING is not supported by the underlying vdup/vbr instructions on current device.
    VFZeroingOnly(op, "vf.full");
    // The fill width follows the DST. The Scalar-mode ``dtype`` kwarg is what
    // the frontend uses to declare dst (a bare int literal is INT64, so it
    // must not be compared against dst nor route a b16/b32 fill into the b64
    // split path); otherwise the assign target's dtype, then the src dtype.
    DataType dstDt = srcDt;
    if (op->HasKwarg("dtype")) {
        dstDt = op->GetKwarg<DataType>("dtype");
    } else if (const auto& t = op->args_[1]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string srcStr = codegen.GetExprAsCode(op->args_[1]);
    // Detect vector-source broadcast: either explicit pos kwarg, or src is a RegTensor variable.
    std::string pos = "";
    if (op->HasKwarg("pos")) {
        pos = VFEnumValueName(ir::EnumToString(static_cast<ir::DuplicatePos>(op->GetKwarg<int>("pos"))));
    }
    bool isVectorSrc = !pos.empty();
    if (!isVectorSrc) {
        isVectorSrc = ir::IsRegTensor(op->args_[1]->GetType());
    }
    if (isVectorSrc) {
        // Vector-source broadcast (Tensor mode): vdup(dst, src_vec, mask, POS_xxx, MODE)
        // pos kwarg: "LOWEST" -> POS_LOWEST, "HIGHEST" -> POS_HIGHEST
        if (pos.empty() || pos == "LOWEST")
            pos = "POS_LOWEST";
        else if (pos == "HIGHEST")
            pos = "POS_HIGHEST";
        std::string mode = VFZeroingOnly(op, "vf.full");
        // FP4/HF8/HF4/INT4/UINT4 lack a vdup overload — reinterpret as uint8_t
        std::string cast = GetB8Cast(srcDt);
        // Tensor mode always requires a mask (vdup(dstReg, srcReg, mask))
        if (op->args_.size() >= 3) {
            std::string mask = codegen.GetExprAsCode(op->args_[2]);
            codegen.Emit("vdup(" + cast + dst + ", " + cast + srcStr + ", " + mask + ", " + pos + ", " + mode + ");");
        } else {
            // No mask provided — create an ALL mask inline for Tensor mode
            static int dupMaskCounter = 0;
            std::string maskVar = "__dup_mask_" + std::to_string(dupMaskCounter++);
            std::string pat = "PAT_ALL";
            std::string psetFn = "pset_b32";
            if (IsB8Type(srcDt))
                psetFn = "pset_b8";
            else if (srcDt.GetBit() == 16)
                psetFn = "pset_b16";
            codegen.Emit("MaskReg " + maskVar + " = " + psetFn + "(" + pat + ");");
            codegen.Emit("vdup(" + cast + dst + ", " + cast + srcStr + ", " + maskVar + ", " + pos + ", " + mode +
                         ");");
        }
    } else if (op->args_.size() >= 3) {
        // Scalar broadcast with mask: vdup(dst, scalar, preg, MODE_ZEROING/MERGING)
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = VFZeroingOnly(op, "vf.full");
        // The fill width follows DST: an int literal is INT64 and must not route
        // a b16/b32 fill into the b64 split path.
        if (dstDt.GetBit() == 64) {
            // B64 has no vdup single-register overload; create two b32 halves,
            // vdup each with packed b32 mask, then vintlv into b64 dst
            // (mirrors AscendC DuplicateB64Impl + MaskPack).
            std::string p = dst + "_dup_";
            std::string packedM = p + "_pm_";
            codegen.Emit("MaskReg " + packedM + ";");
            codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
            std::string loD = p + "_lo_";
            std::string hiD = p + "_hi_";
            std::string castType = (dstDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
            codegen.Emit("vdup(" + loD + ", (uint32_t)(" + castType + "(" + srcStr + ")), " + packedM + ", " + mode +
                         ");");
            codegen.Emit("vdup(" + hiD + ", (uint32_t)((" + castType + "(" + srcStr + ")) >> 32), " + packedM + ", " +
                         mode + ");");
            EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
        } else {
            codegen.Emit("vdup(" + dst + ", " + srcStr + ", " + mask + ", " + mode + ");");
        }
    } else {
        // Scalar broadcast without mask: vbr(dst, scalar)
        if (dstDt.GetBit() == 64) {
            // B64 has no vbr single-register overload; create two b32 halves,
            // vbr each, then vintlv into b64 dst (mirrors AscendC DuplicateB64Impl).
            std::string p = dst + "_br_";
            std::string loD = p + "_lo_";
            std::string hiD = p + "_hi_";
            std::string castType = (dstDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
            codegen.Emit("vbr(" + loD + ", (uint32_t)(" + castType + "(" + srcStr + ")));");
            codegen.Emit("vbr(" + hiD + ", (uint32_t)((" + castType + "(" + srcStr + ")) >> 32));");
            EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
        } else {
            codegen.Emit("vbr(" + dst + ", " + srcStr + ");");
        }
    }
    return "";
}

// ============================================================================
// Helper: get __ubuf__ pointer from tile or tile-flavored GetItemExpr
// ============================================================================

static std::string GetUBufPtr(codegen::CCECodegen& codegen, const ir::ExprPtr& expr,
                              const std::string& castType = "float", bool isPostUpdate = false)
{
    std::string ptr = codegen.GetOrCreateVFTilePtr(expr, isPostUpdate);
    std::string tileCtype = GetExprDtype(expr, DataType::FP32).ToCTypeString();
    if (castType == tileCtype)
        return ptr;
    return "(__ubuf__ " + castType + " *)" + ptr;
}

// ============================================================================
// LoadAlign (unified) — vlds / plds with dist & post_update kwargs
// Replaces: LoadAlign, LoadAlignMode, LoadAlignPostUpdate, LoadAlignPostupdate, LoadAlignUnpackV2
// ============================================================================

static std::string EmitVFLoadAlign(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen — includes the dst register(s)).
    // 1-arg source form: load_align(ptr) — MaskReg dst → plds, RegTensor dst → vlds
    if (op->args_.size() == 2) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.load_align 1-arg form does not support data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.load_align 1-arg form does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.load_align 1-arg form does not support repeat_stride";
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        DataType dstDt = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") :
                                                 GetTypeDtype(op->args_[0]->GetType(), GetExprDtype(op->args_[1]));
        bool dstIsMask = ir::IsMaskReg(op->args_[0]->GetType());
        if (dstIsMask) {
            std::string mode = "NORM";
            if (op->HasKwarg("dist")) {
                mode = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/true, "vf.load_align");
                PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, mode == "NORM" || mode == "US" || mode == "DS")
                    << "vf.load_align 1-arg (MaskReg) only supports NORM/US/DS dist, got " << mode;
            }
            std::string pldsPtr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + pldsPtr + ", 0, " + mode + ");");
        } else {
            // RegTensor 1-arg form: vlds with hardcoded NORM, dist kwarg not supported
            if (op->HasKwarg("dist")) {
                auto distVal = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/true, "vf.load_align");
                PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, distVal == "NORM")
                    << "vf.load_align 1-arg (RegTensor) only supports NORM dist, got " << distVal;
            }
            std::string ptrType = DtypeToPtrType(dstDt);
            std::string ubPtr = GetUBufPtr(codegen, op->args_[1], ptrType);
            std::string cast = GetB8Cast(dstDt);
            // B64 types (INT64/UINT64): load as b32 (mirrors AscendC DataCopyImpl
            // b64 RegTraitNumOne: vlds((RegTensor<uint32_t>&)reg, uint32_t* ptr, 0, NORM))
            if (dstDt.GetBit() == 64) {
                std::string b32Ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
                codegen.Emit("vlds((RegTensor<uint32_t>&)" + dst + ", " + b32Ptr + ", 0, NORM);");
            } else {
                codegen.Emit("vlds(" + cast + dst + ", " + ubPtr + ", 0, NORM);");
            }
        }
        return "";
    }
    // 2-arg source form: [dst, src_ptr, offset|mask]; or the 2-dst DINTLV form
    // [dst0, dst1, src_ptr, offset].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3 || op->args_.size() == 4)
        << "vf.load_align requires 2 or 3 args (dst, src_ptr[, offset | mask])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    // The dual-dst form carries [dst0, dst1, src_ptr, (offset)]: the pointer is
    // always at index 2 and the offset is optional, so it cannot share the
    // single-dst offset/pointer index computation below.
    const bool dualDst = ir::As<ir::TupleType>(op->GetType()) != nullptr;
    const size_t srcPtrIdx = (dualDst || op->args_.size() == 4) ? 2 : 1;
    std::string offsetStr = "0";
    if (op->args_.size() == 4) {
        offsetStr = ResolveOffsetArg(codegen, op->args_[3], op->args_[2]);
    } else if (!dualDst) {
        offsetStr = ResolveOffsetArg(codegen, op->args_[2], op->args_[1]);
    }
    DataType dstDt = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") :
                                             GetTypeDtype(op->args_[0]->GetType(), GetExprDtype(op->args_[srcPtrIdx]));
    // Pointer type follows the register's declared dtype. When it differs from
    // the tile's dtype, adjust the offset by the element-width ratio so the byte
    // address stays correct (mirrors b64→b32 stride*2).
    std::string ptrType = DtypeToPtrType(dstDt);
    DataType tileDt = GetExprDtype(op->args_[srcPtrIdx], dstDt);
    // Is the destination a MaskReg? (routes to pld/plds instead of vld/vlds)
    bool dstIsMask = ir::IsMaskReg(op->args_[0]->GetType());
    // Offset adjustment only for RegTensor path (MaskReg always uses uint32_t ptr)
    std::string effectiveOffset = offsetStr;
    if (!dstIsMask && tileDt.GetBit() != dstDt.GetBit()) {
        effectiveOffset = "(" + offsetStr + ") * " + std::to_string(tileDt.GetBit()) + " / " +
                          std::to_string(dstDt.GetBit());
    }
    // Determine mode from kwargs (dist kwarg is legacy alias for mode).
    // MaskReg (plds) and RegTensor (vlds) paths share the same LoadDist enum:
    // LoadDist includes NORM/US/DS/BRC/... and EnumToString yields the bare
    // name (e.g. "DS"), which plds accepts directly and vlds maps to Bxx suffix.
    std::string mode = "NORM";
    if (op->HasKwarg("dist"))
        mode = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/true, "vf.load_align");
    // DINTLV de-interleave: the dst pair is the result target's tuple elements.
    if (mode == "DINTLV_B8" || mode == "DINTLV_B16" || mode == "DINTLV_B32") {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.load_align (de-interleave) does not support data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.load_align (de-interleave) does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.load_align (de-interleave) does not support repeat_stride";
        auto tupleTy = ir::As<ir::TupleType>(op->GetType());
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tupleTy != nullptr && tupleTy->types_.size() == 2)
            << "vf.load_align with DINTLV dist requires the 2-dst assignment form: "
               "dst0, dst1 = vf.load_align(src, offset, dist=DINTLV_B*)";
        DataType dualTileDt = GetExprDtype(op->args_[2], dstDt);
        DataType dstDt0 = GetTypeDtype(tupleTy->types_[0], dualTileDt);
        DataType dstDt1 = GetTypeDtype(tupleTy->types_[1], dualTileDt);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dstDt0.GetBit() == dstDt1.GetBit())
            << "vf.load_align (de-interleave) requires dst0 and dst1 to have the same bit width, got dst0="
            << DTypeStr(dstDt0) << " dst1=" << DTypeStr(dstDt1);
        for (size_t i = 0; i < tupleTy->types_.size(); ++i) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !ir::IsMaskReg(tupleTy->types_[i]))
                << "vf.load_align (de-interleave) requires RegTensor dst, but tuple element " << i << " is a MaskReg";
        }
        std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
        std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
        std::string ptrType0 = DtypeToPtrType(dstDt0);
        std::string ubPtr = GetUBufPtr(codegen, op->args_[2], ptrType0);
        std::string cast0 = GetB8Cast(dstDt0);
        std::string effOffset = offsetStr;
        if (dualTileDt.GetBit() != dstDt0.GetBit() && !offsetStr.empty()) {
            effOffset = "(" + offsetStr + ") * " + std::to_string(dualTileDt.GetBit()) + " / " +
                        std::to_string(dstDt0.GetBit());
        }
        bool dintlvPost = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
        if (dintlvPost) {
            codegen.Emit("vlds(" + cast0 + dst0 + ", " + cast0 + dst1 + ", " + ubPtr + ", " + effOffset + ", " + mode +
                         ", POST_UPDATE);");
        } else {
            codegen.Emit("vlds(" + cast0 + dst0 + ", " + cast0 + dst1 + ", " + ubPtr + ", " + effOffset + ", " + mode +
                         ");");
        }
        return "";
    }
    // AddrReg offset path: MaskReg dst -> pld, RegTensor dst -> vld
    if (op->args_.size() > srcPtrIdx + 1 && ir::IsAddrReg(op->args_[srcPtrIdx + 1]->GetType())) {
        std::string ubPtr = GetUBufPtr(codegen, op->args_[srcPtrIdx], dstIsMask ? "uint32_t" : ptrType);
        if (dstIsMask) {
            codegen.Emit("pld(" + dst + ", " + ubPtr + ", " + offsetStr + ", " + mode + ");");
        } else {
            codegen.Emit("vld(" + dst + ", " + ubPtr + ", " + effectiveOffset + ", " + mode + ");");
        }
        return "";
    }
    // Check for DataBlock load path (vsldb)
    std::string dataCopyMode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        dataCopyMode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    bool postUpdate = false;
    if (op->HasKwarg("post_update")) {
        postUpdate = op->GetKwarg<bool>("post_update");
    }
    // vsldb path (non-contiguous datablock load). Accept both DATA_BLOCK_LOAD
    // (pypto legacy name) and DATA_BLOCK_COPY so
    // code written against either naming does not silently fall back to vlds.
    if (dataCopyMode == "DATA_BLOCK_LOAD" || dataCopyMode == "DATA_BLOCK_COPY") {
        // In DataBlock mode, args[src_ptr_idx+1] is a mask register, not an offset.
        // Verify the user didn't pass an integer/AddrReg offset by mistake.
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY requires 3 args (dst, src_ptr, mask)";
        auto maskVar = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, maskVar != nullptr)
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY requires args[2] to be a "
            << "mask register, but got an offset value (offset is not supported in DataBlock mode)";
        // Verify args[2] is actually a MaskReg, not a Tile or other Var type
        std::string maskName = codegen.GetVarName(maskVar);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsMaskReg(maskVar->GetType()))
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY requires args[2] to be a "
            << "mask register, but got a non-mask variable '" << maskName << "'";
        // DataBlock mode requires RegTensor dst (not MaskReg)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !dstIsMask)
            << "vf.load_align with data_copy_mode=DATA_BLOCK_COPY does not support MaskReg dst";
        // vsldb only supports b8/b16/b32 (not b64)
        std::string maskReg = codegen.GetExprAsCode(op->args_[2]);
        std::string blockStride = "0";
        std::string repeatStride = "0";
        if (op->HasKwarg("block_stride")) {
            blockStride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->HasKwarg("repeat_stride")) {
            repeatStride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (postUpdate) {
            std::string ubPtr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("vsldb(" + dst + ", " + ubPtr + ", (" + blockStride + " << 16u) | (" + repeatStride +
                         " & 0xFFFFU), " + maskReg + ", POST_UPDATE);");
        } else {
            std::string ubPtr = GetUBufPtr(codegen, op->args_[1], ptrType);
            codegen.Emit("vsldb(" + dst + ", " + ubPtr + ", (" + blockStride + " << 16u), " + maskReg + ");");
        }
        return "";
    }
    // Route by dst kind: MaskReg → plds, RegTensor → vlds
    if (dstIsMask) {
        // plds path: dst is MaskReg, pointer is always uint32_t*
        // pld/plds: only supports NORM/US/DS dist
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, mode == "NORM" || mode == "US" || mode == "DS")
            << "vf.load_align (MaskReg) only supports NORM/US/DS dist, got " << mode;
        std::string pldsPtr;
        if (postUpdate) {
            pldsPtr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            codegen.Emit("plds(" + dst + ", " + pldsPtr + ", " + offsetStr + ", " + mode + ", POST_UPDATE);");
        } else {
            pldsPtr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("plds(" + dst + ", " + pldsPtr + ", " + offsetStr + ", " + mode + ");");
        }
        return "";
    }
    // vlds path: dst is RegTensor
    // Get UB pointer (post_update uses reference-based ptr)
    std::string ubPtr;
    if (postUpdate) {
        ubPtr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
    } else {
        ubPtr = GetUBufPtr(codegen, op->args_[1], ptrType);
    }
    // Determine vlds mode string
    std::string vldsMode;
    if (mode == "NORM") {
        vldsMode = "NORM";
    } else if (mode == "BRC") {
        if (IsB8Type(dstDt))
            vldsMode = "BRC_B8";
        else if (IsB16Type(dstDt))
            vldsMode = "BRC_B16";
        else
            vldsMode = "BRC_B32";
    } else if (mode == "US") {
        if (IsB8Type(dstDt))
            vldsMode = "US_B8";
        else
            vldsMode = "US_B16";
    } else if (mode == "DS") {
        if (IsB8Type(dstDt))
            vldsMode = "DS_B8";
        else
            vldsMode = "DS_B16";
    } else if (mode == "UNPK") {
        if (IsB8Type(dstDt))
            vldsMode = "UNPK_B8";
        else if (IsB16Type(dstDt))
            vldsMode = "UNPK_B16";
        else
            vldsMode = "UNPK_B32";
    } else if (mode == "UNPK4") {
        vldsMode = "UNPK4_B8";
    } else if (mode == "BLK") {
        vldsMode = "BLK";
    } else if (mode == "E2B") {
        if (IsB16Type(dstDt))
            vldsMode = "E2B_B16";
        else
            vldsMode = "E2B_B32";
    } else {
        // Fallback: pass through directly (e.g. BRC_B32, DS_B16, E2B_B32, etc.)
        vldsMode = mode;
    }
    std::string cast = GetB8Cast(dstDt);
    // B64 types (INT64/UINT64): load as b32 (mirrors AscendC DataCopyImpl b64
    // path, which passes the resolved dist through unchanged)
    if (dstDt.GetBit() == 64) {
        std::string b32Ptr;
        if (postUpdate) {
            b32Ptr = codegen.GetOrCreateVFTilePtr(op->args_[1], /*is_post_update=*/true);
            std::string postOffset = "(" + effectiveOffset + ") * 2";
            codegen.Emit("vlds((RegTensor<uint32_t>&)" + dst + ", (__ubuf__ uint32_t*&)" + b32Ptr + ", " + postOffset +
                         ", " + vldsMode + ", POST_UPDATE);");
        } else {
            b32Ptr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("vlds((RegTensor<uint32_t>&)" + dst + ", " + b32Ptr + ", " + effectiveOffset + ", " +
                         vldsMode + ");");
        }
    } else if (postUpdate) {
        std::string postOffset = effectiveOffset;
        codegen.Emit("vlds(" + cast + dst + ", " + ubPtr + ", " + postOffset + ", " + vldsMode + ", POST_UPDATE);");
    } else {
        codegen.Emit("vlds(" + cast + dst + ", " + ubPtr + ", " + effectiveOffset + ", " + vldsMode + ");");
    }
    return "";
}

// ============================================================================
// StoreAlign — vsts
// ============================================================================

static std::string EmitVFStoreAlign(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // args: [dst_ptr, src_reg, mask, (optional) block_stride, (optional) repeat_stride]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2)
        << "vf.store_align requires at least 2 args (dst_ptr, src_reg)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    // vsts supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(srcDt) || srcDt.GetBit() == 16 || srcDt.GetBit() == 32 || srcDt.GetBit() == 64)
        << "vf.store_align only supports b8/b16/b32/b64 types, got " << DTypeStr(srcDt);
    std::string srcReg = codegen.GetExprAsCode(op->args_[1]);
    std::string cast = GetB8Cast(srcDt);
    // MaskReg src path: when args[1] is a MaskReg, dispatch to psts/pst (mask store)
    bool srcIsMask = false;
    if (auto srcV = ir::As<ir::Var>(op->args_[1])) {
        srcIsMask = ir::IsMaskReg(srcV->GetType());
    }
    // vsts stores raw register bits; GetUBufPtr already casts the dst pointer
    // to the src register type (equivalent to AscendC's (RegTensor<T>&) cast),
    // so no runtime dtype compatibility check is needed here.
    if (srcIsMask) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.store_align (MaskReg src) does not support data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.store_align (MaskReg src) does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.store_align (MaskReg src) does not support repeat_stride";
        std::string dist = "NORM";
        if (op->HasKwarg("dist")) {
            dist = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/false, "vf.store_align");
            // pst/psts: only supports NORM and PACK dist
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, dist == "NORM" || dist == "PACK" || dist == "PK")
                << "vf.store_align (MaskReg) only supports NORM/PACK dist, got " << dist;
            // psts/pst use PK (not PACK) for packed mode
            if (dist == "PACK")
                dist = "PK";
        }
        // AddrReg offset path: 3rd arg is AddrReg -> pst(mask, ptr, areg, dist)
        if (op->args_.size() >= 3) {
            std::string thirdArg = codegen.GetExprAsCode(op->args_[2]);
            if (ir::IsAddrReg(op->args_[2]->GetType())) {
                std::string ubPtr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
                codegen.Emit("pst(" + srcReg + ", " + ubPtr + ", " + thirdArg + ", " + dist + ");");
                return "";
            }
            // Post-update path: int offset with post_update kwarg
            bool postUpdate = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
            if (postUpdate) {
                std::string ptrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
                codegen.Emit("psts(" + srcReg + ", " + ptrVar + ", " + thirdArg + ", " + dist + ", POST_UPDATE);");
                return "";
            }
        }
        // Default: psts with offset=0
        std::string ubPtr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
        codegen.Emit("psts(" + srcReg + ", " + ubPtr + ", 0, " + dist + ");");
        return "";
    }
    // Resolve the dist kwarg for the single-source paths (vst/vsts). The MaskReg
    // path above handles its own NORM/PACK dist. Coarse names (NORM,
    // FIRST_ELEMENT, PACK, PACK4, INTLV) auto-expand to the element-width
    // variant based on the src dtype; width-qualified names map to the CCE
    // DistVST constants (mirrors AscendC Reg::StoreDist / GetStoreDist).
    std::string dist = "";
    if (op->HasKwarg("dist")) {
        dist = VFCheckedDistName(op->GetKwarg<int>("dist"), /*is_load=*/false, "vf.store_align");
    }
    if (dist.empty()) {
        // Auto-select default dist based on src dtype (uses NORM_B8/B16/B32 by element width)
        DataType srcDtype = DataType::FP32;
        if (auto scalarType = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
            srcDtype = scalarType->dtype_;
        } else if (auto tileTypeTmp = ir::As<ir::TileType>(op->args_[0]->GetType())) {
            srcDtype = tileTypeTmp->dtype_;
        }
        if (IsB8Type(srcDtype))
            dist = "NORM_B8";
        else if (IsB16Type(srcDtype))
            dist = "NORM_B16";
        else
            dist = "NORM_B32";
    }
    // Auto-expand shorthand dist names to element-width-qualified CCE constants
    if (dist == "NORM") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "NORM_B8";
        else if (IsB16Type(sd))
            dist = "NORM_B16";
        else
            dist = "NORM_B32";
    } else if (dist == "FIRST_ELEMENT" || dist == "FIRST_ELE") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "ONEPT_B8";
        else if (IsB16Type(sd))
            dist = "ONEPT_B16";
        else
            dist = "ONEPT_B32";
    } else if (dist == "PACK") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd) || IsB16Type(sd))
            dist = "PK_B16";
        else if (sd.GetBit() == 32)
            dist = "PK_B32";
        else
            dist = "PK_B64";
    } else if (dist == "PACK4") {
        dist = "PK4_B32";
    } else if (dist == "INTLV") {
        DataType sd = GetExprDtype(op->args_[1]);
        if (IsB8Type(sd))
            dist = "INTLV_B8";
        else if (IsB16Type(sd))
            dist = "INTLV_B16";
        else
            dist = "INTLV_B32";
    } else if (dist == "FIRST_ELEMENT_B8" || dist == "FIRST_ELEMENT_B16" || dist == "FIRST_ELEMENT_B32") {
        // Explicit-width variants (mirrors AscendC Reg::StoreDist): map to the CCE
        // DistVST constants; NORM_Bx/INTLV_Bx already match and pass through as-is.
        dist = "ONEPT" + dist.substr(std::strlen("FIRST_ELEMENT"));
    } else if (dist == "PACK_B16" || dist == "PACK_B32" || dist == "PACK_B64") {
        dist = "PK" + dist.substr(std::strlen("PACK"));
    } else if (dist == "PACK4_B32") {
        dist = "PK4_B32";
    }
    // AddrReg offset path: when 4th arg is an AddrReg variable,
    // emit vst(src, ptr, areg, dist, mask) — 5 args (note: vst, not vsts).
    // The resolved dist is honored here as well (previously the dist kwarg was
    // silently ignored in this path and NORM_Bx was always used).
    if (op->args_.size() >= 4) {
        std::string addrReg = codegen.GetExprAsCode(op->args_[3]);
        if (ir::IsAddrReg(op->args_[3]->GetType())) {
            // Verify args[2] is a MaskReg
            auto maskVar4 = ir::As<ir::Var>(op->args_[2]);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, maskVar4 != nullptr)
                << "vf.store_align (AddrReg) requires args[2] to be a mask register, but got non-Var type";
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsMaskReg(maskVar4->GetType()))
                << "vf.store_align (AddrReg) requires args[2] to be a mask register";
            // vst has no post mode — the AddrReg offset must be recreated each
            // iteration via vf.create_addr_reg
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                              !(op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update")))
                << "vf.store_align (AddrReg) does not support post_update; recreate the AddrReg offset each "
                   "iteration via vf.create_addr_reg";
            // vsstb (DataBlock copy) has no AddrReg form
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
                << "vf.store_align (AddrReg) does not support data_copy_mode";
            // vst (AddrReg) is a single-source intrinsic: INTLV dists need two
            // source registers (vsts dual form)
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                              dist != "INTLV_B8" && dist != "INTLV_B16" && dist != "INTLV_B32")
                << "vf.store_align (AddrReg) does not support INTLV dist (requires two source registers), got " << dist;
            std::string maskReg = codegen.GetExprAsCode(op->args_[2]);
            std::string ptrType = "float";
            if (auto scalarType = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
                ptrType = DtypeToPtrType(scalarType->dtype_);
            }
            std::string dstPtr = GetUBufPtr(codegen, op->args_[0], ptrType);
            std::string vstSrc = cast + srcReg;
            std::string vstMask = maskReg;
            if (srcDt.GetBit() == 64) {
                // b64: simulate with b32 vst — reinterpret the register/pointer as
                // int32 and expand the b64-element mask to per-b32-lane bits
                // (ppack + pintlv_b32, mirrors the AscendC DataCopyImpl vst-areg
                // b64 path); dist passes through unchanged
                std::string packedMask = maskReg + "_b32lo_";
                std::string dumpMask = maskReg + "_b32hi_";
                codegen.Emit("MaskReg " + packedMask + ";");
                codegen.Emit("MaskReg " + dumpMask + ";");
                codegen.Emit("ppack(" + packedMask + ", " + maskReg + ", LOWER);");
                codegen.Emit("pintlv_b32(" + packedMask + ", " + dumpMask + ", " + packedMask + ", " + packedMask +
                             ");");
                vstSrc = "(RegTensor<int32_t>&)" + srcReg;
                vstMask = packedMask;
                dstPtr = GetUBufPtr(codegen, op->args_[0], "int32_t");
            }
            codegen.Emit("vst(" + vstSrc + ", " + dstPtr + ", " + addrReg + ", " + dist + ", " + vstMask + ");");
            return "";
        }
    }
    bool postUpdate = false;
    if (op->HasKwarg("post_update")) {
        postUpdate = op->GetKwarg<bool>("post_update");
    }
    std::string dataCopyMode = "NORM";
    if (op->HasKwarg("data_copy_mode")) {
        dataCopyMode = VFEnumValueName(
            ir::EnumToString(static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"))));
    }
    // Pointer type follows the register's declared dtype. When it differs from
    // the tile's dtype, the offset/stride must be adjusted by the element-width
    // ratio so the byte address stays correct (mirrors b64→b32
    // stride*2). No register cast needed — the register is already src_dt.
    std::string ptrType = DtypeToPtrType(srcDt);
    DataType tileDt = DataType::FP32;
    auto tileType = ir::As<ir::TileType>(op->args_[0]->GetType());
    if (tileType) {
        tileDt = tileType->dtype_;
    } else if (auto scalarType = ir::As<ir::ScalarType>(op->args_[1]->GetType())) {
        tileDt = scalarType->dtype_;
    } else {
        tileDt = srcDt;
    }
    std::string widthOp;
    if (tileDt.GetBit() != srcDt.GetBit()) {
        widthOp = " * " + std::to_string(tileDt.GetBit()) + " / " + std::to_string(srcDt.GetBit());
    }
    // INTLV modes need two src registers: args = [dst_ptr, src_reg, src1, mask]
    bool isIntlv = (dist == "INTLV_B8" || dist == "INTLV_B16" || dist == "INTLV_B32");
    // INTLV requires 4 args; 4-arg with args[2] as register (not mask) requires INTLV
    // Skip this check for post_update path (4 args = dst, src, mask, stride is valid)
    if (isIntlv) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
            << "vf.store_align INTLV requires 4 args (dst_ptr, src_reg, src1, mask)";
    } else if (op->args_.size() == 4 && !srcIsMask && !postUpdate) {
        auto thirdArgVar = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          thirdArgVar == nullptr || ir::IsMaskReg(thirdArgVar->GetType()))
            << "vf.store_align with 4 args where args[2] is a register (not mask) requires INTLV dist, " << "got "
            << dist;
    }
    if (isIntlv) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
            << "vf.store_align INTLV requires 4 args (dst_ptr, src_reg, src1, mask)";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.store_align INTLV mode is incompatible with data_copy_mode";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("post_update"))
            << "vf.store_align INTLV mode does not support post_update";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
            << "vf.store_align INTLV mode does not support block_stride";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("repeat_stride"))
            << "vf.store_align INTLV mode does not support repeat_stride";
        // INTLV mode: args[2] is src1 (second source register), args[3] is mask
        auto intlvMaskVar = ir::As<ir::Var>(op->args_[3]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, intlvMaskVar != nullptr)
            << "vf.store_align INTLV requires args[3] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsMaskReg(intlvMaskVar->GetType()))
            << "vf.store_align INTLV requires args[3] to be a mask register";
        std::string src1 = codegen.GetExprAsCode(op->args_[2]);
        std::string maskReg = codegen.GetExprAsCode(op->args_[3]);
        // vsts 2-source overload (__VF_VSTSX2) does not exist for FP32;
        // cast to UINT32 (same 32-bit width) to use the UINT32 overload.
        std::string intlvPtrType = ptrType;
        if (srcDt == DataType::FP32) {
            srcReg = "(RegTensor<uint32_t> &)" + srcReg;
            src1 = "(RegTensor<uint32_t> &)" + src1;
            intlvPtrType = "uint32_t";
        }
        std::string dstPtr = GetUBufPtr(codegen, op->args_[0], intlvPtrType);
        codegen.Emit("vsts(" + cast + srcReg + ", " + cast + src1 + ", " + dstPtr + ", 0, " + dist + ", " + maskReg +
                     ");");
    } else if (dataCopyMode == "DATA_BLOCK_LOAD" || dataCopyMode == "DATA_BLOCK_COPY") {
        // Accept both DATA_BLOCK_LOAD (pypto legacy name) and DATA_BLOCK_COPY
        //, consistent with load_align behavior.
        // vsstb only supports b8/b16/b32 (not b64)
        DataType dcSrcDt = GetExprDtype(op->args_[1]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          dcSrcDt.GetBit() == 8 || dcSrcDt.GetBit() == 16 || dcSrcDt.GetBit() == 32)
            << "vf.store_align (DATA_BLOCK_COPY) only supports b8/b16/b32, got " << DTypeStr(dcSrcDt);
        // In DataBlock mode, args[2] is a mask register
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
            << "vf.store_align (DATA_BLOCK_COPY) requires at least 3 args (dst_ptr, src_reg, mask)";
        auto dbMaskVar = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dbMaskVar != nullptr)
            << "vf.store_align (DATA_BLOCK_COPY) requires args[2] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsMaskReg(dbMaskVar->GetType()))
            << "vf.store_align (DATA_BLOCK_COPY) requires args[2] to be a mask register";
        std::string maskReg = codegen.GetExprAsCode(op->args_[2]);
        std::string blockStride = "0";
        std::string repeatStride = "0";
        if (op->args_.size() >= 4) {
            blockStride = codegen.GetExprAsCode(op->args_[3]);
        } else if (op->HasKwarg("block_stride")) {
            blockStride = std::to_string(op->GetKwarg<int>("block_stride"));
        }
        if (op->args_.size() >= 5) {
            repeatStride = codegen.GetExprAsCode(op->args_[4]);
        } else if (op->HasKwarg("repeat_stride")) {
            repeatStride = std::to_string(op->GetKwarg<int>("repeat_stride"));
        }
        if (postUpdate) {
            std::string ptrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
            codegen.Emit("vsstb(" + cast + srcReg + ", " + ptrVar + ", " + "(" + blockStride + " << 16u) | (" +
                         repeatStride + " & 0xFFFFU), " + maskReg + ", POST_UPDATE);");
        } else {
            std::string dstPtr = GetUBufPtr(codegen, op->args_[0], ptrType);
            codegen.Emit("vsstb(" + cast + srcReg + ", " + dstPtr + ", " + "(" + blockStride + " << 16u) | (" +
                         repeatStride + " & 0xFFFFU), " + maskReg + ");");
        }
    } else if (postUpdate) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
            << "vf.store_align (post_update) requires at least 3 args (dst_ptr, src_reg, mask)";
        auto puMaskVar = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, puMaskVar != nullptr)
            << "vf.store_align requires args[2] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsMaskReg(puMaskVar->GetType()))
            << "vf.store_align requires args[2] to be a mask register";
        std::string maskReg = codegen.GetExprAsCode(op->args_[2]);
        std::string stride = (op->args_.size() >= 4) ? codegen.GetExprAsCode(op->args_[3]) : "0";
        // B64 register types need stride doubled (postUpdateStride * 2)
        std::string effectiveStride = stride;
        if (srcDt.GetBit() == 64) {
            effectiveStride = "(" + stride + ") * 2";
        } else if (!widthOp.empty()) {
            effectiveStride = "(" + stride + ")" + widthOp;
        }
        std::string ptrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        // B64 types: use b32 vsts with ppack+pintlv_b32 mask splitting (mirrors AscendC
        // DataCopyImpl post_update b64 path). The b64 register is reinterpreted as
        // RegTensor<uint32_t>& and stored via b32 vsts with the split mask.
        if (srcDt.GetBit() == 64) {
            std::string p = srcReg + "_b64pu_";
            std::string tm = p + "tm_";
            std::string m0 = p + "m0_";
            std::string m1 = p + "m1_";
            codegen.Emit("MaskReg " + tm + ";");
            codegen.Emit("ppack(" + tm + ", " + maskReg + ", LOWER);");
            codegen.Emit("MaskReg " + m0 + ", " + m1 + ";");
            codegen.Emit("pintlv_b32(" + m0 + ", " + m1 + ", " + tm + ", " + tm + ");");
            codegen.Emit("vsts((RegTensor<uint32_t>&)" + srcReg + ", (__ubuf__ uint32_t*&)" + ptrVar + ", " +
                         effectiveStride + ", " + dist + ", " + m0 + ", POST_UPDATE);");
        } else {
            codegen.Emit("vsts(" + cast + srcReg + ", " + ptrVar + ", " + effectiveStride + ", " + dist + ", " +
                         maskReg + ", POST_UPDATE);");
        }
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
            << "vf.store_align requires at least 3 args (dst_ptr, src_reg, mask)";
        auto defMaskVar = ir::As<ir::Var>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, defMaskVar != nullptr)
            << "vf.store_align requires args[2] to be a mask register, but got non-Var type";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsMaskReg(defMaskVar->GetType()))
            << "vf.store_align requires args[2] to be a mask register";
        std::string maskReg = codegen.GetExprAsCode(op->args_[2]);
        std::string dstPtr = GetUBufPtr(codegen, op->args_[0], ptrType);
        std::string offsetStr = "0";
        if (op->args_.size() >= 4) {
            offsetStr = ResolveOffsetArg(codegen, op->args_[3], op->args_[0]);
        }
        std::string effectiveOffset = offsetStr;
        if (!widthOp.empty()) {
            effectiveOffset = "(" + offsetStr + ")" + widthOp;
        }
        // B64 types (INT64/UINT64): use b32 vsts with ppack+pintlv_b32 mask splitting
        // (mirrors AscendC DataCopyImpl b64 path). The b64 register is reinterpreted
        // as RegTensor<uint32_t>& and stored via b32 vsts with the split mask.
        if (srcDt.GetBit() == 64) {
            std::string p = srcReg + "_b64st_";
            std::string tm = p + "tm_";
            std::string m0 = p + "m0_";
            std::string m1 = p + "m1_";
            codegen.Emit("MaskReg " + tm + ";");
            codegen.Emit("ppack(" + tm + ", " + maskReg + ", LOWER);");
            codegen.Emit("MaskReg " + m0 + ", " + m1 + ";");
            codegen.Emit("pintlv_b32(" + m0 + ", " + m1 + ", " + tm + ", " + tm + ");");
            std::string b32Ptr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
            codegen.Emit("vsts((RegTensor<uint32_t>&)" + srcReg + ", " + b32Ptr + ", " + effectiveOffset + ", " + dist +
                         ", " + m0 + ");");
        } else {
            codegen.Emit("vsts(" + cast + srcReg + ", " + dstPtr + ", " + effectiveOffset + ", " + dist + ", " +
                         maskReg + ");");
        }
    }
    return "";
}

// ============================================================================
// MemBar — mem_bar
// ============================================================================

static std::string EmitVFMemBar(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    std::string mode = "VST_VLD";
    if (op->HasKwarg("mode")) {
        mode = VFEnumValueName(ir::EnumToString(static_cast<ir::MemBarMode>(op->GetKwarg<int>("mode"))));
    }
    codegen.Emit("mem_bar(" + mode + ");");
    return "";
}

// ============================================================================
// Max — vmax
// ============================================================================

static std::string EmitVFMax(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.max requires 4 args (dst, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0Dt) || s0Dt == DataType::FP16 ||
                                                    s0Dt == DataType::FP32 || s0Dt == DataType::BF16))
        << "vf.max src only supports supported types, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s1Dt) || s1Dt == DataType::FP16 ||
                                                    s1Dt == DataType::FP32 || s1Dt == DataType::BF16))
        << "vf.max src only supports supported types, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.max requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // MERGING is not supported by the underlying vmax instruction on current device.
    std::string mode = VFZeroingOnly(op, "vf.max");
    if (s0Dt.GetBit() == 64) {
        // B64 max via deinterleave + b32 compare + vsel + interleave.
        // For INT64: compare high halves (signed), then low halves (unsigned) on tie.
        // For UINT64: compare high halves (unsigned), then low halves (unsigned) on tie.
        std::string p = dst + "_max_";
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        // Compare high halves
        std::string hiGt = p + "_higt_";
        std::string hiEq = p + "_hieq_";
        std::string loGt = p + "_logt_";
        codegen.Emit("MaskReg " + hiGt + ";");
        codegen.Emit("MaskReg " + hiEq + ";");
        codegen.Emit("MaskReg " + loGt + ";");
        if (s0Dt == DataType::INT64) {
            codegen.Emit("vcmp_gt(" + hiGt + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         allM + ");");
        } else {
            codegen.Emit("vcmp_gt(" + hiGt + ", " + hi0 + ", " + hi1 + ", " + allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", " + hi0 + ", " + hi1 + ", " + allM + ");");
        }
        codegen.Emit("vcmp_gt(" + loGt + ", " + lo0 + ", " + lo1 + ", " + allM + ");");
        // Combine: src0 > src1 iff hi_gt OR (hi_eq AND lo_gt)
        std::string eqLo = p + "_eqlo_";
        std::string s0Gt = p + "_s0gt_";
        codegen.Emit("MaskReg " + eqLo + ";");
        codegen.Emit("MaskReg " + s0Gt + ";");
        codegen.Emit("pand(" + eqLo + ", " + hiEq + ", " + loGt + ", " + allM + ");");
        codegen.Emit("por(" + s0Gt + ", " + hiGt + ", " + eqLo + ", " + allM + ");");
        // Select: dst = src0 > src1 ? src0 : src1 (per b32 half)
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiDst);
        codegen.Emit("vsel(" + loDst + ", " + lo0 + ", " + lo1 + ", " + s0Gt + ");");
        codegen.Emit("vsel(" + hiDst + ", " + hi0 + ", " + hi1 + ", " + s0Gt + ");");
        // Interleave back to b64
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        // Apply ZEROING
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmax(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Add — vadd
// ============================================================================

static std::string EmitVFAdd(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.add requires 4 args (dst, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0Dt) || s0Dt == DataType::FP16 ||
                                                    s0Dt == DataType::FP32 || s0Dt == DataType::BF16))
        << "vf.add src0 only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.add requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // MERGING is not supported by the underlying vadd instruction on current device.
    std::string mode = VFZeroingOnly(op, "vf.add");
    if (s0Dt.GetBit() == 64) {
        // B64 add: Mirrors AscendC AddB64Impl (vaddc + vaddcs carry-chain).
        // Uses CalTraitOneByTransToTraitTwo pattern: ppack mask, deinterleave both sources,
        // AddB64 (vaddc + vaddcs), interleave back.
        std::string p = dst + "_add_";
        std::string b32Cast = (s0Dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string carry = p + "_carry_";
        std::string loD = p + "_lod_";
        std::string hiD = p + "_hid_";
        codegen.Emit("MaskReg " + carry + ";");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
        codegen.Emit("vaddc(" + carry + ", " + b32Cast + loD + ", " + b32Cast + lo0 + ", " + b32Cast + lo1 + ", " +
                     packedM + ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32Cast + hiD + ", " + b32Cast + hi0 + ", " + b32Cast + hi1 + ", " +
                     carry + ", " + packedM + ");");
        EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
    } else {
        codegen.Emit("vadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Sub — vsub
// ============================================================================

static std::string EmitVFSub(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.sub requires 4 args (dst, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0Dt) || s0Dt == DataType::FP16 ||
                                                    s0Dt == DataType::FP32 || s0Dt == DataType::BF16))
        << "vf.sub src0 only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.sub requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.sub");
    if (s0Dt.GetBit() == 64) {
        // B64 sub: Mirrors AscendC SubB64Impl (vsubc + vsubcs borrow-chain).
        // Uses CalTraitOneByTransToTraitTwo pattern: ppack mask, deinterleave both sources,
        // SubB64 (vsubc + vsubcs), interleave back.
        std::string p = dst + "_sub_";
        std::string b32Cast = (s0Dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string borrow = p + "_borrow_";
        std::string loD = p + "_lod_";
        std::string hiD = p + "_hid_";
        codegen.Emit("MaskReg " + borrow + ";");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
        codegen.Emit("vsubc(" + borrow + ", " + b32Cast + loD + ", " + b32Cast + lo0 + ", " + b32Cast + lo1 + ", " +
                     packedM + ");");
        codegen.Emit("vsubcs(" + borrow + ", " + b32Cast + hiD + ", " + b32Cast + hi0 + ", " + b32Cast + hi1 + ", " +
                     borrow + ", " + packedM + ");");
        EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
    } else {
        codegen.Emit("vsub(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// And — vand
// ============================================================================

// ============================================================================
// And — vand, Xor — vxor, Or — vor
// Bitwise operations: type-agnostic, only requires same bit width across
// dst/src0/src1.  Any b8/b16/b32/b64 type (including FP16/BF16/FP32) is valid
// because vand/vxor/vor operate on raw bits.
// ============================================================================

static std::string EmitVFAnd(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.and_ requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // Mask-form dispatch follows the first source: the dst kind mirrors src0
    // (the registry deduce guarantees it), so args_[1] carries the kind.
    if (IsSrcMaskReg(op, 1)) {
        codegen.Emit("pand(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s0Dt == DataType::INT8 || s0Dt == DataType::UINT8 || s0Dt == DataType::BOOL ||
                          s0Dt == DataType::FP8E4M3FN || s0Dt == DataType::FP8E5M2 || s0Dt == DataType::FP8E8M0 ||
                          s0Dt == DataType::HF8 || s0Dt == DataType::INT16 || s0Dt == DataType::UINT16 ||
                          s0Dt == DataType::FP16 || s0Dt == DataType::BF16 || s0Dt == DataType::INT32 ||
                          s0Dt == DataType::UINT32 || s0Dt == DataType::FP32 || s0Dt == DataType::INT64 ||
                          s0Dt == DataType::UINT64)
        << "vf.and_ src0 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s1Dt == DataType::INT8 || s1Dt == DataType::UINT8 || s1Dt == DataType::BOOL ||
                          s1Dt == DataType::FP8E4M3FN || s1Dt == DataType::FP8E5M2 || s1Dt == DataType::FP8E8M0 ||
                          s1Dt == DataType::HF8 || s1Dt == DataType::INT16 || s1Dt == DataType::UINT16 ||
                          s1Dt == DataType::FP16 || s1Dt == DataType::BF16 || s1Dt == DataType::INT32 ||
                          s1Dt == DataType::UINT32 || s1Dt == DataType::FP32 || s1Dt == DataType::INT64 ||
                          s1Dt == DataType::UINT64)
        << "vf.and_ src1 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.and_ requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string castPrefix = "(RegTensor<" + s0Dt.ToCTypeString() + "> &)";
    std::string s0Expr = (s0Dt == s1Dt) ? src0 : (castPrefix + src0);
    std::string s1Expr = (s1Dt == s0Dt) ? src1 : (castPrefix + src1);
    std::string mode = VFZeroingOnly(op, "vf.and_");
    if (s0Dt.GetBit() == 64) {
        // B64 bitwise via b32 vand (mirrors AscendC AndImpl b64 path, see EmitB64Bitwise).
        EmitB64Bitwise(codegen, "vand", {s0Expr, s1Expr}, dst, mask, mode, dst + "_and_");
    } else {
        codegen.Emit("vand(" + dst + ", " + s0Expr + ", " + s1Expr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Xor — vxor, Or — vor
// Bitwise operations: type-agnostic, only requires same bit width across
// dst/src0/src1.  Any b8/b16/b32/b64 type (including FP16/BF16/FP32) is valid
// because vxor/vor operate on raw bits.
// ============================================================================

static std::string EmitVFBinaryBitwise(const ir::CallPtr& op, codegen::CodegenBase& codegenBase,
                                       const std::string& opName, const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << opName << " requires 4 args (dst, src0, src1, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // Mask-form dispatch follows the first source (dst kind mirrors src0).
    if (IsSrcMaskReg(op, 1)) {
        std::string pInstr = "p" + instruction.substr(1);
        codegen.Emit(pInstr + "(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ");");
        return "";
    }
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s0Dt == DataType::INT8 || s0Dt == DataType::UINT8 || s0Dt == DataType::BOOL ||
                          s0Dt == DataType::FP8E4M3FN || s0Dt == DataType::FP8E5M2 || s0Dt == DataType::FP8E8M0 ||
                          s0Dt == DataType::HF8 || s0Dt == DataType::INT16 || s0Dt == DataType::UINT16 ||
                          s0Dt == DataType::FP16 || s0Dt == DataType::BF16 || s0Dt == DataType::INT32 ||
                          s0Dt == DataType::UINT32 || s0Dt == DataType::FP32 || s0Dt == DataType::INT64 ||
                          s0Dt == DataType::UINT64)
        << opName
        << " src0 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      s1Dt == DataType::INT8 || s1Dt == DataType::UINT8 || s1Dt == DataType::BOOL ||
                          s1Dt == DataType::FP8E4M3FN || s1Dt == DataType::FP8E5M2 || s1Dt == DataType::FP8E8M0 ||
                          s1Dt == DataType::HF8 || s1Dt == DataType::INT16 || s1Dt == DataType::UINT16 ||
                          s1Dt == DataType::FP16 || s1Dt == DataType::BF16 || s1Dt == DataType::INT32 ||
                          s1Dt == DataType::UINT32 || s1Dt == DataType::FP32 || s1Dt == DataType::INT64 ||
                          s1Dt == DataType::UINT64)
        << opName
        << " src1 only supports "
           "INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/FP8E4M3FN/FP8E5M2/FP8E8M0/HF8/INT64/UINT64, got "
        << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << opName << " requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string mode = VFZeroingOnly(op, opName);
    if (s0Dt.GetBit() == 64) {
        // B64 bitwise via b32 instruction (mirrors AscendC XorImpl/OrImpl b64 path, see EmitB64Bitwise).
        std::string p = dst + "_" + instruction.substr(1) + "_";
        EmitB64Bitwise(codegen, instruction, {src0, src1}, dst, mask, mode, p);
    } else {
        codegen.Emit(instruction + "(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

static std::string EmitVFXor(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return EmitVFBinaryBitwise(op, codegenBase, "vf.xor", "vxor");
}

// ============================================================================
// Or — vor
// ============================================================================

static std::string EmitVFOr(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return EmitVFBinaryBitwise(op, codegenBase, "vf.or_", "vor");
}

// ============================================================================
// Reduce — vcadd/vcmax/vcmin + vcgadd/vcgmax/vcgmin (unified)
// Supports both new-style (mode=SUM/MAX/MIN, datablock) and legacy (reduce_type=ADD/MAX, merge_mode)
// ============================================================================

// Shared reduction emitter. `reduce_mode` is one of "SUM"/"MAX"/"MIN".
static std::string EmitVFReduceImpl(const ir::CallPtr& op, codegen::CodegenBase& codegenBase,
                                    const std::string& reduceMode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << op->name_ << " requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    bool datablock = false;
    if (op->HasKwarg("datablock")) {
        datablock = op->GetKwarg<bool>("datablock");
    }
    if (datablock) {
        // Datablock reduce only supports b16/b32 (no b64, no BF16)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          (srcDt.GetBit() == 16 || srcDt.GetBit() == 32) &&
                              (IsArithIntType(srcDt) || srcDt == DataType::FP16 || srcDt == DataType::FP32))
            << op->name_ << " (datablock) src only supports b16/b32 INT/UINT/FP16/FP32, got " << DTypeStr(srcDt);
    } else {
        // Non-datablock reduce supports b16/b32/b64 (no BF16)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          (srcDt.GetBit() == 16 || srcDt.GetBit() == 32 || srcDt.GetBit() == 64) &&
                              (IsArithIntType(srcDt) || srcDt == DataType::FP16 || srcDt == DataType::FP32))
            << op->name_ << " src only supports b16/b32/b64 INT/UINT/FP16/FP32, got " << DTypeStr(srcDt);
    }
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // vcadd for 16-bit int requires 32-bit accumulator: s16→s32, u16→u32
    if (!datablock && reduceMode == "SUM") {
        if (srcDt == DataType::INT16) {
            dst = "(RegTensor<int32_t>&)" + dst;
        } else if (srcDt == DataType::UINT16) {
            dst = "(RegTensor<uint32_t>&)" + dst;
        }
    }
    std::string intrinsic;
    if (reduceMode == "SUM")
        intrinsic = datablock ? "vcgadd" : "vcadd";
    else if (reduceMode == "MAX")
        intrinsic = datablock ? "vcgmax" : "vcmax";
    else
        intrinsic = datablock ? "vcgmin" : "vcmin";
    // reduce ops use "merge_mode" kwarg (not "mode")
    std::string mode = "MODE_ZEROING";
    if (op->HasKwarg("merge_mode")) {
        auto mergeMode = static_cast<ir::MergeMode>(op->GetKwarg<int>("merge_mode"));
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, mergeMode == ir::MergeMode::ZEROING)
            << op->name_ << " only supports ZEROING mode on current device, but got MERGING";
        mode = mergeMode == ir::MergeMode::MERGING ? "MODE_MERGING" : "MODE_ZEROING";
    }
    if (!datablock && srcDt.GetBit() == 64) {
        // B64 reduce (INT64/UINT64): no native b64 vcadd/vcmax/vcmin. Defer to
        // the b64 emulation helpers (mirrors AscendC ReduceSumB64Impl /
        // ReduceMaxB64Impl / ReduceMinB64Impl). The result lands in element 0
        // of dst; the remaining dst lanes are don't-care.
        if (reduceMode == "SUM") {
            EmitB64ReduceSum(codegen, dst, src, mask);
        } else {
            EmitB64ReduceMaxMin(codegen, dst, src, mask, reduceMode == "MAX", srcDt == DataType::INT64);
        }
        return "";
    }
    codegen.Emit(intrinsic + "(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return EmitVFReduceImpl(op, codegenBase, "SUM");
}

static std::string EmitVFReduceMax(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return EmitVFReduceImpl(op, codegenBase, "MAX");
}

static std::string EmitVFReduceMin(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return EmitVFReduceImpl(op, codegenBase, "MIN");
}

// ============================================================================
// Mul — vmul
// ============================================================================

static std::string EmitVFMul(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.mul requires 4 args (dst, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s0Dt.GetBit() == 16 || s0Dt.GetBit() == 32 || s0Dt.GetBit() == 64))
        << "vf.mul src only supports supported types, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s1Dt.GetBit() == 16 || s1Dt.GetBit() == 32 || s1Dt.GetBit() == 64))
        << "vf.mul src only supports supported types, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.mul requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mul");
    if (s0Dt.GetBit() == 64) {
        // B64 mul: Mirrors AscendC MulB64Impl (vmull + vmula).
        // MulB64Impl uses B64TraitOneToTraitTwo to split both sources into b32 halves,
        // then vmull (32x32→64 long multiply) + vmula (cross-term accumulate).
        std::string p = dst + "_mul_";
        // AscendC MulB64Impl: vmull always uses uint32_t (unsigned 32x32→64);
        // vmula uses int32_t for int64_t (signed cross-term), uint32_t for uint64_t.
        std::string vmullCast = "(RegTensor<uint32_t>&)";
        std::string vmulaCast = (s0Dt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        // Deinterleave src0 and src1 to b32 halves
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        // MulB64: vmull (lo0 × lo1 → dst_lo, dst_hi) + vmula (lo0 × hi1 → dst_hi) + vmula (hi0 × lo1 → dst_hi)
        std::string loD = p + "_lod_";
        std::string hiD = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
        codegen.Emit("vmull(" + vmullCast + loD + ", " + vmullCast + hiD + ", " + vmullCast + lo0 + ", " + vmullCast +
                     lo1 + ", " + packedM + ");");
        codegen.Emit("vmula(" + vmulaCast + hiD + ", " + vmulaCast + lo0 + ", " + vmulaCast + hi1 + ", " + packedM +
                     ", MODE_ZEROING);");
        codegen.Emit("vmula(" + vmulaCast + hiD + ", " + vmulaCast + hi0 + ", " + vmulaCast + lo1 + ", " + packedM +
                     ", MODE_ZEROING);");
        // Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
    } else {
        codegen.Emit("vmul(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// MulAddDst — vmula (hardware FMA: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulAddDst(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask], or
    // [dst, old, src0, src1, mask] on a rebinding assignment (the previous dst
    // value is an explicit argument, copied into the fresh dst before the
    // accumulate instruction).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4 || op->args_.size() == 5)
        << op->name_ << " requires 4 args (dst, src0, src1, mask) or 5 (dst, old, src0, src1, mask)";
    size_t base = op->args_.size() == 5 ? 1 : 0;
    DataType s0Dt = GetExprDtype(op->args_[base + 1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s0Dt.GetBit() == 16 || s0Dt.GetBit() == 32 || s0Dt.GetBit() == 64))
        << "vf.mul_add_dst src only supports supported types, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[base + 2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s1Dt.GetBit() == 16 || s1Dt.GetBit() == 32 || s1Dt.GetBit() == 64))
        << "vf.mul_add_dst src only supports supported types, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << op->name_ << " requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[base + 1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[base + 2]);
    std::string mask = codegen.GetExprAsCode(op->args_[base + 3]);
    if (base == 1) {
        codegen.Emit(dst + " = " + codegen.GetExprAsCode(op->args_[1]) + ";");
    }
    std::string mode = VFZeroingOnly(op, op->name_);
    codegen.Emit("vmula(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Div — vdiv
// ============================================================================

static std::string EmitVFDiv(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.div requires 4 args (dst, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[1]);
    // No vdiv overloads for 8-bit ints (mirrors AscendC DivImpl: u16..i64 + half/float).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithIntType(s0Dt) || s0Dt == DataType::FP16 || s0Dt == DataType::FP32) &&
                          s0Dt != DataType::INT8 && s0Dt != DataType::UINT8)
        << "vf.div src0 only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithIntType(s1Dt) || s1Dt == DataType::FP16 || s1Dt == DataType::FP32) &&
                          s1Dt != DataType::INT8 && s1Dt != DataType::UINT8)
        << "vf.div src1 only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.div requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    const bool usePrecision = op->HasKwarg("precision") && op->GetKwarg<bool>("precision");
    // High-precision mode supports FP16/FP32 only; validated up front before
    // any emission (B64 division and integer types take the standard path).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !usePrecision || s0Dt == DataType::FP16 || s0Dt == DataType::FP32)
        << "vf.div high-precision mode only supports FP16/FP32, got " << DTypeStr(s0Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.div");
    if (s0Dt.GetBit() == 64) {
        // B64 division (INT64/UINT64): Newton-Raphson reciprocal refinement
        // mirroring AscendC DivS64Impl / DivU64Impl. No native b64 vdiv.
        EmitB64Div(codegen, dst, src0, src1, mask, s0Dt == DataType::INT64);
        return "";
    }
    if (usePrecision) {
        // Mirrors AscendC high-precision division: error-complementation
        // quotient correction with inf/nan/zero bypass and subnormal-input
        // scaling. FP32 follows DivPrecisionImpl (vec_binary_impl.h:868-976);
        // FP16 follows DivIEEE754HalfImpl (vec_binary_impl.h:1199-1413).
        if (s0Dt == DataType::FP16) {
            // IEEE754 manual half division: normalize subnormal inputs by 2^10,
            // standardize exponents, divide, then compensate, clamp overflow/
            // underflow and bypass inf/zero/nan lanes.
            const std::string h = dst + "_h_";
            const std::string t0 = h + "t0", t1 = h + "t1", t2 = h + "t2", z1 = h + "z1", z2 = h + "z2";
            const std::string a0abs = h + "a0abs", a0sub = h + "a0sub", a0nrm = h + "a0nrm";
            const std::string a0all = h + "a0all", a0abn = h + "a0abn", a0exp = h + "a0exp";
            const std::string b1abs = h + "b1abs", b1sub = h + "b1sub", b1nrm = h + "b1nrm";
            const std::string b1all = h + "b1all", b1abn = h + "b1abn", b1exp = h + "b1exp";
            const std::string dsgn = h + "dsgn", scl = h + "scl";
            const std::string m0 = h + "m0", ms0n = h + "ms0n", ms0s = h + "ms0s", ms1n = h + "ms1n";
            const std::string ms1s = h + "ms1s", mt = h + "mt", mnan = h + "mnan", minf = h + "minf";
            const std::string mz0 = h + "mz0", mz1 = h + "mz1", mv = h + "mv", mn = h + "mn";
            codegen.Emit("union { uint16_t i; half f; } " + h + "thr_ = {0x03FF};");
            codegen.Emit("union { uint16_t i; half f; } " + h + "enl_ = {0x6400};");
            codegen.Emit("union { uint16_t i; half f; } " + h + "erd_ = {0x1400};");
            EmitRtWrapDecl(codegen, ir::DataType::FP16,
                           a0abs + ", " + a0sub + ", " + a0nrm + ", " + a0all + ", " + a0abn);
            EmitRtWrapDecl(codegen, ir::DataType::FP16,
                           b1abs + ", " + b1sub + ", " + b1nrm + ", " + b1all + ", " + b1abn);
            EmitRtWrapDecl(codegen, ir::DataType::FP16, z1 + ", " + z2);
            EmitRtWrapDecl(codegen, ir::DataType::UINT16, t0 + ", " + t2 + ", " + a0exp + ", " + b1exp + ", " + dsgn);
            EmitRtWrapDecl(codegen, ir::DataType::INT16, t1 + ", " + scl);
            codegen.Emit("MaskReg " + m0 + ", " + ms0n + ", " + ms0s + ", " + ms1n + ", " + ms1s + ", " + mt + ";");
            codegen.Emit("MaskReg " + mnan + ", " + minf + ", " + mz0 + ", " + mz1 + ", " + mv + ", " + mn + ";");
            // Acquire valid numbers (no inf, no 0): DivIEEE754HalfImpl:1259-1281
            codegen.Emit("vabs(" + a0abs + ", " + src0 + ", " + mask + ", " + mode + ");");
            codegen.Emit("vabs(" + b1abs + ", " + src1 + ", " + mask + ", " + mode + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7C00, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + minf + ", (RegTensor<uint16_t>&)" + a0abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("vcmp_eq(" + mt + ", (RegTensor<uint16_t>&)" + b1abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("por(" + mv + ", " + minf + ", " + mt + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x0, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + mz0 + ", (RegTensor<uint16_t>&)" + a0abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("por(" + mv + ", " + mv + ", " + mz0 + ", " + mask + ");");
            codegen.Emit("vcmp_eq(" + mz1 + ", (RegTensor<uint16_t>&)" + b1abs + ", " + t0 + ", " + mask + ");");
            codegen.Emit("por(" + mv + ", " + mv + ", " + mz1 + ", " + mask + ");");
            codegen.Emit("pnot(" + mv + ", " + mv + ", " + mask + ");");
            // Normalize subnormal elements of src0/src1 (1284-1298)
            codegen.Emit("vcmps_lt(" + ms0s + ", " + a0abs + ", " + h + "thr_.f, " + mask + ");");
            codegen.Emit("pnot(" + ms0n + ", " + ms0s + ", " + mask + ");");
            codegen.Emit("vmuls(" + a0sub + ", " + src0 + ", " + h + "enl_.f, " + ms0s + ", " + mode + ");");
            codegen.Emit("vcmps_lt(" + ms1s + ", " + b1abs + ", " + h + "thr_.f, " + mask + ");");
            codegen.Emit("pnot(" + ms1n + ", " + ms1s + ", " + mask + ");");
            codegen.Emit("vmuls(" + b1sub + ", " + src1 + ", " + h + "enl_.f, " + ms1s + ", " + mode + ");");
            codegen.Emit("vsel(" + a0all + ", " + src0 + ", " + a0sub + ", " + ms0n + ");");
            codegen.Emit("vsel(" + b1all + ", " + src1 + ", " + b1sub + ", " + ms1n + ");");
            // Standardize the exponent bits of src0 and src1 (1301-1317).
            // Both steps are u16 bit-pattern ops (mask the exponent field away,
            // then integer-add the 1.0h exponent bits back in), so dst and the
            // recursive src0 are taken through the u16 register view as well:
            // vand/vadd require all three data operands to share one type.
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x83FF, " + mask + ", " + mode + ");");
            codegen.Emit("vand((RegTensor<uint16_t>&)" + a0nrm + ", (RegTensor<uint16_t>&)" + a0all + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vand((RegTensor<uint16_t>&)" + b1nrm + ", (RegTensor<uint16_t>&)" + b1all + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x3C00, " + mask + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + a0nrm + ", (RegTensor<uint16_t>&)" + a0nrm + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + b1nrm + ", (RegTensor<uint16_t>&)" + b1nrm + ", " + t0 + ", " +
                         mv + ", " + mode + ");");
            codegen.Emit("vsel(" + a0nrm + ", " + a0nrm + ", " + a0all + ", " + mv + ");");
            codegen.Emit("vsel(" + b1nrm + ", " + b1nrm + ", " + b1all + ", " + mv + ");");
            codegen.Emit("vabs(" + a0abn + ", " + a0nrm + ", " + mv + ", " + mode + ");");
            codegen.Emit("vabs(" + b1abn + ", " + b1nrm + ", " + mv + ", " + mode + ");");
            codegen.Emit("vcmp_le(" + mn + ", " + a0abn + ", " + b1abn + ", " + mv + ");");
            codegen.Emit("vdiv(" + dst + ", " + a0nrm + ", " + b1nrm + ", " + mask + ", " + mode + ");");
            // Normalization compensation for subnormal operands (1324-1334)
            codegen.Emit("pand(" + m0 + ", " + ms0s + ", " + ms1n + ", " + mask + ");");
            codegen.Emit("vmuls(" + z1 + ", " + dst + ", " + h + "erd_.f, " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pand(" + m0 + ", " + ms0n + ", " + ms1s + ", " + mask + ");");
            codegen.Emit("vmuls(" + z1 + ", " + dst + ", " + h + "enl_.f, " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // Preserve sign for the exception handling below (1337-1339)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x8000, " + mask + ", " + mode + ");");
            codegen.Emit("vand(" + dsgn + ", (RegTensor<uint16_t>&)" + dst + ", " + t0 + ", " + mask + ", " + mode +
                         ");");
            // Exponent subtraction (effectively fp number division) (1342-1356)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7C00, " + mask + ", " + mode + ");");
            codegen.Emit("vand(" + a0exp + ", (RegTensor<uint16_t>&)" + a0all + ", " + t0 + ", " + mask + ", " + mode +
                         ");");
            codegen.Emit("vand(" + b1exp + ", (RegTensor<uint16_t>&)" + b1all + ", " + t0 + ", " + mask + ", " + mode +
                         ");");
            codegen.Emit("vshrs(" + a0exp + ", " + a0exp + ", (int16_t)10, " + mask + ", " + mode + ");");
            codegen.Emit("vshrs(" + b1exp + ", " + b1exp + ", (int16_t)10, " + mask + ", " + mode + ");");
            codegen.Emit("vsub(" + scl + ", (RegTensor<int16_t>&)" + a0exp + ", (RegTensor<int16_t>&)" + b1exp + ", " +
                         mask + ", " + mode + ");");
            codegen.Emit("vadds(" + scl + ", " + scl + ", (int16_t)15, " + mask + ", " + mode + ");");
            // scale == -9: clamp to signed min denormal (1360-1368)
            codegen.Emit("vdup(" + t1 + ", (int16_t)-9, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + m0 + ", " + scl + ", " + t1 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x1, " + m0 + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z1 + ", " + dsgn + ", " + t0 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vdup(" + t2 + ", (uint16_t)0x0, " + m0 + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z2 + ", " + dsgn + ", " + t2 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + z1 + ", " + z2 + ", " + z1 + ", " + mn + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mask + ");");
            codegen.Emit("pand(" + mv + ", " + m0 + ", " + mv + ", " + mask + ");");
            // scale < -9: underflow to signed zero (1370-1375)
            codegen.Emit("vcmp_lt(" + m0 + ", " + scl + ", " + t1 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x0, " + mask + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z1 + ", " + dsgn + ", " + t0 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mask + ");");
            codegen.Emit("pand(" + mv + ", " + m0 + ", " + mv + ", " + mask + ");");
            // scale == 31: double the result and fix the exponent (1377-1384)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x1F, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_eq(" + m0 + ", " + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t1 + ", (int16_t)0x1, " + m0 + ", " + mode + ");");
            codegen.Emit("vsub(" + t1 + ", " + scl + ", " + t1 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + scl + ", " + t1 + ", " + scl + ", " + m0 + ");");
            codegen.Emit("vmuls(" + z1 + ", " + dst + ", 2.0f, " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // scale > 31: overflow to signed infinity (1386-1394)
            codegen.Emit("vcmp_gt(" + m0 + ", " + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + mask + ");");
            codegen.Emit("pand(" + m0 + ", " + m0 + ", " + mv + ", " + mask + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7C00, " + mask + ", " + mode + ");");
            codegen.Emit("vadd((RegTensor<uint16_t>&)" + z1 + ", " + dsgn + ", " + t0 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mask + ");");
            codegen.Emit("pand(" + mv + ", " + m0 + ", " + mv + ", " + mask + ");");
            // scale > 0: rescale by 2^(10*scale) (1396-1403)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x0, " + mv + ", " + mode + ");");
            codegen.Emit("vcmp_gt(" + m0 + ", " + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + mv + ");");
            codegen.Emit("vshls(" + t1 + ", " + scl + ", (int16_t)10, " + m0 + ", " + mode + ");");
            codegen.Emit("vmul(" + z1 + ", " + dst + ", (RegTensor<half>&)" + t1 + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // scale <= 0: subnormal result, scale by half(512 >> |scale|) (1405-1416)
            codegen.Emit("pnot(" + m0 + ", " + m0 + ", " + mv + ");");
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x200, " + m0 + ", " + mode + ");");
            codegen.Emit("vabs(" + scl + ", " + scl + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vshr(" + scl + ", (RegTensor<int16_t>&)" + t0 + ", " + scl + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vmul(" + z1 + ", " + dst + ", (RegTensor<half>&)" + scl + ", " + m0 + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + z1 + ", " + dst + ", " + m0 + ");");
            // Set output with nan input to nan (1418-1424)
            codegen.Emit("vdup(" + t0 + ", (uint16_t)0x7E00, " + mask + ", " + mode + ");");
            codegen.Emit("vcmp_ne(" + mnan + ", " + a0abs + ", " + a0abs + ", " + mask + ");");
            codegen.Emit("vcmp_ne(" + mt + ", " + b1abs + ", " + b1abs + ", " + mask + ");");
            codegen.Emit("por(" + mnan + ", " + mnan + ", " + mt + ", " + mask + ");");
            codegen.Emit("vsel(" + dst + ", (RegTensor<half>&)" + t0 + ", " + dst + ", " + mnan + ");");
            return "";
        }
        const std::string p = dst + "_p_";
        const std::string pall = p + "all", nz = p + "nz", infnan = p + "infn", z = p + "z", q0 = p + "q0";
        const std::string mInf = p + "minf", mZero = p + "mzero", mScale = p + "mscale";
        const std::string mCmp = p + "mcmp", expbits = p + "expb", scratch = p + "scr", expo = p + "expo";
        const std::string kreg = p + "k", thrvec = p + "thr", zerovec = p + "zv", newexp = p + "newexp";
        const std::string scalebits = p + "sb", scale = p + "sc", one = p + "one";
        const std::string asc = p + "asc", bsc = p + "bsc", y = p + "y";
        const std::string r = p + "r", rpre = p + "rpre", rnext = p + "rnext", zpre = p + "zpre", znext = p + "znext";
        codegen.Emit("MaskReg " + pall + " = pset_b8(PAT_ALL);");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, nz);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, infnan);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, z);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, q0);
        codegen.Emit("MaskReg " + mInf + ";");
        codegen.Emit("MaskReg " + mZero + ";");
        // Inf/nan/zero bypass (DivPrecisionImpl:908-914): set the sign bit of the
        // quotient, then >= 0xFF800000 (unsigned) catches +/-Inf, +/-0 and NaNs.
        codegen.Emit("vdup(" + nz + ", (int32_t)0x80000000, " + pall + ", " + mode + ");");
        codegen.Emit("vdiv(" + z + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
        codegen.Emit("vor(" + infnan + ", (RegTensor<uint32_t>&)" + z + ", " + nz + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmov(" + q0 + ", " + z + ");");
        codegen.Emit("vcmps_eq(" + mZero + ", " + z + ", 0.0f, " + mask + ");");
        codegen.Emit("vcmps_ge(" + mInf + ", " + infnan + ", (uint32_t)0xFF800000, " + mask + ");");
        codegen.Emit("por(" + mInf + ", " + mInf + ", " + mZero + ", " + mask + ");");
        // Subnormal-input scaling (DivPrecisionImpl:916-950): scale a/b by 2^k with
        // k = max(-64 - exp(a), 0) so the residual stays in the normal range.
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, expbits);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, scratch);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, expo);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, kreg);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, thrvec);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, zerovec);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, newexp);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, scalebits);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, scale);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, one);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, asc);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, bsc);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, y);
        codegen.Emit("MaskReg " + mScale + ";");
        codegen.Emit("vdup(" + scratch + ", (int32_t)0x7F800000, " + mask + ", " + mode + ");");
        codegen.Emit("vand(" + expbits + ", (RegTensor<uint32_t>&)" + src0 + ", " + scratch + ", " + mask + ", " +
                     mode + ");");
        codegen.Emit("vshrs(" + expbits + ", " + expbits + ", (int16_t)23, " + mask + ", " + mode + ");");
        codegen.Emit("vdup(" + scratch + ", (int32_t)127, " + mask + ", " + mode + ");");
        codegen.Emit("vsub(" + expo + ", (RegTensor<int32_t>&)" + expbits + ", (RegTensor<int32_t>&)" + scratch + ", " +
                     mask + ", " + mode + ");");
        codegen.Emit("vcmps_lt(" + mScale + ", " + expo + ", (int32_t)-64, " + mask + ");");
        codegen.Emit("vdup(" + thrvec + ", (int32_t)-64, " + mask + ", " + mode + ");");
        codegen.Emit("vdup(" + zerovec + ", (int32_t)0, " + mask + ", " + mode + ");");
        codegen.Emit("vsub(" + kreg + ", " + thrvec + ", " + expo + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmax(" + kreg + ", " + kreg + ", " + zerovec + ", " + mask + ", " + mode + ");");
        codegen.Emit("vadds(" + newexp + ", " + kreg + ", (int32_t)127, " + mScale + ", " + mode + ");");
        codegen.Emit("vshls(" + scalebits + ", (RegTensor<uint32_t>&)" + newexp + ", (int16_t)23, " + mScale + ", " +
                     mode + ");");
        codegen.Emit("vdup(" + one + ", 1.0f, " + mask + ", " + mode + ");");
        codegen.Emit("vsel(" + scale + ", (RegTensor<float>&)" + scalebits + ", " + one + ", " + mScale + ");");
        codegen.Emit("vmul(" + asc + ", " + src0 + ", " + scale + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmul(" + bsc + ", " + src1 + ", " + scale + ", " + mask + ", " + mode + ");");
        // Corrected quotient (DivPrecisionImpl:952-975): r = a' - x1*b', pick the
        // bit-pattern neighbor of x1 (-1/+1 ulp) with the smaller |residual|, then
        // bypass invalid lanes back to the raw quotient.
        codegen.Emit("vmuls(" + y + ", " + bsc + ", -1.0f, " + mask + ", " + mode + ");");
        EmitRtWrapDecl(codegen, ir::DataType::FP32, r);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, rpre);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, rnext);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, zpre);
        EmitRtWrapDecl(codegen, ir::DataType::FP32, znext);
        codegen.Emit("MaskReg " + mCmp + ";");
        codegen.Emit("vmov(" + r + ", " + asc + ");");
        codegen.Emit("vmula(" + r + ", " + z + ", " + y + ", " + mask + ", " + mode + ");");
        codegen.Emit("vadds((RegTensor<int32_t>&)" + zpre + ", (RegTensor<int32_t>&)" + z + ", (int32_t)-1, " + mask +
                     ", " + mode + ");");
        codegen.Emit("vadds((RegTensor<int32_t>&)" + znext + ", (RegTensor<int32_t>&)" + z + ", (int32_t)1, " + mask +
                     ", " + mode + ");");
        codegen.Emit("vmov(" + rpre + ", " + asc + ");");
        codegen.Emit("vmov(" + rnext + ", " + asc + ");");
        codegen.Emit("vmula(" + rpre + ", " + zpre + ", " + y + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmula(" + rnext + ", " + znext + ", " + y + ", " + mask + ", " + mode + ");");
        codegen.Emit("vabs(" + r + ", " + r + ", " + mask + ", " + mode + ");");
        codegen.Emit("vabs(" + rpre + ", " + rpre + ", " + mask + ", " + mode + ");");
        codegen.Emit("vabs(" + rnext + ", " + rnext + ", " + mask + ", " + mode + ");");
        codegen.Emit("vcmp_lt(" + mCmp + ", " + r + ", " + rpre + ", " + mask + ");");
        codegen.Emit("vsel(" + r + ", " + r + ", " + rpre + ", " + mCmp + ");");
        codegen.Emit("vsel(" + z + ", " + z + ", " + zpre + ", " + mCmp + ");");
        codegen.Emit("vcmp_lt(" + mCmp + ", " + rnext + ", " + r + ", " + mask + ");");
        codegen.Emit("vsel(" + z + ", " + znext + ", " + z + ", " + mCmp + ");");
        codegen.Emit("vsel(" + dst + ", " + q0 + ", " + z + ", " + mInf + ");");
    } else {
        codegen.Emit("vdiv(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Muls — vmuls
// ============================================================================

static std::string EmitVFMuls(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, scalar, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.muls requires 4 args (dst, src, scalar, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (srcDt == DataType::INT16 || srcDt == DataType::UINT16 || srcDt == DataType::INT32 ||
                       srcDt == DataType::UINT32 || srcDt == DataType::INT64 || srcDt == DataType::UINT64 ||
                       srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.muls src only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(srcDt);
    DataType scalarDt = GetExprDtype(op->args_[2]);
    if (scalarDt == DataType::INDEX || scalarDt == DataType::INT64) {
        scalarDt = srcDt;
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (scalarDt == DataType::INT16 || scalarDt == DataType::UINT16 || scalarDt == DataType::INT32 ||
                       scalarDt == DataType::UINT32 || scalarDt == DataType::INT64 || scalarDt == DataType::UINT64 ||
                       scalarDt == DataType::FP16 || scalarDt == DataType::FP32))
        << "vf.muls scalar only supports INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32, got " << DTypeStr(scalarDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalarStr = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.muls");
    scalarStr = cce::CoerceScalarToInt(op->args_[2], srcDt, scalarStr);
    if (srcDt.GetBit() == 64) {
        // B64 Muls: dst = src * scalar. Mirrors AscendC MulsImpl: Duplicate(scalar) + Mul(src, scalar_reg).
        std::string p = dst + "_muls_";
        std::string castType = (srcDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        // AscendC MulB64Impl: vmull always uses uint32_t (unsigned 32x32→64);
        // vmula uses int32_t for int64_t (signed cross-term), uint32_t for uint64_t.
        std::string vmullCast = "(RegTensor<uint32_t>&)";
        std::string vmulaCast = (srcDt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        // Broadcast scalar to b32 halves (mirrors DuplicateB64Impl)
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string loSc = p + "_losc_";
        std::string hiSc = p + "_hisc_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loSc);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiSc);
        codegen.Emit("vdup(" + loSc + ", (int32_t)(" + castType + "(" + scalarStr + ")), " + allM + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hiSc + ", (int32_t)((" + castType + "(" + scalarStr + ")) >> 32), " + allM +
                     ", MODE_ZEROING);");
        // Deinterleave src to b32 halves (mirrors B64TraitOneToTraitTwo)
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        // MulB64: vmull (lo_s × lo_sc → dst_lo, dst_hi) + vmula (lo_s × hi_sc → dst_hi) + vmula (hi_s × lo_sc → dst_hi)
        std::string loD = p + "_lod_";
        std::string hiD = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
        codegen.Emit("vmull(" + vmullCast + loD + ", " + vmullCast + hiD + ", " + vmullCast + loS + ", " + vmullCast +
                     loSc + ", " + packedM + ");");
        codegen.Emit("vmula(" + vmulaCast + hiD + ", " + vmulaCast + loS + ", " + vmulaCast + hiSc + ", " + packedM +
                     ", MODE_ZEROING);");
        codegen.Emit("vmula(" + vmulaCast + hiD + ", " + vmulaCast + hiS + ", " + vmulaCast + loSc + ", " + packedM +
                     ", MODE_ZEROING);");
        // Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
    } else {
        codegen.Emit("vmuls(" + dst + ", " + src + ", " + scalarStr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Ln — vln (natural logarithm, basic precision)
// ============================================================================

static std::string EmitVFUnary(const ir::CallPtr& op, codegen::CodegenBase& codegenBase, const std::string& opName,
                               const std::string& instruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << opName << " requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, opName);
    codegen.Emit(instruction + "(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFLn(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.ln requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.ln src only supports FP16/FP32, got " << DTypeStr(srcDt);
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src = codegen.GetExprAsCode(op->args_[1]);
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = VFZeroingOnly(op, "vf.ln");
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = srcDt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + dc + ";");
        if (srcDt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -6.931471805599453094172f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -15.9423851528787421f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
        return "";
    }
    return EmitVFUnary(op, codegenBase, "vf.ln", "vln");
}

// ============================================================================
// Log — vln (natural logarithm, same as Ln on A5/dav_3510)
// ============================================================================

static std::string EmitVFLog(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.log requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.log src only supports FP16/FP32, got " << DTypeStr(srcDt);
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src = codegen.GetExprAsCode(op->args_[1]);
        std::string mask = codegen.GetExprAsCode(op->args_[2]);
        std::string mode = VFZeroingOnly(op, "vf.log");
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = srcDt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + dc + ";");
        if (srcDt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -6.931471805599453094172f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -15.9423851528787421f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
        return "";
    }
    return EmitVFUnary(op, codegenBase, "vf.log", "vln");
}

// ============================================================================
// Min — vmin
// ============================================================================

static std::string EmitVFMin(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.min requires 4 args (dst, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0Dt) || s0Dt == DataType::FP16 ||
                                                    s0Dt == DataType::FP32 || s0Dt == DataType::BF16))
        << "vf.min src only supports supported types, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s1Dt) || s1Dt == DataType::FP16 ||
                                                    s1Dt == DataType::FP32 || s1Dt == DataType::BF16))
        << "vf.min src only supports supported types, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.min requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.min");
    if (s0Dt.GetBit() == 64) {
        // B64 min via deinterleave + b32 compare + vsel + interleave.
        // min = src1 when src0 > src1, else src0 (i.e., pick the smaller).
        std::string p = dst + "_min_";
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string hiGt = p + "_higt_";
        std::string hiEq = p + "_hieq_";
        std::string loGt = p + "_logt_";
        codegen.Emit("MaskReg " + hiGt + ";");
        codegen.Emit("MaskReg " + hiEq + ";");
        codegen.Emit("MaskReg " + loGt + ";");
        if (s0Dt == DataType::INT64) {
            codegen.Emit("vcmp_gt(" + hiGt + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 + ", " +
                         allM + ");");
        } else {
            codegen.Emit("vcmp_gt(" + hiGt + ", " + hi0 + ", " + hi1 + ", " + allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", " + hi0 + ", " + hi1 + ", " + allM + ");");
        }
        codegen.Emit("vcmp_gt(" + loGt + ", " + lo0 + ", " + lo1 + ", " + allM + ");");
        std::string eqLo = p + "_eqlo_";
        std::string s0Gt = p + "_s0gt_";
        codegen.Emit("MaskReg " + eqLo + ";");
        codegen.Emit("MaskReg " + s0Gt + ";");
        codegen.Emit("pand(" + eqLo + ", " + hiEq + ", " + loGt + ", " + allM + ");");
        codegen.Emit("por(" + s0Gt + ", " + hiGt + ", " + eqLo + ", " + allM + ");");
        // Select: min = src0 > src1 ? src1 : src0 (pick smaller)
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiDst);
        codegen.Emit("vsel(" + loDst + ", " + lo1 + ", " + lo0 + ", " + s0Gt + ");");
        codegen.Emit("vsel(" + hiDst + ", " + hi1 + ", " + hi0 + ", " + s0Gt + ");");
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmin(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Exp — vexp
// ============================================================================

static std::string EmitVFExp(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.exp requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.exp src only supports FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.exp");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string ms = dst + "_sub_";
        std::string tw = dst + "_two_";
        std::string t0 = dst + "_t0_";
        std::string t1 = dst + "_t1_";
        std::string ctype = srcDt.ToCTypeString();
        codegen.Emit("MaskReg " + ms + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + tw + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + t0 + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + t1 + ";");
        if (srcDt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
        }
        codegen.Emit("vexp(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
        codegen.Emit("vcmps_le(" + ms + ", " + dst + ", " + dst + "_thr_.f, " + mask + ");");
        codegen.Emit("vdup(" + tw + ", 2.0f, " + ms + ", " + mode + ");");
        codegen.Emit("vdiv(" + t0 + ", " + src + ", " + tw + ", " + ms + ", " + mode + ");");
        codegen.Emit("vexp(" + t0 + ", " + t0 + ", " + ms + ", " + mode + ");");
        codegen.Emit("vmul(" + t1 + ", " + t0 + ", " + t0 + ", " + ms + ", " + mode + ");");
        codegen.Emit("vsel(" + dst + ", " + t1 + ", " + dst + ", " + ms + ");");
    } else {
        codegen.Emit("vexp(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Abs — vabs
// ============================================================================

static std::string EmitVFAbs(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.abs requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithSignedIntType(srcDt) || srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.abs src only supports INT8/INT16/INT32/INT64/FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.abs");
    if (srcDt == DataType::INT64) {
        // B64 abs: abs(x) = x if x >= 0, -x if x < 0.
        // Mirrors AscendC AbsB64Impl: vbr(0) + vcmp_lt + Sub(vsubc) + SubC(vsubcs) + vsel.
        std::string p = dst + "_abs_";
        // Deinterleave src to get b32 low/high halves
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        // Zero register (vbr equivalent)
        std::string zeroReg = p + "_zero_";
        EmitRtWrapDecl(codegen, ir::DataType::INT32, zeroReg);
        codegen.Emit("vdup(" + zeroReg + ", 0, " + allM + ", MODE_ZEROING);");
        // Sign mask: hi_s < 0 (signed)
        std::string signM = p + "_sign_";
        codegen.Emit("MaskReg " + signM + ";");
        codegen.Emit("vcmp_lt(" + signM + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + zeroReg + ", " +
                     allM + ");");
        // Negate via borrow-chain sub: neg = 0 - src
        // vsubc: borrow = (0 < lo_s), neg_lo = 0 - lo_s
        // vsubcs: neg_hi = 0 - hi_s - borrow
        std::string borrow = p + "_borrow_";
        std::string loN = p + "_nlo_";
        std::string hiN = p + "_nhi_";
        codegen.Emit("MaskReg " + borrow + ";");
        EmitRtWrapDecl(codegen, ir::DataType::INT32, loN);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, hiN);
        codegen.Emit("vsubc(" + borrow + ", " + loN + ", " + zeroReg + ", (RegTensor<int32_t>&)" + loS + ", " + signM +
                     ");");
        codegen.Emit("vsubcs(" + borrow + ", " + hiN + ", " + zeroReg + ", (RegTensor<int32_t>&)" + hiS + ", " +
                     borrow + ", " + signM + ");");
        // Select: if negative -> neg, else -> src (per b32 half)
        // All vsel operands must be the same type (int32_t).
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::INT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, hiDst);
        codegen.Emit("vsel(" + loDst + ", " + loN + ", (RegTensor<int32_t>&)" + loS + ", " + signM + ");");
        codegen.Emit("vsel(" + hiDst + ", " + hiN + ", (RegTensor<int32_t>&)" + hiS + ", " + signM + ");");
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vabs(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Not — vnot
// ============================================================================

static std::string EmitVFNot(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.not_ requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      srcDt == DataType::INT8 || srcDt == DataType::UINT8 || srcDt == DataType::BOOL ||
                          srcDt == DataType::INT16 || srcDt == DataType::UINT16 || srcDt == DataType::INT32 ||
                          srcDt == DataType::UINT32 || srcDt == DataType::FP16 || srcDt == DataType::FP32 ||
                          srcDt == DataType::INT64 || srcDt == DataType::UINT64)
        << "vf.not_ src only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/FP16/FP32/INT64/UINT64, got "
        << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    if (IsSrcMaskReg(op, 1)) {
        codegen.Emit("pnot(" + dst + ", " + src + ", " + mask + ");");
        return "";
    }
    std::string mode = VFZeroingOnly(op, "vf.not_");
    if (srcDt.GetBit() == 64) {
        // B64 bitwise via b32 vnot (mirrors AscendC NotImpl b64 path, see EmitB64Bitwise).
        EmitB64Bitwise(codegen, "vnot", {src}, dst, mask, mode, dst + "_not_");
    } else {
        codegen.Emit("vnot(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Sqrt — vsqrt
// ============================================================================

static std::string EmitVFSqrt(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.sqrt requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.sqrt src only supports FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.sqrt");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = srcDt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + dc + ";");
        if (srcDt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 4096.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vsqrt(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            // AscendC SqrtImpl PRECISION_1ULP_FTZ_FALSE (half): multiplyFactor1 = 0x2400 = 2^-6.
            // sqrt(2^12) = 2^6, so the scale-down compensation is 2^-6 — NOT the FP32
            // branch's 2^-12, which made subnormal results come out 1/64 too small.
            codegen.Emit("vmuls(" + tp + ", " + dc + ", 0.015625f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 16777216.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vsqrt(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + tp + ", " + dc + ", 0.000244140625f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
    } else {
        codegen.Emit("vsqrt(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Relu — vrelu
// ============================================================================

static std::string EmitVFRelu(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.relu requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::INT32 || srcDt == DataType::INT64 ||
                                                    srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.relu src only supports INT32/INT64/FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.relu");
    if (srcDt == DataType::INT64) {
        // B64 relu (mirrors AscendC ReluB64Impl: Maxs(src, 0)): deinterleave +
        // scalar(0) two-stage compare + vsel + interleave. No native b64 vrelu.
        std::string p = dst + "_relu_";
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string loSc = p + "_losc_", hiSc = p + "_hisc_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loSc);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiSc);
        codegen.Emit("vdup(" + loSc + ", (int32_t)0, " + allM + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hiSc + ", (int32_t)0, " + allM + ", MODE_ZEROING);");
        // src > 0 iff hi > 0 (signed) OR (hi == 0 AND lo > 0, unsigned).
        std::string hiGt = p + "_higt_", hiEq = p + "_hieq_", loGt = p + "_logt_";
        std::string eqLo = p + "_eqlo_", gt = p + "_gt_";
        codegen.Emit("MaskReg " + hiGt + ";");
        codegen.Emit("MaskReg " + hiEq + ";");
        codegen.Emit("MaskReg " + loGt + ";");
        codegen.Emit("MaskReg " + eqLo + ";");
        codegen.Emit("MaskReg " + gt + ";");
        codegen.Emit("vcmp_gt(" + hiGt + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + hiSc + ", " +
                     packedM + ");");
        codegen.Emit("vcmp_eq(" + hiEq + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + hiSc + ", " +
                     packedM + ");");
        codegen.Emit("vcmp_gt(" + loGt + ", " + loS + ", " + loSc + ", " + packedM + ");");
        codegen.Emit("pand(" + eqLo + ", " + hiEq + ", " + loGt + ", " + packedM + ");");
        codegen.Emit("por(" + gt + ", " + hiGt + ", " + eqLo + ", " + packedM + ");");
        std::string loD = p + "_lod_", hiD = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
        codegen.Emit("vsel(" + loD + ", " + loS + ", " + loSc + ", " + gt + ");");
        codegen.Emit("vsel(" + hiD + ", " + hiS + ", " + hiSc + ", " + gt + ");");
        EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
        return "";
    }
    codegen.Emit("vrelu(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Neg — vneg
// ============================================================================

static std::string EmitVFNeg(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.neg requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsArithSignedIntType(srcDt) || srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.neg src only supports INT8/INT16/INT32/INT64/FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.neg");
    if (srcDt == DataType::INT64 || srcDt == DataType::UINT64) {
        // B64 neg: dst = 0 - src. Mirrors AscendC NegB64Impl: Duplicate(0) + Sub(0, src).
        // Sub for b64 uses vsubc + vsubcs borrow-chain (mirrors SubB64Impl).
        std::string p = dst + "_neg_";
        std::string b32Cast = (srcDt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        // Zero register (vbr equivalent)
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string zeroReg = p + "_zero_";
        EmitRtWrapDecl(codegen, ir::DataType::INT32, zeroReg);
        codegen.Emit("vdup(" + zeroReg + ", 0, " + allM + ", MODE_ZEROING);");
        // Deinterleave src to b32 halves (mirrors B64TraitOneToTraitTwo)
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        // SubB64: vsubc + vsubcs (borrow-chain sub: 0 - src)
        std::string borrow = p + "_borrow_";
        std::string loD = p + "_lod_";
        std::string hiD = p + "_hid_";
        codegen.Emit("MaskReg " + borrow + ";");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
        codegen.Emit("vsubc(" + borrow + ", " + b32Cast + loD + ", " + b32Cast + zeroReg + ", " + b32Cast + loS + ", " +
                     packedM + ");");
        codegen.Emit("vsubcs(" + borrow + ", " + b32Cast + hiD + ", " + b32Cast + zeroReg + ", " + b32Cast + hiS +
                     ", " + borrow + ", " + packedM + ");");
        // Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
    } else {
        codegen.Emit("vneg(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Adds — vadds (scalar addition)
// ============================================================================

static std::string EmitVFAdds(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, scalar, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.adds requires 4 args (dst, src, scalar, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(srcDt) || srcDt == DataType::FP16 ||
                                                    srcDt == DataType::FP32 || srcDt == DataType::BF16))
        << "vf.adds src only supports supported types, got " << DTypeStr(srcDt);
    DataType scalarDt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (scalarDt.IsInt() || scalarDt == DataType::FP16 ||
                                                    scalarDt == DataType::FP32 || scalarDt == DataType::BF16))
        << "vf.adds scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalarDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalarStr = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.adds");
    scalarStr = cce::CoerceScalarToInt(op->args_[2], srcDt, scalarStr);
    if (srcDt.GetBit() == 64) {
        // B64 Adds: dst = src + scalar.
        // Mirrors AscendC AddsImpl: Duplicate(scalar) + Add(src, scalar_reg).
        // Duplicate → DuplicateB64Impl (vdup b32 halves → interleave to b64).
        // Add → CalTraitOneByTransToTraitTwo (ppack mask → deinterleave both
        // → AddB64Impl: vaddc + vaddcs → interleave back).
        std::string p = dst + "_adds_";
        std::string castType = (srcDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string b32Cast = (srcDt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // 1. Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        // 2. Broadcast scalar to b32 halves (mirrors DuplicateB64Impl)
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string loSc = p + "_losc_";
        std::string hiSc = p + "_hisc_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loSc);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiSc);
        codegen.Emit("vdup(" + loSc + ", (int32_t)(" + castType + "(" + scalarStr + ")), " + allM + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hiSc + ", (int32_t)((" + castType + "(" + scalarStr + ")) >> 32), " + allM +
                     ", MODE_ZEROING);");
        // 3. Deinterleave src to b32 halves (mirrors B64TraitOneToTraitTwo)
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        // 4. AddB64: vaddc + vaddcs (mirrors AddB64Impl, using packed b32 mask)
        std::string carry = p + "_carry_";
        std::string loD = p + "_lod_";
        std::string hiD = p + "_hid_";
        codegen.Emit("MaskReg " + carry + ";");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiD);
        codegen.Emit("vaddc(" + carry + ", " + b32Cast + loD + ", " + b32Cast + loS + ", " + b32Cast + loSc + ", " +
                     packedM + ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32Cast + hiD + ", " + b32Cast + hiS + ", " + b32Cast + hiSc + ", " +
                     carry + ", " + packedM + ");");
        // 5. Interleave back (mirrors B64TraitTwoToTraitOne)
        EmitB64Interleave(codegen, dst, loD, hiD, p + "_ilv");
        // No separate EmitB64Zeroing — AscendC ZEROING mode relies on vaddc/vaddcs
        // producing 0 for inactive elements (mask-controlled). The packed b32 mask
        // ensures correct element selection.
    } else {
        codegen.Emit("vadds(" + dst + ", " + src + ", " + scalarStr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Mins — vmins (scalar minimum)
// ============================================================================

static std::string EmitVFMins(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, scalar, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.mins requires 4 args (dst, src, scalar, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(srcDt) || srcDt == DataType::FP16 ||
                                                    srcDt == DataType::FP32 || srcDt == DataType::BF16))
        << "vf.mins src only supports supported types, got " << DTypeStr(srcDt);
    DataType scalarDt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (scalarDt.IsInt() || scalarDt == DataType::FP16 ||
                                                    scalarDt == DataType::FP32 || scalarDt == DataType::BF16))
        << "vf.mins scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalarDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalarStr = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.mins");
    scalarStr = cce::CoerceScalarToInt(op->args_[2], srcDt, scalarStr);
    if (srcDt.GetBit() == 64) {
        // B64 mins: min(src, scalar) via deinterleave + scalar compare + vsel + interleave.
        std::string p = dst + "_mins_";
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        // Broadcast scalar to b32 halves
        std::string castType = (srcDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string loScalar = p + "_losc_";
        std::string hiScalar = p + "_hisc_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loScalar);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiScalar);
        codegen.Emit("vdup(" + loScalar + ", (int32_t)(" + castType + "(" + scalarStr + ")), " + allM +
                     ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hiScalar + ", (int32_t)((" + castType + "(" + scalarStr + ")) >> 32), " + allM +
                     ", MODE_ZEROING);");
        // Compare: src < scalar → pick src, else pick scalar
        std::string hiLt = p + "_hilt_";
        std::string hiEq = p + "_hieq_";
        std::string loLt = p + "_lolt_";
        codegen.Emit("MaskReg " + hiLt + ";");
        codegen.Emit("MaskReg " + hiEq + ";");
        codegen.Emit("MaskReg " + loLt + ";");
        if (srcDt == DataType::INT64) {
            codegen.Emit("vcmp_lt(" + hiLt + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + hiScalar +
                         ", " + allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + hiScalar +
                         ", " + allM + ");");
        } else {
            codegen.Emit("vcmp_lt(" + hiLt + ", " + hiS + ", " + hiScalar + ", " + allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", " + hiS + ", " + hiScalar + ", " + allM + ");");
        }
        codegen.Emit("vcmp_lt(" + loLt + ", " + loS + ", " + loScalar + ", " + allM + ");");
        std::string eqLo = p + "_eqlo_";
        std::string sLt = p + "_slt_";
        codegen.Emit("MaskReg " + eqLo + ";");
        codegen.Emit("MaskReg " + sLt + ";");
        codegen.Emit("pand(" + eqLo + ", " + hiEq + ", " + loLt + ", " + allM + ");");
        codegen.Emit("por(" + sLt + ", " + hiLt + ", " + eqLo + ", " + allM + ");");
        // Select: min = src < scalar ? src : scalar
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiDst);
        codegen.Emit("vsel(" + loDst + ", " + loS + ", " + loScalar + ", " + sLt + ");");
        codegen.Emit("vsel(" + hiDst + ", " + hiS + ", " + hiScalar + ", " + sLt + ");");
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmins(" + dst + ", " + src + ", " + scalarStr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Maxs — vmaxs (scalar maximum)
// ============================================================================

static std::string EmitVFMaxs(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, scalar, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.maxs requires 4 args (dst, src, scalar, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(srcDt) || srcDt == DataType::FP16 ||
                                                    srcDt == DataType::FP32 || srcDt == DataType::BF16))
        << "vf.maxs src only supports supported types, got " << DTypeStr(srcDt);
    DataType scalarDt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (scalarDt.IsInt() || scalarDt == DataType::FP16 ||
                                                    scalarDt == DataType::FP32 || scalarDt == DataType::BF16))
        << "vf.maxs scalar only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/FP32/BF16, got "
        << DTypeStr(scalarDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalarStr = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.maxs");
    scalarStr = cce::CoerceScalarToInt(op->args_[2], srcDt, scalarStr);
    if (srcDt.GetBit() == 64) {
        // B64 maxs: max(src, scalar) via deinterleave + scalar compare + vsel + interleave.
        std::string p = dst + "_maxs_";
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string castType = (srcDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string loScalar = p + "_losc_";
        std::string hiScalar = p + "_hisc_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loScalar);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiScalar);
        codegen.Emit("vdup(" + loScalar + ", (int32_t)(" + castType + "(" + scalarStr + ")), " + allM +
                     ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hiScalar + ", (int32_t)((" + castType + "(" + scalarStr + ")) >> 32), " + allM +
                     ", MODE_ZEROING);");
        // Compare: src > scalar → pick src, else pick scalar
        std::string hiGt = p + "_higt_";
        std::string hiEq = p + "_hieq_";
        std::string loGt = p + "_logt_";
        codegen.Emit("MaskReg " + hiGt + ";");
        codegen.Emit("MaskReg " + hiEq + ";");
        codegen.Emit("MaskReg " + loGt + ";");
        if (srcDt == DataType::INT64) {
            codegen.Emit("vcmp_gt(" + hiGt + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + hiScalar +
                         ", " + allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + hiScalar +
                         ", " + allM + ");");
        } else {
            codegen.Emit("vcmp_gt(" + hiGt + ", " + hiS + ", " + hiScalar + ", " + allM + ");");
            codegen.Emit("vcmp_eq(" + hiEq + ", " + hiS + ", " + hiScalar + ", " + allM + ");");
        }
        codegen.Emit("vcmp_gt(" + loGt + ", " + loS + ", " + loScalar + ", " + allM + ");");
        std::string eqLo = p + "_eqlo_";
        std::string sGt = p + "_sgt_";
        codegen.Emit("MaskReg " + eqLo + ";");
        codegen.Emit("MaskReg " + sGt + ";");
        codegen.Emit("pand(" + eqLo + ", " + hiEq + ", " + loGt + ", " + allM + ");");
        codegen.Emit("por(" + sGt + ", " + hiGt + ", " + eqLo + ", " + allM + ");");
        // Select: max = src > scalar ? src : scalar
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiDst);
        codegen.Emit("vsel(" + loDst + ", " + loS + ", " + loScalar + ", " + sGt + ");");
        codegen.Emit("vsel(" + hiDst + ", " + hiS + ", " + hiScalar + ", " + sGt + ");");
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vmaxs(" + dst + ", " + src + ", " + scalarStr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// LeakyRelu — vlrelu
// ============================================================================

static std::string EmitVFLeakyRelu(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, alpha, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.leaky_relu requires 4 args (dst, src, alpha, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.leaky_relu src only supports supported types, got " << DTypeStr(srcDt);
    DataType alphaDt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (alphaDt == DataType::FP16 || alphaDt == DataType::FP32))
        << "vf.leaky_relu scalar only supports FP16/FP32, got " << DTypeStr(alphaDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string alpha = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.leaky_relu");
    codegen.Emit("vlrelu(" + dst + ", " + src + ", " + alpha + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Interleave — vintlv
// ============================================================================

static std::string EmitVFInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst pair prepended by codegen): [dst0, dst1, src0, src1].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsSrcMaskReg(op, 2)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        // MaskInterleave: only supports b8/b16/b32 (not b64)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          dtype.GetBit() == 8 || dtype.GetBit() == 16 || dtype.GetBit() == 32)
            << "vf.interleave (MaskReg) only supports b8/b16/b32, got " << DTypeStr(dtype);
        std::string pintlvOp;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pintlvOp = "pintlv_b8";
        } else if (dtype.GetBit() == 16) {
            pintlvOp = "pintlv_b16";
        } else {
            pintlvOp = "pintlv_b32";
        }
        codegen.Emit(pintlvOp + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
    // vintlv requires src0/src1 to be b8/b16/b32/b64
    DataType src0Dt = GetExprDtype(op->args_[2]);
    DataType src1Dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(src0Dt) || src0Dt.GetBit() == 16 || src0Dt.GetBit() == 32 || src0Dt.GetBit() == 64))
        << "vf.interleave only supports b8/b16/b32/b64 types, got " << DTypeStr(src0Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (src0Dt == src1Dt))
        << "vf.interleave requires src0 and src1 to have the same type, got src0=" << DTypeStr(src0Dt)
        << " src1=" << DTypeStr(src1Dt);
    if (src0Dt.GetBit() == 64) {
        codegen.Emit("vintlv((RegTensor<uint32_t>&)" + dst0 + ", (RegTensor<uint32_t>&)" + dst1 +
                     ", (RegTensor<uint32_t>&)" + src0 + ", (RegTensor<uint32_t>&)" + src1 + ");");
    } else {
        codegen.Emit("vintlv(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
    }
    return "";
}

// ============================================================================
// PairReduceSum — vcpadd
// ============================================================================

static std::string EmitVFPairReduceSum(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.pair_reduce_sum requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.pair_reduce_sum src only supports FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.pair_reduce_sum");
    codegen.Emit("vcpadd(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// AbsSub — vabsdif (absolute difference: |src0 - src1|)
// ============================================================================

static std::string EmitVFAbsSub(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.abs_sub requires 4 args (dst, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s0Dt == DataType::FP16 || s0Dt == DataType::FP32 || s0Dt == DataType::INT64))
        << "vf.abs_sub src0 only supports FP16/FP32/INT64, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s1Dt == DataType::FP16 || s1Dt == DataType::FP32 || s1Dt == DataType::INT64))
        << "vf.abs_sub src1 only supports FP16/FP32/INT64, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.abs_sub requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.abs_sub");
    if (s0Dt == DataType::INT64) {
        // B64 abs_sub: |src0 - src1| = abs(vsub(src0, src1)).
        // vsub has no b64 single-register overload; use deinterleave + vsub/vsubc
        // borrow-chain (mirrors AscendC SubB64Impl for int64_t).
        std::string p = dst + "_abssub_";
        std::string diffM = p + "_diffm_";
        codegen.Emit("MaskReg " + diffM + " = pset_b32(PAT_ALL);");
        // Deinterleave both sources to b32 low/high halves
        EmitB64Deinterleave(codegen, p + "s0", src0);
        EmitB64Deinterleave(codegen, p + "s1", src1);
        std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
        std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
        // Sub with borrow: diff_lo = src0_lo - src1_lo, borrow = (src0_lo < src1_lo)
        std::string borrow = p + "_borrow_";
        std::string loD = p + "_dl_";
        std::string hiD = p + "_dh_";
        codegen.Emit("MaskReg " + borrow + ";");
        EmitRtWrapDecl(codegen, ir::DataType::INT32, loD);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, hiD);
        codegen.Emit("vsubc(" + borrow + ", " + loD + ", (RegTensor<int32_t>&)" + lo0 + ", (RegTensor<int32_t>&)" +
                     lo1 + ", " + diffM + ");");
        codegen.Emit("vsubcs(" + borrow + ", " + hiD + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" +
                     hi1 + ", " + borrow + ", " + diffM + ");");
        // Now diff = {lo_d, hi_d}. Compute abs via sign check + negate.
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        // Zero register (vbr equivalent)
        std::string zeroReg = p + "_zero_";
        EmitRtWrapDecl(codegen, ir::DataType::INT32, zeroReg);
        codegen.Emit("vdup(" + zeroReg + ", 0, " + allM + ", MODE_ZEROING);");
        // Sign mask: hi_d < 0 (signed)
        std::string signM = p + "_sign_";
        codegen.Emit("MaskReg " + signM + ";");
        codegen.Emit("vcmp_lt(" + signM + ", (RegTensor<int32_t>&)" + hiD + ", (RegTensor<int32_t>&)" + zeroReg + ", " +
                     allM + ");");
        // Negate via borrow-chain sub: neg = 0 - diff
        // vsubc: borrow = (0 < lo_d), neg_lo = 0 - lo_d
        // vsubcs: neg_hi = 0 - hi_d - borrow
        std::string borrow2 = p + "_borrow2_";
        std::string loN = p + "_nlo_";
        std::string hiN = p + "_nhi_";
        codegen.Emit("MaskReg " + borrow2 + ";");
        EmitRtWrapDecl(codegen, ir::DataType::INT32, loN);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, hiN);
        codegen.Emit("vsubc(" + borrow2 + ", " + loN + ", " + zeroReg + ", (RegTensor<int32_t>&)" + loD + ", " + signM +
                     ");");
        codegen.Emit("vsubcs(" + borrow2 + ", " + hiN + ", " + zeroReg + ", (RegTensor<int32_t>&)" + hiD + ", " +
                     borrow2 + ", " + signM + ");");
        // Select: if negative -> neg, else -> diff (per b32 half)
        // All vsel operands must be the same type (int32_t).
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::INT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, hiDst);
        codegen.Emit("vsel(" + loDst + ", " + loN + ", (RegTensor<int32_t>&)" + loD + ", " + signM + ");");
        codegen.Emit("vsel(" + hiDst + ", " + hiN + ", (RegTensor<int32_t>&)" + hiD + ", " + signM + ");");
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else {
        codegen.Emit("vabsdif(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Axpy — vaxpy (accumulate: dst = src * scalar + dst)
// ============================================================================

static std::string EmitVFAxpy(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, scalar, mask], or
    // [dst, old, src, scalar, mask] on a rebinding assignment (the previous dst
    // value is an explicit argument, copied into the fresh dst before the
    // accumulate instruction).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4 || op->args_.size() == 5)
        << op->name_ << " requires 4 args (dst, src, scalar, mask) or 5 (dst, old, src, scalar, mask)";
    size_t base = op->args_.size() == 5 ? 1 : 0;
    DataType srcDt = GetExprDtype(op->args_[base + 1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32 ||
                                                    srcDt == DataType::INT64 || srcDt == DataType::UINT64))
        << op->name_ << " src only supports FP16/FP32/INT64/UINT64, got " << DTypeStr(srcDt);
    // vaxpy supports half/float/uint64_t/int64_t
    DataType scalarDt = GetExprDtype(op->args_[base + 2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalarDt == DataType::FP16 || scalarDt == DataType::FP32 ||
                                                       scalarDt == DataType::UINT64 || scalarDt == DataType::INT64 ||
                                                       scalarDt == DataType::INDEX)
        << op->name_ << " scalar only supports FP16/FP32/UINT64/INT64, got " << DTypeStr(scalarDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[base + 1]);
    std::string scalarStr = codegen.GetExprAsCode(op->args_[base + 2]);
    std::string mask = codegen.GetExprAsCode(op->args_[base + 3]);
    std::string mode = VFZeroingOnly(op, op->name_);
    scalarStr = cce::CoerceScalarToInt(op->args_[base + 2], srcDt, scalarStr);
    if (srcDt == DataType::INT64 || srcDt == DataType::UINT64) {
        // B64 Axpy: dst = src * scalar + old → Muls (Duplicate + MulB64Impl) + Add (AddB64Impl).
        // Mirrors AscendC AxpyImpl for b64.
        std::string p = dst + "_axpy_";
        std::string castType = (srcDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        // AscendC MulB64Impl: vmull always uses uint32_t (unsigned 32x32→64);
        // vmula uses int32_t for int64_t (signed cross-term), uint32_t for uint64_t.
        std::string vmullCast = "(RegTensor<uint32_t>&)";
        std::string vmulaCast = (srcDt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        std::string b32Cast = (srcDt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        // 1. Pack b64 mask to b32 (mirrors MaskPack in CalTraitOneByTransToTraitTwo)
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        // 2. Broadcast scalar as b32 halves (mirrors DuplicateB64Impl)
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string loSc = p + "_losc_";
        std::string hiSc = p + "_hisc_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loSc);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiSc);
        codegen.Emit("vdup(" + loSc + ", (int32_t)(" + castType + "(" + scalarStr + ")), " + allM + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + hiSc + ", (int32_t)((" + castType + "(" + scalarStr + ")) >> 32), " + allM +
                     ", MODE_ZEROING);");
        // 3. Deinterleave src
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        // 4. MulB64: vmull + vmula (32×32→64 long multiply)
        std::string mulLo = p + "_mul_lo_";
        std::string mulHi = p + "_mul_hi_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, mulLo);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, mulHi);
        codegen.Emit("vmull(" + vmullCast + mulLo + ", " + vmullCast + mulHi + ", " + vmullCast + loS + ", " +
                     vmullCast + loSc + ", " + packedM + ");");
        codegen.Emit("vmula(" + vmulaCast + mulHi + ", " + vmulaCast + loS + ", " + vmulaCast + hiSc + ", " + packedM +
                     ", MODE_ZEROING);");
        codegen.Emit("vmula(" + vmulaCast + mulHi + ", " + vmulaCast + hiS + ", " + vmulaCast + loSc + ", " + packedM +
                     ", MODE_ZEROING);");
        // 5. Deinterleave old (for AddB64: dst = mul_result + old)
        std::string oldSrc = base == 1 ? codegen.GetExprAsCode(op->args_[1]) : dst;
        EmitB64Deinterleave(codegen, p + "d", oldSrc);
        std::string loD = p + "d_lo_", hiD = p + "d_hi_";
        // 6. AddB64: vaddc + vaddcs (carry-chain add, using packed b32 mask)
        std::string carry = p + "_carry_";
        std::string addLo = p + "_add_lo_";
        std::string addHi = p + "_add_hi_";
        codegen.Emit("MaskReg " + carry + ";");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, addLo);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, addHi);
        codegen.Emit("vaddc(" + carry + ", " + b32Cast + addLo + ", " + b32Cast + mulLo + ", " + b32Cast + loD + ", " +
                     packedM + ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32Cast + addHi + ", " + b32Cast + mulHi + ", " + b32Cast + hiD + ", " +
                     carry + ", " + packedM + ");");
        EmitB64Interleave(codegen, dst, addLo, addHi, p + "_ilv");
    } else {
        if (base == 1) {
            codegen.Emit(dst + " = " + codegen.GetExprAsCode(op->args_[1]) + ";");
        }
        codegen.Emit("vaxpy(" + dst + ", " + src + ", " + scalarStr + ", " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Madd — vmadd (multiply-accumulate: dst = src0 * src1 + dst)
// ============================================================================

static std::string EmitVFMulDstAdd(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask], or
    // [dst, old, src0, src1, mask] on a rebinding assignment (the previous dst
    // value is an explicit argument, copied into the fresh dst before the
    // accumulate instruction).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4 || op->args_.size() == 5)
        << op->name_ << " requires 4 args (dst, src0, src1, mask) or 5 (dst, old, src0, src1, mask)";
    size_t base = op->args_.size() == 5 ? 1 : 0;
    DataType s0Dt = GetExprDtype(op->args_[base + 1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s0Dt == DataType::FP16 || s0Dt == DataType::FP32 || s0Dt == DataType::BF16))
        << op->name_ << " src0 only supports FP16/FP32/BF16, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[base + 2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (s1Dt == DataType::FP16 || s1Dt == DataType::FP32 || s1Dt == DataType::BF16))
        << op->name_ << " src1 only supports FP16/FP32/BF16, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << op->name_ << " requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    // vmadd supports u16/i16/u32/i32/half/float/bf/i64/u64
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[base + 1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[base + 2]);
    std::string mask = codegen.GetExprAsCode(op->args_[base + 3]);
    if (base == 1) {
        codegen.Emit(dst + " = " + codegen.GetExprAsCode(op->args_[1]) + ";");
    }
    std::string mode = VFZeroingOnly(op, op->name_);
    codegen.Emit("vmadd(" + dst + ", " + src0 + ", " + src1 + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// Pack — vpack (narrow data type)
// ============================================================================

static std::string EmitVFPack(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src]. The dst kind
    // mirrors the source (MaskReg src keeps the mask dialect).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2) << "vf.pack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        int partVal = op->GetKwarg<int>("part");
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, partVal == static_cast<int>(ir::PackPart::LOWER) ||
                                                               partVal == static_cast<int>(ir::PackPart::UPPER))
            << "vf.pack part must be PackPart::LOWER or PackPart::UPPER, got " << partVal;
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(partVal)));
    }
    if (IsSrcMaskReg(op, 1)) {
        std::string cceHalf = (part == "LOWER") ? "LOWER" : "HIGHER";
        codegen.Emit("ppack(" + dst + ", " + src + ", " + cceHalf + ");");
        return "";
    }

    DataType srcDt = GetExprDtype(op->args_[1]);
    DataType dstDt = DataType::UINT32;
    if (op->HasKwarg("dtype")) {
        dstDt = op->GetKwarg<DataType>("dtype");
    } else if (const auto& t = op->args_[1]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    // Validate (dst, src) as a pair (mirrors AscendC PackImpl SupportType
    // tuples): dst is always unsigned and half the src width; src may be
    // signed or unsigned.
    //   (UINT8, INT16|UINT16), (UINT16, INT32|UINT32), (UINT32, INT64|UINT64)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (dstDt == DataType::UINT8 && (srcDt == DataType::INT16 || srcDt == DataType::UINT16)) ||
                          (dstDt == DataType::UINT16 && (srcDt == DataType::INT32 || srcDt == DataType::UINT32)) ||
                          (dstDt == DataType::UINT32 && (srcDt == DataType::INT64 || srcDt == DataType::UINT64)))
        << "vf.pack supports UINT8<-INT16/UINT16, UINT16<-INT32/UINT32, UINT32<-INT64/UINT64 pairs, got dst="
        << DTypeStr(dstDt) << " src=" << DTypeStr(srcDt);

    if (srcDt.GetBit() == 64) {
        // 64-bit source → 32-bit dst: use DeInterleave with a zero register,
        // mirroring PackImpl<..., part>(dst, src) for 8-byte src.
        std::string partCheck = (part == "LOWER") ? "LOWEST" : "HIGHEST";
        std::string zeroVar = dst + "_pack_zero_";
        std::string dumpVar = dst + "_pack_dump_";
        std::string maskVar = dst + "_pack_mask_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, zeroVar);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, dumpVar);
        codegen.Emit("MaskReg " + maskVar + " = pset_b32(PAT_ALL);");
        codegen.Emit("vdup(" + zeroVar + ", 0, " + maskVar + ", MODE_ZEROING);");
        if (partCheck == "LOWEST") {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dumpVar + ", " + "(RegTensor<uint32_t>&)" +
                         src + ", " + zeroVar + ");");
        } else {
            codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst + ", " + dumpVar + ", " + zeroVar +
                         ", (RegTensor<uint32_t>&)" + src + ");");
        }
    } else {
        // MODE_UNKNOWN mirrors AscendC: the C API asc_pack_impl (npu_arch_3510)
        // passes MODE_UNKNOWN and Reg::PackImpl omits the mode (intrinsic
        // default), leaving the half not selected by part unspecified.
        std::string ccePart = (part == "LOWER") ? "LOWER" : "HIGHER";
        codegen.Emit("vpack(" + dst + ", " + src + ", " + ccePart + ", MODE_UNKNOWN);");
    }
    return "";
}

// ============================================================================
// Unpack — vunpack (widen data type)
// ============================================================================

static std::string EmitVFUnpack(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src]. The dst kind
    // mirrors the source (MaskReg src keeps the mask dialect).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2) << "vf.unpack requires 2 args (dst, src)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string part = "LOWER";
    if (op->HasKwarg("part")) {
        int partVal = op->GetKwarg<int>("part");
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, partVal == static_cast<int>(ir::PackPart::LOWER) ||
                                                               partVal == static_cast<int>(ir::PackPart::UPPER))
            << "vf.unpack part must be PackPart::LOWER or PackPart::UPPER, got " << partVal;
        part = VFEnumValueName(ir::EnumToString(static_cast<ir::PackPart>(partVal)));
    }

    if (IsSrcMaskReg(op, 1)) {
        std::string cceHalf = (part == "LOWER") ? "LOWER" : "HIGHER";
        codegen.Emit("punpack(" + dst + ", " + src + ", " + cceHalf + ");");
        return "";
    }

    DataType srcDt = GetExprDtype(op->args_[1]);
    DataType dstDt = DataType::UINT32;
    if (op->HasKwarg("dtype")) {
        dstDt = op->GetKwarg<DataType>("dtype");
    } else if (const auto& t = op->args_[1]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    // Validate (dst, src) as a pair (mirrors AscendC UnPackImpl SupportType
    // tuples): dst is double the src width with matching signedness. FP and
    // other types are rejected here instead of failing late in bisheng
    // compilation.
    //   (INT16, INT8), (UINT16, UINT8), (INT32, INT16),
    //   (UINT32, UINT16), (INT64, INT32), (UINT64, UINT32)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (dstDt == DataType::INT16 && srcDt == DataType::INT8) ||
                                                       (dstDt == DataType::UINT16 && srcDt == DataType::UINT8) ||
                                                       (dstDt == DataType::INT32 && srcDt == DataType::INT16) ||
                                                       (dstDt == DataType::UINT32 && srcDt == DataType::UINT16) ||
                                                       (dstDt == DataType::INT64 && srcDt == DataType::INT32) ||
                                                       (dstDt == DataType::UINT64 && srcDt == DataType::UINT32))
        << "vf.unpack supports INT16<-INT8, UINT16<-UINT8, INT32<-INT16, UINT32<-UINT16, INT64<-INT32, "
           "UINT64<-UINT32 pairs, got dst="
        << DTypeStr(dstDt) << " src=" << DTypeStr(srcDt);

    if (dstDt.GetBit() == 64) {
        std::string srcCtype = srcDt.ToCTypeString();
        std::string partCheck = (part == "LOWER") ? "LOWEST" : "HIGHEST";
        std::string padVar = dst + "_unpack_pad_";
        std::string dumpVar = dst + "_unpack_dump_";
        std::string maskVar = dst + "_unpack_mask_";
        codegen.Emit("RtWrap<" + srcCtype + "> " + padVar + ";");
        codegen.Emit("RtWrap<" + srcCtype + "> " + dumpVar + ";");
        codegen.Emit("MaskReg " + maskVar + " = pset_b32(PAT_ALL);");
        if (srcDt == DataType::INT32) {
            codegen.Emit("vshrs(" + padVar + ", " + src + ", 31, " + maskVar + ", MODE_ZEROING);");
        } else {
            codegen.Emit("vdup(" + padVar + ", 0, " + maskVar + ", MODE_ZEROING);");
        }
        if (partCheck == "LOWEST") {
            codegen.Emit("vintlv((RegTensor<" + srcCtype + ">&)" + dst + ", " + dumpVar + ", " + "(RegTensor<" +
                         srcCtype + ">&)" + src + ", " + padVar + ");");
        } else {
            codegen.Emit("vintlv(" + dumpVar + ", (RegTensor<" + srcCtype + ">&)" + dst + ", " + "(RegTensor<" +
                         srcCtype + ">&)" + src + ", " + padVar + ");");
        }
    } else {
        std::string ccePart = (part == "LOWER") ? "LOWER" : "HIGHER";
        codegen.Emit("vunpack(" + dst + ", " + src + ", " + ccePart + ");");
    }
    return "";
}

// ============================================================================
// PRelu — vprelu (parametric ReLU with per-element slope)
// ============================================================================

static std::string EmitVFPRelu(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, slope, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.prelu requires 4 args (dst, src, slope, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.prelu src only supports supported types, got " << DTypeStr(srcDt);
    DataType slopeDt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt == slopeDt)
        << "vf.prelu requires src and slope to have the same type, got src=" << DTypeStr(srcDt)
        << " slope=" << DTypeStr(slopeDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string slope = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, "vf.prelu");
    codegen.Emit("vprelu(" + dst + ", " + src + ", " + slope + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// ShiftLeft — unified left shift: vshl (per-lane, shift is a RegTensor) or
// vshls (uniform, shift is a scalar). The former standalone vf.shift_lefts
// (scalar) op is merged in here; the register-vs-scalar decision is made from
// the arg's IR type (RegType(RegTensor) = vector register, anything else = scalar).
// ============================================================================

// Returns true when the shift-amount arg (op->args_[2]) is a per-lane vector
// register; false for a uniform scalar shift (integer literal or plain scalar).
static bool ShiftAmountIsRegister(const ir::CallPtr& op)
{
    // Integer literals are always scalar shifts.
    if (ir::As<ir::ConstInt>(op->args_[2]))
        return false;
    // A RegType(RegTensor) arg is a vector register shift.
    return ir::IsRegTensor(op->args_[2]->GetType());
}

static std::string EmitVFShift(const ir::CallPtr& op, codegen::CodegenBase& codegenBase, const std::string& opName,
                               const std::string& vectorInstruction, const std::string& scalarInstruction)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, shift, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << opName << " requires 4 args (dst, src, shift, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt == DataType::INT8 || srcDt == DataType::UINT8 ||
                                                       srcDt == DataType::INT16 || srcDt == DataType::UINT16 ||
                                                       srcDt == DataType::INT32 || srcDt == DataType::UINT32 ||
                                                       srcDt == DataType::INT64 || srcDt == DataType::UINT64)
        << opName << " src only supports integer types, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string shift = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    std::string mode = VFZeroingOnly(op, opName);
    if (srcDt.GetBit() == 64) {
        // B64 shift via deinterleave + b32 shift + cross-half carry + interleave.
        // Scalar shift only (shift amount is a compile-time or runtime scalar).
        std::string p = dst + "_shift_";
        EmitB64Deinterleave(codegen, p + "s", src);
        std::string loS = p + "s_lo_", hiS = p + "s_hi_";
        std::string allM = p + "_allm_";
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiDst);
        if (opName == "vf.shift_left") {
            // Left shift: lo_dst = lo << N, carry = lo >> (32-N), hi_dst = (hi << N) | carry
            std::string carry = p + "_carry_";
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, carry);
            codegen.Emit("vshls(" + loDst + ", " + loS + ", (int16_t)(" + shift + "), " + allM + ", MODE_ZEROING);");
            codegen.Emit("vshrs(" + carry + ", " + loS + ", (int16_t)(32 - (" + shift + ")), " + allM +
                         ", MODE_ZEROING);");
            codegen.Emit("vshls(" + hiDst + ", " + hiS + ", (int16_t)(" + shift + "), " + allM + ", MODE_ZEROING);");
            codegen.Emit("vor(" + hiDst + ", " + hiDst + ", " + carry + ", " + allM + ", MODE_ZEROING);");
        } else {
            // Right shift: for INT64 (arithmetic), for UINT64 (logical)
            std::string carry = p + "_carry_";
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, carry);
            if (srcDt == DataType::INT64) {
                // Arithmetic right shift: sign-extend hi. The vshrs overload
                // requires dst and src to have the same signedness, so the hi
                // half is shifted in the int32 domain on both sides (mirrors
                // AscendC ShiftR<int32_t>).
                codegen.Emit("vshrs((RegTensor<int32_t>&)" + hiDst + ", (RegTensor<int32_t>&)" + hiS + ", (int16_t)(" +
                             shift + "), " + allM + ", MODE_ZEROING);");
            } else {
                codegen.Emit("vshrs(" + hiDst + ", " + hiS + ", (int16_t)(" + shift + "), " + allM +
                             ", MODE_ZEROING);");
            }
            codegen.Emit("vshls(" + carry + ", " + hiS + ", (int16_t)(32 - (" + shift + ")), " + allM +
                         ", MODE_ZEROING);");
            codegen.Emit("vshrs(" + loDst + ", " + loS + ", (int16_t)(" + shift + "), " + allM + ", MODE_ZEROING);");
            codegen.Emit("vor(" + loDst + ", " + loDst + ", " + carry + ", " + allM + ", MODE_ZEROING);");
        }
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        EmitB64Zeroing(codegen, dst, dst, mask, p + "_zero");
    } else if (ShiftAmountIsRegister(op)) {
        DataType shiftDt = GetExprDtype(op->args_[2]);
        DataType shiftNeed = DataType::INT8;
        DataType shiftUnsigned = DataType::UINT8;
        if (srcDt.GetBit() == 16) {
            shiftNeed = DataType::INT16;
            shiftUnsigned = DataType::UINT16;
        } else if (srcDt.GetBit() == 32) {
            shiftNeed = DataType::INT32;
            shiftUnsigned = DataType::UINT32;
        }
        // A5's __VF_BINARY_OP overloads take the signed shift operand; A6 (920r1)
        // defines vshl/vshr in the __VF_SHFT_OP registry, which requires the unsigned
        // type of the data width (u8/u16/u32). Accept either same-width form here and
        // reinterpret for the A6 call below.
        const bool isA6 = codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_9201;
        PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, shiftDt == shiftNeed || (isA6 && shiftDt == shiftUnsigned))
            << opName << " (vector) shift register must be the " << (isA6 ? "signed or unsigned " : "signed ")
            << "type matching the " << DTypeStr(srcDt) << " data width, expected " << DTypeStr(shiftNeed)
            << (isA6 ? " or " + DTypeStr(shiftUnsigned) : "") << ", got " << DTypeStr(shiftDt);
        std::string shiftOperand = shift;
        if (isA6 && shiftDt != shiftUnsigned) {
            shiftOperand = "(RegTensor<" + shiftUnsigned.ToCTypeString() + ">&)" + shift;
        }
        codegen.Emit(vectorInstruction + "(" + dst + ", " + src + ", " + shiftOperand + ", " + mask + ", " + mode +
                     ");");
    } else {
        codegen.Emit(scalarInstruction + "(" + dst + ", " + src + ", (int16_t)(" + shift + "), " + mask + ", " + mode +
                     ");");
    }
    return "";
}

static std::string EmitVFShiftLeft(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return EmitVFShift(op, codegenBase, "vf.shift_left", "vshl", "vshls");
}

// ============================================================================
// ShiftRight — unified right shift: vshr (per-lane, shift is a RegTensor) or
// vshrs (uniform, shift is a scalar). The former standalone vf.shift_rights
// (scalar) op is merged in here; dispatch uses ShiftAmountIsRegister (the
// shift arg's IR type), not the op name.
// ============================================================================

static std::string EmitVFShiftRight(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return EmitVFShift(op, codegenBase, "vf.shift_right", "vshr", "vshrs");
}

// ============================================================================
// Mull — vmull (long multiply: 32x32->64, lo/hi split)
// ============================================================================

static std::string EmitVFMull(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst pair prepended by codegen): [dst_lo, dst_hi, src0, src1, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 5)
        << "vf.mull requires 5 args (dst_lo, dst_hi, src0, src1, mask)";
    DataType s0Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s0Dt == DataType::INT32 || s0Dt == DataType::UINT32))
        << "vf.mull src only supports supported types, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (s1Dt == DataType::INT32 || s1Dt == DataType::UINT32))
        << "vf.mull src only supports supported types, got " << DTypeStr(s1Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.mull requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string dstLo = codegen.GetExprAsCode(op->args_[0]);
    std::string dstHi = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string mask = codegen.GetExprAsCode(op->args_[4]);
    codegen.Emit("vmull(" + dstLo + ", " + dstHi + ", " + src0 + ", " + src1 + ", " + mask + ");");
    return "";
}

// ============================================================================
// Addc — vaddcs (add with carry)
// ============================================================================

static std::string EmitVFAddc(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst pair prepended by codegen): [carry, sum, src0, src1, carry_in, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 6)
        << "vf.addc requires 6 args (carry, sum, src0, src1, carry_in, mask)";
    DataType s0Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == DataType::INT32 || s0Dt == DataType::UINT32)
        << "vf.addc src0 only supports INT32/UINT32, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.addc requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string carryOut = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string carryIn = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vaddcs(" + carryOut + ", " + dst + ", " + src0 + ", " + src1 + ", " + carryIn + ", " + mask + ");");
    return "";
}

// ============================================================================
// Subc — vsubcs (subtract with borrow)
// ============================================================================

static std::string EmitVFSubc(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst pair prepended by codegen): [borrow, diff, src0, src1, borrow_in, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 6)
        << "vf.subc requires 6 args (borrow, diff, src0, src1, borrow_in, mask)";
    DataType s0Dt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == DataType::INT32 || s0Dt == DataType::UINT32)
        << "vf.subc src0 only supports INT32/UINT32, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.subc requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    std::string borrowOut = codegen.GetExprAsCode(op->args_[0]);
    std::string dst = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    std::string borrowIn = codegen.GetExprAsCode(op->args_[4]);
    std::string mask = codegen.GetExprAsCode(op->args_[5]);
    codegen.Emit("vsubcs(" + borrowOut + ", " + dst + ", " + src0 + ", " + src1 + ", " + borrowIn + ", " + mask + ");");
    return "";
}

// ============================================================================
// ExpSub — vexpdif
// ============================================================================

// Helpers for multi-step int→int conversion via float intermediate.
// Available int→float vcvt overloads (bisheng intrinsics):
//   b8→f16 (widening), s16→f16 (same-width), s16→f32 (widening), s32→f32 (same-width)
// NOT available: u16→f16, u16→f32, u32→f32, s32→f16

// Emit int→float vcvt step. Returns the float dtype used (FP16 or FP32).
// For unavailable src→f16 paths (u16, u32, s32), routes through f32 instead.
static DataType EmitIntToFloatVcvt(codegen::CCECodegen& codegen, const std::string& dstReg, const std::string& srcReg,
                                   const std::string& mask, const std::string& part, const std::string& round,
                                   const std::string& mode, DataType srcDt)
{
    if (srcDt.GetBit() == 8) {
        // b8→f16: widening → vcvt(dst, src, mask, PART, MODE) — 5 args
        codegen.Emit("vcvt(" + dstReg + ", " + srcReg + ", " + mask + ", " + part + ", " + mode + ");");
        return DataType::FP16;
    }
    if (srcDt == DataType::INT16) {
        // s16→f16: same-width → vcvt(dst, src, mask, ROUND, MODE) — 5 args
        codegen.Emit("vcvt(" + dstReg + ", " + srcReg + ", " + mask + ", " + round + ", " + mode + ");");
        return DataType::FP16;
    }
    // b32 (s32/u32): s32→f32 same-width. u32 needs reinterpret as s32 first.
    std::string effectiveSrc = srcReg;
    if (srcDt == DataType::UINT32)
        effectiveSrc = "(RegTensor<int32_t> &)" + srcReg;
    // s32→f32: same-width → vcvt(dst, src, mask, ROUND, MODE) — 5 args
    codegen.Emit("vcvt(" + dstReg + ", " + effectiveSrc + ", " + mask + ", " + round + ", " + mode + ");");
    return DataType::FP32;
}

// Emit float→int vcvt step. float_dt is always FP16 (from EmitIntToFloatVcvt).
static void EmitFloatToIntVcvt(codegen::CCECodegen& codegen, const std::string& dstReg, const std::string& srcReg,
                               const std::string& mask, const std::string& part, const std::string& round,
                               const std::string& sat, const std::string& mode, DataType dstDt)
{
    if (dstDt.GetBit() == 8) {
        // float→b8: narrower-int → vcvt(dst, src, mask, ROUND, SAT, PART, MODE) — 7 args
        codegen.Emit("vcvt(" + dstReg + ", " + srcReg + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     mode + ");");
    } else if (dstDt.GetBit() == 16) {
        // f16→b16: same-width → vcvt(dst, src, mask, ROUND, RS, MODE) — 5 args
        codegen.Emit("vcvt(" + dstReg + ", " + srcReg + ", " + mask + ", " + round + ", " + sat + ", " + mode + ");");
    } else {
        // f16→b32: wider-int → vcvt(dst, src, mask, ROUND, PART, MODE) — 6 args
        codegen.Emit("vcvt(" + dstReg + ", " + srcReg + ", " + mask + ", " + round + ", " + part + ", " + mode + ");");
    }
}

static std::string EmitVFExpSub(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, max, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.exp_sub requires 4 args (dst, src, max, mask)";
    // vexpdiff: dst must be FP32, src can be FP32 or FP16
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt == DataType::FP32 || srcDt == DataType::FP16)
        << "vf.exp_sub only supports FP32/FP16 src, got " << DTypeStr(srcDt);
    DataType dstDt = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") : GetExprDtype(op->args_[0], srcDt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dstDt == DataType::FP32)
        << "vf.exp_sub result only supports DT_FP32 (pass dtype=DT_FP32 for FP16 src), got " << DTypeStr(dstDt);
    DataType maxDt = GetExprDtype(op->args_[2]);
    // vexpdiff: src0 and src1 must be the same type (both float or both half)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt == maxDt)
        << "vf.exp_sub requires src and max to have the same type, got src=" << DTypeStr(srcDt)
        << " max=" << DTypeStr(maxDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string maxReg = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    // vexpdiff: only supports RegLayout ZERO/ONE
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.exp_sub only supports layout ZERO/ONE, got " << layout;
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vexpdif(" + dst + ", " + src + ", " + maxReg + ", " + mask + ", " + part + ");");
    return "";
}

// ============================================================================
// Cast — vcvt
// ============================================================================

static std::string EmitVFCast(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask], its dtype
    // carried by the mandatory dtype kwarg.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.astype requires 3 args (dst, src, mask)";
    DataType srcDtype = GetExprDtype(op->args_[1], DataType::FP32);
    // vf.astype is a type-changing op: the frontend deduce requires the dtype
    // kwarg (and rejects astype without it), so the destination dtype is
    // carried here — there is no backend-side inference fallback.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->HasKwarg("dtype"))
        << "vf.astype requires the dtype kwarg (the frontend deduce enforces it)";
    DataType dstDtype = op->GetKwarg<DataType>("dtype");
    // vcvt: src and dst must have different types
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDtype != dstDtype)
        << "vf.astype: src and dst must have different types (both are " << DTypeStr(srcDtype) << ")";
    // vcvt has no uint64 overloads on this device, and the AscendC Cast micro
    // instruction it mirrors doesn't support uint64 either — reject early
    // instead of failing late in bisheng.
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                      srcDtype != DataType::UINT64 && dstDtype != DataType::UINT64)
        << "vf.astype does not support DT_UINT64 (no vcvt overload), got src=" << DTypeStr(srcDtype)
        << " dst=" << DTypeStr(dstDtype);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // Get layout and round_mode with defaults
    std::string layout = "ZERO";
    if (op->HasKwarg("layout")) {
        layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        // vcvt: layout must be ZERO/ONE/TWO/THREE
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype only supports layout ZERO/ONE/TWO/THREE, got " << layout;
    }
    std::string roundMode = "CAST_RINT";
    if (op->HasKwarg("round_mode")) {
        roundMode = VFEnumValueName(ir::EnumToString(static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"))));
    }
    // A5 vcvt only supports MODE_ZEROING at the instruction level.
    std::string modeValue = VFZeroingOnly(op, "vf.astype");
    std::string part;
    if (layout == "ZERO")
        part = "PART_EVEN";
    else if (layout == "ONE")
        part = "PART_ODD";
    else if (layout == "TWO")
        part = "PART_TWO";
    else if (layout == "THREE")
        part = "PART_THREE";
    else
        part = "PART_EVEN";
    std::string partPp;
    if (layout == "ZERO")
        partPp = "PART_P0";
    else if (layout == "ONE")
        partPp = "PART_P1";
    else if (layout == "TWO")
        partPp = "PART_P2";
    else if (layout == "THREE")
        partPp = "PART_P3";
    else
        partPp = "PART_P0";
    // Map round_mode to ::ROUND constants (dav_3510 GetRound mapping):
    //   CAST_RINT  → ROUND_R (round to nearest even)
    //   CAST_ROUND → ROUND_A (round to nearest, away from zero)
    //   CAST_FLOOR → ROUND_F
    //   CAST_CEIL  → ROUND_C
    //   CAST_TRUNC → ROUND_Z
    //   CAST_ODD   → ROUND_O (Von Neumann rounding)
    //   CAST_HYBRID→ ROUND_H (3510 only, not supported by all vcvt overloads)
    //   unknown    → ROUND_R (safe fallback, universally supported)
    std::string round;
    if (roundMode == "CAST_RINT")
        round = "ROUND_R";
    else if (roundMode == "CAST_ROUND")
        round = "ROUND_A";
    else if (roundMode == "CAST_FLOOR")
        round = "ROUND_F";
    else if (roundMode == "CAST_CEIL")
        round = "ROUND_C";
    else if (roundMode == "CAST_TRUNC")
        round = "ROUND_Z";
    else if (roundMode == "CAST_ODD")
        round = "ROUND_O"; // only supported by widening float→float vcvt overloads
    else if (roundMode == "CAST_HYBRID")
        round = "ROUND_H"; // only supported by widening float→float vcvt overloads
    else
        round = "ROUND_R"; // unknown → ROUND_R (safe fallback)
    // Get saturation kwarg (default: disabled)
    std::string sat = "RS_DISABLE";
    if (op->HasKwarg("saturate")) {
        std::string satVal = VFEnumValueName(
            ir::EnumToString(static_cast<ir::SaturateMode>(op->GetKwarg<int>("saturate"))));
        if (satVal == "ON" || satVal == "ENABLE")
            sat = "RS_ENABLE";
    }
    // Determine if narrowing or widening conversion based on src/dst dtype.
    // Widening: dst wider than src — no ROUND/RS needed, just PART + MODE
    // Narrowing: dst narrower — needs ROUND + RS + PART + MODE
    // Float→same-width-int (FP32→S32): ROUND + RS + MODE (no PART)
    // INT→FLOAT same-width (S32→FP32): ROUND + MODE (no RS, no PART)
    // FLOAT→FLOAT same-width (FP16→BF16): ROUND + MODE (no RS, no PART)
    // INT→INT narrowing: RS + PART + MODE (no ROUND)
    // Cross-width INT↔FLOAT (S16→FP32, S32→FP16, FP32→S64, S64→FP32): ROUND + PART + MODE
    bool isWidening = false;
    bool isIntToFloat = false;
    bool isFloatToSameInt = false;     // FP32→S32, FP16→S16 (same-width float→int)
    bool isFloatToNarrowerInt = false; // FP16→INT8/UINT8, BF16→INT8/UINT8 (2x narrowing float→int)
    bool isFloatToWiderInt = false;    // FP16→S32, BF16→S32 (widening float→int)
    bool isInt2xNarrowing = false;     // 2x narrowing with RS: b32→b16, b16→u8, b64→b32
    bool isInt4xNarrowing = false;     // 4x narrowing with RS: b32→u8
    bool isInt4xWidening = false;      // 4x int widening: b8→b32 (PART_P0/P1/P2/P3 only)
    bool isCrossWidth = false;         // cross-width INT↔FLOAT with ROUND + PART + MODE
    bool isS4Widening = false;         // INT4→FP16/BF16/INT16 (vcvt_s42f16/bf16/s16)
    bool isS4Narrowing = false;        // FP16→INT4 (vcvt_f162s4)
    bool isS16ToS4 = false;            // INT16→INT4 (two-step: s16→f16→s4, mirroring AscendC Cast)
    bool isIntIntTwoStep = false;      // int→int via f16 intermediate (cross-sign widening, int→INT8 narrowing)
    // FP8/FP4 low-precision conversions
    bool isFpWidenPp = false;        // 4x widening PP: FP8→FP32, FP4→BF16 → vcvt(dst,src,mask,PART_PP,MODE)
    bool isFpNarrowRndSatPp = false; // 4x narrowing RND_SAT_PP: FP32→FP8 → vcvt(dst,src,mask,ROUND,SAT)
    bool isFpNarrowRndPp = false;    // 4x narrowing RND_PP: BF16→FP4 → vcvt(dst,src,mask,ROUND,PART_PP,MODE)
    // S4 (INT4) special instructions: vcvt_s42f16, vcvt_s42bf16, vcvt_s42s16, vcvt_f162s4
    if (srcDtype == DataType::INT4 &&
        (dstDtype == DataType::FP16 || dstDtype == DataType::BF16 || dstDtype == DataType::INT16)) {
        isS4Widening = true;
    } else if (dstDtype == DataType::INT4 && srcDtype == DataType::FP16) {
        isS4Narrowing = true;
    } else if (dstDtype == DataType::INT4 && srcDtype == DataType::INT16) {
        isS16ToS4 = true;
    }
    // Widening cases: vcvt(dst, src, mask, PART, MODE) — no ROUND/RS
    // FP16→FP32, BF16→FP32, INT16→INT32, UINT16→UINT32, INT8→INT16, UINT8→UINT16,
    // INT8/UINT8→FP16, S32→S64, INT16/UINT16→FP32 (__VF_VCVTIF_PART)
    // Cross-sign s→u 2x widening: INT16→UINT32 (AscendC layoutMerge partCondition supports this)
    // NOTE: Cross-sign u→s widening (u8→s16, u16→s32) and s8→u16 are NOT supported by hardware.
    // NOTE: u32→s64 is NOT supported (only s32→s64 is).
    else if ((srcDtype == DataType::FP16 && dstDtype == DataType::FP32) ||
             (srcDtype == DataType::BF16 && dstDtype == DataType::FP32) ||
             (srcDtype == DataType::UINT16 && dstDtype == DataType::UINT32) ||
             (srcDtype == DataType::INT16 && dstDtype == DataType::INT32) ||
             (srcDtype == DataType::INT16 && dstDtype == DataType::UINT32) ||
             (srcDtype == DataType::INT8 && dstDtype == DataType::INT16) ||
             (srcDtype == DataType::UINT8 && dstDtype == DataType::UINT16) ||
             (srcDtype == DataType::INT8 && dstDtype == DataType::FP16) ||
             (srcDtype == DataType::UINT8 && dstDtype == DataType::FP16) ||
             (srcDtype == DataType::INT32 && dstDtype == DataType::INT64) ||
             (srcDtype == DataType::INT16 && dstDtype == DataType::FP32) ||
             (srcDtype == DataType::UINT16 && dstDtype == DataType::FP32) ||
             (srcDtype == DataType::HF8 && dstDtype == DataType::FP16)) {
        isWidening = true;
    }
    // INT→FLOAT same-width or FLOAT→FLOAT same-width: vcvt(dst, src, mask, ROUND, MODE_ZEROING)
    // S32/U32→FP32, S16/U16→FP16, FP16→BF16, BF16→FP16
    else if (((srcDtype == DataType::INT32 || srcDtype == DataType::UINT32) && dstDtype == DataType::FP32) ||
             ((srcDtype == DataType::INT16 || srcDtype == DataType::UINT16) && dstDtype == DataType::FP16) ||
             (srcDtype == DataType::FP16 && dstDtype == DataType::BF16) ||
             (srcDtype == DataType::BF16 && dstDtype == DataType::FP16)) {
        isIntToFloat = true;
    }
    // FLOAT→same-width-INT: vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
    // FP32→S32/U32, FP16→S16/U16 (same element width)
    else if ((srcDtype == DataType::FP32 && (dstDtype == DataType::INT32 || dstDtype == DataType::UINT32)) ||
             (srcDtype == DataType::FP16 && (dstDtype == DataType::INT16 || dstDtype == DataType::UINT16))) {
        isFloatToSameInt = true;
    }
    // FLOAT→narrower-INT (2x narrowing): vcvt(dst, src, mask, ROUND, RS, PART, MODE_ZEROING)
    // FP16→INT8/UINT8, BF16→INT8/UINT8 (b16→b8, uses __VF_VCVTFI_SAT_PART with PART_EVEN/ODD)
    else if ((srcDtype == DataType::FP16 || srcDtype == DataType::BF16) &&
             (dstDtype == DataType::INT8 || dstDtype == DataType::UINT8)) {
        isFloatToNarrowerInt = true;
    }
    // FLOAT→wider-INT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
    // FP16→S32/U32, BF16→S32/U32 (half-width float to full-width int)
    else if ((srcDtype == DataType::FP16 && (dstDtype == DataType::INT32 || dstDtype == DataType::UINT32)) ||
             (srcDtype == DataType::BF16 && (dstDtype == DataType::INT32 || dstDtype == DataType::UINT32))) {
        isFloatToWiderInt = true;
    }
    // Cross-width INT↔FLOAT: vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING)
    // S32/U32→FP16, FP32→S64, S64→FP32
    // (S16/U16→FP32 moved to is_widening — uses __VF_VCVTIF_PART with PART+MODE only)
    else if (((srcDtype == DataType::INT32 || srcDtype == DataType::UINT32) && dstDtype == DataType::FP16) ||
             (srcDtype == DataType::FP32 && dstDtype == DataType::INT64) ||
             (srcDtype == DataType::INT64 && dstDtype == DataType::FP32)) {
        isCrossWidth = true;
    }
    // INT→INT 4x narrowing to UINT8 (b32→u8): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
    // __VF_VCVTII_SAT_PP: uses PART_P0/P1/P2/P3, supports RS parameter
    else if ((srcDtype == DataType::INT32 || srcDtype == DataType::UINT32) && dstDtype == DataType::UINT8) {
        isInt4xNarrowing = true;
    }
    // INT→INT 2x narrowing to UINT8/INT16/UINT16 (b16→u8, b32→b16, b64→b32): vcvt(dst, src, mask, RS, PART, MODE) — 6
    // args
    // __VF_VCVTII_SAT_PART: uses PART_EVEN/ODD, supports RS parameter
    else if (((srcDtype == DataType::INT32 || srcDtype == DataType::UINT32) &&
              (dstDtype == DataType::INT16 || dstDtype == DataType::UINT16)) ||
             ((srcDtype == DataType::INT16 || srcDtype == DataType::UINT16) && dstDtype == DataType::UINT8) ||
             (srcDtype == DataType::INT64 && (dstDtype == DataType::INT32 || dstDtype == DataType::UINT32))) {
        isInt2xNarrowing = true;
    }
    // INT→INT 4x widening (b8→b32): vcvt(dst, src, mask, PART_PP, MODE_ZEROING)
    // __VF_VCVTII_PP: uses PART_P0/P1/P2/P3
    // NOTE: Cross-sign 4x widening (s8→u32, u8→s32) is NOT supported by hardware.
    else if ((srcDtype == DataType::INT8 && dstDtype == DataType::INT32) ||
             (srcDtype == DataType::UINT8 && dstDtype == DataType::UINT32)) {
        isInt4xWidening = true;
    }
    // FP8/FP4 4x widening (PP mode): FP8→FP32, FP4→BF16
    // __VF_VCVTFF_PP: vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
    else if (((srcDtype == DataType::FP8E4M3FN || srcDtype == DataType::FP8E5M2 || srcDtype == DataType::HF8) &&
              dstDtype == DataType::FP32) ||
             ((srcDtype == DataType::FP4E2M1 || srcDtype == DataType::FP4E1M2) && dstDtype == DataType::BF16)) {
        isFpWidenPp = true;
    }
    // FP8/FP4 4x narrowing with SAT (RND_SAT_PP): FP32→FP8
    // __VF_VCVTFF_RND_SAT_PP: vcvt(dst, src, mask, ROUND, SAT) — 5 args
    else if (srcDtype == DataType::FP32 &&
             (dstDtype == DataType::FP8E4M3FN || dstDtype == DataType::FP8E5M2 || dstDtype == DataType::HF8)) {
        isFpNarrowRndSatPp = true;
    }
    // FP4 4x narrowing without SAT (RND_PP): BF16→FP4
    // __VF_VCVTFF_RND_PP: vcvt(dst, src, mask, ROUND, PART_PP, MODE_ZEROING) — 6 args
    else if (srcDtype == DataType::BF16 && (dstDtype == DataType::FP4E2M1 || dstDtype == DataType::FP4E1M2)) {
        isFpNarrowRndPp = true;
    }
    // Remaining int→int paths not directly supported by hardware vcvt overloads:
    // cross-sign widening (u→s, s→u) and int→INT8 narrowing.
    // Route through f16 intermediate (two-step), mirroring AscendC's s16→f16→s4 approach.
    if (srcDtype.IsInt() && dstDtype.IsInt() && !isS4Widening && !isS4Narrowing && !isS16ToS4 && !isWidening &&
        !isInt4xWidening && !isInt4xNarrowing && !isInt2xNarrowing) {
        isIntIntTwoStep = true;
    }

    // Validate round_mode against the specific vcvt overload constraints:
    // - Widening (no precision loss): round_mode is ignored (UNKNOWN), warn if set
    // - is_int_to_float / is_float_to_same_int: R/A/F/C/Z only (no O/H)
    // - is_fp_narrow_rnd_sat_pp (FP32→FP8E4M3FN/FP8E5M2): only ROUND_R
    // - is_fp_narrow_rnd_sat_pp (FP32→HF8): only ROUND_A/ROUND_H
    // - is_fp_narrow_rnd_pp (BF16→FP4): R/A/F/C/Z only (no O/H)
    // - FP16→HF8 (else/fallback path): only ROUND_A/ROUND_H
    // - is_fp_widen_pp (FP8→FP32, FP4→BF16): no round_mode (UNKNOWN)
    // - is_s4_widening (INT4→FP16/BF16/INT16): no round_mode (UNKNOWN)
    // - is_int_2x_narrowing / is_int_4x_narrowing / is_int_4x_widening / is_s16_to_s4: no round_mode
    if (isWidening || isFpWidenPp || isS4Widening || isS16ToS4 || isInt2xNarrowing || isInt4xNarrowing ||
        isInt4xWidening) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("round_mode") || roundMode == "CAST_RINT")
            << "vf.astype: round_mode is not applicable for this widening/no-precision-loss "
            << "conversion path (src=" << DTypeStr(srcDtype) << ", dst=" << DTypeStr(dstDtype)
            << "), only default CAST_RINT is accepted";
    }
    if (isIntToFloat || isFloatToSameInt || isFloatToNarrowerInt || isFpNarrowRndPp || isFloatToWiderInt ||
        isCrossWidth || isIntIntTwoStep) {
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, round != "ROUND_O" && round != "ROUND_H")
            << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for this conversion path "
            << "(src=" << DTypeStr(srcDtype) << ", dst=" << DTypeStr(dstDtype) << "), "
            << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
    }
    if (isFpNarrowRndSatPp) {
        if (dstDtype == DataType::HF8) {
            // FP32→HF8: only CAST_ROUND/CAST_HYBRID. Default CAST_RINT is invalid,
            // so if user didn't specify, silently use CAST_ROUND (the hardware default).
            if (!op->HasKwarg("round_mode")) {
                round = "ROUND_A";
            } else {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, round == "ROUND_A" || round == "ROUND_H")
                    << "vf.astype: FP32→HF8 only supports round_mode CAST_ROUND/CAST_HYBRID, got " << roundMode;
            }
        } else {
            // FP32→FP8E4M3FN/FP8E5M2: only CAST_RINT. Default is CAST_RINT, so
            // not specifying round_mode is fine.
            if (op->HasKwarg("round_mode")) {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, round == "ROUND_R")
                    << "vf.astype: FP32→FP8E4M3FN/FP8E5M2 only supports round_mode CAST_RINT, got " << roundMode;
            }
        }
    }
    if (srcDtype == DataType::FP16 && dstDtype == DataType::HF8) {
        // FP16→HF8: only CAST_ROUND/CAST_HYBRID. Default CAST_RINT is invalid,
        // so if user didn't specify, silently use CAST_ROUND.
        if (!op->HasKwarg("round_mode")) {
            round = "ROUND_A";
        } else {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, round == "ROUND_A" || round == "ROUND_H")
                << "vf.astype: FP16→HF8 only supports round_mode CAST_ROUND/CAST_HYBRID, got " << roundMode;
        }
    }
    // FP16→INT4 (is_s4_narrowing): R/A/F/C/Z only (no O/H)
    if (isS4Narrowing) {
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, round != "ROUND_O" && round != "ROUND_H")
            << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for FP16→INT4, "
            << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
    }
    // Validate saturate against conversion path:
    // - Widening / is_fp_widen_pp / is_s4_widening / is_int_4x_widening: saturate not applicable
    // - is_int_to_float (INT→FLOAT): saturate not applicable (default saturated, no choice)
    // - is_int_4x_narrowing (b32→u8): saturate is mandatory (always RS_ENABLE)
    // - is_int_2x_narrowing (b32→b16, b16→u8, b64→b32): saturate OFF/ON both supported
    // - is_s16_to_s4 (INT16→INT4): saturate OFF/ON both supported (like is_s4_narrowing)
    // - is_float_to_narrower_int (FP16→INT8/UINT8): saturate OFF/ON both supported
    // - is_float_to_same_int (FP32→S32): saturate OFF/ON both supported
    // - BF16→FP16 (float→float same-width): saturate OFF/ON both supported
    // - is_fp_narrow_rnd_sat_pp: saturate is mandatory (always RS_ENABLE)
    // - is_fp_narrow_rnd_pp (BF16→FP4): saturate not applicable (UNKNOWN)
    // - FP→FP32 (widening float): only OFF (non-saturated)
    if (isWidening || isFpWidenPp || isS4Widening || isInt4xWidening || isFpNarrowRndPp ||
        (isIntToFloat && !(srcDtype == DataType::BF16 && dstDtype == DataType::FP16))) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("saturate") || sat == "RS_DISABLE")
            << "vf.astype: saturate is not applicable for this conversion path " << "(src=" << DTypeStr(srcDtype)
            << ", dst=" << DTypeStr(dstDtype) << ")";
    }
    if (isInt4xNarrowing) {
        // b32→b8 (4x narrowing): saturate is always enabled (RS_ENABLE). Default is RS_DISABLE,
        // so if user didn't specify, silently enable it.
        if (!op->HasKwarg("saturate")) {
            sat = "RS_ENABLE";
        } else {
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, sat == "RS_ENABLE")
                << "vf.astype: 4x int narrowing (b32→b8) requires saturate=ON (RS_ENABLE), "
                << "OFF is not supported for this path";
        }
    }
    if (isFpNarrowRndSatPp) {
        // FP32→FP8: saturate is always enabled (RS_ENABLE). Default is RS_DISABLE,
        // so if user didn't specify, silently enable it.
        if (!op->HasKwarg("saturate")) {
            sat = "RS_ENABLE";
        } else {
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, sat == "RS_ENABLE")
                << "vf.astype: FP32→FP8 conversion requires saturate=ON (RS_ENABLE), "
                << "OFF is not supported for this path";
        }
    }
    if (dstDtype == DataType::FP32 && srcDtype.GetBit() < 32 && (srcDtype.IsFloat() || srcDtype == DataType::HF8)) {
        // FP→FP32 widening: only non-saturated mode
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, sat == "RS_DISABLE")
            << "vf.astype: conversion to FP32 only supports saturate=OFF (non-saturated mode)";
    }
    // Validate layout against conversion path:
    // - Same-width conversions (is_int_to_float, is_float_to_same_int): layout not applicable
    // - FP16→BF16 and BF16→FP16 (same-width float→float): layout not applicable
    // - FP32→INT64 and INT64→FP32: same 64-bit width, layout not applicable
    if (isIntToFloat || isFloatToSameInt || (srcDtype == DataType::FP16 && dstDtype == DataType::BF16) ||
        (srcDtype == DataType::BF16 && dstDtype == DataType::FP16) ||
        (srcDtype == DataType::FP32 && dstDtype == DataType::INT64) ||
        (srcDtype == DataType::INT64 && dstDtype == DataType::FP32)) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("layout") || layout == "ZERO")
            << "vf.astype: layout is not applicable for this same-width conversion path "
            << "(src=" << DTypeStr(srcDtype) << ", dst=" << DTypeStr(dstDtype) << ")";
    }
    // FP8/FP4 widening: layout supports ZERO/ONE/TWO/THREE (4x expansion)
    // FP8/FP4 2x widening (HF8→FP16): layout supports ZERO/ONE only
    if (isFpWidenPp) {
        if (srcDtype == DataType::HF8 && dstDtype == DataType::FP16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
                << "vf.astype: HF8→FP16 only supports layout ZERO/ONE, got " << layout;
        } else if (srcDtype == DataType::FP8E8M0 && dstDtype == DataType::BF16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
                << "vf.astype: FP8E8M0→BF16 only supports layout ZERO/ONE, got " << layout;
        }
        // FP8E4M3FN/FP8E5M2→FP32 and FP4→BF16 support all four layouts
    }
    // 2x widening (is_widening): layout supports ZERO/ONE only
    if (isWidening) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.astype: 2x widening conversion only supports layout ZERO/ONE, got " << layout
            << " (src=" << DTypeStr(srcDtype) << ", dst=" << DTypeStr(dstDtype) << ")";
    }
    // 4x int widening (b8→b32): layout supports ZERO/ONE/TWO/THREE
    if (isInt4xWidening) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype: 4x int widening conversion only supports layout ZERO/ONE/TWO/THREE, got " << layout
            << " (src=" << DTypeStr(srcDtype) << ", dst=" << DTypeStr(dstDtype) << ")";
    }
    // 4x int narrowing (b32→u8) and INT16→INT4: layout supports ZERO/ONE/TWO/THREE
    if (isInt4xNarrowing || isS16ToS4) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          layout == "ZERO" || layout == "ONE" || layout == "TWO" || layout == "THREE")
            << "vf.astype: 4x int narrowing conversion only supports layout ZERO/ONE/TWO/THREE, got " << layout
            << " (src=" << DTypeStr(srcDtype) << ", dst=" << DTypeStr(dstDtype) << ")";
    }
    // Float narrowing with PART (partCondition in AscendC): layout supports ZERO/ONE only.
    // This covers FP32→FP16, FP32→BF16, FP16→HF8, FP16→UINT8, FP16→INT8, BF16→INT32, FP32→INT16,
    // FP32→INT64 (all use vcvt(dst,src,mask,ROUND,SAT,PART,MODE) — 7 args).
    // AscendC: static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>());
    if (!isS4Widening && !isS4Narrowing && !isS16ToS4 && !isWidening && !isFpWidenPp && !isIntToFloat &&
        !isFloatToSameInt && !isFloatToNarrowerInt && !isFloatToWiderInt && !isCrossWidth && !isInt2xNarrowing &&
        !isIntIntTwoStep && !isInt4xNarrowing && !isInt4xWidening && !isFpNarrowRndSatPp && !isFpNarrowRndPp &&
        !(srcDtype == DataType::FP16 && dstDtype == DataType::BF16) &&
        !(srcDtype == DataType::BF16 && dstDtype == DataType::FP16) &&
        !(srcDtype == DataType::FP32 && dstDtype == DataType::INT64) &&
        !(srcDtype == DataType::INT64 && dstDtype == DataType::FP32)) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.astype: layout only supports ZERO/ONE for this conversion path " << "(src=" << DTypeStr(srcDtype)
            << ", dst=" << DTypeStr(dstDtype) << "), got " << layout;
        // Round mode validation for fallback (partCondition) paths:
        // FP32→FP16: supports CAST_ODD but NOT CAST_HYBRID
        if (srcDtype == DataType::FP32 && dstDtype == DataType::FP16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, round != "ROUND_H")
                << "vf.astype: FP32→FP16 does not support round_mode CAST_HYBRID, "
                << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC/CAST_ODD";
        } else {
            // All other fallback paths (FP32→BF16, FP16→INT8, FP16→UINT8, BF16→FP16, FP32→INT16, etc.):
            // support CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC only (no CAST_ODD/CAST_HYBRID)
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, round != "ROUND_O" && round != "ROUND_H")
                << "vf.astype: round_mode CAST_ODD/CAST_HYBRID is not supported for this conversion path "
                << "(src=" << DTypeStr(srcDtype) << ", dst=" << DTypeStr(dstDtype) << "), "
                << "supported values are CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
        }
    }
    if (isS4Widening) {
        // INT4→FP16/BF16/INT16: specialized vcvt_s42*
        if (dstDtype == DataType::FP16) {
            codegen.Emit("vcvt_s42f16(" + dst + ", " + src + ", " + mask + ", " + partPp + ", " + modeValue + ");");
        } else if (dstDtype == DataType::BF16) {
            codegen.Emit("vcvt_s42bf16(" + dst + ", " + src + ", " + mask + ", " + partPp + ", " + modeValue + ");");
        } else {
            codegen.Emit("vcvt_s42s16(" + dst + ", " + src + ", " + mask + ", " + partPp + ", " + modeValue + ");");
        }
    } else if (isS4Narrowing) {
        // FP16→INT4: vcvt_f162s4
        codegen.Emit("vcvt_f162s4(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + partPp +
                     ", " + modeValue + ");");
    } else if (isWidening) {
        // vcvt(dst, src, mask, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + part + ", " + modeValue + ");");
    } else if (isIntToFloat) {
        // vcvt(dst, src, mask, ROUND, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + modeValue + ");");
    } else if (isFloatToSameInt) {
        // vcvt(dst, src, mask, ROUND, RS, MODE_ZEROING) — no PART
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + modeValue + ");");
    } else if (srcDtype == DataType::FP32 && dstDtype == DataType::INT64) {
        // FP32→INT64: __VF_VCVTFI_SAT_PART(f32, s64) — the overload takes
        // (dst, src, mask, ROUND, RS, PART, MODE): the 5th arg is the
        // rounding-saturation flag and the 6th is PART_EVEN/PART_ODD (mirrors
        // AscendC CastImpl: vcvt(..., round, sat, part, mode), 7 args).
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.astype FP32->INT64 only supports layout ZERO/ONE, got " << layout;
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     modeValue + ");");
    } else if (isFloatToWiderInt || isCrossWidth) {
        // vcvt(dst, src, mask, ROUND, PART, MODE_ZEROING) — no RS
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + part + ", " + modeValue + ");");
    } else if (isFloatToNarrowerInt) {
        // FLOAT→narrower-INT (2x narrowing): FP16→INT8/UINT8, BF16→INT8/UINT8
        // AscendC CastOperator<T,U,RoundMode,SatMode,RegLayout,MaskMergeMode> emits:
        //   vcvt(dstReg, srcReg, mask, roundModeValue, satModeValue, partModeValue, modeValue)
        // satMode is passed through directly (NO_SAT → RS_DISABLE, SAT → RS_ENABLE).
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     modeValue + ");");
    } else if (isInt4xWidening) {
        // 4x int widening (b8→b32): vcvt(dst, src, mask, PART_PP, MODE_ZEROING) — 5 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + partPp + ", " + modeValue + ");");
    } else if (isInt4xNarrowing) {
        // 4x int narrowing to UINT8 (b32→u8): vcvt(dst, src, mask, RS, PART_PP, MODE) — 6 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + sat + ", " + partPp + ", " + modeValue + ");");
    } else if (isInt2xNarrowing) {
        // 2x int narrowing with RS (b32→b16, b16→u8, b64→b32): vcvt(dst, src, mask, RS, PART, MODE) — 6 args
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + sat + ", " + part + ", " + modeValue + ");");
    } else if (isS16ToS4) {
        // INT16→INT4: two-step conversion mirroring AscendC CastOperator (s16→f16→s4).
        // AscendC hardcodes RoundMode::CAST_RINT for both steps (round_mode not in trait).
        // Step 1: vcvt(tmp_f16, src_s16, mask, ROUND_R, MODE) — s16→f16 (5 args, rndMergeCast)
        // Step 2: vcvt_f162s4(dst_s4, tmp_f16, mask, ROUND_R, SAT, PART_PP, MODE) — f16→s4 (7 args)
        std::string tmp = dst + "_f16_tmp";
        codegen.Emit("RtWrap<half> " + tmp + ";");
        codegen.Emit("vcvt(" + tmp + ", " + src + ", " + mask + ", ROUND_R, " + modeValue + ");");
        codegen.Emit("vcvt_f162s4(" + dst + ", " + tmp + ", " + mask + ", ROUND_R, " + sat + ", " + partPp + ", " +
                     modeValue + ");");
    } else if (isIntIntTwoStep) {
        // Multi-step int→int for paths not directly supported by hardware vcvt overloads.
        // Available int→float overloads: b8→f16, s16→f16, s16→f32, s32→f32
        // NOT available: u16→f16, u16→f32, u32→f32, s32→f16
        //
        // Strategy by src/dst type combination:
        // 1. src→f16 available (b8, s16): two-step src→f16→dst
        // 2. s32/u32 src + b8 dst: three-step src→f32→f16→dst (f32→f16 narrowing needed)
        // 3. u16 src + b8 dst: four-step u16→u32→f32→f16→dst (u16→f16 not available)
        // 4. u16 src + b32 dst: u16→u32 widening + dst cast (zero-extend, semantically correct)
        // 5. same-width cross-sign (u16→s16, s32→u32, u32→s32): reinterpret cast, no instruction
        std::string dstCType = dstDtype.ToCTypeString();
        bool srcToF16Ok = (srcDtype.GetBit() == 8) || (srcDtype == DataType::INT16);
        bool srcIsB32 = (srcDtype == DataType::INT32 || srcDtype == DataType::UINT32);
        bool dstIsB8 = (dstDtype.GetBit() == 8);
        bool dstIsB32 = (dstDtype.GetBit() == 32);
        bool sameWidth = (srcDtype.GetBit() == dstDtype.GetBit());

        if (srcToF16Ok) {
            // Two-step: src→f16→dst
            std::string tmp = dst + "_f16_tmp";
            codegen.Emit("RtWrap<half> " + tmp + ";");
            EmitIntToFloatVcvt(codegen, tmp, src, mask, part, round, modeValue, srcDtype);
            EmitFloatToIntVcvt(codegen, dst, tmp, mask, part, round, sat, modeValue, dstDtype);
        } else if (srcIsB32 && dstIsB8) {
            // Three-step: src→f32→f16→dst (s32→f32→f16→s8, u32 reinterpreted as s32 first)
            std::string tmpF32 = dst + "_f32_tmp";
            std::string tmpF16 = dst + "_f16_tmp";
            codegen.Emit("RtWrap<float> " + tmpF32 + ";");
            codegen.Emit("RtWrap<half> " + tmpF16 + ";");
            EmitIntToFloatVcvt(codegen, tmpF32, src, mask, part, round, modeValue, srcDtype);
            // f32→f16: float narrowing → vcvt(tmp_f16, tmp_f32, mask, ROUND, SAT, PART, MODE) — 7 args
            codegen.Emit("vcvt(" + tmpF16 + ", " + tmpF32 + ", " + mask + ", " + round + ", " + sat + ", " + part +
                         ", " + modeValue + ");");
            EmitFloatToIntVcvt(codegen, dst, tmpF16, mask, part, round, sat, modeValue, dstDtype);
        } else if (srcDtype == DataType::UINT16 && dstIsB8) {
            // Four-step: u16→u32→f32→f16→dst (u16→f16 not available, so widen to u32 first)
            std::string tmpU32 = dst + "_u32_tmp";
            std::string tmpF32 = dst + "_f32_tmp";
            std::string tmpF16 = dst + "_f16_tmp";
            codegen.Emit("RtWrap<uint32_t> " + tmpU32 + ";");
            codegen.Emit("RtWrap<float> " + tmpF32 + ";");
            codegen.Emit("RtWrap<half> " + tmpF16 + ";");
            // u16→u32: widening → vcvt(tmp_u32, src, mask, PART, MODE) — 5 args
            codegen.Emit("vcvt(" + tmpU32 + ", " + src + ", " + mask + ", " + part + ", " + modeValue + ");");
            EmitIntToFloatVcvt(codegen, tmpF32, tmpU32, mask, part, round, modeValue, DataType::UINT32);
            // f32→f16: float narrowing → vcvt(tmp_f16, tmp_f32, mask, ROUND, SAT, PART, MODE) — 7 args
            codegen.Emit("vcvt(" + tmpF16 + ", " + tmpF32 + ", " + mask + ", " + round + ", " + sat + ", " + part +
                         ", " + modeValue + ");");
            EmitFloatToIntVcvt(codegen, dst, tmpF16, mask, part, round, sat, modeValue, dstDtype);
        } else if (srcDtype == DataType::UINT16 && dstIsB32) {
            // u16→s32: u16→u32 widening with dst cast (zero-extend preserves value)
            std::string u32CType = DataType::UINT32.ToCTypeString();
            codegen.Emit("vcvt((RegTensor<" + u32CType + "> &)" + dst + ", " + src + ", " + mask + ", " + part + ", " +
                         modeValue + ");");
        } else if (sameWidth && srcDtype != dstDtype) {
            // Same-width cross-sign (u16→s16, s32→u32, u32→s32): reinterpret cast
            codegen.Emit(dst + " = (RegTensor<" + dstCType + "> &)" + src + ";");
        }
    } else if (isFpWidenPp) {
        // FP8/FP4 4x widening PP: vcvt(dst, src, mask, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + partPp + ", " + modeValue + ");");
    } else if (isFpNarrowRndSatPp) {
        // FP32→FP8 4x narrowing RND_SAT_PP: vcvt(dst, src, mask, ROUND, SAT, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + partPp + ", " +
                     modeValue + ");");
    } else if (isFpNarrowRndPp) {
        // BF16→FP4 4x narrowing RND_PP: vcvt(dst, src, mask, ROUND, PART_PP, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + partPp + ", " + modeValue + ");");
    } else {
        // Float narrowing (FP32→FP16, FP32→BF16, FP16→INT8, etc.)
        // vcvt(dst, src, mask, ROUND, RS, PART, MODE_ZEROING)
        codegen.Emit("vcvt(" + dst + ", " + src + ", " + mask + ", " + round + ", " + sat + ", " + part + ", " +
                     modeValue + ");");
    }
    return "";
}

// ============================================================================
// DeInterleave — vdintlv
// ============================================================================

static std::string EmitVFDeInterleave(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst pair prepended by codegen): [dst0, dst1, src0, src1].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.de_interleave requires 4 args (dst0, dst1, src0, src1)";
    std::string dst0 = codegen.GetExprAsCode(op->args_[0]);
    std::string dst1 = codegen.GetExprAsCode(op->args_[1]);
    std::string src0 = codegen.GetExprAsCode(op->args_[2]);
    std::string src1 = codegen.GetExprAsCode(op->args_[3]);
    if (IsSrcMaskReg(op, 2)) {
        DataType dtype = DataType::FP32;
        if (op->HasKwarg("dtype")) {
            dtype = op->GetKwarg<DataType>("dtype");
        }
        // MaskDeInterleave: only supports b8/b16/b32 (not b64)
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          dtype.GetBit() == 8 || dtype.GetBit() == 16 || dtype.GetBit() == 32)
            << "vf.de_interleave (MaskReg) only supports b8/b16/b32, got " << DTypeStr(dtype);
        std::string pdintlvOp;
        if (dtype == DataType::UINT8 || dtype == DataType::INT8) {
            pdintlvOp = "pdintlv_b8";
        } else if (dtype.GetBit() == 16) {
            pdintlvOp = "pdintlv_b16";
        } else {
            pdintlvOp = "pdintlv_b32";
        }
        codegen.Emit(pdintlvOp + "(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
        return "";
    }
    // DeInterleave requires all four operands to have the same dtype (mirrors
    // AscendC DeInterleaveImpl: dst0/dst1/src0/src1 are one register type U and
    // vdintlv is keyed on that element type). Cross-dtype reinterpretation is
    // the caller's job — cast explicitly first, e.g. the
    // `(RegTensor<uint8_t>&)vreg0U16` pattern in vf_topk.h:60.
    DataType s0Dt = GetExprDtype(op->args_[2]);
    DataType s1Dt = GetExprDtype(op->args_[3]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(s0Dt) || s0Dt.GetBit() == 16 || s0Dt.GetBit() == 32 || s0Dt.GetBit() == 64))
        << "vf.de_interleave only supports b8/b16/b32/b64 types, got " << DTypeStr(s0Dt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
        << "vf.de_interleave requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    if (s0Dt.GetBit() == 64) {
        codegen.Emit("vdintlv((RegTensor<uint32_t>&)" + dst0 + ", (RegTensor<uint32_t>&)" + dst1 +
                     ", (RegTensor<uint32_t>&)" + src0 + ", (RegTensor<uint32_t>&)" + src1 + ");");
    } else {
        codegen.Emit("vdintlv(" + dst0 + ", " + dst1 + ", " + src0 + ", " + src1 + ");");
    }
    return "";
}

// ============================================================================
// Select — vsel
// ============================================================================

static std::string EmitVFSelect(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src_true, src_false, mask].
    // The dst kind mirrors src_true.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.select requires 4 args (dst, src_true, src_false, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string srcTrue = codegen.GetExprAsCode(op->args_[1]);
    std::string srcFalse = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    if (IsSrcMaskReg(op, 1)) {
        codegen.Emit("psel(" + dst + ", " + srcTrue + ", " + srcFalse + ", " + mask + ");");
        return "";
    }
    // Doc: select supports BOOL/INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/INT64/UINT64
    DataType dstDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dstDt) || dstDt.GetBit() == 16 || dstDt.GetBit() == 32 || dstDt.GetBit() == 64)
        << "vf.select only supports b8/b16/b32/b64 types, got " << DTypeStr(dstDt);
    DataType stDt = GetExprDtype(op->args_[1]);
    DataType sfDt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dstDt.GetBit() == stDt.GetBit() && dstDt.GetBit() == sfDt.GetBit())
        << "vf.select requires dst, src_true, src_false to have the same bit width, got dst=" << dstDt.GetBit()
        << "-bit src_true=" << stDt.GetBit() << "-bit src_false=" << sfDt.GetBit() << "-bit";
    if (dstDt.GetBit() == 64) {
        // B64 select: deinterleave both sources, vsel on both b32 halves, interleave back.
        // Mirrors AscendC SelectImpl: B64TraitOneToTraitTwo + vsel(reg[0], reg[1]) + B64TraitTwoToTraitOne.
        std::string p = dst + "_sel_";
        EmitB64Deinterleave(codegen, p + "t", srcTrue);
        EmitB64Deinterleave(codegen, p + "f", srcFalse);
        std::string lot = p + "t_lo_", hit = p + "t_hi_";
        std::string lof = p + "f_lo_", hif = p + "f_hi_";
        // Pack b64 mask to b32 for vsel (mirrors MaskPack in AscendC)
        std::string packedM = p + "_pm_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        std::string loDst = p + "_lod_";
        std::string hiDst = p + "_hid_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loDst);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiDst);
        codegen.Emit("vsel(" + loDst + ", " + lot + ", " + lof + ", " + packedM + ");");
        codegen.Emit("vsel(" + hiDst + ", " + hit + ", " + hif + ", " + packedM + ");");
        EmitB64Interleave(codegen, dst, loDst, hiDst, p + "_ilv");
        return "";
    }
    std::string castPrefix = "(RegTensor<" + dstDt.ToCTypeString() + "> &)";
    std::string stExpr = (stDt == dstDt) ? srcTrue : (castPrefix + srcTrue);
    std::string sfExpr = (sfDt == dstDt) ? srcFalse : (castPrefix + srcFalse);
    codegen.Emit("vsel(" + dst + ", " + stExpr + ", " + sfExpr + ", " + mask + ");");
    return "";
}

// ============================================================================
// UpdateMask — plt_b32/plt_b16
// ============================================================================

static std::string EmitVFUpdateMask(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.update_mask requires 2 args (dst, scalar)";
    std::string scalar = codegen.GetExprAsCode(op->args_[1]);
    std::string regName = codegen.GetExprAsCode(op->args_[0]);
    // Default to b32 (float), use dtype kwarg to select b16 or b8
    bool useB8 = false;
    bool useB16 = false;
    bool useB64 = false;
    if (op->HasKwarg("dtype")) {
        auto dtype = op->GetKwarg<DataType>("dtype");
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                          IsB8Type(dtype) || dtype.GetBit() == 16 || dtype.GetBit() == 32 || dtype.GetBit() == 64)
            << "vf.update_mask dtype must be b8/b16/b32/b64, got " << DTypeStr(dtype);
        useB8 = (dtype == DataType::UINT8 || dtype == DataType::INT8);
        useB16 = (dtype.GetBit() == 16);
        useB64 = (dtype.GetBit() == 64);
    }
    // plt_b32/plt_b16 requires uint32_t& reference, so declare a variable first
    std::string scalarVar = "_vf_mask_scalar_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("uint32_t " + scalarVar + " = (uint32_t)" + scalar + ";");
    if (useB8) {
        codegen.Emit(regName + " = plt_b8(" + scalarVar + ", POST_UPDATE);");
    } else if (useB16) {
        codegen.Emit(regName + " = plt_b16(" + scalarVar + ", POST_UPDATE);");
    } else {
        // b32 and b64 both use plt_b32; b64 additionally needs punpack to expand
        // each bit into a pair, matching the 2-bit-per-element mask width.
        codegen.Emit(regName + " = plt_b32(" + scalarVar + ", POST_UPDATE);");
        if (useB64) {
            codegen.Emit("punpack(" + regName + ", " + regName + ", LOWER);");
        }
    }
    return "";
}

static std::string EmitVFHistograms(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask], or
    // [dst, old, src, mask] on a rebinding assignment (the previous dst value
    // is an explicit argument, copied into the fresh dst before the accumulate
    // instruction).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3 || op->args_.size() == 4)
        << op->name_ << " requires 3 args (dst, src, mask) or 4 (dst, old, src, mask)";
    size_t base = op->args_.size() == 4 ? 1 : 0;
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[base + 1]);
    std::string mask = codegen.GetExprAsCode(op->args_[base + 2]);
    // vhist: src must be uint8_t, dst is derived from src and cast to uint16_t
    DataType srcDt = GetExprDtype(op->args_[base + 1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt == DataType::UINT8)
        << op->name_ << " source only supports UINT8, got " << DTypeStr(srcDt);
    std::string binType = VFEnumValueName(ir::EnumToString(static_cast<ir::BinType>(op->GetKwarg<int>("bin_type"))));
    std::string binConst = (binType == "BIN1") ? "Bin_N1" : "Bin_N0";
    // hist_type: ACCUMULATE (chistv2, default) or FREQUENCY (dhistv2)
    std::string histType = "ACCUMULATE";
    if (op->HasKwarg("hist_type")) {
        histType = VFEnumValueName(ir::EnumToString(static_cast<ir::HistType>(op->GetKwarg<int>("hist_type"))));
    }
    std::string dstCast = "(RegTensor<uint16_t> &)";
    if (base == 1) {
        codegen.Emit(dst + " = " + codegen.GetExprAsCode(op->args_[1]) + ";");
    }
    if (histType == "FREQUENCY") {
        codegen.Emit("dhistv2(" + dstCast + dst + ", " + src + ", " + mask + ", " + binConst + ");");
    } else {
        codegen.Emit("chistv2(" + dstCast + dst + ", " + src + ", " + mask + ", " + binConst + ");");
    }
    return "";
}

static std::string EmitVFCompareImpl(const ir::CallPtr& op, codegen::CodegenBase& codegenBase,
                                     const std::string& cmpMode)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src0, src1, mask_src].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << op->name_ << " requires 4 args (dst, src0, src1, mask)";
    std::string maskDst = codegen.GetExprAsCode(op->args_[0]);
    std::string src0 = codegen.GetExprAsCode(op->args_[1]);
    std::string src1 = codegen.GetExprAsCode(op->args_[2]);
    std::string maskSrc = codegen.GetExprAsCode(op->args_[3]);
    // vcmp supports: u8,s8,u16,s16,u32,s32,half,float,bf16,u64,s64 (no bool, no FP8)
    DataType s0Dt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (IsArithIntType(s0Dt) || s0Dt == DataType::FP16 ||
                                                    s0Dt == DataType::FP32 || s0Dt == DataType::BF16))
        << op->name_ << " source only supports INT/UINT/FP16/FP32/BF16, got " << DTypeStr(s0Dt);
    DataType s1Dt = GetExprDtype(op->args_[2]);
    // FP8/FP4 types have no vcmp overloads (mirrors AscendC CompareImpl):
    // reject them for either compared operand.
    auto isFp8Fp4 = [](DataType dt) {
        return dt == DataType::FP8E4M3FN || dt == DataType::FP8E5M2 || dt == DataType::FP8E8M0 || dt == DataType::HF8 ||
               dt == DataType::FP4 || dt == DataType::FP4E2M1 || dt == DataType::FP4E1M2 || dt == DataType::HF4;
    };
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !isFp8Fp4(s0Dt) && !isFp8Fp4(s1Dt))
        << op->name_ << " does not support FP8/FP4 compare types, got src0=" << DTypeStr(s0Dt)
        << " src1=" << DTypeStr(s1Dt);
    // Bit width check: in scalar path, uses is_convertible (allows int64 scalar -> int32 reg)
    // so we defer the strict check to the vector-vector path only.
    // Here we just ensure s1 is not a wider type that can't convert (e.g. float vs int).
    bool isScalarSrc = !ir::IsRegTensor(op->args_[2]->GetType());
    if (!isScalarSrc) {
        // AscendC CompareImpl takes both sources as the same register type U —
        // mixed dtypes (even equal bit width) are rejected, no reinterpret casts.
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s0Dt == s1Dt)
            << op->name_ << " requires src0 and src1 to have the same type, got src0=" << DTypeStr(s0Dt)
            << " src1=" << DTypeStr(s1Dt);
    } else {
        // Scalar compare: is_convertible allows scalar to be wider than reg
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, s1Dt.GetBit() >= s0Dt.GetBit())
            << op->name_ << " scalar must be convertible to reg type, got src0=" << DTypeStr(s0Dt)
            << " scalar=" << DTypeStr(s1Dt);
    }
    std::string suffix = "eq";
    if (cmpMode == "NE")
        suffix = "ne";
    else if (cmpMode == "LT")
        suffix = "lt";
    else if (cmpMode == "GT")
        suffix = "gt";
    else if (cmpMode == "GE")
        suffix = "ge";
    else if (cmpMode == "LE")
        suffix = "le";
    if (isScalarSrc) {
        src1 = cce::CoerceScalarToInt(op->args_[2], s0Dt, src1);
        if (s0Dt.GetBit() == 64) {
            // B64 scalar compare: deinterleave src0, broadcast scalar halves, compare b32.
            // Must pack b64 mask to b32 (MaskPack) and unpack result (MaskUnPack),
            // matching AscendC CompareImpl for b64 RegTraitNumOne.
            std::string p = maskDst + "_cmps_";
            std::string packedM = p + "_pm_";
            codegen.Emit("MaskReg " + packedM + ";");
            codegen.Emit("ppack(" + packedM + ", " + maskSrc + ", LOWER);");
            EmitB64Deinterleave(codegen, p + "s", src0);
            std::string loS = p + "s_lo_", hiS = p + "s_hi_";
            std::string castType = (s0Dt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
            std::string loSc = p + "_losc_";
            std::string hiSc = p + "_hisc_";
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, loSc);
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiSc);
            codegen.Emit("vdup(" + loSc + ", (int32_t)(" + castType + "(" + src1 + ")), " + packedM +
                         ", MODE_ZEROING);");
            codegen.Emit("vdup(" + hiSc + ", (int32_t)((" + castType + "(" + src1 + ")) >> 32), " + packedM +
                         ", MODE_ZEROING);");
            // Compare high halves
            std::string hiR = p + "_hir_";
            std::string hiEq = p + "_hieq_";
            codegen.Emit("MaskReg " + hiR + ";");
            codegen.Emit("MaskReg " + hiEq + ";");
            if (s0Dt == DataType::INT64) {
                codegen.Emit("vcmp_" + suffix + "(" + hiR + ", (RegTensor<int32_t>&)" + hiS +
                             ", (RegTensor<int32_t>&)" + hiSc + ", " + packedM + ");");
                codegen.Emit("vcmp_eq(" + hiEq + ", (RegTensor<int32_t>&)" + hiS + ", (RegTensor<int32_t>&)" + hiSc +
                             ", " + packedM + ");");
            } else {
                codegen.Emit("vcmp_" + suffix + "(" + hiR + ", " + hiS + ", " + hiSc + ", " + packedM + ");");
                codegen.Emit("vcmp_eq(" + hiEq + ", " + hiS + ", " + hiSc + ", " + packedM + ");");
            }
            // Compare low halves (always unsigned) for equality tie-break
            std::string loR = p + "_lor_";
            codegen.Emit("MaskReg " + loR + ";");
            codegen.Emit("vcmp_" + suffix + "(" + loR + ", " + loS + ", " + loSc + ", " + packedM + ");");
            // For EQ: result = hi_eq AND lo_eq (lo_r is eq result)
            // For NE: result = NOT(hi_eq AND lo_eq)
            // For GT/GE/LT/LE: psel(lowCmp, highCmp, highEq) — if highEq pick lowCmp, else pick highCmp.
            // Matches AscendC CompareScalarImpl b64 path.
            if (cmpMode == "EQ") {
                codegen.Emit("pand(" + maskDst + ", " + hiEq + ", " + loR + ", " + packedM + ");");
            } else if (cmpMode == "NE") {
                // NE: low_ne OR high_ne. lo_r is vcmp_ne(low), hi_r is vcmp_ne(high).
                codegen.Emit("por(" + maskDst + ", " + loR + ", " + hiR + ", " + packedM + ");");
            } else {
                codegen.Emit("psel(" + maskDst + ", " + loR + ", " + hiR + ", " + hiEq + ");");
            }
            codegen.Emit("punpack(" + maskDst + ", " + maskDst + ", LOWER);");
        } else {
            codegen.Emit("vcmps_" + suffix + "(" + maskDst + ", " + src0 + ", " + src1 + ", " + maskSrc + ");");
        }
    } else {
        if (s0Dt.GetBit() == 64) {
            // B64 vector compare: deinterleave both sources, compare b32 halves.
            // Must pack b64 mask to b32 (MaskPack) and unpack result (MaskUnPack),
            // matching AscendC CompareImpl for b64 RegTraitNumOne.
            std::string p = maskDst + "_cmpv_";
            std::string packedM = p + "_pm_";
            codegen.Emit("MaskReg " + packedM + ";");
            codegen.Emit("ppack(" + packedM + ", " + maskSrc + ", LOWER);");
            EmitB64Deinterleave(codegen, p + "s0", src0);
            EmitB64Deinterleave(codegen, p + "s1", src1);
            std::string lo0 = p + "s0_lo_", hi0 = p + "s0_hi_";
            std::string lo1 = p + "s1_lo_", hi1 = p + "s1_hi_";
            std::string hiR = p + "_hir_";
            std::string hiEq = p + "_hieq_";
            codegen.Emit("MaskReg " + hiR + ";");
            codegen.Emit("MaskReg " + hiEq + ";");
            if (s0Dt == DataType::INT64) {
                codegen.Emit("vcmp_" + suffix + "(" + hiR + ", (RegTensor<int32_t>&)" + hi0 +
                             ", (RegTensor<int32_t>&)" + hi1 + ", " + packedM + ");");
                codegen.Emit("vcmp_eq(" + hiEq + ", (RegTensor<int32_t>&)" + hi0 + ", (RegTensor<int32_t>&)" + hi1 +
                             ", " + packedM + ");");
            } else {
                codegen.Emit("vcmp_" + suffix + "(" + hiR + ", " + hi0 + ", " + hi1 + ", " + packedM + ");");
                codegen.Emit("vcmp_eq(" + hiEq + ", " + hi0 + ", " + hi1 + ", " + packedM + ");");
            }
            std::string loR = p + "_lor_";
            codegen.Emit("MaskReg " + loR + ";");
            codegen.Emit("vcmp_" + suffix + "(" + loR + ", " + lo0 + ", " + lo1 + ", " + packedM + ");");
            if (cmpMode == "EQ") {
                codegen.Emit("pand(" + maskDst + ", " + hiEq + ", " + loR + ", " + packedM + ");");
            } else if (cmpMode == "NE") {
                // NE: low_ne OR high_ne. lo_r is vcmp_ne(low), hi_r is vcmp_ne(high).
                codegen.Emit("por(" + maskDst + ", " + loR + ", " + hiR + ", " + packedM + ");");
            } else {
                // GT/GE/LT/LE: psel(lowCmp, highCmp, highEq) — if highEq pick lowCmp, else pick highCmp.
                // Matches AscendC CompareInt64Impl/CompareUint64Impl.
                codegen.Emit("psel(" + maskDst + ", " + loR + ", " + hiR + ", " + hiEq + ");");
            }
            codegen.Emit("punpack(" + maskDst + ", " + maskDst + ", LOWER);");
        } else {
            // Same-type operands (enforced above): no reinterpret casts needed.
            codegen.Emit("vcmp_" + suffix + "(" + maskDst + ", " + src0 + ", " + src1 + ", " + maskSrc + ");");
        }
    }
    return "";
}

static std::string EmitVFSqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask]. The registry
    // deduce gives the dst the src's element type, so no cast is needed.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.squeeze requires 3 args (dst, src, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(
        ExternalError::INVALID_TYPE,
        (srcDt == DataType::INT8 || srcDt == DataType::UINT8 || srcDt == DataType::INT16 || srcDt == DataType::UINT16 ||
         srcDt == DataType::INT32 || srcDt == DataType::UINT32 || srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.squeeze src only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/FP16/FP32, got " << DTypeStr(srcDt);
    std::string dstExpr = dst;
    std::string srcExpr = src;
    // mode kwarg: "STORED" (default) or "NO_STORED"
    // backward compat: gather_mode="STORE_REG" / "NO_STORE_REG"
    std::string mode = "MODE_STORED";
    if (op->HasKwarg("gather_mode")) {
        auto gm = static_cast<ir::SqueezeMode>(op->GetKwarg<int>("gather_mode"));
        if (gm == ir::SqueezeMode::NO_STORE_REG)
            mode = "MODE_NO_STORED";
        else
            mode = "MODE_STORED";
    }
    codegen.Emit("vsqz(" + dstExpr + ", " + srcExpr + ", " + mask + ", " + mode + ");");
    return "";
}

static std::string EmitVFArange(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, start].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.arange requires 2 args (dst, start)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string start = codegen.GetExprAsCode(op->args_[1]);
    DataType dstDt = DataType::INT32;
    if (op->HasKwarg("dtype")) {
        dstDt = op->GetKwarg<DataType>("dtype");
    } else if (const auto& t = op->args_[1]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    // vci: supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/FP16/FP32 (no UINT64)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      dstDt == DataType::INT8 || dstDt == DataType::UINT8 || dstDt == DataType::INT16 ||
                          dstDt == DataType::UINT16 || dstDt == DataType::INT32 || dstDt == DataType::UINT32 ||
                          dstDt == DataType::INT64 || dstDt == DataType::FP16 || dstDt == DataType::FP32)
        << "vf.arange only supports INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/FP16/FP32, got " << DTypeStr(dstDt);
    // vci: scalarValue must be convertible to RegTensor data type
    // is_convertible<U, ActualT>() — start must be convertible to dst type.
    // This allows e.g. int64(index) start with int32 dst (narrowing conversion).
    // Only reject if start is smaller than dst (would lose data).
    // index_order kwarg selects the direction: INCREASE_ORDER (default) ->
    // dst[i] = start + i; DECREASE_ORDER -> dst[i] = start - i.
    bool isDecrease = false;
    if (op->HasKwarg("index_order")) {
        auto o = static_cast<ir::IndexOrder>(op->GetKwarg<int>("index_order"));
        if (o == ir::IndexOrder::DECREASE_ORDER)
            isDecrease = true;
    }
    // vci accepts signed integer types (int8/int16/int32) and float types (half/float).
    // Unsigned types (uint8/uint16/uint32) have no vci overload — cast to signed.
    // Signed and float types are passed directly (no cast needed).
    std::string elemType = dstDt.ToCTypeString();
    if (dstDt == DataType::UINT8)
        elemType = "int8_t";
    else if (dstDt == DataType::UINT16)
        elemType = "int16_t";
    else if (dstDt == DataType::UINT32)
        elemType = "int32_t";
    // b64 (INT64/UINT64): single vci does not support 8-byte elements.
    // Build the index in u32 halves, fold in the 64-bit start scalar with a
    // carry chain (mirrors EmitVFAdds' b64 scalar add: vdup the scalar halves,
    // vaddc/vaddcs), then vintlv the halves into dst:
    //   lo = i + (u32)start;  hi = (u32)(start >> 32) + carry
    // (b64 vadds on a single 256B RegTensor<int64_t> is not callable — the
    // bisheng overload takes vector_2xvl_s64 512B register pairs.)
    // NOTE: DECREASE_ORDER assumes non-negative results (no borrow from the
    // low half into the sign-extended high half).
    if (dstDt == DataType::INT64 || dstDt == DataType::UINT64) {
        std::string lo = dst + "_b64_lo_";
        std::string hi = dst + "_b64_hi_";
        std::string dump = dst + "_b64_dump_";
        std::string m = dst + "_b64_m_";
        std::string castType = (dstDt == DataType::INT64) ? "(int64_t)" : "(uint64_t)";
        std::string b32Cast = (dstDt == DataType::INT64) ? "(RegTensor<int32_t>&)" : "(RegTensor<uint32_t>&)";
        EmitRtWrapDecl(codegen, ir::DataType::INT32, lo);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, hi);
        EmitRtWrapDecl(codegen, ir::DataType::INT32, dump);
        codegen.Emit("MaskReg " + m + " = pset_b32(PAT_ALL);");
        codegen.Emit("vci(" + lo + ", 0, INC_ORDER);");
        if (isDecrease) {
            codegen.Emit("vneg(" + lo + ", " + lo + ", " + m + ", MODE_ZEROING);");
            // Sign-extend the negated low half: lane 0 keeps hi=0, lanes i>=1
            // extend to 0xFFFFFFFF so the pair holds the b64 value -i.
            codegen.Emit("vshrs(" + hi + ", " + lo + ", (int16_t)31, " + m + ", MODE_ZEROING);");
        } else {
            codegen.Emit("vdup(" + hi + ", 0, " + m + ", MODE_ZEROING);");
        }
        std::string scLo = dst + "_b64_sclo_";
        std::string scHi = dst + "_b64_schi_";
        std::string carry = dst + "_b64_carry_";
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, scLo);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, scHi);
        codegen.Emit("MaskReg " + carry + ";");
        codegen.Emit("vdup(" + scLo + ", (int32_t)(" + castType + "(" + start + ")), " + m + ", MODE_ZEROING);");
        codegen.Emit("vdup(" + scHi + ", (int32_t)((" + castType + "(" + start + ")) >> 32), " + m +
                     ", MODE_ZEROING);");
        codegen.Emit("vaddc(" + carry + ", " + b32Cast + lo + ", " + b32Cast + lo + ", " + b32Cast + scLo + ", " + m +
                     ");");
        codegen.Emit("vaddcs(" + carry + ", " + b32Cast + hi + ", " + b32Cast + hi + ", " + b32Cast + scHi + ", " +
                     carry + ", " + m + ");");
        codegen.Emit("vintlv((RegTensor<uint32_t> &)" + dst + ", (RegTensor<uint32_t> &)" + dump +
                     ", (RegTensor<uint32_t> &)" + lo + ", (RegTensor<uint32_t> &)" + hi + ");");
        return "";
    }
    // Non-b64: vci INC_ORDER generates value, value+1, ..., value+VL-1.
    // vci DEC_ORDER generates value+VL-1, value+VL-2, ..., value.
    std::string orderStr = isDecrease ? "DEC_ORDER" : "INC_ORDER";
    if (elemType != dstDt.ToCTypeString()) {
        codegen.Emit("vci((RegTensor<" + elemType + "> &)" + dst + ", " + start + ", " + orderStr + ");");
    } else {
        codegen.Emit("vci(" + dst + ", " + start + ", " + orderStr + ");");
    }
    return "";
}

static std::string EmitVFGather(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Two shapes, dispatched by src argument type; dst is prepended by codegen:
    //   Reg-to-Reg: args = [dst, src_reg, indices]  — no mask, src/dst same type
    //   UB-to-Reg:  args = [dst, src_ub, indices, mask] — with mask, b8 zero-extends to b16
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3 || op->args_.size() == 4)
        << "vf.gather requires 3 args (dst, src, indices) or 4 (dst, src, indices, mask)";
    auto srcTileType = ir::As<ir::TileType>(op->args_[1]->GetType());
    if (!srcTileType) {
        // Reg-to-Reg form
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, op->args_.size() == 3)
            << "vf.gather (reg→reg) requires exactly 3 args (dst, src, indices), "
            << "mask is not supported in reg→reg form; got " << op->args_.size() << " args";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("data_copy_mode"))
            << "vf.gather (reg→reg) does not support data_copy_mode " << "(only Tile→Reg form supports it)";
        // Supports b8/b16/b32; src and dst must have the same type (not just the
        // same bit width — e.g. FP16 src with BF16 dst must be rejected); b64 is
        // not supported (vselr limitation, mirrors AscendC GatherImpl
        // SupportBytes<1,2,4> and the gather.md reg→reg type table).
        DataType srcDt = GetExprDtype(op->args_[1]);
        DataType idxDt = GetExprDtype(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, srcDt.GetBit() != 64)
            << "vf.gather (reg→reg) does not support b64 types (vselr limitation), got " << DTypeStr(srcDt);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt.GetBit() == idxDt.GetBit()))
            << "vf.gather (reg→reg) requires index bit width to match src, got src=" << DTypeStr(srcDt)
            << " index=" << DTypeStr(idxDt);
        DataType dstDt = srcDt;
        std::string dst = codegen.GetExprAsCode(op->args_[0]);
        std::string src = codegen.GetExprAsCode(op->args_[1]);
        std::string indices = codegen.GetExprAsCode(op->args_[2]);
        std::string castType = "uint32_t";
        if (dstDt.GetBit() <= 8) {
            castType = "uint8_t";
        } else if (dstDt.GetBit() == 16) {
            castType = "uint16_t";
        }
        codegen.Emit("vselr((RegTensor<" + castType + ">&)" + dst + ", (RegTensor<" + castType + ">&)" + src +
                     ", (RegTensor<" + castType + ">&)" + indices + ");");
        return "";
    }

    // UB-to-Reg form
    // args: [dst, src_ub, indices, mask]
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.gather requires 4 args (dst, src, indices, mask)";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType srcDt = GetExprDtype(op->args_[1]);
    DataType dstDt = srcDt;
    if (const auto& t = op->args_[1]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    DataType idxDt = GetExprDtype(op->args_[2]);
    std::string indices = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);

    // Check mode: DATA_BLOCK_LOAD -> block gather, otherwise -> per-element gather
    bool isDatablock = false;
    if (op->HasKwarg("data_copy_mode")) {
        auto mode = static_cast<ir::DataCopyMode>(op->GetKwarg<int>("data_copy_mode"));
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                          mode == ir::DataCopyMode::NORM || mode == ir::DataCopyMode::DATA_BLOCK_LOAD)
            << "vf.gather only supports data_copy_mode=NORM or DATA_BLOCK_LOAD, got "
            << VFEnumValueName(ir::EnumToString(mode));
        isDatablock = (mode == ir::DataCopyMode::DATA_BLOCK_LOAD);
        if (!isDatablock) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("block_stride"))
                << "vf.gather (NORM mode) does not support block_stride";
        }
    }

    if (isDatablock) {
        // DataCopyGatherB: dst b8/b16/b32/b64, index must be uint32_t
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT32)
            << "vf.gather (DATA_BLOCK_LOAD) index must be UINT32, got " << DTypeStr(idxDt);
        // Block gather: always casts to signed types (s8/s16/s32/s64)
        // and uses int8_t*/int16_t*/int32_t*/int64_t* for the UB pointer.
        std::string signedCType;
        if (dstDt.GetBit() <= 8) {
            signedCType = "int8_t";
        } else if (dstDt.GetBit() == 16) {
            signedCType = "int16_t";
        } else if (dstDt.GetBit() == 32) {
            signedCType = "int32_t";
        } else {
            signedCType = "int64_t";
        }
        std::string gbUbPtr = GetUBufPtr(codegen, op->args_[1], signedCType);
        codegen.Emit("vgatherb((RegTensor<" + signedCType + ">&)" + dst + ", " + gbUbPtr + ", (RegTensor<uint32_t> &)" +
                     indices + ", " + mask + ");");
    } else {
        // DataCopyGather: specific src-dst-index type combinations
        // src b8 -> dst b16 + idx u16; src b16 -> dst b16 + idx u16/u32;
        // src b32 -> dst b32 + idx u32; src b64 -> dst b64 + idx u32/u64
        bool isB16Src = (dstDt == DataType::INT16 || dstDt == DataType::UINT16 || dstDt == DataType::FP16 ||
                         dstDt == DataType::BF16);
        bool useVgather2Bc = isB16Src && (idxDt.GetBit() >= 32);
        if (dstDt.GetBit() == 16) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt.GetBit() == 8 || srcDt.GetBit() == 16)
                << "vf.gather (NORM) b16 dst requires b8/b16 src, got src=" << DTypeStr(srcDt);
            if (!useVgather2Bc) {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT16)
                    << "vf.gather (NORM) b16 dst requires UINT16 index (or UINT32 for vgather2_bc), got "
                    << DTypeStr(idxDt);
            } else {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT32)
                    << "vf.gather (NORM) b16 dst with vgather2_bc requires UINT32 index, got " << DTypeStr(idxDt);
            }
        } else if (dstDt.GetBit() == 32) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt.GetBit() == 32)
                << "vf.gather (NORM) b32 requires b32 src, got src=" << DTypeStr(srcDt);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT32)
                << "vf.gather (NORM) b32 dst requires UINT32 index, got " << DTypeStr(idxDt);
        } else if (dstDt.GetBit() == 64) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt.GetBit() == 64)
                << "vf.gather (NORM) b64 requires b64 src, got src=" << DTypeStr(srcDt);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT32)
                << "vf.gather (NORM) b64 dst requires UINT32 index, got " << DTypeStr(idxDt);
        }
        if (dstDt.GetBit() == 64) {
            // B64 gather: vgather2 has no b64 single-register overload.
            // Mirrors AscendC DataCopyGatherB64Impl: pack mask, compute odd/even
            // b32 indices (index*2, index*2+1), vgather2 each b32 half, interleave.
            std::string p = dst + "_gather_";
            std::string packedM = p + "_pm_";
            std::string vl32M = p + "_vl32_";
            std::string allM = p + "_allm_";
            std::string andM = p + "_andm_";
            std::string loIdx = p + "_loi_";
            std::string hiIdx = p + "_hii_";
            std::string loReg = p + "_lor_";
            std::string hiReg = p + "_hir_";
            std::string tmpReg = p + "_tmp_";
            codegen.Emit("MaskReg " + packedM + ";");
            codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
            // MaskAnd with VL32: b64 has 32 elements → 64 b32 slots after ppack,
            // but only first 32 b32 slots are valid. Mirrors AscendC
            // DataCopyGatherB64Impl: MaskAnd(dstMask, dstMask, lowerMask, preg).
            codegen.Emit("MaskReg " + vl32M + " = pset_b32(PAT_VL32);");
            codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
            codegen.Emit("MaskReg " + andM + ";");
            codegen.Emit("pand(" + andM + ", " + packedM + ", " + vl32M + ", " + allM + ");");
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, loIdx);
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiIdx);
            codegen.Emit("vmuls(" + loIdx + ", (RegTensor<uint32_t> &)" + indices + ", (uint32_t)2, " + andM +
                         ", MODE_ZEROING);");
            codegen.Emit("vadds(" + hiIdx + ", " + loIdx + ", (uint32_t)1, " + andM + ", MODE_ZEROING);");
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, loReg);
            EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiReg);
            std::string ubPtr = GetUBufPtr(codegen, op->args_[1], "uint32_t");
            codegen.Emit("vgather2(" + loReg + ", " + ubPtr + ", " + loIdx + ", " + andM + ");");
            codegen.Emit("vgather2(" + hiReg + ", " + ubPtr + ", " + hiIdx + ", " + andM + ");");
            EmitB64Interleave(codegen, dst, loReg, hiReg, p + "_ilv");
            return "";
        }
        std::string idxCType = useVgather2Bc ? "uint32_t" : ((dstDt.GetBit() >= 32) ? "uint32_t" : "uint16_t");
        std::string dstExpr = dst;
        std::string ubPtr = GetUBufPtr(codegen, op->args_[1], srcDt.ToCTypeString());
        if (dstDt == DataType::INT8 || dstDt == DataType::UINT8) {
            dstExpr = "(RegTensor<int16_t>&)" + dst;
            ubPtr = GetUBufPtr(codegen, op->args_[1], "int8_t");
        } else if (isB16Src) {
            // vgather2_bc b16 overload expects vector_s16& (RegTensor<int16_t>&).
            // The b16 element occupies the lower 16 bits of each 32-bit slot;
            // the upper 16 bits are zero. Both vgather2 and vgather2_bc use the
            // same dst cast for b16 source types.
            dstExpr = "(RegTensor<int16_t>&)" + dst;
            ubPtr = GetUBufPtr(codegen, op->args_[1], "int16_t");
        } else if (dstDt.GetBit() == 32) {
            dstExpr = "(RegTensor<int32_t>&)" + dst;
            ubPtr = GetUBufPtr(codegen, op->args_[1], "int32_t");
        }
        std::string instr = useVgather2Bc ? "vgather2_bc" : "vgather2";
        codegen.Emit(instr + "(" + dstExpr + ", " + ubPtr + ", (RegTensor<" + idxCType + "> &)" + indices + ", " +
                     mask + ");");
    }
    return "";
}

static std::string EmitVFStoreUnAlign(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2)
        << "vf.store_unalign requires at least 2 args (dst, src)";
    // MaskReg src path: when args[1] is a MaskReg, dispatch to pstu
    bool srcIsMask = false;
    if (auto srcV = ir::As<ir::Var>(op->args_[1])) {
        srcIsMask = ir::IsMaskReg(srcV->GetType());
    }
    if (srcIsMask) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
            << "vf.store_unalign mask path requires 3 args (ptr, mask, ureg)";
        // DataCopyUnAlign: only supports b16/b32 (SupportBytes<T, 2, 4>)
        DataType tileDt = GetExprDtype(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tileDt.GetBit() == 16 || tileDt.GetBit() == 32)
            << "vf.store_unalign (mask path) only supports b16/b32 tile types, got " << DTypeStr(tileDt);
        std::string vreg = codegen.GetExprAsCode(op->args_[1]);
        DataType maskDt = GetExprDtype(op->args_[1], DataType::UINT16);
        int elemBytes = static_cast<int>(maskDt.GetBit() / 8);
        if (elemBytes <= 0)
            elemBytes = 4;
        // pstu only accepts uint16_t* or uint32_t* (DataCopyUnAlignImpl
        // casts to unsigned regardless of template T). b16→uint16_t, b32→uint32_t.
        std::string ptrType = (elemBytes <= 2) ? "uint16_t" : "uint32_t";
        // pstu modifies the pointer in-place (*&), so use post-update ref.
        // AscendC signature: pstu(ureg, mask, (__ubuf__ uint32_t*&)dstAddr)
        // The & in the cast is required so pstu advances the cursor, otherwise
        // vstar in store_unalign_post would overwrite pstu's output at the same address.
        std::string tilePtr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        std::string ureg = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("pstu(" + ureg + ", " + vreg + ", (__ubuf__ " + ptrType + " *&)" + tilePtr + ");");
        return "";
    }
    // 4-arg form: vstus(ureg, stride, vreg, dst, POST_UPDATE) or
    //              vstu(ureg, areg, vreg, dst, POST_UPDATE|NORM) when args[3] is an AddrReg.
    // Strideless vstur mode is in vf.squeeze_store_unalign.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.store_unalign requires 4 args (dst, vreg, ureg, stride|areg); "
        << "use vf.squeeze_store_unalign for strideless (vstur) mode";
    DataType srcDt = GetExprDtype(op->args_[1]);
    // vstus/vstu support b8/b16/b32/b64 element widths. b8 covers the 4-bit
    // FP4 types (packed 2-per-byte), mirroring the load_unalign side and the
    // DataCopyUnAlignImpl b8->uint8_t cast rule.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(srcDt) || srcDt.GetBit() == 16 || srcDt.GetBit() == 32 || srcDt.GetBit() == 64)
        << "vf.store_unalign source only supports b8/b16/b32/b64 types, got " << DTypeStr(srcDt);
    auto uregVar = ir::As<ir::Var>(op->args_[2]);
    if (uregVar) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsUnalignReg(uregVar->GetType()))
            << "vf.store_unalign requires the ureg argument to be an UnalignReg (from vf.unalign_reg_for_store)";
    }
    // AscendC DataCopyUnAlignImpl cast rules (dav_3510 store_impl.h):
    //   b8  → uint8_t
    //   b16 → no cast (native type)
    //   b32 → int32_t
    //   b64 → int32_t (RegTraitNumOne: simulated as b32 with stride×2 for vstus;
    //                  cast to int32_t for vstu — no stride scaling)
    bool isB64 = (srcDt.GetBit() == 64);
    DataType castDt = srcDt;
    if (IsB8Type(srcDt)) {
        castDt = DataType::UINT8;
    } else if (srcDt.GetBit() == 32 || isB64) {
        castDt = DataType::INT32;
    }
    std::string baseCType = castDt.ToCTypeString();
    std::string tilePtrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string vreg = codegen.GetExprAsCode(op->args_[1]);
    std::string ureg = codegen.GetExprAsCode(op->args_[2]);
    std::string vregExpr = (castDt == srcDt) ? vreg : ("(RegTensor<" + baseCType + "> &)" + vreg);
    std::string fourthArg = codegen.GetExprAsCode(op->args_[3]);
    std::string ptrCast = "(__ubuf__ " + baseCType + " *&)";
    if (ir::IsAddrReg(op->args_[3]->GetType())) {
        // vstu(ureg, areg, vreg, dst, POST_UPDATE|NORM) — AddrReg-based unaligned store.
        // AscendC DataCopyUnAlignImpl passes postValue to vstu's 5th arg, so both
        // post modes are valid for the AddrReg form.
        bool postUpdate = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
        std::string pu = postUpdate ? "POST_UPDATE" : "NORM";
        codegen.Emit("vstu(" + ureg + ", " + fourthArg + ", " + vregExpr + ", " + ptrCast + tilePtrVar + ", " + pu +
                     ");");
    } else {
        // vstus(ureg, stride, vreg, dst, POST_UPDATE)
        // Stride requires post_update=True — the stride operand is postUpdateStride in
        // AscendC DataCopyUnAlignImpl, so a bare stride without advancing is meaningless.
        bool postUpdate = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, postUpdate)
            << "vf.store_unalign: stride is only supported with post_update=True. For a non-advancing store, "
            << "use vf.squeeze_store_unalign";
        // b64: stride must be doubled to account for b32 simulation of b64 elements.
        std::string stride = isB64 ? ("(" + fourthArg + ") * 2") : fourthArg;
        codegen.Emit("vstus(" + ureg + ", " + stride + ", " + vregExpr + ", " + ptrCast + tilePtrVar +
                     ", POST_UPDATE);");
    }
    return "";
}

static std::string EmitVFStoreUnAlignPost(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 3)
        << "vf.store_unalign_post requires 3 args (dst, ureg, stride|areg); "
        << "use vf.squeeze_store_unalign_post for strideless (vstar) mode";
    DataType tileDt = GetExprDtype(op->args_[0]);
    auto uregVar = ir::As<ir::Var>(op->args_[1]);
    if (uregVar) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsUnalignReg(uregVar->GetType()))
            << "vf.store_unalign_post requires the ureg argument to be an UnalignReg (from vf.unalign_reg_for_store)";
    }
    // AscendC DataCopyUnAlignPostImpl cast rules (dav_3510 store_impl.h):
    //   b8  → uint8_t
    //   b16 → no cast (native type)
    //   b32 → int32_t
    //   b64 → int32_t (simulated as b32 with stride×2 for vstas; cast to
    //                  int32_t for vsta — no stride scaling)
    bool isB64 = (tileDt.GetBit() == 64);
    DataType castDt = tileDt;
    if (IsB8Type(tileDt)) {
        castDt = DataType::UINT8;
    } else if (tileDt.GetBit() == 32 || isB64) {
        castDt = DataType::INT32;
    }
    std::string baseCType = castDt.ToCTypeString();
    std::string tilePtrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    std::string thirdArg = codegen.GetExprAsCode(op->args_[2]);
    bool postUpdate = op->HasKwarg("post_update") && op->GetKwarg<bool>("post_update");
    std::string ptrCast = "(__ubuf__ " + baseCType + " *&)";
    if (ir::IsAddrReg(op->args_[2]->GetType())) {
        // vsta(ureg, dst, areg) — AddrReg-based unaligned store post.
        // vsta has no post mode; post_update kwarg is not applicable in this mode.
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !op->HasKwarg("post_update"))
            << "vf.store_unalign_post (AddrReg mode) does not support post_update; "
            << "vsta has no post mode parameter";
        codegen.Emit("vsta(" + ureg + ", " + ptrCast + tilePtrVar + ", " + thirdArg + ");");
        return "";
    }
    // vstas always requires __ubuf__ T*& (lvalue ref).
    // POST_UPDATE: 4 args (ureg, ptr, stride, POST_UPDATE)
    // NORM: 3 args (ureg, ptr, stride) — no Post argument
    // b64: stride must be doubled to account for b32 simulation of b64 elements.
    std::string stride = isB64 ? ("(" + thirdArg + ") * 2") : thirdArg;
    if (postUpdate) {
        codegen.Emit("vstas(" + ureg + ", " + ptrCast + tilePtrVar + ", " + stride + ", POST_UPDATE);");
    } else {
        codegen.Emit("vstas(" + ureg + ", " + ptrCast + tilePtrVar + ", " + stride + ");");
    }
    return "";
}

// ============================================================================
// SqueezeStoreUnAlign — vstur (strideless unaligned store, reads AR register
// for byte count). Must be paired with vf.squeeze(STORE_REG) and
// vf.squeeze_store_unalign_post. The AR register (written by squeeze) provides
// the valid byte count; vstur uses it as the implicit stride.
// ============================================================================
static std::string EmitVFSqueezeStoreUnAlign(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.squeeze_store_unalign requires 3 args (dst, src, align_reg)";
    // Squeeze path does not support mask_reg src — use vf.store_unalign (pstu) for mask store.
    if (auto srcV = ir::As<ir::Var>(op->args_[1])) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, !ir::IsMaskReg(srcV->GetType()))
            << "vf.squeeze_store_unalign does not support MaskReg src; use vf.store_unalign instead";
    }
    auto uregVar = ir::As<ir::Var>(op->args_[2]);
    if (uregVar) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsUnalignReg(uregVar->GetType()))
            << "vf.squeeze_store_unalign requires the align_reg argument to be an UnalignReg"
            << " (from vf.unalign_reg_for_store)";
    }
    DataType srcDt = GetExprDtype(op->args_[1]);
    // vstur supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      srcDt.GetBit() == 8 || srcDt.GetBit() == 16 || srcDt.GetBit() == 32 || srcDt.GetBit() == 64)
        << "vf.squeeze_store_unalign source only supports b8/b16/b32/b64 types, got " << DTypeStr(srcDt);
    // AscendC DataCopyUnAlignImpl cast rules (dav_3510 store_impl.h:496-518):
    //   b8  → uint8_t (line 506)
    //   b16 → no cast  (line 515, else branch uses original type)
    //   b32 → int32_t  (line 511)
    //   b64 → int64_t  (line 513)
    DataType castDt = srcDt;
    if (srcDt.GetBit() == 32) {
        castDt = DataType::INT32;
    } else if (srcDt.GetBit() == 64) {
        castDt = DataType::INT64;
    } else if (IsB8Type(srcDt)) {
        castDt = DataType::UINT8;
    }
    std::string baseCType = castDt.ToCTypeString();
    std::string tilePtrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string vreg = codegen.GetExprAsCode(op->args_[1]);
    std::string ureg = codegen.GetExprAsCode(op->args_[2]);
    std::string vregExpr = (castDt == srcDt) ? vreg : ("(RegTensor<" + baseCType + "> &)" + vreg);
    // vstur takes __ubuf__ T* (not *&)
    std::string ptrCast = "(__ubuf__ " + baseCType + " *)";
    codegen.Emit("vstur(" + ureg + ", " + vregExpr + ", " + ptrCast + tilePtrVar + ", POST_UPDATE);");
    return "";
}

// ============================================================================
// SqueezeStoreUnAlignPost — vstar (strideless unaligned store post, reads AR
// register for remaining byte count). Must be paired with
// vf.squeeze_store_unalign.
// ============================================================================
static std::string EmitVFSqueezeStoreUnAlignPost(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.squeeze_store_unalign_post requires 2 args (dst, align_reg)";
    DataType tileDt = GetExprDtype(op->args_[0]);
    // vstar supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(tileDt) || tileDt.GetBit() == 16 || tileDt.GetBit() == 32 || tileDt.GetBit() == 64)
        << "vf.squeeze_store_unalign_post only supports b8/b16/b32/b64 types, got " << DTypeStr(tileDt);
    auto uregVar = ir::As<ir::Var>(op->args_[1]);
    if (uregVar) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsUnalignReg(uregVar->GetType()))
            << "vf.squeeze_store_unalign_post requires the align_reg argument to be an UnalignReg"
            << " (from vf.unalign_reg_for_store)";
    }
    // AscendC DataCopyUnAlignPostImpl cast rules (dav_3510 store_impl.h:520-537):
    //   b8  → uint8_t (line 525)
    //   b16 → no cast  (line 534, else branch uses original type)
    //   b32 → int32_t  (line 530)
    //   b64 → int64_t  (line 532)
    DataType castDt = tileDt;
    if (tileDt.GetBit() == 32) {
        castDt = DataType::INT32;
    } else if (tileDt.GetBit() == 64) {
        castDt = DataType::INT64;
    } else if (IsB8Type(tileDt)) {
        castDt = DataType::UINT8;
    }
    std::string baseCType = castDt.ToCTypeString();
    std::string tilePtrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    // vstar takes __ubuf__ T* (not *&)
    std::string ptrCast = "(__ubuf__ " + baseCType + " *)";
    codegen.Emit("vstar(" + ureg + ", " + ptrCast + tilePtrVar + ");");
    return "";
}

static std::string EmitVFUnalignRegForStore(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    std::string regName = codegen.GetCurrentResultTarget();
    codegen.Emit("UnalignReg " + regName + ";");
    return "";
}

static std::string EmitVFClearSpr(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    codegen.Emit("sprclr(SPR_AR);");
    return "";
}

// ============================================================================
// UnalignRegForLoad — declare unaligned load register
// ============================================================================

static std::string EmitVFUnalignRegForLoad(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    std::string regName = codegen.GetCurrentResultTarget();
    codegen.Emit("UnalignReg " + regName + ";");
    return "";
}

// ============================================================================
// LoadUnalignPre — vldas (setup unaligned load)
// ============================================================================

static std::string EmitVFLoadUnalignPre(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.load_unalign_pre requires 2 args (ureg, src_ptr)";
    std::string ureg = codegen.GetExprAsCode(op->args_[0]);
    auto uregVar = ir::As<ir::Var>(op->args_[0]);
    if (uregVar) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsUnalignReg(uregVar->GetType()))
            << "vf.load_unalign_pre requires the first argument to be an UnalignReg (from vf.load_unalign_init), got "
            << ureg;
    }
    DataType dt = GetExprDtype(op->args_[1], DataType::FP32);
    // vldas supports b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dt) || dt.GetBit() == 16 || dt.GetBit() == 32 || dt.GetBit() == 64)
        << "vf.load_unalign_pre only supports b8/b16/b32/b64 types, got " << DTypeStr(dt);
    int elemBytes = static_cast<int>(dt.GetBit() / 8);
    if (elemBytes <= 0)
        elemBytes = 4;
    std::string ptrType;
    if (elemBytes == 1) {
        ptrType = "uint8_t";
    } else if (elemBytes == 8) {
        ptrType = "uint32_t";
    } else if (elemBytes == 4) {
        ptrType = "int32_t";
    } else {
        if (dt == DataType::FP16 || dt == DataType::BF16)
            ptrType = "half";
        else
            ptrType = "uint16_t";
    }
    std::string srcPtr = GetUBufPtr(codegen, op->args_[1], ptrType);
    codegen.Emit("vldas(" + ureg + ", " + srcPtr + ");");
    return "";
}

// ============================================================================
// LoadUnalign — vldus (unaligned load body, supports 3-arg and 4-arg strided)
// ============================================================================

static std::string EmitVFLoadUnalign(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, ureg, src] without stride,
    // [dst, ureg, src, stride] with stride (positional stride selects
    // POST_UPDATE). vldus has no "stride without advancing" mode (AscendC
    // DataCopyUnAlignImpl static_asserts POST_MODE_UPDATE for its strided
    // overload), so every other combination is rejected instead of silently
    // stepping the pointer.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3 || op->args_.size() == 4)
        << "vf.load_unalign requires 3-4 args (dst, ureg, src_ptr [, stride]), got " << op->args_.size();
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string ureg = codegen.GetExprAsCode(op->args_[1]);
    auto uregVar = ir::As<ir::Var>(op->args_[1]);
    if (uregVar) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::IsUnalignReg(uregVar->GetType()))
            << "vf.load_unalign requires the ureg argument to be an UnalignReg (from vf.load_unalign_init), got "
            << ureg;
    }
    DataType dstDt = GetExprDtype(op->args_[2]);
    if (const auto& t = op->args_[0]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(dstDt) || dstDt.GetBit() == 16 || dstDt.GetBit() == 32 || dstDt.GetBit() == 64))
        << "vf.load_unalign only supports b8/b16/b32/b64 types, got " << DTypeStr(dstDt);
    // vldus supports b8/b16/b32/b64 element widths
    std::string ptrType = dstDt.ToCTypeString();
    if (op->args_.size() == 4) {
        // Passing the stride argument selects POST_UPDATE — vldus has no
        // "stride without advancing" mode.
        std::string stride = codegen.GetExprAsCode(op->args_[3]);
        std::string srcPtr = codegen.GetOrCreateVFTilePtr(op->args_[2], /*is_post_update=*/true);
        int elemBytes = static_cast<int>(dstDt.GetBit() / 8);
        if (elemBytes <= 0)
            elemBytes = 4;
        bool isB64 = (elemBytes == 8);
        // AscendC cast rules (dav_m510 load_impl.h DataCopyUnAlignImpl):
        //   vldus (stride): b64 is simulated as uint32_t with stride*2, else native
        std::string effPtrType = isB64 ? "uint32_t" : ptrType;
        std::string dstExpr = isB64 ? ("(RegTensor<uint32_t>&)" + dst) : dst;
        if (effPtrType != "float") {
            srcPtr = "(__ubuf__ " + effPtrType + " *&)" + srcPtr;
        }
        std::string effectiveStride = isB64 ? ("(" + stride + ") * 2") : stride;
        codegen.Emit("vldus(" + dstExpr + ", " + ureg + ", " + srcPtr + ", " + effectiveStride + ", POST_UPDATE);");
        return "";
    }
    // Stride-less form: non-advancing vldus.
    std::string srcPtr = GetUBufPtr(codegen, op->args_[2], ptrType);
    codegen.Emit("vldus(" + dst + ", " + ureg + ", " + srcPtr + ");");
    return "";
}

// ============================================================================
// Scatter — vscatter (scatter store by indices)
// ============================================================================

static std::string EmitVFScatter(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.scatter requires 4 args (base_ptr, src, index, mask)";
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string index = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    DataType srcDt = GetExprDtype(op->args_[1]);
    DataType idxDt = GetExprDtype(op->args_[2]);
    // Doc: src supports DT_INT8,DT_UINT8,DT_INT16,DT_UINT16,DT_FP16,DT_BF16,
    // DT_INT32,DT_UINT32,DT_FP32,DT_INT64,DT_UINT64
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      srcDt == DataType::INT8 || srcDt == DataType::UINT8 || srcDt == DataType::BOOL ||
                          srcDt == DataType::INT16 || srcDt == DataType::UINT16 || srcDt == DataType::FP16 ||
                          srcDt == DataType::BF16 || srcDt == DataType::INT32 || srcDt == DataType::UINT32 ||
                          srcDt == DataType::FP32 || srcDt == DataType::INT64 || srcDt == DataType::UINT64)
        << "vf.scatter only supports INT8/UINT8/INT16/UINT16/FP16/BF16/INT32/UINT32/FP32/INT64/UINT64, got "
        << DTypeStr(srcDt);
    if (srcDt.GetBit() == 8 || srcDt.GetBit() == 16) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT16)
            << "vf.scatter b8/b16 src requires UINT16 index, got " << DTypeStr(idxDt);
    } else if (srcDt.GetBit() == 32) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT32)
            << "vf.scatter b32 src requires UINT32 index, got " << DTypeStr(idxDt);
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, idxDt == DataType::UINT32 || idxDt == DataType::UINT64)
            << "vf.scatter b64 src requires UINT32/UINT64 index, got " << DTypeStr(idxDt);
    }
    std::string baseCType = srcDt.ToCTypeString();
    std::string basePtr = GetUBufPtr(codegen, op->args_[0], baseCType);
    if (srcDt.GetBit() == 64) {
        // B64 scatter: vscatter has no b64 single-register overload.
        // Mirrors AscendC DataCopyScatterB64Impl: pack mask, compute odd/even
        // b32 indices (index*2, index*2+1), deinterleave src into b32 halves,
        // vscatter each half.
        std::string p = src + "_scatter_";
        std::string packedM = p + "_pm_";
        std::string vl32M = p + "_vl32_";
        std::string allM = p + "_allm_";
        std::string andM = p + "_andm_";
        std::string oddIdx = p + "_oi_";
        std::string evenIdx = p + "_ei_";
        std::string loReg = p + "_lor_";
        std::string hiReg = p + "_hir_";
        std::string dump = p + "_dump_";
        codegen.Emit("MaskReg " + packedM + ";");
        codegen.Emit("ppack(" + packedM + ", " + mask + ", LOWER);");
        codegen.Emit("MaskReg " + vl32M + " = pset_b32(PAT_VL32);");
        codegen.Emit("MaskReg " + allM + " = pset_b32(PAT_ALL);");
        codegen.Emit("MaskReg " + andM + ";");
        codegen.Emit("pand(" + andM + ", " + packedM + ", " + vl32M + ", " + allM + ");");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, oddIdx);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, evenIdx);
        codegen.Emit("vmuls(" + oddIdx + ", (RegTensor<uint32_t> &)" + index + ", (uint32_t)2, " + andM +
                     ", MODE_ZEROING);");
        codegen.Emit("vadds(" + evenIdx + ", " + oddIdx + ", (uint32_t)1, " + andM + ", MODE_ZEROING);");
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, loReg);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, hiReg);
        EmitRtWrapDecl(codegen, ir::DataType::UINT32, dump);
        codegen.Emit("vdintlv(" + loReg + ", " + hiReg + ", (RegTensor<uint32_t>&)" + src + ", (RegTensor<uint32_t>&)" +
                     src + ");");
        std::string ubPtr = GetUBufPtr(codegen, op->args_[0], "uint32_t");
        codegen.Emit("vscatter(" + loReg + ", " + ubPtr + ", " + oddIdx + ", " + andM + ");");
        codegen.Emit("vscatter(" + hiReg + ", " + ubPtr + ", " + evenIdx + ", " + andM + ");");
        return "";
    }
    std::string idxCType = (srcDt.GetBit() >= 32) ? "uint32_t" : "uint16_t";
    codegen.Emit("vscatter(" + src + ", " + basePtr + ", (RegTensor<" + idxCType + "> &)" + index + ", " + mask + ");");
    return "";
}

// ============================================================================
// Unsqueeze — vusqz (expand mask bits into register)
// ============================================================================

static std::string EmitVFUnsqueeze(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, mask]. The dst kind
    // mirrors the source (MaskReg src keeps the mask dialect).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.unsqueeze requires 2 args (dst, mask)";
    // PrefixSum (vusqz): int8/uint8/int16/uint16/int32/uint32 only
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string mask = codegen.GetExprAsCode(op->args_[1]);
    DataType dstDt = GetExprDtype(op->args_[1]);
    if (const auto& t = op->args_[1]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (dstDt == DataType::INT8 || dstDt == DataType::UINT8 || dstDt == DataType::INT16 ||
                       dstDt == DataType::UINT16 || dstDt == DataType::INT32 || dstDt == DataType::UINT32))
        << "vf.unsqueeze dst only supports INT8/UINT8/INT16/UINT16/INT32/UINT32, got " << DTypeStr(dstDt);
    codegen.Emit("vusqz(" + dst + ", " + mask + ");");
    return "";
}

// ============================================================================
// Truncate — vtrc with ROUND_Z (alias for Round with round_mode=TRUNC)
// ============================================================================

static std::string EmitVFTruncate(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.truncate requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      srcDt == DataType::FP16 || srcDt == DataType::BF16 || srcDt == DataType::FP32)
        << "vf.truncate src only supports FP16/BF16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    // Default resolves like AscendC TruncateImpl: an unset round_mode
    // (CAST_NONE) maps to CAST_RINT → ROUND_R.
    std::string roundConst = "ROUND_R";
    if (op->HasKwarg("round_mode")) {
        auto rm = static_cast<ir::VFRoundMode>(op->GetKwarg<int>("round_mode"));
        if (rm == ir::VFRoundMode::CAST_RINT)
            roundConst = "ROUND_R";
        else if (rm == ir::VFRoundMode::CAST_ROUND)
            roundConst = "ROUND_A";
        else if (rm == ir::VFRoundMode::CAST_FLOOR)
            roundConst = "ROUND_F";
        else if (rm == ir::VFRoundMode::CAST_CEIL)
            roundConst = "ROUND_C";
        else if (rm == ir::VFRoundMode::CAST_TRUNC)
            roundConst = "ROUND_Z";
        else if (rm == ir::VFRoundMode::CAST_ODD || rm == ir::VFRoundMode::CAST_HYBRID)
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false)
                << "vf.truncate does not support round_mode CAST_ODD/CAST_HYBRID, supported values are "
                   "CAST_RINT/CAST_ROUND/CAST_FLOOR/CAST_CEIL/CAST_TRUNC";
        else
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
                << "vf.truncate got unknown round_mode " << VFEnumValueName(ir::EnumToString(rm));
    }
    std::string mode = VFZeroingOnly(op, "vf.truncate");
    codegen.Emit("vtrc(" + dst + ", " + src + ", " + roundConst + ", " + mask + ", " + mode + ");");
    return "";
}

// ============================================================================
// MaskGenWithRegTensor — movvp (generate MaskReg from RegTensor bit offset)
// ============================================================================

static std::string EmitVFMaskGenWithRegTensor(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.mask_gen_with_reg_tensor requires 2 args (dst, src)";
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string maskDst = codegen.GetExprAsCode(op->args_[0]);
    std::string offset = "0";
    if (op->HasKwarg("offset")) {
        offset = std::to_string(op->GetKwarg<int>("offset"));
    }
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt.GetBit() == 16 || srcDt.GetBit() == 32)
        << "vf.mask_gen_with_reg_tensor source only supports b16/b32 types, got " << DTypeStr(srcDt);
    // offset must be 0~15 for b16, 0~31 for b32
    int offsetVal = 0;
    if (op->HasKwarg("offset")) {
        offsetVal = op->GetKwarg<int>("offset");
    }
    if (srcDt.GetBit() == 16) {
        PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, offsetVal >= 0 && offsetVal <= 15)
            << "vf.mask_gen_with_reg_tensor offset must be 0~15 for b16, got " << offsetVal;
    } else {
        PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, offsetVal >= 0 && offsetVal <= 31)
            << "vf.mask_gen_with_reg_tensor offset must be 0~31 for b32, got " << offsetVal;
    }
    if (srcDt.GetBit() == 16) {
        codegen.Emit("movvp(" + maskDst + ", (RegTensor<uint16_t> &)" + src + ", " + offset + ");");
    } else {
        codegen.Emit("movvp(" + maskDst + ", (RegTensor<uint32_t> &)" + src + ", " + offset + ");");
    }
    return "";
}

// ============================================================================
// GetMaskSpr (unified) — movp_b32/movp_b16 with width kwarg
// Replaces: GetMaskSprB32, GetMaskSprB16
// ============================================================================

static std::string EmitVFGetMaskSpr(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "vf.get_mask_spr takes no source args (only the synthesized dst)";
    std::string regName = codegen.GetExprAsCode(op->args_[0]);
    std::string width = "B32";
    if (op->HasKwarg("width"))
        width = VFEnumValueName(ir::EnumToString(static_cast<ir::MaskWidth>(op->GetKwarg<int>("width"))));
    // MoveMask: only supports b16/b32 (SupportBytes<T, 2, 4>)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, width == "B16" || width == "B32")
        << "vf.get_mask_spr only supports B16/B32 width, got " << width;
    if (width == "B16")
        codegen.Emit(regName + " = movp_b16();");
    else
        codegen.Emit(regName + " = movp_b32();");
    return "";
}

// ============================================================================
// Registration
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "vf.create_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFCreateMask);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFDuplicate);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLoadAlign);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_align")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFStoreAlign);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMax);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAdd);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFSub);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.and_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAnd);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFXor);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.or_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFOr);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFReduceSum);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFReduceMax);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.reduce_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFReduceMin);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMul);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_add_dst")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMulAddDst);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_add_dst_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMulAddDst);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFDiv);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMuls);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ln")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLn);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLog);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMin);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFExp);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAbs);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.not_")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFNot);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFSqrt);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFRelu);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFNeg);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAdds);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMins);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMaxs);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.leaky_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLeakyRelu);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFInterleave);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.pair_reduce_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFPairReduceSum);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.abs_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAbsSub);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.axpy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAxpy);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.axpy_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAxpy);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_dst_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMulDstAdd);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mul_dst_add_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMulDstAdd);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.pack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFPack);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.unpack")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFUnpack);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.prelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFPRelu);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_left")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFShiftLeft);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.shift_right")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFShiftRight);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mull")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMull);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.addc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFAddc);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.subc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFSubc);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.exp_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFExpSub);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.astype")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFCast);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.de_interleave")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFDeInterleave);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.select")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFSelect);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.update_mask")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFUpdateMask);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mem_bar")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMemBar);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.histograms")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFHistograms);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.histogram_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFHistograms);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.eq")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFCompareImpl(o, c, "EQ"); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ne")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFCompareImpl(o, c, "NE"); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.lt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFCompareImpl(o, c, "LT"); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.gt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFCompareImpl(o, c, "GT"); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.le")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFCompareImpl(o, c, "LE"); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.ge")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFCompareImpl(o, c, "GE"); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFSqueeze);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.arange")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFArange);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.gather")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFGather);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFStoreUnAlign);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.store_unalign_post")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFStoreUnAlignPost(o, c); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze_store_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFSqueezeStoreUnAlign(o, c); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.squeeze_store_unalign_post")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) {
            return EmitVFSqueezeStoreUnAlignPost(o, c);
        });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.unalign_reg_for_store")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFUnalignRegForStore(o, c); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.clear_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFClearSpr);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_init")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(
            op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) { return EmitVFUnalignRegForLoad(o, c); });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign_pre")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLoadUnalignPre);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.load_unalign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLoadUnalign);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.scatter")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFScatter);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.unsqueeze")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFUnsqueeze);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.truncate")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFTruncate);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.mask_gen_with_reg_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, [](const ir::CallPtr& o, codegen::CodegenBase& c) {
            return EmitVFMaskGenWithRegTensor(o, c);
        });
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.get_mask_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFGetMaskSpr);
    });

// ============================================================================
// Log2 — composite: vln + vmuls(1/ln2) = ln(x) * 1.4426950408889634
// ============================================================================

static std::string EmitVFLog2(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.log2 requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.log2 src only supports FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.log2");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = srcDt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + dc + ";");
        if (srcDt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 1.4426950408889634f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -10.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 1.4426950408889634f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -23.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
    } else {
        codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmuls(" + dst + ", " + dst + ", 1.4426950408889634f, " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// Log10 — composite: vln + vmuls(1/ln10) = ln(x) * 0.4342944819032518
// ============================================================================

static std::string EmitVFLog10(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "vf.log10 requires 3 args (dst, src, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, (srcDt == DataType::FP16 || srcDt == DataType::FP32))
        << "vf.log10 src only supports FP16/FP32, got " << DTypeStr(srcDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string mask = codegen.GetExprAsCode(op->args_[2]);
    std::string mode = VFZeroingOnly(op, "vf.log10");
    if (op->HasKwarg("precision") && op->GetKwarg<bool>("precision")) {
        std::string mc = dst + "_mcmp_";
        std::string tp = dst + "_tmp_";
        std::string sc = dst + "_srcc_";
        std::string dc = dst + "_dstc_";
        std::string ctype = srcDt.ToCTypeString();
        codegen.Emit("MaskReg " + mc + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + tp + ";");
        codegen.Emit("RegTensor<" + ctype + "> " + sc + " = (RegTensor<" + ctype + ">&)" + src + ";");
        codegen.Emit("RtWrap<" + ctype + "> " + dc + ";");
        if (srcDt == DataType::FP16) {
            codegen.Emit("union { uint16_t i; half f; } " + dst + "_thr_ = {0x03FF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 1024.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 0.43429448190325176f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -3.01029995663981f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        } else {
            codegen.Emit("union { uint32_t i; float f; } " + dst + "_thr_ = {0x007FFFFF};");
            codegen.Emit("vcmps_lt(" + mc + ", " + sc + ", " + dst + "_thr_.f, " + mask + ");");
            codegen.Emit("vmuls(" + tp + ", " + sc + ", 8388608.0f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + sc + ", " + tp + ", " + sc + ", " + mc + ");");
            codegen.Emit("vln(" + dc + ", " + sc + ", " + mask + ", " + mode + ");");
            codegen.Emit("vmuls(" + dc + ", " + dc + ", 0.43429448190325176f, " + mask + ", " + mode + ");");
            codegen.Emit("vadds(" + tp + ", " + dc + ", -6.923689900271567f, " + mask + ", " + mode + ");");
            codegen.Emit("vsel(" + dst + ", " + tp + ", " + dc + ", " + mc + ");");
        }
    } else {
        codegen.Emit("vln(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
        codegen.Emit("vmuls(" + dst + ", " + dst + ", 0.4342944819032518f, " + mask + ", " + mode + ");");
    }
    return "";
}

// ============================================================================
// MulsCast — vmulscvt (fused multiply-scalar-cast: dst(fp16) = cast(src(fp32) * scalar))
// ============================================================================

static std::string EmitVFMulsCast(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src, scalar, mask].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "vf.muls_cast requires 4 args (dst, src, scalar, mask)";
    DataType srcDt = GetExprDtype(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcDt == DataType::FP32)
        << "vf.muls_cast source only supports FP32, got " << DTypeStr(srcDt);
    DataType dstDt = op->HasKwarg("dtype") ? op->GetKwarg<DataType>("dtype") : GetExprDtype(op->args_[0], srcDt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, dstDt == DataType::FP16)
        << "vf.muls_cast result only supports DT_FP16, got " << DTypeStr(dstDt);
    // FusedMulsCast: scalar must be float (Tuple<half, float, float>)
    DataType scalarDt = GetExprDtype(op->args_[2]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalarDt == DataType::FP32)
        << "vf.muls_cast scalar only supports FP32, got " << DTypeStr(scalarDt);
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    std::string scalarStr = codegen.GetExprAsCode(op->args_[2]);
    std::string mask = codegen.GetExprAsCode(op->args_[3]);
    // layout kwarg selects the result half: ZERO -> PART_EVEN (default), ONE -> PART_ODD.
    // FusedMulsCast: only supports RegLayout ZERO/ONE
    std::string part = "PART_EVEN";
    if (op->HasKwarg("layout")) {
        auto layout = VFEnumValueName(ir::EnumToString(static_cast<ir::CastLayout>(op->GetKwarg<int>("layout"))));
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, layout == "ZERO" || layout == "ONE")
            << "vf.muls_cast only supports layout ZERO/ONE, got " << layout;
        if (layout == "ONE")
            part = "PART_ODD";
    }
    codegen.Emit("vmulscvt(" + dst + ", " + src + ", " + scalarStr + ", " + mask + ", " + part + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.log2")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLog2);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.log10")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLog10);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.muls_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMulsCast);
    });

// ============================================================================
// Load (unified) — vldas + vldus all-in-one unaligned load
// ============================================================================

static std::string EmitVFLoad(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src_ptr, (stride)].
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2 && op->args_.size() <= 3)
        << "vf.load requires 2-3 args (dst, src_ptr[, stride])";
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    DataType dstDt = GetExprDtype(op->args_[1]);
    if (const auto& t = op->args_[1]->GetType(); t != nullptr) {
        dstDt = GetTypeDtype(t);
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      (IsB8Type(dstDt) || dstDt.GetBit() == 16 || dstDt.GetBit() == 32 || dstDt.GetBit() == 64))
        << "vf.load only supports b8/b16/b32/b64 types, got " << DTypeStr(dstDt);
    // vldas/vldus support b8/b16/b32/b64 element widths
    int elemBytes = static_cast<int>(dstDt.GetBit() / 8);
    if (elemBytes <= 0)
        elemBytes = 4;
    // vldas and vldus both use the native dtype pointer (matches asc_load C-API).
    // For b8 types that lack a direct vldas/vldus overload, reinterpret as uint8_t.
    std::string ptrType = NeedsB8Reinterpret(dstDt) ? "uint8_t" : dstDt.ToCTypeString();
    static int loadCounter = 0;
    std::string uregName = "__ureg_ld_" + std::to_string(loadCounter++);
    codegen.Emit("UnalignReg " + uregName + ";");
    if (op->args_.size() == 3) {
        std::string srcPtr = GetUBufPtr(codegen, op->args_[1], ptrType, /*is_post_update=*/true);
        std::string stride = codegen.GetExprAsCode(op->args_[2]);
        codegen.Emit("vldas(" + uregName + ", " + srcPtr + ");");
        codegen.Emit("vldus(" + dst + ", " + uregName + ", " + srcPtr + ", " + stride + ", POST_UPDATE);");
    } else {
        std::string srcPtr = GetUBufPtr(codegen, op->args_[1], ptrType);
        codegen.Emit("vldas(" + uregName + ", " + srcPtr + ");");
        codegen.Emit("vldus(" + dst + ", " + uregName + ", " + srcPtr + ");");
    }
    return "";
}

// ============================================================================
// Store (unified) — vstus + vstas all-in-one unaligned store
// ============================================================================

static std::string EmitVFStore(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 2 && op->args_.size() <= 3)
        << "vf.store requires 2-3 args (dst_ptr, src[, count])";
    DataType srcDt = GetExprDtype(op->args_[1]);
    // vstus/vstas support b8/b16/b32/b64 element widths
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(srcDt) || srcDt.GetBit() == 16 || srcDt.GetBit() == 32 || srcDt.GetBit() == 64)
        << "vf.store only supports b8/b16/b32/b64 types, got " << DTypeStr(srcDt);
    // Check src/dst dtype consistency (doc: src and dst must have the same dtype)
    DataType tileDt = GetExprDtype(op->args_[0], srcDt);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tileDt == srcDt)
        << "vf.store requires src and dst to have the same dtype, got src=" << DTypeStr(srcDt)
        << " dst=" << DTypeStr(tileDt);
    int elemBytes = static_cast<int>(srcDt.GetBit() / 8);
    if (elemBytes <= 0)
        elemBytes = 4;
    // Determine count (positional arg, default 256B/elem_bytes)
    std::string count;
    int maxCount = 256 / elemBytes;
    if (op->args_.size() == 3) {
        count = codegen.GetExprAsCode(op->args_[2]);
        auto constInt = std::dynamic_pointer_cast<const ir::ConstInt>(op->args_[2]);
        if (constInt != nullptr) {
            PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, constInt->value_ <= maxCount)
                << "vf.store count must not exceed 256B/sizeof(dtype) = " << maxCount << ", got " << constInt->value_;
        }
    } else {
        count = std::to_string(maxCount);
    }
    std::string ptrType = NeedsB8Reinterpret(srcDt) ? "uint8_t" : srcDt.ToCTypeString();
    std::string srcExpr = codegen.GetExprAsCode(op->args_[1]);
    // vstus/vstas use POST_UPDATE so the pointer is advanced internally by vstus
    // and vstas flushes the tail at the advanced position. This matches AscendC
    // StoreImpl which always uses POST_MODE_UPDATE via DataCopyUnAlignImpl.
    std::string dstPtr = GetUBufPtr(codegen, op->args_[0], ptrType, /*is_post_update=*/true);
    static int storeCounter = 0;
    std::string uregName = "__ureg_st_" + std::to_string(storeCounter++);
    codegen.Emit("UnalignReg " + uregName + ";");
    // B64 single-register: vstus/vstas have no 8-byte-element overload; reinterpret
    // as uint32_t pairs and double the count (mirrors AscendC DataCopyUnAlignImpl
    // b64 path; bitwise-identical for both INT64 and UINT64).
    if (srcDt.GetBit() == 64) {
        std::string b32Count = "(" + count + ") * 2";
        std::string ptrVar = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/true);
        std::string b32Ptr = "(__ubuf__ uint32_t*&)" + ptrVar;
        codegen.Emit("vstus(" + uregName + ", " + b32Count + ", (RegTensor<uint32_t>&)" +
                     codegen.GetExprAsCode(op->args_[1]) + ", " + b32Ptr + ", POST_UPDATE);");
        codegen.Emit("vstas(" + uregName + ", " + b32Ptr + ", 0, POST_UPDATE);");
    } else {
        codegen.Emit("vstus(" + uregName + ", " + count + ", " + srcExpr + ", " + dstPtr + ", POST_UPDATE);");
        codegen.Emit("vstas(" + uregName + ", " + dstPtr + ", 0, POST_UPDATE);");
    }
    return "";
}

// EmitVFMaskLoad/Store/StoreUnalign have been removed — their logic is now
// unified into EmitVFLoadAlign/EmitVFStoreAlign/EmitVFStoreUnAlign via the
// dst's IR type (RegType MaskReg vs RegTensor kind), matching the
// function-overloading model.

REGISTER_BACKEND_OP(BackendCCE, "vf.load")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFLoad);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.store")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFStore);
    });

// mask_load/mask_store/mask_store_unalign backend registrations removed —
// the parser redirects these to vf.load_align/vf.store_align/vf.store_unalign
// which dispatch via the src/dst IR type (RegType MaskReg vs RegTensor kind).

// ============================================================================
// CreateAddrReg — AddrReg declaration + vag_b8/b16/b32 intrinsic
// ============================================================================

static std::string EmitVFCreateAddrReg(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 1 && op->args_.size() <= 4)
        << "vf.create_addr_reg requires 1-4 strides";
    std::string regName = codegen.GetCurrentResultTarget();
    // Determine element width from dtype kwarg (default b32)
    DataType dt = DataType::FP32;
    if (op->HasKwarg("dtype")) {
        dt = op->GetKwarg<DataType>("dtype");
    }
    // vag_b8/b16/b32 support b8/b16/b32/b64 element widths (b64 uses vag_b32 with doubled stride)
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                      IsB8Type(dt) || dt.GetBit() == 16 || dt.GetBit() == 32 || dt.GetBit() == 64)
        << "vf.create_addr_reg dtype must be b8/b16/b32/b64, got " << DTypeStr(dt);
    std::string vagFn;
    if (dt == DataType::UINT8 || dt == DataType::INT8)
        vagFn = "vag_b8";
    else if (dt.GetBit() == 16)
        vagFn = "vag_b16";
    else
        vagFn = "vag_b32";
    // Collect stride args. For b64, each stride is doubled.
    std::string strideArgs;
    for (size_t i = 0; i < op->args_.size(); ++i) {
        std::string stride = codegen.GetExprAsCode(op->args_[i]);
        if (dt == DataType::UINT64 || dt == DataType::INT64) {
            stride = "(" + stride + ") * 2";
        }
        if (!strideArgs.empty())
            strideArgs += ", ";
        strideArgs += stride;
    }
    // AddrReg (vector_address) must be declared and initialized in a single
    // statement (bisheng rejects a separate declaration + assignment). The
    // vag_* intrinsic is [aicore]-attributed and cannot be called directly
    // from a __simd_vf__ function (bisheng: "vag should be written in the
    // vf"). Route through a __simd_callee__ wrapper (_vag_bXX_N) provided by
    // the framework support header pypto_vf_support.h — same pattern as
    // RtWrap for RegTensor. The wrapper is always_inline, so bisheng inlines
    // it and sees vag_* at the call site, which must be inside a loop ("VAG
    // intrinsic can only be used in loop").
    // Emit inline (not hoisted) so the call site stays inside the user's loop.
    std::string wrapperFn = "_" + vagFn + "_" + std::to_string(op->args_.size());
    codegen.Emit("AddrReg " + regName + " = " + wrapperFn + "(" + strideArgs + ");");
    return "";
}

// ============================================================================
// Move — vmov (RegTensor) / pmov (MaskReg), with or without mask
// ============================================================================

static std::string EmitVFMove(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src] or [dst, src, mask]; a
    // masked rebinding carries the old target as an explicit argument
    // [dst, old, src, mask], because MERGING keeps the unselected lanes of the
    // previous destination value.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      op->args_.size() == 2 || op->args_.size() == 3 || op->args_.size() == 4)
        << op->name_ << " requires 2 args (dst, src), 3 args (dst, src, mask), or 4 (dst, old, src, mask)";
    size_t base = op->args_.size() == 4 ? 1 : 0;
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[base + 1]);
    // Mask-form dispatch follows the source: the dst kind mirrors src.
    bool isMaskDst = ir::IsMaskReg(op->args_[base + 1]->GetType());
    if (!isMaskDst) {
        DataType srcDt = GetExprDtype(op->args_[base + 1]);
        // AscendC Move supports bool + the standard int/float types; the FP8/FP4
        // family and 4-bit ints are not part of the Move contract.
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, srcDt == DataType::BOOL || IsArithIntType(srcDt) ||
                                                                    srcDt == DataType::FP16 ||
                                                                    srcDt == DataType::BF16 || srcDt == DataType::FP32)
            << op->name_
            << " src only supports BOOL/INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64/FP16/BF16/FP32, got "
            << DTypeStr(srcDt);
    }
    // Seed the fresh dst with the old value: the merge semantics below only
    // write the lanes selected by the mask.
    if (base == 1) {
        codegen.Emit(dst + " = " + codegen.GetExprAsCode(op->args_[1]) + ";");
    }
    if (op->args_.size() - base == 3) {
        std::string mask = codegen.GetExprAsCode(op->args_[base + 2]);
        if (isMaskDst) {
            codegen.Emit("pmov(" + dst + ", " + src + ", " + mask + ");");
        } else {
            // vf.move only supports MERGING mode (AscendC Copy/Move default).
            if (op->HasKwarg("mode")) {
                auto modeVal = static_cast<ir::MergeMode>(op->GetKwarg<int>("mode"));
                PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, modeVal == ir::MergeMode::MERGING)
                    << op->name_ << " only supports MERGING mode on current device";
            }
            std::string mode = "MODE_MERGING";
            DataType srcDt = GetExprDtype(op->args_[base + 1]);
            if (srcDt == DataType::BOOL) {
                codegen.Emit("vmov((RegTensor<int8_t>&)" + dst + ", (RegTensor<int8_t>&)" + src + ", " + mask + ", " +
                             mode + ");");
            } else if (srcDt.GetBit() == 64) {
                std::string p = dst + "_mov_";
                std::string tm = p + "tm_";
                std::string m0 = p + "m0_";
                std::string m1 = p + "m1_";
                codegen.Emit("MaskReg " + tm + ";");
                codegen.Emit("ppack(" + tm + ", " + mask + ", LOWER);");
                codegen.Emit("MaskReg " + m0 + ", " + m1 + ";");
                codegen.Emit("pintlv_b32(" + m0 + ", " + m1 + ", " + tm + ", " + tm + ");");
                codegen.Emit("vmov((RegTensor<uint32_t>&)" + dst + ", (RegTensor<uint32_t>&)" + src + ", " + m0 + ", " +
                             mode + ");");
            } else {
                codegen.Emit("vmov(" + dst + ", " + src + ", " + mask + ", " + mode + ");");
            }
        }
    } else {
        if (isMaskDst) {
            codegen.Emit("pmov(" + dst + ", " + src + ");");
        } else {
            codegen.Emit("vmov(" + dst + ", " + src + ");");
        }
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.create_addr_reg")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFCreateAddrReg);
    });

REGISTER_BACKEND_OP(BackendCCE, "vf.move")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMove);
    });

static std::string EmitVFMoveAcc(const ir::CallPtr& op, codegen::CodegenBase& codegen)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << op->name_ << " requires 4 args (dst, old, src, mask)";
    return EmitVFMove(op, codegen);
}

REGISTER_BACKEND_OP(BackendCCE, "vf.move_acc")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFMoveAcc);
    });

// ============================================================================
// BitCast — type reinterpretation (no instruction, just C++ reference cast)
// ============================================================================

static std::string EmitVFBitCast(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // Args order (dst prepended by codegen): [dst, src]. A plain C++
    // reinterpret assignment works because the dst is already declared.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "vf.bit_cast requires 2 args (dst, src)";
    DataType targetDt = op->GetKwarg<DataType>("dtype");
    std::string dst = codegen.GetExprAsCode(op->args_[0]);
    std::string src = codegen.GetExprAsCode(op->args_[1]);
    codegen.Emit(dst + " = (RegTensor<" + targetDt.ToCTypeString() + ">&)" + src + ";");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "vf.bit_cast")

    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return EmitVFValueForm(op, codegen, EmitVFBitCast);
    });

static std::string EmitSimdCall(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    std::ostringstream call;
    call << op->GetKwarg<std::string>("callee") << "(";
    for (size_t i = 0; i < op->args_.size(); ++i) {
        if (i > 0)
            call << ", ";
        const auto& actual = op->args_[i];
        if (auto tileType = ir::As<ir::TileType>(actual->GetType())) {
            // Project a MutexTile wrapper to its native Tile before extracting the
            // UB pointer (the same operand adaptation the SIMT launch uses).
            call << "(__ubuf__ " << tileType->dtype_.ToCTypeString() << " *)" << cce::GetOperandAsCode(codegen, actual)
                 << ".data()";
        } else {
            call << codegen.GetExprAsCode(actual);
        }
    }
    call << ");";
    codegen.Emit(call.str());
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "simd.call")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return EmitSimdCall(op, codegen); });

} // namespace backend
} // namespace pypto
