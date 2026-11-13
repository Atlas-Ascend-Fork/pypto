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
 * @file backend_cce_ops.cpp
 * \brief Backend op registration for BackendCCE
 *
 * This file registers all block operations for the CCE backend.
 * Each registration specifies the pipe type and CCE codegen function.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "backend/common/backend_utils.h"
#include "ir/op_attr_types.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/error.h"
#include "core/logging.h"
#include "interface/inner/dist_utils.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using ir::DataType;
using npu::tile_fwk::ExternalError;

// ============================================================================
// Helper Functions for CCE Code Generation
// ============================================================================

static int NextDebugDumpId()
{
    static int nextDebugDumpId = 0;
    return nextDebugDumpId++;
}

static std::string JoinExpressions(const std::vector<std::string>& expressions, const std::string& delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expressions.size(); ++i) {
        if (i > 0)
            oss << delimiter;
        oss << expressions[i];
    }
    return oss.str();
}

static void EmitDebugLocationHeaderCCE(codegen::CCECodegen& codegen, const ir::Span& span, const std::string& opName)
{
    std::string header = debug_printf::FormatDebugLocationHeader(span, opName);
    if (!header.empty()) {
        codegen.Emit("pypto::aicore::printf(\"" + debug_printf::EscapeStringLiteral(header + "\n") + "\");");
    }
}

static bool NeedsPtoPrintfSignedLongLong(const DataType& dtype, char conversion)
{
    return (conversion == 'd' || conversion == 'i') && (dtype == DataType::INT64 || dtype == DataType::INDEX);
}

static bool NeedsPtoPrintfUnsignedLongLong(const DataType& dtype, char conversion)
{
    return (conversion == 'u' || conversion == 'x') && (dtype == DataType::UINT64 || dtype == DataType::INDEX);
}

static std::string RewritePtoPrintfFormatForScalarType(const std::string& formatSegment, char conversion,
                                                       const DataType& dtype)
{
    const bool signedLongLong = NeedsPtoPrintfSignedLongLong(dtype, conversion);
    const bool unsignedLongLong = NeedsPtoPrintfUnsignedLongLong(dtype, conversion);
    if (!signedLongLong && !unsignedLongLong) {
        return formatSegment;
    }

    size_t convIdx = debug_printf::FindPrintfConversionIndex(formatSegment);
    std::string rewritten = formatSegment;
    if (conversion == 'd') {
        rewritten.replace(convIdx, 1, "lld");
    } else if (conversion == 'i') {
        rewritten.replace(convIdx, 1, "lli");
    } else if (conversion == 'u') {
        rewritten.replace(convIdx, 1, "llu");
    } else {
        rewritten.replace(convIdx, 1, "llx");
    }
    return rewritten;
}

static std::string CastPtoPrintfArgIfNeeded(const std::string& arg, const DataType& dtype, char conversion)
{
    if (conversion == 'f') {
        return arg;
    }
    if (NeedsPtoPrintfSignedLongLong(dtype, conversion)) {
        return "static_cast<long long>(" + arg + ")";
    }
    if (NeedsPtoPrintfUnsignedLongLong(dtype, conversion)) {
        return "static_cast<unsigned long long>(" + arg + ")";
    }
    if (dtype == DataType::BOOL) {
        return conversion == 'u' ? "static_cast<unsigned int>(" + arg + ")" : "static_cast<int>(" + arg + ")";
    }
    if (dtype == DataType::INT8 || dtype == DataType::INT16) {
        return "static_cast<int>(" + arg + ")";
    }
    if (dtype == DataType::UINT8 || dtype == DataType::UINT16) {
        return "static_cast<unsigned int>(" + arg + ")";
    }
    return arg;
}

static std::string BuildPtoPrintfCall(const std::string& format, const std::vector<std::string>& args,
                                      const std::vector<DataType>& argDtypes)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == argDtypes.size())
        << "debug.printf argument/type count mismatch";

    auto segments = debug_printf::ParsePrintfSegments(format);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, segments.size() == args.size())
        << "debug.printf format expects " << segments.size() << " scalar arguments, but got " << args.size();

    if (format.empty()) {
        return "";
    }

    std::string rewrittenFormat = segments.empty() ? format : "";
    std::vector<std::string> rewrittenArgs;
    for (size_t i = 0; i < segments.size(); ++i) {
        rewrittenFormat += RewritePtoPrintfFormatForScalarType(segments[i].formatSegment, segments[i].conversion,
                                                               argDtypes[i]);
        rewrittenArgs.push_back(CastPtoPrintfArgIfNeeded(args[i], argDtypes[i], segments[i].conversion));
    }

    std::string call = "pypto::aicore::printf(\"" + debug_printf::EscapeStringLiteral(rewrittenFormat) + "\"";
    for (const auto& arg : rewrittenArgs) {
        call += ", " + arg;
    }
    return call + ");";
}

static std::vector<std::string> BuildTensorDumpStride(codegen::CCECodegen& codegen, const ir::TensorTypePtr& tensorType)
{
    const size_t rank = tensorType->shape_.size();
    std::vector<std::string> stride(rank, "1");
    if (cce::IsNZTensorType(tensorType)) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, rank >= 2)
            << "debug.dump_tensor NZ lowering requires a tensor rank of at least 2";
        const size_t rowAxis = rank - 2;
        const int64_t c0 = cce::GetNZInnerCols(tensorType->dtype_);
        const std::string paddedRows = codegen.ComputeAlignedShapeDimension(tensorType->shape_[rowAxis], 16);
        const std::string paddedCols = codegen.ComputeAlignedShapeDimension(tensorType->shape_[rank - 1], c0);
        for (size_t axis = 0; axis < rowAxis; ++axis) {
            std::vector<std::string> factors;
            for (size_t inner = axis + 1; inner < rowAxis; ++inner) {
                factors.push_back(codegen.GetExprAsCode(tensorType->shape_[inner]));
            }
            factors.push_back(paddedRows);
            factors.push_back(paddedCols);
            stride[axis] = "(" + JoinExpressions(factors, " * ") + ")";
        }
        stride[rowAxis] = paddedRows;
        stride[rank - 1] = std::to_string(c0);
        return stride;
    }

    return codegen.BuildTensorStrideExpressions(tensorType);
}

static void EmitDumpMetadataCCE(codegen::CCECodegen& codegen, const ir::CallPtr& op, const std::string& opName)
{
    if (op->GetKwarg<bool>("show_location", false)) {
        EmitDebugLocationHeaderCCE(codegen, op->span_, opName);
    }
    std::string dumpFlag = op->HasKwarg("dump_flag") ? op->GetKwarg<std::string>("dump_flag") : "";
    if (dumpFlag.empty()) {
        return;
    }
    codegen.Emit("pypto::aicore::printf(\"[flag] %s\\n\", \"" + debug_printf::EscapeStringLiteral(dumpFlag) + "\");");
}

static std::string MakeDebugDumpTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 3)
        << "debug.dump_tensor requires 1 or 3 arguments, but got " << op->args_.size();
    auto tensorVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensorVar) << "debug.dump_tensor first argument must be a Var";
    auto tensorType = ir::As<ir::TensorType>(tensorVar->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensorType) << "debug.dump_tensor first argument must be TensorType";
    const int debugId = NextDebugDumpId();
    const std::string tensorName = codegen.GetVarName(tensorVar);
    std::string basePtr = codegen.GetPointer(tensorName);
    if (basePtr.empty()) {
        basePtr = tensorName + ".data()";
    }
    const std::string metadataPrefix = "__debug_dump_tensor_" + std::to_string(debugId);
    const std::string sourceShapeName = metadataPrefix + "_source_shape";
    const std::string offsetsName = metadataPrefix + "_offsets";
    const std::string shapeName = metadataPrefix + "_shape";
    const std::string strideName = metadataPrefix + "_stride";
    const size_t rank = tensorType->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, rank >= 1 && rank <= 5)
        << "debug.dump_tensor currently supports tensor rank 1..5, but got " << rank;
    const auto getExprCodes = [&codegen](const std::vector<ir::ExprPtr>& expressions) {
        std::vector<std::string> result;
        result.reserve(expressions.size());
        for (const auto& expression : expressions) {
            result.push_back(codegen.GetExprAsCode(expression));
        }
        return result;
    };
    std::vector<std::string> offsets(rank, "0");
    std::vector<ir::ExprPtr> shapeExprs = tensorType->shape_;
    const bool isNz = cce::IsNZTensorType(tensorType);
    if (op->args_.size() == 3) {
        auto offsetsTuple = ir::As<ir::MakeTuple>(op->args_[1]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, offsetsTuple)
            << "debug.dump_tensor second argument must be a tuple (offsets)";
        auto shapesTuple = ir::As<ir::MakeTuple>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, shapesTuple)
            << "debug.dump_tensor third argument must be a tuple (shapes)";
        offsets = getExprCodes(offsetsTuple->elements_);
        shapeExprs = shapesTuple->elements_;
        if (isNz) {
            const size_t colAxis = tensorType->shape_.size() - 1;
            const int64_t c0 = cce::GetNZInnerCols(tensorType->dtype_);
            const auto knownColOffset = ir::GetConstantDimension(offsetsTuple->elements_[colAxis]);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT,
                              !knownColOffset.has_value() || knownColOffset.value() % c0 == 0)
                << "debug.dump_tensor: NZ column offset must be divisible by C0";
        }
    }
    const auto sourceShape = getExprCodes(tensorType->shape_);
    const auto shape = getExprCodes(shapeExprs);
    const auto stride = BuildTensorDumpStride(codegen, tensorType);

    const std::string layoutName = isNz ? "NZ" : "ND";
    const std::string layout = "pypto::aicore::DumpLayout::" + layoutName;

    EmitDumpMetadataCCE(codegen, op, "dump_tensor");
    const std::string rankStr = std::to_string(rank);
    codegen.Emit("int64_t " + sourceShapeName + "[" + rankStr + "] = {" + JoinExpressions(sourceShape, ", ") + "};");
    codegen.Emit("int64_t " + offsetsName + "[" + rankStr + "] = {" + JoinExpressions(offsets, ", ") + "};");
    codegen.Emit("int64_t " + shapeName + "[" + rankStr + "] = {" + JoinExpressions(shape, ", ") + "};");
    codegen.Emit("int64_t " + strideName + "[" + rankStr + "] = {" + JoinExpressions(stride, ", ") + "};");
    const std::string viewName = metadataPrefix + "_view";
    const std::string innerRows = isNz ? "16" : "1";
    const std::string innerCols = isNz ? strideName + "[" + std::to_string(rank - 1) + "]" : "1";
    codegen.Emit("const pypto::aicore::DumpTensorView " + viewName + " = {" + sourceShapeName + ", " + offsetsName +
                 ", " + shapeName + ", " + strideName + ", " + rankStr + "U, " + innerRows + ", " + innerCols + "};");
    codegen.Emit("PYPTO_DUMP_TENSOR(" + layout + ", " + rankStr + ", " + basePtr + ", " + std::to_string(debugId) +
                 ", " + viewName + ");");
    return "";
}

static std::string MakeDebugPrintfCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);

    std::string format = op->GetKwarg<std::string>("format");
    if (op->GetKwarg<bool>("show_location", false)) {
        std::string location = debug_printf::FormatDebugLocation(op->span_);
        if (!location.empty()) {
            format = location + " " + format;
        }
    }

    std::vector<std::string> args;
    std::vector<DataType> argDtypes;
    args.reserve(op->args_.size());
    argDtypes.reserve(op->args_.size());

    for (size_t i = 0; i < op->args_.size(); ++i) {
        args.emplace_back(codegen.GetExprAsCode(op->args_[i]));
        if (ir::As<ir::PtrType>(op->args_[i]->GetType())) {
            // Pointer arguments are passed directly to PTO printf.
            argDtypes.emplace_back(DataType::INT64);
        } else {
            auto scalarType = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalarType)
                << "debug.printf argument must be ScalarType in CCE lowering";
            argDtypes.emplace_back(scalarType->dtype_);
        }
    }

    std::string call = BuildPtoPrintfCall(format, args, argDtypes);
    if (!call.empty()) {
        codegen.Emit(call);
    }
    return "";
}

static std::string MakeDebugDumpTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 3)
        << "debug.dump_tile requires 1 argument (tile) or 3 arguments (tile, offsets, shapes), but got "
        << op->args_.size();
    const std::string src = cce::GetOperandAsCode(codegen, op->args_[0]);
    auto tileType = ir::As<ir::TileType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tileType) << "debug.dump_tile first argument must be TileType";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, tileType->shape_.size() == 2)
        << "debug.dump_tile CCE lowering currently only supports 2D tiles";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE,
                      ir::As<ir::ConstInt>(tileType->shape_[0]) && ir::As<ir::ConstInt>(tileType->shape_[1]))
        << "debug.dump_tile CCE lowering requires static physical tile shape";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, tileType->memref_.has_value())
        << "debug.dump_tile requires tile to have a memref";

    std::string rowOffset = "0";
    std::string colOffset = "0";
    std::string rows = codegen.GetExprAsCode(tileType->shape_[0]);
    std::string cols = codegen.GetExprAsCode(tileType->shape_[1]);
    if (op->args_.size() == 3) {
        auto offsets = ir::As<ir::MakeTuple>(op->args_[1]);
        auto shapes = ir::As<ir::MakeTuple>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, offsets && shapes)
            << "debug.dump_tile offsets and shapes must be tuples";
        rowOffset = codegen.GetExprAsCode(offsets->elements_[0]);
        colOffset = codegen.GetExprAsCode(offsets->elements_[1]);
        rows = codegen.GetExprAsCode(shapes->elements_[0]);
        cols = codegen.GetExprAsCode(shapes->elements_[1]);
    }

    const int debugId = NextDebugDumpId();
    const auto tileLayout = cce::GetTileTensorLayout(tileType->hardwareInfo_.value_or(ir::HardwareInfo{}));
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, tileLayout.has_value())
        << "debug.dump_tile has an unsupported Tile layout";
    const std::string layoutName = ir::EnumToString(tileLayout.value());
    const std::string layout = "pypto::aicore::DumpLayout::" + layoutName.substr(layoutName.find("::") + 2);

    EmitDumpMetadataCCE(codegen, op, "dump_tile");
    codegen.Emit("PYPTO_DUMP_TILE_WINDOW(" + layout + ", " +
                 JoinExpressions({src, std::to_string(debugId), rowOffset, colOffset, rows, cols}, ", ") + ");");
    return "";
}

// Helper function for get_block_idx (returns value expression).
// Emits the pypto_block_meta.h accessor, which keeps the AscendC GetBlockIdx() contract
// (AIV: global AIV index; AIC: AIC index) and resolves to the SuperKernel per-core
// record or the raw builtin at compile time (__PYPTO_SK_BLOCK__), so the generated
// source is identical for both builds.
static std::string MakeBlockGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0) << "get_block_idx requires no arguments";
    (void)codegenBase;
    return "(int64_t)(__pypto_block_idx_value())";
}

// Helper for dynamic_input_ptr: address of the i-th tensor in a runtime-variadic (DYNAMIC) input
// descriptor. The descriptor layout is read directly -- word[0] is the byte offset from the
// descriptor head to the address array, and the array holds one uint64 address per listed tensor,
// i.e. exactly what AscendC::ListTensorDesc::GetDataPtr does internally:
//     addr(i) = *( (__gm__ uint64_t*)((__gm__ uint8_t*)x + word[0]) + i )
// Reading it directly keeps the op independent of the AscendC list-tensor header (no include is
// injected) and identical on the pto-isa (direct-launch) and OPC paths, matching dynamic_input_dim.
// Returns a pointer expression (not a value); a following ptr.make_tensor turns it into a typed view.
static std::string MakeBlockDynamicInputPtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "dynamic_input_ptr requires 2 arguments";
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    std::string x = cg.GetExprAsCode(op->args_[0]);
    std::string i = cg.GetExprAsCode(op->args_[1]);
    auto ptrType = ir::As<ir::PtrType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ptrType != nullptr) << "dynamic_input_ptr result must be a PtrType";
    std::string ctype = ptrType->dtype_.ToCTypeString();
    // Unmapped dtypes (DT_INT4/DT_UINT4/DT_FP4/DT_HF4) would fall back to "unknown"; fail loudly
    // instead of emitting a broken `__gm__ unknown*`.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ctype != "unknown")
        << "dynamic_input_ptr: descriptor dtype " << ptrType->dtype_.ToString()
        << " has no C type mapping (DT_INT4/DT_UINT4/DT_FP4/DT_HF4 are not usable here)";
    std::string word0 = "*reinterpret_cast<__gm__ uint64_t*>(" + x + ")";
    return "(reinterpret_cast<__gm__ " + ctype + "*>(*(reinterpret_cast<__gm__ uint64_t*>(" +
           "reinterpret_cast<__gm__ uint8_t*>(" + x + ") + " + word0 + ") + (" + i + "))))";
}

// Helper for dynamic_input_dim: read shape(axis) of the i-th tensor in a runtime-variadic
// (DYNAMIC) input descriptor (dim != 0 form). Reads the descriptor layout directly instead of
// AscendC::ListTensorDesc::GetDesc / TensorDesc::GetShape, because those are __NPU_ARCH__-guarded
// and unavailable in the pto-isa (direct-launch) path.
//   word[0] = dataPtrOffset; word[1] = dim(low32) | count(high32);
//   shape structs: <count> entries of (1+dim) words at word[1..];
//   tensor i's shape[axis] is word[1 + i*(1 + dim) + 1 + axis].
static std::string MakeBlockDynamicInputDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "dynamic_input_dim requires 3 arguments";
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    std::string x = cg.GetExprAsCode(op->args_[0]);
    std::string i = cg.GetExprAsCode(op->args_[1]);
    std::string axis = cg.GetExprAsCode(op->args_[2]);
    std::string base = "reinterpret_cast<__gm__ int64_t*>(" + x + ")";
    std::string dim = "(*( " + base + " + 1) & 0xffffffff)";
    std::string a = "(int64_t)(" + axis + ")";
    // Guard: 0 <= axis < dim (out-of-range traps instead of reading a wrong word -- past the
    // shape block for axis >= dim, and before it for a negative axis).
    return "((0 <= " + a + " && " + a + " < (int64_t)(" + dim + ")) ? *( " + base + " + (1 + (int64_t)(" + i +
           ")*(1 + (int64_t)(" + dim + ")) + 1 + " + a + ")) : (trap(), (int64_t)(-1)))";
}

static std::string TileConstructorArgs(codegen::CCECodegen& codegen, const ir::TileTypePtr& type)
{
    // A conv-layout tile is a pto::ConvTile whose whole shape is carried by ConvTileShape<...>;
    // pto's ConvTile has no (dims...) constructor form for static shapes, so emit no ctor args.
    if (type->hardwareInfo_.has_value() && type->hardwareInfo_->convLayout >= 0) {
        return "";
    }
    std::vector<std::string> dims;
    for (size_t i = 0; i < type->shape_.size(); ++i) {
        ir::ExprPtr valid = type->tileView_ && i < type->tileView_->validShape.size() ? type->tileView_->validShape[i] :
                                                                                        nullptr;
        if (!valid || ir::As<ir::ConstInt>(valid)->value_ == -1) {
            dims.push_back(codegen.GetExprAsCode(type->shape_[i]));
        }
    }
    return JoinExpressions(dims, ", ");
}

static std::vector<std::string> InheritedMutexIds(codegen::CCECodegen& codegen, const ir::ExprPtr& source)
{
    std::vector<std::string> ids;
    auto type = ir::As<ir::MutexTileType>(source->GetType());
    const auto value = codegen.GetExprAsCode(source);
    for (uint32_t i = 0; i < type->mutexCount_; ++i)
        ids.push_back(value + ".mutex_ids[" + std::to_string(i) + "]");
    return ids;
}

static void DeclareTile(codegen::CCECodegen& codegen, const ir::TileTypePtr& type, const std::string& target,
                        const std::string& ctor, const std::vector<std::string>& ids)
{
    const auto alias = target + "_Type";
    codegen.Emit("using " + alias + " = " + codegen.GetGeneratedType(type) + ";");
    codegen.Emit(alias + " " + target + "{" + alias + "::PtoTile(" + ctor + "), {" + JoinExpressions(ids, ", ") +
                 "}};");
}

// Construct the complete descriptor at the allocation op, where all operands are in scope.
static std::string MakeBlockCreateMutexTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    const auto target = codegen.GetCurrentResultTarget();
    auto type = ir::As<ir::TileType>(op->GetType());
    std::vector<std::string> ids;
    for (const auto& id : ir::As<ir::MakeTuple>(op->args_[2])->elements_)
        ids.push_back(codegen.GetExprAsCode(id));
    DeclareTile(codegen, type, target, TileConstructorArgs(codegen, type), ids);
    if (type->memref_) {
        const auto addr = codegen.GetConstIntValue((*type->memref_)->addr_);
        std::ostringstream address;
        address << "0x" << std::hex << addr;
        codegen.Emit("TASSIGN(" + target + ".tile, " + address.str() + ");");
    }
    return target;
}

// Helper for ptr.make_tensor (tensor view). Emits the GlobalTensor declaration in place
// at the make_tensor op: the source pointer (op->args_[0]) is already in C++ scope here
// (a function parameter or an earlier ptr.addptr local), so we resolve it directly via
// GetExprAsCode instead of relying on PtrType base/offset annotations (those are ptoas-only).
// The view's access_shape/is_transpose/tile_dims come from the prescanned TensorDefs, looked up
// by the assignment target name -- one per layout its accesses need. Returns "" (no inline value).
static std::string MakeBlockMakeTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    const std::string name = cg.GetCurrentResultTarget();
    // The source of the view is either a raw pointer (PtrType) or an existing tensor (TensorType,
    // re-viewed with a new shape/stride). Resolve the base pointer accordingly and record the
    // source element dtype.
    std::string ptrCode;
    ir::DataType sourceDtype;
    if (auto ptrType = ir::As<ir::PtrType>(op->args_[0]->GetType())) {
        ptrCode = cce::GetOperandAsCode(cg, op->args_[0]);
        sourceDtype = ptrType->dtype_;
    } else if (auto srcTensorType = ir::As<ir::TensorType>(op->args_[0]->GetType())) {
        // Re-view of an existing tensor: reuse its already-registered base pointer (a function
        // parameter's "<name>_ptr" or an earlier make_tensor view's pointer).
        auto srcVar = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcVar != nullptr)
            << "ptr.make_tensor from a tensor requires the source to be a tensor variable";
        const std::string srcName = cg.GetVarName(srcVar);
        ptrCode = cg.HasPointer(srcName) ? cg.GetPointer(srcName) : (srcName + ".data()");
        sourceDtype = srcTensorType->dtype_;
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
            << "ptr.make_tensor source must be a PtrType or TensorType";
    }
    // The view's element dtype may differ from the source's element dtype (e.g. a raw uint8
    // pointer reinterpreted as an fp16 view via ptr.make_tensor(..., dtype=FP16)). The
    // GlobalTensor<element_type> instance is constructed from this pointer, so reinterpret-cast
    // the base pointer to the view element type when the dtypes differ (a no-op when they match).
    auto tensorType = ir::As<ir::TensorType>(op->GetType());
    if (tensorType && !(sourceDtype == tensorType->dtype_)) {
        ptrCode = "(__gm__ " + tensorType->dtype_.ToCTypeString() + "*)(" + ptrCode + ")";
    }
    cg.RegisterPointer(name, ptrCode);
    for (const codegen::TensorDef* def : cg.GetTensorDefs(name)) {
        cg.GenerateGlobalTensorTypeDeclaration(*def);
    }
    return "";
}

// Helper for ptr.addptr / advancing a raw pointer by an element offset. Emits no
// statement; returns the pointer-arithmetic expression so the result var maps to
// it (used as a base address by a subsequent ptr.make_tensor).
static std::string MakePtrAddPtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "ptr.addptr requires 2 arguments: ptr, offset";
    std::string ptr = cce::GetOperandAsCode(codegen, op->args_[0]);
    std::string offset = cce::GetOperandAsCode(codegen, op->args_[1]);
    return "(" + ptr + " + " + offset + ")";
}

// Helper for ptr.make_ptr / reinterpreting a raw pointer (or extracting a pointer from a
// tensor) as a different element type. Emits no statement; returns the expression so the
// result var maps to it (used as a base address by a subsequent ptr.addptr / ptr.make_tensor).
static std::string MakePtrMakePtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "ptr.make_ptr requires 1 argument: ptr";
    std::string ptr;
    if (auto ptrType = ir::As<ir::PtrType>(op->args_[0]->GetType())) {
        ptr = cce::GetOperandAsCode(codegen, op->args_[0]);
    } else if (ir::As<ir::TensorType>(op->args_[0]->GetType())) {
        auto srcVar = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, srcVar != nullptr)
            << "ptr.make_ptr from a tensor requires the source to be a tensor variable";
        const std::string srcName = codegen.GetVarName(srcVar);
        ptr = codegen.HasPointer(srcName) ? codegen.GetPointer(srcName) : (srcName + ".data()");
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "ptr.make_ptr source must be a PtrType or TensorType";
    }
    auto resultPtrType = ir::As<ir::PtrType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, resultPtrType != nullptr) << "ptr.make_ptr result must be a PtrType";
    return "((__gm__ " + resultPtrType->dtype_.ToCTypeString() + "*)(" + ptr + "))";
}

static std::string MakeBlockReinterpretCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(base);
    auto tileType = ir::As<ir::TileType>(op->GetType());
    const std::string target = codegen.GetCurrentResultTarget();
    PRO_CODEGEN_IRCHECK(ExternalError::INVALID_OPERATION, !target.empty())
        << "reinterpret must be assigned to an IR value";
    std::string sourceTile = cce::GetOperandAsCode(codegen, op->args_[0]);
    DeclareTile(codegen, tileType, target, TileConstructorArgs(codegen, tileType),
                InheritedMutexIds(codegen, op->args_[0]));
    codegen.Emit("TASSIGN(" + target + ".tile, (uint64_t)" + sourceTile + ".data());");
    return target;
}

// ============================================================================
// Matmul Operations
// ============================================================================

// ============================================================================
// Elementwise Operations
// ============================================================================

// ============================================================================
// Unary Operations
// ============================================================================

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "block.make_mutex_tile")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockCreateMutexTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "block.tile_mutex_id")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& base) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(base);
        return codegen.GetExprAsCode(op->args_[0]) + ".mutex_ids[" + cce::GetOperandAsCode(codegen, op->args_[1]) + "]";
    });

REGISTER_BACKEND_OP(BackendCCE, "block.reinterpret")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeBlockReinterpretCodegenCCE);

REGISTER_BACKEND_OP(BackendCCE, "ptr.make_tensor")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockMakeTensorCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "ptr.addptr")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakePtrAddPtrCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "ptr.make_ptr")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakePtrMakePtrCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "get_block_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockGetBlockIdxCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dynamic_input_ptr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockDynamicInputPtrCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dynamic_input_dim")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockDynamicInputDimCodegenCCE(op, codegen);
    });

// Helper function for get_spr (reads AR special purpose register via get_ar())
static std::string MakeGetSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    (void)codegenBase;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0) << "get_spr requires no arguments";
    return "get_ar()";
}

REGISTER_BACKEND_OP(BackendCCE, "get_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetSprCodegenCCE(op, codegen); });

// Helper function for get_cycle (reads the system cycle counter via get_sys_cnt())
static std::string MakeGetCycleCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    (void)codegenBase;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0) << "get_cycle requires no arguments";
    return "(int64_t)(get_sys_cnt())";
}

REGISTER_BACKEND_OP(BackendCCE, "get_cycle")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeGetCycleCodegenCCE(op, codegen);
    });

// ============================================================================
// Saturation flag (CTRL register bit manipulation)
// ============================================================================

static int8_t GetSaturationModeBit(ir::SaturationFlagMode mode, const std::string& opName)
{
    switch (mode) {
        case ir::SaturationFlagMode::FLOAT:
            return 48;
        case ir::SaturationFlagMode::FLOAT8:
            return 50;
        case ir::SaturationFlagMode::INT:
            return 53;
        case ir::SaturationFlagMode::CAST:
            return 59;
        default:
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false) << opName << ": unsupported mode";
            return -1;
    }
}

// FLOAT/FLOAT8/CAST: polarity inverted (bit=0 means saturation ON, bit=1 means OFF)
// INT: polarity normal (bit=1 means saturation ON, bit=0 means OFF)
static bool IsInvertedPolarity(ir::SaturationFlagMode mode)
{
    return mode == ir::SaturationFlagMode::FLOAT || mode == ir::SaturationFlagMode::FLOAT8 ||
           mode == ir::SaturationFlagMode::CAST;
}

static std::string MakeSetSaturationFlagCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->HasKwarg("mode"))
        << "set_saturation_flag requires 'mode' kwarg";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->HasKwarg("enable"))
        << "set_saturation_flag requires 'enable' kwarg";
    auto mode = static_cast<ir::SaturationFlagMode>(op->GetKwarg<int>("mode"));
    int8_t bit = GetSaturationModeBit(mode, "set_saturation_flag");
    bool inverted = IsInvertedPolarity(mode);

    bool enable = op->GetKwarg<bool>("enable");

    // For inverted polarity: enable=true → set bit to 0 (sbitset0), enable=false → set bit to 1 (sbitset1)
    // For normal polarity (INT): enable=true → set bit to 1 (sbitset1), enable=false → set bit to 0 (sbitset0)
    std::string setFn;
    if (inverted) {
        setFn = enable ? "sbitset0" : "sbitset1";
    } else {
        setFn = enable ? "sbitset1" : "sbitset0";
    }
    codegen.Emit("set_ctrl(" + setFn + "(get_ctrl(), " + std::to_string(bit) + "));");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "set_saturation_flag")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSetSaturationFlagCodegenCCE(op, codegen);
    });

static std::string MakeGetSaturationFlagCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    (void)codegenBase;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->HasKwarg("mode"))
        << "get_saturation_flag requires 'mode' kwarg";
    auto mode = static_cast<ir::SaturationFlagMode>(op->GetKwarg<int>("mode"));
    int8_t bit = GetSaturationModeBit(mode, "get_saturation_flag");
    bool inverted = IsInvertedPolarity(mode);

    // Read CTRL bit, invert for FLOAT/FLOAT8/CAST modes
    if (inverted) {
        return "((get_ctrl() >> " + std::to_string(bit) + ") & 1) == 0";
    } else {
        return "((get_ctrl() >> " + std::to_string(bit) + ") & 1) != 0";
    }
}

REGISTER_BACKEND_OP(BackendCCE, "get_saturation_flag")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeGetSaturationFlagCodegenCCE(op, codegen);
    });

// ============================================================================
// CTRL SPR direct access (SetCtrlSpr / GetCtrlSpr / ResetCtrlSpr)
// ============================================================================

// 3510 writable CTRL bits: 6-10 (range), 45, 48, 50, 53, 59, 60 (single bits)
static void CheckCtrlBitRange(int8_t startBit, int8_t endBit, const std::string& opName)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, startBit >= 0 && endBit < 64 && startBit <= endBit)
        << opName << ": invalid bit range [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit)
        << "], must be 0 <= startBit <= endBit < 64";
    bool valid = (6 <= startBit && startBit <= 10 && 6 <= endBit && endBit <= 10) ||
                 (startBit == endBit && (startBit == 45 || startBit == 48 || startBit == 50 || startBit == 53 ||
                                         startBit == 59 || startBit == 60));
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, valid)
        << opName << ": bits [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit)
        << "] are not writable on current device. Writable: bits 6-10, 45, 48, 50, 53, 59, 60";
}

static std::string MakeSetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "set_ctrl_spr requires 3 args (start_bit, end_bit, value)";
    auto startVal = ir::As<ir::ConstInt>(op->args_[0]);
    auto endVal = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, startVal != nullptr && endVal != nullptr)
        << "set_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(startVal->value_);
    int8_t endBit = static_cast<int8_t>(endVal->value_);
    CheckCtrlBitRange(startBit, endBit, "set_ctrl_spr");
    auto valueConst = ir::As<ir::ConstInt>(op->args_[2]);
    if (valueConst != nullptr && startBit >= 6 && endBit <= 10) {
        // CTRL[8:6]=3'b111 (atomic operand dtype) and CTRL[10:9]=2'b11 (atomic op
        // type) are undefined encodings that fault the device — reject up front.
        int64_t v = valueConst->value_ & ((int64_t(1) << (endBit - startBit + 1)) - 1);
        if (startBit <= 6 && endBit >= 8 && ((v >> (6 - startBit)) & 0x7) == 0x7) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
                << "set_ctrl_spr: value " << valueConst->value_
                << " sets CTRL[8:6]=3'b111, an undefined atomic operand dtype "
                << "(supported: 0-6 = none/float/half/int16/int32/int8/bfloat16)";
        }
        if (startBit <= 9 && endBit >= 10 && ((v >> (9 - startBit)) & 0x3) == 0x3) {
            PRO_CODEGEN_CHECK(ExternalError::NAME_ERROR, false)
                << "set_ctrl_spr: value " << valueConst->value_
                << " sets CTRL[10:9]=2'b11, an undefined atomic op type "
                << "(supported: 0-2 = ADD/MAX/MIN)";
        }
    }
    std::string value = cce::GetOperandAsCode(codegen, op->args_[2]);
    if (endBit - startBit == 63) {
        codegen.Emit("set_ctrl(" + value + ");");
    } else {
        codegen.Emit("set_ctrl((get_ctrl() & ~(((uint64_t(1) << " + std::to_string(endBit - startBit + 1) +
                     ") - 1) << " + std::to_string(startBit) + ")) | ((" + value + " & ((uint64_t(1) << " +
                     std::to_string(endBit - startBit + 1) + ") - 1)) << " + std::to_string(startBit) + "));");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "set_ctrl_spr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSetCtrlSprCodegenCCE(op, codegen);
    });

static std::string MakeGetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    (void)codegenBase;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "get_ctrl_spr requires 2 args (start_bit, end_bit)";
    auto startVal = ir::As<ir::ConstInt>(op->args_[0]);
    auto endVal = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, startVal != nullptr && endVal != nullptr)
        << "get_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(startVal->value_);
    int8_t endBit = static_cast<int8_t>(endVal->value_);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, startBit >= 0 && endBit < 64 && startBit <= endBit)
        << "get_ctrl_spr: invalid bit range [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit) << "]";
    if (endBit - startBit == 63) {
        return "get_ctrl()";
    }
    return "(get_ctrl() >> " + std::to_string(startBit) + ") & ((uint64_t(1) << " +
           std::to_string(endBit - startBit + 1) + ") - 1)";
}

REGISTER_BACKEND_OP(BackendCCE, "get_ctrl_spr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeGetCtrlSprCodegenCCE(op, codegen);
    });

static std::string MakeResetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "reset_ctrl_spr requires 2 args (start_bit, end_bit)";
    auto startVal = ir::As<ir::ConstInt>(op->args_[0]);
    auto endVal = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, startVal != nullptr && endVal != nullptr)
        << "reset_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(startVal->value_);
    int8_t endBit = static_cast<int8_t>(endVal->value_);
    CheckCtrlBitRange(startBit, endBit, "reset_ctrl_spr");
    constexpr int64_t defaultCtrl = 0x1000000000000008LL;
    if (endBit - startBit == 63) {
        codegen.Emit("set_ctrl(" + std::to_string(defaultCtrl) + "LL);");
    } else {
        uint64_t mask = ((uint64_t(1) << (endBit - startBit + 1)) - 1) << startBit;
        int64_t defaultBits = defaultCtrl & static_cast<int64_t>(mask);
        codegen.Emit("set_ctrl((get_ctrl() & ~(((uint64_t(1) << " + std::to_string(endBit - startBit + 1) +
                     ") - 1) << " + std::to_string(startBit) + ")) | " + std::to_string(defaultBits) + "LL);");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "reset_ctrl_spr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeResetCtrlSprCodegenCCE(op, codegen);
    });

// ============================================================================
// Reduction Operations
// ============================================================================

// ============================================================================
// Broadcast Operations
// ============================================================================

// ============================================================================
// Transform Operations (view/reshape/transpose: same buffer, reinterpret)
// ============================================================================

[[maybe_unused]] static std::string MakeTileTransposeCodegenCCE(const ir::CallPtr& op,
                                                                codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    std::string targetVar = codegen.GetCurrentResultTarget();
    std::string inputVar = cce::GetOperandAsCode(codegen, op->args_[0]);
    auto axis1 = codegen.GetConstIntValue(op->args_[1]);
    auto axis2 = codegen.GetConstIntValue(op->args_[2]);
    int64_t ndim = static_cast<int64_t>(ir::As<ir::TileType>(op->args_[0]->GetType())->shape_.size());

    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ndim == 2)
        << "Codegen only supports 2D tiles, but got " << ndim << "D tile";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, axis1 != axis2)
        << "tile.transpose: axis1 and axis2 must be different, but got axis1=axis2=" << axis1;
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               axis1 >= 0 && axis1 < ndim && axis2 >= 0 && axis2 < ndim)
        << "tile.transpose: axis1 and axis2 must be in range [0, " << ndim << "), but got axis1=" << axis1
        << ", axis2=" << axis2;

    codegen.Emit("TTRANS(" + targetVar + ", " + inputVar + ");");
    return "";
}

// ============================================================================
// Sync / Barrier Operations (inserted by insert_sync_pass)
// ============================================================================

static std::string PipeTypeToCCEString(ir::PipeType pipe)
{
    switch (pipe) {
        case ir::PipeType::MTE1:
            return "PIPE_MTE1";
        case ir::PipeType::MTE2:
            return "PIPE_MTE2";
        case ir::PipeType::MTE3:
            return "PIPE_MTE3";
        case ir::PipeType::M:
            return "PIPE_M";
        case ir::PipeType::V:
            return "PIPE_V";
        case ir::PipeType::S:
            return "PIPE_S";
        case ir::PipeType::FIX:
            return "PIPE_FIX";
        case ir::PipeType::ALL:
            return "PIPE_ALL";
        default:
            return "PIPE_V";
    }
}

static std::string EnumValueName(const char* fullName)
{
    const char* sep = std::strrchr(fullName, ':');
    return sep ? std::string(sep + 1) : std::string(fullName);
}

static std::string NormalizeDcciCacheLine(int cacheLine)
{
    auto cl = static_cast<ir::CacheLine>(cacheLine);
    return EnumValueName(ir::EnumToString(cl));
}

static std::string NormalizeDcciDst(int dst, bool isTile)
{
    auto d = static_cast<ir::DcciDst>(dst);
    if (d == ir::DcciDst::AUTO) {
        return isTile ? "CACHELINE_UB" : "CACHELINE_OUT";
    }
    return EnumValueName(ir::EnumToString(d));
}

static std::string MakeDcciCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 2)
        << "system.dcci requires 1 or 2 arguments, got " << op->args_.size();

    int cacheLineInt = op->HasKwarg("cache_line") ? op->GetKwarg<int>("cache_line") : 1; // ENTIRE_DATA_CACHE
    int dstInt = op->HasKwarg("dst") ? op->GetKwarg<int>("dst") : 0;                     // AUTO

    const std::string cacheLine = NormalizeDcciCacheLine(cacheLineInt);

    auto tensorType = ir::As<ir::TensorType>(op->args_[0]->GetType());
    if (tensorType != nullptr) {
        auto tensorVarPtr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensorVarPtr != nullptr)
            << "system.dcci: tensor target must be a Var";
        std::string tensorVar = codegen.GetVarName(tensorVarPtr);
        std::string offset = "0";
        if (op->args_.size() == 2) {
            auto offsetsTuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
            if (offsetsTuple != nullptr) {
                offset = codegen.ComputeTensorOffset(tensorType, offsetsTuple);
            } else {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                                  ir::As<ir::ScalarType>(op->args_[1]->GetType()) != nullptr)
                    << "system.dcci: tensor target offset must be a tuple or scalar expression";
                offset = cce::GetOperandAsCode(codegen, op->args_[1]);
            }
        }
        std::string tensorPtr = codegen.GetPointer(tensorVar);
        if (tensorPtr.empty()) {
            tensorPtr = tensorVar + ".data()";
        }
        const std::string dstAttr = NormalizeDcciDst(dstInt, false);
        codegen.Emit("dcci(reinterpret_cast<__gm__ void*>(" + tensorPtr + " + " + offset + "), " + cacheLine + ", " +
                     dstAttr + ");");
        return "";
    }

    auto tileType = ir::As<ir::TileType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tileType != nullptr)
        << "system.dcci: target must be TensorType or TileType";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, tileType->memref_.has_value())
        << "system.dcci: tile target must have an allocated memory space";
    if (tileType->memref_.value()->memorySpace_ != ir::MemorySpace::Vec) {
        PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_OPERATION)
            << "system.dcci: tile target must be allocated in Vec memory";
    }

    std::string tile = cce::GetOperandAsCode(codegen, op->args_[0]);
    std::string offset = "0";
    if (op->args_.size() == 2) {
        offset = cce::GetOperandAsCode(codegen, op->args_[1]);
    }
    const std::string dstAttr = NormalizeDcciDst(dstInt, true);
    codegen.Emit("dcci(reinterpret_cast<__ubuf__ void*>(" + tile + ".data() + " + offset + "), " + cacheLine + ", " +
                 dstAttr + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.bar_m")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("pipe_barrier(PIPE_M);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte1")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("pipe_barrier(PIPE_MTE1);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte2")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("pipe_barrier(PIPE_MTE2);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte3")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("pipe_barrier(PIPE_MTE3);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_fix")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("pipe_barrier(PIPE_FIX);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("pipe_barrier(PIPE_ALL);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mask_count")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("set_mask_count();");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mask_norm")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase).Emit("set_mask_norm();");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_vec_mask")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
            << "system.set_vec_mask requires 2 arguments, but got " << op->args_.size();
        std::string maskHigh = cce::GetOperandAsCode(codegen, op->args_[0]);
        std::string maskLow = cce::GetOperandAsCode(codegen, op->args_[1]);
        codegen.Emit("set_vector_mask(" + maskHigh + ", " + maskLow + ");");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.reset_mask")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegenBase)
            .Emit("set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.dcci")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        return MakeDcciCodegenCCE(op, codegenBase);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.icache_preload")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
            << "system.icache_preload requires one argument";
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510)
            << "system.icache_preload is only supported on DAV_3510 (Ascend 950)";
        codegen.Emit("AscendC::ICachePreLoad(" + cce::GetOperandAsCode(codegen, op->args_[0]) + ");");
        return "";
    });

// ============================================================================
// Cross-core Sync Operations
// ============================================================================

// Cross-core SET: ffts_cross_core_sync(PIPE_xxx, getFFTSMsg(mode, event_id))
//   - INTER_BLOCK(0): inter-core sync
//   - INTER_SUBBLOCK(1): intra-core AIV-to-AIV sync
//   - INTRA_BLOCK(2): intra-core AIC↔AIV both subcores (3510 uses set_intra_block)
//   - UNICAST_BLOCK(3): intra-core AIC↔AIV one subcore (3510 uses set_intra_block)
// Cross-core WAIT: wait_flag_dev / wait_intra_block
// SET signals completion from a pipe; WAIT blocks until the other core signals.

static std::string MakeCrossCoreSetCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase, bool isDynamic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    auto pipe = op->GetKwarg<int>("pipe");
    std::string pipeStr = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
    auto syncMode = static_cast<ir::CrossCoreSyncMode>(op->GetKwarg<int>("sync_mode"));
    bool isIntraUnicast = syncMode == ir::CrossCoreSyncMode::INTRA_BLOCK ||
                          syncMode == ir::CrossCoreSyncMode::UNICAST_BLOCK;
    if (isIntraUnicast) {
        // CUBE→VEC: INTRA_BLOCK signals both vectors only on a 1:2 target; UNICAST_BLOCK signals one.
        //     VEC→CUBE: single set
        if (codegen.GetTarget() == ir::SectionKind::Cube && syncMode == ir::CrossCoreSyncMode::INTRA_BLOCK) {
            if (isDynamic) {
                std::string eventId = cce::GetOperandAsCode(codegen, op->args_[0]);
                codegen.Emit("set_intra_block(" + pipeStr + ", " + eventId + ");");
                if (codegen.GetVectorCoresPerCube() == 2) {
                    codegen.Emit("set_intra_block(" + pipeStr + ", " + eventId + " + 16);");
                }
            } else {
                int eventId = op->GetKwarg<int>("event_id");
                codegen.Emit("set_intra_block(" + pipeStr + ", " + std::to_string(eventId) + ");");
                if (codegen.GetVectorCoresPerCube() == 2) {
                    codegen.Emit("set_intra_block(" + pipeStr + ", " + std::to_string(eventId + 16) + ");");
                }
            }
        } else {
            if (isDynamic) {
                std::string eventId = cce::GetOperandAsCode(codegen, op->args_[0]);
                codegen.Emit("set_intra_block(" + pipeStr + ", " + eventId + ");");
            } else {
                int eventId = op->GetKwarg<int>("event_id");
                codegen.Emit("set_intra_block(" + pipeStr + ", " + std::to_string(eventId) + ");");
            }
        }
    } else {
        // 3510/9201 INTER_BLOCK / INTER_SUBBLOCK: ffts_cross_core_sync
        std::string modeStr = std::to_string(static_cast<int>(syncMode));
        if (isDynamic) {
            std::string eventId = cce::GetOperandAsCode(codegen, op->args_[0]);
            codegen.Emit("ffts_cross_core_sync(" + pipeStr + ", getFFTSMsg(" + modeStr + ", " + eventId + "));");
        } else {
            int eventId = op->GetKwarg<int>("event_id");
            codegen.Emit("ffts_cross_core_sync(" + pipeStr + ", getFFTSMsg(" + modeStr + ", " +
                         std::to_string(eventId) + "));");
        }
    }
    return "";
}

static void EmitWaitIntraBlockCCE(codegen::CCECodegen& codegen, const ir::CallPtr& op, const std::string& pipeStr,
                                  bool isDynamic, int eventIdOffset = 0)
{
    if (isDynamic) {
        std::string eventId = cce::GetOperandAsCode(codegen, op->args_[0]);
        if (eventIdOffset != 0) {
            eventId += " + " + std::to_string(eventIdOffset);
        }
        codegen.Emit("wait_intra_block(" + pipeStr + ", " + eventId + ");");
        return;
    }

    int eventId = op->GetKwarg<int>("event_id") + eventIdOffset;
    codegen.Emit("wait_intra_block(" + pipeStr + ", " + std::to_string(eventId) + ");");
}

static std::string MakeCrossCoreWaitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase, bool isDynamic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    auto pipe = op->GetKwarg<int>("pipe");
    std::string pipeStr = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
    auto syncMode = static_cast<ir::CrossCoreSyncMode>(op->GetKwarg<int>("sync_mode"));
    bool isIntraUnicast = (syncMode == ir::CrossCoreSyncMode::INTRA_BLOCK ||
                           syncMode == ir::CrossCoreSyncMode::UNICAST_BLOCK);
    bool waitTwoVecSubcores = (syncMode == ir::CrossCoreSyncMode::INTRA_BLOCK);
    if (isIntraUnicast) {
        // CUBE waiting for VEC: INTRA_BLOCK waits for both vectors only on a 1:2 target.
        //     VEC waiting for CUBE: single wait
        if (codegen.GetTarget() == ir::SectionKind::Cube && waitTwoVecSubcores) {
            EmitWaitIntraBlockCCE(codegen, op, pipeStr, isDynamic);
            if (codegen.GetVectorCoresPerCube() == 2) {
                EmitWaitIntraBlockCCE(codegen, op, pipeStr, isDynamic, 16);
            }
        } else {
            EmitWaitIntraBlockCCE(codegen, op, pipeStr, isDynamic);
        }
    } else {
        // 3510 + INTER_BLOCK(0) / INTER_SUBBLOCK(1): wait_flag_dev
        std::string eventId = isDynamic ? cce::GetOperandAsCode(codegen, op->args_[0]) :
                                          std::to_string(op->GetKwarg<int>("event_id"));
        codegen.Emit("wait_flag_dev(" + pipeStr + ", " + eventId + ");");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.set_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreSetCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.wait_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreWaitCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreSetCodegenCCE(op, codegen, true);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.wait_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeCrossCoreWaitCodegenCCE(op, codegen, true);
    });

// ============================================================================
// Flag synchronization operations
// ============================================================================

static std::string MakeSyncCodegenCCE(const std::string& intrinsic, const ir::CallPtr& op,
                                      codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    auto setPipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
    auto waitPipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
    std::string setPipeStr = PipeTypeToCCEString(setPipe);
    std::string waitPipeStr = PipeTypeToCCEString(waitPipe);
    std::string eventId = cce::GetOperandAsCode(codegen, op->args_[0]);
    codegen.Emit(intrinsic + "(" + setPipeStr + ", " + waitPipeStr + ", (event_t)" + eventId + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_src_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncCodegenCCE("set_flag", op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.sync_dst_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSyncCodegenCCE("wait_flag", op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.dump_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugDumpTensorCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.dump_tile")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugDumpTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.printf")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugPrintfCodegenCCE(op, codegen);
    });

// ============================================================================
// Debug operations: assert and trap
// ============================================================================

static std::string MakeDebugAssertCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);

    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 1)
        << "debug.assert requires at least 1 argument (condition)";

    std::string condition = cce::GetOperandAsCode(codegen, op->args_[0]);
    std::string conditionText = op->GetKwarg<std::string>("condition_text");
    std::string format = op->GetKwarg<std::string>("format");

    codegen.Emit("if (!(" + condition + ")) {");

    if (op->GetKwarg<bool>("show_location", false)) {
        std::string location = debug_printf::FormatDebugLocation(op->span_);
        if (!location.empty()) {
            codegen.Emit("  pypto::aicore::printf(\"" +
                         debug_printf::EscapeStringLiteral(location + " Assertion failed: " + conditionText + "\n") +
                         "\");");
        } else {
            codegen.Emit("  pypto::aicore::printf(\"" +
                         debug_printf::EscapeStringLiteral("Assertion failed: " + conditionText + "\n") + "\");");
        }
    } else {
        codegen.Emit("  pypto::aicore::printf(\"" +
                     debug_printf::EscapeStringLiteral("Assertion failed: " + conditionText + "\n") + "\");");
    }

    if (!format.empty()) {
        std::vector<std::string> args;
        std::vector<DataType> argDtypes;
        for (size_t i = 1; i < op->args_.size(); ++i) {
            args.emplace_back(cce::GetOperandAsCode(codegen, op->args_[i]));
            auto scalarType = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalarType) << "debug.assert argument must be ScalarType";
            argDtypes.emplace_back(scalarType->dtype_);
        }
        codegen.Emit("  " + BuildPtoPrintfCall(format, args, argDtypes));
    }

    codegen.Emit("}");
    return "";
}

static std::string MakeDebugTrapCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    codegen.Emit("trap();");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "debug.assert")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugAssertCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "debug.trap")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDebugTrapCodegenCCE(op, codegen);
    });

// ============================================================================
// Sanitizer log record - device-side write of one per-detection variable-size
// record into the GM log buffer passed as the trailing `sanitizer_log`
// parameter. One op covers every detection type: the leading det_id argument
// selects the record layout on the host side (pypto_pro.runtime.
// sanitizer_replay decodes by det_id); the trailing three arguments are the
// hidden log-buffer pointer / capacity and the span id.
//   args = [det_id, field..., log_ptr, capacity, span_id]
//   record = [det_id, field..., span_id] (rec_u32 = args.size() - 2)
// The buffer region per sub-block is [ctr u32, records...], where ctr is the
// region's u32-word offset past itself and the capacity
// (`sanitizer_log_capacity` scalar) is the per-region u32 word budget for
// records.
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "block.sanitizer_log_region")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
        const auto log = cce::GetOperandAsCode(codegen, op->args_[0]);
        const auto capacity = cce::GetOperandAsCode(codegen, op->args_[1]);
        // AIV regions follow the AIC range. Scalar and SIMT share the AIV region.
        std::string region = "__pypto_block_idx_value()";
        if (codegen.GetTarget() == ir::SectionKind::Vector) {
            region = "__pypto_block_num_value() + " + region;
        }
        return "(" + log + " != 0 ? " + log + " + static_cast<uint64_t>(" + region + ") * (static_cast<uint64_t>(" +
               capacity + ") + 1u) * sizeof(uint32_t) : " + log + ")";
    });

static std::string MakeSanitizerLogCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    // args = [det_id, field..., log_ptr, capacity, span_id] -> at least the
    // det_id, the two hidden buffer parameters and the span id.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, op->args_.size() >= 4)
        << "block.sanitizer_log requires (det_id, field..., log_ptr, capacity, span_id)";

    const size_t n = op->args_.size();
    std::string detId = cce::GetOperandAsCode(codegen, op->args_[0]);
    // Hidden log-buffer parameters: resolve their final (SSA-suffixed) names.
    std::string log = cce::GetOperandAsCode(codegen, op->args_[n - 3]);
    std::string capacity = cce::GetOperandAsCode(codegen, op->args_[n - 2]);
    std::string spanId = cce::GetOperandAsCode(codegen, op->args_[n - 1]);
    // Record size in u32 words: det_id + fields + span_id (fields exclude the
    // hidden log/capacity parameters).
    const uint32_t recU32 = static_cast<uint32_t>(n) - 2u;

    codegen.Emit("if (" + log + " != 0) {");
    codegen.Emit("  __gm__ uint32_t* __log_base = reinterpret_cast<__gm__ uint32_t*>(" + log + ");");
    // All writers CAS-reserve whole records; rejected records advance nothing.
    codegen.Emit("  uint32_t __log_off = atomicCAS(__log_base, 0u, 0u);");
    codegen.Emit("  while (__log_off <= static_cast<uint32_t>(" + capacity + ") && " + std::to_string(recU32) +
                 "u <= static_cast<uint32_t>(" + capacity + ") - __log_off) {");
    codegen.Emit("    uint32_t __log_seen = atomicCAS(__log_base, __log_off, __log_off + " + std::to_string(recU32) +
                 "u);");
    codegen.Emit("    if (__log_seen == __log_off) break;");
    codegen.Emit("    __log_off = __log_seen;");
    codegen.Emit("  }");
    codegen.Emit("  if (__log_off <= static_cast<uint32_t>(" + capacity + ") && " + std::to_string(recU32) +
                 "u <= static_cast<uint32_t>(" + capacity + ") - __log_off) {");
    codegen.Emit("    __gm__ uint32_t* __log_rec = __log_base + 1u + __log_off;");
    // Word atomics bypass Scalar DCache: adjacent records may share a cache line.
    codegen.Emit("    atomicExch(__log_rec + 0, static_cast<uint32_t>(" + detId + "));  // det_id");
    for (size_t i = 1; i + 3 < n; ++i) {
        codegen.Emit("    atomicExch(__log_rec + " + std::to_string(i) + ", static_cast<uint32_t>(" +
                     cce::GetOperandAsCode(codegen, op->args_[i]) + "));");
    }
    codegen.Emit("    atomicExch(__log_rec + " + std::to_string(recU32 - 1u) + ", static_cast<uint32_t>(" + spanId +
                 "));");
    // Preserve the GM ordering primitive for this execution context.
    codegen.Emit(codegen.IsInSimtContext() ? "    __threadfence();" : "    dsb(DSB_DDR);");
    codegen.Emit("  }");
    codegen.Emit("}");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "block.sanitizer_log").set_pipe(ir::PipeType::S).f_codegen(MakeSanitizerLogCodegenCCE);

// Shared block.tile_addr helper. Keep the original Tile expression in the IR so
// CCE codegen can use the same MakeTuple/TileGroup materialization as the
// instrumented operation, then read the selected runtime Tile's data address.
static std::string MakeTileAddrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "block.tile_addr requires one Tile argument";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, ir::As<ir::TileType>(op->args_[0]->GetType()) != nullptr)
        << "block.tile_addr requires a Tile argument";
    const std::string tile = cce::GetOperandAsCode(codegen, op->args_[0]);
    return "(int64_t)((uint64_t)" + tile + ".data())";
}

REGISTER_BACKEND_OP(BackendCCE, "block.tile_addr").set_pipe(ir::PipeType::S).f_codegen(MakeTileAddrCodegenCCE);

// ============================================================================
// Language operations: get_block_num, get_subblock_idx
// ============================================================================

REGISTER_BACKEND_OP(BackendCCE, "get_block_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0)
            << "get_block_num requires no arguments";
        // pypto_block_meta.h accessor: the raw count is the container's grid under
        // SuperKernel; the accessor resolves SK vs raw builtin at compile time
        // (__PYPTO_SK_BLOCK__), so the generated source is identical for both builds.
        return std::string("(int64_t)(__pypto_block_num_value())");
    });

REGISTER_BACKEND_OP(BackendCCE, "get_subblock_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& /*codegen_base*/) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0)
            << "get_subblock_idx requires no arguments";
        return std::string("(int64_t)(get_subblockid())");
    });

REGISTER_BACKEND_OP(BackendCCE, "get_subblock_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0)
            << "get_subblock_num requires no arguments";
        // Matches AscendC GetTaskRation(): AIC returns 1, AIV returns get_subblockdim().
        auto& cg = dynamic_cast<codegen::CCECodegen&>(codegenBase);
        const auto target = cg.GetTarget();
        if (target == ir::SectionKind::Vector) {
            return std::string("(int64_t)(get_subblockdim())");
        }
        return std::string("(int64_t)(1)");
    });

// ============================================================================
// GetVal/SetVal Operations (unified: tile and tensor)
// ============================================================================

static std::string MakeGetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "getval requires 2 arguments, but got " << op->args_.size();

    auto resultType = ir::As<ir::ScalarType>(op->GetType());
    INTERNAL_CHECK(resultType) << "getval result must be ScalarType";
    auto castResultIfDtypeChanged = [&codegen](std::string result, const ir::DataType& resultDtype,
                                               const ir::DataType& inputDtype) {
        return codegen.ScalarCast(result, inputDtype, resultDtype);
    };

    auto firstType = op->args_[0]->GetType();
    if (auto tileType = ir::As<ir::TileType>(firstType)) {
        std::string offset = cce::GetScalarAccessOffset(codegen, op->args_[0], op->args_[1]);
        if (codegen.IsInSimtContext()) {
            std::string tile = codegen.GetExprAsCode(op->args_[0]);
            return castResultIfDtypeChanged(tile + "[" + offset + "]", resultType->dtype_, tileType->dtype_);
        }
        std::string tile = cce::GetOperandAsCode(codegen, op->args_[0]);
        return castResultIfDtypeChanged(tile + ".GetValue(" + offset + ")", resultType->dtype_, tileType->dtype_);
    }

    auto tensorVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorVar)
        << "getval requires tensor to be a Var";
    auto tensorType = ir::As<ir::TensorType>(tensorVar->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorType)
        << "getval requires TensorType";
    std::string tensorName = codegen.GetVarName(tensorVar);
    std::string offset = cce::GetScalarAccessOffset(codegen, op->args_[0], op->args_[1]);
    std::string dtypeStr = codegen.GetTypeString(tensorType->dtype_);

    std::string tensorPtr = codegen.GetPointer(tensorName);

    if (codegen.GetCurrentKernelType() == ir::FunctionType::SIMT_KERNEL) {
        return castResultIfDtypeChanged("(" + tensorPtr + ")[" + offset + "]", resultType->dtype_, tensorType->dtype_);
    }
    return castResultIfDtypeChanged("*((__gm__ " + dtypeStr + "*)" + tensorPtr + " + " + offset + ")",
                                    resultType->dtype_, tensorType->dtype_);
}

static std::string MakeWindowGetCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "dist.window_get requires 2 arguments, but got " << op->args_.size();

    // The window param has no local C++ variable (only its registered pointer), so the
    // pointer name must be resolved through the registry — never GetExprAsCode on the var.
    auto windowVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, windowVar)
        << "dist.window_get requires window to be a Var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::WindowType>(windowVar->GetType()))
        << "dist.window_get requires window to have WindowType";
    auto field = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, field != nullptr)
        << "dist.window_get field index must be a compile-time constant";

    const auto* member = dist::WindowScalarFieldName(field->value_);
    PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, member != nullptr)
        << "dist.window_get field index " << field->value_
        << " is not a scalar field (remote_ptrs array reads are not supported yet)";
    const std::string windowPtr = codegen.GetPointer(codegen.GetVarName(windowVar));
    return "((__gm__ pypto::dist::Window*)" + windowPtr + ")->" + member;
}

static std::string MakeSetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "setval requires 3 arguments, but got " << op->args_.size();

    auto firstType = op->args_[0]->GetType();
    auto scalarValue = [&](const ir::DataType& destination) {
        auto type = ir::As<ir::ScalarType>(op->args_[2]->GetType());
        INTERNAL_CHECK(type) << "setval value must be ScalarType";
        std::string value = codegen.GetExprAsCode(op->args_[2]);
        return codegen.ScalarCast(value, type->dtype_, destination);
    };
    if (auto tileType = ir::As<ir::TileType>(firstType)) {
        std::string offset = cce::GetScalarAccessOffset(codegen, op->args_[0], op->args_[1]);
        std::string value = scalarValue(tileType->dtype_);
        if (codegen.IsInSimtContext()) {
            std::string tile = codegen.GetExprAsCode(op->args_[0]);
            codegen.Emit(tile + "[" + offset + "] = " + value + ";");
            return "";
        }
        std::string tile = cce::GetOperandAsCode(codegen, op->args_[0]);
        codegen.Emit(tile + ".SetValue(" + offset + ", " + value + ");");
        return "";
    }

    auto tensorVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorVar)
        << "setval requires tensor to be a Var";
    auto tensorType = ir::As<ir::TensorType>(tensorVar->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorType)
        << "setval requires TensorType";
    std::string tensorName = codegen.GetVarName(tensorVar);
    std::string offset = cce::GetScalarAccessOffset(codegen, op->args_[0], op->args_[1]);
    std::string value = scalarValue(tensorType->dtype_);
    std::string dtypeStr = codegen.GetTypeString(tensorType->dtype_);

    std::string tensorPtr = codegen.GetPointer(tensorName);

    if (codegen.GetCurrentKernelType() == ir::FunctionType::SIMT_KERNEL) {
        codegen.Emit("(" + tensorPtr + ")[" + offset + "] = " + value + ";");
        return "";
    }
    codegen.Emit("*((__gm__ " + dtypeStr + "*)" + tensorPtr + " + " + offset + ") = " + value + ";");
    return "";
}

static std::string MakeTileValidShapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      op->args_.size() == 1 && ir::As<ir::TileType>(op->args_[0]->GetType()))
        << "block.tile_valid_shape requires one Tile argument";
    int axis = op->GetKwarg<int>("axis");
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, axis >= 0 && axis <= 1)
        << "block.tile_valid_shape axis must be in [0, 1]";

    if (auto dim = codegen.GetStaticTileValidDim(ir::As<ir::TileType>(op->args_[0]->GetType()), axis)) {
        return "(int64_t)(" + std::to_string(*dim) + ")";
    }
    if (codegen.IsInSimtContext() || codegen.IsInSimdContext()) {
        return "(int64_t)(" + codegen.GetVFTileValidShape(op->args_[0], axis) + ")";
    }
    return "(int64_t)(" + cce::GetOperandAsCode(codegen, op->args_[0]) +
           (axis == 0 ? ".GetValidRow())" : ".GetValidCol())");
}

// getval/setval use the "block." IR namespace like every other explicit-output
// block op (block.add, block.matmul, ...). This keeps codegen dispatch (keyed on
// op->name_) and the parser's auto_mutex pipe lookup (get_op_pipe -> "block.<name>")
// consistent, so getval/setval resolve to PIPE_S and participate in auto_mutex.
REGISTER_BACKEND_OP(BackendCCE, "block.getval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetValCodegenCCE(op, codegen); });

REGISTER_BACKEND_OP(BackendCCE, "block.setval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeSetValCodegenCCE(op, codegen); });

// Window metadata reads are scalar GM loads: PIPE_S, like getval.
REGISTER_BACKEND_OP(BackendCCE, "dist.window_get")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeWindowGetCodegenCCE(op, codegen);
    });

static std::string DistTimeoutLiteral(int64_t timeoutCycles)
{
    return timeoutCycles == 0 ? "INT64_MAX" : std::to_string(timeoutCycles) + "LL";
}

static std::string MakeDistBarrierCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "dist.barrier requires 1 argument, but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.barrier requires the Vector target (the barrier helper launches a SIMT VF)";

    auto windowVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, windowVar)
        << "dist.barrier requires window to be a Var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::WindowType>(windowVar->GetType()))
        << "dist.barrier requires window to have WindowType";

    const int64_t timeoutCycles = op->HasKwarg("timeout_cycles") ? op->GetKwarg<int>("timeout_cycles") : 0;
    const bool syncLocal = !op->HasKwarg("sync_local") || op->GetKwarg<bool>("sync_local");
    const bool flushStore = !op->HasKwarg("flush_store") || op->GetKwarg<bool>("flush_store");

    const std::string windowPtr = codegen.GetPointer(codegen.GetVarName(windowVar));
    codegen.Emit("pypto::dist::BarrierSync<" + DistTimeoutLiteral(timeoutCycles) + ", " +
                 (syncLocal ? "true" : "false") + ", " + (flushStore ? "true" : "false") +
                 ">(reinterpret_cast<__gm__ pypto::dist::Window*>(" + windowPtr + "));");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "dist.barrier")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistBarrierCodegenCCE(op, codegen);
    });

static std::string MakeDistNotifyCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "dist.notify requires 2 arguments, but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.notify requires the Vector target (the signal helpers run in the vector section)";

    const int notifyOp = op->HasKwarg("notify_op") ? op->GetKwarg<int>("notify_op") : 0;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, notifyOp == 0 || notifyOp == 1)
        << "dist.notify notify_op must be 0 (SET) or 1 (ATOMIC_ADD), but got " << notifyOp;
    const bool flushStore = !op->HasKwarg("flush_store") || op->GetKwarg<bool>("flush_store");

    auto ptrType = ir::As<ir::PtrType>(op->args_[0]->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ptrType)
        << "dist.notify signal address must have PtrType";
    const std::string dtypeStr = codegen.GetTypeString(ptrType->dtype_);

    const std::string addr = cce::GetOperandAsCode(codegen, op->args_[0]);
    const std::string val = codegen.GetExprAsCode(op->args_[1]);
    codegen.Emit("pypto::dist::Notify<" + dtypeStr + ", " + std::to_string(notifyOp) + ", " +
                 (flushStore ? "true" : "false") + ">(" + addr + ", static_cast<int64_t>(" + val + "));");
    return "";
}

static std::string MakeDistWaitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "dist.wait requires 2 arguments, but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.wait requires the Vector target (the wait helper spins on the vector pipe)";

    const int cmp = op->HasKwarg("cmp") ? op->GetKwarg<int>("cmp") : 0;
    PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, cmp >= 0 && cmp <= 5)
        << "dist.wait cmp must be a CmpMode member (0-5), but got " << cmp;
    const int64_t timeoutCycles = op->HasKwarg("timeout_cycles") ? op->GetKwarg<int>("timeout_cycles") : 0;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, timeoutCycles >= 0)
        << "dist.wait timeout_cycles must be >= 0 (0 = wait forever), but got " << timeoutCycles;

    auto ptrType = ir::As<ir::PtrType>(op->args_[0]->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ptrType)
        << "dist.wait signal address must have PtrType";
    const std::string dtypeStr = codegen.GetTypeString(ptrType->dtype_);

    const std::string addr = cce::GetOperandAsCode(codegen, op->args_[0]);
    const std::string val = codegen.GetExprAsCode(op->args_[1]);
    codegen.Emit("pypto::dist::Wait<" + dtypeStr + ", " + std::to_string(cmp) + ">(" + addr +
                 ", static_cast<int64_t>(" + val + "), " + std::to_string(timeoutCycles) + "LL);");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "dist.notify")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistNotifyCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.wait")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistWaitCodegenCCE(op, codegen);
    });

static std::string DistUbAddrLiteral(int64_t ubAddr)
{
    return "static_cast<uint64_t>(" + std::to_string(ubAddr) + "ULL)";
}

static std::string MakeDistCreateChannelCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "dist.channel requires 2 arguments, but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel requires the Vector target (the write engine runs on the AIV)";

    auto windowVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, windowVar)
        << "dist.channel requires window to be a Var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::WindowType>(windowVar->GetType()))
        << "dist.channel requires window to have WindowType";

    const int64_t ubAddr = op->HasKwarg("ub_addr") ? op->GetKwarg<int>("ub_addr") : 0;
    const std::string target = codegen.GetCurrentResultTarget();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, !target.empty())
        << "dist.channel must be the RHS of an assignment";

    const std::string windowPtr = codegen.GetPointer(codegen.GetVarName(windowVar));
    codegen.Emit("auto " + target + " = pypto::dist::Channel(reinterpret_cast<__gm__ pypto::dist::Window*>(" +
                 windowPtr + "), reinterpret_cast<__ubuf__ uint8_t*>(" + DistUbAddrLiteral(ubAddr) +
                 "), static_cast<int32_t>(" + codegen.GetExprAsCode(op->args_[1]) + "));");
    return "";
}

static std::string MakeDistChannelWriteCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "dist.channel_write requires 4 arguments [channel, dst_ptr, num_bytes, tensor], but got "
        << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_write requires the Vector target (the write engine runs on the AIV)";

    auto sourceVar = ir::As<ir::Var>(op->args_[3]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, sourceVar)
        << "dist.channel_write source must be a tensor Var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::TensorType>(sourceVar->GetType()))
        << "dist.channel_write source must have TensorType";

    const bool submit = op->HasKwarg("submit") ? op->GetKwarg<bool>("submit") : true;
    const int timeoutCycles = op->HasKwarg("timeout_cycles") ? op->GetKwarg<int>("timeout_cycles") : 0;

    const std::string channel = codegen.GetExprAsCode(op->args_[0]);
    const std::string dstPtr = cce::GetOperandAsCode(codegen, op->args_[1]);
    const std::string sourcePtr = codegen.GetPointer(codegen.GetVarName(sourceVar));
    codegen.Emit(channel + ".Write<" + DistTimeoutLiteral(timeoutCycles) + ", " + (submit ? "true" : "false") + ">(" +
                 "reinterpret_cast<GM_ADDR>(" + dstPtr + "), reinterpret_cast<__gm__ uint8_t*>(" + sourcePtr +
                 "), static_cast<int64_t>(" + codegen.GetExprAsCode(op->args_[2]) + "));");
    return "";
}

static std::string MakeDistChannelWriteIndexedCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 7)
        << "dist.channel_write_indexed requires 7 arguments [channel, dst_ptr, tensor, count, row_bytes, "
           "src_indices, dst_indices], but got "
        << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_write_indexed requires the Vector target (the write engine runs on the AIV)";

    auto tensorVar = ir::As<ir::Var>(op->args_[2]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorVar)
        << "dist.channel_write_indexed source must be a tensor Var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::TensorType>(tensorVar->GetType()))
        << "dist.channel_write_indexed source must have TensorType";

    const bool submit = op->HasKwarg("submit") ? op->GetKwarg<bool>("submit") : true;
    const int timeoutCycles = op->HasKwarg("timeout_cycles") ? op->GetKwarg<int>("timeout_cycles") : 0;

    // An index side is either a real Tile or the 0 sentinel; that is a compile-time
    // property, so the helper takes it as template parameters instead of runtime flags.
    // Both tables share the single IndexT of the device helper (IR enforces one dtype).
    const bool hasSrcIndex = ir::As<ir::TileType>(op->args_[5]->GetType()) != nullptr;
    const bool hasDstIndex = ir::As<ir::TileType>(op->args_[6]->GetType()) != nullptr;
    auto indexCType = [&op](size_t index) {
        auto tileType = ir::As<ir::TileType>(op->args_[index]->GetType());
        return tileType != nullptr ? tileType->dtype_.ToCTypeString() : std::string();
    };
    const std::string indexType = hasSrcIndex ? indexCType(5) : (hasDstIndex ? indexCType(6) : "int32_t");
    auto indexPtr = [&codegen, &op](size_t index, bool present, const std::string& type) {
        if (!present) {
            return std::string("nullptr");
        }
        return "reinterpret_cast<__ubuf__ " + type + "*>(" + cce::GetOperandAsCode(codegen, op->args_[index]) +
               ".data())";
    };

    const std::string channel = codegen.GetExprAsCode(op->args_[0]);
    const std::string dstPtr = cce::GetOperandAsCode(codegen, op->args_[1]);
    const std::string tensorPtr = codegen.GetPointer(codegen.GetVarName(tensorVar));
    const std::string commonArgs = "reinterpret_cast<GM_ADDR>(" + dstPtr + "), reinterpret_cast<__gm__ uint8_t*>(" +
                                   tensorPtr + "), " + indexPtr(5, hasSrcIndex, indexType) + ", " +
                                   indexPtr(6, hasDstIndex, indexType) + ", ";

    // The SIMT encoder builds every row's single-SGE WQEBB directly in the GM SQ;
    // only one row stride exists in the IR (contiguous rows).
    codegen.Emit(channel + ".WriteIndexed<" + indexType + ", " + (hasSrcIndex ? "true" : "false") + ", " +
                 (hasDstIndex ? "true" : "false") + ", " + DistTimeoutLiteral(timeoutCycles) + ", " +
                 (submit ? "true" : "false") + ">(" + commonArgs + "static_cast<uint64_t>(" +
                 codegen.GetExprAsCode(op->args_[4]) + "), static_cast<uint64_t>(" +
                 codegen.GetExprAsCode(op->args_[4]) + "), static_cast<uint64_t>(" +
                 codegen.GetExprAsCode(op->args_[4]) + "), static_cast<uint64_t>(" +
                 codegen.GetExprAsCode(op->args_[3]) + "));");
    return "";
}

static std::string MakeDistChannelWriteValueCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "dist.channel_write_value requires 3 arguments [channel, dst_ptr, value], but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_write_value requires the Vector target (the write engine runs on the AIV)";

    const bool submit = op->HasKwarg("submit") ? op->GetKwarg<bool>("submit") : true;
    const int timeoutCycles = op->HasKwarg("timeout_cycles") ? op->GetKwarg<int>("timeout_cycles") : 0;
    auto dstPtrType = ir::As<ir::PtrType>(op->args_[1]->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, dstPtrType != nullptr)
        << "dist.channel_write_value destination must be a pointer";
    const std::string dtypeStr = codegen.GetTypeString(dstPtrType->dtype_);

    const std::string channel = codegen.GetExprAsCode(op->args_[0]);
    const std::string dstPtr = cce::GetOperandAsCode(codegen, op->args_[1]);
    codegen.Emit(channel + ".WriteValue<" + DistTimeoutLiteral(timeoutCycles) + ", " + dtypeStr + ", " +
                 (submit ? "true" : "false") + ">(reinterpret_cast<GM_ADDR>(" + dstPtr + "), static_cast<" + dtypeStr +
                 ">(" + codegen.GetExprAsCode(op->args_[2]) + "));");
    return "";
}

static std::string MakeDistChannelWriteNotifyCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 6)
        << "dist.channel_write_notify requires 6 arguments [channel, dst_ptr, num_bytes, tensor, signal_ptr, val], but "
           "got "
        << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_write_notify requires the Vector target (the write engine runs on the AIV)";

    auto sourceVar = ir::As<ir::Var>(op->args_[3]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, sourceVar)
        << "dist.channel_write_notify source must be a tensor Var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::TensorType>(sourceVar->GetType()))
        << "dist.channel_write_notify source must have TensorType";

    const int timeoutCycles = op->HasKwarg("timeout_cycles") ? op->GetKwarg<int>("timeout_cycles") : 0;

    const std::string channel = codegen.GetExprAsCode(op->args_[0]);
    const std::string dstPtr = cce::GetOperandAsCode(codegen, op->args_[1]);
    const std::string sourcePtr = codegen.GetPointer(codegen.GetVarName(sourceVar));
    const std::string signalPtr = cce::GetOperandAsCode(codegen, op->args_[4]);
    codegen.Emit(channel + ".WriteNotify<" + DistTimeoutLiteral(timeoutCycles) + ">(reinterpret_cast<GM_ADDR>(" +
                 dstPtr + "), reinterpret_cast<__gm__ uint8_t*>(" + sourcePtr + "), static_cast<int64_t>(" +
                 codegen.GetExprAsCode(op->args_[2]) + "), reinterpret_cast<__gm__ uint8_t*>(" + signalPtr +
                 "), static_cast<uint64_t>(" + codegen.GetExprAsCode(op->args_[5]) + "));");
    return "";
}

static std::string MakeDistChannelSubmitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "dist.channel_submit requires 1 argument, but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_submit requires the Vector target";

    codegen.Emit(codegen.GetExprAsCode(op->args_[0]) + ".Submit();");
    return "";
}

static std::string MakeDistChannelDrainCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "dist.channel_drain requires 1 argument, but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_drain requires the Vector target";

    const int timeoutCycles = op->HasKwarg("timeout_cycles") ? op->GetKwarg<int>("timeout_cycles") : 0;
    codegen.Emit(codegen.GetExprAsCode(op->args_[0]) + ".Drain<" + DistTimeoutLiteral(timeoutCycles) + ">();");
    return "";
}

static std::string MakeDistChannelIsExclusiveCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "dist.channel_is_exclusive requires 1 argument, but got " << op->args_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_is_exclusive requires the Vector target";
    return codegen.GetExprAsCode(op->args_[0]) + ".IsExclusive()";
}

REGISTER_BACKEND_OP(BackendCCE, "dist.channel")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistCreateChannelCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_write")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelWriteCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_write_indexed")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelWriteIndexedCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_write_value")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelWriteValueCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_write_notify")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelWriteNotifyCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_submit")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelSubmitCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_drain")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelDrainCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_is_exclusive")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelIsExclusiveCodegenCCE(op, codegen);
    });

// Typed pointer to a registered region of a peer's window. Returns the pointer
// expression so the result var maps to it (a following ptr.addptr / ptr.make_tensor
// consumes it), like the ptr.* ops. Only the UB_MEM plane is directly addressable.
static std::string MakeDistRemotePtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "dist.remote_ptr requires 3 arguments: window, ptr, peer";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.remote_ptr requires the Vector target (the pointer is an AIV address)";

    auto windowVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, windowVar)
        << "dist.remote_ptr requires window to be a Var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::WindowType>(windowVar->GetType()))
        << "dist.remote_ptr requires window to have WindowType";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::PtrType>(op->args_[1]->GetType()) != nullptr)
        << "dist.remote_ptr requires a pointer operand";
    auto ptrType = ir::As<ir::PtrType>(op->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ptrType != nullptr)
        << "dist.remote_ptr must produce a PtrType";
    const std::string dtypeStr = ptrType->dtype_.ToCTypeString();

    const std::string windowPtr = codegen.GetPointer(codegen.GetVarName(windowVar));
    const std::string window = "reinterpret_cast<__gm__ pypto::dist::Window*>(" + windowPtr + ")";
    return "reinterpret_cast<__gm__ " + dtypeStr + "*>(pypto::dist::RemotePtr(" + window +
           ", reinterpret_cast<int64_t>(" + cce::GetOperandAsCode(codegen, op->args_[1]) + "), static_cast<int32_t>(" +
           codegen.GetExprAsCode(op->args_[2]) + ")))";
}

REGISTER_BACKEND_OP(BackendCCE, "dist.remote_ptr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistRemotePtrCodegenCCE(op, codegen);
    });

static std::string MakeDistChannelRemotePtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "dist.channel_remote_ptr requires 2 arguments: channel, ptr";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, codegen.GetTarget() == ir::SectionKind::Vector)
        << "dist.channel_remote_ptr requires the Vector target (the pointer is an AIV address)";

    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                               ir::As<ir::PtrType>(op->args_[1]->GetType()) != nullptr)
        << "dist.channel_remote_ptr requires a pointer operand";
    auto ptrType = ir::As<ir::PtrType>(op->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ptrType != nullptr)
        << "dist.channel_remote_ptr must produce a PtrType";
    const std::string dtypeStr = ptrType->dtype_.ToCTypeString();

    return codegen.GetExprAsCode(op->args_[0]) + ".RemoteAddr<" + dtypeStr + ">(reinterpret_cast<int64_t>(" +
           cce::GetOperandAsCode(codegen, op->args_[1]) + "))";
}

REGISTER_BACKEND_OP(BackendCCE, "dist.channel_remote_ptr")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeDistChannelRemotePtrCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "block.tile_valid_shape")
    .set_pipe(ir::PipeType::S)
    .f_codegen(MakeTileValidShapeCodegenCCE);

// ============================================================================
// block.subview - Tile sub-view with offset and new shape.
//
// Tile in VF section: returns pointer arithmetic expression.
// Tile in non-VF section: emits TASSIGN with new shape, returns the new tile
// variable name.
// ============================================================================
static std::string MakeBlockSubviewCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);

    auto tileType = ir::As<ir::TileType>(op->args_[0]->GetType());
    auto offsetTuple = ir::As<ir::MakeTuple>(op->args_[1]);
    const bool isVf = codegen.IsInSimdContext();

    // VF also uses block.subview for tile + linear_offset.
    if (offsetTuple == nullptr) {
        CHECK(isVf) << "block.subview offset must be a 2D tuple outside VF linear-offset access";
        const std::string basePtr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/false);
        return "(" + basePtr + " + (" + codegen.GetExprAsCode(op->args_[1]) + "))";
    }

    const std::string rowExpr = cce::GetOperandAsCode(codegen, offsetTuple->elements_[0]);
    const std::string colExpr = cce::GetOperandAsCode(codegen, offsetTuple->elements_[1]);
    const std::string baseTile = cce::GetOperandAsCode(codegen, op->args_[0]);
    const std::string elemOff = "pto::GetTileOffset<std::remove_reference_t<decltype(" + baseTile + ")>>(" + rowExpr +
                                ", " + colExpr + ")";

    // VF only needs pointer arithmetic and does not create a Tile descriptor.
    if (isVf) {
        const std::string basePtr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/false);
        return "(" + basePtr + " + " + elemOff + ")";
    }

    const int dtypeBits = static_cast<int>(tileType->dtype_.GetBit());
    const std::string byteOffsetExpr = "((" + elemOff + ")*" + std::to_string(dtypeBits) + "/8)";

    const std::string baseAddr = "(uint64_t)" + baseTile + ".data()";

    // Sub-window valid_shape from args[2] (computed by parser as the intersection
    // of slice size and original valid_shape - start).
    auto shapeTuple = ir::As<ir::MakeTuple>(op->args_[2]);
    std::string vsRow = cce::GetOperandAsCode(codegen, shapeTuple->elements_[0]);
    std::string vsCol = cce::GetOperandAsCode(codegen, shapeTuple->elements_[1]);

    // Build type with original shape (preserves row_stride) but WITHOUT tileView_,
    // so valid_shape template params are -1 (dynamic). This is required because
    // subview calls SetValidShape at runtime, which only works when the template
    // valid_shape params are DYNAMIC (-1).
    auto subviewType = ir::As<ir::TileType>(op->GetType());

    // Sanitize base_tile into a valid C++ identifier prefix
    std::string baseTileId = baseTile;
    for (char& c : baseTileId)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
            c = '_';
    std::string tempName = codegen.GetCurrentResultTarget().empty() ?
                               (baseTileId + "_view_" + std::to_string(codegen.GetTileOffsetCounter())) :
                               codegen.GetCurrentResultTarget();

    // The descriptor and its binding are created at the source value's execution point.
    std::string tempAddr = baseAddr + " + " + byteOffsetExpr;
    const std::string descriptor = tempName + ".tile";
    DeclareTile(codegen, subviewType, tempName, vsRow + ", " + vsCol, InheritedMutexIds(codegen, op->args_[0]));
    codegen.Emit("TASSIGN(" + descriptor + ", " + tempAddr + ");");
    codegen.Emit(descriptor + ".SetValidShape(" + vsRow + ", " + vsCol + ");");

    return tempName;
}

REGISTER_BACKEND_OP(BackendCCE, "block.subview")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        return MakeBlockSubviewCodegenCCE(op, codegenBase);
    });

// ============================================================================
// Mutex (Buffer-ID Token) - 3510 CCE Codegen
// ----------------------------------------------------------------------------
// Lowers system.mutex_lock/unlock to CCE intrinsics get_buf/rls_buf.
// API: get_buf(PIPE_MTE2, mutexId, 0);  rls_buf(PIPE_MTE2, mutexId, 0);
// ============================================================================

static std::string MakeMutexBufCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase,
                                          const std::string& intrinsic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "dynamic mutex requires one scalar mutex ID";
    codegen.Emit(intrinsic + "(" + PipeTypeToCCEString(pipe) + ", " + cce::GetOperandAsCode(codegen, op->args_[0]) +
                 ", 0);");
    return "";
}

static std::string MakeMutexLockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return MakeMutexBufCodegenCCE(op, codegenBase, "get_buf");
}

static std::string MakeMutexUnlockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    return MakeMutexBufCodegenCCE(op, codegenBase, "rls_buf");
}

REGISTER_BACKEND_OP(BackendCCE, "system.mutex_lock_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeMutexLockDynCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.mutex_unlock_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeMutexUnlockDynCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "system.vf_scope_args").f_codegen([](const ir::CallPtr&, codegen::CodegenBase&) {
    return std::string{};
});

// ============================================================================
// Global Core Synchronization (sync_all)
// ============================================================================
// Use AscendC for AIV/MIX so ordinary and SK entries share its synchronization protocol.
// AIC-only retains PTO's dedicated barrier; AscendC::SyncAll<false>() is a MIX barrier.

static std::string MakeSystemSyncAllCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);

    auto coreType = static_cast<ir::SyncCoreType>(op->HasKwarg("core_type") ? op->GetKwarg<int>("core_type") : 2);

    if (coreType == ir::SyncCoreType::AIC_ONLY) {
        codegen.Emit("SYNCALL<SyncCoreType::AICOnly>();");
    } else {
        codegen.Emit(coreType == ir::SyncCoreType::AIV_ONLY ? "AscendC::SyncAll();" : "AscendC::SyncAll<false>();");
    }
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSystemSyncAllCodegenCCE(op, codegen);
    });

static std::string MakeSystemDsbCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);

    int scopeInt = op->HasKwarg("scope") ? op->GetKwarg<int>("scope") : 0; // MemDsb::ALL
    const auto scope = static_cast<ir::MemDsb>(scopeInt);

    // MemDsb members drop the toolchain's DSB_ prefix; prepend it to recover the
    // intrinsic token: MemDsb::DDR -> dsb(DSB_DDR).
    codegen.Emit("dsb(DSB_" + EnumValueName(ir::EnumToString(scope)) + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.dsb").set_pipe(ir::PipeType::S).f_codegen(MakeSystemDsbCodegenCCE);

// ============================================================================
// Early-start inter-task sync (wait_pre_task_end)
// ============================================================================
// Delegates to AscendC::WaitPreTaskEnd(); effective only when the kernel is compiled
// with -D__ASCENDC_SUPERKERNEL_EARLY_START_V3. The __sk__ entry must propagate
// sysArgs->skTaskSyncCfg into g_super_kernel_early_start_config (see jit.py).

REGISTER_BACKEND_OP(BackendCCE, "system.wait_pre_task_end")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        auto& cceCodegen = dynamic_cast<codegen::CCECodegen&>(codegen);
        bool force = op->HasKwarg("force") ? op->GetKwarg<bool>("force") : false;
        cceCodegen.Emit("#if defined(__PYPTO_ACLNN__)");
        cceCodegen.Emit("AscendC::WaitPreTaskEnd();");
        cceCodegen.Emit("#else");
        cceCodegen.Emit("#if defined(__ASCENDC_ENABLE_WAIT_PRE_TASK_END)");
        cceCodegen.Emit(force ? "AscendC::WaitPreTaskEnd<true>();" : "AscendC::WaitPreTaskEnd();");
        cceCodegen.Emit("#endif");
        cceCodegen.Emit("#endif");
        return "";
    });

// Early-start inter-task sync (set_next_task_start, dual of wait_pre_task_end):
// instructions after the call overlap the *next* sub-kernel. FORCE is SetNextTaskStart's
// third template parameter (after the AIV/AIC pipe selectors); the 3510 (__NPU_ARCH__ not
// 1001/2002) defaults are PIPE_MTE3 / PIPE_FIX, so the forced form spells them out.
REGISTER_BACKEND_OP(BackendCCE, "system.set_next_task_start")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        auto& cceCodegen = dynamic_cast<codegen::CCECodegen&>(codegen);
        bool force = op->HasKwarg("force") ? op->GetKwarg<bool>("force") : false;
        cceCodegen.Emit("#if defined(__PYPTO_ACLNN__)");
        cceCodegen.Emit("AscendC::SetNextTaskStart();");
        cceCodegen.Emit("#else");
        cceCodegen.Emit("#if defined(__ASCENDC_ENABLE_SET_NEXT_TASK_START)");
        cceCodegen.Emit(force ? "AscendC::SetNextTaskStart<PIPE_MTE3, PIPE_FIX, true>();" :
                                "AscendC::SetNextTaskStart();");
        cceCodegen.Emit("#endif");
        cceCodegen.Emit("#endif");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mm_layout_transform")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegenBase) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
        auto enabled = op->GetKwarg<int>("enabled");
        // Direct register manipulation: MM_LAYOUT_MODE_BIT = 51
        if (enabled) {
            codegen.Emit("set_ctrl(sbitset1(get_ctrl(), 51));");
        } else {
            codegen.Emit("set_ctrl(sbitset0(get_ctrl(), 51));");
        }
        return "";
    });

} // namespace backend
} // namespace pypto
