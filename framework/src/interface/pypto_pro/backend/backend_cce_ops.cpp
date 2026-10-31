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
    static int next_debug_dump_id = 0;
    return next_debug_dump_id++;
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

static void EmitDebugLocationHeaderCCE(codegen::CCECodegen& codegen, const ir::Span& span, const std::string& op_name)
{
    std::string header = debug_printf::FormatDebugLocationHeader(span, op_name);
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

static std::string RewritePtoPrintfFormatForScalarType(const std::string& format_segment, char conversion,
                                                       const DataType& dtype)
{
    const bool signed_long_long = NeedsPtoPrintfSignedLongLong(dtype, conversion);
    const bool unsigned_long_long = NeedsPtoPrintfUnsignedLongLong(dtype, conversion);
    if (!signed_long_long && !unsigned_long_long) {
        return format_segment;
    }

    size_t conv_idx = debug_printf::FindPrintfConversionIndex(format_segment);
    std::string rewritten = format_segment;
    if (conversion == 'd') {
        rewritten.replace(conv_idx, 1, "lld");
    } else if (conversion == 'i') {
        rewritten.replace(conv_idx, 1, "lli");
    } else if (conversion == 'u') {
        rewritten.replace(conv_idx, 1, "llu");
    } else {
        rewritten.replace(conv_idx, 1, "llx");
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
                                      const std::vector<DataType>& arg_dtypes)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == arg_dtypes.size())
        << "debug.printf argument/type count mismatch";

    auto segments = debug_printf::ParsePrintfSegments(format);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, segments.size() == args.size())
        << "debug.printf format expects " << segments.size() << " scalar arguments, but got " << args.size();

    if (format.empty()) {
        return "";
    }

    std::string rewritten_format = segments.empty() ? format : "";
    std::vector<std::string> rewritten_args;
    for (size_t i = 0; i < segments.size(); ++i) {
        rewritten_format += RewritePtoPrintfFormatForScalarType(segments[i].format_segment, segments[i].conversion,
                                                                arg_dtypes[i]);
        rewritten_args.push_back(CastPtoPrintfArgIfNeeded(args[i], arg_dtypes[i], segments[i].conversion));
    }

    std::string call = "pypto::aicore::printf(\"" + debug_printf::EscapeStringLiteral(rewritten_format) + "\"";
    for (const auto& arg : rewritten_args) {
        call += ", " + arg;
    }
    return call + ");";
}

static std::vector<std::string> BuildTensorDumpStride(codegen::CCECodegen& codegen,
                                                      const ir::TensorTypePtr& tensor_type)
{
    const size_t rank = tensor_type->shape_.size();
    std::vector<std::string> stride(rank, "1");
    if (cce::IsNZTensorType(tensor_type)) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, rank >= 2)
            << "debug.dump_tensor NZ lowering requires a tensor rank of at least 2";
        const size_t row_axis = rank - 2;
        const int64_t c0 = cce::GetNZInnerCols(tensor_type->dtype_);
        const std::string padded_rows = codegen.ComputeAlignedShapeDimension(tensor_type->shape_[row_axis], 16);
        const std::string padded_cols = codegen.ComputeAlignedShapeDimension(tensor_type->shape_[rank - 1], c0);
        for (size_t axis = 0; axis < row_axis; ++axis) {
            std::vector<std::string> factors;
            for (size_t inner = axis + 1; inner < row_axis; ++inner) {
                factors.push_back(codegen.GetExprAsCode(tensor_type->shape_[inner]));
            }
            factors.push_back(padded_rows);
            factors.push_back(padded_cols);
            stride[axis] = "(" + JoinExpressions(factors, " * ") + ")";
        }
        stride[row_axis] = padded_rows;
        stride[rank - 1] = std::to_string(c0);
        return stride;
    }

    return codegen.BuildTensorStrideExpressions(tensor_type);
}

static void EmitDumpMetadataCCE(codegen::CCECodegen& codegen, const ir::CallPtr& op, const std::string& op_name)
{
    if (op->GetKwarg<bool>("show_location", false)) {
        EmitDebugLocationHeaderCCE(codegen, op->span_, op_name);
    }
    std::string dump_flag = op->HasKwarg("dump_flag") ? op->GetKwarg<std::string>("dump_flag") : "";
    if (dump_flag.empty()) {
        return;
    }
    codegen.Emit("pypto::aicore::printf(\"[flag] %s\\n\", \"" + debug_printf::EscapeStringLiteral(dump_flag) + "\");");
}

static std::string MakeDebugDumpTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 3)
        << "debug.dump_tensor requires 1 or 3 arguments, but got " << op->args_.size();
    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_var) << "debug.dump_tensor first argument must be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_type)
        << "debug.dump_tensor first argument must be TensorType";
    const int debug_id = NextDebugDumpId();
    const std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string base_ptr = codegen.GetPointer(tensor_name);
    if (base_ptr.empty()) {
        base_ptr = tensor_name + ".data()";
    }
    const std::string metadata_prefix = "__debug_dump_tensor_" + std::to_string(debug_id);
    const std::string source_shape_name = metadata_prefix + "_source_shape";
    const std::string offsets_name = metadata_prefix + "_offsets";
    const std::string shape_name = metadata_prefix + "_shape";
    const std::string stride_name = metadata_prefix + "_stride";
    const size_t rank = tensor_type->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, rank >= 1 && rank <= 5)
        << "debug.dump_tensor currently supports tensor rank 1..5, but got " << rank;
    const auto get_expr_codes = [&codegen](const std::vector<ir::ExprPtr>& expressions) {
        std::vector<std::string> result;
        result.reserve(expressions.size());
        for (const auto& expression : expressions) {
            result.push_back(codegen.GetExprAsCode(expression));
        }
        return result;
    };
    std::vector<std::string> offsets(rank, "0");
    std::vector<ir::ExprPtr> shape_exprs = tensor_type->shape_;
    const bool is_nz = cce::IsNZTensorType(tensor_type);
    if (op->args_.size() == 3) {
        auto offsets_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, offsets_tuple)
            << "debug.dump_tensor second argument must be a tuple (offsets)";
        auto shapes_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, shapes_tuple)
            << "debug.dump_tensor third argument must be a tuple (shapes)";
        offsets = get_expr_codes(offsets_tuple->elements_);
        shape_exprs = shapes_tuple->elements_;
        if (is_nz) {
            const size_t col_axis = tensor_type->shape_.size() - 1;
            const int64_t c0 = cce::GetNZInnerCols(tensor_type->dtype_);
            const auto known_col_offset = ir::GetConstantDimension(offsets_tuple->elements_[col_axis]);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT,
                              !known_col_offset.has_value() || known_col_offset.value() % c0 == 0)
                << "debug.dump_tensor: NZ column offset must be divisible by C0";
        }
    }
    const auto source_shape = get_expr_codes(tensor_type->shape_);
    const auto shape = get_expr_codes(shape_exprs);
    const auto stride = BuildTensorDumpStride(codegen, tensor_type);

    const std::string layout_name = is_nz ? "NZ" : "ND";
    const std::string layout = "pypto::aicore::DumpLayout::" + layout_name;

    EmitDumpMetadataCCE(codegen, op, "dump_tensor");
    const std::string rank_str = std::to_string(rank);
    codegen.Emit("int64_t " + source_shape_name + "[" + rank_str + "] = {" + JoinExpressions(source_shape, ", ") +
                 "};");
    codegen.Emit("int64_t " + offsets_name + "[" + rank_str + "] = {" + JoinExpressions(offsets, ", ") + "};");
    codegen.Emit("int64_t " + shape_name + "[" + rank_str + "] = {" + JoinExpressions(shape, ", ") + "};");
    codegen.Emit("int64_t " + stride_name + "[" + rank_str + "] = {" + JoinExpressions(stride, ", ") + "};");
    const std::string view_name = metadata_prefix + "_view";
    const std::string inner_rows = is_nz ? "16" : "1";
    const std::string inner_cols = is_nz ? stride_name + "[" + std::to_string(rank - 1) + "]" : "1";
    codegen.Emit("const pypto::aicore::DumpTensorView " + view_name + " = {" + source_shape_name + ", " + offsets_name +
                 ", " + shape_name + ", " + stride_name + ", " + rank_str + "U, " + inner_rows + ", " + inner_cols +
                 "};");
    codegen.Emit("PYPTO_DUMP_TENSOR(" + layout + ", " + rank_str + ", " + base_ptr + ", " + std::to_string(debug_id) +
                 ", " + view_name + ");");
    return "";
}

static std::string MakeDebugPrintfCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    std::string format = op->GetKwarg<std::string>("format");
    if (op->GetKwarg<bool>("show_location", false)) {
        std::string location = debug_printf::FormatDebugLocation(op->span_);
        if (!location.empty()) {
            format = location + " " + format;
        }
    }

    std::vector<std::string> args;
    std::vector<DataType> arg_dtypes;
    args.reserve(op->args_.size());
    arg_dtypes.reserve(op->args_.size());

    for (size_t i = 0; i < op->args_.size(); ++i) {
        args.emplace_back(codegen.GetExprAsCode(op->args_[i]));
        if (ir::As<ir::PtrType>(op->args_[i]->GetType())) {
            // Pointer arguments are passed directly to PTO printf.
            arg_dtypes.emplace_back(DataType::INT64);
        } else {
            auto scalar_type = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_type)
                << "debug.printf argument must be ScalarType in CCE lowering";
            arg_dtypes.emplace_back(scalar_type->dtype_);
        }
    }

    std::string call = BuildPtoPrintfCall(format, args, arg_dtypes);
    if (!call.empty()) {
        codegen.Emit(call);
    }
    return "";
}

static std::string MakeDebugDumpTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 3)
        << "debug.dump_tile requires 1 argument (tile) or 3 arguments (tile, offsets, shapes), but got "
        << op->args_.size();
    const std::string src = cce::GetOperandAsCode(codegen, op->args_[0]);
    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_type) << "debug.dump_tile first argument must be TileType";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, tile_type->shape_.size() == 2)
        << "debug.dump_tile CCE lowering currently only supports 2D tiles";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE,
                      ir::As<ir::ConstInt>(tile_type->shape_[0]) && ir::As<ir::ConstInt>(tile_type->shape_[1]))
        << "debug.dump_tile CCE lowering requires static physical tile shape";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, tile_type->memref_.has_value())
        << "debug.dump_tile requires tile to have a memref";

    std::string row_offset = "0";
    std::string col_offset = "0";
    std::string rows = codegen.GetExprAsCode(tile_type->shape_[0]);
    std::string cols = codegen.GetExprAsCode(tile_type->shape_[1]);
    if (op->args_.size() == 3) {
        auto offsets = ir::As<ir::MakeTuple>(op->args_[1]);
        auto shapes = ir::As<ir::MakeTuple>(op->args_[2]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, offsets && shapes)
            << "debug.dump_tile offsets and shapes must be tuples";
        row_offset = codegen.GetExprAsCode(offsets->elements_[0]);
        col_offset = codegen.GetExprAsCode(offsets->elements_[1]);
        rows = codegen.GetExprAsCode(shapes->elements_[0]);
        cols = codegen.GetExprAsCode(shapes->elements_[1]);
    }

    const int debug_id = NextDebugDumpId();
    const auto tile_layout = cce::GetTileTensorLayout(tile_type->hardwareInfo_.value_or(ir::HardwareInfo{}));
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, tile_layout.has_value())
        << "debug.dump_tile has an unsupported Tile layout";
    const std::string layout_name = ir::EnumToString(tile_layout.value());
    const std::string layout = "pypto::aicore::DumpLayout::" + layout_name.substr(layout_name.find("::") + 2);

    EmitDumpMetadataCCE(codegen, op, "dump_tile");
    codegen.Emit("PYPTO_DUMP_TILE_WINDOW(" + layout + ", " +
                 JoinExpressions({src, std::to_string(debug_id), row_offset, col_offset, rows, cols}, ", ") + ");");
    return "";
}

// Helper function for get_block_idx (returns value expression).
// Emits the pypto_block_meta.h accessor, which keeps the AscendC GetBlockIdx() contract
// (AIV: global AIV index; AIC: AIC index) and resolves to the SuperKernel per-core
// record or the raw builtin at compile time (__PYPTO_SK_BLOCK__), so the generated
// source is identical for both builds.
static std::string MakeBlockGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0) << "get_block_idx requires no arguments";
    (void)codegen_base;
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
static std::string MakeBlockDynamicInputPtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "dynamic_input_ptr requires 2 arguments";
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string x = cg.GetExprAsCode(op->args_[0]);
    std::string i = cg.GetExprAsCode(op->args_[1]);
    auto ptr_type = ir::As<ir::PtrType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ptr_type != nullptr) << "dynamic_input_ptr result must be a PtrType";
    std::string ctype = ptr_type->dtype_.ToCTypeString();
    // Unmapped dtypes (DT_INT4/DT_UINT4/DT_FP4/DT_HF4) would fall back to "unknown"; fail loudly
    // instead of emitting a broken `__gm__ unknown*`.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ctype != "unknown")
        << "dynamic_input_ptr: descriptor dtype " << ptr_type->dtype_.ToString()
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
static std::string MakeBlockDynamicInputDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "dynamic_input_dim requires 3 arguments";
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
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
static std::string MakeBlockCreateMutexTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
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
static std::string MakeBlockMakeTensorCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    const std::string name = cg.GetCurrentResultTarget();
    // The source of the view is either a raw pointer (PtrType) or an existing tensor (TensorType,
    // re-viewed with a new shape/stride). Resolve the base pointer accordingly and record the
    // source element dtype.
    std::string ptr_code;
    ir::DataType source_dtype;
    if (auto ptr_type = ir::As<ir::PtrType>(op->args_[0]->GetType())) {
        ptr_code = cce::GetOperandAsCode(cg, op->args_[0]);
        source_dtype = ptr_type->dtype_;
    } else if (auto src_tensor_type = ir::As<ir::TensorType>(op->args_[0]->GetType())) {
        // Re-view of an existing tensor: reuse its already-registered base pointer (a function
        // parameter's "<name>_ptr" or an earlier make_tensor view's pointer).
        auto src_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_var != nullptr)
            << "ptr.make_tensor from a tensor requires the source to be a tensor variable";
        const std::string src_name = cg.GetVarName(src_var);
        ptr_code = cg.HasPointer(src_name) ? cg.GetPointer(src_name) : (src_name + ".data()");
        source_dtype = src_tensor_type->dtype_;
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
            << "ptr.make_tensor source must be a PtrType or TensorType";
    }
    // The view's element dtype may differ from the source's element dtype (e.g. a raw uint8
    // pointer reinterpreted as an fp16 view via ptr.make_tensor(..., dtype=FP16)). The
    // GlobalTensor<element_type> instance is constructed from this pointer, so reinterpret-cast
    // the base pointer to the view element type when the dtypes differ (a no-op when they match).
    auto tensor_type = ir::As<ir::TensorType>(op->GetType());
    if (tensor_type && !(source_dtype == tensor_type->dtype_)) {
        ptr_code = "(__gm__ " + tensor_type->dtype_.ToCTypeString() + "*)(" + ptr_code + ")";
    }
    cg.RegisterPointer(name, ptr_code);
    for (const codegen::TensorDef* def : cg.GetTensorDefs(name)) {
        cg.GenerateGlobalTensorTypeDeclaration(*def);
    }
    return "";
}

// Helper for ptr.addptr / advancing a raw pointer by an element offset. Emits no
// statement; returns the pointer-arithmetic expression so the result var maps to
// it (used as a base address by a subsequent ptr.make_tensor).
static std::string MakePtrAddPtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "ptr.addptr requires 2 arguments: ptr, offset";
    std::string ptr = cce::GetOperandAsCode(codegen, op->args_[0]);
    std::string offset = cce::GetOperandAsCode(codegen, op->args_[1]);
    return "(" + ptr + " + " + offset + ")";
}

// Helper for ptr.make_ptr / reinterpreting a raw pointer (or extracting a pointer from a
// tensor) as a different element type. Emits no statement; returns the expression so the
// result var maps to it (used as a base address by a subsequent ptr.addptr / ptr.make_tensor).
static std::string MakePtrMakePtrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "ptr.make_ptr requires 1 argument: ptr";
    std::string ptr;
    if (auto ptr_type = ir::As<ir::PtrType>(op->args_[0]->GetType())) {
        ptr = cce::GetOperandAsCode(codegen, op->args_[0]);
    } else if (ir::As<ir::TensorType>(op->args_[0]->GetType())) {
        auto src_var = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, src_var != nullptr)
            << "ptr.make_ptr from a tensor requires the source to be a tensor variable";
        const std::string src_name = codegen.GetVarName(src_var);
        ptr = codegen.HasPointer(src_name) ? codegen.GetPointer(src_name) : (src_name + ".data()");
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "ptr.make_ptr source must be a PtrType or TensorType";
    }
    auto result_ptr_type = ir::As<ir::PtrType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, result_ptr_type != nullptr)
        << "ptr.make_ptr result must be a PtrType";
    return "((__gm__ " + result_ptr_type->dtype_.ToCTypeString() + "*)(" + ptr + "))";
}

static std::string MakeBlockReinterpretCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(base);
    auto tile_type = ir::As<ir::TileType>(op->GetType());
    const std::string target = codegen.GetCurrentResultTarget();
    PRO_CODEGEN_IRCHECK(ExternalError::INVALID_OPERATION, !target.empty())
        << "reinterpret must be assigned to an IR value";
    std::string source_tile = cce::GetOperandAsCode(codegen, op->args_[0]);
    DeclareTile(codegen, tile_type, target, TileConstructorArgs(codegen, tile_type),
                InheritedMutexIds(codegen, op->args_[0]));
    codegen.Emit("TASSIGN(" + target + ".tile, (uint64_t)" + source_tile + ".data());");
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
static std::string MakeGetSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)codegen_base;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0) << "get_spr requires no arguments";
    return "get_ar()";
}

REGISTER_BACKEND_OP(BackendCCE, "get_spr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) { return MakeGetSprCodegenCCE(op, codegen); });

// Helper function for get_cycle (reads the system cycle counter via get_sys_cnt())
static std::string MakeGetCycleCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)codegen_base;
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

static int8_t GetSaturationModeBit(ir::SaturationFlagMode mode, const std::string& op_name)
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
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false) << op_name << ": unsupported mode";
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

static std::string MakeSetSaturationFlagCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
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
    std::string set_fn;
    if (inverted) {
        set_fn = enable ? "sbitset0" : "sbitset1";
    } else {
        set_fn = enable ? "sbitset1" : "sbitset0";
    }
    codegen.Emit("set_ctrl(" + set_fn + "(get_ctrl(), " + std::to_string(bit) + "));");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "set_saturation_flag")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSetSaturationFlagCodegenCCE(op, codegen);
    });

static std::string MakeGetSaturationFlagCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)codegen_base;
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

// A5 writable CTRL bits: 6-10 (range), 45, 48, 50, 53, 59, 60 (single bits)
static void CheckCtrlBitRange(int8_t startBit, int8_t endBit, const std::string& op_name)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, startBit >= 0 && endBit < 64 && startBit <= endBit)
        << op_name << ": invalid bit range [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit)
        << "], must be 0 <= startBit <= endBit < 64";
    bool valid = (6 <= startBit && startBit <= 10 && 6 <= endBit && endBit <= 10) ||
                 (startBit == endBit && (startBit == 45 || startBit == 48 || startBit == 50 || startBit == 53 ||
                                         startBit == 59 || startBit == 60));
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, valid)
        << op_name << ": bits [" << static_cast<int>(startBit) << ", " << static_cast<int>(endBit)
        << "] are not writable on current device. Writable: bits 6-10, 45, 48, 50, 53, 59, 60";
}

static std::string MakeSetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "set_ctrl_spr requires 3 args (start_bit, end_bit, value)";
    auto start_val = ir::As<ir::ConstInt>(op->args_[0]);
    auto end_val = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, start_val != nullptr && end_val != nullptr)
        << "set_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(start_val->value_);
    int8_t endBit = static_cast<int8_t>(end_val->value_);
    CheckCtrlBitRange(startBit, endBit, "set_ctrl_spr");
    auto value_const = ir::As<ir::ConstInt>(op->args_[2]);
    if (value_const != nullptr && startBit >= 6 && endBit <= 10) {
        // CTRL[8:6]=3'b111 (atomic operand dtype) and CTRL[10:9]=2'b11 (atomic op
        // type) are undefined encodings that fault the device — reject up front.
        int64_t v = value_const->value_ & ((int64_t(1) << (endBit - startBit + 1)) - 1);
        if (startBit <= 6 && endBit >= 8 && ((v >> (6 - startBit)) & 0x7) == 0x7) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false)
                << "set_ctrl_spr: value " << value_const->value_
                << " sets CTRL[8:6]=3'b111, an undefined atomic operand dtype "
                << "(supported: 0-6 = none/float/half/int16/int32/int8/bfloat16)";
        }
        if (startBit <= 9 && endBit >= 10 && ((v >> (9 - startBit)) & 0x3) == 0x3) {
            PRO_CODEGEN_CHECK(ExternalError::NAME_ERROR, false)
                << "set_ctrl_spr: value " << value_const->value_
                << " sets CTRL[10:9]=2'b11, an undefined atomic op type " << "(supported: 0-2 = ADD/MAX/MIN)";
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

static std::string MakeGetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    (void)codegen_base;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "get_ctrl_spr requires 2 args (start_bit, end_bit)";
    auto start_val = ir::As<ir::ConstInt>(op->args_[0]);
    auto end_val = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, start_val != nullptr && end_val != nullptr)
        << "get_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(start_val->value_);
    int8_t endBit = static_cast<int8_t>(end_val->value_);
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

static std::string MakeResetCtrlSprCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "reset_ctrl_spr requires 2 args (start_bit, end_bit)";
    auto start_val = ir::As<ir::ConstInt>(op->args_[0]);
    auto end_val = ir::As<ir::ConstInt>(op->args_[1]);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, start_val != nullptr && end_val != nullptr)
        << "reset_ctrl_spr: start_bit and end_bit must be compile-time constants";
    int8_t startBit = static_cast<int8_t>(start_val->value_);
    int8_t endBit = static_cast<int8_t>(end_val->value_);
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
                                                                codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    std::string target_var = codegen.GetCurrentResultTarget();
    std::string input_var = cce::GetOperandAsCode(codegen, op->args_[0]);
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

    codegen.Emit("TTRANS(" + target_var + ", " + input_var + ");");
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

static std::string EnumValueName(const char* full_name)
{
    const char* sep = std::strrchr(full_name, ':');
    return sep ? std::string(sep + 1) : std::string(full_name);
}

static std::string NormalizeDcciCacheLine(int cache_line)
{
    auto cl = static_cast<ir::CacheLine>(cache_line);
    return EnumValueName(ir::EnumToString(cl));
}

static std::string NormalizeDcciDst(int dst, bool is_tile)
{
    auto d = static_cast<ir::DcciDst>(dst);
    if (d == ir::DcciDst::AUTO) {
        return is_tile ? "CACHELINE_UB" : "CACHELINE_OUT";
    }
    return EnumValueName(ir::EnumToString(d));
}

static std::string MakeDcciCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1 || op->args_.size() == 2)
        << "system.dcci requires 1 or 2 arguments, got " << op->args_.size();

    int cache_line_int = op->HasKwarg("cache_line") ? op->GetKwarg<int>("cache_line") : 1; // ENTIRE_DATA_CACHE
    int dst_int = op->HasKwarg("dst") ? op->GetKwarg<int>("dst") : 0;                      // AUTO

    const std::string cache_line = NormalizeDcciCacheLine(cache_line_int);

    auto tensor_type = ir::As<ir::TensorType>(op->args_[0]->GetType());
    if (tensor_type != nullptr) {
        auto tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tensor_var_ptr != nullptr)
            << "system.dcci: tensor target must be a Var";
        std::string tensor_var = codegen.GetVarName(tensor_var_ptr);
        std::string offset = "0";
        if (op->args_.size() == 2) {
            auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
            if (offsets_tuple != nullptr) {
                offset = codegen.ComputeTensorOffset(tensor_type, offsets_tuple);
            } else {
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE,
                                  ir::As<ir::ScalarType>(op->args_[1]->GetType()) != nullptr)
                    << "system.dcci: tensor target offset must be a tuple or scalar expression";
                offset = cce::GetOperandAsCode(codegen, op->args_[1]);
            }
        }
        std::string tensor_ptr = codegen.GetPointer(tensor_var);
        if (tensor_ptr.empty()) {
            tensor_ptr = tensor_var + ".data()";
        }
        const std::string dst_attr = NormalizeDcciDst(dst_int, false);
        codegen.Emit("dcci(reinterpret_cast<__gm__ void*>(" + tensor_ptr + " + " + offset + "), " + cache_line + ", " +
                     dst_attr + ");");
        return "";
    }

    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tile_type != nullptr)
        << "system.dcci: target must be TensorType or TileType";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, tile_type->memref_.has_value())
        << "system.dcci: tile target must have an allocated memory space";
    if (tile_type->memref_.value()->memorySpace_ != ir::MemorySpace::Vec) {
        PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_OPERATION)
            << "system.dcci: tile target must be allocated in Vec memory";
    }

    std::string tile = cce::GetOperandAsCode(codegen, op->args_[0]);
    std::string offset = "0";
    if (op->args_.size() == 2) {
        offset = cce::GetOperandAsCode(codegen, op->args_[1]);
    }
    const std::string dst_attr = NormalizeDcciDst(dst_int, true);
    codegen.Emit("dcci(reinterpret_cast<__ubuf__ void*>(" + tile + ".data() + " + offset + "), " + cache_line + ", " +
                 dst_attr + ");");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.bar_m")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_M);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte1")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_MTE1);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte2")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_MTE2);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_mte3")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_MTE3);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_fix")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_FIX);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.bar_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_ALL);");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mask_count")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("set_mask_count();");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mask_norm")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("set_mask_norm();");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_vec_mask")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
            << "system.set_vec_mask requires 2 arguments, but got " << op->args_.size();
        std::string mask_high = cce::GetOperandAsCode(codegen, op->args_[0]);
        std::string mask_low = cce::GetOperandAsCode(codegen, op->args_[1]);
        codegen.Emit("set_vector_mask(" + mask_high + ", " + mask_low + ");");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.reset_mask")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        (void)op;
        dynamic_cast<codegen::CCECodegen&>(codegen_base)
            .Emit("set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.dcci")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        return MakeDcciCodegenCCE(op, codegen_base);
    });

// ============================================================================
// Cross-core Sync Operations
// ============================================================================

// Cross-core SET: ffts_cross_core_sync(PIPE_xxx, getFFTSMsg(mode, event_id))
//   - INTER_BLOCK(0): inter-core sync
//   - INTER_SUBBLOCK(1): intra-core AIV-to-AIV sync
//   - INTRA_BLOCK(2): intra-core AIC↔AIV both subcores (A5 uses set_intra_block)
//   - UNICAST_BLOCK(3): intra-core AIC↔AIV one subcore (A5 uses set_intra_block)
// Cross-core WAIT: wait_flag_dev / wait_intra_block
// SET signals completion from a pipe; WAIT blocks until the other core signals.

static std::string MakeCrossCoreSetCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                              bool is_dynamic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pipe = op->GetKwarg<int>("pipe");
    std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
    bool is_a5 = (codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510);
    auto sync_mode = static_cast<ir::CrossCoreSyncMode>(op->GetKwarg<int>("sync_mode"));
    bool is_intra_unicast = sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK ||
                            sync_mode == ir::CrossCoreSyncMode::UNICAST_BLOCK;
    if (is_a5 && is_intra_unicast) {
        // A5 + INTRA_BLOCK or UNICAST_BLOCK: set_intra_block
        //     CUBE→VEC: INTRA_BLOCK expands to two calls (v0: id, v1: id+16); UNICAST_BLOCK single call
        //     VEC→CUBE: single set
        if (codegen.GetTarget() == ir::SectionKind::Cube && sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK) {
            if (is_dynamic) {
                std::string event_id = cce::GetOperandAsCode(codegen, op->args_[0]);
                codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + ");");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + " + 16);");
            } else {
                int event_id = op->GetKwarg<int>("event_id");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id + 16) + ");");
            }
        } else {
            if (is_dynamic) {
                std::string event_id = cce::GetOperandAsCode(codegen, op->args_[0]);
                codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + ");");
            } else {
                int event_id = op->GetKwarg<int>("event_id");
                codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
            }
        }
    } else {
        // non-A5, or A5 + INTER_BLOCK / INTER_SUBBLOCK: ffts_cross_core_sync
        std::string mode_str = std::to_string(static_cast<int>(sync_mode));
        if (is_dynamic) {
            std::string event_id = cce::GetOperandAsCode(codegen, op->args_[0]);
            codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(" + mode_str + ", " + event_id + "));");
        } else {
            int event_id = op->GetKwarg<int>("event_id");
            codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(" + mode_str + ", " +
                         std::to_string(event_id) + "));");
        }
    }
    return "";
}

static void EmitWaitIntraBlockCCE(codegen::CCECodegen& codegen, const ir::CallPtr& op, const std::string& pipe_str,
                                  bool is_dynamic, int event_id_offset = 0)
{
    if (is_dynamic) {
        std::string event_id = cce::GetOperandAsCode(codegen, op->args_[0]);
        if (event_id_offset != 0) {
            event_id += " + " + std::to_string(event_id_offset);
        }
        codegen.Emit("wait_intra_block(" + pipe_str + ", " + event_id + ");");
        return;
    }

    int event_id = op->GetKwarg<int>("event_id") + event_id_offset;
    codegen.Emit("wait_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
}

static std::string MakeCrossCoreWaitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                               bool is_dynamic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pipe = op->GetKwarg<int>("pipe");
    std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
    auto sync_mode = static_cast<ir::CrossCoreSyncMode>(op->GetKwarg<int>("sync_mode"));
    bool is_intra_unicast = (sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK ||
                             sync_mode == ir::CrossCoreSyncMode::UNICAST_BLOCK);
    bool wait_two_vec_subcores = (sync_mode == ir::CrossCoreSyncMode::INTRA_BLOCK);
    bool is_a5 = (codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510);
    if (is_a5 && is_intra_unicast) {
        // A5 + INTRA_BLOCK(2) or UNICAST_BLOCK(3): wait_intra_block
        //     CUBE waiting for VEC: INTRA_BLOCK expands to two calls (v0: id, v1: id+16); UNICAST_BLOCK single call
        //     VEC waiting for CUBE: single wait
        if (codegen.GetTarget() == ir::SectionKind::Cube && wait_two_vec_subcores) {
            EmitWaitIntraBlockCCE(codegen, op, pipe_str, is_dynamic);
            EmitWaitIntraBlockCCE(codegen, op, pipe_str, is_dynamic, 16);
        } else {
            EmitWaitIntraBlockCCE(codegen, op, pipe_str, is_dynamic);
        }
    } else {
        // non-A5, or A5 + INTER_BLOCK(0) / INTER_SUBBLOCK(1): wait_flag_dev
        std::string event_id = is_dynamic ? cce::GetOperandAsCode(codegen, op->args_[0]) :
                                            std::to_string(op->GetKwarg<int>("event_id"));
        if (is_a5) {
            codegen.Emit("wait_flag_dev(" + pipe_str + ", " + event_id + ");");
        } else {
            codegen.Emit("wait_flag_dev(" + event_id + ");");
        }
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
                                      codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
    auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
    std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
    std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
    std::string event_id = cce::GetOperandAsCode(codegen, op->args_[0]);
    codegen.Emit(intrinsic + "(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
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

static std::string MakeDebugAssertCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() >= 1)
        << "debug.assert requires at least 1 argument (condition)";

    std::string condition = cce::GetOperandAsCode(codegen, op->args_[0]);
    std::string condition_text = op->GetKwarg<std::string>("condition_text");
    std::string format = op->GetKwarg<std::string>("format");

    codegen.Emit("if (!(" + condition + ")) {");

    if (op->GetKwarg<bool>("show_location", false)) {
        std::string location = debug_printf::FormatDebugLocation(op->span_);
        if (!location.empty()) {
            codegen.Emit("  pypto::aicore::printf(\"" +
                         debug_printf::EscapeStringLiteral(location + " Assertion failed: " + condition_text + "\n") +
                         "\");");
        } else {
            codegen.Emit("  pypto::aicore::printf(\"" +
                         debug_printf::EscapeStringLiteral("Assertion failed: " + condition_text + "\n") + "\");");
        }
    } else {
        codegen.Emit("  pypto::aicore::printf(\"" +
                     debug_printf::EscapeStringLiteral("Assertion failed: " + condition_text + "\n") + "\");");
    }

    if (!format.empty()) {
        std::vector<std::string> args;
        std::vector<DataType> arg_dtypes;
        for (size_t i = 1; i < op->args_.size(); ++i) {
            args.emplace_back(cce::GetOperandAsCode(codegen, op->args_[i]));
            auto scalar_type = ir::As<ir::ScalarType>(op->args_[i]->GetType());
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalar_type) << "debug.assert argument must be ScalarType";
            arg_dtypes.emplace_back(scalar_type->dtype_);
        }
        codegen.Emit("  " + BuildPtoPrintfCall(format, args, arg_dtypes));
    }

    codegen.Emit("}");
    return "";
}

static std::string MakeDebugTrapCodegenCCE(const ir::CallPtr& /*op*/, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
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

namespace {

void EmitSanitizerLogPrologue(codegen::CCECodegen& codegen, const std::string& log, const std::string& capacity)
{
    // Region addressing. A mixed kernel runs the Cube part on the AICs and
    // the Vector part on the AIVs of the SAME launch, so both parts share
    // one log buffer and their region indices must never overlap (the
    // append counter is a plain read-modify-write; two writers on one
    // region silently drop records). The Vector part is therefore offset by
    // the block count: Cube covers [0, block_num), Vector covers
    // [block_num, block_num + block_num * subblockdim) = at most [N, 3N)
    // with the A5 1:2 C/V pairing, which the host allocation's configured
    // region envelope holds exactly. A
    // vector-only kernel wastes the first block_num regions, which is safe.
    // Sub-block addressing follows the established get_block_idx() backend-op
    // contract: the Vector target computes the global AIV index, so each
    // sub-block gets its own region even when several AIVs run under one
    // AIC block.
    std::string block_idx;
    if (codegen.GetTarget() == ir::SectionKind::Vector) {
        block_idx = "(int32_t)(__pypto_block_num_value() + __pypto_block_idx_value())";
    } else {
        block_idx = "(int32_t)(__pypto_block_idx_value())";
    }

    // Append-mode logging, no atomics: each executing sub-block targets its own
    // region (independently addressable via the global block index above), so
    // the leading counter is only written by this single writer. This keeps
    // every loop iteration's record (no overwrite) without relying on a GM
    // atomic instruction. Region layout: [ctr u32, records...], ctr is the
    // u32-word offset of the first free record (read, incremented, written
    // back); buffer size = sub_block_count * (capacity + 1) * 4 bytes.
    codegen.Emit("if (" + log + " != 0) {");
    codegen.Emit("  int32_t __log_block = " + block_idx + ";");
    codegen.Emit("  __gm__ uint32_t* __log_base = reinterpret_cast<__gm__ uint32_t*>(" + log +
                 ") + "
                 "static_cast<uint64_t>(__log_block) * (static_cast<uint64_t>(" +
                 capacity + ") + 1u);");
    codegen.Emit("  uint32_t __log_off = __log_base[0];");
}

} // namespace

static std::string MakeSanitizerLogCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    // args = [det_id, field..., log_ptr, capacity, span_id] -> at least the
    // det_id, the two hidden buffer parameters and the span id.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, op->args_.size() >= 4)
        << "block.sanitizer_log requires (det_id, field..., log_ptr, capacity, span_id)";

    const size_t n = op->args_.size();
    std::string det_id = cce::GetOperandAsCode(codegen, op->args_[0]);
    // Hidden log-buffer parameters: resolve their final (SSA-suffixed) names.
    std::string log = cce::GetOperandAsCode(codegen, op->args_[n - 3]);
    std::string capacity = cce::GetOperandAsCode(codegen, op->args_[n - 2]);
    std::string span_id = cce::GetOperandAsCode(codegen, op->args_[n - 1]);
    // Record size in u32 words: det_id + fields + span_id (fields exclude the
    // hidden log/capacity parameters).
    const uint32_t rec_u32 = static_cast<uint32_t>(n) - 2u;

    EmitSanitizerLogPrologue(codegen, log, capacity);
    codegen.Emit("  if (__log_off + " + std::to_string(rec_u32) + "u <= static_cast<uint32_t>(" + capacity + ")) {");
    codegen.Emit("    __gm__ uint32_t* __log_rec = __log_base + 1u + __log_off;");
    codegen.Emit("    __log_rec[0] = static_cast<uint32_t>(" + det_id + ");  // det_id");
    for (size_t i = 1; i + 3 < n; ++i) {
        codegen.Emit("    __log_rec[" + std::to_string(i) + "] = static_cast<int32_t>(" +
                     cce::GetOperandAsCode(codegen, op->args_[i]) + ");");
    }
    codegen.Emit("    __log_rec[" + std::to_string(rec_u32 - 1u) + "] = static_cast<uint32_t>(" + span_id + ");");
    codegen.Emit("    __log_base[0] = __log_off + " + std::to_string(rec_u32) + "u;");
    // Flush the record (and the counter update) to DDR before the next
    // instruction executes.  The instrumentation record is written by the S
    // pipe as a posted GM write; without a barrier the record can sit in the
    // write buffer while the actual op runs on a different pipe (MTE2/V/M).
    // When that op crashes the device (e.g. an extreme out-of-bounds load
    // triggers a hardware trap like 507015), the unflushed writes are lost
    // and the host-side rescue reads an empty log buffer.  dsb(DSB_DDR)
    // guarantees each record reaches main memory before any subsequent op,
    // at the cost of one barrier per instrumented record (debug-only cost).
    codegen.Emit("    dsb(DSB_DDR);");
    codegen.Emit("  }");
    codegen.Emit("}");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "block.sanitizer_log").set_pipe(ir::PipeType::S).f_codegen(MakeSanitizerLogCodegenCCE);

// Shared block.tile_addr helper. Keep the original Tile expression in the IR so
// CCE codegen can use the same MakeTuple/TileGroup materialization as the
// instrumented operation, then read the selected runtime Tile's data address.
static std::string MakeTileAddrCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
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
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 0)
            << "get_subblock_num requires no arguments";
        // Matches AscendC GetTaskRation(): AIC returns 1, AIV returns get_subblockdim().
        auto& cg = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        const auto target = cg.GetTarget();
        if (target == ir::SectionKind::Vector) {
            return std::string("(int64_t)(get_subblockdim())");
        }
        return std::string("(int64_t)(1)");
    });

// ============================================================================
// GetVal/SetVal Operations (unified: tile and tensor)
// ============================================================================

static std::string MakeGetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 2)
        << "getval requires 2 arguments, but got " << op->args_.size();

    auto result_type = ir::As<ir::ScalarType>(op->GetType());
    INTERNAL_CHECK(result_type) << "getval result must be ScalarType";
    auto cast_result_if_dtype_changed = [](std::string result, const ir::DataType& result_dtype,
                                           const ir::DataType& input_dtype) {
        if (result_dtype != input_dtype) {
            return "(" + result_dtype.ToCTypeString() + ")(" + result + ")";
        }
        return result;
    };

    auto first_type = op->args_[0]->GetType();
    if (auto tile_type = ir::As<ir::TileType>(first_type)) {
        std::string offset = codegen.GetExprAsCode(op->args_[1]);
        if (codegen.IsInSimtContext()) {
            std::string tile = codegen.GetExprAsCode(op->args_[0]);
            return cast_result_if_dtype_changed(tile + "[" + offset + "]", result_type->dtype_, tile_type->dtype_);
        }
        std::string tile = cce::GetOperandAsCode(codegen, op->args_[0]);
        return cast_result_if_dtype_changed(tile + ".GetValue(" + offset + ")", result_type->dtype_, tile_type->dtype_);
    }

    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_var)
        << "getval requires tensor to be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_type)
        << "getval requires TensorType";
    std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

    std::string tensor_ptr = codegen.GetPointer(tensor_name);

    if (codegen.GetCurrentKernelType() == ir::FunctionType::SIMT_KERNEL) {
        return cast_result_if_dtype_changed("(" + tensor_ptr + ")[" + offset + "]", result_type->dtype_,
                                            tensor_type->dtype_);
    }
    return cast_result_if_dtype_changed("*((__gm__ " + dtype_str + "*)" + tensor_ptr + " + " + offset + ")",
                                        result_type->dtype_, tensor_type->dtype_);
}

static std::string MakeSetValCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "setval requires 3 arguments, but got " << op->args_.size();

    auto first_type = op->args_[0]->GetType();
    if (ir::As<ir::TileType>(first_type)) {
        std::string offset = codegen.GetExprAsCode(op->args_[1]);
        std::string value = codegen.GetExprAsCode(op->args_[2]);
        if (codegen.IsInSimtContext()) {
            std::string tile = codegen.GetExprAsCode(op->args_[0]);
            codegen.Emit(tile + "[" + offset + "] = " + value + ";");
            return "";
        }
        std::string tile = cce::GetOperandAsCode(codegen, op->args_[0]);
        codegen.Emit(tile + ".SetValue(" + offset + ", " + value + ");");
        return "";
    }

    auto tensor_var = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_var)
        << "setval requires tensor to be a Var";
    auto tensor_type = ir::As<ir::TensorType>(tensor_var->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensor_type)
        << "setval requires TensorType";
    std::string tensor_name = codegen.GetVarName(tensor_var);
    std::string offset = codegen.GetExprAsCode(op->args_[1]);
    std::string value = codegen.GetExprAsCode(op->args_[2]);
    std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);

    std::string tensor_ptr = codegen.GetPointer(tensor_name);

    if (codegen.GetCurrentKernelType() == ir::FunctionType::SIMT_KERNEL) {
        codegen.Emit("(" + tensor_ptr + ")[" + offset + "] = " + value + ";");
        return "";
    }
    codegen.Emit("*((__gm__ " + dtype_str + "*)" + tensor_ptr + " + " + offset + ") = " + value + ";");
    return "";
}

static std::string MakeTileValidShapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      op->args_.size() == 1 && ir::As<ir::TileType>(op->args_[0]->GetType()))
        << "block.tile_valid_shape requires one Tile argument";
    int axis = op->GetKwarg<int>("axis");
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, axis >= 0 && axis <= 1)
        << "block.tile_valid_shape axis must be in [0, 1]";

    if (codegen.IsInSimtContext()) {
        return "(int64_t)(" + codegen.GetExprAsCode(op->args_[0]) + (axis == 0 ? "__valid_row)" : "__valid_col)");
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
static std::string MakeBlockSubviewCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    auto tile_type = ir::As<ir::TileType>(op->args_[0]->GetType());
    auto offset_tuple = ir::As<ir::MakeTuple>(op->args_[1]);
    const bool is_vf = codegen.IsInVFSection();

    // VF also uses block.subview for tile + linear_offset.
    if (offset_tuple == nullptr) {
        CHECK(is_vf) << "block.subview offset must be a 2D tuple outside VF linear-offset access";
        const std::string base_ptr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/false);
        return codegen.RecordVFTileSubviewPointer("(" + base_ptr + " + (" + codegen.GetExprAsCode(op->args_[1]) + "))");
    }

    const std::string row_expr = cce::GetOperandAsCode(codegen, offset_tuple->elements_[0]);
    const std::string col_expr = cce::GetOperandAsCode(codegen, offset_tuple->elements_[1]);
    const std::string base_tile = cce::GetOperandAsCode(codegen, op->args_[0]);
    const std::string elem_off = "pto::GetTileOffset<std::remove_reference_t<decltype(" + base_tile + ")>>(" +
                                 row_expr + ", " + col_expr + ")";

    // VF only needs pointer arithmetic and does not create a Tile descriptor.
    if (is_vf) {
        const std::string base_ptr = codegen.GetOrCreateVFTilePtr(op->args_[0], /*is_post_update=*/false);
        return codegen.RecordVFTileSubviewPointer("(" + base_ptr + " + " + elem_off + ")");
    }

    const int dtype_bits = static_cast<int>(tile_type->dtype_.GetBit());
    const std::string byte_offset_expr = "((" + elem_off + ")*" + std::to_string(dtype_bits) + "/8)";

    const std::string base_addr = "(uint64_t)" + base_tile + ".data()";

    // Sub-window valid_shape from args[2] (computed by parser as the intersection
    // of slice size and original valid_shape - start).
    auto shape_tuple = ir::As<ir::MakeTuple>(op->args_[2]);
    std::string vs_row = cce::GetOperandAsCode(codegen, shape_tuple->elements_[0]);
    std::string vs_col = cce::GetOperandAsCode(codegen, shape_tuple->elements_[1]);

    // Build type with original shape (preserves row_stride) but WITHOUT tileView_,
    // so valid_shape template params are -1 (dynamic). This is required because
    // subview calls SetValidShape at runtime, which only works when the template
    // valid_shape params are DYNAMIC (-1).
    auto subview_type = ir::As<ir::TileType>(op->GetType());

    // Sanitize base_tile into a valid C++ identifier prefix
    std::string base_tile_id = base_tile;
    for (char& c : base_tile_id)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
            c = '_';
    std::string temp_name = codegen.GetCurrentResultTarget().empty() ?
                                (base_tile_id + "_view_" + std::to_string(codegen.GetTileOffsetCounter())) :
                                codegen.GetCurrentResultTarget();

    // The descriptor and its binding are created at the source value's execution point.
    std::string temp_addr = base_addr + " + " + byte_offset_expr;
    const std::string descriptor = temp_name + ".tile";
    DeclareTile(codegen, subview_type, temp_name, vs_row + ", " + vs_col, InheritedMutexIds(codegen, op->args_[0]));
    codegen.Emit("TASSIGN(" + descriptor + ", " + temp_addr + ");");
    codegen.Emit(descriptor + ".SetValidShape(" + vs_row + ", " + vs_col + ");");

    return temp_name;
}

REGISTER_BACKEND_OP(BackendCCE, "block.subview")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        return MakeBlockSubviewCodegenCCE(op, codegen_base);
    });

// ============================================================================
// Mutex (Buffer-ID Token) - A5 CCE Codegen
// ----------------------------------------------------------------------------
// Lowers system.mutex_lock/unlock to CCE intrinsics get_buf/rls_buf.
// API: get_buf(PIPE_MTE2, mutexId, 0);  rls_buf(PIPE_MTE2, mutexId, 0);
// ============================================================================

static std::string MakeMutexBufCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                          const std::string& intrinsic)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
    auto pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("pipe"));
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 1)
        << "dynamic mutex requires one scalar mutex ID";
    codegen.Emit(intrinsic + "(" + PipeTypeToCCEString(pipe) + ", " + cce::GetOperandAsCode(codegen, op->args_[0]) +
                 ", 0);");
    return "";
}

static std::string MakeMutexLockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "get_buf");
}

static std::string MakeMutexUnlockDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    return MakeMutexBufCodegenCCE(op, codegen_base, "rls_buf");
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
// Delegates to pto-isa SYNCALL<SyncCoreType>().

static std::string MakeSystemSyncAllCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);

    auto core_type = static_cast<ir::SyncCoreType>(op->HasKwarg("core_type") ? op->GetKwarg<int>("core_type") : 2);

    std::string core_type_tok;
    if (core_type == ir::SyncCoreType::AIV_ONLY)
        core_type_tok = "SyncCoreType::AIVOnly";
    else if (core_type == ir::SyncCoreType::AIC_ONLY)
        core_type_tok = "SyncCoreType::AICOnly";
    else
        core_type_tok = "SyncCoreType::Mix";

    codegen.Emit("SYNCALL<" + core_type_tok + ">();");
    return "";
}

REGISTER_BACKEND_OP(BackendCCE, "system.sync_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeSystemSyncAllCodegenCCE(op, codegen);
    });

// ============================================================================
// Early-start inter-task sync (wait_pre_task_end)
// ============================================================================
// Delegates to AscendC::WaitPreTaskEnd(); effective only when the kernel is compiled
// with -D__ASCENDC_SUPERKERNEL_EARLY_START_V3. The __sk__ entry must propagate
// sysArgs->skTaskSyncCfg into g_super_kernel_early_start_config (see jit.py).

REGISTER_BACKEND_OP(BackendCCE, "system.wait_pre_task_end")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        auto& cce_codegen = dynamic_cast<codegen::CCECodegen&>(codegen);
        bool force = op->HasKwarg("force") ? op->GetKwarg<bool>("force") : false;
        cce_codegen.Emit(force ? "AscendC::WaitPreTaskEnd<true>();" : "AscendC::WaitPreTaskEnd();");
        return "";
    });

// Early-start inter-task sync (set_next_task_start, dual of wait_pre_task_end):
// instructions after the call overlap the *next* sub-kernel. FORCE is SetNextTaskStart's
// third template parameter (after the AIV/AIC pipe selectors); the A5 (__NPU_ARCH__ not
// 1001/2002) defaults are PIPE_MTE3 / PIPE_FIX, so the forced form spells them out.
REGISTER_BACKEND_OP(BackendCCE, "system.set_next_task_start")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        auto& cce_codegen = dynamic_cast<codegen::CCECodegen&>(codegen);
        bool force = op->HasKwarg("force") ? op->GetKwarg<bool>("force") : false;
        cce_codegen.Emit(force ? "AscendC::SetNextTaskStart<PIPE_MTE3, PIPE_FIX, true>();" :
                                 "AscendC::SetNextTaskStart();");
        return "";
    });

REGISTER_BACKEND_OP(BackendCCE, "system.set_mm_layout_transform")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
        auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
        auto enabled = op->GetKwarg<int>("enabled");
        if (codegen.GetArch() == npu::tile_fwk::NPUArch::DAV_3510) {
            // Direct register manipulation: MM_LAYOUT_MODE_BIT = 51
            if (enabled) {
                codegen.Emit("set_ctrl(sbitset1(get_ctrl(), 51));");
            } else {
                codegen.Emit("set_ctrl(sbitset0(get_ctrl(), 51));");
            }
        }
        return "";
    });

} // namespace backend
} // namespace pypto
