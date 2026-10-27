/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/backend_utils.h"

#include <any>
#include <cctype>
#include <sstream>

#include "core/error.h"
#include "codegen/cce/cce_codegen.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/scalar_expr.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using npu::tile_fwk::ExternalError;
namespace round_mode {

int FindIndex(const std::string& mode, const std::string& op_name)
{
    static const char* const kRoundModeNames[] = {"none", "rint",  "round", "floor",
                                                  "ceil", "trunc", "odd",   "cast_rint"};
    for (size_t i = 0; i < sizeof(kRoundModeNames) / sizeof(kRoundModeNames[0]); ++i) {
        if (mode == kRoundModeNames[i]) {
            return static_cast<int>(i);
        }
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, false) << op_name << ": unknown round mode '" << mode << "'";
    return -1;
}

} // namespace round_mode

namespace gather {

CompareAttrs GetCompareAttrs(const ir::CallPtr& op)
{
    CompareAttrs attrs;
    for (const auto& [key, val] : op->kwargs_) {
        if (key == "cmp_mode") {
            attrs.has_cmp_mode = true;
            attrs.cmp_mode = std::any_cast<int>(val);
        } else if (key == "offset") {
            attrs.offset = std::any_cast<int>(val);
        }
    }
    return attrs;
}

} // namespace gather

namespace cce {

std::string GetOperandAsCode(codegen::CCECodegen& codegen, const ir::ExprPtr& expr)
{
    auto value = codegen.GetExprAsCode(expr);
    if (ir::As<ir::MutexTileType>(expr->GetType())) {
        value += ".tile";
    }
    return value;
}

AccessArgIndices ResolveAccessArgIndices(const std::string& op_name)
{
    AccessArgIndices indices;
    if (op_name == "block.load") {
        indices.tensor_arg_idx = 1;
        indices.tile_arg_idx = 0;
        indices.offsets_arg_idx = 2;
    } else if (op_name == "block.store") {
        indices.tensor_arg_idx = 0;
        indices.tile_arg_idx = 1;
        indices.offsets_arg_idx = 2;
    } else if (op_name == "debug.dump_tile") {
        indices.tensor_arg_idx = 3;
        indices.tile_arg_idx = 0;
    }
    return indices;
}

bool IsNZTensorType(const ir::TensorTypePtr& tensor_type)
{
    return tensor_type && tensor_type->tensor_view_.has_value() &&
           tensor_type->tensor_view_->layout == ir::TensorLayout::NZ;
}

bool IsMXLoad(const ir::TensorTypePtr& tensor_type, const ir::TileTypePtr& tile_type)
{
    if (!tensor_type || !tile_type || tensor_type->dtype_ != ir::DataType::FP8E8M0 ||
        tile_type->dtype_ != ir::DataType::FP8E8M0 || !tile_type->memref_.has_value() ||
        tile_type->memref_.value()->memorySpace_ != ir::MemorySpace::Mat || !tile_type->hardwareInfo_.has_value()) {
        return false;
    }

    const auto& hw = tile_type->hardwareInfo_.value();
    const bool is_zz = hw.blayout == ir::TileLayout::row_major && hw.slayout == ir::TileLayout::row_major;
    const bool is_nn = hw.blayout == ir::TileLayout::col_major && hw.slayout == ir::TileLayout::col_major;
    return hw.fractal == 32 && (is_zz || is_nn);
}

bool IsMXLoadCall(const ir::CallPtr& op)
{
    if (!op || op->name_ != "block.load" || op->args_.size() < 2) {
        return false;
    }
    return IsMXLoad(ir::As<ir::TensorType>(op->args_[1]->GetType()), ir::As<ir::TileType>(op->args_[0]->GetType()));
}

std::vector<int> MXLoadTileDims(const ir::CallPtr& op, const ir::TensorTypePtr& tensor_type)
{
    const int rank = static_cast<int>(tensor_type->shape_.size());
    // MX scale tensors carry a trailing physical phase axis of size 2.
    PRO_CODEGEN_IRCHECK(ExternalError::INVALID_SHAPE, rank >= 3)
        << "MX scale load requires at least two matrix axes and one physical phase axis at " << op->span_.ToString();
    const auto phase_dim = ir::As<ir::ConstInt>(tensor_type->shape_.back());
    PRO_CODEGEN_IRCHECK(ExternalError::INVALID_SHAPE, phase_dim != nullptr && phase_dim->value_ == 2)
        << "MX scale load trailing physical phase axis must be statically equal to 2 at " << op->span_.ToString();
    const std::vector<int> tile_dims = op->HasKwarg("tile_dims") ? op->GetKwarg<std::vector<int>>("tile_dims") :
                                                                   std::vector<int>{rank - 3, rank - 2};
    PRO_CODEGEN_IRCHECK(ExternalError::INVALID_SHAPE, tile_dims[0] != rank - 1 && tile_dims[1] != rank - 1)
        << "MX scale load order cannot select the trailing physical phase axis at " << op->span_.ToString();
    return tile_dims;
}

std::string MXLoadLayoutName(const ir::CallPtr& op)
{
    if (!IsMXLoadCall(op)) {
        return "";
    }
    const auto& hw = ir::As<ir::TileType>(op->args_[0]->GetType())->hardwareInfo_.value();
    const bool is_scale_a = hw.blayout == ir::TileLayout::row_major;
    const bool is_dn = op->GetKwarg<bool>("is_transpose", false);
    return is_scale_a ? (is_dn ? "Layout::MX_A_DN" : "Layout::MX_A_ND") :
                        (is_dn ? "Layout::MX_B_DN" : "Layout::MX_B_ND");
}

int64_t GetNZInnerCols(const ir::DataType& dtype)
{
    constexpr int64_t nz_inner_block_bits = 256;
    return nz_inner_block_bits / static_cast<int64_t>(dtype.GetBit());
}

void ValidateNZTransfer(const std::string& op_name, const ir::CallPtr& op, const ir::ExprPtr& tile_expr,
                        const ir::MakeTuplePtr& offsets, const ir::TensorTypePtr& tensor_type)
{
    if (!IsNZTensorType(tensor_type)) {
        return;
    }

    auto tile_type = ir::As<ir::TileType>(tile_expr->GetType());
    const auto& hw = tile_type->hardwareInfo_.value();
    const bool is_nz_tile = hw.blayout == ir::TileLayout::col_major && hw.slayout == ir::TileLayout::row_major;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, is_nz_tile)
        << op_name << ": GM NZ transfer requires an NZ Tile layout";

    const size_t ndim = tensor_type->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, offsets->elements_.size() == ndim)
        << op_name << ": offset rank must match GM NZ tensor rank";
    const bool is_transpose = op->HasKwarg("is_transpose") && op->GetKwarg<bool>("is_transpose");
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !is_transpose)
        << op_name << ": CCE NZ transfer does not support order transpose";

    const int64_t c0 = GetNZInnerCols(tensor_type->dtype_);
    const auto tile_rows = ir::GetConstantDimension(tile_type->shape_[0]);
    const auto tile_cols = ir::GetConstantDimension(tile_type->shape_[1]);
    const auto col_offset = ir::GetConstantDimension(offsets->elements_[ndim - 1]);

    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !tile_rows.has_value() || tile_rows.value() % 16 == 0)
        << op_name << ": NZ tile rows must be divisible by 16";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !tile_cols.has_value() || tile_cols.value() % c0 == 0)
        << op_name << ": NZ tile columns must be divisible by C0";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !col_offset.has_value() || col_offset.value() % c0 == 0)
        << op_name << ": NZ column offset must be divisible by C0";

    // In a GM transfer, an Acc tile can only appear as the source of store.
    // Enforce the extra direct-store window restriction for that path.
    if (!tile_type->memref_.has_value() || tile_type->memref_.value()->memorySpace_ != ir::MemorySpace::Acc) {
        return;
    }

    const auto full_rows = ir::GetConstantDimension(tensor_type->shape_[ndim - 2]);
    const auto padded_full_rows = full_rows.has_value() ? std::optional<int64_t>((full_rows.value() + 15) / 16 * 16) :
                                                          std::nullopt;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT,
                      !padded_full_rows.has_value() || !tile_rows.has_value() || !tile_cols.has_value() ||
                          tile_rows.value() == padded_full_rows.value() || tile_cols.value() <= c0)
        << op_name << ": Acc NZ partial-M store spanning multiple N fractals is not supported by direct TSTORE";
}

std::string CoerceScalarToInt(const ir::ExprPtr& scalar_expr, const ir::DataType& src_dt,
                              const std::string& original_str)
{
    if (!src_dt.IsInt()) {
        return original_str;
    }
    auto cf = ir::As<ir::ConstFloat>(scalar_expr);
    if (cf == nullptr) {
        return original_str;
    }
    const size_t bits = src_dt.GetBit();
    if (bits == 0 || bits > 64) {
        return original_str;
    }
    const auto raw = static_cast<uint64_t>(static_cast<int64_t>(cf->value_));
    const uint64_t truncated = (bits == 64) ? raw : (raw & ((static_cast<uint64_t>(1) << bits) - 1));
    if (src_dt.IsUnsignedInt()) {
        return std::to_string(truncated) + "u";
    }
    // Sign-extend back out of the truncated width.
    const uint64_t sign_bit = static_cast<uint64_t>(1) << (bits - 1);
    return std::to_string(static_cast<int64_t>((truncated ^ sign_bit) - sign_bit));
}

} // namespace cce

namespace debug_printf {

std::string EscapeStringLiteral(const std::string& text)
{
    std::ostringstream oss;
    for (char c : text) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\t':
                oss << "\\t";
                break;
            case '\r':
                oss << "\\r";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

std::string QuoteMlirStringLiteral(const std::string& text) { return "\"" + EscapeStringLiteral(text) + "\""; }

std::string FormatDebugLocation(const ir::Span& span)
{
    if (!span.IsValid() || span.Filename().empty() || span.BeginLine() <= 0) {
        return "";
    }

    size_t last_sep = span.Filename().find_last_of("/\\");
    std::string basename = last_sep == std::string::npos ? span.Filename() : span.Filename().substr(last_sep + 1);
    if (basename.empty()) {
        return "";
    }
    return "[" + basename + ":" + std::to_string(span.BeginLine()) + "]";
}

std::string FormatDebugLocationHeader(const ir::Span& span, const std::string& op_name)
{
    std::string location = FormatDebugLocation(span);
    if (location.empty()) {
        return "";
    }
    return location + " " + op_name;
}

bool IsSupportedPrintfConversion(char conversion)
{
    return conversion == 'd' || conversion == 'i' || conversion == 'u' || conversion == 'x' || conversion == 'f' ||
           conversion == 'p';
}

size_t FindPrintfConversionIndex(const std::string& format_segment)
{
    size_t i = 0;
    while (i < format_segment.size()) {
        if (format_segment[i] != '%') {
            ++i;
            continue;
        }
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                          !(i + 1 < format_segment.size() && format_segment[i + 1] == '%'))
            << "debug.printf does not support literal '%%'";

        size_t j = i + 1;
        PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, j < format_segment.size())
            << "debug.printf format ends with an incomplete conversion";
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                          format_segment[j] != '-' && format_segment[j] != '+' && format_segment[j] != ' ' &&
                              format_segment[j] != '#' && format_segment[j] != '0' && format_segment[j] != '.' &&
                              !std::isdigit(static_cast<unsigned char>(format_segment[j])))
            << "debug.printf does not support flags, width, or precision";
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsSupportedPrintfConversion(format_segment[j]))
            << "debug.printf does not support conversion '%" << format_segment[j] << "'";
        return j;
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, false)
        << "debug.printf format segment must contain a supported conversion";
    return std::string::npos;
}

std::vector<PrintfSegment> ParsePrintfSegments(const std::string& format)
{
    std::vector<PrintfSegment> segments;
    std::string pending_text;
    size_t i = 0;
    while (i < format.size()) {
        if (format[i] != '%') {
            pending_text.push_back(format[i]);
            ++i;
            continue;
        }
        if (i + 1 < format.size() && format[i + 1] == '%') {
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false)
                << "debug.printf does not support literal '%%'";
        }

        size_t j = FindPrintfConversionIndex(format.substr(i)) + i;
        char conversion = format[j];
        segments.push_back({pending_text + format.substr(i, j - i + 1), conversion});
        pending_text.clear();
        i = j + 1;
    }

    if (!pending_text.empty()) {
        if (segments.empty()) {
            return segments;
        }
        segments.back().format_segment += pending_text;
    }
    return segments;
}

} // namespace debug_printf
} // namespace backend
} // namespace pypto
