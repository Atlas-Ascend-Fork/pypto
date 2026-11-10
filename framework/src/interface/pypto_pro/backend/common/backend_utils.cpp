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

int FindIndex(const std::string& mode, const std::string& opName)
{
    static const char* const kRoundModeNames[] = {"none", "rint",  "round", "floor",
                                                  "ceil", "trunc", "odd",   "cast_rint"};
    for (size_t i = 0; i < sizeof(kRoundModeNames) / sizeof(kRoundModeNames[0]); ++i) {
        if (mode == kRoundModeNames[i]) {
            return static_cast<int>(i);
        }
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, false) << opName << ": unknown round mode '" << mode << "'";
    return -1;
}

} // namespace round_mode

namespace gather {

CompareAttrs GetCompareAttrs(const ir::CallPtr& op)
{
    CompareAttrs attrs;
    for (const auto& [key, val] : op->kwargs_) {
        if (key == "cmp_mode") {
            attrs.hasCmpMode = true;
            attrs.cmpMode = std::any_cast<int>(val);
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

std::optional<ir::Layout> GetTileTensorLayout(const ir::HardwareInfo& hardwareInfo)
{
    switch (hardwareInfo.slayout) {
        case ir::TileLayout::none_box:
            if (hardwareInfo.blayout == ir::TileLayout::row_major) {
                return ir::Layout::ND;
            }
            if (hardwareInfo.blayout == ir::TileLayout::col_major) {
                return ir::Layout::DN;
            }
            break;
        case ir::TileLayout::row_major:
            if (hardwareInfo.blayout == ir::TileLayout::row_major) {
                return ir::Layout::ZZ;
            }
            if (hardwareInfo.blayout == ir::TileLayout::col_major) {
                return ir::Layout::NZ;
            }
            break;
        case ir::TileLayout::col_major:
            if (hardwareInfo.blayout == ir::TileLayout::row_major) {
                return ir::Layout::ZN;
            }
            if (hardwareInfo.blayout == ir::TileLayout::col_major) {
                return ir::Layout::NN;
            }
            break;
        default:
            return std::nullopt;
    }
    return std::nullopt;
}

AccessArgIndices ResolveAccessArgIndices(const std::string& opName)
{
    AccessArgIndices indices;
    if (opName == "block.load") {
        indices.tensorArgIdx = 1;
        indices.tileArgIdx = 0;
        indices.offsetsArgIdx = 2;
    } else if (opName == "block.store") {
        indices.tensorArgIdx = 0;
        indices.tileArgIdx = 1;
        indices.offsetsArgIdx = 2;
    }
    return indices;
}

bool IsNZTensorType(const ir::TensorTypePtr& tensorType)
{
    return tensorType && tensorType->tensor_view_.has_value() && tensorType->tensor_view_->layout == ir::Layout::NZ;
}

bool IsMXLoad(const ir::TensorTypePtr& tensorType, const ir::TileTypePtr& tileType)
{
    if (!tensorType || !tileType || tensorType->dtype_ != ir::DataType::FP8E8M0 ||
        tileType->dtype_ != ir::DataType::FP8E8M0 || !tileType->memref_.has_value() ||
        tileType->memref_.value()->memorySpace_ != ir::MemorySpace::Mat || !tileType->hardwareInfo_.has_value()) {
        return false;
    }

    const auto& hw = tileType->hardwareInfo_.value();
    const auto layout = GetTileTensorLayout(hw);
    return hw.fractal == 32 && (layout == ir::Layout::ZZ || layout == ir::Layout::NN);
}

std::string MXLoadLayoutName(const ir::CallPtr& op)
{
    if (op->name_ != "block.load") {
        return "";
    }
    auto tensorType = ir::As<ir::TensorType>(op->args_[1]->GetType());
    auto tileType = ir::As<ir::TileType>(op->args_[0]->GetType());
    if (!IsMXLoad(tensorType, tileType)) {
        return "";
    }

    const auto& hw = tileType->hardwareInfo_.value();
    const bool isScaleA = GetTileTensorLayout(hw) == ir::Layout::ZZ;
    const bool isDn = op->GetKwarg<bool>("is_transpose", false);
    return isScaleA ? (isDn ? "Layout::MX_A_DN" : "Layout::MX_A_ND") : (isDn ? "Layout::MX_B_DN" : "Layout::MX_B_ND");
}

int64_t GetNZInnerCols(const ir::DataType& dtype)
{
    constexpr int64_t nzInnerBlockBits = 256;
    return nzInnerBlockBits / static_cast<int64_t>(dtype.GetBit());
}

void ValidateNZTransfer(const std::string& opName, const ir::CallPtr& op, const ir::ExprPtr& tileExpr,
                        const ir::MakeTuplePtr& offsets, const ir::TensorTypePtr& tensorType)
{
    if (!IsNZTensorType(tensorType)) {
        return;
    }

    auto tileType = ir::As<ir::TileType>(tileExpr->GetType());
    const auto& hw = tileType->hardwareInfo_.value();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, GetTileTensorLayout(hw) == ir::Layout::NZ)
        << opName << ": GM NZ transfer requires an NZ Tile layout";

    const size_t ndim = tensorType->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, offsets->elements_.size() == ndim)
        << opName << ": offset rank must match GM NZ tensor rank";
    const bool isTranspose = op->HasKwarg("is_transpose") && op->GetKwarg<bool>("is_transpose");
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !isTranspose)
        << opName << ": CCE NZ transfer does not support order transpose";

    const int64_t c0 = GetNZInnerCols(tensorType->dtype_);
    const auto tileRows = ir::GetConstantDimension(tileType->shape_[0]);
    const auto tileCols = ir::GetConstantDimension(tileType->shape_[1]);
    const auto colOffset = ir::GetConstantDimension(offsets->elements_[ndim - 1]);

    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !tileRows.has_value() || tileRows.value() % 16 == 0)
        << opName << ": NZ tile rows must be divisible by 16";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !tileCols.has_value() || tileCols.value() % c0 == 0)
        << opName << ": NZ tile columns must be divisible by C0";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT, !colOffset.has_value() || colOffset.value() % c0 == 0)
        << opName << ": NZ column offset must be divisible by C0";

    // In a GM transfer, an Acc tile can only appear as the source of store.
    // Enforce the extra direct-store window restriction for that path.
    if (!tileType->memref_.has_value() || tileType->memref_.value()->memorySpace_ != ir::MemorySpace::Acc) {
        return;
    }

    const auto fullRows = ir::GetConstantDimension(tensorType->shape_[ndim - 2]);
    const auto paddedFullRows = fullRows.has_value() ? std::optional<int64_t>((fullRows.value() + 15) / 16 * 16) :
                                                       std::nullopt;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_FORMAT,
                      !paddedFullRows.has_value() || !tileRows.has_value() || !tileCols.has_value() ||
                          tileRows.value() == paddedFullRows.value() || tileCols.value() <= c0)
        << opName << ": Acc NZ partial-M store spanning multiple N fractals is not supported by direct TSTORE";
}

std::string CoerceScalarToInt(const ir::ExprPtr& scalarExpr, const ir::DataType& srcDt, const std::string& originalStr)
{
    if (!srcDt.IsInt()) {
        return originalStr;
    }
    auto cf = ir::As<ir::ConstFloat>(scalarExpr);
    if (cf == nullptr) {
        return originalStr;
    }
    const size_t bits = srcDt.GetBit();
    if (bits == 0 || bits > 64) {
        return originalStr;
    }
    const auto raw = static_cast<uint64_t>(static_cast<int64_t>(cf->value_));
    const uint64_t truncated = (bits == 64) ? raw : (raw & ((static_cast<uint64_t>(1) << bits) - 1));
    if (srcDt.IsUnsignedInt()) {
        return std::to_string(truncated) + "u";
    }
    // Sign-extend back out of the truncated width.
    const uint64_t signBit = static_cast<uint64_t>(1) << (bits - 1);
    return std::to_string(static_cast<int64_t>((truncated ^ signBit) - signBit));
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

    size_t lastSep = span.Filename().find_last_of("/\\");
    std::string basename = lastSep == std::string::npos ? span.Filename() : span.Filename().substr(lastSep + 1);
    if (basename.empty()) {
        return "";
    }
    return "[" + basename + ":" + std::to_string(span.BeginLine()) + "]";
}

std::string FormatDebugLocationHeader(const ir::Span& span, const std::string& opName)
{
    std::string location = FormatDebugLocation(span);
    if (location.empty()) {
        return "";
    }
    return location + " " + opName;
}

bool IsSupportedPrintfConversion(char conversion)
{
    return conversion == 'd' || conversion == 'i' || conversion == 'u' || conversion == 'x' || conversion == 'f' ||
           conversion == 'p';
}

size_t FindPrintfConversionIndex(const std::string& formatSegment)
{
    size_t i = 0;
    while (i < formatSegment.size()) {
        if (formatSegment[i] != '%') {
            ++i;
            continue;
        }
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                          !(i + 1 < formatSegment.size() && formatSegment[i + 1] == '%'))
            << "debug.printf does not support literal '%%'";

        size_t j = i + 1;
        PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, j < formatSegment.size())
            << "debug.printf format ends with an incomplete conversion";
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                          formatSegment[j] != '-' && formatSegment[j] != '+' && formatSegment[j] != ' ' &&
                              formatSegment[j] != '#' && formatSegment[j] != '0' && formatSegment[j] != '.' &&
                              !std::isdigit(static_cast<unsigned char>(formatSegment[j])))
            << "debug.printf does not support flags, width, or precision";
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsSupportedPrintfConversion(formatSegment[j]))
            << "debug.printf does not support conversion '%" << formatSegment[j] << "'";
        return j;
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, false)
        << "debug.printf format segment must contain a supported conversion";
    return std::string::npos;
}

std::vector<PrintfSegment> ParsePrintfSegments(const std::string& format)
{
    std::vector<PrintfSegment> segments;
    std::string pendingText;
    size_t i = 0;
    while (i < format.size()) {
        if (format[i] != '%') {
            pendingText.push_back(format[i]);
            ++i;
            continue;
        }
        if (i + 1 < format.size() && format[i + 1] == '%') {
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false)
                << "debug.printf does not support literal '%%'";
        }

        size_t j = FindPrintfConversionIndex(format.substr(i)) + i;
        char conversion = format[j];
        segments.push_back({pendingText + format.substr(i, j - i + 1), conversion});
        pendingText.clear();
        i = j + 1;
    }

    if (!pendingText.empty()) {
        if (segments.empty()) {
            return segments;
        }
        segments.back().formatSegment += pendingText;
    }
    return segments;
}

} // namespace debug_printf
} // namespace backend
} // namespace pypto
