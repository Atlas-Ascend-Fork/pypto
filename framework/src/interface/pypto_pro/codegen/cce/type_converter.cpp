/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/type_converter.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "ir/op_attr_types.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/type.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {

namespace codegen {
using npu::tile_fwk::ExternalError; // Short enum names in the Enum field (log spec 4.4)

namespace {

// Resolve a valid_shape ExprPtr to a string for use in the Tile<> type template.
// Only ConstInt values are supported (Var �?compile-time unknown, skip).
// Returns "" if the expression cannot be resolved to a constant.
// ConstInt(-1) means "use full shape at runtime" �?emits "-1" in template (dynamic valid_shape).
std::string ResolveValidShapeDim(const ir::ExprPtr& expr, int64_t /*fallback*/)
{
    if (!expr)
        return "-1";
    if (auto c = ir::As<ir::ConstInt>(expr)) {
        return (c->value_ == -1) ? "-1" : std::to_string(c->value_);
    }
    // Var or other dynamic expression: cannot embed in template parameter.
    return "";
}

std::string ConvertTilePadToPTOValue(ir::TilePad pad)
{
    switch (pad) {
        case ir::TilePad::null:
            return "PadValue::Null";
        case ir::TilePad::zero:
            return "PadValue::Zero";
        case ir::TilePad::max:
            return "PadValue::Max";
        case ir::TilePad::min:
            return "PadValue::Min";
        default:
            PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_ARGUMENT) << "Invalid TilePad value";
    }
}

std::string ConvertCompactModeToPTOValue(ir::CompactMode compact)
{
    switch (compact) {
        case ir::CompactMode::null:
            return "";
        case ir::CompactMode::normal:
            return "CompactMode::Normal";
        case ir::CompactMode::row_plus_one:
            return "CompactMode::RowPlusOne";
        default:
            PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_ARGUMENT) << "Invalid CompactMode value";
    }
}

void ApplyTileViewValidShape(const ir::TileTypePtr& tileType, int64_t rows, int64_t cols, std::string& typeRows,
                             std::string& typeCols)
{
    if (!tileType->tileView_.has_value()) {
        return;
    }
    const auto& tv = tileType->tileView_.value();
    if (tv.validShape.empty()) {
        return;
    }
    std::string resolvedRows = tv.validShape.size() >= 1 ? ResolveValidShapeDim(tv.validShape[0], rows) : "";
    std::string resolvedCols = tv.validShape.size() >= 2 ? ResolveValidShapeDim(tv.validShape[1], cols) : "";
    if (!resolvedRows.empty()) {
        typeRows = resolvedRows;
    }
    if (!resolvedCols.empty()) {
        typeCols = resolvedCols;
    }
}

} // namespace

std::string TypeConverter::ConvertTileType(const ir::MutexTileTypePtr& tileType, int64_t rows, int64_t cols) const
{
    return "SyncedTile<" + ConvertPtoTileType(tileType, rows, cols) + ", " + std::to_string(tileType->mutexCount_) +
           ">";
}

std::string TypeConverter::ConvertPtoTileType(const ir::TileTypePtr& tileType, int64_t rows, int64_t cols) const
{
    // A conv-layout TileType is a multi-dim pto::ConvTile; its shape is carried entirely by
    // ConvTileShape<...> and pto has no (rows, cols) constructor form for it.
    if (tileType->hardwareInfo_.has_value() && tileType->hardwareInfo_->convLayout >= 0) {
        return ConvertConvTileType(tileType);
    }
    std::ostringstream typeAlias;
    if (!tileType->memref_.has_value()) {
        // No memref: IfStmt return variable or intermediate tile.
        std::string tileTypeStr = "TileType::Vec";
        std::string BLayout = "RowMajor";
        std::string SLayout = "NoneBox";
        std::string fractal = "512";
        std::string padValue;
        // Default valid_shape template params to -1 (dynamic); overridden when user provides explicit values.
        std::string typeRows = "-1";
        std::string typeCols = "-1";
        std::string compactValue;
        if (tileType->hardwareInfo_.has_value()) {
            const auto& hw = tileType->hardwareInfo_.value();
            BLayout = ConvertTileLayout(hw.blayout);
            SLayout = ConvertTileLayout(hw.slayout);
            fractal = std::to_string(hw.fractal);
            padValue = ConvertTilePadToPTOValue(hw.pad);
            compactValue = ConvertCompactModeToPTOValue(hw.compact);
            using TL = ir::TileLayout;
            if (hw.blayout == TL::col_major && hw.slayout == TL::row_major)
                tileTypeStr = "TileType::Mat";
            else if (hw.blayout == TL::row_major && hw.slayout == TL::row_major)
                tileTypeStr = "TileType::Left";
            else if (hw.blayout == TL::row_major && hw.slayout == TL::col_major)
                tileTypeStr = "TileType::Right";
        }
        // A single-column tile has only its rows left to align, so ColMajor is the one encoding
        // the ISA accepts (BLayout::RowMajor requires Cols * sizeof(DType) % 32 == 0). It settles
        // the block layout alone -- every other hardware field above still applies.
        if (cols == 1) {
            BLayout = "ColMajor";
        }
        ApplyTileViewValidShape(tileType, rows, cols, typeRows, typeCols);
        typeAlias << "Tile<" << tileTypeStr << ", " << tileType->dtype_.ToCTypeString() << ", " << rows << ", " << cols
                  << ", BLayout::" << BLayout << ", " << typeRows << ", " << typeCols << ", SLayout::" << SLayout
                  << ", " << fractal;
        if (!padValue.empty()) {
            typeAlias << ", " << padValue;
        }
        if (!compactValue.empty()) {
            typeAlias << ", " << compactValue;
        }
        typeAlias << ">";
        return typeAlias.str();
    }
    ir::MemorySpace space = (*tileType->memref_)->memorySpace_; // NOLINT(bugprone-unchecked-optional-access)
    std::string tileTypeStr = ConvertMemorySpaceToTileType(space);

    // TODO(YunjiQin): BLayout and SLayout should be determined by the tile format
    std::string BLayout = "RowMajor";
    std::string SLayout = "NoneBox";
    std::string fractal = "512";
    std::string padValue;
    std::string compactValue;
    // Default valid_shape template params to -1 (dynamic); overridden when user provides explicit values.
    std::string typeRows = "-1";
    std::string typeCols = "-1";

    if (tileType->hardwareInfo_.has_value()) {
        const auto& hw = tileType->hardwareInfo_.value();
        BLayout = ConvertTileLayout(hw.blayout);
        SLayout = ConvertTileLayout(hw.slayout);
        fractal = std::to_string(hw.fractal);
        padValue = ConvertTilePadToPTOValue(hw.pad);
        compactValue = ConvertCompactModeToPTOValue(hw.compact);
    }
    // See the no-memref path above: a single-column tile settles its block layout by shape,
    // and only its block layout.
    if (cols == 1) {
        BLayout = "ColMajor";
    }
    ApplyTileViewValidShape(tileType, rows, cols, typeRows, typeCols);
    typeAlias << "Tile<" << tileTypeStr << ", " << tileType->dtype_.ToCTypeString() << ", " << rows << ", " << cols
              << ", BLayout::" << BLayout << ", " << typeRows << ", " << typeCols << ", SLayout::" << SLayout << ", "
              << fractal;
    if (!padValue.empty()) {
        typeAlias << ", " << padValue;
    }
    if (!compactValue.empty()) {
        typeAlias << ", " << compactValue;
    }
    typeAlias << ">";

    return typeAlias.str();
}

namespace {
// Whether a Layout ordinal is one of the conv data layouts that select the pto::ConvTile path.
bool IsConvLayout(const int layout)
{
    switch (static_cast<ir::Layout>(layout)) {
        case ir::Layout::NC1HWC0:
        case ir::Layout::NDC1HWC0:
        case ir::Layout::FRACTAL_Z:
        case ir::Layout::FRACTAL_Z_3D:
            return true;
        default:
            return false;
    }
}
} // namespace

std::string TypeConverter::ConvertConvTileType(const ir::TileTypePtr& tileType) const
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tileType->hardwareInfo_.has_value())
        << "conv TileType is missing hardware_info (conv_layout)";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, IsConvLayout(tileType->hardwareInfo_->convLayout))
        << "TileType carries an unknown conv layout ordinal " << tileType->hardwareInfo_->convLayout;
    // EnumToString spells the value "Layout::<VALUE>" (PYPTO_DECLARE_ENUM), which is the pto
    // spelling under the pto:: namespace.
    const char* ptoLayout = ir::EnumToString(static_cast<ir::Layout>(tileType->hardwareInfo_->convLayout));

    // BufferSize is the element count (the layout-shaped dimensions are already the packed
    // physical form), matching pto TileStorageBytes<ConvTile> = bufferSize * sizeof(DType).
    int64_t bufferSize = 1;
    std::string shapeParams;
    for (size_t i = 0; i < tileType->shape_.size(); ++i) {
        auto dim = ir::As<ir::ConstInt>(tileType->shape_[i]);
        bufferSize *= dim->value_;
        shapeParams += (i ? ", " : "") + std::to_string(dim->value_);
    }

    std::ostringstream oss;
    oss << "ConvTile<TileType::Mat, " << tileType->dtype_.ToCTypeString() << ", " << bufferSize
        << ", pto::" << ptoLayout << ", ConvTileShape<" << shapeParams << ">>";
    return oss.str();
}

std::string TypeConverter::ConvertMemorySpaceToTileType(ir::MemorySpace space) const
{
    switch (space) {
        case ir::MemorySpace::Left:
            return "TileType::Left";
        case ir::MemorySpace::Right:
            return "TileType::Right";
        case ir::MemorySpace::Acc:
            return "TileType::Acc";
        case ir::MemorySpace::Mat:
            return "TileType::Mat";
        case ir::MemorySpace::Vec:
            return "TileType::Vec";
        case ir::MemorySpace::Scaling:
            return "TileType::Scaling";
        case ir::MemorySpace::Bias:
            return "TileType::Bias";
        case ir::MemorySpace::ScaleLeft:
            return "TileType::ScaleLeft";
        case ir::MemorySpace::ScaleRight:
            return "TileType::ScaleRight";
        case ir::MemorySpace::DDR:
            // DDR is for GlobalTensor, not Tile - should not reach here
            PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_VAL)
                << "DDR is for GlobalTensor, not Tile";
        default:
            PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_ARGUMENT) << "Invalid MemorySpace value";
    }
}

std::string TypeConverter::ConvertPipeType(ir::PipeType pipe) const
{
    if (pipe == ir::PipeType::MTE1) {
        return "PIPE_MTE1";
    }
    if (pipe == ir::PipeType::MTE2) {
        return "PIPE_MTE2";
    }
    if (pipe == ir::PipeType::MTE3) {
        return "PIPE_MTE3";
    }
    if (pipe == ir::PipeType::M) {
        return "PIPE_M";
    }
    if (pipe == ir::PipeType::V) {
        return "PIPE_V";
    }
    if (pipe == ir::PipeType::S) {
        return "PIPE_S";
    }
    if (pipe == ir::PipeType::FIX) {
        return "PIPE_FIX";
    }
    if (pipe == ir::PipeType::ALL) {
        return "PIPE_ALL";
    }
    PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_ARGUMENT) << "Invalid PipeType value";
}

std::string TypeConverter::ConvertEventId(int eventId) const
{
    PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, eventId >= 0 && eventId <= 7)
        << "Event ID must be in range [0, 7], got " << eventId;
    return "EVENT_ID" + std::to_string(eventId);
}

std::string TypeConverter::ConvertCastRoundMode(int mode) const
{
    auto m = static_cast<ir::RoundMode>(mode);
    return ir::EnumToString(m);
}

std::string TypeConverter::ConvertTileLayout(ir::TileLayout layout) const
{
    switch (layout) {
        case ir::TileLayout::none_box:
            return "NoneBox";
        case ir::TileLayout::row_major:
            return "RowMajor";
        case ir::TileLayout::col_major:
            return "ColMajor";
        default:
            PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::INVALID_FORMAT) << "Invalid TileLayout value";
    }
}

std::string TypeConverter::GenerateShapeType(const std::vector<int64_t>& dims) const
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, !dims.empty()) << "Cannot generate Shape type for empty dimensions";

    std::ostringstream oss;
    oss << "pto::Shape<";

    // Pad to 5 dimensions with leading 1s
    const size_t targetDims = 5;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, dims.size() <= targetDims)
        << "Cannot generate Shape with more than " << targetDims << " dimensions, got " << dims.size();

    // Add leading 1s for padding
    for (size_t i = 0; i < targetDims - dims.size(); ++i) {
        oss << "1, ";
    }

    // Add actual dimensions
    for (size_t i = 0; i < dims.size(); ++i) {
        oss << dims[i];
        if (i < dims.size() - 1) {
            oss << ", ";
        }
    }

    oss << ">";
    return oss.str();
}

std::string TypeConverter::GenerateStrideType(const std::vector<int64_t>& shape) const
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, !shape.empty()) << "Cannot generate Stride type for empty shape";

    std::ostringstream oss;
    oss << "pto::Stride<";

    // Pad to 5 dimensions with leading 1s
    const size_t targetDims = 5;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, shape.size() <= targetDims)
        << "Cannot generate Stride with more than " << targetDims << " dimensions, got " << shape.size();

    // Add leading 1s for padding
    for (size_t i = 0; i < targetDims - shape.size(); ++i) {
        oss << "1, ";
    }

    // set dynamic strides, will get from runtime
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << "-1";
        if (i < shape.size() - 1) {
            oss << ", ";
        }
    }

    oss << ">";
    return oss.str();
}

} // namespace codegen

} // namespace pypto
