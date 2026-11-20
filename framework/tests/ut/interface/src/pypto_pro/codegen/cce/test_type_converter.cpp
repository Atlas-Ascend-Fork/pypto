/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "codegen/cce/type_converter.h"
#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"
#include "tilefwk/error.h"

namespace pypto {
namespace codegen {

namespace {
ir::TileTypePtr ConvTile(const std::vector<int64_t>& shape, ir::DataType dtype, ir::Layout layout,
                         const std::vector<int64_t>& validShape = {})
{
    std::vector<ir::ExprPtr> dims;
    for (int64_t d : shape) {
        dims.push_back(std::make_shared<ir::ConstInt>(d, ir::DataType::INT64, ir::Span::Unknown()));
    }
    ir::HardwareInfo hw;
    hw.convLayout = static_cast<int>(layout);
    std::optional<ir::TileView> view;
    if (!validShape.empty()) {
        std::vector<ir::ExprPtr> validDims;
        for (int64_t v : validShape) {
            validDims.push_back(std::make_shared<ir::ConstInt>(v, ir::DataType::INT64, ir::Span::Unknown()));
        }
        view = ir::TileView(validDims);
    }
    return std::make_shared<ir::TileType>(dims, dtype, std::nullopt, view, hw);
}
} // namespace

TEST(TypeConverterTest, ConvertsAllCastRoundModesThroughEnumReflection)
{
    TypeConverter converter;
    const std::vector<std::string> expected = {
        "RoundMode::CAST_NONE", "RoundMode::CAST_RINT",  "RoundMode::CAST_ROUND", "RoundMode::CAST_FLOOR",
        "RoundMode::CAST_CEIL", "RoundMode::CAST_TRUNC", "RoundMode::CAST_ODD",
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(converter.ConvertCastRoundMode(static_cast<int>(i)), expected[i]);
    }
    EXPECT_EQ(converter.ConvertCastRoundMode(99), "UNKNOWN");
}

// The conv tile's shape contract (per-layout rank/C0/16 rules) is validated by the Python
// front end for DSL-built tiles (pypto_pro/ir/op/block_ops.py); codegen emits the declared
// shape as given and mirrors a declared valid_shape into the optional ValidShape_ template
// parameter.
TEST(TypeConverterTest, ConvTileTypeEmittedFromDeclaredShape)
{
    TypeConverter converter;
    // Valid NC1HWC0 [N, C1, H, W, C0]: C0 = 32B / 2B = 16 for FP16. The 6th ValidShape_
    // parameter is always present: the IR constructor normalizes an omitted valid_shape to
    // all -1 (DYNAMIC), so the tile is runtime-settable.
    EXPECT_EQ(converter.ConvertPtoTileType(ConvTile({1, 4, 8, 8, 16}, ir::DataType::FP16, ir::Layout::NC1HWC0), 0, 0),
              "ConvTile<TileType::Mat, half, 4096, pto::Layout::NC1HWC0, ConvTileShape<1, 4, 8, 8, 16>, "
              "ConvTileShape<-1, -1, -1, -1, -1>>");
}

TEST(TypeConverterTest, ConvTileValidShapeEmitsSixthTemplateParameter)
{
    TypeConverter converter;
    // All-dynamic declaration: every ValidShape_ dim is -1 (DYNAMIC, runtime-settable).
    EXPECT_EQ(converter.ConvertPtoTileType(
                  ConvTile({1, 4, 8, 8, 16}, ir::DataType::FP16, ir::Layout::NC1HWC0, {-1, -1, -1, -1, -1}), 0, 0),
              "ConvTile<TileType::Mat, half, 4096, pto::Layout::NC1HWC0, ConvTileShape<1, 4, 8, 8, 16>, "
              "ConvTileShape<-1, -1, -1, -1, -1>>");
    // Mixed declaration: static dims keep their constant, -1 dims stay dynamic.
    EXPECT_EQ(converter.ConvertPtoTileType(
                  ConvTile({64, 4, 16, 16}, ir::DataType::FP16, ir::Layout::FRACTAL_Z, {-1, -1, 16, 16}), 0, 0),
              "ConvTile<TileType::Mat, half, 65536, pto::Layout::FRACTAL_Z, ConvTileShape<64, 4, 16, 16>, "
              "ConvTileShape<-1, -1, 16, 16>>");
}

} // namespace codegen
} // namespace pypto
