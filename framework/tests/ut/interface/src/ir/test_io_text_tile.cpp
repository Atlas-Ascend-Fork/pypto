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

#include <array>
#include <any>
#include <type_traits>
#include "ir/stmt.h"
#include "ir/transforms/io_text.h"
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "tilefwk/tile_shape.h"

using namespace pypto::ir;

namespace {
template <typename Tile>
const char* TileAttrName()
{
    if constexpr (std::is_same_v<Tile, CubeTile>) {
        return "cube_tile_shapes";
    }
    if constexpr (std::is_same_v<Tile, ConvTile>) {
        return "conv_tile_shapes";
    }
    if constexpr (std::is_same_v<Tile, ConvBpTile>) {
        return "convbp_tile_shapes";
    }
    return "dist_tile_shapes";
}

std::string DumpAttr(const std::string& key, const std::any& value)
{
    auto stmt = std::make_shared<TensorOpStmt>(
        std::vector<VarPtr>{}, std::vector<VarPtr>{}, "OP_CONFIG", std::vector<ExprPtr>{}, std::vector<VarPtr>{},
        std::vector<std::pair<std::string, std::any>>{{key, value}}, Span::Unknown());
    return TextDump(stmt);
}

// Exercise the public text IO boundary instead of exposing its internal codecs to tests.
template <typename Tile>
std::vector<int64_t> DumpTileValues(const Tile& tile)
{
    std::string error;
    auto stmt = std::dynamic_pointer_cast<const TensorOpStmt>(TextLoadStmt(DumpAttr("values", tile), error));
    EXPECT_NE(stmt, nullptr) << error;
    if (!stmt) {
        return {};
    }
    const auto& values = std::any_cast<const std::vector<std::any>&>(stmt->attrs_.at(0).second);
    std::vector<int64_t> result;
    for (const auto& value : values) {
        result.push_back(std::any_cast<int64_t>(value));
    }
    return result;
}

template <typename Tile>
bool LoadTileValues(const std::vector<int64_t>& values, Tile& out, std::string& error)
{
    auto stmt = std::dynamic_pointer_cast<const TensorOpStmt>(
        TextLoadStmt(DumpAttr(TileAttrName<Tile>(), values), error));
    if (!stmt) {
        return false;
    }
    out = std::any_cast<Tile>(stmt->attrs_.at(0).second);
    return true;
}

bool SameCube(const CubeTile& lhs, const CubeTile& rhs)
{
    return lhs.m == rhs.m && lhs.k == rhs.k && lhs.n == rhs.n && lhs.enableSplitK == rhs.enableSplitK;
}
} // namespace

TEST(IoTextTileTest, CubeKeepsStoredK2Zero)
{
    CubeTile tile;
    tile.m = {1, 2};
    tile.k = {3, 4, 0};
    tile.n = {5, 6};
    tile.enableSplitK = false;

    const std::vector<int64_t> encoded = DumpTileValues(tile);
    ASSERT_EQ(encoded.size(), 8);
    EXPECT_EQ(encoded[4], 0);
    EXPECT_EQ(encoded[7], 0);

    CubeTile decoded;
    decoded.k = {9, 9, 9};
    std::string error;
    ASSERT_TRUE(LoadTileValues(encoded, decoded, error)) << error;
    EXPECT_EQ(decoded.k[2], 0);
    EXPECT_FALSE(decoded.enableSplitK);
}

TEST(IoTextTileTest, ConvKeepsSetL0TileWhenL0IsZero)
{
    ConvTile tile;
    tile.setL0Tile = true;

    const std::vector<int64_t> encoded = DumpTileValues(tile);
    ASSERT_EQ(encoded.size(), 13);
    EXPECT_EQ(encoded[8], 0);
    EXPECT_EQ(encoded[9], 0);
    EXPECT_EQ(encoded[10], 0);
    EXPECT_EQ(encoded[11], 0);
    EXPECT_EQ(encoded[12], 1);

    ConvTile decoded;
    std::string error;
    ASSERT_TRUE(LoadTileValues(encoded, decoded, error)) << error;
    EXPECT_TRUE(decoded.setL0Tile);
    EXPECT_EQ(decoded.tileL0Info.tileH, 0);
    EXPECT_EQ(decoded.tileL0Info.tileW, 0);
    EXPECT_EQ(decoded.tileL0Info.tileK, 0);
    EXPECT_EQ(decoded.tileL0Info.tileN, 0);
}

TEST(IoTextTileTest, DistIntRangeOnArraySlotAndRankId)
{
    const int64_t kMin = std::numeric_limits<int>::min();
    const int64_t kMax = std::numeric_limits<int>::max();

    DistTile decoded;
    std::string error;
    std::vector<int64_t> values{kMin, 2, 3, 4, 5, 6, 7, 8, 9, kMax};
    ASSERT_TRUE(LoadTileValues(values, decoded, error)) << error;
    EXPECT_EQ(decoded.row, (std::array<int, 3>{static_cast<int>(kMin), 2, 3}));
    EXPECT_EQ(decoded.col, (std::array<int, 3>{4, 5, 6}));
    EXPECT_EQ(decoded.rank, (std::array<int, 3>{7, 8, 9}));
    EXPECT_EQ(decoded.rankId, static_cast<int>(kMax));
    EXPECT_EQ(DumpTileValues(decoded), values);

    values[0] = kMin - 1;
    EXPECT_FALSE(LoadTileValues(values, decoded, error));
    EXPECT_NE(error.find("row[0]"), std::string::npos);
    EXPECT_EQ(decoded.row[0], static_cast<int>(kMin));

    values[0] = 1;
    values[9] = kMax + 1;
    EXPECT_FALSE(LoadTileValues(values, decoded, error));
    EXPECT_NE(error.find("rankId"), std::string::npos);
    EXPECT_EQ(decoded.rankId, static_cast<int>(kMax));
}

TEST(IoTextTileTest, ConvBpRoundTrip)
{
    ConvBpTile tile;
    tile.tileL1Info = {16, 32, 64};
    tile.tileL0Info = {8, 16, 32};
    EXPECT_EQ(DumpTileValues(tile), (std::vector<int64_t>{16, 32, 64, 8, 16, 32}));
    ConvBpTile decoded;
    std::string error;
    ASSERT_TRUE(LoadTileValues(DumpTileValues(tile), decoded, error)) << error;
    EXPECT_EQ(decoded.tileL1Info.tileML1, 16);
    EXPECT_EQ(decoded.tileL1Info.tileKL1, 32);
    EXPECT_EQ(decoded.tileL1Info.tileNL1, 64);
    EXPECT_EQ(decoded.tileL0Info.tileML0, 8);
    EXPECT_EQ(decoded.tileL0Info.tileKL0, 16);
    EXPECT_EQ(decoded.tileL0Info.tileNL0, 32);
}

TEST(IoTextTileTest, DimensionWidthsAndConvFieldOrder)
{
    const int64_t wide = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
    CubeTile cube{{wide, 2}, {3, 4, 5}, {6, 7}, true};
    EXPECT_EQ(DumpTileValues(cube), (std::vector<int64_t>{wide, 2, 3, 4, 5, 6, 7, 1}));
    CubeTile decodedCube{};
    std::string error;
    ASSERT_TRUE(LoadTileValues(DumpTileValues(cube), decodedCube, error)) << error;
    EXPECT_TRUE(SameCube(cube, decodedCube));

    const std::vector<int64_t> values{wide, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1};
    ConvTile conv;
    ASSERT_TRUE(LoadTileValues(values, conv, error)) << error;
    EXPECT_EQ(conv.tileL1Info.tileHin, wide);
    EXPECT_EQ(conv.tileL1Info.tileHout, 2);
    EXPECT_EQ(conv.tileL1Info.tileWin, 3);
    EXPECT_EQ(conv.tileL1Info.tileWout, 4);
    EXPECT_EQ(conv.tileL1Info.tileCinFmap, 5);
    EXPECT_EQ(conv.tileL1Info.tileCinWeight, 6);
    EXPECT_EQ(conv.tileL1Info.tileN, 7);
    EXPECT_EQ(conv.tileL1Info.tileBatch, 8);
    EXPECT_EQ(conv.tileL0Info.tileH, 9);
    EXPECT_EQ(conv.tileL0Info.tileW, 10);
    EXPECT_EQ(conv.tileL0Info.tileK, 11);
    EXPECT_EQ(conv.tileL0Info.tileN, 12);
    EXPECT_TRUE(conv.setL0Tile);
    EXPECT_EQ(DumpTileValues(conv), values);
}

namespace {
template <typename Tile>
void CheckInvalidLengths(size_t expected)
{
    Tile original{};
    for (size_t size : {size_t{0}, expected - 1, expected + 1}) {
        Tile out = original;
        std::string error;
        EXPECT_FALSE(LoadTileValues(std::vector<int64_t>(size, 1), out, error));
        EXPECT_NE(error.find("expected " + std::to_string(expected)), std::string::npos);
    }
}
} // namespace

TEST(IoTextTileTest, RejectsIncorrectLengthsForAllTileTypes)
{
    CheckInvalidLengths<CubeTile>(8);
    CheckInvalidLengths<ConvTile>(13);
    CheckInvalidLengths<ConvBpTile>(6);
    CheckInvalidLengths<DistTile>(10);
}
