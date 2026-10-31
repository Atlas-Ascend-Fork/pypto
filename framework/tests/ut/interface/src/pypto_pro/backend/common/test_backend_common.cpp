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

#include "backend/common/backend_registry.h"
#include "backend/common/backend_utils.h"
#include "backend/common/soc.h"
#include "core/error.h"

namespace pypto {
namespace backend {

TEST(BackendCommonTest, RegistersCCEAsSingletonBackend)
{
    auto& registry = BackendRegistry::Instance();
    EXPECT_TRUE(registry.IsRegistered("CCE"));
    EXPECT_FALSE(registry.IsRegistered("910B_CCE"));
    EXPECT_THROW(registry.Create("CCE", nullptr), ir::ValueError);
    EXPECT_THROW(CreateBackendFromRegistry("CCE", nullptr), ir::ValueError);
}

TEST(BackendCommonTest, CreatesSingletonSoC)
{
    const SoC& first = CreateSoC();
    const SoC& second = CreateSoC();

    EXPECT_EQ(&first, &second);
    EXPECT_GT(first.TotalDieCount(), 0);
    EXPECT_GT(first.TotalCoreCount(), 0);
}

TEST(BackendCommonTest, SupportsPointerPrintfConversion)
{
    EXPECT_TRUE(debug_printf::IsSupportedPrintfConversion('p'));
    EXPECT_EQ(debug_printf::FindPrintfConversionIndex("address=%p"), 9);

    auto segments = debug_printf::ParsePrintfSegments("address=%p");
    ASSERT_EQ(segments.size(), 1);
    EXPECT_EQ(segments[0].format_segment, "address=%p");
    EXPECT_EQ(segments[0].conversion, 'p');
}

TEST(BackendCommonTest, ResolvesTileTensorLayout)
{
    using ir::HardwareInfo;
    using ir::TensorLayout;
    using ir::TileLayout;

    EXPECT_EQ(cce::GetTileTensorLayout(HardwareInfo(TileLayout::row_major, TileLayout::none_box)), TensorLayout::ND);
    EXPECT_EQ(cce::GetTileTensorLayout(HardwareInfo(TileLayout::col_major, TileLayout::none_box)), TensorLayout::DN);
    EXPECT_EQ(cce::GetTileTensorLayout(HardwareInfo(TileLayout::col_major, TileLayout::row_major)), TensorLayout::NZ);
    EXPECT_EQ(cce::GetTileTensorLayout(HardwareInfo(TileLayout::row_major, TileLayout::col_major)), TensorLayout::ZN);
    EXPECT_EQ(cce::GetTileTensorLayout(HardwareInfo(TileLayout::col_major, TileLayout::col_major)), TensorLayout::NN);
    EXPECT_EQ(cce::GetTileTensorLayout(HardwareInfo(TileLayout::row_major, TileLayout::row_major)), TensorLayout::ZZ);
    EXPECT_FALSE(cce::GetTileTensorLayout(HardwareInfo(TileLayout::none_box, TileLayout::none_box)).has_value());
}

} // namespace backend
} // namespace pypto
