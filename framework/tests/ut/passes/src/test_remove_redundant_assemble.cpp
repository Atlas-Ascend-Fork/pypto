/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_remove_redundant_assemble.cpp
 * \brief Unit test for helpers of RemoveRedundantAssemble pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"

using namespace npu::tile_fwk;

// remove_redundant_assemble.cpp 内的文件内自由函数(外部链接), 此处补充声明后按白盒方式直接验证
namespace npu {
namespace tile_fwk {
bool CalculateNewRawShapeExpand(const std::vector<int64_t>& newShape, const std::vector<int64_t>& oriRawShape,
                                std::vector<int64_t>& newRawShape);
bool CalculateNewDynRawShapeExpand(const std::vector<int64_t>& newShape,
                                   const std::vector<SymbolicScalar>& oriDynRawShape,
                                   std::vector<SymbolicScalar>& newDynRawShape);
bool CalculateNewDynRawShapeReduce(const std::vector<int64_t>& newShape,
                                   const std::vector<SymbolicScalar>& oriDynRawShape,
                                   std::vector<SymbolicScalar>& newDynRawShape);
void GetDynOffsetBeforeReshape(const std::vector<SymbolicScalar>& oriOffset, const std::vector<int64_t>& oriShape,
                               const std::vector<int64_t>& newShape, std::vector<SymbolicScalar>& newOffset);
} // namespace tile_fwk
} // namespace npu

class TestRemoveRedundantAssemble : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(TestRemoveRedundantAssemble, CalculateNewRawShapeExpandValidCase)
{
    std::vector<int64_t> newShape = {2, 4};
    std::vector<int64_t> oriRawShape = {2, 2, 4};
    std::vector<int64_t> newRawShape;
    EXPECT_TRUE(CalculateNewRawShapeExpand(newShape, oriRawShape, newRawShape));
    EXPECT_EQ(newRawShape, std::vector<int64_t>({4, 4}));
}

// 尾维含0导致newShapeSize为0时无法整除, 打INFO日志并返回false
TEST_F(TestRemoveRedundantAssemble, CalculateNewRawShapeExpandZeroTailDim)
{
    std::vector<int64_t> newShape = {2, 3};
    std::vector<int64_t> oriRawShape = {4, 0};
    std::vector<int64_t> newRawShape;
    EXPECT_FALSE(CalculateNewRawShapeExpand(newShape, oriRawShape, newRawShape));
    EXPECT_EQ(newRawShape, std::vector<int64_t>({4, 0}));
}

// 动态形态expand场景尾维为0, 打ERROR日志并返回false
TEST_F(TestRemoveRedundantAssemble, CalculateNewDynRawShapeExpandZeroTailDim)
{
    std::vector<int64_t> newShape = {2, 3};
    std::vector<SymbolicScalar> oriDynRawShape = CreateTestConstIntVector({4, 0});
    std::vector<SymbolicScalar> newDynRawShape;
    EXPECT_FALSE(CalculateNewDynRawShapeExpand(newShape, oriDynRawShape, newDynRawShape));
    EXPECT_EQ(newDynRawShape.size(), 2U);
}

// 动态形态reduce场景newShape含0维, 打ERROR日志并返回false
TEST_F(TestRemoveRedundantAssemble, CalculateNewDynRawShapeReduceZeroNewShapeDim)
{
    std::vector<int64_t> newShape = {0, 3};
    std::vector<SymbolicScalar> oriDynRawShape = CreateTestConstIntVector({4, 4});
    std::vector<SymbolicScalar> newDynRawShape;
    EXPECT_FALSE(CalculateNewDynRawShapeReduce(newShape, oriDynRawShape, newDynRawShape));
    EXPECT_EQ(newDynRawShape.size(), 2U);
}

// oriShape与oriOffset维数不一致时打ERROR日志并直接返回, 不产出newOffset
TEST_F(TestRemoveRedundantAssemble, GetDynOffsetBeforeReshapeRankMismatch)
{
    std::vector<SymbolicScalar> oriOffset = CreateTestConstIntVector({1, 2});
    std::vector<int64_t> oriShape = {2, 2, 2};
    std::vector<int64_t> newShape = {4, 4, 2};
    std::vector<SymbolicScalar> newOffset;
    GetDynOffsetBeforeReshape(oriOffset, oriShape, newShape, newOffset);
    EXPECT_TRUE(newOffset.empty());
}
