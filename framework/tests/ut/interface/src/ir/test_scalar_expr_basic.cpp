/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scalar_expr_basic.cpp
 * \brief Unit tests for basic scalar expressions (constants only)
 */

#include "gtest/gtest.h"

#include <memory>

#include "core/dtype.h"
#include "ir/scalar_expr_ops.h"

namespace pypto {
namespace ir {

TEST(ScalarExprBasicTest, RejectsStorageOnlyScalarArithmetic)
{
    for (auto dtype : {DataType::FP4, DataType::FP8E4M3FN, DataType::FP8E5M2, DataType::FP8E8M0, DataType::FP4E2M1,
                       DataType::FP4E1M2, DataType::HF4, DataType::HF8, DataType::INT4, DataType::UINT4}) {
        EXPECT_TRUE(dtype.IsStorageOnly());
        auto value = std::make_shared<const Var>("value", std::make_shared<const ScalarType>(dtype), Span::Unknown());
        EXPECT_THROW((void)MakeMin(value, value), ValueError);
        EXPECT_THROW((void)MakeMax(value, value), ValueError);
        EXPECT_THROW((void)MakeAdd(value, value), ValueError);
    }
    for (auto dtype : {DataType::BOOL, DataType::INT8, DataType::UINT8, DataType::INT64, DataType::UINT64,
                       DataType::INDEX, DataType::FP16, DataType::BF16, DataType::FP32, DataType::FP64}) {
        EXPECT_FALSE(dtype.IsStorageOnly());
    }
}

TEST(ScalarExprBasicTest, LogicalOperationsPreserveNumericOperands)
{
    for (auto dtype : {DataType::BOOL, DataType::INT32, DataType::INT64, DataType::UINT64, DataType::FP32}) {
        auto left = std::make_shared<const Var>("left", std::make_shared<const ScalarType>(dtype), Span::Unknown());
        for (auto floating : {DataType::FP16, DataType::BF16, DataType::FP32}) {
            auto right = std::make_shared<const Var>("right", std::make_shared<const ScalarType>(floating),
                                                     Span::Unknown());
            for (auto make_op : {MakeAnd, MakeOr, MakeXor}) {
                for (bool reverse : {false, true}) {
                    const auto& lhs = reverse ? right : left;
                    const auto& rhs = reverse ? left : right;
                    auto result = std::dynamic_pointer_cast<const BinaryExpr>(make_op(lhs, rhs, Span::Unknown()));
                    ASSERT_NE(result, nullptr);
                    EXPECT_EQ(result->left_, lhs);
                    EXPECT_EQ(result->right_, rhs);
                    EXPECT_EQ(GetScalarDtype(result), DataType::BOOL);
                }
            }
        }
    }
}

TEST(ScalarExprBasicTest, IntegerTrueDivisionPromotesToFloat32)
{
    for (auto dtype : {DataType::BOOL, DataType::INT32, DataType::INT64, DataType::UINT64}) {
        auto value = std::make_shared<const Var>("value", std::make_shared<const ScalarType>(dtype), Span::Unknown());
        auto result = std::dynamic_pointer_cast<const FloatDiv>(MakeFloatDiv(value, value));
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(GetScalarDtype(result), DataType::FP32);
        EXPECT_EQ(GetScalarDtype(result->left_), DataType::FP32);
        EXPECT_EQ(GetScalarDtype(result->right_), DataType::FP32);
    }
}

// ============================================================================
// ConstInt Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstIntInt32)
{
    // Test ConstInt with INT32
    auto constInt = std::make_shared<ConstInt>(42, DataType::INT32, Span::Unknown());
    ASSERT_NE(constInt, nullptr);
    ASSERT_EQ(constInt->TypeName(), "ConstInt");
    ASSERT_EQ(constInt->value_, 42);
    ASSERT_EQ(constInt->dtype(), DataType::INT32);
}

TEST(ScalarExprBasicTest, TestConstIntZero)
{
    // Test ConstInt with zero value
    auto constInt = std::make_shared<ConstInt>(0, DataType::INT32, Span::Unknown());
    ASSERT_NE(constInt, nullptr);
    ASSERT_EQ(constInt->value_, 0);
}

TEST(ScalarExprBasicTest, TestConstIntNegative)
{
    // Test ConstInt with negative value
    auto constInt = std::make_shared<ConstInt>(-100, DataType::INT32, Span::Unknown());
    ASSERT_NE(constInt, nullptr);
    ASSERT_EQ(constInt->value_, -100);
}

TEST(ScalarExprBasicTest, TestConstIntInt64)
{
    // Test ConstInt with INT64
    auto constInt = std::make_shared<ConstInt>(1000000, DataType::INT64, Span::Unknown());
    ASSERT_NE(constInt, nullptr);
    ASSERT_EQ(constInt->dtype(), DataType::INT64);
}

TEST(ScalarExprBasicTest, TestConstIntInt8)
{
    // Test ConstInt with INT8
    auto constInt = std::make_shared<ConstInt>(127, DataType::INT8, Span::Unknown());
    ASSERT_NE(constInt, nullptr);
    ASSERT_EQ(constInt->dtype(), DataType::INT8);
}

TEST(ScalarExprBasicTest, TestConstIntUInt32)
{
    // Test ConstInt with UINT32
    auto constInt = std::make_shared<ConstInt>(255, DataType::UINT32, Span::Unknown());
    ASSERT_NE(constInt, nullptr);
    ASSERT_EQ(constInt->dtype(), DataType::UINT32);
}

TEST(ScalarExprBasicTest, TestConstIntWithSpan)
{
    // Test ConstInt with valid span
    Span span("test.py", 10, 5);
    auto constInt = std::make_shared<ConstInt>(42, DataType::INT32, span);
    ASSERT_NE(constInt, nullptr);
    ASSERT_EQ(constInt->span_.Filename(), "test.py");
    ASSERT_EQ(constInt->span_.BeginLine(), 10);
}

// ============================================================================
// ConstFloat Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstFloatFP32)
{
    // Test ConstFloat with FP32
    auto constFloat = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::Unknown());
    ASSERT_NE(constFloat, nullptr);
    ASSERT_EQ(constFloat->TypeName(), "ConstFloat");
    ASSERT_DOUBLE_EQ(constFloat->value_, 3.14);
    ASSERT_EQ(constFloat->dtype(), DataType::FP32);
}

TEST(ScalarExprBasicTest, TestConstFloatZero)
{
    // Test ConstFloat with zero value
    auto constFloat = std::make_shared<ConstFloat>(0.0, DataType::FP32, Span::Unknown());
    ASSERT_NE(constFloat, nullptr);
    ASSERT_DOUBLE_EQ(constFloat->value_, 0.0);
}

TEST(ScalarExprBasicTest, TestConstFloatNegative)
{
    // Test ConstFloat with negative value
    auto constFloat = std::make_shared<ConstFloat>(-2.5, DataType::FP32, Span::Unknown());
    ASSERT_NE(constFloat, nullptr);
    ASSERT_DOUBLE_EQ(constFloat->value_, -2.5);
}

TEST(ScalarExprBasicTest, TestConstFloatFP16)
{
    // Test ConstFloat with FP16
    auto constFloat = std::make_shared<ConstFloat>(1.5, DataType::FP16, Span::Unknown());
    ASSERT_NE(constFloat, nullptr);
    ASSERT_EQ(constFloat->dtype(), DataType::FP16);
}

TEST(ScalarExprBasicTest, TestConstFloatBF16)
{
    // Test ConstFloat with BF16
    auto constFloat = std::make_shared<ConstFloat>(2.5, DataType::BF16, Span::Unknown());
    ASSERT_NE(constFloat, nullptr);
    ASSERT_EQ(constFloat->dtype(), DataType::BF16);
}

TEST(ScalarExprBasicTest, TestConstFloatLargeValue)
{
    // Test ConstFloat with large value
    auto constFloat = std::make_shared<ConstFloat>(1e10, DataType::FP32, Span::Unknown());
    ASSERT_NE(constFloat, nullptr);
    ASSERT_DOUBLE_EQ(constFloat->value_, 1e10);
}

TEST(ScalarExprBasicTest, TestConstFloatSmallValue)
{
    // Test ConstFloat with small value
    auto constFloat = std::make_shared<ConstFloat>(1e-10, DataType::FP32, Span::Unknown());
    ASSERT_NE(constFloat, nullptr);
    ASSERT_DOUBLE_EQ(constFloat->value_, 1e-10);
}

// ============================================================================
// ConstBool Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstBoolTrue)
{
    // Test ConstBool with true value
    auto constBool = std::make_shared<ConstBool>(true, Span::Unknown());
    ASSERT_NE(constBool, nullptr);
    ASSERT_EQ(constBool->TypeName(), "ConstBool");
    ASSERT_TRUE(constBool->value_);
    ASSERT_EQ(constBool->dtype(), DataType::BOOL);
}

TEST(ScalarExprBasicTest, TestConstBoolFalse)
{
    // Test ConstBool with false value
    auto constBool = std::make_shared<ConstBool>(false, Span::Unknown());
    ASSERT_NE(constBool, nullptr);
    ASSERT_FALSE(constBool->value_);
    ASSERT_EQ(constBool->dtype(), DataType::BOOL);
}

TEST(ScalarExprBasicTest, TestConstBoolWithSpan)
{
    // Test ConstBool with valid span
    Span span("test.py", 20, 10);
    auto constBool = std::make_shared<ConstBool>(true, span);
    ASSERT_NE(constBool, nullptr);
    ASSERT_EQ(constBool->span_.Filename(), "test.py");
    ASSERT_EQ(constBool->span_.BeginLine(), 20);
}

// ============================================================================
// Type Checking Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstIntGetType)
{
    // Test ConstInt GetType method
    auto constInt = std::make_shared<ConstInt>(42, DataType::INT32, Span::Unknown());
    auto type = constInt->GetType();
    ASSERT_NE(type, nullptr);
    ASSERT_EQ(type->TypeName(), "ScalarType");

    auto scalarType = std::dynamic_pointer_cast<const ScalarType>(type);
    ASSERT_NE(scalarType, nullptr);
    ASSERT_EQ(scalarType->dtype_, DataType::INT32);
}

TEST(ScalarExprBasicTest, TestConstFloatGetType)
{
    // Test ConstFloat GetType method
    auto constFloat = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::Unknown());
    auto type = constFloat->GetType();
    ASSERT_NE(type, nullptr);
    ASSERT_EQ(type->TypeName(), "ScalarType");

    auto scalarType = std::dynamic_pointer_cast<const ScalarType>(type);
    ASSERT_NE(scalarType, nullptr);
    ASSERT_EQ(scalarType->dtype_, DataType::FP32);
}

TEST(ScalarExprBasicTest, TestConstBoolGetType)
{
    // Test ConstBool GetType method
    auto constBool = std::make_shared<ConstBool>(true, Span::Unknown());
    auto type = constBool->GetType();
    ASSERT_NE(type, nullptr);
    ASSERT_EQ(type->TypeName(), "ScalarType");

    auto scalarType = std::dynamic_pointer_cast<const ScalarType>(type);
    ASSERT_NE(scalarType, nullptr);
    ASSERT_EQ(scalarType->dtype_, DataType::BOOL);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestGetScalarDtype)
{
    // Test GetScalarDtype helper function
    auto constInt = std::make_shared<ConstInt>(42, DataType::INT32, Span::Unknown());
    DataType dtype = GetScalarDtype(constInt);
    ASSERT_EQ(dtype, DataType::INT32);
}

TEST(ScalarExprBasicTest, TestIsBoolDtype)
{
    // Test IsBoolDtype helper function
    ASSERT_TRUE(IsBoolDtype(DataType::BOOL));
    ASSERT_FALSE(IsBoolDtype(DataType::INT32));
    ASSERT_FALSE(IsBoolDtype(DataType::FP32));
}

TEST(ScalarExprBasicTest, TestGetNumericCategoryInt)
{
    // Test GetNumericCategory for integer types
    auto category = GetNumericCategory(DataType::INT32, "test");
    ASSERT_EQ(category, ScalarCategory::INT);
}

TEST(ScalarExprBasicTest, TestGetNumericCategoryFloat)
{
    // Test GetNumericCategory for float types
    auto category = GetNumericCategory(DataType::FP32, "test");
    ASSERT_EQ(category, ScalarCategory::FLOAT);
}

} // namespace ir
} // namespace pypto
