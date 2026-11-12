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

#include <cmath>
#include <cstring>
#include <limits>

#include "codegen/cce/cce_codegen.h"
#include "ir/scalar_expr_ops.h"
#include "pypto_scalar.h"

namespace pypto::codegen {

TEST(ScalarBFloat16Test, RoundsEveryFP32BoundaryToNearestEven)
{
    for (uint32_t high = 0; high <= 0xffff; ++high) {
        for (uint32_t low : {0u, 1u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
            uint32_t bits = (high << 16) | low;
            uint16_t actual = pypto_scalar::FloatBitsToBFloat16(bits);
            if ((bits & 0x7fffffffu) > 0x7f800000u) {
                EXPECT_EQ(actual, high | 0x40u);
            } else {
                uint16_t expected = high + (low > 0x8000u || (low == 0x8000u && (high & 1)));
                ASSERT_EQ(actual, expected) << bits;
            }
        }
    }
}

TEST(ScalarBFloat16Test, IntegerConversionAvoidsDoubleRounding)
{
    for (unsigned shift = 1; shift <= 56; ++shift) {
        for (uint64_t significand : {128u, 129u, 254u, 255u}) {
            const uint64_t midpoint = (significand << shift) + (uint64_t(1) << (shift - 1));
            const uint16_t lower = ((shift + 133u) << 7) + significand;
            EXPECT_EQ(pypto_scalar::IntegerToBFloat16(midpoint - 1), lower);
            EXPECT_EQ(pypto_scalar::IntegerToBFloat16(midpoint), lower + (significand & 1));
            EXPECT_EQ(pypto_scalar::IntegerToBFloat16(midpoint + 1), lower + 1);
            if (shift < 56) {
                EXPECT_EQ(pypto_scalar::IntegerToBFloat16(-int64_t(midpoint + 1)), (lower + 1) | 0x8000);
            }
        }
    }
    EXPECT_EQ(pypto_scalar::IntegerToBFloat16(std::numeric_limits<int64_t>::min()), 0xdf00);
    EXPECT_EQ(pypto_scalar::IntegerToBFloat16(std::numeric_limits<uint64_t>::max()), 0x5f80);
    EXPECT_EQ(pypto_scalar::IntegerToBFloat16(false), 0);
    EXPECT_EQ(pypto_scalar::IntegerToBFloat16(true), 0x3f80);
}

TEST(ScalarBFloat16Test, IntegerAndBoolConversionsMatchAllInRangeEncodings)
{
    for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
        uint32_t fp32_bits = bits << 16;
        float value;
        std::memcpy(&value, &fp32_bits, sizeof(value));
        EXPECT_EQ(pypto_scalar::BFloat16ToInteger<bool>(bits), std::fpclassify(value) != FP_ZERO);
        if (value >= -std::ldexp(1.0f, 63) && value < std::ldexp(1.0f, 63)) {
            ASSERT_EQ(pypto_scalar::BFloat16ToInteger<int64_t>(bits), static_cast<int64_t>(value)) << bits;
        }
        if (value >= 0.0f && value < std::ldexp(1.0f, 64)) {
            ASSERT_EQ(pypto_scalar::BFloat16ToInteger<uint64_t>(bits), static_cast<uint64_t>(value)) << bits;
        }
    }
}

namespace {

ir::ExprPtr LowerExpression(const ir::ExprPtr& expr, ir::FunctionType kind = ir::FunctionType::IN_CORE)
{
    auto span = ir::Span::Unknown();
    auto body = std::make_shared<const ir::AssignStmt>(std::vector<ir::VarPtr>{}, expr, span);
    auto function = std::make_shared<const ir::Function>("kernel", std::vector<ir::VarPtr>{},
                                                         std::vector<ir::TypePtr>{}, body, span, kind, true);
    auto metadata = std::make_shared<ir::IRDebugInfo>();
    auto program = std::make_shared<const ir::Program>(std::vector<ir::FunctionPtr>{function}, "test", span, metadata);
    auto lowered = LowerScalarExpressionsForCCE()(program);
    EXPECT_EQ(lowered->GetDebugInfo(), metadata.get());
    EXPECT_EQ(LowerScalarExpressionsForCCE()(lowered), lowered);
    return ir::As<ir::AssignStmt>(lowered->functions_.at("kernel")->body_->stmts_[0])->value_;
}

ir::VarPtr BFloat16Var(const std::string& name)
{
    return std::make_shared<const ir::Var>(name, std::make_shared<const ir::ScalarType>(ir::DataType::BF16),
                                           ir::Span::Unknown());
}

void ExpectPromotion(const ir::ExprPtr& expr, const ir::ExprPtr& operand)
{
    auto cast = ir::As<ir::Cast>(expr);
    ASSERT_NE(cast, nullptr);
    EXPECT_EQ(ir::GetScalarDtype(cast), ir::DataType::FP32);
    EXPECT_EQ(cast->operand_, operand);
}

} // namespace

TEST(ScalarBFloat16Test, ArithmeticAndComparisonsShareScalarConversion)
{
    auto left = BFloat16Var("left");
    auto right = BFloat16Var("right");
    for (const auto& expr :
         {ir::MakeAdd(left, right), ir::MakeSub(left, right), ir::MakeMul(left, right), ir::MakeFloatDiv(left, right),
          ir::MakeFloorDiv(left, right), ir::MakeFloorMod(left, right), ir::MakeEq(left, right),
          ir::MakeNe(left, right), ir::MakeLt(left, right), ir::MakeLe(left, right), ir::MakeGt(left, right),
          ir::MakeGe(left, right), ir::MakeAnd(left, right), ir::MakeOr(left, right), ir::MakeXor(left, right)}) {
        auto lowered = LowerExpression(expr);
        EXPECT_EQ(ir::GetScalarDtype(lowered), ir::GetScalarDtype(expr));
        if (ir::GetScalarDtype(expr) == ir::DataType::BF16) {
            auto cast = ir::As<ir::Cast>(lowered);
            ASSERT_NE(cast, nullptr);
            lowered = cast->operand_;
            EXPECT_EQ(ir::GetScalarDtype(lowered), ir::DataType::FP32);
        }
        auto binary = ir::As<ir::BinaryExpr>(lowered);
        ASSERT_NE(binary, nullptr);
        EXPECT_EQ(binary->GetKind(), expr->GetKind());
        ExpectPromotion(binary->left_, left);
        ExpectPromotion(binary->right_, right);
    }
}

TEST(ScalarBFloat16Test, LogicalXorConvertsBothOperandsToBool)
{
    CCECodegen codegen(ir::SectionKind::Vector);
    auto left = std::make_shared<const ir::ConstInt>(1, ir::DataType::INT32, ir::Span::Unknown());
    auto right = std::make_shared<const ir::ConstInt>(2, ir::DataType::INT32, ir::Span::Unknown());
    EXPECT_EQ(codegen.GetExprAsCode(ir::MakeXor(left, right)), "((bool)(1) != (bool)(2))");
}

TEST(ScalarBFloat16Test, NestedArithmeticRetainsEveryBFloat16ResultBoundary)
{
    auto left = BFloat16Var("left");
    auto right = BFloat16Var("right");
    auto sum = ir::MakeAdd(left, right);
    auto difference = ir::MakeSub(sum, left);
    auto outer = ir::As<ir::Cast>(LowerExpression(difference));
    ASSERT_NE(outer, nullptr);
    auto subtraction = ir::As<ir::Sub>(outer->operand_);
    ASSERT_NE(subtraction, nullptr);
    auto promotion = ir::As<ir::Cast>(subtraction->left_);
    ASSERT_NE(promotion, nullptr);
    auto rounding = ir::As<ir::Cast>(promotion->operand_);
    ASSERT_NE(rounding, nullptr);
    EXPECT_EQ(ir::GetScalarDtype(rounding), ir::DataType::BF16);
    auto addition = ir::As<ir::Add>(rounding->operand_);
    ASSERT_NE(addition, nullptr);
    EXPECT_EQ(ir::GetScalarDtype(addition), ir::DataType::FP32);
    ExpectPromotion(addition->left_, left);
    ExpectPromotion(addition->right_, right);
    ExpectPromotion(subtraction->right_, left);
    EXPECT_EQ(ir::As<ir::Sub>(difference)->left_, sum);
}

TEST(ScalarBFloat16Test, LogicalOperationsPromoteOnlyBFloat16Operands)
{
    auto left = BFloat16Var("left");
    auto right = std::make_shared<const ir::Var>("right", std::make_shared<const ir::ScalarType>(ir::DataType::FP32),
                                                 ir::Span::Unknown());
    for (const auto& expr : {ir::MakeAnd(left, right), ir::MakeOr(left, right), ir::MakeXor(left, right)}) {
        auto lowered = ir::As<ir::BinaryExpr>(LowerExpression(expr));
        ASSERT_NE(lowered, nullptr);
        ExpectPromotion(lowered->left_, left);
        EXPECT_EQ(lowered->right_, right);
    }
    auto logical_not = ir::As<ir::Not>(LowerExpression(ir::MakeNot(left)));
    ASSERT_NE(logical_not, nullptr);
    ExpectPromotion(logical_not->operand_, left);
}

TEST(ScalarBFloat16Test, PreservesSimtArithmeticInEveryFunctionKind)
{
    auto left = BFloat16Var("left");
    auto right = BFloat16Var("right");
    for (auto kind : {ir::FunctionType::SIMT_VF, ir::FunctionType::SIMT_CALLEE, ir::FunctionType::SIMT_KERNEL}) {
        for (const auto& expr : {ir::MakeAdd(left, right), ir::MakeMin(left, right), ir::MakeMax(left, right)}) {
            EXPECT_EQ(LowerExpression(expr, kind), expr);
        }
    }
}

} // namespace pypto::codegen
