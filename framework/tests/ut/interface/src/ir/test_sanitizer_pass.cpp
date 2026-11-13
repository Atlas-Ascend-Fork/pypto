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

#include <memory>
#include <optional>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

TEST(SanitizerPassTest, PreservesNonSimtExpressionsWhileInsertingRecords)
{
    const Span span("sanitizer_non_simt.py", 12, 1);
    auto scalar_type = std::make_shared<ScalarType>(DataType::INT64);
    auto index = std::make_shared<Var>("index", scalar_type, span);
    auto extent = std::make_shared<ConstInt>(32, DataType::INT64, span);
    auto tensor_type = std::make_shared<TensorType>(std::vector<ExprPtr>{extent}, DataType::INT32);
    auto source = std::make_shared<Var>("source", tensor_type, span);
    auto& registry = OpRegistry::GetInstance();
    auto block = registry.Create("get_block_idx", {}, span);
    auto offset = MakeAdd(index, block, span);
    auto read = registry.Create("block.getval", {source, offset}, {{"preserve_dtype", false}}, span);
    auto value = std::make_shared<Var>("value", read->GetType(), span);
    auto access = std::make_shared<AssignStmt>(std::vector<VarPtr>{value}, read, span);
    auto condition = MakeLt(offset, extent, span);
    auto branch = std::make_shared<IfStmt>(condition, access, std::nullopt, std::vector<VarPtr>{}, span);
    auto result = std::make_shared<Var>("result", scalar_type, span);
    auto arithmetic = std::make_shared<AssignStmt>(std::vector<VarPtr>{result}, MakeMul(offset, extent, span), span);
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{branch, arithmetic}, span);
    auto section = std::make_shared<SectionStmt>(SectionKind::Vector, body, span);
    auto function = std::make_shared<Function>("non_simt", std::vector<VarPtr>{source, index}, std::vector<TypePtr>{},
                                               section, span, FunctionType::OPAQUE, true);
    auto program = std::make_shared<Program>(std::vector<FunctionPtr>{function}, "non_simt", span);

    auto instrumented = pass::Sanitizer()(program)->GetFunction("non_simt");
    ASSERT_NE(instrumented, nullptr);
    ASSERT_EQ(instrumented->params_.size(), function->params_.size() + 2);
    EXPECT_EQ(instrumented->params_[0].get(), source.get());
    EXPECT_EQ(instrumented->params_[1].get(), index.get());
    auto entry_body = instrumented->body_;
    ASSERT_NE(entry_body, nullptr);
    ASSERT_EQ(entry_body->stmts_.size(), 2);
    auto region_assign = As<AssignStmt>(entry_body->stmts_[0]);
    ASSERT_NE(region_assign, nullptr);
    auto region_call = As<Call>(region_assign->value_);
    ASSERT_NE(region_call, nullptr);
    EXPECT_EQ(region_call->name_, "block.sanitizer_log_region");
    auto vector_section = As<SectionStmt>(entry_body->stmts_[1]);
    ASSERT_NE(vector_section, nullptr);
    auto vector_body = vector_section->body_;
    ASSERT_NE(vector_body, nullptr);
    ASSERT_EQ(vector_body->stmts_.size(), 2);
    EXPECT_EQ(vector_body->stmts_[1].get(), arithmetic.get());
    auto rewritten_branch = As<IfStmt>(vector_body->stmts_[0]);
    ASSERT_NE(rewritten_branch, nullptr);
    EXPECT_EQ(rewritten_branch->condition_.get(), condition.get());
    auto then_body = rewritten_branch->thenBody_;
    ASSERT_NE(then_body, nullptr);
    ASSERT_EQ(then_body->stmts_.size(), 2);
    // Pointer identity proves the original call, nested offset, type, kwargs and span survive.
    EXPECT_EQ(then_body->stmts_[1].get(), access.get());
    auto log_stmt = As<AssignStmt>(then_body->stmts_[0]);
    ASSERT_NE(log_stmt, nullptr);
    auto log_call = As<Call>(log_stmt->value_);
    ASSERT_NE(log_call, nullptr);
    EXPECT_EQ(log_call->name_, "block.sanitizer_log");
}

} // namespace ir
} // namespace pypto
