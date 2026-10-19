/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>

#include "gtest/gtest.h"

#include "ir/transforms/passes.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"

#include "program_builder.h"

using namespace npu::tile_fwk;

namespace {

std::vector<ir::TensorOpStmtPtr> GetAssembles(const ir::ProgramPtr& program, const std::string& functionName)
{
    auto function = program->GetFunction(functionName);
    EXPECT_NE(function, nullptr);
    if (!function || !function->body_) {
        return {};
    }

    std::vector<ir::TensorOpStmtPtr> assembles;
    for (const auto& statement : function->body_->stmts_) {
        auto op = std::dynamic_pointer_cast<const Operation>(statement);
        if (op && op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembles.push_back(op);
        }
    }
    return assembles;
}

std::vector<ir::TensorOpStmtPtr> GetOperations(const ir::ProgramPtr& program, const std::string& functionName,
                                               Opcode opcode)
{
    auto function = program->GetFunction(functionName);
    EXPECT_NE(function, nullptr);
    if (!function || !function->body_) {
        return {};
    }

    std::vector<ir::TensorOpStmtPtr> operations;
    for (const auto& statement : function->body_->stmts_) {
        auto operation = std::dynamic_pointer_cast<const Operation>(statement);
        if (operation && operation->GetOpcode() == opcode) {
            operations.push_back(operation);
        }
    }
    return operations;
}

ir::TensorOpStmtPtr GetOperation(const ir::ProgramPtr& program, const std::string& functionName, Opcode opcode)
{
    auto function = program->GetFunction(functionName);
    EXPECT_NE(function, nullptr);
    if (!function || !function->body_) {
        return nullptr;
    }

    for (const auto& statement : function->body_->stmts_) {
        auto operation = std::dynamic_pointer_cast<const Operation>(statement);
        if (operation && operation->GetOpcode() == opcode) {
            return operation;
        }
    }
    return nullptr;
}

bool HasDirectWriteTokenDependency(const ir::TensorOpStmtPtr& producer, const ir::TensorOpStmtPtr& consumer)
{
    for (const auto& resultToken : producer->result_token_) {
        auto tokenType = std::dynamic_pointer_cast<const ir::TokenType>(resultToken->GetType());
        if (!tokenType || tokenType->kind_ != ir::TokenKind::WRITE) {
            continue;
        }
        if (std::find(consumer->tokens_.begin(), consumer->tokens_.end(), resultToken) != consumer->tokens_.end()) {
            return true;
        }
    }
    return false;
}

} // namespace

class TestRemoveRedundantTokenPass : public testing::Test {
protected:
    void SetUp() override
    {
        savedFlag_ = IRContext::Get().AssembleNewLogicalTensor();
        IRContext::Get().SetAssembleNewLogicalTensor(true);
    }

    void TearDown() override { IRContext::Get().SetAssembleNewLogicalTensor(savedFlag_); }

private:
    bool savedFlag_{false};
};

TEST_F(TestRemoveRedundantTokenPass, TensorOps)
{
    auto a = Tensor(DT_FP32, {16, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("TensorOps", {a, out});

    auto lhs = Add(a, Element(DT_FP32, 1));
    auto rhs = Sub(a, Element(DT_FP32, 1));
    Assemble(Add(lhs, rhs), {0, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(result, nullptr);
}

TEST_F(TestRemoveRedundantTokenPass, OverlappingWritesMergeIntoLiveResult)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("OverlappingWritesMerge", {a, out});

    Assemble(Add(a, Element(DT_FP32, 1)), {0, 0}, out);
    Assemble(Sub(a, Element(DT_FP32, 1)), {4, 0}, out);

    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(p.EndFunction());

    ASSERT_NE(result, nullptr);
    auto assembles = GetAssembles(result, "OverlappingWritesMerge");
    ASSERT_EQ(assembles.size(), 2u);
    EXPECT_EQ(assembles[0]->result_[0], assembles[1]->result_[0]);
}

TEST_F(TestRemoveRedundantTokenPass, OverlappingWritesWithWriteTokenMergeResults)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("OverlappingWritesWithWriteToken", {a, out});

    Assemble(Add(a, Element(DT_FP32, 1)), {0, 0}, out);
    Assemble(Sub(a, Element(DT_FP32, 1)), {4, 0}, out);

    auto inferred = pypto::ir::pass::InferTokenPass()(p.EndFunction());
    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(result, nullptr);
    auto assembles = GetAssembles(result, "OverlappingWritesWithWriteToken");
    ASSERT_EQ(assembles.size(), 2u);
    EXPECT_EQ(assembles[0]->result_[0], assembles[1]->result_[0]);
    EXPECT_TRUE(HasDirectWriteTokenDependency(assembles[0], assembles[1]));
}

TEST_F(TestRemoveRedundantTokenPass, FullOutputMergesIntoLiveAssembleResult)
{
    auto source = Tensor(DT_FP32, {16, 16}, "source");

    ProgramBuilder p;
    p.BeginFunction("FullOutputMerges", {source});

    auto output = Full(Element(DT_FP32, 0.0f), DT_FP32, {16, 16});
    Assemble(source, {0, 0}, output);
    (void)Add(output, source);

    auto inferred = pypto::ir::pass::InferTokenPass()(p.EndFunction());
    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(result, nullptr);
    auto full = GetOperation(result, "FullOutputMerges", Opcode::OP_VEC_DUP);
    auto assembles = GetAssembles(result, "FullOutputMerges");
    ASSERT_NE(full, nullptr);
    ASSERT_EQ(assembles.size(), 1u);
    EXPECT_EQ(full->result_[0], assembles[0]->result_[0]);
    EXPECT_TRUE(HasDirectWriteTokenDependency(full, assembles[0]));
}

TEST_F(TestRemoveRedundantTokenPass, DisjointAtomicResultsMergeIntoLiveAtomicResult)
{
    auto source = Tensor(DT_FP32, {8, 16}, "source");
    auto output = Tensor(DT_FP32, {24, 16}, "output");

    ProgramBuilder p;
    p.BeginFunction("DisjointAtomicResults", {source, output});

    AtomicRMW(source, {0, 0}, output, AtomicRMWMode::ADD);
    AtomicRMW(source, {8, 0}, output, AtomicRMWMode::ADD);
    (void)Add(output, output);

    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(p.EndFunction());

    ASSERT_NE(result, nullptr);
    auto atomics = GetOperations(result, "DisjointAtomicResults", Opcode::OP_ATOMIC_RMW);
    ASSERT_EQ(atomics.size(), 2u);
    EXPECT_EQ(atomics[0]->result_[0], atomics[1]->result_[0]);
}

TEST_F(TestRemoveRedundantTokenPass, FirstDanglingVersionMergesIntoNearestLiveVersion)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {24, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("NearestLiveVersion", {a, out});

    Assemble(Add(a, Element(DT_FP32, 1)), {0, 0}, out);
    Assemble(Sub(a, Element(DT_FP32, 1)), {8, 0}, out);
    (void)Add(out, out);
    Assemble(Mul(a, Element(DT_FP32, 2)), {16, 0}, out);
    (void)Add(out, out);

    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(p.EndFunction());

    ASSERT_NE(result, nullptr);
    auto assembles = GetAssembles(result, "NearestLiveVersion");
    ASSERT_EQ(assembles.size(), 3u);
    EXPECT_EQ(assembles[0]->result_[0], assembles[1]->result_[0]);
    EXPECT_NE(assembles[1]->result_[0], assembles[2]->result_[0]);
}

TEST_F(TestRemoveRedundantTokenPass, TokenForkDoesNotChangeNearestLiveVersion)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {24, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("TokenForkNearestLiveVersion", {a, out});

    Assemble(Add(a, Element(DT_FP32, 1)), {0, 0}, out);
    Assemble(Sub(a, Element(DT_FP32, 1)), {8, 0}, out);
    (void)Add(out, out);
    Assemble(Mul(a, Element(DT_FP32, 2)), {16, 0}, out);
    (void)Add(out, out);

    auto inferred = pypto::ir::pass::InferTokenPass()(p.EndFunction());
    auto before = GetAssembles(inferred, "TokenForkNearestLiveVersion");
    ASSERT_EQ(before.size(), 3u);
    ASSERT_FALSE(before[0]->result_token_.empty());
    auto forkTarget = std::const_pointer_cast<ir::TensorOpStmt>(before[2]);
    forkTarget->tokens_.push_back(before[0]->result_token_.front());

    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(result, nullptr);
    auto assembles = GetAssembles(result, "TokenForkNearestLiveVersion");
    ASSERT_EQ(assembles.size(), 3u);
    EXPECT_EQ(assembles[0]->result_[0], assembles[1]->result_[0]);
    EXPECT_NE(assembles[1]->result_[0], assembles[2]->result_[0]);
}

TEST_F(TestRemoveRedundantTokenPass, IdempotentOnSecondRun)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("IdempotentMerge", {a, out});

    Assemble(Add(a, Element(DT_FP32, 1)), {0, 0}, out);
    Assemble(Sub(a, Element(DT_FP32, 1)), {8, 0}, out);
    (void)Add(out, out);

    auto first = pypto::ir::pass::RemoveRedundantTokenPass()(p.EndFunction());
    auto second = pypto::ir::pass::RemoveRedundantTokenPass()(first);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    auto firstAssembles = GetAssembles(first, "IdempotentMerge");
    auto secondAssembles = GetAssembles(second, "IdempotentMerge");
    ASSERT_EQ(firstAssembles.size(), 2u);
    ASSERT_EQ(secondAssembles.size(), 2u);
    EXPECT_EQ(firstAssembles[0]->result_[0], firstAssembles[1]->result_[0]);
    EXPECT_EQ(secondAssembles[0]->result_[0], secondAssembles[1]->result_[0]);
    EXPECT_EQ(secondAssembles[0]->result_[0], firstAssembles[0]->result_[0]);
}

TEST_F(TestRemoveRedundantTokenPass, MultipleDisjointWritesMergeIntoLiveResult)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {24, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("MultipleDisjointWrites", {a, out});

    Assemble(Add(a, Element(DT_FP32, 1)), {0, 0}, out);
    Assemble(Sub(a, Element(DT_FP32, 1)), {8, 0}, out);
    Assemble(Mul(a, Element(DT_FP32, 2)), {16, 0}, out);

    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(p.EndFunction());

    ASSERT_NE(result, nullptr);
    auto assembles = GetAssembles(result, "MultipleDisjointWrites");
    ASSERT_EQ(assembles.size(), 3u);
    EXPECT_EQ(assembles[0]->result_[0], assembles[2]->result_[0]);
    EXPECT_EQ(assembles[1]->result_[0], assembles[2]->result_[0]);
}

TEST_F(TestRemoveRedundantTokenPass, DisjointWrites)
{
    auto a = Tensor(DT_FP32, {8, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("DisjointWrites", {a, out});

    auto v1 = Add(a, Element(DT_FP32, 1));
    Assemble(v1, {0, 0}, out);

    auto v2 = Sub(a, Element(DT_FP32, 1));
    Assemble(v2, {8, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto result = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(result, nullptr);
    auto assembles = GetAssembles(result, "DisjointWrites");
    ASSERT_EQ(assembles.size(), 2u);
    EXPECT_EQ(assembles[0]->result_[0], assembles[1]->result_[0]);
}

TEST_F(TestRemoveRedundantTokenPass, IfStmt)
{
    auto a = Tensor(DT_FP32, {16, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("IfStmt", {a, out});

    auto result = p.If(
        SymbolicScalar("condition") > 0, [&] { p.Yield(Add(a, Element(DT_FP32, 1))); },
        [&] { p.Yield(Sub(a, Element(DT_FP32, 1))); });

    Assemble(p.AsTensor(result[0]), {0, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto removed = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(removed, nullptr);
}

TEST_F(TestRemoveRedundantTokenPass, ForStmt)
{
    auto a = Tensor(DT_FP32, {16, 16}, "a");
    auto out = Tensor(DT_FP32, {16, 16}, "out");

    ProgramBuilder p;
    p.BeginFunction("ForStmt", {a, out});

    auto init = Add(a, Element(DT_FP32, 1));
    auto result = p.For(0, 2, 1, {{"carry", init}}, [&](SymbolicScalar, const std::vector<ir::VarPtr>& carries) {
        p.Continue(Add(p.AsTensor(carries[0]), Element(DT_FP32, 1)));
    });

    Assemble(p.AsTensor(result[0]), {0, 0}, out);

    auto prog = p.EndFunction();

    auto inferred = pypto::ir::pass::InferTokenPass()(prog);
    auto removed = pypto::ir::pass::RemoveRedundantTokenPass()(inferred);

    ASSERT_NE(removed, nullptr);
}
