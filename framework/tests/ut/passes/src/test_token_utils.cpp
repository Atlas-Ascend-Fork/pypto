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
 * \file test_token_utils.cpp
 * \brief Unit test for TokenUtils::SplitMultiProducerTokens.
 */

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "ir/span.h"
#include "ir/type.h"
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "interface/function/function.h"
#undef private
#undef protected
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_utils/token_utils.h"

using namespace npu::tile_fwk;

static ir::VarPtr MakeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetTokenType(), ir::Span::Unknown());
}

class TokenUtilsTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

TEST_F(TokenUtilsTest, SplitMultiProducerTokensBasic)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenSplitBasic")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenSplitBasic");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 3);

    auto tokenT = MakeToken("tokenT");
    auto& dep = func->GetVarDependency();

    opList[0]->result_token_ = {tokenT};
    opList[1]->result_token_ = {tokenT};

    dep.AddProducer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[0]));
    dep.AddProducer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[1]));

    opList[2]->tokens_.push_back(tokenT);
    dep.AddConsumer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[2]));

    EXPECT_EQ(TokenUtils::SplitMultiProducerTokens(*func), SUCCESS);

    EXPECT_EQ(opList[0]->result_token_.front(), tokenT);

    EXPECT_FALSE(opList[1]->result_token_.empty());
    EXPECT_NE(opList[1]->result_token_.front(), tokenT);

    bool hasOriginal = false;
    bool hasNew = false;
    for (const auto& t : opList[2]->tokens_) {
        if (t == tokenT) {
            hasOriginal = true;
        }
        if (t == opList[1]->result_token_.front()) {
            hasNew = true;
        }
    }
    EXPECT_TRUE(hasOriginal);
    EXPECT_TRUE(hasNew);

    const auto& producersOld = dep.GetProducers(tokenT);
    EXPECT_EQ(producersOld.size(), 1);

    const auto& producersNew = dep.GetProducers(opList[1]->result_token_.front());
    EXPECT_EQ(producersNew.size(), 1);

    const auto& consumersNew = dep.GetConsumers(opList[1]->result_token_.front());
    EXPECT_EQ(consumersNew.size(), 1);
}

TEST_F(TokenUtilsTest, SplitMultiProducerTokensThreeProducers)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenSplitThree")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        Tensor c = Add(b, Element(DT_FP32, 1.0));
        output = Sub(c, Element(DT_FP32, 0.5));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenSplitThree");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 4);

    auto tokenT = MakeToken("tokenT3");
    auto& dep = func->GetVarDependency();

    opList[0]->result_token_ = {tokenT};
    opList[1]->result_token_ = {tokenT};
    opList[2]->result_token_ = {tokenT};

    dep.AddProducer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[0]));
    dep.AddProducer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[1]));
    dep.AddProducer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[2]));

    opList[3]->tokens_.push_back(tokenT);
    dep.AddConsumer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[3]));

    EXPECT_EQ(TokenUtils::SplitMultiProducerTokens(*func), SUCCESS);

    EXPECT_EQ(opList[0]->result_token_.front(), tokenT);
    EXPECT_NE(opList[1]->result_token_.front(), tokenT);
    EXPECT_FALSE(opList[1]->result_token_.empty());
    EXPECT_NE(opList[2]->result_token_.front(), tokenT);
    EXPECT_FALSE(opList[2]->result_token_.empty());
    EXPECT_NE(opList[1]->result_token_.front(), opList[2]->result_token_.front());

    EXPECT_EQ(opList[3]->tokens_.size(), 3);

    const auto& producersOld = dep.GetProducers(tokenT);
    EXPECT_EQ(producersOld.size(), 1);

    const auto& producersNew1 = dep.GetProducers(opList[1]->result_token_.front());
    EXPECT_EQ(producersNew1.size(), 1);

    const auto& producersNew2 = dep.GetProducers(opList[2]->result_token_.front());
    EXPECT_EQ(producersNew2.size(), 1);

    const auto& consumersNew1 = dep.GetConsumers(opList[1]->result_token_.front());
    EXPECT_EQ(consumersNew1.size(), 1);

    const auto& consumersNew2 = dep.GetConsumers(opList[2]->result_token_.front());
    EXPECT_EQ(consumersNew2.size(), 1);
}

TEST_F(TokenUtilsTest, SplitMultiProducerTokensSingleProducerNoop)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenSplitNoop")
    {
        Tensor a = Exp(input);
        output = Add(a, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenSplitNoop");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 2);

    auto tokenT = MakeToken("tokenT_single");

    auto& dep = func->GetVarDependency();
    opList[0]->result_token_ = {tokenT};
    dep.AddProducer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[0]));

    opList[1]->tokens_.push_back(tokenT);
    dep.AddConsumer(tokenT, std::static_pointer_cast<const ir::Stmt>(func->operations_[1]));

    EXPECT_EQ(TokenUtils::SplitMultiProducerTokens(*func), SUCCESS);

    EXPECT_EQ(opList[0]->result_token_.front(), tokenT);
    EXPECT_EQ(opList[1]->tokens_.size(), 1);
    EXPECT_EQ(opList[1]->tokens_[0], tokenT);
}

TEST_F(TokenUtilsTest, SplitMultiProducerTokensNoTokensNoop)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenSplitEmpty") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenSplitEmpty");
    ASSERT_NE(func, nullptr);

    EXPECT_EQ(TokenUtils::SplitMultiProducerTokens(*func), SUCCESS);
}

TEST_F(TokenUtilsTest, RebuildTokenDependenciesPrunesStaleAndMatchesFields)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenRebuildStale")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenRebuildStale");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 3);

    // A detached operation never added to function operations, used to seed stale dependency
    // statements that RebuildTokenDependencies must discard.
    auto staleOp = std::make_shared<Operation>(*func, Opcode::OP_ADD, LogicalTensors{}, LogicalTensors{});
    auto staleStmt = std::static_pointer_cast<const ir::Stmt>(staleOp);

    auto tokenA = MakeToken("tokenA_rebuild");
    auto tokenB = MakeToken("tokenB_rebuild");
    auto staleToken = MakeToken("staleToken_rebuild");

    opList[0]->result_token_ = {tokenA};
    opList[1]->result_token_ = {tokenB};
    opList[2]->tokens_.push_back(tokenA);
    opList[2]->tokens_.push_back(tokenB);

    auto op0Stmt = std::static_pointer_cast<const ir::Stmt>(func->operations_[0]);
    auto op1Stmt = std::static_pointer_cast<const ir::Stmt>(func->operations_[1]);
    auto op2Stmt = std::static_pointer_cast<const ir::Stmt>(func->operations_[2]);

    auto& dep = func->GetVarDependency();
    // Current producer/consumer entries plus deliberately stale statements referencing the
    // detached operation as both a producer and a consumer.
    dep.AddProducer(tokenA, op0Stmt);
    dep.AddProducer(tokenB, op1Stmt);
    dep.AddProducer(staleToken, staleStmt);
    dep.AddProducer(tokenA, staleStmt);
    dep.AddConsumer(tokenA, op2Stmt);
    dep.AddConsumer(tokenB, op2Stmt);
    dep.AddConsumer(tokenA, staleStmt);

    EXPECT_EQ(TokenUtils::RebuildTokenDependencies(*func), SUCCESS);

    // The stale-only token lost its only (stale) producer and has no consumer, so the entry is gone.
    EXPECT_FALSE(dep.HasDependency(staleToken));

    // Current producer entries exactly match the operation fields.
    const auto& producersA = dep.GetProducers(tokenA);
    EXPECT_EQ(producersA.size(), 1);
    EXPECT_EQ(producersA.count(op0Stmt), 1);
    EXPECT_EQ(producersA.count(staleStmt), 0);

    const auto& producersB = dep.GetProducers(tokenB);
    EXPECT_EQ(producersB.size(), 1);
    EXPECT_EQ(producersB.count(op1Stmt), 1);

    // Current consumer entries exactly match the operation fields.
    const auto& consumersA = dep.GetConsumers(tokenA);
    EXPECT_EQ(consumersA.size(), 1);
    EXPECT_EQ(consumersA.count(op2Stmt), 1);
    EXPECT_EQ(consumersA.count(staleStmt), 0);

    const auto& consumersB = dep.GetConsumers(tokenB);
    EXPECT_EQ(consumersB.size(), 1);
    EXPECT_EQ(consumersB.count(op2Stmt), 1);

    // No surviving dependency entry references the detached/stale operation.
    for (const auto& entry : dep.GetAllDependencies()) {
        EXPECT_EQ(entry.second.producers.count(staleStmt), 0);
        EXPECT_EQ(entry.second.consumers.count(staleStmt), 0);
    }
}

TEST_F(TokenUtilsTest, RebuildTokenDependenciesAllowsExternalProducer)
{
    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DT_FP32, {32, 32}, "input");
    Tensor output(DT_FP32, {32, 32}, "output");
    FUNCTION("TokenRebuildMissingProducer") { output = Exp(input); }
    auto* function = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenRebuildMissingProducer");
    ASSERT_NE(function, nullptr);
    auto operations = function->Operations(false).DuplicatedOpList();
    ASSERT_FALSE(operations.empty());

    auto externalToken = MakeToken("external_token");
    operations.front()->tokens_.push_back(externalToken);
    EXPECT_EQ(TokenUtils::RebuildTokenDependencies(*function), SUCCESS);

    auto consumer = std::static_pointer_cast<const ir::Stmt>(function->operations_.front());
    EXPECT_TRUE(function->GetVarDependency().GetProducers(externalToken).empty());
    EXPECT_EQ(function->GetVarDependency().GetConsumers(externalToken).count(consumer), 1);
}

TEST_F(TokenUtilsTest, RemoveDanglingTokensDropsIncomplete)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("TokenDanglingRemove")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDanglingRemove");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 3);

    auto& dep = func->GetVarDependency();
    auto op0Stmt = std::static_pointer_cast<const ir::Stmt>(func->operations_[0]);
    auto op1Stmt = std::static_pointer_cast<const ir::Stmt>(func->operations_[1]);
    auto op2Stmt = std::static_pointer_cast<const ir::Stmt>(func->operations_[2]);

    // Read-only token: consumed but never produced -> dangling.
    auto readOnlyToken = MakeToken("token_read_only");
    opList[2]->tokens_.push_back(readOnlyToken);
    dep.AddConsumer(readOnlyToken, op2Stmt);

    // Write-only token: produced but never consumed -> dangling.
    auto writeOnlyToken = MakeToken("token_write_only");
    opList[0]->result_token_.push_back(writeOnlyToken);
    dep.AddProducer(writeOnlyToken, op0Stmt);

    // Complete token: produced and consumed -> must be preserved.
    auto completeToken = MakeToken("token_complete");
    opList[1]->result_token_.push_back(completeToken);
    opList[2]->tokens_.push_back(completeToken);
    dep.AddProducer(completeToken, op1Stmt);
    dep.AddConsumer(completeToken, op2Stmt);

    EXPECT_EQ(TokenUtils::RemoveDanglingTokens(*func), SUCCESS);

    EXPECT_EQ(dep.HasDependency(readOnlyToken), false);
    EXPECT_EQ(std::count(opList[2]->tokens_.begin(), opList[2]->tokens_.end(), readOnlyToken), 0);

    EXPECT_EQ(dep.HasDependency(writeOnlyToken), false);
    EXPECT_EQ(std::count(opList[0]->result_token_.begin(), opList[0]->result_token_.end(), writeOnlyToken), 0);

    EXPECT_EQ(dep.HasDependency(completeToken), true);
    EXPECT_EQ(dep.GetProducers(completeToken).size(), 1);
    EXPECT_EQ(dep.GetConsumers(completeToken).size(), 1);
    EXPECT_EQ(std::count(opList[1]->result_token_.begin(), opList[1]->result_token_.end(), completeToken), 1);
    EXPECT_EQ(std::count(opList[2]->tokens_.begin(), opList[2]->tokens_.end(), completeToken), 1);
}

TEST_F(TokenUtilsTest, RemoveDanglingTokensNoTokensNoop)
{
    TileShape::Current().SetVecTile({32, 32});
    Tensor input(DT_FP32, {32, 32}, "input");
    Tensor output(DT_FP32, {32, 32}, "output");
    FUNCTION("TokenDanglingEmpty") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TokenDanglingEmpty");
    ASSERT_NE(func, nullptr);

    EXPECT_EQ(TokenUtils::RemoveDanglingTokens(*func), SUCCESS);
    EXPECT_TRUE(func->GetVarDependency().GetAllDependencies().empty());
}
