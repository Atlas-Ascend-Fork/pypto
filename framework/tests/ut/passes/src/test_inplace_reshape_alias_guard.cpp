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
 * \file test_inplace_reshape_alias_guard.cpp
 * \brief Tests for GraphUtils::InplaceReshapeRemovalBreaksAlias and the preservation of alias-anchoring
 *        inplace reshapes by DeadOperationEliminator and RemoveRedundantReshape.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>
#include "symbolic_scalar_test_utils.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"

namespace npu {
namespace tile_fwk {

class TestInplaceReshapeAliasGuard : public ::testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "InplaceAliasGuardTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

struct AliasTopology {
    LogicalTensorPtr base;
    LogicalTensorPtr alias;
    LogicalTensorPtr sibling;
    LogicalTensorPtr src;
    Operation* reshape = nullptr;
    Operation* producer = nullptr;
};

static std::shared_ptr<Function> MakeFunction()
{
    return std::make_shared<Function>(Program::GetInstance(), "InplaceAliasGuardTest", "InplaceAliasGuardTest",
                                      nullptr);
}

// base(rawA) --reshape--> alias(rawB, zero consumers); src --view--> sibling(outcast).
// shareRaw makes sibling share alias rawB; siblingHasProducer gives sibling a downstream producer.
static AliasTopology BuildAliasTopology(Function& func, bool inplace, bool shareRaw, bool siblingHasProducer)
{
    AliasTopology t;
    const std::vector<int64_t> shape = {8, 16};
    IRBuilder builder;
    t.base = builder.CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t.alias = builder.CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t.src = builder.CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t.sibling = shareRaw ? builder.CreateTensorVar(t.alias->GetRawTensor(), {0, 0}, shape, {}) :
                           builder.CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t.reshape = &PassOperationUtils::AddOperation(func, Opcode::OP_RESHAPE, {t.base}, {t.alias}, nullptr,
                                                  ir::Span::Unknown(), false);
    if (inplace) {
        t.reshape->SetAttribute("op_attr_isInplace", true);
    }
    if (siblingHasProducer) {
        t.producer = &PassOperationUtils::AddOperation(func, Opcode::OP_VIEW, {t.src}, {t.sibling}, nullptr,
                                                       ir::Span::Unknown(), false);
    }
    func.inCasts_.push_back(t.base);
    func.inCasts_.push_back(t.src);
    func.outCasts_.push_back(t.sibling);
    return t;
}

TEST_F(TestInplaceReshapeAliasGuard, PredicateNonInplaceReturnsFalse)
{
    auto func = MakeFunction();
    auto t = BuildAliasTopology(*func, false, true, true);
    EXPECT_FALSE(GraphUtils::InplaceReshapeRemovalBreaksAlias(*func, *t.reshape, t.base, t.alias));
}

TEST_F(TestInplaceReshapeAliasGuard, PredicateSameRawMagicReturnsFalse)
{
    auto func = MakeFunction();
    const std::vector<int64_t> shape = {8, 16};
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto alias = builder.CreateTensorVar(base->GetRawTensor(), {0, 0}, shape, {});
    auto& reshape = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {base}, {alias}, nullptr,
                                                     ir::Span::Unknown(), false);
    reshape.SetAttribute("op_attr_isInplace", true);
    func->inCasts_.push_back(base);
    func->outCasts_.push_back(alias);
    EXPECT_FALSE(GraphUtils::InplaceReshapeRemovalBreaksAlias(*func, reshape, base, alias));
}

TEST_F(TestInplaceReshapeAliasGuard, PredicateNoSiblingReturnsFalse)
{
    auto func = MakeFunction();
    auto t = BuildAliasTopology(*func, true, false, true);
    EXPECT_FALSE(GraphUtils::InplaceReshapeRemovalBreaksAlias(*func, *t.reshape, t.base, t.alias));
}

TEST_F(TestInplaceReshapeAliasGuard, PredicateSiblingWithProducerReturnsTrue)
{
    auto func = MakeFunction();
    auto t = BuildAliasTopology(*func, true, true, true);
    EXPECT_TRUE(GraphUtils::InplaceReshapeRemovalBreaksAlias(*func, *t.reshape, t.base, t.alias));
}

TEST_F(TestInplaceReshapeAliasGuard, PredicateSiblingWithoutProducerConsumerReturnsFalse)
{
    auto func = MakeFunction();
    auto t = BuildAliasTopology(*func, true, true, false);
    EXPECT_FALSE(GraphUtils::InplaceReshapeRemovalBreaksAlias(*func, *t.reshape, t.base, t.alias));
}

TEST_F(TestInplaceReshapeAliasGuard, DeadOpPreservesInplaceAliasReshape)
{
    auto func = MakeFunction();
    auto t = BuildAliasTopology(*func, true, true, true);
    EXPECT_EQ(DeadOperationEliminator::EliminateDeadOperation(*func), SUCCESS);
    EXPECT_TRUE(func->Operations().Contains(*t.reshape));
}

TEST_F(TestInplaceReshapeAliasGuard, DeadOpEliminatesNonInplaceDeadReshape)
{
    auto func = MakeFunction();
    auto t = BuildAliasTopology(*func, false, true, true);
    EXPECT_EQ(DeadOperationEliminator::EliminateDeadOperation(*func), SUCCESS);
    EXPECT_FALSE(func->Operations().Contains(*t.reshape));
}

TEST_F(TestInplaceReshapeAliasGuard, RemoveRedundantReshapePreservesInplaceAliasReshape)
{
    auto func = MakeFunction();
    auto t = BuildAliasTopology(*func, true, true, true);
    RemoveRedundantReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    EXPECT_TRUE(func->Operations().Contains(*t.reshape));
}

} // namespace tile_fwk
} // namespace npu
