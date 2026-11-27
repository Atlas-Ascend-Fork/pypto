/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_supernode_graph_builder.cpp
 * \brief Unit test for union-find and NodeGraphInfo guards in supernode_graph_builder.
 */

#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/error_manager.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_partition/supernode_graph_builder.h"

namespace npu {
namespace tile_fwk {

// Inline free function defined in supernode_graph_builder.cpp; declared here so white-box tests can link to it.
int32_t FindParent(std::vector<int32_t>& parent, int32_t i);

namespace {
// Exposes the protected SuperNodeGraphBuilder entry points without relying on access-control relaxation.
class TestableSuperNodeBuilder : public SuperNodeGraphBuilder {
public:
    using SuperNodeGraphBuilder::BuildHashValues;
    using SuperNodeGraphBuilder::BuildReduceNodeHash;
    using SuperNodeGraphBuilder::BuildSuperNodeGraph;
    void SetOperationInfo(std::shared_ptr<OperationGraphInfo> operationInfo) { operationInfo_ = operationInfo; }
    void SetSuperNodeInfo(std::shared_ptr<NodeGraphInfo> superNodeInfo) { superNodeInfo_ = superNodeInfo; }
    void DisableReduceBalanceHash() { useReduceBalanceHash_ = false; }
};

Operation* AddAbsOp(ComputationalGraphBuilder& G, const std::string& name)
{
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, {"in_" + name, "out_" + name}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ABS, {"in_" + name}, {"out_" + name}, name));
    return G.GetOp(name);
}

// Error messages are queued; pop the head until one of them contains expectedText.
bool ExpectErrorMessageContaining(const std::string& expectedText, size_t maxPopped = 4)
{
    for (size_t i = 0; i < maxPopped; i++) {
        std::string errMsg;
        if (!ErrorManager::Instance().GetFirstErrorMessage(errMsg)) {
            return false;
        }
        if (errMsg.find(expectedText) != std::string::npos) {
            return true;
        }
        ErrorManager::Instance().OutputErrorMessage(false);
    }
    return false;
}
} // namespace

class SuperNodeGraphBuilderTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }
    void TearDown() override {}
};

// FindParent rejects indices outside [0, size) with the illegal-parameter log.
TEST_F(SuperNodeGraphBuilderTest, FindParentRejectsIllegalIndex)
{
    std::vector<int32_t> parent{0, 1, 2};
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(FindParent(parent, -1), -1);
    EXPECT_EQ(FindParent(parent, 3), -1);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Call FindParent with illegal parameter -1"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// FindParent rejects parent chains that walk outside the union-find array.
TEST_F(SuperNodeGraphBuilderTest, FindParentRejectsIllegalParentChain)
{
    std::vector<int32_t> parent{1, 5}; // node 1 points to out-of-range index 5
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(FindParent(parent, 0), -1);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Find illegal parameter 5 in FindParent"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// AvoidLoop fails when the disjoint-set parent array itself is corrupted: FindParent(0) walks 0 -> 5.
TEST_F(SuperNodeGraphBuilderTest, AvoidLoopRejectsCorruptedParentArray)
{
    ComputationalGraphBuilder G;
    Operation* op = AddAbsOp(G, "abs");
    auto operationInfo = std::make_shared<OperationGraphInfo>();
    operationInfo->opList_ = {op};
    operationInfo->opCoreType_ = {OpCoreType::AIV};
    std::vector<int32_t> parent{5};
    NodeGraphInfo nodeInfo;
    std::vector<std::vector<int32_t>> node2Op;
    bool updated = false;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(nodeInfo.AvoidLoop(operationInfo, parent, node2Op, updated), FAILED);
    // FindParent reports "Find illegal parameter 5" first; AvoidLoop's stage message is queued after it.
    EXPECT_TRUE(ExpectErrorMessageContaining("Find parent in the union set failed in parent-to-node mapping stage"));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// Build fails in the merge pair stage when a merge pair references an out-of-range op index.
TEST_F(SuperNodeGraphBuilderTest, BuildFailsOnOutOfRangeMergePair)
{
    ComputationalGraphBuilder G;
    Operation* op = AddAbsOp(G, "abs");
    auto operationInfo = std::make_shared<OperationGraphInfo>();
    operationInfo->opList_ = {op};
    operationInfo->opCoreType_ = {OpCoreType::AIV};
    NodeGraphInfo nodeInfo;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(nodeInfo.Build(operationInfo, {{0, 7}}, false), FAILED);
    EXPECT_TRUE(ExpectErrorMessageContaining("Build the disjoint set failed in merge pair stage"));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// AvoidLoop merges nodes found by NodeInnerExpand; an expanded AIC node cannot join an AIV island.
TEST_F(SuperNodeGraphBuilderTest, AvoidLoopFailsWhenExpandedNodeHasIncompatibleCoreType)
{
    ComputationalGraphBuilder G;
    Operation* op0 = AddAbsOp(G, "abs0");
    Operation* op1 = AddAbsOp(G, "abs1");
    Operation* op2 = AddAbsOp(G, "abs2");
    auto operationInfo = std::make_shared<OperationGraphInfo>();
    operationInfo->opList_ = {op0, op1, op2};
    operationInfo->opCoreType_ = {OpCoreType::AIV, OpCoreType::AIC, OpCoreType::AIV};
    operationInfo->outGraph_ = {{1}, {}, {}};
    operationInfo->inGraph_ = {{}, {0}, {1}};
    operationInfo->opHashList_ = {1, 2, 3};
    // {op0, op2} belong to one island (root 2); NodeInnerExpand({0, 2}) additionally reaches op1.
    std::vector<int32_t> parent{2, 1, 2};
    NodeGraphInfo nodeInfo;
    std::vector<std::vector<int32_t>> node2Op;
    bool updated = false;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(nodeInfo.AvoidLoop(operationInfo, parent, node2Op, updated), FAILED);
    EXPECT_TRUE(ExpectErrorMessageContaining("Try to merge operations with different OpCoreType"));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// BuildSuperNodeGraph rejects an OperationGraphInfo whose inGraph/outGraph sizes do not match the op list.
TEST_F(SuperNodeGraphBuilderTest, BuildSuperNodeGraphRejectsUninitializedGraphInfo)
{
    TestableSuperNodeBuilder builder;
    auto operationInfo = std::make_shared<OperationGraphInfo>();
    operationInfo->opList_ = {};
    operationInfo->inGraph_ = {{}}; // deliberately inconsistent with the (empty) op list
    operationInfo->outGraph_ = {};
    builder.SetOperationInfo(operationInfo);
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(builder.BuildSuperNodeGraph(), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Operation inGraph and outGraph have not been initialized"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// BuildReduceNodeHash rejects an op-count mismatch between OperationInfo and the reduce node info.
TEST_F(SuperNodeGraphBuilderTest, BuildReduceNodeHashRejectsOpCountMismatch)
{
    TestableSuperNodeBuilder builder;
    auto operationInfo = std::make_shared<OperationGraphInfo>();
    operationInfo->opHashList_ = {1, 2}; // two ops
    builder.SetOperationInfo(operationInfo);
    auto reduceNodeInfo = std::make_shared<NodeGraphInfo>();
    reduceNodeInfo->op2Node_ = {}; // zero ops: mismatch
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(builder.BuildReduceNodeHash(reduceNodeInfo), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Operation number mismatch in OperationInfo and ReduceNodeInfo"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// BuildHashValues rejects an op-count mismatch between SuperNodeInfo and OperationInfo.
TEST_F(SuperNodeGraphBuilderTest, BuildHashValuesRejectsOpCountMismatch)
{
    TestableSuperNodeBuilder builder;
    auto operationInfo = std::make_shared<OperationGraphInfo>();
    operationInfo->opList_ = {}; // zero ops
    builder.SetOperationInfo(operationInfo);
    auto superNodeInfo = std::make_shared<NodeGraphInfo>();
    superNodeInfo->op2Node_ = {0}; // one op: mismatch
    builder.SetSuperNodeInfo(superNodeInfo);
    builder.DisableReduceBalanceHash(); // exercise the plain hashing branch
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(builder.BuildHashValues(), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Operation number mismatch in SuperNodeInfo and OperationInfo"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

} // namespace tile_fwk
} // namespace npu
