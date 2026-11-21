/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_vf_fusion_cluster_identify.cpp
 * \brief Unit test for VFFusionClusterIdentify.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "computational_graph_builder.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "passes/block_graph_pass/vf_fusion_cluster_identify.h"
#include "passes/pass_interface/pass.h"

#include "passes/pass_mgr/pass_manager.h"
#include "passes/pass_mgr/pass_registry.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

class VFFusionClusterIdentifyTestAccessor {
public:
    static void MarkVfFusionCluster(VFFusionClusterIdentify& pass, Operation& op) { pass.SetVfFusionCluster(op, true); }

    static int64_t GetUnrollBand(const Operation* op) { return VFFusionClusterIdentify::GetUnrollBand(op); }

    static bool GetClusterUnrollBand(const std::vector<Operation*>& clusterOps, int64_t& band)
    {
        return VFFusionClusterIdentify::GetClusterUnrollBand(clusterOps, band);
    }

    static bool IsPrefixSetCompatible(VFFusionClusterIdentify& pass,
                                      const std::vector<std::vector<uint64_t>>& taskPrefix,
                                      const std::vector<uint64_t>& prefixA, const std::vector<uint64_t>& prefixB)
    {
        VFFusionClusterIdentify::PrefixContext prefixCtx;
        prefixCtx.taskMembers.resize(taskPrefix.size());
        prefixCtx.taskPrefix = taskPrefix;
        return pass.IsPrefixSetCompatible(prefixCtx, prefixA, prefixB);
    }

    static bool VerifyScheduleTopology(VFFusionClusterIdentify& pass, Function& function,
                                       const std::vector<Operation*>& schedule)
    {
        auto graph = pass.BuildGraph(function);
        return pass.VerifyScheduleTopology(graph, schedule);
    }

    static std::vector<Operation*> BuildDfsPriorityTopoOps(VFFusionClusterIdentify& pass, Function& function)
    {
        auto graph = pass.BuildGraph(function);
        std::vector<size_t> topoOrder;
        if (!pass.BuildDfsPriorityTopoOrder(graph, topoOrder)) {
            return {};
        }
        std::vector<Operation*> result;
        result.reserve(topoOrder.size());
        for (size_t opIndex : topoOrder) {
            if (opIndex >= graph.ops.size()) {
                return {};
            }
            result.emplace_back(graph.ops[opIndex]);
        }
        return result;
    }

    // Member count of every cube task built for the function's graph, in task-id order. Ops in
    // singleton groups (or non-cube ops) map to -1 and never appear in any task.
    static std::vector<size_t> BuildCubeTaskMemberCounts(VFFusionClusterIdentify& pass, Function& function)
    {
        auto graph = pass.BuildGraph(function);
        std::vector<int> cubeTaskOfOp;
        std::vector<std::vector<size_t>> taskMembers;
        std::vector<std::vector<std::string>> taskMatMulOps;
        pass.BuildCubeTasks(graph, cubeTaskOfOp, taskMembers, taskMatMulOps);
        std::vector<size_t> memberCounts;
        memberCounts.reserve(taskMembers.size());
        for (const auto& members : taskMembers) {
            memberCounts.emplace_back(members.size());
        }
        return memberCounts;
    }

    // Cube task id of an op (-1 when the op is not part of any cube task).
    static int CubeTaskOf(VFFusionClusterIdentify& pass, Function& function, Operation* op)
    {
        auto graph = pass.BuildGraph(function);
        auto iter = graph.opToIndex.find(op);
        if (iter == graph.opToIndex.end()) {
            return -1;
        }
        std::vector<int> cubeTaskOfOp;
        std::vector<std::vector<size_t>> taskMembers;
        std::vector<std::vector<std::string>> taskMatMulOps;
        pass.BuildCubeTasks(graph, cubeTaskOfOp, taskMembers, taskMatMulOps);
        return iter->second < cubeTaskOfOp.size() ? cubeTaskOfOp[iter->second] : -1;
    }
};

namespace {
constexpr int VF_CLUSTER_ID_START_FOR_TEST = 200000000;
constexpr int VF_CLUSTER_SIZE_LIMIT_FOR_TEST = 32;
constexpr int USER_ATOMIC_SCOPE_ID_FOR_TEST = 8;

std::shared_ptr<Function> CreateRootFunction(const std::string& name)
{
    auto rootFunc = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    rootFunc->rootFunc_ = rootFunc.get();
    rootFunc->SetGraphType(GraphType::BLOCK_GRAPH);
    rootFunc->paramConfigs_.oooSchedMode = "HLF";
    return rootFunc;
}

std::shared_ptr<Function> CreateLeafFunction(Function& rootFunc, const std::string& name)
{
    auto leafFunc = std::make_shared<Function>(Program::GetInstance(), name, name, &rootFunc);
    leafFunc->rootFunc_ = &rootFunc;
    leafFunc->SetGraphType(GraphType::BLOCK_GRAPH);
    leafFunc->SetFunctionType(FunctionType::STATIC);
    rootFunc.rootFunc_->programs_.emplace(leafFunc->GetFuncMagic(), leafFunc.get());
    return leafFunc;
}

std::vector<int> GetOpMagics(Function& function)
{
    std::vector<int> result;
    for (auto* op : function.Operations(false).DuplicatedOpList()) {
        result.emplace_back(op->GetOpMagic());
    }
    return result;
}

std::vector<int> GetOpMagics(const std::vector<Operation*>& ops)
{
    std::vector<int> result;
    for (auto* op : ops) {
        result.emplace_back(op->GetOpMagic());
    }
    return result;
}

// UT builds the block graph directly, so no tile-graph pass has stamped the isCube attribute
// yet; VFFusionClusterIdentify (like TaskSplitter) classifies ops solely by that attribute.
// Stamp it here for every statically AIC-registered op, mirroring what the tile-graph
// graph-partition passes do in a real compile.
void MarkCubeOps(ComputationalGraphBuilder& graph)
{
    for (auto& item : graph.operations_) {
        if (OpcodeManager::Inst().GetCoreType(item.second->GetOpcode()) == OpCoreType::AIC) {
            item.second->SetAttribute(OpAttributeKey::isCube, true);
        }
    }
}

// Mirror of the frontend stamping: Function::AddRawOperation stamps the current
// SetUnrollBranchId context onto every newly created op. Tests that need ops to look like
// they were recorded inside an unrolled-loop branch can either wrap graph.AddOp with
// SetUnrollBranchId or stamp existing ops directly with this helper.
void StampUnrollBand(Operation* op, int64_t band) { op->SetAttribute(OpAttributeKey::unrollBranchId, band); }

} // namespace

TEST(AncestorBitsTest, PropagatesAncestorsAcrossBranchJoinAndDescendant)
{
    AncestorBits ancestorBits;
    const std::vector<std::vector<size_t>> producers = {
        {},     // 0: left source
        {},     // 1: right source
        {0},    // 2: left branch
        {1},    // 3: right branch
        {2, 3}, // 4: branch join
        {4},    // 5: downstream consumer
        {1},    // 6: independent side branch
    };
    const std::vector<size_t> topoOrder = {0, 1, 2, 3, 4, 5, 6};

    ancestorBits.Build(producers, topoOrder);

    EXPECT_EQ(ancestorBits.Size(), producers.size());
    for (size_t ancestorIndex : {0UL, 1UL, 2UL, 3UL, 4UL}) {
        EXPECT_TRUE(ancestorBits.IsAncestor(5, ancestorIndex));
    }
    EXPECT_TRUE(ancestorBits.IsAncestor(4, 0));
    EXPECT_TRUE(ancestorBits.IsAncestor(4, 1));
    EXPECT_TRUE(ancestorBits.IsAncestor(6, 1));
    EXPECT_FALSE(ancestorBits.IsAncestor(2, 1));
    EXPECT_FALSE(ancestorBits.IsAncestor(6, 0));
    EXPECT_FALSE(ancestorBits.IsAncestor(4, 5));
    EXPECT_FALSE(ancestorBits.IsAncestor(5, 7));
}

TEST(AncestorBitsTest, TracksAncestorsAcrossBitsetWordBoundary)
{
    constexpr size_t nodeCount = 130;
    AncestorBits ancestorBits;
    std::vector<std::vector<size_t>> producers(nodeCount);
    std::vector<size_t> topoOrder;
    topoOrder.reserve(nodeCount);
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
        topoOrder.emplace_back(nodeIndex);
        if (nodeIndex > 0) {
            producers[nodeIndex].emplace_back(nodeIndex - 1);
        }
    }

    ancestorBits.Build(producers, topoOrder);

    EXPECT_EQ(ancestorBits.Size(), nodeCount);
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 0));
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 63));
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 64));
    EXPECT_TRUE(ancestorBits.IsAncestor(nodeCount - 1, 128));
    EXPECT_TRUE(ancestorBits.IsAncestor(64, 63));
    EXPECT_FALSE(ancestorBits.IsAncestor(64, 64));
    EXPECT_FALSE(ancestorBits.IsAncestor(63, 64));
    EXPECT_FALSE(ancestorBits.IsAncestor(nodeCount - 1, nodeCount));
}

class VFFusionClusterIdentifyTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPassGlobalConfig(KEY_ENABLE_VF, true);
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    }

    void TearDown() override {}
};

TEST_F(VFFusionClusterIdentifyTest, AssignsOneClusterToDependentPipeVChain)
{
    auto rootFunc = CreateRootFunction("TestVFFusionChainRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionChainLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Sqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    auto* sqrt = graph.GetOp("Sqrt");
    auto* copyOut = graph.GetOp("CopyOut");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);
    ASSERT_NE(copyOut, nullptr);
    const auto originalOrder = GetOpMagics(*leafFunc);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(sqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(copyOut->GetAtomicScopeId(), -1);
    EXPECT_EQ(GetOpMagics(*leafFunc), originalOrder);
}

TEST_F(VFFusionClusterIdentifyTest, AssignsDistinctClusterIdsAcrossLeafFunctions)
{
    auto rootFunc = CreateRootFunction("TestVFFusionDistinctIdRoot");
    auto firstLeafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDistinctIdFirstLeaf");
    auto secondLeafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDistinctIdSecondLeaf");

    ComputationalGraphBuilder firstGraph(firstLeafFunc.get());
    ASSERT_TRUE(firstGraph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2"}));
    ASSERT_TRUE(firstGraph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "FirstExp"));
    ASSERT_TRUE(firstGraph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "FirstSqrt"));
    auto* firstExp = firstGraph.GetOp("FirstExp");
    auto* firstSqrt = firstGraph.GetOp("FirstSqrt");
    ASSERT_NE(firstExp, nullptr);
    ASSERT_NE(firstSqrt, nullptr);

    ComputationalGraphBuilder secondGraph(secondLeafFunc.get());
    ASSERT_TRUE(secondGraph.AddTensors(DataType::DT_FP32, {16, 16}, {"b0", "b1", "b2"}));
    ASSERT_TRUE(secondGraph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "SecondExp"));
    ASSERT_TRUE(secondGraph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "SecondSqrt"));
    auto* secondExp = secondGraph.GetOp("SecondExp");
    auto* secondSqrt = secondGraph.GetOp("SecondSqrt");
    ASSERT_NE(secondExp, nullptr);
    ASSERT_NE(secondSqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    const int firstClusterId = firstLeafFunc->GetFuncMagic() < secondLeafFunc->GetFuncMagic() ?
                                   firstExp->GetAtomicScopeId() :
                                   secondExp->GetAtomicScopeId();
    const int secondClusterId = firstLeafFunc->GetFuncMagic() < secondLeafFunc->GetFuncMagic() ?
                                    secondExp->GetAtomicScopeId() :
                                    firstExp->GetAtomicScopeId();
    EXPECT_EQ(firstClusterId, VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(secondClusterId, VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_NE(firstClusterId, secondClusterId);
    EXPECT_EQ(firstExp->GetAtomicScopeId(), firstSqrt->GetAtomicScopeId());
    EXPECT_EQ(secondExp->GetAtomicScopeId(), secondSqrt->GetAtomicScopeId());
}

TEST_F(VFFusionClusterIdentifyTest, KeepsUserAtomicScopeSeparateFromGeneratedClusterIds)
{
    auto rootFunc = CreateRootFunction("TestVFFusionUserScopeSeparateRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionUserScopeSeparateLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"u0", "u1", "a0", "a1", "a2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"u0"}, {"u1"}, "UserExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    auto* userExp = graph.GetOp("UserExp");
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    ASSERT_NE(userExp, nullptr);
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    constexpr int userScopeId = USER_ATOMIC_SCOPE_ID_FOR_TEST;
    userExp->SetAtomicScopeId(userScopeId);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(userExp->GetAtomicScopeId(), userScopeId);
    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_NE(userExp->GetAtomicScopeId(), aExp->GetAtomicScopeId());
}

TEST_F(VFFusionClusterIdentifyTest, MovesUnrelatedMiddleOpOutOfClusterWindow)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMoveMiddleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMoveMiddleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "a3", "b0", "b1", "b2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"a2"}, {"a3"}, "ARsqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"a3"}, {"out"}, "CopyOut"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* aRsqrt = graph.GetOp("ARsqrt");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(aRsqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aRsqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(GetOpMagics(*leafFunc),
              std::vector<int>({aExp->GetOpMagic(), aSqrt->GetOpMagic(), aRsqrt->GetOpMagic(),
                                graph.GetOp("CopyOut")->GetOpMagic(), bExp->GetOpMagic(), bSqrt->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, BuildsDfsPriorityTopoOrderIndependentOfInitialSchedule)
{
    auto rootFunc = CreateRootFunction("TestVFFusionDfsOrderRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDfsOrderLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "a3", "a4", "b0", "b1"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"a2"}, {"a3"}, "Consumer"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a3"}, {"a4"}, "Descendant"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* consumer = graph.GetOp("Consumer");
    auto* descendant = graph.GetOp("Descendant");
    auto* bExp = graph.GetOp("BExp");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(consumer, nullptr);
    ASSERT_NE(descendant, nullptr);
    ASSERT_NE(bExp, nullptr);

    leafFunc->ScheduleBy({bExp, aExp, descendant, aSqrt, consumer}, true);

    VFFusionClusterIdentify pass;
    auto topoOps = VFFusionClusterIdentifyTestAccessor::BuildDfsPriorityTopoOps(pass, *leafFunc);

    EXPECT_EQ(GetOpMagics(*leafFunc),
              std::vector<int>({bExp->GetOpMagic(), aExp->GetOpMagic(), descendant->GetOpMagic(), aSqrt->GetOpMagic(),
                                consumer->GetOpMagic()}));
    EXPECT_EQ(GetOpMagics(topoOps), std::vector<int>({aExp->GetOpMagic(), aSqrt->GetOpMagic(), consumer->GetOpMagic(),
                                                      descendant->GetOpMagic(), bExp->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, RejectsTopologicalScheduleWithNonContiguousVfCluster)
{
    auto rootFunc = CreateRootFunction("TestVFFusionNonContiguousClusterRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonContiguousClusterLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(
        DataType::DT_FP32, {16, 16},
        {"clusterInput0", "clusterOutput0", "middleInput", "middleOutput", "clusterInput1", "clusterOutput1"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"clusterInput0"}, {"clusterOutput0"}, "ClusterExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"middleInput"}, {"middleOutput"}, "MiddleNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"clusterInput1"}, {"clusterOutput1"}, "ClusterSqrt"));
    auto* clusterExp = graph.GetOp("ClusterExp");
    auto* middleNeg = graph.GetOp("MiddleNeg");
    auto* clusterSqrt = graph.GetOp("ClusterSqrt");
    ASSERT_NE(clusterExp, nullptr);
    ASSERT_NE(middleNeg, nullptr);
    ASSERT_NE(clusterSqrt, nullptr);

    constexpr int clusterId = VF_CLUSTER_ID_START_FOR_TEST;
    clusterExp->SetAtomicScopeId(clusterId);
    clusterSqrt->SetAtomicScopeId(clusterId);
    VFFusionClusterIdentify pass;
    VFFusionClusterIdentifyTestAccessor::MarkVfFusionCluster(pass, *clusterExp);
    VFFusionClusterIdentifyTestAccessor::MarkVfFusionCluster(pass, *clusterSqrt);

    const std::vector<Operation*> schedule = {clusterExp, middleNeg, clusterSqrt};
    EXPECT_FALSE(VFFusionClusterIdentifyTestAccessor::VerifyScheduleTopology(pass, *leafFunc, schedule));
}

TEST_F(VFFusionClusterIdentifyTest, DissolvesSingleOpClusters)
{
    auto rootFunc = CreateRootFunction("TestVFFusionSingleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionSingleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t1"}, {"t2"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    ASSERT_NE(exp, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesConsumerWithMultipleInputClusters)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMultiInputRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMultiInputLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "b0", "b1", "b2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "b2"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, StampsRecordedOpsWithCurrentUnrollBranchId)
{
    // Function::AddRawOperation must stamp the thread-local unroll branch context onto every
    // newly recorded op and must not stamp anything once the context is cleared back to -1.
    SetUnrollBranchId(-1);
    auto rootFunc = CreateRootFunction("TestVFStampBandRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFStampBandLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3"}));
    SetUnrollBranchId(4);
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "BandedExp"));
    SetUnrollBranchId(-1);
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t1"}, {"t2"}, "PlainExp"));
    auto* bandedExp = graph.GetOp("BandedExp");
    auto* plainExp = graph.GetOp("PlainExp");
    ASSERT_NE(bandedExp, nullptr);
    ASSERT_NE(plainExp, nullptr);

    EXPECT_EQ(VFFusionClusterIdentifyTestAccessor::GetUnrollBand(bandedExp), 4);
    EXPECT_EQ(VFFusionClusterIdentifyTestAccessor::GetUnrollBand(plainExp), -1);
    EXPECT_TRUE(bandedExp->HasAttribute(OpAttributeKey::unrollBranchId));
    EXPECT_FALSE(plainExp->HasAttribute(OpAttributeKey::unrollBranchId));
}

TEST_F(VFFusionClusterIdentifyTest, ClassifiesClusterUnrollBand)
{
    SetUnrollBranchId(-1);
    auto rootFunc = CreateRootFunction("TestVFBandClassRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFBandClassLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3", "t4", "t5", "t6"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Band5First"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t1"}, {"t2"}, "Band5Second"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t2"}, {"t3"}, "Band7"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t3"}, {"t4"}, "Wildcard"));
    auto* band5First = graph.GetOp("Band5First");
    auto* band5Second = graph.GetOp("Band5Second");
    auto* band7 = graph.GetOp("Band7");
    auto* wildcard = graph.GetOp("Wildcard");
    ASSERT_NE(band5First, nullptr);
    ASSERT_NE(band5Second, nullptr);
    ASSERT_NE(band7, nullptr);
    ASSERT_NE(wildcard, nullptr);
    StampUnrollBand(band5First, 5);
    StampUnrollBand(band5Second, 5);
    StampUnrollBand(band7, 7);

    int64_t band = -1;
    EXPECT_TRUE(VFFusionClusterIdentifyTestAccessor::GetClusterUnrollBand({}, band));
    EXPECT_EQ(band, -1);
    EXPECT_TRUE(VFFusionClusterIdentifyTestAccessor::GetClusterUnrollBand({wildcard}, band));
    EXPECT_EQ(band, -1);
    EXPECT_TRUE(VFFusionClusterIdentifyTestAccessor::GetClusterUnrollBand({wildcard, band5First}, band));
    EXPECT_EQ(band, 5);
    EXPECT_TRUE(VFFusionClusterIdentifyTestAccessor::GetClusterUnrollBand({band5First, band5Second, wildcard}, band));
    EXPECT_EQ(band, 5);
    EXPECT_FALSE(VFFusionClusterIdentifyTestAccessor::GetClusterUnrollBand({band5First, band7}, band));
}

TEST_F(VFFusionClusterIdentifyTest, MergesChainWithinSameUnrollBand)
{
    SetUnrollBranchId(-1);
    auto rootFunc = CreateRootFunction("TestVFSameBandRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFSameBandLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "BandExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "BandSqrt"));
    auto* bandExp = graph.GetOp("BandExp");
    auto* bandSqrt = graph.GetOp("BandSqrt");
    ASSERT_NE(bandExp, nullptr);
    ASSERT_NE(bandSqrt, nullptr);
    StampUnrollBand(bandExp, 7);
    StampUnrollBand(bandSqrt, 7);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(bandExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bandSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, RejectsChainAcrossUnrollBands)
{
    SetUnrollBranchId(-1);
    auto rootFunc = CreateRootFunction("TestVFCrossBandRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFCrossBandLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Band0Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Band1Sqrt"));
    auto* band0Exp = graph.GetOp("Band0Exp");
    auto* band1Sqrt = graph.GetOp("Band1Sqrt");
    ASSERT_NE(band0Exp, nullptr);
    ASSERT_NE(band1Sqrt, nullptr);
    StampUnrollBand(band0Exp, 0);
    StampUnrollBand(band1Sqrt, 1);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // The band mismatch forbids the merge; both ops end as dissolved singleton clusters.
    EXPECT_EQ(band0Exp->GetAtomicScopeId(), -1);
    EXPECT_EQ(band1Sqrt->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, AllowsWildcardConsumerIntoBandedCluster)
{
    SetUnrollBranchId(-1);
    auto rootFunc = CreateRootFunction("TestVFWildcardConsumerRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFWildcardConsumerLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3", "t4"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "BandExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "BandSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"t2"}, {"t3"}, "WildcardRsqrt"));
    auto* bandExp = graph.GetOp("BandExp");
    auto* bandSqrt = graph.GetOp("BandSqrt");
    auto* wildcardRsqrt = graph.GetOp("WildcardRsqrt");
    ASSERT_NE(bandExp, nullptr);
    ASSERT_NE(bandSqrt, nullptr);
    ASSERT_NE(wildcardRsqrt, nullptr);
    StampUnrollBand(bandExp, 3);
    StampUnrollBand(bandSqrt, 3);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // Ops without a band (recorded outside any unrolled loop) stay fusible with a banded cluster.
    EXPECT_EQ(bandExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bandSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(wildcardRsqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, RejectsClusterMergeAcrossUnrollBands)
{
    // Band-gated variant of MergesConsumerWithMultipleInputClusters: two input chains from
    // different unroll branches may not be pulled into one cluster through their join consumer.
    SetUnrollBranchId(-1);
    auto rootFunc = CreateRootFunction("TestVFBandClusterMergeRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFBandClusterMergeLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "b0", "b1", "b2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "b2"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);
    StampUnrollBand(aExp, 0);
    StampUnrollBand(aSqrt, 0);
    StampUnrollBand(bExp, 1);
    StampUnrollBand(bSqrt, 1);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // Each branch keeps its own cluster; the wildcard join may only follow one of them.
    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
}

TEST_F(VFFusionClusterIdentifyTest, RejectsMergeWhenExternalMemoryExceedsUbThreshold)
{
    auto rootFunc = CreateRootFunction("TestVFFusionExternalMemoryRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionExternalMemoryLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    const size_t ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    ASSERT_GT(ubSize, 0UL);
    const size_t elementsPerTensor = (ubSize * 35 / 100) / sizeof(float) + 1;
    const std::vector<int64_t> shape = {static_cast<int64_t>(elementsPerTensor), 1};
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, shape, {"a0", "a1", "a2", "b0", "b1", "b2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "b2"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesConsumerOnlyWithLeafClusterThroughNonVfSideBranch)
{
    auto rootFunc = CreateRootFunction("TestVFFusionNonVfSideBranchRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonVfSideBranchLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(
        DataType::DT_FP32, {16, 16},
        {"input", "mainExpOut", "mainSqrtOut", "mainRsqrtOut", "sideNegOut", "sideAbsOut", "sideExpOut", "joinOut"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"input"}, {"mainExpOut"}, "MainExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"mainExpOut"}, {"mainSqrtOut"}, "MainSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"mainSqrtOut"}, {"mainRsqrtOut"}, "MainRsqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"mainExpOut"}, {"sideNegOut"}, "SideNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {"sideNegOut"}, {"sideAbsOut"}, "SideAbs"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"sideAbsOut"}, {"sideExpOut"}, "SideExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"mainRsqrtOut", "sideExpOut"}, {"joinOut"}, "JoinAdd"));
    auto* mainExp = graph.GetOp("MainExp");
    auto* mainSqrt = graph.GetOp("MainSqrt");
    auto* mainRsqrt = graph.GetOp("MainRsqrt");
    auto* sideNeg = graph.GetOp("SideNeg");
    auto* sideAbs = graph.GetOp("SideAbs");
    auto* sideExp = graph.GetOp("SideExp");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(mainExp, nullptr);
    ASSERT_NE(mainSqrt, nullptr);
    ASSERT_NE(mainRsqrt, nullptr);
    ASSERT_NE(sideNeg, nullptr);
    ASSERT_NE(sideAbs, nullptr);
    ASSERT_NE(sideExp, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(mainExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(mainSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(mainRsqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(sideNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(sideAbs->GetAtomicScopeId(), -1);
    EXPECT_EQ(sideExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_NE(mainRsqrt->GetAtomicScopeId(), joinAdd->GetAtomicScopeId());
}

TEST_F(VFFusionClusterIdentifyTest, RejectsMergeWhenNonVfBarrierWouldSplitCluster)
{
    auto rootFunc = CreateRootFunction("TestVFFusionNonVfBarrierRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonVfBarrierLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(
        graph.AddTensors(DataType::DT_FP32, {16, 16},
                         {"branchInput", "branchExpOut", "branchSqrtOut", "branchBarrierOut", "blockedJoinOut"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"branchInput"}, {"branchExpOut"}, "BranchExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"branchExpOut"}, {"branchBarrierOut"}, "BranchBarrierNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"branchExpOut"}, {"branchSqrtOut"}, "BranchSqrt"));
    ASSERT_TRUE(
        graph.AddOp(Opcode::OP_ADD, {"branchSqrtOut", "branchBarrierOut"}, {"blockedJoinOut"}, "BlockedJoinAdd"));
    auto* branchExp = graph.GetOp("BranchExp");
    auto* branchBarrierNeg = graph.GetOp("BranchBarrierNeg");
    auto* branchSqrt = graph.GetOp("BranchSqrt");
    auto* blockedJoinAdd = graph.GetOp("BlockedJoinAdd");
    ASSERT_NE(branchExp, nullptr);
    ASSERT_NE(branchBarrierNeg, nullptr);
    ASSERT_NE(branchSqrt, nullptr);
    ASSERT_NE(blockedJoinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(branchExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(branchSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(branchBarrierNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(blockedJoinAdd->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesOnlyInputClustersThatFitSizeLimit)
{
    // The join consumer's A-side input cluster is already at VF_CLUSTER_SIZE_LIMIT, so only the
    // B-side cluster may take it in; the size-limit gate must keep the full A cluster separate.
    auto rootFunc = CreateRootFunction("TestVFFusionPartialSizeRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionPartialSizeLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    std::vector<std::string> tensorNames;
    for (int i = 0; i <= static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        tensorNames.emplace_back("a" + std::to_string(i));
    }
    tensorNames.insert(tensorNames.end(), {"b0", "b1", "b2", "out"});
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a" + std::to_string(i)}, {"a" + std::to_string(i + 1)},
                                "AExp" + std::to_string(i)));
    }
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"b0"}, {"b1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"b1"}, {"b2"}, "BSqrt"));
    ASSERT_TRUE(
        graph.AddOp(Opcode::OP_ADD, {"a" + std::to_string(VF_CLUSTER_SIZE_LIMIT_FOR_TEST), "b2"}, {"out"}, "JoinAdd"));
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        auto* op = graph.GetOp("AExp" + std::to_string(i));
        ASSERT_NE(op, nullptr);
    }
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST); i++) {
        EXPECT_EQ(graph.GetOp("AExp" + std::to_string(i))->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    }
    EXPECT_EQ(bExp->GetAtomicScopeId(), bSqrt->GetAtomicScopeId());
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), bExp->GetAtomicScopeId());
    EXPECT_NE(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MovesMiddleOpAfterConsumerWhenItDependsOnCluster)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMoveAfterRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMoveAfterLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "side", "sideOut", "a3", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "side"}, {"sideOut"}, "MiddleAdd"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_RSQRT, {"a2"}, {"a3"}, "ARsqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"a3"}, {"out"}, "CopyOut"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* middleAdd = graph.GetOp("MiddleAdd");
    auto* aRsqrt = graph.GetOp("ARsqrt");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(middleAdd, nullptr);
    ASSERT_NE(aRsqrt, nullptr);
    middleAdd->SetAtomicScopeId(USER_ATOMIC_SCOPE_ID_FOR_TEST);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aRsqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(middleAdd->GetAtomicScopeId(), USER_ATOMIC_SCOPE_ID_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MovesMiddleOpBeforeClusterWhenConsumerDependsOnIt)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMoveBeforeRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMoveBeforeLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "side0", "side1", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"side0"}, {"side1"}, "SideNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "side1"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* sideNeg = graph.GetOp("SideNeg");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(sideNeg, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(sideNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(GetOpMagics(*leafFunc), std::vector<int>({sideNeg->GetOpMagic(), aExp->GetOpMagic(), aSqrt->GetOpMagic(),
                                                        joinAdd->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, RejectsMergeWhenMiddleOpMustStayBetweenClusterAndConsumer)
{
    auto rootFunc = CreateRootFunction("TestVFFusionMiddleBarrierRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionMiddleBarrierLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "mid", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"a2"}, {"mid"}, "MiddleNeg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "mid"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* middleNeg = graph.GetOp("MiddleNeg");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(middleNeg, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(middleNeg->GetAtomicScopeId(), -1);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), -1);
    EXPECT_EQ(GetOpMagics(*leafFunc), std::vector<int>({aExp->GetOpMagic(), aSqrt->GetOpMagic(),
                                                        middleNeg->GetOpMagic(), joinAdd->GetOpMagic()}));
}

TEST_F(VFFusionClusterIdentifyTest, MergesPastUserScopedSideInput)
{
    auto rootFunc = CreateRootFunction("TestVFFusionRejectSideInputRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionRejectSideInputLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a0", "a1", "a2", "u0", "u1", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"a0"}, {"a1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"a1"}, {"a2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"u0"}, {"u1"}, "UserSide"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"a2", "u1"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* userSide = graph.GetOp("UserSide");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(userSide, nullptr);
    ASSERT_NE(joinAdd, nullptr);
    userSide->SetAtomicScopeId(USER_ATOMIC_SCOPE_ID_FOR_TEST);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(userSide->GetAtomicScopeId(), USER_ATOMIC_SCOPE_ID_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, RespectsClusterSizeLimit)
{
    auto rootFunc = CreateRootFunction("TestVFFusionSizeLimitRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionSizeLimitLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    std::vector<std::string> tensorNames;
    for (int i = 0; i <= static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST + 1); i++) {
        tensorNames.emplace_back("t" + std::to_string(i));
    }
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST + 1); i++) {
        ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t" + std::to_string(i)}, {"t" + std::to_string(i + 1)},
                                "Exp" + std::to_string(i)));
    }
    std::vector<Operation*> ops;
    for (int i = 0; i < static_cast<int>(VF_CLUSTER_SIZE_LIMIT_FOR_TEST + 1); i++) {
        auto* op = graph.GetOp("Exp" + std::to_string(i));
        ASSERT_NE(op, nullptr);
        ops.emplace_back(op);
    }

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    for (size_t i = 0; i < VF_CLUSTER_SIZE_LIMIT_FOR_TEST; i++) {
        EXPECT_EQ(ops[i]->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    }
    EXPECT_EQ(ops.back()->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, DoesNothingWhenVfDisabled)
{
    config::SetPassGlobalConfig(KEY_ENABLE_VF, false);
    auto rootFunc = CreateRootFunction("TestVFFusionDisabledRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionDisabledLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Sqrt"));
    auto* exp = graph.GetOp("Exp");
    auto* sqrt = graph.GetOp("Sqrt");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
    EXPECT_EQ(sqrt->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, DoesNothingWhenSchedModeNotHlf)
{
    auto rootFunc = CreateRootFunction("TestVFFusionNotHlfRoot");
    rootFunc->paramConfigs_.oooSchedMode = "";
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNotHlfLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Sqrt"));
    auto* exp = graph.GetOp("Exp");
    auto* sqrt = graph.GetOp("Sqrt");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
    EXPECT_EQ(sqrt->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, DoesNothingOnNonA5Arch)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    auto rootFunc = CreateRootFunction("TestVFFusionNonA5Root");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionNonA5Leaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Sqrt"));
    auto* exp = graph.GetOp("Exp");
    auto* sqrt = graph.GetOp("Sqrt");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(sqrt, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
    EXPECT_EQ(sqrt->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, FailsOnCyclicOperationDependencies)
{
    auto rootFunc = CreateRootFunction("TestVFFusionCycleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFFusionCycleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "First"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"t1"}, {"t2"}, "Second"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t2"}, {"t1"}, "Third"));

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), FAILED);
}

TEST_F(VFFusionClusterIdentifyTest, MergesExpandFollowedByElementwise)
{
    auto rootFunc = CreateRootFunction("TestExpandEleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestExpandEleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {1, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXPAND, {"t0"}, {"t1"}, "Expand"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t1"}, {"t2"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* expand = graph.GetOp("Expand");
    auto* exp = graph.GetOp("Exp");
    ASSERT_NE(expand, nullptr);
    ASSERT_NE(exp, nullptr);
    expand->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(expand->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MergesElementwiseFollowedByExpand)
{
    auto rootFunc = CreateRootFunction("TestEleExpandRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestEleExpandLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16, 16}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16, 16}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "Expand"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    auto* expand = graph.GetOp("Expand");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(expand, nullptr);
    expand->SetAttribute(OpAttributeKey::expandDims, std::vector<int>{0});

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // Producer (Exp) is not a reduce op, so merge is allowed regardless of shape difference.
    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(expand->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, MergesElementwiseFollowedByReduce)
{
    auto rootFunc = CreateRootFunction("TestEleReduceRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestEleReduceLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t0"}, {"t1"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"t1"}, {"t2"}, "Reduce"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* exp = graph.GetOp("Exp");
    auto* reduce = graph.GetOp("Reduce");
    ASSERT_NE(exp, nullptr);
    ASSERT_NE(reduce, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(exp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(reduce->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
}

TEST_F(VFFusionClusterIdentifyTest, RejectsMergeWhenReduceFollowedByElementwise)
{
    auto rootFunc = CreateRootFunction("TestReduceEleRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestReduceEleLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t0"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t1"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t2"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ROWSUM_SINGLE, {"t0"}, {"t1"}, "Reduce"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"t1"}, {"t2"}, "Exp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_OUT, {"t2"}, {"t3"}, "CopyOut"));
    auto* reduce = graph.GetOp("Reduce");
    auto* exp = graph.GetOp("Exp");
    ASSERT_NE(reduce, nullptr);
    ASSERT_NE(exp, nullptr);

    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // Reduce producer check rejects merge: exp's producer is a reduce op.
    // Both ops end up without a cluster: exp can't merge, and reduce becomes a
    // singleton cluster that gets dissolved by DissolveSingletonClusters.
    EXPECT_EQ(reduce->GetAtomicScopeId(), -1);
    EXPECT_EQ(exp->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesForkBranchesSharingSingleCubeTaskPrefix)
{
    // Rule A: two vector branches fed by the SAME cube task share an equal prefix and fully fuse.
    // mm1 -> AExp -> ASqrt -->
    //                            > JoinAdd
    // mm1 -> BExp -> BSqrt -->
    auto rootFunc = CreateRootFunction("TestVFRuleAForkRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFRuleAForkLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "b", "m1", "e1", "e2", "s1", "s2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"a", "b"}, {"m1"}, "Mm1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m1"}, {"e1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"e1"}, {"e2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m1"}, {"s1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"s1"}, {"s2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"e2", "s2"}, {"out"}, "JoinAdd"));
    auto* mm1 = graph.GetOp("Mm1");
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(mm1, nullptr);
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(mm1->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, KeepsBranchesUnderDifferentCubeTasksSeparate)
{
    // FA fork pattern: branches under DIFFERENT cube tasks must not fuse through the join consumer.
    // mm1 -> AExp -> ASqrt -->
    //                            > JoinAdd   Both branches stay separate so each can overlap with
    // mm2 -> BExp -> BSqrt -->   its matmul on another core. JoinAdd cannot follow branch A via
    // Rule B either: its extra task T2={mm2} has an empty own prefix (independent parallel root
    // fed from GM), so it is not a serial extension of branch A's common prefix -- the
    // serial-extension check rejects the merge and the singleton JoinAdd cluster dissolves.
    auto rootFunc = CreateRootFunction("TestVFDifferentPrefixForkRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFDifferentPrefixForkLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(
        graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "b", "c", "d", "m1", "m2", "e1", "e2", "s1", "s2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"a", "b"}, {"m1"}, "Mm1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"c", "d"}, {"m2"}, "Mm2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m1"}, {"e1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"e1"}, {"e2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m2"}, {"s1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"s1"}, {"s2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"e2", "s2"}, {"out"}, "JoinAdd"));
    auto* mm1 = graph.GetOp("Mm1");
    auto* mm2 = graph.GetOp("Mm2");
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(mm1, nullptr);
    ASSERT_NE(mm2, nullptr);
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST + 1);
    // JoinAdd's difference task T2={mm2} is an independent root (empty task prefix), so Rule B
    // rejects the merge into branch A; the singleton cluster dissolves to -1.
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), -1);
    EXPECT_EQ(mm1->GetAtomicScopeId(), -1);
    EXPECT_EQ(mm2->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesSubsetPrefixBranchThroughSerialExtension)
{
    // Rule B accept: the extra cube task of branch B serially extends the common prefix and does
    // not depend on the fused ops, so both branches may fuse.
    // mm1 -> Neg0 -> mm2 -> BSqrt -->
    //                                    > JoinAdd
    // mm1 -> AExp ---------------------->
    // P(AExp) = {T1}, P(BSqrt) = {T1, T2}, taskPrefix(T2) = {T1} <= common prefix {T1}.
    auto rootFunc = CreateRootFunction("TestVFRuleBSerialRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFRuleBSerialLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "b", "m1", "n", "m2", "e", "s", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"a", "b"}, {"m1"}, "Mm1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"m1"}, {"n"}, "Neg0"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"n", "b"}, {"m2"}, "Mm2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m1"}, {"e"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"m2"}, {"s"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"e", "s"}, {"out"}, "JoinAdd"));
    auto* neg0 = graph.GetOp("Neg0");
    auto* aExp = graph.GetOp("AExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(neg0, nullptr);
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(neg0->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, RejectsSubsetPrefixWhenExtensionTaskHasExternalCubeDep)
{
    // Rule B reject (serial-extension check): JoinAdd's prefix is a strict SUPERSET of the
    // {AExp, ASqrt} cluster prefix {T1}, and the difference task mm2 depends on the SIDE cube
    // task mm3 which is outside the common prefix, so mm2 forks away instead of serially
    // extending the common prefix. The mm3->Neg->mm2 chain does NOT depend on the cluster, so
    // schedule compaction alone would allow the merge: only the prefix gate rejects it.
    // mm1 -> AExp -> ASqrt ------------> JoinAdd
    // mm3 -> Neg -> mm2 ---------------/
    // T1={mm1}, T2={mm2}, T3={mm3}
    // P(ASqrt)={T1}, P(JoinAdd)={T1,T2,T3}, TP(T2)={T3} not subset of common prefix {T1}
    // -> serial-extension check rejects the fusion.
    auto rootFunc = CreateRootFunction("TestVFRuleBForkExtRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFRuleBForkExtLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(
        graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "b", "c", "d", "m1", "m3", "n", "e1", "e2", "m2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"a", "b"}, {"m1"}, "Mm1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m1"}, {"e1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"e1"}, {"e2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"c", "d"}, {"m3"}, "Mm3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"m3"}, {"n"}, "Neg"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"n", "d"}, {"m2"}, "Mm2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"e2", "m2"}, {"out"}, "JoinAdd"));
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    // JoinAdd is a strict superset-prefix consumer whose difference task mm2 depends on the
    // external task mm3: not a serial extension of the common prefix -> merge rejected.
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesBranchesFedByChainedCubeTask)
{
    // Cube task clustering: directly connected cube ops form ONE cube task, so branches after a
    // mm1 -> mm2 chain share a single-element prefix and fully fuse (Rule A).
    auto rootFunc = CreateRootFunction("TestVFCubeChainRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFCubeChainLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "b", "m1", "m2", "e1", "e2", "s1", "s2", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"a", "b"}, {"m1"}, "Mm1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"m1", "b"}, {"m2"}, "Mm2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m2"}, {"e1"}, "AExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"e1"}, {"e2"}, "ASqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"m2"}, {"s1"}, "BExp"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"s1"}, {"s2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"e2", "s2"}, {"out"}, "JoinAdd"));
    auto* mm1 = graph.GetOp("Mm1");
    auto* mm2 = graph.GetOp("Mm2");
    auto* aExp = graph.GetOp("AExp");
    auto* aSqrt = graph.GetOp("ASqrt");
    auto* bExp = graph.GetOp("BExp");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(mm1, nullptr);
    ASSERT_NE(mm2, nullptr);
    ASSERT_NE(aExp, nullptr);
    ASSERT_NE(aSqrt, nullptr);
    ASSERT_NE(bExp, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    EXPECT_EQ(aExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(aSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bExp->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(mm1->GetAtomicScopeId(), -1);
    EXPECT_EQ(mm2->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, RejectsAccumulatorChainAcrossIndependentCubeTasks)
{
    // FA unrolled-loop pattern: per-iteration matmuls T_j are INDEPENDENT parallel roots (all fed
    // straight from GM, empty task prefixes), while the accumulator chains the iterations
    // serially. The update block of iteration j+1 has a strictly larger prefix
    // ({T1..Tj+1} vs {T1..Tj}) because it consumes the previous accumulator, but the difference
    // task T_{j+1} has an EMPTY own prefix -- it is an independent root, not a serial extension
    // of the common prefix. Rule B must reject the merge so each iteration keeps its own cluster.
    // mm1 -> m1 -------------------> Max1 -> Exp1 -->
    // mm2 -> m2, Exp1 --------------> Max2 -> Exp2 -->   (accumulator passed across iterations)
    // mm3 -> m3, Exp2 --------------> Max3 -> Exp3 -->
    // T1={mm1}, T2={mm2}, T3={mm3}, TP(Tk) = {} for all k (independent roots).
    // P(Max1)={T1}, P(Max2)={T1,T2}, P(Max3)={T1,T2,T3}.
    auto rootFunc = CreateRootFunction("TestVFAccumulatorChainRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFAccumulatorChainLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16},
                                 {"i1", "i2", "i3", "b", "m1", "m2", "m3", "x1", "x2", "x3", "y1", "y2", "y3"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"i1", "b"}, {"m1"}, "Mm1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"i2", "b"}, {"m2"}, "Mm2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"i3", "b"}, {"m3"}, "Mm3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_MAXIMUM, {"m1"}, {"x1"}, "Max1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"x1"}, {"y1"}, "Exp1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_MAXIMUM, {"m2", "y1"}, {"x2"}, "Max2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"x2"}, {"y2"}, "Exp2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_MAXIMUM, {"m3", "y2"}, {"x3"}, "Max3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_EXP, {"x3"}, {"y3"}, "Exp3"));
    auto* mm1 = graph.GetOp("Mm1");
    auto* mm2 = graph.GetOp("Mm2");
    auto* mm3 = graph.GetOp("Mm3");
    auto* max1 = graph.GetOp("Max1");
    auto* max2 = graph.GetOp("Max2");
    auto* max3 = graph.GetOp("Max3");
    auto* exp1 = graph.GetOp("Exp1");
    auto* exp2 = graph.GetOp("Exp2");
    auto* exp3 = graph.GetOp("Exp3");
    ASSERT_NE(mm1, nullptr);
    ASSERT_NE(mm2, nullptr);
    ASSERT_NE(mm3, nullptr);
    ASSERT_NE(max1, nullptr);
    ASSERT_NE(max2, nullptr);
    ASSERT_NE(max3, nullptr);
    ASSERT_NE(exp1, nullptr);
    ASSERT_NE(exp2, nullptr);
    ASSERT_NE(exp3, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // Each iteration fuses only its own MAX+EXP pair (equal prefixes within the iteration).
    EXPECT_EQ(max1->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(exp1->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    // Iteration 2 must NOT merge into iteration 1: difference task T2 has an empty own prefix.
    EXPECT_NE(max2->GetAtomicScopeId(), max1->GetAtomicScopeId());
    EXPECT_EQ(max2->GetAtomicScopeId(), exp2->GetAtomicScopeId());
    // Iteration 3 must NOT merge into the previous iterations either.
    EXPECT_NE(max3->GetAtomicScopeId(), max1->GetAtomicScopeId());
    EXPECT_NE(max3->GetAtomicScopeId(), max2->GetAtomicScopeId());
    EXPECT_EQ(max3->GetAtomicScopeId(), exp3->GetAtomicScopeId());
    EXPECT_EQ(mm1->GetAtomicScopeId(), -1);
    EXPECT_EQ(mm2->GetAtomicScopeId(), -1);
    EXPECT_EQ(mm3->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesComplementaryPrefixMembersThroughClusterUnion)
{
    // Cluster-level union comparison: two branch heads whose prefixes fork pairwise
    // ({T1,T2} vs {T1,T3}) must still fuse through the join consumer. The consumer joins one
    // branch first (Rule B: difference task anchored to the common prefix), then the other
    // branch's cluster merges in: comparing the SIDES AS UNIONS gives {T1,T2,T3} vs {T1,T3},
    // whose difference task T2 has TP(T2)={T1} inside the common prefix -> Rule B accepts.
    // The old pairwise check rejected this merge because BSqrt vs CSqrt fork in isolation.
    // mm1 -> Neg0 -> mm2 -> BSqrt -->
    //                             > JoinAdd
    // mm1 -> Neg1 -> mm3 -> CSqrt ->
    // T1={mm1}, T2={mm2}, T3={mm3}, TP(T2)={T1}, TP(T3)={T1}.
    // P(BSqrt)={T1,T2}, P(CSqrt)={T1,T3}, P(JoinAdd)={T1,T2,T3}.
    auto rootFunc = CreateRootFunction("TestVFUnionPrefixRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFUnionPrefixLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    ASSERT_TRUE(
        graph.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "b", "m1", "n0", "m2", "s2", "n1", "m3", "s3", "out"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"a", "b"}, {"m1"}, "Mm1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"m1"}, {"n0"}, "Neg0"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"n0", "b"}, {"m2"}, "Mm2"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"m2"}, {"s2"}, "BSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_NEG, {"m1"}, {"n1"}, "Neg1"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"n1", "b"}, {"m3"}, "Mm3"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_SQRT, {"m3"}, {"s3"}, "CSqrt"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ADD, {"s2", "s3"}, {"out"}, "JoinAdd"));
    auto* mm1 = graph.GetOp("Mm1");
    auto* neg0 = graph.GetOp("Neg0");
    auto* neg1 = graph.GetOp("Neg1");
    auto* bSqrt = graph.GetOp("BSqrt");
    auto* cSqrt = graph.GetOp("CSqrt");
    auto* joinAdd = graph.GetOp("JoinAdd");
    ASSERT_NE(mm1, nullptr);
    ASSERT_NE(neg0, nullptr);
    ASSERT_NE(neg1, nullptr);
    ASSERT_NE(bSqrt, nullptr);
    ASSERT_NE(cSqrt, nullptr);
    ASSERT_NE(joinAdd, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    // All three vector ops fuse into one cluster via the union-prefix cluster merge.
    EXPECT_EQ(bSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(cSqrt->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(joinAdd->GetAtomicScopeId(), VF_CLUSTER_ID_START_FOR_TEST);
    EXPECT_EQ(mm1->GetAtomicScopeId(), -1);
    EXPECT_EQ(neg0->GetAtomicScopeId(), -1);
    EXPECT_EQ(neg1->GetAtomicScopeId(), -1);
}

TEST_F(VFFusionClusterIdentifyTest, MergesForkedPrefixesWhenDifferenceTasksAreAnchoredAcrossSides)
{
    // Prefixes are not subsets, but each difference task is serially anchored by the other side:
    // P(A)={T0,T1,T6}, P(B)={T0,T1,T8}; TP(T6)={T0,T1}, TP(T8)={T0,T1}.
    const std::vector<std::vector<uint64_t>> taskPrefix = {
        {}, {}, {}, {}, {}, {}, {0b11}, {}, {0b11},
    };
    const std::vector<uint64_t> prefixA = {0b01000011};
    const std::vector<uint64_t> prefixB = {0b100000011};

    VFFusionClusterIdentify pass;
    EXPECT_TRUE(VFFusionClusterIdentifyTestAccessor::IsPrefixSetCompatible(pass, taskPrefix, prefixA, prefixB));
}

TEST_F(VFFusionClusterIdentifyTest, MergesForkedPrefixesWhenDifferenceTaskHasAnyAncestorOnOppositeSide)
{
    // A difference task is allowed when at least one of its ancestors belongs to the opposite
    // prefix, even if another ancestor is outside that prefix. This is weaker than a full subset
    // relation and matches the cross-side anchoring rule.
    // P(A)={T0,T1,T6}, P(B)={T0,T1,T8}; TP(T6)={T0,T2}, TP(T8)={T1,T3}.
    const std::vector<std::vector<uint64_t>> taskPrefix = {
        {}, {}, {}, {}, {}, {}, {0b101}, {}, {0b1010},
    };
    const std::vector<uint64_t> prefixA = {0b01000011};
    const std::vector<uint64_t> prefixB = {0b100000011};

    VFFusionClusterIdentify pass;
    EXPECT_TRUE(VFFusionClusterIdentifyTestAccessor::IsPrefixSetCompatible(pass, taskPrefix, prefixA, prefixB));
}

TEST_F(VFFusionClusterIdentifyTest, MergesForkedPrefixesWhenDifferenceTasksShareACubePrefix)
{
    // Neither difference task is fully anchored by the opposite set, but their own cube-task
    // prefixes overlap at T0: P(A)={T0,T1,T2,T6}, P(B)={T0,T1,T3,T8},
    // TP(T6)={T0,T2}, TP(T8)={T0,T3}.
    const std::vector<std::vector<uint64_t>> taskPrefix = {
        {}, {}, {0b1}, {0b1}, {}, {}, {0b101}, {}, {0b1001},
    };
    const std::vector<uint64_t> prefixA = {0b01000111};
    const std::vector<uint64_t> prefixB = {0b100001011};

    VFFusionClusterIdentify pass;
    EXPECT_TRUE(VFFusionClusterIdentifyTestAccessor::IsPrefixSetCompatible(pass, taskPrefix, prefixA, prefixB));
}

TEST_F(VFFusionClusterIdentifyTest, FormsFourCubeTasksWhenBranchesShareOneL1CopyIn)
{
    // 4-way parallel matmul branches whose L0A path is fed through ONE shared L1 copy
    // (DDR -> L1, multi-consumer):
    //              ┌─ L1_TO_L0A_0 ─ A_MUL_B_0 ─┐
    //   DDR ─ COPY ─┼─ L1_TO_L0A_1 ─ A_MUL_B_1 ─┼─ ...
    //              ├─ L1_TO_L0A_2 ─ A_MUL_B_2 ─┤
    //              └─ L1_TO_L0A_3 ─ A_MUL_B_3 ─┘
    // Expected cube-task partition: exactly FOUR tasks, one {L1_TO_L0A_i, A_MUL_B_i} chain per
    // branch. The shared copy op forms no task at all: the L1 multi-consumer special-case keeps
    // it out of every union (without it all 9 cube ops would chain into ONE task), and the
    // singleton rule then drops its leftover one-op group (keeping it would give FIVE tasks).
    auto rootFunc = CreateRootFunction("TestVFSharedL1CopyInRoot");
    auto leafFunc = CreateLeafFunction(*rootFunc, "TestVFSharedL1CopyInLeaf");
    ComputationalGraphBuilder graph(leafFunc.get());
    const std::vector<std::string> tensorNames = {"a0", "l1", "la0", "la1", "la2", "la3", "b0",
                                                  "b1", "b2", "b3",  "m0",  "m1",  "m2",  "m3"};
    const std::vector<MemoryType> memTypes = {
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1,                              // a0 -> l1
        MemoryType::MEM_L0A,        MemoryType::MEM_L0A,        MemoryType::MEM_L0A, // la0..la2
        MemoryType::MEM_L0A,                                                         // la3
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,                      // b0, b1
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,                      // b2, b3
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,                      // m0, m1
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,                      // m2, m3
    };
    ASSERT_EQ(tensorNames.size(), memTypes.size());
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, tensorNames));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_L1_COPY_IN, {"a0"}, {"l1"}, "SharedCopyIn"));
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(
            graph.AddOp(Opcode::OP_L1_TO_L0A, {"l1"}, {"la" + std::to_string(i)}, "L1ToL0A" + std::to_string(i)));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"la" + std::to_string(i), "b" + std::to_string(i)},
                                {"m" + std::to_string(i)}, "Mm" + std::to_string(i)));
    }
    auto* sharedCopyIn = graph.GetOp("SharedCopyIn");
    std::vector<Operation*> matmuls;
    for (int i = 0; i < 4; i++) {
        auto* mm = graph.GetOp("Mm" + std::to_string(i));
        ASSERT_NE(mm, nullptr);
        matmuls.emplace_back(mm);
    }
    ASSERT_NE(sharedCopyIn, nullptr);

    MarkCubeOps(graph);
    VFFusionClusterIdentify pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFunc), SUCCESS);

    const auto memberCounts = VFFusionClusterIdentifyTestAccessor::BuildCubeTaskMemberCounts(pass, *leafFunc);
    ASSERT_EQ(memberCounts.size(), 4UL);
    for (size_t taskIndex = 0; taskIndex < memberCounts.size(); taskIndex++) {
        EXPECT_EQ(memberCounts[taskIndex], 2UL); // {L1_TO_L0A_i, A_MUL_B_i} per task.
    }
    EXPECT_EQ(VFFusionClusterIdentifyTestAccessor::CubeTaskOf(pass, *leafFunc, sharedCopyIn), -1);
    std::vector<int> matmulTaskIds;
    for (auto* mm : matmuls) {
        matmulTaskIds.emplace_back(VFFusionClusterIdentifyTestAccessor::CubeTaskOf(pass, *leafFunc, mm));
    }
    for (int taskId : matmulTaskIds) {
        EXPECT_GE(taskId, 0); // Every matmul belongs to a real cube task.
    }
    std::sort(matmulTaskIds.begin(), matmulTaskIds.end());
    EXPECT_EQ(std::unique(matmulTaskIds.begin(), matmulTaskIds.end()) - matmulTaskIds.begin(), 4);
}

} // namespace npu::tile_fwk
