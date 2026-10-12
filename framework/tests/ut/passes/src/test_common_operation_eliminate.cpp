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
 * \file test_common_operation_eliminate.cpp
 * \brief Unit test for common_operation_eliminate pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "ir/span.h"
#include "ir/type.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_partition/common_operation_eliminate.h"
#include "symbolic_scalar_test_utils.h"
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {
namespace {
ir::VarPtr MakeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetUnknownType(), ir::Span::Unknown());
}

ir::StmtPtr ToStmtPtr(Operation* operation)
{
    return std::static_pointer_cast<const ir::Stmt>(operation->shared_from_this());
}

LogicalTensorPtr MakeDdrTensor(const std::vector<int64_t>& shape, const std::shared_ptr<RawTensor>& rawTensor = nullptr)
{
    auto tensor = rawTensor == nullptr ?
                      IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape)) :
                      IRBuilder().CreateTensorVar(rawTensor, {0, 0}, shape, CreateTestConstIntVector(shape));
    tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    return tensor;
}

size_t CountOperations(Function& function, Opcode opcode)
{
    size_t count = 0;
    for (const auto* operation : function.Operations().DuplicatedOpList()) {
        if (operation != nullptr && operation->GetOpcode() == opcode) {
            ++count;
        }
    }
    return count;
}

void BuildRedundantAbsGraph(ComputationalGraphBuilder& graph)
{
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(graph.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(graph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(graph.SetInCast({"t1"}), true);
    EXPECT_EQ(graph.SetOutCast({"t4"}), true);
}

Operation* FindKeptAbsProducer(Function& function)
{
    for (auto op : function.Operations().DuplicatedOpList()) {
        if (op != nullptr && op->GetOpcode() == Opcode::OP_ABS) {
            return op;
        }
    }
    return nullptr;
}
} // namespace

class CommonOperationEliminateTest : public testing::Test {
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

namespace {
void ExpectCommonOperationEliminateOpCount(ComputationalGraphBuilder& graph, size_t expectedOpCount)
{
    Function* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate coe;
    EXPECT_EQ(coe.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(function->Operations().size(), expectedOpCount);
}

bool SetAssembleAttributes(ComputationalGraphBuilder& graph)
{
    auto* assemble1 = graph.GetOp("ASSEMBLE1");
    auto* assemble2 = graph.GetOp("ASSEMBLE2");
    if (assemble1 == nullptr || assemble2 == nullptr) {
        return false;
    }
    assemble1->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{0, 0}));
    assemble2->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, std::vector<int64_t>{16, 0}));
    return true;
}

bool BuildVecDupAssembleGraph(ComputationalGraphBuilder& graph, MemoryType outMemoryType,
                              bool setVecDupMemoryType = true, bool setAssembleAttr = true)
{
    std::vector<Opcode> opCodes{Opcode::OP_VEC_DUP, Opcode::OP_VEC_DUP, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"out"}, {"out"}};
    std::vector<std::string> opNames{"VECDUP1", "VECDUP2", "ASSEMBLE1", "ASSEMBLE2"};
    bool vecDupOk = setVecDupMemoryType ? graph.AddTensors(DataType::DT_FP32, {16, 16},
                                                           {MemoryType::MEM_UB, MemoryType::MEM_UB}, {"t1", "t2"}) :
                                          graph.AddTensors(DataType::DT_FP32, {16, 16}, {"t1", "t2"});
    if (!vecDupOk || !graph.AddTensor(DataType::DT_FP32, {32, 16}, outMemoryType, "out") ||
        !graph.AddOps(opCodes, ioperands, ooperands, opNames, true)) {
        return false;
    }
    if (!setAssembleAttr) {
        return graph.SetOutCast({"out"});
    }
    if (!SetAssembleAttributes(graph)) {
        return false;
    }
    return graph.SetOutCast({"out"});
}

bool BuildUnaryOpAssembleGraph(ComputationalGraphBuilder& graph, MemoryType outMemoryType)
{
    std::vector<Opcode> opCodes{Opcode::OP_EXP, Opcode::OP_EXP, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"input"}, {"input"}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"out"}, {"out"}};
    std::vector<std::string> opNames{"EXP1", "EXP2", "ASSEMBLE1", "ASSEMBLE2"};
    if (!graph.AddTensors(DataType::DT_FP32, {16, 16}, {MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB},
                          {"input", "t1", "t2"}) ||
        !graph.AddTensor(DataType::DT_FP32, {32, 16}, outMemoryType, "out") ||
        !graph.AddOps(opCodes, ioperands, ooperands, opNames, true) || !SetAssembleAttributes(graph)) {
        return false;
    }
    return graph.SetInCast({"input"}) && graph.SetOutCast({"out"});
}
} // namespace

TEST_F(CommonOperationEliminateTest, EliminateRedundantOps)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiInputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiOutputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2", "t3"}, {"t4", "t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"RowMax1", "RowMax2", "MUL1", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6", "t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantCascadeOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "EXP1", "EXP2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
    std::shared_ptr<LogicalTensor> tensorPtr = G.GetTensor("t1");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetConsumers().size(), 1);
    tensorPtr = G.GetTensor("t6");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetProducers().size(), 1);
}

TEST_F(CommonOperationEliminateTest, IgnoreSingleInputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t3"}, {"t2", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t3"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreMultiInputOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t4"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentAttr)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ADDS1", "ADDS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ADDS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 1.0));
    opPtr = G.GetOp("ADDS2");
    EXPECT_NE(opPtr, nullptr);
    const double value2 = 2.0;
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, value2));
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3; // 修复后有序遍历tensor，使得连续冗余场景正确消除
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS", "EXP", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentSubgraph)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ABS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(0);
    opPtr = G.GetOp("ABS2");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(1);
    opPtr = G.GetOp("MUL");
    EXPECT_NE(opPtr, nullptr);
    const int subgraphID2 = 2;
    opPtr->UpdateSubgraphID(subgraphID2);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    const int subgraphNum = 3;
    function->SetTotalSubGraphCount(subgraphNum);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, RespectReduceCopyPreSubgraphId)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    auto* abs1 = G.GetOp("ABS1");
    auto* abs2 = G.GetOp("ABS2");
    auto* mul = G.GetOp("MUL");
    ASSERT_NE(abs1, nullptr);
    ASSERT_NE(abs2, nullptr);
    ASSERT_NE(mul, nullptr);
    // The current subgraph id is identical; only the ReduceCopy source subgraph id differs.
    abs1->UpdateSubgraphID(0);
    abs1->SetAttr(OpAttributeKey::reduceCopyPreSubgraphId, static_cast<int64_t>(0));
    abs2->UpdateSubgraphID(0);
    abs2->SetAttr(OpAttributeKey::reduceCopyPreSubgraphId, static_cast<int64_t>(1));
    mul->UpdateSubgraphID(0);
    EXPECT_NE(abs1->DumpAttr().find(OpAttributeKey::reduceCopyPreSubgraphId), std::string::npos);
    EXPECT_NE(abs2->DumpAttr().find(OpAttributeKey::reduceCopyPreSubgraphId), std::string::npos);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate COE;
    EXPECT_EQ(COE.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(function->Operations().size(), 3);
}

TEST_F(CommonOperationEliminateTest, EliminateDupOpsWithDifferentGraphMergeHashOrder)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    auto* abs1 = G.GetOp("ABS1");
    auto* abs2 = G.GetOp("ABS2");
    auto* mul = G.GetOp("MUL");
    ASSERT_NE(abs1, nullptr);
    ASSERT_NE(abs2, nullptr);
    ASSERT_NE(mul, nullptr);
    // Merge passes annotate equivalent producers of one merged subgraph with different
    // merge-group ids ("func<magic>_<order>"); these must not block common-op elimination.
    abs1->UpdateSubgraphID(0);
    abs2->UpdateSubgraphID(0);
    mul->UpdateSubgraphID(0);
    abs1->SetHashOrderInfo(OpAttributeKey::l1ReuseHashOrder, OpAttributeKey::l1ReuseSubgraphCount, "func0_0", 2);
    abs2->SetHashOrderInfo(OpAttributeKey::l1ReuseHashOrder, OpAttributeKey::l1ReuseSubgraphCount, "func0_1", 4);
    abs1->SetHashOrderInfo(OpAttributeKey::cubeMergeHashOrder, OpAttributeKey::cubeMergeSubgraphCount, "func0_0", 2);
    abs2->SetHashOrderInfo(OpAttributeKey::cubeMergeHashOrder, OpAttributeKey::cubeMergeSubgraphCount, "func0_1", 4);
    abs1->SetHashOrderInfo(OpAttributeKey::vecMergeHashOrder, OpAttributeKey::vecMergeSubgraphCount, "func0_0", 2);
    abs2->SetHashOrderInfo(OpAttributeKey::vecMergeHashOrder, OpAttributeKey::vecMergeSubgraphCount, "func0_1", 4);
    ExpectCommonOperationEliminateOpCount(G, 2);
}

TEST_F(CommonOperationEliminateTest, IgnoreSpecialOp)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW,      Opcode::OP_VIEW,      Opcode::OP_MUL,
                                Opcode::OP_L1_TO_FIX, Opcode::OP_L1_TO_FIX, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}, {"t1"}, {"t1"}, {"t5", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"VIEW1", "VIEW2", "MUL1", "COPY1", "COPY2", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4", "t7"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 6;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateVecDupWithoutAssemblePlacement)
{
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VEC_DUP, Opcode::OP_VEC_DUP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"VECDUP1", "VECDUP2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetOutCast({"t3"}), true);
    ExpectCommonOperationEliminateOpCount(G, 2);
}

TEST_F(CommonOperationEliminateTest, PreserveVecDupAssembledToSameMemory)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(BuildVecDupAssembleGraph(G, MemoryType::MEM_UB));
    ExpectCommonOperationEliminateOpCount(G, 4);
}

TEST_F(CommonOperationEliminateTest, EliminateVecDupAssembledToDifferentMemory)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(BuildVecDupAssembleGraph(G, MemoryType::MEM_DEVICE_DDR));
    ExpectCommonOperationEliminateOpCount(G, 3);
}

TEST_F(CommonOperationEliminateTest, PreserveCommonOpAssembledToSameMemory)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(BuildUnaryOpAssembleGraph(G, MemoryType::MEM_UB));
    ExpectCommonOperationEliminateOpCount(G, 4);
}

TEST_F(CommonOperationEliminateTest, PreserveAssembledWhenMemoryTypeUnknown)
{
    ComputationalGraphBuilder attrOnlyGraph;
    ASSERT_TRUE(BuildVecDupAssembleGraph(attrOnlyGraph, MemoryType::MEM_UB, false));
    ExpectCommonOperationEliminateOpCount(attrOnlyGraph, 4);

    ComputationalGraphBuilder unknownMemoryGraph;
    ASSERT_TRUE(BuildVecDupAssembleGraph(unknownMemoryGraph, MemoryType::MEM_UNKNOWN, false, false));
    ExpectCommonOperationEliminateOpCount(unknownMemoryGraph, 4);
}

TEST_F(CommonOperationEliminateTest, TestShmemLoadChecker)
{
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {1, 1}, {"dummy"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {1, 1, 4, 64}, {"shmemData"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {4, 64}, {"out"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_SHMEM_LOAD};
    std::vector<std::vector<std::string>> ioperands{{"dummy", "shmemData"}};
    std::vector<std::vector<std::string>> ooperands{{"out"}};
    std::vector<std::string> opNames{"TILE_SHMEM_LOAD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"dummy", "shmemData"}), true);
    EXPECT_EQ(G.SetOutCast({"out"}), true);
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, SUCCESS) << "COE Precheck failed for OP_SHMEM_LOAD!";
}

TEST_F(CommonOperationEliminateTest, PreCheck_CopyIn_InvalidInputNum)
{
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, {"t1", "t2", "t3"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"COPY_IN_InvalidInput"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(CommonOperationEliminateTest, PreCheck_CopyIn_OffsetShapeMismatch)
{
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, {"t1", "t2"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"COPY_IN_OffsetMismatch"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    Operation* copyOp = G.GetOp("COPY_IN_OffsetMismatch");
    ASSERT_NE(copyOp, nullptr);
    auto opAttr = copyOp->GetOpAttribute();
    ASSERT_NE(opAttr, nullptr);
    auto copyAttr = dynamic_cast<CopyOpAttribute*>(opAttr.get());
    ASSERT_NE(copyAttr, nullptr);
    std::vector<OpImmediate> newFromOffset;
    newFromOffset.emplace_back(0);
    newFromOffset.emplace_back(1);
    newFromOffset.emplace_back(2);
    copyAttr->SetFromOffset(newFromOffset);
    G.GetTensor("t1")->offset = {0, 0};
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(CommonOperationEliminateTest, PreserveCopyInForDistinctRawTensorVersions)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestDistinctRawTensorVersions",
                                               "TestDistinctRawTensorVersions", nullptr);
    ASSERT_NE(function, nullptr);

    const std::vector<int64_t> shape{8, 8};
    auto sourceRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    auto sourceVersion0 = MakeDdrTensor(shape, sourceRaw);
    auto sourceVersion1 = MakeDdrTensor(shape, sourceRaw);
    auto copyOutput0 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto copyOutput1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    copyOutput0->SetMemoryTypeBoth(MEM_UB, true);
    copyOutput1->SetMemoryTypeBoth(MEM_UB, true);
    auto addOutput0 = MakeDdrTensor(shape);
    auto addOutput1 = MakeDdrTensor(shape);

    PassOperationUtils::AddOperation(*function, Opcode::OP_COPY_IN, {sourceVersion0}, {copyOutput0});
    PassOperationUtils::AddOperation(*function, Opcode::OP_COPY_IN, {sourceVersion1}, {copyOutput1});
    auto& add0 = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {copyOutput0, copyOutput0}, {addOutput0});
    auto& add1 = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {copyOutput1, copyOutput1}, {addOutput1});
    add0.SetAttr(OpAttributeKey::dontTouch, true);
    add1.SetAttr(OpAttributeKey::dontTouch, true);
    function->inCasts_ = {sourceVersion0, sourceVersion1};
    function->outCasts_ = {addOutput0, addOutput1};

    CommonOperationEliminate coe;
    ASSERT_EQ(coe.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOperations(*function, Opcode::OP_COPY_IN), 2);
    EXPECT_EQ(add0.GetIOperands().front(), copyOutput0);
    EXPECT_EQ(add1.GetIOperands().front(), copyOutput1);
}

TEST_F(CommonOperationEliminateTest, RewriteResultTokenConsumersAfterMerge)
{
    ComputationalGraphBuilder G;
    BuildRedundantAbsGraph(G);

    Operation* producer1 = G.GetOp("ABS1");
    Operation* producer2 = G.GetOp("ABS2");
    Operation* consumer = G.GetOp("MUL");
    ASSERT_NE(producer1, nullptr);
    ASSERT_NE(producer2, nullptr);
    ASSERT_NE(consumer, nullptr);

    auto token1 = MakeToken("token1");
    auto token2 = MakeToken("token2");
    producer1->result_token_ = {token1};
    producer2->result_token_ = {token2};
    auto producer1Stmt = ToStmtPtr(producer1);
    auto producer2Stmt = ToStmtPtr(producer2);
    consumer->tokens_.push_back(token1);
    consumer->tokens_.push_back(token2);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    auto& dep = function->GetVarDependency();
    dep.AddProducer(token1, ToStmtPtr(producer1));
    dep.AddProducer(token2, ToStmtPtr(producer2));
    dep.AddConsumer(token1, ToStmtPtr(consumer));
    dep.AddConsumer(token2, ToStmtPtr(consumer));
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);

    Operation* keptProducer = FindKeptAbsProducer(*function);
    ASSERT_NE(keptProducer, nullptr);
    EXPECT_EQ(consumer->tokens_.size(), 1);
    EXPECT_EQ(consumer->tokens_[0], keptProducer->result_token_.front());
    auto removedToken = keptProducer->result_token_.front() == token1 ? token2 : token1;
    auto removedProducerStmt = removedToken == token1 ? producer1Stmt : producer2Stmt;
    EXPECT_EQ(dep.GetConsumers(keptProducer->result_token_.front()).count(ToStmtPtr(consumer)), 1);
    EXPECT_EQ(dep.GetConsumers(removedToken).count(ToStmtPtr(consumer)), 0);
    EXPECT_EQ(dep.GetProducers(removedToken).count(removedProducerStmt), 0);
    EXPECT_FALSE(dep.HasDependency(removedToken));
    EXPECT_EQ(function->Operations().size(), 2);
}

TEST_F(CommonOperationEliminateTest, MergeRawMagicBucketResultTokensByProducerIndex)
{
    auto function = std::make_shared<Function>(Program::GetInstance(),
                                               "TestMergeRawMagicBucketResultTokensByProducerIndex",
                                               "TestMergeRawMagicBucketResultTokensByProducerIndex", nullptr);
    ASSERT_NE(function, nullptr);

    const std::vector<int64_t> shape{8, 8};
    std::vector<LogicalTensorPtr> inputs{MakeDdrTensor(shape), MakeDdrTensor(shape)};
    std::vector<LogicalTensorPtr> addInputs{MakeDdrTensor(shape), MakeDdrTensor(shape)};
    auto raw0 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto raw1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::vector<LogicalTensorPtr> raw0Parts{MakeDdrTensor(shape, raw0), MakeDdrTensor(shape, raw0)};
    std::vector<LogicalTensorPtr> raw1Parts{MakeDdrTensor(shape, raw1), MakeDdrTensor(shape, raw1)};
    std::vector<OperationPtr> raw0Producers;
    std::vector<OperationPtr> raw1Producers;
    std::vector<Operation*> raw0Consumers;
    std::vector<Operation*> raw1Consumers;

    for (size_t i = 0; i < inputs.size(); ++i) {
        raw0Producers.emplace_back(
            PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {inputs[i]}, {raw0Parts[i]})
                .shared_from_this());
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        raw1Producers.emplace_back(
            PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {inputs[i]}, {raw1Parts[i]})
                .shared_from_this());
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& raw0Consumer = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {raw0Parts[i], addInputs[i]},
                                                              {MakeDdrTensor(shape)});
        auto& raw1Consumer = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {raw1Parts[i], addInputs[i]},
                                                              {MakeDdrTensor(shape)});
        raw0Consumer.SetAttr(OpAttributeKey::dontTouch, true);
        raw1Consumer.SetAttr(OpAttributeKey::dontTouch, true);
        raw0Consumers.emplace_back(&raw0Consumer);
        raw1Consumers.emplace_back(&raw1Consumer);
    }

    auto& dependency = function->GetVarDependency();
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto oldToken = MakeToken("bucket_old_token_" + std::to_string(i));
        auto newToken = MakeToken("bucket_new_token_" + std::to_string(i));
        raw0Producers[i]->result_token_ = {oldToken};
        raw1Producers[i]->result_token_ = {newToken};
        raw0Consumers[i]->tokens_.push_back(oldToken);
        raw1Consumers[i]->tokens_.push_back(newToken);
        dependency.AddProducer(oldToken, ToStmtPtr(raw0Producers[i].get()));
        dependency.AddProducer(newToken, ToStmtPtr(raw1Producers[i].get()));
        dependency.AddConsumer(oldToken, ToStmtPtr(raw0Consumers[i]));
        dependency.AddConsumer(newToken, ToStmtPtr(raw1Consumers[i]));
    }
    function->inCasts_ = {inputs[0], inputs[1], addInputs[0], addInputs[1]};

    CommonOperationEliminate COE;
    ASSERT_EQ(COE.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOperations(*function, Opcode::OP_ASSEMBLE), 2);

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto* keptProducer = !raw0Producers[i]->IsDeleted() ? raw0Producers[i].get() : raw1Producers[i].get();
        ASSERT_NE(keptProducer, nullptr);
        ASSERT_FALSE(keptProducer->result_token_.empty());
        ASSERT_EQ(raw0Consumers[i]->tokens_.size(), 1U);
        ASSERT_EQ(raw1Consumers[i]->tokens_.size(), 1U);
        EXPECT_EQ(raw0Consumers[i]->tokens_[0], keptProducer->result_token_.front());
        EXPECT_EQ(raw1Consumers[i]->tokens_[0], keptProducer->result_token_.front());
    }
}

TEST_F(CommonOperationEliminateTest, PreserveAssembleAsSideEffectOperation)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(BuildVecDupAssembleGraph(G, MemoryType::MEM_DEVICE_DDR));
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* assemble1 = G.GetOp("ASSEMBLE1");
    auto* assemble2 = G.GetOp("ASSEMBLE2");
    ASSERT_NE(assemble1, nullptr);
    ASSERT_NE(assemble2, nullptr);
    auto assembleToken = MakeToken("assemble_token");
    assemble1->result_token_ = {assembleToken};
    assemble2->tokens_.push_back(assembleToken);
    auto assemble1Stmt = ToStmtPtr(assemble1);
    auto assemble2Stmt = ToStmtPtr(assemble2);
    auto& dep = function->GetVarDependency();
    dep.AddProducer(assembleToken, assemble1Stmt);
    dep.AddConsumer(assembleToken, assemble2Stmt);

    CommonOperationEliminate COE;
    EXPECT_EQ(COE.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOperations(*function, Opcode::OP_ASSEMBLE), 2);
    EXPECT_EQ(assemble1->result_token_.front(), assembleToken);
    ASSERT_EQ(assemble2->tokens_.size(), 1U);
    EXPECT_EQ(assemble2->tokens_[0], assembleToken);
    EXPECT_EQ(dep.GetProducers(assembleToken).count(assemble1Stmt), 1);
    EXPECT_EQ(dep.GetConsumers(assembleToken).count(assemble2Stmt), 1);
}

TEST_F(CommonOperationEliminateTest, MergeInputTokensWithoutResultTokenRewrites)
{
    ComputationalGraphBuilder graph;
    BuildRedundantAbsGraph(graph);
    auto* producer1 = graph.GetOp("ABS1");
    auto* producer2 = graph.GetOp("ABS2");
    auto* consumer = graph.GetOp("MUL");
    auto* function = graph.GetFunction();
    ASSERT_NE(producer1, nullptr);
    ASSERT_NE(producer2, nullptr);
    ASSERT_NE(consumer, nullptr);
    ASSERT_NE(function, nullptr);

    auto inputToken1 = MakeToken("input_token1");
    auto inputToken2 = MakeToken("input_token2");
    producer1->tokens_ = {inputToken1};
    producer2->tokens_ = {inputToken2};
    consumer->tokens_ = {inputToken1};
    auto& dependency = function->GetVarDependency();
    dependency.AddConsumer(inputToken1, ToStmtPtr(producer1));
    dependency.AddConsumer(inputToken2, ToStmtPtr(producer2));
    dependency.AddConsumer(inputToken1, ToStmtPtr(consumer));

    CommonOperationEliminate coe;
    ASSERT_EQ(coe.Run(*function, "", "", 0), SUCCESS);

    auto* keptProducer = FindKeptAbsProducer(*function);
    ASSERT_NE(keptProducer, nullptr);
    EXPECT_TRUE(keptProducer->result_token_.empty());
    EXPECT_EQ(keptProducer->tokens_.size(), 2U);
    EXPECT_EQ(std::count(keptProducer->tokens_.begin(), keptProducer->tokens_.end(), inputToken1), 1);
    EXPECT_EQ(std::count(keptProducer->tokens_.begin(), keptProducer->tokens_.end(), inputToken2), 1);
    ASSERT_EQ(consumer->tokens_.size(), 1U);
    EXPECT_EQ(consumer->tokens_.front(), inputToken1);
    EXPECT_EQ(dependency.GetConsumers(inputToken1).size(), 2U);
    EXPECT_EQ(dependency.GetConsumers(inputToken2).size(), 1U);
    EXPECT_EQ(dependency.GetConsumers(inputToken1).count(ToStmtPtr(keptProducer)), 1U);
    EXPECT_EQ(dependency.GetConsumers(inputToken2).count(ToStmtPtr(keptProducer)), 1U);
    EXPECT_EQ(dependency.GetConsumers(inputToken1).count(ToStmtPtr(consumer)), 1U);
    EXPECT_EQ(function->Operations().size(), 2U);
}

TEST_F(CommonOperationEliminateTest, MergeProducerInputTokensAfterMerge)
{
    ComputationalGraphBuilder G;
    BuildRedundantAbsGraph(G);

    Operation* producer1 = G.GetOp("ABS1");
    Operation* producer2 = G.GetOp("ABS2");
    ASSERT_NE(producer1, nullptr);
    ASSERT_NE(producer2, nullptr);

    auto resultToken1 = MakeToken("result_token1");
    auto resultToken2 = MakeToken("result_token2");
    auto sharedInputToken = MakeToken("shared_input_token");
    auto inputToken1 = MakeToken("input_token1");
    auto inputToken2 = MakeToken("input_token2");
    producer1->result_token_ = {resultToken1};
    producer2->result_token_ = {resultToken2};
    producer1->tokens_.push_back(sharedInputToken);
    producer1->tokens_.push_back(inputToken1);
    producer2->tokens_.push_back(sharedInputToken);
    producer2->tokens_.push_back(inputToken2);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    auto& dep = function->GetVarDependency();
    dep.AddProducer(resultToken1, ToStmtPtr(producer1));
    dep.AddProducer(resultToken2, ToStmtPtr(producer2));
    dep.AddConsumer(sharedInputToken, ToStmtPtr(producer1));
    dep.AddConsumer(inputToken1, ToStmtPtr(producer1));
    dep.AddConsumer(sharedInputToken, ToStmtPtr(producer2));
    dep.AddConsumer(inputToken2, ToStmtPtr(producer2));
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);

    Operation* keptProducer = FindKeptAbsProducer(*function);
    ASSERT_NE(keptProducer, nullptr);
    auto keptProducerStmt = ToStmtPtr(keptProducer);
    EXPECT_EQ(std::count(keptProducer->tokens_.begin(), keptProducer->tokens_.end(), sharedInputToken), 1);
    EXPECT_EQ(std::count(keptProducer->tokens_.begin(), keptProducer->tokens_.end(), inputToken1), 1);
    EXPECT_EQ(std::count(keptProducer->tokens_.begin(), keptProducer->tokens_.end(), inputToken2), 1);
    EXPECT_EQ(dep.GetConsumers(sharedInputToken).count(keptProducerStmt), 1);
    EXPECT_EQ(dep.GetConsumers(inputToken1).count(keptProducerStmt), 1);
    EXPECT_EQ(dep.GetConsumers(inputToken2).count(keptProducerStmt), 1);
    EXPECT_EQ(keptProducer->tokens_.size(), 3);
    EXPECT_EQ(function->Operations().size(), 2);
}

TEST_F(CommonOperationEliminateTest, RejectProducerMergeWhenTokenShapeDiffers)
{
    ComputationalGraphBuilder G;
    BuildRedundantAbsGraph(G);

    Operation* producer1 = G.GetOp("ABS1");
    Operation* producer2 = G.GetOp("ABS2");
    ASSERT_NE(producer1, nullptr);
    ASSERT_NE(producer2, nullptr);
    auto token = MakeToken("only_one_result_token");
    producer1->result_token_ = {token};

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->GetVarDependency().AddProducer(token, ToStmtPtr(producer1));

    CommonOperationEliminate COE;
    ASSERT_EQ(COE.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOperations(*function, Opcode::OP_ABS), 2);
    EXPECT_EQ(function->Operations().size(), 3);
    EXPECT_EQ(producer1->result_token_.front(), token);
}

TEST_F(CommonOperationEliminateTest, RedirectRawMagicSiblingConsumersByProducerOutputPair)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestRedirectRawMagicSiblingConsumers",
                                               "TestRedirectRawMagicSiblingConsumers", nullptr);
    ASSERT_NE(function, nullptr);

    const std::vector<int64_t> shape{8, 8};
    std::vector<LogicalTensorPtr> inputs{MakeDdrTensor(shape), MakeDdrTensor(shape), MakeDdrTensor(shape)};
    std::vector<LogicalTensorPtr> addInputs{MakeDdrTensor(shape), MakeDdrTensor(shape), MakeDdrTensor(shape)};
    auto raw0 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto raw1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::vector<LogicalTensorPtr> raw0Parts{MakeDdrTensor(shape, raw0), MakeDdrTensor(shape, raw0),
                                            MakeDdrTensor(shape, raw0)};
    std::vector<LogicalTensorPtr> raw1Parts{MakeDdrTensor(shape, raw1), MakeDdrTensor(shape, raw1),
                                            MakeDdrTensor(shape, raw1)};

    for (size_t i = 0; i < inputs.size(); ++i) {
        PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {inputs[i]}, {raw0Parts[i]});
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {inputs[i]}, {raw1Parts[i]});
    }

    std::vector<Operation*> raw0Consumers;
    std::vector<Operation*> raw1Consumers;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& raw0Consumer = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {raw0Parts[i], addInputs[i]},
                                                              {MakeDdrTensor(shape)});
        raw0Consumer.SetAttr(OpAttributeKey::dontTouch, true);
        raw0Consumers.emplace_back(&raw0Consumer);
        auto& raw1Consumer = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {raw1Parts[i], addInputs[i]},
                                                              {MakeDdrTensor(shape)});
        raw1Consumer.SetAttr(OpAttributeKey::dontTouch, true);
        raw1Consumers.emplace_back(&raw1Consumer);
    }
    function->inCasts_ = {inputs[0], inputs[1], inputs[2], addInputs[0], addInputs[1], addInputs[2]};

    CommonOperationEliminate COE;
    ASSERT_EQ(COE.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOperations(*function, Opcode::OP_ASSEMBLE), 3);
    for (size_t i = 0; i < inputs.size(); ++i) {
        ASSERT_NE(raw0Consumers[i], nullptr);
        ASSERT_NE(raw1Consumers[i], nullptr);
        EXPECT_EQ(raw0Consumers[i]->GetIOperands()[0], raw1Consumers[i]->GetIOperands()[0]);
    }
}

TEST_F(CommonOperationEliminateTest, SkipRawMagicBucketWithDependOperand)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestSkipRawMagicBucketWithDependOperand",
                                               "TestSkipRawMagicBucketWithDependOperand", nullptr);
    ASSERT_NE(function, nullptr);

    const std::vector<int64_t> shape{8, 8};
    auto input = MakeDdrTensor(shape);
    auto sideInput0 = MakeDdrTensor(shape);
    auto sideInput1 = MakeDdrTensor(shape);
    auto raw0 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto raw1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto raw0Part = MakeDdrTensor(shape, raw0);
    auto raw1Part = MakeDdrTensor(shape, raw1);
    auto output0 = MakeDdrTensor(shape);
    auto output1 = MakeDdrTensor(shape);
    auto dependOutput0 = MakeDdrTensor(shape);
    auto dependOutput1 = MakeDdrTensor(shape);

    auto& assemble0 = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {input}, {raw0Part});
    auto& assemble1 = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {input}, {raw1Part});
    PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {raw0Part, sideInput0}, {output0});
    PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {raw1Part, sideInput1}, {output1});
    auto& depend0 = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {sideInput0, sideInput0},
                                                     {dependOutput0});
    auto& depend1 = PassOperationUtils::AddOperation(*function, Opcode::OP_ADD, {sideInput1, sideInput1},
                                                     {dependOutput1});
    depend0.AddDependOperand(raw0Part);
    depend1.AddDependOperand(raw1Part);
    raw0Part->AddDependOp(depend0);
    raw1Part->AddDependOp(depend1);
    assemble0.AddDependOperand(sideInput0);
    assemble1.AddDependOperand(sideInput1);
    sideInput0->AddDependOp(assemble0);
    sideInput1->AddDependOp(assemble1);
    function->inCasts_ = {input, sideInput0, sideInput1};
    function->outCasts_ = {output0, output1, dependOutput0, dependOutput1};

    CommonOperationEliminate COE;
    ASSERT_EQ(COE.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOperations(*function, Opcode::OP_ASSEMBLE), 2);
}

TEST_F(CommonOperationEliminateTest, MergeProducerDependOperandAfterMerge)
{
    ComputationalGraphBuilder G;
    BuildRedundantAbsGraph(G);
    auto* producer1 = G.GetOp("ABS1");
    auto* producer2 = G.GetOp("ABS2");
    ASSERT_NE(producer1, nullptr);
    ASSERT_NE(producer2, nullptr);

    auto dependTensor = producer1->GetIOperands().front();
    producer1->AddDependOperand(dependTensor);
    dependTensor->AddDependOp(*producer1);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate COE;
    ASSERT_EQ(COE.Run(*function, "", "", 0), SUCCESS);

    auto* keptProducer = FindKeptAbsProducer(*function);
    ASSERT_NE(keptProducer, nullptr);
    EXPECT_EQ(CountOperations(*function, Opcode::OP_ABS), 1);
    ASSERT_EQ(keptProducer->GetDependOperands().size(), 1U);
    EXPECT_EQ(keptProducer->GetDependOperands().front(), dependTensor);
    EXPECT_EQ(dependTensor->GetDependOps().count(keptProducer), 1);
}

TEST_F(CommonOperationEliminateTest, MergeDifferentProducerDependOperandsAfterMerge)
{
    ComputationalGraphBuilder G;
    BuildRedundantAbsGraph(G);
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, {16, 16}, "depend1"));
    ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, {16, 16}, "depend2"));

    auto* producer1 = G.GetOp("ABS1");
    auto* producer2 = G.GetOp("ABS2");
    auto depend1 = G.GetTensor("depend1");
    auto depend2 = G.GetTensor("depend2");
    ASSERT_NE(producer1, nullptr);
    ASSERT_NE(producer2, nullptr);
    ASSERT_NE(depend1, nullptr);
    ASSERT_NE(depend2, nullptr);

    producer1->AddDependOperand(depend1);
    producer2->AddDependOperand(depend2);
    depend1->AddDependOp(*producer1);
    depend2->AddDependOp(*producer2);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->inCasts_ = {G.GetTensor("t1"), depend1, depend2};

    CommonOperationEliminate COE;
    ASSERT_EQ(COE.Run(*function, "", "", 0), SUCCESS);

    auto* keptProducer = FindKeptAbsProducer(*function);
    ASSERT_NE(keptProducer, nullptr);
    ASSERT_EQ(keptProducer->GetDependOperands().size(), 2U);
    EXPECT_NE(std::find(keptProducer->GetDependOperands().begin(), keptProducer->GetDependOperands().end(), depend1),
              keptProducer->GetDependOperands().end());
    EXPECT_NE(std::find(keptProducer->GetDependOperands().begin(), keptProducer->GetDependOperands().end(), depend2),
              keptProducer->GetDependOperands().end());
    EXPECT_EQ(depend1->GetDependOps().count(keptProducer), 1);
    EXPECT_EQ(depend2->GetDependOps().count(keptProducer), 1);
    EXPECT_EQ(function->Operations().size(), 2);
}

TEST_F(CommonOperationEliminateTest, PreserveAssembleSsa)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestPreserveAssembleSsa",
                                               "TestPreserveAssembleSsa", nullptr);
    ASSERT_NE(function, nullptr);

    const std::vector<int64_t> shape{8, 8};
    auto source = MakeDdrTensor(shape);
    auto destinationRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    auto destination = MakeDdrTensor(shape, destinationRaw);
    auto result1 = MakeDdrTensor(shape, destinationRaw);
    auto result2 = MakeDdrTensor(shape, destinationRaw);

    auto& assemble1 = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE_SSA, {source, destination},
                                                       {result1});
    auto& assemble2 = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE_SSA, {source, destination},
                                                       {result2});
    assemble1.SetAssembleOpAttribute({0, 0});
    assemble2.SetAssembleOpAttribute({0, 0});
    function->inCasts_ = {source, destination};

    CommonOperationEliminate COE;
    ASSERT_EQ(COE.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOperations(*function, Opcode::OP_ASSEMBLE_SSA), 2);
}
} // namespace tile_fwk
} // namespace npu
