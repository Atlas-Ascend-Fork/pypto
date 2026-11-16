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
 * \file test_n_buffer_merge.cpp
 * \brief Unit test for n_buffer_merge pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <climits>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_partition/n_buffer_merge.h"
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {

class NBufferMergeTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}

    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    Status TestNBufferMergeWithDifferentVecBufferSetting(std::map<int64_t, int64_t> vecNBufferSetting);
};

namespace {
// Helper that builds a Function with 4 isomorphic subgraphs (subgraph ID 0..3),
// 2 inputs (incast1/incast2), 4 outputs (outcast1..outcast4), tensor shape {8, 16}.
// Each subgraph i contains: MUL(incast1, incast2) → EXP(tensor_i) → outcast(i+1).
// Intended for tests that verify multi-input/output NBufferMerge behavior
// (manualMulityInOutMerge and ByFuncHashOrder path).
std::shared_ptr<Function> BuildMultiInputOutputFunction()
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(function != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& mulOp1 = PassOperationUtils::AddOperation(*function, Opcode::OP_MUL, {incast1, incast2}, {tensor1});
    auto& expOp1 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {tensor1}, {outcast1});
    mulOp1.UpdateSubgraphID(0);
    expOp1.UpdateSubgraphID(0);

    auto& mulOp2 = PassOperationUtils::AddOperation(*function, Opcode::OP_MUL, {incast1, incast2}, {tensor2});
    auto& expOp2 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {tensor2}, {outcast2});
    mulOp2.UpdateSubgraphID(1);
    expOp2.UpdateSubgraphID(1);

    auto& mulOp3 = PassOperationUtils::AddOperation(*function, Opcode::OP_MUL, {incast1, incast2}, {tensor3});
    auto& expOp3 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {tensor3}, {outcast3});
    mulOp3.UpdateSubgraphID(2);
    expOp3.UpdateSubgraphID(2);

    auto& mulOp4 = PassOperationUtils::AddOperation(*function, Opcode::OP_MUL, {incast1, incast2}, {tensor4});
    auto& expOp4 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {tensor4}, {outcast4});
    mulOp4.UpdateSubgraphID(3);
    expOp4.UpdateSubgraphID(3);

    function->inCasts_.push_back(incast1);
    function->inCasts_.push_back(incast2);
    function->outCasts_.push_back(outcast1);
    function->outCasts_.push_back(outcast2);
    function->outCasts_.push_back(outcast3);
    function->outCasts_.push_back(outcast4);
    return function;
}
} // namespace

TEST_F(NBufferMergeTest, TestNBufferMerge)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copy_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    auto& copy_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    auto& copy_out1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);
    auto& copy_out2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    NBufferMerge nPass;
    Pass& nbufferPass = nPass;
    nbufferPass.PreCheck(*currFunctionPtr);
    nbufferPass.Run(*currFunctionPtr, "", "");
    nbufferPass.PostCheck(*currFunctionPtr);
}

TEST_F(NBufferMergeTest, TestMulityInputOutputMode3)
{
    auto currFunctionPtr = BuildMultiInputOutputFunction();

    // Call the pass
    NBufferMerge NBM;
    const int vecParallelNum = 2;
    currFunctionPtr->paramConfigs_.mgVecParallelLb = vecParallelNum;
    currFunctionPtr->paramConfigs_.vecNBufferSetting = {{-2, 0}};
    currFunctionPtr->SetTotalSubGraphCount(4);
    EXPECT_EQ(NBM.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetTotalSubGraphCount(), 2);
    EXPECT_EQ(currFunctionPtr->inCasts_.size(), 2);
    EXPECT_EQ(currFunctionPtr->outCasts_.size(), 4);
}

TEST_F(NBufferMergeTest, TestMulityInputOutputByFuncHashOrder)
{
    auto currFunctionPtr = BuildMultiInputOutputFunction();

    NBufferMerge NBM;
    const int vecParallelNum = 2;
    currFunctionPtr->paramConfigs_.mgVecParallelLb = vecParallelNum;
    const auto funcMagic = currFunctionPtr->GetFuncMagic();
    currFunctionPtr->paramConfigs_.vecNBufferSettingByFunc = {
        {"func" + std::to_string(funcMagic) + "_0", vecParallelNum}};
    currFunctionPtr->SetTotalSubGraphCount(4);
    EXPECT_EQ(NBM.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->inCasts_.size(), 2);
    EXPECT_EQ(currFunctionPtr->outCasts_.size(), 4);
    EXPECT_EQ(currFunctionPtr->GetTotalSubGraphCount(), 2);
}

TEST_F(NBufferMergeTest, TestMode4)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    const int result = 11;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 20;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}),
                  true);
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> iOperands{
            {"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{
            {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();
    function->paramConfigs_.vecNBufferSetting = {{-2, 1}, {-1, 4}, {1, 2}};
    function->paramConfigs_.mgVecParallelLb = vecParallelNum;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), result);
}

TEST_F(NBufferMergeTest, TestNoMergeMode)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int vecParallelNum = 6;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();
    function->paramConfigs_.vecNBufferSetting = {{-1, 1}};
    function->paramConfigs_.mgVecParallelLb = vecParallelNum;
    function->SetTotalSubGraphCount(1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
}

TEST_F(NBufferMergeTest, TestInvalidMode)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-2, 4}, {1, 2}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferSettingKeyMoreThanMaxValue_Tolerated)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-1, 4}, {100, 2}}), SUCCESS);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferSettingValueLessThanMinValue)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-1, 4}, {1, 0}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferSettingValueMoreThanMaxValue)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-1, 4}, {1, INT64_MAX}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndVecNBufferSettingDefaultValueLessThanMinValue)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{-1, 0}}), FAILED);
}

TEST_F(NBufferMergeTest, TestMode2AndNoDefaultValue)
{
    EXPECT_EQ(TestNBufferMergeWithDifferentVecBufferSetting({{0, 2}}), SUCCESS);
}

Status NBufferMergeTest::TestNBufferMergeWithDifferentVecBufferSetting(std::map<int64_t, int64_t> vecNBufferSetting)
{
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 3;
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    const int subGraphNum = 10;
    for (int i = 1; i < subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1" + strID, "tensor2" + strID, "tensor3" + strID}),
                  true);
        std::vector<std::vector<std::string>> iOperands{
            {"incast1"}, {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}};
        std::vector<std::vector<std::string>> oOperands{
            {"tensor1" + strID}, {"tensor2" + strID}, {"tensor3" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();
    function->paramConfigs_.vecNBufferSetting = vecNBufferSetting;
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum);
    NBufferMerge NBM;
    return NBM.RunOnFunction(*function);
}

Function* BuildFunctionWithSubgraphs(ComputationalGraphBuilder& G, const std::vector<int64_t>& tileShape,
                                     int subGraphNum)
{
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    for (int i = 1; i <= subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(
            G.AddTensors(DataType::DT_FP32, tileShape, {"tensor1_" + strID, "tensor2_" + strID, "tensor3_" + strID}),
            true);
        std::vector<std::vector<std::string>> iOperands{
            {"incast1"}, {"tensor1_" + strID}, {"tensor2_" + strID}, {"tensor3_" + strID}};
        std::vector<std::vector<std::string>> oOperands{
            {"tensor1_" + strID}, {"tensor2_" + strID}, {"tensor3_" + strID}, {"outcast"}};
        std::vector<std::string> opNames{"ABS_" + strID, "EXP_" + strID, "ADDS_" + strID, "ASSEMBLE_" + strID};
        std::vector<Opcode> opLists{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
        EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASSEMBLE_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    return G.GetFunction();
}

TEST_F(NBufferMergeTest, TestSemanticLabelSetting)
{
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 3;
    const int subGraphNum = 4;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, tileShape, subGraphNum);

    // Set semantic label for subgraph 1 and 2
    auto vecLabel = std::make_shared<SemanticLabel>("VecLabel", __FILE__, __LINE__);
    for (int i = 1; i <= 2; i++) {
        std::string strID = std::to_string(i);
        G.GetOp("ABS_" + strID)->SetSemanticLabel(vecLabel);
        G.GetOp("EXP_" + strID)->SetSemanticLabel(vecLabel);
    }

    // Default merge=4, but "VecLabel" override to 1 (no merge for labeled subgraphs)
    function->paramConfigs_.vecNBufferSetting = {{-1, 4}};
    function->paramConfigs_.vecNBufferSettingByLabel = {{"VecLabel", 1}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
}

// Unmatched semantic label is ignored (warned, not failed).
TEST_F(NBufferMergeTest, TestSemanticLabelSettingNotFound)
{
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 3;
    const int subGraphNum = 4;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, tileShape, subGraphNum);
    function->paramConfigs_.vecNBufferSetting = {{-1, 4}};
    function->paramConfigs_.vecNBufferSettingByLabel = {{"NoSuchLabel", 1}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
}

// Semantic-label merge count values must be in [1, INT_MAX]:
// 0 / negative / >INT_MAX values are rejected at compile time.
class NBufferMergeLabelMergeCountTest : public NBufferMergeTest, public testing::WithParamInterface<int64_t> {};

TEST_P(NBufferMergeLabelMergeCountTest, TestSemanticLabelInvalidMergeCount)
{
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 3;
    const int subGraphNum = 4;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, tileShape, subGraphNum);

    auto vecLabel = std::make_shared<SemanticLabel>("VecLabel", __FILE__, __LINE__);
    for (int i = 1; i <= 2; i++) {
        std::string strID = std::to_string(i);
        G.GetOp("ABS_" + strID)->SetSemanticLabel(vecLabel);
        G.GetOp("EXP_" + strID)->SetSemanticLabel(vecLabel);
    }

    function->paramConfigs_.vecNBufferSetting = {{-1, 4}};
    function->paramConfigs_.vecNBufferSettingByLabel = {{"VecLabel", GetParam()}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), FAILED);
}

INSTANTIATE_TEST_SUITE_P(SemanticLabelInvalidMergeCount, NBufferMergeLabelMergeCountTest,
                         testing::Values(0, static_cast<int64_t>(INT_MAX) + 1));

// ===== ByFunc Integration Tests =====
TEST_F(NBufferMergeTest, ByFuncMergeDefaultTwo)
{
    // Build 8 AIV subgraphs (same structure, same hash) + 1 COPY_IN subgraph = 9 total
    // ByFunc DEFAULT=2 → all AIV merge in groups of 2 → ceil(8/2)=4 AIV groups + 1 COPY_IN = 5
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    const int mgVecParallelLb = 48;
    const int subGraphNum = 8;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"incast0", "incast1", "outcast"}), true);
    EXPECT_EQ(G.AddOps({Opcode::OP_COPY_IN}, {{"incast0"}}, {{"incast1"}}, {"copy_in"}, true), true);
    G.GetOp("copy_in")->UpdateSubgraphID(0);
    for (int i = 1; i <= subGraphNum; i++) {
        std::string strID = std::to_string(i);
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"t1_" + strID, "t2_" + strID, "t3_" + strID}), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_ABS}, {{"incast1"}}, {{"t1_" + strID}}, {"ABS_" + strID}, true), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_EXP}, {{"t1_" + strID}}, {{"t2_" + strID}}, {"EXP_" + strID}, true), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_ADDS}, {{"t2_" + strID}}, {{"t3_" + strID}}, {"ADDS_" + strID}, true), true);
        EXPECT_EQ(G.AddOps({Opcode::OP_ASSEMBLE}, {{"t3_" + strID}}, {{"outcast"}}, {"ASM_" + strID}, true), true);
        G.GetOp("ABS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("EXP_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ADDS_" + strID)->UpdateSubgraphID(i);
        G.GetOp("ASM_" + strID)->UpdateSubgraphID(i);
    }
    EXPECT_EQ(G.SetInCast({"incast0"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast"}), true);
    Function* function = G.GetFunction();

    function->paramConfigs_.vecNBufferSettingByFunc = {{"DEFAULT", 2}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 5);
}

TEST_F(NBufferMergeTest, ByFuncMergeFuncSpecificOverride)
{
    // BuildFunctionWithSubgraphs: 1 COPY_IN (AIC, hashOrder=0) + 8 AIV subgraphs (hashOrder=1)
    // DEFAULT=4 → AIV ceil(8/4)=2 groups, override func{magic}_1=2 → ceil(8/2)=4 groups
    // Expected with override: 4 AIV groups + 1 COPY_IN = 5
    const int mgVecParallelLb = 48;
    const int subGraphNum = 8;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, {16, 16}, subGraphNum);

    int fm = function->GetFuncMagic();
    function->paramConfigs_.vecNBufferSettingByFunc = {{"DEFAULT", 4}, {"func" + std::to_string(fm) + "_1", 2}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 5);
}

TEST_F(NBufferMergeTest, ByFuncMergeDefaultOneNoMerge)
{
    // ByFunc DEFAULT:1 with size==1 → noMerge mode, graph untouched.
    const int mgVecParallelLb = 48;
    const int subGraphNum = 8;
    ComputationalGraphBuilder G;
    Function* function = BuildFunctionWithSubgraphs(G, {16, 16}, subGraphNum);

    function->paramConfigs_.vecNBufferSettingByFunc = {{"DEFAULT", 1}};
    function->paramConfigs_.mgVecParallelLb = mgVecParallelLb;
    function->SetTotalSubGraphCount(subGraphNum + 1);
    NBufferMerge NBM;
    EXPECT_EQ(NBM.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum + 1);
}

namespace {
// 构造四个交错写入等形状小块的子图。第一个读者读取两组写入，第二个默认只读取偶数组。
// sameReaders 为 true 时，两个读者都读取两组写入。
void BuildGatherReaderGroups(ComputationalGraphBuilder& graph, bool sameReaders = false,
                             Opcode producerOpcode = Opcode::OP_GATHER_IN_UB)
{
    ASSERT_TRUE(graph.AddTensor(DT_FP32, {64, 80}, "source"));
    ASSERT_TRUE(graph.AddTensor(DT_INT32, {1, 4}, "indices"));
    ASSERT_TRUE(graph.AddTensors(DT_FP32, {8, 80}, {"sharedA", "sharedB", "out1", "out2"}));
    for (int i = 0; i < 4; ++i) {
        const auto suffix = std::to_string(i);
        ASSERT_TRUE(graph.AddTensor(DT_FP32, {4, 80}, "tile" + suffix));
        const std::vector<std::string> inputs = producerOpcode == Opcode::OP_GATHER_IN_UB ?
                                                    std::vector<std::string>{"source", "indices"} :
                                                    std::vector<std::string>{"source"};
        ASSERT_TRUE(graph.AddOp(producerOpcode, inputs, {"tile" + suffix}, "produce" + suffix));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_CONTRACT, {"tile" + suffix}, {i % 2 == 0 ? "sharedA" : "sharedB"},
                                "write" + suffix));
        graph.GetOp("produce" + suffix)->UpdateSubgraphID(i);
        graph.GetOp("write" + suffix)->UpdateSubgraphID(i);
        graph.GetOp("write" + suffix)->SetAssembleOpAttribute({(i / 2) * 4, 0});
    }
    ASSERT_TRUE(graph.AddOp(Opcode::OP_A_MUL_B, {"sharedA", "sharedB"}, {"out1"}, "read1"));
    ASSERT_TRUE(graph.AddOp(
        Opcode::OP_A_MUL_B,
        sameReaders ? std::vector<std::string>{"sharedA", "sharedB"} : std::vector<std::string>{"sharedA", "sharedA"},
        {"out2"}, "read2"));
    graph.GetOp("read1")->UpdateSubgraphID(4);
    graph.GetOp("read2")->UpdateSubgraphID(5);
    ASSERT_TRUE(graph.SetInCast({"source", "indices"}));
    ASSERT_TRUE(graph.SetOutCast({"out1", "out2"}));
    graph.GetFunction()->SetTotalSubGraphCount(6);
    graph.GetFunction()->paramConfigs_.vecNBufferSetting = {{-1, 4}};
    graph.GetFunction()->paramConfigs_.mgVecParallelLb = 2;
}
} // namespace

TEST_F(NBufferMergeTest, GatherWritersWithDifferentReadersRemainSeparate)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    auto* function = graph.GetFunction();
    std::vector<nlohmann::json> writeAttrs;
    for (int i = 0; i < 4; ++i) {
        writeAttrs.push_back(graph.GetOp("write" + std::to_string(i))->GetOpAttribute()->DumpDynJson());
    }
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce2")->GetSubgraphID());
    EXPECT_EQ(graph.GetOp("produce1")->GetSubgraphID(), graph.GetOp("produce3")->GetSubgraphID());
    EXPECT_NE(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce1")->GetSubgraphID());
    EXPECT_EQ(function->GetTotalSubGraphCount(), 4);
    for (int i = 0; i < 4; ++i) {
        auto* write = graph.GetOp("write" + std::to_string(i));
        EXPECT_EQ(write->GetOpAttribute()->DumpDynJson(), writeAttrs[i]);
        EXPECT_EQ(write->GetOOperands()[0], graph.GetTensor(i % 2 == 0 ? "sharedA" : "sharedB"));
    }
    EXPECT_EQ(graph.GetOp("read1")->GetIOperands()[1], graph.GetTensor("sharedB"));
    EXPECT_EQ(graph.GetOp("read2")->GetIOperands()[0], graph.GetTensor("sharedA"));
}

TEST_F(NBufferMergeTest, GatherAutoModeRefinesDifferentReaders)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    graph.GetFunction()->paramConfigs_.vecNBufferSetting.clear();
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 4);
    EXPECT_NE(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce1")->GetSubgraphID());
}

TEST_F(NBufferMergeTest, GatherWritersWithSameReadersStillMerge)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph, true);
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 3);
    EXPECT_EQ(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce3")->GetSubgraphID());
}

TEST_F(NBufferMergeTest, GatherWriterMergeRespectsConfiguredLimit)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph, true);
    graph.GetFunction()->paramConfigs_.vecNBufferSetting = {{-1, 2}};
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 4);
}

TEST_F(NBufferMergeTest, GatherRegroupingRespectsConfiguredLimit)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    graph.GetFunction()->paramConfigs_.vecNBufferSetting = {{-1, 2}};
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 4);
    EXPECT_EQ(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce2")->GetSubgraphID());
    EXPECT_NE(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce1")->GetSubgraphID());
}

TEST_F(NBufferMergeTest, GatherPrivateReadersKeepCommonInputMerging)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    ASSERT_TRUE(graph.AddTensor(DT_FP32, {64, 80}, "commonInput"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_VIEW, {"source"}, {"commonInput"}, "commonView"));
    graph.GetOp("commonView")->UpdateSubgraphID(6);
    for (int i = 0; i < 4; ++i) {
        graph.GetOp("produce" + std::to_string(i))->UpdateInputOperand(0, graph.GetTensor("commonInput"));
    }
    graph.GetOp("read1")->UpdateInputOperand(1, graph.GetTensor("sharedA"));
    graph.GetOp("read2")->UpdateInputOperand(0, graph.GetTensor("sharedB"));
    graph.GetOp("read2")->UpdateInputOperand(1, graph.GetTensor("sharedB"));
    graph.GetFunction()->SetTotalSubGraphCount(7);
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce1")->GetSubgraphID());
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 4);
}

TEST_F(NBufferMergeTest, NonGatherWritersKeepExistingGrouping)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph, false, Opcode::OP_EXP);
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 3);
}

TEST_F(NBufferMergeTest, GatherMultiInputOutputModesRefineDifferentReaders)
{
    for (int manual : {0, 1}) {
        SCOPED_TRACE(manual);
        ComputationalGraphBuilder graph;
        BuildGatherReaderGroups(graph);
        graph.GetFunction()->paramConfigs_.vecNBufferSetting = manual == 0 ?
                                                                   std::map<int64_t, int64_t>{{-2, 0}} :
                                                                   std::map<int64_t, int64_t>{{-2, 1}, {-1, 4}};
        NBufferMerge pass;
        ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
        EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 4);
        EXPECT_NE(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce1")->GetSubgraphID());
    }
}

TEST_F(NBufferMergeTest, GatherMixedDependencyGroupRefinesEligibleWriters)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    graph.GetOp("write0")->AddDependOperand(graph.GetTensor("source"));
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 5);
    EXPECT_EQ(graph.GetOp("write0")->GetDependOperands()[0], graph.GetTensor("source"));
}

TEST_F(NBufferMergeTest, GatherMixedControlGroupRefinesEligibleWriters)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    graph.GetOp("write0")->AddOutCtrlOperation(*graph.GetOp("read2"));
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 5);
    EXPECT_EQ(graph.GetOp("write0")->GetOutCtrlOperations().count(graph.GetOp("read2")), 1);
}

TEST_F(NBufferMergeTest, GatherMixedTokenGroupRefinesEligibleWriters)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    auto token = std::make_shared<ir::Var>("writerToken", ir::GetTokenType(), ir::Span::Unknown());
    graph.GetOp("write0")->result_token_ = {token};
    NBufferMerge pass;
    ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
    EXPECT_EQ(graph.GetFunction()->GetTotalSubGraphCount(), 5);
    EXPECT_EQ(graph.GetOp("write0")->result_token_[0], token);
}

TEST_F(NBufferMergeTest, GatherOperationWithoutDependencyRecordIsRejected)
{
    ComputationalGraphBuilder graph;
    BuildGatherReaderGroups(graph);
    EXPECT_FALSE(NBufferMerge::IsGatherGroupOperation(*graph.GetOp("produce0"), 0, {}));
}

TEST_F(NBufferMergeTest, GatherPartialReaderConflictKeepsIndependentMerging)
{
    NBufferMerge pass;
    pass.outColor_ = {{4}, {4}, {5}, {6}, {4}};
    // Colors 0/1 share a minimum reader but have different complete reader sets.
    // Colors 2/3 have independent readers; color 4 is ineligible.
    EXPECT_EQ(pass.RefineGatherConsumerGroups({{0, 1, 2, 3, 4}}, {0, 1, 2, 3, -1}),
              (std::vector<std::vector<int>>{{2, 3}, {4}, {0}, {1}}));
}

TEST_F(NBufferMergeTest, GatherMultiInputOutputModesKeepSameReadersTogether)
{
    for (int manual : {0, 1}) {
        SCOPED_TRACE(manual);
        ComputationalGraphBuilder graph;
        BuildGatherReaderGroups(graph, true);
        graph.GetFunction()->paramConfigs_.vecNBufferSetting = manual == 0 ?
                                                                   std::map<int64_t, int64_t>{{-2, 0}} :
                                                                   std::map<int64_t, int64_t>{{-2, 1}, {-1, 4}};
        NBufferMerge pass;
        ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
        EXPECT_EQ(graph.GetOp("produce0")->GetSubgraphID(), graph.GetOp("produce1")->GetSubgraphID());
    }
}

} // namespace tile_fwk
} // namespace npu
