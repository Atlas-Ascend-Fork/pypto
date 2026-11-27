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
 * \file test_split_reshape_token.cpp
 * \brief Unit test for split_reshape token (WAW scenario) adaptation.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/error_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/pass_attr_defs.h"
#define private public
#define protected public
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"
#undef private
#undef protected
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"

using namespace npu::tile_fwk;

class SplitReshapeTokenTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
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

static ir::VarPtr MakeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetTokenType(), ir::Span::Unknown());
}

TEST_F(SplitReshapeTokenTest, NoTokenNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitReshapeTokenNoToken") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitReshapeTokenNoToken");
    ASSERT_NE(func, nullptr);

    SplitReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

TEST_F(SplitReshapeTokenTest, PassRunsWithTokenOnAssembleNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitReshapeTokenWithToken")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitReshapeTokenWithToken");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    for (auto* op : opList) {
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE && op->result_token_.empty()) {
            auto token = MakeToken("testReshapeToken");
            op->result_token_ = {token};
        }
    }

    SplitReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

TEST_F(SplitReshapeTokenTest, EmptyFunctionNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitReshapeTokenEmpty") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitReshapeTokenEmpty");
    ASSERT_NE(func, nullptr);

    SplitReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

void CheckAliasWriteIsNotBypassed(Opcode writeOpcode, bool separateVersion)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "WritableReshape", "WritableReshape", nullptr);
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, {32, 16});
    for (int64_t row : {0, 16}) {
        auto tile = builder.CreateTensorVar(DT_FP32, {16, 16});
        auto& assemble = PassOperationUtils::AddOperation(*func, writeOpcode, {tile}, {base});
        assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, std::vector<int64_t>{row, 0}));
    }
    auto alias = builder.CreateTensorVar(DT_FP32, {32, 1, 16});
    auto& reshape = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {base}, {alias});
    reshape.SetAttribute("op_attr_isInplace", true);
    auto writtenAlias = separateVersion ? builder.CreateTensorVar(alias->GetRawTensor(), {0, 0, 0}, {32, 1, 16}, {}) :
                                          alias;
    auto patch = builder.CreateTensorVar(DT_FP32, {8, 1, 16});
    auto& write = PassOperationUtils::AddOperation(*func, writeOpcode, {patch}, {writtenAlias});
    write.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, std::vector<int64_t>{8, 0, 0}));
    auto read = builder.CreateTensorVar(DT_FP32, {16, 1, 16});
    auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_SLICE, {writtenAlias}, {read});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0, 0}));

    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    ASSERT_EQ(pass.CollectCopyOut(*func), SUCCESS);
    ASSERT_EQ(pass.CheckCopyIn(*func), SUCCESS);
    EXPECT_FALSE(view.IsDeleted());
    EXPECT_EQ(view.GetIOperands().front(), writtenAlias);
    EXPECT_FALSE(write.IsDeleted());
}

TEST_F(SplitReshapeTokenTest, ReadingWritableAliasKeepsPartialWrite)
{
    CheckAliasWriteIsNotBypassed(Opcode::OP_CONTRACT, false);
    CheckAliasWriteIsNotBypassed(Opcode::OP_ASSEMBLE, false);
}

TEST_F(SplitReshapeTokenTest, ReadingNewAliasVersionKeepsPartialWrite)
{
    CheckAliasWriteIsNotBypassed(Opcode::OP_CONTRACT, true);
}

TEST_F(SplitReshapeTokenTest, ReadTensorWithNullProducerSkipsSplit)
{
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, {32, 16});
    auto alias = builder.CreateTensorVar(DT_FP32, {32, 1, 16});
    auto read = builder.CreateTensorVar(DT_FP32, {16, 1, 16});
    alias->GetProducers().insert(nullptr);
    ASSERT_EQ(alias->GetProducers().size(), 1);

    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    pass.reshapeSources_[alias->GetRawMagic()] = base;
    CheckOutputParam checkOutputParam;
    EXPECT_EQ(pass.CheckReshapeSkip(alias, read, checkOutputParam), WARNING);
}

namespace {
std::shared_ptr<LogicalTensor> MakeRank2TestTensor(int64_t rows, int64_t cols)
{
    return IRBuilder().CreateTensorVar(DT_FP32, Shape{rows, cols});
}

std::shared_ptr<Function> MakeDirectCallFunction(const std::string& name)
{
    return std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
}
} // namespace

// GetAssembleDynShape reports the new tensor-magic error when the assemble output was never registered
// in reshapeOffset_.
TEST_F(SplitReshapeTokenTest, GetAssembleDynShapeReportsMissingOutput)
{
    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    auto input = MakeRank2TestTensor(16, 16);
    auto output = MakeRank2TestTensor(16, 16);
    std::vector<SymbolicScalar> dynValidShape;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.GetAssembleDynShape(input, output, std::vector<int64_t>{0, 0}, dynValidShape), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Cannot find output from reshapeOffset, output tensor magic is"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// AddOperation fails through GetAssembleDynShape when an assembled entry's output has no reshapeOffset_ entry.
TEST_F(SplitReshapeTokenTest, AddOperationFailsWhenAssembleOutputNotRegistered)
{
    auto func = MakeDirectCallFunction("AddOperationAssembleMissing");
    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    auto input = MakeRank2TestTensor(16, 16);
    auto output = MakeRank2TestTensor(16, 16);
    pass.assembles_.push_back(
        AssembleOp{MemoryType::MEM_UB, std::vector<int64_t>{0, 0}, input, output, nullptr, Opcode::OP_ASSEMBLE});

    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.AddOperation(*func), FAILED);
    std::string errMsg;
    // First error is GetAssembleDynShape's own report, the second is AddOperation's wrapper report.
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Cannot find output from reshapeOffset"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(false);
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Get assemble dynamic shape for AddOperation failed"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// GetReshapeDynShape reports the new error when the reshape op has no viewOffset_ entry.
TEST_F(SplitReshapeTokenTest, GetReshapeDynShapeReportsMissingReshapeOp)
{
    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    auto input = MakeRank2TestTensor(16, 16);
    auto output = MakeRank2TestTensor(16, 16);
    auto reshapeOp = std::make_shared<ReshapeOp>(input, output, nullptr);
    std::vector<SymbolicScalar> dynValidShape;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.GetReshapeDynShape(reshapeOp, dynValidShape), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Cannot find reshapeop with sharedptr, reshape input tensor magic"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// AddOperation fails through GetReshapeDynShape when a reshapes_ entry never received a viewOffset_ slot.
TEST_F(SplitReshapeTokenTest, AddOperationFailsWhenReshapeOpNotRegistered)
{
    auto func = MakeDirectCallFunction("AddOperationReshapeMissing");
    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    auto input = MakeRank2TestTensor(16, 16);
    auto output = MakeRank2TestTensor(16, 16);
    pass.reshapes_[0UL] = std::make_shared<ReshapeOp>(input, output, nullptr);

    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.AddOperation(*func), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Cannot find reshapeop with sharedptr"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(false);
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Get reshape dynamic shape for AddOperation failed"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// UpdateForPerfectlyMatchWithAll reuses one newReshapeSource for every overlap; a second overlap whose
// assemble write offset was never recorded resolves to an empty offset and fails the axis check in
// AddAssembleOp for the already-registered output.
TEST_F(SplitReshapeTokenTest, UpdateForPerfectlyMatchWithAllFailsOnUnmappedSecondOverlap)
{
    auto func = MakeDirectCallFunction("UpdateForPerfectlyMatchWithAllSecondOverlap");
    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);

    auto reshapeSource = MakeRank2TestTensor(32, 16);
    auto overlap1 = MakeRank2TestTensor(8, 16);
    auto overlap2 = MakeRank2TestTensor(8, 16);
    auto input = MakeRank2TestTensor(8, 16);
    auto inputView = MakeRank2TestTensor(8, 16);
    auto output = MakeRank2TestTensor(8, 16);

    std::vector<int64_t> viewOffset{0, 0};
    auto& viewOp = PassOperationUtils::AddOperation(
        *func, Opcode::OP_SLICE, {input}, {output}, [&viewOffset](Operation& op) {
            op.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset, MemoryType::MEM_UNKNOWN));
        });

    // Only the first overlap has an assemble write offset recorded in mapOffset_.
    pass.mapOffset_[std::make_pair(overlap1->GetMagic(), reshapeSource->GetMagic())] = std::vector<int64_t>{0, 0};

    CalcOverlapPara para{};
    para.alignedShape = std::vector<int64_t>{32, 16};
    para.reshapeSource = reshapeSource;
    para.newInputViewTileOffset = std::vector<int64_t>{0, 0};
    para.newInputViewTileShape = std::vector<int64_t>{8, 16};
    para.overlaps = {overlap1, overlap2};
    para.newOverlaps = {overlap1, overlap2};
    para.input = input;
    para.inputView = inputView;
    para.output = output;
    ReshapeSourcePara sourcePara{};
    sourcePara.newReshapeSourceTileShape = std::vector<int64_t>{8, 16};
    sourcePara.newReshapeSourceTileOffset = std::vector<int64_t>{0, 0};

    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.UpdateForPerfectlyMatchWithAll(*func, viewOp, para, sourcePara), FAILED);
    std::string errMsg;
    // AddAssembleOp reports the axis mismatch first, then UpdateForPerfectlyMatchWithAll wraps it.
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Incorrect axis, toOffset is"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(false);
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("AddAssembleOp failed in UpdateForPerfectlyMatchWithAll"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

namespace {
// Shared fixture for the ProcessPerfectlyMatch / ProcessBeCovered direct-call cases: a real reshape op
// carries a rank-2 VALID_SHAPE while the crafted reshape output is rank-3, so UpdateDynShape must fail.
struct PerfectlyMatchDirectCallGraph {
    std::shared_ptr<Function> func;
    Operation* reshapeOp = nullptr;
    Operation* viewOp = nullptr;
    LogicalTensorPtr viewInput;
    LogicalTensorPtr viewOutput;
    LogicalTensorPtr overlap;
    LogicalTensorPtr reshapeSource;
    LogicalTensorPtr rank3ReshapeOutput;

    void Build(const std::string& name)
    {
        func = MakeDirectCallFunction(name);
        auto reshapeInput = MakeRank2TestTensor(16, 16);
        auto reshapeOutputBase = MakeRank2TestTensor(16, 16);
        viewInput = MakeRank2TestTensor(16, 16);
        viewOutput = MakeRank2TestTensor(16, 16);
        overlap = MakeRank2TestTensor(16, 16);
        reshapeSource = MakeRank2TestTensor(32, 16);
        rank3ReshapeOutput = IRBuilder().CreateTensorVar(DT_FP32, Shape{4, 4, 4});

        auto& reshape = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {reshapeInput},
                                                         {reshapeOutputBase});
        reshape.SetAttribute(OP_ATTR_VALID_SHAPE, std::vector<SymbolicScalar>{SymbolicScalar(16), SymbolicScalar(16)});
        reshapeOp = &reshape;

        std::vector<int64_t> zeroOffset{0, 0};
        auto& view = PassOperationUtils::AddOperation(
            *func, Opcode::OP_SLICE, {viewInput}, {viewOutput}, [&zeroOffset](Operation& op) {
                op.SetOpAttribute(std::make_shared<ViewOpAttribute>(zeroOffset, MemoryType::MEM_UNKNOWN));
            });
        viewOp = &view;
    }
};
} // namespace

// ProcessPerfectlyMatch fails on the isAddReshapeOp path when the origin reshape's VALID_SHAPE rank does
// not match the crafted reshape output rank.
TEST_F(SplitReshapeTokenTest, ProcessPerfectlyMatchRejectsMismatchedOriginValidShape)
{
    PerfectlyMatchDirectCallGraph graph;
    graph.Build("ProcessPerfectlyMatchMismatch");

    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    pass.reshapeOpPtrs_[graph.viewInput->GetMagic()] = graph.reshapeOp;
    // ObtainReshapeSource builds the new source with this offset, so it must match the overlap rank.
    pass.mapOffset_[std::make_pair(graph.overlap->GetMagic(), graph.reshapeSource->GetMagic())] = std::vector<int64_t>{
        0, 0};

    PerfectlyMatchPara para{};
    para.input = graph.viewInput;
    para.output = graph.viewOutput;
    para.overlap = graph.overlap;
    para.reshapeSource = graph.reshapeSource;
    para.reshapeOutput = graph.rank3ReshapeOutput;

    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.ProcessPerfectlyMatch(*graph.func, *graph.viewOp, para), FAILED);
    std::string errMsg;
    // UpdateDynShape reports the rank mismatch first, then ProcessPerfectlyMatch wraps it.
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("does not equal dynShape"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(false);
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Failed to UpdateDynShape or GroupReshapeOffset for isAddReshapeOp in "
                          "ProcessPerfectlyMatch"),
              std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}

// ProcessBeCovered fails on the isAddReshapeOp path when the origin reshape's VALID_SHAPE rank does not
// match the crafted reshape output rank.
TEST_F(SplitReshapeTokenTest, ProcessBeCoveredRejectsMismatchedOriginValidShape)
{
    PerfectlyMatchDirectCallGraph graph;
    graph.Build("ProcessBeCoveredMismatch");

    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    pass.reshapeOpPtrs_[graph.viewInput->GetMagic()] = graph.reshapeOp;
    pass.mapOffset_[std::make_pair(graph.overlap->GetMagic(), graph.reshapeSource->GetMagic())] = std::vector<int64_t>{
        0, 0};

    BeCoveredPara para{};
    para.overlap = graph.overlap;
    para.input = graph.viewInput;
    para.reshapeOutput = graph.rank3ReshapeOutput;
    para.reshapeSource = graph.reshapeSource;
    para.newOffset = std::vector<int64_t>{0, 0};

    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.ProcessBeCovered(*graph.func, *graph.viewOp, para), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("does not equal dynShape"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(false);
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("Failed to UpdateDynShape or GroupReshapeOffset for isAddReshapeOp in ProcessBeCovered"),
              std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}
