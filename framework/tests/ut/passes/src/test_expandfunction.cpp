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
 * \file test_expandfunction.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "ir/type.h"
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "passes/pass_utils/graph_utils.h"
#define private public
#include "interface/operation/operation.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "computational_graph_builder.h"
#include "passes/pass_check/inplace_conflict_checker.h"

namespace npu {
namespace tile_fwk {
static const size_t kSizeZero = 0UL;
static const size_t kSizeThree = 3UL;
static const size_t kSizeEight = 8UL;
static const size_t kSizeEleven = 11UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumFive = 5u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumForteen = 14u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumSevenTeen = 17u;
static const uint16_t kNumTwentyfive = 25u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

std::shared_ptr<AssembleOpAttribute> CreateAssembleOpAttr();

void MakeExpandGrpah(std::shared_ptr<Function>& currFunctionPtr, LogicalTensorPtr& outCast)
{
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& div_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_DIV, {inCast1, inCast2}, {ubTensor});
    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});
    auto op_attr = CreateAssembleOpAttr();
    assemble_op.SetOpAttribute(op_attr);
    div_op.tileShape_.SetVecTile(tile_shape);
    assemble_op.tileShape_.SetVecTile(tile_shape);

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);
}

std::shared_ptr<AssembleOpAttribute> CreateAssembleOpAttr()
{
    std::vector<int64_t> toOffset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> symbol = {CreateTestScalarVar("sym0"), CreateTestScalarVar("sym1")};
    return std::make_shared<AssembleOpAttribute>(toOffset, symbol);
}

namespace {

void SetViewOpAttribute(Operation& viewOp, const std::vector<int64_t>& offsets, const std::vector<int64_t>& shapes)
{
    auto dynOffsets = SymbolicScalar::FromConcrete(offsets);
    auto validShape = GetViewValidShape(viewOp.GetIOperands()[0]->GetDynValidShape(), offsets, dynOffsets, shapes);
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offsets, dynOffsets, validShape));
}

int CountInputSliceOpsWithSecondDimOffset(Function& function, const LogicalTensorPtr& input, int64_t offset)
{
    int count = 0;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        if (input == nullptr || op.GetIOperands().empty() || op.GetIOperands()[0] != input) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewAttr == nullptr || viewAttr->GetFromOffset().size() < 2) {
            continue;
        }
        if (viewAttr->GetFromOffset()[1] == offset) {
            count++;
        }
    }
    return count;
}

int CountSliceOpsOnInputWithShape(Function& function, const LogicalTensorPtr& input, const std::vector<int64_t>& shape)
{
    int count = 0;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        if (input == nullptr || op.GetIOperands().empty() || op.GetIOperands()[0] != input) {
            continue;
        }
        if (op.GetOOperands().empty() || op.GetOOperands()[0]->GetShape() != shape) {
            continue;
        }
        ++count;
    }
    return count;
}

bool HasNormalTokenNamed(const std::vector<ir::VarPtr>& tokens, const std::string& namePart)
{
    for (const auto& token : tokens) {
        if (token == nullptr || token->name_.find(namePart) == std::string::npos) {
            continue;
        }
        auto tokenType = std::dynamic_pointer_cast<const ir::TokenType>(token->GetType());
        if (tokenType != nullptr && tokenType->kind_ == ir::TokenKind::NORMAL) {
            return true;
        }
    }
    return false;
}

void ExpectSliceContractHaveDistinctRaws(Function& function)
{
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE && op.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        EXPECT_FALSE(op.GetIOperands().empty());
        EXPECT_FALSE(op.GetOOperands().empty());
        if (op.GetIOperands().empty() || op.GetOOperands().empty() || op.GetIOperands()[0] == nullptr ||
            op.GetOOperands()[0] == nullptr) {
            continue;
        }
        EXPECT_NE(op.GetIOperands()[0]->GetRawTensor(), op.GetOOperands()[0]->GetRawTensor())
            << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "] input/output share the same RawTensor";
    }
}

} // namespace

struct ScopeCfg {
    std::string op;
    int id;
    bool parMerge;
    bool crossMerge;
};

void RunScopeInfoTest(const std::vector<std::string>& tensors, size_t numInputs, const std::vector<Opcode>& opcodes,
                      const std::vector<std::vector<std::string>>& inputs,
                      const std::vector<std::vector<std::string>>& outputs, const std::vector<std::string>& opNames,
                      const Status status, const std::vector<ScopeCfg>& scopes = {})
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> shape{kNumExpSix, kNumExpSix};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape, tensors), true);
    EXPECT_EQ(G.AddOps(opcodes, inputs, outputs, opNames, true), true);
    for (const auto& s : scopes) {
        Operation::ScopeInfo info(s.id);
        info.allowParallelMerge = s.parMerge;
        info.allowCrossScopeMerge = s.crossMerge;
        auto op = G.GetOp(s.op);
        op->SetScopeInfo(info);
        if (op->GetCoreType() == CoreType::AIV)
            op->tileShape_.SetVecTile(kNumExpSix, kNumExpSix);
    }
    EXPECT_EQ(G.SetInCast({tensors.begin(), tensors.begin() + numInputs}), true);
    EXPECT_EQ(G.SetOutCast({tensors.begin() + numInputs, tensors.end()}), true);
    G.GetFunction()->SetGraphType(GraphType::TENSOR_GRAPH);
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);
    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.RunOnFunction(*G.GetFunction()), status);
}

class TestExpandFunctionPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPassOption(ENABLE_SLICE, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        // TearDown 会将平台重置为 DAV_UNKNOWN，而 DAV_UNKNOWN 下 GetOpSupportedInputDtypes 返回空表，
        // 同进程内后续用例录制 Exp/Div 等 op 时 dtype 检查必然失败。此处统一恢复有效平台，
        // 保证单进程顺序执行时每个用例的平台状态确定。
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    }
    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN); }
};

/*
TESTExpandFunctionNotTensorGrpah
inCast{8,16}->nop->outCast{8,16}

inCast{8,16}->nop->outCast{8,16}
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& nop_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_NOP, {inCast}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t nop_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_NOP) {
            EXPECT_EQ(nop_op.GetOpMagic(), op.GetOpMagic());
            ++nop_num;
        }
    }
    EXPECT_EQ(nop_num, kNumOne);
}

TEST_F(TestExpandFunctionPass, TestCVSeperate1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_TRUE(GraphUtils::IsCVMixPlatform());
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    EXPECT_FALSE(GraphUtils::IsCVMixPlatform());
}

TEST_F(TestExpandFunctionPass, TestCVSeperate2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);
    TileShape::Current().SetCubeTile({kNumExpFive, kNumExpFive}, {kNumExpFive, kNumExpFive}, {kNumExpFive, kNumExpFive},
                                     false);

    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto L1Tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto L1Tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& opAdd = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {out1});
    auto& opMatmul = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {L1Tensor1, L1Tensor2},
                                                      {out2});

    currFunctionPtr->inCasts_.push_back(ubTensor1);
    currFunctionPtr->inCasts_.push_back(ubTensor2);
    currFunctionPtr->inCasts_.push_back(L1Tensor1);
    currFunctionPtr->inCasts_.push_back(L1Tensor2);
    currFunctionPtr->outCasts_.push_back(out1);
    currFunctionPtr->outCasts_.push_back(out2);

    opAdd.tileShape_.SetVecTile(tile_shape);
    opAdd.SetScopeId(1);
    opMatmul.SetScopeId(1);
    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, FAILED);
}
/*
TESTExpandFunctionNOP
inCast{8,16}->nop->ubTensor2{8,16}->view->outCast{8,16}
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest2)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto op_attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{kNumZero, kNumZero});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_NOP, {inCast}, {ubTensor1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor1}, {outCast});

    std::shared_ptr<Operation> nop_op, view_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex) {
        auto op = currFunctionPtr->Operations().operations_[uIndex];
        if (op->GetOpcode() == Opcode::OP_NOP)
            nop_op = op;
        else if (op->GetOpcode() == Opcode::OP_VIEW)
            view_op = op;
    }

    view_op->SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    uint32_t nop_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_NE(view_op->GetOpMagic(), op.GetOpMagic());
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contract_num;
        } else if (op.GetOpcode() == Opcode::OP_NOP) {
            EXPECT_NE(nop_op->GetOpMagic(), op.GetOpMagic());
            ++nop_num;
        }
    }
    EXPECT_EQ(slice_num, kNumOne);
    EXPECT_EQ(contract_num, kNumOne);
    EXPECT_EQ(nop_num, kNumOne);
    ExpectSliceContractHaveDistinctRaws(*currFunctionPtr);
}

/*
TESTExpandFunctionAssemble
inCast{64,64}->assemble->outCast{64,64}
assemble tile-expands to 1 slice + 1 contract (single tile fallback)
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest3)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto op_attr = CreateAssembleOpAttr();
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {outCast});

    std::shared_ptr<Operation> assemble_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex) {
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_op = currFunctionPtr->Operations().operations_[uIndex];
        }
    }

    assemble_op->SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    uint32_t assemble_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contract_num;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(slice_num, kNumZero);
    EXPECT_EQ(contract_num, kNumZero);
    EXPECT_EQ(assemble_num, kNumOne);
}

/*
TESTExpandFunctionAssemble
inCast{64,64}->assemble->midTensor{64,64}->reshape->outCast{16,256}
assemble 输出仅被一个 reshape 消费且 reshape 输出是 outcast（assemble 无 token）时，
assemble 不展开，保持视图语义；reshape 正常展开为 TILE_RESHAPE
*/
TEST_F(TestExpandFunctionPass, AssembleReshapeToOutcastShouldNotExpand)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleReshapeOutcast",
                                                      "TestAssembleReshapeOutcast", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> outShape = {kNumExpFour, kNumExpSeven * kNumTwo};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {midTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {midTensor}, {outCast});
    assembleOp.SetOpAttribute(CreateAssembleOpAttr());
    assembleOp.tileShape_.SetVecTile(shape);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t sliceNum = kNumZero;
    uint32_t contractNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    uint32_t reshapeNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++sliceNum;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contractNum;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshapeNum;
        }
    }
    EXPECT_EQ(assembleNum, kNumOne);
    EXPECT_EQ(sliceNum, kNumZero);
    EXPECT_EQ(contractNum, kNumZero);
    EXPECT_EQ(reshapeNum, kNumOne);
}

/*
TESTExpandFunctionAssemble
inCast{64,64}->assemble->midTensor{64,64}->reshape->outCast{16,256}
assemble 携带 token（WAW/WAR）时，即使 assemble+reshape 接 outcast 也必须展开，
将 semantic token 传播为 contract 上的 NORMAL token
*/
TEST_F(TestExpandFunctionPass, AssembleReshapeToOutcastWithTokenShouldExpand)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleReshapeOutcastToken",
                                                      "TestAssembleReshapeOutcastToken", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> outShape = {kNumExpFour, kNumExpSeven * kNumTwo};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {midTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {midTensor}, {outCast});
    assembleOp.SetOpAttribute(CreateAssembleOpAttr());
    assembleOp.tileShape_.SetVecTile(shape);
    auto writeSemantic = IRContext::Get().MakeSemanticToken("mid_w", ir::TokenKind::WRITE, ir::Span::Unknown());
    assembleOp.result_token_ = {writeSemantic};

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t sliceNum = kNumZero;
    uint32_t contractNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    bool contractHasWrite = false;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++sliceNum;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contractNum;
            contractHasWrite = contractHasWrite || HasNormalTokenNamed(op.result_token_, "mid_w");
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(assembleNum, kNumZero);
    EXPECT_EQ(sliceNum, kNumOne);
    EXPECT_EQ(contractNum, kNumOne);
    EXPECT_TRUE(contractHasWrite);
    ExpectSliceContractHaveDistinctRaws(*currFunctionPtr);
}

/*
TESTExpandFunctionAssemble
inCast{64,64}->assemble->midTensor{64,64}->reshape->reshapeOut{16,256}->nop->outCast{16,256}
reshape 输出不是 outcast（后接 nop）时，assemble 正常展开为 slice + contract
*/
TEST_F(TestExpandFunctionPass, AssembleReshapeToNonOutcastShouldExpand)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleReshapeNonOutcast",
                                                      "TestAssembleReshapeNonOutcast", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> outShape = {kNumExpFour, kNumExpSeven * kNumTwo};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshapeOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {midTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {midTensor}, {reshapeOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_NOP, {reshapeOut}, {outCast});
    assembleOp.SetOpAttribute(CreateAssembleOpAttr());
    assembleOp.tileShape_.SetVecTile(shape);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t sliceNum = kNumZero;
    uint32_t contractNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++sliceNum;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contractNum;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(assembleNum, kNumZero);
    EXPECT_EQ(sliceNum, kNumOne);
    EXPECT_EQ(contractNum, kNumOne);
    ExpectSliceContractHaveDistinctRaws(*currFunctionPtr);
}

/*
TESTExpandFunctionAssemble
inCast{64,64}->assemble->midTensor{64,64}->reshape(deleted)->reshapeOut{16,256}
                      ->reshape(alive)->outCast2{16,256}
consumers 残留已删除的 RESHAPE 时不应误判为"唯一 RESHAPE 消费者"，
assemble 正常展开为 slice + contract
*/
TEST_F(TestExpandFunctionPass, AssembleReshapeToOutcastWithDeletedReshapeShouldExpand)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleReshapeDeleted",
                                                      "TestAssembleReshapeDeleted", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> outShape = {kNumExpFour, kNumExpSeven * kNumTwo};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto deadOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outShape, CreateTestConstIntVector(outShape));

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {midTensor});
    auto& deadReshape = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {midTensor}, {deadOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {midTensor}, {outCast});
    assembleOp.SetOpAttribute(CreateAssembleOpAttr());
    assembleOp.tileShape_.SetVecTile(shape);
    deadReshape.SetAsDeleted();

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t sliceNum = kNumZero;
    uint32_t contractNum = kNumZero;
    uint32_t assembleNum = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++sliceNum;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contractNum;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleNum;
        }
    }
    EXPECT_EQ(assembleNum, kNumZero);
    EXPECT_EQ(sliceNum, kNumOne);
    EXPECT_EQ(contractNum, kNumOne);
    ExpectSliceContractHaveDistinctRaws(*currFunctionPtr);
}

/*
TESTExpandFunctionAssemble
inCast1{64,64}->div->ubTensor{64,64}->assemble->outCast{64,64}
inCast2{64,64}->
inCast1{64,64}->view*4->div->ubTensor{64,64}->assemble(*4)->outCast{64,64}
inCast2{64,64}->view*4->
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest4)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);
    LogicalTensorPtr outCast;
    MakeExpandGrpah(currFunctionPtr, outCast);

    ExpandFunction expandfunctionpass;
    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest4_before0.json");
    EXPECT_EQ(expandfunctionpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest4_after0.json");
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t div_num = kNumZero;
    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    uint32_t assemble_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_DIV) {
            EXPECT_EQ(op.GetInputOperand(kSizeZero)->shape, tile_shape);
            EXPECT_EQ(op.GetOutputOperand(kSizeZero)->shape, tile_shape);
            EXPECT_NE(op.GetOutputOperand(kSizeZero), outCast);
            ++div_num;
        } else if (op.GetOpcode() == Opcode::OP_SLICE) {
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ++contract_num;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(div_num, kNumFour);
    EXPECT_EQ(slice_num, kNumEight);
    EXPECT_EQ(contract_num, kNumFour);
    EXPECT_EQ(assemble_num, kNumOne);
}

/*
TESTExpandFunctionAssemble
inCast{32,128}->reshape->ubTensor{64,64}->assemble->outCast{32,128}
assemble with reshape producer is view-like and should remain assemble after ExpandFunction.
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest5)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph: reshape -> assemble
    std::vector<int64_t> shape1 = {kNumExpFive, kNumExpSeven};
    std::vector<int64_t> shape2 = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumExpSeven};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));

    auto op_attr = CreateAssembleOpAttr();
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});

    std::shared_ptr<Operation> reshape_op;
    std::shared_ptr<Operation> assemble_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex) {
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_op = currFunctionPtr->Operations().operations_[uIndex];
        }
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_op = currFunctionPtr->Operations().operations_[uIndex];
        }
    }

    assemble_op->SetOpAttribute(op_attr);
    reshape_op->tileShape_.SetVecTile({kNumExpFive, kNumExpFive});
    assemble_op->tileShape_.SetVecTile({kNumExpFive, kNumExpFive});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;

    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest5_before1.json");
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/TestExpandFunctionPass_ExpandFunctionUTest5_after1.json");
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t slice_num = kNumZero;
    uint32_t contract_num = kNumZero;
    uint32_t reshape_num = kNumZero;
    uint32_t assemble_num = kNumZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE) {
            EXPECT_NE(op.GetOpMagic(), assemble_op->GetOpMagic());
            ++slice_num;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            EXPECT_NE(op.GetOpMagic(), assemble_op->GetOpMagic());
            auto attr = op.GetOpAttribute();
            EXPECT_NE(attr, nullptr);
            ++contract_num;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assemble_num;
        }
    }
    EXPECT_EQ(slice_num, kNumZero);
    EXPECT_EQ(contract_num, kNumZero);
    EXPECT_EQ(reshape_num, kNumOne);
    EXPECT_EQ(assemble_num, kNumOne);
}

/*
{64, 64} -> exp -> {64, 64}
{64, 64} -> (view) - > exp -> (assemble) - > {64, 64}
{64, 64} -> (view) -> (view {32, 64}) - > exp -> (assemble {32, 64}) -> (assemble) - > {64, 64}
                   -> (view {32, 64}) - > exp -> (assemble {32, 64})
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionSTest1)
{
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpSix};
    PassManager& passManager = PassManager::Instance();
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    TileShape::Current().SetVecTile(tile_shape);
    config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
    FUNCTION("STCase1") { output = Exp(input); }
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kSizeThree);
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
                                                                   {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                                               });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int exp_num = kNumZero;
    int slice_num = kNumZero;
    int contract_num = kNumZero;
    int view_num = kNumZero;
    int assemble_num = kNumZero;
    for (const auto& op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_EXP) {
            exp_num++;
            EXPECT_EQ(op.GetInputOperand(0)->shape, tile_shape);
            EXPECT_EQ(op.GetOutputOperand(0)->shape, tile_shape);
        } else if (op.GetOpcode() == Opcode::OP_SLICE) {
            slice_num++;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            contract_num++;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
            EXPECT_TRUE(op.HasAttribute(OpAttributeKey::isGlobalInput));
            EXPECT_TRUE(op.GetBoolAttribute(OpAttributeKey::isGlobalInput));
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_num++;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(assemble_num, kNumOne);
    EXPECT_EQ(exp_num, kNumTwo);
    EXPECT_GT(slice_num, kNumZero);
    EXPECT_EQ(slice_num, contract_num);
}

/*
{64, 64} -> exp -> view -> reciprocal
                        -> sqrt -> reshape
{64, 64} -> view -> exp -> view          -> sqrt         -> reshape        ->assemble(end)
                                                         -> assemble(end)
                                         -> reciprocal   -> assemble(end)
                                         -> assemble(end)
                        -> assemble(end)
view -> view(*4) -> exp(*4) -> assemble(*4) ->view  -> view(*4+4)   -> sqrt(*4)         -> assemble(*4)     -> reshape
-> assemble(end)
                                                                                        -> assemble(*4)     ->
assemble(*4) -> assemble(end)
                                            ->assemble(end)         -> reciprocal(*4)   -> assemble(*4)     ->
assemble(end)
*/
void ConstructGraphST2()
{
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> view_shape = {kNumExpSeven, kNumExpFive};
    std::vector<int64_t> reshape_shape = {kNumExpFive, kNumExpSeven};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, view_shape, "view");
    Tensor output1(DT_FP32, view_shape, "output");
    Tensor sqrt(DT_FP32, view_shape, "sqrt");
    Tensor output2(DT_FP32, reshape_shape, "sqrt");

    config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
    FUNCTION("STCase2")
    {
        TileShape::Current().SetVecTile(tile_shape);
        exp = Exp(input);
        view = View(exp, view_shape, {kNumZero, kNumZero});
        output1 = Reciprocal(view);
        sqrt = Sqrt(view);
        output2 = Reshape(sqrt, reshape_shape);
    }
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
}

TEST_F(TestExpandFunctionPass, ExpandFunctionSTest2)
{
    PassManager& passManager = PassManager::Instance();
    ConstructGraphST2();
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
                                                                   {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                                               });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int exp_num = kNumZero;
    int sqrt_num = kNumZero;
    int reshape_num = kNumZero;
    int reciprocal_num = kNumZero;
    int slice_num = kNumZero;
    int contract_num = kNumZero;
    int view_num = kNumZero;
    int assemble_num = kNumZero;
    for (const auto& op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_EXP) {
            exp_num++;
        } else if (op.GetOpcode() == Opcode::OP_SLICE) {
            slice_num++;
        } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            contract_num++;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
            EXPECT_TRUE(op.HasAttribute(OpAttributeKey::isGlobalInput));
            EXPECT_TRUE(op.GetBoolAttribute(OpAttributeKey::isGlobalInput));
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_num++;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_num++;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrt_num++;
        } else if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            reciprocal_num++;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(assemble_num, kNumFive);
    EXPECT_GT(slice_num, kNumZero);
    EXPECT_EQ(slice_num, contract_num);
    EXPECT_EQ(exp_num, kNumFour);
    EXPECT_EQ(reshape_num, kNumOne);
    EXPECT_EQ(sqrt_num, kNumFour);
    EXPECT_EQ(reciprocal_num, kNumFour);
}

/*
TESTExpandFunctionLoop
inCast{64,64}->assemble->view->outCast{64,64}
             <-assemble<-
loop will be detected
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest6)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto op_attr = CreateAssembleOpAttr();
    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {outCast});
    assemble_op.SetOpAttribute(op_attr);

    auto& assemble_op_loop = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {outCast},
                                                              {inCast});
    assemble_op_loop.SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.DefaultEnabledPreCheck(*currFunctionPtr), FAILED);

    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);
    EXPECT_EQ(expandfunctionpass.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(TestExpandFunctionPass, DisableCombineAxisOnA5)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction",
                                                      "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    currFunctionPtr->paramConfigs_.combineAxis = true;
    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->paramConfigs_.combineAxis, true);
}

TEST_F(TestExpandFunctionPass, TestScopeIdMinusOneWithMergeFlag)
{
    RunScopeInfoTest({"in1", "in2", "out1"}, 2, {Opcode::OP_ADD}, {{"in1", "in2"}}, {{"out1"}}, {"add1"}, FAILED,
                     {{"add1", -1, true, false}});
}

TEST_F(TestExpandFunctionPass, TestConflictingScopeInfoSettings)
{
    RunScopeInfoTest({"in1", "in2", "in3", "out1", "out2"}, 3, {Opcode::OP_ADD, Opcode::OP_ADD},
                     {{"in1", "in2"}, {"in2", "in3"}}, {{"out1"}, {"out2"}}, {"add1", "add2"}, FAILED,
                     {{"add1", 1, true, false}, {"add2", 1, false, true}});
}

TEST_F(TestExpandFunctionPass, TestPassScopeInfoSettingsVerify)
{
    RunScopeInfoTest({"in1", "in2", "in3", "out1", "out2"}, 3, {Opcode::OP_ADD, Opcode::OP_ADD},
                     {{"in1", "in2"}, {"in2", "in3"}}, {{"out1"}, {"out2"}}, {"add1", "add2"}, SUCCESS,
                     {{"add1", 1, true, false}, {"add2", 1, true, false}});
}

TEST_F(TestExpandFunctionPass, PreCheckForDisorderIndexOutcast)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape,
                           {"src", "index1", "dst", "index2", "result1", "result2", "tensor1", "outcast1", "outcast2"}),
              true);
    std::vector<Opcode> opLists{Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, Opcode::OP_ASSEMBLE,
                                Opcode::OP_ADDS, Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> iOperands{
        {"src", "index1", "dst"}, {"src", "index2", "dst"}, {"result1"}, {"result2"}, {"tensor1"}};
    std::vector<std::vector<std::string>> oOperands{{"result1"}, {"result2"}, {"outcast1"}, {"tensor1"}, {"outcast2"}};
    std::vector<std::string> opNames{"OP_INDEX_OUTCAST_1", "OP_INDEX_OUTCAST_2", "OP_ASSEMBLE_1", "OP_ADDS_1",
                                     "OP_ASSEMBLE_2"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"src", "index1", "dst", "index2"}), true);
    EXPECT_EQ(G.SetOutCast({"outcast1", "outcast2"}), true);

    Function* function = G.GetFunction();

    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.PreRun(*function), SUCCESS);
}

/*
    Tensor is used by both OP_VIEW and another operation (conflict scenario)
    tensor -> view -> ...
    tensor -> add -> ...
    This should fail CheckInplaceOperationConflict because tensor serves both view and other operations
*/
TEST_F(TestExpandFunctionPass, PreCheckForViewConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape,
                           {"tensor", "view_output", "add_output1", "add_output2", "other_input"}),
              true);

    std::vector<Opcode> opLists{Opcode::OP_VIEW, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"tensor", "other_input"}};
    std::vector<std::vector<std::string>> oOperands{{"view_output"}, {"add_output1"}};
    std::vector<std::string> opNames{"OP_VIEW_1", "OP_ADD_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor", "other_input"}), true);
    EXPECT_EQ(G.SetOutCast({"view_output", "add_output1"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), FAILED);
}

/*
    Tensor is used by both OP_RESHAPE and another operation (conflict scenario)
    tensor -> reshape -> ...
    tensor -> mul -> ...
    This should fail CheckInplaceOperationConflict because tensor serves both reshape and other operations
*/
TEST_F(TestExpandFunctionPass, PreCheckForReshapeConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor", "reshape_output", "mul_output", "other_input"}),
              true);

    std::vector<Opcode> opLists{Opcode::OP_RESHAPE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"tensor", "other_input"}};
    std::vector<std::vector<std::string>> oOperands{{"reshape_output"}, {"mul_output"}};
    std::vector<std::string> opNames{"OP_RESHAPE_1", "OP_MUL_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor", "other_input"}), true);
    EXPECT_EQ(G.SetOutCast({"reshape_output", "mul_output"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), FAILED);
}

/*
    Tensor is used only by OP_VIEW (no conflict scenario)
    tensor -> view -> adds
    This should succeed because tensor only serves view operation (tensor has only one consumer)
*/
TEST_F(TestExpandFunctionPass, PreCheckForViewNoConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor", "view_output", "final_output"}), true);

    std::vector<Opcode> opLists{Opcode::OP_VIEW, Opcode::OP_ADDS};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"view_output"}};
    std::vector<std::vector<std::string>> oOperands{{"view_output"}, {"final_output"}};
    std::vector<std::string> opNames{"OP_VIEW_1", "OP_ADDS_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor"}), true);
    EXPECT_EQ(G.SetOutCast({"final_output"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), SUCCESS);
}

/*
    Tensor is used only by OP_RESHAPE (no conflict scenario)
    tensor -> reshape -> exp
    This should succeed because tensor only serves reshape operation (tensor has only one consumer)
*/
TEST_F(TestExpandFunctionPass, PreCheckForReshapeNoConflict)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"tensor", "reshape_output", "final_output"}), true);

    std::vector<Opcode> opLists{Opcode::OP_RESHAPE, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> iOperands{{"tensor"}, {"reshape_output"}};
    std::vector<std::vector<std::string>> oOperands{{"reshape_output"}, {"final_output"}};
    std::vector<std::string> opNames{"OP_RESHAPE_1", "OP_EXP_1"};
    EXPECT_EQ(G.AddOps(opLists, iOperands, oOperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"tensor"}), true);
    EXPECT_EQ(G.SetOutCast({"final_output"}), true);

    Function* function = G.GetFunction();

    InplaceConflictChecker inplaceConflictChecker;
    EXPECT_EQ(inplaceConflictChecker.CheckInplaceOperationConflict(*function), SUCCESS);
}

/*
 * VIEW/ASSEMBLE whose tiled shape contains dynamic dim (-1) must not tile-expand;
 * otherwise TiledView/TiledAssemble loop over shape[dim]==-1 emits zero ops and breaks the graph.
 */
TEST_F(TestExpandFunctionPass, SkipExpandViewAndAssembleWithDynamicShape)
{
    std::vector<int64_t> dynShape = {-1, kNumExpSix};
    std::vector<int64_t> tileShape = {kNumExpFive, kNumExpFive};

    // ----- view: output shape has -1 -> keep OP_VIEW -----
    {
        auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSkipExpandViewDyn",
                                                          "TestSkipExpandViewDyn", nullptr);
        auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape, CreateTestConstIntVector(dynShape));
        auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape,
                                                                  CreateTestConstIntVector(dynShape));
        PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inCast}, {outCast});

        std::shared_ptr<Operation> viewOp;
        for (auto& opPtr : currFunctionPtr->Operations().operations_) {
            if (opPtr->GetOpcode() == Opcode::OP_VIEW) {
                viewOp = opPtr;
            }
        }
        ASSERT_NE(viewOp, nullptr);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{kNumZero, kNumZero}));
        viewOp->tileShape_.SetVecTile(tileShape);

        currFunctionPtr->inCasts_.push_back(inCast);
        currFunctionPtr->outCasts_.push_back(outCast);
        currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

        ExpandFunction expandPass;
        EXPECT_EQ(expandPass.RunOnFunction(*currFunctionPtr), SUCCESS);

        uint32_t viewNum = kNumZero;
        uint32_t sliceNum = kNumZero;
        uint32_t contractNum = kNumZero;
        for (auto& op : currFunctionPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                ++viewNum;
            } else if (op.GetOpcode() == Opcode::OP_SLICE) {
                ++sliceNum;
            } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
                ++contractNum;
            }
        }
        EXPECT_EQ(viewNum, kNumOne);
        EXPECT_EQ(sliceNum, kNumZero);
        EXPECT_EQ(contractNum, kNumZero);
    }

    // ----- assemble: input shape has -1 -> keep OP_ASSEMBLE -----
    {
        auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSkipExpandAssembleDyn",
                                                          "TestSkipExpandAssembleDyn", nullptr);
        auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape, CreateTestConstIntVector(dynShape));
        auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, dynShape,
                                                                  CreateTestConstIntVector(dynShape));
        PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {outCast});

        std::shared_ptr<Operation> assembleOp;
        for (auto& opPtr : currFunctionPtr->Operations().operations_) {
            if (opPtr->GetOpcode() == Opcode::OP_ASSEMBLE) {
                assembleOp = opPtr;
            }
        }
        ASSERT_NE(assembleOp, nullptr);
        assembleOp->SetOpAttribute(CreateAssembleOpAttr());
        assembleOp->tileShape_.SetVecTile(tileShape);

        currFunctionPtr->inCasts_.push_back(inCast);
        currFunctionPtr->outCasts_.push_back(outCast);
        currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

        ExpandFunction expandPass;
        EXPECT_EQ(expandPass.RunOnFunction(*currFunctionPtr), SUCCESS);

        uint32_t assembleNum = kNumZero;
        uint32_t sliceNum = kNumZero;
        uint32_t contractNum = kNumZero;
        for (auto& op : currFunctionPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                ++assembleNum;
            } else if (op.GetOpcode() == Opcode::OP_SLICE) {
                ++sliceNum;
            } else if (op.GetOpcode() == Opcode::OP_CONTRACT) {
                ++contractNum;
            }
        }
        EXPECT_EQ(assembleNum, kNumOne);
        EXPECT_EQ(sliceNum, kNumZero);
        EXPECT_EQ(contractNum, kNumZero);
    }
}

TEST_F(TestExpandFunctionPass, ExpandFunctionTest)
{
    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ExpandFunctionTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                 });

    std::vector<int64_t> shape{kNumExpSix, kNumExpSix};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);

    FUNCTION("A") { c = Div(a, b); }

    std::string jsonFilePath = "./config/pass/json/expand_function.json";
    auto programJson = Program::GetInstance().DumpJson();
    DumpJsonFile(programJson, jsonFilePath);
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    Function* currentFunction = Program::GetInstance().GetCurrentFunction();

    auto opListBefore = currentFunction->Operations().DuplicatedOpList();
    int divNumBefore = 0;
    int divNumAfter = 0;
    for (auto& op : opListBefore) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumBefore++;
        }
    }
    ExpandFunction expandFunction;
    expandFunction.RunOnFunction(*currentFunction);
    auto opListAfter = currentFunction->Operations().DuplicatedOpList();
    for (auto& op : opListAfter) {
        if (op->GetOpcodeStr().find("DIV") != std::string::npos) {
            divNumAfter++;
        }
    }
    EXPECT_EQ(divNumBefore, kNumOne);
    EXPECT_EQ(divNumAfter, kNumFour);
}

TEST_F(TestExpandFunctionPass, ViewDerivesVecTileFromCubeMatmulConsumer)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ViewCubeTile", "ViewCubeTile", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ViewCubeTile", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    constexpr int64_t kM = 64;
    constexpr int64_t kK = 96;
    constexpr int64_t kN = 96;
    constexpr int64_t kCubeM = 64;
    constexpr int64_t kCubeK = 32;
    constexpr int64_t kCubeN = 64;

    const std::vector<int64_t> shapeA = {kM, kK};
    const std::vector<int64_t> shapeB = {kK, kN};
    const std::vector<int64_t> viewShapeA = {kM, kCubeK};
    const std::vector<int64_t> viewShapeB = {kCubeK, kN};
    const std::vector<int64_t> shapeC = {kM, kN};
    const std::vector<int64_t> offset0 = {0, 0};

    auto incastA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto incastB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto viewOutA = IRBuilder().CreateTensorVar(DT_FP32, viewShapeA, SymbolicScalar::FromConcrete(viewShapeA));
    auto viewOutB = IRBuilder().CreateTensorVar(DT_FP32, viewShapeB, SymbolicScalar::FromConcrete(viewShapeB));
    auto matmulOut = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));

    auto& viewA = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastA}, {viewOutA});
    SetViewOpAttribute(viewA, offset0, viewShapeA);
    auto& viewB = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_VIEW, {incastB}, {viewOutB});
    SetViewOpAttribute(viewB, offset0, viewShapeB);

    TileShape::Current().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    TileShape::Current().GetVecTile().tile.clear();
    auto& matmul = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {viewOutA, viewOutB}, {matmulOut});
    matmul.GetTileShapeForSetting().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    matmul.GetTileShapeForSetting().GetVecTile().tile.clear();

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);

    EXPECT_EQ(CountInputSliceOpsWithSecondDimOffset(*func, incastB, kCubeN), 1);
}

TEST_F(TestExpandFunctionPass, MatmulValidRegionViewDoesNotRetileScore)
{
    for (const auto& reason : {"alias", "copy", "token", "offset"}) {
        SCOPED_TRACE(reason);
        auto func = std::make_shared<Function>(Program::GetInstance(), reason, reason, nullptr);
        func->SetGraphType(GraphType::TENSOR_GRAPH);
        const std::vector<int64_t> shape = {64, 64};
        auto makeTensor = [&]() {
            return IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
        };
        auto lhs = makeTensor();
        auto rhs = makeTensor();
        auto score = makeTensor();
        auto viewed = makeTensor();
        auto output = makeTensor();
        func->inCasts_ = {lhs, rhs};
        func->outCasts_ = {output};
        auto& matmul = PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B, {lhs, rhs}, {score});
        matmul.tileShape_.SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
        auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {score}, {viewed});
        SetViewOpAttribute(view, {0, 0}, shape);
        view.tileShape_.SetVecTile(8, 64);
        auto& exp = PassOperationUtils::AddOperation(*func, Opcode::OP_EXP, {viewed}, {output});
        exp.tileShape_.SetVecTile(8, 64);
        if (std::string(reason) == "copy") {
            view.SetAttribute("NeedCopy", true);
        } else if (std::string(reason) == "token") {
            view.tokens_ = {IRContext::Get().MakeSemanticToken("score_read", ir::TokenKind::READ, ir::Span::Unknown())};
        } else if (std::string(reason) == "offset") {
            SetViewOpAttribute(view, {1, 0}, shape);
        }
        ExpandFunction expandFunction;
        std::unordered_set<LogicalTensor*> gatherOutputs;
        auto skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
        EXPECT_EQ(skipOps.count(&view), std::string(reason) == "alias" ? 1u : 0u);
        EXPECT_TRUE(gatherOutputs.empty());
    }
}

TEST_F(TestExpandFunctionPass, ComputedConcatKeepsReadOnlyCubeViews)
{
    for (bool copyRequired : {false, true}) {
        auto func = std::make_shared<Function>(Program::GetInstance(), "ComputedConcat", "ComputedConcat", nullptr);
        func->SetGraphType(GraphType::TENSOR_GRAPH);
        const std::vector<int64_t> shape = {64, 64};
        const std::vector<int64_t> pieceShape = {64, 32};
        auto makeTensor = [](const std::vector<int64_t>& dims) {
            return IRBuilder().CreateTensorVar(DT_FP32, dims, SymbolicScalar::FromConcrete(dims));
        };
        auto input = makeTensor(pieceShape);
        auto gathered = makeTensor(shape);
        auto viewed = makeTensor(shape);
        auto cropped = makeTensor(pieceShape);
        auto rhs = makeTensor(shape);
        auto rhs2 = makeTensor({32, 64});
        auto result = makeTensor(shape);
        auto result2 = makeTensor(shape);
        func->inCasts_ = {input, rhs, rhs2};
        func->outCasts_ = {result, result2};
        std::vector<Operation*> writers;
        for (int piece = 0; piece < 2; ++piece) {
            auto computed = makeTensor(pieceShape);
            PassOperationUtils::AddOperation(*func, Opcode::OP_REGISTER_COPY, {input}, {computed});
            auto& assemble = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {computed}, {gathered});
            assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, piece * 32}));
            writers.push_back(&assemble);
        }
        auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {gathered}, {viewed});
        SetViewOpAttribute(view, {0, 0}, shape);
        auto& crop = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {viewed}, {cropped});
        SetViewOpAttribute(crop, {0, 0}, pieceShape);
        if (copyRequired) {
            crop.SetAttribute("NeedCopy", true);
        }
        PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B, {viewed, rhs}, {result});
        PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B, {cropped, rhs2}, {result2});
        ExpandFunction expandFunction;
        std::unordered_set<LogicalTensor*> gatherOutputs;
        auto skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
        EXPECT_EQ(skipOps.count(&view), copyRequired ? 0u : 1u);
        EXPECT_EQ(skipOps.count(&crop), copyRequired ? 0u : 1u);
        for (auto* writer : writers) {
            EXPECT_EQ(skipOps.count(writer), copyRequired ? 0u : 1u);
        }
        EXPECT_TRUE(gatherOutputs.empty());
    }
}

TEST_F(TestExpandFunctionPass, IncastColumnConcatKeepsCubeLoadBoundary)
{
    for (int rows : {1, 2}) {
        for (int split : {16, 32}) {
            config::SetPassOption(SG_SET_SCOPE, std::vector<int64_t>{-1, 0, 0});
            const auto name = "ColumnConcat" + std::to_string(rows) + "_" + std::to_string(split);
            auto func = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
            Program::GetInstance().InsertFuncToFunctionMap(name, func);
            func->SetGraphType(GraphType::TENSOR_GRAPH);
            TileShape::Current().SetVecTile(32, 32);
            TileShape::Current().SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
            const std::vector<int64_t> shape = {64, 64};
            auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
            const std::vector<int64_t> gatheredShape = {64 * rows, 64};
            auto gathered = IRBuilder().CreateTensorVar(DT_FP32, gatheredShape,
                                                        SymbolicScalar::FromConcrete(gatheredShape));
            auto output = IRBuilder().CreateTensorVar(DT_FP32, gatheredShape,
                                                      SymbolicScalar::FromConcrete(gatheredShape));
            func->inCasts_.push_back(input);
            func->outCasts_.push_back(output);
            std::vector<Operation*> aliases;
            for (int row = 0; row < rows; ++row) {
                for (int piece = 0; piece < 2; ++piece) {
                    const std::vector<int64_t> pieceShape = {64, piece == 0 ? split : 64 - split};
                    const int column = piece == 0 ? 0 : split;
                    auto slice = IRBuilder().CreateTensorVar(DT_FP32, pieceShape,
                                                             SymbolicScalar::FromConcrete(pieceShape));
                    auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {input}, {slice});
                    SetViewOpAttribute(view, {0, column}, pieceShape);
                    aliases.push_back(&view);
                    auto& assemble = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {slice}, {gathered});
                    assemble.SetOpAttribute(
                        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{row * 64, column}));
                    aliases.push_back(&assemble);
                }
            }
            auto& matmul = PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B, {gathered, input}, {output});
            matmul.tileShape_.SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
            ExpandFunction expandFunction;
            std::unordered_set<LogicalTensor*> gatherOutputs;
            auto skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
            for (auto* alias : aliases) {
                EXPECT_EQ(skipOps.count(alias), 1u);
            }
            EXPECT_EQ(gatherOutputs.count(gathered.get()), rows == 1 ? 0u : 1u);
            for (auto& op : func->Operations()) {
                op.SetScopeId(-1);
            }
            ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);
            int directLoads = 0;
            int concatLoads = 0;
            for (auto& op : func->Operations()) {
                if (op.GetOpcode() != Opcode::OP_SLICE || op.GetIOperands().size() != 1) {
                    continue;
                }
                auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
                if (attr == nullptr || attr->GetTo() != MemoryType::MEM_L1) {
                    continue;
                }
                if (rows == 1) {
                    if (op.GetIOperands().front() == gathered) {
                        ++concatLoads;
                    }
                    if (op.GetIOperands().front() == input) {
                        ++directLoads;
                    }
                }
            }
            if (rows == 1) {
                EXPECT_GT(directLoads, 0);
                if (split == 16) {
                    EXPECT_GT(concatLoads, 0); // A tile crossing a segment boundary must keep its copy.
                } else {
                    EXPECT_EQ(concatLoads, 0);
                }
            }
        }
    }
}

TEST_F(TestExpandFunctionPass, IncastGatherPreservedWhileMatmulExpands)
{
    for (bool symbolic : {false, true}) {
        for (bool globalView : {false, true}) {
            for (int copies : {1, 2}) {
                SCOPED_TRACE(::testing::Message()
                             << "symbolic=" << symbolic << " globalView=" << globalView << " copies=" << copies);
                const auto name = "IncastGather" + std::to_string(symbolic) + std::to_string(globalView) +
                                  std::to_string(copies);
                auto func = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
                Program::GetInstance().InsertFuncToFunctionMap(name, func);
                func->SetGraphType(GraphType::TENSOR_GRAPH);
                TileShape::Current().SetVecTile(32, 32);
                TileShape::Current().SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
                const std::vector<int64_t> shape = {64, 64};
                const std::vector<int64_t> sourceShape = {128, 64};
                const std::vector<int64_t> pieceShape = {32, 64};
                std::vector<std::vector<LogicalTensorPtr>> gathered(2);
                for (int operand = 0; operand < 2; ++operand) {
                    auto input = IRBuilder().CreateTensorVar(DT_FP32, sourceShape,
                                                             SymbolicScalar::FromConcrete(sourceShape));
                    func->inCasts_.push_back(input);
                    if (globalView) {
                        auto global = IRBuilder().CreateTensorVar(DT_FP32, sourceShape,
                                                                  SymbolicScalar::FromConcrete(sourceShape));
                        auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {input}, {global});
                        SetViewOpAttribute(view, {0, 0}, sourceShape);
                        view.SetAttribute(OpAttributeKey::isGlobalInput, true);
                        input = global;
                    }
                    for (int copy = 0; copy < copies; ++copy) {
                        gathered[operand].push_back(
                            IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape)));
                    }
                    for (int piece = 0; piece < 2; ++piece) {
                        auto slice = IRBuilder().CreateTensorVar(DT_FP32, pieceShape,
                                                                 SymbolicScalar::FromConcrete(pieceShape));
                        auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {input}, {slice});
                        std::vector<int64_t> offset = {piece * 32, 0};
                        auto dynOffset = SymbolicScalar::FromConcrete(offset);
                        if (symbolic) {
                            dynOffset[0] = CreateTestScalarVar("gather_row") + piece * 32;
                        }
                        view.SetOpAttribute(std::make_shared<ViewOpAttribute>(
                            offset, dynOffset, SymbolicScalar::FromConcrete(pieceShape)));
                        view.tileShape_.SetVecTile(32, 32);
                        for (const auto& output : gathered[operand]) {
                            auto& assemble = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {slice},
                                                                              {output});
                            assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset));
                            assemble.tileShape_.SetVecTile(32, 32);
                        }
                    }
                }
                for (int copy = 0; copy < copies; ++copy) {
                    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
                    auto& matmul = PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B,
                                                                    {gathered[0][copy], gathered[1][copy]}, {output});
                    matmul.tileShape_.SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
                    func->outCasts_.push_back(output);
                }
                auto originals = func->Operations().operations_;
                ExpandFunction expandFunction;
                ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);
                EXPECT_EQ(func->GetGraphType(), GraphType::TILE_GRAPH);
                for (const auto& original : originals) {
                    if (original->GetOpcode() != Opcode::OP_VIEW && original->GetOpcode() != Opcode::OP_ASSEMBLE) {
                        continue;
                    }
                    int preserved = 0;
                    for (auto& op : func->Operations()) {
                        if (op.GetOpcode() == original->GetOpcode() && op.GetIOperands() == original->GetIOperands() &&
                            op.GetOOperands() == original->GetOOperands()) {
                            EXPECT_EQ(op.GetOpAttribute(), original->GetOpAttribute());
                            ++preserved;
                        }
                    }
                    EXPECT_EQ(preserved, 1) << original->GetOpcodeStr();
                }
                int matmulCount = 0;
                int gatherLoadCount = 0;
                for (auto& op : func->Operations()) {
                    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
                        ++matmulCount;
                        EXPECT_EQ(op.GetOOperands()[0]->GetShape(), (std::vector<int64_t>{32, 32}));
                    }
                    for (const auto& operand : gathered) {
                        for (const auto& input : operand) {
                            if (op.GetIOperands().size() == 1 && op.GetIOperands().front() == input) {
                                EXPECT_EQ(op.GetOpcode(), Opcode::OP_VIEW);
                                auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
                                ASSERT_NE(attr, nullptr);
                                EXPECT_EQ(attr->GetTo(), MemoryType::MEM_L1);
                                ++gatherLoadCount;
                            }
                        }
                    }
                }
                EXPECT_GT(matmulCount, copies);
                EXPECT_GT(gatherLoadCount, 0);
            }
        }
    }
}

TEST_F(TestExpandFunctionPass, StandaloneSymbolicIncastViewFanoutPreservedAndTokenRejected)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "StandaloneSymbolicView", "StandaloneSymbolicView",
                                           nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("StandaloneSymbolicView", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);
    TileShape::Current().SetVecTile(32, 32);
    const std::vector<int64_t> sourceShape = {128, 64};
    const std::vector<int64_t> shape = {64, 64};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, sourceShape, SymbolicScalar::FromConcrete(sourceShape));
    auto viewed = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
    auto rhs = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
    func->inCasts_ = {input, rhs};
    auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {input}, {viewed});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{CreateTestScalarVar("standalone_row"), 0},
        SymbolicScalar::FromConcrete(shape)));
    for (int copy = 0; copy < 2; ++copy) {
        auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
        auto& matmul = PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B, {viewed, rhs}, {output});
        matmul.tileShape_.SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
        func->outCasts_.push_back(output);
    }
    ExpandFunction expandFunction;
    std::unordered_set<LogicalTensor*> gatherOutputs;
    auto skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
    ASSERT_EQ(skipOps.count(&view), 1u);
    EXPECT_TRUE(gatherOutputs.empty());
    view.tokens_ = {IRContext::Get().MakeSemanticToken("standalone_read", ir::TokenKind::READ, ir::Span::Unknown())};
    skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
    EXPECT_EQ(skipOps.count(&view), 0u);
    EXPECT_TRUE(gatherOutputs.empty());
    view.tokens_.clear();

    auto originalView = func->Operations().operations_.front();
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);
    int preserved = 0;
    int matmulCount = 0;
    int standaloneLoadCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW && op.GetIOperands() == originalView->GetIOperands() &&
            op.GetOOperands() == originalView->GetOOperands()) {
            EXPECT_EQ(op.GetOpAttribute(), originalView->GetOpAttribute());
            ++preserved;
        }
        if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
            ++matmulCount;
        }
        if (op.GetIOperands().size() == 1 && op.GetIOperands().front() == viewed) {
            EXPECT_EQ(op.GetOpcode(), Opcode::OP_SLICE);
            ++standaloneLoadCount;
        }
    }
    EXPECT_EQ(preserved, 1);
    EXPECT_GT(matmulCount, 2);
    EXPECT_GT(standaloneLoadCount, 0);
}

TEST_F(TestExpandFunctionPass, IncastGatherRejectsAllWritersIfAnyRequiresExpansion)
{
    for (const std::string reason :
         {"computed", "computed_source", "input_token", "output_token", "copy", "original_memory", "target_memory",
          "view_memory", "assemble_memory", "consumer_token", "view_token", "unregistered_input"}) {
        SCOPED_TRACE(reason);
        auto func = std::make_shared<Function>(Program::GetInstance(), "RejectIncastGather", "RejectIncastGather",
                                               nullptr);
        func->SetGraphType(GraphType::TENSOR_GRAPH);
        TileShape::Current().SetVecTile(32, 32);
        const std::vector<int64_t> shape = {64, 64};
        const std::vector<int64_t> pieceShape = {32, 64};
        auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
        auto gathered = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
        auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
        func->inCasts_.push_back(input);
        func->outCasts_.push_back(output);
        std::vector<Operation*> views;
        std::vector<Operation*> writers;
        for (int piece = 0; piece < 2; ++piece) {
            auto slice = IRBuilder().CreateTensorVar(DT_FP32, pieceShape, SymbolicScalar::FromConcrete(pieceShape));
            auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {input}, {slice});
            SetViewOpAttribute(view, {piece * 32, 0}, pieceShape);
            views.push_back(&view);
            auto& assemble = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {slice}, {gathered});
            assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{piece * 32, 0}));
            writers.push_back(&assemble);
        }
        auto& matmul = PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B, {gathered, input}, {output});
        matmul.tileShape_.SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
        ExpandFunction expandFunction;
        std::unordered_set<LogicalTensor*> gatherOutputs;
        auto skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
        for (auto* writer : writers) {
            ASSERT_EQ(skipOps.count(writer), 1u);
        }
        if (reason == "computed") {
            PassOperationUtils::AddOperation(*func, Opcode::OP_EXP, {input}, {gathered});
        } else if (reason == "computed_source") {
            auto computed = IRBuilder().CreateTensorVar(DT_FP32, shape, SymbolicScalar::FromConcrete(shape));
            PassOperationUtils::AddOperation(*func, Opcode::OP_EXP, {input}, {computed});
            views.back()->ReplaceInput(computed, input);
        } else if (reason == "input_token") {
            writers.back()->tokens_ = {
                IRContext::Get().MakeSemanticToken("gather_dep", ir::TokenKind::WRITE, ir::Span::Unknown())};
        } else if (reason == "output_token") {
            writers.back()->result_token_ = {
                IRContext::Get().MakeSemanticToken("gather_write", ir::TokenKind::WRITE, ir::Span::Unknown())};
        } else if (reason == "copy") {
            writers.back()->SetAttr("NeedCopy", true);
        } else if (reason == "original_memory") {
            writers.back()->GetIOperands()[0]->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
        } else if (reason == "target_memory") {
            writers.back()->GetIOperands()[0]->SetMemoryTypeToBe(MemoryType::MEM_UB);
        } else if (reason == "view_memory") {
            std::dynamic_pointer_cast<ViewOpAttribute>(views.back()->GetOpAttribute())->SetToType(MemoryType::MEM_UB);
        } else if (reason == "assemble_memory") {
            std::dynamic_pointer_cast<AssembleOpAttribute>(writers.back()->GetOpAttribute())
                ->SetFromType(MemoryType::MEM_UB);
        } else if (reason == "consumer_token" || reason == "view_token") {
            auto* target = reason == "consumer_token" ? &matmul : views.back();
            target->tokens_ = {
                IRContext::Get().MakeSemanticToken("gather_read", ir::TokenKind::READ, ir::Span::Unknown())};
        } else if (reason == "unregistered_input") {
            func->inCasts_.clear();
        }
        gatherOutputs.clear();
        skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
        EXPECT_TRUE(gatherOutputs.empty());
        for (auto* writer : writers) {
            EXPECT_EQ(skipOps.count(writer), 0u);
        }
        for (auto* view : views) {
            EXPECT_EQ(skipOps.count(view), 0u);
        }
    }
}

TEST_F(TestExpandFunctionPass, ComputedMatmulAliasesSkipExpansionOnlyWhenSafe)
{
    for (const std::string reason : {"none", "offset", "dynamic_offset", "symbolic_offset", "dtype", "input_token",
                                     "output_token", "producer_token", "writer_token", "consumer_token",
                                     "original_memory", "target_memory", "view_memory", "assemble_memory"}) {
        SCOPED_TRACE(reason);
        auto func = std::make_shared<Function>(Program::GetInstance(), "ComputedMatmulAliases", "ComputedMatmulAliases",
                                               nullptr);
        func->SetGraphType(GraphType::TENSOR_GRAPH);
        TileShape::Current().SetVecTile(32, 32);
        const std::vector<int64_t> shape = {64, 64};
        const std::vector<int64_t> pieceShape = {32, 64};
        auto makeTensor = [](const std::vector<int64_t>& dims, DataType dtype = DT_FP32) {
            return IRBuilder().CreateTensorVar(dtype, dims, SymbolicScalar::FromConcrete(dims));
        };
        auto input = makeTensor(shape);
        auto pieceInput = makeTensor(pieceShape);
        auto assembled = makeTensor(shape);
        auto direct = makeTensor(shape);
        auto assembledView = makeTensor(shape, reason == "dtype" ? DT_FP16 : DT_FP32);
        auto directView = makeTensor(shape, reason == "dtype" ? DT_FP16 : DT_FP32);
        auto output = makeTensor(shape);
        func->inCasts_ = {input, pieceInput};
        func->outCasts_ = {output};
        std::vector<Operation*> producers;
        std::vector<Operation*> writers;
        for (int piece = 0; piece < 2; ++piece) {
            auto computed = makeTensor(pieceShape);
            auto& producer = PassOperationUtils::AddOperation(
                *func, piece == 0 ? Opcode::OP_CAST : Opcode::OP_REGISTER_COPY, {pieceInput}, {computed});
            producers.push_back(&producer);
            auto& writer = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {computed}, {assembled});
            writer.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{piece * 32, 0}));
            writers.push_back(&writer);
        }
        auto& cast = PassOperationUtils::AddOperation(*func, Opcode::OP_CAST, {input}, {direct});
        producers.push_back(&cast);
        auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {assembled}, {assembledView});
        auto& castView = PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {direct}, {directView});
        auto& matmul = PassOperationUtils::AddOperation(*func, Opcode::OP_A_MUL_B, {assembledView, directView},
                                                        {output});
        matmul.tileShape_.SetCubeTile({32, 32}, {32, 32, 32}, {32, 32});
        auto token = IRContext::Get().MakeSemanticToken("computed_alias_read", ir::TokenKind::READ,
                                                        ir::Span::Unknown());
        for (auto* alias : {&view, &castView}) {
            SetViewOpAttribute(*alias, {0, 0}, shape);
            auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(alias->GetOpAttribute());
            if (reason == "offset") {
                attr->GetFromOffset()[0] = 1;
            } else if (reason == "dynamic_offset") {
                attr->GetFromDynOffset()[0] = 1;
            } else if (reason == "symbolic_offset") {
                attr->GetFromDynOffset()[0] = CreateTestScalarVar("computed_alias_row");
            } else if (reason == "input_token") {
                alias->tokens_ = {token};
            } else if (reason == "output_token") {
                alias->result_token_ = {token};
            } else if (reason == "original_memory") {
                alias->GetOOperands()[0]->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
            } else if (reason == "target_memory") {
                alias->GetOOperands()[0]->SetMemoryTypeToBe(MemoryType::MEM_UB);
            } else if (reason == "view_memory") {
                attr->SetToType(MemoryType::MEM_UB);
            }
        }
        if (reason == "producer_token") {
            for (auto* producer : producers) {
                producer->result_token_ = {token};
            }
        } else if (reason == "writer_token") {
            writers.back()->tokens_ = {token};
        } else if (reason == "consumer_token") {
            matmul.tokens_ = {token};
        } else if (reason == "assemble_memory") {
            std::dynamic_pointer_cast<AssembleOpAttribute>(writers.back()->GetOpAttribute())
                ->SetFromType(MemoryType::MEM_UB);
        }
        ExpandFunction expandFunction;
        std::unordered_set<LogicalTensor*> gatherOutputs;
        auto skipOps = expandFunction.CollectSkipExpandOps(*func, func->Operations().operations_, gatherOutputs);
        const size_t assembledSkipped = reason == "none" ? 1u : 0u;
        const size_t directSkipped = reason == "none" || reason == "writer_token" || reason == "assemble_memory" ? 1u :
                                                                                                                   0u;
        EXPECT_EQ(skipOps.count(&view), assembledSkipped);
        EXPECT_EQ(skipOps.count(&castView), directSkipped);
        for (auto* writer : writers) {
            EXPECT_EQ(skipOps.count(writer), assembledSkipped);
        }
        for (auto* producer : producers) {
            EXPECT_EQ(skipOps.count(producer), 0u);
        }
        EXPECT_EQ(skipOps.count(&matmul), 0u);
        EXPECT_EQ(skipOps.size(), 3 * assembledSkipped + directSkipped);
        EXPECT_TRUE(gatherOutputs.empty());
    }
}

TEST_F(TestExpandFunctionPass, AssembleDerivesVecTileFromCubeMatmulProducer)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "AssembleCubeTile", "AssembleCubeTile", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("AssembleCubeTile", func);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    constexpr int64_t kM = 64;
    constexpr int64_t kK = 32;
    constexpr int64_t kN = 256;
    constexpr int64_t kCubeM = 64;
    constexpr int64_t kCubeK = 32;
    constexpr int64_t kCubeN = 128;

    const std::vector<int64_t> shapeA = {kM, kK};
    const std::vector<int64_t> shapeB = {kK, kN};
    const std::vector<int64_t> shapeC = {kM, kN};
    const std::vector<int64_t> shapeOut = {kM, kN};
    const std::vector<int64_t> offset0 = {0, 0};
    const std::vector<int64_t> expectedSliceShape = {kCubeM, kCubeN};

    auto incastA = IRBuilder().CreateTensorVar(DT_FP32, shapeA, SymbolicScalar::FromConcrete(shapeA));
    auto incastB = IRBuilder().CreateTensorVar(DT_FP32, shapeB, SymbolicScalar::FromConcrete(shapeB));
    auto matmulOut = IRBuilder().CreateTensorVar(DT_FP32, shapeC, SymbolicScalar::FromConcrete(shapeC));
    auto outCast = IRBuilder().CreateTensorVar(DT_FP32, shapeOut, SymbolicScalar::FromConcrete(shapeOut));

    TileShape::Current().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    TileShape::Current().SetVecTile(kCubeM, kCubeM);
    auto& matmul = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_A_MUL_B, {incastA, incastB}, {matmulOut});
    matmul.GetTileShapeForSetting().SetCubeTile({kCubeM, kCubeM}, {kCubeK, kCubeK, kCubeK}, {kCubeN, kCubeN});
    matmul.GetTileShapeForSetting().GetVecTile().tile.clear();

    auto& assemble = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_ASSEMBLE, {matmulOut}, {outCast});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset0));
    assemble.GetTileShapeForSetting().GetVecTile().tile.clear();

    ExpandFunction expandFunction;
    ASSERT_EQ(expandFunction.RunOnFunction(*func), SUCCESS);

    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, matmulOut, expectedSliceShape), 2);
    EXPECT_EQ(CountInputSliceOpsWithSecondDimOffset(*func, matmulOut, kCubeN), 1);
    EXPECT_EQ(CountSliceOpsOnInputWithShape(*func, matmulOut, {kCubeM, kCubeM}), 0);
}

TEST_F(TestExpandFunctionPass, ComputeOpReadGoesToSliceAndWriteGoesToContract)
{
    TileShape::Current().SetVecTile(kNumExpSix, kNumExpSix);
    auto func = std::make_shared<Function>(Program::GetInstance(), "ComputeOpTokenAttach", "ComputeOpTokenAttach",
                                           nullptr);
    ASSERT_NE(func, nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto lhs = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto rhs = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto out = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& addOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ADD, {lhs, rhs}, {out});
    addOp.tileShape_.SetVecTile(kNumExpSix, kNumExpSix);

    auto readSemantic = IRContext::Get().MakeSemanticToken("lhs_r", ir::TokenKind::READ, ir::Span::Unknown());
    auto writeSemantic = IRContext::Get().MakeSemanticToken("out_w", ir::TokenKind::WRITE, ir::Span::Unknown());
    auto depSemantic = IRContext::Get().MakeSemanticToken("dep_w", ir::TokenKind::WRITE, ir::Span::Unknown());
    lhs->SetReadToken(readSemantic);
    addOp.result_token_ = {readSemantic, writeSemantic};
    addOp.tokens_ = {depSemantic};

    func->inCasts_.push_back(lhs);
    func->inCasts_.push_back(rhs);
    func->outCasts_.push_back(out);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandPass;
    ASSERT_EQ(expandPass.RunOnFunction(*func), SUCCESS);

    bool sliceHasRead = false;
    bool rhsSliceHasRead = false;
    bool contractHasWrite = false;
    bool contractHasDep = false;
    bool addHasRead = false;
    for (auto& op : func->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.GetIOperands().empty() && op.GetIOperands()[0] == lhs) {
            sliceHasRead = sliceHasRead || HasNormalTokenNamed(op.result_token_, "lhs_r");
        }
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.GetIOperands().empty() && op.GetIOperands()[0] == rhs) {
            rhsSliceHasRead = rhsSliceHasRead || HasNormalTokenNamed(op.result_token_, "lhs_r");
        }
        if (op.GetOpcode() == Opcode::OP_CONTRACT && !op.GetOOperands().empty() && op.GetOOperands()[0] == out) {
            contractHasWrite = contractHasWrite || HasNormalTokenNamed(op.result_token_, "out_w");
            contractHasDep = contractHasDep || HasNormalTokenNamed(op.tokens_, "dep_w");
        }
        if (op.GetOpcode() == Opcode::OP_ADD) {
            addHasRead = addHasRead || HasNormalTokenNamed(op.result_token_, "lhs_r");
        }
    }
    EXPECT_TRUE(sliceHasRead);
    EXPECT_FALSE(rhsSliceHasRead);
    EXPECT_TRUE(contractHasWrite);
    EXPECT_TRUE(contractHasDep);
    EXPECT_FALSE(addHasRead);
}

TEST_F(TestExpandFunctionPass, ComputeTokenDependenciesAreRebuiltBeforeSortingExpandedOps)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ComputeTokenDependency", "ComputeTokenDependency",
                                           nullptr);
    const std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    auto lhs = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto rhs = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto firstOut = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto secondOut = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& readOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ADD, {lhs, rhs}, {firstOut});
    auto& dependentOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ADD, {lhs, rhs}, {secondOut});
    readOp.tileShape_.SetVecTile(kNumEight, kNumExpFour);
    dependentOp.tileShape_.SetVecTile(kNumEight, kNumExpFour);
    auto readToken = IRContext::Get().MakeSemanticToken("lhs_r", ir::TokenKind::READ, ir::Span::Unknown());
    lhs->SetReadToken(readToken);
    readOp.result_token_ = {readToken};
    dependentOp.tokens_ = {readToken};
    func->inCasts_ = {lhs, rhs};
    func->outCasts_ = {firstOut, secondOut};
    func->SetGraphType(GraphType::TENSOR_GRAPH);
    // Earlier passes leave a dependency cache referring to the tensor operations.
    func->RefreshVarDependency();

    ExpandFunction expandPass;
    ASSERT_EQ(expandPass.RunOnFunction(*func), SUCCESS);

    size_t addCount = 0;
    size_t dependentCount = 0;
    for (auto& op : func->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ADD) {
            continue;
        }
        ++addCount;
        if (!op.tokens_.empty()) {
            ++dependentCount;
            EXPECT_FALSE(op.ProducerOpsByToken().empty());
        }
    }
    EXPECT_EQ(addCount, 4UL);
    EXPECT_EQ(dependentCount, 2UL);
}

TEST_F(TestExpandFunctionPass, AssembleWriteGoesToContractAndConsumedOnNextAssemble)
{
    TileShape::Current().SetVecTile(kNumExpSix, kNumExpSix);
    auto func = std::make_shared<Function>(Program::GetInstance(), "AssembleTokenAttach", "AssembleTokenAttach",
                                           nullptr);
    ASSERT_NE(func, nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto src1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto src2 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto out = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto assembleAttr1 = CreateAssembleOpAttr();
    auto assembleAttr2 = CreateAssembleOpAttr();

    auto& firstAssemble = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {src1}, {out});
    auto& secondAssemble = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {src2}, {out});
    firstAssemble.SetOpAttribute(assembleAttr1);
    secondAssemble.SetOpAttribute(assembleAttr2);
    firstAssemble.tileShape_.SetVecTile(kNumExpSix, kNumExpSix);
    secondAssemble.tileShape_.SetVecTile(kNumExpSix, kNumExpSix);

    auto readSemantic = IRContext::Get().MakeSemanticToken("src1_r", ir::TokenKind::READ, ir::Span::Unknown());
    auto firstWrite = IRContext::Get().MakeSemanticToken("out_w1", ir::TokenKind::WRITE, ir::Span::Unknown());
    auto secondWrite = IRContext::Get().MakeSemanticToken("out_w2", ir::TokenKind::WRITE, ir::Span::Unknown());
    src1->SetReadToken(readSemantic);
    firstAssemble.result_token_ = {readSemantic, firstWrite};
    secondAssemble.result_token_ = {secondWrite};
    secondAssemble.tokens_ = {firstWrite};

    func->inCasts_.push_back(src1);
    func->inCasts_.push_back(src2);
    func->outCasts_.push_back(out);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandPass;
    ASSERT_EQ(expandPass.RunOnFunction(*func), SUCCESS);

    bool src1SliceHasRead = false;
    int contractWithFirstWrite = 0;
    int contractWithSecondWrite = 0;
    int contractConsumingFirstWrite = 0;
    for (auto& op : func->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.GetIOperands().empty() && op.GetIOperands()[0] == src1) {
            src1SliceHasRead = src1SliceHasRead || HasNormalTokenNamed(op.result_token_, "src1_r");
        }
        if (op.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        if (HasNormalTokenNamed(op.result_token_, "out_w1")) {
            ++contractWithFirstWrite;
        }
        if (HasNormalTokenNamed(op.result_token_, "out_w2")) {
            ++contractWithSecondWrite;
        }
        if (HasNormalTokenNamed(op.tokens_, "out_w1")) {
            ++contractConsumingFirstWrite;
        }
    }
    EXPECT_TRUE(src1SliceHasRead);
    EXPECT_EQ(contractWithFirstWrite, 1);
    EXPECT_EQ(contractWithSecondWrite, 1);
    EXPECT_EQ(contractConsumingFirstWrite, 1);
}

TEST_F(TestExpandFunctionPass, AtomicRmwConvertsSemanticTokensToNormal)
{
    TileShape::Current().SetVecTile(kNumExpSix, kNumExpSix);
    auto func = std::make_shared<Function>(Program::GetInstance(), "AtomicRmwTokenAttach", "AtomicRmwTokenAttach",
                                           nullptr);
    ASSERT_NE(func, nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto src1 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto src2 = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto dst = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& atomicOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ATOMIC_RMW, {src1}, {dst});
    atomicOp.SetOpAttribute(CreateAssembleOpAttr());
    atomicOp.tileShape_.SetVecTile(kNumExpSix, kNumExpSix);

    auto& assembleOp = PassOperationUtils::AddOperation(*func, Opcode::OP_ASSEMBLE, {src2}, {dst});
    assembleOp.SetOpAttribute(CreateAssembleOpAttr());
    assembleOp.tileShape_.SetVecTile(kNumExpSix, kNumExpSix);

    auto readSemantic = IRContext::Get().MakeSemanticToken("src1_r", ir::TokenKind::READ, ir::Span::Unknown());
    auto atomicWrite = IRContext::Get().MakeSemanticToken("dst_w1", ir::TokenKind::WRITE, ir::Span::Unknown());
    auto depSemantic = IRContext::Get().MakeSemanticToken("dep_w", ir::TokenKind::WRITE, ir::Span::Unknown());
    auto assembleWrite = IRContext::Get().MakeSemanticToken("dst_w2", ir::TokenKind::WRITE, ir::Span::Unknown());
    src1->SetReadToken(readSemantic);
    atomicOp.result_token_ = {readSemantic, atomicWrite};
    atomicOp.tokens_ = {depSemantic};
    assembleOp.result_token_ = {assembleWrite};
    assembleOp.tokens_ = {atomicWrite};

    func->inCasts_.push_back(src1);
    func->inCasts_.push_back(src2);
    func->outCasts_.push_back(dst);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandPass;
    ASSERT_EQ(expandPass.RunOnFunction(*func), SUCCESS);

    uint32_t atomicNum = kNumZero;
    bool atomicHasRead = false;
    bool atomicHasWrite = false;
    bool atomicHasDep = false;
    bool assembleContractConsumesAtomicWrite = false;
    for (auto& op : func->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            ++atomicNum;
            atomicHasRead = atomicHasRead || HasNormalTokenNamed(op.result_token_, "src1_r");
            atomicHasWrite = atomicHasWrite || HasNormalTokenNamed(op.result_token_, "dst_w1");
            atomicHasDep = atomicHasDep || HasNormalTokenNamed(op.tokens_, "dep_w");
        }
        if (op.GetOpcode() == Opcode::OP_CONTRACT && HasNormalTokenNamed(op.tokens_, "dst_w1")) {
            assembleContractConsumesAtomicWrite = true;
        }
    }
    EXPECT_EQ(atomicNum, kNumOne);
    EXPECT_TRUE(atomicHasRead);
    EXPECT_TRUE(atomicHasWrite);
    EXPECT_TRUE(atomicHasDep);
    EXPECT_TRUE(assembleContractConsumesAtomicWrite);
}

// VIEW toDynValidShape is tighter than input-clip. Expanded SLICEs must inherit
// the per-tile clip of that VIEW attr, not GetViewValidShape(input, fromOffset).
TEST_F(TestExpandFunctionPass, ViewExpandSlicesToDynValidShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandViewValidShape",
                                                      "TestExpandViewValidShape", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tileShape = {kNumExpFive, kNumExpSix};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto viewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto expOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto tighterValid = SymbolicScalar::FromConcrete({10, kNumExpSix});
    viewOut->UpdateDynValidShape(tighterValid);

    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inCast}, {viewOut});
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {viewOut}, {expOut});
    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {expOut}, {outCast});

    std::vector<int64_t> offsets = {0, 0};
    viewOp.SetOpAttribute(
        std::make_shared<ViewOpAttribute>(offsets, SymbolicScalar::FromConcrete(offsets), tighterValid));
    assembleOp.SetOpAttribute(CreateAssembleOpAttr());
    viewOp.tileShape_.SetVecTile(tileShape);
    expOp.tileShape_.SetVecTile(tileShape);
    assembleOp.tileShape_.SetVecTile(tileShape);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expand;
    EXPECT_EQ(expand.RunOnFunction(*currFunctionPtr), SUCCESS);

    auto expectFirst = GetViewValidShape(tighterValid, {0, 0}, {}, tileShape);
    auto expectSecond = GetViewValidShape(tighterValid, {kNumExpFive, 0}, {}, tileShape);
    bool sawFirst = false;
    bool sawSecond = false;
    for (auto& op : currFunctionPtr->Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_SLICE || op.GetIOperands().empty() || op.GetIOperands()[0] != inCast) {
            continue;
        }
        auto sliceAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        ASSERT_NE(sliceAttr, nullptr);
        ASSERT_FALSE(op.GetOOperands().empty());
        const auto& fromOffset = sliceAttr->GetFromOffset();
        ASSERT_GE(fromOffset.size(), 1u);
        const auto& expect = (fromOffset[0] == 0) ? expectFirst : expectSecond;
        if (fromOffset[0] == 0) {
            sawFirst = true;
        } else if (fromOffset[0] == kNumExpFive) {
            sawSecond = true;
        } else {
            continue;
        }
        ASSERT_EQ(sliceAttr->GetToDynValidShape().size(), expect.size());
        ASSERT_EQ(op.GetOOperands()[0]->GetDynValidShape().size(), expect.size());
        for (size_t i = 0; i < expect.size(); ++i) {
            EXPECT_EQ(sliceAttr->GetToDynValidShape()[i].Dump(), expect[i].Dump());
            EXPECT_EQ(op.GetOOperands()[0]->GetDynValidShape()[i].Dump(), expect[i].Dump());
        }
    }
    EXPECT_TRUE(sawFirst);
    EXPECT_TRUE(sawSecond);
}

namespace {
// 仅统计以 input 为第 0 个输入的指定 opcode 操作数（GatherInUB 展开会围绕自身 view 输入生成
// SLICE/CONTRACT，需按输入 tensor 区分 view 展开 slice 与 gather 物化 slice）
uint32_t CountOpsWithInput(Function& function, Opcode opcode, const LogicalTensorPtr& input)
{
    uint32_t count = kNumZero;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != opcode || op.GetIOperands().empty() || op.GetIOperands()[kSizeZero] != input) {
            continue;
        }
        ++count;
    }
    return count;
}

// 仅统计以 output 为第 0 个输出的指定 opcode 操作数
uint32_t CountOpsWithOutput(Function& function, Opcode opcode, const LogicalTensorPtr& output)
{
    uint32_t count = kNumZero;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != opcode || op.GetOOperands().empty() || op.GetOOperands()[kSizeZero] != output) {
            continue;
        }
        ++count;
    }
    return count;
}

uint32_t CountOpsWithOpcode(Function& function, Opcode opcode)
{
    uint32_t count = kNumZero;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == opcode) {
            ++count;
        }
    }
    return count;
}
} // namespace

/*
TESTExpandFunctionGatherInUbParamView
kvCache{64,64}(DDR)->view->kvView{64,64}(DDR)
kvView(第0个输入params), bIdx{1,64}, s1Idx{1,64}->gatherInUB->gatherOut{64,64}
view 输出仅被 GatherInUB 作为 params(第 0 个输入)消费时保留穿透别名语义不展开：
不生成以 kvCache 为输入的 slice，也不生成写 kvView 的 contract，view 原样保留；
gatherInUB 正常按 vec tile 展开
*/
TEST_F(TestExpandFunctionPass, GatherInUbParamViewShouldNotExpand)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherInUbParamView",
                                                      "TestGatherInUbParamView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> idxShape = {kNumOne, kNumExpSix};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    std::vector<int64_t> tileShape = {kNumExpFive, kNumExpSix};
    auto kvCache = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    kvCache->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto kvView = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    kvView->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto bIdx = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, idxShape, CreateTestConstIntVector(idxShape));
    auto s1Idx = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, idxShape, CreateTestConstIntVector(idxShape));
    auto gatherOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {kvCache}, {kvView});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    viewOp.tileShape_.SetVecTile(tileShape);
    auto& gatherOp = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_GATHER_IN_UB, {kvView, bIdx, s1Idx}, {gatherOut},
        [](Operation& op) { op.SetAttribute(OpAttributeKey::blockSize, static_cast<int64_t>(kNumExpFour)); });
    gatherOp.tileShape_.SetVecTile(tileShape);

    currFunctionPtr->inCasts_.push_back(kvCache);
    currFunctionPtr->inCasts_.push_back(bIdx);
    currFunctionPtr->inCasts_.push_back(s1Idx);
    currFunctionPtr->outCasts_.push_back(gatherOut);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountOpsWithInput(*currFunctionPtr, Opcode::OP_VIEW, kvCache), kNumOne)
        << "GatherInUB params view should keep alias semantic and not expand";
    EXPECT_EQ(CountOpsWithInput(*currFunctionPtr, Opcode::OP_SLICE, kvCache), kNumZero)
        << "No slice on kvCache should be generated for the skipped params view";
    EXPECT_EQ(CountOpsWithOutput(*currFunctionPtr, Opcode::OP_CONTRACT, kvView), kNumZero)
        << "No contract writing kvView should be generated for the skipped params view";
    EXPECT_EQ(CountOpsWithOpcode(*currFunctionPtr, Opcode::OP_GATHER_IN_UB), kNumTwo)
        << "GatherInUB should expand per vec tile";
}

/*
TESTExpandFunctionGatherInUbParamViewWithOtherConsumer
kvCache{64,64}(DDR)->view->kvView{64,64}(DDR)->gatherInUB(第0个输入params)->gatherOut{64,64}
                                          ->exp->expOut{64,64}
view 输出除 GatherInUB(params) 外还存在其它消费方时不跳过展开：
view 按 vec tile {32,64} 物化为 2 个以 kvCache 为输入的 slice 和 2 个写 kvView 的 contract
*/
TEST_F(TestExpandFunctionPass, GatherInUbParamViewWithOtherConsumerShouldExpand)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestGatherInUbViewOtherConsumer",
                                                      "TestGatherInUbViewOtherConsumer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> idxShape = {kNumOne, kNumExpSix};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    std::vector<int64_t> tileShape = {kNumExpFive, kNumExpSix};
    auto kvCache = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    kvCache->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto kvView = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    kvView->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto bIdx = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, idxShape, CreateTestConstIntVector(idxShape));
    auto s1Idx = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, idxShape, CreateTestConstIntVector(idxShape));
    auto gatherOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto expOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {kvCache}, {kvView});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    viewOp.tileShape_.SetVecTile(tileShape);
    auto& gatherOp = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_GATHER_IN_UB, {kvView, bIdx, s1Idx}, {gatherOut},
        [](Operation& op) { op.SetAttribute(OpAttributeKey::blockSize, static_cast<int64_t>(kNumExpFour)); });
    gatherOp.tileShape_.SetVecTile(tileShape);
    auto& expOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {kvView}, {expOut});
    expOp.tileShape_.SetVecTile(tileShape);

    currFunctionPtr->inCasts_.push_back(kvCache);
    currFunctionPtr->inCasts_.push_back(bIdx);
    currFunctionPtr->inCasts_.push_back(s1Idx);
    currFunctionPtr->outCasts_.push_back(gatherOut);
    currFunctionPtr->outCasts_.push_back(expOut);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountOpsWithInput(*currFunctionPtr, Opcode::OP_VIEW, kvCache), kNumZero)
        << "view with consumer other than GatherInUB params should expand";
    EXPECT_EQ(CountOpsWithInput(*currFunctionPtr, Opcode::OP_SLICE, kvCache), kNumTwo)
        << "view should materialize into 2 slices on kvCache with vec tile {32,64}";
    EXPECT_EQ(CountOpsWithOutput(*currFunctionPtr, Opcode::OP_CONTRACT, kvView), kNumTwo)
        << "view should materialize into 2 contracts writing kvView";
    EXPECT_EQ(CountOpsWithOpcode(*currFunctionPtr, Opcode::OP_GATHER_IN_UB), kNumTwo);
    EXPECT_EQ(CountOpsWithOpcode(*currFunctionPtr, Opcode::OP_EXP), kNumTwo);
}

} // namespace tile_fwk
} // namespace npu
