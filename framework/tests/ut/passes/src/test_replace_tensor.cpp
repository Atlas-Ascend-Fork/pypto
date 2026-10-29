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
 * \file test_replace_tensor.cpp
 * \brief Unit test for ReplaceTensor pass.
 */

#include "gtest/gtest.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_constraint/replace_tensor.h"
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
static const uint32_t kNumZero = 0u;
static const uint32_t kNumOne = 1u;
static const uint32_t kNumTwo = 2u;
static const uint32_t kNumThree = 3u;
static const uint32_t kNumFour = 4u;
static const uint32_t kNumFive = 5u;
static const uint32_t kNumSix = 6u;
static const uint32_t kNumEight = 8u;
static const uint32_t kNumTwelve = 12u;
static const uint32_t kNumSixteen = 16u;

namespace {
ir::StmtPtr ToStmt(const Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

void CheckTokenDependencies(Function& function)
{
    const auto& dependency = function.GetVarDependency();
    for (auto& op : function.Operations(false)) {
        if (!op.result_token_.empty()) {
            EXPECT_TRUE(dependency.HasProducer(op.result_token_.front(), ToStmt(op)));
        }
        for (const auto& token : op.tokens_) {
            EXPECT_TRUE(dependency.HasConsumer(token, ToStmt(op)));
        }
    }
    for (const auto& [token, entry] : dependency.GetAllDependencies()) {
        EXPECT_FALSE(entry.producers.empty());
        EXPECT_FALSE(entry.consumers.empty());
        for (const auto& producerStmt : entry.producers) {
            auto* producer = static_cast<Operation*>(const_cast<ir::Stmt*>(producerStmt.get()));
            ASSERT_NE(producer, nullptr);
            EXPECT_EQ(producer->result_token_.front(), token);
        }
        for (const auto& consumerStmt : entry.consumers) {
            auto* consumer = static_cast<Operation*>(const_cast<ir::Stmt*>(consumerStmt.get()));
            ASSERT_NE(consumer, nullptr);
            EXPECT_TRUE(std::find(consumer->tokens_.begin(), consumer->tokens_.end(), token) !=
                        consumer->tokens_.end());
        }
    }
}

struct TokenRawReuseGraph {
    std::shared_ptr<Function> function;
    Operation* reshape2{nullptr};
    Operation* assemble1{nullptr};
    Operation* assemble2{nullptr};
    Operation* exp1{nullptr};
    Operation* exp2{nullptr};
    Operation* exp3{nullptr};
    Operation* copyInOp2{nullptr};
};

TokenRawReuseGraph BuildTokenRawReuseGraph(const std::string& name, bool reshapeBeforeExp1, bool exp2BeforeView)
{
    TokenRawReuseGraph graph;
    graph.function = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto sharedRaw = std::make_shared<RawTensor>(DT_FP32, shape);

    auto makeTensor = [&](const std::shared_ptr<RawTensor>& raw) {
        auto tensor = IRBuilder().CreateTensorVar(raw, offset, shape, CreateTestConstIntVector(shape));
        tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
        return tensor;
    };
    auto makeUniqueTensor = [&]() { return makeTensor(std::make_shared<RawTensor>(DT_FP32, shape)); };
    auto makeIncast = [&]() {
        auto tensor = makeUniqueTensor();
        graph.function->inCasts_.push_back(tensor);
        return tensor;
    };
    auto addCopyOut = [&](const LogicalTensorPtr& input) {
        auto outcast = makeUniqueTensor();
        PassOperationUtils::AddOperation(*graph.function, Opcode::OP_COPY_OUT, {input}, {outcast});
        graph.function->outCasts_.push_back(outcast);
    };

    auto incast1 = makeIncast();
    auto copyIn1 = makeUniqueTensor();
    PassOperationUtils::AddOperation(*graph.function, Opcode::OP_COPY_IN, {incast1}, {copyIn1});
    auto reshape1Out = makeUniqueTensor();
    PassOperationUtils::AddOperation(*graph.function, Opcode::OP_RESHAPE, {copyIn1}, {reshape1Out});
    graph.assemble1 = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_ASSEMBLE, {reshape1Out},
                                                        {makeTensor(sharedRaw)});
    graph.assemble1->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));

    LogicalTensorPtr branch1Out = graph.assemble1->GetOutputOperand(0);
    LogicalTensorPtr branch1Final = branch1Out;
    if (reshapeBeforeExp1) {
        auto reshape2Out = makeUniqueTensor();
        graph.reshape2 = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_RESHAPE, {branch1Out},
                                                           {reshape2Out});
        branch1Final = reshape2Out;
    }
    graph.exp1 = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_EXP, {branch1Final},
                                                   {makeUniqueTensor()});
    addCopyOut(graph.exp1->GetOutputOperand(0));

    auto incast2 = makeIncast();
    auto copyIn2 = makeUniqueTensor();
    graph.copyInOp2 = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_COPY_IN, {incast2}, {copyIn2});
    LogicalTensorPtr branch2Input = copyIn2;
    if (exp2BeforeView) {
        graph.exp2 = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_EXP, {copyIn2},
                                                       {makeUniqueTensor()});
        branch2Input = graph.exp2->GetOutputOperand(0);
    }
    auto viewOut = makeUniqueTensor();
    auto& view = PassOperationUtils::AddOperation(*graph.function, Opcode::OP_VIEW, {branch2Input}, {viewOut});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    graph.assemble2 = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_ASSEMBLE, {viewOut},
                                                        {makeTensor(sharedRaw)});
    graph.assemble2->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));

    graph.exp3 = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_EXP,
                                                   {graph.assemble2->GetOutputOperand(0)}, {makeUniqueTensor()});
    addCopyOut(graph.exp3->GetOutputOperand(0));
    return graph;
}
} // namespace

class ReplaceTensorTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(ReplaceTensorTest, RawMagicIndexTracksRepeatedSiblingMigration)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "SiblingMigration", "SiblingMigration", nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto rawA = std::make_shared<RawTensor>(DT_FP32, shape);
    auto rawB = std::make_shared<RawTensor>(DT_FP32, shape);
    auto rawC = std::make_shared<RawTensor>(DT_FP32, shape);
    auto makeTensor = [&](const std::shared_ptr<RawTensor>& raw) {
        return IRBuilder().CreateTensorVar(raw, offset, shape, CreateTestConstIntVector(shape));
    };
    auto first = makeTensor(rawA);
    auto sibling = makeTensor(rawA);
    auto existing = makeTensor(rawB);
    auto output = makeTensor(rawC);
    function->inCasts_ = {first, sibling};
    function->outCasts_ = {output};
    PassOperationUtils::AddOperation(*function, Opcode::OP_ABS, {existing}, {output});

    ReplaceTensor pass;
    pass.BuildRawMagicTensorIndex(*function);
    auto checkIndex = [&]() {
        for (const auto& raw : {rawA, rawB, rawC}) {
            EXPECT_EQ(pass.rawMagicTensors_[raw->rawmagic], GraphUtils::GetTensorsByRawMagic(*function, raw->rawmagic));
        }
    };
    checkIndex();
    pass.SetRawTensor(first, rawB);
    pass.SyncSiblingAssembleOutput(*function, first, rawA->rawmagic);
    EXPECT_EQ(sibling->tensor, rawB);
    EXPECT_EQ(pass.backRoots.size(), 1U);
    checkIndex();

    pass.SetRawTensor(first, rawC);
    pass.SyncSiblingAssembleOutput(*function, first, rawB->rawmagic);
    EXPECT_EQ(sibling->tensor, rawC);
    EXPECT_EQ(existing->tensor, rawC);
    EXPECT_EQ(pass.backRoots.size(), 3U);
    EXPECT_EQ(pass.forRoots.size(), 3U);
    checkIndex();

    Function another(Program::GetInstance(), "OtherFunction", "OtherFunction", nullptr);
    another.inCasts_ = {makeTensor(rawA)};
    pass.BuildRawMagicTensorIndex(another);
    ASSERT_EQ(pass.rawMagicTensors_.size(), 1U);
    EXPECT_EQ(pass.rawMagicTensors_.at(rawA->rawmagic), GraphUtils::GetTensorsByRawMagic(another, rawA->rawmagic));
}

TEST_F(ReplaceTensorTest, UniteTensorSkipsAssembleOutputWithoutRawTensor)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "NullRawAssemble", "NullRawAssemble", nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto rawA = std::make_shared<RawTensor>(DT_FP32, shape);
    auto rawB = std::make_shared<RawTensor>(DT_FP32, shape);
    auto makeTensor = [&](const std::shared_ptr<RawTensor>& raw) {
        return IRBuilder().CreateTensorVar(raw, offset, shape, CreateTestConstIntVector(shape));
    };
    auto assembleIn = makeTensor(rawA);
    auto assembleOut = makeTensor(rawB);
    auto output = makeTensor(rawA);
    function->inCasts_ = {assembleIn};
    function->outCasts_ = {output};
    PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {assembleIn}, {assembleOut});
    PassOperationUtils::AddOperation(*function, Opcode::OP_ABS, {assembleOut}, {output});
    // 构图完成后解除assemble输出的raw绑定, 模拟尚未绑定raw tensor的半构建状态
    assembleOut->tensor = nullptr;

    ReplaceTensor pass;
    pass.BuildRawMagicTensorIndex(*function);
    EXPECT_EQ(pass.rawMagicTensors_.find(rawB->rawmagic), pass.rawMagicTensors_.end());
    auto group = pass.rawMagicTensors_.find(rawA->rawmagic);
    ASSERT_NE(group, pass.rawMagicTensors_.end());
    EXPECT_EQ(group->second.count(assembleOut), 0U);
    auto tensorToOrderIndex = pass.BuildTensorOrderIndexMap(*function);
    UnionFind uf(tensorToOrderIndex);
    EXPECT_NO_THROW(pass.UniteTensor(*function, uf));
    // assemble自身的inplace union不受空raw兄弟守卫影响
    EXPECT_EQ(uf.Find(assembleOut), uf.Find(assembleIn));

    assembleOut->tensor = rawB;
    EXPECT_NO_THROW(pass.UniteTensor(*function, uf));
}

TEST_F(ReplaceTensorTest, TestViewAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape1,
                                                               CreateTestConstIntVector(shape1));
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset1, shape1,
                                                               CreateTestConstIntVector(shape1));
    auto copyOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto copyOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto assOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor0, offset0, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto assOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor1, offset1, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape,
                                                              CreateTestConstIntVector(shape));
    /*       Init Graph
                /————> view0 ————> copy ————> assemble \
        incast -                                        - outcast
                \————> view1 ————> copy ————> assemble /
    */
    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {viewOut0}, {copyOut0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {viewOut1}, {copyOut1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyOut0}, {assOut0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyOut1}, {assOut1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assOut0}, {outcast});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assOut1}, {outcast});
    // Init Attribute
    auto viewAttr0 = std::make_shared<ViewOpAttribute>(offset0);
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    viewOp0.SetOpAttribute(viewAttr0);
    viewOp1.SetOpAttribute(viewAttr1);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(incast->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(incast->GetRawMagic(), viewOut1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assOut0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assOut1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshape", "TestReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumOne, kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshape0 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    auto reshape1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset1, shape1,
                                                               CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    /* Init Graph
        incast -> CopyIn -> Reshape -> CopyOut -> outCast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {reshape0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {reshape0}, {reshape1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {reshape1}, {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(reshape0->GetRawMagic(), reshape1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, DdrViewReshapeFanoutSharesProducerStorage)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "DdrReshapeFanout", "DdrReshapeFanout", nullptr);
    std::vector<int64_t> shape = {1, 128};
    std::vector<int64_t> reshapeShape = {1, 1, 128};
    Offset offset(shape.size(), 0);
    Offset reshapeOffset(reshapeShape.size(), 0);
    auto makeTensor = [](const Shape& tensorShape, MemoryType memoryType) {
        auto tensor = IRBuilder().CreateTensorVar(DT_FP32, tensorShape, CreateTestConstIntVector(tensorShape));
        tensor->SetMemoryTypeBoth(memoryType, true);
        return tensor;
    };
    auto source = makeTensor(shape, MEM_UB);
    auto copyOut = makeTensor(shape, MEM_DEVICE_DDR);
    auto& copyOutOp = PassOperationUtils::AddOperation(*function, Opcode::OP_COPY_OUT, {source}, {copyOut});
    copyOutOp.UpdateSubgraphID(0);
    copyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MEM_UB, OpImmediate::Specified(offset), OpImmediate::Specified(shape), OpImmediate::Specified(shape)));

    LogicalTensors views;
    LogicalTensors reshapes;
    for (int branch = 0; branch < 5; ++branch) {
        auto view = makeTensor(shape, MEM_DEVICE_DDR);
        auto reshape = makeTensor(reshapeShape, MEM_DEVICE_DDR);
        auto copyIn = makeTensor(reshapeShape, MEM_UB);
        auto& viewOp = PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {copyOut}, {view});
        viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
        viewOp.UpdateSubgraphID(0);
        auto& reshapeOp = PassOperationUtils::AddOperation(*function, Opcode::OP_RESHAPE, {view}, {reshape});
        reshapeOp.UpdateSubgraphID(0);
        auto& copyInOp = PassOperationUtils::AddOperation(*function, Opcode::OP_COPY_IN, {reshape}, {copyIn});
        copyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(reshapeOffset), MEM_UB,
                                                                  OpImmediate::Specified(reshapeShape),
                                                                  OpImmediate::Specified(reshapeShape)));
        copyInOp.UpdateSubgraphID(branch + 1);
        views.push_back(view);
        reshapes.push_back(reshape);
    }

    ReplaceTensor pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    for (size_t branch = 0; branch < views.size(); ++branch) {
        EXPECT_EQ(views[branch]->GetRawMagic(), copyOut->GetRawMagic());
        EXPECT_EQ(reshapes[branch]->GetRawMagic(), copyOut->GetRawMagic());
        EXPECT_EQ(reshapes[branch]->GetShape(), reshapeShape);
    }
    EXPECT_EQ(copyOut->GetRawTensor()->GetRawShape(), shape);
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestIndexOutCast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexOutCast", "TestIndexOutCast",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                                CreateTestConstIntVector(shape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape,
                                                              CreateTestConstIntVector(shape));
    /* Init Graph
        incast -> Index_OutCast -> outCast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {inTensor0, inTensor1, inTensor2},
                                     {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(inTensor2);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_NE(inTensor2->GetRawMagic(), outcast->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, ForwardIndexPutUpdatesCopyInOffset)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "ForwardIndexPutUpdatesCopyInOffset",
                                                      "ForwardIndexPutUpdatesCopyInOffset", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {kNumFour, kNumFour};
    std::vector<int64_t> indexShape = {kNumFour};
    Offset rootOffset = {kNumFour, kNumZero};
    Offset originalOffset = {kNumZero, kNumZero};
    Offset copyInFromOffset = {kNumOne, kNumTwo};

    auto rootTensor = IRBuilder().CreateTensorVar(std::make_shared<RawTensor>(DT_FP32, shape), rootOffset, shape,
                                                  CreateTestConstIntVector(shape));
    auto values = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto indices = IRBuilder().CreateTensorVar(DT_INT32, indexShape, CreateTestConstIntVector(indexShape));
    auto indexPutOutput = IRBuilder().CreateTensorVar(std::make_shared<RawTensor>(DT_FP32, shape), originalOffset,
                                                      shape, CreateTestConstIntVector(shape));
    auto copyInOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& indexPut = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_PUT,
                                                      {rootTensor, values, indices}, {indexPutOutput});
    indexPut.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    auto& copyIn = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {indexPutOutput},
                                                    {copyInOutput});
    copyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(copyInFromOffset), MEM_UB,
                                                            OpImmediate::Specified(shape),
                                                            OpImmediate::Specified(shape)));

    ReplaceTensor pass;
    EXPECT_EQ(pass.ForwardInputIdx(&indexPut, rootTensor, *currFunctionPtr), SUCCESS);
    EXPECT_EQ(indexPutOutput->GetOffset(), rootOffset);

    auto copyInAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyIn.GetOpAttribute());
    ASSERT_NE(copyInAttr, nullptr);
    EXPECT_EQ(OpImmediate::ToSpecified(copyInAttr->GetFromOffset()), CreateTestConstIntVector({kNumFive, kNumTwo}));
}

TEST_F(ReplaceTensorTest, TestViewType)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewType", "TestViewType", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumTwo};
    std::vector<int64_t> shape1 = {kNumOne, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_INT8, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT8, shape, CreateTestConstIntVector(shape));
    auto viewType0 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                                CreateTestConstIntVector(shape));
    auto viewType1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset1, shape1,
                                                                CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    /* Init Graph
        incast -> CopyIn -> ViewType -> CopyOut -> outCast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {viewType0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW_TYPE, {viewType0}, {viewType1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {viewType1}, {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(viewType0->GetRawMagic(), viewType1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_True)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHasSameConsecutive_True",
                                                      "TestHasSameConsecutive_True", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor2}, {tensor3});

    // 设置操作连接
    tensor2->AddConsumer(&viewOp2);

    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_TRUE(result);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_False)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHasSameConsecutive_False",
                                                      "TestHasSameConsecutive_False", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tensor2}, {tensor3});

    // 设置操作连接
    tensor2->AddConsumer(&assembleOp);

    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_FALSE(result);
}

TEST_F(ReplaceTensorTest, TestPreCheck_FailNoSubgraphID)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPreCheck_FailNoSubgraphID",
                                                      "TestPreCheck_FailNoSubgraphID", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    // 不设置subgraph ID

    ReplaceTensor pass;
    Status result = pass.PreCheck(*currFunctionPtr);
    EXPECT_EQ(result, FAILED);
}

/*
            /————> copy ————> view ————> viewtype ————> assemble \
    incast -                                                      - outcast
            \————> copy ————> view ————> viewtype ————> assemble /
*/
TEST_F(ReplaceTensorTest, TestBackView)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewTypeRaw0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewTypeRaw1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto copy0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto copy1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto viewIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto viewIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset0, shape1,
                                                              CreateTestConstIntVector(shape1));
    auto viewTypeIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewTypeRaw0, offset0, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto viewTypeIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewTypeRaw1, offset0, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto assIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor0, offset0, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto assIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor1, offset1, shape1,
                                                             CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape,
                                                              CreateTestConstIntVector(shape));
    // Init Graph
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copy0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copy1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copy0}, {viewIn0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copy1}, {viewIn1});
    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewIn0}, {viewTypeIn0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewIn1}, {viewTypeIn1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW_TYPE, {viewTypeIn0}, {assIn0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW_TYPE, {viewTypeIn1}, {assIn1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assIn0}, {outcast});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {assIn1}, {outcast});
    // Init Attribute
    auto view_Attr0 = std::make_shared<ViewOpAttribute>(offset0);
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(offset1);
    auto ass_Attr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto ass_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    viewOp0.SetOpAttribute(view_Attr0);
    viewOp1.SetOpAttribute(view_Attr1);
    assOp0.SetOpAttribute(ass_Attr0);
    assOp1.SetOpAttribute(ass_Attr1);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(outcast->GetRawMagic(), assIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assIn1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewTypeIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewTypeIn1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewIn1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestViewPreservesLargerInputRawShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewPreservesLargerInputRawShape",
                                                      "TestViewPreservesLargerInputRawShape", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> inputShape = {kNumSix, kNumSixteen};
    std::vector<int64_t> viewShape = {kNumTwo, kNumSixteen};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto source = IRBuilder().CreateTensorVar(DT_FP32, inputShape, CreateTestConstIntVector(inputShape));
    source->SetMemoryTypeBoth(MEM_UB, true);
    auto viewInputRaw = std::make_shared<RawTensor>(DT_FP32, inputShape);
    auto viewOutputRaw = std::make_shared<RawTensor>(DT_FP32, viewShape);
    auto viewInput = IRBuilder().CreateTensorVar(viewInputRaw, offset, inputShape,
                                                 CreateTestConstIntVector(inputShape));
    auto viewOutput = IRBuilder().CreateTensorVar(viewOutputRaw, offset, viewShape,
                                                  CreateTestConstIntVector(viewShape));
    viewInput->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    viewOutput->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto copyInOutput = IRBuilder().CreateTensorVar(DT_FP32, viewShape, CreateTestConstIntVector(viewShape));
    copyInOutput->SetMemoryTypeBoth(MEM_UB, true);

    auto& copyOutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {source}, {viewInput});
    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {viewInput}, {viewOutput});
    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {viewOutput},
                                                      {copyInOutput});
    copyOutOp.UpdateSubgraphID(1);
    viewOp.UpdateSubgraphID(1);
    copyInOp.UpdateSubgraphID(2);

    copyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified(offset),
                                                               OpImmediate::Specified(inputShape),
                                                               OpImmediate::Specified(inputShape)));
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    copyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset), MEM_UB, OpImmediate::Specified(viewShape), OpImmediate::Specified(viewShape)));

    ReplaceTensor pass;
    ASSERT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    EXPECT_EQ(viewInput->GetRawTensor(), viewInputRaw);
    EXPECT_EQ(viewOutput->GetRawTensor(), viewInputRaw);
    EXPECT_EQ(viewOutput->GetRawTensor()->GetRawShape(), inputShape);
}

TEST_F(ReplaceTensorTest, TestProcessHubAssembleOp_Success)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubAssembleOp_Success",
                                                      "TestProcessHubAssembleOp_Success", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // 准备HUB-ASSEMBLE-OUTCAST链
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};

    // 创建共享的raw tensor
    std::shared_ptr<RawTensor> rawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    // 创建HUB操作相关张量
    auto hubInput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape,
                                                               CreateTestConstIntVector(shape));
    auto hubOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape,
                                                                CreateTestConstIntVector(shape));

    // 创建ASSEMBLE操作相关张量
    auto assembleOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape,
                                                                     CreateTestConstIntVector(shape));

    // 设置内存类型
    hubInput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    hubOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    assembleOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);

    // 创建操作
    auto& hubOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_HUB, {hubInput}, {hubOutput});
    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {hubOutput},
                                                        {assembleOutput});

    // 设置操作连接
    hubOutput->AddConsumer(&assembleOp);

    // 设置ASSEMBLE属性
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);

    // 将assembleOutput设置为outcast
    currFunctionPtr->outCasts_.push_back(assembleOutput);

    ReplaceTensor pass;
    pass.ProcessHubAssembleOp(*currFunctionPtr, hubOp, assembleOp, hubInput, hubOutput);

    // 验证hubInput和hubOutput共享了assembleOutput的tensor
    EXPECT_EQ(hubInput->GetRawTensor(), assembleOutput->GetRawTensor());
    EXPECT_EQ(hubOutput->GetRawTensor(), assembleOutput->GetRawTensor());
}

TEST_F(ReplaceTensorTest, TestProcessHubOpUpdateCopyOutRawShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubOpUpdateCopyOutRawShape",
                                                      "TestProcessHubOpUpdateCopyOutRawShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> fullShape = {kNumEight, kNumEight};
    std::vector<int64_t> partShape = {kNumFour, kNumEight};
    std::vector<int64_t> copyToOffset = {kNumOne, kNumZero};
    std::vector<int64_t> hubOffset = {kNumTwo, kNumZero};

    auto fullRawTensor = std::make_shared<RawTensor>(DT_FP32, fullShape);
    auto copyIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, partShape, CreateTestConstIntVector(partShape));
    copyIn->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto hubInput = npu::tile_fwk::IRBuilder().CreateTensorVar(fullRawTensor, hubOffset, partShape,
                                                               CreateTestConstIntVector(partShape));
    hubInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto hubOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, partShape,
                                                                CreateTestConstIntVector(partShape));
    hubOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto& copyOutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyIn}, {hubInput});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_HUB, {hubInput}, {hubOutput});

    auto copyOutAttr = std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, OpImmediate::Specified(copyToOffset),
                                                         OpImmediate::Specified(partShape),
                                                         OpImmediate::Specified(partShape));
    copyOutOp.SetOpAttribute(copyOutAttr);

    ReplaceTensor pass;
    EXPECT_EQ(pass.ProcessHubOp(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(OpImmediate::ToSpecified(copyOutAttr->GetToOffset()), CreateTestConstIntVector({kNumThree, kNumZero}));
    EXPECT_EQ(OpImmediate::ToSpecified(copyOutAttr->GetRawShape()), CreateTestConstIntVector(fullShape));
}

TEST_F(ReplaceTensorTest, TestA_MULACC_B)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubAssembleOp_BrokenChain",
                                                      "TestProcessHubAssembleOp_BrokenChain", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> mulAccshape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    // init LogicalTensor
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto mulAccIn = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, mulAccshape,
                                                               CreateTestConstIntVector(mulAccshape));
    auto mulAccOut = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    /* Init Graph
        incast -> Index_OutCast -> mulAccOut-> op
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MULACC_B, {inTensor0, inTensor1, mulAccIn},
                                     {mulAccOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {mulAccOut}, {outTensor});
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(mulAccIn);
    currFunctionPtr->outCasts_.push_back(outTensor);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(mulAccIn->GetRawMagic(), mulAccOut->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestSameAssembleOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSameAssembleOut",
                                                      "TestSameAssembleOut", nullptr);
    EXPECT_NE(currFunctionPtr, nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> copyInRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> outRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto copyInOut = npu::tile_fwk::IRBuilder().CreateTensorVar(copyInRawTensor, offset0, shape1,
                                                                CreateTestConstIntVector(shape1));
    copyInOut->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor1, offset0, shape1,
                                                               CreateTestConstIntVector(shape1));
    outcast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    /*       Init Graph
                            /—————> assemble -outcast1
                             /————> assemble \
        incast ————> copyIn -                 - outcast0
                             \————> assemble /
    */
    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyInOut});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {outcast1});
    // Init Attribute
    auto copyInAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset0), MEM_UB,
                                                        OpImmediate::Specified(shape), OpImmediate::Specified(shape));
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1);
    auto assAttr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset0);
    copyInOp.SetOpAttribute(copyInAttr);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    assOp2.SetOpAttribute(assAttr2);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(outcast1);
    int opSumBefore = currFunctionPtr->Operations().size();
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore + 6);
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestNotInplaceReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNotInplaceReshape",
                                                      "TestNotInplaceReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumFour, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> shape2 = {kNumFour, kNumFour};
    std::vector<int64_t> shape3 = {kNumOne, kNumFour, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> reshapeRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshapeOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(reshapeRawTensor0, offset0, shape1,
                                                                  CreateTestConstIntVector(shape1));
    auto viewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor, offset0, shape2,
                                                              CreateTestConstIntVector(shape2));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    /* Init Graph
        incast0 -> Reshape -> View -> Reshape -> outcast
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {incast}, {reshapeOut0});
    auto& viewOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {reshapeOut0}, {viewOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {viewOut}, {outcast});

    auto view_Attr = std::make_shared<ViewOpAttribute>(offset0, MEM_DEVICE_DDR);
    viewOp.SetOpAttribute(view_Attr);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_NE(viewOut->GetRawMagic(), outcast->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, UpdateCopyInAttrAfterBackAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "UpdateCopyInAttrAfterBackAssemble",
                                                      "UpdateCopyInAttrAfterBackAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {4, 4};
    std::vector<int64_t> outshape1 = {8, 4};
    std::vector<int64_t> outshape2 = {2, 4};

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyInout1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutout1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyOutout1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyInout2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape1, CreateTestConstIntVector(outshape1));
    outcast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape2, CreateTestConstIntVector(outshape2));
    outcast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    /* Init Graph
        incast -- CopyIn -- copyInout1 -- CopyOut -- copyOutOut1 -- Assemble -- outcast1
                                                                 -- CopyIn   -- copyInout2 -- CopyOut -- outcast2
    */
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyInout1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInout1}, {copyOutout1});
    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyOutout1},
                                                        {outcast1});
    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {copyOutout1},
                                                      {copyInout2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {copyInout2}, {outcast2});

    Offset assembleToOffset = {4, 0};
    auto assembleOpAttribute = std::make_shared<AssembleOpAttribute>(assembleToOffset,
                                                                     CreateTestConstIntVector(assembleToOffset));
    assembleOp.SetOpAttribute(assembleOpAttribute);
    Offset copyIn2FromOffset = {2, 0};
    auto copyInOpAttribute = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(copyIn2FromOffset), MEM_UB, OpImmediate::Specified(copyInout2->GetShape()),
        OpImmediate::Specified(copyInout2->tensor->GetDynRawShape()),
        OpImmediate::Specified(copyInout2->GetDynValidShape()));
    copyInOp.SetOpAttribute(copyInOpAttribute);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);

    ReplaceTensor replaceTensorPass;
    int opSumBefore = currFunctionPtr->Operations().size();
    replaceTensorPass.RunOnFunction(*currFunctionPtr);
    int opSumExpAfter = opSumBefore + 2;
    EXPECT_EQ(replaceTensorPass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumExpAfter);
}

/*
 * 验证场景：SHMEM_WAIT_UNTIL 的输出 tensor 被多个 ASSEMBLE 消费且输出到不同 outcast 时，
 * ReplaceTensor 不插入 COPY_IN/COPY_OUT。
 */
TEST_F(ReplaceTensorTest, TestShmemWaitUntilWithDiffAssembleOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestShmemWaitUntilAssemble",
                                                      "TestShmemWaitUntilAssemble", nullptr);
    EXPECT_NE(currFunctionPtr, nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    std::shared_ptr<RawTensor> inRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> inRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> shmemRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto incast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor0, offset0, shape);
    incast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor1, offset0, shape);
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto shmemOut = std::make_shared<LogicalTensor>(*currFunctionPtr, shmemRawTensor, offset0, shape);
    shmemOut->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor0, offset0, shape);
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor1, offset1, shape1);
    outcast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SHMEM_WAIT_UNTIL, {incast0, incast1}, {shmemOut});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {shmemOut}, {outcast0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {shmemOut}, {outcast1});
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(outcast1);
    ReplaceTensor pass;
    int opSumBefore = currFunctionPtr->Operations().size();
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore);
    for (auto& op : currFunctionPtr->Operations()) {
        EXPECT_TRUE(op.GetOpcode() != Opcode::OP_COPY_IN && op.GetOpcode() != Opcode::OP_COPY_OUT);
    }
    EXPECT_EQ(shmemOut->GetRawMagic(), outcast0->GetRawMagic());
    EXPECT_EQ(shmemOut->GetRawMagic(), outcast1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

// 场景：L0C2UB copy 携带非零 fromOffset，下游 assemble 又携带 toOffset，
// 源与目的同时携带偏移无法折叠，pass 应报错返回 FAILED
TEST_F(ReplaceTensorTest, FoldL0C2UBCopyOffsetSrcDstConflictFail)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "FoldL0C2UBOffsetConflict",
                                                      "FoldL0C2UBOffsetConflict", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    const std::vector<int64_t> l0cShape = {64, 128};
    const std::vector<int64_t> bigShape = {128, 128};
    const std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    const std::vector<int64_t> srcOffset = {kNumZero, 32};
    const std::vector<int64_t> dstOffset = {kNumZero, 64};

    auto l0a = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0cRaw = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0c = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw, offset0, l0cShape,
                                                          CreateTestConstIntVector(l0cShape));
    l0c->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto ub = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    ub->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto ubBigRaw = std::make_shared<RawTensor>(DT_FP32, bigShape);
    auto ubBig = npu::tile_fwk::IRBuilder().CreateTensorVar(ubBigRaw, offset0, bigShape,
                                                            CreateTestConstIntVector(bigShape));
    ubBig->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto mulIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a, l0b}, {l0c});
    auto& copyOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c}, {ub});
    auto& assOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub}, {ubBig});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubBig, mulIn}, {mulOut});

    // copy 携带非零 fromOffset [0,32]（其余属性与 GenerateMoveOp 产物对齐）
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcOffset), MemoryType::MEM_UB, OpImmediate::Specified(l0cShape),
        OpImmediate::Specified(l0cShape), OpImmediate::Specified(CreateTestConstIntVector(l0cShape)));
    copyAttr->SetToOffset(OpImmediate::Specified(offset0));
    copyOp.SetOpAttribute(copyAttr);
    copyOp.SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    // assemble 携带非零 toOffset [0,64]
    assOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, dstOffset));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), FAILED);
}

// 场景：两个 matmul 结果经 L0C_COPY_UB(EXTRACT) 搬到 UB，各自 ASSEMBLE 到同一个大 tensor 供 MUL 使用。
// ASSEMBLE toOffset 非零([0,64])且 copy fromOffset 全零 → 折叠为 INSERT 并携带 toOffset；
// ASSEMBLE toOffset 全零([0,0]) → 维持 EXTRACT 不动作（参考 chunk_kda tensor 520 计算路径）
TEST_F(ReplaceTensorTest, FoldL0C2UBCopyOffsetForDualAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "FoldL0C2UBCopyOffset",
                                                      "FoldL0C2UBCopyOffset", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    const std::vector<int64_t> l0cShape = {64, 128};
    const std::vector<int64_t> bigShape = {128, 128};
    const std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    const std::vector<int64_t> offset1 = {kNumZero, 64};

    // l0a/l0b 输入与两路 matmul 的 L0C 输出
    auto l0a1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0a2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0cRaw1 = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0cRaw2 = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0c1 = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw1, offset0, l0cShape,
                                                           CreateTestConstIntVector(l0cShape));
    auto l0c2 = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw2, offset0, l0cShape,
                                                           CreateTestConstIntVector(l0cShape));
    l0c1->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    l0c2->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    // L0C2UB copy 输出的 UB 小 tensor
    auto ub1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto ub2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    ub1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    ub2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    // assemble 汇聚的大 UB tensor 与 MUL 侧
    auto ubBigRaw = std::make_shared<RawTensor>(DT_FP32, bigShape);
    auto ubBig = npu::tile_fwk::IRBuilder().CreateTensorVar(ubBigRaw, offset0, bigShape,
                                                            CreateTestConstIntVector(bigShape));
    ubBig->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto mulIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a1, l0b1}, {l0c1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a2, l0b2}, {l0c2});
    auto& copyOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c1}, {ub1});
    auto& copyOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c2}, {ub2});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub1}, {ubBig});
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub2}, {ubBig});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubBig, mulIn}, {mulOut});

    // copy 属性对齐 GenerateMoveOp::SetL0C2UBCopyAttr 产物：EXTRACT + 零 toOffset
    auto makeCopyAttr = [&l0cShape, &offset0](Operation& op) {
        auto attr = std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset0), MemoryType::MEM_UB, OpImmediate::Specified(l0cShape),
            OpImmediate::Specified(l0cShape), OpImmediate::Specified(CreateTestConstIntVector(l0cShape)));
        attr->SetToOffset(OpImmediate::Specified(offset0));
        op.SetOpAttribute(attr);
        op.SetAttr(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    };
    makeCopyAttr(copyOp1);
    makeCopyAttr(copyOp2);
    // assemble1 写大 tensor 偏移 [0,64]，assemble2 写 [0,0]
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));
    assOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));
    const auto assOp1Magic = assOp1.GetOpMagic();
    const auto assOp2Magic = assOp2.GetOpMagic();

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // toOffset 非零分支：折叠为 INSERT，copy toOffset 携带 assemble 偏移
    EXPECT_EQ(copyOp1.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::INSERT));
    auto foldedAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp1.GetOpAttribute());
    ASSERT_NE(foldedAttr, nullptr);
    const auto& foldedToOffset = foldedAttr->GetToOffset();
    ASSERT_EQ(foldedToOffset.size(), offset1.size());
    for (size_t i = 0; i < offset1.size(); i++) {
        EXPECT_EQ(foldedToOffset[i].GetSpecifiedValue().Raw()->GetImmediateValue(), offset1[i]);
    }
    // toOffset 全零分支：也折叠为 INSERT，直接写入原 Assemble 输出
    EXPECT_EQ(copyOp2.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::INSERT));
    auto foldedAttr2 = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp2.GetOpAttribute());
    ASSERT_NE(foldedAttr2, nullptr);
    const auto& foldedToOffset2 = foldedAttr2->GetToOffset();
    for (size_t i = 0; i < offset0.size(); i++) {
        EXPECT_EQ(foldedToOffset2[i].GetSpecifiedValue().Raw()->GetImmediateValue(), offset0[i]);
    }

    bool assemble1Present = false;
    bool assemble2Present = false;
    for (const auto& op : currFunctionPtr->Operations()) {
        assemble1Present = assemble1Present || op.GetOpMagic() == assOp1Magic;
        assemble2Present = assemble2Present || op.GetOpMagic() == assOp2Magic;
    }
    EXPECT_FALSE(assemble1Present);
    EXPECT_FALSE(assemble2Present);
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

// Dual assemble L0C2UB fold: output tensor's dynValidShape must stay [128,128] (not [64,128]).
// The dynValidShape assertion is deterministic. The traversal-snapshot fix is exercised here,
// but iterator invalidation is UB: it only manifests deterministically under ASan/debug builds.
TEST_F(ReplaceTensorTest, FoldL0C2UBCopyOffsetPreservesAssembledValidShape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "FoldPreserveValidShape",
                                                      "FoldPreserveValidShape", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    const std::vector<int64_t> l0cShape = {64, 128};
    const std::vector<int64_t> bigShape = {128, 128};
    const std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    const std::vector<int64_t> offset1 = {kNumZero, 64};

    auto l0a1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0a2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0cRaw1 = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0cRaw2 = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0c1 = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw1, offset0, l0cShape,
                                                           CreateTestConstIntVector(l0cShape));
    auto l0c2 = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw2, offset0, l0cShape,
                                                           CreateTestConstIntVector(l0cShape));
    l0c1->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    l0c2->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto ub1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto ub2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    ub1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    ub2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    ub1->UpdateDynValidShape(CreateTestConstIntVector(l0cShape));
    ub2->UpdateDynValidShape(CreateTestConstIntVector(l0cShape));
    auto ubBigRaw = std::make_shared<RawTensor>(DT_FP32, bigShape);
    auto ubBig = npu::tile_fwk::IRBuilder().CreateTensorVar(ubBigRaw, offset0, bigShape,
                                                            CreateTestConstIntVector(bigShape));
    ubBig->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto mulIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));

    // 构图：A_MUL_B → L0C_COPY_UB → ASSEMBLE(×2, 不同 toOffset) → 大 tensor → MUL
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a1, l0b1}, {l0c1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a2, l0b2}, {l0c2});
    auto& copyOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c1}, {ub1});
    auto& copyOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c2}, {ub2});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub1}, {ubBig});
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub2}, {ubBig});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubBig, mulIn}, {mulOut});

    // AddOperation(OP_ASSEMBLE) 触发 InferShape 会强制覆盖输出的 dynValidShape 为输入形状,
    // 必须在所有 AddOperation 之后再设置 assemble 输出的正确有效形状(多块并集)
    ubBig->UpdateDynValidShape(CreateTestConstIntVector(bigShape));

    auto makeCopyAttr = [&l0cShape, &offset0](Operation& op) {
        auto attr = std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset0), MemoryType::MEM_UB, OpImmediate::Specified(l0cShape),
            OpImmediate::Specified(l0cShape), OpImmediate::Specified(CreateTestConstIntVector(l0cShape)));
        attr->SetToOffset(OpImmediate::Specified(offset0));
        op.SetOpAttribute(attr);
        op.SetAttr(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    };
    makeCopyAttr(copyOp1);
    makeCopyAttr(copyOp2);
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));
    assOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));
    const auto assOp1Magic = assOp1.GetOpMagic();
    const auto assOp2Magic = assOp2.GetOpMagic();

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    bool assemble1Present = false;
    bool assemble2Present = false;
    for (const auto& op : currFunctionPtr->Operations()) {
        assemble1Present = assemble1Present || op.GetOpMagic() == assOp1Magic;
        assemble2Present = assemble2Present || op.GetOpMagic() == assOp2Magic;
    }
    EXPECT_FALSE(assemble1Present);
    EXPECT_FALSE(assemble2Present);
    EXPECT_EQ(copyOp1.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::INSERT));
    EXPECT_EQ(copyOp2.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::INSERT));

    const auto& bigValid = ubBig->GetDynValidShape();
    ASSERT_EQ(bigValid.size(), bigShape.size());
    for (size_t i = 0; i < bigShape.size(); i++) {
        ASSERT_TRUE(bigValid[i].ConcreteValid());
        EXPECT_EQ(bigValid[i].Concrete(), bigShape[i]);
    }
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

// 场景：copy 输出同时被 ASSEMBLE 和普通算子（MUL）消费，consumers 非唯一触发跳过守卫
TEST_F(ReplaceTensorTest, FoldL0C2UBCopyOffsetSkipForMultiConsumer)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "FoldL0C2UBSkipMultiCons",
                                                      "FoldL0C2UBSkipMultiCons", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);
    const std::vector<int64_t> l0cShape = {64, 128};
    const std::vector<int64_t> bigShape = {128, 128};
    const std::vector<int64_t> offset0 = {kNumZero, kNumZero};

    auto l0a = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0b = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto l0cRaw = std::make_shared<RawTensor>(DT_FP32, l0cShape);
    auto l0c = npu::tile_fwk::IRBuilder().CreateTensorVar(l0cRaw, offset0, l0cShape,
                                                          CreateTestConstIntVector(l0cShape));
    l0c->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto ub = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    ub->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto ubBigRaw = std::make_shared<RawTensor>(DT_FP32, bigShape);
    auto ubBig = npu::tile_fwk::IRBuilder().CreateTensorVar(ubBigRaw, offset0, bigShape,
                                                            CreateTestConstIntVector(bigShape));
    ubBig->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto mulIn = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, bigShape, CreateTestConstIntVector(bigShape));
    auto mulIn2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));
    auto mulOut2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, l0cShape, CreateTestConstIntVector(l0cShape));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a, l0b}, {l0c});
    auto& copyOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0C_COPY_UB, {l0c}, {ub});
    auto& assOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ub}, {ubBig});
    // copy 输出同时被普通算子直接消费：fold 改写 toOffset 会使其读错位
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ub, mulIn2}, {mulOut2});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubBig, mulIn}, {mulOut});

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset0), MemoryType::MEM_UB, OpImmediate::Specified(l0cShape),
        OpImmediate::Specified(l0cShape), OpImmediate::Specified(CreateTestConstIntVector(l0cShape)));
    copyAttr->SetToOffset(OpImmediate::Specified(offset0));
    copyOp.SetOpAttribute(copyAttr);
    copyOp.SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    assOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // consumers 非唯一跳过折叠：copy 维持 EXTRACT，toOffset 保持全零不被覆盖
    EXPECT_EQ(copyOp.GetIntAttribute(OpAttributeKey::localCopyLocalMode),
              static_cast<int64_t>(Matrix::CopyMode::EXTRACT));
    auto skipAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp.GetOpAttribute());
    ASSERT_NE(skipAttr, nullptr);
    const auto& skipToOffset = skipAttr->GetToOffset();
    for (size_t i = 0; i < offset0.size(); i++) {
        EXPECT_EQ(skipToOffset[i].GetSpecifiedValue().Raw()->GetImmediateValue(), offset0[i]);
    }
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * A5(DAV_3510)场景下A_MULACC_B支持最多5个输入
 */
TEST_F(ReplaceTensorTest, TestA_MULACC_B_5Inputs_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestA_MULACC_B_5Inputs_A5",
                                                      "TestA_MULACC_B_5Inputs_A5", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> mulAccshape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto mulAccIn = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, mulAccshape,
                                                               CreateTestConstIntVector(mulAccshape));
    auto bias = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape, CreateTestConstIntVector(mulAccshape));
    auto deqScale = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                               CreateTestConstIntVector(mulAccshape));
    auto mulAccOut = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape,
                                                                CreateTestConstIntVector(mulAccshape));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_A_MULACC_B,
                                     {inTensor0, inTensor1, mulAccIn, bias, deqScale}, {mulAccOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {mulAccOut}, {outTensor});
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.insert(currFunctionPtr->inCasts_.end(), {inTensor0, inTensor1, mulAccIn, bias, deqScale});
    currFunctionPtr->outCasts_.push_back(outTensor);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(mulAccIn->GetRawMagic(), mulAccOut->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

/*
 * 新图表达: 两个assemble写入两个不同logical tensor(共享同一rawMagic), 验证pass后
 * 两个输出及对应输入的rawMagic统一到同一地址。
 *
 * 旧图: incast -> view0 -> copy0 -> assemble0 -> outcast (共享)
 *                       -> view1 -> copy1 -> assemble1 -> outcast (共享)
 *
 * 新图: incast -> view0 -> copy0 -> assemble0 -> out0 (rawMagic=R)
 *                       -> view1 -> copy1 -> assemble1 -> out1 (rawMagic=R)
 *       out0 和 out1 共享同一 rawMagic, 但为不同 logical tensor
 */
TEST_F(ReplaceTensorTest, TestMultiAssembleSameRawMagic)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMultiAssembleSameRawMagic",
                                                      "TestMultiAssembleSameRawMagic", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    // 新图表达: out0和out1共享同一rawMagic
    std::shared_ptr<RawTensor> sharedOutRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset1, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out0 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset0, shape,
                                                           CreateTestConstIntVector(shape));
    out0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset1, shape,
                                                           CreateTestConstIntVector(shape));
    out1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut0}, {out0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut1}, {out1});

    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1));
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0));
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1));

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(out0);
    currFunctionPtr->outCasts_.push_back(out1);

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // 两个输出应统一到同一 rawMagic
    EXPECT_EQ(out0->GetRawMagic(), out1->GetRawMagic());
    // assemble 输入也应统一到同一 rawMagic
    EXPECT_EQ(out0->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(out0->GetRawMagic(), viewOut1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * 新图表达: 两个assemble写共享rawMagic的不同logical tensor, 同一输入,
 * 且两个assemble的offset不同(写同一地址的不同位置)。
 * 验证 FindNeedToCopyAssemble 用 rawMagic 比较后不会误插入copy
 * (旧逻辑用 magic 比较会误判输出不同而插入copy)。
 *
 * Graph:
 *                                 /————> Assemble(offset0) -> out0 (rawMagic R, offset0)
 *   incast -> CopyIn -> copyInOut
 *                                 \————> Assemble(offset1) -> out1 (rawMagic R, offset1)
 *   out0 和 out1 共享同一 rawMagic, 但为不同 logical tensor (offset不同)
 */
TEST_F(ReplaceTensorTest, TestMultiAssembleSameInputNoExtraCopy)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMultiAssembleNoCopy",
                                                      "TestMultiAssembleNoCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> copyInRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> sharedOutRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto copyInOut = npu::tile_fwk::IRBuilder().CreateTensorVar(copyInRawTensor, offset0, shape1,
                                                                CreateTestConstIntVector(shape1));
    copyInOut->SetMemoryTypeBoth(MEM_UB, true);
    auto out0 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset0, shape1,
                                                           CreateTestConstIntVector(shape1));
    out0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset1, shape1,
                                                           CreateTestConstIntVector(shape1));
    out1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);

    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyInOut});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {out0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {copyInOut}, {out1});

    copyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset0), MEM_UB, OpImmediate::Specified(shape), OpImmediate::Specified(shape)));
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_UB, offset0));
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_UB, offset1));

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(out0);
    currFunctionPtr->outCasts_.push_back(out1);

    int opSumBefore = currFunctionPtr->Operations().size();
    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    // 两个assemble输出共享rawMagic(同一地址), 不应误插入copy
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore) << "同rawMagic的assemble输出不应触发额外copy插入";
    EXPECT_EQ(out0->GetRawMagic(), out1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * 新图表达: 两个assemble写入两个不同logical tensor(共享同一rawMagic), assemble输出后接reshape。
 * 验证backward处理reshape时, SyncSiblingAssembleOutput 能同步兄弟logicaltensor并继续遍历。
 *
 * Graph:
 *   incast -> view0 -> assemble0 -> out0 (rawMagic R) -> reshape -> reshapeOut -> copyOut -> outcast0
 *         -> view1 -> assemble1 -> out1 (rawMagic R) -> outcast1
 *   out0 和 out1 共享同一 rawMagic, 但为不同 logical tensor
 *
 * 当 backward 从 reshapeOut 处理 reshape 时, out0 被更新, 需同步 out1 (兄弟),
 * 使 out1 的 producer (assemble1) 也被 backward 处理。
 */
TEST_F(ReplaceTensorTest, TestSyncSiblingAssembleOutputReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSyncSiblingReshape",
                                                      "TestSyncSiblingReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumSixteen, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};

    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    // 新图表达: out0和out1共享同一rawMagic(同一RawTensor对象)
    std::shared_ptr<RawTensor> sharedOutRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> reshapeRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset1, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out0 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset0, shape,
                                                           CreateTestConstIntVector(shape));
    out0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset1, shape,
                                                           CreateTestConstIntVector(shape));
    out1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    // reshape 输入输出共享同一 rawMagic 才会 inplace, reshapeOut 不是 outcast
    auto reshapeOut = npu::tile_fwk::IRBuilder().CreateTensorVar(reshapeRawTensor, offset0, shape1,
                                                                 CreateTestConstIntVector(shape1));
    reshapeOut->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut0}, {out0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut1}, {out1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {out0}, {reshapeOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {reshapeOut}, {outcast0});

    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1));
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0));
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1));

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(out1);

    // pass 前记录 reshape 输出的 rawMagic, pass 后用于对比验证 reshape 输出成为复用节点
    int reshapeOutRawMagicBefore = reshapeOut->GetRawMagic();
    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // 两个assemble输出应统一到同一 rawMagic
    EXPECT_EQ(out0->GetRawMagic(), out1->GetRawMagic());
    // assemble输入也应统一到同一 rawMagic (说明兄弟链路被遍历)
    EXPECT_EQ(out0->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(out0->GetRawMagic(), viewOut1->GetRawMagic());
    // reshape 输出应复用 out0 的 rawMagic, 成为复用节点
    EXPECT_EQ(reshapeOut->GetRawMagic(), out0->GetRawMagic());
    EXPECT_NE(reshapeOutRawMagicBefore, reshapeOut->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * 新图表达: 两个assemble写入两个不同logical tensor(共享同一rawMagic), assemble输出后接index_outcast。
 * 验证backward处理index_outcast时, SyncSiblingAssembleOutput 能同步兄弟logicaltensor并继续遍历。
 *
 * Graph:
 *   incast -> view0 -> assemble0 -> out0 (rawMagic R) -- (index 2 of index_outcast)
 *                                                            \
 *   inIdx0, inIdx1 -----------------------------------------> index_outcast -> idxOut (outcast0)
 *   incast -> view1 -> assemble1 -> out1 (rawMagic R) -> outcast1
 *   out0 和 out1 共享同一 rawMagic, 但为不同 logical tensor
 *
 * 当 backward 从 idxOut 处理 index_outcast 时, out0 (input idx 2) 被更新,
 * 需同步 out1 (兄弟), 使 out1 的 producer (assemble1) 也被 backward 处理。
 */
TEST_F(ReplaceTensorTest, TestSyncSiblingAssembleOutputIndexOutcast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSyncSiblingIdxOutcast",
                                                      "TestSyncSiblingIdxOutcast", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};

    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> sharedOutRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> idxOutRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset1, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out0 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset0, shape,
                                                           CreateTestConstIntVector(shape));
    out0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset1, shape,
                                                           CreateTestConstIntVector(shape));
    out1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    // index_outcast 需要3个输入: index0, index1, data(out0)
    auto inIdx0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shape, CreateTestConstIntVector(shape));
    inIdx0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto inIdx1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shape, CreateTestConstIntVector(shape));
    inIdx1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto idxOut = npu::tile_fwk::IRBuilder().CreateTensorVar(idxOutRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    idxOut->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut0}, {out0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut1}, {out1});
    // index_outcast: {inIdx0, inIdx1, out0} -> idxOut, inplace pair = {2, 0}
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {inIdx0, inIdx1, out0}, {idxOut});

    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1));
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0));
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1));

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(idxOut);
    currFunctionPtr->outCasts_.push_back(out1);

    // pass 前记录 idxOut 的 rawMagic, pass 后用于对比验证 idxOut 未成为复用节点 (index_outcast 非inplace)
    int idxOutRawMagicBefore = idxOut->GetRawMagic();
    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // 两个assemble输出应统一到同一 rawMagic
    EXPECT_EQ(out0->GetRawMagic(), out1->GetRawMagic());
    // assemble输入也应统一到同一 rawMagic (说明兄弟链路被遍历)
    EXPECT_EQ(out0->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(out0->GetRawMagic(), viewOut1->GetRawMagic());
    // idxOut 不应复用 out0 的 rawMagic (index_outcast 非inplace, 与reshape不同)
    EXPECT_NE(idxOut->GetRawMagic(), out0->GetRawMagic());
    EXPECT_EQ(idxOutRawMagicBefore, idxOut->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * 新图表达: 两个assemble写入两个不同logical tensor(共享同一rawMagic), assemble输出后接view。
 * 验证backward处理view时, SyncSiblingAssembleOutput 能同步兄弟logicaltensor并继续遍历。
 *
 * Graph:
 *   incast -> view0 -> assemble0 -> out0 (rawMagic R) -> view2 -> viewOut2 (outcast0)
 *          -> view1 -> assemble1 -> out1 (rawMagic R) -> outcast1
 *   out0 和 out1 共享同一 rawMagic, 但为不同 logical tensor
 */
TEST_F(ReplaceTensorTest, TestSyncSiblingAssembleOutputView)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSyncSiblingView",
                                                      "TestSyncSiblingView", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};

    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> sharedOutRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset1, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out0 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset0, shape,
                                                           CreateTestConstIntVector(shape));
    out0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset1, shape,
                                                           CreateTestConstIntVector(shape));
    out1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut2 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor2, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut2->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {viewOut1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut0}, {out0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut1}, {out1});
    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {out0}, {viewOut2});

    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1));
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0));
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1));
    viewOp2.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(viewOut2);
    currFunctionPtr->outCasts_.push_back(out1);

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // 两个assemble输出应统一到同一 rawMagic
    EXPECT_EQ(out0->GetRawMagic(), out1->GetRawMagic());
    // assemble输入也应统一到同一 rawMagic
    EXPECT_EQ(out0->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(out0->GetRawMagic(), viewOut1->GetRawMagic());
    // view输出也应与输入统一
    EXPECT_EQ(out0->GetRawMagic(), viewOut2->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * 新图表达: 两个assemble写入共享rawMagic的不同logical tensor, base为outcast(out1)而非incast。
 * 验证base入队后SyncSiblingAssembleOutput同步兄弟(out0), 使out0的consumer(reshape)也被遍历。
 *
 * Graph (CopyIn打断inplace链, 使incast不在group中, base为outcast):
 *   incast -> CopyIn -> copyOut0 -> view0 -> assemble0 -> out0 (rawMagic R) -> reshape -> reshapeOut -> copyOut ->
 * outcast0
 *         -> CopyIn -> copyOut1 -> view1 -> assemble1 -> out1 (rawMagic R, outcast1)
 *   out0 和 out1 共享同一 rawMagic
 */
TEST_F(ReplaceTensorTest, TestSyncSiblingBaseIsOutcast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSyncSiblingBaseOutcast",
                                                      "TestSyncSiblingBaseOutcast", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumSixteen, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};

    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> copyRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> copyRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> sharedOutRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> reshapeRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape,
                                                             CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto copyOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(copyRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    copyOut0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto copyOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(copyRawTensor1, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    copyOut1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(copyRawTensor0, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(copyRawTensor1, offset0, shape,
                                                               CreateTestConstIntVector(shape));
    viewOut1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out0 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset0, shape,
                                                           CreateTestConstIntVector(shape));
    out0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(sharedOutRawTensor, offset1, shape,
                                                           CreateTestConstIntVector(shape));
    out1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto reshapeOut = npu::tile_fwk::IRBuilder().CreateTensorVar(reshapeRawTensor, offset0, shape1,
                                                                 CreateTestConstIntVector(shape1));
    reshapeOut->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);

    // CopyIn 打断 inplace 链: incast 不在 group 中, base 将是 outcast
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyOut0});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {copyOut1});
    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {copyOut0}, {viewOut0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {copyOut1}, {viewOut1});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut0}, {out0});
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {viewOut1}, {out1});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {out0}, {reshapeOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {reshapeOut}, {outcast0});

    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0));
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0));
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1));

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(out1);

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // 两个assemble输出应统一到同一 rawMagic (base=out1同步了兄弟out0)
    EXPECT_EQ(out0->GetRawMagic(), out1->GetRawMagic());
    // assemble输入也应统一到同一 rawMagic (说明兄弟链路被遍历)
    EXPECT_EQ(out0->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(out0->GetRawMagic(), viewOut1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

/*
 * 用例1:
 *   incast1 -> copyin1 -> reshape1 -> assemble1 -> reshape2 -> exp1 -> copyout1 -> outcast1
 *   incast2 -> copyin2 -> view -> assemble2 -> exp3 -> copyout2 -> outcast2
 *
 * assemble1 和 assemble2 的输出共享同一个 RawTensor。
 * Before: reshape2 --produce T0--> assemble2
 * After:  exp1 --produce T1--> assemble2
 */
TEST_F(ReplaceTensorTest, TestTokenProducerMove)
{
    auto graph = BuildTokenRawReuseGraph("TokenProducerMove", true, false);
    ASSERT_NE(graph.reshape2, nullptr);
    ASSERT_NE(graph.exp1, nullptr);
    ASSERT_NE(graph.assemble1, nullptr);
    ASSERT_NE(graph.assemble2, nullptr);

    IRBuilder builder;
    auto originalToken = builder.CreateTokenVar(ir::Span::Unknown());
    graph.reshape2->result_token_ = {originalToken};
    graph.assemble2->tokens_.push_back(originalToken);
    graph.function->GetVarDependency().AddProducer(originalToken, ToStmt(*graph.reshape2));
    graph.function->GetVarDependency().AddConsumer(originalToken, ToStmt(*graph.assemble2));

    ReplaceTensor pass;
    pass.RunOnFunction(*graph.function);
    EXPECT_EQ(graph.assemble1->GetOutputOperand(0)->GetRawTensor(),
              graph.assemble2->GetOutputOperand(0)->GetRawTensor());

    ASSERT_FALSE(graph.exp1->result_token_.empty());
    EXPECT_NE(graph.exp1->result_token_.front(), originalToken);
    EXPECT_EQ(graph.assemble2->tokens_.size(), 0);
    EXPECT_EQ(graph.copyInOp2->tokens_.size(), 1);
    EXPECT_EQ(graph.assemble2->tokens_.front(), graph.exp1->result_token_.front());
    EXPECT_TRUE(graph.reshape2->result_token_.empty());
    EXPECT_FALSE(graph.function->GetVarDependency().HasDependency(originalToken));
    CheckTokenDependencies(*graph.function);
}

/*
 * 用例2:
 *   incast1 -> copyin1 -> reshape1 -> assemble1 -> exp1 -> copyout1 -> outcast1
 *   incast2 -> copyin2 -> exp2 -> view -> assemble2 -> exp3 -> copyout2 -> outcast2
 *
 * assemble1 和 assemble2 的输出共享同一个 RawTensor。
 * Before: exp1 --produce T--> assemble2
 * After:  exp1 --produce T--> exp2
 */
TEST_F(ReplaceTensorTest, TestTokenConsumerMove)
{
    auto graph = BuildTokenRawReuseGraph("TokenConsumerMove", false, true);
    ASSERT_NE(graph.exp1, nullptr);
    ASSERT_NE(graph.exp2, nullptr);
    ASSERT_NE(graph.assemble1, nullptr);
    ASSERT_NE(graph.assemble2, nullptr);

    IRBuilder builder;
    auto token = builder.CreateTokenVar(ir::Span::Unknown());
    graph.exp1->result_token_ = {token};
    graph.assemble2->tokens_.push_back(token);
    graph.function->GetVarDependency().AddProducer(token, ToStmt(*graph.exp1));
    graph.function->GetVarDependency().AddConsumer(token, ToStmt(*graph.assemble2));

    EXPECT_EQ(graph.assemble1->GetOutputOperand(0)->GetRawTensor(),
              graph.assemble2->GetOutputOperand(0)->GetRawTensor());

    ReplaceTensor pass;
    pass.RunOnFunction(*graph.function);
    EXPECT_EQ(graph.assemble1->GetOutputOperand(0)->GetRawTensor(),
              graph.assemble2->GetOutputOperand(0)->GetRawTensor());

    EXPECT_EQ(graph.exp1->result_token_.front(), token);
    EXPECT_TRUE(graph.assemble2->tokens_.empty());
    ASSERT_EQ(graph.exp2->tokens_.size(), 1);
    EXPECT_EQ(graph.exp2->tokens_.front(), token);
    EXPECT_TRUE(graph.function->GetVarDependency().HasProducer(token, ToStmt(*graph.exp1)));
    EXPECT_TRUE(graph.function->GetVarDependency().HasConsumer(token, ToStmt(*graph.exp2)));
    EXPECT_FALSE(graph.function->GetVarDependency().HasConsumer(token, ToStmt(*graph.assemble2)));
    CheckTokenDependencies(*graph.function);
}

/*
 * 用例3:
 *   incast1 -> copyin1 -> reshape1 -> assemble1 -> reshape2 -> exp1 -> copyout1 -> outcast1
 *   incast2 -> copyin2 -> exp2 -> view -> assemble2 -> exp3 -> copyout2 -> outcast2
 *
 * assemble1 和 assemble2 的输出共享同一个 RawTensor。
 * Before: reshape2 --produce T0--> assemble2
 * After:  exp1 --produce T1--> exp2
 */
TEST_F(ReplaceTensorTest, TestTokenProducerAndConsumerMove)
{
    auto graph = BuildTokenRawReuseGraph("TokenProducerAndConsumerMove", true, true);
    ASSERT_NE(graph.reshape2, nullptr);
    ASSERT_NE(graph.exp1, nullptr);
    ASSERT_NE(graph.exp2, nullptr);
    ASSERT_NE(graph.assemble1, nullptr);
    ASSERT_NE(graph.assemble2, nullptr);

    IRBuilder builder;
    auto originalToken = builder.CreateTokenVar(ir::Span::Unknown());
    graph.reshape2->result_token_ = {originalToken};
    graph.assemble2->tokens_.push_back(originalToken);
    graph.function->GetVarDependency().AddProducer(originalToken, ToStmt(*graph.reshape2));
    graph.function->GetVarDependency().AddConsumer(originalToken, ToStmt(*graph.assemble2));

    ReplaceTensor pass;
    pass.RunOnFunction(*graph.function);
    EXPECT_EQ(graph.assemble1->GetOutputOperand(0)->GetRawTensor(),
              graph.assemble2->GetOutputOperand(0)->GetRawTensor());

    ASSERT_FALSE(graph.exp1->result_token_.empty());
    EXPECT_NE(graph.exp1->result_token_.front(), originalToken);
    EXPECT_EQ(graph.exp2->tokens_.size(), 1);
    EXPECT_EQ(graph.exp2->tokens_.front(), graph.exp1->result_token_.front());
    EXPECT_TRUE(graph.reshape2->result_token_.empty());
    EXPECT_TRUE(graph.assemble2->tokens_.empty());
    EXPECT_TRUE(graph.function->GetVarDependency().HasProducer(graph.exp1->result_token_.front(), ToStmt(*graph.exp1)));
    EXPECT_TRUE(graph.function->GetVarDependency().HasConsumer(graph.exp1->result_token_.front(), ToStmt(*graph.exp2)));
    EXPECT_FALSE(graph.function->GetVarDependency().HasDependency(originalToken));
    CheckTokenDependencies(*graph.function);
}

TEST_F(ReplaceTensorTest, DoesNotMoveTokenProducerToItsConsumers)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TokenConsumerBoundary", "TokenConsumerBoundary",
                                               nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto sharedRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    auto makeTensor = [&](const std::shared_ptr<RawTensor>& raw) {
        auto tensor = IRBuilder().CreateTensorVar(raw, offset, shape, CreateTestConstIntVector(shape));
        tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
        return tensor;
    };

    auto incast = makeTensor(sharedRaw);
    function->inCasts_.push_back(incast);
    auto viewOut1 = makeTensor(sharedRaw);
    auto viewOut2 = makeTensor(sharedRaw);
    auto& view1 = PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {incast}, {viewOut1});
    auto& view2 = PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {incast}, {viewOut2});
    auto& consumer1 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {viewOut1},
                                                       {makeTensor(std::make_shared<RawTensor>(DT_FP32, shape))});
    auto& consumer2 = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {viewOut2},
                                                       {makeTensor(std::make_shared<RawTensor>(DT_FP32, shape))});

    IRBuilder builder;
    auto token1 = builder.CreateTokenVar(ir::Span::Unknown());
    auto token2 = builder.CreateTokenVar(ir::Span::Unknown());
    view1.result_token_ = {token1};
    view2.result_token_ = {token2};
    consumer1.tokens_ = {token1, token2};
    consumer2.tokens_ = {token1, token2};
    auto& dependency = function->GetVarDependency();
    dependency.AddProducer(token1, ToStmt(view1));
    dependency.AddProducer(token2, ToStmt(view2));
    dependency.AddConsumer(token1, ToStmt(consumer1));
    dependency.AddConsumer(token1, ToStmt(consumer2));
    dependency.AddConsumer(token2, ToStmt(consumer1));
    dependency.AddConsumer(token2, ToStmt(consumer2));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    EXPECT_TRUE(view1.result_token_.empty());
    EXPECT_TRUE(view2.result_token_.empty());
    EXPECT_TRUE(consumer1.tokens_.empty());
    EXPECT_TRUE(consumer2.tokens_.empty());
    EXPECT_FALSE(dependency.HasDependency(token1));
    EXPECT_FALSE(dependency.HasDependency(token2));
    CheckTokenDependencies(*function);
}

TEST_F(ReplaceTensorTest, DoesNotMoveTokenProducerPastItsConsumer)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TokenConsumerBeforeBoundary",
                                               "TokenConsumerBeforeBoundary", nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto sharedRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    auto makeTensor = [&](const std::shared_ptr<RawTensor>& raw) {
        auto tensor = IRBuilder().CreateTensorVar(raw, offset, shape, CreateTestConstIntVector(shape));
        tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
        return tensor;
    };

    auto incast = makeTensor(sharedRaw);
    function->inCasts_.push_back(incast);
    auto firstViewOutput = makeTensor(sharedRaw);
    auto secondViewOutput = makeTensor(sharedRaw);
    auto boundaryOutput = makeTensor(std::make_shared<RawTensor>(DT_FP32, shape));
    auto& tokenProducer = PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {incast}, {firstViewOutput});
    auto& tokenConsumer = PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {firstViewOutput},
                                                           {secondViewOutput});
    auto& boundary = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {secondViewOutput}, {boundaryOutput});

    IRBuilder builder;
    auto token = builder.CreateTokenVar(ir::Span::Unknown());
    tokenProducer.result_token_ = {token};
    tokenConsumer.tokens_ = {token};
    auto& dependency = function->GetVarDependency();
    dependency.AddProducer(token, ToStmt(tokenProducer));
    dependency.AddConsumer(token, ToStmt(tokenConsumer));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_TRUE(tokenProducer.result_token_.empty());
    EXPECT_TRUE(tokenConsumer.tokens_.empty());
    EXPECT_TRUE(boundary.result_token_.empty());
    EXPECT_FALSE(dependency.HasDependency(token));
    EXPECT_NO_THROW(function->GetSortedOperations());
    CheckTokenDependencies(*function);
}

TEST_F(ReplaceTensorTest, DoesNotMoveTokenConsumerToProducerAncestor)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TokenConsumerProducerAncestor",
                                               "TokenConsumerProducerAncestor", nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto sharedRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    auto makeTensor = [&](const std::shared_ptr<RawTensor>& raw) {
        auto tensor = IRBuilder().CreateTensorVar(raw, offset, shape, CreateTestConstIntVector(shape));
        tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
        return tensor;
    };

    auto incast = makeTensor(std::make_shared<RawTensor>(DT_FP32, shape));
    function->inCasts_.push_back(incast);
    auto base = makeTensor(sharedRaw);
    auto& copyInOp = PassOperationUtils::AddOperation(*function, Opcode::OP_COPY_IN, {incast}, {base});

    auto reshape1Out = makeTensor(sharedRaw);
    PassOperationUtils::AddOperation(*function, Opcode::OP_RESHAPE, {base}, {reshape1Out});
    auto viewOut = makeTensor(sharedRaw);
    auto& view = PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {reshape1Out}, {viewOut});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    auto& exp = PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {viewOut},
                                                 {makeTensor(std::make_shared<RawTensor>(DT_FP32, shape))});

    auto reshape2Out = makeTensor(sharedRaw);
    auto& tokenConsumer = PassOperationUtils::AddOperation(*function, Opcode::OP_RESHAPE, {base}, {reshape2Out});

    IRBuilder builder;
    auto token = builder.CreateTokenVar(ir::Span::Unknown());
    exp.result_token_ = {token};
    tokenConsumer.tokens_.push_back(token);
    auto& dependency = function->GetVarDependency();
    dependency.AddProducer(token, ToStmt(exp));
    dependency.AddConsumer(token, ToStmt(tokenConsumer));

    ReplaceTensor pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    ASSERT_EQ(tokenConsumer.tokens_.size(), 1);
    EXPECT_EQ(tokenConsumer.tokens_.front(), token);
    EXPECT_TRUE(copyInOp.tokens_.empty());
    EXPECT_TRUE(dependency.HasProducer(token, ToStmt(exp)));
    EXPECT_TRUE(dependency.HasConsumer(token, ToStmt(tokenConsumer)));
    EXPECT_FALSE(dependency.HasConsumer(token, ToStmt(copyInOp)));
    EXPECT_TRUE(function->LoopCheck().empty());
    CheckTokenDependencies(*function);
}

TEST_F(ReplaceTensorTest, TestOversizedSharedDdrAssembleInputSkipsCopyInOut)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "OversizedSharedDdrInput",
                                               "OversizedSharedDdrInput", nullptr);
    std::vector<int64_t> shape = {128, 2048};
    std::vector<int64_t> offset = {0, 0};
    auto makeDdrTensor = [&shape]() {
        auto tensor = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
        return tensor;
    };

    auto input = makeDdrTensor();
    auto sharedInput = makeDdrTensor();
    auto assembleOutput = makeDdrTensor();
    auto otherOutput = makeDdrTensor();
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {input}, {sharedInput});
    auto& assemble = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {sharedInput}, {assembleOutput});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sharedInput}, {otherOutput});

    ReplaceTensor pass;
    EXPECT_EQ(pass.InsertNeedCopy(*function), SUCCESS);
    auto operations = function->Operations();
    EXPECT_EQ(operations.size(), 3u);
    EXPECT_EQ(assemble.GetIOperands().front(), sharedInput);
    for (const auto& op : operations) {
        EXPECT_NE(op.GetOpcode(), Opcode::OP_COPY_IN);
        EXPECT_NE(op.GetOpcode(), Opcode::OP_COPY_OUT);
    }
}

} // namespace tile_fwk
} // namespace npu
