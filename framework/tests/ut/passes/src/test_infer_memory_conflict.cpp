/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/pass_token_utils.h"
#include "symbolic_scalar_test_utils.h"

namespace npu::tile_fwk {

// FUNCTION DSL 用例共享进程级 Program 单例, 用例间不复位会残留函数表,
// 导致后续用例按 calleeHash 查找函数失败, 参照 test_infermemoryconflict.cpp 的做法在 SetUp 中复位
class PassTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(PassTest, InferMemoryConflictInferOutputWriteConflict)
{
    std::vector<int64_t> shape{-1, 32};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");

    Function* loopFunc = nullptr;
    FUNCTION("MAIN", {a, b, c})
    {
        TileShape::Current().SetVecTile(32, 32);
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(32))
        {
            loopFunc = Program::GetInstance().GetCurrentFunction();
            auto t = Full(1.0, DT_FP32, {32, 32});
            Assemble(t, {idx * 32, 0}, a); // assemble of a not overlap, no conflict
            Assemble(t, {(idx + 1) * 32, 0}, a);

            AtomicRMW(t, {idx * 32, 0}, c, AtomicRMWMode::ADD); // atomic always mark as conflict

            Assemble(t, {idx * 32, 0}, b); // assemble of b overlaped, should be conflict
            Assemble(t, {idx * 32 + 16, 0}, b);
        }
    }
    int cnt = 0;
    PassManager::Instance().RegisterStrategy("InferOutputWriteConflictTestStrategy",
                                             {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(
        PassManager::Instance().RunPass(Program::GetInstance(), *loopFunc, "InferOutputWriteConflictTestStrategy"),
        SUCCESS);
    for (auto out : loopFunc->GetOriginOutcast()) {
        if (out->tensor->GetSymbol() == "a") {
            EXPECT_FALSE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        } else if (out->tensor->GetSymbol() == "b") {
            EXPECT_TRUE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        } else if (out->tensor->GetSymbol() == "c") {
            EXPECT_TRUE(out->HasAttr(OpAttributeKey::writeConflict));
            cnt++;
        }
    }
    EXPECT_EQ(cnt, 3);
}

TEST_F(PassTest, InferMemoryConflictAssembleParallelFalseMarksNormalAndCloneKeepsIt)
{
    std::vector<int64_t> shape{-1, 32};
    Tensor serialOut(DT_FP32, shape, "serial_out");
    Tensor parallelOut(DT_FP32, shape, "parallel_out");

    Function* loopFunc = nullptr;
    FUNCTION("MAIN", {serialOut, parallelOut})
    {
        TileShape::Current().SetVecTile(32, 32);
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(32))
        {
            loopFunc = Program::GetInstance().GetCurrentFunction();
            auto t = Full(1.0, DT_FP32, {32, 32});
            // Cross-loop WAW serial write: parallel=false should mark outcast NORMAL.
            Assemble(t, {idx * 32, 0}, serialOut, false);
            // Default parallel=true should not mark NORMAL.
            Assemble(t, {idx * 32, 0}, parallelOut, true);
        }
    }

    const std::string strategyName = "InferAssembleNormalAttrTestStrategy";
    PassManager::Instance().RegisterStrategy(strategyName, {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(PassManager::Instance().RunPass(Program::GetInstance(), *loopFunc, strategyName), SUCCESS);

    int matched = 0;
    for (auto out : loopFunc->GetOriginOutcast()) {
        if (out->tensor->GetSymbol() == "serial_out") {
            EXPECT_TRUE(out->HasAttr("NORMAL"));
            EXPECT_FALSE(out->HasAttr(OpAttributeKey::writeConflict));
            auto cloned = out->Clone(*loopFunc, true);
            ASSERT_NE(cloned, nullptr);
            EXPECT_TRUE(cloned->HasAttr("NORMAL")) << "NORMAL should be inherited by LogicalTensor::Clone";
            matched++;
        } else if (out->tensor->GetSymbol() == "parallel_out") {
            EXPECT_FALSE(out->HasAttr("NORMAL"));
            matched++;
        }
    }
    EXPECT_EQ(matched, 2);
}

namespace {
int CountRegisterCopy(Function* function)
{
    int cnt = 0;
    for (auto& op : function->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            cnt += 1;
        }
    }
    return cnt;
}
} // namespace

/*
self(incast, mem0) -> INDEX_PUT(self, values, indices) -> dst(与self同物理地址) -> ASSEMBLE -> out(outcast, mem1)
内存信息仅经 INDEX_PUT 的第1输入/第1输出对传播, 传播至 assemble 后检测到与 outcast 的地址冲突,
应在 assemble 前插入 register_copy; values/indices 不参与传播
*/
TEST_F(PassTest, InferMemoryConflictIndexPutInsertsRegisterCopy)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexPut", "TestIndexPut", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {4, 4};
    std::vector<int64_t> valuesShape = {2, 4};
    std::vector<int64_t> indicesShape = {2};
    std::vector<int64_t> offset = {0, 0};

    std::shared_ptr<RawTensor> selfRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    selfRaw->SetSymbol("self");
    selfRaw->memoryId = 0;
    auto self = npu::tile_fwk::IRBuilder().CreateTensorVar(selfRaw, offset, shape, CreateTestConstIntVector(shape));
    auto values = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, valuesShape,
                                                             CreateTestConstIntVector(valuesShape));
    auto indices = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, indicesShape,
                                                              CreateTestConstIntVector(indicesShape));

    std::shared_ptr<RawTensor> dstRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    dstRaw->SetSymbol("index_put_dst");
    dstRaw->memoryId = 0; // 与 self 同物理地址
    auto dst = npu::tile_fwk::IRBuilder().CreateTensorVar(dstRaw, offset, shape, CreateTestConstIntVector(shape));

    std::shared_ptr<RawTensor> outRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    outRaw->SetSymbol("out");
    outRaw->memoryId = 1;
    auto out = npu::tile_fwk::IRBuilder().CreateTensorVar(outRaw, offset, shape, CreateTestConstIntVector(shape));

    currFunctionPtr->inCasts_.push_back(self);
    currFunctionPtr->outCasts_.push_back(out);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_PUT, {self, values, indices}, {dst});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {dst}, {out});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));

    PassManager::Instance().RegisterStrategy("IndexPutMemoryConflictTestStrategy",
                                             {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(
        PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "IndexPutMemoryConflictTestStrategy"),
        SUCCESS);

    EXPECT_EQ(CountRegisterCopy(currFunctionPtr.get()), 1);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), dst);
    EXPECT_EQ(assembleOp.GetIOperands().front(), copy->GetOOperands().front());
}

/*
self(incast, mem0) -> INDEX_ADD(self, src, indices) -> dst(与self同物理地址) -> ASSEMBLE -> out(outcast, mem1)
                                              |-> tmp(第2输出, 临时buffer) -> ASSEMBLE -> out2(outcast, mem2)
内存信息仅经第1输入/第1输出对传播: dst 分支检测到冲突应在 assemble 前插入 register_copy;
tmp 分支不传播内存信息, 不插入 register_copy, INDEX_ADD 前也不插入
*/
TEST_F(PassTest, InferMemoryConflictIndexAddInsertsRegisterCopy)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexAdd", "TestIndexAdd", nullptr);
    ASSERT_NE(currFunctionPtr, nullptr);

    std::vector<int64_t> shape = {4, 4};
    std::vector<int64_t> srcShape = {2, 4};
    std::vector<int64_t> indicesShape = {2};
    std::vector<int64_t> offset = {0, 0};

    std::shared_ptr<RawTensor> selfRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    selfRaw->SetSymbol("self");
    selfRaw->memoryId = 0;
    auto self = npu::tile_fwk::IRBuilder().CreateTensorVar(selfRaw, offset, shape, CreateTestConstIntVector(shape));
    auto src = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, srcShape, CreateTestConstIntVector(srcShape));
    auto indices = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, indicesShape,
                                                              CreateTestConstIntVector(indicesShape));

    std::shared_ptr<RawTensor> dstRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    dstRaw->SetSymbol("index_add_dst");
    dstRaw->memoryId = 0; // 与 self 同物理地址
    auto dst = npu::tile_fwk::IRBuilder().CreateTensorVar(dstRaw, offset, shape, CreateTestConstIntVector(shape));
    auto tmp = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    std::shared_ptr<RawTensor> outRaw = std::make_shared<RawTensor>(DT_FP32, shape);
    outRaw->SetSymbol("out");
    outRaw->memoryId = 1;
    auto out = npu::tile_fwk::IRBuilder().CreateTensorVar(outRaw, offset, shape, CreateTestConstIntVector(shape));
    std::shared_ptr<RawTensor> out2Raw = std::make_shared<RawTensor>(DT_FP32, shape);
    out2Raw->SetSymbol("out2");
    out2Raw->memoryId = 2;
    auto out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(out2Raw, offset, shape, CreateTestConstIntVector(shape));

    currFunctionPtr->inCasts_.push_back(self);
    currFunctionPtr->outCasts_.push_back(out);
    currFunctionPtr->outCasts_.push_back(out2);

    auto& indexAddOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_ADD, {self, src, indices},
                                                        {dst, tmp});

    auto& assembleOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {dst}, {out});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));
    auto& assembleTmpOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {tmp}, {out2});
    assembleTmpOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset));

    PassManager::Instance().RegisterStrategy("IndexAddMemoryConflictTestStrategy",
                                             {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    EXPECT_EQ(
        PassManager::Instance().RunPass(Program::GetInstance(), *currFunctionPtr, "IndexAddMemoryConflictTestStrategy"),
        SUCCESS);

    EXPECT_EQ(CountRegisterCopy(currFunctionPtr.get()), 1);
    Operation* copy = nullptr;
    for (auto& op : currFunctionPtr->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
            copy = op;
        }
    }
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->GetIOperands().front(), dst);
    EXPECT_EQ(assembleOp.GetIOperands().front(), copy->GetOOperands().front());
    EXPECT_EQ(indexAddOp.GetIOperands().front(), self);
    EXPECT_EQ(assembleTmpOp.GetIOperands().front(), tmp);
}

namespace {
struct InputOutputAliasGraph {
    std::shared_ptr<Function> function;
    LogicalTensorPtr input;
    LogicalTensorPtr view;
    LogicalTensorPtr weight;
    LogicalTensorPtr result;
    LogicalTensorPtr output;
    Operation* viewOp;
    Operation* matmul;
    Operation* assemble;
};

InputOutputAliasGraph BuildInputOutputAliasGraph(const Shape& viewShape = {8, 512}, DataType dtype = DT_FP16)
{
    InputOutputAliasGraph graph;
    graph.function = std::make_shared<Function>(Program::GetInstance(), "TestInplaceAlias", "TestInplaceAlias",
                                                nullptr);
    auto createTensor = [dtype](const Shape& shape, const std::string& symbol) {
        auto raw = std::make_shared<RawTensor>(dtype, shape);
        raw->SetSymbol(symbol);
        raw->memoryId = 0;
        return IRBuilder().CreateTensorVar(raw, Offset(shape.size(), 0), shape, CreateTestConstIntVector(shape));
    };
    graph.input = createTensor({-1, -1}, "inplace");
    graph.view = createTensor(viewShape, "view");
    graph.weight = createTensor({512, 512}, "weight");
    graph.result = createTensor({8, 512}, "result");
    graph.output = createTensor({8, 512}, "output");
    graph.function->inCasts_ = {graph.input, graph.weight};
    graph.function->outCasts_ = {graph.output};
    graph.viewOp = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_VIEW, {graph.input}, {graph.view});
    graph.viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset(viewShape.size(), 0), MEM_UNKNOWN));
    graph.matmul = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_A_MUL_B, {graph.view, graph.weight},
                                                     {graph.result});
    graph.assemble = &PassOperationUtils::AddOperation(*graph.function, Opcode::OP_ASSEMBLE, {graph.result},
                                                       {graph.output});
    graph.assemble->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, Offset(viewShape.size(), 0)));
    TileShape tile;
    tile.SetVecTile({8, 128});
    graph.matmul->UpdateTileShape(tile);
    return graph;
}

Status RunAliasMemoryConflictPass(Function& function)
{
    const std::string strategy = "InplaceAliasMemoryConflictTestStrategy";
    PassManager::Instance().RegisterStrategy(strategy, {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    return PassManager::Instance().RunPass(Program::GetInstance(), function, strategy);
}

void ExpectInputSnapshot(const Operation& consumer, size_t index, const LogicalTensorPtr& original)
{
    const auto& input = consumer.GetIOperands()[index];
    ASSERT_NE(input, original);
    ASSERT_EQ(input->GetProducers().size(), 1);
    const auto* copy = *input->GetProducers().begin();
    EXPECT_EQ(copy->GetOpcode(), Opcode::OP_REGISTER_COPY);
    EXPECT_EQ(copy->GetIOperands().front(), original);
    EXPECT_NE(input->GetRawMagic(), original->GetRawMagic());
    EXPECT_EQ(input->GetShape(), original->GetShape());
}
} // namespace

TEST_F(PassTest, InferMemoryConflictInplaceInputOutputCopiesViewInput)
{
    auto graph = BuildInputOutputAliasGraph();
    // SSA 改写后，不同的 RawTensor 仍可能复用同一块输入存储。
    graph.output->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
    ExpectInputSnapshot(*graph.matmul, 0, graph.view);
    EXPECT_EQ(graph.viewOp->GetIOperands().front(), graph.input);
    EXPECT_EQ(graph.matmul->GetIOperands()[1], graph.weight);
    EXPECT_EQ(graph.assemble->GetIOperands().front(), graph.result);
}

TEST_F(PassTest, InferMemoryConflictAlignsAliasSnapshotTile)
{
    for (const auto dtype : {DT_INT8, DT_FP16, DT_FP32, DT_INT64}) {
        for (const int64_t lastTile : {50, 64}) {
            SCOPED_TRACE(std::to_string(dtype) + ":" + std::to_string(lastTile));
            auto graph = BuildInputOutputAliasGraph({2, 112}, dtype);
            graph.output->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
            TileShape tile;
            tile.SetVecTile({1, lastTile});
            graph.viewOp->UpdateTileShape(tile);

            ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
            ASSERT_EQ(CountRegisterCopy(graph.function.get()), 1);
            ExpectInputSnapshot(*graph.matmul, 0, graph.view);
            const auto* copy = *graph.matmul->GetIOperands().front()->GetProducers().begin();
            const int64_t blockElements = 32 / BytesOf(dtype);
            const int64_t alignedTile = (lastTile + blockElements - 1) / blockElements * blockElements;
            EXPECT_EQ(copy->GetTileShape().GetVecTile().tile, (Shape{1, alignedTile}));
            EXPECT_EQ(graph.viewOp->GetTileShape().GetVecTile().tile, (Shape{1, lastTile}));
            EXPECT_EQ(copy->GetOOperands().front()->GetDynValidShape(), graph.view->GetDynValidShape());
        }
    }
}

TEST_F(PassTest, InferMemoryConflictCopiesInputSharingOutputSlot)
{
    auto graph = BuildInputOutputAliasGraph();
    auto scope = std::make_shared<TensorSlotScope>(graph.function.get());
    scope->ioslot.incastSlot = {{17, 19}, {18}};
    scope->ioslot.outcastSlot = {{19}};
    graph.function->SetSlotScope(scope);
    ASSERT_NE(graph.input->GetRawMagic(), graph.output->GetRawMagic());
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
    ExpectInputSnapshot(*graph.matmul, 0, graph.view);
    EXPECT_EQ(graph.matmul->GetIOperands()[1], graph.weight);
}

TEST_F(PassTest, InferMemoryConflictIgnoresMatchingTensorNames)
{
    auto graph = BuildInputOutputAliasGraph();
    graph.output->GetRawTensor()->SetSymbol(graph.input->GetRawTensor()->GetSymbol());
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 0);
    EXPECT_EQ(graph.matmul->GetIOperands().front(), graph.view);
}

TEST_F(PassTest, InferMemoryConflictIgnoresIntermediateStorageAliases)
{
    auto graph = BuildInputOutputAliasGraph();
    graph.view->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 0);
    EXPECT_EQ(graph.matmul->GetIOperands().front(), graph.view);
}

TEST_F(PassTest, InferMemoryConflictSkipsDynamicViewSnapshot)
{
    auto graph = BuildInputOutputAliasGraph({-1, 512});
    graph.output->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 0);
    EXPECT_EQ(graph.matmul->GetIOperands().front(), graph.view);
}

TEST_F(PassTest, InferMemoryConflictWaitsForConcreteViewShape)
{
    auto graph = BuildInputOutputAliasGraph();
    graph.output->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    const Shape dynamicShape{-1, 512};
    auto dynamicView = IRBuilder().CreateTensorVar(DT_FP16, dynamicShape, CreateTestConstIntVector(dynamicShape));
    auto& view = PassOperationUtils::AddOperation(*graph.function, Opcode::OP_VIEW, {graph.input}, {dynamicView});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset(dynamicShape.size(), 0), MEM_UNKNOWN));
    graph.viewOp->ReplaceInput(dynamicView, graph.input);
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
    ExpectInputSnapshot(*graph.matmul, 0, graph.view);
    EXPECT_EQ(graph.viewOp->GetIOperands().front(), dynamicView);
}

TEST_F(PassTest, InferMemoryConflictCopiesAllAliasedConsumerInputs)
{
    auto graph = BuildInputOutputAliasGraph();
    graph.output->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    graph.weight->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 2);
    ExpectInputSnapshot(*graph.matmul, 0, graph.view);
    ExpectInputSnapshot(*graph.matmul, 1, graph.weight);
}

TEST_F(PassTest, InferMemoryConflictReusesExistingInputSnapshot)
{
    auto graph = BuildInputOutputAliasGraph();
    graph.output->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    auto snapshot = IRBuilder().CreateTensorVar(DT_FP16, graph.view->GetShape(), graph.view->GetDynValidShape());
    PassOperationUtils::AddOperation(*graph.function, Opcode::OP_REGISTER_COPY, {graph.view}, {snapshot});
    graph.matmul->ReplaceInput(snapshot, graph.view);
    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
    EXPECT_EQ(graph.matmul->GetIOperands().front(), snapshot);
    ExpectInputSnapshot(*graph.matmul, 0, graph.view);
}

TEST_F(PassTest, InferMemoryConflictPreservesInplaceInput)
{
    for (auto opcode : {Opcode::OP_INDEX_PUT, Opcode::OP_INDEX_ADD, Opcode::OP_INDEX_OUTCAST}) {
        SCOPED_TRACE(static_cast<int>(opcode));
        auto function = std::make_shared<Function>(Program::GetInstance(), "TestInplaceAlias", "TestInplaceAlias",
                                                   nullptr);
        const Shape shape = {4, 4};
        const Shape valuesShape = {2, 4};
        const Shape indicesShape = opcode == Opcode::OP_INDEX_OUTCAST ? Shape{1, 2} : Shape{2};
        auto inputRaw = std::make_shared<RawTensor>(DT_FP32, shape);
        inputRaw->memoryId = 0;
        auto input = IRBuilder().CreateTensorVar(inputRaw, Offset(shape.size(), 0), shape,
                                                 CreateTestConstIntVector(shape));
        auto values = IRBuilder().CreateTensorVar(DT_FP32, valuesShape, CreateTestConstIntVector(valuesShape));
        auto indices = IRBuilder().CreateTensorVar(DT_INT32, indicesShape, CreateTestConstIntVector(indicesShape));
        auto output = IRBuilder().CreateTensorVar(inputRaw, Offset(shape.size(), 0), shape,
                                                  CreateTestConstIntVector(shape));
        LogicalTensors inputs = {input, values, indices};
        const size_t inplaceIndex = opcode == Opcode::OP_INDEX_OUTCAST ? 2 : 0;
        if (opcode == Opcode::OP_INDEX_OUTCAST) {
            inputs = {values, indices, input};
        }
        LogicalTensors outputs = {output};
        if (opcode == Opcode::OP_INDEX_ADD) {
            outputs.push_back(IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape)));
        }
        function->inCasts_ = {input, values, indices};
        function->outCasts_ = outputs;
        auto& inplaceOp = PassOperationUtils::AddOperation(*function, opcode, inputs, outputs);

        ASSERT_EQ(RunAliasMemoryConflictPass(*function), SUCCESS);
        EXPECT_EQ(CountRegisterCopy(function.get()), 0);
        EXPECT_EQ(inplaceOp.GetIOperands()[inplaceIndex], input);
        EXPECT_EQ(inplaceOp.GetIOperands(), inputs);
        EXPECT_EQ(inplaceOp.GetOOperands(), outputs);

        // 同一算子的普通读取输入仍需缓存，只有原地目标输入保持不变。
        values->GetRawTensor()->actualRawmagic = input->GetRawMagic();
        ASSERT_EQ(RunAliasMemoryConflictPass(*function), SUCCESS);
        EXPECT_EQ(CountRegisterCopy(function.get()), 1);
        EXPECT_EQ(inplaceOp.GetIOperands()[inplaceIndex], input);
        ExpectInputSnapshot(inplaceOp, inplaceIndex == 2 ? 0 : 1, values);
    }
}

TEST_F(PassTest, InferMemoryConflictPreservesAssembleSsaInplaceInput)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestAssembleSsaAlias", "TestAssembleSsaAlias",
                                               nullptr);
    const Shape shape = {2, 2};
    auto source = IRBuilder().CreateTensorVar(DT_FP16, shape, CreateTestConstIntVector(shape));
    auto outputRaw = std::make_shared<RawTensor>(DT_FP16, shape);
    outputRaw->memoryId = 0;
    auto output = IRBuilder().CreateTensorVar(outputRaw, Offset(shape.size(), 0), shape,
                                              CreateTestConstIntVector(shape));
    auto view = IRBuilder().CreateTensorVar(DT_FP16, shape, CreateTestConstIntVector(shape));
    auto result = IRBuilder().CreateTensorVar(outputRaw, Offset(shape.size(), 0), shape,
                                              CreateTestConstIntVector(shape));
    function->inCasts_ = {source, output};
    function->outCasts_ = {result};

    auto& viewOp = PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {output}, {view});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset(shape.size(), 0), MEM_UNKNOWN));
    auto& assembleSsa = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE_SSA, {source, view}, {result});
    assembleSsa.SetAssembleOpAttribute(Offset(shape.size(), 0), {});
    assembleSsa.SetAttribute(OpAttributeKey::inplaceIdx, 1);

    ASSERT_EQ(RunAliasMemoryConflictPass(*function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(function.get()), 0);
    EXPECT_EQ(assembleSsa.GetIOperands()[0], source);
    EXPECT_EQ(assembleSsa.GetIOperands()[1], view);
    ASSERT_EQ(view->GetProducers().size(), 1);
    EXPECT_EQ((*view->GetProducers().begin())->GetOpcode(), Opcode::OP_VIEW);
}

TEST_F(PassTest, InferMemoryConflictPreservesIndependentWriteOrdering)
{
    auto graph = BuildInputOutputAliasGraph();
    graph.output->GetRawTensor()->actualRawmagic = graph.input->GetRawMagic();
    auto writeSource = IRBuilder().CreateTensorVar(DT_FP16, graph.output->GetShape(), graph.output->GetDynValidShape());
    writeSource->GetRawTensor()->memoryId = 0;
    graph.function->inCasts_.push_back(writeSource);
    graph.function->outCasts_.push_back(graph.result);
    graph.assemble->ReplaceInput(writeSource, graph.result);
    // 读写分支没有数据边，由原有 token 保证先读后写，插入快照后仍应保持该约束。
    PassTokenUtils::LinkTokenDependency(*graph.function, *graph.matmul, *graph.assemble);
    const auto tokens = graph.assemble->tokens_;
    ASSERT_EQ(graph.assemble->ProducerOps().count(graph.matmul), 0);

    ASSERT_EQ(RunAliasMemoryConflictPass(*graph.function), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
    ExpectInputSnapshot(*graph.matmul, 0, graph.view);
    EXPECT_EQ(graph.assemble->GetIOperands().front(), writeSource);
    EXPECT_EQ(graph.assemble->tokens_, tokens);
    EXPECT_EQ(graph.assemble->ProducerOpsByToken().count(graph.matmul), 1);
}

namespace {
// 首轴动态连续 RESHAPE 豁免用例的构图工具。
// 所有 rawtensor 共用 memoryId, 使 REGISTER_COPY 仅由 RESHAPE 的 rawshape 冲突路径驱动,
// 排除 CheckConflict 真实内存冲突路径对 copy 计数的干扰
LogicalTensorPtr CreateExemptionTestTensor(const std::string& symbol, const Shape& shape,
                                           const std::vector<SymbolicScalar>& valid = {})
{
    auto rawTensor = std::make_shared<RawTensor>(DT_FP16, shape);
    rawTensor->SetSymbol(symbol);
    rawTensor->memoryId = 0;
    auto validShape = valid.empty() ? CreateTestConstIntVector(shape) : valid;
    return IRBuilder().CreateTensorVar(rawTensor, Offset(shape.size(), 0), shape, validShape);
}

struct ViewReshapeTestGraph {
    std::shared_ptr<Function> function;
    LogicalTensorPtr incast;
    LogicalTensorPtr viewOut;
    LogicalTensorPtr reshapeOut;
    LogicalTensorPtr transOut;
    Operation* viewOp = nullptr;
    Operation* reshapeOp = nullptr;
    Operation* transOp = nullptr;
};

// 构图: incast(incastShape) -> VIEW(viewOffset) -> viewOut(viewOutShape) -> RESHAPE -> reshapeOut
//       -> [TRANSDATA -> transOut ->] ASSEMBLE -> out(outcast)。
// withTransDataConsumer=false 时 RESHAPE 输出被 ASSEMBLE 直接消费(豁免须拒绝的场景)
ViewReshapeTestGraph BuildViewReshapeGraph(const std::string& name, const Shape& incastShape, const Shape& viewOutShape,
                                           const Shape& reshapeOutShape, const Offset& viewOffset,
                                           const std::vector<int64_t>& reshapeTile,
                                           const std::vector<SymbolicScalar>& viewOutValid = {},
                                           const std::vector<SymbolicScalar>& viewDynOffset = {},
                                           bool withTransDataConsumer = true)
{
    ViewReshapeTestGraph graph;
    graph.function = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    graph.incast = CreateExemptionTestTensor("incast", incastShape);
    graph.viewOut = CreateExemptionTestTensor("view_out", viewOutShape, viewOutValid);
    graph.reshapeOut = CreateExemptionTestTensor("reshape_out", reshapeOutShape);
    auto out = CreateExemptionTestTensor("out", reshapeOutShape);
    graph.function->inCasts_.push_back(graph.incast);
    graph.function->outCasts_.push_back(out);

    auto& viewOp = PassOperationUtils::AddOperation(*graph.function, Opcode::OP_VIEW, {graph.incast}, {graph.viewOut});
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset, MemoryType::MEM_UNKNOWN, viewDynOffset));
    graph.viewOp = &viewOp;

    auto& reshapeOp = PassOperationUtils::AddOperation(*graph.function, Opcode::OP_RESHAPE, {graph.viewOut},
                                                       {graph.reshapeOut});
    TileShape tileShape;
    tileShape.SetVecTile(reshapeTile);
    reshapeOp.UpdateTileShape(tileShape);
    graph.reshapeOp = &reshapeOp;

    LogicalTensorPtr assembleSrc = graph.reshapeOut;
    if (withTransDataConsumer) {
        graph.transOut = CreateExemptionTestTensor("trans_out", reshapeOutShape);
        auto transShapeOut = CreateExemptionTestTensor("trans_shape_out", reshapeOutShape);
        auto& transOp = PassOperationUtils::AddOperation(*graph.function, Opcode::OP_NCHW2Fractal_Z, {graph.reshapeOut},
                                                         {graph.transOut, transShapeOut});
        graph.transOp = &transOp;
        assembleSrc = graph.transOut;
    }

    auto& assembleOp = PassOperationUtils::AddOperation(*graph.function, Opcode::OP_ASSEMBLE, {assembleSrc}, {out});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, Offset(reshapeOutShape.size(), 0)));
    return graph;
}

Status RunInferMemoryConflictPass(Function& function, const std::string& strategyName)
{
    PassManager::Instance().RegisterStrategy(strategyName, {{"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT}});
    return PassManager::Instance().RunPass(Program::GetInstance(), function, strategyName);
}
} // namespace

// [-1,8,3,3] 布局来源经单 VIEW 物化为 [80,8,3,3], 尾轴展平为 [80,72]: 布局保持, 免插 REGISTER_COPY
TEST_F(PassTest, InferMemoryConflictExemptLeadingDynamicFlattenReshape)
{
    auto graph = BuildViewReshapeGraph("ExemptFlatten", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 72}, Offset{0, 0, 0, 0},
                                       {16, 8, 3, 3});
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "ExemptFlattenStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 0);
    EXPECT_EQ(graph.reshapeOp->GetIOperands().front(), graph.viewOut); // RESHAPE 输入未被 copy 替换
}

// conv1d 权重链(3D->2D): [-1,8,3] -> VIEW -> [80,8,3] -> RESHAPE -> [80,24]。
// conv1d 构图的 3D->4D 升维 RESHAPE 已被先于本 pass 的 RemoveRedundantReshape 折叠
TEST_F(PassTest, InferMemoryConflictExemptLeadingDynamicFlatten3DReshape)
{
    auto graph = BuildViewReshapeGraph("ExemptFlatten3D", {-1, 8, 3}, {80, 8, 3}, {80, 24}, Offset{0, 0, 0},
                                       {16, 8, 3});
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "ExemptFlatten3DStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 0);
    EXPECT_EQ(graph.reshapeOp->GetIOperands().front(), graph.viewOut);
}

// 同函数存在全动态 incast(BMM 的函数级卫兵场景)不影响豁免; VIEW 首轴动态滑窗偏移合法
TEST_F(PassTest, InferMemoryConflictExemptWithDynamicSiblingAndLeadingOffset)
{
    std::vector<SymbolicScalar> viewDynOffset{CreateTestScalarVar("loop_n")};
    auto trailingZero = CreateTestConstIntVector({0, 0, 0});
    viewDynOffset.insert(viewDynOffset.end(), trailingZero.begin(), trailingZero.end());
    auto graph = BuildViewReshapeGraph("ExemptWithSibling", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 72}, Offset{0, 0, 0, 0},
                                       {16, 8, 3, 3}, {}, viewDynOffset);
    graph.function->inCasts_.push_back(CreateExemptionTestTensor("fmap", {-1, -1, -1, -1}));
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "ExemptWithSiblingStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 0);
}

// VIEW 窗口切了尾轴([-1,8,3,4] -> [80,8,3,3]): 行块不完整, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenViewCutsTrailingAxis)
{
    auto graph = BuildViewReshapeGraph("ViewCutsTrailing", {-1, 8, 3, 4}, {80, 8, 3, 3}, {80, 72}, Offset{0, 0, 0, 0},
                                       {16, 8, 3, 3});
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "ViewCutsTrailingStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
}

// 源 raw 尾轴大于逻辑尾轴([-1,8,3,3] 逻辑 / [-1,8,3,4] raw): 行间存在空洞, 展平应读取
// 0,1,2,4,5,6,... 而非连续 0..N, 跳过物化会读入间隔数据, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenViewInputRawTailExceedsLogicalTail)
{
    auto graph = BuildViewReshapeGraph("RawTailMismatch", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 72}, Offset{0, 0, 0, 0},
                                       {16, 8, 3, 3});
    graph.incast->GetRawTensor()->UpdateRawShape({-1, 8, 3, 4});
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "RawTailMismatchStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
}

// VIEW 窗口偏移落在尾轴上: 起点不在行块边界, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenViewOffsetsTrailingAxis)
{
    auto graph = BuildViewReshapeGraph("ViewOffsetsTrailing", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 72},
                                       Offset{0, 0, 1, 0}, {16, 8, 3, 3});
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "ViewOffsetsTrailingStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
}

// 直接 VIEW 输入的尾轴含动态维度([-1,8,3,-1]): 行块大小不可静态证明, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenViewInputHasDynamicTrailingAxis)
{
    auto graph = BuildViewReshapeGraph("DynTrailing", {-1, 8, 3, -1}, {80, 8, 3, 3}, {80, 72}, Offset{0, 0, 0, 0},
                                       {16, 8, 3, 3});
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "DynTrailingStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
}

// RESHAPE 输入尾轴有效范围不满(valid [80,8,3,2]): 展平块内存在空洞, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenTrailingValidNotFull)
{
    auto graph = BuildViewReshapeGraph("TrailingValidNotFull", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 72},
                                       Offset{0, 0, 0, 0}, {16, 8, 3, 3}, CreateTestConstIntVector({80, 8, 3, 2}));
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "TrailingValidNotFullStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
}

// RESHAPE 首轴不保持([80,8,3,3] -> [40,144]): 不属于尾轴展平模式, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenLeadingAxisNotPreserved)
{
    auto graph = BuildViewReshapeGraph("LeadingNotPreserved", {-1, 8, 3, 3}, {80, 8, 3, 3}, {40, 144},
                                       Offset{0, 0, 0, 0}, {16, 8, 3, 3});
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "LeadingNotPreservedStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
}

// RESHAPE 输出为 NZ 格式: 连续性论证仅对 ND 布局成立, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenOutputNotNdFormat)
{
    auto graph = BuildViewReshapeGraph("OutputNotNd", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 72}, Offset{0, 0, 0, 0},
                                       {16, 8, 3, 3});
    graph.reshapeOut->GetRawTensor()->format = TileOpFormat::TILEOP_NZ;
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "OutputNotNdStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
}

// RESHAPE 输出被非 TRANSDATA 算子(ASSEMBLE)直接消费: 别名逸散到通用图会破坏
// 后续 view-assemble 融合的地址解析(assembled scratch 精度事故), 必须物化, 不豁免
TEST_F(PassTest, InferMemoryConflictNotExemptWhenOutputNotConsumedByTransData)
{
    auto graph = BuildViewReshapeGraph("NoTransDataConsumer", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 72},
                                       Offset{0, 0, 0, 0}, {16, 8, 3, 3}, {}, {}, false);
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "NoTransDataConsumerStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(graph.function.get()), 1);
    EXPECT_NE(graph.reshapeOp->GetIOperands().front(), graph.viewOut);
}

// RESHAPE 输入无 VIEW 生产者(直接消费 incast): 不满足豁免构图前提, 不豁免。
// 输入侧即动态 incast: 前向 rawshape 冲突插前置 copy, 反向传播的冲突路径再插后置 copy, 共 2 个
TEST_F(PassTest, InferMemoryConflictNotExemptWithoutViewProducer)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "NoViewProducer", "NoViewProducer", nullptr);
    auto incast = CreateExemptionTestTensor("incast", {-1, 8, 3, 3});
    auto reshapeOut = CreateExemptionTestTensor("reshape_out", {80, 72});
    auto out = CreateExemptionTestTensor("out", {80, 72});
    function->inCasts_.push_back(incast);
    function->outCasts_.push_back(out);

    auto& reshapeOp = PassOperationUtils::AddOperation(*function, Opcode::OP_RESHAPE, {incast}, {reshapeOut});
    TileShape tileShape;
    tileShape.SetVecTile(std::vector<int64_t>{16, 8, 3, 3});
    reshapeOp.UpdateTileShape(tileShape);
    auto& assembleOp = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {reshapeOut}, {out});
    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, Offset{0, 0}));

    EXPECT_EQ(RunInferMemoryConflictPass(*function, "NoViewProducerStrategy"), SUCCESS);
    EXPECT_EQ(CountRegisterCopy(function.get()), 2);
    EXPECT_NE(reshapeOp.GetIOperands().front(), incast); // RESHAPE 输入已被前置 copy 输出替换
}

// 函数含 NZ 多动态轴 incast 且 RESHAPE 目标非 2D(不满足豁免的维度模式): 落回 BMM 判定, 既有 NZ 非法输入报错保留
TEST_F(PassTest, InferMemoryConflictNzMultiDynamicIncastStillFails)
{
    auto graph = BuildViewReshapeGraph("NzMultiDynamic", {-1, 8, 3, 3}, {80, 8, 3, 3}, {80, 24, 3}, Offset{0, 0, 0, 0},
                                       {16, 8, 3, 3});
    auto nzRaw = std::make_shared<RawTensor>(DT_FP16, Shape{-1, -1, 3, 3});
    nzRaw->SetSymbol("nz_incast");
    nzRaw->format = TileOpFormat::TILEOP_NZ;
    auto nzIncast = IRBuilder().CreateTensorVar(nzRaw, Offset(4, 0), Shape{-1, -1, 3, 3},
                                                CreateTestConstIntVector({-1, -1, 3, 3}));
    graph.function->inCasts_.push_back(nzIncast);
    EXPECT_EQ(RunInferMemoryConflictPass(*graph.function, "NzMultiDynamicStrategy"), FAILED);
}
} // namespace npu::tile_fwk
