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
 * \file test_merge_view_assemble.cpp
 * \brief Unit test for merge_view_assemble pass.
 */

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "pass_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_optimization/merge_view_assemble.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/tile_graph_pass/graph_optimization/process_atomic.h"
#include <fstream>
#include <vector>
#include <string>
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
constexpr int NUM5 = 5;
constexpr int NUM6 = 6;

class MergeViewAssembleTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ViewAssembleTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IRContext::Get().SetAssembleNewLogicalTensor(false);
    }
    void TearDown() override { IRContext::Get().SetAssembleNewLogicalTensor(false); }
};

namespace {
ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

ir::VarPtr AddTokenEdge(Function& function, Operation& producer, Operation& consumer)
{
    if (producer.result_token_.empty()) {
        producer.result_token_ = {IRBuilder().CreateTokenVar(producer.GetSpan())};
    }
    auto token = producer.result_token_.front();
    function.GetVarDependency().AddProducer(token, ToStmtPtr(producer));
    function.GetVarDependency().AddConsumer(token, ToStmtPtr(consumer));
    if (std::find(consumer.tokens_.begin(), consumer.tokens_.end(), token) == consumer.tokens_.end()) {
        consumer.tokens_.emplace_back(token);
    }
    return token;
}

void BuildTokenBoundaryGraph(ComputationalGraphBuilder& graph, Opcode opcode)
{
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {4, 4},
                                 {"input", "middle", "output", "sync_input", "sync_output", "side_output"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {"sync_input"}, {"sync_output"}, "sync", true));
    ASSERT_TRUE(graph.AddOp(opcode, {"input"}, {"middle"}, "first", true));
    ASSERT_TRUE(graph.AddOp(opcode, {"middle"}, {"output"}, "second", true));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {"sync_input"}, {"side_output"}, "side", true));
    for (const auto& name : {"first", "second"}) {
        if (opcode == Opcode::OP_VIEW) {
            graph.GetOp(name)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        } else {
            graph.GetOp(name)->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        }
    }
    ASSERT_TRUE(graph.SetInCast({"input", "sync_input"}));
    ASSERT_TRUE(graph.SetOutCast({"output", "sync_output", "side_output"}));
}

void BuildLiveTokenChains(ComputationalGraphBuilder& graph, int count)
{
    ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {4, 4}, {"input", "sync_output"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {"input"}, {"sync_output"}, "sync", true));
    std::vector<std::string> outputs{"sync_output"};
    for (int index = 0; index < count; ++index) {
        std::string id = std::to_string(index);
        std::string middle = "middle_" + id;
        std::string output = "output_" + id;
        std::string first = "first_" + id;
        std::string second = "second_" + id;
        Opcode opcode = index % 2 == 0 ? Opcode::OP_VIEW : Opcode::OP_ASSEMBLE;
        ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {4, 4}, {middle, output}));
        ASSERT_TRUE(graph.AddOp(opcode, {"input"}, {middle}, first, true));
        ASSERT_TRUE(graph.AddOp(opcode, {middle}, {output}, second, true));
        for (const auto& name : {first, second}) {
            if (opcode == Opcode::OP_VIEW) {
                graph.GetOp(name)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
            } else {
                graph.GetOp(name)->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
            }
        }
        AddTokenEdge(*graph.GetFunction(), *graph.GetOp("sync"), *graph.GetOp(first));
        outputs.emplace_back(output);
    }
    ASSERT_TRUE(graph.SetInCast({"input"}));
    ASSERT_TRUE(graph.SetOutCast(outputs));
}

void AddUnrelatedTokenAndMiddleReader(ComputationalGraphBuilder& graph, bool unusedMarker)
{
    graph.GetOp("side")->UpdateInputOperand(0, graph.GetTensor("middle"));
    graph.GetOp("sync")->UpdateInputOperand(0, graph.GetTensor("input"));
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, {4, 4}, "token_output"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {"input"}, {"token_output"}, "token_consumer", true));
    AddTokenEdge(*graph.GetFunction(), *graph.GetOp("sync"), *graph.GetOp("token_consumer"));
    if (unusedMarker) {
        auto* first = graph.GetOp("first");
        first->result_token_ = {IRBuilder().CreateTokenVar(first->GetSpan())};
        graph.GetFunction()->GetVarDependency().AddProducer(first->result_token_.front(), ToStmtPtr(*first));
    }
    ASSERT_TRUE(graph.SetOutCast({"output", "sync_output", "side_output", "token_output"}));
}

void BuildLargeSharedTensorMergeableChains(ComputationalGraphBuilder& graph, int assembleChainCount, int viewChainCount)
{
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    inputs.reserve(assembleChainCount);
    outputs.reserve(viewChainCount);
    std::vector<int64_t> shape{16, 16};
    ASSERT_TRUE(graph.AddTensor(DataType::DT_FP32, shape, "shared"));
    for (int index = 0; index < assembleChainCount; ++index) {
        std::string id = std::to_string(index);
        std::string input = "assemble_input_" + id;
        std::string producerOut = "assemble_producer_out_" + id;
        std::string mid = "assemble_mid_" + id;
        std::string producerName = "assemble_producer_" + id;
        std::string firstAssembleName = "assemble_first_" + id;
        std::string secondAssembleName = "assemble_second_" + id;
        ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, shape, {input, producerOut, mid}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {input}, {producerOut}, producerName, true));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ASSEMBLE, {producerOut}, {mid}, firstAssembleName, true));
        graph.GetOp(firstAssembleName)
            ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ASSEMBLE, {mid}, {"shared"}, secondAssembleName, true));
        graph.GetOp(secondAssembleName)
            ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}));
        inputs.emplace_back(std::move(input));
    }
    for (int index = 0; index < viewChainCount; ++index) {
        std::string id = std::to_string(index);
        std::string mid = "view_mid_" + id;
        std::string output = "view_output_" + id;
        std::string firstViewName = "view_first_" + id;
        std::string secondViewName = "view_second_" + id;
        ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, shape, {mid, output}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_VIEW, {"shared"}, {mid}, firstViewName, true));
        graph.GetOp(firstViewName)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_VIEW, {mid}, {output}, secondViewName, true));
        graph.GetOp(secondViewName)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 0}));
        outputs.emplace_back(std::move(output));
    }
    ASSERT_TRUE(graph.SetInCast(inputs));
    ASSERT_TRUE(graph.SetOutCast(outputs));
}
} // namespace

TEST_F(MergeViewAssembleTest, LegacyStopperUsesArchitectureDespiteUnrelatedTokenAndMarker)
{
    for (auto npuArch : {NPUArch::DAV_2201, NPUArch::DAV_3510}) {
        NpuArchScope arch(npuArch);
        for (bool constructionMode : {false, true}) {
            IRContext::Get().SetAssembleNewLogicalTensor(constructionMode);
            for (bool unusedMarker : {false, true}) {
                ComputationalGraphBuilder graph;
                BuildTokenBoundaryGraph(graph, Opcode::OP_ASSEMBLE);
                AddUnrelatedTokenAndMiddleReader(graph, unusedMarker);
                MergeViewAssemble pass;
                ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
                auto output = graph.GetTensor("output");
                ASSERT_EQ(output->GetProducers().size(), 1);
                auto expected = graph.GetTensor(npuArch == NPUArch::DAV_3510 ? "input" : "middle");
                EXPECT_EQ((*output->GetProducers().begin())->GetIOperands().front(), expected);
                EXPECT_FALSE(graph.GetTensor("middle")->GetProducers().empty());
                EXPECT_EQ(graph.GetOp("side")->GetIOperands().front(), graph.GetTensor("middle"));
                auto* consumer = graph.GetOp("token_consumer");
                ASSERT_EQ(consumer->tokens_.size(), 1);
                EXPECT_TRUE(graph.GetFunction()->GetVarDependency().HasProducer(consumer->tokens_.front(),
                                                                                ToStmtPtr(*graph.GetOp("sync"))));
            }
        }
    }
}

TEST_F(MergeViewAssembleTest, AssembleWithoutInputFailsInBothConstructionModes)
{
    for (bool constructionMode : {false, true}) {
        IRContext::Get().SetAssembleNewLogicalTensor(constructionMode);
        ComputationalGraphBuilder graph;
        BuildTokenBoundaryGraph(graph, Opcode::OP_ASSEMBLE);
        auto* first = graph.GetOp("first");
        graph.GetTensor("input")->RemoveConsumer(first);
        first->GetIOperands().clear();
        MergeViewAssemble pass;
        EXPECT_EQ(pass.RunOnFunction(*graph.GetFunction()), FAILED);
    }
}

TEST_F(MergeViewAssembleTest, LegacyAssembleKeepsProducerFreeIncast)
{
    for (bool constructionMode : {false, true}) {
        IRContext::Get().SetAssembleNewLogicalTensor(constructionMode);
        ComputationalGraphBuilder graph;
        ASSERT_TRUE(graph.AddTensors(DataType::DT_FP32, {4, 4}, {"input", "output"}));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ASSEMBLE, {"input"}, {"output"}, "assemble", true));
        graph.GetOp("assemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        ASSERT_TRUE(graph.SetInCast({"input"}));
        ASSERT_TRUE(graph.SetOutCast({"output"}));
        MergeViewAssemble pass;
        ASSERT_EQ(pass.RunOnFunction(*graph.GetFunction()), SUCCESS);
        ASSERT_EQ(graph.GetTensor("output")->GetProducers().size(), 1);
        EXPECT_EQ((*graph.GetTensor("output")->GetProducers().begin())->GetIOperands().front(),
                  graph.GetTensor("input"));
    }
}

static void ExpectTopologicalOrder(Function* function)
{
    auto ops = function->Operations().DuplicatedOpList();
    std::unordered_map<Operation*, size_t> opToIndex;
    opToIndex.reserve(ops.size());
    for (size_t index = 0; index < ops.size(); ++index) {
        opToIndex.emplace(ops[index], index);
    }
    for (auto* op : ops) {
        for (const auto& input : op->GetIOperands()) {
            for (auto* producer : input->GetProducers()) {
                if (producer->BelongTo() == function && producer != op) {
                    EXPECT_LT(opToIndex.at(producer), opToIndex.at(op));
                }
            }
        }
        for (const auto& depend : op->GetDependOperands()) {
            for (auto* producer : depend->GetProducers()) {
                if (producer->BelongTo() == function && producer != op) {
                    EXPECT_LT(opToIndex.at(producer), opToIndex.at(op));
                }
            }
        }
    }
}

TEST_F(MergeViewAssembleTest, MergeViewAssembleKeepsTopologicalOrderForSharedTensorFanout)
{
    constexpr int producerNum = 4;
    constexpr int consumerNum = 5;
    std::vector<int64_t> shape{16, 16};
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, shape, {"input", "shared"}));
    for (int index = 0; index < consumerNum; ++index) {
        auto indexStr = std::to_string(index);
        ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, shape, {"view_mid_" + indexStr, "view_out_" + indexStr}));
        ASSERT_TRUE(G.AddOp(Opcode::OP_VIEW, {"shared"}, {"view_mid_" + indexStr}, "view_mid_" + indexStr, true));
        G.GetOp("view_mid_" + indexStr)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        ASSERT_TRUE(
            G.AddOp(Opcode::OP_VIEW, {"view_mid_" + indexStr}, {"view_out_" + indexStr}, "view_out_" + indexStr, true));
        G.GetOp("view_out_" + indexStr)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    }
    for (int index = 0; index < producerNum; ++index) {
        auto indexStr = std::to_string(index);
        ASSERT_TRUE(G.AddTensor(DataType::DT_FP32, shape, "producer_out_" + indexStr));
        ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"input"}, {"producer_out_" + indexStr}, "producer_" + indexStr, true));
        ASSERT_TRUE(
            G.AddOp(Opcode::OP_ASSEMBLE, {"producer_out_" + indexStr}, {"shared"}, "assemble_" + indexStr, true));
        G.GetOp("assemble_" + indexStr)
            ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    }
    std::vector<std::string> outcasts;
    for (int index = 0; index < consumerNum; ++index) {
        outcasts.emplace_back("view_out_" + std::to_string(index));
    }
    ASSERT_TRUE(G.SetInCast({"input"}));
    ASSERT_TRUE(G.SetOutCast(outcasts));

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*G.GetFunction()), SUCCESS);

    ExpectTopologicalOrder(G.GetFunction());
}

TEST_F(MergeViewAssembleTest, TestMergeViewAssemble)
{
    constexpr int32_t tilex = 8;
    constexpr int32_t tiley = 16;
    constexpr int expectedOps = 8;
    constexpr int expected_view = 0;
    constexpr int expected_slice = 4;
    constexpr int expected_add = 2;
    constexpr int expected_assemble = 0;
    constexpr int expected_contract = 2;
    std::vector<int64_t> shape{16, 16};
    Tensor a(DT_FP32, shape, "a");
    Tensor in_tensor(DT_FP32, shape, "in_tensor");
    Tensor out_tensor(DT_FP32, shape, "out_tensor");

    TileShape::Current().SetVecTile(tilex, tiley);

    PassManager& passManager = PassManager::Instance();
    passManager.RegisterStrategy("ViewAssembleTestStrategy",
                                 {
                                     {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
                                     {"InferMemoryConflict", PassName::INFER_MEMORY_CONFLICT},
                                     {"ExpandFunction", PassName::EXPAND_FUNCTION},
                                     {"SplitReshape", PassName::SPLIT_RESHAPE},
                                     {"SplitRawTensor", PassName::SPLIT_RAW_TENSOR},
                                     {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                 });

    Function* originFunction = nullptr;
    std::vector<int64_t> originOpmagic;
    FUNCTION("AddFunction")
    {
        out_tensor = Add(in_tensor, a);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "Current function pointer is null";
        auto operations = originFunction->Operations();
        for (const auto& op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/merge_view_assemble.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_AddFunction");

    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currentFunction);
    mergeViewAssemble.RunOnFunction(*currentFunction);
    mergeViewAssemble.PostCheck(*currentFunction);

    // ================== Verify Pass Effect ==================
    auto updated_operations = currentFunction->Operations();
    EXPECT_EQ(updated_operations.size(), expectedOps) << "8 operations should remain";
    int view_count = 0;
    int slice_count = 0;
    int add_count = 0;
    int assemble_count = 0;
    int contract_count = 0;
    for (const auto& op : updated_operations) {
        if (op.GetOpcodeStr() == "VIEW") {
            view_count++;
        } else if (op.GetOpcodeStr() == "ADD") {
            add_count++;
        } else if (op.GetOpcodeStr() == "SLICE") {
            slice_count++;
        } else if (op.GetOpcodeStr() == "ASSEMBLE") {
            assemble_count++;
        } else if (op.GetOpcodeStr() == "CONTRACT") {
            contract_count++;
        }
    }

    EXPECT_EQ(view_count, expected_view) << "0 VIEW operations should remain";
    EXPECT_EQ(slice_count, expected_slice) << "4 SLICE operations should remain";
    EXPECT_EQ(add_count, expected_add) << "2 ADD operations should remain";
    EXPECT_EQ(assemble_count, expected_assemble) << "0 boundary ASSEMBLE operation should remain";
    EXPECT_EQ(contract_count, expected_contract) << "2 CONTRACT operation should remain";

    // Check the offset of the View operation
}

TEST_F(MergeViewAssembleTest, MergeTwoConsecutiveViews)
{
    Program program;
    std::string funcMagicName = "test_function";
    std::string funcRawName = "test_function_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);
    // 创建原始输入tensor
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);
    // 创建第一个VIEW操作，偏移量[1,2]
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{8, 8},
                                                                CreateTestConstIntVector(std::vector<int64_t>{8, 8}));
    auto view1Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2},    // from_offset
                                                       std::vector<SymbolicScalar>{}, // from_dyn_offset
                                                       std::vector<SymbolicScalar>{}  // to_dyn_valid_shape
    );
    auto& view1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {inputTensor}, {midTensor});
    view1Op.SetOpAttribute(view1Attr);

    // 2. 创建第2个VIEW操作，偏移量[3,4]
    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 6}, CreateTestConstIntVector(std::vector<int64_t>{6, 6}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto view2Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{3, 4},    // from_offset
                                                       std::vector<SymbolicScalar>{}, // from_dyn_offset
                                                       std::vector<SymbolicScalar>{}  // to_dyn_valid_shape
    );
    auto& view2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {midTensor}, {outputTensor});
    view2Op.SetOpAttribute(view2Attr);

    // 3. 执行MergeViewAssemble pass
    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // 4. 验证结果
    // 4.1 检查原始VIEW操作是否被标记为删除

    const auto& operations = function->Operations();
    EXPECT_FALSE(operations.Contains(view1Op));
    EXPECT_FALSE(operations.Contains(view2Op));

    // 4.2 检查合并后的VIEW操作
    int viewOpCount = 0;
    Operation* mergedViewOp = nullptr;
    for (auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW && !op.IsDeleted()) {
            viewOpCount++;
            mergedViewOp = &op;
        }
    }

    ASSERT_EQ(viewOpCount, 1) << "只有一个合并后的VIEW操作";
    ASSERT_NE(mergedViewOp, nullptr);

    // 4.3 检查合并后的偏移量是否正确
    auto mergedAttr = dynamic_cast<ViewOpAttribute*>(mergedViewOp->GetOpAttribute().get());
    ASSERT_NE(mergedAttr, nullptr);

    const auto& mergedOffset = mergedAttr->GetFromOffset();
    ASSERT_EQ(mergedOffset.size(), 2);
    EXPECT_EQ(mergedOffset[0], 4) << "第一个维度偏移量应为4";
    EXPECT_EQ(mergedOffset[1], 6) << "第二个维度偏移量应为6";

    // 4.4 检查输入输出tensor是否正确
    ASSERT_EQ(mergedViewOp->GetIOperands().size(), 1);
    ASSERT_EQ(mergedViewOp->GetOOperands().size(), 1);
    EXPECT_EQ(mergedViewOp->GetInputOperand(0), inputTensor);
    EXPECT_EQ(mergedViewOp->GetOutputOperand(0), outputTensor);

    // 4.5 检查中间tensor是否被清理
    bool midTensorExists = false;
    for (const auto& item : GraphUtils::GetAllTensors(*function)) {
        if (item == midTensor) {
            midTensorExists = true;
            break;
        }
    }
    EXPECT_FALSE(midTensorExists) << "中间tensor应该被清理";
}

TEST_F(MergeViewAssembleTest, MergeThreeConsecutiveAssembles)
{
    Program program;
    std::string funcMagicName = "test_function";
    std::string funcRawName = "test_function_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);

    // 1.创建原始输入tensor并设置incast
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);

    // 2.创建三个连续的ASSEMBLE操作
    // 第一个ASSEMBLE：偏移量[1,0]
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{9, 10},
                                                                 CreateTestConstIntVector(std::vector<int64_t>{9, 10}));
    auto assemble1Attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0},   // to_offset
                                                               std::vector<SymbolicScalar>{} // to_dyn_offset
    );
    auto& assemble1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {inputTensor}, {midTensor1});
    assemble1Op.SetOpAttribute(assemble1Attr);

    // 第二个ASSEMBLE：偏移量[0,2]
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{9, 8},
                                                                 CreateTestConstIntVector(std::vector<int64_t>{9, 8}));
    auto assemble2Attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2},   // to_offset
                                                               std::vector<SymbolicScalar>{} // to_dyn_offset
    );
    auto& assemble2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {midTensor1}, {midTensor2});
    assemble2Op.SetOpAttribute(assemble2Attr);

    // 第三个ASSEMBLE：偏移量[3,0]
    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 8}, CreateTestConstIntVector(std::vector<int64_t>{6, 8}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto assemble3Attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{3, 0},   // to_offset
                                                               std::vector<SymbolicScalar>{} // to_dyn_offset
    );
    auto& assemble3Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {midTensor2}, {outputTensor});
    assemble3Op.SetOpAttribute(assemble3Attr);

    // 3.执行MergeViewAssemble pass
    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // 4.验证结果
    // 4.1检查原始ASSEMBLE操作是否被标记为已删除
    const auto& operations = function->Operations();
    EXPECT_FALSE(operations.Contains(assemble1Op));
    EXPECT_FALSE(operations.Contains(assemble2Op));
    EXPECT_FALSE(operations.Contains(assemble3Op));

    // 4.2检查合并后的ASSEMBLE操作。输入是合法 incast，合并后的写操作必须保留。
    ASSERT_EQ(operations.size(), 1);
    EXPECT_EQ(operations.begin()->GetOpcode(), Opcode::OP_ASSEMBLE);
    EXPECT_EQ(operations.begin()->GetIOperands().front(), inputTensor);
    EXPECT_EQ(operations.begin()->GetOOperands().front(), outputTensor);

    // 4.3检查中间tensor是否被清理
    bool midTensor1Exists = false;
    bool midTensor2Exists = false;
    for (const auto& item : GraphUtils::GetAllTensors(*function)) {
        if (item == midTensor1) {
            midTensor1Exists = true;
        }
        if (item == midTensor2) {
            midTensor2Exists = true;
        }
    }
    EXPECT_FALSE(midTensor1Exists) << "中间tensor1应该被清理";
    EXPECT_FALSE(midTensor2Exists) << "中间tensor2应该被清理";
}

TEST_F(MergeViewAssembleTest, ViewAssembleChainShouldNotMerge)
{
    // 创建计算图构建器
    ComputationalGraphBuilder G;

    // 创建测试tensor
    std::vector<std::string> tensorNames = {"input", "view1_out", "assemble_out", "view2_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));

    // 创建op链: VIEW1 -> ASSEMBLE -> VIEW2
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"input"}, {"view1_out"}, {"assemble_out"}};
    std::vector<std::vector<std::string>> ooperands{{"view1_out"}, {"assemble_out"}, {"view2_out"}};
    std::vector<std::string> opNames{"VIEW1", "ASSEMBLE", "VIEW2"};

    // 添加op
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 设置输入输出
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"view2_out"}));

    // 获取Function并验证
    Function* function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    // 为VIEW设置必要的属性
    size_t op_index = 0;
    for (auto& op : function->Operations()) {
        switch (op_index) {
            case 0: {                                                                        // VIEW1
                auto attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0},    // offset
                                                              MemoryType::MEM_UNKNOWN,       // toType
                                                              std::vector<SymbolicScalar>{}, // dynOffset
                                                              std::vector<SymbolicScalar>{}  // dynValidShape
                );
                op.SetOpAttribute(attr);
                break;
            }
            case 2: {                                                                        // VIEW2
                auto attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0},    // offset
                                                              MemoryType::MEM_UNKNOWN,       // toType
                                                              std::vector<SymbolicScalar>{}, // dynOffset
                                                              std::vector<SymbolicScalar>{}  // dynValidShape
                );
                op.SetOpAttribute(attr);
                break;
            }
            default:
                break;
        }
        op_index++;
    }

    // 记录原始op数量
    const size_t originalOpCount = function->Operations().size();
    ASSERT_EQ(originalOpCount, 3);

    // 执行pass
    MergeViewAssemble mva;
    Status status = mva.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // 验证op数量不变
    EXPECT_EQ(function->Operations().size(), originalOpCount);

    // 验证op顺序和类型保持不变
    const auto& ops = function->Operations();
    EXPECT_EQ(ops[0].GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(ops[1].GetOpcode(), Opcode::OP_ASSEMBLE);
    EXPECT_EQ(ops[2].GetOpcode(), Opcode::OP_VIEW);
}

TEST_F(MergeViewAssembleTest, Test2View2Assemble2View2AssembleChain)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames{"input",     "view1_out", "view2_out",     "assemble1_out", "assemble2_out",
                                         "view3_out", "view4_out", "assemble3_out", "assemble4_out", "final_out"};

    std::vector<Opcode> opCodes{Opcode::OP_VIEW,     Opcode::OP_VIEW,     Opcode::OP_ASSEMBLE,
                                Opcode::OP_ASSEMBLE, Opcode::OP_VIEW,     Opcode::OP_VIEW,
                                Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};

    std::vector<std::vector<std::string>> ioperands{{"input"},         {"view1_out"},     {"view2_out"},
                                                    {"assemble1_out"}, {"assemble2_out"}, {"view3_out"},
                                                    {"view4_out"},     {"assemble3_out"}, {"assemble4_out"}};

    std::vector<std::vector<std::string>> ooperands{{"view1_out"},     {"view2_out"},     {"assemble1_out"},
                                                    {"assemble2_out"}, {"view3_out"},     {"view4_out"},
                                                    {"assemble3_out"}, {"assemble4_out"}, {"final_out"}};

    std::vector<std::string> opNames{"view1", "view2",     "assemble1", "assemble2", "view3",
                                     "view4", "assemble3", "assemble4", "abs"};

    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10, 10}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_EQ(G.SetInCast({"input"}), true);
    EXPECT_EQ(G.SetOutCast({"final_out"}), true);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    size_t op_index = 0;
    for (auto& op : function->Operations()) {
        switch (op_index) {
            case 0: { // view1
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{1, 1, 1}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 1: { // view2
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{2, 2, 2}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 2: { // assemble1
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{3, 3, 3},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 3: { // assemble2
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{4, 4, 4},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 4: { // view3
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{5, 5, 5}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 5: { // view4
                auto attr = std::make_shared<ViewOpAttribute>(
                    std::vector<int64_t>{6, 6, 6}, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 6: { // assemble3
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{7, 7, 7},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            case 7: { // assemble4
                auto attr = std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{8, 8, 8},
                                                                  std::vector<SymbolicScalar>{});
                op.SetOpAttribute(attr);
                break;
            }
            default:
                // No attribute needed for other operations (e.g. ABS)
                break;
        }
        op_index++;
    }

    MergeViewAssemble pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    const auto& operations = function->Operations();
    int view_count = 0;
    int assemble_count = 0;

    for (const auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count++;
        }
    }

    // Should have 2 merged views and 2 merged assembles
    EXPECT_EQ(view_count, 2);
    EXPECT_EQ(assemble_count, 2);

    // Verify final graph structure
    bool found_structure = false;
    for (const auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_ABS) {
            const auto* abs_input = op.GetIOperands()[0].get();
            ASSERT_NE(abs_input, nullptr);

            const auto* last_assemble = *abs_input->GetProducers().begin();
            ASSERT_NE(last_assemble, nullptr);
            EXPECT_EQ(last_assemble->GetOpcode(), Opcode::OP_ASSEMBLE);

            const auto* last_view = *last_assemble->GetIOperands()[0]->GetProducers().begin();
            ASSERT_NE(last_view, nullptr);
            EXPECT_EQ(last_view->GetOpcode(), Opcode::OP_VIEW);

            const auto* first_assemble = *last_view->GetIOperands()[0]->GetProducers().begin();
            ASSERT_NE(first_assemble, nullptr);
            EXPECT_EQ(first_assemble->GetOpcode(), Opcode::OP_ASSEMBLE);

            const auto* first_view = *first_assemble->GetIOperands()[0]->GetProducers().begin();
            ASSERT_NE(first_view, nullptr);
            EXPECT_EQ(first_view->GetOpcode(), Opcode::OP_VIEW);

            found_structure = true;
            break;
        }
    }
    EXPECT_TRUE(found_structure);
}

TEST_F(MergeViewAssembleTest, TestMixedBranchWithViewAndAssemble)
{
    ComputationalGraphBuilder G;

    // 定义张量 (包含主分支和两个不同类型的子分支)
    std::vector<std::string> tensorNames{"input",
                                         // 主分支
                                         "view1_out", "view2_out", "assemble1_out",
                                         // 分支1 (VIEW分支)
                                         "view3_out", "view4_out", "assemble2_out",
                                         // 分支2 (Assemble分支)
                                         "view5_out", "assemble3_out", "assemble4_out",
                                         // 合并输出
                                         "merged_out", "final_out"};

    // 操作序列 (包含混合类型分支)
    std::vector<Opcode> opCodes{
        // 主分支
        Opcode::OP_VIEW,     // view1
        Opcode::OP_VIEW,     // view2
        Opcode::OP_ASSEMBLE, // assemble1
        // 分支1 (VIEW分支)
        Opcode::OP_VIEW,     // view3
        Opcode::OP_VIEW,     // view4
        Opcode::OP_ASSEMBLE, // assemble2
        // 分支2 (Assemble分支)
        Opcode::OP_ASSEMBLE, // assemble3
        Opcode::OP_VIEW,     // view5
        Opcode::OP_ASSEMBLE, // assemble4
        // 合并
        Opcode::OP_ASSEMBLE, // merge_assemble
        Opcode::OP_ABS       // final
    };

    // 定义输入输出关系
    std::vector<std::vector<std::string>> ioperands{
        {"input"},                          // view1
        {"view1_out"},                      // view2
        {"view2_out"},                      // assemble1
        {"assemble1_out"},                  // view3 (分支1)
        {"view3_out"},                      // view4 (分支1)
        {"view4_out"},                      // assemble2 (分支1)
        {"assemble1_out"},                  // assemble3 (分支2)
        {"assemble3_out"},                  // view5 (分支2)
        {"view5_out"},                      // assemble4 (分支2)
        {"assemble2_out", "assemble4_out"}, // merge_assemble
        {"merged_out"}                      // abs
    };

    std::vector<std::vector<std::string>> ooperands{
        {"view1_out"},     {"view2_out"}, {"assemble1_out"}, {"view3_out"},  {"view4_out"}, {"assemble2_out"},
        {"assemble3_out"}, {"view5_out"}, {"assemble4_out"}, {"merged_out"}, {"final_out"}};

    // 添加操作名称列表
    std::vector<std::string> opNames{
        "view1",          "view2",    "assemble1", // 主分支
        "view3",          "view4",    "assemble2", // 分支1 (VIEW分支)
        "assemble3",      "view5",    "assemble4", // 分支2 (Assemble分支)
        "merge_assemble", "abs_final"              // 合并与最终操作
    };

    // 构建计算图
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {20, 20, 20}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 辅助函数：安全设置属性
    auto set_attr = [&G](const std::string& op_name, auto attr) {
        auto* op = G.GetOp(op_name);
        ASSERT_NE(op, nullptr) << "Operation " << op_name << " not found!";
        op->SetOpAttribute(attr);
    };

    // ------------------------- 主分支属性设置 -------------------------
    // view1: offset=[1,0,0]
    set_attr("view1", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 0, 0}, // offset
                                                        std::vector<SymbolicScalar>{}, // stride (空表示默认)
                                                        std::vector<SymbolicScalar>{} // shape (空表示保持输入形状)
                                                        ));

    // view2: offset=[0,2,0]
    set_attr("view2", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 2, 0}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // assemble1: offset=[1,2,0]
    set_attr("assemble1", std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 2, 0}, // offset
                                                                std::vector<SymbolicScalar>{} // 其他参数（如无则空）
                                                                ));

    // ------------------------- 分支1 (VIEW分支) 属性设置 -------------------------
    // view3: offset=[0,0,3]
    set_attr("view3", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0, 3}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // view4: offset=[4,0,0]
    set_attr("view4", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{4, 0, 0}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // assemble2: offset=[4,0,3]
    set_attr("assemble2",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{4, 0, 3}, std::vector<SymbolicScalar>{}));

    // ------------------------- 分支2 (Assemble分支) 属性设置 -------------------------
    // assemble3: offset=[5,0,0]
    set_attr("assemble3",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{5, 0, 0}, std::vector<SymbolicScalar>{}));

    // view5: offset=[0,6,0]
    set_attr("view5", std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 6, 0}, std::vector<SymbolicScalar>{},
                                                        std::vector<SymbolicScalar>{}));

    // assemble4: offset=[5,6,0]
    set_attr("assemble4",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{5, 6, 0}, std::vector<SymbolicScalar>{}));

    // ------------------------- 合并操作属性设置 -------------------------
    // merge_assemble: offset=[9,9,9]
    set_attr("merge_assemble",
             std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{9, 9, 9}, std::vector<SymbolicScalar>{}));

    // ABS操作无需属性
    // ------------------------- 设置输入输出 -------------------------
    EXPECT_EQ(G.SetInCast({"input"}), true);
    EXPECT_EQ(G.SetOutCast({"final_out"}), true);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    // 执行pass
    MergeViewAssemble pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // 验证结果
    const auto& operations = function->Operations();

    // 1. 验证操作合并情况
    int view_count = 0;
    int assemble_count = 0;
    for (const auto& op : operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count++;
        }
    }
    EXPECT_EQ(view_count, 3);     // 应合并为3个VIEW操作
    EXPECT_EQ(assemble_count, 4); // 带 VIEW 旁路消费的 assemble 链不能合并

    // 验证final_out的生成路径
    bool found_abs = false;
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ABS) {
            // 1. 检查输入数量是否为1
            const auto& abs_inputs = op.GetIOperands();
            ASSERT_EQ(abs_inputs.size(), 1) << "OP_ABS should have exactly 1 input tensor";

            // 2. 获取输入Tensor及其生产者
            const auto* input_tensor = abs_inputs[0].get();
            ASSERT_NE(input_tensor, nullptr) << "Input tensor is null";

            const auto& producers = input_tensor->GetProducers();
            ASSERT_EQ(producers.size(), 2) << "Input tensor should have 2 producers";

            // 3. 验证两个生产者均为ASSEMBLE操作
            int assemble_producer_count = 0;
            for (const auto* producer : producers) {
                ASSERT_NE(producer, nullptr) << "Producer is null";
                if (producer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                    assemble_producer_count++;
                }
            }
            EXPECT_EQ(assemble_producer_count, 2)
                << "Expected 2 ASSEMBLE producers, but got " << assemble_producer_count;

            found_abs = true;
            break;
        }
    }
    EXPECT_TRUE(found_abs) << "OP_ABS operation not found in the graph";
}
void MergeViewL1DataMoveGraph(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> input_cast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> input_cast1_view = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> input_cast2_view = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> redundant_view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> redundant_view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> op_view_L1_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> op_view_L1_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{64, 16}, CreateTestConstIntVector(std::vector<int64_t>{64, 16}));
    std::shared_ptr<LogicalTensor> view_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> view_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> view_out3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> view_out4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> l0a_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> l0a_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    std::shared_ptr<LogicalTensor> l0b_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> l0b_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> a_mul_b_out1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    std::shared_ptr<LogicalTensor> a_mul_b_out2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 16}, CreateTestConstIntVector(std::vector<int64_t>{32, 16}));
    // std::shared_ptr<LogicalTensor> output_cast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
    // CreateTestConstIntVector(shape));
    auto& head_view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1},
                                                         {input_cast1_view});
    head_view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& head_view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2},
                                                         {input_cast2_view});
    head_view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));

    auto& head_view_op11 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast1_view},
                                                          {redundant_view_out1});
    head_view_op11.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& head_view_op12 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {input_cast2_view},
                                                          {redundant_view_out2});
    head_view_op12.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));

    auto& view_L1_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {redundant_view_out1},
                                                       {op_view_L1_out1});
    std::vector<int> newoffset{0, 0};
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    view_L1_op1.SetOpAttribute(viewAttribute);

    auto& view_L1_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {redundant_view_out2},
                                                       {op_view_L1_out2});
    view_L1_op2.SetOpAttribute(viewAttribute);

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out1}, {view_out1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out1}, {view_out2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 32}));
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out2}, {view_out3});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {op_view_L1_out2}, {view_out4});
    view_op4.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 0}));

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {view_out1}, {l0a_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0A, {view_out2}, {l0a_out2});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {view_out3}, {l0b_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_L1_TO_L0B, {view_out4}, {l0b_out2});

    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a_out1, l0b_out1}, {a_mul_b_out1});
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {l0a_out2, l0b_out2}, {a_mul_b_out2});

    currFunctionPtr->inCasts_.push_back(input_cast1);
    currFunctionPtr->inCasts_.push_back(input_cast2);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out1);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out2);
}
TEST_F(MergeViewAssembleTest, MergeViewL1DataMove)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "MergeViewL1DataMove",
                                                      "MergeViewL1DataMove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("MergeViewL1DataMove", currFunctionPtr);

    MergeViewL1DataMoveGraph(currFunctionPtr);

    // 验证构图
    int view_count = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        }
    }
    EXPECT_EQ(view_count, 10);

    std::stringstream ssBefore;
    ssBefore << "Before_MergeViewAssemble";

    // Call the pass
    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_before.json");
    mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_after.json");
    mergeViewAssemble.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_MergeViewAssemble";

    // Validate the results
    int view_count_after_pass = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count_after_pass++;
        }
    }
    EXPECT_EQ(view_count_after_pass, NUM6);
}
void MergeViewWithAttr(std::shared_ptr<Function>& currFunctionPtr)
{
    std::shared_ptr<LogicalTensor> view_in = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));
    std::shared_ptr<LogicalTensor> view_out = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 64}, CreateTestConstIntVector(std::vector<int64_t>{32, 64}));

    auto& view_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {view_in}, {tensor1});
    auto viewAttribute1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute1->SetToType(MemoryType::MEM_UNKNOWN);
    view_op1.SetOpAttribute(viewAttribute1);
    auto& view_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto viewAttribute2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute2->SetToType(MemoryType::MEM_BT);
    view_op2.SetOpAttribute(viewAttribute2);
    auto& view_op3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor2}, {tensor3});
    auto viewAttribute3 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute3->SetToType(MemoryType::MEM_BT);
    view_op3.SetOpAttribute(viewAttribute3);
    auto& view_op4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor3}, {tensor4});
    auto viewAttribute4 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute4->SetToType(MemoryType::MEM_UB);
    view_op4.SetOpAttribute(viewAttribute4);
    auto& view_op5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {tensor4}, {view_out});
    auto viewAttribute5 = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute5->SetToType(MemoryType::MEM_UB);
    view_op5.SetOpAttribute(viewAttribute5);

    currFunctionPtr->inCasts_.push_back(view_in);
    currFunctionPtr->outCasts_.push_back(view_out);
}
TEST_F(MergeViewAssembleTest, MergeViewWithAttr)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "MergeViewWithAttr", "MergeViewWithAttr",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("MergeViewWithAttr", currFunctionPtr);

    MergeViewWithAttr(currFunctionPtr);

    // 验证构图
    int view_count = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count++;
        }
    }
    EXPECT_EQ(view_count, NUM5);

    std::stringstream ssBefore;
    ssBefore << "Before_MergeViewAssemble";

    // Call the pass
    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_before.json");
    mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/mergeViewAssemble_L1DataMove_after.json");
    mergeViewAssemble.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_MergeViewAssemble";

    // Validate the results
    int view_count_after_pass = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count_after_pass++;
        }
    }
    EXPECT_EQ(view_count_after_pass, NUM2);
}

TEST_F(MergeViewAssembleTest, MergeViewInheritIsGemv)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "MergeViewInheritIsGemv",
                                                      "MergeViewInheritIsGemv", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("MergeViewInheritIsGemv", currFunctionPtr);

    MergeViewWithAttr(currFunctionPtr);
    // 给链尾 VIEW 设置 isGemv 属性（模拟 GEMV 的 DDR→L1 VIEW 信号）
    Operation* lastUbView = nullptr;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            lastUbView = &op;
        }
    }
    ASSERT_NE(lastUbView, nullptr);
    lastUbView->SetAttribute(OpAttributeKey::isGemv, static_cast<int64_t>(1));

    MergeViewAssemble mergeViewAssemble;
    mergeViewAssemble.PreCheck(*currFunctionPtr);
    mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    mergeViewAssemble.PostCheck(*currFunctionPtr);

    // 白名单继承：合并后的 VIEW 中至少一个带 isGemv 属性
    bool isGemvInherited = false;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            int64_t isGemv = 0;
            if (op.GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv) && isGemv != 0) {
                isGemvInherited = true;
                break;
            }
        }
    }
    EXPECT_TRUE(isGemvInherited);
}

TEST_F(MergeViewAssembleTest, TestPreCheck)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMergeViewAssemble",
                                                      "TestMergeViewAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {8, 4};
    std::vector<int64_t> shape2 = {1, 8, 4};
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {inCast}, {ubTensor});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {ubTensor}, {});

    currFunctionPtr->inCasts_.push_back(inCast);

    MergeViewAssemble mergeViewAssemble;
    Status status = mergeViewAssemble.PreCheck(*currFunctionPtr);
    EXPECT_EQ(status, FAILED);
}

/*
TestSpan
inCast{16,16}->view->ubTensor1{8,4}->assemble->ubTensor2{8,4}->assemble->outCast{8,4}
inCast{16,16}->view->ubTensor1{8,4}->assemble->outCast{8,4}
*/
TEST_F(MergeViewAssembleTest, TestSpan)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSpan", "TestSpan", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {16, 16};
    std::vector<int64_t> shape2 = {8, 4};
    auto span1 = ir::Span("view", NUM1, 0);
    auto span2 = ir::Span("view", NUM2, 0);
    auto inCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));

    auto& viewOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_VIEW, {inCast}, {ubTensor1}, span1);
    auto viewAttribute = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0});
    viewAttribute->SetToType(MemoryType::MEM_UB);
    viewOp.SetOpAttribute(viewAttribute);
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor1}, {ubTensor2}, span2);
    IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ASSEMBLE, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->inCasts_.push_back(outCast);
    MergeViewAssemble mergeViewAssemble;
    Status status = mergeViewAssemble.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            EXPECT_EQ(op.GetSpan().BeginLine(), NUM2);
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            EXPECT_EQ(op.GetSpan().BeginLine(), NUM1);
        }
    }
}

TEST_F(MergeViewAssembleTest, ViewChainSameScopeIdsShouldMerge)
{
    Program program;
    std::string funcMagicName = "test_view_same_scope";
    std::string funcRawName = "test_view_same_scope_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);

    // VIEW1(scopeId=1) -> VIEW2(scopeId=1), should merge
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{8, 8},
                                                                CreateTestConstIntVector(std::vector<int64_t>{8, 8}));
    auto view1Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {inputTensor}, {midTensor});
    view1Op.SetOpAttribute(view1Attr);
    view1Op.SetScopeId(1);

    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 6}, CreateTestConstIntVector(std::vector<int64_t>{6, 6}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto view2Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{3, 4}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {midTensor}, {outputTensor});
    view2Op.SetOpAttribute(view2Attr);
    view2Op.SetScopeId(1);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // Verify: 1 merged VIEW with scopeId=1
    int viewCount = 0;
    int mergedScopeId = -1;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW && !op.IsDeleted()) {
            viewCount++;
            mergedScopeId = op.GetScopeId();
        }
    }
    EXPECT_EQ(viewCount, 1);
    EXPECT_EQ(mergedScopeId, 1);
}

TEST_F(MergeViewAssembleTest, ViewChainMixedDefaultAndValidScopeIdShouldMerge)
{
    Program program;
    std::string funcMagicName = "test_view_mixed_scope";
    std::string funcRawName = "test_view_mixed_scope_raw";
    std::unique_ptr<Function> function = std::make_unique<Function>(program, funcMagicName, funcRawName, nullptr);
    auto rawTensor = std::make_shared<RawTensor>(DataType::DT_FP32, std::vector<int64_t>{10, 10},
                                                 TileOpFormat::TILEOP_ND, "input_tensor");
    std::shared_ptr<LogicalTensor> inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{10, 10},
        CreateTestConstIntVector(std::vector<int64_t>{10, 10}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetIncast()).push_back(inputTensor);

    // VIEW1(scopeId=-1, default) -> VIEW2(scopeId=1), should merge and inherit scopeId=1
    auto midTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DataType::DT_FP32, std::vector<int64_t>{8, 8},
                                                                CreateTestConstIntVector(std::vector<int64_t>{8, 8}));
    auto view1Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{1, 2}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view1Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {inputTensor}, {midTensor});
    view1Op.SetOpAttribute(view1Attr);

    auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, std::vector<int64_t>{6, 6}, CreateTestConstIntVector(std::vector<int64_t>{6, 6}));
    const_cast<std::vector<std::shared_ptr<LogicalTensor>>&>(function->GetOutcast()).push_back(outputTensor);
    auto view2Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{3, 4}, std::vector<SymbolicScalar>{},
                                                       std::vector<SymbolicScalar>{});
    auto& view2Op = IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_VIEW, {midTensor}, {outputTensor});
    view2Op.SetOpAttribute(view2Attr);
    view2Op.SetScopeId(1);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // Verify: 1 merged VIEW with scopeId=1
    int viewCount = 0;
    int mergedScopeId = -2;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW && !op.IsDeleted()) {
            viewCount++;
            mergedScopeId = op.GetScopeId();
        }
    }
    EXPECT_EQ(viewCount, 1);
    EXPECT_EQ(mergedScopeId, 1);
}

TEST_F(MergeViewAssembleTest, AssembleChainDifferentScopeIdsShouldNotMerge)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames = {"input", "view_out", "assemble1_out", "assemble2_out", "final_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, tensorNames));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"view_out"}, {"assemble1_out"}, {"assemble2_out"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"view_out"}, {"assemble1_out"}, {"assemble2_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "assemble1", "assemble2", "abs"};

    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* view1 = G.GetOp("view1");
    view1->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
                                                            std::vector<SymbolicScalar>{}));

    auto* assemble1 = G.GetOp("assemble1");
    assemble1->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{}));
    assemble1->SetScopeId(1);

    auto* assemble2 = G.GetOp("assemble2");
    assemble2->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2}, std::vector<SymbolicScalar>{}));
    assemble2->SetScopeId(2);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // Verify: 2 ASSEMBLEs remain (not merged due to different scopeIds)
    int assembleCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            assembleCount++;
        }
    }
    EXPECT_EQ(assembleCount, 2);
}

TEST_F(MergeViewAssembleTest, MergeAssembleChainShouldInheritAtomicAddAttr)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames = {"input", "view_out", "assemble1_out", "assemble2_out", "final_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, tensorNames));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"view_out"}, {"assemble1_out"}, {"assemble2_out"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"view_out"}, {"assemble1_out"}, {"assemble2_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "assemble1", "assemble2", "abs"};

    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* view1 = G.GetOp("view1");
    ASSERT_NE(view1, nullptr);
    view1->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
                                                            std::vector<SymbolicScalar>{}));

    auto* assemble1 = G.GetOp("assemble1");
    ASSERT_NE(assemble1, nullptr);
    assemble1->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{}));
    assemble1->SetAttribute(RMW_MODE_ATTR_ADD, 1L);

    auto* assemble2 = G.GetOp("assemble2");
    ASSERT_NE(assemble2, nullptr);
    assemble2->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2}, std::vector<SymbolicScalar>{}));

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    int assembleCount = 0;
    Operation* mergedAssemble = nullptr;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            assembleCount++;
            mergedAssemble = &op;
        }
    }
    ASSERT_EQ(assembleCount, 1);
    ASSERT_NE(mergedAssemble, nullptr);
    EXPECT_TRUE(mergedAssemble->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(MergeViewAssembleTest, MergeAssembleChainShouldInheritAtomicSemanticAttrs)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames = {"input", "view_out", "assemble1_out", "assemble2_out", "final_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, tensorNames));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"view_out"}, {"assemble1_out"}, {"assemble2_out"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"view_out"}, {"assemble1_out"}, {"assemble2_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "assemble1", "assemble2", "abs"};

    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* view1 = G.GetOp("view1");
    ASSERT_NE(view1, nullptr);
    view1->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
                                                            std::vector<SymbolicScalar>{}));

    auto* assemble1 = G.GetOp("assemble1");
    ASSERT_NE(assemble1, nullptr);
    assemble1->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{}));
    assemble1->SetAttribute(RMW_MODE_ATTR_ADD, 1L);
    assemble1->SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);

    auto* assemble2 = G.GetOp("assemble2");
    ASSERT_NE(assemble2, nullptr);
    assemble2->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2}, std::vector<SymbolicScalar>{}));
    assemble2->SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    int assembleCount = 0;
    Operation* mergedAssemble = nullptr;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            assembleCount++;
            mergedAssemble = &op;
        }
    }
    ASSERT_EQ(assembleCount, 1);
    ASSERT_NE(mergedAssemble, nullptr);
    EXPECT_TRUE(mergedAssemble->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_TRUE(mergedAssemble->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR));
    EXPECT_TRUE(mergedAssemble->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR));
}

TEST_F(MergeViewAssembleTest, AssembleChainWithDifferentRmwModeAttrsShouldNotMerge)
{
    ComputationalGraphBuilder G;

    std::vector<std::string> tensorNames = {"input", "view_out", "assemble1_out", "assemble2_out", "final_out"};
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, tensorNames));

    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"view_out"}, {"assemble1_out"}, {"assemble2_out"}};
    std::vector<std::vector<std::string>> ooperands = {
        {"view_out"}, {"assemble1_out"}, {"assemble2_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "assemble1", "assemble2", "abs"};

    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);

    auto* view1 = G.GetOp("view1");
    ASSERT_NE(view1, nullptr);
    view1->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{},
                                                            std::vector<SymbolicScalar>{}));

    auto* assemble1 = G.GetOp("assemble1");
    ASSERT_NE(assemble1, nullptr);
    assemble1->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{}));
    assemble1->SetAttribute(RMW_MODE_ATTR_ADD, 1L);

    auto* assemble2 = G.GetOp("assemble2");
    ASSERT_NE(assemble2, nullptr);
    assemble2->SetOpAttribute(
        std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 2}, std::vector<SymbolicScalar>{}));
    assemble2->SetAttribute(RMW_MODE_ATTR_MIN, 1L);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    int assembleCount = 0;
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            assembleCount++;
        }
    }
    EXPECT_EQ(assembleCount, 2);
}

TEST_F(MergeViewAssembleTest, LinearAssembleChainPreservesTokenBoundary)
{
    Program program;
    auto function = std::make_unique<Function>(program, "linear_token", "linear_token", nullptr);
    IRBuilder builder;
    std::vector<int64_t> shape{4, 4};
    auto input = builder.CreateTensorVar(*function, DataType::DT_FP32, shape);
    auto tokenInput = builder.CreateTensorVar(*function, DataType::DT_FP32, shape);
    auto assembleInput = builder.CreateTensorVar(*function, DataType::DT_FP32, shape);
    auto tokenOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, shape);
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, shape);
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, shape);
    auto finalOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, shape);
    function->inCasts_ = {input, tokenInput};
    function->outCasts_ = {tokenOutput, finalOutput};

    auto& dataProducer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {input}, {assembleInput});
    auto& tokenProducer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {tokenInput}, {tokenOutput});
    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {assembleInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& finalConsumer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {output}, {finalOutput});
    auto inputToken = AddTokenEdge(*function, tokenProducer, first);
    AddTokenEdge(*function, second, finalConsumer);

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    Operation* merged = nullptr;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ASSERT_EQ(merged, nullptr);
            merged = &op;
        }
    }
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->GetIOperands().front(), assembleInput);
    EXPECT_EQ(merged->GetOOperands().front(), output);
    EXPECT_NE(std::find(merged->tokens_.begin(), merged->tokens_.end(), inputToken), merged->tokens_.end());
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(inputToken, ToStmtPtr(*merged)));
    ASSERT_FALSE(merged->result_token_.empty());
    EXPECT_TRUE(function->GetVarDependency().HasProducer(merged->result_token_.front(), ToStmtPtr(*merged)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(merged->result_token_.front(), ToStmtPtr(finalConsumer)));
    EXPECT_TRUE(function->Operations(false).Contains(dataProducer));
}

TEST_F(MergeViewAssembleTest, CompleteProducerGroupFusesAndPreservesTokens)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group", "producer_group", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto finalOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {finalOutput};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& finalConsumer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {output}, {finalOutput});
    auto wawToken = AddTokenEdge(*function, first, second);
    AddTokenEdge(*function, downstream, finalConsumer);
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    std::vector<Operation*> replacements;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            replacements.emplace_back(&op);
            EXPECT_EQ(op.GetOOperands().front(), output);
        }
    }
    ASSERT_EQ(replacements.size(), 2);
    ASSERT_EQ(output->GetProducers().size(), 2);
    // 旧 wawToken 被清理：两条合并链写入 output 的不相交区域（coverage 保证），
    // WAW 顺序随之消解；downstream 的结果 token 扇出为两条链各自的新 token。
    EXPECT_EQ(function->GetVarDependency().GetProducers(wawToken).size(), 0);
    EXPECT_EQ(finalConsumer.tokens_.size(), 2);
    for (auto* replacement : replacements) {
        ASSERT_FALSE(replacement->result_token_.empty());
        bool waitedByFinalConsumer = false;
        for (const auto& token : replacement->result_token_) {
            waitedByFinalConsumer = waitedByFinalConsumer ||
                                    function->GetVarDependency().HasConsumer(token, ToStmtPtr(finalConsumer));
        }
        EXPECT_TRUE(waitedByFinalConsumer);
    }
    EXPECT_EQ(function->GetTensorMap().GetTensorByMagic(middle->GetMagic()), nullptr);
}

TEST_F(MergeViewAssembleTest, ProducerGroupMapsDataCoveredTokensToMatchingReplacement)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_data_tokens", "producer_group_data_tokens",
                                               nullptr);
    IRBuilder builder;
    auto firstSource = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondSource = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto syncSource = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto firstIntermediate = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondIntermediate = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto syncOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstSource, secondSource, syncSource};
    function->outCasts_ = {output, syncOutput};

    auto& firstDataProducer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {firstSource}, {firstIntermediate});
    auto& secondDataProducer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {secondSource},
                                                          {secondIntermediate});
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {firstIntermediate}, {firstInput});
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {secondIntermediate}, {secondInput});
    auto& syncProducer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {syncSource}, {syncOutput});
    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto firstToken = AddTokenEdge(*function, firstDataProducer, downstream);
    auto secondToken = AddTokenEdge(*function, secondDataProducer, downstream);
    auto syncToken = AddTokenEdge(*function, syncProducer, downstream);
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    std::vector<Operation*> replacements;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            replacements.emplace_back(&op);
        }
    }
    ASSERT_EQ(replacements.size(), 2);
    for (auto* replacement : replacements) {
        ASSERT_EQ(replacement->tokens_.size(), 2);
        auto ownToken = replacement->GetInputOperand(0) == firstInput ? firstToken : secondToken;
        auto otherToken = replacement->GetInputOperand(0) == firstInput ? secondToken : firstToken;
        EXPECT_NE(std::find(replacement->tokens_.begin(), replacement->tokens_.end(), ownToken),
                  replacement->tokens_.end());
        EXPECT_EQ(std::find(replacement->tokens_.begin(), replacement->tokens_.end(), otherToken),
                  replacement->tokens_.end());
        EXPECT_NE(std::find(replacement->tokens_.begin(), replacement->tokens_.end(), syncToken),
                  replacement->tokens_.end());
    }
    EXPECT_EQ(function->GetVarDependency().GetConsumers(firstToken).size(), 1);
    EXPECT_EQ(function->GetVarDependency().GetConsumers(secondToken).size(), 1);
    EXPECT_EQ(function->GetVarDependency().GetConsumers(syncToken).size(), 2);
}

TEST_F(MergeViewAssembleTest, CompleteProducerGroupWithConcreteDynOffsetsFuses)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_dyn", "producer_group_dyn", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {output};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
        std::vector<int64_t>{0, 0}, std::vector<SymbolicScalar>{SymbolicScalar(0), SymbolicScalar(0)}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
        std::vector<int64_t>{1, 0}, std::vector<SymbolicScalar>{SymbolicScalar(1), SymbolicScalar(0)}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t assembleCount = 0;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        ++assembleCount;
        EXPECT_EQ(op.GetOOperands().front(), output);
    }
    EXPECT_EQ(assembleCount, 2);
    EXPECT_TRUE(middle->GetProducers().empty());
    EXPECT_TRUE(middle->GetConsumers().empty());
}

TEST_F(MergeViewAssembleTest, NestedCompleteProducerGroupFusionsCollapseThroughAbsorption)
{
    Program program;
    auto function = std::make_unique<Function>(program, "nested_producer_groups", "nested_producer_groups", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto thirdInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto fourthInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto firstMiddle = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondMiddle = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto concatMiddle = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 8});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 8});
    function->inCasts_ = {firstInput, secondInput, thirdInput, fourthInput};
    function->outCasts_ = {output};

    auto addConcreteAssemble = [&builder, &function](const LogicalTensorPtr& input, const LogicalTensorPtr& target,
                                                     int64_t rowOffset) {
        auto& op = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {input}, {target});
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
            std::vector<int64_t>{rowOffset, 0},
            std::vector<SymbolicScalar>{SymbolicScalar(rowOffset), SymbolicScalar(0)}));
    };
    addConcreteAssemble(firstInput, firstMiddle, 0);
    addConcreteAssemble(secondInput, firstMiddle, 1);
    addConcreteAssemble(thirdInput, secondMiddle, 0);
    addConcreteAssemble(fourthInput, secondMiddle, 1);

    auto& firstConcat = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstMiddle}, {concatMiddle});
    firstConcat.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& secondConcat = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondMiddle}, {concatMiddle});
    secondConcat.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 4}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {concatMiddle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t assembleCount = 0;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        ++assembleCount;
        ASSERT_EQ(op.GetIOperands().size(), 1);
        ASSERT_EQ(op.GetOOperands().size(), 1);
        EXPECT_EQ(op.GetOOperands().front(), output);
    }
    // The outer group absorbs both inner fusions: every leaf writes the final output
    // directly at the composed offset; the legacy DCE removes all live middle edges.
    EXPECT_EQ(assembleCount, 4);
    EXPECT_EQ(output->GetProducers().size(), 4);
    EXPECT_TRUE(firstMiddle->GetProducers().empty());
    EXPECT_TRUE(firstMiddle->GetConsumers().empty());
    EXPECT_TRUE(secondMiddle->GetProducers().empty());
    EXPECT_TRUE(secondMiddle->GetConsumers().empty());
    EXPECT_TRUE(concatMiddle->GetProducers().empty());
    EXPECT_TRUE(concatMiddle->GetConsumers().empty());
}

TEST_F(MergeViewAssembleTest, ProducerGroupWithIntermediateViewConsumerKeepsDataEdge)
{
    NpuArchScope arch(NPUArch::DAV_3510);
    for (bool withToken : {false, true}) {
        Program program;
        auto function = std::make_unique<Function>(program, "producer_group_fanout", "producer_group_fanout", nullptr);
        IRBuilder builder;
        auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
        auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
        auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto viewOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        function->inCasts_ = {firstInput, secondInput};
        function->outCasts_ = {viewOutput, output};

        auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstInput}, {middle});
        first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
        second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
        auto& view = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {middle}, {viewOutput});
        view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
        downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        if (withToken) {
            AddTokenEdge(*function, first, view);
        }
        function->BuildTensorMap();

        MergeViewAssemble pass;
        ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
        EXPECT_EQ(middle->GetProducers().size(), 2);
        EXPECT_EQ(view.GetIOperands().front(), middle);
        EXPECT_FALSE(first.IsDeleted());
        EXPECT_FALSE(second.IsDeleted());
        if (withToken) {
            ASSERT_EQ(output->GetProducers().size(), 1);
            EXPECT_EQ((*output->GetProducers().begin())->GetIOperands().front(), middle);
            ASSERT_EQ(view.tokens_.size(), 1);
            EXPECT_TRUE(function->GetVarDependency().HasProducer(view.tokens_.front(), ToStmtPtr(first)));
        } else {
            ASSERT_EQ(output->GetProducers().size(), 2);
            for (auto* producer : output->GetProducers()) {
                const auto& input = producer->GetIOperands().front();
                EXPECT_TRUE(input == firstInput || input == secondInput);
                auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
                ASSERT_NE(attr, nullptr);
                EXPECT_EQ(attr->GetToOffset(), (std::vector<int64_t>{input == firstInput ? 0 : 2, 0}));
            }
        }
    }
}

// A rejected sibling must keep the shared tail, including when the conflict is
// inside the suffix rather than at its last op. Exercise both producer orders.
TEST_F(MergeViewAssembleTest, NestedSiblingRejectionPreservesEveryInputContribution)
{
    NpuArchScope arch(NPUArch::DAV_3510);
    for (bool tokenMode : {false, true}) {
        for (bool reverse : {false, true}) {
            for (int conflict = 0; conflict < 3; ++conflict) {
                IRContext::Get().SetAssembleNewLogicalTensor(tokenMode);
                Program program;
                auto function = std::make_unique<Function>(program, "rejected_sibling", "rejected_sibling", nullptr);
                IRBuilder builder;
                auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
                auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
                auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
                auto next = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
                auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
                auto sideOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
                function->inCasts_ = {firstInput, secondInput};
                function->outCasts_ = {output, sideOutput};
                auto add = [&](Opcode opcode, const LogicalTensorPtr& input, const LogicalTensorPtr& target,
                               int64_t row) -> Operation& {
                    auto& op = builder.CreateTensorOpStmt(*function, opcode, {input}, {target});
                    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{row, 0}));
                    return op;
                };
                Operation* first = nullptr;
                Operation* second = nullptr;
                for (bool secondProducer : {reverse, !reverse}) {
                    if (secondProducer) {
                        second = &add(conflict == 0 ? Opcode::OP_CONTRACT : Opcode::OP_ASSEMBLE, secondInput, middle,
                                      2);
                    } else {
                        first = &add(Opcode::OP_ASSEMBLE, firstInput, middle, 0);
                    }
                }
                auto& downstream = add(conflict == 0 ? Opcode::OP_CONTRACT : Opcode::OP_ASSEMBLE, middle, next, 0);
                auto& tail = add(Opcode::OP_ASSEMBLE, next, output, 0);
                builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {middle}, {sideOutput});
                if (conflict == 1) {
                    second->SetScopeId(2);
                    downstream.SetScopeId(1);
                } else if (conflict == 2) {
                    second->SetAttribute(RMW_MODE_ATTR_MIN, 1L);
                    downstream.SetAttribute(RMW_MODE_ATTR_ADD, 1L);
                }
                if (tokenMode) {
                    auto marker = builder.CreateTokenVar(first->GetSpan());
                    first->result_token_ = {marker};
                    function->GetVarDependency().AddProducer(marker, ToStmtPtr(*first));
                }
                std::vector<std::shared_ptr<Operation>> originalOps;
                for (auto& op : function->Operations(false)) {
                    originalOps.emplace_back(op.shared_from_this());
                }
                function->BuildTensorMap();
                MergeViewAssemble pass;
                ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
                EXPECT_FALSE(first->IsDeleted());
                EXPECT_FALSE(second->IsDeleted());
                EXPECT_FALSE(downstream.IsDeleted());
                EXPECT_FALSE(tail.IsDeleted());
                ASSERT_EQ(output->GetProducers().size(), 1);
                EXPECT_EQ(*output->GetProducers().begin(), &tail);
                ASSERT_EQ(tail.GetIOperands().size(), 1);
                EXPECT_EQ(tail.GetIOperands().front(), next);
                EXPECT_EQ(middle->GetProducers().size(), 2);
            }
        }
    }
}

TEST_F(MergeViewAssembleTest, RejectedSiblingCycleKeepsSharedTailAndTokens)
{
    for (bool reverse : {false, true}) {
        Program program;
        auto function = std::make_unique<Function>(program, "sibling_cycle", "sibling_cycle", nullptr);
        IRBuilder builder;
        auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
        auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
        auto sideInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
        auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto sideOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
        function->inCasts_ = {firstInput, secondInput, sideInput};
        function->outCasts_ = {output, sideOutput};
        Operation* first = nullptr;
        Operation* second = nullptr;
        for (bool secondProducer : {reverse, !reverse}) {
            auto& op = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE,
                                                  {secondProducer ? secondInput : firstInput}, {middle});
            op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{secondProducer ? 2 : 0, 0}));
            (secondProducer ? second : first) = &op;
        }
        auto& side = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {sideInput}, {sideOutput});
        auto& tail = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
        tail.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
        auto firstToken = AddTokenEdge(*function, *second, side);
        auto secondToken = AddTokenEdge(*function, side, tail);
        std::vector<std::shared_ptr<Operation>> originalOps;
        for (auto& op : function->Operations(false)) {
            originalOps.emplace_back(op.shared_from_this());
        }
        function->BuildTensorMap();
        MergeViewAssemble pass;
        ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
        EXPECT_FALSE(first->IsDeleted());
        EXPECT_FALSE(second->IsDeleted());
        EXPECT_FALSE(tail.IsDeleted());
        EXPECT_EQ(middle->GetProducers().size(), 2);
        ASSERT_EQ(output->GetProducers().size(), 1);
        EXPECT_EQ(*output->GetProducers().begin(), &tail);
        EXPECT_TRUE(function->GetVarDependency().HasProducer(firstToken, ToStmtPtr(*second)));
        EXPECT_TRUE(function->GetVarDependency().HasConsumer(firstToken, ToStmtPtr(side)));
        EXPECT_TRUE(function->GetVarDependency().HasProducer(secondToken, ToStmtPtr(side)));
        EXPECT_TRUE(function->GetVarDependency().HasConsumer(secondToken, ToStmtPtr(tail)));
    }
}

TEST_F(MergeViewAssembleTest, ProducerGroupWithCoverageHoleDoesNotFuse)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_hole", "producer_group_hole", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {1, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {output};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t assembleCount = 0;
    for (auto& op : function->Operations(false)) {
        assembleCount += op.GetOpcode() == Opcode::OP_ASSEMBLE;
    }
    EXPECT_EQ(assembleCount, 3);
    EXPECT_NE(function->GetTensorMap().GetTensorByMagic(middle->GetMagic()), nullptr);
}

TEST_F(MergeViewAssembleTest, ProducerGroupWithContractDownstreamMergesToContract)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_contract", "producer_group_contract", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto finalOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {finalOutput};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {output}, {finalOutput});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    std::vector<Operation*> replacements;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            replacements.emplace_back(&op);
            EXPECT_EQ(op.GetOOperands().front(), output);
        }
        EXPECT_NE(op.GetOpcode(), Opcode::OP_ASSEMBLE);
    }
    ASSERT_EQ(replacements.size(), 2);
    EXPECT_EQ(output->GetProducers().size(), 2);
    for (auto* replacement : replacements) {
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(replacement->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        auto input = replacement->GetIOperands().front();
        EXPECT_EQ(attr->GetToOffset(),
                  input == firstInput ? std::vector<int64_t>({1, 0}) : std::vector<int64_t>({3, 0}));
    }
    EXPECT_TRUE(middle->GetProducers().empty());
    EXPECT_TRUE(middle->GetConsumers().empty());
}

TEST_F(MergeViewAssembleTest, ProducerGroupWithAssembleDownstreamKeepsAssembleOpcode)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_assemble", "producer_group_assemble", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto finalOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {finalOutput};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {output}, {finalOutput});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    std::vector<Operation*> replacements;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            replacements.emplace_back(&op);
            EXPECT_EQ(op.GetOOperands().front(), output);
        }
        EXPECT_NE(op.GetOpcode(), Opcode::OP_CONTRACT);
    }
    ASSERT_EQ(replacements.size(), 2);
    EXPECT_EQ(output->GetProducers().size(), 2);
    for (auto* replacement : replacements) {
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(replacement->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        auto input = replacement->GetIOperands().front();
        EXPECT_EQ(attr->GetToOffset(),
                  input == firstInput ? std::vector<int64_t>({1, 0}) : std::vector<int64_t>({3, 0}));
    }
    EXPECT_TRUE(middle->GetProducers().empty());
    EXPECT_TRUE(middle->GetConsumers().empty());
}

TEST_F(MergeViewAssembleTest, ProducerGroupWithContractProducersMergesToContract)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_contract_producer",
                                               "producer_group_contract_producer", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto finalOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {finalOutput};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {output}, {finalOutput});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    std::vector<Operation*> replacements;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            replacements.emplace_back(&op);
            EXPECT_EQ(op.GetOOperands().front(), output);
        }
    }
    ASSERT_EQ(replacements.size(), 2);
    EXPECT_EQ(output->GetProducers().size(), 2);
    for (auto* replacement : replacements) {
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(replacement->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        auto input = replacement->GetIOperands().front();
        EXPECT_EQ(attr->GetToOffset(),
                  input == firstInput ? std::vector<int64_t>({1, 0}) : std::vector<int64_t>({3, 0}));
    }
    EXPECT_TRUE(middle->GetProducers().empty());
    EXPECT_TRUE(middle->GetConsumers().empty());
}

TEST_F(MergeViewAssembleTest, ProducerGroupWithMixedProducersMergesPerPairOpcode)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_mixed_producer",
                                               "producer_group_mixed_producer", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto finalOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {finalOutput};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {output}, {finalOutput});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    Operation* contractReplacement = nullptr;
    Operation* assembleReplacement = nullptr;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT) {
            ASSERT_EQ(contractReplacement, nullptr);
            contractReplacement = &op;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ASSERT_EQ(assembleReplacement, nullptr);
            assembleReplacement = &op;
        }
    }
    ASSERT_NE(contractReplacement, nullptr);
    ASSERT_NE(assembleReplacement, nullptr);
    EXPECT_EQ(contractReplacement->GetIOperands().front(), firstInput);
    EXPECT_EQ(contractReplacement->GetOOperands().front(), output);
    EXPECT_EQ(assembleReplacement->GetIOperands().front(), secondInput);
    EXPECT_EQ(assembleReplacement->GetOOperands().front(), output);
    auto contractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(contractReplacement->GetOpAttribute());
    ASSERT_NE(contractAttr, nullptr);
    EXPECT_EQ(contractAttr->GetToOffset(), std::vector<int64_t>({1, 0}));
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleReplacement->GetOpAttribute());
    ASSERT_NE(assembleAttr, nullptr);
    EXPECT_EQ(assembleAttr->GetToOffset(), std::vector<int64_t>({3, 0}));
    EXPECT_EQ(output->GetProducers().size(), 2);
    EXPECT_TRUE(middle->GetProducers().empty());
    EXPECT_TRUE(middle->GetConsumers().empty());
}

TEST_F(MergeViewAssembleTest, ProducerGroupWithContractProducerAndContractDownstreamDoesNotFuse)
{
    Program program;
    auto function = std::make_unique<Function>(program, "producer_group_contract_pair", "producer_group_contract_pair",
                                               nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto finalOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {finalOutput};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {firstInput}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondInput}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{1, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {output}, {finalOutput});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t contractCount = 0;
    size_t assembleCount = 0;
    for (auto& op : function->Operations(false)) {
        contractCount += op.GetOpcode() == Opcode::OP_CONTRACT;
        assembleCount += op.GetOpcode() == Opcode::OP_ASSEMBLE;
    }
    EXPECT_EQ(contractCount, 2);
    EXPECT_EQ(assembleCount, 1);
    EXPECT_EQ(output->GetProducers().size(), 1);
    EXPECT_NE(function->GetTensorMap().GetTensorByMagic(middle->GetMagic()), nullptr);
}

TEST_F(MergeViewAssembleTest, SplitLogicalTensorVersionStopsLinearFusion)
{
    Program program;
    auto function = std::make_unique<Function>(program, "split_version", "split_version", nullptr);
    IRBuilder builder;
    auto sharedRaw = builder.CreateRawTensor(DataType::DT_FP32, Shape{4, 4});
    auto oldInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto newInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto oldVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, {4, 4});
    auto newVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, {4, 4});
    auto readOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {oldInput, newInput};
    function->outCasts_ = {readOutput, output};

    auto& oldWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {oldInput}, {oldVersion});
    oldWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& read = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {oldVersion}, {readOutput});
    auto& newWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {newInput}, {newVersion});
    newWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {newVersion}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto warToken = AddTokenEdge(*function, read, newWrite);
    function->BuildTensorMap();
    function->GetTensorMap().Insert(oldVersion, false);
    function->GetTensorMap().Insert(newVersion, false);

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t assembleCount = 0;
    for (auto& op : function->Operations(false)) {
        assembleCount += op.GetOpcode() == Opcode::OP_ASSEMBLE;
    }
    EXPECT_EQ(assembleCount, 3);
    EXPECT_NE(function->GetTensorMap().GetTensorByMagic(oldVersion->GetMagic()), nullptr);
    EXPECT_NE(function->GetTensorMap().GetTensorByMagic(newVersion->GetMagic()), nullptr);
    EXPECT_TRUE(function->GetVarDependency().HasProducer(warToken, ToStmtPtr(read)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(warToken, ToStmtPtr(newWrite)));
}

TEST_F(MergeViewAssembleTest, SplitLogicalTensorVersionWithContractProducerStopsLinearFusion)
{
    Program program;
    auto function = std::make_unique<Function>(program, "split_version_contract", "split_version_contract", nullptr);
    IRBuilder builder;
    auto sharedRaw = builder.CreateRawTensor(DataType::DT_FP32, Shape{4, 4});
    auto oldInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto newInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto oldVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, {4, 4});
    auto newVersion = builder.CreateTensorVar(*function, sharedRaw, {0, 0}, {4, 4});
    auto readOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {oldInput, newInput};
    function->outCasts_ = {readOutput, output};

    auto& oldWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_CONTRACT, {oldInput}, {oldVersion});
    oldWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& read = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {oldVersion}, {readOutput});
    auto& newWrite = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {newInput}, {newVersion});
    newWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {newVersion}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto warToken = AddTokenEdge(*function, read, newWrite);
    function->BuildTensorMap();
    function->GetTensorMap().Insert(oldVersion, false);
    function->GetTensorMap().Insert(newVersion, false);

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t assembleCount = 0;
    size_t contractCount = 0;
    for (auto& op : function->Operations(false)) {
        assembleCount += op.GetOpcode() == Opcode::OP_ASSEMBLE;
        contractCount += op.GetOpcode() == Opcode::OP_CONTRACT;
    }
    EXPECT_EQ(assembleCount, 2);
    EXPECT_EQ(contractCount, 1);
    EXPECT_NE(function->GetTensorMap().GetTensorByMagic(oldVersion->GetMagic()), nullptr);
    EXPECT_NE(function->GetTensorMap().GetTensorByMagic(newVersion->GetMagic()), nullptr);
    EXPECT_TRUE(function->GetVarDependency().HasProducer(warToken, ToStmtPtr(read)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(warToken, ToStmtPtr(newWrite)));
}

TEST_F(MergeViewAssembleTest, AssembleChainWithIntermediateViewConsumerKeepsDataEdge)
{
    NpuArchScope arch(NPUArch::DAV_3510);
    Program program;
    auto function = std::make_unique<Function>(program, "assemble_view_fanout", "assemble_view_fanout", nullptr);
    IRBuilder builder;
    auto input = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto viewOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {input};
    function->outCasts_ = {viewOutput, output};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {input}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {middle}, {viewOutput});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t assembleCount = 0;
    size_t viewCount = 0;
    Operation* remainingView = nullptr;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleCount;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewCount;
            remainingView = &op;
        }
    }
    EXPECT_EQ(assembleCount, 2);
    EXPECT_EQ(viewCount, 1);
    ASSERT_NE(remainingView, nullptr);
    ASSERT_EQ(remainingView->GetIOperands().size(), 1);
    EXPECT_EQ(remainingView->GetIOperands().front(), middle);
    EXPECT_FALSE(middle->GetProducers().empty());
    ASSERT_EQ(output->GetProducers().size(), 1);
    EXPECT_EQ((*output->GetProducers().begin())->GetIOperands().front(), input);
}

TEST_F(MergeViewAssembleTest, AssembleChainWithIntermediateTokenConsumerKeepsDataEdge)
{
    Program program;
    auto function = std::make_unique<Function>(program, "assemble_view_fanout", "assemble_view_fanout", nullptr);
    IRBuilder builder;
    auto input = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto viewOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {input};
    function->outCasts_ = {viewOutput, output};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {input}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {middle}, {viewOutput});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto token = AddTokenEdge(*function, first, view);
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t assembleCount = 0;
    size_t viewCount = 0;
    Operation* remainingView = nullptr;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleCount;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewCount;
            remainingView = &op;
        }
    }
    EXPECT_EQ(assembleCount, 2);
    EXPECT_EQ(viewCount, 1);
    ASSERT_NE(remainingView, nullptr);
    ASSERT_EQ(remainingView->GetIOperands().size(), 1);
    EXPECT_EQ(remainingView->GetIOperands().front(), middle);
    EXPECT_FALSE(middle->GetProducers().empty());
    ASSERT_EQ(output->GetProducers().size(), 1);
    EXPECT_EQ((*output->GetProducers().begin())->GetIOperands().front(), middle);
    EXPECT_TRUE(function->GetVarDependency().HasProducer(token, ToStmtPtr(first)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(token, ToStmtPtr(view)));
}

TEST_F(MergeViewAssembleTest, ShmemSetPredicateConsumerStopsViewChainFusion)
{
    Program program;
    auto function = std::make_unique<Function>(program, "view_shmem_set_pred", "view_shmem_set_pred", nullptr);
    IRBuilder builder;
    auto input = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto shmem = builder.CreateTensorVar(*function, DataType::DT_BF16, {4, 4});
    auto shmemOut = builder.CreateTensorVar(*function, DataType::DT_INT32, {1, 1});
    auto buffer = builder.CreateTensorVar(*function, DataType::DT_BF16, {16});
    function->inCasts_ = {input, shmem};
    function->outCasts_ = {output, shmemOut};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {input}, {middle});
    first.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {middle}, {output});
    second.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& shmemSet = builder.CreateTensorOpStmt(*function, Opcode::OP_SHMEM_SET, {middle, shmem}, {shmemOut, buffer});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    size_t viewCount = 0;
    size_t shmemSetCount = 0;
    for (auto& op : function->Operations(false)) {
        viewCount += op.GetOpcode() == Opcode::OP_VIEW;
        shmemSetCount += op.GetOpcode() == Opcode::OP_SHMEM_SET;
    }
    EXPECT_EQ(viewCount, 2);
    EXPECT_EQ(shmemSetCount, 1);
    ASSERT_EQ(shmemSet.GetIOperands().size(), 2);
    EXPECT_EQ(shmemSet.GetIOperands().front(), middle);
    EXPECT_EQ(shmemSet.GetIOperands().back(), shmem);
    EXPECT_FALSE(middle->GetProducers().empty());
}

TEST_F(MergeViewAssembleTest, ViewChainFusionDoesNotDependOnConstructionMode)
{
    for (bool tokenMode : {false, true}) {
        for (bool unusedMarker : {false, true}) {
            IRContext::Get().SetAssembleNewLogicalTensor(tokenMode);
            Program program;
            auto function = std::make_unique<Function>(program, "view_data_fanout", "view_data_fanout", nullptr);
            IRBuilder builder;
            auto input = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
            auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
            auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
            auto sideOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
            function->inCasts_ = {input};
            function->outCasts_ = {output, sideOutput};

            auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {input}, {middle});
            first.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
            if (unusedMarker) {
                auto marker = builder.CreateTokenVar(first.GetSpan());
                first.result_token_ = {marker};
                function->GetVarDependency().AddProducer(marker, ToStmtPtr(first));
            }
            auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {middle}, {output});
            second.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
            builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {middle}, {sideOutput});
            function->BuildTensorMap();

            MergeViewAssemble pass;
            ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

            size_t viewCount = 0;
            for (auto& op : function->Operations(false)) {
                viewCount += op.GetOpcode() == Opcode::OP_VIEW;
            }
            EXPECT_EQ(viewCount, 2);
            EXPECT_FALSE(middle->GetProducers().empty());
            ASSERT_EQ(output->GetProducers().size(), 1);
            EXPECT_EQ((*output->GetProducers().begin())->GetIOperands().front(), input);
        }
    }
}

TEST_F(MergeViewAssembleTest, ViewChainContractionPreservesExternalTokenPath)
{
    for (bool tokenMode : {false, true}) {
        IRContext::Get().SetAssembleNewLogicalTensor(tokenMode);
        Program program;
        auto function = std::make_unique<Function>(program, "view_token_path", "view_token_path", nullptr);
        IRBuilder builder;
        auto input = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto sideInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        auto sideOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
        function->inCasts_ = {input, sideInput};
        function->outCasts_ = {output, sideOutput};
        auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {input}, {middle});
        first.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {middle}, {output});
        second.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
        auto& side = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {sideInput}, {sideOutput});
        auto firstToken = AddTokenEdge(*function, first, side);
        auto secondToken = AddTokenEdge(*function, side, second);
        function->BuildTensorMap();

        MergeViewAssemble pass;
        ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
        ASSERT_EQ(output->GetProducers().size(), 1);
        EXPECT_EQ((*output->GetProducers().begin())->GetIOperands().front(), middle);
        EXPECT_TRUE(function->GetVarDependency().HasProducer(firstToken, ToStmtPtr(first)));
        EXPECT_TRUE(function->GetVarDependency().HasConsumer(firstToken, ToStmtPtr(side)));
        EXPECT_TRUE(function->GetVarDependency().HasConsumer(secondToken, ToStmtPtr(second)));
    }
}

TEST_F(MergeViewAssembleTest, TokenBoundaryRulesPreserveDependencies)
{
    for (Opcode opcode : {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE}) {
        for (int direction : {0, 1, 2}) {
            SCOPED_TRACE(static_cast<int>(opcode));
            SCOPED_TRACE(direction);
            ComputationalGraphBuilder graph;
            BuildTokenBoundaryGraph(graph, opcode);
            auto* function = graph.GetFunction();
            ASSERT_NE(function, nullptr);
            ir::VarPtr inputToken;
            if (direction != 1) {
                // A token entering the tail requires the single-exit fast path.
                inputToken = AddTokenEdge(*function, *graph.GetOp("sync"),
                                          *graph.GetOp(direction == 0 ? "second" : "first"));
            }
            if (direction != 0) {
                // A token leaving the head requires the single-entry fast path.
                AddTokenEdge(*function, *graph.GetOp("first"), *graph.GetOp("side"));
            }
            MergeViewAssemble pass;
            ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
            const auto& producers = graph.GetTensor("output")->GetProducers();
            ASSERT_EQ(producers.size(), 1);
            auto* merged = *producers.begin();
            EXPECT_EQ(merged->GetIOperands().front(), graph.GetTensor("input"));
            auto& dependency = function->GetVarDependency();
            if (inputToken != nullptr) {
                EXPECT_TRUE(dependency.HasConsumer(inputToken, ToStmtPtr(*merged)));
                EXPECT_NE(std::find(merged->tokens_.begin(), merged->tokens_.end(), inputToken), merged->tokens_.end());
            }
            if (direction != 0) {
                ASSERT_EQ(merged->result_token_.size(), 1);
                EXPECT_TRUE(dependency.HasProducer(merged->result_token_.front(), ToStmtPtr(*merged)));
                EXPECT_TRUE(dependency.HasConsumer(merged->result_token_.front(), ToStmtPtr(*graph.GetOp("side"))));
            }
        }
    }
}

TEST_F(MergeViewAssembleTest, InputTokenWithExternalDataPathStillChecksCycle)
{
    for (Opcode opcode : {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE}) {
        ComputationalGraphBuilder graph;
        ASSERT_TRUE(
            graph.AddTensors(DataType::DT_FP32, {4, 4}, {"input", "middle", "output", "side_middle", "side_output"}));
        ASSERT_TRUE(graph.AddOp(opcode, {"input"}, {"middle"}, "first", true));
        ASSERT_TRUE(graph.AddOp(opcode, {"middle"}, {"output"}, "second", true));
        ASSERT_TRUE(graph.AddOp(opcode, {"middle"}, {"side_middle"}, "side", true));
        ASSERT_TRUE(graph.AddOp(Opcode::OP_ABS, {"side_middle"}, {"side_output"}, "reader", true));
        for (const auto& name : {"first", "second", "side"}) {
            if (opcode == Opcode::OP_VIEW) {
                graph.GetOp(name)->SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
            } else {
                graph.GetOp(name)->SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
            }
        }
        ASSERT_TRUE(graph.SetInCast({"input"}));
        ASSERT_TRUE(graph.SetOutCast({"output", "side_output"}));
        auto* function = graph.GetFunction();
        auto token = AddTokenEdge(*function, *graph.GetOp("reader"), *graph.GetOp("second"));
        MergeViewAssemble pass;
        ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
        ASSERT_EQ(graph.GetTensor("output")->GetProducers().size(), 1);
        auto* tail = *graph.GetTensor("output")->GetProducers().begin();
        EXPECT_EQ(tail->GetIOperands().front(), graph.GetTensor("middle"));
        EXPECT_FALSE(graph.GetTensor("middle")->GetProducers().empty());
        EXPECT_TRUE(function->GetVarDependency().HasConsumer(token, ToStmtPtr(*tail)));
    }
}

TEST_F(MergeViewAssembleTest, IndependentLiveTokenChainsPerfGuard)
{
    constexpr int count = 1024;
    ComputationalGraphBuilder graph;
    BuildLiveTokenChains(graph, count);
    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    auto token = graph.GetOp("sync")->result_token_.front();
    MergeViewAssemble pass;
    const auto start = std::chrono::steady_clock::now();
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                                 start)
                               .count();
    size_t mergedCount = 0;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_VIEW && op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        ++mergedCount;
        EXPECT_EQ(op.GetIOperands().front(), graph.GetTensor("input"));
        EXPECT_NE(std::find(op.tokens_.begin(), op.tokens_.end(), token), op.tokens_.end());
        EXPECT_TRUE(function->GetVarDependency().HasConsumer(token, ToStmtPtr(op)));
    }
    EXPECT_EQ(mergedCount, static_cast<size_t>(count));
    std::cout << "IndependentLiveTokenChainsPerfGuard elapsed: " << elapsedMs << " ms" << std::endl;
    constexpr int64_t thresholdMs = 3000;
    EXPECT_LT(elapsedMs, thresholdMs);
}

TEST_F(MergeViewAssembleTest, BuilderLargeSharedTensorMergeableChainsPerfGuard)
{
    int assembleChainCount = 4096;
    int viewChainCount = 4096;
    int64_t thresholdMs = 1000;

    ComputationalGraphBuilder graph;
    BuildLargeSharedTensorMergeableChains(graph, assembleChainCount, viewChainCount);
    Function* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    ASSERT_EQ(function->Operations(false).size(), static_cast<size_t>(assembleChainCount * 3 + viewChainCount * 2));

    MergeViewAssemble pass;
    ASSERT_EQ(pass.PreCheck(*function), SUCCESS);
    auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    auto end = std::chrono::steady_clock::now();
    int64_t elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_EQ(pass.PostCheck(*function), SUCCESS);

    size_t viewOpCount = 0;
    size_t assembleOpCount = 0;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++viewOpCount;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ++assembleOpCount;
        }
    }
    EXPECT_EQ(viewOpCount, static_cast<size_t>(viewChainCount));
    EXPECT_EQ(assembleOpCount, static_cast<size_t>(assembleChainCount));
    std::cout << "BuilderLargeSharedTensorMergeableChainsPerfGuard elapsed: " << elapsedMs << " ms" << std::endl;
    EXPECT_LT(elapsedMs, thresholdMs);
}

static void SetSimpleViewAttr(Operation* op, const std::vector<int64_t>& offset)
{
    ASSERT_NE(op, nullptr);
    op->SetOpAttribute(
        std::make_shared<ViewOpAttribute>(offset, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{}));
}

static void SetSimpleAssembleAttr(Operation* op, const std::vector<int64_t>& offset)
{
    ASSERT_NE(op, nullptr);
    op->SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset, std::vector<SymbolicScalar>{}));
}

TEST_F(MergeViewAssembleTest, ViewSliceChainShouldMergeToSlice)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_VIEW, {"input"}, {"mid"}, "view", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_SLICE, {"mid"}, {"output"}, "slice", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleViewAttr(G.GetOp("view"), {1, 2});
    SetSimpleViewAttr(G.GetOp("slice"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int sliceCount = 0;
    Operation* mergedSlice = nullptr;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.IsDeleted()) {
            sliceCount++;
            mergedSlice = &op;
        }
        EXPECT_NE(op.GetOpcode(), Opcode::OP_VIEW);
    }
    ASSERT_EQ(sliceCount, 1);
    ASSERT_NE(mergedSlice, nullptr);
    auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(mergedSlice->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->GetFromOffset(), (std::vector<int64_t>{4, 6}));
}

TEST_F(MergeViewAssembleTest, SliceSliceChainShouldNotMerge)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_SLICE, {"input"}, {"mid"}, "slice1", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_SLICE, {"mid"}, {"output"}, "slice2", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleViewAttr(G.GetOp("slice1"), {1, 2});
    SetSimpleViewAttr(G.GetOp("slice2"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int sliceCount = 0;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.IsDeleted()) {
            sliceCount++;
        }
    }
    EXPECT_EQ(sliceCount, 2);
}

TEST_F(MergeViewAssembleTest, AssembleContractChainShouldMergeToContract)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ASSEMBLE, {"input"}, {"mid"}, "assemble", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_CONTRACT, {"mid"}, {"output"}, "contract", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleAssembleAttr(G.GetOp("assemble"), {1, 2});
    SetSimpleAssembleAttr(G.GetOp("contract"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int contractCount = 0;
    Operation* mergedContract = nullptr;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT && !op.IsDeleted()) {
            contractCount++;
            mergedContract = &op;
        }
        EXPECT_NE(op.GetOpcode(), Opcode::OP_ASSEMBLE);
    }
    ASSERT_EQ(contractCount, 1);
    ASSERT_NE(mergedContract, nullptr);
    auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(mergedContract->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->GetToOffset(), (std::vector<int64_t>{4, 6}));
}

TEST_F(MergeViewAssembleTest, ContractContractChainShouldNotMerge)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_CONTRACT, {"input"}, {"mid"}, "contract1", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_CONTRACT, {"mid"}, {"output"}, "contract2", true));
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    SetSimpleAssembleAttr(G.GetOp("contract1"), {1, 2});
    SetSimpleAssembleAttr(G.GetOp("contract2"), {3, 4});

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*G.GetFunction()), SUCCESS);

    int contractCount = 0;
    for (auto& op : G.GetFunction()->Operations()) {
        if (op.GetOpcode() == Opcode::OP_CONTRACT && !op.IsDeleted()) {
            contractCount++;
        }
    }
    EXPECT_EQ(contractCount, 2);
}

/*
 * 扇出场景的 WAR token 接管：
 *   shared --SLICE(head, result_token=warToken)--> mid --VIEW_i--> out_i (outcast), i = 0..2
 *   contract_in --CONTRACT--> contract_out (outcast), contract 等待 warToken
 * slice_head 与每个 view_i 各合并成一条链，产出 3 个 merged SLICE。每个 merged SLICE
 * 必须产出自己的新 token 且都被 contract 等待（token 单生产者约束），旧 warToken 被清理。
 */
TEST_F(MergeViewAssembleTest, FanoutViewChainResultTokenPropagatesToEveryMergedSlice)
{
    ComputationalGraphBuilder G;
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {10, 10},
                             {"shared", "mid", "out_0", "out_1", "out_2", "contract_in", "contract_out"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_SLICE, {"shared"}, {"mid"}, "slice_head", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_VIEW, {"mid"}, {"out_0"}, "view_0", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_VIEW, {"mid"}, {"out_1"}, "view_1", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_VIEW, {"mid"}, {"out_2"}, "view_2", true));
    EXPECT_TRUE(G.AddOp(Opcode::OP_CONTRACT, {"contract_in"}, {"contract_out"}, "contract", true));
    EXPECT_TRUE(G.SetInCast({"shared", "contract_in"}));
    EXPECT_TRUE(G.SetOutCast({"out_0", "out_1", "out_2", "contract_out"}));
    SetSimpleViewAttr(G.GetOp("slice_head"), {0, 0});
    SetSimpleViewAttr(G.GetOp("view_0"), {0, 0});
    SetSimpleViewAttr(G.GetOp("view_1"), {0, 0});
    SetSimpleViewAttr(G.GetOp("view_2"), {0, 0});
    SetSimpleAssembleAttr(G.GetOp("contract"), {0, 0});

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    Operation* sliceHead = G.GetOp("slice_head");
    Operation* contract = G.GetOp("contract");
    ASSERT_NE(sliceHead, nullptr);
    ASSERT_NE(contract, nullptr);
    auto warToken = AddTokenEdge(*function, *sliceHead, *contract);

    MergeViewAssemble mergePass;
    ASSERT_EQ(mergePass.RunOnFunction(*function), SUCCESS);

    // 扇出的 3 条 [slice_head, view_i] 链合并成 3 个 SLICE
    std::vector<Operation*> mergedSlices;
    std::vector<ir::VarPtr> newTokens;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_SLICE && !op.IsDeleted()) {
            mergedSlices.emplace_back(&op);
            for (const auto& token : op.result_token_) {
                newTokens.emplace_back(token);
            }
        }
    }
    ASSERT_EQ(mergedSlices.size(), 3);
    ASSERT_EQ(newTokens.size(), 3);

    // 每个 merged slice 恰好生产一个新 token（单生产者），且全部被 contract 等待
    auto& dependency = function->GetVarDependency();
    for (auto* mergedSlice : mergedSlices) {
        ASSERT_EQ(mergedSlice->result_token_.size(), 1);
        EXPECT_EQ(dependency.GetProducers(mergedSlice->result_token_.front()).size(), 1);
        EXPECT_TRUE(dependency.HasProducer(mergedSlice->result_token_.front(), ToStmtPtr(*mergedSlice)));
        EXPECT_NE(std::find(contract->tokens_.begin(), contract->tokens_.end(), mergedSlice->result_token_.front()),
                  contract->tokens_.end());
        EXPECT_TRUE(dependency.HasConsumer(mergedSlice->result_token_.front(), ToStmtPtr(*contract)));
    }

    // 旧 token 已被全部新 token 接管，不得残留
    EXPECT_EQ(std::find(contract->tokens_.begin(), contract->tokens_.end(), warToken), contract->tokens_.end());
    EXPECT_FALSE(dependency.HasDependency(warToken));
}

TEST_F(MergeViewAssembleTest, MixedChainsKeepLegacyFusionAndMergedTokenAssemble)
{
    Program program;
    auto function = std::make_unique<Function>(program, "mixed_chains", "mixed_chains", nullptr);
    IRBuilder builder;
    auto assembleInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto assembleMiddle = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 8});
    auto assembleOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 16});
    auto source = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto tokenOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto viewInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto viewMiddle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto viewOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto sideOutput = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {assembleInput, source};
    function->outCasts_ = {assembleOutput, viewOutput, sideOutput};

    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {assembleInput}, {assembleMiddle});
    SetSimpleAssembleAttr(&first, {0, 0});
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {assembleMiddle}, {assembleOutput});
    SetSimpleAssembleAttr(&second, {0, 4});
    auto& tokenConsumer = builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {source}, {tokenOutput});
    AddTokenEdge(*function, first, tokenConsumer);
    // The ordinary ABS separates the token region from the legacy view chain.
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {tokenOutput}, {viewInput});
    auto& viewFirst = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {viewInput}, {viewMiddle});
    SetSimpleViewAttr(&viewFirst, {0, 0});
    auto& viewSecond = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {viewMiddle}, {viewOutput});
    SetSimpleViewAttr(&viewSecond, {0, 0});
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {viewMiddle}, {sideOutput});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    ASSERT_EQ(viewOutput->GetProducers().size(), 1);
    EXPECT_EQ((*viewOutput->GetProducers().begin())->GetIOperands().front(), viewInput);
    EXPECT_FALSE(viewMiddle->GetProducers().empty());
    ASSERT_EQ(assembleOutput->GetProducers().size(), 1);
    auto* merged = *assembleOutput->GetProducers().begin();
    EXPECT_EQ(merged->GetIOperands().front(), assembleInput);
    ASSERT_EQ(tokenConsumer.tokens_.size(), 1);
    const auto& token = tokenConsumer.tokens_.front();
    EXPECT_TRUE(function->GetVarDependency().HasProducer(token, ToStmtPtr(*merged)));
    EXPECT_TRUE(function->GetVarDependency().HasConsumer(token, ToStmtPtr(tokenConsumer)));
}

// Legacy mode (create_new_logical_tensor off): basic assemble chain merge still works.
TEST_F(MergeViewAssembleTest, LegacyModeMergesLinearAssembleChain)
{
    IRContext::Get().SetAssembleNewLogicalTensor(false);
    Program program;
    auto function = std::make_unique<Function>(program, "legacy_linear", "legacy_linear", nullptr);
    IRBuilder builder;
    auto input = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto viewOut = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto midTensor = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 8});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 16});
    function->inCasts_ = {input};
    function->outCasts_ = {output};
    // The producer view supplies the assemble input.
    auto& view = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {input}, {viewOut});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {viewOut}, {midTensor});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {midTensor}, {output});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 4}));
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    // Legal mode deletes only the chain tail; the head dies later via dead-code
    // elimination, leaving exactly one live assemble (the merged op).
    Operation* merged = nullptr;
    for (auto& op : function->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && !op.IsDeleted()) {
            ASSERT_EQ(merged, nullptr);
            merged = &op;
        }
    }
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->GetIOperands().front(), viewOut);
    EXPECT_EQ(merged->GetOOperands().front(), output);
    auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(merged->GetOpAttribute());
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->GetToOffset(), (std::vector<int64_t>{0, 4}));
}

// Legal mode deletes only the chain tail: intermediates survive for out-of-chain
// readers, so the ABS reader of middle keeps a live producer.
TEST_F(MergeViewAssembleTest, LegacyModeKeepsIntermediateForSiblingConsumer)
{
    IRContext::Get().SetAssembleNewLogicalTensor(false);
    Program program;
    auto function = std::make_unique<Function>(program, "legacy_sibling", "legacy_sibling", nullptr);
    IRBuilder builder;
    auto firstInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondInput = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto firstView = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto secondView = builder.CreateTensorVar(*function, DataType::DT_FP32, {2, 4});
    auto middle = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    auto otherOut = builder.CreateTensorVar(*function, DataType::DT_FP32, {4, 4});
    function->inCasts_ = {firstInput, secondInput};
    function->outCasts_ = {output, otherOut};
    // Producer views supply the assemble inputs.
    auto& view1 = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {firstInput}, {firstView});
    view1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& view2 = builder.CreateTensorOpStmt(*function, Opcode::OP_VIEW, {secondInput}, {secondView});
    view2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& first = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {firstView}, {middle});
    first.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto& second = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {secondView}, {middle});
    second.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{2, 0}));
    // Sibling consumer of middle: an ABS op. Legal mode merges the chain through the
    // middle but deletes only the tail, so the sibling keeps live producers.
    auto& downstream = builder.CreateTensorOpStmt(*function, Opcode::OP_ASSEMBLE, {middle}, {output});
    downstream.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    builder.CreateTensorOpStmt(*function, Opcode::OP_ABS, {middle}, {otherOut});
    function->BuildTensorMap();

    MergeViewAssemble pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    EXPECT_FALSE(first.IsDeleted()) << "legal mode must keep the intermediate producer alive";
    EXPECT_EQ(function->GetTensorMap().GetTensorByMagic(middle->GetMagic()) != nullptr, true);
}
} // namespace tile_fwk
} // namespace npu
