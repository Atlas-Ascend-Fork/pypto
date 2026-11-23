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
 * \file test_build_tree_from_reduce_pass.cpp
 * \brief Unit tests for BuildTreeFromReduce.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "computational_graph_builder.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/program/program.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/tile_graph_pass/graph_optimization/build_tree_from_reduce.h"
#include "passes/tile_graph_pass/graph_optimization/process_atomic.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {
namespace {

const Shape DEFAULT_TEST_SHAPE{16, 16};
const Shape WIDE_OUTPUT_SHAPE{32, 16};
const std::vector<int64_t> ZERO_OFFSET{0, 0};
const std::vector<int64_t> OFFSET_ROW_16{16, 0};
constexpr int64_t HALF_UB_FACTOR = 2;
constexpr int64_t MIN_LARGE_ROWS = 3;
constexpr int64_t SPLIT_FACTOR = 2;
constexpr int SCOPE_ID_A = 1;
constexpr int SCOPE_ID_B = 2;
constexpr size_t FINAL_ASSEMBLE_COUNT = 1;
constexpr size_t VIEW_OPS_PER_SPLIT_INPUT = 2;

size_t CountOpcode(Function& function, Opcode opcode)
{
    size_t count = 0;
    for (const auto& op : function.Operations()) {
        count += op.GetOpcode() == opcode ? 1 : 0;
    }
    return count;
}

Shape LargeShapeExceedingHalfUb(bool oddFirstDim = false)
{
    constexpr int64_t innerDim = 16;
    const auto ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    const auto dataBytes = BytesOf(DataType::DT_FP16);
    EXPECT_GT(ubSize, 0);
    int64_t rows = static_cast<int64_t>(ubSize / (HALF_UB_FACTOR * dataBytes * innerDim) + HALF_UB_FACTOR);
    rows = std::max<int64_t>(rows, MIN_LARGE_ROWS);
    if (oddFirstDim) {
        rows += rows % 2 == 0 ? 1 : 0;
    } else {
        rows += rows % 2 == 0 ? 0 : 1;
    }
    return {rows, innerDim};
}

std::vector<Operation*> GetOutputAssembles(Function& function, const LogicalTensorPtr& output)
{
    std::vector<Operation*> assembles;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE || op.GetOOperands().size() != 1 ||
            op.GetOutputOperand(0) != output) {
            continue;
        }
        assembles.push_back(&op);
    }
    std::sort(assembles.begin(), assembles.end(), [](const auto* lhs, const auto* rhs) {
        auto lhsAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(lhs->GetOpAttribute());
        auto rhsAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(rhs->GetOpAttribute());
        return lhsAttr->GetToOffset()[0] < rhsAttr->GetToOffset()[0];
    });
    return assembles;
}

std::vector<LogicalTensorPtr> CollectCopyInOutputs(Function& function)
{
    std::vector<LogicalTensorPtr> outputs;
    for (const auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            outputs.emplace_back(op.GetOutputOperand(0));
        }
    }
    return outputs;
}

void ExpectAddOpsInUb(Function& function)
{
    for (const auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ADD) {
            continue;
        }
        EXPECT_EQ(op.GetInputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        EXPECT_EQ(op.GetInputOperand(1)->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        EXPECT_EQ(op.GetOutputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    }
}

class BuildTreeFromReduceTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        ConfigManagerNg::SetGlobalConfig(COMPUTE_DETERMINISM_LEVEL, static_cast<int64_t>(1));
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
        Platform::Instance().ReloadMemoryPaths("3510");
    }

    void TearDown() override
    {
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
        Platform::Instance().ReloadMemoryPaths("");
    }

    static void AddAtomicAssemble(ComputationalGraphBuilder& graph, const std::string& input, const std::string& output,
                                  const std::string& name, const std::vector<int64_t>& offset = ZERO_OFFSET,
                                  bool fromReduceAcc = true, bool fromExplicitRmw = false,
                                  Opcode opcode = Opcode::OP_ASSEMBLE)
    {
        graph.AddOp(opcode, {input}, {output}, name);
        auto* assemble = graph.GetOp(name);
        auto inputTensor = graph.GetTensor(input);
        assemble->SetOpAttribute(std::make_shared<AssembleOpAttribute>(inputTensor->GetMemoryTypeOriginal(), offset,
                                                                       SymbolicScalar::FromConcrete(offset),
                                                                       inputTensor->GetDynValidShape()));
        assemble->SetAttribute(RMW_MODE_ATTR_ADD, 1L);
        if (fromReduceAcc) {
            assemble->SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);
        }
        if (fromExplicitRmw) {
            assemble->SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
        }
    }

    static void BuildInputs(ComputationalGraphBuilder& graph, size_t inputCount,
                            MemoryType inputMemory = MemoryType::MEM_UB, Shape inputShape = DEFAULT_TEST_SHAPE)
    {
        std::vector<std::string> sources;
        for (size_t i = 0; i < inputCount; ++i) {
            auto suffix = std::to_string(i);
            auto source = "source" + suffix;
            auto input = "input" + suffix;
            graph.AddTensor(DataType::DT_FP16, inputShape, source);
            graph.GetTensor(source)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            graph.AddTensor(DataType::DT_FP16, inputShape, input);
            graph.AddOp(Opcode::OP_COPY_IN, {source}, {input}, "producer" + suffix);
            graph.GetTensor(input)->SetMemoryTypeBoth(inputMemory, true);
            sources.emplace_back(source);
        }
        graph.SetInCast(sources);
    }
};

TEST_F(BuildTreeFromReduceTest, BuildsBalancedTreeInInputProducerTopologyOrderAndInfersMemory)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 4;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // Deliberately shuffle the assemble insertion order. Pairing must follow the input producers' topo order.
    AddAtomicAssemble(graph, "input2", "output", "assemble2");
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    AddAtomicAssemble(graph, "input3", "output", "assemble3");
    AddAtomicAssemble(graph, "input1", "output", "assemble1");
    graph.SetOutCast({"output"});

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SortOperations(SortOperationsMode::LIGHTWEIGHT);
    auto inputsInProducerOrder = CollectCopyInOutputs(*function);
    ASSERT_EQ(inputsInProducerOrder.size(), inputCount);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), FINAL_ASSEMBLE_COUNT);

    constexpr size_t firstLevelPairs = inputCount / 2;
    std::array<bool, firstLevelPairs> expectedFirstLevel = {};
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ADD) {
            continue;
        }
        const auto& inputs = op.GetIOperands();
        for (size_t pair = 0; pair < firstLevelPairs; ++pair) {
            if (inputs[0] == inputsInProducerOrder[pair * 2] && inputs[1] == inputsInProducerOrder[pair * 2 + 1]) {
                expectedFirstLevel[pair] = true;
            }
        }
    }
    for (size_t pair = 0; pair < firstLevelPairs; ++pair) {
        EXPECT_TRUE(expectedFirstLevel[pair]);
    }

    ExpectAddOpsInUb(*function);

    auto* finalAssemble = *graph.GetTensor("output")->GetProducers().begin();
    ASSERT_NE(finalAssemble, nullptr);
    auto finalAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(finalAssemble->GetOpAttribute());
    ASSERT_NE(finalAttr, nullptr);
    EXPECT_EQ(finalAttr->GetFrom(), MemoryType::MEM_UB);
}

TEST_F(BuildTreeFromReduceTest, SkipsWhenDeterminismIsDisabled)
{
    ConfigManagerNg::SetGlobalConfig(COMPUTE_DETERMINISM_LEVEL, static_cast<int64_t>(0));

    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    AddAtomicAssemble(graph, "input1", "output", "assemble1");
    graph.SetOutCast({"output"});

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), 0U);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), inputCount);
}

TEST_F(BuildTreeFromReduceTest, CarriesOddInputToNextLevel)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 5;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    for (size_t i = 0; i < inputCount; ++i) {
        AddAtomicAssemble(graph, "input" + std::to_string(i), "output", "assemble" + std::to_string(i));
    }

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), FINAL_ASSEMBLE_COUNT);

    auto* finalAssemble = *graph.GetTensor("output")->GetProducers().begin();
    ASSERT_NE(finalAssemble, nullptr);
    ASSERT_EQ(finalAssemble->GetIOperands().size(), 1);
    ASSERT_EQ(finalAssemble->GetInputOperand(0)->GetProducers().size(), 1);
    EXPECT_EQ((*finalAssemble->GetInputOperand(0)->GetProducers().begin())->GetOpcode(), Opcode::OP_ADD);
}

TEST_F(BuildTreeFromReduceTest, DoesNotMergeDifferentDestinationRegions)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, WIDE_OUTPUT_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0", ZERO_OFFSET);
    AddAtomicAssemble(graph, "input1", "output", "assemble1", OFFSET_ROW_16);

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), 0);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), inputCount);
}

TEST_F(BuildTreeFromReduceTest, DoesNotBuildTreeAcrossExecutionScopes)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    AddAtomicAssemble(graph, "input1", "output", "assemble1");
    graph.GetOp("assemble0")->SetScopeId(SCOPE_ID_A);
    graph.GetOp("assemble1")->SetScopeId(SCOPE_ID_B);

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), 0);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), inputCount);
}

TEST_F(BuildTreeFromReduceTest, BuildsTreeForExplicitAtomicRmwOnlyAndKeepsAtomicAdd)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0", ZERO_OFFSET, false, true);
    AddAtomicAssemble(graph, "input1", "output", "assemble1", ZERO_OFFSET, false, true);

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), FINAL_ASSEMBLE_COUNT);

    auto* finalAssemble = *graph.GetTensor("output")->GetProducers().begin();
    ASSERT_NE(finalAssemble, nullptr);
    EXPECT_TRUE(finalAssemble->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_FALSE(finalAssemble->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR));
    EXPECT_FALSE(finalAssemble->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR));
}

TEST_F(BuildTreeFromReduceTest, KeepsAtomicAddForReduceAccFeedingExplicitAtomicRmw)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0", ZERO_OFFSET, true, true);
    AddAtomicAssemble(graph, "input1", "output", "assemble1", ZERO_OFFSET, true, true);

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), FINAL_ASSEMBLE_COUNT);

    auto* finalAssemble = *graph.GetTensor("output")->GetProducers().begin();
    ASSERT_NE(finalAssemble, nullptr);
    EXPECT_TRUE(finalAssemble->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_FALSE(finalAssemble->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR));
    EXPECT_FALSE(finalAssemble->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR));
}

TEST_F(BuildTreeFromReduceTest, IgnoresTensorExplicitRmwProvenance)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    graph.GetTensor("output")->SetAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    AddAtomicAssemble(graph, "input1", "output", "assemble1");

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), FINAL_ASSEMBLE_COUNT);

    auto* finalAssemble = *graph.GetTensor("output")->GetProducers().begin();
    ASSERT_NE(finalAssemble, nullptr);
    EXPECT_FALSE(finalAssemble->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(BuildTreeFromReduceTest, SplitsLargeInputOnAxisZeroWhenOverHalfUb)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    auto inputShape = LargeShapeExceedingHalfUb(false);
    BuildInputs(graph, inputCount, MemoryType::MEM_UB, inputShape);
    graph.AddTensor(DataType::DT_FP16, inputShape, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    AddAtomicAssemble(graph, "input1", "output", "assemble1");

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), inputCount);
    EXPECT_EQ(CountOpcode(*function, config::GetSliceOpcode()), inputCount * VIEW_OPS_PER_SPLIT_INPUT);

    auto outputAssembles = GetOutputAssembles(*function, graph.GetTensor("output"));
    ASSERT_EQ(outputAssembles.size(), inputCount);
    auto leftAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(outputAssembles[0]->GetOpAttribute());
    auto rightAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(outputAssembles[1]->GetOpAttribute());
    ASSERT_NE(leftAttr, nullptr);
    ASSERT_NE(rightAttr, nullptr);
    EXPECT_EQ(leftAttr->GetToOffset(), ZERO_OFFSET);
    EXPECT_EQ(rightAttr->GetToOffset(), (std::vector<int64_t>{inputShape[0] / SPLIT_FACTOR, 0}));
    EXPECT_TRUE(leftAttr->GetToDynOffset().empty());
    EXPECT_TRUE(rightAttr->GetToDynOffset().empty());
    EXPECT_EQ(outputAssembles[0]->GetInputOperand(0)->GetShape(), (Shape{inputShape[0] / SPLIT_FACTOR, inputShape[1]}));
    EXPECT_EQ(outputAssembles[1]->GetInputOperand(0)->GetShape(),
              (Shape{inputShape[0] - inputShape[0] / SPLIT_FACTOR, inputShape[1]}));
    for (auto& op : function->Operations()) {
        if (op.GetOpcode() != config::GetSliceOpcode()) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        ASSERT_NE(viewAttr, nullptr);
        EXPECT_TRUE(viewAttr->GetFromDynOffset().empty());
    }
    EXPECT_FALSE(outputAssembles[0]->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_FALSE(outputAssembles[1]->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(BuildTreeFromReduceTest, SplitsLargeInputAndKeepsAtomicAddForExplicitRmw)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    auto inputShape = LargeShapeExceedingHalfUb(true);
    BuildInputs(graph, inputCount, MemoryType::MEM_UB, inputShape);
    graph.AddTensor(DataType::DT_FP16, inputShape, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0", ZERO_OFFSET, false, true);
    AddAtomicAssemble(graph, "input1", "output", "assemble1", ZERO_OFFSET, false, true);

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), inputCount);
    EXPECT_EQ(CountOpcode(*function, config::GetSliceOpcode()), inputCount * VIEW_OPS_PER_SPLIT_INPUT);

    auto outputAssembles = GetOutputAssembles(*function, graph.GetTensor("output"));
    ASSERT_EQ(outputAssembles.size(), inputCount);
    auto leftAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(outputAssembles[0]->GetOpAttribute());
    auto rightAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(outputAssembles[1]->GetOpAttribute());
    ASSERT_NE(leftAttr, nullptr);
    ASSERT_NE(rightAttr, nullptr);
    EXPECT_EQ(leftAttr->GetToOffset(), ZERO_OFFSET);
    EXPECT_EQ(rightAttr->GetToOffset(), (std::vector<int64_t>{inputShape[0] / SPLIT_FACTOR, 0}));
    EXPECT_TRUE(leftAttr->GetToDynOffset().empty());
    EXPECT_TRUE(rightAttr->GetToDynOffset().empty());
    EXPECT_TRUE(outputAssembles[0]->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_TRUE(outputAssembles[1]->HasAttr(RMW_MODE_ATTR_ADD));
    EXPECT_FALSE(outputAssembles[0]->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR));
    EXPECT_FALSE(outputAssembles[0]->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR));
    EXPECT_FALSE(outputAssembles[1]->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR));
    EXPECT_FALSE(outputAssembles[1]->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR));
}

TEST_F(BuildTreeFromReduceTest, UsesConfiguredSliceOpcodeForLargeInputSplit)
{
    ASSERT_EQ(config::GetSliceOpcode(), Opcode::OP_SLICE);

    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    auto inputShape = LargeShapeExceedingHalfUb(false);
    BuildInputs(graph, inputCount, MemoryType::MEM_UB, inputShape);
    graph.AddTensor(DataType::DT_FP16, inputShape, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    AddAtomicAssemble(graph, "input1", "output", "assemble1");

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOpcode(*function, config::GetSliceOpcode()), inputCount * VIEW_OPS_PER_SPLIT_INPUT);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_VIEW), 0);
}

TEST_F(BuildTreeFromReduceTest, BuildsTreeForConfiguredContractOpcode)
{
    ASSERT_EQ(config::GetContractOpcode(), Opcode::OP_CONTRACT);

    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "contract0", ZERO_OFFSET, false, true, config::GetContractOpcode());
    AddAtomicAssemble(graph, "input1", "output", "contract1", ZERO_OFFSET, false, true, config::GetContractOpcode());

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_CONTRACT), FINAL_ASSEMBLE_COUNT);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), 0);
    auto* finalContract = *graph.GetTensor("output")->GetProducers().begin();
    ASSERT_NE(finalContract, nullptr);
    EXPECT_TRUE(finalContract->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(BuildTreeFromReduceTest, LeavesNonAtomicProducerOutsideAtomicTree)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 3;
    constexpr size_t nonAtomicCount = 1;
    const size_t atomicCount = inputCount - nonAtomicCount;
    BuildInputs(graph, inputCount);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    graph.AddOp(Opcode::OP_ASSEMBLE, {"input1"}, {"output"}, "plainAssemble");
    graph.GetOp("plainAssemble")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(ZERO_OFFSET));
    AddAtomicAssemble(graph, "input2", "output", "assemble2");

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), atomicCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), nonAtomicCount + FINAL_ASSEMBLE_COUNT);
    EXPECT_FALSE(graph.GetOp("plainAssemble")->HasAttr(RMW_MODE_ATTR_ADD));
}

TEST_F(BuildTreeFromReduceTest, IntegratesWithProcessAtomicAfterMemoryAssignment)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 4;
    BuildInputs(graph, inputCount);
    std::vector<std::string> partials;
    for (size_t i = 0; i < inputCount; ++i) {
        auto suffix = std::to_string(i);
        auto partial = "partial" + suffix;
        graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, partial);
        graph.GetTensor(partial)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        graph.AddOp(Opcode::OP_ASSEMBLE, {"input" + suffix}, {partial}, "assemble" + suffix);
        graph.GetOp("assemble" + suffix)
            ->SetOpAttribute(std::make_shared<AssembleOpAttribute>(
                graph.GetTensor("input" + suffix)->GetMemoryTypeOriginal(), ZERO_OFFSET,
                SymbolicScalar::FromConcrete(ZERO_OFFSET), graph.GetTensor("input" + suffix)->GetDynValidShape()));
        partials.emplace_back(partial);
    }
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    graph.AddOp(Opcode::OP_REDUCE_ACC, partials, {"output"}, "reduceAcc");
    graph.SetOutCast({"output"});

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    ProcessAtomic processAtomic;
    ASSERT_EQ(processAtomic.Run(*function, "", "", 0), SUCCESS);
    BuildTreeFromReduce buildTree;
    ASSERT_EQ(buildTree.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_REDUCE_ACC), 0);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ASSEMBLE), FINAL_ASSEMBLE_COUNT);
    auto* finalAssemble = *graph.GetTensor("output")->GetProducers().begin();
    ASSERT_NE(finalAssemble, nullptr);
    EXPECT_FALSE(finalAssemble->HasAttr(RMW_MODE_ATTR_ADD));

    ExpectAddOpsInUb(*function);
}

TEST_F(BuildTreeFromReduceTest, RepairsL0CLeavesToUbForGeneratedAdds)
{
    ComputationalGraphBuilder graph;
    constexpr size_t inputCount = 2;
    BuildInputs(graph, inputCount, MemoryType::MEM_L0C);
    graph.AddTensor(DataType::DT_FP16, DEFAULT_TEST_SHAPE, "output");
    graph.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    AddAtomicAssemble(graph, "input0", "output", "assemble0");
    AddAtomicAssemble(graph, "input1", "output", "assemble1");
    graph.SetOutCast({"output"});

    auto* function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    BuildTreeFromReduce pass;
    ASSERT_EQ(pass.Run(*function, "", "", 0), SUCCESS);

    EXPECT_EQ(CountOpcode(*function, Opcode::OP_ADD), inputCount - 1);
    ExpectAddOpsInUb(*function);
    for (const auto& op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_ADD) {
            continue;
        }
        for (const auto& input : op.GetIOperands()) {
            ASSERT_EQ(input->GetProducers().size(), 1);
            auto* producer = *input->GetProducers().begin();
            ASSERT_NE(producer, nullptr);
            if (producer->GetOpcode() == Opcode::OP_CONVERT) {
                auto attr = std::dynamic_pointer_cast<ConvertOpAttribute>(producer->GetOpAttribute());
                ASSERT_NE(attr, nullptr);
                auto [from, to] = attr->GetConvertPath();
                EXPECT_EQ(from, MemoryType::MEM_L0C);
                EXPECT_EQ(to, MemoryType::MEM_UB);
                ASSERT_EQ(producer->GetIOperands().size(), 1);
                EXPECT_EQ(producer->GetInputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_L0C);
            } else {
                // With a single UB consumer, ConvertInserter may retarget the producer output
                // directly instead of materializing a standalone Convert operation.
                ASSERT_EQ(producer->GetOOperands().size(), 1);
                EXPECT_EQ(producer->GetOutputOperand(0), input);
                EXPECT_EQ(producer->GetOutputOperand(0)->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
            }
        }
    }
}

} // namespace
} // namespace npu::tile_fwk
