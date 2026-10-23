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
 * \file test_subgraph_to_function.cpp
 * \brief Unit test for SubgraphToFunction pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "tilefwk/data_type.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/tile_graph_pass/subgraph_to_function.h"
#include "passes/tile_graph_pass/graph_partition/graph_partition_token_dependency.h"
#include "passes/tile_graph_pass/static_subgraph_processor.h"
#include "passes/pass_mgr/pass_manager.h"
#include "passes/statistics/execute_graph_statistic.h"
#include "ut_json/ut_json_tool.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {

class SubgraphToFunctionTest : public testing::Test {
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

bool ArePsgHashesUnique(const Function& function)
{
    std::unordered_set<size_t> hashSet;
    for (const auto& [psgId, program] : function.programs_) {
        (void)psgId;
        size_t hashValue = program->ComputeHash().GetHash();
        if (hashSet.find(hashValue) != hashSet.end()) {
            return false;
        }
        hashSet.insert(hashValue);
    }
    return true;
}

bool IsPSgToESgMapOneToOne(const std::multimap<int, int>& PSgToESgMap)
{
    std::unordered_map<int, int> esgToPsgMap;
    for (const auto& [psgId, esgId] : PSgToESgMap) {
        if (esgToPsgMap.find(esgId) != esgToPsgMap.end()) {
            return false;
        }
        esgToPsgMap[esgId] = psgId;
    }
    return true;
}

std::multimap<int, int> GetPSgToESgMap(Function* rootFunc)
{
    std::multimap<int, int> PSgToESgMap;

    for (size_t i = 0; i < rootFunc->Operations().size(); i++) {
        auto iter = rootFunc->Operations()[i].GetSubFuncInvokeInfo();
        int PSgId = iter.GetProgramId();
        int ESgId = rootFunc->Operations()[i].GetSubgraphID();
        PSgToESgMap.insert({PSgId, ESgId});
    }
    return PSgToESgMap;
}

static std::shared_ptr<LogicalTensor> BuildDifferentOffsetSubgraph0(const std::shared_ptr<Function>& func)
{
    auto shape3Imme = OpImmediate::Specified({16, 8, 8});
    auto shape1Imme = OpImmediate::Specified({16, 64});

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{32, 8, 8},
                                                             CreateTestConstIntVector(std::vector<int64_t>{32, 8, 8}));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    incast->SetMagic(3);
    func->inCasts_.push_back(incast);

    auto tensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{16, 8, 8},
                                                              CreateTestConstIntVector(std::vector<int64_t>{16, 8, 8}));
    tensor0->SetMemoryTypeBoth(MEM_UB);
    tensor0->SetMagic(15);

    auto& copyopin0 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {incast}, {tensor0});
    copyopin0.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({16, 0, 0}), MEM_UB, shape3Imme,
                                                               shape3Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin0.UpdateSubgraphID(0);
    copyopin0.opmagic = 10032;

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, std::vector<int64_t>{16, 64},
                                                              CreateTestConstIntVector(std::vector<int64_t>{16, 64}));
    tensor1->SetMemoryTypeBoth(MEM_UB);
    tensor1->SetMagic(66);

    auto& reshapeop = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {tensor0}, {tensor1});
    reshapeop.UpdateSubgraphID(0);
    reshapeop.opmagic = 10039;

    auto input_tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 64}, CreateTestConstIntVector(std::vector<int64_t>{16, 64}));
    input_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor->SetMagic(79);

    auto& copyoutop0 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {tensor1}, {input_tensor});
    copyoutop0.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape1Imme,
                                                                shape1Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop0.UpdateSubgraphID(0);
    copyoutop0.opmagic = 10043;

    return input_tensor;
}

static void BuildDifferentOffsetSubgraph1(const std::shared_ptr<Function>& func,
                                          const std::shared_ptr<LogicalTensor>& input_tensor,
                                          const std::shared_ptr<LogicalTensor>& output_tensor)
{
    auto shape2Imme = OpImmediate::Specified({16, 32});

    auto inner_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    inner_tensor1->SetMemoryTypeBoth(MEM_UB);
    inner_tensor1->UpdateOffset({0, 0});
    inner_tensor1->SetMagic(30);

    auto& copyopin1 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor1});
    copyopin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme,
                                                               shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin1.UpdateSubgraphID(1);
    copyopin1.opmagic = 10021;

    auto result_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    result_tensor1->SetMemoryTypeBoth(MEM_UB);
    result_tensor1->SetMagic(29);

    auto& expopin1 = PassOperationUtils::AddOperation(*func, Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(1);
    expopin1.opmagic = 10023;

    auto& copyoutop1 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {result_tensor1}, {output_tensor});
    copyoutop1.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape2Imme,
                                                                shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop1.UpdateSubgraphID(1);
    copyoutop1.opmagic = 10029;
}

static void BuildDifferentOffsetSubgraph2(const std::shared_ptr<Function>& func,
                                          const std::shared_ptr<LogicalTensor>& input_tensor,
                                          const std::shared_ptr<LogicalTensor>& output_tensor)
{
    auto shape2Imme = OpImmediate::Specified({16, 32});

    auto inner_tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    inner_tensor2->SetMemoryTypeBoth(MEM_UB);
    inner_tensor2->UpdateOffset({0, 32});
    inner_tensor2->SetMagic(35);

    auto& copyopin2 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor2});
    copyopin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 32}), MEM_UB, shape2Imme,
                                                               shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin2.UpdateSubgraphID(2);
    copyopin2.opmagic = 10024;

    auto result_tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    result_tensor2->SetMemoryTypeBoth(MEM_UB);
    result_tensor2->SetMagic(34);

    auto& expopin2 = PassOperationUtils::AddOperation(*func, Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(2);
    expopin2.opmagic = 10026;

    auto& copyoutop2 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {result_tensor2}, {output_tensor});
    copyoutop2.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 32}), shape2Imme,
                                                                shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop2.UpdateSubgraphID(2);
    copyoutop2.opmagic = 10030;
}

TEST_F(SubgraphToFunctionTest, DifferentOffset)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TILE_DifferentOffset", "TILE_DifferentOffset",
                                           nullptr);
    EXPECT_TRUE(func != nullptr);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "use_max_freq_label", true);
    Program::GetInstance().InsertFuncToFunctionMap("TILE_DifferentOffset", func);

    auto input_tensor = BuildDifferentOffsetSubgraph0(func);

    auto output_tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 64}, CreateTestConstIntVector(std::vector<int64_t>{16, 64}));
    output_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    output_tensor->SetMagic(7);
    func->outCasts_.push_back(output_tensor);

    BuildDifferentOffsetSubgraph1(func, input_tensor, output_tensor);
    BuildDifferentOffsetSubgraph2(func, input_tensor, output_tensor);

    func->SetTotalSubGraphCount(3);

    SubgraphToFunction pass;
    pass.PreCheck(*func);
    pass.RunOnFunction(*func);
    pass.PostCheck(*func);

    auto rootFunc = func->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto& PSgToESgMap = GetPSgToESgMap(rootFunc);

    std::unordered_set<int> uniquePSgIds;
    for (const auto& pair : PSgToESgMap) {
        uniquePSgIds.insert(pair.first);
    }

    EXPECT_EQ(uniquePSgIds.size(), func->GetTotalSubGraphCount());
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

static void BuildSameOffsetSubgraph0(const std::shared_ptr<Function>& func,
                                     const std::shared_ptr<LogicalTensor>& input_tensor,
                                     const std::shared_ptr<LogicalTensor>& output_tensor)
{
    auto shape2Imme = OpImmediate::Specified({16, 32});

    auto inner_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    inner_tensor1->UpdateOffset({0, 0});
    inner_tensor1->SetMagic(30);
    inner_tensor1->SetMemoryTypeBoth(MEM_UB);

    auto& copyopin1 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor1});
    copyopin1.UpdateSubgraphID(0);
    copyopin1.opmagic = 10021;
    copyopin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme,
                                                               shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));

    auto result_tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    result_tensor1->SetMemoryTypeBoth(MEM_UB);
    result_tensor1->SetMagic(29);

    auto& expopin1 = PassOperationUtils::AddOperation(*func, Opcode::OP_EXP, {inner_tensor1}, {result_tensor1});
    expopin1.UpdateSubgraphID(0);
    expopin1.opmagic = 10023;

    auto& copyoutop1 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {result_tensor1}, {output_tensor});
    copyoutop1.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shape2Imme,
                                                                shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop1.UpdateSubgraphID(0);
    copyoutop1.opmagic = 10029;
}

static void BuildSameOffsetSubgraph1(const std::shared_ptr<Function>& func,
                                     const std::shared_ptr<LogicalTensor>& input_tensor,
                                     const std::shared_ptr<LogicalTensor>& output_tensor)
{
    auto shape2Imme = OpImmediate::Specified({16, 32});

    auto inner_tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    inner_tensor2->SetMemoryTypeBoth(MEM_UB);
    inner_tensor2->UpdateOffset({0, 0});
    inner_tensor2->SetMagic(35);

    auto& copyopin2 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_IN, {input_tensor}, {inner_tensor2});
    copyopin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme,
                                                               shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyopin2.UpdateSubgraphID(1);
    copyopin2.opmagic = 10024;

    auto result_tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 32}, CreateTestConstIntVector(std::vector<int64_t>{16, 32}));
    result_tensor2->SetMemoryTypeBoth(MEM_UB);
    result_tensor2->SetMagic(34);

    auto& expopin2 = PassOperationUtils::AddOperation(*func, Opcode::OP_EXP, {inner_tensor2}, {result_tensor2});
    expopin2.UpdateSubgraphID(1);
    expopin2.opmagic = 10026;

    auto& copyoutop2 = PassOperationUtils::AddOperation(*func, Opcode::OP_COPY_OUT, {result_tensor2}, {output_tensor});
    copyoutop2.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({16, 0}), shape2Imme,
                                                                shape2Imme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyoutop2.UpdateSubgraphID(1);
    copyoutop2.opmagic = 10030;
}

TEST_F(SubgraphToFunctionTest, SameOffset)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TILE_SameOffset", "TILE_SameOffset", nullptr);
    EXPECT_TRUE(func != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TILE_SameOffset", func);

    auto input_tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{16, 64}, CreateTestConstIntVector(std::vector<int64_t>{16, 64}));
    input_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor->SetMagic(79);

    auto output_tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DT_FP32, std::vector<int64_t>{32, 32}, CreateTestConstIntVector(std::vector<int64_t>{32, 32}));
    output_tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    output_tensor->SetMagic(7);

    BuildSameOffsetSubgraph0(func, input_tensor, output_tensor);
    BuildSameOffsetSubgraph1(func, input_tensor, output_tensor);

    func->inCasts_.push_back(input_tensor);
    func->outCasts_.push_back(output_tensor);
    func->SetTotalSubGraphCount(2);

    Json progDump;
    progDump["version"] = "2.0";
    progDump["functions"].push_back(func->DumpJson());
    std::ofstream file("Before_subgraphIsomorphismPass.json");
    file << progDump.dump() << std::endl;
    file.close();

    SubgraphToFunction pass;
    pass.PreCheck(*func);
    pass.RunOnFunction(*func);
    pass.PostCheck(*func);

    auto rootFunc = func->rootFunc_;
    EXPECT_NE(rootFunc, nullptr);
    const auto& PSgToESgMap = GetPSgToESgMap(rootFunc);

    std::unordered_set<int> uniquePSgIds;
    for (const auto& pair : PSgToESgMap) {
        uniquePSgIds.insert(pair.first);
    }
    size_t mergedSubgraphCount = uniquePSgIds.size();
    EXPECT_EQ(mergedSubgraphCount, 2);
    EXPECT_TRUE(ArePsgHashesUnique(*rootFunc));
    EXPECT_TRUE(IsPSgToESgMapOneToOne(PSgToESgMap));
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1)
{
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK")
    {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, k};
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, "A");
        auto output = std::make_tuple(Tensor(DT_FP32, output_shape, "npu_val"),
                                      Tensor(DT_FP32, output_shape, "resDics"));
        config::SetBuildStatic(true);
        FUNCTION("TOPK_T", {input_a, std::get<0>(output), std::get<1>(output)})
        {
            output = TopK(input_a, k, -1, isLargest);
        }
    }

    Json programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

#ifndef PRIOR_SCHEDULING
    EXPECT_EQ(programJsonNew.dump(), programJsonNewNew.dump());
#endif
    config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
}

TEST_F(SubgraphToFunctionTest, test_json_dump_and_load_1_cov)
{
    int32_t shape0 = 4;
    int32_t shape1 = 32;
    int32_t k = 8;
    bool isLargest = true;

    PROGRAM("TOPK")
    {
        std::vector<int64_t> input_shape = {shape0, shape1};
        std::vector<int64_t> output_shape = {shape0, k};
        TileShape::Current().SetVecTile({shape0, shape1});
        Tensor input_a(DT_FP32, input_shape, "A");
        auto output = std::make_tuple(Tensor(DT_FP32, output_shape, "npu_val"),
                                      Tensor(DT_FP32, output_shape, "resDics"));
        config::SetBuildStatic(true);
        FUNCTION("TOPK_T", {input_a, std::get<0>(output), std::get<1>(output)})
        {
            output = TopK(input_a, k, -1, isLargest);
        }
    }

    Json programJson = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJson);
    Json programJsonNew = Program::GetInstance().DumpJson();

    Program::GetInstance().LoadJson(programJsonNew);
    Json programJsonNewNew = Program::GetInstance().DumpJson();

    config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
}

/*
 * input -> view1(01) -> view1_out -/
 *                                  add(03) -> add_out -> abc(04) -> final_out
 * input -> View2(01) -> view2_out -/
 */
void InitGraphBuilder(ComputationalGraphBuilder& G, std::vector<int64_t> tileShape)
{
    // 1. 定义张量和操作
    std::vector<std::string> tensorNames = {"input", "view1_out", "view2_out", "add_out", "final_out"};
    std::vector<Opcode> opCodes = {Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands = {
        {"input"},                  // view1
        {"input"},                  // view2 (确保与view1不同输出)
        {"view1_out", "view2_out"}, // add (两个不同输入)
        {"add_out"}                 // abs
    };
    std::vector<std::vector<std::string>> ooperands = {{"view1_out"}, {"view2_out"}, {"add_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"view1", "view2", "add", "abs_final"};

    // 2. 添加张量和操作
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 3. 设置内存类型和边界张量
    auto input_tensor = G.GetTensor("input");
    auto final_out_tensor = G.GetTensor("final_out");
    input_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    final_out_tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    // 4. 设置输入输出转换
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));

    // 5. 设置子图ID（所有操作在同一个子图）
    for (const auto& opName : opNames) {
        G.GetOp(opName)->UpdateSubgraphID(0);
    }
}

TEST_F(SubgraphToFunctionTest, TestBasicSubgraphConversion)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    // 初始化
    InitGraphBuilder(G, tileShape);

    // 获取Function并执行子图转换Pass
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(1); // 总子图数=1

    SubgraphToFunction pass;
    Status status = pass.RunOnFunction(*function);
    status = pass.PostCheck(*function);
    EXPECT_EQ(status, SUCCESS);

    // 验证结果
    Function* rootFunc = function->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->GetGraphType(), GraphType::EXECUTE_GRAPH);

    // 检查子图调用信息
    const auto& topoInfo = rootFunc->topoInfo_;
    EXPECT_EQ(topoInfo.topology_.size(), 1); // 应有一个子图调用

    // 检查子图内部操作（应保留VIEW+VIEW+ADD+ABS）
    auto leafFunc = rootFunc->programs_.begin()->second;
    EXPECT_EQ(leafFunc->Operations().size(), 4);
}

TEST_F(SubgraphToFunctionTest, RestoredInplaceViewUsesNormalizedCopyInValidShape)
{
    ComputationalGraphBuilder graph;
    InitGraphBuilder(graph, {16, 16});
    auto function = graph.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetTotalSubGraphCount(1);

    for (const auto& [name, validRows] : {std::pair{"view1", 1}, std::pair{"view2", 0}}) {
        auto view = graph.GetOp(name);
        ASSERT_NE(view, nullptr);
        view->SetAttribute(OpAttributeKey::inplaceIdx, 0);
        view->SetOpAttribute(std::make_shared<ViewOpAttribute>(Offset{0, 0}, MEM_UB, std::vector<SymbolicScalar>{},
                                                               std::vector<SymbolicScalar>{validRows, 16}));
    }

    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    ASSERT_NE(function->rootFunc_, nullptr);

    bool passesZeroValidShape = false;
    bool passesOneValidShape = false;
    for (auto& op : function->rootFunc_->Operations(false)) {
        auto callAttr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
        if (callAttr == nullptr) {
            continue;
        }
        for (const auto& args : callAttr->GetArgList()) {
            const size_t validShapeIndex = COA_INDEX_DIM_BASE + 2 * 3;
            if (args.size() <= validShapeIndex + 1 || args[validShapeIndex + 1].Dump() != "16") {
                continue;
            }
            passesZeroValidShape |= args[validShapeIndex].Dump() == "0";
            passesOneValidShape |= args[validShapeIndex].Dump() == "1";
        }
    }
    EXPECT_TRUE(passesZeroValidShape);
    EXPECT_TRUE(passesOneValidShape);

    int foundViews = 0;
    for (const auto& [id, leaf] : function->rootFunc_->programs_) {
        (void)id;
        for (auto& op : leaf->Operations(false)) {
            if (op.GetOpcode() != Opcode::OP_VIEW || !op.HasAttribute(OpAttributeKey::inplaceIdx)) {
                continue;
            }
            auto attr = std::static_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            ASSERT_EQ(attr->GetToDynValidShape().size(), 2U);
            EXPECT_NE(attr->GetToDynValidShape()[0].Dump().find("RUNTIME_COA_GET_PARAM_VALID_SHAPE"),
                      std::string::npos);
            foundViews++;
        }
    }
    EXPECT_EQ(foundViews, 2);
}

TEST_F(SubgraphToFunctionTest, IgnoresRemovedTrailingSubgraph)
{
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    InitGraphBuilder(G, tileShape);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2);

    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    ASSERT_NE(function->rootFunc_, nullptr);
    EXPECT_EQ(function->rootFunc_->Operations().size(), 1);
    ASSERT_EQ(function->rootFunc_->programs_.size(), 1);
    EXPECT_EQ(function->rootFunc_->programs_.begin()->second->Operations().size(), 4);
}

static Function* BuildIsomorphicSubgraphsWithDifferentVecHashOrder(ComputationalGraphBuilder& G)
{
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, {"input", "out0", "out1"}));
    EXPECT_TRUE(G.AddOps({Opcode::OP_ABS, Opcode::OP_ABS}, {{"input"}, {"input"}}, {{"out0"}, {"out1"}},
                         {"abs0", "abs1"}, true));

    G.GetTensor("input")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    G.GetTensor("out0")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    G.GetTensor("out1")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"out0", "out1"}));

    G.GetOp("abs0")->UpdateSubgraphID(0);
    G.GetOp("abs1")->UpdateSubgraphID(1);
    G.GetOp("abs0")->SetHashOrderInfo(OpAttributeKey::vecMergeHashOrder, OpAttributeKey::vecMergeSubgraphCount,
                                      "func8_0", 4);
    G.GetOp("abs1")->SetHashOrderInfo(OpAttributeKey::vecMergeHashOrder, OpAttributeKey::vecMergeSubgraphCount,
                                      "func9_0", 1);

    auto* function = G.GetFunction();
    function->SetTotalSubGraphCount(2);
    return function;
}

TEST_F(SubgraphToFunctionTest, PreservesHashOrderInfoOnIsomorphicCallOps)
{
    ComputationalGraphBuilder G;
    Function* function = BuildIsomorphicSubgraphsWithDifferentVecHashOrder(G);

    SubgraphToFunction pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

    Function* rootFunc = function->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->programs_.size(), 1);
    const auto& callOps = rootFunc->Operations();
    ASSERT_EQ(callOps.size(), 2);

    auto firstInfo = callOps[0].GetHashOrderInfo(OpAttributeKey::vecMergeHashOrder,
                                                 OpAttributeKey::vecMergeSubgraphCount);
    auto secondInfo = callOps[1].GetHashOrderInfo(OpAttributeKey::vecMergeHashOrder,
                                                  OpAttributeKey::vecMergeSubgraphCount);
    EXPECT_EQ(firstInfo.first, "func8_0");
    EXPECT_EQ(firstInfo.second, 4);
    EXPECT_EQ(secondInfo.first, "func9_0");
    EXPECT_EQ(secondInfo.second, 1);
}

ir::StmtPtr ToSubgraphFunctionStmt(Operation& op)
{
    return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this());
}

bool HasSubgraphFunctionToken(const Operation& op, const ir::VarPtr& token)
{
    return std::find(op.tokens_.begin(), op.tokens_.end(), token) != op.tokens_.end();
}

Operation* FindCallOpBySubgraphId(Function& rootFunc, int subgraphId)
{
    for (auto& op : rootFunc.Operations()) {
        if (op.GetOpcode() == Opcode::OP_CALL && op.GetSubgraphID() == subgraphId) {
            return &op;
        }
    }
    return nullptr;
}

Function* BuildSubgraphTokenDependencyGraph(ComputationalGraphBuilder& G)
{
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, {"input", "mid", "output"}));
    EXPECT_TRUE(G.AddOp(Opcode::OP_ABS, {"input"}, {"mid"}, "producer"));
    EXPECT_TRUE(G.AddOp(Opcode::OP_EXP, {"mid"}, {"output"}, "consumer"));
    G.GetTensor("input")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    G.GetTensor("mid")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    G.GetTensor("output")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"output"}));
    G.GetOp("producer")->UpdateSubgraphID(0);
    G.GetOp("consumer")->UpdateSubgraphID(1);
    auto token = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    G.GetFunction()->GetVarDependency().AddProducer(token, ToSubgraphFunctionStmt(*G.GetOp("producer")));
    G.GetFunction()->GetVarDependency().AddConsumer(token, ToSubgraphFunctionStmt(*G.GetOp("consumer")));
    G.GetOp("producer")->result_token_ = {token};
    G.GetOp("consumer")->tokens_.push_back(token);
    G.GetFunction()->SetTotalSubGraphCount(2);
    return G.GetFunction();
}

TEST_F(SubgraphToFunctionTest, BuildsCallOpTokenDependency)
{
    ComputationalGraphBuilder G;
    auto* function = BuildSubgraphTokenDependencyGraph(G);
    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    auto* rootFunc = function->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    auto* producerCallOp = FindCallOpBySubgraphId(*rootFunc, 0);
    auto* consumerCallOp = FindCallOpBySubgraphId(*rootFunc, 1);
    ASSERT_NE(producerCallOp, nullptr);
    ASSERT_NE(consumerCallOp, nullptr);

    ASSERT_FALSE(producerCallOp->result_token_.empty());
    EXPECT_TRUE(rootFunc->GetVarDependency().HasProducer(producerCallOp->result_token_.front(),
                                                         ToSubgraphFunctionStmt(*producerCallOp)));
    EXPECT_TRUE(rootFunc->GetVarDependency().HasConsumer(producerCallOp->result_token_.front(),
                                                         ToSubgraphFunctionStmt(*consumerCallOp)));
    EXPECT_TRUE(HasSubgraphFunctionToken(*consumerCallOp, producerCallOp->result_token_.front()));

    bool producerSeen = false;
    for (auto* op : rootFunc->Operations(false).DuplicatedOpList()) {
        if (op == producerCallOp) {
            producerSeen = true;
        }
        if (op == consumerCallOp) {
            EXPECT_TRUE(producerSeen);
            break;
        }
    }
}

TEST_F(SubgraphToFunctionTest, MergesCyclicSubgraphsCreatedAfterPartition)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, {"input", "producer_out", "middle_out", "output"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"input"}, {"producer_out"}, "producer"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_EXP, {"producer_out"}, {"middle_out"}, "middle"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_NEG, {"middle_out"}, {"output"}, "consumer"));
    for (const auto& tensorName : {"input", "producer_out", "middle_out", "output"}) {
        G.GetTensor(tensorName)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    }
    ASSERT_TRUE(G.SetInCast({"input"}));
    ASSERT_TRUE(G.SetOutCast({"output"}));

    auto* producer = G.GetOp("producer");
    auto* middle = G.GetOp("middle");
    auto* consumer = G.GetOp("consumer");
    producer->UpdateSubgraphID(0);
    middle->UpdateSubgraphID(1);
    consumer->UpdateSubgraphID(0);
    G.GetFunction()->SetTotalSubGraphCount(2);
    auto token = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    G.GetFunction()->GetVarDependency().AddProducer(token, ToSubgraphFunctionStmt(*producer));
    G.GetFunction()->GetVarDependency().AddConsumer(token, ToSubgraphFunctionStmt(*consumer));
    producer->result_token_ = {token};
    consumer->tokens_.push_back(token);

    ASSERT_EQ(FinalizePartitionWithTokenDependency(*G.GetFunction()), SUCCESS);
    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*G.GetFunction()), SUCCESS);

    EXPECT_EQ(producer->GetSubgraphID(), middle->GetSubgraphID());
    EXPECT_EQ(middle->GetSubgraphID(), consumer->GetSubgraphID());
    EXPECT_EQ(G.GetFunction()->GetTotalSubGraphCount(), 1U);
    ASSERT_NE(G.GetFunction()->rootFunc_, nullptr);
    EXPECT_EQ(G.GetFunction()->rootFunc_->Operations().size(), 1U);
}

TEST_F(SubgraphToFunctionTest, SplitsMixedCoreCycleCreatedAfterPartition)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16},
                             {"inputA", "inputB", "inputC", "producer_out", "middle_out", "output"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"inputA", "inputB"}, {"producer_out"}, "producer"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_EXP, {"producer_out"}, {"middle_out"}, "middle"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"middle_out", "inputC"}, {"output"}, "consumer"));
    for (const auto& tensorName : {"inputA", "inputB", "inputC", "producer_out", "middle_out", "output"}) {
        G.GetTensor(tensorName)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    }
    ASSERT_TRUE(G.SetInCast({"inputA", "inputB", "inputC"}));
    ASSERT_TRUE(G.SetOutCast({"output"}));

    auto* producer = G.GetOp("producer");
    auto* middle = G.GetOp("middle");
    auto* consumer = G.GetOp("consumer");
    producer->UpdateSubgraphID(0);
    middle->UpdateSubgraphID(1);
    consumer->UpdateSubgraphID(0);
    G.GetFunction()->SetTotalSubGraphCount(2);
    auto token = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    G.GetFunction()->GetVarDependency().AddProducer(token, ToSubgraphFunctionStmt(*producer));
    G.GetFunction()->GetVarDependency().AddConsumer(token, ToSubgraphFunctionStmt(*consumer));
    producer->result_token_ = {token};
    consumer->tokens_.push_back(token);

    ASSERT_EQ(FinalizePartitionWithTokenDependency(*G.GetFunction()), SUCCESS);
    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*G.GetFunction()), SUCCESS);

    EXPECT_NE(producer->GetSubgraphID(), middle->GetSubgraphID());
    EXPECT_NE(middle->GetSubgraphID(), consumer->GetSubgraphID());
    EXPECT_NE(producer->GetSubgraphID(), consumer->GetSubgraphID());
    EXPECT_EQ(G.GetFunction()->GetTotalSubGraphCount(), 3U);
    ASSERT_NE(G.GetFunction()->rootFunc_, nullptr);
    EXPECT_EQ(G.GetFunction()->rootFunc_->Operations().size(), 3U);
}

TEST_F(SubgraphToFunctionTest, KeepsLocalBufferInsideSubgraphWhenSplittingMixedCoreCycle)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16},
                             {"inputA", "inputB", "inputC", "local", "cube_out", "vector_out", "output"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"inputA", "inputB"}, {"local"}, "local_producer"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"local", "inputC"}, {"cube_out"}, "cube_consumer"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_EXP, {"cube_out"}, {"vector_out"}, "vector_middle"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_NEG, {"vector_out"}, {"output"}, "vector_consumer"));
    for (const auto& tensorName : {"inputA", "inputB", "inputC", "cube_out", "vector_out", "output"}) {
        G.GetTensor(tensorName)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    }
    G.GetTensor("local")->SetMemoryTypeBoth(MemoryType::MEM_L0B);
    ASSERT_TRUE(G.SetInCast({"inputA", "inputB", "inputC"}));
    ASSERT_TRUE(G.SetOutCast({"output"}));

    auto* localProducer = G.GetOp("local_producer");
    auto* cubeConsumer = G.GetOp("cube_consumer");
    auto* vectorMiddle = G.GetOp("vector_middle");
    auto* vectorConsumer = G.GetOp("vector_consumer");
    localProducer->UpdateSubgraphID(0);
    cubeConsumer->UpdateSubgraphID(1);
    vectorMiddle->UpdateSubgraphID(1);
    vectorConsumer->UpdateSubgraphID(0);
    G.GetFunction()->SetTotalSubGraphCount(2);

    ASSERT_EQ(FinalizePartitionWithTokenDependency(*G.GetFunction()), SUCCESS);
    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*G.GetFunction()), SUCCESS);

    EXPECT_EQ(localProducer->GetSubgraphID(), cubeConsumer->GetSubgraphID());
    EXPECT_NE(cubeConsumer->GetSubgraphID(), vectorMiddle->GetSubgraphID());
    EXPECT_EQ(vectorMiddle->GetSubgraphID(), vectorConsumer->GetSubgraphID());
    EXPECT_EQ(G.GetFunction()->GetTotalSubGraphCount(), 2U);
}

TEST_F(SubgraphToFunctionTest, CoalescesCompatibleCoreRunsAfterSplittingMixedCoreCycle)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16},
                             {"inputA", "inputB", "inputC", "l1", "cube_out", "vector_out", "output"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"inputA", "inputB"}, {"l1"}, "aic_first"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_A_MUL_B, {"l1", "inputC"}, {"cube_out"}, "aic_second"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_EXP, {"cube_out"}, {"vector_out"}, "aiv_first"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_NEG, {"vector_out"}, {"output"}, "aiv_second"));
    for (const auto& tensorName : {"inputA", "inputB", "inputC", "cube_out", "vector_out", "output"}) {
        G.GetTensor(tensorName)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    }
    G.GetTensor("l1")->SetMemoryTypeBoth(MemoryType::MEM_L1);
    ASSERT_TRUE(G.SetInCast({"inputA", "inputB", "inputC"}));
    ASSERT_TRUE(G.SetOutCast({"output"}));

    auto* aicFirst = G.GetOp("aic_first");
    auto* aicSecond = G.GetOp("aic_second");
    auto* aivFirst = G.GetOp("aiv_first");
    auto* aivSecond = G.GetOp("aiv_second");
    aicFirst->UpdateSubgraphID(0);
    aicSecond->UpdateSubgraphID(1);
    aivFirst->UpdateSubgraphID(1);
    aivSecond->UpdateSubgraphID(0);
    G.GetFunction()->SetTotalSubGraphCount(2);

    ASSERT_EQ(FinalizePartitionWithTokenDependency(*G.GetFunction()), SUCCESS);
    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*G.GetFunction()), SUCCESS);

    EXPECT_EQ(aicFirst->GetSubgraphID(), aicSecond->GetSubgraphID());
    EXPECT_NE(aicSecond->GetSubgraphID(), aivFirst->GetSubgraphID());
    EXPECT_EQ(aivFirst->GetSubgraphID(), aivSecond->GetSubgraphID());
    EXPECT_EQ(G.GetFunction()->GetTotalSubGraphCount(), 2U);
}

TEST_F(SubgraphToFunctionTest, BuildsTokenOnlyStaticTopologyDependency)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, {"input", "producer_out", "consumer_out"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_ABS, {"input"}, {"producer_out"}, "producer"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_EXP, {"input"}, {"consumer_out"}, "consumer"));
    for (const auto& tensorName : {"input", "producer_out", "consumer_out"}) {
        G.GetTensor(tensorName)->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    }
    ASSERT_TRUE(G.SetInCast({"input"}));
    ASSERT_TRUE(G.SetOutCast({"producer_out", "consumer_out"}));

    auto* producer = G.GetOp("producer");
    auto* consumer = G.GetOp("consumer");
    producer->UpdateSubgraphID(1);
    consumer->UpdateSubgraphID(0);
    auto token = IRBuilder().CreateTokenVar(ir::Span::Unknown());
    producer->result_token_ = {token};
    consumer->tokens_.push_back(token);
    G.GetFunction()->SetTotalSubGraphCount(2);

    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*G.GetFunction()), SUCCESS);
    auto* rootFunc = G.GetFunction()->rootFunc_;
    ASSERT_NE(rootFunc, nullptr);
    ASSERT_EQ(rootFunc->topoInfo_.topology_.size(), 2);
    EXPECT_EQ(rootFunc->topoInfo_.topology_[1].readyState, 0);
    EXPECT_EQ(rootFunc->topoInfo_.topology_[0].readyState, -1);
    const auto& producerSuccessors = rootFunc->topoInfo_.topology_[1].outGraph;
    EXPECT_NE(std::find(producerSuccessors.begin(), producerSuccessors.end(), 0), producerSuccessors.end());

    auto* producerCallOp = FindCallOpBySubgraphId(*rootFunc, 1);
    auto* consumerCallOp = FindCallOpBySubgraphId(*rootFunc, 0);
    ASSERT_NE(producerCallOp, nullptr);
    ASSERT_NE(consumerCallOp, nullptr);
    const auto callOps = rootFunc->Operations(false).DuplicatedOpList();
    EXPECT_LT(std::find(callOps.begin(), callOps.end(), producerCallOp),
              std::find(callOps.begin(), callOps.end(), consumerCallOp));
}

static void BuildMultiSubgraphDependencyGraph(ComputationalGraphBuilder& G)
{
    std::vector<std::string> tensorNames = {"input", "aic_out", "aiv_out", "final_out"};
    std::vector<Opcode> opCodes = {Opcode::OP_A_MUL_B, Opcode::OP_ADD, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands = {{"input"}, {"aic_out"}, {"aiv_out"}};
    std::vector<std::vector<std::string>> ooperands = {{"aic_out"}, {"aiv_out"}, {"final_out"}};
    std::vector<std::string> opNames = {"matmul_aic", "add_aiv", "exp_aicpu"};

    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    G.GetOp("matmul_aic")->UpdateSubgraphID(0);
    G.GetOp("matmul_aic")->SetCoreType(CoreType::AIC);
    G.GetOp("matmul_aic")->SetAttribute(OpAttributeKey::isCube, true);

    G.GetOp("add_aiv")->UpdateSubgraphID(1);
    G.GetOp("add_aiv")->SetCoreType(CoreType::AIV);

    G.GetOp("exp_aicpu")->UpdateSubgraphID(2);
    G.GetOp("exp_aicpu")->SetCoreType(CoreType::AICPU);

    G.GetTensor("input")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    G.GetTensor("final_out")->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);

    EXPECT_TRUE(G.SetInCast({"input"}));
    EXPECT_TRUE(G.SetOutCast({"final_out"}));
}

static void VerifyMultiSubgraphDependency(Function* rootFunc)
{
    ASSERT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->programs_.size(), 3);

    const auto& topoInfo = rootFunc->topoInfo_;
    EXPECT_EQ(topoInfo.topology_.size(), 3);
    EXPECT_EQ(topoInfo.topology_[0].outGraph, std::unordered_set<int>{1}); // 0 -> 1
    EXPECT_EQ(topoInfo.topology_[1].outGraph, std::unordered_set<int>{2}); // 1 -> 2
    EXPECT_TRUE(topoInfo.topology_[2].outGraph.empty());                   // 2 无后继

    EXPECT_EQ(topoInfo.topology_[0].readyState, 0);
    EXPECT_EQ(topoInfo.topology_[1].readyState, -1);
    EXPECT_EQ(topoInfo.topology_[2].readyState, -1);

    const auto& callOps = rootFunc->Operations();
    ASSERT_EQ(callOps.size(), 3);

    auto check_graph_type = [&callOps](size_t idx, CoreType expected) {
        auto attr = dynamic_cast<CallOpAttribute*>(callOps[idx].GetOpAttribute().get());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->invokeInfo_->GetGraphType(), expected);
    };

    check_graph_type(0, CoreType::AIC);
    check_graph_type(1, CoreType::AIV);
    check_graph_type(2, CoreType::AICPU);

    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AIC), 1);
    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AIV), 0);
    EXPECT_EQ(rootFunc->GetReadySubGraphCount(CoreType::AICPU), 0);
}

TEST_F(SubgraphToFunctionTest, MultiSubgraphDependencyWithMixedOps)
{
    ComputationalGraphBuilder G;
    BuildMultiSubgraphDependencyGraph(G);

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(3);

    SubgraphToFunction pass;
    Status status = pass.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    VerifyMultiSubgraphDependency(function->rootFunc_);
}

TEST_F(SubgraphToFunctionTest, EliminateRedundantEdges)
{
    ComputationalGraphBuilder G;

    // 定义张量（需要更多张量来创建冗余路径）
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6"};

    // 定义操作和子图分配 - 创建两条路径到MAX_SG3
    std::vector<Opcode> opCodes{
        Opcode::OP_ADD,    // 子图0
        Opcode::OP_CONV2D, // 子图1
        Opcode::OP_ABS,    // 子图2
        Opcode::OP_ADD,    // 子图3
        Opcode::OP_MAXIMUM // 子图4
    };

    std::vector<std::vector<std::string>> ioperands{
        {"t0", "t1"}, // ADD1_SG0
        {"t2"},       // CONV_SG1
        {"t2"},       // ABS_SG2
        {"t3", "t4"}, // ADD2_SG3
        {"t4", "t5"}  // MAX_SG4 (接收来自ADD和ABS的输入)
    };

    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};

    std::vector<std::string> opNames{"ADD1_SG0", "CONV_SG1", "ABS_SG2", "ADD2_SG3", "MAX_SG4"};

    // 创建图和操作
    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 设置子图ID - 创建跨子图冗余
    G.GetOp("ADD1_SG0")->UpdateSubgraphID(0);
    G.GetOp("CONV_SG1")->UpdateSubgraphID(1);
    G.GetOp("ABS_SG2")->UpdateSubgraphID(2);
    G.GetOp("ADD2_SG3")->UpdateSubgraphID(3);
    G.GetOp("MAX_SG4")->UpdateSubgraphID(4);

    // 设置边界张量
    EXPECT_TRUE(G.SetInCast({"t0", "t1"}));
    EXPECT_TRUE(G.SetOutCast({"t6"}));

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(5); // 共5个子图
    // 2. 运行SubgraphToFunction pass
    SubgraphToFunction pass;
    pass.SetupStaticProcessor();
    // 构建基础图结构
    pass.staticProcessor_.BuildGraph(*function);
    pass.RecordIncastOutcast(*function);

    // 3. 验证初始边关系（通过消费者关系）
    auto* abs_op = G.GetOp("ABS_SG2");
    auto* add2_op = G.GetOp("ADD2_SG3");
    auto* max_op = G.GetOp("MAX_SG4");

    // 验证ABS_SG2的消费者包含ADD2_SG3、MAX_SG4
    auto abs_consumers1 = abs_op->ConsumerOps();
    EXPECT_TRUE(abs_consumers1.find(add2_op) != abs_consumers1.end());
    EXPECT_TRUE(abs_consumers1.find(max_op) != abs_consumers1.end());

    // 4. 构建颜色图并消除冗余边
    pass.staticProcessor_.BuildColorGraph(*function);
    pass.staticProcessor_.EraseRedundantColorEdges(*function);

    // 5. 验证冗余边已被移除
    const auto& colorOutGraph = pass.staticProcessor_.colorOutGraph;
    const int abs_sgid = G.GetOp("ABS_SG2")->GetSubgraphID();
    const int max_sgid = G.GetOp("MAX_SG4")->GetSubgraphID();
    const auto& abs_out_edges = colorOutGraph[abs_sgid];
    bool found = std::find(abs_out_edges.begin(), abs_out_edges.end(), max_sgid) != abs_out_edges.end();
    EXPECT_FALSE(found) << "Redundant edge not removed!";
}

TEST_F(SubgraphToFunctionTest, ReshapeDependencyHandling)
{
    ComputationalGraphBuilder G;

    // 1. 构建测试图：包含一个RESHAPE操作和其消费者
    std::vector<std::string> tensorNames{"t0", "t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE, Opcode::OP_ABS};
    std::vector<std::vector<std::string>> ioperands{
        {"t0"}, // RESHAPE_SG0 (无输入子图)
        {"t1"}  // ABS_SG1 (输入来自RESHAPE)
    };
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}};
    std::vector<std::string> opNames{"RESHAPE_SG0", "ABS_SG1"};

    EXPECT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames));
    EXPECT_TRUE(G.AddOps(opCodes, ioperands, ooperands, opNames, true));

    // 2. 设置子图ID（确保RESHAPE是独立子图）
    auto set_subgraph_id = [&G](const std::string& op_name, int id) {
        auto* op = G.GetOp(op_name);
        ASSERT_NE(op, nullptr) << "Operation " << op_name << " not found!";
        op->UpdateSubgraphID(id);
    };
    set_subgraph_id("RESHAPE_SG0", 0); // RESHAPE单独子图且无输入子图
    set_subgraph_id("ABS_SG1", 1);

    // 3. 构建函数并运行pass
    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(2); // 2个子图

    SubgraphToFunction pass;
    pass.SetupStaticProcessor(); // 初始化静态处理器
    // 通过静态处理器调用BuildGraph
    pass.staticProcessor_.BuildGraph(*function);
    pass.RecordIncastOutcast(*function);
    pass.ConstructParamMap(*function);

    // 4. 验证RESHAPE子图的特殊处理
    auto* reshape_op = G.GetOp("RESHAPE_SG0");
    ASSERT_NE(reshape_op, nullptr);
    const int reshape_sgid = reshape_op->GetSubgraphID();

    // 4.1 验证RESHAPE子图被正确标记
    EXPECT_TRUE(pass.staticProcessor_.isReshape[reshape_sgid])
        << "RESHAPE subgraph should be marked when it has no input subgraph and single reshape op";

    EXPECT_TRUE(function->topoInfo_.GetSuccs(reshape_sgid).empty())
        << "RESHAPE subgraph should have empty successors set";

    int expected_out_degree = 0; // 根据实际图结构调整这个值
    bool found = false;
    for (const auto& entry : function->topoInfo_.GetTopology()) {
        if (entry.esgId == 1) { // ABS_SG1的子图ID
            EXPECT_EQ(entry.readyState, -1 * expected_out_degree)
                << "Consumer subgraph's readyOrNot should exclude RESHAPE inputs";
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "ABS_SG1 subgraph entry not found in topology";
}

TEST_F(SubgraphToFunctionTest, ExportsFunctionOutcastWrittenAndReadInSameSubgraph)
{
    ComputationalGraphBuilder G;
    ASSERT_TRUE(G.AddTensors(DataType::DT_FP32, {16, 16},
                             {MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                              MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR},
                             {"x", "ub", "buf", "ub2", "y"}));
    ASSERT_TRUE(G.AddOp(Opcode::OP_COPY_IN, {"x"}, {"ub"}, "copy_in_x"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_COPY_OUT, {"ub"}, {"buf"}, "copy_out_buf"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_COPY_IN, {"buf"}, {"ub2"}, "copy_in_buf"));
    ASSERT_TRUE(G.AddOp(Opcode::OP_COPY_OUT, {"ub2"}, {"y"}, "copy_out_y"));
    ASSERT_TRUE(G.SetInCast({"x"}));
    ASSERT_TRUE(G.SetOutCast({"buf", "y"}));
    for (const char* name : {"copy_in_x", "copy_out_buf", "copy_in_buf", "copy_out_y"}) {
        G.GetOp(name)->UpdateSubgraphID(0);
    }

    Function* function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    function->SetTotalSubGraphCount(1);

    SubgraphToFunction pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    ASSERT_NE(function->rootFunc_, nullptr);
    ASSERT_EQ(function->rootFunc_->programs_.size(), 1U);

    const int bufRawMagic = G.GetTensor("buf")->GetRawMagic();
    auto containsBuf = [bufRawMagic](const auto& tensors) {
        return std::any_of(tensors.begin(), tensors.end(), [bufRawMagic](const auto& tensor) {
            return tensor != nullptr && tensor->GetRawMagic() == bufRawMagic;
        });
    };
    EXPECT_TRUE(containsBuf(function->rootFunc_->programs_.begin()->second->GetOutcast()));
    bool callProducesBuf = false;
    for (const auto& op : function->rootFunc_->Operations()) {
        callProducesBuf = callProducesBuf || containsBuf(op.GetOOperands());
    }
    EXPECT_TRUE(callProducesBuf);
}

TEST_F(SubgraphToFunctionTest, LightweightStableSortIgnoresInplaceSelfDependency)
{
    Program program;
    auto function = std::make_unique<Function>(program, "inplace_call_sort", "inplace_call_sort", nullptr);
    IRBuilder builder;
    auto inplaceTensor = builder.CreateTensorVar(*function, DataType::DT_FP32, {16, 16});
    auto output = builder.CreateTensorVar(*function, DataType::DT_FP32, {16, 16});

    auto& inplaceCall = function->AddRawOperation(Opcode::OP_CALL, {inplaceTensor}, {inplaceTensor});
    auto& consumer = function->AddRawOperation(Opcode::OP_ABS, {inplaceTensor}, {output});

    function->SortOperations(SortOperationsMode::LIGHTWEIGHT_STABLE);
    ASSERT_EQ(function->Operations(false).size(), 2);
    EXPECT_EQ(&function->Operations(false)[0], &inplaceCall);
    EXPECT_EQ(&function->Operations(false)[1], &consumer);
}

std::shared_ptr<LogicalTensor> GetTensorDataSubgraphTensor(Operation& refOp);

TEST_F(SubgraphToFunctionTest, UB2L1DynamicAttributesUseL1Tensor)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "UB2L1Dependency", "UB2L1Dependency", nullptr);
    const Shape shape{32, 64};
    auto input = IRBuilder().CreateTensorVar(DT_FP16, shape, CreateTestConstIntVector(shape));
    auto output = IRBuilder().CreateTensorVar(DT_FP16, shape, CreateTestConstIntVector(shape));
    input->SetMemoryTypeBoth(MEM_UB);
    output->SetMemoryTypeBoth(MEM_L1);
    auto& op = func->AddOperation(Opcode::OP_UB_COPY_L1, {input}, {output});
    auto offset = CreateTestScalarVar("ub2l1_offset");
    IRBuilder builder;
    SymbolicScalar getTensorData(AddRuntimePrefix("GetTensorDataInt32Dim1"));
    auto validDim = getTensorData(builder.CreateConstInt(0), builder.CreateConstInt(0), builder.CreateConstInt(0),
                                  builder.CreateConstInt(0), builder.CreateConstInt(0));
    auto attr = std::make_shared<CopyOpAttribute>(
        std::vector<OpImmediate>{OpImmediate::Specified(offset), OpImmediate::Parameter(1)}, MEM_L1,
        OpImmediate::Specified(shape), OpImmediate::Specified(shape),
        std::vector<OpImmediate>{OpImmediate::Specified(validDim), OpImmediate::Parameter(2)});
    op.SetOpAttribute(attr);

    auto dynamicAttrs = op.GetDynamicAttributeList();
    ASSERT_EQ(dynamicAttrs.size(), 2);
    EXPECT_EQ(GetTensorDataDict(dynamicAttrs).size(), 1);
    EXPECT_EQ(&dynamicAttrs[0].get(), &attr->GetFromOffset()[0].GetSpecifiedValue());
    EXPECT_EQ(&dynamicAttrs[1].get(), &attr->GetToDynValidShape()[0].GetSpecifiedValue());
    // Rebinding must update the original attribute rather than a temporary copy.
    dynamicAttrs[1].get() = IRBuilder().CreateConstInt(17);
    EXPECT_EQ(attr->GetToDynValidShape()[0].GetSpecifiedValue().Concrete(), 17);
    EXPECT_EQ(GetTensorDataSubgraphTensor(op), output);
}
} // namespace tile_fwk
} // namespace npu
