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
 * \file test_loop_unroll.cpp
 * \brief Unit test for LoopUnroll pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/error_manager.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "interface/interpreter/raw_tensor_data.h"
#define private public
#include "passes/tensor_graph_pass/loop_unroll.h"
#undef private
#include "computational_graph_builder.h"

#include <fstream>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {
class LoopUnrollTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "FunctionUnroll");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
        std::vector<std::string> funcName = {"TENSOR_main"};
        config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
        int s = 32;
        TileShape::Current().SetVecTile(s, s);
        TileShape::Current().SetCubeTile({s, s}, {s, s}, {s, s});
    }
    void TearDown() override {}
};

// Test nested LOOPs
TEST_F(LoopUnrollTest, TestInnerLoopOrder)
{
    ConfigManager::Instance();
    int vecLen = 128;
    int loopNum = 5;
    int tileNum = 4;
    Tensor inputA(DT_FP32, {loopNum, vecLen}, "inputA");
    Tensor inputB(DT_FP32, {tileNum, vecLen}, "inputB");
    Tensor output(DT_FP32, {1, vecLen}, "out");

    FUNCTION("main", {inputA, inputB}, {output})
    {
        LOOP("Outer", FunctionType::DYNAMIC_LOOP, i, LoopRange(tileNum))
        {
            Tensor tileB(DT_FP32, {1, vecLen}, "tileB");
            LOOP("Inner", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
            {
                (void)j;
                auto tile = View(inputB, {1, vecLen}, {i, 0});
                tileB = Mul(tile, Element(DataType::DT_FP32, 1.0));
            }

            LOOP("Inner2", FunctionType::DYNAMIC_LOOP, k, LoopRange(loopNum))
            {
                auto tileA = View(inputA, {1, vecLen}, {k, 0});
                tileB = Add(tileA, tileB);
            }

            LOOP("Inner3", FunctionType::DYNAMIC_LOOP, l, LoopRange(1))
            {
                (void)l;
                tileB = Mul(tileB, Element(DataType::DT_FP32, 1.0));
                Assemble(tileB, {i, 0}, output);
            }
        }
    }
}

// Test GetTileShape
TEST_F(LoopUnrollTest, test_only_reshape2)
{
    int b = 1;
    int sq = 128;
    int d = 64;
    int bSq = (b == -1) ? -1 : b * sq;
    std::vector<int64_t> qShape = {b, sq, d};

    Tensor q(DT_FP32, qShape, "q");
    Tensor out(DT_FP32, {bSq, d}, "out");

    FUNCTION("main", {q}, {out})
    {
        Tensor qReshape(DT_FP32, {bSq, d}, "qReshape");
        LOOP("LOOP_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(0, 1, 1), {}, true)
        {
            (void)batchId;
            qReshape = Reshape(q, {bSq, d}, true);
        }

        LOOP("L0_AF", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(GetInputShape(q, 0)), {}, true)
        {
            Tensor q0 = View(qReshape, {sq, d}, {batchId * sq, 0});
            auto tmp = Exp(q0);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }
}

TEST_F(LoopUnrollTest, IsNoOverlapWAWAssembleAttrNull)
{
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, {8, 8}, "a");
    G.AddTensor(DataType::DT_FP32, {8, 8}, "b");
    G.AddTensor(DataType::DT_FP32, {8, 8}, "c");
    G.AddOp(Opcode::OP_ASSEMBLE, {"a"}, {"b"}, "assemble1");
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"c"}, "assemble2");
    G.SetInCast({"a"});
    G.SetOutCast({"c"});
    LoopUnroll pass;
    pass.lastWriteMap_[0] = std::make_pair(G.GetTensor("b"), true);
    std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap;
    EXPECT_FALSE(pass.IsNoOverlapWAW(0, G.GetTensor("c"), opDynOffsetMap));
}

// Test IF and nested FUNCTION
TEST_F(LoopUnrollTest, TestLoopIfWithRank)
{
    int s = 32;
    int n = 10;

    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {s, s}, "r0");
    Tensor out(DT_FP32, {s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 0.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int len = 8;
    int three = 3;
    int six = 6;

    FUNCTION("main", {t0, r0}, {out})
    {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(len))
        {
            IF(i < three)
            {
                IF(IsLoopEnd(i, len)) { r0 = Add(r0, Element(DataType::DT_FP32, 1.0)); }
                ELSE { r0 = Add(r0, Element(DataType::DT_FP32, 1.0)); }
            }
            ELSE
            {
                IF(i < six) { r0 = Add(r0, Element(DataType::DT_FP32, 0.0)); }
                ELSE
                {
                    Tensor t0v = View(t0, {s, s}, {s * i, 0});
                    r0 = Add(t0v, r0);
                }
            }
        }
        config::SetBuildStatic(true);
        FUNCTION("S1") { out = Add(r0, Element(DataType::DT_FP32, 1.0)); }
    }
}

namespace {
// 逐条弹出 ErrorManager 队列，查找包含 substr 的错误消息；未命中的消息打印后丢弃。
// 命中后不清空队列，便于同一用例内继续断言后续消息；用例末尾需自行清空队列。
bool FindReportedErrorContains(const std::string& substr, std::string& foundMsg)
{
    std::string errMsg;
    while (ErrorManager::Instance().GetFirstErrorMessage(errMsg)) {
        if (errMsg.find(substr) != std::string::npos) {
            foundMsg = errMsg;
            return true;
        }
        ErrorManager::Instance().OutputErrorMessage(false);
    }
    return false;
}

// 构造指向 calleeMagicName 的 CALL 算子
Operation* MakeCallop(Function* caller, const std::string& calleeMagicName)
{
    auto& call = caller->AddOperation(Opcode::OP_CALL, {}, {});
    auto attr = std::make_shared<CallOpAttribute>();
    attr->SetCalleeMagicName(calleeMagicName);
    call.SetOpAttribute(attr);
    return &call;
}
} // namespace

// CALL 算子指向 Program 中不存在的被调函数，GetCallee 报错返回 FAILED
TEST_F(LoopUnrollTest, GetCalleeMissingFunctionFails)
{
    auto caller = std::make_shared<Function>(Program::GetInstance(), "caller_magic", "TENSOR_caller", nullptr);
    Operation* callop = MakeCallop(caller.get(), "missing_callee");

    LoopUnroll pass;
    ErrorManager::Instance().OutputErrorMessage(true);
    Function* callee = nullptr;
    EXPECT_EQ(pass.GetCallee(callop, callee), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("Get callee function missing_callee failed in LoopUnroll.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// ExpandDynamicLoop 的 CALL 算子指向不存在的被调函数，报 "GetCallee failed in ExpandDynamicLoop."
TEST_F(LoopUnrollTest, ExpandDynamicLoopCalleeMissingFails)
{
    auto caller = std::make_shared<Function>(Program::GetInstance(), "outer_caller_magic", "TENSOR_outer_caller",
                                             nullptr);
    Operation* callop = MakeCallop(caller.get(), "missing_callee");

    LoopUnroll pass;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.ExpandDynamicLoop(callop), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("GetCallee failed in ExpandDynamicLoop.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// ExpandDynamicFunction 的 CALL 算子指向不存在的被调函数，报 "GetCallee failed in ExpandDynamicFunction."
TEST_F(LoopUnrollTest, ExpandDynamicFunctionCalleeMissingFails)
{
    auto caller = std::make_shared<Function>(Program::GetInstance(), "expand_caller_magic", "TENSOR_expand_caller",
                                             nullptr);
    Operation* callop = MakeCallop(caller.get(), "missing_callee");

    LoopUnroll pass;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.ExpandDynamicFunction(callop), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("GetCallee failed in ExpandDynamicFunction.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// 循环路径展开时选中 path 里的 CALL 算子指向不存在的被调函数，
// 嵌套 ExpandDynamicFunction 失败后报 "ExpandDynamic failed in ExpandDynamicLoop."
TEST_F(LoopUnrollTest, ExpandDynamicLoopNestedExpandFails)
{
    auto loopFunc = std::make_shared<Function>(Program::GetInstance(), "loop_func_magic", "TENSOR_loop_func", nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("loop_func_magic", loopFunc);
    // 循环体内 path 指向的 CALL 算子，其被调函数不存在
    Operation* innerCallop = MakeCallop(loopFunc.get(), "missing_inner_callee");
    auto loopAttr = std::make_shared<DynloopFunctionAttribute>("i", LoopRange(1), LoopRange(1), false);
    loopAttr->pathList.emplace_back(nullptr, std::vector<DynloopFunctionPathCondition>{}, innerCallop);
    loopFunc->SetDynloopAttribute(loopAttr);
    // 外层 CALL 算子指向 loopFunc，保证进入循环展开
    Operation* outerCallop = MakeCallop(loopFunc.get(), "loop_func_magic");

    LoopUnroll pass;
    pass.evaluateSymbol_ = std::make_shared<EvaluateSymbol>();
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.ExpandDynamicLoop(outerCallop), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("GetCallee failed in ExpandDynamicFunction.", errMsg));
    EXPECT_TRUE(FindReportedErrorContains("ExpandDynamic failed in ExpandDynamicLoop.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// functionMap 中已存在同名 Loop_Unroll 函数时，CreateLoopUnrollFunc 报 "already exists in functionMap."
TEST_F(LoopUnrollTest, CreateLoopUnrollFuncDuplicateNameFails)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "dup_top_magic", "TENSOR_dup_top", nullptr);
    auto existing = std::make_shared<Function>(Program::GetInstance(), "occupied_magic", "occupied_raw", nullptr);
    // 提前用与 pass 相同的规则占用 magic 名（此后不能再构造 Function，否则 CurId 变化）
    std::string funcMagicName = function->GetRawName() + "_Loop_Unroll" + "_" +
                                std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
    Program::GetInstance().InsertFuncToFunctionMap(funcMagicName, existing);
    auto* previousFunction = Program::GetInstance().GetCurrentFunction();

    LoopUnroll pass;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.CreateLoopUnrollFunc(function.get()), FAILED);
    EXPECT_EQ(Program::GetInstance().GetCurrentFunction(), previousFunction);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("Function[" + funcMagicName + "] already exists in functionMap.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// 静态转换入口函数的 CALL 算子指向不存在的被调函数，
// TopFunctionUnroll 内 ExpandDynamicFunction 失败后报 "ExpandDynamic failed in TopFunctionUnroll."
TEST_F(LoopUnrollTest, TopFunctionUnrollExpandDynamicFails)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "top_magic", "TENSOR_top", nullptr);
    (void)MakeCallop(func.get(), "missing_callee");

    LoopUnroll pass;
    pass.staticFuncNames_ = {"TENSOR_top"};
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.TraverseCallOp(func.get()), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("GetCallee failed in ExpandDynamicFunction.", errMsg));
    EXPECT_TRUE(FindReportedErrorContains("ExpandDynamic failed in TopFunctionUnroll.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// 非静态转换入口函数的 CALL 算子指向不存在的被调函数，TraverseCallOp 报 "GetCallee failed in TraverseCallOp."
TEST_F(LoopUnrollTest, TraverseCallOpNonStaticCalleeMissingFails)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "child_magic", "TENSOR_child", nullptr);
    (void)MakeCallop(func.get(), "missing_callee");

    LoopUnroll pass;
    pass.staticFuncNames_ = {};
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.TraverseCallOp(func.get()), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("GetCallee failed in TraverseCallOp.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// slot 已有同 loop 迭代内的写入（crossLoop=false）、无 WAR 依赖且非无重叠 WAW，
// 落入既非新建也非复用的场景，报 "Unexpected WAW case for tensor[%d] in slot[%d]."
TEST_F(LoopUnrollTest, FindOutputGlobalTensorUnexpectedWawFails)
{
    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "a"));
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "b"));

    LoopUnroll pass;
    pass.lastWriteMap_[0] = std::make_pair(graph.GetTensor("a"), false);
    std::unordered_map<int, LogicalTensorPtr> tensor2Global;
    std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.FindOutputGlobalTensor(0, tensor2Global, {}, graph.GetTensor("b"), opDynOffsetMap), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("Unexpected WAW case for tensor", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// 输入 tensor 的 magic 已存在于 tensor2Global 映射，报 "Tensor[%d] already exists in tensor2Global."
TEST_F(LoopUnrollTest, FindInputGlobalTensorDuplicateMappingFails)
{
    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "a"));
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "b"));

    LoopUnroll pass;
    pass.lastWriteMap_[0] = std::make_pair(graph.GetTensor("a"), false);
    std::unordered_map<int, LogicalTensorPtr> tensor2Global;
    tensor2Global[graph.GetTensor("b")->GetMagic()] = graph.GetTensor("a");
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.FindInputGlobalTensor(0, tensor2Global, graph.GetTensor("b")), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("already exists in tensor2Global.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// 输入 tensor 在函数入侧占用多个 slot，CreateGlobalTensor 报
// "Input tensor[%d] has multi slot[%zu], not support now."
TEST_F(LoopUnrollTest, CreateGlobalTensorInputMultiSlotFails)
{
    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "a"));
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "b"));
    EXPECT_TRUE(graph.AddOp(Opcode::OP_MUL, {"a"}, {"b"}, "mul"));
    EXPECT_TRUE(graph.SetInCast({"a"}));
    Function* func = graph.GetFunction();
    func->SetSlotScope(std::make_shared<TensorSlotScope>(func));
    func->GetSlotScope()->ioslot.incastSlot = {{0, 1}}; // 同一 incast tensor 挂两个 slot

    LoopUnroll pass;
    std::unordered_map<int, LogicalTensorPtr> tensor2Global;
    std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap;
    Operation* op = graph.GetOp("mul");
    ASSERT_NE(op, nullptr);
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.CreateGlobalTensor(opDynOffsetMap, tensor2Global, op, func), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("Input tensor[", errMsg));
    EXPECT_TRUE(FindReportedErrorContains("] has multi slot[2], not support now.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}

// 输出 tensor 在函数出侧占用多个 slot，CreateGlobalTensor 报
// "Output tensor[%d] has multi slot[%zu], not support now."
TEST_F(LoopUnrollTest, CreateGlobalTensorOutputMultiSlotFails)
{
    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "b"));
    EXPECT_TRUE(graph.AddOp(Opcode::OP_MUL, {}, {"b"}, "mul"));
    EXPECT_TRUE(graph.SetOutCast({"b"}));
    Function* func = graph.GetFunction();
    func->SetSlotScope(std::make_shared<TensorSlotScope>(func));
    func->GetSlotScope()->ioslot.outcastSlot = {{0, 1}}; // 同一 outcast tensor 挂两个 slot

    LoopUnroll pass;
    std::unordered_map<int, LogicalTensorPtr> tensor2Global;
    std::unordered_map<Operation*, std::vector<int64_t>> opDynOffsetMap;
    Operation* op = graph.GetOp("mul");
    ASSERT_NE(op, nullptr);
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.CreateGlobalTensor(opDynOffsetMap, tensor2Global, op, func), FAILED);
    std::string errMsg;
    EXPECT_TRUE(FindReportedErrorContains("Output tensor[", errMsg));
    EXPECT_TRUE(FindReportedErrorContains("] has multi slot[2], not support now.", errMsg));
    ErrorManager::Instance().OutputErrorMessage(true);
}
} // namespace tile_fwk
} // namespace npu
