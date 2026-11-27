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
 * \file test_axiscombine.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class TestAxisCombine : public ::testing::Test {
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

constexpr int64_t K_1 = 1;
constexpr int64_t K_2 = 2;
constexpr int64_t K_4 = 4;
constexpr int64_t K_8 = 8;
constexpr int64_t K_16 = 16;
constexpr int64_t K_32 = 32;
constexpr int64_t K_64 = 64;
constexpr int64_t K_128 = 128;

TEST_F(TestAxisCombine, Test1)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 127}, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 127}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"add"};
    EXPECT_EQ(graph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t brcbCnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++brcbCnt;
            auto outputTensor = op.GetOOperands()[0];
            EXPECT_TRUE(outputTensor->GetConsumers().size() != 0);
        }
        if (op.HasAttr(OpAttributeKey::brcbIdx)) {
            auto idx = op.GetIntAttribute(OpAttributeKey::brcbIdx) - 1;
            auto tensor = op.GetIOperands()[idx];
            EXPECT_TRUE(tensor != nullptr);
            EXPECT_EQ(tensor->shape[0], K_4);
            EXPECT_EQ(tensor->shape[1], K_8);
        }
    }
    EXPECT_EQ(brcbCnt, K_1);
}

TEST_F(TestAxisCombine, Test2)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 128}, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 1}, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 128}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ROWSUM_SINGLE, Opcode::OP_SUB};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}};
    std::vector<std::string> opNames{"rowmax", "add"};
    EXPECT_EQ(graph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    graph.GetOp("rowmax")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);
    auto* rootFuncPtr = graph.GetFunction();
    AxisCombine pass;
    rootFuncPtr->paramConfigs_.combineAxis = true;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
            auto outputTensor = op.GetOOperands()[0];
            EXPECT_TRUE(outputTensor->GetConsumers().size() != 0);
        }
        if (op.HasAttr(OpAttributeKey::brcbIdx)) {
            auto idx = op.GetIntAttribute(OpAttributeKey::brcbIdx) - 1;
            auto tensor = op.GetIOperands()[idx];
            EXPECT_TRUE(tensor != nullptr);
            EXPECT_EQ(tensor->shape[1], K_8);
            EXPECT_EQ(tensor->shape[0], K_4);
        }
    }
    EXPECT_EQ(cnt, K_1);
}

TEST_F(TestAxisCombine, Test3)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 128}, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t5"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 16}, "t6"), true);
    std::vector<Opcode> opCodes{Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2", "t3"}, {"t2", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t6"}};
    std::vector<std::string> opNames{"max", "add1", "add2"};
    EXPECT_EQ(graph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    graph.GetOp("max")->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);

    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // ================== Verify Pass Effect ==================
    auto updatedOperations = rootFuncPtr->Operations();
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            auto outputTensor = op.GetOOperands()[0];
            EXPECT_TRUE(outputTensor->GetConsumers().size() != 0);
        }
        if (op.HasAttr(OpAttributeKey::brcbIdx)) {
            auto idx = op.GetIntAttribute(OpAttributeKey::brcbIdx) - 1;
            auto tensor = op.GetIOperands()[idx];
            EXPECT_TRUE(tensor != nullptr);
            EXPECT_EQ(tensor->shape[0], K_16);
            EXPECT_EQ(tensor->shape[1], K_8);
            EXPECT_EQ(tensor->GetRawTensor()->GetRawShape()[0], K_16);
            EXPECT_EQ(tensor->GetRawTensor()->GetRawShape()[1], K_8);
        }
    }
}

// Skip insert when Both inputs have last dim shape of 1.
TEST_F(TestAxisCombine, Test4)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {-1, 1}, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {-1, 1}, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 1}, "t5"), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_EXPANDEXPDIF};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t3"}, {"t2", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"c1", "c2", "expanddif"};
    EXPECT_EQ(graph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // ================== Verify Pass Effect ==================
    int cnt = 0;
    for (const auto& op : rootFuncPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND || op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
}

TEST_F(TestAxisCombine, TestDD)
{
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

    TileShape::Current().SetVecTile(K_1, K_1, K_32, K_32);
    std::vector<int64_t> tshape = {K_2, K_2, K_64, K_64};

    Tensor T(DT_FP32, tshape, "T");
    Tensor d;
    Tensor output;
    FUNCTION("Test")
    {
        d = SoftmaxNew(T);
        output = Amax(d, -1, true);
    }

    auto funcMap = Program::GetInstance().GetFunctionMap();
}

/*
 * Graph: gm1[8,16] -> ci1[8,1] -> assemble1 -> t2[8,1]
 *        gm2[8,1]  -> ci2[8,1] -> assemble2 -> t4[8,1]
 * t2 and t4 share one RawTensor. The first copy-in disables axis combine,
 * while the second one enables it. The shared raw magic must propagate the
 * disable status to t4, so ADD inserts EXPAND instead of BRCB.
 */
TEST_F(TestAxisCombine, AssembleSharedRawMagicDisable)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_DEVICE_DDR, "gm1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "ci1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm1"}, {"ci1"}, "copy_in1", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "gm2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "ci2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm2"}, {"ci2"}, "copy_in2", true), true);

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"ci1"}, {"t2"}, "assemble1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"ci2"}, {"t4"}, "assemble2", true), true);

    auto t2 = graph.GetTensor("t2");
    auto t4 = graph.GetTensor("t4");
    t4->tensor = t2->tensor;

    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "rhs"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "out"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t4", "rhs"}, {"out"}, "add", true), true);

    auto* rootFuncPtr = graph.GetFunction();
    rootFuncPtr->paramConfigs_.combineAxis = true;
    AxisCombine pass;
    EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);

    int expandCount = 0;
    int brcbCount = 0;
    for (const auto& op : rootFuncPtr->Operations()) {
        expandCount += op.GetOpcode() == Opcode::OP_EXPAND;
        brcbCount += op.GetOpcode() == Opcode::OP_BRCB;
    }
    EXPECT_EQ(expandCount, 1);
    EXPECT_EQ(brcbCount, 0);
}

TEST_F(TestAxisCombine, MoveAssembleConsumerTokenToInsertedOp)
{
    auto runCase = [](bool disableAxisCombine, Opcode expectedOpcode) {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

        ComputationalGraphBuilder graph;
        if (disableAxisCombine) {
            EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_DEVICE_DDR, "gm"), true);
            EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "ci"), true);
            EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"gm"}, {"ci"}, "copy_in", true), true);
        } else {
            EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "assemble_in"), true);
        }
        EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "assembled"), true);
        EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {disableAxisCombine ? "ci" : "assemble_in"}, {"assembled"},
                              "assemble", true),
                  true);
        EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "rhs"), true);
        EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "out"), true);
        EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"assembled", "rhs"}, {"out"}, "consumer", true), true);

        IRBuilder builder;
        auto token = builder.CreateTokenVar(ir::Span::Unknown());
        auto* consumer = graph.GetOp("consumer");
        ASSERT_NE(consumer, nullptr);
        consumer->result_token_ = {token};

        auto* function = graph.GetFunction();
        function->paramConfigs_.combineAxis = true;
        AxisCombine pass;
        EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);

        Operation* inserted = nullptr;
        for (auto& operation : function->Operations(false)) {
            if (operation.GetOpcode() == expectedOpcode) {
                inserted = &operation;
                break;
            }
        }
        ASSERT_NE(inserted, nullptr);
        EXPECT_EQ(inserted->result_token_.front(), token);
        EXPECT_TRUE(consumer->result_token_.empty());
    };

    runCase(false, Opcode::OP_BRCB);
    runCase(true, Opcode::OP_EXPAND);
}

// 看护：广播输入 pad 基数随 dtype 变化（GetPaddingValue 委托 AlignmentUtils::GetLastDimAlignBase）
TEST_F(TestAxisCombine, BroadcastPadBaseFollowsDtype)
{
    auto runCase = [](DataType dtype, int64_t expectBase) {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);

        ComputationalGraphBuilder graph;
        EXPECT_EQ(graph.AddTensor(dtype, {4, 127}, "t1"), true);
        EXPECT_EQ(graph.AddTensor(dtype, {4, 1}, "t2"), true);
        EXPECT_EQ(graph.AddTensor(dtype, {4, 127}, "t3"), true);
        std::vector<Opcode> opCodes{Opcode::OP_ADD};
        std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
        std::vector<std::vector<std::string>> ooperands{{"t3"}};
        std::vector<std::string> opNames{"add"};
        EXPECT_EQ(graph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

        auto* rootFuncPtr = graph.GetFunction();
        rootFuncPtr->paramConfigs_.combineAxis = true;
        AxisCombine pass;
        EXPECT_EQ(pass.RunOnFunction(*rootFuncPtr), SUCCESS);

        int64_t brcbCnt = 0;
        for (const auto& op : rootFuncPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_BRCB) {
                ++brcbCnt;
            }
            if (op.HasAttr(OpAttributeKey::brcbIdx)) {
                auto idx = op.GetIntAttribute(OpAttributeKey::brcbIdx) - 1;
                auto tensor = op.GetIOperands()[idx];
                EXPECT_TRUE(tensor != nullptr);
                EXPECT_EQ(tensor->shape[0], K_4);
                EXPECT_EQ(tensor->shape[1], expectBase);
            }
        }
        EXPECT_EQ(brcbCnt, K_1);
    };
    runCase(DataType::DT_BF16, K_16);
    runCase(DataType::DT_INT8, K_32);
}

// axis_combine.cpp 内的自由函数(外部链接), 补充声明后直接验证
namespace npu {
namespace tile_fwk {
Status AlignedIfNeed(int64_t& currentDim, int64_t padValue);
} // namespace tile_fwk
} // namespace npu

// padBase为0时报ERROR并返回FAILED, 不修改currentDim
// (覆盖axis_combine.cpp "invalid pad base %ld, expected a non-zero positive pad base."日志分支)
TEST_F(TestAxisCombine, AlignedIfNeedRejectsZeroPadBase)
{
    int64_t currentDim = K_8;
    EXPECT_EQ(AlignedIfNeed(currentDim, 0), FAILED);
    EXPECT_EQ(currentDim, K_8);
}

// 正常场景: currentDim按padValue向上取整对齐
TEST_F(TestAxisCombine, AlignedIfNeedPadsCurrentDim)
{
    int64_t currentDim = K_8;
    EXPECT_EQ(AlignedIfNeed(currentDim, K_16), SUCCESS);
    EXPECT_EQ(currentDim, K_16);
}
