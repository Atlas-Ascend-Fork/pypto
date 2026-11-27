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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>

#include "interface/program/program.h"
#include "interface/function/function.h"
#include "passes/statistics/tensor_and_tile_graph_statistic.h"
#include "computational_graph_builder.h"
#include "symbolic_scalar_test_utils.h"

using json = nlohmann::json;
namespace npu::tile_fwk {

Status CalRedundantCopy(Function& function, json& report);

TEST(GraphStatisticsTest, IgnoresCopyInWithSymbolicShapeOrOffset)
{
    Function function(Program::GetInstance(), "symbolic_copy", "symbolic_copy", nullptr);
    ComputationalGraphBuilder graph(&function);
    ASSERT_TRUE(graph.AddTensors(DT_FP32, {16}, {"input", "first", "second", "third"}));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_IN, {"input"}, {"first"}, "static_copy"));
    const auto shape = OpImmediate::Specified({16});
    const auto offset = OpImmediate::Specified({0});
    const std::vector<OpImmediate> symbolic{OpImmediate(CreateTestScalarVar("dimension"))};
    graph.GetOp("static_copy")->SetOpAttribute(std::make_shared<CopyOpAttribute>(offset, MEM_UB, shape, shape));
    json baseline;
    ASSERT_EQ(CalRedundantCopy(function, baseline), SUCCESS);
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_IN, {"input"}, {"second"}, "dynamic_shape"));
    ASSERT_TRUE(graph.AddOp(Opcode::OP_COPY_IN, {"input"}, {"third"}, "dynamic_offset"));
    graph.GetOp("dynamic_shape")->SetOpAttribute(std::make_shared<CopyOpAttribute>(offset, MEM_UB, symbolic, shape));
    graph.GetOp("dynamic_offset")->SetOpAttribute(std::make_shared<CopyOpAttribute>(symbolic, MEM_UB, shape, shape));
    json report;
    ASSERT_EQ(CalRedundantCopy(function, report), SUCCESS);
    EXPECT_EQ(report.at("redundantCopyCount(Bytes)"), baseline.at("redundantCopyCount(Bytes)"));
}

using LogStrategy = std::function<void(Function&)>;
void Strategy_PeakMemory(Function& func)
{
    func.SetFunctionType(FunctionType::DYNAMIC);
    json j;
    CalcOperatorInfo(func, j);
}

void Strategy_CreateDirFailed(Function& func) { HealthCheckTileGraph(func, "/invalid/readonly/path", ""); }

void Strategy_OpenFileFailed(Function& func) { HealthCheckTileGraph(func, "./", ""); }

void Strategy_AllRemainingLogs(Function& func)
{
    HealthCheckTileGraph(func, "./", "test");
    HealthCheckTensorGraph(func, "./", "test");
}

TEST(AllLogsCover, LOG01_PeakMemory)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "test", nullptr);
    Strategy_PeakMemory(*func);
}

TEST(AllLogsCover, LOG02_CreateDirFailed)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "test", nullptr);
    Strategy_CreateDirFailed(*func);
}

TEST(AllLogsCover, LOG03_OpenFileFailed)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "test", nullptr);
    Strategy_OpenFileFailed(*func);
}

TEST(AllLogsCover, LOG04_TO_12_ALL)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "test", "test", nullptr);
    Strategy_AllRemainingLogs(*func);
}
} // namespace npu::tile_fwk
