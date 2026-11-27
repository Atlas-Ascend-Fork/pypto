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
 * \file test_infer_param_index.cpp
 * \brief Unit test for InferParamIndex pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/error_manager.h"
#include "interface/inner/tilefwk.h"
#include "passes/block_graph_pass/infer_param_index.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

class InferParamIndexTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

// CopyIn 算子输入个数（1）少于输出个数（2）时，coaIndex 不够分配，
// InsertAddr2ValidShapeSpecified 报 "Copyin[%d] does not support fewer inputs (%zu) than outputs (%zu)." 并返回 FAILED
TEST_F(InferParamIndexTest, CopyInWithFewerInputsThanOutputsFails)
{
    ComputationalGraphBuilder graph;
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "gm"));
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "l1a"));
    EXPECT_TRUE(graph.AddTensor(DataType::DT_FP32, {8, 8}, "l1b"));
    EXPECT_TRUE(graph.AddOp(Opcode::OP_COPY_IN, {"gm"}, {"l1a", "l1b"}, "copyin"));
    Operation* op = graph.GetOp("copyin");
    ASSERT_NE(op, nullptr);

    InferParamIndex pass;
    std::map<int, std::vector<SymbolicScalar>> addr2ValidShape;
    std::map<int, std::vector<SymbolicScalar>> addr2ValidShapeSpecified;
    std::map<int, int> magic2Coa;
    ErrorManager::Instance().OutputErrorMessage(true);
    EXPECT_EQ(pass.InsertAddr2ValidShapeSpecified(*op, addr2ValidShape, addr2ValidShapeSpecified, magic2Coa), FAILED);
    std::string errMsg;
    EXPECT_TRUE(ErrorManager::Instance().GetFirstErrorMessage(errMsg));
    EXPECT_NE(errMsg.find("does not support fewer inputs (1) than outputs (2)"), std::string::npos);
    ErrorManager::Instance().OutputErrorMessage(true);
}
