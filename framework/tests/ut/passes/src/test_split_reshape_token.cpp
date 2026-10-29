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
 * \file test_split_reshape_token.cpp
 * \brief Unit test for split_reshape token (WAW scenario) adaptation.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_utils/pass_operation_utils.h"
#define private public
#define protected public
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"
#undef private
#undef protected
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"

using namespace npu::tile_fwk;

class SplitReshapeTokenTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

static ir::VarPtr MakeToken(const std::string& name)
{
    return std::make_shared<ir::Var>(name, ir::GetTokenType(), ir::Span::Unknown());
}

TEST_F(SplitReshapeTokenTest, NoTokenNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitReshapeTokenNoToken") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitReshapeTokenNoToken");
    ASSERT_NE(func, nullptr);

    SplitReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

TEST_F(SplitReshapeTokenTest, PassRunsWithTokenOnAssembleNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitReshapeTokenWithToken")
    {
        Tensor a = Exp(input);
        Tensor b = Mul(a, Element(DT_FP32, 2.0));
        output = Add(b, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitReshapeTokenWithToken");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    for (auto* op : opList) {
        if (op->GetOpcode() == Opcode::OP_ASSEMBLE && op->result_token_.empty()) {
            auto token = MakeToken("testReshapeToken");
            op->result_token_ = {token};
        }
    }

    SplitReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

TEST_F(SplitReshapeTokenTest, EmptyFunctionNoCrash)
{
    TileShape::Current().SetVecTile({16, 16});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("SplitReshapeTokenEmpty") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_SplitReshapeTokenEmpty");
    ASSERT_NE(func, nullptr);

    SplitReshape pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
}

void CheckAliasWriteIsNotBypassed(Opcode writeOpcode, bool separateVersion)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "WritableReshape", "WritableReshape", nullptr);
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, {32, 16});
    for (int64_t row : {0, 16}) {
        auto tile = builder.CreateTensorVar(DT_FP32, {16, 16});
        auto& assemble = PassOperationUtils::AddOperation(*func, writeOpcode, {tile}, {base});
        assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, std::vector<int64_t>{row, 0}));
    }
    auto alias = builder.CreateTensorVar(DT_FP32, {32, 1, 16});
    auto& reshape = PassOperationUtils::AddOperation(*func, Opcode::OP_RESHAPE, {base}, {alias});
    reshape.SetAttribute("op_attr_isInplace", true);
    auto writtenAlias = separateVersion ? builder.CreateTensorVar(alias->GetRawTensor(), {0, 0, 0}, {32, 1, 16}, {}) :
                                          alias;
    auto patch = builder.CreateTensorVar(DT_FP32, {8, 1, 16});
    auto& write = PassOperationUtils::AddOperation(*func, writeOpcode, {patch}, {writtenAlias});
    write.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, std::vector<int64_t>{8, 0, 0}));
    auto read = builder.CreateTensorVar(DT_FP32, {16, 1, 16});
    auto& view = PassOperationUtils::AddOperation(*func, Opcode::OP_SLICE, {writtenAlias}, {read});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0, 0}));

    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    ASSERT_EQ(pass.CollectCopyOut(*func), SUCCESS);
    ASSERT_EQ(pass.CheckCopyIn(*func), SUCCESS);
    EXPECT_FALSE(view.IsDeleted());
    EXPECT_EQ(view.GetIOperands().front(), writtenAlias);
    EXPECT_FALSE(write.IsDeleted());
}

TEST_F(SplitReshapeTokenTest, ReadingWritableAliasKeepsPartialWrite)
{
    CheckAliasWriteIsNotBypassed(Opcode::OP_CONTRACT, false);
    CheckAliasWriteIsNotBypassed(Opcode::OP_ASSEMBLE, false);
}

TEST_F(SplitReshapeTokenTest, ReadingNewAliasVersionKeepsPartialWrite)
{
    CheckAliasWriteIsNotBypassed(Opcode::OP_CONTRACT, true);
}

TEST_F(SplitReshapeTokenTest, ReadTensorWithNullProducerSkipsSplit)
{
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, {32, 16});
    auto alias = builder.CreateTensorVar(DT_FP32, {32, 1, 16});
    auto read = builder.CreateTensorVar(DT_FP32, {16, 1, 16});
    alias->GetProducers().insert(nullptr);
    ASSERT_EQ(alias->GetProducers().size(), 1);

    SplitReshape pass;
    ASSERT_EQ(pass.Init(), SUCCESS);
    pass.reshapeSources_[alias->GetRawMagic()] = base;
    CheckOutputParam checkOutputParam;
    EXPECT_EQ(pass.CheckReshapeSkip(alias, read, checkOutputParam), WARNING);
}
