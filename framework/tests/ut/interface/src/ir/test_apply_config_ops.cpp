/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file test_apply_config_ops.cpp
 * \brief Configuration replay tests for text-loaded IR.
 */

#include "gtest/gtest.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "ir/transforms/io_text.h"
#include "ir/transforms/passes.h"
#include "interface/configs/config_manager_ng.h"

namespace pypto::ir {
TEST(ApplyConfigOpsTest, TextConfigListsAndFunctionConfigs)
{
    std::string error;
    auto body = TextLoadStmt("{ OP_CONFIG() token() #vec_tile_shapes([8, 16]) "
                             "#pass.cube_nbuffer_setting([[-1, 4]]); ADD() token(); "
                             "OP_CONFIG() token() #pass.cube_nbuffer_setting([[-1, 6]]); }",
                             error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_NE(body, nullptr);
    auto func = std::make_shared<Function>("test", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                           Span::Unknown());
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "test", Span::Unknown());
    auto applied = pass::ApplyConfigOps()(prog);
    auto result = applied->functions_.at("test");
    ASSERT_EQ(result->body_->stmts_.size(), 2);
    auto pending = std::dynamic_pointer_cast<const TensorOpStmt>(result->body_->stmts_[0]);
    ASSERT_NE(pending, nullptr);
    EXPECT_EQ(pending->opcode_, "OP_CONFIG");
    ASSERT_EQ(pending->attrs_.size(), 1);
    EXPECT_EQ(pending->attrs_[0].first, "_config_scope");
    auto config = std::any_cast<npu::tile_fwk::ConfigScopePtr>(pending->attrs_[0].second);
    ASSERT_NE(config, nullptr);
    using IntMap = std::map<int64_t, int64_t>;
    EXPECT_EQ(std::any_cast<IntMap>(config->GetAnyConfig("pass.cube_nbuffer_setting")), (IntMap{{-1, 6}}));
    EXPECT_TRUE(pending->result_.empty());
    EXPECT_TRUE(pending->result_token_.empty());
    EXPECT_TRUE(pending->args_.empty());
    EXPECT_TRUE(pending->tokens_.empty());
    // Text-loaded functions retain the current entry scope for deferred lowering.
    auto entry = result->GetAttr<npu::tile_fwk::ConfigScopePtr>("_config_scope");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry, npu::tile_fwk::ConfigManagerNg::CurrentScope());
    auto op = std::dynamic_pointer_cast<const TensorOpStmt>(result->body_->stmts_[1]);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->opcode_, "ADD");
    bool found = false;
    for (const auto& [key, value] : op->attrs_) {
        if (key == "vec_tile_shapes") {
            EXPECT_EQ(std::any_cast<std::vector<int64_t>>(value), (std::vector<int64_t>{8, 16}));
            found = true;
        }
    }
    EXPECT_TRUE(found);
    // The scope pointer is lowering metadata. Text dump does not serialize its values.
    const auto dump = TextDump(result);
    EXPECT_NE(dump.find("#_config_scope(Unsupported)"), std::string::npos);
    EXPECT_EQ(dump.find("cube_nbuffer_setting"), std::string::npos);
    EXPECT_EQ(pass::ApplyConfigOps()(applied)->functions_.at("test"), result);
}
} // namespace pypto::ir
