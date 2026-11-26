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
    // Function/loop scope tile overrides serialize as #_scope_*; OP_CONFIG keeps its metadata placeholder.
    const auto dump = TextDump(result);
    EXPECT_NE(dump.find("#_config_scope(Unsupported)"), std::string::npos);
    EXPECT_EQ(dump.find("cube_nbuffer_setting"), std::string::npos);
    EXPECT_EQ(pass::ApplyConfigOps()(applied)->functions_.at("test"), result);
}

namespace {

constexpr const char* kScopeTileAttrs = "#_scope_vec_tile([8, 8]) #_scope_cube_tile([32, 32, 64, 64, 64, 64, 64, 1]) "
                                        "#_scope_conv_tile([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 1]) "
                                        "#_scope_convbp_input_tile([8, 9, 10, 4, 5, 6]) "
                                        "#_scope_convbp_filter_tile([10, 11, 12, 7, 8, 9]) "
                                        "#_scope_dist_tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])";

npu::tile_fwk::ConfigScopePtr LoadLoopScope(const std::string& attrs, std::string& error)
{
    auto stmt = TextLoadStmt("for %i inrange 0, 4, 1 iter { } " + attrs + " { }", error);
    if (stmt == nullptr) {
        return nullptr;
    }
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(stmt);
    if (forStmt == nullptr) {
        return nullptr;
    }
    return forStmt->GetAttr<npu::tile_fwk::ConfigScopePtr>("_config_scope");
}

} // namespace

TEST(ConfigScopeTileTest, LoopScopeDecodesAllTileKinds)
{
    std::string error;
    auto scope = LoadLoopScope(kScopeTileAttrs, error);
    ASSERT_NE(scope, nullptr) << error;
    EXPECT_EQ(scope->GetConfig<std::vector<int64_t>>("vec_tile_shapes"), (std::vector<int64_t>{8, 8}));

    const auto cube = scope->GetConfig<CubeTile>("cube_tile_shapes");
    EXPECT_EQ(cube.m, (std::array<int64_t, 2>{32, 32}));
    EXPECT_EQ(cube.k, (std::array<int64_t, 3>{64, 64, 64}));
    EXPECT_EQ(cube.n, (std::array<int64_t, 2>{64, 64}));
    EXPECT_TRUE(cube.enableSplitK);

    const auto conv = scope->GetConfig<ConvTile>("conv_tile_shapes");
    EXPECT_EQ(conv.tileL1Info.tileHin, 16);
    EXPECT_EQ(conv.tileL1Info.tileBatch, 23);
    EXPECT_EQ(conv.tileL0Info.tileH, 24);
    EXPECT_EQ(conv.tileL0Info.tileN, 27);
    EXPECT_TRUE(conv.setL0Tile);

    const auto convBpInput = scope->GetConfig<ConvBpTile>("convbp_input_tile_shapes");
    EXPECT_EQ(convBpInput.tileL1Info.tileML1, 8);
    EXPECT_EQ(convBpInput.tileL0Info.tileNL0, 6);
    const auto convBpFilter = scope->GetConfig<ConvBpTile>("convbp_filter_tile_shapes");
    EXPECT_EQ(convBpFilter.tileL1Info.tileML1, 10);
    EXPECT_EQ(convBpFilter.tileL0Info.tileKL0, 8);

    const auto dist = scope->GetConfig<DistTile>("dist_tile_shapes");
    EXPECT_EQ(dist.row[0], 1);
    EXPECT_EQ(dist.col[2], 6);
    EXPECT_EQ(dist.rank[2], 9);
    EXPECT_EQ(dist.rankId, 0);
}

TEST(ConfigScopeTileTest, LoopScopeDumpKeepsTileAttrs)
{
    std::string error;
    auto stmt = TextLoadStmt(std::string("for %i inrange 0, 4, 1 iter { } ") + kScopeTileAttrs + " { }", error);
    ASSERT_NE(stmt, nullptr) << error;
    const auto dump = TextDump(stmt);
    EXPECT_NE(dump.find("#_scope_vec_tile([8, 8])"), std::string::npos);
    EXPECT_NE(dump.find("#_scope_cube_tile([32, 32, 64, 64, 64, 64, 64, 1])"), std::string::npos);
    EXPECT_NE(dump.find("#_scope_conv_tile([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 1])"), std::string::npos);
    EXPECT_NE(dump.find("#_scope_convbp_input_tile([8, 9, 10, 4, 5, 6])"), std::string::npos);
    EXPECT_NE(dump.find("#_scope_convbp_filter_tile([10, 11, 12, 7, 8, 9])"), std::string::npos);
    EXPECT_NE(dump.find("#_scope_dist_tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])"), std::string::npos);
}

TEST(ConfigScopeTileTest, FunctionScopeRoundTripsTileAttrs)
{
    const std::string text = std::string("program #entry(f) {\n"
                                         "    function f incast() outcast() ") +
                             kScopeTileAttrs + " #entry(true) #type(Opaque) {\n    }\n}";
    std::string error;
    auto program = TextLoadProgram(text, error);
    ASSERT_NE(program, nullptr) << error;
    auto scope = program->functions_.at("f")->GetAttr<npu::tile_fwk::ConfigScopePtr>("_config_scope");
    ASSERT_NE(scope, nullptr);
    EXPECT_EQ(scope->GetConfig<std::vector<int64_t>>("vec_tile_shapes"), (std::vector<int64_t>{8, 8}));
    EXPECT_EQ(scope->GetConfig<CubeTile>("cube_tile_shapes").m[0], 32);

    const auto dump = TextDump(program);
    EXPECT_NE(dump.find("#_scope_vec_tile([8, 8])"), std::string::npos);
    auto reloaded = TextLoadProgram(dump, error);
    ASSERT_NE(reloaded, nullptr) << error;
    EXPECT_EQ(TextDump(reloaded), dump);
}

TEST(ConfigScopeTileTest, EmptyVecTileIsNotAnOverride)
{
    std::string error;
    auto scope = LoadLoopScope("#_scope_vec_tile([])", error);
    EXPECT_TRUE(error.empty()) << error;
    EXPECT_EQ(scope, nullptr);
}

} // namespace pypto::ir
