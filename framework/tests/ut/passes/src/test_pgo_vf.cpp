/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_pgo_vf.cpp
 * \brief UT for pgo_vf plugin loading.
 */

#include <gtest/gtest.h>
#include <functional>
#include <stdexcept>
#include <string>

#include "interface/plugin/plugin.h"
#include "interface/utils/common.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/stmt.h"
#include "passes/block_graph_pass/pgo_vf/pgo_vf_api.h"

TEST(PgoVfTest, LoadSharedLibraryPlugin)
{
    using namespace npu::tile_fwk;

    const auto span = pypto::ir::Span::Unknown();
    auto function = std::make_shared<const pypto::ir::Function>(
        "pgo_vf_ut", std::vector<pypto::ir::VarPtr>{}, std::vector<pypto::ir::TypePtr>{},
        std::make_shared<const pypto::ir::ReturnStmt>(std::vector<pypto::ir::ExprPtr>{}, span), span);
    auto program = std::make_shared<const pypto::ir::Program>(std::vector<pypto::ir::FunctionPtr>{function},
                                                              "pgo_vf_ut", span);
    PluginManager& manager = PluginManager::GetInstance();
    ASSERT_TRUE(
        manager.AddSharedLibraryPlugin<PluginPassPGOVF>("pgo_vf", "pypto_pass_pgo_vf", kPgoVfVersion, "PGOVFRun"));
    const auto plugins = manager.GetPlugin<PluginPassPGOVF>();
    ASSERT_EQ(plugins.size(), 1U);
    // Both branches must either run the implementation or report that it is unavailable.
    auto run = [&](const pypto::ir::ProgramPtr& inputProgram, const pypto::ir::FunctionPtr& inputFunction) {
        pypto::ir::ProgramPtr outputProgram;
        pypto::ir::FunctionPtr outputFunction;
        try {
            EXPECT_EQ(plugins.front()->Call(inputProgram, inputFunction, outputProgram, outputFunction), SUCCESS);
            if (inputFunction) {
                EXPECT_NE(outputFunction, nullptr);
            } else {
                EXPECT_NE(outputProgram, nullptr);
            }
            return true;
        } catch (const Error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("ExternalError::NOT_IMPLEMENTED_ERROR"), std::string::npos);
            EXPECT_NE(message.find("PGOVFRun"), std::string::npos);
            EXPECT_EQ(outputProgram, nullptr);
            EXPECT_EQ(outputFunction, nullptr);
            return false;
        }
    };
    const bool functionImplemented = run(nullptr, function);
    const bool programImplemented = run(program, nullptr);
    EXPECT_EQ(functionImplemented, programImplemented);
    manager.ClearPlugin();
}

TEST(PgoVfTest, SharedLibraryLoadUnloadAndReload)
{
    npu::tile_fwk::SharedLibrary library("pypto_pass_pgo_vf", npu::tile_fwk::kPgoVfVersion);
    EXPECT_FALSE(library.Valid());
    EXPECT_EQ(library.GetSymbolAddress("PGOVFRun"), nullptr);

    ASSERT_TRUE(library.Load());
    EXPECT_TRUE(library.Valid());
    using GetVersionEntry = const char* (*)();
    auto getVersion = reinterpret_cast<GetVersionEntry>(library.GetSymbolAddress(PLUGIN_STR(PLUGIN_GET_VERSION_NAME)));
    ASSERT_NE(getVersion, nullptr);
    EXPECT_STREQ(getVersion(), npu::tile_fwk::kPgoVfVersion);
    auto entry = library.GetSymbolAddress("PGOVFRun");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(library.GetSymbolAddress("PGOVFRunMissing"), nullptr);

    ASSERT_TRUE(library.Load());
    EXPECT_EQ(library.GetSymbolAddress("PGOVFRun"), entry);
    library.Unload();
    EXPECT_FALSE(library.Valid());
    EXPECT_EQ(library.GetSymbolAddress("PGOVFRun"), nullptr);
    library.Unload();
    EXPECT_FALSE(library.Valid());

    ASSERT_TRUE(library.Load());
    EXPECT_TRUE(library.Valid());
    EXPECT_NE(library.GetSymbolAddress("PGOVFRun"), nullptr);
}

TEST(PgoVfTest, SharedLibraryRejectsVersionMismatch)
{
    npu::tile_fwk::SharedLibrary library("pypto_pass_pgo_vf", std::string(npu::tile_fwk::kPgoVfVersion) + "-mismatch");

    EXPECT_FALSE(library.Load());
    EXPECT_FALSE(library.Valid());
    EXPECT_EQ(library.GetSymbolAddress("PGOVFRun"), nullptr);
}

#ifdef PGO_VF_SKIP
TEST(PgoVfTest, UnavailableImplementationReturnsFailureDuringUnwind)
{
    using namespace npu::tile_fwk;

    PluginPassPGOVF plugin("pgo_vf_unwind", "pypto_pass_pgo_vf", kPgoVfVersion, "PGOVFRun");
    ASSERT_TRUE(plugin.IsValid());
    const auto span = pypto::ir::Span::Unknown();
    auto function = std::make_shared<const pypto::ir::Function>(
        "pgo_vf_ut", std::vector<pypto::ir::VarPtr>{}, std::vector<pypto::ir::TypePtr>{},
        std::make_shared<const pypto::ir::ReturnStmt>(std::vector<pypto::ir::ExprPtr>{}, span), span);
    auto program = std::make_shared<const pypto::ir::Program>(std::vector<pypto::ir::FunctionPtr>{function},
                                                              "pgo_vf_ut", span);
    pypto::ir::ProgramPtr outputProgram = program;
    pypto::ir::FunctionPtr outputFunction = function;
    Status status = SUCCESS;
    struct RunOnUnwind {
        std::function<void()> run;
        ~RunOnUnwind() { run(); }
    };

    // CHECK preserves an active exception instead of throwing a second one.
    EXPECT_THROW(
        {
            RunOnUnwind guard{[&] { status = plugin.Call(program, nullptr, outputProgram, outputFunction); }};
            throw std::runtime_error("original exception");
        },
        std::runtime_error);
    EXPECT_EQ(status, FAILED);
    EXPECT_EQ(outputProgram, program);
    EXPECT_EQ(outputFunction, function);
}
#endif
