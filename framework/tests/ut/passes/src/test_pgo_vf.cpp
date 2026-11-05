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

#include "interface/function/function.h"
#include "interface/plugin/plugin.h"
#include "interface/program/program.h"
#include "interface/utils/common.h"
#include "passes/block_graph_pass/pgo_vf/pgo_vf_api.h"

TEST(PgoVfTest, LoadSharedLibraryPlugin)
{
    using namespace npu::tile_fwk;

    Function function(Program::GetInstance(), "pgo_vf_ut", "pgo_vf_ut", nullptr);
    PluginManager& manager = PluginManager::GetInstance();
    ASSERT_TRUE(manager.AddSharedLibraryPlugin<PluginPassPGOVF>("pgo_vf", "pypto_pass_pgo_vf", kPgoVfVersion,
                                                                "PGOVFRunOnFunction"));
    const std::vector<Status> results = manager.RunPlugin<PluginPassPGOVF>(function);
    manager.ClearPlugin();
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0], SUCCESS);
}
