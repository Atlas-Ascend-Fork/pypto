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
 * \file test_plugin.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <string>

#include "tilefwk/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"

#include "interface/plugin/plugin.h"
#include "interface/utils/string_utils.h"

using namespace npu::tile_fwk;

TEST(PluginTest, Basic)
{
    struct Compute {
        static int ComputeAdd(const std::string& filepath, std::string& source)
        {
            source += "Add" + filepath;
            return 0;
        }
        static int ComputeSub(const std::string& filepath, std::string& source)
        {
            source += "Sub" + filepath;
            return 0;
        }
    };
    PluginManager& manager = PluginManager::GetInstance();

    EXPECT_TRUE(manager.AddPlugin<PluginCodegenSrc>("add", Compute::ComputeAdd));
    EXPECT_FALSE(manager.AddPlugin<PluginCodegenSrc>("add", Compute::ComputeSub));
    EXPECT_TRUE(manager.AddPlugin<PluginCodegenSrc>("sub", Compute::ComputeSub));

    EXPECT_EQ(2, manager.GetPlugin<PluginCodegenSrc>().size());

    std::string code = "2";
    manager.RunPlugin<PluginCodegenSrc>("1", code);
    EXPECT_EQ("2Add1Sub1", code);

    manager.ClearPlugin();
    EXPECT_EQ(0, manager.GetPlugin<PluginCodegenSrc>().size());

    std::string code2 = "2";
    manager.RunPlugin<PluginCodegenSrc>("1", code2);
    EXPECT_EQ("2", code2);
}
