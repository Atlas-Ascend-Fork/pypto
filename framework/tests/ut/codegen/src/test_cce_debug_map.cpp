/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/codegen.h"
#include "gtest/gtest.h"
#include "interface/configs/config_manager.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace npu::tile_fwk {
namespace {
class CceDebugMapTest : public testing::Test {
protected:
    void SetUp() override
    {
        char path[] = "/tmp/pypto-cce-map-XXXXXX";
        ASSERT_NE(mkdtemp(path), nullptr);
        tempDir_ = path;
    }

    void TearDown() override
    {
        config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, previousDebugMode_);
        if (!tempDir_.empty()) {
            std::filesystem::remove_all(tempDir_);
        }
    }

    const int64_t previousDebugMode_ = config::GetDebugOption<int64_t>(CFG_COMPILE_DBEUG_MODE);
    std::filesystem::path tempDir_;
};

TEST_F(CceDebugMapTest, WritesMapsOnlyWithFullCompileDebug)
{
    const auto cceDir = tempDir_ / "kernel's cce";
    ASSERT_TRUE(std::filesystem::create_directory(cceDir));
    const auto opMap = cceDir / "cce_opmagic_map.json";
    const auto labelMap = cceDir / "cce_semantic_label.json";

    config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, CFG_DEBUG_NONE);
    InvokeCceDebugMapScript(cceDir.string());
    EXPECT_TRUE(std::filesystem::is_empty(cceDir));

    config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, CFG_DEBUG_ALL);
    InvokeCceDebugMapScript(cceDir.string());
    EXPECT_TRUE(std::filesystem::is_regular_file(opMap));
    EXPECT_TRUE(std::filesystem::is_regular_file(labelMap));
}

TEST_F(CceDebugMapTest, LeavesOutputUntouchedForInvalidPaths)
{
    config::SetDebugOption(CFG_COMPILE_DBEUG_MODE, CFG_DEBUG_ALL);
    const auto missingDir = tempDir_ / "missing";
    InvokeCceDebugMapScript(missingDir.string());
    EXPECT_TRUE(std::filesystem::is_empty(tempDir_));

    const auto regularFile = tempDir_ / "not-a-directory";
    std::ofstream(regularFile) << "existing CCE file";
    ASSERT_TRUE(std::filesystem::is_regular_file(regularFile));
    InvokeCceDebugMapScript(regularFile.string());
    std::ifstream input(regularFile);
    std::string contents;
    std::getline(input, contents);
    EXPECT_EQ(contents, "existing CCE file");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(tempDir_), std::filesystem::directory_iterator()), 1);
}
} // namespace
} // namespace npu::tile_fwk
