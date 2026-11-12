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
 * \file test_kernel_binary_op_info.cpp
 * \brief UT for ascend_package_load.ini based op info selection
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "machine/runtime/runner/kernel_binary.h"

using namespace npu::tile_fwk::dynamic;

namespace {
class PyptoAicpuKernelConfigTest : public testing::Test {
protected:
    void SetUp() override
    {
        SaveEnv("ASCEND_AICPU_PATH", oldAicpuPath_, hasAicpuPath_);
        SaveEnv("ASCEND_HOME_PATH", oldHomePath_, hasHomePath_);
        root_ = std::filesystem::temp_directory_path() /
                ("pypto_aicpu_cfg_" + std::to_string(getpid()) + "_" + CurrentTestName());
        std::filesystem::remove_all(root_);
        std::filesystem::create_directories(root_ / "conf");
    }

    void TearDown() override
    {
        std::filesystem::remove_all(root_);
        RestoreEnv("ASCEND_AICPU_PATH", oldAicpuPath_, hasAicpuPath_);
        RestoreEnv("ASCEND_HOME_PATH", oldHomePath_, hasHomePath_);
    }

    void WriteIni(const std::string& content)
    {
        std::ofstream ofs(root_ / "conf" / "ascend_package_load.ini");
        ofs << content;
    }

    void WriteTar()
    {
        const auto kernelDir = root_ / "opp" / "built-in" / "op_impl" / "aicpu" / "kernel";
        std::filesystem::create_directories(kernelDir);
        std::ofstream ofs(kernelDir / "pypto_aicpu.tar.gz");
        ofs << "dummy";
    }

    static std::string CurrentTestName()
    {
        const auto* info = testing::UnitTest::GetInstance()->current_test_info();
        return info == nullptr ? "" : std::string(info->name());
    }

    static void SaveEnv(const char* name, std::string& value, bool& hasValue)
    {
        const char* env = std::getenv(name);
        hasValue = (env != nullptr);
        value = hasValue ? env : "";
    }

    static void RestoreEnv(const char* name, const std::string& value, bool hasValue)
    {
        if (hasValue) {
            (void)setenv(name, value.c_str(), 1);
        } else {
            (void)unsetenv(name);
        }
    }

    std::filesystem::path root_;
    std::string oldAicpuPath_;
    std::string oldHomePath_;
    bool hasAicpuPath_ = false;
    bool hasHomePath_ = false;
};
} // namespace

TEST_F(PyptoAicpuKernelConfigTest, ConfiguredWhenNameMatches)
{
    (void)setenv("ASCEND_AICPU_PATH", root_.c_str(), 1);
    WriteTar();
    WriteIni("name:pypto_aicpu.tar.gz\ninstall_path:2\n");
    EXPECT_TRUE(detail::IsPyptoAicpuKernelConfigured());
}

TEST_F(PyptoAicpuKernelConfigTest, NotConfiguredWhenNameDoesNotMatch)
{
    (void)setenv("ASCEND_AICPU_PATH", root_.c_str(), 1);
    WriteTar();
    WriteIni("name:other_kernel.tar.gz\ninstall_path:2\n");
    EXPECT_FALSE(detail::IsPyptoAicpuKernelConfigured());
}

TEST_F(PyptoAicpuKernelConfigTest, NotConfiguredWhenIniMissing)
{
    (void)setenv("ASCEND_AICPU_PATH", root_.c_str(), 1);
    WriteTar();
    EXPECT_FALSE(detail::IsPyptoAicpuKernelConfigured());
}

TEST_F(PyptoAicpuKernelConfigTest, FallsBackToAscendHomePath)
{
    (void)unsetenv("ASCEND_AICPU_PATH");
    (void)setenv("ASCEND_HOME_PATH", root_.c_str(), 1);
    WriteTar();
    WriteIni("name:pypto_aicpu.tar.gz\n");
    EXPECT_TRUE(detail::IsPyptoAicpuKernelConfigured());
}

TEST_F(PyptoAicpuKernelConfigTest, MatchesNameWithLeadingSpace)
{
    (void)setenv("ASCEND_AICPU_PATH", root_.c_str(), 1);
    WriteTar();
    WriteIni("  name:pypto_aicpu.tar.gz\n");
    EXPECT_TRUE(detail::IsPyptoAicpuKernelConfigured());
}

TEST_F(PyptoAicpuKernelConfigTest, NotConfiguredWhenOnlyPathContainsName)
{
    (void)setenv("ASCEND_AICPU_PATH", root_.c_str(), 1);
    WriteTar();
    WriteIni("package:pypto_aicpu.tar.gz\n");
    EXPECT_FALSE(detail::IsPyptoAicpuKernelConfigured());
}

TEST_F(PyptoAicpuKernelConfigTest, NotConfiguredWhenTarMissing)
{
    (void)setenv("ASCEND_AICPU_PATH", root_.c_str(), 1);
    WriteIni("name:pypto_aicpu.tar.gz\n");
    EXPECT_FALSE(detail::IsPyptoAicpuKernelConfigured());
}
