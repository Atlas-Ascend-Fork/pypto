/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/npu/cce_split.h"
#include "gtest/gtest.h"
#include "tilefwk/error.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <unistd.h>

namespace npu::tile_fwk {
namespace {
class CceSplitBuild : public testing::Test {
protected:
    void SetUp() override
    {
        char path[] = "/tmp/pypto-split-build-XXXXXX";
        const auto result = mkdtemp(path);
        ASSERT_NE(result, nullptr);
        dir = std::string(result) + "/with space";
        std::filesystem::create_directory(dir);
        Write("part0.cpp", "a");
        Write("part1.cpp", "b");
        Write("parts.h", "header");
    }
    void TearDown() override
    {
        if (!dir.empty()) {
            std::filesystem::remove_all(std::filesystem::path(dir).parent_path());
        }
    }
    void Write(const std::string& name, const std::string& contents) { std::ofstream(dir + "/" + name) << contents; }
    std::string Read(const std::string& name)
    {
        std::ifstream file(dir + "/" + name);
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    std::string Q(const std::string& name) { return QuoteCceBuildArg(dir + "/" + name); }
    std::vector<CompileTaskInfo> Tasks()
    {
        std::vector<CompileTaskInfo> tasks;
        for (int i = 0; i < 2; ++i) {
            const std::string stem = "part" + std::to_string(i);
            CompileTaskInfo task;
            task.inputPath = dir + "/" + stem + ".cpp";
            task.outputPath = dir + "/" + stem + ".o";
            task.dependencies = {dir + "/parts.h"};
            task.compileCmd = "cp -- " + Q(stem + ".cpp") + " " + Q(stem + ".o");
            tasks.push_back(task);
        }
        CompileTaskInfo link;
        link.outputPath = dir + "/leaf.o";
        link.publishFrom = dir + "/leaf.tmp";
        link.dependencies = {dir + "/part0.o", dir + "/part1.o"};
        link.compileCmd = "cat " + Q("part0.o") + " " + Q("part1.o") + " > " + Q("leaf.tmp");
        tasks.push_back(link);
        return tasks;
    }
    int Make(const std::vector<CompileTaskInfo>& tasks)
    {
        Write("Makefile", RenderCceMakefile(tasks));
        const auto command = "make --no-print-directory -j4 -f " + Q("Makefile");
        return std::system(command.c_str());
    }
    std::string dir;
};
} // namespace

TEST_F(CceSplitBuild, ParallelBuildWaitsForAllObjectsAndPublishes)
{
    ASSERT_EQ(Make(Tasks()), 0);
    EXPECT_EQ(Read("leaf.o"), "ab");
    EXPECT_FALSE(std::filesystem::exists(dir + "/leaf.tmp"));
}

TEST_F(CceSplitBuild, CompileFailureDoesNotPublishOldOrPartialResult)
{
    Write("leaf.o", "previous");
    auto tasks = Tasks();
    tasks[0].compileCmd = "false";
    EXPECT_NE(Make(tasks), 0);
    EXPECT_EQ(Read("leaf.o"), "previous");
    EXPECT_FALSE(std::filesystem::exists(dir + "/leaf.tmp"));
}

TEST_F(CceSplitBuild, LinkFailureDoesNotPublish)
{
    Write("leaf.o", "previous");
    auto tasks = Tasks();
    tasks.back().compileCmd = "false";
    EXPECT_NE(Make(tasks), 0);
    EXPECT_EQ(Read("leaf.o"), "previous");
}

TEST_F(CceSplitBuild, CommonHeaderInvalidatesObjects)
{
    auto tasks = Tasks();
    ASSERT_EQ(Make(tasks), 0);
    Write("part0.o", "stale");
    // Avoid depending on filesystem timestamp resolution or sleeps.
    std::filesystem::last_write_time(dir + "/part0.o",
                                     std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
    ASSERT_EQ(Make(tasks), 0);
    EXPECT_EQ(Read("leaf.o"), "ab");
}

TEST_F(CceSplitBuild, OrderIndependentAndOnlyListsCurrentObjects)
{
    auto tasks = Tasks();
    const auto a = RenderCceMakefile(tasks);
    std::swap(tasks[0], tasks[2]);
    EXPECT_EQ(a, RenderCceMakefile(tasks));
    EXPECT_EQ(a.find("*.o"), std::string::npos);
    EXPECT_NE(a.find("parts.h"), std::string::npos);
}

TEST_F(CceSplitBuild, SpecialCharactersRemainLiteral)
{
    const auto originalDir = dir;
    for (const std::string suffix :
         {"a&b", "a#b", "a'b", "a`b", "a\\b", "a$HOME", "a$(id)", "a\\ b", "a\\#b", "a:b", "a%b"}) {
        SCOPED_TRACE(suffix);
        dir = originalDir + "/" + suffix;
        std::filesystem::create_directory(dir);
        Write("part0.cpp", "a");
        Write("part1.cpp", "b");
        Write("parts.h", "header");
        // Exercise the same quoted argument both directly and through Make recipes.
        EXPECT_EQ(std::system(("cp -- " + Q("part0.cpp") + " " + Q("direct.o")).c_str()), 0);
        EXPECT_EQ(Read("direct.o"), "a");
        EXPECT_EQ(Make(Tasks()), 0);
        EXPECT_EQ(Read("leaf.o"), "ab");
        EXPECT_EQ(std::system(("make --no-print-directory -f " + Q("Makefile") + " clean").c_str()), 0);
        EXPECT_FALSE(std::filesystem::exists(dir + "/leaf.o"));
    }
    dir = originalDir;
}

TEST_F(CceSplitBuild, UnsupportedMakeSyntaxUsesFrameworkError)
{
    for (const std::string suffix : {"a;b", "a|b"}) {
        auto tasks = Tasks();
        tasks[0].inputPath += suffix;
        try {
            (void)RenderCceMakefile(tasks);
            FAIL() << "Expected unsupported Makefile path to fail";
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find("CMD_CHECK_FAILED"), std::string::npos);
        }
    }
}

TEST(CceBuildArguments, InvalidPathsUseFrameworkError)
{
    EXPECT_EQ(QuoteCceBuildArg("a b.cpp"), "'a b.cpp'");
    for (const auto& bad : {std::string("a\nb"), std::string("a\rb"), std::string("a\0b", 3)}) {
        try {
            (void)QuoteCceBuildArg(bad);
            FAIL() << "Expected invalid path to fail";
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find("CMD_CHECK_FAILED"), std::string::npos);
        }
    }
}
} // namespace npu::tile_fwk
