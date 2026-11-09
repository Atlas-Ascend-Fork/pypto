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
 * \file test_monitor_watchdog.cpp
 * \brief Unit tests for compiler monitor total-timeout watchdog behavior.
 */

#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

#define private public
#include "interface/compiler_monitor/monitor_manager.h"
#undef private
#include "gtest/gtest.h"
#include "tilefwk/pypto_fwk_log.h"
#include "monitor_test_fixture.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "utils/host_log/log_manager.h"

using namespace npu::tile_fwk;

namespace {
std::string FindLatestHostLogFile()
{
    const std::string dir = LogManager::Instance().GetHostLogDir();
    DIR* dirPtr = opendir(dir.c_str());
    if (dirPtr == nullptr) {
        return "";
    }
    const std::string prefix = "pypto-log-" + std::to_string(getpid()) + "_";
    std::string latest;
    struct dirent* entry = nullptr;
    while ((entry = readdir(dirPtr)) != nullptr) {
        std::string name = entry->d_name;
        if (name.rfind(prefix, 0) == 0 && name.size() > 4 && name.substr(name.size() - 4) == ".log" && name > latest) {
            latest = name;
        }
    }
    closedir(dirPtr);
    return latest.empty() ? "" : dir + "/" + latest;
}

std::string ReadFileContent(const std::string& path)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return "";
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string ReadLogBetweenMarkers(const std::string& beginMarker, const std::string& endMarker)
{
    const std::string content = ReadFileContent(FindLatestHostLogFile());
    auto beginPos = content.rfind(beginMarker);
    if (beginPos == std::string::npos) {
        return "";
    }
    beginPos = content.find('\n', beginPos);
    if (beginPos == std::string::npos) {
        return "";
    }
    auto endPos = content.find(endMarker, beginPos);
    if (endPos == std::string::npos) {
        return content.substr(beginPos);
    }
    return content.substr(beginPos, endPos - beginPos);
}
} // namespace

class TestMonitorWatchdog : public MonitorTestFixtureBase {};

TEST_F(TestMonitorWatchdog, PassDetailEnableFlag)
{
    MonitorManager::Instance().Initialize(true, 600, false);
    EXPECT_FALSE(MonitorManager::Instance().IsPassDetailEnabled());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 600, true);
    EXPECT_TRUE(MonitorManager::Instance().IsPassDetailEnabled());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(false, 600, false);
    EXPECT_FALSE(MonitorManager::Instance().IsEnabled());
}

TEST_F(TestMonitorWatchdog, TotalTimeoutFlagClearedOnReinit)
{
    MonitorManager::Instance().Initialize(true, 600, false);
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        MonitorManager::Instance().totalTimeoutEmitted_ = true;
    }
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 600, false);
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        EXPECT_FALSE(MonitorManager::Instance().totalTimeoutEmitted_);
    }
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorWatchdog, PrintCurrentTotalElapsedNoCrash)
{
    MonitorManager::Instance().Initialize(true, 600, false);
    MonitorManager::Instance().PrintCurrentTotalElapsed("test");
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 600, true);
    MonitorManager::Instance().PrintCurrentTotalElapsed("test");
}

TEST_F(TestMonitorWatchdog, TryEndPrepareStageNoCrash)
{
    MonitorManager::Instance().Initialize(true, 600, false);
    MonitorManager::Instance().TryEndPrepareStage();
    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Prepare"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 600, true);
    MonitorManager::Instance().TryEndPrepareStage();
}

TEST_F(TestMonitorWatchdog, EndStageRecordsElapsed)
{
    MonitorManager::Instance().Initialize(true, 600, false);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    MonitorManager::Instance().StartStage("CodeGen");
    MonitorManager::Instance().EndStage("CodeGen");
    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("CodeGen"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();

    MonitorManager::Instance().Initialize(true, 600, true);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    MonitorManager::Instance().StartStage("CodeGen");
    MonitorManager::Instance().EndStage("CodeGen");
}

TEST_F(TestMonitorWatchdog, DetailModeTotalTimeoutViaTick)
{
    MonitorManager::Instance().Initialize(true, 1, true);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");

    MonitorManager::Instance().StartStage("CodeGen");
    std::this_thread::sleep_for(std::chrono::milliseconds(2100));
    MonitorManager::Instance().EndStage("CodeGen");
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        EXPECT_TRUE(MonitorManager::Instance().totalTimeoutEmitted_);
    }
}

TEST_F(TestMonitorWatchdog, CheckAndEmitTotalTimeoutOnce)
{
    MonitorManager::Instance().Initialize(true, 10, false);
    MonitorManager::Instance().impl_->StopMonitoring();
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        MonitorManager::Instance().stageElapsedTotals_["Prepare"] = 1.0;
        MonitorManager::Instance().stageElapsedTotals_["Pass"] = 1.0;
        MonitorManager::Instance().totalStart_ = std::chrono::steady_clock::now() - std::chrono::seconds(12);
    }

    MonitorManager::Instance().CheckAndEmitTotalTimeout();
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        EXPECT_TRUE(MonitorManager::Instance().totalTimeoutEmitted_);
    }
    EXPECT_NEAR(MonitorManager::Instance().GetTotalElapsed(), 12.0, 0.05);

    double recordedElapsed = MonitorManager::Instance().GetTotalElapsed();
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        MonitorManager::Instance().stageElapsedTotals_["CodeGen"] = 8.0;
    }
    MonitorManager::Instance().CheckAndEmitTotalTimeout();
    EXPECT_DOUBLE_EQ(MonitorManager::Instance().GetTotalElapsed(), recordedElapsed);
}

TEST_F(TestMonitorWatchdog, DisabledModeIgnoresTotalTimeout)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 10, false);
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        MonitorManager::Instance().totalTimeoutEmitted_ = false;
        MonitorManager::Instance().totalStart_ = std::chrono::steady_clock::now() - std::chrono::seconds(100);
    }
    MonitorManager::Instance().CheckAndEmitTotalTimeout();
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        EXPECT_FALSE(MonitorManager::Instance().totalTimeoutEmitted_);
    }
}

TEST_F(TestMonitorWatchdog, TimeoutWarningListsCompletedAndCurrentStagesOnly)
{
    MonitorManager::Instance().Initialize(true, 10, false);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);
    MonitorManager::Instance().SetCurrentFunctionName("func_0");
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        MonitorManager::Instance().stageElapsedTotals_["Prepare"] = 1.0;
        MonitorManager::Instance().stageElapsedTotals_["Pass"] = 2.0;
    }

    MonitorManager::Instance().StartStage("HostMachine");
    MonitorManager::Instance().impl_->StopMonitoring();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        std::string summary = MonitorManager::Instance().BuildStageElapsedSummaryLocked();
        EXPECT_NE(summary.find("Prepare"), std::string::npos);
        EXPECT_NE(summary.find("Pass"), std::string::npos);
        EXPECT_NE(summary.find("HostMachine(current)"), std::string::npos);
        EXPECT_EQ(summary.find("CodeGen"), std::string::npos);

        std::string warning = MonitorManager::Instance().BuildTimeoutWarningLocked(12.0);
        EXPECT_NE(warning.find("Threshold: 10s"), std::string::npos);
        EXPECT_NE(warning.find("pypto.set_host_options(compile_timeout=...)"), std::string::npos);
    }

    MonitorManager::Instance().EndStage("HostMachine");
}

TEST_F(TestMonitorWatchdog, DisabledModeWritesNoMonitorLogToFile)
{
    MonitorManager::Instance().Initialize(false, 600, false);
    COMPILER_EVENT("MONITOR_LOG_MARK_BEGIN_DISABLED");
    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
    COMPILER_EVENT("MONITOR_LOG_MARK_END_DISABLED");

    const std::string segment = ReadLogBetweenMarkers("MONITOR_LOG_MARK_BEGIN_DISABLED",
                                                      "MONITOR_LOG_MARK_END_DISABLED");
    EXPECT_EQ(segment.find("[Compiler Monitor]"), std::string::npos);
    EXPECT_EQ(segment.find("Compile timeout"), std::string::npos);
}

TEST_F(TestMonitorWatchdog, ExistingShortCaseWritesNoTimeoutWarningToFile)
{
    COMPILER_EVENT("MONITOR_LOG_MARK_BEGIN_NO_TIMEOUT");
    MonitorManager::Instance().Initialize(true, 600, false);
    MonitorManager::Instance().impl_->StopMonitoring();
    MonitorManager::Instance().CheckAndEmitTotalTimeout();
    COMPILER_EVENT("MONITOR_LOG_MARK_END_NO_TIMEOUT");
    MonitorManager::Instance().NotifyCompilationFinished();

    const std::string segment = ReadLogBetweenMarkers("MONITOR_LOG_MARK_BEGIN_NO_TIMEOUT",
                                                      "MONITOR_LOG_MARK_END_NO_TIMEOUT");
    EXPECT_EQ(segment.find("Threshold:"), std::string::npos);
    EXPECT_EQ(segment.find("Compile timeout"), std::string::npos);
}

TEST_F(TestMonitorWatchdog, DetailModeInitWritesNoThresholdBannerToFile)
{
    COMPILER_EVENT("MONITOR_LOG_MARK_BEGIN_NO_BANNER_MODE2");
    MonitorManager::Instance().Initialize(true, 600, true);
    MonitorManager::Instance().impl_->StopMonitoring();
    COMPILER_EVENT("MONITOR_LOG_MARK_END_NO_BANNER_MODE2");
    MonitorManager::Instance().NotifyCompilationFinished();

    const std::string segment = ReadLogBetweenMarkers("MONITOR_LOG_MARK_BEGIN_NO_BANNER_MODE2",
                                                      "MONITOR_LOG_MARK_END_NO_BANNER_MODE2");
    EXPECT_EQ(segment.find("Threshold:"), std::string::npos);
}

TEST_F(TestMonitorWatchdog, Mode1TimeoutWritesNoWarningToScreen)
{
    MonitorManager::Instance().Initialize(true, 1, false);
    MonitorManager::Instance().impl_->StopMonitoring();
    {
        std::lock_guard<std::mutex> lock(MonitorManager::Instance().mutex_);
        MonitorManager::Instance().totalTimeoutEmitted_ = false;
        MonitorManager::Instance().totalStart_ = std::chrono::steady_clock::now() - std::chrono::seconds(2);
        MonitorManager::Instance().stageStart_ = MonitorManager::Instance().totalStart_;
    }
    testing::internal::CaptureStdout();
    MonitorManager::Instance().CheckAndEmitTotalTimeout();
    const std::string screen = testing::internal::GetCapturedStdout();
    MonitorManager::Instance().NotifyCompilationFinished();

    EXPECT_EQ(screen.find("Compile timeout"), std::string::npos);
}
