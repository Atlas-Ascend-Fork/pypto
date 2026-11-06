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
 * \file test_monitor_manager.cpp
 * \brief Unit tests for MonitorManager state management, static functions, getters/setters (no sleep)
 */

#include <cstdlib>
#include <string>
#include <map>
#include <thread>
#include <chrono>
#include "gtest/gtest.h"
#define private public
#include "interface/compiler_monitor/monitor_manager.h"
#undef private
#include "monitor_test_fixture.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_util.h"

using namespace npu::tile_fwk;

class TestMonitorManagerState : public MonitorTestFixtureBase {};

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsEnvVars)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 120, false);
    EXPECT_EQ(MonitorManager::Instance().IsEnabled(), true);
    EXPECT_EQ(MonitorManager::Instance().GetTotalTimeoutSec(), 120);
    EXPECT_EQ(MonitorManager::Instance().IsPassDetailEnabled(), false);

    const char* totalEnv = std::getenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC");
    EXPECT_NE(totalEnv, nullptr);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsNegativeTotalTimeout)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, -1, false);
    EXPECT_EQ(MonitorManager::Instance().GetTotalTimeoutSec(), 600);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsPassDetailEnable)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 600, true);
    EXPECT_EQ(MonitorManager::Instance().IsPassDetailEnabled(), true);
}

TEST_F(TestMonitorManagerState, SetCompilerMonitorOptionsDisable)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(true, 600, false);
    EXPECT_EQ(MonitorManager::Instance().IsEnabled(), true);
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600, false);
    EXPECT_EQ(MonitorManager::Instance().IsEnabled(), false);
}

TEST_F(TestMonitorManagerState, InitializeAndShutdownLifecycle)
{
    MonitorManager::Instance().Initialize(true, 600);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "Prepare");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, InitializeNotEnabledNoThread)
{
    MonitorManager::Instance().Initialize(false, 600);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, SetTotalFunctionCountAndIndex)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(10);
    EXPECT_EQ(MonitorManager::Instance().GetTotalFunctionCount(), 10);

    int idx1 = MonitorManager::Instance().GetAndIncrementNextFunctionIndex();
    EXPECT_EQ(idx1, 1);
    int idx2 = MonitorManager::Instance().GetAndIncrementNextFunctionIndex();
    EXPECT_EQ(idx2, 2);

    MonitorManager::Instance().SetCurrentFunctionIndex(5);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentFunctionIndex(), 5);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetAndIncrementNextFunctionIndexDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    int idx = MonitorManager::Instance().GetAndIncrementNextFunctionIndex();
    EXPECT_EQ(idx, 0);
}

TEST_F(TestMonitorManagerState, SetCurrentFunctionIndexDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    MonitorManager::Instance().SetCurrentFunctionIndex(5);
}

TEST_F(TestMonitorManagerState, SetCurrentFunctionNameAndOpSize)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionName("test_func_1");
    EXPECT_EQ(MonitorManager::Instance().GetCurrentFunctionName(), "test_func_1");

    MonitorManager::Instance().SetCurrentFuncOpSize(5000);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentFuncOpSize(), 5000);

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, SetCurrentFuncOpSizeDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    MonitorManager::Instance().SetCurrentFuncOpSize(100);
}

TEST_F(TestMonitorManagerState, SetCurrentFunctionNameDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    MonitorManager::Instance().SetCurrentFunctionName("name");
}

TEST_F(TestMonitorManagerState, SwitchStageReset)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionName("func_before_reset");
    MonitorManager::Instance().SetCurrentFuncOpSize(9999);

    MonitorManager::Instance().SwitchStageReset();

    EXPECT_EQ(MonitorManager::Instance().GetCurrentFuncOpSize(), 0);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, SetCurrentFuncOpSizeUpdateActiveStage)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionIndex(2);
    MonitorManager::Instance().SetCurrentFunctionName("func_2");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().SetCurrentFuncOpSize(12345, true);

    auto stages = MonitorManager::Instance().GetActiveStages();
    bool found = false;
    for (const auto& s : stages) {
        if (s.stageName == "Pass" && s.functionIndex == 2 && s.functionOpSize == 12345) {
            found = true;
        }
    }
    EXPECT_TRUE(found);

    MonitorManager::Instance().EndStage("Pass");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetProgressWidthSingleFunction)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    int pw = MonitorManager::Instance().GetProgressWidth();
    EXPECT_EQ(pw, 3);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetProgressWidthMultipleFunctions)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(100);
    int pw = MonitorManager::Instance().GetProgressWidth();
    EXPECT_GE(pw, 5);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetProgressWidthWithRootFuncs)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(10);
    MonitorManager::Instance().SetRootFuncCount(50);
    int pw = MonitorManager::Instance().GetProgressWidth();
    EXPECT_GE(pw, 5);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetCurrentStageElapsedNotInitialized)
{
    double elapsed = MonitorManager::Instance().GetCurrentStageElapsed("Pass");
    EXPECT_EQ(elapsed, 0.0);
}

TEST_F(TestMonitorManagerState, StartStageAndEndStagePass)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "Pass");
    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages[0].stageName, "Pass");

    MonitorManager::Instance().EndStage("Pass");
    stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 0u);

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Pass"), totals.end());
    EXPECT_GT(totals["Pass"], 0.0);

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, StartStageAndEndStageCodeGen)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);

    MonitorManager::Instance().StartStage("CodeGen");
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "CodeGen");
    MonitorManager::Instance().EndStage("CodeGen");

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("CodeGen"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, StartStageFuncToBinRemovesCodeGen)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetRootFuncCount(1);

    MonitorManager::Instance().StartStage("CodeGen");
    auto stages = MonitorManager::Instance().GetActiveStages();
    EXPECT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages[0].stageName, "CodeGen");

    int idx = MonitorManager::Instance().PrepareNextRootFunc();
    MonitorManager::Instance().StartStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");
    stages = MonitorManager::Instance().GetActiveStages();
    bool hasCodeGen = false;
    bool hasFuncToBin = false;
    for (const auto& s : stages) {
        if (s.stageName == "CodeGen")
            hasCodeGen = true;
        if (s.stageName == STAGE_FUNC_TO_BIN)
            hasFuncToBin = true;
    }
    EXPECT_FALSE(hasCodeGen);
    EXPECT_TRUE(hasFuncToBin);
    EXPECT_TRUE(MonitorManager::Instance().hasSuspendedCodeGen_);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    MonitorManager::Instance().EndStage(STAGE_FUNC_TO_BIN, idx, "root_func_1");
    MonitorManager::Instance().EndStage("CodeGen");

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    ASSERT_NE(totals.find("CodeGen"), totals.end());
    // FuncToBin is folded into CodeGen wall-clock and not listed separately.
    EXPECT_TRUE(totals.find(STAGE_FUNC_TO_BIN) == totals.end());
    EXPECT_GE(totals.at("CodeGen"), 0.15);
    EXPECT_FALSE(MonitorManager::Instance().hasSuspendedCodeGen_);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, StartStageHostMachineRemovesCodeGen)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().BeginHostMachineCompileGroup(3);

    MonitorManager::Instance().StartStage("CodeGen");
    int stepIdx = MonitorManager::Instance().AllocHostMachineStepIndex();
    MonitorManager::Instance().StartStage(STAGE_HOST_MACHINE, stepIdx, "host_step_1");

    auto stages = MonitorManager::Instance().GetActiveStages();
    bool hasCodeGen = false;
    bool hasHostMachine = false;
    for (const auto& s : stages) {
        if (s.stageName == "CodeGen")
            hasCodeGen = true;
        if (s.stageName == STAGE_HOST_MACHINE)
            hasHostMachine = true;
    }
    EXPECT_FALSE(hasCodeGen);
    EXPECT_TRUE(hasHostMachine);

    MonitorManager::Instance().EndStage(STAGE_HOST_MACHINE, stepIdx, "host_step_1");
    MonitorManager::Instance().EndStage("CodeGen");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, TryEndPrepareStage)
{
    MonitorManager::Instance().Initialize(true, 600);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentStageName(), "Prepare");

    MonitorManager::Instance().TryEndPrepareStage();

    auto totals = MonitorManager::Instance().GetStageElapsedTotals();
    EXPECT_NE(totals.find("Prepare"), totals.end());
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, PrintCurrentTotalElapsed)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().PrintCurrentTotalElapsed("Test checkpoint");
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, PrintCurrentTotalElapsedDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    MonitorManager::Instance().PrintCurrentTotalElapsed("Should not print");
}

TEST_F(TestMonitorManagerState, HostMachineStepGroup)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().BeginHostMachineCompileGroup(4);
    EXPECT_EQ(MonitorManager::Instance().GetHostMachineTotalSteps(), 4);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 1);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 2);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 3);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), 4);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, HostMachineStepGroupDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    EXPECT_EQ(MonitorManager::Instance().AllocHostMachineStepIndex(), -1);
}

TEST_F(TestMonitorManagerState, BeginHostMachineCompileGroupDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    MonitorManager::Instance().BeginHostMachineCompileGroup(3);
}

TEST_F(TestMonitorManagerState, GetHostMachineTotalStepsDefault)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().BeginHostMachineCompileGroup(0);
    EXPECT_EQ(MonitorManager::Instance().GetHostMachineTotalSteps(), 1);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, RootFuncTracking)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetRootFuncCount(3);
    EXPECT_EQ(MonitorManager::Instance().GetRootFuncCount(), 3);
    EXPECT_EQ(MonitorManager::Instance().GetCurrentRootFuncIndex(), 0);

    int idx1 = MonitorManager::Instance().PrepareNextRootFunc();
    EXPECT_EQ(idx1, 1);

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, RootFuncTrackingDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    EXPECT_EQ(MonitorManager::Instance().PrepareNextRootFunc(), 0);
}

TEST_F(TestMonitorManagerState, SetRootFuncCountDisabled)
{
    MonitorManager::Instance().SetCompilerMonitorOptions(false, 600);
    MonitorManager::Instance().SetRootFuncCount(5);
}

TEST_F(TestMonitorManagerState, RecordPassCompileTimeAndRetrieve)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartPassCompile("StrategyA", "Pass1", 0, "func_1", 1);
    MonitorManager::Instance().RecordPassCompileTime("StrategyA", "Pass1", 0, "func_1", 1, 5000, 0.5, true);

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 1u);
    EXPECT_EQ(timings[0].strategy, "StrategyA");
    EXPECT_EQ(timings[0].passIdentifier, "Pass1");
    EXPECT_EQ(timings[0].functionIndex, 1);
    EXPECT_EQ(timings[0].functionOpSize, 5000);
    EXPECT_NEAR(timings[0].elapsedSec, 0.5, 0.01);
    EXPECT_EQ(timings[0].success, true);

    auto elapsedTotals = MonitorManager::Instance().GetPassElapsedTotals();
    EXPECT_NE(elapsedTotals.find("StrategyA::Pass1"), elapsedTotals.end());

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, RecordPassCompileTimeNotInitialized)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().NotifyCompilationFinished();
    size_t sizeBefore = MonitorManager::Instance().GetPassCompileTimings().size();
    MonitorManager::Instance().RecordPassCompileTime("S", "P", 0, "f", 1, 100, 1.0, true);
    size_t sizeAfter = MonitorManager::Instance().GetPassCompileTimings().size();
    EXPECT_EQ(sizeAfter, sizeBefore);
}

TEST_F(TestMonitorManagerState, StartPassCompileNotInitialized)
{
    MonitorManager::Instance().StartPassCompile("S", "P", 0, "f", 1);
}

TEST_F(TestMonitorManagerState, MultiplePassCompileRecords)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(2);

    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "f1", 1);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass1", 0, "f1", 1, 1000, 0.3, true);

    MonitorManager::Instance().StartPassCompile("StratA", "Pass2", 1, "f1", 1);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass2", 1, "f1", 1, 1000, 0.7, true);

    MonitorManager::Instance().StartPassCompile("StratB", "Pass1", 0, "f2", 2);
    MonitorManager::Instance().RecordPassCompileTime("StratB", "Pass1", 0, "f2", 2, 2000, 0.4, false);

    auto timings = MonitorManager::Instance().GetPassCompileTimings();
    EXPECT_EQ(timings.size(), 3u);

    auto totals = MonitorManager::Instance().GetPassElapsedTotals();
    EXPECT_NE(totals.find("StratA::Pass1"), totals.end());
    EXPECT_NE(totals.find("StratA::Pass2"), totals.end());
    EXPECT_NE(totals.find("StratB::Pass1"), totals.end());

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, PassDetailEnableMode)
{
    MonitorManager::Instance().Initialize(true, 600, true);
    EXPECT_EQ(MonitorManager::Instance().IsPassDetailEnabled(), true);

    MonitorManager::Instance().SetTotalFunctionCount(2);
    MonitorManager::Instance().SetCurrentFunctionIndex(1);
    MonitorManager::Instance().SetCurrentFunctionName("func_1");

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().StartPassCompile("StratA", "Pass1", 0, "func_1", 1);
    MonitorManager::Instance().RecordPassCompileTime("StratA", "Pass1", 0, "func_1", 1, 1000, 0.1, true);
    MonitorManager::Instance().EndStage("Pass");

    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, GetTotalElapsedBeforeFinish)
{
    MonitorManager::Instance().Initialize(true, 600);
    double elapsed = MonitorManager::Instance().GetTotalElapsed();
    EXPECT_GE(elapsed, 0.0);
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, NotifyCompilationFinishedPrintsSummary)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(3);
    MonitorManager::Instance().SetCurrentFunctionIndex(3);

    MonitorManager::Instance().StartStage("Pass");
    MonitorManager::Instance().EndStage("Pass");

    MonitorManager::Instance().NotifyCompilationFinished();
    double totalElapsed = MonitorManager::Instance().GetTotalElapsed();
    EXPECT_GT(totalElapsed, 0.0);
}

TEST_F(TestMonitorManagerState, NotifyCompilationFinishedNotInitialized)
{
    MonitorManager::Instance().NotifyCompilationFinished();
}

TEST_F(TestMonitorManagerState, ShutdownNotInitialized) { MonitorManager::Instance().Shutdown(); }

TEST_F(TestMonitorManagerState, EndStageNoMatchStillCompletes)
{
    MonitorManager::Instance().Initialize(true, 600);
    MonitorManager::Instance().SetTotalFunctionCount(1);
    MonitorManager::Instance().SetCurrentFunctionIndex(0);

    MonitorManager::Instance().EndStage("NonExistentStage");
    MonitorManager::Instance().NotifyCompilationFinished();
}
