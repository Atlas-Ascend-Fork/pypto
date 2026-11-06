/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace npu::tile_fwk {
const std::string STAGE_PASS = "Pass";
const std::string STAGE_FUNC_TO_BIN = "FuncToBin";
const std::string STAGE_HOST_MACHINE = "HostMachine";

class MonitorImpl;

struct ActiveStageInfo {
    std::string stageName;
    std::chrono::steady_clock::time_point startTime;
    int rootFuncIndex{0};
    std::string rootFuncName;
    int functionIndex{0};
    std::string functionName;
    int functionOpSize{0};
    int rootFuncOpSize{0};
};

struct PassCompileTiming {
    int functionIndex;
    std::string functionName;
    int functionOpSize;
    std::string strategy;
    std::string passIdentifier;
    size_t passIndex;
    double elapsedSec;
    bool success;
};

class MonitorManager {
public:
    static MonitorManager& Instance();

    void Initialize(bool enable, int totalTimeoutSec, bool passDetailEnable = false);
    void Shutdown();

    void StartStage(const std::string& name, int rootFuncIndex = -1, const std::string& rootFuncName = "",
                    int rootFuncOpSize = 0);
    void EndStage(const std::string& name, int rootFuncIndex = -1, const std::string& rootFuncName = "",
                  int rootFuncOpSize = 0);
    double GetCurrentStageElapsed(const std::string& name);

    void SetTotalFunctionCount(int n);
    int GetAndIncrementNextFunctionIndex();
    void SetCurrentFunctionIndex(int k);

    void SwitchStageReset();

    void SetRootFuncCount(int n);
    int PrepareNextRootFunc();
    std::string GetCurrentRootFuncName() const;
    int GetRootFuncCount() const;
    int GetCurrentRootFuncIndex() const;

    // HostMachine sub-stage progress for CompileDyndevFunction:
    // 1) Call BeginHostMachineCompileGroup once with the number of monitored sub-steps (3 or 4).
    // 2) Call AllocHostMachineStepIndex() once before each MonitorStageScope(STAGE_HOST_MACHINE, step, ...).
    // step is 1-based while monitor is enabled, or -1 when disabled.
    void BeginHostMachineCompileGroup(int totalSteps);
    int AllocHostMachineStepIndex();
    int GetHostMachineTotalSteps() const;

    void TryEndPrepareStage();

    void CheckAndEmitTotalTimeout();
    void NotifyCompilationFinished();

    void SetCompilerMonitorOptions(bool enable, int totalTimeoutSec, bool passDetailEnable = false);
    bool IsEnabled() const;
    bool IsPassDetailEnabled() const;
    int GetTotalTimeoutSec() const;
    std::string GetCurrentStageName() const;
    std::chrono::steady_clock::time_point GetStageStartTime() const;
    std::chrono::steady_clock::time_point GetTotalStartTime() const;
    int GetTotalFunctionCount() const;
    int GetCurrentFunctionIndex() const;
    std::unordered_map<std::string, double> GetStageElapsedTotals() const;

    std::string GetCurrentFunctionName() const;
    void SetCurrentFunctionName(const std::string& name);
    int GetCurrentFuncOpSize() const;
    void SetCurrentFuncOpSize(int opSize, bool updateActiveStage = false);
    double GetTotalElapsed() const;
    void PrintCurrentTotalElapsed(std::string strTemp = "");

    std::vector<ActiveStageInfo> GetActiveStages() const;

    int GetProgressWidth() const;
    void StartPassCompile(const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
                          const std::string& functionName, int functionIndex);
    void RecordPassCompileTime(const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
                               const std::string& functionName, int functionIndex, int functionOpSize,
                               double elapsedSec, bool success);
    std::vector<PassCompileTiming> GetPassCompileTimings() const;
    std::map<std::string, double> GetPassElapsedTotals() const;

    MonitorManager() = default;
    ~MonitorManager();
    MonitorManager(const MonitorManager&) = delete;
    MonitorManager& operator=(const MonitorManager&) = delete;

private:
    void MaybeStartTotalClock();
    void PrintCompilationFinished();
    double ComputeTotalElapsed() const;
    void PrintDetailSummary(double totalElapsed);
    std::string BuildTimeoutWarningLocked(double totalElapsed) const;
    std::string BuildStageElapsedSummaryLocked() const;
    void EndStageInternal(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                          const std::chrono::steady_clock::time_point& startTime, int rootFuncIndexOriginal,
                          int rootFuncOpSize, int functionIndex, const std::string& functionName, int functionOpSize);
    std::string BuildStageFinishMsg(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                                    double elapsed, double totalElapsed, int functionIndex,
                                    const std::string& functionName, int functionOpSize, int rootFuncOpSize) const;
    std::string BuildPassFunctionHeaderLocked(int functionIndex, const std::string& functionName,
                                              const std::string& strategy = "") const;
    std::string BuildPassProgressLineLocked(const std::string& passIdentifier, size_t passIndex,
                                            const std::string& status, double elapsedSec, bool success) const;

    mutable std::mutex mutex_;
    mutable std::mutex passDetailPrintMutex_;
    MonitorImpl* impl_{nullptr};
    bool initialized_{false};
    bool pythonStageEnded_{false};

    std::atomic<bool> enable_{false};
    std::atomic<bool> passDetailEnable_{false};
    bool stageDoing_{false};
    bool totalTimeoutEmitted_{false};
    std::atomic<int> totalTimeoutSec_{600};

    std::string currentFunction_;
    std::string currentStage_;
    std::chrono::steady_clock::time_point totalStart_;
    std::chrono::steady_clock::time_point stageStart_;
    std::unordered_map<std::string, double> stageElapsedTotals_;
    std::vector<PassCompileTiming> passCompileTimings_;
    std::map<std::string, double> passElapsedTotals_;
    int lastPassDetailFunctionIndex_{-1};
    std::string lastPassDetailFunctionName_;
    std::string lastPassDetailStrategy_;

    std::vector<ActiveStageInfo> activeStages_;
    // CodeGen is suspended (removed from activeStages_) while nested FuncToBin/HostMachine runs.
    // Keep its startTime so EndStage("CodeGen") still covers the full outer wall-clock span.
    bool hasSuspendedCodeGen_{false};
    ActiveStageInfo suspendedCodeGen_;

    int currentFuncOpsize_{0};
    int totalFunctionCount_{0};
    int currentFunctionIndex_{0};
    int nextFunctionIndex_{1};
    int rootFuncCount_{0};
    int currentRootFuncIndex_{0};
    int nextRootFuncIndex_{1};
    std::string currentRootFunc_;
    double lastTotalElapsed_{0.0};

    int hostMachineTotalSteps_{0};
    int hostMachineNextStep_{1};
};

} // namespace npu::tile_fwk
