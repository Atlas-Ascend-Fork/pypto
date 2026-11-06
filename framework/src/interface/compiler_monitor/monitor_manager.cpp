/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <thread>

#include "tilefwk/pypto_fwk_log.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_util.h"

namespace npu::tile_fwk {
namespace {
constexpr double kMillisecondsPerSecond = 1000.0;
constexpr int kPassDurationPrecision = 3;

std::string FormatPassDuration(double seconds)
{
    if (seconds < 0.0) {
        return "-1";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(kPassDurationPrecision) << seconds * kMillisecondsPerSecond << "ms";
    return oss.str();
}
} // namespace

MonitorManager::~MonitorManager() { Shutdown(); }

MonitorManager& MonitorManager::Instance()
{
    static MonitorManager instance;
    return instance;
}

void MonitorManager::Initialize(bool enable, int totalTimeoutSec, bool passDetailEnable)
{
    std::lock_guard<std::mutex> lock(mutex_);
    this->SetCompilerMonitorOptions(enable, totalTimeoutSec, enable && passDetailEnable);
    if (initialized_) {
        return;
    }
    if (!enable) {
        return;
    }
    impl_ = new MonitorImpl(this);
    currentStage_ = "Prepare";
    totalStart_ = std::chrono::steady_clock::now();
    stageStart_ = totalStart_;
    nextFunctionIndex_ = 1;
    impl_->Start();
    initialized_ = true;
    stageDoing_ = true;
    totalTimeoutEmitted_ = false;
    stageElapsedTotals_.clear();
    passCompileTimings_.clear();
    passElapsedTotals_.clear();
    lastPassDetailFunctionIndex_ = -1;
    lastPassDetailFunctionName_.clear();
    lastPassDetailStrategy_.clear();
    pythonStageEnded_ = false;
    hasSuspendedCodeGen_ = false;
    suspendedCodeGen_ = ActiveStageInfo{};
    stageElapsedTotals_["Prepare"] = 0.0;
    // Mark that Prepare stage has started (use env var for cross-.so communication)
    (void)setenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED", "1", 1);
    impl_->StartMonitoring();
}

void MonitorManager::Shutdown()
{
    MonitorImpl* implToStop = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return;
        }
        if (impl_) {
            implToStop = impl_;
            impl_ = nullptr;
        }
        initialized_ = false;
        stageDoing_ = false;
    }
    if (implToStop) {
        implToStop->Stop();
        delete implToStop;
    }
}

void MonitorManager::MaybeStartTotalClock()
{
    if (currentStage_.empty()) {
        totalStart_ = std::chrono::steady_clock::now();
        stageStart_ = totalStart_;
    }
}

double MonitorManager::GetCurrentStageElapsed(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return 0.0;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    if (stageDoing_ && name == currentStage_) {
        elapsed = std::chrono::duration<double>(now - stageStart_).count();
    }
    return elapsed;
}

void MonitorManager::SetTotalFunctionCount(int n)
{
    if (!enable_) {
        return;
    }
    MonitorImpl* toStart = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Check if Prepare stage was started via Python (env var indicates this)
        bool prepareStarted = (std::getenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED") != nullptr);
        if (!initialized_ && enable_ && !prepareStarted) {
            // First time initialization (no Prepare stage from Python)
            if (!impl_) {
                impl_ = new MonitorImpl(this);
            }
            currentStage_ = "Prepare";
            totalStart_ = std::chrono::steady_clock::now();
            stageStart_ = totalStart_;
            stageElapsedTotals_.clear();
            passCompileTimings_.clear();
            passElapsedTotals_.clear();
            lastPassDetailFunctionIndex_ = -1;
            lastPassDetailFunctionName_.clear();
            lastPassDetailStrategy_.clear();
            totalTimeoutEmitted_ = false;
            pythonStageEnded_ = false;
            hasSuspendedCodeGen_ = false;
            suspendedCodeGen_ = ActiveStageInfo{};
            toStart = impl_;
            initialized_ = true;
        }
        totalFunctionCount_ = n;
        nextFunctionIndex_ = 1;
        currentFunctionIndex_ = 0;
        (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", "0", 1); // 进程内唯一，避免多 .so 多单例
    }
    if (toStart != nullptr) {
        toStart->Start();
    }
}

int MonitorManager::GetAndIncrementNextFunctionIndex()
{
    if (!enable_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int k = nextFunctionIndex_++;
    return k;
}

void MonitorManager::SetCurrentFunctionIndex(int k)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    currentFunctionIndex_ = k;
    std::string val = std::to_string(k);
    (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", val.c_str(), 1);
}

void MonitorManager::SetRootFuncCount(int n)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    rootFuncCount_ = n;
    nextRootFuncIndex_ = 1;
    currentRootFuncIndex_ = 0;
}

int MonitorManager::PrepareNextRootFunc()
{
    if (!enable_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int k = nextRootFuncIndex_++;
    return k;
}

std::string MonitorManager::GetCurrentRootFuncName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentRootFunc_;
}

int MonitorManager::GetRootFuncCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return rootFuncCount_;
}

int MonitorManager::GetCurrentRootFuncIndex() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentRootFuncIndex_;
}

void MonitorManager::BeginHostMachineCompileGroup(int totalSteps)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    hostMachineTotalSteps_ = std::max(1, totalSteps);
    hostMachineNextStep_ = 1;
}

int MonitorManager::AllocHostMachineStepIndex()
{
    if (!enable_) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const int k = hostMachineNextStep_;
    ++hostMachineNextStep_;
    return k;
}

int MonitorManager::GetHostMachineTotalSteps() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::max(1, hostMachineTotalSteps_);
}

void MonitorManager::TryEndPrepareStage()
{
    if (!impl_ || !enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->StopMonitoring();
    if (!initialized_ || pythonStageEnded_) {
        return;
    }
    pythonStageEnded_ = true;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - stageStart_).count();
    stageElapsedTotals_["Prepare"] += elapsed;
    stageDoing_ = false;

    if (currentStage_ == "Prepare" && enable_) {
        double totalElapsedPrepare = std::chrono::duration<double>(std::chrono::steady_clock::now() - totalStart_)
                                         .count();
        std::string msg = "[Compiler Monitor] Stage: " + currentStage_ +
                          "(completed) | Stashed function: " + std::to_string(totalFunctionCount_) +
                          " | Stage elapsed: " + FormatElapsed(elapsed) +
                          " | Total elapsed: " + FormatElapsed(totalElapsedPrepare);
        COMPILER_LOGI(
            "%s. current thread_id:%s pid:%ld ppid:%ld", msg.c_str(),
            []() {
                std::stringstream ss;
                ss << std::this_thread::get_id();
                return ss.str();
            }()
                .c_str(),
            static_cast<long>(getpid()), static_cast<long>(getppid()));
        if (passDetailEnable_) {
            (void)fprintf(stdout, "%s\n", msg.c_str());
            (void)fflush(stdout);
        }
    }
}

void MonitorManager::NotifyCompilationFinished()
{
    MonitorImpl* implToStop = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return;
        }
        PrintCompilationFinished();
        if (impl_) {
            implToStop = impl_;
            impl_ = nullptr;
        }
        initialized_ = false;
        stageDoing_ = false;
    }
    if (implToStop) {
        implToStop->Stop();
        delete implToStop;
    }
}

double MonitorManager::ComputeTotalElapsed() const
{
    auto now = std::chrono::steady_clock::now();
    double totalElapsed = std::chrono::duration<double>(now - totalStart_).count();
    double stageTotal = 0.0;
    for (const auto& kv : stageElapsedTotals_) {
        // HostMachine is nested under the CodeGen MonitorStageScope in DYNAMIC path;
        // skip to avoid double-count against CodeGen wall-clock.
        if (kv.first == STAGE_HOST_MACHINE || kv.first == STAGE_FUNC_TO_BIN) {
            continue;
        }
        stageTotal += kv.second;
    }
    if (stageTotal > totalElapsed) {
        totalElapsed = stageTotal;
    }
    return totalElapsed;
}

std::string MonitorManager::BuildStageElapsedSummaryLocked() const
{
    auto now = std::chrono::steady_clock::now();
    std::map<std::string, double> elapsedTotals(stageElapsedTotals_.begin(), stageElapsedTotals_.end());
    std::map<std::string, bool> activeStages;
    for (const auto& info : activeStages_) {
        if (info.stageName == STAGE_FUNC_TO_BIN && hasSuspendedCodeGen_) {
            continue;
        }
        activeStages[info.stageName] = true;
        elapsedTotals[info.stageName] += std::chrono::duration<double>(now - info.startTime).count();
    }
    if (hasSuspendedCodeGen_ && activeStages.count("CodeGen") == 0) {
        activeStages["CodeGen"] = true;
        elapsedTotals["CodeGen"] += std::chrono::duration<double>(now - suspendedCodeGen_.startTime).count();
    }
    if (stageDoing_ && !currentStage_.empty() && currentStage_ != STAGE_FUNC_TO_BIN &&
        activeStages.count(currentStage_) == 0) {
        activeStages[currentStage_] = true;
        elapsedTotals[currentStage_] += std::chrono::duration<double>(now - stageStart_).count();
    }
    auto funcToBinIt = elapsedTotals.find(STAGE_FUNC_TO_BIN);
    if (funcToBinIt != elapsedTotals.end()) {
        elapsedTotals["CodeGen"] += funcToBinIt->second;
        elapsedTotals.erase(funcToBinIt);
    }
    if (activeStages.erase(STAGE_FUNC_TO_BIN) > 0) {
        activeStages["CodeGen"] = true;
    }

    static const std::vector<std::string> stageOrder = {"Prepare", "Pass", "HostMachine", "CodeGen"};
    std::ostringstream oss;
    bool firstStage = true;
    auto appendStage = [&oss, &firstStage](const std::string& name, double seconds, bool current) {
        if (!firstStage) {
            oss << " | ";
        }
        firstStage = false;
        oss << name << (current ? "(current)" : "") << ": " << FormatElapsed(seconds);
    };
    for (const auto& name : stageOrder) {
        auto it = elapsedTotals.find(name);
        if (it == elapsedTotals.end() && activeStages.count(name) == 0) {
            continue;
        }
        appendStage(name, it == elapsedTotals.end() ? 0.0 : it->second, activeStages.count(name) > 0);
    }
    for (const auto& [name, sec] : elapsedTotals) {
        if (std::find(stageOrder.begin(), stageOrder.end(), name) != stageOrder.end()) {
            continue;
        }
        appendStage(name, sec, activeStages.count(name) > 0);
    }
    return oss.str();
}

std::string MonitorManager::BuildTimeoutWarningLocked(double totalElapsed) const
{
    std::ostringstream oss;
    oss << "[Compiler Monitor] Compile timeout | Threshold: " << totalTimeoutSec_.load()
        << "s | You can adjust the timeout threshold via pypto.set_host_options(compile_timeout=...) | Total elapsed: "
        << FormatElapsed(totalElapsed) << " | Stage elapsed: " << BuildStageElapsedSummaryLocked();
    return oss.str();
}

void MonitorManager::CheckAndEmitTotalTimeout()
{
    if (!enable_) {
        return;
    }
    int thresholdSec = totalTimeoutSec_.load();
    if (thresholdSec <= 0) {
        return;
    }
    std::string warningMsg;
    bool printToScreen = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || totalTimeoutEmitted_) {
            return;
        }
        auto now = std::chrono::steady_clock::now();
        double totalElapsed = std::chrono::duration<double>(now - totalStart_).count();
        if (totalElapsed < static_cast<double>(thresholdSec)) {
            return;
        }
        totalTimeoutEmitted_ = true;
        lastTotalElapsed_ = totalElapsed;
        warningMsg = BuildTimeoutWarningLocked(totalElapsed);
        printToScreen = passDetailEnable_;
    }
    COMPILER_WARN_EVENT("%s", warningMsg.c_str());
    if (printToScreen) {
        (void)fprintf(stdout, "%s\n", warningMsg.c_str());
        (void)fflush(stdout);
    }
}

void MonitorManager::PrintDetailSummary(double totalElapsed)
{
    std::string compilationMsg = "[Compiler Monitor] Compilation finished " + std::to_string(currentFunctionIndex_) +
                                 "/" + std::to_string(totalFunctionCount_ > 0 ? totalFunctionCount_ : 1) +
                                 " | Total functions: " +
                                 std::to_string(totalFunctionCount_ > 0 ? totalFunctionCount_ : 1);
    (void)fprintf(stdout, "%s\n", compilationMsg.c_str());
    (void)fflush(stdout);
    COMPILER_LOGI("%s", compilationMsg.c_str());

    int n = totalFunctionCount_ > 0 ? totalFunctionCount_ : 1;
    std::ostringstream stageMsg;
    for (const auto& [stage, sec] : stageElapsedTotals_) {
        if (stage == "Pass" || stage == "CodeGen" || stage == STAGE_HOST_MACHINE) {
            stageMsg << " " << ("[" + stage + "]:") << std::fixed << std::setprecision(1) << sec << "s"
                     << " ";
        } else {
            stageMsg << " " << ("[" + stage + "]:") << std::fixed << std::setprecision(1) << sec << "s  (sum over " << n
                     << " functions)\n";
        }
    }
    std::string stageTimingMsg = "[Compiler Monitor] Stage timing (aggregated by stage):" + stageMsg.str();
    COMPILER_LOGI("%s", stageTimingMsg.c_str());
    (void)fprintf(stdout, "%s\n", stageTimingMsg.c_str());
    (void)fflush(stdout);

    std::string finalMsg = "[Compiler Monitor] Monitoring stopped | Total elapsed: " + FormatElapsed(totalElapsed);
    COMPILER_LOGI("%s", finalMsg.c_str());
    (void)fprintf(stdout, "%s\n", finalMsg.c_str());
    (void)fflush(stdout);

    lastTotalElapsed_ = totalElapsed;
}

void MonitorManager::PrintCompilationFinished()
{
    if (!enable_) {
        return;
    }
    double totalElapsed = ComputeTotalElapsed();
    lastTotalElapsed_ = totalElapsed;
    if (passDetailEnable_) {
        PrintDetailSummary(totalElapsed);
    }
}

std::string MonitorManager::BuildPassFunctionHeaderLocked(int functionIndex, const std::string& functionName,
                                                          const std::string& strategy) const
{
    int totalFunctions = totalFunctionCount_ > 0 ? totalFunctionCount_ : 1;
    std::ostringstream oss;
    oss << "  |__ [Compiler Monitor] Function: " << functionIndex << "/" << totalFunctions << " | Pass timing detail";
    if (!strategy.empty()) {
        oss << " | Strategy:[" << strategy << "]";
    }
    oss << " | Func:[" << functionName << "]\n";
    return oss.str();
}

std::string MonitorManager::BuildPassProgressLineLocked(const std::string& passIdentifier, size_t passIndex,
                                                        const std::string& status, double elapsedSec,
                                                        bool success) const
{
    std::ostringstream oss;
    oss << "  |__ [Compiler Monitor] [" << std::setw(2) << std::setfill('0') << passIndex << std::setfill(' ') << "] "
        << passIdentifier;
    if (status == "running") {
        oss << " start";
    } else {
        oss << " | Pass elapsed: " << FormatPassDuration(elapsedSec);
    }
    if (!success) {
        oss << " | Result:[FAILED]";
    }
    oss << "\n";
    return oss.str();
}

void MonitorManager::SetCompilerMonitorOptions(bool enable, int totalTimeoutSec, bool passDetailEnable)
{
    enable_ = enable;
    passDetailEnable_ = passDetailEnable;
    totalTimeoutSec_.store((totalTimeoutSec >= 0) ? totalTimeoutSec : 600);
    std::string totalTimeoutStr = std::to_string(totalTimeoutSec_.load());
    (void)setenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC", totalTimeoutStr.c_str(), 1);
}

bool MonitorManager::IsEnabled() const { return enable_.load(); }

bool MonitorManager::IsPassDetailEnabled() const { return passDetailEnable_.load(); }

int MonitorManager::GetTotalTimeoutSec() const { return totalTimeoutSec_.load(); }

std::string MonitorManager::GetCurrentStageName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentStage_;
}

void MonitorManager::SwitchStageReset()
{
    this->SetCurrentFunctionName("");
    this->SetCurrentFuncOpSize(0);
}

std::string MonitorManager::GetCurrentFunctionName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentFunction_;
}

void MonitorManager::SetCurrentFunctionName(const std::string& name)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    currentFunction_ = name;
}

int MonitorManager::GetCurrentFuncOpSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentFuncOpsize_;
}

void MonitorManager::SetCurrentFuncOpSize(int opSize, bool updateActiveStage)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    currentFuncOpsize_ = opSize;
    if (!updateActiveStage) {
        return;
    }
    for (auto& info : activeStages_) {
        if (info.stageName == "Pass" && info.functionIndex == currentFunctionIndex_ &&
            info.functionName == currentFunction_) {
            info.functionOpSize = opSize;
        }
    }
}

void MonitorManager::PrintCurrentTotalElapsed(std::string strTemp)
{
    if (!enable_ || !passDetailEnable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    double totalElapsed = std::chrono::duration<double>(now - totalStart_).count();
    std::string stageFinishMsg = "[Compiler Monitor] " + strTemp + " | Total elapsed: " + FormatElapsed(totalElapsed);
    (void)fprintf(stdout, "%s\n", stageFinishMsg.c_str());
    (void)fflush(stdout);
}

std::chrono::steady_clock::time_point MonitorManager::GetStageStartTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stageStart_;
}

std::chrono::steady_clock::time_point MonitorManager::GetTotalStartTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return totalStart_;
}

int MonitorManager::GetTotalFunctionCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return totalFunctionCount_;
}

int MonitorManager::GetCurrentFunctionIndex() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentFunctionIndex_;
}

std::unordered_map<std::string, double> MonitorManager::GetStageElapsedTotals() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stageElapsedTotals_;
}

void MonitorManager::RecordPassCompileTime(const std::string& strategy, const std::string& passIdentifier,
                                           size_t passIndex, const std::string& functionName, int functionIndex,
                                           int functionOpSize, double elapsedSec, bool success)
{
    std::string progressLine;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_ || !initialized_) {
            return;
        }
        PassCompileTiming timing{functionIndex,  functionName, functionOpSize, strategy,
                                 passIdentifier, passIndex,    elapsedSec,     success};
        passCompileTimings_.push_back(timing);
        passElapsedTotals_[strategy + "::" + passIdentifier] += elapsedSec;
        if (passDetailEnable_ && currentStage_ == STAGE_PASS) {
            progressLine = BuildPassProgressLineLocked(passIdentifier, passIndex, "completed", elapsedSec, success);
        }
    }
    if (!progressLine.empty()) {
        std::lock_guard<std::mutex> printLock(passDetailPrintMutex_);
        (void)fprintf(stdout, "%s", progressLine.c_str());
        (void)fflush(stdout);
        if (success) {
            COMPILER_LOGI("%s", progressLine.c_str());
        } else {
            COMPILER_LOGW("%s", progressLine.c_str());
        }
    }
}

std::vector<PassCompileTiming> MonitorManager::GetPassCompileTimings() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return passCompileTimings_;
}

std::map<std::string, double> MonitorManager::GetPassElapsedTotals() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return passElapsedTotals_;
}

void MonitorManager::StartPassCompile(const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
                                      const std::string& functionName, int functionIndex)
{
    std::string progressLine;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_ || !initialized_) {
            return;
        }
        if (passDetailEnable_ && currentStage_ == STAGE_PASS) {
            if (lastPassDetailFunctionIndex_ != functionIndex || lastPassDetailFunctionName_ != functionName ||
                lastPassDetailStrategy_ != strategy) {
                progressLine += BuildPassFunctionHeaderLocked(functionIndex, functionName, strategy);
                lastPassDetailFunctionIndex_ = functionIndex;
                lastPassDetailFunctionName_ = functionName;
                lastPassDetailStrategy_ = strategy;
            }
            progressLine += BuildPassProgressLineLocked(passIdentifier, passIndex, "running", 0.0, true);
        }
    }
    if (!progressLine.empty()) {
        std::lock_guard<std::mutex> printLock(passDetailPrintMutex_);
        (void)fprintf(stdout, "%s", progressLine.c_str());
        (void)fflush(stdout);
        COMPILER_LOGI("%s", progressLine.c_str());
    }
}

void MonitorManager::StartStage(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                                int rootFuncOpSize)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }
    COMPILER_LOGI("Stage ==[%s]== begin.", name.c_str());

    // Suspend CodeGen while nested FuncToBin / HostMachine runs (avoid dual tick warnings),
    // but keep its startTime so the outer CodeGen EndStage still covers the full wall-clock span.
    if (name == STAGE_FUNC_TO_BIN || name == STAGE_HOST_MACHINE) {
        auto it = std::find_if(activeStages_.begin(), activeStages_.end(),
                               [](const ActiveStageInfo& info) { return info.stageName == "CodeGen"; });
        if (it != activeStages_.end()) {
            suspendedCodeGen_ = *it;
            hasSuspendedCodeGen_ = true;
            activeStages_.erase(it);
        }
    }

    if (rootFuncIndex == -1) {
        impl_->StartMonitoring();
    }
    MaybeStartTotalClock();
    currentStage_ = name;
    stageStart_ = std::chrono::steady_clock::now();
    stageDoing_ = true;

    ActiveStageInfo info;
    info.stageName = name;
    info.startTime = stageStart_;
    info.functionIndex = currentFunctionIndex_;
    info.functionName = currentFunction_;
    info.functionOpSize = currentFuncOpsize_;
    info.rootFuncIndex = (rootFuncIndex < 0) ? currentRootFuncIndex_ : rootFuncIndex;
    info.rootFuncName = (rootFuncIndex < 0) ? currentRootFunc_ : rootFuncName;
    info.rootFuncOpSize = rootFuncOpSize;
    activeStages_.push_back(info);
}

void MonitorManager::EndStage(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                              int rootFuncOpSize)
{
    std::lock_guard<std::mutex> lock(mutex_);
    int actualRootFuncIndex = rootFuncIndex;
    std::string actualRootFuncName = rootFuncName;
    auto stageStartTime = stageStart_;
    int actualFunctionIndex = currentFunctionIndex_;
    std::string actualFunctionName = currentFunction_;
    int actualFunctionOpSize = currentFuncOpsize_;

    auto it = activeStages_.rend();
    if (rootFuncIndex < 0) {
        it = std::find_if(activeStages_.rbegin(), activeStages_.rend(),
                          [&name](const ActiveStageInfo& info) { return info.stageName == name; });
        if (it != activeStages_.rend()) {
            actualRootFuncIndex = it->rootFuncIndex;
            actualRootFuncName = it->rootFuncName;
            stageStartTime = it->startTime;
            actualFunctionIndex = it->functionIndex;
            actualFunctionName = it->functionName;
            actualFunctionOpSize = it->functionOpSize;
        } else if (name == "CodeGen" && hasSuspendedCodeGen_) {
            stageStartTime = suspendedCodeGen_.startTime;
            actualRootFuncIndex = suspendedCodeGen_.rootFuncIndex;
            actualRootFuncName = suspendedCodeGen_.rootFuncName;
            actualFunctionIndex = suspendedCodeGen_.functionIndex;
            actualFunctionName = suspendedCodeGen_.functionName;
            actualFunctionOpSize = suspendedCodeGen_.functionOpSize;
            hasSuspendedCodeGen_ = false;
            suspendedCodeGen_ = ActiveStageInfo{};
        } else {
            actualRootFuncIndex = currentRootFuncIndex_;
            actualRootFuncName = currentRootFunc_;
        }
    } else {
        it = std::find_if(activeStages_.rbegin(), activeStages_.rend(), [&](const ActiveStageInfo& info) {
            return info.stageName == name && info.rootFuncIndex == rootFuncIndex;
        });
        if (it != activeStages_.rend()) {
            stageStartTime = it->startTime;
            actualFunctionIndex = it->functionIndex;
            actualFunctionName = it->functionName;
            actualFunctionOpSize = it->functionOpSize;
        }
    }

    if (it != activeStages_.rend()) {
        activeStages_.erase(std::prev(it.base()));
    }
    EndStageInternal(name, actualRootFuncIndex, actualRootFuncName, stageStartTime, rootFuncIndex, rootFuncOpSize,
                     actualFunctionIndex, actualFunctionName, actualFunctionOpSize);
}

std::string MonitorManager::BuildStageFinishMsg(const std::string& name, int rootFuncIndex,
                                                const std::string& rootFuncName, double elapsed, double totalElapsed,
                                                int functionIndex, const std::string& functionName, int functionOpSize,
                                                int rootFuncOpSize) const
{
    if (name == STAGE_FUNC_TO_BIN) {
        int pw = GetProgressWidth();
        return "[Compiler Monitor] " + PadLabel("Function(parallel): ") +
               PadRight(std::to_string(rootFuncIndex) + "/" + std::to_string(rootFuncCount_), pw) +
               " | Stage: " + PadStageName("CodeGen[" + name + "]") +
               "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
               " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) + " | Func:[" + rootFuncName +
               "] Ops: " + FormatOpCount(rootFuncOpSize);
    }
    if (name == STAGE_HOST_MACHINE) {
        int pw = GetProgressWidth();
        const int denom = std::max(hostMachineTotalSteps_, std::max(1, rootFuncIndex));
        const std::string& hostMachineStage = rootFuncName.empty() ? STAGE_HOST_MACHINE : rootFuncName;
        return "[Compiler Monitor] " + PadLabel("HostMachine: ") +
               PadRight(std::to_string(rootFuncIndex) + "/" + std::to_string(denom), pw) +
               " | Stage: " + PadStageName(hostMachineStage) +
               "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
               " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) +
               " | Ops: " + FormatOpCount(rootFuncOpSize);
    }
    if (name == "CodeGen") {
        return "[Compiler Monitor] Stage: " + name +
               "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
               " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed));
    }
    int pw = GetProgressWidth();
    return "[Compiler Monitor] " + PadLabel("Function: ") +
           PadRight(std::to_string(functionIndex) + "/" + std::to_string(totalFunctionCount_), pw) +
           " | Stage: " + PadStageName(name) + "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
           " | Total elapsed: " + PadElapsed(FormatElapsed(totalElapsed)) + " | Func:[" + functionName +
           "] Ops: " + FormatOpCount(functionOpSize);
}

void MonitorManager::EndStageInternal(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                                      const std::chrono::steady_clock::time_point& startTime, int rootFuncIndexOriginal,
                                      int rootFuncOpSize, int functionIndex, const std::string& functionName,
                                      int functionOpSize)
{
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }
    if (rootFuncIndexOriginal == -1) {
        impl_->StopMonitoring();
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - startTime).count();
    // FuncToBin is a CodeGen sub-stage: covered by CodeGen wall-clock, not listed separately.
    if (name != STAGE_FUNC_TO_BIN) {
        stageElapsedTotals_[name] += elapsed;
    }
    stageDoing_ = false;
    COMPILER_LOGI("Stage ==[%s]== end, sub stage cost %lfs.", name.c_str(),
                  name == STAGE_FUNC_TO_BIN ? elapsed : stageElapsedTotals_[name]);

    double totalElapsed = std::chrono::duration<double>(now - totalStart_).count();
    std::string stageFinishMsg = BuildStageFinishMsg(name, rootFuncIndex, rootFuncName, elapsed, totalElapsed,
                                                     functionIndex, functionName, functionOpSize, rootFuncOpSize);

    if (passDetailEnable_) {
        (void)fprintf(stdout, "%s\n", stageFinishMsg.c_str());
        (void)fflush(stdout);
    }
    COMPILER_LOGI("%s", stageFinishMsg.c_str());
}

std::vector<ActiveStageInfo> MonitorManager::GetActiveStages() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return activeStages_;
}

int MonitorManager::GetProgressWidth() const
{
    auto digits = [](int n) { return static_cast<int>(std::to_string(std::max(n, 1)).size()); };
    int maxDigits = std::max({digits(totalFunctionCount_), digits(rootFuncCount_), digits(hostMachineTotalSteps_)});
    return 2 * maxDigits + 1;
}

double MonitorManager::GetTotalElapsed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return lastTotalElapsed_;
}

} // namespace npu::tile_fwk
