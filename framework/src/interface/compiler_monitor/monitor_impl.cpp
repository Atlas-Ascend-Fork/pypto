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
#include <mutex>
#include <thread>

#include "tilefwk/pypto_fwk_log.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_manager.h"

namespace npu::tile_fwk {
namespace {
constexpr int kMillisecondsPerSecond = 1000;
constexpr int kPassDetailMonitorIntervalMs = 1;
} // namespace

MonitorImpl::MonitorImpl(MonitorManager* manager) : manager_(manager) {}

MonitorImpl::~MonitorImpl() { Stop(); }

void MonitorImpl::Start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_ && thread_->joinable()) {
        return;
    }
    stop_.store(false);
    thread_ = std::make_unique<std::thread>(&MonitorImpl::MonitorLoop, this);
}

void MonitorImpl::Stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
        stageStartFlag_.store(false);
        cv_.notify_all();
    }
    if (thread_ && thread_->joinable()) {
        thread_->join();
    }
    thread_.reset();
}

void MonitorImpl::StartMonitoring()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stageStartFlag_.store(true);
    }
    cv_.notify_all(); // 唤醒等待的线程
}

void MonitorImpl::StopMonitoring()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stageStartFlag_.store(false);
    }
    cv_.notify_all(); // 唤醒等待的线程
}

void MonitorImpl::MonitorLoop()
{
    bool checkEnable = manager_->IsEnabled();

    COMPILER_LOGI("[Compiler Monitor] total_timeout_sec=%d, check_enable=%d", manager_->GetTotalTimeoutSec(),
                  checkEnable);

    int checkIntervalSec = 1;

    while (!stop_.load()) {
        // 检查 start_flag，如果为 false 则等待
        if (!stageStartFlag_.load()) {
            std::unique_lock<std::mutex> lock(mutex_);
            // 等待直到 start_flag 变为 true 或 stop_ 变为 true
            cv_.wait(lock, [this] { return stageStartFlag_.load() || stop_.load(); });
            if (stop_.load()) {
                break;
            }
            lock.unlock();
        }

        manager_->CheckAndEmitTotalTimeout();

        auto waitDuration = std::chrono::milliseconds(checkIntervalSec * kMillisecondsPerSecond);
        if (checkEnable && manager_->GetCurrentStageName() == STAGE_PASS && manager_->IsPassDetailEnabled()) {
            waitDuration = std::chrono::milliseconds(kPassDetailMonitorIntervalMs);
        }
        std::unique_lock<std::mutex> lock(mutex_);

        // 修改等待条件：检查 stop_ 和 stageStartFlag_
        cv_.wait_for(lock, waitDuration, [this] { return stop_.load() || !stageStartFlag_.load(); });
        if (stop_.load()) {
            break;
        }
        lock.unlock();

        // 如果 start_flag 变为 false，则回到循环开头重新等待
        if (!stageStartFlag_.load()) {
            continue;
        }

        manager_->CheckAndEmitTotalTimeout();
    }
}

} // namespace npu::tile_fwk
