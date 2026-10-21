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
 * \file fed_prof_manager.h
 * \brief
 */

#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include "machine/runtime/runner/fed_prof_holder.h"

struct DeviceArgs;
struct ToSubMachineConfig;

namespace npu::tile_fwk {

class FedProfManager {
public:
    static FedProfManager& GetInstance();
    FedProfManager();
    ~FedProfManager();
    void InitDeviceArgs(const DeviceArgs& args);
    void AicpuProfDumpThreadStart();
    uint64_t DeviceDataAllocate(FedProfKind kind);
    void PerCoreProfResetData() const;
    void PerCoreProfSync(bool debugEnable);
    int SetDebugEnable(const ToSubMachineConfig& profLevelConfig);

private:
    uint32_t GetCoreCount() const;
    uint64_t PerCoreProfDevAddr() const
    {
        return profList_[FED_PROF_PER_CORE_PROF] != nullptr ? profList_[FED_PROF_PER_CORE_PROF]->GetDevAddr() : 0;
    }
    uint64_t AicpuProfDevAddr() const
    {
        return profList_[FED_PROF_AICPU_PROF] != nullptr ? profList_[FED_PROF_AICPU_PROF]->GetDevAddr() : 0;
    }
    void PerCoreProfReleaseData();
    void AicpuProfReleaseData();
    void PerCoreProfResetMetadata(bool multipleLaunch);

    void PerCoreProfDumpSwimlane();
    void PerCoreProfDumpSwimlaneDraw(const json& perCoreProfObjList);

    void AicpuProfDumpThreadStop();
    void AicpuProfDumpThreadEntry();
    void AicpuProfMaybeDumpSwimlane(bool isLast);
    void AicpuProfDumpSwimlane(bool isLast);
    void AicpuProfDumpSwimlaneDraw(const json& launchObjList, uint32_t currLaunchRound);
    int64_t AicpuProfDumpSwimlaneCheck(bool isLast);

    FedProfDeviceArgs args_;

    std::unique_ptr<FedProfHolderBase> profList_[FED_PROF_KIND_MAX];

    uint32_t lastDumpRound{0};
    bool perCoreProfDataInited_{false};
    std::thread dumpThread_;
    std::atomic<bool> dumpThreadStopFlag_{false};
};
} // namespace npu::tile_fwk
