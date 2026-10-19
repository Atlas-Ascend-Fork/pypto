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
 * \file fed_prof.h
 * \brief
 */

#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include "interface/machine/device/tilefwk/aikernel_profile.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct DeviceArgs;
struct ToSubMachineConfig;

namespace npu::tile_fwk {

enum class ArchInfo : uint32_t;

struct FedProfDeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint32_t scheCpuNum{0};
    uint64_t deviceId{0};
    uint64_t sharedBuffer{0};
    uint64_t devDfxArgAddr{0};
    ArchInfo archInfo{};

    FedProfDeviceArgs() = default;
    explicit FedProfDeviceArgs(const DeviceArgs& args);
    uint64_t GetBlockNum() const { return nrValidAic * (nrAiv / nrAic + 1); }
};

class FedProf {
public:
    static FedProf& GetInstance();
    FedProf();
    ~FedProf();
    void InitAndStartDumpThread(const DeviceArgs& args);
    uint64_t AicpuProfAllocateData();
    uint64_t PerCoreProfAllocateData(uint32_t aicCount, uint32_t aivCount);
    void PerCoreProfResetData() const;
    void PerCoreProfSync(bool debugEnable);
    int SetDebugEnable(const ToSubMachineConfig& profLevelConfig);

private:
    uint32_t GetCoreCount() const;
    void PerCoreProfReleaseData();
    void PerCoreProfResetMetadata(const uint32_t coreId);

    void PerCoreProfDumpSwimlane();
    void PerCoreProfDumpSwimlaneDraw(const json& perCoreProfObjList);
    void PerCoreProfCopyToHost(uint32_t blockNum, std::vector<FedProfPerCoreProf*>& perCoreProfList,
                               std::vector<std::vector<uint8_t>>& hostBufferList);
    void PerCoreProfClearDevice(uint32_t blockNum, const std::vector<FedProfPerCoreProf*>& perCoreProfList);

    void AicpuProfDumpThreadStart();
    void AicpuProfDumpThreadStop();
    void AicpuProfDumpThreadEntry();
    void AicpuProfCopyToHost(uint32_t currLaunchRound, std::vector<FedProfAicpuProf>& hostBufferList);
    void AicpuProfMaybeDumpSwimlane(bool isLast);
    void AicpuProfDumpSwimlane(uint32_t freq, bool isLast);
    void AicpuProfDumpSwimlaneDraw(const json& launchObjList, uint32_t currLaunchRound);
    int64_t AicpuProfDumpSwimlaneCheck(bool isLast);

    FedProfDeviceArgs args_;

    bool perCoreProfAllocated{false};
    uint64_t perCoreProfDevAddr_{0};
    bool perCoreProfDataInited_{false};

    uint64_t aicpuProfDevAddr_{0};
    bool aicpuProfAllocated_{false};

    uint32_t lastDumpRound{0};
    std::thread dumpThread_;
    std::atomic<bool> dumpThreadStopFlag_{false};
};
} // namespace npu::tile_fwk
