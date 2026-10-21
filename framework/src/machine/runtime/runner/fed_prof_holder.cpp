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
 * \file fed_prof_holder.cpp
 * \brief
 */

#include "machine/runtime/runner/fed_prof_holder.h"
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/runner/fed_prof_io_json.h"
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "machine/device/dynamic/device_common.h"
#include "interface/machine/device/tilefwk/aikernel_profile_aicpu.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
namespace {
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
} // namespace

FedProfDeviceArgs::FedProfDeviceArgs(const DeviceArgs& args)
    : nrAic(args.nrAic),
      nrAiv(args.nrAiv),
      nrAicpu(args.nrAicpu),
      nrValidAic(args.nrValidAic),
      scheCpuNum(args.scheCpuNum),
      deviceId(args.deviceId),
      sharedBuffer(args.sharedBuffer),
      devDfxArgAddr(args.devDfxArgAddr),
      archInfo(args.archInfo)
{}

uint64_t FedProfPerCoreProfHolder::DeviceDataAllocate(const FedProfDeviceArgs& devArgs)
{
    if (IsAllocated()) {
        return GetDevAddr();
    }
    bool debugEnable = !dynamic::DeviceLauncher::IsCaptureMode() && config::IsRuntimeDebugAllEnabled();
    if (!debugEnable && GetEnvVar("DUMP_DEVICE_PERF") != "true" && HostProf::GetInstance().GetHostProfType() != 1) {
        return 0;
    }
    auto ptr = DevMallocWithAlignSize(
        static_cast<uint64_t>(devArgs.nrAic + devArgs.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS) *
            FED_PROF_PER_CORE_PROF_ALIGNED_SIZE,
        TWO_MB_HUGE_PAGE_FLAGS);
    if (ptr == nullptr) {
        MACHINE_LOGE(DevCommonErr::MALLOC_FAILED, "PerfData Malloc failed");
        return 0;
    }
    devAddr_ = dynamic::PtrToValue(ptr);
    allocated_ = true;
    dataCount_ = devArgs.nrAic + devArgs.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS;
    return devAddr_;
}

void FedProfPerCoreProfHolder::DeviceDataClear()
{
    size_t leafTaskCountOffset = offsetof(FedProfPerCoreProf, status) + offsetof(FedProfPerCoreStatus, leafTaskCount);
    for (uint64_t index = 0; index < dataCount_; ++index) {
        void* devPtr = dynamic::ValueToPtr(devAddr_ + index * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE +
                                           leafTaskCountOffset);
        int ret = RuntimeMemset(devPtr, sizeof(int64_t), 0, sizeof(int64_t));
        if (ret != 0) {
            MACHINE_LOGW("task perf clear failed ret: %d, index: %lu", ret, index);
        }
    }
}

void FedProfPerCoreProfHolder::DeviceDataGetBase(std::vector<uint8_t>& hostBuffer,
                                                 std::vector<FedProfDataBase*>& dataList,
                                                 const FedProfDeviceArgs& devArgs, FedProfLaunchArgs launchArgs)
{
    UNUSED(launchArgs);
    uint32_t totalBlockNum = devArgs.nrAic + devArgs.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS;
    uint64_t dataSize = static_cast<uint64_t>(totalBlockNum) * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE;
    hostBuffer.resize(dataSize);
    auto ret = NormalizedRtMemcpy(hostBuffer.data(), dataSize, dynamic::ValueToPtr(devAddr_), dataSize,
                                  RtMemcpyKind::DEVICE_TO_HOST);
    if (ret != 0) {
        MACHINE_LOGW("task perf D2H copy failed ret: %d, block num: %u", ret, totalBlockNum);
    }
    dataList.resize(totalBlockNum);
    for (uint32_t index = 0; index < totalBlockNum; ++index) {
        dataList[index] = reinterpret_cast<FedProfDataBase*>(
            hostBuffer.data() + static_cast<uint64_t>(index) * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE);
    }
}

void FedProfPerCoreProfHolder::DumpJsonBase(json& perCoreProfObjList, const std::vector<uint8_t>& hostBuffer,
                                            const std::vector<FedProfDataBase*>& dataList,
                                            const FedProfDeviceArgs& devArgs, const FedProfLaunchArgs& launchArgs)
{
    UNUSED(hostBuffer, launchArgs);
    uint32_t blockNum = devArgs.GetBlockNum();
    uint32_t aicoreBlockNum = devArgs.nrAic + devArgs.nrAiv;
    uint32_t totalBlockNum = aicoreBlockNum + AICPU_NUM_OF_RUN_AICPU_TASKS;
    for (uint32_t i = 0; i < blockNum; i++) {
        std::string coreType = (i < devArgs.nrValidAic) ? "AIC" : "AIV";
        json perCoreProfObj;
        dynamic::FedProfPerCoreDumpJson(perCoreProfObj, static_cast<FedProfPerCoreProf*>(dataList[i]), i, coreType);
        perCoreProfObjList.push_back(perCoreProfObj);
    }
    for (uint32_t i = aicoreBlockNum; i < totalBlockNum; i++) {
        json perCoreProfObj;
        dynamic::FedProfPerCoreDumpJson(perCoreProfObj, static_cast<FedProfPerCoreProf*>(dataList[i]), i, "AI-CPU");
        perCoreProfObjList.push_back(perCoreProfObj);
    }
}

void FedProfPerCoreProfHolder::DumpJsonAicpuProf(json& launchObjList, const std::vector<uint8_t>& hostBuffer,
                                                 const std::vector<FedProfDataBase*>& dataList,
                                                 const FedProfDeviceArgs& devArgs, const FedProfLaunchArgs& launchArgs)
{
    UNUSED(hostBuffer);
    std::vector<FedProfPerCoreProf*> perCoreProfList(dataList.size());
    for (size_t index = 0; index < dataList.size(); ++index) {
        perCoreProfList[index] = static_cast<FedProfPerCoreProf*>(dataList[index]);
    }
    uint32_t freq = (devArgs.archInfo == ArchInfo::DAV_2201) ? dynamic::FREQ_DAV_2201 : dynamic::FREQ_DAV_3510;
    dynamic::FedProfLaunchAicoreDumpJson(launchObjList, devArgs.GetBlockNum(), devArgs.nrValidAic, freq,
                                         launchArgs.launchRoundEnd, launchArgs.launchRoundBegin, perCoreProfList);
}

uint64_t FedProfAicpuProfHolder::DeviceDataAllocate(const FedProfDeviceArgs&)
{
    if (IsAllocated()) {
        return GetDevAddr();
    }
    if (GetEnvVar("DUMP_DEVICE_PERF") != "true") {
        return 0;
    }
    auto aicpuDevPtr = DevMallocWithAlignSize(FED_PROF_DEV_TASK_MAX_COUNT * sizeof(FedProfAicpuProf),
                                              TWO_MB_HUGE_PAGE_FLAGS);
    if (aicpuDevPtr == nullptr) {
        MACHINE_LOGW("Aicpu prof addr malloc failed");
        return 0;
    }
    devAddr_ = dynamic::PtrToValue(aicpuDevPtr);
    allocated_ = true;
    return devAddr_;
}

void FedProfAicpuProfHolder::DeviceDataGetBase(std::vector<uint8_t>& hostBuffer,
                                               std::vector<FedProfDataBase*>& dataList,
                                               const FedProfDeviceArgs& devArgs, FedProfLaunchArgs launchArgs)
{
    UNUSED(devArgs);
    uint32_t roundNum = (launchArgs.launchRoundEnd > launchArgs.launchRoundBegin) ?
                            launchArgs.launchRoundEnd - launchArgs.launchRoundBegin :
                            0;
    hostBuffer.resize(static_cast<uint64_t>(roundNum) * sizeof(FedProfAicpuProf));
    dataList.resize(roundNum);
    for (uint32_t index = 0; index < roundNum; ++index) {
        dataList[index] = reinterpret_cast<FedProfDataBase*>(hostBuffer.data() +
                                                             static_cast<uint64_t>(index) * sizeof(FedProfAicpuProf));
    }
    if (roundNum == 0) {
        return;
    }
    uint64_t offset = static_cast<uint64_t>(launchArgs.launchRoundBegin) * sizeof(FedProfAicpuProf);
    uint64_t dataSize = static_cast<uint64_t>(roundNum) * sizeof(FedProfAicpuProf);
    auto ret = NormalizedRtMemcpy(hostBuffer.data(), dataSize, dynamic::ValueToPtr(devAddr_ + offset), dataSize,
                                  RtMemcpyKind::DEVICE_TO_HOST);
    if (ret != 0) {
        MACHINE_LOGW("launch prof D2H copy failed ret: %d, turn: %u~%u", ret, launchArgs.launchRoundBegin,
                     launchArgs.launchRoundEnd);
    }
}

void FedProfAicpuProfHolder::DumpJsonBase(json& launchObjList, const std::vector<uint8_t>& hostBuffer,
                                          const std::vector<FedProfDataBase*>& dataList,
                                          const FedProfDeviceArgs& devArgs, const FedProfLaunchArgs& launchArgs)
{
    UNUSED(hostBuffer);
    std::vector<FedProfAicpuProf*> aicpuProfList(dataList.size());
    for (size_t index = 0; index < dataList.size(); ++index) {
        aicpuProfList[index] = static_cast<FedProfAicpuProf*>(dataList[index]);
    }
    uint32_t freq = (devArgs.archInfo == ArchInfo::DAV_2201) ? dynamic::FREQ_DAV_2201 : dynamic::FREQ_DAV_3510;
    dynamic::FedProfLaunchAicpuDumpJson(launchObjList, devArgs.nrAicpu, devArgs.scheCpuNum, freq,
                                        launchArgs.launchRoundEnd, launchArgs.launchRoundBegin, aicpuProfList);
}

void FedProfAicpuProfHolder::DeviceDataClear()
{
    if (devAddr_ == 0) {
        return;
    }
    size_t devTaskPerfsOffset = offsetof(FedProfAicpuProf, devTaskPerfs);
    uint64_t dataSize = sizeof(FedProfAicpuProf) - devTaskPerfsOffset;
    for (uint32_t round = 0; round < FED_PROF_DEV_TASK_MAX_COUNT; ++round) {
        void* devPtr = dynamic::ValueToPtr(devAddr_ + static_cast<uint64_t>(round) * sizeof(FedProfAicpuProf) +
                                           devTaskPerfsOffset);
        int ret = RuntimeMemset(devPtr, dataSize, 0, dataSize);
        if (ret != 0) {
            MACHINE_LOGW("aicpu prof clear failed ret: %d, turn: %u", ret, round);
        }
    }
}

void FedProfHolderBase::DeviceDataRelease()
{
    if (devAddr_ != 0) {
        RuntimeFree(dynamic::ValueToPtr(devAddr_));
        devAddr_ = 0;
        allocated_ = false;
    }
}

void FedProfHolderBase::DeviceDataClear() {}

} // namespace npu::tile_fwk
