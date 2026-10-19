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
 * \file fed_prof_aicpu_prof.cpp
 * \brief
 */

#include "machine/runtime/runner/fed_prof.h"
#include "machine/runtime/runner/fed_prof_io_json.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "utils/file_utils.h"
#include "interface/configs/config_manager.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/device/distributed/common.h"
#include "machine/utils/checkinject.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace npu::tile_fwk::dynamic {
namespace {
constexpr int DUMP_LEVEL_FOUR = 4;
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
} // namespace

} // namespace npu::tile_fwk::dynamic

namespace npu::tile_fwk {
uint64_t FedProf::AicpuProfAllocateData()
{
    if (aicpuProfAllocated_) {
        return aicpuProfDevAddr_;
    }
    if (GetEnvVar("DUMP_DEVICE_PERF") == "true") {
        auto aicpuDevPtr = DevMallocWithAlignSize(FED_PROF_DEV_TASK_MAX_COUNT * sizeof(FedProfAicpuProf),
                                                  TWO_MB_HUGE_PAGE_FLAGS);
        if (aicpuDevPtr == nullptr) {
            MACHINE_LOGW("Aicpu prof addr malloc failed");
            return 0;
        }
        aicpuProfDevAddr_ = dynamic::PtrToValue(aicpuDevPtr);
        aicpuProfAllocated_ = true;
    }
    return aicpuProfDevAddr_;
}

void FedProf::AicpuProfDumpThreadStart()
{
    if (aicpuProfDevAddr_ == 0) {
        return;
    }
    if (dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(false);
    dumpThread_ = std::thread(&FedProf::AicpuProfDumpThreadEntry, this);
    MACHINE_LOGI("Dump thread started");
}

void FedProf::AicpuProfDumpThreadStop()
{
    if (!dumpThread_.joinable()) {
        return;
    }
    dumpThreadStopFlag_.store(true);
    if (dumpThread_.joinable()) {
        dumpThread_.join();
    }
    MACHINE_LOGD("Dump thread stopped");

    if (aicpuProfDevAddr_ != 0) {
        void* ptr = npu::tile_fwk::dynamic::ValueToPtr(aicpuProfDevAddr_);
        if (ptr != nullptr) {
            RuntimeFree(ptr);
            aicpuProfDevAddr_ = 0;
            aicpuProfAllocated_ = false;
        }
    }
}

void FedProf::AicpuProfDumpThreadEntry()
{
    MACHINE_LOGD("Dump thread starts to dump machine perf trace data");
    int32_t deviceId = static_cast<int32_t>(args_.deviceId);
    if (RuntimeSetDevice(deviceId) != 0) {
        MACHINE_LOGW("Dump perf thread failed to set Device[%d]", deviceId);
    }
    while (!dumpThreadStopFlag_.load()) {
        usleep(10000);
        AicpuProfMaybeDumpSwimlane(false);
    }
    MACHINE_LOGD("Dump thread final dump");
    AicpuProfMaybeDumpSwimlane(true);
}

void FedProf::AicpuProfMaybeDumpSwimlane(bool isLast)
{
    if (GetEnvVar("DUMP_DEVICE_PERF") == "true" && perCoreProfDevAddr_ != 0) {
        uint64_t freq = (args_.archInfo == ArchInfo::DAV_2201) ? dynamic::FREQ_DAV_2201 : dynamic::FREQ_DAV_3510;
        AicpuProfDumpSwimlane(freq, isLast);
    }
}

int64_t FedProf::AicpuProfDumpSwimlaneCheck(bool isLast)
{
    void* devPtr = dynamic::ValueToPtr(perCoreProfDevAddr_);
    size_t dataSize = FED_PROF_PER_CORE_PROF_SIZE;
    std::vector<uint8_t> hostBuffer(dataSize);
    auto memcpyRet = NormalizedRtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RtMemcpyKind::DEVICE_TO_HOST);
    if (memcpyRet != 0) {
        MACHINE_LOGW("aicpu perf header D2H copy failed ret: %d", memcpyRet);
    }
    FedProfPerCoreProf* perCoreProf = reinterpret_cast<FedProfPerCoreProf*>(hostBuffer.data());
    auto currLaunchRound = (perCoreProf->status.currLaunchRound > FED_PROF_DEV_TASK_MAX_COUNT) ?
                               FED_PROF_DEV_TASK_MAX_COUNT :
                               perCoreProf->status.currLaunchRound;
    MACHINE_LOGD("CoreId 0 devAddr: %p, currLaunchRound: %ld", devPtr, currLaunchRound);
    if (currLaunchRound == lastDumpRound) {
        return 0;
    }
    if ((currLaunchRound < 50 || currLaunchRound % 50 != 0) && !isLast) {
        return 0;
    }
    return currLaunchRound;
}

void FedProf::AicpuProfDumpSwimlane(uint32_t freq, bool isLast)
{
    int64_t currLaunchRound = AicpuProfDumpSwimlaneCheck(isLast);
    if (currLaunchRound <= 0) {
        return;
    }
    uint32_t blockNum = args_.GetBlockNum();
    std::vector<FedProfPerCoreProf*> perCoreProfList(blockNum);
    std::vector<std::vector<uint8_t>> hostBufferList(blockNum);
    PerCoreProfCopyToHost(blockNum, perCoreProfList, hostBufferList);

    json launchObjList = json::array();
    std::vector<FedProfAicpuProf> aicpuProfList;
    AicpuProfCopyToHost(currLaunchRound, aicpuProfList);
    dynamic::FedProfLaunchAicpuDumpJson(launchObjList, args_.nrAicpu, args_.scheCpuNum, freq, currLaunchRound,
                                        lastDumpRound, aicpuProfList);
    dynamic::FedProfLaunchAicoreDumpJson(launchObjList, blockNum, args_.nrValidAic, freq, currLaunchRound,
                                         lastDumpRound, perCoreProfList);
    AicpuProfDumpSwimlaneDraw(launchObjList, currLaunchRound);
}

void FedProf::AicpuProfDumpSwimlaneDraw(const json& launchObjList, uint32_t currLaunchRound)
{
    std::string aicpuPerfilePath = config::LogTopFolder() + "/machine_trace_perf_data_" +
                                   std::to_string(lastDumpRound) + ".json";
    if (!SaveFile(aicpuPerfilePath, launchObjList.dump(dynamic::DUMP_LEVEL_FOUR))) {
        MACHINE_LOGW("Construct custom op json failed");
        return;
    }

    // toolkit drawm aicpu preffto
    std::string scriptPath = GetPyptoLibPath() + "/scripts/machine_perf_trace.py";
    std::string cmd = "python3 " + scriptPath + " gen_perfetto " + aicpuPerfilePath + " " +
                      npu::tile_fwk::config::LogTopFolder() + "/machine_runtime_operator_trace_" +
                      std::to_string(lastDumpRound) + ".json " + npu::tile_fwk::config::LogTopFolder() +
                      "/merged_swimlane.json";
    int ret = Checkinject(cmd.c_str(), cmd.size());
    if (ret != 0) {
        MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "Draw swimlane cmd illegal char.");
        return;
    }
    if (system(cmd.c_str()) != 0) {
        MACHINE_LOGW("Failed to execute machine_perf_trace.py, cannot get aicpu perfetto.json.");
    }
    lastDumpRound = currLaunchRound;
    npu::tile_fwk::config::SetRunDataOption(KEY_AICPU_PERF_GRAPH_PATH, npu::tile_fwk::config::GetAbsoluteTopFolder() +
                                                                           "/machine_runtime_operator_trace_" +
                                                                           std::to_string(lastDumpRound) + ".json");
}
} // namespace npu::tile_fwk
