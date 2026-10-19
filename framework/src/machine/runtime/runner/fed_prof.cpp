/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fed_prof.cpp
 * \brief
 */

#include "machine/runtime/runner/fed_prof.h"
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
#include "machine/utils/checkinject.h"
#include "utils/file_utils.h"
#include "nlohmann/json.hpp"

namespace npu::tile_fwk {
namespace {
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
constexpr int DUMP_LEVEL_FOUR = 4;
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

FedProf& FedProf::GetInstance()
{
    static FedProf instance;
    return instance;
}

FedProf::FedProf() : args_() {}
FedProf::~FedProf()
{
    MACHINE_LOGD("Start to clean up profile data.");
    AicpuProfDumpThreadStop();
    PerCoreProfReleaseData();
}

void FedProf::InitAndStartDumpThread(const DeviceArgs& args)
{
    args_ = FedProfDeviceArgs(args);
    AicpuProfDumpThreadStart();
}

uint32_t FedProf::GetCoreCount() const { return args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; }

uint64_t FedProf::PerCoreProfAllocateData(uint32_t aicCount, uint32_t aivCount)
{
    if (perCoreProfAllocated) {
        return perCoreProfDevAddr_;
    }
    bool debugEnable = !dynamic::DeviceLauncher::IsCaptureMode() && config::IsRuntimeDebugAllEnabled();
    if (!debugEnable && GetEnvVar("DUMP_DEVICE_PERF") != "true" && HostProf::GetInstance().GetHostProfType() != 1) {
        return 0;
    }
    // init perf data
    auto ptr = DevMallocWithAlignSize(
        static_cast<uint64_t>(aicCount + aivCount + AICPU_NUM_OF_RUN_AICPU_TASKS) * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE,
        TWO_MB_HUGE_PAGE_FLAGS);
    if (ptr == nullptr) {
        MACHINE_LOGE(DevCommonErr::MALLOC_FAILED, "PerfData Malloc failed");
        return 0;
    }
    perCoreProfDevAddr_ = dynamic::PtrToValue(ptr);
    perCoreProfAllocated = true;
    return perCoreProfDevAddr_;
}

void FedProf::PerCoreProfResetData() const
{
    for (uint32_t i = 0; i < GetCoreCount(); i++) {
        int rc = RuntimeMemset(dynamic::ValueToPtr(perCoreProfDevAddr_ + i * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE),
                               FED_PROF_PER_CORE_PROF_SIZE, 0, FED_PROF_PER_CORE_PROF_SIZE);
        if (rc != 0) {
            MACHINE_LOGW("CoreId %u, RuntimeMemset failed, rc: %d", i, rc);
        }
    }
}

void FedProf::PerCoreProfReleaseData()
{
    if (perCoreProfDevAddr_ != 0) {
        RuntimeFree(dynamic::ValueToPtr(perCoreProfDevAddr_));
        perCoreProfDevAddr_ = 0;
    }
    perCoreProfAllocated = false;
}

void FedProf::PerCoreProfResetMetadata(const uint32_t coreId)
{
    if (perCoreProfDevAddr_ == 0 || coreId >= GetCoreCount()) {
        return;
    }
    if (aicpuProfDevAddr_ != 0) {
        if (!perCoreProfDataInited_) {
            RuntimeMemset(dynamic::ValueToPtr(perCoreProfDevAddr_ + coreId * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE),
                          sizeof(FedProfPerCoreProf), 0, sizeof(FedProfPerCoreProf));
            perCoreProfDataInited_ = true;
        }
    } else {
        RuntimeMemset(dynamic::ValueToPtr(perCoreProfDevAddr_ + coreId * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE),
                      sizeof(FedProfPerCoreProf), 0, sizeof(FedProfPerCoreProf));
    }
}

void FedProf::PerCoreProfSync(bool debugEnable)
{
    if (debugEnable) {
        // 多轮控核，nrValidAic和scheCpuNum需实时刷新，否则泳道图会出错
        args_.nrValidAic = GetCfgBlockdim();
        args_.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args_.nrValidAic, args_.nrAicpu, args_.archInfo);
        PerCoreProfDumpSwimlane();
    }
}

int FedProf::SetDebugEnable(const ToSubMachineConfig& profLevelConfig)
{
    if (PerCoreProfAllocateData(args_.nrAic, args_.nrAiv) == 0) {
        return static_cast<int>(DevCommonErr::MALLOC_FAILED);
    }
    int32_t ret = 0;
    for (uint32_t i = 0; i < GetCoreCount(); i++) {
        PerCoreProfResetMetadata(i);
    }
    if (IsAicoreResolveEnabled()) {
        DevDfxArgs devDfxarg;
        ret = RuntimeMemcpyDirect(&devDfxarg, sizeof(DevDfxArgs), reinterpret_cast<uint8_t*>(args_.devDfxArgAddr),
                                  sizeof(DevDfxArgs), RtMemcpyKind::DEVICE_TO_HOST);
        if (ret != RT_SUCCESS) {
            MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED,
                         "RuntimeMemcpyDirect DEVICE_TO_HOST devDfxArg failed, ret: %d, kind=%d, size=%lu bytes, "
                         "dst=%p, src=%p",
                         static_cast<int>(ret), static_cast<int>(RtMemcpyKind::DEVICE_TO_HOST), sizeof(DevDfxArgs),
                         reinterpret_cast<void*>(&devDfxarg), reinterpret_cast<void*>(args_.devDfxArgAddr));
            return ret;
        }
        devDfxarg.profLevel = profLevelConfig.profConfig.Contains(FedProfFlag::AICORE_TIME) ? 2 : 0;
        ret = RuntimeMemcpyDirect(reinterpret_cast<uint8_t*>(args_.devDfxArgAddr), sizeof(DevDfxArgs), &devDfxarg,
                                  sizeof(DevDfxArgs), RtMemcpyKind::HOST_TO_DEVICE);
        if (ret != RT_SUCCESS) {
            MACHINE_LOGE(RtErr::RT_MEMCPY_FAILED,
                         "RuntimeMemcpyDirect HOST_TO_DEVICE devDfxArg failed, ret: %d, kind=%d, size=%lu bytes, "
                         "dst=%p, src=%p",
                         static_cast<int>(ret), static_cast<int>(RtMemcpyKind::HOST_TO_DEVICE), sizeof(DevDfxArgs),
                         reinterpret_cast<void*>(args_.devDfxArgAddr), reinterpret_cast<void*>(&devDfxarg));
            return ret;
        }
    }
    MACHINE_LOGD("Set debug enable aicore 0 devPtr: %p", dynamic::ValueToPtr(perCoreProfDevAddr_));
    return 0;
}

void FedProf::PerCoreProfCopyToHost(uint32_t blockNum, std::vector<FedProfPerCoreProf*>& perCoreProfList,
                                    std::vector<std::vector<uint8_t>>& hostBufferList)
{
    for (uint32_t index = 0; index < blockNum; ++index) {
        void* devPtr = dynamic::ValueToPtr(perCoreProfDevAddr_ + index * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE);
        size_t dataSize = FED_PROF_PER_CORE_PROF_SIZE;
        hostBufferList[index].resize(dataSize);
        auto ret = NormalizedRtMemcpy(hostBufferList[index].data(), dataSize, devPtr, dataSize,
                                      RtMemcpyKind::DEVICE_TO_HOST);
        if (ret != 0) {
            MACHINE_LOGW("task perf D2H copy failed ret: %d, index: %u", ret, index);
        }
        perCoreProfList[index] = reinterpret_cast<FedProfPerCoreProf*>(hostBufferList[index].data());
    }
}

void FedProf::PerCoreProfClearDevice(uint32_t blockNum, const std::vector<FedProfPerCoreProf*>& perCoreProfList)
{
    for (uint32_t index = 0; index < blockNum; ++index) {
        FedProfPerCoreProf* perCoreProf = perCoreProfList[index];
        perCoreProf->status.leafTaskCount = 0;
        auto ret = NormalizedRtMemcpy(
            dynamic::ValueToPtr(perCoreProfDevAddr_ + index * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE),
            sizeof(FedProfPerCoreProf), perCoreProf, sizeof(FedProfPerCoreProf), RtMemcpyKind::HOST_TO_DEVICE);
        if (ret != 0) {
            MACHINE_LOGW("task perf H2D copy failed ret: %d, index: %u", ret, index);
        }
    }
}

void FedProf::AicpuProfCopyToHost(uint32_t currLaunchRound, std::vector<FedProfAicpuProf>& hostBufferList)
{
    hostBufferList.resize(currLaunchRound);
    for (uint32_t turnIdx = lastDumpRound; turnIdx < currLaunchRound; ++turnIdx) {
        auto devPtr = dynamic::ValueToPtr(aicpuProfDevAddr_ + turnIdx * sizeof(FedProfAicpuProf));
        auto ret = NormalizedRtMemcpy(&hostBufferList[turnIdx], sizeof(FedProfAicpuProf), devPtr,
                                      sizeof(FedProfAicpuProf), RtMemcpyKind::DEVICE_TO_HOST);
        if (ret != 0) {
            MACHINE_LOGW("launch prof D2H copy failed ret: %d, turn: %u", ret, turnIdx);
        }
    }
}

void FedProf::PerCoreProfDumpSwimlane()
{
    auto blockNum = args_.GetBlockNum();
    uint32_t aicoreBlockNum = args_.nrAic + args_.nrAiv;
    uint32_t totalBlockNum = aicoreBlockNum + AICPU_NUM_OF_RUN_AICPU_TASKS;
    MACHINE_LOGI("GetBlockNum : %lu", blockNum);

    std::vector<FedProfPerCoreProf*> perCoreProfList(totalBlockNum);
    std::vector<std::vector<uint8_t>> hostBufferList(totalBlockNum);
    PerCoreProfCopyToHost(totalBlockNum, perCoreProfList, hostBufferList);

    json perCoreProfObjList = json::array();
    for (uint32_t i = 0; i < blockNum; i++) {
        std::string coreType = (i < args_.nrValidAic) ? "AIC" : "AIV";
        json perCoreProfObj;
        dynamic::FedProfPerCoreDumpJson(perCoreProfObj, perCoreProfList[i], i, coreType);
        perCoreProfObjList.push_back(perCoreProfObj);
    }
    for (uint32_t i = aicoreBlockNum; i < totalBlockNum; i++) {
        json perCoreProfObj;
        dynamic::FedProfPerCoreDumpJson(perCoreProfObj, perCoreProfList[i], i, "AI-CPU");
        perCoreProfObjList.push_back(perCoreProfObj);
    }
    PerCoreProfClearDevice(totalBlockNum, perCoreProfList);
    PerCoreProfDumpSwimlaneDraw(perCoreProfObjList);
}

void FedProf::PerCoreProfDumpSwimlaneDraw(const json& perCoreProfObjList)
{
    std::string jsonFilePath = config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    if (!SaveFile(jsonFilePath, perCoreProfObjList.dump(DUMP_LEVEL_FOUR))) {
        MACHINE_LOGW("Construct custom op json failed");
        return;
    }
    MACHINE_LOGD("tilefwk_L1_prof_data has been saved in: %s", jsonFilePath.c_str());
    std::string topo_txt_path = config::LogTopFolder() + "/dyn_topo.txt";
    std::string program_json_path = config::LogTopFolder() + "/program.json";
    std::string mix_event_path = config::GetAbsoluteTopFolder() + "/mix_event_info.json";
    std::string draw_swim_lane_py_path = GetPyptoLibPath() + "/scripts/draw_swim_lane.py";
    config::SetRunDataOption(KEY_SWIM_GRAPH_PATH, config::GetAbsoluteTopFolder() + "/merged_swimlane.json");
    uint64_t freq = (args_.archInfo == ArchInfo::DAV_2201) ? dynamic::FREQ_DAV_2201 : dynamic::FREQ_DAV_3510;

    if (IsPathExist(program_json_path) && IsPathExist(topo_txt_path)) {
        MACHINE_LOGI("The files program.json and dyn_topo.txt exist. Start merging the swimlane.");
        std::string command = "python3 " + draw_swim_lane_py_path + " \"" + jsonFilePath + "\" \"" + topo_txt_path +
                              "\" \"" + program_json_path +
                              "\" --label_type=1 --time_convert_denominator=" + std::to_string(freq) +
                              " --mix_event_info=\"" + mix_event_path + "\"";
        int ret = Checkinject(command.c_str(), command.size());
        if (ret != 0) {
            MACHINE_LOGE(DevCommonErr::SYSTEM_CALL_FAILED, "Draw swimlane cmd illegal char.");
            return;
        }
        if (system(command.c_str()) != 0) {
            MACHINE_LOGW("Failed to execute draw_swim_lane.py. Stop merging the swimlane.");
        }
    } else {
        MACHINE_LOGW("program.json or dyn_topo.txt missing. Stop merging the swimlane.");
    }
}
} // namespace npu::tile_fwk
