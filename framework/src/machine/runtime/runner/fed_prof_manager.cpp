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
 * \file fed_prof_manager.cpp
 * \brief
 */

#include "machine/runtime/runner/fed_prof_manager.h"
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/device/dynamic/device_common.h"
#include "interface/machine/device/tilefwk/aikernel_profile_aicpu.h"
#include "interface/configs/config_manager.h"
#include "machine/utils/checkinject.h"
#include "utils/file_utils.h"

namespace npu::tile_fwk {
namespace {
constexpr uint32_t AICPU_NUM_OF_RUN_AICPU_TASKS = 1;
constexpr int DUMP_LEVEL_FOUR = 4;
} // namespace

FedProfManager& FedProfManager::GetInstance()
{
    static FedProfManager instance;
    return instance;
}

FedProfManager::FedProfManager() : args_()
{
    profList_[FED_PROF_PER_CORE_PROF] = std::make_unique<FedProfPerCoreProfHolder>();
    profList_[FED_PROF_AICPU_PROF] = std::make_unique<FedProfAicpuProfHolder>();
}
FedProfManager::~FedProfManager()
{
    MACHINE_LOGD("Start to clean up profile data.");
    AicpuProfDumpThreadStop();
    AicpuProfReleaseData();
    PerCoreProfReleaseData();
}

void FedProfManager::InitDeviceArgs(const DeviceArgs& args) { args_ = FedProfDeviceArgs(args); }

uint32_t FedProfManager::GetCoreCount() const { return args_.nrAic + args_.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; }

uint64_t FedProfManager::DeviceDataAllocate(FedProfKind kind) { return profList_[kind]->DeviceDataAllocate(args_); }

void FedProfManager::PerCoreProfResetData() const
{
    for (uint32_t i = 0; i < GetCoreCount(); i++) {
        int rc = RuntimeMemset(dynamic::ValueToPtr(PerCoreProfDevAddr() + i * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE),
                               FED_PROF_PER_CORE_PROF_SIZE, 0, FED_PROF_PER_CORE_PROF_SIZE);
        if (rc != 0) {
            MACHINE_LOGW("CoreId %u, RuntimeMemset failed, rc: %d", i, rc);
        }
    }
}

void FedProfManager::PerCoreProfReleaseData() { profList_[FED_PROF_PER_CORE_PROF]->DeviceDataRelease(); }

void FedProfManager::AicpuProfReleaseData() { profList_[FED_PROF_AICPU_PROF]->DeviceDataRelease(); }

void FedProfManager::PerCoreProfResetMetadata(bool multipleLaunch)
{
    if (PerCoreProfDevAddr() == 0 || (multipleLaunch && perCoreProfDataInited_)) {
        return;
    }
    for (uint32_t coreId = 0; coreId < GetCoreCount(); coreId++) {
        RuntimeMemset(dynamic::ValueToPtr(PerCoreProfDevAddr() + coreId * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE),
                      sizeof(FedProfPerCoreProf), 0, sizeof(FedProfPerCoreProf));
    }
    if (multipleLaunch) {
        perCoreProfDataInited_ = true;
    }
}

void FedProfManager::PerCoreProfSync(bool debugEnable)
{
    if (debugEnable) {
        // 多轮控核，nrValidAic和scheCpuNum需实时刷新，否则泳道图会出错
        args_.nrValidAic = GetCfgBlockdim();
        args_.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(args_.nrValidAic, args_.nrAicpu, args_.archInfo);
        PerCoreProfDumpSwimlane();
    }
}

int FedProfManager::SetDebugEnable(const ToSubMachineConfig& profLevelConfig)
{
    if (DeviceDataAllocate(FED_PROF_PER_CORE_PROF) == 0) {
        return static_cast<int>(DevCommonErr::MALLOC_FAILED);
    }
    int32_t ret = 0;
    PerCoreProfResetMetadata(AicpuProfDevAddr() != 0);
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
    MACHINE_LOGD("Set debug enable aicore 0 devPtr: %p", dynamic::ValueToPtr(PerCoreProfDevAddr()));
    return 0;
}

void FedProfManager::PerCoreProfDumpSwimlane()
{
    MACHINE_LOGI("GetBlockNum : %lu", args_.GetBlockNum());
    std::vector<uint8_t> hostBuffer;
    std::vector<FedProfDataBase*> dataList;
    profList_[FED_PROF_PER_CORE_PROF]->DeviceDataGet(hostBuffer, dataList, args_, FedProfLaunchArgs{});

    json perCoreProfObjList = json::array();
    profList_[FED_PROF_PER_CORE_PROF]->DumpJson(perCoreProfObjList, hostBuffer, dataList, args_, FedProfLaunchArgs{});
    profList_[FED_PROF_PER_CORE_PROF]->DeviceDataClear();
    PerCoreProfDumpSwimlaneDraw(perCoreProfObjList);
}

void FedProfManager::PerCoreProfDumpSwimlaneDraw(const json& perCoreProfObjList)
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
