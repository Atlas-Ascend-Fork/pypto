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
 * \file fed_prof_io_json.cpp
 * \brief
 */

#include "machine/runtime/runner/fed_prof_io_json.h"

#include <algorithm>
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk::dynamic {
void FedProfPerCoreMixSyncEventDumpJsonList(json& mixSyncEventArray, const FedProfLeafTaskProf& taskStat,
                                            const FedProfPerCoreProf* perCoreProf)
{
    mixSyncEventArray = json::array();

    const uint8_t* perCoreProfBase = reinterpret_cast<const uint8_t*>(perCoreProf);
    const uint64_t* setEventBase = reinterpret_cast<const uint64_t*>(perCoreProfBase + taskStat.setEventAddr);
    const uint64_t* waitEventBase = reinterpret_cast<const uint64_t*>(perCoreProfBase + taskStat.waitEventAddr);

    for (int k = 0; k < taskStat.setEventNum; ++k) {
        uint64_t timestamp = setEventBase[k];
        if (timestamp != 0) {
            json setEvent;
            setEvent["idx"] = k;
            setEvent["type"] = "CV_SYNC_SET";
            setEvent["time"] = timestamp;
            mixSyncEventArray.push_back(setEvent);
        }
    }
    for (int k = 0; k < taskStat.waitEventNum; ++k) {
        uint64_t timestamp = waitEventBase[k];
        if (timestamp != 0) {
            json waitEvent;
            waitEvent["idx"] = k;
            waitEvent["type"] = "CV_SYNC_WAIT";
            waitEvent["time"] = timestamp;
            mixSyncEventArray.push_back(waitEvent);
        }
    }

    std::sort(mixSyncEventArray.begin(), mixSyncEventArray.end(),
              [](const json& a, const json& b) { return a.value("time", 0) < b.value("time", 0); });
}

void FedProfPerCoreDumpJson(json& perCoreProfObj, FedProfPerCoreProf* perCoreProf, uint32_t coreIdx,
                            const std::string& coreType)
{
    if (perCoreProf->status.leafTaskCount > FED_PROF_LEAF_TASK_MAX_COUNT) {
        perCoreProf->status.leafTaskCount = FED_PROF_LEAF_TASK_MAX_COUNT;
    }
    FedProfLeafTaskProf* taskStats = perCoreProf->leafTaskProfList;
    size_t numTasks = perCoreProf->status.leafTaskCount;
    perCoreProfObj["blockIdx"] = coreIdx;
    perCoreProfObj["coreType"] = coreType;
    if (!IsAicoreResolveEnabled() && perCoreProf->status.coreType != -1) {
        perCoreProfObj["coreType"] = perCoreProf->status.coreType == static_cast<int16_t>(CoreType::AIC) ? "AIC" :
                                                                                                           "AIV";
    }
    json tasksArr = json::array();
    for (size_t j = 0; j < numTasks; ++j) {
        if (taskStats[j].timestampEnd != 0) {
            json taskObj;
            taskObj["seqNo"] = taskStats[j].devTaskIdx;
            taskObj["leafIndex"] = taskStats[j].leafFuncIdx;
            taskObj["taskId"] = taskStats[j].taskId;
            taskObj["execStart"] = taskStats[j].timestampStart;
            taskObj["execEnd"] = taskStats[j].timestampEnd;
            json syncEventsArr;
            FedProfPerCoreMixSyncEventDumpJsonList(syncEventsArr, taskStats[j], perCoreProf);
            if (!syncEventsArr.empty()) {
                taskObj["syncEvents"] = syncEventsArr;
            }
            tasksArr.push_back(taskObj);
        }
    }
    perCoreProfObj["tasks"] = tasksArr;
}

void FedProfLaunchAicoreRoundDumpJson(json& launchPerCoreObjList, FedProfPerCoreProf* perCoreProf,
                                      const uint32_t& currLaunchRound, const uint32_t& lastDumpRound)
{
    uint64_t curCycle = 0;
    for (uint32_t turnIdx = lastDumpRound; turnIdx < currLaunchRound; turnIdx++) {
        FedProfDevTaskProf* devTaskProf = &(perCoreProf->aicoreDevTaskInfo[turnIdx]);
        uint64_t cntNum = std::min<uint64_t>(
            devTaskProf->cnt, static_cast<uint64_t>(PERF_TRACE_CORE_KIND_MAX) * PERF_TRACE_CORE_PER_KIND_MAX_COUNT);
        for (uint64_t cnt = 0; cnt < cntNum; cnt++) {
            FedProfDevTaskCoreEvent perf = devTaskProf->aicoreEveryDevTypeTimeStamp[cnt];

            curCycle = perf.timestamp;
            if (curCycle == 0) {
                continue;
            }

            uint32_t type = static_cast<uint32_t>(perf.type);
            if (type >= PERF_TRACE_CORE_KIND_MAX) {
                continue;
            }

            json aicoreTaskType;
            std::string name = FedProfPerfTraceKindName[type];
            name = name + "_" + std::to_string(turnIdx);
            if (perf.devTaskIdx != INVALID_DEV_TASK_ID) {
                name = name + "(" + std::to_string(perf.devTaskIdx) + ")";
            }
            aicoreTaskType["name"] = name;
            aicoreTaskType["end"] = curCycle;
            launchPerCoreObjList.push_back(aicoreTaskType);
        }
    }
}

void FedProfLaunchAicoreDumpJson(json& launchObjList, uint32_t blockNum, uint32_t nrValidAic, const uint32_t& freq,
                                 const uint32_t& currLaunchRound, const uint32_t& lastDumpRound,
                                 const std::vector<FedProfPerCoreProf*>& perCoreProfList)
{
    for (uint32_t i = 0; i < blockNum; i++) {
        FedProfPerCoreProf* perCoreProf = perCoreProfList[i];
        std::string coreType = perCoreProf->status.coreType == static_cast<int16_t>(CoreType::AIC) ? "AIC" : "AIV";
        if (IsAicoreResolveEnabled()) {
            coreType = i < nrValidAic ? "AIC" : "AIV";
        }
        json aicoreTask;
        aicoreTask["blockIdx"] = i + 1;
        aicoreTask["coreType"] = "SCHED" + std::to_string(perCoreProf->status.scheCpuIdx) + "-" + coreType;
        aicoreTask["freq"] = freq;
        json launchPerCoreObjList = json::array();
        FedProfLaunchAicoreRoundDumpJson(launchPerCoreObjList, perCoreProf, currLaunchRound, lastDumpRound);
        aicoreTask["tasks"] = launchPerCoreObjList;
        launchObjList.push_back(aicoreTask);
    }
}

void FedProfLaunchAicpuDevTaskDumpJson(uint32_t tid, uint32_t type, json& devTaskJson, const FedProfAicpuProf* aicpuPer,
                                       const uint32_t& launchIndex)
{
    json per_dev_task;
    uint32_t idx = DEVTASK_PERF_ARRY_INDEX(type);
    const FedProfLaunchDevTaskProf& perfSlot = aicpuPer->devTaskPerfs[tid][idx];
    uint64_t cnt = std::min<uint64_t>(perfSlot.cnt, PERF_TRACE_COUNT_DEVTASK_MAX_NUM);
    for (uint32_t i = 0; i < cnt; i++) {
        std::string name = PerfTraceName[type];
        name = name + "_" + std::to_string(launchIndex);
        if (type != PERF_TRACE_DEV_TASK_SEND_FIRST_LEAF_TASK) {
            name = name + "(" + std::to_string(i) + ")";
        }
        per_dev_task["name"] = name;
        per_dev_task["end"] = perfSlot.timeStamp[i];
        devTaskJson.push_back(per_dev_task);
    }
}

void FedProfLaunchAicpuDumpJson(json& launchObjList, uint32_t nrAicpu, uint32_t scheCpuNum, const uint32_t& freq,
                                const uint32_t& currLaunchRound, const uint32_t& lastDumpRound,
                                const std::vector<FedProfAicpuProf*>& aicpuProfList)
{
    for (uint32_t i = 0; i < nrAicpu - 1; i++) {
        json aicpu;
        std::string coreType = "AICPU";
        if (i == 0) {
            coreType = "AICPU-CTRL";
        } else if (i <= scheCpuNum) {
            coreType = "AICPU-SCHED";
        }
        aicpu["blockIdx"] = i;
        aicpu["coreType"] = coreType;
        aicpu["freq"] = freq;
        json aicpuDevTasks = json::array();
        for (uint32_t launchIndex = lastDumpRound; launchIndex < currLaunchRound; launchIndex++) {
            const FedProfAicpuProf* aicpuMetric = aicpuProfList[launchIndex - lastDumpRound];
            for (uint32_t type = 0; type < PERF_TRACE_MAX; type++) {
                if (IsDevTaskType(type)) {
                    FedProfLaunchAicpuDevTaskDumpJson(i, type, aicpuDevTasks, aicpuMetric, launchIndex);
                    continue;
                }
                if (aicpuMetric->perfAicpuTrace[i][type] == 0) {
                    continue;
                }
                json schCtrAicpu;
                std::string name = PerfTraceName[type];
                schCtrAicpu["name"] = name + "_" + std::to_string(launchIndex);
                schCtrAicpu["end"] = aicpuMetric->perfAicpuTrace[i][type];
                aicpuDevTasks.push_back(schCtrAicpu);
            }
        }
        aicpu["tasks"] = aicpuDevTasks;
        launchObjList.push_back(aicpu);
    }
}
} // namespace npu::tile_fwk::dynamic
