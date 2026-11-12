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
 * \file device_machine.h
 * \brief
 */

#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "machine/device/tilefwk/aikernel_constants.h"
#include "device_utils.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
const uint32_t MAX_DAV_2201_SCHEDULE_AICPU_NUM = 3;
const uint32_t MAX_DAV_3510_SCHEDULE_AICPU_NUM = 4;
const uint32_t MAX_DAV_9201_SCHEDULE_AICPU_NUM = 6;
inline uint32_t CalcSchAicpuNumByBlockDim(uint32_t blockDim, uint32_t aiCpuNum, ArchInfo archInfo)
{
    uint32_t maxScheCore = aiCpuNum - dynamic::MAX_CONTROL_FLOW_AICPU_NUM;
    if (archInfo == ArchInfo::DAV_2201) {
        maxScheCore = maxScheCore >= MAX_DAV_2201_SCHEDULE_AICPU_NUM ? MAX_DAV_2201_SCHEDULE_AICPU_NUM : maxScheCore;
    } else if (archInfo == ArchInfo::DAV_3510) {
        maxScheCore = maxScheCore >= MAX_DAV_3510_SCHEDULE_AICPU_NUM ? MAX_DAV_3510_SCHEDULE_AICPU_NUM : maxScheCore;
    } else if (archInfo == ArchInfo::DAV_9201) {
        maxScheCore = maxScheCore >= MAX_DAV_9201_SCHEDULE_AICPU_NUM ? MAX_DAV_9201_SCHEDULE_AICPU_NUM : maxScheCore;
    }

    if (blockDim > (maxScheCore - 1) * dynamic::MAX_MNG_AICORE_AVG_NUM) {
        return maxScheCore;
    }

    if (blockDim % dynamic::MAX_MNG_AICORE_AVG_NUM == 0) {
        return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM;
    }

    return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM + 1;
}

// Per-launch effective AIC count: prefer ctrlBlockNum (host 控核), else DevProg capacity.
inline uint32_t ResolveRoundNrValidAic(uint64_t ctrlBlockNum, uint32_t fallbackNrValidAic)
{
    return ctrlBlockNum != 0 ? static_cast<uint32_t>(ctrlBlockNum) : fallbackNrValidAic;
}

// Per-launch sche count from effective AIC; never exceed DevProg capacity (queue/shm sized by capacity).
inline uint32_t ResolveRoundScheCpuNum(uint32_t nrValidAic, uint32_t capacityScheCpuNum, ArchInfo archInfo)
{
    // Reconstruct CalcSch's aiCpuNum scale from capacity (launch nrAicpu may already be shrunk / == sche).
    uint32_t aiCpuNum = capacityScheCpuNum + dynamic::MAX_CONTROL_FLOW_AICPU_NUM;
    uint32_t roundSche = CalcSchAicpuNumByBlockDim(nrValidAic, aiCpuNum, archInfo);
    return roundSche < capacityScheCpuNum ? roundSche : capacityScheCpuNum;
}

const uint16_t AICPU_EXECUTE_TIMEOUT = 1080; // 18min

} // namespace npu::tile_fwk::dynamic
