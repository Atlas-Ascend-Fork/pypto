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
 * \file fed_prof_io_json.h
 * \brief
 */

#ifndef FED_PROF_IO_JSON_H
#define FED_PROF_IO_JSON_H

#include <string>
#include "interface/machine/device/tilefwk/aikernel_profile.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace npu::tile_fwk::dynamic {
void FedProfPerCoreMixSyncEventDumpJsonList(json& mixSyncEventArray, const FedProfLeafTaskProf& taskStat,
                                            const FedProfPerCoreProf* perCoreProf);
void FedProfPerCoreDumpJson(json& perCoreProfObj, FedProfPerCoreProf* perCoreProf, uint32_t coreIdx,
                            const std::string& coreType);
void FedProfLaunchAicoreRoundDumpJson(json& launchPerCoreObjList, FedProfPerCoreProf* perCoreProf,
                                      const uint32_t& currLaunchRound, const uint32_t& lastDumpRound);
void FedProfLaunchAicoreDumpJson(json& launchObjList, uint32_t blockNum, uint32_t nrValidAic, const uint32_t& freq,
                                 const uint32_t& currLaunchRound, const uint32_t& lastDumpRound,
                                 const std::vector<FedProfPerCoreProf*>& perCoreProfList);
void FedProfLaunchAicpuDevTaskDumpJson(uint32_t tid, uint32_t type, json& devTaskJson, const FedProfAicpuProf* aicpuPer,
                                       const uint32_t& turnIdx);
void FedProfLaunchAicpuDumpJson(json& launchObjList, uint32_t nrAicpu, uint32_t scheCpuNum, const uint32_t& freq,
                                const uint32_t& currLaunchRound, const uint32_t& lastDumpRound,
                                const std::vector<FedProfAicpuProf>& hostBufferList);
} // namespace npu::tile_fwk::dynamic
#endif
