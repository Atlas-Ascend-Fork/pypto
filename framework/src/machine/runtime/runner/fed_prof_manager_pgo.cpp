/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fed_prof_manager_pgo.cpp
 * \brief PGO (Profiling Guided Optimization) option sync and data dump.
 *
 * Option handling: PGO only records data. `experimental_pgo_path` / `experimental_pgo_update` are read
 * from the operator scope (ConfigManagerNg, prefix `generic.`) during the launch and are never written
 * into the process-global config. `experimental_pgo_path` is the output directory; the data is dumped
 * as one json per FedProfKind:
 *
 *     <experimental_pgo_path>/<fed_prof_name_list[kind]>.json      // e.g. per_core.json, aicpu.json
 *
 * Reading and using the dumped data belongs to the consumer of the PGO data.
 */

#include "machine/runtime/runner/fed_prof_manager.h"

#include <cstdint>
#include <string>
#include <vector>

#include "interface/configs/config_manager_ng.h"
#include "interface/machine/device/tilefwk/aikernel_profile.h"
#include "tilefwk/pypto_fwk_log.h"
#include "utils/file_utils.h"

namespace npu::tile_fwk {
namespace {
constexpr int DUMP_JSON_INDENT = 4;
constexpr uint32_t FED_PROF_DUMP_ROUND_BEGIN = 0;
constexpr uint32_t FED_PROF_DUMP_ROUND_END = FED_PROF_DEV_TASK_MAX_COUNT;
} // namespace

void FedProfManager::PgoUpdatePath()
{
    // 每次 launch（算子作用域生效期间）同步一次选项：仅在 experimental_pgo_update=true 时
    // 记下 experimental_pgo_path 作为导出目录；不做算子维度的数据切分。
    auto scope = ConfigManagerNg::CurrentScope();
    if (scope == nullptr || !scope->HasConfig(std::string(GENERIC_PREFIX) + EXPERIMENTAL_PGO_PATH)) {
        return; // 当前作用域未配置 PGO
    }
    if (!scope->HasConfig(std::string(GENERIC_PREFIX) + EXPERIMENTAL_PGO_UPDATE) ||
        !scope->GetGenericConfig<bool>(EXPERIMENTAL_PGO_UPDATE)) {
        return; // 未要求导出
    }
    pgoPath_ = scope->GetGenericConfig<std::string>(EXPERIMENTAL_PGO_PATH);
    if (pgoPath_.empty()) {
        MACHINE_LOGW("PGO update is enabled but experimental_pgo_path is empty, skip PGO dump");
        return;
    }
    MACHINE_LOGI("PGO update requested to %s", pgoPath_.c_str());
}

void FedProfManager::PgoDump()
{
    if (pgoPath_.empty()) {
        return; // 本次运行没有算子要求导出 PGO 数据
    }
    (void)CreateDir(pgoPath_, true);

    // 枚举所有 prof 数据，逐类调用其 DumpJson 输出，分别保存为 <pgoPath_>/<kind 名称>.json
    FedProfLaunchArgs launchArgs{FED_PROF_DUMP_ROUND_BEGIN, FED_PROF_DUMP_ROUND_END};
    for (uint32_t kind = FED_PROF_PER_CORE_PROF; kind < FED_PROF_KIND_MAX; ++kind) {
        auto& holder = profList_[kind];
        if (holder == nullptr || holder->GetDevAddr() == 0) {
            continue; // 该类数据未使能
        }
        std::vector<uint8_t> hostBuffer;
        std::vector<FedProfDataBase*> dataList;
        holder->DeviceDataGet(hostBuffer, dataList, args_, launchArgs);
        json pgoJson = json::array();
        holder->DumpJson(pgoJson, hostBuffer, dataList, args_, launchArgs);

        std::string filePath = pgoPath_ + "/" + fed_prof_name_list[kind] + ".json";
        if (!SaveFile(filePath, pgoJson.dump(DUMP_JSON_INDENT))) {
            MACHINE_LOGW("Failed to dump PGO data to %s", filePath.c_str());
            continue;
        }
        MACHINE_LOGI("PGO data dumped to %s", filePath.c_str());
    }
}
} // namespace npu::tile_fwk
