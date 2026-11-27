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
 * \file ArchRegistry.cpp
 * \brief Cost-model architecture registry: soc resolution, defaults and factories.
 */

#include "cost_model/simulation/arch/ArchRegistry.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

#include "cost_model/simulation/arch/A2A3/L2CacheImplA2A3.h"
#include "cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h"
#include "cost_model/simulation/arch/A5/L2CacheImplA5.h"
#include "cost_model/simulation/arch/A5/PostSimulatorA5.h"
#include "cost_model/simulation/arch/Kirin9050/PostSimulatorKirin9050.h"
#include "cost_model/simulation/arch/PipeSimulatorFast.h"

namespace CostModel {
namespace {

using SimulatorFactory = UnifiedPipeMachinePtr (*)();
using CacheFactory = std::unique_ptr<CacheMachineImpl> (*)();

std::unique_ptr<CacheMachineImpl> MakeA2A3Cache() { return std::make_unique<L2CacheImplA2A3>(); }

std::unique_ptr<CacheMachineImpl> MakeA5Cache() { return std::make_unique<L2CacheImplA5>(); }

struct ArchEntry {
    npu::tile_fwk::NPUArch npuArch;
    const char* name;
    // Empty means the arch keeps the legacy deviceArch/config arguments.
    std::vector<const char*> defaults;
    SimulatorFactory createSimulator;
    CacheFactory createCache;
};

const std::vector<ArchEntry>& ArchEntries()
{
    static const std::vector<ArchEntry> entries = {
        {npu::tile_fwk::NPUArch::DAV_2201, "A2A3", {}, &CreatePipeSimulatorFast<PostSimulatorA2A3>, &MakeA2A3Cache},
        {npu::tile_fwk::NPUArch::DAV_3510, "A5", {}, &CreatePipeSimulatorFast<PostSimulatorA5>, &MakeA5Cache},
        // Kirin9050 (DAV_5101): CV-mix, 4 mixed cores, no L2.
        {npu::tile_fwk::NPUArch::DAV_5101,
         "Kirin9050",
         {
             "Model.aicpuMachineNumber=0",
             "Model.cubeMachineNumberPerAICPU=0",
             "Model.vecMachineNumberPerAICPU=4",
             "Model.coreMachineNumberPerAICPU=4",
             "Model.cubeVecMixMode=true",
             "Model.drawPngThresholdCycle=5000000",
             "Model.mteUseL2Cache=false",
         },
         &CreatePipeSimulatorFast<PostSimulatorKirin9050>,
         &MakeA5Cache},
        // Kirin9030 (DAV_3113) / KirinX90 (DAV_3003): add entries here once calibrated.
    };
    return entries;
}

const ArchEntry* FindEntryByName(const std::string& archName)
{
    for (const auto& entry : ArchEntries()) {
        if (archName == entry.name) {
            return &entry;
        }
    }
    return nullptr;
}

const ArchEntry* FindEntryByNPUArch(npu::tile_fwk::NPUArch npuArch)
{
    for (const auto& entry : ArchEntries()) {
        if (entry.npuArch == npuArch) {
            return &entry;
        }
    }
    return nullptr;
}

const ArchEntry* FindEntryByPlatformArch()
{
    const auto platformArch = npu::tile_fwk::Platform::Instance().GetSoc().GetNPUArch();
    if (platformArch == npu::tile_fwk::NPUArch::DAV_UNKNOWN) {
        return nullptr;
    }
    return FindEntryByNPUArch(platformArch);
}

const ArchEntry& RequireEntry(const std::string& archName)
{
    const ArchEntry* entry = FindEntryByName(archName);
    if (entry != nullptr) {
        return *entry;
    }
    throw std::invalid_argument("unknown arch type " + archName);
}

bool HasCostModelArg(const std::vector<std::string>& args, const std::string& key)
{
    const std::string prefix = key + "=";
    return std::any_of(args.begin(), args.end(),
                       [&prefix](const std::string& arg) { return arg.rfind(prefix, 0) == 0; });
}

void AppendCostModelArgIfAbsent(std::vector<std::string>& args, const std::string& arg)
{
    const auto pos = arg.find('=');
    if (pos == std::string::npos) {
        return;
    }
    if (!HasCostModelArg(args, arg.substr(0, pos))) {
        args.push_back(arg);
    }
}

void SetArchArg(std::vector<std::string>& args, const char* archName)
{
    const std::string prefix = "Model.deviceArch=";
    args.erase(std::remove_if(args.begin(), args.end(),
                              [&prefix](const std::string& arg) { return arg.rfind(prefix, 0) == 0; }),
               args.end());
    args.emplace_back(prefix + archName);
}

} // namespace

void ArchRegistry::ApplyCurrentArchDefaults(std::vector<std::string>& args)
{
    const ArchEntry* entry = FindEntryByPlatformArch();
    if (entry == nullptr) {
        entry = FindEntryByName("A2A3");
    }

    SetArchArg(args, entry->name);
    for (const char* arg : entry->defaults) {
        AppendCostModelArgIfAbsent(args, arg);
    }
    SIMULATION_LOGI("Cost model arch defaults applied (arch=%s).", entry->name);
}

std::string ArchRegistry::ResolveArchName()
{
    const ArchEntry* entry = FindEntryByPlatformArch();
    if (entry != nullptr) {
        return entry->name;
    }
    SIMULATION_LOGW("Fallback to A2A3: cannot resolve arch from Platform.");
    return "A2A3";
}

UnifiedPipeMachinePtr ArchRegistry::CreateSimulator(const std::string& archName)
{
    return RequireEntry(archName).createSimulator();
}

std::unique_ptr<CacheMachineImpl> ArchRegistry::CreateCache(const std::string& archName)
{
    return RequireEntry(archName).createCache();
}

} // namespace CostModel
