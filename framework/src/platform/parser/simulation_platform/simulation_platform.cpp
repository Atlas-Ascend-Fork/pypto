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
 * \file simulation_platform.cpp
 * \brief
 */

#include <climits>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include "simulation_platform.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
const std::string PLATFORM_INFO_RELATIVE_PATH = "/configs/";
const std::string FWK_CONFIG_RELATIVE_PATH = "tile_fwk_config.json";
const std::string DEFAULT_SOC_VERSION = "Ascend910B2";
const std::string INI_EXTENSION = ".ini";
const uint32_t PLATFORM_FAILED = 0xFFFFFFFF;
const uint32_t PLATFORM_SUCCESS = 0;

namespace {
// Deprecated device_platform formats (e.g. "ASCEND_910B2") mapped to standard soc_version, for compatibility only.
// Legacy values whose names did not match the actual chip bin are normalized to the real soc_version
// of the ini they used to load: ASCEND_950DT_9579 -> Ascend950DT_9572, ASCEND_950PR_9582 -> Ascend950PR_9589.
const std::unordered_map<std::string, std::string> legacyDevicePlatformMap = {
    {"ASCEND_910B1", "Ascend910B1"},
    {"ASCEND_910B2", "Ascend910B2"},
    {"ASCEND_910B3", "Ascend910B3"},
    {"ASCEND_910B4", "Ascend910B4"},
    {"ASCEND_910_9363", "Ascend910_9363"},
    {"ASCEND_950DT_9572", "Ascend950DT_9572"},
    {"ASCEND_950DT_9579", "Ascend950DT_9572"},
    {"ASCEND_950PR_9579", "Ascend950PR_9579"},
    {"ASCEND_950DT_9581", "Ascend950DT_9581"},
    {"ASCEND_950DT_9581X", "Ascend950DT_9581X"},
    {"ASCEND_950DT_9582", "Ascend950DT_9582"},
    {"ASCEND_950PR_9582", "Ascend950PR_9589"},
    {"ASCEND_950PR_9589", "Ascend950PR_9589"},
    {"KIRIN_9030", "Kirin9030"},
    {"KIRIN_X90", "KirinX90"},
    {"KIRIN_9050", "Kirin9050"},
};

std::string NormalizeLegacyDevicePlatform(const std::string& platform)
{
    auto it = legacyDevicePlatformMap.find(platform);
    if (it != legacyDevicePlatformMap.end()) {
        PLATFORM_LOGW("Deprecated device_platform format:%s, use standard soc_version:%s instead.", platform.c_str(),
                      it->second.c_str());
        return it->second;
    }
    return platform;
}
} // namespace

std::string SimulationPlatform::GetDevicePlatform()
{
    const std::string configPath = RealPath(GetPyptoLibPath() + PLATFORM_INFO_RELATIVE_PATH + FWK_CONFIG_RELATIVE_PATH);
    if (configPath.empty()) {
        PLATFORM_LOGW("Failed to open tile_fwk_config.json, use default platform.");
        return DEFAULT_SOC_VERSION;
    }
    std::ifstream jsonFile(configPath);
    nlohmann::json jsonData = nlohmann::json::parse(jsonFile);
    jsonFile.close();

    std::string socVersion;
    if (jsonData.contains("global") && jsonData["global"].contains("platform") &&
        jsonData["global"]["platform"].contains("device_platform") &&
        jsonData["global"]["platform"]["device_platform"].is_string()) {
        socVersion = jsonData["global"]["platform"]["device_platform"].get<std::string>();
        PLATFORM_LOGD("Key 'global.platform.device_platform' specified soc version:%s.", socVersion.c_str());
    } else {
        PLATFORM_LOGW("Key 'global.platform.device_platform' not found in %s, use default platform.",
                      configPath.c_str());
        return DEFAULT_SOC_VERSION;
    }

    socVersion = NormalizeLegacyDevicePlatform(socVersion);
    if (StringToDPlatform(socVersion) == DPlatform::UNKNOWN_DEVICE) {
        PLATFORM_LOGW("Invalid soc version:%s, use default platform.", socVersion.c_str());
        return DEFAULT_SOC_VERSION;
    }
    return socVersion;
}

bool SimulationPlatform::GetCostModelPlatformRealPath(const std::string& socVersion, std::string& realPath)
{
    std::string platformSocVersion = socVersion;
    if (platformSocVersion.empty()) {
        platformSocVersion = GetDevicePlatform();
        PLATFORM_LOGD("Config specified SoC version:%s.", platformSocVersion.c_str());
    }
    realPath = RealPath(GetPyptoLibPath() + PLATFORM_INFO_RELATIVE_PATH + platformSocVersion + INI_EXTENSION);
    if (realPath.empty()) {
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu
