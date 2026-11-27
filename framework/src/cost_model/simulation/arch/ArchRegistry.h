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
 * \file ArchRegistry.h
 * \brief Cost-model architecture registry: soc resolution, defaults and factories.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cost_model/simulation/arch/CacheMachineImpl.h"
#include "cost_model/simulation/arch/PipeMachineImpl.h"

namespace CostModel {
class ArchRegistry {
public:
    // Apply the current soc arch's default CostModel arguments.
    static void ApplyCurrentArchDefaults(std::vector<std::string>& args);
    // Resolve the registered arch from the current Platform (A2A3 fallback).
    static std::string ResolveArchName();
    static UnifiedPipeMachinePtr CreateSimulator(const std::string& archName);
    static std::unique_ptr<CacheMachineImpl> CreateCache(const std::string& archName);
};
} // namespace CostModel
