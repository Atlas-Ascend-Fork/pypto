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
 * \file PostSimulatorKirin9050.h
 * \brief Kirin9050 per-tile feature extraction for the confirmed single-op rules.
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "cost_model/simulation/arch/Kirin9050/Kirin9050CostCoeffs.h"
#include "cost_model/simulation/arch/PipeMachineImpl.h"

namespace CostModel {
class PostSimulatorKirin9050 {
public:
    static constexpr bool kUseExprCostEngine = true;

    static const CostRule* FindTileOpRule(const std::string& opcode) { return Kirin9050::FindTileOpRule(opcode); }
};
} // namespace CostModel
