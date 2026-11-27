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
 * \file PipeSimulatorFast.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include "PipeMachineImpl.h"

namespace CostModel {

struct OpCostRequestC {
    const char* opcode;
    const int* inShapeData;       // required when inCount > 0
    const int* inShapeDims;       // required when inCount > 0
    int32_t inCount;              // 0 for output-only ops
    const int32_t* inDtypes;      // required when inCount > 0
    const int* outShapeData;      // required
    const int* outShapeDims;      // required
    int32_t outCount;             // > 0
    const int32_t* outDtypes;     // required
    const int* inValidShapeData;  // optional
    const int* inValidShapeDims;  // optional
    const int* outValidShapeData; // optional
    const int* outValidShapeDims; // optional
    uint32_t opModeFlag;          // optional, default 0
};

extern "C" __attribute__((visibility("default"))) int64_t GetCyclesForPass(const OpCostRequestC* request);

__attribute__((visibility("default"))) int64_t GetCyclesForPassCPP(
    const std::string& opcode, const std::vector<std::vector<int>>& inShapes,
    const std::vector<std::vector<int>>& outShapes, const std::vector<DataType>& inDtypes,
    const std::vector<DataType>& outDtypes, uint32_t opModeFlag = 0,
    const std::vector<std::vector<int>>& inValidShapes = {}, const std::vector<std::vector<int>>& outValidShapes = {});

template <typename PostSimulator>
class PipeSimulatorFast : public PipeMachineImpl {
public:
    uint64_t Simulate(const TileOpPtr& tileOp) override;
    uint64_t PostSimulate(const TileOpPtr& tileOp) override;
    uint64_t PostSimulateForPass(const std::string& opcode, const std::vector<std::vector<int>>& inShapes,
                                 const std::vector<std::vector<int>>& outShapes, const std::vector<DataType>& inDtypes,
                                 const std::vector<DataType>& outDtypes, uint32_t opModeFlag,
                                 const std::vector<std::vector<int>>& inValidShapes,
                                 const std::vector<std::vector<int>>& outValidShapes) override;

private:
    std::unordered_map<std::string, uint64_t> tileopLatencyCacheMp;
};

template <typename Simulator>
inline UnifiedPipeMachinePtr CreatePipeSimulatorFast()
{
    return UnifiedPipeMachinePtr(new PipeSimulatorFast<Simulator>());
}
} // namespace CostModel
