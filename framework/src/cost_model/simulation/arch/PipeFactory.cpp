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
 * \file PipeFactory.cpp
 * \brief Pipe/cache dispatch; arch-specific implementation selection lives in ArchRegistry.
 */

#include "PipeFactory.h"

#include <stdexcept>
#include <string>

#include "CallPipeImpl.h"
#include "TileAllocPipeImpl.h"
#include "cost_model/simulation/arch/ArchRegistry.h"

namespace CostModel {
UnifiedPipeMachinePtr PipeFactory::Create(CorePipeType pipeType, std::string archType)
{
    switch (pipeType) {
        case CorePipeType::PIPE_TILE_ALLOC:
            return CreateTileAllocPipeImpl();
        case CorePipeType::PIPE_CALL:
            return CreateCallPipeImpl();
        case CorePipeType::PIPE_MTE_IN:
        case CorePipeType::PIPE_MTE1:
        case CorePipeType::PIPE_MTE_OUT:
        case CorePipeType::PIPE_FIX:
        case CorePipeType::PIPE_VECTOR_ALU:
        case CorePipeType::PIPE_S:
        case CorePipeType::PIPE_CUBE:
            return ArchRegistry::CreateSimulator(archType);
        default:
            throw std::invalid_argument("unknown pipe type " + CorePipeName(pipeType));
    }
    return nullptr;
}

std::unique_ptr<CacheMachineImpl> PipeFactory::CreateCache(CostModel::CacheType type, std::string archType)
{
    if (type == CacheType::L2CACHE) {
        return ArchRegistry::CreateCache(archType);
    }
    throw std::invalid_argument("unknown cache type " + CacheName(type));
}
} // namespace CostModel
