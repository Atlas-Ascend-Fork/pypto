/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/backend.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/common/soc.h"
#include "core/logging.h"
#include "tilefwk/error.h"
#include "ir/memref.h"
#include "ir/pipe.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using npu::tile_fwk::ExternalError;

namespace {

template <typename CoreT>
std::optional<uint64_t> FindMemSizeInCore(const CoreT& core, ir::MemorySpace memType)
{
    for (const auto& mem : core.GetMems()) {
        if (mem.GetMemType() == memType) {
            return mem.GetMemSize();
        }
    }
    return std::nullopt;
}

template <typename ClusterT>
std::optional<uint64_t> FindMemSizeInCluster(const ClusterT& cluster, ir::MemorySpace memType)
{
    for (const auto& coreEntry : cluster.GetCoreCounts()) {
        if (auto size = FindMemSizeInCore(coreEntry.first, memType)) {
            return size;
        }
    }
    return std::nullopt;
}

template <typename DieT>
std::optional<uint64_t> FindMemSizeInDie(const DieT& die, ir::MemorySpace memType)
{
    for (const auto& clusterEntry : die.GetClusterCounts()) {
        if (auto size = FindMemSizeInCluster(clusterEntry.first, memType)) {
            return size;
        }
    }
    return std::nullopt;
}

} // namespace

// ========== Backend Implementation ==========

std::vector<ir::MemorySpace> Backend::FindMemPath(ir::MemorySpace from, ir::MemorySpace to) const
{
    if (from == to) {
        return {from};
    }

    const auto& memGraph = soc_->GetMemoryGraph();

    // BFS to find shortest path
    std::queue<ir::MemorySpace> queue;
    std::unordered_map<ir::MemorySpace, ir::MemorySpace> parent;
    std::set<ir::MemorySpace> visited;

    queue.push(from);
    visited.insert(from);
    parent[from] = from; // Mark root

    bool found = false;
    while (!queue.empty() && !found) {
        auto current = queue.front();
        queue.pop();

        auto it = memGraph.find(current);
        if (it == memGraph.end()) {
            continue;
        }

        for (auto neighbor : it->second) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                parent[neighbor] = current;
                queue.push(neighbor);

                if (neighbor == to) {
                    found = true;
                    break;
                }
            }
        }
    }

    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, found)
        << "No path found from " << static_cast<int>(from) << " to " << static_cast<int>(to);

    // Reconstruct path
    std::vector<ir::MemorySpace> path;
    ir::MemorySpace current = to;
    while (current != from) {
        path.push_back(current);
        current = parent[current];
    }
    path.push_back(from);
    std::reverse(path.begin(), path.end());

    return path;
}

uint64_t Backend::GetMemSize(ir::MemorySpace memType) const
{
    for (const auto& dieEntry : soc_->GetDieCounts()) {
        if (auto size = FindMemSizeInDie(dieEntry.first, memType)) {
            return *size;
        }
    }
    return 0;
}

// ========== Operator Registration ==========

BackendOpRegistryEntry Backend::RegisterOp(const std::string& opName) { return BackendOpRegistryEntry(this, opName); }

void Backend::FinalizeOpRegistration(const std::string& opName, std::optional<ir::PipeType> pipe,
                                     BackendCodegenFunc func)
{
    if (backendOpRegistry_.find(opName) != backendOpRegistry_.end()) {
        return;
    }
    backendOpRegistry_[opName] = BackendOpInfo{pipe, std::move(func)};
}

const Backend::BackendOpInfo* Backend::GetOpInfo(const std::string& opName) const
{
    auto it = backendOpRegistry_.find(opName);
    if (it != backendOpRegistry_.end()) {
        return &it->second;
    }
    return nullptr;
}

// ========== BackendOpRegistryEntry Implementation ==========

BackendOpRegistryEntry& BackendOpRegistryEntry::set_pipe(ir::PipeType pipe)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, !pipe_.has_value())
        << "Pipe type already set for op '" << opName_ << "'";
    pipe_ = pipe;
    return *this;
}

BackendOpRegistryEntry& BackendOpRegistryEntry::f_codegen(BackendCodegenFunc func)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, !codegenFunc_.has_value())
        << "Codegen function already set for op '" << opName_ << "'";
    codegenFunc_ = std::move(func);
    return *this;
}

BackendOpRegistryEntry::~BackendOpRegistryEntry()
{
    if (backend_ && codegenFunc_.has_value()) {
        backend_->FinalizeOpRegistration(opName_, pipe_, std::move(*codegenFunc_));
    }
}

} // namespace backend
} // namespace pypto
