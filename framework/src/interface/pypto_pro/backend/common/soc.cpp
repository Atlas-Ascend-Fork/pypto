/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/soc.h"

#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "backend/common/cann_runtime.h"
#include "core/logging.h"
#include "ir/memref.h"
#include "ir/pipe.h"

namespace pypto {
namespace backend {

// ========== Mem Implementation ==========

Mem::Mem(ir::MemorySpace memType, uint64_t memSize, uint64_t alignment)
    : memType_(memType), memSize_(memSize), alignment_(alignment)
{}

bool Mem::operator<(const Mem& other) const
{
    return std::tie(memType_, memSize_, alignment_) < std::tie(other.memType_, other.memSize_, other.alignment_);
}

bool Mem::operator==(const Mem& other) const
{
    return memType_ == other.memType_ && memSize_ == other.memSize_ && alignment_ == other.alignment_;
}

// ========== Core Implementation ==========

Core::Core(ir::CoreType coreType, std::vector<Mem> mems) : coreType_(coreType), mems_(std::move(mems)) {}

bool Core::operator<(const Core& other) const
{
    if (coreType_ != other.coreType_) {
        return coreType_ < other.coreType_;
    }
    return mems_ < other.mems_;
}

bool Core::operator==(const Core& other) const { return coreType_ == other.coreType_ && mems_ == other.mems_; }

// ========== Cluster Implementation ==========

Cluster::Cluster(std::map<Core, int> coreCounts) : coreCounts_(std::move(coreCounts)) {}

Cluster::Cluster(const Core& core, int count) : coreCounts_({{core, count}}) {}

int Cluster::TotalCoreCount() const
{
    return std::accumulate(coreCounts_.begin(), coreCounts_.end(), 0,
                           [](int sum, const auto& pair) { return sum + pair.second; });
}

bool Cluster::operator<(const Cluster& other) const { return coreCounts_ < other.coreCounts_; }

bool Cluster::operator==(const Cluster& other) const { return coreCounts_ == other.coreCounts_; }

// ========== Die Implementation ==========

Die::Die(std::map<Cluster, int> clusterCounts) : clusterCounts_(std::move(clusterCounts)) {}

Die::Die(const Cluster& cluster, int count) : clusterCounts_({{cluster, count}}) {}

int Die::TotalClusterCount() const
{
    return std::accumulate(clusterCounts_.begin(), clusterCounts_.end(), 0,
                           [](int sum, const auto& pair) { return sum + pair.second; });
}

int Die::TotalCoreCount() const
{
    return std::accumulate(clusterCounts_.begin(), clusterCounts_.end(), 0,
                           [](int sum, const auto& pair) { return sum + pair.first.TotalCoreCount() * pair.second; });
}

bool Die::operator<(const Die& other) const { return clusterCounts_ < other.clusterCounts_; }

bool Die::operator==(const Die& other) const { return clusterCounts_ == other.clusterCounts_; }

// ========== SoC Implementation ==========

SoC::SoC(std::map<Die, int> dieCounts, std::map<ir::MemorySpace, std::vector<ir::MemorySpace>> memGraph)
    : dieCounts_(std::move(dieCounts)), memGraph_(std::move(memGraph))
{}

SoC::SoC(const Die& die, int count, std::map<ir::MemorySpace, std::vector<ir::MemorySpace>> memGraph)
    : dieCounts_({{die, count}}), memGraph_(std::move(memGraph))
{}

int SoC::TotalDieCount() const
{
    return std::accumulate(dieCounts_.begin(), dieCounts_.end(), 0,
                           [](int sum, const auto& pair) { return sum + pair.second; });
}

int SoC::TotalClusterCount() const
{
    return std::accumulate(dieCounts_.begin(), dieCounts_.end(), 0, [](int sum, const auto& pair) {
        return sum + pair.first.TotalClusterCount() * pair.second;
    });
}

int SoC::TotalCoreCount() const
{
    return std::accumulate(dieCounts_.begin(), dieCounts_.end(), 0,
                           [](int sum, const auto& pair) { return sum + pair.first.TotalCoreCount() * pair.second; });
}

// ========== SoC Factory ==========

const SoC& CreateSoC()
{
    // Singleton instance shared by all backends
    static SoC soc = []() {
        auto& rt = CannRuntime::Instance();

        auto queryU64 = [&](const std::string& col, const std::string& key, uint64_t fallback) -> uint64_t {
            std::string val;
            if (rt.IsAvailable() && rt.GetSocSpec(col, key, val)) {
                return std::stoull(val);
            }
            IR_LOGW() << "SoC spec query failed for [" << col << "]." << key << ", using default: " << fallback;
            return fallback;
        };

        // AIC (CUBE) core configuration
        Core aicCore(ir::CoreType::CUBE,
                     {Mem(ir::MemorySpace::Mat, queryU64("AICoreSpec", "l1_size", 512ULL * 1024), 128),
                      Mem(ir::MemorySpace::Left, queryU64("AICoreSpec", "l0_a_size", 64ULL * 1024), 64),
                      Mem(ir::MemorySpace::Right, queryU64("AICoreSpec", "l0_b_size", 64ULL * 1024), 64),
                      Mem(ir::MemorySpace::Scaling, queryU64("AICoreSpec", "fbuf_size", 7ULL * 1024), 32),
                      Mem(ir::MemorySpace::Acc, queryU64("AICoreSpec", "l0_c_size", 128ULL * 1024), 128)});

        // AIV (VECTOR) core configuration
        Core aivCore(ir::CoreType::VECTOR,
                     {Mem(ir::MemorySpace::Vec, queryU64("AICoreSpec", "ub_size", 192ULL * 1024), 128)});

        Cluster aicCluster(aicCore, 1); // 1 core per cluster
        Cluster aivCluster(aivCore, 1); // 1 core per cluster

        int aicCount = static_cast<int>(queryU64("SoCInfo", "cube_core_cnt", 24));
        int aivCount = static_cast<int>(queryU64("SoCInfo", "vector_core_cnt", 48));
        Die die({{aicCluster, aicCount}, {aivCluster, aivCount}});

        // Memory hierarchy graph for path finding
        std::map<ir::MemorySpace, std::vector<ir::MemorySpace>> memGraph;
        memGraph[ir::MemorySpace::DDR] = {ir::MemorySpace::Vec, ir::MemorySpace::Mat};
        memGraph[ir::MemorySpace::Vec] = {ir::MemorySpace::DDR};
        memGraph[ir::MemorySpace::Mat] = {ir::MemorySpace::Left, ir::MemorySpace::Right, ir::MemorySpace::Scaling};
        memGraph[ir::MemorySpace::Acc] = {ir::MemorySpace::Mat, ir::MemorySpace::DDR};

        return SoC(die, 1, std::move(memGraph));
    }();
    return soc;
}

} // namespace backend
} // namespace pypto
