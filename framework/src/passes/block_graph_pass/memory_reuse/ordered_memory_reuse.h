/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "connection_matrix.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

class Allocator;
class TensorBucket;
struct TensorsDesc;
class OrderedMemoryReuse;

struct OrderedReuseLifetime {
    bool supportsOrderedReuse = true;
    std::unordered_set<Operation*> producerOps;
    std::unordered_set<Operation*> lifetimeEndOps;
    uint64_t firstProducerOpIdx = ConnectionMatrix::INVALID_INDEX;
    uint64_t lastUseOpIdx = ConnectionMatrix::INVALID_INDEX;
};

class OrderedReuseBucket {
    friend class OrderedMemoryReuse;

    void Update(const OrderedReuseLifetime& next);
    bool HasNonOverlappingOrderedInterval(const OrderedReuseLifetime& next) const;

    std::unordered_set<Operation*> lifetimeEndOps_;
    uint64_t lastUseOpIdx_{ConnectionMatrix::INVALID_INDEX};
    bool supportsOrderedReuse_{false};
};

struct OrderedReuseAllocatorState {
    int64_t minimumParallelism{-1};
    OrderedMemoryReuse* active{nullptr};
    std::unordered_map<Function*, bool> communicationLeafCache;
};

// All root operations participate in reachability; only CALLs contribute to width.
class OrderedReuseGraph {
public:
    using Endpoints = std::unordered_set<Operation*>;
    bool Init(Function& function);
    bool AllReachable(const Endpoints& ends, const Endpoints& producers) const;
    // Transactional: rejection leaves the entire graph, matching and witness unchanged.
    bool TryOrder(const Endpoints& ends, const Endpoints& producers, size_t floor);
    bool SameCallOrder(const OrderedReuseGraph& other) const;
    size_t Width() const { return calls_.size() - matched_; }
    const std::vector<Operation*>& Operations() const { return operations_; }

private:
    bool AddEdge(Operation* from, Operation* to);
    bool RefreshReachability();
    bool Reaches(size_t from, size_t to) const;
    bool Order(const Endpoints& ends, const Endpoints& producers);
    bool ProveWidth(size_t floor);
    bool WitnessSurvives() const;
    void FindAntichain(const std::vector<std::vector<size_t>>& edges);

    std::vector<Operation*> operations_;
    std::unordered_map<Operation*, size_t> indices_;
    std::vector<size_t> indegrees_;
    std::vector<std::set<size_t>> successors_;
    std::vector<LargeBitmap> ancestors_;
    std::vector<size_t> calls_;
    std::vector<size_t> leftMatch_;
    std::vector<size_t> rightMatch_;
    std::vector<size_t> antichain_;
    size_t matched_{0};
};

class OrderedMemoryReuse {
public:
    explicit OrderedMemoryReuse(int64_t minimum) : floor_(static_cast<size_t>(std::max<int64_t>(1, minimum))) {}
    Status Build(Allocator& allocator);
    TensorBucket& GetBestFitBucket(Allocator& allocator, const TensorsDesc& next, int64_t minimumSize);
    static TensorBucket& GetBestFitNaturalBucket(Allocator& allocator, const TensorsDesc& next, int64_t minimumSize,
                                                 const std::unordered_set<int>& visited);
    static uint64_t CollectLifetime(Allocator& allocator, TensorsDesc& tensorsDesc, Operation& op, bool isProducer);

private:
    using Endpoints = OrderedReuseGraph::Endpoints;
    struct AddedToken {
        Operation* end;
        ir::VarPtr token;
        Endpoints consumers;
    };
    static void BuildBuckets(Allocator& allocator);
    static Status Preflight(const Allocator& allocator);
    // SHMEM calls and unresolved callees cannot introduce ordering across ranks.
    static bool IsCommCallOp(Allocator& allocator, const Operation& op);
    bool NaturallyOrdered(const TensorBucket& bucket, const TensorsDesc& next) const;
    bool TryAddDependencies(const TensorBucket& bucket, const TensorsDesc& next);
    bool Materialize(Function& function);
    void Rollback(Function& function);
    static uint64_t Bytes(const std::vector<TensorBucket>& buckets, const TensorBucket& dummy);

    size_t floor_;
    OrderedReuseGraph graph_;
    std::unordered_map<Operation*, Endpoints> addedConsumers_;
    std::vector<AddedToken> addedTokens_;
};
} // namespace npu::tile_fwk
