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
 * \file event_instance_tracker.h
 * \brief Tracks event instance state and validates reuse within InsertSync.
 */

#ifndef PASS_INSERT_SYNC_EVENT_INSTANCE_TRACKER_H
#define PASS_INSERT_SYNC_EVENT_INSTANCE_TRACKER_H

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include "passes/block_graph_pass/insert_sync/sync_types.h"

namespace npu {
namespace tile_fwk {
// Owns all event-instance transitions; the scheduler only reads instances.
class EventInstanceTracker {
    using ConsumptionProof = PipeSyncTypes::ConsumptionProof;
    using ConsumedInstances = PipeSyncTypes::ConsumedInstances;
    using EventInstance = PipeSyncTypes::EventInstance;
    using EventResource = PipeSyncTypes::EventResource;
    using EventResourceHash = PipeSyncTypes::EventResourceHash;
    using IndexVecHash = PipeSyncTypes::IndexVecHash;
    using InstanceState = PipeSyncTypes::InstanceState;
    using PipeCoreRealEx = PipeSyncTypes::PipeCoreRealEx;
    using PipeCoreRealExHash = PipeSyncTypes::PipeCoreRealExHash;
    using PipePairEx = PipeSyncTypes::PipePairEx;

public:
    const std::vector<EventInstance>& Instances() const { return eventInstances_; }
    const EventInstance* Find(size_t setIdx, size_t waitIdx) const;
    const EventInstance* FindActive(const EventResource& resource) const;
    ConsumedInstances Known(const PipeCoreRealEx& pipe) const;
    const ConsumedInstances& KnownForAllocationCheck(const PipeCoreRealEx& pipe) const;
    void MergeConsumed(ConsumedInstances& destination, const ConsumedInstances& snapshot) const;
    bool HasActive(const EventResource& resource) const { return FindActive(resource) != nullptr; }
    const EventResource& ProofResource(ConsumptionProof proof) const { return slots_[proof.slot].resource; }
    bool CanAllocate(const EventResource& resource, const ConsumedInstances& consumed,
                     bool requireActive = false) const;
    Status Allocate(const EventResource& resource, const PipePairEx& pipes, size_t setIdx, size_t waitIdx);
    Status Allocate(const EventResource& resource, const PipePairEx& pipes, size_t setIdx, size_t waitIdx,
                    ConsumedInstances consumed);
    Status RecordSet(const EventResource& resource, const PipePairEx& pipes, size_t setIdx, size_t waitIdx,
                     Operation& syncOp, bool intraCore);
    Status Consume(size_t setIdx, size_t waitIdx, Operation& syncOp);
    Status CancelPending(size_t setIdx, size_t waitIdx);
    Status MergePending(size_t set1, size_t wait1, size_t set2, size_t wait2);
    bool RetargetPendingWait(size_t setIdx, size_t waitIdx, size_t earlierWaitIdx);
    Status RetireConsumed(size_t setIdx, size_t waitIdx);
    void ForgetProofs();

private:
    struct EventSlot {
        EventResource resource;
        uint64_t generation{0};
        std::optional<size_t> active;
    };
    std::optional<size_t> FindSlot(const EventResource& resource) const;
    size_t GetOrCreateSlot(const EventResource& resource);
    const EventInstance* FindActive(const EventSlot& slot) const;
    bool CanAllocate(const EventSlot* slot, const ConsumedInstances& consumed, bool requireActive) const;

    std::vector<EventInstance> eventInstances_;
    // Indices remain valid when the tracker is copied for a staged drain.
    std::vector<EventSlot> slots_;
    std::unordered_map<EventResource, size_t, EventResourceHash> slotByResource_;
    std::unordered_map<PipeCoreRealEx, ConsumedInstances, PipeCoreRealExHash> knownConsumed_;
    std::unordered_map<std::pair<size_t, size_t>, size_t, IndexVecHash> eventInstanceByDep_;
};
} // namespace tile_fwk
} // namespace npu

#endif // PASS_INSERT_SYNC_EVENT_INSTANCE_TRACKER_H
