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
 * \file event_instance_tracker.cpp
 * \brief
 */

#include "passes/block_graph_pass/insert_sync/event_instance_tracker.h"
#include <algorithm>
#include <limits>
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InsertSync"

namespace npu {
namespace tile_fwk {

Status EventInstanceTracker::RetireConsumed(size_t setIdx, size_t waitIdx)
{
    const auto* found = Find(setIdx, waitIdx);
    if (found == nullptr || found->state != InstanceState::WaitInserted) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot retire dependency %zu -> %zu without a consumed event.", setIdx,
                          waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[eventInstanceByDep_.at({setIdx, waitIdx})];
    instance.state = InstanceState::Retired;
    auto& slot = slots_[instance.slot];
    const auto* active = FindActive(slot);
    if (active != nullptr && active->id == instance.id) {
        slot.active.reset();
    }
    return SUCCESS;
}

void EventInstanceTracker::ForgetProofs()
{
    // A barrier is not permission to forward an old generation from a new set.
    // Keeping only active facts also bounds frozen proof propagation after force.
    for (auto& entry : knownConsumed_) {
        entry.second = Known(entry.first);
    }
}

const PipeSyncTypes::EventInstance* EventInstanceTracker::Find(size_t setIdx, size_t waitIdx) const
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    return it == eventInstanceByDep_.end() ? nullptr : &eventInstances_[it->second];
}

std::optional<size_t> EventInstanceTracker::FindSlot(const EventResource& resource) const
{
    const auto it = slotByResource_.find(resource);
    return it == slotByResource_.end() ? std::nullopt : std::optional<size_t>(it->second);
}

size_t EventInstanceTracker::GetOrCreateSlot(const EventResource& resource)
{
    const auto [it, inserted] = slotByResource_.try_emplace(resource, slots_.size());
    if (inserted) {
        slots_.push_back({resource, 0, std::nullopt});
    }
    return it->second;
}

const PipeSyncTypes::EventInstance* EventInstanceTracker::FindActive(const EventSlot& slot) const
{
    return slot.active ? &eventInstances_[*slot.active] : nullptr;
}

const PipeSyncTypes::EventInstance* EventInstanceTracker::FindActive(const EventResource& resource) const
{
    const auto slot = FindSlot(resource);
    return slot ? FindActive(slots_[*slot]) : nullptr;
}

void EventInstanceTracker::MergeConsumed(ConsumedInstances& destination, const ConsumedInstances& snapshot) const
{
    for (const auto& proof : snapshot) {
        const auto* active = FindActive(slots_[proof.slot]);
        if (active != nullptr && active->id.generation == proof.generation) {
            destination.insert(proof);
        }
    }
}

const PipeSyncTypes::ConsumedInstances& EventInstanceTracker::KnownForAllocationCheck(const PipeCoreRealEx& pipe) const
{
    static const ConsumedInstances empty;
    auto it = knownConsumed_.find(pipe);
    return it == knownConsumed_.end() ? empty : it->second;
}

PipeSyncTypes::ConsumedInstances EventInstanceTracker::Known(const PipeCoreRealEx& pipe) const
{
    ConsumedInstances result;
    const auto& source = KnownForAllocationCheck(pipe);
    result.reserve(source.size());
    MergeConsumed(result, source);
    return result;
}

bool EventInstanceTracker::CanAllocate(const EventResource& resource, const ConsumedInstances& consumed,
                                       bool requireActive) const
{
    const auto slot = FindSlot(resource);
    return CanAllocate(slot ? &slots_[*slot] : nullptr, consumed, requireActive);
}

bool EventInstanceTracker::CanAllocate(const EventSlot* slot, const ConsumedInstances& consumed,
                                       bool requireActive) const
{
    if (slot == nullptr) {
        return !requireActive;
    }
    if (slot->generation == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    const auto* instance = FindActive(*slot);
    if (instance == nullptr) {
        return !requireActive;
    }
    // Only the proof of the exact active generation can authorize reuse.
    return instance->state == InstanceState::WaitInserted && slot->generation == instance->id.generation &&
           consumed.count(instance->Proof()) != 0;
}

Status EventInstanceTracker::Allocate(const EventResource& resource, const PipePairEx& pipes, size_t setIdx,
                                      size_t waitIdx)
{
    return Allocate(resource, pipes, setIdx, waitIdx, Known(pipes.first));
}

Status EventInstanceTracker::Allocate(const EventResource& resource, const PipePairEx& pipes, size_t setIdx,
                                      size_t waitIdx, ConsumedInstances consumed)
{
    const auto existing = FindSlot(resource);
    if (Find(setIdx, waitIdx) != nullptr || !CanAllocate(existing ? &slots_[*existing] : nullptr, consumed, false)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot allocate event %d for dependency %zu -> %zu.", resource.eventId,
                          setIdx, waitIdx);
        return FAILED;
    }
    const size_t slotIndex = existing ? *existing : GetOrCreateSlot(resource);
    auto& slot = slots_[slotIndex];
    if (slot.active) {
        auto& previous = eventInstances_[*slot.active];
        consumed.erase(previous.Proof());
        previous.state = InstanceState::Retired;
    }
    const auto index = eventInstances_.size();
    eventInstances_.push_back({{resource, ++slot.generation},
                               slotIndex,
                               pipes,
                               setIdx,
                               waitIdx,
                               std::move(consumed),
                               nullptr,
                               nullptr,
                               InstanceState::Allocated,
                               true});
    slot.active = index;
    eventInstanceByDep_[{setIdx, waitIdx}] = index;
    return SUCCESS;
}

Status EventInstanceTracker::RecordSet(const EventResource& resource, const PipePairEx& pipes, size_t setIdx,
                                       size_t waitIdx, Operation& syncOp, bool intraCore)
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    if (!intraCore) {
        const auto slotIndex = it == eventInstanceByDep_.end() ? std::optional<size_t>(GetOrCreateSlot(resource)) :
                                                                 std::nullopt;
        if (!slotIndex || slots_[*slotIndex].generation == std::numeric_limits<uint64_t>::max()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Invalid cross-core event instance for dependency %zu -> %zu, event %d.", setIdx, waitIdx,
                              resource.eventId);
            return FAILED;
        }
        const auto index = eventInstances_.size();
        auto snapshot = Known(pipes.first);
        eventInstances_.push_back({{resource, ++slots_[*slotIndex].generation},
                                   *slotIndex,
                                   pipes,
                                   setIdx,
                                   waitIdx,
                                   std::move(snapshot),
                                   &syncOp,
                                   nullptr,
                                   InstanceState::Allocated,
                                   false});
        eventInstanceByDep_[{setIdx, waitIdx}] = index;
        return SUCCESS;
    }
    if (it == eventInstanceByDep_.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Missing allocation for set dependency %zu -> %zu.", setIdx, waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[it->second];
    if (instance.state != InstanceState::Allocated || instance.setOp != nullptr ||
        !(instance.id.resource == resource)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid set state for dependency %zu -> %zu, event %d.", setIdx,
                          waitIdx, resource.eventId);
        return FAILED;
    }
    instance.setOp = &syncOp;
    return SUCCESS;
}

Status EventInstanceTracker::Consume(size_t setIdx, size_t waitIdx, Operation& syncOp)
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    if (it == eventInstanceByDep_.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Missing event instance for wait dependency %zu -> %zu.", setIdx,
                          waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[it->second];
    if (instance.state != InstanceState::Allocated || instance.setOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid wait state for dependency %zu -> %zu.", setIdx, waitIdx);
        return FAILED;
    }
    auto& consumed = knownConsumed_[instance.pipes.second];
    consumed.EraseIf([this](const ConsumptionProof& proof) {
        const auto* active = FindActive(slots_[proof.slot]);
        return active == nullptr || active->id.generation != proof.generation;
    });
    MergeConsumed(consumed, instance.setSnapshot);
    if (instance.intraCore) {
        consumed.insert(instance.Proof());
    }
    instance.waitOp = &syncOp;
    instance.state = InstanceState::WaitInserted;
    return SUCCESS;
}

Status EventInstanceTracker::CancelPending(size_t setIdx, size_t waitIdx)
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    if (it == eventInstanceByDep_.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Missing pending dependency %zu -> %zu during cancellation.", setIdx,
                          waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[it->second];
    if (instance.state != InstanceState::Allocated) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot cancel consumed dependency %zu -> %zu.", setIdx, waitIdx);
        return FAILED;
    }
    if (instance.intraCore) {
        auto& slot = slots_[instance.slot];
        const auto* active = FindActive(slot);
        if (active == nullptr || !(active->id == instance.id)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cannot cancel dependency %zu -> %zu: its event generation does not own the active slot.",
                              setIdx, waitIdx);
            return FAILED;
        }
        slot.active.reset();
    }
    instance.state = InstanceState::Cancelled;
    eventInstanceByDep_.erase(it);
    return SUCCESS;
}

Status EventInstanceTracker::MergePending(size_t set1, size_t wait1, size_t set2, size_t wait2)
{
    const auto* first = Find(set1, wait1);
    const auto* second = Find(set2, wait2);
    if (first == nullptr || second == nullptr || first == second || first->state != InstanceState::Allocated ||
        second->state != InstanceState::Allocated || !first->intraCore || !second->intraCore ||
        first->pipes != second->pipes || set1 >= set2 || set2 >= wait1 || wait1 >= wait2 ||
        Find(set2, wait1) != nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid pending-event merge: %zu -> %zu and %zu -> %zu.", set1, wait1,
                          set2, wait2);
        return FAILED;
    }
    const auto index = eventInstanceByDep_.at({set2, wait2});
    if (CancelPending(set1, wait1) != SUCCESS) {
        return FAILED;
    }
    eventInstanceByDep_.erase({set2, wait2});
    eventInstanceByDep_[{set2, wait1}] = index;
    eventInstances_[index].waitIdx = wait1;
    return SUCCESS;
}

bool EventInstanceTracker::RetargetPendingWait(size_t setIdx, size_t waitIdx, size_t earlierWaitIdx)
{
    const auto* instance = Find(setIdx, waitIdx);
    if (instance == nullptr || !instance->intraCore || instance->state != InstanceState::Allocated ||
        instance->setOp == nullptr || instance->waitOp != nullptr || setIdx >= earlierWaitIdx ||
        earlierWaitIdx >= waitIdx || Find(setIdx, earlierWaitIdx) != nullptr) {
        return false;
    }
    const auto* active = FindActive(instance->id.resource);
    if (active == nullptr || !(active->id == instance->id) ||
        slots_[instance->slot].generation != instance->id.generation) {
        return false;
    }
    const auto index = eventInstanceByDep_.at({setIdx, waitIdx});
    eventInstanceByDep_.erase({setIdx, waitIdx});
    eventInstanceByDep_[{setIdx, earlierWaitIdx}] = index;
    eventInstances_[index].waitIdx = earlierWaitIdx;
    return true;
}

} // namespace tile_fwk
} // namespace npu
