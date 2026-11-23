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
 * \file event_allocation.cpp
 * \brief
 */

#include "passes/block_graph_pass/insert_sync/insert_sync.h"
#include <algorithm>
#include <functional>
#include <thread>
#include <unordered_set>
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_common_defs.h"

#define MODULE_NAME "InsertSync"

namespace npu {
namespace tile_fwk {

// Event allocation and capacity recovery

std::deque<int>* PipeSync::GetCrossCoreEventIdQPtr(const PipePairEx& pp)
{
    if (pp.first.core == CoreType::AIV) {
        return &crossCoreFreeEventId_[pp.first][0];
    }
    if (pp.second.aivCore == AIVCore::AIV0) {
        return &crossCoreFreeEventId_[pp.first][0];
    }
    if (pp.second.aivCore != AIVCore::AIV1) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV op is neither AIV0 nor AIV1, GetEventId failed.");
        return nullptr;
    }
    return &crossCoreFreeEventId_[pp.first][1];
}

void PipeSync::RemoveEventIdFromCrossCoreQueues(int eventId)
{
    for (auto& [otherP, otherQArr] : crossCoreFreeEventId_) {
        (void)otherP;
        for (auto& otherQ : otherQArr) {
            otherQ.erase(std::remove(otherQ.begin(), otherQ.end(), eventId), otherQ.end());
        }
    }
}

size_t PipeSync::FindReclaimPrefix(size_t boundary, const std::unordered_set<Operation*>& emitted) const
{
    for (size_t i = 0; i < boundary; ++i) {
        if (emitted.count(oriOpList_[i]) == 0) {
            boundary = i;
            break;
        }
    }
    // A prefix wait must never depend on a suffix set: the barrier would
    // otherwise block the producer needed to finish draining that prefix.
    bool moved = true;
    while (moved && boundary != 0) {
        moved = false;
        for (size_t i = boundary; i < depOps_.size(); ++i) {
            for (auto wait : depOps_[i].setPipe) {
                if (wait < boundary) {
                    boundary = wait;
                    moved = true;
                }
            }
        }
    }
    return boundary;
}

Status PipeSync::StageReclaimedEvents(size_t boundary, const PipePairEx& pp, bool lite,
                                      const std::unordered_set<Operation*>& emitted, EventInstanceTracker& stagedState,
                                      std::vector<size_t>& covered) const
{
    for (size_t i = 0; i < eventState_.Instances().size(); ++i) {
        const auto& instance = eventState_.Instances()[i];
        if (!instance.intraCore || instance.state != InstanceState::WaitInserted ||
            (!lite &&
             (instance.pipes.first.core != pp.first.core || instance.pipes.first.aivCore != pp.first.aivCore ||
              instance.pipes.second.core != pp.first.core || instance.pipes.second.aivCore != pp.first.aivCore)) ||
            instance.setIdx >= boundary || instance.waitIdx >= boundary) {
            continue;
        }
        if (emitted.count(instance.setOp) == 0 || emitted.count(instance.waitOp) == 0 ||
            stagedState.RetireConsumed(instance.setIdx, instance.waitIdx) != SUCCESS) {
            return FAILED;
        }
        covered.push_back(i);
    }
    return SUCCESS;
}

bool PipeSync::HasReclaimCapacity(size_t blockedIdx, size_t boundary, const PipePairEx& pp,
                                  const EventInstanceTracker& stagedState, size_t covered)
{
    // Pending sets are deliberately preserved: a drain does not consume flags
    // whose waits are still in the suffix, and cannot authorize their reuse.
    const auto consumed = ProjectConsumedInstances(depOps_[blockedIdx], pp.first);
    size_t available = 0;
    for (int id = INTRA_CORE_START_EVENT_ID; id <= GetMaxEventId(pp); ++id) {
        available += stagedState.CanAllocate(GetEventResource(pp, id), consumed) ? 1 : 0;
    }
    size_t required = 0;
    const auto& source = depOps_[blockedIdx].selfPipeCore;
    for (auto wait : depOps_[blockedIdx].setPipe) {
        const auto& target = depOps_[wait].selfPipeCore;
        const PipePairEx pair{{source.pipeEnd, source.core, source.aivCore},
                              {target.pipeStart, target.core, target.aivCore}};
        required += pair == pp ? 1 : 0;
    }
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "Event reuse prefix drain: blocked=%zu, boundary=%zu, covered=%zu, available=%zu, required=%zu.",
                      blockedIdx, boundary, covered, available, required);
    return covered != 0 && available >= required;
}

Status PipeSync::ReclaimIntraCoreEvents(Function& function, size_t boundary, const PipePairEx& pp,
                                        std::vector<IndexOp>& syncedOpLog)
{
    const bool lite = IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch());
    if (boundary == 0 || boundary >= depOps_.size() || !IsIntraCoreEvent(pp) ||
        (!lite && (pp.first.core != pp.second.core || pp.first.aivCore != pp.second.aivCore))) {
        return FAILED;
    }
    const auto blockedIdx = boundary;
    // Replay may issue a later producer before an earlier consumer. Drain only
    // a complete original-order prefix, not the entire issued scheduling prefix.
    std::unordered_set<Operation*> emitted;
    for (const auto& entry : syncedOpLog) {
        emitted.insert(&entry.second.get());
    }
    boundary = FindReclaimPrefix(boundary, emitted);
    if (boundary == 0) {
        return FAILED;
    }
    auto stagedState = eventState_;
    std::vector<size_t> covered;
    if (StageReclaimedEvents(boundary, pp, lite, emitted, stagedState, covered) != SUCCESS) {
        return FAILED;
    }
    if (!HasReclaimCapacity(blockedIdx, boundary, pp, stagedState, covered.size())) {
        return FAILED;
    }
    uint64_t insertIdx = boundary * SEQUENCE_IDX - RECLAIM_PREFIX_OFFSET;
    for (const auto& entry : syncedOpLog) {
        if (entry.first > insertIdx && entry.first < boundary * SEQUENCE_IDX - HALF_SEQUENCE_IDX) {
            insertIdx = entry.first;
        }
    }
    if (insertIdx + 1 >= boundary * SEQUENCE_IDX - HALF_SEQUENCE_IDX) {
        return FAILED;
    }
    return CommitIntraCoreReclaim(function, boundary, pp, insertIdx, stagedState, covered, syncedOpLog);
}

Status PipeSync::GetCrossCoreBarrierBoundary(size_t idx, size_t instructionCount,
                                             const std::vector<IndexOp>& syncedOpLog, uint64_t& insertIdx) const
{
    if (idx == 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot insert a cross-core barrier before the first operation.");
        return FAILED;
    }
    std::unordered_set<const Operation*> emitted;
    insertIdx = 0;
    for (const auto& entry : syncedOpLog) {
        // EnqueueOp emits future VIEW/NOP operations eagerly. They do not belong to this prefix.
        if (entry.first < idx * SEQUENCE_IDX) {
            emitted.insert(&entry.second.get());
            insertIdx = std::max(insertIdx, entry.first);
        }
    }
    for (size_t i = 0; i < idx; ++i) {
        if (emitted.count(oriOpList_[i]) == 0) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cross-core barrier before op %zu requires a complete emitted prefix; op %zu is missing.",
                              idx, i);
            return FAILED;
        }
    }
    const uint64_t waitAnchor = idx * SEQUENCE_IDX - HALF_SEQUENCE_IDX;
    if (insertIdx >= waitAnchor || instructionCount >= waitAnchor - insertIdx) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Insufficient sequence space before op %zu: need %zu slots, insertion %llu, wait anchor %llu.", idx,
            instructionCount, static_cast<unsigned long long>(insertIdx), static_cast<unsigned long long>(waitAnchor));
        return FAILED;
    }
    for (const auto& pending : NoWaitCVPairs_) {
        if (pending.first.first >= idx || pending.first.second < idx) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Pending cross-core edge %zu -> %zu does not cross the barrier before op %zu.",
                              pending.first.first, pending.first.second, idx);
            return FAILED;
        }
    }
    return SUCCESS;
}

bool PipeSync::UseCrossCoreProof() const
{
    // A5 opts into proof-based reuse and the local handshake. Other architectures keep their legacy policy.
    return enableLocalCrossCoreBarrier_ && Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
}

CoreType PipeSync::GetCrossCoreHandshakeSender(int eventId) const
{
    // A pending set is removed by the drain. Its direction must not replace
    // the last signal whose wait remains in the emitted prefix.
    return crossCoreCompletedSenders_[eventId].value_or(*crossCoreEventSenders_[eventId]);
}

void PipeSync::MergeCrossCoreProof(CrossCoreClock& clock, const CrossCoreSnapshot& snapshot)
{
    for (size_t id = 0; id < clock.size(); ++id) {
        clock[id] = std::max(clock[id], snapshot.consumed[id]);
    }
    if (snapshot.eventId >= 0) {
        clock[snapshot.eventId] = std::max(clock[snapshot.eventId], snapshot.generation);
    }
}

PipeSync::CrossCoreClock PipeSync::ProjectCrossCoreConsumption(size_t idx) const
{
    const auto& current = depOps_[idx].selfPipeCore;
    const PipeCoreRealEx source(current.pipeEnd, current.core, current.aivCore);
    CrossCoreClock clock{};
    auto known = crossCoreConsumed_.find(source);
    if (known != crossCoreConsumed_.end()) {
        clock = known->second;
    }
    // A multi-pipe operation does not establish entry-pipe -> exit-pipe FIFO order.
    if (current.pipeStart == current.pipeEnd) {
        for (size_t producer : depOps_[idx].waitPipe) {
            auto snapshot = crossCoreSnapshots_.find({producer, idx});
            if (snapshot != crossCoreSnapshots_.end()) {
                MergeCrossCoreProof(clock, snapshot->second);
            }
        }
    }
    return clock;
}

bool PipeSync::CanReuseCrossCoreEventId(int eventId, const CrossCoreClock& clock) const
{
    return crossCoreGenerations_[eventId] != std::numeric_limits<uint64_t>::max() &&
           (!crossCoreEventSenders_[eventId].has_value() ||
            (crossCoreGenerations_[eventId] != 0 && clock[eventId] == crossCoreGenerations_[eventId]));
}

bool PipeSync::CanOrderCrossCoreWaits(int eventId, size_t idx, size_t pool) const
{
    if (crossCoreWaitGenerations_[0][eventId] == 0 && crossCoreWaitGenerations_[1][eventId] == 0) {
        return true; // No earlier wait in either direction needs ordering.
    }
    const auto& current = depOps_[idx].selfPipeCore;
    for (size_t consumer : depOps_[idx].setPipe) {
        const auto& target = depOps_[consumer].selfPipeCore;
        if (current.core == target.core) {
            continue;
        }
        const auto vector = current.core == CoreType::AIV ? current.aivCore : target.aivCore;
        if ((vector == AIVCore::AIV0 ? 0U : 1U) != pool) {
            continue;
        }
        const auto generation = crossCoreWaitGenerations_[static_cast<size_t>(target.core)][eventId];
        if (generation == 0) {
            continue;
        }
        const PipeCoreRealEx receivingPipe(target.pipeStart, target.core, target.aivCore);
        const auto known = crossCoreConsumed_.find(receivingPipe);
        // Do not use the new set->wait edge to prove that wait's own ordering.
        // Otherwise a different receiving pipe can consume an older signal.
        if (known == crossCoreConsumed_.end() || known->second[eventId] < generation) {
            return false;
        }
    }
    return true;
}

void PipeSync::RefreshCrossCoreEventIds(const PipeCoreRealEx& source, size_t pool, const CrossCoreClock& clock,
                                        size_t idx)
{
    auto& queue = crossCoreFreeEventId_[source][source.core == CoreType::AIC ? pool : 0];
    const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
    queue.erase(std::remove_if(queue.begin(), queue.end(),
                               [&](int id) {
                                   return id < base || id >= base + CROSS_CORE_EVENT_NUM ||
                                          !CanReuseCrossCoreEventId(id, clock) ||
                                          !CanOrderCrossCoreWaits(id, idx, pool);
                               }),
                queue.end());
    // Keep fresh IDs in FIFO order. Reusable IDs are appended deterministically.
    for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
        if (crossCoreEventSenders_[id].has_value() && CanReuseCrossCoreEventId(id, clock) &&
            CanOrderCrossCoreWaits(id, idx, pool)) {
            PushEventIdIfAbsent(queue, id);
        }
    }
}

Status PipeSync::PlanCrossCoreBarriers(size_t idx, const std::array<size_t, NUM2>& demand, CrossCoreBarrierPlan& plan)
{
    const auto& current = depOps_[idx].selfPipeCore;
    const PipeCoreRealEx source(current.pipeEnd, current.core, current.aivCore);
    const auto projected = ProjectCrossCoreConsumption(idx);
    size_t exhaustedPools = 0;
    for (size_t pool = 0; pool < demand.size(); ++pool) {
        if (demand[pool] > CROSS_CORE_EVENT_NUM) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Operation %zu requires %zu cross-core IDs from pool %zu of size %d.", idx, demand[pool],
                              pool, CROSS_CORE_EVENT_NUM);
            return FAILED;
        }
        if (demand[pool] == 0) {
            continue;
        }
        RefreshCrossCoreEventIds(source, pool, projected, idx);
        if (demand[pool] <= crossCoreFreeEventId_[source][current.core == CoreType::AIC ? pool : 0].size()) {
            continue;
        }
        ++plan.localCount;
        ++exhaustedPools;
        const bool scalarWaits = HasScalarCrossCoreWaits(idx, pool);
        const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
        for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
            if (!crossCoreEventSenders_[id].has_value() ||
                crossCoreGenerations_[id] > std::numeric_limits<uint64_t>::max() - NUM2) {
                continue;
            }
            // H remains available only on its final receiving S pipe, ordered after the final wait.
            const bool canUseHandshake = source.pipe == PIPE_S && source.core != GetCrossCoreHandshakeSender(id) &&
                                         scalarWaits;
            if (demand[pool] <= CROSS_CORE_EVENT_NUM - 1 + static_cast<size_t>(canUseHandshake)) {
                plan.handshakeIds[pool] = id;
                break;
            }
        }
        plan.globalBarrier = plan.globalBarrier || plan.handshakeIds[pool] == -1;
    }
    // A single local handshake is scoped to one C/V pool. If both pools are
    // exhausted for this operation, drain them together with the legacy
    // cross-core barrier instead of emitting two independent local barriers.
    plan.globalBarrier = plan.globalBarrier || exhaustedPools > 1;
    return SUCCESS;
}

Status PipeSync::EnsureCrossCoreCapacity(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    std::array<size_t, NUM2> demand{};
    if (CollectCrossCoreDemand(idx, demand) != SUCCESS) {
        return FAILED;
    }
    if (demand[0] == 0 && demand[1] == 0) {
        return SUCCESS;
    }
    // A barrier can cancel an input wait that carried proof for the other pool. Recompute after each reset.
    for (size_t attempt = 0;; ++attempt) {
        CrossCoreBarrierPlan plan;
        if (PlanCrossCoreBarriers(idx, demand, plan) != SUCCESS) {
            return FAILED;
        }
        if (plan.localCount == 0) {
            return SUCCESS;
        }
        if (attempt == NUM2) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cross-core capacity still insufficient at op %zu after %zu drains.",
                              idx, attempt);
            return FAILED;
        }
        if (ApplyCrossCoreBarrierPlan(function, idx, demand, plan, syncedOpLog) != SUCCESS) {
            return FAILED;
        }
    }
}

Status PipeSync::RecordCrossCoreDrain(size_t boundary, const std::vector<IndexOp>& syncedOpLog, ReclaimBarrier barrier)
{
    if (barrier.ops.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cross-core drain before op %zu has no instructions.", boundary);
        return FAILED;
    }
    const auto first = std::find_if(syncedOpLog.begin(), syncedOpLog.end(),
                                    [&](const IndexOp& entry) { return &entry.second.get() == barrier.ops.front(); });
    if (first == syncedOpLog.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cross-core drain before op %zu is missing its first instruction.",
                          boundary);
        return FAILED;
    }
    std::unordered_set<Operation*> emitted;
    for (const auto& entry : syncedOpLog) {
        if (entry.first < first->first) {
            emitted.insert(&entry.second.get());
        }
    }
    std::vector<size_t> covered;
    for (size_t i = 0; i < eventState_.Instances().size(); ++i) {
        const auto& instance = eventState_.Instances()[i];
        const CoreTypeDetail core{instance.pipes.first.core, instance.pipes.first.aivCore};
        if (!instance.intraCore || instance.state != InstanceState::WaitInserted || instance.setIdx >= boundary ||
            instance.waitIdx >= boundary ||
            std::find(barrier.cores.begin(), barrier.cores.end(), core) == barrier.cores.end()) {
            continue;
        }
        if (emitted.count(instance.setOp) == 0 || emitted.count(instance.waitOp) == 0) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cross-core drain before op %zu is missing the emitted set or wait for %zu -> %zu.",
                              boundary, instance.setIdx, instance.waitIdx);
            return FAILED;
        }
        covered.push_back(i);
    }
    // A cancelled cross-core input may have carried intra-core capacity proof.
    // The existing full drain covers completed waits on these cores instead;
    // pending flags and consumed instances on other cores remain occupied.
    for (auto index : covered) {
        const auto& instance = eventState_.Instances()[index];
        if (eventState_.RetireConsumed(instance.setIdx, instance.waitIdx) != SUCCESS) {
            return FAILED;
        }
        PushEventIdIfAbsent(GetFreeEventIdQueue(instance.pipes), instance.id.resource.eventId);
    }
    reclaimBarriers_.push_back(std::move(barrier));
    return SUCCESS;
}

Status PipeSync::RemovePendingCrossCoreSyncs(size_t pool, std::vector<IndexOp>& syncedOpLog)
{
    const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
    auto inPool = [base](int id) { return id >= base && id < base + CROSS_CORE_EVENT_NUM; };
    auto stagedState = eventState_;
    std::unordered_set<Operation*> removedSets;
    for (const auto& [edge, setOp] : NoWaitCVPairs_) {
        if (!inPool(setOp->syncQueue_.eventId_)) {
            continue;
        }
        if (stagedState.CancelPending(edge.first, edge.second) != SUCCESS) {
            return FAILED;
        }
        removedSets.insert(setOp);
    }
    eventState_ = std::move(stagedState);
    ErasePendingCrossCoreSets(removedSets);
    syncedOpLog.erase(std::remove_if(syncedOpLog.begin(), syncedOpLog.end(),
                                     [&](const IndexOp& entry) { return removedSets.count(&entry.second.get()) != 0; }),
                      syncedOpLog.end());
    return SUCCESS;
}

Status PipeSync::ResetCrossCorePool(size_t pool, int handshakeId, std::vector<IndexOp>& syncedOpLog)
{
    if (RemovePendingCrossCoreSyncs(pool, syncedOpLog) != SUCCESS) {
        return FAILED;
    }
    const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
    const auto vector = pool == 0 ? AIVCore::AIV0 : AIVCore::AIV1;
    const auto handshakeSender = GetCrossCoreHandshakeSender(handshakeId);
    const auto finalReceiver = handshakeSender == CoreType::AIC ? CoreType::AIV : CoreType::AIC;
    crossCoreEventSenders_[handshakeId] = handshakeSender;
    crossCoreCompletedSenders_[handshakeId] = handshakeSender;
    auto clearPool = [base](CrossCoreClock& clock) {
        std::fill(clock.begin() + base, clock.begin() + base + CROSS_CORE_EVENT_NUM, 0);
    };
    for (auto& clock : crossCoreWaitGenerations_) {
        clearPool(clock);
    }
    for (auto& [pipe, clock] : crossCoreConsumed_) {
        (void)pipe;
        clearPool(clock);
    }
    for (auto& [edge, snapshot] : crossCoreSnapshots_) {
        (void)edge;
        clearPool(snapshot.consumed);
    }
    crossCoreGenerations_[handshakeId] += NUM2;
    // Both handshake waits remain live; retain their ordering on the receiving S pipes.
    for (auto core : {handshakeSender, finalReceiver}) {
        const auto generation = crossCoreGenerations_[handshakeId] - (core == handshakeSender ? 1U : 0U);
        crossCoreWaitGenerations_[static_cast<size_t>(core)][handshakeId] = generation;
        const PipeCoreRealEx receivingPipe(PIPE_S, core, core == CoreType::AIV ? vector : AIVCore::UNSPECIFIED);
        crossCoreConsumed_[receivingPipe][handshakeId] = generation;
    }
    RefillCrossCorePool(pool, handshakeId, finalReceiver);
    return SUCCESS;
}

Status PipeSync::CollectCrossCoreDemand(size_t idx, std::array<size_t, NUM2>& demand) const
{
    const auto& current = depOps_[idx].selfPipeCore;
    for (size_t consumer : depOps_[idx].setPipe) {
        const auto& target = depOps_[consumer].selfPipeCore;
        if (current.core == target.core) {
            continue;
        }
        const auto vector = current.core == CoreType::AIV ? current.aivCore : target.aivCore;
        if (vector != AIVCore::AIV0 && vector != AIVCore::AIV1) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cross-core dependency %zu -> %zu has invalid AIV core %d; expected AIV0 or AIV1.", idx,
                              consumer, static_cast<int>(vector));
            return FAILED;
        }
        ++demand[vector == AIVCore::AIV0 ? 0 : 1];
    }
    return SUCCESS;
}

Status PipeSync::ApplyCrossCoreBarrierPlan(Function& function, size_t idx, const std::array<size_t, NUM2>& demand,
                                           const CrossCoreBarrierPlan& plan, std::vector<IndexOp>& syncedOpLog)
{
    constexpr size_t localInstructionCount = 6;
    constexpr size_t globalInstructionCount = 12;
    uint64_t insertIdx = 0;
    if (GetCrossCoreBarrierBoundary(
            idx, plan.globalBarrier ? globalInstructionCount : plan.localCount * localInstructionCount, syncedOpLog,
            insertIdx) != SUCCESS) {
        return FAILED;
    }
    if (plan.globalBarrier) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Inserting global cross-core barrier before op %zu: %s; pool demand %zu/%zu.", idx,
            plan.localCount > 1 ? "both pools exhausted" : "no eligible handshake ID", demand[0], demand[1]);
        if (AddGlobalCrossCoreBarrier(function, idx, insertIdx, syncedOpLog) != SUCCESS) {
            return FAILED;
        }
        return SUCCESS;
    }
    for (size_t pool = 0; pool < plan.handshakeIds.size(); ++pool) {
        if (plan.handshakeIds[pool] == -1) {
            continue;
        }
        if (AddLocalCrossCoreBarrier(function, idx, pool, plan.handshakeIds[pool], insertIdx, syncedOpLog) != SUCCESS) {
            return FAILED;
        }
        if (ResetCrossCorePool(pool, plan.handshakeIds[pool], syncedOpLog) != SUCCESS) {
            return FAILED;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Inserted 1C1V barrier before op %zu for pool %zu with event ID %d.",
                          idx, pool, plan.handshakeIds[pool]);
    }
    return SUCCESS;
}

Status PipeSync::CommitIntraCoreReclaim(Function& function, size_t boundary, const PipePairEx& pp, uint64_t insertIdx,
                                        EventInstanceTracker& stagedState, const std::vector<size_t>& covered,
                                        std::vector<IndexOp>& syncedOpLog)
{
    std::vector<IndexOp> generated;
    CreateForceSyncOp(Opcode::OP_BAR_ALL, PipeType::PIPE_ALL, pp.first.core, pp.first.aivCore, insertIdx, function,
                      generated);
    ReclaimBarrier barrier;
    barrier.cores = {{pp.first.core, pp.first.aivCore}};
    barrier.ops = {&generated.front().second.get()};
    stagedState.ForgetProofs();
    eventState_ = std::move(stagedState);
    for (auto index : covered) {
        const auto& instance = eventState_.Instances()[index];
        auto& ids = GetFreeEventIdQueue(instance.pipes);
        if (std::find(ids.begin(), ids.end(), instance.id.resource.eventId) == ids.end()) {
            ids.push_back(instance.id.resource.eventId);
        }
    }
    syncedOpLog.insert(syncedOpLog.end(), generated.begin(), generated.end());
    reclaimBarriers_.push_back(std::move(barrier));
    APASS_LOG_INFO_F(Elements::Operation, "Drained core %d/%d before op %zu; reclaimed %zu consumed intra-core events.",
                     static_cast<int>(pp.first.core), static_cast<int>(pp.first.aivCore), boundary, covered.size());
    return SUCCESS;
}

void PipeSync::RefillCrossCorePool(size_t pool, int handshakeId, CoreType finalReceiver)
{
    const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
    const auto vector = pool == 0 ? AIVCore::AIV0 : AIVCore::AIV1;
    for (const auto& pipe : cvPipeCoreEx) {
        if (pipe.core != CoreType::AIC && pipe.aivCore != vector) {
            continue;
        }
        auto& queue = crossCoreFreeEventId_[pipe][pipe.core == CoreType::AIC ? pool : 0];
        queue.clear();
        for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
            if (id != handshakeId || (pipe.core == finalReceiver && pipe.pipe == PIPE_S)) {
                queue.push_back(id);
            }
        }
    }
    for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
        if (id != handshakeId) {
            crossCoreEventSenders_[id].reset();
            crossCoreCompletedSenders_[id].reset();
        }
    }
}

bool PipeSync::HasScalarCrossCoreWaits(size_t idx, size_t pool) const
{
    const auto& current = depOps_[idx].selfPipeCore;
    return std::all_of(depOps_[idx].setPipe.begin(), depOps_[idx].setPipe.end(), [&](size_t consumer) {
        const auto& target = depOps_[consumer].selfPipeCore;
        const auto vector = current.core == CoreType::AIV ? current.aivCore : target.aivCore;
        return current.core == target.core || target.pipeStart == PIPE_S || (vector == AIVCore::AIV0 ? 0U : 1U) != pool;
    });
}

Status PipeSync::RecoverCrossCoreEventQueue(const PipePairEx& pp, int& eventId, size_t setIdx,
                                            std::vector<IndexOp>& syncedOpLog, Function& function)
{
    // insert ffts to relax all eventid
    if (RemoveSetIntraBlockAndOpDep(syncedOpLog) != SUCCESS) {
        return FAILED;
    }
    AddCrossCoreForceSyncOps(setIdx, syncedOpLog, function);
    InitCVEventIdQ();
    std::deque<int>* newCvEventIdQPtr = GetCrossCoreEventIdQPtr(pp);
    if (newCvEventIdQPtr == nullptr || newCvEventIdQPtr->empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "EventId still exhausted after InitCVEventIdQ, GetEventId failed.");
        return FAILED;
    }
    eventId = newCvEventIdQPtr->front();
    newCvEventIdQPtr->pop_front();
    RemoveEventIdFromCrossCoreQueues(eventId);
    return SUCCESS;
}

Status PipeSync::AllocateCrossCoreEvent(const PipePairEx& pp, int& eventId, size_t setIdx,
                                        std::vector<IndexOp>& syncedOpLog, Function& function)
{
    std::deque<int>* cvEventIdQPtr = GetCrossCoreEventIdQPtr(pp);
    if (cvEventIdQPtr == nullptr) {
        return FAILED;
    }
    auto& cvEventIdQ = *cvEventIdQPtr;
    if (UseCrossCoreProof()) {
        const size_t pool = (pp.first.core == CoreType::AIV ? pp.first.aivCore : pp.second.aivCore) == AIVCore::AIV0 ?
                                0 :
                                1;
        RefreshCrossCoreEventIds(pp.first, pool, crossCoreConsumed_[pp.first], setIdx);
        if (cvEventIdQ.empty()) {
            // InjectSync reserves the whole output group before emitting input waits.
            // Draining here could cancel sets after those waits have already committed.
            APASS_LOG_ERROR_F(Elements::Operation, "Cross-core ID reservation is exhausted at op %zu.", setIdx);
            return FAILED;
        }
        eventId = cvEventIdQ.front();
        ++crossCoreGenerations_[eventId];
        crossCoreEventSenders_[eventId] = pp.first.core;
    }
    if (!cvEventIdQ.empty()) {
        eventId = cvEventIdQ.front();
        cvEventIdQ.pop_front();
        RemoveEventIdFromCrossCoreQueues(eventId);
        return SUCCESS;
    }

    return RecoverCrossCoreEventQueue(pp, eventId, setIdx, syncedOpLog, function);
}

bool PipeSync::IsIntraCoreEvent(const PipePairEx& pp) const
{
    return pp.first.core == pp.second.core || IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch());
}

PipeSync::EventResource PipeSync::GetEventResource(const PipePairEx& pp, int eventId) const
{
    return {
        eventId, {pp.first.core, pp.first.aivCore}, {pp.second.core, pp.second.aivCore}, pp.first.pipe, pp.second.pipe};
}

PipeSync::ConsumedInstances PipeSync::ProjectConsumedInstances(const DepOp& op, const PipeCoreRealEx& source) const
{
    auto result = eventState_.Known(source);
    const PipeCoreRealEx entry(op.selfPipeCore.pipeStart, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    if (source != entry) {
        // A multi-pipe op does not prove an entry-to-exit hardware ordering edge.
        return result;
    }
    // This is a preview only. InjectWaitFlag must commit these same waits before
    // allocation; PopFromQueue stops batches at waits so earlier proofs are committed.
    for (auto setIdx : op.waitPipe) {
        const auto* instance = eventState_.Find(setIdx, op.idx);
        if (instance == nullptr || instance->state != InstanceState::Allocated || instance->setOp == nullptr) {
            continue;
        }
        eventState_.MergeConsumed(result, instance->setSnapshot);
        if (instance->intraCore) {
            result.insert(instance->Proof());
        }
    }
    return result;
}

PipeSync::IntraEventIds PipeSync::GetAvailableIntraEventIds(const PipePairEx& pp, const ConsumedInstances& consumed,
                                                            size_t limit)
{
    IntraEventIds available;
    std::array<bool, EVENT_NUM> seen{};
    const int maxId = GetMaxEventId(pp);
    auto appendIfAvailable = [&](int id, bool requireActive) {
        if (id >= INTRA_CORE_START_EVENT_ID && id <= maxId && !seen[id]) {
            seen[id] = true;
            if (eventState_.CanAllocate(GetEventResource(pp, id), consumed, requireActive)) {
                available.ids[available.count++] = id;
            }
        }
        return available.count == limit;
    };
    for (int id : GetFreeEventIdQueue(pp)) {
        if (appendIfAvailable(id, false)) {
            return available;
        }
    }
    for (int id = INTRA_CORE_START_EVENT_ID; id <= maxId; ++id) {
        if (appendIfAvailable(id, true)) {
            break;
        }
    }
    return available;
}

std::optional<PipeSync::IntraEventCapacityFailure> PipeSync::AnalyzeIntraEventCapacity(const DepOp& op)
{
    if (op.setPipe.empty()) {
        return std::nullopt;
    }
    const PipeCoreRealEx source(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    std::optional<ConsumedInstances> consumed;
    // Neither the proof nor the capacity changes while checking this operation.
    std::unordered_map<PipePairEx, std::pair<size_t, size_t>, PipePairExHash> capacity;
    for (auto waitIdx : op.setPipe) {
        const auto& target = depOps_[waitIdx].selfPipeCore;
        const PipePairEx pp{source, {target.pipeStart, target.core, target.aivCore}};
        if (!IsIntraCoreEvent(pp) || pp.first == pp.second) {
            continue;
        }
        if (pp.first.core == CoreType::AIV && pp.second.core == CoreType::AIV &&
            pp.first.aivCore != pp.second.aivCore) {
            return IntraEventCapacityFailure{op.idx, waitIdx, pp, 0, 0};
        }
        auto [it, inserted] = capacity.try_emplace(pp, 0, 0);
        auto& [required, available] = it->second;
        if (inserted) {
            if (!consumed) {
                consumed.emplace(ProjectConsumedInstances(op, source));
            }
            available = GetAvailableIntraEventIds(pp, *consumed).size();
        }
        if (++required > available) {
            return IntraEventCapacityFailure{op.idx, waitIdx, pp, required, available};
        }
    }
    return std::nullopt;
}

void PipeSync::LogIntraEventCapacityFailure(const IntraEventCapacityFailure& failure) const
{
    const auto& source = failure.pipes.first;
    const auto& target = failure.pipes.second;
    if (source.core == CoreType::AIV && target.core == CoreType::AIV && source.aivCore != target.aivCore) {
        APASS_LOG_ERROR_F(Elements::Operation, "Unsupported AIV-to-AIV dependency %zu -> %zu, cores %d -> %d.",
                          failure.setIdx, failure.waitIdx, static_cast<int>(source.aivCore),
                          static_cast<int>(target.aivCore));
        return;
    }
    APASS_LOG_ERROR_F(Elements::Operation,
                      "Insufficient intra-core events for op %zu -> %zu: pipe %s -> %s, core %d/%d -> %d/%d, "
                      "required %zu, reusable %zu. Consumption has not returned to the source pipe.",
                      failure.setIdx, failure.waitIdx, GetPipeTypeDict().Find(source.pipe).c_str(),
                      GetPipeTypeDict().Find(target.pipe).c_str(), static_cast<int>(source.core),
                      static_cast<int>(source.aivCore), static_cast<int>(target.core), static_cast<int>(target.aivCore),
                      failure.required, failure.available);
}

Status PipeSync::AllocateIntraEvent(const PipePairEx& pp, int& eventId, size_t setIdx, size_t waitIdx)
{
    auto consumed = eventState_.Known(pp.first);
    const auto available = GetAvailableIntraEventIds(pp, consumed, 1);
    if (available.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "No proven reusable intra-core event for dependency %zu -> %zu.", setIdx,
                          waitIdx);
        return FAILED;
    }
    const int id = available.front();
    if (eventState_.Allocate(GetEventResource(pp, id), pp, setIdx, waitIdx, std::move(consumed)) != SUCCESS) {
        return FAILED;
    }
    auto& queue = GetFreeEventIdQueue(pp);
    queue.erase(std::remove(queue.begin(), queue.end(), id), queue.end());
    eventId = id;
    return SUCCESS;
}

Status PipeSync::RecordEventSet(const PipePairEx& pp, int eventId, size_t setIdx, size_t waitIdx, Operation& syncOp)
{
    if (eventState_.RecordSet(GetEventResource(pp, eventId), pp, setIdx, waitIdx, syncOp, IsIntraCoreEvent(pp)) !=
        SUCCESS) {
        return FAILED;
    }
    if (UseCrossCoreProof()) {
        auto& snapshot = crossCoreSnapshots_[{setIdx, waitIdx}];
        snapshot.consumed = crossCoreConsumed_[pp.first];
        if (pp.first.core != pp.second.core) {
            snapshot.eventId = eventId;
            snapshot.generation = crossCoreGenerations_[eventId];
        }
    }
    return SUCCESS;
}

Status PipeSync::RecordEventWait(size_t setIdx, size_t waitIdx, Operation& syncOp)
{
    if (eventState_.Consume(setIdx, waitIdx, syncOp) != SUCCESS) {
        return FAILED;
    }
    const auto* instance = eventState_.Find(setIdx, waitIdx);
    if (instance->intraCore) {
        // Preserve the original wait-return FIFO policy. Queue membership is
        // only a candidate: allocation still requires consumption at the source.
        GetFreeEventIdQueue(instance->pipes).push_back(instance->id.resource.eventId);
    }
    if (UseCrossCoreProof()) {
        if (!instance->intraCore) {
            const int id = instance->id.resource.eventId;
            crossCoreCompletedSenders_[id] = instance->pipes.first.core;
            crossCoreWaitGenerations_[static_cast<size_t>(instance->pipes.second.core)][id] = crossCoreGenerations_[id];
        }
        auto snapshot = crossCoreSnapshots_.find({setIdx, waitIdx});
        if (snapshot != crossCoreSnapshots_.end()) {
            MergeCrossCoreProof(crossCoreConsumed_[instance->pipes.second], snapshot->second);
            crossCoreSnapshots_.erase(snapshot);
        }
    }
    return SUCCESS;
}

int PipeSync::GetMaxEventId(const PipePairEx& pp)
{
    PipePairEx ppReverse = {pp.second, pp.first};
    auto it1 = doublePipeOp.find(pp);
    auto it2 = doublePipeOp.find(ppReverse);
    if (it1 == doublePipeOp.end() && it2 == doublePipeOp.end()) {
        return EVENT_ID7;
    }
    return EVENT_ID7;
}

std::deque<int>& PipeSync::GetFreeEventIdQueue(const PipePairEx& pp)
{
    auto [it, inserted] = freeEventId_.try_emplace(pp);
    if (inserted) {
        for (int i = INTRA_CORE_START_EVENT_ID; i <= GetMaxEventId(pp); i++) {
            it->second.push_back(i);
        }
    }
    return it->second;
}

} // namespace tile_fwk
} // namespace npu
