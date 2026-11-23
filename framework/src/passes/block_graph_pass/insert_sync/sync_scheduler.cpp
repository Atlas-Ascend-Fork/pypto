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
 * \file sync_scheduler.cpp
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

// Pipe scheduling and dependency updates

std::string PipeSync::DumpLatestPipeDepMap()
{
    std::stringstream ss;
    for (auto pair : latestPipeDep_) {
        ss << "current PipeCore: " << GetPipeTypeDict().Find(pair.first.pipe) << " "
           << GetCoreTypeDict().Find(pair.first.core) << " aivCore: " << static_cast<int>(pair.first.aivCore) << "\n";
        ss << pair.second.DumpPipeDepInfo() << "\n";
    }
    return ss.str();
}

std::string PipeSync::PipeSeqName(PipeSeq seq) const
{
    switch (seq) {
        case PipeSeq::AIC_MTE2:
            return "AIC_MTE2";
        case PipeSeq::AIC_MTE1:
            return "AIC_MTE1";
        case PipeSeq::AIC_M:
            return "AIC_M";
        case PipeSeq::AIC_FIX:
            return "AIC_FIX";
        case PipeSeq::AIV0_MTE2:
            return "AIV0_MTE2";
        case PipeSeq::AIV1_MTE2:
            return "AIV1_MTE2";
        case PipeSeq::AIV0_V:
            return "AIV0_V";
        case PipeSeq::AIV1_V:
            return "AIV1_V";
        case PipeSeq::AIV0_MTE3:
            return "AIV0_MTE3";
        case PipeSeq::AIV1_MTE3:
            return "AIV1_MTE3";
        case PipeSeq::AIC_MTE3:
            return "AIC_MTE3";
        case PipeSeq::AIV0_S:
            return "AIV0_S";
        case PipeSeq::AIV1_S:
            return "AIV1_S";
        case PipeSeq::AIC_S:
            return "AIC_S";
        case PipeSeq::PIPE_END:
            return "PIPE_END";
        default:
            return "ILLEGAL";
    }
}

void PipeSync::InitIssueQueue()
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        issueState_.emplace_back(GetPipeFromSeq(static_cast<PipeSeq>(i)));
    }
}

void PipeSync::EnqueueOp(DepOp& op, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    if (opLogPtr[op.idx]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_NOP || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_HUB ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW_TYPE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_RESHAPE) {
        syncedOpLog.emplace_back(std::make_pair(op.idx * SEQUENCE_IDX, std::ref(*opLogPtr[op.idx])));
        return;
    }
    PipeCoreRealEx opPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    auto& issueQ = issueState_[static_cast<int>(GetPipeSeq(opPipeCoreEx))];
    issueQ.ops.emplace_back(op.idx);
    op.idxInPipe = issueQ.ops.size() - 1;
    // 若op的pipeStart和pipeEnd不同, 进行记录
    if (op.selfPipeCore.pipeStart != op.selfPipeCore.pipeEnd) {
        PipeCoreRealEx opPipeCoreExStart(op.selfPipeCore.pipeStart, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipeCoreRealEx opPipeCoreExEnd(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipePairEx pp{opPipeCoreExStart, opPipeCoreExEnd};
        int opMagic = opLogPtr[op.idx]->GetOpMagic();
        doublePipeOp[pp].emplace_back(opMagic);
    }
    orderedOpList_.emplace(op.idx);
}

Status PipeSync::ProcessSameCoreCase(const PipePairEx& pp, EventIdProcessContext& ctx,
                                     std::optional<ConsumedInstances>& projected)
{
    const auto issued = ctx.issuenum.currIssueNum.try_emplace(pp, 0).first->second;
    // Reserve the complete outgoing group before PopFromQueue commits this op.
    const auto required = std::count_if(ctx.op.setPipe.begin(), ctx.op.setPipe.end(), [&](size_t idx) {
        const auto& target = depOps_[idx].selfPipeCore;
        return target.pipeStart == pp.second.pipe && target.core == pp.second.core &&
               target.aivCore == pp.second.aivCore;
    });
    const auto needed = issued + std::max<size_t>(required, 1);
    const auto* proofs = &eventState_.KnownForAllocationCheck(pp.first);
    // With no same-pipe wait to apply, the exact generation lookup already ignores stale proofs.
    if (!ctx.op.waitPipe.empty() && ctx.op.selfPipeCore.pipeStart == pp.first.pipe) {
        if (!projected) {
            projected.emplace(ProjectConsumedInstances(ctx.op, pp.first));
        }
        proofs = &*projected;
    }
    const auto available = GetAvailableIntraEventIds(pp, *proofs, needed).size();
    if (available < needed) {
        if (!ctx.deadlock) {
            ctx.eventIdOk = false;
            return SUCCESS;
        }
        // A yield can rewrite the waits used to project consumption.
        projected.reset();
        if (AdjustOpDep(ctx.op, ctx.eleIdx, ctx.issueQ, ctx.failedFlag) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessSameCoreCase failed at AdjustOpDep.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessEventIdElement(EventIdProcessContext& ctx, std::optional<ConsumedInstances>& projected)
{
    auto& op = ctx.op;
    auto ele = ctx.eleIdx;

    if (op.selfPipeCore.pipeEnd == depOps_[ele].selfPipeCore.pipeStart) {
        return SUCCESS;
    }

    PipeCoreRealEx currEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    PipeCoreRealEx eleEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                         depOps_[ele].selfPipeCore.aivCore);
    PipePairEx pp{currEx, eleEx};

    // 当前逻辑该函数不需要处理cv同步 (if mix cv core, need handle mix cv core sync)
    if (!IsMixCvCore(currEx.core, eleEx.core) && (currEx.core != eleEx.core)) {
        return SUCCESS;
    }
    return ProcessSameCoreCase(pp, ctx, projected);
}

Status PipeSync::HandleEventID(DepOp& op, IssueQueue& issueQ, IssueNum& issuenum, bool& deadlock, bool& res,
                               std::vector<IndexOp>& syncedOpLog)
{
    bool eventIdOk = true;
    bool failedFlag = false;

    // Yield may rewrite several dependencies. Do not retain a partial rewrite
    // when another destination prevents this operation from being issued.
    const auto originalDeps = deadlock ? SnapshotYieldDependencies(op, issueQ) : std::unordered_map<size_t, DepOp>{};
    std::optional<ConsumedInstances> projected;
    // Only the deadlock path can rewrite the dependencies during iteration.
    const auto originalSetPipe = deadlock ? op.setPipe : std::vector<size_t>{};
    const auto& oriSetPipe = deadlock ? originalSetPipe : op.setPipe;
    for (auto ele : oriSetPipe) {
        EventIdProcessContext ctx{op, ele, issueQ, issuenum, syncedOpLog, deadlock, eventIdOk, failedFlag};
        if (ProcessEventIdElement(ctx, projected) != SUCCESS) {
            RestoreOpDeps(originalDeps);
            APASS_LOG_ERROR_F(Elements::Operation, "HandleEventID failed at ProcessEventIdElement.");
            return FAILED;
        }
        eventIdOk = ctx.eventIdOk;
        failedFlag = ctx.failedFlag;
        if (!eventIdOk) {
            break;
        }
    }

    if (failedFlag) {
        deadlock = true;
        res = false;
    } else {
        deadlock = false;
        res = eventIdOk;
    }
    if (!res) {
        RestoreOpDeps(originalDeps);
    }
    return SUCCESS;
}

bool PipeSync::CheckIssuedOp(const DepOp& op)
{
    // current op will be issued only when all of the waitop are issued
    for (const auto& waitOp : op.waitPipe) {
        if (!depOps_[waitOp].issued) {
            return false;
        }
    }
    return true;
}

Status PipeSync::PopFromQueue(IssueQueue& issueQ, std::vector<size_t>& poped, bool& deadlock,
                              std::vector<IndexOp>& syncedOpLog)
{
    IssueNum issuenum;

    for (uint64_t i = 0; i < MAX_POP; i++) {
        if (issueQ.currOp >= issueQ.ops.size()) {
            break;
        }
        auto& op = depOps_[issueQ.ops[issueQ.currOp]];
        if (!poped.empty() && !op.waitPipe.empty()) {
            break;
        }
        if (op.idx != orderedOpList_.front()) {
            break;
        }
        if (op.issued) {
            APASS_LOG_ERROR_F(Elements::Operation, "Try to issue a op which is already issued, PopFromQueue failed.");
            return FAILED;
        }
        if (!CheckIssuedOp(op)) {
            break;
        }
        bool res = false;
        // cv同步不需要在此处理eventid
        if (HandleEventID(op, issueQ, issuenum, deadlock, res, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PopFromQueue failed at function HandleEventID.");
            return FAILED;
        }
        if (!res) {
            break;
        }
        op.issued = true;
        poped.emplace_back(op.idx);
        orderedOpList_.pop();
        CountIssuedEvents(op, issuenum);
        issueQ.currOp++;
        // Commit waits and their proofs before checking the next operation.
        // A batch preflight cannot see the returning waits in its own batch.
        if (!op.waitPipe.empty()) {
            break;
        }
    }
    return SUCCESS;
}

PipeSync::PipeCoreRealEx PipeSync::NormalizeToAICCoreForCVMixSync(PipeCoreRealEx p) const
{
    if (p.core == CoreType::AIV && p.pipe != PipeType::PIPE_V) {
        p.core = CoreType::AIC;
        p.aivCore = AIVCore::UNSPECIFIED;
    }
    return p;
}

bool PipeSync::IsMixCvCore(CoreType coreOne, CoreType coreTwo)
{
    return Platform::Instance().GetSoc().GetCoreWrap().IsCVMix() &&
           ((coreOne == CoreType::AIV && coreTwo == CoreType::AIC) ||
            (coreOne == CoreType::AIC && coreTwo == CoreType::AIV));
}

Status PipeSync::InjectSync(Function& function, const std::vector<Operation*>& opLogPtr, size_t idx,
                            std::vector<IndexOp>& syncedOpLog)
{
    // check idx range
    if (idx > std::numeric_limits<uint64_t>::max() / SEQUENCE_IDX) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation index is out of range, InjectSync failed.");
        return FAILED;
    }

    // A cross-core drain may retire a completed instance that carries an
    // intra-core proof. Perform it before the final intra-core capacity check.
    if (UseCrossCoreProof() && EnsureCrossCoreCapacity(function, idx, syncedOpLog) != SUCCESS) {
        return FAILED;
    }

    // No waits, sets, or CV force operations may commit when an intra-core
    // destination cannot be allocated for this operation.
    if (auto failure = AnalyzeIntraEventCapacity(depOps_[idx])) {
        LogIntraEventCapacityFailure(*failure);
        return FAILED;
    }

    // insert wait_flag
    if (InjectWaitFlag(function, idx, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InjectSync failed at function InjectWaitFlag.");
        return FAILED;
    }

    // insert current operation
    syncedOpLog.emplace_back(std::make_pair(idx * SEQUENCE_IDX, std::ref(*opLogPtr[idx])));
    depOps_[idx].issued = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert %d %s", opLogPtr[idx]->GetOpMagic(),
                      opLogPtr[idx]->GetOpcodeStr().c_str());

    // The complete output group was preflighted before waits. They have now
    // committed exactly the projected proofs; allocation uses committed state.
    if (InjectSetFlag(function, idx, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InjectSync failed at function InjectSetFlag.");
        return FAILED;
    }

    return SUCCESS;
}

Status PipeSync::ProcessDeadLock(uint64_t& eventIdDeadlockEnterTimes, bool& eventIdDeadlock,
                                 std::vector<IndexOp>& syncedOpLog, bool* mergeSearchExhausted)
{
    eventIdDeadlockEnterTimes++;
    APASS_LOG_DEBUG_F(Elements::Operation, "Event reuse stall: attempt=%lu, emitted=%zu, pending=%zu.",
                      static_cast<unsigned long>(eventIdDeadlockEnterTimes), syncedOpLog.size(), orderedOpList_.size());
    // eventID deadlock
    if (eventIdDeadlockEnterTimes > DEADLOCK_TIME_THRESHOLD) {
        eventIdDeadlock = true;
    }
    if (mergeSearchExhausted == nullptr || !*mergeSearchExhausted) {
        bool merged = false;
        if (RelaxFakeDataDep(syncedOpLog, merged) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessDeadLock failed at function RelaxFakeDataDep.");
            return FAILED;
        }
        if (mergeSearchExhausted != nullptr) {
            *mergeSearchExhausted = !merged;
        }
    }
    if (eventIdDeadlockEnterTimes >= EVENTID_DEADLOCK_ENTER_TIME) {
        if (!orderedOpList_.empty()) {
            // Emit the blocked op and exact directed pool before the terminal error.
            if (auto failure = AnalyzeIntraEventCapacity(depOps_[orderedOpList_.front()])) {
                LogIntraEventCapacityFailure(*failure);
            }
        }
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Unbreakable deadlock: no legal dependency merge or yield position after %lu attempts; "
                          "intra-core IDs with consumed waits still require a return path to their source pipe.",
                          static_cast<unsigned long>(eventIdDeadlockEnterTimes));
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::IssueOpPipeSeq(Function& function, const std::vector<Operation*>& opLogPtr,
                                std::vector<IndexOp>& syncedOpLog, bool& eventIdDeadlock, size_t& issued)
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        std::vector<size_t> issuedOps;
        if (PopFromQueue(issueState_[i], issuedOps, eventIdDeadlock, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function PopFromQueue.");
            return FAILED;
        }
        issued += issuedOps.size();
        for (auto idx : issuedOps) {
            if (InjectSync(function, opLogPtr, idx, syncedOpLog) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function InjectSync.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::IssueOp(Function& function, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += issueState_[i].ops.size();
        APASS_LOG_DEBUG_F(Elements::Operation, "Pipe seq %d: %s %s", i, PipeSeqName(static_cast<PipeSeq>(i)).c_str(),
                          issueState_[i].DumpIssueQueue(opLogPtr).c_str());
    }
    if (IssueSyncOp(function, opLogPtr, syncedOpLog, totalIssued, allIssued) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed with IssueSyncOp.");
        return FAILED;
    }
    if (totalIssued != allIssued) {
        APASS_LOG_ERROR_F(Elements::Operation, "Issue error, IssueOp failed.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "ALL op issued: %zu", totalIssued);
    return SUCCESS;
}

bool PipeSync::FindMaxOverlap(DataDepInfo& depInfo, int& maxOverlapDepIdx)
{
    if (depInfo.opDepList.size() < 2) {
        return false;
    }
    int maxOverlap = -1;
    for (int idx = 0; idx < static_cast<int>(depInfo.opDepList.size() - 1); idx++) {
        const auto [set1, wait1] = depInfo.opDepList[idx];
        const auto [set2, wait2] = depInfo.opDepList[idx + 1];
        // SynDependency delays the first wait to the later producer. Nested or
        // reversed endpoints cannot be merged with that rewrite.
        if (set1 >= set2 || wait1 >= wait2 || wait1 <= set2) {
            continue;
        }
        // max_overlap初始值为阈值，相邻两个依赖之间的重叠超过阈值，可以合并
        // max_overlap用来记录遍历到的最大overlap
        if ((depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first) > maxOverlap) {
            maxOverlapDepIdx = idx;
            maxOverlap = depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first;
        }
    }
    if (maxOverlapDepIdx == -1) { // 8个依赖中，前后依赖都没有重叠，或者重叠全部小于阈值。
        return false;
    }
    return true;
}

Status PipeSync::MergePendingDependencies(int set1, int wait1, int set2, int wait2, int eventId1, int eventId2)
{
    std::unordered_map<size_t, DepOp> originalDeps{{static_cast<size_t>(set1), depOps_[set1]},
                                                   {static_cast<size_t>(wait1), depOps_[wait1]},
                                                   {static_cast<size_t>(set2), depOps_[set2]},
                                                   {static_cast<size_t>(wait2), depOps_[wait2]}};
    // AddOpDep may replace another wait on set2's target pipe. Keep its
    // reverse dependency available even if a later tracker transition fails.
    for (auto waitIdx : depOps_[set2].setPipe) {
        originalDeps.emplace(waitIdx, depOps_[waitIdx]);
    }
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "Event reuse pending merge: (%d,%d), id=%d and (%d,%d), id=%d become (%d,%d).", set1, wait1,
                      eventId1, set2, wait2, eventId2, set2, wait1);
    RemoveOpDep(depOps_[set1], depOps_[wait1]);
    RemoveOpDep(depOps_[set2], depOps_[wait2]);
    if (AddOpDep(depOps_[set2], depOps_[wait1]) != SUCCESS) {
        RestoreOpDeps(originalDeps);
        APASS_LOG_ERROR_F(Elements::Operation, "RelaxFakeDataDep failed at function AddOpDep.");
        return FAILED;
    }
    if (eventState_.MergePending(set1, wait1, set2, wait2) != SUCCESS) {
        RestoreOpDeps(originalDeps);
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::SynDependency(int maxOverlapDepIdx, const DataDepInfo& depInfo, const PipePairEx& pipePairEx,
                               std::vector<IndexOp>& syncedOpLog)
{
    int set1 = depInfo.opDepList[maxOverlapDepIdx].first;
    int wait1 = depInfo.opDepList[maxOverlapDepIdx].second;
    int set2 = depInfo.opDepList[maxOverlapDepIdx + 1].first;
    int wait2 = depInfo.opDepList[maxOverlapDepIdx + 1].second;
    int eventId1 = depInfo.setOpEventIdList[maxOverlapDepIdx];
    int eventId2 = depInfo.setOpEventIdList[maxOverlapDepIdx + 1];
    int syncOpIdx1 = depInfo.setOpIdList[maxOverlapDepIdx];
    const auto* first = eventState_.Find(set1, wait1);
    const auto* second = eventState_.Find(set2, wait2);
    if (ValidatePendingMerge(maxOverlapDepIdx, depInfo, pipePairEx, syncedOpLog, first, second) != SUCCESS) {
        return FAILED;
    }
    const auto& firstInstance = *first;
    const auto& secondInstance = *second;
    if (MergePendingDependencies(set1, wait1, set2, wait2, eventId1, eventId2) != SUCCESS) {
        return FAILED;
    }
    // Rewritten dependencies cannot inherit an old cross-core consumption snapshot.
    crossCoreSnapshots_.erase({set1, wait1});
    crossCoreSnapshots_.erase({set2, wait2});
    GetFreeEventIdQueue(pipePairEx).push_back(eventId1);
    setWaitPairMap_.erase({set1, wait1});
    setWaitPairMap_.erase({set2, wait2});
    setWaitPairMap_[{set2, wait1}] = eventId2;
    setOpMap.erase(firstInstance.setOp);
    setOpMap[secondInstance.setOp] = oriOpList_[wait1];
    DiscardMergedSet(syncedOpLog, syncOpIdx1);
    return SUCCESS;
}

Status PipeSync::GetDepInfo(std::vector<IndexOp>& syncedOpLog, const PipePairEx& pipePairEx, DataDepInfo& depInfo)
{
    depInfo.Reset(pipePairEx);
    // Exhausted slots can already have a wait, without a returning proof.
    // Only pending waits may participate in dependency merging.
    std::unordered_map<Operation*, const EventInstance*> pending;
    // Each directed slot has at most one pending generation. Look it up
    // directly instead of walking the history of retired and cancelled sets.
    for (int id = INTRA_CORE_START_EVENT_ID; id <= GetMaxEventId(pipePairEx); ++id) {
        const auto* instance = eventState_.FindActive(GetEventResource(pipePairEx, id));
        if (instance != nullptr && instance->state == InstanceState::Allocated && instance->setOp != nullptr &&
            !depOps_[instance->waitIdx].issued) {
            pending.emplace(instance->setOp, instance);
        }
    }
    if (!pending.empty()) {
        depInfo.CollectPending(syncedOpLog, pending);
    }
    return SUCCESS;
}

Status PipeSync::RelaxFakeDataDep(std::vector<IndexOp>& syncedOpLog, bool& merged)
{
    const auto dataDepPairEx = BuildMergePipePairs();
    for (const auto& pipePairEx : dataDepPairEx) {
        // 合并CV同步释放cv eventid 当前逻辑不需要 (if mix cv core, need handle mix cv core sync)
        if (!(IsMixCvCore(pipePairEx.first.core, pipePairEx.second.core)) &&
            (pipePairEx.first.core != pipePairEx.second.core)) {
            continue;
        }
        // CanAllocate checks the active generation, so stale proofs cannot match.
        const auto& consumed = eventState_.KnownForAllocationCheck(pipePairEx.first);
        if (!GetAvailableIntraEventIds(pipePairEx, consumed, 1).empty()) {
            continue;
        }
        // 找到free id exhausted的pipe pair
        DataDepInfo depInfo;
        if (GetDepInfo(syncedOpLog, pipePairEx, depInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetDepInfo failed.");
            return FAILED;
        }
        // 找到前一个依赖和后一个依赖间重叠最大的地方
        int maxOverlapDepIdx{-1};
        if (!(FindMaxOverlap(depInfo, maxOverlapDepIdx))) {
            continue;
        }
        // 合并依赖
        if (SynDependency(maxOverlapDepIdx, depInfo, pipePairEx, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SynDependency failed.");
            return FAILED;
        }
        merged = true;
    }

    return SUCCESS;
}

Status PipeSync::GetEventId(const PipePairEx& pp, int& eventId, size_t setIdx, size_t waitIdx,
                            std::vector<IndexOp>& syncedOpLog, Function& function)
{
    if (pp.first.pipe == pp.second.pipe && pp.first.core == pp.second.core) {
        // Pipe Barrier
        eventId = -1;
        return SUCCESS;
    }

    // CV eventid获取
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) && pp.first.core != pp.second.core) {
        return AllocateCrossCoreEvent(pp, eventId, setIdx, syncedOpLog, function);
    }

    return AllocateIntraEvent(pp, eventId, setIdx, waitIdx);
}

void PipeSync::InitCVEventIdQ()
{
    for (const auto& pipecore : cvPipeCoreEx) {
        int base = (pipecore.aivCore == AIVCore::AIV0) ? 0 : CROSS_CORE_EVENT_NUM;
        if (pipecore.core == CoreType::AIC) {
            for (int i = 0; i < CROSS_CORE_EVENT_NUM; i++) {
                crossCoreFreeEventId_[pipecore][0].push_back(i);
            }
            for (int i = CROSS_CORE_EVENT_NUM; i < CROSS_CORE_EVENT_NUM * NUM2; i++) {
                crossCoreFreeEventId_[pipecore][1].push_back(i);
            }
        } else {
            for (int i = base; i < base + CROSS_CORE_EVENT_NUM; i++) {
                crossCoreFreeEventId_[pipecore][0].push_back(i);
            }
        }
    }
}

void PipeSync::PushEventIdIfAbsent(std::deque<int>& queue, int eventId)
{
    if (std::find(queue.begin(), queue.end(), eventId) == queue.end()) {
        queue.push_back(eventId);
    }
}

Status PipeSync::ValidatePendingMerge(int maxOverlapDepIdx, const DataDepInfo& depInfo, const PipePairEx& pipePairEx,
                                      const std::vector<IndexOp>& syncedOpLog, const EventInstance* first,
                                      const EventInstance* second)
{
    int set1 = depInfo.opDepList[maxOverlapDepIdx].first;
    int wait1 = depInfo.opDepList[maxOverlapDepIdx].second;
    int set2 = depInfo.opDepList[maxOverlapDepIdx + 1].first;
    int wait2 = depInfo.opDepList[maxOverlapDepIdx + 1].second;
    int eventId1 = depInfo.setOpEventIdList[maxOverlapDepIdx];
    int eventId2 = depInfo.setOpEventIdList[maxOverlapDepIdx + 1];
    int syncOpIdx1 = depInfo.setOpIdList[maxOverlapDepIdx];
    if (set1 >= set2 || wait1 >= wait2 || set2 >= wait1 || first == nullptr || second == nullptr ||
        depOps_[wait1].issued || depOps_[wait2].issued) {
        APASS_LOG_ERROR_F(Elements::Operation, "Dependency error, RelaxFakeDataDep failed.");
        return FAILED;
    }
    const auto& firstInstance = *first;
    const auto& secondInstance = *second;
    if (firstInstance.state != InstanceState::Allocated || secondInstance.state != InstanceState::Allocated ||
        !firstInstance.intraCore || !secondInstance.intraCore || firstInstance.pipes != pipePairEx ||
        secondInstance.pipes != pipePairEx || firstInstance.id.resource.eventId != eventId1 ||
        secondInstance.id.resource.eventId != eventId2 ||
        firstInstance.setOp != &syncedOpLog[syncOpIdx1].second.get()) {
        return FAILED;
    }
    // Validate the core pair before changing dependency or event-instance state.
    if (!(IsMixCvCore(pipePairEx.first.core, pipePairEx.second.core)) &&
        (pipePairEx.first.core != pipePairEx.second.core)) {
        APASS_LOG_ERROR_F(Elements::Operation, "CoreType is not same, SynDependency failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::ValidateWaitGroup(size_t idx) const
{
    const PipeCore currPipe = depOps_[idx].selfPipeCore;
    for (auto ele : depOps_[idx].waitPipe) {
        const auto& setPipe = depOps_[ele].selfPipeCore;
        if ((setPipe.aivCore == AIVCore::AIV0 && currPipe.aivCore == AIVCore::AIV1) ||
            (setPipe.aivCore == AIVCore::AIV1 && currPipe.aivCore == AIVCore::AIV0)) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Sync between AIV0 and AIV1 happened. wait op idx: %zu, wait op: %s[%d], set op idx: %zu, "
                "set op: %s[%d], setPipe: {pipeStart %s, pipeEnd %s, core %d, aivCore %d}, currPipe: "
                "{pipeStart %s, pipeEnd %s, core %d, aivCore %d}.",
                idx, oriOpList_[idx]->GetOpcodeStr().c_str(), oriOpList_[idx]->GetOpMagic(), ele,
                oriOpList_[ele]->GetOpcodeStr().c_str(), oriOpList_[ele]->GetOpMagic(),
                GetPipeTypeDict().Find(setPipe.pipeStart).c_str(), GetPipeTypeDict().Find(setPipe.pipeEnd).c_str(),
                static_cast<int>(setPipe.core), static_cast<int>(setPipe.aivCore),
                GetPipeTypeDict().Find(currPipe.pipeStart).c_str(), GetPipeTypeDict().Find(currPipe.pipeEnd).c_str(),
                static_cast<int>(currPipe.core), static_cast<int>(currPipe.aivCore));
            return FAILED;
        }
        auto pair = setWaitPairMap_.find({ele, idx});
        if (pair == setWaitPairMap_.end()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Missing set for wait dependency %zu -> %zu.", ele, idx);
            return FAILED;
        }
        if (pair->second >= 0) {
            const auto* event = eventState_.Find(ele, idx);
            if (event == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Missing instance for wait dependency %zu -> %zu, event %d.",
                                  ele, idx, pair->second);
                return FAILED;
            }
            if (event->state != InstanceState::Allocated || event->setOp == nullptr ||
                event->id.resource.eventId != pair->second) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Invalid instance for wait dependency %zu -> %zu: event %d, expected %d, state %d, "
                                  "set present %d.",
                                  ele, idx, event->id.resource.eventId, pair->second, static_cast<int>(event->state),
                                  static_cast<int>(event->setOp != nullptr));
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void PipeSync::NormalizeMergePipePairs(std::vector<PipePairEx>& dataDepPairEx) const
{
    // for cv mix architecture, normalize all non PIPE_V instructions to AIC, to maintain consistency in a singular
    // core since they share a singular eventid
    if (Platform::Instance().GetSoc().GetCoreWrap().IsCVMix()) {
        std::unordered_set<PipePairEx, PipePairExHash> seen;
        std::vector<PipePairEx> normalized;
        for (const auto& pe : dataDepPairEx) {
            PipePairEx npe = {NormalizeToAICCoreForCVMixSync(pe.first), NormalizeToAICCoreForCVMixSync(pe.second)};
            if (seen.insert(npe).second) {
                normalized.emplace_back(npe);
            }
        }
        dataDepPairEx = std::move(normalized);
    }
}

std::vector<PipeSync::PipePairEx> PipeSync::BuildMergePipePairs() const
{
    // 合并阈值。只有前后两个set-wait对之间的重叠（以op数量度量）超过该阈值时，才对这两个set-wait对进行合并。
    std::vector<PipePairEx> dataDepPairEx;
    for (const auto& pipePair : dataDepPair) {
        if (pipePair.first.core == CoreType::AIC) {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::UNSPECIFIED);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::UNSPECIFIED);
            PipePairEx ppEx = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx);
        } else {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV0);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV0);
            PipePairEx ppEx1 = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx1);
            PipeCoreRealEx pp3(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV1);
            PipeCoreRealEx pp4(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV1);
            PipePairEx ppEx2 = {pp3, pp4};
            dataDepPairEx.emplace_back(ppEx2);
        }
    }

    if (Platform::Instance().GetSoc().GetCoreWrap().IsCVMix()) {
        for (const auto& pipePair : mixCVDataDepPair) {
            if (pipePair.first.core == CoreType::AIC && pipePair.second.core == CoreType::AIV) {
                PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::UNSPECIFIED);
                PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV0);
                PipePairEx ppEx = {pp1, pp2};
                dataDepPairEx.emplace_back(ppEx);
            } else if (pipePair.first.core == CoreType::AIV && pipePair.second.core == CoreType::AIC) {
                PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV0);
                PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::UNSPECIFIED);
                PipePairEx ppEx = {pp1, pp2};
                dataDepPairEx.emplace_back(ppEx);
            }
        }
    }

    NormalizeMergePipePairs(dataDepPairEx);
    return dataDepPairEx;
}

std::unordered_map<size_t, PipeSync::DepOp> PipeSync::SnapshotYieldDependencies(const DepOp& op,
                                                                                const IssueQueue& issueQ) const
{
    std::unordered_map<size_t, DepOp> originalDeps;
    {
        originalDeps.emplace(op.idx, op);
        for (auto waitIdx : op.setPipe) {
            originalDeps.emplace(waitIdx, depOps_[waitIdx]);
        }
        if (issueQ.currOp + 1 < issueQ.ops.size()) {
            const auto nextIdx = issueQ.ops[issueQ.currOp + 1];
            originalDeps.emplace(nextIdx, depOps_[nextIdx]);
            // AddOpDep may replace a later wait already attached to nextOp.
            for (auto waitIdx : depOps_[nextIdx].setPipe) {
                originalDeps.emplace(waitIdx, depOps_[waitIdx]);
            }
        }
    }
    return originalDeps;
}

void PipeSync::CountIssuedEvents(const DepOp& op, IssueNum& issuenum) const
{
    for (auto ele : op.setPipe) {
        PipeCoreRealEx currPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipeCoreRealEx elePipeCoreEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                                     depOps_[ele].selfPipeCore.aivCore);
        auto pp = PipePairEx{currPipeCoreEx, elePipeCoreEx};
        issuenum.currIssueNum[pp] = issuenum.currIssueNum[pp] + 1;
    }
}

// Replay scheduling

void PipeSync::CacheReplayOrder(const std::vector<size_t>& order)
{
    if (order.size() != depOps_.size()) {
        replayRank_.clear();
        return;
    }
    replayRank_.resize(order.size());
    replayBoundary_.resize(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        replayRank_[order[i]] = i;
    }
    size_t maxRank = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        maxRank = std::max(maxRank, replayRank_[i]);
        // A cross-core endpoint must separate all lower indices from all higher ones.
        replayBoundary_[i] = maxRank == i && replayRank_[i] == i;
    }
}

bool PipeSync::PreservesReplayOrder(size_t from, size_t to) const
{
    if (replayRank_[from] >= replayRank_[to]) {
        return false;
    }
    const auto& source = depOps_[from].selfPipeCore;
    const auto& target = depOps_[to].selfPipeCore;
    return (replayBoundary_[from] && replayBoundary_[to]) ||
           IsIntraCoreEvent(
               {{source.pipeEnd, source.core, source.aivCore}, {target.pipeStart, target.core, target.aivCore}});
}

bool PipeSync::CanAddReplayDependency(size_t from, size_t to)
{
    if (!replayRank_.empty() && PreservesReplayOrder(from, to)) {
        return true;
    }
    std::vector<size_t> order;
    if (!BuildReplayIssueOrder(depOps_, order, std::make_pair(from, to))) {
        return false;
    }
    CacheReplayOrder(order);
    return true;
}

bool PipeSync::MergeDependenciesForReplay(std::vector<DepOp>& plan, const PipePairEx& pipes,
                                          std::vector<size_t>& order) const
{
    const auto dependencies = CollectReplayDependencies(plan, pipes);
    for (size_t i = 0; i + 1 < dependencies.size(); ++i) {
        const auto [set1, wait1] = dependencies[i];
        const auto [set2, wait2] = dependencies[i + 1];
        // A merge wholly after the blocked producer cannot reduce its event demand.
        if (!orderedOpList_.empty() && set1 > orderedOpList_.front()) {
            continue;
        }
        if (!orderedOpList_.empty() && set1 < orderedOpList_.front()) {
            const auto* instance = eventState_.Find(set1, wait1);
            // Removing an already returned event cannot relieve the blocked pool.
            if (instance == nullptr ||
                (instance->state != InstanceState::Allocated && instance->state != InstanceState::WaitInserted)) {
                continue;
            }
        }
        // The output is sorted by original operation indices, not replay order.
        // An acyclic backward edge would therefore emit wait1 before its set2.
        if (set1 >= set2 || wait1 >= wait2 || set2 >= wait1 ||
            std::find(plan[set2].setPipe.begin(), plan[set2].setPipe.end(), wait1) != plan[set2].setPipe.end()) {
            continue;
        }
        // Check the strengthening edge before removing either original dependency.
        // Source FIFO and destination FIFO then prove coverage of both old edges.
        auto candidate = plan;
        candidate[set2].setPipe.push_back(wait1);
        if (!BuildReplayIssueOrder(candidate, order)) {
            continue;
        }
        auto erase = [](std::vector<size_t>& indices, size_t idx) {
            indices.erase(std::remove(indices.begin(), indices.end(), idx), indices.end());
        };
        erase(candidate[set1].setPipe, wait1);
        erase(candidate[wait1].waitPipe, set1);
        erase(candidate[set2].setPipe, wait2);
        erase(candidate[wait2].waitPipe, set2);
        candidate[wait1].waitPipe.push_back(set2);
        APASS_LOG_DEBUG_F(Elements::Operation, "Event reuse replay merge: (%zu,%zu) and (%zu,%zu) become (%zu,%zu).",
                          set1, wait1, set2, wait2, set2, wait1);
        plan = std::move(candidate);
        return true;
    }
    return false;
}

bool PipeSync::ReturnsEventCapacity(const EventInstance& instance, const IntraEventCapacityFailure& failure,
                                    const ConsumedInstances& consumed) const
{
    if (!instance.intraCore || instance.state != InstanceState::Allocated || instance.setOp == nullptr ||
        instance.pipes.second != failure.pipes.first || instance.setIdx >= failure.setIdx ||
        instance.waitIdx <= failure.setIdx) {
        return false;
    }
    for (const auto& proof : instance.setSnapshot) {
        const auto& resource = eventState_.ProofResource(proof);
        if (resource == GetEventResource(failure.pipes, resource.eventId) && eventState_.HasActive(resource) &&
            eventState_.CanAllocate(resource, instance.setSnapshot) && !eventState_.CanAllocate(resource, consumed)) {
            return true;
        }
    }
    return false;
}

bool PipeSync::AdvanceReturningWaitInPlace(std::vector<DepOp>& plan, const IntraEventCapacityFailure& failure)
{
    const auto blocked = failure.setIdx;
    const auto& target = depOps_[blocked].selfPipeCore;
    if (orderedOpList_.empty() || orderedOpList_.front() != blocked || depOps_[blocked].issued ||
        target.pipeStart != target.pipeEnd ||
        PipeCoreRealEx(target.pipeStart, target.core, target.aivCore) != failure.pipes.first) {
        return false;
    }
    const auto consumed = ProjectConsumedInstances(depOps_[blocked], failure.pipes.first);
    for (const auto& instance : eventState_.Instances()) {
        if (!ReturnsEventCapacity(instance, failure, consumed)) {
            continue;
        }
        const auto set = instance.setIdx;
        const auto wait = instance.waitIdx;
        // Restrict the fast path to one logical core and one destination FIFO.
        // Lite dependencies across logical core types retain replay recovery.
        if (!depOps_[set].issued || depOps_[wait].issued || depOps_[wait].selfPipeCore != target ||
            instance.pipes.first.core != instance.pipes.second.core ||
            instance.pipes.first.aivCore != instance.pipes.second.aivCore) {
            continue;
        }
        if (!CanRetargetPendingWait(instance, blocked, plan)) {
            continue;
        }
        auto moveEdge = [set, wait, blocked](std::vector<DepOp>& deps) {
            auto& targets = deps[set].setPipe;
            *std::find(targets.begin(), targets.end(), wait) = blocked;
            auto& sources = deps[wait].waitPipe;
            sources.erase(std::remove(sources.begin(), sources.end(), set), sources.end());
            deps[blocked].waitPipe.push_back(set);
        };
        auto candidate = plan;
        moveEdge(candidate);
        std::vector<size_t> order;
        if (!BuildReplayIssueOrder(candidate, order) || !eventState_.RetargetPendingWait(set, wait, blocked)) {
            continue;
        }
        // The set and its frozen consumption proof remain unchanged. Only its
        // unissued wait moves earlier; destination FIFO still covers the old wait.
        CommitPendingWaitMove(set, wait, instance.setOp, blocked, plan, candidate);
        return true;
    }
    return false;
}

bool PipeSync::AdvanceReturningWaitForReplay(std::vector<DepOp>& plan, const IntraEventCapacityFailure& failure,
                                             std::vector<size_t>& order) const
{
    const auto blocked = failure.setIdx;
    const auto& source = failure.pipes.first;
    if (plan[blocked].selfPipeCore.pipeStart != source.pipe) {
        return false;
    }
    const auto consumed = ProjectConsumedInstances(depOps_[blocked], source);
    for (const auto& instance : eventState_.Instances()) {
        if (!ReturnsEventCapacity(instance, failure, consumed)) {
            continue;
        }
        const auto set = instance.setIdx;
        const auto wait = instance.waitIdx;
        // Pending merges may differ from the saved replay plan. Only move an
        // existing edge, preserving its old wait through destination FIFO.
        auto candidate = plan;
        auto& targets = candidate[set].setPipe;
        auto old = std::find(targets.begin(), targets.end(), wait);
        if (old == targets.end() || std::find(targets.begin(), targets.end(), blocked) != targets.end()) {
            continue;
        }
        *old = blocked;
        auto& sources = candidate[wait].waitPipe;
        sources.erase(std::remove(sources.begin(), sources.end(), set), sources.end());
        candidate[blocked].waitPipe.push_back(set);
        if (!BuildReplayIssueOrder(candidate, order)) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Event reuse return wait: dependency (%zu,%zu) moves to (%zu,%zu).", set,
                          wait, set, blocked);
        plan = std::move(candidate);
        return true;
    }
    return false;
}

bool PipeSync::AddReturningDependencyForReplay(std::vector<DepOp>& plan, const IntraEventCapacityFailure& failure,
                                               std::vector<size_t>& order) const
{
    const auto& pipes = failure.pipes;
    const auto blocked = failure.setIdx;
    if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) ||
        Platform::Instance().GetSoc().GetCoreWrap().IsCVMix() ||
        plan[blocked].selfPipeCore.pipeStart != pipes.first.pipe) {
        return false;
    }
    const auto consumers = CollectReturningConsumers(failure);
    for (auto consumer : consumers) {
        const auto& core = plan[consumer].selfPipeCore;
        if (core.pipeStart != pipes.second.pipe || core.pipeEnd != pipes.second.pipe ||
            core.core != pipes.second.core || core.aivCore != pipes.second.aivCore ||
            std::find(plan[consumer].setPipe.begin(), plan[consumer].setPipe.end(), blocked) !=
                plan[consumer].setPipe.end()) {
            continue;
        }
        auto candidate = plan;
        if (!RewriteReturningDependency(candidate, consumer, failure)) {
            continue;
        }
        if (!BuildReplayIssueOrder(candidate, order)) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Event reuse directed return: dependency (%zu,%zu).", consumer, blocked);
        plan = std::move(candidate);
        return true;
    }
    return false;
}

void PipeSync::ResetForReplay(const std::vector<DepOp>& plan, const std::vector<IssueQueue>& queues,
                              const std::vector<size_t>& order)
{
    depOps_ = plan;
    CacheReplayOrder(order);
    issueState_ = queues;
    std::unordered_set<size_t> queued;
    for (const auto& queue : queues) {
        queued.insert(queue.ops.begin(), queue.ops.end());
    }
    orderedOpList_ = {};
    for (auto idx : order) {
        if (queued.count(idx) != 0) {
            orderedOpList_.push(idx);
        }
    }
    eventState_ = EventInstanceTracker{};
    reclaimBarriers_.clear();
    freeEventId_.clear();
    // Replay rebuilds the entire emitted stream; no cross-core proof from the old stream survives.
    crossCoreFreeEventId_.clear();
    crossCoreEventSenders_.fill(std::nullopt);
    crossCoreCompletedSenders_.fill(std::nullopt);
    crossCoreWaitGenerations_ = {};
    crossCoreGenerations_.fill(0);
    crossCoreConsumed_.clear();
    crossCoreSnapshots_.clear();
    NoWaitCVPairs_.clear();
    setWaitPairMap_.clear();
    setOpMap.clear();
    waitOpMap.clear();
    forceSyncOps_.clear();
}

Status PipeSync::IssueSyncOp(Function& function, const std::vector<Operation*>& opLogPtr,
                             std::vector<IndexOp>& syncedOpLog, size_t& totalIssued, size_t& allIssued)
{
    bool eventIdDeadlock = false;
    bool mergeSearchExhausted = false;
    uint64_t eventIdDeadlockEnterTimes = 0;
    // Keep the pre-emission plan: pending-only recovery may have already edited
    // the working dependencies when a consumed pool eventually exhausts.
    ReplayContext replay{depOps_, issueState_, syncedOpLog, {opLogPtr.begin(), opLogPtr.end()}};
    InitCVEventIdQ();
    while (totalIssued < allIssued) {
        size_t issued = 0;
        if (IssueOpPipeSeq(function, opLogPtr, syncedOpLog, eventIdDeadlock, issued) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function IssueOpPipeSeq.");
            return FAILED;
        }
        totalIssued += issued;
        // eventIdDeadlockEnterTimes eventIdDeadlock syncedOpLog
        if (issued == 0) {
            if (eventIdDeadlockEnterTimes + 1 >= EVENTID_DEADLOCK_ENTER_TIME &&
                RecoverIssueStall(function, replay, syncedOpLog, totalIssued)) {
                eventIdDeadlock = false;
                mergeSearchExhausted = false;
                eventIdDeadlockEnterTimes = 0;
                continue;
            }
            // With no committed issue or recovery, yielding has rolled back its edits.
            if (ProcessDeadLock(eventIdDeadlockEnterTimes, eventIdDeadlock, syncedOpLog, &mergeSearchExhausted) !=
                SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function ProcessDeadLock.");
                return FAILED;
            }
            continue;
        }
        eventIdDeadlock = false;
        mergeSearchExhausted = false;
        eventIdDeadlockEnterTimes = 0;
    }
    return SUCCESS;
}

std::vector<size_t> PipeSync::CollectReturningConsumers(const IntraEventCapacityFailure& failure) const
{
    const auto& pipes = failure.pipes;
    const auto blocked = failure.setIdx;
    const auto consumed = ProjectConsumedInstances(depOps_[blocked], pipes.first);
    std::vector<size_t> consumers;
    for (const auto& instance : eventState_.Instances()) {
        if (instance.intraCore && instance.pipes == pipes && instance.state == InstanceState::WaitInserted &&
            instance.waitIdx < blocked && eventState_.HasActive(instance.id.resource) &&
            !eventState_.CanAllocate(instance.id.resource, consumed)) {
            consumers.push_back(instance.waitIdx);
        }
    }
    std::sort(consumers.begin(), consumers.end(), std::greater<size_t>());
    return consumers;
}

bool PipeSync::RecoverIssueStall(Function& function, ReplayContext& replay, std::vector<IndexOp>& syncedOpLog,
                                 size_t& totalIssued)
{
    if (!orderedOpList_.empty()) {
        const auto failure = AnalyzeIntraEventCapacity(depOps_[orderedOpList_.front()]);
        std::vector<size_t> replayOrder;
        if (failure && AdvanceReturningWaitInPlace(replay.plan, *failure)) {
            return true;
        }
        if (failure && (AdvanceReturningWaitForReplay(replay.plan, *failure, replayOrder) ||
                        MergeDependenciesForReplay(replay.plan, failure->pipes, replayOrder) ||
                        AddReturningDependencyForReplay(replay.plan, *failure, replayOrder))) {
            ResetLeafForReplay(function, replay, syncedOpLog, replayOrder);
            totalIssued = 0;
            InitCVEventIdQ();
            APASS_LOG_INFO_F(Elements::Operation,
                             "Replaying sync generation after safe intra-core dependency recovery.");
            return true;
        }
        // Drain only after pending merges, yielding and replay recovery have failed.
        if (failure &&
            ReclaimIntraCoreEvents(function, orderedOpList_.front(), failure->pipes, syncedOpLog) == SUCCESS) {
            return true;
        }
    }
    return false;
}

bool PipeSync::CanRetargetPendingWait(const EventInstance& instance, size_t blocked,
                                      const std::vector<DepOp>& plan) const
{
    const auto set = instance.setIdx;
    const auto wait = instance.waitIdx;
    const auto mapping = setWaitPairMap_.find({set, wait});
    const auto setOp = setOpMap.find(instance.setOp);
    auto hasEdge = [set, wait, blocked](const std::vector<DepOp>& deps) {
        const auto& targets = deps[set].setPipe;
        const auto& sources = deps[wait].waitPipe;
        return std::find(targets.begin(), targets.end(), wait) != targets.end() &&
               std::find(targets.begin(), targets.end(), blocked) == targets.end() &&
               std::find(sources.begin(), sources.end(), set) != sources.end() &&
               std::find(deps[blocked].waitPipe.begin(), deps[blocked].waitPipe.end(), set) ==
                   deps[blocked].waitPipe.end();
    };
    if (!hasEdge(depOps_) || !hasEdge(plan) || mapping == setWaitPairMap_.end() ||
        mapping->second != instance.id.resource.eventId || setWaitPairMap_.count({set, blocked}) != 0 ||
        crossCoreSnapshots_.count({set, blocked}) != 0 || setOp == setOpMap.end() ||
        setOp->second != oriOpList_[wait]) {
        return false;
    }
    return true;
}

void PipeSync::CommitPendingWaitMove(size_t set, size_t wait, Operation* syncSet, size_t blocked,
                                     std::vector<DepOp>& plan, std::vector<DepOp>& candidate)
{
    const auto mapping = setWaitPairMap_.find({set, wait});
    const auto setOp = setOpMap.find(syncSet);
    const auto eventId = mapping->second;
    setWaitPairMap_.erase(mapping);
    setWaitPairMap_[{set, blocked}] = eventId;
    setOp->second = oriOpList_[blocked];
    auto snapshot = crossCoreSnapshots_.extract({set, wait});
    if (!snapshot.empty()) {
        snapshot.key() = {set, blocked};
        crossCoreSnapshots_.insert(std::move(snapshot));
    }
    auto& targets = depOps_[set].setPipe;
    *std::find(targets.begin(), targets.end(), wait) = blocked;
    auto& sources = depOps_[wait].waitPipe;
    sources.erase(std::remove(sources.begin(), sources.end(), set), sources.end());
    depOps_[blocked].waitPipe.push_back(set);
    plan = std::move(candidate);
    replayRank_.clear();
    APASS_LOG_DEBUG_F(Elements::Operation, "Advanced pending event wait from %zu to %zu for set %zu.", wait, blocked,
                      set);
}

} // namespace tile_fwk
} // namespace npu
