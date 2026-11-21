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
 * \file ooo_scheduler_spill.cpp
 * \brief OoOScheduler spill 决策、选组与应用。
 */

#include "ooo_scheduler.h"

#include "passes/pass_log/pass_log.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOSchedule"

namespace npu::tile_fwk {

Status OoOScheduler::SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> orderFirstPair)
{
    APASS_LOG_INFO_F(Elements::Operation, "Start to spillOnBlock at %s",
                     coreTypeToString(orderFirstPair.first).c_str());
    SpillContext ctx;
    if (GenBufferSpill(state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front(), ctx, true) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed at GenBufferSpill.");
        return FAILED;
    }
    if (ApplySpillContext(ctx, state_.allocIssueQueue[orderFirstPair.first][orderFirstPair.second].Front()) !=
        SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ApplySpillContext failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType& spillMemType)
{
    bool anyNotEmpty = false;
    for (auto& kv : state_.allocIssueQueue[coreLocation]) {
        if (!kv.second.Empty()) {
            anyNotEmpty = true;
            break;
        }
    }
    if (!anyNotEmpty) {
        return FAILED;
    }
    if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_UB].Empty()) {
        spillMemType = MemoryType::MEM_UB;
    } else if (!state_.allocIssueQueue[coreLocation][MemoryType::MEM_L1].Empty()) {
        spillMemType = MemoryType::MEM_L1;
    } else if (coreLocation == CoreLocationType::AIC &&
               !state_.allocIssueQueue[coreLocation][MemoryType::MEM_L0C].Empty()) {
        spillMemType = MemoryType::MEM_L0C;
    } else {
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::FindFirstOrder(std::pair<CoreLocationType, MemoryType>& orderFirstPair)
{
    std::unordered_map<CoreLocationType, MemoryType> spillMemTypeMap;
    std::vector<CoreLocationType> coreVec;
    for (auto& coreLocation : state_.coreInitConfigs) {
        MemoryType spillMemType;
        if (FindCoreLocationMemoryType(coreLocation, spillMemType) != SUCCESS) {
            APASS_LOG_INFO_F(Elements::Operation, "FindCoreLocationMemoryType %s failed.",
                             coreTypeToString(coreLocation).c_str());
            continue;
        }
        spillMemTypeMap[coreLocation] = spillMemType;
        coreVec.push_back(coreLocation);
    }
    if (spillMemTypeMap.empty() || coreVec.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "All coreLocation have no spillMemType.");
        return FAILED;
    }
    std::sort(coreVec.begin(), coreVec.end(),
              [&spillMemTypeMap, this](const CoreLocationType& a, const CoreLocationType& b) {
                  return state_.schedInfoMap[state_.allocIssueQueue[a][spillMemTypeMap[a]].Front()].execOrder <
                         state_.schedInfoMap[state_.allocIssueQueue[b][spillMemTypeMap[b]].Front()].execOrder;
              });
    orderFirstPair = std::make_pair(coreVec[0], spillMemTypeMap[coreVec[0]]);
    return SUCCESS;
}

Status OoOScheduler::SpillOnBlock()
{
    std::pair<CoreLocationType, MemoryType> orderFirstPair;
    if (FindFirstOrder(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "FindFirstOrder failed.");
        return FAILED;
    }

    if (DetectSpillDeadLoop() != SUCCESS)
        return FAILED;

    bool isAivUb = orderFirstPair.second == MemoryType::MEM_UB &&
                   (orderFirstPair.first == CoreLocationType::AIV0 || orderFirstPair.first == CoreLocationType::AIV1);
    if (state_.enableDualDst && isAivUb)
        return SpillDualDstAivUbBlock();

    if (SpillOnCoreBlock(orderFirstPair) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SpillOnCoreBlock failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::DetectSpillDeadLoop()
{
    int curPreSpillTotalQSize = 0;
    for (auto& corePair : state_.allocIssueQueue) {
        for (auto& memPair : corePair.second) {
            curPreSpillTotalQSize += static_cast<int>(memPair.second.Size());
        }
    }
    if (state_.lastPreSpillTotalQSize >= 0) {
        if (curPreSpillTotalQSize >= state_.lastPreSpillTotalQSize) {
            state_.spillNoProgressCnt++;
            if (state_.spillNoProgressCnt >= MAX_SPILL_NO_PROGRESS_ROUNDS) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Spill dead-loop detected: %d consecutive spills without total queue size reduction "
                                  "(current=%d, last=%d).",
                                  state_.spillNoProgressCnt, curPreSpillTotalQSize, state_.lastPreSpillTotalQSize);
                return FAILED;
            }
        } else {
            state_.spillNoProgressCnt = 0;
        }
    }
    state_.lastPreSpillTotalQSize = curPreSpillTotalQSize;
    return SUCCESS;
}

Status OoOScheduler::SpillDualDstAivUbBlock()
{
    auto& queue0 = state_.allocIssueQueue[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& queue1 = state_.allocIssueQueue[CoreLocationType::AIV1][MemoryType::MEM_UB];
    if (queue0.Empty() || queue1.Empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB spill queues are not synchronized.");
        return FAILED;
    }
    Operation* front0 = queue0.Front();
    Operation* front1 = queue1.Front();
    bool isDual0 = state_.IsDualDstAlloc(front0);
    bool isDual1 = state_.IsDualDstAlloc(front1);
    if (isDual0 != isDual1 || (isDual0 && (state_.schedInfoMap[front0].pairedDualDstAlloc != front1 ||
                                           state_.schedInfoMap[front1].pairedDualDstAlloc != front0))) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV UB spill queue fronts are inconsistent.");
        return FAILED;
    }
    if (SpillOnCoreBlock({CoreLocationType::AIV0, MemoryType::MEM_UB}) != SUCCESS ||
        SpillOnCoreBlock({CoreLocationType::AIV1, MemoryType::MEM_UB}) != SUCCESS) {
        return FAILED;
    }
    if (CheckAivUbPoolSlicesEqual() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Tensor, "AIV0/AIV1 UB pool slices diverged after spill.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::AllocateAfterSpill(Operation* op, LocalBufferPtr allocBuffer, CoreLocationType coreLocation,
                                        bool isDualDst)
{
    if (!isDualDst) {
        if (state_.bufferManagerMap[coreLocation][allocBuffer->memType].Allocate(allocBuffer) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Allocate tensor[%d] failed.", allocBuffer->id);
            return FAILED;
        }
        return SUCCESS;
    }

    bool allocated = false;
    if (dualDstEngine_.AllocateDualDstAtCurrent(op, allocated) != SUCCESS)
        return FAILED;
    if (!allocated) {
        APASS_LOG_ERROR_F(Elements::Operation, "DualDst alloc still infeasible after spill. %s",
                          GetFormatBacktrace(*op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

bool OoOScheduler::HasEnoughBuffer(Operation* allocOp, MemoryType memType)
{
    return !state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].IsFull(
        state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]], false);
}

Status OoOScheduler::RearrangeBuffer(Operation* allocOp, MemoryType memType)
{
    std::vector<int> memIds = state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType]
                                  .GetAddrSortedBufs();
    for (auto memId : memIds) {
        auto op = spillEngine_.GetSpillOp(memId);
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] latest write op.", memId);
            return FAILED;
        }
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            return SUCCESS;
        }
    }
    std::vector<BufferAddrChange> changes;
    auto status = state_.bufferManagerMap[state_.schedInfoMap[allocOp].coreLocation][memType].CompactBufferSlices(
        state_.localBufferMap, changes);
    if (status == SUCCESS) {
        NotifyBufferRearrange(allocOp, memType, std::move(changes));
    }
    return status;
}

std::vector<std::vector<int>> OoOScheduler::GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill,
                                                          const std::vector<std::pair<uint64_t, uint64_t>>& avoidRanges)
{
    return pool.GetSpillGroup(sizeNeedSpill, avoidRanges);
}

std::vector<OoOScheduler::DualSpillGroup> OoOScheduler::GetDualSpillGroup(BufferPool& poolA, BufferPool& poolB,
                                                                          size_t sizeNeedSpill)
{
    std::vector<DualSpillGroup> result;
    auto bufsA = poolA.GetSortedAllocatedBufs();
    auto bufsB = poolB.GetSortedAllocatedBufs();

    size_t iA = 0;
    while (iA < bufsA.size()) {
        size_t startAddrA = poolA.ObtainStartAddr(iA, bufsA);
        if ((poolA.GetMemSize() - startAddrA) < sizeNeedSpill) {
            break;
        }
        size_t jA = poolA.UpdateIdx(iA, sizeNeedSpill, startAddrA, bufsA);
        size_t iB = 0;
        while (iB < bufsB.size()) {
            size_t startAddrB = poolB.ObtainStartAddr(iB, bufsB);
            if (startAddrB != startAddrA) {
                if (startAddrB < startAddrA) {
                    iB++;
                    continue;
                }
                break;
            }
            if ((poolB.GetMemSize() - startAddrB) < sizeNeedSpill) {
                break;
            }
            size_t jB = poolB.UpdateIdx(iB, sizeNeedSpill, startAddrB, bufsB);
            if (jA < iA || jB < iB || (jA == iA && jB == iB)) {
                iB++;
                continue;
            }
            DualSpillGroup group;
            group.aiv0MemIds.reserve(jA - iA);
            group.aiv1MemIds.reserve(jB - iB);
            for (size_t k = iA; k < jA; k++) {
                group.aiv0MemIds.push_back(std::get<0>(bufsA[k]));
            }
            for (size_t k = iB; k < jB; k++) {
                group.aiv1MemIds.push_back(std::get<0>(bufsB[k]));
            }
            result.push_back(std::move(group));
            iB++;
        }
        iA++;
    }
    return result;
}

Status OoOScheduler::GetGroupNextUseTime(std::vector<int> group, Operation* allocOp, std::vector<int>& groupNextUseTime,
                                         std::unordered_map<int, size_t>& nextUseTimeCache)
{
    size_t minNextUseTime = INT_MAX;
    for (auto& memId : group) {
        Operation* spillOp = spillEngine_.GetSpillOp(memId);
        if (spillOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find spill Tensor[%d] occupy op.", memId);
            return FAILED;
        }
        if (spillEngine_.IsBelongSpillBlackList(memId, allocOp)) {
            groupNextUseTime.push_back(-1);
            return SUCCESS;
        }
        if (nextUseTimeCache.find(memId) != nextUseTimeCache.end()) {
            minNextUseTime = std::min(minNextUseTime, nextUseTimeCache[memId]);
        } else {
            int nextUseTime = spillEngine_.GetBufNextUseTime(memId);
            if (nextUseTime == -1) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Cannot find Tensor[%d] next used time.", memId);
                return FAILED;
            }
            nextUseTimeCache[memId] = static_cast<size_t>(nextUseTime);
            minNextUseTime = std::min(minNextUseTime, static_cast<size_t>(nextUseTime));
        }
    }
    groupNextUseTime.push_back(minNextUseTime);
    return SUCCESS;
}

std::vector<int> OoOScheduler::SelectSpillBuffers(Operation* allocOp)
{
    LocalBufferPtr allocBuffer = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]];
    auto coreType = state_.schedInfoMap[allocOp].coreLocation;
    auto& pool = state_.bufferManagerMap[coreType][allocBuffer->memType];
    // vf 受限分配的 avoidRanges: SeqSchedule 阶段登记表尚未登记, 天然为空。
    std::vector<std::pair<uint64_t, uint64_t>> avoidRanges;
    if (allocOp->GetAtomicScopeId() >= VF_CLUSTER_ID_START) {
        auto it = scopeTensorRanges_.find(allocOp->GetAtomicScopeId());
        if (it != scopeTensorRanges_.end()) {
            avoidRanges = it->second;
        }
    }
    std::vector<std::vector<int>> canSpillGroups = GetSpillGroup(pool, allocBuffer->size, avoidRanges);
    if (canSpillGroups.empty()) {
        return pool.GetAddrSortedBufs();
    }
    std::unordered_map<int, size_t> nextUseTimeCache;
    std::vector<int> groupNextUseTime;
    for (auto& group : canSpillGroups) {
        if (GetGroupNextUseTime(group, allocOp, groupNextUseTime, nextUseTimeCache) != SUCCESS) {
            APASS_LOG_WARN_F(Elements::Operation, "Get group next use time failed, begin spill all tensor.");
            return pool.GetAddrSortedBufs();
        }
    }
    size_t groupSel = std::max_element(groupNextUseTime.begin(), groupNextUseTime.end()) - groupNextUseTime.begin();
    if (groupNextUseTime[groupSel] == -1) {
        APASS_LOG_WARN_F(Elements::Tensor, "Cannot find tensor to spill, begin spill all tensor.");
        return pool.GetAddrSortedBufs();
    }
    return canSpillGroups[groupSel];
}

Status OoOScheduler::GenBufferSpill(Operation* allocOp, SpillContext& ctx, bool isMainLoop)
{
    auto reqMemType = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->memType;
    auto reqSize = state_.localBufferMap[state_.opReqMemIdsMap[allocOp][0]]->size;
    std::vector<int> spillGroup;
    SpillContext peerCtx;
    Operation* peerAlloc = nullptr;
    DualSpillGroup dualGroup;
    bool useDualGroup = !isMainLoop && state_.IsDualDstAlloc(allocOp);
    if (useDualGroup) {
        peerAlloc = state_.schedInfoMap[allocOp].pairedDualDstAlloc;
        if (SelectDualDstSpillTargets(allocOp, peerAlloc, dualGroup) != SUCCESS)
            return FAILED;
    } else {
        spillGroup = SelectSpillBuffers(allocOp);
    }
    if ((!useDualGroup && spillGroup.empty()) ||
        (useDualGroup && dualGroup.aiv0MemIds.empty() && dualGroup.aiv1MemIds.empty())) {
        APASS_LOG_ERROR_F(Elements::Operation, "Select buffer to spill failed.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    // DualDst 需要共同 offset，不使用单侧 compact 和容量检查。
    if (useDualGroup) {
        if (ApplyDualSpill(allocOp, peerAlloc, dualGroup, ctx, peerCtx, reqMemType, reqSize) != SUCCESS)
            return FAILED;
    } else {
        if (ApplySingleSpill(allocOp, spillGroup, ctx, reqMemType, reqSize) != SUCCESS)
            return FAILED;
        if (FinalizeSingleSideSpill(allocOp, reqMemType, reqSize) != SUCCESS)
            return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::SelectDualDstSpillTargets(Operation* allocOp, Operation* peerAlloc, DualSpillGroup& dualGroup)
{
    DualDstAllocCtx allocCtx;
    if (peerAlloc == nullptr || dualDstEngine_.ResolveDualDstAllocCtx(allocOp, allocCtx) != SUCCESS)
        return FAILED;
    auto& pool0 = state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& pool1 = state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    auto candidates = GetDualSpillGroup(pool0, pool1, allocCtx.bufA->size);
    if (candidates.empty()) {
        dualGroup.aiv0MemIds = pool0.GetAddrSortedBufs();
        dualGroup.aiv1MemIds = pool1.GetAddrSortedBufs();
    } else {
        long long bestScore = -1;
        for (auto& candidate : candidates) {
            std::vector<int> score0;
            std::vector<int> score1;
            std::unordered_map<int, size_t> cache0;
            std::unordered_map<int, size_t> cache1;
            bool allocOnAiv0 = state_.schedInfoMap[allocOp].coreLocation == CoreLocationType::AIV0;
            Operation* alloc0 = allocOnAiv0 ? allocOp : peerAlloc;
            Operation* alloc1 = allocOnAiv0 ? peerAlloc : allocOp;
            if (GetGroupNextUseTime(candidate.aiv0MemIds, alloc0, score0, cache0) != SUCCESS ||
                GetGroupNextUseTime(candidate.aiv1MemIds, alloc1, score1, cache1) != SUCCESS || score0.empty() ||
                score1.empty()) {
                continue;
            }
            long long score = std::min<long long>(score0.back(), score1.back());
            if (score > bestScore) {
                bestScore = score;
                dualGroup = candidate;
            }
        }
        if (bestScore < 0) {
            dualGroup.aiv0MemIds = pool0.GetAddrSortedBufs();
            dualGroup.aiv1MemIds = pool1.GetAddrSortedBufs();
        }
    }
    return SUCCESS;
}

Status OoOScheduler::ApplyDualSpill(Operation* allocOp, Operation* peerAlloc, const DualSpillGroup& dualGroup,
                                    SpillContext& ctx, SpillContext& peerCtx, MemoryType reqMemType, uint64_t reqSize)
{
    auto spillOneSide = [this](const std::vector<int>& memIds, Operation* trigger, SpillContext& spillCtx) {
        spillCtx.spillMemIds = memIds;
        for (int memId : memIds) {
            if (spillEngine_.SpillBuffer(memId, trigger, spillCtx) != SUCCESS)
                return FAILED;
        }
        return SUCCESS;
    };
    Operation* alloc0 = state_.schedInfoMap[allocOp].coreLocation == CoreLocationType::AIV0 ? allocOp : peerAlloc;
    Operation* alloc1 = alloc0 == allocOp ? peerAlloc : allocOp;
    SpillContext& ctx0 = alloc0 == allocOp ? ctx : peerCtx;
    SpillContext& ctx1 = alloc1 == allocOp ? ctx : peerCtx;
    if (spillOneSide(dualGroup.aiv0MemIds, alloc0, ctx0) != SUCCESS ||
        spillOneSide(dualGroup.aiv1MemIds, alloc1, ctx1) != SUCCESS) {
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    return SUCCESS;
}

Status OoOScheduler::ApplySingleSpill(Operation* allocOp, const std::vector<int>& spillGroup, SpillContext& ctx,
                                      MemoryType reqMemType, uint64_t reqSize)
{
    ctx.spillMemIds = spillGroup;
    for (auto& memId : spillGroup) {
        if (spillEngine_.SpillBuffer(memId, allocOp, ctx) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill tensor[%d] for %s failed!", memId,
                              state_.GetOpInfo(allocOp).c_str());
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOScheduler::FinalizeSingleSideSpill(Operation* allocOp, MemoryType reqMemType, uint64_t reqSize)
{
    if (RearrangeBuffer(allocOp, reqMemType) != SUCCESS) {
        APASS_LOG_WARN_F(Elements::Operation, "RearrangeBuffer failed for %s.", GetFormatBacktrace(*allocOp).c_str());
    }
    if (!HasEnoughBuffer(allocOp, reqMemType)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Spill all buffer failed! %s", GetFormatBacktrace(*allocOp).c_str());
        if (PrintSpillFailedInfo(allocOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PrintSpillFailedInfo failed!");
            NotifyAllocFail(allocOp, reqMemType, reqSize);
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Possible causes: incorrect memory reuse, memory fragmentation, or spill not supported for "
            "L0C_COPY_TO_L1."
            "Please check tile shape and OOO spill failed info. Consider avoiding cube-aligned matrix sizes.");
        NotifyAllocFail(allocOp, reqMemType, reqSize);
        return FAILED;
    }
    return SUCCESS;
}

// Inserts the copyout's skip chain (reshape/view/viewType producers, if not already
// present) starting at copyoutPos, in order. Returns the position right after the last
// inserted skip op -- where the copyout itself must go so it lands after its producers.
size_t OoOScheduler::InsertSkipOpsBeforeCopyout(Operation* copyoutOp, size_t copyoutPos)
{
    auto& skipOps = state_.schedInfoMap[copyoutOp].skipOps;
    size_t insertPos = copyoutPos;
    for (auto* skipOp : skipOps) {
        auto existing = std::find(state_.newOperations.begin(), state_.newOperations.end(), skipOp);
        if (existing == state_.newOperations.end()) {
            state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), skipOp);
            insertPos++;
        }
    }
    return insertPos;
}

Status OoOScheduler::InsertSpillCopyoutOp(Operation* copyoutOp)
{
    int spillMemid = copyoutOp->GetInputOperand(0)->memoryrange.memId;
    Operation* insertOp = nullptr;
    for (auto op : state_.newOperations) {
        auto& reqMemids = state_.GetOpMemIds(op);
        if (std::find(reqMemids.begin(), reqMemids.end(), spillMemid) != reqMemids.end()) {
            insertOp = op;
        }
    }
    if (insertOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Insert spillout %s failed.", state_.GetOpInfo(copyoutOp).c_str());
        return FAILED;
    }
    if (insertOp->GetAtomicScopeId() >= VF_CLUSTER_ID_START) {
        int scopeId = insertOp->GetAtomicScopeId();
        auto opsIt = state_.vfScopeOpsByScope.find(scopeId);
        if (opsIt == state_.vfScopeOpsByScope.end() || opsIt->second.empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Insert spillout %s failed: unknown vf scope %d.",
                              state_.GetOpInfo(copyoutOp).c_str(), scopeId);
            return FAILED;
        }
        std::unordered_set<Operation*> scopeOpSet(opsIt->second.begin(), opsIt->second.end());
        Operation* scopeLastOp = nullptr;
        size_t foundCnt = 0;
        for (auto op : state_.newOperations) {
            if (scopeOpSet.count(op) > 0) {
                foundCnt++;
                scopeLastOp = op;
            }
        }
        if (foundCnt != scopeOpSet.size()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Insert spillout %s failed: vf scope %d not fully launch(%zu/%zu).",
                              state_.GetOpInfo(copyoutOp).c_str(), scopeId, foundCnt, scopeOpSet.size());
            return FAILED;
        }
        auto lastIt = find(state_.newOperations.begin(), state_.newOperations.end(), scopeLastOp);
        size_t copyoutPos = static_cast<size_t>(std::distance(state_.newOperations.begin(), lastIt)) + 1;
        size_t insertPos = InsertSkipOpsBeforeCopyout(copyoutOp, copyoutPos);
        state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), copyoutOp);
        return SUCCESS;
    }
    auto it = find(state_.newOperations.begin(), state_.newOperations.end(), insertOp);
    if (it != state_.newOperations.end()) {
        size_t copyoutPos = static_cast<size_t>(std::distance(state_.newOperations.begin(), it)) + 1;
        size_t insertPos = InsertSkipOpsBeforeCopyout(copyoutOp, copyoutPos);
        state_.newOperations.insert(state_.newOperations.begin() + static_cast<long>(insertPos), copyoutOp);
    }
    return SUCCESS;
}

Status OoOScheduler::ApplySpillContext(SpillContext& ctx, Operation* allocOp)
{
    for (auto* copyoutOp : ctx.newCopyoutOps) {
        if (InsertSpillCopyoutOp(copyoutOp) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ApplySpillContext: insert copyout %s failed.",
                              state_.GetOpInfo(copyoutOp).c_str());
            return FAILED;
        }
    }
    state_.numTotalIssues += ctx.newNotRetiredCopyOutSize - ctx.deleteNotRetiredOpSize;
    MemoryType memType = allocOp->GetOutputOperand(0)->GetMemoryTypeOriginal();

    for (auto deleteOp : ctx.deleteAllocOps) {
        state_.allocIssueQueue[std::get<2>(deleteOp)][std::get<1>(deleteOp)].DeleteOp(std::get<0>(deleteOp));
    }
    for (auto& newAllocOp : ctx.newAllocOps) {
        state_.allocIssueQueue[state_.schedInfoMap[newAllocOp].coreLocation][memType].Insert(newAllocOp);
    }

    // spill 改写了依赖的非退休 op 若已被旧前驱唤醒进 issue 队列，摘出后按新前驱重新武装，
    // 防止新 alloc 尚未执行 (tensorOccupyMap 无账) 时携带旧内存归属被发射。
    std::unordered_set<Operation*> requeued;
    for (auto* op : ctx.requeueOps) {
        if (!requeued.insert(op).second) {
            continue;
        }
        auto& pipeQueue = issueQueues[state_.schedInfoMap[op].coreLocation][state_.schedInfoMap[op].pipeType];
        if (pipeQueue.curOp == op) {
            continue;
        }
        pipeQueue.DeleteOp(op);
        bool ready = true;
        for (auto* pred : state_.depManager.GetPredecessors(op)) {
            if (!state_.schedInfoMap[pred].isRetired) {
                ready = false;
                break;
            }
        }
        if (ready) {
            pipeQueue.Insert(op);
            APASS_LOG_DEBUG_F(Elements::Operation, "    Requeue: %s, execOrder: %d", state_.GetOpInfo(op).c_str(),
                              state_.schedInfoMap[op].execOrder);
        }
    }

    for (auto memId : ctx.spillMemIds) {
        state_.localBufferMap[memId]->retireCycle = state_.clock;
    }
    return SUCCESS;
}

Status OoOScheduler::PrintSpillFailedInfo(Operation* allocOp)
{
    auto memType = state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->memType;
    APASS_LOG_ERROR_F(Elements::Operation, "======== OoO Spill failed info ===========");
    APASS_LOG_ERROR_F(Elements::Operation, "Spill failed memoryType: %s. %s", MemoryTypeToString(memType).c_str(),
                      GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- alloc request ----");
    APASS_LOG_ERROR_F(Elements::Operation, "op:%s need buffer size: %lu. %s", state_.GetOpInfo(allocOp).c_str(),
                      state_.localBufferMap[state_.GetOpMemIds(allocOp)[0]]->size,
                      GetFormatBacktrace(*allocOp).c_str());

    APASS_LOG_ERROR_F(Elements::Operation, "---- current buffer occupancy ----");
    for (auto& [memId, occupyOp] : state_.tensorOccupyMap) {
        if (state_.localBufferMap[memId]->memType != memType) {
            continue;
        }
        APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d], size:%lu, range[%lu,%lu], last writer: %s. %s", memId,
                          state_.localBufferMap[memId]->size, state_.localBufferMap[memId]->start,
                          state_.localBufferMap[memId]->end, state_.GetOpInfo(occupyOp).c_str(),
                          GetFormatBacktrace(*occupyOp).c_str());
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
