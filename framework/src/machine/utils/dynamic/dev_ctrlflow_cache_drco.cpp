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
 * \file dev_ctrlflow_cache_drco.cpp
 * \brief AICore dependency resolution (DRCO) ctrl-flow cache backup, replay, and reloc.
 */

#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk::dynamic {

namespace {

template <typename T>
void PublishOther(T*& published, T* ping, T* pong)
{
    /* Sparse slots are never backed up; ping and pong are published together. */
    if (ping == nullptr) {
        return;
    }
    published = (published == ping) ? pong : ping;
}

void RefreshInactive(void* published, void* ping, void* pong, const void* canon, size_t bytes)
{
    void* inactive = (published == ping) ? pong : ping;
    DevMemcpyS(inactive, bytes, canon, bytes);
}

size_t PerCoreImageBytes(const npu::tile_fwk::PerCorePendingQueue* queue)
{
    return sizeof(npu::tile_fwk::PerCorePendingQueue) +
           static_cast<size_t>(queue->size) * sizeof(npu::tile_fwk::LeafTaskId);
}

size_t LocalReadyImageBytes(const npu::tile_fwk::DrcoLocalReadyQueue* queue)
{
    return sizeof(npu::tile_fwk::DrcoLocalReadyQueue) +
           static_cast<size_t>(queue->size) * sizeof(npu::tile_fwk::LeafTaskId);
}

/* canon and pong are fresh copies; ping stays the recording-time live object. Null src is a hole. */
template <typename T, typename Alloc>
bool BackupShadowPair(Alloc&& alloc, T* src, size_t bytes, T*& canon, T*& ping, T*& pong)
{
    if (src == nullptr) {
        return true;
    }
    auto* canonBuf = static_cast<T*>(alloc(bytes));
    auto* pongBuf = static_cast<T*>(alloc(bytes));
    if (canonBuf == nullptr || pongBuf == nullptr) {
        return false;
    }
    (void)memcpy_s(canonBuf, bytes, src, bytes);
    (void)memcpy_s(pongBuf, bytes, src, bytes);
    canon = canonBuf;
    ping = src;
    pong = pongBuf;
    return true;
}

} // namespace

bool DevControlFlowCache::BackupDrcoPredCount(DynFuncDataBackup* dynDataBackup, uint32_t pingOff, uint32_t opSize)
{
    size_t drcoBytes = static_cast<size_t>(opSize) * sizeof(int32_t);
    auto* image = reinterpret_cast<int32_t*>(AllocateCache(drcoBytes));
    auto* pong = reinterpret_cast<int32_t*>(AllocateCache(drcoBytes));
    if (image == nullptr || pong == nullptr) {
        return false;
    }
    for (uint32_t i = 0; i < opSize; ++i) {
        image[i] = static_cast<int32_t>(dynDataBackup->predCountBackup[i]);
    }
    DevMemcpyS(pong, drcoBytes, image, drcoBytes);
    uint64_t pongRaw = reinterpret_cast<uint64_t>(pong) - reinterpret_cast<uint64_t>(this);
    /* Low 31 bits are the cache offset; the high bit selects metadata vs ctrl-cache base. */
    if (pongRaw >= (1ULL << 31)) {
        DEV_ERROR(ProgEncodeErr::METADATA_SIZE_OVERFLOW_4G,
                  "#ctrl.cache.record: DRCO predcount pong offset %lu does not fit in 31 bits.",
                  static_cast<unsigned long>(pongRaw));
        isRecordingStopped = true;
        return false;
    }
    dynDataBackup->drcoPredCountImage = image;
    dynDataBackup->drcoPredCountPingOffset = pingOff;
    dynDataBackup->drcoPredCountPongOffset = static_cast<uint32_t>(pongRaw);
    dynDataBackup->drcoPredCountBytes = static_cast<uint32_t>(drcoBytes);
    return true;
}

namespace {

void RestoreDrcoPredCountDup(DynDeviceTaskBase* base, size_t dupIndex)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    uint64_t metadataBase = reinterpret_cast<uint64_t>(base->drcoRootFuncList->metadataBase);
    uint64_t controlFlowCacheBase = reinterpret_cast<uint64_t>(base->drcoRootFuncList->controlFlowCacheBase);
    uint64_t funcDynamicDataOffset = dynFuncDataList->GetDrcoRootFuncData(dupIndex).funcDynamicDataOffset;
    uint32_t predCountOffset = static_cast<uint32_t>(funcDynamicDataOffset);
    uint64_t predCountBase = (predCountOffset & 0x80000000u) ? metadataBase : controlFlowCacheBase;
    int32_t* drcoPredCount = reinterpret_cast<int32_t*>(predCountBase + (predCountOffset & 0x7fffffffu));
    predcount_t* predCountBackup = base->dynFuncDataBackupList->At(dupIndex).predCountBackup;
    uint32_t opSize = base->dynFuncDataCacheList->At(dupIndex).duppedData->GetOperationSize();
    for (uint32_t i = 0; i < opSize; ++i) {
        drcoPredCount[i] = static_cast<int32_t>(predCountBackup[i]);
    }
}

} // namespace

void DevControlFlowCache::DrcoPredCountDataRestore(DynDeviceTaskBase* base)
{
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        RestoreDrcoPredCountDup(base, dupIndex);
    }
}

bool DevControlFlowCache::BackupDrcoPerCorePending(ReadyQueueCache* readyQueueBackup, DynDeviceTaskBase* base)
{
    for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
        auto* src = base->drcoRootFuncList->perCorePendingQueueArray[i];
        if (src == nullptr) {
            continue;
        }
        size_t backupSize = sizeof(npu::tile_fwk::PerCorePendingQueue) + sizeof(npu::tile_fwk::LeafTaskId) * src->size;
        auto* dst = reinterpret_cast<npu::tile_fwk::PerCorePendingQueue*>(AllocateCache(backupSize));
        if (dst == nullptr) {
            /* AllocateCache already stopped recording. Do not publish a backup with a missing canon. */
            return false;
        }
        memcpy_s(dst, backupSize, src, backupSize);
        readyQueueBackup->perCorePendingQueueList[i] = dst;
    }
    return true;
}

void DevControlFlowCache::DrcoReadyQueueDataRestore(DynDeviceTaskBase* base, uint32_t nrValidAic)
{
    auto* perCorePendingQueueArray = base->drcoRootFuncList->perCorePendingQueueArray;
    for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
        auto* dst = perCorePendingQueueArray[i];
        if (dst == nullptr) {
            continue;
        }
        dst->head = 0;
        dst->tail = 0;
        dst->size = 0;
    }
    // middleQueue 回放无需恢复：队列内容只读不变量（设备不写回 GM），slab 镜像跨轮有效
    // ——与 v2（9a6f7a7ff）验证通过的形态一致；v3 曾加 backup/Restore 回退（见 Backup 注释）
    const uint32_t nrAivCores = nrValidAic * 2;
    ReadyCoreFunctionQueue* aivQueue = base->readyQueue[0];
    ReadyCoreFunctionQueue* aicQueue = base->readyQueue[1];
    uint32_t wrapCoreIdx = 0;

    // Restore wrap (mix) tasks: AIC task to core w, AIV tasks to paired vector cores.
    // Mirrors DispatchReadyQueueToCores so mix-task dispatch is not lost on cache replay.
    WrapInfoQueue* wrapQueue = reinterpret_cast<WrapInfoQueue*>(base->devTask.mixTaskData.readyWrapCoreFunctionQue);
    if (wrapQueue != nullptr && nrValidAic > 0) {
        constexpr uint8_t MIX_TYPE_1C2V = 2;
        constexpr uint8_t MIX_TYPE_1C1V = 1;
        for (uint8_t targetType : {MIX_TYPE_1C2V, MIX_TYPE_1C1V}) {
            for (uint32_t idx = wrapQueue->head; idx < wrapQueue->tail; idx++) {
                WrapInfo& info = wrapQueue->elem[idx];
                if (info.mixResourceType != targetType) {
                    continue;
                }
                uint32_t aicCore = wrapCoreIdx++ % nrValidAic;
                uint32_t aiv0Core = nrValidAic + aicCore * 2;
                perCorePendingQueueArray[aicCore]->UnsafeEnqueue(info.tasklist[WRAP_IDX_AIC]);
                perCorePendingQueueArray[aiv0Core]->UnsafeEnqueue(info.tasklist[WRAP_IDX_AIV0]);
                if (targetType == MIX_TYPE_1C2V) {
                    perCorePendingQueueArray[aiv0Core + 1]->UnsafeEnqueue(info.tasklist[WRAP_IDX_AIV1]);
                }
            }
        }
    }

    uint32_t aicIdx = wrapCoreIdx;
    for (const auto* it = aicQueue->begin(); it != aicQueue->end(); ++it) {
        perCorePendingQueueArray[aicIdx++ % nrValidAic]->UnsafeEnqueue(*it);
    }
    uint32_t aivIdx = wrapCoreIdx * 2;
    for (const auto* it = aivQueue->begin(); it != aivQueue->end(); ++it) {
        perCorePendingQueueArray[nrValidAic + (aivIdx++ % nrAivCores)]->UnsafeEnqueue(*it);
    }

    __sync_synchronize();
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        base->drcoRootFuncList->devTaskCountList.count[ct].staticExecCount = 0;
        // 动态分片计数复位（cache 重放不残留上次运行的动态 leaf 计数）
        for (uint32_t s = 0; s < npu::tile_fwk::DRCO_COUNT_SHARD_NUM; s++) {
            base->drcoRootFuncList->devTaskCountList.shardExecCount[ct][s].value = 0;
        }
    }
    // 就绪队列复位按各队列自身容量清理（MIX 行容量 = coreFunctionCnt + HUB_MIX op 数，
    // 大于 coreFunctionCnt）；三行（AIC/AIV/MIX）均全组分配
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            auto* dst = base->drcoRootFuncList->localReadyQueueArray[ct][i];
            if (dst == nullptr) {
                continue;
            }
            dst->head = 0;
            dst->tail = 0;
            uint32_t taskListSize = dst->size * sizeof(npu::tile_fwk::LeafTaskId);
            (void)memset_s(reinterpret_cast<uint8_t*>(dst) + sizeof(DrcoLocalReadyQueue), taskListSize, 0,
                           taskListSize);
        }
    }
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        auto* matrix = base->drcoRootFuncList->localReadyMatrixArray[ct];
        if (matrix == nullptr) {
            continue;
        }
        // 定长矩阵：整块清零行区（含未用行/槽，覆盖所有可能残留的任务）
        (void)memset_s(matrix->rowList, sizeof(matrix->rowList), 0, sizeof(matrix->rowList));
    }
    // 槽内为 stitch 节点指针，复位为 nullptr，避免 cache 重放后残留悬空指针
    auto* stitchNodeMatrix = base->drcoRootFuncList->stitchNodeMatrix;
    if (stitchNodeMatrix != nullptr) {
        (void)memset_s(stitchNodeMatrix->stitchNodeList, sizeof(stitchNodeMatrix->stitchNodeList), 0,
                       sizeof(stitchNodeMatrix->stitchNodeList));
    }
    // 兜底 hub 矩阵槽内存 taskId，复位为 0，避免 cache 重放后残留任务
    auto* hubTaskMatrix = base->drcoRootFuncList->hubTaskMatrix;
    if (hubTaskMatrix != nullptr) {
        (void)memset_s(hubTaskMatrix->hubTaskList, sizeof(hubTaskMatrix->hubTaskList), 0,
                       sizeof(hubTaskMatrix->hubTaskList));
    }
    base->drcoRootFuncList->devTaskFinished = 0;
    auto* finishFlagList = &base->drcoRootFuncList->devTaskFinishFlagList;
    (void)memset_s(reinterpret_cast<uint8_t*>(finishFlagList), sizeof(*finishFlagList), 0, sizeof(*finishFlagList));
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        base->drcoRootFuncList->readyMatrixPushCol[ct] = 0; // 重放复位全局 col 游标
    }
}

void DevControlFlowCache::BackupDrcoReplayShadow(DynDeviceTaskBase* base)
{
    ReadyQueueCache* backup = base->readyQueueBackup;
    auto* root = base->drcoRootFuncList;
    if (backup == nullptr || root == nullptr || isRecordingStopped) {
        return;
    }
    auto alloc = [this](size_t bytes) { return AllocateCache(bytes); };
    const uint32_t capacity = base->devTask.coreFunctionCnt;
    for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; ++i) {
        auto* src = root->perCorePendingQueueArray[i];
        /* Sparse cores leave both src and canon null; ReadyQueueDataBackup publishes them together. */
        if (src == nullptr) {
            continue;
        }
        auto* canon = backup->perCorePendingQueueList[i];
        /* Pong is sized for later enqueue, not just the recorded occupancy. Canon already exists. */
        uint32_t cap = capacity > src->size ? capacity : src->size;
        size_t workBytes = sizeof(npu::tile_fwk::PerCorePendingQueue) +
                           static_cast<size_t>(cap) * sizeof(npu::tile_fwk::LeafTaskId);
        auto* pong = static_cast<npu::tile_fwk::PerCorePendingQueue*>(AllocateCache(workBytes));
        if (pong == nullptr) {
            return;
        }
        (void)memcpy_s(pong, workBytes, canon, PerCoreImageBytes(canon));
        backup->perCorePendingPing[i] = src;
        backup->perCorePendingPong[i] = pong;
    }
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ++ct) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; ++i) {
            auto* ready = root->localReadyQueueArray[ct][i];
            if (ready != nullptr &&
                !BackupShadowPair(alloc, ready, LocalReadyImageBytes(ready), backup->localReadyCanon[ct][i],
                                  backup->localReadyPing[ct][i], backup->localReadyPong[ct][i])) {
                return;
            }
        }
        auto* matrix = root->localReadyMatrixArray[ct];
        if (matrix != nullptr &&
            !BackupShadowPair(alloc, matrix, sizeof(npu::tile_fwk::DrcoLocalReadyMatrix), backup->localMatrixCanon[ct],
                              backup->localMatrixPing[ct], backup->localMatrixPong[ct])) {
            return;
        }
    }
    if (!BackupShadowPair(alloc, root->stitchNodeMatrix, sizeof(npu::tile_fwk::DrcoGlobalStitchNodeMatrix),
                          backup->stitchCanon, backup->stitchPing, backup->stitchPong) ||
        !BackupShadowPair(alloc, root->hubTaskMatrix, sizeof(npu::tile_fwk::DrcoGlobalHubTaskMatrix), backup->hubCanon,
                          backup->hubPing, backup->hubPong) ||
        isRecordingStopped) {
        return;
    }
    backup->drcoShadowReady = 1;
}

void DevControlFlowCache::DrcoShadowPingPongSwap(DynDeviceTaskBase* base)
{
    ReadyQueueCache* backup = base->readyQueueBackup;
    auto* root = base->drcoRootFuncList;
    if (backup == nullptr || root == nullptr) {
        return;
    }
    for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; ++i) {
        PublishOther(root->perCorePendingQueueArray[i], backup->perCorePendingPing[i], backup->perCorePendingPong[i]);
    }
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ++ct) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; ++i) {
            PublishOther(root->localReadyQueueArray[ct][i], backup->localReadyPing[ct][i],
                         backup->localReadyPong[ct][i]);
        }
        PublishOther(root->localReadyMatrixArray[ct], backup->localMatrixPing[ct], backup->localMatrixPong[ct]);
    }
    PublishOther(root->stitchNodeMatrix, backup->stitchPing, backup->stitchPong);
    PublishOther(root->hubTaskMatrix, backup->hubPing, backup->hubPong);
    root->devTaskFinished = 0;
    /* Match mainline rebuild path: recording/prior launch dirties these; replay must clear them. */
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        root->readyMatrixPushCol[ct] = 0;
        for (uint32_t s = 0; s < npu::tile_fwk::DRCO_COUNT_SHARD_NUM; s++) {
            root->devTaskCountList.shardExecCount[ct][s].value = 0;
        }
    }
    npu::tile_fwk::DrcoRootFuncListRestorePrecount(root, base->drcoPrecountExecuted, base->drcoPrecountFinishFlag);
}

void DevControlFlowCache::DrcoShadowPingPongRestore(DynDeviceTaskBase* base)
{
    ReadyQueueCache* backup = base->readyQueueBackup;
    auto* root = base->drcoRootFuncList;
    /* No shadow means this launch already rebuilt the live queues. Refreshing here would copy a partial backup. */
    if (backup == nullptr || root == nullptr || backup->drcoShadowReady == 0) {
        return;
    }
    for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; ++i) {
        auto* canon = backup->perCorePendingQueueList[i];
        if (canon == nullptr) {
            continue;
        }
        RefreshInactive(root->perCorePendingQueueArray[i], backup->perCorePendingPing[i], backup->perCorePendingPong[i],
                        canon, PerCoreImageBytes(canon));
    }
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ++ct) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; ++i) {
            auto* readyCanon = backup->localReadyCanon[ct][i];
            if (readyCanon != nullptr) {
                RefreshInactive(root->localReadyQueueArray[ct][i], backup->localReadyPing[ct][i],
                                backup->localReadyPong[ct][i], readyCanon, LocalReadyImageBytes(readyCanon));
            }
        }
        auto* matrixCanon = backup->localMatrixCanon[ct];
        if (matrixCanon != nullptr) {
            RefreshInactive(root->localReadyMatrixArray[ct], backup->localMatrixPing[ct], backup->localMatrixPong[ct],
                            matrixCanon, sizeof(npu::tile_fwk::DrcoLocalReadyMatrix));
        }
    }
    if (backup->stitchCanon != nullptr) {
        RefreshInactive(root->stitchNodeMatrix, backup->stitchPing, backup->stitchPong, backup->stitchCanon,
                        sizeof(npu::tile_fwk::DrcoGlobalStitchNodeMatrix));
    }
    if (backup->hubCanon != nullptr) {
        RefreshInactive(root->hubTaskMatrix, backup->hubPing, backup->hubPong, backup->hubCanon,
                        sizeof(npu::tile_fwk::DrcoGlobalHubTaskMatrix));
    }
}

void DevControlFlowCache::DrcoPredCountPingPongSwap(DynDeviceTaskBase* base)
{
    if (base->drcoRootFuncList == nullptr) {
        return;
    }
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncDataBackup* dynDataBackup = &base->dynFuncDataBackupList[dupIndex];
        if (dynDataBackup->drcoPredCountImage == nullptr || dynDataBackup->drcoPredCountBytes == 0) {
            /* Pong was not allocated. Rewrite the live buffer from the uint16 backup. */
            if (dynDataBackup->predCountBackup != nullptr) {
                RestoreDrcoPredCountDup(base, dupIndex);
            }
            continue;
        }
        auto& packed = dynFuncDataList->GetDrcoRootFuncData(static_cast<int>(dupIndex)).funcDynamicDataOffset;
        uint32_t current = static_cast<uint32_t>(packed);
        uint32_t next = (current == dynDataBackup->drcoPredCountPingOffset) ? dynDataBackup->drcoPredCountPongOffset :
                                                                              dynDataBackup->drcoPredCountPingOffset;
        packed = (packed & 0xffffffff00000000ull) | next;
    }
}

void DevControlFlowCache::DrcoPredCountPingPongRestore(DynDeviceTaskBase* base)
{
    if (base->drcoRootFuncList == nullptr) {
        return;
    }
    DynFuncHeader* dynFuncDataList = base->GetDynFuncDataList();
    uint64_t cacheBase = reinterpret_cast<uint64_t>(base->drcoRootFuncList->controlFlowCacheBase);
    uint64_t metaBase = reinterpret_cast<uint64_t>(base->drcoRootFuncList->metadataBase);
    auto addrOf = [&](uint32_t off) -> int32_t* {
        uint64_t baseAddr = (off & 0x80000000u) ? metaBase : cacheBase;
        return reinterpret_cast<int32_t*>(baseAddr + (off & 0x7fffffffu));
    };
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
        DynFuncDataBackup* dynDataBackup = &base->dynFuncDataBackupList[dupIndex];
        if (dynDataBackup->drcoPredCountImage == nullptr || dynDataBackup->drcoPredCountBytes == 0) {
            continue;
        }
        uint32_t current = static_cast<uint32_t>(
            dynFuncDataList->GetDrcoRootFuncData(static_cast<int>(dupIndex)).funcDynamicDataOffset);
        uint32_t inactiveOff = (current == dynDataBackup->drcoPredCountPingOffset) ?
                                   dynDataBackup->drcoPredCountPongOffset :
                                   dynDataBackup->drcoPredCountPingOffset;
        DevMemcpyS(addrOf(inactiveOff), dynDataBackup->drcoPredCountBytes, dynDataBackup->drcoPredCountImage,
                   dynDataBackup->drcoPredCountBytes);
    }
}

void DevControlFlowCache::RelocDrcoReadyQueueBackup(RelocRange& relocCtrlCache, ReadyQueueCache* readyQueueBackup)
{
    for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
        if (readyQueueBackup->perCorePendingQueueList[i] != nullptr) {
            RelocControlFlowCachePointer(readyQueueBackup->perCorePendingQueueList[i], relocCtrlCache);
        }
        relocCtrlCache.RelocNullable(readyQueueBackup->perCorePendingPing[i]);
        relocCtrlCache.RelocNullable(readyQueueBackup->perCorePendingPong[i]);
    }
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            relocCtrlCache.RelocNullable(readyQueueBackup->localReadyCanon[ct][i]);
            relocCtrlCache.RelocNullable(readyQueueBackup->localReadyPing[ct][i]);
            relocCtrlCache.RelocNullable(readyQueueBackup->localReadyPong[ct][i]);
        }
        relocCtrlCache.RelocNullable(readyQueueBackup->localMatrixCanon[ct]);
        relocCtrlCache.RelocNullable(readyQueueBackup->localMatrixPing[ct]);
        relocCtrlCache.RelocNullable(readyQueueBackup->localMatrixPong[ct]);
    }
    relocCtrlCache.RelocNullable(readyQueueBackup->stitchCanon);
    relocCtrlCache.RelocNullable(readyQueueBackup->stitchPing);
    relocCtrlCache.RelocNullable(readyQueueBackup->stitchPong);
    relocCtrlCache.RelocNullable(readyQueueBackup->hubCanon);
    relocCtrlCache.RelocNullable(readyQueueBackup->hubPing);
    relocCtrlCache.RelocNullable(readyQueueBackup->hubPong);
}

void DevControlFlowCache::RelocDrcoRootFuncList(RelocRange& relocCtrlCache, RelocRange& relocProgram,
                                                DynDeviceTaskBase* dynTaskBase)
{
    npu::tile_fwk::DrcoRootFuncList*& drcoRootFuncListRef = dynTaskBase->drcoRootFuncList;
    npu::tile_fwk::DrcoRootFuncList* drcoRootFuncList = nullptr;
    if (drcoRootFuncListRef != nullptr) {
        drcoRootFuncList = RelocControlFlowCachePointer(drcoRootFuncListRef, relocCtrlCache);
    }
    if (drcoRootFuncList != nullptr) {
        for (size_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
            relocCtrlCache.Reloc(drcoRootFuncList->perCorePendingQueueArray[i]);
            for (size_t sl = 0; sl < npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM; sl++) {
                relocCtrlCache.RelocNullable(drcoRootFuncList->perCoreMiddleQueueArray[i][sl]);
            }
        }

        for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
            for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
                relocCtrlCache.Reloc(drcoRootFuncList->localReadyQueueArray[ct][i]);
            }
        }
        for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
            relocCtrlCache.Reloc(drcoRootFuncList->localReadyMatrixArray[ct]);
        }
        relocCtrlCache.Reloc(drcoRootFuncList->stitchNodeMatrix);
        // 基址与 stitch 节点指针同域，走同一平移；slot 内偏移相对基址，平移不变
        relocCtrlCache.Reloc(drcoRootFuncList->stitchNodeBase);
        relocCtrlCache.Reloc(drcoRootFuncList->hubTaskMatrix);
        relocCtrlCache.Reloc(drcoRootFuncList->metadataBase);
        relocCtrlCache.Reloc(drcoRootFuncList->controlFlowCacheBase);
        relocProgram.Reloc(drcoRootFuncList->programBase);
    }
}

} // namespace npu::tile_fwk::dynamic
