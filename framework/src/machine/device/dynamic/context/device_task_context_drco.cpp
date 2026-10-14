/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_task_context_drco.cpp
 * \brief AICore dependency resolve (DRCO) helpers of DeviceTaskContext, gated by enableAicoreResolve.
 */

#include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {

void DeviceTaskContext::InitDrcoRootFuncList(DynDeviceTask* dyntask)
{
    auto* rootFuncList = reinterpret_cast<npu::tile_fwk::DrcoRootFuncList*>(
        ControlFlowAllocateSlab(
            devProg_, sizeof(npu::tile_fwk::DrcoRootFuncList),
            workspace_->SlabAlloc(sizeof(npu::tile_fwk::DrcoRootFuncList), WsAicpuSlabMemType::DYN_FUNC_DATA))
            .ptr);
    dyntask->drcoRootFuncList = rootFuncList;
    uint32_t queueCapacity = dyntask->devTask.coreFunctionCnt;
    uint32_t aicTaskCnt = 0;
    uint32_t aivTaskCnt = 0;
    for (uint64_t funcId = 0; funcId < dyntask->dynFuncDataCacheListSize; ++funcId) {
        auto callList = dyntask->dynFuncDataCacheList[funcId].calleeList;
        for (size_t opIndex = 0; opIndex < dyntask->dynFuncDataCacheList[funcId].devFunc->GetOperationSize();
             ++opIndex) {
            auto coreType = dyntask->cceBinary[callList[opIndex]].coreType;
            if (coreType == static_cast<int>(CoreType::AIC)) {
                aicTaskCnt++;
            } else if (coreType == static_cast<int>(CoreType::AIV)) {
                aivTaskCnt++;
            }
        }
    }
    // 完成计数表内嵌 rootFuncList（承接原全局队列的 executedCount/size），MIX 恒为 0
    new (&rootFuncList->devTaskCountList) npu::tile_fwk::DrcoDevTaskCountList();
    rootFuncList->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIC].size = aicTaskCnt;
    rootFuncList->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIV].size = aivTaskCnt;
    uint32_t perCoreSize = sizeof(npu::tile_fwk::PerCorePendingQueue) +
                           queueCapacity * sizeof(npu::tile_fwk::LeafTaskId);
    for (uint32_t i = 0; i < npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE; i++) {
        auto* perCoreQueue = workspace_->AllocatePerCorePendingQueue(perCoreSize);
        new (perCoreQueue) npu::tile_fwk::PerCorePendingQueue();
        rootFuncList->perCorePendingQueueArray[i] = perCoreQueue;
    }
    uint32_t nrValidAic = devProg_->devArgs.nrValidAic;
    uint32_t localSize = sizeof(npu::tile_fwk::DrcoLocalReadyQueue);
    localSize += queueCapacity * sizeof(npu::tile_fwk::LeafTaskId);
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        for (uint32_t i = 0; i < npu::tile_fwk::NUM_LOCAL_GROUPS; i++) {
            auto* localQueue = workspace_->AllocateDrcoLocalReadyQueue(localSize);
            new (localQueue) npu::tile_fwk::DrcoLocalReadyQueue(queueCapacity);
            (void)memset_s(reinterpret_cast<uint8_t*>(localQueue) + sizeof(DrcoLocalReadyQueue),
                           queueCapacity * sizeof(LeafTaskId), 0, queueCapacity * sizeof(LeafTaskId));
            rootFuncList->localReadyQueueArray[ct][i] = localQueue;
        }
    }
    // LocalMatrix 分配（每类型单实例，定长 cacheline 行隔离布局：push 逐槽 u32 CAS /
    // pop 整字 u64 exch）：行 = 类型内本地编号（AIC 本地编号 = blockIdx ∈ [0, nrValidAic)，
    // AIV 本地编号 = blockIdx - nrValidAic ∈ [0, 2*nrValidAic)，MIX 行承载 HUB_MIX 后继、
    // 消费者为 AIC，validCoreNum 同 AIC 行）；
    // 每行列数 colNum（缓冲深度）按解依赖扇出自适应：rootFuncMaxCallOpsize（单任务最大直接后继数）
    // > 默认深度时加深到上限，容量 = colNum × validCoreNum，超出溢出 local ready queue
    uint32_t matrixColNum = devProg_->rootFuncMaxCallOpsize > npu::tile_fwk::DRCO_MATRIX_COL_DEFAULT ?
                                npu::tile_fwk::DRCO_MATRIX_COL_MAX :
                                npu::tile_fwk::DRCO_MATRIX_COL_DEFAULT;
    const uint32_t matrixValidCoreNums[npu::tile_fwk::DRCO_QUEUE_MAX] = {nrValidAic * 2, nrValidAic, nrValidAic};
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        auto* localMatrix = workspace_->AllocateDrcoLocalReadyMatrix(sizeof(npu::tile_fwk::DrcoLocalReadyMatrix));
        new (localMatrix) npu::tile_fwk::DrcoLocalReadyMatrix(matrixColNum, matrixValidCoreNums[ct]);
        rootFuncList->localReadyMatrixArray[ct] = localMatrix;
    }
    // 全局共享 stitch 节点矩阵（单实例）：行 = 全局 blockIdx（AIC [0,nrValidAic) + AIV
    // [nrValidAic,3*nrValidAic)），任何核可 push 任意行、每核只 pop 自己行；
    // slot 语义为节点相对 stitch pool 基址的 u32 偏移（0 = 空闲），基址随 rootFuncList
    // 透传给 aicore 侧
    auto* stitchNodeMatrix = workspace_->AllocateDrcoStitchNodeMatrix(
        sizeof(npu::tile_fwk::DrcoGlobalStitchNodeMatrix));
    new (stitchNodeMatrix) npu::tile_fwk::DrcoGlobalStitchNodeMatrix();
    rootFuncList->stitchNodeMatrix = stitchNodeMatrix;
    auto* ctrlCache = devProg_->GetControlFlowCache();
    rootFuncList->stitchNodeBase = (ctrlCache != nullptr && ctrlCache->IsRecording()) ?
                                       reinterpret_cast<uint64_t>(ctrlCache) :
                                       workspace_->GetStitchPoolBase();
    // 全核共享 hub 任务矩阵（单实例）：承接 hubStack 溢出与 stitch 类型越界的兜底改投，
    // 行 = 全局 blockIdx，任何核可 push 任意行、每核只 pop 自己行，fetch 循环无条件扫
    auto* hubTaskMatrix = workspace_->AllocateDrcoHubTaskMatrix(sizeof(npu::tile_fwk::DrcoGlobalHubTaskMatrix));
    new (hubTaskMatrix) npu::tile_fwk::DrcoGlobalHubTaskMatrix();
    rootFuncList->hubTaskMatrix = hubTaskMatrix;
    rootFuncList->controlFlowCacheBase = reinterpret_cast<void*>(ctrlCache);
    rootFuncList->metadataBase = reinterpret_cast<void*>(workspace_->GetMetadataBase());
    rootFuncList->programBase = reinterpret_cast<void*>(devProg_);
    rootFuncList->totalTaskCount = dyntask->devTask.coreFunctionCnt;
    rootFuncList->devTaskFinished = 0;
    new (&rootFuncList->devTaskFinishFlagList) npu::tile_fwk::DrcoDevTaskFinishFlagList();
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        rootFuncList->readyMatrixPushCol[ct] = 0; // 全局 col 游标起点
    }
}

void DeviceTaskContext::DispatchReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    ReadyCoreFunctionQueue* aivQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
    ReadyCoreFunctionQueue* aicQueue = dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];

    uint32_t nrValidAic = devProg->devArgs.nrValidAic;
    uint32_t nrAivCores = 2 * nrValidAic;

    // perCorePendingQueue: distribute wrap (mix) tasks, AIC task to core w, AIV tasks to paired vector cores
    WrapInfoQueue* wrapQueue = reinterpret_cast<WrapInfoQueue*>(dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue);
    auto* perCorePendingQueueArray = dyntask->drcoRootFuncList->perCorePendingQueueArray;
    uint32_t wrapCoreIdx = 0;
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

    // perCorePendingQueue: distribute aic tasks to aicore cores (0..nrValidAic-1), continue from wrap dispatch
    uint32_t aicIdx = wrapCoreIdx;
    for (const auto* it = aicQueue->begin(); it != aicQueue->end(); ++it) {
        uint32_t coreIdx = aicIdx % nrValidAic;
        perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
        aicIdx++;
    }

    // perCorePendingQueue: distribute aiv tasks to aiv cores (nrValidAic..3*nrValidAic-1)
    uint32_t aivIdx = wrapCoreIdx * 2;
    for (const auto* it = aivQueue->begin(); it != aivQueue->end(); ++it) {
        uint32_t coreIdx = nrValidAic + (aivIdx % nrAivCores);
        perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
        aivIdx++;
    }
}

void DeviceTaskContext::DispatchDieReadyQueueToCores(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    if (!IsMultiDie(devProg)) {
        return;
    }
    uint32_t nrValidAic = devProg->devArgs.nrValidAic;
    uint32_t nrAivCores = 2 * nrValidAic;
    uint32_t halfAic = nrValidAic / 2;
    uint32_t halfAiv = nrAivCores / 2;
    // perCorePendingQueue: distribute die tasks, die0 to first-half cores, die1 to second-half cores
    for (uint32_t dieId = 0; dieId < DIE_NUM; ++dieId) {
        uint32_t aicCoreBase = dieId * halfAic;
        uint32_t aicCoreCnt = (dieId == 0) ? halfAic : (nrValidAic - halfAic);
        ReadyCoreFunctionQueue* dieAicQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAicCoreFunctionQue[dieId]);
        uint32_t dieAicIdx = 0;
        for (const auto* it = dieAicQue->begin(); it != dieAicQue->end(); ++it) {
            uint32_t coreIdx = aicCoreBase + (dieAicIdx % aicCoreCnt);
            dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            dieAicIdx++;
        }

        uint32_t aivCoreBase = nrValidAic + dieId * halfAiv;
        uint32_t aivCoreCnt = (dieId == 0) ? halfAiv : (nrAivCores - halfAiv);
        ReadyCoreFunctionQueue* dieAivQue = reinterpret_cast<ReadyCoreFunctionQueue*>(
            dyntask->devTask.dieReadyFunctionQue.readyDieAivCoreFunctionQue[dieId]);
        uint32_t dieAivIdx = 0;
        for (const auto* it = dieAivQue->begin(); it != dieAivQue->end(); ++it) {
            uint32_t coreIdx = aivCoreBase + (dieAivIdx % aivCoreCnt);
            dyntask->drcoRootFuncList->perCorePendingQueueArray[coreIdx]->UnsafeEnqueue(*it);
            dieAivIdx++;
        }
    }
}

void DeviceTaskContext::DrcoRefreshSpecialTaskId(DynDeviceTask* dyntask)
{
    auto encodeTaskId = [&](uint32_t taskId) -> uint32_t {
        uint32_t funcIdx = npu::tile_fwk::FuncID(taskId);
        uint64_t funcInfoAndStaticDataOffset = dyntask->dynFuncDataList->GetDrcoRootFuncData(funcIdx)
                                                   .funcInfoAndStaticDataOffset;
        DevAscendFunction* devFunc = dyntask->dynFuncDataCacheList[funcIdx].devFunc;
        int opIdx = static_cast<int>(npu::tile_fwk::TaskID(taskId));
        size_t succSize = 0;
        const uint32_t* succList = devFunc->GetOperationDepGraphSuccAddr(opIdx, succSize);
        if (succSize == 1 && npu::tile_fwk::TaskID(succList[0]) == devFunc->dummyEndingOpIdx_ &&
            (funcInfoAndStaticDataOffset & npu::tile_fwk::DRCO_ROOT_FUNC_MEMORY_REUSE_STITCHED_BIT) == 0 &&
            dyntask->dynFuncDataCacheList[funcIdx].duppedData->GetOperationStitch(opIdx).IsNull()) {
            return taskId | ((npu::tile_fwk::DRCO_SUCC_SPECIAL_PERCORE_SKIP_RESOLVE &
                              npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_MASK)
                             << npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_SHIFT);
        }
        return taskId;
    };

    auto refreshReadyQueue = [&](ReadyCoreFunctionQueue* queue) {
        for (uint32_t i = 0; i < queue->UnsafeSize(); ++i) {
            queue->Data()[i] = encodeTaskId(queue->Data()[i]);
        }
    };
    refreshReadyQueue(dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)]);
    refreshReadyQueue(dyntask->readyQueue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)]);

    WrapInfoQueue* wrapQueue = reinterpret_cast<WrapInfoQueue*>(dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue);
    if (wrapQueue != nullptr) {
        for (uint32_t idx = wrapQueue->head; idx < wrapQueue->tail; ++idx) {
            for (uint32_t j = 0; j < MAX_WRAP_TASK_NUM; ++j) {
                if (wrapQueue->elem[idx].tasklist[j] != AICORE_TASK_INIT) {
                    wrapQueue->elem[idx].tasklist[j] = encodeTaskId(wrapQueue->elem[idx].tasklist[j]);
                }
            }
        }
    }
}

void DeviceTaskContext::BuildDrcoRootFuncData(DynFuncData* dyndata, DrcoRootFuncData* rootFuncData,
                                              DevAscendFunctionDupped& stitchedFunc)
{
    DevAscendFunction* source = stitchedFunc.GetSource();
    uint32_t opSize = stitchedFunc.GetOperationSize();
    // AICore drco uses int32_t predCount because aicore atomicAdd only supports 32bit/64bit operands,
    // while the aicpu-side dupped data predCount stays uint16_t (sizeof(predcount_t)).
    uint32_t drcoPredCountSize = opSize * sizeof(int32_t);
    WsAllocation predAlloc = ControlFlowAllocateSlab(
        devProg_, drcoPredCountSize, workspace_->SlabAlloc(drcoPredCountSize, WsAicpuSlabMemType::PRED_COUNT));
    int32_t* aicorePredCount = predAlloc.As<int32_t>();
    // DRCO pair resolve does u64 atomicAdd on &predCount[even opIdx] (aikernel_data.h). The slab
    // allocator's 64B granularity keeps the base 64B aligned so even slots stay 8B aligned; an
    // unaligned base would fault the hardware atomic, so guard the implicit contract here.
    DEV_ASSERT_MSG(ProgEncodeErr::DYNFUNC_DATA_ALIGNMENT_ERROR, (predAlloc.ptr % 8ULL) == 0,
                   "#drco.predcount.align: base=%llu is not 8B aligned",
                   static_cast<unsigned long long>(predAlloc.ptr));
    for (uint32_t i = 0; i < opSize; ++i) {
        aicorePredCount[i] = static_cast<int32_t>(stitchedFunc.GetOperationCurrPredCount(i));
    }
    DevAscendFunctionDuppedStitch** succStitchList = &stitchedFunc.DupDataForDynFuncData()->GetStitch(0).Head();

    uint64_t programBase = reinterpret_cast<uint64_t>(devProg_);
    uint64_t metadataBase = workspace_->GetMetadataBase();

    auto* ctrlFlowCache = devProg_->GetControlFlowCache();
    uint64_t controlFlowCacheBase = reinterpret_cast<uint64_t>(ctrlFlowCache);
    uint64_t controlFlowCacheSize = (ctrlFlowCache != nullptr) ? ctrlFlowCache->GetSize() : 0;

    // DRCO 静态后继统一以 succNode 节点存储（program memory, zero copy）；节点区偏移打包进
    // funcInfoAndStaticDataOffset（bit59-30），DrcoRootFuncData 保持两字段形态。
    // The source operationSuccList_ stays untouched (shared with DRCU/aicpu resolve).
    auto* succInfoList = &source->GetOperationSuccInfo(0);
    uint64_t succInfoListOffset = reinterpret_cast<uint64_t>(succInfoList) - programBase;
    DEV_ASSERT_MSG(ProgEncodeErr::METADATA_SIZE_OVERFLOW_4G, succInfoListOffset < (1ULL << 30),
                   "#drco.succInfoList.offset: offset=%lu is over 30bit", succInfoListOffset);
    // 节点区偏移（相对 programBase，平移不变无需 reloc）：空表 func 写 0（设备侧仅在
    // succNodeSize > 0 时解引用）；位域 30bit 限定节点区末端 < 2^30（静态 succ slot 低
    // 32 位偏移同以 programBase 为原点，u32 跨度随之满足）
    uint64_t staticSuccNodeListOffset = 0;
    auto* staticSuccNodeList = source->GetStaticSuccNodeAddr();
    if (staticSuccNodeList != nullptr) {
        staticSuccNodeListOffset = reinterpret_cast<uint64_t>(staticSuccNodeList) - programBase;
        uint64_t staticSuccNodeListEnd = staticSuccNodeListOffset + source->GetStaticSuccNodeByteSize();
        DEV_ASSERT_MSG(ProgEncodeErr::METADATA_SIZE_OVERFLOW_4G, staticSuccNodeListEnd < (1ULL << 30),
                       "#drco.staticsucc.span: static succ node region end %llu exceeds 30bit offset from "
                       "program base",
                       static_cast<unsigned long long>(staticSuccNodeListEnd));
    }
    rootFuncData->funcInfoAndStaticDataOffset = succInfoListOffset | (staticSuccNodeListOffset << 30) |
                                                (stitchedFunc.GetDummyEndingUsed() ?
                                                     DRCO_ROOT_FUNC_MEMORY_REUSE_STITCHED_BIT :
                                                     0ULL);

    // predCount/succStitchList 可能落在 control flow cache（recording）或 metadata（非 recording），
    // offset 最高位（bit31）作为 base 标记：0 = controlFlowCacheBase，1 = metadataBase，低 31 位为字节偏移。
    uint64_t predCountAddr = reinterpret_cast<uint64_t>(aicorePredCount);
    uint64_t succStitchListAddr = reinterpret_cast<uint64_t>(succStitchList);
    bool predCountInCache = (ctrlFlowCache != nullptr) && predCountAddr >= controlFlowCacheBase &&
                            predCountAddr < controlFlowCacheBase + controlFlowCacheSize;
    bool succStitchListInCache = (ctrlFlowCache != nullptr) && succStitchListAddr >= controlFlowCacheBase &&
                                 succStitchListAddr < controlFlowCacheBase + controlFlowCacheSize;
    uint64_t predCountRawOffset = predCountAddr - (predCountInCache ? controlFlowCacheBase : metadataBase);
    uint64_t succStitchListRawOffset = succStitchListAddr -
                                       (succStitchListInCache ? controlFlowCacheBase : metadataBase);
    // bit31 保留为 base 标记，偏移须 < 2^31，否则与标记位冲突。
    DEV_ASSERT_MSG(ProgEncodeErr::METADATA_SIZE_OVERFLOW_4G, predCountRawOffset < (1ULL << 31),
                   "#drco.predcount.offset: offset=%lu is over 2G", predCountRawOffset);
    // Stitch successors are encoded at build time (MakeDrcoStitchTaskId in device_stitch_context),
    // the shared linked nodes are consumed directly, no flattened copy here.
    DEV_ASSERT_MSG(ProgEncodeErr::METADATA_SIZE_OVERFLOW_4G, succStitchListRawOffset < (1ULL << 31),
                   "#drco.succStitchList.offset: offset=%lu is over 2G", succStitchListRawOffset);
    uint32_t predCountOffset = static_cast<uint32_t>(predCountRawOffset) | (predCountInCache ? 0u : 0x80000000u);
    uint32_t succStitchListOffset = static_cast<uint32_t>(succStitchListRawOffset) |
                                    (succStitchListInCache ? 0u : 0x80000000u);
    rootFuncData->funcDynamicDataOffset = (static_cast<uint64_t>(succStitchListOffset) << 32) | predCountOffset;

    dyndata->cceBinaryIndexList = source->GetCalleeIndexAddr();
    // 子链长度编码（nodeNext 低 6 位）：录制/非录制路径统一编码；录制路径下
    // TaskAddrRelocProgramAndCtrlCache 遍历已按解码地址取 NextRaw，编码值可安全参与重定位。
    auto* duppedData = stitchedFunc.DupDataForDynFuncData();
    for (uint32_t i = 0; i < duppedData->GetStitchSize(); i++) {
        duppedData->GetStitch(i).EncodeStitchNodes();
    }
}

} // namespace npu::tile_fwk::dynamic
