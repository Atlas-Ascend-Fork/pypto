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

namespace {

// 收编门槛与静态缓冲容量：后继数不足门槛或超容量均回退老路径（fast path 是纯优化，
// 回退语义安全；容量按大扇出头上限留一倍余量）
constexpr uint32_t MIDDLE_QUEUE_MIN_SUCC = 16;
constexpr uint32_t MIDDLE_QUEUE_MAX_SUCC = 2048;

// per-func predCount 数组基址缓存：双基址解码（与设备侧 DrcoGetCurrentRootFuncData 同
// 编码），按目标函数单项缓存——同头成员通常集中在同一目标函数，命中率极高
struct PredArrCache {
    uint64_t metadataBase = 0;
    uint64_t controlFlowCacheBase = 0;
    uint32_t funcCached = 0xFFFFFFFFu;
    int32_t* arrCached = nullptr;
};

// 收集 op 的全量直接后继到 succBuf：dep graph 条目为本函数 opIdx（无 funcIdx 位，须用
// 头的 funcIdx 重组完整 taskId）、stitch 条目为跨函数完整 taskId（两者编码不同，见
// DumpTopo 同款解码）；大扇出头的后继可全挂 stitch 链（dep graph 为空）。粗过滤（dep
// 后继数 + stitch 空判）先行——绝大多数 op 两者皆小，免展开直接返回 0。返回后继数，
// 超静态容量返回 0（回退老路径）
uint32_t CollectDirectSuccessors(DevAscendFunction* devFunc, DevAscendFunctionDuppedData* duppedData, uint32_t headFunc,
                                 int32_t op, uint32_t* succBuf)
{
    size_t depSuccSize = 0;
    const uint32_t* depSuccs = devFunc->GetOperationDepGraphSuccAddr(static_cast<int>(op), depSuccSize);
    const auto& stitch = duppedData->GetOperationStitch(static_cast<int>(op), true);
    if (depSuccSize < MIDDLE_QUEUE_MIN_SUCC && stitch.IsNull()) {
        return 0;
    }
    uint32_t cnt = 0;
    for (size_t k = 0; k < depSuccSize; k++) {
        if (cnt >= MIDDLE_QUEUE_MAX_SUCC) {
            return 0; // 超静态容量：回退老路径
        }
        succBuf[cnt++] = npu::tile_fwk::MakeTaskID(headFunc, depSuccs[k] & TASKID_FROM_CTRL_TOPO_MASK);
    }
    for (auto* n = stitch.Head(); n != nullptr; n = n->Next()) {
        const uint32_t nodeSize = n->Size();
        for (uint32_t i = 0; i < nodeSize; i++) {
            if (cnt >= MIDDLE_QUEUE_MAX_SUCC) {
                return 0;
            }
            succBuf[cnt++] = n->At(i) & TASKID_FROM_CTRL_TOPO_MASK;
        }
    }
    return cnt;
}

// resolve 跳过判定（perCoreQueue 的 SKIP_RESOLVE 与 middleQueue 的 MIDDLE_QUEUE_MEMBER_SKIP 共用判据）：
// 唯一后继为本函数 dummyEnding 且无 stitch、本 func 未内存复用缝合 —— 该任务 resolve 到的
// 全是 dummy 边（设备侧本就跳过递减），无任何下游可解
bool IsSkipResolveSucc(DynDeviceTask* dyntask, uint32_t taskId)
{
    const uint32_t funcIdx = npu::tile_fwk::FuncID(taskId);
    const int opIdx = static_cast<int>(npu::tile_fwk::TaskID(taskId));
    DevAscendFunction* devFunc = dyntask->dynFuncDataCacheList[funcIdx].devFunc;
    size_t succSize = 0;
    const uint32_t* succList = devFunc->GetOperationDepGraphSuccAddr(opIdx, succSize);
    if (succSize != 1 || npu::tile_fwk::TaskID(succList[0]) != devFunc->dummyEndingOpIdx_) {
        return false;
    }
    const uint64_t funcInfoAndStaticDataOffset = dyntask->dynFuncDataList->GetDrcoRootFuncData(funcIdx)
                                                     .funcInfoAndStaticDataOffset;
    if ((funcInfoAndStaticDataOffset & npu::tile_fwk::DRCO_ROOT_FUNC_MEMORY_REUSE_STITCHED_BIT) != 0) {
        return false;
    }
    auto* duppedData = dyntask->dynFuncDataCacheList[funcIdx].duppedData;
    return duppedData == nullptr || duppedData->GetOperationStitch(opIdx, true).IsNull();
}

// HUB 搬入防御判定：编码侧会把"HUB 后继的后继表"整体搬入其唯一前驱头的解依赖表
// （dev_encode.cpp hubExpanded：HUB 本体表清空、边只在头表中解，HUB 链上按唯一前驱
// 奇偶交替展开保证每边恰搬一次）。middleQueue 跳过头 resolve 会使这些搬入边永不被解
// → 搬入目标 predCount 永不归零 → 挂死（HUB 成员消费侧 resolve 到的是空表，搬入边
// 物理上只存在于头表中）。
// 理论上可把搬入后继一并收编（全量做单前驱校验），但 hubExpanded 的展开
// 判定依赖编码期的唯一前驱链奇偶标记，host 精确复刻对齐成本高、且校验集扩大拖慢
// Host 热路径——保守处理：头后继中存在会被搬入展开的 HUB（自身有后继）即整体拒绝，
// 该头交老路径/主线散射
bool IsRelocatedHubSuccessor(DynDeviceTask* dyntask, uint32_t succFunc, uint32_t succOp)
{
    DevAscendFunction* hubFunc = dyntask->dynFuncDataCacheList[succFunc].devFunc;
    size_t hubDepSucc = 0;
    (void)hubFunc->GetOperationDepGraphSuccAddr(static_cast<int>(succOp), hubDepSucc);
    auto* duppedData = dyntask->dynFuncDataCacheList[succFunc].duppedData;
    const bool hubStitchEmpty = duppedData == nullptr ||
                                duppedData->GetOperationStitch(static_cast<int>(succOp), true).IsNull();
    return hubDepSucc > 0 || !hubStitchEmpty;
}

// 满覆盖 + 成员资格校验，合格成员原地压实（写位恒 ≤ 读位，免二次缓冲）：
// ① 全部后继唯一前驱 = 头（predCount==1，老路径永不发射，零哨兵前提）；② 核类型
// ∈ {AIC, AIV, HUB}——HUB_MIX 是 wrap 组头不可直接执行（老路径由 AIC 核经
// ResolveHubMixTask 解析 + C2V 派发），AICPU/GMATOMIC/MIX 同理不走 leaf 消费；
// ③ HUB 成员须未被搬入展开（见 IsRelocatedHubSuccessor）。任一不满足返回 0（整体拒绝）
uint32_t CollectEligibleMembers(DynDeviceTask* dyntask, uint64_t funcCount, PredArrCache& cache,
                                const uint32_t* succBuf, uint32_t succCnt, uint32_t* memberTask, uint32_t* memberCt)
{
    uint32_t memberCnt = 0;
    for (uint32_t k = 0; k < succCnt; k++) {
        const uint32_t succTask = succBuf[k] & TASKID_FROM_CTRL_TOPO_MASK;
        const uint32_t succFunc = npu::tile_fwk::FuncID(succTask);
        const uint32_t succOp = npu::tile_fwk::TaskID(succTask);
        if (succFunc >= funcCount) {
            return 0;
        }
        if (succFunc != cache.funcCached) {
            cache.funcCached = succFunc;
            const uint32_t offset = static_cast<uint32_t>(
                dyntask->dynFuncDataList->GetDrcoRootFuncData(succFunc).funcDynamicDataOffset);
            const uint64_t base = (offset & 0x80000000u) ? cache.metadataBase : cache.controlFlowCacheBase;
            cache.arrCached = reinterpret_cast<int32_t*>(base + (offset & 0x7fffffffu));
        }
        if (cache.arrCached[succOp] != 1) {
            return 0;
        }
        const int sct = dyntask->cceBinary[dyntask->dynFuncDataCacheList[succFunc].calleeList[succOp]].coreType;
        if (sct != static_cast<int>(CoreType::AIC) && sct != static_cast<int>(CoreType::AIV) &&
            sct != static_cast<int>(CoreType::HUB)) {
            return 0;
        }
        if (sct == static_cast<int>(CoreType::HUB) && IsRelocatedHubSuccessor(dyntask, succFunc, succOp)) {
            return 0;
        }
        memberTask[memberCnt] = succTask;
        memberCt[memberCnt] = static_cast<uint32_t>(sct);
        memberCnt++;
    }
    return memberCnt;
}

// 一解多直通成员撒核 + 入队：① 按核类型分组顺序轮转（AIC 组 [0, nrValidAic)、AIV 组
// [nrValidAic, 3*nrValidAic)）——顺序轮转保证每核至多 ⌈n/核数⌉ 个，避免取模撞核把长
// 成员波尾拉串行；② 容量校验（任一核超 queueCapacity 返回 false，由调用方回退注册走
// 老路径——失败发生在入队之前，无半途状态）；③ 入队打标（单槽 0，懒分配精确尺寸，
// leaf 打 MIDDLE_QUEUE_MEMBER / hub 打 MIDDLE_QUEUE_MEMBER_HUB；stitch 条目自带的 ct 位与 special 位段
// 重叠，入队前先清位段再打标，否则设备侧 ct 解码错乱）
bool SpreadMiddleQueueMembers(npu::tile_fwk::DrcoRootFuncList* rootFuncList, DeviceWorkspaceAllocator* workspace,
                              const uint32_t* memberTask, const uint32_t* memberCt, const bool* memberSkip,
                              uint32_t memberCnt, uint32_t headFlat, uint32_t slot, uint32_t nrValidAic,
                              uint32_t queueCapacity)
{
    uint32_t memberCore[MIDDLE_QUEUE_MAX_SUCC];
    uint32_t perCore[3 * npu::tile_fwk::MAX_AICORE_NUM_FOR_QUEUE];
    for (uint32_t c = 0; c < 3 * nrValidAic; c++) {
        perCore[c] = 0;
    }
    uint32_t aicSeq = 0;
    uint32_t aivSeq = 0;
    for (uint32_t mi = 0; mi < memberCnt; mi++) {
        memberCore[mi] = (memberCt[mi] == static_cast<uint32_t>(CoreType::AIC)) ?
                             (aicSeq++ % nrValidAic) :
                             nrValidAic + ((aivSeq++) % (2 * nrValidAic));
    }
    for (uint32_t mi = 0; mi < memberCnt; mi++) {
        perCore[memberCore[mi]]++;
    }
    for (uint32_t c = 0; c < 3 * nrValidAic; c++) {
        if (perCore[c] > queueCapacity) {
            return false;
        }
    }
    constexpr uint32_t specialShift = npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_SHIFT;
    auto* perCoreMiddleQueueArray = rootFuncList->perCoreMiddleQueueArray;
    for (uint32_t mi = 0; mi < memberCnt; mi++) {
        const uint32_t coreIdx = memberCore[mi];
        if (perCoreMiddleQueueArray[coreIdx][slot] == nullptr) {
            const uint32_t middleSize = sizeof(npu::tile_fwk::PerCoreMiddleQueue) +
                                        perCore[coreIdx] * sizeof(npu::tile_fwk::LeafTaskId);
            auto* middleQueue = reinterpret_cast<npu::tile_fwk::PerCoreMiddleQueue*>(
                workspace->AllocatePerCorePendingQueue(middleSize));
            new (middleQueue) npu::tile_fwk::PerCoreMiddleQueue();
            perCoreMiddleQueueArray[coreIdx][slot] = middleQueue;
        }
        auto* middleQueue = perCoreMiddleQueueArray[coreIdx][slot];
        middleQueue->headTaskId = headFlat;
        // leaf 成员命中 IsSkipResolveSucc 时打 MIDDLE_QUEUE_MEMBER_SKIP（消费侧跳过 resolve）；
        // hub 成员保持 MIDDLE_QUEUE_MEMBER_HUB 原行为（仅 resolve 不执行）
        uint32_t memberSpecial;
        if (memberCt[mi] == static_cast<uint32_t>(CoreType::HUB)) {
            memberSpecial = npu::tile_fwk::DRCO_SUCC_SPECIAL_MIDDLE_QUEUE_MEMBER_HUB;
        } else if (memberSkip != nullptr && memberSkip[mi]) {
            memberSpecial = npu::tile_fwk::DRCO_SUCC_SPECIAL_MIDDLE_QUEUE_MEMBER_SKIP;
        } else {
            memberSpecial = npu::tile_fwk::DRCO_SUCC_SPECIAL_MIDDLE_QUEUE_MEMBER;
        }
        middleQueue->UnsafeEnqueue((memberTask[mi] & ~(0x3Fu << specialShift)) | (memberSpecial << specialShift));
    }
    return true;
}

} // namespace

void DeviceTaskContext::InitDrcoRootFuncList(DynDeviceTask* dyntask)
{
    auto* rootFuncList = reinterpret_cast<npu::tile_fwk::DrcoRootFuncList*>(
        ControlFlowAllocateSlab(
            devProg_, sizeof(npu::tile_fwk::DrcoRootFuncList),
            workspace_->SlabAlloc(sizeof(npu::tile_fwk::DrcoRootFuncList), WsAicpuSlabMemType::DYN_FUNC_DATA))
            .ptr);
    dyntask->drcoRootFuncList = rootFuncList;
    // middleQueue 懒分配数组 + 区域注册表计数：rootFuncList 整体不清零（reinterpret_cast
    // 分配，默认成员初始化不执行），未赋值槽位残留 workspace 旧数据指针（设备侧仅判非空即
    // 解引用 → taskId 越界，AIV aicore 271，MxQuantize 实证）；mqRegionHeadCount 残留上一
    // devTask 的合法值（slab 复用）更凶——当前图注册追加在陈旧列表后，设备侧 TryOpen 撞
    // 陈旧头即假命中，非头任务被写 0xff + 陈旧叶计数入账 → executedCount 污染。分配点建立
    // nullptr/0 契约（谁分配谁初始化）。
    (void)memset_s(rootFuncList->perCoreMiddleQueueArray, sizeof(rootFuncList->perCoreMiddleQueueArray), 0,
                   sizeof(rootFuncList->perCoreMiddleQueueArray));
    rootFuncList->middleQueueRegionHeadCount = 0;
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
    // 完成计数表内嵌 rootFuncList（承接原全局队列的 staticExecCount/size），MIX 恒为 0
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
        if (IsSkipResolveSucc(dyntask, taskId)) {
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

bool DeviceTaskContext::TryHeadReleaseFastPath(DynDeviceTask* dyntask, DevAscendProgram* devProg)
{
    // 一解多直通：头的执行逻辑不变（初始就绪走正常分发、运行期 fire 走矩阵），仅当头的
    // 全部直接后继唯一前驱=头（满覆盖零哨兵）且数量达门槛时，把后继预放入各核
    // middleQueue 并注册头（完成点命中注册表 → 写 predCount=0xff 开门并跳过 resolve 展开，
    // 省掉单核串行 push 大扇出后继到矩阵的分发）；成员执行后正常 resolve（第二层起走
    // 老路径），hub 成员不可执行只 resolve。随 DRCO 总开关生效（调用点 enableAicoreResolve_ 块内）
    const uint32_t nrValidAic = devProg->devArgs.nrValidAic;
    const uint32_t queueCapacity = dyntask->devTask.coreFunctionCnt;
    auto* rootFuncList = dyntask->drcoRootFuncList;
    const uint64_t funcCount = dyntask->dynFuncDataCacheListSize;
    PredArrCache predCache{reinterpret_cast<uint64_t>(rootFuncList->metadataBase),
                           reinterpret_cast<uint64_t>(rootFuncList->controlFlowCacheBase)};
    uint32_t succBuf[MIDDLE_QUEUE_MAX_SUCC]; // 收集缓冲兼成员压实缓冲（写位恒 ≤ 读位）
    uint32_t memberCt[MIDDLE_QUEUE_MAX_SUCC];
    bool memberSkip[MIDDLE_QUEUE_MAX_SUCC];
    uint32_t regionApplied = 0;
    for (uint64_t f = 0; f < funcCount; f++) {
        DevAscendFunction* devFunc = dyntask->dynFuncDataCacheList[f].devFunc;
        const int32_t opSize = static_cast<int32_t>(devFunc->GetOperationSize());
        auto* duppedData = dyntask->dynFuncDataCacheList[f].duppedData;
        if (opSize <= 0 || duppedData == nullptr) {
            continue;
        }
        for (int32_t o = 0; o < opSize; o++) {
            // 槽满即止（regionApplied 只增不减，槽满后候选头已无槽可入）
            if (regionApplied >= npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM) {
                continue;
            }
            const uint32_t succCnt = CollectDirectSuccessors(devFunc, duppedData, static_cast<uint32_t>(f), o, succBuf);
            if (succCnt < MIDDLE_QUEUE_MIN_SUCC) {
                continue;
            }
            const uint32_t memberCnt = CollectEligibleMembers(dyntask, funcCount, predCache, succBuf, succCnt, succBuf,
                                                              memberCt);
            if (memberCnt == 0) {
                continue;
            }
            // 头注册（头调度不动；完成点命中 → 写 0xff 开门并跳过整个 resolve）
            if (rootFuncList->middleQueueRegionHeadCount >=
                npu::tile_fwk::DrcoRootFuncList::DRCO_MIDDLE_QUEUE_REGION_MAX) {
                continue;
            }
            const uint32_t headFlat = npu::tile_fwk::MakeTaskID(static_cast<uint32_t>(f), static_cast<uint32_t>(o));
            const uint32_t regionIdx = rootFuncList->middleQueueRegionHeadCount;
            rootFuncList->middleQueueRegionHeadList[rootFuncList->middleQueueRegionHeadCount++] = headFlat;
            for (uint32_t mi = 0; mi < memberCnt; mi++) {
                memberSkip[mi] = IsSkipResolveSucc(dyntask, succBuf[mi]);
            }
            uint32_t aicLeafCnt = 0;
            uint32_t aivLeafCnt = 0;
            for (uint32_t mi = 0; mi < memberCnt; mi++) {
                if (memberCt[mi] == static_cast<uint32_t>(CoreType::AIC)) {
                    aicLeafCnt++;
                } else if (memberCt[mi] == static_cast<uint32_t>(CoreType::AIV)) {
                    aivLeafCnt++;
                }
            }
            rootFuncList->middleQueueRegionLeafCountAic[regionIdx] = aicLeafCnt;
            rootFuncList->middleQueueRegionLeafCountAiv[regionIdx] = aivLeafCnt;

            // 撒核 + 入队。slot = regionApplied（候选序）：同 devTask 多头各自独立槽位/
            // 独立门——共享单槽会 headTaskId 互相覆盖，后入队头开门时放行未完成头的成员
            if (!SpreadMiddleQueueMembers(rootFuncList, workspace_, succBuf, memberCt, memberSkip, memberCnt, headFlat,
                                          regionApplied, nrValidAic, queueCapacity)) {
                rootFuncList->middleQueueRegionHeadCount--; // 容量不足回退本次注册
                continue;
            }
            regionApplied++;
        }
    }
    return regionApplied > 0;
}

} // namespace npu::tile_fwk::dynamic
