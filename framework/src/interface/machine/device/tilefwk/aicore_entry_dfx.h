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
 * \file aicore_entry_dfx.h
 * \brief DFX/perf helpers for aicore_entry.h (include only after ExecuteContext is defined).
 */

#pragma once

enum DFX_STAGE_STATUS {
    STAGE_HANDSHAKE_START = 1,
    STAGE_HANDSHAKE_END = 2,
    STAGE_CORE_EXIT = 3,
    STAGE_GET_NEXT_TASK_STOP = 4,
    STAGE_PRE_EXEC_COREFUNC_KERNEL = 5,
    STAGE_FINISH_EXEC_COREFUNC_KERNEL = 6,
    STAGE_FINISH_PIPE_SYNC = 7,
    STAGE_FINISH_CUR_TASK = 8,
    STAGE_GET_PARALLEL_DEVTASK_TIMEOUT = 9,
    STAGE_GET_NEXT_TASK_TIMEOUT = 10,
    STAGE_WAVE_TIMEOUT = 11,
    STAGE_GET_HIGH_REG_TIMEOUT = 12,
    STAGE_UPDATE_PARALLEL_DEVTASK_TIMEOUT = 13,
    STAGE_UPDATE_PARALLEL_DEVTASK_ID_TIMEOUT = 14,
    STAGE_RUN_LEAFTASK_TIMEOUT = 15,
    STAGE_GET_FUNCDATA_STOP = 16
};

struct FedProfDevTaskProfStatus {
    bool profEnabled{false};
    __gm__ FedProfDevTaskProf* devTaskProf{nullptr};
};

struct ExecuteContext {
    __gm__ KernelArgs* args;
    int32_t blockIdx;
    __gm__ FedProfPerCoreProf* perCoreProf{nullptr};
    volatile __gm__ ParallelDevTask* parallelDevTask{nullptr};
    uint32_t curLeafTaskParallelIdx{0};
    uint32_t seqNo{0};
    uint32_t profLevel{0};
    uint32_t aicCoreNum{0};
    uint32_t drcoGroupBeg[npu::tile_fwk::DRCO_QUEUE_MAX]{};
    uint32_t drcoGroupEnd[npu::tile_fwk::DRCO_QUEUE_MAX]{};
    struct CachedDevTask {
        uint32_t seqNo{0};
        __gm__ DynFuncHeader* header{nullptr};
        __gm__ DynFuncData* funcDataList{nullptr};
        __gm__ DrcoRootFuncData* drcoRootFuncDataList{nullptr};
        __gm__ DynFuncBin* cceBinary{nullptr};
    } cachedDevTasks[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM];
    CachedDevTask* cachedDevTaskCurr;
    uint64_t lastTaskFinishCycle{0};
    FedProfDevTaskProfStatus devTaskProfStatus;
#if ENABLE_AICORE_PRINT
    AicoreLogger logger;
#endif
    uint32_t SeqNo() { return cachedDevTasks[curLeafTaskParallelIdx].seqNo; }
};

INLINE void Barrier()
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    __asm__ __volatile__("" ::: "memory");
#else
    __asm__ __volatile__("");
#endif
}

INLINE void SetStatus(__gm__ KernelArgs* args, int64_t val)
{
    if (!IS_AICORE || DEBUG_SWITCH) {
        Barrier();
        args->dfxBuffer[2] = val;
        dcci(args->dfxBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
}

INLINE void SetLastWordStatus(__gm__ KernelArgs* args, int64_t val)
{
    Barrier();
    args->dfxBuffer[3] = val;
    dcci(args->dfxBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void SetWarningStatus(__gm__ KernelArgs* args, int64_t val)
{
    Barrier();
    args->dfxBuffer[6] = val;
    dcci(args->dfxBuffer, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

INLINE void FedProfDevTaskProfRecord(uint32_t devTaskId, __gm__ FedProfDevTaskProf* devTaskProf,
                                     FedProfPerfTraceKind type, uint64_t cycle = 0)
{
    if (devTaskProf != nullptr) {
        uint64_t cnt = devTaskProf->cnt;
        if (cnt < PERF_TRACE_CORE_KIND_MAX * PERF_TRACE_CORE_PER_KIND_MAX_COUNT) {
            __gm__ FedProfDevTaskCoreEvent* perf = &(devTaskProf->aicoreEveryDevTypeTimeStamp[cnt]);
            perf->type = static_cast<uint64_t>(type);
            perf->timestamp = (cycle == 0) ? get_sys_cnt() : cycle;
            perf->devTaskIdx = static_cast<uint64_t>(devTaskId);

            devTaskProf->cnt = cnt + 1;
        }
    }
}

INLINE void FedProfPerCoreProfFlush(__gm__ volatile FedProfPerCoreProf* perCoreProf)
{
    if (perCoreProf == nullptr) {
        return;
    }

    for (uint32_t i = 0; i < perCoreProf->status.leafTaskCount; i++) {
        dcci(&perCoreProf->leafTaskProfList[i], SINGLE_CACHE_LINE, CACHELINE_OUT);
    }

    perCoreProf->status.isProfFinish = 1;
    dcci(perCoreProf, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}

INLINE void FedProfLeafTaskProfRecord(ExecuteContext* ctx, uint32_t seqNo, uint32_t taskId, int32_t subGraphId,
                                      int64_t t1)
{
    auto perCoreProf = ctx->perCoreProf;
    if (perCoreProf && perCoreProf->status.leafTaskCount < FED_PROF_LEAF_TASK_MAX_COUNT) {
        perCoreProf->leafTaskProfList[perCoreProf->status.leafTaskCount].leafFuncIdx = subGraphId;
        perCoreProf->leafTaskProfList[perCoreProf->status.leafTaskCount].devTaskIdx = seqNo;
        perCoreProf->leafTaskProfList[perCoreProf->status.leafTaskCount].taskId = taskId & TASKID_FROM_CTRL_TOPO_MASK;
        perCoreProf->leafTaskProfList[perCoreProf->status.leafTaskCount].timestampStart = t1;
        ctx->lastTaskFinishCycle = get_sys_cnt();
        perCoreProf->leafTaskProfList[perCoreProf->status.leafTaskCount].timestampEnd = ctx->lastTaskFinishCycle;
        perCoreProf->status.leafTaskCount++;
    }
}

INLINE void FedProfLeafTaskProfRecordMix(ExecuteContext* ctx, uint32_t taskId, int32_t subGraphId)
{
    if (unlikely(ctx->profLevel == PRO_LEVEL2 || ctx->profLevel == PRO_LEVEL1)) {
#ifdef __DAV_V310
        // set mix event info for new taskstat
        auto perCoreProf = ctx->perCoreProf;
        if (perCoreProf && perCoreProf->status.leafTaskCount < FED_PROF_LEAF_TASK_MAX_COUNT) {
            __gm__ FedProfLeafTaskProf* taskStat = &perCoreProf->leafTaskProfList[perCoreProf->status.leafTaskCount];
            __gm__ FedProfLeafTaskProf* prevTask = &perCoreProf->leafTaskProfList[perCoreProf->status.leafTaskCount -
                                                                                  1]; // hub task wont be first taskStat
            taskStat->perfDataBaseAddr = reinterpret_cast<uint64_t>(perCoreProf);
            taskStat->setEventAddr = prevTask->setEventAddr + prevTask->setEventNum * sizeof(uint64_t);
            taskStat->waitEventAddr = prevTask->waitEventAddr + prevTask->waitEventNum * sizeof(uint64_t);
            taskStat->setEventNum = 0;
            taskStat->waitEventNum = 0;
        }
#endif
        int64_t t1 = get_sys_cnt();
        FedProfLeafTaskProfRecord(ctx, ctx->SeqNo(), taskId, subGraphId, t1);
    }
}

INLINE void DfxProcWhenCoreExit(ExecuteContext* ctx, __gm__ KernelArgs* args, __gm__ FedProfPerCoreProf* metric)
{
    FedProfDevTaskProfRecord(INVALID_DEV_TASK_ID, ctx->devTaskProfStatus.devTaskProf, PERF_TRACE_CORE_WAIT_EXIT_NOTIFY);
    if (unlikely(args->taskEntry.reserved[0] == PRO_LEVEL2 || args->taskEntry.reserved[0] == PRO_LEVEL1 ||
                 ctx->devTaskProfStatus.profEnabled)) {
        metric->status.currLaunchRound++;
        FedProfPerCoreProfFlush(metric);
    }
}

INLINE void DfxProcWhenDevTaskStop(ExecuteContext* ctx, __gm__ KernelArgs* args, __gm__ FedProfPerCoreProf* metric,
                                   bool isExectedTask = false)
{
    if (isExectedTask) {
        FedProfDevTaskProfRecord(ctx->SeqNo(), ctx->devTaskProfStatus.devTaskProf,
                                 PERF_TRACE_CORE_DEV_TASK_LEAF_TASK_EXEC, ctx->lastTaskFinishCycle);
    }

    (void)metric;
    (void)args;
}
