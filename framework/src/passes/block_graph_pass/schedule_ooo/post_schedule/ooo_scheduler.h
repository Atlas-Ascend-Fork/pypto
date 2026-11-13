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
 * \file ooo_scheduler.h
 * \brief
 */

#ifndef PASS_SCHEDULER_H
#define PASS_SCHEDULER_H

#include <climits>
#include <unordered_set>

#include "interface/tensor/irbuilder.h"
#include "passes/block_graph_pass/schedule_ooo/common/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/common/dep_manager.h"
#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/dualdst_engine.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/spill_engine.h"
#include "passes/pass_check/schedule_ooo_checker.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/statistics/schedule_observer.h"

namespace npu::tile_fwk {

class OoOScheduler {
public:
    ScheduleState state_;

    template <typename Scheduler>
    friend Status RunSchedulerMainLoop(Scheduler& self);

public:
    Status Schedule(const std::vector<Operation*>& opList,
                    const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);
    OoOScheduler(Function& function)
        : function_(function), spillEngine_(state_, function_), dualDstEngine_(state_, function_)
    {}

    void AddObserver(ScheduleObserver* observer) { state_.observers_.push_back(observer); }
    std::vector<Operation*> GetNewOperations();

    // DualDst switch is configured by OoOSchedule before Schedule().
    void SetEnableDualDst(bool v) { state_.enableDualDst = v; }
    void SetDualDstPairs(std::unordered_map<Operation*, Operation*> pairs) { state_.dualDstPairs = std::move(pairs); }
    void SetDualDstOpPairs(std::unordered_map<Operation*, Operation*> pairs)
    {
        state_.dualDstOpPairs = std::move(pairs);
    }

private:
    struct DualSpillGroup {
        std::vector<int> aiv0MemIds;
        std::vector<int> aiv1MemIds;
    };
    Function& function_;
    SpillEngine spillEngine_;
    DualDstEngine dualDstEngine_;

    std::unordered_map<CoreLocationType, std::map<PipeType, OpQueue>> issueQueues;

    uint64_t spillIssueCnt{0};
    std::vector<ScheduleObserver*>& observers_ = state_.observers_;

    IRBuilder irBuilder_;
    std::unordered_map<int, DDRBufferKind> ddrKindMap_;
    // 输出在 DDR 上的 skip op: 没有消费者的 skipOps 接管它, 由 PlaceDdrSkipOps 定稿时紧跟生产者插入。
    // 按 opList 原序存放, 祖先在前, 链上后继才能看到已落位的前驱。
    std::vector<Operation*> ddrSkipOps_;

    // 每 core 闸门持有的活动 vf scope id（-1 表示闸门打开）
    std::unordered_map<CoreLocationType, int> gateActiveScope_;
    // scopeId -> 已发射（launch）的 scope op 计数，闸门关闭期间随同 scope op 发射递增
    std::unordered_map<int, int> scopeLaunchedCnt_;
    // scopeId -> launch 计数集合大小：scope 内走 PIPE_V 发射路径的非 skip/ALLOC op 数
    std::unordered_map<int, int> scopeLaunchTotal_;
    std::unordered_map<int, std::vector<std::pair<uint64_t, uint64_t>>> scopeTensorRanges_;
    // 本次 alloc 下发后是否激活了后继 op（就绪进 launch 队列），用于 vf alloc 的"激活即停"节流
    bool allocActivatedOp_{false};

    void NotifyOpLaunch(Operation* op, int cycleEnd);
    void NotifyOpRetire(Operation* op, const std::vector<int>& freedMemIds);
    void NotifyAllocExec(Operation* op, int memId);
    void NotifyBufferRearrange(Operation* triggerOp, MemoryType memType,
                               std::vector<BufferRearrangeEvent::Change> changes);
    void NotifyAllocFail(Operation* triggerOp, MemoryType memType, uint64_t requestSize);
    void NotifyScheduleEnd(bool success);
    void NotifyInitDDRBuffers();
    void NotifyMainLoopBegin();
    void NotifyMainLoopEnd();
    void EmitInitDDRBuffer(const LogicalTensorPtr& t, DDRBufferKind kind);

    static CoreLocation ToCoreLocation(CoreLocationType c);

    std::vector<DDRRef> BuildDDRRefs(Operation* op) const;

    // scheduler
    Status Init(const std::vector<Operation*>& opList,
                const std::unordered_set<CoreLocationType> fixCoreConfig = CORE_INIT_CONFIGS_HARDWARE_ONE);

    Status InitOpEntry(Operation* op);
    Status InitOpCoreType(Operation* op);
    void InitSkipOps(Operation* op);
    void PlaceDdrSkipOps();

    void InitCoreConfig(const std::vector<Operation*>& opList);
    void InitTensorCoreMap();
    void InitIssueQueuesAndBufferManager();
    void InitVfScopeGate();
    bool IsScopeInputsReady(int scopeId);
    bool GateAllowsIssue(CoreLocationType coreLocation, PipeType pipeType, Operation* op);
    bool ShouldDelayScopeGate(CoreLocationType coreLocation, Operation* scopeFirstOp);
    void UpdateVfScopeGate(CoreLocationType coreLocation, PipeType pipeType, Operation* op);
    void RecordScopeTensorRange(int scopeId, int memId);
    void ClearScopeTensorRanges(int scopeId);
    Status InsertSpillCopyoutOp(Operation* copyoutOp);
    size_t InsertSkipOpsBeforeCopyout(Operation* copyoutOp, size_t copyoutPos);
    Status QueryAllocPlacement(Operation* op, bool& hasMatchedOffset, uint64_t& matchedOffset,
                               std::vector<std::pair<uint64_t, uint64_t>>& avoidRanges);

    Status AllocWorkspaceGM(const std::vector<Operation*>& opList);
    void UpdateDualDstL0MXRange();
    Status FinalizeScheduleResult(const std::vector<Operation*>& opList);
    Status SeqSchedule();
    Status ProcessSeqOp(Operation* op, size_t& pcIdx);
    Status ExecuteAllocIssue(Operation* op, size_t& pcIdx);
    Status AllocateAfterSpill(Operation* op, LocalBufferPtr allocBuffer, CoreLocationType coreLocation, bool isDualDst);
    Status RetireIssue(Operation* op);
    Status ScheduleMainLoop();
    void LaunchReadyIssue();
    Status RetireCoreIssue(CoreLocationType targetCore, uint64_t& commitCnt, int& nextCycle);
    Status PreMainLoop();
    Status PostMainLoop();
    Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle);
    Status RetireOpAndAwakeSucc(Operation* op, uint64_t& commitCnt);
    Status FreeBuffer(Operation* op, std::vector<int>& freedMemIds);
    Status BufferAllocStage(uint64_t& commitCnt);
    Status ExecuteAivUbAllocRound(uint64_t& commitCnt);
    Status ExecuteAivRegularAllocStage(uint64_t& commitCnt, OpQueue& queue0, OpQueue& queue1);
    Status CheckAivUbPoolSlicesEqual();
    Status ExecuteAllocIssue(uint64_t& commitCnt, MemoryType memType, OpQueue& pipe);
    Status TryDualDstAllocPair(Operation* aiv0Alloc, Operation* aiv1Alloc, uint64_t& commitCnt, bool& allocated);
    Status TryRegularAllocOnce(Operation* op, MemoryType memType, CoreLocationType coreLocation,
                               const std::vector<int>& reqMemIds, uint64_t& commitCnt, bool& allocated);
    void HandleSkipOp(Operation* op);
    Status LaunchIssueStage(int& nextCycle);
    Status AllocTensorMemRange(Operation* op);
    Status AllocSkipTensorMemRange(Operation& operation);
    Status CheckAndUpdateLifecycle();
    void UpdateIssueExecOrder();
    void PrintOpList(std::vector<Operation*> opList);
    bool HasEnoughBuffer(Operation* allocOp, MemoryType memType);
    Status RearrangeBuffer(Operation* allocOp, MemoryType memType);
    Status GenBufferSpill(Operation* allocOp, SpillContext& ctx, bool isMainLoop);
    Status SelectDualDstSpillTargets(Operation* allocOp, Operation* peerAlloc, DualSpillGroup& dualGroup);
    Status ApplyDualSpill(Operation* allocOp, Operation* peerAlloc, const DualSpillGroup& dualGroup, SpillContext& ctx,
                          SpillContext& peerCtx, MemoryType reqMemType, uint64_t reqSize);
    Status ApplySingleSpill(Operation* allocOp, const std::vector<int>& spillGroup, SpillContext& ctx,
                            MemoryType reqMemType, uint64_t reqSize);
    Status FinalizeSingleSideSpill(Operation* allocOp, MemoryType reqMemType, uint64_t reqSize);
    std::vector<int> SelectSpillBuffers(Operation* allocOp);
    Status ApplySpillContext(SpillContext& ctx, Operation* allocOp);
    Status PrintSpillFailedInfo(Operation* allocOp);
    std::vector<std::vector<int>> GetSpillGroup(BufferPool& pool, size_t sizeNeedSpill,
                                                const std::vector<std::pair<uint64_t, uint64_t>>& avoidRanges = {});
    std::vector<DualSpillGroup> GetDualSpillGroup(BufferPool& poolA, BufferPool& poolB, size_t sizeNeedSpill);
    Status GetGroupNextUseTime(std::vector<int> group, Operation* allocOp, std::vector<int>& groupNextUseTime,
                               std::unordered_map<int, size_t>& nextUseTimeCache);

    Status SpillOnBlock();
    Status DetectSpillDeadLoop();
    Status SpillDualDstAivUbBlock();
    Status SpillOnCoreBlock(std::pair<CoreLocationType, MemoryType> coreLocation);
    Status FindFirstOrder(std::pair<CoreLocationType, MemoryType>& orderFirstPair);
    Status FindCoreLocationMemoryType(CoreLocationType coreLocation, MemoryType& spillMemType);

    std::vector<Operation*>& GetSkipOps(Operation* op) { return state_.schedInfoMap[op].skipOps; }
    void SetIsRetired(Operation* op, bool isRetired) { state_.schedInfoMap[op].isRetired = isRetired; }
    void SetCoreLocation(Operation* op, CoreLocationType loc) { state_.schedInfoMap[op].coreLocation = loc; }

    void UpdateL0MXMap(const std::vector<Operation*>& opList);
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULER_H
