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
 * \file schedule_state.h
 * \brief Shared mutable state container for OoOScheduler, SpillEngine, DualDstEngine,
 *        LatencyEstimator, OptimizeSort, and MemoryAwareTopoSort.
 *        Each scheduler holds its own ScheduleState instance via composition.
 */

#ifndef PASS_SCHEDULE_STATE_H
#define PASS_SCHEDULE_STATE_H

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "passes/block_graph_pass/schedule_ooo/common/buffer_pool.h"
#include "passes/block_graph_pass/schedule_ooo/common/dep_manager.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_common_defs.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/statistics/schedule_observer.h"
#include "tilefwk/error_code.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOScheduleState"

namespace npu::tile_fwk {

constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;
constexpr int32_t MAX_SPILL_NO_PROGRESS_ROUNDS = 10;
constexpr uint64_t MAX_NUM_TOTAL_ISSUES_RATIO = 3;

const std::unordered_set<Opcode> USE_LESS_OPS = {Opcode::OP_NOP,       Opcode::OP_RESHAPE,  Opcode::OP_SHMEM_WAIT_UNTIL,
                                                 Opcode::OP_VIEW,      Opcode::OP_ASSEMBLE, Opcode::OP_BIND_TENSOR,
                                                 Opcode::OP_VIEW_TYPE, Opcode::OP_HUB,      Opcode::OP_SHMEM_STORE};

inline int BytesPerElement(DataType dataType) { return BytesOf(dataType); }

inline uint64_t CeilAlign(uint64_t a, int b) { return ((a + b - 1) / b) * b; }

// === Shared types used by OoOScheduler, SpillEngine, DualDstEngine ===

enum class CoreLocationType { AIC = 0, AIV0 = 1, AIV1 = 2, UNKNOWN = 3 };

const std::unordered_set<Opcode> COPY_IN_OPS = {
    Opcode::OP_COPY_IN,        Opcode::OP_UB_COPY_IN, Opcode::OP_L1_COPY_IN,  Opcode::OP_L1_COPY_IN_FRACTAL_Z,
    Opcode::OP_L1_COPY_IN_DMA, Opcode::OP_L1_COPY_UB, Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1,
    Opcode::OP_UB_COPY_L1_ND};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_ONE = {CoreLocationType::AIC,
                                                                             CoreLocationType::AIV0};

const std::unordered_set<CoreLocationType> CORE_INIT_CONFIGS_HARDWARE_TWO = {
    CoreLocationType::AIC, CoreLocationType::AIV0, CoreLocationType::AIV1};

struct OpSchedInfo {
    int execOrder{-1};
    PipeType pipeType{PipeType::PIPE_FIX};
    bool isAlloc{false};
    bool isRetired{false};
    bool isDualDstAlloc{false};
    Operation* pairedDualDstAlloc{nullptr};
    // 生产者在前: 会照这个次序写回调度序列, 反了则消费者排在生产者之前。
    std::vector<Operation*> skipOps;
    CoreLocationType coreLocation{CoreLocationType::UNKNOWN};
};

struct SpillContext {
    std::vector<Operation*> newCopyoutOps;
    std::vector<Operation*> newAllocOps;
    std::vector<int> spillMemIds;
    int newNotRetiredCopyOutSize{0};
    int deleteRetiredOpSize{0};
    int deleteNotRetiredOpSize{0};
    std::vector<std::tuple<Operation*, MemoryType, CoreLocationType>> deleteAllocOps;
    // 依赖被 spill 改写的非退休 op：可能已被旧前驱唤醒进 issue 队列，需摘出按新前驱重新入队。
    std::vector<Operation*> requeueOps;
};

struct SingleSpillCreatedOps {
    Operation* copyoutOp{nullptr};
    Operation* allocOp{nullptr};
    Operation* copyinOp{nullptr};
    LogicalTensorPtr gmTensor{nullptr};

    void Record(Operation* copyout = nullptr, Operation* alloc = nullptr, Operation* copyin = nullptr,
                LogicalTensorPtr gm = nullptr)
    {
        if (copyout)
            copyoutOp = copyout;
        if (alloc)
            allocOp = alloc;
        if (copyin)
            copyinOp = copyin;
        if (gm)
            gmTensor = gm;
    }
};

struct DualDstPair {
    Operation* opEarly{nullptr};
    Operation* opLate{nullptr};
    Operation* allocEarly{nullptr};
    Operation* allocLate{nullptr};
    LogicalTensorPtr tensorEarly;
    LogicalTensorPtr tensorLate;
};

struct DualDstAllocCtx {
    int memIdA{-1};
    int memIdB{-1};
    LocalBufferPtr bufA;
    LocalBufferPtr bufB;
    CoreLocationType coreA{CoreLocationType::UNKNOWN};
    CoreLocationType coreB{CoreLocationType::UNKNOWN};
};

struct OpQueue {
    bool busy{false};
    Operation* curOp = nullptr;
    int curOpRetireCycle{-1};
    std::vector<Operation*> queue;
    std::function<bool(Operation*, Operation*)> compareFunc;

    OpQueue() {}
    ~OpQueue() {}

    void SetCompareFunc(std::function<bool(Operation*, Operation*)> func) { compareFunc = func; }

    void Insert(Operation* op)
    {
        queue.push_back(op);
        if (compareFunc) {
            std::push_heap(queue.begin(), queue.end(), compareFunc);
        }
    }

    bool Empty() { return queue.empty(); }

    size_t Size() const { return queue.size(); }

    Operation* Front() { return queue[0]; }

    Operation* PopFront()
    {
        if (compareFunc) {
            std::pop_heap(queue.begin(), queue.end(), compareFunc);
            Operation* op = queue.back();
            queue.pop_back();
            return op;
        }
        Operation* op = queue.front();
        queue.erase(queue.begin());
        return op;
    }

    // 弹窗降级扫描：队首被闸门拒绝时不跳过整条 queue，
    // 向后找第一个通过 allow 判定的 op 弹出；被拒成员保持在场。
    Operation* PopAllowing(const std::function<bool(Operation*)>& allow)
    {
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (allow(*it)) {
                Operation* op = *it;
                queue.erase(it);
                if (compareFunc) {
                    std::make_heap(queue.begin(), queue.end(), compareFunc);
                }
                return op;
            }
        }
        return nullptr;
    }

    void DeleteOp(Operation* op)
    {
        auto it = std::find(queue.begin(), queue.end(), op);
        if (it != queue.end()) {
            queue.erase(it);
            if (compareFunc) {
                std::make_heap(queue.begin(), queue.end(), compareFunc);
            }
        }
    }
};

class ScheduleState {
public:
    ScheduleState() {}
    ~ScheduleState() {}

    std::vector<ScheduleObserver*> observers_;
    bool HasObservers() const { return !observers_.empty(); }

    // === Buffer and dependency state ===
    std::unordered_map<int, int> bufRefCount;
    std::unordered_map<MemoryType, int64_t> localMemSize;
    std::unordered_map<MemoryType, int64_t> localMemoryCurrentSize;
    std::unordered_map<int, LocalBufferPtr> localBufferMap;
    std::unordered_map<Operation*, LogicalTensors> inOutOperandsCache;
    std::unordered_map<Operation*, std::vector<int>> opReqMemIdsMap;
    std::vector<Operation*> operations;
    DependencyManager depManager;

    // === OoO scheduling metadata ===
    std::unordered_map<Operation*, OpSchedInfo> schedInfoMap;
    std::unordered_map<int, Operation*> tensorOccupyMap;
    std::unordered_map<int, Operation*> tensorAllocMap;
    std::unordered_map<CoreLocationType, std::map<MemoryType, BufferPool>> bufferManagerMap;
    std::vector<Operation*> newOperations;
    std::unordered_set<CoreLocationType> coreInitConfigs;
    // DualDst 总开关 (OoOScheduler 在 Schedule() 前设置)。各引擎共享。
    bool enableDualDst{false};
    std::unordered_map<Operation*, Operation*> dualDstPairs;   // AIV0 alloc -> AIV1 alloc
    std::unordered_map<Operation*, Operation*> dualDstOpPairs; // AIV0 op -> AIV1 op, non-alloc
    int64_t workspaceOffset{0};
    std::unordered_map<PipeType, int> pipeEndTime;
    int workspaceMemId{SYMBOL_STACK_BASE};

    // === Main loop scheduling state ===
    std::vector<Operation*> orderedOps;
    int clock{0};
    uint64_t numTotalIssues{0};
    std::unordered_map<CoreLocationType, std::map<MemoryType, OpQueue>> allocIssueQueue;
    // 当前 alloc stage 因 DualDst 分段而主动结束，需要保持 clock 继续下一轮，不触发 spill。
    bool continueAllocStage{false};

    // === VF scope tensor memId blacklist for spill ===
    std::unordered_map<int, std::unordered_set<int>> vfScopeTensorMemIds;
    // scopeId -> 按原 op 顺序排列的 vf op 列表(首 op 在前)。spill 锚定 copyin 用。
    std::unordered_map<int, std::vector<Operation*>> vfScopeOpsByScope;

    // === Spill dead-loop detection ===
    int spillNoProgressCnt{0};
    int lastPreSpillTotalQSize{-1};
    uint64_t originalNumTotalIssues{0};

    // === State operation methods ===

    std::vector<int>& GetOpMemIds(Operation* op);
    void SetOpMemIds(Operation* op, const std::vector<int>& memIds) { opReqMemIdsMap[op] = memIds; }
    void ClearOpMemIds(Operation* op) { opReqMemIdsMap[op].clear(); }
    void AddOpMemId(Operation* op, int memId) { opReqMemIdsMap[op].push_back(memId); }
    void ClearAllOpMemIds() { opReqMemIdsMap.clear(); }
    bool ReplaceOpMemId(Operation* op, int oldMemId, int newMemId);

    size_t CalcTensorMemoryPaddingSize(LogicalTensorPtr tensor);
    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId);
    std::string GetOpInfo(Operation* op) const;
    Status DelBufRefCount(const int memId);
    uint64_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype);
    const LogicalTensors& GetInOutOperandCached(Operation* op);
    void UpdateBufRefCount(Operation* op, LogicalTensorPtr tensor);
    Status InitBufRefCount(std::vector<Operation*>& list);
    bool IsOpAlloc(Operation* op);
    Status CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t>& bufferSize, std::set<int>& memIdMap);
    std::string DumpOpInfo(Operation& op);
    Status CheckOpBufferSize(Operation* op);
    Status Init(std::vector<Operation*>& opList);

    std::vector<Operation*>& GetSkipOps(Operation* op) { return schedInfoMap[op].skipOps; }
    void SetIsRetired(Operation* op, bool isRetired) { schedInfoMap[op].isRetired = isRetired; }
    void SetCoreLocation(Operation* op, CoreLocationType loc) { schedInfoMap[op].coreLocation = loc; }
    CoreLocationType GetCoreLocation(Operation* op) const;
    int GetExecOrder(Operation* op) const;
    bool IsOpAllocInSchedInfo(Operation* op) const;
    bool IsOpRetired(Operation* op) const;
    void InsertOrdered(Operation* insertOp);
    CoreLocationType ResolveCoreForFree(int memId);
    bool IsDualDstAlloc(Operation* allocOp);
};

CoreLocation ToCoreLocation(CoreLocationType c);

void NotifySpill(ScheduleState& state, LogicalTensorPtr spillTensor, int spillMemId, Operation* spillAllocOp,
                 const SingleSpillCreatedOps& created);

template <typename Scheduler>
Status RunSchedulerMainLoop(Scheduler& self)
{
    if (self.PreMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "PreMainLoop failed.");
        return FAILED;
    }
    uint64_t commitCnt = 0;
    bool isAllRetired = false;
    while (!isAllRetired) {
        int nextCycle = -1;
        APASS_LOG_DEBUG_F(Elements::Operation, "     clock: %d", self.state_.clock);
        if (self.RetireIssueStage(commitCnt, nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "RetireIssueStage failed.");
            return FAILED;
        }
        self.state_.continueAllocStage = false;
        if (self.BufferAllocStage(commitCnt) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "BufferAllocStage failed.");
            return FAILED;
        }
        if (self.LaunchIssueStage(nextCycle) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "LaunchIssueStage failed.");
            return FAILED;
        }
        if (self.state_.numTotalIssues == commitCnt && nextCycle == -1) {
            isAllRetired = true;
            break;
        }
        if (nextCycle == -1 && self.state_.continueAllocStage) {
            APASS_LOG_DEBUG_F(Elements::Operation, "AIV UB alloc stage completed; continue at current clock.");
            continue;
        }
        if (nextCycle == -1) {
            if (self.SpillOnBlock() != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "SpillOnBlock failed.");
                return FAILED;
            }
        } else {
            self.state_.clock = nextCycle;
        }
        if (self.state_.numTotalIssues > self.state_.originalNumTotalIssues * MAX_NUM_TOTAL_ISSUES_RATIO) {
            APASS_LOG_ERROR_F(Elements::Operation, "Spill dead-loop.");
            return FAILED;
        }
    }
    if (self.PostMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "PostMainLoop failed.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_STATE_H
