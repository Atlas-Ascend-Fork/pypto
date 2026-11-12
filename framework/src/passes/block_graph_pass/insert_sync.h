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
 * \file insert_sync.h
 * \brief
 */

#ifndef PASS_INSERT_SYNC_H
#define PASS_INSERT_SYNC_H

#include <queue>
#include <optional>
#include "interface/utils/common.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/opcode.h"
#include "passes/block_graph_pass/insert_sync_types.h"
#include "passes/pass_interface/pass.h"

namespace npu {
namespace tile_fwk {
constexpr uint64_t EVENTID_DEADLOCK_ENTER_TIME = 5;
constexpr uint64_t DEADLOCK_TIME_THRESHOLD = 2;
constexpr uint64_t MAX_POP = 8;
// 两个op之间插入的set_flag/wait_flag数量的最大值为192
// Reserve separate slots for the previous op's sets, a reclaim barrier, and this op's waits.
constexpr uint64_t SEQUENCE_IDX = 1024;
constexpr uint64_t HALF_SEQUENCE_IDX = 512;
constexpr uint64_t RECLAIM_PREFIX_OFFSET = 768;
constexpr uint64_t FORCE_SYNC_OP_NUM = 15;
constexpr uint64_t NUM15 = 15;
constexpr uint64_t NUM31 = 31;
struct Interval {
    int start;
    int end;
    int idx;
    Interval(int s, int e, int id) : start(s), end(e), idx(id) {}
};

struct IntervalTreeNode {
    Interval interval;
    int max;
    IntervalTreeNode* left;
    IntervalTreeNode* right;
    explicit IntervalTreeNode(Interval i) : interval(i), max(i.end), left(nullptr), right(nullptr) {}
};

class RangeSearchTree {
public:
    ~RangeSearchTree() { FreeTree(); }
    void Insert(int left, int right, int idx);
    std::set<int> GetCovered(int left, int right);

private:
    void FreeTree();
    Status ProcessTreeNode(const Interval& interval, IntervalTreeNode* currPtr,
                           std::vector<IntervalTreeNode*>& intervalStack);
    Status InsertInterval(const Interval& interval);
    void OverlapSearch(const Interval& interval, std::set<int>& result);
    IntervalTreeNode* treeRoot = nullptr;
};

class DataDependencySearcher {
public:
    std::set<int> Find(Operation* opWait);
    void Insert(const Operation* opSet, int idx);
    std::unordered_map<int, TileRange> ubTensorRangeMap;

private:
    void CheckRAWSearchTree(Operation* opWait, std::set<int>& res);
    void CheckWAWSearchTree(Operation* opWait, std::set<int>& res);
    void CheckWARSearchTree(Operation* opWait, std::set<int>& res);
    void InsertWAWSearchTree(const Operation* opSet, int idx);
    void InsertRAWSearchTree(const Operation* opSet, int idx);
    void InsertWARSearchTree(const Operation* opSet, int idx);

    std::unordered_map<MemoryType, RangeSearchTree> wawSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> warSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> rawSearchTree_;
    std::unordered_map<int, std::set<int>> readDdrMemMap;
    std::unordered_map<int, std::set<int>> writeDdrMemMap;
    std::unordered_map<const Operation*, int> opIndices_;
};

enum class PipeSeq {
    AIC_MTE2 = 0,
    AIC_MTE1,
    AIC_M,
    AIC_FIX,
    AIV0_MTE2,
    AIV1_MTE2,
    AIV0_V,
    AIV1_V,
    AIV0_MTE3,
    AIV1_MTE3,
    AIC_MTE3,
    AIV0_S,
    AIV1_S,
    AIC_S,
    PIPE_END
};

class PipeSync {
public:
    PipeSync() { InitIssueQueue(); }
    Status InsertSync(Function& function, std::vector<Operation*>& syncedOpLog);
    void PhaseKernelProcess(Function& function, const std::vector<Operation*>& srcLog, std::vector<Operation*>& dstLog);
    const std::vector<Operation*>& GetOriOpList() const { return oriOpList_; }
    std::unordered_map<Operation*, Operation*> setOpMap;
    std::unordered_map<Operation*, Operation*> waitOpMap;

private:
    friend class TuneTileOpSeqForVF;
    friend class TuneSyncForVF;
    friend class InsertSync;

    using PipeCoreReal = PipeSyncTypes::PipeCoreReal;
    using PipeCoreRealEx = PipeSyncTypes::PipeCoreRealEx;
    using PipeCoreRealExCompare = PipeSyncTypes::PipeCoreRealExCompare;
    using PipeCoreRealCompare = PipeSyncTypes::PipeCoreRealCompare;
    using PipeCore = PipeSyncTypes::PipeCore;
    using PipeCoreCompare = PipeSyncTypes::PipeCoreCompare;
    using PipePair = PipeSyncTypes::PipePair;
    using PipePairEx = PipeSyncTypes::PipePairEx;
    using CoreTypeDetail = PipeSyncTypes::CoreTypeDetail;
    using CorePair = PipeSyncTypes::CorePair;
    using PipePairHash = PipeSyncTypes::PipePairHash;
    using PipePairExHash = PipeSyncTypes::PipePairExHash;
    using CorePairHash = PipeSyncTypes::CorePairHash;
    using IndexVecHash = PipeSyncTypes::IndexVecHash;
    using DepOp = PipeSyncTypes::DepOp;
    using IssueQueue = PipeSyncTypes::IssueQueue;
    using PipeDepInfo = PipeSyncTypes::PipeDepInfo;
    using DataDepInfo = PipeSyncTypes::DataDepInfo;
    using IssueNum = PipeSyncTypes::IssueNum;
    using EventIdProcessContext = PipeSyncTypes::EventIdProcessContext;
    using EventResource = PipeSyncTypes::EventResource;
    using EventResourceHash = PipeSyncTypes::EventResourceHash;
    using PipeCoreRealExHash = PipeSyncTypes::PipeCoreRealExHash;
    using EventInstanceId = PipeSyncTypes::EventInstanceId;
    using ConsumedInstances = PipeSyncTypes::ConsumedInstances;
    using InstanceState = PipeSyncTypes::InstanceState;
    using EventInstance = PipeSyncTypes::EventInstance;
    using EventInstanceTracker = PipeSyncTypes::EventInstanceTracker;

    struct ReclaimBarrier {
        std::vector<Operation*> ops;
        std::vector<CoreTypeDetail> cores;
    };
    std::vector<ReclaimBarrier> reclaimBarriers_;
    size_t FindReclaimPrefix(size_t boundary, const std::unordered_set<Operation*>& emitted) const;
    Status StageReclaimedEvents(size_t boundary, const PipePairEx& pp, bool lite,
                                const std::unordered_set<Operation*>& emitted, EventInstanceTracker& stagedState,
                                std::vector<size_t>& covered) const;
    bool HasReclaimCapacity(size_t blockedIdx, size_t boundary, const PipePairEx& pp,
                            const EventInstanceTracker& stagedState, size_t covered);
    Status ReclaimIntraCoreEvents(Function& function, size_t boundary, const PipePairEx& pp,
                                  std::vector<IndexOp>& syncedOpLog);
    bool IsIntraCoreEvent(const PipePairEx& pp) const;
    EventResource GetEventResource(const PipePairEx& pp, int eventId) const;
    ConsumedInstances ProjectConsumedInstances(const DepOp& op, const PipeCoreRealEx& source) const;
    static constexpr int EVENT_ID7 = 7;
    static constexpr int EVENT_NUM = EVENT_ID7 + 1;
    static constexpr int INTRA_CORE_START_EVENT_ID = 1; // 核内 eventid 使用 1 ~ 7
    struct IntraEventIds {
        std::array<int, EVENT_NUM> ids{};
        size_t count{0};
        bool empty() const { return count == 0; }
        size_t size() const { return count; }
        int front() const { return ids.front(); }
    };
    IntraEventIds GetAvailableIntraEventIds(const PipePairEx& pp, const ConsumedInstances& consumed,
                                            size_t limit = EVENT_NUM);
    Status AllocateIntraEvent(const PipePairEx& pp, int& eventId, size_t setIdx, size_t waitIdx);
    Status RecordEventSet(const PipePairEx& pp, int eventId, size_t setIdx, size_t waitIdx, Operation& syncOp);
    Status RecordEventWait(size_t setIdx, size_t waitIdx, Operation& syncOp);
    struct IntraEventCapacityFailure {
        size_t setIdx;
        size_t waitIdx;
        PipePairEx pipes;
        size_t required;
        size_t available;
    };
    std::optional<IntraEventCapacityFailure> AnalyzeIntraEventCapacity(const DepOp& op);
    void LogIntraEventCapacityFailure(const IntraEventCapacityFailure& failure) const;
    struct EventDependencyGraph {
        std::vector<std::vector<size_t>> successors;
        std::vector<size_t> inDegree;
        std::vector<std::pair<size_t, bool>> nodes;   // Event-instance index and whether the node is a set.
        std::vector<std::array<size_t, 2>> positions; // Set and wait node indices for each instance.
    };
    Status BuildEventDependencyGraph(const std::vector<Operation*>& opList, EventDependencyGraph& graph) const;
    Status AddReclaimBarrierEdges(const std::vector<Operation*>& opList, EventDependencyGraph& graph) const;
    Status ValidateEventGraphAcyclic(const EventDependencyGraph& graph) const;
    Status ValidateEventReuseOrder(const EventDependencyGraph& graph) const;
    Status ValidateEventReuse(const std::vector<Operation*>& opList) const;
    bool BuildReplayIssueOrder(const std::vector<DepOp>& plan, std::vector<size_t>& order,
                               std::optional<std::pair<size_t, size_t>> extraDependency = std::nullopt) const;
    void CacheReplayOrder(const std::vector<size_t>& order);
    bool PreservesReplayOrder(size_t from, size_t to) const;
    bool CanAddReplayDependency(size_t from, size_t to);
    void RestoreOpDeps(const std::unordered_map<size_t, DepOp>& originalDeps);
    bool MergeDependenciesForReplay(std::vector<DepOp>& plan, const PipePairEx& pipes,
                                    std::vector<size_t>& order) const;
    bool AdvanceReturningWaitForReplay(std::vector<DepOp>& plan, const IntraEventCapacityFailure& failure,
                                       std::vector<size_t>& order) const;
    static bool RewriteReturningDependency(std::vector<DepOp>& candidate, size_t consumer,
                                           const IntraEventCapacityFailure& failure);
    bool AddReturningDependencyForReplay(std::vector<DepOp>& plan, const IntraEventCapacityFailure& failure,
                                         std::vector<size_t>& order) const;
    void ResetForReplay(const std::vector<DepOp>& plan, const std::vector<IssueQueue>& queues,
                        const std::vector<size_t>& order);

    std::string PipeSeqName(PipeSeq seq) const;
    PipeSeq GetPipeSeq(PipeCoreRealEx pipe);
    PipeCoreRealEx GetPipeFromSeq(PipeSeq seq);
    Status PipeDispatch(const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog);
    Status AdjustCopyInCfg(TileOpCfg& opcfg, const Operation& op);
    Status AdjustCopyOutCfg(TileOpCfg& opcfg, const Operation& op);
    Status AdjustOpCfg(TileOpCfg& opcfg, const Operation& op);
    void InitIssueQueue();
    void EnqueueOp(DepOp& op, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog);
    void RemoveOpDep(DepOp& setOp, DepOp& waitOp) const;
    void AddPhaseOp1(Function& function, const std::vector<Operation*>& srcLog, std::vector<Operation*>& dstLog,
                     size_t& i, size_t& prerun);
    void AddPhaseOp2(Function& function, std::vector<Operation*>& dstLog, size_t& prerun);
    Status AddOpDep(DepOp& setOp, DepOp& waitOp, bool isMergeCvSyncBase = false);
    Status AdjustOpDep(DepOp& op, size_t waitOpIdx, IssueQueue& issueQ, bool& failedFlag);
    Status HandleEventID(DepOp& op, IssueQueue& issueQ, IssueNum& issuenum, bool& deadlock, bool& res,
                         std::vector<IndexOp>& syncedOpLog);
    Status ProcessEventIdElement(EventIdProcessContext& ctx, std::optional<ConsumedInstances>& projected);
    Status ProcessSameCoreCase(const PipePairEx& pp, EventIdProcessContext& ctx,
                               std::optional<ConsumedInstances>& projected);
    Status PopFromQueue(IssueQueue& issueQ, std::vector<size_t>& poped, bool& deadlock,
                        std::vector<IndexOp>& syncedOpLog);
    Status InjectWaitFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog);
    Status InjectSetFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog);
    Status InjectSync(Function& function, const std::vector<Operation*>& opLogPtr, size_t idx,
                      std::vector<IndexOp>& syncedOpLog);
    Status IssueOpPipeSeq(Function& function, const std::vector<Operation*>& opLogPtr,
                          std::vector<IndexOp>& syncedOpLog, bool& eventIdDeadlock, size_t& issued);
    Status IssueSyncOp(Function& function, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog,
                       size_t& totalIssued, size_t& allIssued);
    Status IssueOp(Function& function, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog);
    Status ProcessDeadLock(uint64_t& eventIdDeadlockEnterTimes, bool& eventIdDeadlock,
                           std::vector<IndexOp>& syncedOpLog, bool* mergeSearchExhausted = nullptr);
    Status MergePendingDependencies(int set1, int wait1, int set2, int wait2, int eventId1, int eventId2);
    Status SynDependency(int maxOverlapDepIdx, const DataDepInfo& depInfo, const PipePairEx& pipePairEx,
                         std::vector<IndexOp>& syncedOpLog);
    Status GetDepInfo(std::vector<IndexOp>& syncedOpLog, const PipePairEx& pipePairEx, DataDepInfo& depInfo);
    Status RelaxFakeDataDep(std::vector<IndexOp>& syncedOpLog, bool& merged);
    bool CheckIssuedOp(const DepOp& op);
    bool FindMaxOverlap(DataDepInfo& depInfo, int& maxOverlapDepIdx);
    bool GenSyncOp(PipeCoreRealEx set, PipeCoreRealEx wait, int eventId, bool isSet, Operation& op);
    Status GetEventId(const PipePairEx& pp, int& eventId, size_t setIdx, size_t waitIdx,
                      std::vector<IndexOp>& syncedOpLog, Function& function);
    bool HasFreeEventId(const PipePairEx& pp);
    bool BufOverlap(const TileRange& range1, const TileRange& range2) const;
    bool CheckWawDependency(const Operation& opSet, const Operation& opWait);
    bool CheckRawDependency(const Operation& opSet, const Operation& opWait);
    bool CheckWarDependency(const Operation& opSet, const Operation& opWait);
    bool HasDataDependency(const Operation& opSet, const Operation& opWait);
    void UpdateDep(DepOp& currOp, DepOp& prevOp);
    bool IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation*>& opLogPtr);
    void FindDep(DepOp& op, const std::vector<Operation*>& opLogPtr, size_t idx,
                 DataDependencySearcher& dataDependencySearcher);
    void InitCVEventIdQ();
    static void PushEventIdIfAbsent(std::deque<int>& queue, int eventId);
    std::deque<int>* GetCrossCoreEventIdQPtr(const PipePairEx& pp);
    void RemoveEventIdFromCrossCoreQueues(int eventId);
    Status RemoveSetIntraBlockAndOpDep(std::vector<IndexOp>& syncedOpLog);
    void AddCrossCoreForceSyncOps(size_t setIdx, std::vector<IndexOp>& syncedOpLog, Function& function);
    static constexpr int CROSS_CORE_EVENT_NUM = 16;
    using CrossCoreClock = std::array<uint64_t, CROSS_CORE_EVENT_NUM * NUM2>;
    struct CrossCoreSnapshot {
        CrossCoreClock consumed{};
        int eventId{-1};
        uint64_t generation{0};
    };
    struct CrossCoreBarrierPlan {
        std::array<int, NUM2> handshakeIds{-1, -1};
        size_t localCount{0};
        bool globalBarrier{false};
    };
    bool UseCrossCoreProof() const;
    static void MergeCrossCoreProof(CrossCoreClock& clock, const CrossCoreSnapshot& snapshot);
    CrossCoreClock ProjectCrossCoreConsumption(size_t idx) const;
    bool CanReuseCrossCoreEventId(int eventId, const CrossCoreClock& clock) const;
    void RefreshCrossCoreEventIds(const PipeCoreRealEx& source, size_t pool, const CrossCoreClock& clock);
    Status PlanCrossCoreBarriers(size_t idx, const std::array<size_t, NUM2>& demand, CrossCoreBarrierPlan& plan);
    Status EnsureCrossCoreCapacity(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog);
    Status GetCrossCoreBarrierBoundary(size_t idx, size_t instructionCount, const std::vector<IndexOp>& syncedOpLog,
                                       uint64_t& insertIdx) const;
    Status AddGlobalCrossCoreBarrier(Function& function, size_t boundary, uint64_t insertIdx,
                                     std::vector<IndexOp>& syncedOpLog);
    Status AddLocalCrossCoreBarrier(Function& function, size_t boundary, size_t pool, int eventId, uint64_t& insertIdx,
                                    std::vector<IndexOp>& syncedOpLog);
    Status RecordCrossCoreDrain(size_t boundary, const std::vector<IndexOp>& syncedOpLog, ReclaimBarrier barrier);
    void ErasePendingCrossCoreSets(const std::unordered_set<Operation*>& removedSets);
    Status RemovePendingCrossCoreSyncs(size_t pool, std::vector<IndexOp>& syncedOpLog);
    Status ResetCrossCorePool(size_t pool, int handshakeId, std::vector<IndexOp>& syncedOpLog);
    void CreateForceSyncOp(Opcode opcode, PipeType pipe, CoreType core, AIVCore aivCore, uint64_t& insertIdx,
                           Function& function, std::vector<IndexOp>& syncedOpLog);
    void CreateBarAllOp(AIVCore aivCore, uint64_t& insertIdx, Function& function, std::vector<IndexOp>& syncedOpLog);
    // CV eventId 耗尽时插入的兜底 force 同步序列（BAR_ALL/FFTS/WAIT_FLAG_DEV）
    const std::unordered_set<Operation*>& GetForceSyncOps() const { return forceSyncOps_; }
    std::deque<int>& GetFreeEventIdQueue(const PipePairEx& pp);
    int GetMaxEventId(const PipePairEx& pp);
    std::string DumpLatestPipeDepMap();
    void BuildTensorRangeMap(Operation* op);
    bool IsMixCvCore(CoreType coreOne, CoreType coreTwo);
    // Kirin9030/KirinX90 (CVMix lite NPU): AIC (cube) 与 AIV (vector) 为同一个物理核，仅 PIPE_V 相对独立。
    // 对非 PIPE_V 的 AIV 流水归一化为 AIC，使其与 AIC 共用同一套 eventid，避免 AIC/AIV 各自独立分配 1~7 造成冲突。
    PipeCoreRealEx NormalizeToAICCoreForCVMixSync(PipeCoreRealEx p) const;

    std::vector<DepOp> depOps_;
    // A nonempty rank certifies that the current dependencies admit this order.
    std::vector<size_t> replayRank_;
    std::vector<bool> replayBoundary_;
    // Cube: MTE2, MTE1, M, FIX, Vector: MTE2, V, MTE3
    std::vector<IssueQueue> issueState_;
    std::unordered_map<PipePairEx, std::deque<int>, PipePairExHash> freeEventId_;
    EventInstanceTracker eventState_;

    // set pipe可用的eventid
    std::unordered_map<PipeCoreRealEx, std::array<std::deque<int>, NUM2>, PipeCoreRealExHash> crossCoreFreeEventId_;
    std::vector<std::pair<std::pair<size_t, size_t>, Operation*>> NoWaitCVPairs_;

    std::unordered_map<std::pair<size_t, size_t>, int, IndexVecHash> setWaitPairMap_;
    std::map<PipeCoreRealEx, PipeDepInfo, PipeCoreRealExCompare> latestPipeDep_;
    static std::map<PipeCoreRealEx, PipeSeq, PipeCoreRealExCompare> pipe2Seq;
    static std::map<PipeSeq, PipeCoreRealEx> seq2pipe;
    static std::vector<PipePair> dataDepPair;
    static std::vector<PipePair> mixCVDataDepPair;
    static std::vector<PipeCoreRealEx> cvPipeCoreEx;

    // Both directions share each physical slot. Proofs refer to a specific set generation.
    std::array<std::optional<CoreType>, CROSS_CORE_EVENT_NUM * NUM2> crossCoreEventSenders_{};
    CrossCoreClock crossCoreGenerations_{};
    std::unordered_map<PipeCoreRealEx, CrossCoreClock, PipeCoreRealExHash> crossCoreConsumed_;
    std::unordered_map<std::pair<size_t, size_t>, CrossCoreSnapshot, IndexVecHash> crossCoreSnapshots_;
    bool enableLocalCrossCoreBarrier_{true};
    std::unordered_map<PipePairEx, std::vector<int>, PipePairExHash> doublePipeOp; // pipepair, opmagic
    std::queue<size_t> orderedOpList_;
    std::vector<Operation*> oriOpList_;
    std::unordered_set<Operation*> forceSyncOps_;
    std::unordered_map<int, TileRange> ubTensorRangeMap;
    IRBuilder irBuilder_;
};

class InsertSync : public Pass {
public:
    InsertSync() : Pass("InsertSync") {}
    ~InsertSync() override {}
    void SetEnableDebug(bool enableDebug) { enableDebug_ = enableDebug; }
    void SetCvHardSync(bool cvHardSync) { cvHardSync_ = cvHardSync; }

private:
    Status RunOnFunction(Function& function) override;
    void InsertPipeAll(Function* subGraphFunc);
    void InsertCvPipeAll(Function* subGraphFunc);
    void InsertCvSyncOps(Function* subGraphFunc, std::vector<Operation*>& newOpList);
    Status GenNewOpList(Function* subGraphFunc, std::vector<Operation*>& opListNew);
    Status ApplyCvHardSync(Function* subGraphFunc, std::vector<Operation*>& opListNew);
    Status CheckNewOpListSeq(const std::vector<Operation*>& oriOpList, const std::vector<Operation*>& opListNew);
    Status InsertSyncMainLoop(Function* subGraphFunc);
    Status AdjustSyncByAtomicScope(std::vector<Operation*>& opList, const std::unordered_set<Operation*>& forceSyncOps);

    // --- atomic scope 后处理辅助方法 ---
    static bool IsSyncOpcode(Opcode opcode);
    static bool IsImmovableSync(Opcode opcode);
    // 配对签名：(eventId, setPipe, waitPipe) 编码为 uint64
    using SyncSignature = uint64_t;
    static SyncSignature GetSyncSignature(int eventId, PipeType setPipe, PipeType waitPipe);
    // 配对标识 = 硬件事件三元组 (eventId, setPipe, waitPipe)：GenSyncOp 对 OP_SYNC_SRC/OP_SYNC_DST
    // 均按 {set.pipe, wait.pipe} 写入 syncQueue_，codegen 的 set_flag/wait_flag 用相同参数序
    // (PrintSyncInSingleKernel 只换助记符)，故 set/wait 两侧取同一签名；核内 eventId
    // 按有序 pipe 对 (PipePairEx) 分池，并由 EventInstanceTracker 校验复用；反向同号为独立资源。
    // cluster 信息
    struct ClusterInfo {
        size_t firstIdx{SIZE_MAX};
        size_t lastIdx{0};
        AIVCore aivCore{AIVCore::UNSPECIFIED};
        // cluster 内出现 force 同步序列时置位：该 cluster 整体不再进行 op 顺序调整
        bool skipAdjust{false};
        std::vector<Operation*> waitSyncs;
        std::vector<Operation*> setSyncs;
        std::unordered_set<SyncSignature> setSignatures;
        std::unordered_set<int> cvSetEventIds;
    };
    Status BuildClusterRanges(const std::vector<Operation*>& opList, std::map<int, ClusterInfo>& clusters);
    Status ValidateAndCollectClusterSyncOps(const std::vector<Operation*>& opList, int scopeId, ClusterInfo& ci,
                                            const std::unordered_set<Operation*>& forceSyncOps);
    void ReorderOpListForClusters(std::vector<Operation*>& opList, const std::map<int, ClusterInfo>& clusters);

    bool enableDebug_{false};
    bool cvHardSync_{false};
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu

#endif // PASS_INSERT_SYNC_H
