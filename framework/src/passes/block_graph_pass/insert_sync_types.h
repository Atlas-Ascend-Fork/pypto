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
 * \file insert_sync_types.h
 * \brief Internal scheduling and event tracking types for InsertSync.
 */

#ifndef PASS_INSERT_SYNC_TYPES_H
#define PASS_INSERT_SYNC_TYPES_H

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "interface/operation/operation.h"
#include "interface/utils/common.h"

namespace npu {
namespace tile_fwk {
constexpr uint64_t LEFT_OFFSET_32 = 32;
constexpr uint64_t LEFT_OFFSET_16 = 16;
constexpr uint64_t LEFT_OFFSET_8 = 8;
constexpr uint64_t LEFT_OFFSET_24 = 24;

using IndexOp = std::pair<uint64_t, std::reference_wrapper<Operation>>;

struct PipeSyncTypes {
    struct PipeCoreReal {
        PipeCoreReal(PipeType p, CoreType c) : pipe(p), core(c) {}
        PipeType pipe;
        CoreType core;

        bool operator==(const PipeCoreReal& t) const { return (this->pipe == t.pipe && this->core == t.core); }

        bool operator!=(const PipeCoreReal& t) const { return !(*this == t); }
    };

    // 包含AIVCore类型的PipeCoreReal
    struct PipeCoreRealEx {
        PipeCoreRealEx(PipeType p, CoreType c, AIVCore a = AIVCore::UNSPECIFIED) : pipe(p), core(c), aivCore(a) {}
        PipeCoreRealEx(PipeCoreReal p, AIVCore a = AIVCore::UNSPECIFIED) : pipe(p.pipe), core(p.core), aivCore(a) {}
        PipeType pipe;
        CoreType core;
        AIVCore aivCore{AIVCore::UNSPECIFIED};

        bool operator==(const PipeCoreRealEx& t) const
        {
            return (this->pipe == t.pipe && this->core == t.core && this->aivCore == t.aivCore);
        }

        bool operator!=(const PipeCoreRealEx& t) const { return !(*this == t); }
    };

    struct PipeCoreRealExCompare {
        bool operator()(const PipeCoreRealEx& lhs, const PipeCoreRealEx& rhs) const
        {
            if (lhs.core != rhs.core) {
                return lhs.core < rhs.core;
            }
            if (lhs.pipe != rhs.pipe) {
                return lhs.pipe < rhs.pipe;
            }
            return lhs.aivCore < rhs.aivCore;
        }
    };

    struct PipeCoreRealCompare {
        bool operator()(const PipeCoreReal& lhs, const PipeCoreReal& rhs) const
        {
            return ((static_cast<uint64_t>(lhs.core) << LEFT_OFFSET_16) | (lhs.pipe << LEFT_OFFSET_8)) <
                   ((static_cast<uint64_t>(rhs.core) << LEFT_OFFSET_16) | (rhs.pipe << LEFT_OFFSET_8));
        }
    };

    struct PipeCore {
        PipeCore(PipeType ps, PipeType pe, CoreType c, AIVCore a) : pipeStart(ps), pipeEnd(pe), core(c), aivCore(a) {}
        PipeType pipeStart;
        PipeType pipeEnd;
        CoreType core;
        AIVCore aivCore;

        bool operator==(const PipeCore& t) const
        {
            return (this->pipeStart == t.pipeStart && this->pipeEnd == t.pipeEnd && this->core == t.core &&
                    this->aivCore == t.aivCore);
        }

        bool operator!=(const PipeCore& t) const { return !(*this == t); }
    };

    struct PipeCoreCompare {
        bool operator()(const PipeCore& lhs, const PipeCore& rhs) const
        {
            return ((static_cast<uint64_t>(lhs.core) << LEFT_OFFSET_32) | (lhs.pipeStart << LEFT_OFFSET_24) |
                    (lhs.pipeEnd << LEFT_OFFSET_16) | (static_cast<uint64_t>(lhs.aivCore) << LEFT_OFFSET_8)) <
                   ((static_cast<uint64_t>(rhs.core) << LEFT_OFFSET_32) | (rhs.pipeStart << LEFT_OFFSET_24) |
                    (rhs.pipeEnd << LEFT_OFFSET_16) | (static_cast<uint64_t>(rhs.aivCore) << LEFT_OFFSET_8));
        }
    };

    using PipePair = std::pair<PipeCoreReal, PipeCoreReal>; // setPipe, waitPipe
    using PipePairEx = std::pair<PipeCoreRealEx, PipeCoreRealEx>;
    using CoreTypeDetail = std::pair<CoreType, AIVCore>;
    using CorePair = std::pair<CoreTypeDetail, CoreTypeDetail>;

    struct PipePairHash {
        std::size_t operator()(const PipePair& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first.pipe);
            HashCombine(res, pp.first.core);
            HashCombine(res, pp.second.pipe);
            HashCombine(res, pp.second.core);
            return res;
        };
    };

    struct PipePairExHash {
        std::size_t operator()(const PipePairEx& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first.pipe);
            HashCombine(res, pp.first.core);
            HashCombine(res, pp.first.aivCore);
            HashCombine(res, pp.second.pipe);
            HashCombine(res, pp.second.core);
            HashCombine(res, pp.second.aivCore);
            return res;
        };
    };

    struct CorePairHash {
        std::size_t operator()(const CorePair& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first.first);
            HashCombine(res, pp.first.second);
            HashCombine(res, pp.second.first);
            HashCombine(res, pp.second.second);
            return res;
        };
    };

    struct IndexVecHash {
        std::size_t operator()(const std::pair<size_t, size_t>& pp) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, pp.first);
            HashCombine(res, pp.second);
            return res;
        };
    };

    struct DepOp {
        DepOp(size_t i, PipeCore pipeCore) : idx(i), selfPipeCore(pipeCore) {}
        size_t idx = SIZE_MAX;       // idx in oplog
        size_t idxInPipe = SIZE_MAX; // idx in the pipe belonging to
        PipeCore selfPipeCore;
        bool issued{false};

        std::vector<size_t> setPipe;  // this op will set_flag for op in setPipe; 后
        std::vector<size_t> waitPipe; // this op will wait_flag for op in waitPipe; 前
        std::string DumpDepOp(const std::vector<Operation*>& opLog);
    };

    struct IssueQueue {
        explicit IssueQueue(PipeCoreRealEx pipe) : selfPipeCore(pipe) {}
        PipeCoreRealEx selfPipeCore;
        size_t currOp{0};
        std::vector<size_t> ops;
        std::string DumpIssueQueue(const std::vector<Operation*>& opLogPtr);
    };

    struct PipeDepInfo {
        size_t waitIdx;
        std::map<PipeCoreRealEx, size_t, PipeCoreRealExCompare> setPipes;
        std::string DumpPipeDepInfo();
    };

    struct DataDepInfo {
        PipeType setp;
        CoreType setc;
        PipeType waitp;
        CoreType waitc;
        AIVCore setaivc;
        AIVCore waitaivc;
        std::vector<int> setOpIdList{};      // 对应sync_src/cv_sync_src在syncedOpLog中的idx
        std::vector<int> setOpEventIdList{}; // eventid
        // sync_src/cv_sync_src对应的setop和waitop的idx pair {setop idx, waitop idx}
        std::vector<std::pair<int, int>> opDepList{};
        std::string DumpDataDepInfo(const std::vector<IndexOp>& syncedOpLog, const std::vector<Operation*>& oriOpList);
    };

    struct IssueNum {
        // already issued op this round
        std::unordered_map<PipePairEx, size_t, PipePairExHash> currIssueNum;
        // max cv op can be issued this round
        std::unordered_map<CorePair, size_t, CorePairHash> maxCvIssueNum;
        // already issued cv op this round
        std::unordered_map<CorePair, size_t, CorePairHash> currCvIssueNum;
    };

    // HandleEventID 辅助结构：封装处理上下文
    struct EventIdProcessContext {
        DepOp& op;
        size_t eleIdx;
        IssueQueue& issueQ;
        IssueNum& issuenum;
        std::vector<IndexOp>& syncedOpLog;
        bool deadlock;
        bool eventIdOk{true};
        bool failedFlag{false};
    };

    struct EventResource {
        int eventId;
        CoreTypeDetail srcCore;
        CoreTypeDetail dstCore;
        PipeType srcPipe;
        PipeType dstPipe;

        bool operator==(const EventResource& other) const
        {
            return eventId == other.eventId && srcCore == other.srcCore && dstCore == other.dstCore &&
                   srcPipe == other.srcPipe && dstPipe == other.dstPipe;
        }
    };

    struct EventResourceHash {
        std::size_t operator()(const EventResource& er) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, er.eventId);
            HashCombine(res, er.srcCore.first);
            HashCombine(res, er.srcCore.second);
            HashCombine(res, er.dstCore.first);
            HashCombine(res, er.dstCore.second);
            HashCombine(res, er.srcPipe);
            HashCombine(res, er.dstPipe);
            return res;
        }
    };

    struct PipeCoreRealExHash {
        std::size_t operator()(const PipeCoreRealEx& p) const noexcept
        {
            std::size_t res = 0;
            HashCombine(res, p.pipe);
            HashCombine(res, p.core);
            HashCombine(res, p.aivCore);
            return res;
        }
    };

    // EventResource already identifies a directed, physical intra-core slot.
    // A consumed generation must never authorize reuse of a newer instance.
    struct EventInstanceId {
        EventResource resource;
        uint64_t generation;
        bool operator==(const EventInstanceId& other) const
        {
            return resource == other.resource && generation == other.generation;
        }
    };
    struct ConsumptionProof {
        size_t slot;
        uint64_t generation;
        bool operator==(const ConsumptionProof& other) const
        {
            return slot == other.slot && generation == other.generation;
        }
    };
    class ConsumedInstances {
    public:
        auto begin() const { return proofs_.begin(); }
        auto end() const { return proofs_.end(); }
        size_t size() const { return proofs_.size(); }
        bool empty() const { return proofs_.empty(); }
        void reserve(size_t count) { proofs_.reserve(count); }
        size_t count(const ConsumptionProof& proof) const
        {
            const auto it = std::lower_bound(proofs_.begin(), proofs_.end(), proof, Less);
            return it != proofs_.end() && *it == proof;
        }
        void insert(const ConsumptionProof& proof)
        {
            if (proofs_.empty() || Less(proofs_.back(), proof)) {
                proofs_.push_back(proof);
                return;
            }
            const auto it = std::lower_bound(proofs_.begin(), proofs_.end(), proof, Less);
            if (!(*it == proof)) {
                proofs_.insert(it, proof);
            }
        }
        void erase(const ConsumptionProof& proof)
        {
            const auto it = std::lower_bound(proofs_.begin(), proofs_.end(), proof, Less);
            if (it != proofs_.end() && *it == proof) {
                proofs_.erase(it);
            }
        }
        template <typename Predicate>
        void EraseIf(Predicate&& pred)
        {
            proofs_.erase(std::remove_if(proofs_.begin(), proofs_.end(), pred), proofs_.end());
        }

    private:
        static bool Less(const ConsumptionProof& lhs, const ConsumptionProof& rhs)
        {
            return std::tie(lhs.slot, lhs.generation) < std::tie(rhs.slot, rhs.generation);
        }
        std::vector<ConsumptionProof> proofs_;
    };
    enum class InstanceState { Allocated, WaitInserted, Retired, Cancelled };
    struct EventInstance {
        EventInstanceId id;
        size_t slot;
        PipePairEx pipes;
        size_t setIdx;
        size_t waitIdx;
        ConsumedInstances setSnapshot;
        Operation* setOp{nullptr};
        Operation* waitOp{nullptr};
        InstanceState state{InstanceState::Allocated};
        bool intraCore;
        ConsumptionProof Proof() const { return {slot, id.generation}; }
    };

    // Owns all event-instance transitions; the scheduler only reads instances.
    class EventInstanceTracker {
    public:
        const std::vector<EventInstance>& Instances() const { return eventInstances_; }
        const EventInstance* Find(size_t setIdx, size_t waitIdx) const;
        const EventInstance* FindActive(const EventResource& resource) const;
        ConsumedInstances Known(const PipeCoreRealEx& pipe) const;
        const ConsumedInstances& KnownForAllocationCheck(const PipeCoreRealEx& pipe) const;
        void MergeConsumed(ConsumedInstances& destination, const ConsumedInstances& snapshot) const;
        bool HasActive(const EventResource& resource) const { return FindActive(resource) != nullptr; }
        const EventResource& ProofResource(ConsumptionProof proof) const { return slots_[proof.slot].resource; }
        bool CanAllocate(const EventResource& resource, const ConsumedInstances& consumed,
                         bool requireActive = false) const;
        Status Allocate(const EventResource& resource, const PipePairEx& pipes, size_t setIdx, size_t waitIdx);
        Status Allocate(const EventResource& resource, const PipePairEx& pipes, size_t setIdx, size_t waitIdx,
                        ConsumedInstances consumed);
        Status RecordSet(const EventResource& resource, const PipePairEx& pipes, size_t setIdx, size_t waitIdx,
                         Operation& syncOp, bool intraCore);
        Status Consume(size_t setIdx, size_t waitIdx, Operation& syncOp);
        Status CancelPending(size_t setIdx, size_t waitIdx);
        Status MergePending(size_t set1, size_t wait1, size_t set2, size_t wait2);
        Status RetireConsumed(size_t setIdx, size_t waitIdx);
        void ForgetProofs();

    private:
        struct EventSlot {
            EventResource resource;
            uint64_t generation{0};
            std::optional<size_t> active;
        };
        std::optional<size_t> FindSlot(const EventResource& resource) const;
        size_t GetOrCreateSlot(const EventResource& resource);
        const EventInstance* FindActive(const EventSlot& slot) const;
        bool CanAllocate(const EventSlot* slot, const ConsumedInstances& consumed, bool requireActive) const;

        std::vector<EventInstance> eventInstances_;
        // Indices remain valid when the tracker is copied for a staged drain.
        std::vector<EventSlot> slots_;
        std::unordered_map<EventResource, size_t, EventResourceHash> slotByResource_;
        std::unordered_map<PipeCoreRealEx, ConsumedInstances, PipeCoreRealExHash> knownConsumed_;
        std::unordered_map<std::pair<size_t, size_t>, size_t, IndexVecHash> eventInstanceByDep_;
    };
};
} // namespace tile_fwk
} // namespace npu

#endif // PASS_INSERT_SYNC_TYPES_H
