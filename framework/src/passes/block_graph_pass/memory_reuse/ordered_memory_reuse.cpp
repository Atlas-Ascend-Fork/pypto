/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ordered_memory_reuse.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <queue>
#include "global_memory_reuse.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_token_utils.h"

#define MODULE_NAME "GlobalMemoryReuse"

namespace npu::tile_fwk {

bool OrderedReuseBucket::HasNonOverlappingOrderedInterval(const OrderedReuseLifetime& tensorsDesc) const
{
    return supportsOrderedReuse_ && tensorsDesc.supportsOrderedReuse &&
           lastUseOpIdx_ != ConnectionMatrix::INVALID_INDEX &&
           tensorsDesc.firstProducerOpIdx != ConnectionMatrix::INVALID_INDEX &&
           lastUseOpIdx_ < tensorsDesc.firstProducerOpIdx;
}

bool OrderedMemoryReuse::IsCommCallOp(Allocator& allocator, const Operation& op)
{
    auto cacheValue = Program::GetInstance().TryHitCahce(op.GetCalleeHash());
    if (cacheValue == std::nullopt || cacheValue->GetFunction() == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Cannot resolve callee hash %lu for op %d; disabling ordered memory reuse.",
                         op.GetCalleeHash().GetHash(), op.GetOpMagic());
        return true;
    }
    Function* callee = cacheValue->GetFunction();
    auto [entry, inserted] = allocator.orderedReuse_.communicationLeafCache.emplace(callee, false);
    if (inserted) {
        for (const Operation& leafOp : callee->Operations()) {
            if (OpcodeManager::Inst().IsSharedMemory(leafOp.GetOpcode())) {
                entry->second = true;
                break;
            }
        }
    }
    return entry->second;
}

void OrderedMemoryReuse::BuildBuckets(Allocator& allocator)
{
    allocator.buckets_.clear();
    allocator.dummyPackets_ = TensorBucket();
    allocator.tensorMagicToBucketIdx_.clear();
    allocator.bucketsIdxToSize_.clear();
    allocator.size_ = 0;
    APASS_LOG_INFO_F(Elements::Tensor, "Starting memory allocation with %zu storage entries.",
                     allocator.storageNeedToAllocate_.size());
    for (auto& tensorsDesc : allocator.storageNeedToAllocate_) {
        auto& tensor = *(tensorsDesc.tensors.begin());
        TensorBucket& bucket = allocator.GetBestFitBucket(tensorsDesc);
        bucket.AddTensorGroup(tensorsDesc);
        bucket.orderedReuse_.Update(tensorsDesc.orderedReuse);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Allocating storage for tensor: rawmagic=%d size=%lu.",
                          tensor->GetRawMagic(), static_cast<unsigned long>(tensor->storage_->length_));
    }
}

Status OrderedMemoryReuse::Preflight(const Allocator& allocator)
{
    for (const auto& desc : allocator.storageNeedToAllocate_) {
        if (desc.tensors.empty()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Memory allocation aborted: storage group has no tensors.");
            return FAILED;
        }
        for (const auto& tensor : desc.tensors) {
            if (tensor == nullptr || tensor->storage_ == nullptr) {
                APASS_LOG_ERROR_F(Elements::Tensor,
                                  "Memory allocation aborted: storage group has a null tensor or missing storage.");
                return FAILED;
            }
        }
    }
    for (const Operation& op : allocator.function_->Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        const auto attr = std::dynamic_pointer_cast<CallOpAttribute>(op.GetOpAttribute());
        if (attr == nullptr || attr->invokeInfo_ == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Memory allocation aborted: CALL op %d has no call attribute or invoke information.",
                              op.GetOpMagic());
            return FAILED;
        }
        if (attr->invokeInfo_->GetIncastTensorParamList().size() > op.iOperand.size() ||
            attr->invokeInfo_->GetOutcastTensorParamList().size() > op.oOperand.size()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Memory allocation aborted: CALL op %d has more incast/outcast parameters than operands.",
                              op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

void OrderedReuseBucket::Update(const OrderedReuseLifetime& next)
{
    // Existing topology or inserted tokens order successive groups, so the latest
    // lifetime endpoints cover all previous groups transitively.
    lifetimeEndOps_ = next.lifetimeEndOps;
    lastUseOpIdx_ = next.lastUseOpIdx;
    supportsOrderedReuse_ = next.supportsOrderedReuse;
}

uint64_t OrderedMemoryReuse::CollectLifetime(Allocator& allocator, TensorsDesc& tensorsDesc, Operation& op,
                                             bool isProducer)
{
    if (allocator.orderedReuse_.minimumParallelism < 0) {
        return op.GetOpcode() == Opcode::OP_CALL ? allocator.connectionMatrix_.GetIndex(op) :
                                                   ConnectionMatrix::INVALID_INDEX;
    }
    const uint64_t index = allocator.connectionMatrix_.GetIndex(op);
    // Keep all endpoints independently of DATA-only consumer index reduction.
    tensorsDesc.orderedReuse.lifetimeEndOps.emplace(&op);
    if (isProducer) {
        tensorsDesc.orderedReuse.producerOps.emplace(&op);
    }
    if (op.GetOpcode() == Opcode::OP_CALL) {
        if (IsCommCallOp(allocator, op)) {
            tensorsDesc.orderedReuse.supportsOrderedReuse = false;
        }
    } else {
        tensorsDesc.orderedReuse.supportsOrderedReuse = false;
    }
    if (index != ConnectionMatrix::INVALID_INDEX) {
        if (isProducer) {
            tensorsDesc.orderedReuse.firstProducerOpIdx = std::min(tensorsDesc.orderedReuse.firstProducerOpIdx, index);
        }
        tensorsDesc.orderedReuse.lastUseOpIdx = tensorsDesc.orderedReuse.lastUseOpIdx ==
                                                        ConnectionMatrix::INVALID_INDEX ?
                                                    index :
                                                    std::max(tensorsDesc.orderedReuse.lastUseOpIdx, index);
    }
    return index;
}

TensorBucket& OrderedMemoryReuse::GetBestFitNaturalBucket(Allocator& allocator, const TensorsDesc& tensorsDesc,
                                                          int64_t minimumSize, const std::unordered_set<int>& visited)
{
    const auto& first = *tensorsDesc.tensors.begin();
    const int64_t rawDataSize = first->tensor->GetRawDataSize();
    size_t best = allocator.buckets_.size();
    for (size_t index = 0; index < allocator.buckets_.size(); ++index) {
        if (visited.count(static_cast<int>(index)) == 0 && allocator.bucketsIdxToSize_[index] >= minimumSize &&
            (best == allocator.buckets_.size() ||
             allocator.bucketsIdxToSize_[index] < allocator.bucketsIdxToSize_[best]) &&
            allocator.buckets_[index].HasTopoDependency(tensorsDesc.connectionOpsBitmap)) {
            best = index;
        }
    }
    if (best == allocator.buckets_.size()) {
        return allocator.HandleNewBuckets(tensorsDesc, rawDataSize, first->magic);
    }
    allocator.bucketsIdxToSize_[best] = rawDataSize;
    allocator.UpdateTensorMagicToBucketIdx(tensorsDesc.tensors, static_cast<int>(best));
    return allocator.buckets_[best];
}

TensorBucket& OrderedMemoryReuse::GetBestFitBucket(Allocator& allocator, const TensorsDesc& tensorsDesc,
                                                   int64_t minimumSize)
{
    const auto& first = *tensorsDesc.tensors.begin();
    const int64_t rawDataSize = first->tensor->GetRawDataSize();
    auto select = [&](size_t index) -> TensorBucket& {
        allocator.bucketsIdxToSize_[index] = rawDataSize;
        allocator.UpdateTensorMagicToBucketIdx(tensorsDesc.tensors, static_cast<int>(index));
        return allocator.buckets_[index];
    };
    std::deque<LogicalTensorPtr> predecessorTensors(tensorsDesc.tensors.begin(), tensorsDesc.tensors.end());
    std::unordered_set<int> visitedTensor;
    std::unordered_set<size_t> visitedBucket;
    std::vector<size_t> candidates;
    while (!predecessorTensors.empty()) {
        LogicalTensorPtr ptr = predecessorTensors.front();
        predecessorTensors.pop_front();
        if (!visitedTensor.emplace(ptr->GetMagic()).second) {
            continue;
        }
        auto it = allocator.tensorMagicToBucketIdx_.find(ptr->GetMagic());
        if (it != allocator.tensorMagicToBucketIdx_.end()) {
            const size_t index = it->second;
            if (visitedBucket.emplace(index).second && allocator.bucketsIdxToSize_[index] >= minimumSize) {
                candidates.push_back(index);
            }
        }
        for (Operation* op : ptr->GetProducers()) {
            for (const auto& tensor : op->GetIOperands()) {
                predecessorTensors.push_back(tensor);
            }
        }
    }
    std::vector<size_t> globalCandidates;
    for (size_t index = 0; index < allocator.buckets_.size(); ++index) {
        if (visitedBucket.count(index) == 0 && allocator.bucketsIdxToSize_[index] >= minimumSize) {
            globalCandidates.push_back(index);
        }
    }
    std::stable_sort(globalCandidates.begin(), globalCandidates.end(), [&](size_t a, size_t b) {
        return allocator.bucketsIdxToSize_[a] < allocator.bucketsIdxToSize_[b];
    });
    candidates.insert(candidates.end(), globalCandidates.begin(), globalCandidates.end());
    // Prefer reuse requiring no additional ordering across both search paths.
    for (size_t index : candidates) {
        if (NaturallyOrdered(allocator.buckets_[index], tensorsDesc)) {
            return select(index);
        }
    }
    for (size_t index : candidates) {
        // A failed best-fit proof does not prevent trying the next bucket.
        if (TryAddDependencies(allocator.buckets_[index], tensorsDesc)) {
            return select(index);
        }
    }

    return allocator.HandleNewBuckets(tensorsDesc, rawDataSize, first->magic);
}

bool OrderedReuseGraph::Init(Function& function)
{
    operations_ = function.Operations(false).DuplicatedOpList();
    indices_.clear();
    indegrees_.assign(operations_.size(), 0);
    successors_.assign(operations_.size(), {});
    for (size_t i = 0; i < operations_.size(); ++i) {
        if (operations_[i]->BelongTo() != &function) {
            return false;
        }
        indices_.emplace(operations_[i], i);
    }
    for (Operation* op : operations_) {
        // DATA self references describe aliasing, as in ConnectionMatrix. Explicit
        // token/control/depend self edges are real scheduling cycles and are rejected.
        for (Operation* producer : op->ProducerOps()) {
            if (producer != op && !AddEdge(producer, op)) {
                return false;
            }
        }
        for (Operation* producer : op->ProducerOpsByToken()) {
            if (!AddEdge(producer, op)) {
                return false;
            }
        }
        for (Operation* producer : op->GetInCtrlOperations()) {
            if (!AddEdge(producer, op)) {
                return false;
            }
        }
        for (const auto& operand : op->GetDependOperands()) {
            for (Operation* producer : operand->GetProducers()) {
                if (!AddEdge(producer, op)) {
                    return false;
                }
            }
        }
    }
    calls_.clear();
    for (size_t i = 0; i < operations_.size(); ++i) {
        if (operations_[i]->GetOpcode() == Opcode::OP_CALL) {
            calls_.push_back(i);
        }
    }
    leftMatch_.assign(calls_.size(), calls_.size());
    rightMatch_ = leftMatch_;
    antichain_.clear();
    matched_ = 0;
    return RefreshReachability() && ProveWidth(0);
}

bool OrderedReuseGraph::AddEdge(Operation* from, Operation* to)
{
    auto before = indices_.find(from);
    auto after = indices_.find(to);
    if (before == indices_.end() || after == indices_.end() || from == to) {
        return false;
    }
    if (successors_[before->second].insert(after->second).second) {
        ++indegrees_[after->second];
    }
    return true;
}

bool OrderedReuseGraph::RefreshReachability()
{
    ancestors_.assign(operations_.size(), LargeBitmap(operations_.size()));
    auto indegrees = indegrees_;
    std::queue<size_t> ready;
    for (size_t i = 0; i < operations_.size(); ++i) {
        if (indegrees[i] == 0) {
            ready.push(i);
        }
    }
    size_t visited = 0;
    while (!ready.empty()) {
        const size_t current = ready.front();
        ready.pop();
        ++visited;
        for (size_t next : successors_[current]) {
            ancestors_[next].Or(ancestors_[current]);
            ancestors_[next].SetBit(current);
            if (--indegrees[next] == 0) {
                ready.push(next);
            }
        }
    }
    return visited == operations_.size();
}

bool OrderedReuseGraph::Reaches(size_t from, size_t to) const { return from != to && ancestors_[to].GetBit(from); }

bool OrderedReuseGraph::AllReachable(const Endpoints& ends, const Endpoints& producers) const
{
    if (ends.empty() || producers.empty()) {
        return false;
    }
    for (Operation* end : ends) {
        const auto before = indices_.find(end);
        if (before == indices_.end()) {
            return false;
        }
        for (Operation* producer : producers) {
            const auto after = indices_.find(producer);
            if (after == indices_.end() || !Reaches(before->second, after->second)) {
                return false;
            }
        }
    }
    return true;
}

bool OrderedReuseGraph::Order(const Endpoints& ends, const Endpoints& producers)
{
    if (ends.empty() || producers.empty()) {
        return false;
    }
    for (Operation* end : ends) {
        const auto before = indices_.find(end);
        if (before == indices_.end()) {
            return false;
        }
        for (Operation* producer : producers) {
            const auto after = indices_.find(producer);
            if (after == indices_.end() || end == producer || Reaches(after->second, before->second)) {
                return false;
            }
        }
    }
    // Every new edge goes from ends to producers. Any cycle would need an original
    // producer-to-end path, which the complete Cartesian precheck above rejects.
    // Update the closure one edge at a time, propagating all ancestors of the
    // source to all descendants of the sink.
    for (Operation* end : ends) {
        const size_t from = indices_.at(end);
        for (Operation* producer : producers) {
            const size_t to = indices_.at(producer);
            if (Reaches(from, to)) {
                continue;
            }
            AddEdge(end, producer);
            LargeBitmap prefix = ancestors_[from];
            prefix.SetBit(from);
            for (size_t node = 0; node < operations_.size(); ++node) {
                if (node == to || Reaches(to, node)) {
                    ancestors_[node].Or(prefix);
                }
            }
        }
    }
    return true;
}

bool OrderedReuseGraph::WitnessSurvives() const
{
    if (antichain_.size() != Width()) {
        return false;
    }
    for (size_t i = 0; i < antichain_.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            const size_t a = calls_[antichain_[i]];
            const size_t b = calls_[antichain_[j]];
            if (Reaches(a, b) || Reaches(b, a)) {
                return false;
            }
        }
    }
    return true;
}

bool OrderedReuseGraph::ProveWidth(size_t floor)
{
    const size_t count = calls_.size();
    if (floor > count) {
        return false;
    }
    // Adding edges cannot increase width. A surviving maximum antichain is an
    // exact proof that the previous width is unchanged, without another matching.
    if (WitnessSurvives()) {
        return Width() >= floor;
    }
    std::vector<std::vector<size_t>> edges(count);
    for (size_t from = 0; from < count; ++from) {
        for (size_t to = 0; to < count; ++to) {
            if (Reaches(calls_[from], calls_[to])) {
                edges[from].push_back(to);
            }
        }
    }
    // Hopcroft-Karp on strict transitive CALL reachability. The previous maximum
    // matching remains valid after edge insertion and is extended, never reset.
    const size_t infinity = count + 1;
    std::vector<size_t> distance(count);
    while (true) {
        std::queue<size_t> ready;
        std::fill(distance.begin(), distance.end(), infinity);
        for (size_t left = 0; left < count; ++left) {
            if (leftMatch_[left] == count) {
                distance[left] = 0;
                ready.push(left);
            }
        }
        size_t shortest = infinity;
        while (!ready.empty()) {
            const size_t left = ready.front();
            ready.pop();
            if (distance[left] >= shortest) {
                continue;
            }
            for (size_t right : edges[left]) {
                const size_t next = rightMatch_[right];
                if (next == count) {
                    shortest = distance[left] + 1;
                } else if (distance[next] == infinity) {
                    distance[next] = distance[left] + 1;
                    ready.push(next);
                }
            }
        }
        if (shortest == infinity) {
            break;
        }
        std::function<bool(size_t)> augment = [&](size_t left) {
            for (size_t right : edges[left]) {
                const size_t next = rightMatch_[right];
                if ((next == count && distance[left] + 1 == shortest) ||
                    (next != count && distance[next] == distance[left] + 1 && augment(next))) {
                    leftMatch_[left] = right;
                    rightMatch_[right] = left;
                    return true;
                }
            }
            distance[left] = infinity;
            return false;
        };
        for (size_t left = 0; left < count; ++left) {
            if (leftMatch_[left] == count && augment(left)) {
                ++matched_;
                if (count - matched_ < floor) {
                    return false; // An upper bound already disproves the requested floor.
                }
            }
        }
    }
    FindAntichain(edges);
    return Width() >= floor;
}

void OrderedReuseGraph::FindAntichain(const std::vector<std::vector<size_t>>& edges)
{
    const size_t count = calls_.size();
    std::vector<bool> leftSeen(count, false), rightSeen(count, false);
    std::queue<size_t> ready;
    for (size_t left = 0; left < count; ++left) {
        if (leftMatch_[left] == count) {
            leftSeen[left] = true;
            ready.push(left);
        }
    }
    while (!ready.empty()) {
        const size_t left = ready.front();
        ready.pop();
        for (size_t right : edges[left]) {
            if (leftMatch_[left] == right || rightSeen[right]) {
                continue;
            }
            rightSeen[right] = true;
            const size_t next = rightMatch_[right];
            if (next != count && !leftSeen[next]) {
                leftSeen[next] = true;
                ready.push(next);
            }
        }
    }
    antichain_.clear();
    for (size_t call = 0; call < count; ++call) {
        if (leftSeen[call] && !rightSeen[call]) {
            antichain_.push_back(call);
        }
    }
}

bool OrderedReuseGraph::TryOrder(const Endpoints& ends, const Endpoints& producers, size_t floor)
{
    OrderedReuseGraph trial = *this;
    if (!trial.Order(ends, producers) || !trial.ProveWidth(floor)) {
        return false;
    }
    *this = std::move(trial);
    return true;
}

bool OrderedReuseGraph::SameCallOrder(const OrderedReuseGraph& other) const
{
    if (operations_ != other.operations_ || calls_ != other.calls_) {
        return false;
    }
    for (size_t from : calls_) {
        for (size_t to : calls_) {
            if (Reaches(from, to) != other.Reaches(from, to)) {
                return false;
            }
        }
    }
    return true;
}

bool OrderedMemoryReuse::NaturallyOrdered(const TensorBucket& bucket, const TensorsDesc& next) const
{
    return graph_.AllReachable(bucket.orderedReuse_.lifetimeEndOps_, next.orderedReuse.producerOps);
}

bool OrderedMemoryReuse::TryAddDependencies(const TensorBucket& bucket, const TensorsDesc& next)
{
    if (!bucket.orderedReuse_.HasNonOverlappingOrderedInterval(next.orderedReuse)) {
        return false;
    }
    const auto& ends = bucket.orderedReuse_.lifetimeEndOps_;
    const auto& producers = next.orderedReuse.producerOps;
    std::vector<std::pair<Operation*, Operation*>> missing;
    for (Operation* end : ends) {
        for (Operation* producer : producers) {
            if (!graph_.AllReachable({end}, {producer})) {
                missing.emplace_back(end, producer);
            }
        }
    }
    if (!graph_.TryOrder(ends, producers, floor_)) {
        return false;
    }
    for (const auto& [end, producer] : missing) {
        addedConsumers_[end].insert(producer);
    }
    return true;
}

bool OrderedMemoryReuse::Materialize(Function& function)
{
    for (Operation* end : graph_.Operations()) {
        auto it = addedConsumers_.find(end);
        if (it == addedConsumers_.end()) {
            continue;
        }
        // A NORMAL completion token has one producer. Never reuse an existing
        // semantic/shared token, whose other producers would add unplanned edges.
        auto token = IRBuilder().CreateTokenVar(end->GetSpan());
        addedTokens_.push_back({end, token, it->second});
        end->result_token_.push_back(token);
        function.GetVarDependency().AddProducer(token,
                                                std::static_pointer_cast<const ir::Stmt>(end->shared_from_this()));
        for (Operation* producer : it->second) {
            PassTokenUtils::AddTokenConsumer(function, token, *producer);
        }
    }
    OrderedReuseGraph actual;
    return actual.Init(function) && actual.Width() >= floor_ && graph_.SameCallOrder(actual);
}

void OrderedMemoryReuse::Rollback(Function& function)
{
    for (const auto& entry : addedTokens_) {
        for (Operation* consumer : entry.consumers) {
            auto& tokens = consumer->tokens_;
            tokens.erase(std::remove(tokens.begin(), tokens.end(), entry.token), tokens.end());
            function.GetVarDependency().RemoveConsumer(
                entry.token, std::static_pointer_cast<const ir::Stmt>(consumer->shared_from_this()));
        }
        auto& results = entry.end->result_token_;
        results.erase(std::remove(results.begin(), results.end(), entry.token), results.end());
        function.GetVarDependency().RemoveProducer(
            entry.token, std::static_pointer_cast<const ir::Stmt>(entry.end->shared_from_this()));
        function.GetVarDependency().RemoveVar(entry.token);
    }
    addedTokens_.clear();
}

uint64_t OrderedMemoryReuse::Bytes(const std::vector<TensorBucket>& buckets, const TensorBucket& dummy)
{
    uint64_t result = dummy.GetSize();
    for (const auto& bucket : buckets) {
        result += bucket.GetSize();
    }
    return result;
}

Status OrderedMemoryReuse::Build(Allocator& allocator)
{
    if (Preflight(allocator) != SUCCESS) {
        return FAILED;
    }
    BuildBuckets(allocator);
    if (allocator.skipReuseJudgment_) {
        return SUCCESS;
    }
    if (!graph_.Init(*allocator.function_)) {
        APASS_LOG_WARN_F(Elements::Graph, "Cannot prove root scheduling DAG; retaining natural memory reuse.");
        return SUCCESS;
    }
    const size_t naturalWidth = graph_.Width();
    const uint64_t naturalBytes = Bytes(allocator.buckets_, allocator.dummyPackets_);
    if (naturalWidth <= floor_) {
        APASS_LOG_INFO_F(Elements::Graph,
                         "Retaining natural memory reuse: initial CALL width=%zu does not exceed floor=%zu; "
                         "natural bucket bytes=%lu, added dependencies=0.",
                         naturalWidth, floor_, naturalBytes);
        return SUCCESS;
    }
    auto natural = std::move(allocator.buckets_);
    auto naturalDummy = std::move(allocator.dummyPackets_);
    allocator.orderedReuse_.active = this;
    BuildBuckets(allocator);
    allocator.orderedReuse_.active = nullptr;
    const uint64_t orderedBytes = Bytes(allocator.buckets_, allocator.dummyPackets_);
    size_t addedDependencies = 0;
    for (const auto& entry : addedConsumers_) {
        addedDependencies += entry.second.size();
    }
    APASS_LOG_INFO_F(Elements::Graph,
                     "Ordered memory reuse plan: CALL width=%zu -> %zu, floor=%zu, "
                     "bucket bytes=%lu -> %lu, planned added dependencies=%zu.",
                     naturalWidth, graph_.Width(), floor_, naturalBytes, orderedBytes, addedDependencies);
    bool keepNatural = orderedBytes >= naturalBytes;
    if (keepNatural) {
        APASS_LOG_INFO_F(Elements::Graph, "Retaining natural memory reuse: ordered plan has no byte savings.");
    } else if (!Materialize(*allocator.function_)) {
        APASS_LOG_WARN_F(Elements::Graph,
                         "Retaining natural memory reuse: materialized scheduling graph failed validation.");
        keepNatural = true;
    } else {
        APASS_LOG_INFO_F(Elements::Graph, "Selected ordered memory reuse: materialized scheduling graph validated.");
    }
    if (keepNatural) {
        Rollback(*allocator.function_);
        allocator.buckets_ = std::move(natural);
        allocator.dummyPackets_ = std::move(naturalDummy);
        // The accepted trial already has complete mappings; only fallback needs rebuilding.
        allocator.tensorMagicToBucketIdx_.clear();
        allocator.bucketsIdxToSize_.clear();
        for (size_t i = 0; i < allocator.buckets_.size(); ++i) {
            for (const auto& tensors : allocator.buckets_[i].tensorGroups_) {
                allocator.UpdateTensorMagicToBucketIdx(tensors, static_cast<int>(i));
                allocator.bucketsIdxToSize_[i] = (*tensors.begin())->tensor->GetRawDataSize();
            }
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
