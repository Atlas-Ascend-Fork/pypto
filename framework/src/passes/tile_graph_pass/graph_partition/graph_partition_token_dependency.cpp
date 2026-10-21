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
 * \file graph_partition_token_dependency.cpp
 * \brief Token dependency handling for graph partition finalization.
 */

#include "passes/tile_graph_pass/graph_partition/graph_partition_token_dependency.h"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/configs/config_manager_ng.h"
#include "interface/function/function.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/token_utils.h"

#define MODULE_NAME "GraphPartition"

namespace npu::tile_fwk {
namespace {

Operation* ToOperation(const ir::StmtPtr& stmt)
{
    if (stmt == nullptr) {
        return nullptr;
    }
    return static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get()));
}

int GetMaxSubgraphId(Function& function)
{
    int maxSubgraphId = -1;
    for (auto& op : function.Operations()) {
        maxSubgraphId = std::max(maxSubgraphId, op.GetSubgraphID());
    }
    return maxSubgraphId;
}

void AddSubgraphEdge(std::vector<std::set<int>>& graph, int src, int dst)
{
    if (src < 0 || dst < 0 || src == dst) {
        return;
    }
    const int maxId = std::max(src, dst);
    if (maxId >= static_cast<int>(graph.size())) {
        graph.resize(static_cast<size_t>(maxId + 1));
    }
    graph[src].insert(dst);
}

std::vector<std::set<int>> BuildSubgraphDependencyGraph(Function& function)
{
    std::vector<std::set<int>> graph(static_cast<size_t>(GetMaxSubgraphId(function) + 1));
    for (auto& op : function.Operations()) {
        int srcSubgraphId = op.GetSubgraphID();
        for (const auto& output : op.GetOOperands()) {
            if (output == nullptr) {
                continue;
            }
            for (auto* consumer : output->GetConsumers()) {
                if (consumer == nullptr) {
                    continue;
                }
                AddSubgraphEdge(graph, srcSubgraphId, consumer->GetSubgraphID());
            }
        }
    }
    const auto dependencies = function.GetVarDependency().GetAllDependencies();
    for (const auto& [token, entry] : dependencies) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            auto* producerOp = ToOperation(producerStmt);
            if (producerOp == nullptr) {
                continue;
            }
            for (const auto& consumerStmt : entry.consumers) {
                auto* consumerOp = ToOperation(consumerStmt);
                if (consumerOp == nullptr) {
                    continue;
                }
                AddSubgraphEdge(graph, producerOp->GetSubgraphID(), consumerOp->GetSubgraphID());
            }
        }
    }
    return graph;
}

std::vector<std::vector<int>> FindStronglyConnectedComponents(const std::vector<std::set<int>>& graph)
{
    std::vector<int> dfn(graph.size(), -1);
    std::vector<int> low(graph.size(), -1);
    std::vector<int> stack;
    std::vector<bool> inStack(graph.size(), false);
    std::vector<std::vector<int>> sccs;
    int timestamp = 0;
    std::function<void(int)> dfs = [&](int node) {
        dfn[node] = low[node] = timestamp++;
        stack.push_back(node);
        inStack[node] = true;
        for (int next : graph[node]) {
            if (dfn[next] < 0) {
                dfs(next);
                low[node] = std::min(low[node], low[next]);
            } else if (inStack[next]) {
                low[node] = std::min(low[node], dfn[next]);
            }
        }
        if (low[node] != dfn[node]) {
            return;
        }
        std::vector<int> component;
        while (!stack.empty()) {
            int curr = stack.back();
            stack.pop_back();
            inStack[curr] = false;
            component.push_back(curr);
            if (curr == node) {
                break;
            }
        }
        sccs.push_back(std::move(component));
    };
    for (size_t i = 0; i < graph.size(); ++i) {
        if (dfn[i] < 0) {
            dfs(static_cast<int>(i));
        }
    }
    return sccs;
}

class SubgraphUnionFind {
public:
    explicit SubgraphUnionFind(size_t size) : parent_(size) { std::iota(parent_.begin(), parent_.end(), 0); }

    int Find(int node)
    {
        if (parent_[node] != node) {
            parent_[node] = Find(parent_[node]);
        }
        return parent_[node];
    }

    void Union(int lhs, int rhs)
    {
        int lhsRoot = Find(lhs);
        int rhsRoot = Find(rhs);
        if (lhsRoot == rhsRoot) {
            return;
        }
        parent_[std::max(lhsRoot, rhsRoot)] = std::min(lhsRoot, rhsRoot);
    }

private:
    std::vector<int> parent_;
};

constexpr uint32_t kAicCoreFlag = 1U;
constexpr uint32_t kAivCoreFlag = 2U;

uint32_t GetOperationCoreFlag(const Operation& op)
{
    const auto coreType = OpcodeManager::Inst().GetCoreType(op.GetOpcode());
    if (coreType == OpCoreType::AIC) {
        return kAicCoreFlag;
    }
    if (coreType == OpCoreType::AIV) {
        return kAivCoreFlag;
    }

    auto getTensorCoreFlag = [](const LogicalTensorPtr& tensor) {
        if (tensor == nullptr) {
            return 0U;
        }
        const auto memoryType = tensor->GetMemoryTypeOriginal();
        if (memoryType == MemoryType::MEM_L1 || memoryType == MemoryType::MEM_L0A ||
            memoryType == MemoryType::MEM_L0B || memoryType == MemoryType::MEM_L0C) {
            return kAicCoreFlag;
        }
        if (memoryType == MemoryType::MEM_UB) {
            return kAivCoreFlag;
        }
        return 0U;
    };
    uint32_t tensorCoreFlags = 0U;
    for (const auto& tensor : op.GetIOperands()) {
        tensorCoreFlags |= getTensorCoreFlag(tensor);
    }
    for (const auto& tensor : op.GetOOperands()) {
        tensorCoreFlags |= getTensorCoreFlag(tensor);
    }
    for (const auto& tensor : op.GetDependOperands()) {
        tensorCoreFlags |= getTensorCoreFlag(tensor);
    }
    return tensorCoreFlags;
}

bool HasAicAivConflict(uint32_t coreFlags)
{
    return (coreFlags & kAicCoreFlag) != 0U && (coreFlags & kAivCoreFlag) != 0U;
}

uint32_t GetFixedOperationCoreFlag(const Operation& op)
{
    const auto coreType = OpcodeManager::Inst().GetCoreType(op.GetOpcode());
    if (coreType == OpCoreType::AIC) {
        return kAicCoreFlag;
    }
    if (coreType == OpCoreType::AIV) {
        return kAivCoreFlag;
    }
    return 0U;
}

uint32_t GetPostLoweringOperationCoreFlag(const Operation& op)
{
    const uint32_t fixedCoreFlag = GetFixedOperationCoreFlag(op);
    if (fixedCoreFlag != 0U) {
        return fixedCoreFlag;
    }

    const auto opcode = op.GetOpcode();
    const bool willGenerateMove = opcode == Opcode::OP_ASSEMBLE_SSA || opcode == config::GetContractOpcode() ||
                                  opcode == config::GetSliceOpcode() || opcode == Opcode::OP_CONVERT ||
                                  opcode == Opcode::OP_DUPLICATE;
    const bool isCopy = opcode == Opcode::OP_COPY_IN || opcode == Opcode::OP_COPY_OUT;
    if ((!willGenerateMove && !isCopy) || op.GetIOperands().empty() || op.GetOOperands().empty() ||
        op.GetIOperands().front() == nullptr || op.GetOOperands().front() == nullptr) {
        return 0U;
    }

    const auto from = op.GetIOperands().front()->GetMemoryTypeOriginal();
    const auto to = op.GetOOperands().front()->GetMemoryTypeOriginal();
    if (from == to) {
        return 0U;
    }
    if (from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) {
        return kAicCoreFlag;
    }
    if (from == MemoryType::MEM_UB && to == MemoryType::MEM_L1) {
        return kAivCoreFlag;
    }
    const uint32_t coreFlag = GetOperationCoreFlag(op);
    return HasAicAivConflict(coreFlag) ? 0U : coreFlag;
}

bool IsProducerAnchoredPostLoweringMove(const Operation& op)
{
    const auto opcode = op.GetOpcode();
    const bool willGenerateMove = opcode == Opcode::OP_ASSEMBLE_SSA || opcode == config::GetContractOpcode() ||
                                  opcode == config::GetSliceOpcode() || opcode == Opcode::OP_CONVERT ||
                                  opcode == Opcode::OP_DUPLICATE;
    if ((!willGenerateMove && opcode != Opcode::OP_COPY_OUT) || op.GetIOperands().empty() ||
        op.GetOOperands().empty() || op.GetIOperands().front() == nullptr || op.GetOOperands().front() == nullptr) {
        return false;
    }
    const auto from = op.GetIOperands().front()->GetMemoryTypeOriginal();
    const auto to = op.GetOOperands().front()->GetMemoryTypeOriginal();
    return (from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) ||
           (from == MemoryType::MEM_UB && to == MemoryType::MEM_L1) ||
           ((from == MemoryType::MEM_L0C || from == MemoryType::MEM_UB) && to == MemoryType::MEM_DEVICE_DDR);
}

bool IsConsumerAnchoredPostLoweringMove(const Operation& op)
{
    const auto opcode = op.GetOpcode();
    const bool willGenerateMove = opcode == Opcode::OP_ASSEMBLE_SSA || opcode == config::GetContractOpcode() ||
                                  opcode == config::GetSliceOpcode() || opcode == Opcode::OP_CONVERT ||
                                  opcode == Opcode::OP_DUPLICATE;
    if ((!willGenerateMove && opcode != Opcode::OP_COPY_IN) || op.GetIOperands().empty() || op.GetOOperands().empty() ||
        op.GetIOperands().front() == nullptr || op.GetOOperands().front() == nullptr) {
        return false;
    }
    const auto from = op.GetIOperands().front()->GetMemoryTypeOriginal();
    const auto to = op.GetOOperands().front()->GetMemoryTypeOriginal();
    return from == MemoryType::MEM_DEVICE_DDR && (to == MemoryType::MEM_L1 || to == MemoryType::MEM_UB);
}

std::unordered_set<int> CollectMixedPostLoweringCoreSubgraphIds(Function& function)
{
    std::unordered_map<int, uint32_t> coreFlagsBySubgraph;
    for (auto& op : function.Operations()) {
        const int subgraphId = op.GetSubgraphID();
        if (subgraphId < 0) {
            continue;
        }
        coreFlagsBySubgraph[subgraphId] |= GetPostLoweringOperationCoreFlag(op);
    }

    std::unordered_set<int> mixedSubgraphIds;
    for (const auto& [subgraphId, coreFlags] : coreFlagsBySubgraph) {
        if (HasAicAivConflict(coreFlags)) {
            mixedSubgraphIds.insert(subgraphId);
        }
    }
    return mixedSubgraphIds;
}

std::vector<uint32_t> CollectSubgraphCoreFlags(Function& function, int maxSubgraphId,
                                               bool inferPostLoweringCore = false)
{
    std::vector<uint32_t> subgraphCoreFlags(static_cast<size_t>(maxSubgraphId + 1), 0U);
    for (auto& op : function.Operations()) {
        const int subgraphId = op.GetSubgraphID();
        if (subgraphId < 0) {
            continue;
        }
        uint32_t coreFlag = GetOperationCoreFlag(op);
        if (inferPostLoweringCore) {
            const uint32_t postLoweringCoreFlag = GetPostLoweringOperationCoreFlag(op);
            if (postLoweringCoreFlag != 0U) {
                coreFlag = postLoweringCoreFlag;
            }
        }
        subgraphCoreFlags[subgraphId] |= coreFlag;
    }
    return subgraphCoreFlags;
}

std::vector<std::set<int>> BuildMergedSubgraphDependencyGraph(Function& function, SubgraphUnionFind& unionFind)
{
    const auto graph = BuildSubgraphDependencyGraph(function);
    std::vector<std::set<int>> mergedGraph(graph.size());
    for (size_t src = 0; src < graph.size(); ++src) {
        for (int dst : graph[src]) {
            AddSubgraphEdge(mergedGraph, unionFind.Find(static_cast<int>(src)), unionFind.Find(dst));
        }
    }
    return mergedGraph;
}

std::vector<bool> FindReachableSubgraphs(const std::vector<std::set<int>>& graph, int start)
{
    std::vector<bool> reachable(graph.size(), false);
    if (start < 0 || start >= static_cast<int>(graph.size())) {
        return reachable;
    }
    std::vector<int> pending{start};
    reachable[start] = true;
    while (!pending.empty()) {
        const int node = pending.back();
        pending.pop_back();
        for (int next : graph[node]) {
            if (!reachable[next]) {
                reachable[next] = true;
                pending.push_back(next);
            }
        }
    }
    return reachable;
}

std::vector<int> CollectPathClosure(const std::vector<std::set<int>>& graph, int producer, int consumer)
{
    const auto reachableFromProducer = FindReachableSubgraphs(graph, producer);
    if (consumer < 0 || consumer >= static_cast<int>(reachableFromProducer.size()) ||
        !reachableFromProducer[consumer]) {
        return {};
    }

    std::vector<std::set<int>> reverseGraph(graph.size());
    for (size_t src = 0; src < graph.size(); ++src) {
        for (int dst : graph[src]) {
            reverseGraph[dst].insert(static_cast<int>(src));
        }
    }
    const auto canReachConsumer = FindReachableSubgraphs(reverseGraph, consumer);
    std::vector<int> pathClosure;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (reachableFromProducer[i] && canReachConsumer[i]) {
            pathClosure.push_back(static_cast<int>(i));
        }
    }
    return pathClosure;
}

void UpdateMergedSubgraphIds(Function& function, SubgraphUnionFind& unionFind)
{
    std::map<int, int> rootToNewId;
    for (auto& op : function.Operations()) {
        int oldId = op.GetSubgraphID();
        if (oldId < 0) {
            continue;
        }
        int root = unionFind.Find(oldId);
        auto [it, inserted] = rootToNewId.emplace(root, static_cast<int>(rootToNewId.size()));
        (void)inserted;
        op.UpdateSubgraphID(it->second);
    }
    function.SetTotalSubGraphCount(rootToNewId.size());
}

bool IsValidSubgraphBoundaryTensor(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return true;
    }
    const auto memoryType = tensor->GetMemoryTypeOriginal();
    return memoryType == MemoryType::MEM_UB || memoryType == MemoryType::MEM_L1 || memoryType == MemoryType::MEM_L0C ||
           memoryType == MemoryType::MEM_DEVICE_DDR;
}

Status MergeInvalidBoundaryConnectedSubgraphs(Function& function, bool inferPostLoweringCore = false)
{
    const int maxSubgraphId = GetMaxSubgraphId(function);
    if (maxSubgraphId < 0) {
        return SUCCESS;
    }
    SubgraphUnionFind unionFind(static_cast<size_t>(maxSubgraphId + 1));
    auto subgraphCoreFlags = CollectSubgraphCoreFlags(function, maxSubgraphId, inferPostLoweringCore);
    auto mergedGraph = BuildMergedSubgraphDependencyGraph(function, unionFind);
    std::unordered_set<const LogicalTensor*> visitedTensors;
    bool changed = false;
    for (auto& op : function.Operations()) {
        for (const auto& tensor : op.GetOOperands()) {
            if (IsValidSubgraphBoundaryTensor(tensor) || !visitedTensors.insert(tensor.get()).second) {
                continue;
            }
            for (auto* producer : tensor->GetProducers()) {
                if (producer == nullptr || producer->GetSubgraphID() < 0) {
                    continue;
                }
                for (auto* consumer : tensor->GetConsumers()) {
                    if (consumer == nullptr || consumer->GetSubgraphID() < 0) {
                        continue;
                    }
                    const int producerRoot = unionFind.Find(producer->GetSubgraphID());
                    const int consumerRoot = unionFind.Find(consumer->GetSubgraphID());
                    if (producerRoot == consumerRoot) {
                        continue;
                    }
                    const auto pathClosure = CollectPathClosure(mergedGraph, producerRoot, consumerRoot);
                    uint32_t mergedCoreFlags = 0U;
                    for (int subgraphId : pathClosure) {
                        mergedCoreFlags |= subgraphCoreFlags[unionFind.Find(subgraphId)];
                    }
                    if (pathClosure.empty() || HasAicAivConflict(mergedCoreFlags)) {
                        APASS_LOG_ERROR_F(Elements::Tensor,
                                          "Cannot keep local tensor %d inside a core-homogeneous subgraph.",
                                          tensor->GetMagic());
                        return FAILED;
                    }
                    for (int subgraphId : pathClosure) {
                        unionFind.Union(producerRoot, subgraphId);
                    }
                    subgraphCoreFlags[unionFind.Find(producerRoot)] = mergedCoreFlags;
                    mergedGraph = BuildMergedSubgraphDependencyGraph(function, unionFind);
                    changed = true;
                }
            }
        }
    }
    if (changed) {
        UpdateMergedSubgraphIds(function, unionFind);
    }
    return SUCCESS;
}

void CoalesceCompatibleAffectedSubgraphs(Function& function,
                                         const std::unordered_set<const Operation*>& affectedOperations,
                                         bool inferPostLoweringCore = false)
{
    const int maxSubgraphId = GetMaxSubgraphId(function);
    if (maxSubgraphId < 0) {
        return;
    }
    SubgraphUnionFind unionFind(static_cast<size_t>(maxSubgraphId + 1));
    auto subgraphCoreFlags = CollectSubgraphCoreFlags(function, maxSubgraphId, inferPostLoweringCore);
    auto mergedGraph = BuildMergedSubgraphDependencyGraph(function, unionFind);
    std::vector<uint8_t> affectedState(static_cast<size_t>(maxSubgraphId + 1), 0U);
    for (auto& op : function.Operations()) {
        const int subgraphId = op.GetSubgraphID();
        if (subgraphId < 0) {
            continue;
        }
        if (affectedOperations.count(&op) == 0U) {
            affectedState[subgraphId] = 2U;
        } else if (affectedState[subgraphId] == 0U) {
            affectedState[subgraphId] = 1U;
        }
    }

    bool changed = false;
    for (auto& producer : function.Operations()) {
        if (affectedOperations.count(&producer) == 0U) {
            continue;
        }
        for (const auto& output : producer.GetOOperands()) {
            if (output == nullptr) {
                continue;
            }
            for (auto* consumer : output->GetConsumers()) {
                if (consumer == nullptr || affectedOperations.count(consumer) == 0U) {
                    continue;
                }
                const int producerRoot = unionFind.Find(producer.GetSubgraphID());
                const int consumerRoot = unionFind.Find(consumer->GetSubgraphID());
                if (producerRoot == consumerRoot) {
                    continue;
                }
                const auto pathClosure = CollectPathClosure(mergedGraph, producerRoot, consumerRoot);
                uint32_t mergedCoreFlags = 0U;
                bool allAffected = !pathClosure.empty();
                for (int subgraphId : pathClosure) {
                    const int root = unionFind.Find(subgraphId);
                    mergedCoreFlags |= subgraphCoreFlags[root];
                    allAffected = allAffected && affectedState[root] == 1U;
                }
                if (!allAffected || HasAicAivConflict(mergedCoreFlags)) {
                    continue;
                }
                for (int subgraphId : pathClosure) {
                    unionFind.Union(producerRoot, subgraphId);
                }
                const int mergedRoot = unionFind.Find(producerRoot);
                subgraphCoreFlags[mergedRoot] = mergedCoreFlags;
                affectedState[mergedRoot] = 1U;
                mergedGraph = BuildMergedSubgraphDependencyGraph(function, unionFind);
                changed = true;
            }
        }
    }
    if (changed) {
        UpdateMergedSubgraphIds(function, unionFind);
    }
}

void UpdateAffectedSubgraphCoreAttributes(Function& function,
                                          const std::unordered_set<const Operation*>& affectedOperations,
                                          bool inferPostLoweringCore)
{
    std::unordered_set<int> affectedSubgraphIds;
    for (const auto* op : affectedOperations) {
        affectedSubgraphIds.insert(op->GetSubgraphID());
    }

    std::unordered_map<int, uint32_t> coreFlagsBySubgraph;
    for (auto& op : function.Operations()) {
        if (affectedSubgraphIds.count(op.GetSubgraphID()) != 0U) {
            const uint32_t coreFlag = inferPostLoweringCore ? GetPostLoweringOperationCoreFlag(op) :
                                                              GetOperationCoreFlag(op);
            coreFlagsBySubgraph[op.GetSubgraphID()] |= coreFlag;
        }
    }
    for (auto& op : function.Operations()) {
        const auto iter = coreFlagsBySubgraph.find(op.GetSubgraphID());
        if (iter == coreFlagsBySubgraph.end()) {
            continue;
        }
        if (iter->second == kAicCoreFlag) {
            op.SetAttribute(OpAttributeKey::isCube, true);
        } else if (iter->second == kAivCoreFlag) {
            op.SetAttribute(OpAttributeKey::isCube, false);
        }
    }
}

Status SplitMixedCoreSubgraphs(Function& function, const std::unordered_set<int>& subgraphIds,
                               bool inferPostLoweringCore = false)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT_STABLE);
    const auto sortedOperations = function.Operations(false).DuplicatedOpList();
    std::unordered_map<const Operation*, int> originalSubgraphIds;
    for (const auto* op : sortedOperations) {
        originalSubgraphIds.emplace(op, op->GetSubgraphID());
    }
    std::unordered_set<const Operation*> affectedOperations;
    std::unordered_map<int, int> preservedSubgraphIds;
    std::unordered_map<const Operation*, int> assignedSubgraphIds;
    std::unordered_map<int, uint32_t> splitCoreFlags;
    int nextSubgraphId = 0;
    int currentSplitSubgraphId = -1;
    int previousOldSubgraphId = -1;
    uint32_t currentCoreFlags = 0U;
    bool previousWasSplit = false;
    for (auto* op : sortedOperations) {
        const int oldSubgraphId = op->GetSubgraphID();
        if (subgraphIds.count(oldSubgraphId) == 0U) {
            auto [iter, inserted] = preservedSubgraphIds.emplace(oldSubgraphId, nextSubgraphId);
            if (inserted) {
                ++nextSubgraphId;
            }
            op->UpdateSubgraphID(iter->second);
            assignedSubgraphIds.emplace(op, iter->second);
            previousWasSplit = false;
            continue;
        }

        affectedOperations.insert(op);
        uint32_t opCoreFlag = GetOperationCoreFlag(*op);
        if (inferPostLoweringCore) {
            const uint32_t postLoweringCoreFlag = GetPostLoweringOperationCoreFlag(*op);
            if (postLoweringCoreFlag != 0U) {
                opCoreFlag = postLoweringCoreFlag;
            }
        }
        int anchoredSubgraphId = -1;
        if (inferPostLoweringCore && IsProducerAnchoredPostLoweringMove(*op)) {
            for (const auto* producer : op->GetIOperands().front()->GetProducers()) {
                const auto originalIter = originalSubgraphIds.find(producer);
                const auto assignedIter = assignedSubgraphIds.find(producer);
                if (originalIter == originalSubgraphIds.end() || originalIter->second != oldSubgraphId ||
                    assignedIter == assignedSubgraphIds.end()) {
                    continue;
                }
                if (anchoredSubgraphId >= 0 && anchoredSubgraphId != assignedIter->second) {
                    anchoredSubgraphId = -1;
                    break;
                }
                anchoredSubgraphId = assignedIter->second;
            }
        }
        if (anchoredSubgraphId >= 0 && !HasAicAivConflict(splitCoreFlags[anchoredSubgraphId] | opCoreFlag)) {
            op->UpdateSubgraphID(anchoredSubgraphId);
            assignedSubgraphIds.emplace(op, anchoredSubgraphId);
            splitCoreFlags[anchoredSubgraphId] |= opCoreFlag;
            currentSplitSubgraphId = anchoredSubgraphId;
            currentCoreFlags = splitCoreFlags[anchoredSubgraphId];
            previousOldSubgraphId = oldSubgraphId;
            previousWasSplit = true;
            continue;
        }
        if (!previousWasSplit || previousOldSubgraphId != oldSubgraphId ||
            HasAicAivConflict(currentCoreFlags | opCoreFlag)) {
            currentSplitSubgraphId = nextSubgraphId++;
            currentCoreFlags = 0U;
        }
        currentCoreFlags |= opCoreFlag;
        op->UpdateSubgraphID(currentSplitSubgraphId);
        assignedSubgraphIds.emplace(op, currentSplitSubgraphId);
        splitCoreFlags[currentSplitSubgraphId] |= opCoreFlag;
        previousOldSubgraphId = oldSubgraphId;
        previousWasSplit = true;
    }
    bool consumerAnchorChanged = false;
    if (inferPostLoweringCore) {
        std::vector<std::set<int>> dependencyGraph;
        std::vector<uint32_t> coreFlags;
        bool dependencyGraphDirty = true;
        for (auto iter = sortedOperations.rbegin(); iter != sortedOperations.rend(); ++iter) {
            auto* op = *iter;
            if (!IsConsumerAnchoredPostLoweringMove(*op)) {
                continue;
            }
            const int oldSubgraphId = originalSubgraphIds.at(op);
            int anchoredSubgraphId = -1;
            for (auto* consumer : op->GetOOperands().front()->GetConsumers()) {
                const auto originalIter = originalSubgraphIds.find(consumer);
                const auto assignedIter = assignedSubgraphIds.find(consumer);
                if (originalIter == originalSubgraphIds.end() || originalIter->second != oldSubgraphId ||
                    assignedIter == assignedSubgraphIds.end()) {
                    continue;
                }
                if (anchoredSubgraphId >= 0 && anchoredSubgraphId != assignedIter->second) {
                    anchoredSubgraphId = -1;
                    break;
                }
                anchoredSubgraphId = assignedIter->second;
            }
            const uint32_t opCoreFlag = GetPostLoweringOperationCoreFlag(*op);
            if (anchoredSubgraphId < 0 || anchoredSubgraphId == op->GetSubgraphID() ||
                HasAicAivConflict(splitCoreFlags[anchoredSubgraphId] | opCoreFlag)) {
                continue;
            }
            // Moving a copy past an intervening core segment can recreate a token dependency cycle.
            if (dependencyGraphDirty) {
                dependencyGraph = BuildSubgraphDependencyGraph(function);
                coreFlags = CollectSubgraphCoreFlags(function, nextSubgraphId - 1, true);
                dependencyGraphDirty = false;
            }
            const auto pathClosure = CollectPathClosure(dependencyGraph, op->GetSubgraphID(), anchoredSubgraphId);
            uint32_t pathCoreFlags = opCoreFlag;
            for (int subgraphId : pathClosure) {
                pathCoreFlags |= coreFlags[subgraphId];
            }
            if (HasAicAivConflict(pathCoreFlags)) {
                continue;
            }
            op->UpdateSubgraphID(anchoredSubgraphId);
            assignedSubgraphIds[op] = anchoredSubgraphId;
            splitCoreFlags[anchoredSubgraphId] |= opCoreFlag;
            consumerAnchorChanged = true;
            dependencyGraphDirty = true;
        }
    }
    if (consumerAnchorChanged) {
        SubgraphUnionFind identityUnionFind(static_cast<size_t>(nextSubgraphId));
        UpdateMergedSubgraphIds(function, identityUnionFind);
    } else {
        function.SetTotalSubGraphCount(static_cast<size_t>(nextSubgraphId));
    }
    if (MergeInvalidBoundaryConnectedSubgraphs(function, inferPostLoweringCore) != SUCCESS) {
        return FAILED;
    }
    CoalesceCompatibleAffectedSubgraphs(function, affectedOperations, inferPostLoweringCore);
    UpdateAffectedSubgraphCoreAttributes(function, affectedOperations, inferPostLoweringCore);
    return SUCCESS;
}

Status MergeCyclicSubgraphsOnce(Function& function, bool& changed)
{
    changed = false;
    auto graph = BuildSubgraphDependencyGraph(function);
    if (graph.empty()) {
        return SUCCESS;
    }
    auto sccs = FindStronglyConnectedComponents(graph);
    SubgraphUnionFind unionFind(graph.size());
    std::vector<uint32_t> coreFlagsBySubgraph(graph.size(), 0U);
    for (auto& op : function.Operations()) {
        const int subgraphId = op.GetSubgraphID();
        if (subgraphId >= 0 && subgraphId < static_cast<int>(coreFlagsBySubgraph.size())) {
            coreFlagsBySubgraph[subgraphId] |= GetOperationCoreFlag(op);
        }
    }
    std::unordered_set<int> mixedCoreCyclicSubgraphIds;
    for (const auto& scc : sccs) {
        if (scc.size() <= 1) {
            continue;
        }
        uint32_t coreFlags = 0U;
        for (int subgraphId : scc) {
            coreFlags |= coreFlagsBySubgraph[subgraphId];
        }
        if (HasAicAivConflict(coreFlags)) {
            mixedCoreCyclicSubgraphIds.insert(scc.begin(), scc.end());
            continue;
        }
        changed = true;
        for (size_t i = 1; i < scc.size(); ++i) {
            unionFind.Union(scc[0], scc[i]);
        }
    }
    if (!mixedCoreCyclicSubgraphIds.empty()) {
        // Infer the executing core of cross-memory moves and preserve their computation anchors.
        if (SplitMixedCoreSubgraphs(function, mixedCoreCyclicSubgraphIds, true) != SUCCESS) {
            return FAILED;
        }
        changed = true;
        return SUCCESS;
    }
    if (!changed) {
        return SUCCESS;
    }

    UpdateMergedSubgraphIds(function, unionFind);
    return SUCCESS;
}

Status MergeCyclicSubgraphs(Function& function)
{
    const size_t maxIteration = function.GetOperationSize() + 1;
    for (size_t i = 0; i < maxIteration; ++i) {
        bool changed = false;
        if (MergeCyclicSubgraphsOnce(function, changed) != SUCCESS) {
            return FAILED;
        }
        if (!changed) {
            return SUCCESS;
        }
    }
    APASS_LOG_ERROR_F(Elements::Function, "Merge cyclic subgraphs failed after %zu iterations.", maxIteration);
    return FAILED;
}

using TensorAccesses = std::unordered_map<int, std::unordered_set<const LogicalTensor*>>;

void RecordTensorAccesses(TensorAccesses& accesses, const LogicalTensors& tensors)
{
    for (const auto& tensor : tensors) {
        if (tensor != nullptr && tensor->GetRawTensor() != nullptr) {
            accesses[tensor->GetRawMagic()].insert(tensor.get());
        }
    }
}

std::unordered_map<int, TensorAccesses> CollectSubgraphTensorAccesses(Function& function)
{
    std::unordered_map<int, TensorAccesses> subgraphAccesses;
    for (auto& op : function.Operations()) {
        const int subgraphId = op.GetSubgraphID();
        if (subgraphId < 0) {
            continue;
        }
        auto& accesses = subgraphAccesses[subgraphId];
        RecordTensorAccesses(accesses, op.GetIOperands());
        RecordTensorAccesses(accesses, op.GetOOperands());
        RecordTensorAccesses(accesses, op.GetDependOperands());
    }
    return subgraphAccesses;
}

bool HasAliasedRawTensorAccess(const TensorAccesses& producerAccesses, const TensorAccesses& consumerAccesses)
{
    for (const auto& [rawMagic, producerTensors] : producerAccesses) {
        const auto consumerIter = consumerAccesses.find(rawMagic);
        if (consumerIter == consumerAccesses.end()) {
            continue;
        }
        const auto& consumerTensors = consumerIter->second;
        const bool hasSharedLogicalTensor = std::any_of(
            producerTensors.begin(), producerTensors.end(),
            [&consumerTensors](const auto* tensor) { return consumerTensors.count(tensor) != 0U; });
        if (!hasSharedLogicalTensor) {
            return true;
        }
    }
    return false;
}

Status MergeAliasedTokenConnectedSubgraphs(Function& function)
{
    const int maxSubgraphId = GetMaxSubgraphId(function);
    if (maxSubgraphId < 0) {
        return SUCCESS;
    }

    const auto subgraphAccesses = CollectSubgraphTensorAccesses(function);
    auto subgraphCoreFlags = CollectSubgraphCoreFlags(function, maxSubgraphId);
    SubgraphUnionFind unionFind(static_cast<size_t>(maxSubgraphId + 1));
    auto mergedGraph = BuildMergedSubgraphDependencyGraph(function, unionFind);
    bool changed = false;
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            auto* producerOp = ToOperation(producerStmt);
            if (producerOp == nullptr || producerOp->GetSubgraphID() < 0) {
                continue;
            }
            for (const auto& consumerStmt : entry.consumers) {
                auto* consumerOp = ToOperation(consumerStmt);
                if (consumerOp == nullptr || consumerOp->GetSubgraphID() < 0) {
                    continue;
                }
                const int producerId = producerOp->GetSubgraphID();
                const int consumerId = consumerOp->GetSubgraphID();
                if (producerId == consumerId ||
                    !HasAliasedRawTensorAccess(subgraphAccesses.at(producerId), subgraphAccesses.at(consumerId))) {
                    continue;
                }
                const int producerRoot = unionFind.Find(producerId);
                const int consumerRoot = unionFind.Find(consumerId);
                if (producerRoot == consumerRoot) {
                    continue;
                }
                const auto pathClosure = CollectPathClosure(mergedGraph, producerRoot, consumerRoot);
                if (pathClosure.empty()) {
                    continue;
                }
                uint32_t mergedCoreFlags = 0U;
                for (int subgraphId : pathClosure) {
                    mergedCoreFlags |= subgraphCoreFlags[unionFind.Find(subgraphId)];
                }
                if (HasAicAivConflict(mergedCoreFlags)) {
                    continue;
                }
                for (int subgraphId : pathClosure) {
                    unionFind.Union(producerRoot, subgraphId);
                }
                subgraphCoreFlags[unionFind.Find(producerRoot)] = mergedCoreFlags;
                mergedGraph = BuildMergedSubgraphDependencyGraph(function, unionFind);
                changed = true;
            }
        }
    }
    if (changed) {
        UpdateMergedSubgraphIds(function, unionFind);
    }
    const auto sccs = FindStronglyConnectedComponents(BuildSubgraphDependencyGraph(function));
    const bool hasCycle = std::any_of(sccs.begin(), sccs.end(), [](const auto& scc) { return scc.size() > 1; });
    if (hasCycle) {
        APASS_LOG_ERROR_F(Elements::Function, "Aliased token-connected subgraph merge created a cycle.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace

Status FinalizePartitionWithTokenDependency(Function& function, bool splitPostLoweringMixedCoreSubgraphs,
                                            bool* postLoweringSplitOccurred)
{
    if (postLoweringSplitOccurred != nullptr) {
        *postLoweringSplitOccurred = false;
    }
    if (TokenUtils::RebuildTokenDependencies(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to rebuild token dependencies after graph partition.");
        return FAILED;
    }
    if (splitPostLoweringMixedCoreSubgraphs) {
        const auto mixedSubgraphIds = CollectMixedPostLoweringCoreSubgraphIds(function);
        if (!mixedSubgraphIds.empty() && SplitMixedCoreSubgraphs(function, mixedSubgraphIds, true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Split mixed AIC/AIV subgraphs failed.");
            return FAILED;
        }
        if (!mixedSubgraphIds.empty() && postLoweringSplitOccurred != nullptr) {
            *postLoweringSplitOccurred = true;
        }
    }
    if (MergeCyclicSubgraphs(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Merge cyclic subgraphs failed.");
        return FAILED;
    }
    if (MergeAliasedTokenConnectedSubgraphs(function) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
