/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PASS_VF_AFFINITY_SORT_H
#define PASS_VF_AFFINITY_SORT_H

#include "optimize_sort.h"

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace npu::tile_fwk {

class VfAffinitySort : public OptimizeSort {
public:
    using OptimizeSort::OptimizeSort;

private:
    using OpToSuperNodeMap = std::unordered_map<Operation*, int>;
    using SuperNodeToOpsMap = std::map<int, std::vector<Operation*>>;
    using SuperNodeWitness = std::map<std::pair<int, int>, std::pair<Operation*, Operation*>>;

    Status DoSortOps() override;
    bool IsVfOp(Operation* op) const;
    bool IsFrontNonVfOp(Operation* op);
    bool IsBackNonVfOp(Operation* op);
    bool HasBackNonVfConsumer(Operation* allocOp);
    int GetEntryCount(Operation* op);
    int GetNodePriority(Operation* op) const;
    int64_t GetOutSize(Operation* op) const;
    int64_t GetBranchNetRelease(Operation* op);
    std::set<int> GetFrontScopeSet(Operation* op);
    Status SortScope(std::vector<Operation*>& result);
    void AssignSuperNodes(OpToSuperNodeMap& opToSuperNode, SuperNodeToOpsMap& superNodeToOps);
    void BuildSuperNodeGraph(const OpToSuperNodeMap& opToSuperNode, const SuperNodeToOpsMap& superNodeToOps,
                             std::map<int, int>& snToIdx, std::vector<int>& snIds, std::vector<std::set<int>>& inGraph,
                             std::vector<std::set<int>>& outGraph, SuperNodeWitness& quotientWitness);
    Status SortSuperNodeTopo(const SuperNodeToOpsMap& superNodeToOps, const std::vector<int>& snIds,
                             const std::vector<std::set<int>>& inGraph, const std::vector<std::set<int>>& outGraph,
                             const SuperNodeWitness& quotientWitness, std::vector<int>& topo);
    Status AppendSortedSuperNodes(const std::vector<int>& topo, const std::vector<int>& snIds,
                                  const SuperNodeToOpsMap& superNodeToOps, std::vector<Operation*>& result);
    Status SortOneSuperNode(const std::vector<Operation*>& ops, std::vector<Operation*>& out);
    void ReverseDfs(Operation* op, std::unordered_set<Operation*>& visited, std::vector<Operation*>& result);
    Status PostProcess(std::vector<Operation*>& result);

    std::unordered_set<Operation*> opSet_;
    std::unordered_set<Operation*> vfOpSet_;
    std::unordered_map<Operation*, int> entryCountCache_;
    std::unordered_map<Operation*, int64_t> branchReleaseCache_;
    std::unordered_map<Operation*, std::set<int>> frontScopeSetCache_;
    std::unordered_map<Operation*, std::set<int>> backScopeSetCache_;
    std::unordered_set<Operation*> frontScopeVisiting_;
    std::unordered_set<Operation*> backScopeVisiting_;
    std::unordered_map<Operation*, bool> frontCache_;
    std::unordered_map<Operation*, bool> backCache_;
};

} // namespace npu::tile_fwk
#endif // PASS_VF_AFFINITY_SORT_H
