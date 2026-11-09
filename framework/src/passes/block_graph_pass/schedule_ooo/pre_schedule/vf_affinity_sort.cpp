/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vf_affinity_sort.h"
#include "passes/pass_log/pass_log.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace npu::tile_fwk {

namespace {
const std::unordered_map<Opcode, int>& GetPreNodePriority()
{
    static const std::unordered_map<Opcode, int> preNodePriority = {
        {Opcode::OP_UB_ALLOC, 0},
        {Opcode::OP_L1_ALLOC, 0},
        {Opcode::OP_L0A_ALLOC, 0},
        {Opcode::OP_L0B_ALLOC, 0},
        {Opcode::OP_L0C_ALLOC, 0},
        {Opcode::OP_BT_ALLOC, 0},
        {Opcode::OP_FIX_ALLOC, 0},
        {Opcode::OP_L1_TO_L0A, 1},
        {Opcode::OP_L1_TO_L0B, 1},
        {Opcode::OP_L1_TO_L0_AT, 1},
        {Opcode::OP_L1_TO_L0_BT, 1},
        {Opcode::OP_L1_TO_FIX, 1},
        {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
        {Opcode::OP_L1_TO_FIX_RELU_PRE, 1},
        {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
        {Opcode::OP_L1_TO_FIX_QUANT_POST, 1},
        {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
        {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1},
        {Opcode::OP_L1_TO_BT, 1},
        {Opcode::OP_COPY_IN, 2},
        {Opcode::OP_UB_COPY_IN, 2},
        {Opcode::OP_L1_COPY_IN, 2},
        {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
        {Opcode::OP_L1_COPY_UB, 2},
        {Opcode::OP_L0C_COPY_UB, 2},
        {Opcode::OP_UB_COPY_L1, 2},
    };
    return preNodePriority;
}
} // namespace

bool VfAffinitySort::IsVfOp(Operation* op) const { return op->GetAtomicScopeId() >= VF_CLUSTER_ID_START; }

bool VfAffinitySort::IsFrontNonVfOp(Operation* op)
{
    if (IsVfOp(op)) {
        return false;
    }
    auto it = frontCache_.find(op);
    if (it != frontCache_.end()) {
        return it->second;
    }
    frontCache_[op] = false;
    for (auto* succ : state_.depManager.GetSuccessors(op)) {
        if (opSet_.count(succ) == 0) {
            continue;
        }
        if (IsVfOp(succ) || IsFrontNonVfOp(succ)) {
            frontCache_[op] = true;
            return true;
        }
    }
    return false;
}

bool VfAffinitySort::IsBackNonVfOp(Operation* op)
{
    if (IsVfOp(op)) {
        return false;
    }
    auto it = backCache_.find(op);
    if (it != backCache_.end()) {
        return it->second;
    }
    backCache_[op] = false;
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (opSet_.count(pred) == 0) {
            continue;
        }
        if (IsVfOp(pred) || IsBackNonVfOp(pred)) {
            backCache_[op] = true;
            return true;
        }
    }
    return false;
}

bool VfAffinitySort::HasBackNonVfConsumer(Operation* allocOp)
{
    for (auto* succ : state_.depManager.GetSuccessors(allocOp)) {
        if (!IsVfOp(succ) && !IsFrontNonVfOp(succ) && IsBackNonVfOp(succ)) {
            return true;
        }
    }
    return false;
}

int VfAffinitySort::GetEntryCount(Operation* op)
{
    auto it = entryCountCache_.find(op);
    if (it != entryCountCache_.end()) {
        return it->second;
    }
    entryCountCache_[op] = 1;
    int count = 0;
    bool hasPred = false;
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (opSet_.count(pred) == 0) {
            continue;
        }
        hasPred = true;
        count += GetEntryCount(pred);
    }
    if (!hasPred) {
        count = 1;
    }
    entryCountCache_[op] = count;
    return count;
}

int VfAffinitySort::GetNodePriority(Operation* op) const
{
    const auto& preNodePriority = GetPreNodePriority();
    int prior = 10;
    auto it = preNodePriority.find(op->GetOpcode());
    if (it != preNodePriority.end()) {
        prior = it->second;
    }
    return prior;
}

int64_t VfAffinitySort::GetOutSize(Operation* op) const
{
    int64_t outSize = 0;
    for (auto& oOperand : op->GetOOperands()) {
        if (oOperand != nullptr && oOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR &&
            oOperand->tensor != nullptr) {
            outSize += oOperand->tensor->GetRawDataSize();
        }
    }
    return outSize;
}

int64_t VfAffinitySort::GetBranchNetRelease(Operation* op)
{
    auto it = branchReleaseCache_.find(op);
    if (it != branchReleaseCache_.end()) {
        return it->second;
    }
    // 断环占位: 依赖图若带环, 递归回到未完成的 op 时直接取 0 贡献; 环由上层商图回边检测报出。
    branchReleaseCache_[op] = 0;
    int64_t total = 0;
    if (!IsVfOp(op)) {
        total = GetOutSize(op);
    } else {
        for (auto* pred : state_.depManager.GetPredecessors(op)) {
            if (opSet_.count(pred) > 0) {
                total += GetBranchNetRelease(pred);
            }
        }
    }
    branchReleaseCache_[op] = total;
    return total;
}

std::set<int> VfAffinitySort::GetFrontScopeSet(Operation* op)
{
    auto it = frontScopeSetCache_.find(op);
    if (it != frontScopeSetCache_.end()) {
        return it->second;
    }
    // 断环占位: 依赖图若带环, 正在访问的 op 按空集合参与, 环由上层商图回边检测报出。
    if (frontScopeVisiting_.count(op) > 0) {
        return {};
    }
    frontScopeVisiting_.insert(op);
    std::set<int> scopes;
    for (auto* succ : state_.depManager.GetSuccessors(op)) {
        if (opSet_.count(succ) == 0) {
            continue;
        }
        if (IsVfOp(succ)) {
            scopes.insert(succ->GetAtomicScopeId());
        } else {
            for (auto scopeId : GetFrontScopeSet(succ)) {
                scopes.insert(scopeId);
            }
        }
    }
    frontScopeVisiting_.erase(op);
    frontScopeSetCache_[op] = scopes;
    return scopes;
}

void VfAffinitySort::ReverseDfs(Operation* op, std::unordered_set<Operation*>& visited, std::vector<Operation*>& result)
{
    if (visited.count(op) > 0) {
        return;
    }
    visited.insert(op);

    std::vector<Operation*> preds;
    for (auto* pred : state_.depManager.GetPredecessors(op)) {
        if (opSet_.count(pred) > 0) {
            preds.push_back(pred);
        }
    }
    std::sort(preds.begin(), preds.end(), [this](Operation* a, Operation* b) {
        int priorA = GetNodePriority(a);
        int priorB = GetNodePriority(b);
        if (priorA != priorB) {
            return priorA > priorB;
        }
        int entryA = GetEntryCount(a);
        int entryB = GetEntryCount(b);
        if (entryA != entryB) {
            return entryA > entryB;
        }
        int64_t releaseA = GetBranchNetRelease(a);
        int64_t releaseB = GetBranchNetRelease(b);
        if (releaseA != releaseB) {
            return releaseA > releaseB;
        }
        return a->GetOpMagic() < b->GetOpMagic();
    });

    for (auto* pred : preds) {
        ReverseDfs(pred, visited, result);
    }

    result.push_back(op);
}

Status VfAffinitySort::PostProcess(std::vector<Operation*>& result)
{
    std::vector<Operation*> front;
    std::vector<Operation*> mid;
    std::vector<Operation*> back;
    for (auto* op : result) {
        if (IsVfOp(op)) {
            mid.push_back(op);
        } else if (IsAllocOpCode(op->GetOpcode())) {
            if (HasBackNonVfConsumer(op)) {
                back.push_back(op);
            } else {
                front.push_back(op);
            }
        } else if (IsFrontNonVfOp(op)) {
            front.push_back(op);
        } else if (IsBackNonVfOp(op)) {
            back.push_back(op);
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "VfAffinitySort: non-vf op %s is neither front nor back.",
                              state_.GetOpInfo(op).c_str());
            return FAILED;
        }
    }

    result.clear();
    result.insert(result.end(), front.begin(), front.end());
    result.insert(result.end(), mid.begin(), mid.end());
    result.insert(result.end(), back.begin(), back.end());
    return SUCCESS;
}

Status VfAffinitySort::SortOneSuperNode(const std::vector<Operation*>& ops, std::vector<Operation*>& out)
{
    std::unordered_set<Operation*> subset(ops.begin(), ops.end());
    std::unordered_set<Operation*> savedOpSet = std::move(opSet_);
    opSet_ = subset;
    frontCache_.clear();
    backCache_.clear();
    entryCountCache_.clear();
    branchReleaseCache_.clear();

    std::vector<Operation*> exits;
    for (auto* op : ops) {
        bool hasSucc = false;
        for (auto* succ : state_.depManager.GetSuccessors(op)) {
            if (subset.count(succ) > 0) {
                hasSucc = true;
                break;
            }
        }
        if (!hasSucc) {
            exits.push_back(op);
        }
    }

    std::unordered_set<Operation*> visited;
    std::vector<Operation*> subResult;
    subResult.reserve(ops.size());
    for (auto* op : exits) {
        ReverseDfs(op, visited, subResult);
    }

    Status st = SUCCESS;
    if (subResult.size() != ops.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "VfAffinitySort: super-node ordered %zu vs %zu ops (cycle?).",
                          subResult.size(), ops.size());
        st = FAILED;
    } else if (PostProcess(subResult) != SUCCESS) {
        st = FAILED;
    } else {
        out.insert(out.end(), subResult.begin(), subResult.end());
    }

    opSet_ = std::move(savedOpSet);
    return st;
}

Status VfAffinitySort::SortScope(std::vector<Operation*>& result)
{
    // 1. 每个 op 归属一个超节点: scope op → atomicScopeId;
    OpToSuperNodeMap opToSuperNode;
    SuperNodeToOpsMap superNodeToOps;
    AssignSuperNodes(opToSuperNode, superNodeToOps);

    // 2. 构建超节点粗粒度图
    std::map<int, int> snToIdx;
    std::vector<int> snIds;
    std::vector<std::set<int>> inGraph;
    std::vector<std::set<int>> outGraph;
    SuperNodeWitness quotientWitness;
    BuildSuperNodeGraph(opToSuperNode, superNodeToOps, snToIdx, snIds, inGraph, outGraph, quotientWitness);

    // 3. 块间排序: 商图反向 DFS, 与 fine 层同构——出口超节点反向递归、后序 append;
    //    fan-in 点按 优先级/entryCount/branchNetRelease/snId 镜像 fine 层排序前驱分支;
    //    fan-out 不排序, 保持 DFS 自然序; 在栈节点重入(回边)即报环错, visited 计数兜底断开 sink 的环。
    std::vector<int> topo;
    if (SortSuperNodeTopo(superNodeToOps, snIds, inGraph, outGraph, quotientWitness, topo) != SUCCESS) {
        return FAILED;
    }

    // 4. 逐超节点排序并拼接
    return AppendSortedSuperNodes(topo, snIds, superNodeToOps, result);
}

void VfAffinitySort::AssignSuperNodes(OpToSuperNodeMap& opToSuperNode, SuperNodeToOpsMap& superNodeToOps)
{
    for (auto* op : operations) {
        if (IsVfOp(op)) {
            opToSuperNode[op] = op->GetAtomicScopeId();
        }
    }
    int nextSingletonId = -1;
    for (auto* op : operations) {
        if (IsVfOp(op)) {
            continue;
        }
        int scopeId = 0;
        auto frontScopes = GetFrontScopeSet(op);
        if (frontScopes.size() == 1) {
            scopeId = *frontScopes.begin();
        }
        if (scopeId >= VF_CLUSTER_ID_START) {
            opToSuperNode[op] = scopeId;
        } else {
            opToSuperNode[op] = nextSingletonId--;
        }
    }
    for (auto* op : operations) {
        superNodeToOps[opToSuperNode[op]].push_back(op);
    }
}

void VfAffinitySort::BuildSuperNodeGraph(const OpToSuperNodeMap& opToSuperNode, const SuperNodeToOpsMap& superNodeToOps,
                                         std::map<int, int>& snToIdx, std::vector<int>& snIds,
                                         std::vector<std::set<int>>& inGraph, std::vector<std::set<int>>& outGraph,
                                         SuperNodeWitness& quotientWitness)
{
    for (auto& [sn, ops] : superNodeToOps) {
        (void)ops;
        snToIdx[sn] = static_cast<int>(snIds.size());
        snIds.push_back(sn);
    }
    int n = static_cast<int>(snIds.size());
    inGraph.resize(n);
    outGraph.resize(n);
    for (auto* op : operations) {
        int srcIdx = snToIdx.at(opToSuperNode.at(op));
        for (auto* succ : state_.depManager.GetSuccessors(op)) {
            if (opSet_.count(succ) == 0 || IsAllocOpCode(succ->GetOpcode())) {
                continue;
            }
            int dstIdx = snToIdx.at(opToSuperNode.at(succ));
            if (srcIdx != dstIdx) {
                if (outGraph[srcIdx].insert(dstIdx).second) {
                    quotientWitness[{srcIdx, dstIdx}] = {op, succ};
                }
                inGraph[dstIdx].insert(srcIdx);
            }
        }
    }
}

Status VfAffinitySort::SortSuperNodeTopo(const SuperNodeToOpsMap& superNodeToOps, const std::vector<int>& snIds,
                                         const std::vector<std::set<int>>& inGraph,
                                         const std::vector<std::set<int>>& outGraph,
                                         const SuperNodeWitness& quotientWitness, std::vector<int>& topo)
{
    int n = static_cast<int>(snIds.size());
    topo.reserve(n);
    // 超节点优先级: 单 op 超节点查 preNodePriority, 多 op 超节点默认 10 (镜像 fine 层)。
    auto getSnPriority = [&](int idx) -> int {
        const auto& ops = superNodeToOps.at(snIds[idx]);
        if (ops.size() == 1) {
            return GetNodePriority(ops[0]);
        }
        return 10;
    };
    // fan-in 分支度量取商边 witness 的 src op, 复用 op 级缓存 (语义与 fine 层 5.3 一致)。
    auto getEdgeEntryCount = [&](int srcIdx, int dstIdx) -> int {
        auto wit = quotientWitness.find({srcIdx, dstIdx});
        if (wit == quotientWitness.end()) {
            return 0;
        }
        return GetEntryCount(wit->second.first);
    };
    auto getEdgeRelease = [&](int srcIdx, int dstIdx) -> int64_t {
        auto wit = quotientWitness.find({srcIdx, dstIdx});
        if (wit == quotientWitness.end()) {
            return 0;
        }
        return GetBranchNetRelease(wit->second.first);
    };
    // 环检测: 两态标记(0 未访问, 1 在栈, 2 已完成), pred 仍在栈上即回边→环, 直接 FAILED。
    std::vector<int> color(n, 0);
    std::function<Status(int)> dfs = [&](int u) -> Status {
        if (color[u] == 2) {
            return SUCCESS;
        }
        if (color[u] == 1) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "VfAffinitySort: super-node graph has cycle (back edge revisits %d).", u);
            return FAILED;
        }
        color[u] = 1;
        std::vector<int> preds(inGraph[u].begin(), inGraph[u].end());
        std::sort(preds.begin(), preds.end(), [&](int a, int b) {
            int priA = getSnPriority(a);
            int priB = getSnPriority(b);
            if (priA != priB) {
                return priA > priB;
            }
            int entryA = getEdgeEntryCount(a, u);
            int entryB = getEdgeEntryCount(b, u);
            if (entryA != entryB) {
                return entryA > entryB;
            }
            int64_t releaseA = getEdgeRelease(a, u);
            int64_t releaseB = getEdgeRelease(b, u);
            if (releaseA != releaseB) {
                return releaseA > releaseB;
            }
            return a < b;
        });
        for (int p : preds) {
            if (dfs(p) != SUCCESS) {
                return FAILED;
            }
        }
        color[u] = 2;
        topo.push_back(u);
        return SUCCESS;
    };
    for (int i = 0; i < n; i++) {
        if (outGraph[i].empty()) {
            if (dfs(i) != SUCCESS) {
                return FAILED;
            }
        }
    }
    // visited 计数兜底: 与所有 sink 断开的环不会被任何 dfs 触达, 计数必然不足。
    if (static_cast<int>(topo.size()) != n) {
        APASS_LOG_ERROR_F(Elements::Operation, "VfAffinitySort: super-node graph has cycle (%d vs %d).",
                          static_cast<int>(topo.size()), n);
        return FAILED;
    }
    return SUCCESS;
}

Status VfAffinitySort::AppendSortedSuperNodes(const std::vector<int>& topo, const std::vector<int>& snIds,
                                              const SuperNodeToOpsMap& superNodeToOps, std::vector<Operation*>& result)
{
    for (int idx : topo) {
        int sn = snIds[idx];
        const auto& ops = superNodeToOps.at(sn);
        if (sn < 0 && ops.size() == 1 && !IsVfOp(ops[0])) {
            result.push_back(ops[0]);
            continue;
        }
        if (SortOneSuperNode(ops, result) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

Status VfAffinitySort::DoSortOps()
{
    opSet_.clear();
    vfOpSet_.clear();
    entryCountCache_.clear();
    branchReleaseCache_.clear();
    frontCache_.clear();
    backCache_.clear();
    frontScopeSetCache_.clear();
    frontScopeVisiting_.clear();
    for (auto* op : operations) {
        opSet_.insert(op);
        if (IsVfOp(op)) {
            vfOpSet_.insert(op);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "VfAffinitySort: %zu ops, %zu vf ops.", operations.size(), vfOpSet_.size());

    // 单 scope 同样走 SortScope: 全部 op 落进唯一 scope 超节点, 输出与旧单 scope 路径一致
    // (同一 ReverseDfs + PostProcess 于同一 op 集合); 孤立 op 由报错放宽为单例。
    std::vector<Operation*> result;
    result.reserve(operations.size());
    if (SortScope(result) != SUCCESS) {
        return FAILED;
    }

    operations = std::move(result);
    APASS_LOG_INFO_F(Elements::Operation, "VfAffinitySort: sorted %zu ops.", operations.size());

    if (ExecuteOp() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ExecuteOp failed.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
