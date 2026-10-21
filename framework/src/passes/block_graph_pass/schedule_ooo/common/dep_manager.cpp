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
 * \file dep_manager.cpp
 * \brief Dependency manager implementation
 */

#include "passes/block_graph_pass/schedule_ooo/common/dep_manager.h"
#include <unordered_set>
#include "passes/pass_log/pass_log.h"

#ifndef MODULE_NAME
#define MODULE_NAME "DependencyManager"
#endif

namespace npu::tile_fwk {

void DependencyManager::RegisterOp(Operation* op)
{
    if (op == nullptr) {
        return;
    }
    if (inGraph_.find(op) == inGraph_.end()) {
        inGraph_[op] = std::set<Operation*, Operation::OperationComparator>();
    }
    if (outGraph_.find(op) == outGraph_.end()) {
        outGraph_[op] = std::set<Operation*, Operation::OperationComparator>();
    }
}

void DependencyManager::Clear()
{
    inGraph_.clear();
    outGraph_.clear();
}

void DependencyManager::ClearDependencies()
{
    for (auto& [op, preds] : inGraph_) {
        (void)op;
        preds.clear();
    }
    for (auto& [op, succs] : outGraph_) {
        (void)op;
        succs.clear();
    }
}

bool DependencyManager::IsOpAlloc(Operation* op)
{
    if (op == nullptr) {
        return false;
    }
    return op->GetOpcodeStr().find("ALLOC") != std::string::npos;
}

void DependencyManager::AddDependency(Operation* preOp, Operation* postOp)
{
    if (preOp == nullptr || postOp == nullptr) {
        return;
    }
    if (!IsOpAlloc(preOp) && !IsOpAlloc(postOp)) {
        outGraph_[preOp].insert(postOp);
        inGraph_[postOp].insert(preOp);
    }
}

void DependencyManager::AddAllocDependency(Operation* preOp, Operation* postOp)
{
    if (preOp == nullptr || postOp == nullptr) {
        return;
    }
    outGraph_[preOp].insert(postOp);
    inGraph_[postOp].insert(preOp);
}

bool DependencyManager::RemoveDependency(Operation* preOp, Operation* postOp)
{
    if (preOp == nullptr || postOp == nullptr) {
        return false;
    }
    bool removedFromSucc = false;
    bool removedFromPred = false;

    if (outGraph_.find(preOp) != outGraph_.end()) {
        removedFromSucc = outGraph_[preOp].erase(postOp) > 0;
    }
    if (inGraph_.find(postOp) != inGraph_.end()) {
        removedFromPred = inGraph_[postOp].erase(preOp) > 0;
    }

    return removedFromSucc && removedFromPred;
}

int DependencyManager::InsertSuccessor(Operation* op, Operation* succ)
{
    if (op == nullptr || succ == nullptr) {
        return 0;
    }
    auto result = outGraph_[op].insert(succ);
    return result.second ? 1 : 0;
}

int DependencyManager::RemoveSuccessor(Operation* op, Operation* succ)
{
    if (op == nullptr || succ == nullptr) {
        return 0;
    }
    return outGraph_[op].erase(succ);
}

void DependencyManager::RemoveOp(Operation* op)
{
    if (op == nullptr) {
        return;
    }
    auto inIt = inGraph_.find(op);
    if (inIt != inGraph_.end()) {
        for (auto* pred : inIt->second) {
            auto predIt = outGraph_.find(pred);
            if (predIt != outGraph_.end()) {
                predIt->second.erase(op);
            }
        }
    }
    auto outIt = outGraph_.find(op);
    if (outIt != outGraph_.end()) {
        for (auto* succ : outIt->second) {
            auto succIt = inGraph_.find(succ);
            if (succIt != inGraph_.end()) {
                succIt->second.erase(op);
            }
        }
    }
    inGraph_.erase(op);
    outGraph_.erase(op);
    opConsumers.erase(op);
    opProducers.erase(op);
}

int DependencyManager::InsertPredecessor(Operation* op, Operation* pred)
{
    if (op == nullptr || pred == nullptr) {
        return 0;
    }
    auto result = inGraph_[op].insert(pred);
    return result.second ? 1 : 0;
}

int DependencyManager::RemovePredecessor(Operation* op, Operation* pred)
{
    if (op == nullptr || pred == nullptr) {
        return 0;
    }
    return inGraph_[op].erase(pred);
}

std::set<Operation*, Operation::OperationComparator>& DependencyManager::GetSuccessors(Operation* op)
{
    return outGraph_[op];
}

std::set<Operation*, Operation::OperationComparator>& DependencyManager::GetPredecessors(Operation* op)
{
    return inGraph_[op];
}

bool DependencyManager::HasOp(Operation* op) const { return inGraph_.find(op) != inGraph_.end(); }

std::string DependencyManager::PrintOp(Operation* op)
{
    return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
}

void DependencyManager::PrintDependencies(const std::vector<Operation*>& ops)
{
    for (const auto& op : ops) {
        if (inGraph_.find(op) == inGraph_.end() || outGraph_.find(op) == outGraph_.end()) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s", PrintOp(op).c_str());
        for (const auto& preOp : inGraph_.at(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Predecessors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", PrintOp(preOp).c_str());
        }
        for (const auto& succOp : outGraph_.at(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Successors:");
            APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", PrintOp(succOp).c_str());
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "\n");
    }
}

Status DependencyManager::InitAllocDependencies(Operation* op, std::unordered_map<int, Operation*>& tensor2AllocOpMap)
{
    for (auto& tensor : op->GetOOperands()) {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            auto it = tensor2AllocOpMap.find(memId);
            if (it == tensor2AllocOpMap.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc. magic: %d, op: %s", memId,
                                  tensor->GetMagic(), PrintOp(op).c_str());
                return FAILED;
            }
            AddAllocDependency(it->second, op);
        }
    }
    return SUCCESS;
}

void DependencyManager::HandleScaleOpDependency(Operation* op, MemoryType memType)
{
    auto matmulOp = *(op->GetOutputOperand(0))->GetConsumers().begin();
    if (matmulOp == nullptr) {
        return;
    }
    for (auto& input : matmulOp->GetIOperands()) {
        if (input->GetMemoryTypeOriginal() == memType) {
            AddDependency(*input->GetProducers().begin(), op);
        }
    }
}

Operation* UniqueSkipProducer(Operation* op)
{
    Operation* skipProducer = nullptr;
    size_t skipCnt = 0;
    for (auto* producer : op->GetInputOperand(0)->GetProducers()) {
        if (producer != nullptr && IsSkipOp(*producer)) {
            skipProducer = producer;
            ++skipCnt;
        }
    }
    // 截断而不终止流程: 该形状实测 0 例, 调用方按链到此为止继续。
    if (skipCnt > 1) {
        APASS_LOG_WARN_F(Elements::Operation, "%s[%d] input has %zu skip producers, skip chain is truncated.",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), skipCnt);
        return nullptr;
    }
    return skipProducer;
}

// skip op 永不 retire, 挂上去就是永不满足的边。岔口写者与链尾同内存, 都要等。
// ProducerOps() 的 op 按 BelongTo 过滤且已 RegisterOp, 必在图内;
// 穿过 skip op 上溯到的 op 可能在调度集合外, 需 HasOp 判断。
void DependencyManager::AddProducerDependencies(Operation* op)
{
    for (auto* producer : op->ProducerOps()) {
        const auto path = SkipChainPath(producer);
        if (path.empty()) {
            AddDependency(producer, op);
            continue;
        }
        for (auto* link : path) {
            for (auto* realProducer : link->ProducerOps()) {
                if (!IsSkipOp(*realProducer) && HasOp(realProducer)) {
                    AddDependency(realProducer, op);
                }
            }
        }
    }
}

void DependencyManager::FindDependencies(Operation* op, bool needView)
{
    if (op->GetOpcode() == Opcode::OP_L1_TO_L0A_SCALE) {
        HandleScaleOpDependency(op, MemoryType::MEM_L0A);
    }
    if (op->GetOpcode() == Opcode::OP_L1_TO_L0B_SCALE) {
        HandleScaleOpDependency(op, MemoryType::MEM_L0B);
    }

    for (auto* producer : op->ProducerOpsByToken()) {
        if (producer != op && producer->BelongTo() == op->BelongTo() && HasOp(producer)) {
            AddDependency(producer, op);
        }
    }

    if (needView) {
        for (auto& producer : opProducers[op]) {
            AddDependency(producer, op);
        }
        for (auto& consumer : opConsumers[op]) {
            AddDependency(op, consumer);
        }
        return;
    }

    AddProducerDependencies(op);
}

void DependencyManager::InitOpConsumerAndProducer(const std::vector<Operation*>& ops, bool needView)
{
    opConsumers.clear();
    opProducers.clear();
    if (!needView) {
        return;
    }
    std::unordered_set<Operation*> opSet;
    for (auto op : ops) {
        opSet.insert(op);
    }
    for (auto op : ops) {
        for (auto* consumer : op->ConsumerOps()) {
            if (opSet.find(consumer) != opSet.end()) {
                opConsumers[op].insert(consumer);
            }
        }
        for (auto* producer : op->ProducerOps()) {
            if (opSet.find(producer) != opSet.end()) {
                opProducers[op].insert(producer);
            }
        }
    }
}

std::vector<Operation*> DependencyManager::ExpandDirtySet(const std::vector<Operation*>& seeds) const
{
    std::unordered_set<Operation*> dirty;
    for (auto* seed : seeds) {
        if (seed == nullptr || seed->IsDeleted()) {
            continue;
        }
        dirty.insert(seed);
        auto inIt = inGraph_.find(seed);
        if (inIt != inGraph_.end()) {
            dirty.insert(inIt->second.begin(), inIt->second.end());
        }
        auto outIt = outGraph_.find(seed);
        if (outIt != outGraph_.end()) {
            dirty.insert(outIt->second.begin(), outIt->second.end());
        }
        for (auto* succ : seed->ConsumerOpsByToken()) {
            if (succ != seed && succ->BelongTo() == seed->BelongTo()) {
                dirty.insert(succ);
            }
        }
        std::vector<Operation*> pending{seed};
        std::unordered_set<Operation*> walked{seed};
        while (!pending.empty()) {
            Operation* cur = pending.back();
            pending.pop_back();
            for (auto* consumer : cur->ConsumerOps()) {
                dirty.insert(consumer);
                if (IsSkipOp(*consumer) && walked.insert(consumer).second) {
                    pending.push_back(consumer);
                }
            }
        }
    }
    std::vector<Operation*> result;
    result.reserve(dirty.size());
    for (auto* op : dirty) {
        if (op != nullptr && !op->IsDeleted() && inGraph_.find(op) != inGraph_.end()) {
            result.push_back(op);
        }
    }
    return result;
}

void DependencyManager::ClearInEdges(Operation* op)
{
    auto inIt = inGraph_.find(op);
    if (inIt == inGraph_.end()) {
        return;
    }
    for (auto* pred : inIt->second) {
        auto predIt = outGraph_.find(pred);
        if (predIt != outGraph_.end()) {
            predIt->second.erase(op);
        }
    }
    inIt->second.clear();
}

Status DependencyManager::RefreshDependencies(const std::vector<Operation*>& seeds,
                                              const std::unordered_map<int, Operation*>& allocMap)
{
    const auto dirty = ExpandDirtySet(seeds);
    for (auto* op : dirty) {
        ClearInEdges(op);
    }
    for (auto* op : dirty) {
        if (IsOpAlloc(op)) {
            continue;
        }
        FindDependencies(op, false);
        for (auto& tensor : op->GetOOperands()) {
            if (tensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            auto it = allocMap.find(tensor->memoryrange.memId);
            if (it == allocMap.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc. magic: %d, op: %s",
                                  tensor->memoryrange.memId, tensor->GetMagic(), PrintOp(op).c_str());
                return FAILED;
            }
            AddAllocDependency(it->second, op);
        }
    }
    return SUCCESS;
}

Status DependencyManager::InitDependencies(const std::vector<Operation*>& ops, bool needView)
{
    std::unordered_map<int, Operation*> tensor2AllocOpMap;
    InitOpConsumerAndProducer(ops, needView);
    for (const auto& op : ops) {
        if (IsOpAlloc(op)) {
            if (op->GetOOperands().size() != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] oOperand must be 1.", op->GetOpMagic());
                return FAILED;
            }
            int memId = op->GetOutputOperand(0)->memoryrange.memId;
            tensor2AllocOpMap[memId] = op;
        }
    }

    Clear();
    for (const auto& op : ops) {
        RegisterOp(op);
    }

    for (const auto& op : ops) {
        if (!IsOpAlloc(op)) {
            FindDependencies(op, needView);
            if (InitAllocDependencies(op, tensor2AllocOpMap) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "InitAllocDependencies failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
