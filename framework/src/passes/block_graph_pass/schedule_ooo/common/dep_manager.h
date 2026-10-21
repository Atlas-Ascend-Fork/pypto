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
 * \file dep_manager.h
 * \brief Dependency manager for operation scheduling
 */

#ifndef PASS_DEP_MANAGER_H_
#define PASS_DEP_MANAGER_H_

#include <unordered_map>
#include <set>
#include <vector>
#include <functional>
#include <string>
#include "interface/operation/operation.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

inline bool IsSkipOp(const Operation& op)
{
    const auto opc = op.GetOpcode();
    return opc == Opcode::OP_VIEW || opc == Opcode::OP_VIEW_TYPE || opc == Opcode::OP_RESHAPE;
}

// inplace 别名写下生产者不唯一, 只跟 skip 那个。
// 前置条件: skip op 单进单出, 只取 operand 0; 新增 skip opcode 须先满足, 否则静默漏其余 operand。
// 多于一个 skip 生产者时截断, 截断处更深的真写入者收不到。
Operation* UniqueSkipProducer(Operation* op);

// from 在最前, 最远的 skip op 在最后; from 不是 skip op 时为空。跨函数处停止。
inline std::vector<Operation*> SkipChainPath(Operation* from)
{
    std::vector<Operation*> path;
    if (from == nullptr) {
        return path;
    }
    for (Operation* op = from; op != nullptr && IsSkipOp(*op) && op->BelongTo() == from->BelongTo();) {
        path.push_back(op);
        op = UniqueSkipProducer(op);
    }
    return path;
}

inline Operation* SkipChain(Operation* from)
{
    const auto path = SkipChainPath(from);
    return path.empty() ? nullptr : path.back();
}

class DependencyManager {
public:
    void RegisterOp(Operation* op);

    void ClearDependencies();

    static bool IsOpAlloc(Operation* op);

    void AddDependency(Operation* preOp, Operation* postOp);

    void AddAllocDependency(Operation* preOp, Operation* postOp);

    Status InitAllocDependencies(Operation* op, std::unordered_map<int, Operation*>& tensor2AllocOpMap);

    bool RemoveDependency(Operation* preOp, Operation* postOp);

    int InsertSuccessor(Operation* op, Operation* succ);
    int RemoveSuccessor(Operation* op, Operation* succ);
    int InsertPredecessor(Operation* op, Operation* pred);
    int RemovePredecessor(Operation* op, Operation* pred);

    void RemoveOp(Operation* op);

    std::set<Operation*, Operation::OperationComparator>& GetSuccessors(Operation* op);
    std::set<Operation*, Operation::OperationComparator>& GetPredecessors(Operation* op);
    bool HasOp(Operation* op) const;

    std::string PrintOp(Operation* op);

    void FindDependencies(Operation* op, bool needView);
    void InitOpConsumerAndProducer(const std::vector<Operation*>& ops, bool needView);

    Status InitDependencies(const std::vector<Operation*>& ops, bool needView);

    Status RefreshDependencies(const std::vector<Operation*>& seeds,
                               const std::unordered_map<int, Operation*>& allocMap);

    void PrintDependencies(const std::vector<Operation*>& ops);

private:
    void Clear();

    void HandleScaleOpDependency(Operation* op, MemoryType memType);
    void AddProducerDependencies(Operation* op);

    std::vector<Operation*> ExpandDirtySet(const std::vector<Operation*>& seeds) const;
    void ClearInEdges(Operation* op);

    std::unordered_map<Operation*, std::set<Operation*, Operation::OperationComparator>> opConsumers;
    std::unordered_map<Operation*, std::set<Operation*, Operation::OperationComparator>> opProducers;
    std::unordered_map<Operation*, std::set<Operation*, Operation::OperationComparator>> inGraph_;
    std::unordered_map<Operation*, std::set<Operation*, Operation::OperationComparator>> outGraph_;
};

} // namespace npu::tile_fwk

#endif
