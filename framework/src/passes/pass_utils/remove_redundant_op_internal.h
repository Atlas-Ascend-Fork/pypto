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
 * \file remove_redundant_op_internal.h
 * \brief Internal shared helpers for RemoveRedundantOp and RemoveRedundantOpUtils.
 */

#ifndef PASS_REMOVE_REDUNDANT_OP_INTERNAL_H_
#define PASS_REMOVE_REDUNDANT_OP_INTERNAL_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk::remove_redundant_op_internal {

std::vector<Operation*> GetTensorConsumers(const LogicalTensorPtr& tensor);
std::vector<Operation*> GetTensorProducers(const LogicalTensorPtr& tensor);
std::vector<Operation*> CollectViewAssembleCascadeOps(const LogicalTensorPtr& startTensor,
                                                      const LogicalTensorPtr& endTensor);

bool CanMigrateRemovedOpsTokenDependency(Function& function, const std::vector<Operation*>& removedOps,
                                         const std::vector<Operation*>& targetConsumers,
                                         const std::vector<Operation*>& targetProducers);

void MigrateRemovedOpsTokenDependency(Function& function, const std::vector<Operation*>& removedOps,
                                      const std::vector<Operation*>& targetConsumers,
                                      const std::vector<Operation*>& targetProducers);

bool HasOtherAssembleOutputOnSameRaw(Function& function, const LogicalTensorPtr& output,
                                     const Operation* ignoredOp = nullptr);

// 单次子 pass 内需对大量候选判定"是否存在同 raw 的其他 assemble 输出（拆分版本兄弟）"时，
// 先构建本索引再做 O(1) 查询，避免逐候选 O(ops×operands) 全图扫描。索引取子 pass 起始时刻
// 快照：子 pass 内部改写造成的删减不做增量维护（可达场景由其余结构条件排除）。
using AssembleOutputRawIndex = std::unordered_map<int, std::unordered_set<int>>;

AssembleOutputRawIndex CollectAssembleOutputRawIndex(Function& function);

bool HasSiblingAssembleOutputOnRaw(const AssembleOutputRawIndex& rawTensorMagics, const LogicalTensorPtr& tensor);

} // namespace npu::tile_fwk::remove_redundant_op_internal

#endif // PASS_REMOVE_REDUNDANT_OP_INTERNAL_H_
