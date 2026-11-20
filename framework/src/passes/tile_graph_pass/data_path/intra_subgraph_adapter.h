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
 * \file intra_subgraph_adapter.h
 * \brief
 */

#ifndef PASS_INTRA_SUBGRAPH_ADAPTER_H_
#define PASS_INTRA_SUBGRAPH_ADAPTER_H_

#include <unordered_set>
#include <vector>
#include "interface/operation/opcode.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
// InsertOpBetween 的 newOpSubgraphID 传该值表示继承 ops[0] 的 subgraph id
constexpr int kInheritOpsSubgraphId = -1;

class IntraSubgraphAdapter : public Pass {
public:
    IntraSubgraphAdapter() : Pass("IntraSubgraphAdapter") {}
    ~IntraSubgraphAdapter() override = default;
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;

private:
    Status AdaptBoundaryTensors(Function& function);
    Status AdaptSingleBoundaryTensor(Function& function, const LogicalTensorPtr& tensor);
    Status AdaptSharedColorBoundaryTensor(Function& function, const LogicalTensorPtr& tensor,
                                          const std::set<int>& commonColors, const std::set<int>& producerColors,
                                          const std::set<int>& consumerColors);
    void LogSharedColorBoundaryTensor(Function& function, const LogicalTensorPtr& tensor, int mainSubgraphID,
                                      const std::set<int>& producerColors, const std::set<int>& consumerColors);
    Status CheckColorCount(Function& function, LogicalTensorPtr tensor, size_t commonColorCount);
    Status CheckBoundaryTensor(LogicalTensorPtr tensor);
    Status SplitBoundaryTensor(Function& function, LogicalTensorPtr tensor, int mainSubgraphID,
                               LogicalTensors& newBoundaryTensors);
    Status SplitProducerToMainSubgraph(Function& function, const LogicalTensorPtr& tensor, Operation* producer,
                                       int mainSubgraphID, LogicalTensors& newBoundaryTensors);
    Status InsertContractBeforeSubsidiaryConsumers(Function& function, const LogicalTensorPtr& tensor,
                                                   int mainSubgraphID, LogicalTensors& newBoundaryTensors);
    LogicalTensors CollectBoundaryTensors(Function& function);
    Status ProcessBoundaryTensors(Function& function, LogicalTensors tensors);
    Status ProcessBoundaryTensor(Function& function, LogicalTensorPtr tensor);
    Status AdapteTensorProducers(Function& function, LogicalTensorPtr tensor);
    Status AdaptMultiProducersToDdr(const LogicalTensorPtr& tensor);
    Status AdaptSingleProducerToDdr(Function& function, const LogicalTensorPtr& tensor);
    void RewriteCrossCoreProducerToCopyOut(Operation* producer, const LogicalTensorPtr& tensor);
    Status AdapteTensorConsumers(Function& function, LogicalTensorPtr tensor);
    LogicalTensorPtr InsertOpBetween(Function& function, Opcode opcode, Operation& op, LogicalTensorPtr tensor);
    LogicalTensorPtr InsertOpBetween(Function& function, Opcode opcode, LogicalTensorPtr tensor,
                                     const std::vector<Operation*>& ops, int newOpSubgraphID = kInheritOpsSubgraphId);
    LogicalTensorPtr CreateInsertedOpTensor(const LogicalTensorPtr& tensor);
    Operation* CreateInsertedOp(Function& function, Opcode opcode, const LogicalTensorPtr& tensor,
                                const std::vector<Operation*>& ops, int newOpSubgraphID,
                                const LogicalTensorPtr& newTensor);
    void CollectProducerColors(LogicalTensorPtr tensor, std::set<int>& colors);
    void CollectConsumerColors(LogicalTensorPtr tensor, std::set<int>& colors);
    std::set<int> SetIntersection(std::set<int>& a, std::set<int>& b);
    bool IsCrossCoreMoveOps(Operation* op);

    // Delay sibling routing until every boundary tensor has completed structural adaptation.
    void RouteBoundaryTensorToDDR(LogicalTensorPtr tensor);
    Status RouteVersionGroupsToDDR(Function& function);
    void InsertDdrToUbViewsForSibling(Function& function, LogicalTensorPtr sibling);
    void TransferResultToken(Function& function, Operation& from, Operation& to);

    std::vector<Operation*> newOps;
    std::unordered_set<int> ddrRoutedRawMagics_;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_INTRA_SUBGRAPH_ADAPTER_H_
