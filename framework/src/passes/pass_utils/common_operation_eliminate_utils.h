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
 * \file common_operation_eliminate_utils.h
 * \brief utils of common operation elimination
 */

#ifndef PASS_COMMON_OPERATION_ELIMINATE_UTILS_H_
#define PASS_COMMON_OPERATION_ELIMINATE_UTILS_H_

#include <cstdint>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {
class CommonOperationEliminateUtils {
public:
    CommonOperationEliminateUtils() = default;
    ~CommonOperationEliminateUtils() = default;

    static Status EliminateCommonOperation(Function& function);

    Status Process(Function& function);

private:
    static const std::unordered_set<Opcode>& GetSkipEliminateOpcodes();
    static std::string DumpAttr(const Operation& op);
    void SortedProducer(std::vector<Operation*>& sortedProducers) const;
    void CollectProducerInfo(const std::vector<Operation*>& sortedProducers, const LogicalTensorPtr& curTensor,
                             std::vector<std::string>& opStrList, std::stringstream& ss) const;
    unsigned long ComputeHash(const std::vector<Operation*>& producers, LogicalTensorPtr curTensor) const;
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> GetTensorProducers(
        Function& function, std::vector<LogicalTensorPtr>& sequence);
    bool UpdateConnection(Function& function, LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor,
                          const std::vector<Operation*>& oldProducers, const std::vector<Operation*>& newProducers);
    bool BuildOutputTensorRedirectPairs(
        Function& function, const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor,
        const std::vector<Operation*>& sortedOldProducers, const std::vector<Operation*>& sortedNewProducers,
        std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>>& tensorRedirectPairs) const;
    uint32_t GetTensorCoreFlag(const LogicalTensorPtr& tensor) const;
    void CollectSubgraphIds(const std::set<Operation*, LogicalTensor::CompareOp>& ops,
                            std::unordered_set<int>& subgraphIds) const;
    void UpdateInternalTensorCoreFlag(const LogicalTensorPtr& tensor,
                                      std::unordered_map<int, uint32_t>& subgraphCoreFlags) const;
    std::unordered_set<int> GetMixSubgraphIds(Function& function) const;
    // mix 子图分析: mixSubgraphIds_(供 WouldExpose 门) + aivBranchOfOp_(供 branch 门, 仅 DAV_3510)
    void AnalyzeMixSubgraphs(Function& function);
    std::unordered_map<Operation*, int> BuildAivBranchMap(Function& function,
                                                          const std::unordered_set<int>& mixSubgraphIds);
    // aivOps 按直接数据依赖边(AIV→AIV, 同子图)的连通分量, op -> 分量 id
    std::unordered_map<Operation*, int> BuildAivConnectedComponents(const std::vector<Operation*>& aivOps);
    bool ShouldSkipCrossBranchMerge(const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor,
                                    const std::vector<Operation*>& oldProducers,
                                    const std::vector<Operation*>& newProducers) const;
    MemoryType GetAssembleInputMemoryType(Operation* consumer, const LogicalTensorPtr& input) const;
    MemoryType GetAssembleOutputMemoryType(Operation* consumer) const;
    bool HasSameMemoryAssembleUse(const LogicalTensorPtr& tensor, std::unordered_set<int>& visitedTensorMagics) const;
    bool ShouldSkipSameMemoryAssembleMerge(const LogicalTensorPtr& oldTensor) const;
    bool WouldExposeMixInternalTensorAfterMerge(const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor,
                                                const std::unordered_set<int>& mixSubgraphIds) const;
    std::pair<LogicalTensorPtr, std::vector<Operation*>> TensorHashExist(
        const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
        const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap,
        std::vector<Operation*>& currentBucketProducers);
    bool ShouldSkipProducers(const std::vector<Operation*>& producers) const;
    void AugmentProducersWithSiblings(const LogicalTensorPtr& orderedTensor,
                                      std::vector<Operation*>& augmentedProducers);
    bool TensorProducersMerge(Function& function, const LogicalTensorPtr orderedTensor,
                              std::unordered_set<Operation*>& cacheProducers,
                              const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap,
                              std::vector<Operation*>& oldBucketProducers);
    const TensorSet& GetSameRawMagicTensors(const LogicalTensorPtr& tensor) const;
    std::vector<Operation*> CollectConsumersWithSameRawMagic(Function& function, const LogicalTensorPtr& tensor) const;
    void MarkRedundantProducersDeleted(Function& function, const std::vector<Operation*>& oldBucketProducers,
                                       std::unordered_set<Operation*>& cacheProducers);
    void UpdateView(ViewOpAttribute* viewOpAttribute, const std::shared_ptr<LogicalTensor> oldTensor,
                    const std::shared_ptr<LogicalTensor> newTensor) const;
    void UpdateCopy(CopyOpAttribute* copyOpAttribute, const std::shared_ptr<LogicalTensor> oldTensor,
                    const std::shared_ptr<LogicalTensor> newTensor) const;
    bool ProducerPairingValid(LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor,
                              const std::vector<Operation*>& oldProducers,
                              const std::vector<Operation*>& newProducers) const;
    void UpdateTensorConsumers(LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor) const;

    std::unordered_map<uint64_t, std::pair<LogicalTensorPtr, std::vector<Operation*>>> hashCache_;
    std::unordered_set<int> mixSubgraphIds_;
    // op -> AIV 连通分量 id。空表 = branch 门不生效
    std::unordered_map<Operation*, int> aivBranchOfOp_;
    Function* function_ = nullptr;
    RawMagicTensorMap tensorsByRawMagic_;
    // A RawTensor bucket is processed once, including all of its logical tensors.
    std::unordered_set<int> processedRawMagicTensors_;
};
} // namespace npu::tile_fwk
#endif // PASS_COMMON_OPERATION_ELIMINATE_UTILS_H_
