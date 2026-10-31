/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file merge_view_assemble_utils.h
 * \brief utils of view and assemble operation merging
 */

#ifndef PASS_MERGE_VIEW_ASSEMBLE_UTILS_H_
#define PASS_MERGE_VIEW_ASSEMBLE_UTILS_H_

#include <unordered_map>
#include <unordered_set>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/irbuilder.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
// Merge semantics are selected from each chain's IR, independently of the frontend
// logical-tensor construction option. Coverage and shared-tail checks protect both modes.
class MergeViewAssembleUtils {
public:
    MergeViewAssembleUtils() = default;
    ~MergeViewAssembleUtils() = default;

    struct TokenDependency {
        std::vector<ir::VarPtr> inputTokens;
        std::vector<ir::VarPtr> resultTokens;
        // Per resultTokens entry: the external consumers of that token.
        std::vector<std::vector<ir::StmtPtr>> resultTokenConsumers;
        std::vector<ir::VarPtr> touchedTokens;
    };

    struct ViewOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int64_t> offset;
        std::vector<SymbolicScalar> dynOffset;
        std::vector<SymbolicScalar> dynValidShape;
        MemoryType toType = MemoryType::MEM_UNKNOWN;
        bool hasCopyInMode;       // 是否有copy_in_mode属性
        std::any copyInModeValue; // copy_in_mode属性值
        bool hasL1PaddingMode;    // 是否有copy_in_l1_padding_mode属性
        std::any l1PaddingMode;   // copy_in_l1_padding_mode属性值
        bool hasKIndex;           // 是否有copy_in_l1_k_index属性
        std::any kIndex;          // copy_in_l1_k_index属性值（K维度在dynValidShape中的索引，0或1）
        bool hasIsGemv;           // 是否有 isGemv 属性
        std::any isGemvValue;     // isGemv 属性值（0=false，非0=true）
        ir::Span span;            // 链路最早操作的span
        Operation::ScopeInfo scopeInfo;
        Opcode opcode = Opcode::OP_VIEW;
        TokenDependency tokenDependency;
    };
    struct AssembleOp {
        std::shared_ptr<LogicalTensor> input;
        std::shared_ptr<LogicalTensor> output;
        std::vector<int64_t> offset;
        std::vector<SymbolicScalar> dynOffset;
        ir::Span span; // 链路最早操作的span
        Operation::ScopeInfo scopeInfo;
        std::string rmwModeAttr;
        Opcode opcode = Opcode::OP_ASSEMBLE;
        TokenDependency tokenDependency;
        bool atomicFromReduceAcc = false;
        bool atomicFromExplicitRmw = false;
        int subgraphId = -1;
    };
    struct ConsumerCacheEntry {
        std::vector<Operation*> viewConsumers;
        std::vector<Operation*> assembleConsumers;
        // A dependency-sensitive non-view consumer stops a view chain at this tensor.
        bool hasViewChainStopper = false;
        // Any non-assemble consumer stops an assemble chain at this tensor.
        bool hasAssembleChainStopper = false;
        size_t producerCount = 0;
        bool allProducersAreAssembleLike = false;
    };
    struct DependencyCacheEntry {
        std::vector<Operation*> producers;
        std::vector<Operation*> consumers;
    };

    static Status MergeViewAssemble(Function& function);

    Status Process(Function& function);

    // View chain processing methods
    /**
     * @brief Merge a chain of view operations into a single view.
     *
     * @param function the target function for the operation to be processed.
     * @param operation the starting operation of the view chain.
     * @param chain the list of operations in the view chain.
     * @return Status indicating success or failed.
     */
    Status MergeViewChain(Function& function, Operation& operation, std::vector<Operation*>& chain,
                          int effectiveScopeId = -1);

    void InitOperationChain(Operation& operation, std::vector<Operation*>& chain);

    /**
     * @brief Process the consumer chain of a view.
     *
     * @param function the target function for the operation to be processed.
     * @param consumers the consumers for the view to be processed.
     * @param chain the list of operations in the view chain.
     * @param chainEnd a flag indicating whether the chain has ended.
     * @return Status indicating success or failed.
     */
    Status ProcessConsumerChain(Function& function, const ConsumerCacheEntry& consumers, std::vector<Operation*>& chain,
                                bool& chainEnd, int effectiveScopeId);

    Status ProcessChainEnd(Function& function, std::vector<Operation*>& chain);

    /**
     * @brief Calculate the merged offsets and dynamic vaildshapes for the chain of a view.
     *
     * @param chain the list of operations in the view chain.
     * @param newOffset the calculated newoffset.
     * @param newDynOffset the calculated newDynOffset.
     * @param newDynValidShape the calculated newDynValidShape.
     * @return Status indicating success or failed.
     */
    Status CalculateMergedOffsets(const std::vector<Operation*>& chain, std::vector<int64_t>& newOffset,
                                  std::vector<SymbolicScalar>& newDynOffset,
                                  std::vector<SymbolicScalar>& newDynValidShape);

    /**
     * @brief Recode the merged offsets and dynamic vaildshapes for the chain of a view.
     *
     * @param lastViewOp the list of operations in the view chain.
     * @param startTensor the start tensor of the chain.
     * @param endTensor the end tensor of the chain.
     * @param newOffset the calculated newoffset.
     * @param newDynOffset the calculated newDynOffset.
     * @param newDynValidShape the calculated newDynValidShape.
     */
    void RecordMergedViewOperation(Operation* lastViewOp, const std::shared_ptr<LogicalTensor>& startTensor,
                                   const std::shared_ptr<LogicalTensor>& endTensor,
                                   const std::vector<int64_t>& newOffset,
                                   const std::vector<SymbolicScalar>& newDynOffset,
                                   const std::vector<SymbolicScalar>& newDynValidShape, const ir::Span& span,
                                   const Operation::ScopeInfo& scopeInfo, Opcode opcode,
                                   const TokenDependency& tokenDependency);

    // Assemble chain processing methods
    /**
     * @brief Merge a chain of assemble operations into a single assemble.
     *
     * @param function the target function for the operation to be processed.
     * @param operation the starting operation of the assemble chain.
     * @param chain the list of operations in the assemble chain.
     * @return Status indicating success or failed.
     */
    Status MergeAssembleChain(Function& function, Operation& operation, std::vector<Operation*>& chain,
                              int effectiveScopeId = -1);

    void InitAssembleChain(Operation& operation, std::vector<Operation*>& chain);

    /**
     * @brief Process the consumer chain of a assemble.
     *
     * @param function the target function for the operation to be processed.
     * @param consumers the consumers for the assemble to be processed.
     * @param chain the list of operations in the assemble chain.
     * @param chainEnd a flag indicating whether the chain has ended.
     * @return Status indicating success or failed.
     */
    Status ProcessAssembleConsumers(Function& function, const ConsumerCacheEntry& consumers,
                                    std::vector<Operation*>& chain, bool& chainEnd, int effectiveScopeId);

    // Middle extension gate: split-version contribution plus fan-in
    // producer coverage. Returns false when the chain must stop at this middle.
    bool CanExtendThroughMiddle(Function& function, Operation& currentOp);
    // Stops extension when version contribution or coverage cannot be preserved.
    bool StopsAtMiddle(Function& function, const ConsumerCacheEntry& consumers, Operation* currentOp);
    bool ShouldStopAssembleChain(const std::vector<Operation*>& chain, const ConsumerCacheEntry& consumers) const;

    Status ProcessAssembleChainEnd(Function& function, std::vector<Operation*>& chain, Operation& operation);

    // Safety gates applied before merging an assemble chain.
    bool HasBlockingChainCycle(Function& function, const std::vector<Operation*>& chain);
    bool HasCompleteAssembleGroup(const std::vector<size_t>& group) const;
    Status CommitAssembleChains(Function& function);
    Status CommitAssembleChain(Function& function, const std::vector<Operation*>& chain);
    // Rewrites inputTokens in-place: keeps tokens owned by this chain, drops the ones
    // owned by sibling-producer chains (data-coverage precision).
    void FilterOwnedInputTokens(Function& function, TokenDependency& tokenDependency,
                                const std::vector<Operation*>& chain);
    // True when an intermediate output still has readers outside the chain: a live op,
    // or a merge recorded for an earlier chain whose merged op (created in the append
    // phase) reads it.
    bool HasChainOutSurvivor(const std::vector<Operation*>& chain) const;
    // Deletes merged chain ops; intermediates with side consumers survive. Intermediates
    // whose output still has live readers outside the chain also survive (the reader's
    // own chain merge may have been rejected, leaving it dependent on the intermediate).
    void DeleteMergedChainOps(const std::vector<Operation*>& chain, const std::shared_ptr<LogicalTensor>& endTensor);

    std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> CalculateAssembleOffsets(
        const std::vector<Operation*>& chain, size_t offsetSize);

    void RecordAssembleOperation(const std::shared_ptr<LogicalTensor>& input,
                                 const std::shared_ptr<LogicalTensor>& output, const std::vector<int64_t>& offset,
                                 const std::vector<SymbolicScalar>& dynOffset, const ir::Span& span,
                                 const Operation::ScopeInfo& scopeInfo, const std::string& rmwModeAttr, Opcode opcode,
                                 const TokenDependency& tokenDependency, bool atomicFromReduceAcc,
                                 bool atomicFromExplicitRmw, int subgraphId);

    // Common methods
    Status Initialize();
    Status BuildConsumerCache(Function& function);
    void RecordTensorVersion(const LogicalTensorPtr& tensor);
    void CacheOperationConsumers(Function& function, Operation& operation);
    bool HasVersionedAssembleOutput(const Operation& operation) const;
    void RecordTokenDependency(Operation& operation);
    void BuildTokenSemanticCache(Function& function);
    void BuildDependencyCache(Function& function);
    bool HasSafeChainBoundary(const std::vector<Operation*>& chain) const;
    void PropagateTokenSemantics(std::vector<Operation*>& tokenWorklist);
    bool IsTokenSemanticOperation(const Operation& operation) const;
    bool HasChainTokenDependency(const std::vector<Operation*>& chain, const ConsumerCacheEntry& consumers) const;
    bool HasSplitVersionContribution(const LogicalTensorPtr& middle,
                                     const std::vector<Operation*>& currentProducers) const;
    // Static coverage check with per-middle memoization: producers and their attributes
    // are stable during chain processing, so a middle's result is computed once per
    // invocation (guards the cell x region blow-up on large fan-in middles).
    bool HasCompleteStaticCoverage(const LogicalTensorPtr& middle, const std::vector<Operation*>& producers) const;
    bool ComputeCompleteStaticCoverage(const LogicalTensorPtr& middle, const std::vector<Operation*>& producers) const;
    const ConsumerCacheEntry& BuildTensorConsumerCache(Function& function, const LogicalTensorPtr& tensor);
    const ConsumerCacheEntry& GetConsumers(const Operation& operation) const;
    static ir::Span GetFirstSpan(const std::vector<Operation*>& chain);
    static Operation::ScopeInfo GetChainScopeInfo(const std::vector<Operation*>& chain);

    // Processing methods
    Status ProcessOperations(Function& function);
    Status ProcessCandidateChains(Function& function);
    Status AppendMergedOperations(Function& function);

    // Operation appending methods
    Status AppendMergedViewOperations(Function& function);
    Status AppendMergedAssembleOperations(Function& function);

    // Remove legacy result tokens replaced by per-merged-op new tokens.
    void CleanupLegacyResultTokens(Function& function);

    // Cleanup methods
    void EraseLegacyRedundantAssembles(Function& function) const;
    Status CleanUp(Function& function);
    // Current chain semantics. It is reset at each candidate chain start.
    bool useTokenSemantics_ = false;
    // Whole-graph preparation state; it does not select the semantics of a chain.
    bool hasAnyTokenSemantics_ = false;
    std::unordered_set<Operation*> tokenSemanticOps_;
    std::unordered_set<int> tokenDependencyTensorMagics_;
    std::unordered_set<Operation*> tokenDependencyOps_;
    // Original data/token edges remain conservative while chain planning only removes
    // edges. Replacement operations are appended after all chain safety checks.
    std::unordered_map<Operation*, DependencyCacheEntry> dependencyCache_;
    std::unordered_set<int> visitedOp_;
    std::unordered_map<int, const ConsumerCacheEntry*> consumerCache_;
    std::unordered_map<int, ConsumerCacheEntry> tensorConsumerCache_;
    std::unordered_map<int, std::vector<LogicalTensorPtr>> rawTensorVersions_;
    std::vector<Operation*> candidateOps_;
    std::vector<ViewOp> viewOpToAppend_;
    std::vector<AssembleOp> assembleOpToAppend_;
    // Candidate paths are validated together before deleting their shared tail.
    std::vector<std::vector<Operation*>> pendingAssembleChains_;
    // Input tensor magics of all merges recorded so far (view + assemble); lets the
    // chain-out survivor check see readers that only materialize in the append phase.
    std::unordered_set<int> recordedMergeInputMagics_;
    // Per-middle coverage results for the current invocation (see HasCompleteStaticCoverage).
    mutable std::unordered_map<int, bool> coverageCache_;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_MERGE_VIEW_ASSEMBLE_UTILS_H_
