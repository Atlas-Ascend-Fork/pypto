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
 * \file reduce_copy.h
 * \brief
 */

#ifndef PASS_REDUCE_COPY_H_
#define PASS_REDUCE_COPY_H_

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/tensor/logical_tensor.h"
#include <vector>
#include <set>
#include <unordered_set>
#include <queue>
#include <utility>

namespace npu::tile_fwk {
// 核型属于单个边界 op 自身, 不属于其所在子图/root(子图级聚合核型会掩盖真实触碰边界的 op)。
enum class EndpointCoreKind { AIC, AIV, AICPU, UNKNOWN };

struct BoundaryEndpoint {
    int subgraphId;            // 端点 op 所在子图, 门控按其当前 root 过滤候选组
    EndpointCoreKind coreKind; // 端点 op 自身核型
    bool directPath{false};    // 本 op 与本 boundary tensor 间的直连通路资格(L0C2UB/UB2L1)
};

struct BoundaryTensorInfo {
    int tensorMagic;
    std::vector<int> producerSubgraphs;
    std::vector<int> consumerSubgraphs;
    bool isDDR{false};                  // 目的: 仅 DDR tensor 参与 inner-external-use 检查
    std::vector<int> producerCvFuseIds; // 与 producerSubgraphs 逐项平行, 端点级 CvFuseId, 未分配为 -1
    std::vector<int> consumerCvFuseIds; // 与 consumerSubgraphs 逐项平行
    std::vector<BoundaryEndpoint> producers{};
    std::vector<BoundaryEndpoint> consumers{};
};

struct MergeInput {
    int numSubgraph{0};
    int maxLatency{0};
    int maxSubgraphAICOpNum{0}; // 合并后子图 AIC op 数上限, 超限拒绝合并
    int maxSubgraphAIVOpNum{0}; // 合并后子图 AIV op 数上限, 超限拒绝合并
    std::pair<double, double> aivRatio;
    std::vector<int> subgraphAICLatency;
    std::vector<int> subgraphAIVLatency;
    std::vector<int> subgraphAICOpNum;
    std::vector<int> subgraphAIVOpNum;
    std::vector<std::set<int>> subGraphInGraph;
    std::vector<std::set<int>> subGraphOutGraph;
    std::vector<std::vector<int>> mergeGroup;
    std::vector<bool> isEnforceMergeGroup;
    std::vector<bool> isValidMergeGroup;
    bool hasScopedOp{false}; // 图中存在 cvFuseId>=0 的 op(仅 CV 混合 scope 分配 cvFuseId, 纯 scope 保持 -1)
    std::vector<BoundaryTensorInfo> boundaryTensors;
    std::vector<std::vector<int>> subgraphToBoundaryTensorIds;
    // 每个 sg 所属的 feedback loop-carry 通路集合(空=不在任何环路上), 通路按"环"归类:
    // 共享子图的多条 slot 链属同一环(如 online softmax 的 out/sum/max), 同环内可互相合并。
    // 环路上的 sg 只允许与"通路归属完全相同"的 sg 合并, 防止通路外分支被吞进 carry 链。
    // 空向量 = 前端无 slot scope / 无 feedback slot, 门控不启用。
    std::vector<std::set<int>> subgraphLoopPaths;
};

struct MergeOutput {
    int numSubgraphUpdated;
    std::vector<int> subgraphIdUpdated;
};

class MixGraphMerger {
public:
    MixGraphMerger() = default;
    ~MixGraphMerger() = default;
    MergeOutput Merge(const MergeInput& input);
    bool enableAutoMix{true};

private:
    MergeInput mInput;
    MergeOutput mOutput;
    std::vector<int> mParent;
    std::vector<int> mRank;
    std::vector<std::vector<int>> mRootToBoundaryTensorIds;
    std::vector<int> mTensorVisitStamp;
    int mVisitStamp{0};
    std::unordered_set<int> mGlobalOutputSinks; // 出度0子图: 不作为任何 boundary tensor producer, 即最终输出端点
    std::unordered_set<int> mWarnedInnerTensorMagics; // 已输出过 WARN 的 tensor, 跨 merge loop 迭代去重防打屏
    // 每个 root 的 loop-carry 通路集合(成员原始集合的并集), 供同通路门控使用
    std::vector<std::set<int>> mRootLoopPaths;
    // cached merged graph (avoid redundant rebuild in CanMergeWithoutCycle)
    std::vector<std::set<int>> mCachedOutGraph;
    std::vector<std::set<int>> mCachedInGraph;
    // HasCycle scratch buffers (avoid per-call reallocation)
    std::vector<int> mInDegreeBuf;
    std::vector<bool> mIsRootBuf;
    std::queue<int> mQueueBuf;

    void Initialize(const MergeInput& input);
    void InitBoundaryTensorIndex();
    int FindParent(int x);
    void UnionSets(int x, int y);
    bool CanMergeWithoutCycle(const std::vector<int>& actualGroup);
    void WarnIfEnforceOpNumExceeds(const std::vector<int>& actualGroup);
    bool CanMergeWithConstraints(const std::vector<int>& actualGroup, bool allowSinkMerge);
    // 单个候选组的合并尝试: enforce 组走 CvFuseId 判定(同 scope 组外端点豁免),
    // auto-mix 组走约束检查; 返回是否执行了合并(用于轮次收敛判定)
    bool TryMergeGroup(size_t groupIdx, const MergeInput& input, int mergeLoopStep, bool allowSinkMerge);
    void PerformMerge(const std::vector<int>& actualGroup);
    void UpdateBoundaryTensorIndex(const std::vector<int>& actualGroup);
    void ApplyMergeToGraph(const std::vector<int>& actualGroup);
    void MergeNodesInCachedGraph(int root, const std::set<int>& del);
    void UpdateOutput();
    bool CheckTwoStageNmBenefit(const std::vector<int>& actualGroup,
                                const std::set<std::pair<int, int>>& directPathRootPairs, std::set<int>& twoStageRoots);
    // 双侧分类: 组内每个 root 须只有前驱或只有后继(中间/孤立 root 拒绝),
    // 且生产/消费两侧各 >= kMultiBranchThreshold 个 root
    bool ClassifyTwoStageSides(const std::vector<int>& actualGroup, std::set<int>& producers, std::set<int>& consumers);
    // 完全二部校验: 每个 prod->cons 依赖均须存在且带直连证据(directPathRootPairs)
    bool VerifyTwoStageDependencies(const std::set<int>& producers, const std::set<int>& consumers,
                                    const std::set<std::pair<int, int>>& directPathRootPairs);
    // 双侧并行损失约束: sum-max 均不超过 kSerialLossRatio x 对侧 max 且两侧 max > 0
    bool IsTwoStageLossAcceptable(const std::set<int>& producers, const std::set<int>& consumers);
    bool CheckLatencyConstraint(const std::vector<int>& actualGroup);
    // 候选组内 loop-carry 通路集合是否混杂(存在与首个 root 通路集合不同的成员)
    bool IsLoopPathMixed(const std::vector<int>& actualGroup);
    bool CheckLoopPathConsistency(const std::vector<int>& actualGroup);
    // Tensor topology and endpoint-level direct-path evidence for one candidate.
    struct StructuralPattern {
        // root 级核构成(纯C/纯V/混合): 混合 root 的跨核代价已沉没, 不触发直连门
        std::unordered_map<int, std::pair<bool, bool>> rootCoreKinds;
        std::set<int> allProdRoots;
        std::unordered_map<int, std::set<int>> rootPreds;
        std::unordered_map<int, std::set<int>> rootSuccs;
        std::unordered_map<int, std::set<int>> directConsumersByRoot;
        std::set<std::pair<int, int>> directPathRootPairs;
        bool hasSimpleTensorEdge{false};
        bool hasDirectPathEdge{false};
        bool hasCrossCoreEdge{false};
    };
    void CollectStructuralPattern(const std::vector<int>& actualGroup, const std::unordered_set<int>& mergedRoot,
                                  StructuralPattern& pattern);
    // 单个 boundary tensor 的 op 级跨核端点对收集: 填充 directConsumersByRoot/directPathRootPairs
    // 与 hasCrossCoreEdge/hasDirectPathEdge 直连证据(组外端点对只计入 directConsumersByRoot)
    void CollectTensorEndpointPairs(int tensorId, const std::unordered_set<int>& mergedRoot,
                                    StructuralPattern& pattern);
    // 建 root 级方向边 (prodRoot -> consRoot, 跳过 self-loop), 填充 rootPreds/rootSuccs
    void BuildRootLevelEdges(const std::set<int>& prodRoots, const std::set<int>& consRoots,
                             StructuralPattern& pattern);
    // 输出级 consolidate 旁路: 全图 >=2 个出度0 sink 且本候选含其全部时,
    // 视为合并并行输出级(下游无并行可损失), 跳过 sink 保护
    bool AllOutputSinksInCandidate(const std::unordered_set<int>& mergedRoot);
    // 收集 root 的全部 producer root(排除自身), 用于 sink 入边统计
    std::set<int> CollectIncomingRoots(int root);
    // sink(汇聚点): 不产出任何 boundary tensor 的 root(纯消费输入、无对外输出);
    // 本函数做 sink 串行损失保护、输出级 consolidate 旁路与 sink 轮推迟门控
    bool CheckStructuralSinkBenefit(const std::vector<int>& actualGroup, const std::unordered_set<int>& mergedRoot,
                                    const std::set<int>& allProdRoots, bool allowSinkMerge);
    // hinge(铰链端点): 多个组内 root 共同挂靠的边界 tensor 对端(1:N 扇出 / N:1 扇入),
    // 端点可在组内或组外; 本函数防并行分支被串进单 kernel
    bool CheckStructuralHingeBenefit(const std::vector<int>& actualGroup, const std::unordered_set<int>& mergedRoot,
                                     const StructuralPattern& pattern, const std::set<int>& twoStageRoots);
    // hinge 端点 -> 组内分支集合: 端点(组内或组外)通过 boundary tensor 直连的组内 root 全集
    std::unordered_map<int, std::set<int>> CollectHingeToBranchRoots(const std::vector<int>& actualGroup,
                                                                     const std::unordered_set<int>& mergedRoot);
    // 同环豁免: 分支两两共享 loop-carry 通路(同一 ring 的多链)时非独立并行分支
    bool AreBranchesRingCoupled(const std::set<int>& branches);
    // 规模豁免: 独立分支串行化损失(Σ-max)可被 hinge 规模消化, 或直连 1:N hinge 的
    // 共享 GM 往返收益可摊销串行损失时放行
    bool IsHingeSerialLossAbsorbable(int hingeRoot, const std::set<int>& branches, const StructuralPattern& pattern);
    bool CheckMergeBenefitByStructuralPattern(const std::vector<int>& actualGroup, bool allowSinkMerge);
    // 判断汇聚 sink 的合并是否无收益需拒绝(串行损失模型): 串行化损失 Σbranch − max(branch)
    // 超过 kSerialLossRatio × 合入root 总 latency 时拒绝; 调用方保证入边数 >= 2
    bool IsSinkMergeUnbeneficial(int root, const std::set<int>& incomingRoots);
    bool CheckNoExternalUseOfMergedInnerTensor(const std::vector<int>& actualGroup, bool checkByCvFuseId = false);
    // 首次发现的非法 inner tensor 打 WARN(含成因推导), 同一 tensorMagic 重复出现时降级 DEBUG,
    // 避免同一 tensor 在多个候选组反复告警刷屏
    void LogMergedInnerTensorRejection(const std::vector<int>& actualGroup, int tensorId,
                                       const std::vector<int>& prodIn, const std::vector<int>& prodOut,
                                       const std::vector<int>& consIn, const std::vector<int>& consOut,
                                       bool checkByCvFuseId);
    bool IsInvalidMergedInnerTensor(int tensorId, const std::unordered_set<int>& mergedRoots, std::vector<int>& prodIn,
                                    std::vector<int>& prodOut, std::vector<int>& consIn, std::vector<int>& consOut);
    bool IsInvalidMergedInnerTensorByCvFuseId(int tensorId, const std::unordered_set<int>& mergedRoots,
                                              std::vector<int>& prodIn, std::vector<int>& prodOut,
                                              std::vector<int>& consIn, std::vector<int>& consOut);
    std::vector<int> GetActualGroup(const std::vector<int>& group);
    void BuildMergedGraph(std::vector<std::set<int>>& outGraph, std::vector<std::set<int>>& inGraph);
    bool HasCycle(const std::vector<std::set<int>>& outGraph, const std::vector<std::set<int>>& inGraph);
};

class ReduceCopyMerge : public Pass {
public:
    ReduceCopyMerge() : Pass("ReduceCopyMerge") { SetSupportedArches({NPUArch::DAV_3510}); }
    ~ReduceCopyMerge() override = default;

private:
    Status BuildGraph(Function& function, MergeInput& mergeInput);
    Status BuildMergeGroup(Function& function, MergeInput& mergeInput);
    void CombineForkSubgraph(Function& function, MergeInput& mergeInput);
    Status MarkNoMergeSubgraph(Function& function, MergeInput& mergeInput);
    void MarkFeedbackSubgraphs(Function& function, MergeInput& mergeInput);
    void UpdateConnectRecord(Function& function, MergeInput& mergeInput);
    void UpdateBoundaryTensorSize(LogicalTensorPtr& tensor, int tensorSize);
    void RecordBoundaryTensorInfo(LogicalTensorPtr& tensor, MergeInput& mergeInput, const std::set<int>& connectGraphs);
    void UpdateMergeInput(MergeInput& mergeInput, std::multimap<int, std::vector<int>>& sortedMergeGroup);
    bool IsEnforceMergeBoundary(LogicalTensorPtr& tensor);
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;
    std::unordered_map<int, std::vector<int>> subgraphToOutputTensors;
    std::unordered_map<int, std::vector<int>> subgraphToInputTensors;
    std::unordered_map<int, std::vector<int>> tensorToMergeGroup;
    std::unordered_set<int> noMergeSubgraph;
    std::unordered_set<int> noMergeSubgraphEnforce;
    std::map<std::vector<int>, int> mergeGroupToPriority;
    std::set<std::vector<int>> enforceMergeGroup;
    std::vector<int> subgraphInputSize;
    std::vector<int> subgraphOutputSize;
};

} // namespace npu::tile_fwk
#endif
