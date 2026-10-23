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
 * \file expand_function.cpp
 * \brief
 */

#include "passes/tensor_graph_pass/expand_function.h"
#include <map>
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/operation/tile_shape_resolver.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_check/expand_function_checker.h"
#include "passes/statistics/tensor_and_tile_graph_statistic.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/token_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/tile_shape.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#define MODULE_NAME "ExpandFunction"

using namespace npu::tile_fwk;

namespace {

bool SameLogicalTensor(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs);

bool IsTransparentForTileTraverse(Opcode opcode)
{
    return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_RESHAPE;
}

bool IsViewAssembleHopTransparent(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE; }

void AppendTraverseNeighbors(bool traverseConsumers, Operation* op, std::vector<Operation*>& neighbors)
{
    const auto nextOps = traverseConsumers ? op->ConsumerOps() : op->ProducerOps();
    for (auto* nextOp : nextOps) {
        neighbors.push_back(nextOp);
    }
}

bool OperandShapesHaveDynamicDim(const LogicalTensors& operands)
{
    for (const auto& tensor : operands) {
        if (tensor == nullptr) {
            continue;
        }
        for (const auto dim : tensor->GetShape()) {
            if (dim == -1) {
                return true;
            }
        }
    }
    return false;
}

// 从 currentLevel 层出发逐层 BFS（方向由 traverseConsumers 决定）。
// 若某一层（跳过 VIEW/ASSEMBLE 穿透层后）的非穿透节点全部为 RESHAPE 则返回 true；
// 否则返回 false，遍历耗尽也返回 false。
bool TraversePathContainsOnlyReshape(std::vector<Operation*> currentLevel, bool traverseConsumers)
{
    std::unordered_set<Operation*> visited;
    while (!currentLevel.empty()) {
        bool levelHasReshape = false;
        bool levelHasNonReshapeCompute = false;
        std::vector<Operation*> nextLevel;
        for (auto* curOp : currentLevel) {
            if (curOp == nullptr || !visited.insert(curOp).second) {
                continue;
            }
            const auto opcode = curOp->GetOpcode();
            if (IsViewAssembleHopTransparent(opcode)) {
                AppendTraverseNeighbors(traverseConsumers, curOp, nextLevel);
            } else if (opcode == Opcode::OP_RESHAPE) {
                levelHasReshape = true;
            } else {
                levelHasNonReshapeCompute = true;
            }
        }
        if (levelHasNonReshapeCompute) {
            return false;
        }
        if (levelHasReshape) {
            return true;
        }
        if (nextLevel.empty()) {
            break;
        }
        currentLevel = std::move(nextLevel);
    }
    return false;
}

// VIEW/ASSEMBLE 不参与 tile 展开的条件：
// 1) 参与展开循环的 shape 含动态维 -1（VIEW 看输出，ASSEMBLE 看输入；TiledView/TiledAssemble
//    对 -1 的 for 循环次数为 0，会断图）；
// 2) 在 consumer/producer 方向 BFS 时，若某一层（跳过 VIEW/ASSEMBLE 穿透层后）
//    首次出现的非穿透节点全部为 RESHAPE。
bool ShouldSkipViewAssembleExpand(const Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_VIEW && op.GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }

    // VIEW 按输出 shape 切分，ASSEMBLE 按输入 shape 切分；对应 shape 含 -1 则跳过展开。
    if (op.GetOpcode() == Opcode::OP_VIEW) {
        if (OperandShapesHaveDynamicDim(op.GetOOperands())) {
            return true;
        }
    } else if (OperandShapesHaveDynamicDim(op.GetIOperands())) {
        return true;
    }

    const bool traverseConsumers = (op.GetOpcode() == Opcode::OP_VIEW);
    std::vector<Operation*> currentLevel;
    if (traverseConsumers) {
        for (auto* consumer : op.ConsumerOps()) {
            currentLevel.push_back(consumer);
        }
    } else {
        for (auto* producer : op.ProducerOps()) {
            currentLevel.push_back(producer);
        }
    }

    return TraversePathContainsOnlyReshape(std::move(currentLevel), traverseConsumers);
}

std::unordered_set<Operation*> CollectViewAssembleSkipExpandOps(const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr) {
            continue;
        }
        if (opPtr->GetOpcode() != Opcode::OP_VIEW && opPtr->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (ShouldSkipViewAssembleExpand(*opPtr)) {
            skipExpandOps.insert(opPtr.get());
        }
    }
    return skipExpandOps;
}

// ASSEMBLE_SSA 的 in-place dst 输入（INPLACE_IDX 指定的 operand）是一个 VIEW，
// 它表示"当前 dst 在指定 offset 处的视图"。ASSEMBLE_SSA 自身已经包含了
// "从 dst 读取 → 与 src 合并 → 写回 result"的完整搬运动作语义。
// 如果将该 VIEW 展开为 SLICE + CONTRACT(NeedCopy=true)，会产生一个中间 buffer，
// 下游 Pass 无法将其与 ASSEMBLE_SSA 的 in-place 语义正确合并，
// 导致残留 CONTRACT → 额外 COPY_OUT → OoOSchedule 产生 zero predicate → 部分数据为 0。
// 因此这类 VIEW 不应展开，保持视图语义即可。
std::unordered_set<Operation*> CollectAssembleSsaDstViewSkipExpandOps(const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_ASSEMBLE_SSA) {
            continue;
        }
        if (!opPtr->HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        int inplaceIdx = opPtr->GetIntAttribute(OpAttributeKey::inplaceIdx);
        if (inplaceIdx < 0 || static_cast<size_t>(inplaceIdx) >= opPtr->GetInputOperandSize()) {
            continue;
        }
        auto dstTensor = opPtr->GetInputOperand(static_cast<size_t>(inplaceIdx));
        if (dstTensor == nullptr) {
            continue;
        }
        for (auto* producer : opPtr->ProducerOps()) {
            if (producer == nullptr || producer->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            const auto& oOperands = producer->GetOOperands();
            if (!oOperands.empty() && SameLogicalTensor(oOperands[0], dstTensor)) {
                skipExpandOps.insert(producer);
            }
        }
    }
    return skipExpandOps;
}

// ASSEMBLE 的输出是函数的 outcast（即直接作为函数输出张量）时，暂不展开为SLICE + CONTRACT。
// 理论上，由于assemble前的实际运算op会展开并在末尾插入contract，此时assemble可以单纯表示视图语义，表达本次结果再整个tensor中的位置
// 流程中，当assemble前存在多个不同tile的op汇聚时，会导致assemble的tile无法与前面的展开一一匹配，无法完美处理，导致冗余搬运。
// 但若assemble携带token依赖（WAW/WAR），仍需展开以将semantic token传播并转换为contract上的NORMAL
// token，保证下游同步正确。
// 同理，ASSEMBLE 的输出仅被一个 RESHAPE 消费且该 RESHAPE 的输出是 outcast 时，ASSEMBLE 也不展开：
// RESHAPE 展开后仅生成 TILE_RESHAPE dummy 标记 op（不搬数据），ASSEMBLE 仍可保持视图语义。
bool AssembleOutputIsOutcast(const std::vector<std::shared_ptr<LogicalTensor>>& outcasts,
                             const LogicalTensorPtr& assembleOutput)
{
    for (const auto& outcast : outcasts) {
        if (SameLogicalTensor(outcast, assembleOutput)) {
            return true;
        }
    }
    return false;
}

bool AssembleFeedsOutcastThroughReshape(const std::vector<std::shared_ptr<LogicalTensor>>& outcasts,
                                        const LogicalTensorPtr& assembleOutput)
{
    if (assembleOutput == nullptr) {
        return false;
    }
    const auto& consumers = assembleOutput->GetConsumers();
    if (consumers.size() != 1) {
        return false;
    }
    auto* reshapeOp = *consumers.begin();
    if (reshapeOp == nullptr || reshapeOp->IsDeleted() || reshapeOp->GetOpcode() != Opcode::OP_RESHAPE ||
        reshapeOp->GetOOperands().empty()) {
        return false;
    }
    return AssembleOutputIsOutcast(outcasts, reshapeOp->GetOOperands()[0]);
}

std::unordered_set<Operation*> CollectAssembleOutcastSkipExpandOps(Function& function,
                                                                   const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    const auto& outcasts = function.GetOutcast();
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        const auto& oOperands = opPtr->GetOOperands();
        if (oOperands.empty()) {
            continue;
        }
        const bool outputReachesOutcast = AssembleOutputIsOutcast(outcasts, oOperands[0]) ||
                                          AssembleFeedsOutcastThroughReshape(outcasts, oOperands[0]);
        if (outputReachesOutcast && opPtr->result_token_.empty() && opPtr->tokens_.empty()) {
            skipExpandOps.insert(opPtr.get());
        }
    }
    return skipExpandOps;
}

// GatherInUB 的 params 输入（第 0 个操作数）通常是对 GM 大缓冲的整体别名 view（如 KV cache）。
// GatherInUB 的输入本身要求 DDR（GM 直读），该类 view 保留穿透别名语义不展开，与 legacy 行为一致。
std::unordered_set<Operation*> CollectGatherInUbParamViewSkipOps(const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_VIEW || opPtr->GetOOperands().size() != 1) {
            continue;
        }
        const auto& output = opPtr->GetOOperands().front();
        if (output == nullptr) {
            continue;
        }
        // 仅当输出全部被 GatherInUB 作为 params(第 0 个输入)消费时跳过展开，
        // 其它消费方仍按原逻辑展开，避免影响需要物化的场景
        const auto& consumers = output->GetConsumers();
        if (consumers.empty()) {
            continue;
        }
        bool gatherParamOnly = true;
        for (auto* consumer : consumers) {
            if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_GATHER_IN_UB ||
                consumer->GetIOperands().empty() || consumer->GetIOperands().front() != output) {
                gatherParamOnly = false;
                break;
            }
        }
        if (gatherParamOnly) {
            skipExpandOps.insert(opPtr.get());
        }
    }
    return skipExpandOps;
}

// VIEW/ASSEMBLE 的输入或输出 tensor shape 含动态维 -1 时，展开为 SLICE/CONTRACT 会引入
// 带 NeedCopy 的强制搬运，动态维下界未知导致该类搬运无法被后续 Pass 合并消除。
// 若该 view/assemble 不携带 token 依赖（无 WAW/WAR 同步语义），保持 VIEW/ASSEMBLE
// 视图语义不展开，与 legacy 行为一致；携带 token 时仍需展开以传播同步依赖。
std::unordered_set<Operation*> CollectDynamicShapeViewAssembleSkipExpandOps(
    const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr) {
            continue;
        }
        if (opPtr->GetOpcode() != Opcode::OP_VIEW && opPtr->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        const bool hasDynamicDim = OperandShapesHaveDynamicDim(opPtr->GetIOperands()) ||
                                   OperandShapesHaveDynamicDim(opPtr->GetOOperands());
        const bool hasNoToken = opPtr->result_token_.empty() && opPtr->tokens_.empty();
        if (hasDynamicDim && hasNoToken) {
            skipExpandOps.insert(opPtr.get());
        }
    }
    return skipExpandOps;
}

// gather_in_l1 的输出是驻留 L1 的 tile 数据，语义上由后续 cube 的 COPY_IN(L1) 直读以实现
// L1 复用。若将消费该输出的 ASSEMBLE 展开为 SLICE + CONTRACT，会把 L1 数据物化搬运
// （产生 L1 spill 类搬运算子，其 copyout 仅支持 ND2ND 模式），同时使 GATHER_IN_L1 输出
// 变为多 consumer 触发 DuplicateOp 复制，破坏 L1 复用语义。因此这类 ASSEMBLE 保持
// 视图语义不展开，与 legacy（enable_slice=false）行为一致。
std::unordered_set<Operation*> CollectAssembleGatherInL1SrcSkipExpandOps(
    const std::vector<OperationPtr>& tensorOperations)
{
    std::unordered_set<Operation*> skipExpandOps;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        for (const auto& input : opPtr->GetIOperands()) {
            if (input == nullptr) {
                continue;
            }
            for (auto* producer : input->GetProducers()) {
                if (producer != nullptr && producer->GetOpcode() == Opcode::OP_GATHER_IN_L1) {
                    skipExpandOps.insert(opPtr.get());
                }
            }
        }
    }
    return skipExpandOps;
}

bool IsMatmulOpcode(Opcode opcode)
{
    static const std::unordered_set<Opcode> kMatmulOps = {Opcode::OP_A_MUL_B,  Opcode::OP_A_MULACC_B,
                                                          Opcode::OP_A_MUL_BT, Opcode::OP_A_MULACC_BT,
                                                          Opcode::OP_AT_MUL_B, Opcode::OP_AT_MUL_BT};
    return kMatmulOps.count(opcode) > 0;
}

bool SameLogicalTensor(const LogicalTensorPtr& lhs, const LogicalTensorPtr& rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    return lhs.get() == rhs.get() || lhs->magic == rhs->magic;
}

std::optional<int> GetInputOperandIndex(const Operation& op, const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return std::nullopt;
    }
    const auto& iOperands = op.GetIOperands();
    for (size_t i = 0; i < iOperands.size(); ++i) {
        if (SameLogicalTensor(iOperands[i], tensor)) {
            return static_cast<int>(i);
        }
    }
    return std::nullopt;
}

std::vector<int64_t> AlignVecTileToViewRank(const std::vector<int64_t>& matmulVecTile,
                                            const std::vector<int64_t>& viewShape)
{
    if (viewShape.empty()) {
        return matmulVecTile;
    }
    if (viewShape.size() <= matmulVecTile.size()) {
        if (viewShape.size() == matmulVecTile.size()) {
            return matmulVecTile;
        }
        return std::vector<int64_t>(matmulVecTile.end() - viewShape.size(), matmulVecTile.end());
    }
    std::vector<int64_t> aligned;
    const size_t leadingDims = viewShape.size() - matmulVecTile.size();
    aligned.reserve(viewShape.size());
    for (size_t i = 0; i < leadingDims; ++i) {
        aligned.push_back(viewShape[i]);
    }
    aligned.insert(aligned.end(), matmulVecTile.begin(), matmulVecTile.end());
    return aligned;
}

TileShape GetEffectiveTileShapeForTraverse(const Operation& op, const LogicalTensorPtr& incomingTensor,
                                           const std::vector<int64_t>& viewShape)
{
    TileShape resolved;
    if (incomingTensor != nullptr) {
        if (auto inputIndex = GetInputOperandIndex(op, incomingTensor)) {
            resolved = TileShapeResolver::Instance().GetInputTileShape(op, *inputIndex);
        } else {
            resolved = TileShapeResolver::Instance().GetOutputTileShape(op, 0);
        }
    } else {
        resolved = TileShapeResolver::Instance().GetOutputTileShape(op, 0);
    }

    if (resolved.GetVecTile().valid()) {
        if (!viewShape.empty()) {
            resolved.SetVecTile(AlignVecTileToViewRank(resolved.GetVecTile().tile, viewShape));
        }
        if (IsMatmulOpcode(op.GetOpcode()) && op.GetTileShape().GetCubeTile().valid()) {
            resolved.GetCubeTile() = op.GetTileShape().GetCubeTile();
        }
        return resolved;
    }
    return TileShape::Current();
}

TileShape GetEffectiveTileShapeForTraverse(const Operation& op)
{
    return GetEffectiveTileShapeForTraverse(op, nullptr, {});
}

int64_t VecTileDimProduct(const VecTile& vecTile)
{
    int64_t product = 1;
    for (int64_t dim : vecTile.tile) {
        product *= dim;
    }
    return product;
}

std::optional<TileShape> SelectMaxTileShape(const std::vector<TileShape>& candidates)
{
    if (candidates.empty()) {
        return std::nullopt;
    }
    const TileShape* best = &candidates.front();
    int64_t bestProduct = VecTileDimProduct(best->GetVecTile());
    for (size_t i = 1; i < candidates.size(); ++i) {
        int64_t product = VecTileDimProduct(candidates[i].GetVecTile());
        if (product > bestProduct) {
            best = &candidates[i];
            bestProduct = product;
        }
    }
    return *best;
}

struct TileTraverseState {
    Operation* op{nullptr};
    LogicalTensorPtr tensor;
};

bool AdvanceTransparentConsumerHop(const Operation& curOp, const LogicalTensorPtr& tensor,
                                   std::vector<TileTraverseState>& nextLevel)
{
    bool tensorIsInput = false;
    for (const auto& input : curOp.GetIOperands()) {
        if (SameLogicalTensor(input, tensor)) {
            tensorIsInput = true;
            break;
        }
    }
    if (!tensorIsInput || curOp.GetOOperands().empty()) {
        return false;
    }
    const auto outTensor = curOp.GetOOperands()[0];
    for (auto* consumer : curOp.ConsumerOps()) {
        nextLevel.push_back({consumer, outTensor});
    }
    return true;
}

std::optional<TileShape> FindTileShapeFromConsumers(const Operation& op)
{
    if (op.GetOOperands().empty()) {
        return std::nullopt;
    }
    const auto viewOutput = op.GetOOperands()[0];
    const auto& viewShape = viewOutput->GetShape();

    std::vector<TileTraverseState> currentLevel;
    for (auto* consumer : op.ConsumerOps()) {
        currentLevel.push_back({consumer, viewOutput});
    }
    std::unordered_set<Operation*> visited;

    while (!currentLevel.empty()) {
        std::vector<TileShape> computeCandidates;
        std::vector<TileTraverseState> nextLevel;
        for (const auto& state : currentLevel) {
            Operation* curOp = state.op;
            if (curOp == nullptr || visited.count(curOp) > 0) {
                continue;
            }
            if (IsTransparentForTileTraverse(curOp->GetOpcode())) {
                if (AdvanceTransparentConsumerHop(*curOp, state.tensor, nextLevel)) {
                    visited.insert(curOp);
                }
                continue;
            }
            visited.insert(curOp);
            computeCandidates.push_back(GetEffectiveTileShapeForTraverse(*curOp, state.tensor, viewShape));
        }
        if (!computeCandidates.empty()) {
            return SelectMaxTileShape(computeCandidates);
        }
        currentLevel = std::move(nextLevel);
    }
    return std::nullopt;
}

std::optional<TileShape> FindTileShapeFromProducers(const Operation& op)
{
    std::vector<Operation*> currentLevel;
    for (auto* producer : op.ProducerOps()) {
        currentLevel.push_back(producer);
    }
    std::unordered_set<Operation*> visited;

    while (!currentLevel.empty()) {
        std::vector<TileShape> computeCandidates;
        std::vector<Operation*> nextLevel;
        for (auto* curOp : currentLevel) {
            if (curOp == nullptr || !visited.insert(curOp).second) {
                continue;
            }
            if (IsTransparentForTileTraverse(curOp->GetOpcode())) {
                for (auto* nextOp : curOp->ProducerOps()) {
                    nextLevel.push_back(nextOp);
                }
                continue;
            }
            computeCandidates.push_back(GetEffectiveTileShapeForTraverse(*curOp));
        }
        if (!computeCandidates.empty()) {
            return SelectMaxTileShape(computeCandidates);
        }
        currentLevel = std::move(nextLevel);
    }
    return std::nullopt;
}

} // namespace

namespace npu::tile_fwk {

// 不需要展开的操作码集合
// 这些操作在展开过程中保持原样，不进行 tile-level 展开
// OP_SLICE / OP_CONTRACT 仅在 ExpandFunction 展开过程中由 AddOperation 插入，本身已是 tile 级语义
const std::unordered_set<Opcode> ExpandFunction::kNotNeedExpandOps = {Opcode::OP_SLICE, Opcode::OP_CONTRACT,
                                                                      Opcode::OP_NOP, Opcode::OP_ATOMIC_RMW};
thread_local Operation* ExpandFunction::currentTileOp_ = nullptr;
thread_local std::unordered_map<ir::VarPtr, ir::VarPtr> ExpandFunction::semanticToNormal_;

ir::VarPtr ExpandFunction::GetNormalToken(const ir::VarPtr& semantic)
{
    if (semantic == nullptr) {
        return nullptr;
    }
    auto it = semanticToNormal_.find(semantic);
    if (it != semanticToNormal_.end()) {
        return it->second;
    }
    auto normal = IRContext::Get().MakeVar(semantic->name_ + "_n", ir::GetTokenType(ir::TokenKind::NORMAL),
                                           semantic->span_);
    semanticToNormal_.emplace(semantic, normal);
    return normal;
}

Status ExpandFunction::ClearIOOperand(const std::vector<OperationPtr>& tensorOperations) const
{
    for (auto& op : tensorOperations) {
        // clear consumers and producers
        for (auto& iOperand : op->GetIOperands()) {
            if (iOperand == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op:%s[%d] input is null.%s", op->GetOpcodeStr().c_str(),
                                  op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            iOperand->GetConsumers().clear();
            iOperand->GetProducers().clear();
        }
        for (auto& oOperand : op->GetOOperands()) {
            if (oOperand == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op:%s[%d] output is null.%s", op->GetOpcodeStr().c_str(),
                                  op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            oOperand->GetConsumers().clear();
            oOperand->GetProducers().clear();
        }
    }
    return SUCCESS;
}

void ExpandFunction::RefreshViewAssembleTileShapes(const std::vector<OperationPtr>& tensorOperations,
                                                   const std::unordered_set<Operation*>& skipExpandOps) const
{
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || skipExpandOps.count(opPtr.get()) > 0) {
            continue;
        }
        Operation& op = *opPtr;
        std::optional<TileShape> refreshedTileShape;
        const std::vector<int64_t>* ownShape = nullptr;
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (op.GetOOperands().empty()) {
                continue;
            }
            // VIEW 按输出 shape 切分；找不到计算 op 时用输出 shape 作为 tile。
            ownShape = &op.GetOOperands()[0]->GetShape();
            refreshedTileShape = FindTileShapeFromConsumers(op);
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (op.GetIOperands().empty()) {
                continue;
            }
            // ASSEMBLE 按输入 shape 切分；找不到计算 op 时用输入 shape 作为 tile。
            ownShape = &op.GetIOperands()[0]->GetShape();
            refreshedTileShape = FindTileShapeFromProducers(op);
        } else {
            continue;
        }

        TileShape tileShape;
        if (refreshedTileShape.has_value() && refreshedTileShape->GetVecTile().valid()) {
            tileShape = *refreshedTileShape;
        } else {
            // 未找到可继承的计算 op tile，或结果无效：用自身 shape，避免回退 Current() 被污染。
            tileShape.GetVecTile().tile = *ownShape;
        }
        op.GetTileShapeForSetting() = tileShape;
    }
}

namespace {

void ConvertTokensToNormal(Operation& dst, const Operation& src)
{
    auto convert = [](const ir::VarPtr& token) -> ir::VarPtr {
        if (token == nullptr) {
            return nullptr;
        }
        auto tokenType = std::dynamic_pointer_cast<const ir::TokenType>(token->GetType());
        if (tokenType != nullptr && tokenType->kind_ == ir::TokenKind::NORMAL) {
            return token;
        }
        return ExpandFunction::GetNormalToken(token);
    };

    dst.result_token_.clear();
    dst.tokens_.clear();
    for (const auto& token : src.result_token_) {
        auto normal = convert(token);
        if (normal != nullptr) {
            dst.result_token_.push_back(normal);
        }
    }
    for (const auto& token : src.tokens_) {
        auto normal = convert(token);
        if (normal != nullptr) {
            dst.tokens_.push_back(normal);
        }
    }
}

void CopyEmuOpAttrsForAdds(const OperationsViewer& opListPost, const OperationPtr& op, size_t opListPreSize)
{
    if (op->GetOpcode() == Opcode::OP_ADDS) {
        for (size_t i = opListPreSize; i < opListPost.size(); i++) {
            auto& newOp = opListPost[i];
            newOp.CopyAttrFrom(*op, OP_EMUOP_PREFIX);
        }
    }
}

void InheritTokenContract(const OperationsViewer& opListPost, const OperationPtr& op, size_t opListPreSize)
{
    if (!op->result_token_.empty() || !op->tokens_.empty()) {
        // VIEW/ASSEMBLE operations inserted by expansion are implementation
        // details. Only tile operations representing the original operation
        // inherit its token contract.
        for (size_t i = opListPreSize; i < opListPost.size(); i++) {
            auto& newOp = opListPost[i];
            if (newOp.GetOpcode() == op->GetOpcode()) {
                newOp.result_token_ = op->result_token_;
                newOp.tokens_ = op->tokens_;
            }
        }
    }
}

Status FinalizeExpansion(Function& function)
{
    function.BuildTensorMap();
    function.expandFunctionAccelerate = false;
    if (TokenUtils::SplitMultiProducerTokens(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SplitMultiProducerTokens failed.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace

void ExpandFunction::ProcessForNotExpandOp(Function& function, Operation& op) const
{
    auto& newOp = PassOperationUtils::AddOperation(function, op.GetOpcode(), op.GetIOperands(), op.GetOOperands(),
                                                   nullptr, ir::Span::Unknown(), false);
    if (config::EnableSlice() && op.GetOpcode() == Opcode::OP_VIEW && op.HasAttribute(OpAttributeKey::isGlobalInput)) {
        newOp.SetAttribute(OpAttributeKey::isGlobalInput, op.GetBoolAttribute(OpAttributeKey::isGlobalInput));
    }
    newOp.SetOpAttribute(op.GetOpAttribute());
    newOp.SetScopeInfo(op.GetScopeInfo());
    newOp.SetAtomicScopeId(op.GetAtomicScopeId());
    newOp.CopyAttrFrom(op, OP_EMUOP_PREFIX);
    if (op.HasAttribute(OpAttributeKey::inplaceIdx)) {
        newOp.SetAttribute(OpAttributeKey::inplaceIdx, op.GetIntAttribute(OpAttributeKey::inplaceIdx));
    }
    if (op.HasAttribute(OpAttributeKey::rmwMode)) {
        newOp.SetAttribute(OpAttributeKey::rmwMode, op.GetIntAttribute(OpAttributeKey::rmwMode));
    }
    // ATOMIC_RMW is kept as-is (not sliced into SLICE/CONTRACT). Semantic READ/WRITE
    // tokens must still become the paired NORMAL tokens so later expanded ops share
    // the same tile-graph dependency.
    if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        ConvertTokensToNormal(newOp, op);
    }
}

Status ExpandFunction::DefaultEnabledPreCheck(Function& function)
{
    ExpandFunctionChecker checker;
    return checker.DoDefaultEnabledPreCheck(function);
}

Status ExpandFunction::PostCheck(Function& function)
{
    ExpandFunctionChecker checker;
    return checker.DoPostCheck(function);
}

Status ExpandFunction::VerifyScopeInfo(Function& function, std::ostringstream& oss) const
{
    std::unordered_map<int, Operation::ScopeInfo> scopeInfoMap;
    std::unordered_map<int, std::unordered_set<CoreType>> scopeCoreTypes;
    for (auto& op : function.Operations(false)) {
        const auto& info = op.GetScopeInfo();
        if (info.scopeId == -1 && (info.allowParallelMerge || info.allowCrossScopeMerge)) {
            oss << "Op " << op.GetOpcodeStr() << "[" << op.GetOpMagic()
                << "]: allowParallelMerge and allowCrossScopeMerge must be false when scopeId is -1.";
            return FAILED;
        }
        if (info.scopeId != -1) {
            auto it = scopeInfoMap.find(info.scopeId);
            if (it != scopeInfoMap.end()) {
                const auto& existing = it->second;
                if (existing.allowParallelMerge != info.allowParallelMerge ||
                    existing.allowCrossScopeMerge != info.allowCrossScopeMerge) {
                    oss << "Op " << op.GetOpcodeStr() << "[" << op.GetOpMagic() << "]: scopeId=" << info.scopeId
                        << " has conflicting allowParallelMerge or allowCrossScopeMerge settings.";
                    return FAILED;
                }
            } else {
                scopeInfoMap[info.scopeId] = info;
            }
            scopeCoreTypes[info.scopeId].insert(op.GetCoreType());
        }
    }
    for (auto& [scopeId, coreTypes] : scopeCoreTypes) {
        if (coreTypes.count(CoreType::AIC) > 0 && coreTypes.count(CoreType::AIV) > 0) {
            if (!GraphUtils::IsCVMixPlatform()) {
                oss << "Cannot mix cube and vector op on a CV separate platform in function: " << function.GetRawName()
                    << ", please check your setting: sg_set_scope=" << scopeId;
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ExpandFunction::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "Start ExpandFunction function [%s].", function.GetRawName().c_str());
    std::ostringstream oss;
    if (VerifyScopeInfo(function, oss) != SUCCESS) {
        APASS_LOG_ERROR_C(OperationErr::OP_SCOPE_ERROR, Elements::Function,
                          "Function[%s] ScopeInfo verification failed: %s", function.GetRawName().c_str(),
                          oss.str().c_str());
        return FAILED;
    }
    bool verifyResult = true;
    for (auto& op : function.Operations(false)) {
        auto verifyOperationEntry = OpcodeManager::Inst().GetVerifyOperationEntry(op.GetOpcode());
        if (verifyOperationEntry) {
            verifyResult = verifyResult && verifyOperationEntry(function, op, oss);
        }
    }
    if (!verifyResult) {
        APASS_LOG_ERROR_F(Elements::Function, "Function[%s] ExpandFunction failed: %s", function.GetRawName().c_str(),
                          oss.str().c_str());
        return FAILED;
    }
    if (Expandfunction(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Function[%s] ExpandFunction failed.", function.GetRawName().c_str());
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "Function[%s] operation size is: %zu after expansion.",
                     function.GetMagicName().c_str(), function.Operations(false).size());
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    APASS_LOG_INFO_F(Elements::Function, "End ExpandFunction function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

namespace {
struct IncastColumnSource {
    LogicalTensorPtr tensor;
    std::vector<SymbolicScalar> offset;
    std::vector<SymbolicScalar> valid;
};

bool GetColumnSegment(Operation& writer, const LogicalTensorPtr& concat, int64_t& begin)
{
    if (writer.GetOpcode() != Opcode::OP_ASSEMBLE || writer.GetIOperands().size() != 1 || !writer.tokens_.empty() ||
        !writer.result_token_.empty()) {
        return false;
    }
    auto assemble = std::dynamic_pointer_cast<AssembleOpAttribute>(writer.GetOpAttribute());
    auto piece = writer.GetIOperands().front();
    bool needCopy = false;
    if ((writer.GetAttr<bool>("NeedCopy", needCopy) && needCopy) || writer.HasAttribute(OpAttributeKey::inplaceIdx) ||
        writer.HasAttribute(OpAttributeKey::rmwMode) || assemble == nullptr || piece->GetShape().size() != 2 ||
        piece->Datatype() != concat->Datatype() || piece->Format() != concat->Format() ||
        piece->GetShape()[0] != concat->GetShape()[0]) {
        return false;
    }
    auto to = assemble->GetToDynOffset().empty() ? SymbolicScalar::FromConcrete(assemble->GetToOffset()) :
                                                   assemble->GetToDynOffset();
    if (to.size() != 2 || !to[0].Simplify().ConcreteValid() || to[0].Simplify().Concrete() != 0 ||
        !to[1].Simplify().ConcreteValid()) {
        return false;
    }
    begin = to[1].Simplify().Concrete();
    return true;
}

bool IsColumnSourceView(Operation& view, const std::shared_ptr<ViewOpAttribute>& attr, const LogicalTensorPtr& piece)
{
    bool needCopy = false;
    if (view.GetOpcode() != Opcode::OP_VIEW || attr == nullptr || view.GetIOperands().size() != 1 ||
        (view.GetAttr<bool>("NeedCopy", needCopy) && needCopy) || view.HasAttribute(OpAttributeKey::inplaceIdx) ||
        view.HasAttribute(OpAttributeKey::rmwMode) ||
        (attr->GetTo() != MemoryType::MEM_UNKNOWN && attr->GetTo() != MemoryType::MEM_DEVICE_DDR) ||
        !view.tokens_.empty() || !view.result_token_.empty()) {
        return false;
    }
    const auto& input = view.GetIOperands().front();
    if (input->GetShape().size() != 2 || input->Datatype() != piece->Datatype() || input->Format() != piece->Format()) {
        return false;
    }
    for (const auto& tensor : {piece, input}) {
        for (auto memory : {tensor->GetMemoryTypeOriginal(), tensor->GetMemoryTypeToBe()}) {
            if (memory != MemoryType::MEM_UNKNOWN && memory != MemoryType::MEM_DEVICE_DDR) {
                return false;
            }
        }
    }
    return true;
}

bool UpdateColumnSourceOffset(ViewOpAttribute& attr, IncastColumnSource& source)
{
    auto from = attr.GetFromDynOffset().empty() ? SymbolicScalar::FromConcrete(attr.GetFromOffset()) :
                                                  attr.GetFromDynOffset();
    if (from.size() != 2) {
        return false;
    }
    for (size_t dim = 0; dim < 2; ++dim) {
        const auto& viewValid = attr.GetToDynValidShape();
        if (!viewValid.empty()) {
            if (viewValid.size() != 2) {
                return false;
            }
            source.valid[dim] = source.valid[dim].Min((viewValid[dim] - source.offset[dim]).Max(0)).Simplify();
        }
        source.offset[dim] = (source.offset[dim] + from[dim]).Simplify();
    }
    return true;
}

bool TraceColumnSourceToIncast(Function& function, LogicalTensorPtr piece, IncastColumnSource& source)
{
    std::unordered_set<LogicalTensor*> seen;
    while (!piece->GetProducers().empty()) {
        if (piece->GetProducers().size() != 1 || !seen.insert(piece.get()).second) {
            return false;
        }
        auto* view = *piece->GetProducers().begin();
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
        if (!IsColumnSourceView(*view, attr, piece) || !UpdateColumnSourceOffset(*attr, source)) {
            return false;
        }
        piece = view->GetIOperands().front();
    }
    if (std::find(function.GetIncast().begin(), function.GetIncast().end(), piece) == function.GetIncast().end()) {
        return false;
    }
    source.tensor = piece;
    return true;
}

bool CoversAllColumns(std::vector<std::pair<int64_t, int64_t>>& segments, int64_t columns)
{
    std::sort(segments.begin(), segments.end());
    int64_t end = 0;
    for (const auto& segment : segments) {
        if (segment.first != end || segment.second <= segment.first) {
            return false;
        }
        end = segment.second;
    }
    return end == columns;
}

bool FindIncastColumnSource(Function& function, const LogicalTensorPtr& concat,
                            const std::vector<SymbolicScalar>& offset, int64_t width, IncastColumnSource& source)
{
    int64_t column = offset[1].Simplify().Concrete();
    std::vector<std::pair<int64_t, int64_t>> segments;
    for (auto* writer : concat->GetProducers()) {
        int64_t begin = 0;
        if (!GetColumnSegment(*writer, concat, begin)) {
            return false;
        }
        auto piece = writer->GetIOperands().front();
        segments.emplace_back(begin, begin + piece->GetShape()[1]);
        if (column < begin || column + width > begin + piece->GetShape()[1]) {
            continue;
        }
        if (source.tensor != nullptr) { // Overlapping writers cannot be resolved independently.
            return false;
        }
        source.offset = offset;
        source.offset[1] = source.offset[1] - begin;
        if (!TraceColumnSourceToIncast(function, piece, source)) {
            return false;
        }
    }
    return source.tensor != nullptr && width > 0 && CoversAllColumns(segments, concat->GetShape()[1]);
}

// Resolve a cube tile wholly contained in one input column segment directly to
// its incast. This avoids materializing a concatenated DDR row merely to load it.
void ResolveIncastColumnLoad(Function& function, Operation& load)
{
    if (load.GetOpcode() != Opcode::OP_SLICE || load.GetIOperands().size() != 1 || load.GetOOperands().size() != 1 ||
        !load.tokens_.empty() || !load.result_token_.empty()) {
        return;
    }
    auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(load.GetOpAttribute());
    auto concat = load.GetIOperands().front();
    if (attr == nullptr || attr->GetTo() != MemoryType::MEM_L1 || concat->GetShape().size() != 2 ||
        concat->GetProducers().empty() || concat->GetConsumers().empty() || attr->GetFromOffset().size() != 2) {
        return;
    }
    auto offset = attr->GetFromDynOffset().empty() ? SymbolicScalar::FromConcrete(attr->GetFromOffset()) :
                                                     attr->GetFromDynOffset();
    if (offset.size() != 2 || !offset[1].Simplify().ConcreteValid()) {
        return;
    }
    int64_t column = offset[1].Simplify().Concrete();
    int64_t width = std::min(load.GetOOperands().front()->GetShape()[1], concat->GetShape()[1] - column);
    IncastColumnSource source;
    source.valid = attr->GetToDynValidShape().empty() ? load.GetOOperands().front()->GetDynValidShape() :
                                                        attr->GetToDynValidShape();
    if (source.valid.size() != 2 || !FindIncastColumnSource(function, concat, offset, width, source)) {
        return;
    }
    auto replacement = std::make_shared<ViewOpAttribute>(*attr);
    Offset concreteOffset(source.offset.size(), 0);
    for (size_t dim = 0; dim < source.offset.size(); ++dim) {
        auto simplified = source.offset[dim].Simplify();
        if (simplified.ConcreteValid()) {
            concreteOffset[dim] = simplified.Concrete();
        }
    }
    replacement->SetFromOffset(concreteOffset, source.offset);
    replacement->SetToDynValidShape(source.valid);
    load.ReplaceIOperand(0, source.tensor);
    load.SetOpAttribute(replacement);
}

bool IsDdrAlias(Operation* op)
{
    if (op == nullptr || !op->tokens_.empty() || !op->result_token_.empty() || op->GetIOperands().size() != 1 ||
        op->GetOOperands().size() != 1) {
        return false;
    }
    bool needCopy = false;
    if ((op->GetAttr<bool>("NeedCopy", needCopy) && needCopy) || op->HasAttribute(OpAttributeKey::inplaceIdx) ||
        op->HasAttribute(OpAttributeKey::rmwMode)) {
        return false;
    }
    const auto& input = op->GetIOperands().front();
    const auto& output = op->GetOOperands().front();
    if (input == nullptr || output == nullptr || input->Datatype() != output->Datatype() ||
        input->Format() != output->Format() || input->GetShape().size() != output->GetShape().size()) {
        return false;
    }
    for (const auto& tensor : {input, output}) {
        for (auto memory : {tensor->GetMemoryTypeOriginal(), tensor->GetMemoryTypeToBe()}) {
            if (memory != MemoryType::MEM_UNKNOWN && memory != MemoryType::MEM_DEVICE_DDR) {
                return false;
            }
        }
    }
    auto view = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    auto assemble = std::dynamic_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
    auto memory = view != nullptr ? view->GetTo() : assemble != nullptr ? assemble->GetFrom() : MemoryType::MEM_UB;
    return memory == MemoryType::MEM_UNKNOWN || memory == MemoryType::MEM_DEVICE_DDR;
}

bool CollectIncastViewChain(Function& function, LogicalTensorPtr input, std::unordered_set<Operation*>& viewChain)
{
    while (!input->GetProducers().empty()) {
        if (input->GetProducers().size() != 1) {
            return false;
        }
        auto* view = *input->GetProducers().begin();
        if (view->GetOpcode() != Opcode::OP_VIEW || !IsDdrAlias(view) || !viewChain.insert(view).second) {
            return false;
        }
        input = view->GetIOperands().front();
    }
    return !viewChain.empty() &&
           std::find(function.GetIncast().begin(), function.GetIncast().end(), input) != function.GetIncast().end();
}

bool CollectIncastGatherWriters(Function& function, const LogicalTensorPtr& output,
                                std::unordered_set<Operation*>& gatherOps)
{
    for (auto* writer : output->GetProducers()) {
        if (writer->GetOpcode() != Opcode::OP_ASSEMBLE || !IsDdrAlias(writer)) {
            return false;
        }
        gatherOps.insert(writer);
        std::unordered_set<Operation*> viewChain;
        if (!CollectIncastViewChain(function, writer->GetIOperands().front(), viewChain)) {
            return false;
        }
        gatherOps.insert(viewChain.begin(), viewChain.end());
    }
    return true;
}

bool ShouldKeepGatherL1Alias(const LogicalTensorPtr& output)
{
    // Preserve L1 aliases for block gathers. A pure column concat of complete
    // input rows needs a cube SLICE boundary to avoid reloading each full row.
    bool completeRows = true;
    bool blockGather = !output->GetShape().empty();
    for (auto* writer : output->GetProducers()) {
        const auto& input = writer->GetIOperands().front();
        blockGather = blockGather && input->GetShape().front() < output->GetShape().front();
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(writer->GetOpAttribute());
        if (attr == nullptr || attr->GetToOffset().size() != output->GetShape().size() ||
            (!attr->GetToDynOffset().empty() && attr->GetToDynOffset().size() != output->GetShape().size())) {
            completeRows = false;
            blockGather = false;
            break;
        }
        for (size_t dim = 1; dim < output->GetShape().size(); ++dim) {
            auto offset = (attr->GetToDynOffset().empty() ? SymbolicScalar(attr->GetToOffset()[dim]) :
                                                            attr->GetToDynOffset()[dim])
                              .Simplify();
            if (input->GetShape()[dim] != output->GetShape()[dim] || !offset.ConcreteValid() ||
                offset.Concrete() != 0) {
                completeRows = false;
                break;
            }
        }
    }
    return completeRows || blockGather;
}

void CollectAssembledIncastGatherOps(Function& function, const std::vector<OperationPtr>& tensorOperations,
                                     std::unordered_set<Operation*>& skipOps,
                                     std::unordered_set<LogicalTensor*>& incastGatherOutputs)
{
    std::unordered_set<LogicalTensor*> visitedOutputs;
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_ASSEMBLE || opPtr->GetOOperands().size() != 1) {
            continue;
        }
        const auto& output = opPtr->GetOOperands().front();
        // A shared L1 assembly would bind all cube consumers into one supernode,
        // even when a vector dependency separates those consumers.
        if (output == nullptr || !visitedOutputs.insert(output.get()).second || function.IsFromOutCast(output) ||
            function.IsFromInCast(output) || output->GetConsumers().size() != 1) {
            continue;
        }
        bool eligible = true;
        for (auto* consumer : output->GetConsumers()) {
            if (!IsMatmulOpcode(consumer->GetOpcode()) || !consumer->tokens_.empty() ||
                !consumer->result_token_.empty()) {
                eligible = false;
                break;
            }
        }
        std::unordered_set<Operation*> gatherOps;
        if (!eligible || !CollectIncastGatherWriters(function, output, gatherOps)) {
            continue;
        }
        skipOps.insert(gatherOps.begin(), gatherOps.end());
        if (ShouldKeepGatherL1Alias(output)) {
            incastGatherOutputs.insert(output.get());
        }
    }
}

void CollectDynamicIncastViews(Function& function, const std::vector<OperationPtr>& tensorOperations,
                               std::unordered_set<Operation*>& skipOps)
{
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_VIEW || !IsDdrAlias(opPtr.get())) {
            continue;
        }
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(opPtr->GetOpAttribute());
        if (attr == nullptr ||
            !std::any_of(attr->GetFromDynOffset().begin(), attr->GetFromDynOffset().end(),
                         [](const SymbolicScalar& offset) { return !offset.Simplify().ConcreteValid(); })) {
            continue;
        }
        std::unordered_set<Operation*> viewChain;
        if (CollectIncastViewChain(function, opPtr->GetOOperands().front(), viewChain)) {
            skipOps.insert(opPtr.get());
        }
    }
}

bool IsFullShapeZeroOffsetView(Operation& op)
{
    const auto& input = op.GetIOperands().front();
    const auto& output = op.GetOOperands().front();
    auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
    return input->GetShape() == output->GetShape() && !output->GetConsumers().empty() && attr != nullptr &&
           std::all_of(attr->GetFromOffset().begin(), attr->GetFromOffset().end(),
                       [](int64_t offset) { return offset == 0; }) &&
           std::all_of(attr->GetFromDynOffset().begin(), attr->GetFromDynOffset().end(),
                       [](const SymbolicScalar& offset) {
                           auto simplified = offset.Simplify();
                           return simplified.ConcreteValid() && simplified.Concrete() == 0;
                       });
}

bool CollectCubeAliasConsumers(const LogicalTensorPtr& output, std::unordered_set<Operation*>& seenConsumers)
{
    std::vector<Operation*> consumers(output->GetConsumers().begin(), output->GetConsumers().end());
    for (size_t i = 0; i < consumers.size(); ++i) {
        auto* consumer = consumers[i];
        if (!seenConsumers.insert(consumer).second) {
            continue;
        }
        if (consumer->GetOpcode() == Opcode::OP_VIEW && IsDdrAlias(consumer)) {
            const auto& next = consumer->GetOOperands().front()->GetConsumers();
            if (next.empty()) {
                return false;
            }
            consumers.insert(consumers.end(), next.begin(), next.end());
        } else if (!IsMatmulOpcode(consumer->GetOpcode()) || !consumer->tokens_.empty() ||
                   !consumer->result_token_.empty()) {
            return false;
        }
    }
    return true;
}

bool CollectComputedAliases(const LogicalTensorPtr& input, std::unordered_set<Operation*>& aliases)
{
    for (auto* writer : input->GetProducers()) {
        if (writer->GetOpcode() == Opcode::OP_ASSEMBLE && IsDdrAlias(writer) && input->GetConsumers().size() == 1) {
            aliases.insert(writer);
            const auto& producers = writer->GetIOperands().front()->GetProducers();
            if (producers.size() != 1) {
                return false;
            }
            writer = *producers.begin();
        }
        if ((writer->GetOpcode() != Opcode::OP_CAST && writer->GetOpcode() != Opcode::OP_REGISTER_COPY) ||
            !writer->tokens_.empty() || !writer->result_token_.empty()) {
            return false;
        }
    }
    return true;
}

// A full-shape alias between a computed assembly and cube consumers should not
// retile data that its producers have already tiled.
void CollectComputedViewSkipExpandOps(const std::vector<OperationPtr>& tensorOperations,
                                      std::unordered_set<Operation*>& skipOps)
{
    for (const auto& opPtr : tensorOperations) {
        if (opPtr == nullptr || opPtr->GetOpcode() != Opcode::OP_VIEW || !IsDdrAlias(opPtr.get()) ||
            !IsFullShapeZeroOffsetView(*opPtr)) {
            continue;
        }
        const auto& input = opPtr->GetIOperands().front();
        // A same-shape zero-offset view only restricts the valid region. Retiling
        // a computed DDR result would introduce storage for a pure alias.
        if (input->GetProducers().size() == 1 && IsMatmulOpcode((*input->GetProducers().begin())->GetOpcode())) {
            skipOps.insert(opPtr.get());
            continue;
        }
        std::unordered_set<Operation*> seenConsumers;
        if (!CollectCubeAliasConsumers(opPtr->GetOOperands().front(), seenConsumers) || input->GetProducers().empty()) {
            continue;
        }
        std::unordered_set<Operation*> aliases{opPtr.get()};
        if (!CollectComputedAliases(input, aliases)) {
            continue;
        }
        skipOps.insert(aliases.begin(), aliases.end());
        for (auto* consumer : seenConsumers) {
            if (consumer->GetOpcode() == Opcode::OP_VIEW) {
                skipOps.insert(consumer);
            }
        }
    }
}

// Keep input gather descriptions intact until cube loads are tiled. Independently
// tiling their views and assembles creates copy-required intermediate buffers.
std::unordered_set<Operation*> CollectIncastGatherSkipExpandOps(Function& function,
                                                                const std::vector<OperationPtr>& tensorOperations,
                                                                std::unordered_set<LogicalTensor*>& incastGatherOutputs)
{
    std::unordered_set<Operation*> skipOps;
    CollectAssembledIncastGatherOps(function, tensorOperations, skipOps, incastGatherOutputs);
    CollectDynamicIncastViews(function, tensorOperations, skipOps);
    CollectComputedViewSkipExpandOps(tensorOperations, skipOps);
    return skipOps;
}

void MergeSkipExpandOps(const char* rule, const std::unordered_set<Operation*>& candidates,
                        std::unordered_set<Operation*>& skipExpandOps)
{
    for (auto* op : candidates) {
        if (op == nullptr || !skipExpandOps.insert(op).second) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "ExpandFunction skip VIEW/ASSEMBLE op=%d opcode=%s rule=%s.",
                          op->GetOpMagic(), op->GetOpcodeStr().c_str(), rule);
    }
}

void PreserveIncastGatherLoads(const OperationsViewer& operations, size_t begin,
                               const std::unordered_set<LogicalTensor*>& incastGatherOutputs)
{
    for (size_t i = begin; i < operations.size(); ++i) {
        auto& load = operations[i];
        if (load.GetOpcode() != Opcode::OP_SLICE || load.GetIOperands().size() != 1 ||
            incastGatherOutputs.count(load.GetIOperands().front().get()) == 0) {
            continue;
        }
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(load.GetOpAttribute());
        // A VIEW load lets memory inference push L1 through the gather aliases
        // instead of materializing an intermediate DDR buffer.
        if (attr != nullptr && attr->GetTo() == MemoryType::MEM_L1 && load.tokens_.empty() &&
            load.result_token_.empty()) {
            load.SetOpCode(Opcode::OP_VIEW);
        }
    }
}
} // namespace

std::unordered_set<Operation*> ExpandFunction::CollectSkipExpandOps(
    Function& function, const std::vector<OperationPtr>& tensorOperations,
    std::unordered_set<LogicalTensor*>& incastGatherOutputs) const
{
    auto skipExpandOps = CollectViewAssembleSkipExpandOps(tensorOperations);
    auto ssaDstViewSkipOps = CollectAssembleSsaDstViewSkipExpandOps(tensorOperations);
    MergeSkipExpandOps("assemble_ssa_dst_view", ssaDstViewSkipOps, skipExpandOps);
    const bool expandViewAssemble = config::EnableSlice();
    if (!expandViewAssemble) {
        for (const auto& opPtr : tensorOperations) {
            if (opPtr == nullptr) {
                continue;
            }
            if (opPtr->GetOpcode() == Opcode::OP_VIEW || opPtr->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (skipExpandOps.insert(opPtr.get()).second) {
                    APASS_LOG_DEBUG_F(Elements::Operation,
                                      "ExpandFunction skip VIEW/ASSEMBLE op=%d opcode=%s rule=legacy_disable_slice.",
                                      opPtr->GetOpMagic(), opPtr->GetOpcodeStr().c_str());
                }
            }
        }
    }
    if (expandViewAssemble) {
        auto outcastSkipOps = CollectAssembleOutcastSkipExpandOps(function, tensorOperations);
        MergeSkipExpandOps("assemble_outcast", outcastSkipOps, skipExpandOps);
        auto gatherParamSkipOps = CollectGatherInUbParamViewSkipOps(tensorOperations);
        MergeSkipExpandOps("gather_ub_param_view", gatherParamSkipOps, skipExpandOps);
        auto dynamicShapeSkipOps = CollectDynamicShapeViewAssembleSkipExpandOps(tensorOperations);
        MergeSkipExpandOps("dynamic_shape_no_token", dynamicShapeSkipOps, skipExpandOps);
        auto gatherInL1SrcSkipOps = CollectAssembleGatherInL1SrcSkipExpandOps(tensorOperations);
        MergeSkipExpandOps("gather_in_l1_src", gatherInL1SrcSkipOps, skipExpandOps);
        auto incastGatherSkipOps = CollectIncastGatherSkipExpandOps(function, tensorOperations, incastGatherOutputs);
        MergeSkipExpandOps("incast_gather_alias", incastGatherSkipOps, skipExpandOps);
        RefreshViewAssembleTileShapes(tensorOperations, skipExpandOps);
    }
    return skipExpandOps;
}

bool ExpandFunction::ShouldKeepOpUnexpanded(const OperationPtr& op, const std::unordered_set<Operation*>& skipExpandOps)
{
    if (config::EnableSlice() && op->GetOpcode() == Opcode::OP_VIEW &&
        op->HasAttribute(OpAttributeKey::isGlobalInput)) {
        return true;
    }
    if (kNotNeedExpandOps.count(op->GetOpcode())) {
        return true;
    }
    return skipExpandOps.count(op.get()) > 0;
}

Status ExpandFunction::ExpandTensorOperations(Function& function, const std::vector<OperationPtr>& tensorOperations,
                                              const std::unordered_set<Operation*>& skipExpandOps,
                                              const std::unordered_set<LogicalTensor*>& incastGatherOutputs) const
{
    for (auto& op : tensorOperations) {
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Encountered null operation in function.");
            return FAILED;
        }
        if (op->GetOpcode() == Opcode::OP_PRINT) {
            continue;
        }
        if (ShouldKeepOpUnexpanded(op, skipExpandOps)) {
            ProcessForNotExpandOp(function, *op);
            continue;
        }

        ir::Span::SetCurrent(op->GetSpan());
        config::SetSemanticLabel(op->GetSemanticLabel());
        size_t opListPreSize = function.Operations(false).size();
        if (ExpandOperation(function, *op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExpandOperation failed.");
            return FAILED;
        }
        auto opListPost = function.Operations(false);
        if (IsMatmulOpcode(op->GetOpcode())) {
            PreserveIncastGatherLoads(opListPost, opListPreSize, incastGatherOutputs);
        }
        CopyEmuOpAttrsForAdds(opListPost, op, opListPreSize);
        InheritTokenContract(opListPost, op, opListPreSize);
        ir::Span::ClearCurrent();
    }
    return SUCCESS;
}

Status ExpandFunction::Expandfunction(Function& function) const
{
    if (!function.IsGraphType(GraphType::TENSOR_GRAPH)) {
        APASS_LOG_INFO_F(Elements::Function, "Function %s is not static tensor graph, skip expanding.",
                         function.GetRawName().c_str());
        return SUCCESS;
    }
    function.expandFunctionAccelerate = true;
    SemanticToNormalGuard semanticToNormalGuard;
    function.SetGraphType(GraphType::TILE_GRAPH);

    std::vector<OperationPtr> tensorOperations;
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        tensorOperations.emplace_back(operationViewer.operations_[i]);
    }

    function.ResetOperations();
    std::unordered_set<LogicalTensor*> incastGatherOutputs;
    auto skipExpandOps = CollectSkipExpandOps(function, tensorOperations, incastGatherOutputs);
    if (ClearIOOperand(tensorOperations) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ClearIOOperand failed.");
        return FAILED;
    }

    if (ExpandTensorOperations(function, tensorOperations, skipExpandOps, incastGatherOutputs) != SUCCESS) {
        return FAILED;
    }
    if (config::EnableSlice()) {
        // FinalizeExpansion rebuilds token dependencies before the final sort.
        // Until then, the dependency cache still refers to tensor operations.
        for (auto& op : function.Operations(false)) {
            ResolveIncastColumnLoad(function, op);
        }
    }
    return FinalizeExpansion(function);
}

Status ExpandFunction::ExpandOperation(Function& function, Operation& op) const
{
    const auto& info = op.GetScopeInfo();
    std::vector<int64_t> scopeVec = {static_cast<int64_t>(info.scopeId), static_cast<int64_t>(info.allowParallelMerge),
                                     static_cast<int64_t>(info.allowCrossScopeMerge)};
    config::SetPassOption(SG_SET_SCOPE, scopeVec);
    config::SetAtomicScopeId(op.GetAtomicScopeId());
    struct ScopeConfigGuard {
        ~ScopeConfigGuard()
        {
            config::SetPassOption(SG_SET_SCOPE, std::vector<int64_t>{-1, 0, 0});
            config::SetAtomicScopeId(-1);
        }
    } scopeConfigGuard;
    CurrentTileOpGuard currentTileOpGuard(op);
    ExpandOperationInto(function, op.GetTileShape(), op.GetOpcode(), op.GetIOperands(), op.GetOOperands(), op);
    return SUCCESS;
}

void ExpandFunction::DoHealthCheckBefore(Function& function, const std::string& folderPath)
{
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph START");
    std::string fileName = GetDumpFilePrefix(function, true);
    HealthCheckTensorGraph(function, folderPath, fileName);
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph END");
}

} // namespace npu::tile_fwk
