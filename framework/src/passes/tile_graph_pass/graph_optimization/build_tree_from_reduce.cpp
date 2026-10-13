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
 * \file build_tree_from_reduce.cpp
 * \brief Convert atomic-add assemble fan-in into an explicit balanced add tree.
 */

#include "build_tree_from_reduce.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface/operation/attribute.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/platform.h"

#define MODULE_NAME "BuildTreeFromReduce"

namespace npu::tile_fwk {
namespace {
constexpr size_t MIN_TREE_GROUP_SIZE = 2;
constexpr size_t ADD_TREE_FAN_IN = 2;
} // namespace

bool BuildTreeFromReducePass::IsTreeAssembleOpcode(Opcode opcode)
{
    return IsAssembleLike(opcode) || (!config::EnableSlice() && opcode == Opcode::OP_ASSEMBLE_SSA);
}

bool BuildTreeFromReducePass::IsAtomicAddCandidate(const Operation& op)
{
    if (!IsTreeAssembleOpcode(op.GetOpcode()) || !op.HasAttr(RMW_MODE_ATTR_ADD) || op.GetIOperands().size() != 1 ||
        op.GetOOperands().size() != 1) {
        return false;
    }
    // ProcessAtomic attaches one of these provenance attributes to every
    // atomic-add Assemble/Contract it creates.  Accept either source here:
    // ReduceAcc and explicit AtomicRMW are both reducible when they form an
    // independent fan-in group.
    return op.HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR) || op.HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR);
}

bool BuildTreeFromReducePass::TensorDependsOn(const LogicalTensorPtr& tensor, const LogicalTensorPtr& dependency,
                                              std::unordered_set<const LogicalTensor*>& visited)
{
    if (tensor == nullptr || dependency == nullptr) {
        return false;
    }
    if (tensor == dependency || tensor->GetRawMagic() == dependency->GetRawMagic()) {
        return true;
    }
    if (!visited.insert(tensor.get()).second) {
        return false;
    }
    for (const auto* producer : tensor->GetProducers()) {
        if (producer == nullptr || producer->IsDeleted()) {
            continue;
        }
        for (const auto& input : producer->GetIOperands()) {
            if (TensorDependsOn(input, dependency, visited)) {
                return true;
            }
        }
    }
    return false;
}

bool BuildTreeFromReducePass::HasAtomicTargetReadDependency(const Operation& candidate, const LogicalTensorPtr& output)
{
    if (candidate.GetIOperands().size() != 1 || output == nullptr) {
        return true;
    }
    std::unordered_set<const LogicalTensor*> visited;
    return TensorDependsOn(candidate.GetIOperands().front(), output, visited);
}

bool BuildTreeFromReducePass::IsSameDynShape(const std::vector<SymbolicScalar>& lhs,
                                             const std::vector<SymbolicScalar>& rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].Dump() != rhs[i].Dump()) {
            return false;
        }
    }
    return true;
}

bool BuildTreeFromReducePass::IsSameAssembleRegion(const Operation& lhs, const Operation& rhs)
{
    auto lhsAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(lhs.GetOpAttribute());
    auto rhsAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(rhs.GetOpAttribute());
    if (lhsAttr == nullptr || rhsAttr == nullptr) {
        return lhsAttr == rhsAttr;
    }
    return lhsAttr->GetToOffset() == rhsAttr->GetToOffset() &&
           IsSameDynShape(lhsAttr->GetToDynOffset(), rhsAttr->GetToDynOffset());
}

bool BuildTreeFromReducePass::CanBuildOneTree(const Operation& lhs, const Operation& rhs)
{
    if (lhs.GetScopeId() != rhs.GetScopeId()) {
        return false;
    }
    if (lhs.GetIOperands().size() != 1 || rhs.GetIOperands().size() != 1 || !IsSameAssembleRegion(lhs, rhs)) {
        return false;
    }
    // Assemble fan-in is reducible only when all partials target the same
    // logical tensor.  Matching shape/offset alone is insufficient: separate
    // tensors can have identical regions but must never be added together.
    if (lhs.GetOOperands().size() != 1 || rhs.GetOOperands().size() != 1 || lhs.GetOOperands().front() == nullptr ||
        rhs.GetOOperands().front() == nullptr ||
        lhs.GetOOperands().front()->GetRawMagic() != rhs.GetOOperands().front()->GetRawMagic()) {
        return false;
    }
    const auto& lhsInput = lhs.GetIOperands().front();
    const auto& rhsInput = rhs.GetIOperands().front();
    return lhsInput != nullptr && rhsInput != nullptr && lhsInput->Datatype() == rhsInput->Datatype() &&
           lhsInput->GetShape() == rhsInput->GetShape() && lhsInput->Format() == rhsInput->Format() &&
           IsSameDynShape(lhsInput->GetDynValidShape(), rhsInput->GetDynValidShape());
}

bool BuildTreeFromReducePass::TryGetStaticTensorBytes(const LogicalTensorPtr& tensor, uint64_t& bytes)
{
    if (tensor == nullptr || tensor->GetShape().empty()) {
        return false;
    }

    uint64_t elements = 1;
    for (const auto dim : tensor->GetShape()) {
        if (dim <= 0) {
            return false;
        }
        const auto uDim = static_cast<uint64_t>(dim);
        if (elements > std::numeric_limits<uint64_t>::max() / uDim) {
            return false;
        }
        elements *= uDim;
    }

    const auto dataBytes = static_cast<uint64_t>(BytesOf(tensor->Datatype()));
    if (dataBytes == 0) {
        return false;
    }
    if (elements > std::numeric_limits<uint64_t>::max() / dataBytes) {
        return false;
    }
    bytes = elements * dataBytes;
    return true;
}

bool BuildTreeFromReducePass::HasOnlyImmediateDynOffset(const std::vector<SymbolicScalar>& dynOffset)
{
    return std::all_of(dynOffset.begin(), dynOffset.end(),
                       [](const SymbolicScalar& scalar) { return scalar.IsImmediate(); });
}

// If the dynamic offset is already fully concrete, treat it as static offset
// instead of manufacturing a fake dynamic attribute on split outputs.
void BuildTreeFromReducePass::NormalizeStaticDynOffset(Offset& staticOffset, std::vector<SymbolicScalar>& dynOffset)
{
    if (dynOffset.empty() || dynOffset.size() != staticOffset.size() || !HasOnlyImmediateDynOffset(dynOffset)) {
        return;
    }
    staticOffset = AddStaticOffset(staticOffset, SymbolicScalar::Concrete(dynOffset, 0));
    dynOffset.clear();
}

std::vector<SymbolicScalar> BuildTreeFromReducePass::AddStaticOffsetToDynOffset(
    const std::vector<SymbolicScalar>& baseDynOffset, const Offset& concreteOffset)
{
    if (baseDynOffset.empty()) {
        return {};
    }
    auto dynOffset = baseDynOffset;
    for (size_t i = 0; i < dynOffset.size() && i < concreteOffset.size(); ++i) {
        dynOffset[i] = (dynOffset[i] + concreteOffset[i]).Simplify();
    }
    return dynOffset;
}

Offset BuildTreeFromReducePass::AddStaticOffset(const Offset& baseOffset, const Offset& concreteOffset)
{
    auto offset = baseOffset;
    for (size_t i = 0; i < offset.size() && i < concreteOffset.size(); ++i) {
        offset[i] += concreteOffset[i];
    }
    return offset;
}

std::vector<SymbolicScalar> BuildTreeFromReducePass::MakeSliceDynValidShape(
    const LogicalTensorPtr& input, const Offset& inputOffset, const std::vector<SymbolicScalar>& inputDynOffset,
    const Shape& sliceShape)
{
    if (input == nullptr) {
        return {};
    }
    if (input->GetDynValidShape().empty()) {
        return SymbolicScalar::FromConcrete(sliceShape);
    }
    return GetViewValidShape(input->GetDynValidShape(), inputOffset, inputDynOffset, sliceShape);
}

bool BuildTreeFromReducePass::ValidateSplitRegionSource(const Operation& anchor, const LogicalTensorPtr& input,
                                                        uint64_t& inputBytes)
{
    if (anchor.GetIOperands().size() != 1 || input == nullptr || input->GetShape().empty() ||
        input->GetShape().front() < MIN_SPLITTABLE_DIM) {
        return false;
    }
    if (!TryGetStaticTensorBytes(input, inputBytes)) {
        return false;
    }
    const auto ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    return ubSize != 0 && inputBytes > ubSize / 2;
}

std::vector<int64_t> BuildTreeFromReducePass::DeriveAddVecTile(const Operation& anchor, const LogicalTensorPtr& lhs,
                                                               uint64_t ubSize)
{
    auto vecTile = lhs->GetShape();
    const auto bytes = static_cast<uint64_t>(BytesOf(lhs->Datatype()));
    const uint64_t maxTileElements = bytes > 0 ? ubSize / (ADD_TILE_BUFFER_COUNT * bytes) : 0;
    if (maxTileElements == 0) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "BuildTree ADD[%d] keeps the original tile shape because UB budget is insufficient.",
                         anchor.GetOpMagic());
        return vecTile;
    }
    uint64_t suffixElements = 1;
    for (size_t i = vecTile.size(); i > 0; --i) {
        const size_t dim = i - 1;
        const uint64_t shapeDim = static_cast<uint64_t>(std::max<int64_t>(lhs->GetShape()[dim], 1));
        const uint64_t allowed = suffixElements < maxTileElements ? maxTileElements / suffixElements : 1;
        vecTile[dim] = static_cast<int64_t>(std::max<uint64_t>(1, std::min(shapeDim, allowed)));
        suffixElements *= static_cast<uint64_t>(vecTile[dim]);
    }
    return vecTile;
}

bool BuildTreeFromReducePass::TryBuildSplitRegions(const Operation& anchor,
                                                   std::array<SplitRegion, SPLIT_REGION_COUNT>& regions)
{
    const auto& input = anchor.GetIOperands().front();
    uint64_t inputBytes = 0;
    if (!ValidateSplitRegionSource(anchor, input, inputBytes)) {
        return false;
    }

    const auto rank = input->GetShape().size();
    if (!input->GetDynValidShape().empty() && input->GetDynValidShape().size() != rank) {
        return false;
    }

    const auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(anchor.GetOpAttribute());
    if (assembleAttr == nullptr || assembleAttr->GetToOffset().size() != rank ||
        (!assembleAttr->GetToDynOffset().empty() && assembleAttr->GetToDynOffset().size() != rank)) {
        return false;
    }

    auto leftShape = input->GetShape();
    auto rightShape = input->GetShape();
    leftShape[0] = input->GetShape()[0] / SPLIT_REGION_COUNT;
    rightShape[0] = input->GetShape()[0] - leftShape[0];
    if (leftShape[0] <= 0 || rightShape[0] <= 0) {
        return false;
    }

    Offset leftInputOffset(rank, 0);
    Offset rightInputOffset(rank, 0);
    rightInputOffset[0] = leftShape[0];

    auto baseOutputOffset = assembleAttr->GetToOffset();
    auto baseOutputDynOffset = assembleAttr->GetToDynOffset();
    NormalizeStaticDynOffset(baseOutputOffset, baseOutputDynOffset);
    regions[0] = {leftInputOffset,
                  leftShape,
                  AddStaticOffset(baseOutputOffset, leftInputOffset),
                  {},
                  AddStaticOffsetToDynOffset(baseOutputDynOffset, leftInputOffset),
                  MakeSliceDynValidShape(input, leftInputOffset, {}, leftShape)};
    regions[1] = {rightInputOffset,
                  rightShape,
                  AddStaticOffset(baseOutputOffset, rightInputOffset),
                  {},
                  AddStaticOffsetToDynOffset(baseOutputDynOffset, rightInputOffset),
                  MakeSliceDynValidShape(input, rightInputOffset, {}, rightShape)};
    return true;
}

LogicalTensorPtr BuildTreeFromReducePass::CreateSliceView(Function& function, const LogicalTensorPtr& input,
                                                          const SplitRegion& region, const Operation& anchor,
                                                          size_t index)
{
    IRBuilder builder;
    auto slice = builder.CreateTensorVar(
        function, input->Datatype(), region.shape, region.dynValidShape, input->Format(),
        "atomic_reduce_slice_" + std::to_string(anchor.GetOpMagic()) + "_" + std::to_string(index));
    slice->SetMemoryTypeBoth(input->GetMemoryTypeOriginal(), true);
    auto& viewOp = builder.CreateTensorOpStmt(function, config::GetSliceOpcode(), {input}, {slice}, anchor.GetSpan());
    viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(region.inputOffset, input->GetMemoryTypeOriginal(),
                                                            region.inputDynOffset, region.dynValidShape));
    viewOp.UpdateSubgraphID(anchor.GetSubgraphID());
    viewOp.SetScopeInfo(anchor.GetScopeInfo());
    return slice;
}

void BuildTreeFromReducePass::UpdateAssembleRegion(Operation& assemble, const SplitRegion& region)
{
    auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(assemble.GetOpAttribute());
    if (attr == nullptr) {
        attr = std::make_shared<AssembleOpAttribute>(region.outputOffset, region.outputDynOffset);
        assemble.SetOpAttribute(attr);
    } else {
        attr->SetToOffset(region.outputOffset, region.outputDynOffset);
        auto dynValidShape = region.dynValidShape;
        attr->SetFromDynValidShape(dynValidShape);
    }
}

void BuildTreeFromReducePass::MergeAtomicSemanticAttrsToRetained(const std::vector<Operation*>& assembles)
{
    if (assembles.empty() || assembles.front() == nullptr) {
        return;
    }
    auto* retained = assembles.front();
    for (auto* assemble : assembles) {
        if (assemble == nullptr) {
            continue;
        }
        if (assemble->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR)) {
            retained->SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);
        }
        if (assemble->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR)) {
            retained->SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
        }
    }
}

size_t BuildTreeFromReducePass::GetInputProducerRank(const Operation& assemble, const OperationRanks& ranks)
{
    size_t rank = 0;
    for (const auto* producer : assemble.GetIOperands().front()->GetProducers()) {
        auto iter = ranks.find(producer);
        if (iter != ranks.end()) {
            rank = std::max(rank, iter->second + 1);
        }
    }
    return rank;
}

LogicalTensorPtr BuildTreeFromReducePass::CreateAdd(Function& function, const LogicalTensorPtr& lhs,
                                                    const LogicalTensorPtr& rhs, const Operation& anchor, size_t index,
                                                    TreeRewriteInfo& rewriteInfo)
{
    IRBuilder builder;
    auto output = builder.CreateTensorVar(
        function, lhs->Datatype(), lhs->GetShape(), lhs->GetDynValidShape(), lhs->Format(),
        "atomic_reduce_add_" + std::to_string(anchor.GetOpMagic()) + "_" + std::to_string(index));
    auto& addOp = PassOperationUtils::AddOperation(function, Opcode::OP_ADD, {lhs, rhs}, {output}, nullptr,
                                                   anchor.GetSpan());
    // The balanced reduction tree has explicit producer/consumer dependencies between
    // successive ADD levels.  Reusing an earlier partial-product buffer as an ADD
    // destination breaks those anti-dependencies in the OOO memory passes, so keep
    // generated ADD outputs in distinct buffers.
    addOp.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
    addOp.UpdateSubgraphID(anchor.GetSubgraphID());
    addOp.SetScopeInfo(anchor.GetScopeInfo());
    auto addTileShape = anchor.GetTileShape();
    const auto ubSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    auto vecTile = DeriveAddVecTile(anchor, lhs, ubSize);
    addTileShape.SetVecTile(vecTile);
    addOp.UpdateTileShape(addTileShape);
    output->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    rewriteInfo.addedAddOps.push_back(&addOp);
    return output;
}

void BuildTreeFromReducePass::BuildTree(Function& function, std::vector<Operation*>& assembles,
                                        TreeRewriteInfo& rewriteInfo)
{
    std::vector<LogicalTensorPtr> level;
    level.reserve(assembles.size());
    for (const auto* assemble : assembles) {
        level.emplace_back(assemble->GetIOperands().front());
    }

    size_t addIndex = 0;
    while (level.size() > 1) {
        std::vector<LogicalTensorPtr> nextLevel;
        nextLevel.reserve((level.size() + ADD_TREE_FAN_IN - 1) / ADD_TREE_FAN_IN);
        for (size_t i = 0; i < level.size(); i += ADD_TREE_FAN_IN) {
            if (i + 1 == level.size()) {
                nextLevel.emplace_back(level[i]);
                continue;
            }
            nextLevel.emplace_back(
                CreateAdd(function, level[i], level[i + 1], *assembles.front(), addIndex++, rewriteInfo));
        }
        level = std::move(nextLevel);
    }

    MergeAtomicSemanticAttrsToRetained(assembles);
    assembles.front()->ReplaceIOperand(0, level.front());
    rewriteInfo.retainedAssembles.push_back(assembles.front());
    for (size_t i = 1; i < assembles.size(); ++i) {
        assembles[i]->SetAsDeleted();
    }
}

void BuildTreeFromReducePass::BuildSplitTrees(Function& function, std::vector<Operation*>& assembles,
                                              const std::array<SplitRegion, SPLIT_REGION_COUNT>& regions,
                                              TreeRewriteInfo& rewriteInfo)
{
    std::vector<Operation*> leftAssembles;
    std::vector<Operation*> rightAssembles;
    leftAssembles.reserve(assembles.size());
    rightAssembles.reserve(assembles.size());

    size_t sliceIndex = 0;
    for (auto* assemble : assembles) {
        if (assemble == nullptr || assemble->GetIOperands().size() != 1 || assemble->GetOOperands().size() != 1) {
            continue;
        }
        auto leftInput = CreateSliceView(function, assemble->GetIOperands().front(), regions[0], *assemble,
                                         sliceIndex++);
        auto rightInput = CreateSliceView(function, assemble->GetIOperands().front(), regions[1], *assemble,
                                          sliceIndex++);

        assemble->ReplaceIOperand(0, leftInput);
        UpdateAssembleRegion(*assemble, regions[0]);
        leftAssembles.push_back(assemble);

        auto& rightAssemble = assemble->CloneOperation(function, {rightInput}, assemble->GetOOperands());
        UpdateAssembleRegion(rightAssemble, regions[1]);
        rightAssembles.push_back(&rightAssemble);
    }

    BuildTree(function, leftAssembles, rewriteInfo);
    BuildTree(function, rightAssembles, rewriteInfo);
}

template <typename ConvertInserterT>
Status BuildTreeFromReducePass::FinalizeAddedAddMemory(ConvertInserterT& inserter,
                                                       const std::vector<Operation*>& addedAddOps)
{
    for (auto* addOp : addedAddOps) {
        if (addOp == nullptr || addOp->IsDeleted() || addOp->GetIOperands().size() != ADD_TREE_FAN_IN ||
            addOp->GetOOperands().size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "BuildTree generated an invalid ADD operation.");
            return FAILED;
        }
        auto output = addOp->GetOutputOperand(0);
        output->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
        for (size_t inputIndex = 0; inputIndex < addOp->GetIOperands().size(); ++inputIndex) {
            auto input = addOp->GetInputOperand(inputIndex);
            if (input == nullptr || input->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "BuildTree ADD[%d] input has unknown memory type; cannot repair memory path.",
                                  addOp->GetOpMagic());
                return FAILED;
            }
            inserter.UpdateTensorTobeMap(input, *addOp, MemoryType::MEM_UB, "BuildTree ADD input");
        }
    }
    return SUCCESS;
}

template <typename ConvertInserterT>
Status BuildTreeFromReducePass::FinalizeRetainedAssemblesMemory(ConvertInserterT& inserter,
                                                                const std::vector<Operation*>& retainedAssembles)
{
    for (auto* assemble : retainedAssembles) {
        if (assemble == nullptr || assemble->IsDeleted() || assemble->GetIOperands().size() != 1) {
            APASS_LOG_ERROR_F(Elements::Operation, "BuildTree retained an invalid Assemble operation.");
            return FAILED;
        }
        auto input = assemble->GetInputOperand(0);
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(assemble->GetOpAttribute());
        if (input == nullptr || attr == nullptr || input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "BuildTree final Assemble[%d] must consume a UB tensor with Assemble attributes.",
                              assemble->GetOpMagic());
            return FAILED;
        }
        attr->SetFromType(MemoryType::MEM_UB);
        auto dynValidShape = input->GetDynValidShape();
        attr->SetFromDynValidShape(dynValidShape);
        // The ADD tree has already reduced all Split-K partials.  A plain
        // ReduceAcc therefore ends in a normal write.  For ReduceAcc feeding
        // an explicit AtomicRMW, however, the final write still has to remain
        // atomic so independent reduction groups can accumulate into the same
        // destination.
        const bool fromReduceAcc = assemble->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR);
        const bool fromExplicitRmw = assemble->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR);
        if (fromReduceAcc) {
            assemble->RemoveAttr(ATOMIC_FROM_REDUCE_ACC_ATTR);
        }
        if (fromExplicitRmw) {
            assemble->RemoveAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR);
        } else {
            assemble->RemoveAttr(OpAttributeKey::atomicAdd);
        }
        inserter.UpdateTensorTobeMap(input, *assemble, MemoryType::MEM_UB, "BuildTree final Assemble input");
    }
    return SUCCESS;
}

void BuildTreeFromReducePass::CollectNewOperations(Function& function,
                                                   const std::unordered_set<Operation*>& existingOps,
                                                   std::vector<Operation*>& addedOps)
{
    for (auto& op : function.Operations(false)) {
        if (existingOps.find(&op) == existingOps.end()) {
            addedOps.push_back(&op);
            for (const auto& output : op.GetOOperands()) {
                if (output != nullptr && output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
                    output->SetMemoryTypeToBe(output->GetMemoryTypeOriginal());
                }
            }
        }
    }
}

Status BuildTreeFromReducePass::CheckAddedAddOps(const std::vector<Operation*>& addedAddOps)
{
    for (auto* addOp : addedAddOps) {
        if (addOp == nullptr || addOp->IsDeleted()) {
            continue;
        }
        if (addOp->GetOutputOperand(0)->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            APASS_LOG_ERROR_F(Elements::Operation, "BuildTree ADD[%d] output is not UB after repair.",
                              addOp->GetOpMagic());
            return FAILED;
        }
        for (const auto& input : addOp->GetIOperands()) {
            if (input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildTree ADD[%d] input is not UB after repair.",
                                  addOp->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

template <typename ConvertInserterT>
Status BuildTreeFromReducePass::FinalizeBuildTreeMemory(Function& function, const TreeRewriteInfo& rewriteInfo,
                                                        ConvertInserterT& inserter)
{
    std::unordered_set<Operation*> existingOps;
    for (auto& op : function.Operations()) {
        existingOps.insert(&op);
    }

    if (FinalizeAddedAddMemory(inserter, rewriteInfo.addedAddOps) != SUCCESS) {
        return FAILED;
    }

    if (FinalizeRetainedAssemblesMemory(inserter, rewriteInfo.retainedAssembles) != SUCCESS) {
        return FAILED;
    }

    if (inserter.DoInsertion(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "BuildTree memory conversion insertion failed.");
        return FAILED;
    }
    std::vector<Operation*> addedOps = rewriteInfo.addedAddOps;
    CollectNewOperations(function, existingOps, addedOps);
    if (!addedOps.empty() && InferShapeUtils::InferShape(function, addedOps) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "BuildTree memory repair shape inference failed.");
        return FAILED;
    }

    if (CheckAddedAddOps(rewriteInfo.addedAddOps) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status BuildTreeFromReducePass::ProcessOutputGroup(Function& function, const LogicalTensorPtr& output,
                                                   const OperationRanks& ranks, TreeRewriteInfo& rewriteInfo,
                                                   bool& changed)
{
    std::vector<std::vector<Operation*>> groups;
    for (auto* producer : output->GetProducers()) {
        if (producer == nullptr || producer->IsDeleted() || !IsAtomicAddCandidate(*producer) ||
            HasAtomicTargetReadDependency(*producer, output)) {
            continue;
        }
        auto iter = std::find_if(groups.begin(), groups.end(),
                                 [producer](const auto& group) { return CanBuildOneTree(*group.front(), *producer); });
        if (iter == groups.end()) {
            groups.push_back({producer});
        } else {
            iter->push_back(producer);
        }
    }

    for (auto& group : groups) {
        if (group.size() < MIN_TREE_GROUP_SIZE) {
            continue;
        }
        std::stable_sort(group.begin(), group.end(), [&ranks](const auto* lhs, const auto* rhs) {
            size_t lhsRank = GetInputProducerRank(*lhs, ranks);
            size_t rhsRank = GetInputProducerRank(*rhs, ranks);
            if (lhsRank != rhsRank) {
                return lhsRank < rhsRank;
            }
            return ranks.at(lhs) < ranks.at(rhs);
        });
        std::array<SplitRegion, SPLIT_REGION_COUNT> regions;
        if (TryBuildSplitRegions(*group.front(), regions)) {
            BuildSplitTrees(function, group, regions, rewriteInfo);
        } else {
            BuildTree(function, group, rewriteInfo);
        }
        changed = true;
    }
    return SUCCESS;
}

Status BuildTreeFromReducePass::RunOnFunction(Function& function)
{
    const auto determinismLevel = ConfigManagerNg::GetGlobalConfig<int64_t>(COMPUTE_DETERMINISM_LEVEL);
    if (determinismLevel == 0) {
        APASS_LOG_INFO_F(Elements::Function, "Skip BuildTreeFromReduce because compute determinism is disabled.");
        return SUCCESS;
    }

    APASS_LOG_INFO_F(Elements::Function, "===> Start BuildTreeFromReduce (determinism level %ld).", determinismLevel);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);

    OperationRanks ranks;
    std::vector<LogicalTensorPtr> outputs;
    size_t rank = 0;
    for (auto& op : function.Operations()) {
        ranks.emplace(&op, rank++);
        if (!IsAtomicAddCandidate(op)) {
            continue;
        }
        const auto& output = op.GetOOperands().front();
        if (std::find(outputs.begin(), outputs.end(), output) == outputs.end()) {
            outputs.emplace_back(output);
        }
    }

    bool changed = false;
    TreeRewriteInfo rewriteInfo;
    for (const auto& output : outputs) {
        if (ProcessOutputGroup(function, output, ranks, rewriteInfo, changed) != SUCCESS) {
            return FAILED;
        }
    }

    if (changed) {
        function.EraseOperations(true);
        function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
        Status memoryStatus = SUCCESS;
        AssignMemoryTypeMode mode = config::EnableSlice() ? AssignMemoryTypeMode::SLICE_CONTRACT :
                                                            AssignMemoryTypeMode::VIEW_ASSEMBLE;
        ConvertInserter inserter(mode);
        memoryStatus = FinalizeBuildTreeMemory(function, rewriteInfo, inserter);
        if (memoryStatus != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "BuildTree memory repair failed.");
            return FAILED;
        }
        function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End BuildTreeFromReduce.");
    return SUCCESS;
}

} // namespace npu::tile_fwk
