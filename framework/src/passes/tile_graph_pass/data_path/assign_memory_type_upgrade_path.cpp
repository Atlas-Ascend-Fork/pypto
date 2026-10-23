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
 * \file assign_memory_type_upgrade_path.cpp
 * \brief AssignMemoryType 平台路径升级链路（自 assign_memory_type.cpp 按主题拆出，仅移动不改逻辑）。
 */

#include "assign_memory_type.h"

#include <algorithm>
#include <utility>

#include "passes/pass_log/pass_log.h"
#include "passes/tile_graph_pass/data_path/memory_path_utils.h"

#define MODULE_NAME "AssignMemoryType"

#define RETURN_IF_NOT_SUCCESS(expr)                \
    do {                                           \
        Status assignMemoryReturnStatus = (expr);  \
        if (assignMemoryReturnStatus != SUCCESS) { \
            return assignMemoryReturnStatus;       \
        }                                          \
    } while (0)

namespace npu::tile_fwk {

namespace {
// 二维 matmul tile 用 CheckUBTileShape 分形校验 (前两维 16 对齐)；
// conv 多维 tile 用末维 (W) 对齐校验，两者分开判断
bool CheckUBTileShapeByRank(const LogicalTensorPtr& tensor)
{
    return (tensor != nullptr && tensor->GetShape().size() > SHAPE_DIM2) ?
               MemoryPathUtils::CheckUBConvTileShape(tensor) :
               MemoryPathUtils::CheckUBTileShape(tensor);
}

// ND 全量元素数: 全部维度乘积
size_t ShapeNumel(const Shape& shape)
{
    size_t elems = 1;
    for (auto dim : shape) {
        elems *= static_cast<size_t>(dim);
    }
    return elems;
}

// L0C 分形 (外轴, 内轴) 按物理布局取值: 二维 matmul tile 即 (M, N); conv 多维 tile (NCHW/NCDHW)
// 对应 M = 除 C 外全部维度乘积 (N·D·H·W), N = C 维, 与 ConstrucCopyOutTile 的 L0C (M, N) 映射一致;
// 只取末两维会丢失 N/C/D 前导维, 导致容量估算严重偏小
std::pair<size_t, size_t> GetNzOuterInnerAxes(const Shape& shape)
{
    constexpr size_t kConvChannelIdx = 1;
    if (shape.size() == SHAPE_DIM2) {
        return {static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1])};
    }
    size_t outer = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i == kConvChannelIdx) {
            continue;
        }
        outer *= static_cast<size_t>(shape[i]);
    }
    return {outer, static_cast<size_t>(shape[kConvChannelIdx])};
}
} // namespace

Status AssignMemoryType::ApplyPlatformPathUpgradeRules(Function& function)
{
    RETURN_IF_NOT_SUCCESS(ProcessDdrMultiReshape(function));
    RETURN_IF_NOT_SUCCESS(ProcessL1DdrL1(function));
    RETURN_IF_NOT_SUCCESS(ProcessL0C2L1SmallToLarge(function));
    RETURN_IF_NOT_SUCCESS(ProcessL0C2L1LargeToSmall(function));
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 ||
        IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        RETURN_IF_NOT_SUCCESS(ProcessL0C2UBSmallToLarge(function));
        RETURN_IF_NOT_SUCCESS(ProcessL0C2UBLargeToSmall(function));
        RETURN_IF_NOT_SUCCESS(ProcessUB2L1SmallToLarge(function));
        RETURN_IF_NOT_SUCCESS(ProcessUB2L1LargeToSmall(function));
    }
    RETURN_IF_NOT_SUCCESS(ProcessUB2UBContractSlice(function));
    return SUCCESS;
}

Status AssignMemoryType::ProcessDdrMultiReshape(Function& function)
{
    constexpr MemoryType kUbMemoryType = MemoryType::MEM_UB;
    constexpr const char* kReason = "ProcessDdrMultiReshape";
    for (auto& contract : function.Operations()) {
        if (contract.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        auto contractInput = contract.iOperand.front();
        auto contractOutput = contract.oOperand.front();
        if (contractOutput->GetProducers().size() != 1 || contractInput->GetMemoryTypeOriginal() != kUbMemoryType ||
            contractOutput->GetConsumers().size() != 1) {
            continue;
        }
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(contract, contractOutput, aligned));
        if (!aligned) {
            continue;
        }
        auto* firstReshape = *contractOutput->GetConsumers().begin();
        if (firstReshape->GetOpcode() != Opcode::OP_RESHAPE || firstReshape->iOperand.size() != 1 ||
            firstReshape->oOperand.size() != 1 || firstReshape->iOperand.front() != contractOutput) {
            continue;
        }
        auto firstReshapeOutput = firstReshape->oOperand.front();
        if (firstReshapeOutput->GetConsumers().empty()) {
            continue;
        }

        std::vector<Operation*> branchViews;
        std::vector<Operation*> branchReshapes;
        std::vector<Operation*> branchSlices;
        std::vector<Operation*> directSlices;
        bool matches = true;
        for (auto* branch : firstReshapeOutput->GetConsumers()) {
            if (branch->GetOpcode() == Opcode::OP_SLICE) {
                if (branch->iOperand.size() != 1 || branch->oOperand.size() != 1 ||
                    branch->iOperand.front() != firstReshapeOutput ||
                    branch->oOperand.front()->GetMemoryTypeOriginal() != kUbMemoryType ||
                    std::dynamic_pointer_cast<ViewOpAttribute>(branch->GetOpAttribute()) == nullptr) {
                    matches = false;
                    break;
                }
                directSlices.push_back(branch);
                continue;
            }
            if (branch->GetOpcode() != Opcode::OP_VIEW || branch->iOperand.size() != 1 ||
                branch->oOperand.size() != 1 || branch->iOperand.front() != firstReshapeOutput ||
                std::dynamic_pointer_cast<ViewOpAttribute>(branch->GetOpAttribute()) == nullptr) {
                matches = false;
                break;
            }
            auto viewOutput = branch->oOperand.front();
            if (viewOutput->GetConsumers().empty()) {
                matches = false;
                break;
            }
            branchViews.push_back(branch);
            for (auto* reshape : viewOutput->GetConsumers()) {
                if (reshape->GetOpcode() != Opcode::OP_RESHAPE || reshape->iOperand.size() != 1 ||
                    reshape->oOperand.size() != 1 || reshape->iOperand.front() != viewOutput) {
                    matches = false;
                    break;
                }
                auto reshapeOutput = reshape->oOperand.front();
                if (reshapeOutput->GetConsumers().empty()) {
                    matches = false;
                    break;
                }
                branchReshapes.push_back(reshape);
                for (auto* slice : reshapeOutput->GetConsumers()) {
                    if (slice->GetOpcode() != Opcode::OP_SLICE || slice->iOperand.size() != 1 ||
                        slice->oOperand.size() != 1 || slice->iOperand.front() != reshapeOutput ||
                        slice->oOperand.front()->GetMemoryTypeOriginal() != kUbMemoryType ||
                        std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute()) == nullptr) {
                        matches = false;
                        break;
                    }
                    branchSlices.push_back(slice);
                }
                if (!matches) {
                    break;
                }
            }
            if (!matches) {
                break;
            }
        }
        // A single reshape followed only by slices keeps the boundary selected by
        // normal inference. This upgrade is for paths with another reshape branch.
        if (!matches || branchReshapes.empty()) {
            continue;
        }

        auto contractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(contract.GetOpAttribute());
        if (contractAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessDdrMultiReshape failed because contract attr is null.");
            return FAILED;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, contractInput, contract, kUbMemoryType, kReason);
        contractAttr->SetFromType(kUbMemoryType);
        MemoryPathUtils::ForceSetOriginal(contractOutput, kUbMemoryType, kReason);
        MemoryPathUtils::ForceSetRequirement(inserter, contractOutput, *firstReshape, kUbMemoryType, kReason);
        MemoryPathUtils::ForceSetOriginal(firstReshapeOutput, kUbMemoryType, kReason);

        for (auto* view : branchViews) {
            auto viewOutput = view->oOperand.front();
            MemoryPathUtils::ForceSetRequirement(inserter, firstReshapeOutput, *view, kUbMemoryType, kReason);
            MemoryPathUtils::ForceSetOriginal(viewOutput, kUbMemoryType, kReason);
            std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute())->SetToType(kUbMemoryType);
        }
        for (auto* reshape : branchReshapes) {
            auto reshapeInput = reshape->iOperand.front();
            MemoryPathUtils::ForceSetRequirement(inserter, reshapeInput, *reshape, kUbMemoryType, kReason);
            MemoryPathUtils::ForceSetOriginal(reshape->oOperand.front(), kUbMemoryType, kReason);
        }
        for (auto* slice : branchSlices) {
            auto sliceInput = slice->iOperand.front();
            MemoryPathUtils::ForceSetRequirement(inserter, sliceInput, *slice, kUbMemoryType, kReason);
            MemoryPathUtils::ForceSetOriginal(slice->oOperand.front(), kUbMemoryType, kReason);
            std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute())->SetToType(kUbMemoryType);
        }
        for (auto* slice : directSlices) {
            MemoryPathUtils::ForceSetRequirement(inserter, firstReshapeOutput, *slice, kUbMemoryType, kReason);
            MemoryPathUtils::ForceSetOriginal(slice->oOperand.front(), kUbMemoryType, kReason);
            std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute())->SetToType(kUbMemoryType);
        }
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Upgrade contract[%d] reshape branches from DDR to UB for UB slice outputs.",
                          contract.GetOpMagic());
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL1DdrL1(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visitedOutputs;
    for (auto& contract : function.Operations()) {
        if (contract.GetOpcode() != Opcode::OP_CONTRACT || contract.oOperand.empty()) {
            continue;
        }
        auto middle = contract.oOperand.front();
        if (middle == nullptr || !visitedOutputs.insert(middle).second ||
            (middle->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN &&
             middle->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) ||
            MemoryPathUtils::IsOversizedLocalBuffer(middle, MemoryType::MEM_L1, false, true) ||
            !HasOnlyContractProducers(middle) || !HasOnlySliceConsumers(middle)) {
            continue;
        }

        bool allContractInputsFromL1 = true;
        for (auto* producer : middle->GetProducers()) {
            auto input = producer->iOperand.front();
            if (input == nullptr) {
                allContractInputsFromL1 = false;
                break;
            }
            MemoryType requirement = inserter.GetRequirementOrUnknown(input, *producer);
            if (requirement != MemoryType::MEM_L1 &&
                (requirement != MemoryType::MEM_UNKNOWN || input->GetMemoryTypeOriginal() != MemoryType::MEM_L1)) {
                allContractInputsFromL1 = false;
                break;
            }
        }
        if (!allContractInputsFromL1) {
            continue;
        }

        bool allSliceConsumersToL1 = true;
        for (auto* consumer : middle->GetConsumers()) {
            auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
            if (viewAttr == nullptr || viewAttr->GetTo() != MemoryType::MEM_L1 ||
                inserter.GetRequirementOrUnknown(middle, *consumer) != MemoryType::MEM_DEVICE_DDR) {
                allSliceConsumersToL1 = false;
                break;
            }
        }
        if (!allSliceConsumersToL1) {
            continue;
        }

        MemoryPathUtils::ForceSetOriginal(middle, MemoryType::MEM_L1, "ProcessL1DdrL1");
        for (auto* consumer : middle->GetConsumers()) {
            MemoryPathUtils::ForceSetRequirement(inserter, middle, *consumer, MemoryType::MEM_L1, "ProcessL1DdrL1");
        }
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Upgrade tensor[%d] from L1 -> DDR -> L1 path to L1 for contract/slice layout ops.",
                          middle->GetMagic());
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2L1SmallToLarge(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSliceContractPath(op, MemoryType::MEM_L0C, MemoryType::MEM_L1,
                                                          "ProcessL0C2L1SmallToLarge", false, false, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessUB2UBContractSlice(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT || op.oOperand.empty() || op.oOperand.front() == nullptr) {
            continue;
        }
        auto middle = op.oOperand.front();
        if (!MemoryPathUtils::FitsAssembleOutputMemoryLimit(middle, MemoryType::MEM_UB)) {
            continue;
        }
        bool canKeepProducers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepContractProducersInUb(middle, canKeepProducers));
        if (!canKeepProducers) {
            continue;
        }
        bool canKeepConsumers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepSliceConsumersInUb(middle, canKeepConsumers));
        if (!canKeepConsumers) {
            continue;
        }
        bool hasNonZero = false;
        RETURN_IF_NOT_SUCCESS(HasNonZeroSliceFromOffset(middle, hasNonZero));
        if (hasNonZero) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                                                "ProcessUB2UBContractSlice", false, false, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2L1LargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT || op.oOperand.empty() || op.oOperand.front() == nullptr) {
            continue;
        }
        // Single-contract restriction is enforced inside TryUpgradeSingleContractSlicePath.
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_L0C, MemoryType::MEM_L1,
                                                                "ProcessL0C2L1LargeToSmall", false, false, false));
    }
    return SUCCESS;
}

bool AssignMemoryType::CheckConsumerMoveInShapeMultiple(const LogicalTensorPtr& output, const LogicalTensorPtr& input)
{
    for (auto& consumerOp : output->GetConsumers()) {
        if (!IsModeMoveIn(mode_, consumerOp->GetOpcode())) {
            continue;
        }
        if (!IsSliceMode()) {
            // VIEW is defined as a single-input/single-output operation.
            ASSERT(consumerOp->GetOOperands().size() == 1) << "VIEW should have 1 output";
        }
        if (!MemoryPathUtils::IsDimMultiple(consumerOp->GetOOperands().front()->GetShape(), input->GetShape())) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::AreAllSliceConsumerShapesPreserved(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetConsumers().empty()) {
        return false;
    }
    const auto& inputShape = tensor->GetShape();
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_SLICE) {
            return false;
        }
        if (consumer->iOperand.empty() || consumer->iOperand.front() != tensor) {
            return false;
        }
        if (consumer->oOperand.empty() || consumer->oOperand.front() == nullptr ||
            consumer->oOperand.front()->GetShape() != inputShape) {
            return false;
        }
    }
    return true;
}

Status AssignMemoryType::ProcessL0C2UBSmallToLarge(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        // conv 多维 tile 放开 2 维限制 (requireMatrixShape=false)，对齐校验按维数分派
        RETURN_IF_NOT_SUCCESS(TryUpgradeSliceContractPath(op, MemoryType::MEM_L0C, MemoryType::MEM_UB,
                                                          "ProcessL0C2UBSmallToLarge", false, true, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2UBLargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        // conv 多维 tile 放开 2 维限制 (requireMatrixShape=false)，对齐校验按维数分派
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_L0C, MemoryType::MEM_UB,
                                                                "ProcessL0C2UBLargeToSmall", false, true, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessUB2L1SmallToLarge(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSliceContractPath(op, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                                          "ProcessUB2L1SmallToLarge", true, false, true));
    }
    return SUCCESS;
}

bool AssignMemoryType::ShouldSkipUB2L1SmallToLarge(const LogicalTensorPtr& iOperand,
                                                   const LogicalTensorPtr& oOperand) const
{
    const size_t UB_LIMIT = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                UB_THRESHOLD_NORMAL);
    size_t tensorSize = IsSliceMode() ? CalcNZTensorSize(iOperand) : CalcNZTensorSizeLegacy(iOperand);
    if (tensorSize > UB_LIMIT) {
        return true;
    }
    // 检查 consumer slice 是否有 copy_in_mode=0 属性
    for (auto& consumerOp : oOperand->GetConsumers()) {
        if (IsModeMoveIn(mode_, consumerOp->GetOpcode())) {
            int64_t copyInModeValue = 0;
            if (consumerOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0) {
                return true;
            }
            // MXMatmul场景K轴非64对齐不支持UB2L1直连（MX补齐仅由DDR路径支持），回退DDR
            if (MemoryPathUtils::IsMxPaddingMode(*consumerOp)) {
                return true;
            }
        }
    }
    return !MemoryPathUtils::CheckInnerAxisC0Size(iOperand, oOperand);
}

Status AssignMemoryType::ProcessUB2L1LargeToSmall(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_CONTRACT) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                                                "ProcessUB2L1LargeToSmall", true, false, true));
    }
    return SUCCESS;
}

bool AssignMemoryType::CanUseMiddleTensorForUpgrade(const LogicalTensorPtr& middle, MemoryType targetType) const
{
    if (middle == nullptr || targetType == MemoryType::MEM_UNKNOWN) {
        return false;
    }
    MemoryType currentType = middle->GetMemoryTypeOriginal();
    return currentType == MemoryType::MEM_UNKNOWN || currentType == MemoryType::MEM_DEVICE_DDR ||
           currentType == targetType;
}

bool AssignMemoryType::HasOnlyContractProducers(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetProducers().empty()) {
        return false;
    }
    return std::all_of(tensor->GetProducers().begin(), tensor->GetProducers().end(), [](Operation* producer) {
        return producer != nullptr && producer->GetOpcode() == Opcode::OP_CONTRACT && !producer->iOperand.empty() &&
               !producer->oOperand.empty() &&
               std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute()) != nullptr;
    });
}

bool AssignMemoryType::HasOnlySliceConsumers(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetConsumers().empty()) {
        return false;
    }
    return std::all_of(tensor->GetConsumers().begin(), tensor->GetConsumers().end(), [](Operation* consumer) {
        return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE && !consumer->iOperand.empty() &&
               !consumer->oOperand.empty() &&
               std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute()) != nullptr;
    });
}

bool AssignMemoryType::IsSliceOutputTarget(Operation& sliceOp, MemoryType targetType) const
{
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.oOperand.empty() || sliceOp.oOperand.front() == nullptr) {
        return false;
    }
    MemoryType outputOriginal = sliceOp.oOperand.front()->GetMemoryTypeOriginal();
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        return outputOriginal == targetType;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    return viewOpAttribute != nullptr && viewOpAttribute->GetTo() == targetType;
}

Status AssignMemoryType::EnsureSliceOutputTarget(Operation& sliceOp, MemoryType targetType, const std::string& reason)
{
    if (sliceOp.oOperand.empty() || sliceOp.oOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because slice output is invalid.", reason.c_str(),
                          sliceOp.GetOpMagic());
        return FAILED;
    }
    auto output = sliceOp.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal != MemoryType::MEM_UNKNOWN && outputOriginal != targetType) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Upgrade %s[%d] failed because slice output tensor[%d] original %s conflicts with target %s.",
                          reason.c_str(), sliceOp.GetOpMagic(), output->GetMagic(),
                          BriefMemoryTypeToString(outputOriginal).c_str(), BriefMemoryTypeToString(targetType).c_str());
        return FAILED;
    }
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, targetType, reason));
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because slice attr is null.", reason.c_str(),
                          sliceOp.GetOpMagic());
        return FAILED;
    }
    viewOpAttribute->SetToType(targetType);
    return SUCCESS;
}

Status AssignMemoryType::ApplySliceContractUpgrade(Operation& sliceOp, MemoryType sourceType, MemoryType targetType,
                                                   const std::string& reason)
{
    auto middle = sliceOp.iOperand.front();
    MemoryPathUtils::ForceSetOriginal(middle, targetType, reason);
    for (auto* consumer : middle->GetConsumers()) {
        RETURN_IF_NOT_SUCCESS(EnsureSliceOutputTarget(*consumer, targetType, reason));
        MemoryPathUtils::ForceSetRequirement(inserter, middle, *consumer, targetType, reason);
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract input is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, *producer, sourceType, reason);
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (assembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        assembleOpAttribute->SetFromType(sourceType);
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Upgrade middle tensor[%d] to %s for %s contract-side special path.",
                      middle->GetMagic(), BriefMemoryTypeToString(targetType).c_str(), reason.c_str());
    return SUCCESS;
}

Status AssignMemoryType::ApplySingleContractSliceUpgrade(Operation& contractOp, MemoryType sourceType,
                                                         MemoryType targetType, const std::string& reason)
{
    auto middle = contractOp.oOperand.front();
    MemoryPathUtils::ForceSetOriginal(middle, sourceType, reason);
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(contractOp.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.", reason.c_str(),
                          contractOp.GetOpMagic());
        return FAILED;
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract input is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, *producer, sourceType, reason);
        auto producerAssembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (producerAssembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        producerAssembleOpAttribute->SetFromType(sourceType);
    }
    for (auto* consumer : middle->GetConsumers()) {
        RETURN_IF_NOT_SUCCESS(EnsureSliceOutputTarget(*consumer, targetType, reason));
        MemoryPathUtils::ForceSetRequirement(inserter, middle, *consumer, sourceType, reason);
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Upgrade middle tensor[%d] to %s for %s slice-side special path.",
                      middle->GetMagic(), BriefMemoryTypeToString(sourceType).c_str(), reason.c_str());
    return SUCCESS;
}

bool AssignMemoryType::CanUseL0C2L1UpgradePath(Operation& operation)
{
    if (operation.iOperand.empty() || operation.oOperand.empty() || operation.iOperand.front() == nullptr ||
        operation.oOperand.front() == nullptr) {
        return false;
    }
    return inserter.FitL0C2L1(operation);
}

Status AssignMemoryType::TryUpgradeSliceContractPath(Operation& sliceOp, MemoryType sourceType, MemoryType targetType,
                                                     const std::string& reason, bool requireMatrixShape,
                                                     bool checkUbTileShape, bool checkUb2L1Constraints)
{
    constexpr size_t kMatrixShapeDimCount = 2;
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.iOperand.empty() || sliceOp.oOperand.empty()) {
        return SUCCESS;
    }
    auto middle = sliceOp.iOperand.front();
    auto target = sliceOp.oOperand.front();
    if (middle == nullptr || target == nullptr || !CanUseMiddleTensorForUpgrade(middle, targetType) ||
        !HasOnlyContractProducers(middle) || !HasOnlySliceConsumers(middle) ||
        !IsSliceOutputTarget(sliceOp, targetType)) {
        return SUCCESS;
    }
    // 与 TryUpgradeSingleContractSlicePath 一致：middle 的 shape 存在 -1（动态维度）时不参与升级，
    // 动态 shape 的 tensor 无法作为本地 scratch 静态分配。
    const auto& middleShape = middle->GetShape();
    if (std::any_of(middleShape.begin(), middleShape.end(), [](int64_t dim) { return dim < 0; })) {
        return SUCCESS;
    }
    for (auto* consumer : middle->GetConsumers()) {
        if (consumer != nullptr && !IsSliceOutputTarget(*consumer, targetType)) {
            return SUCCESS;
        }
    }
    if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
        !AreAllSliceConsumerShapesPreserved(middle)) {
        return SUCCESS;
    }
    if (requireMatrixShape &&
        (middle->GetShape().size() != kMatrixShapeDimCount || target->GetShape().size() != kMatrixShapeDimCount)) {
        return SUCCESS;
    }
    if (checkUbTileShape && !MemoryPathUtils::FitsAssembleOutputMemoryLimit(middle, targetType)) {
        return SUCCESS;
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr || input->GetMemoryTypeOriginal() != sourceType) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_UB &&
            !inserter.IsL0C2UbSupportedDtype(input)) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_UB && targetType == MemoryType::MEM_L1 &&
            !inserter.IsUb2L1SupportedDtype(input)) {
            return SUCCESS;
        }
        if (requireMatrixShape && input->GetShape().size() != kMatrixShapeDimCount) {
            return SUCCESS;
        }
        if (checkUbTileShape && !CheckUBTileShapeByRank(input)) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*producer)) {
            return SUCCESS;
        }
        for (auto* consumer : middle->GetConsumers()) {
            if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
                !CanUseL0C2L1UpgradePath(*consumer)) {
                return SUCCESS;
            }
            if (targetType != MemoryType::MEM_UB) {
                continue;
            }
            auto output = consumer->oOperand.front();
            if (HasDifferentConsumerRequirement(output, targetType)) {
                return SUCCESS;
            }
        }
        if (HasParallelDifferentConsumerRequirement(input, targetType) ||
            !MemoryPathUtils::IsDimMultiple(middle->GetShape(), input->GetShape()) ||
            !CheckConsumerMoveInShapeMultiple(middle, input) ||
            !MemoryPathUtils::IsDimMultiple(target->GetShape(), input->GetShape())) {
            return SUCCESS;
        }
        if (checkUb2L1Constraints && ShouldSkipUB2L1SmallToLarge(input, middle)) {
            return SUCCESS;
        }
    }
    return ApplySliceContractUpgrade(sliceOp, sourceType, targetType, reason);
}

Status AssignMemoryType::TryUpgradeSingleContractSlicePath(Operation& contractOp, MemoryType sourceType,
                                                           MemoryType targetType, const std::string& reason,
                                                           bool requireMatrixShape, bool checkUbTileShape,
                                                           bool checkUb2L1Constraints)
{
    constexpr size_t kMatrixShapeDimCount = 2;
    if (contractOp.GetOpcode() != Opcode::OP_CONTRACT || contractOp.iOperand.empty() || contractOp.oOperand.empty()) {
        return SUCCESS;
    }
    auto input = contractOp.iOperand.front();
    auto middle = contractOp.oOperand.front();
    if (input == nullptr || middle == nullptr || input->GetMemoryTypeOriginal() != sourceType ||
        !CanUseMiddleTensorForUpgrade(middle, sourceType) || !HasOnlyContractProducers(middle) ||
        !HasOnlySliceConsumers(middle)) {
        return SUCCESS;
    }
    // middle 的 shape 存在 -1（动态维度）时不能参与升级：
    // 动态 shape 的 tensor 无法作为本地 scratch 静态分配（OoOSchedule requires static rawShape）。
    const auto& middleShape = middle->GetShape();
    if (std::any_of(middleShape.begin(), middleShape.end(), [](int64_t dim) { return dim < 0; })) {
        return SUCCESS;
    }
    // Large-to-small must be a single-contract path: the slice/view side already reads with a
    // from-offset, and multiple contracts would write the middle with their own to-offsets.
    // Offsets on both sides cannot be handled by the underlying direct path and break accuracy.
    // The same-memory path (sourceType == targetType, e.g. UB2UB) is exempt: multiple contracts
    // fan-in is its designed pattern (CanKeepContractProducersInUb checks producers one by one).
    if (sourceType != targetType &&
        (middle->GetProducers().size() != 1 || *middle->GetProducers().begin() != &contractOp)) {
        return SUCCESS;
    }
    // For large-to-small the middle stays in sourceType after the upgrade; reject when it cannot
    // fit. UB residency holds both the ND original and the NZ copy, L0C residency holds NZ only.
    // The same-memory path is exempt: its caller already checks the middle against the
    // assemble-output UB limit and the middle carries no NZ copy there.
    if (sourceType != targetType && middle->GetShape().size() >= kMatrixShapeDimCount) {
        if (sourceType == MemoryType::MEM_UB) {
            const size_t ubLimit = static_cast<size_t>(
                Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_NORMAL);
            if (CalcNZTensorSize(middle) > ubLimit) {
                return SUCCESS;
            }
        } else if (sourceType == MemoryType::MEM_L0C) {
            const size_t l0cLimit = static_cast<size_t>(
                Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C) * L0C_THRESHOLD);
            if (CalcNzStorageSize(middle) > l0cLimit) {
                return SUCCESS;
            }
        }
    }
    if (requireMatrixShape && input->GetShape().size() != kMatrixShapeDimCount) {
        return SUCCESS;
    }
    if (HasParallelDifferentConsumerRequirement(input, targetType)) {
        return SUCCESS;
    }
    if (HasParallelDifferentConsumerRequirement(middle, targetType)) {
        return SUCCESS;
    }
    for (auto* producer : middle->GetProducers()) {
        auto producerInput = producer->iOperand.front();
        if (producerInput == nullptr || producerInput->GetMemoryTypeOriginal() != sourceType) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_UB &&
            !inserter.IsL0C2UbSupportedDtype(producerInput)) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_UB && targetType == MemoryType::MEM_L1 &&
            !inserter.IsUb2L1SupportedDtype(producerInput)) {
            return SUCCESS;
        }
        if (requireMatrixShape && producerInput->GetShape().size() != kMatrixShapeDimCount) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*producer)) {
            return SUCCESS;
        }
        if (HasParallelDifferentConsumerRequirement(producerInput, targetType)) {
            return SUCCESS;
        }
    }
    for (auto* consumer : middle->GetConsumers()) {
        if (!IsSliceOutputTarget(*consumer, targetType)) {
            return SUCCESS;
        }
        auto output = consumer->oOperand.front();
        if (requireMatrixShape && output->GetShape().size() != kMatrixShapeDimCount) {
            return SUCCESS;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*consumer)) {
            return SUCCESS;
        }
        bool shapeCompatible = MemoryPathUtils::IsDimMultiple(input->GetShape(), output->GetShape());
        if (sourceType != MemoryType::MEM_L0C || targetType != MemoryType::MEM_L1) {
            shapeCompatible = shapeCompatible || MemoryPathUtils::IsDimMultiple(output->GetShape(), input->GetShape());
        }
        if (!shapeCompatible) {
            return SUCCESS;
        }
        if (checkUbTileShape && !CheckUBTileShapeByRank(output)) {
            return SUCCESS;
        }
        if (checkUb2L1Constraints) {
            const size_t ubLimit = static_cast<size_t>(
                Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_NORMAL);
            int64_t copyInModeValue = 0;
            // MXMatmul场景K轴非64对齐不支持UB2L1直连（MX补齐仅由DDR路径支持），回退DDR
            if (CalcNZTensorSize(input) > ubLimit ||
                (consumer->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0) ||
                MemoryPathUtils::IsMxPaddingMode(*consumer)) {
                return SUCCESS;
            }
        }
    }
    return ApplySingleContractSliceUpgrade(contractOp, sourceType, targetType, reason);
}

size_t AssignMemoryType::CalcNzAlignedSize(const LogicalTensorPtr& tensor) const
{
    constexpr int64_t kC0AlignBytes = 32;
    const auto& shape = tensor->GetShape();
    if (shape.size() < 2) {
        return 0;
    }
    DataType dtype = tensor->Datatype();
    int64_t bytes = BytesOf(dtype);
    // 外轴/内轴按 L0C 物理布局取值: conv 多维 tile 为 (N·D·H·W, C), 二维 matmul tile 为 (M, N)
    auto [outer, inner] = GetNzOuterInnerAxes(shape);
    // 外轴对齐：INT8/FP8 对齐到 32，其他对齐到 16
    size_t outerAlign = (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8) ? 32 : 16;
    // 内轴对齐：C0 size = 32 / 元素字节数
    size_t c0 = 0;
    if (bytes > 0) {
        c0 = static_cast<size_t>(kC0AlignBytes / bytes);
    }
    if (c0 == 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "CalcNzAlignedSize: invalid C0 size, c0=%zu", c0);
        return 0;
    }
    size_t alignedOuter = (outer + outerAlign - 1) / outerAlign * outerAlign + 1;
    size_t alignedInner = (inner + c0 - 1) / c0 * c0;
    return alignedOuter * alignedInner * static_cast<size_t>(bytes);
}

size_t AssignMemoryType::CalcNZTensorSize(const LogicalTensorPtr& tensor) const
{
    constexpr int64_t kAlignBytes = 4;
    const auto& shape = tensor->GetShape();
    if (shape.size() < 2) {
        return 0;
    }
    int64_t bytes = BytesOf(tensor->Datatype());
    // ND 为全量元素数 (全部维度乘积): conv 多维 tile 不能只取末两维, 否则丢失 N/C/D 前导维
    size_t numel = ShapeNumel(shape);
    size_t ndSize = numel * static_cast<size_t>(bytes);
    size_t nzSize = CalcNzAlignedSize(tensor);
    if (nzSize == 0) {
        return numel * static_cast<size_t>(bytes > 0 ? bytes : kAlignBytes);
    }
    return ndSize + nzSize;
}

size_t AssignMemoryType::CalcNzStorageSize(const LogicalTensorPtr& tensor) const
{
    constexpr int64_t kAlignBytes = 4;
    const auto& shape = tensor->GetShape();
    if (shape.size() < 2) {
        return 0;
    }
    int64_t bytes = BytesOf(tensor->Datatype());
    size_t nzSize = CalcNzAlignedSize(tensor);
    if (nzSize == 0) {
        // C0 无法推导时退化为 ND 全量大小 (全部维度乘积)
        return ShapeNumel(shape) * static_cast<size_t>(bytes > 0 ? bytes : kAlignBytes);
    }
    return nzSize;
}
} // namespace npu::tile_fwk
