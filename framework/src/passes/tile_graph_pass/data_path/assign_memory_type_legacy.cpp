/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file assign_memory_type.cpp
 * \brief
 */

#include "assign_memory_type.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/checker_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/tile_graph_pass/data_path/memory_path_utils.h"
#include "tilefwk/tilefwk.h"

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
// 穿透时视为透明的 op 集合（shape 兜底/端点收集跳过它们继续穿透）
const std::unordered_set<Opcode> TRANSPARENT_OPS = {Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_REGISTER_COPY};

// conv L0C tile 判定: 与 GenerateMoveOp::SetOpcodeByMemPath 选择 OP_L0C_COPY_UB_CONV 的判据对齐,
// 即 L0C tensor 的 producer 为 OP_TRANS_FORMAT_L0C (conv expand 产物) 且 rank 为 4/5 (NCHW/NCDHW);
// 其余 rank>2 场景 (如 batched matmul) 不视为 conv, 不允许按 conv 末维对齐规则进入 L0C2UB 通路
bool IsConvL0CTile(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    size_t rank = tensor->GetShape().size();
    if (rank != SHAPE_DIM4 && rank != SHAPE_DIM5) {
        return false;
    }
    for (const auto& producer : tensor->GetProducers()) {
        if (producer != nullptr && producer->GetOpcode() == Opcode::OP_TRANS_FORMAT_L0C) {
            return true;
        }
    }
    return false;
}
} // namespace
Status AssignMemoryType::AssignConfirmedMemoryTypesLegacy(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            RETURN_IF_NOT_SUCCESS(AssignViewAttrMemoryTypeLegacy(op));
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            RETURN_IF_NOT_SUCCESS(AssignAssembleAttrMemoryTypeLegacy(op));
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignReduceAccInputRequirements(inserter, op));
        }
        if (OpChecker::check(op, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
            RETURN_IF_NOT_SUCCESS(AssignMatmulInputRequirements(op));
        }
        RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignOpcodeDefinedMemoryTypes(inserter, op));
    }
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignInOutCastMemoryTypes(function));
    return MemoryPathUtils::EnsureAllConsumerRequirementsExist(inserter, function);
}

Status AssignMemoryType::AssignViewAttrMemoryTypeLegacy(Operation& operation)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "View attribute is null for %s[%d] while assigning view attr memory type.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType attrToType = viewOpAttribute->GetTo();
    if (attrToType == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    return MemoryPathUtils::SetOriginalChecked(operation.oOperand.front(), attrToType, "AssignViewAttrMemoryType");
}

Status AssignMemoryType::AssignAssembleAttrMemoryTypeLegacy(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assemble attribute is null for %s[%d] while assigning assemble attr memory type.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType attrFromType = assembleOpAttribute->GetFrom();
    if (attrFromType == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    return MemoryPathUtils::SetRequirementChecked(inserter, operation.iOperand.front(), operation, attrFromType,
                                                  "AssignAssembleAttrMemoryType");
}

Status AssignMemoryType::InferUncertainMemoryTypesLegacy(Function& function)
{
    std::unordered_set<LogicalTensorPtr> inferredAssembleOutputs;
    for (auto& op : function.Operations()) {
        switch (op.GetOpcode()) {
            case Opcode::OP_VIEW:
                RETURN_IF_NOT_SUCCESS(InferViewMemoryTypeLegacy(op));
                break;
            case Opcode::OP_VIEW_TYPE:
                RETURN_IF_NOT_SUCCESS(InferViewTypeMemoryType(op));
                break;
            case Opcode::OP_ASSEMBLE:
                RETURN_IF_NOT_SUCCESS(InferAssembleMemoryTypeLegacy(function, op, inferredAssembleOutputs));
                break;
            case Opcode::OP_RESHAPE:
                RETURN_IF_NOT_SUCCESS(InferReshapeMemoryTypeLegacy(op));
                RETURN_IF_NOT_SUCCESS(InferReshapeL0C2UBAndUB2L1PatternLiteNPU(op));
                break;
            default:
                break;
        }
    }

    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::ApplyOtherSpecialOpcodeRules(inserter, function));
    RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(function));
    return ApplyPlatformPathFallbackRules(function);
}

Status AssignMemoryType::InferViewMemoryTypeLegacy(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_VIEW, "Infer OP_VIEW memory type", input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_VIEW[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    RETURN_IF_NOT_SUCCESS(InferViewOutputFromRequirement(output, outputOriginal));
    // HasPermuteProducerAndTransDataDownstream 为 conv 算子问题的临时规避，正式方案落地后删除
    bool forceInputDdr = HasDynOffsetViewAndReshape(operation, output) ||
                         HasPermuteProducerAndTransDataDownstream(input, output);
    bool handled = TryHandleUnalignedView(operation, input, inputOriginal, outputOriginal);
    if (!handled && inputOriginal != MemoryType::MEM_UNKNOWN && outputOriginal != MemoryType::MEM_UNKNOWN) {
        RETURN_IF_NOT_SUCCESS(InferViewKnownInputOutput(operation, input, inputOriginal, outputOriginal));
        handled = true;
    }
    if (!handled && inputOriginal != MemoryType::MEM_UNKNOWN && outputOriginal == MemoryType::MEM_UNKNOWN) {
        RETURN_IF_NOT_SUCCESS(InferViewKnownInputUnknownOutput(operation, input, output, inputOriginal));
    }
    if (forceInputDdr) {
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR,
                                             "InferDynamicOffsetViewInputDdr");
    }
    return SUCCESS;
}

Status AssignMemoryType::InferViewOutputFromRequirement(const LogicalTensorPtr& output, MemoryType& outputOriginal)
{
    MemoryType uniqueOutputRequirement = output == nullptr ? MemoryType::MEM_UNKNOWN :
                                                             inserter.TryGetUniqueKnownRequiredType(output);
    if (outputOriginal != MemoryType::MEM_UNKNOWN || uniqueOutputRequirement == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(
        MemoryPathUtils::SetOriginalChecked(output, uniqueOutputRequirement, "InferViewOutputRequirement"));
    outputOriginal = output->GetMemoryTypeOriginal();
    return SUCCESS;
}

Status AssignMemoryType::InferViewKnownInputOutput(Operation& operation, const LogicalTensorPtr& input,
                                                   MemoryType inputOriginal, MemoryType outputOriginal)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_VIEW[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    viewOpAttribute->SetToType(outputOriginal);
    if (CanUseDirectViewPath(operation, inputOriginal, outputOriginal)) {
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, inputOriginal, "InferViewDirectPath");
        APASS_LOG_DEBUG_F(Elements::Operation, "Infer OP_VIEW[%d] direct %s for input tensor[%d].",
                          operation.GetOpMagic(), BriefMemoryTypeToString(inputOriginal).c_str(), input->GetMagic());
        return SUCCESS;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR,
                                         "InferViewFallbackDdr");
    return SUCCESS;
}

Status AssignMemoryType::InferViewKnownInputUnknownOutput(Operation& operation, const LogicalTensorPtr& input,
                                                          const LogicalTensorPtr& output, MemoryType inputOriginal)
{
    if (inputOriginal == MemoryType::MEM_L0C)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, inputOriginal, "InferViewReuseInputOriginal"));
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, inputOriginal, "InferViewReuseInputOriginal");
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_VIEW[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    viewOpAttribute->SetToType(inputOriginal);
    APASS_LOG_DEBUG_F(Elements::Operation, "Infer OP_VIEW[%d] reuse %s for output tensor[%d].", operation.GetOpMagic(),
                      BriefMemoryTypeToString(inputOriginal).c_str(), output->GetMagic());
    return SUCCESS;
}

bool AssignMemoryType::TryHandleUnalignedView(Operation& operation, const LogicalTensorPtr& input,
                                              MemoryType inputOriginal, MemoryType outputOriginal)
{
    if (inputOriginal == MemoryType::MEM_UNKNOWN) {
        return false;
    }
    // cube 数据加载路径不受 32 字节对齐约束
    if (outputOriginal == MemoryType::MEM_L0A || outputOriginal == MemoryType::MEM_L0B ||
        outputOriginal == MemoryType::MEM_L0AMX || outputOriginal == MemoryType::MEM_L0BMX) {
        return false;
    }
    if (IsViewFromOffsetAligned(operation)) {
        return false;
    }
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
        if (viewOpAttribute != nullptr) {
            viewOpAttribute->SetToType(outputOriginal);
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR,
                                             "InferViewUnalignedOffset");
        return true;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR,
                                         "InferViewUnknownOutputUnaligned");
    return true;
}

bool AssignMemoryType::HasParallelDifferentConsumerRequirementLegacy(const LogicalTensorPtr& tensor,
                                                                     MemoryType targetType) const
{
    if (tensor == nullptr || tensor->GetConsumers().size() <= 1) {
        return false;
    }
    auto requirements = inserter.GetConsumerRequirements(tensor);
    return std::any_of(requirements.begin(), requirements.end(), [this, targetType](const auto& item) {
        auto resolveOutputRequirement = [this](const LogicalTensorPtr& output) {
            return MemoryPathUtils::InferUniqueRequirementThroughViewConsumers(inserter, output);
        };
        MemoryType requirement = MemoryPathUtils::ResolveEffectiveConsumerRequirement(
            item.first, item.second, targetType, resolveOutputRequirement);
        return MemoryPathUtils::IsDifferentKnownRequirement(requirement, targetType);
    });
}

bool AssignMemoryType::IsViewFromOffsetAligned(Operation& operation) const
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr || operation.iOperand.empty() || operation.oOperand.empty()) {
        return false;
    }
    auto fromOffset = viewOpAttribute->GetFromOffset();
    if (fromOffset.empty()) {
        return true;
    }
    auto input = operation.iOperand.front();
    if (input == nullptr || input->GetRawTensor() == nullptr) {
        return false;
    }
    int64_t lineOffset = CalcLineOffsetLegacy(input->GetRawTensor()->rawshape, fromOffset);
    if (lineOffset == -1) {
        return true;
    }
    static constexpr int VIEW_ALIGN_BYTES = 32;
    auto output = operation.oOperand.front();
    return (BytesOf(output->Datatype()) * lineOffset) % VIEW_ALIGN_BYTES == 0;
}

bool AssignMemoryType::HasDynOffsetViewAndReshape(Operation& operation, const LogicalTensorPtr& output) const
{
    if (operation.GetOpcode() != Opcode::OP_VIEW) {
        return false;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr || viewOpAttribute->GetFromDynOffset().empty()) {
        return false;
    }
    const auto& fromOffset = viewOpAttribute->GetFromOffset();
    const auto& fromDynOffset = viewOpAttribute->GetFromDynOffset();
    bool hasDynamicOffset = false;
    if (fromOffset.size() != fromDynOffset.size()) {
        hasDynamicOffset = true;
    } else {
        for (size_t i = 0; i < fromDynOffset.size(); ++i) {
            if (!fromDynOffset[i].ConcreteValid() || fromDynOffset[i].Concrete() != fromOffset[i]) {
                hasDynamicOffset = true;
                break;
            }
        }
    }
    if (!hasDynamicOffset || output == nullptr) {
        return false;
    }
    for (const auto& consumerOp : output->GetConsumers()) {
        if (consumerOp != nullptr && consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

bool AssignMemoryType::HasTransDataConsumer(const LogicalTensorPtr& tensor) const
{
    for (const auto& consumerOp : tensor->GetConsumers()) {
        if (consumerOp != nullptr && consumerOp->HasAttr(OpAttributeKey::transDataOffset)) {
            return true;
        }
    }
    return false;
}

bool AssignMemoryType::HasPermuteProducerAndTransDataDownstream(const LogicalTensorPtr& input,
                                                                const LogicalTensorPtr& output) const
{
    bool hasPermuteProducer = false;
    for (const auto& producerOp : input->GetProducers()) {
        if (producerOp != nullptr && producerOp->GetOpcode() == Opcode::OP_PERMUTE) {
            hasPermuteProducer = true;
            break;
        }
    }
    if (!hasPermuteProducer) {
        return false;
    }
    if (HasTransDataConsumer(output)) {
        return true;
    }
    for (const auto& consumerOp : output->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_REGISTER_COPY ||
            consumerOp->oOperand.empty()) {
            continue;
        }
        const auto& registerCopyOutput = consumerOp->oOperand.front();
        if (registerCopyOutput == nullptr) {
            continue;
        }
        for (const auto& assembleOp : registerCopyOutput->GetConsumers()) {
            if (assembleOp == nullptr || assembleOp->GetOpcode() != Opcode::OP_ASSEMBLE ||
                assembleOp->oOperand.empty()) {
                continue;
            }
            if (HasTransDataConsumer(assembleOp->oOperand.front())) {
                return true;
            }
        }
    }
    return false;
}

Status AssignMemoryType::AssignAssembleToOutCastRequirement(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Handle OP_ASSEMBLE[%d] to outcast failed because operand is empty.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.front();
    if (input == nullptr || output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Handle OP_ASSEMBLE[%d] to outcast failed because operand tensor is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType inputRequirement = input->GetMemoryTypeOriginal();
    // A5 cannot write an L1 assemble result directly to GM. Route only the outcast edge through UB.
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && inputRequirement == MemoryType::MEM_L1) {
        inputRequirement = MemoryType::MEM_UB;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, inputRequirement,
                                         "AssignAssembleToOutCastRequirement");
    MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "AssignAssembleToOutCastRequirement");
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Handle OP_ASSEMBLE[%d] to outcast failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (inputRequirement != MemoryType::MEM_UNKNOWN) {
        assembleOpAttribute->SetFromType(inputRequirement);
    }
    return SUCCESS;
}

Status AssignMemoryType::InferAssembleMemoryTypeLegacy(Function& function, Operation& operation,
                                                       std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    if (!operation.oOperand.empty() && std::find(function.outCasts_.begin(), function.outCasts_.end(),
                                                 operation.oOperand.front()) != function.outCasts_.end()) {
        return AssignAssembleToOutCastRequirement(operation);
    }
    if (!operation.oOperand.empty() && operation.oOperand.front() != nullptr &&
        !inferredAssembleOutputs.insert(operation.oOperand.front()).second) {
        return SUCCESS;
    }
    return InferAssembleMemoryTypeLegacy(operation);
}

Status AssignMemoryType::InferAssembleMemoryTypeLegacy(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] memory type failed because operand is empty.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute()) == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_ASSEMBLE[%d] memory type failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    return InferAssembleOutputMemoryType(operation.oOperand.front());
}

Status AssignMemoryType::InferAssembleOutputMemoryType(const LogicalTensorPtr& output)
{
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Infer assemble output memory type failed because output tensor is null.");
        return FAILED;
    }
    if (HasAssembleInputOutputElementCountMismatch(output)) {
        RETURN_IF_NOT_SUCCESS(ApplyAssembleDdrOutputWithInputOriginals(output, "InferAssembleElementCountMismatch",
                                                                       "InferAssembleElementCountMismatchFillInput"));
        return SUCCESS;
    }
    MemoryType tempOriginal = InferAssembleTempOriginal(output);
    bool handled = false;
    RETURN_IF_NOT_SUCCESS(TryInferAssembleOutputByTempOriginal(output, tempOriginal, handled));
    if (handled)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(TryInferAssembleOutputByProducerCandidate(output, handled));
    if (handled)
        return SUCCESS;
    MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferAssembleUnknownFallbackDdr");
    return SUCCESS;
}

bool AssignMemoryType::HasAssembleInputOutputElementCountMismatch(const LogicalTensorPtr& output) const
{
    if (output == nullptr)
        return false;
    int64_t assembleInputElements = 0;
    bool hasAssembleProducer = false;
    for (const auto& producerOp : output->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_ASSEMBLE || producerOp->iOperand.empty() ||
            producerOp->iOperand.front() == nullptr) {
            continue;
        }
        hasAssembleProducer = true;
        assembleInputElements += CommonUtils::Numel(producerOp->iOperand.front()->GetShape());
    }
    if (!hasAssembleProducer)
        return false;
    int64_t assembleOutputElements = CommonUtils::Numel(output->GetShape());
    if (assembleInputElements == assembleOutputElements)
        return false;
    return true;
}

Status AssignMemoryType::TryInferAssembleOutputByTempOriginal(const LogicalTensorPtr& output, MemoryType tempOriginal,
                                                              bool& handled)
{
    handled = tempOriginal != MemoryType::MEM_UNKNOWN;
    if (!handled)
        return SUCCESS;
    if (tempOriginal == MemoryType::MEM_DEVICE_DDR) {
        RETURN_IF_NOT_SUCCESS(
            ApplyAssembleDdrOutputWithInputOriginals(output, "InferAssembleTempDdr", "InferAssembleDdrFillInput"));
        return SUCCESS;
    }
    if (AreAssembleDirectPathsSupported(output, tempOriginal)) {
        RETURN_IF_NOT_SUCCESS(ApplyAssembleDirectOutputOriginal(output, tempOriginal));
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(ApplyAssembleDdrOutputWithInputOriginals(output, "InferAssembleUnsupportedPath",
                                                                   "InferAssembleFallbackFillInput"));
    return SUCCESS;
}

bool AssignMemoryType::AreAssembleDirectPathsSupported(const LogicalTensorPtr& output, MemoryType targetOriginal)
{
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp))
            continue;
        auto input = producerOp->iOperand.front();
        if (input == nullptr)
            return false;
        MemoryType fromType = MemoryPathUtils::GetAssembleInputType(inserter, *producerOp);
        if (fromType == MemoryType::MEM_UNKNOWN)
            return false;
        bool checkOffsetAlignment = !MemoryPathUtils::IsAdvancedMemoryPath(fromType, targetOriginal);
        if ((checkOffsetAlignment && !IsAssembleToOffsetAlignedLegacy(*producerOp, output)) ||
            !CanUseDirectAssemblePath(*producerOp, fromType, targetOriginal)) {
            return false;
        }
    }
    return true;
}

Status AssignMemoryType::ApplyAssembleDirectOutputOriginal(const LogicalTensorPtr& output, MemoryType targetOriginal)
{
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, targetOriginal, "InferAssembleTempOriginal"));
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp))
            continue;
        RETURN_IF_NOT_SUCCESS(SyncAssembleInputRequirementAndAttr(*producerOp, MemoryType::MEM_UNKNOWN,
                                                                  "InferAssembleFillInputRequirement"));
    }
    return SUCCESS;
}

Status AssignMemoryType::SyncAssembleInputRequirementAndAttr(Operation& operation, MemoryType fallbackType,
                                                             const std::string& reason)
{
    auto input = operation.iOperand.front();
    if (input == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because input tensor is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
    MemoryType resolvedFallback = fallbackType == MemoryType::MEM_UNKNOWN ? input->GetMemoryTypeOriginal() :
                                                                            fallbackType;
    if (fromType == MemoryType::MEM_UNKNOWN && resolvedFallback != MemoryType::MEM_UNKNOWN) {
        fromType = resolvedFallback;
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, fromType, reason);
    }
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (fromType != MemoryType::MEM_UNKNOWN) {
        assembleOpAttribute->SetFromType(fromType);
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyAssembleDdrOutputWithInputOriginals(const LogicalTensorPtr& output,
                                                                  const std::string& originalReason,
                                                                  const std::string& inputReason)
{
    MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, originalReason);
    return FillAssembleInputRequirementsFromOriginal(output, inputReason);
}

Status AssignMemoryType::FillAssembleInputRequirementsFromOriginal(const LogicalTensorPtr& output,
                                                                   const std::string& reason)
{
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp))
            continue;
        auto input = producerOp->iOperand.front();
        if (input != nullptr && inserter.GetRequirementOrUnknown(input, *producerOp) == MemoryType::MEM_UNKNOWN &&
            input->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
            MemoryPathUtils::ForceSetRequirement(inserter, input, *producerOp, input->GetMemoryTypeOriginal(), reason);
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::TryInferAssembleOutputByProducerCandidate(const LogicalTensorPtr& output, bool& handled)
{
    bool hasConflict = false;
    MemoryType producerCandidate = InferAssembleProducerCandidate(output, hasConflict);
    handled = !hasConflict && producerCandidate != MemoryType::MEM_UNKNOWN &&
              MemoryPathUtils::FitsAssembleOutputMemoryLimit(output, producerCandidate) &&
              AreAssembleDirectPathsSupported(output, producerCandidate);
    if (!handled)
        return SUCCESS;
    return ApplyAssembleProducerCandidate(output, producerCandidate);
}

MemoryType AssignMemoryType::InferAssembleProducerCandidate(const LogicalTensorPtr& output, bool& hasConflict) const
{
    MemoryType producerCandidate = MemoryType::MEM_UNKNOWN;
    hasConflict = false;
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp))
            continue;
        MemoryType fromType = MemoryPathUtils::GetAssembleInputType(inserter, *producerOp);
        if (fromType == MemoryType::MEM_UNKNOWN) {
            hasConflict = true;
            break;
        }
        if (producerCandidate == MemoryType::MEM_UNKNOWN) {
            producerCandidate = fromType;
        } else if (producerCandidate != fromType) {
            hasConflict = true;
            break;
        }
    }
    return hasConflict ? MemoryType::MEM_UNKNOWN : producerCandidate;
}

Status AssignMemoryType::ApplyAssembleProducerCandidate(const LogicalTensorPtr& output, MemoryType producerCandidate)
{
    RETURN_IF_NOT_SUCCESS(
        MemoryPathUtils::SetOriginalChecked(output, producerCandidate, "InferAssembleProducerCandidate"));
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp))
            continue;
        RETURN_IF_NOT_SUCCESS(
            SyncAssembleInputRequirementAndAttr(*producerOp, producerCandidate, "InferAssembleProducerCandidate"));
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Infer assemble output tensor[%d] original as %s by producer candidate.",
                      output->GetMagic(), BriefMemoryTypeToString(producerCandidate).c_str());
    return SUCCESS;
}

MemoryType AssignMemoryType::InferAssembleTempOriginal(const LogicalTensorPtr& output) const
{
    if (output == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    MemoryType tempOriginal = MemoryType::MEM_UNKNOWN;
    bool hasL1ViewTarget = false;
    auto requirements = inserter.GetConsumerRequirements(output);
    for (const auto& item : requirements) {
        auto consumerOp = item.first;
        MemoryType candidate = item.second;
        if (consumerOp != nullptr && consumerOp->GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
            if (viewOpAttribute != nullptr) {
                hasL1ViewTarget = hasL1ViewTarget || viewOpAttribute->GetTo() == MemoryType::MEM_L1;
                if (candidate == MemoryType::MEM_UNKNOWN) {
                    candidate = viewOpAttribute->GetTo();
                }
            }
        }
        if (candidate == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        if (tempOriginal == MemoryType::MEM_UNKNOWN) {
            tempOriginal = candidate;
        } else if (tempOriginal != candidate) {
            return MemoryType::MEM_DEVICE_DDR;
        }
    }
    // 因 L1 View 消费者推导出 L1 时，要求输出的所有消费者都是目标 L1 的 View，
    // 否则设置为MEM_UNKNOWN，避免为其他消费者生成以 L1（Mat 型 tile）为源的 TStore（不支持 Mat 源存储）
    if (tempOriginal == MemoryType::MEM_L1 && hasL1ViewTarget && !AreAllConsumersL1Views(output)) {
        return MemoryType::MEM_UNKNOWN;
    }
    return tempOriginal;
}

bool AssignMemoryType::AreAllConsumersL1Views(const LogicalTensorPtr& output) const
{
    if (output == nullptr) {
        return false;
    }
    const auto& consumers = output->GetConsumers();
    if (consumers.empty()) {
        return false;
    }
    for (const auto& consumerOp : consumers) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
            return false;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewOpAttribute == nullptr || viewOpAttribute->GetTo() != MemoryType::MEM_L1) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::IsAssembleToOffsetAlignedLegacy(Operation& operation, const LogicalTensorPtr& output)
{
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr || output == nullptr) {
        return false;
    }
    int64_t lineOffset = CalcLineOffsetLegacy(output->GetRawTensor()->rawshape, assembleOpAttribute->GetToOffset());
    if (lineOffset == -1) {
        return true;
    }
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(output->Datatype()));
    return (tensorBytes * lineOffset) % ASSEMBLE_ALIGN_BYTES == 0;
}

Status AssignMemoryType::InferReshapeMemoryTypeLegacy(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_RESHAPE, "Infer OP_RESHAPE memory type", input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    MemoryType inputRequirement = MemoryPathUtils::GetReshapeInputRequirement(inserter, operation, input,
                                                                              inputOriginal);
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::InferReshapeOutputFromRequirement(inserter, output, outputOriginal));
    bool isConv1DGroupReshape = false;
    operation.GetAttr(OpAttributeKey::groupReshapeNoSplit, isConv1DGroupReshape);
    if (KeepSplitReshapeUbLegacy(operation, input, output) && !isConv1DGroupReshape) {
        return SUCCESS;
    }
    bool isDynamic = MemoryPathUtils::IsDynamicReshape(operation, output);
    bool canUseUb = MemoryPathUtils::CanUseUbForReshape(input, output, inputRequirement, outputOriginal) &&
                    !isConv1DGroupReshape;
    return MemoryPathUtils::ApplyReshapeMemoryType(inserter, operation, input, output, isDynamic, canUseUb);
}

bool AssignMemoryType::KeepSplitReshapeUbLegacy(Operation& operation, const LogicalTensorPtr& input,
                                                const LogicalTensorPtr& output)
{
    if (input == nullptr || output == nullptr) {
        return false;
    }
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    if (producers.empty() || consumers.empty()) {
        return false;
    }
    Operation* producer = *producers.begin();
    Operation* consumer = *consumers.begin();
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_ASSEMBLE);
    int64_t inputDataSize = input->GetDataSize();
    if (producer != nullptr && consumer != nullptr && producer->GetOpcode() == Opcode::OP_ASSEMBLE &&
        consumer->GetOpcode() == Opcode::OP_VIEW && input->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
        output->GetMemoryTypeOriginal() == MemoryType::MEM_UB && inputDataSize >= 0 &&
        static_cast<size_t>(inputDataSize) <= ubThreshold) {
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_UB, "InferSplitReshapeUb");
        for (const auto& consumerOp : output->GetConsumers()) {
            if (consumerOp != nullptr && !consumerOp->oOperand.empty() &&
                consumerOp->oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                MemoryPathUtils::ForceSetRequirement(inserter, output, *consumerOp, MemoryType::MEM_UB,
                                                     "InferSplitReshapeUb");
            }
        }
        return true;
    }
    return false;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallbackLegacy(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE && operation.GetOpcode() != Opcode::OP_VIEW) {
        return SUCCESS;
    }
    if (operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because output operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto output = operation.oOperand.front();
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because output tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    bool isAssemble = operation.GetOpcode() == Opcode::OP_ASSEMBLE;
    // op_view的输出不做L1内存类型回退，避免tile_shape设置异常场景下，回退到DDR导致出现非预期的view
    if (!MemoryPathUtils::IsOversizedLocalBuffer(output, output->GetMemoryTypeOriginal(), isAssemble, isAssemble)) {
        return SUCCESS;
    }
    MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedLocalBufferFallback");
    APASS_LOG_DEBUG_F(Elements::Operation, "Force %s[%d] output tensor[%d] to DDR by size limit.",
                      operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), output->GetMagic());
    if (operation.GetOpcode() == Opcode::OP_VIEW) {
        RETURN_IF_NOT_SUCCESS(DowngradeOversizedMoveInInputRequirement(operation));
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
        if (viewAttr != nullptr) {
            viewAttr->SetToType(MemoryType::MEM_DEVICE_DDR);
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyPlatformPathFallbackRules(Function& function)
{
    ProcessL0C2L1SmallToLargeLegacy(function);
    ProcessL0C2L1LargeToSmallLegacy(function);
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        ProcessL0C2UBSmallToLargeLegacy(function);
        ProcessL0C2UBLargeToSmallLegacy(function);
        ProcessUB2L1SmallToLargeLegacy(function);
        ProcessUB2L1LargeToSmallLegacy(function);
    }
    ProcessShapeTransportFallback(function);
    return SUCCESS;
}

bool AssignMemoryType::AreAllConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType memoryType) const
{
    auto requirements = inserter.GetConsumerRequirements(tensor);
    return std::all_of(requirements.begin(), requirements.end(),
                       [memoryType](const auto& item) { return item.second == memoryType; });
}

void AssignMemoryType::DowngradeConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType fromType)
{
    for (const auto& [consumerOp, memoryType] : inserter.GetConsumerRequirements(tensor)) {
        if (memoryType == fromType) {
            inserter.UpdateTensorTobeMap(tensor, *consumerOp, MemoryType::MEM_DEVICE_DDR);
        }
    }
}

int64_t AssignMemoryType::CalcLineOffsetLegacy(const Shape& shape, const Offset& offset) const
{
    if (shape.size() != offset.size()) {
        return -1;
    }
    if (shape.size() == 0) {
        return 0;
    }
    int64_t lineOffset = 0;
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        lineOffset += offset[i - 1] * stride;
        stride *= shape[i - 1];
    }
    return lineOffset;
}

void AssignMemoryType::ProcessL0C2L1SmallToLargeLegacy(Function& function)
{
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto oOperand = op.GetOOperands().front();
        auto iOperand = op.GetIOperands().front();
        if (oOperand->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            continue;
        }
        if (iOperand->GetMemoryTypeOriginal() != MEM_L0C) {
            continue;
        }
        bool isConsumerOutputMultiple = CheckConsumerMoveInShapeMultiple(oOperand, iOperand);
        if (HasParallelDifferentConsumerRequirementLegacy(iOperand, MemoryType::MEM_L1) ||
            !AreAllConsumerRequirements(oOperand, MemoryType::MEM_L1) ||
            !MemoryPathUtils::IsDimMultiple(oOperand->GetShape(), iOperand->GetShape()) || !isConsumerOutputMultiple) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, MemoryType::MEM_L0C);
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Set tensor %d original memory type "
                              "to DDR since not towards L1 or not multiple dimensions.",
                              oOperand->magic);
        }
    }
}

void AssignMemoryType::ProcessL0C2L1LargeToSmallLegacy(Function& function)
{
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewOpAttribute->GetTo() != MEM_L1) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            HasParallelDifferentConsumerRequirementLegacy(iOperand, MemoryType::MEM_L1)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            !MemoryPathUtils::IsDimMultiple(iOperand->GetShape(), oOperand->GetShape())) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 &&
            iOperand->GetMemoryTypeOriginal() == MEM_UB && oOperand->shape != iOperand->shape) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
    }
}

static bool IsViewConsumerToUb(Operation* consumerOp)
{
    if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW || consumerOp->oOperand.empty()) {
        return false;
    }
    auto output = consumerOp->oOperand.front();
    if (output != nullptr && output->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        return true;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
    return viewOpAttribute != nullptr && viewOpAttribute->GetTo() == MemoryType::MEM_UB;
}

static bool IsConsumerRequirementTowardsUb(const LogicalTensorPtr& tensor, Operation* consumerOp,
                                           MemoryType requirement)
{
    if (requirement == MemoryType::MEM_UB) {
        return true;
    }
    if (requirement != tensor->GetMemoryTypeOriginal()) {
        return false;
    }
    return IsViewConsumerToUb(consumerOp);
}

static bool AreAllConsumerRequirementsTowardsUb(ConvertInserter& inserter, const LogicalTensorPtr& tensor)
{
    auto requirements = inserter.GetConsumerRequirements(tensor);
    if (requirements.empty()) {
        return false;
    }
    return std::all_of(requirements.begin(), requirements.end(), [&tensor](const auto& item) {
        return IsConsumerRequirementTowardsUb(tensor, item.first, item.second);
    });
}

void AssignMemoryType::ProcessL0C2UBSmallToLargeLegacy(Function& function)
{
    constexpr size_t kMatrixShapeDimCount = 2; // 矩阵形状维度数 (M, N)
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto oOperand = op.GetOOperands().front();
        auto iOperand = op.GetIOperands().front();
        if (iOperand->GetMemoryTypeOriginal() != MEM_L0C) {
            continue;
        }
        // conv 多维 tile (rank 4/5, producer 为 TRANS_FORMAT_L0C) 走末维 (W) 对齐校验;
        // 非 conv 维持原二维 guard: rank != 2 直接跳过, 不落入 L0C2UB 通路
        bool isConvTile = IsConvL0CTile(iOperand);
        if (!isConvTile && (iOperand->GetShape().size() != kMatrixShapeDimCount ||
                            oOperand->GetShape().size() != kMatrixShapeDimCount)) {
            continue;
        }
        bool isConsumerOutputMultiple = CheckConsumerMoveInShapeMultiple(oOperand, iOperand);
        bool isVecTileShapeValid = isConvTile ? MemoryPathUtils::CheckUBConvTileLastDimAligned(iOperand) :
                                                MemoryPathUtils::CheckUBTileShape(iOperand);
        bool canUseUb = !HasParallelDifferentConsumerRequirementLegacy(iOperand, MemoryType::MEM_UB) &&
                        AreAllConsumerRequirementsTowardsUb(inserter, oOperand) &&
                        inserter.IsL0C2UbSupportedDtype(iOperand) &&
                        MemoryPathUtils::IsDimMultiple(oOperand->GetShape(), iOperand->GetShape()) &&
                        isConsumerOutputMultiple && isVecTileShapeValid &&
                        MemoryPathUtils::FitsAssembleOutputMemoryLimit(oOperand, MemoryType::MEM_UB);
        if (!canUseUb) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, MemoryType::MEM_L0C);
            APASS_LOG_DEBUG_F(Elements::Tensor,
                              "Set tensor %d original memory type to DDR since "
                              "not towards UB or not multiple dimensions.",
                              oOperand->magic);
            continue;
        }
        MemoryPathUtils::ForceSetOriginal(oOperand, MemoryType::MEM_UB, "ProcessL0C2UBSmallToLarge");
        for (const auto& [consumerOp, memoryType] : inserter.GetConsumerRequirements(oOperand)) {
            if (memoryType != MemoryType::MEM_UB && IsViewConsumerToUb(consumerOp)) {
                inserter.UpdateTensorTobeMap(oOperand, *consumerOp, MemoryType::MEM_UB, "ProcessL0C2UBSmallToLarge");
            }
        }
    }
}

void AssignMemoryType::ProcessL0C2UBLargeToSmallLegacy(Function& function)
{
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewOpAttribute->GetTo() != MEM_UB) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        // conv 判定看 L0C 源 (producer 为 TRANS_FORMAT_L0C 且 rank 4/5), 对齐校验对象是 view 窗口 oOperand;
        // 非 conv 维持 CheckUBTileShape 前两维分形校验
        bool isConvTile = iOperand->GetMemoryTypeOriginal() == MEM_L0C && IsConvL0CTile(iOperand);
        bool isVecTileShapeValid = isConvTile ? MemoryPathUtils::CheckUBConvTileLastDimAligned(oOperand) :
                                                MemoryPathUtils::CheckUBTileShape(oOperand);
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            HasParallelDifferentConsumerRequirementLegacy(iOperand, MemoryType::MEM_UB)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C && !inserter.IsL0C2UbSupportedDtype(iOperand)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "L0C2UB large to small: dtype %s of tensor %d is not supported by L0C2UB, "
                              "downgrade to DDR",
                              DataType2String(iOperand->Datatype()), iOperand->magic);
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (iOperand->GetMemoryTypeOriginal() == MEM_L0C &&
            (!MemoryPathUtils::IsDimMultiple(iOperand->GetShape(), oOperand->GetShape()) || !isVecTileShapeValid)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
    }
}

void AssignMemoryType::ProcessUB2L1SmallToLargeLegacy(Function& function)
{
    constexpr size_t kMatrixShapeDimCount = 2; // 矩阵形状维度数 (M, N)
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto oOperand = op.GetOOperands().front();
        auto iOperand = op.GetIOperands().front();
        if (iOperand->GetMemoryTypeOriginal() != MEM_UB || oOperand->GetMemoryTypeOriginal() != MEM_L1) {
            continue;
        }
        if (iOperand->GetShape().size() != kMatrixShapeDimCount ||
            oOperand->GetShape().size() != kMatrixShapeDimCount) {
            continue;
        }
        if (!inserter.IsUb2L1SupportedDtype(iOperand)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "UB2L1 small to large: dtype %s of tensor %d is not supported by UB2L1, "
                              "downgrade to DDR",
                              DataType2String(iOperand->Datatype()), iOperand->magic);
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            continue;
        }
        if (ShouldSkipUB2L1SmallToLarge(iOperand, oOperand)) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            continue;
        }
        if (HasParallelDifferentConsumerRequirementLegacy(iOperand, MemoryType::MEM_L1) ||
            !AreAllConsumerRequirements(oOperand, MemoryType::MEM_L1) ||
            !MemoryPathUtils::IsDimMultiple(oOperand->GetShape(), iOperand->GetShape()) ||
            !CheckConsumerMoveInShapeMultiple(oOperand, iOperand)) {
            oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, MemoryType::MEM_UB);
        }
    }
}

void AssignMemoryType::ProcessUB2L1LargeToSmallLegacy(Function& function)
{
    constexpr size_t kMatrixShapeDimCount = 2; // 矩阵形状维度数 (M, N)
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        MemoryType attrToType = viewOpAttribute->GetTo();
        if (attrToType != MEM_L1) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        if (iOperand->GetMemoryTypeOriginal() != MEM_UB) {
            continue;
        }
        if (HasParallelDifferentConsumerRequirementLegacy(iOperand, MemoryType::MEM_L1)) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (iOperand->GetShape().size() != kMatrixShapeDimCount ||
            oOperand->GetShape().size() != kMatrixShapeDimCount) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (!inserter.IsUb2L1SupportedDtype(iOperand)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "UB2L1 large to small: dtype %s of tensor %d is not supported by UB2L1, "
                              "downgrade to DDR",
                              DataType2String(iOperand->Datatype()), iOperand->magic);
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        const size_t UB_LIMIT = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                    UB_THRESHOLD_NORMAL);
        size_t totalSize = CalcNZTensorSizeLegacy(iOperand);
        if (totalSize > UB_LIMIT) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "UB2L1 large to small: totalSize %zu exceeds UB_LIMIT %zu, downgrade to DDR", totalSize,
                              UB_LIMIT);
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        int64_t copyInModeValue = 0;
        if (op.GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "UB2L1 large to small skip: bias/scale tensor (copy_in_mode=%ld), View Op[%d]",
                              static_cast<long>(copyInModeValue), op.GetOpMagic());
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        // MXMatmul场景K轴非64对齐不支持UB2L1直连（MX补齐仅由DDR路径支持），回退DDR
        if (MemoryPathUtils::IsMxPaddingMode(op)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "UB2L1 large to small skip: MX matmul K not 64-aligned (copy_in_l1_padding_mode), "
                              "View Op[%d]",
                              op.GetOpMagic());
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
            continue;
        }
        if (!MemoryPathUtils::IsDimMultiple(iOperand->GetShape(), oOperand->GetShape())) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
        }
    }
}

// 判定两 tensor 是否满足任意方向的逐维整除关系（含等大放行）
bool AssignMemoryType::IsAllowedTransport(const LogicalTensorPtr& prodOut, const LogicalTensorPtr& consIn) const
{
    if (prodOut == nullptr || consIn == nullptr) {
        return false;
    }
    return MemoryPathUtils::IsDimMultiple(prodOut->GetShape(), consIn->GetShape()) ||
           MemoryPathUtils::IsDimMultiple(consIn->GetShape(), prodOut->GetShape());
}

namespace {
// 端点收集方向：沿 producer 方向找真实生产者输出 / 沿 consumer 方向找真实消费者输入
enum class TraverseDirection { PRODUCER, CONSUMER };

// 沿 direction 方向穿透 TRANSPARENT_OPS，限定在 boundType 层级内；
// 若透明 op 的对侧 tensor 离开 boundType，把当前 t 标记为端点（层级边界，不再继续）
// registerCopyOps 出参收集穿透路径上遇到的 OP_REGISTER_COPY（两侧都收集），调用方传 nullptr 可跳过
std::vector<LogicalTensorPtr> CollectRealEndpoints(const LogicalTensorPtr& tensor, MemoryType boundType,
                                                   TraverseDirection direction,
                                                   std::vector<Operation*>* registerCopyOps = nullptr)
{
    if (tensor == nullptr) {
        return {};
    }
    std::vector<LogicalTensorPtr> endpoints;
    std::unordered_set<LogicalTensorPtr> visited;
    std::vector<LogicalTensorPtr> stack = {tensor};
    while (!stack.empty()) {
        auto t = stack.back();
        stack.pop_back();
        if (t == nullptr || !visited.insert(t).second) {
            continue;
        }
        bool isEndpoint = false;
        auto& ops = (direction == TraverseDirection::PRODUCER) ? t->GetProducers() : t->GetConsumers();
        for (auto* op : ops) {
            if (TRANSPARENT_OPS.count(op->GetOpcode()) == 0) {
                isEndpoint = true;
                continue;
            }
            if (registerCopyOps != nullptr && op->GetOpcode() == Opcode::OP_REGISTER_COPY) {
                if (std::find(registerCopyOps->begin(), registerCopyOps->end(), op) == registerCopyOps->end()) {
                    registerCopyOps->push_back(op);
                }
            }
            auto& nextTensors = (direction == TraverseDirection::PRODUCER) ? op->GetIOperands() : op->GetOOperands();
            for (auto& nextT : nextTensors) {
                if (nextT->GetMemoryTypeOriginal() == boundType) {
                    stack.push_back(nextT);
                } else {
                    isEndpoint = true;
                }
            }
        }
        if (isEndpoint) {
            endpoints.push_back(t);
        }
    }
    return endpoints;
}
} // namespace

// 穿透式 shape 兜底：对仍走 L0C↔L1/L0C↔UB/UB↔L1 片上直连的 view/assemble，
// 用真实生产者输出×真实消费者输入做双向逐维整除判断，不满足则拦截回退 DDR
void AssignMemoryType::ProcessShapeTransportFallback(Function& function)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    bool is3510 = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (op.GetIOperands().empty() || op.GetOOperands().empty()) {
            continue;
        }
        auto iOperand = op.GetIOperands().front();
        auto oOperand = op.GetOOperands().front();
        MemoryType fromType = iOperand->GetMemoryTypeOriginal();
        MemoryType toType = MEM_DEVICE_DDR;
        bool isTransport = false;
        if (opcode == Opcode::OP_VIEW) {
            auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
            if (viewAttr == nullptr) {
                continue;
            }
            toType = viewAttr->GetTo();
            // L0C→L1 全平台; L0C→UB / UB→L1 仅 DAV_3510
            bool pathL0C2L1 = (fromType == MEM_L0C && toType == MEM_L1);
            bool path3510 = is3510 &&
                            ((fromType == MEM_L0C && toType == MEM_UB) || (fromType == MEM_UB && toType == MEM_L1));
            isTransport = (pathL0C2L1 || path3510) && inserter.GetRequirementOrUnknown(iOperand, op) != MEM_DEVICE_DDR;
        } else if (opcode == Opcode::OP_ASSEMBLE) {
            toType = oOperand->GetMemoryTypeOriginal();
            if (toType == MEM_DEVICE_DDR) {
                continue;
            }
            // L0C→L1(iOperand=L0C, oOperand≠UB) 全平台; L0C→UB / UB→L1 仅 DAV_3510
            bool pathL0C2L1 = (fromType == MEM_L0C && toType != MEM_UB);
            bool path3510 = is3510 &&
                            ((fromType == MEM_L0C && toType == MEM_UB) || (fromType == MEM_UB && toType == MEM_L1));
            isTransport = (pathL0C2L1 || path3510);
        }
        if (!isTransport) {
            continue;
        }
        // 收集穿透端点：iOperand 向前找真实生产者输出，oOperand 向后找真实消费者输入；
        // 穿透限定在各自层级内(fromType/toType)，跨层级边界即停
        // 同时收集穿透路径上的 REGISTER_COPY（两侧都收集），用于后续 tileshape 整数倍校验
        std::vector<Operation*> prodRegisterCopies;
        std::vector<Operation*> consRegisterCopies;
        auto prodOutputs = CollectRealEndpoints(iOperand, fromType, TraverseDirection::PRODUCER, &prodRegisterCopies);
        auto consInputs = CollectRealEndpoints(oOperand, toType, TraverseDirection::CONSUMER, &consRegisterCopies);
        // 所有端点对 IsAllowedTransport 全成立才放行；端点集为空或任一不成立则拦截。
        // 空端点集（如函数出口/无 producer 的 tensor）无法证明兼容，保守拦截回退 DDR
        Shape firstBadProdShape;
        Shape firstBadConsShape;
        bool allowed = !prodOutputs.empty() && !consInputs.empty();
        for (const auto& prodOut : prodOutputs) {
            for (const auto& consIn : consInputs) {
                if (!IsAllowedTransport(prodOut, consIn)) {
                    allowed = false;
                    firstBadProdShape = prodOut->GetShape();
                    firstBadConsShape = consIn->GetShape();
                    break;
                }
            }
            if (!allowed) {
                break;
            }
        }
        // 额外条件：穿透路径上有 REGISTER_COPY 时，其输入输出 tensor shape 必须是端点对中
        // 逐维较小 shape 的整数倍，否则拦截回退 DDR（避免搬运粒度不匹配导致数据错位）
        Shape firstBadRcShape;
        if (allowed && (!prodRegisterCopies.empty() || !consRegisterCopies.empty())) {
            for (const auto& prodOut : prodOutputs) {
                for (const auto& consIn : consInputs) {
                    const auto& prodShape = prodOut->GetShape();
                    const auto& consShape = consIn->GetShape();
                    Shape smallerShape;
                    smallerShape.reserve(prodShape.size());
                    for (size_t i = 0; i < prodShape.size(); i++) {
                        smallerShape.push_back(std::min(prodShape[i], consShape[i]));
                    }
                    auto checkRegisterCopies = [&](std::vector<Operation*>& rcOps) -> bool {
                        std::sort(rcOps.begin(), rcOps.end(), [](const auto* lhs, const auto* rhs) {
                            return lhs->GetOpMagic() < rhs->GetOpMagic();
                        });
                        for (auto* rcOp : rcOps) {
                            // REGISTER_COPY is defined as a single-input/single-output copy operation.
                            ASSERT(rcOp->GetIOperands().size() == 1) << "REGISTER_COPY should have 1 input";
                            ASSERT(rcOp->GetOOperands().size() == 1) << "REGISTER_COPY should have 1 output";
                            const auto& rcShape = rcOp->GetIOperands().front()->GetShape();
                            if (rcShape.size() != smallerShape.size()) {
                                firstBadRcShape = rcShape;
                                return false;
                            }
                            if (!MemoryPathUtils::IsDimMultiple(rcShape, smallerShape)) {
                                firstBadRcShape = rcShape;
                                return false;
                            }
                        }
                        return true;
                    };
                    if (!checkRegisterCopies(prodRegisterCopies) || !checkRegisterCopies(consRegisterCopies)) {
                        allowed = false;
                        firstBadProdShape = prodShape;
                        firstBadConsShape = consShape;
                        break;
                    }
                }
                if (!allowed) {
                    break;
                }
            }
        }
        if (allowed) {
            continue;
        }
        // 拦截：view→tobe=DDR; assemble→origin=DDR+DowngradeConsumerRequirements
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "ShapeTransportFallback intercepts %s[%d] on-chip path %s->%s to DDR: prodOut=%s consIn=%s "
                          "rcShape=%s.",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic(), BriefMemoryTypeToString(fromType).c_str(),
                          BriefMemoryTypeToString(toType).c_str(), IntVecToStr(firstBadProdShape).c_str(),
                          IntVecToStr(firstBadConsShape).c_str(), IntVecToStr(firstBadRcShape).c_str());
        if (opcode == Opcode::OP_VIEW) {
            inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
        } else {
            oOperand->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
            DowngradeConsumerRequirements(oOperand, fromType);
        }
    }
}

size_t AssignMemoryType::CalcNZTensorSizeLegacy(const LogicalTensorPtr& tensor) const
{
    constexpr int64_t kAlignBytes = 4;
    constexpr int64_t kC0AlignBytes = 32;
    DataType dtype = tensor->Datatype();
    int64_t bytes = BytesOf(dtype);
    size_t outer = tensor->GetShape()[0];
    size_t inner = tensor->GetShape()[1];
    // 外轴对齐：INT8/FP8 对齐到 32，其他对齐到 16
    size_t outerAlign = (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8) ? 32 : 16;
    // 内轴对齐：C0 size = 32 / 元素字节数
    size_t c0 = 0;
    if (bytes > 0) {
        c0 = static_cast<size_t>(kC0AlignBytes / bytes);
    }
    if (c0 <= 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "CalcNZTensorSize: invalid C0 size, c0=%zu", c0);
        // 返回原始 ND 格式大小作为 fallback
        return outer * inner * static_cast<size_t>(bytes > 0 ? bytes : kAlignBytes);
    }
    size_t alignedOuter = (outer + outerAlign - 1) / outerAlign * outerAlign + 1;
    size_t alignedInner = (inner + c0 - 1) / c0 * c0;
    // NZ 格式大小
    size_t nzSize = alignedOuter * alignedInner * static_cast<size_t>(bytes);
    // ND 格式原始大小
    size_t ndSize = outer * inner * static_cast<size_t>(bytes);
    // ND + NZ 同时存在，需要两者之和
    return ndSize + nzSize;
}
} // namespace npu::tile_fwk
