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
#include <set>
#include <unordered_map>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/simt_utils.h"
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
void AssignMemoryType::Init(Function& function)
{
    mode_ = config::EnableSlice() ? AssignMemoryTypeMode::SLICE_CONTRACT : AssignMemoryTypeMode::VIEW_ASSEMBLE;
    inserter = ConvertInserter(mode_);
    assembleTensorsByRaw_.clear();
    for (const auto& tensor : GraphUtils::GetAllTensors(function)) {
        if (tensor->tensor != nullptr) {
            assembleTensorsByRaw_[tensor->tensor->rawmagic].insert(tensor);
        }
    }
    outcastRawMagics_.emplace();
    for (const auto& tensor : function.GetOutcast()) {
        if (tensor != nullptr) {
            outcastRawMagics_->insert(tensor->GetRawMagic());
        }
    }
}

// 先推断并修正张量驻留及消费需求，再补齐搬运算子与形状信息。
Status AssignMemoryType::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start AssignMemoryType.");
    Init(function);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    if (IsSliceMode()) {
        RETURN_IF_NOT_SUCCESS(AssignConfirmedMemoryTypes(function));
        RETURN_IF_NOT_SUCCESS(InferUncertainMemoryTypes(function));
    } else {
        RETURN_IF_NOT_SUCCESS(AssignConfirmedMemoryTypesLegacy(function));
        RETURN_IF_NOT_SUCCESS(InferUncertainMemoryTypesLegacy(function));
    }
    RETURN_IF_NOT_SUCCESS(ResolveMemoryUnknowns(function));
    RETURN_IF_NOT_SUCCESS(ResolveInconsistentRawTensorMemoryTypes(function));
    if (IsSliceMode()) {
        RETURN_IF_NOT_SUCCESS(ResolveWritableReshapeMemoryTypes(function));
    }
    RETURN_IF_NOT_SUCCESS(SyncViewAssembleMemoryAttrs(function));
    RETURN_IF_NOT_SUCCESS(FixViewAssembleSemanticMismatch(function));
    RETURN_IF_NOT_SUCCESS(ResolveUnalignedUbMoves(function));
    RETURN_IF_NOT_SUCCESS(InsertConvertOpsAndInferShape(function));
    RETURN_IF_NOT_SUCCESS(NormalizeViewAssembleOpcodes(function));
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SyncTensorToBe(function));
    RETURN_IF_NOT_SUCCESS(MarkSimtOps(function));
    APASS_LOG_INFO_F(Elements::Function, "===> End AssignMemoryType.");
    return SUCCESS;
}

// 从图边界、操作码和搬运属性写入已知类型，为后续推断建立初始约束。
Status AssignMemoryType::AssignConfirmedMemoryTypes(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_SLICE) {
            RETURN_IF_NOT_SUCCESS(AssignViewAttrMemoryType(op));
            RETURN_IF_NOT_SUCCESS(AssignSliceInputRequirement(op));
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_CONTRACT) {
            RETURN_IF_NOT_SUCCESS(AssignAssembleAttrMemoryType(op));
        }
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignReduceAccInputRequirements(inserter, op));
        }
        if (OpChecker::check(op, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
            RETURN_IF_NOT_SUCCESS(AssignMatmulInputRequirements(op));
        }
        RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignOpcodeDefinedMemoryTypes(inserter, op));
        // shape 存在 -1（动态维度）的输出 tensor 无法作为本地 buffer 静态分配, 统一绑定 DDR
        RETURN_IF_NOT_SUCCESS(AssignDynamicShapeOutputMemoryType(op));
    }
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignInOutCastMemoryTypes(function));
    return MemoryPathUtils::EnsureAllConsumerRequirementsExist(inserter, function);
}

Status AssignMemoryType::AssignDynamicShapeOutputMemoryType(Operation& operation)
{
    for (auto& output : operation.oOperand) {
        const auto& outShape = output->GetShape();
        if (std::any_of(outShape.begin(), outShape.end(), [](int64_t dim) { return dim < 0; })) {
            MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "DynamicShapeTensorAsDdr");
        }
    }
    return SUCCESS;
}

// 为矩阵乘输入建立存储需求，并检查已有输入搬运是否符合约束。
Status AssignMemoryType::AssignMatmulInputRequirements(Operation& operation)
{
    for (auto& tensor : operation.iOperand) {
        for (const auto& producerOp : tensor->GetProducers()) {
            auto producerOpcode = producerOp->GetOpcode();
            MemoryType requirement = MemoryType::MEM_DEVICE_DDR;
            if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
                // kirin9050 宏指令：无 L0C，partial sum 与输出均驻留 L1
                requirement = IsMatmulMacroInstrOpcode(operation.GetOpcode()) ? MemoryType::MEM_L1 :
                                                                                MemoryType::MEM_L0C;
            } else if (IsModeMoveIn(mode_, producerOpcode)) {
                auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
                if (viewOpAttribute == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation,
                                      "View attribute is null for %s[%d] while assigning matmul input.",
                                      producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
                    return FAILED;
                }
                requirement = viewOpAttribute->GetTo();
                if (requirement == MemoryType::MEM_UNKNOWN) {
                    requirement = MemoryType::MEM_DEVICE_DDR;
                }
            } else if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                        OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                        OpChecker::OutputMemTypeChecker(MemoryType::MEM_L0A))) {
                requirement = MemoryType::MEM_L0A;
            } else if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                        OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                        OpChecker::OutputMemTypeChecker(MemoryType::MEM_L0B))) {
                requirement = MemoryType::MEM_L0B;
            }
            RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetRequirementChecked(inserter, tensor, operation, requirement,
                                                                         "AssignMatmulInputRequirements"));
            if (requirement != MemoryType::MEM_DEVICE_DDR && requirement != MemoryType::MEM_UNKNOWN) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Infer %s[%d] input tensor[%d] as %s.",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), tensor->GetMagic(),
                                  BriefMemoryTypeToString(requirement).c_str());
            }
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignViewAttrMemoryType(Operation& operation)
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
    RETURN_IF_NOT_SUCCESS(
        MemoryPathUtils::SetOriginalChecked(operation.oOperand.front(), attrToType, "AssignViewAttrMemoryType"));
    if (operation.GetOpcode() == Opcode::OP_VIEW) {
        RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetRequirementChecked(inserter, operation.iOperand.front(), operation,
                                                                     attrToType, "AssignViewAttrMemoryType"));
    }
    return SUCCESS;
}

// 从 SLICE 搬运属性确认输入需求，同时处理图边界和特殊直连约束。
Status AssignMemoryType::AssignSliceInputRequirement(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_SLICE) {
        return SUCCESS;
    }
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assign OP_SLICE[%d] input requirement failed because input operand is empty or null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.empty() ? nullptr : operation.oOperand.front();
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assign OP_SLICE[%d] input requirement failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType targetType = viewOpAttribute->GetTo();
    if (output != nullptr && output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        targetType = output->GetMemoryTypeOriginal();
    }
    if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
        (targetType == MemoryType::MEM_L0A || targetType == MemoryType::MEM_L0B)) {
        return MemoryPathUtils::SetRequirementChecked(inserter, input, operation, MemoryType::MEM_L1,
                                                      "AssignSliceInputRequirementLocalCopyIn");
    }
    MemoryType requirement = MemoryType::MEM_DEVICE_DDR;
    for (const auto& producerOp : input->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto producerViewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
        if (producerViewOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Assign OP_SLICE[%d] input requirement failed because producer OP_SLICE[%d] view attr is null.",
                operation.GetOpMagic(), producerOp->GetOpMagic());
            return FAILED;
        }
        if (producerViewOpAttribute->GetTo() == MemoryType::MEM_L1) {
            requirement = MemoryType::MEM_L1;
            break;
        }
    }
    return MemoryPathUtils::SetRequirementChecked(inserter, input, operation, requirement,
                                                  "AssignSliceInputRequirement");
}

Status AssignMemoryType::AssignAssembleAttrMemoryType(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_ASSEMBLE && opcode != Opcode::OP_CONTRACT)
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
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetRequirementChecked(inserter, operation.iOperand.front(), operation,
                                                                 attrFromType, "AssignAssembleAttrMemoryType"));
    if (opcode != Opcode::OP_ASSEMBLE) {
        return SUCCESS;
    }
    auto output = operation.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal == MemoryType::MEM_UNKNOWN) {
        return MemoryPathUtils::SetOriginalChecked(output, attrFromType, "AssignAssembleAttrMemoryType");
    }
    if (outputOriginal == attrFromType) {
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation,
                      "Assign assemble attr memory type failed because view operation memory type conflict exists. "
                      "%s[%d] from type is %s, output tensor[%d] original type is %s.",
                      operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                      BriefMemoryTypeToString(attrFromType).c_str(), output->GetMagic(),
                      BriefMemoryTypeToString(outputOriginal).c_str());
    return FAILED;
}

// 推断未定型的布局算子，处理特殊算子和缓冲区超限后，再尝试升级硬件直连路径。
Status AssignMemoryType::InferUncertainMemoryTypes(Function& function)
{
    if (!outcastRawMagics_.has_value()) {
        APASS_LOG_WARN_F(Elements::Function,
                         "AssignMemoryType inference called before Init; outcast protection is unavailable.");
    }
    std::unordered_set<LogicalTensorPtr> inferredAssembleOutputs;
    for (auto& op : function.Operations()) {
        switch (op.GetOpcode()) {
            case Opcode::OP_VIEW:
                RETURN_IF_NOT_SUCCESS(InferViewMemoryType(op));
                break;
            case Opcode::OP_SLICE:
                RETURN_IF_NOT_SUCCESS(InferSliceMemoryType(op));
                break;
            case Opcode::OP_VIEW_TYPE:
                RETURN_IF_NOT_SUCCESS(InferViewTypeMemoryType(op));
                break;
            case Opcode::OP_ASSEMBLE:
                RETURN_IF_NOT_SUCCESS(InferAssembleMemoryType(op, inferredAssembleOutputs));
                break;
            case Opcode::OP_CONTRACT:
                RETURN_IF_NOT_SUCCESS(InferContractMemoryType(op));
                break;
            case Opcode::OP_RESHAPE:
                RETURN_IF_NOT_SUCCESS(InferReshapeMemoryType(op));
                RETURN_IF_NOT_SUCCESS(InferReshapeL0C2UBAndUB2L1PatternLiteNPU(op));
                break;
            default:
                break;
        }
    }

    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::ApplyOtherSpecialOpcodeRules(inserter, function));
    RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(function));
    return ApplyPlatformPathUpgradeRules(function);
}

static bool IsViewSemanticOpcode(Opcode opcode)
{
    return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_RESHAPE;
}

Status AssignMemoryType::InferViewMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_VIEW, "Infer OP_VIEW memory type", input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryType inferredType = InferRequirementFromInputOriginals(input);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        inferredType = InferOriginalFromOutputRequirements(output);
    }
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, inferredType, "InferViewSemanticType");
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, inferredType, "InferViewSemanticType"));
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute != nullptr) {
        viewOpAttribute->SetToType(inferredType);
    }
    return SUCCESS;
}

// 推断 SLICE 的输出驻留并同步属性，无法确定时默认使用 UB，高维非卷积搬运使用 DDR。
Status AssignMemoryType::InferSliceMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_SLICE, "Infer OP_SLICE memory type", input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_SLICE[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    // 向量搬入最多支持五维，卷积使用独立的 L1 搬运路径。
    if (output->GetShape().size() >= SHAPE_DIM6 && !operation.GetBoolAttribute(OpAttributeKey::isConv)) {
        MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferHighRankSliceOutputOriginal");
        viewOpAttribute->SetToType(MemoryType::MEM_DEVICE_DDR);
        return SUCCESS;
    }
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        viewOpAttribute->SetToType(output->GetMemoryTypeOriginal());
        return SUCCESS;
    }
    MemoryType inferredType = InferOriginalFromOutputRequirements(output);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "InferSliceOutputOriginal: %s[%d] tensor[%d] defaults to UB after ambiguous inference.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), output->GetMagic());
        inferredType = MemoryType::MEM_UB;
    }
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, inferredType, "InferSliceOutputOriginal"));
    viewOpAttribute->SetToType(inferredType);
    return SUCCESS;
}

// 推断 CONTRACT 的输入需求并同步属性，高维搬出直接回退到 DDR。
Status AssignMemoryType::InferContractMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_CONTRACT, "Infer OP_CONTRACT memory type", input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_CONTRACT[%d] memory type failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    // 向量搬出最多支持五维，更高维的输入保留在 DDR。
    if (input->GetShape().size() >= SHAPE_DIM6) {
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR,
                                             "InferHighRankContractInputRequirement");
        assembleOpAttribute->SetFromType(MemoryType::MEM_DEVICE_DDR);
        return SUCCESS;
    }
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    if (inputRequirement != MemoryType::MEM_UNKNOWN) {
        assembleOpAttribute->SetFromType(inputRequirement);
        return SUCCESS;
    }
    MemoryType inferredType = InferRequirementFromInputOriginals(input);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "InferContractInputRequirement: %s[%d] tensor[%d] defaults to UB after ambiguous inference.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), input->GetMagic());
        inferredType = MemoryType::MEM_UB;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, inferredType, "InferContractInputRequirement");
    assembleOpAttribute->SetFromType(inferredType);
    return SUCCESS;
}

MemoryType AssignMemoryType::InferOriginalFromOutputRequirements(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    MemoryType inferred = InferOriginalFromOutputRequirements(tensor, visitedTensors);
    if (tensor != nullptr && inferred == MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Cannot infer tensor[%d] from consumer requirements: no unique known type.",
                          tensor->GetMagic());
    }
    return inferred;
}

MemoryType AssignMemoryType::InferOriginalFromOutputRequirements(
    const LogicalTensorPtr& tensor, std::unordered_set<const LogicalTensor*>& visitedTensors) const
{
    if (tensor == nullptr || !visitedTensors.insert(tensor.get()).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    auto addCandidate = [&candidates](MemoryType candidate) {
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    };
    auto consumerRequirements = inserter.GetConsumerRequirements(tensor);
    for (const auto& item : consumerRequirements) {
        Operation* consumerOp = item.first;
        MemoryType requirement = item.second;
        if (requirement != MemoryType::MEM_UNKNOWN) {
            addCandidate(requirement);
            continue;
        }
        if (consumerOp == nullptr || !IsViewSemanticOpcode(consumerOp->GetOpcode())) {
            continue;
        }
        for (const auto& output : consumerOp->oOperand) {
            addCandidate(InferOriginalFromOutputRequirements(output, visitedTensors));
        }
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
}

MemoryType AssignMemoryType::InferRequirementFromInputOriginals(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    MemoryType inferred = InferRequirementFromInputOriginals(tensor, visitedTensors);
    if (tensor != nullptr && inferred == MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Cannot infer tensor[%d] from producer originals: no unique known type.",
                          tensor->GetMagic());
    }
    return inferred;
}

MemoryType AssignMemoryType::InferRequirementFromInputOriginals(
    const LogicalTensorPtr& tensor, std::unordered_set<const LogicalTensor*>& visitedTensors) const
{
    if (tensor == nullptr || !visitedTensors.insert(tensor.get()).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    auto addCandidate = [&candidates](MemoryType candidate) {
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    };
    addCandidate(tensor->GetMemoryTypeOriginal());
    for (const auto& producerOp : tensor->GetProducers()) {
        if (producerOp == nullptr || !IsViewSemanticOpcode(producerOp->GetOpcode())) {
            continue;
        }
        for (const auto& input : producerOp->iOperand) {
            addCandidate(InferRequirementFromInputOriginals(input, visitedTensors));
        }
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
}

// 检查 A5 的 UB 到 L1 直连条件，MX 补齐和不支持的形状或类型保留 DDR 路径。
void AssignMemoryType::SetA5UbToL1DirectPath(Operation& operation, const LogicalTensorPtr& input, MemoryType from,
                                             MemoryType to, bool& directPath)
{
    if (MemoryPathUtils::IsMxPaddingMode(operation)) {
        directPath = false;
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Disable direct %s -> %s path for %s[%d] because MX matmul with K not 64-aligned "
                          "does not support UB2L1.",
                          BriefMemoryTypeToString(from).c_str(), BriefMemoryTypeToString(to).c_str(),
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return;
    }
    directPath = inserter.FitUB2L1(operation.iOperand.front());
    if (!directPath && input != nullptr) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Reject UB-to-L1 rank or dtype for %s[%d]: tensor[%d] shape=%s, dtype=%s.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), input->GetMagic(),
                          IntVecToStr(input->GetShape()).c_str(), DataType2String(input->Datatype()));
    }
}

// 判断特殊搬运是否可直连，源张量存在并行需求冲突时优先回退。
bool AssignMemoryType::TryHandleSpecialDirectMemoryPath(Operation& operation, MemoryType from, MemoryType to,
                                                        bool& directPath)
{
    LogicalTensorPtr input = operation.iOperand.empty() ? nullptr : operation.iOperand.front();
    if (MemoryPathUtils::IsSpecialDirectMemoryPath(from, to) && HasParallelDifferentConsumerRequirement(input, to)) {
        directPath = false;
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Disable direct %s -> %s path for %s[%d] because source tensor has parallel different requirements.",
            BriefMemoryTypeToString(from).c_str(), BriefMemoryTypeToString(to).c_str(),
            operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return true;
    }
    if (from == MemoryType::MEM_L0C && to == MemoryType::MEM_L1) {
        directPath = inserter.FitL0C2L1(operation);
        if (!directPath && input != nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Reject L0C-to-L1 transfer constraints for %s[%d]: tensor[%d] shape=%s, dtype=%s.",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), input->GetMagic(),
                              IntVecToStr(input->GetShape()).c_str(), DataType2String(input->Datatype()));
        }
        return true;
    }
    bool isA5 = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if (isA5 && from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) {
        directPath = (input != nullptr) && inserter.IsL0C2UbSupportedDtype(input);
        if (!directPath && input != nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Reject L0C-to-UB dtype for %s[%d]: tensor[%d] shape=%s, dtype=%s.",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), input->GetMagic(),
                              IntVecToStr(input->GetShape()).c_str(), DataType2String(input->Datatype()));
        }
        return true;
    }
    if (isA5 && from == MemoryType::MEM_UB && to == MemoryType::MEM_L1) {
        SetA5UbToL1DirectPath(operation, input, from, to, directPath);
        return true;
    }
    return false;
}

// 特殊进阶数据通路，不满足特定条件时回退到通过DDR搬运：L0C2L1, L0C2UB, UB2L1

bool AssignMemoryType::HasParallelDifferentConsumerRequirementStandard(const LogicalTensorPtr& tensor,
                                                                       MemoryType targetType) const
{
    if (tensor == nullptr || tensor->GetConsumers().size() <= 1) {
        return false;
    }
    return HasDifferentConsumerRequirement(tensor, targetType);
}

bool AssignMemoryType::HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor,
                                                               MemoryType targetType) const
{
    return IsSliceMode() ? HasParallelDifferentConsumerRequirementStandard(tensor, targetType) :
                           HasParallelDifferentConsumerRequirementLegacy(tensor, targetType);
}

// 检测 UB 分支末端的搬出是否必须拷贝，或明确要求不同的驻留类型。
static bool HasTerminalDifferentMoveConsumer(const LogicalTensorPtr& branchTensor, MemoryType targetType)
{
    if (targetType != MemoryType::MEM_UB || branchTensor == nullptr) {
        return false;
    }
    return std::any_of(
        branchTensor->GetConsumers().begin(), branchTensor->GetConsumers().end(), [targetType](Operation* consumerOp) {
            if (consumerOp == nullptr ||
                (consumerOp->GetOpcode() != Opcode::OP_CONTRACT && consumerOp->GetOpcode() != Opcode::OP_ASSEMBLE) ||
                consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                return false;
            }
            auto output = consumerOp->oOperand.front();
            bool needCopy = false;
            return output->GetConsumers().empty() &&
                   ((consumerOp->GetAttr<bool>(OpAttributeKey::needCopy, needCopy) && needCopy) ||
                    MemoryPathUtils::IsDifferentKnownRequirement(output->GetMemoryTypeOriginal(), targetType));
        });
}

// 检查直接和经视图传递的消费需求，识别会阻止直连升级的并行分支。
bool AssignMemoryType::HasDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const
{
    if (tensor == nullptr) {
        return false;
    }
    auto requirements = inserter.GetConsumerRequirements(tensor);
    bool hasDifferentRequirement = std::any_of(
        requirements.begin(), requirements.end(), [this, targetType](const auto& item) {
            auto resolveOutputRequirement = [this](const LogicalTensorPtr& output) {
                return MemoryPathUtils::InferUniqueRequirementThroughViewConsumers(inserter, output);
            };
            Operation* consumerOp = item.first;
            if (consumerOp != nullptr &&
                (consumerOp->GetOpcode() == Opcode::OP_VIEW || consumerOp->GetOpcode() == Opcode::OP_SLICE) &&
                !consumerOp->oOperand.empty() &&
                HasTerminalDifferentMoveConsumer(consumerOp->oOperand.front(), targetType)) {
                return true;
            }
            MemoryType requirement = MemoryPathUtils::ResolveEffectiveConsumerRequirement(
                consumerOp, item.second, targetType, resolveOutputRequirement);
            if (MemoryPathUtils::IsDifferentKnownRequirement(requirement, targetType)) {
                return true;
            }
            if (consumerOp == nullptr ||
                (consumerOp->GetOpcode() != Opcode::OP_CONTRACT && consumerOp->GetOpcode() != Opcode::OP_ASSEMBLE) ||
                consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                return false;
            }
            auto output = consumerOp->oOperand.front();
            auto outputRequirement = output->GetMemoryTypeOriginal();
            return output->GetConsumers().empty() &&
                   MemoryPathUtils::IsDifferentKnownRequirement(outputRequirement, targetType);
        });
    if (hasDifferentRequirement) {
        return true;
    }
    return HasTerminalDifferentMoveConsumer(tensor, targetType);
}

bool AssignMemoryType::CanUseDirectViewPath(Operation& operation, MemoryType from, MemoryType to)
{
    if (from == MemoryType::MEM_UNKNOWN || to == MemoryType::MEM_UNKNOWN)
        return false;
    if (from == to)
        return true;
    if (from != MemoryType::MEM_DEVICE_DDR && to == MemoryType::MEM_DEVICE_DDR)
        return false;
    bool directPath = false;
    if (TryHandleSpecialDirectMemoryPath(operation, from, to, directPath))
        return directPath;
    std::vector<MemoryType> paths;
    bool pathFound = Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (!pathFound || paths.empty())
        return false;
    static constexpr size_t DIRECT_MEMORY_PATH_LENGTH = 2;
    bool isDirectPath = paths.size() == DIRECT_MEMORY_PATH_LENGTH && paths.front() == from && paths.back() == to;
    return isDirectPath;
}

Status AssignMemoryType::InferAssembleMemoryType(Operation& operation,
                                                 std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_ASSEMBLE, "Infer OP_ASSEMBLE memory type", input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    if (inserter.GetRequirementOrUnknown(input, operation) != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    if (!inferredAssembleOutputs.insert(output).second) {
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(InferAssembleMemoryType(operation));
    PropagateMemoryTypeToRawTensorSiblings(output, inferredAssembleOutputs);
    return SUCCESS;
}

Status AssignMemoryType::InferAssembleMemoryType(Operation& operation)
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
    auto output = operation.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal == MemoryType::MEM_UNKNOWN) {
        outputOriginal = InferOriginalFromOutputRequirements(output);
    }
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        RETURN_IF_NOT_SUCCESS(
            MemoryPathUtils::SetOriginalChecked(output, outputOriginal, "InferAssembleOutputRequirement"));
        return SetParallelAssembleInputRequirements(output, outputOriginal, "InferAssembleOutputRequirement");
    }
    MemoryType inputRequirement = InferParallelAssembleInputRequirement(output);
    if (inputRequirement == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(
        MemoryPathUtils::SetOriginalChecked(output, inputRequirement, "InferAssembleInputRequirement"));
    return SetParallelAssembleInputRequirements(output, inputRequirement, "InferAssembleInputRequirement");
}

void AssignMemoryType::PropagateMemoryTypeToRawTensorSiblings(
    const LogicalTensorPtr& output, std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs)
{
    if (output == nullptr || output->tensor == nullptr) {
        return;
    }
    auto tensorsIt = assembleTensorsByRaw_.find(output->tensor->rawmagic);
    if (tensorsIt == assembleTensorsByRaw_.end()) {
        return;
    }
    const auto& siblings = tensorsIt->second;
    if (siblings.size() <= 1) {
        return;
    }
    MemoryType inferredType = output->GetMemoryTypeOriginal();
    for (const auto& sibling : siblings) {
        if (sibling == output || sibling == nullptr) {
            continue;
        }
        inferredAssembleOutputs.insert(sibling);
        if (sibling->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
            sibling->SetMemoryTypeOriginal(inferredType, false);
        }
    }
}

static bool IsMoveOutOutput(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    return std::any_of(tensor->GetProducers().begin(), tensor->GetProducers().end(), [](const auto* producer) {
        return producer != nullptr && IsAssembleLike(producer->GetOpcode());
    });
}

// 同一底层存储的搬出结果出现不同的已知驻留类型时，将这些结果统一回退到 DDR。
void AssignMemoryType::ResolveRawTensorSiblingMemoryTypes(int64_t rawMagic)
{
    const auto& siblings = assembleTensorsByRaw_.at(rawMagic);
    if (siblings.size() <= 1) {
        return;
    }
    MemoryType firstType = MemoryType::MEM_UNKNOWN;
    bool hasMoveOutOutput = false;
    bool inconsistent = false;
    for (const auto& sibling : siblings) {
        if (!IsMoveOutOutput(sibling)) {
            continue;
        }
        hasMoveOutOutput = true;
        MemoryType siblingType = sibling->GetMemoryTypeOriginal();
        if (siblingType == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        if (firstType == MemoryType::MEM_UNKNOWN) {
            firstType = siblingType;
        } else if (siblingType != firstType) {
            inconsistent = true;
            break;
        }
    }
    if (hasMoveOutOutput && inconsistent) {
        APASS_LOG_WARN_F(Elements::Tensor,
                         "Inconsistent memory types detected on rawMagic %ld across assemble/contract outputs, "
                         "falling back to DDR.",
                         static_cast<long>(rawMagic));
        for (const auto& sibling : siblings) {
            if (IsMoveOutOutput(sibling)) {
                MemoryPathUtils::ForceSetOriginal(sibling, MemoryType::MEM_DEVICE_DDR,
                                                  "ResolveInconsistentRawTensorMemoryTypes");
            }
        }
    }
}

// 按图中首次出现的顺序处理底层张量，每组搬出结果只检查一次。
Status AssignMemoryType::ResolveInconsistentRawTensorMemoryTypes(Function& function)
{
    std::unordered_set<int64_t> visitedRawMagic;
    for (auto& op : function.Operations()) {
        if (!IsAssembleLike(op.GetOpcode())) {
            continue;
        }
        if (op.oOperand.empty() || op.oOperand.front() == nullptr) {
            continue;
        }
        auto output = op.oOperand.front();
        if (output->tensor == nullptr) {
            continue;
        }
        int64_t rawMagic = output->tensor->rawmagic;
        if (!visitedRawMagic.insert(rawMagic).second) {
            continue;
        }
        ResolveRawTensorSiblingMemoryTypes(rawMagic);
    }
    return SUCCESS;
}

MemoryType AssignMemoryType::InferParallelAssembleInputRequirement(const LogicalTensorPtr& output) const
{
    if (output == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp)) {
            continue;
        }
        auto input = producerOp->iOperand.front();
        MemoryType candidate = inserter.GetRequirementOrUnknown(input, *producerOp);
        if (candidate == MemoryType::MEM_UNKNOWN) {
            candidate = InferRequirementFromInputOriginals(input);
        }
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    }
    if (candidates.empty()) {
        return MemoryType::MEM_UNKNOWN;
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Infer assemble output tensor[%d] as DDR: %zu conflicting input types.",
                      output->GetMagic(), candidates.size());
    return MemoryType::MEM_DEVICE_DDR;
}

Status AssignMemoryType::SetParallelAssembleInputRequirements(const LogicalTensorPtr& output, MemoryType memoryType,
                                                              const std::string& reason)
{
    if (output == nullptr || memoryType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp)) {
            continue;
        }
        auto input = producerOp->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because input tensor is null.",
                              producerOp->GetOpMagic());
            return FAILED;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, *producerOp, memoryType, reason);
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (assembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because assemble attr is null.",
                              producerOp->GetOpMagic());
            return FAILED;
        }
        assembleOpAttribute->SetFromType(memoryType);
    }
    return SUCCESS;
}

bool AssignMemoryType::CanUseDirectAssemblePath(Operation& operation, MemoryType from, MemoryType to)
{
    if (from == MemoryType::MEM_UNKNOWN || to == MemoryType::MEM_UNKNOWN) {
        return false;
    }
    if (from == to) {
        return true;
    }
    bool directPath = false;
    if (TryHandleSpecialDirectMemoryPath(operation, from, to, directPath)) {
        return directPath;
    }
    std::vector<MemoryType> paths;
    Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (paths.empty()) {
        return false;
    }
    bool hasDdr = std::find(paths.begin(), paths.end(), MemoryType::MEM_DEVICE_DDR) != paths.end();
    return !hasDdr;
}

Status AssignMemoryType::IsPaddedAssembleOffsetAligned(const Shape& rawShape, const Offset& toOffset,
                                                       int64_t tensorBytes, int64_t alignmentBytes, size_t padIdx,
                                                       bool& padAligned)
{
    int64_t alignElements = alignmentBytes / tensorBytes;
    padAligned = false;
    if (rawShape[padIdx] <= 0) {
        return SUCCESS;
    }
    Shape paddedShape = rawShape;
    paddedShape[padIdx] = (rawShape[padIdx] + alignElements - 1) / alignElements * alignElements;
    int64_t paddedOffset = 0;
    RETURN_IF_NOT_SUCCESS(CalcLineOffset(paddedShape, toOffset, paddedOffset));
    padAligned = (tensorBytes * paddedOffset) % alignmentBytes == 0;
    return SUCCESS;
}

// 模拟局部缓冲区可能采用的补齐方式，检查 ASSEMBLE 的写入起点是否均按 32 字节对齐。
Status AssignMemoryType::IsAssembleToOffsetAligned(Operation& operation, const LogicalTensorPtr& output, bool& aligned)
{
    aligned = false;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr || output == nullptr || output->GetRawTensor() == nullptr) {
        return FAILED;
    }
    const auto& rawShape = output->GetRawTensor()->rawshape;
    const auto& toOffset = assembleOpAttribute->GetToOffset();
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    static constexpr int64_t DEGENERATE_AXIS_SIZE = 1;
    static constexpr size_t MIN_RANK_WITH_SECOND_LAST_AXIS = 2;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(output->Datatype()));
    if (tensorBytes <= 0 || ASSEMBLE_ALIGN_BYTES % tensorBytes != 0 || rawShape.empty()) {
        aligned = true;
        return SUCCESS;
    }
    // 含动态维度（-1）时，编译期无法确定补齐形状或对齐结果。
    if (std::find(rawShape.begin(), rawShape.end(), -1) != rawShape.end()) {
        return SUCCESS;
    }
    size_t lastIdx = rawShape.size() - 1;
    // 未启用轴合并或张量不满足合并条件时，PadLocalBuffer 将末轴补齐到 32 字节。
    bool tailPadAligned = false;
    RETURN_IF_NOT_SUCCESS(
        IsPaddedAssembleOffsetAligned(rawShape, toOffset, tensorBytes, ASSEMBLE_ALIGN_BYTES, lastIdx, tailPadAligned));
    // 末轴为 1 且存在倒数第二轴时，AxisCombine 可能为 ASSEMBLE 补齐倒数第二轴。
    // 此时还无法确定实际补齐的轴，两种结果都对齐才可避免回退到 DDR。
    if (rawShape[lastIdx] == DEGENERATE_AXIS_SIZE && rawShape.size() >= MIN_RANK_WITH_SECOND_LAST_AXIS) {
        bool secondLastPadAligned = false;
        RETURN_IF_NOT_SUCCESS(IsPaddedAssembleOffsetAligned(rawShape, toOffset, tensorBytes, ASSEMBLE_ALIGN_BYTES,
                                                            lastIdx - 1, secondLastPadAligned));
        aligned = tailPadAligned && secondLastPadAligned;
        return SUCCESS;
    }
    aligned = tailPadAligned;
    return SUCCESS;
}

// 推断 reshape 的输入需求与输出驻留，处理可写别名和分组 Conv1D 的特殊约束。
Status AssignMemoryType::InferReshapeMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_RESHAPE, "Infer OP_RESHAPE memory type", input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    // 保留源驻留类型，待内存路径回退后再统一处理别名组。
    if (operation.GetInplaceIndex(0) >= 0 && output->GetProducers().size() > 1) {
        MemoryType storageType = input->GetMemoryTypeOriginal();
        if (storageType == MemoryType::MEM_UNKNOWN) {
            storageType = MemoryType::MEM_DEVICE_DDR;
        }
        MemoryPathUtils::ForceSetOriginal(input, storageType, "MutableInplaceReshape");
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, storageType, "MutableInplaceReshape");
        MemoryPathUtils::ForceSetOriginal(output, storageType, "MutableInplaceReshape");
        return SUCCESS;
    }
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    MemoryType inputRequirement = MemoryPathUtils::GetReshapeInputRequirement(inserter, operation, input,
                                                                              inputOriginal);
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::InferReshapeOutputFromRequirement(inserter, output, outputOriginal));
    bool isConv1DGroupReshape = false;
    operation.GetAttr(OpAttributeKey::groupReshapeNoSplit, isConv1DGroupReshape);
    bool kept = false;
    // 沿用旧路径的规则，分组 Conv1D 的 reshape 须保留在 DDR。
    if (!isConv1DGroupReshape) {
        RETURN_IF_NOT_SUCCESS(KeepSplitReshapeUb(operation, input, output, kept));
    }
    if (kept) {
        return SUCCESS;
    }
    bool isDynamic = MemoryPathUtils::IsDynamicReshape(operation, output);
    bool canUseUb = MemoryPathUtils::CanUseUbForReshape(input, output, inputRequirement, outputOriginal) &&
                    !isConv1DGroupReshape;
    return MemoryPathUtils::ApplyReshapeMemoryType(inserter, operation, input, output, isDynamic, canUseUb);
}

// 收集图边界和算子的张量，按底层存储编号归组。
static std::unordered_map<int64_t, TensorSet> CollectReshapeTensorsByRaw(Function& function,
                                                                         const TensorSet& interfaceTensors)
{
    std::unordered_map<int64_t, TensorSet> tensorsByRaw;
    for (const auto& tensor : interfaceTensors) {
        tensorsByRaw[tensor->GetRawMagic()].insert(tensor);
    }
    for (auto& op : function.Operations()) {
        for (const auto& tensor : op.GetIOperands()) {
            tensorsByRaw[tensor->GetRawMagic()].insert(tensor);
        }
        for (const auto& tensor : op.GetOOperands()) {
            tensorsByRaw[tensor->GetRawMagic()].insert(tensor);
        }
    }
    return tensorsByRaw;
}

// 沿原地 reshape 的别名边收集连通分量，同一底层存储只访问一次。
static TensorSet CollectReshapeAliasComponent(int64_t raw, std::unordered_set<int64_t>& visited,
                                              std::unordered_map<int64_t, TensorSet>& tensorsByRaw,
                                              std::unordered_map<int64_t, std::vector<int64_t>>& aliases)
{
    TensorSet tensors;
    std::vector<int64_t> pending{raw};
    while (!pending.empty()) {
        auto current = pending.back();
        pending.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        tensors.insert(tensorsByRaw[current].begin(), tensorsByRaw[current].end());
        pending.insert(pending.end(), aliases[current].begin(), aliases[current].end());
    }
    return tensors;
}

// 为可写别名组选择统一驻留类型，遇到边界、容量、对齐或不可直连约束时回退到 DDR。
Status AssignMemoryType::InferWritableReshapeStorageType(const TensorSet& tensors, const TensorSet& interfaceTensors,
                                                         MemoryType& storageType)
{
    std::set<MemoryType> types;
    for (const auto& tensor : tensors) {
        types.insert(tensor->GetMemoryTypeOriginal());
        if (interfaceTensors.count(tensor) != 0) {
            types.insert(MemoryType::MEM_DEVICE_DDR);
        }
    }
    storageType = types.size() == 1 ? *types.begin() : MemoryType::MEM_DEVICE_DDR;
    if (storageType == MemoryType::MEM_UNKNOWN) {
        storageType = MemoryType::MEM_DEVICE_DDR;
    }
    for (const auto& tensor : tensors) {
        if (storageType == MemoryType::MEM_DEVICE_DDR) {
            break;
        }
        if (tensor->GetDataSize() < 0 || MemoryPathUtils::IsOversizedLocalBuffer(tensor, storageType, true, true)) {
            storageType = MemoryType::MEM_DEVICE_DDR;
            break;
        }
        for (auto* producer : tensor->GetProducers()) {
            if (producer == nullptr || !IsAssembleLike(producer->GetOpcode())) {
                continue;
            }
            MemoryType fromType = MemoryPathUtils::GetAssembleInputType(inserter, *producer);
            bool checkOffsetAlignment = !MemoryPathUtils::IsAdvancedMemoryPath(fromType, storageType);
            bool aligned = false;
            RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(*producer, tensor, aligned));
            if ((checkOffsetAlignment && !aligned) || !CanUseDirectAssemblePath(*producer, fromType, storageType)) {
                storageType = MemoryType::MEM_DEVICE_DDR;
                break;
            }
        }
    }
    return SUCCESS;
}

// 先统一张量的驻留类型，再按图中原有的 reshape 顺序写回需求。
void AssignMemoryType::ApplyWritableReshapeStorageType(
    const TensorSet& tensors, MemoryType storageType, const std::vector<Operation*>& reshapes,
    const std::unordered_map<int, std::vector<size_t>>& reshapeIndicesByInputMagic)
{
    for (const auto& tensor : tensors) {
        MemoryPathUtils::ForceSetOriginal(tensor, storageType, "ResolveWritableReshape");
    }
    // TensorSet 按逻辑编号识别张量，写回需求时恢复图中的 reshape 顺序。
    std::vector<size_t> componentReshapeIndices;
    for (const auto& tensor : tensors) {
        auto it = reshapeIndicesByInputMagic.find(tensor->GetMagic());
        if (it != reshapeIndicesByInputMagic.end()) {
            componentReshapeIndices.insert(componentReshapeIndices.end(), it->second.begin(), it->second.end());
        }
    }
    std::sort(componentReshapeIndices.begin(), componentReshapeIndices.end());
    for (auto index : componentReshapeIndices) {
        auto* reshape = reshapes[index];
        MemoryPathUtils::ForceSetRequirement(inserter, reshape->iOperand.front(), *reshape, storageType,
                                             "ResolveWritableReshape");
    }
}

// 按原地 reshape 的别名关系为共享存储统一定型，仅处理含可写输出的分量。
Status AssignMemoryType::ResolveWritableReshapeMemoryTypes(Function& function)
{
    std::unordered_map<int64_t, std::vector<int64_t>> aliases;
    std::vector<Operation*> reshapes;
    std::vector<int64_t> writableRaws;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE || op.GetInplaceIndex(0) < 0) {
            continue;
        }
        auto inputRaw = op.iOperand.front()->GetRawMagic();
        auto outputRaw = op.oOperand.front()->GetRawMagic();
        aliases[inputRaw].push_back(outputRaw);
        aliases[outputRaw].push_back(inputRaw);
        reshapes.push_back(&op);
        if (op.oOperand.front()->GetProducers().size() > 1) {
            writableRaws.push_back(inputRaw);
        }
    }
    if (writableRaws.empty()) {
        return SUCCESS;
    }
    TensorSet interfaceTensors(function.inCasts_.begin(), function.inCasts_.end());
    interfaceTensors.insert(function.outCasts_.begin(), function.outCasts_.end());
    auto tensorsByRaw = CollectReshapeTensorsByRaw(function, interfaceTensors);
    std::unordered_set<int64_t> visited;
    std::unordered_map<int, std::vector<size_t>> reshapeIndicesByInputMagic;
    for (size_t i = 0; i < reshapes.size(); ++i) {
        reshapeIndicesByInputMagic[reshapes[i]->iOperand.front()->GetMagic()].push_back(i);
    }
    for (auto raw : writableRaws) {
        if (visited.count(raw) != 0) {
            continue;
        }
        const auto tensors = CollectReshapeAliasComponent(raw, visited, tensorsByRaw, aliases);
        MemoryType storageType = MemoryType::MEM_UNKNOWN;
        RETURN_IF_NOT_SUCCESS(InferWritableReshapeStorageType(tensors, interfaceTensors, storageType));
        ApplyWritableReshapeStorageType(tensors, storageType, reshapes, reshapeIndicesByInputMagic);
    }
    return SUCCESS;
}

Status AssignMemoryType::InferViewTypeMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_VIEW_TYPE, "Infer OP_VIEW_TYPE memory type", input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    MemoryType outputRequirement = output == nullptr ? MemoryType::MEM_UNKNOWN :
                                                       inserter.TryGetUniqueKnownRequiredType(output);
    // 输出 toBeMap 未知时，沿后续未推导的视图链向前查找有效内存类型
    if (outputRequirement == MemoryType::MEM_UNKNOWN) {
        MemoryType forwarded = MemoryPathUtils::InferTargetTypeThroughForwardViews(inserter, output);
        if (forwarded != MemoryType::MEM_UNKNOWN) {
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Infer OP_VIEW_TYPE[%d] memory type reused from forward view requirement %s for output tensor[%d].",
                operation.GetOpMagic(), BriefMemoryTypeToString(forwarded).c_str(), output->GetMagic());
            outputRequirement = forwarded;
        }
    }
    MemoryType targetType = outputRequirement != MemoryType::MEM_UNKNOWN ? outputRequirement : outputOriginal;
    bool handled = false;
    RETURN_IF_NOT_SUCCESS(TryInferViewTypeFromProducerMoveIn(operation, input, output, targetType, handled));
    if (handled) {
        return SUCCESS;
    }
    return MemoryPathUtils::InferViewTypeInput(inserter, operation, input, output, targetType);
}

Status AssignMemoryType::TryInferViewTypeFromProducerMoveIn(Operation& operation, const LogicalTensorPtr& input,
                                                            const LogicalTensorPtr& output, MemoryType targetType,
                                                            bool& handled)
{
    handled = false;
    auto& producers = input->GetProducers();
    if (producers.empty()) {
        return SUCCESS;
    }
    auto producer = *producers.begin();
    if (producer == nullptr || !IsModeMoveIn(mode_, producer->GetOpcode())) {
        return SUCCESS;
    }
    handled = true;
    if (producer->iOperand.empty()) {
        const char* opcodeName = IsSliceMode() ? "OP_SLICE" : "OP_VIEW";
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_VIEW_TYPE[%d] memory type failed because producer %s[%d] input is empty.",
                          operation.GetOpMagic(), opcodeName, producer->GetOpMagic());
        return FAILED;
    }
    auto moveInput = producer->iOperand.front();
    MemoryType moveInputRequirement = inserter.GetRequirementOrUnknown(moveInput, *producer);
    if (moveInputRequirement == MemoryType::MEM_UNKNOWN) {
        moveInputRequirement = moveInput->GetMemoryTypeOriginal();
    }
    const char* reason = IsSliceMode() ? "InferViewTypeProducerSlice" : "InferViewTypeProducerView";
    const char* fallbackReason = IsSliceMode() ? "InferViewTypeProducerSliceFallback" :
                                                 "InferViewTypeProducerViewFallback";
    if (targetType != MemoryType::MEM_UNKNOWN && CanUseDirectViewPath(*producer, moveInputRequirement, targetType)) {
        MemoryPathUtils::ForceSetOriginal(input, targetType, reason);
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, targetType, reason);
        MemoryPathUtils::ForceSetOriginal(output, targetType, reason);
        return SUCCESS;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, fallbackReason);
    MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, fallbackReason);
    return SUCCESS;
}

// Batch matmul 的 batch 轴偏移会导致 L0C2UB 精度异常，跳过 UB 推导
static bool HasBatchMatmulContractInput(const LogicalTensorPtr& input, const Operation& reshape)
{
    for (auto* contract : input->GetProducers()) {
        for (const auto& tile : contract->iOperand) {
            if (tile == nullptr || tile->GetShape().size() <= SHAPE_DIM2) {
                continue;
            }
            for (auto* source : tile->GetProducers()) {
                if (source != nullptr && OpChecker::check(source, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
                    APASS_LOG_DEBUG_F(Elements::Operation,
                                      "Skip split reshape[%d] UB retention: contract[%d] input tensor[%d] "
                                      "shape=%s comes from batch matmul[%d].",
                                      reshape.GetOpMagic(), contract->GetOpMagic(), tile->GetMagic(),
                                      IntVecToStr(tile->GetShape()).c_str(), source->GetOpMagic());
                    return true;
                }
            }
        }
    }
    return false;
}

// 容量和搬出约束满足后才修改 reshape 及其并行切片分支的驻留类型。
Status AssignMemoryType::TrySetSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input,
                                              const LogicalTensorPtr& output, bool allProducersContract,
                                              bool allConsumersSlice, bool& kept)
{
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_ASSEMBLE);
    int64_t inputDataSize = input->GetDataSize();
    if (allProducersContract && allConsumersSlice && inputDataSize >= 0 &&
        static_cast<size_t>(inputDataSize) <= ubThreshold) {
        if (HasBatchMatmulContractInput(input, operation)) {
            return SUCCESS;
        }
        bool canKeepProducers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepContractProducersInUb(input, canKeepProducers));
        if (canKeepProducers) {
            constexpr const char* kReason = "InferSplitReshapeUb";
            MemoryPathUtils::ForceSetOriginal(input, MemoryType::MEM_UB, kReason);
            MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_UB, kReason);
            // 仅在确认 UB 升级后写回并行切片的需求。
            for (auto* consumer : input->GetConsumers()) {
                if (consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE) {
                    MemoryPathUtils::ForceSetRequirement(inserter, input, *consumer, MemoryType::MEM_UB, kReason);
                }
            }
            MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_UB, kReason);
            for (const auto& consumerOp : output->GetConsumers()) {
                if (consumerOp != nullptr) {
                    MemoryPathUtils::ForceSetRequirement(inserter, output, *consumerOp, MemoryType::MEM_UB, kReason);
                }
            }
            kept = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}

// 先推断输入张量的并行切片，若有分支需要其他驻留类型，则保留搬入边界。
Status AssignMemoryType::InferParallelSplitReshapeSlices(Operation& operation, const LogicalTensorPtr& input,
                                                         bool& compatible)
{
    for (auto* consumer : input->GetConsumers()) {
        if (consumer == &operation || consumer->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(InferSliceMemoryType(*consumer));
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
        if (attr != nullptr && attr->GetTo() != MemoryType::MEM_UNKNOWN && attr->GetTo() != MemoryType::MEM_UB) {
            compatible = false;
            return SUCCESS;
        }
    }
    compatible = true;
    return SUCCESS;
}

// 先推断上下游切片，再判断 CONTRACT → reshape → SLICE 是否可保留在 UB。
Status AssignMemoryType::KeepSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input,
                                            const LogicalTensorPtr& output, bool& kept)
{
    kept = false;
    if (input == nullptr || output == nullptr || IsOutCastTensor(input) || IsOutCastTensor(output)) {
        return SUCCESS;
    }
    bool parallelSlicesCompatible = false;
    RETURN_IF_NOT_SUCCESS(InferParallelSplitReshapeSlices(operation, input, parallelSlicesCompatible));
    if (!parallelSlicesCompatible) {
        return SUCCESS;
    }
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    if (producers.empty() || consumers.empty()) {
        return SUCCESS;
    }
    bool allProducersContract = std::all_of(producers.begin(), producers.end(), [](const auto& producer) {
        return producer != nullptr && producer->GetOpcode() == Opcode::OP_CONTRACT;
    });
    // 预先根据下游需求确定切片目标，避免 UB 升级后未知目标又回退到 DDR。
    for (auto* consumer : consumers) {
        if (consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE) {
            RETURN_IF_NOT_SUCCESS(InferSliceMemoryType(*consumer));
        }
    }
    bool allConsumersSlice = std::all_of(consumers.begin(), consumers.end(), [](const auto& consumer) {
        if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_SLICE) {
            return false;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
        return viewOpAttribute != nullptr && viewOpAttribute->GetTo() == MemoryType::MEM_UB;
    });
    return TrySetSplitReshapeUb(operation, input, output, allProducersContract, allConsumersSlice, kept);
}

Status AssignMemoryType::CanKeepContractProducersInUb(const LogicalTensorPtr& tensor, bool& canKeep)
{
    canKeep = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* producerOp : tensor->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_CONTRACT || producerOp->iOperand.empty()) {
            return SUCCESS;
        }
        MemoryType fromType = MemoryPathUtils::GetAssembleInputType(inserter, *producerOp);
        constexpr MemoryType targetType = MemoryType::MEM_UB;
        bool checkOffsetAlignment = !MemoryPathUtils::IsAdvancedMemoryPath(fromType, targetType);
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(*producerOp, tensor, aligned));
        if ((checkOffsetAlignment && !aligned) || !CanUseDirectAssemblePath(*producerOp, fromType, targetType)) {
            return SUCCESS;
        }
    }
    canKeep = true;
    return SUCCESS;
}

Status AssignMemoryType::IsSliceFromOffsetAligned(Operation& sliceOp, const LogicalTensorPtr& input, bool& aligned)
{
    aligned = false;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    if (viewOpAttribute == nullptr || input == nullptr) {
        return SUCCESS;
    }
    int64_t lineOffset = 0;
    RETURN_IF_NOT_SUCCESS(
        CalcLineOffset(input->GetRawTensor()->rawshape, viewOpAttribute->GetFromOffset(), lineOffset));
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(input->Datatype()));
    aligned = (tensorBytes * lineOffset) % ASSEMBLE_ALIGN_BYTES == 0;
    return SUCCESS;
}

Status AssignMemoryType::CanKeepSliceConsumersInUb(const LogicalTensorPtr& tensor, bool& canKeep)
{
    canKeep = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* consumerOp : tensor->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_SLICE || consumerOp->oOperand.empty() ||
            consumerOp->oOperand.front() == nullptr) {
            return SUCCESS;
        }
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsSliceFromOffsetAligned(*consumerOp, tensor, aligned));
        if (!aligned) {
            return SUCCESS;
        }
    }
    canKeep = true;
    return SUCCESS;
}

Status AssignMemoryType::HasNonZeroSliceFromOffset(const LogicalTensorPtr& tensor, bool& hasNonZero)
{
    hasNonZero = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* consumerOp : tensor->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewAttr == nullptr) {
            continue;
        }
        int64_t lineOffset = 0;
        RETURN_IF_NOT_SUCCESS(CalcLineOffset(tensor->GetRawTensor()->rawshape, viewAttr->GetFromOffset(), lineOffset));
        if (lineOffset > 0) {
            hasNonZero = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallback(Function& function)
{
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_ASSEMBLE);
    const size_t ubNormalThreshold = static_cast<size_t>(
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_NORMAL);
    const size_t l1Threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) *
                                                   L1_THRESHOLD);
    if (IsSliceMode()) {
        APASS_LOG_INFO_F(Elements::Function, "Memory threshold: UB strict %zu, UB normal %zu, L1 %zu.", ubThreshold,
                         ubNormalThreshold, l1Threshold);
    } else {
        APASS_LOG_INFO_F(Elements::Function,
                         "Memory threshold: UB assemble %zu bytes, UB normal %zu bytes, L1 %zu bytes.", ubThreshold,
                         ubNormalThreshold, l1Threshold);
    }
    for (auto& op : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(op));
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallback(Operation& operation)
{
    return IsSliceMode() ? ApplyOversizedLocalBufferFallbackStandard(operation) :
                           ApplyOversizedLocalBufferFallbackLegacy(operation);
}

Status AssignMemoryType::ApplyOversizedContractInputFallback(Operation& operation, const LogicalTensorPtr& input)
{
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    if (!MemoryPathUtils::IsOversizedLocalBuffer(input, inputRequirement, true, true)) {
        return SUCCESS;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR,
                                         "ApplyOversizedContractInputFallback");
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for OP_CONTRACT[%d] failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    assembleAttr->SetFromType(MemoryType::MEM_DEVICE_DDR);
    APASS_LOG_DEBUG_F(Elements::Operation, "Force OP_CONTRACT[%d] input tensor[%d] requirement to DDR by size limit.",
                      operation.GetOpMagic(), input->GetMagic());
    return SUCCESS;
}

// 对超限的 CONTRACT 输入或 SLICE 输出回退到 DDR，并同步对应的搬运属性。
Status AssignMemoryType::ApplyOversizedLocalBufferFallbackStandard(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_SLICE && opcode != Opcode::OP_CONTRACT) {
        return SUCCESS;
    }
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Apply oversized fallback for %s[%d] failed because operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.front();
    if (input == nullptr || output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because operand tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }

    if (opcode == Opcode::OP_CONTRACT) {
        return ApplyOversizedContractInputFallback(operation, input);
    }

    if (MemoryPathUtils::IsOversizedLocalBuffer(output, output->GetMemoryTypeOriginal(), false, false)) {
        MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedSliceOutputFallback");
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
        if (viewAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Apply oversized fallback for OP_SLICE[%d] failed because view attr is null.",
                              operation.GetOpMagic());
            return FAILED;
        }
        viewAttr->SetToType(MemoryType::MEM_DEVICE_DDR);
        APASS_LOG_DEBUG_F(Elements::Operation, "Force OP_SLICE[%d] output tensor[%d] to DDR by size limit.",
                          operation.GetOpMagic(), output->GetMagic());
    }
    return DowngradeOversizedMoveInInputRequirement(operation);
}

Status AssignMemoryType::DowngradeOversizedMoveInInputRequirement(Operation& operation)
{
    const char* opcodeName = IsSliceMode() ? "OP_SLICE" : "OP_VIEW";
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because of invalid input operand.", opcodeName,
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    MemoryType inputType = inserter.GetRequirementOrUnknown(input, operation);
    if (!MemoryPathUtils::IsOversizedLocalBuffer(input, inputType, false, false)) {
        return SUCCESS;
    }
    const char* reason = IsSliceMode() ? "ApplyOversizedSliceInputFallback" : "ApplyOversizedViewInputFallback";
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, reason);
    APASS_LOG_DEBUG_F(Elements::Operation, "Force %s[%d] input tensor[%d] requirement to DDR by size limit.",
                      opcodeName, operation.GetOpMagic(), input->GetMagic());
    return SUCCESS;
}

Status AssignMemoryType::ResolveMemoryUnknowns(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visited;
    auto resolveTensor = [this, &function, &visited](const LogicalTensorPtr& tensor) -> Status {
        if (tensor != nullptr && !visited.insert(tensor).second) {
            return SUCCESS;
        }
        if (ShouldResolveExplicitUnknownRequirementToDdr(function, tensor)) {
            MemoryPathUtils::ForceSetOriginal(tensor, MemoryType::MEM_DEVICE_DDR,
                                              "ResolveExplicitUnknownSliceFromInCast");
            for (auto* consumer : tensor->GetConsumers()) {
                if (consumer != nullptr) {
                    MemoryPathUtils::ForceSetRequirement(inserter, tensor, *consumer, MemoryType::MEM_DEVICE_DDR,
                                                         "ResolveExplicitUnknownSliceFromInCast");
                }
            }
        }
        return MemoryPathUtils::ResolveTensorMemoryUnknowns(inserter, tensor);
    };
    for (auto& op : function.Operations()) {
        for (auto& input : op.iOperand) {
            RETURN_IF_NOT_SUCCESS(resolveTensor(input));
        }
        for (auto& output : op.oOperand) {
            RETURN_IF_NOT_SUCCESS(resolveTensor(output));
        }
    }
    return SUCCESS;
}

// 区分操作码在该输入位置显式声明未知类型，与未定义该位置的存储类型。
static bool HasExplicitUnknownConsumerTypes(const LogicalTensorPtr& tensor)
{
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr) {
            return false;
        }
        auto it = std::find(consumer->iOperand.begin(), consumer->iOperand.end(), tensor);
        if (it == consumer->iOperand.end()) {
            return false;
        }
        size_t inputIndex = static_cast<size_t>(std::distance(consumer->iOperand.begin(), it));
        const auto& definedTypes = OpcodeManager::Inst().GetInputsMemType(consumer->GetOpcode());
        if (inputIndex >= definedTypes.size() || definedTypes[inputIndex] != MemoryType::MEM_UNKNOWN) {
            return false;
        }
    }
    return true;
}

// 入参直接切出的张量仅在所有消费者均显式要求未知类型时保留在 DDR。
bool AssignMemoryType::ShouldResolveExplicitUnknownRequirementToDdr(const Function& function,
                                                                    const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetConsumers().empty()) {
        return false;
    }
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || inserter.GetRequirementOrUnknown(tensor, *consumer) != MemoryType::MEM_UNKNOWN) {
            return false;
        }
    }

    // 仅匹配由图入参直接供数的 SLICE。
    Operation* slice = nullptr;
    for (auto* producer : tensor->GetProducers()) {
        if (producer == nullptr || producer->GetOpcode() != Opcode::OP_SLICE || producer->oOperand.empty() ||
            producer->oOperand.front() != tensor || producer->iOperand.empty() ||
            producer->iOperand.front() == nullptr) {
            return false;
        }
        if (slice != nullptr) {
            return false;
        }
        slice = producer;
    }
    if (slice == nullptr || std::find(function.inCasts_.begin(), function.inCasts_.end(), slice->iOperand.front()) ==
                                function.inCasts_.end()) {
        return false;
    }
    if (std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute()) == nullptr) {
        return false;
    }

    return HasExplicitUnknownConsumerTypes(tensor);
}

Status AssignMemoryType::SyncViewAssembleMemoryAttrs(Function& function)
{
    for (auto& operation : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(SyncViewMemoryAttr(operation));
        RETURN_IF_NOT_SUCCESS(SyncAssembleMemoryAttr(operation));
    }
    return SUCCESS;
}

Status AssignMemoryType::SyncViewMemoryAttr(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_VIEW && operation.GetOpcode() != Opcode::OP_SLICE) {
        return SUCCESS;
    }
    if (operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because output operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because view attr is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto output = operation.oOperand.front();
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because output tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType toType = output->GetMemoryTypeOriginal();
    if (toType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    viewOpAttribute->SetToType(toType);
    return SUCCESS;
}

Status AssignMemoryType::SyncAssembleMemoryAttr(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE && operation.GetOpcode() != Opcode::OP_CONTRACT) {
        return SUCCESS;
    }
    if (operation.iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because input operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because assemble attr is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    if (input == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because input tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
    if (fromType == MemoryType::MEM_UNKNOWN) {
        fromType = input->GetMemoryTypeOriginal();
    }
    if (fromType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    assembleOpAttribute->SetFromType(fromType);
    return SUCCESS;
}

// 将不满足对齐要求的 UB 搬运回退到 DDR，并同步张量需求及搬运属性。
Status AssignMemoryType::ResolveUnalignedUbMoves(Function& function)
{
    for (auto& op : function.Operations()) {
        const auto opcode = op.GetOpcode();
        const bool isSlice = opcode == Opcode::OP_SLICE;
        const bool isAssembleLike = opcode == Opcode::OP_CONTRACT || opcode == Opcode::OP_ASSEMBLE;
        if (!isSlice && !isAssembleLike) {
            continue;
        }
        if (op.iOperand.empty() || op.oOperand.empty()) {
            continue;
        }
        auto input = op.iOperand.front();
        auto output = op.oOperand.front();
        if (input == nullptr || output == nullptr) {
            continue;
        }
        if (input->GetMemoryTypeOriginal() != MemoryType::MEM_UB ||
            output->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            continue;
        }
        bool aligned = false;
        if (isSlice) {
            RETURN_IF_NOT_SUCCESS(IsSliceFromOffsetAligned(op, input, aligned));
            if (aligned) {
                continue;
            }
            // 同内存 SLICE 后续会转成共享缓冲区 VIEW，非对齐时转到 DDR，保证访问对齐。
            MemoryPathUtils::ForceSetRequirement(inserter, input, op, MemoryType::MEM_DEVICE_DDR,
                                                 "ResolveUnalignedUbSlice");
            continue;
        }

        // 搬出类的 CONTRACT 和 ASSEMBLE
        RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(op, output, aligned));
        if (aligned) {
            continue;
        }
        // 非对齐拼接输出回退 DDR，避免后续访问共享 UB 的非对齐区域。
        MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "ResolveUnalignedUbMoveOutput");
    }
    return SUCCESS;
}

// 根据驻留和消费需求区分布局变换与搬运，并统一相关操作码。
Status AssignMemoryType::NormalizeViewAssembleOpcodes(Function& function)
{
    if (!IsSliceMode()) {
        return SUCCESS;
    }
    for (auto& operation : function.Operations()) {
        auto opcode = operation.GetOpcode();
        bool isViewFamily = opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE;
        bool isAssembleFamily = opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT;
        if (!isViewFamily && !isAssembleFamily) {
            continue;
        }
        if (operation.iOperand.empty() || operation.oOperand.empty() || operation.iOperand.front() == nullptr ||
            operation.oOperand.front() == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Normalize view/assemble opcode for %s[%d] failed because operand is invalid.",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
            return FAILED;
        }
        MemoryType inputType = operation.iOperand.front()->GetMemoryTypeOriginal();
        MemoryType outputType = operation.oOperand.front()->GetMemoryTypeOriginal();
        if (inputType == MemoryType::MEM_UNKNOWN || outputType == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        Opcode normalizedOpcode;
        if (isViewFamily) {
            normalizedOpcode = inputType == outputType ? Opcode::OP_VIEW : Opcode::OP_SLICE;
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
            if (viewOpAttribute != nullptr) {
                viewOpAttribute->SetToType(outputType);
            }
        } else {
            normalizedOpcode = inputType == outputType ? Opcode::OP_ASSEMBLE : Opcode::OP_CONTRACT;
            auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
            if (assembleOpAttribute != nullptr) {
                assembleOpAttribute->SetFromType(inputType);
            }
        }
        if (opcode == normalizedOpcode) {
            continue;
        }
        std::string opcodeBeforeNormalization = operation.GetOpcodeStr();
        operation.SetOpCode(normalizedOpcode);
        APASS_LOG_DEBUG_F(Elements::Operation, "Normalize %s[%d] to %s because input is %s and output is %s.",
                          opcodeBeforeNormalization.c_str(), operation.GetOpMagic(), operation.GetOpcodeStr().c_str(),
                          BriefMemoryTypeToString(inputType).c_str(), BriefMemoryTypeToString(outputType).c_str());
    }
    return SUCCESS;
}

Status AssignMemoryType::MarkSimtOps(Function& function)
{
    const auto arch = Platform::Instance().GetSoc().GetNPUArch();
    if (arch != NPUArch::DAV_3510 && arch != NPUArch::DAV_9201) {
        return SUCCESS;
    }
    for (auto& op : function.Operations()) {
        if (IsGmGatherElement(op) || IsGmScatter(op)) {
            op.SetAttribute(OP_ATTR_PREFIX + "requires_simt", true);
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::InsertConvertOpsAndInferShape(Function& function)
{
    std::vector<Operation*> addedOps;
    RETURN_IF_NOT_SUCCESS(inserter.DoInsertion(function, &addedOps));
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    if (!addedOps.empty()) {
        if (InferShapeUtils::InferShape(function, addedOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape for added ops failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::PreCheck(Function& function) { return checker.DoPreCheck(function); }

Status AssignMemoryType::PostCheck(Function& function) { return checker.DoPostCheck(function); }

Status AssignMemoryType::CalcLineOffset(const Shape& shape, const Offset& offset, int64_t& lineOffset) const
{
    if (shape.size() != offset.size() || shape.empty()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "CalcLineOffset failed because shape size %zu != offset size %zu or shape is empty.",
                          shape.size(), offset.size());
        return FAILED;
    }
    lineOffset = 0;
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        lineOffset += offset[i - 1] * stride;
        stride *= shape[i - 1];
    }
    return SUCCESS;
}

// 修正 VIEW 或 ASSEMBLE 与搬运方向不一致的情况，避免生成方向相反的拷贝。
Status AssignMemoryType::FixViewAssembleSemanticMismatch(Function& function)
{
    for (auto& op : function.Operations(false)) {
        if (op.iOperand.empty() || op.oOperand.empty()) {
            continue;
        }
        auto input = op.iOperand.front();
        auto output = op.oOperand.front();
        if (input == nullptr || output == nullptr) {
            continue;
        }
        MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, op);
        MemoryType outputOriginal = output->GetMemoryTypeOriginal();
        if (inputRequirement == MemoryType::MEM_UNKNOWN || outputOriginal == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            // VIEW 表达搬入语义；当 output==DDR 且 input requirement!=DDR 时语义违反，
            // 需要在 VIEW 之前插入 ASSEMBLE(local→DDR)。
            // 给 input 增加 DDR requirement（指向 VIEW 自身），制造 local vs DDR 冲突。
            if (outputOriginal == MemoryType::MEM_DEVICE_DDR && inputRequirement != MemoryType::MEM_DEVICE_DDR) {
                inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR, "FixViewSemanticMismatch");
                APASS_LOG_INFO_F(
                    Elements::Operation,
                    "VIEW[%d] output=DDR input requirement=%s, update input[%d] requirement to DDR to trigger "
                    "ASSEMBLE insertion.",
                    op.GetOpMagic(), BriefMemoryTypeToString(inputRequirement).c_str(), input->GetMagic());
            }
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            // ASSEMBLE 表达搬出语义；当 input requirement==DDR 且 output!=DDR 时语义违反，
            // 需要在 ASSEMBLE 之前插入 VIEW(DDR→local)。
            // 给 input 增加 output memType requirement（指向 ASSEMBLE 自身），制造 DDR vs local 冲突。
            if (inputRequirement == MemoryType::MEM_DEVICE_DDR && outputOriginal != MemoryType::MEM_DEVICE_DDR) {
                inserter.UpdateTensorTobeMap(input, op, outputOriginal, "FixAssembleSemanticMismatch");
                RETURN_IF_NOT_SUCCESS(SyncAssembleMemoryAttr(op));
                APASS_LOG_INFO_F(
                    Elements::Operation,
                    "ASSEMBLE[%d] input requirement=DDR output=%s, update input[%d] requirement to %s to trigger "
                    "VIEW insertion.",
                    op.GetOpMagic(), BriefMemoryTypeToString(outputOriginal).c_str(), input->GetMagic(),
                    BriefMemoryTypeToString(outputOriginal).c_str());
            }
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
