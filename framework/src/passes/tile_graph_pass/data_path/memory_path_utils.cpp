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
 * \file memory_path_utils.cpp
 * \brief
 */

#include "memory_path_utils.h"

#include <algorithm>
#include <set>

#include "interface/operation/attribute.h"
#include "interface/utils/common.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

bool MemoryPathUtils::IsMxPaddingMode(const Operation& operation)
{
    int64_t paddingMode = static_cast<int64_t>(Matrix::PaddingMode::NORMAL_PADDING_MODE);
    if (!operation.GetAttr<int64_t>("op_attr_copy_in_l1_padding_mode", paddingMode) ||
        paddingMode != static_cast<int64_t>(Matrix::PaddingMode::MX_PADDING_MODE)) {
        return false;
    }
    // K维索引由copy_in_l1_k_index指示（transA时K在shape第0维，transB时K在第1维），
    // 缺失或非法时无法判定K轴，保守回退DDR
    int64_t kIndex = -1;
    if (!operation.GetAttr<int64_t>("op_attr_copy_in_l1_k_index", kIndex) || kIndex < 0) {
        return true;
    }
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        return true;
    }
    // K轴取拷入源tensor（view输入）的dynValidShape（validK）：validK非concrete（如viewK动态K）
    // 时无法确定MX K向补齐量，直接回退DDR
    const auto& dynValidShape = operation.iOperand.front()->GetDynValidShape();
    if (dynValidShape.size() <= static_cast<size_t>(kIndex) || !dynValidShape[kIndex].ConcreteValid()) {
        return true;
    }
    // validK为concrete且按64（MX scale块大小）对齐时无需K向补齐，允许UB->L1直连
    return dynValidShape[kIndex].Concrete() % ALIGN_SIZE_64 != 0;
}

bool MemoryPathUtils::HasMxPaddingModeConsumer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto& consumers = tensor->GetConsumers();
    return std::any_of(consumers.begin(), consumers.end(),
                       [](const Operation* consumer) { return consumer != nullptr && IsMxPaddingMode(*consumer); });
}

bool MemoryPathUtils::IsSpecialDirectMemoryPath(MemoryType from, MemoryType to)
{
    return (from == MemoryType::MEM_L0C && to == MemoryType::MEM_L1) ||
           (from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) ||
           (from == MemoryType::MEM_UB && to == MemoryType::MEM_L1);
}

bool MemoryPathUtils::IsDifferentKnownRequirement(MemoryType requirement, MemoryType targetType)
{
    return requirement != MemoryType::MEM_UNKNOWN && requirement != targetType;
}

bool MemoryPathUtils::ShouldUseDdrForSpecialPath(bool hasParallelDifferentRequirement, MemoryType from, MemoryType to)
{
    return hasParallelDifferentRequirement && IsSpecialDirectMemoryPath(from, to);
}

MemoryType MemoryPathUtils::ResolveEffectiveConsumerRequirement(
    Operation* consumerOp, MemoryType directRequirement, MemoryType targetType,
    const OutputRequirementResolver& resolveOutputRequirement)
{
    if (consumerOp == nullptr) {
        return directRequirement;
    }
    auto opcode = consumerOp->GetOpcode();
    if (opcode != Opcode::OP_VIEW && opcode != Opcode::OP_SLICE && opcode != Opcode::OP_ASSEMBLE &&
        opcode != Opcode::OP_CONTRACT) {
        return directRequirement;
    }
    std::set<MemoryType> branchRequirements;
    auto addRequirement = [&branchRequirements](MemoryType requirement) {
        if (requirement != MemoryType::MEM_UNKNOWN) {
            branchRequirements.insert(requirement);
        }
    };
    if (opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE) {
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewOpAttribute != nullptr) {
            addRequirement(viewOpAttribute->GetTo());
        }
    }
    if (opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT) {
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(consumerOp->GetOpAttribute());
        if (assembleOpAttribute != nullptr) {
            MemoryType fromType = assembleOpAttribute->GetFrom();
            if (IsSpecialDirectMemoryPath(fromType, targetType) && !consumerOp->oOperand.empty() &&
                consumerOp->oOperand.front() != nullptr && !consumerOp->oOperand.front()->GetConsumers().empty()) {
                addRequirement(targetType);
            }
        }
    }
    if (!consumerOp->oOperand.empty() && consumerOp->oOperand.front() != nullptr) {
        auto output = consumerOp->oOperand.front();
        addRequirement(output->GetMemoryTypeOriginal());
        if (resolveOutputRequirement) {
            addRequirement(resolveOutputRequirement(output));
        }
    }
    if (branchRequirements.count(targetType) > 0) {
        return targetType;
    }
    if (branchRequirements.size() == 1) {
        return *branchRequirements.begin();
    }
    return directRequirement;
}

// ==================== 以下为 AssignMemoryType（含 legacy）两版本完全一致的公共实现 ====================

Status MemoryPathUtils::GetFirstInputOutputIfOpcode(Operation& operation, Opcode expectedOpcode,
                                                    const std::string& action, LogicalTensorPtr& input,
                                                    LogicalTensorPtr& output, bool& shouldHandle)
{
    shouldHandle = operation.GetOpcode() == expectedOpcode;
    if (!shouldHandle)
        return SUCCESS;
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s for %s[%d] failed because operand is empty.", action.c_str(),
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    input = operation.iOperand.front();
    output = operation.oOperand.front();
    if (input == nullptr || output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s for %s[%d] failed because operand tensor is null.", action.c_str(),
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

// 特殊进阶数据通路，不满足特定条件时回退到通过DDR搬运：L0C2L1, L0C2UB, UB2L1
bool MemoryPathUtils::IsAdvancedMemoryPath(MemoryType from, MemoryType to)
{
    if (from == MemoryType::MEM_L0C && to == MemoryType::MEM_L1) {
        return true;
    }
    bool isA5 = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    return isA5 && ((from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) ||
                    (from == MemoryType::MEM_UB && to == MemoryType::MEM_L1));
}

bool MemoryPathUtils::IsAssembleProducer(Operation* operation)
{
    return operation != nullptr && operation->GetOpcode() == Opcode::OP_ASSEMBLE && !operation->iOperand.empty();
}

void MemoryPathUtils::CollectProducerAIVFlags(Operation* op, std::vector<bool>& isProducerVector)
{
    for (auto& opInput : op->iOperand) {
        for (auto& producer : opInput->GetProducers()) {
            isProducerVector.push_back(producer->GetCoreType() == CoreType::AIV);
        }
    }
}

void MemoryPathUtils::CollectConsumerAICFlags(Operation* op, std::vector<bool>& isConsumerCube)
{
    for (auto& opOutput : op->oOperand) {
        for (auto& consumer : opOutput->GetConsumers()) {
            isConsumerCube.push_back(consumer->GetCoreType() == CoreType::AIC);
        }
    }
}

bool MemoryPathUtils::FitsAssembleOutputMemoryLimit(const LogicalTensorPtr& output, MemoryType memoryType)
{
    if (output == nullptr) {
        return false;
    }
    if (memoryType == MemoryType::MEM_UB) {
        const size_t ubThreshold = static_cast<size_t>(
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_ASSEMBLE);
        return static_cast<size_t>(output->GetDataSize()) <= ubThreshold;
    }
    if (memoryType == MemoryType::MEM_L1) {
        const size_t l1Threshold = static_cast<size_t>(
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) * L1_THRESHOLD);
        return static_cast<size_t>(output->GetDataSize()) <= l1Threshold;
    }
    return true;
}

bool MemoryPathUtils::FitsTensorInUb(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_NORMAL);
    int64_t dataSize = tensor->GetDataSize();
    return dataSize >= 0 && static_cast<size_t>(dataSize) <= ubThreshold;
}

bool MemoryPathUtils::ExceedsMemoryLimit(const LogicalTensorPtr& tensor, size_t threshold)
{
    if (tensor == nullptr) {
        return false;
    }
    int64_t dataSize = tensor->GetDataSize();
    if (dataSize < 0) {
        return false;
    }
    return static_cast<size_t>(dataSize) > threshold;
}

bool MemoryPathUtils::IsOversizedLocalBuffer(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                             bool useStrictUbLimit, bool allowL1Fallback)
{
    if (memoryType == MemoryType::MEM_UB) {
        double ubLimitRatio = useStrictUbLimit ? UB_THRESHOLD_ASSEMBLE : UB_THRESHOLD_NORMAL;
        size_t threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                               ubLimitRatio);
        return ExceedsMemoryLimit(tensor, threshold);
    }
    if (memoryType == MemoryType::MEM_L1 && allowL1Fallback) {
        size_t threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) *
                                               L1_THRESHOLD);
        return ExceedsMemoryLimit(tensor, threshold);
    }
    return false;
}

bool MemoryPathUtils::IsDynamicReshape(Operation& operation, const LogicalTensorPtr& output)
{
    static const std::string validShapeAttr = "op_attr_validShape";
    if (operation.HasAttr(validShapeAttr)) {
        return true;
    }
    if (output == nullptr) {
        return false;
    }
    for (const auto& dim : output->GetDynValidShape()) {
        if (!dim.IsImmediate()) {
            return true;
        }
    }
    return false;
}

bool MemoryPathUtils::CheckUBTileShape(const LogicalTensorPtr& moveTensor)
{
    if (moveTensor == nullptr || moveTensor->GetShape().size() < 2) {
        return false;
    }
    const int64_t alignElems = (moveTensor->Datatype() == DataType::DT_INT8) ? INT8_ALIGN_SIZE : L0C_TILE_SIZE;
    if (moveTensor->GetShape()[0] % alignElems == 0 && moveTensor->GetShape()[1] % alignElems == 0) {
        return true;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor,
                      "Set tensor %d original memory type to DDR since tile shape of moved block "
                      "is not 16-element aligned (int8 requires 32-element).",
                      moveTensor->magic);
    return false;
}

bool MemoryPathUtils::CheckUBConvTileShape(const LogicalTensorPtr& moveTensor)
{
    // 仅约束 conv 多维 tile 的末维 (W)；二维 matmul tile 的分形校验由 CheckUBTileShape 负责，分开判断
    if (moveTensor == nullptr || moveTensor->GetShape().size() <= SHAPE_DIM2) {
        return false;
    }
    const int64_t alignElems = (moveTensor->Datatype() == DataType::DT_INT8) ? INT8_ALIGN_SIZE : L0C_TILE_SIZE;
    if (moveTensor->GetShape().back() % alignElems == 0 && moveTensor->GetShape()[1] % alignElems == 0) {
        return true;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor,
                      "Set tensor %d original memory type to DDR since last dim of moved conv tile "
                      "is not 16-element aligned (int8 requires 32-element).",
                      moveTensor->magic);
    return false;
}

bool MemoryPathUtils::IsDimMultiple(const Shape& shape1, const Shape& shape2)
{
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] <= 0 || shape2[i] <= 0 || shape1[i] % shape2[i] != 0) {
            return false;
        }
    }
    return true;
}

bool MemoryPathUtils::CheckInnerAxisC0Size(const LogicalTensorPtr& input, const LogicalTensorPtr& output)
{
    constexpr int64_t kC0AlignBytes = 32;
    size_t inputInnerAxis = input->GetShape().back();
    size_t outputInnerAxis = output->GetShape().back();
    // 如果输入内轴大小等于输出内轴大小，说明内轴未被切分
    // 这种情况不需要检查对齐，直接返回 true
    if (inputInnerAxis == outputInnerAxis) {
        return true;
    }
    int64_t inputDtypeBytes = BytesOf(input->Datatype());
    int64_t outputDtypeBytes = BytesOf(output->Datatype());
    // 检查数据类型字节数是否有效（避免除零）
    int64_t inputC0Size = (inputDtypeBytes > 0) ? (kC0AlignBytes / inputDtypeBytes) : 0;
    int64_t outputC0Size = (outputDtypeBytes > 0) ? (kC0AlignBytes / outputDtypeBytes) : 0;
    if (inputC0Size <= 0 || outputC0Size <= 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "CheckInnerAxisC0Size: invalid C0 size, inputC0Size=%ld, outputC0Size=%ld",
                          static_cast<long>(inputC0Size), static_cast<long>(outputC0Size));
        return false;
    }
    // 分别检查 input 和 output 的内轴是否满足各自的 C0 size 切分
    if (inputInnerAxis % static_cast<size_t>(inputC0Size) != 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "CheckInnerAxisC0Size: input inner=%zu, dtypeBytes=%ld, c0Size=%ld, not aligned",
                          inputInnerAxis, static_cast<long>(inputDtypeBytes), static_cast<long>(inputC0Size));
        return false;
    }
    if (outputInnerAxis % static_cast<size_t>(outputC0Size) != 0) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "CheckInnerAxisC0Size: output inner=%zu, dtypeBytes=%ld, c0Size=%ld, not aligned",
                          outputInnerAxis, static_cast<long>(outputDtypeBytes), static_cast<long>(outputC0Size));
        return false;
    }
    return true;
}

Status MemoryPathUtils::SyncTensorToBe(Function& function)
{
    size_t syncCount = 0;
    std::unordered_set<LogicalTensorPtr> visited;
    auto syncTensor = [&syncCount, &visited](const LogicalTensorPtr& tensor) {
        if (tensor == nullptr) {
            return;
        }
        if (!visited.insert(tensor).second) {
            return;
        }
        tensor->SetMemoryTypeToBe(tensor->GetMemoryTypeOriginal());
        ++syncCount;
    };
    // Function boundary tensors can remain after all of their operand references are removed.
    for (const auto& tensor : function.inCasts_) {
        syncTensor(tensor);
    }
    for (const auto& tensor : function.outCasts_) {
        syncTensor(tensor);
    }
    for (auto& op : function.Operations()) {
        for (auto& input : op.iOperand) {
            syncTensor(input);
        }
        for (auto& output : op.oOperand) {
            syncTensor(output);
        }
    }
    return SUCCESS;
}

Status MemoryPathUtils::SetOriginalChecked(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                           const std::string& reason, bool allowOverride)
{
    std::string context = reason.empty() ? "unknown" : reason;
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "SetOriginalChecked failed because tensor is null, reason: %s.",
                          context.c_str());
        return FAILED;
    }
    if (memoryType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryType currentType = tensor->GetMemoryTypeOriginal();
    if (currentType != MemoryType::MEM_UNKNOWN && currentType != memoryType && !allowOverride) {
        APASS_LOG_WARN_F(
            Elements::Tensor,
            "Skip tensor %d original memory type update because current %s conflicts with new %s, reason: %s.",
            tensor->GetMagic(), BriefMemoryTypeToString(currentType).c_str(),
            BriefMemoryTypeToString(memoryType).c_str(), context.c_str());
        return SUCCESS;
    }
    if (currentType != MemoryType::MEM_UNKNOWN && currentType != memoryType) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Override tensor[%d] original %s -> %s, reason: %s.", tensor->GetMagic(),
                          BriefMemoryTypeToString(currentType).c_str(), BriefMemoryTypeToString(memoryType).c_str(),
                          context.c_str());
    } else if (currentType == MemoryType::MEM_UNKNOWN && memoryType != currentType &&
               context.compare(0, 5, "Infer") == 0) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Infer tensor[%d] original as %s, reason: %s.", tensor->GetMagic(),
                          BriefMemoryTypeToString(memoryType).c_str(), context.c_str());
    }
    tensor->SetMemoryTypeOriginal(memoryType, allowOverride);
    return SUCCESS;
}

void MemoryPathUtils::ForceSetOriginal(const LogicalTensorPtr& tensor, MemoryType memoryType, const std::string& reason)
{
    if (tensor != nullptr && memoryType != MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Force tensor[%d] original as %s, reason %s.", tensor->GetMagic(),
                          BriefMemoryTypeToString(memoryType).c_str(), reason.c_str());
    }
    SetOriginalChecked(tensor, memoryType, reason, true);
}

bool MemoryPathUtils::CanUseUbForReshape(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                         MemoryType inputRequirement, MemoryType outputOriginal)
{
    if (inputRequirement != outputOriginal) {
        return false;
    }
    return inputRequirement == MemoryType::MEM_UB && FitsTensorInUb(input) && FitsTensorInUb(output);
}

Status MemoryPathUtils::AssignInOutCastMemoryTypes(Function& function)
{
    for (auto& incast : function.inCasts_) {
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(incast, MemoryType::MEM_DEVICE_DDR, "AssignIncastMemoryType", true));
    }

    for (auto& outcast : function.outCasts_) {
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(outcast, MemoryType::MEM_DEVICE_DDR, "AssignOutcastMemoryType", true));
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
