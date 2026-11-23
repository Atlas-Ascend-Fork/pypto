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
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
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

// 只有可直连且仍有下游消费者的拼接输出才贡献目标存储类型。
static bool HasDirectAssembleOutputRequirement(const Operation& consumerOp, MemoryType targetType)
{
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(consumerOp.GetOpAttribute());
    if (assembleOpAttribute != nullptr) {
        MemoryType fromType = assembleOpAttribute->GetFrom();
        if (MemoryPathUtils::IsSpecialDirectMemoryPath(fromType, targetType) && !consumerOp.oOperand.empty() &&
            consumerOp.oOperand.front() != nullptr && !consumerOp.oOperand.front()->GetConsumers().empty()) {
            return true;
        }
    }
    return false;
}

// 综合搬运属性、输出驻留类型和下游需求，优先保留目标类型，其次采用唯一已知类型。
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
    if ((opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT) &&
        HasDirectAssembleOutputRequirement(*consumerOp, targetType)) {
        addRequirement(targetType);
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

bool MemoryPathUtils::HasConvProducer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    return std::any_of(tensor->GetProducers().begin(), tensor->GetProducers().end(), [](const Operation* producer) {
        if (producer == nullptr) {
            return false;
        }
        if (producer->GetBoolAttribute(OpAttributeKey::isConv)) {
            return true;
        }
        if (producer->GetOpcode() != Opcode::OP_CONTRACT) {
            return false;
        }
        return std::any_of(producer->GetIOperands().begin(), producer->GetIOperands().end(), HasConvProducer);
    });
}

bool MemoryPathUtils::HasConvConsumer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    return std::any_of(tensor->GetConsumers().begin(), tensor->GetConsumers().end(), [](const Operation* consumer) {
        if (consumer == nullptr) {
            return false;
        }
        if (consumer->GetBoolAttribute(OpAttributeKey::isConv)) {
            return true;
        }
        if (consumer->GetOpcode() != Opcode::OP_SLICE) {
            return false;
        }
        return std::any_of(consumer->GetOOperands().begin(), consumer->GetOOperands().end(), HasConvConsumer);
    });
}

// batch matmul 多维 tile 要求 batch 轴均为 1，矩阵语义只读尾两维 (M, N)
bool MemoryPathUtils::AllBatchAxesAreOne(const Shape& shape)
{
    if (shape.size() <= SHAPE_DIM2) {
        return false;
    }
    return std::all_of(shape.begin(), shape.end() - SHAPE_DIM2, [](int64_t dim) { return dim == 1; });
}

// L0C2UB大于2维场景仅支持卷积
bool MemoryPathUtils::IsL0C2UbSupportedShape(const LogicalTensorPtr& tensor)
{
    return tensor != nullptr && (tensor->GetShape().size() == SHAPE_DIM2 || HasConvProducer(tensor));
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

bool MemoryPathUtils::CheckUBBatchMatmulTileShapeLiteNPU(const LogicalTensorPtr& moveTensor)
{
    // LiteNPU batch matmul 多维 tile（batch 轴全 1）矩阵语义只读尾两维 (M, N)。LiteNPU 的
    // L1→UB 与 L0C→UB 直连一致：均为目的端 UB 保持 ND 布局的 fixpipe 写（kirin9050 L1→UB 见
    // TCopyL12UBImpl：L1 源为 mmad 直出的 NZ Mat，UB 目的为 ND），与 TStore NZ2ND 的 DDR 回退路径
    // 同语义——fixpipe ND 写仅约束末维 (N) 对齐，M 为有效行数不做 16 对齐约束。
    // LiteNPU 当前不存在 conv 场景（conv 产品矩阵：Kirin9030/KirinX90/Kirin9050 均不支持），
    // rank>2 且前导维退化全 1 的 conv tile 不可达，不会误入本分支（若后续放开需回到 conv 双维校验）；
    // 仅 LiteNPU 直通场景可达（CheckUBTileShapeByRank 分派器带 IsLiteNPU 门控）
    if (moveTensor == nullptr || moveTensor->GetShape().empty()) {
        return false;
    }
    const int64_t alignElems = (moveTensor->Datatype() == DataType::DT_INT8) ? INT8_ALIGN_SIZE : L0C_TILE_SIZE;
    if (moveTensor->GetShape().back() % alignElems == 0) {
        return true;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor,
                      "Set tensor %d original memory type to DDR since last dim of moved batch matmul tile "
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

// 将各张量的 MemoryTypeToBe 同步为已确定的 MemoryTypeOriginal（实际驻留类型），覆盖算子引用张量与图边界张量。
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
    // 图边界的张量在失去所有算子引用后仍可能存在。
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

// 校验单个消费者的存储需求，检测冲突并按允许的覆盖策略更新。
Status MemoryPathUtils::SetRequirementChecked(ConvertInserter& inserter, const LogicalTensorPtr& tensor,
                                              Operation& operation, MemoryType memoryType, const std::string& reason,
                                              bool allowOverride)
{
    std::string context = reason.empty() ? "unknown" : reason;
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "SetRequirementChecked failed because tensor is null for operation %s[%d], reason: %s.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), context.c_str());
        return FAILED;
    }
    if (!tensor->HasConsumer(operation)) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Operation %s[%d] is not a consumer of tensor %d, reason: %s.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), tensor->GetMagic(),
                          context.c_str());
        return FAILED;
    }
    bool hasRequirement = inserter.HasRequirement(tensor, operation);
    MemoryType currentType = inserter.GetRequirementOrUnknown(tensor, operation);
    if (hasRequirement && currentType != MemoryType::MEM_UNKNOWN && memoryType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    if (currentType != MemoryType::MEM_UNKNOWN && memoryType != MemoryType::MEM_UNKNOWN && currentType != memoryType &&
        !allowOverride) {
        APASS_LOG_WARN_F(
            Elements::Tensor,
            "Skip tensor %d requirement update for operation %s[%d] because current %s conflicts with new %s, "
            "reason: %s.",
            tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
            BriefMemoryTypeToString(currentType).c_str(), BriefMemoryTypeToString(memoryType).c_str(), context.c_str());
        return SUCCESS;
    }
    inserter.UpdateTensorTobeMap(tensor, operation, memoryType, context.c_str());
    return SUCCESS;
}

void MemoryPathUtils::ForceSetRequirement(ConvertInserter& inserter, const LogicalTensorPtr& tensor,
                                          Operation& operation, MemoryType memoryType, const std::string& reason)
{
    if (tensor != nullptr && memoryType != MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Force tensor[%d] requirement for %s[%d] as %s, reason %s.",
                          tensor->GetMagic(), operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                          BriefMemoryTypeToString(memoryType).c_str(), reason.c_str());
    }
    SetRequirementChecked(inserter, tensor, operation, memoryType, reason, true);
}

void MemoryPathUtils::FillUnknownRequirementsWith(ConvertInserter& inserter, const LogicalTensorPtr& tensor,
                                                  MemoryType memoryType, const char* reason)
{
    if (tensor == nullptr || memoryType == MemoryType::MEM_UNKNOWN) {
        return;
    }
    auto requirements = inserter.GetConsumerRequirements(tensor);
    for (const auto& item : requirements) {
        if (item.second == MemoryType::MEM_UNKNOWN) {
            inserter.UpdateTensorTobeMap(tensor, *item.first, memoryType, reason);
        }
    }
}

Status MemoryPathUtils::EnsureAllConsumerRequirementsExist(ConvertInserter& inserter, Function& function)
{
    std::unordered_set<LogicalTensorPtr> visited;
    auto ensureTensor = [&inserter, &visited](const LogicalTensorPtr& tensor) -> Status {
        if (tensor == nullptr || !visited.insert(tensor).second)
            return SUCCESS;
        for (const auto& consumerOp : tensor->GetConsumers()) {
            if (inserter.HasRequirement(tensor, *consumerOp))
                continue;
            RETURN_IF_NOT_SUCCESS(SetRequirementChecked(inserter, tensor, *consumerOp, MemoryType::MEM_UNKNOWN,
                                                        "EnsureAllConsumerRequirementsExist"));
        }
        return SUCCESS;
    };
    for (auto& op : function.Operations()) {
        for (auto& input : op.iOperand) {
            RETURN_IF_NOT_SUCCESS(ensureTensor(input));
        }
        for (auto& output : op.oOperand) {
            RETURN_IF_NOT_SUCCESS(ensureTensor(output));
        }
    }
    return SUCCESS;
}

Status MemoryPathUtils::HandleNopMemoryType(ConvertInserter& inserter, Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(GetFirstInputOutputIfOpcode(operation, Opcode::OP_NOP, "Handle OP_NOP memory type", input,
                                                      output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (inputRequirement == MemoryType::MEM_UNKNOWN || outputOriginal == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    if (inputRequirement != outputOriginal) {
        ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, "HandleNopMismatchFallbackDdr");
        ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "HandleNopMismatchFallbackDdr");
        return SUCCESS;
    }
    return SUCCESS;
}

// 识别由入参直接切片且所有消费者都显式声明未知需求的情况，原地覆盖时保留在 DDR。
static bool IsScatterSliceFromIncast(const ConvertInserter& inserter, const Function& function,
                                     const LogicalTensorPtr& self)
{
    if (self->GetConsumers().empty()) {
        return false;
    }
    for (auto* consumer : self->GetConsumers()) {
        if (consumer == nullptr || inserter.GetRequirementOrUnknown(self, *consumer) != MemoryType::MEM_UNKNOWN) {
            return false;
        }
        auto it = std::find(consumer->iOperand.begin(), consumer->iOperand.end(), self);
        if (it == consumer->iOperand.end()) {
            return false;
        }
        size_t inputIndex = static_cast<size_t>(std::distance(consumer->iOperand.begin(), it));
        const auto& definedTypes = OpcodeManager::Inst().GetInputsMemType(consumer->GetOpcode());
        if (inputIndex >= definedTypes.size() || definedTypes[inputIndex] != MemoryType::MEM_UNKNOWN) {
            return false;
        }
    }
    Operation* slice = nullptr;
    for (auto* producer : self->GetProducers()) {
        if (producer == nullptr || producer->GetOpcode() != Opcode::OP_SLICE || producer->oOperand.empty() ||
            producer->oOperand.front() != self || producer->iOperand.empty() || producer->iOperand.front() == nullptr ||
            slice != nullptr) {
            return false;
        }
        slice = producer;
    }
    return slice != nullptr &&
           std::find(function.inCasts_.begin(), function.inCasts_.end(), slice->iOperand.front()) !=
               function.inCasts_.end() &&
           std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute()) != nullptr;
}

// 沿 VIEW、ASSEMBLE 和 RESHAPE 向上追溯，仅在候选驻留类型唯一时采用该类型。
static MemoryType InferScatterInputMemoryType(const LogicalTensorPtr& tensor,
                                              std::unordered_set<const LogicalTensor*>& visited)
{
    if (tensor == nullptr || !visited.insert(tensor.get()).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        candidates.insert(tensor->GetMemoryTypeOriginal());
    }
    for (auto* producer : tensor->GetProducers()) {
        if (producer == nullptr ||
            (producer->GetOpcode() != Opcode::OP_VIEW && producer->GetOpcode() != Opcode::OP_ASSEMBLE &&
             producer->GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        for (const auto& input : producer->iOperand) {
            auto candidate = InferScatterInputMemoryType(input, visited);
            if (candidate != MemoryType::MEM_UNKNOWN) {
                candidates.insert(candidate);
            }
        }
    }
    return candidates.size() == 1 ? *candidates.begin() : MemoryType::MEM_UNKNOWN;
}

// 沿布局算子向下收集消费者需求，递归时复用已访问集合避免重复访问。
static MemoryType InferScatterOutputMemoryType(const ConvertInserter& inserter, const LogicalTensorPtr& tensor,
                                               std::unordered_set<const LogicalTensor*>& visited)
{
    if (tensor == nullptr || !visited.insert(tensor.get()).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    for (const auto& [consumer, requirement] : inserter.GetConsumerRequirements(tensor)) {
        if (requirement != MemoryType::MEM_UNKNOWN) {
            candidates.insert(requirement);
        } else if (consumer != nullptr &&
                   (consumer->GetOpcode() == Opcode::OP_VIEW || consumer->GetOpcode() == Opcode::OP_ASSEMBLE ||
                    consumer->GetOpcode() == Opcode::OP_RESHAPE)) {
            for (const auto& next : consumer->oOperand) {
                auto candidate = InferScatterOutputMemoryType(inserter, next, visited);
                if (candidate != MemoryType::MEM_UNKNOWN) {
                    candidates.insert(candidate);
                }
            }
        }
    }
    return candidates.size() == 1 ? *candidates.begin() : MemoryType::MEM_UNKNOWN;
}

// 为原地 Scatter 统一输入需求与输出驻留类型，UB 场景改用普通 Scatter。
Status MemoryPathUtils::HandleScatterInplaceMemoryType(ConvertInserter& inserter, Function& function,
                                                       Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_SCATTER_INPLACE &&
        operation.GetOpcode() != Opcode::OP_SCATTER_ELEMENT_INPLACE) {
        return SUCCESS;
    }
    if (!operation.HasAttr(OP_ATTR_PREFIX + "scatter_mode") ||
        operation.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode") != 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "In-place Scatter only supports overwrite.");
        return FAILED;
    }
    if (operation.iOperand.empty() || operation.oOperand.empty() || operation.iOperand[0] == nullptr ||
        operation.oOperand[0] == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Handle SCATTER_INPLACE memory type failed because operand is invalid.");
        return FAILED;
    }

    auto self = operation.iOperand[0];
    auto output = operation.oOperand[0];
    MemoryType memoryType = self->GetMemoryTypeOriginal();
    if (inserter.GetMode() == AssignMemoryTypeMode::SLICE_CONTRACT) {
        if (IsScatterSliceFromIncast(inserter, function, self)) {
            memoryType = MemoryType::MEM_DEVICE_DDR;
            ForceSetOriginal(self, memoryType, "ScatterInplaceSliceFromIncast");
        } else if (memoryType == MemoryType::MEM_UNKNOWN) {
            std::unordered_set<const LogicalTensor*> visited;
            memoryType = InferScatterInputMemoryType(self, visited);
        }
        if (memoryType == MemoryType::MEM_UNKNOWN) {
            std::unordered_set<const LogicalTensor*> visited;
            memoryType = InferScatterOutputMemoryType(inserter, output, visited);
        }
    } else if (memoryType == MemoryType::MEM_UNKNOWN) {
        memoryType = output->GetMemoryTypeOriginal();
    }
    if (memoryType == MemoryType::MEM_UNKNOWN) {
        memoryType = MemoryType::MEM_UB;
    }
    ForceSetRequirement(inserter, self, operation, memoryType, "ScatterInplaceSameMemory");
    ForceSetOriginal(output, memoryType, "ScatterInplaceSameMemory");
    if (memoryType == MemoryType::MEM_UB) {
        operation.SetOpCode(operation.GetOpcode() == Opcode::OP_SCATTER_INPLACE ? Opcode::OP_SCATTER :
                                                                                  Opcode::OP_SCATTER_ELEMENT);
    }
    return SUCCESS;
}

Status MemoryPathUtils::ApplyOtherSpecialOpcodeRules(ConvertInserter& inserter, Function& function)
{
    for (auto& op : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(HandleScatterInplaceMemoryType(inserter, function, op));
        RETURN_IF_NOT_SUCCESS(HandleNopMemoryType(inserter, op));
    }
    return SUCCESS;
}

Status MemoryPathUtils::AssignReduceAccInputRequirements(ConvertInserter& inserter, Operation& operation)
{
    for (auto& tensor : operation.iOperand) {
        RETURN_IF_NOT_SUCCESS(SetRequirementChecked(inserter, tensor, operation, MemoryType::MEM_DEVICE_DDR,
                                                    "AssignReduceAccInputRequirements"));
    }
    return SUCCESS;
}

Status MemoryPathUtils::AssignOpcodeDefinedMemoryTypes(ConvertInserter& inserter, Operation& operation)
{
    auto opcode = operation.GetOpcode();
    bool hasSpecialInputRule = opcode == Opcode::OP_REDUCE_ACC ||
                               OpChecker::check(operation, OpChecker::CalcTypeChecker(OpCalcType::MATMUL));
    const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
    if (!hasSpecialInputRule) {
        for (size_t i = 0; i < operation.iOperand.size(); ++i) {
            MemoryType inputMemType = (i < inputsMemType.size()) ? inputsMemType[i] : MemoryType::MEM_UNKNOWN;
            RETURN_IF_NOT_SUCCESS(SetRequirementChecked(inserter, operation.iOperand[i], operation, inputMemType,
                                                        "AssignOpcodeDefinedInput"));
        }
    }
    const auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(opcode);
    for (size_t i = 0; i < operation.oOperand.size(); ++i) {
        if (i >= outputsMemType.size())
            continue;
        RETURN_IF_NOT_SUCCESS(
            SetOriginalChecked(operation.oOperand[i], outputsMemType[i], "AssignOpcodeDefinedOutput"));
    }
    return SUCCESS;
}

MemoryType MemoryPathUtils::GetAssembleInputType(const ConvertInserter& inserter, Operation& operation)
{
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr)
        return MemoryType::MEM_UNKNOWN;
    auto input = operation.iOperand.front();
    MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
    return fromType != MemoryType::MEM_UNKNOWN ? fromType : input->GetMemoryTypeOriginal();
}

MemoryType MemoryPathUtils::GetReshapeInputRequirement(ConvertInserter& inserter, Operation& operation,
                                                       const LogicalTensorPtr& input, MemoryType inputOriginal)
{
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    if (inputRequirement != MemoryType::MEM_UNKNOWN || inputOriginal == MemoryType::MEM_UNKNOWN) {
        return inputRequirement;
    }
    ForceSetRequirement(inserter, input, operation, inputOriginal, "InferReshapeInputOriginal");
    return inputOriginal;
}

// 经 VIEW 推断 reshape 的输出驻留，需求不唯一时优先保留可达的 UB 需求。
Status MemoryPathUtils::InferReshapeOutputFromRequirement(ConvertInserter& inserter, const LogicalTensorPtr& output,
                                                          MemoryType& outputOriginal)
{
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryType outputRequirement = InferUniqueRequirementThroughViewConsumers(inserter, output);
    if (outputRequirement == MemoryType::MEM_UNKNOWN) {
        std::unordered_set<const LogicalTensor*> visitedTensors;
        if (HasRequirementThroughViewConsumers(inserter, output, MemoryType::MEM_UB, visitedTensors)) {
            outputRequirement = MemoryType::MEM_UB;
        }
    }
    if (outputRequirement == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(SetOriginalChecked(output, outputRequirement, "InferReshapeOutputRequirement"));
    outputOriginal = output->GetMemoryTypeOriginal();
    return SUCCESS;
}

MemoryType MemoryPathUtils::InferUniqueRequirementThroughViewConsumers(const ConvertInserter& inserter,
                                                                       const LogicalTensorPtr& tensor)
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    return InferUniqueRequirementThroughViewConsumers(inserter, tensor, visitedTensors);
}

MemoryType MemoryPathUtils::InferUniqueRequirementThroughViewConsumers(
    const ConvertInserter& inserter, const LogicalTensorPtr& tensor,
    std::unordered_set<const LogicalTensor*>& visitedTensors)
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
        addCandidate(item.second);
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewOpAttribute != nullptr) {
            addCandidate(viewOpAttribute->GetTo());
        }
        if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
            continue;
        }
        auto viewOutput = consumerOp->oOperand.front();
        addCandidate(viewOutput->GetMemoryTypeOriginal());
        addCandidate(InferUniqueRequirementThroughViewConsumers(inserter, viewOutput, visitedTensors));
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
}

bool MemoryPathUtils::HasRequirementThroughViewConsumers(const ConvertInserter& inserter,
                                                         const LogicalTensorPtr& tensor, MemoryType targetRequirement,
                                                         std::unordered_set<const LogicalTensor*>& visitedTensors)
{
    if (tensor == nullptr || targetRequirement == MemoryType::MEM_UNKNOWN ||
        !visitedTensors.insert(tensor.get()).second) {
        return false;
    }
    auto consumerRequirements = inserter.GetConsumerRequirements(tensor);
    for (const auto& item : consumerRequirements) {
        Operation* consumerOp = item.first;
        if (item.second == targetRequirement) {
            return true;
        }
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewOpAttribute != nullptr && viewOpAttribute->GetTo() == targetRequirement) {
            return true;
        }
        if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
            continue;
        }
        auto viewOutput = consumerOp->oOperand.front();
        if (viewOutput->GetMemoryTypeOriginal() == targetRequirement ||
            HasRequirementThroughViewConsumers(inserter, viewOutput, targetRequirement, visitedTensors)) {
            return true;
        }
    }
    return false;
}

Status MemoryPathUtils::ApplyReshapeMemoryType(ConvertInserter& inserter, Operation& operation,
                                               const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                               bool isDynamic, bool canUseUb)
{
    if (canUseUb) {
        const char* reason = isDynamic ? "InferDynamicReshapeUb" : "InferStaticReshapeUb";
        ForceSetRequirement(inserter, input, operation, MemoryType::MEM_UB, reason);
        ForceSetOriginal(output, MemoryType::MEM_UB, reason);
        return SUCCESS;
    }
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "Reshape[%d] falls back to DDR: dynamic=%d, input tensor[%d] original=%s, "
                      "requirement=%s, output tensor[%d] original=%s.",
                      operation.GetOpMagic(), isDynamic, input->GetMagic(),
                      BriefMemoryTypeToString(input->GetMemoryTypeOriginal()).c_str(),
                      BriefMemoryTypeToString(inserter.GetRequirementOrUnknown(input, operation)).c_str(),
                      output->GetMagic(), BriefMemoryTypeToString(output->GetMemoryTypeOriginal()).c_str());
    const char* reason = isDynamic ? "InferDynamicReshapeFallbackDdr" : "InferStaticReshapeFallbackDdr";
    ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, reason);
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, reason);
    return SUCCESS;
}

Status MemoryPathUtils::InferViewTypeInput(ConvertInserter& inserter, Operation& operation,
                                           const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                           MemoryType targetType)
{
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    if (targetType != MemoryType::MEM_UNKNOWN && inputOriginal == targetType) {
        ForceSetRequirement(inserter, input, operation, targetType, "InferViewTypeSameMemory");
        ForceSetOriginal(output, targetType, "InferViewTypeSameMemory");
        return SUCCESS;
    }
    ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, "InferViewTypeFallbackDdr");
    ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "InferViewTypeFallbackDdr");
    return SUCCESS;
}

MemoryType MemoryPathUtils::InferTargetTypeThroughForwardViews(const ConvertInserter& inserter,
                                                               const LogicalTensorPtr& tensor)
{
    std::unordered_set<LogicalTensorPtr> visitedTensors;
    return InferTargetTypeThroughForwardViews(inserter, tensor, visitedTensors);
}

MemoryType MemoryPathUtils::InferTargetTypeThroughForwardViews(const ConvertInserter& inserter,
                                                               const LogicalTensorPtr& tensor,
                                                               std::unordered_set<LogicalTensorPtr>& visitedTensors)
{
    if (tensor == nullptr || !visitedTensors.insert(tensor).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    // 仅当唯一 consumer 为 OP_VIEW 时沿视图链前向推导，规避多分支分歧
    const auto& consumers = tensor->GetConsumers();
    if (consumers.size() != 1) {
        return MemoryType::MEM_UNKNOWN;
    }
    auto consumerOp = *consumers.begin();
    if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_VIEW) {
        return MemoryType::MEM_UNKNOWN;
    }
    if (consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    auto viewOutput = consumerOp->oOperand.front();
    if (viewOutput->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return viewOutput->GetMemoryTypeOriginal();
    }
    MemoryType viewOutputRequirement = inserter.TryGetUniqueKnownRequiredType(viewOutput);
    if (viewOutputRequirement != MemoryType::MEM_UNKNOWN) {
        return viewOutputRequirement;
    }
    return InferTargetTypeThroughForwardViews(inserter, viewOutput, visitedTensors);
}

Status MemoryPathUtils::ResolveTensorMemoryUnknowns(ConvertInserter& inserter, const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Resolve tensor memory unknown failed because tensor is null.");
        return FAILED;
    }
    MemoryType original = tensor->GetMemoryTypeOriginal();
    if (original == MemoryType::MEM_UNKNOWN) {
        MemoryType inferredOriginal = InferOriginalFromRequirements(inserter, tensor);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Resolve unknown tensor[%d] original as %s from consumer requirements.",
                          tensor->GetMagic(), BriefMemoryTypeToString(inferredOriginal).c_str());
        RETURN_IF_NOT_SUCCESS(SetOriginalChecked(tensor, inferredOriginal, "ResolveMemoryUnknowns"));
        original = tensor->GetMemoryTypeOriginal();
    }
    FillUnknownRequirementsWith(inserter, tensor, original, "ResolveMemoryUnknowns");
    return SUCCESS;
}

MemoryType MemoryPathUtils::InferOriginalFromRequirements(const ConvertInserter& inserter,
                                                          const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return MemoryType::MEM_DEVICE_DDR;
    }
    auto knownRequirements = inserter.GetKnownRequiredTypes(tensor);
    if (knownRequirements.size() == 1) {
        return *knownRequirements.begin();
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Default tensor[%d] to DDR: %zu distinct known consumer requirements.",
                      tensor->GetMagic(), knownRequirements.size());
    return MemoryType::MEM_DEVICE_DDR;
}

} // namespace npu::tile_fwk
