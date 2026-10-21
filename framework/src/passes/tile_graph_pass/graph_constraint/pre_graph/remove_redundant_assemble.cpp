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
 * \file remove_redundant_assemble.cpp
 * \brief
 */

#include "remove_redundant_assemble.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_token_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/remove_redundant_op_utils.h"
#include "interface/configs/config_manager.h"

#include <sstream>
#include <unordered_set>

#define MODULE_NAME "PreGraphProcess"

namespace npu::tile_fwk {
namespace {
constexpr const char* KEY_ENABLE_TRUE = "enable_true";

bool HasInputOperand(const Operation& op, const LogicalTensorPtr& tensor)
{
    for (const auto& input : op.GetIOperands()) {
        if (input == tensor) {
            return true;
        }
    }
    return false;
}

void AddMissingCopyInConsumers(Function& function, const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return;
    }
    for (auto& op : function.Operations(false)) {
        if (!op.IsDeleted() && IsCopyIn(op.GetOpcode()) && HasInputOperand(op, tensor)) {
            tensor->AddConsumer(op);
        }
    }
}

// 静态offset全零, 且符号offset为空或全部可化简为常量0; 无法证明为零的符号偏移视为非零
bool IsZeroTensorOffset(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto& offset = tensor->GetOffset();
    if (std::any_of(offset.begin(), offset.end(), [](int64_t value) { return value != 0; })) {
        return false;
    }
    const auto& dynOffset = tensor->GetDynOffset();
    return std::all_of(dynOffset.begin(), dynOffset.end(), [](const SymbolicScalar& value) {
        auto simplified = value.Simplify();
        return simplified.ConcreteValid() && simplified.Concrete() == 0;
    });
}
} // namespace

std::string OpImmediateVecToStr(const std::vector<OpImmediate>& values)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << values[i].Dump();
    }
    ss << "]";
    return ss.str();
}

namespace {
ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

std::vector<Operation*> GetExternalProducers(Operation& op, const std::unordered_set<Operation*>& deletedOps)
{
    std::vector<Operation*> producers;
    for (auto* producer : op.ProducerOps()) {
        if (producer == nullptr || deletedOps.count(producer) != 0) {
            continue;
        }
        producers.push_back(producer);
    }
    return producers;
}

std::vector<Operation*> GetExternalConsumers(Operation& op, const std::unordered_set<Operation*>& deletedOps)
{
    std::vector<Operation*> consumers;
    for (auto* consumer : op.ConsumerOps()) {
        if (consumer == nullptr || deletedOps.count(consumer) != 0) {
            continue;
        }
        consumers.push_back(consumer);
    }
    return consumers;
}

std::unordered_set<ir::StmtPtr> BuildDeletedStmtSet(const std::vector<Operation*>& deletedOps)
{
    std::unordered_set<ir::StmtPtr> deletedStmts;
    for (auto* op : deletedOps) {
        if (op == nullptr) {
            continue;
        }
        deletedStmts.insert(ToStmtPtr(*op));
    }
    return deletedStmts;
}

std::unordered_set<ir::VarPtr> BuildDeletedResultTokenSet(const std::vector<Operation*>& deletedOps)
{
    std::unordered_set<ir::VarPtr> deletedResultTokens;
    for (auto* op : deletedOps) {
        if (op == nullptr) {
            continue;
        }
        deletedResultTokens.insert(op->result_token_.begin(), op->result_token_.end());
    }
    return deletedResultTokens;
}

void MoveInputTokensToConsumers(Function& function, const std::vector<Operation*>& deletedOps,
                                const std::vector<Operation*>& targetConsumers,
                                const std::unordered_set<ir::VarPtr>& deletedResultTokens)
{
    auto& dependency = function.GetVarDependency();
    for (auto* op : deletedOps) {
        if (op == nullptr) {
            continue;
        }
        auto inputTokens = op->tokens_;
        for (const auto& token : inputTokens) {
            dependency.RemoveConsumer(token, ToStmtPtr(*op));
            if (deletedResultTokens.count(token) != 0) {
                continue;
            }
            for (auto* consumer : targetConsumers) {
                if (consumer == nullptr || consumer == op) {
                    continue;
                }
                PassTokenUtils::AddTokenConsumer(function, token, *consumer);
            }
        }
        op->tokens_.clear();
    }
}

void MoveReshapeViewTokenBeforeRemove(Function& function, Operation& firstReshape, Operation& viewOp)
{
    std::vector<Operation*> deletedOps = {&firstReshape, &viewOp};
    std::unordered_set<Operation*> deletedOpSet(deletedOps.begin(), deletedOps.end());
    auto targetProducers = GetExternalProducers(firstReshape, deletedOpSet);
    auto targetConsumers = GetExternalConsumers(viewOp, deletedOpSet);
    auto deletedResultTokens = BuildDeletedResultTokenSet(deletedOps);
    auto deletedStmts = BuildDeletedStmtSet(deletedOps);
    MoveInputTokensToConsumers(function, deletedOps, targetConsumers, deletedResultTokens);
    PassTokenUtils::MoveResultTokensToProducers(function, deletedOps, targetProducers, deletedStmts);
    PassTokenUtils::CleanupDeletedTokenDependency(function, deletedOps);
}
} // namespace

std::vector<OpImmediate> SumOffset(const std::vector<OpImmediate> offset1, const std::vector<OpImmediate> offset2)
{
    std::vector<OpImmediate> res;
    for (size_t i = 0; i < offset1.size(); i++) {
        res.push_back(offset1[i] + offset2[i]);
    }
    return res;
}

bool GetViewOffset(const ViewOpAttribute& viewAttr, std::vector<OpImmediate>& viewOffset)
{
    if (!viewAttr.GetFromDynOffset().empty()) {
        viewOffset = OpImmediate::Specified(viewAttr.GetFromDynOffset());
    } else {
        viewOffset = OpImmediate::Specified(viewAttr.GetFromOffset());
    }
    return !viewOffset.empty();
}

bool CanMergeViewToCopyIn(const Operation& viewOp, std::vector<Operation*>& copyIns,
                          std::vector<std::vector<OpImmediate>>& mergedOffsets)
{
    if (viewOp.GetIOperands().size() != 1 || viewOp.GetOOperands().size() != 1) {
        return false;
    }
    const auto& viewInput = viewOp.GetIOperands().front();
    const auto& viewOutput = viewOp.GetOOperands().front();
    if (viewInput == nullptr || viewOutput == nullptr || viewInput->GetRawTensor() == nullptr ||
        viewInput->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR ||
        viewInput->GetShape().size() != viewOutput->GetShape().size() || viewOutput->GetConsumers().empty()) {
        return false;
    }
    // ReplaceTensor(ForUpdateView)可能已把view输入的offset吸收进view的from并同步刷新view输出的offset,
    // 此时from不再是相对view输入的纯窗口偏移; 继续折叠会使copyIn新输入(view输入)自身的offset与from中
    // 已吸收的输入offset重复累加, 导致搬运地址错误。仅当view输入的offset(静态+符号)全为零时,
    // from无论是否被吸收过取值都一致, 折叠才是安全的。
    if (!IsZeroTensorOffset(viewInput)) {
        return false;
    }
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(viewOp.GetOpAttribute());
    std::vector<OpImmediate> viewOffset;
    if (viewAttr == nullptr || !GetViewOffset(*viewAttr, viewOffset) ||
        viewOffset.size() != viewInput->GetShape().size()) {
        return false;
    }
    for (auto* consumer : viewOutput->GetConsumers()) {
        if (consumer == nullptr ||
            (consumer->GetOpcode() != Opcode::OP_COPY_IN && consumer->GetOpcode() != Opcode::OP_L1_COPY_IN_CONV) ||
            consumer->GetIOperands().size() != 1 || consumer->GetOOperands().size() != 1) {
            return false;
        }
        // ReplaceTensor::InsertCopyDDROp插入的copyIn, 其fromOffset已吸收view窗口偏移(绝对量),
        // codegen对DDR copyIn只使用fromOffset; 再将view偏移折叠进fromOffset会重复累加, 跳过折叠
        if (consumer->HasAttr(COPY_IN_VIEW_OFFSET_ABSORBED)) {
            return false;
        }
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(consumer->GetOpAttribute());
        if (copyAttr == nullptr) {
            mergedOffsets.push_back(viewOffset);
        } else {
            const auto& copyOffset = copyAttr->GetFromOffset();
            if (copyAttr->IsCopyOut() || copyOffset.size() != viewOffset.size() ||
                (!copyAttr->GetRawShape().empty() && copyAttr->GetRawShape().size() != viewOffset.size())) {
                return false;
            }
            mergedOffsets.push_back(SumOffset(viewOffset, copyOffset));
        }
        copyIns.push_back(consumer);
    }
    return true;
}

Status RemoveRedundantAssemble::ProcessViewToCopyIn(Function& function) const
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW || op.IsDeleted()) {
            continue;
        }
        std::vector<Operation*> copyIns;
        std::vector<std::vector<OpImmediate>> mergedOffsets;
        if (!CanMergeViewToCopyIn(op, copyIns, mergedOffsets)) {
            continue;
        }
        auto viewInput = op.GetIOperands().front();
        for (size_t idx = 0; idx < copyIns.size(); ++idx) {
            auto& copyIn = *copyIns[idx];
            auto copyOutput = copyIn.GetOOperands().front();
            // L1_COPY_IN_CONV 的 attr 约定 shape 为源窗口形状（与 rawShape 同维），其余 CopyIn 沿用输出 shape
            auto copyShape = copyIn.GetOpcode() == Opcode::OP_L1_COPY_IN_CONV ?
                                 OpImmediate::Specified(copyIn.GetIOperands().front()->GetShape()) :
                                 OpImmediate::Specified(copyOutput->GetShape());
            auto rawShape = OpImmediate::Specified(viewInput->tensor->GetDynRawShape());
            auto toDynValidShape = OpImmediate::Specified(copyOutput->GetDynValidShape());
            auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyIn.GetOpAttribute());
            if (copyAttr == nullptr) {
                copyAttr = std::make_shared<CopyOpAttribute>(mergedOffsets[idx], copyOutput->GetMemoryTypeOriginal(),
                                                             copyShape, rawShape, toDynValidShape);
                copyIn.SetOpAttribute(copyAttr);
            } else {
                copyAttr->SetFromOffset(mergedOffsets[idx]);
                copyAttr->SetShape(copyShape);
                copyAttr->SetRawShape(rawShape);
                copyAttr->SetToDynValidShape(toDynValidShape);
            }
            copyIn.ReplaceIOperand(0, viewInput);
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Remove View op:%s[%d] before CopyIn.", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic());
        op.SetAsDeleted();
    }
    return SUCCESS;
}

// 当前op为Copy Out时，需要将后继Assemble上的offset累加到当前op的CopyOpAttr上
void UpdateCopyOutAttr(Operation& op, Operation& opNext)
{
    auto opAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto opNextAttr = std::static_pointer_cast<AssembleOpAttribute>(opNext.GetOpAttribute());
    if (opNextAttr->GetToDynOffset().size() != 0) {
        if (op.GetOpcode() != Opcode::OP_COPY_OUT) {
            opAttr->SetToOffset(OpImmediate::Specified(opNextAttr->GetToDynOffset()));
        } else {
            opAttr->SetToOffset(SumOffset(OpImmediate::Specified(opNextAttr->GetToDynOffset()), opAttr->GetToOffset()));
        }
    } else {
        if (op.GetOpcode() != Opcode::OP_COPY_OUT) {
            opAttr->SetToOffset(OpImmediate::Specified(opNextAttr->GetToOffset()));
        } else {
            opAttr->SetToOffset(SumOffset(OpImmediate::Specified(opNextAttr->GetToOffset()), opAttr->GetToOffset()));
        }
    }
    opAttr->SetRawShape(OpImmediate::Specified(op.GetOOperands().front()->tensor->GetDynRawShape()));
}

bool CalculateNewRawShapeExpand(const std::vector<int64_t>& newShape, const std::vector<int64_t>& oriRawShape,
                                std::vector<int64_t>& newRawShape)
{
    newRawShape.resize(newShape.size());
    if (oriRawShape.size() < newShape.size()) {
        return false;
    }
    size_t diff = oriRawShape.size() - newShape.size();
    std::copy(oriRawShape.begin() + diff, oriRawShape.end(), newRawShape.begin());
    int64_t newShapeSize = 1;
    if (newRawShape.size() > 1) {
        newShapeSize = std::accumulate(newRawShape.begin() + 1, newRawShape.end(), INT64_C(1),
                                       std::multiplies<int64_t>());
    }
    int64_t oriShapeSize = std::accumulate(oriRawShape.begin(), oriRawShape.end(), INT64_C(1),
                                           std::multiplies<int64_t>());
    if (newShapeSize == 0 || oriShapeSize % newShapeSize != 0) {
        APASS_LOG_INFO_F(Elements::Function, "Cannot calculate NewRawShape as the dimension is not divisible.");
        return false;
    }
    newRawShape[0] = oriShapeSize / newShapeSize;
    return true;
}

bool CalculateNewRawShapeReduce(const std::vector<int64_t>& newShape, const std::vector<int64_t>& oriRawShape,
                                std::vector<int64_t>& newRawShape)
{
    newRawShape.resize(newShape.size());
    std::copy(newShape.begin(), newShape.end(), newRawShape.begin());
    int64_t newShapeSize = std::accumulate(newRawShape.begin(), newRawShape.end(), INT64_C(1),
                                           std::multiplies<int64_t>());
    int64_t oriShapeSize = std::accumulate(oriRawShape.begin(), oriRawShape.end(), INT64_C(1),
                                           std::multiplies<int64_t>());
    int64_t remainShapeSize = 1;
    if (newShapeSize > 0) {
        remainShapeSize = oriShapeSize / newShapeSize;
    }
    if (remainShapeSize <= 0 || oriShapeSize % newShapeSize != 0) {
        APASS_LOG_INFO_F(Elements::Function, "Cannot calculate NewRawShape as the dimension is not divisible.");
        return false;
    }
    newRawShape.back() = newRawShape.back() * remainShapeSize;
    return true;
}

bool CalculateNewDynRawShapeExpand(const std::vector<int64_t>& newShape,
                                   const std::vector<SymbolicScalar>& oriDynRawShape,
                                   std::vector<SymbolicScalar>& newDynRawShape)
{
    newDynRawShape.resize(newShape.size());
    if (oriDynRawShape.size() < newShape.size()) {
        APASS_LOG_ERROR_F(
            Elements::Function,
            "Cannot calculate NewDynRawShape as oriDynRawShape size [%zu] is smaller than newShape size [%zu].",
            oriDynRawShape.size(), newShape.size());
        return false;
    }
    size_t diff = oriDynRawShape.size() - newShape.size();
    std::copy(oriDynRawShape.begin() + diff, oriDynRawShape.end(), newDynRawShape.begin());
    SymbolicScalar newShapeSize(1);
    if (newDynRawShape.size() > 1) {
        newShapeSize = std::accumulate(newDynRawShape.begin() + 1, newDynRawShape.end(), IRBuilder().CreateConstInt(1),
                                       std::multiplies<>());
    }
    SymbolicScalar oriShapeSize = std::accumulate(oriDynRawShape.begin(), oriDynRawShape.end(),
                                                  IRBuilder().CreateConstInt(1), std::multiplies<>());
    if (newShapeSize.ConcreteValid() && newShapeSize == 0) {
        APASS_LOG_ERROR_F(Elements::Function, "Cannot calculate NewDynRawShape as the dimension is zero.");
        return false;
    }
    // Max(1) only avoids building a symbolic division-by-zero expression; runtime size >= 1 is guaranteed upstream.
    SymbolicScalar safeNewShapeSize = newShapeSize.Max(IRBuilder().CreateConstInt(1));
    newDynRawShape[0] = oriShapeSize / safeNewShapeSize;
    return true;
}

bool CalculateNewDynRawShapeExpand(const std::vector<int64_t>& newShape,
                                   const std::vector<SymbolicScalar>& oriDynRawShape,
                                   const std::vector<int64_t>& newRawShape, std::vector<SymbolicScalar>& newDynRawShape)
{
    if (oriDynRawShape.empty()) {
        newDynRawShape = CommonUtils::CreateConstIntVector(newRawShape);
        return true;
    }
    return CalculateNewDynRawShapeExpand(newShape, oriDynRawShape, newDynRawShape);
}

bool CalculateNewDynRawShapeReduce(const std::vector<int64_t>& newShape,
                                   const std::vector<SymbolicScalar>& oriDynRawShape,
                                   std::vector<SymbolicScalar>& newDynRawShape)
{
    newDynRawShape = CommonUtils::CreateConstIntVector(newShape);
    SymbolicScalar newShapeSize = std::accumulate(newDynRawShape.begin(), newDynRawShape.end(),
                                                  IRBuilder().CreateConstInt(1), std::multiplies<>());
    SymbolicScalar oriShapeSize = std::accumulate(oriDynRawShape.begin(), oriDynRawShape.end(),
                                                  IRBuilder().CreateConstInt(1), std::multiplies<>());
    if (newShapeSize.ConcreteValid() && newShapeSize == 0) {
        APASS_LOG_ERROR_F(Elements::Function, "Cannot calculate NewDynRawShape as the dimension is zero.");
        return false;
    }
    // Max(1) only avoids building a symbolic division-by-zero expression; runtime size >= 1 is guaranteed upstream.
    SymbolicScalar safeNewShapeSize = newShapeSize.Max(IRBuilder().CreateConstInt(1));
    SymbolicScalar remainShapeSize = oriShapeSize / safeNewShapeSize;
    newDynRawShape.back() = newDynRawShape.back() * remainShapeSize;
    return true;
}

bool CalculateNewDynRawShapeReduce(const std::vector<int64_t>& newShape,
                                   const std::vector<SymbolicScalar>& oriDynRawShape,
                                   const std::vector<int64_t>& newRawShape, std::vector<SymbolicScalar>& newDynRawShape)
{
    if (oriDynRawShape.empty()) {
        newDynRawShape = CommonUtils::CreateConstIntVector(newRawShape);
        return true;
    }
    return CalculateNewDynRawShapeReduce(newShape, oriDynRawShape, newDynRawShape);
}

void RemoveRedundantAssemble::HandleForAssembleFromInOut(
    Function& function, Operation& assembleOp, std::set<Operation*, LogicalTensor::CompareOp>& producersBackup) const
{
    LogicalTensorPtr inOrOutTensor = nullptr;
    if (function.IsFromInCast(assembleOp.iOperand[0]) || function.IsFromOutCast(assembleOp.iOperand[0])) {
        inOrOutTensor = assembleOp.iOperand[0];
    }
    if (inOrOutTensor == nullptr) {
        return;
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Find incast or outcast, tensor magic: %d, raw magic: %d.",
                      inOrOutTensor->magic, inOrOutTensor->GetRawMagic());
    for (const auto& producer : producersBackup) {
        producer->oOperand[0]->tensor = inOrOutTensor->tensor;
        for (auto& cons : producer->oOperand[0]->GetConsumers()) {
            if (cons->GetOpcode() == Opcode::OP_RESHAPE && cons->oOperand[0]->tensor->actualRawmagic != -1) {
                APASS_LOG_DEBUG_F(Elements::Operation, "consumer[%d] is OP_RESHAPE.", cons->GetOpMagic());
                cons->oOperand[0]->tensor->actualRawmagic = inOrOutTensor->GetRawMagic();
            }
        }
    }
}

void GetDynOffsetBeforeReshape(const std::vector<SymbolicScalar>& oriOffset, const std::vector<int64_t>& oriShape,
                               const std::vector<int64_t>& newShape, std::vector<SymbolicScalar>& newOffset)
{
    IRBuilder builder;
    // 计算原始shape的步长（stride）
    if (oriShape.size() != oriOffset.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "OriShape and oriOffset size mismatch.");
        return;
    }
    size_t oriSize = oriOffset.size();
    size_t newSize = newShape.size();
    std::vector<int64_t> oriStride(oriShape.size());
    int64_t currentStride = 1;
    for (int i = oriSize - 1; i >= 0; --i) {
        oriStride[i] = currentStride;
        currentStride *= oriShape[i];
    }
    std::vector<SymbolicScalar> oriStrideScalar;
    oriStrideScalar.reserve(oriStride.size());
    for (auto stride : oriStride) {
        oriStrideScalar.emplace_back(builder.CreateConstInt(stride));
    }
    // 计算原始偏移量对应的线性索引
    SymbolicScalar linearIndex = oriOffset[0] * oriStrideScalar[0];
    for (size_t i = 1; i < oriOffset.size(); ++i) {
        linearIndex = linearIndex + oriOffset[i] * oriStrideScalar[i];
    }

    // 计算新shape的步长
    std::vector<int64_t> newStride(newSize);
    currentStride = 1;
    for (int i = newSize - 1; i >= 0; --i) {
        newStride[i] = currentStride;
        currentStride *= newShape[i];
    }
    std::vector<SymbolicScalar> newStrideScalar;
    newStrideScalar.reserve(newStride.size());
    for (auto stride : newStride) {
        newStrideScalar.emplace_back(builder.CreateConstInt(stride));
    }

    // 根据线性索引计算新的偏移量
    newOffset.resize(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        newOffset[i] = (linearIndex / newStrideScalar[i]).Simplify();
        linearIndex = (linearIndex % newStrideScalar[i]).Simplify();
    }
}

void GetDynOffsetBeforeDynReshape(const std::vector<SymbolicScalar>& oriOffset,
                                  const std::vector<SymbolicScalar>& oriShape,
                                  const std::vector<SymbolicScalar>& newShape, std::vector<SymbolicScalar>& newOffset)
{
    if (oriShape.size() != oriOffset.size()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "OriDynShape and oriOffset size mismatch.");
        return;
    }
    size_t oriSize = oriOffset.size();
    size_t newSize = newShape.size();
    std::vector<SymbolicScalar> oriStride(oriShape.size());
    SymbolicScalar currentStride(1);
    for (int i = oriSize - 1; i >= 0; --i) {
        oriStride[i] = currentStride;
        currentStride = currentStride * oriShape[i];
    }
    SymbolicScalar linearIndex = oriOffset[0] * oriStride[0];
    for (size_t i = 1; i < oriOffset.size(); ++i) {
        linearIndex = linearIndex + oriOffset[i] * oriStride[i];
    }

    std::vector<SymbolicScalar> newStride(newSize);
    currentStride = IRBuilder().CreateConstInt(1);
    for (int i = newSize - 1; i >= 0; --i) {
        newStride[i] = currentStride;
        currentStride = currentStride * newShape[i];
    }

    newOffset.resize(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        newOffset[i] = (linearIndex / newStride[i]).Simplify();
        linearIndex = (linearIndex % newStride[i]).Simplify();
    }
}

void UpdateCopyInOffsetAfterReshapeInput(LogicalTensorPtr reshapeInput, const std::vector<int64_t>& newRawShape,
                                         const std::vector<OpImmediate>& reshapeOffset)
{
    auto oriShape = reshapeInput->GetShape();
    auto oriShapeScalar = CommonUtils::CreateConstIntVector(oriShape);
    auto newShapeScalar = CommonUtils::CreateConstIntVector(newRawShape);
    auto reshapeDynOffset = OpImmediate::ToSpecified(reshapeOffset);
    for (auto& consumer : reshapeInput->GetConsumers()) {
        if (consumer == nullptr || consumer->IsDeleted() || !IsCopyIn(consumer->GetOpcode()) ||
            !HasInputOperand(*consumer, reshapeInput)) {
            continue;
        }
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(consumer->GetOpAttribute());
        if (copyAttr == nullptr) {
            APASS_LOG_INFO_F(Elements::Operation, "CopyIn Op %d Attribute is nullptr, skip updating fromOffset.",
                             consumer->GetOpMagic());
            continue;
        }
        auto oriFromDynOffset = OpImmediate::ToSpecified(copyAttr->GetFromOffset());
        if (oriFromDynOffset.empty()) {
            oriFromDynOffset = CommonUtils::CreateConstIntVector(std::vector<int64_t>(oriShape.size(), 0));
        }
        std::vector<SymbolicScalar> newFromDynOffset;
        GetDynOffsetBeforeDynReshape(oriFromDynOffset, oriShapeScalar, newShapeScalar, newFromDynOffset);
        if (newFromDynOffset.size() != newShapeScalar.size()) {
            APASS_LOG_WARN_F(Elements::Operation,
                             "Skip updating CopyIn op:%s[%d] fromOffset because reshape offset remapping failed.",
                             consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic());
            continue;
        }
        for (size_t i = 0; i < newFromDynOffset.size() && i < reshapeDynOffset.size(); i++) {
            newFromDynOffset[i] = (newFromDynOffset[i] + reshapeDynOffset[i]).Simplify();
        }
        copyAttr->SetFromOffset(OpImmediate::Specified(newFromDynOffset));
        copyAttr->SetRawShape(OpImmediate::Specified(newRawShape));
    }
}

std::vector<int64_t> removeAllOnes(const std::vector<int64_t>& vec)
{
    std::vector<int64_t> result = vec;
    result.erase(std::remove(result.begin(), result.end(), 1), result.end());
    return result;
}

bool MatchReshapePattern(const LogicalTensorPtr& reshapeInput, const LogicalTensorPtr& reshapeOutput)
{
    constexpr size_t kShapeSizeThreshold = 3;
    auto inputShape = reshapeInput->GetShape();
    auto outputShape = reshapeOutput->GetShape();
    if (std::max(inputShape.size(), outputShape.size()) < kShapeSizeThreshold) {
        return false;
    }
    return removeAllOnes(inputShape) == removeAllOnes(outputShape);
}

// Build the axis mapping for a reshape that only inserts singleton axes.
// Unlike a regular reshape, a VIEW may expose only a slice of its source
// tensor, so the source raw shape and the reshape logical shape can have
// different element counts.  In that case raw-shape calculation must retain
// the source axes and only insert the new singleton axes.
bool BuildViewReshapeAxisMap(const Shape& reshapeInputShape, const Shape& reshapeOutputShape,
                             std::vector<size_t>& axisMap)
{
    if (reshapeOutputShape.size() <= reshapeInputShape.size()) {
        return false;
    }
    axisMap.clear();
    axisMap.resize(reshapeInputShape.size());
    size_t outputIdx = 0;
    for (size_t inputIdx = 0; inputIdx < reshapeInputShape.size(); ++inputIdx) {
        while (outputIdx < reshapeOutputShape.size() && reshapeOutputShape[outputIdx] != reshapeInputShape[inputIdx]) {
            // Extra axes are allowed only when they are singleton axes.
            if (reshapeOutputShape[outputIdx] != 1) {
                return false;
            }
            ++outputIdx;
        }
        if (outputIdx == reshapeOutputShape.size()) {
            return false;
        }
        axisMap[inputIdx] = outputIdx++;
    }
    while (outputIdx < reshapeOutputShape.size()) {
        if (reshapeOutputShape[outputIdx] != 1) {
            return false;
        }
        ++outputIdx;
    }
    return true;
}

Shape GetStaticShapeForDynAxes(const Shape& rawShape)
{
    Shape staticShape = rawShape;
    for (size_t idx = 0; idx < staticShape.size(); ++idx) {
        if (staticShape[idx] < 0) {
            staticShape[idx] = -1;
        }
    }
    return staticShape;
}

void UpdateCopyInAttrAfterRemoveView(Operation& reshapeOp, const Shape& newRawShape,
                                     const std::vector<SymbolicScalar>& newDynOffset,
                                     const std::vector<SymbolicScalar>& oriReshapeDynValidShape,
                                     const SymbolicScalar& validShapeExpr)
{
    for (const auto& copyIn : reshapeOp.GetOOperands().front()->GetConsumers()) {
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyIn->GetOpAttribute());
        if (copyAttr == nullptr) {
            APASS_LOG_INFO_F(Elements::Operation, "CopyIn Op %d Attribute is nullptr.", copyIn->GetOpMagic());
            continue;
        }
        auto oriCopyOffset = copyAttr->GetFromOffset();
        std::vector<OpImmediate> newOffset = OpImmediate::Specified(newDynOffset);
        if (newOffset.size() < oriCopyOffset.size()) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Cannot update CopyIn Op %d offset as newOffset size [%zu] is smaller than oriCopyOffset size [%zu].",
                copyIn->GetOpMagic(), newOffset.size(), oriCopyOffset.size());
            continue;
        }
        for (size_t i = 0; i < oriCopyOffset.size(); i++) {
            newOffset[i] = newOffset[i] + oriCopyOffset[i];
        }
        copyAttr->SetFromOffset(newOffset);
        copyAttr->SetRawShape(OpImmediate::Specified(newRawShape));
        auto copyOut = copyIn->GetOOperands().front();
        auto copyDynValidShape = copyOut->GetDynValidShape();
        if (copyDynValidShape.empty()) {
            copyDynValidShape = oriReshapeDynValidShape;
        }
        for (auto& validShape : copyDynValidShape) {
            validShape = (validShape * validShapeExpr).Simplify();
        }
        copyOut->UpdateDynValidShape(copyDynValidShape);
        copyAttr->SetToDynValidShape(OpImmediate::Specified(copyDynValidShape));
    }
}

void RemoveRedundantAssemble::UpdateReshapeShape(Operation& reshapeOp, LogicalTensorPtr tensorPtr,
                                                 const Shape& newRawShape,
                                                 const std::vector<SymbolicScalar>& newDynRawShape) const
{
    reshapeOp.SetAttr(OP_ATTR_PREFIX + "validShape", tensorPtr->dynValidShape_);
    tensorPtr->shape = newRawShape;
    tensorPtr->tensor->UpdateRawShape(newRawShape);
    tensorPtr->tensor->UpdateDynRawShape(newDynRawShape);
}

bool RemoveRedundantAssemble::CalculateViewReshapeRawShape(const Shape& inputShape, const Shape& outputShape,
                                                           const Shape& viewRawShape, Shape& rawShape,
                                                           std::vector<size_t>& axisMap) const
{
    // Direction follows the physical raw rank.  A VIEW can expose a logical
    // slice with fewer dimensions than its backing raw tensor; using logical
    // ranks here would incorrectly classify that case as axis insertion.
    bool isReduce = viewRawShape.size() < outputShape.size();
    rawShape = outputShape;
    if (isReduce && viewRawShape.size() == inputShape.size() &&
        BuildViewReshapeAxisMap(inputShape, outputShape, axisMap)) {
        rawShape.assign(outputShape.size(), 1);
        for (size_t i = 0; i < axisMap.size(); ++i) {
            rawShape[axisMap[i]] = viewRawShape[i];
        }
        return true;
    }
    return isReduce ? CalculateNewRawShapeReduce(outputShape, viewRawShape, rawShape) :
                      CalculateNewRawShapeExpand(outputShape, viewRawShape, rawShape);
}

bool RemoveRedundantAssemble::CalculateViewReshapeDynRawShape(const Shape& inputShape, const Shape& outputShape,
                                                              const std::vector<SymbolicScalar>& viewDynRawShape,
                                                              const Shape& rawShape, const std::vector<size_t>& axisMap,
                                                              std::vector<SymbolicScalar>& dynRawShape) const
{
    // Use the source dynamic raw rank, before the remapped output shape is
    // written into `rawShape`.
    bool isReduce = viewDynRawShape.size() < outputShape.size();
    if (isReduce && !axisMap.empty() && (viewDynRawShape.empty() || viewDynRawShape.size() == inputShape.size())) {
        if (viewDynRawShape.empty()) {
            dynRawShape = CommonUtils::CreateConstIntVector(rawShape);
        } else {
            dynRawShape.assign(outputShape.size(), IRBuilder().CreateConstInt(1));
            for (size_t i = 0; i < axisMap.size(); ++i) {
                dynRawShape[axisMap[i]] = viewDynRawShape[i];
            }
        }
        return true;
    }
    return isReduce ? CalculateNewDynRawShapeReduce(outputShape, viewDynRawShape, rawShape, dynRawShape) :
                      CalculateNewDynRawShapeExpand(outputShape, viewDynRawShape, rawShape, dynRawShape);
}

void RemoveRedundantAssemble::CalculateViewReshapeValidInfo(const LogicalTensorPtr& viewInput,
                                                            const ViewOpAttribute& viewAttr,
                                                            const LogicalTensorPtr& reshapeOutput,
                                                            ViewReshapeInfo& info) const
{
    info.reshapeDynValidShape = reshapeOutput->GetDynValidShape();
    if (info.reshapeDynValidShape.empty()) {
        info.reshapeDynValidShape = CommonUtils::CreateConstIntVector(reshapeOutput->shape);
    }
    auto validShape = viewInput->GetDynValidShape();
    if (validShape.empty()) {
        validShape = CommonUtils::CreateConstIntVector(viewInput->shape);
    }
    auto offset = viewAttr.GetFromDynOffset();
    if (offset.empty()) {
        offset = CommonUtils::CreateConstIntVector(viewAttr.GetFromOffset());
    }
    IRBuilder builder;
    info.validShapeExpr = builder.CreateConstInt(1);
    constexpr size_t dim4D = 4;
    if (validShape.size() == dim4D && offset.size() == dim4D) {
        info.validShapeExpr = (validShape[1] - offset[1]).Max(builder.CreateConstInt(0));
        info.validShapeExpr = info.validShapeExpr.Min(builder.CreateConstInt(1)).Simplify();
    } else if (validShape.size() == dim4D) {
        APASS_LOG_WARN_F(Elements::Tensor,
                         "Cannot update reshape dyn valid shape as validShape size [%zu] is not equal to offset "
                         "size [%zu].",
                         validShape.size(), offset.size());
    }
    auto physicalShape = viewInput->tensor->GetDynRawShape().empty() ?
                             CommonUtils::CreateConstIntVector(viewInput->shape) :
                             viewInput->tensor->GetDynRawShape();
    GetDynOffsetBeforeDynReshape(offset, physicalShape, info.dynRawShape, info.dynOffset);
}

bool RemoveRedundantAssemble::CalculateViewReshapeInfo(const Operation& reshapeOp, const LogicalTensorPtr& viewInput,
                                                       const ViewOpAttribute& viewAttr, ViewReshapeInfo& info) const
{
    const auto& inputShape = reshapeOp.GetIOperands().front()->shape;
    const auto& output = reshapeOp.GetOOperands().front();
    std::vector<size_t> axisMap;
    if (!CalculateViewReshapeRawShape(inputShape, output->shape, viewInput->tensor->GetRawShape(), info.rawShape,
                                      axisMap)) {
        return false;
    }
    if (!CalculateViewReshapeDynRawShape(inputShape, output->shape, viewInput->tensor->GetDynRawShape(), info.rawShape,
                                         axisMap, info.dynRawShape)) {
        return false;
    }
    CalculateViewReshapeValidInfo(viewInput, viewAttr, output, info);
    return true;
}

Status RemoveRedundantAssemble::ProcessView(Function& function) const
{
    std::vector<std::pair<Operation*, Operation*>> multiReshapeVector;
    if (SplitMultiConsumerReshape(function, multiReshapeVector) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "SplitMultiConsumerReshape failed.");
        return FAILED;
    }
    if (RemoveViewMultiReshape(multiReshapeVector) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "RemoveViewMultiReshape failed.");
        return FAILED;
    }
    if (RemoveViewSingleReshape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "RemoveViewSingleReshape failed.");
        return FAILED;
    }
    return SUCCESS;
}

/*
删除冗余的VIEW处理场景:
Brfore：
VIEW -> RESHAPE -> COPYIN
                -> COPYIN
After：
RESHAPE -> COPYIN
        -> COPYIN
*/
Status RemoveRedundantAssemble::RemoveViewSingleReshape(Function& function) const
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE)
            continue;
        auto& reshapeOp = op;
        if (!MatchReshapePattern(reshapeOp.GetIOperands().front(), reshapeOp.GetOOperands().front()))
            continue;
        auto producers = reshapeOp.GetIOperands().front()->GetProducers();
        if (producers.empty()) {
            APASS_LOG_INFO_F(Elements::Operation, "No producers found for RESHAPE op's input %d.",
                             reshapeOp.GetOpMagic());
            continue;
        }
        auto producerOp = *producers.begin();
        if (producerOp == nullptr || producers.size() != 1 || producerOp->GetOpcode() != Opcode::OP_VIEW)
            continue;
        auto viewInput = producerOp->GetIOperands().front();
        bool allConsumersCopyIn = true;
        for (auto reshapeConsumer : reshapeOp.GetOOperands().front()->GetConsumers()) {
            if (!IsCopyIn(reshapeConsumer->GetOpcode())) {
                allConsumersCopyIn = false;
                break;
            }
        }
        if (!allConsumersCopyIn) {
            APASS_LOG_INFO_F(Elements::Operation, "Skip RESHAPE op %d as output has non-CopyIn consumer.",
                             reshapeOp.GetOpMagic());
            continue;
        }
        auto opAttr = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
        if (opAttr == nullptr) {
            APASS_LOG_INFO_F(Elements::Operation, "Op %d Attribute is nullptr.", producerOp->GetOpMagic());
            continue;
        }
        ViewReshapeInfo reshapeInfo;
        if (!CalculateViewReshapeInfo(reshapeOp, viewInput, *opAttr, reshapeInfo)) {
            APASS_LOG_INFO_F(Elements::Operation, "Skip RESHAPE op %d as view reshape info calculation failed.",
                             reshapeOp.GetOpMagic());
            continue;
        }
        // Shape and offset details were calculated by the helper above.
        APASS_LOG_DEBUG_F(Elements::Operation, "Process View[%d] Tensor[%d]: newRawshape: %s, newOffset: %s.",
                          producerOp->GetOpMagic(), reshapeOp.GetOOperands().front()->GetMagic(),
                          IntVecToStr(reshapeInfo.rawShape).c_str(), IntVecToStr(reshapeInfo.dynOffset).c_str());
        Shape staticNewRawShape = GetStaticShapeForDynAxes(reshapeInfo.rawShape);
        UpdateCopyInAttrAfterRemoveView(reshapeOp, staticNewRawShape, reshapeInfo.dynOffset,
                                        reshapeInfo.reshapeDynValidShape, reshapeInfo.validShapeExpr);
        UpdateReshapeShape(reshapeOp, reshapeOp.GetOOperands().front(), staticNewRawShape, reshapeInfo.dynRawShape);
        PassTokenUtils::MoveTokenDependencyBeforeRemoveOp(function, *producerOp);
        reshapeOp.ReplaceIOperand(0, viewInput);
        producerOp->SetAsDeleted();
    }
    return SUCCESS;
}

// 检测并优化特定的reshape模式：当输入张量的前两个维度中有一个为1时，
// 这两个维度可以合并为单个维度（例如 [1, N, ...] 或 [N, 1, ...] → [N, ...]）
bool RemoveViewMultiReshapePattern(const LogicalTensorPtr& reshapeInput, const LogicalTensorPtr& reshapeOutput)
{
    auto longerRawShape = reshapeInput->GetRawTensor()->GetRawShape();
    auto shorterRawShape = reshapeOutput->GetRawTensor()->GetRawShape();
    // 确保longerRawShape是维度数更大的一个
    if (longerRawShape.size() < shorterRawShape.size()) {
        std::swap(longerRawShape, shorterRawShape);
    }

    // 维度数一致或太小（小于1）则失败
    if (longerRawShape.size() == shorterRawShape.size() ||
        std::min(longerRawShape.size(), shorterRawShape.size()) < 1) {
        return false;
    }

    // 检查总元素数量是否一致
    auto longerRawShapeSize = reshapeInput->GetRawTensor()->GetRawShapeSize();
    auto shorterRawShapeSize = reshapeOutput->GetRawTensor()->GetRawShapeSize();
    if (longerRawShapeSize != shorterRawShapeSize) {
        return false;
    }

    // 检查维度为1的条件
    auto haveOne = longerRawShape[0] == 1 || longerRawShape[1] == 1;
    return haveOne && longerRawShape[0] * longerRawShape[1] == shorterRawShape[0];
}

/*
拆分RESHAPE
Before:
RESHAPE -> VIEW -> RESHAPE
        -> COPYIN
        -> COPYIN

After:
RESHAPE -> VIEW -> RESHAPE
RESHAPE -> COPYIN
        -> COPYIN
*/
Status RemoveRedundantAssemble::SplitMultiConsumerReshape(
    Function& function, std::vector<std::pair<Operation*, Operation*>>& multiReshapeVector) const
{
    for (auto op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        auto firstReshape = op;
        if (!RemoveViewMultiReshapePattern(firstReshape->GetIOperands().front(),
                                           firstReshape->GetOOperands().front())) {
            continue;
        }
        auto consumer = firstReshape->GetOOperands().front()->GetConsumers();
        for (auto consumerOp : consumer) {
            if (consumerOp->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            auto viewConsumers = consumerOp->GetOOperands().front()->GetConsumers();
            auto viewConsumerOp = *viewConsumers.begin();
            if (viewConsumerOp == nullptr || viewConsumers.size() != 1 ||
                viewConsumerOp->GetOpcode() != Opcode::OP_RESHAPE) {
                continue;
            }
            multiReshapeVector.push_back(std::make_pair(firstReshape, consumerOp));
        }
    }
    return SUCCESS;
}

// 处理 RESHAPE 拆分逻辑
Status RemoveRedundantAssemble::ProcessReshape(Function& function, Operation*& operation,
                                               std::vector<std::pair<Operation*, Operation*>>& multiReshapeVector) const
{
    if (operation == nullptr) {
        return FAILED;
    }
    auto iOperand = operation->iOperand[0];
    auto oOperand = operation->oOperand[0];
    if (oOperand == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Null output operands detected while iterating over the output operands of the operation [%d].%s",
            operation->opmagic, GetFormatBacktrace(operation).c_str());
        return FAILED;
    }
    if (iOperand == nullptr) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Null input operands detected while iterating over the input operands of the operation [%d].%s",
            operation->opmagic, GetFormatBacktrace(operation).c_str());
        return FAILED;
    }
    auto consumers = oOperand->GetConsumers();
    for (auto& consumer : consumers) {
        if (consumer == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor,
                              "Null consumer detected while iterating over the consumers of the output operand [%d].",
                              oOperand->magic);
            return FAILED;
        }
        if (consumer->GetOpcode() == Opcode::OP_COPY_IN) {
            continue;
        }
        auto dst = oOperand->Clone(function, true);
        if (dst == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Clone failed for output operand [%d].", oOperand->magic);
            return FAILED;
        }
        consumer->ReplaceInput(dst, oOperand);
        IRBuilder builder;
        auto& newReshapeOp = builder.CreateTensorOpStmt(function, Opcode::OP_RESHAPE, {iOperand}, {dst});
        const std::shared_ptr<OpAttribute> oriReshapeAttr = operation->GetOpAttribute();
        if (oriReshapeAttr != nullptr) {
            newReshapeOp.SetOpAttribute(oriReshapeAttr);
        }
        multiReshapeVector.emplace_back(&newReshapeOp, consumer);
    }
    return SUCCESS;
}

/*
删除冗余RESHAPE
Before:
input --> RESHAPE1 -> VIEW -> RESHAPE2 -> XXX

After:
input --> RESHAPE2 -> XXX

*/
Status RemoveRedundantAssemble::RemoveViewMultiReshape(
    const std::vector<std::pair<Operation*, Operation*>>& multiReshapeVector) const
{
    for (const auto& pair : multiReshapeVector) {
        auto firstReshape = pair.first;
        auto viewOp = pair.second;
        if (viewOp->GetOutputOperand(0) == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "VIEW operator [%d]: OutputOperand[0] is a null pointer. %s",
                              viewOp->GetOpMagic(), GetFormatBacktrace(viewOp).c_str());
            return FAILED;
        }
        auto consumers = viewOp->GetOutputOperand(0)->GetConsumers();
        if (consumers.empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "VIEW operator [%d]: OutputOperand[0] has no consumers. %s",
                              viewOp->GetOpMagic(), GetFormatBacktrace(viewOp).c_str());
            return FAILED;
        }
        auto firstReshapeOutput = firstReshape->GetOOperands()[0];
        auto viewOutput = viewOp->GetOOperands()[0];
        if (SubgraphUtils::IsBoundary(firstReshapeOutput) || SubgraphUtils::IsBoundary(viewOutput)) {
            continue;
        }
        auto secondReshape = *consumers.begin();
        if (secondReshape == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "SecondReshape is null.");
            return FAILED;
        }
        auto oriRawShape = secondReshape->GetIOperands().front()->GetRawTensor()->GetRawShape();
        auto oriDynRawShape = secondReshape->GetIOperands().front()->GetRawTensor()->GetDynRawShape();
        Shape newShape;
        std::remove_copy_if(oriRawShape.begin(), oriRawShape.end(), std::back_inserter(newShape),
                            [](const auto& e) { return e == 1; });
        std::vector<SymbolicScalar> newDynShape;
        if (!oriDynRawShape.empty() && oriDynRawShape.size() == oriRawShape.size()) {
            for (size_t idx = 0; idx < oriRawShape.size(); ++idx) {
                if (oriRawShape[idx] != 1) {
                    newDynShape.push_back(oriDynRawShape[idx]);
                }
            }
        }
        if (newDynShape.empty() && !newShape.empty()) {
            newDynShape = CommonUtils::CreateConstIntVector(newShape);
        }
        secondReshape->GetOOperands().front()->GetRawTensor()->UpdateRawShape(newShape);
        secondReshape->GetOOperands().front()->GetRawTensor()->UpdateDynRawShape(newDynShape);
        MoveReshapeViewTokenBeforeRemove(*secondReshape->BelongTo(), *firstReshape, *viewOp);
        secondReshape->ReplaceIOperand(0, firstReshape->GetIOperands().front());
        firstReshape->SetAsDeleted();
        viewOp->SetAsDeleted();
    }
    return SUCCESS;
}

/*
生效场景:
Assemble拆分了最高轴，认为可以透传，不需要拷贝，前序在ExpandFunction中做了判断，属性NeedCopy=false
Copy_Out --> tensor(GM) --> Reshape --> oriBackUp [16, 16] --> Assemble(offset, dynOffset) --> OCAST(offset, dynOffset)
[16, 64] 因此需要: 重新计算Reshape输入的RawShape, offset, dynOffset
*/
bool RemoveRedundantAssemble::CalculateReshapeToAssembleInfo(Operation& assembleOp, Operation& reshapeOp,
                                                             const std::vector<SymbolicScalar>& dynOffset,
                                                             std::vector<int64_t>& newRawShape,
                                                             std::vector<SymbolicScalar>& newDynRawShape,
                                                             std::vector<SymbolicScalar>& newDynOffset) const
{
    auto& assembleOutShape = assembleOp.GetOOperands()[0]->tensor->rawshape;
    auto assembleOutDynShape = assembleOp.GetOOperands()[0]->tensor->GetDynRawShape();
    bool isReduce = assembleOutShape.size() < reshapeOp.GetIOperands()[0]->shape.size();
    bool isSuccess = isReduce ?
                         CalculateNewRawShapeReduce(reshapeOp.GetIOperands()[0]->shape, assembleOutShape, newRawShape) :
                         CalculateNewRawShapeExpand(reshapeOp.GetIOperands()[0]->shape, assembleOutShape, newRawShape);
    if (!isSuccess) {
        return false;
    }
    isSuccess = isReduce ? CalculateNewDynRawShapeReduce(reshapeOp.GetIOperands()[0]->shape, assembleOutDynShape,
                                                         newRawShape, newDynRawShape) :
                           CalculateNewDynRawShapeExpand(reshapeOp.GetIOperands()[0]->shape, assembleOutDynShape,
                                                         newRawShape, newDynRawShape);
    if (!isSuccess) {
        return false;
    }
    auto dynAssembleOutShape = assembleOutDynShape.empty() ? CommonUtils::CreateConstIntVector(assembleOutShape) :
                                                             assembleOutDynShape;
    GetDynOffsetBeforeDynReshape(dynOffset, dynAssembleOutShape, newDynRawShape, newDynOffset);
    return true;
}

Status RemoveRedundantAssemble::UpdateCopyOutBeforeReshape(LogicalTensorPtr reshapeInput,
                                                           const std::vector<int64_t>& newRawShape,
                                                           const std::vector<SymbolicScalar>& newDynOffset) const
{
    if (reshapeInput == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Failed to update CopyOut before Reshape because reshape input is nullptr.");
        return FAILED;
    }
    // The reshape offset is common to all CopyOut and CopyIn operations using reshapeInput.
    const std::vector<OpImmediate> reshapeOffset = OpImmediate::Specified(newDynOffset);
    Function* function = nullptr;
    bool needUpdateCopyIn = false;
    auto updateCopyIn = [&]() {
        if (!needUpdateCopyIn) {
            return;
        }
        if (function != nullptr) {
            AddMissingCopyInConsumers(*function, reshapeInput);
        }
        UpdateCopyInOffsetAfterReshapeInput(reshapeInput, newRawShape, reshapeOffset);
    };
    for (auto copyOut : reshapeInput->GetProducers()) {
        if (copyOut->IsDeleted()) {
            continue;
        }
        if (!IsCopyOut(copyOut->GetOpcode())) {
            updateCopyIn();
            return SUCCESS;
        }
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOut->GetOpAttribute());
        if (copyAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Failed to update CopyOut before Reshape because CopyOut op:%s[%d] attr is nullptr.",
                              copyOut->GetOpcodeStr().c_str(), copyOut->GetOpMagic());
            return FAILED;
        }
        auto oriCopyOffset = copyAttr->GetToOffset();
        std::vector<OpImmediate> newOffset = reshapeOffset;
        if (newOffset.size() < oriCopyOffset.size()) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Failed to update CopyOut op:%s[%d] before Reshape because new offset size [%zu] is smaller than "
                "original CopyOut offset size [%zu]. newDynOffset: %s, oriCopyOffset: %s.",
                copyOut->GetOpcodeStr().c_str(), copyOut->GetOpMagic(), newOffset.size(), oriCopyOffset.size(),
                IntVecToStr(newDynOffset).c_str(), OpImmediateVecToStr(oriCopyOffset).c_str());
            return FAILED;
        }
        for (size_t i = 0; i < oriCopyOffset.size(); i++) {
            newOffset[i] = newOffset[i] + oriCopyOffset[i];
        }
        copyAttr->SetRawShape(OpImmediate::Specified(newRawShape));
        copyAttr->SetToOffset(newOffset);
        if (copyOut->BelongTo() != nullptr) {
            function = copyOut->BelongTo();
        }
        needUpdateCopyIn = true;
    }
    updateCopyIn();
    return SUCCESS;
}

Status RemoveRedundantAssemble::HandleReshapeToAssemble(
    Operation& assembleOp, const std::set<Operation*, LogicalTensor::CompareOp>& producers) const
{
    auto opAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp.GetOpAttribute());
    if (opAttr == nullptr)
        return FAILED;
    auto dynOffset = opAttr->GetToDynOffset();
    if (dynOffset.empty()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] does not have DynOffset attributes",
                          assembleOp.GetOpcodeStr().c_str(), assembleOp.GetOpMagic());
        dynOffset = OpImmediate::ToSpecified(OpImmediate::Specified(opAttr->GetToOffset()));
    }
    if (producers.size() != 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] has multiple producer operations, size: %zu",
                          assembleOp.GetOpcodeStr().c_str(), assembleOp.GetOpMagic(), producers.size());
        return SUCCESS;
    }
    auto producer = *(producers.begin());
    if (producer->GetOpcode() != Opcode::OP_RESHAPE) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Producer op:%s[%d] is not Reshape", producer->GetOpcodeStr().c_str(),
                          producer->GetOpMagic());
        return SUCCESS;
    }
    std::vector<int64_t> newRawShape;
    std::vector<SymbolicScalar> newDynRawShape;
    std::vector<SymbolicScalar> newDynOffset;
    if (!CalculateReshapeToAssembleInfo(assembleOp, *producer, dynOffset, newRawShape, newDynRawShape, newDynOffset)) {
        return SUCCESS;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Process Assemble %d Tensor[%d]: newRawshape: %s, newOffset: %s.",
                      assembleOp.GetOpMagic(), producer->GetIOperands()[0]->GetMagic(),
                      IntVecToStr(newRawShape).c_str(), IntVecToStr(newDynOffset).c_str());
    Shape newShape = GetStaticShapeForDynAxes(newRawShape);
    if (UpdateCopyOutBeforeReshape(producer->GetIOperands()[0], newShape, newDynOffset) != SUCCESS) {
        return FAILED;
    }
    producer->GetIOperands()[0]->tensor->UpdateRawShape(newShape);
    producer->GetIOperands()[0]->tensor->UpdateDynRawShape(newDynRawShape);
    return SUCCESS;
}

Status RemoveRedundantAssemble::HandleCopyOutToAssemble(
    Operation& assembleOp, LogicalTensorPtr output,
    const std::set<Operation*, LogicalTensor::CompareOp>& producers) const
{
    for (auto& producer : producers) {
        auto oriOutputBackUp = producer->oOperand[0]; // producer --> oriOutputBackUp(input) --> assembleOp
        producer->ReplaceOutput(output, oriOutputBackUp);
        if (!IsCopyOut(producer->GetOpcode())) {
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "The producer op:[%d] is copyOut, update its CopyOpAttr. %s",
                          producer->GetOpMagic(), producer->GetOpcodeStr().c_str());
        UpdateCopyOutAttr(*producer, assembleOp);
    }
    return SUCCESS;
}

/* 将某个op的输入是expected的替换为newTensor并刷新Producer、Consumer关系 */
void SubstituteInput(Operation& op, LogicalTensorPtr& expected, LogicalTensorPtr& newTensor)
{
    for (auto& input : op.iOperand) {
        if (input == expected) {
            newTensor->AddConsumer(op);
            input->RemoveConsumer(op);
            input = newTensor;
        }
    }
}

bool RemoveRedundantAssemble::IsCandidateAssembleOp(Function& function, Operation& op) const
{
    if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }
    auto& output = op.GetOOperands().front();
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR || op.IsDeleted()) {
        return false;
    }
    for (auto& prod : function.FindProducers(op)) {
        if (prod->GetOpcode() != Opcode::OP_VIEW) {
            if (prod->GetOpcode() == Opcode::OP_HUB) {
                return false;
            }
            return true;
        }
    }
    return false;
}

void RemoveRedundantAssemble::HandleForReshapeToOutcast(Function& function) const
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        auto dstRT = op.GetOOperands().front()->tensor;
        auto srcRT = op.GetIOperands().front()->tensor;
        if (function.outIncastLinkMap.count(dstRT) && function.outIncastLinkMap[dstRT] == srcRT) {
            continue;
        }
        if (!function.IsFromOutCast(op.GetOOperands()[0])) {
            continue;
        }
        // INPUT --> RESHAPE --> OCAST
        if (op.GetIOperands()[0]->tensor->actualRawmagic != -1) {
            // 说明输入也来自于reshape，需要把这条actual raw链上的tensor重定向到新的outcast raw
            int inputActualRawId = op.GetIOperands()[0]->tensor->actualRawmagic;
            auto inputRaw = GraphUtils::GetRawTensorByRawMagic(function, inputActualRawId);
            if (inputRaw != nullptr) {
                inputRaw->actualRawmagic = op.GetOOperands()[0]->GetRawMagic();
            }
        }
        op.GetIOperands()[0]->tensor->actualRawmagic = op.GetOOperands()[0]->GetRawMagic();
    }
}

/*
                /--> Assemble1-1(self) --> Tensor
op1 --> tensor1 ---> Assemble1-2 --> OCAST
                \--> Reshape --> tensor2
*/
/*
                /--> Assemble1-1(self) --> Tensor
op1 --> tensor1 ---> Assemble1-2 --> OCAST

                /--> Assemble2-1 --> Tensor
op2 --> tensor2 ---> Assemble2-2 --> OCAST
*/
void RemoveRedundantAssemble::HandleForAssembleToOutcast(
    Function& function, Operation& assembleOp, std::set<Operation*, LogicalTensor::CompareOp>& producersBackup) const
{
    int outCastMagic = -1;
    if (function.IsFromOutCast(assembleOp.oOperand[0]) &&
        FunctionUtils::GetNodeType(*(assembleOp.oOperand[0]), function) == NodeType::OUTCAST) {
        outCastMagic = assembleOp.oOperand[0]->GetMagic();
    }
    if (outCastMagic != -1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Find outCastMagic: %d.", outCastMagic);
        for (auto& producer : producersBackup) {
            producer->oOperand[0]->SetMagic(outCastMagic);
        }
    }
}

void RemoveRedundantAssemble::HanldeForMultiAssemble(Function& function,
                                                     std::unordered_set<Operation*>& concurrentAssembles) const
{
    LogicalTensorPtr replaceTensor = nullptr;
    for (auto& assemble : concurrentAssembles) {
        if (function.IsFromInCast(assemble->iOperand[0]) || function.IsFromOutCast(assemble->iOperand[0])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d]'s iOperand comes from Incast or Outcast",
                              assemble->GetOpcodeStr().c_str(), assemble->GetOpMagic());
            replaceTensor = assemble->iOperand[0];
            break;
        } else if (function.IsFromInCast(assemble->oOperand[0]) || function.IsFromOutCast(assemble->oOperand[0])) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d]'s oOperand comes from Incast or Outcast",
                              assemble->GetOpcodeStr().c_str(), assemble->GetOpMagic());
            replaceTensor = assemble->oOperand[0];
            break;
        }
    }
    for (auto& assemble : concurrentAssembles) {
        if (replaceTensor == nullptr)
            replaceTensor = assemble->oOperand[0];
        auto& input = assemble->GetIOperands().front();
        auto& output = assemble->GetOOperands().front();
        input->tensor = replaceTensor->tensor;
        output->tensor = replaceTensor->tensor;
    }
}

Status RemoveRedundantAssemble::HanldeForSingleAssemble(Function& function, LogicalTensorPtr input,
                                                        LogicalTensorPtr output, Operation& op) const
{
    auto producersBackup = input->GetProducers();
    auto& consumers = input->GetConsumers();
    // Only fold the additional COPY_OUT -> ASSEMBLE -> RESHAPE -> OUTCAST pattern.
    // The caller guarantees a single sibling ASSEMBLE. ReplaceTensor has already refreshed COPY_IN attributes.
    const auto& outputConsumers = output->GetConsumers();
    bool reshapeToOutcast = outputConsumers.size() == 1 &&
                            (*outputConsumers.begin())->GetOpcode() == Opcode::OP_RESHAPE &&
                            FunctionUtils::GetNodeType(*(*outputConsumers.begin())->GetOOperands().front(), function) ==
                                NodeType::OUTCAST;
    const auto& outputProducers = output->GetProducers();
    bool rewireCopyIns = reshapeToOutcast && outputProducers.size() == 1 && *outputProducers.begin() == &op &&
                         consumers.size() > 1 && input->GetRawMagic() == output->GetRawMagic() &&
                         std::all_of(producersBackup.begin(), producersBackup.end(),
                                     [](Operation* producer) { return IsCopyOut(producer->GetOpcode()); }) &&
                         std::all_of(consumers.begin(), consumers.end(), [&op](Operation* consumer) {
                             return consumer == &op || IsCopyIn(consumer->GetOpcode());
                         });
    if (consumers.size() == 1 || rewireCopyIns) {
        PassTokenUtils::MoveTokenDependencyBeforeRemoveOp(function, op);
        if (rewireCopyIns) {
            auto consumersBackup = consumers;
            for (auto* consumer : consumersBackup) {
                if (consumer != &op) {
                    // Preserve the absolute COPY_IN attributes refreshed by ReplaceTensor.
                    consumer->ReplaceInput(output, input);
                }
            }
        }
        op.SetAsDeleted();
        if (HandleCopyOutToAssemble(op, output, producersBackup) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "HandleCopyOutToAssemble for op:[%d] failed. %s", op.GetOpMagic(),
                              op.GetOpcodeStr().c_str());
            return FAILED;
        }
        HandleForAssembleFromInOut(function, op, producersBackup);
        HandleForAssembleToOutcast(function, op, producersBackup);
    }
    if (HandleReshapeToAssemble(op, producersBackup) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "HandleReshapeToAssemble for op:[%d] failed. %s", op.GetOpMagic(),
                          op.GetOpcodeStr().c_str());
        return FAILED;
    }
    return SUCCESS;
}

/*
    Producer1 -->
                 \
    Producer2 --> input --> consAssemble -----> Tensor1 --> Op1
                    \---> consumer ------> Tensor2 --> Op2
    will be modified to:
    Producer1 -->
                 \
    Producer2 --> Tensor1 --> Op1
                    \---> Consumer ------> Tensor2 --> Op2
*/
Status RemoveRedundantAssemble::DeleteRedundantAssemble(Function& function) const
{
    bool operationUpdated = false;
    if (config::GetPassGlobalConfig(KEY_ENABLE_TRUE, false)) {
        if (RemoveRedundantOpUtils::ProcessViewCopyout(function, operationUpdated) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessViewCopyout failed.");
            return FAILED;
        }
        if (RemoveRedundantOpUtils::ProcessCopyinAssemble(function, operationUpdated) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessCopyinAssemble failed.");
            return FAILED;
        }
        if (operationUpdated) {
            function.EraseOperations(true, false);
        }
    }
    for (auto& op : function.Operations()) {
        if (!IsCandidateAssembleOp(function, op)) {
            continue;
        }
        auto& input = op.GetIOperands().front();
        auto& output = op.GetOOperands().front();
        auto& consumers = input->GetConsumers();
        std::unordered_set<Operation*> concurrentAssembles;
        for (auto& cons : consumers) {
            if (cons->GetOpcode() == Opcode::OP_ASSEMBLE) {
                concurrentAssembles.emplace(cons);
            }
        }
        if (concurrentAssembles.size() > 1) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Op:%s[%d] has %zu parallel assemble op", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic(), concurrentAssembles.size());
            HanldeForMultiAssemble(function, concurrentAssembles);
        } else {
            if (HanldeForSingleAssemble(function, input, output, op) != SUCCESS)
                return FAILED;
        }
    }
    if (ProcessView(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessView failed.");
        return FAILED;
    }
    function.EraseOperations(false, true, SortOperationsMode::LIGHTWEIGHT);
    HandleForReshapeToOutcast(function);
    return SUCCESS;
}
} // namespace npu::tile_fwk
