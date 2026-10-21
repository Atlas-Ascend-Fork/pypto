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
 * \file replace_tensor.cpp
 * \brief
 */

#include "replace_tensor.h"
#include <algorithm>
#include <deque>
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/tensor_utils.h"
#include "tilefwk/error_code.h"
#include "passes/pass_utils/alignment_utils.h"

#define MODULE_NAME "ReplaceTensor"

namespace npu {
namespace tile_fwk {
namespace {
using Token = ir::VarPtr;

ir::StmtPtr GetStmt(const Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

bool HasSameRawTensorInput(const Operation& lhs, const Operation& rhs)
{
    for (const auto& lhsInput : lhs.GetIOperands()) {
        if (lhsInput == nullptr || lhsInput->GetRawTensor() == nullptr) {
            continue;
        }
        for (const auto& rhsInput : rhs.GetIOperands()) {
            if (rhsInput != nullptr && lhsInput->tensor->GetRawMagic() == rhsInput->tensor->GetRawMagic()) {
                return true;
            }
        }
    }
    return false;
}

bool HasInputMatchingOutputRawTensor(const Operation& inputOp, const Operation& outputOp)
{
    for (const auto& input : inputOp.GetIOperands()) {
        if (input == nullptr || input->GetRawTensor() == nullptr) {
            continue;
        }
        for (const auto& output : outputOp.GetOOperands()) {
            if (output != nullptr && output->GetRawTensor() != nullptr &&
                input->GetRawTensor()->GetRawMagic() == output->GetRawTensor()->GetRawMagic()) {
                return true;
            }
        }
    }
    return false;
}

bool IsInplaceIdxOperation(const Operation& op)
{
    return (op.GetOpcode() == Opcode::OP_INDEX_PUT || op.GetOpcode() == Opcode::OP_SCATTER_INPLACE ||
            op.GetOpcode() == Opcode::OP_SCATTER_ELEMENT_INPLACE || op.GetOpcode() == Opcode::OP_INDEX_ADD) &&
           op.HasAttribute(OpAttributeKey::inplaceIdx);
}

bool IsInplaceOperation(const Operation& op)
{
    if (inplaceOpSet.count(op.GetOpcode()) != 0) {
        return true;
    }
    return (op.GetOpcode() == Opcode::OP_COPY_OUT && op.HasAttribute(OpAttributeKey::inplaceIdx)) ||
           IsInplaceIdxOperation(op);
}

std::vector<Operation*> FindFirstNonInplaceConsumers(Operation& op)
{
    const auto orderedConsumers = op.ConsumerOpsOrdered();
    std::deque<Operation*> pending(orderedConsumers.begin(), orderedConsumers.end());
    std::unordered_set<Operation*> visited;
    std::vector<Operation*> result;
    while (!pending.empty()) {
        auto* current = pending.front();
        pending.pop_front();
        if (current == nullptr || !visited.insert(current).second) {
            continue;
        }
        if (!IsInplaceOperation(*current)) {
            result.push_back(current);
            continue;
        }
        for (auto* consumer : current->ConsumerOpsOrdered()) {
            pending.push_back(consumer);
        }
    }
    return result;
}

std::vector<Operation*> FindFirstNonInplaceProducers(Operation& op)
{
    const auto orderedProducers = op.ProducerOpsOrdered();
    std::deque<Operation*> pending(orderedProducers.begin(), orderedProducers.end());
    std::unordered_set<Operation*> visited;
    std::vector<Operation*> result;
    while (!pending.empty()) {
        auto* current = pending.front();
        pending.pop_front();
        if (current == nullptr || !visited.insert(current).second) {
            continue;
        }
        if (!IsInplaceOperation(*current)) {
            result.push_back(current);
            continue;
        }
        for (auto* producer : current->ProducerOpsOrdered()) {
            pending.push_back(producer);
        }
    }
    return result;
}

bool HasDependencyPath(Operation& from, Operation& to)
{
    std::deque<Operation*> pending{&from};
    std::unordered_set<Operation*> visited;
    while (!pending.empty()) {
        auto* current = pending.front();
        pending.pop_front();
        if (current == nullptr || !visited.insert(current).second) {
            continue;
        }
        if (current == &to) {
            return true;
        }
        for (auto* consumer : current->ConsumerOpsOrdered()) {
            pending.push_back(consumer);
        }
        for (auto* consumer : current->ConsumerOpsByToken()) {
            pending.push_back(consumer);
        }
    }
    return false;
}

void RemoveTokenConsumer(Operation& op, const Token& token, Function& function)
{
    op.tokens_.erase(std::remove(op.tokens_.begin(), op.tokens_.end(), token), op.tokens_.end());
    function.GetVarDependency().RemoveConsumer(token, GetStmt(op));
}

void AddTokenConsumer(Operation& op, const Token& token, Function& function)
{
    if (std::find(op.tokens_.begin(), op.tokens_.end(), token) == op.tokens_.end()) {
        op.tokens_.push_back(token);
    }
    function.GetVarDependency().AddConsumer(token, GetStmt(op));
}
} // namespace

bool ReplaceTensor::CheckAddrConflict(const Operation& op)
{
    auto tensorIn = op.GetIOperands().front();
    auto tensorOut = op.GetOOperands().front();
    if (tensorIn->GetRawMagic() != tensorOut->GetRawMagic() &&
        tensorIn->GetRawTensor()->memoryId != tensorOut->GetRawTensor()->memoryId) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "%s op[%d] invalid or conflict. tensorIn magic: %d, rawMagic: %d, tensorOut magic: %d, rawMagic: %d",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), tensorIn->GetMagic(), tensorIn->GetRawMagic(),
            tensorOut->GetMagic(), tensorOut->GetRawMagic());
        return true;
    }
    return false;
}

/*
用于校验assemble节点的输入输出是否存在冲突
注意，若输入由OP_INDEX_OUTCAST构造，则不会出现冲突
*/
bool ReplaceTensor::CheckIndexProducer(const Operation& op)
{
    for (const auto& producer : op.ProducerOps()) {
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            return true;
        }
    }
    return false;
}

bool ReplaceTensor::CheckAssembleConflict(const Operation& op)
{
    if (!CheckIndexProducer(op) && CheckAddrConflict(op)) {
        return true;
    }
    return false;
}

/*
用于校验index_outcast节点的输入输出是否存在冲突
注意，若输出后接assemble节点，则不会出现冲突
*/
bool ReplaceTensor::CheckIndexOutcastConflict(const Operation& op, Function& function)
{
    int index = 2;
    auto indexIn = op.GetInputOperand(index);
    auto indexOut = op.GetOOperands().front();
    if (forwardOps.find(op.GetOpMagic()) != forwardOps.end()) {
        if (function.IsFromInCast(indexIn) && function.IsFromOutCast(indexOut)) {
            return false;
        }
    }
    if (backwardOps.find(op.GetOpMagic()) != backwardOps.end()) {
        if (indexIn->GetRawMagic() != indexOut->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

/*
用于校验reshape节点的输入输出是否存在冲突
需要校验的场景：
    shape输入输出的rawtensor除了首轴之外都一致
*/
bool ReplaceTensor::CheckReshapeConflict(const Operation& op, Function& function)
{
    if (op.GetBoolAttribute(OP_ATTR_PREFIX + "isInplace"))
        return false;
    if (forwardOps.find(op.GetOpMagic()) != forwardOps.end()) {
        auto tensorOut = op.GetOOperands().front();
        if (function.IsFromOutCast(tensorOut)) {
            return false;
        }
    }
    if (backwardOps.find(op.GetOpMagic()) != backwardOps.end()) {
        if (CheckAddrConflict(op)) {
            return true;
        }
    }
    return false;
}

/*
用于校验a_mulacc_b节点的输入输出是否存在冲突
*/
bool ReplaceTensor::CheckAMulAccBConflict(const Operation& op)
{
    int index = 2;
    auto tensorIn = op.GetInputOperand(index);
    auto tensorOut = op.GetOOperands().front();
    auto& inOp = *tensorIn->GetProducers().begin();
    auto& outOp = *tensorOut->GetConsumers().begin();
    if (inOp == nullptr && outOp == nullptr) {
        return false;
    }
    if (tensorIn->GetRawMagic() != tensorOut->GetRawMagic()) {
        return true;
    }
    return false;
}

Status ReplaceTensor::InplaceCheck(Function& function)
{
    struct OpValidator {
        std::function<bool(const Operation&)> validate;
        std::function<bool(const Operation&, Function&)> validateWithFunc;
        std::function<bool(size_t)> inputCountValidator;
        std::function<bool(size_t)> outputCountValidator;
    };

    std::unordered_map<Opcode, OpValidator> opValidators = {
        {Opcode::OP_VIEW,
         {[this](const Operation& op) { return this->CheckAddrConflict(op); }, nullptr,
          [](size_t inputCount) { return inputCount == OperandCount::VIEW_INPUT; },
          [](size_t outputCount) { return outputCount == OperandCount::VIEW_OUTPUT; }}},
        {Opcode::OP_ASSEMBLE,
         {[this](const Operation& op) { return this->CheckAssembleConflict(op); }, nullptr,
          [](size_t inputCount) { return inputCount == OperandCount::ASSEMBLE_INPUT; },
          [](size_t outputCount) { return outputCount == OperandCount::ASSEMBLE_OUTPUT; }}},
        {Opcode::OP_INDEX_OUTCAST,
         {nullptr, [this](const Operation& op, Function& func) { return this->CheckIndexOutcastConflict(op, func); },
          [](size_t inputCount) { return inputCount == OperandCount::INDEX_OUTCAST_INPUTS; },
          [](size_t outputCount) { return outputCount == OperandCount::INDEX_OUTCAST_OUTPUT; }}},
        {Opcode::OP_RESHAPE,
         {nullptr, [this](const Operation& op, Function& func) { return this->CheckReshapeConflict(op, func); },
          [](size_t inputCount) { return inputCount == OperandCount::RESHAPE_INPUT; },
          [](size_t outputCount) { return outputCount == OperandCount::RESHAPE_OUTPUT; }}},
        {Opcode::OP_A_MULACC_B,
         {[this](const Operation& op) { return this->CheckAMulAccBConflict(op); }, nullptr,
          [](size_t inputCount) {
              size_t maxInputs = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 ?
                                     OperandCount::A_MULACC_B_MAX_INPUTS_A5 :
                                     OperandCount::A_MULACC_B_MAX_INPUTS;
              return inputCount >= OperandCount::A_MULACC_B_MIN_INPUTS && inputCount <= maxInputs;
          },
          [](size_t outputCount) { return outputCount == OperandCount::A_MULACC_B_OUTPUT; }}},
    };

    for (const auto& op : function.Operations()) {
        auto it = opValidators.find(op.GetOpcode());
        if (it == opValidators.end())
            continue;
        const auto& validator = it->second;
        if (!validator.inputCountValidator(op.GetInputOperandSize()) ||
            !validator.outputCountValidator(op.GetOutputOperandSize()) ||
            (validator.validate && validator.validate(op)) ||
            (validator.validateWithFunc && validator.validateWithFunc(op, function))) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s op[%d] invalid or conflict.", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

bool ReplaceTensor::CheckInplace(const Operation& op)
{
    if (inplaceOpSet.find(op.GetOpcode()) != inplaceOpSet.end()) {
        return true;
    }
    return false;
}

bool ReplaceTensor::HasSameConsecutive(Operation& op)
{
    for (auto& nextOp : op.ConsumerOps()) {
        if (nextOp->GetOpcode() == op.GetOpcode()) {
            return true;
        }
    }
    return false;
}
Status ReplaceTensor::PreCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for ReplaceTensor.");
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_F(Elements::Function,
                          "Loopcheck failed before PreGraph; Please check whether there is a loop.");
        return FAILED;
    }
    for (auto& op : function.Operations()) {
        if (op.GetSubgraphID() == NOT_IN_SUBGRAPH) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] is not partitioned; Please check subGraphIDs. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if ((op.GetOpcode() != Opcode::OP_ASSEMBLE) && (op.GetOpcode() != Opcode::OP_VIEW) &&
            (op.GetOpcode() != Opcode::OP_RESHAPE)) {
            continue;
        }
        if (HasSameConsecutive(op)) {
            APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] has the same Opcode child op; Please check child ops. %s",
                              op.GetOpcodeStr().c_str(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        auto tensorIn = op.GetIOperands().front();
        auto tensorOut = op.GetOOperands().front();
        if (tensorIn->GetMemoryTypeOriginal() != tensorOut->GetMemoryTypeOriginal()) {
            APASS_LOG_ERROR_F(
                Elements::Tensor,
                "unmatched input output memory type for reshape opmagic: %d, input mem type: %s, output mem type: %s; "
                "Please check the input and output.",
                op.opmagic, MemoryTypeToString(tensorIn->GetMemoryTypeOriginal()).c_str(),
                MemoryTypeToString(tensorOut->GetMemoryTypeOriginal()).c_str());
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for ReplaceTensor success.");
    return SUCCESS;
}

Status ReplaceTensor::PostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor.");
    if (InplaceCheck(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InplaceCheck failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor success.");
    return SUCCESS;
}

void ReplaceTensor::BuildRawMagicTensorIndex(Function& function)
{
    rawMagicTensors_.clear();
    // 与 GetTensorsByRawMagic 保持一致，按原始 rawmagic 分组；GetTensorsGroupedByRawMagic 使用有效值。
    for (const auto& tensor : GraphUtils::GetAllTensors(function)) {
        if (tensor->tensor != nullptr) {
            rawMagicTensors_[tensor->tensor->rawmagic].insert(tensor);
        }
    }
}

void ReplaceTensor::SetRawTensor(const LogicalTensorPtr& tensor, const std::shared_ptr<RawTensor>& rawTensor)
{
    if (tensor == nullptr) {
        return;
    }
    if (tensor->tensor == nullptr || rawTensor == nullptr || tensor->tensor->rawmagic != rawTensor->rawmagic) {
        if (tensor->tensor != nullptr) {
            rawMagicTensors_[tensor->tensor->rawmagic].erase(tensor);
        }
        if (rawTensor != nullptr) {
            rawMagicTensors_[rawTensor->rawmagic].insert(tensor);
        }
    }
    tensor->tensor = rawTensor;
}

void ReplaceTensor::UniteTensor(Function& function, UnionFind& uf)
{
    for (const auto& op : function.Operations()) {
        if (CheckInplace(op)) {
            if (inplaceOpMap.find(op.GetOpcode()) != inplaceOpMap.end()) {
                for (const auto& pair : inplaceOpMap.at(op.GetOpcode())) {
                    uf.Unite(op.GetInputOperand(pair.first), op.GetOutputOperand(pair.second));
                    APASS_LOG_INFO_F(Elements::Operation, "Unite %s op[%d] iOperand %d and oOperand %d.",
                                     op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands()[0]->GetMagic(),
                                     op.GetOOperands()[0]->GetMagic());
                }
            } else {
                uf.Unite(op.GetIOperands().front(), op.GetOOperands().front());
                APASS_LOG_INFO_F(Elements::Operation, "Unite %s op[%d] iOperand %d and oOperand %d.",
                                 op.GetOpcodeStr().c_str(), op.GetOpMagic(), op.GetIOperands()[0]->GetMagic(),
                                 op.GetOOperands()[0]->GetMagic());
            }
        }
        // 新图表达下, 多个assemble写入不同logical tensor但共享同一rawMagic(同一地址)。
        // 需要将同rawMagic的兄弟logical tensor也union到同一组, 保证FindBaseTensor统一选base。
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && op.GetOOperands().front()->tensor != nullptr) {
            auto assembleOut = op.GetOOperands().front();
            for (const auto& sibling : rawMagicTensors_[assembleOut->tensor->rawmagic]) {
                if (sibling != nullptr && sibling->GetMagic() != assembleOut->GetMagic()) {
                    uf.Unite(assembleOut, sibling);
                }
            }
        }
        if (op.HasAttribute(OpAttributeKey::inplaceIdx)) {
            uf.Unite(op.GetIOperands()[op.GetIntAttribute(OpAttributeKey::inplaceIdx)], op.GetOOperands().front());
        }
    }
}

Status ReplaceTensor::FindBaseTensor(Function& function,
                                     const std::unordered_map<LogicalTensorPtr, int>& tensorToOrderIndex,
                                     LogicalTensors& group, LogicalTensorPtr& baseTensor)
{
    for (const auto& curTensor : group) {
        if (function.IsFromInCast(curTensor) || function.IsFromOutCast(curTensor)) {
            if (baseTensor == nullptr) {
                baseTensor = curTensor;
                APASS_LOG_INFO_F(Elements::Tensor, "Set base Tensor %d", curTensor->GetMagic());
            } else if (baseTensor->Symbol() != curTensor->Symbol() &&
                       baseTensor->GetRawTensor()->memoryId != curTensor->GetRawTensor()->memoryId &&
                       baseTensor->tensor->actualRawmagic != curTensor->tensor->actualRawmagic) {
                APASS_LOG_ERROR_F(Elements::Tensor, "baseTensor %d and curTensor %d has conflict.",
                                  baseTensor->GetMagic(), curTensor->GetMagic());
                return FAILED;
            } else if (function.IsFromInCast(curTensor)) {
                baseTensor = curTensor;
                APASS_LOG_INFO_F(Elements::Tensor, "Set base Tensor %d", curTensor->GetMagic());
            }
        }
    }
    if (baseTensor != nullptr) {
        return SUCCESS;
    }
    LogicalTensors boundTensors;
    for (auto& curTensor : group) {
        if (!isBoundTensor(curTensor)) {
            continue;
        }
        boundTensors.push_back(curTensor);
        AddViewInputsToBoundTensors(curTensor, boundTensors);
    }
    LogicalTensors baseGroup = boundTensors.empty() ? group : boundTensors;
    baseTensor = baseGroup.front();
    int64_t baseShape = abs(baseTensor->tensor->GetRawDataSize());
    for (auto& curTensor : baseGroup) {
        int64_t curShape = abs(curTensor->tensor->GetRawDataSize());
        if (curShape > baseShape) {
            APASS_LOG_INFO_F(Elements::Tensor, "Replace curTensor %d size %ld to baseTensor %d size %ld.",
                             curTensor->GetMagic(), curShape, baseTensor->GetMagic(), baseShape);
            baseTensor = curTensor;
            baseShape = curShape;
        } else if (curShape == baseShape && tensorToOrderIndex.at(curTensor) < tensorToOrderIndex.at(baseTensor)) {
            APASS_LOG_INFO_F(Elements::Tensor, "Replace curTensor %d idx %d to baseTensor %d idx %d.",
                             curTensor->GetMagic(), tensorToOrderIndex.at(curTensor), baseTensor->GetMagic(),
                             tensorToOrderIndex.at(baseTensor));
            baseTensor = curTensor;
        }
    }
    return SUCCESS;
}

void ReplaceTensor::AddViewInputsToBoundTensors(const LogicalTensorPtr& curTensor, LogicalTensors& boundTensors) const
{
    auto consumers = curTensor->GetConsumers();
    bool hasCopyInConsumer = std::any_of(consumers.begin(), consumers.end(), [](const Operation* consumer) {
        return consumer != nullptr && IsCopyIn(consumer->GetOpcode());
    });
    if (!hasCopyInConsumer) {
        return;
    }
    for (auto* producer : curTensor->GetProducers()) {
        if (producer == nullptr || producer->GetOpcode() != Opcode::OP_VIEW || producer->GetIOperands().empty()) {
            continue;
        }
        auto viewInput = producer->GetIOperands().front();
        if (viewInput != nullptr &&
            std::find(boundTensors.begin(), boundTensors.end(), viewInput) == boundTensors.end()) {
            boundTensors.push_back(viewInput);
        }
    }
}

Status ReplaceTensor::ForwardView(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    if (ForUpdateView(op) == FAILED) {
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    SetRawTensor(op->GetOOperands()[0], rootTensor->tensor);
    forwardOps.insert(op->GetOpMagic());
    forRoots.push(op->GetOOperands()[0]);
    if (!function.IsFromOutCast(op->GetOOperands()[0])) {
        function.UpdateLinkMap(op->GetOOperands()[0], op->GetIOperands()[0]);
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardReshape(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    processedOp.insert(op->GetOpMagic());
    (void)function;
    if (function.IsFromOutCast(op->GetOOperands()[0])) {
        APASS_LOG_INFO_F(Elements::Operation, "OP_RESHAPE %d oOperand is OutCast, Skip inplace.", op->GetOpMagic());
        return SUCCESS;
    }
    op->GetOOperands()[0]->tensor->actualRawmagic = rootTensor->GetRawMagic();
    forwardOps.insert(op->GetOpMagic());
    forRoots.push(op->GetOOperands()[0]);
    // 新图表达下, reshape的输入可能是assemble输出(共享rawMagic), 需同步兄弟并继续遍历
    SyncSiblingAssembleOutput(function, op->GetIOperands()[0]);
    return SUCCESS;
}

Status ReplaceTensor::ForwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto reusePairs = inplaceOpMap.at(op->GetOpcode());
    for (const auto& reusePair : reusePairs) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        auto tensorIn = op->GetIOperands()[inputIdx];
        auto tensorOut = op->GetOOperands()[outputIdx];
        if (tensorIn != rootTensor) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(),
                             rootTensor->GetMagic());
            return SUCCESS;
        }
        processedOp.insert(op->GetOpMagic());
        if (function.IsFromInCast(tensorIn) && function.IsFromOutCast(tensorOut)) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is incast, tensorOut %d is outcast.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(), tensorOut->GetMagic());
            return SUCCESS;
        }
        SetRawTensor(tensorOut, tensorIn->tensor);
        forwardOps.insert(op->GetOpMagic());
        forRoots.push(tensorOut);
        tensorOut->UpdateOffset(tensorIn->GetOffset());
    }
    return SUCCESS;
}

Status ReplaceTensor::ForwardViewType(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto viewTypeIn = op->GetIOperands()[0];
    auto viewTypeOut = op->GetOOperands()[0];
    if (viewTypeIn != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_VIEW_TYPE %d rootTensor %d is not same as viewTypeIn %d.",
                          op->GetOpMagic(), rootTensor->GetMagic(), viewTypeIn->GetMagic());
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    viewTypeOut->tensor->actualRawmagic = viewTypeIn->GetRawMagic();
    forwardOps.insert(op->GetOpMagic());
    if (AdjustOffsetAndRawShape(viewTypeIn, viewTypeOut) == FAILED) {
        return FAILED;
    }
    forRoots.push(viewTypeOut);
    return SUCCESS;
}

bool isInplaceAssemble(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    for (auto inOp : assembleIn->GetProducers()) {
        if (inplaceOpSet.find(inOp->GetOpcode()) != inplaceOpSet.end()) {
            return true;
        }
    }
    return false;
}

bool isMultiAssemble(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    for (auto outOp : assembleIn->GetConsumers()) {
        if (outOp->GetOpMagic() != op->GetOpMagic() && outOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            return true;
        }
    }
    return false;
}

Status ReplaceTensor::ForwardAssemble(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto assembleIn = op->GetIOperands()[0];
    auto assembleOut = op->GetOOperands()[0];
    if (assembleIn != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_ASSEMBLE %d rootTensor %d is not same as viewTypeIn %d.",
                          op->GetOpMagic(), rootTensor->GetMagic(), assembleIn->GetMagic());
        return FAILED;
    }
    if (isInplaceAssemble(op) || isMultiAssemble(op)) {
        auto& inOp = *(assembleIn)->GetProducers().begin();
        processedOp.insert(op->GetOpMagic());
        forRoots.push(assembleOut);
        if (inOp != nullptr && inOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            APASS_LOG_INFO_F(Elements::Operation,
                             "OP_ASSEMBLE %d parentOp is OP_INDEX_OUTCAST %d, skip replace tensor.", op->GetOpMagic(),
                             inOp->GetOpMagic());
            return SUCCESS;
        }
        // 保存原始 rawMagic, 因为 tensor 指针替换后 rawmagic 会变, 需按原始值查找兄弟
        int origRawMagic = assembleOut->tensor->rawmagic;
        SetRawTensor(assembleOut, assembleIn->tensor);
        forwardOps.insert(op->GetOpMagic());
        // 新图表达下, Assemble的输出是assemble输出(共享rawMagic), 需同步兄弟并继续遍历
        SyncSiblingAssembleOutput(function, assembleOut, origRawMagic);
        return SUCCESS;
    } else {
        forRoots.push(assembleOut);
        backRoots.push(assembleOut);
        return SUCCESS;
    }
}

Status ReplaceTensor::ForwardCopyOut(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto index = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
    auto inTensor = op->GetIOperands()[index];
    auto outTensor = op->GetOOperands().front();
    if (inTensor != rootTensor) {
        APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), rootTensor->GetMagic());
        return SUCCESS;
    }
    processedOp.insert(op->GetOpMagic());
    if (function.IsFromInCast(inTensor) && function.IsFromOutCast(outTensor)) {
        APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] input tensor %d is Incast, output tensor %d is OutCast",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), outTensor->GetMagic());
        return SUCCESS;
    }
    if (!function.IsFromOutCast(outTensor)) {
        function.UpdateLinkMap(outTensor, inTensor);
    }
    SetRawTensor(outTensor, rootTensor->tensor);
    outTensor->UpdateOffset(rootTensor->GetOffset());
    forRoots.push(outTensor);
    return SUCCESS;
}

Status ReplaceTensor::ForwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto index = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
    auto inTensor = op->GetIOperands()[index];
    auto outTensor = op->GetOOperands().front();
    if (inTensor != rootTensor) {
        APASS_LOG_INFO_F(Elements::Operation, "op %s[%d] tensorIn %d is not same as rootTensor %d.",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), rootTensor->GetMagic());
        return SUCCESS;
    }
    processedOp.insert(op->GetOpMagic());
    if (!function.IsFromOutCast(outTensor)) {
        function.UpdateLinkMap(outTensor, inTensor);
    }
    SetRawTensor(outTensor, rootTensor->tensor);
    const bool isScatterInplace = op->GetOpcode() == Opcode::OP_SCATTER_INPLACE ||
                                  op->GetOpcode() == Opcode::OP_SCATTER_ELEMENT_INPLACE;
    const auto oldOffset = outTensor->GetOffset();
    const auto newOffset = rootTensor->GetOffset();
    if (isScatterInplace) {
        outTensor->UpdateOffset(rootTensor->GetTensorOffset());
    } else {
        auto dynOffset = outTensor->GetDynOffset();
        for (size_t i = 0; i < dynOffset.size(); ++i) {
            dynOffset[i] = dynOffset[i] + newOffset[i] - oldOffset[i];
        }
        outTensor->UpdateOffset(TensorOffset(newOffset, dynOffset));
    }
    for (auto* consumer : outTensor->GetConsumers()) {
        if (!IsCopyIn(consumer->GetOpcode())) {
            continue;
        }
        if (UpdateCopyInAttr(consumer, isScatterInplace || oldOffset != newOffset) == FAILED) {
            APASS_LOG_ERROR_F(Elements::Operation, "Update copyIn[%d] attr failed.", consumer->GetOpMagic());
            return FAILED;
        }
    }
    forRoots.push(outTensor);
    return SUCCESS;
}

Status ReplaceTensor::BackwardReshape(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    processedOp.insert(op->GetOpMagic());
    op->GetIOperands()[0]->tensor->actualRawmagic = rootTensor->GetRawMagic();
    backwardOps.insert(op->GetOpMagic());
    backRoots.push(op->GetIOperands()[0]);
    // 新图表达下, reshape的输入可能是assemble输出(共享rawMagic), 需同步兄弟并继续遍历
    SyncSiblingAssembleOutput(function, op->GetIOperands()[0]);
    return SUCCESS;
}

Status ReplaceTensor::BackwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto reusePairs = inplaceOpMap.at(op->GetOpcode());
    for (const auto& reusePair : reusePairs) {
        auto inputIdx = reusePair.first;
        auto outputIdx = reusePair.second;
        auto tensorIn = op->GetIOperands()[inputIdx];
        auto tensorOut = op->GetOOperands()[outputIdx];
        if (tensorOut != rootTensor) {
            APASS_LOG_INFO_F(Elements::Operation, "OP %s[%d] tensorIn %d is not same as rootTensor %d.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), tensorIn->GetMagic(),
                             rootTensor->GetMagic());
            return SUCCESS;
        }
        processedOp.insert(op->GetOpMagic());
        // 保存原始 rawMagic, 因为 tensor 指针替换后 rawmagic 会变, 需按原始值查找兄弟
        int origRawMagic = tensorIn->tensor->rawmagic;
        SetRawTensor(tensorIn, tensorOut->tensor);
        backwardOps.insert(op->GetOpMagic());
        backRoots.push(tensorIn);
        tensorOut->UpdateOffset(tensorIn->GetOffset());
        // 新图表达下, inplace op的输入可能是assemble输出(共享rawMagic), 需同步兄弟并继续遍历
        SyncSiblingAssembleOutput(function, tensorIn, origRawMagic);
    }
    return SUCCESS;
}

Status ReplaceTensor::BackwardView(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto viewIn = op->GetIOperands()[0];
    auto viewOut = op->GetOOperands()[0];
    (void)rootTensor;
    processedOp.insert(op->GetOpMagic());
    backRoots.push(viewIn);
    // 保存原始 rawMagic, 因为 tensor 指针替换后 rawmagic 会变, 需按原始值查找兄弟
    int origRawMagic = viewIn->tensor->rawmagic;
    SetRawTensor(viewIn, viewOut->tensor);
    backwardOps.insert(op->GetOpMagic());
    // 新图表达下, view的输入可能是assemble输出(共享rawMagic), 需同步兄弟并继续遍历
    SyncSiblingAssembleOutput(function, viewIn, origRawMagic);
    return SUCCESS;
}

Status ReplaceTensor::BackwardViewType(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto viewTypeIn = op->GetIOperands()[0];
    auto viewTypeOut = op->GetOOperands()[0];
    if (viewTypeOut != rootTensor) {
        APASS_LOG_ERROR_F(Elements::Operation, "OP_VIEW_TYPE %d rootTensor %d is not same as viewTypeOut %d.",
                          op->GetOpMagic(), rootTensor->GetMagic(), viewTypeOut->GetMagic());
        return FAILED;
    }
    processedOp.insert(op->GetOpMagic());
    viewTypeIn->tensor->actualRawmagic = viewTypeOut->GetRawMagic();
    backwardOps.insert(op->GetOpMagic());
    if (AdjustOffsetAndRawShape(viewTypeOut, viewTypeIn) == FAILED) {
        return FAILED;
    }
    backRoots.push(viewTypeIn);
    return SUCCESS;
}

Status ReplaceTensor::BackwardAssemble(Operation* op, LogicalTensorPtr& rootTensor)
{
    auto& inOp = *(op->GetIOperands()[0])->GetProducers().begin();
    backRoots.push(op->GetIOperands()[0]);
    processedOp.insert(op->GetOpMagic());
    if (inOp != nullptr && inOp->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
        APASS_LOG_INFO_F(Elements::Operation, "OP_ASSEMBLE %d parent op is OP_INDEX_OUTCAST %d, skip inplace.",
                         op->GetOpMagic(), inOp->GetOpMagic());
        return SUCCESS;
    }
    if (BackUpdateAssemble(op) == FAILED) {
        return FAILED;
    }
    SetRawTensor(op->GetIOperands()[0], rootTensor->tensor);
    if (FoldL0C2UBCopyOffset(op) == FAILED) {
        return FAILED;
    }
    backwardOps.insert(op->GetOpMagic());
    if (op->GetIOperands()[0]->GetConsumers().size() > 1) {
        forRoots.push(op->GetIOperands()[0]);
        for (auto& consumer : op->GetIOperands()[0]->GetConsumers()) {
            if (IsCopyIn(consumer->GetOpcode())) {
                if (UpdateCopyInAttr(consumer) == FAILED) {
                    APASS_LOG_ERROR_F(Elements::Operation, "Update copyIn[%d] attr failed.", consumer->GetOpMagic());
                    return FAILED;
                }
            }
        }
    }
    return SUCCESS;
}

Status ReplaceTensor::BackwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function)
{
    auto index = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
    auto inTensor = op->GetIOperands()[index];
    auto outTensor = op->GetOOperands().front();
    if (outTensor != rootTensor) {
        APASS_LOG_INFO_F(Elements::Operation, "op %s[%d] tensorIn %d is not same as rootTensor %d.",
                         op->GetOpcodeStr().c_str(), op->GetOpMagic(), inTensor->GetMagic(), rootTensor->GetMagic());
        return SUCCESS;
    }
    processedOp.insert(op->GetOpMagic());
    if (!function.IsFromOutCast(outTensor)) {
        function.UpdateLinkMap(outTensor, inTensor);
    }
    SetRawTensor(inTensor, rootTensor->tensor);
    inTensor->UpdateOffset(rootTensor->GetOffset());
    forRoots.push(inTensor);
    return SUCCESS;
}

Status ReplaceTensor::ForwardProcess(Function& function)
{
    while (!forRoots.empty()) {
        auto rootTensor = forRoots.front();
        forRoots.pop();
        // 防御性快照：现查 Forward* 各 helper（SetRawTensor/UpdateLinkMap/UpdateOffset/入队）
        // 均不改写 rootTensor 的 consumer 集合，无已知失效路径，此处防患未然。
        // 快照语义：仅保证遍历入口时集合内的元素被完整处理；循环中新加入集合的 consumer
        // 本轮不被访问（现状下无 helper 在遍历中加 consumer，行为与裸遍历一致）
        auto consumersSnapshot = rootTensor->GetConsumers();
        for (auto& consumerOp : consumersSnapshot) {
            if (processedOp.find(consumerOp->GetOpMagic()) != processedOp.end()) {
                continue;
            }
            if (consumerOp->GetOpcode() == Opcode::OP_VIEW) {
                if (ForwardView(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (ForwardAssemble(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                if (ForwardReshape(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
                if (ForwardViewType(consumerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (inplaceOpMap.find(consumerOp->GetOpcode()) != inplaceOpMap.end()) {
                if (ForwardInplaceOp(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (consumerOp->GetOpcode() == Opcode::OP_COPY_OUT &&
                       consumerOp->HasAttribute(OpAttributeKey::inplaceIdx)) {
                if (ForwardCopyOut(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (IsInplaceIdxOperation(*consumerOp)) {
                if (ForwardInputIdx(consumerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else {
                continue;
            }
        }
    }
    return SUCCESS;
}

void ReplaceTensor::SyncSiblingAssembleOutput(Function&, const LogicalTensorPtr& tensor, int origRawMagic)
{
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return;
    }
    int rawMagic = (origRawMagic >= 0) ? origRawMagic : tensor->tensor->rawmagic;
    // 只处理 assemble 输出: 检查同 rawMagic 的兄弟 logical tensor
    auto it = rawMagicTensors_.find(rawMagic);
    if (it == rawMagicTensors_.end()) {
        return;
    }
    // Updating a sibling can move it to another rawMagic group.
    const auto siblings = it->second;
    for (const auto& sibling : siblings) {
        if (sibling == nullptr || sibling->GetMagic() == tensor->GetMagic()) {
            continue;
        }
        // 同步 tensor 指针(与当前 tensor 保持一致)
        SetRawTensor(sibling, tensor->tensor);
        // push 到 backRoots 和 forRoots, 使兄弟的 producer/consumer 链路继续被遍历
        backRoots.push(sibling);
        forRoots.push(sibling);
    }
}

Status ReplaceTensor::BackwardProcess(Function& function)
{
    while (!backRoots.empty()) {
        auto rootTensor = backRoots.front();
        backRoots.pop();
        // 遍历中 BackwardAssemble→FoldL0C2UBCopyOffset 会经 UpdateOperandBeforeRemoveOp
        // 的 RemoveProducer 删除集合中当前遍历元素，先做快照避免迭代器失效导致的 segfault。
        // 快照语义：仅保证遍历入口时集合内的元素被完整处理；循环中新加入集合的 producer
        // 本轮不被访问（现状下新增者为 L0C_COPY_UB，不在 Backward 处理范围）
        auto producersSnapshot = rootTensor->GetProducers();
        for (auto& producerOp : producersSnapshot) {
            if (processedOp.find(producerOp->GetOpMagic()) != processedOp.end()) {
                continue;
            }
            if (producerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (BackwardAssemble(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_VIEW) {
                if (BackwardView(producerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_RESHAPE) {
                if (BackwardReshape(producerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (inplaceOpMap.find(producerOp->GetOpcode()) != inplaceOpMap.end()) {
                if (BackwardInplaceOp(producerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else if (producerOp->GetOpcode() == Opcode::OP_VIEW_TYPE) {
                if (BackwardViewType(producerOp, rootTensor) == FAILED) {
                    return FAILED;
                }
            } else if (IsInplaceIdxOperation(*producerOp)) {
                if (BackwardInputIdx(producerOp, rootTensor, function) == FAILED) {
                    return FAILED;
                }
            } else {
                continue;
            }
        }
    }
    return SUCCESS;
}

LogicalTensorPtr ReplaceTensor::FindReplaceSource(Function& function, Operation& op,
                                                  std::unordered_map<Operation*, LogicalTensorPtr>& visited)
{
    if (visited.count(&op) > 0) {
        return visited.at(&op);
    }
    auto inplaceIdx = op.GetIntAttribute(OpAttributeKey::inplaceIdx);
    ASSERT(OperationErr::OP_INVALID_OPERAND_COUNT,
           inplaceIdx >= 0 && inplaceIdx < static_cast<int>(op.GetIOperands().size()));
    auto inplaceIOperand = op.GetInputOperand(inplaceIdx);
    LogicalTensorPtr res = nullptr;
    for (auto producer : inplaceIOperand->GetProducers()) {
        if (!producer->HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        auto tmp = FindReplaceSource(function, *producer, visited);
        if (res == nullptr) {
            res = tmp;
        } else {
            ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, res == tmp); // inplace路径应总是交汇于同一起点
        }
    }
    if (res == nullptr) {
        res = inplaceIOperand; // 向前没有inplace了，自己就是起点
    }
    visited.emplace(&op, res);
    return res;
}

Status ReplaceTensor::RefactorViewConnectForReplace(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start RefactorViewConnectForInplace.");
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (op.GetInputOperand(0)->GetRawTensor() == op.GetOutputOperand(0)->GetRawTensor()) {
            op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        }
    }
    std::unordered_map<Operation*, LogicalTensorPtr> visited;
    for (Operation& op : function.Operations()) {
        if (!op.HasAttribute(OpAttributeKey::inplaceIdx) || visited.count(&op) > 0) {
            continue;
        }
        FindReplaceSource(function, op, visited);
    }

    for (auto& [op, srcTensor] : visited) {
        if (op->GetOpcode() != Opcode::OP_VIEW) { // 仅重构View连接
            continue;
        }
        auto inplaceIdx = op->GetIntAttribute(OpAttributeKey::inplaceIdx);
        ASSERT(OperationErr::OP_SPECIAL_CONSTRAINT, inplaceIdx == 0);
        auto iOperand = op->GetInputOperand(inplaceIdx);
        auto oOperand = op->GetOutputOperand(0);
        if (iOperand == srcTensor) { // 开头的VIEW不需要插入NOP来控制顺序
            continue;
        }
        bool hasNonInplaceConsumer = false;
        for (auto consumer : oOperand->GetConsumers()) {
            if (consumer->GetOpcode() != Opcode::OP_NOP && !consumer->HasAttribute(OpAttributeKey::inplaceIdx)) {
                hasNonInplaceConsumer = true;
                break;
            }
        }
        if (hasNonInplaceConsumer) {
            continue;
        }
        ASSERT(TensorErr::TENSOR_SHAPE_MISMATCH, iOperand->GetRawTensor() == srcTensor->GetRawTensor());
        ASSERT(TensorErr::TENSOR_SHAPE_MISMATCH, oOperand->GetRawTensor() == srcTensor->GetRawTensor());
        op->ReplaceIOperand(0, srcTensor);
        auto nopOutput = irBuilder_.CreateTensorVar(srcTensor->GetRawTensor(), Offset(srcTensor->GetOffset().size()),
                                                    srcTensor->GetShape(), std::vector<SymbolicScalar>{});
        nopOutput->SetMemoryTypeBoth(oOperand->GetMemoryTypeOriginal());
        auto& nop = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_NOP, {iOperand, oOperand}, {nopOutput});
        nop.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        nop.UpdateSubgraphID(op->GetSubgraphID());
        auto consumers = oOperand->GetConsumers();
        for (auto consumer : consumers) {
            if (consumer->GetOpcode() == Opcode::OP_NOP || !consumer->HasAttribute(OpAttributeKey::inplaceIdx)) {
                continue;
            }
            consumer->ReplaceIOperand(consumer->GetIntAttribute(OpAttributeKey::inplaceIdx), nopOutput);
        }
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End RefactorViewConnectForInplace.");
    return SUCCESS;
}

void ReplaceTensor::ProcessHubAssembleOp(Function& function, Operation& hubOp, Operation& assembleOp,
                                         std::shared_ptr<LogicalTensor> hubInput,
                                         std::shared_ptr<LogicalTensor> hubOutput)
{
    auto assembleInput = assembleOp.GetIOperands()[0];
    auto assembleOutput = assembleOp.GetOOperands()[0];
    if (assembleInput.get() != hubOutput.get()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Assemble input[%d] is not HUB output[%d], chain may be broken",
                         assembleInput->GetMagic(), hubOutput->GetMagic());
        return;
    }
    bool isExactOutcast = false;
    auto outcasts = function.GetOutcast();
    for (auto& outcast : outcasts) {
        if (outcast.get() == assembleOutput.get()) {
            isExactOutcast = true;
            break;
        }
    }
    if (!isExactOutcast) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Assemble[%d] output is not exact outcast, skip HUB memory reuse processing.",
                         assembleOp.GetOpMagic());
        return;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Found exact HUB-ASSEMBLE-OUTCAST chain: HUB[%d] -> ASSEMBLE[%d] -> OUTCAST[%d]",
                     hubOp.GetOpMagic(), assembleOp.GetOpMagic(), assembleOutput->GetMagic());
    auto hubInputMemType = hubInput->GetMemoryTypeOriginal();
    auto hubOutputMemType = hubOutput->GetMemoryTypeOriginal();
    auto assembleOutputMemType = assembleOutput->GetMemoryTypeOriginal();
    if (hubInputMemType != hubOutputMemType || hubInputMemType != assembleOutputMemType) {
        APASS_LOG_WARN_F(Elements::Tensor, "Memory type mismatch: HUB input=%d, HUB output=%d, ASSEMBLE output=%d",
                         hubInputMemType, hubOutputMemType, assembleOutputMemType);
        return;
    }
    // RunOnFunction 在处理 HUB 前已清空 rawMagicTensors_。
    hubInput->tensor = assembleOutput->tensor;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "HUB assemble op %d attribute is nullptr, skip HUB memory reuse.",
                         assembleOp.GetOpMagic());
        return;
    }
    hubInput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    hubOutput->tensor = assembleOutput->tensor;
    hubOutput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    APASS_LOG_INFO_F(Elements::Tensor, "Complete memory reuse established: all tensors share HUB input[%d] memory",
                     hubInput->GetMagic());
}

Status ReplaceTensor::ProcessHubOp(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_HUB) {
            continue;
        }
        auto hubInput = op.GetIOperands()[0];  // HUB 的输入 tensor
        auto hubOutput = op.GetOOperands()[0]; // HUB 的输出 tensor
        for (auto consumerOp : hubOutput->GetConsumers()) {
            if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                ProcessHubAssembleOp(function, op, *consumerOp, hubInput, hubOutput);
            }
        }
        for (auto producerOp : hubInput->GetProducers()) {
            if (!OpcodeManager::Inst().IsCopyOut(producerOp->GetOpcode())) {
                continue;
            }
            auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
            if (copyAttr == nullptr) {
                APASS_LOG_INFO_F(Elements::Operation, "Copy Op %d Attribute is nullptr.", producerOp->GetOpMagic());
                continue;
            }
            auto attrOffset = copyAttr->GetToOffset(); // OpImm
            auto tensorOffset = OpImmediate::Specified(hubInput->GetTensorOffset());
            std::vector<OpImmediate> newOffset;
            for (size_t i = 0; i < attrOffset.size(); i++) {
                newOffset.push_back(attrOffset[i] + tensorOffset[i]);
            }
            copyAttr->SetToOffset(newOffset);
            copyAttr->SetRawShape(OpImmediate::Specified(hubInput->tensor->GetDynRawShape()));
        }
    }
    return SUCCESS;
}

std::unordered_map<LogicalTensorPtr, int> ReplaceTensor::BuildTensorOrderIndexMap(Function& function)
{
    std::unordered_map<LogicalTensorPtr, int> tensorToOrderIndex;
    int index = 0;
    for (const auto& op : function.Operations()) {
        for (const auto& inTensor : op.GetIOperands()) {
            if (!tensorToOrderIndex.count(inTensor)) {
                tensorToOrderIndex[inTensor] = index++;
            }
        }
        for (const auto& outTensor : op.GetOOperands()) {
            if (!tensorToOrderIndex.count(outTensor)) {
                tensorToOrderIndex[outTensor] = index++;
            }
        }
    }
    return tensorToOrderIndex;
}

/**
 * @brief 为 UB 内存类型的输入插入拷贝序列 (UB → DDR → UB)
 */
Status ReplaceTensor::InsertCopyUBOp(Function& function, Operation* needInsertCopyAssOp, LogicalTensorPtr& input)
{
    auto copyShape = input->GetShape();
    auto copyRawShape = input->tensor->GetDynRawShape();
    auto copyDynShape = input->GetDynValidShape();
    Offset offset(copyShape.size(), 0);
    auto copyOutOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyOutOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto& copyOutOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_OUT, {input}, {copyOutOutputPtr},
        [&input, &offset, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                input->GetMemoryTypeOriginal(), OpImmediate::Specified(offset), OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    copyOutOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    auto copyInOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyInOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto& copyInOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_IN, {copyOutOutputPtr}, {copyInOutputPtr},
        [&offset, &input, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offset), input->GetMemoryTypeOriginal(), OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    copyInOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    needInsertCopyAssOp->ReplaceInput(copyInOutputPtr, input);
    return SUCCESS;
}

/**
 * @brief 为 DDR 内存类型的输入插入拷贝序列 (DDR → UB → DDR)
 */
Status ReplaceTensor::InsertCopyDDROp(Function& function, Operation* needInsertCopyAssOp, LogicalTensorPtr& input)
{
    auto copyShape = input->GetShape();
    auto copyRawShape = input->tensor->GetDynRawShape();
    auto copyDynShape = input->GetDynValidShape();
    Offset offset(copyShape.size(), 0);
    Offset inOffset = input->GetOffset();
    auto& inOp = *(input)->GetProducers().begin();
    bool viewOffsetAbsorbed = false;
    if (needInsertCopyAssOp->GetOpcode() == Opcode::OP_RESHAPE && inOp->GetOpcode() == Opcode::OP_VIEW) {
        auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(inOp->GetOpAttribute());
        inOffset = viewOpAttr->GetFrom();
        // codegen对DDR copyIn只使用fromOffset(绝对量), 烤入view窗口偏移是必需的;
        // 打标记防止后续pass(如PreGraph的view-copyin折叠)再次叠加view偏移
        viewOffsetAbsorbed = true;
    }
    auto copyInOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyInOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    const int UB_SIZE_THRESHOLD = static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB));
    auto memType = copyInOutputPtr->GetMemoryTypeOriginal();
    if ((memType == MemoryType::MEM_UB) && (copyInOutputPtr->GetDataSize() > UB_SIZE_THRESHOLD)) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "Tensor [%d] can not copy to UB, tensor size [%ld] exceeds the UB size [%d] limit.",
                          input->magic, input->GetDataSize(), UB_SIZE_THRESHOLD);
        return FAILED;
    }

    // 为copy到Ub的Tensor进行32B对齐
    AlignmentUtils::ProcessLastDim32BAlignedOnUB(copyInOutputPtr);
    auto& copyInOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_IN, {input}, {copyInOutputPtr},
        [&inOffset, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(inOffset), MemoryType::MEM_UB, OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    if (viewOffsetAbsorbed) {
        copyInOp.SetAttr(COPY_IN_VIEW_OFFSET_ABSORBED, true);
    }
    copyInOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    auto copyOutOutputPtr = irBuilder_.CreateTensorVar(input->Datatype(), copyShape, std::vector<SymbolicScalar>{});
    copyOutOutputPtr->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto& copyOutOp = PassOperationUtils::AddOperation(
        function, Opcode::OP_COPY_OUT, {copyInOutputPtr}, {copyOutOutputPtr},
        [&offset, &copyShape, &copyRawShape, &copyDynShape](Operation& op) {
            op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                MemoryType::MEM_UB, OpImmediate::Specified(offset), OpImmediate::Specified(copyShape),
                OpImmediate::Specified(copyRawShape), OpImmediate::Specified(copyDynShape)));
        });
    copyOutOp.UpdateSubgraphID(needInsertCopyAssOp->GetSubgraphID());

    needInsertCopyAssOp->ReplaceInput(copyOutOutputPtr, input);
    return SUCCESS;
}

/**
 * @brief 判断 ASSEMBLE 是否为就地修改 GM 入参的同址自写回（no-op，可跳过防御拷贝）
 */
bool ReplaceTensor::IsAssembleSameGmWriteBack(Function& function, Operation* assembleOp) const
{
    // step0: 仅处理单输入单输出 ASSEMBLE
    if (assembleOp == nullptr || assembleOp->GetIOperands().size() != 1 || assembleOp->GetOOperands().size() != 1) {
        return false;
    }

    auto assembleIn = assembleOp->GetIOperands()[0];
    auto assembleOut = assembleOp->GetOOperands()[0];
    auto inRaw = assembleIn->GetRawTensor();
    auto outRaw = assembleOut->GetRawTensor();

    APASS_LOG_INFO_F(Elements::Tensor,
                     "[ReplaceTensor] IsAssembleSameGmWriteBack ENTRY: ASSEMBLE[%d] numInput=%zu numOutput=%zu "
                     "numIncastLink=%zu",
                     assembleOp->GetOpMagic(), assembleOp->GetIOperands().size(), assembleOp->GetOOperands().size(),
                     function.outIncastLinkMap.size());

    bool inIsDdr = assembleIn->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;

    // step1: 输出必须是 linked-inplace outcast
    auto itOut = function.outIncastLinkMap.find(outRaw);
    if (itOut == function.outIncastLinkMap.end()) {
        return false;
    }
    auto incastRaw = itOut->second;

    // step2: 校验所有带 inplaceIdx 的 producer 的 self 都与 incastRaw 同源
    bool sameSource = false;
    std::shared_ptr<RawTensor> srcRaw = nullptr;
    std::string producerOpcode = "NONE";
    int producerInplaceIdx = -1;
    for (const auto& producerOp : assembleIn->GetProducers()) {
        if (!producerOp->HasAttribute(OpAttributeKey::inplaceIdx)) {
            continue;
        }
        const int idx = producerOp->GetIntAttribute(OpAttributeKey::inplaceIdx);
        if (idx < 0 || idx >= static_cast<int>(producerOp->GetIOperands().size())) {
            return false;
        }
        auto candRaw = producerOp->GetIOperands()[idx]->GetRawTensor();
        const bool rawShapeMatch = (candRaw->rawshape == incastRaw->rawshape);
        bool curMatch = (candRaw == incastRaw);
        if (!curMatch && rawShapeMatch && candRaw->memoryId != -1 && incastRaw->memoryId != -1) {
            curMatch = (candRaw->memoryId == incastRaw->memoryId);
        }
        if (!curMatch && rawShapeMatch && candRaw->actualRawmagic != -1 && incastRaw->actualRawmagic != -1) {
            curMatch = (candRaw->actualRawmagic == incastRaw->actualRawmagic);
        }
        if (!curMatch) {
            return false;
        }
        if (srcRaw == nullptr) {
            producerOpcode = producerOp->GetOpcodeStr();
            producerInplaceIdx = idx;
            srcRaw = candRaw;
        }
    }
    sameSource = (srcRaw != nullptr);

    APASS_LOG_INFO_F(
        Elements::Tensor,
        "[ReplaceTensor] IsAssembleSameGmWriteBack: ASSEMBLE[%d] "
        "in[magic=%d raw=%d memId=%d memType=%s size=%ld] "
        "out[magic=%d raw=%d memId=%d memType=%s size=%ld] "
        "inIsDdr=%d incastRaw=%d producer=%s producerInplaceIdx=%d srcRaw=%d sameSource=%d outIncastLinkMapSize=%zu",
        assembleOp->GetOpMagic(), assembleIn->GetMagic(), inRaw->rawmagic, inRaw->memoryId,
        MemoryTypeToString(assembleIn->GetMemoryTypeOriginal()).c_str(), assembleIn->GetDataSize(),
        assembleOut->GetMagic(), outRaw->rawmagic, outRaw->memoryId,
        MemoryTypeToString(assembleOut->GetMemoryTypeOriginal()).c_str(), assembleOut->GetDataSize(), inIsDdr,
        incastRaw->rawmagic, producerOpcode.c_str(), producerInplaceIdx, srcRaw ? srcRaw->rawmagic : -1, sameSource,
        function.outIncastLinkMap.size());

    return inIsDdr && sameSource;
}

/**
 * @brief 递归查找需要插入拷贝的 ASSEMBLE 操作
 */
Status ReplaceTensor::FindNeedToCopyAssemble(std::unordered_set<Operation*>& needInsertCopyAssOps,
                                             std::unordered_set<int>& visitedAssOps, Operation& op, Function& function)
{
    visitedAssOps.insert(op.GetOpMagic());
    auto assembleIn = op.GetIOperands()[0];
    auto assembleOut = op.GetOOperands()[0];
    auto producers = assembleIn->GetProducers();
    auto& inOp = *(assembleIn)->GetProducers().begin();
    if ((!producers.empty()) && (((*producers.begin())->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT) ||
                                 (*producers.begin())->GetOpcode() == Opcode::OP_SHMEM_WAIT_UNTIL)) {
        return FAILED;
    }
    const bool hasScatterInplaceProducer = std::any_of(
        producers.begin(), producers.end(), [](const Operation* producer) {
            return producer->GetOpcode() == Opcode::OP_SCATTER_INPLACE ||
                   producer->GetOpcode() == Opcode::OP_SCATTER_ELEMENT_INPLACE;
        });
    if (assembleIn->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR && hasScatterInplaceProducer &&
        !IsAssembleSameGmWriteBack(function, &op)) {
        needInsertCopyAssOps.insert(&op);
        return SUCCESS;
    }
    const int UB_SIZE_THRESHOLD = static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB));
    auto inProducerOps = inOp->ProducerOps();
    bool allProducerAssemble = !inProducerOps.empty();

    // BMM优化场景：Reshape前驱全为Assemble时不插copy。
    for (const auto& producer : inProducerOps) {
        allProducerAssemble = allProducerAssemble && producer->GetOpcode() == Opcode::OP_ASSEMBLE;
    }

    if (inOp->GetOpcode() == Opcode::OP_RESHAPE && !allProducerAssemble && !function.IsFromOutCast(assembleOut) &&
        !isBoundTensor(assembleOut) && op.GetIOperands()[0]->tensor->GetRawDataSize() <= UB_SIZE_THRESHOLD) {
        needInsertCopyAssOps.insert(&op);
        return SUCCESS;
    }
    auto consumers = assembleIn->GetConsumers();
    bool sameAssembleOut = true;
    for (const auto& con : consumers) {
        if (con->GetOOperands()[0]->GetRawMagic() != op.GetOOperands()[0]->GetRawMagic()) {
            sameAssembleOut = false;
            break;
        }
    }
    if (!sameAssembleOut) {
        for (const auto& con : consumers) {
            if (con->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            // 同址自写回为 no-op，无需防御拷贝
            if (IsAssembleSameGmWriteBack(function, con)) {
                visitedAssOps.insert(con->GetOpMagic());
                continue;
            }
            if (con->GetIOperands()[0]->GetDataSize() <= UB_SIZE_THRESHOLD) {
                visitedAssOps.insert(con->GetOpMagic());
                needInsertCopyAssOps.insert(con);
            }
        }
    }
    return SUCCESS;
}

bool ReplaceTensor::isBoundTensor(LogicalTensorPtr& curTensor)
{
    std::set<int> boundTensorIDs;
    for (auto& inOp : curTensor->GetProducers()) {
        boundTensorIDs.insert(inOp->GetSubgraphID());
    }
    for (auto& outOp : curTensor->GetConsumers()) {
        boundTensorIDs.insert(outOp->GetSubgraphID());
    }
    if (boundTensorIDs.size() > 1) {
        return true;
    }
    return false;
}

Status ReplaceTensor::FindNeedToCopyReshape(std::unordered_set<Operation*>& needInsertCopyAssOps,
                                            std::unordered_set<int>& visitedReshapeOps, Operation& op,
                                            Function& function)
{
    visitedReshapeOps.insert(op.GetOpMagic());
    if (op.GetIOperands()[0]->tensor->GetRawShapeSize() > 0 && op.GetOOperands()[0]->tensor->GetRawShapeSize() > 0 &&
        op.GetIOperands()[0]->tensor->GetRawShapeSize() != op.GetOOperands()[0]->tensor->GetRawShapeSize()) {
        needInsertCopyAssOps.insert(&op);
        return SUCCESS;
    }
    auto producerOps = op.ProducerOps();
    auto consumerOps = op.ConsumerOps();
    for (auto producesOp : producerOps) {
        if (producesOp->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (isBoundTensor(op.GetOOperands().front()) && !isBoundTensor(producesOp->GetIOperands().front()) &&
            !function.IsFromInCast(producesOp->GetIOperands().front()) &&
            producesOp->GetIOperands().front()->tensor->GetRawShapeSize() !=
                op.GetOOperands().front()->tensor->GetRawShapeSize()) {
            needInsertCopyAssOps.insert(&op);
            continue;
        }
        for (auto consumerOp : consumerOps) {
            if (inplaceOpSet.find(consumerOp->GetOpcode()) != inplaceOpSet.end()) {
                needInsertCopyAssOps.insert(&op);
            }
        }
    }
    return SUCCESS;
}

/**
 * @brief 遍历所有 ASSEMBLE 操作，为需要拷贝的操作插入拷贝序列，避免多个 ASSEMBLE 操作共享同一个输入导致的内存冲突
 * Tensor1 ---> Assemble ---> Tensor2
 *         ---> Assemble ---> Tensor3
 *         ---> Assemble ---> Tensor4

 * Tensor1 ---> View ---> Reshape ---> OP(非CopyIn) ---> Tensor2

 * Tensor1 ---> Reshape ---> Assemble ---> Tensor2(可能造成CopyOut+Reshape+Assemble的一些场景性能损失)
 */
Status ReplaceTensor::InsertNeedCopy(Function& function)
{
    std::unordered_set<int> visitedAssOps;
    std::unordered_set<int> visitedReshapeOps;
    std::unordered_set<Operation*> needInsertCopyAssOps;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE && (!visitedAssOps.count(op.GetOpMagic()))) {
            FindNeedToCopyAssemble(needInsertCopyAssOps, visitedAssOps, op, function);
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE && (!visitedReshapeOps.count(op.GetOpMagic()))) {
            FindNeedToCopyReshape(needInsertCopyAssOps, visitedReshapeOps, op, function);
        }
    }
    std::vector<Operation*> sortedOps(needInsertCopyAssOps.begin(), needInsertCopyAssOps.end());
    std::sort(sortedOps.begin(), sortedOps.end(),
              [](const Operation* a, const Operation* b) { return a->GetOpMagic() < b->GetOpMagic(); });
    for (auto& needInsertCopyAssOp : sortedOps) {
        auto input = needInsertCopyAssOp->GetIOperands()[0];
        constexpr size_t kMaxCopyShapeDim = 5;
        if (input->GetShape().size() > kMaxCopyShapeDim) {
            APASS_LOG_WARN_F(Elements::Tensor,
                             "Skip inserting copy for op [%s:%d]: input [%d] shape dim [%zu] exceeds pto-isa copy "
                             "limit [%zu].",
                             needInsertCopyAssOp->GetOpcodeStr().c_str(), needInsertCopyAssOp->GetOpMagic(),
                             input->GetMagic(), input->GetShape().size(), kMaxCopyShapeDim);
            continue;
        }
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            InsertCopyUBOp(function, needInsertCopyAssOp, input);
        } else if (input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            if (InsertCopyDDROp(function, needInsertCopyAssOp, input) == FAILED) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void ReplaceTensor::RebuildTokenProducer(Function& function)
{
    std::vector<Operation*> operations = function.Operations(false).DuplicatedOpList();
    std::vector<std::pair<Operation*, Token>> tokenProducers;
    // Find all ops that produce tokens
    for (auto* producer : operations) {
        if (producer == nullptr) {
            continue;
        }
        for (const auto& token : producer->result_token_) {
            tokenProducers.emplace_back(producer, token);
        }
    }

    for (const auto& [producer, token] : tokenProducers) {
        // Do not move tokens for non-inplace
        if (!IsInplaceOperation(*producer)) {
            continue;
        }
        const auto tokenConsumers = producer->ConsumerOpsByToken();
        std::vector<Operation*> consumers(tokenConsumers.begin(), tokenConsumers.end());
        // Find all consumers of the token that need to be moved
        std::vector<Operation*> matchedConsumers;
        for (auto* consumer : consumers) {
            // Only tokens where the producer and consumer inputs must have equal actualRawMagic need to be moved
            if (consumer != nullptr && HasSameRawTensorInput(*producer, *consumer)) {
                matchedConsumers.push_back(consumer);
            }
        }
        if (matchedConsumers.empty() || matchedConsumers.size() != consumers.size()) {
            continue;
        }

        // Clear the token relationship(s)
        auto& producerTokens = producer->result_token_;
        producerTokens.erase(std::remove(producerTokens.begin(), producerTokens.end(), token), producerTokens.end());
        function.GetVarDependency().RemoveProducer(token, GetStmt(*producer));
        for (auto* consumer : matchedConsumers) {
            RemoveTokenConsumer(*consumer, token, function);
        }
        function.GetVarDependency().RemoveVar(token);

        // Produce new tokens everyone in matchedConsumers.
        std::unordered_map<Operation*, Token> replacementTokens;
        for (auto* boundary : FindFirstNonInplaceConsumers(*producer)) {
            bool createsCycle = false;
            for (auto* consumer : matchedConsumers) {
                if (consumer != nullptr && HasDependencyPath(*consumer, *boundary)) {
                    createsCycle = true;
                    break;
                }
            }
            if (createsCycle) {
                continue;
            }
            auto tokenIt = replacementTokens.find(boundary);
            if (tokenIt == replacementTokens.end()) {
                if (boundary->result_token_.empty()) {
                    auto newToken = irBuilder_.CreateTokenVar(boundary->GetSpan());
                    boundary->result_token_.push_back(newToken);
                    function.GetVarDependency().AddProducer(newToken, GetStmt(*boundary));
                }
                tokenIt = replacementTokens.emplace(boundary, boundary->result_token_.front()).first;
            }
            for (auto* consumer : matchedConsumers) {
                AddTokenConsumer(*consumer, tokenIt->second, function);
            }
        }
    }
}

void ReplaceTensor::RebuildTokenConsumer(Function& function)
{
    std::vector<Operation*> operations = function.Operations(false).DuplicatedOpList();
    std::vector<std::pair<Operation*, std::vector<Token>>> tokenConsumers;
    // Find all ops that consume tokens
    for (auto* consumer : operations) {
        if (consumer != nullptr && !consumer->tokens_.empty()) {
            tokenConsumers.emplace_back(consumer, consumer->tokens_);
        }
    }

    for (const auto& [consumer, tokens] : tokenConsumers) {
        // Do not move tokens for non-inplace
        if (!IsInplaceOperation(*consumer)) {
            continue;
        }
        for (const auto& token : tokens) {
            std::vector<Operation*> matchedProducers;
            for (const auto& stmt : function.GetVarDependency().GetProducers(token)) {
                auto* producer = static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get()));
                // Only tokens where the producer and consumer inputs must have equal actualRawMagic need to be moved
                if (producer != nullptr && (HasSameRawTensorInput(*producer, *consumer) ||
                                            HasInputMatchingOutputRawTensor(*producer, *consumer))) {
                    matchedProducers.push_back(producer);
                }
            }
            if (matchedProducers.empty()) {
                continue;
            }

            auto boundaries = FindFirstNonInplaceProducers(*consumer);
            bool createsCycle = boundaries.empty();
            for (auto* boundary : boundaries) {
                for (auto* producer : matchedProducers) {
                    if (HasDependencyPath(*boundary, *producer)) {
                        createsCycle = true;
                        break;
                    }
                }
                if (createsCycle) {
                    break;
                }
            }
            if (createsCycle) {
                continue;
            }
            RemoveTokenConsumer(*consumer, token, function);
            for (auto* boundary : boundaries) {
                AddTokenConsumer(*boundary, token, function);
            }
        }
    }
}

Status ReplaceTensor::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "===> Start ReplaceTensor.");
    if (InsertNeedCopy(function) == FAILED) {
        return FAILED;
    }
    BuildRawMagicTensorIndex(function);
    auto tensorToOrderIndex = BuildTensorOrderIndexMap(function);
    UnionFind uf(tensorToOrderIndex);
    UniteTensor(function, uf);
    std::vector<LogicalTensors> tensorGroups = uf.GetGroups();
    for (auto& group : tensorGroups) {
        LogicalTensorPtr baseTensor = nullptr;
        if (group.size() == 1) {
            continue;
        }
        if (FindBaseTensor(function, tensorToOrderIndex, group, baseTensor) == FAILED || baseTensor == nullptr) {
            return FAILED;
        }
        backRoots.push(baseTensor);
        forRoots.push(baseTensor);
        // 新图表达下, base本身可能是assemble输出(有同rawMagic兄弟), 需同步兄弟并push到roots,
        // 否则兄弟链路可能无法通过producer/consumer关系到达。
        SyncSiblingAssembleOutput(function, baseTensor);
        while (!forRoots.empty() || !backRoots.empty()) {
            if (BackwardProcess(function) == FAILED || ForwardProcess(function) == FAILED) {
                return FAILED;
            }
        }
    }

    rawMagicTensors_.clear();

    // Some tensor replacements (for example L0C2UB offset folding) bypass an
    // obsolete Assemble operation by marking it deleted.  The replacement
    // traversal above must not erase operations in-place, because it is still
    // walking producer/consumer relationships.  Once all backward/forward
    // processing is complete, remove those tombstones before the remaining
    // ReplaceTensor stages inspect the graph; otherwise later stages may still
    // see a deleted Assemble and retain stale graph edges.
    function.EraseOperations(true, false);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);

    if (RefactorViewConnectForReplace(function) == FAILED) {
        return FAILED;
    }
    if (ProcessHubOp(function) == FAILED) {
        return FAILED;
    }
    RebuildTokenProducer(function);
    RebuildTokenConsumer(function);
    if (MarkTensorAsPartialMem(function) == FAILED) {
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End ReplaceTensor.");
    return SUCCESS;
}

Status ReplaceTensor::AdjustOffsetAndRawShape(LogicalTensorPtr& fromView, LogicalTensorPtr& toView) const
{
    auto fromType = fromView->tensor->datatype;
    auto toType = toView->tensor->datatype;
    auto inEntry = viewTypeTable.find(fromType);
    auto outEntry = viewTypeTable.find(toType);
    if (inEntry == viewTypeTable.end() || outEntry == viewTypeTable.end()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ViewType Input Tensor OR Output Tensor DataType is not in viewType, Please check it!");
        return FAILED;
    }
    int inSize = inEntry->second;
    int outSize = outEntry->second;
    int ratio = inSize > outSize ? inSize / outSize : outSize / inSize;
    bool isExpand = inSize > outSize;
    std::vector<int64_t> fromOffset = fromView->GetOffset();
    std::vector<int64_t> toOffset(fromOffset.size(), 0);
    std::vector<int64_t> inShape = fromView->GetRawTensor()->rawshape;
    std::vector<int64_t> outShape(inShape.size(), 0);
    for (size_t i = 0; i < fromOffset.size(); ++i) {
        if (i != fromOffset.size() - 1) {
            toOffset[i] = fromOffset[i];
            outShape[i] = inShape[i];
            continue;
        }
        if (isExpand) {
            toOffset[i] = fromOffset[i] * ratio;
            outShape[i] = inShape[i] * ratio;
        } else {
            if (fromOffset[i] % ratio != 0 || inShape[i] % ratio != 0) {
                APASS_LOG_ERROR_F(Elements::Operation, "ViewType Offset is not Even.");
                return FAILED;
            }
            toOffset[i] = fromOffset[i] / ratio;
            outShape[i] = inShape[i] / ratio;
        }
    }
    toView->UpdateOffset(toOffset);
    toView->GetRawTensor()->rawshape = outShape;
    return SUCCESS;
}

Status ReplaceTensor::ForUpdateView(Operation* op)
{
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    auto viewIn = op->GetIOperands()[0];
    auto viewOut = op->GetOOperands()[0];
    std::vector<int64_t> inputOffset = viewIn->GetOffset();
    if (viewAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ReplaceTensor::ForUpdateView: View op %d Attribute is nullptr.",
                          op->GetOpMagic());
        return FAILED;
    }
    std::vector<int64_t> viewOpOffset = viewAttr->GetFrom();
    auto inputDynOffset = viewIn->GetDynOffset();
    if (inputDynOffset.empty()) {
        inputDynOffset = std::vector<SymbolicScalar>(inputOffset.size(), 0);
    }
    auto attrDynOffset = viewAttr->GetFromDynOffset();
    std::vector<SymbolicScalar> outTensorOffset;
    for (size_t i = 0; i < inputOffset.size(); i++) {
        viewOpOffset[i] = inputOffset[i] + viewOpOffset[i];
        if (attrDynOffset.size() == inputOffset.size()) {
            attrDynOffset[i] = inputDynOffset[i] + attrDynOffset[i];
        }
    }
    viewAttr->SetFromOffset(viewOpOffset, viewAttr->GetFromDynOffset());
    TensorOffset newOffset(viewOpOffset, attrDynOffset);
    viewOut->UpdateOffset(newOffset);
    return SUCCESS;
}

std::vector<OpImmediate> ReplaceTensor::SumOffsetForCopyIn(const std::vector<OpImmediate> offset1,
                                                           const std::vector<OpImmediate> offset2)
{
    std::vector<OpImmediate> res;
    for (size_t i = 0; i < offset1.size(); i++) {
        res.push_back(offset1[i] + offset2[i]);
    }
    return res;
}

Status ReplaceTensor::UpdateCopyInAttr(Operation* copyInOp, bool updateOffset)
{
    auto input = copyInOp->GetIOperands()[0];
    auto copyInOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyInOp->GetOpAttribute());
    if (copyInOpAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "CopyInOp[%d] don not have attr.", copyInOp->GetOpMagic());
        return FAILED;
    } else {
        std::vector<OpImmediate> inputOffset;
        if (input->GetDynOffset().empty()) {
            inputOffset = OpImmediate::Specified(input->GetOffset());
        } else {
            inputOffset = OpImmediate::Specified(input->GetDynOffset());
        }
        std::vector<OpImmediate> oldFromOffset = copyInOpAttr->GetFromOffset();
        if (updateOffset && !inputOffset.empty() && !oldFromOffset.empty() &&
            (inputOffset.size() == oldFromOffset.size())) {
            copyInOpAttr->SetFromOffset(SumOffsetForCopyIn(inputOffset, oldFromOffset));
        }
        copyInOpAttr->SetRawShape(OpImmediate::Specified(input->tensor->GetDynRawShape()));
    }
    return SUCCESS;
}

Status ReplaceTensor::BackUpdateAssemble(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    auto assembleOut = op->GetOOperands()[0];
    auto assAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
    if (assAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ReplaceTensor::BackUpdateAssemble: Assemble op %d Attribute is nullptr.", op->GetOpMagic());
        return FAILED;
    }
    std::vector<int64_t> assOffset = assAttr->GetToOffset();
    std::vector<int64_t> outOffset = assembleOut->GetOffset();
    auto outDynOffset = assembleOut->GetDynOffset();
    auto assDynOffset = assAttr->GetToDynOffset();
    std::vector<SymbolicScalar> inTensorOffset;
    if (outDynOffset.empty()) {
        outDynOffset = std::vector<SymbolicScalar>(outOffset.size(), 0);
    }
    for (size_t i = 0; i < outOffset.size(); i++) {
        assOffset[i] = outOffset[i] + assOffset[i];
        if (assDynOffset.size() == outOffset.size()) {
            assDynOffset[i] = outDynOffset[i] + assDynOffset[i];
        }
    }
    assAttr->SetToOffset(assOffset, assAttr->GetToDynOffset());
    TensorOffset newOffset(assOffset, assDynOffset);
    assembleIn->UpdateOffset(newOffset);
    return SUCCESS;
}

// L0C2UB 直写折叠：BackUpdateAssemble 已把 assemble 目的偏移刷新到输入 tensor 上，
// 若该输入唯一 producer 是 L0C2UB copy：目的偏移携带非零值且 copy 源偏移可证明为立即零时，
// 把目的偏移落入 copy 的 toOffset 并改为 INSERT；源与目的同时携带偏移时报错
Status ReplaceTensor::FoldL0C2UBCopyOffset(Operation* op)
{
    auto assembleIn = op->GetIOperands()[0];
    const auto& producers = assembleIn->GetProducers();
    if (producers.size() != 1) {
        return SUCCESS;
    }
    auto copyOp = *producers.begin();
    if (copyOp->GetOpcode() != Opcode::OP_L0C_COPY_UB) {
        return SUCCESS;
    }
    // copy 输出被多个 consumer 使用时不折叠，避免不同 assemble 的 toOffset 互相覆盖
    if (assembleIn->GetConsumers().size() != 1) {
        return SUCCESS;
    }
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(copyOp->GetOpAttribute());
    if (copyAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ReplaceTensor::FoldL0C2UBCopyOffset: copy op %d attr is nullptr.",
                          copyOp->GetOpMagic());
        return FAILED;
    }
    // 源偏移携带判定：非立即零（参数、符号或非零立即数）均视为携带；该组合无法安全折叠，终止本次 Pass。
    bool srcHasOffset = false;
    for (const auto& offset : copyAttr->GetFromOffset()) {
        if (!offset.IsSpecified() || !offset.GetSpecifiedValue().IsImmediate() ||
            offset.GetSpecifiedValue().Raw()->GetImmediateValue() != 0) {
            srcHasOffset = true;
            break;
        }
    }
    if (srcHasOffset) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ReplaceTensor::FoldL0C2UBCopyOffset: copy op %d carries non-zero fromOffset while its "
                          "assemble consumer op %d carries toOffset, cannot fold offsets; aborting pass.",
                          copyOp->GetOpMagic(), op->GetOpMagic());
        return FAILED;
    }
    // Even a zero Assemble offset must be materialized as an INSERT before
    // removing the Assemble; otherwise the original EXTRACT copy would not
    // have the same destination semantics.
    copyAttr->SetToOffset(OpImmediate::Specified(TensorOffset(assembleIn->GetOffset(), assembleIn->GetDynOffset())));
    copyOp->SetAttribute(OpAttributeKey::localCopyLocalMode, static_cast<int64_t>(Matrix::CopyMode::INSERT));

    // The copy now writes directly into the Assemble destination.  Bypass the
    // obsolete Assemble node so its consumers use the copy result directly.
    // Do not erase from the operation list while ReplaceTensor is traversing
    // it; mark the op deleted and let RunOnFunction perform the cleanup.
    auto* function = op->BelongTo();
    if (function == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "ReplaceTensor::FoldL0C2UBCopyOffset: assemble op %d has no owner function.",
                          op->GetOpMagic());
        return FAILED;
    }
    // Keep the Assemble output tensor (which may be an outcast/shared raw
    // tensor).  Replacing consumers with the input view would lose that raw
    // tensor identity and can make the direct copy write to the wrong view.
    function->UpdateOperandBeforeRemoveOp(*op, true);
    op->SetAsDeleted();
    APASS_LOG_INFO_F(Elements::Operation,
                     "ReplaceTensor::FoldL0C2UBCopyOffset: remove folded assemble op %d, copy op %d writes "
                     "directly to destination.",
                     op->GetOpMagic(), copyOp->GetOpMagic());
    return SUCCESS;
}

Status ReplaceTensor::MarkTensorAsPartialMem(Function& func)
{
    for (auto& op : func.Operations()) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto iOperand = op.GetInputOperand(0);
        auto oOperand = op.GetOutputOperand(0);
        if (iOperand->GetRawTensor() != oOperand->GetRawTensor()) {
            continue;
        }
        iOperand->SetAttr("isPartialMem", true);
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
