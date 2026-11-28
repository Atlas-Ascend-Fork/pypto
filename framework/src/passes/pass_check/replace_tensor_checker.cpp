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
 * \file replace_tensor_checker.cpp
 * \brief
 */

#include "replace_tensor_checker.h"

#include <functional>
#include <unordered_map>

#include "passes/pass_log/pass_log.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"

#define MODULE_NAME "ReplaceTensor"

namespace npu {
namespace tile_fwk {
bool ReplaceTensorChecker::HasSameConsecutive(const Operation& op) const
{
    for (auto* nextOp : op.ConsumerOps()) {
        if (nextOp->GetOpcode() == op.GetOpcode()) {
            return true;
        }
    }
    return false;
}

Status ReplaceTensorChecker::DoPreCheck(Function& function)
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

bool ReplaceTensorChecker::CheckAddrConflict(const Operation& op)
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

bool ReplaceTensorChecker::CheckIndexProducer(const Operation& op)
{
    for (const auto& producer : op.ProducerOps()) {
        if (producer->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            return true;
        }
    }
    return false;
}

/*
用于校验assemble节点的输入输出是否存在冲突
注意，若输入由OP_INDEX_OUTCAST构造，则不会出现冲突
*/
bool ReplaceTensorChecker::CheckAssembleConflict(const Operation& op)
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
bool ReplaceTensorChecker::CheckIndexOutcastConflict(const Operation& op, Function& function)
{
    int index = 2;
    auto indexIn = op.GetInputOperand(index);
    auto indexOut = op.GetOOperands().front();
    if (forwardOps_.find(op.GetOpMagic()) != forwardOps_.end()) {
        if (function.IsFromInCast(indexIn) && function.IsFromOutCast(indexOut)) {
            return false;
        }
    }
    if (backwardOps_.find(op.GetOpMagic()) != backwardOps_.end()) {
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
bool ReplaceTensorChecker::CheckReshapeConflict(const Operation& op, Function& function)
{
    if (op.GetBoolAttribute(OP_ATTR_PREFIX + "isInplace"))
        return false;
    if (forwardOps_.find(op.GetOpMagic()) != forwardOps_.end()) {
        auto tensorOut = op.GetOOperands().front();
        if (function.IsFromOutCast(tensorOut)) {
            return false;
        }
    }
    if (backwardOps_.find(op.GetOpMagic()) != backwardOps_.end()) {
        if (CheckAddrConflict(op)) {
            return true;
        }
    }
    return false;
}

/*
用于校验a_mulacc_b节点的输入输出是否存在冲突
*/
bool ReplaceTensorChecker::CheckAMulAccBConflict(const Operation& op)
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

Status ReplaceTensorChecker::InplaceCheck(Function& function)
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

Status ReplaceTensorChecker::DoPostCheck(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor.");
    if (InplaceCheck(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InplaceCheck failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for ReplaceTensor success.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
