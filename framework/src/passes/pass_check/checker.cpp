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
 * \file remove_redundant_op_checker.cpp
 * \brief
 */

#include "remove_redundant_op_checker.h"
#include "interface/utils/common.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/error_code.h"

#define MODULE_NAME "Checker"

namespace npu {
namespace tile_fwk {
Status Checker::DoPreCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::DoPostCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::DoDefaultEnabledPreCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::DoDefaultEnabledPostCheck(Function& function)
{
    (void)function;
    return SUCCESS;
}

Status Checker::CheckConsumerProducer(const LogicalTensorPtr& tensor)
{
    for (const auto& producers : tensor->GetProducers()) {
        if (producers == nullptr) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "Found null producer in tensor.");
            return FAILED;
        }
    }
    for (const auto& consumers : tensor->GetConsumers()) {
        if (consumers == nullptr) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "Found null consumer in tensor.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckValidOp(Function& function)
{
    for (const auto& op : function.Operations().DuplicatedOpList()) {
        if (op == nullptr) {
            APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                              "Found null op in function.Operations().");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckOpIOValid(Function& function)
{
    for (const auto& op : function.Operations().DuplicatedOpList()) {
        for (const auto& input : op->iOperand) {
            if (input == nullptr) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "The input of op[%d] is null",
                                  op->opmagic);
                return FAILED;
            }
            if (CheckConsumerProducer(input) != SUCCESS) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation,
                                  "CheckConsumerProducer for op[%d]'s input failed!", op->opmagic);
                return FAILED;
            }
        }
        for (const auto& output : op->oOperand) {
            if (output == nullptr) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation, "The output of op[%d] is null",
                                  op->opmagic);
                return FAILED;
            }
            if (CheckConsumerProducer(output) != SUCCESS) {
                APASS_LOG_ERROR_C(TensorErr::TENSOR_NULL_POINTER, Elements::Operation,
                                  "CheckConsumerProducer for op[%d]'s output failed!", op->opmagic);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status Checker::CheckCompleteness(Function& function)
{
    if (function.GetIncast().empty()) {
        APASS_LOG_WARN_F(Elements::Function, "The incast of function[%d] is empty.", function.GetFuncMagic());
        return SUCCESS;
    }
    for (const auto& incast : function.GetIncast()) {
        if (incast == nullptr) {
            APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                              "The function[%d] contains incast which is null.", function.GetFuncMagic());
            return FAILED;
        }
        if (incast->GetConsumers().empty()) {
            APASS_LOG_WARN_F(Elements::Operation, "The incast[%d] has no consumer.", incast->GetMagic());
            continue;
        }
    }
    if (function.GetOutcast().empty()) {
        APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                          "The outcast of function[%d] is empty.", function.GetFuncMagic());
        return FAILED;
    }
    for (const auto& outcast : function.GetOutcast()) {
        if (outcast == nullptr) {
            APASS_LOG_ERROR_C(FunctionErr::FUNCTION_GRAPH_STRUCTURE, Elements::Function,
                              "The function[%d] contains outcast which is null.", function.GetFuncMagic());
            return FAILED;
        }
        if (outcast->GetProducers().empty()) {
            APASS_LOG_WARN_F(Elements::Operation, "The outcast[%d] has no producer.", outcast->GetMagic());
            continue;
        }
    }
    return SUCCESS;
}

Status Checker::CheckGraphLoop(Function& function)
{
    if (function.GetTotalSubGraphCount() == 0 && !function.OperationLoopCheck()) {
        APASS_LOG_ERROR_C(GraphErr::GRAPH_LOOP_DETECTION, Elements::Operation,
                          "OperationLoopCheck failed, there is a loop in function[%d].", function.GetFuncMagic());
        return FAILED;
    }
    if (!function.LoopCheck().empty()) {
        APASS_LOG_ERROR_C(GraphErr::GRAPH_LOOP_DETECTION, Elements::Function,
                          "Loopcheck failed, there is a loop in function[%d].", function.GetFuncMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status Checker::PublicCheck(Function& function)
{
    if (CheckValidOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckValidOp for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckOpIOValid(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckOpIOValid for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckCompleteness(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckCompleteness for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckGraphLoop(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckGraphLoop for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckLocalTensor(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckLocalTensor for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status Checker::CheckDynAttrForView(Function& function)
{
    std::unordered_set<Operation*> needCheckViewOps;
    // 边界张量无动态轴（shape 不含 -1）时，View 无需携带动态属性即可表达完整语义，跳过校验；
    // 含 -1 动态轴的边界张量（incast/outcast）侧的 View 仍要求携带动态属性
    for (const auto& incast : function.GetIncast()) {
        if (!HasNegativeNum<int64_t>(incast->GetShape())) {
            continue;
        }
        for (auto& consumer : incast->GetConsumers()) {
            if (consumer->GetOpcode() == Opcode::OP_VIEW) {
                needCheckViewOps.insert(consumer);
            }
        }
    }
    for (const auto& outcast : function.GetOutcast()) {
        if (!HasNegativeNum<int64_t>(outcast->GetShape())) {
            continue;
        }
        for (auto& producer : outcast->GetProducers()) {
            if (producer->GetOpcode() == Opcode::OP_VIEW) {
                needCheckViewOps.insert(producer);
            }
        }
    }
    for (const auto& op : needCheckViewOps) {
        const int opMagic = op->GetOpMagic();
        const int funcMagic = function.GetFuncMagic();
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
        std::vector<SymbolicScalar>& viewFromDynOffset = viewAttr->GetFromDynOffset();
        if (viewFromDynOffset.empty()) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_DYNAMIC_ATTR, Elements::Operation,
                              "CheckDynAttrForView failed, fromDynOffset_ of op[%d] in function[%d] is empty.", opMagic,
                              funcMagic);
            return FAILED;
        }
        std::vector<SymbolicScalar>& viewToDynValidShape = viewAttr->GetToDynValidShape();
        if (viewToDynValidShape.empty()) {
            APASS_LOG_ERROR_C(TensorErr::TENSOR_DYNAMIC_ATTR, Elements::Operation,
                              "CheckDynAttrForView failed, toDynValidShape_ of op[%d] in function[%d] is empty.",
                              opMagic, funcMagic);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckToDynOffsetForAssemble(Function& function)
{
    // 仅校验动态 outcast（shape 含 -1）的写入描述：outcast 无动态轴时写入位置可由静态 offset 完整描述，
    // 不强制其 Assemble 携带 toDynOffset_；incast 侧展开后不存在 Assemble 直接连接 incast 的场景，故不收集
    for (const auto& outcast : function.GetOutcast()) {
        if (!HasNegativeNum<int64_t>(outcast->GetShape())) {
            continue;
        }
        for (auto& producer : outcast->GetProducers()) {
            if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            const int opMagic = producer->GetOpMagic();
            const int funcMagic = function.GetFuncMagic();
            auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
            std::vector<SymbolicScalar>& assembleToDynOffset = assembleAttr->GetToDynOffset();
            if (assembleToDynOffset.empty()) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_DYNAMIC_ATTR, Elements::Operation,
                    "CheckToDynOffsetForAssemble failed, toDynOffset_ of op[%d] in function[%d] is empty.", opMagic,
                    funcMagic);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status Checker::CheckShapeForCopyOp(Function& function)
{
    for (auto& op : function.Operations()) {
        if (!IsCopyIn(op.GetOpcode()) && !IsCopyOut(op.GetOpcode())) {
            continue;
        }
        auto copyOpAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
        if (copyOpAttr == nullptr || copyOpAttr->GetShape().size() == copyOpAttr->GetRawShape().size()) {
            continue;
        }
        APASS_LOG_ERROR_C(TensorErr::TENSOR_SHAPE_MISMATCH, Elements::Operation,
                          "CheckShapeForCopyOp failed, shape dimension: %zu and rawShape dimension: %zu of op[%d] in "
                          "function[%d] are mismatched.",
                          copyOpAttr->GetShape().size(), copyOpAttr->GetRawShape().size(), op.GetOpMagic(),
                          function.GetFuncMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status Checker::CheckLocalTensor(Function& function)
{
    auto opList = function.Operations().DuplicatedOpList();
    for (const auto& op : opList) {
        for (auto& iOperand : op->GetIOperands()) {
            if (iOperand == nullptr) {
                APASS_LOG_ERROR_C(OperationErr::OP_NULL_POINTER, Elements::Operation,
                                  "The iOperand of op[%d][%s] is null", op->GetOpMagic(), op->GetOpcodeStr().c_str());
                return FAILED;
            }
            if (iOperand->GetProducers().empty() &&
                FunctionUtils::GetNodeType(*iOperand, function) != NodeType::INCAST) {
                APASS_LOG_ERROR_C(
                    TensorErr::TENSOR_NULL_POINTER, Elements::Operation,
                    "A locally defined temporary tensor[%d] cannot be used as an input to an operation[%d].",
                    iOperand->GetMagic(), op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
