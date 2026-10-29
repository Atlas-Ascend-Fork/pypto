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
 * \file remove_redundant_op.cpp
 * \brief
 */
#include <algorithm>
#include <climits>
#include <unordered_set>
#include "remove_redundant_op.h"
#include "passes/pass_utils/infer_discontinuous_input_utils.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_check/remove_redundant_op_checker.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/remove_redundant_op_internal.h"
#include "passes/pass_utils/remove_redundant_op_utils.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/tilefwk_op.h"

#define MODULE_NAME "RemoveRedundantOp"

namespace npu {
namespace tile_fwk {
namespace {
bool IsAssembleLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT; }

bool ProtectsPartialWriteSnapshot(const Operation& op)
{
    auto rawMagic = op.GetIOperands().front()->GetRawMagic();
    for (auto* consumer : op.ConsumerOpsByToken()) {
        if (IsAssembleLikeOpcode(consumer->GetOpcode()) &&
            consumer->GetOOperands().front()->GetRawMagic() == rawMagic &&
            consumer->GetIOperands().front()->GetShape() != consumer->GetOOperands().front()->GetShape()) {
            return true;
        }
    }
    return false;
}

std::vector<Operation*> GetExternalConsumers(Operation& op, const std::unordered_set<Operation*>& removedOps)
{
    std::vector<Operation*> consumers;
    for (auto* consumer : op.ConsumerOps()) {
        if (consumer == nullptr || removedOps.count(consumer) != 0) {
            continue;
        }
        if (std::find(consumers.begin(), consumers.end(), consumer) != consumers.end()) {
            continue;
        }
        consumers.emplace_back(consumer);
    }
    return consumers;
}

std::vector<Operation*> GetExternalProducers(Operation& op, const std::unordered_set<Operation*>& removedOps)
{
    std::vector<Operation*> producers;
    for (auto* producer : op.ProducerOps()) {
        if (producer == nullptr || removedOps.count(producer) != 0) {
            continue;
        }
        if (std::find(producers.begin(), producers.end(), producer) != producers.end()) {
            continue;
        }
        producers.emplace_back(producer);
    }
    return producers;
}

bool MoveTokenDependencyBeforeRemoveOp(Function& function, Operation& op)
{
    std::vector<Operation*> removedOps = {&op};
    std::unordered_set<Operation*> removedSet(removedOps.begin(), removedOps.end());
    auto targetConsumers = GetExternalConsumers(op, removedSet);
    auto targetProducers = GetExternalProducers(op, removedSet);
    if (!remove_redundant_op_internal::CanMigrateRemovedOpsTokenDependency(function, removedOps, targetConsumers,
                                                                           targetProducers)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Move token dependency before removing op[%d] would be invalid.",
                          op.GetOpMagic());
        return false;
    }
    remove_redundant_op_internal::MigrateRemovedOpsTokenDependency(function, removedOps, targetConsumers,
                                                                   targetProducers);
    return true;
}

bool HasAtomicWriteSemantic(const Operation& op)
{
    return op.HasAttr(RMW_MODE_ATTR_ADD) || op.HasAttr(RMW_MODE_ATTR_MIN) || op.HasAttr(RMW_MODE_ATTR_MAX);
}

bool CanRemoveAssembleByWriteRelation(Function& function, const Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_ASSEMBLE || op.GetOOperands().empty()) {
        return true;
    }
    auto assembleOutput = op.GetOOperands().front();
    if (assembleOutput->GetProducers().size() > 1) {
        APASS_LOG_DEBUG_F(Elements::Operation, "assembleOutput[%d] has more than one producer, skip removing.",
                          assembleOutput->GetMagic());
        return false;
    }
    if (remove_redundant_op_internal::HasOtherAssembleOutputOnSameRaw(function, assembleOutput, &op)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "assemble op[%d]'s output raw[%d] has another assemble output, skip removing.",
                          op.GetOpMagic(), assembleOutput->GetRawMagic());
        return false;
    }
    if (HasAtomicWriteSemantic(op)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "assemble op[%d] has atomic write semantic, skip removing.",
                          op.GetOpMagic());
        return false;
    }
    return true;
}

// 有效形状恒非负，最外层 RUNTIME_Max(expr, 0) 截断不改变取值；剥离该无效截断返回内部表达式串，
// 用于识别仅相差一层截断的等价表达式。不满足剥离条件时原样返回。
std::string StripUselessMaxZeroClamp(const std::string& dump)
{
    constexpr char kMaxPrefix[] = "RUNTIME_Max(";
    constexpr char kZeroSuffix[] = ", 0)";
    const size_t prefixLen = sizeof(kMaxPrefix) - 1;
    const size_t suffixLen = sizeof(kZeroSuffix) - 1;
    if (dump.size() <= prefixLen + suffixLen || dump.compare(0, prefixLen, kMaxPrefix) != 0 ||
        dump.compare(dump.size() - suffixLen, suffixLen, kZeroSuffix) != 0) {
        return dump;
    }
    std::string inner = dump.substr(prefixLen, dump.size() - prefixLen - suffixLen);
    // 中间表达式括号必须平衡，确保 ", 0)" 属于最外层 Max 的第二个参数
    int32_t depth = 0;
    for (char c : inner) {
        if (c == '(') {
            depth++;
        } else if (c == ')') {
            depth--;
            if (depth < 0) {
                return dump;
            }
        }
    }
    if (depth != 0) {
        return dump;
    }
    return inner;
}

// VIEW 输出是否仅被 ReduceAcc 消费；放宽有效形状比较仅限该场景
bool IsReduceAccExclusiveConsumer(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto& consumers = tensor->GetConsumers();
    if (consumers.empty()) {
        return false;
    }
    return std::all_of(consumers.begin(), consumers.end(), [](const Operation* consumer) {
        return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_REDUCE_ACC;
    });
}
} // namespace

bool EqualInOutShape(const Operation& op)
{
    auto in = op.GetIOperands().front();
    auto out = op.GetOOperands().front();
    // 比较memtype
    bool equalMemType = (in->GetMemoryTypeOriginal() == out->GetMemoryTypeOriginal());
    // 比较静态shape
    bool equalShape = (in->GetShape() == out->GetShape());
    return (equalMemType && equalShape);
}

bool EqualInOut(const Operation& op, bool relaxMaxZeroClamp)
{
    auto in = op.GetIOperands().front();
    auto out = op.GetOOperands().front();
    bool equalShape = EqualInOutShape(op);
    bool equalDynValidShape = true;
    if (!in->GetDynValidShape().empty() && !out->GetDynValidShape().empty()) {
        auto inDynValidShape = in->GetDynValidShape();
        auto outDynValidShape = out->GetDynValidShape();
        for (size_t i = 0; i < inDynValidShape.size(); i++) {
            if (inDynValidShape[i].Dump() == outDynValidShape[i].Dump()) {
                continue;
            }
            // 仅 ReduceAcc 消费的 VIEW 允许剥离无效的 RUNTIME_Max(expr, 0) 外层截断后再比较：
            // slice 模式下该别名 VIEW 与其输入（Contract 输出）的有效形状可能仅相差这一层截断
            if (!relaxMaxZeroClamp || StripUselessMaxZeroClamp(inDynValidShape[i].Dump()) !=
                                          StripUselessMaxZeroClamp(outDynValidShape[i].Dump())) {
                equalDynValidShape = false;
                break;
            }
        }
    } else if (in->GetDynValidShape().empty() && out->GetDynValidShape().empty()) {
        equalDynValidShape = true;
    } else {
        equalDynValidShape = false;
    }
    return (equalShape && equalDynValidShape);
}

bool RemoveRedundantOp::ProcessRedundantOpWithDynShape(Operation& op) const
{
    bool relaxMaxZeroClamp = op.GetOpcode() == Opcode::OP_VIEW && op.GetOOperands().size() == 1 &&
                             IsReduceAccExclusiveConsumer(op.GetOOperands().front());
    if (!EqualInOut(op, relaxMaxZeroClamp)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "op[%d]'s input and output has unequal shape and dynshape, skip removing.", op.GetOpMagic());
        return false;
    }
    return true;
}

bool RemoveRedundantOp::ProcessRedundantOpWithoutDynShape(Operation& op) const
{
    if (!EqualInOutShape(op)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op[%d]'s input and output has unequal shape, skip removing.",
                          op.opmagic);
        return false;
    }
    if (IsAssembleLikeOpcode(op.GetOpcode())) {
        auto assembleOutput = op.GetOOperands().front();
        if (assembleOutput->GetProducers().size() > 1) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "assemble-like output[%d] has more than one producer, skip removing.",
                              assembleOutput->GetMagic());
            return false;
        }
        auto assembleInput = op.GetIOperands().front();
        bool hasParallelAssembleSameOutput = false;
        for (const auto& consumer : assembleInput->GetConsumers()) {
            if (consumer == &op || !IsAssembleLikeOpcode(consumer->GetOpcode()) || consumer->GetOOperands().empty()) {
                continue;
            }
            if (consumer->GetOOperands().front() == assembleOutput) {
                hasParallelAssembleSameOutput = true;
                break;
            }
        }
        bool hasReshapeConsumer = false;
        for (const auto& consumer : assembleOutput->GetConsumers()) {
            if (consumer->GetOpcode() == Opcode::OP_RESHAPE) {
                hasReshapeConsumer = true;
                break;
            }
        }
        if (hasParallelAssembleSameOutput && hasReshapeConsumer)
            return false;
    }
    return true;
}

Status RemoveRedundantOp::RemoveDummyOp(Function& function)
{
    bool anyRemoved = false;
    for (auto& op : function.Operations()) {
        // Keep the snapshot read before a partial overwrite of the same raw tensor.
        if ((op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_SLICE) &&
            ProtectsPartialWriteSnapshot(op)) {
            continue;
        }
        bool canRemove = false;
        if (matchOpcodeWithDynshape.find(op.GetOpcode()) != matchOpcodeWithDynshape.end()) {
            canRemove = ProcessRedundantOpWithDynShape(op);
        } else if (matchOpcodeWithoutDynshape.find(op.GetOpcode()) != matchOpcodeWithoutDynshape.end()) {
            canRemove = ProcessRedundantOpWithoutDynShape(op);
            if (canRemove && op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                canRemove = CanRemoveAssembleByWriteRelation(function, op);
            }
        }
        // outcast tensor 的 ASSEMBLE 不能移除（当 input 有其他 consumer 时）：
        // 移除后 outcast 会被提前到 input tensor，继承其内部 consumer，
        // 导致 SubgraphToFunction 的 isFromCast 误判 outcast 为非 boundary tensor，
        // 进而 AllocWorkspaceGM 误设 workspaceBaseOffset，跨 kernel 读取到未初始化数据
        if (canRemove && op.GetOpcode() == Opcode::OP_ASSEMBLE && function.IsFromOutCast(op.GetOOperands().front()) &&
            op.GetIOperands().front()->GetConsumers().size() > 1) {
            canRemove = false;
        }
        if (canRemove) {
            if (!MoveTokenDependencyBeforeRemoveOp(function, op)) {
                continue;
            }
            operationUpdated = true;
            anyRemoved = true;
            function.UpdateOperandBeforeRemoveOp(op, false);
        }
    }
    if (anyRemoved) {
        DeadOperationEliminator::EliminateDeadOperation(function);
    }
    return SUCCESS;
}

Status RemoveRedundantOp::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveRedundantOp");
    operationUpdated = true;
    iterTime = 0U;
    newOps_.clear();
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    while (operationUpdated) {
        operationUpdated = false;
        if (RemoveDummyOps(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "RemoveDummyOps failed.");
            return FAILED;
        }
        if (RemoveDummyOp(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "RemoveDummyOp failed.");
            return FAILED;
        }
        iterTime++;
    }
    // 冗余删除（REGISTER_COPY 等）会引入新的 view-assemble 结构，需重新推测不连续输入。
    // 放在 MergeViewAssemble 之前：MergeViewAssemble 仅合并已有的 view-assemble 模式，
    // 不引入新的不连续输入冲突；而 re-infer 需在合并前处理冗余删除引入的新结构。
    // 跳过 NoViewConflict：REGISTER_COPY 删除后 ASSEMBLE 输入可能穿透到上游 VIEW(inCast)，
    // 但 ASSEMBLE 输出位置不重叠即无冲突，仅靠 PerfectOffsetOverlap 判断即可
    InferDiscontinuousInputUtils utils;
    if (utils.Process(function, false) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "InferDiscontinuousInput after RemoveRedundantOp failed.");
        return FAILED;
    }
    Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
        return status;
    }
    if (!newOps_.empty()) {
        status = InferShapeUtils::InferShape(function, newOps_);
        if (status != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape for new operations failed.");
            return status;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveRedundantOp");
    return SUCCESS;
}

Status RemoveRedundantOp::PreCheck(Function& function)
{
    RemoveRedundantOpChecker checker;
    return checker.DoPreCheck(function);
}

Status RemoveRedundantOp::PostCheck(Function& function)
{
    RemoveRedundantOpChecker checker;
    return checker.DoPostCheck(function);
}

Status RemoveRedundantOp::RemoveDummyOps(Function& function)
{
    if (ProcessReshape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessReshape failed.");
        return FAILED;
    }
    if (iterTime == 0U) {
        if (ProcessViewAssemble(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessViewAssemble failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOp::ProcessViewAssemble(Function& function)
{
    if (RemoveRedundantOpUtils::ProcessViewAssembleLike(function, newOps_, operationUpdated) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessViewAssembleLike failed.");
        return FAILED;
    }
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}

Status RemoveRedundantOp::ProcessReshape(Function& function)
{
    bool anyRemoved = false;
    for (auto& op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_RESHAPE) {
            // 跳过非reshape的op
            continue;
        }
        auto in = op.GetIOperands().front();
        auto out = op.GetOOperands().front();
        bool canRemove = false;
        if (in->shape == out->shape && !CommonUtils::ContainsNegativeOne(in->GetShape()) &&
            !CommonUtils::ContainsNegativeOne(out->GetShape())) {
            APASS_LOG_DEBUG_F(Elements::Operation, "op[%d]'s in->shape == out->shape.", op.GetOpMagic());
            canRemove = true;
        } else if (!op.ConsumerOps().empty()) {
            canRemove = true;
            for (auto& consumerOp : op.ConsumerOps()) {
                if (consumerOp->GetOpcode() != Opcode::OP_RESHAPE) {
                    canRemove = false;
                    continue;
                }
                consumerOp->ReplaceInput(in, out);
            }
            if (canRemove) {
                APASS_LOG_DEBUG_F(Elements::Operation, "All consummers of op [%d] are reshape.", op.GetOpMagic());
            }
        }
        if (canRemove) {
            if (!MoveTokenDependencyBeforeRemoveOp(function, op)) {
                continue;
            }
            function.UpdateOperandBeforeRemoveOp(op, false);
            operationUpdated = true;
            anyRemoved = true;
        }
    }
    if (anyRemoved) {
        DeadOperationEliminator::EliminateDeadOperation(function);
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
