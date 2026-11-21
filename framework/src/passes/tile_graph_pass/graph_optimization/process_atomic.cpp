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
 * \file process_atomic.cpp
 * \brief Process atomic operations including ReduceAcc and AtomicRMW
 */

#include "process_atomic.h"
#include "interface/configs/config_manager_ng.h"
#include "passes/pass_check/process_atomic_checker.h"
#include "interface/operation/attribute.h"
#include "tilefwk/tilefwk_op.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/view_reshape_assemble_reorder_utils.h"
#include "passes/pass_log/pass_log.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <set>

#define MODULE_NAME "ProcessAtomic"

namespace npu {
namespace tile_fwk {

namespace {
ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

Operation* ToOperation(const ir::StmtPtr& stmt)
{
    if (stmt == nullptr) {
        return nullptr;
    }
    return static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get()));
}

template <typename T>
void AddUnique(std::vector<T>& values, const T& value)
{
    if (value == nullptr || std::find(values.begin(), values.end(), value) != values.end()) {
        return;
    }
    values.emplace_back(value);
}

void RemoveToken(Operation& op, const ir::VarPtr& token)
{
    if (token == nullptr) {
        return;
    }
    op.tokens_.erase(std::remove(op.tokens_.begin(), op.tokens_.end(), token), op.tokens_.end());
}

void AddTokenConsumer(Function& function, const ir::VarPtr& token, Operation& consumer)
{
    if (token == nullptr) {
        return;
    }
    function.GetVarDependency().AddConsumer(token, ToStmtPtr(consumer));
    AddUnique(consumer.tokens_, token);
}

ir::VarPtr EnsureResultToken(Function& function, Operation& producer)
{
    if (producer.result_token_.empty()) {
        producer.result_token_.push_back(IRBuilder().CreateTokenVar(producer.GetSpan()));
    }
    function.GetVarDependency().AddProducer(producer.result_token_.front(), ToStmtPtr(producer));
    return producer.result_token_.front();
}

std::vector<SymbolicScalar> GetSymbolicShapeOrStatic(const std::shared_ptr<LogicalTensor>& tensor)
{
    if (tensor == nullptr) {
        return {};
    }
    const auto& dynShape = tensor->GetDynValidShape();
    return dynShape.size() == tensor->GetShape().size() ? dynShape : SymbolicScalar::FromConcrete(tensor->GetShape());
}

bool IsContractOpcode(Opcode opcode)
{
    // ProcessAtomic may run on graphs that contain a mixture of legacy
    // Assemble nodes and slice-mode Contract nodes.  This is expected for
    // example when an AtomicRMW consumes a tensor assembled before the
    // slice/contract lowering pass.  Do not gate the check on the current
    // ENABLE_SLICE setting: both opcodes have the same assemble-like
    // producer semantics here.  ASSEMBLE_SSA is kept for the non-slice SSA
    // form as well.
    return IsAssembleLike(opcode) || opcode == Opcode::OP_ASSEMBLE_SSA;
}

Status CollectReduceAccAssembleProducers(const Operation& op, std::vector<Operation*>& assembleProducers)
{
    for (const auto& input : op.GetIOperands()) {
        if (input == nullptr || input->GetProducers().empty()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "ReduceAcc[%d] input has no producer; expected Assemble before ProcessAtomic.%s",
                              op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        for (auto* producer : input->GetProducers()) {
            if (producer == nullptr || !IsContractOpcode(producer->GetOpcode())) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "ReduceAcc[%d] input producer must be Contract/Assemble, but got %s.%s",
                                  op.GetOpMagic(), producer == nullptr ? "nullptr" : producer->GetOpcodeStr().c_str(),
                                  GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            assembleProducers.push_back(producer);
        }
    }
    return SUCCESS;
}

bool IsAssembleLike(const Operation* op)
{
    return op != nullptr && (op->GetOpcode() == Opcode::OP_ASSEMBLE || op->GetOpcode() == Opcode::OP_ASSEMBLE_SSA ||
                             op->GetOpcode() == Opcode::OP_CONTRACT);
}

std::vector<Operation*> CollectRawAssembleProducers(Function& function, const std::shared_ptr<LogicalTensor>& tensor)
{
    std::vector<Operation*> producers;
    if (tensor == nullptr) {
        return producers;
    }
    auto rawMagic = tensor->GetRawMagic();
    for (auto& op : function.Operations(false)) {
        if (op.IsDeleted() || !IsAssembleLike(&op) || op.GetOOperands().empty()) {
            continue;
        }
        auto output = op.GetOOperands().front();
        if (output != nullptr && output->GetRawMagic() == rawMagic) {
            AddUnique(producers, &op);
        }
    }
    std::sort(producers.begin(), producers.end(),
              [](const Operation* lhs, const Operation* rhs) { return lhs->GetOpMagic() < rhs->GetOpMagic(); });
    return producers;
}

bool IsExactOutputProducer(const Operation& producer, const std::shared_ptr<LogicalTensor>& tensor)
{
    return !producer.GetOOperands().empty() && producer.GetOOperands().front() != nullptr && tensor != nullptr &&
           producer.GetOOperands().front()->GetMagic() == tensor->GetMagic();
}

bool HasRawVersionProducer(const std::vector<Operation*>& producers, const std::shared_ptr<LogicalTensor>& tensor)
{
    return std::any_of(producers.begin(), producers.end(), [&tensor](const Operation* producer) {
        return producer != nullptr && !IsExactOutputProducer(*producer, tensor);
    });
}

bool AreAllContractProducers(const std::set<Operation*, LogicalTensor::CompareOp>& producers)
{
    return std::all_of(producers.begin(), producers.end(), [](const Operation* producer) {
        return producer != nullptr && IsContractOpcode(producer->GetOpcode());
    });
}

bool HasRawExternalConsumer(Function& function, const std::shared_ptr<LogicalTensor>& tensor, const Operation& atomicOp)
{
    auto rawProducers = CollectRawAssembleProducers(function, tensor);
    for (auto* producer : rawProducers) {
        if (producer == nullptr || producer->GetOOperands().empty()) {
            continue;
        }
        auto output = producer->GetOOperands().front();
        if (output == nullptr) {
            continue;
        }
        for (auto* consumer : output->GetConsumers()) {
            if (consumer != nullptr && consumer->GetOpMagic() != atomicOp.GetOpMagic()) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<LogicalTensor> CreateLogicalTensorOnRaw(Function& function, const std::shared_ptr<RawTensor>& rawTensor,
                                                        const std::shared_ptr<LogicalTensor>& origin)
{
    if (rawTensor == nullptr || origin == nullptr) {
        return nullptr;
    }
    auto cloned = IRBuilder().CreateTensorVar(function, rawTensor, origin->GetOffset(), origin->GetShape(),
                                              origin->GetDynValidShape());
    cloned->CopyMemoryType(origin);
    return cloned;
}

void CopyBasicOpMetadata(const Operation& source, Operation& target)
{
    target.UpdateSubgraphID(source.GetSubgraphID());
    target.SetCoreType(source.GetCoreType());
    target.SetSpan(source.GetSpan());
    target.SetScopeInfo(source.GetScopeInfo());
    if (source.GetOpAttribute() != nullptr) {
        target.SetOpAttribute(source.GetOpAttribute()->Clone());
    }
}

bool IsTensorOnRaw(const std::shared_ptr<LogicalTensor>& tensor, int rawMagic)
{
    return tensor != nullptr && tensor->GetRawMagic() == rawMagic;
}

bool ReadsRawTensor(const Operation& op, int rawMagic)
{
    return std::any_of(
        op.GetIOperands().begin(), op.GetIOperands().end(),
        [rawMagic](const std::shared_ptr<LogicalTensor>& tensor) { return IsTensorOnRaw(tensor, rawMagic); });
}

std::shared_ptr<LogicalTensor> CloneAuxiliaryTensor(Function& function, const std::shared_ptr<LogicalTensor>& origin,
                                                    std::unordered_map<int, std::shared_ptr<LogicalTensor>>& tensorMap)
{
    if (origin == nullptr) {
        return nullptr;
    }
    auto iter = tensorMap.find(origin->GetMagic());
    if (iter != tensorMap.end()) {
        return iter->second;
    }
    auto cloned = origin->Clone(function, true);
    if (cloned == nullptr) {
        return nullptr;
    }
    tensorMap.emplace(origin->GetMagic(), cloned);
    return cloned;
}

Operation* CloneTokenBoundaryProducer(
    Function& function, Operation& source, int rawMagic,
    const std::function<std::shared_ptr<LogicalTensor>(const std::shared_ptr<LogicalTensor>&)>& getClonedRawTensor,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>>& auxTensorMap)
{
    if (source.IsDeleted() || source.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        return nullptr;
    }

    LogicalTensors clonedInputs;
    clonedInputs.reserve(source.GetIOperands().size());
    for (const auto& input : source.GetIOperands()) {
        auto mapped = input;
        if (IsTensorOnRaw(input, rawMagic)) {
            mapped = getClonedRawTensor(input);
        } else if (input != nullptr) {
            auto iter = auxTensorMap.find(input->GetMagic());
            if (iter != auxTensorMap.end()) {
                mapped = iter->second;
            }
        }
        if (mapped == nullptr) {
            return nullptr;
        }
        clonedInputs.emplace_back(mapped);
    }

    LogicalTensors clonedOutputs;
    clonedOutputs.reserve(source.GetOOperands().size());
    for (const auto& output : source.GetOOperands()) {
        auto mapped = IsTensorOnRaw(output, rawMagic) ? getClonedRawTensor(output) :
                                                        CloneAuxiliaryTensor(function, output, auxTensorMap);
        if (mapped == nullptr) {
            return nullptr;
        }
        clonedOutputs.emplace_back(mapped);
    }

    auto& clonedProducer = source.CloneOperation(function, clonedInputs, clonedOutputs);
    CopyBasicOpMetadata(source, clonedProducer);
    return &clonedProducer;
}

Status CloneTokenBoundaryProducers(
    Function& function, int rawMagic,
    const std::function<std::shared_ptr<LogicalTensor>(const std::shared_ptr<LogicalTensor>&)>& getClonedRawTensor,
    std::unordered_map<int, std::shared_ptr<LogicalTensor>>& auxTensorMap,
    std::unordered_map<Operation*, Operation*>& cloneMap)
{
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<std::pair<ir::VarPtr, VarDependency::Entry>> dependencies;
        dependencies.reserve(function.GetVarDependency().GetAllDependencies().size());
        for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
            dependencies.emplace_back(token, entry);
        }

        for (const auto& dependency : dependencies) {
            const auto& entry = dependency.second;
            bool hasClonedConsumer = std::any_of(
                entry.consumers.begin(), entry.consumers.end(),
                [&cloneMap](const ir::StmtPtr& consumerStmt) { return cloneMap.count(ToOperation(consumerStmt)) > 0; });
            if (!hasClonedConsumer) {
                continue;
            }
            for (const auto& producerStmt : entry.producers) {
                auto* producer = ToOperation(producerStmt);
                if (producer == nullptr || cloneMap.count(producer) > 0 || !ReadsRawTensor(*producer, rawMagic)) {
                    continue;
                }
                auto* clonedProducer = CloneTokenBoundaryProducer(function, *producer, rawMagic, getClonedRawTensor,
                                                                  auxTensorMap);
                if (clonedProducer == nullptr) {
                    return FAILED;
                }
                cloneMap.emplace(producer, clonedProducer);
                changed = true;
            }
        }
    }
    return SUCCESS;
}

void CopyCloneInternalTokenDependency(Function& function, const std::unordered_map<Operation*, Operation*>& cloneMap)
{
    auto& dependency = function.GetVarDependency();
    std::vector<std::pair<ir::VarPtr, VarDependency::Entry>> dependencies;
    dependencies.reserve(dependency.GetAllDependencies().size());
    for (const auto& [token, entry] : dependency.GetAllDependencies()) {
        dependencies.emplace_back(token, entry);
    }

    for (const auto& [token, entry] : dependencies) {
        std::vector<Operation*> oldProducers;
        std::vector<Operation*> oldConsumers;
        std::unordered_map<Operation*, ir::VarPtr> clonedProducerTokens;
        for (const auto& producerStmt : entry.producers) {
            AddUnique(oldProducers, ToOperation(producerStmt));
        }
        for (const auto& consumerStmt : entry.consumers) {
            AddUnique(oldConsumers, ToOperation(consumerStmt));
        }
        for (auto* oldConsumer : oldConsumers) {
            auto consumerIter = cloneMap.find(oldConsumer);
            if (consumerIter == cloneMap.end()) {
                continue;
            }
            auto* clonedConsumer = consumerIter->second;
            bool hasClonedProducer = false;
            for (auto* oldProducer : oldProducers) {
                auto producerIter = cloneMap.find(oldProducer);
                if (producerIter == cloneMap.end()) {
                    continue;
                }
                hasClonedProducer = true;
                auto tokenIter = clonedProducerTokens.find(producerIter->second);
                if (tokenIter == clonedProducerTokens.end()) {
                    auto clonedToken = IRBuilder().CreateTokenVar(producerIter->second->GetSpan());
                    producerIter->second->result_token_.push_back(clonedToken);
                    dependency.AddProducer(clonedToken, ToStmtPtr(*producerIter->second));
                    tokenIter = clonedProducerTokens.emplace(producerIter->second, clonedToken).first;
                }
                auto clonedToken = tokenIter->second;
                AddTokenConsumer(function, clonedToken, *clonedConsumer);
            }
            if (!hasClonedProducer) {
                AddTokenConsumer(function, token, *clonedConsumer);
            }
        }
    }
}

std::shared_ptr<LogicalTensor> CloneRawAssembleSetForAtomic(Function& function, Operation& atomicOp,
                                                            const std::shared_ptr<LogicalTensor>& input,
                                                            const std::vector<Operation*>& rawProducers)
{
    if (input == nullptr || rawProducers.empty() || input->GetRawTensor() == nullptr) {
        return nullptr;
    }
    auto originRaw = input->GetRawTensor();
    auto clonedRaw = IRBuilder().CreateRawTensor(originRaw->GetDataType(), originRaw->GetRawShape(), input->Format());
    clonedRaw->UpdateDynRawShape(originRaw->GetDynRawShape());
    clonedRaw->memoryId = originRaw->memoryId;

    std::unordered_map<int, std::shared_ptr<LogicalTensor>> rawTensorMap;
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> auxTensorMap;
    std::unordered_map<Operation*, Operation*> cloneMap;
    auto getClonedTensor = [&](const std::shared_ptr<LogicalTensor>& origin) -> std::shared_ptr<LogicalTensor> {
        if (origin == nullptr) {
            return nullptr;
        }
        auto iter = rawTensorMap.find(origin->GetMagic());
        if (iter != rawTensorMap.end()) {
            return iter->second;
        }
        auto cloned = CreateLogicalTensorOnRaw(function, clonedRaw, origin);
        rawTensorMap.emplace(origin->GetMagic(), cloned);
        return cloned;
    };

    for (auto* producer : rawProducers) {
        if (!IsAssembleLike(producer) || producer->GetOOperands().empty()) {
            return nullptr;
        }
        auto clonedOutput = getClonedTensor(producer->GetOOperands().front());
        if (clonedOutput == nullptr) {
            return nullptr;
        }
        auto& clonedProducer = producer->CloneOperation(function, producer->GetIOperands(), {clonedOutput});
        CopyBasicOpMetadata(*producer, clonedProducer);
        cloneMap.emplace(producer, &clonedProducer);
    }

    if (CloneTokenBoundaryProducers(function, originRaw->rawmagic, getClonedTensor, auxTensorMap, cloneMap) !=
        SUCCESS) {
        return nullptr;
    }
    CopyCloneInternalTokenDependency(function, cloneMap);
    auto clonedInput = getClonedTensor(input);
    if (clonedInput == nullptr) {
        return nullptr;
    }
    atomicOp.ReplaceInput(clonedInput, input);
    return clonedInput;
}

Status MigrateFoldedOpInputTokens(Function& function, Operation& foldedOp, const std::vector<Operation*>& replacements)
{
    auto& dependency = function.GetVarDependency();
    auto foldedStmt = ToStmtPtr(foldedOp);

    auto inputTokens = foldedOp.tokens_;
    for (const auto& token : inputTokens) {
        auto tokenProducers = dependency.GetProducers(token);
        bool producedByReplacement = std::any_of(
            tokenProducers.begin(), tokenProducers.end(), [&replacements](const ir::StmtPtr& producerStmt) {
                auto* producer = ToOperation(producerStmt);
                return std::find(replacements.begin(), replacements.end(), producer) != replacements.end();
            });
        dependency.RemoveConsumer(token, foldedStmt);
        RemoveToken(foldedOp, token);
        if (producedByReplacement) {
            continue;
        }
        if (replacements.empty()) {
            return FAILED;
        }
        for (auto* replacement : replacements) {
            AddTokenConsumer(function, token, *replacement);
        }
    }
    foldedOp.tokens_.clear();
    return SUCCESS;
}

Status MigrateFoldedOpResultTokens(Function& function, Operation& foldedOp, const std::vector<Operation*>& replacements)
{
    auto oldResultTokens = foldedOp.result_token_;
    if (oldResultTokens.empty()) {
        return SUCCESS;
    }
    auto& dependency = function.GetVarDependency();
    auto foldedStmt = ToStmtPtr(foldedOp);
    std::vector<Operation*> externalConsumers;
    for (const auto& oldResultToken : oldResultTokens) {
        for (const auto& consumerStmt : dependency.GetConsumers(oldResultToken)) {
            AddUnique(externalConsumers, ToOperation(consumerStmt));
        }
    }
    if (!externalConsumers.empty() && replacements.empty()) {
        return FAILED;
    }
    for (const auto& oldResultToken : oldResultTokens) {
        auto oldConsumers = dependency.GetConsumers(oldResultToken);
        for (auto* replacement : replacements) {
            auto replacementToken = EnsureResultToken(function, *replacement);
            for (auto* consumer : externalConsumers) {
                AddTokenConsumer(function, replacementToken, *consumer);
            }
        }
        for (const auto& consumerStmt : oldConsumers) {
            auto* consumer = ToOperation(consumerStmt);
            dependency.RemoveConsumer(oldResultToken, consumerStmt);
            if (consumer != nullptr) {
                RemoveToken(*consumer, oldResultToken);
            }
        }
        dependency.RemoveProducer(oldResultToken, foldedStmt);
        dependency.RemoveVar(oldResultToken);
    }
    foldedOp.result_token_.clear();
    return SUCCESS;
}

Status MigrateFoldedOpTokenDependency(Function& function, Operation& foldedOp,
                                      const std::vector<Operation*>& replacementProducers)
{
    std::vector<Operation*> replacements;
    for (auto* producer : replacementProducers) {
        AddUnique(replacements, producer);
    }
    if (MigrateFoldedOpInputTokens(function, foldedOp, replacements) != SUCCESS) {
        return FAILED;
    }
    return MigrateFoldedOpResultTokens(function, foldedOp, replacements);
}

} // namespace

Status ProcessAtomic::PreCheck(Function& function)
{
    ProcessAtomicChecker checker;
    const auto status = checker.DoPreCheck(function);
    return status;
}

Status ProcessAtomic::PostCheck(Function& function)
{
    ProcessAtomicChecker checker;
    const auto status = checker.DoPostCheck(function);
    return status;
}

Status ProcessAtomic::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start ProcessAtomic.");
    if (CheckAtomicRMWUnsupportedMode(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Unsupported AtomicRMW mode detected.");
        return FAILED;
    }
    bool hasReduceAccCascade = false;
    if (EliminateVecDupBranch(function, hasReduceAccCascade) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate VecDup branch failed.");
        return FAILED;
    }
    if (EliminateReduceAcc(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate ReduceAcc failed.");
        return FAILED;
    }
    if (EliminateAtomicRMW(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate AtomicRMW failed.");
        return FAILED;
    }
    if (hasReduceAccCascade) {
        Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
        if (status != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
            return status;
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End ProcessAtomic.");
    return SUCCESS;
}

Status ProcessAtomic::CheckAtomicRMWUnsupportedMode(Function& function)
{
    for (const auto& op : function.Operations(true, SortOperationsMode::LIGHTWEIGHT)) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            int rmwModeValue = op.GetIntAttribute(OpAttributeKey::rmwMode);
            AtomicRMWMode rmwMode = static_cast<AtomicRMWMode>(rmwModeValue);
            if (rmwMode == AtomicRMWMode::MAX || rmwMode == AtomicRMWMode::MIN) {
                APASS_LOG_ERROR_F(
                    Elements::Operation,
                    "Op[%d] AtomicRMW mode '%s' is not supported yet. "
                    "Currently only ADD mode is supported. Please use ADD mode or wait for future support.%s",
                    op.GetOpMagic(), (rmwMode == AtomicRMWMode::MAX ? "MAX" : "MIN"), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::EliminateReduceAcc(Function& function)
{
    bool anyDeleted = false;
    for (auto& op : function.Operations(true, SortOperationsMode::LIGHTWEIGHT)) {
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            APASS_LOG_INFO_F(Elements::Operation, "ATOMIC_ADD, opmagic: %d", op.GetOpMagic());
            std::vector<Operation*> assembleProducers;
            if (CollectReduceAccAssembleProducers(op, assembleProducers) != SUCCESS) {
                return FAILED;
            }
            auto reduceOut = op.GetOOperands().front();
            reduceOut->GetProducers().clear();
            std::vector<Operation*> replacementProducers;
            for (auto* producer : assembleProducers) {
                if (CheckAndSetRmwAttr(*producer, AtomicRMWMode::ADD, RMW_MODE_ATTR_ADD) != SUCCESS) {
                    return FAILED;
                }
                producer->SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);
                producer->ReplaceOOperand(0, reduceOut);
                AddUnique(replacementProducers, producer);
            }
            if (MigrateFoldedOpTokenDependency(function, op, replacementProducers) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "Migrate token dependency for ReduceAcc op[%d] failed.",
                                  op.GetOpMagic());
                return FAILED;
            }
            op.SetAsDeleted();
            anyDeleted = true;
            APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] will be deleted.", op.GetOpcodeStr().c_str(),
                              op.GetOpMagic());
        }
    }
    if (anyDeleted) {
        function.EraseOperations(true, true, SortOperationsMode::LIGHTWEIGHT);
        if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function,
                              "Eliminate dead operation failed for ReduceAcc in CommonOperationEliminate.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::EliminateAtomicRMW(Function& function)
{
    std::vector<Operation*> atomicRmwOps;
    for (auto& op : function.Operations(true, SortOperationsMode::LIGHTWEIGHT)) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            atomicRmwOps.emplace_back(&op);
        }
    }
    if (atomicRmwOps.empty()) {
        return SUCCESS;
    }
    if (PrepareAtomicRMWSharedInputs(function, atomicRmwOps) != SUCCESS) {
        return FAILED;
    }
    bool anyDeleted = false;
    for (auto* op : atomicRmwOps) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        if (ProcessSingleAtomicRMW(function, *op) != SUCCESS) {
            return FAILED;
        }
        anyDeleted = true;
    }
    if (anyDeleted) {
        function.EraseOperations(true, true, SortOperationsMode::LIGHTWEIGHT);
        if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function,
                              "Eliminate dead operation failed for AtomicRMW in CommonOperationEliminate.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::ProcessSingleAtomicRMW(Function& function, Operation& op)
{
    APASS_LOG_INFO_F(Elements::Operation, "ATOMIC_RMW, opmagic: %d", op.GetOpMagic());

    auto rmwOut = op.GetOOperands().front();
    auto contractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
    if (contractAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] missing contract op attribute; Cannot eliminate.",
                          op.GetOpMagic());
        return FAILED;
    }

    auto& rmwOffset = contractAttr->GetToOffset();
    auto& rmwDynOffset = contractAttr->GetToDynOffset();

    int rmwModeValue = op.GetIntAttribute(OpAttributeKey::rmwMode);
    AtomicRMWMode rmwMode = static_cast<AtomicRMWMode>(rmwModeValue);
    if (GetRmwAttrKey(rmwMode).empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] has invalid rmwMode value %d.", op.GetOpMagic(), rmwModeValue);
        return FAILED;
    }
    std::vector<Operation*> replacementProducers;
    for (const auto& input : op.GetIOperands()) {
        if (ProcessAtomicInput(op, input, rmwOut, rmwMode, rmwOffset, rmwDynOffset, replacementProducers) != SUCCESS) {
            return FAILED;
        }
    }
    if (MigrateFoldedOpTokenDependency(function, op, replacementProducers) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Migrate token dependency for AtomicRMW op[%d] failed.",
                          op.GetOpMagic());
        return FAILED;
    }
    op.SetAsDeleted();
    APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] will be deleted.", op.GetOpcodeStr().c_str(), op.GetOpMagic());
    return SUCCESS;
}

Status ProcessAtomic::ProcessAtomicInput(Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input,
                                         const std::shared_ptr<LogicalTensor>& output, AtomicRMWMode rmwMode,
                                         const std::vector<int64_t>& rmwOffset,
                                         const std::vector<SymbolicScalar>& rmwDynOffset,
                                         std::vector<Operation*>& replacementProducers)
{
    auto* function = atomicOp.BelongTo();
    if (function == nullptr) {
        return FAILED;
    }
    auto producersBackup = input->GetProducers();
    auto rawProducers = CollectRawAssembleProducers(*function, input);
    if (!producersBackup.empty() && AreAllContractProducers(producersBackup) &&
        !HasRawVersionProducer(rawProducers, input)) {
        for (auto* producerOp : producersBackup) {
            if (ProcessAtomicContractProducer(atomicOp, *producerOp, output, rmwMode, rmwOffset, rmwDynOffset,
                                              replacementProducers) != SUCCESS) {
                return FAILED;
            }
        }
        return SUCCESS;
    }
    if (!rawProducers.empty()) {
        for (auto* producerOp : rawProducers) {
            if (producerOp == nullptr) {
                return FAILED;
            }
            if (IsExactOutputProducer(*producerOp, input)) {
                if (ProcessAtomicAssembleProducer(atomicOp, *producerOp, output, rmwMode, rmwOffset, rmwDynOffset,
                                                  replacementProducers) != SUCCESS) {
                    return FAILED;
                }
                continue;
            }
            if (MarkContractProducerAtomic(*producerOp, rmwMode, rmwOffset, rmwDynOffset) != SUCCESS) {
                return FAILED;
            }
            AddUnique(replacementProducers, producerOp);
        }
        return SUCCESS;
    }
    if (HasReshapeProducer(input)) {
        return ProcessAtomicThroughReshape(atomicOp, input, output, rmwMode, rmwOffset, rmwDynOffset,
                                           replacementProducers);
    }
    APASS_LOG_ERROR_F(Elements::Operation,
                      "AtomicRMW[%d] input producers must be Contract or a supported Reshape chain.%s",
                      atomicOp.GetOpMagic(), GetFormatBacktrace(atomicOp).c_str());
    return FAILED;
}

Status ProcessAtomic::ProcessAtomicContractProducer(Operation& atomicOp, Operation& producerOp,
                                                    const std::shared_ptr<LogicalTensor>& output, AtomicRMWMode rmwMode,
                                                    const std::vector<int64_t>& rmwOffset,
                                                    const std::vector<SymbolicScalar>& rmwDynOffset,
                                                    std::vector<Operation*>& replacementProducers)
{
    if (producerOp.GetIOperands().size() != 1 || !HasReshapeProducer(producerOp.GetInputOperand(0))) {
        if (ProcessContractProducer(producerOp, output, rmwMode, rmwOffset, rmwDynOffset) != SUCCESS) {
            return FAILED;
        }
        AddUnique(replacementProducers, &producerOp);
        return SUCCESS;
    }
    std::vector<int64_t> combinedOffset;
    std::vector<SymbolicScalar> combinedDynOffset;
    if (CombineContractOffset(producerOp, rmwOffset, rmwDynOffset, combinedOffset, combinedDynOffset) != SUCCESS) {
        return FAILED;
    }
    return ProcessAtomicThroughReshape(atomicOp, producerOp.GetInputOperand(0), output, rmwMode, combinedOffset,
                                       combinedDynOffset, replacementProducers);
}

Status ProcessAtomic::ProcessAtomicAssembleProducer(Operation& atomicOp, Operation& producerOp,
                                                    const std::shared_ptr<LogicalTensor>& output, AtomicRMWMode rmwMode,
                                                    const std::vector<int64_t>& rmwOffset,
                                                    const std::vector<SymbolicScalar>& rmwDynOffset,
                                                    std::vector<Operation*>& replacementProducers)
{
    if (producerOp.GetIOperands().size() != 1 || !HasReshapeProducer(producerOp.GetInputOperand(0))) {
        if (ProcessContractProducer(producerOp, output, rmwMode, rmwOffset, rmwDynOffset) != SUCCESS) {
            return FAILED;
        }
        AddUnique(replacementProducers, &producerOp);
        return SUCCESS;
    }
    std::vector<int64_t> combinedOffset;
    std::vector<SymbolicScalar> combinedDynOffset;
    if (CombineContractOffset(producerOp, rmwOffset, rmwDynOffset, combinedOffset, combinedDynOffset) != SUCCESS) {
        return FAILED;
    }
    return ProcessAtomicThroughReshape(atomicOp, producerOp.GetInputOperand(0), output, rmwMode, combinedOffset,
                                       combinedDynOffset, replacementProducers);
}

Status ProcessAtomic::ProcessAtomicThroughReshape(Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input,
                                                  const std::shared_ptr<LogicalTensor>& output, AtomicRMWMode rmwMode,
                                                  const std::vector<int64_t>& rmwOffset,
                                                  const std::vector<SymbolicScalar>& rmwDynOffset,
                                                  std::vector<Operation*>& replacementProducers)
{
    ReshapeRemapResult remapResult;
    if (FindUpstreamAssembleAndRemapOffset(input, output, rmwOffset, rmwDynOffset, remapResult) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] cannot remap AtomicRMW offset through Reshape; Cannot eliminate.",
                          atomicOp.GetOpMagic());
        return FAILED;
    }
    for (auto* contract : remapResult.assembles) {
        if (MarkContractProducerAtomic(*contract, rmwMode, remapResult.mappedOffset, remapResult.mappedDynOffset) !=
            SUCCESS) {
            return FAILED;
        }
        AddUnique(replacementProducers, contract);
    }
    return RetargetReshapeChain(atomicOp, output, remapResult);
}

bool ProcessAtomic::HasReshapeProducer(const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* producer : input->GetProducers()) {
        if (producer != nullptr && producer->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

Status ProcessAtomic::FindUpstreamAssembleAndRemapOffset(const std::shared_ptr<LogicalTensor>& input,
                                                         const std::shared_ptr<LogicalTensor>& outputBase,
                                                         const std::vector<int64_t>& offset,
                                                         const std::vector<SymbolicScalar>& dynOffset,
                                                         ReshapeRemapResult& result) const
{
    auto current = input;
    result = {};
    result.mappedOffset = offset;
    result.mappedDynOffset = dynOffset;
    if (outputBase == nullptr || outputBase->GetShape().size() != offset.size()) {
        return FAILED;
    }
    auto currentBaseShape = outputBase->GetShape();
    auto currentBaseDynShape = GetSymbolicShapeOrStatic(outputBase);
    std::set<int> visited;
    while (current != nullptr) {
        bool found = false;
        if (CollectTerminalAssembles(current, visited, result, found) != SUCCESS) {
            return FAILED;
        }
        if (found) {
            result.assembleOutputShape = std::move(currentBaseShape);
            result.assembleOutputDynShape = std::move(currentBaseDynShape);
            result.terminalTensor = current;
            return SUCCESS;
        }
        const auto& producers = current->GetProducers();
        if (producers.size() != 1) {
            return FAILED;
        }
        auto* producer = *producers.begin();
        if (producer == nullptr || !visited.insert(producer->GetOpMagic()).second) {
            return FAILED;
        }
        if (RemapThroughReshape(*producer, current, currentBaseShape, currentBaseDynShape, result) != SUCCESS) {
            return FAILED;
        }
    }
    return FAILED;
}

Status ProcessAtomic::CollectTerminalAssembles(const std::shared_ptr<LogicalTensor>& current, std::set<int>& visited,
                                               ReshapeRemapResult& result, bool& found) const
{
    found = false;
    const auto& producers = current->GetProducers();
    if (producers.empty()) {
        return FAILED;
    }
    if (!AreAllContractProducers(producers)) {
        return SUCCESS;
    }
    auto* function = (*producers.begin())->BelongTo();
    if (function == nullptr) {
        return FAILED;
    }
    auto rawProducers = CollectRawAssembleProducers(*function, current);
    if (rawProducers.empty()) {
        return FAILED;
    }
    for (auto* producer : rawProducers) {
        if (!visited.insert(producer->GetOpMagic()).second) {
            return FAILED;
        }
        result.assembles.push_back(producer);
        if (IsExactOutputProducer(*producer, current)) {
            result.retargetAssembles.push_back(producer);
        }
    }
    if (result.retargetAssembles.empty()) {
        return FAILED;
    }
    found = true;
    return SUCCESS;
}

Status ProcessAtomic::RemapThroughReshape(Operation& producer, std::shared_ptr<LogicalTensor>& current,
                                          std::vector<int64_t>& currentBaseShape,
                                          std::vector<SymbolicScalar>& currentBaseDynShape,
                                          ReshapeRemapResult& result) const
{
    if (producer.GetOpcode() != Opcode::OP_RESHAPE || producer.GetIOperands().size() != 1) {
        return FAILED;
    }
    auto next = producer.GetInputOperand(0);
    std::vector<int64_t> nextBaseShape, nextOffset;
    std::vector<SymbolicScalar> nextBaseDynShape, nextDynOffset;
    result.reshapeOps.push_back(&producer);
    result.reshapeOutputShapes.push_back(currentBaseShape);
    result.reshapeOutputDynShapes.push_back(currentBaseDynShape);
    if (!ViewReshapeAssembleReorderUtils::RemapOffsetBackwardThroughReshape(
            next, current, currentBaseShape, currentBaseDynShape, result.mappedOffset, result.mappedDynOffset,
            nextBaseShape, nextBaseDynShape, nextOffset, nextDynOffset)) {
        return FAILED;
    }
    currentBaseShape = std::move(nextBaseShape);
    currentBaseDynShape = std::move(nextBaseDynShape);
    result.mappedOffset = std::move(nextOffset);
    result.mappedDynOffset = std::move(nextDynOffset);
    current = next;
    return SUCCESS;
}

Status ProcessAtomic::RetargetReshapeChain(Operation& atomicOp, const std::shared_ptr<LogicalTensor>& output,
                                           const ReshapeRemapResult& remapResult)
{
    auto* function = atomicOp.BelongTo();
    if (function == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op[%d] does not belong to a function; Cannot retarget Reshape chain.",
                          atomicOp.GetOpMagic());
        return FAILED;
    }
    if (remapResult.retargetAssembles.empty()) {
        return FAILED;
    }
    auto current = remapResult.retargetAssembles.front()->GetOutputOperand(0);
    if (current == nullptr || current->GetShape() != remapResult.assembleOutputShape) {
        auto original = current;
        current = irBuilder_.CreateTensorVar(*function, output->Datatype(), remapResult.assembleOutputShape,
                                             remapResult.assembleOutputDynShape, output->Format());
        if (original != nullptr) {
            current->CopyMemoryType(original);
        }
        for (auto* assemble : remapResult.retargetAssembles) {
            assemble->ReplaceOOperand(0, current);
        }
    }
    for (size_t index = remapResult.reshapeOps.size(); index-- > 0;) {
        auto* reshapeOp = remapResult.reshapeOps[index];
        auto originalReshapeOutput = reshapeOp->GetOutputOperand(0);
        auto reshapeOutput = index == 0 ? output :
                                          irBuilder_.CreateTensorVar(
                                              *function, output->Datatype(), remapResult.reshapeOutputShapes[index],
                                              remapResult.reshapeOutputDynShapes[index], output->Format());
        if (index != 0 && originalReshapeOutput != nullptr) {
            reshapeOutput->CopyMemoryType(originalReshapeOutput);
        }
        reshapeOp->ReplaceIOperand(0, current);
        reshapeOp->ReplaceOOperand(0, reshapeOutput);
        reshapeOp->SetAttribute("reshape", reshapeOutput->GetShape());
        reshapeOp->SetAttribute(OP_ATTR_PREFIX + "validShape", remapResult.reshapeOutputDynShapes[index]);
        reshapeOutput->UpdateDynValidShape(remapResult.reshapeOutputDynShapes[index]);
        current = std::move(reshapeOutput);
    }
    return SUCCESS;
}

Status ProcessAtomic::CombineContractOffset(const Operation& contract, const std::vector<int64_t>& offset,
                                            const std::vector<SymbolicScalar>& dynOffset,
                                            std::vector<int64_t>& combinedOffset,
                                            std::vector<SymbolicScalar>& combinedDynOffset) const
{
    auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(contract.GetOpAttribute());
    if (attr == nullptr || attr->GetToOffset().size() != offset.size()) {
        return FAILED;
    }
    combinedOffset = attr->GetToOffset();
    for (size_t i = 0; i < offset.size(); ++i) {
        combinedOffset[i] += offset[i];
    }
    combinedDynOffset.clear();
    if (!attr->GetToDynOffset().empty() || !dynOffset.empty()) {
        auto lhs = attr->GetToDynOffset().size() == offset.size() ? attr->GetToDynOffset() :
                                                                    SymbolicScalar::FromConcrete(attr->GetToOffset());
        auto rhs = dynOffset.size() == offset.size() ? dynOffset : SymbolicScalar::FromConcrete(offset);
        combinedDynOffset.reserve(offset.size());
        for (size_t i = 0; i < offset.size(); ++i) {
            combinedDynOffset.push_back((lhs[i] + rhs[i]).Simplify());
        }
    }
    return SUCCESS;
}

bool ProcessAtomic::HasContractProducer(const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* producerOp : input->GetProducers()) {
        if (producerOp == nullptr) {
            continue;
        }
        if (IsContractOpcode(producerOp->GetOpcode())) {
            return true;
        }
    }
    return false;
}

bool ProcessAtomic::HasConsumerExcept(const std::shared_ptr<LogicalTensor>& input, const Operation& op) const
{
    if (input == nullptr) {
        return false;
    }
    for (auto* consumerOp : input->GetConsumers()) {
        if (consumerOp != nullptr && consumerOp->GetOpMagic() != op.GetOpMagic()) {
            return true;
        }
    }
    return false;
}

Status ProcessAtomic::PrepareAtomicRMWSharedInputs(Function& function,
                                                   const std::vector<Operation*>& atomicRmwOps) const
{
    for (auto* op : atomicRmwOps) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        auto inputsBackup = op->GetIOperands();
        for (const auto& input : inputsBackup) {
            if (PrepareExclusiveAtomicInput(function, *op, input) == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Prepare shared input failed for AtomicRMW op[%d].",
                                  op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

std::shared_ptr<LogicalTensor> ProcessAtomic::PrepareExclusiveAtomicInput(
    Function& function, Operation& atomicOp, const std::shared_ptr<LogicalTensor>& input) const
{
    if (input == nullptr) {
        return input;
    }

    auto rawProducers = CollectRawAssembleProducers(function, input);
    if (!rawProducers.empty() && HasRawExternalConsumer(function, input, atomicOp)) {
        auto clonedInput = CloneRawAssembleSetForAtomic(function, atomicOp, input, rawProducers);
        if (clonedInput == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Clone atomic raw producer set for tensor[%d] failed.",
                              input->GetMagic());
            return nullptr;
        }
        APASS_LOG_INFO_F(Elements::Tensor, "Clone raw producer set for AtomicRMW input tensor[%d].", input->GetMagic());
        return clonedInput;
    }

    if (!HasConsumerExcept(input, atomicOp) || !HasContractProducer(input)) {
        return input;
    }
    auto producersBackup = input->GetProducers();
    auto contractClonedInput = input->Clone(function, true);
    if (contractClonedInput == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "Clone atomic input tensor[%d] failed.", input->GetMagic());
        return nullptr;
    }
    atomicOp.ReplaceInput(contractClonedInput, input);

    for (auto* producerOp : producersBackup) {
        if (producerOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Null producer detected for atomic input tensor[%d].",
                              input->GetMagic());
            return nullptr;
        }
        if (!IsContractOpcode(producerOp->GetOpcode())) {
            continue;
        }
        auto& clonedProducer = producerOp->CloneOperation(function, producerOp->GetIOperands(),
                                                          producerOp->GetOOperands());
        clonedProducer.UpdateSubgraphID(producerOp->GetSubgraphID());
        clonedProducer.SetCoreType(producerOp->GetCoreType());
        clonedProducer.SetSpan(producerOp->GetSpan());
        clonedProducer.SetScopeInfo(producerOp->GetScopeInfo());
        if (producerOp->GetOpAttribute() != nullptr) {
            clonedProducer.SetOpAttribute(producerOp->GetOpAttribute()->Clone());
        }
        clonedProducer.ReplaceOutput(contractClonedInput, input);
    }
    return contractClonedInput;
}

std::string ProcessAtomic::GetRmwAttrKey(AtomicRMWMode mode)
{
    switch (mode) {
        case AtomicRMWMode::ADD:
            return RMW_MODE_ATTR_ADD;
        case AtomicRMWMode::MAX:
            return RMW_MODE_ATTR_MAX;
        case AtomicRMWMode::MIN:
            return RMW_MODE_ATTR_MIN;
        default:
            return "";
    }
}

Status ProcessAtomic::ProcessContractProducer(Operation& producerOp, std::shared_ptr<LogicalTensor> rmwOut,
                                              AtomicRMWMode rmwMode, const std::vector<int64_t>& rmwOffset,
                                              const std::vector<SymbolicScalar>& rmwDynOffset)
{
    std::string rmwAttrKey = GetRmwAttrKey(rmwMode);
    if (CheckAndSetRmwAttr(producerOp, rmwMode, rmwAttrKey) != SUCCESS) {
        return FAILED;
    }
    producerOp.SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);

    producerOp.ReplaceOOperand(0, rmwOut);

    auto producerContractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp.GetOpAttribute());
    if (producerContractAttr != nullptr &&
        AccumulateContractOffset(producerContractAttr, rmwOffset, rmwDynOffset) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::MarkContractProducerAtomic(Operation& producerOp, AtomicRMWMode rmwMode,
                                                 const std::vector<int64_t>& rmwOffset,
                                                 const std::vector<SymbolicScalar>& rmwDynOffset)
{
    std::string rmwAttrKey = GetRmwAttrKey(rmwMode);
    if (CheckAndSetRmwAttr(producerOp, rmwMode, rmwAttrKey) != SUCCESS) {
        return FAILED;
    }
    producerOp.SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
    auto producerAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp.GetOpAttribute());
    if (producerAttr != nullptr && AccumulateContractOffset(producerAttr, rmwOffset, rmwDynOffset) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::CheckAndSetRmwAttr(Operation& producerOp, AtomicRMWMode rmwMode, const std::string& rmwAttrKey)
{
    bool hasAdd = producerOp.HasAttr(RMW_MODE_ATTR_ADD);
    bool hasMax = producerOp.HasAttr(RMW_MODE_ATTR_MAX);
    bool hasMin = producerOp.HasAttr(RMW_MODE_ATTR_MIN);
    if (!hasAdd && !hasMax && !hasMin) {
        producerOp.SetAttribute(rmwAttrKey, 1L);
        return SUCCESS;
    }

    bool attrConflict = (rmwMode == AtomicRMWMode::ADD && !hasAdd) || (rmwMode == AtomicRMWMode::MAX && !hasMax) ||
                        (rmwMode == AtomicRMWMode::MIN && !hasMin);

    if (attrConflict) {
        std::string existingAttrType;
        if (hasAdd)
            existingAttrType = "atomic_add";
        else if (hasMax)
            existingAttrType = "atomic_max";
        else if (hasMin)
            existingAttrType = "atomic_min";

        APASS_LOG_ERROR_F(Elements::Operation,
                          "Op[%d] rmwMode conflict: producer contract op already has '%s' attribute, "
                          "but current wants to set '%s'. Cannot set different rmwMode to the same contract op.",
                          producerOp.GetOpMagic(), existingAttrType.c_str(), rmwAttrKey.c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessAtomic::AccumulateContractOffset(std::shared_ptr<AssembleOpAttribute> producerAttr,
                                               const std::vector<int64_t>& rmwOffset,
                                               const std::vector<SymbolicScalar>& rmwDynOffset)
{
    auto& producerOffset = producerAttr->GetToOffset();
    auto& producerDynOffset = producerAttr->GetToDynOffset();
    if (producerOffset.size() != rmwOffset.size() ||
        (!producerDynOffset.empty() && producerDynOffset.size() != producerOffset.size()) ||
        (!rmwDynOffset.empty() && rmwDynOffset.size() != rmwOffset.size())) {
        return FAILED;
    }

    auto originalOffset = producerOffset;
    if (!producerDynOffset.empty() || !rmwDynOffset.empty()) {
        auto lhs = producerDynOffset.empty() ? SymbolicScalar::FromConcrete(originalOffset) : producerDynOffset;
        auto rhs = rmwDynOffset.empty() ? SymbolicScalar::FromConcrete(rmwOffset) : rmwDynOffset;
        producerDynOffset.clear();
        producerDynOffset.reserve(producerOffset.size());
        for (size_t i = 0; i < producerOffset.size(); ++i) {
            producerDynOffset.push_back((lhs[i] + rhs[i]).Simplify());
        }
    }
    for (size_t i = 0; i < producerOffset.size(); ++i) {
        producerOffset[i] += rmwOffset[i];
    }
    return SUCCESS;
}

void ProcessAtomic::CollectReduceAccUpstream(Operation& op, std::set<int>& visited,
                                             std::vector<Operation*>& result) const
{
    if (visited.count(op.GetOpMagic()) > 0 || op.IsDeleted()) {
        return;
    }
    visited.insert(op.GetOpMagic());
    if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
        result.push_back(&op);
        return;
    }
    for (const auto& input : op.GetIOperands()) {
        if (input == nullptr) {
            continue;
        }
        for (auto* producer : input->GetProducers()) {
            if (producer == nullptr || producer->IsDeleted() || producer->GetOpcode() == Opcode::OP_ATOMIC_RMW) {
                continue;
            }
            CollectReduceAccUpstream(*producer, visited, result);
        }
    }
}

Status ProcessAtomic::TraceBackAndRemoveVecDup(Function& function, Operation& op, std::set<int>& visited,
                                               bool& anyRemoved)
{
    if (visited.count(op.GetOpMagic()) > 0 || op.IsDeleted()) {
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
        return SUCCESS;
    }
    visited.insert(op.GetOpMagic());

    if (op.GetOpcode() == Opcode::OP_A_MUL_B || op.GetOpcode() == Opcode::OP_A_MULACC_B) {
        if (RemoveVecDupBranchFromCubeOp(op, anyRemoved) != SUCCESS) {
            return FAILED;
        }
    }

    for (const auto& input : op.GetIOperands()) {
        if (input == nullptr) {
            continue;
        }
        for (auto* producer : input->GetProducers()) {
            if (producer == nullptr || producer->IsDeleted()) {
                continue;
            }
            if (TraceBackAndRemoveVecDup(function, *producer, visited, anyRemoved) != SUCCESS) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status ProcessAtomic::RemoveVecDupBranchFromCubeOp(Operation& cubeOp, bool& anyRemoved)
{
    auto inputsBackup = cubeOp.GetIOperands();
    for (const auto& input : inputsBackup) {
        auto producersBackup = input->GetProducers();
        for (auto* producer : producersBackup) {
            if (producer == nullptr || producer->IsDeleted()) {
                continue;
            }
            if (IsContractOpcode(producer->GetOpcode()) && IsVecDupContractInput(*producer)) {
                input->RemoveConsumer(&cubeOp);
                cubeOp.EraseInput(input);
                anyRemoved = true;
                break;
            }
        }
    }
    return SUCCESS;
}

bool ProcessAtomic::IsVecDupContractInput(const Operation& contractOp) const
{
    for (const auto& input : contractOp.GetIOperands()) {
        for (auto* producer : input->GetProducers()) {
            if (producer != nullptr && producer->GetOpcode() == Opcode::OP_VEC_DUP) {
                return true;
            }
        }
    }
    return false;
}

Status ProcessAtomic::EliminateVecDupBranch(Function& function, bool& hasReduceAccCascade)
{
    std::vector<Operation*> reduceAccOps;
    std::set<int> collectVisited;
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            CollectReduceAccUpstream(op, collectVisited, reduceAccOps);
        }
    }
    hasReduceAccCascade = !reduceAccOps.empty();
    if (reduceAccOps.empty()) {
        return SUCCESS;
    }

    bool anyRemoved = false;
    std::set<int> traceVisited;
    for (auto* reduceAccOp : reduceAccOps) {
        if (TraceBackAndRemoveVecDup(function, *reduceAccOp, traceVisited, anyRemoved) != SUCCESS) {
            return FAILED;
        }
    }

    if (!anyRemoved) {
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Function, "EliminateVecDupBranch removed VecDup contract input branch.");
    function.EraseOperations(true);
    if (DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Eliminate dead operation failed for VecDup branch.");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
