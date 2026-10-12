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
 * \file common_operation_eliminate_utils.cpp
 * \brief utils of common operation elimination
 */

#include "common_operation_eliminate_utils.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_utils/pass_utils.h"

#define MODULE_NAME "CommonOperationEliminateUtils"

namespace npu::tile_fwk {
const std::unordered_set<Opcode>& CommonOperationEliminateUtils::GetSkipEliminateOpcodes()
{
    // Opcodes in this set are excluded from common operation elimination on purpose:
    //   - OP_VIEW: works with GraphPartition processing logic;
    //   - OP_ASSEMBLE_SSA writes an explicit SSA destination and its version/token chain;
    //   - OP_ATOMIC_RMW writes external state and is not an ordinary common
    //     subexpression.
    static const std::unordered_set<Opcode> skipOpcodes = {
        Opcode::OP_VIEW,
        Opcode::OP_SLICE,
        Opcode::OP_ASSEMBLE_SSA,
        Opcode::OP_ATOMIC_RMW,
    };
    return skipOpcodes;
}

namespace {
using TokenRewritePair = std::pair<pypto::ir::VarPtr, pypto::ir::VarPtr>;

pypto::ir::VarPtr RewriteToken(const pypto::ir::VarPtr& token, const std::vector<TokenRewritePair>& tokenRewritePairs)
{
    for (const auto& tokenRewritePair : tokenRewritePairs) {
        if (token == tokenRewritePair.first) {
            return tokenRewritePair.second;
        }
    }
    return token;
}

bool TokenExists(const std::vector<pypto::ir::VarPtr>& tokens, const pypto::ir::VarPtr& targetToken)
{
    return std::find(tokens.begin(), tokens.end(), targetToken) != tokens.end();
}

bool SymbolicScalarsEqual(const std::vector<SymbolicScalar>& lhs, const std::vector<SymbolicScalar>& rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].Dump() != rhs[i].Dump()) {
            return false;
        }
    }
    return true;
}

bool TensorRedirectCompatible(const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor)
{
    return oldTensor != nullptr && newTensor != nullptr && oldTensor->tensor != nullptr &&
           newTensor->tensor != nullptr && oldTensor->shape == newTensor->shape &&
           oldTensor->offset == newTensor->offset &&
           oldTensor->tensor->GetDataType() == newTensor->tensor->GetDataType() &&
           SymbolicScalarsEqual(oldTensor->GetDynOffset(), newTensor->GetDynOffset()) &&
           SymbolicScalarsEqual(oldTensor->GetDynValidShape(), newTensor->GetDynValidShape());
}

bool HasDuplicateProducerKey(const std::vector<std::string>& producerKeys)
{
    std::unordered_set<std::string> visitedKeys;
    for (const auto& producerKey : producerKeys) {
        if (!visitedKeys.emplace(producerKey).second) {
            return true;
        }
    }
    return false;
}

void AddTokenIfAbsent(std::vector<pypto::ir::VarPtr>& tokens, const pypto::ir::VarPtr& token)
{
    if (token != nullptr && !TokenExists(tokens, token)) {
        tokens.emplace_back(token);
    }
}

pypto::ir::StmtPtr GetOperationStmtPtr(Operation& operation)
{
    return std::static_pointer_cast<const pypto::ir::Stmt>(operation.shared_from_this());
}

void UpdateOperationTokens(Function& function, Operation& operation, const std::vector<pypto::ir::VarPtr>& newTokens)
{
    auto oldTokens = operation.tokens_;
    if (oldTokens == newTokens) {
        return;
    }
    auto stmt = GetOperationStmtPtr(operation);
    auto& varDependency = function.GetVarDependency();
    auto removeIfUnused = [&varDependency](const pypto::ir::VarPtr& token) {
        if (token != nullptr && varDependency.GetProducers(token).empty() &&
            varDependency.GetConsumers(token).empty()) {
            varDependency.RemoveVar(token);
        }
    };
    for (const auto& oldToken : oldTokens) {
        if (oldToken != nullptr && !TokenExists(newTokens, oldToken)) {
            varDependency.RemoveConsumer(oldToken, stmt);
            removeIfUnused(oldToken);
        }
    }
    for (const auto& newToken : newTokens) {
        if (newToken != nullptr && !TokenExists(oldTokens, newToken)) {
            varDependency.AddConsumer(newToken, stmt);
        }
    }
    operation.tokens_ = newTokens;
}

void DetachOperationTokens(Function& function, Operation& operation)
{
    auto stmt = GetOperationStmtPtr(operation);
    auto& varDependency = function.GetVarDependency();
    auto removeIfUnused = [&varDependency](const pypto::ir::VarPtr& token) {
        if (token != nullptr && varDependency.GetProducers(token).empty() &&
            varDependency.GetConsumers(token).empty()) {
            varDependency.RemoveVar(token);
        }
    };
    for (const auto& token : operation.tokens_) {
        if (token != nullptr) {
            varDependency.RemoveConsumer(token, stmt);
            removeIfUnused(token);
        }
    }
    for (const auto& token : operation.result_token_) {
        if (token != nullptr) {
            varDependency.RemoveProducer(token, stmt);
            removeIfUnused(token);
        }
    }
    operation.tokens_.clear();
    operation.result_token_.clear();
}

bool HasControlDependencies(Function& function)
{
    for (const auto* operation : function.Operations(false).DuplicatedOpList()) {
        if (operation != nullptr && (!operation->tokens_.empty() || !operation->result_token_.empty() ||
                                     !operation->GetDependOperands().empty())) {
            return true;
        }
    }
    return false;
}

bool BuildResultTokenRewritePairs(const std::vector<Operation*>& oldProducers,
                                  const std::vector<Operation*>& newProducers,
                                  std::vector<TokenRewritePair>& tokenRewritePairs)
{
    if (oldProducers.size() != newProducers.size()) {
        return false;
    }
    for (size_t i = 0; i < oldProducers.size(); ++i) {
        Operation* oldProducer = oldProducers[i];
        Operation* newProducer = newProducers[i];
        if (oldProducer == nullptr || newProducer == nullptr) {
            return false;
        }
        const auto& oldTokens = oldProducer->result_token_;
        const auto& newTokens = newProducer->result_token_;
        if (oldTokens.size() != newTokens.size()) {
            return false;
        }
        for (size_t j = 0; j < oldTokens.size(); ++j) {
            const auto& oldToken = oldTokens[j];
            const auto& newToken = newTokens[j];
            if ((oldToken == nullptr) != (newToken == nullptr)) {
                return false;
            }
            if (oldToken == nullptr) {
                continue;
            }
            if (oldToken != newToken) {
                for (const auto& existingPair : tokenRewritePairs) {
                    if ((existingPair.first == oldToken && existingPair.second != newToken) ||
                        (existingPair.second == newToken && existingPair.first != oldToken)) {
                        return false;
                    }
                }
                tokenRewritePairs.emplace_back(oldToken, newToken);
            }
        }
    }
    return true;
}

bool BuildMergedProducerTokenInputs(const std::vector<Operation*>& oldProducers,
                                    const std::vector<Operation*>& newProducers,
                                    const std::vector<TokenRewritePair>& tokenRewritePairs,
                                    std::vector<std::vector<pypto::ir::VarPtr>>& mergedProducerTokens)
{
    if (oldProducers.size() != newProducers.size()) {
        return false;
    }
    mergedProducerTokens.clear();
    mergedProducerTokens.reserve(newProducers.size());
    for (size_t i = 0; i < oldProducers.size(); ++i) {
        Operation* oldProducer = oldProducers[i];
        Operation* newProducer = newProducers[i];
        if (oldProducer == nullptr || newProducer == nullptr) {
            return false;
        }
        const auto& newResultTokens = newProducer->result_token_;
        std::vector<pypto::ir::VarPtr> mergedTokens;
        for (const auto& token : newProducer->tokens_) {
            auto rewrittenToken = RewriteToken(token, tokenRewritePairs);
            if (TokenExists(newResultTokens, rewrittenToken)) {
                return false;
            }
            AddTokenIfAbsent(mergedTokens, rewrittenToken);
        }
        for (const auto& token : oldProducer->tokens_) {
            auto rewrittenToken = RewriteToken(token, tokenRewritePairs);
            if (TokenExists(newResultTokens, rewrittenToken)) {
                return false;
            }
            AddTokenIfAbsent(mergedTokens, rewrittenToken);
        }
        mergedProducerTokens.emplace_back(std::move(mergedTokens));
    }
    return true;
}

bool BuildRewrittenTokenInputs(const Operation& operation, const std::vector<TokenRewritePair>& tokenRewritePairs,
                               std::vector<pypto::ir::VarPtr>& rewrittenTokens)
{
    if (tokenRewritePairs.empty() || operation.tokens_.empty()) {
        return false;
    }
    bool changed = false;
    rewrittenTokens.clear();
    for (const auto& token : operation.tokens_) {
        auto rewrittenToken = RewriteToken(token, tokenRewritePairs);
        if (rewrittenToken != token) {
            changed = true;
        }
        AddTokenIfAbsent(rewrittenTokens, rewrittenToken);
    }
    return changed;
}

void UpdateMergedProducerTokens(Function& function, const std::vector<Operation*>& producers,
                                const std::vector<std::vector<pypto::ir::VarPtr>>& mergedProducerTokens)
{
    for (size_t i = 0; i < producers.size(); ++i) {
        UpdateOperationTokens(function, *producers[i], mergedProducerTokens[i]);
    }
}

void RewriteFunctionTokenInputs(Function& function, const std::vector<TokenRewritePair>& tokenRewritePairs)
{
    if (tokenRewritePairs.empty()) {
        return;
    }
    for (auto op : function.Operations(true).DuplicatedOpList()) {
        std::vector<pypto::ir::VarPtr> rewrittenTokens;
        if (op != nullptr && BuildRewrittenTokenInputs(*op, tokenRewritePairs, rewrittenTokens)) {
            UpdateOperationTokens(function, *op, rewrittenTokens);
        }
    }
}

bool LogicalTensorExists(const std::vector<LogicalTensorPtr>& tensors, const LogicalTensorPtr& target)
{
    return std::find(tensors.begin(), tensors.end(), target) != tensors.end();
}

LogicalTensorPtr RewriteTensor(const LogicalTensorPtr& tensor,
                               const std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>>& tensorRedirectPairs)
{
    for (const auto& [oldTensor, newTensor] : tensorRedirectPairs) {
        if (tensor == oldTensor) {
            return newTensor;
        }
    }
    return tensor;
}

bool BuildMergedProducerDependOperands(
    const std::vector<Operation*>& oldProducers, const std::vector<Operation*>& newProducers,
    const std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>>& tensorRedirectPairs,
    std::vector<std::vector<LogicalTensorPtr>>& mergedDependOperands)
{
    if (oldProducers.size() != newProducers.size()) {
        return false;
    }
    mergedDependOperands.clear();
    mergedDependOperands.reserve(newProducers.size());
    for (size_t i = 0; i < oldProducers.size(); ++i) {
        if (oldProducers[i] == nullptr || newProducers[i] == nullptr) {
            return false;
        }
        std::vector<LogicalTensorPtr> merged;
        auto addDependOperand = [&](const LogicalTensorPtr& depend) {
            if (depend == nullptr) {
                return false;
            }
            auto rewrittenDepend = RewriteTensor(depend, tensorRedirectPairs);
            if (rewrittenDepend == depend) {
                for (const auto* producer : oldProducers) {
                    if (producer == nullptr) {
                        continue;
                    }
                    const auto& outputs = producer->GetOOperands();
                    if (std::find(outputs.begin(), outputs.end(), depend) != outputs.end()) {
                        return false;
                    }
                }
            }
            if (!LogicalTensorExists(merged, rewrittenDepend)) {
                merged.emplace_back(rewrittenDepend);
            }
            return true;
        };
        for (const auto& depend : newProducers[i]->GetDependOperands()) {
            if (!addDependOperand(depend)) {
                return false;
            }
        }
        for (const auto& depend : oldProducers[i]->GetDependOperands()) {
            if (!addDependOperand(depend)) {
                return false;
            }
        }
        mergedDependOperands.emplace_back(std::move(merged));
    }
    return true;
}

void UpdateMergedProducerDependOperands(const std::vector<Operation*>& producers,
                                        const std::vector<std::vector<LogicalTensorPtr>>& mergedDependOperands)
{
    for (size_t i = 0; i < producers.size(); ++i) {
        auto& operation = *producers[i];
        const auto oldDependOperands = operation.GetDependOperands();
        for (const auto& depend : oldDependOperands) {
            if (!LogicalTensorExists(mergedDependOperands[i], depend)) {
                depend->RemoveDependOp(operation);
            }
        }
        for (const auto& depend : mergedDependOperands[i]) {
            if (!LogicalTensorExists(oldDependOperands, depend)) {
                depend->AddDependOp(operation);
            }
        }
        operation.GetDependOperands() = mergedDependOperands[i];
    }
}

} // namespace

std::string CommonOperationEliminateUtils::DumpAttr(const Operation& op)
{
    // Excluded attribute keys: graph-merge attributes are annotated by merge passes (L1CopyInReuse /
    // CubeNBuffer / VecNBuffer) with merge-group identifiers (e.g. "func<magic>_<order>") that differ
    // across equivalent producer groups, so they must not participate in the common-operation hash.
    static const std::unordered_set<std::string> excludedAttrKeys = {
        OpAttributeKey::l1ReuseHashOrder,   OpAttributeKey::l1ReuseSubgraphCount,
        OpAttributeKey::cubeMergeHashOrder, OpAttributeKey::cubeMergeSubgraphCount,
        OpAttributeKey::vecMergeHashOrder,  OpAttributeKey::vecMergeSubgraphCount,
    };
    std::ostringstream oss;
    size_t index = 0;
    for (const auto& attr : op.GetAllAttr()) {
        if (excludedAttrKeys.count(attr.first) != 0) {
            continue;
        }
        oss << ((index++ == 0) ? "" : " ");
        oss << "#" << attr.first << "{" << op.DumpAttr(attr.first) << "}";
    }
    return oss.str();
}

void CommonOperationEliminateUtils::SortedProducer(std::vector<Operation*>& sortedProducers) const
{
    // Keep the original producer order for ties so hash generation stays deterministic.
    std::stable_sort(sortedProducers.begin(), sortedProducers.end(), [](const Operation* op1, const Operation* op2) {
        const auto& iOp1 = op1->GetIOperands();
        const auto& iOp2 = op2->GetIOperands();
        size_t minLen = std::min(iOp1.size(), iOp2.size());
        for (size_t i = 0; i < minLen; ++i) {
            LogicalTensorPtr ptr1 = iOp1[i];
            LogicalTensorPtr ptr2 = iOp2[i];
            if (ptr1 != ptr2) {
                return ptr1 < ptr2;
            }
        }
        if (iOp1.size() != iOp2.size()) {
            return iOp1.size() < iOp2.size();
        }
        std::stringstream ss1, ss2;
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op1->GetOpcode())) {
            ss1 << " attr: [" << attr << " : " << op1->DumpAttr(attr) << "]";
        }
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op2->GetOpcode())) {
            ss2 << " attr: [" << attr << " : " << op2->DumpAttr(attr) << "]";
        }
        return ss1.str() < ss2.str();
    });
}

void CommonOperationEliminateUtils::CollectProducerInfo(const std::vector<Operation*>& sortedProducers,
                                                        const LogicalTensorPtr& curTensor,
                                                        std::vector<std::string>& opStrList,
                                                        std::stringstream& ss) const
{
    for (const auto& op : sortedProducers) {
        if (op == nullptr) {
            continue;
        }
        ss.str("");
        ss.clear();
        ss << op->GetOpcodeStr(true);
        for (const auto& iOperands : op->GetIOperands()) {
            if (iOperands == nullptr || iOperands->tensor == nullptr) {
                continue;
            }
            ss << "[i";
            ss << "$" << iOperands->tensor->DumpSSA(false, false);
            // Logical tensors sharing one raw tensor can represent different
            // assemble versions. Keep those versions distinct for CSE.
            const auto rawTensorIt = tensorsByRawMagic_.find(iOperands->GetRawMagic());
            if (rawTensorIt != tensorsByRawMagic_.end() && rawTensorIt->second.size() > 1U) {
                ss << " logicalMagic" << iOperands->GetMagic();
            }
            ss << iOperands->DumpType();
            ss << "(";
            for (size_t i = 0; i < iOperands->offset.size(); ++i) {
                ss << iOperands->offset[i];
                if (i != iOperands->offset.size() - 1) {
                    ss << ", ";
                }
            }
            if (curTensor && !curTensor->GetDynValidShape().empty()) {
                std::string shapeStr;
                for (size_t i = 0; i < curTensor->GetDynValidShape().size(); i++) {
                    shapeStr += curTensor->GetDynValidShape()[i].Dump();
                }
                ss << "[" << shapeStr << "]";
                ss << "memoryType: [" << MemoryTypeToString(curTensor->GetMemoryTypeOriginal()) << "]";
            }
            ss << ")]";
        }
        if (op->GetOpAttribute() != nullptr) {
            ss << " " << op->GetOpAttribute()->Dump();
        }
        std::string opAttrStr = DumpAttr(*op);
        if (!opAttrStr.empty()) {
            ss << " " << opAttrStr;
        }
        for (const auto& attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : " << op->DumpAttr(attr) << "]";
        }
        ss << "id" << op->GetSubgraphID();
        opStrList.emplace_back(ss.str());
    }
    ss.str("");
    ss.clear();
    for (const auto& str : opStrList) {
        ss << str;
    }
}

unsigned long CommonOperationEliminateUtils::ComputeHash(const std::vector<Operation*>& producers,
                                                         LogicalTensorPtr curTensor) const
{
    std::vector<std::string> opStrList;
    std::stringstream ss;
    std::vector<Operation*> sortedProducers = producers;
    SortedProducer(sortedProducers);
    CollectProducerInfo(sortedProducers, curTensor, opStrList, ss);
    std::hash<std::string> hasher;
    return hasher(ss.str());
}

Status CommonOperationEliminateUtils::EliminateCommonOperation(Function& function)
{
    CommonOperationEliminateUtils commonOperationEliminateUtils;
    return commonOperationEliminateUtils.Process(function);
}

Status CommonOperationEliminateUtils::Process(Function& function)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    std::vector<LogicalTensorPtr> sequence;
    auto tensorProducerMap = GetTensorProducers(function, sequence);
    mixSubgraphIds_ = GetMixSubgraphIds(function);
    function_ = &function;
    tensorsByRawMagic_ = GraphUtils::GetTensorsGroupedByRawMagic(function);
    processedRawMagicTensors_.clear();
    hashCache_.clear();
    std::unordered_set<Operation*> cacheProducers;
    bool anyDeleted = false;
    for (auto& orderedTensor : sequence) {
        if (orderedTensor == nullptr || processedRawMagicTensors_.count(orderedTensor->GetRawMagic()) != 0) {
            continue;
        }
        auto& producerGroup = tensorProducerMap[orderedTensor];
        std::vector<Operation*> oldBucketProducers;
        if (producerGroup.empty() ||
            !TensorProducersMerge(function, orderedTensor, cacheProducers, tensorProducerMap, oldBucketProducers)) {
            continue;
        }
        anyDeleted = true;
        MarkRedundantProducersDeleted(function, oldBucketProducers, cacheProducers);
    }
    if (anyDeleted) {
        function.EraseOperations(true, false);
        if (!HasControlDependencies(function) && DeadOperationEliminator::EliminateDeadOperation(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Eliminate dead operation failed in CommonOperationEliminateUtils.");
            return FAILED;
        }
    }
    return SUCCESS;
}

std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> CommonOperationEliminateUtils::GetTensorProducers(
    Function& function, std::vector<LogicalTensorPtr>& sequence)
{
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> tensorProducerMap;
    std::unordered_set<int> visitedTensors;
    auto allOps = function.Operations(true).DuplicatedOpList();
    for (const auto& op : allOps) {
        if (op == nullptr) {
            continue;
        }
        auto& outputTensors = op->GetOOperands();
        for (const auto& tensor : outputTensors) {
            if (tensor == nullptr || visitedTensors.count(tensor->GetMagic())) {
                continue;
            }
            visitedTensors.insert(tensor->GetMagic());
            for (const auto& producer : tensor->GetProducers()) {
                if (producer == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation, "Producer operation nullptr for Tensor[%d].",
                                      tensor->GetMagic());
                    continue;
                }
                if (tensorProducerMap.count(tensor) == 0) {
                    sequence.push_back(tensor);
                }
                tensorProducerMap[tensor].push_back(producer);
            }
        }
    }
    return tensorProducerMap;
}

const TensorSet& CommonOperationEliminateUtils::GetSameRawMagicTensors(const LogicalTensorPtr& tensor) const
{
    static const TensorSet empty;
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return empty;
    }
    const auto iter = tensorsByRawMagic_.find(tensor->GetRawMagic());
    return iter == tensorsByRawMagic_.end() ? empty : iter->second;
}

void CommonOperationEliminateUtils::MarkRedundantProducersDeleted(Function& function,
                                                                  const std::vector<Operation*>& oldBucketProducers,
                                                                  std::unordered_set<Operation*>& cacheProducers)
{
    for (auto* op : oldBucketProducers) {
        if (op == nullptr) {
            continue;
        }
        if (cacheProducers.count(op) == 0) {
            DetachOperationTokens(function, *op);
            op->SetAsDeleted();
        }
    }
}

std::vector<Operation*> CommonOperationEliminateUtils::CollectConsumersWithSameRawMagic(
    Function& function, const LogicalTensorPtr& tensor) const
{
    std::vector<Operation*> result;
    std::unordered_set<Operation*> seen;
    for (const auto& sameRawTensor : GetSameRawMagicTensors(tensor)) {
        if (sameRawTensor == nullptr) {
            continue;
        }
        for (auto* consumer : sameRawTensor->GetConsumers()) {
            if (consumer == nullptr || consumer->IsDeleted() || consumer->BelongTo() != &function) {
                continue;
            }
            if (seen.insert(consumer).second) {
                result.emplace_back(consumer);
            }
        }
    }
    return result;
}

bool CommonOperationEliminateUtils::ShouldSkipProducers(const std::vector<Operation*>& producers) const
{
    for (auto* operation : producers) {
        if (operation == nullptr) {
            continue;
        }
        const auto& inputsMemType = OpcodeManager::Inst().GetInputsMemType(operation->GetOpcode());
        const auto& outputsMemType = OpcodeManager::Inst().GetOutputsMemType(operation->GetOpcode());
        const auto opCalcType = OpcodeManager::Inst().GetOpCalcType(operation->GetOpcode());
        const bool inputCheck = inputsMemType.size() == 1 && inputsMemType[0] == MemoryType::MEM_L1;
        const bool calcTypeCheck = opCalcType == OpCalcType::MOVE_LOCAL || opCalcType == OpCalcType::MOVE_IN;
        const bool outputCheck = outputsMemType.size() == 1 && outputsMemType[0] != MemoryType::MEM_L1;
        if ((inputCheck && calcTypeCheck && outputCheck) ||
            GetSkipEliminateOpcodes().count(operation->GetOpcode()) != 0 || IsCopyOut(operation->GetOpcode()) ||
            operation->GetBoolAttribute(OpAttributeKey::dontTouch)) {
            return true;
        }
    }
    return false;
}

void CommonOperationEliminateUtils::AugmentProducersWithSiblings(const LogicalTensorPtr& orderedTensor,
                                                                 std::vector<Operation*>& augmentedProducers)
{
    if (function_ == nullptr || orderedTensor == nullptr) {
        return;
    }
    for (const auto& sameRawTensor : GetSameRawMagicTensors(orderedTensor)) {
        if (sameRawTensor == nullptr) {
            continue;
        }
        for (auto* producer : sameRawTensor->GetProducers()) {
            if (producer != nullptr && !producer->IsDeleted() && producer->BelongTo() == function_ &&
                std::find(augmentedProducers.begin(), augmentedProducers.end(), producer) == augmentedProducers.end()) {
                augmentedProducers.emplace_back(producer);
            }
        }
    }
    processedRawMagicTensors_.insert(orderedTensor->GetRawMagic());
}

std::pair<LogicalTensorPtr, std::vector<Operation*>> CommonOperationEliminateUtils::TensorHashExist(
    const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
    const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap,
    std::vector<Operation*>& currentBucketProducers)
{
    const auto producerIt = tensorProducerMap.find(orderedTensor);
    if (producerIt == tensorProducerMap.end()) {
        return {nullptr, {}};
    }
    currentBucketProducers = producerIt->second;
    AugmentProducersWithSiblings(orderedTensor, currentBucketProducers);
    if (ShouldSkipProducers(currentBucketProducers)) {
        return {nullptr, {}};
    }
    const uint64_t groupHash = ComputeHash(currentBucketProducers, orderedTensor);
    if (hashCache_.count(groupHash) != 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d] are marked as hash already existed tensor.",
                          orderedTensor->GetMagic());
        return hashCache_[groupHash];
    }
    hashCache_.emplace(groupHash, std::make_pair(orderedTensor, currentBucketProducers));
    for (auto producer : currentBucketProducers) {
        if (producer != nullptr) {
            cacheProducers.insert(producer);
        }
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Tensor[%d] hash already existed.", orderedTensor->GetMagic());
    return {nullptr, {}};
}

void CommonOperationEliminateUtils::UpdateView(ViewOpAttribute* viewOpAttribute,
                                               const std::shared_ptr<LogicalTensor> oldTensor,
                                               const std::shared_ptr<LogicalTensor> newTensor) const
{
    auto& fromOffset = viewOpAttribute->GetFromOffset();
    for (size_t j = 0; j < fromOffset.size(); j++) {
        fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
    }
}

void CommonOperationEliminateUtils::UpdateCopy(CopyOpAttribute* copyOpAttribute,
                                               const std::shared_ptr<LogicalTensor> oldTensor,
                                               const std::shared_ptr<LogicalTensor> newTensor) const
{
    if (!copyOpAttribute->IsCopyOut()) {
        auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
        (void)memType;
        for (size_t j = 0; j < fromOffset.size(); j++) {
            fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
        }
        copyOpAttribute->SetFromOffset(fromOffset);
    }
}

bool CommonOperationEliminateUtils::ProducerPairingValid(LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor,
                                                         const std::vector<Operation*>& oldProducers,
                                                         const std::vector<Operation*>& newProducers) const
{
    std::vector<std::string> oldProducerKeys;
    std::vector<std::string> newProducerKeys;
    std::stringstream oldProducerStream;
    std::stringstream newProducerStream;
    CollectProducerInfo(oldProducers, oldTensor, oldProducerKeys, oldProducerStream);
    CollectProducerInfo(newProducers, newTensor, newProducerKeys, newProducerStream);
    return oldProducerKeys == newProducerKeys && !HasDuplicateProducerKey(oldProducerKeys) &&
           !HasDuplicateProducerKey(newProducerKeys);
}

void CommonOperationEliminateUtils::UpdateTensorConsumers(LogicalTensorPtr oldTensor, LogicalTensorPtr newTensor) const
{
    auto consumers = oldTensor->GetConsumers();
    for (auto& cur : consumers) {
        if (cur == nullptr) {
            continue;
        }
        cur->ReplaceInput(newTensor, oldTensor);
        auto attrPtr = cur->GetOpAttribute().get();
        if (attrPtr == nullptr) {
            continue;
        }
        if (cur->GetOpcode() == Opcode::OP_VIEW) {
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(attrPtr)) {
                UpdateView(viewOpAttribute, oldTensor, newTensor);
                continue;
            }
        } else if (cur->GetOpcode() == Opcode::OP_COPY_IN) {
            if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(attrPtr)) {
                UpdateCopy(copyOpAttribute, oldTensor, newTensor);
                continue;
            }
        }
    }
}

bool CommonOperationEliminateUtils::UpdateConnection(Function& function, LogicalTensorPtr oldTensor,
                                                     LogicalTensorPtr newTensor,
                                                     const std::vector<Operation*>& oldProducers,
                                                     const std::vector<Operation*>& newProducers)
{
    std::vector<Operation*> sortedOldProducers = oldProducers;
    std::vector<Operation*> sortedNewProducers = newProducers;
    SortedProducer(sortedOldProducers);
    SortedProducer(sortedNewProducers);
    if (!ProducerPairingValid(oldTensor, newTensor, sortedOldProducers, sortedNewProducers)) {
        return false;
    }

    std::vector<TokenRewritePair> tokenRewritePairs;
    if (!BuildResultTokenRewritePairs(sortedOldProducers, sortedNewProducers, tokenRewritePairs)) {
        return false;
    }
    std::vector<std::vector<pypto::ir::VarPtr>> mergedProducerTokens;
    if (!BuildMergedProducerTokenInputs(sortedOldProducers, sortedNewProducers, tokenRewritePairs,
                                        mergedProducerTokens)) {
        return false;
    }

    std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>> tensorRedirectPairs;
    if (!BuildOutputTensorRedirectPairs(function, oldTensor, newTensor, sortedOldProducers, sortedNewProducers,
                                        tensorRedirectPairs)) {
        return false;
    }
    std::vector<std::vector<LogicalTensorPtr>> mergedDependOperands;
    if (!BuildMergedProducerDependOperands(sortedOldProducers, sortedNewProducers, tensorRedirectPairs,
                                           mergedDependOperands)) {
        return false;
    }

    for (const auto& [oldOutput, newOutput] : tensorRedirectPairs) {
        UpdateTensorConsumers(oldOutput, newOutput);
    }
    UpdateMergedProducerDependOperands(sortedNewProducers, mergedDependOperands);
    UpdateMergedProducerTokens(function, sortedNewProducers, mergedProducerTokens);
    RewriteFunctionTokenInputs(function, tokenRewritePairs);
    return true;
}

uint32_t CommonOperationEliminateUtils::GetTensorCoreFlag(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr) {
        return 0;
    }
    constexpr uint32_t kAicTensorFlag = 1U;
    constexpr uint32_t kAivTensorFlag = 2U;
    switch (tensor->GetMemoryTypeOriginal()) {
        case MemoryType::MEM_L1:
        case MemoryType::MEM_L0A:
        case MemoryType::MEM_L0B:
        case MemoryType::MEM_L0C:
            return kAicTensorFlag;
        case MemoryType::MEM_UB:
            return kAivTensorFlag;
        default:
            return 0;
    }
}

void CommonOperationEliminateUtils::CollectSubgraphIds(const std::set<Operation*, LogicalTensor::CompareOp>& ops,
                                                       std::unordered_set<int>& subgraphIds) const
{
    for (const auto& op : ops) {
        if (op == nullptr) {
            continue;
        }
        if (op->GetSubgraphID() >= 0) {
            subgraphIds.insert(op->GetSubgraphID());
        }
    }
}

void CommonOperationEliminateUtils::UpdateInternalTensorCoreFlag(
    const LogicalTensorPtr& tensor, std::unordered_map<int, uint32_t>& subgraphCoreFlags) const
{
    uint32_t tensorFlag = GetTensorCoreFlag(tensor);
    if (tensorFlag == 0) {
        return;
    }

    std::unordered_set<int> producerSubgraphIds;
    std::unordered_set<int> consumerSubgraphIds;
    CollectSubgraphIds(tensor->GetProducers(), producerSubgraphIds);
    CollectSubgraphIds(tensor->GetConsumers(), consumerSubgraphIds);
    for (const auto producerSubgraphId : producerSubgraphIds) {
        if (consumerSubgraphIds.count(producerSubgraphId) != 0) {
            subgraphCoreFlags[producerSubgraphId] |= tensorFlag;
        }
    }
}

std::unordered_set<int> CommonOperationEliminateUtils::GetMixSubgraphIds(Function& function) const
{
    constexpr uint32_t kAicTensorFlag = 1U;
    constexpr uint32_t kAivTensorFlag = 2U;
    std::unordered_map<int, uint32_t> internalTensorFlagsBySubgraph;
    std::unordered_set<int> handledTensorMagics;
    for (const auto& opPtr : function.Operations(true).DuplicatedOpList()) {
        if (opPtr == nullptr) {
            continue;
        }
        for (const auto& outputTensor : opPtr->GetOOperands()) {
            if (outputTensor == nullptr) {
                continue;
            }
            if (!handledTensorMagics.insert(outputTensor->GetMagic()).second) {
                continue;
            }
            UpdateInternalTensorCoreFlag(outputTensor, internalTensorFlagsBySubgraph);
        }
    }

    std::unordered_set<int> result;
    for (const auto& entry : internalTensorFlagsBySubgraph) {
        if ((entry.second & kAicTensorFlag) == 0 || (entry.second & kAivTensorFlag) == 0) {
            continue;
        }
        result.insert(entry.first);
    }
    return result;
}

MemoryType CommonOperationEliminateUtils::GetAssembleInputMemoryType(Operation* consumer,
                                                                     const LogicalTensorPtr& input) const
{
    if (input != nullptr && input->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return input->GetMemoryTypeOriginal();
    }
    if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
        return MemoryType::MEM_UNKNOWN;
    }
    auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(consumer->GetOpAttribute());
    if (assembleAttr == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    return assembleAttr->GetFrom();
}

MemoryType CommonOperationEliminateUtils::GetAssembleOutputMemoryType(Operation* consumer) const
{
    if (consumer == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    const auto& outputs = consumer->GetOOperands();
    if (!outputs.empty() && outputs.front() != nullptr &&
        outputs.front()->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return outputs.front()->GetMemoryTypeOriginal();
    }
    return MemoryType::MEM_UNKNOWN;
}

bool CommonOperationEliminateUtils::HasSameMemoryAssembleUse(const LogicalTensorPtr& tensor,
                                                             std::unordered_set<int>& visitedTensorMagics) const
{
    if (tensor == nullptr || !visitedTensorMagics.insert(tensor->GetMagic()).second) {
        return false;
    }
    for (const auto& consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        MemoryType inputType = GetAssembleInputMemoryType(consumer, tensor);
        MemoryType outputType = GetAssembleOutputMemoryType(consumer);
        if (inputType == MemoryType::MEM_UNKNOWN || outputType == MemoryType::MEM_UNKNOWN || inputType == outputType) {
            return true;
        }
        for (const auto& output : consumer->GetOOperands()) {
            if (HasSameMemoryAssembleUse(output, visitedTensorMagics)) {
                return true;
            }
        }
    }
    return false;
}

bool CommonOperationEliminateUtils::ShouldSkipSameMemoryAssembleMerge(const LogicalTensorPtr& oldTensor) const
{
    if (function_ != nullptr && oldTensor != nullptr) {
        // MR 4503 represents multiple writes to one RawTensor with distinct
        // LogicalTensors.  Such a bucket must go through the multi-assemble
        // merge path even when the assemble memory types are equal.
        const auto& sameRawTensors = GetSameRawMagicTensors(oldTensor);
        if (sameRawTensors.size() > 1) {
            return false;
        }
    }
    std::unordered_set<int> visitedTensorMagics;
    if (HasSameMemoryAssembleUse(oldTensor, visitedTensorMagics)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip eliminating Tensor[%d] because it has same-memory assemble consumers.",
                          oldTensor->GetMagic());
        return true;
    }
    return false;
}

bool CommonOperationEliminateUtils::BuildOutputTensorRedirectPairs(
    Function& function, const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor,
    const std::vector<Operation*>& sortedOldProducers, const std::vector<Operation*>& sortedNewProducers,
    std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>>& tensorRedirectPairs) const
{
    if (oldTensor == nullptr || newTensor == nullptr || oldTensor->tensor == nullptr || newTensor->tensor == nullptr ||
        sortedOldProducers.size() != sortedNewProducers.size()) {
        return false;
    }
    tensorRedirectPairs.clear();
    std::unordered_map<int, LogicalTensorPtr> oldToNew;
    std::unordered_map<int, LogicalTensorPtr> newToOld;
    for (size_t i = 0; i < sortedOldProducers.size(); ++i) {
        const auto* oldProducer = sortedOldProducers[i];
        const auto* newProducer = sortedNewProducers[i];
        if (oldProducer == nullptr || newProducer == nullptr ||
            oldProducer->GetOOperands().size() != newProducer->GetOOperands().size()) {
            return false;
        }
        for (size_t j = 0; j < oldProducer->GetOOperands().size(); ++j) {
            const auto& oldOutput = oldProducer->GetOOperands()[j];
            const auto& newOutput = newProducer->GetOOperands()[j];
            if (oldOutput == nullptr || newOutput == nullptr || oldOutput->tensor == nullptr ||
                newOutput->tensor == nullptr) {
                return false;
            }
            if (function.IsFromOutCast(oldOutput) || !oldOutput->GetDependOps().empty() ||
                !TensorRedirectCompatible(oldOutput, newOutput)) {
                return false;
            }
            const auto oldIt = oldToNew.find(oldOutput->GetMagic());
            if (oldIt != oldToNew.end()) {
                if (oldIt->second != newOutput) {
                    return false;
                }
                continue;
            }
            const auto newIt = newToOld.find(newOutput->GetMagic());
            if (newIt != newToOld.end() && newIt->second != oldOutput) {
                return false;
            }
            oldToNew.emplace(oldOutput->GetMagic(), newOutput);
            newToOld.emplace(newOutput->GetMagic(), oldOutput);
            tensorRedirectPairs.emplace_back(oldOutput, newOutput);
        }
    }

    for (const auto& oldSibling : GetSameRawMagicTensors(oldTensor)) {
        if (oldSibling == nullptr) {
            continue;
        }
        if (!oldSibling->GetDependOps().empty()) {
            return false;
        }
        if (oldSibling->GetConsumers().empty()) {
            continue;
        }
        if (oldToNew.count(oldSibling->GetMagic()) == 0) {
            return false;
        }
    }
    return !tensorRedirectPairs.empty();
}

bool CommonOperationEliminateUtils::WouldExposeMixInternalTensorAfterMerge(
    const LogicalTensorPtr& oldTensor, const LogicalTensorPtr& newTensor,
    const std::unordered_set<int>& mixSubgraphIds) const
{
    if (oldTensor == nullptr || newTensor == nullptr || mixSubgraphIds.empty()) {
        return false;
    }

    std::unordered_set<int> producerMixSubgraphIds;
    for (const auto& producer : newTensor->GetProducers()) {
        if (producer != nullptr && mixSubgraphIds.count(producer->GetSubgraphID()) != 0) {
            producerMixSubgraphIds.insert(producer->GetSubgraphID());
        }
    }
    for (const auto mixSubgraphId : producerMixSubgraphIds) {
        bool hasConsumerInMix = false;
        bool hasOtherSubgraphOp = false;
        auto updateConsumerSubgraphUse = [&](Operation* op) {
            if (op == nullptr) {
                return;
            }
            if (op->GetSubgraphID() == mixSubgraphId) {
                hasConsumerInMix = true;
            } else {
                hasOtherSubgraphOp = true;
            }
        };
        for (const auto& producer : newTensor->GetProducers()) {
            if (producer != nullptr && producer->GetSubgraphID() != mixSubgraphId) {
                hasOtherSubgraphOp = true;
            }
        }
        for (const auto& consumer : newTensor->GetConsumers()) {
            updateConsumerSubgraphUse(consumer);
        }
        for (const auto& consumer : oldTensor->GetConsumers()) {
            updateConsumerSubgraphUse(consumer);
        }
        if (hasConsumerInMix && hasOtherSubgraphOp) {
            return true;
        }
    }
    return false;
}

bool CommonOperationEliminateUtils::TensorProducersMerge(
    Function& function, const LogicalTensorPtr orderedTensor, std::unordered_set<Operation*>& cacheProducers,
    const std::unordered_map<LogicalTensorPtr, std::vector<Operation*>>& tensorProducerMap,
    std::vector<Operation*>& oldBucketProducers)
{
    auto& producers = tensorProducerMap.at(orderedTensor);
    if (producers.empty()) {
        return false;
    }
    oldBucketProducers.clear();
    auto existOp = TensorHashExist(orderedTensor, cacheProducers, tensorProducerMap, oldBucketProducers);
    if (existOp.first == nullptr || orderedTensor == nullptr || existOp.second.empty()) {
        return false;
    }
    if (orderedTensor->shape != existOp.first->shape) {
        return false;
    }
    if (orderedTensor->tensor->GetDataType() != existOp.first->tensor->GetDataType()) {
        return false;
    }
    LogicalTensorPtr oldTensor = orderedTensor;
    LogicalTensorPtr newTensor = existOp.first;
    for (const auto& sameRawTensor : GetSameRawMagicTensors(oldTensor)) {
        if (sameRawTensor != nullptr && function.IsFromOutCast(sameRawTensor)) {
            return false;
        }
    }
    if (oldBucketProducers.size() == existOp.second.size()) {
        bool allSame = true;
        std::vector<Operation*> sortedOldProducers = oldBucketProducers;
        std::vector<Operation*> sortedNewProducers = existOp.second;
        SortedProducer(sortedOldProducers);
        SortedProducer(sortedNewProducers);
        for (size_t i = 0; i < sortedNewProducers.size() && allSame; i++) {
            allSame = (sortedOldProducers[i] == sortedNewProducers[i]);
        }
        if (allSame) {
            return false;
        }
    }
    const auto oldConsumers = CollectConsumersWithSameRawMagic(function, oldTensor);
    const auto newConsumers = CollectConsumersWithSameRawMagic(function, newTensor);
    if (newConsumers.empty() || oldConsumers.empty()) {
        return false;
    }
    if (ShouldSkipSameMemoryAssembleMerge(oldTensor)) {
        return false;
    }
    if (WouldExposeMixInternalTensorAfterMerge(oldTensor, newTensor, mixSubgraphIds_)) {
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Skip eliminating Tensor[%d] to avoid exposing mix subgraph internal Tensor[%d] to other subgraphs.",
            oldTensor->GetMagic(), newTensor->GetMagic());
        return false;
    }
    if (!UpdateConnection(function, oldTensor, newTensor, oldBucketProducers, existOp.second)) {
        return false;
    }
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "In CommonOperationEliminateUtils, Tensor[%d] and producersgroup are marked as redundant.",
                      oldTensor->GetMagic());
    return true;
}
} // namespace npu::tile_fwk
