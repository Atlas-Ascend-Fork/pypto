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
 * \file insert_sync.cpp
 * \brief
 */

#include "passes/block_graph_pass/insert_sync.h"
#include <algorithm>
#include <functional>
#include <thread>
#include <unordered_set>
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/pass_common_defs.h"

#define MODULE_NAME "InsertSync"

namespace npu {
namespace tile_fwk {
Status RangeSearchTree::ProcessTreeNode(const Interval& interval, IntervalTreeNode* currPtr,
                                        std::vector<IntervalTreeNode*>& intervalStack)
{
    int start = currPtr->interval.start;
    if (interval.start < start) {
        if (currPtr->left != nullptr) {
            intervalStack.push_back(currPtr->left);
            return SUCCESS;
        }
        currPtr->left = new IntervalTreeNode(interval);
        if (currPtr->left == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "New created left tree node is nullptr, ProcessTreeNode failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    if (currPtr->right != nullptr) {
        intervalStack.push_back(currPtr->right);
        return SUCCESS;
    }
    currPtr->right = new IntervalTreeNode(interval);
    if (currPtr->right == nullptr) {
        APASS_LOG_ERROR_F(Elements::Tensor, "New created right tree node is nullptr, ProcessTreeNode failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status RangeSearchTree::InsertInterval(const Interval& interval)
{
    std::vector<IntervalTreeNode*> intervalStack;
    if (treeRoot == nullptr) {
        treeRoot = new IntervalTreeNode(interval);
        if (treeRoot == nullptr) {
            APASS_LOG_ERROR_F(Elements::Tensor, "TreeRoot is nullptr, InsertInterval failed.");
            return FAILED;
        }
        return SUCCESS;
    }
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (ProcessTreeNode(interval, currPtr, intervalStack) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Tensor, "InsertInterval failed at function ProcessTreeNode.");
            return FAILED;
        }
        if (currPtr->max < interval.end) {
            currPtr->max = interval.end;
        }
    }
    return SUCCESS;
}

void RangeSearchTree::OverlapSearch(const Interval& interval, std::set<int>& result)
{
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);

    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        if (interval.start <= currPtr->interval.end && interval.end >= currPtr->interval.start) {
            result.insert(currPtr->interval.idx);
        }
        if (currPtr->left != nullptr && currPtr->left->max >= interval.start) {
            intervalStack.push_back(currPtr->left);
        }
        intervalStack.push_back(currPtr->right);
    }
}

void RangeSearchTree::FreeTree()
{
    std::vector<IntervalTreeNode*> intervalStack;
    intervalStack.push_back(treeRoot);
    while (intervalStack.size() > 0) {
        IntervalTreeNode* currPtr = intervalStack.back();
        intervalStack.pop_back();
        if (currPtr == nullptr) {
            continue;
        }
        intervalStack.push_back(currPtr->left);
        intervalStack.push_back(currPtr->right);
        delete currPtr;
    }
}

void RangeSearchTree::Insert(int left, int right, int idx)
{
    Interval interval(left, right, idx);
    InsertInterval(interval);
}

std::set<int> RangeSearchTree::GetCovered(int left, int right)
{
    Interval givenInterval(left, right, 0);
    std::set<int> overlappingIdx;
    OverlapSearch(givenInterval, overlappingIdx);
    return overlappingIdx;
}

void DataDependencySearcher::CheckWAWSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType currMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(currMemoryType) > 0) {
            TileRange rg = currMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                                  tensor->memoryrange;
            std::set<int> found = wawSearchTree_[currMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckRAWSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t inIdx = 0; inIdx < opWait->GetIOperands().size(); inIdx++) {
        auto tensor = opWait->GetIOperands()[inIdx];
        MemoryType readMemoryType = tensor->GetMemoryTypeOriginal();
        int readDDRmemId = tensor->memoryrange.memId;
        if (readDDRmemId != -1 && writeDdrMemMap.count(readDDRmemId) > 0) {
            std::set<int> found = writeDdrMemMap[readDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (rawSearchTree_.count(readMemoryType) > 0) {
            TileRange rg = readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                                  tensor->memoryrange;
            std::set<int> found = rawSearchTree_[readMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

void DataDependencySearcher::CheckWARSearchTree(Operation* opWait, std::set<int>& res)
{
    for (size_t outIdx = 0; outIdx < opWait->GetOOperands().size(); outIdx++) {
        auto tensor = opWait->GetOOperands()[outIdx];
        MemoryType writeMemoryType = tensor->GetMemoryTypeOriginal();
        int writeDDRmemId = tensor->memoryrange.memId;
        if (writeDDRmemId != -1 && readDdrMemMap.count(writeDDRmemId) > 0) {
            std::set<int> found = readDdrMemMap[writeDDRmemId];
            res.insert(found.begin(), found.end());
        }
        if (warSearchTree_.count(writeMemoryType) > 0) {
            TileRange rg = writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                                   tensor->memoryrange;
            std::set<int> found = warSearchTree_[writeMemoryType].GetCovered(rg.start, rg.end);
            res.insert(found.begin(), found.end());
        }
    }
}

std::set<int> DataDependencySearcher::Find(Operation* opWait)
{
    std::set<int> res;

    std::string opStr = opWait->GetOpcodeStr();
    // check WAW
    CheckWAWSearchTree(opWait, res);
    // check RAW
    CheckRAWSearchTree(opWait, res);
    // check WAR
    CheckWARSearchTree(opWait, res);
    for (auto* producer : opWait->ProducerOpsByToken()) {
        auto iter = opIndices_.find(producer);
        if (iter != opIndices_.end()) {
            res.emplace(iter->second);
        }
    }
    return res;
}

void DataDependencySearcher::InsertWAWSearchTree(const Operation* opSet, int idx)
{
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        auto tensor = opSet->GetOOperands()[outIdx];
        MemoryType prevMemoryType = tensor->GetMemoryTypeOriginal();
        if (wawSearchTree_.count(prevMemoryType) == 0) {
            wawSearchTree_[prevMemoryType] = RangeSearchTree();
        }
        TileRange rg = prevMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                              tensor->memoryrange;
        wawSearchTree_[prevMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertRAWSearchTree(const Operation* opSet, int idx)
{
    for (size_t outIdx = 0; outIdx < opSet->GetOOperands().size(); outIdx++) {
        auto tensor = opSet->GetOOperands()[outIdx];
        MemoryType writeMemoryType = tensor->GetMemoryTypeOriginal();
        int writeDDRmemId = tensor->memoryrange.memId;
        if (writeDDRmemId != -1) {
            if (writeDdrMemMap.count(writeDDRmemId) == 0) {
                writeDdrMemMap[writeDDRmemId] = std::set<int>{};
            }
            writeDdrMemMap[writeDDRmemId].insert(idx);
        }
        if (rawSearchTree_.count(writeMemoryType) == 0) {
            rawSearchTree_[writeMemoryType] = RangeSearchTree();
        }
        TileRange rg = writeMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                               tensor->memoryrange;
        rawSearchTree_[writeMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::InsertWARSearchTree(const Operation* opSet, int idx)
{
    for (size_t inIdx = 0; inIdx < opSet->GetIOperands().size(); inIdx++) {
        auto tensor = opSet->GetIOperands()[inIdx];
        MemoryType readMemoryType = tensor->GetMemoryTypeOriginal();
        int readDDRmemId = tensor->memoryrange.memId;
        if (readDDRmemId != -1) {
            if (readDdrMemMap.count(readDDRmemId) == 0) {
                readDdrMemMap[readDDRmemId] = std::set<int>{};
            }
            readDdrMemMap[readDDRmemId].insert(idx);
        }
        if (warSearchTree_.count(readMemoryType) == 0) {
            warSearchTree_[readMemoryType] = RangeSearchTree();
        }
        TileRange rg = readMemoryType == MemoryType::MEM_UB ? ubTensorRangeMap[tensor->GetMagic()] :
                                                              tensor->memoryrange;
        warSearchTree_[readMemoryType].Insert(rg.start, rg.end, idx);
    }
}

void DataDependencySearcher::Insert(const Operation* opSet, int idx)
{
    InsertWAWSearchTree(opSet, idx);
    InsertRAWSearchTree(opSet, idx);
    InsertWARSearchTree(opSet, idx);
    opIndices_.emplace(opSet, idx);
}

void PipeSync::BuildTensorRangeMap(Operation* op)
{
    auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
    if (opcfg.coreType_ != CoreType::AIV) {
        return;
    }
    bool isAIV1 = op->GetAIVCore() == AIVCore::AIV1;
    auto inTensors = op->GetIOperands();
    auto outTensors = op->GetOOperands();
    LogicalTensors inOutTensors;
    inOutTensors.reserve(inTensors.size() + outTensors.size());
    inOutTensors.insert(inOutTensors.end(), inTensors.begin(), inTensors.end());
    inOutTensors.insert(inOutTensors.end(), outTensors.begin(), outTensors.end());
    for (const auto& tensor : inOutTensors) {
        if (ubTensorRangeMap.count(tensor->GetMagic()) || tensor->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            continue;
        }
        TileRange tensorRange;
        if (!isAIV1) {
            tensorRange.start = tensor->memoryrange.start;
            tensorRange.end = tensor->memoryrange.end;
        } else {
            // 将AIV1中的tensor地址映射到的区间
            size_t curUBSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
            tensorRange.start = tensor->memoryrange.start + curUBSize;
            tensorRange.end = tensor->memoryrange.end + curUBSize;
        }
        ubTensorRangeMap.emplace(std::make_pair(tensor->GetMagic(), tensorRange));
    }
}

Status PipeSync::InsertSync(Function& function, std::vector<Operation*>& syncedOpLog)
{
    std::vector<IndexOp> synced;
    std::vector<Operation*> opLogPtr(function.Operations(false).DuplicatedOpList());
    oriOpList_ = opLogPtr;
    uint64_t idxInput = 0;
    for (const auto& op : opLogPtr) {
        BuildTensorRangeMap(op);
        APASS_LOG_DEBUG_F(Elements::Operation, "Input operation %lu %lu: %s.", static_cast<unsigned long>(idxInput),
                          static_cast<unsigned long>(op->GetOpMagic()), op->GetOpcodeStr().c_str());
        idxInput++;
    }
    if (PipeDispatch(opLogPtr, synced) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSync failed at function PipeDispatch.");
        return FAILED;
    }

    if (IssueOp(function, opLogPtr, synced) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSync failed at function IssueOp.");
        return FAILED;
    }

    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });

    for (auto& log : synced) {
        syncedOpLog.push_back(&log.second.get());
    }
    return ValidateEventReuse(syncedOpLog);
}

std::string PipeSyncTypes::DepOp::DumpDepOp(const std::vector<Operation*>& opLog)
{
    std::stringstream ss;
    ss << "idx: " << idx << " opmagic: " << opLog[idx]->GetOpMagic() << ", " << opLog[idx]->GetOpcodeStr() << ", ";
    ss << "setPipe: {";
    for (auto i : setPipe) {
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}, waitPipe: {";
    for (auto i : waitPipe) {
        ss << opLog[i]->GetOpMagic() << " " << opLog[i]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSyncTypes::IssueQueue::DumpIssueQueue(const std::vector<Operation*>& opLogPtr)
{
    std::stringstream ss;
    ss << " Op in this pipe: {";
    for (auto op : ops) {
        ss << opLogPtr[op]->GetOpMagic() << " " << opLogPtr[op]->GetOpcodeStr() << ", ";
    }
    ss << "}";
    return ss.str();
}

std::string PipeSyncTypes::PipeDepInfo::DumpPipeDepInfo()
{
    std::stringstream ss;
    ss << "    wait idx: " << waitIdx << "\n";
    ss << "    setPipeCoreEx:"
       << "\n";
    for (auto pair : setPipes) {
        ss << "        pipetype: " << GetPipeTypeDict().Find(pair.first.pipe) << " "
           << GetCoreTypeDict().Find(pair.first.core) << "  aivCore: " << static_cast<int>(pair.first.aivCore)
           << "  opidx: " << pair.second << "\n";
    }
    return ss.str();
}

std::string PipeSync::DumpLatestPipeDepMap()
{
    std::stringstream ss;
    for (auto pair : latestPipeDep_) {
        ss << "current PipeCore: " << GetPipeTypeDict().Find(pair.first.pipe) << " "
           << GetCoreTypeDict().Find(pair.first.core) << " aivCore: " << static_cast<int>(pair.first.aivCore) << "\n";
        ss << pair.second.DumpPipeDepInfo() << "\n";
    }
    return ss.str();
}

std::string PipeSync::PipeSeqName(PipeSeq seq) const
{
    switch (seq) {
        case PipeSeq::AIC_MTE2:
            return "AIC_MTE2";
        case PipeSeq::AIC_MTE1:
            return "AIC_MTE1";
        case PipeSeq::AIC_M:
            return "AIC_M";
        case PipeSeq::AIC_FIX:
            return "AIC_FIX";
        case PipeSeq::AIV0_MTE2:
            return "AIV0_MTE2";
        case PipeSeq::AIV1_MTE2:
            return "AIV1_MTE2";
        case PipeSeq::AIV0_V:
            return "AIV0_V";
        case PipeSeq::AIV1_V:
            return "AIV1_V";
        case PipeSeq::AIV0_MTE3:
            return "AIV0_MTE3";
        case PipeSeq::AIV1_MTE3:
            return "AIV1_MTE3";
        case PipeSeq::AIC_MTE3:
            return "AIC_MTE3";
        case PipeSeq::AIV0_S:
            return "AIV0_S";
        case PipeSeq::AIV1_S:
            return "AIV1_S";
        case PipeSeq::AIC_S:
            return "AIC_S";
        case PipeSeq::PIPE_END:
            return "PIPE_END";
        default:
            return "ILLEGAL";
    }
}

std::map<PipeSync::PipeCoreRealEx, PipeSeq, PipeSync::PipeCoreRealExCompare> PipeSync::pipe2Seq = {
    {{PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE2},
    {{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE1},
    {{PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_M},
    {{PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_FIX},
    {{PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_MTE3},
    {{PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED}, PipeSeq::AIC_S},
    {{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_MTE2},
    {{PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_MTE2},
    {{PIPE_V, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_V},
    {{PIPE_V, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_V},
    {{PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_MTE3},
    {{PIPE_MTE3, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_MTE3},
    {{PIPE_S, CoreType::AIV, AIVCore::AIV0}, PipeSeq::AIV0_S},
    {{PIPE_S, CoreType::AIV, AIVCore::AIV1}, PipeSeq::AIV1_S},
};

std::map<PipeSeq, PipeSync::PipeCoreRealEx> PipeSync::seq2pipe = {
    {PipeSeq::AIC_MTE2, {PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_MTE1, {PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_M, {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_FIX, {PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_MTE3, {PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIC_S, {PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED}},
    {PipeSeq::AIV0_MTE2, {PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_MTE2, {PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_V, {PIPE_V, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_V, {PIPE_V, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_MTE3, {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_MTE3, {PIPE_MTE3, CoreType::AIV, AIVCore::AIV1}},
    {PipeSeq::AIV0_S, {PIPE_S, CoreType::AIV, AIVCore::AIV0}},
    {PipeSeq::AIV1_S, {PIPE_S, CoreType::AIV, AIVCore::AIV1}},
};

PipeSeq PipeSync::GetPipeSeq(PipeSync::PipeCoreRealEx pipe) { return pipe2Seq.at(pipe); }

PipeSync::PipeCoreRealEx PipeSync::GetPipeFromSeq(PipeSeq seq) { return seq2pipe.at(seq); }

Status PipeSync::AdjustCopyInCfg(TileOpCfg& opcfg, const Operation& op)
{
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d COPYIN op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto dstMemType = attr->GetCopyInAttr().second;
    if (dstMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
        return SUCCESS;
    }
    if (dstMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
        opcfg.coreType_ = CoreType::AIV;
    }
    return SUCCESS;
}

Status PipeSync::AdjustCopyOutCfg(TileOpCfg& opcfg, const Operation& op)
{
    if (op.GetOpAttribute() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "%d COPYOUT op attr is nullptr, AdjustOpCfg failed.%s", op.GetOpMagic(),
                          GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    auto srcMemType = attr->GetCopyOutAttr().first;
    if (srcMemType == MemoryType::MEM_L0C) {
        opcfg.pipeIdStart_ = PipeType::PIPE_FIX;
        opcfg.pipeIdEnd_ = PipeType::PIPE_FIX;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_UB) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIV;
        return SUCCESS;
    }
    if (srcMemType == MemoryType::MEM_L1) {
        opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
        opcfg.coreType_ = CoreType::AIC;
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
    }
    return SUCCESS;
}

Status PipeSync::AdjustOpCfg(TileOpCfg& opcfg, const Operation& op)
{
    opcfg.aivCore_ = op.GetAIVCore();
    if (opcfg.coreType_ == CoreType::AIC) {
        opcfg.aivCore_ = AIVCore::UNSPECIFIED;
    }
    if (opcfg.coreType_ == CoreType::AIV && opcfg.aivCore_ == AIVCore::UNSPECIFIED) {
        opcfg.aivCore_ = AIVCore::AIV0;
    }
    if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        if (AdjustCopyInCfg(opcfg, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AdjustCopyInCfg failed.");
            return FAILED;
        }
    }
    if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
        if (AdjustCopyOutCfg(opcfg, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AdjustCopyOutCfg failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::PipeDispatch(const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    replayRank_.clear();
    DataDependencySearcher dataDependencySearcher;
    dataDependencySearcher.ubTensorRangeMap = ubTensorRangeMap;
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        if (opLogPtr[i]->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "%d ALLOC op should not appear in InsertSync, PipeDispatch failed.%s",
                              opLogPtr[i]->GetOpMagic(), GetFormatBacktrace(opLogPtr[i]).c_str());
            return FAILED;
        }
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        if (AdjustOpCfg(opcfg, *opLogPtr[i]) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PipeDispatch failed at function AdjustOpCfg.");
            return FAILED;
        }
        // for cv mix architecture, normalize all non PIPE_V instructions to AIC, to maintain consistency in a singular
        // core since they share a singular eventid
        PipeCoreRealEx normalizedAICOpCore(opcfg.pipeIdStart_, opcfg.coreType_, opcfg.aivCore_);
        if (Platform::Instance().GetSoc().GetCoreWrap().IsCVMix()) {
            normalizedAICOpCore = NormalizeToAICCoreForCVMixSync(normalizedAICOpCore);
            opcfg.coreType_ = normalizedAICOpCore.core;
            opcfg.aivCore_ = normalizedAICOpCore.aivCore;
        }
        DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
        DepOp& opRef = depOps_.emplace_back(op);
        FindDep(opRef, opLogPtr, i, dataDependencySearcher);
        EnqueueOp(opRef, opLogPtr, syncedOpLog);
    }
    return SUCCESS;
}

void PipeSync::InitIssueQueue()
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        issueState_.emplace_back(GetPipeFromSeq(static_cast<PipeSeq>(i)));
    }
}

void PipeSync::EnqueueOp(DepOp& op, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    if (opLogPtr[op.idx]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_NOP || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_HUB ||
        opLogPtr[op.idx]->GetOpcode() == Opcode::OP_VIEW_TYPE || opLogPtr[op.idx]->GetOpcode() == Opcode::OP_RESHAPE) {
        syncedOpLog.emplace_back(std::make_pair(op.idx * SEQUENCE_IDX, std::ref(*opLogPtr[op.idx])));
        return;
    }
    PipeCoreRealEx opPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    auto& issueQ = issueState_[static_cast<int>(GetPipeSeq(opPipeCoreEx))];
    issueQ.ops.emplace_back(op.idx);
    op.idxInPipe = issueQ.ops.size() - 1;
    // 若op的pipeStart和pipeEnd不同, 进行记录
    if (op.selfPipeCore.pipeStart != op.selfPipeCore.pipeEnd) {
        PipeCoreRealEx opPipeCoreExStart(op.selfPipeCore.pipeStart, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipeCoreRealEx opPipeCoreExEnd(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
        PipePairEx pp{opPipeCoreExStart, opPipeCoreExEnd};
        int opMagic = opLogPtr[op.idx]->GetOpMagic();
        doublePipeOp[pp].emplace_back(opMagic);
    }
    orderedOpList_.emplace(op.idx);
}

void PipeSync::RemoveOpDep(DepOp& setOp, DepOp& waitOp) const
{
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;
    std::vector<size_t> newSetDep;
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) {
            continue;
        }
        newSetDep.emplace_back(ele);
    }
    setOp.setPipe = newSetDep;

    std::vector<size_t> newWaitDep;
    for (auto ele : waitOp.waitPipe) {
        if (ele == setOpIdx) {
            continue;
        }
        newWaitDep.emplace_back(ele);
    }
    waitOp.waitPipe = newWaitDep;
}

Status PipeSync::AddOpDep(DepOp& setOp, DepOp& waitOp, bool isMergeCvSyncBase)
{
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;

    size_t depWaitIdx = static_cast<size_t>(-1);
    for (auto ele : setOp.setPipe) {
        if (ele == waitOpIdx) {
            APASS_LOG_ERROR_F(Elements::Operation, "This dependency should not exist, AddOpDep failed.");
            return FAILED;
        }
        PipeCoreRealEx elePipeCoreEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                                     depOps_[ele].selfPipeCore.aivCore);
        PipeCoreRealEx waitOpPipeCoreEx(depOps_[waitOpIdx].selfPipeCore.pipeStart, depOps_[waitOpIdx].selfPipeCore.core,
                                        depOps_[waitOpIdx].selfPipeCore.aivCore);
        if (elePipeCoreEx == waitOpPipeCoreEx) {
            if (ele <= waitOpIdx && !isMergeCvSyncBase) {
                APASS_LOG_ERROR_F(Elements::Operation, "New waitidx should less than old, AddOpDep failed.");
                return FAILED;
            } else if (ele <= waitOpIdx && isMergeCvSyncBase) {
                return SUCCESS;
            }
            depWaitIdx = ele;
            break;
        }
    }
    // 同一种pipecore，不需要存记录依赖关系
    if (depWaitIdx != static_cast<size_t>(-1)) {
        RemoveOpDep(setOp, depOps_[depWaitIdx]);
    }
    setOp.setPipe.emplace_back(waitOpIdx);
    waitOp.waitPipe.emplace_back(setOpIdx);
    if (!replayRank_.empty() && !PreservesReplayOrder(setOpIdx, waitOpIdx)) {
        replayRank_.clear();
    }
    return SUCCESS;
}

Status PipeSync::AdjustOpDep(DepOp& op, size_t waitOpIdx, IssueQueue& issueQ, bool& failedFlag)
{
    // op为靠前的， waitOp为靠后的
    auto& waitOp = depOps_[waitOpIdx];

    if (issueQ.currOp + 1 == issueQ.ops.size()) {
        failedFlag = true;
        return SUCCESS;
    }

    auto& nextOpIdx = issueQ.ops[issueQ.currOp + 1];

    // next op的idx要小于waitop
    if (nextOpIdx > waitOpIdx) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot AdjustOpDep because nextop idx > waitop idx.");
        failedFlag = true;
        return SUCCESS;
    }

    // Replay can introduce backward dependencies: original indices alone no
    // longer prove that delaying this wait to the next producer is acyclic.
    if (!CanAddReplayDependency(nextOpIdx, waitOpIdx)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Cannot AdjustOpDep because delaying the wait creates a cycle.");
        failedFlag = true;
        return SUCCESS;
    }

    APASS_LOG_INFO_F(Elements::Operation, "Event reuse yield: dependency (%zu,%zu) moves to (%zu,%zu).", op.idx,
                     waitOpIdx, nextOpIdx, waitOpIdx);
    auto& nextOp = depOps_[nextOpIdx];
    RemoveOpDep(op, waitOp);
    if (AddOpDep(nextOp, waitOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustOpDep failed at function AddOpDep.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSyncTypes::EventInstanceTracker::RetireConsumed(size_t setIdx, size_t waitIdx)
{
    const auto* found = Find(setIdx, waitIdx);
    if (found == nullptr || found->state != InstanceState::WaitInserted) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot retire dependency %zu -> %zu without a consumed event.", setIdx,
                          waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[eventInstanceByDep_.at({setIdx, waitIdx})];
    instance.state = InstanceState::Retired;
    auto& slot = slots_[instance.slot];
    const auto* active = FindActive(slot);
    if (active != nullptr && active->id == instance.id) {
        slot.active.reset();
    }
    return SUCCESS;
}

void PipeSyncTypes::EventInstanceTracker::ForgetProofs()
{
    // A barrier is not permission to forward an old generation from a new set.
    // Keeping only active facts also bounds frozen proof propagation after force.
    for (auto& entry : knownConsumed_) {
        entry.second = Known(entry.first);
    }
}

bool PipeSync::IsIntraCoreEvent(const PipePairEx& pp) const
{
    return pp.first.core == pp.second.core || IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch());
}

PipeSync::EventResource PipeSync::GetEventResource(const PipePairEx& pp, int eventId) const
{
    return {
        eventId, {pp.first.core, pp.first.aivCore}, {pp.second.core, pp.second.aivCore}, pp.first.pipe, pp.second.pipe};
}

const PipeSyncTypes::EventInstance* PipeSyncTypes::EventInstanceTracker::Find(size_t setIdx, size_t waitIdx) const
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    return it == eventInstanceByDep_.end() ? nullptr : &eventInstances_[it->second];
}

std::optional<size_t> PipeSyncTypes::EventInstanceTracker::FindSlot(const EventResource& resource) const
{
    const auto it = slotByResource_.find(resource);
    return it == slotByResource_.end() ? std::nullopt : std::optional<size_t>(it->second);
}

size_t PipeSyncTypes::EventInstanceTracker::GetOrCreateSlot(const EventResource& resource)
{
    const auto [it, inserted] = slotByResource_.try_emplace(resource, slots_.size());
    if (inserted) {
        slots_.push_back({resource, 0, std::nullopt});
    }
    return it->second;
}

const PipeSyncTypes::EventInstance* PipeSyncTypes::EventInstanceTracker::FindActive(const EventSlot& slot) const
{
    return slot.active ? &eventInstances_[*slot.active] : nullptr;
}

const PipeSyncTypes::EventInstance* PipeSyncTypes::EventInstanceTracker::FindActive(const EventResource& resource) const
{
    const auto slot = FindSlot(resource);
    return slot ? FindActive(slots_[*slot]) : nullptr;
}

void PipeSyncTypes::EventInstanceTracker::MergeConsumed(ConsumedInstances& destination,
                                                        const ConsumedInstances& snapshot) const
{
    for (const auto& proof : snapshot) {
        const auto* active = FindActive(slots_[proof.slot]);
        if (active != nullptr && active->id.generation == proof.generation) {
            destination.insert(proof);
        }
    }
}

const PipeSyncTypes::ConsumedInstances& PipeSyncTypes::EventInstanceTracker::KnownForAllocationCheck(
    const PipeCoreRealEx& pipe) const
{
    static const ConsumedInstances empty;
    auto it = knownConsumed_.find(pipe);
    return it == knownConsumed_.end() ? empty : it->second;
}

PipeSyncTypes::ConsumedInstances PipeSyncTypes::EventInstanceTracker::Known(const PipeCoreRealEx& pipe) const
{
    ConsumedInstances result;
    const auto& source = KnownForAllocationCheck(pipe);
    result.reserve(source.size());
    MergeConsumed(result, source);
    return result;
}

bool PipeSyncTypes::EventInstanceTracker::CanAllocate(const EventResource& resource, const ConsumedInstances& consumed,
                                                      bool requireActive) const
{
    const auto slot = FindSlot(resource);
    return CanAllocate(slot ? &slots_[*slot] : nullptr, consumed, requireActive);
}

bool PipeSyncTypes::EventInstanceTracker::CanAllocate(const EventSlot* slot, const ConsumedInstances& consumed,
                                                      bool requireActive) const
{
    if (slot == nullptr) {
        return !requireActive;
    }
    if (slot->generation == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    const auto* instance = FindActive(*slot);
    if (instance == nullptr) {
        return !requireActive;
    }
    // Only the proof of the exact active generation can authorize reuse.
    return instance->state == InstanceState::WaitInserted && slot->generation == instance->id.generation &&
           consumed.count(instance->Proof()) != 0;
}

Status PipeSyncTypes::EventInstanceTracker::Allocate(const EventResource& resource, const PipePairEx& pipes,
                                                     size_t setIdx, size_t waitIdx)
{
    return Allocate(resource, pipes, setIdx, waitIdx, Known(pipes.first));
}

Status PipeSyncTypes::EventInstanceTracker::Allocate(const EventResource& resource, const PipePairEx& pipes,
                                                     size_t setIdx, size_t waitIdx, ConsumedInstances consumed)
{
    const auto existing = FindSlot(resource);
    if (Find(setIdx, waitIdx) != nullptr || !CanAllocate(existing ? &slots_[*existing] : nullptr, consumed, false)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot allocate event %d for dependency %zu -> %zu.", resource.eventId,
                          setIdx, waitIdx);
        return FAILED;
    }
    const size_t slotIndex = existing ? *existing : GetOrCreateSlot(resource);
    auto& slot = slots_[slotIndex];
    if (slot.active) {
        auto& previous = eventInstances_[*slot.active];
        consumed.erase(previous.Proof());
        previous.state = InstanceState::Retired;
    }
    const auto index = eventInstances_.size();
    eventInstances_.push_back({{resource, ++slot.generation},
                               slotIndex,
                               pipes,
                               setIdx,
                               waitIdx,
                               std::move(consumed),
                               nullptr,
                               nullptr,
                               InstanceState::Allocated,
                               true});
    slot.active = index;
    eventInstanceByDep_[{setIdx, waitIdx}] = index;
    return SUCCESS;
}

Status PipeSyncTypes::EventInstanceTracker::RecordSet(const EventResource& resource, const PipePairEx& pipes,
                                                      size_t setIdx, size_t waitIdx, Operation& syncOp, bool intraCore)
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    if (!intraCore) {
        const auto slotIndex = it == eventInstanceByDep_.end() ? std::optional<size_t>(GetOrCreateSlot(resource)) :
                                                                 std::nullopt;
        if (!slotIndex || slots_[*slotIndex].generation == std::numeric_limits<uint64_t>::max()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Invalid cross-core event instance for dependency %zu -> %zu, event %d.", setIdx, waitIdx,
                              resource.eventId);
            return FAILED;
        }
        const auto index = eventInstances_.size();
        auto snapshot = Known(pipes.first);
        eventInstances_.push_back({{resource, ++slots_[*slotIndex].generation},
                                   *slotIndex,
                                   pipes,
                                   setIdx,
                                   waitIdx,
                                   std::move(snapshot),
                                   &syncOp,
                                   nullptr,
                                   InstanceState::Allocated,
                                   false});
        eventInstanceByDep_[{setIdx, waitIdx}] = index;
        return SUCCESS;
    }
    if (it == eventInstanceByDep_.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Missing allocation for set dependency %zu -> %zu.", setIdx, waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[it->second];
    if (instance.state != InstanceState::Allocated || instance.setOp != nullptr ||
        !(instance.id.resource == resource)) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid set state for dependency %zu -> %zu, event %d.", setIdx,
                          waitIdx, resource.eventId);
        return FAILED;
    }
    instance.setOp = &syncOp;
    return SUCCESS;
}

Status PipeSyncTypes::EventInstanceTracker::Consume(size_t setIdx, size_t waitIdx, Operation& syncOp)
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    if (it == eventInstanceByDep_.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Missing event instance for wait dependency %zu -> %zu.", setIdx,
                          waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[it->second];
    if (instance.state != InstanceState::Allocated || instance.setOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid wait state for dependency %zu -> %zu.", setIdx, waitIdx);
        return FAILED;
    }
    auto& consumed = knownConsumed_[instance.pipes.second];
    consumed.EraseIf([this](const ConsumptionProof& proof) {
        const auto* active = FindActive(slots_[proof.slot]);
        return active == nullptr || active->id.generation != proof.generation;
    });
    MergeConsumed(consumed, instance.setSnapshot);
    if (instance.intraCore) {
        consumed.insert(instance.Proof());
    }
    instance.waitOp = &syncOp;
    instance.state = InstanceState::WaitInserted;
    return SUCCESS;
}

Status PipeSyncTypes::EventInstanceTracker::CancelPending(size_t setIdx, size_t waitIdx)
{
    auto it = eventInstanceByDep_.find({setIdx, waitIdx});
    if (it == eventInstanceByDep_.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Missing pending dependency %zu -> %zu during cancellation.", setIdx,
                          waitIdx);
        return FAILED;
    }
    auto& instance = eventInstances_[it->second];
    if (instance.state != InstanceState::Allocated) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot cancel consumed dependency %zu -> %zu.", setIdx, waitIdx);
        return FAILED;
    }
    if (instance.intraCore) {
        auto& slot = slots_[instance.slot];
        const auto* active = FindActive(slot);
        if (active == nullptr || !(active->id == instance.id)) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cannot cancel dependency %zu -> %zu: its event generation does not own the active slot.",
                              setIdx, waitIdx);
            return FAILED;
        }
        slot.active.reset();
    }
    instance.state = InstanceState::Cancelled;
    eventInstanceByDep_.erase(it);
    return SUCCESS;
}

Status PipeSyncTypes::EventInstanceTracker::MergePending(size_t set1, size_t wait1, size_t set2, size_t wait2)
{
    const auto* first = Find(set1, wait1);
    const auto* second = Find(set2, wait2);
    if (first == nullptr || second == nullptr || first == second || first->state != InstanceState::Allocated ||
        second->state != InstanceState::Allocated || !first->intraCore || !second->intraCore ||
        first->pipes != second->pipes || set1 >= set2 || set2 >= wait1 || wait1 >= wait2 ||
        Find(set2, wait1) != nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid pending-event merge: %zu -> %zu and %zu -> %zu.", set1, wait1,
                          set2, wait2);
        return FAILED;
    }
    const auto index = eventInstanceByDep_.at({set2, wait2});
    if (CancelPending(set1, wait1) != SUCCESS) {
        return FAILED;
    }
    eventInstanceByDep_.erase({set2, wait2});
    eventInstanceByDep_[{set2, wait1}] = index;
    eventInstances_[index].waitIdx = wait1;
    return SUCCESS;
}

PipeSync::ConsumedInstances PipeSync::ProjectConsumedInstances(const DepOp& op, const PipeCoreRealEx& source) const
{
    auto result = eventState_.Known(source);
    const PipeCoreRealEx entry(op.selfPipeCore.pipeStart, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    if (source != entry) {
        // A multi-pipe op does not prove an entry-to-exit hardware ordering edge.
        return result;
    }
    // This is a preview only. InjectWaitFlag must commit these same waits before
    // allocation; PopFromQueue stops batches at waits so earlier proofs are committed.
    for (auto setIdx : op.waitPipe) {
        const auto* instance = eventState_.Find(setIdx, op.idx);
        if (instance == nullptr || instance->state != InstanceState::Allocated || instance->setOp == nullptr) {
            continue;
        }
        eventState_.MergeConsumed(result, instance->setSnapshot);
        if (instance->intraCore) {
            result.insert(instance->Proof());
        }
    }
    return result;
}

PipeSync::IntraEventIds PipeSync::GetAvailableIntraEventIds(const PipePairEx& pp, const ConsumedInstances& consumed,
                                                            size_t limit)
{
    IntraEventIds available;
    std::array<bool, EVENT_NUM> seen{};
    const int maxId = GetMaxEventId(pp);
    auto appendIfAvailable = [&](int id, bool requireActive) {
        if (id >= INTRA_CORE_START_EVENT_ID && id <= maxId && !seen[id]) {
            seen[id] = true;
            if (eventState_.CanAllocate(GetEventResource(pp, id), consumed, requireActive)) {
                available.ids[available.count++] = id;
            }
        }
        return available.count == limit;
    };
    for (int id : GetFreeEventIdQueue(pp)) {
        if (appendIfAvailable(id, false)) {
            return available;
        }
    }
    for (int id = INTRA_CORE_START_EVENT_ID; id <= maxId; ++id) {
        if (appendIfAvailable(id, true)) {
            break;
        }
    }
    return available;
}

std::optional<PipeSync::IntraEventCapacityFailure> PipeSync::AnalyzeIntraEventCapacity(const DepOp& op)
{
    if (op.setPipe.empty()) {
        return std::nullopt;
    }
    const PipeCoreRealEx source(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    std::optional<ConsumedInstances> consumed;
    // Neither the proof nor the capacity changes while checking this operation.
    std::unordered_map<PipePairEx, std::pair<size_t, size_t>, PipePairExHash> capacity;
    for (auto waitIdx : op.setPipe) {
        const auto& target = depOps_[waitIdx].selfPipeCore;
        const PipePairEx pp{source, {target.pipeStart, target.core, target.aivCore}};
        if (!IsIntraCoreEvent(pp) || pp.first == pp.second) {
            continue;
        }
        if (pp.first.core == CoreType::AIV && pp.second.core == CoreType::AIV &&
            pp.first.aivCore != pp.second.aivCore) {
            return IntraEventCapacityFailure{op.idx, waitIdx, pp, 0, 0};
        }
        auto [it, inserted] = capacity.try_emplace(pp, 0, 0);
        auto& [required, available] = it->second;
        if (inserted) {
            if (!consumed) {
                consumed.emplace(ProjectConsumedInstances(op, source));
            }
            available = GetAvailableIntraEventIds(pp, *consumed).size();
        }
        if (++required > available) {
            return IntraEventCapacityFailure{op.idx, waitIdx, pp, required, available};
        }
    }
    return std::nullopt;
}

void PipeSync::LogIntraEventCapacityFailure(const IntraEventCapacityFailure& failure) const
{
    const auto& source = failure.pipes.first;
    const auto& target = failure.pipes.second;
    if (source.core == CoreType::AIV && target.core == CoreType::AIV && source.aivCore != target.aivCore) {
        APASS_LOG_ERROR_F(Elements::Operation, "Unsupported AIV-to-AIV dependency %zu -> %zu, cores %d -> %d.",
                          failure.setIdx, failure.waitIdx, static_cast<int>(source.aivCore),
                          static_cast<int>(target.aivCore));
        return;
    }
    APASS_LOG_ERROR_F(Elements::Operation,
                      "Insufficient intra-core events for op %zu -> %zu: pipe %s -> %s, core %d/%d -> %d/%d, "
                      "required %zu, reusable %zu. Consumption has not returned to the source pipe.",
                      failure.setIdx, failure.waitIdx, GetPipeTypeDict().Find(source.pipe).c_str(),
                      GetPipeTypeDict().Find(target.pipe).c_str(), static_cast<int>(source.core),
                      static_cast<int>(source.aivCore), static_cast<int>(target.core), static_cast<int>(target.aivCore),
                      failure.required, failure.available);
}

Status PipeSync::AllocateIntraEvent(const PipePairEx& pp, int& eventId, size_t setIdx, size_t waitIdx)
{
    auto consumed = eventState_.Known(pp.first);
    const auto available = GetAvailableIntraEventIds(pp, consumed, 1);
    if (available.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "No proven reusable intra-core event for dependency %zu -> %zu.", setIdx,
                          waitIdx);
        return FAILED;
    }
    const int id = available.front();
    if (eventState_.Allocate(GetEventResource(pp, id), pp, setIdx, waitIdx, std::move(consumed)) != SUCCESS) {
        return FAILED;
    }
    auto& queue = GetFreeEventIdQueue(pp);
    queue.erase(std::remove(queue.begin(), queue.end(), id), queue.end());
    eventId = id;
    return SUCCESS;
}

Status PipeSync::RecordEventSet(const PipePairEx& pp, int eventId, size_t setIdx, size_t waitIdx, Operation& syncOp)
{
    if (eventState_.RecordSet(GetEventResource(pp, eventId), pp, setIdx, waitIdx, syncOp, IsIntraCoreEvent(pp)) !=
        SUCCESS) {
        return FAILED;
    }
    if (UseCrossCoreProof()) {
        auto& snapshot = crossCoreSnapshots_[{setIdx, waitIdx}];
        snapshot.consumed = crossCoreConsumed_[pp.first];
        if (pp.first.core != pp.second.core) {
            snapshot.eventId = eventId;
            snapshot.generation = crossCoreGenerations_[eventId];
        }
    }
    return SUCCESS;
}

Status PipeSync::RecordEventWait(size_t setIdx, size_t waitIdx, Operation& syncOp)
{
    if (eventState_.Consume(setIdx, waitIdx, syncOp) != SUCCESS) {
        return FAILED;
    }
    const auto* instance = eventState_.Find(setIdx, waitIdx);
    if (instance->intraCore) {
        // Preserve the original wait-return FIFO policy. Queue membership is
        // only a candidate: allocation still requires consumption at the source.
        GetFreeEventIdQueue(instance->pipes).push_back(instance->id.resource.eventId);
    }
    if (UseCrossCoreProof()) {
        auto snapshot = crossCoreSnapshots_.find({setIdx, waitIdx});
        if (snapshot != crossCoreSnapshots_.end()) {
            MergeCrossCoreProof(crossCoreConsumed_[instance->pipes.second], snapshot->second);
            crossCoreSnapshots_.erase(snapshot);
        }
    }
    return SUCCESS;
}

Status PipeSync::BuildEventDependencyGraph(const std::vector<Operation*>& opList, EventDependencyGraph& graph) const
{
    // Reconstruct only real FIFO and matched set->wait edges. In particular,
    // never add a wait->next-set edge merely because the ID was reused.
    std::unordered_map<Operation*, std::pair<size_t, bool>> syncInstances;
    syncInstances.reserve(eventState_.Instances().size() * 2);
    for (size_t i = 0; i < eventState_.Instances().size(); ++i) {
        const auto& instance = eventState_.Instances()[i];
        if (instance.state == InstanceState::Cancelled) {
            continue;
        }
        if (instance.setOp == nullptr || instance.waitOp == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unmatched event instance in final sync sequence.");
            return FAILED;
        }
        syncInstances.emplace(instance.setOp, std::make_pair(i, true));
        syncInstances.emplace(instance.waitOp, std::make_pair(i, false));
    }
    graph.positions.assign(eventState_.Instances().size(), {SIZE_MAX, SIZE_MAX});
    graph.nodes.reserve(syncInstances.size());
    graph.successors.reserve(syncInstances.size());
    graph.inDegree.reserve(syncInstances.size());
    std::unordered_map<PipeCoreRealEx, size_t, PipeCoreRealExHash> lastInPipe;
    auto addEdge = [&](size_t from, size_t to) {
        graph.successors[from].push_back(to);
        ++graph.inDegree[to];
    };
    for (auto* op : opList) {
        auto it = syncInstances.find(op);
        if (it == syncInstances.end()) {
            continue;
        }
        const auto instanceIdx = it->second.first;
        const bool isSet = it->second.second;
        auto& position = graph.positions[instanceIdx][isSet ? 0 : 1];
        if (position != SIZE_MAX) {
            const auto& instance = eventState_.Instances()[instanceIdx];
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Duplicate event %s for dependency %zu -> %zu in final sync sequence.",
                              isSet ? "set" : "wait", instance.setIdx, instance.waitIdx);
            return FAILED;
        }
        position = graph.nodes.size();
        graph.nodes.push_back(it->second);
        graph.successors.emplace_back();
        graph.inDegree.push_back(0);
        const auto& instance = eventState_.Instances()[instanceIdx];
        const auto& pipe = isSet ? instance.pipes.first : instance.pipes.second;
        auto last = lastInPipe.find(pipe);
        if (last != lastInPipe.end()) {
            addEdge(last->second, position);
        }
        lastInPipe[pipe] = position;
    }
    for (size_t i = 0; i < eventState_.Instances().size(); ++i) {
        if (eventState_.Instances()[i].state == InstanceState::Cancelled) {
            continue;
        }
        if (graph.positions[i][0] == SIZE_MAX || graph.positions[i][1] == SIZE_MAX) {
            const auto& instance = eventState_.Instances()[i];
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Missing event set or wait for dependency %zu -> %zu in final sync sequence.",
                              instance.setIdx, instance.waitIdx);
            return FAILED;
        }
        if (graph.positions[i][0] >= graph.positions[i][1]) {
            APASS_LOG_ERROR_F(Elements::Operation, "Event wait precedes its matching set in the emitted sequence.");
            return FAILED;
        }
        addEdge(graph.positions[i][0], graph.positions[i][1]);
    }
    return AddReclaimBarrierEdges(opList, graph);
}

Status PipeSync::AddReclaimBarrierEdges(const std::vector<Operation*>& opList, EventDependencyGraph& graph) const
{
    if (reclaimBarriers_.empty()) {
        return SUCCESS;
    }
    const auto realNodeCount = graph.nodes.size();
    auto addEdge = [&](size_t from, size_t to) {
        graph.successors[from].push_back(to);
        ++graph.inDegree[to];
    };
    // Reclaim regions are immutable. BAR.ALL drains the entire local prefix;
    // all local prefix waits complete before suffix instructions execute.
    std::unordered_map<Operation*, size_t> listPosition;
    listPosition.reserve(opList.size());
    for (size_t i = 0; i < opList.size(); ++i) {
        listPosition[opList[i]] = i;
    }
    for (const auto& barrier : reclaimBarriers_) {
        if (barrier.ops.empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Empty event reclaim region in final sync sequence.");
            return FAILED;
        }
        auto first = listPosition.find(barrier.ops.front());
        if (first == listPosition.end() || first->second + barrier.ops.size() > opList.size()) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Missing or incomplete event reclaim region in final sync sequence.");
            return FAILED;
        }
        for (size_t i = 0; i < barrier.ops.size(); ++i) {
            if (opList[first->second + i] != barrier.ops[i]) {
                APASS_LOG_ERROR_F(Elements::Operation, "Event reclaim region was split or reordered.");
                return FAILED;
            }
        }
        const auto end = first->second + barrier.ops.size();
        std::vector<size_t> entries;
        std::vector<size_t> exits;
        for (size_t i = 0; i < barrier.cores.size(); ++i) {
            const size_t entry = graph.nodes.size();
            graph.nodes.emplace_back(SIZE_MAX, false);
            graph.successors.emplace_back();
            graph.inDegree.push_back(0);
            const size_t exit = graph.nodes.size();
            graph.nodes.emplace_back(SIZE_MAX, false);
            graph.successors.emplace_back();
            graph.inDegree.push_back(0);
            entries.push_back(entry);
            exits.push_back(exit);
            addEdge(entry, exit);
        }
        for (size_t node = 0; node < realNodeCount; ++node) {
            const auto& instance = eventState_.Instances()[graph.nodes[node].first];
            const bool isSet = graph.nodes[node].second;
            const auto& pipe = isSet ? instance.pipes.first : instance.pipes.second;
            const auto pos = listPosition.at(isSet ? instance.setOp : instance.waitOp);
            const CoreTypeDetail core{pipe.core, pipe.aivCore};
            for (size_t c = 0; c < barrier.cores.size(); ++c) {
                // Lite's logical AIC/AIV endpoints execute on one physical core.
                if (core != barrier.cores[c] && !IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
                    continue;
                }
                if (pos < first->second) {
                    addEdge(node, entries[c]);
                } else if (pos >= end) {
                    addEdge(exits[c], node);
                }
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::ValidateEventGraphAcyclic(const EventDependencyGraph& graph) const
{
    auto remaining = graph.inDegree;
    std::queue<size_t> ready;
    for (size_t i = 0; i < remaining.size(); ++i) {
        if (remaining[i] == 0) {
            ready.push(i);
        }
    }
    size_t visited = 0;
    while (!ready.empty()) {
        auto node = ready.front();
        ready.pop();
        ++visited;
        for (auto next : graph.successors[node]) {
            if (--remaining[next] == 0) {
                ready.push(next);
            }
        }
    }
    if (visited != graph.nodes.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cycle detected in final event dependency graph.");
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::ValidateEventReuseOrder(const EventDependencyGraph& graph) const
{
    std::unordered_map<EventResource, size_t, EventResourceHash> lastUse;
    // Each set node is a unique query stamp; proofs remain specific to one generation.
    std::vector<size_t> reached(graph.nodes.size(), SIZE_MAX);
    std::vector<size_t> pending;
    for (size_t node = 0; node < graph.nodes.size(); ++node) {
        if (graph.nodes[node].first == SIZE_MAX) {
            continue;
        }
        const auto& instance = eventState_.Instances()[graph.nodes[node].first];
        if (!graph.nodes[node].second || !instance.intraCore) {
            continue;
        }
        auto old = lastUse.find(instance.id.resource);
        if (old != lastUse.end()) {
            const auto& previous = eventState_.Instances()[old->second];
            if (instance.id.generation <= previous.id.generation) {
                APASS_LOG_ERROR_F(Elements::Operation, "Non-increasing generation for intra-core event %d.",
                                  instance.id.resource.eventId);
                return FAILED;
            }
            pending.clear();
            pending.push_back(graph.positions[old->second][1]);
            while (!pending.empty()) {
                auto current = pending.back();
                pending.pop_back();
                if (reached[current] == node) {
                    continue;
                }
                reached[current] = node;
                if (current == node) {
                    break;
                }
                pending.insert(pending.end(), graph.successors[current].begin(), graph.successors[current].end());
            }
            if (reached[node] != node) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Missing wait-to-set ordering for intra-core event %d, generation %lu.",
                                  instance.id.resource.eventId, static_cast<unsigned long>(instance.id.generation));
                return FAILED;
            }
        }
        lastUse[instance.id.resource] = graph.nodes[node].first;
    }
    return SUCCESS;
}

Status PipeSync::ValidateEventReuse(const std::vector<Operation*>& opList) const
{
    EventDependencyGraph graph;
    if (BuildEventDependencyGraph(opList, graph) != SUCCESS || ValidateEventGraphAcyclic(graph) != SUCCESS ||
        ValidateEventReuseOrder(graph) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::ProcessSameCoreCase(const PipePairEx& pp, EventIdProcessContext& ctx,
                                     std::optional<ConsumedInstances>& projected)
{
    const auto issued = ctx.issuenum.currIssueNum.try_emplace(pp, 0).first->second;
    // Reserve the complete outgoing group before PopFromQueue commits this op.
    const auto required = std::count_if(ctx.op.setPipe.begin(), ctx.op.setPipe.end(), [&](size_t idx) {
        const auto& target = depOps_[idx].selfPipeCore;
        return target.pipeStart == pp.second.pipe && target.core == pp.second.core &&
               target.aivCore == pp.second.aivCore;
    });
    const auto needed = issued + std::max<size_t>(required, 1);
    const auto* proofs = &eventState_.KnownForAllocationCheck(pp.first);
    // With no same-pipe wait to apply, the exact generation lookup already ignores stale proofs.
    if (!ctx.op.waitPipe.empty() && ctx.op.selfPipeCore.pipeStart == pp.first.pipe) {
        if (!projected) {
            projected.emplace(ProjectConsumedInstances(ctx.op, pp.first));
        }
        proofs = &*projected;
    }
    const auto available = GetAvailableIntraEventIds(pp, *proofs, needed).size();
    if (available < needed) {
        if (!ctx.deadlock) {
            ctx.eventIdOk = false;
            return SUCCESS;
        }
        // A yield can rewrite the waits used to project consumption.
        projected.reset();
        if (AdjustOpDep(ctx.op, ctx.eleIdx, ctx.issueQ, ctx.failedFlag) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessSameCoreCase failed at AdjustOpDep.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status PipeSync::ProcessEventIdElement(EventIdProcessContext& ctx, std::optional<ConsumedInstances>& projected)
{
    auto& op = ctx.op;
    auto ele = ctx.eleIdx;

    if (op.selfPipeCore.pipeEnd == depOps_[ele].selfPipeCore.pipeStart) {
        return SUCCESS;
    }

    PipeCoreRealEx currEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
    PipeCoreRealEx eleEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                         depOps_[ele].selfPipeCore.aivCore);
    PipePairEx pp{currEx, eleEx};

    // 当前逻辑该函数不需要处理cv同步 (if mix cv core, need handle mix cv core sync)
    if (!IsMixCvCore(currEx.core, eleEx.core) && (currEx.core != eleEx.core)) {
        return SUCCESS;
    }
    return ProcessSameCoreCase(pp, ctx, projected);
}

Status PipeSync::HandleEventID(DepOp& op, IssueQueue& issueQ, IssueNum& issuenum, bool& deadlock, bool& res,
                               std::vector<IndexOp>& syncedOpLog)
{
    bool eventIdOk = true;
    bool failedFlag = false;

    // Yield may rewrite several dependencies. Do not retain a partial rewrite
    // when another destination prevents this operation from being issued.
    std::unordered_map<size_t, DepOp> originalDeps;
    if (deadlock) {
        originalDeps.emplace(op.idx, op);
        for (auto waitIdx : op.setPipe) {
            originalDeps.emplace(waitIdx, depOps_[waitIdx]);
        }
        if (issueQ.currOp + 1 < issueQ.ops.size()) {
            const auto nextIdx = issueQ.ops[issueQ.currOp + 1];
            originalDeps.emplace(nextIdx, depOps_[nextIdx]);
            // AddOpDep may replace a later wait already attached to nextOp.
            for (auto waitIdx : depOps_[nextIdx].setPipe) {
                originalDeps.emplace(waitIdx, depOps_[waitIdx]);
            }
        }
    }
    std::optional<ConsumedInstances> projected;
    // Only the deadlock path can rewrite the dependencies during iteration.
    const auto originalSetPipe = deadlock ? op.setPipe : std::vector<size_t>{};
    const auto& oriSetPipe = deadlock ? originalSetPipe : op.setPipe;
    for (auto ele : oriSetPipe) {
        EventIdProcessContext ctx{op, ele, issueQ, issuenum, syncedOpLog, deadlock, eventIdOk, failedFlag};
        if (ProcessEventIdElement(ctx, projected) != SUCCESS) {
            RestoreOpDeps(originalDeps);
            APASS_LOG_ERROR_F(Elements::Operation, "HandleEventID failed at ProcessEventIdElement.");
            return FAILED;
        }
        eventIdOk = ctx.eventIdOk;
        failedFlag = ctx.failedFlag;
        if (!eventIdOk) {
            break;
        }
    }

    if (failedFlag) {
        deadlock = true;
        res = false;
    } else {
        deadlock = false;
        res = eventIdOk;
    }
    if (!res) {
        RestoreOpDeps(originalDeps);
    }
    return SUCCESS;
}

bool PipeSync::CheckIssuedOp(const DepOp& op)
{
    // current op will be issued only when all of the waitop are issued
    for (const auto& waitOp : op.waitPipe) {
        if (!depOps_[waitOp].issued) {
            return false;
        }
    }
    return true;
}

Status PipeSync::PopFromQueue(IssueQueue& issueQ, std::vector<size_t>& poped, bool& deadlock,
                              std::vector<IndexOp>& syncedOpLog)
{
    IssueNum issuenum;

    for (uint64_t i = 0; i < MAX_POP; i++) {
        if (issueQ.currOp >= issueQ.ops.size()) {
            break;
        }
        auto& op = depOps_[issueQ.ops[issueQ.currOp]];
        if (!poped.empty() && !op.waitPipe.empty()) {
            break;
        }
        if (op.idx != orderedOpList_.front()) {
            break;
        }
        if (op.issued) {
            APASS_LOG_ERROR_F(Elements::Operation, "Try to issue a op which is already issued, PopFromQueue failed.");
            return FAILED;
        }
        if (!CheckIssuedOp(op)) {
            break;
        }
        bool res = false;
        // cv同步不需要在此处理eventid
        if (HandleEventID(op, issueQ, issuenum, deadlock, res, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PopFromQueue failed at function HandleEventID.");
            return FAILED;
        }
        if (!res) {
            break;
        }
        op.issued = true;
        poped.emplace_back(op.idx);
        orderedOpList_.pop();
        for (auto ele : op.setPipe) {
            PipeCoreRealEx currPipeCoreEx(op.selfPipeCore.pipeEnd, op.selfPipeCore.core, op.selfPipeCore.aivCore);
            PipeCoreRealEx elePipeCoreEx(depOps_[ele].selfPipeCore.pipeStart, depOps_[ele].selfPipeCore.core,
                                         depOps_[ele].selfPipeCore.aivCore);
            auto pp = PipePairEx{currPipeCoreEx, elePipeCoreEx};
            issuenum.currIssueNum[pp] = issuenum.currIssueNum[pp] + 1;
        }
        issueQ.currOp++;
        // Commit waits and their proofs before checking the next operation.
        // A batch preflight cannot see the returning waits in its own batch.
        if (!op.waitPipe.empty()) {
            break;
        }
    }
    return SUCCESS;
}

PipeSync::PipeCoreRealEx PipeSync::NormalizeToAICCoreForCVMixSync(PipeCoreRealEx p) const
{
    if (p.core == CoreType::AIV && p.pipe != PipeType::PIPE_V) {
        p.core = CoreType::AIC;
        p.aivCore = AIVCore::UNSPECIFIED;
    }
    return p;
}

// check if platform architecture is mix cv core, and coreOne and coreTwo are AIV/AIC or vice versa
bool PipeSync::IsMixCvCore(CoreType coreOne, CoreType coreTwo)
{
    return Platform::Instance().GetSoc().GetCoreWrap().IsCVMix() &&
           ((coreOne == CoreType::AIV && coreTwo == CoreType::AIC) ||
            (coreOne == CoreType::AIC && coreTwo == CoreType::AIV));
}

Status PipeSync::InjectWaitFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    // Validate the complete input group before consuming any instance. A
    // malformed later wait must not leave an earlier wait partially committed.
    for (auto ele : depOps_[idx].waitPipe) {
        const auto& setPipe = depOps_[ele].selfPipeCore;
        if ((setPipe.aivCore == AIVCore::AIV0 && currPipe.aivCore == AIVCore::AIV1) ||
            (setPipe.aivCore == AIVCore::AIV1 && currPipe.aivCore == AIVCore::AIV0)) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Sync between AIV0 and AIV1 happened. wait op idx: %zu, wait op: %s[%d], set op idx: %zu, "
                "set op: %s[%d], setPipe: {pipeStart %s, pipeEnd %s, core %d, aivCore %d}, currPipe: "
                "{pipeStart %s, pipeEnd %s, core %d, aivCore %d}.",
                idx, oriOpList_[idx]->GetOpcodeStr().c_str(), oriOpList_[idx]->GetOpMagic(), ele,
                oriOpList_[ele]->GetOpcodeStr().c_str(), oriOpList_[ele]->GetOpMagic(),
                GetPipeTypeDict().Find(setPipe.pipeStart).c_str(), GetPipeTypeDict().Find(setPipe.pipeEnd).c_str(),
                static_cast<int>(setPipe.core), static_cast<int>(setPipe.aivCore),
                GetPipeTypeDict().Find(currPipe.pipeStart).c_str(), GetPipeTypeDict().Find(currPipe.pipeEnd).c_str(),
                static_cast<int>(currPipe.core), static_cast<int>(currPipe.aivCore));
            return FAILED;
        }
        auto pair = setWaitPairMap_.find({ele, idx});
        if (pair == setWaitPairMap_.end()) {
            APASS_LOG_ERROR_F(Elements::Operation, "Missing set for wait dependency %zu -> %zu.", ele, idx);
            return FAILED;
        }
        if (pair->second >= 0) {
            const auto* event = eventState_.Find(ele, idx);
            if (event == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Missing instance for wait dependency %zu -> %zu, event %d.",
                                  ele, idx, pair->second);
                return FAILED;
            }
            if (event->state != InstanceState::Allocated || event->setOp == nullptr ||
                event->id.resource.eventId != pair->second) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "Invalid instance for wait dependency %zu -> %zu: event %d, expected %d, state %d, "
                                  "set present %d.",
                                  ele, idx, event->id.resource.eventId, pair->second, static_cast<int>(event->state),
                                  static_cast<int>(event->setOp != nullptr));
                return FAILED;
            }
        }
    }
    // serch the waitpipe of current op
    uint64_t waitIdx = idx == 0 ? 0 : idx * SEQUENCE_IDX - HALF_SEQUENCE_IDX;
    for (const auto& ele : depOps_[idx].waitPipe) {
        PipeCore setPipe = depOps_[ele].selfPipeCore;
        PipeCoreRealEx setPipeRealEx(setPipe.pipeEnd, setPipe.core, setPipe.aivCore);
        PipeCoreRealEx currPipeRealEx(currPipe.pipeStart, currPipe.core, currPipe.aivCore);
        const int eventId = setWaitPairMap_.at({ele, idx}); // Validated with the complete wait group.
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& syncOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_SYNC_DST, input, output);
        bool res = GenSyncOp(setPipeRealEx, currPipeRealEx, eventId, false, syncOp);
        if (!res) {
            syncOp.SetAsDeleted();
            continue;
        }
        // insert wait_flag
        syncOp.SetAIVCore(currPipe.aivCore);
        syncedOpLog.emplace_back(std::make_pair(++waitIdx, std::ref(syncOp)));
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d, setAIVCore type: %d, waitAIVCore type: %d",
            syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(),
            GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
            GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_,
            static_cast<int>(syncOp.syncQueue_.setAivCore_), static_cast<int>(syncOp.syncQueue_.waitAivCore_));
        if (eventId >= 0 && RecordEventWait(ele, idx, syncOp) != SUCCESS) {
            syncOp.SetAsDeleted();
            syncedOpLog.pop_back();
            return FAILED;
        }
        // 记录 set op 和 waitflag的对应关系
        waitOpMap.emplace(&syncOp, oriOpList_[ele]);
        for (auto it = NoWaitCVPairs_.begin(); it != NoWaitCVPairs_.end(); ++it) {
            if (it->first == std::make_pair(ele, idx)) {
                NoWaitCVPairs_.erase(it);
                break;
            }
        }
    }
    return SUCCESS;
}

Status PipeSync::InjectSetFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    // InjectSync preflights the whole group before committing its waits.
    // Allocation below uses those committed proofs, without another projection.
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    uint64_t setIdx = idx * SEQUENCE_IDX + FORCE_SYNC_OP_NUM;
    std::vector<size_t> setPipeCopy = depOps_[idx].setPipe;
    for (size_t& ele : setPipeCopy) {
        PipeCore waitPipe = depOps_[ele].selfPipeCore;
        PipeCoreRealEx waitPipeRealEx(waitPipe.pipeStart, waitPipe.core, waitPipe.aivCore);
        PipeCoreRealEx currPipeRealEx(currPipe.pipeEnd, currPipe.core, currPipe.aivCore);
        int eventId{0};
        if (GetEventId({currPipeRealEx, waitPipeRealEx}, eventId, idx, ele, syncedOpLog, function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InjectSetFlag failed at function GetEventId.");
            return FAILED;
        }
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& syncOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_SYNC_SRC, input, output);
        bool res = GenSyncOp(currPipeRealEx, waitPipeRealEx, eventId, true, syncOp);
        if (res) {
            if (eventId >= 0 &&
                RecordEventSet({currPipeRealEx, waitPipeRealEx}, eventId, idx, ele, syncOp) != SUCCESS) {
                syncOp.SetAsDeleted();
                return FAILED;
            }
            // insert set_flag
            syncOp.SetAIVCore(currPipe.aivCore);
            syncedOpLog.emplace_back(std::make_pair(++setIdx, std::ref(syncOp)));
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Insert %d %s, setpipe: %s, waitpipe: %s, eventid: %d, setAIVCore type: %d, waitAIVCore type: %d",
                syncOp.GetOpMagic(), syncOp.GetOpcodeStr().c_str(),
                GetPipeTypeDict().Find(syncOp.syncQueue_.pipeId_).c_str(),
                GetPipeTypeDict().Find(syncOp.syncQueue_.trigPipeId_).c_str(), syncOp.syncQueue_.eventId_,
                static_cast<int>(syncOp.syncQueue_.setAivCore_), static_cast<int>(syncOp.syncQueue_.waitAivCore_));
            setWaitPairMap_[{idx, ele}] = eventId;
            // 记录wait op 和 setflag的对应关系
            setOpMap.emplace(&syncOp, oriOpList_[ele]);
            if (currPipe.core != waitPipe.core) {
                NoWaitCVPairs_.push_back({{idx, ele}, &syncOp});
            }
            continue;
        }
        syncOp.SetAsDeleted();
        setWaitPairMap_[{idx, ele}] = eventId;
    }
    return SUCCESS;
}

Status PipeSync::InjectSync(Function& function, const std::vector<Operation*>& opLogPtr, size_t idx,
                            std::vector<IndexOp>& syncedOpLog)
{
    // check idx range
    if (idx > std::numeric_limits<uint64_t>::max() / SEQUENCE_IDX) {
        APASS_LOG_ERROR_F(Elements::Operation, "Operation index is out of range, InjectSync failed.");
        return FAILED;
    }

    // A cross-core drain may retire a completed instance that carries an
    // intra-core proof. Perform it before the final intra-core capacity check.
    if (UseCrossCoreProof() && EnsureCrossCoreCapacity(function, idx, syncedOpLog) != SUCCESS) {
        return FAILED;
    }

    // No waits, sets, or CV force operations may commit when an intra-core
    // destination cannot be allocated for this operation.
    if (auto failure = AnalyzeIntraEventCapacity(depOps_[idx])) {
        LogIntraEventCapacityFailure(*failure);
        return FAILED;
    }

    // insert wait_flag
    if (InjectWaitFlag(function, idx, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InjectSync failed at function InjectWaitFlag.");
        return FAILED;
    }

    // insert current operation
    syncedOpLog.emplace_back(std::make_pair(idx * SEQUENCE_IDX, std::ref(*opLogPtr[idx])));
    depOps_[idx].issued = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Insert %d %s", opLogPtr[idx]->GetOpMagic(),
                      opLogPtr[idx]->GetOpcodeStr().c_str());

    // The complete output group was preflighted before waits. They have now
    // committed exactly the projected proofs; allocation uses committed state.
    if (InjectSetFlag(function, idx, syncedOpLog) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InjectSync failed at function InjectSetFlag.");
        return FAILED;
    }

    return SUCCESS;
}

int PipeSync::GetMaxEventId(const PipePairEx& pp)
{
    PipePairEx ppReverse = {pp.second, pp.first};
    auto it1 = doublePipeOp.find(pp);
    auto it2 = doublePipeOp.find(ppReverse);
    if (it1 == doublePipeOp.end() && it2 == doublePipeOp.end()) {
        return EVENT_ID7;
    }
    return EVENT_ID7;
}

Status PipeSync::ProcessDeadLock(uint64_t& eventIdDeadlockEnterTimes, bool& eventIdDeadlock,
                                 std::vector<IndexOp>& syncedOpLog, bool* mergeSearchExhausted)
{
    eventIdDeadlockEnterTimes++;
    APASS_LOG_INFO_F(Elements::Operation, "Event reuse stall: attempt=%lu, emitted=%zu, pending=%zu.",
                     static_cast<unsigned long>(eventIdDeadlockEnterTimes), syncedOpLog.size(), orderedOpList_.size());
    // eventID deadlock
    if (eventIdDeadlockEnterTimes > DEADLOCK_TIME_THRESHOLD) {
        eventIdDeadlock = true;
    }
    if (mergeSearchExhausted == nullptr || !*mergeSearchExhausted) {
        bool merged = false;
        if (RelaxFakeDataDep(syncedOpLog, merged) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ProcessDeadLock failed at function RelaxFakeDataDep.");
            return FAILED;
        }
        if (mergeSearchExhausted != nullptr) {
            *mergeSearchExhausted = !merged;
        }
    }
    if (eventIdDeadlockEnterTimes >= EVENTID_DEADLOCK_ENTER_TIME) {
        if (!orderedOpList_.empty()) {
            // Emit the blocked op and exact directed pool before the terminal error.
            if (auto failure = AnalyzeIntraEventCapacity(depOps_[orderedOpList_.front()])) {
                LogIntraEventCapacityFailure(*failure);
            }
        }
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Unbreakable deadlock: no legal dependency merge or yield position after %lu attempts; "
                          "intra-core IDs with consumed waits still require a return path to their source pipe.",
                          static_cast<unsigned long>(eventIdDeadlockEnterTimes));
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::IssueOpPipeSeq(Function& function, const std::vector<Operation*>& opLogPtr,
                                std::vector<IndexOp>& syncedOpLog, bool& eventIdDeadlock, size_t& issued)
{
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        std::vector<size_t> issuedOps;
        if (PopFromQueue(issueState_[i], issuedOps, eventIdDeadlock, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function PopFromQueue.");
            return FAILED;
        }
        issued += issuedOps.size();
        for (auto idx : issuedOps) {
            if (InjectSync(function, opLogPtr, idx, syncedOpLog) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function InjectSync.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

void PipeSync::CacheReplayOrder(const std::vector<size_t>& order)
{
    if (order.size() != depOps_.size()) {
        replayRank_.clear();
        return;
    }
    replayRank_.resize(order.size());
    replayBoundary_.resize(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        replayRank_[order[i]] = i;
    }
    size_t maxRank = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        maxRank = std::max(maxRank, replayRank_[i]);
        // A cross-core endpoint must separate all lower indices from all higher ones.
        replayBoundary_[i] = maxRank == i && replayRank_[i] == i;
    }
}

bool PipeSync::PreservesReplayOrder(size_t from, size_t to) const
{
    if (replayRank_[from] >= replayRank_[to]) {
        return false;
    }
    const auto& source = depOps_[from].selfPipeCore;
    const auto& target = depOps_[to].selfPipeCore;
    return (replayBoundary_[from] && replayBoundary_[to]) ||
           IsIntraCoreEvent(
               {{source.pipeEnd, source.core, source.aivCore}, {target.pipeStart, target.core, target.aivCore}});
}

bool PipeSync::CanAddReplayDependency(size_t from, size_t to)
{
    if (!replayRank_.empty() && PreservesReplayOrder(from, to)) {
        return true;
    }
    std::vector<size_t> order;
    if (!BuildReplayIssueOrder(depOps_, order, std::make_pair(from, to))) {
        return false;
    }
    CacheReplayOrder(order);
    return true;
}

void PipeSync::RestoreOpDeps(const std::unordered_map<size_t, DepOp>& originalDeps)
{
    for (const auto& [idx, dep] : originalDeps) {
        depOps_[idx] = dep;
        if (!replayRank_.empty() && std::any_of(dep.setPipe.begin(), dep.setPipe.end(), [this, idx = idx](size_t to) {
                return !PreservesReplayOrder(idx, to);
            })) {
            replayRank_.clear();
        }
    }
}

bool PipeSync::BuildReplayIssueOrder(const std::vector<DepOp>& plan, std::vector<size_t>& order,
                                     std::optional<std::pair<size_t, size_t>> extraDependency) const
{
    // A flat edge list avoids allocating a separate buffer for each operation.
    std::vector<size_t> firstEdge(plan.size(), SIZE_MAX);
    std::vector<std::pair<size_t, size_t>> edges;
    edges.reserve(plan.size());
    std::vector<size_t> remaining(plan.size(), 0);
    auto addEdge = [&](size_t from, size_t to) {
        edges.emplace_back(to, firstEdge[from]);
        firstEdge[from] = edges.size() - 1;
        ++remaining[to];
    };
    std::unordered_map<PipeCoreRealEx, size_t, PipeCoreRealExHash> lastInPipe;
    std::vector<bool> crossCoreBoundary(plan.size(), false);
    auto addDependency = [&](const DepOp& op, size_t waitIdx) {
        addEdge(op.idx, waitIdx);
        const auto& source = op.selfPipeCore;
        const auto& target = plan[waitIdx].selfPipeCore;
        if (!IsIntraCoreEvent(
                {{source.pipeEnd, source.core, source.aivCore}, {target.pipeStart, target.core, target.aivCore}})) {
            crossCoreBoundary[op.idx] = true;
            crossCoreBoundary[waitIdx] = true;
        }
    };
    for (const auto& op : plan) {
        for (auto waitIdx : op.setPipe) {
            addDependency(op, waitIdx);
        }
        if (extraDependency && extraDependency->first == op.idx) {
            addDependency(op, extraDependency->second);
        }
        // Preserve both endpoints of multi-pipe operations conservatively.
        const auto& core = op.selfPipeCore;
        for (auto pipe : {core.pipeStart, core.pipeEnd}) {
            const PipeCoreRealEx endpoint(pipe, core.core, core.aivCore);
            auto last = lastInPipe.find(endpoint);
            if (last != lastInPipe.end() && last->second != op.idx) {
                addEdge(last->second, op.idx);
            }
            lastInPipe[endpoint] = op.idx;
        }
    }
    // The legacy CV force path uses original indices as prefix boundaries.
    // Keep every cross-core endpoint's prefix intact during replay; otherwise
    // it could cancel a pending set whose producer sorts after the force.
    std::optional<size_t> lastBoundary;
    for (size_t i = 0; i < plan.size(); ++i) {
        if (lastBoundary) {
            addEdge(*lastBoundary, i);
        }
        if (crossCoreBoundary[i]) {
            for (size_t j = lastBoundary ? *lastBoundary + 1 : 0; j < i; ++j) {
                addEdge(j, i);
            }
            lastBoundary = i;
        }
    }
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t i = 0; i < plan.size(); ++i) {
        if (remaining[i] == 0) {
            ready.push(i);
        }
    }
    order.clear();
    order.reserve(plan.size());
    while (!ready.empty()) {
        const auto idx = ready.top();
        ready.pop();
        order.push_back(idx);
        // All successors become ready before the next pop; the min-heap preserves replay order.
        for (auto edge = firstEdge[idx]; edge != SIZE_MAX; edge = edges[edge].second) {
            const auto next = edges[edge].first;
            if (--remaining[next] == 0) {
                ready.push(next);
            }
        }
    }
    return order.size() == plan.size();
}

bool PipeSync::MergeDependenciesForReplay(std::vector<DepOp>& plan, const PipePairEx& pipes,
                                          std::vector<size_t>& order) const
{
    std::vector<std::pair<size_t, size_t>> dependencies;
    // Both endpoints must stay on this pipe: merging relies on its FIFO at set and wait.
    auto matches = [](const DepOp& op, const PipeCoreRealEx& pipe) {
        const auto& core = op.selfPipeCore;
        return core.pipeStart == pipe.pipe && core.pipeEnd == pipe.pipe && core.core == pipe.core &&
               core.aivCore == pipe.aivCore;
    };
    for (const auto& op : plan) {
        if (!matches(op, pipes.first)) {
            continue;
        }
        for (auto waitIdx : op.setPipe) {
            if (matches(plan[waitIdx], pipes.second)) {
                dependencies.emplace_back(op.idx, waitIdx);
            }
        }
    }
    for (size_t i = 0; i + 1 < dependencies.size(); ++i) {
        const auto [set1, wait1] = dependencies[i];
        const auto [set2, wait2] = dependencies[i + 1];
        // A merge wholly after the blocked producer cannot reduce its event demand.
        if (!orderedOpList_.empty() && set1 > orderedOpList_.front()) {
            continue;
        }
        if (!orderedOpList_.empty() && set1 < orderedOpList_.front()) {
            const auto* instance = eventState_.Find(set1, wait1);
            // Removing an already returned event cannot relieve the blocked pool.
            if (instance == nullptr ||
                (instance->state != InstanceState::Allocated && instance->state != InstanceState::WaitInserted)) {
                continue;
            }
        }
        // The output is sorted by original operation indices, not replay order.
        // An acyclic backward edge would therefore emit wait1 before its set2.
        if (set1 >= set2 || wait1 >= wait2 || set2 >= wait1 ||
            std::find(plan[set2].setPipe.begin(), plan[set2].setPipe.end(), wait1) != plan[set2].setPipe.end()) {
            continue;
        }
        // Check the strengthening edge before removing either original dependency.
        // Source FIFO and destination FIFO then prove coverage of both old edges.
        auto candidate = plan;
        candidate[set2].setPipe.push_back(wait1);
        if (!BuildReplayIssueOrder(candidate, order)) {
            continue;
        }
        auto erase = [](std::vector<size_t>& indices, size_t idx) {
            indices.erase(std::remove(indices.begin(), indices.end(), idx), indices.end());
        };
        erase(candidate[set1].setPipe, wait1);
        erase(candidate[wait1].waitPipe, set1);
        erase(candidate[set2].setPipe, wait2);
        erase(candidate[wait2].waitPipe, set2);
        candidate[wait1].waitPipe.push_back(set2);
        APASS_LOG_INFO_F(Elements::Operation, "Event reuse replay merge: (%zu,%zu) and (%zu,%zu) become (%zu,%zu).",
                         set1, wait1, set2, wait2, set2, wait1);
        plan = std::move(candidate);
        return true;
    }
    return false;
}

bool PipeSync::AdvanceReturningWaitForReplay(std::vector<DepOp>& plan, const IntraEventCapacityFailure& failure,
                                             std::vector<size_t>& order) const
{
    const auto blocked = failure.setIdx;
    const auto& source = failure.pipes.first;
    if (plan[blocked].selfPipeCore.pipeStart != source.pipe) {
        return false;
    }
    const auto consumed = ProjectConsumedInstances(depOps_[blocked], source);
    for (const auto& instance : eventState_.Instances()) {
        if (!instance.intraCore || instance.state != InstanceState::Allocated || instance.setOp == nullptr) {
            continue;
        }
        if (instance.pipes.second != source || instance.setIdx >= blocked || instance.waitIdx <= blocked) {
            continue;
        }
        bool returnsSlot = false;
        for (const auto& proof : instance.setSnapshot) {
            const auto& resource = eventState_.ProofResource(proof);
            if (resource == GetEventResource(failure.pipes, resource.eventId) && eventState_.HasActive(resource) &&
                eventState_.CanAllocate(resource, instance.setSnapshot) &&
                !eventState_.CanAllocate(resource, consumed)) {
                returnsSlot = true;
                break;
            }
        }
        if (!returnsSlot) {
            continue;
        }
        const auto set = instance.setIdx;
        const auto wait = instance.waitIdx;
        // Pending merges may differ from the saved replay plan. Only move an
        // existing edge, preserving its old wait through destination FIFO.
        auto candidate = plan;
        auto& targets = candidate[set].setPipe;
        auto old = std::find(targets.begin(), targets.end(), wait);
        if (old == targets.end() || std::find(targets.begin(), targets.end(), blocked) != targets.end()) {
            continue;
        }
        *old = blocked;
        auto& sources = candidate[wait].waitPipe;
        sources.erase(std::remove(sources.begin(), sources.end(), set), sources.end());
        candidate[blocked].waitPipe.push_back(set);
        if (!BuildReplayIssueOrder(candidate, order)) {
            continue;
        }
        APASS_LOG_INFO_F(Elements::Operation, "Event reuse return wait: dependency (%zu,%zu) moves to (%zu,%zu).", set,
                         wait, set, blocked);
        plan = std::move(candidate);
        return true;
    }
    return false;
}

bool PipeSync::RewriteReturningDependency(std::vector<DepOp>& candidate, size_t consumer,
                                          const IntraEventCapacityFailure& failure)
{
    const auto& pipes = failure.pipes;
    const auto blocked = failure.setIdx;
    auto& targets = candidate[consumer].setPipe;
    // AddOpDep keeps one wait per destination pipe. Only the entry pipe matters here;
    // the consumer above must be a single-pipe op so its completed wait precedes this set.
    const auto oldWait = std::find_if(targets.begin(), targets.end(), [&](size_t idx) {
        const auto& target = candidate[idx].selfPipeCore;
        return target.pipeStart == pipes.first.pipe && target.core == pipes.first.core &&
               target.aivCore == pipes.first.aivCore;
    });
    if (oldWait != targets.end()) {
        // Destination FIFO covers the old wait after moving it earlier.
        // Keeping both edges would require two IDs in the reverse pool.
        if (*oldWait <= blocked) {
            // The existing return already reaches blocked through destination FIFO.
            return false;
        }
        auto& sources = candidate[*oldWait].waitPipe;
        sources.erase(std::remove(sources.begin(), sources.end(), consumer), sources.end());
        *oldWait = blocked;
    } else {
        targets.push_back(blocked);
    }
    candidate[blocked].waitPipe.push_back(consumer);
    return true;
}

bool PipeSync::AddReturningDependencyForReplay(std::vector<DepOp>& plan, const IntraEventCapacityFailure& failure,
                                               std::vector<size_t>& order) const
{
    const auto& pipes = failure.pipes;
    const auto blocked = failure.setIdx;
    if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) ||
        Platform::Instance().GetSoc().GetCoreWrap().IsCVMix() ||
        plan[blocked].selfPipeCore.pipeStart != pipes.first.pipe) {
        return false;
    }
    const auto consumed = ProjectConsumedInstances(depOps_[blocked], pipes.first);
    std::vector<size_t> consumers;
    for (const auto& instance : eventState_.Instances()) {
        if (instance.intraCore && instance.pipes == pipes && instance.state == InstanceState::WaitInserted &&
            instance.waitIdx < blocked && eventState_.HasActive(instance.id.resource) &&
            !eventState_.CanAllocate(instance.id.resource, consumed)) {
            consumers.push_back(instance.waitIdx);
        }
    }
    std::sort(consumers.begin(), consumers.end(), std::greater<size_t>());
    for (auto consumer : consumers) {
        const auto& core = plan[consumer].selfPipeCore;
        if (core.pipeStart != pipes.second.pipe || core.pipeEnd != pipes.second.pipe ||
            core.core != pipes.second.core || core.aivCore != pipes.second.aivCore ||
            std::find(plan[consumer].setPipe.begin(), plan[consumer].setPipe.end(), blocked) !=
                plan[consumer].setPipe.end()) {
            continue;
        }
        auto candidate = plan;
        if (!RewriteReturningDependency(candidate, consumer, failure)) {
            continue;
        }
        if (!BuildReplayIssueOrder(candidate, order)) {
            continue;
        }
        APASS_LOG_INFO_F(Elements::Operation, "Event reuse directed return: dependency (%zu,%zu).", consumer, blocked);
        plan = std::move(candidate);
        return true;
    }
    return false;
}

void PipeSync::ResetForReplay(const std::vector<DepOp>& plan, const std::vector<IssueQueue>& queues,
                              const std::vector<size_t>& order)
{
    depOps_ = plan;
    CacheReplayOrder(order);
    issueState_ = queues;
    std::unordered_set<size_t> queued;
    for (const auto& queue : queues) {
        queued.insert(queue.ops.begin(), queue.ops.end());
    }
    orderedOpList_ = {};
    for (auto idx : order) {
        if (queued.count(idx) != 0) {
            orderedOpList_.push(idx);
        }
    }
    eventState_ = EventInstanceTracker{};
    reclaimBarriers_.clear();
    freeEventId_.clear();
    crossCoreFreeEventId_.clear();
    crossCoreEventSenders_.fill(std::nullopt);
    crossCoreGenerations_.fill(0);
    crossCoreConsumed_.clear();
    crossCoreSnapshots_.clear();
    NoWaitCVPairs_.clear();
    setWaitPairMap_.clear();
    setOpMap.clear();
    waitOpMap.clear();
    forceSyncOps_.clear();
}

Status PipeSync::IssueSyncOp(Function& function, const std::vector<Operation*>& opLogPtr,
                             std::vector<IndexOp>& syncedOpLog, size_t& totalIssued, size_t& allIssued)
{
    bool eventIdDeadlock = false;
    bool mergeSearchExhausted = false;
    uint64_t eventIdDeadlockEnterTimes = 0;
    // Keep the pre-emission plan: pending-only recovery may have already edited
    // the working dependencies when a consumed pool eventually exhausts.
    auto replayPlan = depOps_;
    const auto initialQueues = issueState_;
    const auto initialLog = syncedOpLog;
    const std::unordered_set<Operation*> originalOps(opLogPtr.begin(), opLogPtr.end());
    InitCVEventIdQ();
    while (totalIssued < allIssued) {
        size_t issued = 0;
        if (IssueOpPipeSeq(function, opLogPtr, syncedOpLog, eventIdDeadlock, issued) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function IssueOpPipeSeq.");
            return FAILED;
        }
        totalIssued += issued;
        // eventIdDeadlockEnterTimes eventIdDeadlock syncedOpLog
        if (issued == 0) {
            if (eventIdDeadlockEnterTimes + 1 >= EVENTID_DEADLOCK_ENTER_TIME && !orderedOpList_.empty()) {
                const auto failure = AnalyzeIntraEventCapacity(depOps_[orderedOpList_.front()]);
                std::vector<size_t> replayOrder;
                if (failure && (AdvanceReturningWaitForReplay(replayPlan, *failure, replayOrder) ||
                                MergeDependenciesForReplay(replayPlan, failure->pipes, replayOrder) ||
                                AddReturningDependencyForReplay(replayPlan, *failure, replayOrder))) {
                    // A failed attempt's snapshots and generated ops must never
                    // survive into the rebuilt event-instance graph.
                    for (auto& op : function.Operations(false)) {
                        if (originalOps.count(&op) == 0) {
                            op.SetAsDeleted();
                        }
                    }
                    ResetForReplay(replayPlan, initialQueues, replayOrder);
                    syncedOpLog = initialLog;
                    function.EraseOperations(true, false);
                    totalIssued = 0;
                    eventIdDeadlock = false;
                    mergeSearchExhausted = false;
                    eventIdDeadlockEnterTimes = 0;
                    InitCVEventIdQ();
                    APASS_LOG_INFO_F(Elements::Operation,
                                     "Replaying sync generation after safe intra-core dependency recovery.");
                    continue;
                }
                // Drain only after pending merges, yielding and replay recovery have failed.
                if (failure &&
                    ReclaimIntraCoreEvents(function, orderedOpList_.front(), failure->pipes, syncedOpLog) == SUCCESS) {
                    eventIdDeadlock = false;
                    mergeSearchExhausted = false;
                    eventIdDeadlockEnterTimes = 0;
                    continue;
                }
            }
            // With no committed issue or recovery, yielding has rolled back its edits.
            if (ProcessDeadLock(eventIdDeadlockEnterTimes, eventIdDeadlock, syncedOpLog, &mergeSearchExhausted) !=
                SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed at function ProcessDeadLock.");
                return FAILED;
            }
            continue;
        }
        eventIdDeadlock = false;
        mergeSearchExhausted = false;
        eventIdDeadlockEnterTimes = 0;
    }
    return SUCCESS;
}

Status PipeSync::IssueOp(Function& function, const std::vector<Operation*>& opLogPtr, std::vector<IndexOp>& syncedOpLog)
{
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += issueState_[i].ops.size();
        APASS_LOG_DEBUG_F(Elements::Operation, "Pipe seq %d: %s %s", i, PipeSeqName(static_cast<PipeSeq>(i)).c_str(),
                          issueState_[i].DumpIssueQueue(opLogPtr).c_str());
    }
    if (IssueSyncOp(function, opLogPtr, syncedOpLog, totalIssued, allIssued) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "IssueOp failed with IssueSyncOp.");
        return FAILED;
    }
    if (totalIssued != allIssued) {
        APASS_LOG_ERROR_F(Elements::Operation, "Issue error, IssueOp failed.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "ALL op issued: %zu", totalIssued);
    return SUCCESS;
}

std::vector<PipeSync::PipePair> PipeSync::dataDepPair = {
    // PIPE_MTE1只有AIC PIPE_V只有AIV PIPE_M只有AIC PIPE_FIX只有AIC
    // PIPE_MTE3->PIPE_MTE2
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE3
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE2->PIPE_V
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE2
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE3->PIPE_V
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_MTE3
    {{PIPE_V, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_S->PIPE_V
    {{PIPE_S, CoreType::AIV}, {PIPE_V, CoreType::AIV}},
    // PIPE_V->PIPE_S
    {{PIPE_V, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_M
    {{PIPE_S, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_S
    {{PIPE_M, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE1
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_S
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_S->PIPE_MTE2
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE2, CoreType::AIV}},
    // PIPE_MTE2->PIPE_S
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE2, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_MTE3
    {{PIPE_S, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    {{PIPE_S, CoreType::AIV}, {PIPE_MTE3, CoreType::AIV}},
    // PIPE_MTE3->PIPE_S
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    {{PIPE_MTE3, CoreType::AIV}, {PIPE_S, CoreType::AIV}},
    // PIPE_S->PIPE_FIX
    {{PIPE_S, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_S
    {{PIPE_FIX, CoreType::AIC}, {PIPE_S, CoreType::AIC}},
    // PIPE_M->PIPE_MTE1
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_M
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE2
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_M
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_MTE3
    {{PIPE_M, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_M
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_M->PIPE_FIX
    {{PIPE_M, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_M
    {{PIPE_FIX, CoreType::AIC}, {PIPE_M, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE2
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE2->PIPE_MTE1
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_MTE3
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
    // PIPE_MTE3->PIPE_MTE1
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE1->PIPE_FIX
    {{PIPE_MTE1, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE1
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE1, CoreType::AIC}},
    // PIPE_MTE2->PIPE_FIX
    {{PIPE_MTE2, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE2
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE2, CoreType::AIC}},
    // PIPE_MTE3->PIPE_FIX
    {{PIPE_MTE3, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}},
    // PIPE_FIX->PIPE_MTE3
    {{PIPE_FIX, CoreType::AIC}, {PIPE_MTE3, CoreType::AIC}},
};

// mix aic/aiv core dependency pairs
std::vector<PipeSync::PipePair> PipeSync::mixCVDataDepPair = {
    // PIPE_FIX->PIPE_V
    {{PIPE_FIX, CoreType::AIC}, {PIPE_V, CoreType::AIV}}};

bool PipeSync::FindMaxOverlap(DataDepInfo& depInfo, int& maxOverlapDepIdx)
{
    if (depInfo.opDepList.size() < 2) {
        return false;
    }
    int maxOverlap = -1;
    for (int idx = 0; idx < static_cast<int>(depInfo.opDepList.size() - 1); idx++) {
        const auto [set1, wait1] = depInfo.opDepList[idx];
        const auto [set2, wait2] = depInfo.opDepList[idx + 1];
        // SynDependency delays the first wait to the later producer. Nested or
        // reversed endpoints cannot be merged with that rewrite.
        if (set1 >= set2 || wait1 >= wait2 || wait1 <= set2) {
            continue;
        }
        // max_overlap初始值为阈值，相邻两个依赖之间的重叠超过阈值，可以合并
        // max_overlap用来记录遍历到的最大overlap
        if ((depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first) > maxOverlap) {
            maxOverlapDepIdx = idx;
            maxOverlap = depInfo.opDepList[idx].second - depInfo.opDepList[idx + 1].first;
        }
    }
    if (maxOverlapDepIdx == -1) { // 8个依赖中，前后依赖都没有重叠，或者重叠全部小于阈值。
        return false;
    }
    return true;
}

Status PipeSync::MergePendingDependencies(int set1, int wait1, int set2, int wait2, int eventId1, int eventId2)
{
    std::unordered_map<size_t, DepOp> originalDeps{{static_cast<size_t>(set1), depOps_[set1]},
                                                   {static_cast<size_t>(wait1), depOps_[wait1]},
                                                   {static_cast<size_t>(set2), depOps_[set2]},
                                                   {static_cast<size_t>(wait2), depOps_[wait2]}};
    // AddOpDep may replace another wait on set2's target pipe. Keep its
    // reverse dependency available even if a later tracker transition fails.
    for (auto waitIdx : depOps_[set2].setPipe) {
        originalDeps.emplace(waitIdx, depOps_[waitIdx]);
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Event reuse pending merge: (%d,%d), id=%d and (%d,%d), id=%d become (%d,%d).", set1, wait1,
                     eventId1, set2, wait2, eventId2, set2, wait1);
    RemoveOpDep(depOps_[set1], depOps_[wait1]);
    RemoveOpDep(depOps_[set2], depOps_[wait2]);
    if (AddOpDep(depOps_[set2], depOps_[wait1]) != SUCCESS) {
        RestoreOpDeps(originalDeps);
        APASS_LOG_ERROR_F(Elements::Operation, "RelaxFakeDataDep failed at function AddOpDep.");
        return FAILED;
    }
    if (eventState_.MergePending(set1, wait1, set2, wait2) != SUCCESS) {
        RestoreOpDeps(originalDeps);
        return FAILED;
    }
    return SUCCESS;
}

Status PipeSync::SynDependency(int maxOverlapDepIdx, const DataDepInfo& depInfo, const PipePairEx& pipePairEx,
                               std::vector<IndexOp>& syncedOpLog)
{
    int set1 = depInfo.opDepList[maxOverlapDepIdx].first;
    int wait1 = depInfo.opDepList[maxOverlapDepIdx].second;
    int set2 = depInfo.opDepList[maxOverlapDepIdx + 1].first;
    int wait2 = depInfo.opDepList[maxOverlapDepIdx + 1].second;
    int eventId1 = depInfo.setOpEventIdList[maxOverlapDepIdx];
    int eventId2 = depInfo.setOpEventIdList[maxOverlapDepIdx + 1];
    int syncOpIdx1 = depInfo.setOpIdList[maxOverlapDepIdx];
    const auto* first = eventState_.Find(set1, wait1);
    const auto* second = eventState_.Find(set2, wait2);
    if (set1 >= set2 || wait1 >= wait2 || set2 >= wait1 || first == nullptr || second == nullptr ||
        depOps_[wait1].issued || depOps_[wait2].issued) {
        APASS_LOG_ERROR_F(Elements::Operation, "Dependency error, RelaxFakeDataDep failed.");
        return FAILED;
    }
    const auto& firstInstance = *first;
    const auto& secondInstance = *second;
    if (firstInstance.state != InstanceState::Allocated || secondInstance.state != InstanceState::Allocated ||
        !firstInstance.intraCore || !secondInstance.intraCore || firstInstance.pipes != pipePairEx ||
        secondInstance.pipes != pipePairEx || firstInstance.id.resource.eventId != eventId1 ||
        secondInstance.id.resource.eventId != eventId2 ||
        firstInstance.setOp != &syncedOpLog[syncOpIdx1].second.get()) {
        return FAILED;
    }
    // Validate the core pair before changing dependency or event-instance state.
    if (!(IsMixCvCore(pipePairEx.first.core, pipePairEx.second.core)) &&
        (pipePairEx.first.core != pipePairEx.second.core)) {
        APASS_LOG_ERROR_F(Elements::Operation, "CoreType is not same, SynDependency failed.");
        return FAILED;
    }
    if (MergePendingDependencies(set1, wait1, set2, wait2, eventId1, eventId2) != SUCCESS) {
        return FAILED;
    }
    // Rewritten dependencies cannot inherit an old cross-core consumption snapshot.
    crossCoreSnapshots_.erase({set1, wait1});
    crossCoreSnapshots_.erase({set2, wait2});
    GetFreeEventIdQueue(pipePairEx).push_back(eventId1);
    setWaitPairMap_.erase({set1, wait1});
    setWaitPairMap_.erase({set2, wait2});
    setWaitPairMap_[{set2, wait1}] = eventId2;
    setOpMap.erase(firstInstance.setOp);
    setOpMap[secondInstance.setOp] = oriOpList_[wait1];
    // 将靠前的一对有依赖关系op中插入的SYNC_SRC op删除
    syncedOpLog[syncOpIdx1].second.get().SetAsDeleted();
    syncedOpLog.erase(syncedOpLog.begin() + syncOpIdx1);
    return SUCCESS;
}

Status PipeSync::GetDepInfo(std::vector<IndexOp>& syncedOpLog, const PipePairEx& pipePairEx, DataDepInfo& depInfo)
{
    depInfo.setp = pipePairEx.first.pipe;
    depInfo.setc = pipePairEx.first.core;
    depInfo.waitp = pipePairEx.second.pipe;
    depInfo.waitc = pipePairEx.second.core;
    depInfo.setaivc = pipePairEx.first.aivCore;
    depInfo.waitaivc = pipePairEx.second.aivCore;
    // Exhausted slots can already have a wait, without a returning proof.
    // Only pending waits may participate in dependency merging.
    std::unordered_map<Operation*, const EventInstance*> pending;
    // Each directed slot has at most one pending generation. Look it up
    // directly instead of walking the history of retired and cancelled sets.
    for (int id = INTRA_CORE_START_EVENT_ID; id <= GetMaxEventId(pipePairEx); ++id) {
        const auto* instance = eventState_.FindActive(GetEventResource(pipePairEx, id));
        if (instance != nullptr && instance->state == InstanceState::Allocated && instance->setOp != nullptr &&
            !depOps_[instance->waitIdx].issued) {
            pending.emplace(instance->setOp, instance);
        }
    }
    depInfo.opDepList.clear();
    depInfo.setOpIdList.clear();
    depInfo.setOpEventIdList.clear();
    if (pending.empty()) {
        return SUCCESS;
    }
    // Each generated set occurs once in the log. Pending sets are usually near
    // its tail; stop once all are found, then restore the original candidate order.
    size_t remaining = pending.size();
    for (size_t i = syncedOpLog.size(); i-- > 0;) {
        auto it = pending.find(&syncedOpLog[i].second.get());
        if (it != pending.end()) {
            const auto& instance = *it->second;
            depInfo.opDepList.emplace_back(instance.setIdx, instance.waitIdx);
            depInfo.setOpIdList.push_back(static_cast<int>(i));
            depInfo.setOpEventIdList.push_back(instance.id.resource.eventId);
            if (--remaining == 0) {
                break;
            }
        }
    }
    std::reverse(depInfo.opDepList.begin(), depInfo.opDepList.end());
    std::reverse(depInfo.setOpIdList.begin(), depInfo.setOpIdList.end());
    std::reverse(depInfo.setOpEventIdList.begin(), depInfo.setOpEventIdList.end());
    return SUCCESS;
}

std::vector<PipeSync::PipeCoreRealEx> PipeSync::cvPipeCoreEx = {
    {PIPE_S, CoreType::AIV, AIVCore::AIV0},           {PIPE_S, CoreType::AIV, AIVCore::AIV1},
    {PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},        {PIPE_MTE2, CoreType::AIV, AIVCore::AIV1},
    {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0},        {PIPE_MTE3, CoreType::AIV, AIVCore::AIV1},
    {PIPE_V, CoreType::AIV, AIVCore::AIV0},           {PIPE_V, CoreType::AIV, AIVCore::AIV1},
    {PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED},    {PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED},
    {PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED}, {PIPE_MTE3, CoreType::AIC, AIVCore::UNSPECIFIED},
    {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED},    {PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED},
};

std::string PipeSyncTypes::DataDepInfo::DumpDataDepInfo(const std::vector<IndexOp>& syncedOpLog,
                                                        const std::vector<Operation*>& oriOpList)
{
    std::stringstream ss;
    ss << "    CV_SYNC_SRC magic: ";
    for (auto sync : setOpIdList) {
        ss << syncedOpLog[sync].second.get().GetOpMagic() << " ";
    }
    ss << "\n    CV EventID: ";
    for (auto id : setOpEventIdList) {
        ss << id << " ";
    }
    ss << "\n    Set Wait Pair: \n";
    for (auto [setidx, waitidx] : opDepList) {
        ss << "        " << setidx << " " << oriOpList[setidx]->GetOpMagic() << " "
           << oriOpList[setidx]->GetOpcodeStr();
        ss << "  " << waitidx << " " << oriOpList[waitidx]->GetOpMagic() << " " << oriOpList[waitidx]->GetOpcodeStr()
           << "\n";
    }
    return ss.str();
}

Status PipeSync::RelaxFakeDataDep(std::vector<IndexOp>& syncedOpLog, bool& merged)
{
    // 合并阈值。只有前后两个set-wait对之间的重叠（以op数量度量）超过该阈值时，才对这两个set-wait对进行合并。
    std::vector<PipePairEx> dataDepPairEx;
    for (const auto& pipePair : dataDepPair) {
        if (pipePair.first.core == CoreType::AIC) {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::UNSPECIFIED);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::UNSPECIFIED);
            PipePairEx ppEx = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx);
        } else {
            PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV0);
            PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV0);
            PipePairEx ppEx1 = {pp1, pp2};
            dataDepPairEx.emplace_back(ppEx1);
            PipeCoreRealEx pp3(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV1);
            PipeCoreRealEx pp4(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV1);
            PipePairEx ppEx2 = {pp3, pp4};
            dataDepPairEx.emplace_back(ppEx2);
        }
    }

    if (Platform::Instance().GetSoc().GetCoreWrap().IsCVMix()) {
        for (const auto& pipePair : mixCVDataDepPair) {
            if (pipePair.first.core == CoreType::AIC && pipePair.second.core == CoreType::AIV) {
                PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::UNSPECIFIED);
                PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::AIV0);
                PipePairEx ppEx = {pp1, pp2};
                dataDepPairEx.emplace_back(ppEx);
            } else if (pipePair.first.core == CoreType::AIV && pipePair.second.core == CoreType::AIC) {
                PipeCoreRealEx pp1(pipePair.first.pipe, pipePair.first.core, AIVCore::AIV0);
                PipeCoreRealEx pp2(pipePair.second.pipe, pipePair.second.core, AIVCore::UNSPECIFIED);
                PipePairEx ppEx = {pp1, pp2};
                dataDepPairEx.emplace_back(ppEx);
            }
        }
    }

    // for cv mix architecture, normalize all non PIPE_V instructions to AIC, to maintain consistency in a singular
    // core since they share a singular eventid
    if (Platform::Instance().GetSoc().GetCoreWrap().IsCVMix()) {
        std::unordered_set<PipePairEx, PipePairExHash> seen;
        std::vector<PipePairEx> normalized;
        for (const auto& pe : dataDepPairEx) {
            PipePairEx npe = {NormalizeToAICCoreForCVMixSync(pe.first), NormalizeToAICCoreForCVMixSync(pe.second)};
            if (seen.insert(npe).second) {
                normalized.emplace_back(npe);
            }
        }
        dataDepPairEx = std::move(normalized);
    }

    for (const auto& pipePairEx : dataDepPairEx) {
        // 合并CV同步释放cv eventid 当前逻辑不需要 (if mix cv core, need handle mix cv core sync)
        if (!(IsMixCvCore(pipePairEx.first.core, pipePairEx.second.core)) &&
            (pipePairEx.first.core != pipePairEx.second.core)) {
            continue;
        }
        if (HasFreeEventId(pipePairEx)) {
            continue;
        }
        // 找到free id exhausted的pipe pair
        DataDepInfo depInfo;
        if (GetDepInfo(syncedOpLog, pipePairEx, depInfo) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "GetDepInfo failed.");
            return FAILED;
        }
        // 找到前一个依赖和后一个依赖间重叠最大的地方
        int maxOverlapDepIdx{-1};
        if (!(FindMaxOverlap(depInfo, maxOverlapDepIdx))) {
            continue;
        }
        // 合并依赖
        if (SynDependency(maxOverlapDepIdx, depInfo, pipePairEx, syncedOpLog) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SynDependency failed.");
            return FAILED;
        }
        merged = true;
    }

    return SUCCESS;
}

bool PipeSync::GenSyncOp(PipeCoreRealEx set, PipeCoreRealEx wait, int eventId, bool isSet, Operation& op)
{
    if (set.core != wait.core) {
        if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
            op.SetOpCode(isSet ? Opcode::OP_CV_SYNC_SRC : Opcode::OP_CV_SYNC_DST);
        }
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && !isSet && wait.core == CoreType::AIV) {
            set.pipe = PipeType::PIPE_V;
        }
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
        return true;
    }
    if (set.core == CoreType::AIV && set.aivCore != wait.aivCore) {
        APASS_LOG_WARN_F(Elements::Operation, "Find dependency between AIV0 and AIV1, remove it.");
        return false;
    }
    if (set.pipe != wait.pipe) {
        op.SetOpCode(isSet ? Opcode::OP_SYNC_SRC : Opcode::OP_SYNC_DST);
        op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
        return true;
    }
    if (isSet || set.pipe == PipeType::PIPE_S) {
        return false;
    }
    // 同步相关的信息放在operation属性里
    op.syncQueue_ = {set.pipe, wait.pipe, set.core, wait.core, eventId, set.aivCore, wait.aivCore};
    if (set.core == CoreType::AIV && set.pipe == PipeType::PIPE_V && wait.pipe == PipeType::PIPE_V) {
        if ((Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) ||
            (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()))) {
            return false;
        }
        op.SetOpCode(Opcode::OP_BAR_V);
        return true;
    }
    op.SetOpCode(Opcode::OP_BAR_M);
    return true;
}

std::deque<int>* PipeSync::GetCrossCoreEventIdQPtr(const PipePairEx& pp)
{
    if (pp.first.core == CoreType::AIV) {
        return &crossCoreFreeEventId_[pp.first][0];
    }
    if (pp.second.aivCore == AIVCore::AIV0) {
        return &crossCoreFreeEventId_[pp.first][0];
    }
    if (pp.second.aivCore != AIVCore::AIV1) {
        APASS_LOG_ERROR_F(Elements::Operation, "AIV op is neither AIV0 nor AIV1, GetEventId failed.");
        return nullptr;
    }
    return &crossCoreFreeEventId_[pp.first][1];
}

void PipeSync::RemoveEventIdFromCrossCoreQueues(int eventId)
{
    for (auto& [otherP, otherQArr] : crossCoreFreeEventId_) {
        (void)otherP;
        for (auto& otherQ : otherQArr) {
            otherQ.erase(std::remove(otherQ.begin(), otherQ.end(), eventId), otherQ.end());
        }
    }
}

Status PipeSync::GetEventId(const PipePairEx& pp, int& eventId, size_t setIdx, size_t waitIdx,
                            std::vector<IndexOp>& syncedOpLog, Function& function)
{
    if (pp.first.pipe == pp.second.pipe && pp.first.core == pp.second.core) {
        // Pipe Barrier
        eventId = -1;
        return SUCCESS;
    }

    // CV eventid获取
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) && pp.first.core != pp.second.core) {
        std::deque<int>* cvEventIdQPtr = GetCrossCoreEventIdQPtr(pp);
        if (cvEventIdQPtr == nullptr) {
            return FAILED;
        }
        auto& cvEventIdQ = *cvEventIdQPtr;
        if (UseCrossCoreProof()) {
            const size_t pool = (pp.first.core == CoreType::AIV ? pp.first.aivCore : pp.second.aivCore) ==
                                        AIVCore::AIV0 ?
                                    0 :
                                    1;
            RefreshCrossCoreEventIds(pp.first, pool, crossCoreConsumed_[pp.first]);
            if (cvEventIdQ.empty()) {
                // InjectSync reserves the whole output group before emitting input waits.
                // Draining here could cancel sets after those waits have already committed.
                APASS_LOG_ERROR_F(Elements::Operation, "Cross-core ID reservation is exhausted at op %zu.", setIdx);
                return FAILED;
            }
            eventId = cvEventIdQ.front();
            ++crossCoreGenerations_[eventId];
            crossCoreEventSenders_[eventId] = pp.first.core;
        }
        if (!cvEventIdQ.empty()) {
            eventId = cvEventIdQ.front();
            cvEventIdQ.pop_front();
            RemoveEventIdFromCrossCoreQueues(eventId);
            return SUCCESS;
        }

        // insert ffts to relax all eventid
        if (RemoveSetIntraBlockAndOpDep(syncedOpLog) != SUCCESS) {
            return FAILED;
        }
        AddCrossCoreForceSyncOps(setIdx, syncedOpLog, function);
        InitCVEventIdQ();
        std::deque<int>* newCvEventIdQPtr = GetCrossCoreEventIdQPtr(pp);
        if (newCvEventIdQPtr == nullptr || newCvEventIdQPtr->empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, "EventId still exhausted after InitCVEventIdQ, GetEventId failed.");
            return FAILED;
        }
        eventId = newCvEventIdQPtr->front();
        newCvEventIdQPtr->pop_front();
        RemoveEventIdFromCrossCoreQueues(eventId);
        return SUCCESS;
    }

    return AllocateIntraEvent(pp, eventId, setIdx, waitIdx);
}

size_t PipeSync::FindReclaimPrefix(size_t boundary, const std::unordered_set<Operation*>& emitted) const
{
    for (size_t i = 0; i < boundary; ++i) {
        if (emitted.count(oriOpList_[i]) == 0) {
            boundary = i;
            break;
        }
    }
    // A prefix wait must never depend on a suffix set: the barrier would
    // otherwise block the producer needed to finish draining that prefix.
    bool moved = true;
    while (moved && boundary != 0) {
        moved = false;
        for (size_t i = boundary; i < depOps_.size(); ++i) {
            for (auto wait : depOps_[i].setPipe) {
                if (wait < boundary) {
                    boundary = wait;
                    moved = true;
                }
            }
        }
    }
    return boundary;
}

Status PipeSync::StageReclaimedEvents(size_t boundary, const PipePairEx& pp, bool lite,
                                      const std::unordered_set<Operation*>& emitted, EventInstanceTracker& stagedState,
                                      std::vector<size_t>& covered) const
{
    for (size_t i = 0; i < eventState_.Instances().size(); ++i) {
        const auto& instance = eventState_.Instances()[i];
        if (!instance.intraCore || instance.state != InstanceState::WaitInserted ||
            (!lite &&
             (instance.pipes.first.core != pp.first.core || instance.pipes.first.aivCore != pp.first.aivCore ||
              instance.pipes.second.core != pp.first.core || instance.pipes.second.aivCore != pp.first.aivCore)) ||
            instance.setIdx >= boundary || instance.waitIdx >= boundary) {
            continue;
        }
        if (emitted.count(instance.setOp) == 0 || emitted.count(instance.waitOp) == 0 ||
            stagedState.RetireConsumed(instance.setIdx, instance.waitIdx) != SUCCESS) {
            return FAILED;
        }
        covered.push_back(i);
    }
    return SUCCESS;
}

bool PipeSync::HasReclaimCapacity(size_t blockedIdx, size_t boundary, const PipePairEx& pp,
                                  const EventInstanceTracker& stagedState, size_t covered)
{
    // Pending sets are deliberately preserved: a drain does not consume flags
    // whose waits are still in the suffix, and cannot authorize their reuse.
    const auto consumed = ProjectConsumedInstances(depOps_[blockedIdx], pp.first);
    size_t available = 0;
    for (int id = INTRA_CORE_START_EVENT_ID; id <= GetMaxEventId(pp); ++id) {
        available += stagedState.CanAllocate(GetEventResource(pp, id), consumed) ? 1 : 0;
    }
    size_t required = 0;
    const auto& source = depOps_[blockedIdx].selfPipeCore;
    for (auto wait : depOps_[blockedIdx].setPipe) {
        const auto& target = depOps_[wait].selfPipeCore;
        const PipePairEx pair{{source.pipeEnd, source.core, source.aivCore},
                              {target.pipeStart, target.core, target.aivCore}};
        required += pair == pp ? 1 : 0;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Event reuse prefix drain: blocked=%zu, boundary=%zu, covered=%zu, available=%zu, required=%zu.",
                     blockedIdx, boundary, covered, available, required);
    return covered != 0 && available >= required;
}

Status PipeSync::ReclaimIntraCoreEvents(Function& function, size_t boundary, const PipePairEx& pp,
                                        std::vector<IndexOp>& syncedOpLog)
{
    const bool lite = IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch());
    if (boundary == 0 || boundary >= depOps_.size() || !IsIntraCoreEvent(pp) ||
        (!lite && (pp.first.core != pp.second.core || pp.first.aivCore != pp.second.aivCore))) {
        return FAILED;
    }
    const auto blockedIdx = boundary;
    // Replay may issue a later producer before an earlier consumer. Drain only
    // a complete original-order prefix, not the entire issued scheduling prefix.
    std::unordered_set<Operation*> emitted;
    for (const auto& entry : syncedOpLog) {
        emitted.insert(&entry.second.get());
    }
    boundary = FindReclaimPrefix(boundary, emitted);
    if (boundary == 0) {
        return FAILED;
    }
    auto stagedState = eventState_;
    std::vector<size_t> covered;
    if (StageReclaimedEvents(boundary, pp, lite, emitted, stagedState, covered) != SUCCESS) {
        return FAILED;
    }
    if (!HasReclaimCapacity(blockedIdx, boundary, pp, stagedState, covered.size())) {
        return FAILED;
    }
    uint64_t insertIdx = boundary * SEQUENCE_IDX - RECLAIM_PREFIX_OFFSET;
    for (const auto& entry : syncedOpLog) {
        if (entry.first > insertIdx && entry.first < boundary * SEQUENCE_IDX - HALF_SEQUENCE_IDX) {
            insertIdx = entry.first;
        }
    }
    if (insertIdx + 1 >= boundary * SEQUENCE_IDX - HALF_SEQUENCE_IDX) {
        return FAILED;
    }
    std::vector<IndexOp> generated;
    CreateForceSyncOp(Opcode::OP_BAR_ALL, PipeType::PIPE_ALL, pp.first.core, pp.first.aivCore, insertIdx, function,
                      generated);
    ReclaimBarrier barrier;
    barrier.cores = {{pp.first.core, pp.first.aivCore}};
    barrier.ops = {&generated.front().second.get()};
    stagedState.ForgetProofs();
    eventState_ = std::move(stagedState);
    for (auto index : covered) {
        const auto& instance = eventState_.Instances()[index];
        auto& ids = GetFreeEventIdQueue(instance.pipes);
        if (std::find(ids.begin(), ids.end(), instance.id.resource.eventId) == ids.end()) {
            ids.push_back(instance.id.resource.eventId);
        }
    }
    syncedOpLog.insert(syncedOpLog.end(), generated.begin(), generated.end());
    reclaimBarriers_.push_back(std::move(barrier));
    APASS_LOG_INFO_F(Elements::Operation, "Drained core %d/%d before op %zu; reclaimed %zu consumed intra-core events.",
                     static_cast<int>(pp.first.core), static_cast<int>(pp.first.aivCore), boundary, covered.size());
    return SUCCESS;
}

Status PipeSync::RemoveSetIntraBlockAndOpDep(std::vector<IndexOp>& syncedOpLog)
{
    for (auto& [key, op] : NoWaitCVPairs_) {
        if (eventState_.CancelPending(key.first, key.second) != SUCCESS) {
            return FAILED;
        }
        crossCoreSnapshots_.erase(key);
        RemoveOpDep(depOps_[key.first], depOps_[key.second]);
        op->SetAsDeleted();
        setOpMap.erase(op);
        Operation* opPtr = op;
        syncedOpLog.erase(std::remove_if(syncedOpLog.begin(), syncedOpLog.end(),
                                         [opPtr](const IndexOp& entry) { return &entry.second.get() == opPtr; }),
                          syncedOpLog.end());
    }
    NoWaitCVPairs_.clear();
    crossCoreFreeEventId_.clear();
    return SUCCESS;
}

void PipeSync::CreateForceSyncOp(Opcode opcode, PipeType pipe, CoreType core, AIVCore aivCore, uint64_t& insertIdx,
                                 Function& function, std::vector<IndexOp>& syncedOpLog)
{
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    Operation& op = irBuilder_.CreateTensorOpStmt(function, opcode, input, output);
    op.syncQueue_ = {pipe, pipe, core, core, EVENT_ID7, aivCore, aivCore};
    syncedOpLog.emplace_back(std::make_pair(++insertIdx, std::ref(op)));
    op.SetAIVCore(aivCore);
    forceSyncOps_.insert(&op);
}

void PipeSync::CreateBarAllOp(AIVCore aivCore, uint64_t& insertIdx, Function& function,
                              std::vector<IndexOp>& syncedOpLog)
{
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    Operation& op = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_BAR_ALL, input, output);
    op.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1, aivCore, aivCore};
    syncedOpLog.emplace_back(std::make_pair(++insertIdx, std::ref(op)));
    op.SetAIVCore(aivCore);
    forceSyncOps_.insert(&op);
}

void PipeSync::AddCrossCoreForceSyncOps(size_t setIdx, std::vector<IndexOp>& syncedOpLog, Function& function)
{
    PipeType prevAicPipe = PipeType::PIPE_FIX;
    PipeType prevAiv0Pipe = PipeType::PIPE_MTE3;
    PipeType prevAiv1Pipe = PipeType::PIPE_MTE3;

    uint64_t insertIdx = setIdx * SEQUENCE_IDX;

    CreateBarAllOp(AIVCore::UNSPECIFIED, insertIdx, function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_FFTS_CROSS_CORE_SYNC, prevAicPipe, CoreType::AIC, AIVCore::UNSPECIFIED, insertIdx,
                      function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED, insertIdx,
                      function, syncedOpLog);
    CreateBarAllOp(AIVCore::UNSPECIFIED, insertIdx, function, syncedOpLog);
    CreateBarAllOp(AIVCore::AIV0, insertIdx, function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV0, insertIdx, function,
                      syncedOpLog);
    CreateForceSyncOp(Opcode::OP_FFTS_CROSS_CORE_SYNC, prevAiv0Pipe, CoreType::AIV, AIVCore::AIV0, insertIdx, function,
                      syncedOpLog);
    CreateBarAllOp(AIVCore::AIV0, insertIdx, function, syncedOpLog);
    CreateBarAllOp(AIVCore::AIV1, insertIdx, function, syncedOpLog);
    CreateForceSyncOp(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV1, insertIdx, function,
                      syncedOpLog);
    CreateForceSyncOp(Opcode::OP_FFTS_CROSS_CORE_SYNC, prevAiv1Pipe, CoreType::AIV, AIVCore::AIV1, insertIdx, function,
                      syncedOpLog);
    CreateBarAllOp(AIVCore::AIV1, insertIdx, function, syncedOpLog);
}

Status PipeSync::GetCrossCoreBarrierBoundary(size_t idx, size_t instructionCount,
                                             const std::vector<IndexOp>& syncedOpLog, uint64_t& insertIdx) const
{
    if (idx == 0) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cannot insert a cross-core barrier before the first operation.");
        return FAILED;
    }
    std::unordered_set<const Operation*> emitted;
    insertIdx = 0;
    for (const auto& entry : syncedOpLog) {
        // EnqueueOp emits future VIEW/NOP operations eagerly. They do not belong to this prefix.
        if (entry.first < idx * SEQUENCE_IDX) {
            emitted.insert(&entry.second.get());
            insertIdx = std::max(insertIdx, entry.first);
        }
    }
    for (size_t i = 0; i < idx; ++i) {
        if (emitted.count(oriOpList_[i]) == 0) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cross-core barrier before op %zu requires a complete emitted prefix; op %zu is missing.",
                              idx, i);
            return FAILED;
        }
    }
    const uint64_t waitAnchor = idx * SEQUENCE_IDX - HALF_SEQUENCE_IDX;
    if (insertIdx >= waitAnchor || instructionCount >= waitAnchor - insertIdx) {
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "Insufficient sequence space before op %zu: need %zu slots, insertion %llu, wait anchor %llu.", idx,
            instructionCount, static_cast<unsigned long long>(insertIdx), static_cast<unsigned long long>(waitAnchor));
        return FAILED;
    }
    for (const auto& pending : NoWaitCVPairs_) {
        if (pending.first.first >= idx || pending.first.second < idx) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Pending cross-core edge %zu -> %zu does not cross the barrier before op %zu.",
                              pending.first.first, pending.first.second, idx);
            return FAILED;
        }
    }
    return SUCCESS;
}

bool PipeSync::UseCrossCoreProof() const
{
    return enableLocalCrossCoreBarrier_ && Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
}

void PipeSync::MergeCrossCoreProof(CrossCoreClock& clock, const CrossCoreSnapshot& snapshot)
{
    for (size_t id = 0; id < clock.size(); ++id) {
        clock[id] = std::max(clock[id], snapshot.consumed[id]);
    }
    if (snapshot.eventId >= 0) {
        clock[snapshot.eventId] = std::max(clock[snapshot.eventId], snapshot.generation);
    }
}

PipeSync::CrossCoreClock PipeSync::ProjectCrossCoreConsumption(size_t idx) const
{
    const auto& current = depOps_[idx].selfPipeCore;
    const PipeCoreRealEx source(current.pipeEnd, current.core, current.aivCore);
    CrossCoreClock clock{};
    auto known = crossCoreConsumed_.find(source);
    if (known != crossCoreConsumed_.end()) {
        clock = known->second;
    }
    // A multi-pipe operation does not establish entry-pipe -> exit-pipe FIFO order.
    if (current.pipeStart == current.pipeEnd) {
        for (size_t producer : depOps_[idx].waitPipe) {
            auto snapshot = crossCoreSnapshots_.find({producer, idx});
            if (snapshot != crossCoreSnapshots_.end()) {
                MergeCrossCoreProof(clock, snapshot->second);
            }
        }
    }
    return clock;
}

bool PipeSync::CanReuseCrossCoreEventId(int eventId, const CrossCoreClock& clock) const
{
    return crossCoreGenerations_[eventId] != std::numeric_limits<uint64_t>::max() &&
           (!crossCoreEventSenders_[eventId].has_value() ||
            (crossCoreGenerations_[eventId] != 0 && clock[eventId] == crossCoreGenerations_[eventId]));
}

void PipeSync::RefreshCrossCoreEventIds(const PipeCoreRealEx& source, size_t pool, const CrossCoreClock& clock)
{
    auto& queue = crossCoreFreeEventId_[source][source.core == CoreType::AIC ? pool : 0];
    const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
    queue.erase(std::remove_if(queue.begin(), queue.end(),
                               [&](int id) {
                                   return id < base || id >= base + CROSS_CORE_EVENT_NUM ||
                                          !CanReuseCrossCoreEventId(id, clock);
                               }),
                queue.end());
    // Keep fresh IDs in FIFO order. Reusable IDs are appended deterministically.
    for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
        if (crossCoreEventSenders_[id].has_value() && CanReuseCrossCoreEventId(id, clock)) {
            PushEventIdIfAbsent(queue, id);
        }
    }
}

Status PipeSync::PlanCrossCoreBarriers(size_t idx, const std::array<size_t, NUM2>& demand, CrossCoreBarrierPlan& plan)
{
    const auto& current = depOps_[idx].selfPipeCore;
    const PipeCoreRealEx source(current.pipeEnd, current.core, current.aivCore);
    const auto projected = ProjectCrossCoreConsumption(idx);
    size_t exhaustedPools = 0;
    for (size_t pool = 0; pool < demand.size(); ++pool) {
        if (demand[pool] > CROSS_CORE_EVENT_NUM) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Operation %zu requires %zu cross-core IDs from pool %zu of size %d.", idx, demand[pool],
                              pool, CROSS_CORE_EVENT_NUM);
            return FAILED;
        }
        if (demand[pool] == 0) {
            continue;
        }
        RefreshCrossCoreEventIds(source, pool, projected);
        if (demand[pool] <= crossCoreFreeEventId_[source][current.core == CoreType::AIC ? pool : 0].size()) {
            continue;
        }
        ++plan.localCount;
        ++exhaustedPools;
        const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
        for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
            if (!crossCoreEventSenders_[id].has_value() ||
                crossCoreGenerations_[id] > std::numeric_limits<uint64_t>::max() - NUM2) {
                continue;
            }
            // H remains available only on its final receiving S pipe, ordered after the final wait.
            const bool canUseHandshake = source.pipe == PIPE_S && source.core != *crossCoreEventSenders_[id];
            if (demand[pool] <= CROSS_CORE_EVENT_NUM - 1 + static_cast<size_t>(canUseHandshake)) {
                plan.handshakeIds[pool] = id;
                break;
            }
        }
        plan.globalBarrier = plan.globalBarrier || plan.handshakeIds[pool] == -1;
    }
    // A single local handshake is scoped to one C/V pool. If both pools are
    // exhausted for this operation, drain them together with the legacy
    // cross-core barrier instead of emitting two independent local barriers.
    plan.globalBarrier = plan.globalBarrier || exhaustedPools > 1;
    return SUCCESS;
}

Status PipeSync::EnsureCrossCoreCapacity(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    const auto& current = depOps_[idx].selfPipeCore;
    std::array<size_t, NUM2> demand{};
    for (size_t consumer : depOps_[idx].setPipe) {
        const auto& target = depOps_[consumer].selfPipeCore;
        if (current.core == target.core) {
            continue;
        }
        const auto vector = current.core == CoreType::AIV ? current.aivCore : target.aivCore;
        if (vector != AIVCore::AIV0 && vector != AIVCore::AIV1) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cross-core dependency %zu -> %zu has invalid AIV core %d; expected AIV0 or AIV1.", idx,
                              consumer, static_cast<int>(vector));
            return FAILED;
        }
        ++demand[vector == AIVCore::AIV0 ? 0 : 1];
    }
    if (demand[0] == 0 && demand[1] == 0) {
        return SUCCESS;
    }
    // A barrier can cancel an input wait that carried proof for the other pool. Recompute after each reset.
    for (size_t attempt = 0;; ++attempt) {
        CrossCoreBarrierPlan plan;
        if (PlanCrossCoreBarriers(idx, demand, plan) != SUCCESS) {
            return FAILED;
        }
        if (plan.localCount == 0) {
            return SUCCESS;
        }
        if (attempt == NUM2) {
            APASS_LOG_ERROR_F(Elements::Operation, "Cross-core capacity still insufficient at op %zu after %zu drains.",
                              idx, attempt);
            return FAILED;
        }
        constexpr size_t localInstructionCount = 6;
        constexpr size_t globalInstructionCount = 12;
        uint64_t insertIdx = 0;
        if (GetCrossCoreBarrierBoundary(
                idx, plan.globalBarrier ? globalInstructionCount : plan.localCount * localInstructionCount, syncedOpLog,
                insertIdx) != SUCCESS) {
            return FAILED;
        }
        if (plan.globalBarrier) {
            if (AddGlobalCrossCoreBarrier(function, idx, insertIdx, syncedOpLog) != SUCCESS) {
                return FAILED;
            }
            continue;
        }
        for (size_t pool = 0; pool < plan.handshakeIds.size(); ++pool) {
            if (plan.handshakeIds[pool] == -1) {
                continue;
            }
            if (AddLocalCrossCoreBarrier(function, idx, pool, plan.handshakeIds[pool], insertIdx, syncedOpLog) !=
                SUCCESS) {
                return FAILED;
            }
            if (ResetCrossCorePool(pool, plan.handshakeIds[pool], syncedOpLog) != SUCCESS) {
                return FAILED;
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "Inserted 1C1V barrier before op %zu for pool %zu with event ID %d.",
                              idx, pool, plan.handshakeIds[pool]);
        }
    }
}

Status PipeSync::AddGlobalCrossCoreBarrier(Function& function, size_t boundary, uint64_t insertIdx,
                                           std::vector<IndexOp>& syncedOpLog)
{
    for (const auto& pending : NoWaitCVPairs_) {
        setWaitPairMap_.erase(pending.first);
    }
    if (RemoveSetIntraBlockAndOpDep(syncedOpLog) != SUCCESS) {
        return FAILED;
    }
    std::vector<IndexOp> forceSyncs;
    AddCrossCoreForceSyncOps(0, forceSyncs, function);
    ReclaimBarrier barrier;
    barrier.cores = {
        {CoreType::AIC, AIVCore::UNSPECIFIED}, {CoreType::AIV, AIVCore::AIV0}, {CoreType::AIV, AIVCore::AIV1}};
    for (auto& entry : forceSyncs) {
        syncedOpLog.emplace_back(insertIdx + entry.first, entry.second);
        barrier.ops.push_back(&entry.second.get());
    }
    if (RecordCrossCoreDrain(boundary, syncedOpLog, std::move(barrier)) != SUCCESS) {
        return FAILED;
    }
    InitCVEventIdQ();
    crossCoreEventSenders_.fill(std::nullopt);
    crossCoreConsumed_.clear();
    crossCoreSnapshots_.clear();
    return SUCCESS;
}

Status PipeSync::RecordCrossCoreDrain(size_t boundary, const std::vector<IndexOp>& syncedOpLog, ReclaimBarrier barrier)
{
    if (barrier.ops.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cross-core drain before op %zu has no instructions.", boundary);
        return FAILED;
    }
    const auto first = std::find_if(syncedOpLog.begin(), syncedOpLog.end(),
                                    [&](const IndexOp& entry) { return &entry.second.get() == barrier.ops.front(); });
    if (first == syncedOpLog.end()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Cross-core drain before op %zu is missing its first instruction.",
                          boundary);
        return FAILED;
    }
    std::unordered_set<Operation*> emitted;
    for (const auto& entry : syncedOpLog) {
        if (entry.first < first->first) {
            emitted.insert(&entry.second.get());
        }
    }
    std::vector<size_t> covered;
    for (size_t i = 0; i < eventState_.Instances().size(); ++i) {
        const auto& instance = eventState_.Instances()[i];
        const CoreTypeDetail core{instance.pipes.first.core, instance.pipes.first.aivCore};
        if (!instance.intraCore || instance.state != InstanceState::WaitInserted || instance.setIdx >= boundary ||
            instance.waitIdx >= boundary ||
            std::find(barrier.cores.begin(), barrier.cores.end(), core) == barrier.cores.end()) {
            continue;
        }
        if (emitted.count(instance.setOp) == 0 || emitted.count(instance.waitOp) == 0) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Cross-core drain before op %zu is missing the emitted set or wait for %zu -> %zu.",
                              boundary, instance.setIdx, instance.waitIdx);
            return FAILED;
        }
        covered.push_back(i);
    }
    // A cancelled cross-core input may have carried intra-core capacity proof.
    // The existing full drain covers completed waits on these cores instead;
    // pending flags and consumed instances on other cores remain occupied.
    for (auto index : covered) {
        const auto& instance = eventState_.Instances()[index];
        if (eventState_.RetireConsumed(instance.setIdx, instance.waitIdx) != SUCCESS) {
            return FAILED;
        }
        PushEventIdIfAbsent(GetFreeEventIdQueue(instance.pipes), instance.id.resource.eventId);
    }
    reclaimBarriers_.push_back(std::move(barrier));
    return SUCCESS;
}

Status PipeSync::AddLocalCrossCoreBarrier(Function& function, size_t boundary, size_t pool, int eventId,
                                          uint64_t& insertIdx, std::vector<IndexOp>& syncedOpLog)
{
    const auto vector = pool == 0 ? AIVCore::AIV0 : AIVCore::AIV1;
    const PipeCoreRealEx cubeSet(PIPE_FIX, CoreType::AIC);
    const PipeCoreRealEx vectorSet(PIPE_MTE3, CoreType::AIV, vector);
    const PipeCoreRealEx cubeWait(PIPE_S, CoreType::AIC);
    const PipeCoreRealEx vectorWait(PIPE_S, CoreType::AIV, vector);
    ReclaimBarrier barrier;
    barrier.cores = {{CoreType::AIC, AIVCore::UNSPECIFIED}, {CoreType::AIV, vector}};
    auto emit = [&](Opcode opcode, PipeCoreRealEx set, PipeCoreRealEx wait) {
        auto& op = irBuilder_.CreateTensorOpStmt(function, opcode, {}, {});
        if (opcode == Opcode::OP_BAR_ALL) {
            // GenBarrier emits PIPE_ALL independently of coreType_. SetAIVCore below
            // selects the executing core; retain the actual core in this metadata.
            op.syncQueue_ = {PIPE_ALL, PIPE_ALL, set.core, set.core, -1, set.aivCore, set.aivCore};
        } else {
            GenSyncOp(set, wait, eventId, opcode == Opcode::OP_CV_SYNC_SRC, op);
        }
        op.SetAIVCore(opcode == Opcode::OP_CV_SYNC_DST ? wait.aivCore : set.aivCore);
        syncedOpLog.emplace_back(++insertIdx, std::ref(op));
        forceSyncOps_.insert(&op);
        barrier.ops.push_back(&op);
    };
    emit(Opcode::OP_BAR_ALL, cubeSet, cubeWait);
    emit(Opcode::OP_BAR_ALL, vectorSet, vectorWait);
    // Reverse the old sender first; the return hop uses H in its old direction.
    const bool oldCubeSender = *crossCoreEventSenders_[eventId] == CoreType::AIC;
    const auto firstSet = oldCubeSender ? vectorSet : cubeSet;
    const auto firstWait = oldCubeSender ? cubeWait : vectorWait;
    const auto returnSet = oldCubeSender ? cubeSet : vectorSet;
    const auto returnWait = oldCubeSender ? vectorWait : cubeWait;
    emit(Opcode::OP_CV_SYNC_SRC, firstSet, firstWait);
    emit(Opcode::OP_CV_SYNC_DST, firstSet, firstWait);
    emit(Opcode::OP_CV_SYNC_SRC, returnSet, returnWait);
    emit(Opcode::OP_CV_SYNC_DST, returnSet, returnWait);
    return RecordCrossCoreDrain(boundary, syncedOpLog, std::move(barrier));
}

void PipeSync::ErasePendingCrossCoreSets(const std::unordered_set<Operation*>& removedSets)
{
    for (const auto& [edge, setOp] : NoWaitCVPairs_) {
        if (removedSets.count(setOp) == 0) {
            continue;
        }
        RemoveOpDep(depOps_[edge.first], depOps_[edge.second]);
        setWaitPairMap_.erase(edge);
        crossCoreSnapshots_.erase(edge);
        setOpMap.erase(setOp);
        setOp->SetAsDeleted();
    }
    NoWaitCVPairs_.erase(std::remove_if(NoWaitCVPairs_.begin(), NoWaitCVPairs_.end(),
                                        [&](const auto& pending) { return removedSets.count(pending.second) != 0; }),
                         NoWaitCVPairs_.end());
}

Status PipeSync::RemovePendingCrossCoreSyncs(size_t pool, std::vector<IndexOp>& syncedOpLog)
{
    const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
    auto inPool = [base](int id) { return id >= base && id < base + CROSS_CORE_EVENT_NUM; };
    auto stagedState = eventState_;
    std::unordered_set<Operation*> removedSets;
    for (const auto& [edge, setOp] : NoWaitCVPairs_) {
        if (!inPool(setOp->syncQueue_.eventId_)) {
            continue;
        }
        if (stagedState.CancelPending(edge.first, edge.second) != SUCCESS) {
            return FAILED;
        }
        removedSets.insert(setOp);
    }
    eventState_ = std::move(stagedState);
    ErasePendingCrossCoreSets(removedSets);
    syncedOpLog.erase(std::remove_if(syncedOpLog.begin(), syncedOpLog.end(),
                                     [&](const IndexOp& entry) { return removedSets.count(&entry.second.get()) != 0; }),
                      syncedOpLog.end());
    return SUCCESS;
}

Status PipeSync::ResetCrossCorePool(size_t pool, int handshakeId, std::vector<IndexOp>& syncedOpLog)
{
    if (RemovePendingCrossCoreSyncs(pool, syncedOpLog) != SUCCESS) {
        return FAILED;
    }
    const int base = static_cast<int>(pool) * CROSS_CORE_EVENT_NUM;
    const auto vector = pool == 0 ? AIVCore::AIV0 : AIVCore::AIV1;
    const auto finalReceiver = *crossCoreEventSenders_[handshakeId] == CoreType::AIC ? CoreType::AIV : CoreType::AIC;
    auto clearPool = [base](CrossCoreClock& clock) {
        std::fill(clock.begin() + base, clock.begin() + base + CROSS_CORE_EVENT_NUM, 0);
    };
    for (auto& [pipe, clock] : crossCoreConsumed_) {
        (void)pipe;
        clearPool(clock);
    }
    for (auto& [edge, snapshot] : crossCoreSnapshots_) {
        (void)edge;
        clearPool(snapshot.consumed);
    }
    crossCoreGenerations_[handshakeId] += NUM2;
    const PipeCoreRealEx receivingPipe(PIPE_S, finalReceiver,
                                       finalReceiver == CoreType::AIV ? vector : AIVCore::UNSPECIFIED);
    crossCoreConsumed_[receivingPipe][handshakeId] = crossCoreGenerations_[handshakeId];
    for (const auto& pipe : cvPipeCoreEx) {
        if (pipe.core != CoreType::AIC && pipe.aivCore != vector) {
            continue;
        }
        auto& queue = crossCoreFreeEventId_[pipe][pipe.core == CoreType::AIC ? pool : 0];
        queue.clear();
        for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
            if (id != handshakeId || (pipe.core == finalReceiver && pipe.pipe == PIPE_S)) {
                queue.push_back(id);
            }
        }
    }
    for (int id = base; id < base + CROSS_CORE_EVENT_NUM; ++id) {
        if (id != handshakeId) {
            crossCoreEventSenders_[id].reset();
        }
    }
    return SUCCESS;
}

bool PipeSync::HasFreeEventId(const PipePairEx& pp)
{
    // CanAllocate looks up the exact active generation, so stale proofs cannot match.
    return !GetAvailableIntraEventIds(pp, eventState_.KnownForAllocationCheck(pp.first), 1).empty();
}

bool PipeSync::BufOverlap(const TileRange& range1, const TileRange& range2) const
{
    return range1.end > range2.start && range2.end > range1.start;
}

bool PipeSync::CheckWawDependency(const Operation& opSet, const Operation& opWait)
{
    for (size_t setIdx = 0; setIdx < opSet.GetOOperands().size(); setIdx++) {
        for (size_t waitIdx = 0; waitIdx < opWait.GetOOperands().size(); waitIdx++) {
            if (opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal() !=
                opWait.GetOOperands()[waitIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetOOperands()[setIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetOOperands()[setIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[waitIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] :
                                                               opSet.GetOOperands()[setIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] :
                                                               opWait.GetOOperands()[waitIdx]->memoryrange;
            if (BufOverlap(range1, range2)) {
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckRawDependency(const Operation& opSet, const Operation& opWait)
{
    for (size_t outIdx = 0; outIdx < opSet.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opWait.GetIOperands().size(); inIdx++) {
            if (opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal() !=
                opSet.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opWait.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opWait.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opSet.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] :
                                                               opWait.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] :
                                                               opSet.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, range2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::CheckWarDependency(const Operation& opSet, const Operation& opWait)
{
    for (size_t outIdx = 0; outIdx < opWait.GetOOperands().size(); outIdx++) {
        for (size_t inIdx = 0; inIdx < opSet.GetIOperands().size(); inIdx++) {
            if (opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal() !=
                opWait.GetOOperands()[outIdx]->GetMemoryTypeOriginal()) {
                continue;
            }
            auto memType = opSet.GetIOperands()[inIdx]->GetMemoryTypeOriginal();
            int magic1 = opSet.GetIOperands()[inIdx]->GetMagic();
            int magic2 = opWait.GetOOperands()[outIdx]->GetMagic();
            TileRange range1 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic1] :
                                                               opSet.GetIOperands()[inIdx]->memoryrange;
            TileRange range2 = memType == MemoryType::MEM_UB ? ubTensorRangeMap[magic2] :
                                                               opWait.GetOOperands()[outIdx]->memoryrange;
            auto overlap = BufOverlap(range1, range2);
            auto ddrTensorSame = memType == MemoryType::MEM_DEVICE_DDR && range1.memId == range2.memId;
            if (overlap || ddrTensorSame) {
                return true;
            }
        }
    }
    return false;
}

bool PipeSync::HasDataDependency(const Operation& opSet, const Operation& opWait)
{
    for (const auto* producer : opWait.ProducerOpsByToken()) {
        if (producer == &opSet) {
            return true;
        }
    }

    std::string opSetStr = opSet.GetOpcodeStr();
    std::string opWaitStr = opWait.GetOpcodeStr();

    // check WAW
    bool checkWaw = true;
    auto setCfg = OpcodeManager::Inst().GetTileOpCfg(opSet.GetOpcode());
    auto waitCfg = OpcodeManager::Inst().GetTileOpCfg(opWait.GetOpcode());
    AdjustOpCfg(setCfg, opSet);
    AdjustOpCfg(waitCfg, opWait);
    if (waitCfg.pipeIdStart_ == setCfg.pipeIdStart_ &&
        (opSetStr.find("CUBE_A_MUL") == std::string::npos || opWaitStr.find("CUBE_A_MUL") == std::string::npos)) {
        checkWaw = false;
    }
    if (checkWaw) {
        if (CheckWawDependency(opSet, opWait)) {
            return true;
        }
    }

    // check RAW
    if (CheckRawDependency(opSet, opWait)) {
        return true;
    }

    // check WAR
    if (CheckWarDependency(opSet, opWait)) {
        return true;
    }

    return false;
}

void PipeSync::UpdateDep(DepOp& currOp, DepOp& prevOp)
{
    PipeCoreRealEx currPipe(currOp.selfPipeCore.pipeStart, currOp.selfPipeCore.core, currOp.selfPipeCore.aivCore);
    PipeCoreRealEx prevPipe(prevOp.selfPipeCore.pipeEnd, prevOp.selfPipeCore.core, prevOp.selfPipeCore.aivCore);
    auto& currPipeDep = latestPipeDep_[currPipe];
    currPipeDep.waitIdx = currOp.idx;

    auto currSetPipeIter = currPipeDep.setPipes.find(prevPipe);
    if (currSetPipeIter == currPipeDep.setPipes.end() || currSetPipeIter->second < prevOp.idx) {
        // no indirect dependency exist, save current dependency
        currOp.waitPipe.emplace_back(prevOp.idx);
        prevOp.setPipe.emplace_back(currOp.idx);
        currPipeDep.setPipes[prevPipe] = prevOp.idx;
        auto prevPipeDepIter = latestPipeDep_.find(prevPipe);
        auto prevWaitPipeIdx = prevPipeDepIter->second.waitIdx;
        if (prevPipeDepIter != latestPipeDep_.end() && prevWaitPipeIdx <= prevOp.idx) {
            // merge dependency
            std::map<PipeCoreRealEx, size_t, PipeCoreRealExCompare> prevSetPipes = prevPipeDepIter->second.setPipes;
            for (auto [prevSetPipeType, prevSetPipeIdx] : prevSetPipes) {
                auto res = currPipeDep.setPipes.emplace(prevSetPipeType, prevSetPipeIdx);
                // isExist == isPrevSetPipeTypeExist
                bool isExist = !res.second;
                size_t& existIdx = res.first->second;
                if (isExist && existIdx < prevSetPipeIdx) {
                    // overwrite dependency
                    existIdx = prevSetPipeIdx;
                }
            }
        }
    }
}

bool PipeSync::IgnorableIntraPipeDep(size_t prev, size_t curr, const std::vector<Operation*>& opLogPtr)
{
    // true表示依赖关系可忽略，false表示依赖关系不可忽略
    // VIEW or ASSEMBLE data dependency can be ignored
    if (opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_VIEW_TYPE || opLogPtr[curr]->GetOpcode() == Opcode::OP_VIEW_TYPE ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_ASSEMBLE || opLogPtr[curr]->GetOpcode() == Opcode::OP_ASSEMBLE ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_NOP || opLogPtr[curr]->GetOpcode() == Opcode::OP_NOP ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_HUB || opLogPtr[curr]->GetOpcode() == Opcode::OP_HUB ||
        opLogPtr[prev]->GetOpcode() == Opcode::OP_RESHAPE || opLogPtr[curr]->GetOpcode() == Opcode::OP_RESHAPE) {
        return true;
    }
    return false;
}

// find depend op in opLog for 0 to idx
void PipeSync::FindDep(DepOp& op, const std::vector<Operation*>& opLogPtr, size_t idx,
                       DataDependencySearcher& dataDependencySearcher)
{
    const auto currOp = opLogPtr[idx];
    APASS_LOG_DEBUG_F(Elements::Operation, "=== OP: %d %zu %s ===", currOp->GetOpMagic(), idx,
                      currOp->GetOpcodeStr().c_str());
    // check dependency from latest op to oldest
    auto dataDependencySet = dataDependencySearcher.Find(currOp);
    for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
        size_t k = *it;
        const Operation* prevAOp = opLogPtr[k];
        DepOp& prevOp = depOps_[k];

        if (HasDataDependency(*prevAOp, *currOp)) {
            bool ignorable = false;
            if (IgnorableIntraPipeDep(k, idx, opLogPtr)) {
                ignorable = true;
            }
            if (!ignorable) {
                UpdateDep(op, prevOp);
            }
        }
    }
    dataDependencySearcher.Insert(currOp, idx);
}

void PipeSync::InitCVEventIdQ()
{
    for (const auto& pipecore : cvPipeCoreEx) {
        int base = (pipecore.aivCore == AIVCore::AIV0) ? 0 : CROSS_CORE_EVENT_NUM;
        if (pipecore.core == CoreType::AIC) {
            for (int i = 0; i < CROSS_CORE_EVENT_NUM; i++) {
                crossCoreFreeEventId_[pipecore][0].push_back(i);
            }
            for (int i = CROSS_CORE_EVENT_NUM; i < CROSS_CORE_EVENT_NUM * NUM2; i++) {
                crossCoreFreeEventId_[pipecore][1].push_back(i);
            }
        } else {
            for (int i = base; i < base + CROSS_CORE_EVENT_NUM; i++) {
                crossCoreFreeEventId_[pipecore][0].push_back(i);
            }
        }
    }
}

void PipeSync::PushEventIdIfAbsent(std::deque<int>& queue, int eventId)
{
    if (std::find(queue.begin(), queue.end(), eventId) == queue.end()) {
        queue.push_back(eventId);
    }
}

// 该函数仅用于核内同步
std::deque<int>& PipeSync::GetFreeEventIdQueue(const PipePairEx& pp)
{
    auto [it, inserted] = freeEventId_.try_emplace(pp);
    if (inserted) {
        for (int i = INTRA_CORE_START_EVENT_ID; i <= GetMaxEventId(pp); i++) {
            it->second.push_back(i);
        }
    }
    return it->second;
}

void PipeSync::AddPhaseOp1(Function& function, const std::vector<Operation*>& srcLog, std::vector<Operation*>& dstLog,
                           size_t& i, size_t& prerun)
{
    constexpr size_t prerunNum = 2;
    for (; i < srcLog.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(srcLog[i]->GetOpcode());
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_IN) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
        }
        if (srcLog[i]->GetOpcode() == Opcode::OP_COPY_OUT) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
        }
        if ((opcfg.pipeIdStart_ != PIPE_S && opcfg.pipeIdStart_ != PIPE_MTE2 &&
             srcLog[i]->GetOpcode() != Opcode::OP_RESHAPE && srcLog[i]->GetOpcode() != Opcode::OP_VEC_DUP) ||
            prerun == prerunNum) {
            break;
        }
        if (opcfg.pipeIdStart_ == PIPE_MTE2) {
            if (prerun == 0) {
                std::vector<std::shared_ptr<LogicalTensor>> input;
                std::vector<std::shared_ptr<LogicalTensor>> output;
                Operation& phaseOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_PHASE1, input,
                                                                   output);
                Operation* phaseOpPtr = &phaseOp;
                dstLog.emplace_back(phaseOpPtr);
            }
            prerun++;
        }
        dstLog.emplace_back(srcLog[i]);
    }
}

void PipeSync::AddPhaseOp2(Function& function, std::vector<Operation*>& dstLog, size_t& prerun)
{
    if (prerun > 0) {
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& phaseOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_PHASE2, input, output);
        Operation* phaseOpPtr = &phaseOp;
        dstLog.emplace_back(phaseOpPtr);
    }
}

void PipeSync::PhaseKernelProcess(Function& function, const std::vector<Operation*>& srcLog,
                                  std::vector<Operation*>& dstLog)
{
    size_t prerun = 0;
    size_t i = 0;
    AddPhaseOp1(function, srcLog, dstLog, i, prerun);
    AddPhaseOp2(function, dstLog, prerun);
    for (; i < srcLog.size(); i++) {
        dstLog.emplace_back(srcLog[i]);
    }
}

void InsertSync::InsertCvSyncOps(Function* subGraphFunc, std::vector<Operation*>& newOpList)
{
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    auto createBarAll = [&](AIVCore aivCore) {
        Operation& op = irBuilder_.CreateTensorOpStmt(*subGraphFunc, Opcode::OP_BAR_ALL, input, output);
        op.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1, aivCore, aivCore};
        op.SetAIVCore(aivCore);
        newOpList.push_back(&op);
    };
    // 核间全同步固定使用EVENT_ID7
    auto createForceSync = [&](Opcode opcode, PipeType pipe, CoreType core, AIVCore aivCore) {
        Operation& op = irBuilder_.CreateTensorOpStmt(*subGraphFunc, opcode, input, output);
        op.syncQueue_ = {pipe, pipe, core, core, PipeSync::EVENT_ID7, aivCore, aivCore};
        op.SetAIVCore(aivCore);
        newOpList.push_back(&op);
    };

    createBarAll(AIVCore::UNSPECIFIED);
    createForceSync(Opcode::OP_FFTS_CROSS_CORE_SYNC, PipeType::PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    createForceSync(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED);
    createBarAll(AIVCore::UNSPECIFIED);
    createBarAll(AIVCore::AIV0);
    createForceSync(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV0);
    createForceSync(Opcode::OP_FFTS_CROSS_CORE_SYNC, PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    createBarAll(AIVCore::AIV0);
    createBarAll(AIVCore::AIV1);
    createForceSync(Opcode::OP_WAIT_FLAG_DEV, PipeType::PIPE_S, CoreType::AIV, AIVCore::AIV1);
    createForceSync(Opcode::OP_FFTS_CROSS_CORE_SYNC, PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV1);
    createBarAll(AIVCore::AIV1);
}

void InsertSync::InsertCvPipeAll(Function* subGraphFunc)
{
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    std::vector<Operation*> newOpList;
    PipeSync ps;
    for (size_t i = 0; i < oriOpList.size(); i++) {
        newOpList.push_back(oriOpList[i]);
        if (i == oriOpList.size() - 1) {
            continue;
        }
        auto currOpCfg = OpcodeManager::Inst().GetTileOpCfg(oriOpList[i]->GetOpcode());
        ps.AdjustOpCfg(currOpCfg, *oriOpList[i]);
        auto nextOpCfg = OpcodeManager::Inst().GetTileOpCfg(oriOpList[i + 1]->GetOpcode());
        ps.AdjustOpCfg(nextOpCfg, *oriOpList[i + 1]);
        if (currOpCfg.coreType_ == nextOpCfg.coreType_) {
            std::vector<std::shared_ptr<LogicalTensor>> input;
            std::vector<std::shared_ptr<LogicalTensor>> output;
            Operation& syncOp = irBuilder_.CreateTensorOpStmt(*subGraphFunc, npu::tile_fwk::Opcode::OP_BAR_ALL, input,
                                                              output);
            AIVCore nextAIVCore = oriOpList[i + 1]->GetAIVCore();
            syncOp.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1,
                                 nextAIVCore,        nextAIVCore};
            syncOp.SetAIVCore(nextAIVCore);
            newOpList.push_back(&syncOp);
            continue;
        }
        // 插入固定序列1C2V全同步
        InsertCvSyncOps(subGraphFunc, newOpList);
    }
    subGraphFunc->ScheduleBy(newOpList, true);
    subGraphFunc->oriOpList = oriOpList;
}

Status InsertSync::CheckNewOpListSeq(const std::vector<Operation*>& oriOpList, const std::vector<Operation*>& opListNew)
{
    if (oriOpList.size() <= 1) {
        return SUCCESS;
    }
    size_t i = 0;
    size_t j = 0;
    while (i < oriOpList.size() && j < opListNew.size()) {
        if (oriOpList[i] == opListNew[j]) {
            ++i;
            ++j;
        } else {
            ++j;
        }
    }
    if (i != oriOpList.size()) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "NewOpList sequence is not equal to OriOpList sequence, CheckNewOpListSeq failed.");
        return FAILED;
    }
    return SUCCESS;
}

// --- atomic scope 后处理：静态辅助方法 ---

bool InsertSync::IsSyncOpcode(Opcode opcode)
{
    return opcode == Opcode::OP_SYNC_SRC || opcode == Opcode::OP_SYNC_DST || opcode == Opcode::OP_CV_SYNC_SRC ||
           opcode == Opcode::OP_CV_SYNC_DST || opcode == Opcode::OP_FFTS_CROSS_CORE_SYNC ||
           opcode == Opcode::OP_WAIT_FLAG_DEV || opcode == Opcode::OP_BAR_V || opcode == Opcode::OP_BAR_M ||
           opcode == Opcode::OP_BAR_ALL;
}

bool InsertSync::IsImmovableSync(Opcode opcode)
{
    return opcode == Opcode::OP_FFTS_CROSS_CORE_SYNC || opcode == Opcode::OP_WAIT_FLAG_DEV ||
           opcode == Opcode::OP_BAR_ALL || opcode == Opcode::OP_BAR_V || opcode == Opcode::OP_BAR_M;
}

InsertSync::SyncSignature InsertSync::GetSyncSignature(int eventId, PipeType setPipe, PipeType waitPipe)
{
    // 位布局：低 8 bit 为 waitPipe，次 8 bit 为 setPipe，其余高位为 eventId。
    constexpr int PIPE_FIELD_BITS = 8;
    return (static_cast<uint64_t>(eventId) << (2 * PIPE_FIELD_BITS)) |
           (static_cast<uint64_t>(setPipe) << PIPE_FIELD_BITS) | static_cast<uint64_t>(waitPipe);
}

// Pass 1: 确定 cluster 范围（只看非 sync op 的 scopeId），O(N)
Status InsertSync::BuildClusterRanges(const std::vector<Operation*>& opList, std::map<int, ClusterInfo>& clusters)
{
    for (size_t i = 0; i < opList.size(); ++i) {
        if (IsSyncOpcode(opList[i]->GetOpcode())) {
            continue;
        }
        int scopeId = opList[i]->GetAtomicScopeId();
        if (scopeId < VF_CLUSTER_ID_START) {
            continue;
        }
        auto& ci = clusters[scopeId];
        ci.firstIdx = std::min(ci.firstIdx, i);
        ci.lastIdx = std::max(ci.lastIdx, i);
        if (ci.firstIdx == i) {
            ci.aivCore = opList[i]->GetAIVCore();
        }
    }
    return SUCCESS;
}

// Pass 2: 扫描单个 cluster 范围 — 连续性 + 指令类型 + 收集 sync op + 配对检查
// OOO 发射序列跨核交错：区间内另一核（AIV0/AIV1/AIC）的 op 不属于本 cluster，跳过挪动判定
Status InsertSync::ValidateAndCollectClusterSyncOps(const std::vector<Operation*>& opList, int scopeId, ClusterInfo& ci,
                                                    const std::unordered_set<Operation*>& forceSyncOps)
{
    for (size_t i = ci.firstIdx + 1; i < ci.lastIdx; ++i) {
        Operation* op = opList[i];
        if (!IsSyncOpcode(op->GetOpcode())) {
            if (op->GetAIVCore() != ci.aivCore) {
                // OOO 发射序列跨核交错：区间内另一核（AIV0/AIV1/AIC）的 op 不属于本 cluster，跳过
                continue;
            }
            if (op->GetAtomicScopeId() != scopeId) {
                // 窗口内夹带同核外部 op（scope=-1 的未压实搬运，或异 scope 交错）：
                // 挪动 sync 跨越同发射流 op 存在顺序风险，降级为整 cluster 不调整
                ci.skipAdjust = true;
                ci.waitSyncs.clear();
                ci.setSyncs.clear();
                APASS_LOG_WARN_F(Elements::Operation,
                                 "AdjustSyncByAtomicScope: cluster %d not continuous at index %zu, "
                                 "op:%s[%d] scope:%d aivcore:%d, cluster first:%s[%d] last:%s[%d], skip whole cluster.",
                                 scopeId, i, op->GetOpcodeStr().c_str(), op->GetOpMagic(), op->GetAtomicScopeId(),
                                 static_cast<int>(op->GetAIVCore()), opList[ci.firstIdx]->GetOpcodeStr().c_str(),
                                 opList[ci.firstIdx]->GetOpMagic(), opList[ci.lastIdx]->GetOpcodeStr().c_str(),
                                 opList[ci.lastIdx]->GetOpMagic());
                return SUCCESS;
            }
            continue;
        }
        bool isImmovable = IsImmovableSync(op->GetOpcode());
        // CV eventId 耗尽时插入的兜底 force 同步序列（BAR_ALL/FFTS/WAIT_FLAG_DEV）落在 cluster 内：
        // 挪动 sync 跨越 force 序列存在事件重置风险，该 cluster 整体不再进行 op 顺序调整
        if (isImmovable && forceSyncOps.count(op) > 0) {
            ci.skipAdjust = true;
            ci.waitSyncs.clear();
            ci.setSyncs.clear();
            APASS_LOG_WARN_F(Elements::Operation,
                             "AdjustSyncByAtomicScope: force sync %s[%d] inside cluster %d at index %zu, "
                             "cluster first:%s[%d] last:%s[%d], skip whole cluster adjust.",
                             op->GetOpcodeStr().c_str(), op->GetOpMagic(), scopeId, i,
                             opList[ci.firstIdx]->GetOpcodeStr().c_str(), opList[ci.firstIdx]->GetOpMagic(),
                             opList[ci.lastIdx]->GetOpcodeStr().c_str(), opList[ci.lastIdx]->GetOpMagic());
            return SUCCESS;
        }
        if (op->GetAIVCore() != ci.aivCore) {
            // OOO 发射序列跨核交错：区间内另一核（AIV0/AIV1/AIC）的 op 不属于本 cluster，跳过
            continue;
        }
        if (isImmovable) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "AdjustSyncByAtomicScope failed: immovable sync %s inside cluster %d, "
                              "sync aivcore:%d, cluster aivcore:%d.",
                              op->GetOpcodeStr().c_str(), scopeId, static_cast<int>(op->GetAIVCore()),
                              static_cast<int>(ci.aivCore));
            return FAILED;
        }
        // CV 配对检查（同簇同 eventId 的 set/wait 即拒绝）：AIC<->AIV 双向握手共享 eventId 数值
        // 空间且 wait 生成后即回收复用，同一 cluster 可能收集到数值相同的 CV set 与 CV wait（分属
        // 两个握手、均在本核执行的一端）。此时异核配对端 op 保持原位，若把本核 wait 挪到 cluster
        // 前、set 挪到 cluster 后，会与原位配对端形成顺序颠倒（如 V set C wait C set V wait 挪动后
        // 变为 V wait C wait C set V set）互等死锁，须报错拒绝调整。普通 SYNC_SRC/DST 为同核跨
        // pipe 同步，配对可能同簇出现，签名检查同理保留。
        // 同签名 wait 先于 set 既可能来自不同代次，也可能来自回放合并后的反向文本依赖。
        // 文本顺序不等于跨 pipe 的执行顺序；这里只做保守签名检查，最终安全性由
        // GenNewOpList 中的真实 FIFO/set->wait 图校验确认。
        if (op->GetOpcode() == Opcode::OP_SYNC_DST) {
            if (ci.setSignatures.count(GetSyncSignature(op->syncQueue_.eventId_, op->syncQueue_.pipeId_,
                                                        op->syncQueue_.trigPipeId_)) > 0) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "AdjustSyncByAtomicScope failed: paired set/wait (eventId=%d, setPipe=%s, "
                                  "waitPipe=%s) in cluster %d.",
                                  op->syncQueue_.eventId_, GetPipeTypeDict().Find(op->syncQueue_.pipeId_).c_str(),
                                  GetPipeTypeDict().Find(op->syncQueue_.trigPipeId_).c_str(), scopeId);
                return FAILED;
            }
            ci.waitSyncs.push_back(op);
        } else if (op->GetOpcode() == Opcode::OP_SYNC_SRC) {
            ci.setSyncs.push_back(op);
            ci.setSignatures.insert(
                GetSyncSignature(op->syncQueue_.eventId_, op->syncQueue_.pipeId_, op->syncQueue_.trigPipeId_));
        } else if (op->GetOpcode() == Opcode::OP_CV_SYNC_DST) {
            if (ci.cvSetEventIds.count(op->syncQueue_.eventId_) > 0) {
                APASS_LOG_ERROR_F(Elements::Operation,
                                  "AdjustSyncByAtomicScope failed: paired cv set/wait (eventId=%d) in cluster %d.",
                                  op->syncQueue_.eventId_, scopeId);
                return FAILED;
            }
            ci.waitSyncs.push_back(op);
        } else if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC) {
            ci.setSyncs.push_back(op);
            ci.cvSetEventIds.insert(op->syncQueue_.eventId_);
        }
    }
    return SUCCESS;
}

// Pass 3: 重排 — wait-like 移到 cluster 前，set-like 移到 cluster 后
// 区间可能跨核交错重叠（异核 cluster 嵌套），被收集的 sync 从原位移除、在所属 cluster 边界重发，
// 未被收集的 op（另一核 op 与 sync）保持原位
void InsertSync::ReorderOpListForClusters(std::vector<Operation*>& opList, const std::map<int, ClusterInfo>& clusters)
{
    std::unordered_set<Operation*> collectedSyncs;
    std::unordered_map<size_t, std::vector<const ClusterInfo*>> waitAt;
    std::unordered_map<size_t, std::vector<const ClusterInfo*>> setAt;
    for (const auto& cluster : clusters) {
        const ClusterInfo& ci = cluster.second;
        if (ci.skipAdjust) {
            continue;
        }
        for (auto* w : ci.waitSyncs) {
            collectedSyncs.insert(w);
        }
        for (auto* s : ci.setSyncs) {
            collectedSyncs.insert(s);
        }
        waitAt[ci.firstIdx].emplace_back(&ci);
        setAt[ci.lastIdx].emplace_back(&ci);
    }

    std::vector<Operation*> newOpList;
    newOpList.reserve(opList.size());
    for (size_t i = 0; i < opList.size(); ++i) {
        auto waitIt = waitAt.find(i);
        if (waitIt != waitAt.end()) {
            for (auto* ci : waitIt->second) {
                for (auto* w : ci->waitSyncs) {
                    newOpList.push_back(w);
                }
            }
        }
        if (collectedSyncs.count(opList[i]) == 0) {
            newOpList.push_back(opList[i]);
        }
        auto setIt = setAt.find(i);
        if (setIt != setAt.end()) {
            for (auto* ci : setIt->second) {
                for (auto* s : ci->setSyncs) {
                    newOpList.push_back(s);
                }
            }
        }
    }
    opList = std::move(newOpList);
}

Status InsertSync::AdjustSyncByAtomicScope(std::vector<Operation*>& opList,
                                           const std::unordered_set<Operation*>& forceSyncOps)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        return SUCCESS;
    }

    std::map<int, ClusterInfo> clusters;
    if (BuildClusterRanges(opList, clusters) != SUCCESS) {
        return FAILED;
    }
    if (clusters.empty()) {
        return SUCCESS;
    }

    // skipAdjust 仅由 ValidateAndCollectClusterSyncOps（Pass 2）置位，此处无需预判；
    // 被 Pass 2 置位的 cluster 在 ReorderOpListForClusters 中按 skipAdjust 跳过
    for (auto& [scopeId, ci] : clusters) {
        auto status = ValidateAndCollectClusterSyncOps(opList, scopeId, ci, forceSyncOps);
        if (status != SUCCESS) {
            return status;
        }
    }

    ReorderOpListForClusters(opList, clusters);
    return SUCCESS;
}

// cv_hard_sync debug mode: keep the intra-core syncs derived by the normal path
// (SYNC_SRC/SYNC_DST, BAR_V/BAR_M), but replace the cross-core part with the 1C2V FFTS hard sync
// (the fixed sequence from InsertCvSyncOps: BAR.ALL + ffts_cross_core_sync(CV_CORES_SYNC) +
// wait_flag_dev covering AIC/AIV0/AIV1) at every CV boundary in the schedule instead of relying
// on the dependency-analysis-hit CV_SYNC_SRC/CV_SYNC_DST intra_block flag handshakes.
// Difference from the enableDebug_ full-sync mode: no OP_BAR_ALL between adjacent same-core ops.
Status InsertSync::ApplyCvHardSync(Function* subGraphFunc, std::vector<Operation*>& opListNew)
{
    size_t boundaryNum = 0;
    PipeSync ps;
    bool hasLastTileCore = false;
    CoreType lastTileCore = CoreType::AIC;
    std::vector<Operation*> hardSyncList;
    hardSyncList.reserve(opListNew.size());
    for (Operation* op : opListNew) {
        if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC || op->GetOpcode() == Opcode::OP_CV_SYNC_DST) {
            // cross-core flag handshake is superseded by the CV-boundary hard sync
            op->SetAsDeleted();
            continue;
        }
        if (IsSyncOpcode(op->GetOpcode())) {
            // intra-core syncs stay as derived, transparent for boundary detection
            hardSyncList.push_back(op);
            continue;
        }
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(op->GetOpcode());
        if (ps.AdjustOpCfg(opcfg, *op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ApplyCvHardSync failed at AdjustOpCfg.");
            return FAILED;
        }
        if (hasLastTileCore && opcfg.coreType_ != lastTileCore) {
            // the executing core of tile ops switches: insert the all-core rendezvous so data
            // written before the switch is visible to the core after the switch
            InsertCvSyncOps(subGraphFunc, hardSyncList);
            boundaryNum++;
        }
        lastTileCore = opcfg.coreType_;
        hasLastTileCore = true;
        hardSyncList.push_back(op);
    }
    // 无CV边界也要完成已置删除同步算子的换出与擦除，保持与有边界路径一致
    opListNew.swap(hardSyncList);
    subGraphFunc->EraseOperations(true, false);
    if (boundaryNum == 0) {
        return SUCCESS;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "CvHardSync mode: inserted 1C2V hard sync at %zu CV boundary(ies) in function %s.", boundaryNum,
                     subGraphFunc->GetMagicName().c_str());
    return SUCCESS;
}

Status InsertSync::GenNewOpList(Function* subGraphFunc, std::vector<Operation*>& opListNew)
{
    PipeSync ps;
    ps.enableLocalCrossCoreBarrier_ = !cvHardSync_;
    std::vector<Operation*> syncedOpLogPtr;
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    if (ps.InsertSync(*subGraphFunc, syncedOpLogPtr) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenNewOpList failed at function InsertSync.");
        return FAILED;
    }
    if (cvHardSync_) {
        // skip the phase split (PHASE1/PHASE2) in hard-sync mode: cross-kernel phase overlap
        // races with GM view dependencies between mix and pure kernels, matching the debug mode
        opListNew = syncedOpLogPtr;
    } else {
        ps.PhaseKernelProcess(*subGraphFunc, syncedOpLogPtr, opListNew);
    }
    subGraphFunc->EraseOperations(true, false);
    if (AdjustSyncByAtomicScope(opListNew, ps.GetForceSyncOps()) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenNewOpList failed at function AdjustSyncByAtomicScope.");
        return FAILED;
    }
    if (CheckNewOpListSeq(oriOpList, opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "GenNewOpList failed at function CheckNewOpListSeq.");
        return FAILED;
    }
    if (ps.ValidateEventReuse(opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Event reuse proof invalidated by sync postprocessing.");
        return FAILED;
    }
    subGraphFunc->setOpMap = ps.setOpMap;
    subGraphFunc->waitOpMap = ps.waitOpMap;
    subGraphFunc->oriOpList = ps.GetOriOpList();
    return SUCCESS;
}

Status InsertSync::InsertSyncMainLoop(Function* subGraphFunc)
{
    if (config::GetPassGlobalConfig<bool>(KEY_ENABLE_PIPE_ALL, false)) {
        InsertPipeAll(subGraphFunc);
        return SUCCESS;
    }
    if (enableDebug_) {
        InsertCvPipeAll(subGraphFunc);
        return SUCCESS;
    }
    std::vector<Operation*> opListNew;
    if (GenNewOpList(subGraphFunc, opListNew) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "InsertSyncMainLoop failed at GenNewOpList.");
        return FAILED;
    }
    if (cvHardSync_) {
        if (ApplyCvHardSync(subGraphFunc, opListNew) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InsertSyncMainLoop failed at ApplyCvHardSync.");
            return FAILED;
        }
    }
    subGraphFunc->ScheduleBy(opListNew, true);
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "==========================================================================================");
    for (const auto& op : subGraphFunc->Operations(false).DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("SYNC_SRC") != std::string::npos ||
            op->GetOpcodeStr().find("SYNC_DST") != std::string::npos || op->GetOpcode() == Opcode::OP_BAR_V ||
            op->GetOpcode() == Opcode::OP_BAR_M) {
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Output operation %d: %s, setpipe type: %s, setcore type: %s, waitpipe type: %s, waitcore type: %s, "
                "eventid: %d, setAIVcore type: %d, waitAIVcore type: %d",
                op->GetOpMagic(), op->GetOpcodeStr().c_str(), GetPipeTypeDict().Find(op->syncQueue_.pipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.coreType_).c_str(),
                GetPipeTypeDict().Find(op->syncQueue_.trigPipeId_).c_str(),
                GetCoreTypeDict().Find(op->syncQueue_.trigCoreType_).c_str(), op->syncQueue_.eventId_,
                static_cast<int>(op->syncQueue_.setAivCore_), static_cast<int>(op->syncQueue_.waitAivCore_));
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Output operation %d: %s, AIV core type: %d", op->GetOpMagic(),
                          op->GetOpcodeStr().c_str(), static_cast<int>(op->GetAIVCore()));
    }
    return SUCCESS;
}

void InsertSync::InsertPipeAll(Function* subGraphFunc)
{
    std::vector<Operation*> oriOpList(subGraphFunc->Operations(false).DuplicatedOpList());
    std::vector<Operation*> newOpList;
    for (size_t i = 0; i < oriOpList.size(); i++) {
        newOpList.push_back(oriOpList[i]);
        if (oriOpList[i]->GetOpcode() == Opcode::OP_RESHAPE || oriOpList[i]->GetOpcode() == Opcode::OP_VIEW ||
            oriOpList[i]->GetOpcode() == Opcode::OP_VIEW_TYPE || oriOpList[i]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            continue;
        }
        if (i != oriOpList.size() - 1) {
            std::vector<std::shared_ptr<LogicalTensor>> input;
            std::vector<std::shared_ptr<LogicalTensor>> output;
            Operation& syncOp = irBuilder_.CreateTensorOpStmt(*subGraphFunc, npu::tile_fwk::Opcode::OP_BAR_ALL, input,
                                                              output);
            AIVCore nextAIVCore = oriOpList[i + 1]->GetAIVCore();
            syncOp.syncQueue_ = {PipeType::PIPE_ALL, PipeType::PIPE_ALL, CoreType::AIV, CoreType::AIV, -1,
                                 nextAIVCore,        nextAIVCore};
            syncOp.SetAIVCore(nextAIVCore);
            newOpList.push_back(&syncOp);
        }
    }
    subGraphFunc->ScheduleBy(newOpList, true);
    subGraphFunc->oriOpList = oriOpList;
}

// regist pass
Status InsertSync::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Start InsertSync.");
    if (cvHardSync_) {
        APASS_LOG_INFO_F(Elements::Operation,
                         "CvHardSync debug mode enabled, all cross-core syncs use 1C2V hard sync.");
    }
    const unsigned hardwareConcurrency = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 1);
    uint64_t index = 0;
    std::vector<std::pair<uint64_t, Function*>> subPrograms;
    for (auto& subProgram : function.rootFunc_->programs_) {
        subPrograms.push_back(subProgram);
    }
    std::atomic<size_t> nextIdx(0);
    size_t leafFuncSize = function.rootFunc_->programs_.size();
    // The max thread number by std::thread::hardware_concurrency()

    const unsigned threadNum = std::min(static_cast<unsigned>(leafFuncSize), hardwareConcurrency);
    std::vector<std::thread> workers;
    bool status{true};
    for (unsigned i = 0; i < threadNum; ++i) {
        workers.emplace_back([&subPrograms, &nextIdx, leafFuncSize, &index, this, &status] {
            for (size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed); idx < leafFuncSize;
                 idx = nextIdx.fetch_add(1, std::memory_order_relaxed)) {
                auto program = subPrograms[idx];
                APASS_LOG_DEBUG_F(
                    Elements::Operation,
                    "====================================Program %zu ===========================================",
                    index);
                if (InsertSyncMainLoop(program.second) != SUCCESS) {
                    status = false;
                    break;
                }
                index++;
            }
        });
    }
    // Wait for all threads to finish
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (!status) {
        if (threadNum == 1) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Single Thread scenario.");
            return FAILED;
        }
        APASS_LOG_ERROR_F(
            Elements::Operation,
            "InsertSync RunOnFunction failed at function InsertSyncMainLoop in Multiple Threads scenario.");
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Finish InsertSync.");
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
