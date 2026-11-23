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
 * \file data_dependency.cpp
 * \brief
 */

#include "passes/block_graph_pass/insert_sync/insert_sync.h"
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

// Data dependency discovery

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
        if (currPtr->interval.start <= interval.end) {
            intervalStack.push_back(currPtr->right);
        }
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
            const auto& found = writeDdrMemMap[readDDRmemId];
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
            const auto& found = readDdrMemMap[writeDDRmemId];
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

void PipeSync::RemoveRedundantAivDependencies()
{
    if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        return;
    }
    auto crossesAivCores = [](const DepOp& set, const DepOp& wait) {
        return set.selfPipeCore.core == CoreType::AIV && wait.selfPipeCore.core == CoreType::AIV &&
               set.selfPipeCore.aivCore != wait.selfPipeCore.aivCore;
    };
    auto singlePipe = [](const DepOp& op) { return op.selfPipeCore.pipeStart == op.selfPipeCore.pipeEnd; };
    std::vector<size_t> nextInPipe;
    for (auto& source : depOps_) {
        if (!singlePipe(source)) {
            continue;
        }
        std::vector<size_t> candidates;
        for (auto waitIdx : source.setPipe) {
            if (crossesAivCores(source, depOps_[waitIdx])) {
                candidates.push_back(waitIdx);
            }
        }
        if (candidates.empty()) {
            continue;
        }
        if (nextInPipe.empty()) {
            BuildNextInPipe(nextInPipe);
        }
        // latestPipeDep_ cannot represent an old producer's history after a newer
        // operation has updated that pipe. Prove redundancy from retained edges.
        const auto reached = ReachRetainedDependencies(source.idx, nextInPipe);
        for (auto waitIdx : candidates) {
            if (reached[waitIdx]) {
                RemoveOpDep(source, depOps_[waitIdx]);
            }
        }
    }
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

void PipeSync::BuildNextInPipe(std::vector<size_t>& nextInPipe) const
{
    auto singlePipe = [](const DepOp& op) { return op.selfPipeCore.pipeStart == op.selfPipeCore.pipeEnd; };
    nextInPipe.assign(depOps_.size(), SIZE_MAX);
    std::map<PipeCoreRealEx, size_t, PipeCoreRealExCompare> lastInPipe;
    for (const auto& op : depOps_) {
        if (!singlePipe(op)) {
            continue;
        }
        const auto& core = op.selfPipeCore;
        const PipeCoreRealEx pipe(core.pipeStart, core.core, core.aivCore);
        auto last = lastInPipe.find(pipe);
        if (last != lastInPipe.end()) {
            nextInPipe[last->second] = op.idx;
        }
        lastInPipe[pipe] = op.idx;
    }
}

std::vector<bool> PipeSync::ReachRetainedDependencies(size_t sourceIdx, const std::vector<size_t>& nextInPipe) const
{
    auto singlePipe = [](const DepOp& op) { return op.selfPipeCore.pipeStart == op.selfPipeCore.pipeEnd; };
    auto crossesAivCores = [](const DepOp& set, const DepOp& wait) {
        return set.selfPipeCore.core == CoreType::AIV && wait.selfPipeCore.core == CoreType::AIV &&
               set.selfPipeCore.aivCore != wait.selfPipeCore.aivCore;
    };
    std::vector<bool> reached(depOps_.size(), false);
    std::vector<size_t> pending{sourceIdx};
    reached[sourceIdx] = true;
    while (!pending.empty()) {
        const auto idx = pending.back();
        pending.pop_back();
        auto enqueue = [&](size_t next) {
            // Never justify one unsupported edge with another. A multi-pipe
            // operation also does not prove entry-to-exit hardware ordering.
            if (next <= idx || reached[next] || !singlePipe(depOps_[next]) ||
                crossesAivCores(depOps_[idx], depOps_[next])) {
                return;
            }
            reached[next] = true;
            pending.push_back(next);
        };
        for (auto next : depOps_[idx].setPipe) {
            enqueue(next);
        }
        // Only the same physical core, subcore and pipe share this FIFO.
        if (nextInPipe[idx] != SIZE_MAX) {
            enqueue(nextInPipe[idx]);
        }
    }
    return reached;
}

// Initial dependency graph

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
    RemoveRedundantAivDependencies();
    return SUCCESS;
}

// Dependency edge updates

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

std::optional<PipeSync::MatchingWait> PipeSync::FindWaitOnSamePipe(const DepOp& setOp, const DepOp& waitOp) const
{
    const auto& target = waitOp.selfPipeCore;
    const PipeCoreRealEx targetPipe(target.pipeStart, target.core, target.aivCore);
    for (const auto idx : setOp.setPipe) {
        const auto& existing = depOps_[idx].selfPipeCore;
        if (PipeCoreRealEx(existing.pipeStart, existing.core, existing.aivCore) == targetPipe) {
            return MatchingWait{idx, idx <= waitOp.idx};
        }
    }
    return std::nullopt;
}

Status PipeSync::AddOpDep(DepOp& setOp, DepOp& waitOp, bool isMergeCvSyncBase)
{
    size_t setOpIdx = setOp.idx;
    size_t waitOpIdx = waitOp.idx;

    if (const auto existing = FindWaitOnSamePipe(setOp, waitOp)) {
        if (existing->blocksInsertion) {
            if (existing->idx == waitOpIdx) {
                APASS_LOG_ERROR_F(Elements::Operation, "This dependency should not exist, AddOpDep failed.");
                return FAILED;
            }
            if (isMergeCvSyncBase) {
                return SUCCESS;
            }
            APASS_LOG_ERROR_F(Elements::Operation, "New waitidx should less than old, AddOpDep failed.");
            return FAILED;
        }
        RemoveOpDep(setOp, depOps_[existing->idx]);
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

    auto& nextOp = depOps_[nextOpIdx];
    const auto existing = FindWaitOnSamePipe(nextOp, waitOp);
    if (existing && existing->blocksInsertion) {
        // Check the same conflict as AddOpDep before removing the original edge.
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Cannot AdjustOpDep: producer %zu already has wait %zu before or at %zu.", nextOpIdx,
                          existing->idx, waitOpIdx);
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

    APASS_LOG_DEBUG_F(Elements::Operation, "Event reuse yield: dependency (%zu,%zu) moves to (%zu,%zu).", op.idx,
                      waitOpIdx, nextOpIdx, waitOpIdx);
    RemoveOpDep(op, waitOp);
    if (AddOpDep(nextOp, waitOp) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "AdjustOpDep failed at function AddOpDep.");
        return FAILED;
    }
    return SUCCESS;
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

// Dependency graph for replay

bool PipeSync::BuildReplayIssueOrder(const std::vector<DepOp>& plan, std::vector<size_t>& order,
                                     std::optional<std::pair<size_t, size_t>> extraDependency) const
{
    ReplayGraph graph(plan.size());
    std::vector<bool> crossCoreBoundary(plan.size(), false);
    AddReplayDependencies(plan, extraDependency, graph, crossCoreBoundary);
    AddReplayBoundaries(crossCoreBoundary, graph);
    return graph.Sort(order);
}

void PipeSync::AddReplayDependencies(const std::vector<DepOp>& plan,
                                     std::optional<std::pair<size_t, size_t>> extraDependency, ReplayGraph& graph,
                                     std::vector<bool>& crossCoreBoundary) const
{
    std::unordered_map<PipeCoreRealEx, size_t, PipeCoreRealExHash> lastInPipe;
    auto addDependency = [&](const DepOp& op, size_t waitIdx) {
        graph.AddEdge(op.idx, waitIdx);
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
                graph.AddEdge(last->second, op.idx);
            }
            lastInPipe[endpoint] = op.idx;
        }
    }
}

void PipeSync::AddReplayBoundaries(const std::vector<bool>& crossCoreBoundary, ReplayGraph& graph) const
{
    // The legacy CV force path uses original indices as prefix boundaries.
    // Keep every cross-core endpoint's prefix intact during replay; otherwise
    // it could cancel a pending set whose producer sorts after the force.
    std::optional<size_t> lastBoundary;
    for (size_t i = 0; i < crossCoreBoundary.size(); ++i) {
        if (lastBoundary) {
            graph.AddEdge(*lastBoundary, i);
        }
        if (crossCoreBoundary[i]) {
            for (size_t j = lastBoundary ? *lastBoundary + 1 : 0; j < i; ++j) {
                graph.AddEdge(j, i);
            }
            lastBoundary = i;
        }
    }
}

void PipeSync::ReplayGraph::AddEdge(size_t from, size_t to)
{
    edges.emplace_back(to, firstEdge[from]);
    firstEdge[from] = edges.size() - 1;
    ++remaining[to];
}

bool PipeSync::ReplayGraph::Sort(std::vector<size_t>& order)
{
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t i = 0; i < firstEdge.size(); ++i) {
        if (remaining[i] == 0) {
            ready.push(i);
        }
    }
    order.clear();
    order.reserve(firstEdge.size());
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
    return order.size() == firstEdge.size();
}

std::vector<std::pair<size_t, size_t>> PipeSync::CollectReplayDependencies(const std::vector<DepOp>& plan,
                                                                           const PipePairEx& pipes) const
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
    return dependencies;
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

} // namespace tile_fwk
} // namespace npu
