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
 * \file loopaxes_proc.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/common.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_interface/pass.h"
#include "loopaxes_proc.h"
#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "tilefwk/error_code.h"

#undef MODULE_NAME
#define MODULE_NAME "LoopaxesProc"

namespace npu {
namespace tile_fwk {

namespace {
constexpr size_t MIN_SHAPE_DIM = 2;

void SetOpLoopEnd(Operation* op)
{
    op->SetAttribute(OpAttributeKey::loopGroupEnd, true);
    APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup --End--", op->GetOpcodeStr().c_str(),
                     op->GetOpMagic());
}

void SetOpDynLoopEnd(Operation* op)
{
    op->SetAttribute(OpAttributeKey::dynloopGroupEnd, true);
    APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set dynloopGroup --End--", op->GetOpcodeStr().c_str(),
                     op->GetOpMagic());
}

bool IsReduceAxisInLoopTile(const Operation& op)
{
    if (!op.HasAttribute(OP_ATTR_PREFIX + "AXIS")) {
        return true;
    }
    const auto& inputs = op.GetIOperands();
    if (inputs.empty()) {
        return false;
    }
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    const auto& srcShape = inputs.front()->GetShape();
    return axis >= static_cast<int64_t>(srcShape.size()) - static_cast<int64_t>(MIN_SHAPE_DIM);
}

bool NeedClearStatus(const Operation& op)
{
    auto opCode = op.GetOpcode();
    if (SUPPORT_VF_FUSE_OPS.find(opCode) == SUPPORT_VF_FUSE_OPS.end()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "%d %s doesn't support VF fuse", op.GetOpMagic(),
                          op.GetOpcodeStr().c_str());
        return true;
    }
    if (!IsReduceAxisInLoopTile(op)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "%d %s reduce axis is not within last 2 dims, doesn't support VF fuse",
                          op.GetOpMagic(), op.GetOpcodeStr().c_str());
        return true;
    }
    return false;
}

void GetOpLoopAxes(const Operation& op, std::vector<int64_t>& loopAxes, std::vector<SymbolicScalar>& dynloopAxes)
{
    auto output = op.GetOOperands().front();
    auto shape = output->GetShape();
    auto dynShape = output->GetDynValidShape();

    if (op.HasAttr(OpAttributeKey::dynloopAxes)) {
        dynloopAxes = op.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes);
    } else {
        for (size_t i = 0; i < dynShape.size() - MIN_SHAPE_DIM; ++i) {
            dynloopAxes.push_back(dynShape[i]);
        }
    }

    if (op.HasAttr(OpAttributeKey::loopAxes)) {
        loopAxes = op.GetVectorIntAttribute(OpAttributeKey::loopAxes);
    } else {
        for (size_t i = 0; i < shape.size() - MIN_SHAPE_DIM; ++i) {
            loopAxes.push_back(shape[i]);
        }
    }
}

void HandleSmallShapeOp(Operation& op)
{
    op.SetAttribute(OpAttributeKey::dynloopGroup, INVALID_LOOP_GROUPID);
    op.SetAttribute(OpAttributeKey::loopGroup, INVALID_LOOP_GROUPID);
}
} // namespace

Status LoopaxesProc::RunOnFunction(Function& function)
{
    bool enableVF = config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
    bool useMarkFor = enableVF || config::GetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, false);
    if (!useMarkFor) {
        return SUCCESS;
    }

    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Start LoopaxesProc.");
    Status status = UpdateFuncLoopAxes(function);
    APASS_LOG_INFO_F(Elements::Operation,
                     "===============================================================> Finish LoopaxesProc.");
    return status;
}

void LoopaxesProc::ClearStatus()
{
    lastGroupIdx = INVALID_LOOP_GROUPID;
    previousOutputMagic = INVALID_LOOP_GROUPID;
    previousLoopAxes.clear();
    if (lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
        lastOpInLoop = nullptr;
    }

    dynLastGroupIdx = INVALID_LOOP_GROUPID;
    dynPreviousOutputMagic = INVALID_LOOP_GROUPID;
    dynPreviousLoopAxes.clear();
    if (dynLastOpInLoop != nullptr) {
        SetOpDynLoopEnd(dynLastOpInLoop);
        dynLastOpInLoop = nullptr;
    }
}

Status LoopaxesProc::UpdateOpLoopAxes(Operation& op, Function& subFunc)
{
    if (SKIP_OPCODE_FOR_CODEGEN.find(op.GetOpcode()) != SKIP_OPCODE_FOR_CODEGEN.end()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Op Code %s, Op[%d] ignore this op", op.GetOpcodeStr().c_str(),
                          op.GetOpMagic());
        return SUCCESS;
    }

    if (NeedClearStatus(op)) {
        ClearStatus();
        return SUCCESS;
    }

    auto output = op.GetOOperands().front();
    auto shape = output->GetShape();
    auto dynShape = output->GetDynValidShape();
    if (shape.size() != dynShape.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Op Code %s, Op[%d] output dynShape size != shape size.",
                          op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return FAILED;
    }

    if (dynShape.size() <= MIN_SHAPE_DIM) {
        HandleSmallShapeOp(op);
        ClearStatus();
        return SUCCESS;
    }

    std::vector<int64_t> loopAxes;
    std::vector<SymbolicScalar> dynloopAxes;
    GetOpLoopAxes(op, loopAxes, dynloopAxes);

    if (ProcessDynLoopGroup(op, dynloopAxes, subFunc) != SUCCESS) {
        return FAILED;
    }
    if (ProcessStaticLoopGroup(op, loopAxes) != SUCCESS) {
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Operation,
                     "Op Code %s, Op[%d] groupIdx=%ld, loopAxes=%s, dynGroupIdx=%ld, dynLoopAxes=%s",
                     op.GetOpcodeStr().c_str(), op.GetOpMagic(), groupIdx, IntVecToStr(loopAxes).c_str(), dynGroupIdx,
                     IntVecToStr(dynloopAxes).c_str());

    return SUCCESS;
}

Status LoopaxesProc::ProcessDynLoopGroup(Operation& op, const std::vector<SymbolicScalar>& dynloopAxes,
                                         const Function& subFunc)
{
    if (!SameDynLoopAxes(dynloopAxes, subFunc)) {
        if (CheckAddrOverLap(false, sameDynLoopOpGroup, addrDynConflictIdx, addrDynRecordMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unresolvable address conflict in dyn loop group");
            return FAILED;
        }
        dynLastGroupIdx = dynGroupIdx++;
        dynPreviousLoopAxes = dynloopAxes;
        op.SetAttribute(OpAttributeKey::dynloopGroupStart, true);
        if (dynLastOpInLoop != nullptr) {
            SetOpDynLoopEnd(dynLastOpInLoop);
        }
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set dynloopGroup ++Start++",
                         op.GetOpcodeStr().c_str(), op.GetOpMagic());
        sameDynLoopOpGroup.clear();
        addrDynConflictIdx.clear();
        addrDynRecordMap.clear();
    }
    sameDynLoopOpGroup.push_back(&op);
    op.SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
    op.SetAttribute(OpAttributeKey::dynloopAxes, dynloopAxes);
    dynLastOpInLoop = &op;
    dynPreviousOutputMagic = op.GetOOperands().front()->GetMagic();
    return SUCCESS;
}

namespace {
struct Interval {
    int l, r;
};

std::vector<Interval> BuildIntervals(const std::set<std::pair<int, int>>& conflicts)
{
    std::vector<Interval> intervals;
    for (const auto& p : conflicts) {
        int a = p.first, b = p.second;
        if (a == b)
            continue;
        if (a > b)
            std::swap(a, b);
        if (a <= b - 1)
            intervals.push_back({a, b - 1});
    }
    return intervals;
}

bool IsIntervalCovered(const Interval& inv, const std::vector<int>& cuts)
{
    for (int cut : cuts) {
        if (cut >= inv.l && cut <= inv.r)
            return true;
    }
    return false;
}

int CalculateMaxSegment(const std::vector<int>& sortedCuts, int groupSize)
{
    int prev = 0, maxSeg = 0;
    for (int cut : sortedCuts) {
        int segSize = cut - prev + 1;
        maxSeg = std::max(maxSeg, segSize);
        prev = cut + 1;
    }
    return std::max(maxSeg, groupSize - prev);
}

void UpdateBestSolution(const std::vector<int>& cuts, int groupSize, std::vector<int>& bestCuts, int& bestCutCount,
                        int& bestMaxSeg)
{
    int curCutCount = static_cast<int>(cuts.size());
    std::vector<int> sortedCuts = cuts;
    std::sort(sortedCuts.begin(), sortedCuts.end());
    int maxSeg = CalculateMaxSegment(sortedCuts, groupSize);
    if (curCutCount < bestCutCount || (curCutCount == bestCutCount && maxSeg > bestMaxSeg)) {
        bestCutCount = curCutCount;
        bestMaxSeg = maxSeg;
        bestCuts = cuts;
    }
}
} // namespace

std::vector<int> LoopaxesProc::FindCuts(const std::set<std::pair<int, int>>& conflicts, int groupSize,
                                        const std::set<int>& forbiddenCuts)
{
    if (groupSize <= 1)
        return {};
    auto intervals = BuildIntervals(conflicts);
    if (intervals.empty())
        return {};

    int totalPos = groupSize - 1;
    int bestCutCount = INT_MAX, bestMaxSeg = -1;
    std::vector<int> bestCuts;

    for (int mask = 0; mask < (1 << totalPos); ++mask) {
        std::vector<int> cuts;
        bool hasForbidden = false;
        for (int i = 0; i < totalPos; ++i) {
            if (mask & (1 << i)) {
                if (forbiddenCuts.count(i) > 0) {
                    hasForbidden = true;
                    break;
                }
                cuts.push_back(i);
            }
        }
        if (hasForbidden)
            continue;
        bool allCovered = true;
        for (const auto& inv : intervals) {
            if (!IsIntervalCovered(inv, cuts)) {
                allCovered = false;
                break;
            }
        }
        if (allCovered)
            UpdateBestSolution(cuts, groupSize, bestCuts, bestCutCount, bestMaxSeg);
    }
    return bestCutCount == INT_MAX ? std::vector<int>{} : bestCuts;
}

void LoopaxesProc::IsOverLap(std::vector<size_t>& addrRange, bool& isAdd, int& conflictIdx,
                             std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap,
                             std::set<std::pair<int, int>>& addrConflictIdx, int& idx)
{
    for (auto& entry : addrRecordMap) {
        if (entry.first == idx) {
            continue;
        }
        bool noOverlapWithInput = addrRange[0] >= entry.second[0][1] || addrRange[1] <= entry.second[0][0];
        bool noOverlapWithOutput = addrRange[0] >= entry.second[1][1] || addrRange[1] <= entry.second[1][0];
        if (noOverlapWithInput && noOverlapWithOutput) {
            isAdd = true;
            conflictIdx = INVALID_LOOP_GROUPID;
        } else {
            isAdd = true;
            conflictIdx = entry.first;
            std::pair<int, int> conflictPair(conflictIdx, idx);
            addrConflictIdx.insert(conflictPair);
        }
    }
}

void LoopaxesProc::RecordAddrOverLap(Operation* op, int& idx, std::set<std::pair<int, int>>& addrConflictIdx,
                                     std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap)
{
    std::vector<size_t> inAddrRange;
    std::vector<size_t> outAddrRange;
    inAddrRange.push_back(op->GetIOperands().front()->memoryrange.start);
    inAddrRange.push_back(op->GetIOperands().front()->memoryrange.end);
    outAddrRange.push_back(op->GetOOperands().front()->memoryrange.start);
    outAddrRange.push_back(op->GetOOperands().front()->memoryrange.end);
    if (addrRecordMap.empty()) {
        addrRecordMap[idx].push_back(inAddrRange);
        addrRecordMap[idx].push_back(outAddrRange);
        return;
    }
    bool isAdd{false};
    int conflictIdx = INVALID_LOOP_GROUPID;
    addrRecordMap[idx].push_back(inAddrRange);
    addrRecordMap[idx].push_back(outAddrRange);
    IsOverLap(inAddrRange, isAdd, conflictIdx, addrRecordMap, addrConflictIdx, idx);
    IsOverLap(outAddrRange, isAdd, conflictIdx, addrRecordMap, addrConflictIdx, idx);
    return;
}

Status LoopaxesProc::CheckAddrOverLap(bool isStaticLoop, std::vector<Operation*>& sameLoopOpGroup,
                                      std::set<std::pair<int, int>>& addrConflictIdx,
                                      std::map<int, std::vector<std::vector<size_t>>>& addrRecordMap)
{
    if (sameLoopOpGroup.size() != 1) {
        for (int idx = 0; idx < static_cast<int>(sameLoopOpGroup.size()); idx++) {
            APASS_LOG_INFO_F(Elements::Operation, "RecordAddrOverLap %s[%d].",
                             sameLoopOpGroup[idx]->GetOpcodeStr().c_str(), sameLoopOpGroup[idx]->GetOpMagic());
            RecordAddrOverLap(sameLoopOpGroup[idx], idx, addrConflictIdx, addrRecordMap);
        }
    }
    if (addrConflictIdx.empty()) {
        return SUCCESS;
    }
    int groupSize = static_cast<int>(sameLoopOpGroup.size());
    std::set<int> forbiddenCuts = GetForbiddenCutPositions(sameLoopOpGroup);
    std::vector<int> cutResult = FindCuts(addrConflictIdx, groupSize, forbiddenCuts);
    if (cutResult.empty()) {
        APASS_LOG_WARN_F(Elements::Operation, "No legal cut position found for address conflict.");
        return SUCCESS;
    }
    if (isStaticLoop) {
        ProcessCutStaticGroup(cutResult, sameLoopOpGroup);
    } else {
        ProcessCutDynGroup(cutResult, sameLoopOpGroup);
    }
    return SUCCESS;
}

void LoopaxesProc::ProcessCutStaticGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup)
{
    for (size_t i = 0; i < cutResult.size(); i++) {
        lastGroupIdx = groupIdx++;
        lastOpInLoop1 = sameLoopOpGroup[cutResult[i]];
        if (lastOpInLoop1 != nullptr) {
            SetOpLoopEnd(lastOpInLoop1);
        }
        sameLoopOpGroup[cutResult[i] + 1]->SetAttribute(OpAttributeKey::loopGroupStart, true);
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++",
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpcodeStr().c_str(),
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpMagic());
        if (i != cutResult.size() - 1) {
            for (int opIdx = cutResult[i] + 1; opIdx <= cutResult[i + 1]; opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::loopGroup, groupIdx);
            }
        } else {
            for (int opIdx = cutResult[i] + 1; opIdx < static_cast<int>(sameLoopOpGroup.size()); opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::loopGroup, groupIdx);
            }
        }
    }
}

void LoopaxesProc::ProcessCutDynGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup)
{
    for (size_t i = 0; i < cutResult.size(); i++) {
        dynLastGroupIdx = dynGroupIdx++;
        lastOpInLoop1 = sameLoopOpGroup[cutResult[i]];
        if (lastOpInLoop1 != nullptr) {
            SetOpDynLoopEnd(lastOpInLoop1);
        }
        sameLoopOpGroup[cutResult[i] + 1]->SetAttribute(OpAttributeKey::dynloopGroupStart, true);
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++",
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpcodeStr().c_str(),
                         sameLoopOpGroup[cutResult[i] + 1]->GetOpMagic());
        if (i != cutResult.size() - 1) {
            for (int opIdx = cutResult[i] + 1; opIdx <= cutResult[i + 1]; opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
            }
        } else {
            for (int opIdx = cutResult[i] + 1; opIdx < static_cast<int>(sameLoopOpGroup.size()); opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
            }
        }
    }
}

Status LoopaxesProc::ProcessStaticLoopGroup(Operation& op, const std::vector<int64_t>& loopAxes)
{
    if (!SameLoopAxes(loopAxes)) {
        if (CheckAddrOverLap(true, sameStaticLoopOpGroup, addrStaticConflictIdx, addrStaticRecordMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Unresolvable address conflict in static loop group");
            return FAILED;
        }
        lastGroupIdx = groupIdx++;
        previousLoopAxes = loopAxes;
        op.SetAttribute(OpAttributeKey::loopGroupStart, true);
        if (lastOpInLoop != nullptr) {
            SetOpLoopEnd(lastOpInLoop);
        }
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++", op.GetOpcodeStr().c_str(),
                         op.GetOpMagic());
        sameStaticLoopOpGroup.clear();
        addrStaticConflictIdx.clear();
        addrStaticRecordMap.clear();
    }
    sameStaticLoopOpGroup.push_back(&op);
    op.SetAttribute(OpAttributeKey::loopGroup, groupIdx);
    op.SetAttribute(OpAttributeKey::loopAxes, loopAxes);
    lastOpInLoop = &op;
    previousOutputMagic = op.GetOOperands().front()->GetMagic();
    return SUCCESS;
}

Status LoopaxesProc::UpdateFuncLoopAxes(Function& function)
{
    DynAttrToStatic dyn2Static;
    if (dyn2Static.BuildLeafToCaller(&function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to call BuildLeafToCaller.");
        return FAILED;
    }

    for (auto& pair : dyn2Static.leaf2Caller) {
        if (pair.first == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation, "subProgram of Function is nullptr.");
            continue;
        }

        ResetGroupState();

        for (auto& op : pair.first->Operations(false)) {
            if (UpdateOpLoopAxes(op, *pair.first) != SUCCESS) {
                return FAILED;
            }
        }

        if (FinalizeLoopGroups() != SUCCESS) {
            return FAILED;
        }
        if (ValidateAtomicScopeGrouping(*pair.first) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

void LoopaxesProc::ResetGroupState()
{
    groupIdx = INVALID_LOOP_GROUPID;
    lastGroupIdx = groupIdx;
    lastOpInLoop = nullptr;

    dynGroupIdx = INVALID_LOOP_GROUPID;
    dynLastGroupIdx = dynGroupIdx;
    dynLastOpInLoop = nullptr;

    previousLoopAxes.clear();
    dynPreviousLoopAxes.clear();
    sameStaticLoopOpGroup.clear();
    addrStaticConflictIdx.clear();
    addrStaticRecordMap.clear();
    sameDynLoopOpGroup.clear();
    addrDynConflictIdx.clear();
    addrDynRecordMap.clear();
}

Status LoopaxesProc::FinalizeLoopGroups()
{
    if (CheckAddrOverLap(false, sameDynLoopOpGroup, addrDynConflictIdx, addrDynRecordMap) != SUCCESS) {
        return FAILED;
    }
    if (CheckAddrOverLap(true, sameStaticLoopOpGroup, addrStaticConflictIdx, addrStaticRecordMap) != SUCCESS) {
        return FAILED;
    }

    if (groupIdx != INVALID_LOOP_GROUPID && lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
    }
    if (dynGroupIdx != INVALID_LOOP_GROUPID && dynLastOpInLoop != nullptr) {
        SetOpDynLoopEnd(dynLastOpInLoop);
    }
    return SUCCESS;
}

bool LoopaxesProc::SameLoopAxes(const std::vector<int64_t>& curLoopAxes)
{
    if (curLoopAxes.size() != previousLoopAxes.size()) {
        return false;
    }
    for (size_t i = 0; i < curLoopAxes.size(); i++) {
        if (curLoopAxes[i] != previousLoopAxes[i]) {
            return false;
        }
    }
    return true;
}

bool LoopaxesProc::SameDynLoopAxes(const std::vector<SymbolicScalar>& curLoopAxes, const Function& subFunc)
{
    if (curLoopAxes.size() != dynPreviousLoopAxes.size()) {
        return false;
    }

    auto dynParamTable = subFunc.GetDynParamTable();
    bool allReplacedSymbolsMatch = true;
    bool allExprsMatch = true;

    for (size_t i = 0; i < curLoopAxes.size(); ++i) {
        auto curExpr = SymbolicExpressionTable::BuildExpression(curLoopAxes[i]);
        auto prevExpr = SymbolicExpressionTable::BuildExpression(dynPreviousLoopAxes[i]);
        if (dynParamTable.find(curExpr) != dynParamTable.end() && dynParamTable.find(prevExpr) != dynParamTable.end()) {
            auto curParamInfo = dynParamTable[curExpr];
            auto preParamInfo = dynParamTable[prevExpr];
            if (curParamInfo.replacedSymbol.empty() || preParamInfo.replacedSymbol.empty() ||
                curParamInfo.replacedSymbol != preParamInfo.replacedSymbol) {
                allReplacedSymbolsMatch = false;
            }
        } else {
            allReplacedSymbolsMatch = false;
        }

        if (curExpr != prevExpr) {
            allExprsMatch = false;
        }
    }

    return allReplacedSymbolsMatch || allExprsMatch;
}

Status LoopaxesProc::ValidateAtomicScopeGrouping(Function& subFunc) const
{
    // dynloopGroup 与 loopGroup 是两个独立编号空间（各自独立递增），聚合 key 使用 (isDyn, group)
    // 二元组区分，避免跨空间同号（如 dyn 组 1 与 static 组 1）被折叠为同组而漏检
    std::map<int, std::set<std::pair<bool, int64_t>>> scopeToGroups;
    for (auto& op : subFunc.Operations(false)) {
        if (op.IsNOP()) {
            continue;
        }
        int scopeId = op.GetAtomicScopeId();
        if (scopeId <= 0) {
            continue;
        }
        if (SUPPORT_VF_FUSE_OPS.find(op.GetOpcode()) == SUPPORT_VF_FUSE_OPS.end()) {
            continue;
        }
        int64_t group = INVALID_LOOP_GROUPID;
        bool isDynGroup = op.HasAttr(OpAttributeKey::dynloopGroup);
        if (isDynGroup) {
            group = op.GetIntAttribute(OpAttributeKey::dynloopGroup);
        } else if (op.HasAttr(OpAttributeKey::loopGroup)) {
            group = op.GetIntAttribute(OpAttributeKey::loopGroup);
        }
        if (group != INVALID_LOOP_GROUPID) {
            scopeToGroups[scopeId].insert({isDynGroup, group});
        }
    }
    for (auto& entry : scopeToGroups) {
        if (entry.second.size() > 1) {
            APASS_LOG_WARN_F(Elements::Operation, "AtomicScope %d ops span %zu loop groups, expected 1", entry.first,
                             entry.second.size());
        }
    }
    return SUCCESS;
}

std::set<int> LoopaxesProc::GetForbiddenCutPositions(const std::vector<Operation*>& opGroup) const
{
    std::set<int> forbidden;
    for (size_t i = 0; i + 1 < opGroup.size(); i++) {
        int scope1 = opGroup[i]->GetAtomicScopeId();
        int scope2 = opGroup[i + 1]->GetAtomicScopeId();
        if (scope1 > 0 && scope1 == scope2) {
            forbidden.insert(static_cast<int>(i));
        }
    }
    return forbidden;
}

} // namespace tile_fwk
} // namespace npu
