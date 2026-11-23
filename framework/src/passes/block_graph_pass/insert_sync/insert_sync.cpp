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

// check if platform architecture is mix cv core, and coreOne and coreTwo are AIV/AIC or vice versa

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

// find depend op in opLog for 0 to idx

// 该函数仅用于核内同步

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

void PipeSync::PhaseKernelProcess(Function& function, const std::vector<Operation*>& srcLog,
                                  std::vector<Operation*>& dstLog)
{
    size_t prerun = 0;
    size_t i = 0;
    AddPhaseOp1(function, srcLog, dstLog, i, prerun);
    if (prerun > 0) {
        std::vector<std::shared_ptr<LogicalTensor>> input;
        std::vector<std::shared_ptr<LogicalTensor>> output;
        Operation& phaseOp = irBuilder_.CreateTensorOpStmt(function, npu::tile_fwk::Opcode::OP_PHASE2, input, output);
        Operation* phaseOpPtr = &phaseOp;
        dstLog.emplace_back(phaseOpPtr);
    }
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
            const auto status = ValidateClusterContinuity(opList, i, scopeId, ci);
            if (status != SUCCESS || ci.skipAdjust) {
                return status;
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
        if (CollectClusterSync(op, scopeId, ci) != SUCCESS) {
            return FAILED;
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
    APASS_LOG_DEBUG_F(Elements::Operation,
                      "===============================================================> Start InsertSync.");
    if (cvHardSync_) {
        APASS_LOG_INFO_F(Elements::Operation,
                         "CvHardSync debug mode enabled, all cross-core syncs use 1C2V hard sync.");
    }
    const unsigned hardwareConcurrency = config::GetPassGlobalConfig(KEY_PASS_THREAD_NUM, 1);
    std::vector<std::pair<uint64_t, Function*>> subPrograms;
    for (auto& subProgram : function.rootFunc_->programs_) {
        subPrograms.push_back(subProgram);
    }
    std::atomic<size_t> nextIdx(0);
    size_t leafFuncSize = function.rootFunc_->programs_.size();
    // The max thread number by std::thread::hardware_concurrency()

    const unsigned threadNum = std::min(static_cast<unsigned>(leafFuncSize), hardwareConcurrency);
    std::vector<std::thread> workers;
    std::atomic<bool> status{true};
    for (unsigned i = 0; i < threadNum; ++i) {
        workers.emplace_back([&subPrograms, &nextIdx, this, &status] { RunSubPrograms(subPrograms, nextIdx, status); });
    }
    // Wait for all threads to finish
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (!status.load(std::memory_order_relaxed)) {
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

    APASS_LOG_DEBUG_F(Elements::Operation,
                      "===============================================================> Finish InsertSync.");
    return SUCCESS;
}

void PipeSyncTypes::DataDepInfo::Reset(const PipePairEx& pipes)
{
    setp = pipes.first.pipe;
    setc = pipes.first.core;
    waitp = pipes.second.pipe;
    waitc = pipes.second.core;
    setaivc = pipes.first.aivCore;
    waitaivc = pipes.second.aivCore;
    opDepList.clear();
    setOpIdList.clear();
    setOpEventIdList.clear();
}

void PipeSyncTypes::DataDepInfo::CollectPending(const std::vector<IndexOp>& log,
                                                const std::unordered_map<Operation*, const EventInstance*>& pending)
{
    // Each generated set occurs once in the log. Pending sets are usually near
    // its tail; stop once all are found, then restore the original candidate order.
    size_t remaining = pending.size();
    for (size_t i = log.size(); i-- > 0;) {
        auto it = pending.find(&log[i].second.get());
        if (it != pending.end()) {
            const auto& instance = *it->second;
            opDepList.emplace_back(instance.setIdx, instance.waitIdx);
            setOpIdList.push_back(static_cast<int>(i));
            setOpEventIdList.push_back(instance.id.resource.eventId);
            if (--remaining == 0) {
                break;
            }
        }
    }
    std::reverse(opDepList.begin(), opDepList.end());
    std::reverse(setOpIdList.begin(), setOpIdList.end());
    std::reverse(setOpEventIdList.begin(), setOpEventIdList.end());
}

void InsertSync::RunSubPrograms(const std::vector<std::pair<uint64_t, Function*>>& subPrograms,
                                std::atomic<size_t>& nextIdx, std::atomic<bool>& status)
{
    const size_t leafFuncSize = subPrograms.size();
    for (size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed); idx < leafFuncSize;
         idx = nextIdx.fetch_add(1, std::memory_order_relaxed)) {
        auto program = subPrograms[idx];
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "====================================Program %zu ===========================================", idx);
        if (InsertSyncMainLoop(program.second) != SUCCESS) {
            status.store(false, std::memory_order_relaxed);
            break;
        }
    }
}

Status InsertSync::CollectClusterSync(Operation* op, int scopeId, ClusterInfo& ci)
{
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
        if (ci.setSignatures.count(
                GetSyncSignature(op->syncQueue_.eventId_, op->syncQueue_.pipeId_, op->syncQueue_.trigPipeId_)) > 0) {
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

    return SUCCESS;
}

Status InsertSync::ValidateClusterContinuity(const std::vector<Operation*>& opList, size_t i, int scopeId,
                                             ClusterInfo& ci)
{
    Operation* op = opList[i];
    if (op->GetAIVCore() != ci.aivCore) {
        // OOO 发射序列跨核交错：区间内另一核（AIV0/AIV1/AIC）的 op 不属于本 cluster，跳过
        return SUCCESS;
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
    return SUCCESS;
}

// Leaf synchronization instructions

Status PipeSync::InjectWaitFlag(Function& function, size_t idx, std::vector<IndexOp>& syncedOpLog)
{
    PipeCore currPipe = depOps_[idx].selfPipeCore;
    // Validate the complete input group before consuming any instance. A
    // malformed later wait must not leave an earlier wait partially committed.
    if (ValidateWaitGroup(idx) != SUCCESS) {
        return FAILED;
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

void PipeSync::DiscardMergedSet(std::vector<IndexOp>& syncedOpLog, size_t syncOpIdx1)
{
    // 将靠前的一对有依赖关系op中插入的SYNC_SRC op删除
    syncedOpLog[syncOpIdx1].second.get().SetAsDeleted();
    syncedOpLog.erase(syncedOpLog.begin() + syncOpIdx1);
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
    crossCoreCompletedSenders_.fill(std::nullopt);
    crossCoreWaitGenerations_ = {};
    crossCoreConsumed_.clear();
    crossCoreSnapshots_.clear();
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
    const bool oldCubeSender = GetCrossCoreHandshakeSender(eventId) == CoreType::AIC;
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

void PipeSync::ResetLeafForReplay(Function& function, ReplayContext& replay, std::vector<IndexOp>& syncedOpLog,
                                  const std::vector<size_t>& replayOrder)
{
    // A failed attempt's snapshots and generated ops must never
    // survive into the rebuilt event-instance graph.
    for (auto& op : function.Operations(false)) {
        if (replay.originalOps.count(&op) == 0) {
            op.SetAsDeleted();
        }
    }
    ResetForReplay(replay.plan, replay.queues, replayOrder);
    syncedOpLog = replay.log;
    function.EraseOperations(true, false);
}

// Validation of the generated synchronization

Status PipeSync::BuildEventDependencyGraph(const std::vector<Operation*>& opList, EventDependencyGraph& graph) const
{
    // Reconstruct only real FIFO and matched set->wait edges. In particular,
    // never add a wait->next-set edge merely because the ID was reused.
    std::unordered_map<Operation*, std::pair<size_t, bool>> syncInstances;
    syncInstances.reserve(eventState_.Instances().size() * 2);
    if (CollectEventSyncInstances(syncInstances) != SUCCESS) {
        return FAILED;
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
    if (AddMatchedEventEdges(graph) != SUCCESS) {
        return FAILED;
    }
    return AddReclaimBarrierEdges(opList, graph);
}

Status PipeSync::AddReclaimBarrierEdges(const std::vector<Operation*>& opList, EventDependencyGraph& graph) const
{
    if (reclaimBarriers_.empty()) {
        return SUCCESS;
    }
    const auto realNodeCount = graph.nodes.size();
    // Reclaim regions are immutable. BAR.ALL drains the entire local prefix;
    // all local prefix waits complete before suffix instructions execute.
    std::unordered_map<Operation*, size_t> listPosition;
    listPosition.reserve(opList.size());
    for (size_t i = 0; i < opList.size(); ++i) {
        listPosition[opList[i]] = i;
    }
    for (const auto& barrier : reclaimBarriers_) {
        if (AddReclaimRegionEdges(barrier, opList, listPosition, realNodeCount, graph) != SUCCESS) {
            return FAILED;
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

Status PipeSync::CollectEventSyncInstances(std::unordered_map<Operation*, std::pair<size_t, bool>>& syncInstances) const
{
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
    return SUCCESS;
}

Status PipeSync::AddMatchedEventEdges(EventDependencyGraph& graph) const
{
    auto addEdge = [&](size_t from, size_t to) {
        graph.successors[from].push_back(to);
        ++graph.inDegree[to];
    };
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
    return SUCCESS;
}

Status PipeSync::AddReclaimRegionEdges(const ReclaimBarrier& barrier, const std::vector<Operation*>& opList,
                                       const std::unordered_map<Operation*, size_t>& listPosition, size_t realNodeCount,
                                       EventDependencyGraph& graph) const
{
    auto addEdge = [&](size_t from, size_t to) {
        graph.successors[from].push_back(to);
        ++graph.inDegree[to];
    };
    if (barrier.ops.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Empty event reclaim region in final sync sequence.");
        return FAILED;
    }
    auto first = listPosition.find(barrier.ops.front());
    if (first == listPosition.end() || first->second + barrier.ops.size() > opList.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Missing or incomplete event reclaim region in final sync sequence.");
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
    AddReclaimEndpoints(barrier.cores.size(), graph, entries, exits);
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

    return SUCCESS;
}

void PipeSync::AddReclaimEndpoints(size_t count, EventDependencyGraph& graph, std::vector<size_t>& entries,
                                   std::vector<size_t>& exits) const
{
    auto addEdge = [&](size_t from, size_t to) {
        graph.successors[from].push_back(to);
        ++graph.inDegree[to];
    };
    for (size_t i = 0; i < count; ++i) {
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
}

} // namespace tile_fwk
} // namespace npu
