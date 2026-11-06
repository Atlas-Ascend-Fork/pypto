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
 * \file codegen_op_cloudnpu.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"

namespace npu::tile_fwk {

CceSplitRegion CodeGenOpCloudNPU::GetSplitRegion()
{
    CceSplitRegion region;
    region.allowCutAfter = GetLoopAxes().empty() && (forBlkMgr_ == nullptr || !forBlkMgr_->IsInLoop());

    // Collect operands for every TileTensor operation, including temporary buffers.
    // Loop code also refers to the full-dimensional descriptors when computing addresses.
    if (isSupportTileTensor) {
        for (int i = 0; i < operandCnt; ++i) {
            const auto* full = sm->QueryTileTensorByMagic(operandWithMagic[i], originalOp.GetOpMagic());
            ASSERT(GenCodeErr::TENSOR_NOT_FOUND, full != nullptr) << "Missing split operand " << i;
            region.declarations.emplace_back(full->tensorName, full->ToString());
            const auto* inner = sm->QueryTileTensorInLoopByMagic(operandWithMagic[i], originalOp.GetOpMagic());
            if (inner != nullptr && inner->tensorName != full->tensorName) {
                region.declarations.emplace_back(inner->tensorName, inner->ToString());
            }
        }
    }

    // Operations use the default NON_SYNC_OP effect. Explicit synchronization and
    // nonlocal state are classified separately for boundary analysis.
    switch (opCode) {
        case Opcode::OP_SYNC_SRC:
        case Opcode::OP_SYNC_DST:
            // These generators emit local set_flag/wait_flag, regardless of core annotations.
            region.effect = opCode == Opcode::OP_SYNC_SRC ? CceSplitEffect::SET : CceSplitEffect::WAIT;
            break;
        case Opcode::OP_BAR_V:
        case Opcode::OP_BAR_M:
        case Opcode::OP_BAR_ALL:
            // All three opcodes use GenBarrier(), whose scope comes from pipeId_.
            region.effect = syncQueue.pipeId_ == PipeType::PIPE_ALL ? CceSplitEffect::FULL_BARRIER :
                                                                      CceSplitEffect::BARRIER;
            break;
        case Opcode::OP_CV_SYNC_SRC:
        case Opcode::OP_CV_SYNC_DST:
        case Opcode::OP_FFTS_CROSS_CORE_SYNC:
        case Opcode::OP_WAIT_FLAG_DEV:
            region.effect = CceSplitEffect::CROSS_CORE_SYNC;
            break;
        case Opcode::OP_AICPU_CALL_AIC:
        case Opcode::OP_AICPU_CALL_AIV:
            region.effect = CceSplitEffect::EXTERNAL_STATE;
            break;
        default:
            if (OpcodeManager::Inst().IsSharedMemory(opCode)) {
                region.effect = CceSplitEffect::EXTERNAL_STATE;
            }
            break;
    }
    return region;
}

CodeGenOpCloudNPU::CodeGenOpCloudNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOpNPU(ctx)
{
    InitOpsGenMap();
    forBlkMgr_ = ctx.forBlockManager;
    CodeGenOp::Init(ctx.operation);
    GetGmParamIdx(ctx.operation);
    UpdateTileTensorInfo();
    UpdateLoopInfo();
}

void CodeGenOpCloudNPU::GetGmParamIdx(const Operation& oper)
{
    if (oper.HasAttribute(OpAttributeKey::gmTensorParamIdxInCall)) {
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute(OpAttributeKey::gmTensorParamIdxInCall);
    }

    for (size_t i = 0; i < oper.GetOOperands().size(); ++i) {
        if (oper.GetOOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
            paramLocation[i] = oper.GetOOpAttrOffset(i);
        }
    }

    size_t iOffset = oper.GetOOperands().size() == 0 ? 1 : oper.GetOOperands().size();
    for (size_t i = 0; i < oper.GetIOperands().size(); ++i) {
        if (oper.GetIOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
            paramLocation[i + iOffset] = oper.GetIOpAttrOffset(i);
        }
    }
}

} // namespace npu::tile_fwk
