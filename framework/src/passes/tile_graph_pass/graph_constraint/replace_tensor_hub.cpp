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
 * \file replace_tensor_hub.cpp
 * \brief HUB memory reuse processing for ReplaceTensor.
 */

#include "replace_tensor_hub.h"

#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "ReplaceTensor"

namespace npu {
namespace tile_fwk {
void ReplaceTensorHub::ProcessAssemble(Function& function, Operation& hubOp, Operation& assembleOp,
                                       const LogicalTensorPtr& hubInput, const LogicalTensorPtr& hubOutput)
{
    auto assembleInput = assembleOp.GetIOperands()[0];
    auto assembleOutput = assembleOp.GetOOperands()[0];
    if (assembleInput.get() != hubOutput.get()) {
        APASS_LOG_WARN_F(Elements::Tensor, "Assemble input[%d] is not HUB output[%d], chain may be broken",
                         assembleInput->GetMagic(), hubOutput->GetMagic());
        return;
    }
    bool isExactOutcast = false;
    auto outcasts = function.GetOutcast();
    for (auto& outcast : outcasts) {
        if (outcast.get() == assembleOutput.get()) {
            isExactOutcast = true;
            break;
        }
    }
    if (!isExactOutcast) {
        APASS_LOG_WARN_F(Elements::Operation,
                         "Assemble[%d] output is not exact outcast, skip HUB memory reuse processing.",
                         assembleOp.GetOpMagic());
        return;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "Found exact HUB-ASSEMBLE-OUTCAST chain: HUB[%d] -> ASSEMBLE[%d] -> OUTCAST[%d]",
                     hubOp.GetOpMagic(), assembleOp.GetOpMagic(), assembleOutput->GetMagic());
    auto hubInputMemType = hubInput->GetMemoryTypeOriginal();
    auto hubOutputMemType = hubOutput->GetMemoryTypeOriginal();
    auto assembleOutputMemType = assembleOutput->GetMemoryTypeOriginal();
    if (hubInputMemType != hubOutputMemType || hubInputMemType != assembleOutputMemType) {
        APASS_LOG_WARN_F(Elements::Tensor, "Memory type mismatch: HUB input=%d, HUB output=%d, ASSEMBLE output=%d",
                         hubInputMemType, hubOutputMemType, assembleOutputMemType);
        return;
    }
    // RunOnFunction 在处理 HUB 前已清空 rawMagicTensors_。
    hubInput->tensor = assembleOutput->tensor;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(assembleOp.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_WARN_F(Elements::Operation, "HUB assemble op %d attribute is nullptr, skip HUB memory reuse.",
                         assembleOp.GetOpMagic());
        return;
    }
    hubInput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    hubOutput->tensor = assembleOutput->tensor;
    hubOutput->UpdateOffset(assembleOpAttribute->GetToTensorOffset());
    APASS_LOG_INFO_F(Elements::Tensor, "Complete memory reuse established: all tensors share HUB input[%d] memory",
                     hubInput->GetMagic());
}

void ReplaceTensorHub::Process(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_HUB) {
            continue;
        }
        auto hubInput = op.GetIOperands()[0];  // HUB 的输入 tensor
        auto hubOutput = op.GetOOperands()[0]; // HUB 的输出 tensor
        for (auto consumerOp : hubOutput->GetConsumers()) {
            if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
                ProcessAssemble(function, op, *consumerOp, hubInput, hubOutput);
            }
        }
        for (auto producerOp : hubInput->GetProducers()) {
            if (!OpcodeManager::Inst().IsCopyOut(producerOp->GetOpcode())) {
                continue;
            }
            auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(producerOp->GetOpAttribute());
            if (copyAttr == nullptr) {
                APASS_LOG_INFO_F(Elements::Operation, "Copy Op %d Attribute is nullptr.", producerOp->GetOpMagic());
                continue;
            }
            auto attrOffset = copyAttr->GetToOffset(); // OpImm
            auto tensorOffset = OpImmediate::Specified(hubInput->GetTensorOffset());
            std::vector<OpImmediate> newOffset;
            for (size_t i = 0; i < attrOffset.size(); i++) {
                newOffset.push_back(attrOffset[i] + tensorOffset[i]);
            }
            copyAttr->SetToOffset(newOffset);
            copyAttr->SetRawShape(OpImmediate::Specified(hubInput->tensor->GetDynRawShape()));
        }
    }
}
} // namespace tile_fwk
} // namespace npu
