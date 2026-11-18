/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

inline uint32_t GetSimtDynamicUbSize(NPUArch arch)
{
    static const std::unordered_map<NPUArch, uint32_t> budgets = {
        {NPUArch::DAV_3510, A5_SIMT_DYNAMIC_UB_SIZE},
        {NPUArch::DAV_9201, A6_SIMT_DYNAMIC_UB_SIZE},
    };
    const auto it = budgets.find(arch);
    return it != budgets.end() ? it->second : 0u;
}

inline bool IsGmGatherElement(const Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_GATHER_ELEMENT || op.GetIOperands().size() != 2U) {
        return false;
    }
    const auto& inputs = op.GetIOperands();
    return inputs[0] != nullptr && inputs[1] != nullptr &&
           inputs[0]->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR &&
           inputs[1]->GetMemoryTypeToBe() == MemoryType::MEM_UB;
}

inline bool IsGmScatter(const Operation& op)
{
    const bool scalar = op.GetOpcode() == Opcode::OP_SCATTER_ELEMENT_INPLACE;
    if ((!scalar && op.GetOpcode() != Opcode::OP_SCATTER_INPLACE) || op.GetIOperands().size() != (scalar ? 2U : 3U) ||
        op.GetOOperands().empty() || !op.HasAttr(OP_ATTR_PREFIX + "scatter_mode") ||
        op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode") != 0) {
        return false;
    }
    const auto& inputs = op.GetIOperands();
    const auto& outputs = op.GetOOperands();
    if (inputs[1] == nullptr || (!scalar && inputs[2] == nullptr) || outputs[0] == nullptr ||
        outputs[0]->GetMemoryTypeToBe() != MemoryType::MEM_DEVICE_DDR ||
        inputs[1]->GetMemoryTypeToBe() != MemoryType::MEM_UB ||
        (!scalar && inputs[2]->GetMemoryTypeToBe() != MemoryType::MEM_UB)) {
        return false;
    }
    const auto arch = Platform::Instance().GetSoc().GetNPUArch();
    return arch == NPUArch::DAV_3510 || arch == NPUArch::DAV_9201;
}

} // namespace npu::tile_fwk
