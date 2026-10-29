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
 * \file operation_utils.h
 * \brief Trace 期公共判定：动态/静态函数、是否需要先物化连续副本。
 */

#ifndef INTERFACE_OPERATION_OPERATION_UTILS_H
#define INTERFACE_OPERATION_OPERATION_UTILS_H

#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {

// 当前是否处于静态图函数，trace 期用于区分静态形状与动态形状的处理策略。
inline bool IsStaticFunction() { return Program::GetInstance().GetCurrentFunction()->IsStatic(); }

// reshape / 合轴前是否需要先物化一份连续副本：带偏移或裁剪的 VIEW/SLICE 视图内存不连续，
// 直接按连续内存处理（把 stride 当连续）会得到错误结果。
inline bool TensorNeedsContiguousCopy(const Tensor& operand)
{
    if (operand.GetShape() != operand.GetStorage()->tensor->rawshape) {
        return true;
    }
    if (operand.GetStorage()->GetProducers().empty()) {
        return false;
    }
    auto op = *operand.GetStorage()->GetProducers().begin();
    while (op != nullptr && (op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_SLICE)) {
        auto input = op->GetInputOperand(0);
        auto output = op->GetOutputOperand(0);
        if (input->GetShape() != output->GetShape()) {
            return true;
        }
        op = input->GetProducers().empty() ? nullptr : *input->GetProducers().begin();
    }
    return false;
}

} // namespace npu::tile_fwk

#endif // INTERFACE_OPERATION_OPERATION_UTILS_H
