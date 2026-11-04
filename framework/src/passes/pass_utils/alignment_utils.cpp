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
 * \file alignment_utils.cpp
 * \brief
 */

#include "passes/pass_utils/alignment_utils.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

bool AlignmentUtils::CheckPadSecondLast(const LogicalTensorPtr& tensor, bool enableAxisCombine)
{
    if (!enableAxisCombine || !IsValidForLastDimCheck(tensor)) {
        return false;
    }
    const auto& shape = tensor->shape;
    const auto& rawShape = tensor->tensor->rawshape;
    return shape.size() > 1 && rawShape.size() == shape.size() && rawShape.back() == 1;
}

bool AlignmentUtils::IsValidForLastDimCheck(const LogicalTensorPtr& tensor)
{
    return tensor != nullptr && tensor->tensor != nullptr && !tensor->shape.empty();
}

bool AlignmentUtils::IsCombinedAxis(const std::vector<bool>& combineAxis, size_t index)
{
    return index < combineAxis.size() && combineAxis[index];
}

int64_t AlignmentUtils::GetLastDimAlignBase(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return 0;
    }
    constexpr int64_t blockBits = 32 * 8;
    auto bits = BitsOf(tensor->Datatype());
    if (blockBits % bits != 0) {
        return 0;
    }
    return blockBits / bits;
}

bool AlignmentUtils::IsLastDim32BAligned(const LogicalTensorPtr& tensor)
{
    if (!IsValidForLastDimCheck(tensor)) {
        return false;
    }
    auto alignBase = GetLastDimAlignBase(tensor);
    if (alignBase <= 0) {
        return false;
    }
    auto lastDim = tensor->shape.back();
    if (lastDim <= 0) {
        return false;
    }
    return lastDim % alignBase == 0;
}

int64_t AlignmentUtils::Pad(int64_t dim, int64_t padValue)
{
    if (padValue == 0) {
        return dim;
    }
    return (dim + padValue - 1) / padValue * padValue;
}

void AlignmentUtils::ProcessLastDim32BAlignedOnUB(LogicalTensorPtr tensor)
{
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return;
    }
    auto memType = tensor->GetMemoryTypeOriginal();
    if (memType == MemoryType::MEM_UB && !IsLastDim32BAligned(tensor)) {
        size_t lastIdx = tensor->shape.size() - 1;
        size_t paddingValue = GetLastDimAlignBase(tensor); // 根据数据类型，判断需要pad到几个元素

        // 保存原始值
        int64_t oriRawshapeValue = tensor->tensor->rawshape[lastIdx];

        // pad 32B
        auto rawShapePadded = Pad(oriRawshapeValue, paddingValue);
        tensor->tensor->rawshape[lastIdx] = Pad(oriRawshapeValue, rawShapePadded);
    }
}

size_t AlignmentUtils::GetLastDimBytes(const LogicalTensorPtr& tensor)
{
    if (tensor == nullptr || tensor->GetRawTensor() == nullptr) {
        return 0;
    }
    auto& rawshape = tensor->GetRawTensor()->GetRawShape();
    if (rawshape.empty()) {
        return 0;
    }
    constexpr int64_t byteBits = 8;
    auto bits = BitsOf(tensor->GetRawTensor()->GetDataType());
    return static_cast<size_t>((rawshape.back() * bits + byteBits - 1) / byteBits);
}

int64_t AlignmentUtils::PadRowDim(int64_t dim, int64_t padValue) { return dim + padValue - 1; }

} // namespace npu::tile_fwk
