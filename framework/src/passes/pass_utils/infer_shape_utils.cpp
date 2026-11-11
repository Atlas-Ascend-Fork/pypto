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
 * \file infer_shape_utils.cpp
 * \brief 公共的 InferShape 方法实现
 */

#include <unordered_set>
#include "infer_shape_utils.h"

namespace npu {
namespace tile_fwk {
Status InferShapeUtils::InferShape(Function& function, const std::vector<Operation*>& targetOps)
{
    auto sortedOperations = function.Operations(true, SortOperationsMode::LIGHTWEIGHT);
    if (targetOps.empty()) {
        for (auto& op : sortedOperations) {
            InferShapeRegistry::GetInstance().CallInferShapeFunc(&op);
        }
        return SUCCESS;
    }

    std::unordered_set<Operation*> remaining(targetOps.begin(), targetOps.end());
    remaining.erase(nullptr);
    std::vector<Operation*> opList;
    opList.reserve(remaining.size());
    for (auto& op : sortedOperations) {
        if (remaining.erase(&op) != 0) {
            opList.push_back(&op);
        }
        if (remaining.empty()) {
            break;
        }
    }
    for (auto* op : opList) {
        InferShapeRegistry::GetInstance().CallInferShapeFunc(op);
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
