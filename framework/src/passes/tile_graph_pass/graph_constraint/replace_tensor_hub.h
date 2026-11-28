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
 * \file replace_tensor_hub.h
 * \brief HUB memory reuse processing for ReplaceTensor.
 */

#ifndef REPLACE_TENSOR_HUB_H
#define REPLACE_TENSOR_HUB_H

#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu {
namespace tile_fwk {
class ReplaceTensorHub {
public:
    ReplaceTensorHub() = default;
    ~ReplaceTensorHub() = default;

    void Process(Function& function);
    void ProcessAssemble(Function& function, Operation& hubOp, Operation& assembleOp, const LogicalTensorPtr& hubInput,
                         const LogicalTensorPtr& hubOutput);
};
} // namespace tile_fwk
} // namespace npu
#endif // REPLACE_TENSOR_HUB_H
