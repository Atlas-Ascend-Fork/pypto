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
 * \file replace_tensor_checker.h
 * \brief
 */

#ifndef REPLACE_TENSOR_CHECKER_H
#define REPLACE_TENSOR_CHECKER_H

#include <unordered_set>

#include "checker.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"

namespace npu {
namespace tile_fwk {
class ReplaceTensorChecker : Checker {
public:
    ReplaceTensorChecker(const std::unordered_set<int>& forwardOps, const std::unordered_set<int>& backwardOps)
        : forwardOps_(forwardOps), backwardOps_(backwardOps)
    {}

    Status DoPreCheck(Function& function) override;
    Status DoPostCheck(Function& function) override;

private:
    struct OperandCount {
        constexpr static size_t VIEW_INPUT = 1;
        constexpr static size_t VIEW_OUTPUT = 1;
        constexpr static size_t ASSEMBLE_INPUT = 1;
        constexpr static size_t ASSEMBLE_OUTPUT = 1;
        constexpr static size_t RESHAPE_INPUT = 1;
        constexpr static size_t RESHAPE_OUTPUT = 1;
        constexpr static size_t INDEX_OUTCAST_INPUTS = 3;
        constexpr static size_t INDEX_OUTCAST_OUTPUT = 1;
        constexpr static size_t A_MULACC_B_MIN_INPUTS = 3;
        constexpr static size_t A_MULACC_B_MAX_INPUTS = 4;
        constexpr static size_t A_MULACC_B_MAX_INPUTS_A5 = 5;
        constexpr static size_t A_MULACC_B_OUTPUT = 1;
    };

    bool HasSameConsecutive(const Operation& op) const;
    bool CheckAddrConflict(const Operation& op);
    bool CheckIndexProducer(const Operation& op);
    bool CheckAssembleConflict(const Operation& op);
    bool CheckIndexOutcastConflict(const Operation& op, Function& function);
    bool CheckReshapeConflict(const Operation& op, Function& function);
    bool CheckAMulAccBConflict(const Operation& op);
    Status InplaceCheck(Function& function);

    const std::unordered_set<int>& forwardOps_;
    const std::unordered_set<int>& backwardOps_;
};
} // namespace tile_fwk
} // namespace npu
#endif // REPLACE_TENSOR_CHECKER_H
