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
 * \file expand_function.h
 * \brief
 */

#ifndef PASS_EXPAND_FUNCTION_H_
#define PASS_EXPAND_FUNCTION_H_

#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "ir/expr.h"

namespace npu::tile_fwk {
class ExpandFunction : public Pass {
public:
    ExpandFunction() : Pass("ExpandFunction") {}
    ~ExpandFunction() override = default;
    Status DefaultEnabledPreCheck(Function& function) override;
    Status PostCheck(Function& function) override;

    static Operation* GetCurrentTileOp() { return currentTileOp_; }
    static ir::VarPtr GetNormalToken(const ir::VarPtr& semantic);

private:
    Status RunOnFunction(Function& function) override;
    Status Expandfunction(Function& function) const;
    Status ExpandTensorOperations(Function& function, const std::vector<OperationPtr>& tensorOperations,
                                  const std::unordered_set<Operation*>& skipExpandOps,
                                  const std::unordered_set<LogicalTensor*>& incastGatherOutputs) const;
    Status ExpandOperation(Function& function, Operation& op) const;
    Status VerifyScopeInfo(Function& function, std::ostringstream& oss) const;
    Status ClearIOOperand(const std::vector<OperationPtr>& tensorOperations) const;
    void RefreshViewAssembleTileShapes(const std::vector<OperationPtr>& tensorOperations,
                                       const std::unordered_set<Operation*>& skipExpandOps) const;
    std::unordered_set<Operation*> CollectSkipExpandOps(Function& function,
                                                        const std::vector<OperationPtr>& tensorOperations,
                                                        std::unordered_set<LogicalTensor*>& incastGatherOutputs) const;
    static bool ShouldKeepOpUnexpanded(const OperationPtr& op, const std::unordered_set<Operation*>& skipExpandOps);
    void ProcessForNotExpandOp(Function& function, Operation& op) const;
    void DoHealthCheckBefore(Function& function, const std::string& folderPath) override;

    struct CurrentTileOpGuard {
        explicit CurrentTileOpGuard(Operation& op) { currentTileOp_ = &op; }
        ~CurrentTileOpGuard() { currentTileOp_ = nullptr; }
        CurrentTileOpGuard(const CurrentTileOpGuard&) = delete;
        CurrentTileOpGuard& operator=(const CurrentTileOpGuard&) = delete;
    };

    struct SemanticToNormalGuard {
        SemanticToNormalGuard() { semanticToNormal_.clear(); }
        ~SemanticToNormalGuard() { semanticToNormal_.clear(); }
    };

    static thread_local Operation* currentTileOp_;
    static thread_local std::unordered_map<ir::VarPtr, ir::VarPtr> semanticToNormal_;
    static const std::unordered_set<Opcode> kNotNeedExpandOps;
};
} // namespace npu::tile_fwk
#endif // PASS_EXPAND_FUNCTION_H_
