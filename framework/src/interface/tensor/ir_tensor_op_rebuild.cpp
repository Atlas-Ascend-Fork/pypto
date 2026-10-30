/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir_tensor_op_rebuild.h"

#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/error.h"

using namespace pypto;

namespace npu::tile_fwk {

namespace {

LogicalTensors ExprsToLogicalTensors(const std::vector<ir::ExprPtr>& exprs)
{
    LogicalTensors out;
    out.reserve(exprs.size());
    for (auto& expr : exprs) {
        if (!expr) {
            continue;
        }
        out.push_back(std::const_pointer_cast<LogicalTensor>(std::static_pointer_cast<const LogicalTensor>(expr)));
    }
    return out;
}

LogicalTensors VarsToLogicalTensors(const std::vector<ir::VarPtr>& vars)
{
    LogicalTensors out;
    out.reserve(vars.size());
    for (auto& var : vars) {
        if (!var) {
            continue;
        }
        out.push_back(std::const_pointer_cast<LogicalTensor>(std::static_pointer_cast<const LogicalTensor>(var)));
    }
    return out;
}

} // namespace

void RemapTensorAccessTokens(const ir::TensorOpStmtPtr& src, const std::vector<ir::VarPtr>& results,
                             const std::vector<ir::VarPtr>& resultTokens, const std::vector<ir::ExprPtr>& args)
{
    FE_ASSERT(src->result_token_.size() == resultTokens.size());
    FE_ASSERT(src->args_.size() == args.size() && src->result_.size() == results.size());
    for (size_t t = 0; t < resultTokens.size(); ++t) {
        const auto& oldToken = src->result_token_[t];
        for (size_t i = 0; i < args.size(); ++i) {
            auto original = std::dynamic_pointer_cast<const LogicalTensor>(src->args_[i]);
            auto mapped = std::dynamic_pointer_cast<const LogicalTensor>(args[i]);
            if (original && mapped && original->HasReadToken(oldToken)) {
                std::const_pointer_cast<LogicalTensor>(mapped)->AddReadToken(resultTokens[t]);
            }
        }
        for (size_t i = 0; i < results.size(); ++i) {
            auto original = std::dynamic_pointer_cast<const LogicalTensor>(src->result_[i]);
            auto mapped = std::dynamic_pointer_cast<const LogicalTensor>(results[i]);
            if (original && mapped && original->HasWriteToken(oldToken)) {
                std::const_pointer_cast<LogicalTensor>(mapped)->AddWriteToken(resultTokens[t]);
            }
        }
    }
}

ir::StmtPtr RebuildTensorOpStmt(const ir::TensorOpStmtPtr& src, std::vector<ir::VarPtr> results,
                                std::vector<ir::VarPtr> resultTokens, std::vector<ir::ExprPtr> args,
                                std::vector<ir::VarPtr> tokens, ir::Span span, Function* targetFunc)
{
    ir::Span outSpan = span.IsUnknown() ? src->span_ : span;

    auto srcOp = std::dynamic_pointer_cast<const Operation>(src);
    if (srcOp == nullptr) {
        return std::make_shared<const ir::TensorOpStmt>(std::move(results), std::move(resultTokens), src->opcode_,
                                                        std::move(args), std::move(tokens), src->attrs_, outSpan);
    }

    FE_ASSERT(srcOp->BelongTo() != nullptr)
        << "RebuildTensorOpStmt: Operation has no owning function, opcode=" << src->opcode_;
    auto iOperands = ExprsToLogicalTensors(args);
    auto oOperands = VarsToLogicalTensors(results);
    return srcOp->CloneTensorOpStmt(iOperands, oOperands, resultTokens, tokens, outSpan, targetFunc);
}

} // namespace npu::tile_fwk
