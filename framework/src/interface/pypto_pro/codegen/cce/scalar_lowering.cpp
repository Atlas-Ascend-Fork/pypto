/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/cce_codegen.h"

#include "ir/transforms/base/mutator.h"

namespace pypto::codegen {
namespace {

class ScalarExpressionLowerer : public ir::IRMutator {
public:
    ir::FunctionPtr VisitFunction(const ir::FunctionPtr& func) override
    {
        in_simt_ = func->funcType_ == ir::FunctionType::SIMT_VF || func->funcType_ == ir::FunctionType::SIMT_CALLEE ||
                   func->funcType_ == ir::FunctionType::SIMT_KERNEL;
        return ir::IRMutator::VisitFunction(func);
    }

private:
    using ir::IRMutator::VisitExpr_;
    using ir::IRMutator::VisitStmt_;

    static ir::ExprPtr Promote(const ir::ExprPtr& operand)
    {
        return ir::GetScalarDtype(operand) == ir::DataType::BF16 ?
                   ir::MaybeCast(operand, ir::DataType::FP32, operand->span_) :
                   operand;
    }

    template <typename Op>
    ir::ExprPtr LowerBinary(const std::shared_ptr<const Op>& op)
    {
        auto left = VisitExpr(op->left_);
        auto right = VisitExpr(op->right_);
        auto dtype = ir::GetScalarDtype(op);
        auto compute_dtype = dtype;
        if (!in_simt_) {
            left = Promote(left);
            right = Promote(right);
            if (dtype == ir::DataType::BF16) {
                compute_dtype = ir::DataType::FP32;
            }
        }
        if (left == op->left_ && right == op->right_ && compute_dtype == dtype) {
            return op;
        }
        // Retain every BF16 result boundary, including inside nested expressions.
        auto result = std::make_shared<const Op>(left, right, compute_dtype, op->span_);
        return ir::MaybeCast(result, dtype, op->span_);
    }

#define LOWER_BINARY(Op) \
    ir::ExprPtr VisitExpr_(const ir::Op##Ptr& op) override { return LowerBinary(op); }
    LOWER_BINARY(Add)
    LOWER_BINARY(Sub)
    LOWER_BINARY(Mul)
    LOWER_BINARY(FloatDiv)
    LOWER_BINARY(FloorDiv)
    LOWER_BINARY(FloorMod)
    LOWER_BINARY(Eq)
    LOWER_BINARY(Ne)
    LOWER_BINARY(Lt)
    LOWER_BINARY(Le)
    LOWER_BINARY(Gt)
    LOWER_BINARY(Ge)
    LOWER_BINARY(And)
    LOWER_BINARY(Or)
    LOWER_BINARY(Xor)
#undef LOWER_BINARY

    ir::ExprPtr VisitExpr_(const ir::NotPtr& op) override
    {
        auto operand = VisitExpr(op->operand_);
        if (!in_simt_) {
            operand = Promote(operand);
        }
        return operand == op->operand_ ? op : std::make_shared<const ir::Not>(operand, ir::DataType::BOOL, op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::IfStmtPtr& op) override
    {
        auto visited = ir::As<ir::IfStmt>(ir::IRMutator::VisitStmt_(op));
        auto condition = ir::MaybeCast(visited->condition_, ir::DataType::BOOL, visited->condition_->span_);
        if (condition == visited->condition_) {
            return visited;
        }
        return std::make_shared<ir::IfStmt>(condition, visited->thenBody_, visited->elseBody_, visited->returnVars_,
                                            visited->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::WhileStmtPtr& op) override
    {
        auto visited = ir::As<ir::WhileStmt>(ir::IRMutator::VisitStmt_(op));
        auto condition = ir::MaybeCast(visited->condition_, ir::DataType::BOOL, visited->condition_->span_);
        if (condition == visited->condition_) {
            return visited;
        }
        return std::make_shared<ir::WhileStmt>(condition, visited->iterArgs_, visited->body_, visited->returnVars_,
                                               visited->span_);
    }

    bool in_simt_ = false;
};

} // namespace

ir::Pass LowerScalarExpressionsForCCE()
{
    return ir::pass::CreateProgramPass(
        [](const ir::ProgramPtr& program) {
            ScalarExpressionLowerer lowerer;
            return lowerer.VisitProgram(program);
        },
        "LowerScalarExpressionsForCCE");
}

} // namespace pypto::codegen
