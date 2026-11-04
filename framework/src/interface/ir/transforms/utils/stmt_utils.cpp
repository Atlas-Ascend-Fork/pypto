/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "stmt_utils.h"

#include <memory>
#include <unordered_set>
#include <vector>

#include "ir/expr.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"

#include "interface/tensor/ir.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/tensor/logical_tensor.h"

namespace pypto {
namespace ir {
namespace utils {

namespace {

class VarUseCollector : public IRVisitor {
public:
    std::unordered_set<const Var*> varUses;

    // When set, skip Yield/Break/Continue statements (and their carried-value
    // expressions) at any nesting depth: a terminator only forwards loop
    // carries, it is not a real use of the carried variable.
    explicit VarUseCollector(bool skipIterUpdates = false) : skipIterUpdates_(skipIterUpdates) {}

private:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;
    void VisitExpr_(const VarPtr& op) override
    {
        varUses.insert(op.get());
        IRVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const MemRefPtr& op) override
    {
        varUses.insert(op.get());
        IRVisitor::VisitExpr_(op);
    }
    // Symbolic scalar expressions (if conditions, scalar yield/continue values) keep their
    // operand vars in RawSymbolicExpression, invisible to the base visitor's traversal.
    void VisitExpr_(const ScalarExprPtr& op) override
    {
        npu::tile_fwk::SymbolicScalar::FromExpr(op).GetVarRefs(varUses);
    }
    void VisitStmt_(const YieldStmtPtr& op) override
    {
        if (!skipIterUpdates_) {
            IRVisitor::VisitStmt_(op);
            CollectValueTypeRefs(op->value_);
        }
    }
    void VisitStmt_(const BreakStmtPtr& op) override
    {
        if (!skipIterUpdates_) {
            IRVisitor::VisitStmt_(op);
            CollectValueTypeRefs(op->value_);
        }
    }
    void VisitStmt_(const ContinueStmtPtr& op) override
    {
        if (!skipIterUpdates_) {
            IRVisitor::VisitStmt_(op);
            CollectValueTypeRefs(op->value_);
        }
    }

    void VisitStmt_(const TensorOpStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        CollectScalarVarRefs(op, varUses);
    }

    // A terminator forwards carried values, and a carried tensor's dynvalidshape may reference
    // scalars (symbolic merge dims) whose defining if lives earlier in the body: those refs are
    // uses too, or the scalar's carry gets pruned while the exported tensor type still names it.
    void CollectValueTypeRefs(const std::vector<ExprPtr>& values)
    {
        for (const auto& v : values) {
            if (auto lt = npu::tile_fwk::AsLogicalTensor(v)) {
                for (auto& shape : lt->GetDynValidShape()) {
                    shape.GetVarRefs(varUses);
                }
            }
        }
    }

    bool skipIterUpdates_;
};

} // namespace

std::unordered_set<const Var*> CollectVarUses(const ExprPtr& expr)
{
    VarUseCollector collector;
    if (expr)
        collector.VisitExpr(expr);
    return std::move(collector.varUses);
}

std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt)
{
    VarUseCollector collector;
    collector.VisitStmt(stmt);
    return std::move(collector.varUses);
}

std::unordered_set<const Var*> CollectStmtVarRefs(const std::vector<StmtPtr>& stmts, bool skipIterUpdates)
{
    VarUseCollector collector(skipIterUpdates);
    for (const auto& s : stmts) {
        collector.VisitStmt(s);
    }
    return std::move(collector.varUses);
}

const std::vector<StmtPtr>& FlattenBody(const SeqStmtsPtr& body)
{
    static const std::vector<StmtPtr> empty;
    if (!body)
        return empty;
    return body->stmts_;
}

bool StmtsEqual(const std::vector<StmtPtr>& a, const std::vector<StmtPtr>& b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].get() != b[i].get())
            return false;
    }
    return true;
}

SeqStmtsPtr MakeSeqBody(const std::vector<StmtPtr>& stmts, const Span& span)
{
    return std::make_shared<SeqStmts>(stmts, span);
}

ExprPtr LookupVarInExpr(ExprPtr expr, const VarExprMap& varMap)
{
    auto cur = std::dynamic_pointer_cast<const Var>(expr);
    if (!cur) {
        return expr;
    }
    // Resolve transitively: the map may chain (A->B, B->C); follow to the terminal.
    // Guard against self/cyclic entries; stop at the first revisit.
    std::unordered_set<const Var*> seen;
    while (seen.insert(cur.get()).second) {
        auto it = varMap.find(cur);
        if (it == varMap.end()) {
            return cur;
        }
        cur = std::dynamic_pointer_cast<const Var>(it->second);
        if (!cur) {
            return it->second;
        }
    }
    return cur;
}
} // namespace utils
} // namespace ir
} // namespace pypto
